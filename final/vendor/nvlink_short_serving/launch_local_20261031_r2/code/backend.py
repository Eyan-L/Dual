"""Short serving workload: local FlashInfer prefill and tiered decode.

The peer attention arithmetic and 2048-token migration policy stay frozen.
Local-only prefill uses the original backend in both custom and peer cases.
"""

from pathlib import Path as _FinalPath
_SLACK_FINAL_ROOT = next(p for p in _FinalPath(__file__).resolve().parents if (p / 'release.json').is_file())
_SLACK_PROJECT_ROOT = _SLACK_FINAL_ROOT.parent
from collections import deque
from collections import Counter
import importlib.util
from itertools import islice
from pathlib import Path
import sys
import time
import types
import torch
from paged_decode import PagedDecode
from page_ops import restore, write_prefill_layout

SOURCE=Path(str(_SLACK_FINAL_ROOT / 'vendor/nvlink_prefill_opt/launch_20261031_r3/code/backend.py'))
spec=importlib.util.spec_from_file_location('frozen_prefill5_for_trace',SOURCE)
frozen=importlib.util.module_from_spec(spec);sys.modules[spec.name]=frozen;spec.loader.exec_module(frozen)
scope=frozen.scope


class ReplayBackend(frozen.TimingBackend):
    def __init__(self,raw,peer,cfg,old_control,case):
        native_extend=raw.attn_backend.forward_extend
        super().__init__(raw,peer,cfg,old_control,case)
        self.validation=cfg['validation'];self.banks={};self.queries={};self.last_fb=None
        parent_decode=self.backend.forward_decode;parent_extend=self.backend.forward_extend
        parent_forward=raw.forward;control=self
        def decode(backend,q,k,v,layer,fb,save_kv_cache=True):
            if not control.enabled:return parent_decode(q,k,v,layer,fb,save_kv_cache=save_kv_cache)
            backend.token_to_kv_pool.set_kv_buffer(layer,fb.out_cache_loc,k,v,layer.k_scale_float,layer.v_scale_float)
            query=q.contiguous().view(-1,32,128);engine=control.banks[fb.batch_size]
            if control.validation:control.queries[fb.batch_size][layer.layer_id].copy_(query)
            return engine.run(query,layer.layer_id).view(-1,4096)
        def extend(backend,q,k,v,layer,fb,save_kv_cache=True):
            # Prefill is eager: select on the requests actually in this forward.
            # A remote page owned by another request does not forbid FlashInfer.
            method=native_extend if control.prefill_all_local else parent_extend
            out=method(q,k,v,layer,fb,save_kv_cache=save_kv_cache)
            if control.validation:control.check_attention_output(q.view(-1,32,128),out.view(-1,32,128),layer.layer_id,fb)
            return out
        def forward(fb,*args,**kwargs):
            control.last_fb=fb
            if fb.forward_mode.is_extend():
                control.prefill_all_local=not any(control.remote_counts.get(int(r.kv.req_pool_idx),0) for r in control.req_objects)
            if not control.enabled:
                control.requests=[int(r.kv.req_pool_idx) for r in control.req_objects]
                control.lengths=fb.seq_lens_cpu.tolist()
                control.live.update(zip(control.requests,control.lengths))
            result=parent_forward(fb,*args,**kwargs)
            if fb.forward_mode.is_extend():
                control.frames[-1]['attention_path']='native_local_prefill' if control.prefill_all_local else 'tiered_prefill5'
            else:control.frames[-1]['attention_path']='tiered_decode' if control.enabled else 'native_decode'
            return result
        self.backend.forward_decode=types.MethodType(decode,self.backend)
        self.backend.forward_extend=types.MethodType(extend,self.backend)
        raw.forward=forward

    def reset(self,batch,profile=False):
        super().reset(batch,profile)
        self.remote_positions={};self.restorations=[];self.releases=[];self.layer_checks=[]
        self.remote_counts=Counter();self.prefill_all_local=True
        self.pending_coordinates=[];self.pending_targets=[];self.req_objects=[]
        self.max_local=0;self.max_remote=0

    def prepare_graphs(self,max_batch):
        self.reset(1)
        if self.enabled:
            self.tier.zero_();self.slot.zero_()
            for layer in range(36):
                for plane in self.raw.token_to_kv_pool.get_kv_buffer(layer):plane[0].zero_()
            # Graph sizes run sequentially. Share scratch rather than allocate
            # sum(B) storage, which is excessive for short-context large batches.
            owner=PagedDecode(self.raw,self.peer,self.tier,self.slot,max_batch,self.cfg['decode_splits'])
            queries=torch.empty((36,max_batch,32,128),device=self.raw.device,dtype=torch.bfloat16) if self.validation else None
            for b in range(1,max_batch+1):
                engine=object.__new__(PagedDecode);engine.__dict__.update(owner.__dict__);engine.batch=b
                for name in ['requests','seq']:setattr(engine,name,getattr(owner,name)[:b])
                for name in ['acc','maximum','den','output']:setattr(engine,name,getattr(owner,name)[:,:b])
                self.banks[b]=engine
                if self.validation:self.queries[b]=queries[:,:b]
            self.engine=self.banks[1]
        graph=self.raw.decode_cuda_graph_runner
        graph.capture_bs=list(range(1,max_batch+1));graph.capture();torch.cuda.synchronize()
        self.reset(1)

    def begin(self,fb):
        self.batch=fb.batch_size
        self.requests=[int(r.kv.req_pool_idx) for r in self.req_objects]
        self.lengths=fb.seq_lens_cpu.tolist()
        if fb.forward_mode.is_decode():
            self.engine=self.banks[fb.batch_size]
            # One vector operation per table, rather than two launches/request.
            rid=fb.req_pool_indices;pos=fb.seq_lens-1
            self.slot[rid,pos]=fb.out_cache_loc.to(torch.int32);self.tier[rid,pos]=0
            coordinates=list(zip(self.requests,[n-1 for n in self.lengths]))
            self.engine.prepare(fb)
        else:
            write_prefill_layout(self.tier,self.slot,fb.req_pool_indices,fb.extend_start_loc,
                fb.extend_seq_lens,fb.extend_prefix_lens,fb.out_cache_loc,max(fb.extend_seq_lens_cpu))
            coordinates=[(rid,pos) for rid,n,q in zip(self.requests,self.lengths,fb.extend_seq_lens_cpu) for pos in range(n-q,n)]
        self.residents.extend(coordinates)
        self.created.update((key,len(self.frames)) for key in coordinates)
        self.live.update(zip(self.requests,self.lengths))
        local,remote,_=self.budget();self.max_local=max(self.max_local,local);self.max_remote=max(self.max_remote,remote)

    def warm_prefill_kernels(self):
        if not self.enabled:return
        from optimized_prefill import prefill
        # Compile every query-block count used by the measured chunk bound.
        # Only sentinel slot0 is touched; no request payload or KV mirror.
        self.tier[0].zero_();self.slot[0].zero_()
        lk,lv=self.raw.token_to_kv_pool.get_kv_buffer(0)
        req=torch.zeros(1,device=self.raw.device,dtype=torch.int32)
        seq=torch.ones_like(req);start=torch.zeros_like(req);prefix=torch.zeros_like(req)
        for size in range(1,self.cfg['prefill_chunk_tokens']+1,32):
            q=torch.zeros((size,32,128),device=self.raw.device,dtype=torch.bfloat16)
            qlen=torch.full_like(req,size)
            args=(q,lk,lv,self.peer[0,0],self.peer[0,1],self.tier,self.slot,req,seq,start,qlen,prefix,size,128**-.5)
            frozen.original_prefill(*args)
            prefill(*args,block_m=64,splits=1)
        torch.cuda.synchronize();self.tier[0].fill_(-1)

    def allocation_budget(self):
        free=self.physical_allocator.available_size()
        return free+(self.budget()[2] if self.case=='tiered_peer' else 0)

    def launch(self,needed,asynchronous):
        n=self.amount(needed)[0]
        coordinates=list(islice(self.residents,n));targets=list(islice(self.peer_free,n))
        super().launch(needed,asynchronous)
        self.pending_coordinates=coordinates;self.pending_targets=targets

    def commit(self):
        if self.validation:
            rid,pos,source,targets,done,record=self.pending;done.synchronize()
            for i in range(36):
                for plane,local in enumerate(self.raw.token_to_kv_pool.get_kv_buffer(i)):
                    assert torch.equal(local.index_select(0,source),self.peer[i,plane].index_select(0,targets)),('migration payload',i,plane)
        super().commit()
        self.remote_positions.update(zip(self.pending_coordinates,self.pending_targets))
        self.remote_counts.update(rid for rid,pos in self.pending_coordinates)
        self.pending_coordinates=[];self.pending_targets=[]

    def ensure_headroom(self,needed):
        start=time.perf_counter()
        if self.pending is not None:self.commit()
        if self.physical_allocator.available_size()<needed:
            assert self.case=='tiered_peer'
            # A new batch/prefill boundary may have no eligible prior prefetch.
            # Keep the cost visible rather than asserting a fixed batch exists.
            self.launch(needed,False);self.commit()
        return (time.perf_counter()-start)*1000

    def prefetch_next(self,has_next):
        if not self.enabled or self.case!='tiered_peer' or not has_next:return
        if self.physical_allocator.available_size()>=self.batch:return
        if self.allocation_budget()<self.batch:return
        amount,_,_,_=self.amount(self.batch)
        oldest=list(islice(self.residents,amount))
        # KV first written by this forward cannot be copied from its start.
        if any(self.created[item]>=self.frames[-1]['uid'] for item in oldest):return
        super().prefetch_next(True)

    def release_requests(self,requests):
        requests=[r for r in requests if r.kv.holds_kv]
        with scope(f'SLACK_COST_RELEASE_{len(self.frames)}',self.profile):
            return self._release_requests(requests)

    def _release_requests(self,requests):
        if not requests:return
        start=time.perf_counter_ns();events=self.events();events[0].record()
        if self.pending is not None:self.commit()
        released=[]
        if self.enabled:
            # Remove the whole departing group in one pass. Previously each
            # request rebuilt BOTH full-pool containers, O(group * active KV).
            gone={int(req.kv.req_pool_idx) for req in requests}
            positions=[key for key in self.remote_positions if key[0] in gone]
            for key in positions:self.peer_free.append(self.remote_positions.pop(key))
            self.committed_remote-=len(positions)
            for rid in gone:self.remote_counts.pop(rid,None)
            self.residents=deque(key for key in self.residents if key[0] not in gone)
        for req in requests:
            rid=int(req.kv.req_pool_idx);length=self.live.pop(rid)
            if self.enabled:
                far=self.tier[rid,:length]==1;near=self.slot[rid,:length][~far].long()
                # Only this request's logical positions need deleting. This
                # also removes entries for pages previously migrated remotely.
                for pos in range(length):self.created.pop((rid,pos))
                self.tier[rid,:length].fill_(-1);self.slot[rid,:length].zero_()
            else:near=self.raw.req_to_token_pool.req_to_token[rid,:length].long()
            self.physical_allocator.free(near)
            self.raw.req_to_token_pool.req_to_token[rid,:length].zero_()
            self.raw.req_to_token_pool.free(req)
            released.append(dict(request=rid,logical_tokens=length,local_tokens=near.numel()))
        self.repair_active_ratio()
        events[1].record()
        self.releases.append(dict(uid=len(self.frames),requests=released,cpu_ms=(time.perf_counter_ns()-start)/1e6,_events=events))

    def repair_active_ratio(self):
        local,remote,_=self.budget()
        if not remote or 10000*remote<599*local:return
        count=(10000*remote-599*local)//10599+1
        selected=list(islice(self.remote_positions,count));assert len(selected)==count
        source=torch.tensor([self.remote_positions[k] for k in selected],device=self.raw.device,dtype=torch.int64)
        targets=self.physical_allocator.alloc(count);assert targets is not None
        rid=torch.tensor([k[0] for k in selected],device=self.raw.device,dtype=torch.int64)
        pos=torch.tensor([k[1] for k in selected],device=self.raw.device,dtype=torch.int64)
        if not hasattr(self,'local_pointers'):
            self.local_pointers=torch.tensor([v.data_ptr() for i in range(36) for v in self.raw.token_to_kv_pool.get_kv_buffer(i)],device=self.raw.device,dtype=torch.int64)
        start=time.perf_counter_ns();events=self.events();events[0].record()
        with scope(f'SLACK_COST_RESTORE_{len(self.frames)}',self.profile):
            restore(self.local_pointers,self.peer,source,targets)
            if self.validation:
                torch.cuda.synchronize()
                for i in range(36):
                    for plane,near in enumerate(self.raw.token_to_kv_pool.get_kv_buffer(i)):
                        assert torch.equal(near.index_select(0,targets),self.peer[i,plane].index_select(0,source)),('restore payload',i,plane)
            self.tier[rid,pos]=0;self.slot[rid,pos]=targets.to(torch.int32)
            self.raw.req_to_token_pool.req_to_token[rid,pos]=targets.to(torch.int32)
        for key in selected:
            self.peer_free.append(self.remote_positions.pop(key));self.remote_counts[key[0]]-=1
        self.committed_remote-=count
        self.residents.extend(selected);self.residents=deque(sorted(self.residents,key=lambda k:(self.created[k],k)))
        events[1].record();after_local,after_remote,_=self.budget()
        assert not after_remote or 10000*after_remote<599*after_local
        self.restorations.append(dict(uid=len(self.frames),tokens=count,local_before=local,remote_before=remote,
            local_after=after_local,remote_after=after_remote,copy_bytes=count*147456,cpu_ms=(time.perf_counter_ns()-start)/1e6,_events=events))

    def audit(self):
        near=[];far=[]
        for rid,length in self.live.items():
            if self.enabled:
                tiers=self.tier[rid,:length];slots=self.slot[rid,:length];remote=tiers==1
                assert ((tiers==0)|remote).all().item()
                assert torch.equal(slots[~remote],self.raw.req_to_token_pool.req_to_token[rid,:length][~remote])
                assert (self.raw.req_to_token_pool.req_to_token[rid,:length][remote]==0).all().item()
                near.append(slots[~remote]);far.append(slots[remote])
            else:near.append(self.raw.req_to_token_pool.req_to_token[rid,:length])
        n=torch.cat(near).long() if near else torch.empty(0,device=self.raw.device,dtype=torch.int64)
        f=torch.cat(far).long() if far else torch.empty(0,device=self.raw.device,dtype=torch.int64)
        assert torch.unique(n).numel()==n.numel() and (n>0).all().item()
        assert torch.unique(f).numel()==f.numel()
        assert n.numel()==self.cfg['kv_capacity_tokens']-self.physical_allocator.available_size()
        assert f.numel()==self.committed_remote==len(self.remote_positions)
        if self.enabled:assert +self.remote_counts==Counter(rid for rid,pos in self.remote_positions)
        reserved=len(self.pending_targets) if self.pending is not None else 0
        assert f.numel()+len(self.peer_free)+reserved==self.peer.shape[2]
        assert n.numel()+f.numel()==sum(self.live.values())
        assert not f.numel() or 10000*f.numel()<599*n.numel()
        return dict(local_active_tokens=n.numel(),remote_active_tokens=f.numel(),logical_tokens=n.numel()+f.numel(),
            page_tables_exact=True,unique_local_slots=True,unique_peer_slots=True,remote_local_shadow_present=False)

    def check_attention_output(self,q,out,layer,fb):
        lengths=fb.seq_lens_cpu.tolist()
        qlens=fb.extend_seq_lens_cpu if fb.forward_mode.is_extend() else [1]*fb.batch_size
        prefixes=fb.extend_prefix_lens_cpu if fb.forward_mode.is_extend() else [n-1 for n in lengths]
        references=[];offset=0;lk,lv=self.raw.token_to_kv_pool.get_kv_buffer(layer)
        for req,length,qlen,prefix in zip(self.req_objects,lengths,qlens,prefixes):
            rid=int(req.kv.req_pool_idx)
            if self.enabled:
                slots=self.slot[rid,:length].long();remote=self.tier[rid,:length]==1
            else:
                slots=self.raw.req_to_token_pool.req_to_token[rid,:length].long();remote=torch.zeros(length,device=q.device,dtype=torch.bool)
            kv=[]
            for local,peer in ((lk,self.peer[layer,0]),(lv,self.peer[layer,1])):
                value=torch.empty((length,8,128),device=q.device,dtype=torch.float32)
                value[~remote]=local.index_select(0,slots[~remote]).float()
                value[remote]=peer.index_select(0,slots[remote]).float();kv.append(value)
            query=q[offset:offset+qlen].float().transpose(0,1)
            key=kv[0].repeat_interleave(4,1).transpose(0,1);value=kv[1].repeat_interleave(4,1).transpose(0,1)
            score=(query@key.transpose(1,2))*(128**-.5)
            visible=torch.arange(length,device=q.device)[None,:]<=prefix+torch.arange(qlen,device=q.device)[:,None]
            truth=(score.masked_fill(~visible[None,:,:],float('-inf')).softmax(-1)@value).transpose(0,1)
            references.append(truth);offset+=qlen
        expected=torch.cat(references)
        torch.testing.assert_close(out.float(),expected,atol=self.cfg['reference_atol'],rtol=self.cfg['reference_rtol'])
        self.layer_checks.append(dict(uid=len(self.frames),layer=layer,mode='prefill' if fb.forward_mode.is_extend() else 'decode',
            batch=fb.batch_size,max_abs_error=float((out.float()-expected).abs().max()),remote_tokens=self.remote_active))

    def validate_decode(self):
        if not self.validation or not self.enabled:return
        fb=self.last_fb;engine=self.banks[fb.batch_size]
        for layer in range(36):self.check_attention_output(self.queries[fb.batch_size][layer],engine.output[layer],layer,fb)

    def finish(self):
        result=frozen.migration.TimingBackend.finish(self)
        for collection in [self.releases,self.restorations]:
            for row in collection:
                a,z=row.pop('_events');row['gpu_stream_ms']=a.elapsed_time(z)
        return result
