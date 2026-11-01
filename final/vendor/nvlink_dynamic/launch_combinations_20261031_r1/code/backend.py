"""Allocation-policy experiments around frozen attention/timing implementations."""

from pathlib import Path as _FinalPath
_SLACK_FINAL_ROOT = next(p for p in _FinalPath(__file__).resolve().parents if (p / 'release.json').is_file())
_SLACK_PROJECT_ROOT = _SLACK_FINAL_ROOT.parent
import importlib.util
from pathlib import Path
import sys
import time
import torch

SOURCE=Path(str(_SLACK_FINAL_ROOT / 'vendor/nvlink_prefill_timing/launch_20261031_r2/code/backend.py'))
spec=importlib.util.spec_from_file_location('frozen_timing_backend_r2',SOURCE)
frozen=importlib.util.module_from_spec(spec);sys.modules[spec.name]=frozen;spec.loader.exec_module(frozen)
scope=frozen.scope


class TimingBackend(frozen.TimingBackend):
    def configure(self,variant):
        if not hasattr(self,'physical_allocator'):
            self.physical_allocator=self.raw.token_to_kv_pool_allocator
            self.native_store=self.raw.token_to_kv_pool.set_kv_buffer
            control=self
            def store(layer,loc,k,v,*args,**kwargs):
                if not control.variant.get('direct_remote',False):return control.native_store(layer,loc,k,v,*args,**kwargs)
                from sglang.srt.mem_cache.memory_pool import unwrap_write_loc
                from direct_write import write
                locations,_,_=unwrap_write_loc(loc)
                i=layer.layer_id
                return write(control.raw.token_to_kv_pool.get_kv_buffer(i),control.peer[i],locations,k,v,control.cfg['kv_capacity_tokens'])
            self.raw.token_to_kv_pool.set_kv_buffer=store
        if getattr(self,'variant',{}).get('direct_remote',False)!=variant.get('direct_remote',False):self.engine=None
        self.variant=dict(variant)
        if variant.get('direct_remote',False):
            from direct_write import Allocator
            self.raw.token_to_kv_pool_allocator=Allocator(self,self.physical_allocator)
        else:self.raw.token_to_kv_pool_allocator=self.physical_allocator

    def begin(self,fb):
        if not self.variant.get('direct_remote',False):
            result=super().begin(fb)
            if self.variant.get('async_copy',False):
                qlens=fb.extend_seq_lens_cpu if fb.forward_mode.is_extend() else [1]*fb.batch_size
                for rid,length,q in zip(self.requests,self.lengths,qlens):
                    self.created.update({(rid,pos):len(self.frames) for pos in range(length-q,length)})
            return result
        self.requests=[int(req.kv.req_pool_idx) for req in self.req_objects]
        self.lengths=fb.seq_lens_cpu.tolist()
        if fb.forward_mode.is_extend():qlens,prefix=fb.extend_seq_lens_cpu,fb.extend_prefix_lens_cpu
        else:qlens,prefix=[1]*fb.batch_size,[n-1 for n in self.lengths]
        offset=0;c=self.cfg['kv_capacity_tokens']
        for rid,start,length in zip(self.requests,prefix,qlens):
            loc=fb.out_cache_loc[offset:offset+length];remote=loc>c
            self.slot[rid,start:start+length].copy_(torch.where(remote,loc-c-1,loc))
            self.tier[rid,start:start+length].copy_(remote.to(torch.int32))
            local_length=max(0,min(length,self.last_allocation_near-offset))
            self.residents.extend((rid,pos) for pos in range(start,start+local_length))
            if self.variant.get('async_copy',False):
                self.created.update({(rid,pos):len(self.frames) for pos in range(start,start+local_length)})
            self.live[rid]=start+length;offset+=length
        if self.engine is not None and fb.forward_mode.is_decode():self.engine.prepare(fb)

    def audit(self):
        if not self.variant.get('direct_remote',False):return super().audit()
        local=[];remote=[];requests=[];c=self.cfg['kv_capacity_tokens']
        for rid,length in self.live.items():
            tiers=self.tier[rid,:length];slots=self.slot[rid,:length];far=tiers==1;near=tiers==0
            assert (far|near).all().item()
            expected=torch.where(far,slots+c+1,slots)
            assert torch.equal(expected,self.raw.req_to_token_pool.req_to_token[rid,:length])
            local.append(slots[near]);remote.append(slots[far])
            requests.append(dict(request=rid,context=length,local=int(near.sum()),remote=int(far.sum())))
        near=torch.cat(local);far=torch.cat(remote)
        assert torch.unique(near).numel()==near.numel() and torch.unique(far).numel()==far.numel()
        assert near.numel()==c-self.physical_allocator.available_size()
        assert far.numel()+len(self.peer_free)==self.peer.shape[2]
        assert (near>0).all().item() and (near<=c).all().item()
        assert (far>=0).all().item() and (far<self.peer.shape[2]).all().item()
        return dict(local_active_tokens=near.numel(),remote_active_tokens=far.numel(),logical_tokens=sum(self.live.values()),per_request=requests,page_tables_exact=True,unique_local_slots=True,unique_peer_slots=True,remote_local_shadow_present=False)

    def reset(self,batch,profile=False):
        super().reset(batch,profile)
        self.created={};self.copy_references=[]
        if self.variant.get('async_copy',False) and not hasattr(self,'copy_stream'):
            self.copy_stream=torch.cuda.Stream(device=self.raw.device)

    def target_headroom(self,needed):
        window=self.variant.get('reserve_steps',0)
        remaining=self.context.get('remaining_decode_steps',0)
        reserve=self.batch*min(window,remaining)
        if self.context.get('phase')=='prefill' and self.context.get('prefill_last_chunk'):
            return needed+reserve,True
        if self.context.get('phase')=='decode':return max(needed,reserve),False
        return needed,False

    def ensure_headroom(self,needed):
        start=time.perf_counter();free=self.raw.token_to_kv_pool_allocator.available_size()
        if self.variant.get('direct_remote',False):
            assert needed<=free,(needed,free)
            # Pure direct writing needs no old-page migration. The hybrid
            # spills appended input directly, then reserves local decode slots.
            if not self.variant.get('reserve_steps',0) or (self.context['phase']=='prefill' and not self.context.get('reserve_after_write',False)):
                return (time.perf_counter()-start)*1000
            free=self.physical_allocator.available_size()
        target,at_input_boundary=self.target_headroom(needed)
        required=max(0,needed-free)
        assert required<=len(self.peer_free),(needed,free,len(self.peer_free))
        # During decode refill only on exhaustion, rather than topping up after
        # every token. The last prefill chunk may reserve the next known suffix.
        if free>=needed and not (at_input_boundary and free<target):return (time.perf_counter()-start)*1000
        deficit=min(max(0,target-free),len(self.peer_free))
        if not deficit:return (time.perf_counter()-start)*1000
        if self.variant.get('async_copy',False):return self.migrate_async(deficit,start)
        events=self.events();events[0].record();uid=len(self.frames)
        with scope(f'SLACK_COST_MIGRATION_{uid}',self.profile):
            selected=[self.residents.popleft() for _ in range(deficit)]
            rid=torch.tensor([r for r,p in selected],device=self.raw.device,dtype=torch.int64)
            pos=torch.tensor([p for r,p in selected],device=self.raw.device,dtype=torch.int64)
            source=self.slot[rid,pos].long()
            targets=torch.tensor([self.peer_free.popleft() for _ in selected],device=self.raw.device,dtype=torch.int64)
            self.copy_pages(source,targets)
            torch.cuda.synchronize()
            self.raw.token_to_kv_pool_allocator.free(source)
            self.raw.req_to_token_pool.req_to_token[rid,pos]=(targets+self.cfg['kv_capacity_tokens']+1).to(torch.int32) if self.variant.get('direct_remote',False) else 0
            self.tier[rid,pos]=1;self.slot[rid,pos]=targets.to(torch.int32)
        events[1].record();wall=(time.perf_counter()-start)*1000
        self.migrations.append(dict(uid=uid,**self.context,tokens=deficit,wall_ms=wall,
            remote_active=self.remote_active,_events=events))
        return wall

    def migrate_async(self,deficit,start):
        uid=len(self.frames);selected=[self.residents.popleft() for _ in range(deficit)]
        assert self.frames
        previous=self.frames[-1]
        previous_uid=previous['uid']
        old_before_previous=all(self.created[item]<previous_uid for item in selected)
        # KV from earlier forwards is immutable and ready at the preceding
        # model's start. A page first written by that model needs its end event.
        ready=previous['_model_events'][0 if old_before_previous else 1]
        events=self.events();done=torch.cuda.Event();main=torch.cuda.current_stream()
        with scope(f'SLACK_COST_MIGRATION_{uid}',self.profile):
            self.copy_stream.wait_event(ready)
            with torch.cuda.stream(self.copy_stream):
                events[0].record()
                rid=torch.tensor([r for r,p in selected],device=self.raw.device,dtype=torch.int64)
                pos=torch.tensor([p for r,p in selected],device=self.raw.device,dtype=torch.int64)
                source=self.slot[rid,pos].long()
                targets=torch.tensor([self.peer_free.popleft() for _ in selected],device=self.raw.device,dtype=torch.int64)
                self.copy_pages(source,targets)
                events[1].record();done.record()
            # All earlier readers are ahead of this main-stream fence. Both
            # completed copy and completed readers precede slot reuse/writes.
            main.wait_event(done)
            self.raw.token_to_kv_pool_allocator.free(source)
            self.raw.req_to_token_pool.req_to_token[rid,pos]=(targets+self.cfg['kv_capacity_tokens']+1).to(torch.int32) if self.variant.get('direct_remote',False) else 0
            self.tier[rid,pos]=1;self.slot[rid,pos]=targets.to(torch.int32)
            # Keep cross-stream metadata alive until finish() synchronizes.
            self.copy_references.append((rid,pos,source,targets,done))
        wall=(time.perf_counter()-start)*1000
        self.migrations.append(dict(uid=uid,**self.context,tokens=deficit,wall_ms=wall,
            remote_active=self.remote_active,_events=events,async_copy=True,
            readiness='previous_model_start' if old_before_previous else 'previous_model_end',
            ownership_commit='main stream after copy_done and preceding readers',
            reserved_peer_tokens=deficit))
        return wall

    def copy_pages(self,source,targets):
        if self.variant['copy_impl']=='torch':
            for i in range(36):
                for plane,local in enumerate(self.raw.token_to_kv_pool.get_kv_buffer(i)):
                    self.peer[i,plane].index_copy_(0,targets,local.index_select(0,source))
        else:
            from fused_copy import copy_pages
            if not hasattr(self,'local_pointers'):
                self.local_pointers=torch.tensor([v.data_ptr() for i in range(36) for v in self.raw.token_to_kv_pool.get_kv_buffer(i)],device=self.raw.device,dtype=torch.int64)
            copy_pages(self.local_pointers,self.peer,source,targets)
