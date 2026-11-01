"""Timing adapter around unchanged r5 kernels; no numerical-oracle work."""
from collections import deque
from contextlib import nullcontext
import time
import types
import torch
from paged_decode import PagedDecode
from paged_prefill import prefill
from tiered_backend import TieredBackend


def scope(name,enabled):
    return torch.profiler.record_function(name) if enabled else nullcontext()


class TimingBackend:
    audit=TieredBackend.audit

    def __init__(self,raw,peer,cfg,old_control,case):
        self.raw,self.peer,self.cfg,self.case=raw,peer,cfg,case
        self.enabled=case!='native_local';self.profile=False;self.engine=None
        old_control.backend.forward_decode=old_control.native
        self.backend=raw.attn_backend
        assert self.backend is (raw.decode_attn_backend or raw.attn_backend)
        native_extend=self.backend.forward_extend;native_decode=self.backend.forward_decode
        native_forward=raw.forward;native_sample=raw.sample
        shape=raw.req_to_token_pool.req_to_token.shape
        self.tier=torch.empty(shape,device=raw.device,dtype=torch.int32)
        self.slot=torch.empty_like(self.tier)
        self.graph_captures=0;self.context={};self.req_objects=[]
        control=self
        def extend(backend,q,k,v,layer,fb,save_kv_cache=True):
            if not control.enabled:return native_extend(q,k,v,layer,fb,save_kv_cache=save_kv_cache)
            backend.token_to_kv_pool.set_kv_buffer(layer,fb.out_cache_loc,k,v,layer.k_scale_float,layer.v_scale_float)
            lk,lv=raw.token_to_kv_pool.get_kv_buffer(layer.layer_id)
            output,_=prefill(q.contiguous().view(-1,32,128),lk,lv,peer[layer.layer_id,0],peer[layer.layer_id,1],
                control.tier,control.slot,fb.req_pool_indices,fb.seq_lens,fb.extend_start_loc,
                fb.extend_seq_lens,fb.extend_prefix_lens,max(fb.extend_seq_lens_cpu),layer.scaling)
            return output.view(-1,4096)
        def decode(backend,q,k,v,layer,fb,save_kv_cache=True):
            if not control.enabled:return native_decode(q,k,v,layer,fb,save_kv_cache=save_kv_cache)
            backend.token_to_kv_pool.set_kv_buffer(layer,fb.out_cache_loc,k,v,layer.k_scale_float,layer.v_scale_float)
            return control.engine.run(q.contiguous().view(-1,32,128),layer.layer_id).view(-1,4096)
        def forward(fb,*args,**kwargs):
            entry=time.perf_counter();uid=len(control.frames)
            page_events=None;page_wall=0.
            if control.enabled:
                page_events=control.events();t=time.perf_counter();page_events[0].record()
                with scope(f'SLACK_COST_PAGE_{uid}',control.profile):control.begin(fb)
                page_events[1].record();page_wall=(time.perf_counter()-t)*1000
            model_events=control.events()
            with scope(f'SLACK_PHASE_FORWARD_{case}_sample{uid}',control.profile):
                model_events[0].record();result=native_forward(fb,*args,**kwargs);model_events[1].record()
            control.frames.append(dict(uid=uid,**control.context,forward_entry=entry,
                prefix_lengths=(list(fb.extend_prefix_lens_cpu) if fb.forward_mode.is_extend() else [n-1 for n in fb.seq_lens_cpu.tolist()]),
                query_lengths=(list(fb.extend_seq_lens_cpu) if fb.forward_mode.is_extend() else [1]*fb.batch_size),
                context_lengths=fb.seq_lens_cpu.tolist(),remote_active=control.remote_active,
                custom_page_wall_ms=page_wall,_model_events=model_events,_page_events=page_events))
            return result
        def sample(*args,**kwargs):
            uid=max(0,len(control.frames)-1)
            with scope(f'SLACK_COST_SAMPLE_{uid}',control.profile):return native_sample(*args,**kwargs)
        self.backend.forward_extend=types.MethodType(extend,self.backend)
        self.backend.forward_decode=types.MethodType(decode,self.backend)
        raw.forward=forward;raw.sample=sample

    @staticmethod
    def events():return torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)

    @property
    def remote_active(self):return self.peer.shape[2]-len(self.peer_free)

    def reset(self,batch,profile=False):
        self.batch=batch;self.profile=profile;self.frames=[];self.migrations=[]
        self.requests=[];self.lengths=[];self.live={};self.residents=deque()
        self.peer_free=deque(range(self.peer.shape[2]))
        if self.enabled:self.tier.fill_(-1);self.slot.zero_()

    def ensure_headroom(self,needed):
        start=time.perf_counter();free=self.raw.token_to_kv_pool_allocator.available_size()
        deficit=max(0,needed-free)
        assert deficit<=len(self.peer_free),(needed,free,len(self.peer_free))
        if not deficit:return (time.perf_counter()-start)*1000
        assert self.case=='tiered_peer'
        events=self.events();events[0].record()
        uid=len(self.frames)
        with scope(f'SLACK_COST_MIGRATION_{uid}',self.profile):
            # The deque contains resident pages only; migration removes entries.
            selected=[self.residents.popleft() for _ in range(deficit)]
            rid=torch.tensor([r for r,p in selected],device=self.raw.device,dtype=torch.int64)
            pos=torch.tensor([p for r,p in selected],device=self.raw.device,dtype=torch.int64)
            source=self.slot[rid,pos].long()
            targets=torch.tensor([self.peer_free.popleft() for _ in selected],device=self.raw.device,dtype=torch.int64)
            for i in range(36):
                for plane,local in enumerate(self.raw.token_to_kv_pool.get_kv_buffer(i)):
                    self.peer[i,plane].index_copy_(0,targets,local.index_select(0,source))
            # Retain the validated completion boundary before releasing source slots.
            torch.cuda.synchronize()
            self.raw.token_to_kv_pool_allocator.free(source)
            self.raw.req_to_token_pool.req_to_token[rid,pos]=0
            self.tier[rid,pos]=1;self.slot[rid,pos]=targets.to(torch.int32)
        events[1].record();wall=(time.perf_counter()-start)*1000
        self.migrations.append(dict(uid=uid,**self.context,tokens=deficit,wall_ms=wall,
            remote_active=self.remote_active,_events=events))
        return wall

    def begin(self,fb):
        self.requests=[int(req.kv.req_pool_idx) for req in self.req_objects]
        self.lengths=fb.seq_lens_cpu.tolist()
        if fb.forward_mode.is_extend():qlens,prefix=fb.extend_seq_lens_cpu,fb.extend_prefix_lens_cpu
        else:qlens,prefix=[1]*fb.batch_size,[n-1 for n in self.lengths]
        offset=0
        for rid,start,length in zip(self.requests,prefix,qlens):
            self.slot[rid,start:start+length].copy_(fb.out_cache_loc[offset:offset+length])
            self.tier[rid,start:start+length]=0
            self.residents.extend((rid,pos) for pos in range(start,start+length))
            self.live[rid]=start+length;offset+=length
        if self.engine is not None and fb.forward_mode.is_decode():self.engine.prepare(fb)

    def enable_decode(self):
        if not self.enabled or self.engine is not None:return
        self.engine=PagedDecode(self.raw,self.peer,self.tier,self.slot,self.batch,self.cfg['decode_splits'])
        self.engine.requests.copy_(torch.tensor(self.requests,device=self.raw.device,dtype=torch.int32))
        self.engine.seq.copy_(torch.tensor(self.lengths,device=self.raw.device,dtype=torch.int32))
        self.raw.decode_cuda_graph_runner.capture_bs=[self.batch]
        self.raw.decode_cuda_graph_runner.capture();torch.cuda.synchronize();self.graph_captures+=1

    def finish(self):
        torch.cuda.synchronize()
        for f in self.frames:
            a,z=f.pop('_model_events');f['model_event_ms']=a.elapsed_time(z)
            p=f.pop('_page_events');f['custom_page_gpu_ms']=p[0].elapsed_time(p[1]) if p else 0.
        for m in self.migrations:
            a,z=m.pop('_events');m['gpu_stream_ms']=a.elapsed_time(z)
        if self.enabled:return self.audit()
        used=self.cfg['kv_capacity_tokens']-self.raw.token_to_kv_pool_allocator.available_size()
        return dict(local_active_tokens=used,remote_active_tokens=0,logical_tokens=used)
