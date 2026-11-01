"""Fixed native pool with real peer prefixes; custom attention only on decode."""
import types

import torch
from flashinfer import BatchDecodeWithPagedKVCacheWrapper

from optimized_pages import OptimizedAttention


class Controller:
    def __init__(self,raw,peer,cfg,workload):
        self.raw,self.peer,self.cfg,self.workload=raw,peer,cfg,workload
        self.backend=raw.decode_attn_backend or raw.attn_backend
        self.native=self.backend.forward_decode
        self.mode='native';self.trace=False
        self.active={};self.engines={};self.buffers={}
        self.context=1;self.batch=1;self.remote_context=0
        controller=self

        def decode(backend,q,k,v,layer,forward_batch,save_kv_cache=True):
            b=q.numel()//4096;i=layer.layer_id
            query=q.contiguous().view(b,32,128) if controller.trace or controller.mode!='native' else None
            if controller.trace:controller.buffers[b][0][i].copy_(query)
            if controller.mode=='native':
                value=controller.native(q,k,v,layer,forward_batch,save_kv_cache=save_kv_cache)
            else:
                assert save_kv_cache and k is not None and v is not None
                backend.token_to_kv_pool.set_kv_buffer(layer,forward_batch.out_cache_loc,k,v,
                                                       layer.k_scale_float,layer.v_scale_float)
                value=controller.active[b].run(query,i).view(-1,4096)
            if controller.trace:controller.buffers[b][1][i].copy_(value.view(b,32,128))
            return value
        self.backend.forward_decode=types.MethodType(decode,self.backend)

    def add_engine(self,batch,remote,tuning):
        config=dict(self.cfg,batch=batch,local_full_context=self.workload['final_context']-remote,
                    remote_context_per_request=remote)
        engine=OptimizedAttention(self,config,tuning)
        engine.seq.fill_(self.workload['final_context'])
        self.engines[(batch,remote,engine.label)]=engine
        if batch not in self.buffers:
            q=torch.empty((36,batch,32,128),device=self.raw.device,dtype=torch.bfloat16)
            self.buffers[batch]=(q,torch.empty_like(q))
        return engine

    def activate(self,mode,engines,trace=False):
        self.mode,self.trace=mode,trace
        self.active=dict(engines)
        torch.cuda.synchronize()
        self.raw.decode_cuda_graph_runner.capture()
        torch.cuda.synchronize()

    def plan(self,batch,indices=True):
        self.batch=len(batch.seq_lens_cpu)
        self.context=int(batch.seq_lens_cpu[0])
        assert batch.seq_lens_cpu.tolist()==[self.context]*self.batch
        self.remote_context=0 if self.mode=='native' else self.active[self.batch].remote
        if indices:
            self.local_indices=self.raw.req_to_token_pool.req_to_token[
                batch.req_pool_indices,self.remote_context:self.context].reshape(-1).to(torch.int64)
        if self.mode!='native':
            engine=self.active[self.batch]
            previous=getattr(engine,'last_context',None)
            if getattr(engine,'last_batch',None) is batch and self.context==previous:
                pass
            elif getattr(engine,'last_batch',None) is batch and previous is not None and self.context==previous+1:
                engine.seq.copy_(batch.seq_lens)
                source=self.raw.req_to_token_pool.req_to_token[batch.req_pool_indices,previous:self.context]
                engine.slot[:,previous:self.context].copy_(source)
            else:engine.update(batch)
            engine.last_batch,engine.last_context=batch,self.context
        self.queries,self.outputs=self.buffers[self.batch]

    def audit_pages(self,batch):
        if self.mode=='native':return dict(custom_table=False)
        b=self.batch;e=self.active[b];r=e.remote;length=self.context
        source=self.raw.req_to_token_pool.req_to_token[batch.req_pool_indices,r:length]
        assert torch.equal(e.slot[:,r:length],source)
        assert torch.equal(e.seq,batch.seq_lens)
        if r:
            expected=torch.arange(b*r,device=self.raw.device,dtype=torch.int32).view(b,r)
            assert torch.equal(e.slot[:,:r],expected)
        return dict(custom_table=True,exact_local_slot_match=True,exact_peer_slot_match=True,
                    local_slots=b*(length-r),peer_slots=b*r)

    def offload(self,batch):
        b=len(batch.seq_lens_cpu)
        r=0 if self.mode=='native' else self.active[b].remote
        if not r:return dict(reclaimed_slots=0,exact_payload_layers=0)
        table=self.raw.req_to_token_pool.req_to_token
        slots=table[batch.req_pool_indices,:r].reshape(-1).to(torch.int64)
        assert slots.numel()==b*r and slots.min().item()>0 and torch.unique(slots).numel()==slots.numel()
        for i in range(36):
            for plane,local in enumerate(self.raw.token_to_kv_pool.get_kv_buffer(i)):
                expected=local.index_select(0,slots)
                self.peer[i,plane,:b*r].copy_(expected)
                assert torch.equal(self.peer[i,plane,:b*r],expected)
        torch.cuda.synchronize()
        before=self.raw.token_to_kv_pool_allocator.available_size()
        self.raw.token_to_kv_pool_allocator.free(slots)
        after=self.raw.token_to_kv_pool_allocator.available_size()
        assert after-before==slots.numel()
        table[batch.req_pool_indices[:,None],torch.arange(r,device=self.raw.device)[None,:]]=0
        self.freed_slots=slots
        self.plan(batch)
        assert not torch.isin(self.local_indices,slots).any().item()
        return dict(reclaimed_slots=slots.numel(),exact_payload_layers=36,local_shadow_present=False)

    def check(self):
        """Native oracle stages only one layer, outside all throughput intervals."""
        b,r,length=self.batch,self.remote_context,self.context
        total=b*length
        wrapper=BatchDecodeWithPagedKVCacheWrapper(
            torch.zeros(128*1024*1024,device=self.raw.device,dtype=torch.uint8),'NHD',
            use_tensor_cores=self.backend.decode_use_tensor_cores,backend=self.backend.decode_backend)
        wrapper.plan(torch.arange(b+1,dtype=torch.int32)*length,
                     torch.arange(total,dtype=torch.int32),torch.ones(b,dtype=torch.int32),
                     32,8,128,1,pos_encoding_mode='NONE',q_data_type=torch.bfloat16,
                     kv_data_type=torch.bfloat16,sm_scale=128**-.5)
        k=torch.empty((total,1,8,128),device=self.raw.device,dtype=torch.bfloat16);v=torch.empty_like(k)
        expected_outputs=torch.empty_like(self.outputs)
        errors=[]
        for i in range(36):
            for plane,dest in enumerate((k,v)):
                shape=dest.view(b,length,8,128)
                if r:shape[:,:r].copy_(self.peer[i,plane,:b*r].view(b,r,8,128))
                source=self.raw.token_to_kv_pool.get_kv_buffer(i)[plane]
                shape[:,r:].copy_(source.index_select(0,self.local_indices).view(b,length-r,8,128))
            layer=self.raw.model.model.layers[i].self_attn.attn
            expected=wrapper.run(self.queries[i],(k,v),k_scale=layer.k_scale_float,v_scale=layer.v_scale_float)
            torch.testing.assert_close(self.outputs[i],expected,atol=self.cfg['reference_atol'],rtol=self.cfg['reference_rtol'])
            expected_outputs[i].copy_(expected)
            errors.append(float((self.outputs[i]-expected).abs().max().item()))
        torch.cuda.synchronize()
        return dict(context=length,layers=36,max_abs_error=max(errors)),expected_outputs
