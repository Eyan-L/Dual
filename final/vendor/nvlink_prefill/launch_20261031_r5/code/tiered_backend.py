"""Chunked prefill integration; local-first allocation, peer-prefix spill on demand.

Only this prototype's controlled benchmark uses zero sentinel entries for remote
pages in SGLang's local table. Tier/Slot remain authoritative for custom kernels.
"""
from collections import deque
import json
import types
import torch
from flashinfer import BatchPrefillWithPagedKVCacheWrapper
from paged_decode import PagedDecode
from paged_prefill import prefill


class KVCapacityExhausted(RuntimeError):
    pass


class TieredBackend:
    def __init__(self,raw,peer,cfg,old_control):
        self.raw,self.peer,self.cfg=raw,peer,cfg
        # Remove the old decode-only wrapper; leave its immutable source alone.
        old_control.backend.forward_decode=old_control.native
        self.backend=raw.attn_backend
        assert self.backend is (raw.decode_attn_backend or raw.attn_backend)
        self.original_extend=self.backend.forward_extend;self.original_decode=self.backend.forward_decode
        self.original_forward=raw.forward
        self.enabled=False;self.engine=None
        shape=raw.req_to_token_pool.req_to_token.shape
        self.tier=torch.zeros(shape,device=raw.device,dtype=torch.int32)
        self.slot=torch.zeros_like(self.tier)
        self.workspace=torch.empty(128*1024*1024,device=raw.device,dtype=torch.uint8)
        control=self
        def extend(backend,q,k,v,layer,fb,save_kv_cache=True):
            if not control.enabled:return control.original_extend(q,k,v,layer,fb,save_kv_cache=save_kv_cache)
            assert save_kv_cache and k is not None and v is not None
            assert not layer.is_cross_attention and layer.logit_cap in (None,0,0.)
            backend.token_to_kv_pool.set_kv_buffer(layer,fb.out_cache_loc,k,v,layer.k_scale_float,layer.v_scale_float)
            query=q.contiguous().view(-1,32,128);i=layer.layer_id
            lk,lv=raw.token_to_kv_pool.get_kv_buffer(i)
            output,kernel=prefill(query,lk,lv,peer[i,0],peer[i,1],control.tier,control.slot,
                fb.req_pool_indices,fb.seq_lens,fb.extend_start_loc,fb.extend_seq_lens,fb.extend_prefix_lens,
                max(fb.extend_seq_lens_cpu),layer.scaling)
            control.kernel_metadata=dict(registers=kernel.n_regs,spills=kernel.n_spills,shared_bytes=kernel.metadata.shared)
            alternate=None
            if len(control.forwards)==1:
                alternate,_=prefill(query,lk,lv,peer[i,0],peer[i,1],control.tier,control.slot,
                    fb.req_pool_indices,fb.seq_lens,fb.extend_start_loc,fb.extend_seq_lens,fb.extend_prefix_lens,
                    max(fb.extend_seq_lens_cpu),layer.scaling,probability_residual=False)
            control.check_layer(query,output,i,'prefill',alternate)
            return output.view(-1,4096)
        def decode(backend,q,k,v,layer,fb,save_kv_cache=True):
            if not control.enabled:return control.original_decode(q,k,v,layer,fb,save_kv_cache=save_kv_cache)
            assert save_kv_cache and control.engine is not None
            backend.token_to_kv_pool.set_kv_buffer(layer,fb.out_cache_loc,k,v,layer.k_scale_float,layer.v_scale_float)
            query=q.contiguous().view(-1,32,128);output=control.engine.run(query,layer.layer_id)
            control.decode_queries[layer.layer_id].copy_(query);control.decode_outputs[layer.layer_id].copy_(output)
            return output.view(-1,4096)
        def forward(fb,*args,**kwargs):
            if control.enabled and not torch.cuda.is_current_stream_capturing():control.begin(fb)
            return control.original_forward(fb,*args,**kwargs)
        self.backend.forward_extend=types.MethodType(extend,self.backend)
        self.backend.forward_decode=types.MethodType(decode,self.backend)
        raw.forward=forward

    def reset(self,batch):
        self.batch=batch;self.tier.fill_(-1);self.slot.zero_();self.engine=None
        self.requests=[];self.lengths=[];self.live={};self.residents=deque()
        self.peer_free=deque(range(self.peer.shape[2]));self.migrations=[]
        self.checks=[];self.forwards=[];self.decode_graph_captures=0;self.reference_audits=[]

    def ensure_headroom(self,needed):
        free=self.raw.token_to_kv_pool_allocator.available_size()
        deficit=max(0,needed-free)
        if deficit>len(self.peer_free):
            raise KVCapacityExhausted(f'Need{needed} local slots, only{free} free and{len(self.peer_free)} peer slots remain')
        if not deficit:return
        selected=[]
        while len(selected)<deficit:
            assert self.residents
            rid,pos=self.residents.popleft()
            if rid in self.live and int(self.tier[rid,pos])==0:selected.append((rid,pos))
        rid=torch.tensor([r for r,p in selected],device=self.raw.device,dtype=torch.int64)
        pos=torch.tensor([p for r,p in selected],device=self.raw.device,dtype=torch.int64)
        table=self.raw.req_to_token_pool.req_to_token
        source=self.slot[rid,pos].long()
        assert source.min().item()>0 and torch.unique(source).numel()==source.numel()
        targets=torch.tensor([self.peer_free.popleft() for _ in selected],device=self.raw.device,dtype=torch.int64)
        for i in range(36):
            for plane,local in enumerate(self.raw.token_to_kv_pool.get_kv_buffer(i)):
                payload=local.index_select(0,source)
                self.peer[i,plane].index_copy_(0,targets,payload)
                assert torch.equal(self.peer[i,plane].index_select(0,targets),payload)
                local.index_fill_(0,source,float('nan'))
        torch.cuda.synchronize()
        self.raw.token_to_kv_pool_allocator.free(source)
        table[rid,pos]=0;self.tier[rid,pos]=1;self.slot[rid,pos]=targets.to(torch.int32)
        self.migrations.append(dict(released_local_slots=source.numel(),exact_payload_layers=36,
            request_positions=selected,old_local_payload_poisoned=True,
            remote_active=self.peer.shape[2]-len(self.peer_free),
            available_after=self.raw.token_to_kv_pool_allocator.available_size()))

    def begin(self,fb):
        self.requests=fb.req_pool_indices.cpu().tolist();self.lengths=fb.seq_lens_cpu.tolist()
        if fb.forward_mode.is_extend():
            qlens=fb.extend_seq_lens_cpu;prefix=fb.extend_prefix_lens_cpu
        else:
            qlens=[1]*self.batch;prefix=[n-1 for n in self.lengths]
        offset=0
        for rid,start,length in zip(self.requests,prefix,qlens):
            loc=fb.out_cache_loc[offset:offset+length]
            assert (loc>0).all().item()
            self.slot[rid,start:start+length].copy_(loc)
            assert (self.tier[rid,start:start+length]==-1).all().item()
            self.tier[rid,start:start+length]=0
            self.residents.extend((rid,pos) for pos in range(start,start+length))
            self.live[rid]=start+length;offset+=length
        assert offset==fb.input_ids.numel()
        self.qlens,self.prefix=qlens,prefix
        self.oracle=BatchPrefillWithPagedKVCacheWrapper(self.workspace,'NHD')
        qi=torch.tensor([0]+list(__import__('itertools').accumulate(qlens)),dtype=torch.int32)
        ki=torch.tensor([0]+list(__import__('itertools').accumulate(self.lengths)),dtype=torch.int32)
        self.oracle.plan(qi,ki,torch.arange(sum(self.lengths),dtype=torch.int32),
            torch.ones(len(self.requests),dtype=torch.int32),32,8,128,1,causal=True,
            q_data_type=torch.bfloat16,kv_data_type=torch.bfloat16,sm_scale=128**-.5)
        if self.engine is not None and fb.forward_mode.is_decode():self.engine.prepare(fb)
        self.forwards.append(dict(mode='prefill' if fb.forward_mode.is_extend() else 'decode',
            lengths=list(self.lengths),query_lengths=list(qlens),
            remote_active=self.peer.shape[2]-len(self.peer_free),
            remote_per_request={str(r):int((self.tier[r,:n]==1).sum()) for r,n in self.live.items()},
            local_active=self.cfg['kv_capacity_tokens']-self.raw.token_to_kv_pool_allocator.available_size()))

    def logical_kv(self,layer):
        lk,lv=self.raw.token_to_kv_pool.get_kv_buffer(layer);ks=[];vs=[]
        for rid,length in zip(self.requests,self.lengths):
            slots=self.slot[rid,:length].long();remote=self.tier[rid,:length].bool()
            k=torch.empty((length,8,128),device=self.raw.device,dtype=torch.bfloat16);v=torch.empty_like(k)
            for dest,local,peer in ((k,lk,self.peer[layer,0]),(v,lv,self.peer[layer,1])):
                dest[~remote]=local.index_select(0,slots[~remote]);dest[remote]=peer.index_select(0,slots[remote])
            ks.append(k);vs.append(v)
        return torch.cat(ks).unsqueeze(1),torch.cat(vs).unsqueeze(1)

    def check_layer(self,q,out,layer,mode,alternate=None):
        kv=self.logical_kv(layer)
        expected=self.oracle.run(q,kv)
        mismatch=~torch.isclose(out,expected,atol=self.cfg['reference_atol'],rtol=self.cfg['reference_rtol'])
        count=int(mismatch.sum())
        audit=self.audit_fp64(q,out,expected,kv,mismatch,layer,mode,alternate) if count else None
        if audit is not None:
            assert all(v['candidate_fp64_failures']==0 for v in audit['rows']), 'Candidate failed independent FP64 reference at original tolerance'
        else:torch.testing.assert_close(out,expected,atol=self.cfg['reference_atol'],rtol=self.cfg['reference_rtol'])
        err=out.float()-expected.float()
        row=dict(mode=mode,layer=layer,context=list(self.lengths),
            native_elementwise_pass=(count==0),native_mismatch_count=count,
            fp64_adjudicated=(audit is not None),
            max_abs_error=float(err.abs().max()),relative_l2=float(err.norm()/expected.float().norm().clamp_min(1.e-12)))
        if alternate is not None:
            row['plain_bf16_probability_mismatches']=int((~torch.isclose(alternate,expected,atol=self.cfg['reference_atol'],rtol=self.cfg['reference_rtol'])).sum())
            row['plain_bf16_probability_relative_l2']=float((alternate.float()-expected.float()).norm()/expected.float().norm().clamp_min(1.e-12))
        self.checks.append(row)

    def audit_fp64(self,q,out,expected,kv,mismatch,layer,mode,alternate):
        """Independent mathematical reference at every disagreeing query/head.

        Keep native disagreement visible. Candidate must pass the original
        tolerances against FP64 over all128 channels of each disputed row.
        FP64 elementwise products/reductions avoid Tensor Core/TF32 arithmetic.
        """
        coordinates=mismatch.nonzero().cpu().tolist()
        rows=[]
        for token,head in sorted({(t,h) for t,h,d in coordinates}):
            qo=0;ko=0
            for b,n in enumerate(self.qlens):
                if token<qo+n:break
                qo+=n;ko+=self.lengths[b]
            visible=self.prefix[b]+token-qo+1
            key=kv[0][ko:ko+visible,0,head//4].double()
            value=kv[1][ko:ko+visible,0,head//4].double()
            score=(key*q[token,head].double()).sum(-1)*(128**-.5)
            truth=(score.softmax(0)[:,None]*value).sum(0)
            candidate=out[token,head].double();native=expected[token,head].double()
            bad_candidate=~torch.isclose(candidate,truth,atol=self.cfg['reference_atol'],rtol=self.cfg['reference_rtol'])
            bad_native=~torch.isclose(native,truth,atol=self.cfg['reference_atol'],rtol=self.cfg['reference_rtol'])
            details=[]
            for t,h,d in coordinates:
                if (t,h)!=(token,head):continue
                details.append(dict(channel=d,candidate=float(candidate[d]),native=float(native[d]),
                    fp64=float(truth[d]),plain_bf16=None if alternate is None else float(alternate[t,h,d]),
                    candidate_error=float(abs(candidate[d]-truth[d])),native_error=float(abs(native[d]-truth[d]))))
            rows.append(dict(token=token,head=head,request=self.requests[b],visible_keys=visible,
                candidate_fp64_failures=int(bad_candidate.sum()),native_fp64_failures=int(bad_native.sum()),
                candidate_max_abs_error=float((candidate-truth).abs().max()),
                native_max_abs_error=float((native-truth).abs().max()),disagreements=details))
        report=dict(status='FP64_ADJUDICATION_RECORDED',mode=mode,layer=layer,
            forward_index=len(self.forwards)-1,query_lengths=self.qlens,prefix_lengths=self.prefix,
            context=self.lengths,native_mismatched_elements=len(coordinates),total_elements=out.numel(),
            remote_active=self.peer.shape[2]-len(self.peer_free),rows=rows,
            reference='FP64 sum of elementwise Q*K, causal softmax, then FP64 weighted V. Same BF16 Q/K/V as both implementations. Candidate is checked with unchanged atol/rtol on all128 output channels of every disputed query/head; native disagreement is explicitly retained.')
        self.reference_audits.append(report)
        (self.diagnostic_root/'reference_adjudications.json').write_text(json.dumps(self.reference_audits,indent=2)+'\n')
        if len(self.reference_audits)==1:
            torch.save(dict(q=q.cpu(),k=kv[0].cpu(),v=kv[1].cpu(),candidate=out.cpu(),native=expected.cpu(),
                plain_bf16=None if alternate is None else alternate.cpu(),query_lengths=self.qlens,
                prefix_lengths=self.prefix,context=self.lengths,coordinates=coordinates),
                self.diagnostic_root/'first_reference_adjudication.pt')
        print('FP64_ADJUDICATION',mode,layer,len(coordinates),
              sum(v['candidate_fp64_failures'] for v in rows),flush=True)
        return report

    def enable_decode(self):
        if self.engine is not None:return
        self.engine=PagedDecode(self.raw,self.peer,self.tier,self.slot,self.batch,self.cfg['decode_splits'])
        self.engine.requests.copy_(torch.tensor(self.requests,device=self.raw.device,dtype=torch.int32))
        self.engine.seq.copy_(torch.tensor(self.lengths,device=self.raw.device,dtype=torch.int32))
        self.decode_queries=torch.empty((36,self.batch,32,128),device=self.raw.device,dtype=torch.bfloat16)
        self.decode_outputs=torch.empty_like(self.decode_queries)
        graph=self.raw.decode_cuda_graph_runner;graph.capture_bs=[self.batch]
        graph.capture();torch.cuda.synchronize();self.decode_graph_captures+=1

    def check_decode(self):
        for i in range(36):self.check_layer(self.decode_queries[i],self.decode_outputs[i],i,'decode')

    def audit(self):
        local=[];remote=[];per_request=[];table=self.raw.req_to_token_pool.req_to_token
        for rid,length in self.live.items():
            tiers=self.tier[rid,:length];slots=self.slot[rid,:length]
            assert ((tiers==0)|(tiers==1)).all().item()
            far=tiers==1;near=~far
            assert torch.equal(slots[near],table[rid,:length][near])
            assert (table[rid,:length][far]==0).all().item()
            local.append(slots[near]);remote.append(slots[far])
            per_request.append(dict(request=rid,context=length,local=int(near.sum()),remote=int(far.sum())))
        near=torch.cat(local);far=torch.cat(remote)
        assert torch.unique(near).numel()==near.numel()
        assert torch.unique(far).numel()==far.numel()
        assert far.numel()+len(self.peer_free)==self.peer.shape[2]
        assert near.numel()==self.cfg['kv_capacity_tokens']-self.raw.token_to_kv_pool_allocator.available_size()
        return dict(local_active_tokens=near.numel(),remote_active_tokens=far.numel(),
                    logical_tokens=sum(self.live.values()),per_request=per_request,
                    page_tables_exact=True,unique_local_slots=True,unique_peer_slots=True,
                    remote_local_shadow_present=False)
