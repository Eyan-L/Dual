"""Keep native prefill routing/ragged/merge/writes;extend paged KV addresses."""
import functools
import types
import torch
import triton
from flashinfer import BatchPrefillWithPagedKVCacheWrapper as Wrapper
from flashinfer_peer import activate
from backend import Backend as DecodeBackend
from packing import prepare
from layout import history_order


class Backend(DecodeBackend):
    def __init__(self,raw,peer,cfg,old,case):
        native_extend=raw.attn_backend.forward_extend
        super().__init__(raw,peer,cfg,old,case)
        self.prefill_active=False;self.check_prefill=False;self.prefill_checks=[];self.prefill_calls=[]
        self.prefill_source=cfg['source'];self.remote_rows=cfg['remote_rows'];self.tile_rows=cfg['tile_rows'];self.profile_label=''
        self.stage_depth=cfg.get('stage_depth',0);self.stager=None
        self.full_indices=torch.empty(cfg['kv_capacity_tokens']+cfg['remote_capacity_tokens'],device=raw.device,dtype=torch.int32)
        self.history_indices=torch.empty_like(self.full_indices)
        self.capture_attention=False;self.captured={}
        control=self;original_forward=raw.forward;original_run=Wrapper.run

        def run(wrapper,q,kv,*args,**kwargs):
            if not getattr(wrapper,'slack_prefill_model_wrapper',False) or not control.prefill_active:
                return original_run(wrapper,q,kv,*args,**kwargs)
            assert not args and wrapper._custom_mask_buf is None and wrapper._window_left==-1
            assert wrapper._jit_module is not None
            ragged=control.backend.forward_metadata.use_ragged
            indices=control.history_indices if ragged else control.full_indices
            i=control.current_layer
            remote=control.stager.wait(i) if control.stage_depth else [bank[i] for bank in control.read_peer.banks]
            output=original_run(wrapper,q,kv,*[tensor for bank in remote for tensor in bank],indices,
                control.cfg['kv_capacity_tokens'],control.current_scale,**kwargs)
            if control.stage_depth:control.stager.next(i)
            return output
        if self.enabled:
            Wrapper.run=run
            # FlashInfer0.6.18 captures the function in partialmethod;update
            # this path as well as run,otherwise cached-prefix LSE bypasses it.
            Wrapper.run_return_lse=functools.partialmethod(run,return_lse=True)
        def forward(fb,*args,**kwargs):
            if fb.forward_mode.is_extend():
                current=sum(control.remote_counts.get(int(r.kv.req_pool_idx),0) for r in control.req_objects)
                control.prefill_active=control.enabled and current>0
                for wrapper in control.backend.prefill_wrappers_paged:
                    wrapper.slack_prefill_model_wrapper=True
                    if control.prefill_active:activate(wrapper)
                    else:wrapper._jit_module=None
            return original_forward(fb,*args,**kwargs)
        def extend(backend,q,k,v,layer,fb,save_kv_cache=True):
            assert not layer.is_cross_attention and layer.logit_cap in (None,0,0.)
            control.current_layer=layer.layer_id;control.current_scale=layer.scaling
            out=native_extend(q,k,v,layer,fb,save_kv_cache=save_kv_cache)
            if control.capture_attention:
                control.captured[layer.layer_id]=(q.detach().view(-1,32,128).cpu().clone(),out.detach().view(-1,32,128).cpu().clone())
            if control.check_prefill:control.check_prefill_layer(q.view(-1,32,128),out.view(-1,32,128),layer.layer_id,fb)
            return out
        raw.forward=forward;self.backend.forward_extend=types.MethodType(extend,self.backend)

    def begin(self,fb):
        super().begin(fb)
        if fb.forward_mode.is_extend() and self.prefill_active:
            prepare[(fb.batch_size,)](self.tier,self.slot,fb.req_pool_indices,fb.seq_lens,fb.extend_prefix_lens,
                self.full_indices,self.history_indices,self.tier.shape[1],self.cfg['kv_capacity_tokens'],
                fb.batch_size,triton.next_power_of_2(fb.batch_size),triton.next_power_of_2(max(fb.seq_lens_cpu.tolist())),
                self.remote_rows,self.tile_rows,num_warps=8)
            if self.stage_depth:
                from staging import LayerStager
                if self.stager is None:self.stager=LayerStager(self)
                self.stager.start()

    def audit(self):
        value=super().audit()
        value['remote_local_shadow_present']=self.prefill_source=='local_shadow'
        if self.check_prefill and self.prefill_active and self.live:
            fb=self.last_fb;full=[];history=[]
            if fb.forward_mode.is_extend():
                for rid,n,p in zip(fb.req_pool_indices.cpu().tolist(),fb.seq_lens_cpu.tolist(),fb.extend_prefix_lens_cpu):
                    tier=self.tier[rid,:n].cpu();encoded=(self.slot[rid,:n].cpu()+torch.where(tier==1,self.cfg['kv_capacity_tokens']+1,0)).int()
                    order=history_order(encoded,tier==1,p,self.remote_rows,self.tile_rows);full.append(order);history.append(order[:p])
                full=torch.cat(full);history=torch.cat(history)
                assert torch.equal(full,self.full_indices[:len(full)].cpu())
                assert torch.equal(history,self.history_indices[:len(history)].cpu())
                value['history_permutation_exact']=True;value['new_query_suffix_unchanged']=True
        return value

    def check_prefill_layer(self,q,out,layer,fb):
        # Native full-causal same-QKV oracle is independent of SGLang's
        # cached-prefix/ragged merge and our tier-aware address resolver.
        lengths=fb.seq_lens_cpu.tolist();qlens=fb.extend_seq_lens_cpu;prefix=fb.extend_prefix_lens_cpu
        k=[];v=[];offset=0
        for rid,n in zip(fb.req_pool_indices.cpu().tolist(),lengths):
            tier=self.tier[rid,:n] if self.enabled else torch.zeros(n,device=q.device,dtype=torch.int32)
            slots=self.slot[rid,:n].long() if self.enabled else self.raw.req_to_token_pool.req_to_token[rid,:n].long()
            far=tier==1
            for plane,target in enumerate([k,v]):
                near=self.raw.token_to_kv_pool.get_kv_buffer(layer)[plane]
                payload=near.index_select(0,slots)
                payload[far]=self.peer[layer,plane].index_select(0,slots[far])
                target.append(payload)
        k=torch.cat(k).unsqueeze(1);v=torch.cat(v).unsqueeze(1)
        qi=torch.tensor([0]+list(__import__('itertools').accumulate(qlens)),dtype=torch.int32)
        ki=torch.tensor([0]+list(__import__('itertools').accumulate(lengths)),dtype=torch.int32)
        oracle=Wrapper(torch.empty(128*1024*1024,device=q.device,dtype=torch.uint8),'NHD',backend='fa2')
        oracle.plan(qi,ki,torch.arange(sum(lengths),device=q.device,dtype=torch.int32),torch.ones(len(lengths),dtype=torch.int32),
            32,8,128,1,causal=True,q_data_type=torch.bfloat16,kv_data_type=torch.bfloat16,sm_scale=128**-.5)
        expected=oracle.run(q,(k,v));bad=~torch.isclose(out,expected,atol=.01,rtol=.02)
        disputed=bad.nonzero().cpu().tolist();adjudications=[];cascade=None;cascade_check=None
        for token,head in sorted({(t,h) for t,h,d in disputed}):
            request=next(i for i in range(len(qlens)) if int(qi[i])<=token<int(qi[i+1]))
            visible=prefix[request]+token-int(qi[request])+1;lo=int(ki[request]);kh=head//4
            key=k[lo:lo+visible,0,kh].double();value=v[lo:lo+visible,0,kh].double()
            scores=(key*q[token,head].double()).sum(-1)*(128**-.5);truth=(scores.softmax(0)[:,None]*value).sum(0)
            candidate=out[token,head].double();native=expected[token,head].double()
            failures=int((~torch.isclose(candidate,truth,atol=.01,rtol=.02)).sum())
            native_failures=int((~torch.isclose(native,truth,atol=.01,rtol=.02)).sum())
            detail=dict(token=token,head=head,candidate_fp64_failures=failures,native_fp64_failures=native_failures)
            if failures and self.prefill_active:
                if cascade is None:
                    from cascade_oracle import native_cascade
                    cascade=native_cascade(self,q,k,v,fb)
                    cascade_check=dict(bitwise_equal=torch.equal(out,cascade),native_tolerance_match=bool(torch.isclose(out,cascade,atol=.01,rtol=.02).all()),max_abs_error=float((out-cascade).abs().max()),
                        scope='Unmodified native FA2 current-input ragged + same-order gathered-local historical paged + native safe merge. No peer pointer,JIT address resolver,or staged source in oracle.')
                detail['native_cascade_equivalence']=cascade_check
                bad_channels=(~torch.isclose(candidate,truth,atol=.01,rtol=.02)).nonzero().flatten().cpu().tolist()
                detail['fp64_disputed_values']=[dict(channel=d,candidate=float(candidate[d]),truth=float(truth[d]),
                    native_full_causal=float(native[d]),native_same_order_cascade=float(cascade[token,head,d])) for d in bad_channels]
                assert cascade_check['native_tolerance_match'],detail
            adjudications.append(detail)
        self.prefill_checks.append(dict(layer=layer,query_lengths=qlens,prefix_lengths=prefix,
            native_mismatches=len(disputed),max_abs_error=float((out-expected).abs().max()),
            fp64_adjudications=adjudications,custom_path=self.prefill_active,native_cascade_check=cascade_check,
            tolerance_note='Original FP64 .01/.02 retained and violations reported. If a violation occurs,compare independent unmodified native same-order cascade at original native tolerance;retain bitwise outcome;never mark it as FP64 tolerance passing.'))
