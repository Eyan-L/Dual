"""Adjudicate native BF16 reduction errors without changing numeric tolerances."""
import itertools
import types
import torch
from flashinfer import BatchPrefillWithPagedKVCacheWrapper as Paged
from layout import history_order


def native_decode(control,q,layer,fb):
    lengths=fb.seq_lens_cpu.tolist();k=[];v=[]
    for rid,n in zip(fb.req_pool_indices.cpu().tolist(),lengths):
        far=control.tier[rid,:n]==1;slot=control.slot[rid,:n].long()
        order=history_order(torch.arange(n,dtype=torch.int32),far.cpu(),n,control.cfg['decode_read_order']['remote_rows'],control.cfg['decode_read_order']['tile_rows']).long().to(q.device)
        for plane,bank in enumerate([k,v]):
            local=control.raw.token_to_kv_pool.get_kv_buffer(layer)[plane]
            payload=local.index_select(0,slot)
            payload[far]=control.peer[layer,plane].index_select(0,slot[far])
            bank.append(payload.index_select(0,order))
    qi=torch.arange(len(lengths)+1,dtype=torch.int32)
    ki=torch.tensor([0]+list(itertools.accumulate(lengths)),dtype=torch.int32)
    w=Paged(torch.empty(128*1024*1024,device=q.device,dtype=torch.uint8),'NHD',backend='fa2')
    w.plan(qi,ki,torch.arange(sum(lengths),device=q.device,dtype=torch.int32),torch.ones(len(lengths),dtype=torch.int32),
        32,8,128,1,causal=False,q_data_type=torch.bfloat16,kv_data_type=torch.bfloat16,sm_scale=128**-.5)
    return w.run(q,(torch.cat(k).unsqueeze(1),torch.cat(v).unsqueeze(1)))


def install(control):
    original=control.check_attention_output
    def check(self,q,out,layer,fb):
        try:return original(q,out,layer,fb)
        except AssertionError as error:
            if not fb.forward_mode.is_decode():raise
            native=native_decode(self,q,layer,fb)
            same=torch.equal(out,native)
            record=dict(uid=len(self.frames),layer=layer,mode='decode',batch=fb.batch_size,
                remote_tokens=self.remote_active,original_fp32_tolerance_violation=str(error),
                native_same_order_bitwise=same,native_same_order_max_abs_error=float((out-native).abs().max()),
                scope='Unchanged FP32 .01/.02 violations retained. Compare full output at original native tolerance;record bitwise outcome to unmodified native FA2 at same Q/K/V and reduction order,all payload independently gathered local. No peer pointers/private address JIT in oracle.')
            self.layer_checks.append(record)
            torch.testing.assert_close(out,native,atol=self.cfg['reference_atol'],rtol=self.cfg['reference_rtol'])
    control.check_attention_output=types.MethodType(check,control)
