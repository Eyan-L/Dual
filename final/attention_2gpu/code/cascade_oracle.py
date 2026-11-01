"""Independent unmodified FlashInfer cascade on gathered local KV,same order."""
import itertools
import torch
from flashinfer import BatchPrefillWithPagedKVCacheWrapper as Paged,BatchPrefillWithRaggedKVCacheWrapper as Ragged
from sglang.srt.layers.attention.flashinfer_backend import _safe_merge_state
from layout import history_order


def native_cascade(control,q,k,v,fb):
    assert control.backend.forward_metadata.use_ragged
    lengths=fb.seq_lens_cpu.tolist();prefixes=fb.extend_prefix_lens_cpu;qlens=fb.extend_seq_lens_cpu
    qi=torch.tensor([0]+list(itertools.accumulate(qlens)),dtype=torch.int32)
    pi=torch.tensor([0]+list(itertools.accumulate(prefixes)),dtype=torch.int32)
    hk=[];hv=[];ck=[];cv=[];offset=0
    for rid,n,p in zip(fb.req_pool_indices.cpu().tolist(),lengths,prefixes):
        far=(control.tier[rid,:n].cpu()==1)
        order=history_order(torch.arange(n,dtype=torch.int32),far,p,control.remote_rows,control.tile_rows)
        order=order[:p].long().to(q.device)
        hk.append(k[offset:offset+n,0].index_select(0,order));hv.append(v[offset:offset+n,0].index_select(0,order))
        ck.append(k[offset+p:offset+n,0]);cv.append(v[offset+p:offset+n,0]);offset+=n
    a=Ragged(torch.empty(128*1024*1024,device=q.device,dtype=torch.uint8),'NHD',backend='fa2')
    a.plan(qi,qi,32,8,128,causal=True,q_data_type=torch.bfloat16,kv_data_type=torch.bfloat16,sm_scale=128**-.5)
    x,lx=a.run(q,torch.cat(ck),torch.cat(cv),return_lse=True)
    b=Paged(torch.empty(128*1024*1024,device=q.device,dtype=torch.uint8),'NHD',backend='fa2')
    b.plan(qi,pi,torch.arange(sum(prefixes),device=q.device,dtype=torch.int32),torch.ones(len(lengths),dtype=torch.int32),
        32,8,128,1,causal=False,q_data_type=torch.bfloat16,kv_data_type=torch.bfloat16,sm_scale=128**-.5)
    y,ly=b.run(q,(torch.cat(hk).unsqueeze(1),torch.cat(hv).unsqueeze(1)),return_lse=True)
    return _safe_merge_state(x,lx,y,ly)[0]
