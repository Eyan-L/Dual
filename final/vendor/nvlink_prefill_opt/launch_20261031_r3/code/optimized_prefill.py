"""Split-KV causal paged prefill; FP32 softmax states, BF16 KV/residual PV.

Splits partition logical positions, not physical placement. Arbitrary local /
peer page maps and GQA use the same arithmetic. Empty splits have zero weight.
The one-split variant writes directly and does not launch a merge kernel.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _split_causal_prefill(Q, LK, LV, RK, RV, Tier, Slot, Req, Seq, Qstart, Qlen, Prefix,
             Partial, Maximum, Denominator, Out,
             STRIDE:tl.constexpr, HQ:tl.constexpr, HKV:tl.constexpr, D:tl.constexpr,
             SCALE:tl.constexpr, BM:tl.constexpr, BN:tl.constexpr,
             SPLITS:tl.constexpr, QB:tl.constexpr, PROB_RESIDUAL:tl.constexpr):
    block=tl.program_id(0);head=tl.program_id(1)
    batch=tl.program_id(2)//SPLITS;split=tl.program_id(2)%SPLITS
    rid=tl.load(Req+batch)
    qstart=tl.load(Qstart+batch);qlen=tl.load(Qlen+batch)
    prefix=tl.load(Prefix+batch);length=tl.load(Seq+batch)
    row=block*BM+tl.arange(0,BM);dim=tl.arange(0,D)
    keyrow=tl.arange(0,BN)
    q=tl.load(Q+((qstart+row[:,None])*HQ+head)*D+dim[None,:],row[:,None]<qlen,0)
    maximum=tl.full((BM,),float('-inf'),tl.float32)
    denominator=tl.zeros((BM,),tl.float32)
    numerator=tl.zeros((BM,D),tl.float32)
    end=tl.minimum(length,prefix+tl.minimum((block+1)*BM,qlen))
    tiles=tl.cdiv(end,BN)
    # Align split boundaries to KV tiles; avoid partial-page duplicate reads.
    first=tiles*split//SPLITS;last=tiles*(split+1)//SPLITS
    for tile in range(first,last):
        pos=tile*BN+keyrow;valid=pos<end
        tier=tl.load(Tier+rid*STRIDE+pos,valid,0)
        slot=tl.load(Slot+rid*STRIDE+pos,valid,0)
        kp=tl.where(tier[:,None]==1,RK,LK)
        vp=tl.where(tier[:,None]==1,RV,LV)
        offset=(slot[:,None]*HKV+head//(HQ//HKV))*D+dim[None,:]
        k=tl.load(kp+offset,valid[:,None],0,cache_modifier='.cg')
        v=tl.load(vp+offset,valid[:,None],0,cache_modifier='.cg')
        score=tl.dot(q,tl.trans(k),out_dtype=tl.float32)*(SCALE*1.4426950408889634)
        visible=valid[None,:] & (pos[None,:]<=prefix+row[:,None]) & (row[:,None]<qlen)
        score=tl.where(visible,score,float('-inf'))
        updated=tl.maximum(maximum,tl.max(score,1))
        safe=tl.where(updated==float('-inf'),0.,updated)
        factor=tl.where(denominator>0,tl.exp2(maximum-safe),0.)
        probability=tl.exp2(score-safe[:,None])
        high=probability.to(tl.bfloat16)
        value=tl.dot(high,v,out_dtype=tl.float32)
        if PROB_RESIDUAL:
            low=(probability-high.to(tl.float32)).to(tl.bfloat16)
            value+=tl.dot(low,v,out_dtype=tl.float32)
        numerator=numerator*factor[:,None]+value
        denominator=denominator*factor+tl.sum(probability,1)
        maximum=updated
    if SPLITS==1:
        output=numerator/tl.maximum(denominator[:,None],1.e-20)
        tl.store(Out+((qstart+row[:,None])*HQ+head)*D+dim[None,:],output,row[:,None]<qlen)
    else:
        # Dense per-CTA states include padded rows and empty splits explicitly.
        state=((((batch*HQ+head)*QB+block)*SPLITS+split)*BM+tl.arange(0,BM))
        tl.store(Maximum+state,maximum)
        tl.store(Denominator+state,denominator)
        tl.store(Partial+state[:,None]*D+dim[None,:],numerator)


@triton.jit
def _merge_causal_prefill(Partial,Maximum,Denominator,Qstart,Qlen,Out,
           HQ:tl.constexpr,D:tl.constexpr,BM:tl.constexpr,QB:tl.constexpr,
           SPLITS:tl.constexpr,SS:tl.constexpr):
    block=tl.program_id(0);head=tl.program_id(1);batch=tl.program_id(2)
    row=block*BM+tl.arange(0,BM)
    # One program per query tile; reduce splits for each row, then accumulate V.
    split=tl.arange(0,SS)
    state=((((batch*HQ+head)*QB+block)*SPLITS+split[:,None])*BM+tl.arange(0,BM)[None,:])
    denom=tl.load(Denominator+state,split[:,None]<SPLITS,0.)
    maxima=tl.load(Maximum+state,split[:,None]<SPLITS,float('-inf'))
    maximum=tl.max(tl.where(denom>0,maxima,float('-inf')),0)
    safe=tl.where(maximum==float('-inf'),0.,maximum)
    weights=tl.where(denom>0,tl.exp2(maxima-safe[None,:]),0.)
    total=tl.sum(denom*weights,0)
    # Iterate compile-time splits to avoid a SPLITS*BM*D live register tensor.
    dim=tl.arange(0,D);numerator=tl.zeros((BM,D),tl.float32)
    for s in tl.static_range(SPLITS):
        index=((((batch*HQ+head)*QB+block)*SPLITS+s)*BM+tl.arange(0,BM))
        den=tl.load(Denominator+index);m=tl.load(Maximum+index)
        weight=tl.where(den>0,tl.exp2(m-safe),0.)
        part=tl.load(Partial+index[:,None]*D+dim[None,:])
        numerator+=part*weight[:,None]
    start=tl.load(Qstart+batch);qlen=tl.load(Qlen+batch)
    output=numerator/tl.maximum(total[:,None],1.e-20)
    tl.store(Out+((start+row[:,None])*HQ+head)*D+dim[None,:],output,row[:,None]<qlen)


def prefill(q,local_k,local_v,peer_k,peer_v,tier,slot,requests,seq,qstart,qlen,prefix,
            max_query_length,scale,probability_residual=True,*,block_m=64,splits=4):
    assert q.is_contiguous() and q.dtype==torch.bfloat16
    assert local_k.is_contiguous() and local_v.is_contiguous() and peer_k.is_contiguous() and peer_v.is_contiguous()
    assert local_k.shape[1:]==peer_k.shape[1:] and q.shape[-1]==128
    assert q.dtype==local_k.dtype==local_v.dtype==peer_k.dtype==peer_v.dtype
    assert block_m in (32,64) and splits in (1,2,4,8) and max_query_length>0
    heads,dim=q.shape[1:];kvheads=local_k.shape[1]
    assert heads%kvheads==0
    qb=triton.cdiv(max_query_length,block_m);batch=requests.numel()
    out=torch.empty_like(q)
    if splits>1:
        shape=(batch,heads,qb,splits,block_m)
        partial=torch.empty((*shape,dim),device=q.device,dtype=torch.float32)
        maximum=torch.empty(shape,device=q.device,dtype=torch.float32)
        denominator=torch.empty_like(maximum)
    else:partial=maximum=denominator=out
    kernel=_split_causal_prefill[(qb,heads,batch*splits)](q,local_k,local_v,peer_k,peer_v,tier,slot,requests,seq,
        qstart,qlen,prefix,partial,maximum,denominator,out,tier.shape[1],heads,kvheads,dim,scale,
        block_m,64,splits,qb,probability_residual,num_warps=4,num_stages=2)
    if splits>1:
        _merge_causal_prefill[(qb,heads,batch)](partial,maximum,denominator,qstart,qlen,out,
            heads,dim,block_m,qb,splits,triton.next_power_of_2(splits),num_warps=4)
    return out,kernel
