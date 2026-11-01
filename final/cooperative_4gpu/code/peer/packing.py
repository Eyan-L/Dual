"""Stable tier compaction and prefix-only interleave;arbitrary historical pages."""
import triton
import triton.language as tl


@triton.jit
def prepare(Tier,Slot,Req,Seq,Prefix,Full,History,
            STRIDE:tl.constexpr,CAP:tl.constexpr,B:tl.constexpr,BPAD:tl.constexpr,
            PAD:tl.constexpr,RN:tl.constexpr,TN:tl.constexpr):
    b=tl.program_id(0);rid=tl.load(Req+b);n=tl.load(Seq+b);p=tl.load(Prefix+b)
    lane=tl.arange(0,BPAD);seq=tl.load(Seq+lane,lane<B,0);prefix=tl.load(Prefix+lane,lane<B,0)
    full_start=tl.sum(tl.where(lane<b,seq,0),0);history_start=tl.sum(tl.where(lane<b,prefix,0),0)
    pos=tl.arange(0,PAD);valid=pos<n;old=pos<p
    tier=tl.load(Tier+rid*STRIDE+pos,valid,0);slot=tl.load(Slot+rid*STRIDE+pos,valid,0)
    encoded=slot+tl.where(tier==1,CAP+1,0)
    if RN==0:
        destination=pos
    else:
        far=old&(tier==1);near=old&(tier==0)
        r=tl.sum(far.to(tl.int32),0);l=p-r
        rr=tl.cumsum(far.to(tl.int32),0)-1;ll=tl.cumsum(near.to(tl.int32),0)-1
        LN=TN-RN;groups=tl.minimum(r//RN,l//LN);base=groups*TN
        rleft=r-groups*RN;lleft=l-groups*LN
        tail_r=tl.minimum(rleft,RN);tail_l=tl.minimum(lleft,LN)
        extra_l=tl.minimum(TN-tail_r-tail_l,lleft-tail_l);tail_l+=extra_l
        tail_r+=tl.minimum(TN-tail_r-tail_l,rleft-tail_r)
        tail_size=tail_r+tail_l
        remote_position=tl.where(rr<groups*RN,(rr//RN)*TN+rr%RN,
            tl.where(rr<groups*RN+tail_r,base+rr-groups*RN,
                base+tail_size+rr-groups*RN-tail_r))
        local_position=tl.where(ll<groups*LN,(ll//LN)*TN+RN+ll%LN,
            tl.where(ll<groups*LN+tail_l,base+tail_r+ll-groups*LN,
                base+tail_size+ll-groups*LN-tail_l))
        destination=tl.where(old,tl.where(far,remote_position,local_position),pos)
    tl.store(Full+full_start+destination,encoded,valid)
    tl.store(History+history_start+destination,encoded,old)
