"""Decode with arbitrary per-token local/peer ownership and changing page maps."""
import torch
import triton
import triton.language as tl


@triton.jit
def _decode(Q,LK,LV,RK,RV,Tier,Slot,Req,Seq,Acc,Max,Den,
            STRIDE:tl.constexpr,S:tl.constexpr,N:tl.constexpr):
    head,split,batch=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    rid=tl.load(Req+batch);length=tl.load(Seq+batch)
    chunk=tl.cdiv(length,S);start=split*chunk;end=tl.minimum(start+chunk,length)
    row=tl.arange(0,16);dim=tl.arange(0,128);col=tl.arange(0,N)
    q=tl.load(Q+(batch*32+head*4+row[:,None])*128+dim[None,:],row[:,None]<4,0)
    maximum=tl.full((16,),float('-inf'),tl.float32);denominator=tl.zeros((16,),tl.float32)
    numerator=tl.zeros((16,128),tl.float32)
    for tile in range(0,tl.cdiv(chunk,N)):
        pos=start+tile*N+col;valid=pos<end
        tier=tl.load(Tier+rid*STRIDE+pos,valid,0);slot=tl.load(Slot+rid*STRIDE+pos,valid,0)
        kp=tl.where(tier[:,None]==1,RK,LK);vp=tl.where(tier[:,None]==1,RV,LV)
        offset=slot[:,None]*1024+head*128+dim[None,:]
        k=tl.load(kp+offset,valid[:,None],0,cache_modifier='.cg')
        v=tl.load(vp+offset,valid[:,None],0,cache_modifier='.cg')
        score=tl.dot(q,tl.trans(k),out_dtype=tl.float32)*(0.08838834764831845*1.4426950408889634)
        score=tl.where(valid[None,:],score,float('-inf'))
        updated=tl.maximum(maximum,tl.max(score,1));safe=tl.where(updated==float('-inf'),0.,updated)
        factor=tl.where(denominator>0,tl.exp2(maximum-safe),0.)
        probability=tl.exp2(score-safe[:,None])
        high=probability.to(tl.bfloat16)
        low=(probability-high.to(tl.float32)).to(tl.bfloat16)
        value=tl.dot(high,v,out_dtype=tl.float32)+tl.dot(low,v,out_dtype=tl.float32)
        numerator=numerator*factor[:,None]+value
        denominator=denominator*factor+tl.sum(probability,1);maximum=updated
    offset=((batch*8+head)*S+split)*4+row
    tl.store(Acc+offset[:,None]*128+dim[None,:],numerator,row[:,None]<4)
    tl.store(Max+offset,maximum,row<4);tl.store(Den+offset,denominator,row<4)


@triton.jit
def _merge(Acc,Max,Den,Out,S:tl.constexpr,PAD:tl.constexpr):
    head,batch=tl.program_id(0),tl.program_id(1);split=tl.arange(0,PAD);dim=tl.arange(0,128)
    row=((batch*8+head//4)*S+split)*4+head%4
    maximum=tl.load(Max+row,split<S,float('-inf'));den=tl.load(Den+row,split<S,0.)
    totalmax=tl.max(maximum,0);safe=tl.where(totalmax==float('-inf'),0.,totalmax)
    weight=tl.where(den>0,tl.exp2(maximum-safe),0.)
    acc=tl.load(Acc+row[:,None]*128+dim[None,:],split[:,None]<S,0.)
    output=tl.sum(acc*weight[:,None],0)/tl.maximum(tl.sum(den*weight,0),1.e-20)
    tl.store(Out+(batch*32+head)*128+dim,output)


class PagedDecode:
    def __init__(self,raw,peer,tier,slot,batch,splits=4):
        self.raw,self.peer,self.tier,self.slot=raw,peer,tier,slot;self.batch,self.splits=batch,splits
        self.requests=torch.zeros(batch,device=raw.device,dtype=torch.int32)
        self.seq=torch.ones_like(self.requests)
        shape=(36,batch,8,splits,4)
        self.acc=torch.empty((*shape,128),device=raw.device,dtype=torch.float32)
        self.maximum=torch.empty(shape,device=raw.device,dtype=torch.float32);self.den=torch.empty_like(self.maximum)
        self.output=torch.empty((36,batch,32,128),device=raw.device,dtype=torch.bfloat16)

    def prepare(self,fb):
        assert fb.batch_size==self.batch
        self.requests.copy_(fb.req_pool_indices);self.seq.copy_(fb.seq_lens)

    def run(self,q,layer):
        lk,lv=self.raw.token_to_kv_pool.get_kv_buffer(layer)
        _decode[(8,self.splits,self.batch)](q,lk,lv,self.peer[layer,0],self.peer[layer,1],
            self.tier,self.slot,self.requests,self.seq,self.acc[layer],self.maximum[layer],self.den[layer],
            self.tier.shape[1],self.splits,128,num_warps=4,num_stages=2)
        _merge[(32,self.batch)](self.acc[layer],self.maximum[layer],self.den[layer],self.output[layer],
            self.splits,triton.next_power_of_2(self.splits),num_warps=4)
        return self.output[layer]
