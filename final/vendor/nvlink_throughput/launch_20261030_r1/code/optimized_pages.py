"""Mixed-page tuning and scheme-1 asymmetric CTA work in a single dispatch."""
import torch
import triton
import triton.language as tl

from original_pages import PageAttention, _unified_pages, _reduce_pages


@triton.jit
def _tier_ctas(Q, LK, LV, RK, RV, Tier, Slot, Seq, Acc, Max, Den,
               STRIDE:tl.constexpr, R:tl.constexpr, LS:tl.constexpr, RS:tl.constexpr,
               N:tl.constexpr, STAGES:tl.constexpr):
    h, split, b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    m, d, row = tl.arange(0,16), tl.arange(0,128), tl.arange(0,N)
    remote = split >= LS
    length = tl.where(remote, R, tl.load(Seq+b)-R)
    count = tl.where(remote, RS, LS)
    part = tl.where(remote, split-LS, split)
    chunk = tl.cdiv(length,count)
    lo = part*chunk
    end = tl.minimum(lo+chunk,length)
    q=tl.load(Q+(b*32+h*4+m[:,None])*128+d[None,:],m[:,None]<4,0)
    maximum=tl.full((16,),float('-inf'),tl.float32)
    denominator=tl.zeros((16,),tl.float32)
    numerator=tl.zeros((16,128),tl.float32)
    for tile in tl.range(0,tl.cdiv(chunk,N),num_stages=STAGES):
        pos=lo+tile*N+row
        valid=pos<end
        logical=tl.where(remote,pos,R+pos)
        tier=tl.load(Tier+b*STRIDE+logical,valid,0)
        slot=tl.load(Slot+b*STRIDE+logical,valid,0)
        kp=tl.where(tier[:,None]==1,RK,LK)
        vp=tl.where(tier[:,None]==1,RV,LV)
        offset=slot[:,None]*1024+h*128+d[None,:]
        k=tl.load(kp+offset,valid[:,None],0,cache_modifier='.cg')
        v=tl.load(vp+offset,valid[:,None],0,cache_modifier='.cg')
        score=tl.dot(q,tl.trans(k),out_dtype=tl.float32)*(0.08838834764831845*1.4426950408889634)
        score=tl.where(valid[None,:],score,float('-inf'))
        updated=tl.maximum(maximum,tl.max(score,axis=1))
        factor=tl.exp2(maximum-updated)
        probability=tl.exp2(score-updated[:,None])
        numerator=numerator*factor[:,None]+tl.dot(probability.to(tl.bfloat16),v,out_dtype=tl.float32)
        denominator=denominator*factor+tl.sum(probability,axis=1)
        maximum=updated
    output_row=((b*8+h)*(LS+RS)+split)*4+m
    tl.store(Acc+output_row[:,None]*128+d[None,:],numerator,m[:,None]<4)
    tl.store(Max+output_row,maximum,m<4)
    tl.store(Den+output_row,denominator,m<4)


class OptimizedAttention(PageAttention):
    def __init__(self,control,cfg,tuning):
        tuning=dict(tuning)
        self.mode=tuning.get('mode','mixed')
        if self.mode=='tier_ctas':tuning['splits']=tuning['local_splits']+tuning['remote_splits']
        super().__init__(control,cfg,tuning)
        self.remote_rows=tuning.get('remote_rows',self.tile_n//16)
        assert 0 < self.remote_rows < self.tile_n

    @property
    def label(self):
        t=self.tuning
        stem=(f"T_L{t['local_splits']}_R{t['remote_splits']}" if self.mode=='tier_ctas' else f"M_S{self.splits}_RN{self.remote_rows}")
        return stem+f"_N{t['tile_n']}_P{t['num_stages']}_W{t['num_warps']}"

    def run(self,q,layer,shadow=False):
        c=self.control
        lk,lv=c.raw.token_to_kv_pool.get_kv_buffer(layer)
        remote=c.shadow if shadow else c.peer
        args=(q,lk,lv,remote[layer,0],remote[layer,1],self.tier,self.slot,self.seq,
              self.acc[layer],self.maximum[layer],self.denominator[layer],self.maximum_context,self.remote)
        launch=dict(num_warps=self.tuning['num_warps'],num_stages=self.tuning['num_stages'])
        if self.mode=='tier_ctas':
            self.kernel=_tier_ctas[(8,self.splits,self.batch)](*args,self.tuning['local_splits'],
                self.tuning['remote_splits'],self.tile_n,self.tuning['num_stages'],**launch)
        else:
            self.kernel=_unified_pages[(8,self.splits,self.batch)](*args,self.splits,self.tile_n,
                self.remote_rows,self.tuning['num_stages'],**launch)
        self.reduce_kernel=_reduce_pages[(32,self.batch)](self.acc[layer],self.maximum[layer],
            self.denominator[layer],self.output[layer],self.splits,triton.next_power_of_2(self.splits),num_warps=4)
        return self.output[layer]

    def metadata(self):
        meta=super().metadata()
        meta['mode']=self.mode
        if self.mode=='tier_ctas':
            meta.update(local_attention_blocks=8*self.batch*self.tuning['local_splits'],
                        remote_attention_blocks=8*self.batch*self.tuning['remote_splits'],
                        local_rows_per_tile=self.tile_n,remote_rows_per_tile=self.tile_n)
        return meta
