"""Reverse of the frozen all-layer fused copy, for active-ratio repair on exit."""
import triton
import triton.language as tl


@triton.jit
def _write_prefill_layout(Tier,Slot,Req,Start,Length,Prefix,Locations,STRIDE:tl.constexpr,BLOCK:tl.constexpr):
    request=tl.program_id(0);offset=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    rid=tl.load(Req+request);start=tl.load(Start+request)
    length=tl.load(Length+request);prefix=tl.load(Prefix+request);valid=offset<length
    location=tl.load(Locations+start+offset,valid,0)
    index=rid*STRIDE+prefix+offset
    tl.store(Slot+index,location,valid);tl.store(Tier+index,0,valid)


def write_prefill_layout(tier,slot,requests,start,length,prefix,locations,max_query):
    _write_prefill_layout[(requests.numel(),triton.cdiv(max_query,256))](
        tier,slot,requests,start,length,prefix,locations,tier.shape[1],256,num_warps=4)


@triton.jit
def _restore_pages(Pointers,Peer,Source,Target,N:tl.constexpr,R:tl.constexpr,BLOCK:tl.constexpr):
    plane=tl.program_id(1);x=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    valid=x<N*1024;page=x//1024;element=x%1024
    src=tl.load(Source+page,valid,0);dst=tl.load(Target+page,valid,0)
    pointer=tl.load(Pointers+plane).to(tl.pointer_type(tl.bfloat16))
    value=tl.load(Peer+plane*R*1024+src*1024+element,valid,0)
    tl.store(pointer+dst*1024+element,value,valid)


def restore(pointers,peer,source,target):
    n=source.numel();assert n==target.numel()
    if n:_restore_pages[(triton.cdiv(n*1024,1024),72)](pointers,peer,source,target,n,peer.shape[2],1024,num_warps=4)
