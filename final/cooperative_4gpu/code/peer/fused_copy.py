"""One kernel covers all72planes and three distinct peer allocations."""
import triton
import triton.language as tl


@triton.jit
def _copy(Local,Remote,Source,Target,N:tl.constexpr,REVERSE:tl.constexpr,BLOCK:tl.constexpr):
    plane=tl.program_id(1);x=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    valid=x<N*1024;page=x//1024;element=x%1024
    src=tl.load(Source+page,valid,0);dst=tl.load(Target+page,valid,0)
    global_remote=src if REVERSE else dst
    local_slot=dst if REVERSE else src
    bank=global_remote%3;remote_slot=global_remote//3
    lp=tl.load(Local+plane).to(tl.pointer_type(tl.bfloat16))
    rp=tl.load(Remote+bank*72+plane).to(tl.pointer_type(tl.bfloat16))
    if REVERSE:
        value=tl.load(rp+remote_slot*1024+element,valid,0)
        tl.store(lp+local_slot*1024+element,value,valid)
    else:
        value=tl.load(lp+local_slot*1024+element,valid,0)
        tl.store(rp+remote_slot*1024+element,value,valid)


def copy_pages(pointers,peer,source,target):
    n=source.numel();assert n==target.numel()
    if n:
        block=1024 if n<=8 else 4096
        _copy[(triton.cdiv(n*1024,block),72)](pointers,peer.pointers,source,target,n,False,block,num_warps=4)


def restore(pointers,peer,source,target):
    n=source.numel();assert n==target.numel()
    if n:_copy[(triton.cdiv(n*1024,1024),72)](pointers,peer.pointers,source,target,n,True,1024,num_warps=4)
