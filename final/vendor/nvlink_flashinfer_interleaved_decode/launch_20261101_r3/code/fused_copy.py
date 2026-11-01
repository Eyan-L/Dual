"""One paged gather/scatter launch covers all36 K/V layer pairs."""
import torch
import triton
import triton.language as tl


@triton.jit
def _copy_kv_pages(Pointers,Peer,Source,Target,N:tl.constexpr,R:tl.constexpr,BLOCK:tl.constexpr):
    lp=tl.program_id(1)
    x=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
    valid=x<N*1024;page=x//1024;element=x%1024
    src=tl.load(Source+page,valid,0);dst=tl.load(Target+page,valid,0)
    base=tl.load(Pointers+lp).to(tl.pointer_type(tl.bfloat16))
    value=tl.load(base+src*1024+element,valid,0)
    tl.store(Peer+lp*R*1024+dst*1024+element,value,valid)


def copy_pages(pointers,peer,source,target):
    n=source.numel();assert n==target.numel()
    if n:
        block=1024 if n<=8 else 4096
        return _copy_kv_pages[(triton.cdiv(n*1024,block),72)](pointers,peer,source,target,n,peer.shape[2],block,num_warps=4)


def validate(control):
    raw=control.raw;device=raw.device
    pointers=torch.tensor([v.data_ptr() for i in range(36) for v in raw.token_to_kv_pool.get_kv_buffer(i)],device=device,dtype=torch.int64)
    control.local_pointers=pointers;rows=[]
    for n in sorted({2,17,200,control.cfg.get('validate_max_copy',200)}):
        source=torch.randperm(control.cfg['kv_capacity_tokens'],device=device)[:n]+1
        target=torch.randperm(control.peer.shape[2],device=device)[:n]
        expected=[]
        for i in range(36):
            for plane,local in enumerate(raw.token_to_kv_pool.get_kv_buffer(i)):
                value=((torch.arange(n*1024,device=device).reshape(n,8,128)%127)/64+i/16+plane/8).to(torch.bfloat16)
                local.index_copy_(0,source,value);expected.append(value)
        copy_pages(pointers,control.peer,source,target);torch.cuda.synchronize()
        for lp,value in enumerate(expected):
            actual=control.peer[lp//2,lp%2].index_select(0,target)
            assert torch.equal(actual.view(torch.int16),value.view(torch.int16)),(n,lp)
        rows.append(dict(tokens=n,layers=36,planes=2,bitwise_equal=True))
    return rows
