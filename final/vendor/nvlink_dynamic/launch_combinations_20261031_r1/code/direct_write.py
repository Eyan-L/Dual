"""Local-first virtual slot allocation and direct BF16 peer KV stores."""
import torch
import triton
import triton.language as tl


@triton.jit
def _write_tiered_kv(K,V,LK,LV,RK,RV,Loc,C:tl.constexpr,KS:tl.constexpr,VS:tl.constexpr):
    token=tl.program_id(0);x=tl.arange(0,1024)
    loc=tl.load(Loc+token);remote=loc>C;slot=tl.where(remote,loc-C-1,loc)
    k=tl.load(K+token*KS+x);v=tl.load(V+token*VS+x)
    tl.store(LK+slot*1024+x,k,~remote & (loc>0))
    tl.store(LV+slot*1024+x,v,~remote & (loc>0))
    tl.store(RK+slot*1024+x,k,remote)
    tl.store(RV+slot*1024+x,v,remote)


def write(local,peer,loc,k,v,capacity):
    assert k.dtype==v.dtype==torch.bfloat16
    k=k.view(-1,1024);v=v.view(-1,1024)
    return _write_tiered_kv[(loc.numel(),)](k,v,*local,*peer,loc,capacity,k.stride(0),v.stride(0),num_warps=4)


class Allocator:
    def __init__(self,control,physical):self.control=control;self.physical=physical
    def __getattr__(self,name):return getattr(self.physical,name)
    def available_size(self):return self.physical.available_size()+len(self.control.peer_free)
    def evict_to_free_tokens(self,tree_cache,num_tokens):
        assert num_tokens<=self.available_size(),(num_tokens,self.available_size())
        return True
    def alloc(self,n):
        if n>self.available_size():return None
        near=min(n,self.physical.available_size());far=n-near
        self.control.last_allocation_near=near
        local=self.physical.alloc(near)
        if not far:return local
        control=self.control
        remote=torch.tensor([control.peer_free.popleft()+control.cfg['kv_capacity_tokens']+1 for _ in range(far)],device=control.raw.device,dtype=torch.int64)
        return torch.cat((local,remote))


def validate(control):
    rows=[];c=control.cfg['kv_capacity_tokens'];device=control.raw.device
    for destinations in ([1,2],[c+1,c+2],[3,c+3,4,c+4]):
        n=len(destinations);loc=torch.tensor(destinations,device=device,dtype=torch.int64)
        # V has the row stride of a slice from a fused QKV activation.
        for layer in range(36):
            activation=(torch.arange(n*6144,device=device).reshape(n,6144)%191/128+layer/64).to(torch.bfloat16)
            k=activation[:,:1024].contiguous();v=activation[:,4096:5120]
            local=control.raw.token_to_kv_pool.get_kv_buffer(layer)
            write(local,control.peer[layer],loc,k,v,c);torch.cuda.synchronize()
            for j,dst in enumerate(destinations):
                pool=control.peer[layer] if dst>c else local;slot=dst-c-1 if dst>c else dst
                for plane,expected in enumerate((k,v)):
                    assert torch.equal(pool[plane][slot].reshape(-1).view(torch.int16),expected[j].view(torch.int16)),(layer,j,plane)
        rows.append(dict(destinations=destinations,layers=36,bf16_bitwise_equal=True,strided_v=True))
    return rows
