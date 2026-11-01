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


from fused_copy import restore
