"""Permute immutable fully-visible history only;causal query suffix is fixed."""
import torch


def history_order(encoded,far,prefix,remote_rows,tile_rows=128):
    assert encoded.ndim==far.ndim==1 and 0<=prefix<=len(encoded)
    assert not far[prefix:].any(),'Current input KV must stay local and ordered'
    if remote_rows==0:return encoded.clone()
    assert 0<remote_rows<tile_rows
    remote=encoded[:prefix][far[:prefix]].tolist();local=encoded[:prefix][~far[:prefix]].tolist()
    result=[];ri=li=0
    while ri<len(remote) or li<len(local):
        rr=remote[ri:ri+remote_rows];ll=local[li:li+tile_rows-remote_rows]
        ri+=len(rr);li+=len(ll);missing=tile_rows-len(rr)-len(ll)
        extra=local[li:li+missing];ll+=extra;li+=len(extra);missing-=len(extra)
        extra=remote[ri:ri+missing];rr+=extra;ri+=len(extra)
        result.extend(rr+ll)
    result+=encoded[prefix:].tolist()
    out=torch.tensor(result,dtype=encoded.dtype)
    assert sorted(out[:prefix].tolist())==sorted(encoded[:prefix].tolist())
    assert torch.equal(out[prefix:],encoded[prefix:])
    return out
