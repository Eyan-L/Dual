"""Causal chunk prefill over page-size-1 local and CUDA-IPC peer KV.

Q is packed by request. Tier/Slot use logical request IDs and token positions;
remote pages need not be contiguous or form a prefix. GQA maps h -> h // group.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _paged_causal_prefill(Q, LK, LV, RK, RV, Tier, Slot, Req, Seq,
                          Qstart, Qlen, Prefix, Out,
                          STRIDE: tl.constexpr, HQ: tl.constexpr,
                          HKV: tl.constexpr, D: tl.constexpr,
                          SCALE: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                          PROB_RESIDUAL: tl.constexpr):
    block, head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rid = tl.load(Req + batch)
    qstart, qlen = tl.load(Qstart + batch), tl.load(Qlen + batch)
    prefix, length = tl.load(Prefix + batch), tl.load(Seq + batch)
    row = block * BM + tl.arange(0, BM)
    dim = tl.arange(0, D)
    keyrow = tl.arange(0, BN)
    q = tl.load(Q + ((qstart + row[:, None]) * HQ + head) * D + dim[None, :],
                row[:, None] < qlen, 0)
    maximum = tl.full((BM,), float('-inf'), tl.float32)
    denominator = tl.zeros((BM,), tl.float32)
    numerator = tl.zeros((BM, D), tl.float32)
    end = tl.minimum(length, prefix + tl.minimum((block + 1) * BM, qlen))
    for start in range(0, tl.cdiv(end, BN)):
        pos = start * BN + keyrow
        valid = pos < end
        tier = tl.load(Tier + rid * STRIDE + pos, valid, 0)
        slot = tl.load(Slot + rid * STRIDE + pos, valid, 0)
        kp = tl.where(tier[:, None] == 1, RK, LK)
        vp = tl.where(tier[:, None] == 1, RV, LV)
        offset = (slot[:, None] * HKV + head // (HQ // HKV)) * D + dim[None, :]
        k = tl.load(kp + offset, valid[:, None], 0, cache_modifier='.cg')
        v = tl.load(vp + offset, valid[:, None], 0, cache_modifier='.cg')
        score = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * (SCALE * 1.4426950408889634)
        visible = valid[None, :] & (pos[None, :] <= prefix + row[:, None]) & (row[:, None] < qlen)
        score = tl.where(visible, score, float('-inf'))
        updated = tl.maximum(maximum, tl.max(score, 1))
        # Padded query rows have no visible keys. Keep their recurrence finite.
        safe = tl.where(updated == float('-inf'), 0., updated)
        factor = tl.where(denominator > 0, tl.exp2(maximum - safe), 0.)
        probability = tl.exp2(score - safe[:, None])
        high = probability.to(tl.bfloat16)
        value = tl.dot(high, v, out_dtype=tl.float32)
        if PROB_RESIDUAL:
            low = (probability - high.to(tl.float32)).to(tl.bfloat16)
            value += tl.dot(low, v, out_dtype=tl.float32)
        numerator = numerator * factor[:, None] + value
        denominator = denominator * factor + tl.sum(probability, 1)
        maximum = updated
    output = numerator / tl.maximum(denominator[:, None], 1.e-20)
    tl.store(Out + ((qstart + row[:, None]) * HQ + head) * D + dim[None, :],
             output, row[:, None] < qlen)


def prefill(q, local_k, local_v, peer_k, peer_v, tier, slot,
            requests, seq, qstart, qlen, prefix, max_query_length, scale, probability_residual=True):
    assert q.ndim == 3 and q.is_contiguous()
    assert local_k.is_contiguous() and peer_k.is_contiguous()
    assert local_k.shape[1:] == peer_k.shape[1:]
    heads, dim = q.shape[1:]; kvheads = local_k.shape[1]
    assert dim == 128 and heads % kvheads == 0
    assert q.dtype == local_k.dtype == peer_k.dtype == torch.bfloat16
    assert max_query_length > 0
    out = torch.empty_like(q)
    kernel = _paged_causal_prefill[(triton.cdiv(max_query_length, 32), heads, requests.numel())](
        q, local_k, local_v, peer_k, peer_v, tier, slot, requests, seq,
        qstart, qlen, prefix, out, tier.shape[1], heads, kvheads, dim, scale, 32, 64, probability_residual,
        num_warps=4, num_stages=2)
    return out, kernel
