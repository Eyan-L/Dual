from array import array
import math

def prepare(bench, wrapped, tokenizer, batch_size, context, chunk, max_outputs):
    import torch
    raw = wrapped.torch_runner
    wrapped.clear()
    material = tokenizer.encode('A controlled serving experiment measures attention, KV growth and request latency. ', add_special_tokens=False)
    ids = (material * math.ceil(context/len(material)))[:context]
    reqs = bench.prepare_synthetic_inputs_for_latency_test(batch_size, context, [ids[:] for _ in range(batch_size)])
    merged, next_ids = None, []
    for req in reqs:
        req.sampling_params.ignore_eos = True
        req.sampling_params.max_new_tokens = max_outputs
        full = array('q', req.full_untruncated_fill_ids)
        for lo in range(0, len(full), chunk):
            hi = min(lo+chunk, len(full))
            if lo:
                req.prefix_indices = raw.req_to_token_pool.req_to_token[req.kv.req_pool_idx, :lo].to(req.prefix_indices.dtype)
            req.origin_input_ids = array('q', full[:hi])
            req.full_untruncated_fill_ids = array('q', full[:hi])
            req.set_extend_range(lo, hi)
            token, _, prepared = bench.extend([req], raw)
        next_ids.append(token)
        if merged is None:
            merged = prepared
        else:
            merged.merge_batch(prepared)
    token = torch.cat(next_ids).to(torch.int64)
    torch.cuda.synchronize()
    assert merged.seq_lens_cpu.tolist() == [context]*batch_size
    assert raw.max_total_num_tokens-raw.token_to_kv_pool_allocator.available_size() == batch_size*context
    # SamplingParams normalizes temperature=0 into temperature=1, top_k=1.
    # Check the effective native greedy path, rather than the input spelling.
    assert all(r.sampling_params.ignore_eos and r.sampling_params.top_k == 1 for r in reqs)
    assert merged.sampling_info.is_all_greedy
    return merged, token, ids
