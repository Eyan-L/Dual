"""Native local/peer partial attention with physically reclaimed local KV slots."""
import types

import torch
from flashinfer import BatchDecodeWithPagedKVCacheWrapper, merge_states

from measure import AttentionEvents


class CooperativeAttention(AttentionEvents):
    def __init__(self, raw, peer, cfg):
        self.cfg = cfg
        self.remote_context = cfg['remote_context_per_request']
        self.capacity = cfg['kv_capacity_tokens']
        super().__init__(raw, peer, cfg['batch'])
        native = self.backend.forward_decode
        self.local_buffers = [tuple(x.unsqueeze(1) for x in raw.token_to_kv_pool.get_kv_buffer(i)) for i in range(36)]
        self.stream = torch.cuda.Stream(device=raw.device)
        self.ready = torch.cuda.Event()
        self.done = torch.cuda.Event()
        self.partial_output = torch.empty((2, 36, self.batch, 32, 128), device=raw.device, dtype=torch.bfloat16)
        self.partial_lse = torch.empty((2, 36, self.batch, 32), device=raw.device, dtype=torch.float32)
        self.local_wrapper = self.new_wrapper(self.capacity)
        self.remote_wrapper = self.new_wrapper(self.batch * self.remote_context)
        self.dummy_plan()
        self.shadow = None
        controller = self

        def forward_decode(backend, q, k, v, layer, forward_batch, save_kv_cache=True):
            if controller.mode == 'local':
                return native(q, k, v, layer, forward_batch, save_kv_cache=save_kv_cache)
            assert controller.mode in ('hybrid', 'shadow')
            assert k is not None and v is not None and save_kv_cache
            assert layer.logit_cap in (None, 0, 0.)
            backend.token_to_kv_pool.set_kv_buffer(layer, forward_batch.out_cache_loc, k, v,
                                                   layer.k_scale_float, layer.v_scale_float)
            query = q.contiguous().view(-1, 32, 128)
            i = layer.layer_id
            controller.queries[i].copy_(query)
            start, end = controller.events[i]
            start.record()
            output = controller.partial(query, i)
            end.record()
            controller.outputs[i].copy_(output)
            return output.view(-1, 4096)

        self.backend.forward_decode = types.MethodType(forward_decode, self.backend)

    def new_wrapper(self, slots):
        return BatchDecodeWithPagedKVCacheWrapper(
            torch.zeros(128 * 1024 * 1024, device=self.raw.device, dtype=torch.uint8),
            'NHD', use_cuda_graph=True,
            use_tensor_cores=self.backend.decode_use_tensor_cores, backend=self.backend.decode_backend,
            paged_kv_indptr_buffer=torch.empty(self.batch + 1, device=self.raw.device, dtype=torch.int32),
            paged_kv_indices_buffer=torch.empty(slots, device=self.raw.device, dtype=torch.int32),
            paged_kv_last_page_len_buffer=torch.empty(self.batch, device=self.raw.device, dtype=torch.int32))

    def plan_wrapper(self, wrapper, indices, length):
        assert indices.numel() == self.batch * length
        wrapper.plan(torch.arange(self.batch + 1, dtype=torch.int32) * length,
                     indices.to(dtype=torch.int32), torch.ones(self.batch, dtype=torch.int32),
                     32, 8, 128, 1, pos_encoding_mode='NONE', q_data_type=torch.bfloat16,
                     kv_data_type=torch.bfloat16, sm_scale=128 ** -.5)

    def dummy_plan(self):
        for wrapper in (self.local_wrapper, self.remote_wrapper):
            self.plan_wrapper(wrapper, torch.zeros(self.batch, dtype=torch.int32), 1)

    def recapture(self, mode):
        if mode != 'local':
            self.dummy_plan()
        super().recapture(mode)

    def plan(self, batch):
        context = int(batch.seq_lens_cpu[0])
        assert batch.seq_lens_cpu.tolist() == [context] * self.batch
        self.context = context
        self.local_indices = self.raw.req_to_token_pool.req_to_token[
            batch.req_pool_indices, self.remote_context:context].reshape(-1).to(torch.int64)
        assert self.local_indices.numel() <= self.capacity
        self.plan_wrapper(self.local_wrapper, self.local_indices, context - self.remote_context)
        self.plan_wrapper(self.remote_wrapper,
                          torch.arange(self.batch * self.remote_context, dtype=torch.int32), self.remote_context)

    def offload_prefix(self, batch):
        assert self.mode == 'hybrid' and self.shadow is None
        table = self.raw.req_to_token_pool.req_to_token
        slots = table[batch.req_pool_indices, :self.remote_context].reshape(-1).to(torch.int64)
        assert torch.unique(slots).numel() == self.batch * self.remote_context and slots.min().item() > 0
        records = []
        for i in range(36):
            k, v = self.raw.token_to_kv_pool.get_kv_buffer(i)
            expected_k, expected_v = k.index_select(0, slots), v.index_select(0, slots)
            self.peer[i, 0].copy_(expected_k)
            self.peer[i, 1].copy_(expected_v)
            assert torch.equal(self.peer[i, 0], expected_k) and torch.equal(self.peer[i, 1], expected_v)
            records.append(dict(layer=i, exact_peer_payload_match=True))
        torch.cuda.synchronize()
        before = self.raw.token_to_kv_pool_allocator.available_size()
        self.raw.token_to_kv_pool_allocator.free(slots)
        after = self.raw.token_to_kv_pool_allocator.available_size()
        assert after - before == slots.numel()
        # These are remote logical positions. A safe dummy replaces the stale
        # physical index; only our remote page table serves this prefix.
        table[batch.req_pool_indices[:, None], torch.arange(self.remote_context, device=self.raw.device)[None, :]] = 0
        self.freed_slots = slots.clone()
        self.plan(batch)
        assert not torch.isin(self.local_indices, slots).any().item()
        return dict(reclaimed_slots=int(slots.numel()), available_before=before,
                    available_after=after, layer_payload_checks=records,
                    local_reference_retained=False, placement='remote prefix, local suffix')

    def partial(self, query, i):
        layer = self.raw.model.model.layers[i].self_attn.attn
        buffers = self.peer_buffers if self.mode == 'hybrid' else self.shadow_buffers
        self.ready.record()
        with torch.cuda.stream(self.stream):
            self.stream.wait_event(self.ready)
            self.remote_wrapper.run(query, buffers[i], out=self.partial_output[1, i],
                                    lse=self.partial_lse[1, i], return_lse=True,
                                    k_scale=layer.k_scale_float, v_scale=layer.v_scale_float)
            self.done.record(self.stream)
        self.local_wrapper.run(query, self.local_buffers[i], out=self.partial_output[0, i],
                               lse=self.partial_lse[0, i], return_lse=True,
                               k_scale=layer.k_scale_float, v_scale=layer.v_scale_float)
        torch.cuda.current_stream().wait_event(self.done)
        output, _ = merge_states(torch.stack([self.partial_output[j, i] for j in range(2)], dim=1),
                                 torch.stack([self.partial_lse[j, i] for j in range(2)], dim=1))
        return output

    def check_full(self):
        """Stage ONE layer at a time solely for an untimed native local oracle."""
        total = self.batch * self.context
        wrapper = self.new_wrapper(total)
        self.plan_wrapper(wrapper, torch.arange(total, dtype=torch.int32), self.context)
        k_full = torch.empty((total, 1, 8, 128), device=self.raw.device, dtype=torch.bfloat16)
        v_full = torch.empty_like(k_full)
        rows = []
        for i in range(36):
            for plane, full in enumerate((k_full, v_full)):
                shaped = full.view(self.batch, self.context, 8, 128)
                shaped[:, :self.remote_context].copy_(self.peer[i, plane].view(self.batch, self.remote_context, 8, 128))
                local = self.raw.token_to_kv_pool.get_kv_buffer(i)[plane]
                shaped[:, self.remote_context:].copy_(local.index_select(0, self.local_indices).view(self.batch, self.context - self.remote_context, 8, 128))
            layer = self.raw.model.model.layers[i].self_attn.attn
            expected = wrapper.run(self.queries[i], (k_full, v_full),
                                   k_scale=layer.k_scale_float, v_scale=layer.v_scale_float)
            torch.testing.assert_close(self.outputs[i], expected,
                                       atol=self.cfg['reference_atol'], rtol=self.cfg['reference_rtol'])
            rows.append(dict(layer=i, max_abs_error=float((self.outputs[i] - expected).abs().max().item())))
        torch.cuda.synchronize()
        return dict(context=self.context, layers=rows, temporary_reference_layers=1,
                    reference_staging_excluded_from_primary_timing=True)

    def sensitivity(self):
        reference = self.partial(self.queries[0], 0).clone()
        saved = self.peer[0, 1].clone()
        self.peer[0, 1].add_(.5)
        changed = self.partial(self.queries[0], 0)
        difference = float((changed - reference).abs().max().item())
        self.peer[0, 1].copy_(saved)
        restored = self.partial(self.queries[0], 0)
        assert difference > 1e-3 and torch.equal(restored, reference)
        return dict(remote_V_change_affects_attention=difference, restored_exactly=True)

    def create_shadow(self):
        assert self.shadow is None
        # This extra local KV exists only AFTER all primary hybrid samples.
        self.shadow = self.peer.clone()
        assert torch.equal(self.shadow, self.peer)
        self.shadow_buffers = [(self.shadow[i, 0].unsqueeze(1), self.shadow[i, 1].unsqueeze(1)) for i in range(36)]
        torch.cuda.synchronize()
