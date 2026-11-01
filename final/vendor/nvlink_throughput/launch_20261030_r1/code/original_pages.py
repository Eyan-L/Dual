"""One GQA attention kernel reads tier-tagged local/peer pages, then split reduce."""
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _unified_pages(Q, LK, LV, RK, RV, Tier, Slot, Seq, Acc, Max, Den,
                   STRIDE: tl.constexpr, R: tl.constexpr, SPLITS: tl.constexpr,
                   N: tl.constexpr, RN: tl.constexpr, STAGES: tl.constexpr):
    h = tl.program_id(0)
    split = tl.program_id(1)
    b = tl.program_id(2)
    m = tl.arange(0, 16)
    d = tl.arange(0, 128)
    row = tl.arange(0, N)
    length = tl.load(Seq+b)
    local_length = length-R
    lc = tl.cdiv(local_length, SPLITS)
    rc = tl.cdiv(R, SPLITS)
    lo, ro = split*lc, split*rc
    lend, rend = tl.minimum(lo+lc, local_length), tl.minimum(ro+rc, R)
    q = tl.load(Q+(b*32+h*4+m[:, None])*128+d[None, :], m[:, None] < 4, 0)
    maximum = tl.full((16,), float('-inf'), tl.float32)
    denominator = tl.zeros((16,), tl.float32)
    numerator = tl.zeros((16, 128), tl.float32)
    trips = tl.maximum(tl.cdiv(lc, N-RN), tl.cdiv(rc, RN))
    for tile in tl.range(0, trips, num_stages=STAGES):
        remote_row = row < RN
        rt = ro+tile*RN+row
        lt = lo+tile*(N-RN)+row-RN
        logical = tl.where(remote_row, rt, R+lt)
        valid = tl.where(remote_row, rt < rend, lt < lend)
        tier = tl.load(Tier+b*STRIDE+logical, valid, 0)
        slot = tl.load(Slot+b*STRIDE+logical, valid, 0)
        kp = tl.where(tier[:, None] == 1, RK, LK)
        vp = tl.where(tier[:, None] == 1, RV, LV)
        offset = slot[:, None]*1024+h*128+d[None, :]
        k = tl.load(kp+offset, valid[:, None], 0, cache_modifier='.cg')
        v = tl.load(vp+offset, valid[:, None], 0, cache_modifier='.cg')
        score = tl.dot(q, tl.trans(k), out_dtype=tl.float32)*(0.08838834764831845*1.4426950408889634)
        score = tl.where(valid[None, :], score, float('-inf'))
        updated = tl.maximum(maximum, tl.max(score, axis=1))
        factor = tl.exp2(maximum-updated)
        probability = tl.exp2(score-updated[:, None])
        numerator = numerator*factor[:, None]+tl.dot(probability.to(tl.bfloat16), v, out_dtype=tl.float32)
        denominator = denominator*factor+tl.sum(probability, axis=1)
        maximum = updated
    output_row = ((b*8+h)*SPLITS+split)*4+m
    tl.store(Acc+output_row[:, None]*128+d[None, :], numerator, m[:, None] < 4)
    tl.store(Max+output_row, maximum, m < 4)
    tl.store(Den+output_row, denominator, m < 4)


@triton.jit
def _reduce_pages(Acc, Max, Den, Out, SPLITS: tl.constexpr, PAD: tl.constexpr):
    h, b = tl.program_id(0), tl.program_id(1)
    s = tl.arange(0, PAD)
    d = tl.arange(0, 128)
    row = ((b*8+h//4)*SPLITS+s)*4+h%4
    maximum = tl.load(Max+row, s < SPLITS, float('-inf'))
    den = tl.load(Den+row, s < SPLITS, 0.)
    factor = tl.exp2(maximum-tl.max(maximum, axis=0))
    acc = tl.load(Acc+row[:, None]*128+d[None, :], s[:, None] < SPLITS, 0.)
    value = tl.sum(acc*factor[:, None], axis=0)/tl.sum(den*factor, axis=0)
    tl.store(Out+(b*32+h)*128+d, value)


class PageAttention:
    def __init__(self, control, cfg, tuning):
        self.control, self.cfg, self.tuning = control, cfg, tuning
        self.batch = cfg['batch']
        self.remote = cfg['remote_context_per_request']
        self.maximum_context = cfg['local_full_context']+self.remote
        self.splits, self.tile_n = tuning['splits'], tuning['tile_n']
        self.remote_rows = self.tile_n//16
        dev = control.raw.device
        self.tier = torch.zeros((self.batch, self.maximum_context), device=dev, dtype=torch.int32)
        self.slot = torch.zeros_like(self.tier)
        self.tier[:, :self.remote] = 1
        self.slot[:, :self.remote] = torch.arange(self.batch*self.remote, device=dev, dtype=torch.int32).view(self.batch, self.remote)
        self.seq = torch.empty(self.batch, device=dev, dtype=torch.int32)
        shape = (36, self.batch, 8, self.splits, 4)
        self.acc = torch.empty((*shape, 128), device=dev, dtype=torch.float32)
        self.maximum = torch.empty(shape, device=dev, dtype=torch.float32)
        self.denominator = torch.empty_like(self.maximum)
        self.output = torch.empty((36, self.batch, 32, 128), device=dev, dtype=torch.bfloat16)
        self.kernel = None

    @property
    def label(self):
        t = self.tuning
        return f"S{t['splits']}_N{t['tile_n']}_P{t['num_stages']}"

    def update(self, batch):
        context = int(batch.seq_lens_cpu[0])
        assert self.remote < context <= self.maximum_context
        assert batch.seq_lens_cpu.tolist() == [context]*self.batch
        self.seq.copy_(batch.seq_lens)
        source = self.control.raw.req_to_token_pool.req_to_token[batch.req_pool_indices, self.remote:context]
        self.slot[:, self.remote:context].copy_(source)

    def run(self, q, layer, shadow=False):
        c = self.control
        lk, lv = c.raw.token_to_kv_pool.get_kv_buffer(layer)
        remote = c.shadow if shadow else c.peer
        self.kernel = _unified_pages[(8, self.splits, self.batch)](
            q, lk, lv, remote[layer, 0], remote[layer, 1], self.tier, self.slot, self.seq,
            self.acc[layer], self.maximum[layer], self.denominator[layer],
            self.maximum_context, self.remote, self.splits, self.tile_n, self.remote_rows,
            self.tuning['num_stages'], num_warps=self.tuning['num_warps'], num_stages=self.tuning['num_stages'])
        self.reduce_kernel = _reduce_pages[(32, self.batch)](
            self.acc[layer], self.maximum[layer], self.denominator[layer], self.output[layer],
            self.splits, triton.next_power_of_2(self.splits), num_warps=4)
        return self.output[layer]

    def metadata(self):
        return dict(name=self.label, tuning=self.tuning,
                    registers_per_thread=getattr(self.kernel, 'n_regs', None),
                    spills=getattr(self.kernel, 'n_spills', None),
                    shared_memory_bytes=getattr(self.kernel.metadata, 'shared', None),
                    generated_cp_async='cp.async' in self.kernel.asm['ptx'],
                    generated_tensorcore_mma='mma.sync' in self.kernel.asm['ptx'],
                    attention_blocks=8*self.splits*self.batch,
                    remote_rows_per_tile=self.remote_rows, local_rows_per_tile=self.tile_n-self.remote_rows)
