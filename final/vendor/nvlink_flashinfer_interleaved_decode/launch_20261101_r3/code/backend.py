"""Native FlashInfer decode plus the frozen1+2+4/2048 KV manager."""

from pathlib import Path as _FinalPath
_SLACK_FINAL_ROOT = next(p for p in _FinalPath(__file__).resolve().parents if (p / 'release.json').is_file())
_SLACK_PROJECT_ROOT = _SLACK_FINAL_ROOT.parent
import importlib.util
from pathlib import Path
import sys
import types

import torch
import triton
import triton.language as tl
from flashinfer_peer import activate,module

SOURCE=Path(str(_SLACK_FINAL_ROOT / 'vendor/nvlink_short_serving/launch_local_20261031_r2/code/backend.py'))
spec=importlib.util.spec_from_file_location('frozen_124_2048_flashinfer',SOURCE)
base=importlib.util.module_from_spec(spec);sys.modules[spec.name]=base;spec.loader.exec_module(base)
scope=base.scope


@triton.jit
def _count_remote(Tier,Req,Seq,Remote,STRIDE:tl.constexpr,PAD:tl.constexpr):
    b=tl.program_id(0);pos=tl.arange(0,PAD)
    rid=tl.load(Req+b);length=tl.load(Seq+b)
    tier=tl.load(Tier+rid*STRIDE+pos,pos<length,0)
    tl.store(Remote+b,tl.sum((tier==1).to(tl.int32),0))


@triton.jit
def _pack_indices(Tier,Slot,Req,Seq,Remote,Out,STRIDE:tl.constexpr,CAP:tl.constexpr,B:tl.constexpr,PAD:tl.constexpr,BLOCK:tl.constexpr):
    b=tl.program_id(0);pos=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    lanes=tl.arange(0,PAD);lengths=tl.load(Seq+lanes,lanes<B,0)
    start=tl.sum(tl.where(lanes<b,lengths,0),0)
    length=tl.load(Seq+b);rid=tl.load(Req+b);valid=pos<length
    r=tl.load(Remote+b);l=length-r
    # Invert the diagnosed8remote/120local permutation. FIFO migration leaves
    # a remote logical prefix;the validation audit checks that invariant.
    # Handle either tier exhausting first,including a partial mixed tail tile.
    groups=tl.minimum(r//8,l//120);base=groups*128
    rr=r-groups*8;ll=l-groups*120
    tail_r=tl.minimum(rr,8);tail_l=tl.minimum(ll,120)
    missing=128-tail_r-tail_l
    extra_l=tl.minimum(missing,ll-tail_l);tail_l+=extra_l
    tail_r+=tl.minimum(missing-extra_l,rr-tail_r)
    lane=pos%128;group=pos//128
    in_full=pos<base;tail_pos=pos-base;in_tail=tail_pos<tail_r+tail_l
    full_logical=tl.where(lane<8,group*8+lane,r+group*120+lane-8)
    tail_logical=tl.where(tail_pos<tail_r,groups*8+tail_pos,r+groups*120+tail_pos-tail_r)
    rest=tail_pos-tail_r-tail_l
    rest_logical=tl.where(rr>tail_r,groups*8+tail_r+rest,r+groups*120+tail_l+rest)
    logical=tl.where(in_full,full_logical,tl.where(in_tail,tail_logical,rest_logical))
    slot=tl.load(Slot+rid*STRIDE+logical,valid,0);tier=tl.load(Tier+rid*STRIDE+logical,valid,0)
    tl.store(Out+start+pos,slot+tl.where(tier==1,CAP+1,0),valid)


class Engine:
    def __init__(self,control,batch):
        self.control,self.batch=control,batch;device=control.raw.device
        self.requests=torch.zeros(batch,device=device,dtype=torch.int32)
        self.seq=torch.ones_like(self.requests)
        self.remote=torch.zeros_like(self.requests)
        self.indices=torch.zeros(control.cfg['kv_capacity_tokens']+control.cfg['remote_capacity_tokens'],device=device,dtype=torch.int32)
        self.output=torch.empty((36,batch,32,128),device=device,dtype=torch.bfloat16)

    def prepare(self,fb):
        self.requests.copy_(fb.req_pool_indices);self.seq.copy_(fb.seq_lens)
        assert sum(fb.seq_lens_cpu.tolist())<=self.indices.numel()
        _count_remote[(self.batch,)](self.control.tier,self.requests,self.seq,self.remote,
            self.control.tier.shape[1],triton.next_power_of_2(max(fb.seq_lens_cpu.tolist())),num_warps=4)
        _pack_indices[(self.batch,triton.cdiv(max(fb.seq_lens_cpu.tolist()),256))](
            self.control.tier,self.control.slot,self.requests,self.seq,self.remote,self.indices,
            self.control.tier.shape[1],self.control.cfg['kv_capacity_tokens'],self.batch,
            triton.next_power_of_2(self.batch),256,num_warps=4)


class Backend(base.ReplayBackend):
    def audit(self):
        result=super().audit()
        if self.numerical_shadow is not None:
            result['remote_local_shadow_present']=True
            result['numerical_reference_only']=True
        if self.enabled:
            for rid,length in self.live.items():
                r=self.remote_counts.get(rid,0)
                assert (self.tier[rid,:r]==1).all().item(),'Interleaving requires FIFO remote prefix'
                assert (self.tier[rid,r:length]==0).all().item(),'Interleaving requires FIFO remote prefix'
            result['fifo_remote_prefix_exact']=True
            if self.validation and self.live and self.last_fb is not None and self.last_fb.forward_mode.is_decode():
                wanted=[]
                for rid,length in zip(self.engine.requests.cpu().tolist(),self.engine.seq.cpu().tolist()):
                    r=self.remote_counts.get(rid,0)
                    encoded=self.slot[rid,:length].cpu()+torch.where(torch.arange(length)<r,self.cfg['kv_capacity_tokens']+1,0)
                    far=encoded[:r].tolist();near=encoded[r:].tolist();ri=li=0
                    while ri<len(far) or li<len(near):
                        rr=far[ri:ri+8];ll=near[li:li+120];ri+=len(rr);li+=len(ll)
                        missing=128-len(rr)-len(ll)
                        extra=near[li:li+missing];ll+=extra;li+=len(extra);missing-=len(extra)
                        extra=far[ri:ri+missing];rr+=extra;ri+=len(extra)
                        wanted.extend(rr+ll)
                assert torch.equal(self.engine.indices[:len(wanted)].cpu(),torch.tensor(wanted,dtype=torch.int32)),'GPU read order differs from diagnosed permutation'
                result['interleaved_index_order_exact']=True
        return result

    def __init__(self,raw,peer,cfg,old_control,case):
        native_decode=old_control.native
        super().__init__(raw,peer,cfg,old_control,case)
        self.current_layer=0;self.current_scale=128**-.5
        self.read_peer=peer;self.numerical_shadow=None
        if self.enabled:self.install_flashinfer()
        control=self
        def decode(backend,q,k,v,layer,fb,save_kv_cache=True):
            if not control.enabled:return native_decode(q,k,v,layer,fb,save_kv_cache=save_kv_cache)
            assert layer.logit_cap in (None,0,0.) and not layer.is_cross_attention
            control.current_layer=layer.layer_id;control.current_scale=layer.scaling
            if control.validation:
                control.queries[fb.batch_size][layer.layer_id].copy_(q.contiguous().view(-1,32,128))
            output=native_decode(q,k,v,layer,fb,save_kv_cache=save_kv_cache)
            if control.validation:control.banks[fb.batch_size].output[layer.layer_id].copy_(output.view(-1,32,128))
            return output
        self.backend.forward_decode=types.MethodType(decode,self.backend)

    def install_flashinfer(self):
        from flashinfer import BatchDecodeWithPagedKVCacheWrapper as Wrapper
        module()  # Compile before capture and timed execution.
        control=self;original_init=Wrapper.__init__;original_run=Wrapper.run
        def init(wrapper,*args,**kwargs):
            original_init(wrapper,*args,**kwargs)
            assert wrapper.use_tensor_cores,'This Qwen3 BF16 integration expects the observed tensor-core decode path'
            activate(wrapper)
        def run(wrapper,q,paged_kv,*args,**kwargs):
            if not hasattr(wrapper,'slack_peer_metadata'):return original_run(wrapper,q,paged_kv,*args,**kwargs)
            assert not args
            engine=control.banks[q.shape[0]];i=control.current_layer
            return original_run(wrapper,q,paged_kv,control.read_peer[i,0],control.read_peer[i,1],engine.indices,
                control.cfg['kv_capacity_tokens'],control.current_scale,**kwargs)
        Wrapper.__init__=init;Wrapper.run=run
        # ModelRunner created native wrappers before this adapter was installed.
        # Capture may reuse them;activate existing wrapper containers as well.
        def visit(value,depth):
            if isinstance(value,Wrapper):
                assert value.use_tensor_cores;activate(value)
            elif depth and isinstance(value,dict):
                for child in value.values():visit(child,depth-1)
            elif depth and isinstance(value,(tuple,list)):
                for child in value:visit(child,depth-1)
        for value in vars(self.backend).values():visit(value,3)

    def prepare_graphs(self,max_batch):
        self.reset(1)
        if self.enabled:
            self.tier.zero_();self.slot.zero_()
            for i in range(36):
                for plane in self.raw.token_to_kv_pool.get_kv_buffer(i):plane[0].zero_()
            self.banks={b:Engine(self,b) for b in range(1,max_batch+1)}
            self.queries={b:torch.empty((36,b,32,128),device=self.raw.device,dtype=torch.bfloat16)
                          for b in range(1,max_batch+1)} if self.validation else {}
            self.engine=self.banks[1]
        graph=self.raw.decode_cuda_graph_runner
        graph.capture_bs=list(range(1,max_batch+1));graph.capture();torch.cuda.synchronize();self.reset(1)

    def commit(self):
        super().commit()
        self.migrations[-1].update({k:self.context.get(k) for k in ['phase','step']})
        if self.numerical_shadow is not None:
            # Numerical reference only:identical page order and arithmetic,
            # but far-page payload is independently copied to local memory.
            self.numerical_shadow.copy_(self.peer)
