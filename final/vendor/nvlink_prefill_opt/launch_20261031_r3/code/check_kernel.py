"""Actual IPC page correctness and isolated prefill timing, outside model timers."""

from pathlib import Path as _FinalPath
_SLACK_FINAL_ROOT = next(p for p in _FinalPath(__file__).resolve().parents if (p / 'release.json').is_file())
_SLACK_PROJECT_ROOT = _SLACK_FINAL_ROOT.parent
import functools
import importlib.util
from pathlib import Path
import statistics
import torch
from optimized_prefill import prefill
from backend import original_prefill
from experiment_common import save

CANDIDATES=[dict(name='bm32_s1',block_m=32,splits=1),dict(name='bm64_s1',block_m=64,splits=1),
            dict(name='bm64_s2',block_m=64,splits=2),dict(name='bm32_s4',block_m=32,splits=4),
            dict(name='bm64_s4',block_m=64,splits=4)]


def run(peer,root):
    path=Path(str(_SLACK_FINAL_ROOT / 'vendor/nvlink_prefill/launch_20261031_r5/code/validate_kernel.py'))
    spec=importlib.util.spec_from_file_location('frozen_prefill_validator',path)
    validator=importlib.util.module_from_spec(spec);spec.loader.exec_module(validator)
    checks=[]
    for choice in CANDIDATES:
        validator.prefill=functools.partial(prefill,block_m=choice['block_m'],splits=choice['splits'])
        checks.append(dict(kernel=choice,checks=validator.validate(peer)))
        save(root/'kernel_checks.json',checks)
        print('PREFILL5_KERNEL_VALIDATED',choice['name'],flush=True)
    rows=[];device=peer.device
    for prefix,qlen in [(0,256),(2048,64),(2048,256),(8192,256),(24576,64),(24576,256)]:
        batch=2;length=prefix+qlen;total=batch*length
        local_k=torch.randn((total+1,8,128),device=device,dtype=torch.bfloat16)
        local_v=torch.randn_like(local_k);peer_k,peer_v=peer[0,0],peer[0,1]
        q=torch.randn((batch*qlen,32,128),device=device,dtype=torch.bfloat16)
        for placement in ['local','mixed']:
            # Deterministic sparse/noncontiguous logical placement; local slots
            # are shuffled, so no contiguous-prefix assumption can pass by luck.
            nr=0 if placement=='local' else min(peer.shape[2],599*total//10599)
            is_remote=torch.zeros(total,device=device,dtype=torch.bool)
            if nr:is_remote[torch.linspace(0,total-1,nr,device=device).long()]=True
            assert int(is_remote.sum())==nr
            slots=torch.empty(total,device=device,dtype=torch.int32)
            slots[~is_remote]=torch.randperm(total-nr,device=device,dtype=torch.int64).to(torch.int32)+1
            slots[is_remote]=torch.arange(nr,device=device,dtype=torch.int32)
            peer_k[:nr].normal_();peer_v[:nr].normal_()
            tier=is_remote.to(torch.int32).reshape(batch,length).contiguous()
            slot=slots.reshape(batch,length).contiguous()
            args=(q,local_k,local_v,peer_k,peer_v,tier,slot,
                torch.arange(batch,device=device,dtype=torch.int32),
                torch.full((batch,),length,device=device,dtype=torch.int32),
                torch.arange(batch,device=device,dtype=torch.int32)*qlen,
                torch.full((batch,),qlen,device=device,dtype=torch.int32),
                torch.full((batch,),prefix,device=device,dtype=torch.int32),qlen,128**-.5)
            for choice in [dict(name='r5')]+CANDIDATES:
                fn=original_prefill if choice['name']=='r5' else functools.partial(prefill,block_m=choice['block_m'],splits=choice['splits'])
                for _ in range(3):fn(*args)
                torch.cuda.synchronize();samples=[]
                for _ in range(5):
                    a=torch.cuda.Event(enable_timing=True);z=torch.cuda.Event(enable_timing=True)
                    a.record()
                    for _ in range(10):output,_=fn(*args)
                    z.record();z.synchronize();samples.append(a.elapsed_time(z)/10)
                rows.append(dict(prefix=prefix,query_length=qlen,batch=batch,placement=placement,
                    remote_tokens=nr,kernel=choice,mean_ms=statistics.mean(samples),samples_ms=samples))
                del output
            save(root/'kernel_timing.json',rows)
            print('PREFILL5_MICRO_TIMED',prefix,qlen,placement,flush=True)
    return rows
