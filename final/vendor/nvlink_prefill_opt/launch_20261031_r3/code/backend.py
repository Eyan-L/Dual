"""Prefill-only kernel dispatch on the frozen 2048-page decode policy."""

from pathlib import Path as _FinalPath
_SLACK_FINAL_ROOT = next(p for p in _FinalPath(__file__).resolve().parents if (p / 'release.json').is_file())
_SLACK_PROJECT_ROOT = _SLACK_FINAL_ROOT.parent
import importlib.util
from pathlib import Path
import sys
import torch
from optimized_prefill import prefill as optimized_prefill

SOURCE=Path(str(_SLACK_FINAL_ROOT / 'vendor/nvlink_migration_latency/launch_large_20261031_r2/code/backend.py'))
spec=importlib.util.spec_from_file_location('frozen_migration_backend_for_prefill5',SOURCE)
migration=importlib.util.module_from_spec(spec);sys.modules[spec.name]=migration;spec.loader.exec_module(migration)
scope=migration.scope
original_prefill=migration.frozen.frozen.prefill


class TimingBackend(migration.TimingBackend):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.kernel_choice=dict(self.cfg.get('prefill_kernel',dict(name='r5',block_m=32,splits=1)))
        self.check_attention=False;self.attention_checks=[]
        # The original forward hook keeps its native KV writer and decode.
        migration.frozen.frozen.prefill=self.prefill

    def prefill(self,*args,**kwargs):
        if self.kernel_choice['name']=='r5' or (self.kernel_choice['name']=='peer_auto' and
                (self.remote_active==0 or args[12]<128)):
            return original_prefill(*args,**kwargs)
        actual,kernel=optimized_prefill(*args,**kwargs,
            block_m=self.kernel_choice['block_m'],splits=self.kernel_choice['splits'])
        if self.check_attention:
            expected,_=original_prefill(*args,**kwargs)
            mismatch=~torch.isclose(actual,expected,atol=self.cfg['reference_atol'],rtol=self.cfg['reference_rtol'])
            disputed=int(mismatch.sum());adjudications=[]
            if disputed:
                q,lk,lv,rk,rv,tier,slot,requests,seq,qstart,qlen,prefix,maxq,scale=args
                starts=qstart.cpu().tolist();lengths=qlen.cpu().tolist();prefixes=prefix.cpu().tolist()
                reqs=requests.cpu().tolist()
                for token,head in sorted({(t,h) for t,h,d in mismatch.nonzero().cpu().tolist()}):
                    b=next(i for i,(s,n) in enumerate(zip(starts,lengths)) if s<=token<s+n)
                    visible=prefixes[b]+token-starts[b]+1;rid=reqs[b]
                    positions=slot[rid,:visible].long();remote=tier[rid,:visible].bool();kh=head//(q.shape[1]//lk.shape[1])
                    key=torch.empty((visible,128),device=q.device,dtype=torch.float64)
                    value=torch.empty_like(key)
                    for dst,local,peer in ((key,lk,rk),(value,lv,rv)):
                        dst[~remote]=local[positions[~remote],kh].double()
                        dst[remote]=peer[positions[remote],kh].double()
                    scores=(key*q[token,head].double()).sum(-1)*scale
                    truth=(scores.softmax(0)[:,None]*value).sum(0)
                    candidate=actual[token,head].double();old=expected[token,head].double()
                    bad=~torch.isclose(candidate,truth,atol=self.cfg['reference_atol'],rtol=self.cfg['reference_rtol'])
                    assert not bad.any(),('optimized prefill failed FP64',self.kernel_choice,token,head,float((candidate-truth).abs().max()))
                    adjudications.append(dict(token=token,head=head,visible_keys=visible,candidate_fp64_failures=int(bad.sum()),
                        original_fp64_failures=int((~torch.isclose(old,truth,atol=self.cfg['reference_atol'],rtol=self.cfg['reference_rtol'])).sum()),
                        candidate_max_abs_error=float((candidate-truth).abs().max())))
            self.attention_checks.append(dict(forward=len(self.frames),kernel=self.kernel_choice,
                compared_elements=actual.numel(),original_disputed_elements=disputed,
                bitwise_equal=torch.equal(actual,expected),max_abs_error=float((actual.float()-expected.float()).abs().max()),
                relative_l2=float((actual.float()-expected.float()).norm()/expected.float().norm().clamp_min(1.e-12)),
                fp64_adjudications=adjudications))
        return actual,kernel

    def ensure_headroom(self,needed):
        if not self.enabled:return 0.
        if self.context.get('phase')=='prefill':
            import time
            start=time.perf_counter()
            if self.pending is not None:self.commit()
            if self.physical_allocator.available_size()<needed:
                # An unannounced appended input cannot be prefetched by decode.
                # Reserve/copy before its first allocation, with the same bound.
                self.launch(needed,False);self.commit()
            return (time.perf_counter()-start)*1000
        return super().ensure_headroom(needed)

    def prefetch_next(self,has_next):
        if self.enabled:return super().prefetch_next(has_next)

    def finish(self):
        result=super().finish()
        assert result['remote_active_tokens']*10000<result['local_active_tokens']*599
        return result
