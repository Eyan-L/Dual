"""Pressure-triggered migration, bounded by strict active remote/local ratio.

Async starts after forward submission; ownership commits at the next allocation.
The current token may become CPU-visible before that commit. No KV writer or
attention arithmetic changes from the validated dynamic-combinations release.
"""

from pathlib import Path as _FinalPath
_SLACK_FINAL_ROOT = next(p for p in _FinalPath(__file__).resolve().parents if (p / 'release.json').is_file())
_SLACK_PROJECT_ROOT = _SLACK_FINAL_ROOT.parent
import importlib.util
from pathlib import Path
import sys
import time
import torch

SOURCE=Path(str(_SLACK_FINAL_ROOT / 'vendor/nvlink_dynamic/launch_combinations_20261031_r1/code/backend.py'))
spec=importlib.util.spec_from_file_location('frozen_dynamic_combinations',SOURCE)
frozen=importlib.util.module_from_spec(spec);sys.modules[spec.name]=frozen;spec.loader.exec_module(frozen)
scope=frozen.scope


def limit_pages(local,remote,numerator=599,denominator=10000):
    # denominator*(remote+m) < numerator*(local-m), strictly, without float rounding.
    return max(0,(numerator*local-denominator*remote-1)//(denominator+numerator))


class TimingBackend(frozen.TimingBackend):
    def reset(self,batch,profile=False):
        super().reset(batch,profile)
        self.pending=None;self.committed_remote=0;self.launched=None

    @property
    def remote_active(self):return self.committed_remote

    def budget(self):
        local=self.cfg['kv_capacity_tokens']-self.physical_allocator.available_size()
        remote=self.committed_remote
        return local,remote,min(limit_pages(local,remote),len(self.peer_free),len(self.residents))

    def amount(self,needed):
        local,remote,limit=self.budget()
        amount=min(self.variant['migration_tokens'],limit)
        required=max(0,needed-self.physical_allocator.available_size())
        assert amount>=required,(needed,required,amount,local,remote,limit)
        return amount,local,remote,limit

    def launch(self,needed,asynchronous):
        assert self.pending is None
        amount,local,remote,limit=self.amount(needed)
        selected=[self.residents.popleft() for _ in range(amount)]
        target_ids=[self.peer_free.popleft() for _ in selected]
        main=torch.cuda.current_stream();stream=self.copy_stream if asynchronous else main
        previous=self.frames[-1]
        assert all(self.created[item]<previous['uid'] for item in selected),'Migration victims must be immutable before the preceding model starts'
        event=self.events();done=torch.cuda.Event();begin=time.perf_counter()
        if asynchronous:stream.wait_event(previous['_model_events'][0])
        with scope(f'SLACK_COST_MIGRATION_LAUNCH_{previous["uid"]}',self.profile):
            with torch.cuda.stream(stream):
                event[0].record()
                rid=torch.tensor([r for r,p in selected],device=self.raw.device,dtype=torch.int64)
                pos=torch.tensor([p for r,p in selected],device=self.raw.device,dtype=torch.int64)
                source=self.slot[rid,pos].long()
                targets=torch.tensor(target_ids,device=self.raw.device,dtype=torch.int64)
                self.copy_pages(source,targets)
                event[1].record();done.record()
        self.copy_references.append((rid,pos,source,targets,done))
        record=dict(tokens=amount,requested_tokens=self.variant['migration_tokens'],budget_tokens=limit,
            local_before=local,remote_before=remote,local_after=local-amount,remote_after=remote+amount,
            ratio_after=(remote+amount)/(local-amount),ratio_strict_pass=10000*(remote+amount)<599*(local-amount),
            copy_bytes=amount*147456,async_copy=asynchronous,launch_step=(previous['uid'] if asynchronous else len(self.frames)),
            launch_cpu_ms=(time.perf_counter()-begin)*1000,_events=event,uid=-1,
            pending_peer_reservation=amount,source_ready='preceding model start; immutable older KV')
        assert record['ratio_strict_pass']
        self.pending=(rid,pos,source,targets,done,record)
        self.launched=record

    def commit(self):
        assert self.pending is not None
        rid,pos,source,targets,done,record=self.pending
        begin=time.perf_counter();wait=self.events();wait[0].record()
        with scope(f'SLACK_COST_MIGRATION_COMMIT_{len(self.frames)}',self.profile):
            if record['async_copy']:torch.cuda.current_stream().wait_event(done)
            else:done.synchronize()
            wait[1].record()
            # Main-stream ordering places reuse after both copy completion
            # and the previous forward's readers. No writer uses copy_stream.
            self.physical_allocator.free(source)
            self.raw.req_to_token_pool.req_to_token[rid,pos]=0
            self.tier[rid,pos]=1;self.slot[rid,pos]=targets.to(torch.int32)
        self.committed_remote+=record['tokens']
        local,remote,_=self.budget()
        assert (local,remote)==(record['local_after'],record['remote_after'])
        assert 10000*remote<599*local
        record.update(uid=len(self.frames),commit_step=len(self.frames),commit_cpu_ms=(time.perf_counter()-begin)*1000,
            _wait_events=wait,phase='decode',turn=0,step=len(self.frames),wall_ms=record['launch_cpu_ms']+(time.perf_counter()-begin)*1000)
        self.migrations.append(record);self.pending=None

    def ensure_headroom(self,needed):
        begin=time.perf_counter()
        if self.pending is not None:self.commit()
        if self.physical_allocator.available_size()<needed:
            assert not self.variant['async_copy'],'Async prefetch was not scheduled before local exhaustion'
            self.launch(needed,False);self.commit()
        return (time.perf_counter()-begin)*1000

    def prefetch_next(self,has_next):
        self.launched=None
        if has_next and self.variant['async_copy'] and self.physical_allocator.available_size()<self.batch:
            self.launch(self.batch,True)

    def finish(self):
        assert self.pending is None
        torch.cuda.synchronize()
        for record in self.migrations:
            a,z=record.pop('_wait_events');record['commit_wait_gpu_ms']=a.elapsed_time(z)
        return super().finish()
