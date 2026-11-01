"""Extend the validated1+2+4/2048 policy to a measured aggregate remote bound."""
from collections import deque
from contextlib import nullcontext
from itertools import islice
import time
import torch
from fused_copy import restore


def scope(name,enabled):return torch.profiler.record_function(name) if enabled else nullcontext()


def audit(self):
    if not self.enabled:
        used=self.cfg['kv_capacity_tokens']-self.physical_allocator.available_size()
        assert used==sum(self.live.values())
        return dict(local_active_tokens=used,remote_active_tokens=0,logical_tokens=used,donor_active_tokens=[0,0,0])
    near=[];far=[];requests=[]
    for rid,n in self.live.items():
        t=self.tier[rid,:n];s=self.slot[rid,:n];f=t==1
        assert ((t==0)|f).all().item()
        assert torch.equal(self.raw.req_to_token_pool.req_to_token[rid,:n],torch.where(f,0,s))
        near.append(s[~f]);far.append(s[f])
        requests.append(dict(request=rid,context=n,local=int((~f).sum()),remote=int(f.sum())))
    device=self.raw.device
    n=torch.cat(near) if near else torch.empty(0,device=device,dtype=torch.int32)
    f=torch.cat(far) if far else torch.empty(0,device=device,dtype=torch.int32)
    reserved=self.pending[-1]['tokens'] if self.pending is not None else 0
    assert torch.unique(n).numel()==n.numel() and torch.unique(f).numel()==f.numel()
    assert n.numel()==self.cfg['kv_capacity_tokens']-self.physical_allocator.available_size()
    assert f.numel()==self.committed_remote==len(self.remote_positions)
    assert f.numel()+len(self.peer_free)+reserved==self.peer.shape[2]
    assert (n>0).all().item() and (n<=self.cfg['kv_capacity_tokens']).all().item()
    assert (f>=0).all().item() and (f<self.peer.shape[2]).all().item()
    assert not f.numel() or self.cfg['remote_ratio_denominator']*f.numel()<self.cfg['remote_ratio_numerator']*n.numel()
    assert n.numel()+f.numel()==sum(self.live.values())
    return dict(local_active_tokens=n.numel(),remote_active_tokens=f.numel(),logical_tokens=sum(self.live.values()),
        per_request=requests,page_tables_exact=True,unique_local_slots=True,unique_peer_slots=True,
        remote_local_shadow_present=False,donor_active_tokens=torch.bincount((f%3).long(),minlength=3).cpu().tolist())


def budget(self):
    local=self.cfg['kv_capacity_tokens']-self.physical_allocator.available_size()
    remote=self.committed_remote;a=self.cfg['remote_ratio_numerator'];b=self.cfg['remote_ratio_denominator']
    limit=max(0,(a*local-b*remote-1)//(a+b))
    return local,remote,min(limit,len(self.peer_free),len(self.residents))


def launch(self,needed,asynchronous):
    assert self.pending is None
    amount,local,remote,limit=self.amount(needed)
    selected=[self.residents.popleft() for _ in range(amount)]
    targets=[self.peer_free.popleft() for _ in selected]
    previous=self.frames[-1]
    assert all(self.created[item]<previous['uid'] for item in selected)
    stream=self.copy_stream if asynchronous else torch.cuda.current_stream()
    events=self.events();done=torch.cuda.Event();begin=time.perf_counter()
    if asynchronous:stream.wait_event(previous['_model_events'][0])
    with scope(f'SLACK_COST_MIGRATION_LAUNCH_{previous["uid"]}',self.profile),torch.cuda.stream(stream):
        events[0].record()
        rid=torch.tensor([r for r,p in selected],device=self.raw.device,dtype=torch.int64)
        pos=torch.tensor([p for r,p in selected],device=self.raw.device,dtype=torch.int64)
        source=self.slot[rid,pos].long();target=torch.tensor(targets,device=self.raw.device,dtype=torch.int64)
        self.copy_pages(source,target);events[1].record();done.record()
    self.copy_references.append((rid,pos,source,target,done))
    a=self.cfg['remote_ratio_numerator'];b=self.cfg['remote_ratio_denominator']
    record=dict(tokens=amount,requested_tokens=self.variant['migration_tokens'],budget_tokens=limit,
        local_before=local,remote_before=remote,local_after=local-amount,remote_after=remote+amount,
        ratio_after=(remote+amount)/(local-amount),ratio_strict_pass=b*(remote+amount)<a*(local-amount),
        copy_bytes=amount*147456,async_copy=asynchronous,launch_step=previous['uid'] if asynchronous else len(self.frames),
        launch_cpu_ms=(time.perf_counter()-begin)*1000,_events=events,uid=-1,pending_peer_reservation=amount,
        source_ready='preceding model start;immutable olderKV',donor_target_counts=[sum(t%3==d for t in targets) for d in range(3)])
    assert record['ratio_strict_pass']
    self.pending=(rid,pos,source,target,done,record);self.launched=record
    self.pending_coordinates=selected;self.pending_targets=targets


def commit(self):
    assert self.pending is not None
    rid,pos,source,target,done,record=self.pending
    if self.validation:
        done.synchronize()
        for layer in range(36):
            for plane,local in enumerate(self.raw.token_to_kv_pool.get_kv_buffer(layer)):
                assert torch.equal(local.index_select(0,source),self.peer[layer,plane].index_select(0,target)),('copy payload',layer,plane)
    begin=time.perf_counter();wait=self.events();wait[0].record()
    if record['async_copy']:torch.cuda.current_stream().wait_event(done)
    else:done.synchronize()
    wait[1].record()
    self.physical_allocator.free(source)
    self.raw.req_to_token_pool.req_to_token[rid,pos]=0
    self.tier[rid,pos]=1;self.slot[rid,pos]=target.to(torch.int32)
    self.committed_remote+=record['tokens']
    local,remote,_=self.budget()
    assert (local,remote)==(record['local_after'],record['remote_after'])
    assert self.cfg['remote_ratio_denominator']*remote<self.cfg['remote_ratio_numerator']*local
    record.update(uid=len(self.frames),commit_step=len(self.frames),commit_cpu_ms=(time.perf_counter()-begin)*1000,
        _wait_events=wait,**{k:self.context.get(k) for k in ['phase','turn','chunk','step']},
        wall_ms=record['launch_cpu_ms']+(time.perf_counter()-begin)*1000)
    self.migrations.append(record);self.pending=None
    self.remote_positions.update(zip(self.pending_coordinates,self.pending_targets))
    self.remote_counts.update(r for r,p in self.pending_coordinates)
    self.pending_coordinates=[];self.pending_targets=[]
    if self.numerical_shadow is not None:self.numerical_shadow.copy_(self.peer)


def repair_active_ratio(self):
    local,remote,_=self.budget();a=self.cfg['remote_ratio_numerator'];b=self.cfg['remote_ratio_denominator']
    if not remote or b*remote<a*local:return
    count=(b*remote-a*local)//(a+b)+1
    selected=list(islice(reversed(self.remote_positions),count));assert len(selected)==count
    source=torch.tensor([self.remote_positions[k] for k in selected],device=self.raw.device,dtype=torch.int64)
    target=self.physical_allocator.alloc(count);assert target is not None
    rid=torch.tensor([r for r,p in selected],device=self.raw.device,dtype=torch.int64)
    pos=torch.tensor([p for r,p in selected],device=self.raw.device,dtype=torch.int64)
    if not hasattr(self,'local_pointers'):
        self.local_pointers=torch.tensor([v.data_ptr() for i in range(36) for v in self.raw.token_to_kv_pool.get_kv_buffer(i)],device=self.raw.device,dtype=torch.int64)
    begin=time.perf_counter_ns();events=self.events();events[0].record()
    restore(self.local_pointers,self.peer,source,target)
    self.tier[rid,pos]=0;self.slot[rid,pos]=target.to(torch.int32)
    self.raw.req_to_token_pool.req_to_token[rid,pos]=target.to(torch.int32)
    for key in selected:
        self.peer_free.append(self.remote_positions.pop(key));self.remote_counts[key[0]]-=1
    self.committed_remote-=count
    self.residents.extend(selected);self.residents=deque(sorted(self.residents,key=lambda k:(self.created[k],k)))
    events[1].record();l,r,_=self.budget();assert not r or b*r<a*l
    self.restorations.append(dict(uid=len(self.frames),tokens=count,local_before=local,remote_before=remote,
        local_after=l,remote_after=r,cpu_ms=(time.perf_counter_ns()-begin)/1e6,_events=events))
