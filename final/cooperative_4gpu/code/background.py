"""Protected foreground agents and lower-priority trace-derived background calls."""
from array import array
from collections import deque
import json
import time

from worker import save,subset


def replay(bench,wrapped,agents,origin,capacity,folder,cfg,transport=None):
    import torch
    raw=wrapped.torch_runner;wrapped.clear();bg=cfg['background']
    if transport is not None:transport.reset()
    assert bg['warmup_seconds']==0
    nodes=[];foreground=[];pending=deque();records=[];completed_calls=[]
    steps=0;max_used=0;max_remote=0;sequence=0;fg_done_ns=None;retractions=0;recomputed=0
    files={name:(folder/(name+'.jsonl')).open('w',buffering=1)for name in ['calls','kv_events','background','arrivals','retractions']}
    def emit(name,row):files[name].write(json.dumps(row,separators=(',',':'))+'\n')
    def call(n):return n['agent']['calls'][n['turn']]
    def make_req(n,length):
        req=bench.prepare_synthetic_inputs_for_latency_test(1,length,[[n['token']]*length])[0]
        req.rid=n['agent']['id'];req.sampling_params.ignore_eos=True;req.sampling_params.max_new_tokens=65536
        return req
    for a in agents:
        n=dict(agent=a,turn=0,context=0,generated=0,batch=None,phase='waiting',background=False,
            ready_ns=origin,first_start_ns=None,finish_ns=None,token=a['forced_token_id'])
        n['req']=make_req(n,a['calls'][0]['input_tokens']);nodes.append(n);foreground.append(n)
    assert sum(a['reserved_peak_tokens']for a in agents)<=capacity
    interval=round(bg['arrival_interval_seconds']*1e9);next_arrival=origin+round(bg['first_arrival_seconds']*1e9)
    def release(n):
        req=n['req']
        if req is not None and req.kv.holds_kv and transport is not None:
            transport.release(n)
        elif req is not None and req.kv.holds_kv:
            rid=req.kv.req_pool_idx;slots=raw.req_to_token_pool.req_to_token[rid,:n['context']].long()
            raw.token_to_kv_pool_allocator.free(slots);raw.req_to_token_pool.req_to_token[rid,:n['context']].zero_();raw.req_to_token_pool.free(req)
        n['req']=None
    def bg_reserved():return sum(n['peak']for n in nodes if n['background'] and n['phase']!='done')
    def fg_used():return sum(n['context']for n in foreground if n['phase']!='done')
    def fg_budget():return capacity-fg_used()-bg_reserved()
    def queue_arrivals(now):
        nonlocal next_arrival,sequence
        while next_arrival<=now:
            template=bg['templates'][sequence%len(bg['templates'])]
            r=dict(id=f'bg_{sequence}',planned_ns=next_arrival,observed_ns=now,status='queued',
                input_tokens=template['input_tokens'],output_tokens=template['output_tokens'],
                source_agent=template['source_agent'],source_round=template['source_round'],scale=template['scale'],
                admissions=0,retractions=0,recomputed_tokens=0,generated_tokens=0)
            a=dict(id=r['id'],group='background',calls=[dict(input_tokens=r['input_tokens'],output_tokens=r['output_tokens'],wait_after_ms=0)])
            n=dict(agent=a,req=None,turn=0,context=0,generated=0,batch=None,phase='queued',background=True,
                peak=r['input_tokens']+r['output_tokens']-1,prefill_target=r['input_tokens'],saved_generated=0,recompute_until=0,
                ready_ns=next_arrival,first_visible_ns=None,token=template['forced_token_id'],record=r)
            pending.append(n);records.append(r);emit('arrivals',r.copy());sequence+=1;next_arrival+=interval
    def ensure_foreground(needed):
        nonlocal retractions
        while fg_budget()<needed or (transport is not None and transport.available()<min(needed,1+sum(n['phase']=='decode'for n in foreground))):
            victims=[n for n in nodes if n['background']and n['phase']!='done']
            assert victims,'Protected foreground must fit independently in its original pool'
            n=max(victims,key=lambda n:n['record']['last_admitted_ns']);r=n['record'];discard=n['context'];start=time.perf_counter_ns()
            # No foreground node can enter this path. Preserve logical output progress.
            assert n['background']
            release(n);nodes.remove(n)
            n.update(saved_generated=n['generated'],prefill_target=r['input_tokens']+max(0,n['generated']-1),
                recompute_until=max(discard,n['recompute_until']),context=0,batch=None,phase='queued')
            r.update(status='queued',generated_tokens=n['generated'])
            if discard:r['retractions']+=1;retractions+=1
            pending.appendleft(n)
            emit('retractions',dict(perf_ns=time.perf_counter_ns(),id=r['id'],protected_foreground=True,
                released_tokens=discard,generated_tokens=n['generated'],needed_foreground_tokens=needed,
                elapsed_cpu_ms=(time.perf_counter_ns()-start)/1e6))
        assert fg_budget()>=needed
    while True:
        now=time.perf_counter_ns()
        if (folder.parent/'stop_background.json').exists():
            stop=json.loads((folder.parent/'stop_background.json').read_text())['perf_ns'];queue_arrivals(stop);break
        if fg_done_ns is not None and 'protocol_stop_after_foreground_seconds'in bg:
            if now-fg_done_ns>=bg['protocol_stop_after_foreground_seconds']*1e9:stop=now;break
        queue_arrivals(now)
        if transport is not None:transport.commit()
        for n in foreground:
            if n['phase']=='waiting'and now>=n['ready_ns']:
                n.update(phase='prefill',call_start_ns=now,first_visible_ns=None,generated=0)
        selected=[];budget=4096
        for n in foreground:
            if n['phase']!='prefill':continue
            size=min(call(n)['input_tokens']-n['context'],2048,budget)
            if size>0:selected.append((n,size));budget-=size
        fg_decode=sum(n['phase']=='decode'and n['generated']<call(n)['output_tokens']for n in foreground)
        fg_decode+=sum(n['context']+size==call(n)['input_tokens']and call(n)['output_tokens']>1 for n,size in selected)
        needed=sum(s for _,s in selected)+fg_decode
        # Foreground allocations win over every background reservation, even on a sudden append.
        ensure_foreground(needed)
        if transport is not None:
            # At the strict R/L boundary, bound an eager chunk by currently
            # reachable pages. Decode is always retained; chunking still progresses.
            ceiling=transport.available()
            if needed>ceiling:
                reserve=sum(n['phase']=='decode'for n in foreground)+len(selected)
                budget=max(0,ceiling-reserve);revised=[]
                for n,size in selected:
                    size=min(size,budget)
                    if size:revised.append((n,size));budget-=size
                assert revised or not selected,'Protected prefill must make progress'
                selected=revised
                fg_decode=sum(n['phase']=='decode'and n['generated']<call(n)['output_tokens']for n in foreground)
                fg_decode+=sum(n['context']+size==call(n)['input_tokens']and call(n)['output_tokens']>1 for n,size in selected)
                needed=sum(size for _,size in selected)+fg_decode
            budget=min(4096-sum(size for _,size in selected),max(0,ceiling-needed))
        while pending and sum(n['phase']!='done'for n in nodes)<bg['max_decode_batch']:
            n=pending[0]
            if fg_used()+bg_reserved()+n['peak']+needed+bg['admission_headroom_tokens']>capacity:break
            pending.popleft();r=n['record'];r.setdefault('admitted_ns',time.perf_counter_ns())
            r.update(status='running',last_admitted_ns=time.perf_counter_ns(),admissions=r['admissions']+1)
            n['req']=make_req(n,n['prefill_target']);n['phase']='prefill';nodes.append(n)
        isolate_small=bg.get('isolate_small_foreground_prefill',False) and selected and max(size for _,size in selected)<=32
        for n in nodes:
            if isolate_small:break
            if not n['background']or n['phase']!='prefill':continue
            size=min(n['prefill_target']-n['context'],2048,budget)
            if size>0:selected.append((n,size));budget-=size
        prefill_ms=0;decode_ms=0;redo_tokens=0
        if selected:
            if transport is not None:
                for n,size in selected:n['_selected_size']=size
                transport.before([n for n,_ in selected],'prefill')
            assert raw.token_to_kv_pool_allocator.available_size()>=sum(s for _,s in selected)
            for n,size in selected:
                lo=n['context'];hi=lo+size;req=n['req']
                if not n['background']and n['first_start_ns']is None:n['first_start_ns']=time.perf_counter_ns()
                if lo:req.prefix_indices=raw.req_to_token_pool.req_to_token[req.kv.req_pool_idx,:lo].to(req.prefix_indices.dtype)
                req.origin_input_ids=array('q',[n['token']]*hi);req.full_untruncated_fill_ids=req.origin_input_ids;req.set_extend_range(lo,hi)
            begin=time.perf_counter_ns();tokens,_,batch=bench.extend([n['req']for n,_ in selected],raw);tokens.cpu();visible=time.perf_counter_ns()
            prefill_ms=(visible-begin)/1e6;steps+=1
            if transport is not None:transport.after('prefill')
            for i,(n,size)in enumerate(selected):
                if n['background']:
                    redo=max(0,min(n['context']+size,n['recompute_until'])-min(n['context'],n['recompute_until']))
                    redo_tokens+=redo;n['record']['recomputed_tokens']+=redo
                n['context']+=size;target=n['prefill_target']if n['background']else call(n)['input_tokens']
                if n['context']==target:
                    generated=max(1,n['saved_generated'])if n['background']else 1
                    n.update(batch=subset(batch,[i]),generated=generated,phase='decode')
                    if n['first_visible_ns']is None:n['first_visible_ns']=visible
        decoders=[n for n in nodes if n['phase']=='decode'and n['generated']<call(n)['output_tokens']]
        ready_fg=[n for n in foreground if n['phase']=='decode'and n['generated']<call(n)['output_tokens']]
        if transport is not None:
            room=transport.available();assert room>=len(ready_fg)
            decoders=ready_fg+[n for n in decoders if n['background']][:max(0,room-len(ready_fg))]
        assert all(any(n is d for d in decoders)for n in ready_fg)
        if decoders:
            if transport is not None:transport.before(decoders,'decode')
            assert raw.token_to_kv_pool_allocator.available_size()>=len(decoders)
            batch=subset(decoders[0]['batch'],[0])
            for n in decoders[1:]:batch.merge_batch(subset(n['batch'],[0]))
            begin=time.perf_counter_ns();batch.input_ids=torch.tensor([n['token']for n in decoders],device=raw.device,dtype=torch.int64)
            batch.prepare_for_decode();bench._maybe_prepare_mlp_sync_batch(batch,raw)
            fb=bench.ForwardBatch.init_new(batch,raw,return_hidden_states_before_norm=False);result=raw.forward(fb);assert result.can_run_graph
            sampled=raw.sample(result.logits_output,fb)
            if transport is not None:transport.prefetch()
            sampled.cpu();visible=time.perf_counter_ns();decode_ms=(visible-begin)/1e6;steps+=1
            if transport is not None:transport.after('decode')
            for i,n in enumerate(decoders):n.update(batch=subset(batch,[i]),context=n['context']+1,generated=n['generated']+1)
        occupancy=transport.snapshot()if transport is not None else dict(local_active_tokens=capacity-raw.token_to_kv_pool_allocator.available_size(),remote_active_tokens=0,pending_remote_tokens=0)
        used=occupancy['local_active_tokens']+occupancy['remote_active_tokens'];assert used==sum(n['context']for n in nodes if n['phase']!='done')
        max_used=max(max_used,used);max_remote=max(max_remote,occupancy['remote_active_tokens']);recomputed+=redo_tokens
        emit('kv_events',dict(perf_ns=time.perf_counter_ns(),loop_start_ns=now,fg_relative_ms=(time.perf_counter_ns()-origin)/1e6,
            **occupancy,used_tokens=used,foreground_tokens=fg_used(),background_tokens=used-fg_used(),queued_background=len(pending),
            active_background=sum(n['background']and n['phase']!='done'for n in nodes),fg_budget=fg_budget(),
            foreground_decode_ready=len(ready_fg),foreground_decode_executed=len(ready_fg),foreground_memory_waits=0,
            prefill_ms=prefill_ms,decode_ms=decode_ms,background_recomputed_tokens=redo_tokens))
        for n in nodes:
            if n['background']:n['record']['generated_tokens']=n['generated']
            if n['phase']!='decode'or n['generated']!=call(n)['output_tokens']:continue
            c=call(n);end=time.perf_counter_ns();assert n['context']==c['input_tokens']+c['output_tokens']-1
            if n['background']:
                r=n['record'];r.update(status='completed',finished_ns=end,e2e_ms=(end-r['planned_ns'])/1e6,
                    queue_ms=(r['admitted_ns']-r['planned_ns'])/1e6,ttft_ms=(n['first_visible_ns']-r['planned_ns'])/1e6)
                emit('background',r);n['phase']='done';release(n)
            else:
                row=dict(agent=n['agent']['id'],group=n['agent']['group'],round=n['turn'],input_tokens=c['input_tokens'],output_tokens=n['generated'],
                    ready_ms=(n['ready_ns']-origin)/1e6,start_ms=(n['call_start_ns']-origin)/1e6,end_ms=(end-origin)/1e6,
                    ttft_ms=(n['first_visible_ns']-n['ready_ns'])/1e6,round_service_ms=(end-n['call_start_ns'])/1e6,
                    context_at_end=n['context'],pool_used_tokens=used,wait_after_ms=c['wait_after_ms'])
                completed_calls.append(row);emit('calls',row);print('FOREGROUND_CALL',n['agent']['id'],n['turn'],round(row['end_ms'],2),flush=True)
                n['turn']+=1
                if n['turn']==len(n['agent']['calls']):n.update(phase='done',finish_ns=end);release(n)
                else:n.update(phase='waiting',ready_ns=end+round(c['wait_after_ms']*1e6),batch=None)
        nodes=[n for n in nodes if not n['background']or n['phase']!='done']
        if fg_done_ns is None and all(n['phase']=='done'for n in foreground):
            fg_done_ns=time.perf_counter_ns();save(folder/'foreground_done.json',dict(done=True,perf_ns=fg_done_ns))
        if not selected and not decoders:time.sleep(.001)
    for r in records:
        if r['status']!='completed':
            r.update(status='unfinished',cutoff_ns=stop,censored_age_ms=max(0,(stop-r['planned_ns'])/1e6));emit('background',r)
    for n in nodes:
        if n['phase']!='done':assert n['background'];release(n)
    torch.cuda.synchronize()
    for f in files.values():f.close()
    physical_capacity=capacity if transport is None else cfg['cooperation']['offpath_physical_capacity_tokens']
    assert raw.token_to_kv_pool_allocator.available_size()==physical_capacity
    if transport is not None:transport.finish(folder)
    assert raw.req_to_token_pool.available_size()==raw.req_to_token_pool.size
    assert len(completed_calls)==sum(len(a['calls'])for a in agents)
    result=dict(status='INDEPENDENT_GPU_TRACE_COMPLETE',origin_perf_ns=origin,calls=completed_calls,forwards=steps,
        max_used_tokens=max_used,capacity_tokens=capacity,memory_queue_events=0,foreground_retractions=0,remote_tokens=max_remote,
        agents=[dict(id=n['agent']['id'],group=n['agent']['group'],arrival_offset_ms=0,
            first_start_ms=(n['first_start_ns']-origin)/1e6,finish_ms=(n['finish_ns']-origin)/1e6,
            service_wall_ms=(n['finish_ns']-origin)/1e6,calls=n['turn'],input_tokens=sum(c['input_tokens']for c in n['agent']['calls']),
            output_tokens=sum(c['output_tokens']for c in n['agent']['calls']),reserved_peak_tokens=n['agent']['reserved_peak_tokens'])for n in foreground],
        final_free_tokens=physical_capacity,logical_capacity_tokens=capacity,background_summary=dict(arrived=len(records),completed=sum(r['status']=='completed'for r in records),
            unfinished=sum(r['status']!='completed'for r in records),retractions=retractions,recomputed_tokens=recomputed,
            interval_seconds=bg['arrival_interval_seconds'],warmup_seconds=0,start_perf_ns=origin,cutoff_perf_ns=stop),scope=cfg['scope'])
    save(folder/'result.json',result);return result
