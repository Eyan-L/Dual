"""Finite saturated workload with actual prefill, decode and KV reclamation.

Sampled mode feeds each request's previous model token back autoregressively.
Fixed generation length is known to admission identically in all modes.
"""
from array import array
import copy
import time
import torch
from experiment_common import base
from backend import scope


def subset(batch,indices):
    work=copy.copy(batch);work.sampling_info=copy.copy(batch.sampling_info)
    work.sampling_info.penalizer_orchestrator=copy.copy(batch.sampling_info.penalizer_orchestrator)
    assert not work.sampling_info.penalizer_orchestrator.is_required
    work.filter_batch(keep_indices=indices)
    return work


def replay(bench,wrapped,control,cfg,records,*,profile=False,max_forwards=None):
    raw=control.raw;wrapped.clear();control.reset(1,profile);control.req_objects=[]
    pending=list(records);active={};completed=[];running=None;steps=[]
    total_output=sum(r['output_tokens'] for r in records);token_count=0;prefill_work=0
    c=cfg['kv_capacity_tokens'];remote_extra=min(cfg['remote_capacity_tokens'],(599*c-1)//10000) if control.case=='tiered_peer' else 0
    capacity=c+remote_extra;maximum_running=0;maximum_active=0
    torch.cuda.synchronize();start_ns=time.perf_counter_ns();start_unix_ns=time.time_ns()

    def next_token(node):
        return node['samples'][-1] if cfg.get('decode_source')=='sampled' else node['record']['output_ids'][node['generated']-1]

    def complete(node,visible):
        nonlocal token_count
        times=node['token_times_ns'];record=node['record']
        assert len(times)==record['output_tokens']
        completed.append(dict(id=record['id'],instance=node['instance'],case=record.get('case','smoke'),
            input_tokens=record['input_tokens'],output_tokens=record['output_tokens'],
            arrival_ms=0.,admitted_ms=(node['admitted_ns']-start_ns)/1e6,
            queue_ms=(node['admitted_ns']-start_ns)/1e6,
            ttft_ms=(times[0]-start_ns)/1e6,service_ttft_ms=(times[0]-node['admitted_ns'])/1e6,
            e2e_ms=(visible-start_ns)/1e6,service_e2e_ms=(visible-node['admitted_ns'])/1e6,
            token_visible_ms=[(v-start_ns)/1e6 for v in times],sampled_tokens=node['samples']))

    while pending or active:
        # First-fit scans the fixed FIFO backlog; both modes see the same order.
        # Admissions depend on token capacity, never a hard-coded target batch.
        reserved=sum(n['record']['peak_kv_tokens'] for n in active.values())
        remaining=[]
        for record in pending:
            if reserved+record['peak_kv_tokens']<=capacity:
                req=bench.prepare_synthetic_inputs_for_latency_test(1,record['input_tokens'],[record['prompt_ids']])[0]
                instance=len(completed)+len(active)
                req.rid=f'replay_{instance}_{record["id"]}'
                req.sampling_params.ignore_eos=True;req.sampling_params.max_new_tokens=record['output_tokens']
                active[id(req)]=dict(req=req,record=record,prefill=0,generated=0,token_times_ns=[],samples=[],
                    admitted_ns=time.perf_counter_ns(),instance=instance)
                reserved+=record['peak_kv_tokens']
            else:remaining.append(record)
        pending=remaining;maximum_active=max(maximum_active,len(active))
        assert len(active)<=cfg['max_batch'],('capacity-derived graph bound exceeded',len(active),cfg['max_batch'])
        progress=False

        # One bounded prefill forward, then one decode forward per iteration.
        # Existing decodes continue while newly admitted calls fill their KV.
        filling=[n for n in active.values() if n['prefill']<n['record']['input_tokens']]
        if filling:
            control.context=dict(phase='prefill',turn=-1,chunk=len(steps),step=-1)
            if control.pending is not None:control.commit()
            budget=min(cfg['prefill_token_budget'],control.allocation_budget())
            selected=[];lengths=[]
            for node in filling:
                length=min(cfg['prefill_chunk_tokens'],node['record']['input_tokens']-node['prefill'],budget)
                if length<=0:break
                selected.append(node);lengths.append(length);budget-=length
            if selected:
                entered=time.perf_counter_ns();control.ensure_headroom(sum(lengths))
                requests=[n['req'] for n in selected];control.req_objects=requests
                for node,length in zip(selected,lengths):
                    req=node['req'];old=node['prefill'];end=old+length
                    if old:req.prefix_indices=raw.req_to_token_pool.req_to_token[req.kv.req_pool_idx,:old].to(req.prefix_indices.dtype)
                    req.origin_input_ids=array('q',node['record']['prompt_ids'][:end])
                    req.full_untruncated_fill_ids=req.origin_input_ids;req.set_extend_range(old,end)
                with scope(f'SLACK_COST_SCHEDULE_{len(control.frames)}',profile):sampled,logits,batch=bench.extend(requests,raw)
                samples=sampled.cpu().tolist();visible=time.perf_counter_ns();prefill_work+=sum(lengths)
                model_local,model_remote,_=control.budget()
                newly_running=[];finished=[]
                for i,(node,length) in enumerate(zip(selected,lengths)):
                    node['prefill']+=length
                    if node['prefill']==node['record']['input_tokens']:
                        node['generated']=1;node['token_times_ns'].append(visible);node['samples'].append(samples[i]);token_count+=1
                        if node['generated']==node['record']['output_tokens']:finished.append(node)
                        else:newly_running.append(i)
                if newly_running:
                    batch.filter_batch(keep_indices=newly_running)
                    batch.input_ids=torch.tensor([next_token(active[id(req)]) for req in batch.reqs],device=raw.device,dtype=torch.int64)
                    if running is None:running=batch
                    else:running.merge_batch(batch)
                if finished:
                    for node in finished:complete(node,visible)
                    control.release_requests([n['req'] for n in finished])
                    for node in finished:active.pop(id(node['req']))
                local,remote,_=control.budget()
                steps.append(dict(uid=len(control.frames)-1,phase='prefill',batch=len(selected),input_tokens=sum(lengths),
                    output_tokens=len(newly_running)+len(finished),elapsed_ms=(visible-start_ns)/1e6,
                    operation_ms=(visible-entered)/1e6,local_active=model_local,remote_active=model_remote,
                    local_after_release=local,remote_after_release=remote,
                    running=0 if running is None else len(running.reqs),active=len(active),queued=len(pending),
                    active_reserved_tokens=sum(n['record']['peak_kv_tokens'] for n in active.values())))
                progress=True

        if max_forwards is not None and len(steps)>=max_forwards:break
        if running is not None:
            control.context=dict(phase='decode',turn=-1,chunk=-1,step=len(steps))
            if control.pending is not None:control.commit()
            available=control.allocation_budget()
            count=min(len(running.reqs),available)
            if count:
                indices=list(range(count))
                work=running if count==len(running.reqs) else subset(running,indices)
                nodes=[active[id(req)] for req in work.reqs];control.req_objects=list(work.reqs)
                entered=time.perf_counter_ns();control.ensure_headroom(count)
                tokens=torch.tensor([next_token(n) for n in nodes],device=raw.device,dtype=torch.int64)
                with scope(f'SLACK_COST_DECODE_SCHEDULE_{len(control.frames)}',profile):forward=base.make_forward(bench,raw,work,tokens)
                result=raw.forward(forward);sampled=raw.sample(result.logits_output,forward)
                finishing=any(n['generated']+1==n['record']['output_tokens'] for n in nodes)
                control.batch=count;control.prefetch_next(not finishing)
                samples=sampled.cpu().tolist();visible=time.perf_counter_ns()
                model_local,model_remote,_=control.budget()
                if control.validation:
                    control.validate_decode();control.audit()
                if work is not running:
                    index=torch.tensor(indices,device=raw.device,dtype=torch.int64)
                    running.seq_lens.index_copy_(0,index,work.seq_lens)
                    running.orig_seq_lens.index_copy_(0,index,work.orig_seq_lens)
                    running.seq_lens_cpu[indices]=work.seq_lens_cpu;running.seq_lens_sum=None
                finished=[]
                for node,sample in zip(nodes,samples):
                    node['generated']+=1;node['token_times_ns'].append(visible);node['samples'].append(sample);token_count+=1
                    if node['generated']==node['record']['output_tokens']:finished.append(node)
                if finished:
                    gone={id(n['req']) for n in finished}
                    running.filter_batch(keep_indices=[i for i,r in enumerate(running.reqs) if id(r) not in gone])
                    if not running.reqs:running=None
                    for node in finished:complete(node,visible)
                    control.release_requests([n['req'] for n in finished])
                    for node in finished:active.pop(id(node['req']))
                local,remote,_=control.budget()
                steps.append(dict(uid=len(control.frames)-1,phase='decode',batch=count,input_tokens=count,output_tokens=count,
                    elapsed_ms=(visible-start_ns)/1e6,operation_ms=(visible-entered)/1e6,local_active=model_local,remote_active=model_remote,
                    local_after_release=local,remote_after_release=remote,
                    running=0 if running is None else len(running.reqs),active=len(active),queued=len(pending),
                    active_reserved_tokens=sum(n['record']['peak_kv_tokens'] for n in active.values())))
                maximum_running=max(maximum_running,count);progress=True
        if control.validation:control.audit()
        if max_forwards is not None and len(steps)>=max_forwards:break
        assert progress or not (pending or active),('replay made no progress',len(pending),len(active),control.budget(),capacity)

    incomplete=[dict(id=n['record']['id'],prefill=n['prefill'],generated=n['generated']) for n in active.values()]
    # Partial diagnostic runs explicitly release unfinished requests after the
    # window; never count those requests as completed or zero latency.
    if active:
        control.release_requests([n['req'] for n in active.values()]);active.clear()
    torch.cuda.synchronize();end_ns=time.perf_counter_ns();end_unix_ns=time.time_ns()
    final=control.finish()
    assert final['logical_tokens']==0 and control.physical_allocator.available_size()==c
    assert raw.req_to_token_pool.available_size()==raw.req_to_token_pool.size
    if max_forwards is None:
        assert not pending and not incomplete and len(completed)==len(records)
        assert token_count==total_output
        assert prefill_work==sum(r['input_tokens'] for r in records)
    return dict(status='REPLAY_COMPLETE' if not (pending or incomplete) else 'DIAGNOSTIC_WINDOW_COMPLETE',
        wall_ms=(end_ns-start_ns)/1e6,start_unix_ns=start_unix_ns,end_unix_ns=end_unix_ns,
        replayed_output_tokens=token_count,actual_prefill_tokens=prefill_work,completed_requests=len(completed),
        output_tokens_per_second=token_count/((end_ns-start_ns)/1e9),completed_requests_per_second=len(completed)/((end_ns-start_ns)/1e9),
        maximum_running=maximum_running,maximum_active=maximum_active,request_order=[r['id'] for r in records],
        requests=completed,steps=steps,forwards=control.frames,migrations=control.migrations,restorations=control.restorations,
        releases=control.releases,attention_checks=control.layer_checks,final=final,
        unfinished=incomplete,unadmitted=len(pending),
        decode_source=cfg.get('decode_source','teacher_forced'),
        semantics='Saturated finite backlog; actual model forward/sampling/KV growth. '+
            ('Greedy autoregressive sampled tokens, fixed generation length, EOS ignored. ' if cfg.get('decode_source')=='sampled' else 'Teacher-forced validation tokens. ')+
            'Entire prefill/decode/admission/migration/release/ratio-repair wall included. Tokenization,model load,graph capture,warmup,validation/profiler excluded from primary timing. No tool/DAG wait or task-success claim.')
