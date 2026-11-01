"""Persistent-KV four-round workload,with explicit pending output semantics."""
from array import array
import hashlib
import json
from pathlib import Path
import time
import torch
from experiment_common import base,save
from prefill_backend import Backend


def session(bench,wrapped,control,cfg,inputs,*,forced=None,check=False,profile=False):
    b=cfg['batch'];n=cfg['decode_forwards_per_turn'];raw=control.raw
    wrapped.clear();control.reset(b,profile);control.req_objects=[]
    control.check_prefill=False;control.prefill_checks=[]
    initial=[r['initial_ids'] for r in inputs['requests'][:b]]
    reqs=bench.prepare_synthetic_inputs_for_latency_test(b,cfg['initial_tokens'],initial)
    for req in reqs:req.sampling_params.ignore_eos=True;req.sampling_params.max_new_tokens=cfg['final_context']+4
    control.req_objects=reqs;control.batch=b
    history=[[] for _ in reqs];pending=None;rounds=[];steps=[];chunks=[];outputs=[];used=[];checkpoints={}
    replay=None if forced is None else torch.tensor(forced['decode_inputs'],device=raw.device,dtype=torch.int64)[:,:,:b]
    torch.cuda.synchronize();began=time.perf_counter_ns();unix=time.time_ns()
    for turn in range(cfg['turns']):
        suffix=initial if turn==0 else [[pending[i]]+inputs['requests'][i]['tool_ids'][turn-1] for i in range(b)]
        assert all(len(s)==cfg['input_tokens_by_turn'][turn] for s in suffix)
        prefix=len(history[0]);full=[h+s for h,s in zip(history,suffix)]
        start=time.perf_counter_ns();turn_uid=len(control.frames);before_migrations=len(control.migrations)
        for lo in range(0,len(suffix[0]),cfg['chunk_tokens']):
            size=min(cfg['chunk_tokens'],len(suffix[0])-lo);last=lo+size==len(suffix[0])
            control.context=dict(phase='prefill',turn=turn,chunk=lo//cfg['chunk_tokens'],step=-1)
            control.check_prefill=check and last
            uid=len(control.frames);t=time.perf_counter_ns()
            management=control.ensure_headroom(b*size)
            for req,ids in zip(reqs,full):
                a=prefix+lo;z=a+size
                if a:req.prefix_indices=raw.req_to_token_pool.req_to_token[req.kv.req_pool_idx,:a].to(req.prefix_indices.dtype)
                assert len(req.prefix_indices)==a
                req.origin_input_ids=array('q',ids[:z]);req.full_untruncated_fill_ids=req.origin_input_ids;req.set_extend_range(a,z)
            token,logits,batch=bench.extend(reqs,raw)
            chunks.append(dict(uid=uid,turn=turn,chunk=lo//cfg['chunk_tokens'],prefix=prefix+lo,query=size,
                management_cpu_ms=management,submission_cpu_ms=(time.perf_counter_ns()-t)/1e6,remote_active=control.remote_active))
        actual_first=token.cpu().tolist();first_visible=time.perf_counter_ns();control.check_prefill=False
        if check:checkpoints[f'prefill_{turn}']=logits.detach().cpu().clone();control.audit()
        # The reference's last pending output belongs to its emitted history.
        # Matching cases replay that history even if BF16 argmax differs.
        if forced is not None:assert suffix==forced['prefill_suffixes'][turn][:b]
        history=full;turn_outputs=[actual_first];turn_inputs=[]
        decode_start=time.perf_counter_ns()
        for step in range(n):
            uid=len(control.frames);control.context=dict(phase='decode',turn=turn,chunk=-1,step=step)
            t=time.perf_counter_ns();events=control.events();events[0].record()
            control.ensure_headroom(b)
            supplied=token if replay is None else replay[turn,step]
            fb=base.make_forward(bench,raw,batch,supplied)
            result=raw.forward(fb);assert result.can_run_graph and raw.decode_cuda_graph_runner.bs==b
            token=raw.sample(result.logits_output,fb)
            control.prefetch_next(step+1<n)
            events[1].record();actual=token.cpu().tolist();visible=time.perf_counter_ns()
            local,remote,_=control.budget()
            steps.append(dict(uid=uid,turn=turn,step=step,context=prefix+len(suffix[0])+step+1,
                wall_ms=(visible-t)/1e6,elapsed_ms=(visible-began)/1e6,local_active=local,remote_active=remote,
                pending_tokens=control.pending[-1]['tokens'] if control.pending is not None else 0,_events=events))
            turn_outputs.append(actual);turn_inputs.append(supplied)
            if check and step in [0,n-1]:
                assert torch.equal(fb.input_ids,supplied)
                if control.enabled:control.validate_decode()
                control.audit();checkpoints[f'decode_{turn}_{step}']=result.logits_output.next_token_logits.detach().cpu().clone()
        torch.cuda.synchronize();end=time.perf_counter_ns()
        actual_inputs=torch.stack(turn_inputs).cpu().tolist()
        if replay is None:assert actual_inputs==turn_outputs[:-1]
        else:assert actual_inputs==[r[:b] for r in forced['decode_inputs'][turn]]
        for i in range(b):history[i].extend(row[i] for row in actual_inputs)
        pending=turn_outputs[-1] if forced is None else forced['last_pending'][turn][:b]
        expected_context=prefix+len(suffix[0])+n
        assert batch.seq_lens_cpu.tolist()==[expected_context]*b
        rounds.append(dict(turn=turn,kind='initial' if turn==0 else 'append',prefix=prefix,query=len(suffix[0]),
            ttft_wall_ms=(first_visible-start)/1e6,decode_wall_ms=(end-decode_start)/1e6,
            round_wall_ms=(end-start)/1e6,input_tokens=b*len(suffix[0]),output_tokens=b*(n+1),
            decode_forward_tokens=b*n,decode_tokens_per_second=b*n*1e9/(end-decode_start),
            first_uid=turn_uid,last_uid=len(control.frames)-1,migrations=len(control.migrations)-before_migrations,
            local_active=control.budget()[0],remote_active=control.remote_active,
            actual_first_token=actual_first,pending_last_output=pending,prefill_suffixes=suffix))
        outputs.append(turn_outputs);used.append(actual_inputs)
        print('ROUND',cfg['label'],turn,expected_context,'R',control.remote_active,flush=True)
    torch.cuda.synchronize();ended=time.perf_counter_ns()
    assert all(len(h)==cfg['final_context'] for h in history)
    assert control.pending is None
    endpoint=control.audit()
    assert endpoint['logical_tokens']==b*cfg['final_context']
    control.release_requests(reqs);torch.cuda.synchronize();final=control.finish()
    assert final['logical_tokens']==0 and control.physical_allocator.available_size()==cfg['kv_capacity_tokens']
    for row in steps:
        a,z=row.pop('_events');row['main_stream_ms']=a.elapsed_time(z)
    for m in control.migrations:
        frame=control.frames[m['uid']];m.update(phase=frame['phase'],turn=frame['turn'],chunk=frame.get('chunk',-1),step=frame['step'])
    output_tokens=b*cfg['turns']*(n+1);wall=(ended-began)/1e6
    result=dict(status='PERSISTENT_MULTITURN_COMPLETE',label=cfg['label'],batch=b,wall_ms=wall,
        output_tokens=output_tokens,output_tokens_per_second=output_tokens*1000/wall,
        decode_forward_tokens=b*cfg['turns']*n,input_tokens=sum(r['input_tokens'] for r in rounds),
        external_input_tokens=sum(r['input_tokens'] for r in rounds)-b*(cfg['turns']-1),
        rounds=rounds,steps=steps,chunks=chunks,forwards=control.frames,migrations=control.migrations,
        initial_remote_tokens=0,endpoint=endpoint,final=final,start_unix_ns=unix,end_unix_ns=time.time_ns(),
        decode_inputs=used,sampled_outputs=outputs,last_pending=[r['pending_last_output'] for r in rounds],
        prefill_suffixes=[r['prefill_suffixes'] for r in rounds],
        attention_checks=control.layer_checks if check else [],prefill_checks=control.prefill_checks if check else [],
        primary_query_copies=control.validation,stage_depth=control.stage_depth,
        stage_buffer_bytes=control.stager.bytes if control.stager is not None else 0,
        source_request_ids=[r['source_id'] for r in inputs['requests'][:b]],
        history_sha256=[hashlib.sha256(str(h).encode()).hexdigest() for h in history],
        scope=cfg['scope'])
    return result,checkpoints


def collect(bench,wrapped,tokenizer,old,cfg,w,root):
    inputs=json.loads(Path(cfg['inputs_path']).read_text())
    actual_peer=old.peer
    control=Backend(old.raw,actual_peer,dict(cfg,validation=not cfg.get('reference_only',False)),old,cfg['timing_case'])
    from decode_oracle import install
    install(control)
    control.configure(dict(name='batch2048_fused_async',migration_tokens=2048,async_copy=True,copy_impl='fused',reserve_steps=0))
    control.stage_depth=cfg['stage_depth']
    control.prepare_graphs(cfg['batch'])
    if cfg.get('reference_only',False):
        result,values=session(bench,wrapped,control,cfg,inputs,check=True)
        save(root/'validation.json',result);torch.save(values,root/'checkpoints.pt')
        save(root/'case_collection.json',dict(status='MULTITURN_CASE_COMPLETE',reference_only=True,scope='Enlarged pool,numerical/token-history reference only;no performance claim.'))
        return
    reference=json.loads((root.parent/'native8_reference/validation.json').read_text())
    try:checked,values=session(bench,wrapped,control,cfg,inputs,forced=reference,check=True)
    except BaseException:
        save(root/'numerical_progress.json',dict(context=control.context,prefill_checks=control.prefill_checks,decode_checks=control.layer_checks))
        raise
    save(root/'validation.json',checked);torch.save(values,root/'checkpoints.pt')
    if cfg['timing_case']=='tiered_peer':
        assert any(m['phase']=='prefill' for m in checked['migrations'])
        assert any(m['phase']=='decode' for m in checked['migrations'])
        shadow=torch.empty_like(actual_peer)
        control.peer=shadow;control.read_peer=shadow
        control.validation=False;control.prepare_graphs(cfg['batch'])
        ordered,local_values=session(bench,wrapped,control,cfg,inputs,forced=reference,check=True)
        comparison=[dict(checkpoint=k,bitwise_equal=torch.equal(values[k],local_values[k]),
            max_abs_error=float((values[k]-local_values[k]).abs().max())) for k in values]
        save(root/'same_order_local_shadow.json',dict(checkpoints=comparison,
            matched_inputs=ordered['history_sha256']==checked['history_sha256'],matched_outputs=ordered['sampled_outputs']==checked['sampled_outputs'],
            extra_local_bytes=shadow.numel()*shadow.element_size(),scope='Independent whole trajectory with identical ownership/allocation and read order,far payload backed by extra local memory;numerical only.'))
        assert all(r['bitwise_equal'] for r in comparison)
        assert ordered['sampled_outputs']==checked['sampled_outputs']
        control.peer=actual_peer;control.read_peer=actual_peer
        control.validation=False;control.prepare_graphs(cfg['batch'])
        shadow.fill_(float('nan'));torch.cuda.synchronize();del shadow,local_values
    else:
        control.validation=False;control.prepare_graphs(cfg['batch'])
    session(bench,wrapped,control,cfg,inputs,forced=reference) # Full exact-trajectory warmup.
    plain,_=session(bench,wrapped,control,cfg,inputs,forced=reference)
    assert not plain['primary_query_copies'] and not plain['attention_checks'] and not plain['prefill_checks']
    save(root/'plain_0.json',plain)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profiler:
        diagnostic,_=session(bench,wrapped,control,cfg,inputs,forced=reference,profile=True)
    profiler.export_chrome_trace(str(root/'multiturn.trace.json'));save(root/'profile.json',diagnostic)
    assert plain['history_sha256']==diagnostic['history_sha256']==checked['history_sha256']
    save(root/'case_collection.json',dict(status='MULTITURN_CASE_COMPLETE',primary_trials=1,
        wall_ms=plain['wall_ms'],output_tokens_per_second=plain['output_tokens_per_second'],
        profiler_wall_ratio=diagnostic['wall_ms']/plain['wall_ms'],scope=cfg['scope']))
