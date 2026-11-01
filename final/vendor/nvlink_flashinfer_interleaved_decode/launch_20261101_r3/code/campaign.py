from array import array
import hashlib
import math
from pathlib import Path
import time
import torch
from backend import Backend,scope
from experiment_common import base,save


def initialize(bench,wrapped,tokenizer,control,cfg):
    raw=control.raw;b=cfg['batch'];wrapped.clear();control.reset(b);control.req_objects=[]
    checking=control.validation;control.validation=False  # Decode checks do not require prefill oracles.
    material=tokenizer.encode('A controlled serving experiment measures attention, KV growth and request latency. ',add_special_tokens=False)
    full=(material*math.ceil(cfg['initial_context']/len(material)))[:cfg['initial_context']]
    prefix=cfg['setup_context'];reqs=bench.prepare_synthetic_inputs_for_latency_test(b,prefix,[full[:prefix] for _ in range(b)])
    for req in reqs:
        req.sampling_params.ignore_eos=True;req.sampling_params.max_new_tokens=cfg['initial_context']-prefix+cfg['decode_steps']+3
    # Simultaneous requests:each native prefill forward advances every request
    # by the same chunk. FIFO victim order is the real KV creation order.
    control.req_objects=list(reqs);control.batch=b
    for lo in range(0,prefix,cfg['preparation_chunk_tokens']):
        hi=min(prefix,lo+cfg['preparation_chunk_tokens']);control.context=dict(phase='setup_prefill',step=-1)
        for req in reqs:
            if lo:req.prefix_indices=raw.req_to_token_pool.req_to_token[req.kv.req_pool_idx,:lo].to(req.prefix_indices.dtype)
            req.origin_input_ids=array('q',full[:hi]);req.full_untruncated_fill_ids=req.origin_input_ids;req.set_extend_range(lo,hi)
        token,logits,merged=bench.extend(reqs,raw)
    for position in range(prefix,cfg['initial_context']):
        control.context=dict(phase='setup_decode',step=-1);control.ensure_headroom(b)
        inputs=torch.full((b,),full[position],device=raw.device,dtype=torch.int64)
        fb=base.make_forward(bench,raw,merged,inputs)
        if checking and position in [prefix,cfg['initial_context']-1]:assert torch.equal(fb.input_ids,inputs)
        raw.forward(fb)
    torch.cuda.synchronize()
    assert merged.seq_lens_cpu.tolist()==[cfg['initial_context']]*b
    assert not control.pending
    audit=control.audit()
    assert audit['remote_active_tokens']==0 and not control.migrations,'Every trajectory must start all-local'
    control.validation=checking
    return merged,reqs,audit,hashlib.sha256(str(full).encode()).hexdigest()


def trajectory(bench,wrapped,tokenizer,control,cfg,*,forced=None,validate=False,profile=False,prepared=None):
    batch,reqs,audit,prompt_hash=prepared or initialize(bench,wrapped,tokenizer,control,cfg)
    raw=control.raw;b=cfg['batch'];n=cfg['decode_steps'];start_uid=len(control.frames)
    # Controlled first decode input;the native trajectory supplies subsequent
    # matched inputs. All measured forwards really append KV and sample output.
    if forced is None:
        first=tokenizer.encode(' The',add_special_tokens=False)[0]
        token=torch.full((b,),first,device=raw.device,dtype=torch.int64)
    else:
        token=forced[0]
    values={};inputs=[];outputs=[];steps=[];events=[];prediction_checks=[];control.profile=profile
    torch.cuda.synchronize();whole_start=time.perf_counter_ns();unix_start=time.time_ns()
    for step in range(n):
        uid=len(control.frames);control.context=dict(phase='decode',step=step)
        with scope(f'DECODE_DETAIL_STEP_{step}',profile):
            began=time.perf_counter_ns();gpu=control.events();gpu[0].record();cpu={}
            def timed(name,call):
                t=time.perf_counter_ns()
                with scope(f'DECODE_DETAIL_{name}_{step}',profile):result=call()
                cpu[name.lower()+'_cpu_ms']=(time.perf_counter_ns()-t)/1e6
                return result
            timed('HEADROOM',lambda:control.ensure_headroom(b))
            used=token if forced is None else forced[step]
            fb=timed('SCHEDULE',lambda:base.make_forward(bench,raw,batch,used))
            if validate:assert torch.equal(fb.input_ids,used),'Prepared model input differs from requested token'
            result=timed('FORWARD_CALL',lambda:raw.forward(fb))
            assert result.can_run_graph and raw.decode_cuda_graph_runner.bs==b
            sampled=timed('SAMPLING',lambda:raw.sample(result.logits_output,fb))
            timed('PREFETCH',lambda:control.prefetch_next(step+1<n))
            gpu[1].record()
            actual=timed('VISIBILITY',lambda:sampled.cpu().tolist());visible=time.perf_counter_ns()
            wall=(visible-began)/1e6;events.append(gpu)
            outputs.append(actual)
            # Avoid per-step host copies of input tensors in the primary path.
            inputs.append(fb.input_ids);token=sampled
            local,remote,_=control.budget()
            steps.append(dict(step=step,uid=uid,context=cfg['initial_context']+step+1,
                wall_ms=wall,elapsed_ms=(visible-whole_start)/1e6,**cpu,
                cpu_residual_ms=wall-sum(cpu.values()),local_active=local,remote_active=remote,
                pending_tokens=control.pending[-1]['tokens'] if control.pending is not None else 0))
        if validate and step in cfg['validation_steps_zero_based']:
            assert torch.isfinite(result.logits_output.next_token_logits).all().item()
            values[str(step)]=result.logits_output.next_token_logits.detach().cpu().clone()
            if control.enabled:control.validate_decode()
            control.audit()
        if (validate or profile) and step==19:
            top=result.logits_output.next_token_logits.topk(2,dim=-1)
            prediction_checks.append(dict(step=step,top_ids=top.indices.cpu().tolist(),top_values=top.values.cpu().tolist()))
    torch.cuda.synchronize();whole_end=time.perf_counter_ns();unix_end=time.time_ns();control.profile=False
    assert batch.seq_lens_cpu.tolist()==[cfg['final_context']]*b
    endpoint=control.audit();layout=[dict(request=int(r.kv.req_pool_idx),remote=control.remote_counts.get(int(r.kv.req_pool_idx),0)) for r in reqs]
    raw_inputs=torch.stack(inputs).cpu().tolist()
    if forced is not None:assert raw_inputs==forced.cpu().tolist(),'Actual model token history is not matched'
    else:assert raw_inputs[1:]==outputs[:-1]
    control.release_requests(reqs);torch.cuda.synchronize();final=control.finish()
    assert final['logical_tokens']==0 and control.physical_allocator.available_size()==cfg['kv_capacity_tokens']
    for row,(a,z) in zip(steps,events):row['main_stream_pre_visibility_ms']=a.elapsed_time(z)
    frames=[r for r in control.frames if r['phase']=='decode']
    assert len(frames)==n and not control.pending
    for row,frame in zip(steps,frames):
        row['other_main_stream_ms']=row['main_stream_pre_visibility_ms']-frame['model_event_ms']-frame['custom_page_gpu_ms']
        assert row['other_main_stream_ms']>=-.02
    result=dict(status='DECODE_TRAJECTORY_COMPLETE',label=cfg['label'],batch=b,steps=steps,
        decode_wall_ms=(whole_end-whole_start)/1e6,generated_tokens=b*n,
        decode_tokens_per_second=b*n/((whole_end-whole_start)/1e9),
        start_unix_ns=unix_start,end_unix_ns=unix_end,measurement_start_uid=start_uid,
        initial_audit=audit,endpoint_audit=endpoint,remote_per_request_at_end=layout,prompt_sha256=prompt_hash,
        forwards=frames,migrations=control.migrations,final=final,
        consumed_ids=raw_inputs,sampled_ids=outputs,validation_checks=control.layer_checks if validate else [],
        prediction_checkpoints=prediction_checks,
        primary_query_copies=control.validation,
        semantics=f"One{n}-step real decode trajectory;KV grows{cfg['initial_context']}->{cfg['final_context']}. All requests prefill together in native chunks;remote starts0;only local exhaustion triggers production FIFO/2048/async migration. Prefill,capture,oracle,warmup,release outside timer. Same controlled token history. Step starts before pending commit and ends at visible output. CPU/GPU budgets are separate.")
    return result,values


def collect(bench,wrapped,tokenizer,old,cfg,w,root):
    import inspect
    (root/'runtime_prepare_for_decode.py').write_text(inspect.getsource(bench.ScheduleBatch.prepare_for_decode))
    (root/'runtime_make_forward.py').write_text(inspect.getsource(base.make_forward))
    control=Backend(old.raw,old.peer,dict(cfg,validation=True),old,cfg['timing_case'])
    control.configure(dict(name='batch2048_fused_async',migration_tokens=2048,async_copy=True,copy_impl='fused',reserve_steps=0))
    if control.enabled:
        from flashinfer_peer import module
        save(root/'backend.json',dict(implementation='FlashInfer0.6.18 FA2 with tiered addresses',jit=module()[1],
            policy=cfg['policy'],manual_split_used=False))
    control.prepare_graphs(cfg['batch'])
    forced=None
    if cfg['label']!=cfg['native_label']:
        native=__import__('json').loads((Path(cfg['reference_root'])/'validation.json').read_text())
        ids=native['consumed_ids']
        if cfg['batch']==cfg['expanded_batch']:ids=[row+[row[0]] for row in ids]
        forced=torch.tensor(ids,device=old.raw.device,dtype=torch.int64)
    reuse=Path(cfg['reuse_peer_validation_root'])
    checked=__import__('json').loads((reuse/'validation.json').read_text())
    values=torch.load(reuse/'checkpoints.pt',weights_only=True,map_location='cpu')
    assert checked['consumed_ids']==forced.cpu().tolist()
    assert len(checked['validation_checks'])==288
    save(root/'validation_reuse.json',dict(source=str(reuse),source_job_id='26074678.pbs101',
        scope='Reuse completed peer trajectory:288same-query native attention checks,exact GPU index-order audits,776steps. Pointer selection defaults to the same actual peer source;counter/packing kernel sources unchanged. No repeated peer validation trajectory.'))
    save(root/'validation.json',checked);torch.save(values,root/'checkpoints.pt')
    if cfg['label'] in cfg['reference_cases']:
        expected=torch.load(Path(cfg['reference_measurement_root'])/cfg['reference_cases'][cfg['label']]/'checkpoints.pt',weights_only=True,map_location='cpu')
        checks=[]
        for name,value in values.items():
            checks.append(dict(step=int(name),original_tolerance_pass=bool(torch.isclose(value,expected[name],atol=cfg['reference_atol'],rtol=cfg['reference_rtol']).all()),
                bitwise_equal=torch.equal(value,expected[name]),max_abs_error=float((value-expected[name]).abs().max())))
        save(root/'native_checkpoint_checks.json',checks)
        # Retain this independent diagnostic unchanged. Permuting BF16 FA2
        # tiles changes intermediate probability rounding and sum order;
        # original-order whole-model logits are not assumed bitwise invariant.
    if cfg.get('validation_only',False):
        save(root/'case_collection.json',dict(status='DECODE_DETAIL_CASE_COMPLETE',validation_only=True,
            scope='Enlarged all-local pool is an equal-work numerical reference only;no timing baseline or original-pool admission claim.'))
        return
    if forced is None:forced=torch.tensor(checked['consumed_ids'],device=old.raw.device,dtype=torch.int64)
    native_reference=__import__('json').loads((Path(cfg['reference_measurement_root'])/'native8_ref/validation.json').read_text())
    assert checked['sampled_ids']==native_reference['sampled_ids'],'Peer greedy tokens differ from native reference'
    # Full-model numerical reference at IDENTICAL read order. This temporary
    # local far-payload shadow exists only during reference validation;it is
    # poisoned and destroyed before warmup or performance measurement.
    shadow=torch.empty_like(control.peer);shadow.fill_(float('nan'))
    control.read_peer=shadow;control.numerical_shadow=shadow
    control.validation=True;control.prepare_graphs(cfg['batch'])
    ordered,ordered_values=trajectory(bench,wrapped,tokenizer,control,cfg,forced=forced,validate=True)
    same_order=[]
    for name,value in values.items():
        same_order.append(dict(step=int(name),bitwise_equal=torch.equal(value,ordered_values[name]),
            max_abs_error=float((value-ordered_values[name]).abs().max())))
    save(root/'same_order_local_shadow_checks.json',dict(checkpoints=same_order,
        matched_actual_inputs=ordered['consumed_ids']==checked['consumed_ids'],
        matched_sampled_tokens=ordered['sampled_ids']==checked['sampled_ids'],
        scope='Numerical reference only;extra local memory. Same realQwen full forward,KV growth and index order. Explicit actual-IPC migration payload copied into a local shadow after ownership commit. No performance baseline or capacity claim.'))
    save(root/'local_shadow_validation.json',ordered);torch.save(ordered_values,root/'local_shadow_checkpoints.pt')
    assert all(v['bitwise_equal'] for v in same_order),same_order
    assert ordered['consumed_ids']==checked['consumed_ids'] and ordered['sampled_ids']==checked['sampled_ids']
    control.read_peer=control.peer;control.numerical_shadow=None
    control.validation=False;control.prepare_graphs(cfg['batch'])
    shadow.fill_(float('nan'));torch.cuda.synchronize();del shadow,ordered_values
    trajectory(bench,wrapped,tokenizer,control,cfg,forced=forced)  # Full exact-point warmup.
    plain,_=trajectory(bench,wrapped,tokenizer,control,cfg,forced=forced)
    assert not plain['validation_checks'] and not plain['primary_query_copies']
    save(root/'plain_0.json',plain)
    print('DECODE_DETAIL_PLAIN',cfg['label'],plain['decode_wall_ms'],plain['decode_tokens_per_second'],flush=True)
    prepared=initialize(bench,wrapped,tokenizer,control,cfg)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,with_stack=False,profile_memory=False) as profiler:
        profiled,_=trajectory(bench,wrapped,tokenizer,control,cfg,forced=forced,profile=True,prepared=prepared)
    profiler.export_chrome_trace(str(root/'decode.trace.json'));save(root/'profile.json',profiled)
    assert profiled['consumed_ids']==plain['consumed_ids']
    matches=sum(a==b for x,y in zip(profiled['sampled_ids'],plain['sampled_ids']) for a,b in zip(x,y))
    save(root/'profile_pair_checks.json',dict(matched_actual_inputs=True,matched_prompt=True,
        greedy_matches=matches,predictions=cfg['batch']*cfg['decode_steps'],greedy_match_fraction=matches/(cfg['batch']*cfg['decode_steps']),
        mismatched_steps=[i for i,(a,b) in enumerate(zip(profiled['sampled_ids'],plain['sampled_ids'])) if a!=b],
        note='Paired timing requires actual identical consumed token inputs. BF16 argmax diagnostics recorded independently. Unchanged attention math tolerance and exact migration payload checks remain enforced in validation.'))
    save(root/'case_collection.json',dict(status='DECODE_DETAIL_CASE_COMPLETE',config=cfg,
        primary_trials=1,decode_wall_ms=plain['decode_wall_ms'],decode_tokens_per_second=plain['decode_tokens_per_second'],
        profiled_wall_ratio=profiled['decode_wall_ms']/plain['decode_wall_ms'],
        scope='One primary trajectory and one independent profiler diagnostic;no3trial averaging. FlashInfer native planner/math/merge;no manual split. Initial remote0;unchanged1+2+4/2048 migration policy. Batched native chunk-prefill is setup only.'))
