import json
from pathlib import Path
import random
import time
import torch
from backend import TimingBackend,scope
from experiment_common import base,save


CHECKS={0,1,2,3,7,15,31,63,127,255,511,1023}


def prepare(bench,wrapped,tokenizer,control,cfg):
    # Identical native prefill is preparation only; real KV is rebuilt for
    # each trial. No saved full-KV mirror or fixed-context replay is used.
    control.enabled=False
    control.reset(cfg['batch'],False)
    batch,token,prompt=base.prepare(bench,wrapped,tokenizer,cfg['batch'],cfg['initial_context'],
        cfg['preparation_chunk_tokens'],cfg['decode_steps']+2)
    control.enabled=True;control.reset(cfg['batch'],False)
    control.req_objects=batch.reqs
    control.requests=[int(r.kv.req_pool_idx) for r in batch.reqs]
    control.lengths=batch.seq_lens_cpu.tolist()
    for rid,length in zip(control.requests,control.lengths):
        control.tier[rid,:length]=0
        control.slot[rid,:length].copy_(control.raw.req_to_token_pool.req_to_token[rid,:length])
        control.live[rid]=length
        control.residents.extend((rid,pos) for pos in range(length))
        control.created.update({(rid,pos):-1 for pos in range(length)})
    control.enable_decode();torch.cuda.synchronize()
    return batch,token,prompt


def session(bench,wrapped,tokenizer,control,cfg,profile=False,checkpoints=False,prepared=None):
    batch,token,prompt=prepared if prepared is not None else prepare(bench,wrapped,tokenizer,control,cfg)
    control.profile=profile;rows=[];saved={};outputs=[]
    initial_token=token.cpu().tolist();start=time.perf_counter_ns();last_visible=start
    for step in range(cfg['decode_steps']):
        entered=time.perf_counter_ns();events=control.events();events[0].record()
        before=len(control.migrations)
        control.context=dict(phase='decode',turn=0,step=step,remaining_decode_steps=cfg['decode_steps']-step)
        control.ensure_headroom(cfg['batch'])
        forward=base.make_forward(bench,control.raw,batch,token)
        result=control.raw.forward(forward)
        token=control.raw.sample(result.logits_output,forward)
        if checkpoints and step in CHECKS:saved[step]=result.logits_output.next_token_logits.detach().cpu().clone()
        # Launch while this forward may still run, but leave ownership/local
        # slots unchanged until the next step. The current token's D2H does
        # not wait for copy_done merely to report that token to the caller.
        control.prefetch_next(step+1<cfg['decode_steps'])
        events[1].record()
        actual=token.cpu().tolist();visible=time.perf_counter_ns();outputs.append(actual)
        local,remote,_=control.budget()
        assert remote*10000<local*599
        row=dict(step=step,token_interval_ms=(visible-last_visible)/1e6,
            step_wall_ms=(visible-entered)/1e6,visible_unix_ns=time.time_ns(),
            local_active=local,remote_active=remote,ratio=remote/local,
            reserved_peer=control.cfg['remote_capacity_tokens']-len(control.peer_free),
            migration_committed=len(control.migrations)>before,
            migration_launched_for_next=control.launched is not None,
            context_lengths=batch.seq_lens_cpu.tolist(),_events=events)
        rows.append(row);last_visible=visible
    total=(last_visible-start)/1e6
    final=control.finish()
    for row in rows:
        a,z=row.pop('_events');row['step_gpu_stream_ms']=a.elapsed_time(z)
        row['model_event_ms']=control.frames[row['step']]['model_event_ms']
        row['custom_page_gpu_ms']=control.frames[row['step']]['custom_page_gpu_ms']
    assert final['logical_tokens']==cfg['batch']*(cfg['initial_context']+cfg['decode_steps'])
    return dict(session_wall_ms=total,steps=rows,migrations=control.migrations,forwards=control.frames,
        final=final,initial_token=initial_token,outputs=outputs,context=cfg['initial_context'],
        prompt_ids=prompt,profile=profile,checkpoints=checkpoints),saved


def collect(bench,wrapped,tokenizer,old_control,cfg,w,root):
    control=TimingBackend(old_control.raw,old_control.peer,cfg,old_control,'tiered_peer')
    if cfg['operation']=='golden':
        control.configure(dict(name='all_local_golden',copy_impl='fused',reserve_steps=0,async_copy=False,migration_tokens=0))
        result,checks=session(bench,wrapped,tokenizer,control,cfg,checkpoints=True)
        assert not result['migrations'] and result['final']['remote_active_tokens']==0
        save(root/'reference.json',dict(initial_token=result['initial_token'],outputs=result['outputs'],
            prompt_ids=result['prompt_ids'],initial_context=cfg['initial_context'],batch=cfg['batch'],
            equal_work_local_capacity=cfg['kv_capacity_tokens'],checks=sorted(checks)))
        torch.save(checks,root/'reference_logits.pt')
        save(root/'stage_collection.json',dict(status='DYNAMIC_STAGE_COMPLETE',operation='golden',config=cfg))
        return
    reference_root=Path(cfg['reference_root'])
    reference=json.loads((reference_root/'reference.json').read_text())
    expected=torch.load(reference_root/'reference_logits.pt',map_location='cpu',weights_only=True)
    assert reference['initial_context']==cfg['initial_context'] and reference['batch']==cfg['batch']
    from fused_copy import validate
    save(root/'fused_copy_checks.json',validate(control))
    variants=[dict(name=f'{mode}_m{amount}',migration_tokens=amount,async_copy=(mode=='async'),
        copy_impl='fused',reserve_steps=0) for amount in cfg['migration_amounts'] for mode in ['sync','async']]
    for variant in variants:
        out=root/variant['name'];out.mkdir();control.configure(variant)
        result,actual=session(bench,wrapped,tokenizer,control,cfg,checkpoints=True)
        assert result['initial_token']==reference['initial_token']
        assert result['outputs']==reference['outputs'][:cfg['decode_steps']]
        for step,value in actual.items():assert torch.equal(value,expected[step]),(variant,step,float((value-expected[step]).abs().max()))
        save(out/'validation.json',dict(checkpoints=sorted(actual),all_logits_bitwise_equal=True,
            generated_tokens_matching=cfg['decode_steps']*cfg['batch'],final=result['final'],migrations=result['migrations']))
        print('MIGRATION_VALIDATED',variant['name'],len(actual),flush=True)
    # Validation above also compiles and warms every kernel specialization.
    # Pair modes closely, randomizing pair order and sync/async order per trial.
    rng=random.Random(cfg['random_seed']);orders=[];completed=[]
    for trial in range(cfg['plain_trials']):
        amounts=list(cfg['migration_amounts']);rng.shuffle(amounts);order=[]
        for amount in amounts:
            pair=[v for v in variants if v['migration_tokens']==amount];rng.shuffle(pair)
            for variant in pair:
                name=variant['name'];control.configure(variant)
                result,_=session(bench,wrapped,tokenizer,control,cfg)
                assert result['initial_token']==reference['initial_token']
                assert result['outputs']==reference['outputs'][:cfg['decode_steps']],(name,trial,'greedy mismatch')
                result.update(variant=variant,trial=trial)
                save(root/name/f'plain_{trial}.json',result)
                order.append(name);completed.append(dict(name=name,trial=trial,session_wall_ms=result['session_wall_ms']))
                save(root/'progress.json',completed)
                print('MIGRATION_PLAIN',name,trial,result['session_wall_ms'],len(result['migrations']),flush=True)
        orders.append(order)
    for variant in variants:
        if variant['migration_tokens'] not in cfg.get('profile_amounts',[]):continue
        control.configure(variant)
        prepared=prepare(bench,wrapped,tokenizer,control,cfg)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profiler:
            result,_=session(bench,wrapped,tokenizer,control,cfg,profile=True,prepared=prepared)
        # Scope/profile starts after fresh native prefill and Graph preparation.
        profiler.export_chrome_trace(str(root/variant['name']/'trace.json'))
        save(root/variant['name']/'profile.json',result)
    save(root/'stage_collection.json',dict(status='DYNAMIC_STAGE_COMPLETE',config=cfg,
        variants=variants,paired_orders=orders,completed=completed,
        timing='CPU-visible generated token every step; ordinary inter-token time includes allocator, migration commit, model, sample, token D2H and harness gaps. Async launch overhead is counted on the preceding step. Also inspect the two-step sum, copy events, residual fence and ordinary-step slowdown. Fresh native prefill per trial is outside decode measurement. No checkpoint copies/profiler in primary runs.',
        policy='Pressure-triggered min(requested migration pages, admissible current-active ratio budget, peer free pages); strict10000*R <599*L checked on every commit and step. Ratio uses active local KV, not pool size; no future input oracle. Async preserves old ownership until next allocation and joins after copy_done and preceding readers. Same amounts/layout between paired sync and async runs.',
        kernels='Attention and KV writer frozen at validated r5/dynamic-combinations release; fused page copy only. Donor idle.'))
