import json
from pathlib import Path
import random
import statistics
import torch
from backend import TimingBackend
from campaign import session
from experiment_common import save

CHOICES=[dict(name='r5',block_m=32,splits=1),dict(name='bm64_s1',block_m=64,splits=1),
         dict(name='bm64_s2',block_m=64,splits=2),
         dict(name='bm32_s4',block_m=32,splits=4),dict(name='bm64_s4',block_m=64,splits=4)]


def compare_logits(actual,expected,cfg,enforce=True):
    rows=[]
    for point,value in actual.items():
        reference=expected[point].to(value.device)
        if enforce:torch.testing.assert_close(value,reference,atol=cfg['reference_atol'],rtol=cfg['reference_rtol'])
        rows.append(dict(checkpoint=point,bitwise_equal=torch.equal(value,reference),
            original_tolerance_pass=bool(torch.isclose(value,reference,atol=cfg['reference_atol'],rtol=cfg['reference_rtol']).all()),
            max_abs_error=float((value-reference).abs().max()),
            relative_l2=float((value-reference).norm()/reference.norm().clamp_min(1.e-12))))
    assert len(rows)==6
    return rows


def collect(bench,wrapped,tokenizer,old_control,cfg,w,root):
    control=TimingBackend(old_control.raw,old_control.peer,cfg,old_control,cfg['timing_case'])
    control.configure(dict(name='batch2048_fused_async',migration_tokens=2048,async_copy=True,copy_impl='fused',reserve_steps=0))
    fixture=json.loads((Path(__file__).resolve().parent.parent/'session_inputs.json').read_text())
    if cfg['timing_case']=='custom_all_local':
        if cfg.get('reuse_kernel_checks'):
            import hashlib
            source=Path(cfg['reuse_kernel_checks'])
            reused=[]
            for name in ['kernel_checks.json','kernel_timing.json']:
                payload=(source/name).read_bytes();(root/name).write_bytes(payload)
                reused.append(dict(source=str(source/name),sha256=hashlib.sha256(payload).hexdigest()))
            save(root/'kernel_check_provenance.json',dict(samples=reused,source_job='26071989.pbs101',
                policy='Identical kernel code. Reuse completed25 actual-IPC correctness cases and isolated timings; fresh model validation/warmup and ordinary timing below.'))
        else:
            from check_kernel import run
            run(old_control.peer,root)
    choices=cfg.get('kernel_choices',CHOICES) if cfg['timing_case']!='native_local' else CHOICES[:1]
    expected=None;reference_file=Path(cfg['reference_logits']);eligible=[];rejected=[]
    for choice in choices:
        out=root/choice['name'];out.mkdir();control.kernel_choice=choice
        control.check_attention=(choice['name']!='r5')
        control.attention_checks=[]
        validation,values=session(bench,wrapped,control,cfg,fixture,checkpoints=True)
        if expected is None:
            if cfg['timing_case']=='custom_all_local':
                expected={k:v.detach().cpu() for k,v in values.items()};torch.save(expected,reference_file)
            else:expected=torch.load(reference_file,map_location='cpu',weights_only=True)
        # Persist diagnostics before enforcing admission to performance trials.
        # Failed candidates retain original strict gates and are not timed.
        comparisons=compare_logits(values,expected,cfg,enforce=False)
        accepted=(cfg['timing_case']=='native_local' or
            (all(v['original_tolerance_pass'] for v in comparisons) and
             validation['greedy_matches_to_fixture']==validation['greedy_comparisons']==102))
        save(out/'validation.json',dict(logits=comparisons,attention=control.attention_checks,
            accepted_for_performance=accepted,
            actual_output_tokens=validation['actual_output_tokens'],greedy_matches_to_fixture=validation['greedy_matches_to_fixture'],
            greedy_comparisons=validation['greedy_comparisons'],final=validation['final'],migrations=validation['migrations']))
        torch.save({k:v.cpu() for k,v in values.items()},out/'matched_checkpoints.pt')
        control.check_attention=False
        if not accepted:
            rejected.append(choice)
            print('PREFILL5_MODEL_REJECTED',cfg['timing_case'],choice['name'],
                [v['checkpoint'] for v in comparisons if not v['original_tolerance_pass']],
                validation['greedy_matches_to_fixture'],flush=True)
            continue
        eligible.append(choice)
        session(bench,wrapped,control,cfg,fixture)
        print('PREFILL5_MODEL_VALIDATED',cfg['timing_case'],choice['name'],flush=True)
    orders=[];completed=[];rng=random.Random(cfg['random_seed'])
    for trial in range(cfg['plain_trials']):
        order=list(eligible);rng.shuffle(order);orders.append([c['name'] for c in order])
        for choice in order:
            control.kernel_choice=choice
            result,_=session(bench,wrapped,control,cfg,fixture)
            if cfg['timing_case']!='native_local':assert result['greedy_matches_to_fixture']==result['greedy_comparisons']==102,(choice,'greedy output mismatch')
            result.update(kernel=choice,case=cfg['timing_case'],trial=trial,profile=False)
            save(root/choice['name']/f'plain_{trial}.json',result)
            completed.append(dict(kernel=choice['name'],trial=trial,session_wall_ms=result['session_wall_ms'],
                ttft_ms=[r['ttft_wall_ms'] for r in result['rounds']]))
            save(root/'progress.json',completed)
            print('PREFILL5_PLAIN',cfg['timing_case'],choice['name'],trial,completed[-1]['ttft_ms'],flush=True)
    assert eligible and eligible[0]['name']=='r5'
    means={c['name']:statistics.mean(sum(r['ttft_ms']) for r in completed if r['kernel']==c['name']) for c in eligible}
    best=min(means,key=means.get)
    profiled=sorted({'r5',best})
    for name in profiled:
        control.kernel_choice=next(c for c in choices if c['name']==name)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as profiler:
            result,_=session(bench,wrapped,control,cfg,fixture,profile=True)
        profiler.export_chrome_trace(str(root/name/'trace.json'))
        save(root/name/'profile.json',result)
    save(root/'case_collection.json',dict(status='PREFILL5_CASE_COMPLETE',config=cfg,kernels=eligible,rejected_kernels=rejected,orders=orders,
        completed=completed,best_prefill_ttft=best,mean_three_turn_ttft_ms=means,profiled=profiled,
        note='Same controlled histories/chunking;real model/KV. Ordinary timing excludes layer oracles,checkpoints and profiler. Decode uses unchanged r5 attention,2048 requested batch,fused copy,previous-step async prefetch. Unexpected input admission copies before allocation. Strict active remote/local<5.99%.'))
