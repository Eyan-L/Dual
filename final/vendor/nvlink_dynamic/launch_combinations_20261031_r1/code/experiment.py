import json
from pathlib import Path
import statistics
import torch
from backend import TimingBackend
from campaign import session
from experiment_common import save


def collect(bench,wrapped,tokenizer,old_control,cfg,w,root):
    fixture=json.loads((Path(__file__).resolve().parent.parent/'session_inputs.json').read_text())
    golden=torch.load(cfg['golden_logits'],map_location='cpu',weights_only=True)
    control=TimingBackend(old_control.raw,old_control.peer,cfg,old_control,'tiered_peer')
    if any(v.get('direct_remote',False) for v in cfg['variants']):
        from direct_write import validate
        save(root/'direct_write_checks.json',validate(control))
    if any(v['copy_impl']=='fused' for v in cfg['variants']):
        from fused_copy import validate
        save(root/'fused_copy_checks.json',validate(control))
    completed=[]
    for variant in cfg['variants']:
        name=variant['name'];out=root/name;out.mkdir();control.configure(variant)
        validation,check=session(bench,wrapped,control,cfg,fixture,checkpoints=True)
        comparisons=[]
        for point,actual in check.items():
            actual=actual.cpu();assert torch.equal(actual,golden[point]),(name,point,float((actual-golden[point]).abs().max()))
            comparisons.append(dict(checkpoint=point,bitwise_equal=True,max_abs=0.))
        assert len(comparisons)==6
        save(out/'validation.json',dict(checkpoints=comparisons,final=validation['final'],migrations=validation['migrations']))
        for _ in range(cfg['warmup_sessions']):session(bench,wrapped,control,cfg,fixture)
        rows=[]
        for trial in range(cfg['plain_trials']):
            result,_=session(bench,wrapped,control,cfg,fixture)
            assert result['greedy_matches_to_fixture']==result['greedy_comparisons']==102,(name,trial,'greedy output mismatch')
            result.update(variant=variant,trial=trial,profile=False);save(out/f'plain_{trial}.json',result);rows.append(result)
            print('DYNAMIC_PLAIN',name,trial,result['session_wall_ms'],len(result['migrations']),flush=True)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=False,with_stack=False,profile_memory=False) as profiler:
            profile,_=session(bench,wrapped,control,cfg,fixture,profile=True)
        profiler.export_chrome_trace(str(out/'trace.json'));save(out/'profile.json',profile)
        row=dict(variant=variant,session_wall_ms=statistics.mean(v['session_wall_ms'] for v in rows),
            session_stdev_ms=statistics.stdev(v['session_wall_ms'] for v in rows),
            ttft_ms=[statistics.mean(v['rounds'][t]['ttft_wall_ms'] for v in rows) for t in range(3)],
            decode_wall_ms=[statistics.mean((v['rounds'][t]['decode_complete_unix_ns']-v['rounds'][t]['end_unix_ns'])/1e6 for v in rows) for t in range(3)],
            migrations=len(rows[0]['migrations']),moved_tokens=sum(v['tokens'] for v in rows[0]['migrations']),
            migration_gpu_ms=statistics.mean(sum(m['gpu_stream_ms'] for m in v['migrations']) for v in rows),
            final=rows[0]['final'],validation='6 bitwise-equal logits checkpoints; no diagnostics in timed sessions')
        completed.append(row);save(out/'summary.json',row);save(root/'progress.json',completed)
    save(root/'stage_collection.json',dict(status='DYNAMIC_STAGE_COMPLETE',config=cfg,results=completed,
        attention_kernels='Frozen nvlink_prefill/launch_20261031_r5; strategy5 not applied',
        measurement='All3 input rounds +16 decode forwards each, including all intervening migrations/management. No checkpoint copies inside timed sessions. No artificial slack. Original capacities, page_size1, same fixture.'))
