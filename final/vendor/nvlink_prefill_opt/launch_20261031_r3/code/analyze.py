"""Unprofiled TTFT paired trials; separate CUDA stage diagnosis incl. merge."""

from pathlib import Path as _FinalPath
_SLACK_FINAL_ROOT = next(p for p in _FinalPath(__file__).resolve().parents if (p / 'release.json').is_file())
_SLACK_PROJECT_ROOT = _SLACK_FINAL_ROOT.parent
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import statistics
import sys


def read(p):return json.loads(p.read_text())
def save(p,d):p.write_text(json.dumps(d,indent=2)+'\n')
def write_csv(p,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def main(root):
    collection=read(root/'collection.json');assert collection['status']=='PREFILL5_COMPLETE'
    source=Path(str(_SLACK_FINAL_ROOT / 'vendor/nvlink_prefill_timing/launch_20261031_r2/code'))
    sys.path.insert(0,str(source))
    spec=importlib.util.spec_from_file_location('frozen_prefill_stage_analysis_for5',source/'analyze.py')
    stage=importlib.util.module_from_spec(spec);spec.loader.exec_module(stage)
    old_kind=stage.common.kind
    def kind(name):
        if '_split_causal_prefill' in name:return 'attention_main'
        if '_merge_causal_prefill' in name:return 'attention_reduce'
        return old_kind(name)
    stage.common.kind=kind
    summaries=[];trials=[];frames=[];phases=[];outside=[];migrations=[];rejected={}
    for case in collection['config']['case_order']:
        p=root/case;cfg=read(p/'case_collection.json')
        rejected[case]=cfg.get('rejected_kernels',[])
        for choice in cfg['kernels']:
            name=choice['name'];out=p/name
            runs=[read(out/f'plain_{i}.json') for i in range(collection['config']['plain_trials'])]
            for run in runs:
                assert len(run['forwards'])==59 and run['final']['logical_tokens']==4296
                assert all(run['rounds'][i]['start_unix_ns']>=run['rounds'][i-1]['decode_complete_unix_ns'] for i in (1,2))
                for m in run['migrations']:migrations.append(dict(case=case,kernel=name,trial=run['trial'],**m))
                for r in run['rounds']:trials.append(dict(case=case,kernel=name,trial=run['trial'],**r))
            for turn in range(3):
                ttft=[v['rounds'][turn]['ttft_wall_ms'] for v in runs]
                summaries.append(dict(case=case,kernel=name,turn=turn,ttft_mean_ms=statistics.mean(ttft),
                    ttft_sd_ms=statistics.stdev(ttft),session_mean_ms=statistics.mean(v['session_wall_ms'] for v in runs),
                    remote_tokens=runs[0]['rounds'][turn]['remote_after_prefill'],
                    plain_model_mean_ms=statistics.mean(sum(f['model_event_ms'] for f in v['forwards'] if f['phase']=='prefill' and f['turn']==turn) for v in runs)))
            if (out/'profile.json').exists():
                profile=read(out/'profile.json');events=read(out/'trace.json')['traceEvents']
                samples=stage.common.extract_samples(events);assert len(samples)==59
                outside.extend(dict(kernel=name,**v) for v in stage.outside_activities(events,case))
                for f in profile['forwards']:
                    uid=f['uid'];values,parts=stage.stages(samples[uid],f['model_event_ms'],case,uid)
                    plain=statistics.mean(v['forwards'][uid]['model_event_ms'] for v in runs)
                    frames.append(dict(case=case,kernel=name,uid=uid,phase=f['phase'],turn=f['turn'],
                        prefix_lengths=f['prefix_lengths'],query_lengths=f['query_lengths'],profile_model_ms=f['model_event_ms'],
                        profile_to_plain_ratio=f['model_event_ms']/plain,**values))
                    phases.extend(dict(kernel=name,**v) for v in parts)
    report=dict(status='PREFILL5_REVIEW_COMPLETE',results=summaries,local_peer_checks=collection['local_peer_checks'],
        external_control_job=collection['config'].get('external_control_job'),
        qualified_in_both_cases=collection['qualified_in_both_cases'],rejected_kernels=rejected,
        decode='Unchanged r5 decode attention; requested2048 total KV tokens, fused all-layer copy,previous-step asynchronous launch; strict active remote/local<5.99%. This B2 small fixture clips2048 to admissible budget. Unexpected appended-input admission migrates before allocating; not future-input prediction.',
        capacity='Native and custom-local4352 slots are equal-work controls. Peer compute4096. Donor allocates5888 slots for long-context kernel probes, but measured multi-turn active usage is strictly ratio-limited. No full-KV mirror during measured runs.',
        timing='Primary TTFT includes page admission/migration/all chunks/sample and CPU-visible first token;previous decode completed before input. Checkpoints/per-layer reference/FP64 probes outside primary runs. Profile stages include split merge as Attention; other/gaps close whole Event. CPU and GPU intervals are nonadditive. Controlled replay histories,not new online serving or autoregressive throughput claim.',
        correctness='Candidate full-model logits checked at original atol.01/rtol.02 against newly collected r5 local; every disputed per-layer same-QKV attention row independently checked against FP64 at unchanged tolerances. Native/custom differences reported separately. Candidate greedy102/102. Each custom kernel local/peer6 checkpoints bitwise equal.')
    save(root/'summary.json',report)
    for name,rows in [('prefill_rounds.csv',summaries),('ttft_trials.csv',trials),('profile_stages.csv',frames),('phase_intervals.csv',phases),('outside_forward.csv',outside),('migration_events.csv',migrations)]:write_csv(root/name,rows)
    print('PREFILL5_REVIEW_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
