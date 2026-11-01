"""Separate unprofiled end-to-end/turn metrics and complete GPU phase budgets."""

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

SOURCE=Path(str(_SLACK_FINAL_ROOT / 'vendor/nvlink_flashinfer_interleaved_decode/launch_20261101_r3/code'))
sys.path.insert(0,str(SOURCE))
from stage_budget import stages
from trace_index import extract_samples
PHASE=Path(str(_SLACK_FINAL_ROOT / 'vendor/nvlink_prefill_timing/launch_20261031_r2/code'))
sys.path.insert(0,str(PHASE))
spec=importlib.util.spec_from_file_location('frozen_stage_classification',PHASE/'analyze.py')
classification=importlib.util.module_from_spec(spec);spec.loader.exec_module(classification)


def read(p):return json.loads(p.read_text())
def table(p,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def main(root):
    assert read(root/'collection.json')['status']=='PERSISTENT_MULTITURN_COLLECTION_COMPLETE'
    cfg=read(root/'config.json');totals=[];turns=[];steps=[];chunks=[];management=[];phases=[];kernels=[];intervals=[]
    ref=read(root/'native8_reference/validation.json')
    for choice in cfg['cases']:
        if choice.get('reference_only'):continue
        case=choice['label'];folder=root/case;plain=read(folder/'plain_0.json');prof=read(folder/'profile.json');checked=read(folder/'validation.json')
        b=choice['batch'];assert plain['initial_remote_tokens']==0
        assert plain['history_sha256']==prof['history_sha256']==checked['history_sha256']==ref['history_sha256'][:b]
        assert plain['final']['logical_tokens']==0 and not plain['primary_query_copies']
        assert not plain['attention_checks'] and not plain['prefill_checks']
        assert all(not s['remote_active'] or 10000*s['remote_active']<599*s['local_active'] for s in plain['steps'])
        if case=='peer8':assert all(c['bitwise_equal'] for c in read(folder/'same_order_local_shadow.json')['checkpoints'])
        totals.append(dict(case=case,batch=b,wall_ms=plain['wall_ms'],output_tokens=plain['output_tokens'],
            output_tokens_per_second=plain['output_tokens_per_second'],input_tokens=plain['input_tokens'],
            external_input_tokens=plain['external_input_tokens'],decode_forward_tokens=plain['decode_forward_tokens'],
            initial_ttft_ms=plain['rounds'][0]['ttft_wall_ms'],append_ttft_max_ms=max(r['ttft_wall_ms'] for r in plain['rounds'][1:]),
            migration_count=len(plain['migrations']),migration_tokens=sum(m['tokens'] for m in plain['migrations']),
            final_local=plain['endpoint']['local_active_tokens'],final_remote=plain['endpoint']['remote_active_tokens'],
            profiler_wall_ratio=prof['wall_ms']/plain['wall_ms'],primary_trials=1))
        turns.extend(dict(case=case,**{k:v for k,v in r.items() if k not in ['prefill_suffixes','pending_last_output','actual_first_token']}) for r in plain['rounds'])
        steps.extend(dict(case=case,**s) for s in plain['steps']);chunks.extend(dict(case=case,**c) for c in plain['chunks'])
        management.extend(dict(case=case,**m) for m in plain['migrations'])
        events=read(folder/'multiturn.trace.json')['traceEvents'];samples=extract_samples(events,classification.common.kind)
        assert len(samples)==len(prof['forwards'])
        for f in prof['forwards']:
            uid=f['uid'];value,parts=stages(samples[uid],f['model_event_ms'],case,uid)
            phases.append(dict(case=case,uid=uid,turn=f['turn'],phase=f['phase'],chunk=f.get('chunk',-1),step=f['step'],
                query_lengths=f['query_lengths'],context_lengths=f['context_lengths'],model_ms=f['model_event_ms'],**value))
            intervals.extend(parts);kernels.extend(dict(case=case,uid=uid,**k) for k in samples[uid])
    for name,rows in [('summary.csv',totals),('rounds.csv',turns),('decode_steps.csv',steps),('prefill_chunks.csv',chunks),
        ('migrations.csv',management),('profile_stages.csv',phases),('phase_intervals.csv',intervals),('kernels.csv',kernels)]:table(root/name,rows)
    report=dict(status='PERSISTENT_MULTITURN_REVIEW_COMPLETE',results=totals,config=cfg,
        fairness='First7 shortened trace-material tasks/history/token inputs identical;B8 addsone request. Same fixed localC98304 baseline;dual charges23MiB/164slots,C98140. Native8C104192 solely free-generation/token-history reference. Formal measurements replay reference-generated tokens while executing real prefill/decode/sample/KV writes. No original agent DAG/tool waits or task-quality claim.',
        timing='One primary/configuration,first prefill included,all turns/migrations/page management/token visibility charged. First output per turn comes from final prefill,plus128 outputs from128decode forwards. Final pending token committed on next prefill,not silently discarded;last round pending need not be cached. Independent exact-point warmup and full profiler. Copy overlap is not added to model phases;profiler impact disclosed.',
        rounds=turns,migrations=management)
    (root/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],results=totals),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
