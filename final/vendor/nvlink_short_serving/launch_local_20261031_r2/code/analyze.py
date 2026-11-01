"""Short-context serving: complete workload wall and request latency."""

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
def quantile(values,f):
    v=sorted(values);x=(len(v)-1)*f;i=int(x);j=min(i+1,len(v)-1)
    return v[i]+(v[j]-v[i])*(x-i) if v else None
def csv_write(p,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def profiles(root,cfg):
    source=Path(str(_SLACK_FINAL_ROOT / 'vendor/nvlink_prefill_timing/launch_20261031_r2/code'))
    sys.path.insert(0,str(source))
    spec=importlib.util.spec_from_file_location('trace_stage_analysis',source/'analyze.py')
    stage=importlib.util.module_from_spec(spec);spec.loader.exec_module(stage)
    original=stage.common.kind
    def kind(name):
        if '_split_causal_prefill' in name:return 'attention_main'
        if '_merge_causal_prefill' in name:return 'attention_reduce'
        return original(name)
    stage.common.kind=kind;rows=[];phases=[];outside=[]
    for case in cfg['case_order']:
        p=root/case
        if not (p/'profile.json').exists():continue
        run=read(p/'profile.json');events=read(p/'trace.json')['traceEvents']
        samples=stage.common.extract_samples(events)
        assert len(samples)==len(run['forwards'])
        outside.extend(stage.outside_activities(events,case))
        for f in run['forwards']:
            uid=f['uid'];value,parts=stage.stages(samples[uid],f['model_event_ms'],case,uid)
            rows.append(dict(case=case,uid=uid,phase=f['phase'],query_lengths=f['query_lengths'],
                context_lengths=f['context_lengths'],profile_model_ms=f['model_event_ms'],**value))
            phases.extend(parts)
    csv_write(root/'profile_stages.csv',rows);csv_write(root/'profile_phase_intervals.csv',phases)
    csv_write(root/'profile_outside_forward.csv',outside)


def main(root):
    collection=read(root/'collection.json');assert collection['status']=='TRACE_REPLAY_COMPLETE'
    cfg=collection['config'];trials=[];requests=[];steps=[];management=[];all_runs={}
    references=cfg.get('reference_runs',{})
    cases=list(dict.fromkeys([*references,*cfg['case_order']]))
    sources={case:Path(references[case]) if case in references else root/case for case in cases}
    for case in cases:
        for trial in range(cfg['plain_trials']):
            run=read(sources[case]/f'plain_{trial}.json');all_runs[case,trial]=run
            assert run['status']=='REPLAY_COMPLETE' and not run['unfinished'] and not run['unadmitted']
            assert run['final']['logical_tokens']==0
            assert run['replayed_output_tokens']==sum(r['output_tokens'] for r in run['requests'])
            assert run['actual_prefill_tokens']==sum(r['input_tokens'] for r in run['requests'])
            assert all(not s['remote_active'] or 10000*s['remote_active']<599*s['local_active'] for s in run['steps'])
            assert all(not s['remote_after_release'] or 10000*s['remote_after_release']<599*s['local_after_release'] for s in run['steps'])
            tpot=[b-a for r in run['requests'] for a,b in zip(r['token_visible_ms'],r['token_visible_ms'][1:])]
            row=dict(case=case,trial=trial,wall_ms=run['wall_ms'],output_tokens_per_second=run['output_tokens_per_second'],
                completed_requests_per_second=run['completed_requests_per_second'],completed_requests=run['completed_requests'],
                input_tokens=run['actual_prefill_tokens'],output_tokens=run['replayed_output_tokens'],
                ttft_p50_ms=quantile([r['ttft_ms'] for r in run['requests']],.5),ttft_p95_ms=quantile([r['ttft_ms'] for r in run['requests']],.95),
                service_ttft_p95_ms=quantile([r['service_ttft_ms'] for r in run['requests']],.95),
                request_e2e_p95_ms=quantile([r['e2e_ms'] for r in run['requests']],.95),
                tpot_p50_ms=quantile(tpot,.5),tpot_p95_ms=quantile(tpot,.95),tpot_p99_ms=quantile(tpot,.99),
                maximum_running=run['maximum_running'],maximum_active=run['maximum_active'],
                maximum_local_active=max(s['local_active'] for s in run['steps']),maximum_remote_active=max(s['remote_active'] for s in run['steps']),
                migrations=len(run['migrations']),migrated_tokens=sum(m['tokens'] for m in run['migrations']),
                restorations=len(run['restorations']),restored_tokens=sum(m['tokens'] for m in run['restorations']))
            row.update(release_cpu_ms=sum(r['cpu_ms'] for r in run['releases']),source=str(sources[case]),
                reused_reference=case in references)
            for phase in ['prefill','decode']:
                frames=[f for f in run['forwards'] if f['phase']==phase]
                row[phase+'_forwards']=len(frames)
                row[phase+'_model_ms']=sum(f['model_event_ms'] for f in frames)
                row[phase+'_page_gpu_ms']=sum(f['custom_page_gpu_ms'] for f in frames)
                row[phase+'_page_cpu_ms']=sum(f['custom_page_wall_ms'] for f in frames)
            trials.append(row)
            for r in run['requests']:requests.append(dict(case=case,trial=trial,trace_case=r.get('case'),**{k:v for k,v in r.items() if k not in ['case','sampled_tokens','token_visible_ms']}))
            steps.extend(dict(case=case,trial=trial,**s) for s in run['steps'])
            for key in ['migrations','restorations','releases']:
                management.extend(dict(case=case,trial=trial,operation=key,**v) for v in run[key])
    baseline=cases[0]
    for trial in range(cfg['plain_trials']):
        ref=all_runs[baseline,trial]
        for case in cases[1:]:
            run=all_runs[case,trial]
            assert read(sources[case]/'input_checks.json')['prompt_ids_sha256']==read(sources[baseline]/'input_checks.json')['prompt_ids_sha256']
            assert run['request_order']==ref['request_order']
            assert (run['actual_prefill_tokens'],run['replayed_output_tokens'],run['completed_requests'])==(ref['actual_prefill_tokens'],ref['replayed_output_tokens'],ref['completed_requests'])
    summaries=[]
    for case in cases:
        group=[r for r in trials if r['case']==case]
        tokens=sum(r['output_tokens'] for r in group);wall=sum(r['wall_ms'] for r in group)
        summaries.append(dict(case=case,trials=len(group),output_tokens_per_second=tokens/(wall/1000),
            completed_requests_per_second=sum(r['completed_requests'] for r in group)/(wall/1000),
            wall_mean_ms=statistics.mean(r['wall_ms'] for r in group),wall_sd_ms=statistics.stdev(r['wall_ms'] for r in group) if len(group)>1 else None))
    paired=[]
    if 'tiered_peer' in cases:
        for trial in range(cfg['plain_trials']):
            dual=all_runs['tiered_peer',trial]
            for case in ['native_local','custom_all_local']:
                if case not in cases:continue
                local=all_runs[case,trial]
                paired.append(dict(trial=trial,baseline=case,throughput_change=dual['output_tokens_per_second']/local['output_tokens_per_second']-1,
                    wall_saved_ms=local['wall_ms']-dual['wall_ms']))
    local_comparison=[]
    if 'native_local' in cases and 'custom_all_local' in cases:
        for trial in range(cfg['plain_trials']):
            native=all_runs['native_local',trial];custom=all_runs['custom_all_local',trial]
            local_comparison.append(dict(trial=trial,custom_over_native=custom['output_tokens_per_second']/native['output_tokens_per_second'],
                custom_minus_native_wall_ms=custom['wall_ms']-native['wall_ms']))
    report=dict(status='SHORT_SERVING_REVIEW_COMPLETE',config=cfg,results=summaries,paired_trials=paired,local_comparison=local_comparison,
        source_runs={case:dict(path=str(sources[case]),reused=case in references) for case in cases},
        metrics='All requests offered t0. Complete tokenized-input to final drain and KV recycle wall is primary. Fixed generation length, actual greedy autoregressive tokens; EOS ignored. Queue TTFT and service TTFT separate; TPOT contains interleaved prefill, migration and reclamation. Model intervals use whole-forward CUDA Events; page GPU/CPU totals separately. One measured run per configuration, no statistical confidence claim.',
        fairness='Same model,BF16,TP1,local C,short prompt IDs,request order,prefill budgets and capacity admission policy. Reserve prompt+output-1 slots identically. Actual batch capacity-driven. Local-only prefill dispatches to native FlashInfer in both custom-local and dual; custom decode remains tiered-aware. Only dual has remote capacity, donor runs no model. Initialization and warmup excluded.',
        limitations='Controlled short synthetic requests, independent saturated backlog; no tool waits, DAG, task-success or quality claim. Sampled token sequences may differ numerically between backends; lengths and forward work counts fixed. Prefix reuse disabled. Uses SGLang batch APIs rather than the full native online scheduler.')
    save(root/'summary.json',report)
    for name,rows in [('trials.csv',trials),('requests.csv',requests),('steps.csv',steps),('management.csv',management)]:csv_write(root/name,rows)
    if cfg.get('profile_forwards',0):profiles(root,cfg)
    plot(root,summaries)
    print(json.dumps(dict(status=report['status'],results=summaries,paired_trials=paired,local_comparison=local_comparison),indent=2),flush=True)


def plot(root,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'STIXGeneral','font.size':12,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(7,4.8));labels={'native_local':'Native local C','custom_all_local':'Custom local C','tiered_peer':'Dual C + peer'}
    bars=ax.bar([labels[r['case']] for r in rows],[r['output_tokens_per_second'] for r in rows],color=['#A4A7A2','#65B4AD','#2E7BD6'][:len(rows)])
    for bar,row in zip(bars,rows):ax.annotate(f'{row["output_tokens_per_second"]:.2f}',(bar.get_x()+bar.get_width()/2,bar.get_height()),ha='center',va='bottom')
    ax.set_ylabel('Generated output tokens/s');ax.set_ylim(0,max(r['output_tokens_per_second'] for r in rows)*1.15)
    fig.text(.5,.015,'Complete prefill + decode + migration + release wall\nShort synthetic requests · one measured run · same local C',ha='center',fontsize=10,color='#666666')
    fig.tight_layout(rect=(0,.09,1,1))
    for ext in ['png','pdf','svg']:fig.savefig(root/f'throughput.{ext}',dpi=230)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
