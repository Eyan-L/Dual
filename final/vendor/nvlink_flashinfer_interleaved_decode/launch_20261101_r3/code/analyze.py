"""Actual kernel activity and closed CPU/step budgets;no bandwidth model."""

from pathlib import Path as _FinalPath
_SLACK_FINAL_ROOT = next(p for p in _FinalPath(__file__).resolve().parents if (p / 'release.json').is_file())
_SLACK_PROJECT_ROOT = _SLACK_FINAL_ROOT.parent
import argparse
import collections
import csv
import importlib.util
import json
from pathlib import Path
import re
import statistics
import sys
from trace_index import Timeline,extract_samples


def read(p):return json.loads(p.read_text())
def table(p,rows):
    if not rows:return
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def quantile(values,p):
    a=sorted(values);x=(len(a)-1)*p;i=int(x)
    return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(x-i)


def activities(events):
    """Attribute ALL step GPU activity by launch correlation,including copies."""
    scopes=[]
    for event in events:
        if event.get('cat')!='user_annotation' or event.get('ph')!='X':continue
        name=event.get('name','');step=re.fullmatch(r'DECODE_DETAIL_STEP_(\d+)',name)
        component=re.fullmatch(r'DECODE_DETAIL_(HEADROOM|SCHEDULE|FORWARD_CALL|SAMPLING|PREFETCH|VISIBILITY)_(\d+)',name)
        if step or component:scopes.append(dict(lo=event['ts'],hi=event['ts']+event['dur'],tid=event['tid'],
            step=int(step[1] if step else component[2]),component='WHOLE_STEP' if step else component[1]))
    outer=Timeline([s for s in scopes if s['component']=='WHOLE_STEP'])
    for event in events:
        if event.get('cat')!='user_annotation' or event.get('ph')!='X':continue
        name=event.get('name','')
        component='PAGE_TABLE' if name.startswith('SLACK_COST_PAGE_') else ('MODEL' if name.startswith('SLACK_PHASE_FORWARD_') else None)
        if component:
            parent=outer.at(event['tid'],event['ts']);assert parent is not None
            scopes.append(dict(lo=event['ts'],hi=event['ts']+event['dur'],tid=event['tid'],step=parent['step'],component=component))
    components=collections.defaultdict(list)
    for scope in scopes:components[scope['tid'],scope['step']].append(scope)
    launch={}
    for event in events:
        if event.get('cat') not in ['cuda_runtime','cuda_driver'] or event.get('ph')!='X':continue
        correlation=event.get('args',{}).get('correlation')
        parent=outer.at(event['tid'],event['ts'])
        matches=[s for s in components[event['tid'],parent['step']] if s['lo']<=event['ts']<=s['hi']] if parent else []
        if matches and correlation is not None:launch[correlation]=min(matches,key=lambda v:v['hi']-v['lo'])
    rows=[]
    for event in events:
        if event.get('cat') not in ['kernel','gpu_memcpy','gpu_memset'] or event.get('ph')!='X':continue
        scope=launch.get(event.get('args',{}).get('correlation'))
        if scope:rows.append(dict(step=scope['step'],component=scope['component'],category=event['cat'],name=event['name'],
            start_us=event['ts'],end_us=event['ts']+event['dur'],duration_us=event['dur'],
            stream=event.get('args',{}).get('stream'),correlation=event.get('args',{}).get('correlation')))
    return rows


def union_us(rows):
    merged=[]
    for a,z in sorted((r['start_us'],r['end_us']) for r in rows):
        if merged and a<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],z)
        else:merged.append([a,z])
    return sum(z-a for a,z in merged)


def main(root):
    assert read(root/'collection.json')['status']=='DECODE_DETAIL_COMPLETE'
    cfg=read(root/'config.json');assert cfg['plain_trials']==1
    representative=cfg['representative_step_1based']-1
    source=Path(str(_SLACK_FINAL_ROOT / 'vendor/nvlink_prefill_timing/launch_20261031_r2/code'))
    sys.path.insert(0,str(source));spec=importlib.util.spec_from_file_location('decode_detail_phase_extractor',source/'analyze.py')
    phases=importlib.util.module_from_spec(spec);spec.loader.exec_module(phases)
    from stage_budget import stages
    phases.stages=stages
    summaries=[];budgets=[];intervals=[];kernels=[];all_activity=[];migration_rows=[];operators=[];windows=[]
    for case in cfg['cases']:
        label=case['label'];folder=root/label;plain=read(folder/'plain_0.json');profile=read(folder/'profile.json')
        assert not plain['primary_query_copies'] and not plain['validation_checks']
        assert plain['initial_audit']['remote_active_tokens']==profile['initial_audit']['remote_active_tokens']==0
        assert all(m['phase']=='decode' for m in plain['migrations'])
        assert plain['prompt_sha256']==profile['prompt_sha256']
        assert plain['consumed_ids']==profile['consumed_ids']
        paired=read(folder/'profile_pair_checks.json')
        events=read(folder/'decode.trace.json')['traceEvents'];samples=extract_samples(events,phases.common.kind)
        assert len(samples)==cfg['decode_steps']
        gpu_activity=activities(events);all_activity.extend(dict(case=label,**r) for r in gpu_activity)
        activity_by_step=collections.defaultdict(list)
        for row in gpu_activity:activity_by_step[row['step']].append(row)
        frames={f['uid']:f for f in profile['forwards']};plain_frames={f['uid']:f for f in plain['forwards']}
        phase_values=[];by_operator=collections.defaultdict(lambda:[0,0.])
        previous_visible=0.
        for step in plain['steps']:
            uid=step['uid'];assert uid in samples and uid in frames
            assert abs(sum(v for k,v in step.items() if k.endswith('_cpu_ms'))+step['cpu_residual_ms']-step['wall_ms'])<1.e-6
            token_gap=step['elapsed_ms']-previous_visible;previous_visible=step['elapsed_ms']
            bookkeeping=token_gap-step['wall_ms'];assert bookkeeping>=-.001
            value,parts=phases.stages(samples[uid],frames[uid]['model_event_ms'],label,uid)
            for part in parts:intervals.append(dict(case=label,step=step['step'],**{k:v for k,v in part.items() if k!='case'}))
            for kernel in samples[uid]:
                kernels.append(dict(case=label,step=step['step'],uid=uid,**kernel))
                by_operator[kernel['name']][0]+=1;by_operator[kernel['name']][1]+=kernel['duration_us']/1000
            active=activity_by_step[step['step']]
            active_union=union_us(active)/1000
            span=(max((r['end_us'] for r in active),default=0)-min((r['start_us'] for r in active),default=0))/1000
            roles=collections.defaultdict(float)
            for part in parts:roles[part['phase']]+=part['duration_ms']
            row=dict(case=label,**step,token_visible_gap_ms=token_gap,before_step_bookkeeping_ms=bookkeeping,
                token_visible_edge_residual_ms=token_gap-step['main_stream_pre_visibility_ms'],
                plain_model_gpu_ms=plain_frames[uid]['model_event_ms'],
                plain_page_gpu_ms=plain_frames[uid]['custom_page_gpu_ms'],plain_page_cpu_ms=plain_frames[uid]['custom_page_wall_ms'],
                forward_submission_cpu_excluding_page_ms=step['forward_call_cpu_ms']-plain_frames[uid]['custom_page_wall_ms'],
                profile_model_gpu_ms=frames[uid]['model_event_ms'],profile_model_ratio=frames[uid]['model_event_ms']/plain_frames[uid]['model_event_ms'],
                profile_all_stream_active_union_ms=active_union,profile_all_stream_span_ms=span,
                profile_page_activity_ms=union_us([r for r in active if r['component']=='PAGE_TABLE'])/1000,
                profile_scheduling_activity_ms=union_us([r for r in active if r['component']=='SCHEDULE'])/1000,
                profile_sampling_activity_ms=union_us([r for r in active if r['component']=='SAMPLING'])/1000,
                profile_all_stream_idle_ms=span-active_union,**{'profile_role_'+k.lower()+'_ms':v for k,v in roles.items()},
                **{'profile_'+k:v for k,v in value.items()})
            assert row['profile_all_stream_idle_ms']>=-.001
            assert row['token_visible_edge_residual_ms']>=-.02
            assert abs(row['plain_model_gpu_ms']+row['plain_page_gpu_ms']+row['other_main_stream_ms']+
                row['token_visible_edge_residual_ms']-token_gap)<1.e-6
            budgets.append(row);phase_values.append(row)
        for name,(count,total) in by_operator.items():operators.append(dict(case=label,kernel=name,count=count,total_ms=total))
        for migration in plain['migrations']:
            if migration.get('setup_only'):continue
            commit=migration['commit_step'];launch=migration['launch_step']
            byuid={r['uid']:r for r in plain['steps']}
            migration_rows.append(dict(case=label,**migration,launch_decode_step=byuid.get(launch,{}).get('step'),
                commit_decode_step=byuid.get(commit,{}).get('step'),
                launch_step_wall_ms=byuid.get(launch,{}).get('wall_ms'),commit_step_wall_ms=byuid.get(commit,{}).get('wall_ms')))
        summary=dict(case=label,batch=case['batch'],primary_trials=1,steps=len(plain['steps']),
            decode_tokens_per_second=plain['decode_tokens_per_second'],decode_wall_ms=plain['decode_wall_ms'],
            step_wall_median_ms=statistics.median(s['token_visible_gap_ms'] for s in phase_values),step_wall_p95_ms=quantile([s['token_visible_gap_ms'] for s in phase_values],.95),
            step_wall_max_ms=max(s['token_visible_gap_ms'] for s in phase_values),loop_tail_ms=plain['decode_wall_ms']-plain['steps'][-1]['elapsed_ms'],
            model_gpu_median_ms=statistics.median(f['model_event_ms'] for f in plain['forwards']),
            model_event_sum_ms=sum(f['model_event_ms'] for f in plain['forwards']),
            non_model_wall_residual_ms=plain['decode_wall_ms']-sum(f['model_event_ms'] for f in plain['forwards']),
            primary_representative_visible_ms=phase_values[representative]['token_visible_gap_ms'],
            primary_representative_model_gpu_ms=phase_values[representative]['plain_model_gpu_ms'],
            primary_representative_page_gpu_ms=phase_values[representative]['plain_page_gpu_ms'],
            primary_representative_other_main_stream_ms=phase_values[representative]['other_main_stream_ms'],
            primary_representative_edge_ms=phase_values[representative]['token_visible_edge_residual_ms'],
            profile_plain_model_ratio=sum(f['model_event_ms'] for f in profile['forwards'])/sum(f['model_event_ms'] for f in plain['forwards']),
            local_endpoint=plain['endpoint_audit']['local_active_tokens'],remote_endpoint=plain['endpoint_audit']['remote_active_tokens'],
            timed_migrations=sum(not m.get('setup_only',False) for m in plain['migrations']),
            profile_plain_prediction_fraction=paired['greedy_match_fraction'],
            profile_representative_step=representative+1,
            **{k:phase_values[representative][k] for k in phase_values[representative] if k.startswith('profile_role_')},
            **{k.replace('_ms','_median_ms'):statistics.median(v[k] for v in phase_values)
                for k in ['profile_attention_ms','profile_gemm_ms','profile_kv_store_ms','profile_other_and_gaps_ms']},
            **{k:phase_values[representative][k] for k in ['profile_attention_ms','profile_gemm_ms','profile_kv_store_ms','profile_other_and_gaps_ms']})
        assert abs(sum(s['token_visible_gap_ms'] for s in phase_values)+summary['loop_tail_ms']-plain['decode_wall_ms'])<1.e-6
        for window in cfg['analysis_windows']:
            chosen=[v for v in phase_values if window['first_step']<=v['step']+1<=window['last_step']]
            assert len(chosen)==window['last_step']-window['first_step']+1
            elapsed=sum(v['token_visible_gap_ms'] for v in chosen)
            windows.append(dict(case=label,window=window['name'],first_step=window['first_step'],last_step=window['last_step'],
                first_context=chosen[0]['context'],last_context=chosen[-1]['context'],wall_ms=elapsed,
                decode_tokens_per_second=case['batch']*len(chosen)*1000/elapsed,
                step_median_ms=statistics.median(v['token_visible_gap_ms'] for v in chosen),
                step_p95_ms=quantile([v['token_visible_gap_ms'] for v in chosen],.95),
                remote_start=chosen[0]['remote_active'],remote_end=chosen[-1]['remote_active'],
                **{k:statistics.median(v[k] for v in chosen) for k in ['plain_model_gpu_ms','plain_page_gpu_ms','profile_attention_ms','profile_gemm_ms','profile_kv_store_ms','profile_other_and_gaps_ms']}))
        late=next(v for v in windows if v['case']==label and v['window']=='late128')
        summary.update(late_tokens_per_second=late['decode_tokens_per_second'],late_step_median_ms=late['step_median_ms'],
            late_model_median_ms=late['plain_model_gpu_ms'],late_profile_attention_ms=late['profile_attention_ms'],late_profile_gemm_ms=late['profile_gemm_ms'])
        summaries.append(summary)
    table(root/'step_budgets.csv',budgets);table(root/'model_phase_intervals.csv',intervals);table(root/'model_kernel_times.csv',kernels)
    table(root/'all_step_gpu_activity.csv',all_activity);table(root/'migration_steps.csv',migration_rows);table(root/'operator_totals.csv',operators);table(root/'case_summary.csv',summaries)
    all_ops=collections.defaultdict(lambda:[0,0.])
    for row in all_activity:
        key=(row['case'],row['component'],row['category'],row['name']);all_ops[key][0]+=1;all_ops[key][1]+=row['duration_us']/1000
    table(root/'all_step_operator_totals.csv',[dict(case=key[0],component=key[1],category=key[2],name=key[3],count=value[0],total_active_ms=value[1])
        for key,value in sorted(all_ops.items())])
    deltas=[];by={r['case']:r for r in summaries}
    for a,b,meaning in cfg['comparison_pairs']:
        deltas.append(dict(baseline=a,variant=b,meaning=meaning,
            **{k:by[b][k]-by[a][k] for k in ['step_wall_median_ms','model_gpu_median_ms','profile_attention_ms','profile_gemm_ms','profile_kv_store_ms','profile_other_and_gaps_ms']}))
    table(root/'case_deltas.csv',deltas);table(root/'windows.csv',windows)
    op_delta=[];op_map={(v['case'],v['kernel']):v for v in operators}
    for a,b,_ in cfg['comparison_pairs']:
        names={v['kernel'] for v in operators if v['case'] in [a,b]}
        for name in names:
            left=op_map.get((a,name),{});right=op_map.get((b,name),{})
            op_delta.append(dict(baseline=a,variant=b,kernel=name,baseline_count=left.get('count',0),variant_count=right.get('count',0),
                baseline_profile_total_ms=left.get('total_ms',0),variant_profile_total_ms=right.get('total_ms',0),
                delta_profile_total_ms=right.get('total_ms',0)-left.get('total_ms',0)))
    op_delta.sort(key=lambda r:(r['baseline'],r['variant'],-abs(r['delta_profile_total_ms'])))
    table(root/'operator_deltas.csv',op_delta)
    result=dict(status='DECODE_DETAIL_REVIEW_COMPLETE',config=cfg,cases=summaries,deltas=deltas,
        windows=windows,
        definitions='One primary776-step growing-KV trajectory per case;no3trial average. AllCPU/GPU timestamps kept. CPU API+residual close operation wall;bookkeeping closes token-visible gap;sum gaps+loop tail closes full decode wall. GPU main model,page,other-stream-time and edge residual separately close visible step. CPU/GPU budgets overlap and must not be added. Independent profiler preserves native fusion,full kernel names and actual timestamps;attention includes reduce,GEMM includes split-K. No profiling data rescaled into primary. Profile overview uses same actual final step;per-phase and last128-step summaries predeclared before measurement.',
        layout='Only twoGPU B8 rerun. Native256-token/request chunk-prefill to12224,initial remote0;real776-step KV growth. Unchanged1+2+4/FIFO/2048/fused/async policy and strict R/L<5.99%. Native FlashInfer plan/math/merge;packed read indices interleave8remote/120local pages. Metadata generation is included in page/step cost. Existing native7/native8 numerical references reused;no singleGPU performance rerun.',
        decode_read_order=cfg['decode_read_order'],reference_measurement_root=cfg['reference_measurement_root'])
    overlap_rows=[dict(case=v['case'],step_1based=v['step']+1,overlap_ms=v['profile_timestamp_overlap_ms'],segments=v['profile_timestamp_overlap_segments']) for v in budgets if v['profile_timestamp_overlap_ms']>0]
    table(root/'trace_timestamp_overlaps.csv',overlap_rows)
    result['timestamp_overlap_accounting']=dict(rows=overlap_rows,definition='Reported phase timestamp intersections on one serial CUDA stream are excluded from BOTH phase budgets and retained in Other/gaps. Raw start/end and raw_duration_ms remain unchanged;exclusive duration_ms and trace_timestamp_overlap_ms are explicit. No arbitrary time threshold or manufactured timestamps.')
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n');plot(root,summaries,budgets)
    print(json.dumps(dict(status=result['status'],cases=summaries,deltas=deltas),indent=2))


def plot(root,rows,steps):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'STIXGeneral','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.2));colors=['#333333','#999999','#2E7BD6','#65B4AD']
    for i,row in enumerate(rows):
        chosen=[s for s in steps if s['case']==row['case']]
        axes[0].plot([s['step']+1 for s in chosen],[s['token_visible_gap_ms'] for s in chosen],label=row['case'],color=colors[i],lw=1.2)
    axes[0].set(xlabel='Decode step',ylabel='Complete token-visible step (ms)');axes[0].legend(frameon=False,fontsize=9)
    bottom=[0.]*len(rows)
    for key,label,color in [('profile_attention_ms','Attention + merge','#2E7BD6'),('profile_gemm_ms','GEMM + split-K','#65B4AD'),('profile_kv_store_ms','KV write','#B8A578'),('profile_other_and_gaps_ms','Other / gaps','#AAAAAA')]:
        values=[r[key] for r in rows];axes[1].bar(range(len(rows)),values,bottom=bottom,label=label,color=color);bottom=[a+b for a,b in zip(bottom,values)]
    axes[1].set_xticks(range(len(rows)),[r['case'] for r in rows],rotation=20);axes[1].set_ylabel('Profiled common final step (ms)');axes[1].legend(frameon=False,fontsize=8)
    fig.tight_layout()
    for ext in ['png','pdf','svg']:fig.savefig(root/f'decode_detail.{ext}',dpi=220)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
