"""Actual CUDA timestamps; model and outside-forward costs stay distinct."""
import argparse
import collections
import csv
import json
from pathlib import Path
import re
import statistics
import phase_common as common

original_kind=common.kind
def kind(name):
    n=name.lower()
    if '_paged_causal_prefill' in n or n=='_decode':return 'attention_main'
    if n=='_merge':return 'attention_reduce'
    if any(s in n for s in ('store_kvcache','store_kv_cache','reshape_and_cache','append_paged_kv_cache')):return 'kv_store'
    return original_kind(name)
common.kind=kind

def read(p):return json.loads(p.read_text())
def write_csv(p,rows):
    if not rows:return
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def stages(kernels,event_ms,case,uid):
    gemms=[]
    for k in kernels:
        if k['kind']!='gemm':continue
        if 'splitkreduce' in k['name'].lower() or 'splitk_reduce' in k['name'].lower():
            assert gemms and gemms[-1]['stream']==k['stream']
            gemms[-1]['duration_us']=k['start_us']+k['duration_us']-gemms[-1]['start_us']
        else:gemms.append(dict(k))
    assert len(gemms)==145,(case,uid,'logical GEMMs',len(gemms))
    phases=[];active=collections.defaultdict(float)
    for k in kernels:
        if k['kind']=='gemm':active['gemm_ms']+=k['duration_us']/1000
        elif k['kind'].startswith('attention_'):active['attention_ms']+=k['duration_us']/1000
        elif k['kind']=='kv_store':active['kv_store_ms']+=k['duration_us']/1000
    for i,k in enumerate(gemms):
        phases.append(dict(case=case,uid=uid,layer=i//4,phase='GEMM_'+(['QKV','O','MLP_UP','MLP_DOWN'][i%4] if i<144 else 'LM_HEAD'),
                           start_us=k['start_us'],end_us=k['start_us']+k['duration_us']))
    main_count=0
    for layer in range(36):
        lo=gemms[layer*4]['start_us']+gemms[layer*4]['duration_us'];hi=gemms[layer*4+1]['start_us']
        att=[k for k in kernels if k['kind'].startswith('attention_') and lo<=k['start_us']<hi]
        mains=[k for k in att if k['kind']=='attention_main'];assert mains,(case,uid,layer)
        main_count+=len(mains)
        phases.append(dict(case=case,uid=uid,layer=layer,phase='ATTENTION',
            start_us=min(k['start_us'] for k in att),end_us=max(k['start_us']+k['duration_us'] for k in att)))
    stores=[k for k in kernels if k['kind']=='kv_store'];assert len(stores)==36,(case,uid,'KV stores',len(stores))
    for layer,k in enumerate(stores):phases.append(dict(case=case,uid=uid,layer=layer,phase='KV_STORE',start_us=k['start_us'],end_us=k['start_us']+k['duration_us']))
    phases.sort(key=lambda p:p['start_us'])
    assert all(a['end_us']<=b['start_us']+.5 for a,b in zip(phases,phases[1:])),(case,uid,'phase overlap')
    totals=collections.defaultdict(float)
    for p in phases:p['duration_ms']=(p['end_us']-p['start_us'])/1000;totals[p['phase']]+=p['duration_ms']
    gemm=sum(v for k,v in totals.items() if k.startswith('GEMM_'));attention=totals['ATTENTION'];store=totals['KV_STORE']
    span=(max(k['start_us']+k['duration_us'] for k in kernels)-min(k['start_us'] for k in kernels))/1000
    other=event_ms-gemm-attention-store;assert other>=-.01,(case,uid,event_ms,gemm,attention,store)
    return dict(attention_ms=attention,gemm_ms=gemm,kv_store_ms=store,other_and_gaps_ms=other,
        kernel_other_and_gaps_ms=event_ms-sum(active.values()),
        gpu_first_last_span_ms=span,event_minus_kernel_span_ms=event_ms-span,attention_main_kernels=main_count,
        **{k.replace('_ms','_kernel_active_ms'):v for k,v in active.items()},
        **{k.lower()+'_ms':v for k,v in totals.items() if k.startswith('GEMM_')}),phases

def outside_activities(events,case):
    scopes=[]
    for e in events:
        if e.get('cat')!='user_annotation' or e.get('ph')!='X':continue
        m=re.fullmatch(r'SLACK_COST_(.+)_(\d+)',e.get('name',''));model=re.fullmatch(r'SLACK_PHASE_FORWARD_.+_sample(\d+)',e.get('name',''))
        if m or model:scopes.append((e['ts'],e['ts']+e['dur'],e['tid'],m[1] if m else 'MODEL',int(m[2]) if m else int(model[1])))
    owners={}
    for e in events:
        if e.get('cat') not in ('cuda_runtime','cuda_driver') or e.get('ph')!='X':continue
        corr=e.get('args',{}).get('correlation')
        match=[s for s in scopes if s[2]==e.get('tid') and s[0]<=e['ts']<=s[1]]
        if corr is not None and match:owners[corr]=min(match,key=lambda s:s[1]-s[0])[3:]
    rows=[]
    for e in events:
        if e.get('cat') not in ('kernel','gpu_memcpy','gpu_memset') or e.get('ph')!='X':continue
        own=owners.get(e.get('args',{}).get('correlation'))
        if own and own[0]!='MODEL':rows.append(dict(case=case,uid=own[1],owner=own[0],category=e['cat'],name=e['name'],start_us=e['ts'],duration_us=e['dur']))
    return rows

def main(root):
    collection=read(root/'collection.json');assert collection['status']=='MATCHED_PREFILL_TIMING_COLLECTION_COMPLETE'
    cfg=collection['config'];summaries=[];chunk_rows=[];phase_rows=[];kernel_rows=[];cost_rows=[];plain_rows=[]
    for case in cfg['case_order']:
        path=root/case;profile=read(path/'profile.json');plain=[read(path/f'plain_{i}.json') for i in range(cfg['plain_trials'])]
        assert len(profile['forwards'])==59 and len(profile['chunks'])==11
        assert all(v['final']['logical_tokens']==4296 for v in plain)
        for run in plain:
            assert all(run['rounds'][i]['start_unix_ns']>=run['rounds'][i-1]['decode_complete_unix_ns'] for i in (1,2))
            assert [(v['prefix_lengths'],v['query_lengths'],v['context_lengths']) for v in run['chunks']]==[(v['prefix_lengths'],v['query_lengths'],v['context_lengths']) for v in profile['chunks']]
            for t in run['rounds']:plain_rows.append(dict(case=case,trial=run['trial'],**t))
        trace=read(path/'prefill.trace.json');samples=common.extract_samples(trace['traceEvents']);assert len(samples)==59
        costs=outside_activities(trace['traceEvents'],case);cost_rows+=costs
        frames={v['uid']:v for v in profile['forwards']};chunk_by={v['uid']:v for v in profile['chunks']}
        rows=[]
        for uid,c in chunk_by.items():
            f=frames[uid];v,phases=stages(samples[uid],f['model_event_ms'],case,uid);phase_rows+=phases
            row=dict(case=case,uid=uid,turn=c['turn'],chunk=c['chunk'],prefix_lengths=c['prefix_lengths'],query_lengths=c['query_lengths'],
                context_lengths=c['context_lengths'],remote_active=c['remote_active'],profile_model_event_ms=f['model_event_ms'],
                profile_custom_page_wall_ms=f['custom_page_wall_ms'],profile_custom_page_gpu_ms=f['custom_page_gpu_ms'],**v)
            rows.append(row);chunk_rows.append(row)
            for k in samples[uid]:kernel_rows.append(dict(case=case,uid=uid,**k))
        for turn in range(3):
            chosen=[v for v in rows if v['turn']==turn]
            normal=[v['rounds'][turn] for v in plain]
            model=[sum(v['model_event_ms'] for v in run['forwards'] if v['phase']=='prefill' and v['turn']==turn) for run in plain]
            def average_phase(field,source):return statistics.mean(sum(v[field] for v in run[source] if v['turn']==turn) for run in plain)
            migrations=[sum(v['wall_ms'] for v in run['migrations'] if v['phase']=='prefill' and v['turn']==turn) for run in plain]
            migration_gpu=[sum(v['gpu_stream_ms'] for v in run['migrations'] if v['phase']=='prefill' and v['turn']==turn) for run in plain]
            decode_migrations=[sum(v['wall_ms'] for v in run['migrations'] if v['phase']=='decode' and v['turn']==turn) for run in plain]
            decode_migration_gpu=[sum(v['gpu_stream_ms'] for v in run['migrations'] if v['phase']=='decode' and v['turn']==turn) for run in plain]
            page=[sum(v['custom_page_wall_ms'] for v in run['forwards'] if v['phase']=='prefill' and v['turn']==turn) for run in plain]
            prep=[sum(v['scheduling_before_forward_wall_ms']+v['request_fields_wall_ms'] for v in run['chunks'] if v['turn']==turn) for run in plain]
            sums={k:sum(v[k] for v in chosen) for k in ('attention_ms','gemm_ms','kv_store_ms','other_and_gaps_ms','kernel_other_and_gaps_ms','profile_model_event_ms','attention_kernel_active_ms','gemm_kernel_active_ms','kv_store_kernel_active_ms')}
            summaries.append(dict(case=case,turn=turn,input_type='initial' if turn==0 else 'append',batch=2,chunks=len(chosen),
                plain_ttft_ms=statistics.mean(v['ttft_wall_ms'] for v in normal),plain_ttft_stdev_ms=statistics.stdev(v['ttft_wall_ms'] for v in normal),
                plain_model_sum_ms=statistics.mean(model),profile_ttft_ms=profile['rounds'][turn]['ttft_wall_ms'],
                profile_model_to_plain_ratio=sums['profile_model_event_ms']/statistics.mean(model),
                prefill_migration_wall_ms=statistics.mean(migrations),prefill_migration_gpu_stream_ms=statistics.mean(migration_gpu),
                decode_migration_wall_ms=statistics.mean(decode_migrations),decode_migration_gpu_stream_ms=statistics.mean(decode_migration_gpu),
                custom_page_update_wall_ms=statistics.mean(page),schedule_and_request_preparation_wall_ms=statistics.mean(prep),
                peer_active_after_prefill=normal[0]['remote_after_prefill'],**sums))
    by={(v['case'],v['turn']):v for v in summaries}
    deltas=[]
    for turn in range(3):
        for baseline in ('native_local','custom_all_local'):
            a,b=by[baseline,turn],by['tiered_peer',turn]
            deltas.append(dict(turn=turn,baseline=baseline,ttft_change=b['plain_ttft_ms']/a['plain_ttft_ms']-1,
                profile_attention_delta_ms=b['attention_ms']-a['attention_ms'],profile_gemm_delta_ms=b['gemm_ms']-a['gemm_ms'],
                profile_kv_store_delta_ms=b['kv_store_ms']-a['kv_store_ms'],plain_ttft_delta_ms=b['plain_ttft_ms']-a['plain_ttft_ms']))
    result=dict(status='MATCHED_PREFILL_TIMING_ANALYSIS_COMPLETE',config=cfg,rounds=summaries,deltas=deltas,
        matched_output_checks=read(root/'matched_output_checks.json'),
        definitions='Primary plain TTFT: tokenized input ready through CPU-visible first sampled token, includes all chunks, required migration, native scheduling/allocation, custom page management and sampling. Previous decode completes before each new input timer. Stage columns use separate diagnostic traces; both phase envelopes and kernel-active durations are retained. Model Other/gaps closes the profiled model Event interval, including capture-induced stalls. CPU preparation/page/migration intervals may overlap GPU work and are not added to GPU stages. Migration API wall time can include waiting for preceding queued GPU work at the completion boundary; its CUDA-event interval is separately reported. Migration during decode is separate.',
        capacity='Native/custom-all-local pool4352 versus peer local4096+donor256. Equal logical work, not an assertion that fixed local4096 can admit4296 tokens.',
        replay='Same B2 token histories, query/prefix lengths and chunk boundaries from validated26070196; inter-round decode consumes saved token IDs while performing real forward/sample/KV append. No serving queue or tokenizer time included.')
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    for name,rows in [('prefill_rounds.csv',summaries),('prefill_deltas.csv',deltas),('prefill_chunks.csv',chunk_rows),('prefill_phase_intervals.csv',phase_rows),('prefill_kernel_times.csv',kernel_rows),('outside_forward_cuda.csv',cost_rows),('plain_ttft_trials.csv',plain_rows)]:write_csv(root/name,rows)
    plot(root,summaries)
    print(json.dumps(dict(status=result['status'],rounds=summaries,output_checks=result['matched_output_checks']),indent=2))

def plot(root,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'STIXGeneral','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,3,figsize=(12.5,7));labels=['Native\nlocal','Custom\nlocal','Peer\nKV']
    colors=[('attention_kernel_active_ms','Attention kernels','#2E7BD6'),('gemm_kernel_active_ms','GEMM kernels','#65B4AD'),('kv_store_kernel_active_ms','KV write','#B8A578'),('kernel_other_and_gaps_ms','Other / gaps','#A4A7A2')]
    for turn in range(3):
        chosen=[v for v in rows if v['turn']==turn];x=np.arange(3);bottom=np.zeros(3)
        for key,label,color in colors:
            values=[r[key] for r in chosen];axes[0,turn].bar(x,values,bottom=bottom,color=color,label=label);bottom+=values
        axes[0,turn].set_title('Initial input' if turn==0 else f'Append {turn}');axes[0,turn].set_xticks(x,labels);axes[0,turn].set_ylabel('Profiled model sum (ms)' if turn==0 else '')
        axes[1,turn].bar(x,[v['plain_ttft_ms'] for v in chosen],yerr=[v['plain_ttft_stdev_ms'] for v in chosen],color=['#A4A7A2','#65B4AD','#2E7BD6'],capsize=3)
        axes[1,turn].set_xticks(x,labels);axes[1,turn].set_ylabel('Plain input-to-first-token (ms)' if turn==0 else '')
    handles,legend=axes[0,0].get_legend_handles_labels();fig.legend(handles,legend,loc='upper center',ncol=4,frameon=False,bbox_to_anchor=(.5,1.01))
    fig.subplots_adjust(left=.075,right=.98,top=.91,bottom=.13,hspace=.45,wspace=.3)
    fig.text(.5,.025,'Same B2 / token histories / chunks · local controls4352 vs peer4096+256\nUpper: separate CUDA trace diagnostics; lower: unprofiled complete input processing incl. migration/page management',ha='center',fontsize=10,color='#666666')
    for ext in ('png','pdf','svg'):fig.savefig(root/f'prefill_stage_comparison.{ext}',dpi=240,bbox_inches='tight')
    plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
