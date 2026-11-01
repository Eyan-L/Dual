import argparse
import csv
import json
from pathlib import Path
import statistics


def main(root):
    done=json.loads((root/'throughput_collection.json').read_text())
    assert done['status']=='THROUGHPUT_COLLECTION_COMPLETE'
    cfg=done['config'];C=cfg['kv_capacity_tokens'];M=cfg['remote_capacity_tokens']
    assert done['completed']==[w['name'] for w in cfg['workloads']]
    trial_rows,table=[],[]
    for w in cfg['workloads']:
        data=json.loads((root/w['name']/'workload_collection.json').read_text())
        assert data['status']=='WORKLOAD_COMPLETE' and data['workload']==w
        assert w['native_batch']==C//w['final_context']
        assert w['extended_batch']==(C+M)//w['final_context']
        for calibration in data['calibrations']:
            assert calibration['oracle']['layers']==36
            m=json.loads((root/w['name']/(calibration['source']+'.json')).read_text())
            assert len(m['checks'])==len(m['variants'])
            assert all(r['samples']==cfg['calibration_groups']*cfg['calibration_replays'] for r in m['variants'])
        grouped={name:[] for name in ('native_max','unified_local_max','peer_same_batch','peer_max')}
        for r in data['trials']:
            assert r['name'] in grouped and r['decode_steps']==cfg['decode_steps']==1024
            assert r['generated_tokens']==r['batch']*cfg['decode_steps']
            assert r['wall_seconds']>0 and r['gpu_stream_interval_ms']>0
            assert r['gpu_stream_interval_ms']<=r['wall_seconds']*1000*1.01
            assert r['tokens_per_second']==r['generated_tokens']/r['wall_seconds']
            assert not r['primary_query_copies'] and not r['primary_per_layer_events'] and not r['local_shadow_present']
            assert len(r['correctness'])==2 and all(c['layers']==36 for c in r['correctness'])
            assert [c['context'] for c in r['correctness']]==[w['initial_context'],w['final_context']]
            assert r['local_active_tokens']==r['batch']*(w['final_context']-r['remote_per_request'])<=C
            assert r['remote_active_tokens']==r['batch']*r['remote_per_request']<=M
            assert r['reclamation']['reclaimed_slots']==r['remote_active_tokens']
            if r['attention_mode']!='native':
                assert r['page_audit']['exact_local_slot_match'] and r['page_audit']['exact_peer_slot_match']
                assert r['page_audit']['local_slots']==r['local_active_tokens']
                assert r['page_audit']['peer_slots']==r['remote_active_tokens']
            assert len(r['bins'])==16 and sum(b['output_tokens'] for b in r['bins'])==r['generated_tokens']
            assert sum(b['wall_seconds'] for b in r['bins'])<=r['wall_seconds']*1.01
            assert [b['first_step'] for b in r['bins']]==list(range(1,1025,64))
            assert [b['last_step'] for b in r['bins']]==list(range(64,1025,64))
            assert all(b['tokens_per_second']==b['output_tokens']/b['wall_seconds'] for b in r['bins'])
            assert r['near_full_tail_tokens_per_second']==r['bins'][-1]['tokens_per_second']
            tokens=json.loads((root/w['name']/f"{r['name']}_trial{r['trial']}_tokens.json").read_text())
            inputs,outputs=tokens['input_token_ids'],tokens['output_token_ids']
            assert len(inputs)==len(outputs)==1024 and inputs[1:]==outputs[:-1]
            assert all(len(x)==r['batch'] for x in inputs+outputs)
            grouped[r['name']].append(r)
            trial_rows.append(dict(workload=w['name'],initial_context=w['initial_context'],final_context=w['final_context'],
                name=r['name'],trial=r['trial'],batch=r['batch'],remote_per_request=r['remote_per_request'],attention_mode=r['attention_mode'],
                local_active_tokens=r['local_active_tokens'],remote_active_tokens=r['remote_active_tokens'],
                wall_seconds=r['wall_seconds'],tokens_per_second=r['tokens_per_second'],
                near_full_tail_tokens_per_second=r['near_full_tail_tokens_per_second']))
        assert len(data['trials'])==4*cfg['trials']
        for rows in grouped.values():assert sorted(r['trial'] for r in rows)==list(range(cfg['trials']))
        rate=lambda rows:sum(r['generated_tokens'] for r in rows)/sum(r['wall_seconds'] for r in rows)
        tail=lambda rows:sum(r['bins'][-1]['output_tokens'] for r in rows)/sum(r['bins'][-1]['wall_seconds'] for r in rows)
        native=grouped['native_max'];peer=grouped['peer_max']
        ratios=[next(r['tokens_per_second'] for r in peer if r['trial']==g)/next(r['tokens_per_second'] for r in native if r['trial']==g)-1 for g in range(cfg['trials'])]
        row=dict(workload=w['name'],initial_context=w['initial_context'],final_context=w['final_context'],
            native_batch=w['native_batch'],extended_batch=w['extended_batch'],concurrency_gain=w['extended_batch']/w['native_batch']-1,
            native_tokens_s=rate(native),unified_local_tokens_s=rate(grouped['unified_local_max']),
            peer_same_batch_tokens_s=rate(grouped['peer_same_batch']),peer_max_tokens_s=rate(peer),
            throughput_gain=rate(peer)/rate(native)-1,paired_gain_min=min(ratios),paired_gain_max=max(ratios),
            paired_gain_median=statistics.median(ratios),all_trials_faster=all(x>0 for x in ratios),
            tail_native_tokens_s=tail(native),tail_peer_tokens_s=tail(peer),tail_gain=tail(peer)/tail(native)-1,
            native_final_pool_fraction=native[0]['local_active_tokens']/C,
            peer_final_pool_fraction=peer[0]['local_active_tokens']/C,
            peer_active_KV_fraction=peer[0]['remote_active_tokens']/C,
            donor_capacity_fraction=M/C,peer_active_tokens=peer[0]['remote_active_tokens'])
        table.append(row)
    for name,rows in [('throughput_trials',trial_rows),('throughput_summary',table)]:
        with (root/(name+'.csv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result=dict(status='THROUGHPUT_CHECKS_PASSED',selection_source=cfg['selection_source'],selected_tuning=cfg['selected_tuning'],
        local_capacity_tokens=C,remote_allocated_tokens=M,workloads=table,
        definition='Aggregate real output tokens divided by aggregate decode wall time across three repetitions; complete native batch preparation, full model forward, greedy sampling, page updates, token logging and 64-step sync included. Prefill, initial KV migration, compilation, calibration, graph capture, warmup and oracle excluded. Tail is separately measured last64 real decode steps. Only offpath computes; donor has no model.',
        interpretation='Concurrency is the largest identical-context batch that fits the complete1024-step trajectory without preemption. Some contexts leave integer packing slack; actual pool fractions are reported. peer_max uses only the remote prefix needed to fit that maximum batch, up to the fixed remote allocation. If no extra request fits, it uses zero remote pages and the original native attention fast path. peer_same_batch deliberately uses the full remote allowance to isolate tier-placement cost. unified_local_max separates custom-kernel efficiency from capacity benefit. Fixed-batch decode throughput is not open-loop serving throughput, two-active-GPU total throughput or per-GPU cost efficiency.')
    (root/'throughput_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    plot(root,table)
    print(json.dumps(result,indent=2))


def plot(root,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'STIXGeneral','font.size':11,'pdf.fonttype':42,'svg.fonttype':'none',
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,1,figsize=(12,7.4),sharex=True,gridspec_kw={'height_ratios':[1.4,1]})
    x=np.arange(len(rows));w=.19
    for offset,key,label,color in [(-1.5,'native_tokens_s','Native / maximum local batch','#92928b'),
                                  (-.5,'unified_local_tokens_s','Optimized / local','#b8cce0'),
                                  (.5,'peer_same_batch_tokens_s','Peer / same batch','#65b4ad'),
                                  (1.5,'peer_max_tokens_s','Peer / maximum batch','#2e7bd6')]:
        axes[0].bar(x+offset*w,[r[key] for r in rows],width=w,label=label,color=color)
    axes[0].set(ylabel='Actual decode output tokens/s');axes[0].legend(frameon=False,ncol=2,fontsize=9)
    gains=np.array([100*r['throughput_gain'] for r in rows])
    axes[1].bar(x,gains,color=['#2e7bd6' if v>=0 else '#92928b' for v in gains],width=.55)
    axes[1].scatter(x,[100*r['paired_gain_min'] for r in rows],marker='_',color='black',s=65,zorder=3)
    axes[1].scatter(x,[100*r['paired_gain_max'] for r in rows],marker='_',color='black',s=65,zorder=3)
    axes[1].axhline(0,color='black',lw=.7)
    axes[1].set(ylabel='Peer maximum vs native gain (%)',xlabel='Cached context before → after1024 true decode steps')
    axes[1].set_xticks(x,[f"{r['initial_context']} → {r['final_context']}\nB {r['native_batch']} → {r['extended_batch']}" for r in rows],fontsize=9)
    for ax in axes:ax.grid(axis='y',color='#deded8',lw=.6);ax.set_axisbelow(True)
    fig.subplots_adjust(left=.08,right=.98,top=.97,bottom=.18,hspace=.16)
    fig.text(.5,.025,'A10040GB · Qwen3-8B BF16 · fixed local KV13.5GiB / peer allocation0.809GiB · idle donor\n3 repetitions · wall time includes sampling and page updates · black ticks show paired trial gain range · decode only',ha='center',fontsize=10,color='#66635d')
    for ext in ('png','pdf','svg'):fig.savefig(root/f'throughput_comparison.{ext}',dpi=240)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
