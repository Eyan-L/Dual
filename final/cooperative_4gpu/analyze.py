"""Review one five-agent/background run and plot the shared observation window."""

from pathlib import Path as _FinalPath
_SLACK_FINAL_ROOT = next(p for p in _FinalPath(__file__).resolve().parents if (p / 'release.json').is_file())
_SLACK_PROJECT_ROOT = _SLACK_FINAL_ROOT.parent
import argparse
import csv
import json
from pathlib import Path

NAMES={
 'element__locate_markdown_logic':'Element / logic',
 'element__locate_markdown_tests':'Element / tests',
 'element__reproduce_markdown_bug':'Element / reproduce',
 'vuls__verify_patch_tests':'Vuls / patch tests',
 'vuls__verify_behavior':'Vuls / behavior',
}

def load(p):return json.loads(p.read_text())
def lines(p):
 with p.open() as f:
  for line in f:
   if line.strip():yield json.loads(line)
def table(path,rows):
 if not rows:return
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main(root):
 import numpy as np
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.lines import Line2D
 from matplotlib.patches import Patch
 assert load(root/'pbs_exit.json')['exit_code']==0
 m=root/'measurement';cfg=load(m/'workload.json');report=load(m/'review.json')
 assert report['status']=='COOPERATIVE_TRACE_JOIN_COMPLETE'
 origin=load(m/'start.json')['perf_ns'];cutoff=load(m/'stop_background.json')['perf_ns'];duration=(cutoff-origin)/1e9
 assert duration>0 and cfg['background']['warmup_seconds']==0
 assert all(a['arrival_offset_ms']==0 for a in cfg['agents'])
 source={a['id']:a for a in cfg['agents']};capacity=cfg['kv_capacity_tokens'];out=root/'analysis';out.mkdir(exist_ok=True)
 branches=[];calls=[];cards=[];background=[];series=[]
 for gpu in [0,1]:
  folder=m/f'gpu{gpu}';r=load(folder/'result.json');physical=cfg['cooperation']['onpath_own_capacity_tokens']if gpu==0 else cfg['cooperation']['offpath_physical_capacity_tokens']
  logical=physical if gpu==0 else cfg['cooperation']['offpath_logical_capacity_tokens']
  assert r['final_free_tokens']==physical and r['max_used_tokens']<=logical
  if gpu==1:
   assert r['remote_tokens']>0
   operations=load(folder/'peer_operations.json')
   assert not operations['primary_query_copies'] and operations['stage_buffer_bytes']==cfg['cooperation']['stage_buffer_bytes']
   assert operations['final']['logical_tokens']==0
   checks=load(folder/'protected_protocol/peer_operations.json')
   assert checks['decode_checks'] and checks['prefill_checks']
  assert r['origin_perf_ns']==origin and r['background_summary']['cutoff_perf_ns']==cutoff
  assert r['foreground_retractions']==r['memory_queue_events']==0
  assert load(folder/'ready.json')['protected_foreground_protocol_passed']
  protocol=load(folder/'protected_protocol/result.json')
  assert protocol['background_summary']['retractions']>0 and protocol['background_summary']['recomputed_tokens']>0
  own=[a for a in cfg['agents'] if a['gpu']==gpu]
  assert {a['id']for a in own}=={a['id']for a in r['agents']}
  for a in r['agents']:
   s=source[a['id']];assert a['calls']==len(s['calls']) and a['output_tokens']==sum(c['output_tokens']for c in s['calls'])
   assert 0<=a['first_start_ms']<=a['finish_ms']<=duration*1000
   branches.append(dict(gpu=gpu,agent=a['id'],label=NAMES[a['id']],group=a['group'],calls=a['calls'],
    first_dispatch_s=a['first_start_ms']/1000,completion_s=a['finish_ms']/1000,output_tokens=a['output_tokens']))
  for c in r['calls']:
   s=source[c['agent']]['calls'][c['round']]
   assert (s['input_tokens'],s['output_tokens'])==(c['input_tokens'],c['output_tokens'])
   assert c['context_at_end']==c['input_tokens']+c['output_tokens']-1
   calls.append(dict(gpu=gpu,**c))
  raw=list(lines(folder/'background.jsonl'));arrivals=list(lines(folder/'arrivals.jsonl'))
  assert len({b['id']for b in raw})==len(raw)==r['background_summary']['arrived']
  assert len(arrivals)==len(raw)
  planned=sorted(b['planned_ns']for b in arrivals)
  expected_first=origin+round(cfg['background']['first_arrival_seconds']*1e9);interval=round(cfg['background']['arrival_interval_seconds']*1e9)
  assert all(t==expected_first+i*interval for i,t in enumerate(planned))
  completed=[b for b in raw if b['status']=='completed' and b['finished_ns']<=cutoff]
  censored=[b for b in raw if b not in completed]
  for b in raw:
   template=cfg['background']['templates'][int(b['id'].split('_')[-1])%len(cfg['background']['templates'])]
   assert (b['input_tokens'],b['output_tokens'],b['source_agent'],b['source_round'])==(template['input_tokens'],template['output_tokens'],template['source_agent'],template['source_round'])
   valid=b['status']=='completed' and b['finished_ns']<=cutoff
   background.append(dict(gpu=gpu,request=b['id'],planned_s=(b['planned_ns']-origin)/1e9,
    source_agent=b['source_agent'],source_round=b['source_round'],input_tokens=b['input_tokens'],output_tokens=b['output_tokens'],retractions=b['retractions'],recomputed_tokens=b['recomputed_tokens'],
    admitted_s=(b['admitted_ns']-origin)/1e9 if 'admitted_ns'in b else '',status='completed'if valid else'unfinished',
    finished_s=(b['finished_ns']-origin)/1e9 if valid else'',latency_s=b['e2e_ms']/1000 if valid else'',
    queue_s=b['queue_ms']/1000 if valid else'',censored_age_s=''if valid else max(0,(cutoff-b['planned_ns'])/1e9)))
  kv=list(lines(folder/'kv_events.jsonl'));assert kv and all(0<=k['used_tokens']<=logical for k in kv)
  assert all(k['used_tokens']==k['foreground_tokens']+k['background_tokens']for k in kv)
  assert all(k['used_tokens']==k['local_active_tokens']+k['remote_active_tokens']and k['local_active_tokens']<=physical for k in kv)
  assert all(k['remote_active_tokens']==0 or 10000*k['remote_active_tokens']<1797*k['local_active_tokens']for k in kv)
  assert all(k['foreground_decode_ready']==k['foreground_decode_executed'] and k['foreground_memory_waits']==0 for k in kv)
  assert all(a['perf_ns']<=b['perf_ns']for a,b in zip(kv,kv[1:]))
  k=[k for k in kv if origin<=k['perf_ns']<=cutoff];assert k
  t=np.array([(v['perf_ns']-origin)/1e9 for v in k]);fg=np.array([v['foreground_tokens']/capacity*100 for v in k]);bg=np.array([v['background_tokens']/capacity*100 for v in k])
  series.append((t,fg,bg,np.array([v['remote_active_tokens']/capacity*100 for v in k])))
  fgout=sum(a['output_tokens']for a in r['agents']);bgout=sum(b['output_tokens']for b in completed)
  latency=[b['e2e_ms']/1000 for b in completed]
  cards.append(dict(gpu=gpu,subagents_completed=len(r['agents']),foreground_calls_completed=len(r['calls']),
   background_arrived=len(raw),background_completed=len(completed),background_unfinished=len(censored),
   background_never_admitted_at_stop=sum('admitted_ns'not in b for b in censored),
   foreground_retractions=0,foreground_memory_waits=0,background_retractions=r['background_summary']['retractions'],
   background_requests_retracted=sum(b['retractions']>0 for b in raw),background_recomputed_tokens=r['background_summary']['recomputed_tokens'],
   completed_foreground_output_tokens=fgout,completed_background_output_tokens=bgout,
   completed_output_tokens=fgout+bgout,completed_output_tokens_per_second=(fgout+bgout)/duration,
   completed_calls_per_second=(len(r['calls'])+len(completed))/duration,
   background_latency_p50_s=float(np.median(latency))if latency else None,
   background_latency_p95_s=float(np.percentile(latency,95))if latency else None,
   peak_kv_tokens=max(v['used_tokens']for v in k),peak_kv_percent=max(fg+bg),
   local_capacity_tokens=physical,logical_capacity_tokens=logical,peak_remote_tokens=max(v['remote_active_tokens']for v in k),
   peak_local_tokens=max(v['local_active_tokens']for v in k),peak_foreground_tokens=max(v['foreground_tokens']for v in k),
   peak_active_remote_local_ratio=max(v['remote_active_tokens']/max(1,v['local_active_tokens'])for v in k),
   max_queued_background=max(v['queued_background']for v in k),kv_samples=len(k),
   max_kv_sample_gap_s=float(np.max(np.diff(t)))if len(t)>1 else 0,
   worker_background_completed_after_cutoff=sum(b['status']=='completed'and b['finished_ns']>cutoff for b in raw)))
 assert len(branches)==5 and len(calls)==66
 groups=[dict(group=g,join_s=max(a['completion_s']for a in branches if a['group']==g))for g in ['element','vuls']]
 final=dict(status='COOPERATIVE_TRACE_VALIDATED',job_id=root.name,measurement_seconds=duration,
  latest_foreground_finish_seconds=max(a['completion_s']for a in branches),groups=groups,cards=cards,branches=branches,
  completed_output_tokens=sum(c['completed_output_tokens']for c in cards),
  total_completed_output_tokens_per_second=sum(c['completed_output_tokens_per_second']for c in cards),
  total_completed_calls_per_second=sum(c['completed_calls_per_second']for c in cards),
  throughput_definition='Completed-call output tokens (66 foreground calls plus completed variable-length trace-background calls) divided by one shared t0-to-stop window; excludes partial output of unfinished background requests. Not raw GPU generated-token throughput.',
  timing_definition='Five foreground subagents submitted at common t0; actual first dispatch recorded separately. Agent completion includes all its model calls, queue/capacity waits and recorded tool waits. Background E2E begins at scheduled arrival, including queue delay. Cutoff is shared parent stop after all five complete; rare completions after cutoff excluded.',
  kv_definition='Logical active KV tokens / reference C=98,304 (offpath includes remote), sampled after prefill+decode loop before completed requests are freed; raw loop-boundary samples retained, no 1s smoothing. Both foreground and background shown; preallocated pool is not active usage.',
  limitations='One execution. Two compute models on four physical A100s. Three source HBM controllers and two idle owners do not reproduce an actual single-source NV12 system. Whole-policy comparison changes placement and KV partition together. Only background can retract; migration,restoration,recompute and staging charged to wall time. Valid constant token contents enforce original foreground lengths and I/O ratios, not semantic task success. All FG launch offsets explicitly zero for this run. Background templates are independent length pairs from noncritical traces, not full background agent/DAG replay.')
 for name,rows in [('cards.csv',cards),('branches.csv',branches),('calls.csv',calls),('background.csv',background),('groups.csv',groups)]:table(out/name,rows)
 baseline=load(Path(str(_SLACK_PROJECT_ROOT / 'results' / '05_trace_serving' / 'dual_trace_background/26078142/analysis/review.json')))
 baseline_workload=load(Path(str(_SLACK_PROJECT_ROOT / 'results' / '05_trace_serving' / 'dual_trace_background/26078142/measurement/workload.json')))
 assert cfg['background']==baseline_workload['background']
 for agent in cfg['agents']:
  original=next(a for a in baseline_workload['agents']if a['id']==agent['id'])
  assert {k:v for k,v in agent.items()if k!='gpu'}=={k:v for k,v in original.items()if k!='gpu'}
 assert all(load(m/f'owner_{owner}.json')['model_forwards']==0 for owner in [1,2])
 comparison=[]
 for a in branches:
  before=next(b for b in baseline['branches']if b['agent']==a['agent'])
  comparison.append(dict(agent=a['agent'],independent_seconds=before['completion_s'],cooperative_seconds=a['completion_s'],
   latency_reduction_percent=(1-a['completion_s']/before['completion_s'])*100))
 final['comparison']=dict(baseline_job=baseline['job_id'],branches=comparison,
  independent_groups=baseline['groups'],cooperative_groups=groups,
  independent_output_tokens_per_second=baseline['total_completed_output_tokens_per_second'],
  cooperative_output_tokens_per_second=final['total_completed_output_tokens_per_second'],
  independent_window_seconds=baseline['measurement_seconds'],cooperative_window_seconds=duration,
  limitation=cfg['cooperation']['hardware_scope'])
 table(out/'comparison.csv',comparison)
 (out/'review.json').write_text(json.dumps(final,indent=2)+'\n')
 plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7})
 fig,axes=plt.subplots(3,2,figsize=(13.4,5.16),gridspec_kw={'height_ratios':[1,1.22,1.05]},sharex=True)
 colors=['#3376a5','#c5853d'];endmax=duration*1.08
 for gpu in [0,1]:
  a=axes[0,gpu];own=[b for b in branches if b['gpu']==gpu]
  for i,b in enumerate(own):
   # Entire branch lifetime: model/scheduling and tools, from shared submission.
   a.barh(i,b['completion_s'],height=.50,color=colors[gpu],alpha=.82)
   a.text(b['completion_s']+duration*.012,i,f'{b["completion_s"]:.1f}s',va='center',fontsize=9)
  a.set_yticks(range(len(own)),[b['label']for b in own],fontsize=9);a.invert_yaxis();a.set_ylim(2.5,-.65)
  a.set_title('Subagent completion from t0',loc='left',fontsize=10,pad=3);a.grid(axis='x',alpha=.18)
  a=axes[1,gpu];t,fg,bg,remote=series[gpu];x=np.r_[0,t,duration];ff=np.r_[0,fg,fg[-1]];bb=np.r_[0,bg,bg[-1]]
  a.fill_between(x,0,ff,step='post',color=colors[gpu],alpha=.35)
  a.fill_between(x,ff,ff+bb,step='post',color='#c5c9cd',alpha=.65)
  a.step(x,ff+bb,where='post',color='#222222',lw=.8)
  if gpu==1:a.step(t,remote,where='post',color='#805b9d',lw=.8,ls='--')
  a.axhline(100,color='#888888',ls='--',lw=.7);a.axhline(80 if gpu==0 else 100,color=colors[gpu],ls=':',lw=.7);a.set_ylim(0,125);a.set_yticks([0,50,100]);a.set_ylabel('KV pool (%)',fontsize=9)
  a.set_title(f'Logical KV / C | peak {cards[gpu]["peak_kv_percent"]:.2f}%',loc='left',fontsize=10,pad=4);a.grid(alpha=.15)
  a=axes[2,gpu];done=[b for b in background if b['gpu']==gpu and b['status']=='completed']
  a.scatter([b['finished_s']for b in done],[b['latency_s']for b in done],s=6,color=colors[gpu],alpha=.7,rasterized=True)
  a.set_ylabel('BG E2E (s)',fontsize=9);a.set_xlabel('Time from common t0 (s)',fontsize=9);a.set_ylim(bottom=0);a.grid(alpha=.18)
  p50=cards[gpu]['background_latency_p50_s'];p95=cards[gpu]['background_latency_p95_s']
  a.set_title(f'BG E2E at completion | P50 {p50:.1f}s / P95 {p95:.1f}s'if p50 is not None else'No completed background requests',loc='left',fontsize=10,pad=4)
  for r in range(3):axes[r,gpu].set_xlim(0,endmax)
 fig.subplots_adjust(left=.125,right=.97,top=.965,bottom=.15,hspace=.60,wspace=.44)
 fig.legend(handles=[Patch(facecolor=colors[0],alpha=.35,label='Foreground KV (GPU color)'),Patch(facecolor='#c5c9cd',label='Background KV'),Line2D([0],[0],color='#222222',lw=.8,label='Total active KV'),Line2D([0],[0],color='#805b9d',lw=.8,ls='--',label='Remote subset (offpath)')],loc='lower center',bbox_to_anchor=(.52,.002),frameon=False,ncol=4,fontsize=8)
 fig.savefig(out/'gpu_overview.png',dpi=220);fig.savefig(out/'gpu_overview.pdf');plt.close(fig)
 print(json.dumps(dict(measurement_seconds=duration,cards=cards,total_completed_output_tokens_per_second=final['total_completed_output_tokens_per_second'],groups=groups),indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
