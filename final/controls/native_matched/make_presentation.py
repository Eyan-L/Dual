"""One comparison slide for the native capacity-matched control."""

from pathlib import Path as _FinalPath
_SLACK_FINAL_ROOT = next(p for p in _FinalPath(__file__).resolve().parents if (p / 'release.json').is_file())
_SLACK_PROJECT_ROOT = _SLACK_FINAL_ROOT.parent
import argparse
import json
from pathlib import Path
import shutil


def text(x,y,w,h,value,size=18,color='#333333',bold=False):
 return dict(type='text',x=x,y=y,w=w,h=h,text=value,size=size,color=color,bold=bold,align=1)


def main(root):
 import numpy as np
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.lines import Line2D
 r=json.loads((root/'analysis/review.json').read_text());assert r['status']=='NATIVE_CAPACITY_CONTROL_VALIDATED'
 old=json.loads(Path(str(_SLACK_PROJECT_ROOT / 'results' / '05_trace_serving' / 'dual_trace_background/26078142/analysis/review.json')).read_text())
 peer=json.loads(Path(str(_SLACK_PROJECT_ROOT / 'results' / '05_trace_serving' / 'cooperative_trace_background/26079252/analysis/review.json')).read_text())
 groups={g['group']:g['join_s']for g in r['groups']};pg={g['group']:g['join_s']for g in peer['groups']}
 gain=(r['total_completed_output_tokens_per_second']/peer['total_completed_output_tokens_per_second']-1)*100
 delta={k:(groups[k]/pg[k]-1)*100 for k in groups}
 origin=json.loads((root/'measurement/start.json').read_text())['perf_ns'];duration=r['measurement_seconds'];capacity=98304
 colors=['#aeb4ba','#bc884d','#3376a5']
 plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(2,2,figsize=(13.4,5.1),gridspec_kw={'height_ratios':[1.12,1]})
 a=axes[0,0];values=[old['total_completed_output_tokens_per_second'],peer['total_completed_output_tokens_per_second'],r['total_completed_output_tokens_per_second']]
 labels=['Independent 100 / 100','Cooperative 80 / 117.35','All-local 80 / 117.35']
 for i,(value,color)in enumerate(zip(values,colors)):
  a.barh(i,value,color=color,height=.58);a.text(value+4,i,f'{value:.2f}',va='center',fontsize=10)
 a.set_yticks(range(3),labels,fontsize=9);a.invert_yaxis();a.set_xlim(0,max(values)*1.19)
 a.set_xlabel('Completed-output throughput (tokens/s)',fontsize=9);a.set_title('Total throughput | one run per configuration',loc='left',fontsize=11);a.grid(axis='x',alpha=.16)
 a=axes[0,1]
 order=['element__reproduce_markdown_bug','vuls__verify_behavior','element__locate_markdown_logic','element__locate_markdown_tests','vuls__verify_patch_tests']
 names=['Element / reproduce','Vuls / behavior','Element / logic','Element / tests','Vuls / patch tests']
 for i,ident in enumerate(order):
  before=next(b['completion_s']for b in peer['branches']if b['agent']==ident)
  after=next(b['completion_s']for b in r['branches']if b['agent']==ident)
  a.plot([before,after],[i,i],color='#999999',lw=1.4)
  a.scatter([before],[i],s=27,color=colors[1],zorder=3);a.scatter([after],[i],s=27,color=colors[2],zorder=4)
  a.text(max(before,after)+6,i,f'{before:.1f} / {after:.1f}',va='center',fontsize=8)
 a.set_yticks(range(5),names,fontsize=9);a.set_ylim(4.6,-.6);a.set_xlim(0,410);a.set_xlabel('Completion from shared submission (s)',fontsize=9)
 a.set_title('Five agents | cooperative / all-local',loc='left',fontsize=11);a.grid(axis='x',alpha=.16)
 a.legend(handles=[Line2D([0],[0],marker='o',color=colors[1],ls='',label='Cooperative'),Line2D([0],[0],marker='o',color=colors[2],ls='',label='All-local')],loc='upper left',fontsize=7,frameon=False,ncol=2)
 for gpu in [0,1]:
  a=axes[1,gpu];card=r['cards'][gpu]
  with (root/f'measurement/gpu{gpu}/kv_events.jsonl').open()as f:rows=[json.loads(line)for line in f if line.strip()]
  rows=[v for v in rows if origin<=v['perf_ns']<=origin+duration*1e9]
  t=np.array([0]+[(v['perf_ns']-origin)/1e9 for v in rows]+[duration]);fg=np.array([0]+[v['foreground_tokens']/capacity*100 for v in rows]+[rows[-1]['foreground_tokens']/capacity*100]);bg=np.array([0]+[v['background_tokens']/capacity*100 for v in rows]+[rows[-1]['background_tokens']/capacity*100])
  a.fill_between(t,0,fg,step='post',color=colors[2],alpha=.35);a.fill_between(t,fg,fg+bg,step='post',color='#c5c9cd',alpha=.65)
  a.step(t,fg+bg,where='post',color='#222222',lw=.8)
  a.axhline(card['physical_capacity_tokens']/capacity*100,color='#888888',ls=':',lw=.8)
  a.axhline(100,color='#aaaaaa',ls='--',lw=.6);a.set_xlim(0,duration*1.03);a.set_ylim(0,125);a.set_yticks([0,50,100])
  a.set_xlabel('Time from t0 (s)',fontsize=9);a.set_ylabel('Active KV / C (%)',fontsize=9);a.grid(alpha=.13)
  a.set_title(f'{["Onpath","Offpath"][gpu]} | {card["subagents_completed"]} agents + {card["background_completed"]} BG completed | peak {card["peak_kv_percent"]:.2f}%',loc='left',fontsize=10)
 from matplotlib.patches import Patch
 fig.legend(handles=[Patch(color=colors[2],alpha=.35,label='Original agents KV'),Patch(color='#c5c9cd',label='Background KV'),Line2D([0],[0],color='#222222',lw=.8,label='Total active KV')],loc='lower center',bbox_to_anchor=(.54,.003),frameon=False,ncol=3,fontsize=8)
 fig.subplots_adjust(left=.15,right=.965,bottom=.145,top=.91,wspace=.53,hspace=.68)
 fig.savefig(root/'analysis/capacity_comparison.png',dpi=220);fig.savefig(root/'analysis/capacity_comparison.pdf');plt.close(fig)
 win=Path('/mnt/c/Users/24165/AppData/Local/Temp/slack_trace_capacity_20261101');win.mkdir(exist_ok=True)
 ws='C:\\Users\\24165\\AppData\\Local\\Temp\\slack_trace_capacity_20261101';tag='native_capacity_control_'+root.name.split('.')[0]
 shapes=[text(28,17,910,37,f'等容量全本地对照｜总吞吐比合作高 {gain:.2f}%',24,bold=True),
  text(28,59,910,24,'两卡原生 FlashInfer；KV 上限 78,643 / 115,365 tokens；相同 5 个任务、相同背景流量',14,color='#666666'),
  dict(type='image',path=ws+'\\capacity_comparison.png',x=28,y=88,w=904,h=354),
  dict(type='rect',x=28,y=456,w=904,h=54,color='#EEF3F7'),
  text(40,462,880,24,f'合作 → 全本地：吞吐 {peer["total_completed_output_tokens_per_second"]:.2f} → {r["total_completed_output_tokens_per_second"]:.2f} tok/s（+{gain:.2f}%）',18,bold=True),
  text(40,489,880,18,f'Element {pg["element"]:.2f} → {groups["element"]:.2f}s（+{delta["element"]:.2f}%）；Vuls {pg["vuls"]:.2f} → {groups["vuls"]:.2f}s（+{delta["vuls"]:.2f}%）',13),
  text(28,522,910,15,f'单次实测；前台零撤回 / 零 KV 等待；吞吐仅计完成调用输出。等数据 KV 容量；原生与合作还包含 kernel / 页表 / 迁移 / 暂存差异。{root.name}',8.5,color='#777777')]
 notes=json.dumps(r,ensure_ascii=False,indent=2)+'\n\n'+(root/'measurement/workload.json').read_text()
 notes+='\n\nAll-local and cooperative data KV caps exactly78643/115365. C=98304;117.35pct is shorthand for117.35534668pct. All-local no IPC,no remoteKV,no staging,no customattention. Native100/100 reference has balanced placement;the80/117 controls share the critical/offpath placement. Allfiveagents match lengths,toolwaits,launch0;all16background templates and1req/s/GPU arrivals match. Stop horizons differ by prescribed rule after allfivefinish. Throughput counts completed-call outputs;partial background output excluded,not steady-state/raw emitted throughput. One run/config,no significance claim for1pct differences. Comparison isolates the whole tiered adapter/staging/migration/layout combination against native,not just NVLink. PriorfourGPU run has three sourceHBMcontrollers/twoidle helpers;not exact realdualNV12. Full CSV and raw samples retained.'
 spec=dict(width=959.95,height=540,slides=[dict(title='同容量全本地对照：吞吐与长尾',notes=notes,shapes=shapes)])
 build=Path(str(_SLACK_FINAL_ROOT / 'reporting/powerpoint_build.ps1')).read_text(encoding='utf-8-sig')
 build=build.replace('slack_serving_multiturn_native8_26075216','slack_trace_capacity_20261101').replace('flashinfer_multiturn_native8_26075216',tag)
 build=build.replace('$deck.Close()',"try{$deck.Close()}catch{Write-Output 'Staging saved;temporary source close cleanup failed.'}")
 append=Path(str(_SLACK_FINAL_ROOT / 'reporting/powerpoint_append.ps1')).read_text(encoding='utf-8-sig')
 append=append.replace('slack_trace_join_baseline_20261101','slack_trace_capacity_20261101').replace('dual_trace_join_baseline_26077464',tag)
 out=root/'presentation';out.mkdir(exist_ok=True)
 for folder in [win,out]:
  (folder/'slides.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2)+'\n');(folder/'build.ps1').write_text(build,encoding='utf-8-sig');(folder/'append.ps1').write_text(append,encoding='utf-8-sig')
 shutil.copy2(root/'analysis/capacity_comparison.png',win/'capacity_comparison.png')
 print(win)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
