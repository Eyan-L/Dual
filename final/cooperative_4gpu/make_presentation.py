"""Prepare one concise replacement for existing meeting slide 30."""

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
 r=json.loads((root/'analysis/review.json').read_text());assert r['status']=='COOPERATIVE_TRACE_VALIDATED'
 work=Path('/mnt/c/Users/24165/AppData/Local/Temp/slack_trace_cooperation_20261101');work.mkdir(exist_ok=True)
 win='C:\\Users\\24165\\AppData\\Local\\Temp\\slack_trace_cooperation_20261101'
 tag='cooperative_trace_background_'+root.name.split('.')[0]
 cards=r['cards'];groups={g['group']:g['join_s']for g in r['groups']};before={g['group']:g['join_s']for g in r['comparison']['independent_groups']}
 shapes=[text(28,17,910,37,'合作模式｜长尾缩短 15–20%，总吞吐降低 6.7%',24,bold=True),
  text(28,58,910,23,'同 5 个任务、同背景流量；onpath 自用 ≤80%C，offpath 本地优先、remote/local <17.97%',14,color='#666666')]
 for c in cards:
  x=28+c['gpu']*462;color=['#3376A5','#B47A37'][c['gpu']]
  shapes.extend([dict(type='rect',x=x,y=87,w=442,h=43,color='#EEF3F7'),
   text(x+10,92,422,23,f'{["Onpath","Offpath"][c["gpu"]]}：{c["subagents_completed"]} 个 subagent + {c["background_completed"]} 个背景请求',17,color=color,bold=True),
   text(x+10,114,422,16,f'完成输出吞吐 {c["completed_output_tokens_per_second"]:.1f} tok/s · 共同窗口 {r["measurement_seconds"]:.1f}s',11,color='#555555')])
 shapes.extend([
  dict(type='image',path=win+'\\gpu_overview.png',x=28,y=136,w=904,h=348),
  dict(type='rect',x=28,y=486,w=904,h=32,color='#EEF3F7'),
  text(40,491,885,24,f'Element：{before["element"]:.1f}→{groups["element"]:.1f}s   Vuls：{before["vuls"]:.1f}→{groups["vuls"]:.1f}s   完成输出吞吐 {r["comparison"]["independent_output_tokens_per_second"]:.1f}→{r["total_completed_output_tokens_per_second"]:.1f} tok/s',15,bold=True),
  text(28,522,910,15,f'前台零撤回 / 零 KV 等待；4 卡仅 2 个模型，全部 remote 计入逻辑 onpath 剩余容量；3×NV4 聚合不等价于真实双卡 NV12。{root.name}',8.5,color='#777777')])
 notes=json.dumps(r,ensure_ascii=False,indent=2)+'\n\n'+(root/'measurement/workload.json').read_text()
 notes+='\n\nAppend one cooperation result after existing slide30; preserve all live slides and unsaved edits. Comparison changes placement/partition as authorized;not an isolated transport control. Two source-helper GPUs have no model forwards;their entire remote storage is charged against one logical onpath pool. Stage75MiB is charged as512 offpath KV slots. Total req/s counts 66 foreground model calls plus completed single-call trace-background requests, not five agents. Plots share common t0 and stop. Top bars show whole subagent wall time incl tool waits; bottom dots are individual completed background requests plotted at completion time; Y is E2E since planned arrival, including queue delay. KV samples are raw loop boundaries, without smoothing. Full foreground IDs and detailed per-call/per-background completion timings are in analysis/branches.csv, calls.csv and background.csv. Element join=%.3fs; Vuls join=%.3fs.'%(groups['element'],groups['vuls'])
 spec=dict(width=959.95,height=540,slides=[dict(title='合作模式：原始任务与匹配背景流量',notes=notes,shapes=shapes)])
 build=Path(str(_SLACK_FINAL_ROOT / 'reporting/powerpoint_build.ps1')).read_text(encoding='utf-8-sig')
 build=build.replace('slack_serving_multiturn_native8_26075216','slack_trace_cooperation_20261101').replace('flashinfer_multiturn_native8_26075216',tag)
 build=build.replace('$deck.Close()',"try{$deck.Close()}catch{Write-Output 'Staging saved;temporary source close cleanup failed.'}")
 replace=Path(str(_SLACK_FINAL_ROOT / 'reporting/powerpoint_append.ps1')).read_text(encoding='utf-8-sig')
 replace=replace.replace('slack_trace_join_baseline_20261101','slack_trace_cooperation_20261101').replace('dual_trace_join_baseline_26077464',tag)
 dest=root/'presentation';dest.mkdir(exist_ok=True)
 for folder in [dest,work]:
  (folder/'slides.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2)+'\n')
  (folder/'build.ps1').write_text(build,encoding='utf-8-sig')
  (folder/'append.ps1').write_text(replace,encoding='utf-8-sig')
 shutil.copy2(root/'analysis/gpu_overview.png',work/'gpu_overview.png')
 print(work)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
