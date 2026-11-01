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
 r=json.loads((root/'analysis/review.json').read_text());assert r['status']=='DUAL_BACKGROUND_TRACE_VALIDATED'
 work=Path('/mnt/c/Users/24165/AppData/Local/Temp/slack_trace_background_20261101');work.mkdir(exist_ok=True)
 win='C:\\Users\\24165\\AppData\\Local\\Temp\\slack_trace_background_20261101'
 tag='dual_trace_background_'+root.name.split('.')[0]
 cards=r['cards'];groups={g['group']:g['join_s']for g in r['groups']}
 shapes=[text(28,17,910,37,'双卡独立运行｜5 个 subagent + 持续背景请求',25,bold=True),
  text(28,58,910,23,'5 个任务在 t=0 提交；背景每卡 1 req/s（非 critical trace，约 11K–12K 输入）；五个全部结束后停止',14,color='#666666')]
 for c in cards:
  x=28+c['gpu']*462;color=['#3376A5','#B47A37'][c['gpu']]
  shapes.extend([dict(type='rect',x=x,y=87,w=442,h=43,color='#EEF3F7'),
   text(x+10,92,422,23,f'GPU{c["gpu"]}：{c["subagents_completed"]} 个 subagent + {c["background_completed"]} 个背景请求',17,color=color,bold=True),
   text(x+10,114,422,16,f'完成输出吞吐 {c["completed_output_tokens_per_second"]:.1f} tok/s',11,color='#555555')])
 shapes.extend([
  dict(type='image',path=win+'\\gpu_overview.png',x=28,y=136,w=904,h=348),
  dict(type='rect',x=28,y=486,w=904,h=32,color='#EEF3F7'),
  text(40,491,885,24,f'总吞吐：{r["total_completed_output_tokens_per_second"]:.1f} tok/s  /  {r["total_completed_calls_per_second"]:.2f} req/s    ｜    共同测量窗口 {r["measurement_seconds"]:.1f} s',17,bold=True),
  text(28,522,910,15,f'仅计完成调用输出；前台零撤回 / 零 KV 等待；背景撤回 {cards[0]["background_retractions"]}/{cards[1]["background_retractions"]} 次，未完成 {cards[0]["background_unfinished"]}/{cards[1]["background_unfinished"]}。KV=active/98,304。{root.name}',8.5,color='#777777')])
 notes=json.dumps(r,ensure_ascii=False,indent=2)+'\n\n'+(root/'measurement/workload.json').read_text()
 notes+='\n\nSlide 30 replaces the old no-background result only; all other live deck slides and unsaved edits preserved. Total req/s counts 66 foreground model calls plus completed single-call trace-background requests, not five agents. Plots share common t0 and stop. Top bars show whole subagent wall time incl tool waits; bottom dots are individual completed background requests plotted at completion time; Y is E2E since planned arrival, including queue delay. KV samples are raw loop boundaries, without smoothing. Full foreground IDs and detailed per-call/per-background completion timings are in analysis/branches.csv, calls.csv and background.csv. Element join=%.3fs; Vuls join=%.3fs.'%(groups['element'],groups['vuls'])
 spec=dict(width=959.95,height=540,slides=[dict(title='双卡独立运行：真实 subagent + 连续背景请求',notes=notes,shapes=shapes)])
 build=Path(str(_SLACK_FINAL_ROOT / 'reporting/powerpoint_build.ps1')).read_text(encoding='utf-8-sig')
 build=build.replace('slack_serving_multiturn_native8_26075216','slack_trace_background_20261101').replace('flashinfer_multiturn_native8_26075216',tag)
 build=build.replace('$deck.Close()',"try{$deck.Close()}catch{Write-Output 'Staging saved;temporary source close cleanup failed.'}")
 replace=Path(__file__).with_name('replace_slide30.ps1').read_text(encoding='utf-8-sig').replace('__TAG__',tag)
 dest=root/'presentation';dest.mkdir(exist_ok=True)
 for folder in [dest,work]:
  (folder/'slides.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2)+'\n')
  (folder/'build.ps1').write_text(build,encoding='utf-8-sig')
  (folder/'replace_slide30.ps1').write_text(replace,encoding='utf-8-sig')
 shutil.copy2(root/'analysis/gpu_overview.png',work/'gpu_overview.png')
 print(work)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
