"""Launch two independent native TP1 models;one shared monotonic start clock."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def save(p,v):
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)


def main(a):
    a.out.mkdir(parents=True,exist_ok=False);cfg=json.loads(a.config.read_text());save(a.out/'workload.json',cfg)
    devices=os.environ['CUDA_VISIBLE_DEVICES'].split(',');assert len(devices)==2,devices
    processes=[];logs=[]
    for gpu in [0,1]:
        folder=a.out/f'gpu{gpu}';folder.mkdir();log=(folder/'worker.log').open('w');logs.append(log)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=devices[gpu],OMP_NUM_THREADS='4',SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN='1')
        processes.append(subprocess.Popen([sys.executable,'-B','-u',str(Path(__file__).with_name('worker.py')),'--config',str(a.config),'--gpu',str(gpu),'--out',str(folder)],env=env,stdout=log,stderr=subprocess.STDOUT))
    try:
        while not all((a.out/f'gpu{g}/ready.json').exists() for g in [0,1]):
            if any(p.poll() is not None for p in processes):raise RuntimeError('Workerfailedbeforebarrier')
            time.sleep(1)
        save(a.out/'start.json',dict(perf_ns=time.perf_counter_ns()+2_000_000_000,unix_ns=time.time_ns()+2_000_000_000))
        while not all((a.out/f'gpu{g}/foreground_done.json').exists() for g in [0,1]):
            if any(p.poll() not in [None,0] for p in processes):raise RuntimeError('Workerfailed')
            time.sleep(.01)
        save(a.out/'stop_background.json',dict(perf_ns=time.perf_counter_ns()))
        while not all((a.out/f'gpu{g}/result.json').exists() for g in [0,1]):
            if any(p.poll() not in [None,0] for p in processes):raise RuntimeError('Workerfailed')
            time.sleep(1)
        save(a.out/'release.json',dict(release=True))
        for p in processes:p.wait()
        assert all(p.returncode==0 for p in processes)
        results=[json.loads((a.out/f'gpu{g}/result.json').read_text()) for g in [0,1]]
        agents=[n for r in results for n in r['agents']];assert len(agents)==5
        groups=[]
        for group in ['element','vuls']:
            branches=[n for n in agents if n['group']==group];end=max(n['finish_ms'] for n in branches)
            groups.append(dict(group=group,join_wall_ms=end,tail_agent=max(branches,key=lambda n:n['finish_ms'])['id'],
                completion_spread_ms=end-min(n['finish_ms'] for n in branches),branches=branches))
        report=dict(status='NATIVE_CAPACITY_CONTROL_COMPLETE',groups=groups,placement=cfg['placement'],gpu_results=results,
            source_workload_sha256=hashlib.sha256(a.config.read_bytes()).hexdigest(),scope=cfg['scope'],
            context_scope=cfg['scope'])
        save(a.out/'review.json',report);print(json.dumps(dict(status=report['status'],groups=groups),indent=2))
    finally:
        if any(p.poll() is None for p in processes):
            save(a.out/'stop.json',dict(stop=True))
            for p in processes:
                if p.poll() is None:p.terminate()
            for p in processes:
                try:p.wait(timeout=20)
                except subprocess.TimeoutExpired:p.kill();p.wait()
        for log in logs:log.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
