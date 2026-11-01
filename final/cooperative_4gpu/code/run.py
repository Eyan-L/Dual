"""Two model processes;onpath also owns one of three physical remote banks."""

from pathlib import Path as _FinalPath
_SLACK_FINAL_ROOT = next(p for p in _FinalPath(__file__).resolve().parents if (p / 'release.json').is_file())
_SLACK_PROJECT_ROOT = _SLACK_FINAL_ROOT.parent
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time
import traceback
HERE=Path(__file__).resolve().parent
for path in [str(_SLACK_FINAL_ROOT / 'vendor/nvlink_cooperation/launch_20261030_r1/code'),
 str(_SLACK_FINAL_ROOT / 'vendor/nvlink_throughput/launch_20261030_r1/code'),
 str(_SLACK_FINAL_ROOT / 'vendor/nvlink_prefill/launch_20261031_r5/code'),
 str(_SLACK_FINAL_ROOT / 'vendor/nvlink_flashinfer_interleaved_decode/launch_20261101_r3/code'),str(HERE/'peer'),str(HERE)]:sys.path.insert(0,path)
from worker import save


def worker(role,cfg,root,connections,stop):
    folder=root/('gpu0'if role==0 else'gpu1')if role in [0,3]else root
    log=folder/'worker.log'if role in [0,3]else root/f'owner_{role}.log'
    with log.open('w',buffering=1)as stream:
        os.dup2(stream.fileno(),1);os.dup2(stream.fileno(),2)
        try:
            import torch
            from host4 import receive
            torch.cuda.set_device(role);owner=None
            if role in cfg['cooperation']['donor_devices']:
                from cuda_ipc import Cuda
                from measure import mapped_tensor
                cuda=Cuda();cuda.set_device(role);slots=cfg['cooperation']['remote_capacity_per_donor'];address,handle=cuda.allocate(slots*147456)
                tensor=mapped_tensor(address,slots,role);tensor.zero_();tensor.flatten()[0].fill_(role+1.5);torch.cuda.synchronize()
                connections.send(dict(type='allocation',owner_device=role,slots=slots,bytes=slots*147456,handle=handle,owner=cuda.attributes(address)))
                owner=dict(cuda=cuda,address=address,tensor=tensor)
            if role in [0,3]:
                from worker import model
                return model(role,cfg,folder,root,connections,stop,owner)
            assert receive(connections,stop)['type']=='probe_written'
            assert float(tensor.flatten()[0].item())==role+2.25;connections.send(dict(type='verified'))
            assert receive(connections,stop)['type']=='closed'
            cuda.free(address);connections.send(dict(type='freed'))
            save(root/f'owner_{role}.json',dict(owner=role,model_loaded=False,model_forwards=0,released=True))
        except BaseException:
            save(folder/f'failure_{role}.json',dict(error=traceback.format_exc()));stop.set();raise


def main(a):
    cfg=json.loads(a.config.read_text());a.out.mkdir(parents=True,exist_ok=False)
    for g in [0,1]:(a.out/f'gpu{g}').mkdir()
    save(a.out/'workload.json',cfg)
    from host4 import topology
    topology(a.out,cfg['cooperation'])
    ctx=mp.get_context('spawn');stop=ctx.Event();pairs=[ctx.Pipe(duplex=True)for _ in range(3)]
    processes=[ctx.Process(target=worker,args=(role,cfg,a.out,pairs[role][1],stop))for role in [0,1,2]]
    processes.append(ctx.Process(target=worker,args=(3,cfg,a.out,[p[0]for p in pairs],stop)))
    for p in processes:p.start()
    for pair in pairs:
        for conn in pair:conn.close()
    def healthy():
        if stop.is_set()or any(p.exitcode not in [None,0]for p in processes):raise RuntimeError('Cooperation worker failed;see worker logs')
    try:
        while not all((a.out/f'gpu{g}/ready.json').exists()for g in [0,1]):healthy();time.sleep(.5)
        save(a.out/'start.json',dict(perf_ns=time.perf_counter_ns()+2_000_000_000,unix_ns=time.time_ns()+2_000_000_000))
        while not all((a.out/f'gpu{g}/foreground_done.json').exists()for g in [0,1]):healthy();time.sleep(.01)
        save(a.out/'stop_background.json',dict(perf_ns=time.perf_counter_ns()))
        while not all((a.out/f'gpu{g}/result.json').exists()for g in [0,1]):healthy();time.sleep(.1)
        save(a.out/'release.json',dict(release=True))
        for p in processes:p.join()
        healthy();assert all(p.exitcode==0 for p in processes)
        r=[json.loads((a.out/f'gpu{g}/result.json').read_text())for g in [0,1]]
        assert len([a for v in r for a in v['agents']])==5
        save(a.out/'review.json',dict(status='COOPERATIVE_TRACE_JOIN_COMPLETE',gpu_results=r,scope=cfg['scope'],config=cfg))
    finally:
        stop.set()
        for p in processes:
            if p.is_alive():p.terminate()
        for p in processes:p.join()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
