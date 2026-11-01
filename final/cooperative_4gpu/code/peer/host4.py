"""Four allocated A100s:three immutable CUDAIPC owners and one compute GPU."""

from pathlib import Path as _FinalPath
_SLACK_FINAL_ROOT = next(p for p in _FinalPath(__file__).resolve().parents if (p / 'release.json').is_file())
_SLACK_PROJECT_ROOT = _SLACK_FINAL_ROOT.parent
import argparse
import csv
from io import StringIO
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(_SLACK_FINAL_ROOT / 'vendor/nvlink_cooperation/launch_20261030_r1/code'))
sys.path.insert(0,str(_SLACK_FINAL_ROOT / 'vendor/nvlink_throughput/launch_20261030_r1/code'))
sys.path.insert(0,str(_SLACK_FINAL_ROOT / 'vendor/nvlink_prefill/launch_20261031_r5/code'))
sys.path.insert(0,str(_SLACK_FINAL_ROOT / 'vendor/nvlink_flashinfer_interleaved_decode/launch_20261101_r3/code'))
sys.path.insert(0,str(HERE))
from prototype import save,receive


def topology(root,cfg):
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id,name','--format=csv,noheader'],text=True)
    matrix=subprocess.check_output(['nvidia-smi','topo','-m'],text=True)
    (root/'gpu_inventory.csv').write_text(inventory);(root/'topology.txt').write_text(matrix)
    visible=os.environ['CUDA_VISIBLE_DEVICES'].split(',');assert len(visible)==4
    devices=list(csv.reader(StringIO(inventory)));chosen=[]
    for label in visible:
        found=[r for r in devices if r[0].strip()==label or r[1].strip().startswith(label)]
        assert len(found)==1
        r=found[0];chosen.append(dict(index=int(r[0]),uuid=r[1].strip(),name=r[3].strip()))
    clean=re.sub(r'\x1b\[[0-9;]*m','',matrix)
    header=next(line.split() for line in clean.splitlines() if line.split() and line.split()[0]=='GPU0')
    off=chosen[cfg['offpath_device']]['index']
    row=next(line.split() for line in clean.splitlines() if line.split() and line.split()[0]==f'GPU{off}' and not line.split()[1].startswith('GPU'))
    links={str(d):row[1+header.index(f"GPU{chosen[d]['index']}")] for d in cfg['donor_devices']}
    assert all(v.startswith('NV') for v in links.values()),links
    assert all('A100-SXM4-40GB' in d['name'] for d in chosen)
    save(root/'allocated_topology.json',dict(devices=chosen,visible_devices=visible,compute=cfg['offpath_device'],donor_links=links))


def open_banks(cfg,connections,stop,root):
    import torch
    from cuda_ipc import Cuda
    from measure import mapped_tensor
    from remote_bank import RemoteBank
    cuda=Cuda();cuda.set_device(cfg['offpath_device']);banks=[];addresses=[];records=[]
    for d,c in zip(cfg['donor_devices'],connections):
        allocation=receive(c,stop);assert allocation['slots']==cfg['remote_capacity_per_donor']
        cuda.enable_peer(cfg['offpath_device'],d);address=cuda.open(allocation['handle'])
        tensor=mapped_tensor(address,allocation['slots'],cfg['offpath_device'])
        assert tensor.data_ptr()==address and float(tensor.flatten()[0].item())==d+1.5
        tensor.flatten()[0].fill_(d+2.25);torch.cuda.synchronize()
        c.send(dict(type='probe_written'));assert receive(c,stop)['type']=='verified'
        banks.append(tensor);addresses.append(address);records.append(dict(allocation=allocation,imported_pointer=address,attributes=cuda.attributes(address)))
    save(root/'mapping.json',dict(banks=records,source_order=cfg['donor_devices'],slot_rule='global g -> bank g%3,physical g//3'))
    return RemoteBank(banks),cuda,addresses


def close_banks(cuda,addresses,connections,stop):
    import torch
    torch.cuda.synchronize()
    for address,c in zip(addresses,connections):
        cuda.close(address);c.send(dict(type='closed'));assert receive(c,stop)['type']=='freed'


def worker(role,cfg,root,connections,stop):
    with (root/f'worker_{role}.log').open('w',buffering=1) as log:
        os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
        try:
            import torch
            torch.cuda.set_device(role);torch.manual_seed(cfg['random_seed'])
            if role==cfg['offpath_device']:
                if cfg['operation']=='decode_probe':
                    from decode_probe import consume
                else:
                    from model_host import consume
                return consume(cfg,root,connections,stop)
            from cuda_ipc import Cuda
            from measure import mapped_tensor
            cuda=Cuda();cuda.set_device(role);n=cfg['remote_capacity_per_donor']
            address,handle=cuda.allocate(n*147456);tensor=mapped_tensor(address,n,role)
            tensor.zero_();tensor.flatten()[0].fill_(role+1.5);torch.cuda.synchronize()
            c=connections;c.send(dict(type='allocation',owner_device=role,slots=n,bytes=n*147456,handle=handle,owner=cuda.attributes(address)))
            assert receive(c,stop)['type']=='probe_written'
            assert float(tensor.flatten()[0].item())==role+2.25
            c.send(dict(type='verified'));assert receive(c,stop)['type']=='closed'
            cuda.free(address);c.send(dict(type='freed'))
            save(root/f'owner_{role}.json',dict(owner=role,model_loaded=False,model_forwards=0,bidirectional_probe=True,released=True))
        except BaseException:
            save(root/f'worker_{role}_failure.json',dict(error=traceback.format_exc()));stop.set();raise


def run_group(cfg,root):
    root.mkdir(parents=True);save(root/'config.json',cfg)
    ctx=mp.get_context('spawn');stop=ctx.Event();pairs=[ctx.Pipe(duplex=True) for _ in cfg['donor_devices']]
    workers=[ctx.Process(target=worker,args=(d,cfg,root,pairs[i][1],stop)) for i,d in enumerate(cfg['donor_devices'])]
    workers.append(ctx.Process(target=worker,args=(cfg['offpath_device'],cfg,root,[p[0] for p in pairs],stop)))
    for p in workers:p.start()
    for pair in pairs:
        for c in pair:c.close()
    for p in workers:p.join()
    assert all(p.exitcode==0 for p in workers),[p.exitcode for p in workers]


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    cfg=json.loads(a.config.read_text());a.out.mkdir(parents=True);save(a.out/'config.json',cfg);topology(a.out,cfg)
    run_group(cfg,a.out/'probe')
