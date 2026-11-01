
from pathlib import Path as _FinalPath
_SLACK_FINAL_ROOT = next(p for p in _FinalPath(__file__).resolve().parents if (p / 'release.json').is_file())
_SLACK_PROJECT_ROOT = _SLACK_FINAL_ROOT.parent
import argparse
import copy
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(_SLACK_FINAL_ROOT / 'vendor/nvlink_throughput/launch_20261030_r1/code'))
import benchmark as host
sys.path.insert(0,str(_SLACK_FINAL_ROOT / 'vendor/nvlink_prefill/launch_20261031_r5/code'))
sys.path.insert(0,str(HERE))
from experiment import collect
host.collect=collect


def run_pair(cfg,root):
    root.mkdir(parents=True,exist_ok=True);host.save(root/'config.json',cfg)
    w=dict(name=cfg['stage'],native_batch=cfg['batch'],extended_batch=cfg['batch'],
        initial_context=cfg['initial_context'],final_context=cfg['initial_context']+cfg['decode_steps'])
    ctx=mp.get_context('spawn');stop=ctx.Event();left,right=ctx.Pipe(duplex=True)
    workers=[ctx.Process(target=host.worker,args=(i,cfg,w,root,c,stop)) for i,c in enumerate((left,right))]
    for worker in workers:worker.start()
    left.close();right.close()
    for worker in workers:worker.join()
    assert all(worker.exitcode==0 for worker in workers),[worker.exitcode for worker in workers]
    assert json.loads((root/'stage_collection.json').read_text())['status']=='DYNAMIC_STAGE_COMPLETE'


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();cfg=json.loads(args.config.read_text());args.out.mkdir(parents=True,exist_ok=False)
    host.base.topology(args.out)
    if cfg['operation']=='preflight':
        golden=copy.deepcopy(cfg);golden.update(operation='golden',stage='migration_latency_golden',
            kv_capacity_tokens=104192,decode_steps=1024)
        run_pair(golden,args.out/'golden')
        host.save(args.out/'golden'/'reference_hashes.json',{name:hashlib.sha256((args.out/'golden'/name).read_bytes()).hexdigest() for name in ['reference.json','reference_logits.pt']})
        scan=copy.deepcopy(cfg);scan.update(operation='scan',reference_root=str(args.out/'golden'))
        run_pair(scan,args.out/'sweep')
        host.save(args.out/'stage_collection.json',dict(status='DYNAMIC_STAGE_COMPLETE',operation='preflight',config=cfg,
            reference_root=str(args.out/'golden'),sweep_root=str(args.out/'sweep')))
    else:run_pair(cfg,args.out)
    print('MIGRATION_LATENCY_COMPLETE',cfg['stage'],flush=True)


if __name__=='__main__':main()
