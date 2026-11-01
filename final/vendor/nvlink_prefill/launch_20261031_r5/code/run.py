
from pathlib import Path as _FinalPath
_SLACK_FINAL_ROOT = next(p for p in _FinalPath(__file__).resolve().parents if (p / 'release.json').is_file())
_SLACK_PROJECT_ROOT = _SLACK_FINAL_ROOT.parent
import argparse
import json
import multiprocessing as mp
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(_SLACK_FINAL_ROOT / 'vendor/nvlink_throughput/launch_20261030_r1/code'))
import benchmark as host
sys.path.insert(0,str(HERE))
from campaign import collect
host.collect=collect


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();cfg=json.loads(a.config.read_text());a.out.mkdir(parents=True,exist_ok=False)
    host.save(a.out/'config.json',cfg);host.base.topology(a.out)
    w=dict(name='tiered_prefill',native_batch=1,extended_batch=cfg['batch'],
           initial_context=1,final_context=cfg['max_final_context'])
    ctx=mp.get_context('spawn');stop=ctx.Event();left,right=ctx.Pipe(duplex=True)
    workers=[ctx.Process(target=host.worker,args=(i,cfg,w,a.out,c,stop)) for i,c in enumerate((left,right))]
    for worker in workers:worker.start()
    left.close();right.close()
    for worker in workers:worker.join()
    assert all(worker.exitcode==0 for worker in workers),[worker.exitcode for worker in workers]
    result=json.loads((a.out/'prefill_collection.json').read_text());assert result['status'] in ('MULTITURN_TIERED_KV_CHECKS_PASSED','MULTITURN_TIERED_KV_CHECKS_PASSED_WITH_FP64_ADJUDICATION')
    host.save(a.out/'collection.json',dict(status=result['status'],config=cfg))
    print(result['status'],flush=True)


if __name__=='__main__':main()
