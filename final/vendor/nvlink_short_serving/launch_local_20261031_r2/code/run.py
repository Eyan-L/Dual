
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
sys.path.insert(0,str(_SLACK_FINAL_ROOT / 'vendor/nvlink_prefill/launch_20261031_r5/code'))
sys.path.insert(0,str(HERE))
from experiment import collect
host.collect=collect


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();cfg=json.loads(args.config.read_text());args.out.mkdir(parents=True,exist_ok=False)
    host.save(args.out/'config.json',cfg);host.base.topology(args.out)
    ctx=mp.get_context('spawn')
    for case in cfg['case_order']:
        use=dict(cfg,timing_case=case);root=args.out/case;root.mkdir();host.save(root/'config.json',use)
        w=dict(name='trace_replay',native_batch=1,extended_batch=cfg['max_batch'],initial_context=1,final_context=cfg['context_length_limit'])
        stop=ctx.Event();left,right=ctx.Pipe(duplex=True)
        processes=[ctx.Process(target=host.worker,args=(i,use,w,root,c,stop)) for i,c in enumerate((left,right))]
        for process in processes:process.start()
        left.close();right.close()
        for process in processes:process.join()
        assert all(p.exitcode==0 for p in processes),[p.exitcode for p in processes]
        assert json.loads((root/'case_collection.json').read_text())['status']=='TRACE_REPLAY_CASE_COMPLETE'
    host.save(args.out/'collection.json',dict(status='TRACE_REPLAY_COMPLETE',config=cfg))
    print('TRACE_REPLAY_COMPLETE',cfg['operation'],flush=True)


if __name__=='__main__':main()
