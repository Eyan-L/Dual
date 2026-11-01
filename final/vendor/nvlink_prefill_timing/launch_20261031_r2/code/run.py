
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
from campaign import collect
host.collect=collect


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();cfg=json.loads(a.config.read_text());a.out.mkdir(parents=True,exist_ok=False)
    host.save(a.out/'config.json',cfg);host.base.topology(a.out);ctx=mp.get_context('spawn')
    for case in cfg['case_order']:
        use=dict(cfg,timing_case=case,kv_capacity_tokens=cfg['base_local_capacity_tokens'] if case=='tiered_peer' else cfg['equal_work_local_capacity_tokens'])
        root=a.out/case;root.mkdir();host.save(root/'config.json',use)
        w=dict(name='matched_prefill',native_batch=cfg['batch'],extended_batch=cfg['batch'],
               initial_context=1,final_context=cfg['max_final_context'])
        stop=ctx.Event();left,right=ctx.Pipe(duplex=True)
        workers=[ctx.Process(target=host.worker,args=(i,use,w,root,c,stop)) for i,c in enumerate((left,right))]
        for worker in workers:worker.start()
        left.close();right.close()
        for worker in workers:worker.join()
        assert all(worker.exitcode==0 for worker in workers),[worker.exitcode for worker in workers]
        assert json.loads((root/'case_collection.json').read_text())['status']=='MATCHED_PREFILL_TIMING_CASE_COMPLETE'
    import torch
    local=torch.load(a.out/'custom_all_local/matched_checkpoints.pt',map_location='cpu',weights_only=True)
    peer=torch.load(a.out/'tiered_peer/matched_checkpoints.pt',map_location='cpu',weights_only=True)
    native=torch.load(a.out/'native_local/matched_checkpoints.pt',map_location='cpu',weights_only=True)
    rows=[]
    for name in local:
        torch.testing.assert_close(local[name],peer[name],atol=cfg['reference_atol'],rtol=cfg['reference_rtol'])
        rows.append(dict(checkpoint=name,custom_peer_bitwise_equal=torch.equal(local[name],peer[name]),
            custom_peer_max_abs=float((local[name]-peer[name]).abs().max()),
            custom_peer_relative_l2=float((local[name]-peer[name]).norm()/local[name].norm()),
            native_custom_relative_l2=float((native[name]-local[name]).norm()/native[name].norm())))
    host.save(a.out/'matched_output_checks.json',rows)
    host.save(a.out/'collection.json',dict(status='MATCHED_PREFILL_TIMING_COLLECTION_COMPLETE',config=cfg,
        cases=cfg['case_order'],matched_input_source_job='26070196.pbs101',output_check='matched_output_checks.json'))
    print('MATCHED_PREFILL_TIMING_COLLECTION_COMPLETE',flush=True)


if __name__=='__main__':main()
