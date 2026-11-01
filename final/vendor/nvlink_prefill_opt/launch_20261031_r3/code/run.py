
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
    cfg['reference_logits']=cfg.get('reference_logits',str(args.out/'custom_all_local/r5/matched_checkpoints.pt'))
    ctx=mp.get_context('spawn')
    for case in cfg['case_order']:
        use=dict(cfg,timing_case=case,kv_capacity_tokens=(cfg['base_local_capacity_tokens'] if case=='tiered_peer' else cfg['equal_work_local_capacity_tokens']))
        root=args.out/case;root.mkdir();host.save(root/'config.json',use)
        w=dict(name='prefill5',native_batch=cfg['batch'],extended_batch=cfg['batch'],initial_context=1,final_context=cfg['max_final_context'])
        stop=ctx.Event();left,right=ctx.Pipe(duplex=True)
        workers=[ctx.Process(target=host.worker,args=(i,use,w,root,c,stop)) for i,c in enumerate((left,right))]
        for worker in workers:worker.start()
        left.close();right.close()
        for worker in workers:worker.join()
        assert all(worker.exitcode==0 for worker in workers),[worker.exitcode for worker in workers]
        assert json.loads((root/'case_collection.json').read_text())['status']=='PREFILL5_CASE_COMPLETE'
    import torch
    comparisons=[]
    admitted=[]
    for case in ['custom_all_local','tiered_peer']:
        if case in cfg['case_order']:
            admitted.append({k['name'] for k in json.loads((args.out/case/'case_collection.json').read_text())['kernels']})
    common=sorted(set.intersection(*admitted))
    assert 'r5' in common
    for name in common:
        local_path=args.out/'custom_all_local'/name/'matched_checkpoints.pt'
        if not local_path.exists():local_path=Path(cfg['reference_logits'])
        local=torch.load(local_path,map_location='cpu',weights_only=True)
        peer=torch.load(args.out/'tiered_peer'/name/'matched_checkpoints.pt',map_location='cpu',weights_only=True)
        for point in local:
            assert torch.equal(local[point],peer[point]),(name,point,'local/peer numerical mismatch')
            comparisons.append(dict(kernel=name,checkpoint=point,local_peer_bitwise_equal=True,local_reference=str(local_path)))
    host.save(args.out/'collection.json',dict(status='PREFILL5_COMPLETE',config=cfg,qualified_in_both_cases=common,local_peer_checks=comparisons))
    print('PREFILL5_COMPLETE',flush=True)


if __name__=='__main__':main()
