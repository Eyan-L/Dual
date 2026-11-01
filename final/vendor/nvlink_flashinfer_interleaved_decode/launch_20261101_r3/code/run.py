
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
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path);p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();cfg=json.loads(a.config.read_text());assert cfg['plain_trials']==1
    assert cfg['final_context']-cfg['initial_context']==cfg['decode_steps']
    assert cfg['setup_context']==cfg['initial_context']
    assert cfg['expanded_batch']*cfg['initial_context']<=cfg['kv_capacity_tokens']
    a.out.mkdir(parents=True);host.save(a.out/'config.json',cfg);host.base.topology(a.out)
    ctx=mp.get_context('spawn')
    for case in cfg['case_schedule']:
        reference=Path(cfg['reference_measurement_root'])
        use=dict(cfg,**case,reference_root=str(reference/cfg['native_label']),local_reference_root=str(reference/cfg['local_label']))
        root=a.out/case['label'];root.mkdir();host.save(root/'config.json',use)
        w=dict(name='decode_detail',native_batch=use['batch'],extended_batch=use['batch'],initial_context=cfg['initial_context'],final_context=cfg['final_context'])
        stop=ctx.Event();left,right=ctx.Pipe(duplex=True)
        workers=[ctx.Process(target=host.worker,args=(i,use,w,root,c,stop)) for i,c in enumerate((left,right))]
        for worker in workers:worker.start()
        left.close();right.close()
        for worker in workers:worker.join()
        assert all(worker.exitcode==0 for worker in workers),[worker.exitcode for worker in workers]
        assert json.loads((root/'case_collection.json').read_text())['status']=='DECODE_DETAIL_CASE_COMPLETE'
    host.save(a.out/'collection.json',dict(status='DECODE_DETAIL_COMPLETE',config=cfg))


if __name__=='__main__':main()
