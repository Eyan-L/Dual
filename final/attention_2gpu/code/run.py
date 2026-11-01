
from pathlib import Path as _FinalPath
_SLACK_FINAL_ROOT = next(p for p in _FinalPath(__file__).resolve().parents if (p / 'release.json').is_file())
_SLACK_PROJECT_ROOT = _SLACK_FINAL_ROOT.parent
import argparse
import json
import multiprocessing as mp
import hashlib
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(_SLACK_FINAL_ROOT / 'vendor/nvlink_throughput/launch_20261030_r1/code'))
import benchmark as host
sys.path.insert(0,str(_SLACK_FINAL_ROOT / 'vendor/nvlink_prefill/launch_20261031_r5/code'))
sys.path.insert(0,str(_SLACK_FINAL_ROOT / 'vendor/nvlink_flashinfer_interleaved_decode/launch_20261101_r3/code'))
sys.path.insert(0,str(HERE))
from session import collect
from prepare_inputs import prepare
host.collect=collect


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    cfg=json.loads(a.config.read_text());assert cfg['plain_trials']==1
    assert sum(cfg['input_tokens_by_turn'])+cfg['turns']*cfg['decode_forwards_per_turn']==cfg['final_context']
    a.out.mkdir(parents=True)
    cfg['inputs_path']=str(a.out/'inputs.json')
    prepare(cfg,Path(cfg['inputs_path']))
    host.save(a.out/'config.json',cfg);host.base.topology(a.out)
    reused=cfg.get('reuse_cases',[])
    if reused:
        source=Path(cfg['reuse_measurement_root'])
        assert json.loads((source/'inputs.json').read_text())['requests']==json.loads(Path(cfg['inputs_path']).read_text())['requests']
        provenance=[]
        for label in reused:
            prior=json.loads((source/label/'case_collection.json').read_text())
            assert prior['status']=='MULTITURN_CASE_COMPLETE'
            (a.out/label).symlink_to(source/label,target_is_directory=True)
            names=['config.json','case_collection.json','validation.json']
            if (source/label/'plain_0.json').exists():names.append('plain_0.json')
            provenance.append(dict(case=label,source=str(source/label),sha256={name:hashlib.sha256((source/label/name).read_bytes()).hexdigest() for name in names}))
        host.save(a.out/'reuse_provenance.json',dict(source_job='26075145.pbs101',cases=provenance,
            compatibility='Identical input task/tool IDs,lengths,forced native8 history,native/local production code,pool sizes,warmup/primary/profile. Only peer numerical adjudication adds a same-order unmodified native cascade oracle. Native/reference cases complete in source job;no rerun.'))
    ctx=mp.get_context('spawn')
    for case in cfg['cases']:
        if case['label'] in reused:continue
        use=dict(cfg,**case);root=a.out/case['label'];root.mkdir();host.save(root/'config.json',use)
        w=dict(name='persistent_multiturn',native_batch=case['batch'],extended_batch=case['batch'],initial_context=cfg['initial_tokens'],final_context=cfg['final_context'])
        stop=ctx.Event();left,right=ctx.Pipe(duplex=True)
        workers=[ctx.Process(target=host.worker,args=(i,use,w,root,c,stop)) for i,c in enumerate([left,right])]
        for worker in workers:worker.start()
        left.close();right.close()
        for worker in workers:worker.join()
        assert all(worker.exitcode==0 for worker in workers),[worker.exitcode for worker in workers]
        assert json.loads((root/'case_collection.json').read_text())['status']=='MULTITURN_CASE_COMPLETE'
    host.save(a.out/'collection.json',dict(status='PERSISTENT_MULTITURN_COLLECTION_COMPLETE',config=cfg))
