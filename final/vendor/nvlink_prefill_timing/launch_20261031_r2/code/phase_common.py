"""Verbatim timing classification/correlation helpers from experiment4; no BW model."""
import re

def kind(name):
    n=name.lower()
    if 'batchprefill' in n or 'batchdecode' in n:
        return 'attention_main'
    if 'mergestate' in n or 'merge_state' in n:
        return 'attention_reduce'
    if any(s in n for s in ('gemm','gemv','cublas','splitkreduce','splitk_reduce')):
        return 'gemm'
    return 'other'

def extract_samples(events):
    scopes=[]
    for e in events:
        if e.get('cat')!='user_annotation' or e.get('ph')!='X':continue
        match=re.fullmatch(r'SLACK_PHASE_FORWARD_.+_sample(\d+)',e.get('name',''))
        if match or e.get('name')=='SLACK_FIXED_CONTEXT_FORWARD':
            scopes.append((e['ts'],e['ts']+e['dur'],e['tid'],int(match[1]) if match else 0))
    assert scopes,'No explicit CPU forward annotations'
    launches={}
    for e in events:
        if e.get('cat') not in ('cuda_runtime','cuda_driver') or e.get('ph')!='X':continue
        corr=e.get('args',{}).get('correlation')
        selected=[s for s in scopes if s[2]==e.get('tid') and s[0]<=e['ts']<=s[1]]
        if corr is not None and selected:
            launches[corr]=min(selected,key=lambda s:s[1]-s[0])[3]
    samples={s[3]:[] for s in scopes}
    for e in events:
        if e.get('cat')!='kernel' or e.get('ph')!='X' or e.get('dur',0)<=0:continue
        sample=launches.get(e.get('args',{}).get('correlation'))
        if sample is None:
            assert kind(e['name'])=='other',('Unattributed model kernel',e['name'])
            continue
        samples[sample].append(dict(name=e['name'],kind=kind(e['name']),start_us=e['ts'],duration_us=e['dur'],
                                   stream=e.get('args',{}).get('stream'),graph_id=e.get('args',{}).get('graph id',0)))
    return {i:sorted(k,key=lambda r:r['start_us']) for i,k in samples.items()}
