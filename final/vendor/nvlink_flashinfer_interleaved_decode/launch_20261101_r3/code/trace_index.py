"""Indexed correlation for longer traces;preserves actual GPU timestamps."""
from bisect import bisect_right
from collections import defaultdict
import re


class Timeline:
    def __init__(self,scopes):
        self.rows=defaultdict(list)
        for s in scopes:self.rows[s['tid']].append(s)
        self.starts={}
        for tid,rows in self.rows.items():
            rows.sort(key=lambda s:s['lo'])
            assert all(a['hi']<=b['lo'] for a,b in zip(rows,rows[1:])),('Overlapping peer CPU scopes',tid)
            self.starts[tid]=[s['lo'] for s in rows]

    def at(self,tid,ts):
        if tid not in self.starts:return None
        i=bisect_right(self.starts[tid],ts)-1
        return self.rows[tid][i] if i>=0 and ts<=self.rows[tid][i]['hi'] else None


def extract_samples(events,kind):
    scopes=[]
    for e in events:
        if e.get('cat')!='user_annotation' or e.get('ph')!='X':continue
        match=re.fullmatch(r'SLACK_PHASE_FORWARD_.+_sample(\d+)',e.get('name',''))
        if match:scopes.append(dict(lo=e['ts'],hi=e['ts']+e['dur'],tid=e['tid'],uid=int(match[1])))
    assert scopes
    timeline=Timeline(scopes);launches={};samples={s['uid']:[] for s in scopes}
    for e in events:
        if e.get('cat') not in ['cuda_runtime','cuda_driver'] or e.get('ph')!='X':continue
        scope=timeline.at(e['tid'],e['ts']);corr=e.get('args',{}).get('correlation')
        if scope is not None and corr is not None:launches[corr]=scope['uid']
    for e in events:
        if e.get('cat')!='kernel' or e.get('ph')!='X' or e.get('dur',0)<=0:continue
        uid=launches.get(e.get('args',{}).get('correlation'))
        if uid is None:
            assert kind(e['name'])=='other',('Unattributed model kernel',e['name'])
            continue
        samples[uid].append(dict(name=e['name'],kind=kind(e['name']),start_us=e['ts'],duration_us=e['dur'],
            stream=e.get('args',{}).get('stream'),graph_id=e.get('args',{}).get('graph id',0)))
    return {i:sorted(rows,key=lambda r:r['start_us']) for i,rows in samples.items()}
