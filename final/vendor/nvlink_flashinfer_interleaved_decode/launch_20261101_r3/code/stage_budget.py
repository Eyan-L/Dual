"""Native phase definitions with explicit CUPTI timestamp-overlap accounting.

Derived from frozen prefill_timing r2.stages. Raw intervals remain unchanged.
Overlapping reported timestamps on a serial model stream are uncertainty time:
exclude them from both phase budgets and retain them in Other/gaps,never twice.
"""
import collections


def stages(kernels,event_ms,case,uid):
    classified=[k for k in kernels if k['kind']=='gemm' or k['kind']=='kv_store' or k['kind'].startswith('attention_')]
    assert len({k['stream'] for k in classified})==1,(case,uid,'Model phases require a serial stream')
    gemms=[]
    for k in kernels:
        if k['kind']!='gemm':continue
        if 'splitkreduce' in k['name'].lower() or 'splitk_reduce' in k['name'].lower():
            assert gemms and gemms[-1]['stream']==k['stream']
            gemms[-1]['duration_us']=k['start_us']+k['duration_us']-gemms[-1]['start_us']
        else:gemms.append(dict(k))
    assert len(gemms)==145,(case,uid,'logical GEMMs',len(gemms))
    parts=[];active=collections.defaultdict(float)
    for k in kernels:
        field='gemm_ms' if k['kind']=='gemm' else ('attention_ms' if k['kind'].startswith('attention_') else ('kv_store_ms' if k['kind']=='kv_store' else None))
        if field:active[field]+=k['duration_us']/1000
    for i,k in enumerate(gemms):
        parts.append(dict(case=case,uid=uid,layer=i//4,phase='GEMM_'+(['QKV','O','MLP_UP','MLP_DOWN'][i%4] if i<144 else 'LM_HEAD'),start_us=k['start_us'],end_us=k['start_us']+k['duration_us']))
    main_count=0
    for layer in range(36):
        lo=gemms[layer*4]['start_us']+gemms[layer*4]['duration_us'];hi=gemms[layer*4+1]['start_us']
        att=[k for k in kernels if k['kind'].startswith('attention_') and lo<=k['start_us']<hi]
        mains=[k for k in att if k['kind']=='attention_main'];assert mains,(case,uid,layer)
        main_count+=len(mains)
        parts.append(dict(case=case,uid=uid,layer=layer,phase='ATTENTION',start_us=min(k['start_us'] for k in att),end_us=max(k['start_us']+k['duration_us'] for k in att)))
    stores=[k for k in kernels if k['kind']=='kv_store'];assert len(stores)==36,(case,uid,'KV stores',len(stores))
    for layer,k in enumerate(stores):parts.append(dict(case=case,uid=uid,layer=layer,phase='KV_STORE',start_us=k['start_us'],end_us=k['start_us']+k['duration_us']))
    parts.sort(key=lambda p:p['start_us'])
    endpoints=collections.defaultdict(int)
    for p in parts:endpoints[p['start_us']]+=1;endpoints[p['end_us']]-=1
    overlapping=[];count=0;previous=None
    for t,delta in sorted(endpoints.items()):
        if previous is not None and count>1 and t>previous:overlapping.append((previous,t))
        count+=delta;previous=t
    totals=collections.defaultdict(float)
    for p in parts:
        p['raw_duration_ms']=(p['end_us']-p['start_us'])/1000
        p['trace_timestamp_overlap_ms']=sum(max(0,min(p['end_us'],z)-max(p['start_us'],a)) for a,z in overlapping)/1000
        p['duration_ms']=p['raw_duration_ms']-p['trace_timestamp_overlap_ms']
        assert p['duration_ms']>=0
        totals[p['phase']]+=p['duration_ms']
    gemm=sum(v for k,v in totals.items() if k.startswith('GEMM_'));attention=totals['ATTENTION'];store=totals['KV_STORE']
    span=(max(k['start_us']+k['duration_us'] for k in kernels)-min(k['start_us'] for k in kernels))/1000
    other=event_ms-gemm-attention-store;assert other>=-.01,(case,uid,event_ms,gemm,attention,store)
    return dict(attention_ms=attention,gemm_ms=gemm,kv_store_ms=store,other_and_gaps_ms=other,
        kernel_other_and_gaps_ms=event_ms-sum(active.values()),gpu_first_last_span_ms=span,event_minus_kernel_span_ms=event_ms-span,
        attention_main_kernels=main_count,timestamp_overlap_ms=sum(z-a for a,z in overlapping)/1000,timestamp_overlap_segments=len(overlapping),
        **{k.replace('_ms','_kernel_active_ms'):v for k,v in active.items()},
        **{k.lower()+'_ms':v for k,v in totals.items() if k.startswith('GEMM_')}),parts
