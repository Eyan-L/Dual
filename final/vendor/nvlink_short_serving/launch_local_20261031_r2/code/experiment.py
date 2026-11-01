"""One measured saturated run per mode, after separate lifecycle validation."""
import hashlib
import random
from backend import ReplayBackend
from replay import replay
from experiment_common import save


def workload(cfg,tokenizer):
    if cfg['operation']=='smoke':
        return [dict(id=f'smoke_{i}',case='synthetic_lifecycle_validation',prompt_ids=[1000+i]*60,
                     output_ids=[2000+i]*out,input_tokens=60,output_tokens=out,peak_kv_tokens=60+out-1)
                for i,out in enumerate([31,28,30,31,28,30])]
    records=[]
    material=('A research team is measuring a language model service. Requests enter a queue, '
              'read their context, generate a response, and release their memory. '
              'Summarize the given observations in plain language.\n')
    for i in range(cfg['request_count']):
        text=tokenizer.apply_chat_template([
            dict(role='user',content=f'Sample {i}.\n'+material*40)],
            tokenize=False,add_generation_prompt=True,enable_thinking=False)
        ids=tokenizer.encode(text,add_special_tokens=False)
        # Preserve the chat generation suffix while making the input exact.
        n=cfg['input_tokens'];assert len(ids)>=n
        prompt=ids[:n-32]+ids[-32:]
        records.append(dict(id=f'short_{i}',case='controlled_short_context',prompt_ids=prompt,
            input_tokens=len(prompt),output_tokens=cfg['output_tokens'],peak_kv_tokens=n+cfg['output_tokens']-1))
    return records


def collect(bench,wrapped,tokenizer,old_control,cfg,w,root):
    records=workload(cfg,tokenizer)
    assert all(r['input_tokens']+r['output_tokens']<=cfg['context_length_limit'] for r in records)
    save(root/'input_checks.json',dict(requests=len(records),
        input_tokens=sum(v['input_tokens'] for v in records),output_tokens=sum(v['output_tokens'] for v in records),
        prompt_ids_sha256=hashlib.sha256(str([r['prompt_ids'] for r in records]).encode()).hexdigest(),
        decode_source=cfg.get('decode_source','teacher_forced')))
    # The compact tokenized workload is retained outside primary timing.
    save(root/'workload.json',records)
    control=ReplayBackend(old_control.raw,old_control.peer,cfg,old_control,cfg['timing_case'])
    control.configure(dict(name='batch2048_fused_async',migration_tokens=2048,async_copy=True,copy_impl='fused',reserve_steps=0))
    control.prepare_graphs(cfg['max_batch'])
    if cfg['operation']!='smoke':
        control.warm_prefill_kernels()
        warm=[dict(v,output_tokens=16,peak_kv_tokens=v['input_tokens']+15) for v in records[:16]]
        replay(bench,wrapped,control,cfg,warm)
    assert cfg['plain_trials']==1,'User requested one measured run per configuration'
    order=list(records)
    if cfg['operation']!='smoke':random.Random(cfg['random_seed']).shuffle(order)
    result=replay(bench,wrapped,control,cfg,order)
    result.update(case=cfg['timing_case'],trial=0,validation=cfg['validation'])
    save(root/'plain_0.json',result)
    if cfg['operation']=='smoke' and cfg['timing_case']=='tiered_peer':
        assert result['migrations'] and result['restorations']
        assert any(v['remote_tokens'] for v in result['attention_checks'])
        assert len({s['batch'] for s in result['steps'] if s['phase']=='decode'})>1
    if cfg['operation']=='smoke':
        # Two requests fit together only with peer capacity. Both are still
        # prefilling when C fills, exercising native-to-tiered prefill dispatch.
        extra=[dict(id=f'prefill_{i}',prompt_ids=[3000+i]*n,output_ids=[4000+i]*2,
                    input_tokens=n,output_tokens=2,peak_kv_tokens=n+1) for i,n in enumerate([136,130])]
        checked=replay(bench,wrapped,control,cfg,extra)
        save(root/'prefill_transition_check.json',checked)
        if cfg['timing_case']=='tiered_peer':
            assert any(v['attention_path']=='tiered_prefill5' for v in checked['forwards'])
            assert any(v['mode']=='prefill' and v['remote_tokens'] for v in checked['attention_checks'])
        full=[dict(id='full_prefill_block',prompt_ids=[3500]*256,output_ids=[4500],
                   input_tokens=256,output_tokens=1,peak_kv_tokens=256)]
        save(root/'prefill_full_block_check.json',replay(bench,wrapped,control,
            dict(cfg,prefill_chunk_tokens=256,prefill_token_budget=256),full))
    row=dict(trial=0,wall_ms=result['wall_ms'],output_tokens_per_second=result['output_tokens_per_second'],
        completed_requests=result['completed_requests'],maximum_running=result['maximum_running'],
        migrations=len(result['migrations']),restorations=len(result['restorations']))
    save(root/'progress.json',[row])
    print('SHORT_SERVING_PLAIN',cfg['timing_case'],row,flush=True)
    save(root/'case_collection.json',dict(status='TRACE_REPLAY_CASE_COMPLETE',config=cfg,trials=[row],
        scope='One measured run; fixed local C and known generation lengths. Capacity-driven admission; chunked prefill/decode interleave; all migration, release and ratio repair charged.'))
