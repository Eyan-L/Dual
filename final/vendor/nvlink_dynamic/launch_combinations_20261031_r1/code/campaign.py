"""Matched persistent-KV replay; primary wall/Event timing and separate traces."""
from array import array
import json
import math
import time
import torch
from experiment_common import base,save
from backend import TimingBackend,scope


def session(bench,wrapped,control,cfg,fixture,profile=False,checkpoints=False):
    raw=control.raw;batch_size=cfg['batch'];steps=cfg['decode_steps']
    wrapped.clear();control.reset(batch_size,profile)
    initial=fixture['rounds'][0]['suffixes']
    reqs=bench.prepare_synthetic_inputs_for_latency_test(batch_size,max(map(len,initial)),initial)
    control.req_objects=reqs
    for req in reqs:req.sampling_params.ignore_eos=True;req.sampling_params.max_new_tokens=steps+2
    forced=[torch.tensor(v['decode_inputs'],device=raw.device,dtype=torch.int64) for v in fixture['rounds']]
    generated=torch.empty((3,steps+1,batch_size),device=raw.device,dtype=torch.int64)
    history=[[] for _ in reqs];rounds=[];chunks=[];saved={}
    torch.cuda.synchronize()
    for turn,data in enumerate(fixture['rounds']):
        assert list(map(len,history))==data['history_before']
        full=[h+s for h,s in zip(history,data['suffixes'])]
        parts=math.ceil(max(map(len,data['suffixes']))/cfg['chunk_tokens'])
        prefix=list(map(len,history));round_events=control.events()
        start_ns=time.time_ns();start=time.perf_counter();round_events[0].record()
        for part in range(parts):
            control.context=dict(phase='prefill',turn=turn,chunk=part,step=-1,prefill_last_chunk=(part==parts-1),remaining_decode_steps=steps)
            starts=[prefix[i]+len(v)*part//parts for i,v in enumerate(data['suffixes'])]
            ends=[prefix[i]+len(v)*(part+1)//parts for i,v in enumerate(data['suffixes'])]
            chunk_start=time.perf_counter();uid=len(control.frames)
            migration_ms=control.ensure_headroom(sum(b-a for a,b in zip(starts,ends)))
            with scope(f'SLACK_COST_REQUEST_{uid}',profile):
                for req,ids,a,z in zip(reqs,full,starts,ends):
                    if a:req.prefix_indices=raw.req_to_token_pool.req_to_token[req.kv.req_pool_idx,:a].to(req.prefix_indices.dtype)
                    req.origin_input_ids=array('q',ids[:z]);req.full_untruncated_fill_ids=array('q',ids[:z]);req.set_extend_range(a,z)
            bench_start=time.perf_counter()
            with scope(f'SLACK_COST_SCHEDULE_{uid}',profile):sampled,logits,batch=bench.extend(reqs,raw)
            submitted=time.perf_counter();frame=control.frames[-1]
            chunks.append(dict(uid=uid,turn=turn,chunk=part,prefix_lengths=starts,query_lengths=[b-a for a,b in zip(starts,ends)],
                context_lengths=ends,remote_active=control.remote_active,migration_wall_ms=migration_ms,
                request_fields_wall_ms=(bench_start-chunk_start)*1000-migration_ms,
                scheduling_before_forward_wall_ms=(frame['forward_entry']-bench_start)*1000,
                chunk_cpu_submission_ms=(submitted-chunk_start)*1000))
            if part==parts-1 and control.variant.get('direct_remote',False) and control.variant.get('reserve_steps',0):
                control.context['reserve_after_write']=True
                before=time.perf_counter();extra=control.ensure_headroom(0)
                chunks[-1]['migration_wall_ms']+=extra
                chunks[-1]['chunk_cpu_submission_ms']+=(time.perf_counter()-before)*1000
                chunks[-1]['remote_active']=control.remote_active
                del control.context['reserve_after_write']
        # CPU-visible first token, including required migration, allocation,
        # all chunk forwards, sampling and the final token transfer/sync.
        first_token=sampled.cpu().tolist();elapsed=(time.perf_counter()-start)*1000;end_ns=time.time_ns()
        round_events[1].record();generated[turn,0].copy_(sampled)
        rounds.append(dict(turn=turn,first_or_append='initial' if turn==0 else 'append',
            prefix_lengths=prefix,query_lengths=list(map(len,data['suffixes'])),context_lengths=data['prefill_lengths'],
            ttft_wall_ms=elapsed,start_unix_ns=start_ns,end_unix_ns=end_ns,actual_first_token=first_token,
            remote_after_prefill=control.remote_active,_events=round_events))
        if checkpoints:saved[f'prefill_{turn}']=logits.detach().clone()
        history=full;control.enable_decode()
        for step in range(steps):
            control.context=dict(phase='decode',turn=turn,chunk=-1,step=step,prefill_last_chunk=False,remaining_decode_steps=steps-step)
            control.ensure_headroom(batch_size)
            for i in range(batch_size):history[i].append(data['decode_inputs'][step][i])
            with scope(f'SLACK_COST_DECODE_SCHEDULE_{len(control.frames)}',profile):
                forward=base.make_forward(bench,raw,batch,forced[turn][step])
            result=raw.forward(forward);sampled=raw.sample(result.logits_output,forward)
            generated[turn,step+1].copy_(sampled)
        # The caller appends its next input only after generation has completed.
        # Keep outstanding decode work outside the next input's TTFT window.
        if checkpoints:saved[f'decode_{turn}']=result.logits_output.next_token_logits.detach().clone()
        torch.cuda.synchronize()
        rounds[-1]['decode_complete_unix_ns']=time.time_ns()
        assert list(map(len,history))==data['final_lengths']
    final=control.finish()
    for r in rounds:
        a,z=r.pop('_events');r['ttft_gpu_stream_ms']=a.elapsed_time(z)
    actual=generated.cpu().tolist();expected=[v['expected_outputs'] for v in fixture['rounds']]
    matches=sum(a==b for turn_a,turn_b in zip(actual,expected) for row_a,row_b in zip(turn_a,turn_b) for a,b in zip(row_a,row_b))
    assert final['logical_tokens']==4296
    assert final['remote_active_tokens']==(200 if control.case=='tiered_peer' else 0)
    return dict(session_wall_ms=(rounds[-1]['decode_complete_unix_ns']-rounds[0]['start_unix_ns'])/1e6,rounds=rounds,chunks=chunks,forwards=control.frames,migrations=control.migrations,
        final=final,graph_captures=control.graph_captures,actual_output_tokens=actual,
        greedy_matches_to_fixture=matches,greedy_comparisons=102),saved

