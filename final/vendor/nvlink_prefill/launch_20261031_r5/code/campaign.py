"""Three real input/decode rounds on persistent KV with global peer budgeting."""
from array import array
import math
import torch
from experiment_common import base,save
from tiered_backend import TieredBackend,KVCapacityExhausted
from validate_kernel import validate


def material(tokenizer,length,index,turn):
    text=(f'Request {index}, input round {turn}: '+
          ('A controlled serving experiment measures attention and request latency. ' if index==0
           else 'Remote KV pages extend memory while causal attention computes on the consumer GPU. '))
    ids=tokenizer.encode(text,add_special_tokens=False)
    return (ids*math.ceil(length/len(ids)))[:length]


def extend_segments(bench,raw,control,reqs,history,suffixes,chunks):
    full=[a+b for a,b in zip(history,suffixes)];base_lengths=[len(v) for v in history]
    for part in range(chunks):
        starts=[base_lengths[i]+len(v)*part//chunks for i,v in enumerate(suffixes)]
        ends=[base_lengths[i]+len(v)*(part+1)//chunks for i,v in enumerate(suffixes)]
        assert all(b>a for a,b in zip(starts,ends))
        if control.enabled:control.ensure_headroom(sum(b-a for a,b in zip(starts,ends)))
        before=len(control.checks) if control.enabled else 0
        for req,ids,start,end in zip(reqs,full,starts,ends):
            if start:req.prefix_indices=raw.req_to_token_pool.req_to_token[req.kv.req_pool_idx,:start].to(req.prefix_indices.dtype)
            assert not req.output_ids  # one_batch's decode does not append scheduler output_ids
            req.origin_input_ids=array('q',ids[:end]);req.full_untruncated_fill_ids=array('q',ids[:end]);req.set_extend_range(start,end)
        token,logits,batch=bench.extend(reqs,raw)
        assert torch.isfinite(logits).all().item()
        assert batch.seq_lens_cpu.tolist()==ends
        if control.enabled:
            assert len(control.checks)-before==36
            assert control.lengths==ends
    return batch,token,logits,full


def session(bench,wrapped,tokenizer,control,cfg,indices,references=None):
    raw=wrapped.torch_runner;wrapped.clear();batch_size=len(indices)
    if control.enabled:control.reset(batch_size)
    initial=[material(tokenizer,cfg['initial_lengths'][i],i,0) for i in indices]
    reqs=bench.prepare_synthetic_inputs_for_latency_test(batch_size,max(map(len,initial)),[list(x) for x in initial])
    for req in reqs:req.sampling_params.ignore_eos=True;req.sampling_params.max_new_tokens=cfg['decode_steps']+2
    history=[[] for _ in indices];pending=None;rounds=[]
    for turn in range(1+len(cfg['append_lengths'])):
        suffixes=(initial if turn==0 else [[int(pending[j])]+material(tokenizer,cfg['append_lengths'][turn-1][i],i,turn) for j,i in enumerate(indices)])
        full_sizes=(cfg['initial_lengths'] if turn==0 else [n+1 for n in cfg['append_lengths'][turn-1]])
        chunks=math.ceil(max(full_sizes)/cfg['chunk_tokens'])
        before_lengths=list(map(len,history));pending_consumed=None if pending is None else list(pending)
        batch,token,logits,history=extend_segments(bench,raw,control,reqs,history,suffixes,chunks)
        pref_hist=[list(x) for x in history];generated=[token.cpu().tolist()];inputs=[];comparisons=[]
        snapshots=[logits.detach().clone()] if references is None else None
        matching=[True]*batch_size
        if references is not None:
            matching=[history[j]==references[i]['rounds'][turn]['committed_after_prefill'][0] for j,i in enumerate(indices)]
        def compare(current,position):
            if references is None:return
            for j,i in enumerate(indices):
                if matching[j]:
                    ref=references[i]['rounds'][turn]['logits'][position][0]
                    comparisons.append(dict(request=i,position=position,same_input_prefix=True,
                        relative_l2=float((current[j].float()-ref.float()).norm()/ref.float().norm()),
                        top1_match=int(current[j].argmax())==references[i]['rounds'][turn]['outputs'][position][0]))
        compare(logits,0)
        prefill_audit=control.audit() if control.enabled else None
        if control.enabled:control.enable_decode()
        for step in range(cfg['decode_steps']):
            if control.enabled:control.ensure_headroom(batch_size)
            consumed=token.cpu().tolist();inputs.append(consumed)
            for j,i in enumerate(indices):
                history[j].append(consumed[j])
                if references is not None:
                    matching[j]=matching[j] and consumed[j]==references[i]['rounds'][turn]['inputs'][step][0]
            forward=base.make_forward(bench,raw,batch,token);result=raw.forward(forward)
            if control.enabled:assert result.can_run_graph
            logits=result.logits_output.next_token_logits;assert torch.isfinite(logits).all().item()
            if snapshots is not None:snapshots.append(logits.detach().clone())
            compare(logits,step+1);token=raw.sample(result.logits_output,forward);generated.append(token.cpu().tolist())
            assert batch.seq_lens_cpu.tolist()==list(map(len,history))
            if control.enabled and step in (0,cfg['decode_steps']//2,cfg['decode_steps']-1):control.check_decode()
        assert inputs==generated[:-1]
        pending=generated[-1]
        row=dict(turn=turn,committed_before=before_lengths,new_input_lengths=[len(v) for v in suffixes],
            pending_previous_output_consumed_once=pending_consumed,
            committed_after_prefill=pref_hist,committed_after_decode=[list(x) for x in history],
            inputs=inputs,outputs=generated,prefill=prefill_audit,
            after_decode=control.audit() if control.enabled else None,
            same_prefix_native_logits=comparisons)
        if snapshots is not None:row['logits']=snapshots
        rounds.append(row)
        print('INPUT_DECODE_ROUND_COMPLETE','tiered' if control.enabled else 'native',indices,turn,list(map(len,history)),flush=True)
    return dict(indices=indices,rounds=rounds,final_pending_output=list(pending))


def collect(bench,wrapped,tokenizer,old_control,cfg,w,root):
    raw=old_control.raw;control=TieredBackend(raw,old_control.peer,cfg,old_control)
    control.diagnostic_root=root
    unit=validate(old_control.peer);save(root/'kernel_checks.json',unit)
    references={}
    for i in range(cfg['batch']):
        references[i]=session(bench,wrapped,tokenizer,control,cfg,[i])
        # Save token/KV-history evidence; full native logits are validation-only GPU tensors.
        public=dict(references[i]);public['rounds']=[{k:v for k,v in row.items() if k!='logits'} for row in public['rounds']]
        save(root/f'native_session_request{i}.json',public)
    control.enabled=True
    result=session(bench,wrapped,tokenizer,control,cfg,list(range(cfg['batch'])),references)
    final=control.audit();assert final['logical_tokens']>cfg['kv_capacity_tokens']
    assert any(row['prefill']['remote_active_tokens']>0 for row in result['rounds'])
    assert control.decode_graph_captures==1  # changing maps and input/decode phases reuse the graph
    assert len({v['remote'] for v in final['per_request']})>1  # no forced equal percentage
    before=control.audit();free=raw.token_to_kv_pool_allocator.available_size();peer_free=len(control.peer_free)
    try:control.ensure_headroom(free+peer_free+1)
    except KVCapacityExhausted:capacity_rejected=True
    else:raise AssertionError('Over-budget admission must be rejected before mutation')
    assert control.audit()==before
    result.update(final=final,layer_checks=control.checks,migrations=control.migrations,
        forwards=control.forwards,decode_graph_captures=control.decode_graph_captures,
        reference_adjudications=control.reference_audits,
        all_native_elementwise_pass=not bool(control.reference_audits),
        over_budget_rejected_without_mutation=capacity_rejected,prefill_kernel=control.kernel_metadata)
    save(root/'multiturn_session.json',result)
    status='MULTITURN_TIERED_KV_CHECKS_PASSED_WITH_FP64_ADJUDICATION' if control.reference_audits else 'MULTITURN_TIERED_KV_CHECKS_PASSED'
    save(root/'prefill_collection.json',dict(status=status,config=cfg,
        kernel_checks=unit,session=result,offpath_only_model=True,donor_model_steps=0,
        policy='Real native local allocator first; global donor slot budget. On pressure migrate oldest resident token pages, independent of per-request percentage. Prefill and decode both read Tier/Slot. New writes remain local. Reject if total capacity cannot admit the next chunk; no overwrite.',
        limitation='Controlled benchmark integration, page_size1, BF16 Qwen3 dense causal GQA. No native online queue integration or performance claim. One-layer temporary native gathers only for validation.'))
