"""Native single-GPU persistent-agent trace driver;no attention/page monkeypatches."""
import argparse
from array import array
import copy
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import socket
import time
import traceback


def save(p,v):
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(v,indent=2)+'\n');temp.replace(p)


def subset(batch,indices):
    result=copy.copy(batch);result.reqs=list(batch.reqs);result.sampling_info=copy.copy(batch.sampling_info)
    result.sampling_info.penalizer_orchestrator=copy.copy(batch.sampling_info.penalizer_orchestrator)
    assert not result.sampling_info.penalizer_orchestrator.is_required
    result.filter_batch(keep_indices=indices)
    return result


def execute(bench,wrapped,agents,origin,capacity,folder,*,diagnostic=False):
    import torch
    raw=wrapped.torch_runner;wrapped.clear();nodes=[];calls=[];steps=0;max_used=0;last_logged=-1
    for idx,a in enumerate(agents):
        req=bench.prepare_synthetic_inputs_for_latency_test(1,a['calls'][0]['input_tokens'],[[1000+idx]*a['calls'][0]['input_tokens']])[0]
        req.rid=a['id'];req.sampling_params.ignore_eos=True;req.sampling_params.max_new_tokens=65536
        nodes.append(dict(agent=a,req=req,turn=0,context=0,generated=0,batch=None,phase='waiting',
            ready_ns=origin+round(a.get('arrival_offset_ms',0)*1e6),first_start_ns=None,finish_ns=None,token=a.get('forced_token_id',1000+idx)))
    assert sum(a['reserved_peak_tokens'] for a in agents)<=capacity
    assert raw.max_total_num_tokens==capacity
    log=(folder/'calls.jsonl').open('w',buffering=1);sample_log=(folder/'scheduler.jsonl').open('w',buffering=1)
    while any(n['phase']!='done' for n in nodes):
        now=time.perf_counter_ns()
        for n in nodes:
            if n['phase']=='waiting' and now>=n['ready_ns']:
                n.update(phase='prefill',call_start_ns=now,first_visible_ns=None,generated=0)
                if n['first_start_ns'] is None:n['first_start_ns']=now
        selected=[];budget=4096
        for n in nodes:
            if n['phase']!='prefill':continue
            c=n['agent']['calls'][n['turn']];size=min(c['input_tokens']-n['context'],2048,budget)
            if size>0:selected.append((n,size));budget-=size
        if selected:
            assert raw.token_to_kv_pool_allocator.available_size()>=sum(size for _,size in selected)
            for n,size in selected:
                start=n['context'];end=start+size;req=n['req']
                if start:req.prefix_indices=raw.req_to_token_pool.req_to_token[req.kv.req_pool_idx,:start].to(req.prefix_indices.dtype)
                req.origin_input_ids=array('q',[n['token']]*end);req.full_untruncated_fill_ids=req.origin_input_ids;req.set_extend_range(start,end)
            tokens,_,batch=bench.extend([n['req'] for n,_ in selected],raw)
            tokens.cpu();visible=time.perf_counter_ns();steps+=1
            for i,(n,size) in enumerate(selected):
                n['context']+=size;c=n['agent']['calls'][n['turn']]
                assert n['context']<=c['input_tokens']
                if n['context']==c['input_tokens']:
                    n.update(batch=subset(batch,[i]),generated=1,first_visible_ns=visible,phase='decode')
        decoders=[n for n in nodes if n['phase']=='decode' and n['generated']<n['agent']['calls'][n['turn']]['output_tokens']]
        if decoders:
            batch=subset(decoders[0]['batch'],[0])
            for n in decoders[1:]:batch.merge_batch(subset(n['batch'],[0]))
            assert raw.token_to_kv_pool_allocator.available_size()>=len(decoders)
            batch.input_ids=torch.tensor([n['token'] for n in decoders],device=raw.device,dtype=torch.int64)
            batch.prepare_for_decode();bench._maybe_prepare_mlp_sync_batch(batch,raw)
            fb=bench.ForwardBatch.init_new(batch,raw,return_hidden_states_before_norm=False)
            result=raw.forward(fb);assert result.can_run_graph
            raw.sample(result.logits_output,fb).cpu();visible=time.perf_counter_ns();steps+=1
            for i,n in enumerate(decoders):
                n.update(batch=subset(batch,[i]),context=n['context']+1,generated=n['generated']+1,last_visible_ns=visible)
                assert n['batch'].seq_lens_cpu.tolist()==[n['context']]
        used=capacity-raw.token_to_kv_pool_allocator.available_size();max_used=max(max_used,used)
        for n in nodes:
            if n['phase']!='decode':continue
            c=n['agent']['calls'][n['turn']]
            if n['generated']!=c['output_tokens']:continue
            end=time.perf_counter_ns();assert n['context']==c['input_tokens']+c['output_tokens']-1
            row=dict(agent=n['agent']['id'],group=n['agent']['group'],round=n['turn'],input_tokens=c['input_tokens'],output_tokens=n['generated'],
                ready_ms=(n['ready_ns']-origin)/1e6,start_ms=(n['call_start_ns']-origin)/1e6,end_ms=(end-origin)/1e6,
                ttft_ms=(n['first_visible_ns']-n['ready_ns'])/1e6,round_service_ms=(end-n['call_start_ns'])/1e6,
                context_at_end=n['context'],pool_used_tokens=used,wait_after_ms=c['wait_after_ms'])
            calls.append(row);log.write(json.dumps(row)+'\n');print('CALL',n['agent']['id'],n['turn'],round(row['end_ms'],2),flush=True)
            n['turn']+=1
            if n['turn']==len(n['agent']['calls']):
                n.update(phase='done',finish_ns=end)
                rid=n['req'].kv.req_pool_idx;slots=raw.req_to_token_pool.req_to_token[rid,:n['context']].long()
                raw.token_to_kv_pool_allocator.free(slots);raw.req_to_token_pool.req_to_token[rid,:n['context']].zero_();raw.req_to_token_pool.free(n['req'])
            else:
                # One emitted output is pending,not in KV;next prefill commits it.
                target=n['agent']['calls'][n['turn']]['input_tokens'];assert target>=n['context']+1
                n.update(phase='waiting',ready_ns=end+round(c['wait_after_ms']*1e6),batch=None)
        if steps%64==0 and steps!=last_logged:
            sample_log.write(json.dumps(dict(elapsed_ms=(time.perf_counter_ns()-origin)/1e6,used_tokens=capacity-raw.token_to_kv_pool_allocator.available_size(),states={n['agent']['id']:n['phase'] for n in nodes}))+'\n')
            last_logged=steps
        if not selected and not decoders:
            upcoming=[n['ready_ns'] for n in nodes if n['phase']=='waiting']
            if upcoming:time.sleep(min(.02,max(0,(min(upcoming)-time.perf_counter_ns())/1e9)))
    torch.cuda.synchronize();log.close();sample_log.close()
    assert raw.token_to_kv_pool_allocator.available_size()==capacity
    assert raw.req_to_token_pool.available_size()==raw.req_to_token_pool.size
    assert len(calls)==sum(len(a['calls']) for a in agents)
    result=dict(status='INDEPENDENT_GPU_TRACE_COMPLETE',diagnostic=diagnostic,origin_perf_ns=origin,
        calls=calls,forwards=steps,max_used_tokens=max_used,capacity_tokens=capacity,memory_queue_events=0,remote_tokens=0,
        agents=[dict(id=n['agent']['id'],group=n['agent']['group'],arrival_offset_ms=n['agent'].get('arrival_offset_ms',0),
            first_start_ms=(n['first_start_ns']-origin)/1e6,finish_ms=(n['finish_ns']-origin)/1e6,
            service_wall_ms=(n['finish_ns']-origin)/1e6-n['agent'].get('arrival_offset_ms',0),calls=n['turn'],
            input_tokens=sum(c['input_tokens'] for c in n['agent']['calls']),output_tokens=sum(c['output_tokens'] for c in n['agent']['calls']),
            reserved_peak_tokens=n['agent']['reserved_peak_tokens']) for n in nodes],final_free_tokens=capacity)
    save(folder/'result.json',result);return result



def model(role,cfg,folder,root,connections,stop,owner):
    import torch
    from sglang.benchmark import one_batch as bench
    from host4 import open_banks,close_banks,receive
    gpu=0 if role==0 else 1;c=cfg['cooperation'];agents=[a for a in cfg['agents']if a['gpu']==gpu]
    physical=c['onpath_own_capacity_tokens']if gpu==0 else c['offpath_physical_capacity_tokens']
    capacity=physical if gpu==0 else c['offpath_logical_capacity_tokens']
    torch.cuda.set_device(role)
    def campaign(sa,pa,ba,device,rank):
        bench.publish(sa,role='scheduler',ranks=bench.SpawnRanks(world_rank=bench.spawn_world_rank(sa,tp_rank=0,pp_rank=0),gpu_id=role))
        bench.initialize_moe_config();bench.initialize_fp8_gemm_config();bench.initialize_fp4_gemm_config()
        bench.configure_logger(sa,prefix=f' COOP-GPU{gpu}')
        wrapped,_=bench.load_model(sa,pa,role,0);raw=wrapped.torch_runner
        assert raw.max_total_num_tokens==physical
        assert (raw.model_config.hf_config.num_hidden_layers,raw.model_config.hf_config.num_attention_heads,raw.model_config.hf_config.num_key_value_heads)==(36,32,8)
        peer=cuda=addresses=None
        if gpu==0:
            assert receive(connections,stop)['type']=='probe_written'
            assert float(owner['tensor'].flatten()[0].item())==role+2.25
            connections.send(dict(type='verified'))
        else:
            peer,cuda,addresses=open_banks(c,connections,stop,root)
        save(folder/'inventory.json',dict(device=torch.cuda.get_device_name(role),physical_ordinal=role,
            versions={k:importlib.metadata.version(k)for k in ['torch','sglang','flashinfer-python']},
            physical_capacity_tokens=physical,logical_capacity_tokens=capacity,model_config=raw.model_config.hf_config.to_dict()))
        with torch.no_grad():
            small=[dict(a,arrival_offset_ms=0,reserved_peak_tokens=110,calls=[dict(input_tokens=32,output_tokens=3,wait_after_ms=1),dict(input_tokens=96,output_tokens=4,wait_after_ms=0)])for a in agents]
            d=folder/'native_protocol';d.mkdir();execute(bench,wrapped,small,time.perf_counter_ns(),physical,d,diagnostic=True)
            transport=None
            if gpu==1:
                from transport import Transport
                transport=Transport(raw,peer,cfg,True);transport.capture(cfg['background']['max_decode_batch'],True)
            pressure=[];target=int(capacity*(.90 if gpu else .79))//len(agents)
            for a in agents:
                pressure.append(dict(a,arrival_offset_ms=0,reserved_peak_tokens=target+19,calls=[
                    dict(input_tokens=1024,output_tokens=16,wait_after_ms=0),
                    dict(input_tokens=target,output_tokens=4,wait_after_ms=0),
                    dict(input_tokens=target+16,output_tokens=4,wait_after_ms=0)]))
            check_cfg=copy.deepcopy(cfg);check_cfg['scope']='Diagnostic only: protected agents,peer migration,heterogeneous prefill,release and restore.'
            check_cfg['background'].update(templates=[dict(input_tokens=4096,output_tokens=256,source_agent='diagnostic',source_round=0,scale=1,forced_token_id=2000)],
                arrival_interval_seconds=.01,first_arrival_seconds=0,protocol_stop_after_foreground_seconds=1.5,isolate_small_foreground_prefill=True)
            from background import replay
            d=folder/'protected_protocol';d.mkdir();check=replay(bench,wrapped,pressure,time.perf_counter_ns(),capacity,d,check_cfg,transport)
            assert check['foreground_retractions']==check['memory_queue_events']==0
            assert check['background_summary']['retractions']>0 and check['background_summary']['recomputed_tokens']>0
            if transport is not None:
                assert check['remote_tokens']>0
                assert transport.decode_checks>0 and transport.prefill_checks>0,dict(decode=transport.decode_checks,prefill=transport.prefill_checks)
                assert any(v['custom_path']for v in transport.control.prefill_checks),'Must check a remote-bearing prefill'
                transport.capture(cfg['background']['max_decode_batch'],False)
            save(folder/'ready.json',dict(ready=True,gpu=gpu,protected_foreground_protocol_passed=True,
                actual_peer_protocol_passed=(transport is not None),physical_capacity_tokens=physical,logical_capacity_tokens=capacity))
            start=root/'start.json'
            while not start.exists():
                if stop.is_set():raise RuntimeError('Peer worker failed')
                time.sleep(.1)
            origin=json.loads(start.read_text())['perf_ns'];delay=(origin-time.perf_counter_ns())/1e9
            if delay>0:time.sleep(delay)
            result=replay(bench,wrapped,agents,origin,capacity,folder,cfg,transport)
        if gpu==0:
            assert receive(connections,stop)['type']=='closed'
            owner['cuda'].free(owner['address']);connections.send(dict(type='freed'))
            save(root/'owner_0.json',dict(owner=0,model_loaded=True,model_forwards=result['forwards'],released=True))
        else:close_banks(cuda,addresses,connections,stop)
        while not(root/'release.json').exists():
            if stop.is_set():raise RuntimeError('Peer worker failed')
            time.sleep(.1)
    sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1];sock.close()
    parser=argparse.ArgumentParser();bench.ServerArgs.add_cli_args(parser);bench.BenchArgs.add_cli_args(parser)
    maximum=cfg['background']['max_decode_batch'];graph=dict(prefill=dict(backend='disabled'),decode=dict(backend='full',max_bs=maximum,bs=list(range(1,maximum+1))))
    argv=['--model-path',cfg['model_path'],'--tp-size','1','--dtype','bfloat16','--kv-cache-dtype','bf16','--attention-backend','flashinfer',
        '--disable-radix-cache','--page-size','1','--port',str(port),'--max-running-requests',str(maximum),
        '--max-total-tokens',str(physical),'--mem-fraction-static','0.85','--context-length','65536',
        '--json-model-override-args',json.dumps(dict(max_position_embeddings=65536)),
        '--cuda-graph-config',json.dumps(graph),'--batch-size',str(len(agents)),'--input-len','1','--output-len','2','--random-seed','20261101']
    args=parser.parse_args(argv);bench.latency_test=campaign
    bench.main(bench.ServerArgs.from_cli_args(args),bench.BenchArgs.from_cli_args(args))
