"""Real autoregressive tokens/s with fixed native KV capacity and idle donor."""
import argparse
import importlib.metadata
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import socket
import sys
import time
import traceback

from experiment_common import base, fixture, save


def tuning_candidates(cfg,batch,remote):
    t=dict(cfg['pure_tuning'] if remote==0 else cfg['selected_tuning'])
    target=max(1,round(192/(batch*8)))
    result=[]
    for n in sorted({max(1,target-1),target,target+1,t['splits']}):
        choice=dict(t)
        if t.get('mode','mixed')=='tier_ctas':
            rs=max(1,round(n*t['remote_splits']/t['splits']))
            choice.update(local_splits=max(1,n-rs),remote_splits=rs)
            choice['splits']=choice['local_splits']+choice['remote_splits']
        else:choice['splits']=n
        if choice not in result:result.append(choice)
    return result


def collect(bench,wrapped,tokenizer,control,cfg,w,root):
    import torch
    from controller import Controller
    raw=control.raw
    batches=sorted({w['native_batch'],w['extended_batch']})
    cache={};tuned={}

    def engine(b,r,t):
        key=(b,r,json.dumps(t,sort_keys=True))
        if key not in cache:cache[key]=control.add_engine(b,r,t)
        return cache[key]

    def active_set(b,r,t=None):
        use_peer=r>0
        out={}
        for size in batches:
            remote=cfg['remote_capacity_tokens']//size if use_peer else 0
            out[size]=engine(size,remote,tuning_candidates(cfg,size,remote)[0])
        out[b]=engine(b,r,t or tuning_candidates(cfg,b,r)[0])
        return out

    def initial(b,r,context,mode,active,trace):
        control.activate(mode,active,trace=trace)
        batch,token,_=base.prepare(bench,wrapped,tokenizer,b,context-1,
                                  cfg['preparation_chunk_tokens'],cfg['decode_steps']+3)
        reclaim=control.offload(batch)
        forward=base.make_forward(bench,raw,batch,token)
        control.plan(batch)
        result=raw.forward(forward)
        assert result.can_run_graph
        return batch,forward,result,reclaim

    # Calibrate only at a real, capacity-valid endpoint. Extra endpoint tokens
    # are generated after prefix offload rather than prefilling beyond C.
    calibrations=[]
    keys=list(dict.fromkeys([(w['native_batch'],0),
         (w['native_batch'],cfg['remote_capacity_tokens']//w['native_batch']),
         (w['extended_batch'],w['required_remote_per_request'])]))
    for b,r in keys:
        choices=tuning_candidates(cfg,b,r)
        active=active_set(b,r,choices[0])
        end=w['final_context']
        start=min(end,cfg['kv_capacity_tokens']//b)-1
        batch,forward,result,reclaim=initial(b,r,start,'custom',active,True)
        token=raw.sample(result.logits_output,forward)
        for context in range(start+1,end+1):
            forward=base.make_forward(bench,raw,batch,token)
            control.plan(batch)
            result=raw.forward(forward)
            token=raw.sample(result.logits_output,forward)
        torch.cuda.synchronize()
        check,oracle=control.check()
        previous=control.outputs
        control.outputs=oracle
        variants=[]
        for t in choices:
            e=engine(b,r,t);e.update(batch)
            variants.append(dict(name=e.label,call=lambda q,i,e=e:e.run(q,i),check=True))
        sampling=dict(cfg,fixture_groups=cfg['calibration_groups'],
                      fixture_replays_per_group=cfg['calibration_replays'],
                      fixture_warmup_replays=cfg['calibration_warmup_replays'])
        name=f'calibration_B{b}_R{r}'
        measured,graphs,markers,keepalive=fixture(control,variants,sampling,root,name)
        best=min(measured['variants'],key=lambda x:x['median_ms'])['name']
        chosen=next(engine(b,r,t) for t in choices if engine(b,r,t).label==best)
        tuned[(b,r)]=chosen
        control.outputs=previous
        calibrations.append(dict(batch=b,remote_per_request=r,selected=chosen.metadata(),oracle=check,
                                 source=name,reclamation=reclaim))
        save(root/'calibrations.json',calibrations)
        print('CALIBRATED',w['name'],b,r,best,flush=True)
        del graphs,markers,keepalive

    cases=[dict(name='native_max',batch=w['native_batch'],remote=0,mode='native'),
           dict(name='unified_local_max',batch=w['native_batch'],remote=0,mode='custom'),
           dict(name='peer_same_batch',batch=w['native_batch'],remote=cfg['remote_capacity_tokens']//w['native_batch'],mode='custom'),
           dict(name='peer_max',batch=w['extended_batch'],remote=w['required_remote_per_request'],
                mode='native' if w['required_remote_per_request']==0 else 'custom')]
    rng=random.Random(cfg['random_seed'])
    trials=[]
    for repetition in range(cfg['trials']):
        order=cases.copy();rng.shuffle(order)
        for case in order:
            b,r=case['batch'],case['remote']
            active=active_set(b,r)
            if case['mode']!='native':active[b]=tuned[(b,r)]
            # An instrumented first forward is checked, then recapture without
            # query/output copies or per-layer Events for throughput collection.
            batch,forward,result,reclaim=initial(b,r,w['initial_context'],case['mode'],active,True)
            check_start,_=control.check()
            control.activate(case['mode'],active,trace=False)
            control.plan(batch)
            for _ in range(cfg['warmup_steps']):result=raw.forward(forward)
            token=raw.sample(result.logits_output,forward)
            inputs=torch.empty((cfg['decode_steps'],b),device=raw.device,dtype=torch.int64)
            outputs=torch.empty_like(inputs)
            ends=[]
            whole_start=torch.cuda.Event(enable_timing=True);whole_end=torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            began_ns=time.time_ns();began=time.perf_counter()
            whole_start.record()
            bin_start=began
            for step in range(cfg['decode_steps']):
                inputs[step].copy_(token)
                forward=base.make_forward(bench,raw,batch,token)
                control.plan(batch,indices=False)
                result=raw.forward(forward)
                assert result.can_run_graph
                token=raw.sample(result.logits_output,forward)
                outputs[step].copy_(token)
                if (step+1)%64==0:
                    torch.cuda.synchronize()
                    now=time.perf_counter()
                    ends.append(dict(first_step=step-62,last_step=step+1,wall_seconds=now-bin_start,
                                     output_tokens=64*b,tokens_per_second=64*b/(now-bin_start),
                                     first_context=w['initial_context']+step-62,
                                     last_context=w['initial_context']+step+1))
                    bin_start=now
            whole_end.record();whole_end.synchronize()
            elapsed=time.perf_counter()-began;ended_ns=time.time_ns()
            page_audit=control.audit_pages(batch)
            control.plan(batch)
            actual_context=batch.seq_lens_cpu.tolist()
            assert actual_context==[w['final_context']]*b
            used=cfg['kv_capacity_tokens']-raw.token_to_kv_pool_allocator.available_size()
            assert used==b*(w['final_context']-r) and used<=cfg['kv_capacity_tokens']
            assert torch.unique(control.local_indices).numel()==used
            consumed,generated=inputs.cpu().tolist(),outputs.cpu().tolist()
            assert consumed[1:]==generated[:-1]
            assert all(0<=t<raw.model_config.hf_config.vocab_size for row in generated for t in row)
            assert torch.isfinite(result.logits_output.next_token_logits).all().item()
            control.activate(case['mode'],active,trace=True)
            control.plan(batch);raw.forward(forward);torch.cuda.synchronize()
            check_end,_=control.check()
            payload=dict(workload=w,name=case['name'],trial=repetition,batch=b,remote_per_request=r,
                attention_mode=case['mode'],
                generated_tokens=b*cfg['decode_steps'],decode_steps=cfg['decode_steps'],
                wall_seconds=elapsed,tokens_per_second=b*cfg['decode_steps']/elapsed,
                gpu_stream_interval_ms=whole_start.elapsed_time(whole_end),
                near_full_tail_tokens_per_second=ends[-1]['tokens_per_second'],bins=ends,
                start_unix_ns=began_ns,end_unix_ns=ended_ns,
                local_active_tokens=used,local_capacity_tokens=cfg['kv_capacity_tokens'],
                remote_active_tokens=b*r,remote_allocated_tokens=cfg['remote_capacity_tokens'],
                reclamation=reclaim,correctness=[check_start,check_end],
                page_audit=page_audit,
                selected_tuning=None if case['mode']=='native' else active[b].metadata(),
                primary_query_copies=False,primary_per_layer_events=False,local_shadow_present=False,
                scope='Real native prepare_for_decode + full Qwen forward + greedy sampling + page-table updates and token logging; 64-step synchronization bins included. Prefill/offload/calibration/oracle/capture/warmup excluded. Decode throughput, not online arrival/queue throughput.')
            stem=f"{case['name']}_trial{repetition}"
            save(root/(stem+'_tokens.json'),dict(input_token_ids=consumed,output_token_ids=generated))
            save(root/(stem+'.json'),payload)
            trials.append(payload)
            print('THROUGHPUT',w['name'],case['name'],repetition,payload['tokens_per_second'],flush=True)
    save(root/'workload_collection.json',dict(status='WORKLOAD_COMPLETE',workload=w,calibrations=calibrations,trials=trials))


def consumer(cfg,w,root,connection,stop):
    import torch
    from sglang.benchmark import one_batch as bench
    from controller import Controller
    from cuda_ipc import Cuda
    from measure import mapped_tensor
    torch.cuda.set_device(1);torch.manual_seed(cfg['random_seed'])

    def campaign(sa,pa,ba,gpu,rank):
        bench.publish(sa,role='scheduler',ranks=bench.SpawnRanks(world_rank=bench.spawn_world_rank(sa,tp_rank=0,pp_rank=0),gpu_id=1))
        bench.initialize_moe_config();bench.initialize_fp8_gemm_config();bench.initialize_fp4_gemm_config()
        bench.configure_logger(sa,prefix=' THROUGHPUT')
        wrapped,tokenizer=bench.load_model(sa,pa,1,0);raw=wrapped.torch_runner
        assert raw.max_total_num_tokens==cfg['kv_capacity_tokens'] and raw.dtype==raw.kv_cache_dtype==torch.bfloat16
        hf=raw.model_config.hf_config
        assert (hf.num_hidden_layers,hf.num_attention_heads,hf.num_key_value_heads,hf.head_dim)==(36,32,8,128)
        assert list(raw.decode_cuda_graph_runner.capture_bs)==sorted({w['native_batch'],w['extended_batch']})
        assert sum(x.numel()*x.element_size() for i in range(36) for x in raw.token_to_kv_pool.get_kv_buffer(i))==(cfg['kv_capacity_tokens']+1)*147456
        allocation=base.receive(connection,stop)
        assert allocation['type']=='allocation' and allocation['slots']==cfg['remote_capacity_tokens']
        cuda=Cuda();cuda.enable_peer(1,0);address=cuda.open(allocation['handle'])
        peer=mapped_tensor(address,allocation['slots'],1)
        assert float(peer.flatten()[0].item())==1.5
        peer.flatten()[0].fill_(2.25);torch.cuda.synchronize()
        connection.send(dict(type='probe_written'))
        assert base.receive(connection,stop)['type']=='verified'
        save(root/'mapping.json',dict(allocation=allocation,imported_pointer=address,tensor_pointer=peer.data_ptr(),
             local_capacity_tokens=cfg['kv_capacity_tokens'],versions={k:importlib.metadata.version(k) for k in ('torch','sglang','flashinfer-python')}))
        with torch.no_grad():collect(bench,wrapped,tokenizer,Controller(raw,peer,cfg,w),cfg,w,root)
        torch.cuda.synchronize();cuda.close(address)
        connection.send(dict(type='closed'));assert base.receive(connection,stop)['type']=='freed'

    sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1];sock.close()
    parser=argparse.ArgumentParser();bench.ServerArgs.add_cli_args(parser);bench.BenchArgs.add_cli_args(parser)
    sizes=sorted({w['native_batch'],w['extended_batch']})
    graph=dict(prefill=dict(backend='disabled'),decode=dict(backend='full',max_bs=max(sizes),bs=sizes))
    argv=['--model-path',cfg['model_path'],'--tp-size','1','--dtype','bfloat16','--kv-cache-dtype','bf16',
          '--attention-backend','flashinfer','--disable-radix-cache','--page-size','1','--port',str(port),
          '--mem-fraction-static',str(cfg['mem_fraction_static']),'--max-total-tokens',str(cfg['kv_capacity_tokens']),
          '--max-running-requests',str(max(sizes)),'--context-length',str(cfg['context_length_limit']),
          '--cuda-graph-config',json.dumps(graph),'--batch-size',str(max(sizes)),'--input-len','1','--output-len','2',
          '--random-seed',str(cfg['random_seed'])]
    args=parser.parse_args(argv);bench.latency_test=campaign
    bench.main(bench.ServerArgs.from_cli_args(args),bench.BenchArgs.from_cli_args(args))


def worker(role,cfg,w,root,connection,stop):
    with (root/f'worker_{role}.log').open('w',buffering=1) as log:
        os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
        try:
            if role==1:return consumer(cfg,w,root,connection,stop)
            import torch
            from cuda_ipc import Cuda
            from measure import mapped_tensor
            torch.cuda.set_device(0);cuda=Cuda();cuda.set_device(0)
            slots=cfg['remote_capacity_tokens'];address,handle=cuda.allocate(slots*147456)
            peer=mapped_tensor(address,slots,0);peer.zero_();peer.flatten()[0].fill_(1.5);torch.cuda.synchronize()
            connection.send(dict(type='allocation',slots=slots,bytes=slots*147456,handle=handle,owner=cuda.attributes(address)))
            assert base.receive(connection,stop)['type']=='probe_written'
            assert float(peer.flatten()[0].item())==2.25
            connection.send(dict(type='verified'))
            assert base.receive(connection,stop)['type']=='closed'
            cuda.free(address);connection.send(dict(type='freed'))
        except BaseException:
            save(root/f'worker_{role}_failure.json',dict(error=traceback.format_exc()));stop.set();raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();cfg=json.loads(args.config.read_text())
    assert cfg['selected_tuning'] and cfg['remote_capacity_tokens']==5888
    args.out.mkdir(parents=True,exist_ok=False);save(args.out/'config.json',cfg);base.topology(args.out)
    ctx=mp.get_context('spawn');completed=[]
    for w in cfg['workloads']:
        root=args.out/w['name'];root.mkdir()
        stop=ctx.Event();left,right=ctx.Pipe(duplex=True)
        processes=[ctx.Process(target=worker,args=(role,cfg,w,root,c,stop)) for role,c in enumerate((left,right))]
        for process in processes:process.start()
        left.close();right.close()
        for process in processes:process.join()
        assert all(process.exitcode==0 for process in processes),[p.exitcode for p in processes]
        assert json.loads((root/'workload_collection.json').read_text())['status']=='WORKLOAD_COMPLETE'
        completed.append(w['name']);save(args.out/'progress.json',dict(completed=completed))
    save(args.out/'throughput_collection.json',dict(status='THROUGHPUT_COLLECTION_COMPLETE',completed=completed,config=cfg))
    print('THROUGHPUT_COLLECTION_COMPLETE',flush=True)
