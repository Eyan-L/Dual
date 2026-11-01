"""Two independent TP1 Qwen3 instances, CUDA IPC donor memory, direct peer attention."""
import argparse
import csv
from io import StringIO
import json
import multiprocessing as mp
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parent))
from cuda_ipc import Cuda


def save(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def receive(connection,stop):
    while not connection.poll(.2):
        if stop.is_set():raise RuntimeError('Peer worker stopped before the IPC handshake completed')
    return connection.recv()


def topology(root):
    query=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,pci.bus_id,name','--format=csv,noheader'],text=True)
    matrix=subprocess.check_output(['nvidia-smi','topo','-m'],text=True)
    (root/'gpu_inventory.csv').write_text(query);(root/'topology.txt').write_text(matrix)
    devices=list(csv.reader(StringIO(query)))
    visible=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    assert len(visible)==2,visible
    selected=[]
    for name in visible:
        matches=[r for r in devices if r[0].strip()==name or r[1].strip().startswith(name)]
        assert len(matches)==1,(name,devices)
        selected.append(dict(index=int(matches[0][0]),uuid=matches[0][1].strip(),pci_bus_id=matches[0][2].strip(),name=matches[0][3].strip()))
    header=next(line.split() for line in matrix.splitlines() if line.split() and line.split()[0]=='GPU0')
    row=next(line.split() for line in matrix.splitlines() if len(line.split())>1
             and line.split()[0]==f"GPU{selected[0]['index']}" and not line.split()[1].startswith('GPU'))
    connection=row[1+header.index(f"GPU{selected[1]['index']}")]
    assert connection.startswith('NV'),f'Allocated pair is {connection}, not an NVLink pair'
    assert all('A100' in d['name'] for d in selected)
    result=dict(visible_devices=visible,devices=selected,allocated_pair_topology=connection)
    save(root/'allocated_topology.json',result)
    return result


def nvlink_snapshot(uuid):
    result=dict(cpu_unix_ns=time.time_ns(),fields=[])
    try:
        import pynvml as n
        n.nvmlInit();handle=n.nvmlDeviceGetHandleByUUID(uuid)
        links=[]
        for link in range(n.NVML_NVLINK_MAX_LINKS):
            try:
                if n.nvmlDeviceGetNvLinkState(handle,link):links.append(link)
            except n.NVMLError:pass
        fields=[(field,link) for link in links for field in (n.NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX,n.NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX)]
        for f in n.nvmlDeviceGetFieldValues(handle,fields):
            result['fields'].append(dict(field_id=f.fieldId,scope_link=f.scopeId,timestamp_us=f.timestamp,
                                         status=f.nvmlReturn,value_type=f.valueType,unsigned64_value=int(f.value.ullVal)))
        result['active_links']=links
        n.nvmlShutdown()
    except Exception:
        result['error']=traceback.format_exc()
    return result


def make_forward(bench,raw,batch,token):
    import torch
    batch.input_ids=token.to(torch.int64)
    batch.prepare_for_decode()
    bench._maybe_prepare_mlp_sync_batch(batch,raw)
    return bench.ForwardBatch.init_new(batch,raw,return_hidden_states_before_norm=False)


def worker_body(role,cfg,root,connection,stop,device_info):
    import torch
    from sglang.benchmark import one_batch as bench
    from prefill import prepare
    torch.manual_seed(cfg['random_seed'])
    torch.cuda.set_device(role)
    assert torch.cuda.device_count()==2

    def campaign(sa,pa,ba,unused_gpu,rank):
        from peer_attention import write_probe,read_probe
        bench.publish(sa,role='scheduler',ranks=bench.SpawnRanks(
            world_rank=bench.spawn_world_rank(sa,tp_rank=0,pp_rank=0),gpu_id=role))
        bench.initialize_moe_config();bench.initialize_fp8_gemm_config();bench.initialize_fp4_gemm_config()
        bench.configure_logger(sa,prefix=' ONPATH' if role==0 else ' OFFPATH')
        wrapped,tokenizer=bench.load_model(sa,pa,role,0)
        raw=wrapped.torch_runner
        assert int(torch.cuda.current_device())==role
        assert raw.max_total_num_tokens==cfg['kv_capacity_tokens']
        remote=cfg['offpath_remote_context']
        total_context=cfg['offpath_local_context']+remote
        cuda=Cuda();cuda.set_device(role)
        props=torch.cuda.get_device_properties(role)
        info=dict(role='onpath' if role==0 else 'offpath',logical_device=role,
                  device_name=props.name,process_id=os.getpid(),model='Qwen3-8B',tp_size=1)
        save(root/(info['role']+'_device.json'),info)
        with torch.no_grad():
            if role==0:
                batch,token,_=prepare(bench,wrapped,tokenizer,1,cfg['onpath_context'],cfg['preparation_chunk_tokens'],32768-cfg['onpath_context'])
                size=36*2*remote*8*128*2
                address,handle=cuda.allocate(size)
                attributes=cuda.attributes(address)
                assert attributes['memory_type']==2 and attributes['owner_logical_device']==0
                save(root/'donor_allocation.json',dict(bytes=size,owner=attributes,retains_own_model_and_KV=True))
                write_probe(address,1.5)
                connection.send(dict(type='allocation',handle=handle,bytes=size,owner=attributes))
                message=receive(connection,stop);assert message['type']=='ipc_probe_written'
                owner_observed=read_probe(address)
                assert owner_observed==2.25
                save(root/'ipc_shared_memory_proof.json',dict(owner_written=1.5,consumer_observed=message['consumer_observed'],
                     consumer_written=2.25,owner_observed=owner_observed,owner_logical_device=0,consumer_logical_device=1,
                     mechanism='GPU kernels read/write the same IPC allocation in both processes; proof completed before KV population.'))
                connection.send(dict(type='ipc_probe_verified'))
                message=receive(connection,stop);assert message['type']=='start'
                rows=[];start_unix=time.time_ns()
                while not connection.poll():
                    if stop.is_set():raise RuntimeError('Offpath failed during concurrent generation')
                    if len(rows)>=32768-cfg['onpath_context']-1:
                        raise RuntimeError('Onpath reached native context capacity before offpath finished')
                    forward=make_forward(bench,raw,batch,token)
                    submitted_unix=time.time_ns()
                    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    a.record();result=raw.forward(forward)
                    assert result.can_run_graph
                    token=raw.sample(result.logits_output,forward)
                    b.record();b.synchronize()
                    rows.append(dict(step=len(rows),context=int(batch.seq_lens_cpu[0]),submitted_unix_ns=submitted_unix,
                                     completed_unix_ns=time.time_ns(),gpu_round_ms=a.elapsed_time(b),token_id=int(token.item())))
                message=connection.recv();assert message['type']=='done'
                torch.cuda.synchronize()
                assert rows,'Onpath did not execute concurrently'
                save(root/'onpath.json',dict(status='ONPATH_COMPLETE',start_unix_ns=start_unix,end_unix_ns=time.time_ns(),steps=rows))
                # The consumer acknowledges GPU completion before closing/freeing IPC memory.
                connection.send(dict(type='onpath_complete',steps=len(rows)))
                message=receive(connection,stop);assert message['type']=='ipc_closed'
                cuda.free(address)
                connection.send(dict(type='owner_freed'))
            else:
                from peer_attention import PeerAttention
                batch,token,_=prepare(bench,wrapped,tokenizer,1,total_context,cfg['preparation_chunk_tokens'],cfg['offpath_decode_steps']+1)
                allocation=receive(connection,stop);assert allocation['type']=='allocation'
                cuda.enable_peer(1,0)
                address=cuda.open(allocation['handle'])
                attributes=cuda.attributes(address)
                assert attributes['memory_type']==2 and allocation['owner']['owner_logical_device']==0,attributes
                observed=read_probe(address);assert observed==1.5
                write_probe(address,2.25)
                connection.send(dict(type='ipc_probe_written',consumer_observed=observed))
                message=receive(connection,stop);assert message['type']=='ipc_probe_verified'
                save(root/'consumer_mapping.json',dict(imported_pointer_attributes=attributes,
                     reported_device_is_import_context=True,backing_owner_from_export=allocation['owner'],compute_logical_device=1,
                     mechanism='cudaIpcOpenMemHandle + peer tl.load; owner allocation is device memory, not managed/host memory'))
                controller=PeerAttention(raw,address,remote,1,total_context+cfg['offpath_decode_steps'])
                controller.populate(batch)
                forward=make_forward(bench,raw,batch,token)
                native=raw.forward(forward).logits_output.next_token_logits.detach().clone()
                controller.install();controller.validate_native=True
                raw.forward(forward)
                controller.validate_native=False
                local=raw.forward(forward).logits_output.next_token_logits.detach().clone()
                prefix_audit=controller.audit_prefix(forward)
                save(root/'remote_kv_audit.json',dict(layers=prefix_audit,all_BF16_payloads_equal=True,
                     scope='Diagnostic readback only before generation; attention itself reads peer memory directly.'))
                controller.mode='peer'
                assert len(controller.native_checks)==36
                result=raw.forward(forward)
                peer=result.logits_output.next_token_logits.detach().clone()
                torch.cuda.synchronize()
                assert torch.isfinite(peer).all().item()
                torch.testing.assert_close(peer,local,rtol=.005,atol=.02)
                assert int(peer.argmax().item())==int(native.argmax().item())
                first_layer=raw.model.model.layers[0].self_attn.attn
                sensitivity=controller.sensitivity_check(first_layer)
                save(root/'correctness.json',dict(status='PEER_ATTENTION_CORRECT',
                    local_reference_vs_peer_max_logit_error=float((peer-local).abs().max().item()),
                    local_reference_vs_peer_bitwise_equal=bool(torch.equal(peer,local)),
                    initial_greedy_token_matches_native=True,native_attention_checks=controller.native_checks,
                    sensitivity=sensitivity,
                    local_context_tokens=cfg['offpath_local_context'],remote_context_tokens=remote,
                    local_reference_cache_preserved=True,capacity_reclamation_implemented=False))
                if controller.remote_compiled is not None:
                    (root/'remote_partial.ptx').write_text(controller.remote_compiled.asm['ptx'])
                snapshots=dict(before=nvlink_snapshot(device_info['uuid']))
                connection.send(dict(type='start'))
                rows=[];start_unix=time.time_ns()
                # Same prepared token is executed once more after validation;
                # the native sequence then advances exactly once per next round.
                for step in range(cfg['offpath_decode_steps']):
                    if step:
                        forward=make_forward(bench,raw,batch,token)
                    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    submitted_unix=time.time_ns()
                    a.record()
                    result=raw.forward(forward)
                    token=raw.sample(result.logits_output,forward)
                    b.record();b.synchronize()
                    rows.append(dict(step=step,context=int(batch.seq_lens_cpu[0]),gpu_round_ms=a.elapsed_time(b),
                                     submitted_unix_ns=submitted_unix,completed_unix_ns=time.time_ns(),token_id=int(token.item())))
                torch.cuda.synchronize()
                end_unix=time.time_ns()
                snapshots['after']=nvlink_snapshot(device_info['uuid'])
                save(root/'nvlink_fields.json',snapshots)
                assert controller.remote_calls>=36*cfg['offpath_decode_steps']
                save(root/'offpath.json',dict(status='OFFPATH_COMPLETE',start_unix_ns=start_unix,end_unix_ns=end_unix,
                     steps=rows,remote_attention_calls=controller.remote_calls,
                     mechanism='Two streams compute local and direct-peer partial attention, then merge FP32 softmax states. No KV staging into offpath HBM during attention.',
                     caveat='Functional Triton prototype is not tuned; timings do not establish latency hiding or effective extra capacity. Full local reference retained for validation.'))
                connection.send(dict(type='done'))
                message=receive(connection,stop);assert message['type']=='onpath_complete'
                cuda.close(address)
                connection.send(dict(type='ipc_closed'))
                message=receive(connection,stop);assert message['type']=='owner_freed'

    graph=dict(prefill=dict(backend='disabled'),decode=dict(backend='full' if role==0 else 'disabled',max_bs=1,bs=[1]))
    parser=argparse.ArgumentParser()
    bench.ServerArgs.add_cli_args(parser);bench.BenchArgs.add_cli_args(parser)
    port_socket=socket.socket();port_socket.bind(('127.0.0.1',0));port=port_socket.getsockname()[1];port_socket.close()
    argv=['--model-path',cfg['model_path'],'--tp-size','1','--dtype','bfloat16','--kv-cache-dtype','bf16',
          '--attention-backend','flashinfer','--disable-radix-cache','--page-size','1','--port',str(port),
          '--mem-fraction-static',str(cfg['mem_fraction_static']),'--max-total-tokens',str(cfg['kv_capacity_tokens']),
          '--max-running-requests','1','--context-length',str(cfg['context_length_limit']),
          '--cuda-graph-config',json.dumps(graph),'--batch-size','1','--input-len','1','--output-len','33',
          '--random-seed',str(cfg['random_seed'])]
    args=parser.parse_args(argv)
    bench.latency_test=campaign
    bench.main(bench.ServerArgs.from_cli_args(args),bench.BenchArgs.from_cli_args(args))


def worker(role,cfg,root,connection,stop,device_info):
    os.environ['TRITON_CACHE_DIR']=os.environ.get('XDG_CACHE_HOME','/tmp')+f'/peer_triton_{role}'
    with (root/('onpath.log' if role==0 else 'offpath.log')).open('w',buffering=1) as log:
        os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
        try:
            worker_body(role,cfg,root,connection,stop,device_info)
        except BaseException:
            save(root/f'worker_{role}_failure.json',dict(error=traceback.format_exc()))
            stop.set()
            raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();cfg=json.loads(args.config.read_text())
    prerequisite=json.loads(Path(cfg['prerequisite_summary']).read_text())
    assert prerequisite['status']=='GROWTH_ANALYSIS_PASSED','Experiment3 must complete successfully before this prototype'
    args.out.mkdir(parents=True,exist_ok=False)
    topo=topology(args.out)
    ctx=mp.get_context('spawn');stop=ctx.Event();left,right=ctx.Pipe(duplex=True)
    processes=[ctx.Process(target=worker,args=(role,cfg,args.out,conn,stop,topo['devices'][role])) for role,conn in enumerate((left,right))]
    for process in processes:process.start()
    left.close();right.close()
    for process in processes:process.join()
    assert all(p.exitcode==0 for p in processes),[p.exitcode for p in processes]
    on=json.loads((args.out/'onpath.json').read_text());off=json.loads((args.out/'offpath.json').read_text())
    assert on['start_unix_ns']<off['end_unix_ns'] and off['start_unix_ns']<on['end_unix_ns']
    assert any(off['start_unix_ns']<=s['completed_unix_ns']<=off['end_unix_ns'] for s in on['steps'])
    save(args.out/'summary.json',dict(status='NVLINK_PEER_ATTENTION_PROTOTYPE_PASSED',config=cfg,topology=topo,
         onpath_steps=len(on['steps']),offpath_steps=len(off['steps']),concurrent_execution_verified=True,
         peer_owner_gpu=0,attention_compute_gpu=1,full_local_reference_preserved=True,
         latency_hiding_proven=False,capacity_reclamation_implemented=False))
    print('NVLINK_PEER_ATTENTION_PROTOTYPE_PASSED',flush=True)


if __name__=='__main__':
    main()
