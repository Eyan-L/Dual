"""Native FlashInfer attention: full local pool versus genuinely growing peer-only KV."""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import socket
import statistics
import sys
import time
import traceback
import types

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cuda_ipc import Cuda
from prefill import prepare
from prototype import make_forward, receive, save, topology


class BorrowedCUDA:
    def __init__(self, address, size):
        self.__cuda_array_interface__ = dict(shape=(size,), typestr='|u1',
                                             data=(address, False), version=3)


def mapped_tensor(address, slots, device):
    import torch
    size = 36*2*slots*8*128*2
    owner = BorrowedCUDA(address, size)
    byte_view = torch.as_tensor(owner, device=f'cuda:{device}')
    assert byte_view.data_ptr() == address, 'IPC import must be a zero-copy view'
    result = byte_view.view(torch.bfloat16).view(36, 2, slots, 8, 128)
    assert result.data_ptr() == address and result.device.index == device
    return result


class AttentionEvents:
    def __init__(self, raw, peer, batch):
        import torch
        self.raw, self.peer, self.batch = raw, peer, batch
        self.backend = raw.decode_attn_backend or raw.attn_backend
        assert raw.decode_cuda_graph_runner._replay_attn_backend() is self.backend
        self.events = [(torch.cuda.Event(enable_timing=True, external=True),
                        torch.cuda.Event(enable_timing=True, external=True)) for _ in range(36)]
        self.queries = torch.empty((36, batch, 32, 128), device=raw.device, dtype=torch.bfloat16)
        self.outputs = torch.empty_like(self.queries)
        self.mode = 'local'
        self.peer_buffers = [(peer[layer,0].unsqueeze(1), peer[layer,1].unsqueeze(1)) for layer in range(36)]
        self.original = self.backend.forward_decode
        controller = self

        def forward_decode(backend, q, k, v, layer, forward_batch, save_kv_cache=True):
            assert layer.logit_cap in (None, 0, 0.0)
            i = layer.layer_id
            if controller.mode == 'remote':
                assert k is not None and v is not None and save_kv_cache
            if k is not None and save_kv_cache:
                backend.token_to_kv_pool.set_kv_buffer(layer, forward_batch.out_cache_loc, k, v,
                                                       layer.k_scale_float, layer.v_scale_float)
                if controller.mode == 'remote':
                    loc = forward_batch.out_cache_loc.to(torch.int64)
                    controller.peer[i,0].index_copy_(0, loc, k.reshape(-1,8,128))
                    controller.peer[i,1].index_copy_(0, loc, v.reshape(-1,8,128))
            query = q.contiguous().view(-1,32,128)
            controller.queries[i].copy_(query)
            buffers = (backend.token_to_kv_pool.get_kv_buffer(i) if controller.mode == 'local'
                       else controller.peer_buffers[i])
            wrapper = backend.forward_metadata.decode_wrappers[backend._get_wrapper_idx(layer)]
            start, end = controller.events[i]
            start.record()
            output = wrapper.forward(query, buffers, sm_scale=layer.scaling, logits_soft_cap=layer.logit_cap,
                                     k_scale=layer.k_scale_float, v_scale=layer.v_scale_float)
            end.record()
            controller.outputs[i].copy_(output)
            return output.view(-1,4096)

        self.backend.forward_decode = types.MethodType(forward_decode, self.backend)

    def recapture(self, mode):
        import torch
        self.mode = mode
        torch.cuda.synchronize()
        self.raw.decode_cuda_graph_runner.capture()
        torch.cuda.synchronize()

    def times(self):
        return [a.elapsed_time(b) for a,b in self.events]

    def populate(self, batch):
        import torch
        slots = self.raw.req_to_token_pool.req_to_token[batch.req_pool_indices, :int(batch.seq_lens_cpu[0])].reshape(-1).to(torch.int64)
        assert slots.min().item() > 0 and slots.max().item() < self.peer.shape[2]
        for layer in range(36):
            k,v = self.raw.token_to_kv_pool.get_kv_buffer(layer)
            self.peer[layer,0].index_copy_(0, slots, k.index_select(0,slots))
            self.peer[layer,1].index_copy_(0, slots, v.index_select(0,slots))
        torch.cuda.synchronize()

    def check(self):
        import torch
        rows = []
        graph = self.raw.decode_cuda_graph_runner
        backend = graph._replay_attn_backend()
        wrappers = backend.decode_cuda_graph_metadata[self.batch]
        for i in range(36):
            layer = self.raw.model.model.layers[i].self_attn.attn
            wrapper = wrappers[backend._get_wrapper_idx(layer)]
            expected = wrapper.forward(self.queries[i], self.raw.token_to_kv_pool.get_kv_buffer(i),
                                       sm_scale=layer.scaling, logits_soft_cap=layer.logit_cap,
                                       k_scale=layer.k_scale_float, v_scale=layer.v_scale_float)
            torch.testing.assert_close(self.outputs[i], expected, atol=.01, rtol=.02)
            rows.append(dict(layer=i,max_abs_error=float((self.outputs[i]-expected).abs().max().item())))
        return rows


def timed_forward(raw, forward, controller=None):
    import torch
    start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    start.record(); result=raw.forward(forward); end.record(); end.synchronize()
    assert result.can_run_graph and raw.decode_cuda_graph_runner.bs == forward.batch_size
    value = dict(forward_ms=start.elapsed_time(end))
    if controller:
        values=controller.times()
        assert len(values)==36 and min(values)>0
        value.update(attention_layer_ms=values,attention_sum_ms=sum(values))
    return result,value


def consumer(cfg, root, connection, stop):
    import torch
    import importlib.metadata
    from sglang.benchmark import one_batch as bench
    torch.cuda.set_device(1); torch.manual_seed(cfg['random_seed'])

    def campaign(sa,pa,ba,gpu,rank):
        bench.publish(sa,role='scheduler',ranks=bench.SpawnRanks(
            world_rank=bench.spawn_world_rank(sa,tp_rank=0,pp_rank=0),gpu_id=1))
        bench.initialize_moe_config();bench.initialize_fp8_gemm_config();bench.initialize_fp4_gemm_config()
        bench.configure_logger(sa,prefix=' ATTN-MATCH')
        wrapped,tokenizer=bench.load_model(sa,pa,1,0)
        raw=wrapped.torch_runner
        batch_size=cfg['batch']; capacity=cfg['kv_capacity_tokens']
        assert raw.max_total_num_tokens==capacity
        assert list(raw.decode_cuda_graph_runner.capture_bs)==[batch_size]
        assert raw.dtype==raw.kv_cache_dtype==torch.bfloat16
        hf=raw.model_config.hf_config
        assert (hf.num_hidden_layers,hf.num_attention_heads,hf.num_key_value_heads,hf.head_dim)==(36,32,8,128)
        allocation=receive(connection,stop);assert allocation['type']=='allocation'
        cuda=Cuda();cuda.enable_peer(1,0); address=cuda.open(allocation['handle'])
        attrs=cuda.attributes(address)
        assert attrs['memory_type']==2
        peer=mapped_tensor(address,allocation['slots'],1)
        observed=float(peer.flatten()[0].item());assert observed==1.5
        peer.flatten()[0].fill_(2.25);torch.cuda.synchronize()
        connection.send(dict(type='probe_written',observed=observed))
        message=receive(connection,stop);assert message['type']=='verified'
        save(root/'mapping.json',dict(export=allocation,imported_pointer=address,imported_attributes=attrs,
             zero_copy_tensor_pointer=peer.data_ptr(),compute_device=1,backing_owner_device=0,
             versions={k:importlib.metadata.version(k) for k in ('torch','sglang','flashinfer-python')},
             local_pool_capacity_tokens=capacity,
             local_pool_allocated_bytes=sum(x.numel()*x.element_size() for i in range(36) for x in raw.token_to_kv_pool.get_kv_buffer(i))))
        with torch.no_grad():
            # First measure full-pool native decode without inserted attention markers.
            batch,token,_=prepare(bench,wrapped,tokenizer,batch_size,cfg['local_full_context']-1,
                                  cfg['preparation_chunk_tokens'],2)
            forward=make_forward(bench,raw,batch,token)
            assert capacity-raw.token_to_kv_pool_allocator.available_size()==capacity
            for _ in range(cfg['warmup_steps']): timed_forward(raw,forward)
            uninstrumented=[timed_forward(raw,forward)[1]['forward_ms'] for _ in range(30)]
            controller=AttentionEvents(raw,peer,batch_size)
            controller.recapture('local')
            for _ in range(cfg['warmup_steps']): timed_forward(raw,forward,controller)
            baseline=[dict(sequence=i,**timed_forward(raw,forward,controller)[1]) for i in range(cfg['baseline_samples'])]
            checks=controller.check()
            local_time=statistics.median(r['attention_sum_ms'] for r in baseline)
            save(root/'baseline.json',dict(status='LOCAL_FULL_POOL_MEASURED',batch=batch_size,
                 context=cfg['local_full_context'],active_kv_tokens=capacity,active_kv_bytes=capacity*147456,
                 attention_median_ms=local_time,samples=baseline,native_output_checks=checks,
                 uninstrumented_forward_ms=uninstrumented,
                 instrumentation_forward_time_ratio=statistics.median(r['forward_ms'] for r in baseline)/statistics.median(uninstrumented)))
            print('LOCAL_BASELINE_MS',local_time,flush=True)

            controller.recapture('remote')
            batch,token,_=prepare(bench,wrapped,tokenizer,batch_size,cfg['remote_start_context'],
                                  cfg['preparation_chunk_tokens'],cfg['warmup_steps']+1)
            controller.populate(batch)
            for _ in range(cfg['warmup_steps']):
                forward=make_forward(bench,raw,batch,token)
                result,_=timed_forward(raw,forward,controller)
                token=raw.sample(result.logits_output,forward)

            batch,token,prompt=prepare(bench,wrapped,tokenizer,batch_size,cfg['remote_start_context'],
                                       cfg['preparation_chunk_tokens'],cfg['remote_max_context']-cfg['remote_start_context']+1)
            controller.populate(batch)
            save(root/'prompt_ids.json',dict(per_request_prompt=prompt,prefill_seed=token.cpu().tolist()))
            window=[]; previous_below=False; reason='maximum_remote_context'; count=0; correctness=[]
            started=time.time_ns()
            with (root/'remote_steps.jsonl').open('w',buffering=1) as output:
                for context in range(cfg['remote_start_context']+1,cfg['remote_max_context']+1):
                    consumed=token.cpu().tolist()
                    forward=make_forward(bench,raw,batch,token)
                    assert batch.seq_lens_cpu.tolist()==[context]*batch_size
                    used=capacity-raw.token_to_kv_pool_allocator.available_size()
                    assert used==batch_size*context and int(forward.out_cache_loc.max())<allocation['slots']
                    result,row=timed_forward(raw,forward,controller)
                    token=raw.sample(result.logits_output,forward)
                    generated=token.cpu().tolist()
                    assert torch.isfinite(result.logits_output.next_token_logits).all().item()
                    row.update(sequence=count,context=context,batch=batch_size,active_remote_tokens=used,
                               active_remote_bytes=used*147456,extra_fraction=used/capacity,
                               remote_share_if_appended=used/(capacity+used),input_token_ids=consumed,
                               output_token_ids=generated,completed_unix_ns=time.time_ns())
                    output.write(json.dumps(row,separators=(',',':'))+'\n');window.append(row);count+=1
                    if count==1:
                        correctness.append(dict(context=context,layers=controller.check()))
                    if len(window)==cfg['context_bin_tokens']:
                        value=statistics.median(r['attention_sum_ms'] for r in window)
                        print('REMOTE',context,'x',used/capacity,'attention_ms',value,flush=True)
                        save(root/'progress.json',dict(context=context,steps=count,extra_fraction=used/capacity,
                             window_attention_median_ms=value,local_attention_median_ms=local_time))
                        if value>=local_time*cfg['stop_time_ratio'] and previous_below:
                            reason='crossing_bracket_observed';break
                        previous_below=previous_below or value<local_time
                        window=[]
            correctness.append(dict(context=context,layers=controller.check()))
            slots=raw.req_to_token_pool.req_to_token[batch.req_pool_indices,:context].reshape(-1)
            assert torch.unique(slots).numel()==used and slots.max()<allocation['slots']
            save(root/'correctness.json',dict(checkpoints=correctness,unique_active_slots=used,
                 all_attention_reads_from_peer=True,local_reference_retained=True,
                 no_peer_to_local_KV_staging_during_attention=True))
            save(root/'collection.json',dict(status='ATTENTION_MATCH_COLLECTION_COMPLETE',config=cfg,
                 remote_decode_steps=count,final_context=context,stop_reason=reason,
                 start_unix_ns=started,end_unix_ns=time.time_ns(),local_attention_median_ms=local_time,
                 scope='Independent full-local and peer-only native attention timings; not a concurrency or latency-hiding result. Donor GPU idle.'))
        torch.cuda.synchronize()
        cuda.close(address);connection.send(dict(type='closed'))
        message=receive(connection,stop);assert message['type']=='freed'

    graph=dict(prefill=dict(backend='disabled'),decode=dict(backend='full',max_bs=cfg['batch'],bs=[cfg['batch']]))
    parser=argparse.ArgumentParser();bench.ServerArgs.add_cli_args(parser);bench.BenchArgs.add_cli_args(parser)
    sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1];sock.close()
    argv=['--model-path',cfg['model_path'],'--tp-size','1','--dtype','bfloat16','--kv-cache-dtype','bf16',
          '--attention-backend','flashinfer','--disable-radix-cache','--page-size','1','--port',str(port),
          '--mem-fraction-static',str(cfg['mem_fraction_static']),'--max-total-tokens',str(cfg['kv_capacity_tokens']),
          '--max-running-requests',str(cfg['batch']),'--context-length',str(cfg['context_length_limit']),
          '--cuda-graph-config',json.dumps(graph),'--batch-size',str(cfg['batch']),'--input-len','1','--output-len','2',
          '--random-seed',str(cfg['random_seed'])]
    args=parser.parse_args(argv);bench.latency_test=campaign
    bench.main(bench.ServerArgs.from_cli_args(args),bench.BenchArgs.from_cli_args(args))


def worker(role,cfg,root,connection,stop):
    with (root/f'worker_{role}.log').open('w',buffering=1) as log:
        os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
        try:
            if role==1:
                consumer(cfg,root,connection,stop)
            else:
                import torch
                torch.cuda.set_device(0)
                cuda=Cuda();cuda.set_device(0)
                slots=cfg['batch']*cfg['remote_max_context']+1
                size=36*2*slots*8*128*2
                address,handle=cuda.allocate(size)
                tensor=mapped_tensor(address,slots,0)
                tensor.flatten()[0].fill_(1.5);torch.cuda.synchronize()
                attributes=cuda.attributes(address)
                assert attributes['memory_type']==2 and attributes['owner_logical_device']==0
                connection.send(dict(type='allocation',handle=handle,slots=slots,bytes=size,owner=attributes))
                message=receive(connection,stop);assert message['type']=='probe_written' and message['observed']==1.5
                observed=float(tensor.flatten()[0].item());assert observed==2.25
                save(root/'ipc_proof.json',dict(owner_logical_device=0,consumer_logical_device=1,
                     owner_written=1.5,consumer_written=2.25,owner_observed=observed,
                     donor_state='idle_memory_owner',allocation_bytes=size))
                connection.send(dict(type='verified'))
                message=receive(connection,stop);assert message['type']=='closed'
                cuda.free(address);connection.send(dict(type='freed'))
        except BaseException:
            save(root/f'worker_{role}_failure.json',dict(error=traceback.format_exc()));stop.set();raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();cfg=json.loads(args.config.read_text())
    assert cfg['batch']*cfg['local_full_context']==cfg['kv_capacity_tokens']
    assert cfg['remote_start_context']<cfg['remote_max_context']<=cfg['local_full_context']<=cfg['context_length_limit']
    args.out.mkdir(parents=True,exist_ok=False);save(args.out/'config.json',cfg)
    topology(args.out)
    ctx=mp.get_context('spawn');stop=ctx.Event();left,right=ctx.Pipe(duplex=True)
    workers=[ctx.Process(target=worker,args=(role,cfg,args.out,conn,stop)) for role,conn in enumerate((left,right))]
    for process in workers:process.start()
    left.close();right.close()
    for process in workers:process.join()
    assert all(p.exitcode==0 for p in workers),[p.exitcode for p in workers]
    print('ATTENTION_MATCH_COLLECTION_COMPLETE',flush=True)


if __name__=='__main__': main()
