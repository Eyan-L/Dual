"""One decode GPU, one idle KV owner; actual local capacity reclamation."""
import argparse
import csv
import importlib.metadata
from io import StringIO
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import socket
import statistics
import subprocess
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cuda_ipc import Cuda
from measure import mapped_tensor, timed_forward
from prefill import prepare
from prototype import make_forward, receive, save


def topology(root):
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,pci.bus_id,name', '--format=csv,noheader'], text=True)
    matrix = subprocess.check_output(['nvidia-smi', 'topo', '-m'], text=True)
    (root/'gpu_inventory.csv').write_text(inventory)
    (root/'topology.txt').write_text(matrix)
    devices = list(csv.reader(StringIO(inventory)))
    visible = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    assert len(visible) == 2
    selected = []
    for name in visible:
        rows = [r for r in devices if r[0].strip() == name or r[1].strip().startswith(name)]
        assert len(rows) == 1
        r = rows[0]
        selected.append(dict(index=int(r[0]), uuid=r[1].strip(), pci_bus_id=r[2].strip(), name=r[3].strip()))
    clean = re.sub(r'\x1b\[[0-9;]*m', '', matrix)
    header = next(line.split() for line in clean.splitlines() if line.split() and line.split()[0] == 'GPU0')
    row = next(line.split() for line in clean.splitlines() if line.split() and line.split()[0] == f"GPU{selected[1]['index']}" and not line.split()[1].startswith('GPU'))
    link = row[1 + header.index(f"GPU{selected[0]['index']}")]
    assert link == 'NV4' and all('A100-SXM4-40GB' in d['name'] for d in selected)
    result = dict(devices=selected, visible_devices=visible, allocated_pair_topology=link,
                  offpath_logical_device=1, donor_logical_device=0)
    save(root/'allocated_topology.json', result)
    return result


def fixed_stage(raw, forward, control, cfg, root, name, samples=None):
    for _ in range(cfg['warmup_steps']):
        timed_forward(raw, forward, control)
    rows = []
    for i in range(samples or cfg['fixed_samples']):
        result, row = timed_forward(raw, forward, control)
        rows.append(dict(sequence=i, **row))
    value = dict(stage=name, mode=control.mode, samples=rows,
                 attention_median_ms=statistics.median(r['attention_sum_ms'] for r in rows),
                 forward_median_ms=statistics.median(r['forward_ms'] for r in rows))
    save(root/(name+'.json'), value)
    print(name, 'ATTENTION_MS', value['attention_median_ms'], flush=True)
    return result, value


def consumer(cfg, root, connection, stop):
    import torch
    from sglang.benchmark import one_batch as bench
    torch.cuda.set_device(1)
    torch.manual_seed(cfg['random_seed'])

    def campaign(sa, pa, ba, gpu, rank):
        bench.publish(sa, role='scheduler', ranks=bench.SpawnRanks(world_rank=bench.spawn_world_rank(sa, tp_rank=0, pp_rank=0), gpu_id=1))
        bench.initialize_moe_config()
        bench.initialize_fp8_gemm_config()
        bench.initialize_fp4_gemm_config()
        bench.configure_logger(sa, prefix=' COOPERATIVE')
        wrapped, tokenizer = bench.load_model(sa, pa, 1, 0)
        raw = wrapped.torch_runner
        capacity, batch_size, remote_context = cfg['kv_capacity_tokens'], cfg['batch'], cfg['remote_context_per_request']
        remote_slots = batch_size * remote_context
        assert raw.max_total_num_tokens == capacity
        assert raw.dtype == raw.kv_cache_dtype == torch.bfloat16
        assert list(raw.decode_cuda_graph_runner.capture_bs) == [batch_size]
        allocation = receive(connection, stop)
        assert allocation['type'] == 'allocation' and allocation['slots'] == remote_slots
        cuda = Cuda()
        cuda.enable_peer(1, 0)
        address = cuda.open(allocation['handle'])
        attributes = cuda.attributes(address)
        assert attributes['memory_type'] == 2
        peer = mapped_tensor(address, remote_slots, 1)
        assert float(peer.flatten()[0].item()) == 1.5
        peer.flatten()[0].fill_(2.25)
        torch.cuda.synchronize()
        connection.send(dict(type='probe_written'))
        assert receive(connection, stop)['type'] == 'verified'
        pool_bytes = sum(x.numel()*x.element_size() for i in range(36) for x in raw.token_to_kv_pool.get_kv_buffer(i))
        assert pool_bytes == (capacity + 1) * 147456
        save(root/'mapping.json', dict(export=allocation, imported_attributes=attributes,
             imported_pointer=address, zero_copy_tensor_pointer=peer.data_ptr(),
             local_pool_allocated_bytes=pool_bytes, local_active_capacity_bytes=capacity*147456,
             remote_allocated_bytes=remote_slots*147456,
             versions={k: importlib.metadata.version(k) for k in ('torch', 'sglang', 'flashinfer-python')}))
        from cooperative_attention import CooperativeAttention
        with torch.no_grad():
            # Original native local-full baseline, with the same Event markers.
            batch, token, _ = prepare(bench, wrapped, tokenizer, batch_size, cfg['local_full_context']-1,
                                      cfg['preparation_chunk_tokens'], 2)
            forward = make_forward(bench, raw, batch, token)
            for _ in range(cfg['warmup_steps']):
                timed_forward(raw, forward)
            plain = [timed_forward(raw, forward)[1]['forward_ms'] for _ in range(30)]
            control = CooperativeAttention(raw, peer, cfg)
            control.recapture('local')
            _, baseline = fixed_stage(raw, forward, control, cfg, root, 'native_full_local', cfg['baseline_samples'])
            baseline.update(context=cfg['local_full_context'], active_kv_tokens=capacity,
                            native_output_checks=control.check(), uninstrumented_forward_ms=plain,
                            instrumentation_forward_time_ratio=baseline['forward_median_ms']/statistics.median(plain))
            save(root/'native_full_local.json', baseline)

            control.recapture('hybrid')
            assert all(indices.numel() >= capacity+remote_slots for indices in control.backend.cuda_graph_kv_indices)
            batch, token, prompt = prepare(bench, wrapped, tokenizer, batch_size, cfg['local_full_context']-1,
                                           cfg['preparation_chunk_tokens'], remote_context+2)
            initial_input = token.cpu().tolist()
            forward = make_forward(bench, raw, batch, token)
            assert raw.token_to_kv_pool_allocator.available_size() == 0
            save(root/'capacity_reclamation.json', control.offload_prefix(batch))
            initial_result, initial = fixed_stage(raw, forward, control, cfg, root, 'hybrid_same_total_KV')
            initial.update(context=cfg['local_full_context'], local_active_tokens=capacity-remote_slots,
                           remote_active_tokens=remote_slots, total_active_tokens=capacity)
            save(root/'hybrid_same_total_KV.json', initial)
            checks = [control.check_full()]
            save(root/'remote_sensitivity.json', control.sensitivity())
            token = raw.sample(initial_result.logits_output, forward)
            seed = token.cpu().tolist()
            save(root/'prompt_ids.json', dict(per_request_prompt=prompt, initial_decode_input=initial_input,
                                              growth_seed=seed))
            assert control.shadow is None
            start = time.time_ns()
            with (root/'growth_steps.jsonl').open('w', buffering=1) as output:
                for step in range(remote_context):
                    consumed = token.cpu().tolist()
                    forward = make_forward(bench, raw, batch, token)
                    context = cfg['local_full_context']+step+1
                    assert batch.seq_lens_cpu.tolist() == [context]*batch_size
                    used = capacity-raw.token_to_kv_pool_allocator.available_size()
                    assert used == batch_size*(context-remote_context)
                    control.plan(batch)
                    result, row = timed_forward(raw, forward, control)
                    token = raw.sample(result.logits_output, forward)
                    assert torch.isfinite(result.logits_output.next_token_logits).all().item()
                    row.update(sequence=step, context=context, batch=batch_size, local_active_tokens=used,
                               remote_active_tokens=remote_slots, total_active_tokens=used+remote_slots,
                               total_kv_increment_fraction=(used+remote_slots)/capacity-1,
                               input_token_ids=consumed, output_token_ids=token.cpu().tolist(),
                               completed_unix_ns=time.time_ns(), local_shadow_present=False)
                    output.write(json.dumps(row, separators=(',', ':'))+'\n')
                    if (step+1) % 128 == 0:
                        print('GROWTH', step+1, 'CONTEXT', context, 'LOCAL_USED', used,
                              'ATTENTION_MS', row['attention_sum_ms'], flush=True)
            end = time.time_ns()
            assert raw.token_to_kv_pool_allocator.available_size() == 0
            assert control.local_indices.numel() == capacity
            assert torch.unique(control.local_indices).numel() == capacity
            assert control.local_indices.min().item() > 0 and control.local_indices.max().item() <= capacity
            reused = torch.isin(control.freed_slots, control.local_indices)
            assert reused.all().item()
            final_result, final = fixed_stage(raw, forward, control, cfg, root, 'hybrid_full_local_plus_remote')
            final.update(context=context, local_active_tokens=capacity, remote_active_tokens=remote_slots,
                         total_active_tokens=capacity+remote_slots, local_reference_retained=False,
                         local_shadow_present=False, reused_reclaimed_slots=int(reused.sum().item()))
            save(root/'hybrid_full_local_plus_remote.json', final)
            checks.append(control.check_full())
            final_logits = final_result.logits_output.next_token_logits.detach().clone()
            save(root/'correctness.json', dict(checkpoints=checks, unique_final_local_slots=capacity,
                                               reclaimed_slots_all_reused=True, no_full_local_mirror_in_primary=True))
            # A short post-scan diagnostic only; excluded from primary samples.
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
                for _ in range(cfg['diagnostic_steps']):
                    timed_forward(raw, forward, control)
            profile.export_chrome_trace(str(root/'concurrency.trace.json'))
            # Matched algorithm/Q/KV control: replace only the remote prefix
            # allocation by an extra local copy after primary measurement ends.
            shadow_start = time.time_ns()
            control.create_shadow()
            control.recapture('shadow')
            control.plan(batch)
            shadow_result, shadow = fixed_stage(raw, forward, control, cfg, root, 'all_local_same_split')
            torch.testing.assert_close(shadow_result.logits_output.next_token_logits, final_logits,
                                       atol=cfg['reference_atol'], rtol=cfg['reference_rtol'])
            shadow.update(context=context, total_active_tokens=capacity+remote_slots,
                          extra_local_prefix_bytes=remote_slots*147456, allocated_after_primary=True,
                          max_abs_logit_error=float((shadow_result.logits_output.next_token_logits-final_logits).abs().max().item()))
            save(root/'all_local_same_split.json', shadow)
            save(root/'collection.json', dict(status='COOPERATIVE_COLLECTION_COMPLETE', config=cfg,
                 growth_steps=remote_context, start_unix_ns=start, end_unix_ns=end,
                 shadow_control_start_unix_ns=shadow_start, final_context=context,
                 offpath_only_model=True, donor_decode_steps=0,
                 primary_local_pool_capacity_tokens=capacity, primary_remote_capacity_tokens=remote_slots))
        torch.cuda.synchronize()
        cuda.close(address)
        connection.send(dict(type='closed'))
        assert receive(connection, stop)['type'] == 'freed'

    graph = dict(prefill=dict(backend='disabled'), decode=dict(backend='full', max_bs=cfg['batch'], bs=[cfg['batch']]))
    parser = argparse.ArgumentParser()
    bench.ServerArgs.add_cli_args(parser)
    bench.BenchArgs.add_cli_args(parser)
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    sock.close()
    argv = ['--model-path', cfg['model_path'], '--tp-size', '1', '--dtype', 'bfloat16', '--kv-cache-dtype', 'bf16',
            '--attention-backend', 'flashinfer', '--disable-radix-cache', '--page-size', '1', '--port', str(port),
            '--mem-fraction-static', str(cfg['mem_fraction_static']), '--max-total-tokens', str(cfg['kv_capacity_tokens']),
            '--max-running-requests', str(cfg['batch']), '--context-length', str(cfg['context_length_limit']),
            '--cuda-graph-config', json.dumps(graph), '--batch-size', str(cfg['batch']), '--input-len', '1',
            '--output-len', '2', '--random-seed', str(cfg['random_seed'])]
    args = parser.parse_args(argv)
    bench.latency_test = campaign
    bench.main(bench.ServerArgs.from_cli_args(args), bench.BenchArgs.from_cli_args(args))


def worker(role, cfg, root, connection, stop):
    with (root/f'worker_{role}.log').open('w', buffering=1) as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        try:
            if role == 1:
                return consumer(cfg, root, connection, stop)
            import torch
            torch.cuda.set_device(0)
            cuda = Cuda()
            cuda.set_device(0)
            slots = cfg['batch']*cfg['remote_context_per_request']
            address, handle = cuda.allocate(slots*147456)
            peer = mapped_tensor(address, slots, 0)
            peer.zero_()
            peer.flatten()[0].fill_(1.5)
            torch.cuda.synchronize()
            attributes = cuda.attributes(address)
            assert attributes['memory_type'] == 2 and attributes['owner_logical_device'] == 0
            connection.send(dict(type='allocation', slots=slots, bytes=slots*147456, handle=handle, owner=attributes))
            assert receive(connection, stop)['type'] == 'probe_written'
            assert float(peer.flatten()[0].item()) == 2.25
            save(root/'ipc_owner.json', dict(owner_device=0, consumer_device=1, bidirectional_probe_passed=True,
                                            allocation_slots=slots, allocation_bytes=slots*147456,
                                            model_loaded=False, decode_steps=0))
            connection.send(dict(type='verified'))
            assert receive(connection, stop)['type'] == 'closed'
            cuda.free(address)
            connection.send(dict(type='freed'))
        except BaseException:
            save(root/f'worker_{role}_failure.json', dict(error=traceback.format_exc()))
            stop.set()
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    assert cfg['batch']*cfg['local_full_context'] == cfg['kv_capacity_tokens']
    assert cfg['local_full_context']+cfg['remote_context_per_request'] < cfg['context_length_limit']
    args.out.mkdir(parents=True, exist_ok=False)
    save(args.out/'config.json', cfg)
    topology(args.out)
    ctx = mp.get_context('spawn')
    stop = ctx.Event()
    left, right = ctx.Pipe(duplex=True)
    workers = [ctx.Process(target=worker, args=(role, cfg, args.out, connection, stop)) for role, connection in enumerate((left, right))]
    for process in workers:
        process.start()
    left.close()
    right.close()
    for process in workers:
        process.join()
    assert all(process.exitcode == 0 for process in workers), [p.exitcode for p in workers]
    assert json.loads((args.out/'collection.json').read_text())['status'] == 'COOPERATIVE_COLLECTION_COMPLETE'
    print('COOPERATIVE_COLLECTION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
