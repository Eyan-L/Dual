"""Reuse the frozen real-decode initializer; time fixed-Q/KV candidates fairly."""

from pathlib import Path as _FinalPath
_SLACK_FINAL_ROOT = next(p for p in _FinalPath(__file__).resolve().parents if (p / 'release.json').is_file())
_SLACK_PROJECT_ROOT = _SLACK_FINAL_ROOT.parent
import json
from pathlib import Path
import random
import statistics
import sys
import time

BASE = Path(str(_SLACK_FINAL_ROOT / 'vendor/nvlink_cooperation/launch_20261030_r1/code'))
sys.path.insert(0, str(BASE))
import run_cooperation as base
from prototype import save


def fixture(control, variants, cfg, root, stem):
    import torch
    queries, expected = control.queries.clone(), control.outputs.clone()
    graphs, markers, checks, kept_outputs = {}, {}, [], {}
    stream = torch.cuda.Stream(device=control.raw.device)
    for variant in variants:
        name, call = variant['name'], variant['call']
        result = torch.empty_like(expected) if variant['check'] else None
        if result is not None:
            kept_outputs[name] = result
        torch.cuda.synchronize()
        with torch.cuda.stream(stream):
            for _ in range(3):
                for i in range(36):
                    call(queries[i], i)
        torch.cuda.synchronize()
        events = [(torch.cuda.Event(enable_timing=True, external=True),
                   torch.cuda.Event(enable_timing=True, external=True)) for _ in range(36)]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for i, (begin, end) in enumerate(events):
                begin.record()
                value = call(queries[i], i)
                end.record()
                if result is not None:
                    result[i].copy_(value)
        for _ in range(cfg['fixture_warmup_replays']):
            graph.replay()
        torch.cuda.synchronize()
        if result is not None:
            torch.testing.assert_close(result, expected, atol=cfg['reference_atol'], rtol=cfg['reference_rtol'])
            checks.append(dict(variant=name, max_abs_error=float((result-expected).abs().max().item())))
        graphs[name], markers[name] = graph, events
    values = {v['name']: [] for v in variants}
    orders = []
    rng = random.Random(cfg['random_seed'])
    with (root/(stem+'_steps.jsonl')).open('w', buffering=1) as out:
        for group in range(cfg['fixture_groups']):
            order = list(values)
            rng.shuffle(order)
            orders.append(order)
            for name in order:
                for replay in range(cfg['fixture_replays_per_group']):
                    graphs[name].replay()
                    markers[name][-1][1].synchronize()
                    times = [a.elapsed_time(b) for a, b in markers[name]]
                    envelope = markers[name][0][0].elapsed_time(markers[name][-1][1])
                    assert min(times) > 0 and sum(times) <= envelope*1.01
                    row = dict(variant=name, group=group, replay=replay, attention_layer_ms=times,
                               attention_sum_ms=sum(times), envelope_ms=envelope, completed_unix_ns=time.time_ns())
                    out.write(json.dumps(row, separators=(',', ':'))+'\n')
                    values[name].append(row['attention_sum_ms'])
            print(stem, 'GROUP', group+1, flush=True)
    summary = dict(status='FIXED_KV_FIXTURE_COMPLETE', context=control.context, checks=checks, group_orders=orders,
                   variants=[dict(name=name, samples=len(v), median_ms=statistics.median(v), minimum_ms=min(v), maximum_ms=max(v)) for name, v in values.items()],
                   scope='Immutable endpoint Q/KV; attention-only CUDA graphs; complete per-layer intervals including the candidate reduction/merge; profiler excluded.')
    save(root/(stem+'.json'), summary)
    return summary, graphs, markers, (queries, expected, kept_outputs)


def profile_variants(graphs, markers, names, root, stem):
    import torch
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
        for name in names:
            with torch.profiler.record_function('VARIANT::'+name):
                for _ in range(2):
                    graphs[name].replay()
                    markers[name][-1][1].synchronize()
    prof.export_chrome_trace(str(root/(stem+'.trace.json')))


def driver(experiment):
    original_consumer = base.consumer

    def consumer(cfg, root, connection, stop):
        from cooperative_attention import CooperativeAttention
        state = {}
        original_plan, original_stage, original_save = CooperativeAttention.plan, base.fixed_stage, base.save
        original_prepare = base.prepare

        def prepared(bench, wrapped, tokenizer, *args, **kwargs):
            state.update(bench=bench, wrapped=wrapped, tokenizer=tokenizer)
            return original_prepare(bench, wrapped, tokenizer, *args, **kwargs)

        def plan(control, batch):
            result = original_plan(control, batch)
            state['batch'] = batch
            return result

        def stage(raw, forward, control, *args, **kwargs):
            result = original_stage(raw, forward, control, *args, **kwargs)
            state.update(control=control, forward=forward)
            return result

        def saved(path, value):
            original_save(path, value)
            if path == root/'collection.json' and value.get('status') == 'COOPERATIVE_COLLECTION_COMPLETE':
                assert not state.get('experiment_started')
                state['experiment_started'] = True
                experiment(state['control'], state['forward'], state['batch'], cfg, root, state)

        CooperativeAttention.plan, base.fixed_stage, base.save = plan, stage, saved
        base.prepare = prepared
        return original_consumer(cfg, root, connection, stop)

    base.consumer = consumer
    return base
