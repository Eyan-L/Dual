"""Report KV/latency ratios without making the expected trend a pass condition."""
import argparse
import csv
import json
from pathlib import Path
import statistics

import numpy as np


def table(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def diagnostic(root):
    trace = json.loads((root/'concurrency.trace.json').read_text())
    kernels = [e for e in trace['traceEvents'] if e.get('cat') == 'kernel' and e.get('dur', 0) > 0]
    partial = [e for e in kernels if any(s in e['name'] for s in
               ('BatchDecodeWithPagedKVCacheKernel', 'BatchPrefillWithPagedKVCacheKernel', 'PersistentVariableLengthMergeStatesKernel'))]
    edges = sorted([(e['ts'], 1) for e in partial] + [(e['ts']+e['dur'], -1) for e in partial], key=lambda v: (v[0], v[1]))
    active = 0
    last = None
    durations = {}
    for t, change in edges:
        if last is not None and active:
            durations[active] = durations.get(active, 0.) + t-last
        active += change
        last = t
    union = sum(durations.values())
    value = dict(attention_stream_ids=sorted({e['args']['stream'] for e in partial}),
                 duration_ms_by_concurrent_kernel_count={str(k): v/1000 for k, v in durations.items()},
                 attention_kernel_union_ms=union/1000,
                 overlapping_kernel_fraction=sum(v for k, v in durations.items() if k >= 2)/union if union else None,
                 scope='Post-scan fixed-context profiler; selects actual partial and split/reduce kernel names, not entire streams.')
    (root/'concurrency_summary.json').write_text(json.dumps(value, indent=2)+'\n')
    return value


def analyze(root):
    collection = json.loads((root/'collection.json').read_text())
    assert collection['status'] == 'COOPERATIVE_COLLECTION_COMPLETE'
    cfg = collection['config']
    capacity = cfg['kv_capacity_tokens']
    batch = cfg['batch']
    remote = batch*cfg['remote_context_per_request']
    stages = {name: json.loads((root/(name+'.json')).read_text()) for name in
              ('native_full_local', 'hybrid_same_total_KV', 'hybrid_full_local_plus_remote', 'all_local_same_split')}
    for name, stage in stages.items():
        expected = cfg['baseline_samples'] if name == 'native_full_local' else cfg['fixed_samples']
        assert len(stage['samples']) == expected
        assert [r['sequence'] for r in stage['samples']] == list(range(expected))
        assert all(len(r['attention_layer_ms']) == 36 and min(r['attention_layer_ms']) > 0 and
                   abs(sum(r['attention_layer_ms'])-r['attention_sum_ms']) < 1e-6 and
                   r['attention_sum_ms'] <= r['forward_ms']*1.01 for r in stage['samples'])
    assert stages['native_full_local']['active_kv_tokens'] == capacity
    initial = stages['hybrid_same_total_KV']
    final = stages['hybrid_full_local_plus_remote']
    shadow = stages['all_local_same_split']
    assert initial['local_active_tokens'] == capacity-remote and initial['total_active_tokens'] == capacity
    assert final['local_active_tokens'] == capacity and final['remote_active_tokens'] == remote
    assert final['total_active_tokens'] == capacity+remote and not final['local_shadow_present']
    assert final['reused_reclaimed_slots'] == remote and not final['local_reference_retained']
    assert shadow['total_active_tokens'] == capacity+remote and shadow['allocated_after_primary']
    assert collection['shadow_control_start_unix_ns'] > collection['end_unix_ns']
    mapping = json.loads((root/'mapping.json').read_text())
    assert mapping['local_pool_allocated_bytes'] == (capacity+1)*147456
    assert mapping['remote_allocated_bytes'] == remote*147456
    assert mapping['zero_copy_tensor_pointer'] == mapping['imported_pointer']
    owner = json.loads((root/'ipc_owner.json').read_text())
    assert owner['bidirectional_probe_passed'] and not owner['model_loaded'] and owner['decode_steps'] == 0
    reclaim = json.loads((root/'capacity_reclamation.json').read_text())
    assert reclaim['reclaimed_slots'] == remote and len(reclaim['layer_payload_checks']) == 36
    assert all(r['exact_peer_payload_match'] for r in reclaim['layer_payload_checks'])
    correctness = json.loads((root/'correctness.json').read_text())
    assert correctness['no_full_local_mirror_in_primary'] and correctness['reclaimed_slots_all_reused']
    assert len(correctness['checkpoints']) == 2 and all(len(c['layers']) == 36 for c in correctness['checkpoints'])
    sensitivity = json.loads((root/'remote_sensitivity.json').read_text())
    assert sensitivity['remote_V_change_affects_attention'] > 1e-3 and sensitivity['restored_exactly']
    rows = [json.loads(line) for line in (root/'growth_steps.jsonl').read_text().splitlines()]
    assert len(rows) == cfg['remote_context_per_request'] == collection['growth_steps']
    assert [r['sequence'] for r in rows] == list(range(len(rows)))
    assert [r['context'] for r in rows] == list(range(cfg['local_full_context']+1, collection['final_context']+1))
    seed = json.loads((root/'prompt_ids.json').read_text())['growth_seed']
    assert [r['input_token_ids'] for r in rows] == [seed] + [r['output_token_ids'] for r in rows[:-1]]
    assert all(r['batch'] == batch and len(r['output_token_ids']) == batch and
               r['remote_active_tokens'] == remote and r['local_active_tokens'] == batch*r['context']-remote and
               r['total_active_tokens'] == batch*r['context'] and not r['local_shadow_present'] and
               len(r['attention_layer_ms']) == 36 and min(r['attention_layer_ms']) > 0 and
               abs(sum(r['attention_layer_ms'])-r['attention_sum_ms']) < 1e-6 and
               r['attention_sum_ms'] <= r['forward_ms']*1.01 for r in rows)
    baseline = stages['native_full_local']['attention_median_ms']
    bins = []
    for start in range(0, len(rows), 128):
        chunk = rows[start:start+128]
        times = [r['attention_sum_ms'] for r in chunk]
        bins.append(dict(first_context=chunk[0]['context'], last_context=chunk[-1]['context'], samples=len(chunk),
             total_kv_increment_fraction=statistics.mean(r['total_kv_increment_fraction'] for r in chunk),
             local_pool_fraction=statistics.mean(r['local_active_tokens']/capacity for r in chunk),
             attention_median_ms=statistics.median(times), attention_p05_ms=float(np.percentile(times, 5)),
             attention_p95_ms=float(np.percentile(times, 95)), forward_median_ms=statistics.median(r['forward_ms'] for r in chunk)))
    stage_rows = []
    for name, value in stages.items():
        times = [r['attention_sum_ms'] for r in value['samples']]
        stage_rows.append(dict(stage=name, samples=len(times), attention_median_ms=statistics.median(times),
                               attention_p05_ms=float(np.percentile(times, 5)), attention_p95_ms=float(np.percentile(times, 95)),
                               forward_median_ms=value['forward_median_ms'], ratio_to_native_full_local=statistics.median(times)/baseline))
    fraction = remote/capacity
    ratio = final['attention_median_ms']/baseline
    summary = dict(status='COOPERATIVE_MEASUREMENT_CHECKS_PASSED', config=cfg,
         local_capacity_GiB=capacity*147456/2**30, remote_capacity_GiB=remote*147456/2**30,
         remote_over_original_local_fraction=fraction, remote_over_enlarged_total_fraction=remote/(capacity+remote),
         native_full_local_attention_ms=baseline,
         hybrid_same_total_attention_ms=initial['attention_median_ms'],
         hybrid_full_local_plus_remote_attention_ms=final['attention_median_ms'],
         all_local_same_split_attention_ms=shadow['attention_median_ms'],
         measured_attention_increment_fraction=ratio-1,
         attention_increment_over_KV_increment=(ratio-1)/fraction,
         equal_KV_native_to_hybrid_increment_fraction=initial['attention_median_ms']/baseline-1,
         matched_split_remote_vs_local_increment_fraction=final['attention_median_ms']/shadow['attention_median_ms']-1,
         difference_from_linear_KV_scaling_ms=final['attention_median_ms']-baseline*(1+fraction),
         local_marker_full_forward_time_ratio=stages['native_full_local']['instrumentation_forward_time_ratio'],
         growth_steps=len(rows), correctness=correctness, sensitivity=sensitivity, concurrency=diagnostic(root),
         checks=dict(actual_local_slots_reclaimed_and_reused=True, complete_local_mirror_absent_in_primary=True,
                     remote_KV_capacity_enforced=True, real_autoregressive_growth=True, donor_did_not_decode=True),
         interpretation='The hypothesized attention-increase/KV-increase ratio is reported, not an acceptance threshold. Matched all-local split control holds algorithm and context fixed but uses extra local memory only after primary measurement. Ratios alone do not establish a pure compute cause; link saturation and interference are not measured.')
    table(root/'stages.csv', stage_rows)
    table(root/'growth_bins.csv', bins)
    table(root/'growth_steps.csv', [{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in r.items()} for r in rows])
    plot(root, bins, stage_rows, baseline, fraction)
    (root/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('config', 'correctness')}, indent=2))


def plot(root, bins, stages, baseline, fraction):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.4))
    ax = axes[0]
    x = [r['total_kv_increment_fraction']*100 for r in bins]
    ax.plot(x, [r['attention_median_ms']/baseline for r in bins], color='#2e7bd6', label='Hybrid: real KV growth')
    ax.fill_between(x, [r['attention_p05_ms']/baseline for r in bins], [r['attention_p95_ms']/baseline for r in bins], color='#2e7bd6', alpha=.15)
    ax.plot([0, fraction*100], [1, 1+fraction], color='black', ls='--', label='Reference: time scales with total KV')
    labels = ['Native local', 'Hybrid, same total KV', 'Hybrid, local full + remote', 'All local, same split']
    for i, stage in enumerate(stages):
        if i == 0:
            continue
        px = 0 if i == 1 else fraction*100
        ax.scatter([px], [stage['attention_median_ms']/baseline], marker=['o', 'o', 's', '^'][i], color=['black', '#169b91', '#2e7bd6', '#9164ab'][i], s=38, label=labels[i], zorder=5)
    ax.set(xlabel='Total KV increase over original full local pool (%)', ylabel='Attention time / native full-local attention')
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(axis='y', color='#deded8', lw=.6)
    ax = axes[1]
    colors = ['#92928b', '#169b91', '#2e7bd6', '#9164ab']
    ax.bar(range(4), [s['attention_median_ms'] for s in stages], color=colors, width=.65)
    for i, stage in enumerate(stages):
        ax.plot([i, i], [stage['attention_p05_ms'], stage['attention_p95_ms']], color='black', lw=1)
    ax.set_xticks(range(4), ['Native\nlocal full', 'Hybrid\nsame total KV', 'Hybrid\nlocal full + remote', 'All local\nsame split'], fontsize=9)
    ax.set(ylabel='Sum of 36 attention intervals (ms)')
    ax.grid(axis='y', color='#deded8', lw=.6)
    ax.set_axisbelow(True)
    fig.subplots_adjust(left=.07, right=.98, bottom=.26, top=.94, wspace=.26)
    fig.text(.5, .035, 'B = 4 · one NV4 donor · native FlashInfer partials + global softmax merge\nGrowth curve: 128-step bins (KV changes within each bin); stage bars: fixed-context samples · shadow allocated after primary', ha='center', fontsize=10, color='#66635d')
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(root/f'cooperative_KV_scaling.{ext}', dpi=240)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    analyze(parser.parse_args().root)
