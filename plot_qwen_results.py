#!/usr/bin/env python3
"""Plot benchmark_qwen_zeus.py output. Requires: pip install matplotlib numpy

Examples:
  python plot_qwen_results.py machine1_results_v2
  python plot_qwen_results.py machine1_results_v2 machine2_results --labels Node1 Node2
  python plot_qwen_results.py machine1_results_v2 --gpu 0 --output plots_gpu0

Defaults to gpu=all rows (combined local GPUs); never sums those with individual
GPU rows. Separate folders are separate experiments, never summed together.
Points are repeat means; error bars are sample standard deviations, not confidence
intervals. Power domains are plotted separately, never added together. Timeline
curves retain actual timestamps and are not averaged across independent repeats.
Existing plots with matching filenames in --output are overwritten.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import sys

METRICS = {
    'average_power_w': ('Average GPU power', 'Power (W)'),
    'energy_j': ('GPU energy per measured run', 'Energy (J)'),
    'output_tokens_per_s': ('Output throughput', 'Output tokens/s'),
    'joules_per_output_token': ('Energy per output token', 'J/output token'),
    'requests_per_s': ('Request throughput', 'Requests/s'),
    'joules_per_request': ('Energy per request', 'J/request'),
}


def read_csv(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def load_results(folder, gpu):
    rows = read_csv(folder / 'results.csv')
    selected = [r for r in rows if r.get('gpu', '').strip().lower() == gpu]
    if not selected:
        raise ValueError(f'{folder}: no results for gpu={gpu}; available: '
                         f'{sorted({r.get("gpu", "") for r in rows})}')
    seen = set()
    for row in selected:
        for name in ['batch_per_gpu', 'repeat', 'elapsed_s', *METRICS]:
            row[name] = float(row[name])
            if not math.isfinite(row[name]) or row[name] < 0:
                raise ValueError(f'{folder}: invalid {name}: {row[name]}')
        if row['batch_per_gpu'] <= 0 or row['elapsed_s'] <= 0:
            raise ValueError(f'{folder}: batch and elapsed time must be positive')
        key = (row['batch_per_gpu'], row['repeat'])
        if key in seen:
            raise ValueError(f'{folder}: duplicate batch/repeat {key} for gpu={gpu}')
        seen.add(key)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('folders', nargs='+', type=Path, help='Benchmark output directories')
    parser.add_argument('--labels', nargs='+', help='One legend label per folder')
    parser.add_argument('--gpu', default='all', help='Summary GPU selector: all (default), 0, 1, ...')
    parser.add_argument('--output', type=Path, default=Path('qwen_plots'))
    parser.add_argument('--pdf', action='store_true', help='Also save vector PDF figures')
    parser.add_argument('--skip-power', action='store_true')
    args = parser.parse_args()
    labels = args.labels or [p.name for p in args.folders]
    if len(labels) != len(args.folders) or len(set(labels)) != len(labels):
        parser.error('Provide one distinct label per folder using --labels.')
    import matplotlib
    matplotlib.use('Agg')  # Runs on a remote server without a display.
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.dpi': 120, 'savefig.dpi': 180,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'font.size': 10})
    datasets = [(label, folder, load_results(folder, args.gpu.lower()))
                for label, folder in zip(labels, args.folders)]
    # Warn when key workloads differ; differing GPU counts can be intentional.
    configs = []
    for _, folder, _ in datasets:
        path = folder / 'metadata.json'
        configs.append(json.loads(path.read_text()) if path.exists() else {})
    for key in ['model', 'revision', 'input_tokens', 'output_tokens', 'seconds']:
        values = [repr(c[key]) for c in configs if key in c]
        if len(set(values)) > 1:
            print(f'Warning: {key} differs between experiments: {values}', file=sys.stderr)
    args.output.mkdir(parents=True, exist_ok=True)
    def save(fig, stem):
        fig.savefig(args.output / f'{stem}.png', bbox_inches='tight')
        if args.pdf:
            fig.savefig(args.output / f'{stem}.pdf', bbox_inches='tight')
        plt.close(fig)
    batches = sorted({r['batch_per_gpu'] for _, _, rows in datasets for r in rows})
    positions = {b: i for i, b in enumerate(batches)}
    summary = []
    for label, _, rows in datasets:
        for b in sorted({r['batch_per_gpu'] for r in rows}):
            group = [r for r in rows if r['batch_per_gpu'] == b]
            item = {'experiment': label, 'gpu': args.gpu, 'batch_per_gpu': int(b), 'repeats': len(group)}
            for metric in [*METRICS, 'elapsed_s']:
                values = [r[metric] for r in group]
                item[metric + '_mean'] = statistics.mean(values)
                item[metric + '_std'] = statistics.stdev(values) if len(values) > 1 else 0.0
            summary.append(item)
    with (args.output / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    def draw_metric(ax, metric):
        title, ylabel = METRICS[metric]
        for i, label in enumerate(labels):
            data = [r for r in summary if r['experiment'] == label]
            offset = (i - (len(labels)-1)/2) * min(0.08, 0.5/max(1, len(labels)))
            x = [positions[r['batch_per_gpu']] + offset for r in data]
            ax.errorbar(x, [r[metric+'_mean'] for r in data],
                        yerr=[r[metric+'_std'] for r in data], marker='o',
                        capsize=4, linewidth=1.6, label=label)
        ax.set(title=title, xlabel='Batch size per GPU', ylabel=ylabel)
        ax.set_xticks(range(len(batches)), [str(int(b)) for b in batches])
        ax.set_ylim(bottom=0)
        ax.grid(axis='y', alpha=0.25)
        ax.legend(fontsize=8)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for ax, metric in zip(axes.flat, METRICS):
        draw_metric(ax, metric)
    fig.suptitle(f'Qwen benchmark · GPU {args.gpu}\nMean ± sample SD across repeats', fontsize=15)
    save(fig, 'overview')
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        draw_metric(ax, metric)
        ax.set_title(METRICS[metric][0] + f' · GPU {args.gpu}\nMean ± sample SD')
        save(fig, metric)
    if not args.skip_power:
        for experiment_index, (label, folder, _) in enumerate(datasets, 1):
            path = folder / 'power.csv'
            if not path.exists() or not path.stat().st_size:
                print(f'Skipping absent/empty {path}')
                continue
            rows = read_csv(path)
            if args.gpu.lower() != 'all':
                rows = [r for r in rows if r['gpu'] == args.gpu]
            groups = {}
            for row in rows:
                key = (int(float(row['batch_per_gpu'])), row['domain'])
                t, w = float(row['seconds_from_start']), float(row['power_w'])
                if math.isfinite(t) and math.isfinite(w) and t >= 0 and w >= 0:
                    groups.setdefault(key, {}).setdefault((row['repeat'], row['gpu']), []).append((t, w))
            if not groups:
                print(f'No power samples in {path}; summary plots still generated.')
            for group_index, ((batch, domain), traces) in enumerate(sorted(groups.items()), 1):
                repeats = sorted({r for r, _ in traces}, key=float)
                fig, axes = plt.subplots(len(repeats), 1, figsize=(10, 3*len(repeats)),
                                         squeeze=False, sharex=True, constrained_layout=True)
                for ax, repeat in zip(axes.flat, repeats):
                    for (trace_repeat, gpu), samples in sorted(traces.items()):
                        if trace_repeat != repeat:
                            continue
                        points = sorted(samples)
                        t, w = zip(*points)
                        ax.step(t, w, where='post', linewidth=1, label=f'GPU {gpu}')
                    ax.set(title=f'Repeat {repeat}', ylabel='Power (W)')
                    ax.set_ylim(bottom=0)
                    ax.grid(alpha=0.25)
                    ax.legend()
                axes[-1, 0].set_xlabel('Seconds from measurement start')
                fig.suptitle(f'{label} · Batch {batch} per GPU · {domain}\nSampled power; no smoothing')
                save(fig, f'power_experiment{experiment_index}_batch{batch}_domain{group_index}')
    print(f'Plots and summary.csv saved to {args.output.resolve()}')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f'Error: {exc}') from exc
