#!/usr/bin/env python3
"""Plot original and TP Qwen/Zeus benchmark outputs. Requires matplotlib.

Examples:
  python plot_qwen_results_tp.py tp1_results tp2_results --labels TP1 TP2
  python plot_qwen_results_tp.py tp1_results tp2_results --x-axis total
  python plot_qwen_results_tp.py tp2_results --gpu 0
  python plot_qwen_results_tp.py tp2_results --replica 0

Default: aggregate 'all' rows only. --gpu selects device energy/power; --replica
selects a model replica (new schema only). TP GPU throughput fields are blank
by design and skipped, never treated as zero. Old batch_per_gpu is interpreted
as batch_per_replica because the old benchmark uses one replica per GPU.

--x-axis total uses total concurrent batch across the entire experiment, even
when selecting one GPU/replica. For old files it requires GPU count from
metadata.json or the individual GPU rows. Points are repeat means; error bars
are sample SD, not confidence intervals. Power traces retain individual GPUs,
domains and repeats; no averaging, summing, smoothing or resampling. Output
files with matching names are overwritten. Use a fresh output directory when
changing selections to avoid confusing leftover plots with new output.
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


def number(value, name, positive=False, integer=False):
    v = float(value)
    if not math.isfinite(v) or v < 0 or (positive and v <= 0) or (integer and v != int(v)):
        raise ValueError(f'Invalid {name}: {value!r}')
    return int(v) if integer else v


def batch_value(row):
    return number(row.get('batch_per_replica', row.get('batch_per_gpu')), 'batch', True, True)


def load_results(folder, gpu='all', replica=None):
    raw = read_csv(folder / 'results.csv')
    if not raw:
        raise ValueError(f'{folder}: empty results.csv')
    path = folder / 'metadata.json'
    config = json.loads(path.read_text()) if path.exists() else {}
    new = 'scope' in raw[0]
    if replica is not None and not new:
        raise ValueError(f'{folder}: --replica requires the new TP CSV format; use --gpu for old files.')
    selected = [r for r in raw if (
        (r.get('scope') == 'replica' and str(r.get('replica')) == str(replica))
        if replica is not None else
        (r.get('gpu', '').strip().lower() == gpu and
         (not new or r.get('scope') == ('all' if gpu == 'all' else 'gpu'))))]
    if not selected:
        raise ValueError(f'{folder}: no rows for selected GPU/replica')
    inferred_gpu_count = len({r['gpu'] for r in raw if r.get('gpu', '').isdigit()})
    replicas = len(config.get('gpus', [])) or inferred_gpu_count or None
    seen = set()
    rows = []
    for source in selected:
        b = batch_value(source)
        repeat = number(source['repeat'], 'repeat', True, True)
        if (b, repeat) in seen:
            raise ValueError(f'{folder}: duplicate batch/repeat {(b, repeat)}')
        seen.add((b, repeat))
        tp = number(source.get('tp_size', config.get('tp_size', 1)), 'tp_size', True, True)
        nr = source.get('replicas') or (replicas if not new else None)
        nr = number(nr, 'replicas', True, True) if nr is not None else None
        total = source.get('total_batch') or (b * nr if nr else None)
        total = number(total, 'total_batch', True, True) if total is not None else None
        if total is not None and nr is not None and total != b * nr:
            raise ValueError(f'{folder}: inconsistent total_batch')
        row = dict(batch_per_replica=b, total_batch=total, repeat=repeat,
                   tp_size=tp, replicas=nr,
                   elapsed_s=number(source['elapsed_s'], 'elapsed_s', True))
        for metric in METRICS:
            value = source.get(metric)
            missing = value is None or not str(value).strip()
            allowed_missing = new and source.get('scope') == 'gpu' and tp > 1 and metric not in ('energy_j', 'average_power_w')
            if missing and not allowed_missing:
                raise ValueError(f'{folder}: missing {metric}')
            row[metric] = None if missing else number(value, metric)
        rows.append(row)
    if len({(r['tp_size'], r['replicas']) for r in rows}) != 1:
        raise ValueError(f'{folder}: multiple TP/replica configurations; use separate folders.')
    # Scope is explicit for new files; metadata ties TP replica IDs to devices.
    timeline_gpus = None
    if replica is not None:
        timeline_gpus = {str(r['gpu']) for r in raw if r.get('scope') == 'gpu' and str(r.get('replica')) == str(replica)}
        if not timeline_gpus:
            timeline_gpus = {str(d['gpu']) for d in config.get('devices', []) if str(d.get('replica')) == str(replica)}
        if not timeline_gpus:
            raise ValueError(f'{folder}: cannot determine GPUs belonging to replica {replica}')
    elif gpu != 'all':
        timeline_gpus = {gpu}
    return rows, config, timeline_gpus


def summarize(label, rows):
    result = []
    for batch in sorted({r['batch_per_replica'] for r in rows}):
        group = [r for r in rows if r['batch_per_replica'] == batch]
        if len({r['total_batch'] for r in group}) != 1:
            raise ValueError('Total batch differs across repeats')
        item = dict(experiment=label, tp_size=group[0]['tp_size'], replicas=group[0]['replicas'],
                    batch_per_replica=batch, total_batch=group[0]['total_batch'], repeats=len(group))
        for metric in [*METRICS, 'elapsed_s']:
            values = [r[metric] for r in group if r[metric] is not None]
            item[metric + '_n'] = len(values)
            item[metric + '_mean'] = statistics.mean(values) if values else None
            item[metric + '_std'] = statistics.stdev(values) if len(values) > 1 else (0.0 if values else None)
        result.append(item)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('folders', nargs='+', type=Path)
    p.add_argument('--labels', nargs='+')
    selection = p.add_mutually_exclusive_group()
    selection.add_argument('--gpu', default='all', help='all (default), or a logical GPU index')
    selection.add_argument('--replica', type=int, help='Model replica index in the new schema')
    p.add_argument('--x-axis', choices=['replica', 'total'], default='replica')
    p.add_argument('--output', type=Path, default=Path('qwen_plots'))
    p.add_argument('--pdf', action='store_true')
    p.add_argument('--skip-power', action='store_true')
    args = p.parse_args()
    args.gpu = args.gpu.strip().lower()
    if args.gpu != 'all':
        if not args.gpu.isdigit():
            p.error('--gpu must be all or a nonnegative integer')
        args.gpu = str(int(args.gpu))
    if args.replica is not None and args.replica < 0:
        p.error('--replica must be nonnegative')
    labels = args.labels or [f.name for f in args.folders]
    if len(labels) != len(args.folders) or len(set(labels)) != len(labels):
        p.error('Provide one distinct label per folder with --labels.')
    datasets = [(label, folder, *load_results(folder, args.gpu, args.replica))
                for label, folder in zip(labels, args.folders)]
    for key in ['model', 'revision', 'input_tokens', 'output_tokens', 'seconds']:
        values = [repr(config[key]) for _, _, _, config, _ in datasets if key in config]
        if len(set(values)) > 1:
            print(f'Warning: {key} differs: {values}', file=sys.stderr)
    summary = [s for label, _, rows, _, _ in datasets for s in summarize(label, rows)]
    xkey = 'total_batch' if args.x_axis == 'total' else 'batch_per_replica'
    xlabel = 'Total batch size across all replicas' if args.x_axis == 'total' else 'Batch size per model replica'
    if any(r[xkey] is None for r in summary):
        raise ValueError('Cannot infer total batch; supply original metadata.json or use --x-axis replica.')
    if args.x_axis == 'replica' and len({r['replicas'] for r in summary}) > 1:
        print('Note: replica counts differ; equal per-replica batches have different total concurrency. Use --x-axis total to compare total batch.', file=sys.stderr)
    selection_label = f'Replica {args.replica}' if args.replica is not None else ('All GPUs combined' if args.gpu == 'all' else f'GPU {args.gpu}')
    for r in summary:
        r['selection'] = selection_label
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.dpi': 120, 'savefig.dpi': 180,
                         'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10})
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    def save(fig, stem):
        fig.savefig(args.output / f'{stem}.png', bbox_inches='tight')
        if args.pdf:
            fig.savefig(args.output / f'{stem}.pdf', bbox_inches='tight')
        plt.close(fig)
    batches = sorted({r[xkey] for r in summary})
    positions = {b: i for i, b in enumerate(batches)}
    def draw_metric(ax, metric):
        plotted = False
        for i, label in enumerate(labels):
            data = sorted([r for r in summary if r['experiment'] == label and r[metric+'_mean'] is not None], key=lambda r:r[xkey])
            if not data:
                continue
            plotted = True
            offset = (i-(len(labels)-1)/2) * min(0.08, 0.5/len(labels))
            legend = f"{label} (TP={data[0]['tp_size']}, replicas={data[0]['replicas'] or '?'})"
            ax.errorbar([positions[r[xkey]]+offset for r in data], [r[metric+'_mean'] for r in data],
                        yerr=[r[metric+'_std'] for r in data], color=f'C{i % 10}', marker='o',
                        capsize=4, linewidth=1.6, label=legend)
        ax.set(title=METRICS[metric][0], xlabel=xlabel, ylabel=METRICS[metric][1])
        ax.set_xticks(range(len(batches)), [str(b) for b in batches])
        ax.set_ylim(bottom=0)
        ax.grid(axis='y', alpha=0.25)
        if plotted:
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, 'Not reported for individual TP GPUs\nSelect --gpu all or --replica N',
                    ha='center', va='center', transform=ax.transAxes)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for ax, metric in zip(axes.flat, METRICS):
        draw_metric(ax, metric)
    fig.suptitle(f'Qwen benchmark · {selection_label}\nMean ± sample SD across repeats', fontsize=15)
    save(fig, 'overview')
    for metric in METRICS:
        if not any(r[metric+'_mean'] is not None for r in summary):
            print(f'Skipping unavailable metric: {metric}')
            continue
        fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
        draw_metric(ax, metric)
        ax.set_title(METRICS[metric][0] + f' · {selection_label}\nMean ± sample SD')
        save(fig, metric)
    if not args.skip_power:
        for ei, (label, folder, result_rows, _, timeline_gpus) in enumerate(datasets, 1):
            path = folder / 'power.csv'
            if not path.exists() or not path.stat().st_size:
                print(f'Skipping absent/empty {path}')
                continue
            groups = {}
            for row in read_csv(path):
                if timeline_gpus is not None and row['gpu'] not in timeline_gpus:
                    continue
                b = batch_value(row)
                repeat = number(row['repeat'], 'repeat', True, True)
                t, w = float(row['seconds_from_start']), float(row['power_w'])
                if math.isfinite(t) and math.isfinite(w) and t >= 0 and w >= 0:
                    groups.setdefault((b, row['domain']), {}).setdefault((repeat, row['gpu']), []).append((t,w))
            if not groups:
                print(f'No selected power samples in {path}; summary plots still generated.')
            for gi, ((batch, domain), traces) in enumerate(sorted(groups.items()), 1):
                repeats = sorted({r for r, _ in traces})
                fig, axes = plt.subplots(len(repeats), 1, figsize=(10, 3*len(repeats)),
                                         squeeze=False, sharex=True, constrained_layout=True)
                gpu_ids = sorted({gpu for _, gpu in traces})
                colors = {gpu:f'C{i % 10}' for i,gpu in enumerate(gpu_ids)}
                for ax, repeat in zip(axes.flat, repeats):
                    for (tr, gpu), samples in sorted(traces.items()):
                        if tr != repeat:
                            continue
                        t,w = zip(*sorted(samples))
                        ax.step(t,w,where='post',linewidth=1,color=colors[gpu],label=f'GPU {gpu}')
                    ax.set(title=f'Repeat {repeat}', ylabel='Power (W)')
                    ax.set_ylim(bottom=0)
                    ax.grid(alpha=0.25)
                    ax.legend()
                axes[-1,0].set_xlabel('Seconds from measurement start')
                fig.suptitle(f"{label} · TP={result_rows[0]['tp_size']} · Batch {batch} per replica · {domain}\nSampled power; no smoothing")
                save(fig, f'power_experiment{ei}_batch{batch}_domain{gi}')
    print(f'Plots and summary.csv saved to {args.output.resolve()}')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(f'Error: {exc}') from exc
