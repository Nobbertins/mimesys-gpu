#!/usr/bin/env python3
"""Qwen BF16 inference energy benchmark with independent TP replica groups.

Setup with an existing CUDA-enabled PyTorch (>=2.6):
    python -m pip install 'transformers==4.57.1' accelerate zeus
Run with python, NOT torchrun (this script launches its own workers):
    python benchmark_qwen_zeus_tp.py --gpus 0 1 --tp-size 1
    python benchmark_qwen_zeus_tp.py --gpus 0 1 --tp-size 2
    python benchmark_qwen_zeus_tp.py --gpus 0 1 2 3 --tp-size 2

Batch sizes are PER REPLICA (per GPU only when TP=1). Consecutive entries in
--gpus form groups. CUDA indices respect CUDA_VISIBLE_DEVICES. TP requires
NCCL and a model with a native Transformers TP plan. This script targets the
pinned Transformers API above. Use a model revision hash for reproducibility.

Loading, tokenization and warmup are excluded. Measurements include prefill,
decode, synchronization and control overhead. Energy is whole-device energy,
including idle energy, not whole-server or process-only energy. Keep GPUs free
of other jobs. Synthetic prompts and forced lengths do not test output quality.
Per-GPU TP rows intentionally have blank request/token metrics: they describe
energy only. Replica and 'all' rows count each generated token exactly once.
"""
import argparse
import csv
import importlib.metadata
import inspect
import json
import math
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import socket
import tempfile
import time
import traceback
from datetime import datetime, timezone, timedelta


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='Qwen/Qwen3-4B')
    p.add_argument('--revision', default='main')
    p.add_argument('--gpus', type=int, nargs='+', default=[0, 1])
    p.add_argument('--tp-size', type=int, default=1)
    p.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 4, 8, 16], help='Batch per model replica.')
    p.add_argument('--input-tokens', type=int, default=512)
    p.add_argument('--output-tokens', type=int, default=128)
    p.add_argument('--seconds', type=float, default=60)
    p.add_argument('--warmup-seconds', type=float, default=20)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--worker-timeout', type=float, default=1800, help='Maximum seconds per load/warmup/run phase.')
    p.add_argument('--no-power-timeline', action='store_true')
    p.add_argument('--output', default='qwen_results_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    a = p.parse_args(argv)
    if len(set(a.gpus)) != len(a.gpus) or min(a.gpus) < 0:
        p.error('GPU indices must be distinct and nonnegative.')
    if a.tp_size < 1 or len(a.gpus) % a.tp_size:
        p.error('--tp-size must be positive and divide the number of selected GPUs.')
    if min(a.batch_sizes + [a.input_tokens, a.output_tokens, a.repeats]) <= 0:
        p.error('Batch sizes, token lengths and repeats must be positive.')
    if any(not math.isfinite(v) or v <= 0 for v in (a.seconds, a.warmup_seconds, a.worker_timeout)):
        p.error('Durations must be finite and positive.')
    return a


def worker(gpu, replica, rank, rendezvous, args, connection):
    dist = None
    try:
        import torch
        import torch.distributed as dist
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(1)
        torch.cuda.set_device(gpu)
        device = torch.device(f'cuda:{gpu}')
        tp = args.tp_size > 1
        # Identical seeds and input tokens within each cooperating group.
        torch.manual_seed(1234 + replica)
        if tp:
            os.environ.update(RANK=str(rank), WORLD_SIZE=str(args.tp_size), LOCAL_RANK=str(gpu))
            dist.init_process_group('nccl', init_method=rendezvous, rank=rank,
                                    world_size=args.tp_size, device_id=device,
                                    timeout=timedelta(seconds=args.worker_timeout))
        config = AutoConfig.from_pretrained(args.model, revision=args.revision)
        if tp and not getattr(config, 'base_model_tp_plan', None):
            raise ValueError(f'{args.model} has no native tensor-parallel plan.')
        if tp:
            # Conservative divisibility checks for dense Qwen-style attention.
            for key in ('num_attention_heads', 'num_key_value_heads', 'intermediate_size'):
                size = getattr(config, key, None)
                if size is not None and size % args.tp_size:
                    raise ValueError(f'{key}={size} must be divisible by TP={args.tp_size}.')
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
        placement = {'tp_plan': 'auto'} if tp else {'device_map': {'': str(device)}}
        model = AutoModelForCausalLM.from_pretrained(
            args.model, revision=args.revision, dtype=torch.bfloat16,
            attn_implementation='sdpa', **placement).eval()
        if tp and not getattr(model, '_tp_plan', None):
            raise RuntimeError('Model did not activate a tensor-parallel plan.')
        prompt = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': 'Explain how computers process information. ' * args.input_tokens}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        tokens = (tokens * (args.input_tokens // len(tokens) + 1))[-args.input_tokens:]
        base = torch.tensor(tokens, dtype=torch.long, device=device)
        flag = torch.zeros(1, dtype=torch.int32, device=device)
        connection.send(dict(status='loaded', gpu=gpu, replica=replica, rank=rank,
                             name=torch.cuda.get_device_name(gpu),
                             model_commit=getattr(model.config, '_commit_hash', None),
                             tp_plan=getattr(model, '_tp_plan', None)))
        inputs = mask = None
        current_batch = None
        with torch.inference_mode():
            while True:
                command = connection.recv()
                if command['action'] == 'stop':
                    break
                batch = command['batch']
                if command['action'] == 'warmup':
                    inputs = base.unsqueeze(0).repeat(batch, 1)
                    mask = torch.ones_like(inputs)
                    current_batch = batch
                if batch != current_batch:
                    raise RuntimeError('Run batch must match the preceding warmup.')
                if tp:
                    dist.barrier()
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                batches = output_tokens = 0
                duration = args.warmup_seconds if command['action'] == 'warmup' else args.seconds
                while True:
                    # Only group rank 0 decides when to stop. Independent rank
                    # timers can otherwise cause mismatched collectives/deadlock.
                    keep_running = batches == 0 or time.perf_counter() - start < duration
                    if tp:
                        if rank == 0:
                            flag.fill_(int(keep_running))
                        dist.broadcast(flag, src=0)
                        keep_running = bool(flag.item())
                    if not keep_running:
                        break
                    generated = model.generate(
                        input_ids=inputs, attention_mask=mask, do_sample=False,
                        min_new_tokens=args.output_tokens, max_new_tokens=args.output_tokens,
                        pad_token_id=tokenizer.eos_token_id, use_cache=True,
                        synced_gpus=tp)
                    output_tokens += generated.shape[0] * (generated.shape[1] - inputs.shape[1])
                    del generated
                    torch.cuda.synchronize(device)
                    batches += 1
                if tp:
                    dist.barrier()
                torch.cuda.synchronize(device)
                connection.send(dict(status='done', gpu=gpu, replica=replica, rank=rank,
                                     batches=batches, requests=batches * batch,
                                     output_tokens=output_tokens))
    except BaseException:
        try:
            connection.send(dict(status='error', gpu=gpu, traceback=traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()
        if dist is not None and dist.is_initialized():
            dist.destroy_process_group()


def receive_all(workers, timeout):
    pending = {c: p for c, p in workers}
    results = {}
    deadline = time.monotonic() + timeout
    while pending:
        for c in wait(list(pending), timeout=0.2):
            p = pending[c]
            try:
                result = c.recv()
            except EOFError as exc:
                raise RuntimeError(f'{p.name} disconnected.') from exc
            if result['status'] == 'error':
                raise RuntimeError(result['traceback'])
            results[c] = result
            del pending[c]
        for p in pending.values():
            if not p.is_alive():
                raise RuntimeError(f'{p.name} exited with code {p.exitcode}.')
        if time.monotonic() > deadline:
            raise TimeoutError('Worker phase timed out; increase --worker-timeout if needed.')
    return [results[c] for c, _ in workers]


def make_power_monitor(monitor_class, gpu_indices):
    kwargs = {'gpu_indices': gpu_indices, 'update_period': 0.2}
    if 'cpu_indices' in inspect.signature(monitor_class).parameters:
        kwargs['cpu_indices'] = []
    return monitor_class(**kwargs)


def result_rows(args, batch, repeat, counts, measurement):
    common = dict(tp_size=args.tp_size, replicas=len(args.gpus) // args.tp_size,
                  batch_per_replica=batch, total_batch=batch * len(args.gpus) // args.tp_size,
                  repeat=repeat, elapsed_s=measurement.time)
    def row(scope, gpu, replica, energy, count=None):
        r = dict(common, scope=scope, gpu=gpu, replica=replica, energy_j=energy,
                 average_power_w=energy / measurement.time)
        if count is not None:
            requests, tokens = count['requests'], count['output_tokens']
            r.update(requests=requests, output_tokens=tokens,
                     requests_per_s=requests / measurement.time,
                     output_tokens_per_s=tokens / measurement.time,
                     joules_per_request=energy / requests, joules_per_output_token=energy / tokens)
        return r
    for c in counts:
        yield row('gpu', c['gpu'], c['replica'], measurement.gpu_energy[c['gpu']],
                  c if args.tp_size == 1 else None)
    leaders = [c for c in counts if c['rank'] == 0]
    for leader in leaders:
        group = [c for c in counts if c['replica'] == leader['replica']]
        if any((c['requests'], c['output_tokens']) !=
               (leader['requests'], leader['output_tokens']) for c in group):
            raise RuntimeError('TP ranks reported different counts.')
        energy = sum(measurement.gpu_energy[c['gpu']] for c in group)
        yield row('replica', 'group-' + str(leader['replica']), leader['replica'], energy, leader)
    total = {key: sum(c[key] for c in leaders) for key in ('requests', 'output_tokens')}
    yield row('all', 'all', 'all', sum(measurement.gpu_energy[g] for g in args.gpus), total)


def main():
    args = parse_args()
    import torch
    from zeus.monitor import ZeusMonitor
    from zeus.monitor.power import PowerMonitor
    if not torch.cuda.is_available() or max(args.gpus) >= torch.cuda.device_count():
        raise SystemExit('Requested GPUs unavailable; check CUDA_VISIBLE_DEVICES and PyTorch.')
    if args.tp_size > 1 and not torch.distributed.is_nccl_available():
        raise SystemExit('TP requires a PyTorch build with NCCL.')
    if not args.no_power_timeline and not hasattr(PowerMonitor, 'get_all_power_timelines'):
        raise SystemExit('Upgrade Zeus or pass --no-power-timeline.')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    context = mp.get_context('spawn')
    workers = []
    power = None
    successful = False
    # Distinct file rendezvous avoid collisions and isolate each TP replica.
    with tempfile.TemporaryDirectory(prefix='qwen_tp_') as rendezvous_dir:
        try:
            for index, gpu in enumerate(args.gpus):
                replica, rank = divmod(index, args.tp_size)
                rendezvous = (Path(rendezvous_dir) / f'replica_{replica}').as_uri()
                parent, child = context.Pipe()
                process = context.Process(target=worker,
                    args=(gpu, replica, rank, rendezvous, args, child), name=f'gpu-{gpu}')
                process.start()
                child.close()
                workers.append((parent, process))
            print(f'Loading {len(args.gpus) // args.tp_size} replica(s), TP={args.tp_size}...', flush=True)
            devices = receive_all(workers, args.worker_timeout)
            metadata = dict(vars(args), hostname=socket.gethostname(), devices=devices,
                batch_semantics='per_replica', schema_version=2,
                started_utc=datetime.now(timezone.utc).isoformat(),
                cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                torch_cuda_version=torch.version.cuda,
                versions={n: importlib.metadata.version(n) for n in ['torch', 'transformers', 'zeus', 'accelerate']})
            (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
            monitor = ZeusMonitor(gpu_indices=args.gpus, cpu_indices=[])
            fields = ['tp_size', 'replicas', 'batch_per_replica', 'total_batch', 'repeat', 'scope',
                      'gpu', 'replica', 'elapsed_s', 'energy_j', 'average_power_w', 'requests',
                      'output_tokens', 'requests_per_s', 'output_tokens_per_s',
                      'joules_per_request', 'joules_per_output_token']
            with (output / 'results.csv').open('w', newline='') as rf, \
                 (output / 'power.csv').open('w', newline='') as pf:
                writer = csv.DictWriter(rf, fieldnames=fields)
                writer.writeheader()
                pw = csv.writer(pf)
                pw.writerow(['batch_per_replica', 'repeat', 'domain', 'gpu',
                             'unix_timestamp_s', 'seconds_from_start', 'power_w', 'tp_size'])
                for batch in args.batch_sizes:
                    print(f'Warming up: batch {batch} per replica', flush=True)
                    for c, _ in workers:
                        c.send({'action': 'warmup', 'batch': batch})
                    receive_all(workers, args.worker_timeout)
                    for repeat in range(1, args.repeats + 1):
                        if not args.no_power_timeline:
                            power = make_power_monitor(PowerMonitor, args.gpus)
                        monitor.begin_window('run', sync_execution=False)
                        start_unix = time.time()
                        for c, _ in workers:
                            c.send({'action': 'run', 'batch': batch})
                        counts = receive_all(workers, args.worker_timeout)
                        measurement = monitor.end_window('run', sync_execution=False)
                        end_unix = time.time()
                        if power is not None:
                            timelines = power.get_all_power_timelines(start_time=start_unix, end_time=end_unix)
                            power.stop()
                            power = None
                            for domain, devices_timeline in timelines.items():
                                for gpu, samples in devices_timeline.items():
                                    for timestamp, watts in samples:
                                        pw.writerow([batch, repeat, domain, gpu, timestamp,
                                                     timestamp - start_unix, watts, args.tp_size])
                            pf.flush()
                        for row in result_rows(args, batch, repeat, counts, measurement):
                            writer.writerow(row)
                            if row['scope'] == 'all':
                                print(f"TP={args.tp_size} batch={batch} repeat={repeat}: "
                                      f"{row['energy_j']:.1f} J, {row['average_power_w']:.1f} W, "
                                      f"{row['output_tokens_per_s']:.1f} tokens/s, "
                                      f"{row['joules_per_output_token']:.4f} J/token", flush=True)
                        rf.flush()
            successful = True
            print(f'Results saved in {output.resolve()}')
        finally:
            if power is not None:
                power.stop()
            for c, p in workers:
                if p.is_alive():
                    if successful:
                        try:
                            c.send({'action': 'stop'})
                        except (BrokenPipeError, OSError):
                            pass
                    else:
                        p.terminate()
            for c, p in workers:
                p.join(timeout=5)
                if p.is_alive():
                    p.kill()
                    p.join(timeout=5)
                c.close()


if __name__ == '__main__':
    main()
