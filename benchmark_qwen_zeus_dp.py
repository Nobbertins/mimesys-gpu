#!/usr/bin/env python3
"""Qwen inference benchmark: one independent BF16 model replica per local GPU.

Setup (Python 3.10+, working CUDA-enabled PyTorch and NVIDIA driver):
    python -m pip install -U 'transformers>=4.51.0' accelerate zeus
Run:
    python benchmark_qwen_zeus.py --gpus 0 1
    python benchmark_qwen_zeus.py --gpus 0 --output machine1_results

GPU numbers are logical CUDA indices, respecting CUDA_VISIBLE_DEVICES.
Each batch size is PER GPU. Input tokenization, model loading, and warm-up are
excluded. Measurement includes generation (prefill + decode), synchronization,
and small process-control overhead. GPU device energy includes idle energy;
it is not whole-server energy or energy attributable only to this process.
Keep the selected GPUs free of other jobs. Synthetic repeated-text inputs and
forced output lengths measure performance, not model response quality.
"""
import argparse
import csv
import importlib.metadata
import inspect
import json
import multiprocessing as mp
import os
from pathlib import Path
import socket
import time
import traceback
from datetime import datetime, timezone


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='Qwen/Qwen3-4B')
    p.add_argument('--revision', default='main', help='Use a Hugging Face commit hash for reproducibility.')
    p.add_argument('--gpus', type=int, nargs='+', default=[0, 1])
    p.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 4, 8, 16])
    p.add_argument('--input-tokens', type=int, default=512)
    p.add_argument('--output-tokens', type=int, default=128)
    p.add_argument('--seconds', type=float, default=60, help='Minimum workload duration per repeat.')
    p.add_argument('--warmup-seconds', type=float, default=20)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--no-power-timeline', action='store_true', help='Still record energy and average power.')
    p.add_argument('--output', default='qwen_results_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    a = p.parse_args()
    if len(set(a.gpus)) != len(a.gpus) or min(a.gpus) < 0:
        p.error('GPU indices must be distinct nonnegative integers.')
    if min(a.batch_sizes + [a.input_tokens, a.output_tokens, a.repeats]) <= 0:
        p.error('Batch sizes, token lengths, and repeats must be positive.')
    if a.seconds <= 0 or a.warmup_seconds <= 0:
        p.error('Durations must be positive.')
    return a


def worker(gpu, args, connection):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(1)
        torch.cuda.set_device(gpu)
        torch.manual_seed(1234 + gpu)
        device = torch.device(f'cuda:{gpu}')
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, revision=args.revision, dtype=torch.bfloat16,
            device_map={'': str(device)}, attn_implementation='sdpa',
        ).eval()
        # Render non-thinking chat, then enforce an exact synthetic input length.
        prompt = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': 'Explain how computers process information. ' * args.input_tokens}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        tokens = (tokens * (args.input_tokens // len(tokens) + 1))[-args.input_tokens:]
        base = torch.tensor(tokens, dtype=torch.long, device=device)
        connection.send({'status': 'loaded', 'gpu': gpu,
                         'name': torch.cuda.get_device_name(gpu),
                         'model_commit': getattr(model.config, '_commit_hash', None)})
        inputs = mask = None
        with torch.inference_mode():
            while True:
                command = connection.recv()
                if command['action'] == 'stop':
                    break
                batch = command['batch']
                if command['action'] == 'warmup':
                    inputs = base.unsqueeze(0).repeat(batch, 1)
                    mask = torch.ones_like(inputs)
                def generate():
                    output = model.generate(
                        input_ids=inputs, attention_mask=mask, do_sample=False,
                        min_new_tokens=args.output_tokens, max_new_tokens=args.output_tokens,
                        pad_token_id=tokenizer.eos_token_id, use_cache=True,
                    )
                    n = output.shape[0] * (output.shape[1] - inputs.shape[1])
                    del output
                    torch.cuda.synchronize(device)
                    return n
                start = time.perf_counter()
                batches = output_tokens = 0
                duration = args.warmup_seconds if command['action'] == 'warmup' else args.seconds
                while batches == 0 or time.perf_counter() - start < duration:
                    output_tokens += generate()
                    batches += 1
                connection.send({'status': 'done', 'gpu': gpu, 'batches': batches,
                                 'requests': batches * batch, 'output_tokens': output_tokens})
    except BaseException:
        try:
            connection.send({'status': 'error', 'gpu': gpu, 'traceback': traceback.format_exc()})
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()


def receive(connection, process):
    while not connection.poll(0.2):
        if not process.is_alive():
            raise RuntimeError(f'Worker {process.name} exited with code {process.exitcode}.')
    try:
        result = connection.recv()
    except EOFError as exc:
        raise RuntimeError(f'Worker {process.name} disconnected.') from exc
    if result['status'] == 'error':
        raise RuntimeError(result['traceback'])
    return result


def make_power_monitor(monitor_class, gpu_indices):
    """Disable CPU monitoring only on Zeus releases that expose that option."""
    kwargs = {'gpu_indices': gpu_indices, 'update_period': 0.2}
    if 'cpu_indices' in inspect.signature(monitor_class).parameters:
        kwargs['cpu_indices'] = []
    return monitor_class(**kwargs)


def main():
    args = parse_args()
    import torch
    from zeus.monitor import ZeusMonitor
    from zeus.monitor.power import PowerMonitor
    if not torch.cuda.is_available() or max(args.gpus) >= torch.cuda.device_count():
        raise SystemExit('Requested GPUs unavailable. Check CUDA-enabled PyTorch and CUDA_VISIBLE_DEVICES.')
    if not args.no_power_timeline and not hasattr(PowerMonitor, 'get_all_power_timelines'):
        raise SystemExit('Upgrade Zeus: python -m pip install -U zeus; or use --no-power-timeline.')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)  # Never overwrite an existing experiment.
    context = mp.get_context('spawn')
    workers = []
    power = None
    try:
        for gpu in args.gpus:
            parent, child = context.Pipe()
            process = context.Process(target=worker, args=(gpu, args, child), name=f'gpu-{gpu}')
            process.start()
            child.close()
            workers.append((parent, process))
        print('Loading model replicas (first run downloads weights)...', flush=True)
        devices = [receive(c, p) for c, p in workers]
        metadata = dict(vars(args), hostname=socket.gethostname(), devices=devices,
                        started_utc=datetime.now(timezone.utc).isoformat(),
                        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                        torch_cuda_version=torch.version.cuda,
                        versions={name: importlib.metadata.version(name)
                                  for name in ['torch', 'transformers', 'zeus', 'accelerate']})
        (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        # Parent never dispatches CUDA work. Workers explicitly synchronize before
        # reporting completion, so parent-side CUDA synchronization is unnecessary.
        monitor = ZeusMonitor(gpu_indices=args.gpus, cpu_indices=[])
        fields = ['batch_per_gpu', 'repeat', 'gpu', 'elapsed_s', 'energy_j',
                  'average_power_w', 'requests', 'output_tokens', 'requests_per_s',
                  'output_tokens_per_s', 'joules_per_request', 'joules_per_output_token']
        with (output / 'results.csv').open('w', newline='') as results_file, \
             (output / 'power.csv').open('w', newline='') as power_file:
            writer = csv.DictWriter(results_file, fieldnames=fields)
            writer.writeheader()
            power_writer = csv.writer(power_file)
            power_writer.writerow(['batch_per_gpu', 'repeat', 'domain', 'gpu',
                                   'unix_timestamp_s', 'seconds_from_start', 'power_w'])
            for batch in args.batch_sizes:
                print(f'Warming up: batch {batch} per GPU', flush=True)
                for c, _ in workers:
                    c.send({'action': 'warmup', 'batch': batch})
                for c, p in workers:
                    receive(c, p)
                for repeat in range(1, args.repeats + 1):
                    if not args.no_power_timeline:
                        power = make_power_monitor(PowerMonitor, args.gpus)
                    monitor.begin_window('run', sync_execution=False)
                    start_unix = time.time()
                    for c, _ in workers:
                        c.send({'action': 'run', 'batch': batch})
                    counts = [receive(c, p) for c, p in workers]
                    measurement = monitor.end_window('run', sync_execution=False)
                    end_unix = time.time()
                    if power is not None:
                        timelines = power.get_all_power_timelines(start_time=start_unix, end_time=end_unix)
                        power.stop()
                        for domain, devices_timeline in timelines.items():
                            for gpu, samples in devices_timeline.items():
                                for timestamp, watts in samples:
                                    power_writer.writerow([batch, repeat, domain, gpu, timestamp,
                                                           timestamp - start_unix, watts])
                        power_file.flush()
                        power = None
                    combined = {'gpu': 'all', 'requests': sum(c['requests'] for c in counts),
                                'output_tokens': sum(c['output_tokens'] for c in counts)}
                    for count in counts + [combined]:
                        gpu = count['gpu']
                        energy = (sum(measurement.gpu_energy.values()) if gpu == 'all'
                                  else measurement.gpu_energy[gpu])
                        elapsed = measurement.time
                        row = dict(batch_per_gpu=batch, repeat=repeat, gpu=gpu,
                                   elapsed_s=elapsed, energy_j=energy, average_power_w=energy / elapsed,
                                   requests=count['requests'], output_tokens=count['output_tokens'],
                                   requests_per_s=count['requests'] / elapsed,
                                   output_tokens_per_s=count['output_tokens'] / elapsed,
                                   joules_per_request=energy / count['requests'],
                                   joules_per_output_token=energy / count['output_tokens'])
                        writer.writerow(row)
                        if gpu == 'all':
                            print(f"batch={batch} repeat={repeat}: {energy:.1f} J, "
                                  f"{row['average_power_w']:.1f} W, "
                                  f"{row['output_tokens_per_s']:.1f} tokens/s, "
                                  f"{row['joules_per_output_token']:.4f} J/token", flush=True)
                    results_file.flush()
        print(f'Results saved in {output.resolve()}')
    finally:
        if power is not None:
            power.stop()
        for c, p in workers:
            if p.is_alive():
                try:
                    c.send({'action': 'stop'})
                except (BrokenPipeError, OSError):
                    pass
        for c, p in workers:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
                p.join(timeout=5)
            c.close()


if __name__ == '__main__':
    main()
