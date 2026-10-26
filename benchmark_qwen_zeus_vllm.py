#!/usr/bin/env python3
"""Qwen serving benchmark: vLLM servers + open-loop Poisson load + Zeus GPU energy.

Setup (Python 3.10+, NVIDIA driver, CUDA-enabled PyTorch as pulled in by vLLM):
    python -m pip install -U vllm zeus transformers aiohttp
Optional but recommended: a ShareGPT-style JSON file (list of records with a
"conversations" list of {"from": "human"|"gpt", "value": ...}), e.g.
ShareGPT_V3_unfiltered_cleaned_split.json from Hugging Face. Without --dataset,
a synthetic lognormal length distribution is used instead.
Run:
    python benchmark_qwen_vllm_zeus.py --gpus 0 1 --dataset ShareGPT.json --rates 1 2 4 8

Design:
  * One independent vLLM server (one full BF16 replica) per selected GPU, on
    ports base-port, base-port+1, ... The client round-robins requests over them.
  * Load is OPEN-LOOP: requests arrive as a Poisson process at the offered rate
    regardless of how fast the servers respond, so queues build up when the
    rate exceeds capacity. --rates are requests/s PER GPU (total = rate * #GPUs).
  * Each run = warm-up + measured window + cool-down, all at constant offered
    load. Zeus energy covers exactly the measured window. Requests are counted
    (throughput, latency, energy per request/token) if they COMPLETE inside that
    window, which approximates steady state without drain-tail artifacts.
  * Energy is GPU device energy only (NVML), including idle power. It is not
    whole-server energy. Keep the selected GPUs free of other jobs.
  * Prompt/response tokenization and dataset loading happen before any server
    starts and are excluded from measurement. The client runs on the same host,
    so its CPU use is not measured (cpu_indices=[]) but can perturb latency
    slightly at high request rates.
"""
import argparse
import asyncio
import csv
import importlib.metadata
import inspect
import json
import math
import os
import random
import shlex
import shutil
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

RESULT_FIELDS = [
    'rate_per_gpu', 'repeat', 'gpu', 'offered_rps', 'elapsed_s', 'energy_j',
    'average_power_w', 'completed_requests', 'failed_requests', 'output_tokens',
    'requests_per_s', 'output_tokens_per_s', 'goodput_rps', 'joules_per_request',
    'joules_per_output_token', 'mean_input_tokens', 'mean_output_tokens',
    'ttft_p50_ms', 'ttft_p95_ms', 'ttft_p99_ms',
    'itl_p50_ms', 'itl_p95_ms', 'itl_p99_ms',
    'e2e_p50_ms', 'e2e_p95_ms', 'e2e_p99_ms',
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='Qwen/Qwen3-4B')
    p.add_argument('--revision', default='main', help='Use a Hugging Face commit hash for reproducibility.')
    p.add_argument('--gpus', type=int, nargs='+', default=[0, 1],
                   help='Logical CUDA indices (respect CUDA_VISIBLE_DEVICES).')
    p.add_argument('--rates', type=float, nargs='+', default=[0.5, 1, 2, 4, 8],
                   help='Offered load in requests/s PER GPU.')
    p.add_argument('--dataset', default=None, help='ShareGPT-style JSON. Omit for synthetic lengths.')
    p.add_argument('--output-mode', choices=['natural', 'recorded'], default='natural',
                   help="natural: model stops on EOS (capped by --max-output-tokens). "
                        "recorded: force the dataset's response length with ignore_eos.")
    p.add_argument('--max-input-tokens', type=int, default=1024)
    p.add_argument('--max-output-tokens', type=int, default=512)
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--top-p', type=float, default=0.8)
    p.add_argument('--seconds', type=float, default=120, help='Measured window per repeat.')
    p.add_argument('--warmup-seconds', type=float, default=30)
    p.add_argument('--cooldown-seconds', type=float, default=15,
                   help='Traffic continues after the window so load does not fall off early.')
    p.add_argument('--drain-timeout', type=float, default=180,
                   help='Max wait for in-flight requests after traffic stops.')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--idle-seconds', type=float, default=30,
                   help='Idle-baseline measurement with servers loaded and no traffic (0 to skip).')
    p.add_argument('--slo-ttft-ms', type=float, default=1000)
    p.add_argument('--slo-tpot-ms', type=float, default=50,
                   help='SLO on mean time per output token (after the first) per request.')
    p.add_argument('--max-model-len', type=int, default=4096)
    p.add_argument('--gpu-memory-utilization', type=float, default=0.90)
    p.add_argument('--base-port', type=int, default=8000)
    p.add_argument('--server-timeout', type=float, default=900)
    p.add_argument('--server-args', default='',
                   help='Extra vllm serve flags as one quoted string, e.g. "--max-num-seqs 256 --no-enable-prefix-caching".')
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--no-power-timeline', action='store_true', help='Still record energy and average power.')
    p.add_argument('--output', default='qwen_vllm_results_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    a = p.parse_args()
    if len(set(a.gpus)) != len(a.gpus) or min(a.gpus) < 0:
        p.error('GPU indices must be distinct nonnegative integers.')
    if min(a.rates) <= 0:
        p.error('Rates must be positive.')
    if min([a.max_input_tokens, a.max_output_tokens, a.repeats, a.max_model_len]) <= 0:
        p.error('Token limits and repeats must be positive.')
    if a.seconds <= 0 or a.warmup_seconds < 0 or a.cooldown_seconds < 0 or a.idle_seconds < 0:
        p.error('Durations must be positive (warm-up, cool-down, idle may be zero).')
    if a.max_input_tokens + a.max_output_tokens + 64 > a.max_model_len:
        p.error('--max-model-len must exceed max input + max output tokens (plus chat-template overhead).')
    return a


@dataclass
class Request:
    prompt: str
    input_tokens: int
    output_tokens: int  # recorded/sampled response length (used as max_tokens)


@dataclass
class Outcome:
    server: int
    start: float
    end: float
    ttft: float | None
    token_times: list
    prompt_tokens: int
    completion_tokens: int
    error: str | None


# ----------------------------------------------------------------- workload

def load_requests(args, needed):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    rng = random.Random(args.seed)
    reqs = []
    if args.dataset:
        with open(args.dataset) as f:
            data = json.load(f)
        rng.shuffle(data)
        for item in data:
            conv = item.get('conversations') or []
            if len(conv) < 2 or conv[0].get('from') != 'human':
                continue
            prompt, answer = conv[0]['value'], conv[1]['value']
            n_in = len(tok.encode(prompt, add_special_tokens=False))
            n_out = len(tok.encode(answer, add_special_tokens=False))
            if n_in < 4 or n_out < 4 or n_in > args.max_input_tokens:
                continue
            reqs.append(Request(prompt, n_in, min(n_out, args.max_output_tokens)))
            if len(reqs) >= needed:
                break
        if not reqs:
            raise SystemExit('No usable conversations found in the dataset.')
        if len(reqs) < needed:
            print(f'WARNING: only {len(reqs)} usable prompts for {needed} needed; prompts will repeat, '
                  'which prefix caching can make unrealistically cheap.', flush=True)
    else:
        hi = min(tok.vocab_size, 150000)
        for _ in range(needed):
            n_in = int(min(max(rng.lognormvariate(math.log(200), 0.9), 4), args.max_input_tokens))
            n_out = int(min(max(rng.lognormvariate(math.log(180), 0.8), 4), args.max_output_tokens))
            ids = [rng.randrange(1000, hi) for _ in range(n_in)]
            reqs.append(Request(tok.decode(ids, skip_special_tokens=True), n_in, n_out))
    return reqs


def arrival_times(rate, horizon, rng):
    t, out = 0.0, []
    while True:
        t += rng.expovariate(rate)
        if t >= horizon:
            return out
        out.append(t)


# ------------------------------------------------------------------ servers

def start_servers(args, output):
    if shutil.which('vllm') is None:
        raise SystemExit('`vllm` executable not found. Install with: python -m pip install -U vllm')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    mapping = [x.strip() for x in visible.split(',')] if visible else None
    if mapping is not None and max(args.gpus) >= len(mapping):
        raise SystemExit('Requested GPU index exceeds CUDA_VISIBLE_DEVICES.')
    procs, ports, commands = [], [], []
    for i, gpu in enumerate(args.gpus):
        physical = mapping[gpu] if mapping else str(gpu)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=physical)
        port = args.base_port + i
        cmd = ['vllm', 'serve', args.model, '--revision', args.revision, '--host', '127.0.0.1',
               '--port', str(port), '--dtype', 'bfloat16', '--max-model-len', str(args.max_model_len),
               '--gpu-memory-utilization', str(args.gpu_memory_utilization), '--seed', str(args.seed)]
        cmd += shlex.split(args.server_args)
        log = open(output / f'server_gpu{gpu}.log', 'w')
        procs.append(subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                      start_new_session=True))
        ports.append(port)
        commands.append(' '.join(cmd))
    return procs, ports, commands


def wait_ready(procs, ports, timeout, output, gpus):
    deadline = time.time() + timeout
    pending = set(range(len(ports)))
    while pending:
        for i in list(pending):
            if procs[i].poll() is not None:
                raise RuntimeError(f'vLLM server for GPU {gpus[i]} exited with code {procs[i].returncode}. '
                                   f'See {output / f"server_gpu{gpus[i]}.log"}')
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{ports[i]}/health', timeout=2) as r:
                    if r.status == 200:
                        pending.discard(i)
            except (urllib.error.URLError, OSError):
                pass
        if pending and time.time() > deadline:
            raise TimeoutError('Timed out waiting for vLLM servers to become ready.')
        if pending:
            time.sleep(2)


def stop_servers(procs):
    for p in procs:
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for p in procs:
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            p.wait(timeout=10)


# -------------------------------------------------------------- load driver

async def one_request(session, port, server, req, args, outcomes):
    payload = {
        'model': args.model,
        'messages': [{'role': 'user', 'content': req.prompt}],
        'max_tokens': req.output_tokens if args.output_mode == 'recorded' else args.max_output_tokens,
        'temperature': args.temperature, 'top_p': args.top_p,
        'stream': True, 'stream_options': {'include_usage': True},
        'chat_template_kwargs': {'enable_thinking': False},
    }
    if args.output_mode == 'recorded':
        payload['ignore_eos'] = True
    start = time.perf_counter()
    token_times, usage, error = [], {}, None
    try:
        async with session.post(f'http://127.0.0.1:{port}/v1/chat/completions', json=payload) as resp:
            if resp.status != 200:
                error = f'HTTP {resp.status}: {(await resp.text())[:200]}'
            else:
                async for raw in resp.content:
                    line = raw.decode('utf-8', 'ignore').strip()
                    if not line.startswith('data:'):
                        continue
                    data = line[5:].strip()
                    if data == '[DONE]':
                        break
                    chunk = json.loads(data)
                    if chunk.get('usage'):
                        usage = chunk['usage']
                    choices = chunk.get('choices') or []
                    if choices and (choices[0].get('delta') or {}).get('content'):
                        token_times.append(time.perf_counter())
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # network errors, malformed stream, etc.
        error = repr(exc)
    end = time.perf_counter()
    outcomes.append(Outcome(
        server=server, start=start, end=end,
        ttft=(token_times[0] - start) if token_times else None,
        token_times=token_times,
        prompt_tokens=usage.get('prompt_tokens', req.input_tokens),
        completion_tokens=usage.get('completion_tokens', len(token_times)),
        error=error))


async def drive(args, ports, reqs, rate_total, window_name, monitor, rng):
    horizon = args.warmup_seconds + args.seconds + args.cooldown_seconds
    arrivals = arrival_times(rate_total, horizon, rng)
    w_lo, w_hi = args.warmup_seconds, args.warmup_seconds + args.seconds
    offered = {s: 0 for s in range(len(ports))}
    outcomes, window, tasks, max_lag = [], {}, [], 0.0
    connector = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        t_run = time.perf_counter()

        async def measure():
            await asyncio.sleep(max(0.0, t_run + w_lo - time.perf_counter()))
            monitor.begin_window(window_name, sync_execution=False)
            window['start'], window['start_unix'] = time.perf_counter(), time.time()
            await asyncio.sleep(max(0.0, window['start'] + args.seconds - time.perf_counter()))
            window['measurement'] = monitor.end_window(window_name, sync_execution=False)
            window['end'], window['end_unix'] = time.perf_counter(), time.time()

        measure_task = asyncio.create_task(measure())
        for i, at in enumerate(arrivals):
            delay = t_run + at - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            else:
                max_lag = max(max_lag, -delay)
            server = i % len(ports)
            if w_lo <= at < w_hi:
                offered[server] += 1
            tasks.append(asyncio.create_task(
                one_request(session, ports[server], server, reqs[i % len(reqs)], args, outcomes)))
        await measure_task
        unfinished = 0
        if tasks:
            _, pending = await asyncio.wait(tasks, timeout=args.drain_timeout)
            unfinished = len(pending)
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
    return dict(outcomes=outcomes, window=window, offered=offered, t_run=t_run,
                unfinished=unfinished, max_lag=max_lag, total_requests=len(arrivals))


# ---------------------------------------------------------------- summaries

def pct(values, q):
    if not values:
        return float('nan')
    s = sorted(values)
    k = (len(s) - 1) * q / 100
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def div(a, b):
    return a / b if b else float('nan')


def summarize(outs, w_lo, w_hi, elapsed, energy, offered, args):
    done = [o for o in outs if o.error is None and w_lo <= o.end <= w_hi]
    failed = [o for o in outs if o.error is not None and w_lo <= o.end <= w_hi]
    ttft = [o.ttft * 1000 for o in done if o.ttft is not None]
    e2e = [(o.end - o.start) * 1000 for o in done]
    itl, good = [], 0
    for o in done:
        gaps = [(b - a) * 1000 for a, b in zip(o.token_times, o.token_times[1:])]
        itl += gaps
        tpot = sum(gaps) / len(gaps) if gaps else None
        if o.ttft is not None and o.ttft * 1000 <= args.slo_ttft_ms and \
                (tpot is None or tpot <= args.slo_tpot_ms):
            good += 1
    tokens = sum(o.completion_tokens for o in done)
    return dict(
        offered_rps=offered / args.seconds, elapsed_s=elapsed, energy_j=energy,
        average_power_w=div(energy, elapsed), completed_requests=len(done),
        failed_requests=len(failed), output_tokens=tokens,
        requests_per_s=len(done) / elapsed, output_tokens_per_s=tokens / elapsed,
        goodput_rps=good / elapsed, joules_per_request=div(energy, len(done)),
        joules_per_output_token=div(energy, tokens),
        mean_input_tokens=div(sum(o.prompt_tokens for o in done), len(done)),
        mean_output_tokens=div(tokens, len(done)),
        ttft_p50_ms=pct(ttft, 50), ttft_p95_ms=pct(ttft, 95), ttft_p99_ms=pct(ttft, 99),
        itl_p50_ms=pct(itl, 50), itl_p95_ms=pct(itl, 95), itl_p99_ms=pct(itl, 99),
        e2e_p50_ms=pct(e2e, 50), e2e_p95_ms=pct(e2e, 95), e2e_p99_ms=pct(e2e, 99))


def make_power_monitor(monitor_class, gpu_indices):
    """Disable CPU monitoring only on Zeus releases that expose that option."""
    kwargs = {'gpu_indices': gpu_indices, 'update_period': 0.2}
    if 'cpu_indices' in inspect.signature(monitor_class).parameters:
        kwargs['cpu_indices'] = []
    return monitor_class(**kwargs)


def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


# --------------------------------------------------------------------- main

def main():
    args = parse_args()
    from zeus.monitor import ZeusMonitor
    from zeus.monitor.power import PowerMonitor
    if not args.no_power_timeline and not hasattr(PowerMonitor, 'get_all_power_timelines'):
        raise SystemExit('Upgrade Zeus: python -m pip install -U zeus; or use --no-power-timeline.')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)  # Never overwrite an existing experiment.
    n_gpus = len(args.gpus)
    horizon = args.warmup_seconds + args.seconds + args.cooldown_seconds
    needed = math.ceil(max(args.rates) * n_gpus * horizon * 1.3) + 100
    print(f'Preparing up to {needed} requests...', flush=True)
    reqs = load_requests(args, needed)

    procs = []
    power = None
    try:
        procs, ports, commands = start_servers(args, output)
        print('Starting vLLM servers (first run downloads weights)...', flush=True)
        wait_ready(procs, ports, args.server_timeout, output, args.gpus)
        metadata = dict(vars(args), hostname=socket.gethostname(), ports=ports, server_commands=commands,
                        started_utc=datetime.now(timezone.utc).isoformat(),
                        cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                        prompt_pool_size=len(reqs),
                        pool_mean_input_tokens=sum(r.input_tokens for r in reqs) / len(reqs),
                        pool_mean_recorded_output_tokens=sum(r.output_tokens for r in reqs) / len(reqs),
                        versions={n: version(n) for n in ['vllm', 'torch', 'transformers', 'zeus', 'aiohttp']})
        (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')

        monitor = ZeusMonitor(gpu_indices=args.gpus, cpu_indices=[])

        if args.idle_seconds > 0:
            print(f'Measuring idle baseline for {args.idle_seconds:.0f} s...', flush=True)
            time.sleep(5)
            monitor.begin_window('idle', sync_execution=False)
            time.sleep(args.idle_seconds)
            idle = monitor.end_window('idle', sync_execution=False)
            with (output / 'idle.csv').open('w', newline='') as f:
                w = csv.writer(f)
                w.writerow(['gpu', 'elapsed_s', 'energy_j', 'average_power_w'])
                for gpu in args.gpus:
                    w.writerow([gpu, idle.time, idle.gpu_energy[gpu], idle.gpu_energy[gpu] / idle.time])
            print('Idle power (W): ' + ', '.join(
                f'GPU{g}={idle.gpu_energy[g] / idle.time:.1f}' for g in args.gpus), flush=True)

        with (output / 'results.csv').open('w', newline='') as rf, \
             (output / 'power.csv').open('w', newline='') as pf, \
             (output / 'requests.csv').open('w', newline='') as qf:
            writer = csv.DictWriter(rf, fieldnames=RESULT_FIELDS)
            writer.writeheader()
            power_writer = csv.writer(pf)
            power_writer.writerow(['rate_per_gpu', 'repeat', 'domain', 'gpu',
                                   'unix_timestamp_s', 'seconds_from_start', 'power_w'])
            req_writer = csv.writer(qf)
            req_writer.writerow(['rate_per_gpu', 'repeat', 'gpu', 'start_s', 'ttft_s', 'e2e_s',
                                 'prompt_tokens', 'completion_tokens', 'error'])

            for rate in args.rates:
                for repeat in range(1, args.repeats + 1):
                    rate_total = rate * n_gpus
                    print(f'Running: {rate} req/s per GPU ({rate_total} total), repeat {repeat}...', flush=True)
                    rng = random.Random(args.seed * 1000 + int(rate * 1000) * 10 + repeat)
                    power = None if args.no_power_timeline else make_power_monitor(PowerMonitor, args.gpus)
                    run = asyncio.run(drive(args, ports, reqs, rate_total,
                                            f'rate{rate}_rep{repeat}', monitor, rng))
                    window = run['window']
                    measurement = window['measurement']
                    if power is not None:
                        timelines = power.get_all_power_timelines(start_time=window['start_unix'],
                                                                  end_time=window['end_unix'])
                        power.stop()
                        power = None
                        for domain, per_gpu in timelines.items():
                            for gpu, samples in per_gpu.items():
                                for ts, watts in samples:
                                    power_writer.writerow([rate, repeat, domain, gpu, ts,
                                                           ts - window['start_unix'], watts])
                        pf.flush()
                    if run['unfinished']:
                        print(f"  WARNING: {run['unfinished']} requests unfinished after drain timeout "
                              '(system is overloaded at this rate).', flush=True)
                    if run['max_lag'] > 0.1:
                        print(f"  WARNING: client dispatch lagged by up to {run['max_lag']:.2f} s; "
                              'offered load may be lower than requested.', flush=True)
                    t0 = run['t_run']
                    for o in run['outcomes']:
                        req_writer.writerow([rate, repeat, args.gpus[o.server], o.start - t0,
                                             '' if o.ttft is None else o.ttft, o.end - o.start,
                                             o.prompt_tokens, o.completion_tokens, o.error or ''])
                    qf.flush()
                    elapsed = measurement.time
                    rows = []
                    for s, gpu in enumerate(args.gpus):
                        outs = [o for o in run['outcomes'] if o.server == s]
                        rows.append((gpu, summarize(outs, window['start'], window['end'], elapsed,
                                                    measurement.gpu_energy[gpu], run['offered'][s], args)))
                    rows.append(('all', summarize(run['outcomes'], window['start'], window['end'], elapsed,
                                                  sum(measurement.gpu_energy.values()),
                                                  sum(run['offered'].values()), args)))
                    for gpu, row in rows:
                        writer.writerow(dict(row, rate_per_gpu=rate, repeat=repeat, gpu=gpu))
                    rf.flush()
                    a = rows[-1][1]
                    print(f"  {a['energy_j']:.0f} J, {a['average_power_w']:.1f} W | "
                          f"{a['requests_per_s']:.2f} req/s (offered {a['offered_rps']:.2f}), "
                          f"{a['output_tokens_per_s']:.0f} tok/s, {a['joules_per_output_token']:.4f} J/token | "
                          f"TTFT p50/p99 {a['ttft_p50_ms']:.0f}/{a['ttft_p99_ms']:.0f} ms, "
                          f"ITL p99 {a['itl_p99_ms']:.0f} ms, failed {a['failed_requests']}", flush=True)
        print(f'Results saved in {output.resolve()}')
    finally:
        if power is not None:
            power.stop()
        stop_servers(procs)


if __name__ == '__main__':
    main()