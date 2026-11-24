#!/usr/bin/env python3
"""Run isolated, label-free ScienceQA greedy shards in owned Hygon services.

Launch inside the Hygon container. The input paths must already be present.
Each model copy serves one device; online memory is not sharded here.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def verify_weights(model: Path, manifest: Path) -> None:
    """Manifest is a JSON object mapping model-relative filenames to SHA256."""
    gate = json.loads(manifest.read_text())
    entries = gate.get('files') if isinstance(gate, dict) else None
    if gate.get('verified') is not True:
        raise ValueError('Weight hash gate has not passed')
    if not isinstance(entries, dict) or not entries:
        raise ValueError('Weight manifest must be a nonempty filename-to-sha256 JSON object')
    shards = {p.name for p in model.glob('*.safetensors') if p.is_file()}
    metadata = {p.name for p in model.iterdir() if p.is_file() and
                (p.name.startswith(('tokenizer', 'preprocessor', 'processor', 'vocab', 'merges'))
                 or p.name in ('config.json', 'special_tokens_map.json'))}
    if not shards or not shards.issubset(entries) or not metadata.issubset(entries):
        raise ValueError('Weight gate must include all model shards and tokenizer/processor metadata')
    if 'config.json' not in metadata or not any(x.startswith('tokenizer') for x in metadata):
        raise ValueError('Model config or tokenizer is missing')
    for filename, expected in entries.items():
        path = (model / filename).resolve()
        if not path.is_relative_to(model.resolve()) or not path.is_file():
            raise ValueError(f'Invalid manifest file: {filename}')
        if not isinstance(expected, str) or len(expected) != 64 or digest(path) != expected.lower():
            raise ValueError(f'Weight hash mismatch: {filename}')


def verify_smoke(path: Path, reference: Path) -> None:
    """Smoke is the 20-row output of hygon_scienceqa_greedy.py."""
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != 20 or [r.get('sample_index') for r in rows] != list(range(20)):
        raise ValueError('Smoke must contain the ordered 20-sample prefix')
    refs = [json.loads(line) for line in reference.read_text().splitlines()]
    for row, ref in zip(rows, refs):
        response = row.get('response', {})
        if (row.get('labels_read') is not False or not response.get('completion_tokens') or
                row.get('sample_id') != ref['sample_id'] or
                row.get('image_hashes') != ref['image_hashes']):
            raise ValueError('Smoke has a missing response or label boundary')
        import math
        values = response.get('logprobs', [])
        if len(values) != response['completion_tokens'] or not all(
                math.isfinite(v) and v <= 0 for v in values):
            raise ValueError('Smoke does not have complete finite logprobs')


def free_port(port: int) -> None:
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', port))


def health(port: int) -> bool:
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def cleanup(processes: list[subprocess.Popen]) -> None:
    for proc in processes:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 20
    for proc in processes:
        try:
            proc.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--weight-manifest', type=Path, required=True)
    p.add_argument('--smoke-predictions', type=Path, required=True)
    p.add_argument('--inference', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--devices', default='0,1,2,3,4,5,6,7')
    p.add_argument('--base-port', type=int, default=19100)
    p.add_argument('--count', type=int, default=400)
    p.add_argument('--max-model-len', type=int, default=8192)
    p.add_argument('--gpu-memory-utilization', type=float, default=0.75)
    p.add_argument('--ready-timeout', type=int, default=1200)
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--vllm', default='vllm')
    a = p.parse_args()
    a.project, a.model = a.project.resolve(), a.model.resolve()
    a.inference, a.reference = a.inference.resolve(), a.reference.resolve()
    a.output = a.output.resolve()
    devices = a.devices.split(',')
    if len(devices) != 8 or len(set(devices)) != 8 or not all(x.isdigit() for x in devices):
        raise ValueError('Provide eight distinct numeric HIP device indices')
    if not 8 <= a.count <= 400 or not 1024 <= a.base_port <= 65528:
        raise ValueError('Invalid count or base port')
    if a.output.exists():
        raise FileExistsError(a.output)
    gate = json.loads(a.weight_manifest.read_text())
    if (gate.get('smoke_passed') is not True or
            gate.get('smoke_sha256') != digest(a.smoke_predictions)):
        raise ValueError('Weight gate must bind a passed smoke prediction hash')
    verify_smoke(a.smoke_predictions, a.reference)
    verify_weights(a.model, a.weight_manifest)
    for port in range(a.base_port, a.base_port + 8):
        free_port(port)
    os.chdir(a.project)
    worker_script = str(a.project / 'scripts/hygon_scienceqa_greedy.py')
    common = [a.python, worker_script, '--inference', str(a.inference),
              '--reference', str(a.reference), '--count', str(a.count), '--workers', '8']
    for index in range(8):
        subprocess.run(common + ['--worker', str(index), '--output', str(a.output / 'unused'),
                                 '--base-url', f'http://127.0.0.1:{a.base_port + index}/v1',
                                 '--check-only'], check=True)
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / 'run_config.json').write_text(json.dumps({
        'devices': devices, 'count': a.count, 'base_port': a.base_port,
        'inference_sha256': digest(a.inference), 'reference_sha256': digest(a.reference),
        'weight_manifest_sha256': digest(a.weight_manifest),
        'smoke_sha256': digest(a.smoke_predictions), 'labels_read': False,
        'max_model_len': a.max_model_len, 'gpu_memory_utilization': a.gpu_memory_utilization,
        'source_sha256': {str(path.relative_to(a.project)): digest(path) for path in [
            a.project / 'scripts/run_hygon_scienceqa_parallel.py',
            a.project / 'scripts/hygon_scienceqa_greedy.py',
            a.project / 'core/multimodal/client.py',
            a.project / 'core/multimodal/tf_ttcl_direct_runner.py',
            a.project / 'core/multimodal/tf_ttcl_direct_selection.py']},
        'software_versions': {name: importlib.metadata.version(name)
                              for name in ('torch', 'vllm', 'transformers')},
        'model_revision': 'cc594898137f460bfe9f0759e9844b3ce807cfb5',
        'seed': 0,
    }, indent=2) + '\n')
    services: list[subprocess.Popen] = []
    workers: list[subprocess.Popen] = []
    logs = []
    def interrupt(signum, _frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        for index, device in enumerate(devices):
            log = (a.output / f'vllm_{index}.log').open('x')
            logs.append(log)
            env = os.environ.copy()
            env['HIP_VISIBLE_DEVICES'] = device
            env['VLLM_USE_FLASHINFER_SAMPLER'] = '0'
            cmd = [a.vllm, 'serve', str(a.model), '--served-model-name',
                   'Qwen/Qwen2.5-VL-7B-Instruct', '--host', '127.0.0.1',
                   '--port', str(a.base_port + index), '--trust-remote-code',
                   '--gpu-memory-utilization', str(a.gpu_memory_utilization),
                   '--max-model-len', str(a.max_model_len)]
            services.append(subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                             start_new_session=True))
        deadline = time.monotonic() + a.ready_timeout
        waiting = set(range(8))
        while waiting:
            for index, proc in enumerate(services):
                if proc.poll() is not None:
                    raise RuntimeError(f'Service {index} exited with {proc.returncode}')
            waiting = {i for i in waiting if not health(a.base_port + i)}
            if time.monotonic() >= deadline and waiting:
                raise TimeoutError(f'Services not ready: {sorted(waiting)}')
            if waiting:
                time.sleep(2)
        for index in range(8):
            log = (a.output / f'worker_{index}.log').open('x')
            logs.append(log)
            cmd = common + ['--worker', str(index), '--output', str(a.output / f'predictions_{index}.jsonl'),
                            '--base-url', f'http://127.0.0.1:{a.base_port + index}/v1']
            workers.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        while any(proc.poll() is None for proc in workers):
            for index, proc in enumerate(services):
                if proc.poll() is not None:
                    raise RuntimeError(f'Service {index} exited during inference: {proc.returncode}')
            for index, proc in enumerate(workers):
                if proc.poll() not in (None, 0):
                    raise RuntimeError(f'Worker {index} failed: {proc.returncode}')
            time.sleep(2)
        if any(proc.returncode != 0 for proc in workers):
            raise RuntimeError('At least one worker failed')
        rows = []
        for index in range(8):
            shard = [json.loads(line) for line in (a.output / f'predictions_{index}.jsonl').read_text().splitlines()]
            if [row['sample_index'] for row in shard] != list(range(index, a.count, 8)):
                raise ValueError(f'Shard {index} has missing, duplicate, or out-of-order rows')
            if any(row.get('labels_read') is not False for row in shard):
                raise ValueError('Label boundary violation')
            rows.extend(shard)
        rows.sort(key=lambda row: row['sample_index'])
        with (a.output / 'predictions.jsonl').open('x') as out:
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + '\n')
        (a.output / 'completed.json').write_text(json.dumps({
            'status': 'completed', 'processed': len(rows), 'labels_read': False,
            'predictions_sha256': digest(a.output / 'predictions.jsonl')}, indent=2) + '\n')
        print(f'Completed {len(rows)} label-free predictions in {a.output}', flush=True)
    finally:
        cleanup(workers + services)
        for log in logs:
            log.close()


if __name__ == '__main__':
    main()
