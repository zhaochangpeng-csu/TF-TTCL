#!/usr/bin/env python3
"""Launch isolated HCU services to replay the ScienceQA no-rewrite vote."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_hygon_scienceqa_parallel import cleanup, digest, free_port, health, verify_smoke, verify_weights


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('project', 'model', 'weight-manifest', 'smoke-predictions', 'inference',
                 'reference', 'greedy', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--greedy-sha256', required=True)
    p.add_argument('--vote-smoke', type=Path)
    p.add_argument('--count', type=int, choices=(20, 400), required=True)
    p.add_argument('--devices', default='0,1,2,3,4,5,6,7')
    p.add_argument('--base-port', type=int, default=19200)
    p.add_argument('--ready-timeout', type=int, default=1200)
    a = p.parse_args()
    for key in ('project', 'model', 'weight_manifest', 'smoke_predictions', 'inference',
                'reference', 'greedy', 'output'):
        setattr(a, key, getattr(a, key).resolve())
    devices = a.devices.split(',')
    if len(devices) != 8 or set(devices) != set('01234567') or not 1024 <= a.base_port <= 65528:
        raise ValueError('Expected all 8 distinct device IDs and valid port range')
    if a.output.exists():
        raise FileExistsError(a.output)
    gate = json.loads(a.weight_manifest.read_text())
    if gate.get('smoke_passed') is not True or gate.get('smoke_sha256') != digest(a.smoke_predictions):
        raise ValueError('Original Hygon greedy smoke gate failed')
    verify_smoke(a.smoke_predictions, a.reference)
    verify_weights(a.model, a.weight_manifest)
    if digest(a.greedy) != a.greedy_sha256:
        raise ValueError('Greedy predictions hash mismatch')
    if a.count == 400:
        if not a.vote_smoke or not a.vote_smoke.is_file():
            raise ValueError('A completed vote smoke is required')
        smoke = [json.loads(line) for line in a.vote_smoke.read_text().splitlines()]
        if (len(smoke) != 20 or [x['sample_index'] for x in smoke] != list(range(20))
                or any(x.get('labels_read') is not False or len(x.get('students', [])) != 4
                       for x in smoke)):
            raise ValueError('Incomplete vote smoke')
        from core.multimodal.datasets import load_local_dataset
        from core.multimodal.schema import GenerationResponse, SamplingConfig
        from core.multimodal.tf_ttcl_direct_selection import select_closed
        from scripts.hygon_scienceqa_vote import candidate
        from dataclasses import asdict
        examples = load_local_dataset(str(a.inference), 'scienceqa_img', str(a.inference.parent))
        greeds = [json.loads(line) for line in a.greedy.read_text().splitlines()]
        refs = [json.loads(line) for line in a.reference.read_text().splitlines()]
        for index, row in enumerate(smoke):
            sample = examples[index].sample
            if (row['sample_id'] != sample.sample_id or row['teacher'] != greeds[index]
                    or row['image_hashes'] != refs[index]['image_hashes']):
                raise ValueError(f'Vote smoke alignment mismatch: {index}')
            candidates = [candidate('teacher', GenerationResponse(**row['teacher']['response']), sample)]
            for n, student in enumerate(row['students'], 1):
                role = f'student_{n}'
                expected = SamplingConfig(temperature=.7, top_p=.9, max_tokens=256,
                                          seed=100 * index + n + 1, request_logprobs=True)
                scored = candidate(role, GenerationResponse(**student['response']), sample)
                if (student['role'] != role or student['sampling'] != asdict(expected)
                        or student['answer'] != scored.answer or abs(student['ppl'] - scored.ppl) > 1e-9):
                    raise ValueError(f'Vote smoke candidate mismatch: {index}/{n}')
                candidates.append(scored)
            selection = select_closed(candidates)
            answer = '' if selection.positive.answer.startswith('__unparsed_') else selection.positive.answer
            if (row['answer'], row['positive_role'], row['reason']) != (
                    answer, selection.positive.role, selection.reason):
                raise ValueError(f'Vote smoke selection mismatch: {index}')
    for port in range(a.base_port, a.base_port + 8):
        free_port(port)
    os.chdir(a.project)
    worker = str(a.project / 'scripts/hygon_scienceqa_vote.py')
    common = [sys.executable, worker, '--inference', str(a.inference), '--reference', str(a.reference),
              '--greedy', str(a.greedy), '--count', str(a.count), '--workers', '8']
    for index in range(8):
        subprocess.run(common + ['--worker', str(index), '--output', str(a.output / 'unused'),
                                 '--base-url', f'http://127.0.0.1:{a.base_port + index}/v1',
                                 '--check-only'], check=True)
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / 'run_config.json').write_text(json.dumps({
        'devices': devices, 'count': a.count, 'labels_read': False, 'seed': 0,
        'inference_sha256': digest(a.inference), 'reference_sha256': digest(a.reference),
        'greedy_sha256': digest(a.greedy), 'weight_gate_sha256': digest(a.weight_manifest),
        'vote_smoke_sha256': digest(a.vote_smoke) if a.vote_smoke else None,
        'worker_sha256': digest(Path(worker)), 'launcher_sha256': digest(Path(__file__)),
        'software_versions': {name: importlib.metadata.version(name)
                              for name in ('torch', 'vllm', 'transformers')},
        'max_model_len': 8192, 'gpu_memory_utilization': .75,
    }, indent=2) + '\n')
    services, workers, logs = [], [], []
    def interrupt(signum, _frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        for index, device in enumerate(devices):
            log = (a.output / f'vllm_{index}.log').open('x')
            logs.append(log)
            env = os.environ.copy()
            env.update(HIP_VISIBLE_DEVICES=device, VLLM_USE_FLASHINFER_SAMPLER='0')
            cmd = ['vllm', 'serve', str(a.model), '--served-model-name', 'Qwen/Qwen2.5-VL-7B-Instruct',
                   '--host', '127.0.0.1', '--port', str(a.base_port + index), '--trust-remote-code',
                   '--gpu-memory-utilization', '0.75', '--max-model-len', '8192']
            services.append(subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                             start_new_session=True))
        deadline = time.monotonic() + a.ready_timeout
        waiting = set(range(8))
        while waiting:
            for i, proc in enumerate(services):
                if proc.poll() is not None:
                    raise RuntimeError(f'Service {i} exited: {proc.returncode}')
            waiting = {i for i in waiting if not health(a.base_port + i)}
            if time.monotonic() >= deadline and waiting:
                raise TimeoutError(f'Services not ready: {waiting}')
            if waiting:
                time.sleep(2)
        for i in range(8):
            log = (a.output / f'worker_{i}.log').open('x')
            logs.append(log)
            cmd = common + ['--worker', str(i), '--output', str(a.output / f'predictions_{i}.jsonl'),
                            '--base-url', f'http://127.0.0.1:{a.base_port + i}/v1']
            workers.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                             start_new_session=True))
        while any(proc.poll() is None for proc in workers):
            if any(proc.poll() is not None for proc in services):
                raise RuntimeError('A service exited during inference')
            if any(proc.poll() not in (None, 0) for proc in workers):
                raise RuntimeError('A worker failed; inspect worker logs')
            time.sleep(2)
        if any(proc.returncode != 0 for proc in workers):
            raise RuntimeError('At least one worker failed')
        rows = []
        for i in range(8):
            shard = [json.loads(line) for line in (a.output / f'predictions_{i}.jsonl').read_text().splitlines()]
            if ([x['sample_index'] for x in shard] != list(range(i, a.count, 8))
                    or any(x.get('labels_read') is not False or len(x.get('students', [])) != 4 for x in shard)):
                raise ValueError(f'Incomplete or invalid shard {i}')
            rows.extend(shard)
        rows.sort(key=lambda row: row['sample_index'])
        with (a.output / 'predictions.jsonl').open('x') as out:
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + '\n')
        (a.output / 'completed.json').write_text(json.dumps({
            'status': 'completed', 'processed': len(rows), 'labels_read': False,
            'predictions_sha256': digest(a.output / 'predictions.jsonl')}, indent=2) + '\n')
        print(f'Completed {len(rows)} label-free vote predictions', flush=True)
    finally:
        cleanup(workers + services)
        for log in logs:
            log.close()


if __name__ == '__main__':
    main()
