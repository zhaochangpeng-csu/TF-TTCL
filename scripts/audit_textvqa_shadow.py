#!/usr/bin/env python3
"""Audit completed fixed-history shadow shards without reading target labels.

Recomputes each full-response embedding, similarity partition, PPL and answer
from the recorded generation. Audited request traces allow measuring physical
and logical costs separately. This is an exploratory replay of a scored stream.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.multimodal.answers import normalize_answer
from core.multimodal.image_ops import sha256_file
from core.multimodal.prompts import format_question
from core.multimodal.schema import GenerationResponse, SamplingConfig
from core.multimodal.textvqa_shadow import ARMS, neutralize_rules, system_prompt
from core.multimodal.tf_ttcl_direct_runner import DirectRunner, _digest
from core.multimodal.tf_ttcl_direct_selection import ScoredCandidate, select_open
from core.multimodal.tf_ttcl_text_embedding import cosine
from scripts.run_hygon_textvqa_shadow import checked_anchor, atomic_json
from scripts.run_hygon_tf_ttcl_direct import EMBEDDING_FILES, EMBEDDING_SHA, MODEL_REVISION, source_fingerprint
from scripts.run_hygon_scienceqa_parallel import verify_weights


def audit_item(row, sample, anchor, index, config, tokenizer, embedder, trace_dir, vectors):
    if (row.get('config_sha256'), row.get('sample_index'), row.get('sample_id'),
        row.get('labels_read'), row.get('rules_written')) != (
            _digest(config), index, sample.sample_id, False, False):
        raise ValueError(f'{index}: invalid committed identity')
    if set(row.get('arms', {})) != set(ARMS):
        raise ValueError(f'{index}: incorrect arm set')
    if (sample.question != anchor['original_question'] or
        row.get('image_hashes') != anchor['image_hashes'] or
        [sha256_file(p) for p in sample.image_paths] != anchor['image_hashes'] or
        row.get('variants') != anchor['variants'] or
        row.get('retrieved_rules') != anchor['retrieved_rules'] or
        row.get('anchor_no_memory') != anchor['arms']['select_no_memory']['answer'] or
        row.get('anchor_memory') != anchor['arms']['tf_ttcl_mm_direct']['answer']):
        raise ValueError(f'{index}: anchor question/image/history differs')
    for kind, bucket in row['retrieved_rules'].items():
        if kind not in ('positive', 'negative') or any(rule['sample_index'] >= index for rule in bucket):
            raise ValueError(f'{index}: current/future rules')
    neutral, counts = neutralize_rules(anchor['retrieved_rules'], tokenizer)
    if row.get('neutral_rule_token_counts') != counts:
        raise ValueError(f'{index}: recorded placeholder token counts differ')
    for kind in ('positive', 'negative'):
        for original, substitute, count in zip(anchor['retrieved_rules'][kind], neutral[kind],
                                               [x for x in counts if x['kind'] == kind]):
            if (not re.fullmatch(r'(?:Reference note\.\s*)+', substitute['text']) or
                original['rule_id'] != substitute['rule_id'] or
                count['real_tokens'] != len(tokenizer.encode(original['text'], add_special_tokens=False)) or
                count['neutral_tokens'] != len(tokenizer.encode(substitute['text'], add_special_tokens=False))):
                raise ValueError(f'{index}: invalid neutral placeholder')
    logical, seen_traces = {}, {}
    for name in ARMS:
        content, position = name.split('_')
        bucket = anchor['retrieved_rules'] if content == 'real' else neutral
        record = row['arms'][name]
        calls = record.get('calls')
        if not isinstance(calls, list) or len(calls) != 5:
            raise ValueError(f'{index}/{name}: expected 5 candidate calls')
        candidates = []
        for n, (question, call) in enumerate(zip([sample.question, *anchor['variants']], calls)):
            role = 'teacher' if n == 0 else f'student_{n}'
            sampling = asdict(SamplingConfig(temperature=0.0 if n == 0 else 0.7,
                top_p=None if n == 0 else 0.9, max_tokens=256,
                seed=config['seed'] + 100 * index + (0 if n == 0 else n + 1), request_logprobs=True))
            system = system_prompt(role, bucket, position)
            user = format_question(question, sample.choices, sample.metadata)
            if (call.get('role'), call.get('question_variant'), call.get('system'), call.get('user'),
                call.get('image_paths'), call.get('image_hashes'), call.get('sampling')) != (
                    role, question, system, user, list(sample.image_paths), anchor['image_hashes'], sampling):
                raise ValueError(f'{index}/{name}/{role}: actual request differs')
            response = GenerationResponse(**call['response'])
            answer = normalize_answer(response.content, sample.dataset,
                                      {**sample.metadata, 'choices': sample.choices}) or f'__unparsed_{role}__'
            if call.get('normalized_answer') != answer:
                raise ValueError(f'{index}/{name}/{role}: answer parse mismatch')
            candidate = ScoredCandidate.from_response(role, answer, response)
            if not math.isclose(candidate.ppl, call['ppl'], rel_tol=1e-9):
                raise ValueError(f'{index}/{name}/{role}: PPL mismatch')
            candidates.append(candidate)
            trace_sampling = {k: v for k, v in sampling.items() if k != 'output_schema'}
            request = {'namespace': config, 'model': 'Qwen/Qwen2.5-VL-7B-Instruct',
                       'system_prompt': system, 'user_text': user,
                       'image_hashes': anchor['image_hashes'], 'sampling': trace_sampling}
            key = hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            trace = json.loads((trace_dir / f'{key}.json').read_text())
            if trace != {'request': request, 'response': call['response']}:
                raise ValueError(f'{index}/{name}/{role}: trace differs')
            seen_traces[key] = response
        def vector(text):
            if text not in vectors:
                vectors[text] = embedder.encode(text)
            return vectors[text]
        teacher = vector(candidates[0].response.content)
        similarities = [cosine(teacher, vector(c.response.content)) for c in candidates[1:]]
        expected = DirectRunner._selection_record(select_open(candidates, similarities))
        for key, value in expected.items():
            if key == 'similarity_ranking':
                observed = record[key]
                if len(observed) != 4 or any(a[0] != b[0] or not math.isclose(a[1], b[1], abs_tol=1e-6)
                                               for a, b in zip(observed, value)):
                    raise ValueError(f'{index}/{name}: similarity ranking mismatch')
            elif _digest(record[key]) != _digest(value):
                raise ValueError(f'{index}/{name}: selection mismatch in {key}')
        usage = DirectRunner._usage(calls)
        if _digest(record.get('usage')) != _digest(usage):
            raise ValueError(f'{index}/{name}: usage mismatch')
        logical[name] = usage
    return logical, seen_traces


def audit_shards(output, samples, rows, tokenizer, embedder, limit, shard_count):
    if not (1 <= limit <= 400 and 1 <= shard_count <= limit):
        raise ValueError('Invalid audit range')
    common, vectors, global_seen, row_hashes = None, {}, {}, {}
    logical = {arm: {'prompt_tokens': 0, 'completion_tokens': 0, 'latency_seconds': 0} for arm in ARMS}
    audit_rows = 0
    for shard in range(shard_count):
        root = output / f'shard-{shard:03d}-of-{shard_count:03d}'
        config = json.loads((root / 'config.json').read_text())
        if (config.get('version'), config.get('shard_index'), config.get('shard_count'),
            config.get('limit'), config.get('labels_read'), config.get('rules_written'),
            config.get('arms')) != ('textvqa_shadow_fixed_history_v1', shard, shard_count,
                                    limit, False, False, list(ARMS)):
            raise ValueError(f'Shard {shard}: bad configuration')
        common_config = {k: v for k, v in config.items() if k != 'shard_index'}
        if common is None:
            common = common_config
        elif common != common_config:
            raise ValueError('Shard configurations differ')
        expected_indices = list(range(shard, limit, shard_count))
        expected_files = {f'row-{i:06d}.json' for i in expected_indices}
        if {p.name for p in root.glob('row-*.json')} != expected_files:
            raise ValueError(f'Shard {shard}: missing or extraneous committed rows')
        shard_keys = set()
        for index in expected_indices:
            row_bytes = (root / f'row-{index:06d}.json').read_bytes()
            row = json.loads(row_bytes)
            usage, traces = audit_item(row, samples[index], rows[index], index, config,
                                       tokenizer, embedder, root / 'response_trace', vectors)
            for name in ARMS:
                for key, value in usage[name].items():
                    logical[name][key] += value
            shard_keys.update(traces)
            global_seen.update({(shard, key): response for key, response in traces.items()})
            audit_rows += 1
            row_hashes[str(index)] = hashlib.sha256(row_bytes).hexdigest()
        trace_names = {p.stem for p in (root / 'response_trace').glob('*.json')}
        if trace_names != shard_keys:
            raise ValueError(f'Shard {shard}: orphan or missing trace')
    return {'status': 'passed', 'labels_read': False, 'rules_written': False,
            'count': audit_rows, 'shard_count': shard_count, 'common_config': common,
            'row_sha256': row_hashes,
            'logical_per_arm': logical, 'physical_unique_requests': len(global_seen),
            'physical_unique_prompt_tokens': sum(r.prompt_tokens for r in global_seen.values()),
            'physical_unique_completion_tokens': sum(r.completion_tokens for r in global_seen.values()),
            'physical_unique_model_latency_seconds': sum(r.latency_seconds for r in global_seen.values()),
            'cost_note': 'Physical totals deduplicate stored traces per shard; interrupted attempts without a trace are not included.'}


def verify_report_identity(report, anchor_predictions_sha, anchor_manifest_sha, inference,
                           weight_manifest):
    config = report['common_config']
    if (config['anchor_predictions_sha256'], config['anchor_manifest_sha256'], config['input_sha256'],
        config['embedding_sha256'], config['weights_manifest_sha256'], config['model_revision'],
        config['seed'], config['source_fingerprint']) != (
        anchor_predictions_sha, anchor_manifest_sha, sha256_file(inference),
        EMBEDDING_SHA, sha256_file(weight_manifest), MODEL_REVISION, 0,
        _digest([source_fingerprint(Path(__file__).resolve().parents[1]),
                 sha256_file(Path(__file__).with_name('run_hygon_textvqa_shadow.py')),
                 sha256_file(Path(__file__).resolve().parents[1] / 'core/multimodal/textvqa_shadow.py')])):
        raise ValueError('Audit input or artifact hashes differ from shard configuration')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('anchor', 'inference', 'output', 'embedding-path', 'model-path', 'weight-manifest', 'report'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--anchor-predictions-sha256', required=True)
    parser.add_argument('--anchor-manifest-sha256', required=True)
    parser.add_argument('--limit', type=int, required=True)
    parser.add_argument('--shard-count', type=int, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    _, rows, samples = checked_anchor(args.anchor, args.inference,
                                     args.anchor_predictions_sha256, args.anchor_manifest_sha256)
    for name, expected in EMBEDDING_FILES.items():
        if sha256_file(args.embedding_path / name) != expected:
            raise ValueError('Embedding artifact mismatch')
    verify_weights(args.model_path, args.weight_manifest)
    from transformers import AutoTokenizer
    from core.multimodal.tf_ttcl_text_embedding import Qwen3TextEmbedder
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True)
    embedder = Qwen3TextEmbedder(str(args.embedding_path))
    report = audit_shards(args.output, samples, rows, tokenizer, embedder, args.limit, args.shard_count)
    verify_report_identity(report, args.anchor_predictions_sha256, args.anchor_manifest_sha256,
                           args.inference, args.weight_manifest)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
