#!/usr/bin/env python3
"""Score three independently audited online rule-control streams (400 only).

The streams have distinct evolving memories; matched_random matches targeted
counts only within its own history. Targets are opened after all audits.
"""
import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.multimodal.answers import extract_final_answer
from core.multimodal.datasets import _extract_answers
from core.multimodal.textvqa_selector_online import CachedEmbedder
from core.multimodal.textvqa_rule_control_online import POLICIES
from scripts.audit_textvqa_rule_control_online import audit_rule_control
from scripts.compare_textvqa_format_memory import CONFIG_KEYS, contrast, format_ok, selected_response
from scripts.create_label_isolated_subset import sha256_file
from scripts.rescore_textvqa_mmf import load_evaluator, score_one, MMF_REVISION, MMF_SHA256
from scripts.run_hygon_scienceqa_parallel import verify_weights
from scripts.run_hygon_tf_ttcl_direct import EMBEDDING_FILES
from scripts.run_hygon_textvqa_rule_control_online import load_rule_fresh, control_source_fingerprint


def score_records(streams, targets, evaluator):
    if set(streams) != set(POLICIES) or not targets or any(len(rows) != len(targets) for rows in streams.values()):
        raise ValueError('Missing policy or unequal stream and label counts')
    texts = {f'{policy}_{arm}': [] for policy in POLICIES for arm in ('greedy', 'no_memory', 'memory')}
    for index in range(len(targets)):
        ids = {(rows[index]['sample_index'], rows[index]['sample_id']) for rows in streams.values()}
        if len(ids) != 1 or next(iter(ids))[0] != index:
            raise ValueError('Paired stream order or IDs differ')
        for policy, rows in streams.items():
            arms = rows[index]['arms']
            texts[f'{policy}_greedy'].append(arms['greedy']['calls'][0]['response']['content'])
            texts[f'{policy}_no_memory'].append(selected_response(arms['select_no_memory']))
            texts[f'{policy}_memory'].append(selected_response(arms['tf_ttcl_mm_direct']))
    vals = {key: [score_one(evaluator, extract_final_answer(text), target)
                  for text, target in zip(responses, targets)] for key, responses in texts.items()}
    summaries = {key: {'mean': sum(scores)/len(scores),
        'selected_output_format_rate': sum(format_ok(x) for x in texts[key])/len(scores),
        'missing_final_answer_marker': sum(not bool(re.search(r'(?i)final answer\s*:', x)) for x in texts[key]),
        'mean_extracted_answer_words': sum(len(extract_final_answer(x).split()) for x in texts[key])/len(scores)}
        for key, scores in vals.items()}
    pairs = {'targeted_minus_original_memory': ('original_memory', 'targeted_memory'),
             'targeted_minus_matched_random_memory': ('matched_random_memory', 'targeted_memory'),
             'matched_random_minus_original_memory': ('original_memory', 'matched_random_memory')}
    for policy in POLICIES:
        pairs[f'{policy}_memory_minus_no_memory'] = (f'{policy}_no_memory', f'{policy}_memory')
        pairs[f'{policy}_memory_minus_greedy'] = (f'{policy}_greedy', f'{policy}_memory')
    comparisons = {}
    for key, (old, new) in pairs.items():
        delta = [b-a for a, b in zip(vals[old], vals[new])]
        comparisons[key] = {**contrast(delta), 'improved_items': sum(x > 1e-12 for x in delta),
                            'worsened_items': sum(x < -1e-12 for x in delta),
                            'unchanged_items': sum(abs(x) <= 1e-12 for x in delta)}
    segments = {}
    for policy in POLICIES:
        nomem, memory = vals[f'{policy}_no_memory'], vals[f'{policy}_memory']
        parts = []
        for start in range(0, len(targets), 100):
            stop = min(start + 100, len(targets))
            a, b = nomem[start:stop], memory[start:stop]
            diff = [y-x for x, y in zip(a, b)]
            parts.append({'start_index': start, 'end_index_exclusive': stop,
                'count': stop-start, 'no_memory_mean': sum(a)/len(a),
                'memory_mean': sum(b)/len(b), 'memory_minus_no_memory': sum(diff)/len(diff),
                'improved_items': sum(x > 1e-12 for x in diff),
                'worsened_items': sum(x < -1e-12 for x in diff),
                'unchanged_items': sum(abs(x) <= 1e-12 for x in diff)})
        segments[policy] = parts
    return summaries, comparisons, segments


def gate_summary(rows):
    result = {bucket: {'retrieved': 0, 'targeted_matches': 0, 'replacements': 0,
                       'samples_with_matches': 0, 'samples_with_replacements': 0,
                       'real_tokens_replaced': 0, 'replacement_tokens': 0}
              for bucket in ('positive', 'negative')}
    any_match = any_replacement = 0
    for row in rows:
        gate = row['rule_gate']
        any_match += any(gate['targeted_ids_by_bucket'][bucket] for bucket in result)
        any_replacement += bool(gate['replacements'])
        for bucket in result:
            result[bucket]['retrieved'] += len(row['retrieved_rules'][bucket])
            result[bucket]['targeted_matches'] += len(gate['targeted_ids_by_bucket'][bucket])
            result[bucket]['samples_with_matches'] += bool(gate['targeted_ids_by_bucket'][bucket])
            result[bucket]['samples_with_replacements'] += any(x['bucket'] == bucket for x in gate['replacements'])
        for item in gate['replacements']:
            bucket = item['bucket']
            if bucket not in result:
                raise ValueError('Unknown rule bucket')
            result[bucket]['replacements'] += 1
            result[bucket]['real_tokens_replaced'] += item['real_tokens']
            result[bucket]['replacement_tokens'] += item['replacement_tokens']
    return {'by_bucket': result, 'samples_with_any_match': any_match,
            'samples_with_any_replacement': any_replacement}


def trace_cost(output, manifest, fingerprint, weight_sha):
    paths = list((output / 'response_trace').glob('*.json'))
    if not paths: raise ValueError('Missing physical traces')
    totals = {'stored_unique_requests': 0, 'prompt_tokens': 0, 'completion_tokens': 0, 'latency_seconds': 0.0}
    for path in paths:
        item = json.loads(path.read_text())
        if path.stem != hashlib.sha256(json.dumps(item['request'], sort_keys=True, ensure_ascii=False).encode()).hexdigest():
            raise ValueError('Physical trace request hash mismatch')
        ns, response = item['request']['namespace'], item['response']
        if (ns.get('rule_control_policy'), ns.get('input_sha256'), ns.get('seed'), ns.get('source_fingerprint'),
                ns.get('weight_manifest_sha256')) != (manifest['rule_control_policy'], manifest['input_sha256'],
                manifest['seed'], fingerprint, weight_sha):
            raise ValueError('Physical trace namespace mismatch')
        for key in ('prompt_tokens', 'completion_tokens', 'latency_seconds'):
            value = response[key]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError('Invalid physical trace usage')
            totals[key] += value
        totals['stored_unique_requests'] += 1
    return totals


def score(a, embedder, tokenizer, evaluator):
    if a.count != 400: raise ValueError('Smoke runs cannot be scored')
    _, data = load_rule_fresh(a.inference, a.subset_manifest, a.subset_manifest_sha256, a.excluded_inference, 400)
    shared = embedder if isinstance(embedder, CachedEmbedder) else CachedEmbedder(embedder)
    fingerprint = control_source_fingerprint(Path(__file__).resolve().parents[1])
    weight_sha = sha256_file(a.weight_manifest)
    manifests, streams = {}, {}
    for policy in POLICIES:
        manifests[policy], streams[policy] = audit_rule_control(getattr(a, policy), a.inference, 400, shared, tokenizer,
            expected_fingerprint=fingerprint, expected_weight_sha=weight_sha)
        if manifests[policy].get('rule_control_policy') != policy:
            raise ValueError('Wrong policy stream')
    original = manifests['original']
    for policy, manifest in manifests.items():
        for key in (*CONFIG_KEYS, 'rule_control_experiment', 'rule_gate_version', 'similarity_selector'):
            if key not in original or manifest.get(key) != original[key]:
                raise ValueError(f'Different paired configuration: {policy}/{key}')
        if [r['sample_id'] for r in streams[policy]] != [r['sample_id'] for r in streams['original']]:
            raise ValueError('Paired stream order differs')
    costs, physical, gates = {}, {}, {}
    for policy, rows in streams.items():
        costs[policy] = {arm: {key: sum(r['arms'][arm]['usage'][key] for r in rows)
            for key in ('prompt_tokens', 'completion_tokens', 'latency_seconds')}
            for arm in ('greedy', 'select_no_memory', 'tf_ttcl_mm_direct')}
        physical[policy] = trace_cost(getattr(a, policy), manifests[policy], fingerprint, weight_sha)
        gates[policy] = gate_summary(rows)
    # Targets are first opened here, after all streams and artifact identities pass.
    raw = a.labels.read_bytes()
    if hashlib.sha256(raw).hexdigest() != data['labels_sha256']: raise ValueError('Label checksum mismatch')
    labels = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if [str(x['id']) for x in labels] != [r['sample_id'] for r in streams['original']]:
        raise ValueError('Label ID mismatch')
    targets = [_extract_answers(row, row.get('choices', row.get('options', []))) for row in labels]
    metrics, comparisons, segments = score_records(streams, targets, evaluator)
    budget = {}
    for old_policy, policy in (('original', 'targeted'), ('original', 'matched_random'),
                               ('matched_random', 'targeted')):
        for arm in ('select_no_memory', 'tf_ttcl_mm_direct'):
            old = sum(costs[old_policy][arm][k] for k in ('prompt_tokens', 'completion_tokens'))
            new = sum(costs[policy][arm][k] for k in ('prompt_tokens', 'completion_tokens'))
            budget[f'{policy}_{arm}_vs_{old_policy}'] = {'ratio': new/old if old else None,
                'within_5_percent': abs(new-old) <= 0.05*old if old else False}
    return {'protocol': 'textvqa_rule_control_online_v1', 'n': 400, 'metrics': metrics, 'comparisons': comparisons,
            'stream_segments_100': segments,
            'gate_by_policy_and_bucket': gates, 'cost_logical_per_arm': costs,
            'cost_physical_stored_unique_per_stream': physical, 'cost_matching': budget,
            'audits': {p: {'status': 'passed', 'count': len(streams[p]), 'labels_read': False,
                           'predictions_sha256': sha256_file(getattr(a, p) / 'predictions.jsonl'),
                           'config_sha256': manifests[p]['config_sha256']} for p in POLICIES},
            'input_sha256': sha256_file(a.inference), 'labels_sha256': data['labels_sha256'],
            'upstream_revision': MMF_REVISION, 'upstream_sha256': MMF_SHA256,
            'note': 'Fresh developmental data, three independent online histories and potentially different devices. '
                    'Matched-random matches targeted replacement counts per bucket only within its own evolving history; '
                    'replacement totals need not match across streams. '
                    'Pinned Meta MMF, EvalAI parity unverified. Physical totals exclude failed unrecorded attempts.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('original', 'targeted', 'matched-random', 'inference', 'subset-manifest', 'labels',
                 'embedding-path', 'model-path', 'weight-manifest', 'mmf-evaluator', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--subset-manifest-sha256', required=True)
    p.add_argument('--excluded-inference', type=Path, action='append', required=True)
    p.add_argument('--count', type=int, default=400)
    a = p.parse_args()
    if a.count != 400: raise ValueError('Smoke runs cannot be scored')
    if a.output.exists(): raise FileExistsError(a.output)
    for name, expected in EMBEDDING_FILES.items():
        if sha256_file(a.embedding_path / name) != expected: raise ValueError('Embedding hash mismatch')
    verify_weights(a.model_path, a.weight_manifest)
    evaluator = load_evaluator(a.mmf_evaluator)
    from transformers import AutoTokenizer
    from core.multimodal.tf_ttcl_text_embedding import Qwen3TextEmbedder
    tokenizer = AutoTokenizer.from_pretrained(str(a.model_path), local_files_only=True)
    result = score(a, Qwen3TextEmbedder(str(a.embedding_path)), tokenizer, evaluator)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2); handle.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
