#!/usr/bin/env python3
"""Offline MMF comparison after auditing both complete independent streams."""
import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.multimodal.answers import extract_final_answer
from core.multimodal.textvqa_selector_online import CachedEmbedder
from core.multimodal.datasets import _extract_answers
from scripts.audit_textvqa_selector_online import audit_selector
from scripts.compare_textvqa_format_memory import CONFIG_KEYS, contrast, selected_response, format_ok
from scripts.create_label_isolated_subset import sha256_file
from scripts.rescore_textvqa_mmf import load_evaluator, score_one, MMF_REVISION, MMF_SHA256
from scripts.run_hygon_textvqa_selector_online import load_fresh


def response_for_role(arm, role):
    calls = [c for c in arm['calls'] if c.get('role') == role]
    if len(calls) != 1: raise ValueError('Fixed candidate role mismatch')
    return calls[0]['response']['content']


def score_records(full, answer, targets, evaluator):
    if len(full) != len(answer) or len(full) != len(targets) or not targets:
        raise ValueError('Paired stream length mismatch')
    texts = {key: [] for key in ('greedy', 'full_no_memory', 'answer_no_memory',
        'full_memory', 'answer_memory', 'fixed_full_source_full', 'fixed_full_source_answer')}
    for f, a in zip(full, answer):
        if (f['sample_index'], f['sample_id']) != (a['sample_index'], a['sample_id']):
            raise ValueError('Paired sample identity mismatch')
        fa, aa = f['arms'], a['arms']
        texts['greedy'].append(fa['greedy']['calls'][0]['response']['content'])
        for key, arm in [('full_no_memory', fa['select_no_memory']), ('answer_no_memory', aa['select_no_memory']),
                         ('full_memory', fa['tf_ttcl_mm_direct']), ('answer_memory', aa['tf_ttcl_mm_direct'])]:
            texts[key].append(selected_response(arm))
        for suffix, selector in [('full', 'full_response'), ('answer', 'normalized_answer')]:
            role = f['fixed_no_memory_selection'][selector]['positive_role']
            texts['fixed_full_source_' + suffix].append(response_for_role(fa['select_no_memory'], role))
    scores = {key: [score_one(evaluator, extract_final_answer(text), target) for text, target in zip(values, targets)]
              for key, values in texts.items()}
    summary = {key: {'mean': sum(vals)/len(vals),
                     'selected_output_format_rate': sum(format_ok(x) for x in texts[key])/len(vals),
                     'missing_final_answer_marker': sum(not bool(re.search(r'(?i)final answer\s*:', x)) for x in texts[key]),
                     'mean_extracted_answer_words': sum(len(extract_final_answer(x).split()) for x in texts[key])/len(vals)}
               for key, vals in scores.items()}
    pairs = {'fixed_candidate_answer_minus_full': ('fixed_full_source_full', 'fixed_full_source_answer'),
             'separate_stream_answer_minus_full_no_memory': ('full_no_memory', 'answer_no_memory'),
             'online_answer_minus_full_memory': ('full_memory', 'answer_memory'),
             'full_memory_minus_no_memory': ('full_no_memory', 'full_memory'),
             'answer_memory_minus_no_memory': ('answer_no_memory', 'answer_memory'),
             'answer_memory_minus_greedy': ('greedy', 'answer_memory')}
    comparisons = {}
    for name, (old, new) in pairs.items():
        diff = [b-a for a,b in zip(scores[old], scores[new])]
        comparisons[name] = {**contrast(diff), 'improved_items': sum(x > 1e-12 for x in diff),
                             'worsened_items': sum(x < -1e-12 for x in diff)}
    return summary, comparisons


def stored_trace_cost(output, manifest):
    paths = list((output / 'response_trace').glob('*.json'))
    if not paths: raise ValueError('Missing physical traces')
    totals = {'stored_unique_requests': 0, 'prompt_tokens': 0, 'completion_tokens': 0, 'latency_seconds': 0.0}
    for path in paths:
        item = json.loads(path.read_text())
        namespace, response = item['request']['namespace'], item['response']
        if (namespace.get('selector'), namespace.get('input_sha256'), namespace.get('seed')) != (
                manifest['similarity_selector'], manifest['input_sha256'], manifest['seed']):
            raise ValueError('Physical trace namespace mismatch')
        for key in ('prompt_tokens', 'completion_tokens', 'latency_seconds'):
            value = response[key]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError('Invalid physical trace usage')
            totals[key] += value
        totals['stored_unique_requests'] += 1
    return totals


def score(a, embedder, evaluator):
    if a.count != 400: raise ValueError('Smoke runs cannot be scored')
    embedder = CachedEmbedder(embedder)
    _, data = load_fresh(a.inference, a.subset_manifest, a.subset_manifest_sha256, a.excluded_inference, 400)
    fm, fr = audit_selector(a.full, a.inference, 400, embedder)
    am, ar = audit_selector(a.answer, a.inference, 400, embedder)
    if (fm['similarity_selector'], am['similarity_selector']) != ('full_response', 'normalized_answer'):
        raise ValueError('Wrong selector streams')
    for key in CONFIG_KEYS:
        if key not in fm or fm[key] != am.get(key): raise ValueError('Different paired configuration: ' + key)
    if [r['sample_id'] for r in fr] != [r['sample_id'] for r in ar]:
        raise ValueError('Stream order mismatch')
    physical = {'full_response': stored_trace_cost(a.full, fm),
                'normalized_answer': stored_trace_cost(a.answer, am)}
    # No labels are opened until both independent selector audits pass.
    raw = a.labels.read_bytes()
    if hashlib.sha256(raw).hexdigest() != data['labels_sha256']: raise ValueError('Label checksum mismatch')
    labels = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if [str(x['id']) for x in labels] != [r['sample_id'] for r in fr]: raise ValueError('Label ID mismatch')
    targets = [_extract_answers(row, row.get('choices', row.get('options', []))) for row in labels]
    summary, comparisons = score_records(fr, ar, targets, evaluator)
    costs = {}
    for name, rows in [('full', fr), ('answer', ar)]:
        costs[name] = {arm: {key: sum(r['arms'][arm]['usage'][key] for r in rows)
                      for key in ('prompt_tokens', 'completion_tokens', 'latency_seconds')}
                      for arm in ('greedy', 'select_no_memory', 'tf_ttcl_mm_direct')}
    budget = {}
    for arm in ('select_no_memory', 'tf_ttcl_mm_direct'):
        old = sum(costs['full'][arm][k] for k in ('prompt_tokens', 'completion_tokens'))
        new = sum(costs['answer'][arm][k] for k in ('prompt_tokens', 'completion_tokens'))
        budget[arm] = {'answer_over_full': new / old if old else None,
                       'within_5_percent': abs(new-old) <= 0.05*old if old else False}
    return {'protocol': 'textvqa_selector_online_v1', 'n': 400, 'metrics': summary, 'comparisons': comparisons,
            'audits': {'full_response': {'status': 'passed', 'count': len(fr), 'labels_read': False,
                                        'predictions_sha256': sha256_file(a.full / 'predictions.jsonl'),
                                        'config_sha256': fm['config_sha256']},
                       'normalized_answer': {'status': 'passed', 'count': len(ar), 'labels_read': False,
                                             'predictions_sha256': sha256_file(a.answer / 'predictions.jsonl'),
                                             'config_sha256': am['config_sha256']}},
            'cost_logical_per_arm': costs, 'cost_physical_stored_unique_per_stream': physical,
            'cost_matching': budget,
            'cost_note': 'Logical arms may share calls; physical totals count stored unique requests '
                         'within each isolated stream, including stored interrupted attempts, '
                         'but exclude failed attempts with no stored response.',
            'input_sha256': sha256_file(a.inference),
            'labels_sha256': data['labels_sha256'], 'upstream_revision': MMF_REVISION, 'upstream_sha256': MMF_SHA256,
            'note': 'One fresh developmental stream; labels used only after both audits. '
                    'Pinned Meta MMF evaluator; EvalAI server parity unverified. The two selector streams '
                    'have independent online rule memories and may use different physical devices; '
                    'device/selector confounding and temporal dependence limit causal interpretation.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('full', 'answer', 'inference', 'subset-manifest', 'labels', 'embedding-path', 'mmf-evaluator', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--subset-manifest-sha256', required=True)
    p.add_argument('--excluded-inference', type=Path, action='append', required=True)
    p.add_argument('--count', type=int, default=400)
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    from core.multimodal.tf_ttcl_text_embedding import Qwen3TextEmbedder
    from scripts.run_hygon_tf_ttcl_direct import EMBEDDING_FILES
    for name, expected in EMBEDDING_FILES.items():
        if sha256_file(a.embedding_path / name) != expected: raise ValueError('Embedding hash mismatch')
    result = score(a, Qwen3TextEmbedder(str(a.embedding_path)), load_evaluator(a.mmf_evaluator))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x', encoding='utf-8') as handle: json.dump(result, handle, indent=2); handle.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
