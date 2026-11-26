#!/usr/bin/env python3
"""Independently recompute both selection strategies without opening labels."""
import argparse
import json
import hashlib
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.multimodal.datasets import load_local_dataset
from core.multimodal.answers import normalize_answer
from core.multimodal.image_ops import sha256_file
from core.multimodal.prompts import format_question
from core.multimodal.textvqa_selector_online import SELECTORS, VERSION, CachedEmbedder, choose, recorded_candidates
from core.multimodal.tf_ttcl_direct_runner import (DirectRunner, TextRule, retrieve_rules,
    TEACHER_SYSTEM_OPEN, STUDENT_SYSTEM_OPEN)
from core.multimodal.textvqa_format_factorial import SHORT_FINAL_SUFFIX
from scripts.audit_tf_ttcl_mm_direct import audit
from scripts.compare_textvqa_format_memory import policy_prompt_hash, verify_online_and_sampling


def same_selection(record, selection):
    expected = DirectRunner._selection_record(selection)
    for key, value in expected.items():
        observed = record.get(key)
        if key == 'similarity_ranking':
            if not isinstance(observed, (tuple, list)) or len(observed) != 4:
                raise ValueError('Missing similarity ranking')
            if any(a[0] != b[0] or not math.isclose(a[1], b[1], abs_tol=1e-6)
                   for a, b in zip(observed, value)):
                raise ValueError('Recomputed similarity ranking mismatch')
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            if not isinstance(observed, (int, float)) or not math.isclose(observed, value, rel_tol=1e-7):
                raise ValueError('Selection PPL mismatch')
        elif isinstance(value, tuple):
            if list(observed or []) != list(value):
                raise ValueError('Selected group mismatch')
        elif observed != value:
            raise ValueError('Selected candidate mismatch')


def audit_selector(output, inference, count, embedder):
    if not isinstance(embedder, CachedEmbedder): embedder = CachedEmbedder(embedder)
    manifest, rows = audit(output, inference, count)
    selector = manifest.get('similarity_selector')
    if (selector not in SELECTORS or manifest.get('selector_experiment') != VERSION
            or manifest.get('format_policy') != 'short_final'
            or manifest.get('prompt_sha256') != policy_prompt_hash('short_final')):
        raise ValueError('Selector policy or frozen prompt mismatch')
    verify_online_and_sampling(rows, manifest['seed'], 'short_final')
    examples = load_local_dataset(str(inference), 'textvqa', str(inference.parent), limit=count)
    if len(examples) != count or any(x.target is not None for x in examples):
        raise ValueError('Inference data is incomplete or exposes targets')
    history = []
    trace_dir = output / 'response_trace'
    first = next(trace_dir.glob('*.json'), None)
    if first is None:
        raise ValueError('Missing immutable request traces')
    namespace = json.loads(first.read_text())['request']['namespace']
    for key, value in {'dataset': 'textvqa', 'selector': selector, 'count': count,
                       'input_sha256': manifest['input_sha256'], 'seed': manifest['seed'],
                       'model_revision': manifest['model_revision'],
                       'embedding_sha256': manifest['embedding_revision'],
                       'labels_read': False}.items():
        if namespace.get(key) != value: raise ValueError('Trace namespace mismatch')
    if (not namespace.get('source_fingerprint') or not namespace.get('stack_id') or
            manifest.get('git_sha') != 'archive:' + namespace['source_fingerprint'] + ':' + namespace['stack_id']):
        raise ValueError('Trace source or stack identity mismatch')
    for index, (row, example) in enumerate(zip(rows, examples)):
        sample = example.sample
        if (sample.sample_id != row['sample_id'] or sample.question != row['original_question'] or
                list(sample.choices) != row['choices'] or
                [sha256_file(p) for p in sample.image_paths] != row['image_hashes']):
            raise ValueError('Input question or actual image mismatch')
        expected = retrieve_rules(history, embedder.encode(sample.question, query=True), index) if history else {'positive': [], 'negative': []}
        for kind in expected:
            actual = row['retrieved_rules'][kind]
            if len(actual) != len(expected[kind]):
                raise ValueError('Retrieved rule count mismatch')
            for a, b in zip(actual, expected[kind]):
                if (a['rule_id'], a['sample_index'], a['text']) != (b['rule_id'], b['sample_index'], b['text']) or not math.isclose(a['similarity'], b['similarity'], abs_tol=1e-6):
                    raise ValueError('Retrieved rule or rank mismatch')
        for name in ('vote_no_rewrite', 'select_no_memory', 'tf_ttcl_mm_direct'):
            arm = row['arms'][name]
            calls = [c for c in arm['calls'] if c.get('role')]
            candidates = recorded_candidates(arm)
            for n, call in enumerate(calls):
                role = 'teacher' if n == 0 else f'student_{n}'
                if call['role'] != role:
                    raise ValueError('Candidate role/order mismatch')
                question = sample.question if n == 0 or name == 'vote_no_rewrite' else row['variants'][n-1]
                if call['question_variant'] != question or call['user'] != format_question(question, sample.choices, sample.metadata):
                    raise ValueError('Candidate question mismatch')
                parsed = normalize_answer(call['response']['content'], 'textvqa',
                                          {**sample.metadata, 'choices': sample.choices}) or f'__unparsed_{role}__'
                if parsed != call['normalized_answer']:
                    raise ValueError('Candidate answer extraction mismatch')
                if name == 'vote_no_rewrite':
                    system = (TEACHER_SYSTEM_OPEN if n == 0 else STUDENT_SYSTEM_OPEN) + SHORT_FINAL_SUFFIX
                    if call['system'] != system:
                        raise ValueError('No-rewrite prompt mismatch')
            same_selection(arm, choose(candidates, embedder, selector))
            answer_scores = dict(choose(candidates, embedder, 'normalized_answer').similarity_ranking)
            stored_scores = arm['open_answer_diagnostic']['answer_similarity']
            if set(answer_scores) != set(stored_scores) or any(not math.isclose(answer_scores[k], stored_scores[k], abs_tol=1e-6) for k in answer_scores):
                raise ValueError('Answer-similarity diagnostic mismatch')
        for arm in row['arms'].values():
            for call in arm['calls']:
                sampling = dict(call['sampling'])
                if sampling.get('output_schema') is None: sampling.pop('output_schema', None)
                request = {'namespace': namespace, 'model': manifest['model'], 'system_prompt': call['system'],
                           'user_text': call['user'], 'image_hashes': call['image_hashes'], 'sampling': sampling}
                key = hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                trace = json.loads((trace_dir / (key + '.json')).read_text())
                if trace != {'request': request, 'response': call['response']}:
                    raise ValueError('Original request trace mismatch')
        candidates = recorded_candidates(row['arms']['select_no_memory'])
        if set(row.get('fixed_no_memory_selection', {})) != set(SELECTORS):
            raise ValueError('Missing fixed-candidate comparison')
        for name in SELECTORS:
            same_selection(row['fixed_no_memory_selection'][name], choose(candidates, embedder, name))
        summaries = {c['kind']: c for c in row['arms']['tf_ttcl_mm_direct']['calls'] if c.get('kind')}
        expected_writes = {kind for kind, c in summaries.items() if re.sub(
            r'^(?:Positive|Negative) Rule:\s*', '', c['response']['content'].strip(), flags=re.I)}
        if len(row['written_rules']) != len(expected_writes) or {x['kind'] for x in row['written_rules']} != expected_writes:
            raise ValueError('Missing or duplicate rule writes')
        for item in row['written_rules']:
            kind = item['kind']
            if kind not in summaries or item['sample_index'] != index or item['rule_id'] != f'{kind}_{index}':
                raise ValueError('Rule provenance mismatch')
            raw = summaries[kind]['response']['content'].strip()
            text = re.sub(r'^(?:Positive|Negative) Rule:\s*', '', raw, flags=re.I)
            if item['text'] != text:
                raise ValueError('Rule differs from summary output')
            vec = embedder.encode(text)
            if len(vec) != len(item['vector']) or any(not math.isclose(a, b, abs_tol=1e-6) for a, b in zip(vec, item['vector'])):
                raise ValueError('Rule vector differs from its text')
            history.append(TextRule.from_dict(item))
    return manifest, rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--inference', required=True, type=Path)
    p.add_argument('--count', required=True, type=int)
    p.add_argument('--embedding-path', required=True, type=Path)
    args = p.parse_args()
    from core.multimodal.tf_ttcl_text_embedding import Qwen3TextEmbedder
    manifest, rows = audit_selector(args.output, args.inference, args.count, Qwen3TextEmbedder(str(args.embedding_path)))
    print(json.dumps({'status': 'passed', 'selector': manifest['similarity_selector'], 'count': len(rows), 'labels_read': False}))


if __name__ == '__main__':
    main()
