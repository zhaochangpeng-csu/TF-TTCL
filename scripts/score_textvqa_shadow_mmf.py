#!/usr/bin/env python3
"""Score a fully audited 400-item fixed-history shadow replay with pinned MMF.

Exploratory: the frozen anchor stream was previously scored. The four shadows
change only an immediate response on that fixed history, not the online memory.
No targets are opened until all request/embedding/selection audits succeed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.multimodal.answers import extract_final_answer
from core.multimodal.datasets import _extract_answers
from core.multimodal.textvqa_shadow import ARMS
from scripts.audit_textvqa_shadow import audit_shards, verify_report_identity
from scripts.compare_textvqa_format_memory import contrast, format_ok, selected_response
from scripts.create_label_isolated_subset import sha256_file
from scripts.inspect_baseline_io import read_rows
from scripts.rescore_textvqa_mmf import MMF_REVISION, MMF_SHA256, MMF_URL, load_evaluator, score_one
from scripts.run_hygon_scienceqa_parallel import verify_weights
from scripts.run_hygon_textvqa_shadow import checked_anchor
from scripts.run_hygon_tf_ttcl_direct import EMBEDDING_FILES

ANCHORS = {'anchor_no_memory': 'select_no_memory', 'anchor_memory': 'tf_ttcl_mm_direct'}


def _selected_texts(anchor_rows, shadow_rows):
    if len(anchor_rows) != len(shadow_rows) or not anchor_rows:
        raise ValueError('Unequal or empty paired streams')
    result = {name: [] for name in (*ANCHORS, *ARMS)}
    for index, (anchor, shadow) in enumerate(zip(anchor_rows, shadow_rows)):
        if (anchor['sample_index'], anchor['sample_id']) != (index, shadow['sample_id']) or shadow['sample_index'] != index:
            raise ValueError('Paired predictions are not ordered by anchor ID')
        for output, arm in ANCHORS.items():
            result[output].append(selected_response(anchor['arms'][arm]))
        for arm in ARMS:
            result[arm].append(selected_response(shadow['arms'][arm]))
    return result


def _diff(old, new):
    values = [b-a for a,b in zip(old, new)]
    return {**contrast(values), 'improved_items': sum(x > 1e-12 for x in values),
            'worsened_items': sum(x < -1e-12 for x in values),
            'unchanged_items': sum(abs(x) <= 1e-12 for x in values)}


def score_records(texts, targets, evaluator):
    if set(texts) != {*ANCHORS, *ARMS} or not targets or any(len(v) != len(targets) for v in texts.values()):
        raise ValueError('Missing condition or target count mismatch')
    scores, results = {}, {}
    for name, responses in texts.items():
        answers = [extract_final_answer(text) for text in responses]
        vals = [score_one(evaluator, ans, target) for ans, target in zip(answers, targets)]
        scores[name] = vals
        results[name] = {'mmf_vqa_accuracy': sum(vals) / len(vals),
                         'mean_extracted_answer_words': sum(len(a.split()) for a in answers) / len(answers),
                         'selected_output_format_rate': sum(format_ok(t) for t in responses) / len(responses),
                         'missing_final_answer_marker': sum(not bool(re.search(r'(?i)final answer\s*:', t)) for t in responses)}
    rb, ra, nb, na = (scores[k] for k in ('real_before', 'real_after', 'neutral_before', 'neutral_after'))
    # Factor effects are within-item and averaged across the other factor.
    content = [(a+b-c-d)/2 for a,b,c,d in zip(rb,ra,nb,na)]
    position = [(b-a+d-c)/2 for a,b,c,d in zip(rb,ra,nb,na)]
    interaction = [(b-a)-(d-c) for a,b,c,d in zip(rb,ra,nb,na)]
    def vector(v):
        return {**contrast(v), 'positive_items': sum(x > 1e-12 for x in v),
                'negative_items': sum(x < -1e-12 for x in v),
                'zero_items': sum(abs(x) <= 1e-12 for x in v)}
    contrasts = {'real_minus_neutral_average': vector(content),
                 'after_minus_before_average': vector(position),
                 'interaction_real_vs_neutral_position': vector(interaction),
                 'real_minus_neutral_before': _diff(nb, rb),
                 'real_minus_neutral_after': _diff(na, ra),
                 'after_minus_before_real': _diff(rb, ra),
                 'after_minus_before_neutral': _diff(nb, na)}
    for arm in ARMS:
        for anchor in ANCHORS:
            contrasts[f'{arm}_minus_{anchor}'] = _diff(scores[anchor], scores[arm])
    return results, contrasts


def score_experiment(args, tokenizer, embedder, evaluator):
    # All these checks and all shadow audit work precede even opening labels.
    if args.limit != 400:
        raise ValueError('Only complete 400-item replay may be scored; smoke stays unscored')
    if not re.fullmatch(r'[0-9a-f]{64}', args.subset_manifest_sha256):
        raise ValueError('Need fixed subset manifest SHA')
    if len({p.resolve() for p in (args.inference, args.subset_manifest, args.labels, args.mmf_evaluator)}) != 4:
        raise ValueError('Input, manifest, labels and evaluator must be separate files')
    if sha256_file(args.subset_manifest) != args.subset_manifest_sha256:
        raise ValueError('Subset manifest hash mismatch')
    subset = json.loads(args.subset_manifest.read_text())
    if ((subset.get('dataset_id'), subset.get('split'), subset.get('count')) !=
            ('official/TextVQA_0.5.1', 'train', 400) or
            subset.get('inference_sha256') != sha256_file(args.inference) or
            not re.fullmatch(r'[0-9a-f]{64}', str(subset.get('labels_sha256', '')))):
        raise ValueError('Unexpected frozen TextVQA manifest')
    _, anchors, samples = checked_anchor(args.anchor, args.inference,
        args.anchor_predictions_sha256, args.anchor_manifest_sha256)
    ids = [str(row['sample_id']) for row in anchors]
    if ids != [str(x) for x in subset.get('selected_ids', [])] or len(set(ids)) != 400:
        raise ValueError('Anchor IDs differ from frozen selected stream')
    report = audit_shards(args.shadow, samples, anchors, tokenizer, embedder, 400, args.shard_count)
    if report.get('status') != 'passed' or report.get('count') != 400:
        raise ValueError('Complete label-free shadow audit did not pass')
    verify_report_identity(report, args.anchor_predictions_sha256, args.anchor_manifest_sha256,
                           args.inference, args.weight_manifest)
    shadows = []
    for i in range(400):
        shard = i % args.shard_count
        path = args.shadow / f'shard-{shard:03d}-of-{args.shard_count:03d}' / f'row-{i:06d}.json'
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != report['row_sha256'].get(str(i)):
            raise ValueError('Shadow row changed after audit')
        shadows.append(json.loads(data))
    texts = _selected_texts(anchors, shadows)
    # The target file is first opened here, only after all label-free checks.
    raw_labels = args.labels.read_bytes()
    if hashlib.sha256(raw_labels).hexdigest() != subset['labels_sha256']:
        raise ValueError('Isolated labels SHA256 mismatch')
    labels = [json.loads(line) for line in raw_labels.splitlines() if line.strip()]
    if len(labels) != 400 or [str(row.get('id')) for row in labels] != ids:
        raise ValueError('Isolated target IDs/order mismatch')
    targets = [_extract_answers(row, row.get('choices', row.get('options', []))) for row in labels]
    results, contrasts = score_records(texts, targets, evaluator)
    anchor_usage = {name: {key: sum(row['arms'][arm]['usage'][key] for row in anchors)
                          for key in ('prompt_tokens', 'completion_tokens', 'latency_seconds')}
                    for name, arm in ANCHORS.items()}
    return {'protocol': 'textvqa_fixed_history_shadow_mmf_exploratory_v1', 'n': 400,
            'metric': {'name': 'Meta MMF TextVQAAccuracyEvaluator', 'revision': MMF_REVISION,
                       'sha256': MMF_SHA256, 'upstream_url': MMF_URL,
                       'extraction': 'same existing raw final-answer extraction for every condition; last-line fallback',
                       'note': 'Pinned local upstream evaluator; EvalAI server parity not independently verified.'},
            'data': {'input_sha256': subset['inference_sha256'], 'labels_sha256': subset['labels_sha256'],
                     'subset_manifest_sha256': args.subset_manifest_sha256,
                     'anchor_predictions_sha256': args.anchor_predictions_sha256,
                     'anchor_manifest_sha256': args.anchor_manifest_sha256},
            'results': results, 'paired_contrasts': contrasts,
            'cost': {'shadow_audit': report, 'anchor_logical_per_arm': anchor_usage,
                     'note': 'Shadow physical totals count unique stored requests, excluding failed unrecorded attempts. '
                             'Anchor costs are historical logical costs and cannot be added to physical replay cost.'},
            'interpretation': 'Exploratory fixed-history short-term intervention. Anchor cells are earlier model outputs; '
                              'shadow arms do not write memory and are not separate end-to-end online runs. '
                              'Control rules are length-matched placeholders, not guaranteed semantically inert. '
                              'Single previously scored development stream; paired bootstrap CI describes these items.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('anchor', 'shadow', 'inference', 'subset-manifest', 'labels', 'mmf-evaluator',
                 'embedding-path', 'model-path', 'weight-manifest', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    for name in ('anchor-predictions-sha256', 'anchor-manifest-sha256', 'subset-manifest-sha256'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--shard-count', type=int, required=True)
    parser.add_argument('--limit', type=int, default=400)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.limit != 400:
        raise ValueError('Smoke results may not be scored')
    for name, expected in EMBEDDING_FILES.items():
        if sha256_file(args.embedding_path / name) != expected:
            raise ValueError('Embedding artifact mismatch')
    verify_weights(args.model_path, args.weight_manifest)
    evaluator = load_evaluator(args.mmf_evaluator)
    from transformers import AutoTokenizer
    from core.multimodal.tf_ttcl_text_embedding import Qwen3TextEmbedder
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True)
    embedder = Qwen3TextEmbedder(str(args.embedding_path))
    result = score_experiment(args, tokenizer, embedder, evaluator)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write('\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
