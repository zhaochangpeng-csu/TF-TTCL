#!/usr/bin/env python3
"""Report pre-existing rule coverage by a lexical gate, without labels/scores.

Only IDs and rule text from an existing predictions file are examined. Existing
predictions contain model outputs, but this program never accesses or prints
them. No labels or correctness values may be passed to this program.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.multimodal.textvqa_rule_gate import GATE_VERSION, match_rule


def report(path, expected_sha, expected_count):
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha:
        raise ValueError('prediction file hash mismatch')
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if len(rows) != expected_count or len({str(r['sample_id']) for r in rows}) != expected_count:
        raise ValueError('count or sample identities mismatch')
    patterns, buckets = Counter(), Counter()
    unique_rules, instances, sample_hits, examples = {}, 0, 0, []
    hit_counts = []
    for index, row in enumerate(rows):
        if row.get('sample_index') != index or row.get('labels_read') is not False:
            raise ValueError('unexpected order or label boundary')
        hits_this_sample = 0
        for kind, bucket in row['retrieved_rules'].items():
            if kind not in ('positive', 'negative'):
                raise ValueError('unexpected rule bucket')
            for rule in bucket:
                if rule['sample_index'] >= index:
                    raise ValueError('current/future rule found')
                rid, text = rule['rule_id'], rule['text']
                key = (kind, rid)
                if key in unique_rules and unique_rules[key] != text:
                    raise ValueError('same rule ID has different text')
                unique_rules[key] = text
                instances += 1
                matched = match_rule(text)
                if not matched:
                    continue
                hits_this_sample += 1
                patterns[matched['pattern']] += 1
                buckets[kind] += 1
                if len(examples) < 8:
                    examples.append({'sample_id': str(row['sample_id']), 'sample_index': index,
                        'bucket': kind, 'rule_id': rid, 'text': text, 'match': matched})
        hit_counts.append(hits_this_sample)
        sample_hits += bool(hits_this_sample)
    unique_matches = sum(match_rule(text) is not None for text in unique_rules.values())
    return {'gate_version': GATE_VERSION, 'prediction_sha256': digest, 'samples': len(rows),
            'labels_read': False, 'prediction_values_accessed': False,
            'retrieved_rule_instances': instances, 'matched_retrieval_instances': sum(buckets.values()),
            'matched_by_bucket': dict(buckets), 'matched_by_pattern': dict(patterns),
            'unique_retrieved_rules': len(unique_rules), 'unique_matched_rules': unique_matches,
            'samples_with_match': sample_hits, 'samples_without_match': len(rows)-sample_hits,
            'matches_per_sample': {str(k): v for k, v in sorted(Counter(hit_counts).items())},
            'examples': examples,
            'interpretation': 'First-sentence lexical match only; does not judge semantic conflict or rule truth.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--sha256', required=True)
    p.add_argument('--count', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = report(args.predictions, args.sha256, args.count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'examples'}, indent=2))


if __name__ == '__main__':
    main()
