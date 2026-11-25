#!/usr/bin/env python3
"""Score the complete audited Hygon online run against isolated labels."""
import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_hygon_tf_ttcl_direct import audit
from scripts.create_label_isolated_subset import sha256_file
from evaluate.multimodal import compare_tf_ttcl_mm_direct as scorer

LABEL_SHA = '009cff0b2f94a472311dfb6f112558bfecbb81477f11ae100c15de4c0c451fb7'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'inference', 'reference', 'labels', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    audit(args.run, args.inference, args.reference, 400)
    if sha256_file(args.labels) != LABEL_SHA:
        raise ValueError('Original labels hash mismatch')
    with patch.object(scorer, 'audit', side_effect=lambda output, inference, count:
                      audit(output, inference, args.reference, count)):
        result = scorer.compare(args.run, args.inference, args.labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps({'status': result['status'], 'count': result['n'],
                      'accuracy': {name: row['accuracy'] for name, row in result['results'].items()}}, indent=2))


if __name__ == '__main__':
    main()
