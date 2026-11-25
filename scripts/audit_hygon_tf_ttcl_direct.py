#!/usr/bin/env python3
"""Audit a Hygon ScienceQA online run against the immutable NVIDIA inputs.

The benchmark JSONL retains original absolute image paths. Resolve only the
known old image directory into the local migrated image directory; the input
file itself and its manifest hash remain unchanged. No labels are opened.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import audit_tf_ttcl_mm_direct as original
from scripts.create_label_isolated_subset import sha256_file

INPUT_SHA = '01c6dc83958ea1cee51ab38594b2499e3bfbc4034139674dc799e5dfeb354689'
REFERENCE_SHA = 'b8ac87c5bdf38923f7ec5e9e848b3f1b18e0f183b2ea0deffea98700ffaf4d71'
OLD_ROOT = Path('/home/lthpc/private/code/zcp/TTT/TF-TTCL/data/multimodal/scienceqa_modelscope_train/images')


def mapped_inputs(inputs: list[dict], refs: list[dict], rows: list[dict],
                  image_root: Path, count: int) -> list[dict]:
    if count not in (20, 400) or len(inputs) != 400 or len(refs) != 400 or len(rows) != count:
        raise ValueError('Expected original 400 inputs/reference and complete 20/400 output')
    root = image_root.resolve(strict=True)
    output = []
    prior_rules: dict[str, dict] = {}
    for index, (source, ref, row) in enumerate(zip(inputs[:count], refs[:count], rows)):
        original.require_label_free(source)
        if (str(source['id']) != ref.get('sample_id') or ref.get('sample_index') != index
                or row.get('sample_id') != ref.get('sample_id') or row.get('sample_index') != index):
            raise ValueError(f'Input/reference/run order mismatch at {index}')
        image = source.get('image')
        if not isinstance(image, str) or Path(image).parent != OLD_ROOT:
            raise ValueError(f'Unexpected source image at {index}')
        local = (root / Path(image).name).resolve(strict=True)
        if local.parent != root or not local.is_file():
            raise ValueError(f'Unsafe local image at {index}')
        digest = sha256_file(local)
        if ref.get('image_hashes') != [digest] or row.get('image_hashes') != [digest]:
            raise ValueError(f'Image/reference/run hash mismatch at {index}')
        for kind, bucket in row.get('retrieved_rules', {}).items():
            for rule in bucket:
                previous = prior_rules.get(rule['rule_id'])
                if (not previous or previous['sample_index'] >= index or
                        previous['sample_index'] != rule['sample_index'] or
                        previous['kind'] != kind or previous['text'] != rule['text']):
                    raise ValueError(f'Retrieval is not an earlier written rule at {index}')
        for rule in row.get('written_rules', []):
            if rule['rule_id'] in prior_rules or rule['sample_index'] != index:
                raise ValueError(f'Duplicate or invalid rule write at {index}')
            prior_rules[rule['rule_id']] = rule
        output.append({**source, 'image': str(local)})
    # The original audit scans all 400 inputs for label leakage even in smoke.
    output.extend(inputs[count:])
    return output


def audit(output: Path, inference: Path, reference: Path, count: int):
    if sha256_file(inference) != INPUT_SHA or sha256_file(reference) != REFERENCE_SHA:
        raise ValueError('Frozen inference/reference SHA mismatch')
    read = original.read_rows
    inputs, refs = read(inference), read(reference)
    rows = read(output / 'predictions.jsonl')
    mapped = mapped_inputs(inputs, refs, rows, inference.parent / 'images', count)
    inference_resolved = inference.resolve()
    def mapped_read(path):
        return mapped if Path(path).resolve() == inference_resolved else read(path)
    with patch.object(original, 'read_rows', side_effect=mapped_read):
        manifest, checked = original.audit(output, inference, count)
    if manifest.get('ordered_ids') != [str(row['id']) for row in inputs[:count]]:
        raise ValueError('Manifest ordered IDs do not match the frozen prefix')
    return manifest, checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inference', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--count', type=int, choices=(20, 400), required=True)
    parser.add_argument('--casebook', type=Path)
    args = parser.parse_args()
    _, rows = audit(args.output, args.inference, args.reference, args.count)
    if args.casebook:
        args.casebook.parent.mkdir(parents=True, exist_ok=True)
        with args.casebook.open('x', encoding='utf-8') as f:
            f.write(original.casebook(rows))
    print(json.dumps({'audit': 'passed', 'samples': len(rows), 'labels_read': False,
                      'rule_writes': sum(len(row['written_rules']) for row in rows)}, indent=2))


if __name__ == '__main__':
    main()
