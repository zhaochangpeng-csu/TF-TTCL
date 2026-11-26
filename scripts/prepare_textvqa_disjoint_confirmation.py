#!/usr/bin/env python3
"""Prepare a disjoint 400-item TextVQA train stream; selection uses only IDs.

Annotation loading contains answers, but selection examines no answer values.
Answers are exported to a separate file only during the final export stage.
Existing completed artifacts are never overwritten.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import fetch_textvqa_train_direct_images as image_fetcher
from scripts import prepare_textvqa_train_direct as exporter
from scripts.create_label_isolated_subset import sha256_file
from scripts.inspect_baseline_io import require_label_free

SEED = 2026112602
COUNT = 400
ANNOTATION_SHA = '95f5c407db56cba56a177799dcd685a7cc0ec7c0d851b59910acf7786d31b68a'
EXCLUDES = {
    'e7b3865aa534594634ab9f9c136344d1c2673f317a2578722f09677cc8189b74': 400,
    'f3196d50185da68f150c616825630216fb8d23ebce4678ab26a9ba7b0960f5fd': 400,
    '10b8f9464bc9190a929650e83594b5f0f6a237598e9d9eff47f97f52eb25d978': 5000,
}


def select(annotation: Path, exclusions: list[Path], *, count=COUNT, seed=SEED,
           expected_train=34602, validate_hashes=True):
    if len(exclusions) != 3 or len({p.resolve() for p in exclusions}) != 3:
        raise ValueError('Exactly three distinct prior streams required')
    if validate_hashes and sha256_file(annotation) != ANNOTATION_SHA:
        raise ValueError('Annotation SHA mismatch')
    excluded_ids, excluded_images, exclusion_info, seen_hashes = set(), set(), [], set()
    for path in exclusions:
        digest = sha256_file(path)
        if validate_hashes and (digest not in EXCLUDES or digest in seen_hashes):
            raise ValueError('Unknown or repeated exclusion SHA')
        seen_hashes.add(digest)
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        if validate_hashes and len(rows) != EXCLUDES[digest]:
            raise ValueError('Prior exclusion length mismatch')
        for row in rows:
            require_label_free(row)
            sid = str(row.get('id', ''))
            image = str(row.get('image_id') or Path(row.get('image', '')).stem)
            if not sid or not image:
                raise ValueError('Prior stream missing ID or image')
            excluded_ids.add(sid)
            excluded_images.add(image)
        exclusion_info.append({'sha256': digest, 'count': len(rows)})
    if validate_hashes and seen_hashes != set(EXCLUDES):
        raise ValueError('All specified prior streams required')
    rows = json.loads(annotation.read_text(encoding='utf-8')).get('data')
    if not isinstance(rows, list) or len(rows) != expected_train:
        raise ValueError('Unexpected train annotation count')
    candidates, seen_ids = [], set()
    for row in rows:
        sid, image = str(row.get('question_id', '')), row.get('image_id')
        if (not sid or sid in seen_ids or not isinstance(image, str) or not image
                or Path(image).name != image or image in ('.', '..')
                or row.get('set_name') != 'train' or not isinstance(row.get('question'), str)
                or not row['question'].strip()):
            raise ValueError('Malformed or duplicate train ID')
        seen_ids.add(sid)
        if sid not in excluded_ids and image not in excluded_images:
            candidates.append({'id': sid, 'image_id': image})
    candidates.sort(key=lambda row: hashlib.sha256(f"{seed}:{row['id']}".encode()).hexdigest())
    chosen, images = [], set()
    for row in candidates:
        if row['image_id'] not in images:
            chosen.append(row)
            images.add(row['image_id'])
        if len(chosen) == count:
            break
    if count <= 0 or len(chosen) != count:
        raise ValueError('Not enough unique disjoint images')
    return {'dataset_id': 'official/TextVQA_0.5.1', 'split': 'train', 'seed': seed,
            'count': count, 'selection': chosen, 'unique_images': True,
            'annotation_sha256': sha256_file(annotation),
            'exclusions': sorted(exclusion_info, key=lambda x: x['sha256']),
            'excluded_question_count': len(excluded_ids), 'excluded_image_count': len(excluded_images),
            'labels_used_for_selection': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('select', 'fetch', 'export'))
    parser.add_argument('--annotation', type=Path, required=True)
    parser.add_argument('--exclude-inference', type=Path, action='append', required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--image-dir', type=Path)
    parser.add_argument('--proxy')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    chosen = select(args.annotation, args.exclude_inference)
    if args.stage == 'select':
        args.selection.parent.mkdir(parents=True, exist_ok=True)
        with args.selection.open('x', encoding='utf-8') as handle:
            json.dump(chosen, handle, ensure_ascii=False, indent=2)
            handle.write('\n')
        print(json.dumps({'status': 'selected', 'count': COUNT,
                          'selection_sha256': sha256_file(args.selection), 'labels_used_for_selection': False}))
        return
    if json.loads(args.selection.read_text()) != chosen:
        raise ValueError('Frozen selection does not match input sources')
    if args.output_dir is None:
        raise ValueError('--output-dir required')
    if args.stage == 'fetch':
        if not args.proxy:
            raise ValueError('--proxy required')
        rows = [{'image_id': item['image_id']} for item in chosen['selection']]
        with patch.object(image_fetcher, 'select_train_rows', return_value=rows):
            result = image_fetcher.fetch(args.annotation, args.output_dir, args.proxy,
                                         count=COUNT, seed=SEED, workers=args.workers)
        print(json.dumps({'status': 'images_fetched', 'image_count': result['image_count'],
                          'selection_sha256': sha256_file(args.selection)}))
        return
    if args.image_dir is None:
        raise ValueError('--image-dir required')
    rows = json.loads(args.annotation.read_text(encoding='utf-8'))['data']
    by_id = {str(row['question_id']): row for row in rows}
    rows = [by_id[item['id']] for item in chosen['selection']]
    if any(row['image_id'] != item['image_id'] for row, item in zip(rows, chosen['selection'])):
        raise ValueError('Selected image changed')
    if any(not isinstance(row.get('answers'), list) or len(row['answers']) != 10
           or any(not isinstance(answer, str) for answer in row['answers']) for row in rows):
        raise ValueError('Invalid isolated target record')
    val = next(path for path in args.exclude_inference if sha256_file(path) ==
               '10b8f9464bc9190a929650e83594b5f0f6a237598e9d9eff47f97f52eb25d978')
    with patch.object(exporter, 'select_train_rows', return_value=rows):
        result = exporter.prepare(args.annotation, None, args.output_dir,
                                  image_dir_source=args.image_dir, count=COUNT, seed=SEED,
                                  previous_val=val.parent)
    provenance = args.output_dir / 'selection_provenance.json'
    with provenance.open('x', encoding='utf-8') as handle:
        json.dump({'selection_sha256': sha256_file(args.selection), 'exclusions': chosen['exclusions'],
                   'labels_used_for_selection': False}, handle, indent=2)
        handle.write('\n')
    print(json.dumps({'status': 'exported', 'count': result['count'],
                      'inference_sha256': result['inference_sha256'], 'labels_sha256': result['labels_sha256'],
                      'labels_opened_for_export_only': True}))


if __name__ == '__main__':
    main()
