#!/usr/bin/env python3
"""Prepare a frozen disjoint TextVQA train stream in three explicit stages.

Selection uses only question/image identifiers, never answer values. Images are
fetched from the official ZIP; target answers are copied into a separate file
only by the export stage. Existing outputs are never overwritten.
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

SEED = 20261126
COUNT = 400
ANNOTATION_SHA = '95f5c407db56cba56a177799dcd685a7cc0ec7c0d851b59910acf7786d31b68a'
PREVIOUS_TRAIN_SHA = 'e7b3865aa534594634ab9f9c136344d1c2673f317a2578722f09677cc8189b74'


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
    for row in rows:
        require_label_free(row)
    return rows


def select(annotation: Path, old_train: Path, old_val: Path, *, count: int = COUNT,
           expected_train: int = 34602, expected_previous: int = 400,
           expected_val: int = 5000, validate_hashes: bool = True) -> dict:
    if validate_hashes and (sha256_file(annotation) != ANNOTATION_SHA or
                            sha256_file(old_train) != PREVIOUS_TRAIN_SHA):
        raise ValueError('Official annotation or prior train stream SHA mismatch')
    previous, validation = read_jsonl(old_train), read_jsonl(old_val)
    if len(previous) != expected_previous or len(validation) != expected_val:
        raise ValueError('Previous train/validation stream is incomplete')
    excluded_ids = {str(row['id']) for row in previous + validation}
    excluded_images = {str(row.get('image_id') or Path(row['image']).stem)
                       for row in previous + validation}
    payload = json.loads(annotation.read_text(encoding='utf-8'))
    rows = payload.get('data')
    if not isinstance(rows, list) or len(rows) != expected_train:
        raise ValueError('Unexpected official train annotation size')
    candidates = []
    seen_ids = set()
    for row in rows:
        sample_id = str(row.get('question_id', ''))
        image_id = row.get('image_id')
        question = row.get('question')
        if (not sample_id or sample_id in seen_ids or not isinstance(image_id, str)
                or not image_id or Path(image_id).name != image_id or image_id in ('.', '..')
                or row.get('set_name') != 'train' or not isinstance(question, str) or not question.strip()):
            raise ValueError(f'Malformed or duplicate annotation ID: {sample_id}')
        seen_ids.add(sample_id)
        if sample_id not in excluded_ids and image_id not in excluded_images:
            candidates.append({'id': sample_id, 'image_id': image_id})
    candidates.sort(key=lambda row: hashlib.sha256(f"{SEED}:{row['id']}".encode()).hexdigest())
    selected, used_images = [], set()
    for row in candidates:
        if row['image_id'] not in used_images:
            selected.append(row)
            used_images.add(row['image_id'])
        if len(selected) == count:
            break
    if count <= 0 or len(selected) != count:
        raise ValueError('Not enough disjoint unique-image train questions')
    return {'dataset_id': 'official/TextVQA_0.5.1', 'split': 'train', 'seed': SEED,
            'count': count, 'selection': selected, 'unique_images': True,
            'annotation_sha256': sha256_file(annotation),
            'previous_train_inference_sha256': sha256_file(old_train),
            'previous_val_inference_sha256': sha256_file(old_val),
            'excluded_question_count': len(excluded_ids),
            'excluded_image_count': len(excluded_images), 'labels_used_for_selection': False}


def checked_selection(path: Path, annotation: Path, old_train: Path, old_val: Path) -> dict:
    saved = json.loads(path.read_text(encoding='utf-8'))
    expected = select(annotation, old_train, old_val)
    if saved != expected:
        raise ValueError('Frozen selection differs from current source, exclusions or seed')
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('select', 'fetch', 'export'))
    parser.add_argument('--annotation', type=Path, required=True)
    parser.add_argument('--previous-train-inference', type=Path, required=True)
    parser.add_argument('--previous-val-inference', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--image-dir', type=Path)
    parser.add_argument('--proxy')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    sources = (args.annotation, args.previous_train_inference, args.previous_val_inference)
    if args.stage == 'select':
        result = select(*sources)
        args.selection.parent.mkdir(parents=True, exist_ok=True)
        with args.selection.open('x', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            f.write('\n')
        print(json.dumps({'status': 'selected', 'count': result['count'],
                          'selection_sha256': sha256_file(args.selection),
                          'labels_used_for_selection': False}))
        return
    selection = checked_selection(args.selection, *sources)
    if args.output_dir is None:
        raise ValueError('--output-dir is required')
    if args.stage == 'fetch':
        if not args.proxy:
            raise ValueError('--proxy is required for official ZIP range fetch')
        rows = [{'image_id': item['image_id']} for item in selection['selection']]
        with patch.object(image_fetcher, 'select_train_rows', return_value=rows):
            manifest = image_fetcher.fetch(args.annotation, args.output_dir, args.proxy,
                                           count=COUNT, seed=SEED, workers=args.workers)
        print(json.dumps({'status': 'images_fetched', 'image_count': manifest['image_count'],
                          'selection_sha256': sha256_file(args.selection)}))
        return
    if args.image_dir is None:
        raise ValueError('--image-dir is required for export')
    all_rows = json.loads(args.annotation.read_text(encoding='utf-8'))['data']
    by_id = {str(row['question_id']): row for row in all_rows}
    rows = [by_id[item['id']] for item in selection['selection']]
    if any(row['image_id'] != item['image_id'] for row, item in zip(rows, selection['selection'])):
        raise ValueError('Selected image IDs changed')
    if any(not isinstance(row.get('answers'), list) or len(row['answers']) != 10
           or any(not isinstance(answer, str) for answer in row['answers']) for row in rows):
        raise ValueError('Selected label record must have ten string answers')
    with patch.object(exporter, 'select_train_rows', return_value=rows):
        result = exporter.prepare(args.annotation, None, args.output_dir,
                                  image_dir_source=args.image_dir, count=COUNT, seed=SEED,
                                  previous_val=args.previous_val_inference.parent)
    with (args.output_dir / 'selection_provenance.json').open('x', encoding='utf-8') as f:
        json.dump({'selection_sha256': sha256_file(args.selection),
                   'previous_train_inference_sha256': selection['previous_train_inference_sha256'],
                   'previous_val_inference_sha256': selection['previous_val_inference_sha256'],
                   'labels_used_for_selection': False}, f, indent=2)
        f.write('\n')
    print(json.dumps({'status': 'exported', 'count': result['count'],
                      'inference_sha256': result['inference_sha256'],
                      'labels_sha256': result['labels_sha256'], 'labels_opened_for_export_only': True}))


if __name__ == '__main__':
    main()
