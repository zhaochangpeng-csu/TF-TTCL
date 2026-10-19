#!/usr/bin/env python3
"""Extract an untouched, label-isolated TextVQA train stream for direct transfer.

The official train annotation and train/validation image ZIP must be supplied
locally. Only the deterministic development subset's images are extracted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

from PIL import Image

from scripts.create_label_isolated_subset import sha256_file


def prepare(annotation: Path, image_zip: Path, output: Path, *, count: int = 400,
            seed: int = 20261020, expected_count: int = 34602,
            expected_images: int = 21953, previous_val: Path | None = None) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing export: {output}")
    payload = json.loads(annotation.read_text(encoding="utf-8"))
    rows = payload.get("data")
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} official train questions")
    ids: set[str] = set()
    all_image_ids: set[str] = set()
    for row in rows:
        sample_id, image_id = str(row.get("question_id", "")), row.get("image_id")
        answers = row.get("answers")
        if (not sample_id or sample_id in ids or not isinstance(image_id, str)
                or not image_id or "/" in image_id or "\\" in image_id
                or image_id in {".", ".."} or row.get("set_name") != "train"
                or not isinstance(row.get("question"), str) or not row["question"].strip()
                or not isinstance(answers, list) or len(answers) != 10
                or any(not isinstance(answer, str) for answer in answers)):
            raise ValueError(f"Malformed or duplicate official train annotation: {sample_id}")
        ids.add(sample_id)
        all_image_ids.add(image_id)
    if len(all_image_ids) != expected_images:
        raise ValueError(f"Expected {expected_images} distinct official train images")
    if count <= 0 or count > len(rows):
        raise ValueError("Development count must be within the official train split")
    selected = sorted(rows, key=lambda row: hashlib.sha256(
        f"{seed}:{row['question_id']}".encode()).hexdigest())[:count]
    image_ids = {row["image_id"] for row in selected}
    if previous_val is not None:
        if not (previous_val / "images").is_dir():
            raise FileNotFoundError(f"Prior validation image export is missing: {previous_val}")
        if any((previous_val / "images" / f"{image_id}.jpg").exists()
               for image_id in image_ids):
            raise ValueError("Selected train stream overlaps previously used validation images")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="textvqa_direct_", dir=output.parent) as temporary:
        staged = Path(temporary) / "export"
        image_dir = staged / "images"
        image_dir.mkdir(parents=True)
        with zipfile.ZipFile(image_zip) as archive:
            by_id: dict[str, zipfile.ZipInfo] = {}
            for info in archive.infolist():
                path = Path(info.filename)
                if (not info.is_dir() and path.parent.name == "train_images"
                        and path.suffix.lower() == ".jpg" and path.stem in image_ids):
                    if path.stem in by_id:
                        raise ValueError(f"Duplicate selected image: {path.stem}")
                    by_id[path.stem] = info
            missing = image_ids - by_id.keys()
            if missing:
                raise FileNotFoundError(f"Missing {len(missing)} selected images; first: {sorted(missing)[0]}")
            for image_id in sorted(image_ids):
                target = image_dir / f"{image_id}.jpg"
                with archive.open(by_id[image_id]) as source, target.open("xb") as sink:
                    shutil.copyfileobj(source, sink)
                with Image.open(target) as image:
                    image.verify()

        inference = staged / "inference.jsonl"
        labels = staged / "labels.jsonl"
        with inference.open("x", encoding="utf-8") as public, labels.open("x", encoding="utf-8") as private:
            for row in selected:
                record = {"id": str(row["question_id"]), "image_id": row["image_id"],
                          "question": row["question"], "image": f"images/{row['image_id']}.jpg",
                          "choices": []}
                public.write(json.dumps(record, ensure_ascii=False) + "\n")
                private.write(json.dumps({**record, "answers": row["answers"]}, ensure_ascii=False) + "\n")
        selected_ids = [str(row["question_id"]) for row in selected]
        manifest = {"dataset_id": "official/TextVQA_0.5.1", "split": "train", "seed": seed,
                    "offset": 0, "count": count, "image_count": len(image_ids),
                    "selected_ids": selected_ids,
                    "selected_ids_sha256": hashlib.sha256("\n".join(selected_ids).encode()).hexdigest(),
                    "annotation_sha256": sha256_file(annotation),
                    "image_archive_sha256": sha256_file(image_zip),
                    "inference_sha256": sha256_file(inference),
                    "labels_sha256": sha256_file(labels),
                    "inference_path": str((output / "inference.jsonl").resolve()),
                    "labels_path": str((output / "labels.jsonl").resolve())}
        (staged / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                                encoding="utf-8")
        os.replace(staged, output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--image-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--previous-val", type=Path)
    parser.add_argument("--count", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args()
    print(json.dumps(prepare(args.annotation, args.image_zip, args.output_dir,
                             count=args.count, seed=args.seed, previous_val=args.previous_val), indent=2))


if __name__ == "__main__":
    main()
