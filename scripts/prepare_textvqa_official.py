#!/usr/bin/env python3
"""Import official TextVQA v0.5.1 validation annotations and image archive.

Only images referenced by validation are extracted. The output retains the ten
human answers for the offline evaluator; create_label_isolated_subset.py must
be run before any model call.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

from PIL import Image

from scripts.create_label_isolated_subset import sha256_file


def prepare(annotation: Path, image_zip: Path, output: Path,
            expected_count: int = 5000, expected_images: int = 3166) -> dict:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite dataset export: {output}")
    data = json.loads(annotation.read_text(encoding="utf-8"))
    rows = data.get("data")
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} official validation rows")
    ids: set[str] = set()
    image_ids: set[str] = set()
    for row in rows:
        sample_id = str(row.get("question_id", ""))
        image_id = row.get("image_id")
        answers = row.get("answers")
        if (not sample_id or sample_id in ids or not isinstance(image_id, str)
                or not image_id or "/" in image_id or "\\" in image_id
                or image_id in {".", ".."} or row.get("set_name") != "val"
                or not isinstance(row.get("question"), str) or not row["question"].strip()
                or not isinstance(answers, list) or len(answers) != 10
                or any(not isinstance(answer, str) for answer in answers)):
            raise ValueError(f"Malformed or duplicate validation annotation: {sample_id}")
        ids.add(sample_id)
        image_ids.add(image_id)
    if len(image_ids) != expected_images:
        raise ValueError(f"Expected {expected_images} distinct validation images")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="textvqa_import_", dir=output.parent) as temporary:
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
                        raise ValueError(f"Duplicate image in archive: {path.stem}")
                    by_id[path.stem] = info
            missing = image_ids - by_id.keys()
            if missing:
                raise FileNotFoundError(f"Missing {len(missing)} validation images; first: {sorted(missing)[0]}")
            for image_id in sorted(image_ids):
                target = image_dir / f"{image_id}.jpg"
                with archive.open(by_id[image_id]) as source, target.open("xb") as sink:
                    shutil.copyfileobj(source, sink)
                with Image.open(target) as image:
                    image.verify()

        jsonl = staged / "textvqa_val.jsonl"
        with jsonl.open("x", encoding="utf-8") as handle:
            for row in rows:
                record = {"id": str(row["question_id"]), "image_id": row["image_id"],
                          "question": row["question"],
                          "image": f"images/{row['image_id']}.jpg", "choices": [],
                          "answers": row["answers"]}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        manifest = {"dataset_id": "official/TextVQA_0.5.1", "split": "val",
                    "count": len(rows), "image_count": len(image_ids),
                    "annotation_sha256": sha256_file(annotation),
                    "image_archive_sha256": sha256_file(image_zip),
                    "jsonl_sha256": sha256_file(jsonl),
                    "jsonl": "textvqa_val.jsonl"}
        (staged / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(staged, output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--image-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.annotation, args.image_zip, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
