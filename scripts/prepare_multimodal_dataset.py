#!/usr/bin/env python3
"""Export a ModelScope dataset to EviMem-TTCL's local JSONL schema.

The output may contain labels for offline evaluation.  ``mm_main.py`` strips
them before constructing ``MultimodalSample`` objects.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image


def first(row: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return None


def save_image(value: Any, target: Path) -> str | None:
    if value is None:
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, Image.Image):
        value.convert("RGB").save(target, format="PNG")
    elif isinstance(value, Mapping) and value.get("bytes"):
        Image.open(io.BytesIO(value["bytes"])).convert("RGB").save(target, format="PNG")
    elif isinstance(value, Mapping) and value.get("path"):
        Image.open(value["path"]).convert("RGB").save(target, format="PNG")
    elif isinstance(value, (bytes, bytearray)):
        Image.open(io.BytesIO(value)).convert("RGB").save(target, format="PNG")
    elif isinstance(value, str) and Path(value).is_file():
        Image.open(value).convert("RGB").save(target, format="PNG")
    else:
        raise TypeError(f"Unsupported image value: {type(value)!r}")
    return str(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--dataset-name", required=True, choices=["scienceqa_img", "mathvista", "textvqa", "docvqa"])
    parser.add_argument("--subset")
    parser.add_argument("--split", required=True)
    parser.add_argument("--revision", default="master")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--require-image", action="store_true")
    args = parser.parse_args()

    try:
        from modelscope.msdatasets import MsDataset
    except ImportError as exc:
        missing = getattr(exc, "name", None) or str(exc)
        raise SystemExit(
            f"Cannot import ModelScope MsDataset because dependency {missing!r} is missing"
        ) from exc

    load_kwargs = {"dataset_name": args.dataset_id, "split": args.split, "revision": args.revision}
    if args.subset:
        load_kwargs["subset_name"] = args.subset
    dataset = MsDataset.load(**load_kwargs)

    output_dir = Path(args.output_dir)
    images_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"{args.dataset_name}_{args.split}.jsonl"
    count = 0
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for index, raw in enumerate(dataset):
            if args.limit is not None and count >= args.limit:
                break
            row = dict(raw)
            sample_id = str(first(row, ("id", "pid", "question_id", "questionId")) or index)
            question = first(row, ("question", "query", "instruction", "text"))
            if question is None:
                continue
            image_value = first(row, ("image", "decoded_image", "image_path", "document"))
            if image_value is None and args.require_image:
                continue
            image_path = None
            if image_value is not None:
                image_name = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:20] + ".png"
                saved = save_image(image_value, images_dir / image_name)
                image_path = str(Path(saved).relative_to(output_dir)) if saved else None
            record = {
                "id": sample_id,
                "question": str(question),
                "image": image_path,
                "choices": first(row, ("choices", "options")) or [],
                "answer": first(row, ("answer", "label", "output", "gold")),
                "answers": first(row, ("answers",)),
            }
            for key in ("answer_type", "precision", "question_type"):
                if key in row:
                    record[key] = row[key]
            record = {key: value for key, value in record.items() if value is not None}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    manifest = {
        "dataset_id": args.dataset_id,
        "dataset_name": args.dataset_name,
        "subset": args.subset,
        "split": args.split,
        "revision": args.revision,
        "count": count,
        "jsonl": str(jsonl_path),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
