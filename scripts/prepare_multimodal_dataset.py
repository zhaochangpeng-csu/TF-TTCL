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
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import parse_qs, unquote, urlparse

from PIL import Image


def cached_modelscope_parquet_rows(dataset_id: str, split: str) -> tuple[Iterable[Mapping[str, Any]], list[str]]:
    """Read parquet shards already fetched by ModelScope, bypassing MsDataset.

    ModelScope 1.40's dataset adapter is incompatible with some recent
    ``datasets`` releases. Its download cache records the source URL next to
    every content-addressed file, so the fallback remains provenance-aware and
    never guesses from a bare hash.
    """
    cache_root = Path(os.getenv("MODELSCOPE_CACHE", Path.home() / ".cache/modelscope"))
    downloads = cache_root / "hub" / "datasets" / "downloads"
    shards: list[Path] = []
    source_urls: list[str] = []
    expected_repo = f"/datasets/{dataset_id}/repo".lower()
    for metadata_path in sorted(downloads.glob("*.json")):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            url = str(metadata.get("url", ""))
            parsed = urlparse(url)
            file_path = unquote(parse_qs(parsed.query).get("FilePath", [""])[0])
        except (OSError, ValueError, TypeError):
            continue
        if expected_repo not in parsed.path.lower():
            continue
        if split.lower() not in Path(file_path).name.lower() or not file_path.endswith(".parquet"):
            continue
        data_path = metadata_path.with_suffix("")
        if data_path.is_file():
            shards.append(data_path)
            source_urls.append(url)
    if not shards:
        raise FileNotFoundError(
            f"No cached ModelScope parquet shard found for {dataset_id!r} split {split!r} under {downloads}"
        )

    def rows() -> Iterable[Mapping[str, Any]]:
        import pyarrow.parquet as pq

        for shard in shards:
            parquet = pq.ParquetFile(shard)
            for batch in parquet.iter_batches(batch_size=64):
                yield from batch.to_pylist()

    return rows(), source_urls


def load_modelscope_rows(
    dataset_id: str,
    split: str,
    revision: str,
    subset: str | None,
    cached_only: bool,
) -> tuple[Iterable[Mapping[str, Any]], str, list[str]]:
    if not cached_only:
        try:
            from modelscope.msdatasets import MsDataset

            load_kwargs = {"dataset_name": dataset_id, "split": split, "version": revision}
            if subset:
                load_kwargs["subset_name"] = subset
            return MsDataset.load(**load_kwargs), "modelscope_msdataset", []
        except Exception as exc:  # compatibility failures occur after the shard is cached
            print(
                f"MsDataset failed ({type(exc).__name__}: {exc}); trying its verified parquet cache.",
                file=sys.stderr,
            )
    rows, urls = cached_modelscope_parquet_rows(dataset_id, split)
    return rows, "modelscope_cached_parquet", urls


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
    parser.add_argument(
        "--cached-only",
        action="store_true",
        help="Read ModelScope's provenance metadata and cached parquet without invoking MsDataset",
    )
    args = parser.parse_args()

    dataset, source_backend, source_urls = load_modelscope_rows(
        args.dataset_id, args.split, args.revision, args.subset, args.cached_only
    )

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
        "source_backend": source_backend,
        "source_urls": source_urls,
        "count": count,
        "jsonl": str(jsonl_path),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
