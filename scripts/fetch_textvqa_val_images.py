#!/usr/bin/env python3
"""Fetch only official TextVQA validation images using HTTP ZIP range reads.

Creates the same export as prepare_textvqa_official.py without downloading the
7 GB train/validation archive. Completed images are verified and resumable.
Only a finished 5,000-question export receives manifest.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import threading
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fsspec
from PIL import Image

from scripts.create_label_isolated_subset import sha256_file


DEFAULT_URL = "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip"


def _valid_image(path: Path, info: zipfile.ZipInfo) -> bool:
    if not path.is_file() or path.stat().st_size != info.file_size:
        return False
    checksum = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum = zlib.crc32(block, checksum)
    return (checksum & 0xFFFFFFFF) == info.CRC


def fetch(annotation: Path, output: Path, url: str = DEFAULT_URL,
          expected_count: int = 5000, expected_images: int = 3166,
          workers: int = 4) -> dict:
    if not 1 <= workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    rows = json.loads(annotation.read_text(encoding="utf-8")).get("data")
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count} official validation rows")
    ids = set()
    image_ids = set()
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
    annotation_sha = sha256_file(annotation)
    if (output / "manifest.json").exists():
        raise FileExistsError(f"Complete export already exists: {output}")
    marker = output / ".fetch_in_progress.json"
    image_dir = output / "images"
    output.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(exist_ok=True)
    source_info = {"annotation_sha256": annotation_sha, "image_archive_source": url,
                   "expected_count": expected_count, "expected_images": expected_images}
    if marker.exists():
        if json.loads(marker.read_text()) != source_info:
            raise ValueError("Existing partial export uses different source or annotation")
    elif any(output.iterdir()) and any(image_dir.iterdir()):
        raise ValueError("Unmarked images exist; refusing to adopt them as a partial export")
    else:
        marker.write_text(json.dumps(source_info, indent=2) + "\n")

    started = time.monotonic()
    with fsspec.open(url, "rb", block_size=512 * 1024).open() as remote:
        with zipfile.ZipFile(remote) as archive:
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
            member_index = hashlib.sha256("\n".join(
                f"{image_id}:{by_id[image_id].CRC}:{by_id[image_id].file_size}:{by_id[image_id].compress_size}"
                for image_id in sorted(image_ids)).encode()).hexdigest()
            local = threading.local()
            opened: list[tuple[zipfile.ZipFile, object]] = []
            opened_lock = threading.Lock()

            def download_one(image_id: str) -> bool:
                info = by_id[image_id]
                target = image_dir / f"{image_id}.jpg"
                if _valid_image(target, info):
                    return False
                if not hasattr(local, "archive"):
                    source_file = fsspec.open(url, "rb", block_size=512 * 1024).open()
                    try:
                        worker_archive = zipfile.ZipFile(source_file)
                    except BaseException:
                        source_file.close()
                        raise
                    local.archive = worker_archive
                    with opened_lock:
                        opened.append((worker_archive, source_file))
                temporary = target.with_suffix(".jpg.part")
                with local.archive.open(info) as source, temporary.open("wb") as sink:
                    shutil.copyfileobj(source, sink)
                if not _valid_image(temporary, info):
                    raise ValueError(f"Downloaded image failed CRC/size verification: {image_id}")
                with Image.open(temporary) as image:
                    image.verify()
                os.replace(temporary, target)
                return True

            downloaded = 0
            ordered_ids = sorted(image_ids)
            try:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    for start in range(0, len(ordered_ids), 50):
                        batch = ordered_ids[start:start + 50]
                        futures = [executor.submit(download_one, image_id) for image_id in batch]
                        for future in as_completed(futures):
                            downloaded += int(future.result())
                        print(f"[{start + len(batch)}/{len(image_ids)}] new={downloaded} "
                              f"elapsed={time.monotonic()-started:.0f}s", flush=True)
            finally:
                for worker_archive, source_file in opened:
                    worker_archive.close()
                    source_file.close()

    jsonl = output / "textvqa_val.jsonl"
    temporary_jsonl = output / "textvqa_val.jsonl.part"
    with temporary_jsonl.open("w", encoding="utf-8") as handle:
        for row in rows:
            record = {"id": str(row["question_id"]), "image_id": row["image_id"],
                      "question": row["question"], "image": f"images/{row['image_id']}.jpg",
                      "choices": [], "answers": row["answers"]}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    os.replace(temporary_jsonl, jsonl)
    manifest = {"dataset_id": "official/TextVQA_0.5.1", "split": "val",
                "count": len(rows), "image_count": len(image_ids),
                "annotation_sha256": annotation_sha,
                "image_archive_source": url,
                "image_member_index_sha256": member_index,
                "jsonl_sha256": sha256_file(jsonl), "jsonl": "textvqa_val.jsonl"}
    temporary_manifest = output / "manifest.json.part"
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary_manifest, output / "manifest.json")
    marker.unlink()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(fetch(args.annotation, args.output_dir, args.url, workers=args.workers), indent=2))


if __name__ == "__main__":
    main()
