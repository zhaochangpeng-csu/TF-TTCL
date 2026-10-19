#!/usr/bin/env python3
"""Fetch only 400 frozen official TextVQA train images via ZIP byte ranges.

The documented campus SOCKS5h proxy is passed per command, never persisted.
Images are CRC/size/PIL checked and partial fetches are safely resumable.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import threading
import time
import zipfile
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from scripts.create_label_isolated_subset import sha256_file
from scripts.fetch_textvqa_val_images import _valid_image
from scripts.prepare_textvqa_train_direct import select_train_rows


DEFAULT_URL = "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip"


class ProxyRangeReader(io.RawIOBase):
    """Seekable, block-cached HTTP reader with strict 206/ETag validation."""

    def __init__(self, url: str, proxy: str, block_size: int = 512 * 1024,
                 session: requests.Session | None = None) -> None:
        super().__init__()
        if not proxy.startswith("socks5h://"):
            raise ValueError("A SOCKS5h proxy is required for the managed server")
        self.url, self.block_size = url, block_size
        self.session = session or requests.Session()
        self.session.trust_env = False
        self.session.proxies.update({"http": proxy, "https": proxy})
        if session is None:
            adapter = HTTPAdapter(max_retries=Retry(total=5, backoff_factor=1,
                status_forcelist=(429, 500, 502, 503, 504)))
            self.session.mount("http://", adapter)
            self.session.mount("https://", adapter)
        response = self.session.head(url, allow_redirects=True, timeout=30)
        response.raise_for_status()
        self.size = int(response.headers["Content-Length"])
        self.etag = response.headers.get("ETag")
        self.position = 0
        self.cache: OrderedDict[int, bytes] = OrderedDict()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            position = offset
        elif whence == io.SEEK_CUR:
            position = self.position + offset
        elif whence == io.SEEK_END:
            position = self.size + offset
        else:
            raise ValueError("Invalid seek mode")
        if position < 0:
            raise ValueError("Cannot seek before start")
        self.position = position
        return position

    def _block(self, block_index: int) -> bytes:
        if block_index in self.cache:
            self.cache.move_to_end(block_index)
            return self.cache[block_index]
        start = block_index * self.block_size
        end = min(self.size - 1, start + self.block_size - 1)
        response = self.session.get(self.url, headers={"Range": f"bytes={start}-{end}"},
                                    timeout=60)
        response.raise_for_status()
        expected_range = f"bytes {start}-{end}/{self.size}"
        if (response.status_code != 206 or response.headers.get("Content-Range") != expected_range
                or len(response.content) != end - start + 1
                or (self.etag and response.headers.get("ETag") not in (None, self.etag))):
            raise IOError(f"Official ZIP range response changed or was incomplete at {start}-{end}")
        self.cache[block_index] = response.content
        if len(self.cache) > 8:
            self.cache.popitem(last=False)
        return response.content

    def read(self, size: int = -1) -> bytes:
        if self.position >= self.size:
            return b""
        remaining = self.size - self.position if size is None or size < 0 else min(size, self.size - self.position)
        parts: list[bytes] = []
        while remaining:
            block_index, offset = divmod(self.position, self.block_size)
            block = self._block(block_index)
            piece = block[offset:offset + remaining]
            if not piece:
                raise IOError("Empty ZIP range block")
            parts.append(piece)
            self.position += len(piece)
            remaining -= len(piece)
        return b"".join(parts)

    def close(self) -> None:
        if not self.closed:
            self.session.close()
        super().close()


def fetch(annotation: Path, output: Path, proxy: str, url: str = DEFAULT_URL,
          *, count: int = 400, seed: int = 20261020, workers: int = 4) -> dict:
    if not 1 <= workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    selected = select_train_rows(annotation, count, seed)
    image_ids = {row["image_id"] for row in selected}
    if (output / "manifest.json").exists():
        raise FileExistsError(f"Image fetch already completed: {output}")
    marker = output / ".fetch_in_progress.json"
    image_dir = output / "images"
    output.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(exist_ok=True)
    selected_hash = hashlib.sha256("\n".join(sorted(image_ids)).encode()).hexdigest()
    source = {"annotation_sha256": sha256_file(annotation), "source_url": url,
              "count": count, "seed": seed, "selected_image_ids_sha256": selected_hash}
    if marker.exists():
        if json.loads(marker.read_text()) != source:
            raise ValueError("Partial image export has a different source or selection")
    elif any(image_dir.iterdir()):
        raise ValueError("Unmarked image directory; refusing to adopt it")
    else:
        marker.write_text(json.dumps(source, indent=2) + "\n")

    started = time.monotonic()
    with ProxyRangeReader(url, proxy) as reader:
        source.update({"zip_size": reader.size, "zip_etag": reader.etag})
        with zipfile.ZipFile(reader) as archive:
            by_id: dict[str, zipfile.ZipInfo] = {}
            for info in archive.infolist():
                path = Path(info.filename)
                if (not info.is_dir() and path.parent.name == "train_images"
                        and path.suffix.lower() == ".jpg" and path.stem in image_ids):
                    if path.stem in by_id:
                        raise ValueError(f"Duplicate image in official ZIP: {path.stem}")
                    by_id[path.stem] = info
            missing = image_ids - by_id.keys()
            if missing:
                raise FileNotFoundError(f"Official ZIP lacks selected image: {sorted(missing)[0]}")
            member_index = hashlib.sha256("\n".join(
                f"{image_id}:{by_id[image_id].CRC}:{by_id[image_id].file_size}:{by_id[image_id].compress_size}"
                for image_id in sorted(image_ids)).encode()).hexdigest()
            local = threading.local()
            opened: list[tuple[zipfile.ZipFile, ProxyRangeReader]] = []
            opened_lock = threading.Lock()

            def download_one(image_id: str) -> bool:
                info = by_id[image_id]
                target = image_dir / f"{image_id}.jpg"
                if target.exists():
                    if not _valid_image(target, info):
                        raise ValueError(f"Existing selected image failed CRC/size check: {target}")
                    return False
                if not hasattr(local, "archive"):
                    worker_reader = ProxyRangeReader(url, proxy)
                    if worker_reader.size != reader.size or worker_reader.etag != reader.etag:
                        worker_reader.close()
                        raise IOError("Official ZIP changed during image fetch")
                    try:
                        worker_archive = zipfile.ZipFile(worker_reader)
                    except BaseException:
                        worker_reader.close()
                        raise
                    local.archive = worker_archive
                    with opened_lock:
                        opened.append((worker_archive, worker_reader))
                temporary = target.with_suffix(".jpg.part")
                with local.archive.open(info) as source_handle, temporary.open("wb") as sink:
                    shutil.copyfileobj(source_handle, sink)
                if not _valid_image(temporary, info):
                    raise ValueError(f"Fetched image failed official ZIP CRC: {image_id}")
                with Image.open(temporary) as image:
                    image.verify()
                os.replace(temporary, target)
                return True

            completed = 0
            downloaded = 0
            try:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    for start in range(0, len(image_ids), 20):
                        batch = sorted(image_ids)[start:start + 20]
                        futures = [executor.submit(download_one, image_id) for image_id in batch]
                        for future in as_completed(futures):
                            downloaded += int(future.result())
                            completed += 1
                        print(f"[{completed}/{len(image_ids)}] new={downloaded} "
                              f"elapsed={time.monotonic()-started:.0f}s", flush=True)
            finally:
                for worker_archive, worker_reader in opened:
                    worker_archive.close()
                    worker_reader.close()

    manifest = {**source, "image_count": len(image_ids),
                "image_member_index_sha256": member_index,
                "selected_images_sha256": hashlib.sha256("\n".join(
                    f"{image_id}:{sha256_file(image_dir / f'{image_id}.jpg')}"
                    for image_id in sorted(image_ids)).encode()).hexdigest(),
                "elapsed_seconds": time.monotonic() - started}
    temporary_manifest = output / "manifest.json.part"
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary_manifest, output / "manifest.json")
    marker.unlink()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--proxy", required=True)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    result = fetch(args.annotation, args.output_dir, args.proxy, args.url, workers=args.workers)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
