"""Conservative image views and counterfactual probes."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Sequence

from PIL import Image


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_fidelity_views(
    image_paths: Sequence[str], output_dir: Path, sample_id: str
) -> List[str]:
    """Re-encode without geometric or photometric modifications."""
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_id = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:20]
    views: List[str] = []
    for index, source_path in enumerate(image_paths):
        target = output_dir / f"{safe_id}_{index}_fidelity.png"
        with Image.open(source_path) as image:
            if image.mode not in {"RGB", "RGBA"}:
                image = image.convert("RGB")
            image.save(target, format="PNG", optimize=False)
        views.append(str(target))
    return views


def make_blank_views(
    image_paths: Sequence[str], output_dir: Path, sample_id: str
) -> List[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_id = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:20]
    blanks: List[str] = []
    for index, source_path in enumerate(image_paths):
        target = output_dir / f"{safe_id}_{index}_blank.png"
        with Image.open(source_path) as image:
            blank = Image.new("RGB", image.size, color=(255, 255, 255))
            blank.save(target, format="PNG")
        blanks.append(str(target))
    return blanks
