"""Conservative image views and counterfactual probes."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List, Sequence, Tuple

from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageStat


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


def make_textvqa_photometric_views(
    image_paths: Sequence[str], output_dir: Path, sample_id: str
) -> List[List[str]]:
    """Two fixed, full-frame OCR views with unchanged dimensions and geometry.

    No crop, flip, rotation, resizing, grayscale conversion, or JPEG loss is
    introduced. The interventions are intentionally mild; they are not
    guaranteed to preserve every colour-sensitive answer and must be audited.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_id = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:20]
    variants: List[List[str]] = [[], []]
    for index, source_path in enumerate(image_paths):
        with Image.open(source_path) as opened:
            source = opened.convert("RGB")
            transformed = (
                source.filter(ImageFilter.UnsharpMask(radius=1, percent=60, threshold=3)),
                ImageEnhance.Contrast(source).enhance(1.08),
            )
            for variant_index, view in enumerate(transformed):
                target = output_dir / f"{safe_id}_{index}_photo{variant_index + 1}.png"
                if target.exists():
                    with Image.open(target) as existing:
                        if (existing.size != source.size or existing.mode != "RGB"
                                or ImageChops.difference(existing, view).getbbox() is not None):
                            raise ValueError(f"Existing image view differs from fixed transform: {target}")
                else:
                    view.save(target, format="PNG", optimize=False)
                variants[variant_index].append(str(target))
    return variants


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


def _grid_cell_box(cell: str, width: int, height: int) -> Tuple[int, int, int, int]:
    if len(cell) != 4 or cell[0] != "r" or cell[2] != "c":
        raise ValueError(f"Invalid grid cell: {cell}")
    row, column = int(cell[1]), int(cell[3])
    if row not in range(1, 5) or column not in range(1, 5):
        raise ValueError(f"Invalid grid cell: {cell}")
    return (
        (column - 1) * width // 4,
        (row - 1) * height // 4,
        column * width // 4,
        row * height // 4,
    )


def make_grid_intervention_views(
    image_paths: Sequence[str],
    cells: Sequence[str],
    output_dir: Path,
    sample_id: str,
    tag: str,
) -> Tuple[List[str], List[str]]:
    """Create full-context focus and neutral-drop views for the first image.

    The focus view dims rather than removes the surrounding image, retaining
    global topology. The drop view replaces only the selected cells with the
    image's mean color. Remaining images, if any, are left untouched.
    """
    if not image_paths:
        raise ValueError("Grid interventions require at least one image")
    normalized_cells = tuple(sorted(set(cell.lower() for cell in cells)))
    if not normalized_cells:
        raise ValueError("Grid interventions require at least one cell")
    output_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(
        f"{sample_id}:{tag}:{','.join(normalized_cells)}".encode("utf-8")
    ).hexdigest()[:20]
    focus_path = output_dir / f"{key}_focus.png"
    drop_path = output_dir / f"{key}_drop.png"
    if not focus_path.exists() or not drop_path.exists():
        with Image.open(image_paths[0]) as opened:
            source = opened.convert("RGB")
            mask = Image.new("L", source.size, color=0)
            for cell in normalized_cells:
                mask.paste(255, _grid_cell_box(cell, *source.size))

            dimmed = ImageEnhance.Brightness(source).enhance(0.25)
            focused = Image.composite(source, dimmed, mask)

            mean = tuple(round(value) for value in ImageStat.Stat(source).mean[:3])
            neutral = Image.new("RGB", source.size, color=mean)
            dropped = Image.composite(neutral, source, mask)
            focused.save(focus_path, format="PNG", optimize=False)
            dropped.save(drop_path, format="PNG", optimize=False)
    tail = list(image_paths[1:])
    return [str(focus_path), *tail], [str(drop_path), *tail]
