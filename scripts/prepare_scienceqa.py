#!/usr/bin/env python3
"""Convert the official ScienceQA release into the local label-isolated schema."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _question_with_hint(problem: dict[str, Any]) -> str:
    question = str(problem.get("question", "")).strip()
    hint = str(problem.get("hint", "")).strip()
    if hint:
        return f"Context:\n{hint}\n\nQuestion:\n{question}"
    return question


def _canonical_record(sample_id: str, problem: dict[str, Any], image_path: Path | None) -> dict[str, Any]:
    record = {
        "id": str(sample_id),
        "question": _question_with_hint(problem),
        "image": str(image_path.resolve()) if image_path is not None else None,
        "choices": list(problem.get("choices", [])),
        "answer": problem.get("answer"),
        "question_type": "multi_choice",
        "grade": problem.get("grade"),
        "subject": problem.get("subject"),
        "topic": problem.get("topic"),
        "category": problem.get("category"),
        "skill": problem.get("skill"),
    }
    return {key: value for key, value in record.items() if value is not None}


def convert_scienceqa(
    problems_path: Path,
    images_root: Path,
    output_path: Path,
    split: str,
    image_only: bool = True,
) -> dict[str, Any]:
    problems = json.loads(problems_path.read_text(encoding="utf-8"))
    if not isinstance(problems, dict):
        raise ValueError("Official ScienceQA problems.json must contain an object")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    skipped_without_image = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for sample_id, problem in problems.items():
            if str(problem.get("split", "")) != split:
                continue
            image_name = problem.get("image")
            if not image_name:
                skipped_without_image += 1
                if image_only:
                    continue
            image_path = images_root / split / str(sample_id) / str(image_name) if image_name else None
            if image_path is not None and not image_path.is_file():
                raise FileNotFoundError(f"Missing ScienceQA image: {image_path}")

            record = _canonical_record(str(sample_id), problem, image_path)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    return {
        "source": "lupantech/ScienceQA",
        "source_problems": str(problems_path.resolve()),
        "source_problems_sha256": sha256_file(problems_path),
        "images_root": str(images_root.resolve()),
        "split": split,
        "image_only": image_only,
        "count": written,
        "skipped_without_image": skipped_without_image,
        "output": str(output_path.resolve()),
    }


def convert_modelscope_scienceqa_export(
    problems_path: Path,
    export_path: Path,
    output_path: Path,
    split: str,
) -> dict[str, Any]:
    """Attach official IDs/metadata to an image-bearing ModelScope export.

    AI-ModelScope/ScienceQA contains the official content and images but omits
    the original problem IDs.  Its rows follow the official split order.  We
    rely on that order only after validating every row's question, choices,
    answer, and image presence against ``problems.json``.
    """
    problems = json.loads(problems_path.read_text(encoding="utf-8"))
    if not isinstance(problems, dict):
        raise ValueError("Official ScienceQA problems.json must contain an object")

    official_rows = [
        (str(sample_id), problem)
        for sample_id, problem in problems.items()
        if str(problem.get("split", "")) == split and problem.get("image")
    ]
    exported_rows = [
        json.loads(line)
        for line in export_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(exported_rows) != len(official_rows):
        raise ValueError(
            f"ModelScope export has {len(exported_rows)} image rows; "
            f"official {split} split has {len(official_rows)}"
        )

    canonical_rows: list[dict[str, Any]] = []
    for position, ((sample_id, problem), exported) in enumerate(zip(official_rows, exported_rows)):
        expected = (
            str(problem.get("question", "")).strip(),
            list(problem.get("choices", [])),
            problem.get("answer"),
        )
        observed = (
            str(exported.get("question", "")).strip(),
            list(exported.get("choices", [])),
            exported.get("answer"),
        )
        if observed != expected:
            raise ValueError(
                f"ModelScope row {position} does not match official ScienceQA ID {sample_id}"
            )
        image_value = exported.get("image")
        if not image_value:
            raise ValueError(f"ModelScope row {position} has no image path")
        image_path = Path(str(image_value))
        if not image_path.is_absolute():
            image_path = export_path.parent / image_path
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing ModelScope-exported image: {image_path}")
        canonical_rows.append(_canonical_record(sample_id, problem, image_path))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        for record in canonical_rows:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary_path.replace(output_path)

    return {
        "source": "lupantech/ScienceQA with AI-ModelScope/ScienceQA images",
        "source_problems": str(problems_path.resolve()),
        "source_problems_sha256": sha256_file(problems_path),
        "source_export": str(export_path.resolve()),
        "source_export_sha256": sha256_file(export_path),
        "split": split,
        "image_only": True,
        "mapping_validation": "full_positional_content_match",
        "count": len(canonical_rows),
        "output": str(output_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--images-root")
    source.add_argument(
        "--modelscope-export",
        help="Image-bearing JSONL made by prepare_multimodal_dataset.py",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument("--include-text-only", action="store_true")
    parser.add_argument("--manifest")
    args = parser.parse_args()

    if args.modelscope_export:
        if args.include_text_only:
            parser.error("--include-text-only cannot be used with --modelscope-export")
        manifest = convert_modelscope_scienceqa_export(
            Path(args.problems),
            Path(args.modelscope_export),
            Path(args.output),
            args.split,
        )
    else:
        manifest = convert_scienceqa(
            Path(args.problems),
            Path(args.images_root),
            Path(args.output),
            args.split,
            image_only=not args.include_text_only,
        )
    manifest_path = Path(args.manifest) if args.manifest else Path(args.output).with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
