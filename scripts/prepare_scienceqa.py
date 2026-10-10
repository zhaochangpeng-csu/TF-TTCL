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

            question = str(problem.get("question", "")).strip()
            hint = str(problem.get("hint", "")).strip()
            if hint:
                question = f"Context:\n{hint}\n\nQuestion:\n{question}"
            record = {
                "id": str(sample_id),
                "question": question,
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
            handle.write(
                json.dumps(
                    {key: value for key, value in record.items() if value is not None},
                    ensure_ascii=False,
                )
                + "\n"
            )
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problems", required=True)
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument("--include-text-only", action="store_true")
    parser.add_argument("--manifest")
    args = parser.parse_args()

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
