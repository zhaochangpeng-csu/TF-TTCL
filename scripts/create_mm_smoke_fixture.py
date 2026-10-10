#!/usr/bin/env python3
"""Create a tiny synthetic visual-QA stream for infrastructure smoke tests."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    root = Path("data/multimodal/smoke")
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    rows = []
    specs = [
        ("red_square", "red", "square", "What color is the square?", ["red", "blue"], 0),
        ("blue_circle", "blue", "circle", "Which shape is shown?", ["square", "circle"], 1),
    ]
    for sample_id, color, shape, question, choices, answer in specs:
        path = images / f"{sample_id}.png"
        image = Image.new("RGB", (256, 256), "white")
        draw = ImageDraw.Draw(image)
        if shape == "square":
            draw.rectangle((64, 64, 192, 192), fill=color)
        else:
            draw.ellipse((64, 64, 192, 192), fill=color)
        image.save(path)
        rows.append(
            {
                "id": sample_id,
                "question": question,
                "image": f"images/{path.name}",
                "choices": choices,
                "answer": answer,
            }
        )
    with (root / "smoke.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(root / "smoke.jsonl")


if __name__ == "__main__":
    main()
