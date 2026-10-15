import io
import json
import zipfile
from pathlib import Path

import pytest
from PIL import Image

from core.multimodal.datasets import load_local_dataset
from scripts.create_label_isolated_subset import create_subset
from scripts.prepare_textvqa_official import prepare


def _fixture(tmp_path: Path, *, missing_image: bool = False):
    image = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(image, format="JPEG")
    annotation = tmp_path / "TextVQA_0.5.1_val.json"
    annotation.write_text(json.dumps({"data": [
        {"question_id": 1, "image_id": "abc", "set_name": "val",
         "question": "What word is on the sign?", "answers": ["STOP"] * 10},
        {"question_id": 2, "image_id": "abc", "set_name": "val",
         "question": "What is written?", "answers": ["STOP"] * 10},
    ]}), encoding="utf-8")
    archive = tmp_path / "train_val_images.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        if not missing_image:
            handle.writestr("train_images/abc.jpg", image.getvalue())
        handle.writestr("train_images/unneeded.jpg", image.getvalue())
    return annotation, archive


def test_import_and_isolate(tmp_path):
    annotation, archive = _fixture(tmp_path)
    export = tmp_path / "export"
    manifest = prepare(annotation, archive, export, expected_count=2, expected_images=1)
    assert manifest["count"] == 2 and manifest["image_count"] == 1
    assert len(list((export / "images").iterdir())) == 1
    subset = export / "isolated"
    create_subset(export / "textvqa_val.jsonl", subset.with_suffix(".inference.jsonl"),
                  subset.with_suffix(".labels.jsonl"), subset.with_suffix(".manifest.json"),
                  2, 20261015)
    inference = load_local_dataset(str(subset.with_suffix(".inference.jsonl")), "textvqa", str(export))
    labels = load_local_dataset(str(subset.with_suffix(".labels.jsonl")), "textvqa", str(export))
    assert all(row.target is None for row in inference)
    assert all(row.target and len(row.target.answers) == 10 for row in labels)
    assert all(Path(row.sample.image_paths[0]).is_file() for row in inference)
    with pytest.raises(FileExistsError):
        prepare(annotation, archive, export, expected_count=2, expected_images=1)


def test_incomplete_archive_does_not_leave_export(tmp_path):
    annotation, archive = _fixture(tmp_path, missing_image=True)
    export = tmp_path / "export"
    with pytest.raises(FileNotFoundError):
        prepare(annotation, archive, export, expected_count=2, expected_images=1)
    assert not export.exists()
