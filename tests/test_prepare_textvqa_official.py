import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

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


class TextVQAImportTests(unittest.TestCase):
    def test_import_and_isolate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotation, archive = _fixture(root)
            export = root / "export"
            manifest = prepare(annotation, archive, export, expected_count=2, expected_images=1)
            self.assertEqual((manifest["count"], manifest["image_count"]), (2, 1))
            self.assertEqual(len(list((export / "images").iterdir())), 1)
            subset = export / "isolated"
            create_subset(export / "textvqa_val.jsonl", subset.with_suffix(".inference.jsonl"),
                          subset.with_suffix(".labels.jsonl"), subset.with_suffix(".manifest.json"),
                          2, 20261015)
            inference = load_local_dataset(str(subset.with_suffix(".inference.jsonl")), "textvqa", str(export))
            labels = load_local_dataset(str(subset.with_suffix(".labels.jsonl")), "textvqa", str(export))
            self.assertTrue(all(row.target is None for row in inference))
            self.assertTrue(all(row.target and len(row.target.answers) == 10 for row in labels))
            self.assertTrue(all(Path(row.sample.image_paths[0]).is_file() for row in inference))
            with self.assertRaises(FileExistsError):
                prepare(annotation, archive, export, expected_count=2, expected_images=1)

    def test_incomplete_archive_does_not_leave_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotation, archive = _fixture(root, missing_image=True)
            export = root / "export"
            with self.assertRaises(FileNotFoundError):
                prepare(annotation, archive, export, expected_count=2, expected_images=1)
            self.assertFalse(export.exists())


if __name__ == "__main__":
    unittest.main()
