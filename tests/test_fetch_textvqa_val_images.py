import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from scripts.fetch_textvqa_val_images import fetch


class FetchTextVQAImagesTests(unittest.TestCase):
    def test_selective_zip_fetch_and_reject_existing_complete_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotation = root / "val.json"
            annotation.write_text(json.dumps({"data": [
                {"question_id": 1, "image_id": "abc", "set_name": "val",
                 "question": "What is written?", "answers": ["STOP"] * 10},
                {"question_id": 2, "image_id": "abc", "set_name": "val",
                 "question": "Read the sign", "answers": ["STOP"] * 10},
            ]}))
            image = io.BytesIO()
            Image.new("RGB", (8, 8), "white").save(image, format="JPEG")
            archive = root / "images.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("train_images/abc.jpg", image.getvalue())
                handle.writestr("train_images/unneeded.jpg", image.getvalue())
            output = root / "export"
            manifest = fetch(annotation, output, str(archive), expected_count=2, expected_images=1)
            self.assertEqual((manifest["count"], manifest["image_count"]), (2, 1))
            self.assertEqual(len(list((output / "images").iterdir())), 1)
            self.assertEqual(len((output / "textvqa_val.jsonl").read_text().splitlines()), 2)
            self.assertFalse((output / ".fetch_in_progress.json").exists())
            with self.assertRaises(FileExistsError):
                fetch(annotation, output, str(archive), expected_count=2, expected_images=1)


if __name__ == "__main__":
    unittest.main()
