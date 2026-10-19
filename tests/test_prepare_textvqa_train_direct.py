import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from scripts.prepare_textvqa_train_direct import prepare


class TextVQATrainDirectTests(unittest.TestCase):
    def test_extracts_selected_images_and_isolates_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotation = root / "train.json"
            rows = [{"question_id": i, "image_id": f"image{i}", "set_name": "train",
                     "question": f"What is written in image {i}?", "answers": [str(i)] * 10}
                    for i in range(3)]
            annotation.write_text(json.dumps({"data": rows}))
            image_zip = root / "images.zip"
            with zipfile.ZipFile(image_zip, "w") as archive:
                for i in range(3):
                    image = root / f"image{i}.jpg"
                    Image.new("RGB", (2, 2), "red").save(image)
                    archive.write(image, f"train_images/image{i}.jpg")
            output = root / "export"
            manifest = prepare(annotation, image_zip, output, count=2, seed=5,
                               expected_count=3, expected_images=3)
            public = [json.loads(line) for line in (output / "inference.jsonl").read_text().splitlines()]
            private = [json.loads(line) for line in (output / "labels.jsonl").read_text().splitlines()]
            self.assertEqual(manifest["split"], "train")
            self.assertEqual(len(public), len(private))
            self.assertEqual(len(public), 2)
            self.assertTrue(all("answers" not in row for row in public))
            self.assertTrue(all(len(row["answers"]) == 10 for row in private))
            self.assertEqual(len(list((output / "images").glob("*.jpg"))), 2)
            with self.assertRaises(FileExistsError):
                prepare(annotation, image_zip, output, count=2,
                        expected_count=3, expected_images=3)

    def test_rejects_previous_validation_image_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotation = root / "train.json"
            annotation.write_text(json.dumps({"data": [{"question_id": 1, "image_id": "same",
                "set_name": "train", "question": "What is visible?", "answers": ["x"] * 10}]}))
            image_zip = root / "images.zip"
            with zipfile.ZipFile(image_zip, "w") as archive:
                archive.writestr("train_images/same.jpg", b"not needed for overlap check")
            previous = root / "previous_val" / "images"
            previous.mkdir(parents=True)
            (previous / "same.jpg").write_bytes(b"previous")
            with self.assertRaisesRegex(ValueError, "overlaps"):
                prepare(annotation, image_zip, root / "output", count=1,
                        expected_count=1, expected_images=1, previous_val=previous.parent)


if __name__ == "__main__":
    unittest.main()
