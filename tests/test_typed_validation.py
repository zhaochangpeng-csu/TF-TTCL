from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.check_typed_validation import check_stream
from scripts.create_label_isolated_subset import create_subset


class TypedValidationTests(unittest.TestCase):
    def test_complete_label_isolation_without_opening_label_file(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            Image.new("RGB", (10, 10), "white").save(root / "x.png")
            source = root / "source.jsonl"
            source.write_text(json.dumps({"id": "x", "question": "q", "image_paths": ["x.png"],
                                          "choices": ["yes", "no"], "answer": "SECRET"}) + "\n")
            inference, labels, manifest = (root / name for name in ("inference.jsonl", "labels.jsonl", "manifest.json"))
            create_subset(source, inference, labels, manifest, 1, 20261013)
            # Deliberately invalid label file: checker must not parse it.
            labels.write_text("DO_NOT_READ")
            result = check_stream(manifest, 1)
            self.assertFalse(result["labels_read"])
            self.assertNotIn("SECRET", inference.read_text())
            row = json.loads(inference.read_text())
            row["metadata"] = {"answer": "LEAK"}
            inference.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "Forbidden label"):
                check_stream(manifest, 1)

    def test_configuration_preserves_frozen_algorithm(self):
        import yaml
        root = Path(__file__).resolve().parents[1]
        train = yaml.safe_load((root / "core/config/mm_scienceqa_train_typed.yaml").read_text())
        validation = yaml.safe_load((root / "core/config/mm_scienceqa_validation_typed.yaml").read_text())
        for section in ("model", "evidence", "memory", "ablation"):
            self.assertEqual(train[section], validation[section])
        for key in ("dataset", "seed", "order", "method"):
            self.assertEqual(train["experiment"][key], validation["experiment"][key])


if __name__ == "__main__":
    unittest.main()
