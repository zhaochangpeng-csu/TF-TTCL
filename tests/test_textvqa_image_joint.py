import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from core.multimodal.datasets import load_local_dataset
from core.multimodal.image_ops import make_textvqa_photometric_views, sha256_file
from core.multimodal.schema import GenerationResponse
from core.multimodal.textvqa_image_runner import TextVQAImageRunner
from core.multimodal.textvqa_text_runner import TextVQATextRunner
from evaluate.multimodal.compare_textvqa_image_joint import audit as audit_image
from evaluate.multimodal.compare_textvqa_text import audit as audit_text


class FakeBackend:
    model_name = "fake-vlm"

    def generate(self, system, user, images, sampling):
        if not images:
            return GenerationResponse(content='{"questions":["What word is visible on the sign?",'
                                              '"Which word is on the sign?"]}', finish_reason="stop")
        return GenerationResponse(content="Observed: The sign says STOP.\n"
                                          "Reasoning: Read the sign.\nFinal answer: STOP",
                                  prompt_tokens=10, completion_tokens=15, finish_reason="stop")


class TextVQAImageJointTests(unittest.TestCase):
    def test_views_preserve_geometry_are_deterministic_and_reject_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            image = Image.new("RGB", (120, 60), "white")
            ImageDraw.Draw(image).text((5, 5), "STOP", fill="black")
            image.save(source)
            first = make_textvqa_photometric_views([str(source)], root / "views", "id")
            second = make_textvqa_photometric_views([str(source)], root / "views", "id")
            self.assertEqual(first, second)
            for variant in first:
                with Image.open(variant[0]) as opened:
                    self.assertEqual(opened.size, image.size)
            Image.new("RGB", image.size, "red").save(first[0][0])
            with self.assertRaisesRegex(ValueError, "differs from fixed transform"):
                make_textvqa_photometric_views([str(source)], root / "views", "id")

    def test_pairing_audit_and_no_label_field(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.png"
            Image.new("RGB", (48, 48), "white").save(image)
            inference = root / "inference.jsonl"
            inference.write_text(json.dumps({"id": "1", "question": "What word is on the sign?",
                                             "image": "image.png", "choices": []}) + "\n")
            sample = load_local_dataset(str(inference), "textvqa")[0].sample
            source_sha = hashlib.sha256(inference.read_bytes()).hexdigest()
            baseline = root / "baseline"
            TextVQATextRunner(FakeBackend(), [sample], baseline, source_sha, "fake-revision").run()
            manifest, base_rows = audit_text(baseline, inference, 1)
            run = root / "run"
            runner = TextVQAImageRunner(FakeBackend(), [sample], run, source_sha,
                                        "fake-revision", manifest, base_rows,
                                        sha256_file(baseline / "predictions.jsonl"))
            runner.run()
            _, rows, _ = audit_image(run, baseline, inference, 1)
            self.assertEqual(rows[0]["arms"]["image_only_vote"]["answer"], "stop")
            self.assertEqual(rows[0]["arms"]["joint_vote"]["answer"], "stop")
            self.assertNotIn("answers", rows[0])
            runner.run()  # Completed resume is idempotent.
            rows[0]["new_candidates"][2]["image_hashes"] = ["bad"]
            (run / "predictions.jsonl").write_text(json.dumps(rows[0]) + "\n")
            with self.assertRaisesRegex(ValueError, "candidate contract broken"):
                audit_image(run, baseline, inference, 1)


if __name__ == "__main__":
    unittest.main()
