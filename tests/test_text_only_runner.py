from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from core.multimodal.schema import GenerationResponse, MultimodalSample
from core.multimodal.text_only_runner import TextOnlyRunner, choose_vote, safe_variants
from evaluate.multimodal.compare_text_only import audit_text_only, declared_subgroups, _holm_adjust


class FakeBackend:
    model_name = "fake-vlm"

    def __init__(self):
        self.calls = []

    def generate(self, system, user, images, sampling):
        self.calls.append((system, user, tuple(images), sampling.seed))
        if not images:
            return GenerationResponse(
                content=json.dumps({"questions": ["What is the name of the country that is highlighted?",
                                                  "What country is highlighted in the picture?"]}),
                prompt_tokens=10, completion_tokens=10, finish_reason="stop")
        answer = "B" if ("name of the country" in user or
                         "highlighted in the picture" in user) else "A"
        return GenerationResponse(
            content=f"Observed: A marked region is visible.\nReasoning: Compare the options.\nFinal answer: {answer}",
            prompt_tokens=100, completion_tokens=20, finish_reason="stop")


class TextOnlyRunnerTests(unittest.TestCase):
    def test_declared_validation_slices_are_label_blind(self):
        self.assertEqual(declared_subgroups("Which country is highlighted?"), {"highlighted_map"})
        self.assertEqual(declared_subgroups("Context:\nA map.\n\nQuestion:\nWhich state is highlighted?"),
                         {"highlighted_map"})
        self.assertEqual(declared_subgroups("Which magnet attracts this object?"), {"magnet"})
        self.assertEqual(declared_subgroups("What is the capital of France?"), {"capital"})
        self.assertEqual(declared_subgroups("Which country is shaded?"), set())

    def test_holm_adjustment(self):
        self.assertEqual(_holm_adjust({"a": .01, "b": .04, "c": .03}),
                         {"a": .03, "c": .06, "b": .06})

    def test_rewrite_fallback_and_vote(self):
        variants, changed = safe_variants("Which country is highlighted?", "not json")
        self.assertEqual(variants, ["Which country is highlighted?"] * 2)
        self.assertEqual(changed, [False, False])
        _, changed = safe_variants("Which country is highlighted?",
                                   '{"questions":["Which animal is highlighted?",'
                                   '"What country is highlighted in the picture?"]}')
        self.assertEqual(changed, [False, True])
        _, changed = safe_variants("Which country is highlighted?",
                                   '{"questions":["Which country is highlighted... ?",'
                                   '"Which country is highlighted, Grenada?"]}')
        self.assertEqual(changed, [False, False])
        self.assertEqual(choose_vote([{"answer": "a", "format_ok": True},
                                      {"answer": "b", "format_ok": True},
                                      {"answer": "b", "format_ok": True}]), "b")

    def test_all_arms_share_image_and_pairing_is_auditable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.png"
            image.write_bytes(b"fixed-image-bytes")
            sample = MultimodalSample("1", "scienceqa_img", "Which country is highlighted?",
                                      [str(image)], ["Aland", "Bland"], {"question_type": "multi_choice"})
            backend = FakeBackend()
            runner = TextOnlyRunner(backend, [sample], root / "run", "input-hash", "fake-revision")
            runner.run()
            self.assertEqual(len(backend.calls), 6)
            manifest, rows = audit_text_only(root / "run", 1)
            self.assertFalse(manifest["labels_read"])
            self.assertEqual(rows[0]["arms"]["greedy"]["answer"], "a")
            self.assertEqual(rows[0]["arms"]["same_text_vote"]["answer"], "a")
            self.assertEqual(rows[0]["arms"]["text_aug_vote"]["answer"], "b")
            self.assertEqual(len({tuple(candidate["image_hashes"]) for candidate in rows[0]["candidates"]}), 1)
            runner.run()
            self.assertEqual(len(backend.calls), 6)
            rows[0]["candidates"][4]["image_hashes"] = ["wrong-image"]
            (root / "run" / "predictions.jsonl").write_text(json.dumps(rows[0]) + "\n")
            with self.assertRaisesRegex(ValueError, "contract broken"):
                audit_text_only(root / "run", 1)


if __name__ == "__main__":
    unittest.main()
