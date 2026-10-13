from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from core.multimodal.schema import GenerationResponse, MultimodalSample
from core.multimodal.transfer_runner import TransferRunner, lexical_retrieve, parse_rule, RuleEntry


class FakeBackend:
    model_name = "fake-vlm"

    def __init__(self):
        self.calls = []

    def generate(self, system_prompt, user_text, image_paths, sampling):
        self.calls.append((system_prompt, user_text, sampling.seed))
        if "Distill a broadly reusable" in system_prompt:
            content = json.dumps({"positive_rule": "Inspect the current evidence before choosing the matching option.",
                                  "negative_rule": "", "evidence_summary": "uncertain"})
        else:
            content = "Observed: image not required.\nReasoning: choose the matching option.\nFinal answer: A"
        return GenerationResponse(content, 100, 30, latency_seconds=0.01, finish_reason="stop")


class TransferRunnerTests(unittest.TestCase):
    def test_four_arms_temporal_visibility_same_noisy_random_write_pool_and_resume(self):
        samples = [MultimodalSample(str(i), "scienceqa_img", f"same question {i}?", (), ("yes", "no"),
                                    {"question_type": "multi_choice"}) for i in range(3)]
        with tempfile.TemporaryDirectory() as d:
            backend = FakeBackend()
            runner = TransferRunner(backend, samples, Path(d), "source", "revision")
            runner.run()
            rows = [json.loads(line) for line in (Path(d) / "predictions.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 3)
            for i, row in enumerate(rows):
                self.assertEqual(set(row["arms"]), {"none", "noisy", "gated_weighted", "random"})
                self.assertEqual(row["arms"]["none"]["retrieved"], [])
                for arm in ("noisy", "gated_weighted", "random"):
                    self.assertTrue(all(e["sample_index"] < i for e in row["arms"][arm]["retrieved"]))
                self.assertEqual(row["arms"]["noisy"]["written"], row["arms"]["random"]["written"])
                self.assertIsNone(row["arms"]["gated_weighted"]["written"])
            self.assertTrue(rows[1]["arms"]["noisy"]["retrieved"])
            self.assertTrue(rows[1]["arms"]["random"]["retrieved"])
            calls = len(backend.calls)
            TransferRunner(backend, samples, Path(d), "source", "revision").run()
            self.assertEqual(len(backend.calls), calls)
            self.assertEqual(json.loads((Path(d) / "manifest.json").read_text())["status"], "completed")

    def test_rule_parse_rejects_shortcuts_and_malformed_text(self):
        good = json.dumps({"positive_rule": "Check the requested visible relation on the current image.",
                           "negative_rule": "", "evidence_summary": "example"})
        self.assertTrue(parse_rule(good, False, ["attract", "repel"])["valid"])
        self.assertFalse(parse_rule("Final answer: A", False, ["attract", "repel"])["valid"])
        self.assertFalse(parse_rule(good.replace("Check the requested visible relation", "Option A states the relation"),
                                    False, ["attract", "repel"])["valid"])


if __name__ == "__main__":
    unittest.main()
