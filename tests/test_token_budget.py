from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.multimodal.token_budget import export_plan, load_plan, wants_extra


class TokenBudgetTests(unittest.TestCase):
    def test_independent_runner_spends_tokens_on_real_samples_and_resumes(self):
        from core.multimodal.runner import EviMemRunner
        from core.multimodal.memory import EvidenceMemory
        from core.multimodal.embeddings import HashMultimodalEmbedder
        from core.multimodal.schema import MultimodalSample, GenerationResponse
        from core.multimodal.token_budget import namespace

        class Model:
            model_name = "fake"
            def __init__(self):
                self.calls = []
            def generate(self, system, user, images, sampling):
                self.calls.append(sampling)
                return GenerationResponse("Final answer: A", prompt_tokens=90, completion_tokens=10)

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / "inference.jsonl"
            source.write_text('{"id":"x","question":"q"}\n')
            config = {"experiment": {"dataset": "scienceqa_img", "data_path": str(source), "seed": 0,
                                     "limit": 1}, "model": {"model_name": "fake"}}
            plan = root / "costs.json"
            plan.write_text(json.dumps({"version": "cost_only_v1", "namespace": namespace(config),
                                       "items": [{"sample_id": "x", "target_tokens": 800}]}))
            config["budget_control"] = {"token_plan": str(plan)}
            sample = MultimodalSample("x", "scienceqa_img", "q", choices=["yes", "no"])
            model = Model()
            def runner():
                return EviMemRunner(model, EvidenceMemory(HashMultimodalEmbedder(16)), root / "run",
                                    method="efficient_budget", config_snapshot=config)
            runner().run([sample])
            row = json.loads((root / "run" / "predictions.jsonl").read_text())
            self.assertEqual(len(model.calls), 8)
            self.assertEqual(len(row["candidate_answers"]), 8)
            self.assertEqual(row["usage_delta"]["prompt_tokens"], 720)
            self.assertFalse(row["shared_certificate"])
            self.assertFalse(row["wrote_memory"])
            self.assertTrue(all(call.output_schema is None for call in model.calls))
            manifest_path = root / "run" / "manifest.json"
            # Simulate a crash between append and the atomic sample checkpoint.
            checkpoint = json.loads(manifest_path.read_text())
            checkpoint["prompt_tokens"] = checkpoint["completion_tokens"] = 0
            manifest_path.write_text(json.dumps(checkpoint))
            runner().run([sample])
            self.assertEqual(len(model.calls), 8)
            manifest = json.loads((root / "run" / "manifest.json").read_text())
            self.assertEqual(manifest["prompt_tokens"] + manifest["completion_tokens"], 800)

    def test_cost_only_plan_namespaces_and_labels_excluded(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            source = root / "inference.jsonl"
            source.write_text('{"id":"x","question":"q"}\n')
            config = {"experiment": {"dataset": "scienceqa_img", "data_path": str(source),
                                     "seed": 0, "limit": 1},
                      "model": {"model_name": "frozen", "revision": "r"}}
            predictions = root / "predictions.jsonl"
            predictions.write_text(json.dumps({"sample_id": "x", "answer": "SECRET_ANSWER",
                "shared_certificate": {"evidence": "SECRET_EVIDENCE"},
                "usage_delta": {"prompt_tokens": 100, "completion_tokens": 20}}) + "\n")
            (root / "manifest.json").write_text(json.dumps({"status": "completed", "processed": 1,
                "method": "shared_budget", "config": config, "prompt_tokens": 100, "completion_tokens": 20}))
            path = root / "costs.json"
            export_plan(predictions, path)
            self.assertNotIn("SECRET", path.read_text())
            self.assertEqual(load_plan(str(path), config, ["x"]), [120])
            changed = json.loads(json.dumps(config))
            changed["experiment"]["seed"] = 1
            with self.assertRaises(ValueError):
                load_plan(str(path), changed, ["x"])
            with self.assertRaises(ValueError):
                load_plan(str(path), config, ["future"])
            with self.assertRaises(FileExistsError):
                export_plan(predictions, path)
            plan = json.loads(path.read_text())
            plan["items"][0]["answer"] = "SECRET"
            path.write_text(json.dumps(plan))
            with self.assertRaises(ValueError):
                load_plan(str(path), config, ["x"])

    def test_more_sampling_only_if_closer_to_budget(self):
        self.assertTrue(wants_extra(600, 1000, 300))
        self.assertFalse(wants_extra(900, 1000, 300))
        self.assertFalse(wants_extra(1100, 1000, 300))
        self.assertFalse(wants_extra(0, 0, 0))

    def test_four_arm_guard_precedes_label_load(self):
        from evaluate.multimodal.compare_typed import compare_four
        with tempfile.TemporaryDirectory() as d:
            paths = {}
            for name, method in (("greedy", "zero_shot"), ("shared_budget", "shared_budget"),
                                 ("shared_evidence", "shared_evidence"), ("efficient_budget", "efficient_budget")):
                root = Path(d) / name
                root.mkdir()
                (root / "manifest.json").write_text(json.dumps({"status": "running", "method": method}))
                paths[name] = root / "predictions.jsonl"
            with patch("evaluate.multimodal.compare_typed.load_local_dataset") as extra, \
                 patch("evaluate.multimodal.compare_causal.load_local_dataset") as original:
                with self.assertRaisesRegex(ValueError, "refusing to read labels"):
                    compare_four(paths, "NEVER_OPEN")
                extra.assert_not_called()
                original.assert_not_called()
            # All shared computations complete, but the fourth arm is still running.
            for name, method in (("greedy", "zero_shot"), ("shared_budget", "shared_budget"),
                                 ("shared_evidence", "shared_evidence")):
                root = paths[name].parent
                (root / "manifest.json").write_text(json.dumps({"status": "completed", "method": method,
                                                               "processed": 1}))
                paths[name].write_text('{"sample_id":"x"}\n')
            with patch("evaluate.multimodal.compare_typed.load_local_dataset") as extra, \
                 patch("evaluate.multimodal.compare_causal.load_local_dataset") as original:
                with self.assertRaisesRegex(ValueError, "Independent budget run incomplete"):
                    compare_four(paths, "NEVER_OPEN", expected_n=1)
                extra.assert_not_called()
                original.assert_not_called()


if __name__ == "__main__":
    unittest.main()
