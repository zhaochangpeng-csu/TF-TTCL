from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.inspect_baseline_io import build_report, read_rows, render_markdown, require_label_free, trace_lookup


class BaselineInspectionTests(unittest.TestCase):
    def test_recursive_and_case_insensitive_label_rejection(self):
        for value in ({"answer": "SECRET"}, {"metadata": {"Answer": "SECRET"}}, [{"gold": "SECRET"}]):
            with self.assertRaisesRegex(ValueError, "Forbidden label"):
                require_label_free(value)

    def test_partial_last_prediction_not_in_snapshot(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "rows.jsonl"
            path.write_bytes(b'{"id":"a"}\n{"id":"incomplete"')
            self.assertEqual(read_rows(path), [{"id": "a"}])

    def test_trace_namespaces_and_immutable_request(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            request = {"namespace": {"seed": 0, "dataset": "a"}, "user_text": "q"}
            key = hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            path = root / f"{key}.json"
            path.write_text(json.dumps({"request": request, "response": {"content": "Final answer: A"}}))
            self.assertEqual(trace_lookup(root, request)["provenance"], "recorded_exact")
            self.assertIsNone(trace_lookup(root, {**request, "namespace": {"seed": 1, "dataset": "a"}}))
            path.write_text(json.dumps({"request": {}, "response": {}}))
            with self.assertRaisesRegex(ValueError, "mismatch"):
                trace_lookup(root, request)

    def test_output_answer_not_treated_as_ground_truth(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            inference = root / "inference.jsonl"
            inference.write_text(json.dumps({"id": "x", "question": "q", "choices": ["yes", "no"], "question_type": "multi_choice"}) + "\n")
            # No label file is supplied or opened, including this intentionally invalid one.
            (root / "labels.jsonl").write_text("DO_NOT_READ")
            run = root / "run"
            run.mkdir()
            manifest = {"status": "running", "git_sha": "frozen", "prompt_hash": "p", "seed": 0,
                        "method": "zero_shot", "model": "vlm", "config": {
                            "experiment": {"data_path": str(inference), "dataset": "scienceqa_img"},
                            "model": {"revision": "rev", "max_tokens": 256}}}
            (run / "manifest.json").write_text(json.dumps(manifest))
            row = {"sample_id": "x", "sample_index": 0, "method": "zero_shot", "answer": "a", "response": "Final answer: A",
                   "selection": {}, "usage_delta": {"prompt_tokens": 5, "completion_tokens": 5},
                   "candidate_answers": [{"trajectory": "zero_shot", "answer": "a", "question_variant": "q", "image_paths": [], "image_hashes": []}]}
            (run / "predictions.jsonl").write_text(json.dumps(row) + "\n")
            report = build_report(inference, {"greedy": run}, count=1)
            call = report["cases"][0]["runs"]["greedy"]["candidate_calls"][0]
            self.assertFalse(report["labels_read"])
            self.assertEqual(call["provenance"], "reconstructed_not_wire_capture")
            self.assertIsNone(call["response"])
            self.assertIn("(A) yes", call["request"]["user_text"])
            self.assertNotIn("DO_NOT_READ", render_markdown(report))
            # Reject a future/mismatched sample rather than assigning it this input.
            row["sample_id"] = "future"
            (run / "predictions.jsonl").write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "prefix"):
                build_report(inference, {"greedy": run})


if __name__ == "__main__":
    unittest.main()
