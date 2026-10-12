from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.multimodal.shared_evidence import (
    assess_certificate, execute_operator, parse_observation, route_operator,
)


def magnet(poles):
    opposite = {"N": "S", "S": "N"}
    letters = [opposite[poles[0]], *poles, opposite[poles[1]]] if poles else []
    return parse_observation(json.dumps({"items": [
        {"text": letter, "x": 100 + 250 * i, "y": 500} for i, letter in enumerate(letters)]}), "facing_poles")


class SharedOperatorTests(unittest.TestCase):
    def test_routes_without_labels(self):
        self.assertEqual(route_operator("Will these magnets attract or repel?", ["repel", "attract"]), "facing_poles")
        self.assertEqual(route_operator("During this time, thermal energy was transferred from () to ().",
            ["each vial . . . the surroundings", "the surroundings . . . each vial"]), "temperature_delta")
        self.assertIsNone(route_operator("Which country is highlighted?", ["Samoa", "Kiribati"]))
        self.assertIsNone(route_operator("Magnets nearby, which weather?", ["climate", "weather"]))

    def test_same_evidence_falsifies_opposite_magnet_relation(self):
        for poles, relation in [(["N", "N"], "repel"), (["S", "S"], "repel"),
                                (["N", "S"], "attract"), (["S", "N"], "attract")]:
            for choices in (["attract", "repel"], ["repel", "attract"]):
                result = execute_operator("facing_poles", {"facing_poles": poles}, choices)
                self.assertEqual(result["answer"], chr(97 + choices.index(relation)))

    def test_positional_ocr_is_order_independent_and_rejects_bad_layouts(self):
        items = [{"text": letter, "x": 100 + 250 * i, "y": 500}
                 for i, letter in enumerate(["S", "N", "N", "S"])]
        result = parse_observation(json.dumps({"items": list(reversed(items))}), "facing_poles")
        self.assertEqual(result["facts"]["facing_poles"], ["N", "N"])
        items[1]["y"] = 900
        self.assertFalse(parse_observation(json.dumps({"items": items}), "facing_poles")["visible"])
        items[1]["y"] = 500
        items[1]["x"] = items[0]["x"]
        self.assertFalse(parse_observation(json.dumps({"items": items}), "facing_poles")["visible"])
        items[1]["x"] = 350
        items[1]["text"] = "A"
        observation = parse_observation(json.dumps({"items": items}), "facing_poles")
        self.assertFalse(observation["visible"])
        self.assertFalse(observation["parse_error"])

    def test_table_deltas_and_mixed_abstention(self):
        choices = ["each vial . . . the surroundings", "the surroundings . . . each vial"]
        for final, answer in [(4.9, "b"), (1.0, "a"), (2.2, None)]:
            result = execute_operator("temperature_delta", {"rows": [{"initial": 2.2, "final": final}]}, choices)
            self.assertEqual(result["answer"], answer)
        self.assertIsNone(execute_operator("temperature_delta", {"rows": [
            {"initial": 2, "final": 3}, {"initial": 2, "final": 1}]}, choices)["answer"])

    def test_parser_fails_closed_extra_verdicts_and_bad_types(self):
        for payload in [{"items": [], "answer": "repel"},
                        {"items": [{"text": "unknown", "x": 100, "y": 100}]},
                        {"items": [{"text": "N", "x": True, "y": 100}]},
                        {"items": [{"text": "N", "x": 100, "y": 1001}]}, []]:
            self.assertTrue(parse_observation(json.dumps(payload), "facing_poles")["parse_error"])
        for value in (True, "2.2", float("nan"), float("inf")):
            payload = {"visible": True, "unit": "C", "rows": [{"initial": value, "final": 3}]}
            self.assertTrue(parse_observation(json.dumps(payload), "temperature_delta")["parse_error"])

    def test_blank_mismatch_stability_and_candidate_recall(self):
        facts, blank, other = magnet(["N", "N"]), magnet([]), magnet(["N", "S"])
        accepted = assess_certificate("facing_poles", [facts, facts, blank, other],
                                      ["attract", "repel"], ["a", "b"])
        self.assertTrue(accepted["gate_passed"])
        self.assertEqual(accepted["computed_answer"], "b")
        for observations in ([facts, other, blank], [facts, facts, facts],
                             [facts, facts, blank, facts],
                             [facts, facts, {"parse_error": True}]):
            self.assertFalse(assess_certificate("facing_poles", list(observations),
                ["attract", "repel"], ["a", "b"])["gate_passed"])
        self.assertFalse(assess_certificate("facing_poles", [facts, facts, blank],
            ["attract", "repel"], ["a"])["gate_passed"])


class SharedRunnerTests(unittest.TestCase):
    def test_real_calls_have_no_question_candidates_or_options(self):
        from core.multimodal.runner import EviMemRunner
        from core.multimodal.memory import EvidenceMemory
        from core.multimodal.embeddings import HashMultimodalEmbedder
        from core.multimodal.schema import MultimodalSample, Candidate, GenerationResponse
        from PIL import Image

        class Recorder:
            model_name = "fake"
            calls = []
            def generate(self, system, text, images, sampling):
                self.calls.append((system, text))
                absent = "blank" in images[0]
                return GenerationResponse(json.dumps({"items": [] if absent else [
                    {"text": letter, "x": 100 + i * 250, "y": 500}
                    for i, letter in enumerate(["S", "N", "N", "S"])]}))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real, blank = root / "real.png", root / "blank.png"
            Image.new("RGB", (10, 10)).save(real)
            Image.new("RGB", (10, 10), "white").save(blank)
            sample = MultimodalSample("x", "scienceqa_img",
                "SECRET_CONTEXT\n\nQuestion:\nWill these magnets attract or repel?",
                [str(real)], ["attract", "repel"])
            model = Recorder()
            runner = EviMemRunner(model, EvidenceMemory(HashMultimodalEmbedder(16)),
                root / "run", method="shared_evidence")
            candidates = [Candidate("g", "original_greedy", GenerationResponse("A"), "a", "q", ()),
                          Candidate("s", "original_sampled", GenerationResponse("B"), "b", "q", ())]
            decision, cert = runner._shared_adjudication(sample, candidates, 0, [str(real)], [str(blank)], ())
            self.assertEqual(decision.answer, "b")
            self.assertTrue(cert["changed_from_greedy"])
            self.assertFalse(decision.eligible_for_memory)
            self.assertEqual(len(model.calls), 3)
            for system, text in model.calls:
                self.assertNotIn("SECRET_CONTEXT", system + text)
                self.assertNotIn("Will these magnets", system + text)
                self.assertNotIn("RESPONSE_X", system + text)
                self.assertNotIn("Options:", system + text)
                self.assertNotIn('"N"', system + text)
                self.assertNotIn('"S"', system + text)

    def test_shared_labels_not_opened_before_all_complete(self):
        from evaluate.multimodal.compare_causal import compare_runs
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for name, method in [("greedy", "zero_shot"), ("shared_budget", "shared_budget"),
                                 ("shared_evidence", "shared_evidence")]:
                root = Path(directory) / name
                root.mkdir()
                (root / "manifest.json").write_text(json.dumps({"status": "running", "method": method}))
                paths[name] = root / "predictions.jsonl"
            with patch("evaluate.multimodal.compare_causal.load_local_dataset") as loader:
                with self.assertRaisesRegex(ValueError, "refusing to read labels"):
                    compare_runs(paths, "NEVER_OPEN", "scienceqa_img", family="shared")
                loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
