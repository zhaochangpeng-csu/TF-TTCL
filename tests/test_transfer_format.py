from __future__ import annotations

import unittest

from core.multimodal.schema import GenerationResponse
from scripts.run_transfer_format_smoke import format_ok


class TransferFormatTests(unittest.TestCase):
    def test_observation_not_required_is_valid_not_an_accuracy_test(self):
        response = GenerationResponse("Observed: Image not required.\nReasoning: Use supplied text.\nFinal answer: A", finish_reason="stop")
        self.assertTrue(format_ok(response, ["yes", "no"]))

    def test_answer_only_truncated_or_out_of_range_fails(self):
        for content, finish in [("Final answer: A", "stop"),
                                ("Observed: x\nReasoning: y\nFinal answer: A", "length"),
                                ("Observed: x\nReasoning: y\nFinal answer: Z", "stop"),
                                ("Observed: x\nReasoning: y\nFinal answer: A\nFinal answer: B", "stop")]:
            self.assertFalse(format_ok(GenerationResponse(content, finish_reason=finish), ["yes", "no"]))


if __name__ == "__main__":
    unittest.main()
