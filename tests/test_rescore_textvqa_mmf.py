import os
from pathlib import Path
import tempfile
import unittest

from scripts.rescore_textvqa_mmf import load_evaluator, score_one, score_records


class PinnedMMFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = os.environ.get('MMF_EVALUATOR_PATH')
        if not path:
            raise unittest.SkipTest('Set MMF_EVALUATOR_PATH to the SHA-pinned upstream file')
        cls.evaluator = load_evaluator(Path(path))

    def test_upstream_rejects_modified_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'modified.py'
            path.write_text('pass\n')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                load_evaluator(path)

    def test_punctuation_digits_and_contractions(self):
        pairs = [('1,000', '1000'), ("can't", 'cant'),
                 ('red/blue', 'red blue'), ('twenty-one', 'twenty one'),
                 ('blue!', 'blue'), ('two', '2')]
        for prediction, truth in pairs:
            with self.subTest(prediction=prediction):
                self.assertEqual(score_one(self.evaluator, prediction, [truth] * 10), 1.0)

    def test_leave_one_out_and_ten_answer_requirement(self):
        for count, expected in enumerate((0, 0.3, 0.6, 0.9, 1.0)):
            value = score_one(self.evaluator, 'yes', ['yes'] * count + ['no'] * (10-count))
            self.assertAlmostEqual(value, expected)
        with self.assertRaises(ValueError):
            score_one(self.evaluator, 'yes', ['yes'] * 9)

    def test_raw_selected_response_is_distinct_from_old_normalization(self):
        arm = {'positive_role': 'teacher', 'answer': 'redblue',
               'calls': [{'role': 'teacher', 'normalized_answer': 'redblue',
                          'response': {'content': 'Reasoning: Visible text.\nFinal answer: red/blue'}}]}
        row = {'arms': {'select_no_memory': arm, 'tf_ttcl_mm_direct': arm}}
        summary, contrasts = score_records(([row], [row]), [['red blue'] * 10], self.evaluator)
        self.assertEqual(summary['raw_extracted']['baseline_no_memory']['mean'], 1.0)
        self.assertEqual(summary['recorded_normalized']['baseline_no_memory']['mean'], 0.0)
        self.assertEqual(contrasts['raw_extracted']['interaction']['mean_and_95pct_ci'][0], 0)


if __name__ == '__main__':
    unittest.main()
