import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.score_textvqa_shadow_mmf import ANCHORS, ARMS, _selected_texts, score_experiment, score_records


class FakeEvaluator:
    def eval_pred_list(self, data):
        prediction = data[0]['pred_answer']
        return float(prediction == data[0]['gt_answers'][0])


def fake_contrast(values):
    return {'mean_and_95pct_ci': [sum(values)/len(values), 0, 0], 'paired_sign_randomization_p': 1}


class ScoreTests(unittest.TestCase):
    def test_raw_final_extraction_is_equal_across_all_arms(self):
        texts = {k: ['Reasoning: hello\nFinal answer: red/blue'] for k in (*ANCHORS, *ARMS)}
        with patch('scripts.score_textvqa_shadow_mmf.contrast', fake_contrast):
            results, contrasts = score_records(texts, [['red/blue'] * 10], FakeEvaluator())
        self.assertTrue(all(v['mmf_vqa_accuracy'] == 1 for v in results.values()))
        self.assertEqual(contrasts['interaction_real_vs_neutral_position']['mean_and_95pct_ci'][0], 0)
        self.assertEqual(contrasts['real_before_minus_anchor_memory']['unchanged_items'], 1)

    def test_main_effects_are_paired_and_interaction_has_correct_sign(self):
        truths = [['yes'] * 10] * 2
        texts = {name: ['Final answer: no', 'Final answer: no'] for name in (*ANCHORS, *ARMS)}
        texts['real_before'] = ['Final answer: yes', 'Final answer: no']
        texts['real_after'] = ['Final answer: yes', 'Final answer: yes']
        texts['neutral_before'] = ['Final answer: no', 'Final answer: no']
        texts['neutral_after'] = ['Final answer: no', 'Final answer: yes']
        with patch('scripts.score_textvqa_shadow_mmf.contrast', fake_contrast):
            _, c = score_records(texts, truths, FakeEvaluator())
        self.assertEqual(c['real_minus_neutral_average']['mean_and_95pct_ci'][0], .5)
        self.assertEqual(c['after_minus_before_average']['mean_and_95pct_ci'][0], .5)
        self.assertEqual(c['interaction_real_vs_neutral_position']['mean_and_95pct_ci'][0], 0)
        texts['neutral_after'] = ['Final answer: no', 'Final answer: no']
        with patch('scripts.score_textvqa_shadow_mmf.contrast', fake_contrast):
            _, c = score_records(texts, truths, FakeEvaluator())
        self.assertEqual(c['interaction_real_vs_neutral_position']['mean_and_95pct_ci'][0], .5)

    def test_selected_role_must_match_recorded_answer(self):
        call = {'role': 'teacher', 'normalized_answer': 'red',
                'response': {'content': 'Final answer: red'}}
        arm = {'positive_role': 'teacher', 'answer': 'red', 'calls': [call]}
        anchor = {'sample_index': 0, 'sample_id': 'one', 'arms': {k: arm for k in ANCHORS.values()}}
        shadow = {'sample_index': 0, 'sample_id': 'one', 'arms': {k: arm for k in ARMS}}
        self.assertEqual(_selected_texts([anchor], [shadow])['real_after'], ['Final answer: red'])
        shadow['sample_id'] = 'two'
        with self.assertRaises(ValueError):
            _selected_texts([anchor], [shadow])

    def test_smoke_cannot_score_even_if_labels_exist(self):
        with self.assertRaisesRegex(ValueError, 'smoke'):
            score_experiment(SimpleNamespace(limit=20), None, None, None)

    def test_failed_full_audit_happens_before_label_open(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inference = root / 'inference.jsonl'
            inference.write_text('no labels')
            subset = root / 'manifest.json'
            ids = [str(i) for i in range(400)]
            subset.write_text(json.dumps({'dataset_id': 'official/TextVQA_0.5.1', 'split': 'train', 'count': 400,
                'inference_sha256': hashlib.sha256(inference.read_bytes()).hexdigest(),
                'labels_sha256': 'a'*64, 'selected_ids': ids}))
            args = SimpleNamespace(limit=400, inference=inference, subset_manifest=subset,
                subset_manifest_sha256=hashlib.sha256(subset.read_bytes()).hexdigest(),
                labels=root / 'does-not-exist-labels.jsonl', mmf_evaluator=root / 'mmf.py',
                anchor=root / 'anchor', shadow=root / 'shadow', anchor_predictions_sha256='b'*64,
                anchor_manifest_sha256='c'*64, shard_count=1)
            anchors = [{'sample_id': x} for x in ids]
            with patch('scripts.score_textvqa_shadow_mmf.checked_anchor', return_value=({}, anchors, [])), \
                 patch('scripts.score_textvqa_shadow_mmf.audit_shards', side_effect=ValueError('audit failed')):
                with self.assertRaisesRegex(ValueError, 'audit failed'):
                    score_experiment(args, None, None, None)


if __name__ == '__main__':
    unittest.main()
