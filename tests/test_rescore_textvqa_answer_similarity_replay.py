import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.create_label_isolated_subset import sha256_file
from scripts.rescore_textvqa_answer_similarity_replay import rescore, score_records


class DummyEvaluator:
    def eval_pred_list(self, items):
        item = items[0]
        return float(item['pred_answer'] == item['gt_answers'][0])


class RescoreReplayTests(unittest.TestCase):
    def setUp(self):
        self.calls = [{'role': r, 'normalized_answer': v.lower(),
                       'response': {'content': 'Reasoning: example\nFinal answer: '+v}}
                      for r, v in zip(['teacher','student_1','student_2','student_3','student_4'],
                                      ['No','Yes','Maybe','No','No'])]
        self.rows = [{'sample_index':0, 'sample_id':'id1', 'arms':{'select_no_memory':{
            'calls': self.calls, 'answer':'no', 'positive_role':'teacher'}}}]
        self.records = [{'sample_index':0,'sample_id':'id1','original_positive_role':'teacher',
                         'replay_positive_role':'student_1','original_answer':'no','replay_answer':'yes'}]
        self.truths = [['Yes'] * 10]

    def test_paired_scores_use_raw_selected_responses(self):
        result = score_records(self.rows, self.records, self.truths, DummyEvaluator(), 'select_no_memory')
        self.assertEqual(result['original_mean'], 0)
        self.assertEqual(result['replay_mean'], 1)
        self.assertEqual(result['improved_items'], 1)
        self.assertEqual(result['worsened_items'], 0)
        self.assertEqual(result['paired_delta']['mean_and_95pct_ci'][0], 1)

    def test_rejects_tampered_answer_or_id(self):
        self.records[0]['replay_answer'] = 'fake'
        with self.assertRaises(ValueError):
            score_records(self.rows, self.records, self.truths, DummyEvaluator(), 'select_no_memory')
        self.records[0]['replay_answer'] = 'yes'
        self.records[0]['sample_id'] = 'other'
        with self.assertRaises(ValueError):
            score_records(self.rows, self.records, self.truths, DummyEvaluator(), 'select_no_memory')

    def test_must_have_ten_answers(self):
        with self.assertRaises(ValueError):
            score_records(self.rows, self.records, [['Yes']], DummyEvaluator(), 'select_no_memory')

    def test_replay_mismatch_rejected_before_missing_label_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); inference=root/'input.jsonl'; inference.write_text('{}\n')
            labels=root/'missing.jsonl'; subset=root/'subset.json'; rep=root/'replay.json'
            subset.write_text(json.dumps({'dataset_id':'official/TextVQA_0.5.1','split':'train','count':400,
                'inference_sha256':sha256_file(inference),'labels_sha256':'a'*64}))
            rep.write_text(json.dumps({'arm':'select_no_memory','records':['tampered']}))
            with patch('scripts.rescore_textvqa_answer_similarity_replay.replay', return_value={'different':True}):
                with self.assertRaisesRegex(ValueError, 'source artifacts changed'):
                    rescore(root,inference,subset,labels,rep,DummyEvaluator(),sha256_file(subset))

    def test_subset_manifest_is_pinned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); subset=root/'subset'; subset.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'manifest SHA256 mismatch'):
                rescore(root, root/'input', subset, root/'labels', root/'replay', DummyEvaluator(), '0'*64)


if __name__ == '__main__':
    unittest.main()
