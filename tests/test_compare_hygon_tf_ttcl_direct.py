import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.multimodal.tf_ttcl_direct_runner import ARMS
from scripts import compare_hygon_tf_ttcl_direct as comparison


class HygonOnlineComparisonTests(unittest.TestCase):
    def fixture(self, root):
        ids = [str(i) for i in range(400)]
        def prediction(i, answer):
            return {'sample_index': i, 'sample_id': str(i), 'labels_read': False,
                    'arms': {name: {'answer': answer} for name in ARMS}}
        refs = [prediction(i, 'a') for i in range(400)]
        rows = [prediction(i, 'a') for i in range(400)]
        rows[0] = prediction(0, 'b')
        rows[1] = prediction(1, 'b')
        targets = [{'id': str(i), 'answer': 'b' if i == 0 else 'a'} for i in range(400)]
        def write(path, records):
            path.write_text(''.join(json.dumps(item) + '\n' for item in records))
        inference, reference, labels = (root / x for x in ('inference.jsonl', 'reference.jsonl', 'labels.jsonl'))
        write(inference, [{'id': i} for i in ids])
        write(reference, refs)
        write(labels, targets)
        run = root / 'run'
        run.mkdir()
        write(run / 'predictions.jsonl', rows)
        manifest = {'status': 'completed', 'processed': 400, 'labels_read': False, 'ordered_ids': ids}
        return run, inference, reference, labels, rows, manifest

    def test_all_four_arms_and_paired_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            run, inference, reference, labels, rows, manifest = self.fixture(Path(temp))
            ref_sha = hashlib.sha256(reference.read_bytes()).hexdigest()
            label_sha = hashlib.sha256(labels.read_bytes()).hexdigest()
            with patch.object(comparison, 'REFERENCE_SHA', ref_sha), \
                 patch.object(comparison, 'audit', return_value=(manifest, rows)):
                result = comparison.compare(run, inference, reference, labels, label_sha)
            for entry in result['results'].values():
                self.assertEqual((entry['nvidia_correct'], entry['hygon_correct']), (399, 399))
                self.assertEqual((entry['answers_equal'], entry['improved'], entry['worsened']), (398, 1, 1))
                self.assertAlmostEqual(entry['paired_delta_ci'][0], 0.0)

    def test_audit_failure_prevents_label_open(self):
        with tempfile.TemporaryDirectory() as temp:
            run, inference, reference, labels, rows, manifest = self.fixture(Path(temp))
            ref_sha = hashlib.sha256(reference.read_bytes()).hexdigest()
            labels.unlink()
            with patch.object(comparison, 'REFERENCE_SHA', ref_sha), \
                 patch.object(comparison, 'audit', side_effect=ValueError('incomplete')):
                with self.assertRaisesRegex(ValueError, 'incomplete'):
                    comparison.compare(run, inference, reference, labels)

    def test_wrong_order_or_label_hash_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            run, inference, reference, labels, rows, manifest = self.fixture(Path(temp))
            ref_sha = hashlib.sha256(reference.read_bytes()).hexdigest()
            with patch.object(comparison, 'REFERENCE_SHA', ref_sha), \
                 patch.object(comparison, 'audit', return_value=(manifest, rows)):
                with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
                    comparison.compare(run, inference, reference, labels, '0' * 64)
                rows[1]['sample_id'] = 'wrong'
                with self.assertRaisesRegex(ValueError, 'alignment'):
                    comparison.compare(run, inference, reference, labels)


if __name__ == '__main__':
    unittest.main()
