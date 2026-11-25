import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.audit_hygon_tf_ttcl_direct import OLD_ROOT, mapped_inputs


class HygonAuditMappingTests(unittest.TestCase):
    def fixture(self, root):
        image = root / 'sample.png'
        image.write_bytes(b'known image bytes')
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        inputs = [{'id': str(i), 'question': 'q', 'image': str(OLD_ROOT / image.name)}
                  for i in range(400)]
        refs = [{'sample_id': str(i), 'sample_index': i, 'image_hashes': [digest]}
                for i in range(400)]
        rows = [{'sample_id': str(i), 'sample_index': i, 'image_hashes': [digest],
                 'retrieved_rules': {'positive': [], 'negative': []}, 'written_rules': []}
                for i in range(20)]
        return inputs, refs, rows

    def test_original_path_is_mapped_without_mutating_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs, refs, rows = self.fixture(root)
            mapped = mapped_inputs(inputs, refs, rows, root, 20)
            self.assertEqual(mapped[0]['image'], str((root / 'sample.png').resolve()))
            self.assertEqual(inputs[0]['image'], str(OLD_ROOT / 'sample.png'))
            self.assertEqual(mapped[20], inputs[20])

    def test_bad_path_hash_and_current_rule_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs, refs, rows = self.fixture(root)
            inputs[0]['image'] = '/tmp/sample.png'
            with self.assertRaises(ValueError):
                mapped_inputs(inputs, refs, rows, root, 20)
            inputs, refs, rows = self.fixture(root)
            rows[0]['image_hashes'] = ['0' * 64]
            with self.assertRaises(ValueError):
                mapped_inputs(inputs, refs, rows, root, 20)
            inputs, refs, rows = self.fixture(root)
            rows[0]['retrieved_rules']['positive'] = [{'rule_id': 'x', 'sample_index': 0, 'text': 'x'}]
            with self.assertRaises(ValueError):
                mapped_inputs(inputs, refs, rows, root, 20)

    def test_only_actual_prior_rule_may_be_retrieved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs, refs, rows = self.fixture(root)
            rows[0]['written_rules'] = [{'rule_id': 'r', 'sample_index': 0, 'kind': 'positive', 'text': 'rule'}]
            rows[1]['retrieved_rules']['positive'] = [{'rule_id': 'r', 'sample_index': 0, 'text': 'rule'}]
            mapped_inputs(inputs, refs, rows, root, 20)
            rows[1]['retrieved_rules']['positive'][0]['text'] = 'changed'
            with self.assertRaises(ValueError):
                mapped_inputs(inputs, refs, rows, root, 20)


if __name__ == '__main__':
    unittest.main()
