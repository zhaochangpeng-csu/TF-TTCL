import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_textvqa_fresh_development import select


class FreshSelectionTests(unittest.TestCase):
    def test_deterministic_exclusions_and_label_blindness(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotation, old_train, old_val = (root / name for name in ('ann.json', 'train.jsonl', 'val.jsonl'))
            rows = [{'question_id': i, 'image_id': f'im{i}', 'question': f'Q{i}?',
                     'set_name': 'train', 'answers': ['hidden']} for i in range(10)]
            # A second question for an excluded image must also be excluded.
            rows[8]['image_id'] = 'im0'
            # New stream must not include two questions sharing an image.
            rows[9]['image_id'] = 'im7'
            annotation.write_text(json.dumps({'data': rows}))
            old_train.write_text(json.dumps({'id': '0', 'image_id': 'im0', 'image': 'images/im0.jpg'}) + '\n')
            old_val.write_text(json.dumps({'id': '1', 'image_id': 'im1', 'image': 'images/im1.jpg'}) + '\n')
            kwargs = {'count': 4, 'expected_train': 10, 'expected_previous': 1,
                      'expected_val': 1, 'validate_hashes': False}
            a = select(annotation, old_train, old_val, **kwargs)
            self.assertEqual(a, select(annotation, old_train, old_val, **kwargs))
            self.assertTrue(a['labels_used_for_selection'] is False)
            self.assertEqual(len({item['image_id'] for item in a['selection']}), 4)
            self.assertFalse({'0', '1', '8'} & {item['id'] for item in a['selection']})
            self.assertFalse({'im0', 'im1'} & {item['image_id'] for item in a['selection']})
            for row in rows:
                row['answers'] = ['different hidden answer']
            annotation.write_text(json.dumps({'data': rows}))
            b = select(annotation, old_train, old_val, **kwargs)
            self.assertEqual(a['selection'], b['selection'])

    def test_rejects_previous_labels_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotation, old_train, old_val = (root / name for name in ('ann.json', 'train.jsonl', 'val.jsonl'))
            annotation.write_text(json.dumps({'data': [
                {'question_id': 2, 'image_id': 'a', 'question': 'q', 'set_name': 'train'},
                {'question_id': 2, 'image_id': 'b', 'question': 'q', 'set_name': 'train'}]}))
            old_train.write_text(json.dumps({'id': '0', 'image_id': 'x', 'answers': ['secret']}) + '\n')
            old_val.write_text(json.dumps({'id': '1', 'image_id': 'y'}) + '\n')
            kwargs = {'count': 1, 'expected_train': 2, 'expected_previous': 1,
                      'expected_val': 1, 'validate_hashes': False}
            with self.assertRaises(ValueError):
                select(annotation, old_train, old_val, **kwargs)
            old_train.write_text(json.dumps({'id': '0', 'image_id': 'x'}) + '\n')
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                select(annotation, old_train, old_val, **kwargs)


if __name__ == '__main__':
    unittest.main()
