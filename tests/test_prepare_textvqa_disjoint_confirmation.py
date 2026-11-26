import json
import tempfile
import unittest
from pathlib import Path
from scripts.prepare_textvqa_disjoint_confirmation import select


class DisjointSelectionTest(unittest.TestCase):
    def test_determinism_all_prior_ids_images_and_label_blindness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ann = root / 'annotation.json'
            excluded = [root / f'prior{i}.jsonl' for i in range(3)]
            for i, p in enumerate(excluded):
                p.write_text(json.dumps({'id': str(i), 'image_id': f'im{i}'}) + '\n')
            rows = [{'question_id': i, 'image_id': f'im{i}', 'question': 'q',
                     'set_name': 'train', 'answers': ['hidden']} for i in range(20)]
            rows[18]['image_id'] = 'im0'
            rows[19]['image_id'] = 'im3'
            ann.write_text(json.dumps({'data': rows}))
            opts = {'count': 10, 'expected_train': 20, 'validate_hashes': False}
            first = select(ann, excluded, **opts)
            self.assertEqual(first, select(ann, list(reversed(excluded)), **opts))
            items = first['selection']
            self.assertEqual(len({x['image_id'] for x in items}), 10)
            self.assertFalse({'0', '1', '2', '18'} & {x['id'] for x in items})
            self.assertFalse({'im0', 'im1', 'im2'} & {x['image_id'] for x in items})
            for row in rows:
                row['answers'] = ['different']
            ann.write_text(json.dumps({'data': rows}))
            self.assertEqual(items, select(ann, excluded, **opts)['selection'])
            self.assertFalse(first['labels_used_for_selection'])

    def test_fail_closed_if_previous_has_labels_or_paths_repeat(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ann = root / 'annotation.json'
            ann.write_text(json.dumps({'data': [{'question_id': 10, 'image_id': 'img',
                                                'question': 'q', 'set_name': 'train'}]}))
            paths = [root / f'{i}.jsonl' for i in range(3)]
            for i, p in enumerate(paths):
                p.write_text(json.dumps({'id': str(i), 'image_id': str(i)}) + '\n')
            opts = {'count': 1, 'expected_train': 1, 'validate_hashes': False}
            with self.assertRaises(ValueError):
                select(ann, [paths[0], paths[0], paths[2]], **opts)
            paths[0].write_text(json.dumps({'id': '0', 'image_id': '0', 'answers': ['secret']}) + '\n')
            with self.assertRaises(ValueError):
                select(ann, paths, **opts)


if __name__ == '__main__':
    unittest.main()
