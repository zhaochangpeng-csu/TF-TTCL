import json
import tempfile
import unittest
from pathlib import Path

from scripts.create_label_isolated_subset import sha256_file
from scripts.run_hygon_textvqa_format_factorial import digest_ids, load_checked


class FreshInputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'images').mkdir()
        self.rows = []
        for index in range(400):
            image_id = f'image_{index}'
            (self.root / 'images' / f'{image_id}.jpg').write_bytes(f'pixels:{index}'.encode())
            self.rows.append({'id': str(index), 'image_id': image_id,
                              'question': 'What is written?', 'image': f'images/{image_id}.jpg',
                              'choices': []})
        self.inference = self.root / 'inference.jsonl'
        self.manifest = self.root / 'manifest.json'
        self.prior = self.root / 'prior.jsonl'
        self.prior.write_text(json.dumps({'id': 'old', 'image_id': 'old-image', 'question': 'Old?'}) + '\n')
        self.write_export()

    def write_export(self):
        self.inference.write_text(''.join(json.dumps(row) + '\n' for row in self.rows))
        ids = [row['id'] for row in self.rows]
        image_ids = sorted({row['image_id'] for row in self.rows})
        manifest = {'dataset_id': 'official/TextVQA_0.5.1', 'split': 'train', 'seed': 20261126,
                    'count': 400, 'image_count': len(image_ids), 'selected_ids': ids,
                    'selected_ids_sha256': digest_ids(ids), 'inference_sha256': sha256_file(self.inference),
                    'selected_images_sha256': digest_ids([f'{name}:{sha256_file(self.root / "images" / (name + ".jpg"))}'
                                                          for name in image_ids])}
        self.manifest.write_text(json.dumps(manifest))

    def check(self):
        return load_checked(self.inference, self.manifest, sha256_file(self.manifest), [self.prior], 20,
                            excluded_sha_counts={sha256_file(self.prior): 1})

    def test_checks_all_400_and_opens_no_label_file(self):
        samples, meta, _ = self.check()
        self.assertEqual(len(samples), 20)
        self.assertEqual(meta['count'], 400)
        self.assertFalse((self.root / 'labels.jsonl').exists())

    def test_rejects_overlap_by_question_or_image(self):
        for field, value in [('id', '0'), ('image_id', 'image_0')]:
            with self.subTest(field=field):
                row = {'id': 'old', 'image_id': 'old-image', 'question': 'Old?'}
                row[field] = value
                self.prior.write_text(json.dumps(row) + '\n')
                with self.assertRaisesRegex(ValueError, 'overlaps'):
                    self.check()

    def test_rejects_unfrozen_file_label_and_modified_image(self):
        with self.assertRaisesRegex(ValueError, 'manifest hash'):
            load_checked(self.inference, self.manifest, '0' * 64, [self.prior], 20,
                         excluded_sha_counts={sha256_file(self.prior): 1})
        self.rows[0]['answers'] = ['secret']
        self.write_export()
        with self.assertRaisesRegex(ValueError, 'label'):
            self.check()
        del self.rows[0]['answers']
        self.write_export()
        (self.root / 'images' / 'image_399.jpg').write_bytes(b'modified')
        with self.assertRaisesRegex(ValueError, 'image count or hashes'):
            self.check()

    def test_rejects_duplicate_image_even_if_manifest_recomputed(self):
        self.rows[-1]['image_id'] = 'image_0'
        self.rows[-1]['image'] = 'images/image_0.jpg'
        self.write_export()
        with self.assertRaisesRegex(ValueError, 'image count or hashes'):
            self.check()


if __name__ == '__main__':
    unittest.main()
