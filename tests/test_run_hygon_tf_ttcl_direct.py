import hashlib
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

from core.multimodal.datasets import LoadedExample
from core.multimodal.prompts import format_question
from core.multimodal.schema import EvaluationTarget, MultimodalSample, SamplingConfig
from core.multimodal.tf_ttcl_direct_runner import TEACHER_SYSTEM
from scripts.run_hygon_tf_ttcl_direct import checked_sample


class PathMappingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / 'diagram.png'
        self.image.write_bytes(b'image-bytes')
        self.old = Path('/old/images')
        self.sample = MultimodalSample('1', 'scienceqa_img', 'Which one?',
                                       (str(self.old / 'diagram.png'),), ('x', 'y'))
        image_hash = hashlib.sha256(b'image-bytes').hexdigest()
        sampling = SamplingConfig(temperature=0.0, max_tokens=256, seed=0, request_logprobs=True)
        self.ref = {'sample_index': 0, 'sample_id': '1', 'labels_read': False,
                    'image_hashes': [image_hash], 'arms': {'greedy': {'calls': [{
                        'system': TEACHER_SYSTEM,
                        'user': format_question(self.sample.question, self.sample.choices, self.sample.metadata),
                        'sampling': asdict(sampling), 'image_hashes': [image_hash]}]}}}

    def test_maps_only_to_verified_image(self):
        result = checked_sample(0, LoadedExample(self.sample, None), self.ref, self.root, self.old)
        self.assertEqual(result.image_paths, (str(self.image.resolve()),))
        self.image.write_bytes(b'wrong')
        with self.assertRaisesRegex(ValueError, 'image hash'):
            checked_sample(0, LoadedExample(self.sample, None), self.ref, self.root, self.old)

    def test_rejects_outside_path_labels_and_future_index(self):
        outside = replace(self.sample, image_paths=('/untrusted/diagram.png',))
        with self.assertRaisesRegex(ValueError, 'Unexpected old image path'):
            checked_sample(0, LoadedExample(outside, None), self.ref, self.root, self.old)
        with self.assertRaisesRegex(ValueError, 'Label boundary'):
            checked_sample(0, LoadedExample(self.sample, EvaluationTarget('1', ('a',))), self.ref, self.root, self.old)
        with self.assertRaisesRegex(ValueError, 'sample order'):
            checked_sample(1, LoadedExample(self.sample, None), self.ref, self.root, self.old)


if __name__ == '__main__':
    unittest.main()
