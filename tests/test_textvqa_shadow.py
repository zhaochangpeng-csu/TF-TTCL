import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from core.multimodal.schema import MultimodalSample
from core.multimodal.image_ops import sha256_file
from core.multimodal.textvqa_format_factorial import SHORT_FINAL_SUFFIX
from core.multimodal.textvqa_shadow import ARMS, neutralize_rules, replay_item, system_prompt
from scripts.run_hygon_textvqa_shadow import run_shard
from tests.test_textvqa_format_factorial import Backend, Embedder


class Words:
    def encode(self, text, add_special_tokens=False):
        return text.split()


class ShadowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        image = self.root / 'image.jpg'
        image.write_bytes(b'image fixture')
        self.sample = MultimodalSample('s1', 'textvqa', 'Question?', (str(image),))
        self.rules = {'positive': [{'rule_id': 'positive_0', 'sample_index': 0,
                                     'text': 'use careful visual evidence', 'similarity': 0.5}],
                      'negative': [{'rule_id': 'negative_0', 'sample_index': 0,
                                     'text': 'avoid introducing unsupported detail', 'similarity': 0.4}]}
        self.anchor = {'sample_index': 1, 'sample_id': 's1', 'image_hashes': [sha256_file(image)],
                       'original_question': self.sample.question,
                       'variants': [f'Question variant {i}?' for i in range(4)],
                       'retrieved_rules': self.rules,
                       'arms': {'select_no_memory': {'answer': 'red'},
                                'tf_ttcl_mm_direct': {'answer': 'blue'}}}

    def test_neutral_control_preserves_buckets_without_mutating_history(self):
        old = copy.deepcopy(self.rules)
        result, counts = neutralize_rules(self.rules, Words())
        self.assertEqual(self.rules, old)
        self.assertEqual([len(result[k]) for k in ('positive', 'negative')], [1, 1])
        self.assertTrue(all(abs(c['real_tokens'] - c['neutral_tokens']) <= 1 for c in counts))
        self.assertNotIn('visual', result['positive'][0]['text'])

    def test_position_moves_same_suffix_and_preserves_original_rule_text(self):
        before = system_prompt('teacher', self.rules, 'before')
        after = system_prompt('teacher', self.rules, 'after')
        self.assertEqual(before.count(SHORT_FINAL_SUFFIX), 1)
        self.assertEqual(after.count(SHORT_FINAL_SUFFIX), 1)
        self.assertLess(before.index(SHORT_FINAL_SUFFIX), before.index('<BEGIN_RULES>'))
        self.assertGreater(after.index(SHORT_FINAL_SUFFIX), after.index('<END_RULES>'))
        self.assertEqual(before.replace(SHORT_FINAL_SUFFIX, ''), after.replace(SHORT_FINAL_SUFFIX, ''))

    def test_all_arms_replay_five_with_paired_seeds_and_no_rule_writes(self):
        row = replay_item(Backend(), Embedder(), Words(), self.sample, self.anchor, 1, 0)
        self.assertEqual(set(row['arms']), set(ARMS))
        self.assertFalse(row['labels_read'])
        self.assertFalse(row['rules_written'])
        for arm in row['arms'].values():
            self.assertEqual([c['role'] for c in arm['calls']],
                             ['teacher', 'student_1', 'student_2', 'student_3', 'student_4'])
            self.assertEqual([c['sampling']['seed'] for c in arm['calls']], [100, 102, 103, 104, 105])
            self.assertEqual([c['question_variant'] for c in arm['calls']],
                             [self.sample.question, *self.anchor['variants']])
            self.assertEqual(arm['reason'], 'teacher_similarity_top_half')
        self.assertEqual(self.anchor['retrieved_rules'], self.rules)
        with self.assertRaises(ValueError):
            replay_item(Backend(), Embedder(), Words(), self.sample, self.anchor, 0, 0)
        future = copy.deepcopy(self.anchor)
        future['retrieved_rules']['positive'][0]['sample_index'] = 1
        with self.assertRaisesRegex(ValueError, 'current or future'):
            replay_item(Backend(), Embedder(), Words(), self.sample, future, 1, 0)
        wrong_image = copy.deepcopy(self.anchor)
        wrong_image['image_hashes'] = ['wrong']
        with self.assertRaisesRegex(ValueError, 'image hash'):
            replay_item(Backend(), Embedder(), Words(), self.sample, wrong_image, 1, 0)

    def test_shard_atomic_resume_and_config_mismatch(self):
        args = SimpleNamespace(output=self.root / 'output', shard_index=0, shard_count=1, limit=1)
        anchor = {**self.anchor, 'sample_index': 0}
        anchor['retrieved_rules'] = {'positive': [], 'negative': []}
        cfg = {'a': 1}
        run_shard(args, cfg, [self.sample], [anchor], Backend(), Embedder(), Words())
        path = args.output / 'shard-000-of-001' / 'row-000000.json'
        first = path.read_text()
        run_shard(args, cfg, [self.sample], [anchor], Backend(), Embedder(), Words())
        self.assertEqual(path.read_text(), first)
        with self.assertRaisesRegex(ValueError, 'incompatible shard resume'):
            run_shard(args, {'a': 2}, [self.sample], [anchor], Backend(), Embedder(), Words())
        result = json.loads(first)
        self.assertEqual(len(result['arms']), 4)

    def test_interrupted_item_resumes_from_immutable_traces(self):
        class Interrupted(Backend):
            def __init__(self):
                self.calls = 0

            def generate(self, *args):
                self.calls += 1
                if self.calls == 3:
                    raise RuntimeError('interrupted')
                return super().generate(*args)

        args = SimpleNamespace(output=self.root / 'output', shard_index=0, shard_count=1, limit=1)
        anchor = {**self.anchor, 'sample_index': 0, 'retrieved_rules': {'positive': [], 'negative': []}}
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            run_shard(args, {'seed': 0}, [self.sample], [anchor], Interrupted(), Embedder(), Words())
        shard = args.output / 'shard-000-of-001'
        self.assertFalse((shard / 'row-000000.json').exists())
        self.assertEqual(len(list((shard / 'response_trace').glob('*.json'))), 2)
        run_shard(args, {'seed': 0}, [self.sample], [anchor], Backend(), Embedder(), Words())
        self.assertTrue((shard / 'row-000000.json').is_file())


if __name__ == '__main__':
    unittest.main()
