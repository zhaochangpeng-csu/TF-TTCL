import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from core.multimodal.replay import ResponseTraceBackend
from core.multimodal.schema import MultimodalSample
from core.multimodal.image_ops import sha256_file
from core.multimodal.textvqa_shadow import replay_item
from core.multimodal.tf_ttcl_direct_runner import _digest
from scripts.audit_textvqa_shadow import audit_item, audit_shards
from scripts.run_hygon_textvqa_shadow import run_shard
from tests.test_textvqa_shadow import Words
from tests.test_textvqa_format_factorial import Backend, Embedder


class Model(Backend):
    model_name = 'Qwen/Qwen2.5-VL-7B-Instruct'


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        image = root / 'image.jpg'
        image.write_bytes(b'an image')
        self.sample = MultimodalSample('id', 'textvqa', 'What is shown?', (str(image),))
        self.anchor = {'sample_index': 1, 'sample_id': 'id',
                       'original_question': self.sample.question, 'image_hashes': [sha256_file(image)],
                       'variants': [f'Variant {n}?' for n in range(4)],
                       'retrieved_rules': {'positive': [{'rule_id': 'positive_0', 'sample_index': 0,
                                                         'text': 'look carefully', 'similarity': 0.2}],
                                           'negative': []},
                       'arms': {'select_no_memory': {'answer': 'red'}, 'tf_ttcl_mm_direct': {'answer': 'blue'}}}
        self.config = {'seed': 0}
        self.trace = root / 'traces'
        backend = ResponseTraceBackend(Model(), self.trace, self.config, 'record')
        self.row = replay_item(backend, Embedder(), Words(), self.sample, self.anchor, 1, 0)
        self.row['config_sha256'] = _digest(self.config)

    def audit(self, row):
        return audit_item(row, self.sample, self.anchor, 1, self.config, Words(), Embedder(), self.trace, {})

    def test_audits_requests_ppl_selection_and_trace(self):
        logical, traces = self.audit(self.row)
        self.assertEqual(len(logical), 4)
        self.assertGreater(len(traces), 5)
        self.assertEqual(logical['real_before']['completion_tokens'], 10)

    def test_rejects_corrupted_logprob_prompt_selection_and_history(self):
        for mutate in ('logprob', 'prompt', 'selection', 'history'):
            row = copy.deepcopy(self.row)
            arm = row['arms']['real_before']
            if mutate == 'logprob':
                arm['calls'][0]['response']['logprobs'].pop()
            elif mutate == 'prompt':
                arm['calls'][0]['system'] += ' changed'
            elif mutate == 'selection':
                arm['positive_role'] = 'student_4'
            else:
                row['retrieved_rules']['positive'][0]['sample_index'] = 1
            with self.subTest(mutate=mutate), self.assertRaises((ValueError, KeyError)):
                self.audit(row)

    def test_complete_shard_report_and_orphan_trace_rejection(self):
        root = self.trace.parent / 'shards'
        args = SimpleNamespace(output=root, shard_index=0, shard_count=1, limit=1)
        anchor = {**self.anchor, 'sample_index': 0,
                  'retrieved_rules': {'positive': [], 'negative': []}}
        config = {'seed': 0, 'version': 'textvqa_shadow_fixed_history_v1',
                  'shard_index': 0, 'shard_count': 1, 'limit': 1, 'labels_read': False,
                  'rules_written': False,
                  'arms': ['real_before', 'real_after', 'neutral_before', 'neutral_after']}
        run_shard(args, config, [self.sample], [anchor], Model(), Embedder(), Words())
        report = audit_shards(root, [self.sample], [anchor], Words(), Embedder(), 1, 1)
        self.assertEqual(report['status'], 'passed')
        self.assertEqual(report['physical_unique_requests'], 5)
        self.assertEqual(report['physical_unique_completion_tokens'], 10)
        self.assertEqual(report['logical_per_arm']['real_before']['completion_tokens'], 10)
        trace = root / 'shard-000-of-001' / 'response_trace' / 'orphan.json'
        trace.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'orphan or missing trace'):
            audit_shards(root, [self.sample], [anchor], Words(), Embedder(), 1, 1)


if __name__ == '__main__':
    unittest.main()
