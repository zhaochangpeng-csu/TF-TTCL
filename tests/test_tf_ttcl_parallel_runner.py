import json
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from core.multimodal.schema import GenerationResponse, MultimodalSample, SamplingConfig
from core.multimodal.tf_ttcl_direct_runner import DirectRunner
from core.multimodal.tf_ttcl_parallel_runner import ParallelDirectRunner, RoutedBackend, SingleFlightBackend


class Embedder:
    def encode(self, text, *, query=False):
        return (1.0, 0.0) if query else (0.0, 1.0)


class Backend:
    model_name = 'fake-vlm'
    def __init__(self, output=None, fail=False):
        self.calls = []
        self.lock = threading.Lock()
        self.output = output
        self.fail = fail

    def generate(self, system, user, images, sampling):
        with self.lock:
            self.calls.append((system, user, sampling.seed))
        if self.output and 'Question 1?' in user:
            rows = (self.output / 'predictions.jsonl').read_text().splitlines()
            if len(rows) != 1 or json.loads(rows[0])['sample_index'] != 0:
                raise AssertionError('Next question started before previous commit')
        time.sleep(0.002 * (4 - sampling.seed % 4))
        if self.fail and sampling.seed == 2:
            raise RuntimeError('simulated request failure')
        if 'Rewrite only the question' in system:
            # Invalid variants force fallback and therefore duplicate requests.
            return GenerationResponse('{}', prompt_tokens=4, completion_tokens=1)
        if '<task>' in user:
            return GenerationResponse('Use visual evidence.', prompt_tokens=4, completion_tokens=1)
        answer = 'A' if sampling.seed % 3 != 0 else 'B'
        return GenerationResponse('Final answer: ' + answer, prompt_tokens=4,
                                  completion_tokens=2, logprobs=[-0.1, -0.2])


class ParallelRunnerTests(unittest.TestCase):
    def test_singleflight_deduplicates_concurrent_requests(self):
        backend = Backend()
        wrapper = SingleFlightBackend(backend)
        config = SamplingConfig(seed=3)
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: wrapper.generate('system', 'user', (), config), range(24)))
        self.assertEqual(len(backend.calls), 1)
        self.assertTrue(all(result.content == results[0].content for result in results))

    def test_singleflight_error_propagates_and_retry_works(self):
        backend = Backend(fail=True)
        wrapper = SingleFlightBackend(backend)
        with self.assertRaisesRegex(RuntimeError, 'simulated'):
            wrapper.generate('s', 'u', (), SamplingConfig(seed=2))
        backend.fail = False
        self.assertTrue(wrapper.generate('s', 'u', (), SamplingConfig(seed=2)).content)
        self.assertEqual(len(backend.calls), 2)

    def test_routing_is_deterministic_and_rejects_mixed_model_names(self):
        backends = [Backend() for _ in range(7)]
        routed = RoutedBackend(backends)
        for _ in range(3):
            routed.generate('system', 'user', (), SamplingConfig(seed=7))
        self.assertEqual(sorted(len(backend.calls) for backend in backends), [0, 0, 0, 0, 0, 0, 3])
        backends[0].model_name = 'different'
        with self.assertRaisesRegex(ValueError, 'common model'):
            RoutedBackend(backends)

    def test_parallel_matches_sequential_and_commits_before_next(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / 'image.png'
            image.write_bytes(b'fake-image')
            samples = [MultimodalSample(str(i), 'scienceqa_img', f'Question {i}?',
                                        (str(image),), ('first', 'second')) for i in range(2)]
            sequential = root / 'seq'
            parallel = root / 'par'
            DirectRunner(Backend(), Embedder(), samples, sequential, 'input', 'model', 'embed').run()
            backend = Backend(output=parallel)
            ParallelDirectRunner(backend, Embedder(), samples, parallel, 'input', 'model', 'embed').run()
            read = lambda path: [json.loads(line) for line in (path / 'predictions.jsonl').read_text().splitlines()]
            self.assertEqual(read(sequential), read(parallel))
            rows = read(parallel)
            self.assertTrue(all(rule['sample_index'] == 0 for bucket in rows[1]['retrieved_rules'].values()
                                for rule in bucket))
            resumed = ParallelDirectRunner(Backend(), Embedder(), samples, parallel, 'input', 'model', 'embed')
            resumed.run()
            self.assertEqual(read(parallel), rows)
            with self.assertRaisesRegex(ValueError, 'Resume'):
                ParallelDirectRunner(Backend(), Embedder(), samples, parallel, 'input', 'model', 'embed',
                                     max_workers=4).run()
            # First sample's fallback makes vote/nomem/full Students identical;
            # only four actual Student generations are allowed.
            first_students = [x for x in backend.calls if x[2] in (2, 3, 4, 5) and x[0].startswith('You are a visual')]
            self.assertEqual(len(first_students), 4)

    def test_failure_does_not_advance_or_commit_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / 'image.png'
            image.write_bytes(b'fake-image')
            samples = [MultimodalSample(str(i), 'scienceqa_img', f'Question {i}?',
                                        (str(image),), ('first', 'second')) for i in range(2)]
            backend = Backend(fail=True)
            runner = ParallelDirectRunner(backend, Embedder(), samples, root / 'run', 'input', 'model', 'embed')
            with self.assertRaisesRegex(RuntimeError, 'simulated'):
                runner.run()
            self.assertEqual(runner.rules, [])
            self.assertFalse((root / 'run' / 'predictions.jsonl').exists())
            self.assertFalse(any('Question 1?' in call[1] for call in backend.calls))


if __name__ == '__main__':
    unittest.main()
