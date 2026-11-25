"""Experimental same-question parallelism for DirectRunner.

This prototype is not a replacement for the faithful sequential baseline:
different service scheduling or hardware can change sampled outputs even with
identical seeds. Online questions are still committed one at a time.
"""
from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict

from .image_ops import sha256_file
from .schema import SamplingConfig
from .tf_ttcl_direct_runner import (DirectRunner, TUTOR_SYSTEM, _digest, _rule_prompt,
                                     parse_variants, retrieve_rules, split_question_clause)


class SingleFlightBackend:
    """Deduplicate identical concurrent requests before they reach a trace writer.

    One process must own an output/trace directory. Existing trace I/O still
    fails closed on interrupted partial files; this wrapper does not make the
    underlying trace writer crash-atomic or protect against another process.
    """
    def __init__(self, backend):
        self.backend = backend
        self.model_name = backend.model_name
        self._lock = threading.Lock()
        self._requests: dict[str, Future] = {}

    @property
    def trace_stats(self):
        return getattr(self.backend, 'trace_stats', None)

    def generate(self, system_prompt, user_text, image_paths, sampling):
        request = {'model': self.model_name, 'system': system_prompt, 'user': user_text,
                   'image_hashes': [sha256_file(path) for path in image_paths],
                   'sampling': asdict(sampling)}
        key = hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        with self._lock:
            future = self._requests.get(key)
            leader = future is None
            if leader:
                future = Future()
                self._requests[key] = future
        if leader:
            try:
                future.set_result(self.backend.generate(system_prompt, user_text, image_paths, sampling))
            except BaseException as error:
                future.set_exception(error)
                with self._lock:
                    if self._requests.get(key) is future:
                        del self._requests[key]
                raise
        return future.result()


class RoutedBackend:
    """Deterministically spread distinct requests across identically configured services."""
    def __init__(self, backends):
        if not backends or len({backend.model_name for backend in backends}) != 1:
            raise ValueError('Backends must provide a common model name')
        self.backends = list(backends)
        self.model_name = backends[0].model_name
        self._locks = [threading.Lock() for _ in backends]

    def generate(self, system_prompt, user_text, image_paths, sampling):
        request = {'model': self.model_name, 'system': system_prompt, 'user': user_text,
                   'image_hashes': [sha256_file(path) for path in image_paths],
                   'sampling': asdict(sampling)}
        key = hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False).encode()).digest()
        index = int.from_bytes(key[:8], 'big') % len(self.backends)
        with self._locks[index]:
            return self.backends[index].generate(system_prompt, user_text, image_paths, sampling)


class ParallelDirectRunner(DirectRunner):
    """Parallelize calls within each sample; inherit original ordered run()."""
    def __init__(self, backend, embedder, samples, output, input_sha256,
                 model_revision, embedding_revision, seed=0, max_workers=8):
        if max_workers < 1:
            raise ValueError('max_workers must be positive')
        super().__init__(SingleFlightBackend(backend), embedder, samples, output,
                         input_sha256, model_revision, embedding_revision, seed)
        self.max_workers = max_workers
        self.config['experimental_parallel_workers'] = max_workers
        self.config_sha = _digest(self.config)
        self.code_sha = _digest([self.code_sha, sha256_file(__file__)])

    def _sample(self, sample, index):
        sample.validate_images()
        query_vector = self.embedder.encode(sample.question, query=True) if self.rules else ()
        retrieved = retrieve_rules(self.rules, query_vector, index) if self.rules else {
            'positive': [], 'negative': []}
        rule_text = _rule_prompt(retrieved)
        context_prefix, question_clause = split_question_clause(sample.question)
        # A context manager waits for in-flight requests to finish on failure;
        # no rule is written and run() never advances to the next question.
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            teacher_f = pool.submit(self._candidate, 'teacher', sample, sample.question,
                                    self.teacher_system, 0, index, False)
            vote_f = [pool.submit(self._candidate, f'student_{n}', sample, sample.question,
                                  self.student_system, n + 1, index, True) for n in range(1, 5)]
            tutor_f = pool.submit(self._call, TUTOR_SYSTEM, question_clause, (), SamplingConfig(
                temperature=0.7, top_p=0.9, max_tokens=384, seed=self.seed + 100 * index + 1))
            full_teacher_f = (pool.submit(self._candidate, 'teacher', sample, sample.question,
                                          self.teacher_system + '\n' + rule_text, 0, index, False)
                              if rule_text else None)
            teacher, teacher_call = teacher_f.result()
            vote_pairs = [future.result() for future in vote_f]
            tutor_call = tutor_f.result()
            variants = [context_prefix + item for item in
                        parse_variants(tutor_call['response']['content'], question_clause)]
            nomem_f = [pool.submit(self._candidate, f'student_{n}', sample, variant,
                                   self.student_system, n + 1, index, True)
                       for n, variant in enumerate(variants, 1)]
            full_system = self.student_system + ('\n' + rule_text if rule_text else '')
            full_f = [pool.submit(self._candidate, f'student_{n}', sample, variant,
                                  full_system, n + 1, index, True)
                      for n, variant in enumerate(variants, 1)]
            nomem_pairs = [future.result() for future in nomem_f]
            full_pairs = [future.result() for future in full_f]
            full_teacher, full_teacher_call = (full_teacher_f.result() if full_teacher_f
                                                else (teacher, teacher_call))
        vote = [teacher] + [candidate for candidate, _ in vote_pairs]
        vote_calls = [teacher_call] + [call for _, call in vote_pairs]
        nomem = [teacher] + [candidate for candidate, _ in nomem_pairs]
        nomem_calls = [teacher_call, tutor_call] + [call for _, call in nomem_pairs]
        full = [full_teacher] + [candidate for candidate, _ in full_pairs]
        full_calls = [full_teacher_call, tutor_call] + [call for _, call in full_pairs]
        vote_selection = self._select(vote, sample.dataset)
        nomem_selection = self._select(nomem, sample.dataset)
        full_selection = self._select(full, sample.dataset)
        written, summary_calls = self._summarize(sample, index, full_selection)
        full_calls.extend(summary_calls)
        self.rules.extend(written)
        greedy_answer = '' if teacher.answer.startswith('__unparsed_') else teacher.answer
        return {'sample_index': index, 'sample_id': sample.sample_id, 'dataset': sample.dataset,
                'image_hashes': teacher_call['image_hashes'], 'original_question': sample.question,
                'choices': list(sample.choices), 'variants': variants, 'retrieved_rules': retrieved,
                'memory_size_before': sum(rule.sample_index < index for rule in self.rules),
                'arms': {
                    'greedy': {'answer': greedy_answer, 'calls': [teacher_call],
                               'usage': self._usage([teacher_call])},
                    'vote_no_rewrite': {**self._selection_record(vote_selection),
                        'open_answer_diagnostic': self._open_answer_diagnostic(vote, vote_selection),
                        'calls': vote_calls, 'usage': self._usage(vote_calls)},
                    'select_no_memory': {**self._selection_record(nomem_selection),
                        'open_answer_diagnostic': self._open_answer_diagnostic(nomem, nomem_selection),
                        'calls': nomem_calls, 'usage': self._usage(nomem_calls)},
                    'tf_ttcl_mm_direct': {**self._selection_record(full_selection),
                        'open_answer_diagnostic': self._open_answer_diagnostic(full, full_selection),
                        'calls': full_calls, 'usage': self._usage(full_calls)}},
                'written_rules': [asdict(rule) for rule in written], 'labels_read': False}
