"""Independent online rule-content controls; original TF-TTCL stays unchanged.

Every stream writes its own history after answering. Only the rule text in the
memory-conditioned candidates is substituted; original retrieval, teacher,
Tutor, PPL, summaries, and non-memory candidates are otherwise unchanged.
"""
from copy import deepcopy
from pathlib import Path

from .image_ops import sha256_file
from .schema import SamplingConfig
from .textvqa_rule_gate import GATE_VERSION, match_rule, substitute
from .textvqa_selector_online import SELECTORS, TextVQASelectorRunner, choose, recorded_candidates
from .tf_ttcl_direct_runner import DirectRunner, TUTOR_SYSTEM, _digest, _rule_prompt, parse_variants, retrieve_rules, split_question_clause

POLICIES = ('original', 'targeted', 'matched_random')
VERSION = 'textvqa_rule_control_online_v1'


class TextVQARuleControlRunner(TextVQASelectorRunner):
    def __init__(self, backend, embedder, tokenizer, samples, output, input_sha256,
                 model_revision, embedding_revision, seed=0, policy='original'):
        if policy not in POLICIES:
            raise ValueError('Unknown rule-control policy')
        if tokenizer is None:
            raise ValueError('A pinned tokenizer is required for control length accounting')
        self.rule_policy, self.tokenizer = policy, tokenizer
        super().__init__(backend, embedder, samples, output, input_sha256,
                         model_revision, embedding_revision, seed=seed, selector='full_response')
        self.config.update(rule_control_experiment=VERSION, rule_control_policy=policy,
                           rule_gate_version=GATE_VERSION)
        self.config_sha = _digest(self.config)
        self.code_sha = _digest([self.code_sha, sha256_file(str(Path(__file__))),
                                 sha256_file(str(Path(__file__).with_name('textvqa_rule_gate.py')))])

    def _control(self, retrieved, sample_id):
        if self.rule_policy != 'original':
            return substitute(retrieved, self.tokenizer, mode=self.rule_policy,
                              seed=self.seed, sample_id=sample_id)
        hits = {kind: [r['rule_id'] for r in bucket if match_rule(r['text'])]
                for kind, bucket in retrieved.items()}
        return deepcopy(retrieved), {'gate_version': GATE_VERSION, 'mode': 'original',
                                     'seed': self.seed, 'sample_id': sample_id,
                                     'targeted_ids_by_bucket': hits, 'replacements': []}

    def _sample(self, sample, index):
        # This deliberately mirrors DirectRunner._sample. The exact-input
        # baseline is tested against it; only controlled_rule_text differs.
        sample.validate_images()
        query_vector = self.embedder.encode(sample.question, query=True) if self.rules else ()
        retrieved = retrieve_rules(self.rules, query_vector, index) if self.rules else {
            'positive': [], 'negative': []}
        controlled, gate = self._control(retrieved, sample.sample_id)
        rule_text = _rule_prompt(controlled)
        teacher, teacher_call = self._candidate('teacher', sample, sample.question,
                                                self.teacher_system, 0, index, False)
        vote, vote_calls = [teacher], [teacher_call]
        for number in range(1, 5):
            candidate, call = self._candidate(f'student_{number}', sample, sample.question,
                                               self.student_system, number + 1, index, True)
            vote.append(candidate); vote_calls.append(call)
        vote_selection = self._select(vote, sample.dataset)
        context_prefix, question_clause = split_question_clause(sample.question)
        tutor_call = self._call(TUTOR_SYSTEM, question_clause, (), SamplingConfig(
            temperature=0.7, top_p=0.9, max_tokens=384, seed=self.seed + 100 * index + 1))
        variants = [context_prefix + item for item in parse_variants(tutor_call['response']['content'], question_clause)]
        nomem, nomem_calls = [teacher], [teacher_call, tutor_call]
        for number, variant in enumerate(variants, 1):
            candidate, call = self._candidate(f'student_{number}', sample, variant,
                                               self.student_system, number + 1, index, True)
            nomem.append(candidate); nomem_calls.append(call)
        nomem_selection = self._select(nomem, sample.dataset)
        if rule_text:
            full_teacher, full_teacher_call = self._candidate(
                'teacher', sample, sample.question, self.teacher_system + '\n' + rule_text,
                0, index, False)
        else:
            full_teacher, full_teacher_call = teacher, teacher_call
        full, full_calls = [full_teacher], [full_teacher_call, tutor_call]
        for number, variant in enumerate(variants, 1):
            system = self.student_system + ('\n' + rule_text if rule_text else '')
            candidate, call = self._candidate(f'student_{number}', sample, variant,
                                               system, number + 1, index, True)
            full.append(candidate); full_calls.append(call)
        full_selection = self._select(full, sample.dataset)
        written, summary_calls = self._summarize(sample, index, full_selection)
        full_calls.extend(summary_calls)
        self.rules.extend(written)
        greedy_answer = '' if teacher.answer.startswith('__unparsed_') else teacher.answer
        row = {'sample_index': index, 'sample_id': sample.sample_id, 'dataset': sample.dataset,
               'image_hashes': teacher_call['image_hashes'], 'original_question': sample.question,
               'choices': list(sample.choices), 'variants': variants, 'retrieved_rules': retrieved,
               'controlled_rules': controlled, 'rule_gate': gate,
               'memory_size_before': sum(rule.sample_index < index for rule in self.rules),
               'arms': {
                   'greedy': {'answer': greedy_answer, 'calls': [teacher_call], 'usage': self._usage([teacher_call])},
                   'vote_no_rewrite': {**self._selection_record(vote_selection),
                       'open_answer_diagnostic': self._open_answer_diagnostic(vote, vote_selection),
                       'calls': vote_calls, 'usage': self._usage(vote_calls)},
                   'select_no_memory': {**self._selection_record(nomem_selection),
                       'open_answer_diagnostic': self._open_answer_diagnostic(nomem, nomem_selection),
                       'calls': nomem_calls, 'usage': self._usage(nomem_calls)},
                   'tf_ttcl_mm_direct': {**self._selection_record(full_selection),
                       'open_answer_diagnostic': self._open_answer_diagnostic(full, full_selection),
                       'calls': full_calls, 'usage': self._usage(full_calls)}},
               'written_rules': [vars(rule) for rule in written], 'labels_read': False}
        candidates = recorded_candidates(row['arms']['select_no_memory'])
        row['fixed_no_memory_selection'] = {selector: DirectRunner._selection_record(
            choose(candidates, self.embedder, selector)) for selector in SELECTORS}
        return row
