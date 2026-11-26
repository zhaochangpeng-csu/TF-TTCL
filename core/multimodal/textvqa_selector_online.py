"""Two isolated online streams differing only in open-answer similarity."""
from pathlib import Path

from .image_ops import sha256_file
from .schema import GenerationResponse
from .textvqa_format_factorial import TextVQAFormatRunner
from .tf_ttcl_direct_runner import DirectRunner, _digest
from .tf_ttcl_direct_selection import ScoredCandidate, select_open
from .tf_ttcl_text_embedding import cosine

SELECTORS = ('full_response', 'normalized_answer')
VERSION = 'textvqa_selector_online_v1'


class CachedEmbedder:
    """Exact, process-local immutable embedding cache, including query mode."""
    def __init__(self, backend):
        self.backend = backend
        self.cache = {}

    def encode(self, text, *, query=False):
        key = (text, query)
        if key not in self.cache:
            self.cache[key] = self.backend.encode(text, query=query)
        return self.cache[key]


def choose(candidates, embedder, selector):
    if selector not in SELECTORS:
        raise ValueError('Unknown selector')
    values = [c.response.content if selector == 'full_response' else c.answer for c in candidates]
    teacher = embedder.encode(values[0])
    return select_open(candidates, [cosine(teacher, embedder.encode(x)) for x in values[1:]])


def recorded_candidates(arm):
    calls = [call for call in arm['calls'] if call.get('role')]
    return [ScoredCandidate.from_response(call['role'], call['normalized_answer'],
                                          GenerationResponse(**call['response'])) for call in calls]


class TextVQASelectorRunner(TextVQAFormatRunner):
    def __init__(self, backend, embedder, samples, output, input_sha256, model_revision,
                 embedding_revision, seed=0, selector='full_response'):
        if selector not in SELECTORS:
            raise ValueError('Unknown selector')
        self.selector = selector
        super().__init__(backend, embedder, samples, output, input_sha256, model_revision,
                         embedding_revision, seed=seed, policy='short_final')
        self.embedder = CachedEmbedder(embedder)
        self.config.update(selector_experiment=VERSION, similarity_selector=selector)
        self.config_sha = _digest(self.config)
        self.code_sha = _digest([self.code_sha, sha256_file(str(Path(__file__)))])

    def _select(self, candidates, dataset):
        if dataset != 'textvqa':
            raise ValueError('Selector experiment is TextVQA-only')
        return choose(candidates, self.embedder, self.selector)

    def _sample(self, sample, index):
        row = super()._sample(sample, index)
        candidates = recorded_candidates(row['arms']['select_no_memory'])
        row['fixed_no_memory_selection'] = {
            selector: DirectRunner._selection_record(choose(candidates, self.embedder, selector))
            for selector in SELECTORS}
        return row
