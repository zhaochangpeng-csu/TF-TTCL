# ScienceQA train: label-free audit of transfer-memory content

Date: 2026-10-14. Source: the completed 400-item `transfer_online_v1_seed0_dev400`
development stream. This audit replays recorded writes and retrievals without
model calls, labels, or access to the current/future item's memory. It is a
post-hoc diagnosis of an already scored train stream, **not** a new evaluation.
The exact JSON and example list are ignored local/server artifacts listed below.

The preceding [four-arm result](scienceqa_transfer_online_seed0.md) showed
91.75% for lexical noisy memory versus 91.25% for random memory and 90.50% for
no memory. A [word-count-matched historical-text placebo](scienceqa_transfer_placebo_seed0.md)
reached 91.50%; the noisy/placebo paired difference was one correct item and
not statistically significant. These results do not establish semantic rule
transfer.

## What the label-free trace shows

| Diagnostic | Count | Interpretation limit |
| --- | ---: | --- |
| Noisy-bank writes | 357/400 | Parser acceptance, not rule correctness. |
| Positive rule starts with an image/map/diagram assertion | 123/357 | Lexical proxy for copying an old observation. |
| Either rule mentions the source pseudo-positive answer | 34/357 | Some mentions can be valid general facts. |
| Positive rule mentions its paired pseudo-negative answer | 5/357 | A mention is not necessarily a contradiction. |
| Noisy retrievals with the same official skill as the target | 468/797 | Skill similarity does not prove rule applicability. |
| Random retrievals with the same official skill | 18/797 | The lexical retriever does find related tasks. |
| Typed-gated retrievals with the same official skill | 27/533 | The tiny gated bank is often inapplicable. |

One clear, **label-free** inconsistency is source ID `19528`: the stored
pseudo-positive answer is `Dominica`, the pseudo-negative is `Grenada`, yet
the positive rule asserts that the highlighted area corresponds to `Grenada`.
The original parser accepted this item. Other four negative-answer mentions
include conditional wording and are **not** counted as proven contradictions.
This example explains a contamination mechanism, not the causal effect size of
the rule on later predictions.

We dry-ran a conservative rule-form filter over the existing stream. It keeps
only positive rules beginning with a checking/procedural phrase, rejects
obvious image/map assertions, and rejects an unconditional mention of the
paired negative answer. It accepted 145/357 writes and, in a label-free BM25
replay, produced 787 retrieval events, of which 307 matched the target's
official skill. No VLM final answer was regenerated, so there is **no
accuracy estimate** for this filtered bank. A capitalization screen still
flagged 10 accepted rules, including instructions about the old problem's
`Tyrone and Mason` and `Terrell and Sophie`. Other flags are ordinary terms
such as `Punnett`; the screen is deliberately only a review cue. Conversely,
lower-case or non-name source facts can evade it. The filter is therefore
diagnostic and must not be deployed as a solved trust gate.

The actual inputs and outputs point to two distinct failure modes: (1) a
retriever can find the right *task family* while feeding the target an old
instance's answer or entity; (2) even a well-formed procedure can rely on a
wrong visual observation. The previously recorded magnet certificate for ID
`8823` illustrates the second failure: real and “fidelity” image hashes were
identical, and an OCR position error survived that pseudo-repetition.

## Decision before another GPU run

Do **not** run ScienceQA validation, MathVista, TextVQA, a second model, or a
new 400-item memory evaluation with this parser. First make the retrieved
payload an answer-free, source-instance-free **checking operation**, with a
machine-checkable representation of when it applies and which current-image
evidence must be read. Keep the old question, options, answer, image and
unabstracted rule in an audit-only record; none may enter the target prompt.
Make source/target temporal isolation and forbidden source-token checks hard
tests. An applicability check must use only the current inference input; the
official `skill` field here is an *offline audit proxy*, not a permissible
retrieval oracle unless it is genuinely supplied in every target task.

Before looking at any fresh score, freeze the schema, prompt, validator,
retriever, split IDs, seeds, and cost budget. Then run a 20-item **unscored**
format/leakage smoke on a new disjoint ScienceQA-IMG train slice, followed by
paired no-memory, random-operation, answer-free placebo, budget-matched
reflection, and applicable-operation arms. Preserve the same candidate traces
and final token budget across arms; evaluate only after every arm completes.
Use multiple order seeds to test online transfer. If applicable operations do
not beat placebo and random controls repeatably, stop the memory claim and
reconsider whether the stronger research question is visual evidence
verification rather than online rule accumulation.

Exact label-free audit: server
`core/exp_res/mm/scienceqa_train/qwen25vl/transfer_online_v1_seed0_dev400/memory_audit_v4.json`
and `.md`; local ignored copies
`core/exp_res/inspection/memory_audit_v4.json` and `.md`. The script is
`evaluate/multimodal/audit_transfer_memory.py`. The filter dry run lives in
`core/multimodal/transfer_quality.py`; it is intentionally not connected to
the inference runner.
