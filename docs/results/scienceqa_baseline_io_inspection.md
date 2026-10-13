# ScienceQA baseline I/O review — 2026-10-13

This is **unscored input/output inspection**, not an accuracy result. No new
validation labels were opened. The frozen four-arm regression stopped at its
predeclared protocol audit; do not use its partial arms as a successful result.

## What the model actually receives

Canonical JSONL fields are `id`, `question`, `image`, `choices`,
`question_type`, `grade`, `subject`, `topic`, `category`, and `skill`.
The inference file has no `answer`, `lecture`, or `solution`. Official hints
are preserved as `Context:\n...\n\nQuestion:\n...`; hints are task input, not
the official explanation. 1,331 of 2,097 validation items contain that context.
There are 712 two-choice, 588 three-choice, 753 four-choice and 44 five-choice
items; 1,287 natural-science, 762 social-science, 48 language-science items.

The client sends a system instruction, then a user message whose content is
image(s) followed by text. The user text contains context/question, `(A)` etc.
options, and `Put only the correct option letter after Final answer:`.
Grade/subject/skill metadata are NOT included in this answer prompt. The
answer system asks for a final-answer line but does not require reasoning.

This matters: 2,073/2,097 standalone greedy outputs match exactly
`Final answer: <letter>` (other answer-only formats were not counted in that
number). Mean completion length is 10,419/2,097 = 4.97 tokens. Calling these
outputs "reasoning trajectories" would be inaccurate. Distilling a reasoning
contrast from letters alone invites a newly invented, post-hoc explanation.
Future reasoning prompts must be identical across all transfer arms; simply
requesting reasoning can itself change accuracy and cost.
The extended inspection also counts bare letters: 2,095/2,097 greedy outputs
contain no reasoning. All 2,097 fidelity-candidate image hashes are identical
to their real-image hashes: this particular re-encoding provides repeated
model calls, **not a distinct visual perturbation**.

## Concrete cases, selected without correctness labels

The case snapshot selected the first eight stream items, first supported
operator examples and first two candidate disagreements: 12 cases total.
The shared arms' candidate calls were recovered from exact immutable request
traces (including system/user text, image hash, sampling and raw response).
Standalone greedy has no request trace; its request is explicitly marked
**reconstructed, not a captured wire request**. Selected output is recorded.

| ID | Task/input | Observed behavior and limitation |
| --- | --- | --- |
| 6571 | A letter-number city map; input question literally says `Which i in column 3?` | All six candidates output D; picture is needed to inspect the grid. The wording looks incomplete. Do not silently repair it: provenance must be checked separately before claiming a conversion or dataset error. |
| 16457 | Context already says the organism belongs to the plant kingdom; asks whether its cells have a nucleus | All six candidate raw outputs are just `B`. The photograph shows a fern on rock, not a cell nucleus. Text plus biological knowledge appears sufficient; lack of image sensitivity here need not imply bad reasoning. This is an inspection hypothesis, not a measured text-only result. |
| 10082 | Gobi Desert passage explicitly contrasts weather with climate and supplies definitions | Six B answers. A visual-support gate for every conclusion would be inappropriate for a passage/knowledge question merely accompanied by a picture. Test with a separate text-only ablation, not with validation prompt tuning. |
| 10147 | Two horizontal magnets; choices A=attract, B=repel | Six B answers, yet answer-blind N/S transcription plus the frozen facing-pole operator computes A. Facts are stable, counterfactual support drops, but `computed_answer_not_proposed` forces fallback B. The visible facing labels are S and N. This exposes correlated candidate failure and a candidate-set bottleneck, without opening GT. It is not a substitute for official accuracy scoring. |
| 18331 | Same magnet question, but choices A=repel, B=attract and another picture | Six B answers; here the fixed operator also computes B. The letter B has a different semantic meaning than in 10147. Never persist an isolated option letter as a transferable answer. |
| 19423 | First candidate-disagreement case: magnet question | Five B candidates versus one A; independently transcribed facing poles S/S lead the operator to A and the evidence arm selects it. This is a method disagreement, not a scored correction. |
| 14157 | Refrigerator temperature table, pseudo-candidates split 3/3 | Real/fidelity observations put a nonnumeric row-label header in `columns`, yielding 3 columns but 2 values per row; semantic completeness fails. On the preceding unrelated image, a 9-digit cell violates the schema's 6-integer-digit constraint. Valid JSON is not necessarily a valid or complete evidence certificate. |

## Research implications, not yet validated claims

1. **Image-bearing is not image-dependent.** Distinguish visually necessary
   tasks, text/knowledge-sufficient tasks, and ambiguous cases. Do not force all
   answers to become image-entailed, and do not call image independence an error.
2. **Text-match retrieval can transfer the wrong conclusion.** 94 groups,
   containing 362 items, have identical question-and-options text but may use
   different images. Retrieve a procedure such as inspecting facing endpoints,
   then re-instantiate its evidence on the current image; never copy old letters
   or old observed poles. Count image differences before claiming every group
   is a visual conflict, and score transfer harm only offline.
3. **More candidates do not imply independent evidence.** Six-candidate answer
   disagreement occurs on only 128/2,097 items; repeated deterministic prompts
   and lossless re-encoding may produce effectively repeated answers. Inspect
   request/image hashes rather than treating each call as a separate witness.
4. **Reliable selection is not identical to useful memory.** An uncertified
   answer can suggest a useful checking procedure; a correct conclusion can
   store an image-specific shortcut. Compare net downstream transfer, rather
   than demanding all pseudo-positive answers be GT-correct.
5. **The current verifier is narrow.** Its lexical router supports 91/2,097
   items (70 facing-poles, 18 temperature-delta, 3 table-extreme). Manually
   encoded physics operators are not a general frozen-VLM self-verifier.

## Frozen validation stop and reproducibility

All three executed arms completed 2,097 samples. Shared budget/evidence each
use 4,910,539 total tokens; traces/configurations match; no observation exposed
questions/options/candidate answers, and no memory or policy leak was found.
364 observations contain 38 incomplete responses and 1 `parse_error` (a
schema-contract violation, not broken JSON syntax). There are 65 stable-fact,
64 counterfactual-drop, 39 gate-passing, and 9 override cases. None are scored.
The audit sets `protocol_ok=false`, launcher exits 5, independent budget arm
does not start, evaluator is never called. GPU 2 returned to 2 MiB / 0%.

Frozen run SHA: `aed4fb73331fc3f80dc196f87c1f667d3b391c34`.
Input JSONL SHA: `12d361242e93e14c67620fe3c270054244fa8d4cc71618af29ee1a6ec33a0e19`.
Shared prompt/schema/operator SHA:
`dc4902f3afbf980bcf5c2efa849d8fa93ebf55c2c14cf7790f4ec22d8a6da076`.
Raw snapshot: server `core/exp_res/mm/inspection/validation_typed_v3p1_unscored.json`;
local `core/exp_res/inspection/validation_typed_v3p1_unscored.json` (both ignored
by Git). Extended counts are in server
`core/exp_res/mm/inspection/validation_typed_v3p1_unscored_v2.json`.
Images and large trace caches are not committed.

Run `scripts/inspect_baseline_io.py --help` to create new immutable JSON/Markdown
case cards; it never makes a model call or accepts a label path. Keep this
validation frozen. Newly observed limitations motivate separate **development**
experiments, not a retuned claim of untouched validation performance.
