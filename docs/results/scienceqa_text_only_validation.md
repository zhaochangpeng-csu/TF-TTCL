# ScienceQA-IMG validation: full text-only paired baseline

Completed 2026-10-14, Qwen2.5-VL-7B-Instruct, seed 0, GPU 2, run Git SHA
`99c22d16c6de407c13c1095fd5d49e50c536bce5`. The three arms shared
the same 2,097 ordered validation items, original image bytes and option
lists. The 20-item unscored smoke and full-run label-free paired audit passed;
only then were isolated validation labels opened once for offline scoring.
The owned vLLM service exited and GPU 2 returned to 2 MiB / 0%.

This is a full-set follow-up to the scored 400-item **train** development
experiment, not an untouched benchmark: some validation inputs and prior
label-free outputs had already been inspected. The earlier typed-v3.1
validation run failed its frozen schema audit and was **not** reused or
scored. The [pre-scoring protocol](scienceqa_validation_text_only_protocol.md)
defines these arms, cost accounting, and three secondary slices.

| Arm | Correct / 2,097 | Accuracy | Logical prompt + completion tokens |
| --- | ---: | ---: | ---: |
| Original-question greedy | 1,855 | 88.46% | 1,196,563 + 182,831 |
| Same-text, same-budget vote | 1,868 | 89.08% | 3,999,679 + 602,173 |
| Text-augmented, same-budget vote | 1,868 | 89.08% | 4,001,184 + 604,735 |

The primary comparison, text augmentation minus same-text voting, is **0.00
percentage points**: four corrected and four harmed items, 95% paired
bootstrap CI [−0.286, +0.286] pp, two-sided paired randomization p=1.0.
Both voting arms exceed greedy by 13/2,097 (+0.620 pp), but the same-text
comparison has CI [−0.191, +1.431] pp and p=0.167. More importantly,
voting costs about 3.34 times greedy's logical tokens, so this is not
evidence for the text intervention. The two voting arms differ in total
logical tokens by only 4,067/4,601,852 (0.088%). Identical requests could
be replayed from immutable traces; logical cost is the paired-arm deployment
accounting, not the physical bill of this shared run.

The rewrite guard accepted at least one variant on 617/2,097 questions
(29.42%) and 947/4,194 variant positions (22.58%). Yet the final votes
differed on only 8/2,097 items (0.38%). There were 173 format-invalid
candidate outputs among 10,485 candidate positions (1.65%); the frozen
valid-vote/fallback rule handled them. These figures suggest the current
semantic rewrite is too weak or too often rejected to substantially alter
the model's decision. They do not prove that all text augmentations fail.

## Predeclared secondary slices

| Label-blind question selector | N | Greedy | Same-text | Text-augmented | Text vs same corrected / harmed | Holm-adjusted p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Highlighted country/state/continent map | 122 | 76.23% | 77.05% | 76.23% | 2 / 3 | 1.0 |
| Contains “magnet” | 186 | 69.35% | 68.82% | 68.82% | 0 / 0 | 1.0 |
| Begins “What is the capital of” | 318 | 98.74% | 99.69% | 100.00% | 1 / 0 | 1.0 |

The capital slice's apparent +0.31 pp is exactly **one** corrected item,
with unadjusted and adjusted p=1.0; it is not a defensible narrow-task gain.
The magnet slice is a keyword bucket, not a verified physical-reasoning
category, and its low accuracy does not itself establish visual causality.
The highlighted-map hypothesis arose from two opposed train cases and is
not an independent discovery; on validation it trends slightly negative.

## Actual input/output check after scoring

The eight vote disagreements are five highlighted-country maps (two
corrections, three harms), two specific-humidity maps (one correction, one
harm), and one New Mexico capital question (one correction). For example,
ID `13047` changed a wrong Australia vote to correct New Zealand; IDs
`7101` and `2338` went the other way on nearby-country maps. ID `2325`
corrected Albuquerque to Santa Fe. The accepted map rewrite was generally
the minor wording change from “Which country is highlighted?” to “What
country is highlighted in the image?”, with the second variant falling back
to the original. This is answer instability at a voting margin, not a
demonstrated improvement in image understanding. These examples were
inspected **after** scoring and cannot define a confirmatory slice here.

## Decision and next experiment

Stop scaling this unchanged text-only method: train development (363/400
versus 363/400) and full validation (1,868/2,097 versus 1,868/2,097) both
show zero net gain over the budget-matched arm. Do not search additional
validation categories to manufacture a positive headline. Keep this as a
negative control and a properly budgeted baseline.

Next, develop a **separate** visually grounded, no-memory method on unused
ScienceQA-IMG train items. The label-blind inventory contains 565 train
questions with “magnet”, 338 of which are outside the 2,500 IDs in existing
development manifests; 195/565 are routed by the strict attract/repel
facing-poles operator, including 116 unused. A new development split can
compare raw greedy, budget-matched repeated/structured prompting, and an
image-grounded facing-pole procedure with explicit abstention on unreadable
or inconsistent observations. Report the broad magnet bucket and strict
operator coverage separately; a win on 116 precise cases needs replication,
not a broad multimodal claim. Prior typed certification made a false visual
transcription on ID `8823`, so syntax-valid or repeated same-image OCR must
not be accepted as independent visual evidence. Freeze this new protocol
before inspecting its held-out train labels, and only move to validation or
another model if it first beats the cost control without excessive harms.

Aggregate scoring JSON: server
`core/exp_res/mm/scienceqa_validation/qwen25vl/text_only_v2_seed0_full2097/comparison.json`;
ignored local copy
`core/exp_res/inspection/text_only_validation_comparison.json`. Input
SHA-256: `12d361242e93e14c67620fe3c270054244fa8d4cc71618af29ee1a6ec33a0e19`.
Label SHA-256: `d17cc74729e2d6bb427b8709eb87486c65bf8563f317343b6dc684acdd580897`.
