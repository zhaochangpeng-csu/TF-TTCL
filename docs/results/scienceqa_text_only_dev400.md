# ScienceQA train: paired text-only augmentation baseline, seed 0

Completed 2026-10-14 on 400 disjoint ScienceQA-IMG **train** items using
Qwen2.5-VL-7B-Instruct. Run commit: `e0d4ded28bd98ba0972de562d50ade8b3c2920f5`.
The [frozen protocol](scienceqa_text_only_baseline_protocol.md) uses the same
original images, options, model, sampling temperatures and paired seeds in
both voting arms; only the question text differs. All three arms completed
before the label-free audit passed and the isolated labels were opened.
GPU 2 returned to 2 MiB at process exit. This is a development run, not an
untouched validation or a direct TF-TTCL result.

| Arm | Correct / 400 | Accuracy | Logical prompt + completion tokens |
| --- | ---: | ---: | ---: |
| Original-question greedy | 357 | 89.25% | 228,458 + 35,666 |
| Same-text, same-budget vote | 363 | 90.75% | 763,503 + 117,148 |
| Text-augmented, same-budget vote | 363 | 90.75% | 763,777 + 117,789 |

Same-text voting versus greedy corrected 12 and harmed 6, a +1.50 percentage
point difference with paired 95% bootstrap CI [−0.50, +3.50] pp and two-sided
paired randomization p=0.238. Text augmentation versus same-text voting
corrected **one** and harmed **one**, a 0.00 pp difference, CI [−0.75, +0.75]
pp, p=1.0. Neither contrast shows a convincing gain in this single stream.
The voting arms' logical token totals differ by only 915 tokens (about 0.10%
of the same-text total); both pay for the shared rewrite request. Voting uses
roughly 3.3 times the greedy token budget, so its numerical advantage over
greedy cannot be attributed to text augmentation.

Only 107/400 questions had at least one accepted rewrite; 166/800 rewrite
positions changed. Across those 166 paired sampling positions, the final
candidate letter differed only seven times. The 634 fallback positions had
zero paired answer differences, as expected from identical requests and
seeds. The final majority votes differed on only two items. There were 18
format-invalid responses among 2,000 candidate calls, handled by the frozen
vote fallback. The conservative rewrite guard and frequent identical or
unusable model rewrites limit the effective intervention; this is an
**intention-to-treat** result for the frozen pipeline, not proof that all
semantic rephrasings are useless.

## Actual input/output inspection

The two final-vote disagreements are both `Which country is highlighted?`
map questions. IDs `11615` and `276` have visually near-identical 750×625
maps (59 RGB pixels differ) but different option order. Both offline targets
are Palau. The accepted rewrite is `What country is highlighted in the
image?`; the second variant falls back to the original question in both
cases. On `11615`, the text vote changes the same-text vote from Micronesia to
the correct Palau; on `276`, it changes the correct Palau vote to Micronesia.
Model rationales in both items cite similar western-Pacific locations while
assigning them to different countries. This is evidence of answer instability
under a tiny wording/option-context change, not evidence of a robust new
visual capability. These two scored cases were inspected after the complete
run and must not become a tuned special-case rule.

## Decision

Do not claim a text-only improvement or tune this protocol on the already
scored 400 items. As requested, the next experiment keeps this exact frozen
text-only protocol and evaluates all 2,097 ScienceQA-IMG validation items in
a fresh, isolated three-arm run. Its predeclared slices and limitations are
recorded in [the validation protocol](scienceqa_validation_text_only_protocol.md).
The old typed-v3.1 validation run failed its frozen schema audit and cannot
be scored as a successful four-arm baseline or silently repaired in place.

Aggregate JSON: server
`core/exp_res/mm/scienceqa_train/qwen25vl/text_only_v2_seed0_dev400/comparison.json`;
ignored local copy `core/exp_res/inspection/text_only_v2_dev400_comparison.json`.
Input SHA-256: `09135768833bc74a3bf93aee233845b35a1398283b905b8c0f69d9f7f6980011`.
Label SHA-256: `60c37ed2598412d65de171d1e643c056049a59c52b1962b7812e6dc7f9b7ae92`.
