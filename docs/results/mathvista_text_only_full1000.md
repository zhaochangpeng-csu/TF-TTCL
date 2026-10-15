# MathVista testmini: full paired question-only baseline

Completed 2026-10-15 on all 1,000 cached `evalscope/MathVista` testmini
questions with Qwen2.5-VL-7B-Instruct, seed 0. Run Git SHA:
`55cbfbe3d12955fc78854ef0d6d8bf7632b28b92`. The three arms were
committed for every item, the label-free paired audit passed, and only then
were isolated labels opened once by the offline scorer. The owned vLLM
service exited; GPU 2 returned to 2 MiB / 0%. This uses the project's
**local deterministic MathVista extractor/metric**, not a verified match to
the official three-stage MathVista evaluation or its leaderboard. The
[pre-scoring protocol](mathvista_text_only_protocol.md) froze arms and the
multi-choice/free-form slices.

| Arm | Correct / 1,000 | Local accuracy | Logical prompt + completion tokens |
| --- | ---: | ---: | ---: |
| Original-question greedy | 696 | 69.60% | 723,101 + 82,751 |
| Same-text, same-budget vote | 712 | 71.20% | 2,376,354 + 289,717 |
| Text-augmented, same-budget vote | 710 | 71.00% | 2,376,503 + 289,443 |

The primary paired contrast, **text augmentation minus same-text voting**,
is **−0.20 percentage points**: zero corrected and two harmed items,
95% paired bootstrap CI [−0.50, 0.00] pp, two-sided paired randomization
p=0.5. The voting arms differ by only 125 tokens out of about 2.666 million
logical tokens (<0.01%). Same-text voting beats greedy by 16/1,000 (+1.60 pp;
32 corrected, 16 harmed; CI [+0.30, +2.90] pp; p=0.0287), but consumes
about 3.31 times greedy's tokens. That advantage belongs to the additional
sampling/voting budget, **not** question rewriting. The run shares calls and
can replay exact requests, so logical cost is per-arm accounting rather than
the physical bill of three independent deployments.

| Predeclared question type | N | Greedy | Same-text | Text-augmented | Text vs same corrected / harmed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Multiple choice | 540 | 76.30% | 78.52% | 78.15% | 0 / 2 |
| Free form | 460 | 61.74% | 62.61% | 62.61% | 0 / 0 |

The rewrite guard accepted at least one changed variant on 136/1,000
items; the final votes differed on only 4/1,000. There were 303
format-invalid candidate outputs among 5,000 candidate positions (6.06%),
excluded by the frozen valid-vote/fallback rule. These are important
intention-to-treat limitations. In particular, the inherited lexical guard
was designed on English ScienceQA text; accepted Chinese rewrites are **not
independently proven semantically equivalent**. Do not interpret the result
as a controlled test of every possible high-quality paraphrase.

## Actual input/output check after scoring

Only four final answers disagreed. ID `302` asked what the green curve is:
the same-text vote correctly chose logarithmic, while the rewritten vote
chose exponential. ID `986`, a Chinese geometry angle problem, changed the
correct 15° to wrong 17.5°. IDs `446` (Chinese geometry) and `750`
(missing number) changed between two wrong answers. Thus the two net harmed
items are both multiple-choice, and there is no hidden free-form gain. These
cases were inspected **after** scoring; they are diagnostic, not new
confirmatory subgroups or prompt-tuning examples.

## Service-context incident and provenance

The original service used `--max-model-len 8192` and stopped at item index
226 (ID `240`) because its 3500×2387 image produced a 10,783-token input.
No labels were read and 226 complete paired rows remained committed. A
separate service was launched at `--max-model-len 32768`, using the **same
model, original images, prompts, seeds, code SHA and Git SHA**. The runner
resumed from the verified 226-row prefix and completed all 1,000 items;
the scorer re-audited all rows before opening labels. No image was shrunk,
sample skipped or existing prediction overwritten. The first 226 rows used
the 8192 service and the remaining rows the 32768 service. This mixed
serving configuration must be disclosed; a uniform-context rerun is needed
before treating small differences as a robust benchmark finding.

## Decision

Unchanged question-only rewriting has now failed its matched-budget primary
contrast on both ScienceQA-IMG validation (0/2,097 net) and MathVista
testmini (−2/1,000 net). Do **not** keep testing the same frozen intervention
on new public datasets until a positive slice appears. Treat it as a
negative control. The next development experiment should change the
mechanism—e.g. a task-specific visually grounded check with an explicit
abstain/fallback rule—on unscored training/development data, with greedy and
strict budget controls. Any narrow-task gain must be declared in advance
and replicated on a new stream or model; two harmed geometry/curve cases
cannot define a reliable research direction by themselves.

Aggregate JSON: server
`core/exp_res/mm/mathvista/qwen25vl/text_only_v1_seed0_full1000/comparison.json`;
ignored local copy
`core/exp_res/inspection/mathvista_text_only_full1000_comparison.json`.
Input SHA-256:
`7641ca47f2c98f01a3d510de740f3916b0b1fc8b03b14d3cc836f49aa5440e3a`.
Label SHA-256:
`604a9d8c62d60d97878fa18a66c07aac5ea7528491bcb70b266ffeb201154dd3`.
