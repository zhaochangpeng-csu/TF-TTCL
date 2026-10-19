# Faithful TF-TTCL multimodal direct transfer: TextVQA train development stream

This is an **exploratory development result**, not an unseen validation result.
The frozen Qwen2.5-VL-7B-Instruct protocol ran on the 400 predeclared,
label-isolated TextVQA train questions at seed 0. All four arms use the same
original image; only the question text is paraphrased. The full arm uses the
original open-domain selection logic: rank student **full responses** by
similarity to the Teacher full response, select representatives by generation
PPL, then retrieve top-30 positive and top-30 negative text rules from earlier
questions. No image augmentation, visual verification, memory gate, or final
refinement was added.

The metric is the project's deterministic TextVQA soft score; exact parity
with the official EvalAI answer processor has not yet been verified. These
figures are not leaderboard claims.

The first service had an 8,192-token limit and stopped before sample 173. A
32,768-token service was started; a memory-conditioned replay differed, so
the old 173-row prefix was discarded for scoring and all 400 items were run
afresh. The new run passed the label-free audit (`processed=400`,
`labels_read=false`, 800 rule writes). Only then were isolated targets opened
for offline VQA accuracy. The run's code SHA was
`4d86753b313bd1b52286bd8a43cf53dc33f96830`.

| Arm | Local TextVQA score | Change vs greedy | Logical prompt tokens |
| --- | ---: | ---: | ---: |
| Greedy | 72.50% | — | 434,043 |
| Five candidates, no paraphrase | 72.53% | +0.03 pp | 2,171,815 |
| Original candidate selection, no memory | 71.28% | −1.23 pp | 2,207,218 |
| Full `tf_ttcl_mm_direct` | 65.43% | −7.08 pp | 4,764,130 |

The full arm is 5.85 percentage points below the no-memory selector: paired
bootstrap 95% CI for the difference is [−9.53, −2.18] pp; paired
randomization *p*=0.0017. Against greedy, the difference is −7.08 pp,
95% CI [−10.73, −3.38] pp, *p*=0.0003. These are within-stream tests on one
train subset and one ordering seed, not a claim about TextVQA validation or
another model. The full arm used 4,882,639 logical prompt-plus-completion
tokens, 10.77× greedy; shared requests mean these logical arm totals should
not be added to infer physical cost.

## Input/output inspection and diagnostic limits

Relative to the no-memory selector, retrieved rules changed 151/400 final
answers: 24 items improved, 49 worsened, and 78 retained the same VQA score.
The full arm changed 142/400 answers relative to greedy. In the 49 worsened
memory-vs-no-memory items, 35 outputs contain at least one annotated answer
as a normalized substring. This is only a post-hoc heuristic, not a recovered
accuracy metric: it suggests that extra words and answer extraction contribute
to the failure, but some contained strings may still be invalid responses.

Representative cases: for “what is the lowest number shown on the middle
device?”, the no-memory answer was `0`, while the full arm returned “lowest
figure on middle gadget is 0.” Both express the number, but only the short
answer matches TextVQA's answer format. For “what company is advertising on
the green banner?”, the answer shifted from `td comfort zone` to `td bank`,
which is a substantive error rather than just verbosity. Conversely, on “is
this a telephone booth?”, memory shortened “yes it is telephone booth” to
`yes`, improving the score. Mean final-answer length grew from 1.97 words
without memory to 2.52 with memory; harmed cases averaged 5.14 words, while
improved cases averaged 1.42.

The top-two students chosen by full-response similarity agreed with a
short-answer similarity ranking in 24.25% of no-memory items and 34.00% of
full-memory items. This disagreement is a diagnostic of the original
open-domain selection mechanism on short-answer VQA; it does not by itself
identify the correct answer. The full arm did not improve monotonically over
the stream: its first/middle/final-third scores were 61.13%, 68.80%, and
66.34%, versus greedy's 69.85%, 76.32%, and 71.34%.

## Decision

Do not present the faithful direct transfer as a positive TextVQA result or
expand this exact protocol to unseen validation. The next small, controlled
development experiment should isolate **short-answer preservation** from
rule-memory relevance: evaluate an answer-only output constraint and/or
label-free answer canonicalization under a matched token budget, while
keeping the direct-transfer arm unchanged as a baseline. Use fresh train
development IDs before treating any such fix as evidence of generalization.
The scored 400 items are for diagnosis, not repeated prompt tuning or
confirmatory validation.
