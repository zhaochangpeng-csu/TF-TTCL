# TextVQA format and memory: development results and rescoring

The frozen 400-item train development stream completed on 2026-11-26.
Both independent ordered streams passed label-free audits before scoring:
400 predictions and 800 rule writes each. This is one developmental stream,
not a new validation set. The fixed prompt policy affects generation, candidate
selection and rule-writing, not just the last line of the final answer.

| Policy | No memory | Online memory | Memory minus no memory |
| --- | ---: | ---: | ---: |
| Original | 72.28% | 65.35% | -6.93 pp |
| Short final answer | 79.40% | 71.58% | -7.83 pp |

These post-hoc results apply the pinned Meta MMF TextVQA evaluator to the
selected candidate's *raw extracted answer*. They are exploratory: candidate
selection and memory generation were not rerun, and changing answer extraction
afterward affects only the scored output of the already-selected candidate.
Extraction uses the existing final-answer parser and falls back to the last
line if the marker is absent. The evaluator is not verified against the EvalAI
server. Evaluator revision, exact hash, confidence intervals and results using
the original recorded normalized answers are in
[the aggregate JSON](textvqa_format_factorial_v2_mmf_rescore.json).

Previously reported local scores were 70.48%, 62.78%, 76.85% and 68.95%,
respectively. The old pipeline removed punctuation from predictions before
applying a different normalization to references. For example, punctuation in
numbers or slashed text could cause asymmetric comparisons. The new scores
must not silently replace earlier reports, and changes cannot be attributed
to the evaluator alone: they also use the saved raw extracted answer instead
of the already-normalized prediction.

Under the MMF raw-answer rescore, the short-policy contrast is +7.13 pp
without memory (paired bootstrap 95% CI +4.15 to +10.30 pp) and +6.23 pp
with memory (+2.35 to +10.08 pp). The memory contrasts are -6.93 pp
(-10.85 to -3.10 pp) and -7.83 pp (-11.28 to -4.58 pp), respectively.
These are descriptive within this one development stream. Policies also ran
on different HCU devices; device and policy are confounded. The short policy
used approximately 8.0% and 6.2% more logical prompt-plus-completion tokens
for no-memory and memory arms, respectively, exceeding the 5% matching target.

In the short-policy memory arm 224/400 selected outputs lacked the exact
`Final answer:` marker, compared with 35/400 in the original-policy memory
arm and 1/400 in each no-memory arm. This is a format diagnostic, not an
error count. Looking at selected cases also shows answer-content changes, so
neither the score loss nor the policy effect can be attributed entirely to
format. Fixed-history counterfactual replay and a fresh disjoint stream are
needed before claiming a mechanism or generalization.
