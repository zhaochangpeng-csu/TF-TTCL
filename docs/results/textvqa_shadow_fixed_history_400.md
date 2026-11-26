# TextVQA fixed-history rule and reminder-position replay

The exploratory [predeclared protocol](textvqa_shadow_fixed_history_protocol.md)
completed on the previously scored 400-item train development stream. Its
eight HCU shards passed the full label-free request, rule-history, model,
embedding, selection and PPL audits before isolated labels were scored. The
model services exited before the CPU audits.

| Historical content | Reminder before rules | Reminder after rules |
| --- | ---: | ---: |
| Real rules | 71.33% | 72.80% |
| Length-approximate neutral placeholders | 75.78% | 76.23% |

Scores use the pinned Meta MMF evaluator and raw extracted selected answers;
EvalAI server parity is unverified. Averaged across reminder positions, real
rules score 3.94 percentage points lower than neutral placeholders (paired
bootstrap 95% CI -6.33 to -1.51 pp). This is consistent with a detrimental
effect of the particular rule content on the fixed anchor history, not proof
of an end-to-end online memory effect. Placeholders are not guaranteed to be
semantically inert. Both comparisons differ by less than 1% in logical
prompt-plus-completion tokens.

For the real-rule condition, putting the reminder after the rules increased
the fraction of selected outputs that meet the strict final-answer-line format
from 43.75% to 96.00%. The score changed by +1.48 pp, with a paired bootstrap
95% CI of -1.78 to +4.80 pp. In the neutral condition, corresponding format
rates were 41.75% and 33.75%; the reminder-position contrast is therefore not
a general guarantee of format compliance. The interaction on the accuracy
score had a CI spanning zero. The previously generated no-memory anchor scored
79.40% and the historical memory anchor 71.58%; these are contextual reference
outputs, not newly randomized shadow arms.

The 400 items and particular anchor history had already been examined before
this experiment, and changing a decision in a true online stream could alter
all later rules. Confirm any candidate method with newly generated independent
online memories on disjoint questions and images. The full aggregate, paired
statistics, token counts, hashes and audit provenance are recorded in
[the result JSON](textvqa_shadow_fixed_history_400_mmf.json).
