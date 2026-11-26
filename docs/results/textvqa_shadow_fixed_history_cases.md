# TextVQA fixed-history shadow: exploratory case inspection

The paired replay used the same short-final-policy 400-question development
stream, fixed image, Tutor variants, sampling seed and previously retrieved
online history for all four shadow arms. `real_before` and `real_after` use
the recorded rules; `neutral_before` and `neutral_after` replace their text
with approximate-length neutral placeholders. The suffix specifies whether
the identical format reminder appears before or after the rules. The replay
writes no new rules. These cases were selected after viewing scores and are
illustrative, not representative or independent evidence of causation.

Scores below use the pinned Meta MMF TextVQA evaluator on the selected raw
answer, with the existing last-line fallback. Scores measure matching the ten
annotation strings, not full semantic correctness. EvalAI server parity has
not been independently verified.

| ID and question | Real before | Real after | Neutral before | Neutral after |
| --- | --- | --- | --- | --- |
| 25794: “how many pounds?” | `14 LBS` (0.0) | `14 LBS` (0.0) | `14` (1.0) | `14` (1.0) |
| 29180: “what is the time?” | `10:00` (0.0) | `10:00` (0.0) | `11:00` (1.0) | `11:00` (1.0) |
| 5949: “what kind of sauce is in the bottle?” | `Tabasco Sauce` (0.0) | `Tabasco Sauce` (0.0) | `tabasco` (1.0) | `tabasco` (1.0) |

For ID 25794, nine annotations are `14` and one is `ion`. The retrieved
`negative_30` rule says: “Avoid providing exact weight measurements without
context or units.” Its accompanying explanation specifically mentions LBS.
`14 LBS` and `14` may be semantically equivalent given the question; the
score difference is therefore not proof that one is a worse visual answer.

For ID 29180, eight annotations contain `11:00` (including one with trailing
whitespace), one is `11 ` and one is `11:01`. The retrieved `negative_84`
rule says: “Avoid providing unnecessary details beyond the hour and minute
hands when answering "What time is it?"” The two outputs differ in their
stated time, but the image was not independently inspected for this casebook;
the presence of the rule does not establish why the output changed.

For ID 5949, the annotations include `tabasco`, `tobasco`, and `hot`;
the retrieved `negative_51` rule says: “Avoid unnecessary details that do not
contribute to identifying the brand name.” `Tabasco Sauce` may express an
equivalent answer to `tabasco`, despite the exact-match scoring difference.
The rule's presence alone does not show that it caused the generated text.

## Context and costs

For the completed 400-item anchor, 800 rules were written. A case-insensitive
keyword heuristic counted 216 rules containing `reason`, `explain`, or
`explanation`; 300 containing `context`, `detail`, or `elaborat`; and 113
containing `avoid` followed within 60 non-period, non-newline characters by
`reason` or `explan`. Categories overlap; these are surface matches, not
human-validated semantic categories. Among the 400 questions, 394 retrieved
at least one rule matching the last pattern. Retrieval includes repeated rules
across questions; it does not represent 394 independent source rules.

Across 23,070 retrieved rule instances, the real and placeholder text had
486,087 and 487,152 tokens respectively under the model tokenizer. The mean
per-rule absolute difference was 0.668 token, maximum one token. Logical
prompt-plus-completion totals were 4,880,814 real-before versus 4,885,691
neutral-before (0.10% difference), and 4,911,222 real-after versus 4,879,678
neutral-after (0.64% difference). Total budgets were within 5%, while output
token counts and latency differed; placeholders may themselves alter model
behavior. Logical arm totals share requests and must not be added to claim
physical cost. This one pre-scored development stream cannot establish
generalization to a newly rebuilt online memory.
