# TextVQA fresh selector stream: exploratory memory case inspection

These examples were selected **after scoring** the new 400-question train
development stream. They illustrate hypotheses rather than representative
error categories or a causal effect of an individual rule. The independently
built `normalized_answer` stream is used throughout. Scores below are the
pinned Meta MMF VQA soft score on raw extracted answers; EvalAI server parity
has not been independently verified. The images for these cases have not been
independently inspected, and annotation-string agreement does not by itself
establish visual or semantic correctness.

Relative to its no-memory selector, the memory arm raised the score on 17
items, lowered it on 57, and left it unchanged on 326. Among the 57 lowered
items, 33 selected responses lacked the exact `Final answer:` marker; the
other 24 did contain it. The mean extracted answer length was 4.68 words for
lowered items, 1.47 for raised items, and 2.29 for unchanged items. These
associations do not show that formatting caused a score change.

| Item | Question | No-memory extracted answer and score | Memory extracted answer and score |
| --- | --- | --- | --- |
| 8280 | What route does this bus take? | `15` (1.0) | `The bus takes route 15.` (0.0) |
| 16379 | What company is shown in the middle at the bottom right? | `USDA` (1.0) | `USDA Ad Council` (0.0) |
| 12847 | Can you enter your PIN here? | `Yes` (1.0) | `123` (0.0) |

For item 8280, four of the ten raw annotations are `15`; the other six name
the destination, including one `mainstone 15`. The first retrieved positive
rule says: “**Ensure** that the answer includes clear reasoning, specific
details, and a direct, concise final answer to maintain clarity and accuracy.”
The longer memory answer contains the same route number, but differs from
the annotation strings under MMF scoring. The retrieved rule and the output
are observations, not evidence that the rule caused the change.

For item 16379, nine annotations are `usda` and one is `the usda`. A
retrieved negative rule says:
“Avoid missing key information like the full name of the company in the
answer.” The memory output includes additional text, but visual evidence was
not independently checked. For item 12847, eight of ten annotations say
`yes`, one says `no`, and one says `notecipher`; the memory answer changed
content, not merely length. One retrieved negative rule says: “**Avoid
repeating the user's question exactly without providing additional context or
clarification.**” Again, the observed rule does not establish causation.

## Position in the online stream

| Consecutive sample indices | No-memory score | Memory score | Lowered items | Raised items |
| --- | ---: | ---: | ---: | ---: |
| 0–99 | 83.1% | 78.4% | 7 | 2 |
| 100–199 | 80.4% | 73.7% | 11 | 4 |
| 200–299 | 83.2% | 69.2% | 18 | 5 |
| 300–399 | 78.3% | 66.8% | 21 | 6 |

The later segments show more lowered items in this particular order. Their
question composition also differs, and this single stream cannot establish
that memory accumulation caused the pattern. Independent permutations are
needed to distinguish order and composition effects.

The mean top-1 retrieval similarity of positive rules was approximately
0.544 for lowered items and 0.550 for raised items; for negative rules the
corresponding means were 0.553 and 0.563. These groups contain only 57 and
17 items respectively, so the differences neither establish a reliable gate
nor validate any chosen similarity threshold.
