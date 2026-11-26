# TextVQA fixed-history rule and reminder-position replay

This is a predeclared *exploratory* follow-up on the already scored 400-item
TextVQA train development stream. It uses the completed short-final-policy
stream as an immutable anchor; its earlier labels have already been inspected.
It is not an independent confirmation or an end-to-end online-memory run.

For each question, retain its original image, the four Tutor rewrites, the
exact retrieved historical rules and candidate sampling seeds. Produce five
new Teacher/Student candidates for each cell in a 2×2 comparison:

| Retrieved content | Reminder before rules | Same reminder after rules |
| --- | --- | --- |
| Actual historical rules | `real_before` | `real_after` |
| Neutral placeholders | `neutral_before` | `neutral_after` |

The base policy and reminder are otherwise the same in all four cells.
Placeholders preserve rule counts and approximate each rule's tokenizer length;
they are not guaranteed to be semantically inert or an exact token match.
Report actual prompt and completion tokens per arm. Compare real vs neutral,
and reminder position before vs after, with paired contrasts and interaction.
The old no-memory and memory answers are contextual references, not randomly
assigned arms. The fixed anchor was generated using the original full-response
similarity selection, and the shadow uses the same five-candidate selector.

The shadow never writes rules or reads labels while running. Each item may use
only rules written in earlier anchor samples. Because memory is frozen,
effects are conditional on that particular history: they cannot establish the
effect of a strategy whose decisions would themselves change future memory.
The same reminder in different positions may change generation and selection,
not merely output formatting. The controls do not identify why a rule helps
or harms on their own.

Before a 400-item run, perform an unscored 20-item smoke, audit every request,
image hash, source/anchor hash, sampling seed, history, full log probabilities,
PPL and semantic selection. Run disjoint shards on idle HCU devices and record
logical and deduplicated physical token costs. Release the services before the
final label-free audit. Only a successful complete audit permits offline scoring
with the pinned Meta MMF evaluator and the existing raw-answer extractor; report
that parity with the EvalAI server has not been verified. A later complete
online rerun on fresh questions and images is required for generalization.
