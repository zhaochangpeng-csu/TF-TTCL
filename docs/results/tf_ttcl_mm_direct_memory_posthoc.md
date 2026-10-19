# TF-TTCL multimodal direct transfer: post-hoc ScienceQA memory analysis

This analysis was run **after** the complete 400-item, four-arm, label-free
inference and audit reported in
[the main development result](tf_ttcl_mm_direct_scienceqa_dev400.md). It is
diagnostic, not a new validation result or permission to tune on these labels.
The per-item casebook is stored only in the server's ignored experiment output
as `posthoc_memory_changes_v2.json`.

Adding retrieved rules to the no-memory selector changed the final answer on
24/400 items: 10 improved, 12 worsened, and 2 changed between equally wrong
options. Thus the full arm lost two net correct items (90.25% to 89.75%) while
using 3.62 times the logical prompt tokens. The rule-conditioned Teacher
answer changed on 33/400 items.

The important mechanism is not merely disagreement: rule memory made all five
candidates agree more often, 343/400 rather than 316/400. But unanimously
**wrong** candidate sets also rose from 9/400 to 18/400. In the 24 final-answer
changes, three improvements and three harms ended in all-five agreement. More
consensus is therefore not reliable evidence of correctness in this stream.
This comparison is observational: the retrieved-rule prompt changes candidate
generation, so it does not isolate which individual rule caused a change.

Representative audited cases illustrate both directions. At stream index 364,
the no-memory answer missed the attraction of opposite magnet poles; retrieved
rules emphasized that principle and the full arm corrected it. At index 178,
the no-memory answer correctly compared solute concentration, while the full
arm produced an incorrect unanimous choice; one retrieved negative rule was
about planetary volume, not concentration. At index 169, an originally correct
magnetic-force comparison became an incorrect unanimous answer. These are
examples for mechanism inspection, not a handpicked effectiveness estimate.

The direct transfer already retrieves top-30 positive and top-30 negative
text rules, but it lacks a grounded relevance or trust gate. The observed
wrong-consensus amplification supports investigating evidence-aware retrieval
on a **new development stream**. It does not by itself prove that the proposed
gate will improve accuracy; any such change must be frozen and compared with
budget-matched controls before scoring unseen validation data.
