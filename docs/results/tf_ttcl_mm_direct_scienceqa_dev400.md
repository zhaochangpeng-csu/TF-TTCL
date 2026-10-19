# Faithful TF-TTCL multimodal direct transfer: ScienceQA-IMG train dev400

Completed 2026-10-19 on GPU 3 with Qwen2.5-VL-7B-Instruct. The 400-item
ScienceQA-IMG train stream excluded all 2,500 previously used train IDs.
Inference consumed only isolated label-free inputs. All four arms finished
before the offline evaluator opened the separate labels file; the complete
no-leakage/PPL audit passed. The target model was stopped and GPU 3 returned
to 2 MiB used.

| Arm | Accuracy | Correct / 400 | Change vs greedy | Logical prompt tokens | Logical completion tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| Greedy | 89.00% | 356 | — | 148,691 | 20,895 |
| Five candidates, no rewrite vote | 90.25% | 361 | +1.25 pp | 743,455 | 98,679 |
| TF-TTCL candidate selection, no memory | 90.25% | 361 | +1.25 pp | 778,249 | 123,523 |
| Full `tf_ttcl_mm_direct` | 89.75% | 359 | +0.75 pp | 2,820,036 | 88,759 |

The full arm corrected 13 greedy errors and harmed 10 greedy-correct items;
paired randomization p = 0.681, bootstrap delta CI = [-1.50, +3.00] pp.
Against either five-candidate baseline, it corrected 10 and harmed 12;
delta = -0.50 pp, p ≈ 0.83, CI = [-2.75, +1.75] pp. These are not
significant gains. The no-rewrite vote and rewritten no-memory selector tied
in accuracy, with seven corrections and seven harms relative to each other.

The full arm used 3.62× the no-memory arm's logical prompt tokens while
remaining 0.50 pp worse. Across all arms, immutable response tracing executed
6,390 unique model requests (4,140,331 prompt and 262,291 completion tokens);
64 requests were replayed within the run. The full arm wrote 455 textual
rules and retrieved past rules for 399 of 400 items. Its five candidates
agreed completely on 343 items (85.75%); 385 Tutor results yielded four
distinct, context-preserving question variants.

Exploratory subject slices do not reverse the main conclusion: natural
science (n=235) was 86.81% greedy, 88.51% for either five-candidate baseline,
and 88.09% full; social science (n=158) was 92.41%, 93.04%, and 92.41%,
respectively. No slice was predeclared as a confirmatory endpoint.

This is a single-seed development result, not a validation or test claim.
The direct transfer preserves the [original TF-TTCL](https://aclanthology.org/2026.findings-acl.1482.pdf)
candidate partition, generation PPL, positive/negative text-rule memory,
and separate cosine top-30 retrieval, while adapting answer prompts to
ScienceQA and adding the original image to every answering request. It is
distinct from the earlier simplified `noisy` arm. The result does **not**
support a claim that the direct multimodal transfer improves on a strong
candidate-budget baseline. Next, inspect the 10 corrected/12 harmed memory
changes without using this scored stream for prompt tuning; then test the
open-task branch on fresh TextVQA train data rather than the previously
inspected full validation set.

Reproducibility: run Git SHA `60877daf18cf193821f61fd9b9f527b18452b4cd`;
prompt SHA256 `d5d78e4f2847b2d1cf29a7fede3317b7937bbc3f0604c2c0fd33f0b7bb6a0d24`;
input SHA256 `01c6dc83958ea1cee51ab38594b2499e3bfbc4034139674dc799e5dfeb354689`;
selected-ID SHA256 `0e17a5a94fe7f162c9a057cd7e50c3f907abd0425388a2ed6888f07cdc7524b1`;
label SHA256 `009cff0b2f94a472311dfb6f112558bfecbb81477f11ae100c15de4c0c451fb7`;
VLM revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`;
embedding-weight SHA256 `0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd`.

Server artifacts: `core/exp_res/mm/tf_ttcl_mm_direct/qwen25vl/dev400_seed0/`
(`manifest.json`, `predictions.jsonl`, `scored_comparison.json`, and
`unscored_casebook.md`). Raw responses and per-request traces remain outside
Git.
