# ScienceQA train: first isolated text-augmentation baseline

Protocol frozen before reading this slice's labels. Model:
Qwen2.5-VL-7B-Instruct, revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5`, seed 0, GPU 2.
Data: 400 ScienceQA-IMG **train** examples, SHA256(`20261011:<id>`)
ranks 2100–2499; the manifest check rejects overlap with every earlier `dev*`
slice. Inference inputs and labels are physically separate. The first 20 are
for unscored format/leakage smoke only; the full 400 use the same frozen code.
This is development, not an untouched validation or a direct TF-TTCL result.

The narrow question is: **Does changing only the question wording help beyond
spending the same candidate-generation budget on the original wording?**
No memory, evidence verifier, image transformation, teacher, gradient or
log-probability score is used. All visual candidates see the same original
image bytes, identical options, same answer format, and the same temperature
and seed at each paired sampling position. Only the query clause may change;
ScienceQA passage/context is restored verbatim. An answer-blind model call
proposes two rewrites. A conservative lexical guard rejects changes to
negation, numbers, names, quotes and original content words; rejected or
malformed rewrites fall back to the original query and are counted.

| Arm | Candidate responses | Decision |
| --- | --- | --- |
| `greedy` | Original question, temperature 0 | Its answer. |
| `same_text_vote` | Shared greedy plus two original-question calls at temperature 0.7 | Valid majority; tie prefers first valid response. |
| `text_aug_vote` | Shared greedy plus two rewritten-question calls at temperature 0.7 | Same voting rule. |

The rewrite call is shared in the paired run and included in the **logical**
token budget of both voting arms, even though the same-text arm discards its
output. This isolates the content of the rewrites from the extra call. Prompt
and completion tokens are still reported separately; equal call count does
not guarantee exactly equal tokens. Invalidly formatted candidates are logged
and excluded from voting; the greedy answer is the last-resort fallback.

Run order: unit tests and input-overlap check, 20-item unscored smoke, audit,
then 400-item paired run, audit, and **only then** offline scoring. The audit
checks all arm outputs, sample order, image-hash identity and query variants;
the runner has no label path. The scoring report will show overall accuracy,
paired corrected/harmed counts and paired uncertainty for text augmentation
versus both greedy and same-text vote, plus rewrite coverage and token cost.

Interpretation is deliberately modest. An overall or predeclared narrow-task
gain can motivate the next text-specific experiment, but a post-hoc subgroup
selected by correctness is only a hypothesis. A single 400-item, one-seed
gain is not a main-conference claim. If text augmentation fails, inspect
rewrite coverage and answer-changing cases before adding image variants or
memory; if it succeeds over the same-text vote, replicate on another fixed
stream/model before expanding the method.

Code: `core/multimodal/text_only_runner.py`, `scripts/run_text_only.py`,
`scripts/run_scienceqa_text_only.sh`, and
`evaluate/multimodal/compare_text_only.py`. Raw predictions and response
traces remain outside Git.
