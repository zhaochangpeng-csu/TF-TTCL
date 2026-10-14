# MathVista testmini: frozen text-only cross-dataset baseline

Recorded before the new 1,000-item run and before opening its isolated labels.
MathVista testmini is a 1,000-question set with public labels; this is not a
blind test and its local deterministic evaluator is **not claimed identical**
to the official three-stage MathVista answer-extraction/evaluation pipeline.
See the [official MathVista evaluation instructions](https://github.com/lupantech/MathVista#evaluations-on-mathvista).
The cached `evalscope/MathVista` ModelScope parquet is exported in full into a
new directory; its prior 20-image export is not scored or reused as a full set.
Source and export hashes, model revision, prompt hash, Git SHA and run order
are recorded. All inference inputs are physically separated from labels.

Model: Qwen2.5-VL-7B-Instruct, revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5`, seed 0, GPU 2. This is
**no memory**, no external teacher, no image alteration. It ports the
ScienceQA v2 question-only rewrite guard, candidate seeds/temperature, and
paired cost design without tuning the rewrite prompt on MathVista labels.
Because MathVista includes numeric free-form and multiple-choice questions,
all arms share a new three-section MathVista answer prompt and the same
dataset-aware deterministic extraction. The baseline answer prompt differs
from ScienceQA's option-letter prompt; its effect cannot be compared directly
to the ScienceQA accuracy number.

| Arm | Calls and decision |
| --- | --- |
| `greedy` | Original image/question, temperature 0; its extracted answer. |
| `same_text_vote` | Shared greedy and two original-question samples at temperature 0.7; valid majority with greedy tie preference. |
| `text_aug_vote` | Shared greedy and two rewritten-question samples with paired seeds; same vote. |

Both voting arms pay for the same answer-blind rewrite call in their logical
token budget. Invalid final formats are logged and excluded from voting;
greedy is the fallback. For open answers, voting uses the project's frozen
MathVista normalization, including answer type and float precision.

Run order: full export and 1,000-image/label-isolation precheck; 20-item
**unscored** smoke and audit; full 1,000-item paired run and audit; only then
open labels once and score. Do not change prompts from smoke accuracy (smoke
labels remain closed). The primary contrast is text augmentation versus
same-text vote on all 1,000: accuracy, paired corrected/harmed, 95% paired
bootstrap CI, two-sided paired randomization p and actual tokens. Greedy is
a lower-budget context baseline. Predeclared descriptive slices are
`question_type=multi_choice` and `question_type=free_form`; any slice gain
needs multiplicity caution and independent replication. No after-the-fact
search across tasks or answer types for a positive headline.

One model, one dataset, one seed cannot establish a top-conference method.
Failure to beat the cost control would make unchanged question paraphrasing
a negative cross-dataset baseline, not a reason to keep trying datasets until
one happens to be positive.
