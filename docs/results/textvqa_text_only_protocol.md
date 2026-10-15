# TextVQA validation: exploratory cross-dataset text-only baseline

This run is an **exploratory follow-up**, not a fresh confirmatory chance to
find a positive result: the unchanged question-only rewrite has already
failed its matched-budget contrast on ScienceQA-IMG validation and MathVista
testmini. A positive TextVQA result would need an independent model/stream
replication and should not be presented without the two negative datasets.

Use official TextVQA v0.5.1 validation: 5,000 questions, 3,166 images, ten
human answers each. The [dataset loader](https://huggingface.co/datasets/facebook/textvqa/blob/main/textvqa.py)
locates validation images in `train_val_images.zip` under `train_images/`.
`prepare_textvqa_official.py` imports from a complete local ZIP. When the
server cannot reach the official host, `fetch_textvqa_val_images.py` instead
uses HTTP range requests from a connected machine to fetch only the 3,166
validation images (about 913 MB compressed), with per-image CRC checks and
resumption. Both exporters check sample/image counts and create the same
JSONL schema. Answers are then isolated from inference JSONL.

Model: Qwen2.5-VL-7B-Instruct, revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5`, seed 0, GPU 2, no memory,
no external teacher, unchanged images. All arms use the same TextVQA response
format, answer extraction, and the frozen ScienceQA v2 guarded question rewrite.

| Arm | Calls and decision |
| --- | --- |
| `greedy` | Original image/question, temperature 0; extracted answer. |
| `same_text_vote` | Shared greedy plus two original-question samples at 0.7; valid majority. |
| `text_aug_vote` | Shared greedy plus two rewritten-question samples at 0.7; same vote. |

The voting arms are charged for the same answer-blind rewrite call and use
paired sampling seeds. Vote equality uses the project's TextVQA-normalized
answer; scoring recovers the selected candidate's raw final answer. Invalid
formats are excluded from voting; greedy is the fallback. The primary contrast
is text augmentation minus same-text voting on the complete validation set.
Report local VQA soft score, paired improved/worsened items, paired bootstrap
CI, randomization p, rewrite acceptance, format validity, and tokens. The
current metric is a **local deterministic approximation**, not yet verified
identical to the official EvalAI answer processor; no leaderboard parity claim.

Execution order: obtain official annotation/image ZIP; import and verify full
set; create label-isolated inputs; run 20 unscored smoke and audit; run all
5,000 paired items and audit; only then open isolated labels and score. Smoke
labels stay closed and do not drive prompt changes. The process owns a vLLM
service on GPU 2 and terminates it on exit, including failure.

Pre-run status when this protocol was frozen (2026-10-15): the official
validation annotation was downloaded and hash-checked on the local machine;
selective image acquisition was in progress. The server itself could not
reach the official host. No GPU run or score had started. Final outcomes are
recorded separately in [textvqa_text_only_full5000.md](textvqa_text_only_full5000.md).
