# TextVQA image-only and joint augmentation: exploratory paired protocol

This is a **post-text-only, exploratory** comparison on the same TextVQA
validation labels already opened for the preceding text-only analysis. It
cannot by itself establish a confirmatory benchmark gain. Do not change the
views, prompts, or voting after seeing this run's scores. Any positive result
requires replication on a genuinely untouched stream/model.

The run reuses the audited 5,000-row label-free
[text-only baseline](textvqa_text_only_full5000.md): original greedy response,
two accepted/fallback question rewrites, and the three prior arms. The new
arms use the same Qwen2.5-VL-7B-Instruct revision, TextVQA response format,
seed 0, temperature 0.7, max tokens 384, and greedy-anchored valid-majority
rule. The source image is never cropped, flipped, rotated, downsampled or
made grayscale. Two deterministic, full-frame, same-dimension PNG views are
fixed in advance: mild unsharp mask (radius 1, percent 60, threshold 3) and
contrast 1.08. Colour-sensitive items may still be affected; inspect such
failures after scoring rather than assuming the views are universally safe.

| Arm | Two sampled candidates, plus shared original-image greedy |
| --- | --- |
| `same_text_vote` | Original question + original image (frozen baseline) |
| `text_aug_vote` | Two rewritten questions + original image (frozen baseline) |
| `image_only_vote` | Original question + each fixed image view |
| `joint_vote` | Matched rewritten question + matched image view |

All vote arms are charged for the same answer-blind rewrite request, even
when a particular arm does not use its output. The new calls use the same
sample seeds as the old vote arms; the two new arms use the **same image
views**. Primary comparisons: image-only minus same-text vote, joint minus
text-only vote, and joint minus image-only vote; also report each against
greedy. Record logical token cost and reject a strict budget-matched
interpretation if comparable vote arms differ by more than 5% tokens.

Run a 20-item label-free smoke and audit, then the complete 5,000-row paired
run and audit; only after full completion use isolated labels for local VQA
soft scoring. No smoke accuracy is read. The scorer must verify original and
view image hashes, question pairing, candidate format, vote recomputation,
baseline provenance, ordered IDs and absence of label fields. It reports
paired bootstrap intervals, randomization p-values, score-change counts,
invalid outputs, and cost. Official EvalAI processor parity is not yet
verified. No memory, external teacher, or test-time gradient is used.

GPU 2 is required and the launcher refuses to run while any compute PID is
present. It owns and stops only its own vLLM process. Use `conda activate
ttt`; execute long runs in the existing `ttt_ex` tmux session. At protocol
creation the server was occupied by another compute process, so no GPU
experiment had started. The completed run and its post-score analysis are
reported in [textvqa_image_joint_full5000.md](textvqa_image_joint_full5000.md).
