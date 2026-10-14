# ScienceQA-IMG validation: frozen text-only baseline

This protocol was recorded **before opening validation labels** for the new
run. The 2,097 validation items are a full-set follow-up to the scored,
disjoint 400-item ScienceQA-IMG train development experiment. Earlier
validation items and some inputs/outputs were inspected without labels, so
this is **not** an untouched blind benchmark or independent replication.
The prior typed-v3.1 validation experiment failed its frozen schema audit and
is not reused or scored here.

## Frozen comparison

- Model: Qwen2.5-VL-7B-Instruct, revision
  `cc594898137f460bfe9f0759e9844b3ce807cfb5`; GPU 2; seed 0.
- Input: `validation_typed.inference.jsonl`, ordered by the existing
  `validation_typed.manifest.json` (seed 20261013, offset 0, all 2,097
  ScienceQA-IMG validation items). Inference files have no answer fields;
  labels remain in a separate JSONL until all three arms complete and pass
  the label-free audit.
- Method and prompts: exactly the committed text-only v2 train protocol:
  shared original-question greedy; two sampled original-question responses;
  two sampled rewritten-question responses; shared rewrite request. The
  options and original image bytes are identical across arms. The rewrite
  guard and vote fallback are unchanged from the development run.
- Arms: `greedy`, `same_text_vote`, `text_aug_vote`. The rewrite request is
  charged to both voting arms' logical token budgets. Report actual prompt
  and completion tokens, not just call counts.
- Sequence: unit tests and full image/manifest precheck, unscored 20-item
  smoke and audit, then all 2,097 items in a fresh run directory and audit.
  Only then open isolated validation labels once for offline scoring. No
  prompt or method changes from smoke or scoring.

The primary contrast is full-set accuracy of `text_aug_vote` against
`same_text_vote`, with paired corrected/harmed counts, 95% paired bootstrap
CI, two-sided paired randomization p-value and cost. Compare both voting
arms with greedy as context, while explicitly acknowledging that greedy has
a far smaller token budget. A single seed is not a robustness claim.

## Secondary, predeclared slices

The following selectors use **question text only**, never answer or score:

1. `highlighted_map`: query exactly matches “Which country/state/continent
   is highlighted?” (case-insensitive). Motivated by the two opposed
   development cases; this is hypothesis-generating, not independent.
2. `magnet`: the full question contains “magnet”. Motivated by prior
   label-free I/O inspection. This is a broad keyword slice, not a
   verified magnetic-pole reasoning category.
3. `capital`: query begins “What is the capital of”. Included as a
   predefined knowledge-oriented control slice.

Report slice sizes, rewrite coverage, all three accuracies, and paired
text-versus-same counts/CIs/p-values. Holm-adjust the three slice p-values.
Small or empty slices are reported as such. Do not search more categories
after scoring and present the best one as a confirmatory gain. An isolated
slice gain would still need a new stream or model replication and a causal
mechanism before a main-conference claim.

## Stop rules and provenance

Any incomplete arm, sample/order/hash mismatch, target leakage, or failed
audit prevents scoring. Invalidly formatted model responses are counted and
handled by the already frozen vote fallback; their frequency is reported.
The evaluator refuses overwrite of an existing score. Save Git SHA, model
revision, input/label hashes, prompt hashes, token counts and run logs.
The runner owns its GPU-2 vLLM process and terminates it on normal exit or
signals. No model or image caches enter Git.

No matter which arm wins, a full-set result on one model/dataset is a
baseline measurement, not evidence that EviMem-TTCL or TF-TTCL improves
multimodal reasoning. If text-only has no meaningful gain, retain it as a
negative control and test a *separately frozen* visual/task-specific idea on
development data before another held-out evaluation.
