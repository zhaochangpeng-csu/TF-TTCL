# Train-only reasoning-format smoke — 2026-10-13

**V2 format check passed: 40/40 calls, 20/20 items, all finish reasons `stop`.**
No labels were read and no memory was used. This establishes inspectable output
format only, not accurate observations, answers, or useful online memory.

Input: first 20 IDs of a new disjoint 400-item ScienceQA-IMG train slice,
SHA256(`20261011:<id>`) ranks 1700–2099. No intersection with previous
development subset manifest IDs. Rank slice ordered-ID SHA:
`fe0286939cfa2b5c28b8bc12003b813668fd9cd857c4416dee07cc67c28511d8`.
Inference-file SHA:
`1459dbe9089bdd31fcc9a2078289a07de432669b75c0a9f02fa97fc8568c289c`.
Each item uses the original question/image, greedy and temperature 0.7,
max 384 output tokens, flow seed 0. Model: Qwen2.5-VL-7B-Instruct,
revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`.

| Format development | Passing calls | Prompt tokens | Completion tokens | Summed model latency |
| --- | ---: | ---: | ---: | ---: |
| V1, system instruction only | 1/40 | 18,998 | 1,489 | 22.49 s |
| V2, section requirements repeated in user message | 40/40 | 24,358 | 3,228 | 44.89 s |

V2 was changed solely from unlabeled format failures; the same 20 development
inputs were reused, not treated as a held-out experiment. V1 remains immutable.
V2 prompt SHA:
`335c360b7214d4a77bca136d7598e26ddfbcb23a0f5df2ea94539cac0663ab08`.
V2 script SHA:
`3e1741d4a7e5663851f38c2271d28e47fcb563397ddb609bd6803b6b3cda0a69`.
Runs record server HEAD `aed4fb7` plus a dirty/untracked additive-file status,
because these new diagnostic scripts were deployed before the local commit
was synchronized to server Git. Do not describe the smoke as a clean-HEAD
formal benchmark; exact script/prompt/input/model hashes and requests are
preserved. Commit `d117808` contains V1 tooling; subsequent commit contains V2.
Formal online experiments must start from a clean synchronized code revision.

## What inspecting an explanation reveals

Item 7813 has a reference photograph and a context describing a Steller's sea
eagle as a bird. The actual task is **“Select the bird below”**, with choices
cane toad, tokay gecko, Nile crocodile, cassowary. It is not asking to identify
the photographed species. The candidate answer need not be an entity visible
in the photograph, so demanding image evidence for every answer entity would
be a category error for this task.

Both V2 trajectories select D and provide observation/reasoning sections.
However, their explanation classifies all three other choices as reptiles,
including the cane toad. This is an independently inspectable biological
misclassification; no dataset answer key was opened to find it. A plausible
final choice does not make every intermediate claim reliable, and a flawed
intermediate claim does not make every general rule distilled from it harmful.
Evaluate the actual transferred procedure and downstream harm, not merely
the presence of section labels or agreement on D.

Similarly, an `Observed` section can blend visibly present features with
purpose/causal inferences supplied by text or knowledge. These are proposals,
not certified evidence. Future semantic gates must validate the distinction;
the format checker deliberately does not pretend to do so.

Artifacts: server
`core/exp_res/mm/scienceqa_train/qwen25vl/transfer_format_v{1,2}_seed0_smoke20`
with raw predictions and immutable requests. Local V2 copies are in
`core/exp_res/inspection/transfer_format_v2_{manifest.json,predictions.jsonl}`.
These large/raw files and dataset images are excluded from Git. GPU 2 was
verified released after both runs: 2 MiB, 0% utilization. The regression's
failure and unscored labels remain unchanged. All 75 multimodal tests passed.

Next checkpoint: implement the separately frozen, rule-only four-arm transfer
runner described in [the revised pilot plan](../noisy_experience_transfer_pilot.md).
The four-arm memory comparison has **not** run yet; do not scale datasets/models
on the basis of this format success.
