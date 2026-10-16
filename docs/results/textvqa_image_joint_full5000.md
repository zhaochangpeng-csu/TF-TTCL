# TextVQA image-only and joint augmentation: full paired result

Completed 2026-10-16 (Asia/Shanghai) on all 5,000 TextVQA validation
questions, Qwen2.5-VL-7B-Instruct revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5`, seed 0, GPU 2. Run Git
SHA: `b4ea7d70a746c3a11e7a6c10daa4a24129cf192a`. The frozen baseline Git
SHA was `f5156f5ea37ee227a3e20cd6644f1fe8ea166779`. The
[pre-run protocol](textvqa_image_joint_protocol.md) defines the contrasts.
The 20-item smoke and full 5,000-item paired audit passed **without reading
labels**; isolated labels were opened only afterward by the offline scorer.
No memory, teacher model, or gradient was used.

The metric is the project's **local deterministic TextVQA soft score**, whose
parity with the official EvalAI answer processor is not yet verified. Scores
are therefore not leaderboard claims. This is exploratory: the same
validation labels were previously opened for the text-only baseline.

| Arm | Local VQA score | Score sum / 5,000 | Logical prompt + completion tokens |
| --- | ---: | ---: | ---: |
| Original greedy | 72.532% | 3,626.6 | 5,596,765 + 176,148 |
| Same-text, original-image vote | 72.384% | 3,619.2 | 17,751,925 + 657,044 |
| Text-only rewrite vote | 72.462% | 3,623.1 | 17,756,916 + 658,084 |
| Image-only view vote | **72.810%** | **3,640.5** | 17,751,925 + 657,729 |
| Joint text-and-image vote | 72.778% | 3,638.9 | 17,756,916 + 658,353 |

The largest voting-arm token gap is only **0.034%** of the same-text arm's
18,408,969 logical tokens. Each vote arm costs about 3.19 times greedy in
logical tokens. The image-only and joint arms reused the same original greedy
answer and two fixed full-frame image views. They differed only in whether
the two sampled questions were original or rewritten.

| Paired contrast (candidate − baseline) | Delta | Improved / worsened items | 95% paired bootstrap CI | Two-sided randomization p |
| --- | ---: | ---: | ---: | ---: |
| Image-only − same-text vote | **+0.426 pp** | 101 / 72 | [−0.004, +0.842] pp | 0.0458 |
| Joint − text-only vote | +0.316 pp | 96 / 78 | [−0.104, +0.732] pp | 0.1384 |
| Joint − image-only vote | −0.032 pp | 43 / 50 | [−0.326, +0.270] pp | 0.8477 |
| Image-only − greedy | +0.278 pp | 101 / 81 | [−0.170, +0.712] pp | 0.2196 |
| Joint − greedy | +0.246 pp | 98 / 85 | [−0.202, +0.696] pp | 0.2792 |

The image-only contrast has a *nominal* randomization p just below 0.05,
but its bootstrap interval still includes zero. Applying Holm correction to
the three predeclared primary contrasts gives an adjusted p of about 0.137
for this smallest p. It also does not beat greedy significantly. Thus this is
a **weak exploratory image-only signal, not established improvement**; joint
augmentation has no demonstrated incremental value over image-only.

The two image views had different file hashes from the source on all 5,000
items. Image-only disagreed with the same-text vote on 254 items; 173 of
those changed soft score. There were 50 format-invalid new candidate outputs
among 20,000 new candidate positions. The immutable response trace recorded
13,148 executed requests and 6,852 exact replays because some accepted
rewrite slots fell back to the original question. Logical arm costs count
shared/replayed requests as if each arm ran separately; they are not the
physical incremental bill of this paired run. The two generated PNG views
occupy about 9.7 GB on the server. The owned vLLM process exited, leaving no
GPU 2 compute process from this experiment.

## Post-score input/output diagnosis

These cases were selected **after scoring** and cannot define a new
confirmatory subgroup. ID `39103` asks for a book's name. Same-text voting
selected the publisher despite an `Observed` line containing *The Pencil of
Nature*; image-only voting selected the title and moved the local score from
0 to 1. ID `36880` shows `LAPTOP 1`; the image-only answer `1` scored above
the longer `LAPTOP 1`. That is partly answer granularity, not proof of better
visual recognition. Conversely, ID `36270` asks what *bienvenue* means: an
image-view candidate reasoned that it means “welcome” but finalized
`BIENVENUE`, changing a correct vote to wrong. ID `34977` changed the
expected colour `Black` to `Dark`. The mechanism must account for both
visual reading and question-to-final-answer consistency.

Crucially, **both** view operations also re-encode the source image as PNG.
This run cannot isolate photometric sharpening/contrast from pixel-preserving
re-encoding, altered upload bytes, or sampling variability. Mild contrast
changes can also affect colour-sensitive questions. The next controlled
experiment should add a same-dimension, pixel-preserving PNG re-encode arm
with identical seeds and vote logic, then compare it with sharpening and
contrast on an untouched development/held-out stream. Do not tune on these
validation labels or claim a general multimodal test-time method from this
borderline single-model result.

Input SHA-256:
`10b8f9464bc9190a929650e83594b5f0f6a237598e9d9eff47f97f52eb25d978`.
Isolated label SHA-256:
`3ba53f593010342d727642fb587adfe214e76a39613328e5f4b17fccc3d6d311`.
Aggregate JSON is at
`core/exp_res/mm/textvqa/qwen25vl/image_joint_v1_seed0_full5000/comparison.json`
on the server and in the ignored local copy
`core/exp_res/inspection/textvqa_image_joint_full5000_comparison.json`.
