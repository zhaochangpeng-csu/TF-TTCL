# TextVQA validation: full paired question-only baseline

Completed 2026-10-15 on all 5,000 official TextVQA v0.5.1 validation
questions (3,166 images), Qwen2.5-VL-7B-Instruct revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5`, seed 0, GPU 2. Run Git
SHA: `f5156f5ea37ee227a3e20cd6644f1fe8ea166779`. The
[pre-scoring protocol](textvqa_text_only_protocol.md) declared this an
**exploratory** third-dataset follow-up after negative ScienceQA and MathVista
results. A 20-item unscored smoke and the complete 5,000-item label-free
paired audit passed. Only then were isolated labels opened by the offline
scorer. There is no memory, external teacher, or image augmentation.

Scores below use the project's **local deterministic VQA soft score** over
ten human answers, not an independently verified reproduction of the official
EvalAI answer processor. They are not leaderboard numbers.

| Arm | Local VQA score | Soft-score sum / 5,000 | Logical prompt + completion tokens |
| --- | ---: | ---: | ---: |
| Original-question greedy | 72.532% | 3,626.6 | 5,596,765 + 176,148 |
| Same-text, same-budget vote | 72.384% | 3,619.2 | 17,751,925 + 657,044 |
| Text-augmented, same-budget vote | 72.462% | 3,623.1 | 17,756,916 + 658,084 |

The predeclared primary contrast, **text augmentation minus same-text
voting**, is **+0.078 percentage points** (42 improved, 38 worsened,
+3.9 soft-score points over 5,000). Its 95% paired bootstrap CI is
[−0.200, +0.356] percentage points and its two-sided paired randomization
p=0.5964. Relative to greedy, text augmentation is **−0.070 percentage
points** (83 improved, 88 worsened; CI [−0.482, +0.338] pp; p=0.7408).
Neither contrast supports a reliable improvement. The two voting arms differ
by only 6,031 of roughly 18.41 million logical tokens (0.033%); each uses
about 3.19 times the greedy token budget. Logical cost is per-arm accounting
for shared requests, not the physical bill of three independent runs.

At least one guarded rewrite changed on 2,155/5,000 items (43.1%). The two
vote arms disagreed on 103/5,000 (2.06%), **all** with an accepted rewrite;
80 disagreements changed the local score. There were 62 format-invalid
candidate outputs among 25,000 candidate positions (0.248%) across 18
items; the frozen valid-vote/fallback rule handled them. The 20-item smoke
had zero vote disagreements and therefore did not predict the full-run effect.

## Actual input/output inspection after scoring

These are diagnostic cases, selected **after** opening labels, not new
confirmatory subgroups. In ID `39103`, the image clearly shows the central
book title *The Pencil of Nature* and a publisher line at the bottom. Two
same-question samples described the title in `Observed` but emitted the
publisher as `Final answer`, making the same-text vote wrong; one rewritten
sample restored the title and changed the vote to correct. This is an
observation/final-answer inconsistency that a frozen model might check
without ground truth. In ID `34735`, the first website listed on the page is
`london-underground.blogspot.com`; same-text voting recovered it, while the
rewritten vote selected an unsupported `londongeeker.blogspot.com` from the
greedy/rewrite responses. The two cases point in opposite directions.

The accepted rewrite for ID `39469` added “simultaneously” to “does the
machine measure attack, ratio, and meter?”, changing a condition the
original question did not state. Thus the inherited lexical guard does not
guarantee semantic equivalence. These examples support investigating
visually grounded answer selection and stronger rewrite validity checks;
they do **not** establish that either new method would improve accuracy.

## Interruption and provenance

The first full process stopped after 2,230 complete rows at 10:44; its vLLM
log records a shutdown signal. The server later rebooted at 12:07, so the
reboot cannot be asserted as the immediate cause. A first restart preserved
the 2,230-row prefix but exceeded the launcher's four-minute cold-start
timeout before vLLM became ready; it generated no new rows. The successful
restart used the same model, GPU, prompts, seeds, image bytes, context length
and code/Git SHA, but allowed 20 minutes for model readiness. The runner
verified the complete ordered prefix and continued through 5,000 without
overwriting it. The final label-free audit passed. The owned vLLM service
exited; a later, unrelated `python main.py --config-name level1_pick`
process occupied about 3.1 GiB on GPU 2 and was not touched. The tracked
launcher now also allows 20 minutes for future cold starts; that post-run
maintenance change is **not** part of the recorded run SHA.

Source annotation SHA-256:
`4ceb5aadc1a41719d0a3e4dfdf06838bcfee1db569a9a65ee67d31c99893081d`.
Export JSONL SHA-256:
`58a0d3cc1803cc917cc2bce35033ff5bcec4dad3ae8d680f024280a32a7dd0a4`.
Inference SHA-256:
`10b8f9464bc9190a929650e83594b5f0f6a237598e9d9eff47f97f52eb25d978`.
Isolated label SHA-256:
`3ba53f593010342d727642fb587adfe214e76a39613328e5f4b17fccc3d6d311`.
Prompt SHA-256:
`8a67498626c25b32beb7c6db2b4bb54e1bcc8532c7b00b359704a51b2beea4b5`.
Aggregate JSON is on the server at
`core/exp_res/mm/textvqa/qwen25vl/text_only_v1_seed0_full5000/comparison.json`
and in the ignored local copy
`core/exp_res/inspection/textvqa_text_only_full5000_comparison.json`.

## Decision

The unchanged question-only rewrite has no credible budget-matched gain on
this OCR-heavy dataset, following 0.00 pp on
[ScienceQA-IMG validation](scienceqa_text_only_validation.md) and −0.20 pp
on [MathVista testmini](mathvista_text_only_full1000.md). Stop scanning
additional public datasets for a favorable result. Keep it as a negative
control. The next development method should explicitly target a narrow
visual/OCR failure (for example, region-linked text extraction and a
question–observation–final-answer consistency check with abstention), compare
against greedy and matched-budget repeated prompting on **separate
development data**, and freeze the method before a held-out test. The
official TextVQA scoring processor must be matched before any benchmark or
paper claim.
