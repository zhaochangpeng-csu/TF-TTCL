# ScienceQA train: placebo history control — 2026-10-14

The 400-item placebo replay completed, passed the prompt/order/source audit,
and exited with status 0. GPU 2 returned to 2 MiB. Run SHA:
`e720e80498d21eb86aa323e66a4c6cc9ea478d5b`.
This is a post-hoc diagnostic on the previously scored train stream.

Only the historical rule block of each recorded noisy-arm prompt was replaced
with fixed text saying the earlier record was withheld. Its word count and
line structure matched the retrieved rule block. Current images, candidates,
system prompt, decoding seed and output cap were preserved. Source no-memory,
noisy and random outputs were reused. There were 399 new calls; the first
item had no history and reused its identical source response. No labels were
opened during inference; scoring followed completion and audit. The 20-item
smoke was unscored and had no format failures.

| Context | Accuracy | Correct / 400 | Corrected / harmed vs no memory |
| --- | ---: | ---: | ---: |
| No memory | 90.50% | 362 | — |
| Noisy lexical retrieval | 91.75% | 367 | 6 / 1 |
| Random historical rules | 91.25% | 365 | 3 / 0 |
| Placebo historical text | 91.50% | 366 | 4 / 0 |

Placebo minus noisy: −0.25 percentage points, paired bootstrap 95% CI
[−1.50, +0.75] pp, paired randomization p=1.0. Placebo minus random:
+0.25 pp, CI [−0.50, +1.25] pp, p=1.0. Placebo versus no memory has
p=0.125. Its percentile bootstrap interval excludes zero ([+0.25, +2.00]
pp), but only four discordant pairs support this difference; the discrete
paired test is not significant. Neither test establishes equivalence between
placebo and noisy, and the experiment cannot exclude a small rule benefit.

Final refinement prompt tokens were 256,073 for placebo and 257,506 for
noisy (−0.56%); completion tokens were 31,892 and 31,599 (+0.93%).
Placebo had no format failures. The close token totals make it harder to
attribute the observed difference to a large cost imbalance, although word
matching does not guarantee equal token counts on every item.

## Actual I/O findings

The ignored local casebook
`core/exp_res/inspection/transfer_placebo_dev400_casebook.md` contains the
first three examples and all nine disagreements across the five contexts
(12 cases). Selection used stream order and answer disagreement, not
correctness. Each case retains candidate outputs, retrieved source rules,
exact final inputs and raw outputs, with targets added after audit.

| ID | Behavior | Implication |
| --- | --- | --- |
| 19717, fossil age | All original arms answered wrong A. Placebo answered correct B. Both outputs describe the palm leaf as deeper and older; the final letter changes while the reasoning remains nearly identical. | A correct option can appear without a corresponding improvement in the written reasoning. |
| 13419, prime meridian | Noisy and placebo both correct C; noisy retrieved two statements about the equator intersecting Africa. | This win does not establish transfer of a relevant procedure. |
| 17087, lunch trade | Noisy, random and placebo all correct the no-memory answer. | The observed gain survives removal of historical answer facts. |
| 20681, highlighted country | No memory, random and placebo are correct; noisy is wrong after retrieving a Grenada conclusion. | Source-specific facts can harm a later map task. |
| 19811, 15250, 8823 | Noisy is correct while placebo is wrong. | Some individual gains remain; the current sample cannot establish their mechanism or repeatability. |

Across the five noisy/placebo answer differences, noisy wins three scored
pairs and placebo wins two. Thus the noisy arm's
aggregate advantage amounts to one correct item. These examples motivate
measuring answer/rationale consistency and historical fact copying in addition
to accuracy.

## Research decision

The present results do not demonstrate a reliable semantic memory benefit
over a cheap historical-context placebo. Keep validation and other datasets
frozen. Before expanding, redesign retrieval around transferable checking
operations and their applicability to the current task, with source facts
excluded from the retrieved text. Verify visual observations separately:
the prior run's identical real/fidelity hashes and erroneous magnet
certificate remain unresolved by this experiment.

Freeze any revised rule and applicability protocol before testing it on a new,
disjoint train slice, with no-memory, random and placebo controls and multiple
stream orders. Continue the memory research line only if the revised method
consistently beats those controls. Current results are development evidence
and support no claim of top-conference readiness.

Aggregate JSON is on the server at
`core/exp_res/mm/scienceqa_train/qwen25vl/transfer_placebo_v1_seed0_dev400/comparison.json`
and locally at `core/exp_res/inspection/transfer_placebo_dev400_comparison.json`.
The source input and label hashes are identical to those in the preceding
four-arm report. Raw predictions, images and trace caches stay outside Git.
