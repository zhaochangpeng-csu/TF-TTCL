# ScienceQA-IMG CCIC development pilot (seed 0, 2026-10-12)

**Outcome: the strict CCIC selector has zero coverage and provides no accuracy
gain. Do not expand this version to validation, other datasets, or memory.**

## Data and frozen configuration

This is a development experiment on 400 image-bearing ScienceQA **train**
examples. It is not a held-out benchmark or a comparison with the previous
dev400's accuracy. The new stream uses ranks 400–799 of
`SHA256("20261011:<official-id>")`; its overlap with the earlier dev400 is zero.
The ordered ID-list hash is
`34e2dd95a1b9263253c8c15b3496b08fd5c236be8bb69ca53abade4ea91d961b`.

Inference used `dev400_causal.inference.jsonl`. The separate labels were first
read after all three 400-row inference manifests were completed. The strict
4x4-grid CCIC prompt and selection conditions were unchanged throughout this
experiment. No memory was retrieved or written, no logprobs were requested,
and no external teacher was used.

All full runs used commit `0b62259a5d56960603e614e85d990aa4720e17b4`, frozen
Qwen2.5-VL-7B-Instruct revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5`, prompt hash
`84cc2a353988f438ec07bdd4a165eb6e06adbacfe680bec2ed4b40460b0939b4`.
The `ttt` environment used Python 3.12.14, vLLM 0.30.0, torch 2.13.0,
transformers 5.17.0, and openai 3.19.2. Runs used only GPU 2, an NVIDIA RTX PRO
5000 72GB Blackwell, in the `ttt_ex` tmux session.

## Smoke and exact pairing

The first 20-row smoke executed both arms independently. It observed a
localization-output mismatch on sample `20361`, despite identical candidate
answers and matching request inputs. Total token budgets differed by only
0.0194%, but same-seed independent serving was not an exact selection control.
The original artifacts were retained.

Pairing was corrected without changing CCIC: the budget arm records immutable
responses keyed by prompt, image hashes, sampling configuration, dataset/source
hash, seed, and model revision. The evidence arm strictly replays the same
trace. A replay miss fails rather than executing an unpaired model call.
The second 20-row smoke passed its label-free audit: 144 identical calls,
51,554 tokens per arm, zero parse failures, and zero overrides.

This is a comparison of two **decision policies on one shared computation**.
Evidence-arm token and model-latency totals describe the shared trace's cost;
they are not a second actual expenditure or an independent replication. The
full budget arm made 2,904 model calls; the evidence arm replayed 2,904 and made
zero new model calls. Standalone greedy made 400 calls.

## Scores and costs

| Method | Accuracy | 95% bootstrap CI | Prompt tokens | Completion tokens | Summed model latency |
|---|---:|---:|---:|---:|---:|
| Standalone greedy | 90.75% (363/400) | [87.75%, 93.50%] | 144,177 | 1,997 | 66.89 s |
| CCIC budget | 90.75% (363/400) | [87.75%, 93.50%] | 949,237 | 24,740 | 401.15 s |
| CCIC evidence (shared trace) | 90.75% (363/400) | [87.75%, 93.50%] | 949,237 | 24,740 | 401.15 s |

All three methods produced identical final answers on every sample. Evidence
minus either baseline is 0.00 percentage points, with paired bootstrap CI
[0.00, 0.00] and paired randomization p=1.0. Corrected answers: 0; regressions: 0.
CCIC consumes approximately 6.66 times greedy's token budget and 6.00 times its
summed model latency without changing an answer. This latency excludes service
startup and is not a wall-clock throughput measurement.

## Where the gate failed

| Stage | Samples |
|---|---:|
| Candidate disagreement | 26/400 |
| Stable question and candidate localization | 3/400 |
| Disjoint candidate regions | 0/400 |
| Focus/drop interventions executed | 0/400 |
| Overrides | 0/400 |

There were 374 unanimous candidate sets. Of the 26 disagreements, 23 fell back
for unstable or malformed localization and three for overlapping candidate
regions. The full label-free audit failed its parsing condition: four
question-only responses and seven candidate-conditional responses failed
parsing. They all failed closed; no label exposure or budget-trace mismatch
was detected. Parse-failure counts above refer to responses, not samples.

The following sample-level diagnostics overlap: candidate cell sets disagreed
after order swapping on 20 samples; question cell sets disagreed on four;
question parsing failed on two samples; candidate parsing failed on four.
This points to coarse spatial localization and order sensitivity as upstream
bottlenecks. No causal crossover was actually tested, so this run does **not**
establish that the downstream intervention signal is good or bad.

## Candidate recall and missed repairs

The six-candidate oracle scores 92.50% (370/400), 1.75 percentage points above
greedy. Seven incorrect greedy answers had a correct alternative, exclusively
in the original sampled trajectory. Thirty errors remained wrong across all
six candidates. The oracle is a post-hoc diagnostic that uses labels; it is
not an inference method or a result to include as an achievable baseline.

All seven missed repairs had well-formed localization responses:

| Sample ID | Blocking condition |
|---|---|
| 4275 | Stable localization, but the two candidates share evidence regions |
| 14435 | Both question localization and candidate localization are unstable |
| 3592 | Candidate localization changes after swapping positions |
| 20150 | Candidate localization changes after swapping positions |
| 20281 | Candidate localization changes after swapping positions |
| 11171 | Candidate localization changes after swapping positions |
| 7515 | Candidate localization changes after swapping positions |

Improving JSON formatting alone cannot recover these seven cases. Sample 4275
also shows that spatial separation is too restrictive as a general repair
requirement: a correct alternative can depend on the same region as an
incorrect answer. Zero overrides demonstrates conservative fallback, not a
successful verifier.

## Decision and next research question

The predeclared improvement condition failed. Stop benchmark expansion and
memory integration for strict CCIC v1. Preserve the method and raw traces as a
negative control. Before another scoring run, redesign verification around
**different interpretations or reasoning predicates of shared visual facts**,
rather than requiring the competing answers to cite disjoint grid cells.
Candidate diversity is also a separate bottleneck: even perfect selection can
recover at most seven cases from this candidate set.

The current dev400 labels are now development-visible. Any subsequent claim
must use a fresh, preselected evaluation stream; do not present re-tuned scores
on this stream as held-out gains. Original artifacts are on the server under
`core/exp_res/mm/scienceqa_train/qwen25vl/ccic_v1_paired_seed0_dev400/`.
Aggregated scores, smoke audits, and candidate diagnostics are versioned in
`docs/results/scienceqa_ccic_seed0.json`.

The vLLM process group ended automatically after all inference arms completed.
GPU 2 was checked afterward at 2 MiB used and 0% utilization. Other GPUs and
their processes were untouched.
