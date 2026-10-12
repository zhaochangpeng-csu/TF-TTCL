# ScienceQA-IMG bounded shared-evidence probe (v3.1, 2026-10-12)

Outcome: **a positive, narrow development signal; not a statistically reliable
general improvement**. Evidence repairs four greedy errors with no regressions
and beats independent cost-matched sampling by 0.75 percentage points.
The paired two-sided test against that sampler is not significant (p=0.5078),
and all repairs use the explicitly scripted magnetic-pole operator.

The predeclared directional gate passes (`continue_to_validation=true`).
This licenses a frozen ScienceQA validation re-evaluation, **not** a claim of
top-conference readiness or general multimodal verification. Do not expand the
full model/dataset/memory matrix. Memory remains disabled.

## Inputs and version history

400 ScienceQA-IMG **train** examples, SHA256(`20261011:<official-id>`) ranks
1300–1699, disjoint from prior scored streams and the unscored 100-item
calibration slice (ranks 1200–1299). Ordered-ID SHA256:
`bbb872175d91c146e846d19d4fcbb69a933887aaa7a8c3b50f8fb227e52b291f`.
Separate `dev400_typed.inference.jsonl` and `.labels.jsonl`; inference inputs
were recursively checked for forbidden label fields. Different 400-item
slices do not support comparisons of absolute accuracy across versions.

V3 was frozen at `97f0416` before this stream was prepared. Its full unscored
audit failed: a historical precipitation-chart mismatch for 8817 induced
endless decimal digits and unclosed JSON at 1024 tokens. The gate fell back;
the launcher skipped independent sampling. **No labels were read.** Preserve
[that failed audit](scienceqa_typed_v3_format_failure.json).

V3.1 fixes label-free format/completeness handling: bounded decimal strings,
explicit exclusion of row-label headers, and token-limit finish rejection.
61 tests passed. A complete 100-item unscored calibration flow passed its
audit and four-arm guard; costs matched within 0.0282%. Calibration labels
remain unread. The 400 inputs contributed an unlabeled format diagnostic,
so they are **not completely untouched inputs** for v3.1. No operator,
threshold or prompt was chosen from their correctness.

All four v3.1 runs used frozen commit
`780803a42bbd9d3559d60898d776beb27023c0c9`, Qwen2.5-VL-7B-Instruct configured
revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`, flow seed 0, GPU 2 in
`ttt_ex`, conda `ttt`. Shared prompt/schema/operator hash:
`dc4902f3afbf980bcf5c2efa849d8fa93ebf55c2c14cf7790f4ec22d8a6da076`.
No prompts/policies changed during these runs. The offline evaluator first
loaded these labels only after all four matched 400-row manifests completed.

## Scores, pairing and costs

| Method | Accuracy | 95% bootstrap CI | Total tokens | Summed model latency |
|---|---:|---:|---:|---:|
| Standalone greedy | 86.25% (345/400) | [82.75%, 89.50%] | 141,404 | 66.17 s |
| Shared-budget majority | 86.25% (345/400) | [82.75%, 89.50%] | 928,175 | 420.99 s |
| Shared-evidence | 87.25% (349/400) | [84.00%, 90.50%] | 928,175 | 420.99 s |
| Independent sampling budget | 86.50% (346/400) | [83.00%, 89.75%] | 928,097 | 251.02 s |

| Paired comparison | Delta (pp) | 95% delta CI (pp) | Corrected / regressed | Two-sided p |
|---|---:|---:|---:|---:|
| Evidence minus greedy | +1.00 | [+0.25, +2.00] | 4 / 0 | 0.1250 |
| Evidence minus shared majority | +1.00 | [+0.25, +2.00] | 4 / 0 | 0.1250 |
| Evidence minus independent sampling | +0.75 | [-0.75, +2.25] | 6 / 3 | 0.5078 |
| Independent sampling minus greedy | +0.25 | [-0.75, +1.25] | 3 / 2 | 1.0000 |

Bootstrap uses 10,000 resamples, seed 0. The small-event bootstrap interval
against greedy excludes zero, but the exact paired sign/randomization test
does not reject at 0.05. Do not describe the result as significant or treat
these procedures as interchangeable. Repeated templates also limit generality.

Shared majority and evidence have identical computation: 2,888 executed
reference calls (400 paraphrases + 2,400 candidates + 88 observations), then
2,888 strict replays with **zero** new evidence-arm calls. Majority pays for
observations it ignores; it is only a selection-policy ablation.

Independent sampling executes 2,708 useful original-question answers. It
receives only IDs/integer cost targets, not reference answers/evidence or
labels. Generation counts: 219 items use six, 147 seven, nine eight, one nine,
two ten, and 22 twelve. Whole-stream token difference: **78 tokens / 0.0084%**,
inside ±5%, with no padding or label/confidence-based allocation. Reference
costs are known after reference inference; this is an offline cost-matched
comparator, not a deployable online budget policy or an optimal/SOTA baseline.

Evidence costs 6.56x greedy tokens and 6.36x its summed latency; it is 1.68x
independent sampling latency at essentially equal token cost. Latency excludes
startup/client/file overhead and is not wall-clock throughput. GPU 2 memory,
sampled every ten seconds, peaked at **60,144 MiB** (device-level sampled peak,
not allocator peak). Launcher exit status 0; GPU 2 returned to **2 MiB / 0%**.
Other GPUs/tasks were untouched.

## Coverage and attribution

Strict protocol audit passes: no parsing errors, token-limit responses,
question/options/candidate exposure flags, memory writes/retrieval, policy
violations or paired-trace mismatches.

| Stage | Count |
|---|---:|
| Supported tasks | 22/400 (5.5%): 15 magnets, 6 temperature, 1 table-extreme |
| Object-count tasks | 0 |
| Candidate disagreements | 26/400 |
| Supported disagreements | 7/400 |
| Stable original/fidelity facts | 18/400 |
| Counterfactual drop | 16/400 |
| Accepted certificates | 13/400 (3.25%) |
| Greedy overrides | 4/400 (1.0%) |

All 22 blank controls are literally empty. Eleven observations are incomplete
abstentions rather than format errors. Six tasks fall back for incompleteness,
three because the computed answer is absent from the candidate pool, and 378
because no operator supports them. Mismatches use the immediately preceding
image, never selected by labels, correctness or similarity.

Accepted certificates comprise 12 magnetic tasks and one temperature task
(15649). Temperature/table-extreme tasks produce **no repairs**; counts have
no empirical coverage. Accepted-subset accuracy is 13/13 offline, but this
tiny, domain-selected subset does not establish universal 100%-precision
verification or justify memory writes.

Genuine repairs: **19947, 9795, 4411, 13487**, all `facing_poles`. Five answers
differ from standalone greedy: **8564** has a different inner-greedy answer,
but both answers are wrong; evidence does not override it. The internal anchor
also scores 86.25%, and evidence repairs 4 / regresses 0 relative to it.

All 22 original/fidelity image pairs have **identical byte hashes**. Agreement
is repeated-call stability, not independent visual corroboration or meaningful
image-perturbation robustness. Syntax and blank/mismatch sensitivity cannot
prove OCR truth. Physics/heat-transfer priors are domain-scripted knowledge,
not a learned general verifier or TF-TTCL memory.

[Candidate oracle](scienceqa_typed_candidate_oracle_seed0.json): 355/400
(88.75%) contain a correct six-trajectory candidate; 374/400 are unanimous.
Six recoverable errors remain, while **45 items have no correct candidate**.
Greedy has ten recoverable errors; evidence fixes four. Selection on this pool
has a hard ceiling; changing only the verifier cannot recover missing answers.

## Next decision

Freeze v3.1 and re-evaluate ScienceQA validation with all four controls and
the completion-before-label guard. Do not expand models/datasets, add operator
templates, or enable memory based on this one development win.

ScienceQA validation (2,097 image examples) already contributed development
diagnostics to [the earlier contrastive report](scienceqa_contrastive_gate_seed0.md).
Re-evaluation is therefore a **frozen regression check**, not pristine held-out
confirmation. A main-track claim still needs separately untouched confirmation,
stronger baselines, broader coverage and the original multi-model/multi-dataset/
significance criteria. Do not tune v3.1 from validation or relabel inspected
data as held-out.

Full aggregate: [four-arm results](scienceqa_typed_evidence_seed0.json).
Specification: [typed evidence protocol](../typed_evidence_validation.md).
