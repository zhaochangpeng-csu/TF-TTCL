# ScienceQA-IMG shared-evidence development probe (2026-10-12)

Outcome: **one genuine repair, but not a reliable general improvement**.
The evidence method scores 89.50% versus 89.25% for both greedy and the
six-candidate majority control. The +0.25 percentage-point gain is not
significant (paired randomization p=1.0), covers one override, and the full
protocol audit fails on three malformed counterfactual transcriptions. Do not
expand this version to validation, other architectures/datasets, or memory.

## Prospective development protocol

Use 400 ScienceQA-IMG **train** examples at ranks 800–1199 of
SHA256(`20261011:<official-id>`). Overlap with each previous dev400 is zero.
The ID-list hash is
`708d083f31a72113c53ece3687962a6bcdbd164d9aa78c18b69309f53cbfe06d`.
Inference and labels are physically separated in `dev400_shared.inference.jsonl`
and `.labels.jsonl`. This is a development result, not a benchmark result;
do not compare absolute accuracy against earlier, different train slices.

The operator router supports horizontal two-bar-magnet diagrams and one
initial/final-vial-temperature template. The new stream has 12 magnet tasks
(3.0%) and **zero** supported temperature tasks. Temperature logic has unit
coverage only, not an empirical result here. Unsupported tasks return greedy.
The method uses domain-scripted physics knowledge, not a general learned
verifier or transferable memory. No external teacher, logprobs, gradients,
retrieval, or memory writes are used.

All full inference used commit
`84a438b513b4d40bfed148829dd551df9e3caa96`, Qwen2.5-VL-7B-Instruct revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5`, seed 0, and GPU 2 in `ttt_ex`.
Shared-method prompt/operator hash:
`3ba2b71f9bfd48592f44e1f35e3031b18d9479938e269cfb9e4d23ca266e22ee`.
Standalone greedy keeps its existing prompt and seed schedule. All 400
standalone greedy answers match the original-greedy trajectory in the shared
trace, so the observed gain is not an anchor-call discrepancy.

Only after all three full 400-row manifests were completed were their labels
first opened by the offline evaluator. The scorer additionally rejects a
completed smoke stream when `--expected-n 400` is requested. V2 changed only
after **label-free** v1 transcription failures, before any new-stream scoring.
The inference prompts/operators were unchanged throughout the full run.

## Smoke findings and redesign

The fixed first 20 examples contain no supported tasks. Both v1 and v2
20-item smoke audits pass, but cannot establish core-path functionality.
A separate smoke includes all 12 eligible examples, filtered solely by the
frozen router from the inference file, with no correctness-based selection.

V1 asks for facing-pole fields with populated N/S schema examples. It returns
N/S evidence on **all 12 white blank images**, despite valid JSON and stable
real-image results. Its gate rejects all samples. Schema correctness and
repeatability clearly do not establish visual grounding. The cause is not
proven: prompted answer-shaped fields/examples are a plausible contributor,
not a causally isolated explanation.

V2 instead asks for standalone uppercase-letter transcriptions and positions,
without specifying expected N/S identities, magnetism, question, options, or
candidates. Code orders four visible N/S labels along the horizontal axis and
computes the relation from the two middle labels. No model verdict is executed.
Real/fidelity facts must agree; blank must have no usable facts; the immediately
preceding image must not reproduce the original facts. The calculated answer
must already exist among the six candidate trajectories. Failures return greedy.

V2's 12-item core smoke has six stable observations, five accepted certificates,
one override, and zero parsing/exposure/trace errors. Full-stream historical
mismatches differ from this filtered smoke: mixed-task images expose additional
format failures. All smoke artifacts, including v1 failure, are retained.

## Scores, pairing and costs

| Method | Accuracy | 95% bootstrap CI | Total tokens | Summed model latency |
|---|---:|---:|---:|---:|
| Standalone greedy | 89.25% (357/400) | [86.00%, 92.25%] | 146,175 | 64.66 s |
| Shared-budget majority | 89.25% (357/400) | [86.00%, 92.25%] | 941,335 | 377.58 s |
| Shared-evidence | 89.50% (358/400) | [86.50%, 92.50%] | 941,335 | 377.58 s |

Evidence minus either baseline: +0.25 pp, paired bootstrap 95% CI [0.00, 0.75]
pp, corrected 1, regressed 0, two-sided paired randomization p=1.0.
Bootstrap/permutation budget is 10,000, seed 0. Majority and greedy have
identical answers on all items. One favorable pair cannot establish a general
gain; its CI includes zero. Accepted-certificate accuracy is 4/4, but this tiny
selected sample must not be advertised as a reliable 100%-precision verifier.

Budget executes 2,848 calls: 400 paraphrases, 2,400 candidate answers and 48
observations. Evidence strictly replays those 2,848 responses, executing **zero**
new model calls. Its reported cost is the same shared computation's cost, not
a second expenditure or independent replication. Total tokens match exactly.
The method spends 6.44x greedy tokens and 5.84x summed model latency. Latency
excludes service startup, client overhead, and is not wall-clock throughput.

This majority control deliberately ignores observation calls it pays for. It
is a selection-policy ablation, **not an efficiently allocated strong budget
baseline**. The small gain cannot support a budget-efficiency claim.

## Coverage, failures and one auditable repair

| Stage | Count |
|---|---:|
| Candidate disagreements | 23/400 |
| Supported disagreements | 3/400 |
| Stable real/fidelity facts before counterfactual validation | 6/400 |
| Stable certificates after complete parsing | 5/400 |
| Accepted certificates | 4/400 (1.0%) |
| Overrides | 1/400 (0.25%) |

The audit's stable-facts counter is 5 because v2 fails immediately on any
malformed observation. A separate raw-observation diagnostic finds six stable
real/fidelity pairs; sample 18894 loses its otherwise stable pair due to a
malformed historical mismatch. No unsupported task was overridden.

All 12 full-run blank responses are literally `{"items":[]}`. Three historical
mismatch responses fail strict parsing: 18575 emits a lowercase letter, while
18894 and 15722 emit incomplete JSON after lengthy letter lists. All fail closed.
There are no question/candidate exposure flags, memory writes, policy errors,
or budget/evidence trace mismatches, but zero-malformation protocol acceptance
is not met. Therefore `continue_to_validation=false`.

The repaired item is **14200**. The same image facts are stable on original
and lossless re-encoding: ordered labels `S, N, N, S`; facing poles `N, N`.
The executable equality rule gives repulsion, option A. Five candidate
trajectories—including greedy—say B; only the original sampled trajectory says
A. The blank returns no letters; historical mismatch yields unrelated W/M/K/A
letters, not the required facts. Code selects A, and subsequent offline scoring
confirms it is correct. This demonstrates one instance where shared facts
distinguish reasoning conclusions even though majority voting fails.

Original and fidelity bytes have identical hashes for this case. Their
agreement is a repeated-call/serialization check, not independent visual
corroboration or proof that the transcription is true. The physics rule is
hand-coded and domain-limited; neither this example nor the pilot establishes
novelty, universal grounding, or top-conference readiness.

## Remaining bottlenecks and next step

The post-hoc six-candidate oracle is 92.00% (368/400): 11 greedy errors have a
correct alternative, exclusively at the original-sampled index. The selector
repairs one and misses ten; 32 errors have no correct candidate at all. Oracle
uses labels and is not an inference baseline.

Two oracle repair opportunities are supported magnet tasks. Besides 14200,
item 9952 has stable **vertical** N/S/S/N transcriptions, but the frozen
horizontal-layout check rejects them. Do not retroactively alter that operator
and claim improvement on this now label-visible stream.

Stop benchmark scaling for v2. The next bounded development task should make
the structured evidence protocol robust on mixed-task counterfactual images,
then test direction-independent relations and reusable comparison/count/set
operators rather than another collection of question-specific answer rules.
Separate transcription errors from execution/selection errors and include an
efficiently allocated budget baseline. Freeze any revised method and select
a new disjoint development flow **before** looking at its accuracy. Only after
reliable gains should validation, cross-architecture evaluation and memory
be resumed. This is a follow-up plan, not an already implemented general solver.

There are 46 passing regression/unit tests. GPU 2 was observed at 60,144 MiB
resident service reservation during inference (not a measured model peak).
Its vLLM process group exited automatically afterward; GPU 2 returned to
2 MiB and 0% utilization. No processes on GPUs 0/1/3 were terminated.

Raw artifacts remain on the server under
`core/exp_res/mm/scienceqa_train/qwen25vl/shared_v2_seed0_dev400/`.
Aggregate scores and smoke/observation diagnostics are versioned alongside
this report in `scienceqa_shared_evidence_seed0.json`.
