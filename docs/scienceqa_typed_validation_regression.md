# Frozen ScienceQA validation regression (2026-10-13)

Status: preparation passed 63 tests; automatic smoke/full regression is ready
to launch, with no new validation scores. Method is frozen v3.1;
memory remains disabled. Prompts, schemas, operators, gates, sampling settings,
model revision and ablations are unchanged from the
[400-item development result](results/scienceqa_typed_evidence_seed0.md).

Use all 2,097 ScienceQA-IMG validation items. Preparation separates source
records into `validation_typed.inference.jsonl` and `.labels.jsonl`. All IDs are
permuted once by sorting SHA256(`20261013:<official-id>`); flow seed is 0.
The manifest records source/ordered-ID hashes. Input checking recursively
rejects labels and verifies image paths. The label file is not opened by
smoke or inference. Source-label copying/hashing occurs only in preparation.

Source SHA256:
`1fdc9257bd04416be52e0b7a1702011317c9afb0712d2e220f384dcf57afd74b`.
Ordered-ID SHA256:
`65ae7333261af6b45ad957efa1b1bc599c45acdef2d088a33834eda0b7e7e684`.

Run a first-20-item **unscored** smoke with all four arms. After protocol and
four-arm completion checks pass, run all 2,097 items:

1. Standalone greedy.
2. Shared-budget six-candidate majority (selection-policy ablation).
3. Shared-evidence using strict replay of arm 2.
4. Independent six-to-twelve-sample majority with cost-only token targets.

Only after all four full streams/configurations match does the evaluator open
the label file and compute accuracy, corrected/regressed counts, coverage,
paired intervals/tests and the ±5% token-cost check. No correctness-based
changes or smoke scoring. A format audit failure stops the pipeline, rather
than silently changing the frozen algorithm. Each model-service launcher
releases its own GPU process group on exit, before offline scoring. Use GPU 2;
GPU 0/1/3 are not needed. No embeddings/model/data downloads are required.

Artifacts:
`core/exp_res/mm/scienceqa_validation/qwen25vl/typed_v3p1_seed0_regression`
and a separate `_smoke20` directory. Completed scores will be in
`comparison.json`; interrupted inference can resume with namespace/config
checks and isolated immutable traces.

```sh
# In the server repo, under tmux ttt_ex:
GPU_INDEX=2 bash scripts/run_scienceqa_typed_validation.sh
```

This validation split was previously used for development diagnostics. These
results are a regression check, **not an untouched holdout** or evidence of
top-conference readiness. Keep final confirmation on genuinely unused inputs.

## What candidate reliability means for the next phase

The original paper uses the same frozen model for teacher/tutor/student roles.
Closed-ended grouping uses answer consensus; open-ended grouping uses teacher
similarity; within-group min-PPL selects trajectories. Section 4.2 explicitly
recognizes noisy candidates and hallucination reinforcement. The paper offers
heuristics and empirical ablations, not a correctness or noisy-memory
improvement guarantee. See the
[primary paper, sections 4.1–4.2](https://aclanthology.org/2026.findings-acl.1482.pdf#page=5).

Our interpretation: pseudo-positives need not all be factually correct, and
an imperfect answer can still yield a useful general reasoning constraint.
Conversely, a correct answer can yield an irrelevant or harmful rule. Answer
selection accuracy is therefore **not sufficient, or a theoretical necessary
condition**, for useful memory. The previous selection-first gate is a staged
engineering decision, not a proof that memory fails whenever selection fails.

Do not silently weaken the current gate or start memory in this regression.
For a subsequent separately frozen memory experiment, propose comparing
same-candidate no-memory, original-style noisy pseudo-label memory, gated/
weighted memory and random/poisoned-memory controls. Measure downstream
learning, coverage, evidence-grounding failures and contamination propagation,
with labels used only offline. Memory quality should ultimately mean net
transfer benefit under noise and matched cost, not unachievable zero noise.
These comparisons are a future proposal, not already-validated results or
authorization to tune from this validation stream.
