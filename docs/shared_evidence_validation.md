# Shared visual observations, different executable reasoning (v1)

## Frozen scope and hypothesis

CCIC's spatially disjoint requirement had zero coverage. This probe instead
asks whether **the same extracted facts can falsify one competing reasoning
conclusion**, without a same-VLM verbal correctness vote. It deliberately
supports only two explicit operators, selected from the question and choices:

- Horizontal two-bar-magnet diagrams: equal facing N/S labels imply repulsion;
  unequal facing labels imply attraction.
- Celsius initial/final-temperature tables with the two-vial-flow template:
  all positive deltas imply energy into the vials; all negative deltas imply
  energy out. Zero or mixed signs abstain. This assumes the task's passive
  heat-transfer setting, not arbitrary active heating or internal reactions.

This is **domain-scripted scientific knowledge**, not learned rules or a
general multimodal verifier. Geography, text-only weather/climate, unsupported
quantitative formats, and multiple-image inputs fall back to greedy. Discovery
examples from prior label-visible development streams motivated the scope;
they are not prospective evidence of improvement or a held-out test.

## Extraction and gate

The same frozen VLM transcribes only facing pole letters or numeric table rows.
It sees neither question, context, choices, candidates, nor a requested answer.
Strict schemas reject extra fields, fabricated absence on parse errors,
nonfinite numbers, inferred hidden pole labels, and ambiguous formatting.

For each supported task (including unanimous candidates), extract on the real
image, lossless PNG re-encoding, white blank, and immediately preceding image
(except the first item). Preserve full dimensions; no crop/flip/rotation. The
re-encoding can have the same byte hash: it is a serialization/repeated-call
stability check, **not independent visual corroboration**.

Accept only if real/fidelity observable facts agree exactly, blank has no
facts, mismatch does not reproduce those facts, and the deterministic operator
returns a unique answer already proposed in the six existing trajectories.
Identical facts on an unrelated image fail closed, even by coincidence. Parsing
failures and unsupported/mixed computations fall back. A certificate can
confirm greedy; override coverage is reported separately from gate coverage.
Neither logprobs, labels, external teachers, gradients, nor memory are used.

This signal cannot establish that transcribed facts are true. Correlated OCR
errors remain possible, and counterfactual non-reproduction measures visual
dependence, not correctness. A positive result would motivate broader safe
operators and stronger baselines, not prove generality or conference novelty.

## Prospective experiment and decision

Freeze this specification, prompts, operator code hash and configuration
before generating a new development stream. Use 400 ScienceQA-IMG train items
at ranks **800–1199**, sorting SHA256(`20261011:<official-id>`). This is disjoint
from the first two development streams. It is not ScienceQA validation.
Use separate `dev400_shared.inference.jsonl` and `.labels.jsonl` files.

Run a first-20-item label-free smoke and unit/synthetic tests. Smoke checks
schemas, exposure isolation, traces, deterministic execution and cleanup, not
accuracy. If there are no supported disagreements in smoke, disclose that;
do not choose smoke items using labels or tune by correctness.

Then run all three complete 400-item arms on GPU 2 inside `ttt_ex`:

1. Standalone greedy (existing zero-shot prompt).
2. `shared_budget`: majority of the six candidates, ignoring certificates.
3. `shared_evidence`: greedy fallback with the executable certificate gate.

Budget records immutable calls and evidence replays them exactly. This is a
**decision-policy ablation of the same computation**, not an efficient budget
allocation baseline: the majority control spends extraction calls it ignores.
Report real calls vs replayed calls separately. No claim of superiority over a
strong, optimally allocated budget baseline is permitted from this experiment.
Standalone greedy uses its existing seed schedule; audit and report any
differences from the greedy trajectory inside the shared trace.

Only after all three manifests are complete and streams match may the offline
evaluator load labels. Report accuracy, paired CIs/p-values, corrections and
regressions, supported/gate/override coverage, parsing failures and costs.
Proceed only if evidence beats both greedy and majority; otherwise stop this
version before validation/other datasets/memory. Even success is a pilot,
requiring new frozen evaluation and stronger efficient controls.

Commands (server, repo root):

```sh
METHOD_FAMILY=shared SMOKE_LIMIT=20 RUN_GREEDY=1 RUN_TAG=shared_v1_seed0_smoke20 \
  bash scripts/run_scienceqa_causal_smoke.sh
METHOD_FAMILY=shared SMOKE_LIMIT=400 RUN_GREEDY=1 RUN_TAG=shared_v1_seed0_dev400 \
  bash scripts/run_scienceqa_causal_smoke.sh
```

The launcher kills only its own vLLM process group on exit and releases GPU 2.
