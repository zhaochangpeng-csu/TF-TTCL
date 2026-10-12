# Shared visual observations, different executable reasoning (v2)

Status: development evaluation completed. V2 corrects one answer (+0.25 pp)
but is not significant and fails the strict full-stream format audit. Do not
expand v2 to benchmarks or memory. See
[the result report](results/scienceqa_shared_evidence_seed0.md).

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

The same frozen VLM transcribes standalone uppercase letters with normalized
center positions, or numeric table rows. The letter prompt names neither
magnetism nor expected N/S identities; it gives no populated schema example.
The parser requires exactly four N/S labels, distinct left-to-right positions
(at least 10/1000 apart), and a horizontal band (y spread at most 150/1000).
The middle two labels are the shared facing-pole facts. Stability compares the
ordered letters, not exact coordinate values, avoiding irrelevant OCR jitter.
This restricted geometry is not a learned object detector and cannot prove
that every four-letter observation actually corresponds to two bar magnets.
It sees neither question, context, choices, candidates, nor a requested answer.
Strict schemas reject extra fields, fabricated absence on parse errors,
nonfinite numbers, and ambiguous formatting. Whether letters were hallucinated
still requires empirical diagnostics, not a parser guarantee.

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

The fixed first 20 items have no supported tasks. Add a separate core-path
smoke on all 12 supported magnet items, filtered **only from the new inference
file** by the already-frozen routing predicate. No labels or correctness-based
selection are allowed. This diagnostic does not change the full 400-item
stream, prompts, or operators. The stream contains no supported vial tables,
so temperature inference is not empirically evaluated by this pilot.

The initial v1 core-path smoke returned N/S labels for all 12 white blank
images, despite well-formed JSON and stable real-image outputs. Its gate
correctly rejected every sample. Preserve those artifacts as a failed
observability control. V2 was changed using **only these label-free
transcriptions**, not accuracy: replace prompted facing-pole fields with
literal positional OCR, remove filled-in letter/numeric examples, and re-run
smoke before scoring. No labels from the new stream were opened for redesign.

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
METHOD_FAMILY=shared SMOKE_LIMIT=20 RUN_GREEDY=1 RUN_TAG=shared_v2_seed0_smoke20 \
  bash scripts/run_scienceqa_causal_smoke.sh
METHOD_FAMILY=shared SMOKE_LIMIT=400 RUN_GREEDY=1 RUN_TAG=shared_v2_seed0_dev400 \
  bash scripts/run_scienceqa_causal_smoke.sh
```

The launcher kills only its own vLLM process group on exit and releases GPU 2.
