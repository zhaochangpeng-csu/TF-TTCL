# Bounded shared observations and executable reasoning (typed v3 / v3.1)

Status: v3's full format audit failed unscored. V3.1 passed 61 tests, unscored
recalibration and full four-arm development scoring: evidence 87.25%,
greedy/majority 86.25%, independent matched budget 86.50%. The +0.75 pp
advantage over independent sampling is not significant (p=0.5078); all four
greedy repairs are scripted magnetic-pole tasks. See
[the v3.1 result report](results/scienceqa_typed_evidence_seed0.md).
The directional gate passes for a frozen validation regression check, not
memory or broad benchmark expansion. This is a limited hypothesis probe, not a
general verifier or evidence of top-conference novelty. The v2 pilot is
preserved in [its report](results/scienceqa_shared_evidence_seed0.md).

## What changes, and what does not

Separate syntax validity from semantic reliability. Request bounded JSON
Schema decoding through the same frozen Qwen2.5-VL-7B. See the
[official structured-output API](https://docs.vllm.ai/en/latest/features/structured_outputs/).
The installed vLLM 0.30.0 accepts `response_format.type=json_schema`; an
unsupported API must error rather than silently switch to unconstrained text.
Strict parsing independently checks duplicate keys, extra keys, types,
coordinate bounds, finite numbers and maximum array/string lengths. Overflow
and unreadable operands are semantic abstentions, not successful certificates.
The observation model still sees no question, context, options, candidate
answers, or correct answer. It transcribes the first image only.

V3's numeric JSON cells had no bound on the number of decimal digits. On the
full 400-item format-only run, a historical precipitation-chart mismatch for
8817 produced endlessly repeated decimal digits, reached the 1024-token cap,
and left unclosed JSON. The gate fell back; the strict audit failed and the
launcher exited before the independent budget arm. Its audit is preserved in
[the unscored format-failure artifact](results/scienceqa_typed_v3_format_failure.json).
There were 22 supported tasks, 88 observations, one parse failure, and no
exposure, memory or trace mismatch. **No labels or accuracy were read.**

V3.1 repairs this observed lexical failure, not an accuracy failure: numeric
cells are decimal strings constrained to at most six integer and four decimal
digits (12 characters including sign/point), or null. Out-of-format values must
abstain, not round. Parse validated strings to finite floats before execution.
Clarify that numeric headers exclude the row-label header and row lengths must
match; calibration 3722 had previously mixed the `Bottle` label header into
numeric columns. Record API finish reasons and mark any token-limit response
incomplete even if its JSON can be parsed. Operators, geometry thresholds,
candidate generation and decision policy are otherwise unchanged.

Frozen lexical compilation supports four small operator families:

- Facing poles: four collinear standalone N/S labels, with opposite poles in
  each consecutive pair. The farthest endpoints define an arbitrary axis;
  horizontal and vertical drawings use the same computation. Perpendicular
  deviation is at most 75/1000, adjacent projected separation at least 10/1000.
  Sorted facing/outer labels are invariant to axis reversal. This still
  assumes two aligned bar magnets; it is not learned object detection.
- Temperature deltas: faithful numeric-table transcription with unique
  initial/final Celsius temperature columns; all rows increase or all
  decrease. Generic `each/the <body>` reversed flow choices are supported.
  This uses a passive heat-transfer task assumption, not arbitrary active
  heating, phase transitions or internal reactions.
- Unique table extrema: the bounded `Which ... has the highest/lowest/...`
  template with a uniquely matching column and answer-choice row label.
  Ties, missing cells and ambiguous headers abstain.
- Filtered geometric counts: bounded `How many ... are shown/...` questions,
  numeric options, whitelisted shapes/colors and small/large attributes.
  Unknown required filter attributes, duplicate centers, ambiguous boundaries
  and more than sixteen objects abstain. Empty real observations cannot
  certify zero; empty blank observations only validate a negative control.

No generated code, external teacher, labels, logprobs, gradients or memory.
Maximum transcription tokens: letters 512, numeric tables 1024, objects 1536.
Standalone letters may be lowercase without a format failure, but only four
uppercase N/S labels yield a magnetic fact. This avoids treating an unrelated
lowercase letter on a historical mismatch as malformed JSON.

For supported tasks, transcribe real, full-size lossless PNG re-encoding,
white blank, and immediately preceding image (no mismatch on the first item).
No crop, rotation or flip. Real/fidelity facts must agree, blank output must
be literally empty, and mismatch must not reproduce the facts. Execute a
unique answer already proposed by the original six candidate trajectories;
otherwise retain the original greedy answer. Supported unanimous samples are
also processed, to measure gate coverage separately from answer changes.
The re-encoding may be byte-identical and is not independent visual evidence.
JSON constraints and probe sensitivity cannot guarantee transcription truth;
correlated visual mistakes remain possible.

## Calibration and frozen prospective stream

Use ScienceQA-IMG **train**, sorting SHA256(`20261011:<official-id>`).
Previously scored development streams occupy ranks 0–1199 and are not reused
for prospective evaluation. A separate **100-item format calibration slice
at ranks 1200–1299** is not scored; its label file is never opened by this
workflow. Its ordered-ID SHA256 is
`d84788c5f9d9c4f398244808b54264f79d4fcad601bd6eedfd2dad57210d8b97`.

The first 20 calibration items have no supported operator. Their smoke checks
only candidate generation, paired replay, conservative fallbacks and cleanup,
not the new constrained extraction path. Supplement with **all two supported
calibration items**, selected by the frozen question/choices router only:
12256 (magnets) and 3722 (bottle temperature). No correctness-based selection.
Count/extreme operators have unit fixtures but no calibration-set coverage;
do not claim empirically demonstrated generality for them.

Calibration outcomes (no labels): first-20 protocol audit passed with zero
supported tasks and 51,410 shared tokens. Core-path audit passed on both
supported items with zero parsing errors, exposure, memory writes or replay
mismatches. One real/fidelity observation pair was stable and passed the
blank/mismatch gate; the other abstained on an incomplete observation. A gate
changed one greedy answer; its correctness is intentionally unknown. Shared
compute cost was 4,447 tokens. This verifies execution, not performance.

After smoke and tests pass, freeze prompts, schemas, operators and selection
policy in a git commit. Then prepare **400 new items at ranks 1300–1699** in
`dev400_typed.inference.jsonl`, with a separate `.labels.jsonl`. Do not inspect
the new stream to expand operators and do not change prompts by accuracy.
Only after all four full runs are completed and matched may labels be read.

For v3.1, re-run the **100-item unscored calibration flow**, which includes the
original first-20 smoke and both eligible tasks with realistic historical
mismatches. Then commit/freeze again and re-run the same 400 **label-unread**
development inputs in a new versioned artifact directory. V3.1 used v3's
unlabeled full-flow format failure as a diagnostic; it is not a completely
untouched input holdout, and must not be described as such. No new operators
are chosen from this flow and no prompt change is based on correctness.
Final benchmark confirmation must use different, frozen evaluation inputs.

V3.1 calibration completed all four 100-item arms. Eight observation responses
had zero parse errors, zero token-limit finishes, and two literal-empty blank
controls. One original/fidelity fact pair was stable but its executable answer
was absent from the six candidates; the other task had incomplete observations.
Thus no override was made. Shared compute: 251,616 tokens; independent sampling:
251,545 tokens (0.0282% difference), greedy: 39,583 tokens. Protocol and four-arm
completion checks passed; no calibration accuracy was read. GPU 2 returned to
2 MiB / 0% utilization on launcher exit. These are workflow checks, not positive
accuracy evidence.

## Four controls and accounting

1. `zero_shot`: standalone original greedy.
2. `shared_budget`: six augmented candidates plus paid observations, majority
   selection ignoring certificates. This is a **selection-policy ablation**,
   not an efficient allocation baseline.
3. `shared_evidence`: exactly the same immutable response trace, greedy
   fallback plus certificate-based override. Replay executes no new VLM calls;
   replayed cost is the cost of the shared computation, not a second expense.
4. `efficient_budget`: original greedy plus independent temperature-0.7
   original-question answers, majority with earliest-answer tie-breaking.
   Six mandatory useful generations, up to twelve, using cumulative token
   targets to choose whether an additional estimated-cost generation brings
   total cost closer to the reference. No padding, paraphrases or certificates.

The independent control gets only ordered sample IDs and integer token costs
exported from the complete shared reference: **no predictions or evidence**.
The cost plan rejects other seeds, datasets, model revisions, source-file
hashes, order/limit, ID sequences and extra fields. Its content hash is stored
in configuration so resume rejects modified plans. Budget allocation uses
real prompt+completion tokens and a deterministic average-cost estimate,
without examining answer agreement or labels. Cost matching is measured at
the whole-stream level (required within ±5%), not promised per sample.
Because reference costs are known after reference inference, this comparator
is an offline cost-matched control, not an online deployment budget policy.
Fixed candidate-count bounds can cause a cost mismatch; if so, report it and
do not declare a matched-budget win or retune using accuracy.

All runs record git SHA, model revision, prompt/code hash, seed, raw observation
traces, schema hashes, token limits, per-example/aggregate usage and model
latency. Independent replay caches use a separate directory. The four-arm
evaluator rejects incomplete or nonmatching streams before label loading.
Report supported-task counts (including zero coverage), stability, overflow,
parse failures, blank hallucinations, gates, overrides, paired corrections/
regressions, confidence intervals, p-values, costs and internal-vs-standalone
greedy differences. Keep raw model/data/cache artifacts out of Git.

Continue only if strict protocol audit passes and evidence beats greedy,
the selection-policy ablation, and the independent ±5% budget control.
Even then, one small train-development stream is not confirmation: freeze a
new prospective evaluation before ScienceQA validation, additional models/
datasets, or memory. A nonsignificant or extremely low-coverage gain is not
enough to justify broad benchmark expansion for a top-conference main track.
ScienceQA validation was previously used for development diagnostics; its
re-evaluation is a regression check, not untouched confirmation.

Server commands (repo root, GPU 2, `ttt` environment, `ttt_ex`):

```sh
CONFIG_PATH=core/config/mm_scienceqa_train_typed.yaml METHOD_FAMILY=shared \
DATA_PATH=data/multimodal/scienceqa_modelscope_train/dev100_typed_calibration.inference.jsonl \
RUN_TAG=typed_v3_seed0_calibration_smoke20 SMOKE_LIMIT=20 RUN_GREEDY=1 \
  bash scripts/run_scienceqa_causal_smoke.sh

CONFIG_PATH=core/config/mm_scienceqa_train_typed.yaml METHOD_FAMILY=shared \
RUN_TAG=typed_v3p1_seed0_dev400 SMOKE_LIMIT=400 RUN_GREEDY=1 RUN_EFFICIENT=1 \
  bash scripts/run_scienceqa_causal_smoke.sh
```

The launcher refuses an occupied GPU and kills only its own model-service
process group on exit. GPU 0/1/3 and unrelated tasks are untouched.
