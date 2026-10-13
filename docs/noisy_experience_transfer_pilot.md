# Revised next stage: transfer under noisy pseudo-experiences

## Objective and present status

Ask whether historical, uncertified visual-QA experience improves later
reasoning, and whether evidence-weighted writes reduce its harmful transfer.
Do not assume a pseudo-positive is true, or demand perfect pseudo-label purity.
Answer accuracy, visual grounding and transferable rule utility are separate.

Current implementation supplies read-only baseline I/O case cards, tested
pseudo-pair/write/random-retrieval policies, and a finite reasoning-format
smoke launcher. The [V2 20-item format smoke](results/scienceqa_transfer_format_smoke.md)
passed all 40 calls without opening labels; GPU 2 was released.
**The four-arm online memory runner and its results are not
yet implemented/completed.** The frozen validation regression failed its
contract audit and remains unscored; do not loosen that audit retrospectively.
See [actual input/output review](results/scienceqa_baseline_io_inspection.md).

## Next executable checkpoint

Use unused ScienceQA train rank slice 1700–2099 under the existing
SHA256(`20261011:<id>`) order: `dev400_transfer.*`. Check IDs against all prior
development subset manifests. Preparation copies labels into a separate file;
the smoke has no label argument and opens no labels. The first 20 items are
format-development samples, not an untouched confirmation set.

On GPU 2, generate greedy plus temperature-0.7 responses to the same original
image/question; 40 calls total, max 384 output tokens each. New versioned
prompt asks for `Observed`, `Reasoning`, `Final answer`, permitting “image not
required” or “insufficient” rather than invented visual evidence. Record exact
requests, raw outputs, normalized option letters, finish reasons, model
revision, seed, script/prompt/input hashes, Git SHA/worktree status, token and
latency totals. Format checks require nonempty sections, one valid final
option and nontruncated output; they do NOT judge factual correctness. Stop
after this checkpoint. The launcher releases only its own model process group.
No extra model or embedding download is needed for this format check.

First v1 run (system-only format instruction): 40/40 calls completed, but only
1/40 has the required sections; 18,998 prompt + 1,489 completion tokens,
22.49 seconds summed model latency. No labels read; GPU 2 released afterward.
This is a failed **format** smoke, not a poor-accuracy result. Its exact raw
requests remain in `transfer_format_v1_seed0_smoke20/response_trace`.
V2 additionally puts the same observation/inference section requirements at
the end of the user message, based solely on unlabeled formatting failures.
It uses a new output directory/namespace and keeps v1 untouched. No inference
operator, physics gate or frozen validation prompt was changed. This is
development on the same first-20 inputs, not a new held-out experiment.

```sh
# Server repo, conda ttt, tmux ttt_ex; GPU 2 must be idle:
bash scripts/run_transfer_format_smoke.sh
```

## Subsequent controlled memory pilot (freeze before running)

Start with rule-only memory, not both episode/image memory at once. Equalize
reasoning prompts, candidate generation, evidence computation, distillation
and final refinement calls across arms. The shared candidate bank is
unconditioned on memory. A separate final reasoning call consults historical
rules; only after final answer logging may current experience be appended.
This isolates transfer from a better current-sample selector. It is an
experience-library intervention, **not yet fully on-policy TF-TTCL**: initial
candidate/rule generation remains memory-independent. A subsequent on-policy
comparison must disclose that candidates and memory contents then diverge.

| Arm | Writes | Retrieval |
| --- | --- | --- |
| no-memory | Pays identical rule-distillation cost, discards rules | Same refinement instruction, no historical rules |
| noisy pseudo-memory | Largest-answer-group pseudo-pair; skip all-distinct; all-agree has no negative | Relevant historical rules; weight=1, no hidden evidence filter |
| gated/weighted | Same pseudo-pair; write only if evidence supports that same positive | Same retriever; gate/weight are the only intended changes |
| random-memory | Exactly the noisy arm's write pool | Deterministic random historical rules, same count/length cap |

`core/multimodal/transfer_policy.py` implements these write decisions, CRT-style
grouping and min-PPL representative/tie selection when logprobs exist. Without
logprobs it uses declared trajectory-order tie-breaking, not a fictitious PPL
score. Therefore call it **original-style pseudo-memory**, not an exact paper
reproduction. A different answer is a pseudo-negative, not a proven visual
contradiction. Evidence gated writing must not substitute a new positive
answer and thus confound selection with memory. Consensus is a heuristic
ranking weight, never a calibrated probability. Persist semantic option text
and the source question/options, not bare A/B conclusions.

The evidence gate is not yet a universal implementation: the old typed verifier
is limited to four lexical operators, and its fidelity images may be byte
identical. Declare coverage and correlated observations; do not describe them
as independent confirmations. Freeze a bounded gate on unscored development
inputs before the memory run, including unsupported/overflow/mismatch behavior.
Do not silently use candidate-conditioned self-verification as an oracle.

Requirements before four-arm inference:

- Same ordered IDs/seed/model revision; no labels in inputs, prompts, rules,
  traces or retrieval. No cross-seed/dataset trace reuse. Resume restores the
  exact rule bank and only a committed ordered prefix; current/future entries
  cannot be retrieved. Trace namespaces include gate/prompt/policy versions.
- Use a declared real frozen text embedder or transparent lexical retrieval
  for the initial retrieval ablation. The hash backend is a testing fixture,
  not a research retriever. Later compare Qwen3-VL text/image/joint retrieval
  with embeddings and raw evidence episodes.
- If memory arm costs differ through context length, report exact logical
  prompt/completion costs (shared replay is not new expenditure), and add an
  independent sampling/reflection baseline within ±5% total tokens. Identical
  call count or output caps alone do not establish budget matching.
- Report prompt and completion tokens separately. Earlier controls match
  **total** tokens, not necessarily the original plan's **generation** tokens;
  do not call them strict generation-budget-matched. For the main comparison,
  include a ±5% completion-token-matched control as well as the total-cost
  control (or disclose which budget cannot be matched); also report measured
  latency because prefill and decoding have different costs.
- Include generic checking/reflection without memory, to test whether a
  helpful instruction rather than accumulated experience explains a gain.
- Store exact source and retrieved rule content, rule-use/answer-change trace,
  and any raw observation on every example for later failure analysis. Require
  distilled rules to be nonempty parseable fields; do not promote arbitrary
  malformed model text into a trusted rule.

Only after all planned arms, budget controls and completion/namespace audits
finish, open labels offline. Report final accuracy and paired corrected/harmed
counts, per-question-family and first/middle/last stream segments, write and
retrieval coverage, evidence-candidate disagreement, token/latency/memory cost.
Inspect harmful transfers against the exact historical rule that was exposed;
retrieval association alone is not proof of causal contamination.

Then add a predeclared small fraction of synthetically inverted rules or
mismatched past rules (never GT-selected at inference). This is a perturbation
control, not guaranteed “wrong memory”; actual wrongness is diagnosed offline.
Measure excess downstream harm relative to the same unperturbed write pool.
Run at least three stream-order seeds before drawing conclusions. A model
exceeding its no-memory counterpart is useful; the top-conference claim still
needs budget-matched gains, cross-model/dataset replication and evidence of
reduced harmful transfer. Shrink the claim if gains are only template-specific.
