# ScienceQA train: four-arm online pseudo-rule transfer (development)

Date: 2026-10-14. Frozen run SHA: `0eab28f2d05607fab429975bf40296b66804951a`.
Model: Qwen2.5-VL-7B-Instruct, seed 0, 400 disjoint ScienceQA train items.
This is a single-stream development pilot, not an untouched validation result.

The four arms shared each item's four candidate calls, typed visual observations,
and rule distillation. Only historical rule writes and retrieval differed. All
four final responses were committed before the current rule could enter memory.
The input was label-free during inference. The 400-row, four-arm audit passed;
only then were isolated train labels opened for offline scoring. The run exited
with status 0, and GPU 2 returned to 2 MiB. The `none` arm is **not** a raw
zero-shot baseline: it still receives the shared candidates and final
refinement prompt, but no historical rules.

| Arm | Accuracy | Correct / 400 | Corrected / harmed vs none | Writes | Retrievals | Paired p vs none |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No memory | 90.50% | 362 | — | 0 | 0 | — |
| Noisy memory, lexical top-2 | 91.75% | 367 | 6 / 1 | 357 | 399 | 0.125 |
| Typed-gated, trust-weighted memory | 91.25% | 365 | 3 / 0 | 9 | 272 | 0.25 |
| Random memory from noisy write pool | 91.25% | 365 | 3 / 0 | 357 | 399 | 0.25 |

The paired bootstrap deltas versus no memory were +1.25 pp for noisy
(95% CI [0, 2.50] pp), +0.75 pp for gated ([0, 1.75] pp), and +0.75 pp
for random ([0, 1.75] pp). These are **not statistically persuasive** in this
small, single-seed experiment. Eight of 400 items had any final-answer
disagreement across arms; most retrieved rules did not change the answer.
The observed noisy advantage over random is two correct items, not evidence
that lexical retrieval or rule semantics are reliably causal.

## Coverage and cost

Only 19/400 questions were supported by the fixed typed operator router;
10/400 passed its visual-evidence gate; 9 became gated memory writes. The gate
bank was retrieved on 272 items, often reusing a tiny number of magnet rules
for unrelated questions. A sparse, magnet-heavy gate cannot test general
multimodal evidence-aware memory. Candidate answers disagreed on 48/400;
357/400 rules passed the current structural parser. That parser does **not**
enforce the prompt's ban on example-specific facts.

Total logical tokens per arm (shared candidate/evidence/distillation work plus
that arm's final refinement):

| Arm | Prompt tokens | Completion tokens | Prompt difference vs none |
| --- | ---: | ---: | ---: |
| No memory | 1,419,959 | 197,352 | — |
| Noisy | 1,443,003 | 196,385 | +1.62% |
| Gated | 1,433,110 | 196,784 | +0.93% |
| Random | 1,443,038 | 196,451 | +1.63% |

Shared extraction accounted for 1,185,497 prompt and 164,786 completion
tokens. Aggregate token totals are close, but the final refinement prompt is
longer when memory is present. This is not an independent budget-matched
generation/reflection baseline. The four arms ran in one process and exact
duplicate requests could replay from immutable traces; logical token totals
do not represent a fourfold standalone deployment bill.

## What the actual inputs and outputs show

The local, ignored casebook at
`core/exp_res/inspection/transfer_online_dev400_casebook.md` records the
first three stream items, every arm disagreement and every gate
case (20 total), selected without correctness. It includes the label-free
question/options, original candidate completions, exact final user prompt,
retrieved source rules, raw final response, and offline target added only after
the complete-run audit. The JSON and Markdown casebooks and prediction traces
remain outside Git; this report contains only aggregate metrics and a few
diagnostic cases.

| Sample | Observed result | Research implication |
| --- | --- | --- |
| 19811, equator/continents | No memory chose C; all three memory arms chose correct A, including random and arms retrieving an irrelevant magnet rule. | A retrieved-context effect is not automatically useful knowledge transfer. |
| 20681, highlighted country | No memory was correct C; noisy retrieval of a rule asserting an earlier map showed Grenada changed it to wrong D. | Example-specific conclusions leaked through rule distillation and can pollute later map questions. |
| 15250 and 6937, highlighted countries | Noisy retrieval changed wrong answers to correct ones, despite the same Grenada shortcut in its retrieved pool. | Individual wins do not validate a rule's semantic content. |
| 17087, lunch trade | Noisy, gated and random all fixed B to correct C; gated retrieved unrelated magnet rules, random unrelated US-state map rules. | Prompt/context perturbation is a live alternative explanation. |
| 8823, magnets | The gate certified A/attract, but the isolated label is B/repel; noisy memory alone answered B. The image visibly has N facing N. The model's structured OCR placed one N at the wrong vertical position, yielding a false N-S facing pair. | Syntax-valid, real/fidelity-stable observations are not necessarily true visual facts. |

In sample 8823 the real and fidelity image hashes were identical. Thus their
matching OCR outputs are repeats on the same bytes, **not independent
confirmation**. A blank-image drop establishes image sensitivity, but not
transcription correctness. The gate's one erroneous certificate among ten is
especially relevant because an incorrect certified rule can propagate.

## Decision and next test

This pilot does **not** justify expanding the current method to ScienceQA
validation or claiming a top-conference result. Gated memory failed to beat
noisy or random memory; none of the differences against no memory passes the
reported paired test. The next development experiment should isolate the
retrieved-context effect: compare task-matched rules with shuffled rules and
length/content-matched neutral text on the same held-out train stream,
repeating order seeds. Enforce a source-fact-free rule parser and test true
visual perturbations plus geometry/observation audits before treating typed
certificates as reliable. Only if relevance and trustworthy verification
provide a repeatable gain should the online-memory method move to validation
and additional datasets/models.

Offline aggregate JSON: server
`core/exp_res/mm/scienceqa_train/qwen25vl/transfer_online_v1_seed0_dev400/comparison.json`;
local ignored copy `core/exp_res/inspection/transfer_online_dev400_comparison.json`.
Input SHA-256: `1459dbe9089bdd31fcc9a2078289a07de432669b75c0a9f02fa97fc8568c289c`.
Label SHA-256: `73466c5471c9e58af77f1dc2dade04cfe669e18097608c3a241213c7916bac03`.
