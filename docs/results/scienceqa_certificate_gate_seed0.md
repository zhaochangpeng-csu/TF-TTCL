# ScienceQA-IMG counterfactual evidence-certificate gate (seed 0)

This is a method-development result on 400 image-bearing ScienceQA **train**
examples, not a validation/test benchmark claim. The subset was selected by
the lowest `SHA256("20261011:<official-id>")` values; its ordered ID-list hash
is `feb038bd7daf92334ca4475f9242625b3c935cfa0a9d9c2703269b7ea3fd76b0`.
Inference used a physically label-free JSONL. The separate label file was first
read only after all three manifests reported `completed`.

All runs used frozen Qwen2.5-VL-7B-Instruct revision
`cc594898137f460bfe9f0759e9844b3ce807cfb5` at Git commit `84e1b36`.
`certificate_budget` and `certificate_evidence` made the same candidate and
certificate calls. The evidence arm could replace the original greedy answer
only when original/fidelity views agreed after a response-position swap and the
selected answer lost visible support on every blank/mismatched-image probe.
No memory was read or written.

| Method | Accuracy | 95% bootstrap CI | Prompt tokens | Completion tokens | Summed latency |
|---|---:|---:|---:|---:|---:|
| standalone greedy | 87.25% | [84.00%, 90.50%] | 137,887 | 1,979 | 63.46 s |
| certificate budget | 87.50% | [84.25%, 90.75%] | 904,831 | 26,444 | 423.72 s |
| certificate evidence | 87.25% | [84.00%, 90.50%] | 904,831 | 26,450 | 421.20 s |

The evidence arm did not beat either required baseline. Relative to standalone
greedy its delta was exactly 0.00 percentage points (95% paired bootstrap CI
[-0.75, 0.75], paired randomization p=1.0): one answer improved and one
regressed. Relative to the budget-matched arm it lost 0.25 percentage points
(95% CI [-0.75, 0.00], p=1.0): zero improvements and one regression.

The two certificate arms had identical prompt-token totals and differed by six
completion tokens (0.00064% of total tokens). Candidate answers were identical
for all 400 examples. Two call traces differed only by a newline in an
equivalent paraphrase on unanimous samples that never invoked the certificate;
all candidate answers and image hashes remained identical. The standalone
greedy run differed from the budget arm's internal greedy trajectory once,
improving the budget score by one example; the primary causal comparison is
therefore evidence versus its matched budget arm.

## Gate behavior and failure analysis

- 385/400 samples had no alternative candidate.
- 15/400 had a candidate disagreement.
- 4/400 had stable original/fidelity certificate winners.
- 3/400 also passed the counterfactual-support-drop gate.
- Only 1/400 passed sample would actually change the internal greedy answer.
- All certificate JSON parsed successfully.

The sole gate-induced change, ScienceQA ID `6514`, was a regression. The image
highlights Dominica and the original greedy answer was option A (Dominica).
Both real-image certificate views instead asserted option D (Saint Vincent and
the Grenadines), inventing geographic descriptions such as “southwest of
Barbados” and “north of Grenada.” Blank and mismatched-image probes correctly
removed support, but they established only that the verifier was image
dependent—not that its visual identification was correct.

This is the central negative result: same-VLM invariance plus counterfactual
sensitivity does not break correlated visual-recognition errors. It can produce
a stable, apparently grounded certificate for a wrong answer. The predeclared
gate therefore fails. Validation-set expansion, MathVista/TextVQA runs, and
online memory are stopped for this method version rather than spending more
compute on a selector that underperforms its matched control.
