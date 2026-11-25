# Hygon online TF-TTCL: exploratory same-question parallel smoke

On 2026-11-25, a separate seven-service prototype was checked on the first
20 ScienceQA-IMG train development questions. It uses HCU 1–7 to process
independent requests for the same question concurrently, while committing each
question's rule memory before moving to the next. The original sequential
20-item smoke and a separate sequential 400-item run used HCU 0. Both smoke
runs passed the label-free PPL, selection, image and memory-order audit.

| Arm | Same final answer, parallel versus sequential |
| --- | ---: |
| Greedy | 20/20 |
| No-rewrite five-candidate selection | 20/20 |
| Rewritten selection without memory | 20/20 |
| Full online memory | 19/20 |

Rules written had exactly matching texts on 9/20 questions; retrieved rule-ID
lists matched on 5/20. Differences can propagate through later memory, and
sampling on independently scheduled services does not guarantee identical
results even with the same seeds. The prototype is an experimental execution
variant, not a numerically identical replay.

The prediction files' creation-to-last-write spans were about 215 seconds for
the single-card smoke and 191 seconds for the seven-card smoke; these exclude
startup and the first sample, and the seven-card run competed for CPU with the
ongoing 400-item sequential run. These are not controlled end-to-end timing
measurements or a defensible speedup estimate. Seven-card services terminated
and HCU 1–7 were released after the test; the independent 400-item run kept
HCU 0.

Both output directories are stored on the server under
`/data/cp-experiments/tf-ttcl-hygon-20261124/results/`, as
`online_scienceqa_20_seed0` and `online_parallel_experimental_20_seed0`.
No labels were read or accuracy calculated for this check.
