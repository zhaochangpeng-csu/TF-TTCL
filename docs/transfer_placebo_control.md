# Exact-candidate placebo context control

The 400-item ScienceQA train online pilot was scored before this diagnostic
was designed. This follow-up uses the **same, previously scored development
stream**, so it is exploratory, not fresh confirmation. It reuses the audited
current image, four candidate answers, pseudo-pair, and the original noisy
arm's exact final user prompt. It replaces only the historical rule block:
each retrieved positive and negative rule becomes a fixed, answer-free
"earlier record was withheld" statement with the same regex word count and
line structure. The system prompt, current candidates, source retrieval
positions, greedy decoding seed, image and 384-token output cap are fixed.
Model-reported prompt tokens are logged to audit actual, rather than assumed,
token matching.

Comparison arms are the source run's no-memory, lexical/noisy, and random
history arms, plus the new placebo. The random arm already shuffles retrieval
from the same noisy write pool. This isolates whether extra historical-context
text alone moves answers, but it does not prove that a retrieved rule is
causally used. No additional candidate or distillation call is made.

All inference reads only the source run's label-free records and the public
ScienceQA input file. The source run must pass the complete 400-row audit.
The placebo run has an independent immutable response trace, ordered-prefix
resume checks, model revision, source/input SHA and Git/code/prompt hashes.
The 20-item smoke is never scored. Only after the 400-item placebo audit may
the offline comparator open the isolated labels. The result is reported as a
post-hoc diagnostic because source labels have already been viewed.

On the server, use GPU 2 only in a persistent tmux window:

```sh
PRECHECK_ONLY=1 PLACEBO_COUNT=20 bash scripts/run_transfer_placebo.sh
PLACEBO_COUNT=20 PLACEBO_RUN_NAME=smoke20 bash scripts/run_transfer_placebo.sh
PLACEBO_COUNT=400 PLACEBO_RUN_NAME=dev400 bash scripts/run_transfer_placebo.sh
```

The launcher requires a clean committed worktree, refuses an occupied GPU or
API port, and releases only its owned vLLM process group on exit. After the
full run and audit, evaluate offline:

```sh
/home/lthpc/.conda/envs/ttt/bin/python evaluate/multimodal/compare_placebo.py \
  --source core/exp_res/mm/scienceqa_train/qwen25vl/transfer_online_v1_seed0_dev400 \
  --run core/exp_res/mm/scienceqa_train/qwen25vl/transfer_placebo_v1_seed0_dev400 \
  --labels data/multimodal/scienceqa_modelscope_train/dev400_transfer.labels.jsonl \
  --expected-n 400 \
  --output core/exp_res/mm/scienceqa_train/qwen25vl/transfer_placebo_v1_seed0_dev400/comparison.json
```

If the placebo is comparable to the lexical/noisy arm, the present memory
gain is not distinguishable from a context effect. If lexical/noisy beats
placebo and random on this development stream, freeze the protocol and test
on a **new disjoint** train slice before any validation expansion. Separately,
the typed gate's false visual certificate and identical real/fidelity image
hashes require redesign; this placebo experiment does not repair them.
