# Hygon HCU migration: paired TextVQA greedy replay

The same 400 TextVQA train development questions, images, model weights,
prompts and greedy sampling settings were evaluated on both NVIDIA and Hygon
inference stacks. This evaluates a combined hardware and software migration,
not the effect of the accelerator alone. All 400 outputs passed label-free
alignment checks before isolated targets were opened for scoring.

| Environment | Local TextVQA soft score |
| --- | ---: |
| Original NVIDIA stack | 72.50% |
| Hygon HCU stack | 73.175% |

The final answer strings matched on 364/400 questions (91.00%). Of the 36
changed answers, 11 improved the local soft score, seven worsened it, and 18
had equal scores. The paired score difference was +0.675 percentage points,
95% bootstrap CI [−1.075, +2.425] pp, paired randomization p=0.4764. The
confidence interval does not establish equivalence. These are development
results for one model and one ordering seed.

The local VQA soft-score implementation has not been checked for exact parity
with the official EvalAI processor, so these are not official leaderboard
results. The replay used a 32,768-token model context on eight independent
HCU services; all eight services exited after completing inference. The
20-item label-free smoke is part of the 400 and is not an independent test.

Server output:
`/data/cp-experiments/tf-ttcl-hygon-20261124/results/textvqa_dev400_8gpu/`.
Input SHA-256: `e7b3865aa534594634ab9f9c136344d1c2673f317a2578722f09677cc8189b74`.
NVIDIA predictions SHA-256:
`4750646aa363532c6169f98c332e0e3ef32dc1e04ec13b8feb729b3874e762c4`.
