# Hygon HCU migration: ScienceQA five-candidate selection replay

The original ScienceQA-IMG train development stream of 400 questions was
replayed using Qwen2.5-VL-7B-Instruct. The Hygon run reused the completed
Hygon greedy Teacher response and generated four sampled Student candidates
per question, then applied the original closed-question clustering and
generation-PPL selection rules. It did not add paraphrases or online memory.

| Environment | Correct / 400 | Accuracy |
| --- | ---: | ---: |
| Original NVIDIA stack | 361 | 90.25% |
| Hygon HCU stack | 365 | 91.25% |

Final choices matched for 380/400 questions (95%). Of the 20 changed choices,
10 improved and six worsened; the other four had equal correctness. The paired
difference was +1.00 percentage point, 95% bootstrap CI [−1.00, +3.00] pp;
paired randomization p=0.4545. This is not a test of statistical equivalence.

The smoke and all 400 predictions were audited without reading labels;
offline scoring happened only after all outputs, PPL values and selections
passed audit. Each run used the same original image and question and the same
configured seeds and sampling parameters as its NVIDIA counterpart. The
different inference stacks do not guarantee identical random samples even at
the same seed, so the observed difference includes both software/hardware
and sampling effects.

The 20-item smoke ran before the full replay. The launcher then used a new
port range because immediate rebinding of its previous ports failed; the
completed smoke was retained and validated before the full run. Eight HCU
services were used and all memory was released afterward. Output:
`/data/cp-experiments/tf-ttcl-hygon-20261124/results/vote_dev400_8gpu/`.
