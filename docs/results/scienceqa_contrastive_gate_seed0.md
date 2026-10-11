# ScienceQA-IMG contrastive selection gate (seed 0)

This is a development-set result, not a final benchmark claim. ScienceQA-IMG
validation labels were isolated during inference and opened only after both
paired runs completed. Both runs used Qwen2.5-VL-7B-Instruct at commit
`e5c51b7`.

| Method | Accuracy | 95% bootstrap CI | Prompt tokens | Completion tokens |
|---|---:|---:|---:|---:|
| contrastive budget | 87.506% | [86.075%, 88.889%] | 4,799,354 | 164,005 |
| contrastive evidence | 87.124% | [85.646%, 88.555%] | 4,799,352 | 164,004 |

The paired evidence-minus-budget delta is -0.381 percentage points (95% CI
[-0.858, 0.048], paired randomization p=0.147). The selector changed 32
answers: 8 improved, 16 regressed, and 8 changed between two wrong answers.
Therefore it fails the predeclared gate and must not be connected to online
memory.

Of 2,097 samples, 1,860 had unanimous candidates (91.290% accuracy). The 237
disagreement samples had 57.806% majority-vote accuracy and 89.873% candidate
oracle accuracy, so selection remains the bottleneck. Two-view adjudication was
stable on 133 disagreements but reduced accuracy there from 60.150% to 54.135%.
The stricter explicit-contradiction gate changed only three answers, with zero
wins and two losses.

Trajectory analysis exposed an augmentation defect. `text_variant_1` and
`joint_variant` scored only 84.692% and 84.740%, while `original_greedy` scored
87.601%. The augmenter rewrote the complete ScienceQA prompt, causing its first
variant to omit the original Context and Hint; its second slot often fell back
to the original prompt. The next experiment therefore preserves all context
verbatim, rewrites only the final query clause, and rejects variants that alter
protected numbers, logical operators, quoted terms, or proper names. No memory
experiment proceeds until this corrected candidate stream passes a new paired
selection gate.
