# TextVQA short-answer format and online memory: development protocol

Freeze the experiment before opening labels. The questions are 400 official
TextVQA train samples selected deterministically with seed 20261126, excluding
question and image IDs from the previous 400-item train experiment and 5,000
validation questions; choose at most one question per image. Selection uses
only identifiers, not answer values. Data preparation may read annotations to
export isolated targets, but inference receives no targets or explanations.

Run two independent ordered streams with frozen Qwen2.5-VL-7B-Instruct and
the same seed and images: the original direct-transfer prompts, and the same
prompts with the frozen short-final-answer instruction from
`core/multimodal/textvqa_format_factorial.py`. Both streams retain the
original candidate generation, open-domain full-response similarity, PPL
selection, and online rule-writing algorithms. Each stream has its own memory;
the current sample may only retrieve rules written by earlier samples.

Frozen prompt hashes are `8b48f2132cfe999b2610f2b68424f54c39c3a4ffe4208838ba81a7babfca5193`
for `original` and `88eee4c8df7faa061cb04e3b161c896db20bccfbc0f97a03ee60680c803e0b54`
for `short_final`. The hashes cover the Teacher, Student, Tutor, rule header,
and positive/negative rule templates in the project's canonical JSON digest.

This is protocol v2, frozen before labels were used for scoring or prompt choice.
Dataset preparation had previously read the annotations to isolate labels;
neither inference nor smoke inspection reads them. In the v1 unlabeled
20-item smoke, only 245/320 short-policy answering calls passed the required
reasoning-before-final check (versus 310/320 for original); in the short-policy
memory arm, 38/100 calls lacked a `Final answer:` marker. The formal run was
stopped. Some historical rules could conflict with the requested format, but
this has not been established as the cause. V2 explicitly requires both exact
section labels and two reasoning sentences even when learned rules are present.
The original policy and the audit threshold remain unchanged. V1 outputs remain
preserved separately and no smoke correctness was inspected.

Each stream contains an original candidate-selection arm without memory and
a full online memory arm. The four predeclared factorial cells are therefore
original/no-memory, original/memory, short-final/no-memory, and
short-final/memory. The prompts influence candidate generation, selection
and subsequent rule content as well as final-answer form. Consequently, this
tests the *whole prompt-policy effect*, not an isolated causal intervention on
the final answer alone. No external teacher, labelled examples, image
augmentation or training are introduced.

For each stream first complete 20 label-free smoke items and check message
format, real image paths/hashes, complete PPL, all candidate records and
history-only retrieval. Do not inspect smoke correctness. Then process the
frozen 400-item stream. Both full streams must finish and pass label-free
audit before offline scoring opens the isolated target file. Record model,
software, source/prompt hashes, seed, per-arm logical tokens, physical trace
requests and time. Report observed token budgets; treat comparisons exceeding
5% difference as not token matched, and do not claim equal budgets by design.

Primary contrasts are the two within-format memory effects, the two
within-memory format effects, and their paired interaction:
`(short/memory - short/no-memory) - (original/memory - original/no-memory)`.
Report paired 95% bootstrap intervals and randomization tests, changed-answer
counts and improved/worsened items, output length and explicit format failure
rates. This is a single-stream development experiment, not confirmation or a
license to tune against final benchmark labels. The existing local TextVQA
soft score is not yet verified for official EvalAI parity; label accordingly.
