# TF-TTCL faithful multimodal direct-transfer baseline: staged protocol

This is a **new independent arm**, `tf_ttcl_mm_direct`. The existing
`transfer_online_v1_rule_only_typed_gate` `noisy` arm is a simplified
pseudo-experience mechanism, not the original TF-TTCL algorithm. The original
text implementation under `core/source/` remains unchanged.

The [paper](https://aclanthology.org/2026.findings-acl.1482.pdf) and the
repository's original `core/source/compare/selection_kernel.py` agree on
four sampled Students plus a greedy Teacher; closed-task all-different,
all-same, largest-answer-cluster and min-PPL tie cases; and the open-task
Teacher-similarity top-half split. The paper defines PPL from the token
logprobs of each **generated sequence under its own input context**. Its
reported defaults are temperature 0 for Teacher, temperature 0.7/top-p 0.9
for Tutor and Students, Qwen3-Embedding-0.6B for text rules and queries,
and separate cosine top-30 positive/negative rule retrieval. The direct
transfer changes only each answering input to `(original image, text)`;
the Tutor rewrites text only. There is no image augmentation, visual
verification, trust gate, or extra answer-refinement call.

## Execution gates

1. Pure selection tests must pass for all closed-task cases, open-task
   partitioning, and missing/incomplete/nonfinite logprobs. No candidate
   order or self-reported confidence may replace PPL.
2. On an **idle** GPU 3 (user-authorized change on 2026-10-19), start an owned Qwen2.5-VL-7B service in `ttt`, send
   one label-free ScienceQA image request with `logprobs=True`, and verify
   `len(logprobs) == completion_tokens > 0` and all values/PPL finite.
   Record the raw response and stop the owned service. An absent or
   incomplete vector blocks this arm; do not continue to a purported
   faithful run.
3. Only if gate 2 passes, isolate a new ScienceQA train development stream
   disjoint from previously scored train subsets. Build the five-candidate
   paired no-memory/full-online runner, exact positive/negative rule banks,
   text embeddings and top-30 retrieval. A 20-item label-free smoke must
   pass full I/O, budget and no-leakage audit before scoring any labels.
4. Score the full ScienceQA development comparison only after all arms
   complete and audit passes. TextVQA uses a **separate open-task branch**
   and an untouched input stream; it must not inherit the ScienceQA majority
   vote. Previously inspected validation scores remain exploratory.

The original open-task implementation computes similarity on full Teacher
and Student response texts, then places the Teacher plus the two most
similar of four Students in the positive group. This is especially
questionable for short VQA final answers, so both full-response similarity
and the selected short answers must be logged and diagnosed, not silently
changed into a different algorithm.

The GPU 3 probe completed on commit `5f61849`: both the greedy and sampled
image requests returned two finite token logprobs for two generated tokens;
strict generation PPL was 1.2407 and 1.2692, respectively. `labels_read=false`.
The owned service exited and GPU 3 returned to 2 MiB used. The original GPU 2
queue timed out without stopping unrelated processes.

The 400-item direct-transfer train stream uses seed `20261019` after excluding
all earlier ScienceQA train inference IDs; only isolated public inputs enter
the runner. `smoke20` and `dev400` are separate immutable output namespaces.
The no-memory and online arms share identical Teacher/Tutor/Student requests
through an immutable response trace when their prompts coincide. The full arm
uses CPU Qwen3-Embedding-0.6B with last-token pooling for cosine top-30 rule
retrieval, leaving GPU 3 exclusively to Qwen2.5-VL-7B. The prompt wording and
ScienceQA answer format are task adaptations; they are logged and must not be
described as byte-identical to the original GSM8K prompt implementation.

The first 20-item **unscored** smoke on 2026-10-19 passed structural/PPL
checks (300 physical model requests, 25 rule writes, GPU 3 released), but its
I/O inspection failed the substantive reasoning-format gate: all 20 Teacher
responses were answer-only, so a rule could be invented from a letter rather
than distilled from contrasting reasoning. Five Tutor outputs were valid JSON
arrays that the initial parser discarded. Version `v2` prompts for short
reasoning before `Final answer:`, accepts either JSON object or
array rewrites, and passes only the question clause to Tutor while restoring
the original ScienceQA context verbatim. This change was made without opening
labels; the original smoke remains a diagnostic artifact and is not scored.

The v2 20-item smoke produced four distinct restored-context variants for all
20 items. An initial automated audit incorrectly required the literal prefix
`Reasoning:` even though many outputs included substantive prose before
`Final answer:` without that heading. The corrected format check counts at
least eight words and 25 characters before the final-answer marker. On the
unscored v2 stream this is 273/320 logical answering outputs (85.3%); the
remaining answer-only outputs remain visible in the casebook. No labels were
opened to make or verify this change.
