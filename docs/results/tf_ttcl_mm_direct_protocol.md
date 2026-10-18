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
2. On an **idle** GPU 2, start an owned Qwen2.5-VL-7B service in `ttt`, send
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

At protocol creation on 2026-10-18, GPU 2 was occupied by an unrelated
Qwen3-4B vLLM service and `bokeh_eval` process. That model is not the target
Qwen2.5-VL-7B, so no model request was sent and no process was stopped.
