# Frozen four-arm online rule transfer, ScienceQA train

Version `transfer_online_v1_rule_only_typed_gate` answers the first causal
question: does a historical pseudo-rule help the next image question when
current-sample generation is held fixed? This is a 400-item, single-seed
development pilot, not a final benchmark. The 20-item prefix is used for an
unscored protocol smoke; its labels are opened only after all 400 items have
finished, if the full audit passes.

The common work per sample is four candidate traces on the original image:
one greedy and three temperature-0.7 samples with the train-only V2
Observed/Reasoning/Final-answer format. Majority grouping defines a
pseudo-positive and a different-answer pseudo-negative, neither certified as
correct. All arms share these exact candidates, the same bounded, answer-blind
typed visual observation calls when the lexical router supports the task, and
one constrained JSON rule-distillation call. The gate asks whether the typed
operator produces the *same* positive from stable real/fidelity facts with
blank/mismatched-image drop. A failed or unsupported gate never edits the
current answer; it only controls later memory writing.

Each arm makes a separate final refinement call with identical prompt format,
candidate proposals, decoding seed, and output cap. It sees zero or two
historical checking rules. Pseudo-answer text is stored with its source
question/option mapping; a bare option letter is never retrieved as an answer.
After all four final responses are logged, the current rule may become
visible to the next sample. The random arm writes the exact same rules as the
noisy arm and changes retrieval order only.

| Arm key | Write | Retrieve |
| --- | --- | --- |
| `none` | No | No historical rules; still pays shared distillation logically |
| `noisy` | Parsed reusable pseudo-rule whenever candidates are usable | Top-2 positive BM25 text matches, equal trust |
| `gated_weighted` | Same pseudo-rule only when typed evidence supports the same positive | Top-2 BM25 with consensus-weighted score |
| `random` | Exact `noisy` write pool | Deterministic random top-2 from past |

The retriever is transparent lexical BM25 over prior question text, with
trust-weighted scores for the gated arm. The earlier 256-dimensional hash
embedder is for tests and is not used. This rule-only experiment does **not**
test joint image/text embeddings or episodic memory. It does not reproduce the
paper's original PPL-based teacher choice because this model run requests no
logprobs; group ties use fixed trajectory order. Completion and prompt tokens
are reported separately, because equal call counts do not equal equal cost.

All inputs come from the predeclared disjoint 400-row ScienceQA train slice
`dev400_transfer.inference.jsonl`, source rank offset 1700 under seed 20261011.
Inference code opens only this label-free file and the subset manifest. The
model receives no grade/skill metadata or answer. Exact trace requests and
responses, retrieval content, per-arm writes, evidence observations, prompt
and code hashes, Git SHA, seed, revision and image hashes are retained outside
Git. Resume verifies the complete ordered prefix, code/config/Git fingerprints,
restores past banks from committed rows and replays interrupted calls from
immutable traces. A different seed, model, dataset or input hash has a
different namespace and cannot reuse a trace.

Run in server `ttt_ex` with `ttt` Python, GPU 2 only:

```sh
PRECHECK_ONLY=1 TRANSFER_COUNT=20 bash scripts/run_transfer_online.sh
TRANSFER_COUNT=20 TRANSFER_RUN_NAME=smoke20 bash scripts/run_transfer_online.sh
TRANSFER_COUNT=400 TRANSFER_RUN_NAME=dev400 bash scripts/run_transfer_online.sh
```

The shell wrapper requires a clean committed tree, checks GPU 2 occupancy,
starts its own Qwen2.5-VL server, and releases that server on exit. It opens no
label file. The 20-item smoke is checked for four-arm completion, valid
trace/response presence, ordered writes, own-bank-only past retrieval and
identical noisy/random write pools. Inspect example records and format
coverage before starting 400; do not tune on correctness.

After the full 400-arm audit passes, GPU is released. Only then run offline:

```sh
/home/lthpc/.conda/envs/ttt/bin/python evaluate/multimodal/compare_transfer.py \
  --run core/exp_res/mm/scienceqa_train/qwen25vl/transfer_online_v1_seed0_dev400 \
  --labels data/multimodal/scienceqa_modelscope_train/dev400_transfer.labels.jsonl \
  --expected-n 400 \
  --output core/exp_res/mm/scienceqa_train/qwen25vl/transfer_online_v1_seed0_dev400/comparison.json
```

The evaluator refuses to open labels before all four arms complete, then
reports accuracy, paired corrected/harmed counts and intervals/tests,
front/middle/back stream accuracy, gate/write/retrieval coverage, and separate
prompt/completion token totals. Results from one train stream remain
development evidence. Before a paper claim, add a strong independent
sampling/reflection baseline matched to both generation and total cost, three
order seeds, another VLM, and multiple datasets. A four-arm relative gain alone
does not establish a top-conference contribution.
