# EviMem-TTCL experiment guide

EviMem-TTCL is implemented beside the original text pipeline. The original
`core/main.py` entry point is unchanged; multimodal runs use `core/mm_main.py`.

## Safety and causal protocol

- `MultimodalSample` has no label field. Local adapters put labels in a separate
  `EvaluationTarget`, and prediction JSONL never contains labels.
- Memory retrieval occurs before answering sample `i`; only accepted experience
  from indices `< i` is eligible. Memory is written after the prediction record
  has been decided.
- `hash` embeddings are deterministic pipeline scaffolding only. Do not report
  them as research results. Paper runs use a separately served
  Qwen3-VL-Embedding model through `memory.embedder: vllm_multimodal`.
- Image-preserving augmentation only re-encodes pixels. It never crops, flips,
  rotates, or changes brightness, which is especially important for OCR tasks.

## Server setup

Use the existing environment and only the assigned GPUs:

```bash
conda activate ttt
CUDA_VISIBLE_DEVICES=2 VLLM_USE_FLASHINFER_SAMPLER=0 \
  vllm serve /home/lthpc/private/models/Qwen2.5-VL-7B-Instruct \
  --served-model-name Qwen/Qwen2.5-VL-7B-Instruct \
  --host 127.0.0.1 --port 9000 --trust-remote-code \
  --gpu-memory-utilization 0.85 --max-model-len 8192
```

When GPU 2 has at least 12 GB free, serve Qwen3-VL-Embedding alongside the generator:

```bash
CUDA_VISIBLE_DEVICES=2 VLLM_USE_FLASHINFER_SAMPLER=0 \
  vllm serve /home/lthpc/private/models/Qwen3-VL-Embedding-2B \
  --served-model-name Qwen/Qwen3-VL-Embedding-2B \
  --runner pooling --port 9001 --trust-remote-code \
  --gpu-memory-utilization 0.14 --max-model-len 8192
```

This low-memory pooling service can coexist with the 7B generator on the
72&nbsp;GB GPU 2 used during development. Do not assume that ratio is portable:
check free memory first, and never displace an existing GPU process. GPU 0 is an
alternative only after its current job exits naturally.

The downloaded cross-architecture generators are served one at a time on the
same port. Stop the current port-9000 service before switching models; do not
start competing generators on GPU 2:

```bash
CUDA_VISIBLE_DEVICES=2 VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve \
  /home/lthpc/private/models/Qwen3-VL-8B-Instruct \
  --served-model-name Qwen/Qwen3-VL-8B-Instruct \
  --port 9000 --trust-remote-code --gpu-memory-utilization 0.85 --max-model-len 8192

CUDA_VISIBLE_DEVICES=2 VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve \
  /home/lthpc/private/models/InternVL3_5-8B-HF \
  --served-model-name OpenGVLab/InternVL3.5-8B-HF \
  --port 9000 --trust-remote-code --gpu-memory-utilization 0.85 --max-model-len 8192
```

Both alternate backends passed a two-sample, label-blind MathVista interface
smoke test on 2026-10-10. Qwen3-VL wrote two complete predictions under
`qwen3vl/zero_shot/seed0_smoke`; InternVL3.5 did the same under
`internvl35/zero_shot/seed0_smoke`. These checks establish serving and message
compatibility only and are not benchmark results.

If a model is absent, use `scripts/download_modelscope_model.py`; always record
an immutable revision instead of silently replacing an existing directory.
When a provider only records a mutable revision such as `master`, also record
the SHA-256 of the served weight artifact in the run configuration.  The
ScienceQA development config pins the Qwen2.5-VL Hub commit and fingerprints
the locally served Qwen3-VL-Embedding weights this way.

## Data and runs

`scripts/prepare_multimodal_dataset.py` converts a ModelScope split into local
JSONL plus PNG images. Keep these under `data/multimodal/`, which is ignored by
Git. Validate paths and label isolation before spending GPU time:

For ScienceQA, prefer the official `lupantech/ScienceQA` `problems.json` and
split image archives. Convert an extracted split with
`scripts/prepare_scienceqa.py`; it keeps the optional textual hint, excludes
lecture/solution fields, and can restrict the stream to image-bearing samples.

```bash
python core/mm_main.py --config core/config/mm_qwen25vl.yaml --validate-config
python core/mm_main.py --config core/config/mm_qwen25vl.yaml
```

Supported methods are `zero_shot`, `cot`, `self_consistency`,
`budget_matched`, `contrastive_budget`, `contrastive_evidence`, `direct_ttcl`,
`evidence_only`, and `evimem`. `contrastive_evidence` invokes a position-swapped
two-view adjudicator only when candidate answers disagree; it falls back to
majority voting unless both views select the same evidence-supported answer and
at least one view explicitly contradicts the alternative. `contrastive_budget`
runs the identical calls but discards the adjudicator decision. All evidence,
augmentation, memory, and retrieval-modality ablations are config switches.
The full method only writes experiences whose answer verdict and every claim
are `entailed`. RRF retrieval is also relevance-gated: rules require calibrated
text/joint cosine similarity of at least 0.50 and raw episodes require joint
similarity of at least 0.75. These fixed thresholds separate paraphrases from
wrong-image pairs in the label-free synthetic fixture; they must not be tuned on
benchmark answers.

Offline evaluation reads labels separately:

```bash
python evaluate/multimodal/evaluate.py \
  --predictions core/exp_res/mm/.../predictions.jsonl \
  --labels data/multimodal/scienceqa_img_test.jsonl \
  --dataset scienceqa_img --image-root data/multimodal/images
```

Compare two runs only after inference has finished. The comparator requires
identical sample IDs and reports the paired score delta, bootstrap confidence
interval, per-sample wins/losses, and a paired randomization p-value:

```bash
python evaluate/multimodal/compare.py \
  --baseline core/exp_res/mm/.../zero_shot/.../predictions.jsonl \
  --candidate core/exp_res/mm/.../evimem/.../predictions.jsonl \
  --labels data/multimodal/mathvista/mathvista_testmini.jsonl \
  --dataset mathvista --image-root data/multimodal/mathvista
```

Before changing evidence weights, separate candidate-generation recall from
selection quality:

```bash
python evaluate/multimodal/candidate_oracle.py \
  --predictions core/exp_res/mm/.../predictions.jsonl \
  --labels data/multimodal/mathvista/mathvista_testmini.jsonl \
  --dataset mathvista --image-root data/multimodal/mathvista
```

Every run writes `manifest.json`, `predictions.jsonl`, and `memory.jsonl` with
the Git SHA, prompt/config hashes, model names, token usage, latency, retrieval
trace, and memory write decisions. Reusing an output directory resumes the same
stream; use a new directory for every dataset, method, model, and seed. The
operational `experiment.resume` toggle is excluded from the scientific config
hash, so a run started with `resume: false` can be safely continued with
`resume: true`; every other configuration field remains compatibility-checked.
