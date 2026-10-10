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
CUDA_VISIBLE_DEVICES=2 vllm serve /home/lthpc/private/models/Qwen2.5-VL-7B-Instruct \
  --served-model-name Qwen/Qwen2.5-VL-7B-Instruct \
  --port 9000 --trust-remote-code
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

If a model is absent, use `scripts/download_modelscope_model.py`; always record
an immutable revision instead of silently replacing an existing directory.

## Data and runs

`scripts/prepare_multimodal_dataset.py` converts a ModelScope split into local
JSONL plus PNG images. Keep these under `data/multimodal/`, which is ignored by
Git. Validate paths and label isolation before spending GPU time:

```bash
python core/mm_main.py --config core/config/mm_qwen25vl.yaml --validate-config
python core/mm_main.py --config core/config/mm_qwen25vl.yaml
```

Supported methods are `zero_shot`, `cot`, `self_consistency`,
`budget_matched`, `direct_ttcl`, `evidence_only`, and `evimem`. All evidence,
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

Every run writes `manifest.json`, `predictions.jsonl`, and `memory.jsonl` with
the Git SHA, prompt/config hashes, model names, token usage, latency, retrieval
trace, and memory write decisions. Reusing an output directory resumes the same
stream; use a new directory for every dataset, method, model, and seed.
