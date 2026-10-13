#!/usr/bin/env bash
# Four arms in one ordered stream. Never scores during GPU inference.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
python_bin="${PYTHON_BIN:-/home/lthpc/.conda/envs/ttt/bin/python}"
vllm_bin="${VLLM_BIN:-/home/lthpc/.conda/envs/ttt/bin/vllm}"
inference="data/multimodal/scienceqa_modelscope_train/dev400_transfer.inference.jsonl"
subset_manifest="data/multimodal/scienceqa_modelscope_train/dev400_transfer.manifest.json"
labels="data/multimodal/scienceqa_modelscope_train/dev400_transfer.labels.jsonl"
count="${TRANSFER_COUNT:-20}"
run_name="${TRANSFER_RUN_NAME:-smoke20}"
run_root="core/exp_res/mm/scienceqa_train/qwen25vl/transfer_online_v1_seed0_${run_name}"
model_path="/home/lthpc/private/models/Qwen2.5-VL-7B-Instruct"
api_port="${API_PORT:-9000}"

if [[ "$count" != 20 && "$count" != 400 ]]; then
  echo "Only the frozen 20-item smoke or 400-item development stream is supported." >&2
  exit 2
fi
if [[ "$count" == 20 && "$run_name" != smoke20 ]] || [[ "$count" == 400 && "$run_name" != dev400 ]]; then
  echo "Run name/count mismatch." >&2
  exit 2
fi
if [[ "$(git status --porcelain)" != "" ]]; then
  echo "Run requires a clean, committed server worktree." >&2
  exit 2
fi
if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then
  "$python_bin" - "$inference" "$subset_manifest" "$count" <<'PY'
import sys
from pathlib import Path
from scripts.run_transfer_online import checked_samples
s = checked_samples(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]))
for x in s: x.validate_images()
print(f'Precheck passed: {len(s)} label-free ScienceQA train samples')
PY
  exit 0
fi
active_pids="$(nvidia-smi -i 2 --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')"
if [[ -n "$active_pids" ]]; then
  echo "GPU 2 occupied; refusing to affect existing compute tasks." >&2
  exit 2
fi
if curl --silent --fail "http://127.0.0.1:${api_port}/health" >/dev/null; then
  echo "Port ${api_port} already serves an unowned model." >&2
  exit 2
fi
mkdir -p "$run_root"
service_pid=""
cleanup() {
  if [[ -n "$service_pid" ]]; then
    kill -- "-${service_pid}" 2>/dev/null || true
    wait "$service_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
setsid env CUDA_VISIBLE_DEVICES=2 VLLM_USE_FLASHINFER_SAMPLER=0 \
  "$vllm_bin" serve "$model_path" --served-model-name Qwen/Qwen2.5-VL-7B-Instruct \
  --host 127.0.0.1 --port "$api_port" --trust-remote-code \
  --gpu-memory-utilization 0.85 --max-model-len 8192 >"$run_root/vllm.log" 2>&1 &
service_pid="$!"
ready=0
for _ in $(seq 1 120); do
  if ! kill -0 "$service_pid" 2>/dev/null; then
    echo "Owned vLLM failed; see $run_root/vllm.log" >&2
    exit 3
  fi
  if curl --silent --fail "http://127.0.0.1:${api_port}/health" >/dev/null; then
    ready=1
    break
  fi
  sleep 2
done
if [[ "$ready" != 1 ]]; then
  exit 4
fi
nvidia-smi -i 2 --query-gpu=index,memory.used,utilization.gpu --format=csv >"$run_root/gpu_loaded.csv"
"$python_bin" scripts/run_transfer_online.py --inference "$inference" \
  --subset-manifest "$subset_manifest" --output "$run_root" \
  --count "$count" --seed 0 --base-url "http://127.0.0.1:${api_port}/v1"
# Label-free audit happens before the owned model is released. No accuracy.
"$python_bin" - "$run_root" "$count" <<'PY'
import sys
from pathlib import Path
from evaluate.multimodal.compare_transfer import audit_four
m, rows = audit_four(Path(sys.argv[1]), int(sys.argv[2]))
print(f'Four-arm audit passed for {len(rows)} samples; labels remain unopened.')
PY
