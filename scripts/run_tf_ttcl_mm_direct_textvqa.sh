#!/usr/bin/env bash
# Fresh TextVQA train four-arm direct transfer; no validation labels are read.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
python_bin="${PYTHON_BIN:-/home/lthpc/.conda/envs/ttt/bin/python}"
vllm_bin="${VLLM_BIN:-/home/lthpc/.conda/envs/ttt/bin/vllm}"
model_path="/home/lthpc/private/models/Qwen2.5-VL-7B-Instruct"
subset="data/multimodal/textvqa_train_tf_ttcl_mm_direct"
gpu_index="${GPU_INDEX:-3}"
api_port="${API_PORT:-9000}"
run_kind="${TF_TTCL_MM_RUN_KIND:-smoke20}"
case "$run_kind" in
  smoke20) count=20 ;;
  dev400) count=400 ;;
  *) echo "TF_TTCL_MM_RUN_KIND must be smoke20 or dev400" >&2; exit 2 ;;
esac
if [[ ! "$gpu_index" =~ ^(0|1|2|3)$ ]]; then
  echo "Unsupported GPU index: $gpu_index" >&2
  exit 2
fi
run_root="core/exp_res/mm/tf_ttcl_mm_direct/qwen25vl/textvqa_${run_kind}_seed0"
smoke_root="core/exp_res/mm/tf_ttcl_mm_direct/qwen25vl/textvqa_smoke20_seed0"
if [[ ! -f "$subset/manifest.json" ]]; then
  echo "Fresh official TextVQA train subset is absent; import it before starting a GPU service." >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "Run requires a clean, committed server worktree." >&2
  exit 2
fi
"$python_bin" - "$subset" "$count" <<'PY'
import sys
from pathlib import Path
from scripts.run_tf_ttcl_mm_direct import checked_samples
root = Path(sys.argv[1])
samples = checked_samples(root / "inference.jsonl", root / "manifest.json",
                          int(sys.argv[2]), "textvqa")
for sample in samples:
    sample.validate_images()
print(f"Precheck passed: {len(samples)} label-free fresh TextVQA train images.")
PY
if [[ "$run_kind" == dev400 ]]; then
  "$python_bin" scripts/audit_tf_ttcl_mm_direct.py --output "$smoke_root" \
    --inference "$subset/inference.jsonl" --count 20
fi
if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then
  exit 0
fi
active_pids="$(nvidia-smi -i "$gpu_index" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')"
if [[ -n "$active_pids" ]]; then
  echo "GPU $gpu_index occupied ($active_pids); no service started." >&2
  exit 2
fi
if curl --silent --max-time 2 --fail "http://127.0.0.1:${api_port}/health" >/dev/null; then
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
setsid env CUDA_VISIBLE_DEVICES="$gpu_index" VLLM_USE_FLASHINFER_SAMPLER=0 \
  "$vllm_bin" serve "$model_path" --served-model-name Qwen/Qwen2.5-VL-7B-Instruct \
  --host 127.0.0.1 --port "$api_port" --trust-remote-code \
  --gpu-memory-utilization 0.85 --max-model-len 8192 >"$run_root/vllm.log" 2>&1 &
service_pid="$!"
ready=0
for _ in $(seq 1 600); do
  if ! kill -0 "$service_pid" 2>/dev/null; then
    echo "Owned vLLM failed; see $run_root/vllm.log" >&2
    exit 3
  fi
  if curl --silent --max-time 2 --fail "http://127.0.0.1:${api_port}/health" >/dev/null; then
    ready=1
    break
  fi
  sleep 2
done
if [[ "$ready" != 1 ]]; then
  echo "Owned vLLM did not become ready within 20 minutes." >&2
  exit 4
fi
nvidia-smi -i "$gpu_index" --query-gpu=index,memory.used,utilization.gpu --format=csv >"$run_root/gpu_loaded.csv"
"$python_bin" scripts/run_tf_ttcl_mm_direct.py \
  --dataset textvqa --inference "$subset/inference.jsonl" \
  --subset-manifest "$subset/manifest.json" --output "$run_root" \
  --count "$count" --seed 0 --base-url "http://127.0.0.1:${api_port}/v1"
"$python_bin" scripts/audit_tf_ttcl_mm_direct.py --output "$run_root" \
  --inference "$subset/inference.jsonl" --count "$count" \
  --casebook "$run_root/unscored_casebook.md"
