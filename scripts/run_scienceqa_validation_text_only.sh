#!/usr/bin/env bash
# Full ScienceQA-IMG validation baseline; all inference is label-blind.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
python_bin="${PYTHON_BIN:-/home/lthpc/.conda/envs/ttt/bin/python}"
vllm_bin="${VLLM_BIN:-/home/lthpc/.conda/envs/ttt/bin/vllm}"
data_root="data/multimodal/scienceqa_img"
inference="${data_root}/validation_typed.inference.jsonl"
subset_manifest="${data_root}/validation_typed.manifest.json"
model_path="/home/lthpc/private/models/Qwen2.5-VL-7B-Instruct"
api_port="${API_PORT:-9000}"
run_kind="${TEXT_ONLY_VALIDATION_RUN_KIND:-smoke20}"
case "$run_kind" in
  smoke20) count=20 ;;
  full2097) count=2097 ;;
  *) echo "TEXT_ONLY_VALIDATION_RUN_KIND must be smoke20 or full2097" >&2; exit 2 ;;
esac
run_root="core/exp_res/mm/scienceqa_validation/qwen25vl/text_only_v2_seed0_${run_kind}"
smoke_root="core/exp_res/mm/scienceqa_validation/qwen25vl/text_only_v2_seed0_smoke20"

"$python_bin" scripts/check_typed_validation.py --manifest "$subset_manifest" --expected-n 2097
if [[ "$(git status --porcelain)" != "" ]]; then
  echo "Run requires a clean, committed server worktree." >&2
  exit 2
fi
"$python_bin" - "$inference" "$subset_manifest" "$count" <<'PY'
import sys
from pathlib import Path
from scripts.run_text_only import checked_samples
samples = checked_samples(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]))
for sample in samples:
    sample.validate_images()
print(f'Precheck passed: {len(samples)} label-free validation samples.')
PY
if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then
  exit 0
fi
if [[ "$run_kind" == full2097 ]]; then
  "$python_bin" - "$smoke_root" <<'PY'
import sys
from pathlib import Path
from evaluate.multimodal.compare_text_only import audit_text_only
audit_text_only(Path(sys.argv[1]), 20)
print('Unscored validation smoke audit passed.')
PY
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
  --gpu-memory-utilization 0.85 --max-model-len 8192 >>"$run_root/vllm.log" 2>&1 &
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
  echo "Owned vLLM did not become ready." >&2
  exit 4
fi
nvidia-smi -i 2 --query-gpu=index,memory.used,utilization.gpu --format=csv >"$run_root/gpu_loaded.csv"
"$python_bin" scripts/run_text_only.py --inference "$inference" \
  --subset-manifest "$subset_manifest" --output "$run_root" \
  --count "$count" --seed 0 --base-url "http://127.0.0.1:${api_port}/v1"
"$python_bin" - "$run_root" "$count" <<'PY'
import sys
from pathlib import Path
from evaluate.multimodal.compare_text_only import audit_text_only
_, rows = audit_text_only(Path(sys.argv[1]), int(sys.argv[2]))
print(f'Paired validation audit passed for {len(rows)} items; labels remain unopened.')
PY
