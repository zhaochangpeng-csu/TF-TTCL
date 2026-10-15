#!/usr/bin/env bash
# Label-blind paired TextVQA image-only / joint baseline on an otherwise idle GPU 2.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
python_bin="${PYTHON_BIN:-/home/lthpc/.conda/envs/ttt/bin/python}"
vllm_bin="${VLLM_BIN:-/home/lthpc/.conda/envs/ttt/bin/vllm}"
data_root="data/multimodal/textvqa_val"
subset="${data_root}/val_isolated"
baseline="core/exp_res/mm/textvqa/qwen25vl/text_only_v1_seed0_full5000"
model_path="/home/lthpc/private/models/Qwen2.5-VL-7B-Instruct"
api_port="${API_PORT:-9000}"
run_kind="${TEXTVQA_IMAGE_RUN_KIND:-smoke20}"
case "$run_kind" in
  smoke20) count=20 ;;
  full5000) count=5000 ;;
  *) echo "TEXTVQA_IMAGE_RUN_KIND must be smoke20 or full5000" >&2; exit 2 ;;
esac
run_root="core/exp_res/mm/textvqa/qwen25vl/image_joint_v1_seed0_${run_kind}"
smoke_root="core/exp_res/mm/textvqa/qwen25vl/image_joint_v1_seed0_smoke20"
if [[ "$(git status --porcelain)" != "" ]]; then
  echo "Run requires a clean, committed server worktree." >&2
  exit 2
fi
"$python_bin" - "$baseline" "${subset}.inference.jsonl" "${subset}.manifest.json" "$count" <<'PY'
import sys
from pathlib import Path
from evaluate.multimodal.compare_textvqa_text import audit
from scripts.run_textvqa_text_only import checked_samples
baseline, inference, selection, count = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4])
samples = checked_samples(inference, selection, count)
for sample in samples:
    sample.validate_images()
manifest, rows = audit(baseline, inference, 5000)
if [sample.sample_id for sample in samples] != [row['sample_id'] for row in rows[:count]]:
    raise ValueError('Frozen baseline sample order changed')
print(f'Precheck passed: {count} label-free images and complete frozen baseline.')
PY
if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then
  exit 0
fi
if [[ "$run_kind" == full5000 ]]; then
  "$python_bin" - "$smoke_root" "$baseline" "${subset}.inference.jsonl" <<'PY'
import sys
from pathlib import Path
from evaluate.multimodal.compare_textvqa_image_joint import audit
audit(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), 20)
print('Unscored image/joint smoke audit passed.')
PY
fi
active_pids="$(nvidia-smi -i 2 --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')"
if [[ -n "$active_pids" ]]; then
  echo "GPU 2 occupied; refusing to affect existing compute tasks: $active_pids" >&2
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
  --gpu-memory-utilization 0.85 --max-model-len 32768 >>"$run_root/vllm.log" 2>&1 &
service_pid="$!"
ready=0
for _ in $(seq 1 600); do
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
  echo "Owned vLLM did not become ready within 20 minutes." >&2
  exit 4
fi
nvidia-smi -i 2 --query-gpu=index,memory.used,utilization.gpu --format=csv >"$run_root/gpu_loaded.csv"
"$python_bin" scripts/run_textvqa_image_joint.py \
  --inference "${subset}.inference.jsonl" --subset-manifest "${subset}.manifest.json" \
  --baseline "$baseline" --output "$run_root" --count "$count" --seed 0 \
  --base-url "http://127.0.0.1:${api_port}/v1"
"$python_bin" - "$run_root" "$baseline" "${subset}.inference.jsonl" "$count" <<'PY'
import sys
from pathlib import Path
from evaluate.multimodal.compare_textvqa_image_joint import audit
_, rows, _ = audit(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4]))
print(f'Paired image/joint audit passed for {len(rows)} items; labels remain unopened.')
PY
