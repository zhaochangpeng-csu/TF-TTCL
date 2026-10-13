#!/usr/bin/env bash
# New train inputs only; finite format test, no label reads or memory accuracy.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
python_bin="${PYTHON_BIN:-/home/lthpc/.conda/envs/ttt/bin/python}"
vllm_bin="${VLLM_BIN:-/home/lthpc/.conda/envs/ttt/bin/vllm}"
data_root="data/multimodal/scienceqa_modelscope_train"
subset="${data_root}/dev400_transfer"
service_dir="core/exp_res/mm/scienceqa_train/qwen25vl/transfer_format_v1_service"
output="core/exp_res/mm/scienceqa_train/qwen25vl/transfer_format_v1_seed0_smoke20"
model_path="/home/lthpc/private/models/Qwen2.5-VL-7B-Instruct"
api_port="${API_PORT:-9000}"

if [[ ! -f "${subset}.manifest.json" ]]; then
  "$python_bin" scripts/create_label_isolated_subset.py \
    --source "${data_root}/scienceqa_img_train_canonical.jsonl" \
    --inference-output "${subset}.inference.jsonl" --labels-output "${subset}.labels.jsonl" \
    --manifest "${subset}.manifest.json" --count 400 --seed 20261011 --offset 1700
fi
# Check IDs against previous manifest metadata, not previous labels.
"$python_bin" - "$subset" <<'PY'
import json, sys
from pathlib import Path
from scripts.inspect_baseline_io import read_rows, require_label_free
prefix = Path(sys.argv[1])
m = json.loads(prefix.with_suffix('.manifest.json').read_text())
if (m['count'], m['seed'], m['offset']) != (400, 20261011, 1700):
    raise ValueError('Wrong transfer subset')
ids = set(m['selected_ids'])
for prior in prefix.parent.glob('dev*.manifest.json'):
    if prior == prefix.with_suffix('.manifest.json'):
        continue
    other = json.loads(prior.read_text())
    if ids.intersection(other.get('selected_ids', [])):
        raise ValueError(f'Transfer stream overlaps previous subset {prior}')
rows = read_rows(prefix.with_suffix('.inference.jsonl'))
if [str(row['id']) for row in rows] != m['selected_ids']:
    raise ValueError('Transfer inference IDs changed')
for row in rows:
    require_label_free(row)
print('New disjoint 400-item train stream ready; labels not read.')
PY
if [[ "${PREPARE_ONLY:-0}" == 1 ]]; then
  exit 0
fi
if [[ -e "$output" ]]; then
  echo "Refusing to overwrite an existing format smoke: ${output}" >&2
  exit 2
fi
active_pids="$(nvidia-smi -i 2 --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')"
if [[ -n "$active_pids" ]]; then
  echo "GPU 2 occupied; no existing process will be killed." >&2
  exit 2
fi
if curl --silent --fail "http://127.0.0.1:${api_port}/health" >/dev/null; then
  echo "Port ${api_port} already serves a model; refusing to reuse an unowned server." >&2
  exit 2
fi
mkdir -p "$service_dir"
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
  --gpu-memory-utilization 0.85 --max-model-len 8192 >"${service_dir}/vllm.log" 2>&1 &
service_pid="$!"
ready=0
for _ in $(seq 1 120); do
  if ! kill -0 "$service_pid" 2>/dev/null; then
    echo "Owned vLLM failed to start; see ${service_dir}/vllm.log" >&2
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
nvidia-smi -i 2 --query-gpu=index,memory.used,utilization.gpu --format=csv >"${service_dir}/gpu_loaded.csv"
"$python_bin" scripts/run_transfer_format_smoke.py --inference "${subset}.inference.jsonl" \
  --output "$output" --base-url "http://127.0.0.1:${api_port}/v1" --count 20 --seed 0
