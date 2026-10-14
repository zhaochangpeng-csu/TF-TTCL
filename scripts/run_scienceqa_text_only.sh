#!/usr/bin/env bash
# Paired no-memory text-augmentation baseline on GPU 2; no labels in inference.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
python_bin="${PYTHON_BIN:-/home/lthpc/.conda/envs/ttt/bin/python}"
vllm_bin="${VLLM_BIN:-/home/lthpc/.conda/envs/ttt/bin/vllm}"
data_root="data/multimodal/scienceqa_modelscope_train"
subset="${data_root}/dev400_text_only"
model_path="/home/lthpc/private/models/Qwen2.5-VL-7B-Instruct"
api_port="${API_PORT:-9000}"
run_kind="${TEXT_ONLY_RUN_KIND:-smoke20}"
case "$run_kind" in
  smoke20) count=20 ;;
  dev400) count=400 ;;
  *) echo "TEXT_ONLY_RUN_KIND must be smoke20 or dev400" >&2; exit 2 ;;
esac
run_root="core/exp_res/mm/scienceqa_train/qwen25vl/text_only_v1_seed0_${run_kind}"
smoke_root="core/exp_res/mm/scienceqa_train/qwen25vl/text_only_v1_seed0_smoke20"

if [[ ! -f "${subset}.manifest.json" ]]; then
  "$python_bin" scripts/create_label_isolated_subset.py \
    --source "${data_root}/scienceqa_img_train_canonical.jsonl" \
    --inference-output "${subset}.inference.jsonl" \
    --labels-output "${subset}.labels.jsonl" \
    --manifest "${subset}.manifest.json" --count 400 --seed 20261011 --offset 2100
fi
"$python_bin" - "$subset" <<'PY'
import json, sys
from pathlib import Path
from scripts.inspect_baseline_io import read_rows, require_label_free
prefix = Path(sys.argv[1])
manifest = json.loads(prefix.with_suffix('.manifest.json').read_text())
if (manifest['count'], manifest['seed'], manifest['offset']) != (400, 20261011, 2100):
    raise ValueError('Wrong text-only subset')
ids = set(manifest['selected_ids'])
for prior in prefix.parent.glob('dev*.manifest.json'):
    if prior == prefix.with_suffix('.manifest.json'):
        continue
    if ids.intersection(json.loads(prior.read_text()).get('selected_ids', [])):
        raise ValueError(f'Text-only stream overlaps {prior}')
rows = read_rows(prefix.with_suffix('.inference.jsonl'))
if [str(row['id']) for row in rows] != manifest['selected_ids']:
    raise ValueError('Text-only inference IDs changed')
for row in rows:
    require_label_free(row)
print('Disjoint label-free 400-item text-only stream ready.')
PY
if [[ "${PREPARE_ONLY:-0}" == 1 ]]; then
  exit 0
fi
if [[ "$(git status --porcelain)" != "" ]]; then
  echo "Run requires a clean, committed server worktree." >&2
  exit 2
fi
"$python_bin" - "$subset" "$count" <<'PY'
import sys
from pathlib import Path
from scripts.run_text_only import checked_samples
samples = checked_samples(Path(sys.argv[1]).with_suffix('.inference.jsonl'),
                          Path(sys.argv[1]).with_suffix('.manifest.json'), int(sys.argv[2]))
for sample in samples:
    sample.validate_images()
print(f'Precheck passed: {len(samples)} label-free images.')
PY
if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then
  exit 0
fi
if [[ "$run_kind" == dev400 ]]; then
  "$python_bin" - "$smoke_root" <<'PY'
import sys
from pathlib import Path
from evaluate.multimodal.compare_text_only import audit_text_only
audit_text_only(Path(sys.argv[1]), 20)
print('Unscored 20-item smoke audit passed.')
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
"$python_bin" scripts/run_text_only.py --inference "${subset}.inference.jsonl" \
  --subset-manifest "${subset}.manifest.json" --output "$run_root" \
  --count "$count" --seed 0 --base-url "http://127.0.0.1:${api_port}/v1"
"$python_bin" - "$run_root" "$count" <<'PY'
import sys
from pathlib import Path
from evaluate.multimodal.compare_text_only import audit_text_only
_, rows = audit_text_only(Path(sys.argv[1]), int(sys.argv[2]))
print(f'Paired text-only audit passed for {len(rows)} items; labels remain unopened.')
PY
