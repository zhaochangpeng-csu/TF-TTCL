#!/usr/bin/env bash
# Full cached MathVista testmini, paired no-memory baseline on GPU 2.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
python_bin="${PYTHON_BIN:-/home/lthpc/.conda/envs/ttt/bin/python}"
vllm_bin="${VLLM_BIN:-/home/lthpc/.conda/envs/ttt/bin/vllm}"
data_root="data/multimodal/mathvista_full"
subset="${data_root}/testmini_isolated"
model_path="/home/lthpc/private/models/Qwen2.5-VL-7B-Instruct"
api_port="${API_PORT:-9000}"
run_kind="${MATHVISTA_TEXT_RUN_KIND:-smoke20}"
case "$run_kind" in
  smoke20) count=20 ;;
  full1000) count=1000 ;;
  *) echo "MATHVISTA_TEXT_RUN_KIND must be smoke20 or full1000" >&2; exit 2 ;;
esac
run_root="core/exp_res/mm/mathvista/qwen25vl/text_only_v1_seed0_${run_kind}"
smoke_root="core/exp_res/mm/mathvista/qwen25vl/text_only_v1_seed0_smoke20"

if [[ ! -f "${data_root}/manifest.json" ]]; then
  if [[ -e "$data_root" ]]; then
    echo "Partial MathVista export exists; inspect it before proceeding." >&2
    exit 2
  fi
  "$python_bin" scripts/prepare_multimodal_dataset.py \
    --dataset-id evalscope/MathVista --dataset-name mathvista \
    --split testmini --cached-only --require-image --output-dir "$data_root"
fi
"$python_bin" - "$data_root" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
manifest = json.loads((root / 'manifest.json').read_text())
rows = [json.loads(line) for line in (root / 'mathvista_testmini.jsonl').read_text().splitlines()]
if (manifest.get('dataset_id'), manifest.get('split'), manifest.get('count')) != ('evalscope/MathVista', 'testmini', 1000):
    raise ValueError('Incomplete MathVista full export')
if len(rows) != 1000 or len({str(row['id']) for row in rows}) != 1000:
    raise ValueError('MathVista full export has missing or duplicate rows')
if any(not (root / row['image']).is_file() for row in rows):
    raise ValueError('MathVista export contains missing images')
print('Full 1,000-image MathVista export ready.')
PY
if [[ ! -f "${subset}.manifest.json" ]]; then
  if [[ -e "${subset}.inference.jsonl" || -e "${subset}.labels.jsonl" ]]; then
    echo "Partial isolated MathVista subset exists; inspect it before proceeding." >&2
    exit 2
  fi
  "$python_bin" scripts/create_label_isolated_subset.py \
    --source "${data_root}/mathvista_testmini.jsonl" \
    --inference-output "${subset}.inference.jsonl" \
    --labels-output "${subset}.labels.jsonl" \
    --manifest "${subset}.manifest.json" --count 1000 --seed 20261015 --offset 0
fi
if [[ "$(git status --porcelain)" != "" ]]; then
  echo "Run requires a clean, committed server worktree." >&2
  exit 2
fi
"$python_bin" - "$subset" "$count" <<'PY'
import sys
from pathlib import Path
from scripts.run_mathvista_text_only import checked_samples
prefix = Path(sys.argv[1])
samples = checked_samples(prefix.with_suffix('.inference.jsonl'),
                          prefix.with_suffix('.manifest.json'), int(sys.argv[2]))
for sample in samples:
    sample.validate_images()
print(f'Precheck passed: {len(samples)} label-free MathVista images.')
PY
if [[ "${PRECHECK_ONLY:-0}" == 1 ]]; then
  exit 0
fi
if [[ "$run_kind" == full1000 ]]; then
  "$python_bin" - "$smoke_root" "${subset}.inference.jsonl" <<'PY'
import sys
from pathlib import Path
from evaluate.multimodal.compare_mathvista_text import audit
audit(Path(sys.argv[1]), Path(sys.argv[2]), 20)
print('Unscored MathVista smoke audit passed.')
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
"$python_bin" scripts/run_mathvista_text_only.py --inference "${subset}.inference.jsonl" \
  --subset-manifest "${subset}.manifest.json" --output "$run_root" \
  --count "$count" --seed 0 --base-url "http://127.0.0.1:${api_port}/v1"
"$python_bin" - "$run_root" "${subset}.inference.jsonl" "$count" <<'PY'
import sys
from pathlib import Path
from evaluate.multimodal.compare_mathvista_text import audit
_, rows = audit(Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]))
print(f'Paired MathVista audit passed for {len(rows)} items; labels remain unopened.')
PY
