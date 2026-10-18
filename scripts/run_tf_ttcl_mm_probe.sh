#!/usr/bin/env bash
# One label-free multimodal logprob probe; never touches an occupied GPU.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
python_bin="${PYTHON_BIN:-/home/lthpc/.conda/envs/ttt/bin/python}"
vllm_bin="${VLLM_BIN:-/home/lthpc/.conda/envs/ttt/bin/vllm}"
model_path="/home/lthpc/private/models/Qwen2.5-VL-7B-Instruct"
inference="data/multimodal/scienceqa_modelscope_train/dev400_text_only.inference.jsonl"
image_root="data/multimodal/scienceqa_modelscope_train"
run_root="core/exp_res/mm/tf_ttcl_mm_direct/qwen25vl"
port="${API_PORT:-9000}"
if [[ "$(git status --porcelain)" != "" ]]; then
  echo "Probe requires a clean, committed server worktree." >&2
  exit 2
fi
if [[ ! -f "$inference" || ! -d "$model_path" ]]; then
  echo "Missing isolated input or target model; no service started." >&2
  exit 2
fi
active_pids="$(nvidia-smi -i 2 --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')"
if [[ -n "$active_pids" ]]; then
  echo "GPU 2 occupied ($active_pids); no service started." >&2
  exit 2
fi
if curl --silent --fail "http://127.0.0.1:${port}/health" >/dev/null; then
  echo "Port ${port} serves an unowned model; no service started." >&2
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
  --host 127.0.0.1 --port "$port" --trust-remote-code \
  --gpu-memory-utilization 0.85 --max-model-len 8192 >>"$run_root/probe_vllm.log" 2>&1 &
service_pid="$!"
ready=0
for _ in $(seq 1 600); do
  if ! kill -0 "$service_pid" 2>/dev/null; then
    echo "Owned vLLM failed; see $run_root/probe_vllm.log" >&2
    exit 3
  fi
  if curl --silent --fail "http://127.0.0.1:${port}/health" >/dev/null; then
    ready=1
    break
  fi
  sleep 2
done
if [[ "$ready" != 1 ]]; then
  echo "Owned vLLM did not become ready within 20 minutes." >&2
  exit 4
fi
"$python_bin" scripts/probe_tf_ttcl_mm_logprobs.py --inference "$inference" \
  --image-root "$image_root" --base-url "http://127.0.0.1:${port}/v1" \
  --output "$run_root/logprob_probe.json"
