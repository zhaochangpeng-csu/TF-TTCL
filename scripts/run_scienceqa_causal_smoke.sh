#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

gpu_index="${GPU_INDEX:-2}"
api_port="${API_PORT:-9000}"
smoke_limit="${SMOKE_LIMIT:-20}"
run_tag="${RUN_TAG:-ccic_v1_seed0_smoke20}"
model_path="${MODEL_PATH:-/home/lthpc/private/models/Qwen2.5-VL-7B-Instruct}"
python_bin="${PYTHON_BIN:-/home/lthpc/.conda/envs/ttt/bin/python}"
vllm_bin="${VLLM_BIN:-/home/lthpc/.conda/envs/ttt/bin/vllm}"
config_path="core/config/mm_scienceqa_train_causal.yaml"
run_root="core/exp_res/mm/scienceqa_train/qwen25vl/${run_tag}"
budget_dir="${run_root}/causal_budget"
evidence_dir="${run_root}/causal_evidence"
service_log="${run_root}/vllm.log"
audit_path="${run_root}/protocol_audit.json"

active_pids="$({
  nvidia-smi -i "$gpu_index" --query-compute-apps=pid \
    --format=csv,noheader,nounits 2>/dev/null || true
} | sed '/^[[:space:]]*$/d')"
if [[ -n "$active_pids" ]]; then
  echo "Refusing to start: GPU ${gpu_index} already has compute PID(s): ${active_pids}" >&2
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
trap cleanup EXIT INT TERM

setsid env CUDA_VISIBLE_DEVICES="$gpu_index" VLLM_USE_FLASHINFER_SAMPLER=0 \
  "$vllm_bin" serve "$model_path" \
  --served-model-name Qwen/Qwen2.5-VL-7B-Instruct \
  --host 127.0.0.1 \
  --port "$api_port" \
  --trust-remote-code \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 >"$service_log" 2>&1 &
service_pid="$!"

ready=0
for _ in $(seq 1 120); do
  if ! kill -0 "$service_pid" 2>/dev/null; then
    echo "vLLM exited before becoming ready; inspect ${service_log}" >&2
    exit 3
  fi
  if curl --silent --fail "http://127.0.0.1:${api_port}/health" >/dev/null; then
    ready=1
    break
  fi
  sleep 2
done
if [[ "$ready" -ne 1 ]]; then
  echo "vLLM did not become ready; inspect ${service_log}" >&2
  exit 4
fi

common_overrides=(
  --override "experiment.limit=${smoke_limit}"
  --override "experiment.resume=true"
  --override "model.api_url=http://127.0.0.1:${api_port}/v1"
)

"$python_bin" core/mm_main.py --config "$config_path" \
  "${common_overrides[@]}" \
  --override "experiment.method=causal_budget" \
  --override "experiment.output_dir=${budget_dir}"

"$python_bin" core/mm_main.py --config "$config_path" \
  "${common_overrides[@]}" \
  --override "experiment.method=causal_evidence" \
  --override "experiment.output_dir=${evidence_dir}"

"$python_bin" evaluate/multimodal/audit_causal.py \
  --budget "${budget_dir}/predictions.jsonl" \
  --evidence "${evidence_dir}/predictions.jsonl" >"$audit_path"

echo "CCIC smoke completed. Label-free audit: ${audit_path}"
