#!/usr/bin/env bash
# Run inside the existing Hygon container, owning and cleaning up only this service.
set -euo pipefail
cd /data/cp-experiments/tf-ttcl-hygon-20261124/project
root=/data/cp-experiments/tf-ttcl-hygon-20261124
out="$root/results/smoke20"
subset=data/multimodal/scienceqa_modelscope_train/dev400_tf_ttcl_mm_direct.inference.jsonl
reference=core/exp_res/mm/tf_ttcl_mm_direct/qwen25vl/dev400_seed0/predictions.jsonl
port=19000
if [[ -e "$out" ]]; then echo "Refusing existing output: $out" >&2; exit 2; fi
if curl --silent --max-time 2 --fail "http://127.0.0.1:$port/health" >/dev/null; then
  echo "Port $port already serving a model" >&2; exit 2
fi
mkdir -p "$out"
service_pid=''
cleanup() {
  if [[ -n "$service_pid" ]]; then
    kill -- "-$service_pid" 2>/dev/null || true
    wait "$service_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
python scripts/hygon_scienceqa_greedy.py --inference "$subset" --reference "$reference" \
  --output "$out/not_written.jsonl" --base-url "http://127.0.0.1:$port/v1" --check-only
setsid env HIP_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 \
  vllm serve "$root/modelscope_test" --served-model-name Qwen/Qwen2.5-VL-7B-Instruct \
  --host 127.0.0.1 --port "$port" --trust-remote-code --gpu-memory-utilization 0.75 \
  --max-model-len 8192 >"$out/vllm.log" 2>&1 &
service_pid=$!
ready=0
for _ in $(seq 1 600); do
  if ! kill -0 "$service_pid" 2>/dev/null; then echo "vLLM exited; inspect $out/vllm.log" >&2; exit 3; fi
  if curl --silent --max-time 2 --fail "http://127.0.0.1:$port/health" >/dev/null; then ready=1; break; fi
  sleep 2
done
if [[ "$ready" != 1 ]]; then echo "vLLM readiness timeout" >&2; exit 4; fi
python scripts/hygon_scienceqa_greedy.py --inference "$subset" --reference "$reference" \
  --output "$out/predictions.jsonl" --base-url "http://127.0.0.1:$port/v1" \
  --count 20 --probe-sampling
echo 'Smoke inference completed; labels were not read.'
