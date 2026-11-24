#!/usr/bin/env bash
set -euo pipefail
root=/data/cp-experiments/tf-ttcl-hygon-20261124
cd "$root/project"
out="$root/results/textvqa_smoke20"
inference=data/multimodal/textvqa_train_tf_ttcl_mm_direct/inference.jsonl
reference=core/exp_res/mm/tf_ttcl_mm_direct/qwen25vl/textvqa_dev400_ctx32768_seed0/predictions.jsonl
if [[ -e "$out" ]]; then echo "Refusing existing output: $out" >&2; exit 2; fi
if curl --silent --max-time 2 --fail http://127.0.0.1:19000/health >/dev/null; then
  echo 'Port 19000 already in use' >&2; exit 2
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
python scripts/hygon_textvqa_greedy.py --inference "$inference" --reference "$reference" \
  --output "$out/not_written.jsonl" --base-url http://127.0.0.1:19000/v1 --check-only
setsid env HIP_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 \
  vllm serve "$root/modelscope_test" --served-model-name Qwen/Qwen2.5-VL-7B-Instruct \
  --host 127.0.0.1 --port 19000 --trust-remote-code --gpu-memory-utilization 0.75 \
  --max-model-len 32768 >"$out/vllm.log" 2>&1 &
service_pid=$!
ready=0
for _ in $(seq 1 600); do
  if ! kill -0 "$service_pid" 2>/dev/null; then echo "vLLM exited; see $out/vllm.log" >&2; exit 3; fi
  if curl --silent --max-time 2 --fail http://127.0.0.1:19000/health >/dev/null; then ready=1; break; fi
  sleep 2
done
if [[ "$ready" != 1 ]]; then echo 'vLLM readiness timeout' >&2; exit 4; fi
python scripts/hygon_textvqa_greedy.py --inference "$inference" --reference "$reference" \
  --output "$out/predictions.jsonl" --base-url http://127.0.0.1:19000/v1 --count 20 --probe-sampling
echo 'TextVQA smoke inference completed; labels were not read.'
