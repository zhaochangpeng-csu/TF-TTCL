#!/usr/bin/env bash
# Run one ordered online memory stream inside cp-experiments.
set -euo pipefail
root=/data/cp-experiments/tf-ttcl-hygon-20261124
project="$root/project"
data="$project/data/multimodal/scienceqa_modelscope_train"
inference="$data/dev400_tf_ttcl_mm_direct.inference.jsonl"
subset="$data/dev400_tf_ttcl_mm_direct.manifest.json"
reference="$project/core/exp_res/mm/tf_ttcl_mm_direct/qwen25vl/dev400_seed0/predictions.jsonl"
count="${TF_TTCL_COUNT:-20}"
device="${HYGON_DEVICE:-0}"
port="${HYGON_PORT:-19400}"
if [[ "$count" != 20 && "$count" != 400 ]]; then echo 'Expected count 20 or 400' >&2; exit 2; fi
if [[ ! "$device" =~ ^[0-7]$ ]]; then echo 'Invalid HCU device' >&2; exit 2; fi
out="$root/results/online_scienceqa_${count}_seed0"
cd "$project"
python -c 'from pathlib import Path; from scripts.run_hygon_scienceqa_parallel import verify_weights,verify_smoke; import sys; verify_weights(Path(sys.argv[1]),Path(sys.argv[2])); verify_smoke(Path(sys.argv[3]),Path(sys.argv[4]))' \
  "$root/modelscope_test" "$root/weight_hash_verified.json" "$root/results/smoke20/predictions.jsonl" "$reference"
if [[ "$count" == 400 ]]; then
  python scripts/audit_hygon_tf_ttcl_direct.py --output "$root/results/online_scienceqa_20_seed0" \
    --inference "$inference" --reference "$reference" --count 20
fi
stack_id="hygon_dtk2604_vllm0210_torch210_transformers550_device${device}_ctx8192_bf16"
args=(--inference "$inference" --reference "$reference" --subset-manifest "$subset" \
      --output "$out" --embedding-path "$root/embedding" --count "$count" \
      --base-url "http://127.0.0.1:${port}/v1" --stack-id "$stack_id")
python scripts/run_hygon_tf_ttcl_direct.py "${args[@]}" --check-only
if curl --silent --max-time 2 --fail "http://127.0.0.1:${port}/health" >/dev/null; then
  echo "Port $port already serves a model" >&2; exit 2
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
setsid env HIP_VISIBLE_DEVICES="$device" VLLM_USE_FLASHINFER_SAMPLER=0 \
  vllm serve "$root/modelscope_test" --served-model-name Qwen/Qwen2.5-VL-7B-Instruct \
  --host 127.0.0.1 --port "$port" --trust-remote-code --gpu-memory-utilization 0.75 \
  --max-model-len 8192 >"$out/vllm.log" 2>&1 &
service_pid=$!
ready=0
for _ in $(seq 1 600); do
  if ! kill -0 "$service_pid" 2>/dev/null; then echo "vLLM exited; inspect $out/vllm.log" >&2; exit 3; fi
  if curl --silent --max-time 2 --fail "http://127.0.0.1:${port}/health" >/dev/null; then ready=1; break; fi
  sleep 2
done
if [[ "$ready" != 1 ]]; then echo 'vLLM readiness timeout' >&2; exit 4; fi
python scripts/run_hygon_tf_ttcl_direct.py "${args[@]}"
if [[ ! -e "$out/unscored_casebook.md" ]]; then
  python scripts/audit_hygon_tf_ttcl_direct.py --output "$out" --inference "$inference" \
    --reference "$reference" --count "$count" --casebook "$out/unscored_casebook.md"
else
  python scripts/audit_hygon_tf_ttcl_direct.py --output "$out" --inference "$inference" \
    --reference "$reference" --count "$count"
fi
if [[ "$count" == 400 && ! -e "$out/comparison.json" ]]; then
  python scripts/score_hygon_tf_ttcl_direct.py --run "$out" --inference "$inference" \
    --reference "$reference" --labels "$data/dev400_tf_ttcl_mm_direct.labels.jsonl" --output "$out/comparison.json"
fi
