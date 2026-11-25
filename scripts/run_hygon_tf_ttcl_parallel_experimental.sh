#!/usr/bin/env bash
# Inside cp-experiments, from an ISOLATED project copy; owns HCU 1–7 only.
# Twenty-item label-free experiment. This is not the sequential baseline.
set -euo pipefail
root=/data/cp-experiments/tf-ttcl-hygon-20261124
project="${HYGON_EXPERIMENT_PROJECT:?Set HYGON_EXPERIMENT_PROJECT to an isolated project copy}"
project="$(realpath "$project")"
original="$(realpath "$root/project")"
case "$project/" in "$original/"*) echo 'Experimental source must be outside the running project.' >&2; exit 2 ;; esac
data="$original/data/multimodal/scienceqa_modelscope_train"
inference="$data/dev400_tf_ttcl_mm_direct.inference.jsonl"
subset="$data/dev400_tf_ttcl_mm_direct.manifest.json"
reference="$original/core/exp_res/mm/tf_ttcl_mm_direct/qwen25vl/dev400_seed0/predictions.jsonl"
out="$root/results/online_parallel_experimental_20_seed0"
if [[ -e "$out" ]]; then echo "Output already exists: $out" >&2; exit 2; fi
cd "$project"
ports=(19501 19502 19503 19504 19505 19506 19507)
urls=()
for port in "${ports[@]}"; do urls+=(--base-url "http://127.0.0.1:${port}/v1"); done
stack_id='experimental_hygon_dtk2604_vllm0210_hcu1-7_ctx8192_seed0'
args=(--inference "$inference" --reference "$reference" --subset-manifest "$subset"
      --output "$out" --embedding-path "$root/embedding" --stack-id "$stack_id" "${urls[@]}")
python scripts/run_hygon_tf_ttcl_parallel_experimental.py "${args[@]}" --check-only --check-devices
python -c 'from pathlib import Path; from scripts.run_hygon_scienceqa_parallel import verify_weights, verify_smoke, free_port; import sys; verify_weights(Path(sys.argv[1]),Path(sys.argv[2])); verify_smoke(Path(sys.argv[3]),Path(sys.argv[4])); [free_port(int(port)) for port in sys.argv[5:]]' \
  "$root/modelscope_test" "$root/weight_hash_verified.json" "$root/results/smoke20/predictions.jsonl" "$reference" "${ports[@]}"
mkdir "$out"
pids=()
cleanup() {
  for pid in "${pids[@]}"; do kill -- "-$pid" 2>/dev/null || true; done
  for _ in $(seq 1 20); do
    alive=0
    for pid in "${pids[@]}"; do if kill -0 -- "-$pid" 2>/dev/null; then alive=1; fi; done
    if [[ "$alive" == 0 ]]; then break; fi
    sleep 1
  done
  for pid in "${pids[@]}"; do kill -KILL -- "-$pid" 2>/dev/null || true; done
  for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for device in 1 2 3 4 5 6 7; do
  port=$((19500 + device))
  setsid env HIP_VISIBLE_DEVICES="$device" VLLM_USE_FLASHINFER_SAMPLER=0 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_CACHE_ROOT="$out/vllm-cache-$device" \
    TRITON_CACHE_DIR="$out/triton-cache-$device" TORCHINDUCTOR_CACHE_DIR="$out/torchinductor-$device" \
    vllm serve "$root/modelscope_test" --served-model-name Qwen/Qwen2.5-VL-7B-Instruct \
    --host 127.0.0.1 --port "$port" --trust-remote-code --gpu-memory-utilization 0.75 \
    --max-model-len 8192 >"$out/vllm-$device.log" 2>&1 &
  pids+=("$!")
done
ready=0
for _ in $(seq 1 600); do
  for pid in "${pids[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then echo "Owned service $pid exited" >&2; exit 3; fi
  done
  ready=1
  for port in "${ports[@]}"; do
    if ! curl --silent --max-time 2 --fail "http://127.0.0.1:${port}/health" >/dev/null; then
      ready=0
    fi
  done
  if [[ "$ready" == 1 ]]; then break; fi
  sleep 2
done
if [[ "$ready" != 1 ]]; then echo 'Seven-service readiness timeout' >&2; exit 4; fi
python scripts/run_hygon_tf_ttcl_parallel_experimental.py "${args[@]}"
cleanup
pids=()
python scripts/audit_hygon_tf_ttcl_direct.py --output "$out" --inference "$inference" \
  --reference "$reference" --count 20 --casebook "$out/unscored_casebook.md"
echo 'Experimental parallel 20-item smoke completed and audited; labels were not read.'
