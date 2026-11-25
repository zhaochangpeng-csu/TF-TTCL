#!/usr/bin/env bash
# Inside cp-experiments: two independent ordered streams on HCU 0 and 1.
set -euo pipefail
root=/data/cp-experiments/tf-ttcl-hygon-20261124
project="$root/textvqa_format_project_20261126"
inputs="$root/textvqa_format_fresh_inputs/data/multimodal"
data="$inputs/textvqa_fresh_20261126"
count="${FORMAT_COUNT:-20}"
if [[ "$count" != 20 && "$count" != 400 ]]; then echo 'Expected 20 or 400' >&2; exit 2; fi
out="$root/results/textvqa_format_factorial_${count}_seed0"
manifest_sha=c8916fe1b1aea8650056086be1d868b11b1add52b93a598cc56851b37c431eaf
base_prompt=8b48f2132cfe999b2610f2b68424f54c39c3a4ffe4208838ba81a7babfca5193
short_prompt=6e48938a0cfca4d959d65bb4c476f0a5d43b930c8b50e903ac070bd9ee1268d3
cd "$project"
if [[ -e "$out" ]]; then echo "Refusing existing output: $out" >&2; exit 2; fi
python -c 'import re,subprocess; s=subprocess.check_output(["/opt/hyhal/bin/hy-smi","--showmemuse","--showuse"],text=True); pairs=re.findall(r"HCU\[(\d+)\]\s*:\s*(HCU use \(%\)|HCU memory use \(%\)):\s*([0-9.]+)",s); d={(int(i),k):float(v) for i,k,v in pairs}; assert all(d.get((i,k)) == 0 for i in (0,1) for k in ("HCU use (%)","HCU memory use (%)")), "HCU 0/1 must be idle"'
python -c 'from pathlib import Path; from scripts.run_hygon_scienceqa_parallel import verify_weights; import sys; verify_weights(Path(sys.argv[1]),Path(sys.argv[2]))' \
  "$root/modelscope_test" "$root/weight_hash_verified.json"
if [[ "$count" == 400 ]]; then
  for policy in original short_final; do
    python scripts/audit_textvqa_format_stream.py --output "$root/results/textvqa_format_factorial_20_seed0/$policy" \
      --inference "$data/inference.jsonl" --policy "$policy" --count 20
  done
fi
for port in 19600 19601; do
  python -c 'import socket,sys; s=socket.socket(); s.bind(("127.0.0.1",int(sys.argv[1]))); s.close()' "$port"
done
common=(--inference "$data/inference.jsonl" --subset-manifest "$data/manifest.json"
        --manifest-sha256 "$manifest_sha" --output "$out" --embedding-path "$root/embedding"
        --excluded-inference "$inputs/textvqa_train_tf_ttcl_mm_direct/inference.jsonl"
        --excluded-inference "$inputs/textvqa_val/val_isolated.inference.jsonl" --count "$count")
for policy in original short_final; do
  python scripts/run_hygon_textvqa_format_factorial.py "${common[@]}" --policy "$policy" \
    --stack-id hygon_dtk2604_vllm0210_torch210_transformers550_ctx32768_seed0 --check-only
done
mkdir -p "$out"
services=()
workers=()
cleanup() {
  for pid in "${workers[@]}" "${services[@]}"; do kill -- "-$pid" 2>/dev/null || true; done
  for _ in $(seq 1 20); do
    active=0
    for pid in "${workers[@]}" "${services[@]}"; do if kill -0 -- "-$pid" 2>/dev/null; then active=1; fi; done
    if [[ "$active" == 0 ]]; then break; fi
    sleep 1
  done
  for pid in "${workers[@]}" "${services[@]}"; do kill -KILL -- "-$pid" 2>/dev/null || true; done
  for pid in "${workers[@]}" "${services[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for device in 0 1; do
  port=$((19600 + device))
  setsid env HIP_VISIBLE_DEVICES="$device" VLLM_USE_FLASHINFER_SAMPLER=0 \
    vllm serve "$root/modelscope_test" --served-model-name Qwen/Qwen2.5-VL-7B-Instruct \
    --host 127.0.0.1 --port "$port" --trust-remote-code --gpu-memory-utilization 0.75 \
    --max-model-len 32768 >"$out/vllm-${device}.log" 2>&1 &
  services+=("$!")
done
ready=0
for _ in $(seq 1 600); do
  for pid in "${services[@]}"; do if ! kill -0 "$pid" 2>/dev/null; then echo "Service $pid exited" >&2; exit 3; fi; done
  if curl --silent --max-time 2 --fail http://127.0.0.1:19600/health >/dev/null && \
     curl --silent --max-time 2 --fail http://127.0.0.1:19601/health >/dev/null; then ready=1; break; fi
  sleep 2
done
if [[ "$ready" != 1 ]]; then echo 'Service readiness timeout' >&2; exit 4; fi
for device in 0 1; do
  policy=original
  if [[ "$device" == 1 ]]; then policy=short_final; fi
  port=$((19600 + device))
  setsid env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python scripts/run_hygon_textvqa_format_factorial.py \
    "${common[@]}" --policy "$policy" --base-url "http://127.0.0.1:${port}/v1" \
    --stack-id hygon_dtk2604_vllm0210_torch210_transformers550_ctx32768_seed0 \
    >"$out/worker-${policy}.log" 2>&1 &
  workers+=("$!")
done
for pid in "${workers[@]}"; do
  if ! wait "$pid"; then echo "Worker $pid failed" >&2; exit 5; fi
done
workers=()
cleanup
services=()
for policy in original short_final; do
  python scripts/audit_textvqa_format_stream.py --output "$out/$policy" --inference "$data/inference.jsonl" \
    --policy "$policy" --count "$count" --casebook "$out/$policy/unscored_casebook.md"
done
if [[ "$count" == 400 ]]; then
  python scripts/compare_textvqa_format_memory.py --baseline "$out/original" --strengthened "$out/short_final" \
    --inference "$data/inference.jsonl" --subset-manifest "$data/manifest.json" --labels "$data/labels.jsonl" \
    --baseline-prompt-sha256 "$base_prompt" --strengthened-prompt-sha256 "$short_prompt" \
    --output "$out/comparison.json"
fi
