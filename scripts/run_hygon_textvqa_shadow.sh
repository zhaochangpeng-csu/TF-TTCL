#!/usr/bin/env bash
# Inside the existing container. No labels read. Selects idle HCU devices only.
# Required immutable anchor SHA values: ANCHOR_PRED_SHA and ANCHOR_MANIFEST_SHA.
# Usage: SHADOW_COUNT=20 ... bash scripts/run_hygon_textvqa_shadow.sh
# After an audited smoke: SHADOW_COUNT=400 ... bash scripts/run_hygon_textvqa_shadow.sh
set -euo pipefail
root="${SHADOW_ROOT:-/data/cp-experiments/tf-ttcl-hygon-20261124}"
project="${SHADOW_PROJECT:?Set SHADOW_PROJECT to the frozen deployed source directory}"
anchor="${SHADOW_ANCHOR:-$root/results/textvqa_format_factorial_v2_400_seed0/short_final}"
inference="${SHADOW_INFERENCE:-$root/textvqa_format_fresh_inputs/data/multimodal/textvqa_fresh_20261126/inference.jsonl}"
model="${SHADOW_MODEL:-$root/modelscope_test}"
embedding="${SHADOW_EMBEDDING:-$root/embedding}"
weights="${SHADOW_WEIGHT_MANIFEST:-$root/weight_hash_verified.json}"
evaluator="${SHADOW_MMF_EVALUATOR:-$root/textvqa_rescore_project_280c1a7/m4c_evaluators.py}"
count="${SHADOW_COUNT:-20}"
: "${ANCHOR_PRED_SHA:?Set frozen anchor prediction SHA-256}"
: "${ANCHOR_MANIFEST_SHA:?Set frozen anchor manifest SHA-256}"
: "${SHADOW_STACK_ID:?Set a stable software/model-service stack ID}"
[[ "$count" == 20 || "$count" == 400 ]] || { echo 'SHADOW_COUNT must be 20 or 400' >&2; exit 2; }
[[ "$ANCHOR_PRED_SHA" =~ ^[0-9a-f]{64}$ && "$ANCHOR_MANIFEST_SHA" =~ ^[0-9a-f]{64}$ ]] || { echo 'Malformed anchor SHA' >&2; exit 2; }
cd "$project"
out="${SHADOW_OUTPUT_ROOT:-$root/results/textvqa_shadow_fixed_history_v1}_${count}"
smoke="${SHADOW_OUTPUT_ROOT:-$root/results/textvqa_shadow_fixed_history_v1}_20"
port_base=20000
if [[ "$count" == 400 ]]; then port_base=20100; fi
if [[ -e "$out" && "${SHADOW_RESUME:-0}" != 1 ]]; then echo "Output exists: $out (set SHADOW_RESUME=1 only for same configuration)" >&2; exit 2; fi

devices="${SHADOW_DEVICES:-auto}"
if [[ "$devices" == auto ]]; then
  devices=$(python - "${SHADOW_GPU_COUNT:-2}" <<'PY'
import re, subprocess, sys
n = int(sys.argv[1])
if not 1 <= n <= 8: raise SystemExit('Need 1 to 8 cards')
s = subprocess.check_output(['/opt/hyhal/bin/hy-smi','--showmemuse','--showuse'], text=True)
pairs = re.findall(r'HCU\[(\d+)\]\s*:\s*(HCU use \(%\)|HCU memory use \(%\)):\s*([0-9.]+)', s)
d = {(int(i), key): float(v) for i, key, v in pairs}
ids = [i for i in range(8) if all(d.get((i,key)) == 0 for key in ('HCU use (%)','HCU memory use (%)'))]
if len(ids) < n: raise SystemExit('Not enough idle HCU devices')
print(','.join(map(str, ids[:n])))
PY
  )
fi
IFS=',' read -r -a gpu_ids <<< "$devices"
gpu_count="${#gpu_ids[@]}"
(( gpu_count >= 1 && gpu_count <= 8 && gpu_count <= count )) || { echo 'Invalid device count' >&2; exit 2; }
printf 'Shadow count=%s devices=%s shards=%s stack=%s\n' "$count" "$devices" "$gpu_count" "$SHADOW_STACK_ID"
python - "$devices" <<'PY'
import re, subprocess, sys
items = sys.argv[1].split(',')
if len(set(items)) != len(items) or any(not x.isdigit() or not 0 <= int(x) < 8 for x in items):
    raise SystemExit('Invalid or repeated HCU index')
s = subprocess.check_output(['/opt/hyhal/bin/hy-smi','--showmemuse','--showuse'], text=True)
pairs = re.findall(r'HCU\[(\d+)\]\s*:\s*(HCU use \(%\)|HCU memory use \(%\)):\s*([0-9.]+)', s)
d = {(int(i), key): float(v) for i, key, v in pairs}
if not all(d.get((int(i), key)) == 0 for i in items for key in ('HCU use (%)','HCU memory use (%)')):
    raise SystemExit('Selected HCU device is busy or status unavailable')
PY
for ((i=0; i<gpu_count; i++)); do
  python -c 'import socket,sys; s=socket.socket(); s.bind(("127.0.0.1",int(sys.argv[1]))); s.close()' "$((port_base+i))"
done
mkdir -p "$out/logs"
logrun=$(mktemp -d "$out/logs/run.XXXXXXXX")
common=(--anchor "$anchor" --inference "$inference" --embedding-path "$embedding" --model-path "$model"
        --weight-manifest "$weights" --anchor-predictions-sha256 "$ANCHOR_PRED_SHA"
        --anchor-manifest-sha256 "$ANCHOR_MANIFEST_SHA")
audit_common=("${common[@]}")
if [[ "$count" == 400 ]]; then
  python -c 'from pathlib import Path; from scripts.rescore_textvqa_mmf import load_evaluator; import sys; load_evaluator(Path(sys.argv[1]))' "$evaluator"
  # Full must be preceded by a freshly revalidated completed 20-item smoke.
  smoke_shards="${SHADOW_SMOKE_SHARDS:-$gpu_count}"
  OMP_NUM_THREADS="${SHADOW_AUDIT_THREADS:-8}" MKL_NUM_THREADS="${SHADOW_AUDIT_THREADS:-8}" \
    python scripts/audit_textvqa_shadow.py "${audit_common[@]}" --output "$smoke" --limit 20 \
      --shard-count "$smoke_shards" --report "$logrun/smoke-recheck.json"
fi
python scripts/run_hygon_textvqa_shadow.py "${common[@]}" --output "$out" --limit "$count" \
  --shard-count "$gpu_count" --shard-index 0 --stack-id "$SHADOW_STACK_ID" --check-only
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
for ((i=0; i<gpu_count; i++)); do
  setsid env HIP_VISIBLE_DEVICES="${gpu_ids[$i]}" VLLM_USE_FLASHINFER_SAMPLER=0 \
    vllm serve "$model" --served-model-name Qwen/Qwen2.5-VL-7B-Instruct \
    --host 127.0.0.1 --port "$((port_base+i))" --trust-remote-code --gpu-memory-utilization 0.75 \
    --max-model-len 32768 >"$logrun/vllm-$i.log" 2>&1 &
  services+=("$!")
done
ready=0
for _ in $(seq 1 600); do
  all_ready=1
  for ((i=0; i<gpu_count; i++)); do
    kill -0 "${services[$i]}" 2>/dev/null || { echo "Service $i exited" >&2; exit 3; }
    curl --silent --max-time 2 --fail "http://127.0.0.1:$((port_base+i))/health" >/dev/null || all_ready=0
  done
  if [[ "$all_ready" == 1 ]]; then ready=1; break; fi
  sleep 2
done
[[ "$ready" == 1 ]] || { echo 'Service readiness timeout' >&2; exit 4; }
for ((i=0; i<gpu_count; i++)); do
  setsid env OMP_NUM_THREADS="${SHADOW_WORKER_THREADS:-8}" MKL_NUM_THREADS="${SHADOW_WORKER_THREADS:-8}" \
    python scripts/run_hygon_textvqa_shadow.py "${common[@]}" --output "$out" --limit "$count" \
      --shard-count "$gpu_count" --shard-index "$i" --stack-id "$SHADOW_STACK_ID" \
      --base-url "http://127.0.0.1:$((port_base+i))/v1" >"$logrun/worker-$i.log" 2>&1 &
  workers+=("$!")
done
while (( ${#workers[@]} )); do
  completed=''
  if ! wait -n -p completed "${workers[@]}" "${services[@]}"; then
    echo "A shadow worker or model service failed: ${completed:-unknown}" >&2; exit 5
  fi
  worker_found=0
  pending=()
  for pid in "${workers[@]}"; do
    if [[ "$pid" == "$completed" ]]; then worker_found=1; else pending+=("$pid"); fi
  done
  [[ "$worker_found" == 1 ]] || { echo "Model service exited unexpectedly: $completed" >&2; exit 5; }
  workers=("${pending[@]}")
done
cleanup
services=()
OMP_NUM_THREADS="${SHADOW_AUDIT_THREADS:-8}" MKL_NUM_THREADS="${SHADOW_AUDIT_THREADS:-8}" \
  python scripts/audit_textvqa_shadow.py "${audit_common[@]}" --output "$out" --limit "$count" \
    --shard-count "$gpu_count" --report "$logrun/audit.json"
echo "Shadow audit complete: $logrun/audit.json"
if [[ "$count" == 400 ]]; then
  OMP_NUM_THREADS="${SHADOW_AUDIT_THREADS:-8}" MKL_NUM_THREADS="${SHADOW_AUDIT_THREADS:-8}" \
    python scripts/score_textvqa_shadow_mmf.py --anchor "$anchor" --shadow "$out" \
      --inference "$inference" --subset-manifest "$(dirname "$inference")/manifest.json" \
      --subset-manifest-sha256 c8916fe1b1aea8650056086be1d868b11b1add52b93a598cc56851b37c431eaf \
      --labels "$(dirname "$inference")/labels.jsonl" --mmf-evaluator "$evaluator" \
      --embedding-path "$embedding" --model-path "$model" --weight-manifest "$weights" \
      --anchor-predictions-sha256 "$ANCHOR_PRED_SHA" --anchor-manifest-sha256 "$ANCHOR_MANIFEST_SHA" \
      --shard-count "$gpu_count" --output "$out/score_mmf.json"
fi
