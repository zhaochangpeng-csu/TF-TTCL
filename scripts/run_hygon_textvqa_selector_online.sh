#!/usr/bin/env bash
# Two independent, strictly ordered TextVQA streams. Inside cp-experiments.
set -euo pipefail
root="${SELECTOR_ROOT:-/data/cp-experiments/tf-ttcl-hygon-20261124}"
project="${SELECTOR_PROJECT:?Set frozen project path}"
data="${SELECTOR_DATA:?Set frozen new dataset directory}"
: "${SELECTOR_MANIFEST_SHA:?Set frozen dataset manifest SHA256}"
: "${SELECTOR_PRIOR_TRAIN:?Set first prior train inference path}"
: "${SELECTOR_PRIOR_VAL:?Set prior validation inference path}"
: "${SELECTOR_PRIOR_FRESH:?Set second prior train inference path}"
: "${SELECTOR_STACK_ID:?Set immutable software identity}"
model="${SELECTOR_MODEL:-$root/modelscope_test}"
embedding="${SELECTOR_EMBEDDING:-$root/embedding}"
weights="${SELECTOR_WEIGHT_MANIFEST:-$root/weight_hash_verified.json}"
evaluator="${SELECTOR_MMF_EVALUATOR:-$root/textvqa_rescore_project_280c1a7/m4c_evaluators.py}"
count="${SELECTOR_COUNT:-20}"
[[ "$count" == 20 || "$count" == 400 ]] || { echo 'Need count 20 or 400' >&2; exit 2; }
[[ "$SELECTOR_MANIFEST_SHA" =~ ^[0-9a-f]{64}$ ]] || { echo 'Invalid manifest SHA' >&2; exit 2; }
cd "$project"
out="${SELECTOR_OUTPUT_ROOT:-$root/results/textvqa_selector_online_v1}_${count}"
smoke="${SELECTOR_OUTPUT_ROOT:-$root/results/textvqa_selector_online_v1}_20"
port_base=20200
if [[ "$count" == 400 ]]; then port_base=20300; fi
if [[ -e "$out" && "${SELECTOR_RESUME:-0}" != 1 ]]; then echo "Output exists: $out" >&2; exit 2; fi
devices="${SELECTOR_DEVICES:-auto}"
if [[ "$devices" == auto ]]; then
  devices=$(python - <<'PY'
import re, subprocess
s = subprocess.check_output(['/opt/hyhal/bin/hy-smi','--showmemuse','--showuse'], text=True)
d = {(int(i), k):float(v) for i,k,v in re.findall(r'HCU\[(\d+)\]\s*:\s*(HCU use \(%\)|HCU memory use \(%\)):\s*([0-9.]+)', s)}
ids = [i for i in range(8) if all(d.get((i,k)) == 0 for k in ('HCU use (%)','HCU memory use (%)'))]
if len(ids) < 2: raise SystemExit('Need two idle HCU devices')
print(','.join(map(str, ids[:2])))
PY
  )
fi
IFS=',' read -r -a gpu <<< "$devices"
[[ ${#gpu[@]} == 2 && "${gpu[0]}" != "${gpu[1]}" ]] || { echo 'Need two different device indices' >&2; exit 2; }
check_idle() {
  python - "$devices" <<'PY'
import re,subprocess,sys
ids=sys.argv[1].split(',')
if len(ids)!=2 or len(set(ids))!=2 or any(not x.isdigit() or not 0<=int(x)<8 for x in ids): raise SystemExit('Invalid HCU devices')
s=subprocess.check_output(['/opt/hyhal/bin/hy-smi','--showmemuse','--showuse'],text=True)
d={(int(i),k):float(v) for i,k,v in re.findall(r'HCU\[(\d+)\]\s*:\s*(HCU use \(%\)|HCU memory use \(%\)):\s*([0-9.]+)',s)}
if not all(d.get((int(i),k))==0 for i in ids for k in ('HCU use (%)','HCU memory use (%)')): raise SystemExit('Selected HCU device busy')
PY
}
check_idle
for port in "$port_base" "$((port_base+1))"; do
  python -c 'import socket,sys; s=socket.socket(); s.bind(("127.0.0.1",int(sys.argv[1]))); s.close()' "$port"
done
common=(--inference "$data/inference.jsonl" --subset-manifest "$data/manifest.json"
  --subset-manifest-sha256 "$SELECTOR_MANIFEST_SHA" --excluded-inference "$SELECTOR_PRIOR_TRAIN"
  --excluded-inference "$SELECTOR_PRIOR_VAL" --excluded-inference "$SELECTOR_PRIOR_FRESH")
run_common=("${common[@]}" --output "$out" --embedding-path "$embedding" --model-path "$model"
  --weight-manifest "$weights" --count "$count" --stack-id "$SELECTOR_STACK_ID")
for selector in full_response normalized_answer; do
  python scripts/run_hygon_textvqa_selector_online.py "${run_common[@]}" --selector "$selector" --check-only
done
if [[ "$count" == 400 ]]; then
  python -c 'from pathlib import Path; from scripts.rescore_textvqa_mmf import load_evaluator; import sys; load_evaluator(Path(sys.argv[1]))' "$evaluator"
  for selector in full_response normalized_answer; do
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python scripts/audit_textvqa_selector_online.py --output "$smoke/$selector" --inference "$data/inference.jsonl" --count 20 --embedding-path "$embedding"
  done
fi
# Another job may have acquired a device during the CPU preflight.
check_idle
mkdir -p "$out/logs"
logrun=$(mktemp -d "$out/logs/run.XXXXXXXX")
printf 'count=%s devices=%s stack=%s logs=%s\n' "$count" "$devices" "$SELECTOR_STACK_ID" "$logrun"
services=(); workers=()
cleanup() {
  for pid in "${workers[@]}" "${services[@]}"; do kill -- "-$pid" 2>/dev/null || true; done
  for _ in $(seq 1 20); do
    active=0
    for pid in "${workers[@]}" "${services[@]}"; do if kill -0 -- "-$pid" 2>/dev/null; then active=1; fi; done
    [[ "$active" == 0 ]] && break
    sleep 1
  done
  for pid in "${workers[@]}" "${services[@]}"; do kill -KILL -- "-$pid" 2>/dev/null || true; done
  for pid in "${workers[@]}" "${services[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for i in 0 1; do
  setsid env HIP_VISIBLE_DEVICES="${gpu[$i]}" VLLM_USE_FLASHINFER_SAMPLER=0 \
    vllm serve "$model" --served-model-name Qwen/Qwen2.5-VL-7B-Instruct --host 127.0.0.1 \
    --port "$((port_base+i))" --trust-remote-code --gpu-memory-utilization 0.75 --max-model-len 32768 \
    >"$logrun/vllm-$i.log" 2>&1 &
  services+=("$!")
done
ready=0
for _ in $(seq 1 600); do
  all_ready=1
  for i in 0 1; do
    kill -0 "${services[$i]}" 2>/dev/null || { echo "Model service $i exited" >&2; exit 3; }
    curl --silent --max-time 2 --fail "http://127.0.0.1:$((port_base+i))/health" >/dev/null || all_ready=0
  done
  if [[ "$all_ready" == 1 ]]; then ready=1; break; fi
  sleep 2
done
[[ "$ready" == 1 ]] || { echo 'Model service timeout' >&2; exit 4; }
for i in 0 1; do
  selector=full_response
  [[ "$i" == 1 ]] && selector=normalized_answer
  setsid env OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python scripts/run_hygon_textvqa_selector_online.py \
    "${run_common[@]}" --selector "$selector" --base-url "http://127.0.0.1:$((port_base+i))/v1" \
    >"$logrun/worker-$selector.log" 2>&1 &
  workers+=("$!")
done
while (( ${#workers[@]} )); do
  completed=''
  if ! wait -n -p completed "${workers[@]}" "${services[@]}"; then echo "Process failed: ${completed:-unknown}" >&2; exit 5; fi
  worker_found=0; pending=()
  for pid in "${workers[@]}"; do if [[ "$pid" == "$completed" ]]; then worker_found=1; else pending+=("$pid"); fi; done
  [[ "$worker_found" == 1 ]] || { echo "Model service exited unexpectedly: $completed" >&2; exit 5; }
  workers=("${pending[@]}")
done
cleanup; services=()
if [[ "$count" == 20 ]]; then
  for selector in full_response normalized_answer; do
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python scripts/audit_textvqa_selector_online.py \
      --output "$out/$selector" --inference "$data/inference.jsonl" --count "$count" --embedding-path "$embedding" \
      | tee "$logrun/audit-$selector.json"
  done
else
  # score_textvqa_selector_online first performs both full audits in the same
  # process; only after both pass can it open labels. This avoids duplicating
  # all 400-item CPU embedding work and permits an exact shared embedding cache.
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python scripts/score_textvqa_selector_online.py \
    "${common[@]}" --full "$out/full_response" --answer "$out/normalized_answer" \
    --labels "$data/labels.jsonl" --embedding-path "$embedding" --mmf-evaluator "$evaluator" \
    --output "$out/score_mmf.json"
fi
echo "Selector run, audit and applicable scoring completed: $out"
