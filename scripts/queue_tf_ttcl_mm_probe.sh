#!/usr/bin/env bash
# Bounded wait for an idle GPU; run exactly one target-model logprob probe.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
run_root="core/exp_res/mm/tf_ttcl_mm_direct/qwen25vl"
gpu_index="${GPU_INDEX:-2}"
if [[ ! "$gpu_index" =~ ^(0|2|3)$ ]]; then
  echo "Unsupported GPU index: $gpu_index" >&2
  exit 2
fi
mkdir -p "$run_root"
exec >>"$run_root/probe_queue.log" 2>&1
echo "$(date -Is) Waiting for an idle GPU $gpu_index; no existing process will be stopped."
for _ in $(seq 1 720); do
  active_pids="$(nvidia-smi -i "$gpu_index" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')"
  if [[ -z "$active_pids" ]]; then
    echo "$(date -Is) GPU $gpu_index idle; starting the single label-free PPL probe."
    GPU_INDEX="$gpu_index" bash scripts/run_tf_ttcl_mm_probe.sh
    echo "$(date -Is) Probe passed; owned service released."
    exit 0
  fi
  sleep 60
done
echo "$(date -Is) GPU $gpu_index remained occupied for 12 hours; no target service started." >&2
exit 2
