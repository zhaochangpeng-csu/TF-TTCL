#!/usr/bin/env bash
# Bounded wait for idle GPU 2; run exactly one target-model logprob probe.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
run_root="core/exp_res/mm/tf_ttcl_mm_direct/qwen25vl"
mkdir -p "$run_root"
exec >>"$run_root/probe_queue.log" 2>&1
echo "$(date -Is) Waiting for an idle GPU 2; no existing process will be stopped."
for _ in $(seq 1 720); do
  active_pids="$(nvidia-smi -i 2 --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')"
  if [[ -z "$active_pids" ]]; then
    echo "$(date -Is) GPU 2 idle; starting the single label-free PPL probe."
    bash scripts/run_tf_ttcl_mm_probe.sh
    echo "$(date -Is) Probe passed; owned service released."
    exit 0
  fi
  sleep 60
done
echo "$(date -Is) GPU 2 remained occupied for 12 hours; no target service started." >&2
exit 2
