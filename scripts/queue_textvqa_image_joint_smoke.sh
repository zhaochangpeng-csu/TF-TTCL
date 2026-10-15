#!/usr/bin/env bash
# Wait only for an idle GPU 2, then run the label-free 20-item smoke once.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
run_root="core/exp_res/mm/textvqa/qwen25vl/image_joint_v1_seed0_smoke20"
mkdir -p "$run_root"
exec >>"$run_root/queue.log" 2>&1
echo "$(date -Is) Waiting for GPU 2 to become idle; no current task will be killed."
for _ in $(seq 1 720); do
  active_pids="$(nvidia-smi -i 2 --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')"
  if [[ -z "$active_pids" ]]; then
    echo "$(date -Is) GPU 2 idle; starting label-free smoke."
    TEXTVQA_IMAGE_RUN_KIND=smoke20 bash scripts/run_textvqa_image_joint.sh
    echo "$(date -Is) Smoke and paired audit finished; owned model service released."
    exit 0
  fi
  sleep 60
done
echo "$(date -Is) Waited 12 hours; GPU 2 never became idle. No model started." >&2
exit 2
