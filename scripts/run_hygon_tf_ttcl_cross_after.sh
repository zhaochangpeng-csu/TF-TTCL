#!/usr/bin/env bash
# Host-side watcher: score the cross-platform comparison only after the
# container run and its full label-free audit finish successfully.
set -euo pipefail
root=/data/cp-experiments/tf-ttcl-hygon-20261124
project="$root/project"
data="$project/data/multimodal/scienceqa_modelscope_train"
run="$root/results/online_scienceqa_400_seed0"
reference="$project/core/exp_res/mm/tf_ttcl_mm_direct/qwen25vl/dev400_seed0/predictions.jsonl"
while tmux has-session -t '=tf-ttcl-online-dev400' 2>/dev/null; do sleep 30; done
if [[ ! -f "$run/comparison.json" || ! -f "$run/unscored_casebook.md" ]]; then
  echo 'Full run did not finish its audit and scoring; refusing cross comparison.' >&2
  exit 1
fi
docker exec -w "$project" cp-experiments python scripts/compare_hygon_tf_ttcl_direct.py \
  --run "$run" --inference "$data/dev400_tf_ttcl_mm_direct.inference.jsonl" \
  --reference "$reference" --labels "$data/dev400_tf_ttcl_mm_direct.labels.jsonl" \
  --output "$run/cross_platform_comparison.json"
