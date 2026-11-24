#!/usr/bin/env bash
# Execute inside the Hygon container. The scorer runs only after the launcher
# validates all 400 label-free predictions and cleans up its eight services.
set -euo pipefail
root=/data/cp-experiments/tf-ttcl-hygon-20261124
project="$root/project"
data="$project/data/multimodal/scienceqa_modelscope_train"
reference="$project/core/exp_res/mm/tf_ttcl_mm_direct/qwen25vl/dev400_seed0/predictions.jsonl"
output="$root/results/dev400_8gpu"
cd "$project"
python scripts/run_hygon_scienceqa_parallel.py --project "$project" \
  --model "$root/modelscope_test" --weight-manifest "$root/weight_hash_verified.json" \
  --smoke-predictions "$root/results/smoke20/predictions.jsonl" \
  --inference "$data/dev400_tf_ttcl_mm_direct.inference.jsonl" --reference "$reference" \
  --output "$output"
python scripts/compare_hygon_scienceqa_greedy.py \
  --inference "$data/dev400_tf_ttcl_mm_direct.inference.jsonl" --reference "$reference" \
  --reference-sha256 b8ac87c5bdf38923f7ec5e9e848b3f1b18e0f183b2ea0deffea98700ffaf4d71 \
  --shard "$output/predictions.jsonl" --labels "$data/dev400_tf_ttcl_mm_direct.labels.jsonl" \
  --output "$output/comparison.json"
