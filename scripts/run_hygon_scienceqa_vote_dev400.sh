#!/usr/bin/env bash
set -euo pipefail
root=/data/cp-experiments/tf-ttcl-hygon-20261124
project="$root/project"
data="$project/data/multimodal/scienceqa_modelscope_train"
reference="$project/core/exp_res/mm/tf_ttcl_mm_direct/qwen25vl/dev400_seed0/predictions.jsonl"
greedy="$root/results/dev400_8gpu/predictions.jsonl"
cd "$project"
common=(--project "$project" --model "$root/modelscope_test" --weight-manifest "$root/weight_hash_verified.json"
        --smoke-predictions "$root/results/smoke20/predictions.jsonl"
        --inference "$data/dev400_tf_ttcl_mm_direct.inference.jsonl" --reference "$reference"
        --greedy "$greedy" --greedy-sha256 c79333d6dde5589905efeab450a08039b6d75fc1767aef4ed8e1a72037707a9e)
python scripts/run_hygon_scienceqa_vote_parallel.py "${common[@]}" --count 20 --output "$root/results/vote_smoke20"
python scripts/run_hygon_scienceqa_vote_parallel.py "${common[@]}" --count 400 \
  --vote-smoke "$root/results/vote_smoke20/predictions.jsonl" --output "$root/results/vote_dev400_8gpu"
python scripts/compare_hygon_scienceqa_vote.py --inference "$data/dev400_tf_ttcl_mm_direct.inference.jsonl" \
  --reference "$reference" --greedy "$greedy" --shard "$root/results/vote_dev400_8gpu/predictions.jsonl" \
  --labels "$data/dev400_tf_ttcl_mm_direct.labels.jsonl" --output "$root/results/vote_dev400_8gpu/comparison.json"
