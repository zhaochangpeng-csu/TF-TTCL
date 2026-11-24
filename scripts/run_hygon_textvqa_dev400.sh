#!/usr/bin/env bash
set -euo pipefail
root=/data/cp-experiments/tf-ttcl-hygon-20261124
project="$root/project"
data="$project/data/multimodal/textvqa_train_tf_ttcl_mm_direct"
reference="$project/core/exp_res/mm/tf_ttcl_mm_direct/qwen25vl/textvqa_dev400_ctx32768_seed0/predictions.jsonl"
scienceqa_reference="$project/core/exp_res/mm/tf_ttcl_mm_direct/qwen25vl/dev400_seed0/predictions.jsonl"
output="$root/results/textvqa_dev400_8gpu"
cd "$project"
python -c 'from pathlib import Path; from scripts.run_hygon_scienceqa_parallel import verify_smoke; import sys; verify_smoke(Path(sys.argv[1]), Path(sys.argv[2]))' \
  "$root/results/textvqa_smoke20/predictions.jsonl" "$reference"
python scripts/run_hygon_scienceqa_parallel.py --project "$project" \
  --model "$root/modelscope_test" --weight-manifest "$root/weight_hash_verified.json" \
  --smoke-predictions "$root/results/smoke20/predictions.jsonl" --smoke-reference "$scienceqa_reference" \
  --inference "$data/inference.jsonl" --reference "$reference" --max-model-len 32768 \
  --worker-script scripts/hygon_textvqa_greedy.py --output "$output"
python scripts/compare_hygon_textvqa_greedy.py --inference "$data/inference.jsonl" \
  --reference "$reference" --reference-sha256 4750646aa363532c6169f98c332e0e3ef32dc1e04ec13b8feb729b3874e762c4 \
  --shard "$output/predictions.jsonl" --labels "$data/labels.jsonl" --output "$output/comparison.json"
