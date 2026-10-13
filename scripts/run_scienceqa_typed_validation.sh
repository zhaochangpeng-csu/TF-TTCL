#!/usr/bin/env bash
# Frozen v3.1: label-free smoke -> four full arms -> guarded offline evaluation.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
python_bin="${PYTHON_BIN:-/home/lthpc/.conda/envs/ttt/bin/python}"
data_root="data/multimodal/scienceqa_img"
config_path="core/config/mm_scienceqa_validation_typed.yaml"
run_root="core/exp_res/mm/scienceqa_validation/qwen25vl/typed_v3p1_seed0_regression"
subset_manifest="${data_root}/validation_typed.manifest.json"

if [[ ! -f "$subset_manifest" ]]; then
  "$python_bin" scripts/create_label_isolated_subset.py \
    --source "${data_root}/scienceqa_img_val.jsonl" \
    --inference-output "${data_root}/validation_typed.inference.jsonl" \
    --labels-output "${data_root}/validation_typed.labels.jsonl" \
    --manifest "$subset_manifest" --count 2097 --seed 20261013 --offset 0
fi
"$python_bin" scripts/check_typed_validation.py --manifest "$subset_manifest" --expected-n 2097
"$python_bin" core/mm_main.py --config "$config_path" --validate-config
if [[ "${PREPARE_ONLY:-0}" == 1 ]]; then
  exit 0
fi

env CONFIG_PATH="$config_path" METHOD_FAMILY=shared \
  RUN_ROOT="${run_root}_smoke20" SMOKE_LIMIT=20 RUN_GREEDY=1 RUN_EFFICIENT=1 \
  bash scripts/run_scienceqa_causal_smoke.sh

# Smoke must be complete in all four arms. Never open labels to assess it.
"$python_bin" -c 'import sys; from pathlib import Path; from evaluate.multimodal.compare_typed import validate_four; r=Path(sys.argv[1]); validate_four({n:r/n/"predictions.jsonl" for n in ("greedy","shared_budget","shared_evidence","efficient_budget")},20); print("Four-arm unscored smoke passed")' "${run_root}_smoke20"

env CONFIG_PATH="$config_path" METHOD_FAMILY=shared \
  RUN_ROOT="$run_root" SMOKE_LIMIT=2097 RUN_GREEDY=1 RUN_EFFICIENT=1 \
  bash scripts/run_scienceqa_causal_smoke.sh

# The launcher has released its GPU before this evaluator can read labels.
"$python_bin" evaluate/multimodal/compare_typed.py \
  --greedy "${run_root}/greedy/predictions.jsonl" \
  --budget "${run_root}/shared_budget/predictions.jsonl" \
  --evidence "${run_root}/shared_evidence/predictions.jsonl" \
  --efficient "${run_root}/efficient_budget/predictions.jsonl" \
  --labels "${data_root}/validation_typed.labels.jsonl" \
  --expected-n 2097 >"${run_root}/comparison.json"
echo "Frozen validation regression finished: ${run_root}/comparison.json"
