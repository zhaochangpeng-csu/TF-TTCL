#!/usr/bin/env bash
# Resume the frozen TextVQA direct-transfer stream after the 8k serving cap failed.
# This launcher lives outside the server checkout so the runner's git/code hashes
# remain unchanged. It never reads labels until the complete label-free audit passes.
set -euo pipefail

project_root="${PROJECT_ROOT:-/home/lthpc/private/code/zcp/TTT/TF-TTCL}"
cd "$project_root"
python_bin="${PYTHON_BIN:-/home/lthpc/.conda/envs/ttt/bin/python}"
vllm_bin="${VLLM_BIN:-/home/lthpc/.conda/envs/ttt/bin/vllm}"
model_path="/home/lthpc/private/models/Qwen2.5-VL-7B-Instruct"
subset="data/multimodal/textvqa_train_tf_ttcl_mm_direct"
base_run="core/exp_res/mm/tf_ttcl_mm_direct/qwen25vl/textvqa_dev400_seed0"
fresh_run="core/exp_res/mm/tf_ttcl_mm_direct/qwen25vl/textvqa_dev400_ctx32768_seed0"
api_port="${API_PORT:-9000}"
max_model_len=32768

if [[ -n "$(git status --porcelain)" ]]; then
  echo "Server checkout is dirty; refusing to change the run provenance." >&2
  exit 2
fi
"$python_bin" - "$subset" "$base_run" <<'PY'
import json
import sys
from pathlib import Path
from scripts.run_tf_ttcl_mm_direct import checked_samples

subset, run = map(Path, sys.argv[1:])
samples = checked_samples(subset / "inference.jsonl", subset / "manifest.json", 400, "textvqa")
manifest = json.loads((run / "manifest.json").read_text())
rows = [json.loads(line) for line in (run / "predictions.jsonl").read_text().splitlines()]
if (manifest["status"], manifest["processed"], manifest["labels_read"]) != ("running", 173, False):
    raise ValueError("Expected the exact unscored 173-row interrupted prefix")
if len(rows) != 173 or any(row["sample_index"] != i or
                           row["sample_id"] != samples[i].sample_id for i, row in enumerate(rows)):
    raise ValueError("Interrupted prefix is not complete and ordered")
if manifest["git_sha"] != __import__("subprocess").check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip():
    raise ValueError("Server checkout changed since the interrupted run")
if (run / "unscored_casebook.md").exists() or (run / "offline_comparison.json").exists():
    raise ValueError("Interrupted stream unexpectedly has an audit or score")
print("Verified 173 complete label-free rows; next index is 173.", flush=True)
PY

echo "resume_launcher_sha256=$(sha256sum "$0" | cut -d ' ' -f 1)"
echo "runner_git_sha=$(git rev-parse HEAD) max_model_len=$max_model_len"

service_pid=""
cleanup() {
  if [[ -n "$service_pid" ]]; then
    kill -- "-$service_pid" 2>/dev/null || true
    wait "$service_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

gpu_index=""
while [[ -z "$gpu_index" ]]; do
  if curl --silent --max-time 2 --fail "http://127.0.0.1:${api_port}/health" >/dev/null; then
    echo "Port $api_port already serves another model; waiting."
    sleep 60
    continue
  fi
  for candidate in ${GPU_CANDIDATES:-1 2 0 3}; do
    case "$candidate" in 0|1|2|3) ;; *) echo "Invalid GPU index $candidate" >&2; exit 2 ;; esac
    active="$(nvidia-smi -i "$candidate" --query-compute-apps=pid \
      --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d')"
    used="$(nvidia-smi -i "$candidate" --query-gpu=memory.used \
      --format=csv,noheader,nounits | tr -d ' ')"
    if [[ -z "$active" && "$used" -lt 1024 ]]; then
      gpu_index="$candidate"
      break
    fi
  done
  if [[ -z "$gpu_index" ]]; then
    echo "All permitted GPUs are occupied; waiting 60 seconds."
    sleep 60
  fi
done
echo "Selected idle GPU $gpu_index."

setsid env CUDA_VISIBLE_DEVICES="$gpu_index" VLLM_USE_FLASHINFER_SAMPLER=0 \
  "$vllm_bin" serve "$model_path" --served-model-name Qwen/Qwen2.5-VL-7B-Instruct \
  --host 127.0.0.1 --port "$api_port" --trust-remote-code \
  --gpu-memory-utilization 0.85 --max-model-len "$max_model_len" \
  >"$base_run/vllm_resume_ctx32768.log" 2>&1 &
service_pid="$!"
ready=0
for _ in $(seq 1 600); do
  if ! kill -0 "$service_pid" 2>/dev/null; then
    echo "Owned vLLM failed; see $base_run/vllm_resume_ctx32768.log" >&2
    exit 3
  fi
  if curl --silent --max-time 2 --fail "http://127.0.0.1:${api_port}/health" >/dev/null; then
    ready=1
    break
  fi
  sleep 2
done
if [[ "$ready" != 1 ]]; then
  echo "Owned vLLM did not become ready within 20 minutes." >&2
  exit 4
fi
nvidia-smi -i "$gpu_index" --query-gpu=index,memory.used,utilization.gpu --format=csv \
  >"$base_run/gpu_resume_loaded.csv"

set +e
"$python_bin" - "$base_run" "http://127.0.0.1:${api_port}/v1" <<'PY'
import json
import math
import sys
from pathlib import Path
from core.multimodal.client import OpenAIVLMClient
from core.multimodal.schema import SamplingConfig

run, base_url = Path(sys.argv[1]), sys.argv[2]
rows = [json.loads(line) for line in (run / "predictions.jsonl").read_text().splitlines()]
client = OpenAIVLMClient(base_url, "Qwen/Qwen2.5-VL-7B-Instruct")
for index in (0, 86, 172):
    row = rows[index]
    for arm, call_index in (("greedy", 0), ("tf_ttcl_mm_direct", 2)):
        call = row["arms"][arm]["calls"][call_index]
        actual = client.generate(call["system"], call["user"], call["image_paths"],
                                 SamplingConfig(**call["sampling"]))
        expected = call["response"]
        if (actual.content != expected["content"] or
                actual.prompt_tokens != expected["prompt_tokens"] or
                actual.completion_tokens != expected["completion_tokens"] or
                len(actual.logprobs) != len(expected["logprobs"]) or
                any(not math.isclose(a, b, abs_tol=0.02, rel_tol=1e-4)
                    for a, b in zip(actual.logprobs, expected["logprobs"]))):
            print(f"Probe mismatch: row={index}, arm={arm}", flush=True)
            raise SystemExit(10)
        print(f"Probe matched: row={index}, arm={arm}, "
              f"prompt_tokens={actual.prompt_tokens}", flush=True)
PY
probe_status="$?"
set -e
if [[ "$probe_status" == 0 ]]; then
  run_root="$base_run"
  echo "Probe matched; resuming the immutable 173-row prefix."
elif [[ "$probe_status" == 10 ]]; then
  run_root="$fresh_run"
  if [[ -e "$fresh_run/manifest.json" || -e "$fresh_run/predictions.jsonl" ]]; then
    echo "Probe differed and fresh namespace already exists; refusing to reuse it." >&2
    exit 5
  fi
  mkdir -p "$run_root"
  echo "Probe differed; restarting all 400 samples in $run_root."
else
  echo "Probe request failed (exit $probe_status); leaving both runs unscored." >&2
  exit 6
fi

"$python_bin" scripts/run_tf_ttcl_mm_direct.py \
  --dataset textvqa --inference "$subset/inference.jsonl" \
  --subset-manifest "$subset/manifest.json" --output "$run_root" \
  --count 400 --seed 0 --base-url "http://127.0.0.1:${api_port}/v1"
"$python_bin" scripts/audit_tf_ttcl_mm_direct.py --output "$run_root" \
  --inference "$subset/inference.jsonl" --count 400 \
  --casebook "$run_root/unscored_casebook.md"
"$python_bin" evaluate/multimodal/compare_tf_ttcl_mm_direct.py \
  --run "$run_root" --inference "$subset/inference.jsonl" \
  --labels "$subset/labels.jsonl" --output "$run_root/offline_comparison.json"
echo "Completed full audit and offline comparison in $run_root."
