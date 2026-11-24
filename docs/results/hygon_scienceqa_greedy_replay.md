# Hygon HCU migration: paired ScienceQA greedy replay

On 2026-11-24, Qwen2.5-VL-7B-Instruct was evaluated on the same 400
ScienceQA-IMG train development questions and original greedy prompt used in
the earlier NVIDIA direct-transfer baseline. This measures the combined
effect of changing hardware **and** the inference software stack, not an
isolated hardware precision experiment.

| Environment | Correct / 400 | Accuracy |
| --- | ---: | ---: |
| Original NVIDIA stack | 356 | 89.00% |
| Hygon HCU stack | 360 | 90.00% |

The final answer agreed on 389/400 questions (97.25%). Of the 11 changed
answers, six improved, two worsened, and three had the same correctness.
Prompt token counts agreed on all 400 items; generated completion token counts
agreed on 248/400, so matching final choices does not imply identical reasoning
text or generation traces.
The paired accuracy difference was +1.00 percentage point (95% bootstrap CI
[−0.25, +2.50] pp; two-sided paired randomization p=0.2891). The interval
does not establish that the two stacks are equivalent; one 400-item stream,
one model and one greedy configuration cannot establish general equivalence.

All five ModelScope weight shards and the configuration/tokenizer/processor
files matched SHA-256 hashes calculated on the NVIDIA server. Before inference
the 400 questions, original prompt and sampling configuration, IDs, and image
hashes were checked against the original stored calls. A separate 20-item
label-free smoke had 19/20 answer agreement and successfully returned complete
sampled token logprobs. The 20 items are part of the 400, not an additional
independent evaluation.

The full replay used eight independent 64-GiB Hygon HCU devices in the existing
`cp-experiments` Docker container, with one model service per device and 50
independent greedy questions per worker. Worker elapsed times were 45.25–48.64
seconds after services were ready; this excludes loading, compilation, and
startup, and is not a measured end-to-end NVIDIA speedup. A full sample-level
audit completed before the isolated labels were opened. All eight model
services exited and HCU memory returned to 0% afterward.

Software versions: Hygon PyTorch `2.10.0+das.opt1.dtk2604.2608141812.g28498c`,
vLLM `0.21.0+das.dtk2604.torch2100.2606111143.g8c979d`, Transformers `5.5.0`,
model revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`; max context 8,192.
The full run records code hashes and configuration in `run_config.json`.

The next migration check should compare stochastic student outputs and
generation PPL, followed by an online-memory stream. The latter depends on
all previous questions and cannot be split into eight independent streams and
reported as the same online experiment.

An additional label-free sampled student probe on the first 20 original
questions returned complete finite token logprobs and had 18/20 final-answer
matches with NVIDIA. The same random seed does not guarantee matching samples
across inference stacks; this is a compatibility check, not an accuracy result.

Server output: `/data/cp-experiments/tf-ttcl-hygon-20261124/results/dev400_8gpu/`.
Input SHA-256: `01c6dc83958ea1cee51ab38594b2499e3bfbc4034139674dc799e5dfeb354689`;
reference predictions SHA-256: `b8ac87c5bdf38923f7ec5e9e848b3f1b18e0f183b2ea0deffea98700ffaf4d71`.
