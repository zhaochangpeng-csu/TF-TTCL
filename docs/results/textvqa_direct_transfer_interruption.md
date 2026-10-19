# TextVQA direct-transfer development run: infrastructure interruption

The frozen four-arm TextVQA train stream is **not yet scored**. The first
attempt used Qwen2.5-VL-7B-Instruct on 400 predeclared, label-isolated train
questions. It committed an ordered prefix of 173 complete samples, then the
first visual request for sample index 173 (question ID `25072`) was rejected:

```text
Input length (16245) exceeds model's maximum context length (8192).
```

That sample's original image is 4608 × 3456 pixels. No prediction was
committed for index 173. The manifest says `processed=173` and
`labels_read=false`; neither the full-run audit casebook nor an offline score
exists. The 20-item label-free smoke test had previously passed, but did not
contain this large image. The original 173-row prefix and immutable response
traces are preserved under
`core/exp_res/mm/tf_ttcl_mm_direct/qwen25vl/textvqa_dev400_seed0/`.

Recovery uses the same model, image files, prompts, seed, selection algorithm,
and online memory code, but serves the model with a 32,768-token capacity.
This is a serving limit rather than an image or algorithm change. Before
resuming, the recovery launcher reissues six label-free requests from the
committed prefix (greedy and memory-conditioned student requests at indices
0, 86, and 172) and checks exact output text and token counts plus close
token logprobs. If any comparison differs, it starts a new 400-item namespace
from index zero instead of mixing results. Any request error aborts without
scoring. Only a complete 400-item run that passes the label-free protocol
audit may open the isolated targets for offline scoring.

For this interrupted run, the recovery launcher is copied to a temporary
server path and executed while the runner checkout remains at its recorded
Git SHA `4d86753b313bd1b52286bd8a43cf53dc33f96830`. Do not update that
checkout until completion. The launcher's own SHA-256 and the selected GPU
are printed in its log. It waits for an idle permitted GPU rather than
interrupting another user's process, and cleans up only its owned vLLM
service on exit.
