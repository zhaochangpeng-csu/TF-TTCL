# TextVQA answer-similarity selector: disjoint online development protocol

The fixed-candidate replay on an earlier scored stream suggested that using
answer similarity instead of full-response similarity may improve selection,
but its short-final/no-memory confidence interval included zero. The next
experiment tests that hypothesis on 400 newly selected TextVQA train questions
and images, excluded by both ID and image ID from both previous 400-item train
streams and the prior 5,000-item validation stream. Selection uses identifiers
only; data preparation reads annotation records solely to export isolated
labels. The new subset remains a development set, not a benchmark test set.
Its selection seed is `2026112602`; the frozen selection SHA-256 is
`cb52fa624de46e3d4d8415c02a85afbd4ca3e9331706ac45052f1209c4dc0f2b`.
The exported manifest and inference SHA-256 values are
`53e98605c927b92da5705d5386a6343b6cf3dd411d70ac0d1a7f9276c0f83028` and
`48d194d3b31c85d8e99336e324bd1e74842bbafb0d9e27d22093593b914754bc`.

Run two independent, strictly ordered online streams, both starting with empty
memory. Both use the same frozen Qwen2.5-VL model, short-final prompt, images,
question order, seeds, four Tutor rewrites, generated-token PPL and top-30
positive/negative rule retrieval. The only intended algorithmic difference is
the text embedded when forming the open-task Teacher/top-two-positive group:

* `full_response`: original Teacher and Student complete responses.
* `normalized_answer`: their project-normalized extracted answer strings.

Each stream generates and writes its own rules after committing the current
sample; neither may read current/future samples or evaluation targets. Because
selections affect future rules and prompts, memory streams are genuinely
independent, and the same seeds do not guarantee identical later candidates.
Separate devices may introduce device-level variation. Within each stream,
record both selectors' choices on exactly the same no-memory candidates;
this supplies a paired comparison that does not regenerate candidates.

First perform a 20-item unscored smoke for each policy, audit exact inputs,
image hashes, complete token log probabilities and PPL, selected groups,
rule writes, retrieval and no label leakage. If both pass, run all 400 on
the two streams and audit before opening the isolated targets. Report greedy,
both selectors on shared no-memory candidates, and each full online stream.
Use the pinned Meta MMF evaluator with the same raw-answer extractor for all
conditions; report paired differences, 95% bootstrap intervals, format
failures, logical/physical token costs and runtime. EvalAI server parity is
not yet verified. A single order and single model are development evidence;
confirmation requires further ordering seeds and another model/dataset.
