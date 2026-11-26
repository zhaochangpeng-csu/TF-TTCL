# TextVQA lexical rule-conflict controls: fresh online protocol

The earlier fixed-history replay found that retrieved rule content can change
both output formatting and score, and the next disjoint 400-item online
stream found a roughly nine-point memory deficit. These are development
observations, not proof that any individual rule caused the loss. This
experiment tests one narrower, label-free hypothesis: some retrieved rule
instructions conflict with the existing request to provide reasoning.

Use a newly selected TextVQA train stream disjoint by question and image from
all three previous 400-item train streams and the prior validation stream.
Selection uses identifiers only; annotations are opened during preparation
only to export isolated labels. This is another development stream rather than
an official hidden test set. Freeze the exact input and source hashes before
the unscored smoke test.
The selection seed is `2026112701` and its selection SHA-256 is
`bbc1d677dfc6f7f4610935cdca72a2190f94adf53a635188438e68c3d822bfdc`.
The frozen manifest SHA-256 is
`bdcda110fe725732fa4072ab4cc87b067d6d93c7c3696bdb354510adbdcb01fd`, and
the inference SHA-256 is
`e2a66a58040d117248024bfa06ebb489b3513261551d53dd9768269649dbcf1e`.

The three strictly ordered online streams start with empty independent memories.
All use the same short-final prompt, full-response selector, Teacher/Student
sampling, PPL, Tutor rewrites and top-30 retrieval. Their only intended
difference is which retrieved rule texts appear in the memory-conditioned
prompt:

* `original`: all retrieved rules are unchanged.
* `targeted`: replace rules matched by a frozen first-sentence lexical pattern
  indicating an instruction to omit reasoning or explanation.
* `matched_random`: from each positive/negative bucket replace the same number
  of rules as the lexical matcher identifies **in that stream's own history**,
  choosing deterministic pseudo-random positions.

Replacement uses neutral placeholder text of approximately the original rule's
tokenizer length, preserving bucket, rank and number of entries. Record actual
replacement IDs, token counts, prompt tokens and completion tokens; the
placeholders are not necessarily semantically inert. Each stream builds its
own history, so matched-random counts need not match targeted counts for the
same question across streams. The lexical matcher is a transparent, limited
heuristic, not a semantic conflict verifier or a judgement that a rule is
false. Its definition and version are frozen before any new labels are scored.
The initial pattern was tightened after inspecting unlabeled historical rules:
it now requires the action word to precede the reasoning-related term and
excludes several clear double-negation and non-conflict constructions. The
frozen gate file SHA-256 is
`3fda87925ec31be621fa27c71bdeafa9eb2c2adb57136a686a68f54fc1c4f4e2`.
On the old short-final stream it matched 358 of 23,070 retrieval instances,
affecting 232 of 400 samples and 13 distinct rules; this is coverage, not
proof of semantic conflict or a measure of accuracy. The label-free
[coverage report](textvqa_rule_gate_coverage_400.json) records provenance.

Greedy and no-memory candidates are quality anchors. First run and audit 20
unscored examples per stream. Check actual image and prompt, tokenizer lengths,
strictly earlier rule origins, random selection reproducibility, sampling,
generation-token PPL and the original full-response selection. No stream sees
evaluation targets. Only after all complete 400-item streams pass their audits
may the isolated labels be opened, and the same pinned Meta MMF raw-answer
scorer must be applied to every condition. Report paired effects, format
failures, actual token costs, gate coverage and stream position, and state
whether the compared budgets are within 5%. EvalAI server parity has not been
verified. Different stream histories and devices are possible confounders;
multiple orderings, another model and another dataset are needed before
generalization or conference-level claims.
