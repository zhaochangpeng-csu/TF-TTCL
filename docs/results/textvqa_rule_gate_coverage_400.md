# Lexical reasoning-rule gate: label-free coverage inspection

This is a coverage check on retrieved rules in the previously generated
400-item short-final TextVQA train development stream. It uses sample IDs and
rule text only; it does not open labels, access candidate answer values, or
compute correctness. The full aggregate and examples are in
[the JSON report](textvqa_rule_gate_coverage_400.json), including the source
predictions SHA-256.

The frozen prototype examines only the first sentence. It matches a small set
of action words followed within 60 characters by `reasoning`, `explanation`,
or `explaining`, plus direct negated `explain` statements. It conservatively
excludes obvious reversals such as “do not skip reasoning”, “avoid ... without
reasoning”, and cases about reasoning errors or repetition. The exact patterns
and exclusions are in `core/multimodal/textvqa_rule_gate.py`; its SHA-256 at
inspection was `3fda87925ec31be621fa27c71bdeafa9eb2c2adb57136a686a68f54fc1c4f4e2`.

Across 400 questions and 23,070 retrieved rule instances, 358 instances matched
the gate, corresponding to 13 of 546 unique retrieved rules. At least one
matching rule was retrieved for 232 questions. All matched instances were in
the negative-rule bucket. The same rule may occur in many questions, so these
counts are not independent observations. A targeted neutral substitution can
be compared with a deterministic same-count, same-bucket random substitution;
the audit records rule IDs, positions, selected control seed and tokenizer
lengths. The substitution has not been evaluated by this coverage report.

This gate is a **lexical heuristic**, not a semantic conflict detector or a
claim that matched rules are false. Even “avoid unnecessary reasoning” can be
compatible with providing the required reasoning, depending on context. The
coverage check establishes neither filter precision nor answer improvement.
Inspection informed the conservative exclusions on a previously used
development stream; evaluate the frozen heuristic on a new disjoint stream
before drawing any generalization conclusion.

The same frozen gate was also applied, without opening targets or accessing
candidate answer values, to the two completed selector streams on the newer
400-item train subset. In the full-response-selector stream it matched 1,041
of 23,070 retrieved instances, comprising 24 of 538 unique retrieved rules,
and appeared in at least one retrieved rule for 383/400 questions. In the
normalized-answer-selector stream it matched 604 of 23,070 instances,
comprising 14 of 553 unique rules, and appeared for 312/400 questions. All
matched instances were in negative buckets. Source prediction file hashes,
pattern counts and examples are in [the full-response coverage](textvqa_rule_gate_coverage_selector_full_400.json)
and [the normalized-answer coverage](textvqa_rule_gate_coverage_selector_answer_400.json).
These streams had been scored before the check, so their coverage is a
descriptive inspection, not a blinded accuracy experiment.
