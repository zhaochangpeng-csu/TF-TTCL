# TextVQA selector comparison on a disjoint online development stream

The [frozen protocol](textvqa_selector_fresh_online_protocol.md) ran on 400
previously unused TextVQA train questions and images. Both independent online
streams completed with 400 items and passed label-free audits before scoring;
the GPU services exited before the CPU audits and scoring. This remains one
development stream, one ordering and one model.

| Condition | Meta MMF score |
| --- | ---: |
| Greedy, no memory | 82.05% |
| Full-response similarity, no memory | 79.80% |
| Answer similarity, no memory | 81.25% |
| Full-response similarity, independent online memory | 70.18% |
| Answer similarity, independent online memory | 72.03% |

For exactly the same no-memory candidates, answer similarity gained 1.45 pp
over full-response similarity: paired bootstrap 95% CI +0.05 to +2.98 pp,
paired randomization p=0.0551, with 12 items improved and 3 worsened. It
remained 0.80 pp below greedy. For independently rebuilt online memory, the
answer selector gained 1.85 pp, but its CI was -1.28 to +5.10 pp and p=0.248;
the memory result is not conclusive. Relative to its own no-memory condition,
answer-similarity memory lost 9.23 pp, CI -12.80 to -5.65 pp. These tests are
descriptive for one ordered stream, not confirmation of generalization.

Logical prompt-plus-completion token cost was identical for the two no-memory
conditions and 1.71% higher for answer-similarity memory; both comparisons met
the within-5% budget threshold. Two separate HCU devices were used, so the
online-policy and device are confounded. For the memory conditions the exact
`Final answer:` marker was absent in 151/400 and 187/400 selected outputs,
respectively, versus 1/400 in each no-memory condition. The metric is the
pinned Meta MMF scorer on raw extracted answers; direct parity with the EvalAI
server remains unverified.

The complete score, intervals, hashes and token costs are in
[the aggregate result](textvqa_selector_fresh_online_400_mmf.json). The
answer-similarity selector does not yet exceed greedy, and no memory gain has
been established. Subsequent analysis should identify which historical rules
and output-format failures cause the loss before claiming a reusable method.
