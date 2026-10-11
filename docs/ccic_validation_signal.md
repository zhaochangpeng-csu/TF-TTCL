# CCIC: Contrastive Causal Identifiability Certificate

## Why the previous certificate failed

The ScienceQA train pilot falsified the previous validation signal. Blank and
mismatched images measured whether an answer depended on an image, but did not
measure whether the image supported the *identity* of that answer. On sample
6514, a wrong place name was stable on the real and fidelity views and lost
support on counterfactual views. The certificate therefore validated a
correlated recognition error.

The replacement signal must distinguish three properties:

1. **Relevance:** the answer changes when visual evidence is removed.
2. **Sufficiency:** the answer survives when its proposed evidence is focused.
3. **Contrastive identifiability:** the evidence for two competing answers is
   spatially separable, and interventions selectively exchange their answers.

Only the third property can block the observed failure. If two place names both
refer to the same highlighted map region, the model has shown visual dependence
but has not identified which name is correct. CCIC abstains.

## Strict v1 protocol

CCIC is invoked only when the original greedy answer `g` and the strongest
generated alternative `a` disagree.

1. Treat the first image as a fixed 4x4 grid.
2. In a question-only call that contains neither answer options nor generated
   candidates, ask for the required cells `R_q`.
3. In a separate candidate-conditional call, ask for cells required by `g` and
   `a`. These calls propose regions rather than verdicts.
4. Repeat both calls on a lossless fidelity view and swap candidate order in the
   conditional call. Require all three cell sets to match exactly after
   restoring answer identity.
5. Require `R_g` and `R_a` to be disjoint. Require `R_q` to intersect `R_a` but
   not `R_g`. These exact set tests replace tunable score thresholds.
6. Construct four executable views while preserving image dimensions:
   `focus(R_a)`, `drop(R_a)`, `focus(R_g)`, and `drop(R_g)`. Focus dims the
   surrounding image but preserves global topology; drop replaces selected
   cells with the image mean color.
7. Re-answer the original question directly. The alternative is accepted only
   under the complete crossover:

   ```text
   answer(focus(R_a)) = a       answer(drop(R_a)) = g
   answer(focus(R_g)) = g       answer(drop(R_g)) = a
   ```

Any malformed localization, position sensitivity, shared evidence region,
question-region ambiguity, or missing crossover returns the original greedy
answer. CCIC never writes memory in this phase.

The `causal_budget` arm performs every adaptive call and image transformation
but discards the CCIC decision. `causal_evidence` may apply the decision. This
keeps the causal comparison budget matched.

## What is and is not the research claim

Masking a cited region is not novel by itself. EVICT already applies
counterfactual evidence dropout and explicitly targets a high-precision,
low-coverage regime. Active-Look also reports that zooming loses global context
and that incorrect region proposals can be over-trusted; this motivates the
full-context focus view and order-swapped stability check. Pelican uses claim
decomposition and external tools, while Visual Evidence Prompting uses smaller
specialist visual models; those are outside the no-external-teacher setting.

The hypothesis to test is narrower: **candidate disagreement is safely
actionable only when a frozen black-box VLM exhibits answer-specific causal
evidence separation, not merely evidence sensitivity.** No novelty or priority
claim should be made until the broader literature search and experiments are
complete.

Primary references:

- [EVICT, CVPRW 2026](https://openaccess.thecvf.com/content/CVPR2026W/GRAIL-V/html/Kotte_EVICT_Evidence-Sufficiency_Verification_via_Counterfactual_Dropout_for_Visually-Grounded_Selective_Question_CVPRW_2026_paper.html)
- [Active-Look, Findings of ACL 2026](https://aclanthology.org/2026.findings-acl.745/)
- [Vision-Language Introspection, ACL 2026](https://aclanthology.org/2026.acl-long.1784/)
- [Pelican, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.470/)
- [Visual Evidence Prompting, ACL 2025](https://aclanthology.org/2025.acl-long.205/)

## Pre-registered next experiment

Do not tune CCIC on the already opened dev400 labels. Create a disjoint 400-row
slice from the same deterministic ranking with `--offset 400`. Run a 20-row
label-free smoke test, then finish paired `causal_budget` and `causal_evidence`
runs on all 400 rows before opening the new label file.

Report accuracy, paired wins/losses, gate coverage, localization failure modes,
complete-crossover rate, token/latency cost, and selective precision among
changed answers. Continue to ScienceQA validation only if `causal_evidence`
beats both greedy and `causal_budget`, with zero or very few harmful overrides.

On the experiment server, the paired label-free smoke test is launched from the
`ttt` environment with:

```bash
bash scripts/run_scienceqa_causal_smoke.sh
```

The script refuses to run while GPU 2 has any compute process. It starts its own
port-9000 Qwen2.5-VL service, runs the two matched 20-row arms, writes
`protocol_audit.json`, and terminates only the service process it created on
success, failure, or interruption. It never opens the label file.

