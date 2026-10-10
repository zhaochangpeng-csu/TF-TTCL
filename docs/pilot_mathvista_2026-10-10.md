# MathVista engineering pilot — 2026-10-10

## Scope and validity

These are engineering diagnostics on the first 20 MathVista testmini samples,
not paper results. Labels were inspected after each completed run to diagnose
failure modes, so this slice is permanently excluded from prompt selection,
threshold tuning, formal validation, and final statistical claims.

All runs used Qwen2.5-VL-7B-Instruct as the frozen generator. Runs with memory
used Qwen3-VL-Embedding-2B. The corrected MathVista prompt and type-aware
official-style evaluator were used for all results listed below.

## Results

| Variant | Accuracy | Memory writes | Diagnostic conclusion |
|---|---:|---:|---|
| Zero-shot | 15/20 (75%) | 0 | Corrected reference baseline |
| Evidence selection, no memory | 15/20 (75%) | 0 | Extra verification did not improve this slice |
| Token-budget-matched majority | 15/20 (75%) | 0 | Same answers as zero-shot despite much larger budget |
| Initial EviMem | 10/20 (50%) | 10 | Unrelated episode retrieval caused severe pollution |
| Strict evidence gate + similarity floors | 14/20 (70%) | 4 | Removed four of five baseline-to-error regressions |
| Clean/memory-conditioned dual path | 15/20 (75%) | 4 | Recovered baseline aggregate without a gain |

The initial failure is informative rather than a positive result: forcing a
top-1 episode when no relevant memory exists propagates errors. The remediation
requires every verifier claim to be entailed, applies independently calibrated
raw-similarity floors, and leaves three of six candidate paths free of memory
conditioning. These safeguards were fixed from a synthetic red-square versus
blue-circle fixture, not from benchmark labels.

The dual-path run is stored on the experiment server at
`core/exp_res/mm/mathvista/qwen25vl/evimem/seed0_pilot20_dual`. Its manifest is
complete at Git SHA `4de19d8`, with 91,868 prompt tokens, 6,682 completion
tokens, and 92.96 seconds of summed request latency. Equal aggregate accuracy
does not establish improvement. A strict paired comparison subsequently found
the same per-sample correctness vector as zero-shot: 0 wins, 0 losses, 20 score
ties, delta 0.00 with bootstrap CI `[0.00, 0.00]`, and paired-randomization
`p=1.0`. This is parity on the diagnostic slice, not a positive result.

The token-budget-matched run used 100,779 prompt and 7,985 completion tokens
(114.21 seconds summed request latency). It produced exactly the same 20
answers as zero-shot. EviMem versus this control again had 0 wins and 0 losses;
only sample 1 changed (`0.0` to `0.5`), with both answers scored incorrect.

Candidate-oracle analysis found no selection headroom on either the
budget-matched or dual-path run: selected accuracy and the six-candidate oracle
were both 75%. Ten dual-path samples and eleven budget-matched samples had only
one unique answer across all six trajectories; the mean was 1.5 unique answers.
Thus none of the five remaining errors had a correct candidate available for
the evidence selector. Candidate recall, not Borda weighting, is the immediate
bottleneck on this slice.

## Evaluation correction

The first analysis pass reported 70% for the four non-polluted variants. A
regression audit found that the option parser treated the final `a` in
`Common water flea` as option A, so a correct option-B response was scored
incorrectly. The parser is now anchored to a standalone option letter at the
start of the extracted answer, and offline evaluation consumes the runner's
already-normalized `answer` field. All figures in the table above are the
recomputed values; the earlier 70% figures are superseded.

## Decision rule

Do not expand the full EviMem matrix from this pilot. First establish that the
evidence selector beats a token-budget-matched baseline on an untouched
development stream. If selection works but online memory does not add a stable
gain, the defensible paper scope becomes unsupervised multimodal evidence
selection rather than evidence memory.
