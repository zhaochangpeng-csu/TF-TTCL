# TextVQA fixed-candidate answer-similarity diagnostic

This is a post-hoc exploratory analysis of the 400-item TextVQA train stream
in [the format/memory experiment](textvqa_format_factorial_v2_rescoring.md).
The original open-task selector ranks the four students by similarity of their
complete responses to the Teacher's complete response. This replay instead
uses the stored similarities of their normalized *answers*. The Teacher and
top two students remain the positive group, and the lowest generated-token PPL
in that group determines the selected response. No model is called; candidate
responses, their PPL, and the historical memory remain fixed.

| Prompt | Candidate source | Original score | Answer-similarity replay | Paired difference |
| --- | --- | ---: | ---: | ---: |
| Original | No memory | 72.28% | 74.88% | +2.60 pp |
| Original | Recorded memory | 65.35% | 67.43% | +2.08 pp |
| Short final | No memory | 79.40% | 80.25% | +0.85 pp |
| Short final | Recorded memory | 71.58% | 73.13% | +1.55 pp |

Scores use the pinned Meta MMF evaluator and raw extracted selected answers,
as in the companion report; exact EvalAI server parity is unverified. The
original/no-memory difference has paired bootstrap 95% CI +0.80 to +4.53 pp
(15 items improved, 4 worsened); short/no-memory has CI -1.03 to +2.83 pp
(13 improved, 9 worsened). Other intervals and provenance are in the
[aggregate JSON](textvqa_fixed_candidate_similarity_replay.json). These are
developmental, post-hoc comparisons across multiple contrasts.

Recorded answer similarities were produced during the initial experiment;
the replay verifies their candidate identities but does not recompute their
embeddings. In memory arms, changing the selected answer at earlier samples
would also change rules and later candidates in a true online experiment.
Thus this output-only diagnostic does not establish the behavior of an
answer-similarity selector with independently rebuilt memory. Confirmation
requires a frozen implementation and previously unused images and questions.
