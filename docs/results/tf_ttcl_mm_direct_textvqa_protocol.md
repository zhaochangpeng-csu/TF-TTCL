# Faithful TF-TTCL direct transfer: TextVQA train open-task protocol

This is a fresh **official TextVQA v0.5.1 train** development stream. The
previously scored 5,000-question validation split is not a new confirmation
set. Choose 400 train questions by SHA256 ranking with seed `20261020`, and
extract only their images. Exclude any image present in the prior validation
export. The official annotation and image ZIP are required locally; no GPU
service should start if either is absent.

The four paired arms are greedy, five-candidate no-rewrite selection,
rewritten selection without memory, and full positive/negative text-rule
memory. Unlike ScienceQA, the open-task selector ranks the four Student
**full response texts** by similarity to the Teacher's full response text,
puts the top two with the Teacher, then selects the minimum-generation-PPL
representatives. This follows `core/source/compare/selection_kernel.py`.
The original open-task positive/negative rule templates are used rather than
the ScienceQA closed-task templates. Short final-answer similarities are recorded separately
for diagnosis; they do not change selection. No image augmentation, external
teacher, label-conditioned rule, or extra answer refinement is allowed.

Import only after both official files are available:

```bash
python scripts/prepare_textvqa_train_direct.py \
  --annotation /path/to/TextVQA_0.5.1_train.json \
  --image-zip /path/to/train_val_images.zip \
  --previous-val data/multimodal/textvqa_val \
  --output-dir data/multimodal/textvqa_train_tf_ttcl_mm_direct
```

In the server's `ttt` environment, first run
`TF_TTCL_MM_RUN_KIND=smoke20 bash scripts/run_tf_ttcl_mm_direct_textvqa.sh`.
Inspect the unscored casebook, PPL vectors, open-task similarity rankings,
short-answer disagreement diagnostic and answer format without opening any
labels. If the audit passes, run the same launcher with
`TF_TTCL_MM_RUN_KIND=dev400`. Only after all four arms complete and the
label-free audit passes should
`evaluate/multimodal/compare_tf_ttcl_mm_direct.py` open the isolated labels.
Report VQA accuracy, paired score differences, actual token/time cost, and
how often full-response and short-answer top-two partitions disagree.

The GPU index defaults to user-authorized GPU 3. The launcher refuses an
occupied GPU or already-running local service, owns and stops only its model
process group, and leaves other GPU processes untouched.

At preparation time on 2026-10-19, the server and local workspace lacked the
official train annotation and image archive. The server's official and known
mirror endpoints timed out. No fresh TextVQA train result is claimed until
those assets are obtained and audited.
