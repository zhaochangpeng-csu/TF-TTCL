# Faithful TF-TTCL direct transfer: TextVQA train open-task protocol

This is a fresh **official TextVQA v0.5.1 train** development stream. The
previously scored 5,000-question validation split is not a new confirmation
set. Choose 400 train questions by SHA256 ranking with seed `20261020`, and
extract only their images. Exclude any image present in the prior validation
export. The official annotation and validated images are required locally;
no GPU service should start if either is absent.

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

The managed server cannot access the public internet directly. Its network
guide specifies a SOCKS5h proxy; pass it **per download command**, without
changing global proxy settings. The official annotation was obtained through
that proxy and validated as 34,602 train questions over 21,953 images.
The 7.07 GB official train/validation ZIP supports HTTP byte ranges. Fetch
only the selected images, checking each against the archive's CRC and size:

```bash
python scripts/fetch_textvqa_train_direct_images.py \
  --annotation data/multimodal/textvqa_official_source/TextVQA_0.5.1_train.json \
  --output-dir data/multimodal/textvqa_official_source/train_subset_images \
  --proxy socks5h://122.207.108.6:51080 --workers 4
```

Import the verified selected images only after the fetch finishes:

```bash
python scripts/prepare_textvqa_train_direct.py \
  --annotation data/multimodal/textvqa_official_source/TextVQA_0.5.1_train.json \
  --image-dir data/multimodal/textvqa_official_source/train_subset_images/images \
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

The GPU index defaults to GPU 3; `GPU_INDEX=1` (or another idle card) is
allowed under the user's updated authorization. The launcher refuses an
occupied GPU or already-running local service, owns and stops only its model
process group, and leaves other GPU processes untouched.

At first, direct access to official and mirror endpoints timed out. The
server-network guide resolved this: the official annotation is now present,
and selective image acquisition is the remaining data step. No fresh TextVQA
train result is claimed until the images, four-arm run, and audit complete.
