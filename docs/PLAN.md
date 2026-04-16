# PJ1 Execution Plan

Deadline: 2026-05-10 23:59.

## Task 1 - Image-Text Retrieval

Goal: compare contrastive, matching-based, and projection/query-based alignment
models on COCO val2017 under one retrieval protocol.

Models:

- Contrastive: LAVIS `clip_feature_extractor:base` or OpenCLIP.
- Matching-based: LAVIS `blip_retrieval:coco`.
- Projection/query-based: LAVIS `blip2_feature_extractor:pretrain`.

Protocol:

1. Load COCO val2017 images and `captions_val2017.json`.
2. Extract one image embedding per image.
3. Extract one text embedding per caption.
4. Compute image-text similarity by dot product on normalized embeddings.
5. Evaluate Text-to-Image and Image-to-Text Recall@1, Recall@5, Recall@10.
6. Save per-model JSON and aggregate CSV results under `outputs/task1_retrieval/results/`.

Implementation notes:

- Cache embeddings under `outputs/task1_retrieval/cache/` because model inference is expensive.
- Keep the evaluator independent from LAVIS so metric logic can be unit-tested without GPUs.
- Run smoke tests with `--max-images` before full COCO val2017 evaluation.
- Record commands and observations in `docs/WORKLOG.md`.
- The initial protocol uses final image/text embeddings for all model families.
  BLIP ITM reranking can be added later as a separate protocol to avoid mixing
  embedding retrieval and reranked retrieval in one result table.

Current environment note:

- Use one shared conda environment for all three tasks: `multimodal`.
- Maintain one shared Python dependency file at `requirements.txt`.
- Install PyTorch separately by target device:
  - Linux CUDA 12.4: official cu124 wheels.
  - macOS Apple Silicon: default PyTorch wheels plus MPS fallback.
  - CPU-only: official CPU wheels.
- On macOS arm64, install `decord` from conda-forge before pip installing
  `requirements.txt`, because pip has no compatible `decord` wheel for this
  platform.
- Task 1 sets model cache directories under `outputs/task1_retrieval/` so model
  downloads stay outside the submission archive.
- `scripts/check_environment.py` is the standard pre-run health check.
- `scripts/setup_env.sh` is the standard environment bootstrap script.

## Task 2 - Image Captioning

Working directory: `tasks/task2_captioning/`.

Planned scope:

- Compare BLIP and BLIP-2 caption generation on a fixed COCO val2017 subset or full split.
- Evaluate BLEU-4 and CIDEr at minimum.
- Add METEOR, ROUGE-L, and SPICE if `pycocoevalcap` dependencies are available.

Implementation plan:

1. Use LAVIS caption checkpoints:
   - `lavis:blip_caption:base_coco`
   - `lavis:blip2_opt:caption_coco_opt2.7b`
2. Keep one caption per image under a shared decoding protocol:
   - default `num_beams=5`
   - `max_length=30`
   - `min_length=1`
3. Save generated captions to `outputs/task2_captioning/predictions/`.
4. Evaluate with `pycocoevalcap` scorers through a local wrapper so evaluation
   does not depend on `pycocotools`.
5. Record warnings when Java-based tokenization or optional scorers are
   unavailable.
6. Keep Task 2 runtime flags aligned with Task 1:
   - `--local-files-only`
   - `--hf-endpoint`
   - `--bert-tokenizer-path`

## Task 3 - Representation Analysis

Working directory: `tasks/task3_representation/`.

Planned scope:

- Visualize aligned image/text embedding spaces with PCA, t-SNE, or UMAP.
- Run nearest-neighbor case studies for image-to-text and text-to-image.
- Add optional compositional generalization cases.

## Submission

Use `scripts/package_submission.py` to create a submission archive in `handin/`.
The script excludes datasets, caches, logs, local environments, and model weights.
