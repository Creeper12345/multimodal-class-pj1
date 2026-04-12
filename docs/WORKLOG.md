# Work Log

## 2026-04-08

- Read `PJ1(1).pdf` and extracted project requirements.
- Confirmed COCO val2017 images and caption annotations are present locally:
  - `datasets/val2017/`
  - `datasets/annotations/captions_val2017.json`
- Confirmed the directory is not a git repository.
- Confirmed current Python environment has `numpy` and `PIL`, but does not have
  `torch`, `lavis`, or `transformers`.
- Created task working directories:
  - `tasks/task1_retrieval/`
  - `tasks/task2_captioning/`
  - `tasks/task3_representation/`
- Implemented the Task 1 plan and repository-level ignore/submission rules.
- Implemented Task 1 COCO loading, model adapters, Recall@K evaluator, and CLI.
- Added submission packaging script.
- Ran lightweight validation:
  - `python3 -m compileall code scripts`
  - `python3 code/pj1/task1/run_retrieval.py --dry-run --max-images 10`
  - Metric self-check with identity embeddings.
  - Test packaging to `handin/pj1_submission_test.zip`.
- Checked local conda environments:
  - `base`: Python 3.13, no `torch`, no `lavis`.
  - `llm-26-cpu`: Python 3.12, `torch` available, no `lavis`.
  - `interviewdemo`: Python 3.10, `torch` and `transformers` available, no `lavis`.
- Full LAVIS evaluation remains pending until `salesforce-lavis` is installed in
  a compatible environment.
- Created shared conda environment `multimodal` with Python 3.10 and pip.
- Replaced task-specific requirements with one shared `requirements.txt` for
  Task 1, Task 2, and Task 3.
- Initial `pip install -r requirements.txt` failed because `salesforce-lavis`
  requires `decord`, and pip has no macOS arm64 `decord` wheel.
- Installed `decord` into `multimodal` via conda-forge.
- Re-ran `pip install -r requirements.txt`; installation completed successfully.
- Pinned key package versions in `requirements.txt` to avoid long pip resolver
  backtracking on future installs.
- Found macOS OpenMP duplicate runtime errors when importing torch from the mixed
  conda/pip environment; added a guarded macOS `KMP_DUPLICATE_LIB_OK` default in
  the Task 1 model adapter.
- Redirected HuggingFace and Torch model caches to `outputs/task1_retrieval/`
  runtime directories.

## 2026-04-09

- Re-validated the shared environment in `multimodal`:
  - `conda run -n multimodal python -m compileall code scripts`
  - `conda run -n multimodal python code/pj1/task1/run_retrieval.py --dry-run --max-images 10`
- Verified imports for `torch`, `transformers`, `lavis`, `open_clip`, and
  `pycocoevalcap`.
- Fixed the LAVIS adapter so `lavis:clip_feature_extractor:base` works when
  `extract_features()` returns a plain tensor instead of a feature object.
- Completed a real 1-image smoke test for
  `lavis:clip_feature_extractor:base`.
- Attempted a real 1-image smoke test for `lavis:blip_retrieval:coco`; the
  code path reached model initialization, but the run was stopped before full
  download completion because local storage is tight.
- Observed a partial checkpoint left by the interrupted BLIP retrieval download:
  `outputs/task1_retrieval/torch_cache/hub/checkpoints/blip_coco_retrieval.pth.fce9eb8ff0174252be2ede8a05115a88.partial`
  at about 2.7 GB.

## 2026-04-12

- Refactored environment configuration so device-independent Python packages now
  live in `requirements.txt`, while PyTorch installation is chosen separately by
  backend.
- Added `environment.yml` as the base conda environment definition.
- Added `code/pj1/runtime.py` to centralize runtime cache paths and platform
  compatibility settings.
- Added `scripts/check_environment.py` for pre-run runtime validation.
- Added `scripts/setup_env.sh` for `cuda124`, `cpu`, and `mps` installation
  paths.
- Updated root `README.md` and task README files to align with the server-run
  workflow.
- Added Chinese operator-facing docs:
  - `docs/运行说明.md`
  - `docs/项目说明.md`
  - `docs/执行工作流.md`
- Re-validated the updated runtime wiring with:
  - `conda run -n multimodal python -m compileall code scripts`
  - `conda run -n multimodal python scripts/check_environment.py --check-data`
  - `conda run -n multimodal python code/pj1/task1/run_retrieval.py --dry-run --max-images 10`

## Next Steps

- Decide the storage budget for model checkpoints before resuming BLIP / BLIP-2
  downloads.
- Remove the partial BLIP retrieval checkpoint if disk space is needed before
  the next run.
- Resume 1-image smoke tests for `lavis:blip_retrieval:coco` and
  `lavis:blip2_feature_extractor:pretrain`.
- Run the full COCO val2017 evaluation for the selected model set.
