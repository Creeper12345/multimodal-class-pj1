# Improvement Log

## Task 1

- The evaluator is model-agnostic and operates directly on embedding matrices.
- Embedding extraction is cached per model spec, dataset size, and pooling mode.
- The CLI supports `--max-images` for cheap smoke tests before full evaluation.
- Runtime artifacts are isolated under `outputs/task1_retrieval/`.
- Submission packaging excludes raw COCO data, embedding caches, logs, and model weights.
- Runtime configuration is centralized in `code/pj1/runtime.py`.
- Environment validation is exposed via `scripts/check_environment.py`.
- Installation now distinguishes CUDA 12.4, CPU, and macOS MPS paths.

## Future Improvements

- Add BLIP retrieval ITM reranking as a separate, clearly labeled protocol.
- Add chunked on-disk similarity computation if memory becomes tight on full COCO.
- Add automated report table generation from result JSON files.
- Add Task 2 and Task 3 result templates once those tasks start.
- Add a server-side smoke-test script that checks one sample per model and writes logs.
