# Task 1 - 图文检索

本目录用于追踪 Task 1 的工作状态。运行产物写入
`outputs/task1_retrieval/`。

## 本任务目标

- Text-to-Image Retrieval
- Image-to-Text Retrieval
- Recall@1 / Recall@5 / Recall@10

## 标准运行顺序

### 1. 环境检查

```bash
conda run -n multimodal python scripts/check_environment.py --check-data
```

### 2. Dry-run

```bash
conda run -n multimodal python code/pj1/task1/run_retrieval.py --dry-run --max-images 10
```

### 3. Smoke test

```bash
conda run -n multimodal python code/pj1/task1/run_retrieval.py \
  --max-images 20 \
  --image-batch-size 4 \
  --text-batch-size 32 \
  --eval-batch-size 128 \
  --model-spec lavis:clip_feature_extractor:base
```

### 4. 全量运行

```bash
conda run -n multimodal python code/pj1/task1/run_retrieval.py \
  --model-spec lavis:clip_feature_extractor:base \
  --model-spec lavis:blip_retrieval:coco \
  --model-spec lavis:blip2_feature_extractor:pretrain
```

## 结果位置

- `outputs/task1_retrieval/results/<run_name>.json`
- `outputs/task1_retrieval/results/summary.csv`
