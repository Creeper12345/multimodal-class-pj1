# PJ1 多模态对齐实验

本仓库用于完成课程 PJ1，当前优先实现并验证 Task 1 图文检索，后续扩展到
Task 2 图像描述生成与 Task 3 表征分析。

## 先看这些文档

- [运行说明](docs/运行说明.md)
- [服务器部署说明](docs/服务器部署说明.md)
- [项目说明](docs/项目说明.md)
- [执行工作流](docs/执行工作流.md)
- [工作日志](docs/WORKLOG.md)
- [执行计划](docs/PLAN.md)

## 当前代码状态

- `code/pj1/task1/` 已实现 COCO val2017 图文检索主流程。
- `scripts/check_environment.py` 用于运行前环境自检。
- `scripts/setup_env.sh` 用于创建 `multimodal` 环境并安装依赖。
- `scripts/package_submission.py` 用于整理提交压缩包。

## 最短启动路径

### 1. 创建环境

Linux CUDA 12.4 服务器：

```bash
bash scripts/setup_env.sh multimodal cuda124
```

macOS Apple Silicon：

```bash
bash scripts/setup_env.sh multimodal mps
```

### 2. 环境检查

```bash
conda run -n multimodal python scripts/check_environment.py --check-data
```

### 3. Task 1 数据 dry-run

```bash
conda run -n multimodal python code/pj1/task1/run_retrieval.py --dry-run --max-images 10
```

### 4. 预下载 Task 1 模型

```bash
conda run -n multimodal python scripts/prefetch_task1_models.py
```

### 5. Task 1 正式运行

```bash
conda run -n multimodal python code/pj1/task1/run_retrieval.py \
  --model-spec lavis:clip_feature_extractor:base \
  --model-spec lavis:blip_retrieval:coco \
  --model-spec lavis:blip2_feature_extractor:pretrain
```

如果模型已经预下载完成，并且你希望后续严格只读本地缓存：

```bash
conda run -n multimodal python code/pj1/task1/run_retrieval.py \
  --local-files-only \
  --model-spec lavis:clip_feature_extractor:base \
  --model-spec lavis:blip_retrieval:coco \
  --model-spec lavis:blip2_feature_extractor:pretrain
```

结果会写到 `outputs/task1_retrieval/results/`。
