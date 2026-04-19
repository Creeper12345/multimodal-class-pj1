# Task 3 - 表征分析

这是 Task 3 的工作目录。

## 本任务目标

- 对图像与文本 embedding 做 PCA、t-SNE 或 UMAP 可视化
- 进行 nearest-neighbor case study
- 分析 hard negatives 与组合泛化失败案例

## 代码入口

- `code/pj1/task3/run_representation_analysis.py`
  - 读取 Task 1 embedding cache
  - 生成 PCA / t-SNE / UMAP 可视化
  - 输出 image-to-text 和 text-to-image nearest-neighbor 案例
- `code/pj1/task3/text_to_image_search.py`
  - 支持输入自由文本，基于指定模型 cache 检索最相近图片
- `code/pj1/task3/analysis.py`
  - Task 3 的统计、降维、近邻分析 helper

## 前置条件

Task 3 默认复用 Task 1 的 embedding cache，因此先确保存在：

```text
outputs/task1_retrieval/cache/
```

如果没有 cache，需要先运行 Task 1。

## 表征可视化与案例分析

只分析指定模型：

```bash
conda run -n multimodal python code/pj1/task3/run_representation_analysis.py \
  --run-name lavis_clip_feature_extractor_ViT-B-32_nall_first_first_c1e57b48 \
  --run-name lavis_blip_feature_extractor_base_nall_first_first_69e93c04 \
  --run-name lavis_blip_retrieval_coco_nall_first_first_2abfcb45 \
  --run-name lavis_blip2_feature_extractor_pretrain_nall_first_first_91ffb98f \
  --sample-size 500 \
  --method pca
```

如果要补 t-SNE：

```bash
conda run -n multimodal python code/pj1/task3/run_representation_analysis.py \
  --sample-size 500 \
  --method pca \
  --method tsne
```

输出位置：

- `outputs/task3_representation/task3_representation_analysis.md`
- `outputs/task3_representation/figures/*.png`
- `outputs/task3_representation/results/*_stats.json`

## 文本检索图片

示例：

```bash
CUDA_VISIBLE_DEVICES=1 python code/pj1/task3/text_to_image_search.py \
  --local-files-only \
  --bert-tokenizer-path outputs/task1_retrieval/hf_cache/manual/bert-base-uncased \
  --run-name lavis_blip_retrieval_coco_nall_first_first_2abfcb45 \
  --query "two dogs playing with each other in the grass" \
  --top-k 10
```

说明：

- `--run-name` 必须对应已有 Task 1 cache。
- 脚本会加载同一模型的文本编码器，对 query 编码后检索 cached image embeddings。
- 输出包含图片文件名、COCO image id、相似度分数和参考 caption。
