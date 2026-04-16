# Task 1 补充实验设计与模型说明

## 1. 为什么当前比较不完全公平

当前已经完成的一组结果是：

- `clip_feature_extractor:base`
- `blip_retrieval:coco`
- `blip2_feature_extractor:pretrain`

这组比较有价值，但严格来说不完全公平，原因在于三者的训练目标和使用方式并不一致：

1. `blip_retrieval:coco` 是**专门面向检索任务并在 COCO/Flickr 检索设定下微调过的模型**。
2. `blip2_feature_extractor:pretrain` 是**预训练特征提取器**，并不是检索专门微调版本。
3. `clip_feature_extractor:base` 是**通用 contrastive 对齐模型**，不是专门的 COCO retrieval 微调模型。

所以当前结果更像是在比较：

- 一个通用 contrastive baseline
- 一个检索专门模型
- 一个通用 BLIP-2 特征模型

这能说明“任务专门化训练”的效果，但不能单独作为“架构本身孰优孰劣”的结论。

## 2. 三类模型的特色

### 2.1 CLIP

代表接口：

- `lavis:clip_feature_extractor:base`

特点：

- 用大规模图文对做 contrastive learning
- 学到统一的图像和文本 embedding 空间
- 推理简单，速度快，工程上最干净

优点：

- 训练和推理机制直接
- 适合作为强 baseline
- 零样本泛化通常不错

局限：

- 更偏全局语义对齐
- 对属性词、关系词、计数词等细粒度差异不一定敏感
- 不具备显式的 matching / reranking 结构

### 2.2 BLIP Retrieval

代表接口：

- `lavis:blip_retrieval:coco`

特点：

- 针对图文检索设计
- 包含图像编码器、文本编码器、投影层和匹配机制
- 训练目标更贴近 retrieval

优点：

- 在检索任务上通常最强
- 对 caption 和图像的对齐更细
- 适合做 Task 1 的“检索专门模型”代表

局限：

- 速度比 CLIP 慢
- 结果更依赖任务微调设定
- 如果拿它和纯预训练 feature extractor 比，结论容易掺入“训练协议差异”

### 2.3 BLIP-2 Feature Extractor

代表接口：

- `lavis:blip2_feature_extractor:pretrain`
- 可补充尝试 `lavis:blip2_feature_extractor:coco`

特点：

- 使用视觉编码器 + Q-Former 做跨模态桥接
- 比 BLIP 一代更强调查询 token 和模块化连接
- 既能服务检索表征，也能作为后续生成模型的前端

优点：

- 架构上更灵活
- 与后续 caption / VQA / LLM 连接更自然
- 适合后续 Task 2 和 Task 3 延展

局限：

- 如果只用 `pretrain` 版本做检索，不一定比 retrieval-tuned 模型强
- 需要区分“特征提取器版本”和“生成版本”
- 在当前 Task 1 协议下，`pretrain` 不应直接和 `blip_retrieval:coco` 做单点胜负判断

## 3. 更公平的补充实验

建议补两套协议。

### 协议 A：统一 feature extraction 协议

目标：尽量控制“都是通用预训练特征提取器”，减少任务专门微调带来的偏置。

建议模型：

- `lavis:clip_feature_extractor:ViT-B-32`
- `lavis:blip_feature_extractor:base`
- `lavis:blip2_feature_extractor:pretrain`

这组对比更适合回答：

- 在都不做 retrieval 专门微调时，三类表征谁更适合作为统一 embedding 空间？

推荐命令：

```bash
python code/pj1/task1/run_retrieval.py \
  --local-files-only \
  --bert-tokenizer-path outputs/task1_retrieval/hf_cache/manual/bert-base-uncased \
  --model-spec lavis:clip_feature_extractor:ViT-B-32 \
  --model-spec lavis:blip_feature_extractor:base \
  --model-spec lavis:blip2_feature_extractor:pretrain
```

### 协议 B：任务适配能力对比

目标：看“面向 COCO retrieval 的任务适配”是否真的带来收益。

建议模型：

- `lavis:clip_feature_extractor:ViT-B-32`
- `lavis:blip_retrieval:coco`
- `lavis:blip2_feature_extractor:coco`

说明：

- 这里 `CLIP` 仍然是通用 baseline
- `BLIP retrieval:coco` 和 `BLIP-2 feature extractor:coco` 更接近“任务适配版本”

推荐命令：

```bash
python code/pj1/task1/run_retrieval.py \
  --local-files-only \
  --bert-tokenizer-path outputs/task1_retrieval/hf_cache/manual/bert-base-uncased \
  --model-spec lavis:clip_feature_extractor:ViT-B-32 \
  --model-spec lavis:blip_retrieval:coco \
  --model-spec lavis:blip2_feature_extractor:coco
```

## 4. BLIP-2 pooling 消融

这是当前最值得补的一项小规模消融。

原因：

- `BLIP-2 feature extractor` 返回的是 query-based image features
- 最终怎么把它压成检索向量，会显著影响结果
- 当前 `pretrain` 结果不强，有可能部分来自 pooling 选法

建议至少在 1000 张图上补以下组合：

### 方案 1：默认 first token

```bash
python code/pj1/task1/run_retrieval.py \
  --local-files-only \
  --bert-tokenizer-path outputs/task1_retrieval/hf_cache/manual/bert-base-uncased \
  --max-images 1000 \
  --model-spec lavis:blip2_feature_extractor:pretrain \
  --image-pooling first \
  --text-pooling first
```

### 方案 2：image mean pooling

```bash
python code/pj1/task1/run_retrieval.py \
  --local-files-only \
  --bert-tokenizer-path outputs/task1_retrieval/hf_cache/manual/bert-base-uncased \
  --max-images 1000 \
  --model-spec lavis:blip2_feature_extractor:pretrain \
  --image-pooling mean \
  --text-pooling first
```

### 方案 3：image/text mean pooling

```bash
python code/pj1/task1/run_retrieval.py \
  --local-files-only \
  --bert-tokenizer-path outputs/task1_retrieval/hf_cache/manual/bert-base-uncased \
  --max-images 1000 \
  --model-spec lavis:blip2_feature_extractor:pretrain \
  --image-pooling mean \
  --text-pooling mean
```

建议把这三组结果整理成一张小表：

- pooling 方式
- T2I R@1/R@5/R@10
- I2T R@1/R@5/R@10

这会让你后面解释 BLIP-2 的结果更稳。

## 5. 相似度矩阵与案例分析

为了更具体地解释模型行为，建议对 embedding cache 做两类分析：

1. 相似度矩阵热图
2. retrieval 成功/失败案例

本仓库已补充脚本：

- [analyze_task1_results.py](/Users/daishangzhe/研究生课程/多模态大模型/scripts/analyze_task1_results.py)

它依赖：

- `outputs/task1_retrieval/cache/*.npy`

也就是完整的 image embedding 和 text embedding cache。

如果你已经在服务器上跑完并缓存了 embeddings，请把服务器上的
`outputs/task1_retrieval/cache/` 同步回本地，然后执行：

```bash
python scripts/analyze_task1_results.py
```

输出包括：

- `outputs/task1_retrieval/analysis/*_sim_matrix.png`
- `outputs/task1_retrieval/analysis/task1_similarity_analysis.md`

建议至少整理三类样例：

1. `CLIP` 成功但 `BLIP-2` 失败
2. `BLIP retrieval` 成功但 `CLIP` 失败
3. 三者都容易混淆的 hard negatives

重点观察：

- 主物体依赖
- 颜色词
- 数量词
- 关系词

## 6. 为什么这个分析对 Task 2 也有用

这些补充实验不仅是在完善 Task 1，也会直接影响 Task 2 的模型选择：

- 如果某类模型在 retrieval 上已经明显更擅长细粒度图文对齐，那么它在 caption generation 的条件表示上通常更值得重点观察。
- 如果某类模型的相似度矩阵里出现明显的错配模式，例如总是只抓主物体、忽略关系词，那么在 Task 2 中也很可能生成过于模板化的 caption。

因此，建议在进入 Task 2 前至少补完：

1. 一组更公平的 Task 1 对比
2. 一组相似度矩阵与失败案例分析
