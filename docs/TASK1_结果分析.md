# Task 1 结果分析

本文件基于服务器回传并同步到本地的 Task 1 结果文件撰写，数据来源为：

- [summary.csv](/Users/daishangzhe/研究生课程/多模态大模型/outputs/task1_retrieval/results/summary.csv)
- [lavis_clip_feature_extractor_base_nall_first_first_98f498ec.json](/Users/daishangzhe/研究生课程/多模态大模型/outputs/task1_retrieval/results/lavis_clip_feature_extractor_base_nall_first_first_98f498ec.json)
- [lavis_blip_retrieval_coco_nall_first_first_2abfcb45.json](/Users/daishangzhe/研究生课程/多模态大模型/outputs/task1_retrieval/results/lavis_blip_retrieval_coco_nall_first_first_2abfcb45.json)
- [lavis_blip2_feature_extractor_pretrain_nall_first_first_91ffb98f.json](/Users/daishangzhe/研究生课程/多模态大模型/outputs/task1_retrieval/results/lavis_blip2_feature_extractor_pretrain_nall_first_first_91ffb98f.json)

## 1. 实验设置

- 数据集：COCO `val2017`
- 图像数量：5000
- caption 数量：25014
- 评测任务：
  - Text-to-Image Retrieval
  - Image-to-Text Retrieval
- 评测指标：
  - Recall@1
  - Recall@5
  - Recall@10

当前这组结果对应的是一组 **off-the-shelf retrieval comparison**，参与比较的模型为：

- `lavis:clip_feature_extractor:base`
- `lavis:blip_retrieval:coco`
- `lavis:blip2_feature_extractor:pretrain`

## 2. Off-the-shelf 核心结果

| 模型 | T2I R@1 | T2I R@5 | T2I R@10 | I2T R@1 | I2T R@5 | I2T R@10 | 用时(s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| CLIP feature extractor | 30.36 | 54.78 | 66.09 | 50.02 | 75.00 | 83.24 | 270.59 |
| BLIP retrieval | 64.96 | 86.74 | 92.39 | 83.20 | 96.36 | 98.20 | 522.31 |
| BLIP-2 feature extractor | 30.03 | 52.09 | 62.17 | 40.38 | 68.06 | 78.60 | 581.13 |

## 3. 排名与对比

### Text-to-Image Retrieval

排名：

1. `BLIP retrieval`
2. `CLIP feature extractor`
3. `BLIP-2 feature extractor`

关键差距：

- `BLIP retrieval` 相比 `CLIP`：
  - R@1 提高 **34.61** 个点
  - R@5 提高 **31.96** 个点
  - R@10 提高 **26.29** 个点
- `BLIP-2 feature extractor` 相比 `CLIP`：
  - R@1 低 **0.32** 个点
  - R@5 低 **2.69** 个点
  - R@10 低 **3.92** 个点

### Image-to-Text Retrieval

排名：

1. `BLIP retrieval`
2. `CLIP feature extractor`
3. `BLIP-2 feature extractor`

关键差距：

- `BLIP retrieval` 相比 `CLIP`：
  - R@1 提高 **33.18** 个点
  - R@5 提高 **21.36** 个点
  - R@10 提高 **14.96** 个点
- `BLIP-2 feature extractor` 相比 `CLIP`：
  - R@1 低 **9.64** 个点
  - R@5 低 **6.94** 个点
  - R@10 低 **4.64** 个点

## 4. 这组结果能说明什么

### 4.1 BLIP retrieval 明显是当前实验设置下的最佳模型

这是最清楚的结论。

`BLIP retrieval:coco` 在两个方向、全部 Recall 指标上都显著领先另外两类模型，说明它在当前任务上最贴近评测目标。这个结果本身也符合模型训练方式：

- 它本来就是为检索任务设计和微调的
- 使用了专门的图文对齐结构和检索相关目标
- 在 COCO 检索场景上具备明显优势

因此，更准确的说法应当是：

- `BLIP retrieval:coco` 应被视为**当前实验设置下的任务最优对照模型**

而不是直接把它等同于“统一协议下最公平的主基线”。

### 4.2 CLIP 是一个合理但明显更弱的通用表示学习对照组

`CLIP feature extractor:base` 的表现中规中矩：

- T2I R@1 为 **30.36**
- I2T R@1 为 **50.02**

这说明它具备可用的跨模态匹配能力，但在 COCO 检索这种对细粒度图文对齐要求较高的任务上，仍然明显落后于检索专门模型。

如果报告里需要回答“纯 contrastive learning 的上限或局限”，这一组结果可以作为证据之一，但需要带上限定：

- 它能学到全局语义对齐
- 但对更细粒度的 caption 区分、局部属性、关系词、场景细节，优势不如检索专门模型

更严谨的表述应该是：

- 这一证据来自“检索任务模型”与“通用特征提取器”的对比，因此它更多说明 contrastive baseline 在**当前检索任务设置**下的不足；若要更严格比较不同对齐机制本身，还需补充统一 feature extraction 协议下的实验。

### 4.3 当前使用的 BLIP-2 feature extractor 在检索上并不占优

这是一个比较重要、也比较容易误判的点。

虽然 `BLIP-2` 名字更“新”，但你当前比较的是：

- `blip2_feature_extractor:pretrain`

也就是一个 **预训练特征提取器**，而不是专门为 COCO retrieval 微调的检索模型。

所以它出现下面这种现象是合理的：

- 比 `BLIP retrieval` 差很多
- 甚至整体不如 `CLIP`

这不代表 `BLIP-2` 框架本身弱，而是说明：

- 当前选用的这个 BLIP-2 变体更偏通用预训练特征
- 它并没有在这个检索任务上体现出专门优化后的优势

如果在报告中不解释这一点，读者可能会误以为“BLIP-2 比 CLIP 差”。更准确的说法应该是：

- **当前使用的 `BLIP-2 feature extractor:pretrain` 在此检索协议下不如 CLIP，也明显不如 BLIP retrieval**

### 4.4 这组结果还不能单独回答“哪种跨模态对齐范式更强”

这是当前实验最需要补的一点。

原因在于当前对比对象并没有完全对齐：

1. `BLIP retrieval:coco` 是 task-specific retrieval model
2. `CLIP feature extractor` 是 generic feature extractor
3. `BLIP-2 feature extractor:pretrain` 是 generic pretrain feature extractor

所以这组结果更准确地回答的是：

- 在当前调用的这三个 LAVIS 入口里，谁最适合直接做 COCO retrieval？

而不是：

- 三种跨模态对齐范式谁本质上更强？

前一个问题是成立的，后一个问题还需要补实验。

### 4.5 `clip_feature_extractor:base` 的真实 backbone 需要在正式报告里写清楚

这一点在复现实验时尤其重要。

LAVIS 中 `clip_feature_extractor` 常见 model type 更接近：

- `ViT-B-32`
- `ViT-B-16`
- `ViT-L-14`
- `ViT-L-14-336`
- `RN50`

而当前实验记录里使用的是：

- `lavis:clip_feature_extractor:base`

因此在正式报告中，建议你补充说明：

1. 当前 `base` 是否是本地实验代码或当前安装版本中的一个别名
2. 它实际对应的 backbone 是哪一个 CLIP 变体

否则读者很难严格复现实验。

## 5. 速度与效果权衡

从总用时看：

- `CLIP`：**270.59 s**
- `BLIP retrieval`：**522.31 s**
- `BLIP-2 feature extractor`：**581.13 s**

可以得到两个结论：

1. `BLIP retrieval` 虽然比 `CLIP` 慢，但效果提升非常大，这个代价是划算的。
2. `BLIP-2 feature extractor` 是三者里最慢的，但检索结果并没有相应收益，当前配置下性价比最低。

因此，如果以 Task 1 为目标：

- **最佳效果**：`BLIP retrieval`
- **最快且能接受的对照基线**：`CLIP`
- **当前配置下不推荐作为主检索模型**：`BLIP-2 feature extractor`

## 6. 这组结果在正式报告里应当怎么写

建议把当前结果归入一个单独小节，标题改成更准确的形式，例如：

- `Off-the-shelf retrieval performance under LAVIS model variants`

在这个小节里，可以直接写出以下结论：

1. 在统一 COCO `val2017` 检索协议下，`BLIP retrieval:coco` 在双向检索上全面优于 `CLIP` 和 `BLIP-2 feature extractor`。
2. `CLIP` 提供了一个稳定的 contrastive baseline，但在细粒度检索上存在明显性能差距。
3. 当前使用的 `BLIP-2 feature extractor:pretrain` 并未针对该检索任务做专门优化，因此在 Recall 指标上未体现优势。
4. 从效果与时间综合考虑，Task 1 当前设置下的最优模型应选 `BLIP retrieval:coco`。

## 7. 还需要补什么，才能让 Task 1 更完整

当前结果已经足够作为第一版核心结果，但还不能算“完全完善”。后续至少建议补四项：

1. 加上 `blip_feature_extractor:base`
2. 补一组统一 feature extraction 协议的对照
3. 明确 BLIP-2 的 pooling 方式
4. 做 retrieval case study 和相似度矩阵分析

## 8. 建议的图表与呈现方式

报告中建议至少做两张表：

### 表 1：主结果表

- 行：三个模型
- 列：
  - T2I R@1 / R@5 / R@10
  - I2T R@1 / R@5 / R@10
  - elapsed_seconds

### 表 2：相对 CLIP 的增减

- 用 `BLIP retrieval - CLIP`
- 用 `BLIP-2 - CLIP`

这张表有助于把“检索专门模型”和“通用对照模型”的差距讲清楚。

## 9. 下一步建议

基于当前 Task 1 结果，下一步建议是：

1. 保留当前这组结果，作为第一层 `off-the-shelf retrieval comparison`。
2. 再补一组 `unified feature-based retrieval comparison`：
   - `clip_feature_extractor`
   - `blip_feature_extractor`
   - `blip2_feature_extractor`
3. 对 BLIP-2 做一个小规模 pooling 消融。
4. 补相似度矩阵和 retrieval case study。
5. 再进入 Task 2。
