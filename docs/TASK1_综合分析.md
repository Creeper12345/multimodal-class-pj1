# Task 1 综合分析

本文件在前两份分析基础上，结合你后来补跑的更公平实验、BLIP-2 pooling 消融，以及相似度矩阵与案例分析，给出一版更适合正式报告使用的综合结论。

相关结果文件：

- [summary.csv](/Users/daishangzhe/研究生课程/多模态大模型/outputs/task1_retrieval/results/summary.csv)
- [task1_similarity_analysis.md](/Users/daishangzhe/研究生课程/多模态大模型/outputs/task1_retrieval/analysis/task1_similarity_analysis.md)
- [TASK1_结果分析.md](/Users/daishangzhe/研究生课程/多模态大模型/docs/TASK1_结果分析.md)
- [TASK1_补充实验设计.md](/Users/daishangzhe/研究生课程/多模态大模型/docs/TASK1_补充实验设计.md)

## 1. 结论先行

Task 1 现在已经可以拆成两层结果来讲：

1. **Off-the-shelf retrieval comparison**
   - 直接比较 LAVIS 中现成可调用的模型入口
   - 结论：`blip_retrieval:coco` 明显最好
2. **Unified feature-based retrieval comparison**
   - 统一采用 feature extractor + embedding similarity + Recall evaluator
   - 结论：`blip_feature_extractor:base` 明显优于 `CLIP` 与当前的 `blip2_feature_extractor:pretrain`

同时，BLIP-2 的补充消融表明：

- 当前 `BLIP-2 pretrain` 在检索上的弱势，不只是模型本身的问题
- **pooling 方式会极大影响结果**
- 仅把它的 query-based image feature 取第一个 token，会显著低估它的检索能力

## 2. 需要先说明的实验边界

### 2.1 当前已有结果不是同一种公平性层级

这是你现在写报告时必须主动说明的地方。

当前已完成结果包含两类：

#### 第一类：off-the-shelf task performance

- `lavis:clip_feature_extractor:ViT-B-32`
- `lavis:blip_retrieval:coco`
- `lavis:blip2_feature_extractor:pretrain`

这组更适合回答：

- 在 LAVIS 现成模型入口里，哪一个最适合直接做 COCO retrieval？

#### 第二类：unified feature extraction protocol

- `lavis:clip_feature_extractor:ViT-B-32`
- `lavis:blip_feature_extractor:base`
- `lavis:blip2_feature_extractor:pretrain`

这组更适合回答：

- 在统一抽特征、统一相似度、统一 Recall evaluator 的协议下，不同表示学习方式的检索能力如何？

这两层结果不能混写成同一类实验，否则老师很容易追问“你是不是拿 task-specific retrieval model 去和 generic feature extractor 比”。

### 2.2 当前运行时间不能直接横向比较

这一点也需要说明。

因为你同步回来的部分结果是：

- 首次完整推理得到的结果
- 后续命中 embedding cache 后再次评测得到的结果

所以当前 `elapsed_seconds` 不一定处在同一比较条件下。  
例如某些结果明显已经受到了 cache 命中影响。

因此，**本轮分析以 Recall 指标为主，不把当前同步回来的运行时间作为严格的速度结论。**

## 3. 第一层结果：Off-the-shelf retrieval comparison

### 3.1 核心结果

| 模型 | T2I R@1 | T2I R@5 | T2I R@10 | I2T R@1 | I2T R@5 | I2T R@10 |
|---|---:|---:|---:|---:|---:|---:|
| CLIP feature extractor (ViT-B-32) | 30.36 | 54.78 | 66.09 | 50.02 | 75.00 | 83.26 |
| BLIP retrieval:coco | 64.96 | 86.74 | 92.39 | 83.20 | 96.36 | 98.20 |
| BLIP-2 feature extractor:pretrain | 30.03 | 52.09 | 62.17 | 40.38 | 68.06 | 78.60 |

### 3.2 这组结果说明什么

结论非常清楚：

1. `BLIP retrieval:coco` 是当前实验设置下最强的模型
2. `CLIP` 是稳定但明显更弱的 baseline
3. `BLIP-2 pretrain feature extractor` 在该协议下并不占优

但这里必须加一句限制：

- 这组结果不能直接被解读为“BLIP retrieval 这种对齐范式本质上最强”

更准确的说法是：

- **在当前 LAVIS 的 off-the-shelf 入口下，`blip_retrieval:coco` 最适合直接做 COCO retrieval**

### 3.3 为什么 BLIP retrieval 会领先这么多

因为它不是一个纯通用 feature extractor，而是：

- 面向检索任务训练
- 使用了更贴近 retrieval 的训练目标
- checkpoint 本身也对 COCO retrieval 更适配

所以它的显著领先有两层原因：

1. 模型结构和训练目标对检索更友好
2. checkpoint 与当前任务高度贴合

这不是坏事，但要在报告里说清楚。

## 4. 第二层结果：Unified feature-based retrieval comparison

### 4.1 核心结果

| 模型 | T2I R@1 | T2I R@5 | T2I R@10 | I2T R@1 | I2T R@5 | I2T R@10 |
|---|---:|---:|---:|---:|---:|---:|
| CLIP feature extractor (ViT-B-32) | 30.36 | 54.78 | 66.09 | 50.02 | 75.00 | 83.26 |
| BLIP feature extractor:base | 49.90 | 76.11 | 84.88 | 63.76 | 86.80 | 93.12 |
| BLIP-2 feature extractor:pretrain | 30.03 | 52.09 | 62.17 | 40.38 | 68.06 | 78.60 |

### 4.2 这一组才更适合回答作业核心问题

因为这组更接近统一协议：

- 都是 feature extractor
- 都是抽 image/text embedding
- 都是统一相似度计算
- 都是统一 Recall evaluator

在这个协议下，结果变得很有信息量：

1. `BLIP feature extractor:base` 显著优于 `CLIP`
2. 当前 `BLIP-2 pretrain feature extractor` 仍然不占优
3. 说明“更丰富的图文表征机制”在统一协议下确实可以优于纯 contrastive baseline，但不同实现之间差异很大

### 4.3 最重要的新发现

在这组更公平的对照里：

- `BLIP feature extractor:base` 成了最值得重视的模型

这点很关键，因为它说明：

- 当你去掉 task-specific retrieval checkpoint 的直接优势之后
- **BLIP 作为统一表征模型，本身仍然显著强于 CLIP**

这比只看 `blip_retrieval:coco` 更能支持作业中关于“对齐机制差异”的讨论。

## 5. 第三层结果：BLIP-2 pooling 消融

你补的这组实验非常重要，直接改变了对 BLIP-2 的解释。

实验设置：

- 数据集规模：1000 images / 5003 captions
- 模型：`lavis:blip2_feature_extractor:pretrain`
- 对比不同 pooling 方式

### 5.1 消融结果

| pooling | T2I R@1 | T2I R@5 | T2I R@10 | I2T R@1 | I2T R@5 | I2T R@10 |
|---|---:|---:|---:|---:|---:|---:|
| image=`first`, text=`first` | 48.81 | 75.13 | 84.25 | 63.70 | 88.80 | 94.80 |
| image=`mean`, text=`first` | 71.86 | 92.90 | 97.10 | 76.10 | 95.30 | 98.30 |
| image=`mean`, text=`mean` | 69.62 | 92.06 | 96.84 | 79.80 | 95.70 | 98.60 |

### 5.2 解释

结论很明确：

1. **image pooling 从 `first` 改成 `mean` 后，BLIP-2 检索性能大幅提升**
2. text pooling 从 `first` 改成 `mean` 后：
   - T2I 略降
   - I2T 略升

这说明：

- `BLIP-2` 的检索结果对 pooling 非常敏感
- 你之前看到的“BLIP-2 不如 CLIP”，并不能简单归因于模型本身
- 其中相当一部分可能来自：
  - 当前使用的是 `pretrain` 版
  - 当前默认 pooling 选法不适合 retrieval

### 5.3 对报告的意义

这一组消融能帮助你把结论写得更稳：

- 不要说 “BLIP-2 不适合 retrieval”
- 更准确的说法是：
  - **在当前使用的 `blip2_feature_extractor:pretrain` 和默认 pooling 下，检索结果不如 BLIP feature extractor 与 BLIP retrieval；但补充消融表明，BLIP-2 的结果对 pooling 选择高度敏感。**

## 6. 相似度矩阵与案例分析

详细案例文件已经生成：

- [task1_similarity_analysis.md](/Users/daishangzhe/研究生课程/多模态大模型/outputs/task1_retrieval/analysis/task1_similarity_analysis.md)

代表性热图：

- [CLIP heatmap](/Users/daishangzhe/研究生课程/多模态大模型/outputs/task1_retrieval/analysis/lavis_clip_feature_extractor_ViT-B-32_nall_first_first_c1e57b48_sim_matrix.png)
- [BLIP feature heatmap](/Users/daishangzhe/研究生课程/多模态大模型/outputs/task1_retrieval/analysis/lavis_blip_feature_extractor_base_nall_first_first_69e93c04_sim_matrix.png)
- [BLIP retrieval heatmap](/Users/daishangzhe/研究生课程/多模态大模型/outputs/task1_retrieval/analysis/lavis_blip_retrieval_coco_nall_first_first_2abfcb45_sim_matrix.png)
- [BLIP-2 heatmap](/Users/daishangzhe/研究生课程/多模态大模型/outputs/task1_retrieval/analysis/lavis_blip2_feature_extractor_pretrain_nall_first_first_91ffb98f_sim_matrix.png)

### 6.1 从热图看，BLIP retrieval 的对角结构最清晰

这和它在 Recall 指标上的领先是一致的。

可解释为：

- 正样本对的匹配程度更集中
- 错配相似度被压得更低
- 检索排序边界更清楚

### 6.2 BLIP-2 的失败案例很能说明问题

从生成的案例里，可以看到几种典型错误模式：

1. **室内场景混淆**
   - 例如 `000000000139.jpg`
   - 多个 caption 都在描述“室内 + 桌子 + 人”
   - 模型容易把 dining / living / kitchen 一类图像混在一起

2. **过于依赖主物体或粗语义**
   - 能识别“熊”“卧室”“网球”“滑雪”
   - 但容易丢失更细的关系、动作或上下文限定

3. **对细节词不稳定**
   - 数量、动作、位置关系、场景边界并不总能稳定保留

### 6.3 pooling 改进后，BLIP-2 的错误模式有所缓解

特别是在 `image=mean` 的设置下：

- 熊、卧室、厨房、玩偶等主视觉模式匹配更稳定
- 多 query token 聚合后，图像向量显然比只取第一个 token 更有信息量

这正好支持你在报告里的一个重要论点：

- **BLIP-2 的 retrieval 表现不能脱离 pooling 设计来讨论**

## 7. 现在 Task 1 可以怎么组织进正式报告

我建议直接拆成两个小节：

### 3.1 Off-the-shelf retrieval comparison

放：

- `clip_feature_extractor:ViT-B-32`
- `blip_retrieval:coco`
- `blip2_feature_extractor:pretrain`

这部分回答：

- 在 LAVIS 现成模型入口中，谁直接做 COCO retrieval 最强？

### 3.2 Unified feature-based retrieval comparison

放：

- `clip_feature_extractor:ViT-B-32`
- `blip_feature_extractor:base`
- `blip2_feature_extractor:pretrain`

再补一个：

- `BLIP-2 pooling ablation`

这部分回答：

- 在统一 embedding 检索协议下，不同表征方式谁更强？
- BLIP-2 当前结果到底是模型问题，还是 pooling / checkpoint 问题？

## 8. 进入 Task 2 前，你已经得到的启发

这些 Task 1 结果对 Task 2 很有用：

1. `BLIP retrieval` 在检索上最强，说明它的图文对齐很细，但它不是 caption generation 模型。
2. `BLIP feature extractor` 在统一表征对比里很强，说明 BLIP 一代的视觉-文本表示很扎实。
3. `BLIP-2` 的结果对 pooling 和表示方式敏感，说明它在 Task 2 中也值得重点观察：
   - 它未必在简单检索指标上最强
   - 但它的 Q-Former 架构可能在生成任务上更有优势

因此，进入 Task 2 时，一个合理策略是：

- 把 `BLIP` 和 `BLIP-2` 作为 caption 生成主比较对象
- 同时带着 Task 1 的发现去看：
  - 哪类模型更容易生成泛化但模糊的 caption
  - 哪类模型更能保留细粒度属性和关系

## 9. 目前最稳的总括

如果现在要写一个简洁但严谨的总括，可以这样写：

> 在 Task 1 中，我们首先比较了 LAVIS 中现成可用的检索相关模型入口。结果显示，`blip_retrieval:coco` 在 COCO `val2017` 双向检索上显著优于 `CLIP` 与 `BLIP-2 pretrain feature extractor`，说明面向检索任务微调的模型在 off-the-shelf 设置下具备明显优势。进一步地，在统一 feature extraction 协议下，`blip_feature_extractor:base` 依然显著优于 `CLIP`，说明 BLIP 的表示学习机制本身具有更强的图文对齐能力。与此同时，BLIP-2 的补充消融表明，其检索结果对 pooling 方式高度敏感，因此不能仅凭默认 `pretrain` 配置下的一组结果对其检索能力作出简单结论。
