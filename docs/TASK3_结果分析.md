# Task 3 表征分析实验报告

## 覆盖情况判断

当前 Task 3 结果已经覆盖 PDF 中的核心要求：

- **Part A: Embedding Visualization**
  - 已在同一批 `500` 个 image-text pair 上完成降维可视化
  - 已生成 `PCA` 与 `t-SNE`
  - 已对下面三点给出分析：
    1. 同一图文对是否更接近
    2. 图像点与文本点是否更容易混合
    3. 是否形成明显语义簇
- **Part B: Nearest-Neighbor Case Study**
  - 已分别给出：
    - 给定图像，找最近文本
    - 给定文本，找最近图像
  - 已展示 success case 与 hard negative case
  - 已能支持“基于文本检索图片”的额外能力

因此，**从作业达标角度看，当前结果已经可以写入正式报告**。

但如果你希望和 Task 1 的“统一 feature extraction 协议”更一致，我建议额外补两项增强内容：

1. 再跑一次 `lavis:blip_feature_extractor:base` 的 Task 3 分析  
   这样 Task 3 就能与 Task 1 的统一 feature-based comparison 完整对齐。

2. 再跑 3 到 5 条自由文本检索 demo  
   例如属性词、关系词、计数词查询，用来更直接展示“文本检索图片”的支持能力。

这两项是**增强项**，不是当前版本必须补的硬缺口。

## 实验设置

Task 3 默认复用 Task 1 已保存的 embedding cache，在 `outputs/task1_retrieval/cache/` 上进行分析，不重新提取图像 embedding。

当前已分析的三个模型：

- `lavis:clip_feature_extractor:base`
- `lavis:blip_retrieval:coco`
- `lavis:blip2_feature_extractor:pretrain`

统一设置：

- 数据集：COCO `val2017`
- 每张图选取 1 条 caption 参与可视化
- 可视化样本数：`500` image-text pairs
- 降维方法：`PCA` 与 `t-SNE`

输出文件：

- 主分析 markdown：`outputs/task3_representation/task3_representation_analysis.md`
- 图像文件：`outputs/task3_representation/figures/*.png`
- 统计摘要：`outputs/task3_representation/results/task3_summary.json`

## 定量摘要

| Model | Pair Sim Mean | Random Sim Mean | Margin | T2I Top1 | I2T Top1 | Cross-modal kNN Ratio | Semantic Silhouette |
|---|---:|---:|---:|---:|---:|---:|---:|
| CLIP feature extractor | 0.3057 | 0.1520 | 0.1538 | 0.584 | 0.622 | 0.0001 | -0.0466 |
| BLIP retrieval | 0.4197 | 0.1507 | 0.2690 | 0.846 | 0.888 | 0.0180 | -0.0333 |
| BLIP-2 feature extractor | 0.3045 | 0.0407 | 0.2639 | 0.562 | 0.636 | 0.0167 | -0.0499 |

指标解释：

- `Pair Sim Mean`：配对图文对的平均 cosine 相似度
- `Random Sim Mean`：随机错配图文对的平均 cosine 相似度
- `Margin`：配对相似度减去随机相似度的平均差值
- `T2I Top1`：在采样子集上的 text-to-image top-1 命中率
- `I2T Top1`：在采样子集上的 image-to-text top-1 命中率
- `Cross-modal kNN Ratio`：局部近邻中跨模态邻居占比
- `Semantic Silhouette`：基于粗粒度关键词标签估计的语义簇轮廓系数

## Part A: Embedding Visualization 分析

### 1. 同一图文对在 embedding space 中是否更接近

答案是 **是**。

三个模型的 `Pair Sim Mean` 都明显高于 `Random Sim Mean`：

- CLIP：`0.3057` vs `0.1520`
- BLIP retrieval：`0.4197` vs `0.1507`
- BLIP-2：`0.3045` vs `0.0407`

其中：

- **BLIP retrieval** 的配对相似度最高，`Margin=0.2690`
- **BLIP-2 feature extractor** 的 `Margin=0.2639`，说明它也能形成较强的配对对齐
- **CLIP** 的 `Margin=0.1538` 最小，说明它能对齐，但区分度相对更弱

从这个角度看，三者都满足“配对更近”的基本跨模态对齐特征，但 BLIP retrieval 最强。

### 2. 不同模型的图像点与文本点是否更容易混合在一起

从当前定义的 `Cross-modal kNN Ratio` 看，**三个模型都没有表现出很强的图像点-文本点混合**：

- CLIP：`0.0001`
- BLIP retrieval：`0.0180`
- BLIP-2：`0.0167`

这说明在局部近邻结构里，图像点和文本点大多仍然优先靠近同模态样本，而不是直接形成一个高度交织的 joint manifold。

这个现象有两个含义：

1. 模型确实学习到了对齐，但这种对齐更多体现在“配对关系”和“检索排序”上，而不是模态完全融合。
2. **BLIP retrieval** 和 **BLIP-2** 的跨模态混合程度略高于 CLIP，但提升并不大，说明它们并不是通过简单地把图像点和文本点完全揉在一起取得优势。

因此，对第二个问题的回答应当是：

- **没有哪个模型表现出非常强的图文点混合**
- **BLIP retrieval / BLIP-2 略好于 CLIP**
- **但三者整体仍保留明显模态边界**

### 3. 语义簇结构是否不同

从粗粒度关键词标签计算的 `Semantic Silhouette` 看，三者都没有形成非常清晰的全局语义簇：

- CLIP：`-0.0466`
- BLIP retrieval：`-0.0333`
- BLIP-2：`-0.0499`

这些值接近 0 且偏负，说明：

- “主物体簇”或“场景簇”并不是非常清晰的全局分离结构
- 语义结构更像是局部连续分布，而不是明显的大块簇

但从 nearest-neighbor 案例和局部图形上看，仍然可以观察到一些弱语义团块，例如：

- bedroom / living room / kitchen 这类室内场景
- bear / giraffe / teddy bear 这类主物体
- stop sign 这类高显著物体

因此，更准确的表述是：

- **不存在非常强的全局场景簇或主物体簇**
- **但局部仍可见主题性聚集**
- **BLIP retrieval 在局部结构上更稳定，和其检索性能优势一致**

## Part B: Nearest-Neighbor Case Study 分析

### 1. 模型是否过度依赖“主物体”

**是，三个模型都存在这个问题，但程度不同。**

典型现象：

- `bear` 查询会被大量其他 bear 图像吸引
- `stop sign` 查询会被各种 stop sign 图像吸引
- `bedroom / living room / kitchen` 查询会被同场景模板吸引

模型差异：

- **CLIP** 最明显，往往先抓主物体或主场景模板，再忽略细节
- **BLIP-2 feature extractor** 也有这个问题，尤其容易被“room/table/woman”这类大语义模板吸住
- **BLIP retrieval** 最稳，但仍会被“同主物体但不完全匹配”的图文对拉近

所以，对这个问题的回答是：

- 三者都依赖主物体
- BLIP retrieval 依赖程度相对最轻
- CLIP 和 BLIP-2 feature extractor 更容易出现“主物体对了，但具体匹配错了”

### 2. 是否容易忽略关系词、属性词、计数词

**是，这也是当前 nearest-neighbor 失败样例中的核心误差来源。**

从当前案例可以看到几类典型错误：

- **关系词**：`woman standing at the table` 容易被 `woman standing in a kitchen next to a table` 拉近
- **属性词**：`upside-down stop sign` 若模型不够强，就容易退化成一般性的 `stop sign`
- **计数词/组合结构**：`three teddy bears`、`two laptops and a monitor` 这类结构容易被数量或组合更模糊的近邻替代

模型差异：

- **CLIP** 对关系词、属性词和计数词最不稳定
- **BLIP retrieval** 对这些细节的保持最好，尤其在 `upside-down stop sign` 和 `three teddy bears` 这类例子里更强
- **BLIP-2 feature extractor** 虽然 pair margin 不低，但在 top-1 对齐上仍容易被更泛化的 room/object semantics 干扰

### 3. 是否容易把“语义接近但不匹配”的 hard negatives 拉得很近

**是，而且这正是 Task 3 最值得写进报告的现象。**

典型 hard negatives 包括：

- 同主物体、不同关系：
  - `woman at table` vs `woman in kitchen next to table`
- 同场景、不同精确描述：
  - `bedroom with bookshelf` vs `living room with bookshelf`
- 同类别、不同实例：
  - `three teddy bears` vs `three stuffed animals`
  - `grizzly bear on grass` vs `brown bear near rocks`

模型差异：

- **CLIP** 最容易把模板相近的 hard negatives 拉近
- **BLIP retrieval** 虽然仍会遇到 hard negatives，但其 top-1 准确率明显更高
- **BLIP-2 feature extractor** 在 broad semantic alignment 上表现不差，但在精确判别 hard negatives 上不如 BLIP retrieval

## 综合结论

Task 3 当前结果支持以下结论：

1. **三类模型都学到了跨模态对齐，但对齐强度不同。**
   - BLIP retrieval 最强
   - BLIP-2 feature extractor 次之
   - CLIP 最弱

2. **图像点与文本点并没有完全混合。**
   - 三个模型都仍保留明显模态边界
   - 说明好检索不一定等于完全模态融合

3. **hard negative 是区分模型能力的关键。**
   - BLIP retrieval 对细粒度匹配最稳
   - CLIP 更依赖主物体和场景模板
   - BLIP-2 feature extractor 的 broad semantics 较强，但在精确 top-1 匹配上不如 BLIP retrieval

4. **Task 3 的观察和 Task 1 的检索结果是一致的。**
   - Task 1 中 BLIP retrieval 的 Recall 最强
   - Task 3 中它在配对相似度、top-1 命中率和 hard negative 处理上也最稳

## 建议补充但非必须

当前结果已经满足 PDF 核心要求。若要进一步增强报告，可以再补两项：

### 1. 加入 `blip_feature_extractor:base`

这样 Task 3 会和 Task 1 的 unified feature-based comparison 更对齐：

```bash
python code/pj1/task3/run_representation_analysis.py \
  --run-name lavis_blip_feature_extractor_base_nall_first_first_69e93c04 \
  --sample-size 500 \
  --method pca \
  --method tsne
```

### 2. 增加自由文本检索 demo

例如：

```bash
python code/pj1/task3/text_to_image_search.py \
  --local-files-only \
  --bert-tokenizer-path outputs/task1_retrieval/hf_cache/manual/bert-base-uncased \
  --run-name lavis_blip_retrieval_coco_nall_first_first_2abfcb45 \
  --query "two dogs playing with each other in the grass" \
  --query "a stop sign mounted upside down" \
  --query "three teddy bears hugging on a pillow" \
  --top-k 10
```

这能更直接展示：

- 文本检索图片能力已经实现
- 模型对属性词、关系词、计数词是否敏感
