# Task 2 图像描述生成结果分析

## 实验设置

本实验在 COCO `val2017` 上比较 BLIP 与 BLIP-2 的 image captioning 能力。两组模型使用同一数据集、同一解码协议和同一评测脚本。

- 数据集：COCO `val2017`
- 图像数：5000
- 参考 caption 数：25014
- 每张图生成 caption 数：1
- 解码方式：beam search
- `num_beams`：5
- `max_length`：30
- `min_length`：1
- 指标：`Bleu_4`、`CIDEr`

对比模型：

- BLIP：`lavis:blip_caption:base_coco`
- BLIP-2：`lavis:blip2_opt:caption_coco_opt2.7b`

结果文件：

- BLIP prediction：`outputs/task2_captioning/predictions/lavis_blip_caption_base_coco_nall_beam5_max30_min1_sample0_1e5b638c.json`
- BLIP-2 prediction：`outputs/task2_captioning/predictions/lavis_blip2_opt_caption_coco_opt2.7b_nall_beam5_max30_min1_sample0_14149ed3.json`
- 汇总表：`outputs/task2_captioning/results/summary.csv`

## 定量结果

| Model | BLEU-4 | CIDEr | ROUGE-L* | Time | sec / image | images / sec |
|---|---:|---:|---:|---:|---:|---:|
| BLIP caption `base_coco` | 0.4167 | 1.4053 | 0.5753 | 1612.79s | 0.323 | 3.10 |
| BLIP-2 OPT `caption_coco_opt2.7b` | 0.4753 | 1.5732 | 0.6016 | 3098.89s | 0.620 | 1.61 |

`*` ROUGE-L 是基于已保存 prediction 重新评分得到的补充指标。本地复评时没有 Java，因此 PTBTokenizer 回退到简化 tokenization；正式报告中如果要求严格 COCO tokenization，建议在服务器安装 Java 后重新运行补充分数脚本。

相对 BLIP，BLIP-2 的提升为：

- BLEU-4：+0.0587，约 +14.08%
- CIDEr：+0.1679，约 +11.95%
- ROUGE-L：+0.0263，约 +4.57%
- 运行时间：约 1.92 倍

## 补充指标复评

为了避免重新生成 caption，补充指标通过读取 `outputs/task2_captioning/predictions/` 中的 prediction 文件重新评分，脚本为：

```bash
conda run -n multimodal python scripts/evaluate_task2_predictions.py
```

本地已产出的补充结果位于：

- `outputs/task2_captioning/results/extra_metrics_summary.csv`
- `outputs/task2_captioning/results/lavis_blip_caption_base_coco_nall_beam5_max30_min1_sample0_1e5b638c_extra_metrics.json`
- `outputs/task2_captioning/results/lavis_blip2_opt_caption_coco_opt2.7b_nall_beam5_max30_min1_sample0_14149ed3_extra_metrics.json`

当前本地环境缺少 Java，因此只得到可用的 ROUGE-L，METEOR 和 SPICE 被脚本安全跳过。若需要正式补齐 METEOR 和 SPICE，在服务器上执行：

```bash
conda activate multimodal
conda install -c conda-forge openjdk=11 -y
java -version
```

先补 METEOR 和 ROUGE-L：

```bash
PJ1_ENABLE_JAVA_METRICS=1 python scripts/evaluate_task2_predictions.py \
  --enable-java-metrics \
  --tokenizer-fallback error \
  --metric METEOR \
  --metric ROUGE_L
```

SPICE 额外需要 Stanford CoreNLP 依赖。若服务器能访问 Stanford 下载源，可以先运行：

```bash
python -c "from pycocoevalcap.spice.get_stanford_models import get_stanford_models; get_stanford_models()"
```

然后补 SPICE：

```bash
PJ1_ENABLE_JAVA_METRICS=1 python scripts/evaluate_task2_predictions.py \
  --enable-java-metrics \
  --tokenizer-fallback error \
  --metric SPICE
```

如果 SPICE 依赖下载失败，不影响 Task 2 的主结果；作业明确要求的最低指标是 BLEU-4 和 CIDEr，当前已经完成。

## 主要结论

BLIP-2 在 caption 质量上明显优于 BLIP。BLEU-4 和 CIDEr 都提升，说明它生成的句子不仅有更好的 n-gram 精确匹配，也更贴近 COCO 参考描述的内容表达。

代价也很明确：BLIP-2 的运行时间接近 BLIP 的两倍。对 5000 张 COCO val2017 图像，BLIP 约 26.9 分钟完成，BLIP-2 约 51.6 分钟完成。因此 BLIP-2 更适合作为质量优先的 caption 模型，BLIP 更适合作为轻量、速度优先的基线。

这组结果也和模型结构预期一致。BLIP caption 是较直接的视觉编码器加文本解码器方案；BLIP-2 通过 Q-Former 将视觉表征接入更强的 OPT 语言模型，语言生成能力和细粒度描述能力更强，但参数规模和推理成本也更高。

## 定性案例

下面的案例使用一个简单 token-overlap F1 启发式筛选，只作为定性分析辅助，不作为正式评测指标。正式结论仍以 BLEU-4 和 CIDEr 为准。

### BLIP-2 明显更好的案例

**Image ID 65455**

- BLIP：`a couple of giraffe standing next to each other`
- BLIP-2：`a group of children feeding giraffes at a zoo`
- 参考 caption：
  - `A group of giraffe Standing up against a dirt wall in front of a crowd of children.`
  - `A large group of people holding their arms out to feed giraffes.`
  - `A group of children feeding two giraffes at a zoo.`
- 分析：BLIP 只识别了主物体 giraffe，BLIP-2 进一步捕捉到 children feeding giraffes 这一交互关系。

**Image ID 91500**

- BLIP：`a couple of young men sitting next to each other`
- BLIP-2：`two young men sitting in chairs playing a video game`
- 参考 caption：
  - `The two teens are playing video games in the rec center.`
  - `two people sitting in chairs playing nintendo wii`
  - `Two men are sitting in chairs, playing a video game.`
- 分析：BLIP 描述停留在人和坐姿，BLIP-2 补充了 playing a video game，语义更完整。

**Image ID 323202**

- BLIP：`a white toilet sitting in a bathroom next to a magazine`
- BLIP-2：`a toilet with three seashells on top of it`
- 参考 caption：
  - `A toilet that has three sea shells on top.`
  - `A toilet with three seashells sitting on top of it.`
  - `Small white toilet with seashells sitting on top of it.`
- 分析：BLIP-2 抓住了图中异常且关键的 seashells，BLIP 则生成了更常见但不准确的 bathroom/magazine 描述。

**Image ID 570782**

- BLIP：`a laptop computer sitting on top of a white desk`
- BLIP-2：`a desk with two laptops and a monitor`
- 参考 caption：
  - `Three computers without screens sit on a desk.`
  - `A desktop containing various computers missing their screens.`
  - `a desk with two laptops and a monitor`
- 分析：BLIP-2 对数量和物体组合更准确，BLIP 的描述更泛化。

### BLIP 更好的案例

**Image ID 425906**

- BLIP：`a man flying through the air while riding a snowboard`
- BLIP-2：`a snowboarder doing a trick in front of a crowd`
- 参考 caption：
  - `A man flying through the air while riding a snowboard.`
  - `a person jumping a snow board in the air`
  - `People look on as an airborne snowboarder competes.`
- 分析：BLIP 在这个样例中几乎复现了参考表达；BLIP-2 的描述也合理，但少了 flying through the air 这一高匹配短语。

**Image ID 539883**

- BLIP：`a flat screen tv sitting on top of a wooden table`
- BLIP-2：`a man standing in front of a tv playing a video game`
- 参考 caption：
  - `A big screen TV and a Wii gaming console on a rooftop.`
  - `A wii game of baseball outside on bench`
  - `A flat screen TV sitting on top of a wooden bench.`
- 分析：BLIP 对主要物体和支撑物描述更稳；BLIP-2 可能受 video game 语境影响，加入了不存在或不突出的 man。

**Image ID 404479**

- BLIP：`a large jetliner sitting on top of an airport runway`
- BLIP-2：`a red and white airplane taking off from an airport runway`
- 参考 caption：
  - `A large jetliner sitting on top of an airport runway.`
  - `Japanese airliner just taking off with snow capped mountains in the background.`
  - `A passenger plane sits on a runway with mountains behind it.`
- 分析：该图参考描述本身存在 sitting 与 taking off 两类表述。BLIP 与其中一条参考高度一致，BLIP-2 的动作判断更激进。

### 两者都容易泛化或遗漏的案例

**Image ID 581100**

- BLIP：`a couple of giraffe standing next to each other on a lush green field`
- BLIP-2：`a couple of giraffes standing next to some palm trees`
- 参考 caption：
  - `Two giraffes and several other large animals roam a tropical zoo paddock.`
  - `Two giraffe's and a water buffalo in a zoo type environment`
  - `two giraffe are standing and the other animals are walking`
- 分析：两个模型都抓住 giraffes，但都漏掉了 other large animals / water buffalo 这一关键上下文。

**Image ID 286994**

- BLIP：`a herd of elephants standing on top of a dry grass field`
- BLIP-2：`a herd of elephants standing around a watering hole`
- 参考 caption：
  - `some elephants and one is by some water`
  - `A elephant drinks from a stream with several other elephants walking in the background.`
  - `One elephant standing away from the rest of the herd, drinking water.`
- 分析：BLIP-2 比 BLIP 更接近水域语义，但两个模型都没有明确描述 drinking 这一动作。

**Image ID 470773**

- BLIP：`a chef standing in front of a table filled with food`
- BLIP-2：`a chef standing in front of a counter full of food`
- 参考 caption：
  - `A pastry station, with an assortment of fillings and sauces`
  - `a chef's display of ingredients and pastry creations`
  - `Pastry items and toppings on display on a table.`
- 分析：两个模型都生成了合理但偏泛化的描述，没有捕捉 pastry / toppings / fillings 这些细粒度类别。

## 误差模式

从定性样例看，BLIP-2 的优势主要体现在三类场景：

- 关系和动作：例如 children feeding giraffes、playing a video game。
- 数量和组合：例如 two laptops and a monitor、three seashells。
- 非典型细节：例如 toilet 上的 seashells。

BLIP 的优势主要体现在稳定性和简洁性：

- 对常见场景、常见物体组合有较稳定的模板化描述。
- 有时能与 COCO 参考 caption 的常见句式高度一致。
- 生成速度更快，适合作为轻量 baseline。

两者共同的薄弱点包括：

- 容易忽略背景中的次要物体。
- 对细粒度活动状态仍可能描述不足，例如 drinking、taking off、feeding 等动作是否准确。
- 对 COCO 参考 caption 中存在歧义或多种表达的图像，指标可能受到参考文本差异影响。

## 报告可用结论

在 COCO val2017 caption generation 上，`blip2_opt:caption_coco_opt2.7b` 在 BLEU-4 和 CIDEr 上均优于 `blip_caption:base_coco`，说明 BLIP-2 借助 Q-Former 和更强语言模型后端，能生成更完整、更细粒度的图像描述。与此同时，BLIP-2 的推理成本显著更高，约为 BLIP 的 1.92 倍。因此，本实验支持以下判断：BLIP-2 更适合作为质量优先的 caption 模型，而 BLIP 是更轻量、更快的 caption baseline。
