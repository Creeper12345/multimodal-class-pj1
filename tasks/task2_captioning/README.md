# Task 2 - 图像描述生成

这是 Task 2 的工作目录。

## 本任务目标

- 比较 BLIP 与 BLIP-2 的 caption 生成能力
- 评测 BLEU-4 和 CIDEr
- 视环境情况补充 METEOR、ROUGE-L、SPICE
- 输出定量表格与定性案例

## 代码入口

- `code/pj1/task2/coco.py`
  - 读取 COCO `val2017` 图像和参考 caption
- `code/pj1/task2/models.py`
  - 统一封装 LAVIS caption 模型生成接口
- `code/pj1/task2/metrics.py`
  - 统一封装 caption 指标评测
- `code/pj1/task2/run_captioning.py`
  - Task 2 主入口
- `scripts/prefetch_task2_models.py`
  - 提前下载并验证 caption 模型权重

## 推荐模型

- `lavis:blip_caption:base_coco`
- `lavis:blip2_opt:caption_coco_opt2.7b`

说明：

- `blip_caption:base_coco` 是较轻的 COCO caption 基线。
- `blip2_opt:caption_coco_opt2.7b` 是更强但更重的 BLIP-2 caption 模型。
- 如果服务器显存不足，先用 `--max-images 20` 做 smoke test。

## 标准运行顺序

### 1. 预下载模型

```bash
conda run -n multimodal python scripts/prefetch_task2_models.py \
  --hf-endpoint https://hf-mirror.com
```

默认只做加载检查，不做生成验证，并且默认使用 CPU，避免预下载阶段占用 GPU。
如果要验证生成链路：

```bash
conda run -n multimodal python scripts/prefetch_task2_models.py \
  --hf-endpoint https://hf-mirror.com \
  --device cuda \
  --smoke-generate
```

### 2. Dry-run

```bash
conda run -n multimodal python code/pj1/task2/run_captioning.py --dry-run --max-images 10
```

### 3. 小规模 smoke test

```bash
conda run -n multimodal python code/pj1/task2/run_captioning.py \
  --local-files-only \
  --bert-tokenizer-path outputs/task2_captioning/hf_cache/manual/bert-base-uncased \
  --max-images 20 \
  --batch-size 4 \
  --model-spec lavis:blip_caption:base_coco
```

```bash
conda run -n multimodal python code/pj1/task2/run_captioning.py \
  --local-files-only \
  --bert-tokenizer-path outputs/task2_captioning/hf_cache/manual/bert-base-uncased \
  --max-images 20 \
  --batch-size 1 \
  --model-spec lavis:blip2_opt:caption_coco_opt2.7b
```

### 4. 正式运行

```bash
conda run -n multimodal python code/pj1/task2/run_captioning.py \
  --local-files-only \
  --bert-tokenizer-path outputs/task2_captioning/hf_cache/manual/bert-base-uncased \
  --batch-size 4 \
  --model-spec lavis:blip_caption:base_coco \
  --model-spec lavis:blip2_opt:caption_coco_opt2.7b
```

## 结果位置

- `outputs/task2_captioning/predictions/<run_name>.json`
- `outputs/task2_captioning/results/<run_name>.json`
- `outputs/task2_captioning/results/summary.csv`

运行时会显示 caption 生成进度条。如果 prediction 文件已经存在，脚本会打印
`Loading cached predictions` 并直接复用缓存，只重新执行指标评测。

## 补充指标复评

如果已经生成 prediction，不需要重新跑模型，可以直接补 METEOR、ROUGE-L、SPICE：

```bash
conda run -n multimodal python scripts/evaluate_task2_predictions.py
```

本地无 Java 时，脚本会跳过 `METEOR` 和 `SPICE`，只计算可用指标。服务器上补齐 Java 后：

```bash
conda install -c conda-forge openjdk=11 -y
PJ1_ENABLE_JAVA_METRICS=1 python scripts/evaluate_task2_predictions.py \
  --enable-java-metrics \
  --tokenizer-fallback error \
  --metric METEOR \
  --metric ROUGE_L
```

`SPICE` 还需要 Stanford CoreNLP 依赖。若下载源可访问：

```bash
python -c "from pycocoevalcap.spice.get_stanford_models import get_stanford_models; get_stanford_models()"
PJ1_ENABLE_JAVA_METRICS=1 python scripts/evaluate_task2_predictions.py \
  --enable-java-metrics \
  --tokenizer-fallback error \
  --metric SPICE
```

## 评测注意事项

- 标准 COCO tokenization 依赖 Java。
- 如果机器没有 Java，代码会默认回退到 `identity` tokenization，并在结果 JSON 里写入 warning。
- 默认指标只计算 `Bleu_4` 和 `CIDEr`。
- `METEOR` 和 `SPICE` 依赖 Java；默认会被跳过，避免 smoke test 卡死。
- 若确认 Java 和 SPICE 依赖都已准备好，再设置 `PJ1_ENABLE_JAVA_METRICS=1` 并显式传入 `--metric METEOR` 或 `--metric SPICE`。
- 如果要强制使用标准 tokenization，添加：

```bash
--tokenizer-fallback error --strict-metrics
```
