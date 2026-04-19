"""Run Task 3 embedding visualization and nearest-neighbor analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from code.pj1.progress import progress_iter
from code.pj1.task3.analysis import (
    discover_cache_runs,
    embedding_statistics,
    load_run_dataset,
    load_run_embeddings,
    paired_sample,
    reduce_embeddings,
    save_embedding_plot,
    select_case_indices,
    similarity_text_by_image,
    top_images_for_text,
    top_texts_for_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", default="datasets/annotations/captions_val2017.json")
    parser.add_argument("--image-dir", default="datasets/val2017")
    parser.add_argument("--task1-output-dir", default="outputs/task1_retrieval")
    parser.add_argument("--output-dir", default="outputs/task3_representation")
    parser.add_argument("--run-name", action="append", default=None)
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--method", choices=("pca", "tsne", "umap"), action="append", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--num-cases", type=int, default=5)
    return parser.parse_args()


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def format_neighbor_items(items) -> list[str]:
    lines: list[str] = []
    for item in items:
        flag = "match" if item.is_match else "hard negative"
        lines.append(
            f"  - #{item.rank} `{flag}` score={item.score:.4f} image={item.image_file} caption=`{item.caption}`"
        )
    return lines


def qualitative_notes(stats: dict[str, Any]) -> list[str]:
    notes: list[str] = []
    margin = float(stats["pair_random_margin_mean"])
    cross = float(stats["cross_modal_knn_ratio"])
    silhouette = stats.get("semantic_silhouette_cosine")
    if margin > 0.15:
        notes.append("同一图文对的平均相似度明显高于随机配对，说明该模型形成了有效的跨模态对齐。")
    else:
        notes.append("同一图文对相对随机配对的优势较弱，说明该模型在当前 embedding 协议下对齐不够强。")
    if cross > 0.45:
        notes.append("图像点与文本点在近邻中高度混合，跨模态空间融合较好。")
    elif cross > 0.30:
        notes.append("图像点与文本点有一定混合，但仍保留较明显的模态分离。")
    else:
        notes.append("近邻主要来自同一模态，图像和文本点分离较明显。")
    if silhouette is None:
        notes.append("语义簇轮廓系数不可稳定估计，可能是有效类别过少或分布过散。")
    elif silhouette > 0.08:
        notes.append("主物体/场景标签出现一定聚类，说明 embedding 中存在可见的语义簇结构。")
    else:
        notes.append("主物体/场景标签的聚类较弱，说明语义结构更多是连续混合而非清晰分簇。")
    return notes


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    figure_dir = output_dir / "figures"
    methods = args.method or ["pca"]
    runs = discover_cache_runs(Path(args.task1_output_dir), args.run_name)
    if not runs:
        raise FileNotFoundError("No Task 1 embedding cache runs were found.")

    md_lines = ["# Task 3 表征可视化与 Nearest-Neighbor 分析", ""]
    summary: list[dict[str, Any]] = []

    for run in progress_iter(runs, desc="Task 3 analysis", total=len(runs), unit="run"):
        image_embeddings, text_embeddings = load_run_embeddings(run)
        images, records = load_run_dataset(args.annotation, args.image_dir, run)
        sample = paired_sample(image_embeddings, text_embeddings, images, records, args.sample_size)
        stats = embedding_statistics(
            sample["image_embeddings"],
            sample["text_embeddings"],
            sample["semantic_labels"],
        )
        stats_payload = {
            "run_name": run.run_name,
            "spec": run.spec,
            "image_pooling": run.image_pooling,
            "text_pooling": run.text_pooling,
            **stats,
        }
        summary.append(stats_payload)
        save_json(output_dir / "results" / f"{run.run_name}_stats.json", stats_payload)

        plot_paths: list[Path] = []
        combined = np.vstack([sample["image_embeddings"], sample["text_embeddings"]])
        for method in methods:
            coords = reduce_embeddings(combined, method=method)
            plot_path = figure_dir / f"{run.run_name}_{method}.png"
            save_embedding_plot(
                coords,
                labels=sample["semantic_labels"],
                output_path=plot_path,
                title=f"{run.run_name} ({method.upper()})",
            )
            plot_paths.append(plot_path)

        matrix = similarity_text_by_image(image_embeddings, text_embeddings)
        success_cases, failure_cases = select_case_indices(matrix, records, args.num_cases)

        md_lines.append(f"## {run.run_name}")
        md_lines.append("")
        md_lines.append(f"- 模型：`{run.spec}`")
        md_lines.append(f"- 图像 pooling：`{run.image_pooling}`；文本 pooling：`{run.text_pooling}`")
        md_lines.append(f"- 可视化样本数：`{stats['num_pairs']}` image-text pairs")
        for plot_path in plot_paths:
            md_lines.append(f"- 降维图：[{plot_path.name}]({plot_path.resolve()})")
        md_lines.append("")
        md_lines.append("### 统计指标")
        md_lines.append("")
        md_lines.append(f"- Pair similarity mean：`{stats['paired_similarity_mean']:.4f}`")
        md_lines.append(f"- Random similarity mean：`{stats['random_similarity_mean']:.4f}`")
        md_lines.append(f"- Pair-random margin：`{stats['pair_random_margin_mean']:.4f}`")
        md_lines.append(f"- Sample Text-to-Image top1：`{stats['text_to_image_top1_on_sample']:.3f}`")
        md_lines.append(f"- Sample Image-to-Text top1：`{stats['image_to_text_top1_on_sample']:.3f}`")
        md_lines.append(f"- Cross-modal kNN ratio：`{stats['cross_modal_knn_ratio']:.3f}`")
        md_lines.append(f"- Semantic silhouette：`{stats['semantic_silhouette_cosine']}`")
        md_lines.append("")
        md_lines.append("### 针对三个可视化问题的初步判断")
        md_lines.append("")
        for note in qualitative_notes(stats):
            md_lines.append(f"- {note}")
        md_lines.append("")

        md_lines.append("### Text-to-Image 成功案例")
        md_lines.append("")
        for caption_index in success_cases:
            record = records[caption_index]
            md_lines.append(f"- 查询文本：`{record.caption}`")
            md_lines.extend(format_neighbor_items(top_images_for_text(matrix, caption_index, images, records, args.top_k)))
        md_lines.append("")

        md_lines.append("### Text-to-Image hard negative 案例")
        md_lines.append("")
        for caption_index in failure_cases:
            record = records[caption_index]
            md_lines.append(f"- 查询文本：`{record.caption}`")
            md_lines.extend(format_neighbor_items(top_images_for_text(matrix, caption_index, images, records, args.top_k)))
        md_lines.append("")

        md_lines.append("### Image-to-Text 案例")
        md_lines.append("")
        for image_index in range(min(args.num_cases, len(images), image_embeddings.shape[0])):
            image = images[image_index]
            md_lines.append(f"- 查询图像：`{image.file_name}`，首条参考：`{image.captions[0]}`")
            md_lines.extend(format_neighbor_items(top_texts_for_image(matrix, image_index, images, records, args.top_k)))
        md_lines.append("")

    save_json(output_dir / "results" / "task3_summary.json", {"runs": summary})
    report_path = output_dir / "task3_representation_analysis.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(md_lines).rstrip() + "\n", encoding="utf-8")
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
