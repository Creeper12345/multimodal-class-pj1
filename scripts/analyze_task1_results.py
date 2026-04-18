"""Generate Task 1 similarity heatmaps and retrieval example markdown from cached embeddings."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code.pj1.task1.analysis import (
    collect_image_to_text_examples,
    collect_text_to_image_examples,
    first_caption_indices_by_image,
    load_cached_embeddings,
    save_similarity_heatmap,
    similarity_matrix,
)
from code.pj1.task1.coco import build_caption_records, load_coco_val_captions
from code.pj1.progress import progress_iter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", default="datasets/annotations/captions_val2017.json")
    parser.add_argument("--image-dir", default="datasets/val2017")
    parser.add_argument("--output-dir", default="outputs/task1_retrieval")
    parser.add_argument("--sample-size", type=int, default=40)
    parser.add_argument("--num-examples", type=int, default=5)
    parser.add_argument(
        "--run-name",
        action="append",
        default=None,
        help="Specific run_name values to analyze. Default: all runs in summary.csv that have cached embeddings.",
    )
    return parser.parse_args()


def load_summary_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def discover_run_names(output_dir: Path) -> list[str]:
    run_names: list[str] = []
    for meta_path in sorted((output_dir / "cache").glob("*_meta.json")):
        run_name = meta_path.name[: -len("_meta.json")]
        image_path = output_dir / "cache" / f"{run_name}_image_embeddings.npy"
        text_path = output_dir / "cache" / f"{run_name}_text_embeddings.npy"
        result_path = output_dir / "results" / f"{run_name}.json"
        if image_path.exists() and text_path.exists() and result_path.exists():
            run_names.append(run_name)
    return run_names


def format_examples(title: str, examples: list) -> list[str]:
    lines = [f"### {title}", ""]
    if not examples:
        lines.append("- 无可用样例。")
        lines.append("")
        return lines
    for item in examples:
        lines.append(f"- 查询：`{item.query_text}`")
        lines.append(f"  - Ground truth：`{item.ground_truth}`")
        lines.append(f"  - Prediction：`{item.prediction}`")
        lines.append(f"  - Correct：`{item.is_correct}`")
        lines.append(f"  - Score：`{item.score:.4f}`")
    lines.append("")
    return lines


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    analysis_dir = output_dir / "analysis"
    summary_path = output_dir / "results" / "summary.csv"
    summary_rows = load_summary_rows(summary_path) if summary_path.exists() else []
    selected_run_names = set(args.run_name or discover_run_names(output_dir))

    images = load_coco_val_captions(args.annotation, args.image_dir)
    caption_records = build_caption_records(images)
    first_caption_indices = first_caption_indices_by_image(caption_records, len(images))

    md_lines = ["# Task 1 相似度矩阵与案例分析", ""]
    ordered_run_names = []
    if summary_rows:
        ordered_run_names.extend(
            [row["run_name"] for row in summary_rows if row["run_name"] in selected_run_names]
        )
    ordered_run_names.extend(
        [name for name in sorted(selected_run_names) if name not in ordered_run_names]
    )

    for run_name in progress_iter(
        ordered_run_names,
        desc="Analyze Task 1 runs",
        total=len(ordered_run_names),
        unit="run",
    ):
        if run_name not in selected_run_names:
            continue

        cache_image = output_dir / "cache" / f"{run_name}_image_embeddings.npy"
        cache_text = output_dir / "cache" / f"{run_name}_text_embeddings.npy"
        if not (cache_image.exists() and cache_text.exists()):
            md_lines.append(f"## {run_name}")
            md_lines.append("")
            md_lines.append("- 未找到本地 embedding cache，跳过矩阵与案例分析。")
            md_lines.append("")
            continue

        image_embeddings, text_embeddings = load_cached_embeddings(output_dir, run_name)
        sampled_caption_indices = first_caption_indices[: args.sample_size]
        sampled_image_embeddings = image_embeddings[: len(sampled_caption_indices)]
        sampled_text_embeddings = text_embeddings[sampled_caption_indices]
        matrix = similarity_matrix(sampled_image_embeddings, sampled_text_embeddings)

        heatmap_path = analysis_dir / f"{run_name}_sim_matrix.png"
        save_similarity_heatmap(
            matrix,
            output_path=heatmap_path,
            title=f"{run_name} similarity matrix",
            max_items=args.sample_size,
        )

        full_matrix = similarity_matrix(image_embeddings, text_embeddings)
        t2i_ok, t2i_bad = collect_text_to_image_examples(
            full_matrix, images, caption_records, num_examples=args.num_examples
        )
        i2t_ok, i2t_bad = collect_image_to_text_examples(
            full_matrix, images, caption_records, num_examples=args.num_examples
        )

        result_json = load_json(output_dir / "results" / f"{run_name}.json")
        md_lines.append(f"## {run_name}")
        md_lines.append("")
        md_lines.append(f"- 模型：`{result_json['model_spec']}`")
        md_lines.append(
            f"- Text-to-Image：R@1={result_json['metrics']['text_to_image']['R@1']:.2f}, "
            f"R@5={result_json['metrics']['text_to_image']['R@5']:.2f}, "
            f"R@10={result_json['metrics']['text_to_image']['R@10']:.2f}"
        )
        md_lines.append(
            f"- Image-to-Text：R@1={result_json['metrics']['image_to_text']['R@1']:.2f}, "
            f"R@5={result_json['metrics']['image_to_text']['R@5']:.2f}, "
            f"R@10={result_json['metrics']['image_to_text']['R@10']:.2f}"
        )
        md_lines.append(f"- 相似度矩阵图：[{heatmap_path.name}]({heatmap_path.resolve()})")
        md_lines.append("")
        md_lines.extend(format_examples("Text-to-Image 成功样例", t2i_ok))
        md_lines.extend(format_examples("Text-to-Image 失败样例", t2i_bad))
        md_lines.extend(format_examples("Image-to-Text 成功样例", i2t_ok))
        md_lines.extend(format_examples("Image-to-Text 失败样例", i2t_bad))

    analysis_dir.mkdir(parents=True, exist_ok=True)
    md_path = analysis_dir / "task1_similarity_analysis.md"
    md_path.write_text("\n".join(md_lines).rstrip() + "\n", encoding="utf-8")
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
