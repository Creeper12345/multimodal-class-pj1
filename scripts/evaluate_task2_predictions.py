"""Evaluate saved Task 2 caption predictions without regenerating captions."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code.pj1.task2.coco import build_reference_mapping, load_coco_caption_samples
from code.pj1.task2.metrics import JAVA_METRICS_ENV, evaluate_captions
from code.pj1.progress import progress_iter


DEFAULT_METRICS = ("METEOR", "ROUGE_L", "SPICE")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", default="datasets/annotations/captions_val2017.json")
    parser.add_argument("--image-dir", default="datasets/val2017")
    parser.add_argument("--output-dir", default="outputs/task2_captioning")
    parser.add_argument(
        "--prediction",
        action="append",
        default=None,
        help="Prediction JSON path. Default: all full-split prediction files under output-dir/predictions.",
    )
    parser.add_argument("--metric", action="append", default=None)
    parser.add_argument(
        "--enable-java-metrics",
        action="store_true",
        help="Allow METEOR/SPICE scorers to run. Requires Java and local SPICE dependencies.",
    )
    parser.add_argument(
        "--tokenizer-fallback",
        choices=("identity", "error"),
        default="identity",
    )
    parser.add_argument("--strict-metrics", action="store_true")
    parser.add_argument("--summary-name", default="extra_metrics_summary.csv")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def prediction_mapping(predictions: list[dict[str, Any]]) -> dict[int, list[dict[str, str]]]:
    return {
        int(item["image_id"]): [{"caption": str(item["caption"]).strip()}]
        for item in predictions
    }


def default_prediction_paths(output_dir: Path) -> list[Path]:
    paths = sorted((output_dir / "predictions").glob("*_nall_*.json"))
    if not paths:
        paths = sorted((output_dir / "predictions").glob("*.json"))
    return paths


def infer_run_name(prediction_path: Path) -> str:
    return prediction_path.stem


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if args.enable_java_metrics:
        os.environ[JAVA_METRICS_ENV] = "1"

    metrics = args.metric or list(DEFAULT_METRICS)
    prediction_paths = [Path(path) for path in args.prediction] if args.prediction else default_prediction_paths(output_dir)
    if not prediction_paths:
        raise FileNotFoundError(f"No prediction JSON files found under {output_dir / 'predictions'}")

    samples = load_coco_caption_samples(args.annotation, args.image_dir)
    references = build_reference_mapping(samples)

    rows: list[dict[str, Any]] = []
    for prediction_path in progress_iter(
        prediction_paths,
        desc="Evaluate Task 2 predictions",
        total=len(prediction_paths),
        unit="file",
    ):
        predictions = load_json(prediction_path)
        run_name = infer_run_name(prediction_path)
        evaluated = evaluate_captions(
            references=references,
            predictions=prediction_mapping(predictions),
            metrics=metrics,
            tokenizer_fallback=args.tokenizer_fallback,
            strict=args.strict_metrics,
        )
        payload = {
            "run_name": run_name,
            "prediction_path": str(prediction_path),
            "metrics": evaluated.scores,
            "warnings": list(evaluated.warnings),
            "requested_metrics": metrics,
        }
        save_json(output_dir / "results" / f"{run_name}_extra_metrics.json", payload)
        rows.append(
            {
                "run_name": run_name,
                "num_predictions": len(predictions),
                **evaluated.scores,
                "warnings": " | ".join(evaluated.warnings),
            }
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))

    write_summary(output_dir / "results" / args.summary_name, rows)
    print(output_dir / "results" / args.summary_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
