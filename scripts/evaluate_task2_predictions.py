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


def merge_extra_payload(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return payload

    existing = load_json(path)
    merged_metrics = dict(existing.get("metrics", {}))
    merged_metrics.update(payload.get("metrics", {}))

    merged_warnings = list(existing.get("warnings", []))
    for warning in payload.get("warnings", []):
        if warning not in merged_warnings:
            merged_warnings.append(warning)

    merged_requested = list(existing.get("requested_metrics", []))
    for metric in payload.get("requested_metrics", []):
        if metric not in merged_requested:
            merged_requested.append(metric)

    existing.update(payload)
    existing["metrics"] = merged_metrics
    existing["warnings"] = merged_warnings
    existing["requested_metrics"] = merged_requested
    return existing


def merge_summary_rows(path: Path, new_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged_by_run: dict[str, dict[str, Any]] = {}
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                merged_by_run[str(row["run_name"])] = dict(row)

    for row in new_rows:
        run_name = str(row["run_name"])
        merged = merged_by_run.get(run_name, {})
        merged.update({key: value for key, value in row.items() if value is not None})
        merged_by_run[run_name] = merged

    return list(merged_by_run.values())


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
        extra_path = output_dir / "results" / f"{run_name}_extra_metrics.json"
        payload = {
            "run_name": run_name,
            "prediction_path": str(prediction_path),
            "metrics": evaluated.scores,
            "warnings": list(evaluated.warnings),
            "requested_metrics": metrics,
        }
        payload = merge_extra_payload(extra_path, payload)
        save_json(extra_path, payload)
        rows.append(
            {
                "run_name": run_name,
                "num_predictions": len(predictions),
                **payload["metrics"],
                "warnings": " | ".join(payload["warnings"]),
            }
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))

    summary_path = output_dir / "results" / args.summary_name
    rows = merge_summary_rows(summary_path, rows)
    write_summary(summary_path, rows)
    print(output_dir / "results" / args.summary_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
