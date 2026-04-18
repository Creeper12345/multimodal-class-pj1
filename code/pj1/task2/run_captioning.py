"""Run Task 2 COCO image captioning evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from code.pj1.runtime import configure_hf_endpoint, configure_offline_mode, configure_runtime_env
from code.pj1.task2.coco import build_reference_mapping, load_coco_caption_samples, validate_image_files
from code.pj1.task2.metrics import evaluate_captions
from code.pj1.task2.models import load_model_from_spec


DEFAULT_MODEL_SPECS = (
    "lavis:blip_caption:base_coco",
    "lavis:blip2_opt:caption_coco_opt2.7b",
)
DEFAULT_METRICS = ("Bleu_4", "CIDEr")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", default="datasets/annotations/captions_val2017.json")
    parser.add_argument("--image-dir", default="datasets/val2017")
    parser.add_argument("--output-dir", default="outputs/task2_captioning")
    parser.add_argument("--model-spec", action="append", default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=30)
    parser.add_argument("--min-length", type=int, default=1)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--num-captions", type=int, default=1)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--use-nucleus-sampling", action="store_true")
    parser.add_argument("--metric", action="append", default=None)
    parser.add_argument(
        "--tokenizer-fallback",
        choices=("identity", "error"),
        default="identity",
        help="Fallback behavior when PTBTokenizer/Java is unavailable.",
    )
    parser.add_argument("--strict-metrics", action="store_true")
    parser.add_argument("--skip-metrics", action="store_true")
    parser.add_argument("--overwrite-predictions", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--bert-tokenizer-path", default=None)
    parser.add_argument("--hf-endpoint", default=None)
    return parser.parse_args()


def safe_name(spec: str, max_images: int | None, generation_config: dict[str, Any]) -> str:
    base = spec.replace(":", "_").replace("/", "_")
    suffix = (
        f"n{max_images or 'all'}"
        f"_beam{generation_config['num_beams']}"
        f"_max{generation_config['max_length']}"
        f"_min{generation_config['min_length']}"
        f"_sample{int(generation_config['use_nucleus_sampling'])}"
    )
    digest = hashlib.sha1(f"{spec}:{suffix}:{generation_config.get('prompt')}".encode("utf-8")).hexdigest()[:8]
    return f"{base}_{suffix}_{digest}"


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def append_summary(summary_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_or_generate_predictions(
    spec: str,
    image_ids: list[int],
    image_paths: list[Path],
    output_dir: Path,
    args: argparse.Namespace,
    generation_config: dict[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    run_name = safe_name(spec, args.max_images, generation_config)
    prediction_path = output_dir / "predictions" / f"{run_name}.json"
    if prediction_path.exists() and not args.overwrite_predictions:
        print(f"Loading cached predictions: {prediction_path}")
        with prediction_path.open("r", encoding="utf-8") as f:
            return json.load(f), run_name

    print(f"Generating predictions: {prediction_path}")
    model = load_model_from_spec(
        spec=spec,
        device=args.device,
        local_files_only=args.local_files_only,
        bert_tokenizer_path=args.bert_tokenizer_path,
    )
    captions = model.generate_captions(
        image_paths=image_paths,
        batch_size=args.batch_size,
        generation_config=generation_config,
    )
    predictions = [
        {"image_id": image_id, "caption": caption}
        for image_id, caption in zip(image_ids, captions, strict=True)
    ]
    save_json(prediction_path, predictions)
    return predictions, run_name


def prediction_mapping(predictions: list[dict[str, Any]]) -> dict[int, list[dict[str, str]]]:
    return {
        int(item["image_id"]): [{"caption": str(item["caption"]).strip()}]
        for item in predictions
    }


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    result_dir = output_dir / "results"
    configure_runtime_env(output_dir)
    configure_hf_endpoint(args.hf_endpoint)
    configure_offline_mode(args.local_files_only)

    samples = load_coco_caption_samples(
        annotation_path=args.annotation,
        image_dir=args.image_dir,
        max_images=args.max_images,
    )
    if not samples:
        raise RuntimeError("No COCO caption samples were loaded.")
    missing = validate_image_files(samples)
    if missing:
        preview = ", ".join(str(path) for path in missing[:5])
        raise FileNotFoundError(f"Missing {len(missing)} image files. First missing files: {preview}")

    dataset_summary = {
        "annotation": str(args.annotation),
        "image_dir": str(args.image_dir),
        "num_images": len(samples),
        "num_references": sum(len(sample.references) for sample in samples),
        "max_images": args.max_images,
    }
    save_json(result_dir / "dataset_summary.json", dataset_summary)
    print(json.dumps(dataset_summary, indent=2, ensure_ascii=False))
    if args.dry_run:
        print("Dry run complete; no model was loaded.")
        return 0

    generation_config = {
        "num_beams": args.num_beams,
        "max_length": args.max_length,
        "min_length": args.min_length,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "length_penalty": args.length_penalty,
        "temperature": args.temperature,
        "num_captions": args.num_captions,
        "use_nucleus_sampling": args.use_nucleus_sampling,
        "prompt": args.prompt,
    }
    model_specs = args.model_spec or list(DEFAULT_MODEL_SPECS)
    metrics = args.metric or list(DEFAULT_METRICS)
    image_ids = [sample.image_id for sample in samples]
    image_paths = [sample.image_path for sample in samples]
    references = build_reference_mapping(samples)

    rows: list[dict[str, Any]] = []
    for spec in model_specs:
        start = perf_counter()
        print(f"Running caption evaluation for {spec}")
        predictions, run_name = load_or_generate_predictions(
            spec=spec,
            image_ids=image_ids,
            image_paths=image_paths,
            output_dir=output_dir,
            args=args,
            generation_config=generation_config,
        )
        metric_payload: dict[str, Any] = {}
        warnings: list[str] = []
        if not args.skip_metrics:
            print(f"Evaluating caption metrics for {run_name}: {', '.join(metrics)}")
            evaluated = evaluate_captions(
                references=references,
                predictions=prediction_mapping(predictions),
                metrics=metrics,
                tokenizer_fallback=args.tokenizer_fallback,
                strict=args.strict_metrics,
            )
            metric_payload = evaluated.scores
            warnings = list(evaluated.warnings)

        elapsed_seconds = perf_counter() - start
        payload = {
            "model_spec": spec,
            "run_name": run_name,
            "dataset": dataset_summary,
            "generation_config": generation_config,
            "metrics": metric_payload,
            "warnings": warnings,
            "elapsed_seconds": elapsed_seconds,
            "prediction_path": str((output_dir / "predictions" / f"{run_name}.json").resolve()),
        }
        save_json(result_dir / f"{run_name}.json", payload)
        row = {
            "model_spec": spec,
            "run_name": run_name,
            "num_images": len(samples),
            **{metric: metric_payload.get(metric) for metric in metrics},
            "elapsed_seconds": round(elapsed_seconds, 3),
        }
        rows.append(row)
        print(json.dumps(payload, indent=2, ensure_ascii=False))

    append_summary(result_dir / "summary.csv", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
