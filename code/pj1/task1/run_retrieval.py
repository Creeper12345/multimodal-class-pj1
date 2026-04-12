"""Run Task 1 COCO image-text retrieval evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from code.pj1.task1.coco import build_caption_records, load_coco_val_captions, validate_image_files
from code.pj1.task1.metrics import evaluate_retrieval
from code.pj1.task1.models import load_model_from_spec
from code.pj1.runtime import configure_runtime_env


DEFAULT_MODEL_SPECS = (
    "lavis:clip_feature_extractor:base",
    "lavis:blip_retrieval:coco",
    "lavis:blip2_feature_extractor:pretrain",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", default="datasets/annotations/captions_val2017.json")
    parser.add_argument("--image-dir", default="datasets/val2017")
    parser.add_argument("--output-dir", default="outputs/task1_retrieval")
    parser.add_argument("--model-spec", action="append", default=None)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--image-batch-size", type=int, default=8)
    parser.add_argument("--text-batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-pooling", choices=("first", "mean", "max"), default="first")
    parser.add_argument("--text-pooling", choices=("first", "mean", "max"), default="first")
    parser.add_argument("--overwrite-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def safe_name(spec: str, max_images: int | None, image_pooling: str, text_pooling: str) -> str:
    base = spec.replace(":", "_").replace("/", "_")
    suffix = f"n{max_images or 'all'}_{image_pooling}_{text_pooling}"
    digest = hashlib.sha1(f"{spec}:{suffix}".encode("utf-8")).hexdigest()[:8]
    return f"{base}_{suffix}_{digest}"


def cache_paths(cache_dir: Path, name: str) -> dict[str, Path]:
    return {
        "image_embeddings": cache_dir / f"{name}_image_embeddings.npy",
        "text_embeddings": cache_dir / f"{name}_text_embeddings.npy",
        "meta": cache_dir / f"{name}_meta.json",
    }


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def load_or_extract_embeddings(
    spec: str,
    image_paths: list[Path],
    texts: list[str],
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, str]:
    name = safe_name(spec, args.max_images, args.image_pooling, args.text_pooling)
    paths = cache_paths(output_dir / "cache", name)
    cache_exists = paths["image_embeddings"].exists() and paths["text_embeddings"].exists()

    if cache_exists and not args.overwrite_cache:
        image_embeddings = np.load(paths["image_embeddings"])
        text_embeddings = np.load(paths["text_embeddings"])
        return image_embeddings, text_embeddings, name

    model = load_model_from_spec(
        spec,
        device=args.device,
        image_pooling=args.image_pooling,
        text_pooling=args.text_pooling,
    )
    image_embeddings = model.encode_images(image_paths, batch_size=args.image_batch_size)
    text_embeddings = model.encode_texts(texts, batch_size=args.text_batch_size)

    paths["image_embeddings"].parent.mkdir(parents=True, exist_ok=True)
    np.save(paths["image_embeddings"], image_embeddings.astype(np.float32))
    np.save(paths["text_embeddings"], text_embeddings.astype(np.float32))
    save_json(
        paths["meta"],
        {
            "spec": spec,
            "adapter_name": model.name,
            "num_images": len(image_paths),
            "num_texts": len(texts),
            "image_embedding_shape": list(image_embeddings.shape),
            "text_embedding_shape": list(text_embeddings.shape),
            "image_pooling": args.image_pooling,
            "text_pooling": args.text_pooling,
        },
    )
    return image_embeddings, text_embeddings, name


def append_summary(summary_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    result_dir = output_dir / "results"
    configure_runtime_env(output_dir)

    images = load_coco_val_captions(args.annotation, args.image_dir, max_images=args.max_images)
    if not images:
        raise RuntimeError("No COCO caption records were loaded.")
    missing = validate_image_files(images)
    if missing:
        preview = ", ".join(str(path) for path in missing[:5])
        raise FileNotFoundError(f"Missing {len(missing)} image files. First missing files: {preview}")

    caption_records = build_caption_records(images)
    image_paths = [image.image_path for image in images]
    texts = [record.caption for record in caption_records]
    caption_image_indices = [record.image_index for record in caption_records]

    dataset_summary = {
        "annotation": str(args.annotation),
        "image_dir": str(args.image_dir),
        "num_images": len(images),
        "num_captions": len(caption_records),
        "max_images": args.max_images,
    }
    save_json(result_dir / "dataset_summary.json", dataset_summary)
    print(json.dumps(dataset_summary, indent=2, ensure_ascii=False))

    if args.dry_run:
        print("Dry run complete; no model was loaded.")
        return 0

    model_specs = args.model_spec or list(DEFAULT_MODEL_SPECS)
    rows: list[dict[str, Any]] = []
    for spec in model_specs:
        start = perf_counter()
        print(f"Running retrieval for {spec}")
        image_embeddings, text_embeddings, run_name = load_or_extract_embeddings(
            spec=spec,
            image_paths=image_paths,
            texts=texts,
            output_dir=output_dir,
            args=args,
        )
        metrics = evaluate_retrieval(
            image_embeddings=image_embeddings,
            text_embeddings=text_embeddings,
            caption_image_indices=caption_image_indices,
            eval_batch_size=args.eval_batch_size,
        )
        elapsed_seconds = perf_counter() - start
        payload = {
            "model_spec": spec,
            "run_name": run_name,
            "dataset": dataset_summary,
            "metrics": {
                "text_to_image": metrics.text_to_image,
                "image_to_text": metrics.image_to_text,
            },
            "elapsed_seconds": elapsed_seconds,
        }
        save_json(result_dir / f"{run_name}.json", payload)
        row = {
            "model_spec": spec,
            "run_name": run_name,
            "num_images": len(images),
            "num_captions": len(caption_records),
            **metrics.flat(),
            "elapsed_seconds": round(elapsed_seconds, 3),
        }
        rows.append(row)
        print(json.dumps(row, indent=2, ensure_ascii=False))

    append_summary(result_dir / "summary.csv", rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
