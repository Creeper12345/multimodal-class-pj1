"""Prefetch Task 1 model weights and tokenizer files into local caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code.pj1.runtime import configure_hf_endpoint, configure_runtime_env, managed_bert_tokenizer_dir
from code.pj1.progress import progress_iter
from code.pj1.task1.coco import load_coco_val_captions
from code.pj1.task1.models import load_model_from_spec


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
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-batch-size", type=int, default=1)
    parser.add_argument("--text-batch-size", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=1)
    parser.add_argument("--skip-encode", action="store_true")
    parser.add_argument("--model-spec", action="append", default=None)
    parser.add_argument(
        "--bert-tokenizer-path",
        default=None,
        help="Existing local bert-base-uncased tokenizer directory. If given, it will be copied into the managed cache.",
    )
    parser.add_argument(
        "--hf-endpoint",
        default=None,
        help="Optional Hugging Face mirror endpoint, for example https://hf-mirror.com",
    )
    return parser.parse_args()


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def ensure_local_bert_tokenizer(output_dir: Path, source_path: str | None) -> Path:
    target_dir = managed_bert_tokenizer_dir(output_dir)
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    if source_path:
        source_dir = Path(source_path).resolve()
        if not source_dir.exists():
            raise FileNotFoundError(f"Local tokenizer path does not exist: {source_dir}")
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(source_dir, target_dir)
        return target_dir

    if target_dir.exists():
        return target_dir

    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenizer.save_pretrained(target_dir)
    return target_dir


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    runtime_paths = configure_runtime_env(output_dir)
    configure_hf_endpoint(args.hf_endpoint)
    model_specs = args.model_spec or list(DEFAULT_MODEL_SPECS)
    bert_tokenizer_dir = ensure_local_bert_tokenizer(output_dir, args.bert_tokenizer_path)

    images = load_coco_val_captions(
        args.annotation,
        args.image_dir,
        max_images=args.max_images,
    )
    sample_image_paths = [item.image_path for item in images]
    sample_texts = [images[0].captions[0]] if images else ["a photo"]

    results: list[dict] = []
    for spec in progress_iter(model_specs, desc="Prefetch Task 1 models", unit="model"):
        start = perf_counter()
        print(f"Loading model files for {spec}")
        model = load_model_from_spec(
            spec,
            device=args.device,
            local_files_only=False,
            bert_tokenizer_path=str(bert_tokenizer_dir),
        )
        if not args.skip_encode:
            if sample_image_paths:
                model.encode_images(sample_image_paths, batch_size=args.image_batch_size)
            if sample_texts:
                model.encode_texts(sample_texts, batch_size=args.text_batch_size)
        results.append(
            {
                "model_spec": spec,
                "adapter_name": model.name,
                "did_encode": not args.skip_encode,
                "elapsed_seconds": round(perf_counter() - start, 3),
            }
        )
        print(json.dumps(results[-1], ensure_ascii=False))

    manifest = {
        "runtime_paths": runtime_paths.as_dict(),
        "bert_tokenizer_dir": str(bert_tokenizer_dir),
        "results": results,
    }
    save_json(output_dir / "results" / "prefetch_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
