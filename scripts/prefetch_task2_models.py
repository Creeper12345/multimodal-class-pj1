"""Prefetch Task 2 caption model weights and dependent tokenizers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from code.pj1.runtime import (
    configure_hf_endpoint,
    configure_offline_mode,
    configure_runtime_env,
    managed_bert_tokenizer_dir,
)
from code.pj1.task2.models import load_model_from_spec


DEFAULT_MODEL_SPECS = (
    "lavis:blip_caption:base_coco",
    "lavis:blip2_opt:caption_coco_opt2.7b",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/task2_captioning")
    parser.add_argument("--model-spec", action="append", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--bert-tokenizer-path", default=None)
    parser.add_argument("--hf-endpoint", default=None)
    parser.add_argument(
        "--smoke-generate",
        action="store_true",
        help="After loading the model, run one tiny caption-generation smoke test.",
    )
    parser.add_argument("--skip-generate", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    runtime_paths = configure_runtime_env(output_dir)
    configure_hf_endpoint(args.hf_endpoint)
    configure_offline_mode(args.local_files_only)

    bert_dir = managed_bert_tokenizer_dir(output_dir)
    model_specs = args.model_spec or list(DEFAULT_MODEL_SPECS)
    results: list[dict[str, object]] = []
    sample_image = Path("datasets/val2017/000000000139.jpg")
    generation_config = {
        "num_beams": 3,
        "max_length": 20,
        "min_length": 1,
        "top_p": 0.9,
        "repetition_penalty": 1.0,
        "length_penalty": 1.0,
        "temperature": 1.0,
        "num_captions": 1,
        "use_nucleus_sampling": False,
        "prompt": None,
    }

    for spec in model_specs:
        start = perf_counter()
        model = load_model_from_spec(
            spec=spec,
            device=args.device,
            local_files_only=args.local_files_only,
            bert_tokenizer_path=args.bert_tokenizer_path or str(bert_dir),
        )
        did_generate = False
        if args.smoke_generate and not args.skip_generate and sample_image.exists():
            model.generate_captions(
                [sample_image],
                batch_size=args.batch_size,
                generation_config=generation_config,
            )
            did_generate = True
        result = {
            "model_spec": spec,
            "adapter_name": model.name,
            "did_generate": did_generate,
            "elapsed_seconds": round(perf_counter() - start, 3),
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))

    payload = {
        "runtime_paths": runtime_paths.as_dict(),
        "bert_tokenizer_dir": str(bert_dir),
        "results": results,
    }
    save_json(output_dir / "results" / "prefetch_manifest.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
