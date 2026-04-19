"""Search Task 1 image embeddings with a free-form text query."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from code.pj1.runtime import configure_hf_endpoint, configure_offline_mode, configure_runtime_env
from code.pj1.task1.coco import load_coco_val_captions
from code.pj1.task1.metrics import l2_normalize
from code.pj1.task1.models import load_model_from_spec
from code.pj1.task3.analysis import discover_cache_runs, load_run_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", action="append", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--task1-output-dir", default="outputs/task1_retrieval")
    parser.add_argument("--annotation", default="datasets/annotations/captions_val2017.json")
    parser.add_argument("--image-dir", default="datasets/val2017")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--bert-tokenizer-path", default=None)
    parser.add_argument("--hf-endpoint", default=None)
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    task1_output_dir = Path(args.task1_output_dir)
    configure_runtime_env(task1_output_dir)
    configure_hf_endpoint(args.hf_endpoint)
    configure_offline_mode(args.local_files_only)

    runs = discover_cache_runs(task1_output_dir, [args.run_name])
    if len(runs) != 1:
        raise ValueError(f"Could not find exactly one cache run for {args.run_name}.")
    run = runs[0]
    image_embeddings, _ = load_run_embeddings(run)
    images = load_coco_val_captions(args.annotation, args.image_dir, max_images=run.num_images)

    model = load_model_from_spec(
        run.spec,
        device=args.device,
        image_pooling=run.image_pooling,
        text_pooling=run.text_pooling,
        local_files_only=args.local_files_only,
        bert_tokenizer_path=args.bert_tokenizer_path,
    )
    query_embeddings = model.encode_texts(args.query, batch_size=len(args.query))
    scores = l2_normalize(query_embeddings) @ l2_normalize(image_embeddings).T

    results = []
    for query_index, query in enumerate(args.query):
        order = np.argsort(-scores[query_index])[: args.top_k]
        items = []
        for rank, image_index in enumerate(order, start=1):
            image = images[int(image_index)]
            items.append(
                {
                    "rank": rank,
                    "score": float(scores[query_index, int(image_index)]),
                    "image_id": image.image_id,
                    "file_name": image.file_name,
                    "image_path": str(image.image_path),
                    "reference_captions": list(image.captions),
                }
            )
        results.append({"query": query, "run_name": run.run_name, "model_spec": run.spec, "results": items})

    payload = {"queries": results}
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
