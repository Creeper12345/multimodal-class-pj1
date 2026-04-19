"""Task 3 analysis helpers built on Task 1 embedding caches."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from code.pj1.progress import progress_iter
from code.pj1.task1.coco import CaptionRecord, CocoImage, build_caption_records, load_coco_val_captions
from code.pj1.task1.metrics import l2_normalize
from code.pj1.task1.run_retrieval import cache_paths


COCO_KEYWORDS = (
    "person", "people", "man", "woman", "child", "boy", "girl",
    "dog", "cat", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "bird", "bus", "train", "truck", "car", "motorcycle", "bicycle", "airplane",
    "boat", "traffic light", "stop sign", "bench", "chair", "couch", "bed",
    "table", "toilet", "tv", "laptop", "phone", "pizza", "cake", "donut",
    "sandwich", "skis", "snowboard", "surfboard", "tennis", "baseball",
    "kitchen", "bathroom", "street", "beach", "field", "room",
)


@dataclass(frozen=True)
class CacheRun:
    run_name: str
    spec: str
    num_images: int
    num_texts: int
    image_pooling: str
    text_pooling: str
    image_embeddings_path: Path
    text_embeddings_path: Path
    meta_path: Path


@dataclass(frozen=True)
class NeighborItem:
    rank: int
    score: float
    is_match: bool
    image_id: int
    image_file: str
    caption: str


def discover_cache_runs(task1_output_dir: Path, run_names: Sequence[str] | None = None) -> list[CacheRun]:
    selected = set(run_names or [])
    runs: list[CacheRun] = []
    for meta_path in sorted((task1_output_dir / "cache").glob("*_meta.json")):
        run_name = meta_path.name[: -len("_meta.json")]
        if selected and run_name not in selected:
            continue
        paths = cache_paths(task1_output_dir / "cache", run_name)
        if not paths["image_embeddings"].exists() or not paths["text_embeddings"].exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        runs.append(
            CacheRun(
                run_name=run_name,
                spec=str(meta["spec"]),
                num_images=int(meta["num_images"]),
                num_texts=int(meta["num_texts"]),
                image_pooling=str(meta.get("image_pooling", "first")),
                text_pooling=str(meta.get("text_pooling", "first")),
                image_embeddings_path=paths["image_embeddings"],
                text_embeddings_path=paths["text_embeddings"],
                meta_path=meta_path,
            )
        )
    return runs


def load_run_embeddings(run: CacheRun) -> tuple[np.ndarray, np.ndarray]:
    return np.load(run.image_embeddings_path), np.load(run.text_embeddings_path)


def load_run_dataset(annotation: str | Path, image_dir: str | Path, run: CacheRun) -> tuple[list[CocoImage], list[CaptionRecord]]:
    images = load_coco_val_captions(annotation, image_dir, max_images=run.num_images)
    records = build_caption_records(images)
    if len(records) > run.num_texts:
        records = records[: run.num_texts]
    return images, records


def first_caption_indices_by_image(records: Iterable[CaptionRecord], num_images: int) -> list[int]:
    first: list[int | None] = [None] * num_images
    for index, record in enumerate(records):
        if record.image_index < num_images and first[record.image_index] is None:
            first[record.image_index] = index
    return [idx for idx in first if idx is not None]


def semantic_label(caption: str) -> str:
    text = caption.lower()
    for keyword in COCO_KEYWORDS:
        if re.search(rf"\b{re.escape(keyword)}s?\b", text):
            return keyword
    return "other"


def paired_sample(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    images: list[CocoImage],
    records: list[CaptionRecord],
    sample_size: int,
) -> dict[str, Any]:
    first_caption_indices = first_caption_indices_by_image(records, min(len(images), image_embeddings.shape[0]))
    sample_size = min(sample_size, len(first_caption_indices), image_embeddings.shape[0])
    image_indices = np.arange(sample_size, dtype=np.int64)
    caption_indices = np.asarray(first_caption_indices[:sample_size], dtype=np.int64)
    captions = [records[int(idx)].caption for idx in caption_indices]
    labels = [semantic_label(caption) for caption in captions]
    return {
        "image_indices": image_indices,
        "caption_indices": caption_indices,
        "image_embeddings": image_embeddings[image_indices],
        "text_embeddings": text_embeddings[caption_indices],
        "captions": captions,
        "semantic_labels": labels,
    }


def reduce_embeddings(matrix: np.ndarray, method: str, random_state: int = 42) -> np.ndarray:
    if method == "pca":
        return PCA(n_components=2, random_state=random_state).fit_transform(matrix)
    if method == "tsne":
        perplexity = min(30, max(5, (matrix.shape[0] - 1) // 3))
        return TSNE(
            n_components=2,
            init="pca",
            learning_rate="auto",
            perplexity=perplexity,
            random_state=random_state,
        ).fit_transform(matrix)
    if method == "umap":
        try:
            import umap
        except ImportError as exc:
            raise RuntimeError("UMAP requested but umap-learn is not installed.") from exc
        return umap.UMAP(n_components=2, random_state=random_state).fit_transform(matrix)
    raise ValueError(f"Unsupported reduction method: {method}")


def save_embedding_plot(
    coords: np.ndarray,
    labels: list[str],
    output_path: Path,
    title: str,
    connect_pairs: int = 80,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "pj1_matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path("/tmp") / "pj1_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    half = coords.shape[0] // 2
    image_coords = coords[:half]
    text_coords = coords[half:]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(image_coords[:, 0], image_coords[:, 1], c="#1f77b4", s=18, alpha=0.72, label="image")
    ax.scatter(text_coords[:, 0], text_coords[:, 1], c="#ff7f0e", s=18, alpha=0.72, label="text")

    for idx in range(min(connect_pairs, half)):
        ax.plot(
            [image_coords[idx, 0], text_coords[idx, 0]],
            [image_coords[idx, 1], text_coords[idx, 1]],
            c="#999999",
            linewidth=0.35,
            alpha=0.35,
        )

    ax.set_title(title)
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    label_path = output_path.with_suffix(".labels.tsv")
    with label_path.open("w", encoding="utf-8") as f:
        f.write("pair_index\tsemantic_label\n")
        for idx, label in enumerate(labels):
            f.write(f"{idx}\t{label}\n")


def embedding_statistics(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    semantic_labels: Sequence[str],
    knn_k: int = 10,
) -> dict[str, float | int | None]:
    images = l2_normalize(image_embeddings)
    texts = l2_normalize(text_embeddings)
    pair_scores = np.sum(images * texts, axis=1)

    rng = np.random.default_rng(42)
    shuffled = rng.permutation(texts.shape[0])
    random_scores = np.sum(images * texts[shuffled], axis=1)

    combined = np.vstack([images, texts])
    modalities = np.asarray(["image"] * images.shape[0] + ["text"] * texts.shape[0])
    k = min(knn_k + 1, combined.shape[0])
    neighbors = NearestNeighbors(n_neighbors=k, metric="cosine").fit(combined)
    _, indices = neighbors.kneighbors(combined)
    cross_counts = []
    for row, modality in zip(indices[:, 1:], modalities):
        cross_counts.append(float(np.mean(modalities[row] != modality)))

    label_values = np.asarray(list(semantic_labels) + list(semantic_labels))
    valid_mask = label_values != "other"
    semantic_silhouette: float | None = None
    if valid_mask.sum() >= 10 and len(set(label_values[valid_mask])) >= 2:
        try:
            semantic_silhouette = float(
                silhouette_score(combined[valid_mask], label_values[valid_mask], metric="cosine")
            )
        except ValueError:
            semantic_silhouette = None

    text_to_image_top1 = (texts @ images.T).argmax(axis=1)
    image_to_text_top1 = (images @ texts.T).argmax(axis=1)
    arange = np.arange(images.shape[0])

    return {
        "num_pairs": int(images.shape[0]),
        "paired_similarity_mean": float(pair_scores.mean()),
        "paired_similarity_median": float(np.median(pair_scores)),
        "random_similarity_mean": float(random_scores.mean()),
        "pair_random_margin_mean": float((pair_scores - random_scores).mean()),
        "text_to_image_top1_on_sample": float(np.mean(text_to_image_top1 == arange)),
        "image_to_text_top1_on_sample": float(np.mean(image_to_text_top1 == arange)),
        "cross_modal_knn_ratio": float(np.mean(cross_counts)),
        "semantic_silhouette_cosine": semantic_silhouette,
    }


def top_texts_for_image(
    matrix_text_by_image: np.ndarray,
    image_index: int,
    images: list[CocoImage],
    records: list[CaptionRecord],
    top_k: int,
) -> list[NeighborItem]:
    scores = matrix_text_by_image[:, image_index]
    order = np.argsort(-scores)[:top_k]
    items: list[NeighborItem] = []
    for rank, caption_index in enumerate(order, start=1):
        record = records[int(caption_index)]
        image = images[record.image_index]
        items.append(
            NeighborItem(
                rank=rank,
                score=float(scores[int(caption_index)]),
                is_match=record.image_index == image_index,
                image_id=image.image_id,
                image_file=image.file_name,
                caption=record.caption,
            )
        )
    return items


def top_images_for_text(
    matrix_text_by_image: np.ndarray,
    caption_index: int,
    images: list[CocoImage],
    records: list[CaptionRecord],
    top_k: int,
) -> list[NeighborItem]:
    record = records[caption_index]
    scores = matrix_text_by_image[caption_index]
    order = np.argsort(-scores)[:top_k]
    items: list[NeighborItem] = []
    for rank, image_index in enumerate(order, start=1):
        image = images[int(image_index)]
        items.append(
            NeighborItem(
                rank=rank,
                score=float(scores[int(image_index)]),
                is_match=int(image_index) == record.image_index,
                image_id=image.image_id,
                image_file=image.file_name,
                caption=image.captions[0],
            )
        )
    return items


def similarity_text_by_image(image_embeddings: np.ndarray, text_embeddings: np.ndarray) -> np.ndarray:
    return l2_normalize(text_embeddings) @ l2_normalize(image_embeddings).T


def select_case_indices(matrix: np.ndarray, records: list[CaptionRecord], num_cases: int) -> tuple[list[int], list[int]]:
    text_top1 = matrix.argmax(axis=1)
    success = [idx for idx, image_idx in enumerate(text_top1) if int(image_idx) == records[idx].image_index]
    failure = [idx for idx, image_idx in enumerate(text_top1) if int(image_idx) != records[idx].image_index]
    return success[:num_cases], failure[:num_cases]
