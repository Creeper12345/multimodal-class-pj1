"""Analysis helpers for Task 1 retrieval embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from code.pj1.task1.coco import CocoImage, CaptionRecord
from code.pj1.task1.metrics import l2_normalize
from code.pj1.task1.run_retrieval import cache_paths


@dataclass(frozen=True)
class RetrievalExample:
    query_type: str
    query_id: int
    query_text: str
    ground_truth: str
    prediction: str
    is_correct: bool
    score: float


def load_cached_embeddings(output_dir: Path, run_name: str) -> tuple[np.ndarray, np.ndarray]:
    paths = cache_paths(output_dir / "cache", run_name)
    image_embeddings = np.load(paths["image_embeddings"])
    text_embeddings = np.load(paths["text_embeddings"])
    return image_embeddings, text_embeddings


def similarity_matrix(image_embeddings: np.ndarray, text_embeddings: np.ndarray) -> np.ndarray:
    image_embeddings = l2_normalize(image_embeddings)
    text_embeddings = l2_normalize(text_embeddings)
    return text_embeddings @ image_embeddings.T


def first_caption_indices_by_image(caption_records: Iterable[CaptionRecord], num_images: int) -> list[int]:
    first: list[int | None] = [None] * num_images
    for idx, record in enumerate(caption_records):
        if first[record.image_index] is None:
            first[record.image_index] = idx
    return [idx for idx in first if idx is not None]


def save_similarity_heatmap(
    matrix: np.ndarray,
    output_path: Path,
    title: str,
    max_items: int = 40,
) -> None:
    size = min(max_items, matrix.shape[0], matrix.shape[1])
    plot_matrix = matrix[:size, :size]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(plot_matrix, cmap="viridis", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Image Index")
    ax.set_ylabel("Text Index")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def collect_text_to_image_examples(
    matrix: np.ndarray,
    images: list[CocoImage],
    caption_records: list[CaptionRecord],
    num_examples: int = 5,
) -> tuple[list[RetrievalExample], list[RetrievalExample]]:
    top1 = matrix.argmax(axis=1)
    scores = matrix[np.arange(matrix.shape[0]), top1]
    correct: list[RetrievalExample] = []
    wrong: list[RetrievalExample] = []
    for idx, pred_image_idx in enumerate(top1):
        record = caption_records[idx]
        gt_image = images[record.image_index]
        pred_image = images[int(pred_image_idx)]
        item = RetrievalExample(
            query_type="text_to_image",
            query_id=idx,
            query_text=record.caption,
            ground_truth=gt_image.file_name,
            prediction=pred_image.file_name,
            is_correct=int(pred_image_idx) == record.image_index,
            score=float(scores[idx]),
        )
        if item.is_correct and len(correct) < num_examples:
            correct.append(item)
        if not item.is_correct and len(wrong) < num_examples:
            wrong.append(item)
        if len(correct) >= num_examples and len(wrong) >= num_examples:
            break
    return correct, wrong


def collect_image_to_text_examples(
    matrix: np.ndarray,
    images: list[CocoImage],
    caption_records: list[CaptionRecord],
    num_examples: int = 5,
) -> tuple[list[RetrievalExample], list[RetrievalExample]]:
    image_to_text_matrix = matrix.T
    top1 = image_to_text_matrix.argmax(axis=1)
    scores = image_to_text_matrix[np.arange(image_to_text_matrix.shape[0]), top1]
    correct: list[RetrievalExample] = []
    wrong: list[RetrievalExample] = []
    for image_idx, caption_idx in enumerate(top1):
        image = images[image_idx]
        pred_caption = caption_records[int(caption_idx)]
        gt_caption = image.captions[0]
        item = RetrievalExample(
            query_type="image_to_text",
            query_id=image_idx,
            query_text=image.file_name,
            ground_truth=gt_caption,
            prediction=pred_caption.caption,
            is_correct=pred_caption.image_index == image_idx,
            score=float(scores[image_idx]),
        )
        if item.is_correct and len(correct) < num_examples:
            correct.append(item)
        if not item.is_correct and len(wrong) < num_examples:
            wrong.append(item)
        if len(correct) >= num_examples and len(wrong) >= num_examples:
            break
    return correct, wrong
