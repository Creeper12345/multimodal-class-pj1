"""Retrieval metrics for image-text embedding matrices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class RetrievalMetrics:
    text_to_image: dict[str, float]
    image_to_text: dict[str, float]

    def flat(self) -> dict[str, float]:
        row: dict[str, float] = {}
        for name, value in self.text_to_image.items():
            row[f"text_to_image_{name}"] = value
        for name, value in self.image_to_text.items():
            row[f"image_to_text_{name}"] = value
        return row


def l2_normalize(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    denom = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(denom, eps)


def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Return top-k indices sorted by descending score for each row."""

    if k >= scores.shape[1]:
        top = np.argsort(-scores, axis=1)
        return top[:, :k]

    unsorted = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
    row_indices = np.arange(scores.shape[0])[:, None]
    order = np.argsort(-scores[row_indices, unsorted], axis=1)
    return unsorted[row_indices, order]


def _recall_from_topk(topk: np.ndarray, positives: Sequence[set[int]], ks: Sequence[int]) -> dict[str, float]:
    results: dict[str, float] = {}
    for k in ks:
        hits = 0
        for query_index, positive_indices in enumerate(positives):
            if positive_indices.intersection(topk[query_index, :k]):
                hits += 1
        results[f"R@{k}"] = 100.0 * hits / len(positives)
    return results


def evaluate_retrieval(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    caption_image_indices: Sequence[int],
    ks: Sequence[int] = (1, 5, 10),
    eval_batch_size: int = 512,
) -> RetrievalMetrics:
    """Evaluate bidirectional retrieval with Recall@K.

    Args:
        image_embeddings: shape [num_images, dim].
        text_embeddings: shape [num_captions, dim].
        caption_image_indices: image index for each caption row.
        ks: recall cutoffs.
        eval_batch_size: chunk size for similarity computation.
    """

    if image_embeddings.ndim != 2 or text_embeddings.ndim != 2:
        raise ValueError("Embeddings must be 2D arrays.")
    if image_embeddings.shape[1] != text_embeddings.shape[1]:
        raise ValueError(
            f"Embedding dimensions differ: {image_embeddings.shape[1]} vs {text_embeddings.shape[1]}"
        )
    if len(caption_image_indices) != text_embeddings.shape[0]:
        raise ValueError("caption_image_indices length must equal number of text embeddings.")
    if not ks:
        raise ValueError("At least one Recall@K cutoff is required.")

    max_k = max(ks)
    image_embeddings = l2_normalize(image_embeddings)
    text_embeddings = l2_normalize(text_embeddings)
    caption_image_indices = np.asarray(caption_image_indices, dtype=np.int64)

    text_positives = [{int(image_index)} for image_index in caption_image_indices]
    text_topk_parts: list[np.ndarray] = []
    for start in range(0, text_embeddings.shape[0], eval_batch_size):
        end = min(start + eval_batch_size, text_embeddings.shape[0])
        scores = text_embeddings[start:end] @ image_embeddings.T
        text_topk_parts.append(_topk_indices(scores, min(max_k, image_embeddings.shape[0])))
    text_topk = np.concatenate(text_topk_parts, axis=0)
    text_to_image = _recall_from_topk(text_topk, text_positives, ks)

    image_positives: list[set[int]] = [set() for _ in range(image_embeddings.shape[0])]
    for caption_index, image_index in enumerate(caption_image_indices):
        image_positives[int(image_index)].add(caption_index)

    image_topk_parts: list[np.ndarray] = []
    for start in range(0, image_embeddings.shape[0], eval_batch_size):
        end = min(start + eval_batch_size, image_embeddings.shape[0])
        scores = image_embeddings[start:end] @ text_embeddings.T
        image_topk_parts.append(_topk_indices(scores, min(max_k, text_embeddings.shape[0])))
    image_topk = np.concatenate(image_topk_parts, axis=0)
    image_to_text = _recall_from_topk(image_topk, image_positives, ks)

    return RetrievalMetrics(text_to_image=text_to_image, image_to_text=image_to_text)

