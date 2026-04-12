"""Model adapters for Task 1 embedding extraction."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Protocol, Sequence

import numpy as np
from PIL import Image

from code.pj1.runtime import configure_platform_env


configure_platform_env()


class EmbeddingModel(Protocol):
    name: str

    def encode_images(self, image_paths: Sequence[Path], batch_size: int) -> np.ndarray:
        ...

    def encode_texts(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        ...


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _as_numpy(tensor) -> np.ndarray:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Torch is required for model inference.") from exc
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().float().cpu().numpy()
    return np.asarray(tensor, dtype=np.float32)


def _normalize_np(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    denom = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(denom, eps)


def _pool_feature_array(array: np.ndarray, pooling: str) -> np.ndarray:
    if array.ndim == 2:
        return array
    if array.ndim != 3:
        raise ValueError(f"Expected 2D or 3D feature tensor, got shape {array.shape}.")
    if pooling == "first":
        return array[:, 0, :]
    if pooling == "mean":
        return array.mean(axis=1)
    if pooling == "max":
        return array.max(axis=1)
    raise ValueError(f"Unsupported pooling mode: {pooling}")


def _batched(items: Sequence, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


@dataclass
class LavisFeatureExtractor:
    """LAVIS feature extractor adapter.

    Model specs use `lavis:<name>:<model_type>`, for example:
    `lavis:clip_feature_extractor:base`,
    `lavis:blip_feature_extractor:base`,
    `lavis:blip2_feature_extractor:pretrain`.
    """

    model_name: str
    model_type: str
    device: str
    image_pooling: str = "first"
    text_pooling: str = "first"

    def __post_init__(self) -> None:
        import torch
        from lavis.models import load_model_and_preprocess

        self.name = f"lavis_{self.model_name}_{self.model_type}"
        self.device = resolve_device(self.device)
        self._torch = torch
        self.model, self.vis_processors, self.txt_processors = load_model_and_preprocess(
            name=self.model_name,
            model_type=self.model_type,
            is_eval=True,
            device=self.device,
        )

    def _extract_projected(self, samples: dict, mode: str) -> np.ndarray:
        with self._torch.no_grad():
            try:
                features = self.model.extract_features(samples, mode=mode)
            except TypeError:
                # LAVIS CLIP feature extractor does not accept a `mode` kwarg.
                features = self.model.extract_features(samples)

        if isinstance(features, self._torch.Tensor):
            return _as_numpy(features)

        candidate_names = (
            ("image_embeds_proj", "image_embeds", "multimodal_embeds")
            if mode == "image"
            else ("text_embeds_proj", "text_embeds", "multimodal_embeds")
        )
        for name in candidate_names:
            value = getattr(features, name, None)
            if value is not None:
                return _as_numpy(value)
        available = sorted(name for name in dir(features) if not name.startswith("_"))
        raise RuntimeError(f"LAVIS feature output lacks projected {mode} embeddings. Available: {available}")

    def encode_images(self, image_paths: Sequence[Path], batch_size: int) -> np.ndarray:
        outputs: list[np.ndarray] = []
        processor = self.vis_processors["eval"]
        for batch_paths in _batched(list(image_paths), batch_size):
            images = []
            for path in batch_paths:
                raw = Image.open(path).convert("RGB")
                images.append(processor(raw))
            tensor = self._torch.stack(images, dim=0).to(self.device)
            array = self._extract_projected({"image": tensor}, mode="image")
            outputs.append(_pool_feature_array(array, self.image_pooling))
        return _normalize_np(np.concatenate(outputs, axis=0))

    def encode_texts(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        outputs: list[np.ndarray] = []
        processor = self.txt_processors["eval"]
        for batch_texts in _batched(list(texts), batch_size):
            processed = [processor(text) for text in batch_texts]
            array = self._extract_projected({"text_input": processed}, mode="text")
            outputs.append(_pool_feature_array(array, self.text_pooling))
        return _normalize_np(np.concatenate(outputs, axis=0))


@dataclass
class TransformersClipExtractor:
    model_id: str
    device: str

    def __post_init__(self) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.name = "transformers_" + self.model_id.replace("/", "_")
        self.device = resolve_device(self.device)
        self._torch = torch
        self.processor = CLIPProcessor.from_pretrained(self.model_id)
        self.model = CLIPModel.from_pretrained(self.model_id).to(self.device)
        self.model.eval()

    def encode_images(self, image_paths: Sequence[Path], batch_size: int) -> np.ndarray:
        outputs: list[np.ndarray] = []
        for batch_paths in _batched(list(image_paths), batch_size):
            images = [Image.open(path).convert("RGB") for path in batch_paths]
            inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
            with self._torch.no_grad():
                features = self.model.get_image_features(**inputs)
            outputs.append(_as_numpy(features))
        return _normalize_np(np.concatenate(outputs, axis=0))

    def encode_texts(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        outputs: list[np.ndarray] = []
        for batch_texts in _batched(list(texts), batch_size):
            inputs = self.processor(text=list(batch_texts), return_tensors="pt", padding=True, truncation=True).to(
                self.device
            )
            with self._torch.no_grad():
                features = self.model.get_text_features(**inputs)
            outputs.append(_as_numpy(features))
        return _normalize_np(np.concatenate(outputs, axis=0))


@dataclass
class OpenClipExtractor:
    model_name: str
    pretrained: str
    device: str

    def __post_init__(self) -> None:
        import open_clip
        import torch

        self.name = f"openclip_{self.model_name}_{self.pretrained}".replace("/", "_")
        self.device = resolve_device(self.device)
        self._torch = torch
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
            device=self.device,
        )
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        self.model.eval()

    def encode_images(self, image_paths: Sequence[Path], batch_size: int) -> np.ndarray:
        outputs: list[np.ndarray] = []
        for batch_paths in _batched(list(image_paths), batch_size):
            images = [self.preprocess(Image.open(path).convert("RGB")) for path in batch_paths]
            tensor = self._torch.stack(images, dim=0).to(self.device)
            with self._torch.no_grad():
                features = self.model.encode_image(tensor)
            outputs.append(_as_numpy(features))
        return _normalize_np(np.concatenate(outputs, axis=0))

    def encode_texts(self, texts: Sequence[str], batch_size: int) -> np.ndarray:
        outputs: list[np.ndarray] = []
        for batch_texts in _batched(list(texts), batch_size):
            tokens = self.tokenizer(list(batch_texts)).to(self.device)
            with self._torch.no_grad():
                features = self.model.encode_text(tokens)
            outputs.append(_as_numpy(features))
        return _normalize_np(np.concatenate(outputs, axis=0))


def load_model_from_spec(
    spec: str,
    device: str = "auto",
    image_pooling: str = "first",
    text_pooling: str = "first",
) -> EmbeddingModel:
    """Create a model adapter from a compact spec string."""

    parts = spec.split(":")
    backend = parts[0]

    if backend == "lavis":
        if len(parts) != 3:
            raise ValueError("LAVIS specs must be lavis:<name>:<model_type>.")
        return LavisFeatureExtractor(
            model_name=parts[1],
            model_type=parts[2],
            device=device,
            image_pooling=image_pooling,
            text_pooling=text_pooling,
        )

    if backend == "transformers-clip":
        if len(parts) != 2:
            raise ValueError("Transformers CLIP specs must be transformers-clip:<model_id>.")
        return TransformersClipExtractor(model_id=parts[1], device=device)

    if backend == "openclip":
        if len(parts) != 3:
            raise ValueError("OpenCLIP specs must be openclip:<model_name>:<pretrained>.")
        return OpenClipExtractor(model_name=parts[1], pretrained=parts[2], device=device)

    raise ValueError(f"Unknown model backend in spec: {spec}")
