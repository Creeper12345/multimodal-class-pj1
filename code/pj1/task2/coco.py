"""COCO caption loading helpers for Task 2 caption generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from code.pj1.task1.coco import CocoImage, load_coco_val_captions, validate_image_files


@dataclass(frozen=True)
class CaptionSample:
    image_id: int
    file_name: str
    image_path: Path
    references: tuple[str, ...]


def load_coco_caption_samples(
    annotation_path: str | Path,
    image_dir: str | Path,
    max_images: int | None = None,
) -> list[CaptionSample]:
    images: list[CocoImage] = load_coco_val_captions(
        annotation_path=annotation_path,
        image_dir=image_dir,
        max_images=max_images,
    )
    return [
        CaptionSample(
            image_id=image.image_id,
            file_name=image.file_name,
            image_path=image.image_path,
            references=image.captions,
        )
        for image in images
    ]


def build_reference_mapping(samples: list[CaptionSample]) -> dict[int, list[dict[str, str]]]:
    return {
        sample.image_id: [{"caption": reference} for reference in sample.references]
        for sample in samples
    }


__all__ = [
    "CaptionSample",
    "build_reference_mapping",
    "load_coco_caption_samples",
    "validate_image_files",
]
