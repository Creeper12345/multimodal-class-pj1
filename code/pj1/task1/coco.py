"""COCO caption loading utilities for retrieval evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class CocoImage:
    image_id: int
    file_name: str
    image_path: Path
    captions: tuple[str, ...]


@dataclass(frozen=True)
class CaptionRecord:
    caption_id: int
    image_id: int
    image_index: int
    caption: str


def load_coco_val_captions(
    annotation_path: str | Path,
    image_dir: str | Path,
    max_images: int | None = None,
) -> list[CocoImage]:
    """Load COCO images with all associated captions.

    The return value is sorted by COCO image id to make caches reproducible.
    """

    annotation_path = Path(annotation_path)
    image_dir = Path(image_dir)

    with annotation_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    captions_by_image: dict[int, list[tuple[int, str]]] = {}
    for ann in payload["annotations"]:
        image_id = int(ann["image_id"])
        captions_by_image.setdefault(image_id, []).append(
            (int(ann["id"]), str(ann["caption"]).strip())
        )

    items: list[CocoImage] = []
    for image in sorted(payload["images"], key=lambda item: int(item["id"])):
        image_id = int(image["id"])
        file_name = str(image["file_name"])
        captions = tuple(
            caption
            for _, caption in sorted(captions_by_image.get(image_id, []), key=lambda x: x[0])
        )
        if not captions:
            continue

        items.append(
            CocoImage(
                image_id=image_id,
                file_name=file_name,
                image_path=image_dir / file_name,
                captions=captions,
            )
        )
        if max_images is not None and len(items) >= max_images:
            break

    return items


def build_caption_records(images: Iterable[CocoImage]) -> list[CaptionRecord]:
    """Flatten image captions while preserving each caption's image index."""

    records: list[CaptionRecord] = []
    caption_id = 0
    for image_index, image in enumerate(images):
        for caption in image.captions:
            records.append(
                CaptionRecord(
                    caption_id=caption_id,
                    image_id=image.image_id,
                    image_index=image_index,
                    caption=caption,
                )
            )
            caption_id += 1
    return records


def validate_image_files(images: Iterable[CocoImage]) -> list[Path]:
    """Return missing image files, if any."""

    return [image.image_path for image in images if not image.image_path.exists()]

