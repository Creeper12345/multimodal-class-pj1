"""Small tqdm wrappers used by long-running PJ1 scripts."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TypeVar

from tqdm.auto import tqdm


T = TypeVar("T")


def progress_iter(
    iterable: Iterable[T],
    *,
    desc: str,
    total: int | None = None,
    unit: str = "it",
) -> Iterator[T]:
    """Wrap an iterable with a standard progress bar."""

    yield from tqdm(iterable, desc=desc, total=total, unit=unit)


def num_batches(num_items: int, batch_size: int) -> int:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    return (num_items + batch_size - 1) // batch_size
