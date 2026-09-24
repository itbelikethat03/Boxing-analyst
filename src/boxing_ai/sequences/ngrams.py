"""n-gram extraction and counting over token segments. Pure functions, no boxing knowledge."""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from typing import TypeVar

Ngram = tuple[str, ...]
K = TypeVar("K", bound=Hashable)


def ngrams(tokens: Sequence[str], n: int) -> Iterator[Ngram]:
    """Yield every contiguous ``n``-gram of ``tokens`` in order (none if the sequence is shorter)."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    for i in range(len(tokens) - n + 1):
        yield tuple(tokens[i : i + n])


def count_ngrams(segments: Iterable[Sequence[str]], n: int) -> Counter[Ngram]:
    """Count ``n``-grams per segment; an n-gram never spans two segments."""
    counts: Counter[Ngram] = Counter()
    for tokens in segments:
        counts.update(ngrams(tokens, n))
    return counts


def ranked(
    counts: Mapping[K, int], *, min_count: int = 1, limit: int | None = None
) -> list[tuple[K, int]]:
    """Items by count descending; ties broken by key ascending so the order is fully deterministic."""
    items = sorted(
        ((k, c) for k, c in counts.items() if c >= min_count), key=lambda kc: (-kc[1], kc[0])
    )
    return items if limit is None else items[:limit]
