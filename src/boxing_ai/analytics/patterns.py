"""Sequence patterns: common combinations, what follows an action, how bursts start and end.

All functions take plain events plus a ``StreamSpec`` (categories, tokenizer, burst gap) and optionally the
video's unobserved intervals. Callers choose whose events to pass: one fighter's own events give that fighter's
combinations; both fighters' events with an ``actor_tagged`` tokenizer give reactions ("what follows OPP:JAB").
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from boxing_ai.events import Event, UnobservedInterval
from boxing_ai.sequences import (
    DEFAULT_SPEC,
    Context,
    Ngram,
    StreamSpec,
    Transition,
    count_ngrams,
    followers,
    ranked,
    token_segments,
    transition_counts,
    transitions,
)


@dataclass(frozen=True)
class NgramCount:
    ngram: Ngram
    count: int


def top_ngrams(
    events: Iterable[Event],
    n: int,
    spec: StreamSpec = DEFAULT_SPEC,
    *,
    min_count: int = 1,
    limit: int | None = None,
    unobserved: Iterable[UnobservedInterval] = (),
) -> list[NgramCount]:
    """Most frequent ``n``-grams within bursts (e.g. ``n=2`` for ``JAB -> CROSS``)."""
    counts = count_ngrams(token_segments(events, spec, unobserved), n)
    return [NgramCount(g, c) for g, c in ranked(counts, min_count=min_count, limit=limit)]


def transition_probabilities(
    events: Iterable[Event],
    spec: StreamSpec = DEFAULT_SPEC,
    *,
    order: int = 1,
    min_count: int = 1,
    unobserved: Iterable[UnobservedInterval] = (),
) -> list[Transition]:
    """Every ``P(next | previous order tokens)`` incl. START/END, with counts (see ``sequences.transitions``)."""
    table = transition_counts(token_segments(events, spec, unobserved), order)
    return transitions(table, min_count=min_count)


def next_after(
    events: Iterable[Event],
    context: str | Context,
    spec: StreamSpec = DEFAULT_SPEC,
    *,
    min_count: int = 1,
    unobserved: Iterable[UnobservedInterval] = (),
) -> list[Transition]:
    """What most commonly follows ``context`` (a token, or a tuple of tokens for higher order)."""
    order = 1 if isinstance(context, str) else len(context)
    table = transition_counts(token_segments(events, spec, unobserved), order)
    return followers(table, context, min_count=min_count)


def entry_patterns(
    events: Iterable[Event],
    k: int = 1,
    spec: StreamSpec = DEFAULT_SPEC,
    *,
    min_count: int = 1,
    limit: int | None = None,
    unobserved: Iterable[UnobservedInterval] = (),
) -> list[NgramCount]:
    """The first ``k`` tokens of bursts, most common first. Bursts shorter than ``k`` are not counted."""
    return _edge_patterns(events, k, spec, min_count, limit, unobserved, at_end=False)


def exit_patterns(
    events: Iterable[Event],
    k: int = 1,
    spec: StreamSpec = DEFAULT_SPEC,
    *,
    min_count: int = 1,
    limit: int | None = None,
    unobserved: Iterable[UnobservedInterval] = (),
) -> list[NgramCount]:
    """The last ``k`` tokens of bursts, most common first. Bursts shorter than ``k`` are not counted."""
    return _edge_patterns(events, k, spec, min_count, limit, unobserved, at_end=True)


def _edge_patterns(
    events: Iterable[Event],
    k: int,
    spec: StreamSpec,
    min_count: int,
    limit: int | None,
    unobserved: Iterable[UnobservedInterval],
    *,
    at_end: bool,
) -> list[NgramCount]:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    counts: Counter[Ngram] = Counter()
    for tokens in token_segments(events, spec, unobserved):
        if len(tokens) >= k:
            counts[tuple(tokens[-k:] if at_end else tokens[:k])] += 1
    return [NgramCount(g, c) for g, c in ranked(counts, min_count=min_count, limit=limit)]
