"""Sequence patterns: common combinations, what follows an action, how bursts start and end.

All functions take plain events plus a ``StreamSpec`` (categories, tokenizer, burst gap) and optionally the
video's unobserved intervals. Callers choose whose events to pass: one fighter's own events give that fighter's
combinations; both fighters' events with an ``actor_tagged`` tokenizer give reactions ("what follows OPP:JAB").
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from boxing_ai.events import Event, UnobservedInterval
from boxing_ai.ontology import Category, Outcome
from boxing_ai.sequences import (
    DEFAULT_SPEC,
    Context,
    Ngram,
    StreamSpec,
    Transition,
    count_ngrams,
    followers,
    ranked,
    segment,
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


@dataclass(frozen=True)
class PatternOutcome:
    """How often an ``n``-gram's *final* action was a punch that landed (the pattern's payoff).

    ``known`` counts occurrences ending in a punch with a recorded outcome; patterns ending in a feint or a
    defensive action have ``known == 0`` and no landed rate.
    """

    ngram: Ngram
    count: int
    known: int
    landed: int

    @property
    def landed_rate(self) -> float | None:
        return None if self.known == 0 else self.landed / self.known


def pattern_outcomes(
    events: Iterable[Event],
    n: int,
    spec: StreamSpec = DEFAULT_SPEC,
    *,
    min_count: int = 1,
    limit: int | None = None,
    unobserved: Iterable[UnobservedInterval] = (),
) -> list[PatternOutcome]:
    """``top_ngrams`` plus whether each occurrence ended in a landed punch; same ranking and bursts."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    tally: dict[Ngram, list[int]] = {}  # ngram -> [count, known, landed]
    for burst in segment(events, spec, unobserved):
        for i in range(len(burst) - n + 1):
            window = burst[i : i + n]
            row = tally.setdefault(tuple(spec.tokenizer(e) for e in window), [0, 0, 0])
            row[0] += 1
            last = window[-1]
            if last.category is Category.PUNCH and last.outcome is not None:
                row[1] += 1
                row[2] += last.outcome is Outcome.LANDED
    counts = {g: row[0] for g, row in tally.items()}
    return [
        PatternOutcome(g, c, tally[g][1], tally[g][2])
        for g, c in ranked(counts, min_count=min_count, limit=limit)
    ]


def pattern_occurrences(
    events: Iterable[Event],
    ngram: Sequence[str],
    spec: StreamSpec = DEFAULT_SPEC,
    *,
    unobserved: Iterable[UnobservedInterval] = (),
) -> list[tuple[Event, ...]]:
    """Every place ``ngram`` occurs (same bursts and tokens as ``top_ngrams``), in chronological order.

    Each occurrence is the tuple of its events, so a viewer can jump to ``occ[0].start_ms`` and read the payoff from
    ``occ[-1].outcome``. Overlapping occurrences are all returned (``JAB JAB JAB`` holds ``JAB JAB`` twice).
    """
    target = tuple(ngram)
    n = len(target)
    if n < 1:
        raise ValueError("ngram must not be empty")
    found = []
    for burst in segment(events, spec, unobserved):
        tokens = [spec.tokenizer(e) for e in burst]
        for i in range(len(burst) - n + 1):
            if tuple(tokens[i : i + n]) == target:
                found.append(tuple(burst[i : i + n]))
    return found
