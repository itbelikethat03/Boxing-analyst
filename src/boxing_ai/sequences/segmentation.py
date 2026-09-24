"""Segmenting an event stream into *bursts* (combinations / exchanges).

Naïve n-grams over a whole fight would invent ``JAB -> CROSS`` pairs across a 30-second pause, a camera cut or a
replay. Sequences are therefore only formed **within a burst**. A burst ends (a new one starts) when any of these
holds between consecutive events:

* the silence since the burst's last activity exceeds ``gap_ms``;
* the video, or the round, changes;
* an unobserved interval (replay, cutaway, unusable angle) lies between them.

Bursts are *derived* views — never stored. ``gap_ms`` is a tunable analysis parameter: the defaults below are
starting guesses to be tuned against the real inter-punch gap distribution.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from boxing_ai.events import Event, UnobservedInterval, chronological
from boxing_ai.ontology import Category
from boxing_ai.sequences.tokens import Tokenizer, by_action

DEFAULT_COMBO_GAP_MS = 1000  # one fighter's combination
DEFAULT_EXCHANGE_GAP_MS = 2500  # an exchange between both fighters
DEFAULT_CATEGORIES = frozenset({Category.PUNCH, Category.DEFENSE})  # footwork excluded by default


@dataclass(frozen=True)
class StreamSpec:
    """How to turn events into token sequences.

    ``categories`` are filtered *before* segmentation, so gaps are measured between the events that remain
    (e.g. with STEP excluded, ``JAB, STEP, CROSS`` reads as the adjacent pair ``JAB -> CROSS``).
    """

    categories: frozenset[Category] = DEFAULT_CATEGORIES
    tokenizer: Tokenizer = by_action
    gap_ms: int = DEFAULT_COMBO_GAP_MS

    def __post_init__(self) -> None:
        if self.gap_ms < 0:
            raise ValueError(f"gap_ms must be >= 0, got {self.gap_ms}")
        if not self.categories:
            raise ValueError("categories must not be empty")


DEFAULT_SPEC = StreamSpec()


def segment(
    events: Iterable[Event],
    spec: StreamSpec = DEFAULT_SPEC,
    unobserved: Iterable[UnobservedInterval] = (),
) -> list[list[Event]]:
    """Split events into chronologically ordered bursts (see module docstring)."""
    intervals: dict[str, list[UnobservedInterval]] = defaultdict(list)
    for iv in unobserved:
        intervals[iv.video].append(iv)

    stream = chronological(e for e in events if e.category in spec.categories)
    bursts: list[list[Event]] = []
    current: list[Event] = []
    burst_end = 0  # latest end_ms seen in the current burst (events may overlap)

    for e in stream:
        if current and _starts_new_burst(current[-1], e, burst_end, spec, intervals[e.video]):
            bursts.append(current)
            current = []
        if not current:
            burst_end = e.end_ms
        else:
            burst_end = max(burst_end, e.end_ms)
        current.append(e)

    if current:
        bursts.append(current)
    return bursts


def _starts_new_burst(
    prev: Event,
    e: Event,
    burst_end: int,
    spec: StreamSpec,
    intervals: Sequence[UnobservedInterval],
) -> bool:
    if e.video != prev.video or e.round_number != prev.round_number:
        return True
    if e.start_ms - burst_end > spec.gap_ms:
        return True
    return any(iv.start_ms < e.start_ms and iv.end_ms > burst_end for iv in intervals)


def token_segments(
    events: Iterable[Event],
    spec: StreamSpec = DEFAULT_SPEC,
    unobserved: Iterable[UnobservedInterval] = (),
) -> list[list[str]]:
    """``segment`` followed by tokenization: the input every sequence statistic is computed from."""
    return [[spec.tokenizer(e) for e in burst] for burst in segment(events, spec, unobserved)]
