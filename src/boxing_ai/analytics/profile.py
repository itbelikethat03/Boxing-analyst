"""Descriptive statistics: what a fighter throws, and how much per round.

Rates are per **observed** minute: round time minus replays/cutaways/unusable angles. Raw counts are confounded
by how much of a round the footage actually shows, so counts alone must not be compared across rounds or fights.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from boxing_ai.events import Event, Round, UnobservedInterval, observed_ms
from boxing_ai.ontology import Category

MS_PER_MINUTE = 60_000


@dataclass(frozen=True)
class ActionCount:
    action_type: str
    count: int
    share: float  # fraction of all counted events, 0..1


@dataclass(frozen=True)
class RoundStat:
    video: str
    round_number: int
    count: int
    observed_ms: int  # round duration minus unobserved time

    @property
    def per_observed_minute(self) -> float | None:
        """``None`` when nothing of the round was observed (the rate is undefined, not zero)."""
        return None if self.observed_ms == 0 else self.count / (self.observed_ms / MS_PER_MINUTE)


def action_distribution(
    events: Iterable[Event], *, categories: frozenset[Category] = frozenset({Category.PUNCH})
) -> list[ActionCount]:
    """How many of each action, most frequent first (ties by action code). Default: punches only."""
    counts = Counter(e.action_type for e in events if e.category in categories)
    total = sum(counts.values())
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [ActionCount(action, c, c / total) for action, c in ordered]


def round_stats(
    events: Iterable[Event],
    rounds: Iterable[Round],
    unobserved: Iterable[UnobservedInterval] = (),
    *,
    categories: frozenset[Category] = frozenset({Category.PUNCH}),
) -> list[RoundStat]:
    """Event counts per round with observed duration; one row per ``Round`` (video, then round number).

    Events whose ``round_number`` is unset are not attributed to any round.
    """
    counts = Counter(
        (e.video, e.round_number)
        for e in events
        if e.round_number is not None and e.category in categories
    )
    intervals = list(unobserved)
    stats = []
    for r in sorted(rounds, key=lambda r: (r.video, r.number)):
        same_video = [iv for iv in intervals if iv.video == r.video]
        stats.append(
            RoundStat(
                video=r.video,
                round_number=r.number,
                count=counts[(r.video, r.number)],
                observed_ms=observed_ms(r.start_ms, r.end_ms, same_video),
            )
        )
    return stats
