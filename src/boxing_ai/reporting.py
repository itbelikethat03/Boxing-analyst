"""Thin composition: repository -> analytics -> ``Report``. No statistics are computed here, and no rendering.

``build_report`` is pure (events in, Report out); ``fighter_report`` adds source selection and database reads.

Default source rule (docs/plan.md §12 Q4): per video, use the single HUMAN source; if there is none, the single
MODEL source. Anything more ambiguous raises ``AmbiguousSource`` listing the candidates, and the caller picks one
with ``source="name"`` or ``"name@version"`` — a report never silently mixes or chooses between annotations.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import psycopg

from boxing_ai.analytics import (
    ActionCount,
    NgramCount,
    OutcomeStat,
    PatternOutcome,
    RoundStat,
    action_distribution,
    entry_patterns,
    exit_patterns,
    outcome_breakdown,
    pattern_occurrences,
    pattern_outcomes,
    round_stats,
    transition_probabilities,
)
from boxing_ai.database import (
    SourceInfo,
    fighter_exists,
    list_sources,
    load_events,
    load_rounds,
    load_unobserved,
)
from boxing_ai.events import Event, Round, UnobservedInterval
from boxing_ai.sequences import DEFAULT_SPEC, StreamSpec, Transition


class AmbiguousSource(LookupError):
    """A video has several candidate sources and none was chosen explicitly."""


@dataclass(frozen=True)
class Report:
    fighter: str
    sources: tuple[SourceInfo, ...]
    spec: StreamSpec
    events: int  # the fighter's events of all categories
    punches: int
    observed_ms: int  # summed over the rounds of the selected videos
    actions: list[ActionCount]
    outcomes: list[OutcomeStat]
    rounds: list[RoundStat]
    combos: dict[int, list[PatternOutcome]]  # n -> most common n-grams with landing rates
    entries: list[NgramCount]
    exits: list[NgramCount]
    transitions: list[Transition]

    @property
    def punches_per_observed_minute(self) -> float | None:
        return None if self.observed_ms == 0 else self.punches / (self.observed_ms / 60_000)


def build_report(
    fighter: str,
    events: Iterable[Event],
    rounds: Iterable[Round],
    unobserved: Iterable[UnobservedInterval],
    *,
    sources: Sequence[SourceInfo] = (),
    spec: StreamSpec = DEFAULT_SPEC,
    ngram_sizes: Sequence[int] = (2, 3),
    min_count: int = 1,
    limit: int | None = 10,
) -> Report:
    """Everything the report shows, for ``fighter``'s own events (other fighters' events are ignored)."""
    own = [e for e in events if e.fighter == fighter]
    rounds, unobserved = list(rounds), list(unobserved)
    per_round = round_stats(own, rounds, unobserved)
    actions = action_distribution(own)
    kw = {"min_count": min_count, "limit": limit, "unobserved": unobserved}
    return Report(
        fighter=fighter,
        sources=tuple(sources),
        spec=spec,
        events=len(own),
        punches=sum(a.count for a in actions),
        observed_ms=sum(r.observed_ms for r in per_round),
        actions=actions,
        outcomes=outcome_breakdown(own),
        rounds=per_round,
        combos={n: pattern_outcomes(own, n, spec, **kw) for n in ngram_sizes},
        entries=entry_patterns(own, 1, spec, **kw),
        exits=exit_patterns(own, 1, spec, **kw),
        transitions=transition_probabilities(own, spec, min_count=min_count, unobserved=unobserved),
    )


def select_sources(candidates: Iterable[SourceInfo], source: str | None = None) -> list[SourceInfo]:
    """Apply the default source rule (module docstring) per video; ``source`` narrows by name[@version]."""
    by_video: dict[str, list[SourceInfo]] = defaultdict(list)
    for s in candidates:
        if source is None or source in (s.name, f"{s.name}@{s.version}"):
            by_video[s.video].append(s)
    chosen = []
    for video, options in sorted(by_video.items()):
        human = [s for s in options if s.kind == "HUMAN"]
        pool = human or options
        if len(pool) != 1:
            names = ", ".join(f"{s.name}@{s.version} ({s.kind})" for s in pool)
            raise AmbiguousSource(
                f"video {video!r} has several sources: {names}; pick one with --source"
            )
        chosen.append(pool[0])
    return chosen


@dataclass(frozen=True)
class FighterData:
    """Everything analytics need for one fighter, from the selected sources."""

    sources: list[SourceInfo]
    events: list[Event]
    rounds: list[Round]
    unobserved: list[UnobservedInterval]


def load_fighter_data(
    conn: psycopg.Connection, fighter: str, *, fight: str | None = None, source: str | None = None
) -> FighterData:
    if not fighter_exists(conn, fighter):
        raise LookupError(f"unknown fighter {fighter!r}")
    sources = select_sources(list_sources(conn, fight=fight, fighter=fighter), source)
    if not sources:
        raise LookupError(f"no sources found for {fighter!r}" + (f" in {fight!r}" if fight else ""))
    ids = [s.id for s in sources]
    return FighterData(
        sources,
        load_events(conn, ids, fighter=fighter),
        load_rounds(conn, ids),
        load_unobserved(conn, ids),
    )


def fighter_report(
    conn: psycopg.Connection,
    fighter: str,
    *,
    fight: str | None = None,
    source: str | None = None,
    **options,
) -> Report:
    """Report for ``fighter`` across all their fights (or one ``fight``), from the selected sources."""
    data = load_fighter_data(conn, fighter, fight=fight, source=source)
    return build_report(
        fighter, data.events, data.rounds, data.unobserved, sources=data.sources, **options
    )


def fighter_occurrences(
    conn: psycopg.Connection,
    fighter: str,
    ngram: Sequence[str],
    *,
    fight: str | None = None,
    source: str | None = None,
    spec: StreamSpec = DEFAULT_SPEC,
) -> list[tuple[Event, ...]]:
    """Where ``fighter`` performed ``ngram`` — the same selection and bursts as the report."""
    data = load_fighter_data(conn, fighter, fight=fight, source=source)
    return pattern_occurrences(data.events, ngram, spec, unobserved=data.unobserved)
