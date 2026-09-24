"""Collection-level validation: rules that need context beyond a single event.

``Event`` itself validates everything decidable from its own fields. Whether the fighter took part in the
fight, whether the time is inside the video, and whether the event sits inside a replay/cutaway needs the
surrounding facts, supplied here as an ``EventContext``. All problems are collected (not just the first) so an
annotator can fix a whole file in one pass.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from boxing_ai.events.coverage import Round, UnobservedInterval, spans_overlap
from boxing_ai.events.model import Event


@dataclass(frozen=True)
class EventContext:
    fight: str
    video: str
    fighters: frozenset[str]
    video_duration_ms: int | None = None
    unobserved: tuple[UnobservedInterval, ...] = ()
    rounds: tuple[Round, ...] = ()


@dataclass(frozen=True)
class Problem:
    index: int  # position of the offending event in the input sequence
    message: str


def check_events(events: Iterable[Event], ctx: EventContext) -> list[Problem]:
    """Return every context violation found; an empty list means the events are valid in this context."""
    round_numbers = {r.number for r in ctx.rounds if r.video == ctx.video}
    intervals = [iv for iv in ctx.unobserved if iv.video == ctx.video]
    problems: list[Problem] = []

    for i, e in enumerate(events):
        if e.fight != ctx.fight:
            problems.append(Problem(i, f"event fight {e.fight!r} != {ctx.fight!r}"))
        if e.video != ctx.video:
            problems.append(Problem(i, f"event video {e.video!r} != {ctx.video!r}"))
        if e.fighter not in ctx.fighters:
            known = ", ".join(sorted(ctx.fighters))
            problems.append(
                Problem(i, f"fighter {e.fighter!r} is not in this fight (known: {known})")
            )
        if ctx.video_duration_ms is not None and e.end_ms > ctx.video_duration_ms:
            problems.append(
                Problem(
                    i, f"end_ms {e.end_ms} is beyond the video duration {ctx.video_duration_ms}"
                )
            )
        if e.round_number is not None and e.round_number not in round_numbers:
            problems.append(
                Problem(i, f"round {e.round_number} is not defined for video {ctx.video!r}")
            )
        for iv in intervals:
            if spans_overlap(e.start_ms, e.end_ms, iv.start_ms, iv.end_ms):
                problems.append(
                    Problem(
                        i,
                        f"event [{e.start_ms}, {e.end_ms}] ms lies inside an unobserved "
                        f"{iv.kind} interval [{iv.start_ms}, {iv.end_ms})",
                    )
                )
    return problems
