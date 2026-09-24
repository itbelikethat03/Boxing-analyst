"""In-memory event selection (fighter / fight / video / round / time range).

The database layer filters in SQL for real data; this is the same contract for plain lists, used by tests and
small experiments. ``None`` means "no constraint". The time range is half-open on the event *start*:
``start_ms <= event.start_ms < end_ms``.
"""

from __future__ import annotations

from collections.abc import Iterable

from boxing_ai.events import Event, chronological


def select_events(
    events: Iterable[Event],
    *,
    fighter: str | None = None,
    fight: str | None = None,
    video: str | None = None,
    round_number: int | None = None,
    start_ms: int | None = None,
    end_ms: int | None = None,
) -> list[Event]:
    """Return the matching events in chronological order."""

    def keep(e: Event) -> bool:
        return (
            (fighter is None or e.fighter == fighter)
            and (fight is None or e.fight == fight)
            and (video is None or e.video == video)
            and (round_number is None or e.round_number == round_number)
            and (start_ms is None or e.start_ms >= start_ms)
            and (end_ms is None or e.start_ms < end_ms)
        )

    return chronological(e for e in events if keep(e))
