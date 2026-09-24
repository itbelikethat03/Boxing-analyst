"""Deterministic chronological ordering of events."""

from __future__ import annotations

from collections.abc import Iterable

from boxing_ai.events.model import Event


def sort_key(event: Event) -> tuple:
    """Total order: ``(video, start_ms, end_ms, id, fighter, action_type)``.

    * ``video`` leads because time is only meaningful *within* a video; this keeps videos from interleaving
      while still giving multi-video lists a deterministic total order.
    * Persisted events (with an ``id``) sort before un-persisted ones at a tie, then by ``id``.
    * ``fighter`` and ``action_type`` are the last resort so that the result never depends on input order,
      even for un-persisted events with identical times.
    """
    return (
        event.video,
        event.start_ms,
        event.end_ms,
        event.id is None,
        event.id or 0,
        event.fighter,
        event.action_type,
    )


def chronological(events: Iterable[Event]) -> list[Event]:
    """Return the events in deterministic chronological order (input is not modified)."""
    return sorted(events, key=sort_key)
