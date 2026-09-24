"""Test helpers for building valid events tersely."""

from __future__ import annotations

from typing import Any

from boxing_ai.events import Event
from boxing_ai.ontology import Category, Direction, Side, category_of, resolve_side

_DEFAULT_DIRECTION = {"SLIP": Direction.LEFT, "ROLL": Direction.LEFT, "STEP": Direction.FORWARD}


def ev(action: str = "JAB", start_ms: int = 0, **overrides: Any) -> Event:
    """Build a valid event; anything passed in ``overrides`` (even ``None``) wins over the defaults.

    Defaults: 180 ms long, ``fighter-a`` in ``fight-1`` / ``video-1``. Punches get their implied side (or
    LEAD for hooks/uppercuts); SLIP/ROLL/STEP get a direction.
    """
    fields: dict[str, Any] = {
        "fight": "fight-1",
        "video": "video-1",
        "fighter": "fighter-a",
        "start_ms": start_ms,
        "end_ms": start_ms + 180,
        "action_type": action,
    }
    if category_of(action) is Category.PUNCH:
        fields["side"] = resolve_side(action, None) or Side.LEAD
    elif action in _DEFAULT_DIRECTION:
        fields["direction"] = _DEFAULT_DIRECTION[action]
    fields.update(overrides)
    return Event(**fields)
