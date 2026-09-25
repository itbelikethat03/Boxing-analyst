"""Write Events back to an ``events.csv`` that ``load_annotation`` reads to identical Events.

Used for round-trip checks and, later, for model pre-annotation (a model proposes events, a human corrects the
CSV). ``side`` is left blank where the action implies it and ``round`` is left blank because the loader derives
it from the round spans, so the file stays as terse as a hand-written one.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from pathlib import Path

from boxing_ai.annotations.loader import COLUMNS, UNKNOWN_OUTCOME
from boxing_ai.annotations.meta import AnnotationMeta
from boxing_ai.annotations.timecodes import format_timecode
from boxing_ai.events import Event, chronological
from boxing_ai.ontology import action_spec


def write_events_csv(path: str | Path, events: Iterable[Event], meta: AnnotationMeta) -> None:
    """Write ``events`` (all of ``meta``'s fight/video) in chronological order.

    Raises ``ValueError`` rather than silently dropping data the CSV cannot hold (``confidence``, metadata other
    than ``notes``) or events that do not belong to ``meta``.
    """
    corner_of = {f.slug: f.corner for f in meta.fighters}
    rows = []
    for e in chronological(events):
        if (e.fight, e.video) != (meta.fight.slug, meta.video.slug):
            raise ValueError(f"event of {e.fight}/{e.video} does not belong to {meta.video.slug}")
        if e.fighter not in corner_of:
            raise ValueError(f"fighter {e.fighter!r} is not in {meta.fight.slug}")
        extra = set(e.metadata) - {"notes"}
        if e.confidence is not None or extra:
            raise ValueError(
                f"events.csv cannot hold confidence/metadata {sorted(extra)}; event at {e.start_ms} ms"
            )
        spec = action_spec(e.action_type)
        implied = spec.implied_side is not None
        unknown = UNKNOWN_OUTCOME if spec.takes_outcome else ""
        rows.append(
            {
                "start": format_timecode(e.start_ms),
                "end": format_timecode(e.end_ms),
                "fighter": corner_of[e.fighter],
                "action": e.action_type,
                "side": "" if implied or e.side is None else e.side,
                "target": e.target or "",
                "direction": e.direction or "",
                "outcome": e.outcome or unknown,
                "commitment": e.commitment or "",
                "round": "",
                "notes": e.metadata.get("notes", ""),
            }
        )
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
