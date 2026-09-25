"""Import event sources and read them back as ``Event`` objects. The only module (with ``migrate``) that holds SQL.

Import is one transaction per source: fighters/fight/video/rounds are upserted, then the source is created, left
alone (same content hash) or **replaced atomically** (old events and intervals cascade away). Human annotations and
model runs go through the same ``import_source``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import psycopg
from psycopg.types.json import Jsonb

from boxing_ai.annotations import Annotation, AnnotationMeta, context_of
from boxing_ai.annotations.meta import FighterEntry
from boxing_ai.events import Event, Round, UnobservedInterval, check_events
from boxing_ai.ontology import ACTIONS, action_spec


class ImportConflict(ValueError):
    """The source contradicts what is already stored (e.g. different round spans for the same video)."""


@dataclass(frozen=True)
class ImportResult:
    source_id: int
    status: Literal["created", "replaced", "unchanged"]
    events: int


@dataclass(frozen=True)
class SourceInfo:
    id: int
    fight: str
    video: str
    kind: str
    name: str
    version: str
    ontology_version: str
    events: int


@dataclass(frozen=True)
class FighterInfo:
    slug: str
    name: str
    stance: str | None


@dataclass(frozen=True)
class VideoInfo:
    slug: str
    fight: str
    path: str | None  # relative to the media directory
    fps: float | None
    duration_ms: int | None


# --- import ----------------------------------------------------------------------------------------


def import_annotation(conn: psycopg.Connection, annotation: Annotation) -> ImportResult:
    return import_source(
        conn,
        annotation.meta,
        annotation.events,
        content_sha256=annotation.content_sha256,
    )


def import_source(
    conn: psycopg.Connection,
    meta: AnnotationMeta,
    events: Iterable[Event],
    *,
    content_sha256: str | None = None,
    params: dict[str, Any] | None = None,
) -> ImportResult:
    """Store one source (annotation pass or model run) for ``meta.video``; see the module docstring."""
    events = list(events)
    problems = check_events(events, context_of(meta))
    if problems:
        detail = "; ".join(f"#{p.index}: {p.message}" for p in problems[:5])
        raise ValueError(f"{len(problems)} invalid event(s): {detail}")

    with conn.transaction():
        fight_id = _upsert_fight(conn, meta)
        fighter_ids = {f.slug: _upsert_participant(conn, fight_id, f, meta) for f in meta.fighters}
        video_id = _upsert_video(conn, meta, fight_id)
        _ensure_rounds(conn, video_id, meta)

        key = (video_id, meta.annotator.kind.value, meta.annotator.name, meta.annotator.version)
        existing = conn.execute(
            "SELECT id, content_sha256 FROM event_sources"
            " WHERE video_id = %s AND kind = %s AND name = %s AND version = %s",
            key,
        ).fetchone()
        if existing and content_sha256 is not None and existing[1] == content_sha256:
            return ImportResult(existing[0], "unchanged", len(events))
        if existing:
            conn.execute("DELETE FROM event_sources WHERE id = %s", (existing[0],))

        (source_id,) = conn.execute(
            "INSERT INTO event_sources"
            " (video_id, kind, name, version, ontology_version, content_sha256, params)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id",
            (*key, meta.ontology_version, content_sha256, Jsonb(params or {})),
        ).fetchone()
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO unobserved_intervals (source_id, start_ms, end_ms, kind)"
                " VALUES (%s, %s, %s, %s)",
                [(source_id, u.start_ms, u.end_ms, u.kind.value) for u in meta.to_unobserved()],
            )
            # video_id / fight_id are derived from the source, so a caller cannot store inconsistent ones.
            cur.executemany(
                "INSERT INTO events (source_id, video_id, fight_id, fighter_id, round_number,"
                " start_ms, end_ms, action_type, category, side, target, direction, outcome,"
                " commitment, confidence, metadata)"
                " SELECT s.id, s.video_id, v.fight_id, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,"
                " %s, %s, %s"
                " FROM event_sources s JOIN videos v ON v.id = s.video_id WHERE s.id = %s",
                [_event_row(e, fighter_ids[e.fighter]) + (source_id,) for e in events],
            )
    return ImportResult(source_id, "replaced" if existing else "created", len(events))


def _event_row(e: Event, fighter_id: int) -> tuple:
    def text(value: Any) -> str | None:
        return None if value is None else str(value)

    return (
        fighter_id,
        e.round_number,
        e.start_ms,
        e.end_ms,
        e.action_type,
        action_spec(e.action_type).category.value,
        text(e.side),
        text(e.target),
        text(e.direction),
        text(e.outcome),
        text(e.commitment),
        e.confidence,
        Jsonb(e.metadata),
    )


def _upsert_fight(conn: psycopg.Connection, meta: AnnotationMeta) -> int:
    fight_id, kind = conn.execute(
        "INSERT INTO fights (slug, kind, fought_on) VALUES (%s, %s, %s)"
        " ON CONFLICT (slug) DO UPDATE SET fought_on = COALESCE(fights.fought_on, EXCLUDED.fought_on)"
        " RETURNING id, kind",
        (meta.fight.slug, meta.fight.kind.value, meta.fight.date),
    ).fetchone()
    if kind != meta.fight.kind:
        raise ImportConflict(
            f"fight {meta.fight.slug!r} is stored as {kind}, not {meta.fight.kind}"
        )
    return fight_id


def _upsert_participant(
    conn: psycopg.Connection, fight_id: int, f: FighterEntry, meta: AnnotationMeta
) -> int:
    stance = f.stance.value if f.stance else None
    (fighter_id,) = conn.execute(
        "INSERT INTO fighters (slug, name, stance) VALUES (%s, %s, %s)"
        " ON CONFLICT (slug) DO UPDATE SET stance = COALESCE(fighters.stance, EXCLUDED.stance)"
        " RETURNING id",
        (f.slug, f.name, stance),
    ).fetchone()
    stored = conn.execute(
        "SELECT corner FROM fight_participants WHERE fight_id = %s AND fighter_id = %s",
        (fight_id, fighter_id),
    ).fetchone()
    if stored is not None:
        if stored[0] != f.corner:
            raise ImportConflict(
                f"{f.slug!r} is stored in corner {stored[0]} of {meta.fight.slug!r}, not {f.corner}"
            )
        return fighter_id
    other = conn.execute(
        "SELECT fr.slug FROM fight_participants p JOIN fighters fr ON fr.id = p.fighter_id"
        " WHERE p.fight_id = %s AND p.corner = %s",
        (fight_id, f.corner.value),
    ).fetchone()
    if other is not None:
        raise ImportConflict(
            f"corner {f.corner} of {meta.fight.slug!r} is already {other[0]!r}, not {f.slug!r}"
        )
    conn.execute(
        "INSERT INTO fight_participants (fight_id, fighter_id, corner, stance)"
        " VALUES (%s, %s, %s, %s)",
        (fight_id, fighter_id, f.corner.value, stance),
    )
    return fighter_id


def _upsert_video(conn: psycopg.Connection, meta: AnnotationMeta, fight_id: int) -> int:
    v = meta.video
    video_id, stored_fight = conn.execute(
        "INSERT INTO videos (fight_id, slug, path, sha256, fps, duration_ms)"
        " VALUES (%s, %s, %s, %s, %s, %s)"
        " ON CONFLICT (slug) DO UPDATE SET"
        "  path = COALESCE(EXCLUDED.path, videos.path),"
        "  sha256 = COALESCE(EXCLUDED.sha256, videos.sha256),"
        "  fps = COALESCE(EXCLUDED.fps, videos.fps),"
        "  duration_ms = COALESCE(EXCLUDED.duration_ms, videos.duration_ms)"
        " RETURNING id, fight_id",
        (fight_id, v.slug, v.file, v.sha256, v.fps, v.duration),
    ).fetchone()
    if stored_fight != fight_id:
        raise ImportConflict(f"video {v.slug!r} belongs to another fight")
    return video_id


def _ensure_rounds(conn: psycopg.Connection, video_id: int, meta: AnnotationMeta) -> None:
    """Rounds belong to the video: the first source defines them, later sources must agree."""
    wanted = {(r.number, r.start_ms, r.end_ms) for r in meta.to_rounds()}
    stored = set(
        conn.execute(
            "SELECT round_number, start_ms, end_ms FROM rounds WHERE video_id = %s", (video_id,)
        ).fetchall()
    )
    if not stored:
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO rounds (video_id, round_number, start_ms, end_ms) VALUES (%s, %s, %s, %s)",
                [(video_id, *r) for r in sorted(wanted)],
            )
    elif wanted and wanted != stored:
        raise ImportConflict(
            f"round spans for video {meta.video.slug!r} differ from the stored ones: "
            f"stored {sorted(stored)}, new {sorted(wanted)}"
        )


# --- reads -----------------------------------------------------------------------------------------


def list_sources(
    conn: psycopg.Connection,
    *,
    fight: str | None = None,
    video: str | None = None,
    fighter: str | None = None,
    source_id: int | None = None,
) -> list[SourceInfo]:
    """Sources, optionally only those of one fight / video / fights the fighter took part in / one id."""
    rows = conn.execute(
        "SELECT s.id, f.slug, v.slug, s.kind, s.name, s.version, s.ontology_version,"
        "  (SELECT count(*) FROM events e WHERE e.source_id = s.id)"
        " FROM event_sources s JOIN videos v ON v.id = s.video_id JOIN fights f ON f.id = v.fight_id"
        " WHERE (%(fight)s::text IS NULL OR f.slug = %(fight)s)"
        "   AND (%(video)s::text IS NULL OR v.slug = %(video)s)"
        "   AND (%(id)s::bigint IS NULL OR s.id = %(id)s)"
        "   AND (%(fighter)s::text IS NULL OR EXISTS ("
        "     SELECT 1 FROM fight_participants p JOIN fighters fr ON fr.id = p.fighter_id"
        "     WHERE p.fight_id = f.id AND fr.slug = %(fighter)s))"
        " ORDER BY f.slug, v.slug, s.kind, s.name, s.version",
        {"fight": fight, "video": video, "fighter": fighter, "id": source_id},
    ).fetchall()
    return [SourceInfo(*row) for row in rows]


def load_events(
    conn: psycopg.Connection, source_ids: Sequence[int], *, fighter: str | None = None
) -> list[Event]:
    """Events of the given sources (optionally one fighter) in chronological order, with database ids."""
    rows = conn.execute(
        "SELECT e.id, f.slug, v.slug, fr.slug, e.start_ms, e.end_ms, e.action_type, e.side,"
        "  e.target, e.direction, e.outcome, e.commitment, e.confidence, e.round_number, e.metadata"
        " FROM events e"
        " JOIN fights f ON f.id = e.fight_id"
        " JOIN videos v ON v.id = e.video_id"
        " JOIN fighters fr ON fr.id = e.fighter_id"
        " WHERE e.source_id = ANY(%(sources)s)"
        "   AND (%(fighter)s::text IS NULL OR fr.slug = %(fighter)s)"
        " ORDER BY v.slug, e.start_ms, e.end_ms, e.id",
        {"sources": list(source_ids), "fighter": fighter},
    ).fetchall()
    fields = (
        "id fight video fighter start_ms end_ms action_type side target direction outcome"
        " commitment confidence round_number metadata"
    ).split()
    return [Event(**dict(zip(fields, row, strict=True))) for row in rows]


def load_rounds(conn: psycopg.Connection, source_ids: Sequence[int]) -> list[Round]:
    rows = conn.execute(
        "SELECT DISTINCT v.slug, r.round_number, r.start_ms, r.end_ms"
        " FROM rounds r JOIN videos v ON v.id = r.video_id"
        " JOIN event_sources s ON s.video_id = r.video_id"
        " WHERE s.id = ANY(%s) ORDER BY v.slug, r.round_number",
        (list(source_ids),),
    ).fetchall()
    return [Round(video=v, number=n, start_ms=s, end_ms=e) for v, n, s, e in rows]


def load_unobserved(
    conn: psycopg.Connection, source_ids: Sequence[int]
) -> list[UnobservedInterval]:
    rows = conn.execute(
        "SELECT v.slug, u.kind, u.start_ms, u.end_ms"
        " FROM unobserved_intervals u"
        " JOIN event_sources s ON s.id = u.source_id JOIN videos v ON v.id = s.video_id"
        " WHERE u.source_id = ANY(%s) ORDER BY v.slug, u.start_ms, u.end_ms",
        (list(source_ids),),
    ).fetchall()
    return [UnobservedInterval(video=v, kind=k, start_ms=s, end_ms=e) for v, k, s, e in rows]


def action_counts(
    conn: psycopg.Connection, fighter: str, source_ids: Sequence[int]
) -> dict[str, int]:
    """Per-fighter action totals aggregated in SQL (the multi-fight query the fighter index serves)."""
    rows = conn.execute(
        "SELECT e.action_type, count(*) FROM events e"
        " WHERE e.fighter_id = (SELECT id FROM fighters WHERE slug = %s)"
        "   AND e.source_id = ANY(%s)"
        " GROUP BY e.action_type ORDER BY count(*) DESC, e.action_type",
        (fighter, list(source_ids)),
    ).fetchall()
    return dict(rows)


def fighter_exists(conn: psycopg.Connection, slug: str) -> bool:
    return conn.execute("SELECT 1 FROM fighters WHERE slug = %s", (slug,)).fetchone() is not None


def list_fighters(conn: psycopg.Connection) -> list[FighterInfo]:
    rows = conn.execute("SELECT slug, name, stance FROM fighters ORDER BY slug").fetchall()
    return [FighterInfo(*row) for row in rows]


def get_video(conn: psycopg.Connection, slug: str) -> VideoInfo | None:
    row = conn.execute(
        "SELECT v.slug, f.slug, v.path, v.fps, v.duration_ms"
        " FROM videos v JOIN fights f ON f.id = v.fight_id WHERE v.slug = %s",
        (slug,),
    ).fetchone()
    return None if row is None else VideoInfo(*row)


# Consistency checks the schema cannot express as constraints. Each query returns offending event ids.
_AUDITS = {
    "video/fight differ from the event's source": (
        "SELECT e.id FROM events e JOIN event_sources s ON s.id = e.source_id"
        " JOIN videos v ON v.id = s.video_id"
        " WHERE e.video_id <> s.video_id OR e.fight_id <> v.fight_id"
    ),
    "round_number does not match the round containing start_ms": (
        "SELECT e.id FROM events e LEFT JOIN rounds r ON r.video_id = e.video_id"
        "  AND e.start_ms >= r.start_ms AND e.start_ms < r.end_ms"
        " WHERE e.round_number IS DISTINCT FROM r.round_number"
        "   AND (e.round_number IS NOT NULL"
        "        OR EXISTS (SELECT 1 FROM rounds r2 WHERE r2.video_id = e.video_id))"
    ),
    "event lies inside an unobserved interval of its source": (
        "SELECT e.id FROM events e JOIN unobserved_intervals u ON u.source_id = e.source_id"
        " WHERE e.start_ms < u.end_ms AND GREATEST(e.end_ms, e.start_ms + 1) > u.start_ms"
    ),
}


def audit(conn: psycopg.Connection) -> list[str]:
    """Human-readable problems found in stored data (empty = consistent). Used by ``boxing-ai db-check``."""
    problems = []
    stored = dict(conn.execute("SELECT code, category FROM action_types").fetchall())
    expected = {spec.code: spec.category.value for spec in ACTIONS.values()}
    if stored != expected:
        problems.append("action_types differ from ontology.py (run init-db to sync)")
    for label, sql in _AUDITS.items():
        ids = [row[0] for row in conn.execute(sql + " ORDER BY 1 LIMIT 11").fetchall()]
        if ids:
            shown = ", ".join(map(str, ids[:10])) + (", ..." if len(ids) > 10 else "")
            problems.append(f"{label}: event id(s) {shown}")
    return problems
