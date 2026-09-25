"""Load one annotation folder (``meta.toml`` + ``events.csv``) into validated Events.

Pipeline: parse -> normalise (corner -> fighter slug, JAB/CROSS side fill, timecodes -> ms, round derived from
the round spans) -> validate (ontology, bounds, fighter in fight, not inside an unobserved interval). **Every**
problem is collected with its file and line, so an annotator fixes a whole file in one pass; nothing is returned
unless the folder is entirely valid.
"""

from __future__ import annotations

import csv
import hashlib
import tomllib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, TypeVar

from pydantic import ValidationError

from boxing_ai.annotations.meta import AnnotationMeta
from boxing_ai.annotations.timecodes import parse_timecode
from boxing_ai.events import (
    Event,
    EventContext,
    Round,
    UnobservedInterval,
    check_events,
    chronological,
)
from boxing_ai.ontology import (
    Commitment,
    Corner,
    Direction,
    Outcome,
    Side,
    Target,
    action_spec,
    resolve_side,
)

META_FILE = "meta.toml"
EVENTS_FILE = "events.csv"
REQUIRED_COLUMNS = ("start", "end", "fighter", "action")
COLUMNS = (
    *REQUIRED_COLUMNS,
    "side",
    "target",
    "direction",
    "outcome",
    "commitment",
    "round",
    "notes",
)
# Written in the outcome column when an outcome is required but cannot be seen; stored as None.
UNKNOWN_OUTCOME = "UNKNOWN"

E = TypeVar("E", bound=StrEnum)

# Event field -> CSV column, so validation messages name what the annotator actually typed.
_FIELD_TO_COLUMN = {
    "start_ms": "start",
    "end_ms": "end",
    "action_type": "action",
    "round_number": "round",
}


@dataclass(frozen=True)
class AnnotationProblem:
    file: str
    line: int | None  # None for problems not tied to one line (e.g. anything in meta.toml)
    message: str

    def __str__(self) -> str:
        where = f"{self.file}:{self.line}" if self.line is not None else self.file
        return f"{where}: {self.message}"


class AnnotationError(ValueError):
    def __init__(self, folder: Path, problems: list[AnnotationProblem]) -> None:
        self.problems = problems
        lines = "\n".join(f"  {p}" for p in problems)
        super().__init__(f"{len(problems)} problem(s) in {folder}:\n{lines}")


@dataclass(frozen=True)
class Annotation:
    """A fully validated annotation folder. ``events`` are in chronological order."""

    folder: Path
    meta: AnnotationMeta
    events: tuple[Event, ...]
    content_sha256: str  # of meta.toml + events.csv as read; makes re-imports idempotent

    @property
    def rounds(self) -> tuple[Round, ...]:
        return self.meta.to_rounds()

    @property
    def unobserved(self) -> tuple[UnobservedInterval, ...]:
        return self.meta.to_unobserved()

    @property
    def context(self) -> EventContext:
        return context_of(self.meta)


def context_of(meta: AnnotationMeta) -> EventContext:
    return EventContext(
        fight=meta.fight.slug,
        video=meta.video.slug,
        fighters=frozenset(f.slug for f in meta.fighters),
        video_duration_ms=meta.video.duration,
        unobserved=meta.to_unobserved(),
        rounds=meta.to_rounds(),
    )


def load_annotation(folder: str | Path) -> Annotation:
    """Load and validate ``folder``; raise ``AnnotationError`` listing every problem found."""
    folder = Path(folder)
    problems: list[AnnotationProblem] = []

    meta = _load_meta(folder / META_FILE, problems)
    if meta is None:  # events cannot be checked without fighters/rounds/intervals
        raise AnnotationError(folder, problems)

    events = _load_events(folder / EVENTS_FILE, meta, problems)
    if problems:
        raise AnnotationError(folder, problems)
    digest = hashlib.sha256()
    for name in (META_FILE, EVENTS_FILE):
        digest.update((folder / name).read_bytes())
    return Annotation(
        folder=folder,
        meta=meta,
        events=tuple(chronological(events)),
        content_sha256=digest.hexdigest(),
    )


def _load_meta(path: Path, problems: list[AnnotationProblem]) -> AnnotationMeta | None:
    if not path.is_file():
        problems.append(AnnotationProblem(path.name, None, "file not found"))
        return None
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8-sig"))
    except tomllib.TOMLDecodeError as exc:
        problems.append(AnnotationProblem(path.name, None, f"invalid TOML: {exc}"))
        return None
    try:
        return AnnotationMeta.model_validate(raw)
    except ValidationError as exc:
        problems += [AnnotationProblem(path.name, None, m) for m in _messages(exc)]
        return None


def _load_events(
    path: Path, meta: AnnotationMeta, problems: list[AnnotationProblem]
) -> list[Event]:
    if not path.is_file():
        problems.append(AnnotationProblem(path.name, None, "file not found"))
        return []
    text = path.read_text(encoding="utf-8-sig")  # tolerate the BOM Excel writes
    header = text.split("\n", 1)[0]
    # Excel in comma-decimal locales saves "CSV" with semicolons.
    delimiter = ";" if ";" in header and "," not in header else ","
    reader = csv.DictReader(text.splitlines(), delimiter=delimiter, restkey="\0extra")

    columns = [c.strip().lower() for c in reader.fieldnames or []]
    reader.fieldnames = columns
    unknown = [c for c in columns if c not in COLUMNS]
    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    if unknown or missing:
        if unknown:
            problems.append(
                AnnotationProblem(path.name, 1, f"unknown column(s) {unknown}; allowed: {COLUMNS}")
            )
        if missing:
            problems.append(AnnotationProblem(path.name, 1, f"missing column(s) {missing}"))
        return []

    rows = _RowParser(meta)
    events: list[Event] = []
    lines: list[int] = []
    for row in reader:
        line = reader.line_num
        if "\0extra" in row:
            problems.append(AnnotationProblem(path.name, line, "more cells than header columns"))
            continue
        cells = {k: (v or "").strip() for k, v in row.items()}
        if not any(cells.values()):
            continue
        event, errors = rows.parse(cells)
        problems += [AnnotationProblem(path.name, line, m) for m in errors]
        if event is not None:
            events.append(event)
            lines.append(line)

    for p in check_events(events, context_of(meta)):
        problems.append(AnnotationProblem(path.name, lines[p.index], p.message))
    return events


class _RowParser:
    def __init__(self, meta: AnnotationMeta) -> None:
        self.meta = meta
        self.slug_by_corner = {c: f.slug for c, f in meta.fighter_by_corner().items()}
        self.rounds = meta.to_rounds()

    def parse(self, cells: dict[str, str]) -> tuple[Event | None, list[str]]:
        errors: list[str] = []
        start = _cell(cells, "start", parse_timecode, errors)
        end = _cell(cells, "end", parse_timecode, errors)
        corner = _cell(cells, "fighter", lambda v: _enum(Corner, v), errors)
        side = _cell(cells, "side", lambda v: _enum(Side, v), errors)
        target = _cell(cells, "target", lambda v: _enum(Target, v), errors)
        direction = _cell(cells, "direction", lambda v: _enum(Direction, v), errors)
        outcome_text = cells.get("outcome", "")
        outcome = None
        if outcome_text.upper() != UNKNOWN_OUTCOME:
            outcome = _cell(cells, "outcome", lambda v: _enum(Outcome, v), errors)
        commitment = _cell(cells, "commitment", lambda v: _enum(Commitment, v), errors)
        given_round = _cell(cells, "round", _positive_int, errors)
        action = cells.get("action", "").upper()

        if not action:
            errors.append("action: missing")
        else:
            try:
                side = resolve_side(action, side)
            except ValueError as exc:  # unknown action, or e.g. JAB with side=REAR
                errors.append(str(exc))
            else:
                if self.meta.outcome_required and action_spec(action).takes_outcome:
                    if not outcome_text:
                        errors.append(
                            f"outcome: required for every {action} in a {self.meta.fight.kind} "
                            f"(write {UNKNOWN_OUTCOME} if it cannot be seen)"
                        )
        for col in REQUIRED_COLUMNS[:3]:
            if not cells.get(col):
                errors.append(f"{col}: missing")
        fighter = None
        if corner is not None:
            fighter = self.slug_by_corner.get(corner)
            if fighter is None:
                errors.append(f"fighter: no fighter in corner {corner} in {META_FILE}")
        if errors:
            return None, errors

        round_number = self._round_of(start)
        if self.rounds and round_number is None:
            errors.append(f"start {start} ms is outside every round defined in {META_FILE}")
        elif given_round is not None and given_round != round_number:
            where = f"round {round_number}" if round_number else "no round"
            errors.append(f"round: column says {given_round} but the start time lies in {where}")
        if errors:
            return None, errors

        notes = cells.get("notes", "")
        try:
            event = Event(
                fight=self.meta.fight.slug,
                video=self.meta.video.slug,
                fighter=fighter,
                start_ms=start,
                end_ms=end,
                action_type=action,
                side=side,
                target=target,
                direction=direction,
                outcome=outcome,
                commitment=commitment,
                round_number=round_number,
                metadata={"notes": notes} if notes else {},
            )
        except ValidationError as exc:
            return None, _messages(exc)
        return event, []

    def _round_of(self, start_ms: int) -> int | None:
        for r in self.rounds:
            if r.start_ms <= start_ms < r.end_ms:
                return r.number
        return None


def _cell(cells: dict[str, str], column: str, convert: Any, errors: list[str]) -> Any:
    value = cells.get(column, "")
    if not value:
        return None
    try:
        return convert(value)
    except ValueError as exc:
        errors.append(f"{column}: {exc}")
        return None


def _enum(enum: type[E], value: str) -> E:
    try:
        return enum(value.upper())
    except ValueError:
        allowed = "/".join(enum)
        raise ValueError(f"{value!r} is not one of {allowed}") from None


def _positive_int(value: str) -> int:
    if not value.isdigit() or int(value) < 1:
        raise ValueError(f"{value!r} is not a positive whole number")
    return int(value)


def _messages(exc: ValidationError) -> list[str]:
    """Flatten pydantic errors into ``location: message`` lines (list positions shown 1-based as ``#n``)."""
    out = []
    for err in exc.errors():
        msg = err["msg"].removeprefix("Value error, ")
        parts = [
            f"#{p + 1}" if isinstance(p, int) else _FIELD_TO_COLUMN.get(p, p) for p in err["loc"]
        ]
        out.append(f"{'.'.join(parts)}: {msg}" if parts else msg)
    return out
