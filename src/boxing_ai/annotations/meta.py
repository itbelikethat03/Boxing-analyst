"""``meta.toml``: the small header of an annotation folder (who annotated which video of which fight).

Times are timecode *strings* (``"03:12.400"``) and become integer milliseconds here. Unknown keys are rejected
so a typo (``[[round]]`` for ``[[rounds]]``) is an error instead of silently missing data.
"""

from __future__ import annotations

import datetime as dt
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from boxing_ai.annotations.timecodes import parse_timecode
from boxing_ai.events import Round, UnobservedInterval
from boxing_ai.ontology import Corner, SessionKind, Stance, UnobservedKind

SCHEMA_VERSION = 1


def _timecode(value: Any) -> int:
    if not isinstance(value, str):
        raise ValueError(f'write times as quoted strings such as "03:12.400", got {value!r}')
    return parse_timecode(value)


Timecode = Annotated[int, BeforeValidator(_timecode)]


class SourceKind(StrEnum):
    HUMAN = "HUMAN"
    MODEL = "MODEL"


class _Entry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)


class _SpanEntry(_Entry):
    start: Timecode
    end: Timecode

    @model_validator(mode="after")
    def _end_after_start(self) -> _SpanEntry:
        if self.end <= self.start:
            raise ValueError(f"end ({self.end} ms) must be after start ({self.start} ms)")
        return self


class AnnotatorEntry(_Entry):
    name: str = Field(min_length=1)
    kind: SourceKind = SourceKind.HUMAN
    version: str = ""  # annotation revision, e.g. "r1"


class VideoEntry(_Entry):
    slug: str = Field(min_length=1)
    file: str | None = None  # path relative to data/, informational only
    fps: float | None = Field(default=None, gt=0)
    duration: Timecode | None = None
    sha256: str | None = None


class FightEntry(_Entry):
    """A bout, or a training session (the DB still calls both "fights")."""

    slug: str = Field(min_length=1)
    kind: SessionKind = SessionKind.FIGHT
    date: dt.date | None = None


class FighterEntry(_Entry):
    corner: Corner
    slug: str = Field(min_length=1)
    name: str = Field(min_length=1)
    stance: Stance | None = None


class RoundEntry(_SpanEntry):
    number: int = Field(ge=1)


class UnobservedEntry(_SpanEntry):
    kind: UnobservedKind


class AnnotationMeta(_Entry):
    schema_version: Literal[1]
    ontology_version: str = Field(min_length=1)
    annotator: AnnotatorEntry
    video: VideoEntry
    fight: FightEntry
    fighters: list[FighterEntry] = Field(min_length=1, max_length=2)
    rounds: list[RoundEntry] = Field(default_factory=list)
    unobserved: list[UnobservedEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_consistency(self) -> AnnotationMeta:
        problems: list[str] = []
        needed = self.fight.kind.participants
        if len(self.fighters) != needed:
            problems.append(
                f"a {self.fight.kind} session has {needed} fighter(s), got {len(self.fighters)}"
            )
        for attr in ("corner", "slug"):
            values = [getattr(f, attr) for f in self.fighters]
            if len(set(values)) != len(values):
                problems.append(f"fighters must have different {attr}s, got {values}")

        numbers = [r.number for r in self.rounds]
        if len(set(numbers)) != len(numbers):
            problems.append(f"round numbers must be unique, got {numbers}")
        ordered = sorted(self.rounds, key=lambda r: r.start)
        for a, b in zip(ordered, ordered[1:], strict=False):
            if b.start < a.end:
                problems.append(f"rounds {a.number} and {b.number} overlap")

        duration = self.video.duration
        if duration is not None:
            for label, spans in (("round", self.rounds), ("unobserved interval", self.unobserved)):
                for s in spans:
                    if s.end > duration:
                        problems.append(
                            f"a {label} ends ({s.end} ms) after the video ({duration} ms)"
                        )

        if problems:
            raise ValueError("; ".join(problems))
        return self

    @property
    def outcome_required(self) -> bool:
        """Human annotators must record every punch's outcome when there is an opponent to land on."""
        return self.annotator.kind is SourceKind.HUMAN and self.fight.kind.participants == 2

    def fighter_by_corner(self) -> dict[Corner, FighterEntry]:
        return {f.corner: f for f in self.fighters}

    def to_rounds(self) -> tuple[Round, ...]:
        return tuple(
            Round(video=self.video.slug, number=r.number, start_ms=r.start, end_ms=r.end)
            for r in sorted(self.rounds, key=lambda r: r.start)
        )

    def to_unobserved(self) -> tuple[UnobservedInterval, ...]:
        return tuple(
            UnobservedInterval(video=self.video.slug, kind=u.kind, start_ms=u.start, end_ms=u.end)
            for u in sorted(self.unobserved, key=lambda u: (u.start, u.end))
        )
