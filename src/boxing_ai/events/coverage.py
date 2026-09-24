"""Coverage: which parts of a video are fight rounds, and which parts were *not* observed.

Broadcast footage contains replays, cutaways and unusable camera angles. Events must not exist inside those
stretches, they act as hard breaks between sequences, and they are subtracted when computing observed time
(punches per *observed* minute).

All spans are half-open ``[start_ms, end_ms)`` in milliseconds of *video* time.
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field, model_validator

from boxing_ai.ontology import UnobservedKind


class _Span(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    video: str = Field(min_length=1)
    start_ms: int = Field(ge=0)
    end_ms: int

    @model_validator(mode="after")
    def _end_after_start(self) -> _Span:
        if self.end_ms <= self.start_ms:
            raise ValueError(
                f"end_ms ({self.end_ms}) must be greater than start_ms ({self.start_ms})"
            )
        return self


class UnobservedInterval(_Span):
    kind: UnobservedKind


class Round(_Span):
    """A bell-to-bell span in *video* time (round boundaries differ per video)."""

    number: int = Field(ge=1)


def spans_overlap(start_ms: int, end_ms: int, other_start_ms: int, other_end_ms: int) -> bool:
    """Whether an event span ``[start_ms, end_ms]`` overlaps the half-open ``[other_start, other_end)``.

    A zero-length event is treated as lasting 1 ms, so a point event exactly at ``other_start_ms`` is inside
    and one exactly at ``other_end_ms`` is outside. Spans that merely touch do not overlap.
    """
    effective_end = max(end_ms, start_ms + 1)
    return start_ms < other_end_ms and effective_end > other_start_ms


def observed_ms(start_ms: int, end_ms: int, unobserved: Iterable[UnobservedInterval]) -> int:
    """Length of ``[start_ms, end_ms)`` not covered by any unobserved interval.

    ``unobserved`` must already be restricted to the video the span belongs to. Overlapping or duplicate
    intervals are merged, so nothing is subtracted twice.
    """
    if end_ms < start_ms:
        raise ValueError(f"end_ms ({end_ms}) is before start_ms ({start_ms})")
    clipped = sorted(
        (max(iv.start_ms, start_ms), min(iv.end_ms, end_ms))
        for iv in unobserved
        if iv.start_ms < end_ms and iv.end_ms > start_ms
    )
    covered = 0
    cur_start: int | None = None
    cur_end = 0
    for s, e in clipped:
        if cur_start is None or s > cur_end:
            if cur_start is not None:
                covered += cur_end - cur_start
            cur_start, cur_end = s, e
        else:
            cur_end = max(cur_end, e)
    if cur_start is not None:
        covered += cur_end - cur_start
    return (end_ms - start_ms) - covered
