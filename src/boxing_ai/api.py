"""HTTP API for the web front-end (install the ``api`` extra). Serialises results only; no statistics here.

    .venv\\Scripts\\boxing-ai serve            # or: uvicorn boxing_ai.api:app

Configuration: ``BOXING_AI_DSN`` (database) and ``BOXING_AI_MEDIA_DIR`` (root for ``videos.path``, default the
repository's ``data/``). Video files are streamed with HTTP range support so the browser can seek.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated, Any, Literal

import psycopg
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict

from boxing_ai.database import (
    connect,
    get_video,
    list_fighters,
    list_sources,
    load_events,
    load_rounds,
    load_unobserved,
)
from boxing_ai.events import Event, Round, UnobservedInterval
from boxing_ai.reporting import AmbiguousSource, fighter_occurrences, fighter_report
from boxing_ai.sequences import END, START, TOKENIZERS, StreamSpec

MEDIA_ENV = "BOXING_AI_MEDIA_DIR"
DEFAULT_MEDIA_DIR = Path(__file__).resolve().parents[2] / "data"

TokensName = Literal["action", "direction", "target", "commitment"]
assert set(TokensName.__args__) == set(
    TOKENIZERS
)  # keep the API enum in step with the core registry


# --- response models (the API contract; the front-end's TypeScript types are generated from these) -------


class _Out(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class FighterOut(_Out):
    slug: str
    name: str
    stance: str | None


class SourceOut(_Out):
    id: int
    fight: str
    video: str
    kind: str
    name: str
    version: str
    ontology_version: str
    events: int


class ActionCountOut(_Out):
    action_type: str
    count: int
    share: float


class OutcomeOut(_Out):
    action_type: str
    thrown: int
    landed: int
    blocked: int
    missed: int
    unknown: int
    landed_rate: float | None


class RoundStatOut(_Out):
    video: str
    round_number: int
    count: int
    observed_ms: int
    per_observed_minute: float | None


class PatternOut(_Out):
    ngram: tuple[str, ...]
    count: int
    known: int
    landed: int
    landed_rate: float | None


class NgramCountOut(_Out):
    ngram: tuple[str, ...]
    count: int


class TransitionOut(_Out):
    context: tuple[str, ...]
    next: str
    count: int
    total: int
    probability: float


class ReportOut(_Out):
    fighter: str
    tokens: TokensName
    gap_ms: int
    categories: list[str]
    start_token: str = START
    end_token: str = END
    sources: list[SourceOut]
    events: int
    punches: int
    observed_ms: int
    punches_per_observed_minute: float | None
    actions: list[ActionCountOut]
    outcomes: list[OutcomeOut]
    rounds: list[RoundStatOut]
    combos: dict[int, list[PatternOut]]
    entries: list[NgramCountOut]
    exits: list[NgramCountOut]
    transitions: list[TransitionOut]


class OccurrenceOut(BaseModel):
    video: str
    round_number: int | None
    start_ms: int
    end_ms: int
    final_outcome: str | None
    events: list[Event]


class TimelineOut(BaseModel):
    source: SourceOut
    video_url: str | None
    fps: float | None
    duration_ms: int | None
    rounds: list[Round]
    unobserved: list[UnobservedInterval]
    events: list[Event]


# --- app -------------------------------------------------------------------------------------------


def _db(request: Request) -> Iterator[psycopg.Connection]:
    """One connection per request (plenty for a local single-user app; add a pool when that changes)."""
    with connect(request.app.state.dsn) as conn:
        yield conn


Conn = Annotated[psycopg.Connection, Depends(_db)]


def create_app(*, dsn: str | None = None, media_dir: str | Path | None = None) -> FastAPI:
    media_root = Path(media_dir or os.environ.get(MEDIA_ENV) or DEFAULT_MEDIA_DIR).resolve()
    app = FastAPI(title="boxing-ai", version="0.1.0")
    app.state.dsn = dsn

    @app.exception_handler(AmbiguousSource)
    def _ambiguous(request: Request, exc: AmbiguousSource) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(LookupError)
    def _not_found(request: Request, exc: LookupError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=404)

    @app.exception_handler(ValueError)
    def _invalid(request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.get("/api/fighters", response_model=list[FighterOut])
    def fighters(conn: Conn) -> Any:
        return list_fighters(conn)

    @app.get("/api/sources", response_model=list[SourceOut])
    def sources(conn: Conn, fight: str | None = None, fighter: str | None = None) -> Any:
        return list_sources(conn, fight=fight, fighter=fighter)

    @app.get("/api/fighters/{fighter}/report", response_model=ReportOut)
    def report(
        conn: Conn,
        fighter: str,
        fight: str | None = None,
        source: str | None = None,
        tokens: TokensName = "action",
        gap_ms: Annotated[int, Query(ge=0)] = StreamSpec().gap_ms,
        min_count: Annotated[int, Query(ge=1)] = 1,
        top: Annotated[int, Query(ge=1, le=100)] = 10,
    ) -> Any:
        spec = StreamSpec(tokenizer=TOKENIZERS[tokens], gap_ms=gap_ms)
        r = fighter_report(
            conn, fighter, fight=fight, source=source, spec=spec, min_count=min_count, limit=top
        )
        fields = {name: getattr(r, name) for name in ReportOut.model_fields if hasattr(r, name)}
        return ReportOut.model_validate(
            fields
            | {
                "tokens": tokens,
                "gap_ms": spec.gap_ms,
                "categories": sorted(c.value for c in spec.categories),
            },
            from_attributes=True,
        )

    @app.get("/api/fighters/{fighter}/occurrences", response_model=list[OccurrenceOut])
    def occurrences(
        conn: Conn,
        fighter: str,
        ngram: Annotated[list[str], Query(min_length=1)],
        fight: str | None = None,
        source: str | None = None,
        tokens: TokensName = "action",
        gap_ms: Annotated[int, Query(ge=0)] = StreamSpec().gap_ms,
    ) -> Any:
        spec = StreamSpec(tokenizer=TOKENIZERS[tokens], gap_ms=gap_ms)
        found = fighter_occurrences(conn, fighter, ngram, fight=fight, source=source, spec=spec)
        return [
            OccurrenceOut(
                video=occ[0].video,
                round_number=occ[0].round_number,
                start_ms=occ[0].start_ms,
                end_ms=max(e.end_ms for e in occ),
                final_outcome=occ[-1].outcome,
                events=list(occ),
            )
            for occ in found
        ]

    @app.get("/api/sources/{source_id}/timeline", response_model=TimelineOut)
    def timeline(conn: Conn, source_id: int) -> Any:
        found = list_sources(conn, source_id=source_id)
        if not found:
            raise HTTPException(404, f"unknown source {source_id}")
        video = get_video(conn, found[0].video)
        return TimelineOut(
            source=SourceOut.model_validate(found[0]),
            video_url=f"/api/videos/{video.slug}/file" if video.path else None,
            fps=video.fps,
            duration_ms=video.duration_ms,
            rounds=load_rounds(conn, [source_id]),
            unobserved=load_unobserved(conn, [source_id]),
            events=load_events(conn, [source_id]),
        )

    @app.get("/api/videos/{slug}/file", response_class=FileResponse)
    def video_file(conn: Conn, slug: str) -> Any:
        video = get_video(conn, slug)
        if video is None or not video.path:
            raise HTTPException(404, f"no video file recorded for {slug!r}")
        path = (media_root / video.path).resolve()
        if not path.is_relative_to(media_root):
            raise HTTPException(403, "video path is outside the media directory")
        if not path.is_file():
            raise HTTPException(404, f"video file not found on this machine: {video.path}")
        return FileResponse(path)  # handles Range requests (seeking)

    return app


app = create_app()
