# Database

Status: **implemented (M4)** — `src/boxing_ai/database/`. Design rationale: `docs/plan.md` §3.

## Running PostgreSQL without Docker

This machine has no Docker and no admin rights, so PostgreSQL 17 runs from the official portable zip
(EDB "binaries" download) unpacked to `%LOCALAPPDATA%\Programs\pgsql` (override with `BOXING_AI_PG_BIN`).

```powershell
powershell -ExecutionPolicy Bypass -File scripts\pg.ps1 init     # once: cluster in data\pg\, database boxing_ai
powershell -ExecutionPolicy Bypass -File scripts\pg.ps1 start    # after a reboot
powershell -ExecutionPolicy Bypass -File scripts\pg.ps1 stop
```

- Cluster: `data\pg\` (git-ignored), log `data\pg\server.log`. Listens on **localhost:5432 only**, user
  `postgres`, **trust** authentication — fine for a single-user dev machine, not for anything shared.
- On another machine, any PostgreSQL ≥ 13 works; point `BOXING_AI_DSN` at it.

## Using it

```python
from boxing_ai.annotations import load_annotation
from boxing_ai.database import connect, import_annotation, load_events, migrate

with connect() as conn:  # BOXING_AI_DSN or the local default
    migrate(conn)  # idempotent
    result = import_annotation(conn, load_annotation("data/annotations/sample-synthetic"))
    events = load_events(conn, [result.source_id], fighter="fighter-a")
```

- **Migrations:** `src/boxing_ai/database/migrations/NNNN_*.sql` (inside the package so they are found however it
  is installed), applied once each and recorded in `schema_migrations`. `migrate` then syncs `action_types` from
  `ontology.py`.
- **Import** (`import_annotation` / `import_source`) is one transaction: fighters, fight, participants, video and
  rounds are upserted; the source (unique per video + kind + name + version) is created, left alone when its
  content hash is unchanged, or **replaced atomically**. Conflicts with stored facts (different round spans,
  corners, session kind, a video under another fight) raise `ImportConflict`. Model runs use the same
  `import_source` with `annotator.kind = MODEL`.
- **Reads** return core types: `load_events` (chronological, with ids), `load_rounds`, `load_unobserved`,
  `list_sources`, plus the SQL aggregation `action_counts`. Choosing *which* source to analyse when a video has
  several is still open (`docs/plan.md` §12 Q4) — callers pass source ids explicitly.

## HTTP API (M6)

`src/boxing_ai/api.py` (FastAPI), started with `boxing-ai serve` — localhost only, **no authentication**.
Interactive docs at `/docs`; the OpenAPI schema at `/openapi.json` is the contract the React front-end's
TypeScript types are generated from.

| Endpoint | Returns |
|---|---|
| `GET /api/fighters` | slug, name, stance |
| `GET /api/sources?fight=&fighter=` | imported sources |
| `GET /api/fighters/{slug}/report?fight=&source=&tokens=&gap_ms=&min_count=&top=` | the CLI report as JSON (409 if the source choice is ambiguous) |
| `GET /api/fighters/{slug}/occurrences?ngram=JAB&ngram=CROSS&…` | every occurrence of a sequence: video, times, final outcome, events |
| `GET /api/sources/{id}/timeline` | events, rounds, unobserved intervals, video URL/fps/duration |
| `GET /api/videos/{slug}/file` | the video, with HTTP range support (seeking) |

Video files: `videos.path` (from `meta.toml` `video.file`) is resolved against `BOXING_AI_MEDIA_DIR` (default
`data/`, e.g. `raw/fight.mp4` → `data/raw/fight.mp4`); paths escaping that folder are refused (403).

## Schema changes vs the plan (§3)

- `fights.kind` (FIGHT/SPARRING/PADS/BAG/SHADOW); `action_types.category` includes `FEINT`.
- `events.commitment` (PROBE/FULL); attribute CHECKs are per category: side required for PUNCH and allowed for
  FEINT; target for PUNCH/FEINT; outcome and commitment PUNCH only; direction DEFENSE/MOVEMENT only.
- `events.confidence` is `DOUBLE PRECISION` (not `REAL`) so model scores round-trip exactly.
- `unobserved_intervals` key is `(source_id, start_ms, end_ms)`; `videos.path` is nullable (the file may not be on
  this machine).

## Verification

- `tests/integration/` (28 tests, real PostgreSQL): round trip, idempotent re-import, atomic replace (a crash
  mid-import leaves the old source intact), conflicts, every CHECK/FK rejecting bad raw SQL, SQL `LEAD()` bigram
  counts == Python n-gram counts, and HUMAN vs MODEL sources giving identical analytics.
- `scripts/db_perf.py` — 2,000,000 synthetic events (200 fighters, 1000 fights), measured 2026-09-25 on this
  machine:

| Query | Execution time | Plan |
|---|---|---|
| one fight timeline (2000 events) | 1.2 ms | index scan `events_source_time` |
| one fighter across 10 fights (10k events) | 9.9 ms | bitmap scan `events_fighter` |
| per-fighter action × outcome totals | 4.0 ms | bitmap scan `events_fighter` |
| `load_events` → 2000 `Event` objects (incl. Python) | 32 ms | |

Targets were "tens of ms" for a timeline and "≤ 1 s" for multi-fight aggregation; no extra indexes needed.
