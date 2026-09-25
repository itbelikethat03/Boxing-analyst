# Architecture

Full rationale: `docs/plan.md` §1. This file is the short, always-current reference.

## Layers

```
ontology ─► events ─► sequences ─► analytics
               ▲            └────► evaluation (M8)
annotations ───┘
database (M4, the only SQL)  ─► reporting (M5) ─► cli / api (M6, FastAPI) ─► web/ (React)   (render only)
vision (Phase 3) ─► produces list[Event]   (core never imports vision)
```

## Rules (enforced by `tests/unit/test_architecture.py`)
- Core packages are pure Python and never import `psycopg`, `fastapi`, `cv2`, `torch`, or `boxing_ai.database`.
- `database` never imports `analytics`.
- UI never contains business logic.

## Key decisions
1. Events are immutable facts scoped to a *source run* (human annotation or model run).
2. Analytics consume `Sequence[Event]`, not a DB handle — deterministic and unit-testable.
3. The ontology is defined once in code (`ontology.py`); the DB `action_types` table is synced from it.
4. Tokenization is a projection `Event -> str` chosen per query (`SLIP` vs `SLIP_LEFT` is an analysis choice).
5. Sequences are computed within **bursts** — split by a gap threshold and by hard breaks (unobserved
   intervals, round boundaries) — so n-grams never span pauses, cuts or replays.
6. Times are integer milliseconds; ordering is `(start_ms, end_ms, id)`.

## Status
| Milestone | State |
|---|---|
| M0 scaffold | done |
| M1 ontology + event model | done — see `docs/ontology.md` |
| M2 sequences + analytics | done — see `docs/sequence-analysis.md` |
| M3 annotation format/loader | done — see `docs/annotation.md` (time convention Q1 still to confirm) |
| M4 PostgreSQL | done — see `docs/database.md` (portable PostgreSQL 17, no Docker) |
| M5 reporting + CLI | done — `boxing-ai init-db / import / sources / report / db-check` |
| M6 API + web viewer | API done (`boxing_ai.api`, `boxing-ai serve`); React viewer (`web/`) not started |
| M7+ | not started |
