# AGENTS.md — conventions for working in this repository

Boxing analytics: boxing **actions are stored as individual events**; combinations are *derived* at
analysis time. Human annotation and (later) ML predictions produce the **same** event representation.
The approved architecture/plan is in `docs/plan.md` — read it before making structural changes.

## Environment
- Python **3.11+** (the machine's default `python` is 3.9 — do not use it). Use the project venv:
  `.venv\Scripts\python.exe` (Windows). Create with `py -3.11 -m venv .venv` or the 3.11 interpreter path.
- Install: `.venv\Scripts\python.exe -m pip install -e ".[dev]"` (add `,db` / `,api` when needed).
- Run tests: `.venv\Scripts\python.exe -m pytest` — fast loop: `-m "not integration"`.
- Lint/format: `.venv\Scripts\python.exe -m ruff check . && .venv\Scripts\python.exe -m ruff format .`
- PostgreSQL (from M4): no Docker on this machine — portable binaries, managed by
  `powershell -ExecutionPolicy Bypass -File scripts\pg.ps1 init|start|stop|status` (see `docs/database.md`).
  App DSN: `BOXING_AI_DSN` (default `postgresql://postgres@localhost:5432/boxing_ai`). Integration tests read
  `BOXING_AI_TEST_DSN` (e.g. `postgresql://postgres@localhost:5432/postgres`; they create/drop
  `boxing_ai_pytest`) and are skipped when it is unset.

## Layering (enforced by `tests/unit/test_architecture.py`)
`ontology` → `events` → `sequences` → `analytics`; `evaluation` beside analytics; `annotations` feeds `events`.
- Core packages (`events`, `sequences`, `analytics`, `evaluation`, `annotations`, `ontology`) are **pure Python**:
  they must not import `psycopg`, `fastapi`, `cv2`, `torch`, or `boxing_ai.database`.
- `database` is the only place that contains SQL. It must not import `analytics`.
- UI code (`cli.py`, `api.py`, `web/`) serialises/renders results only — **no business logic in the UI**.
  Web stack: FastAPI + React/TypeScript (Vite); Streamlit was dropped (decision in `docs/plan.md` §10).
- Computer vision (`vision/`, Phase 3) imports core event types; core never imports `vision`.

## Engineering rules
- Prefer simple solutions; no premature abstraction; no new dependency without a stated reason.
- **Never make combinations (`JAB_JAB_CROSS`) a stored data type or ML class.**
- Times are **integer milliseconds** (`start_ms`, `end_ms`); seconds/`MM:SS.mmm` exist only at I/O boundaries.
- Ordering is total and deterministic: `(start_ms, end_ms, id)`. No unseeded randomness in analysis.
- The ontology is defined once in `src/boxing_ai/ontology.py`; anything else derives from it.
- Add tests for non-trivial logic; verify statistics against hand-calculated fixtures.
- Don't refactor unrelated code; don't silently change public APIs.
- Raw video and bulk pose data are never committed; annotation files are.
