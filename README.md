# boxing-ai

Structured boxing intelligence built on **individual action events**.

- Individual actions (`JAB`, `CROSS`, `HOOK`, `UPPERCUT`, later `SLIP`, `ROLL`, …) are the fundamental data.
- Combinations/sequences are **derived** (n-grams, transition probabilities) — never stored or trained as classes.
- Events come from human annotation today and from computer vision later; downstream code can't tell the difference.

**Status:** M0–M5 — first usable MVP: annotate a clip, import it, get a fighter report. See `docs/plan.md` for the full plan and milestones.

## Quick start

```powershell
py -3.11 -m venv .venv          # or any Python 3.11+ (e.g. py -3.13)
.venv\Scripts\python.exe -m pip install -e ".[dev,db,api]"
.venv\Scripts\python.exe -m pytest
```

## Using it

```powershell
powershell -ExecutionPolicy Bypass -File scripts\pg.ps1 start   # local PostgreSQL (docs/database.md)
.venv\Scripts\boxing-ai init-db
.venv\Scripts\boxing-ai import data\annotations               # every annotation folder in there
.venv\Scripts\boxing-ai sources --fighter fighter-a
.venv\Scripts\boxing-ai report fighter-a                       # all of the fighter's fights
.venv\Scripts\boxing-ai report fighter-a --tokens commitment --gap-ms 1500 --min-count 2
.venv\Scripts\boxing-ai db-check
.venv\Scripts\boxing-ai serve                                  # HTTP API on http://127.0.0.1:8000 (docs at /docs)
```

The report shows punch totals and rate per observed minute, landed/blocked/missed per punch, per-round
counts, the most common 2- and 3-action sequences with how often their final punch landed, how bursts open
and close, and what follows each action. Annotation format: `docs/annotation.md`.

## Layout

See `docs/architecture.md`. Conventions for contributors and coding agents: `AGENTS.md`.
