# boxing-ai

Structured boxing intelligence built on **individual action events**.

- Individual actions (`JAB`, `CROSS`, `HOOK`, `UPPERCUT`, later `SLIP`, `ROLL`, …) are the fundamental data.
- Combinations/sequences are **derived** (n-grams, transition probabilities) — never stored or trained as classes.
- Events come from human annotation today and from computer vision later; downstream code can't tell the difference.

**Status:** M0–M2 (domain model + sequence analytics, no database yet). See `docs/plan.md` for the full plan and milestones.

## Quick start

```powershell
py -3.11 -m venv .venv          # or point at any Python 3.11+
.venv\Scripts\python.exe -m pip install -e ".[dev]"
.venv\Scripts\python.exe -m pytest
```

## Layout

See `docs/architecture.md`. Conventions for contributors and coding agents: `AGENTS.md`.
