# Ontology (version 0.1)

Source of truth: `src/boxing_ai/ontology.py`. This page describes it; if they disagree, the code wins.

## Principles
- Actions are **individual** actions. Combinations (`JAB_JAB_CROSS`) are never actions (a test enforces that no
  action code contains `_`).
- Variants are **attributes**: a slip to the left is `SLIP` + `direction=LEFT`. Whether an n-gram token is `SLIP`
  or `SLIP_LEFT` is chosen per analysis by a tokenizer, so it never requires re-annotation or retraining.
- Adding an action = adding one `ActionSpec` row and bumping `ONTOLOGY_VERSION`. No schema change.

## Actions

| Action | Category | Side | Direction | Notes |
|---|---|---|---|---|
| `JAB` | PUNCH | always `LEAD` | — | lead-hand straight |
| `CROSS` | PUNCH | always `REAR` | — | rear-hand straight |
| `HOOK` | PUNCH | `LEAD`/`REAR` (required) | — | |
| `UPPERCUT` | PUNCH | `LEAD`/`REAR` (required) | — | |
| `SLIP` | DEFENSE | — | `LEFT`/`RIGHT` (required) | |
| `ROLL` | DEFENSE | — | `LEFT`/`RIGHT` (required) | |
| `PULL` | DEFENSE | — | none | |
| `BLOCK` | DEFENSE | — | none | |
| `PARRY` | DEFENSE | — | none | |
| `STEP` | MOVEMENT | — | any of `LEFT`/`RIGHT`/`FORWARD`/`BACK` (required) | excluded from streams by default |

Defensive actions and `STEP` are registered now so the model is proven generic; annotating them is optional in
the MVP.

## Attributes
- `side` (`LEAD`/`REAR`): punches only, always stored. Relative to the fighter's stance. `JAB` ⇒ `LEAD`,
  `CROSS` ⇒ `REAR` is validated; the annotation loader may fill it in (`ontology.resolve_side`).
- `target` (`HEAD`/`BODY`): punches only; **optional** — `null` means unknown.
- `outcome` (`LANDED`/`BLOCKED`/`MISSED`): punches only; optional.
- `direction`: non-punches only, in the *fighter's own* frame of reference.
- `confidence`: `null` for human annotations (never a fake `1.0`).

## Unobserved intervals (broadcast footage)
`REPLAY`, `CUTAWAY`, `UNSUPPORTED_ANGLE`, `UNCERTAIN`. Events may not lie inside these; they are hard breaks
between sequences and are subtracted when computing observed time.
