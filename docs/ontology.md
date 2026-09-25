# Ontology (version 0.2)

0.2 added the `FEINT` category/action and the punch attribute `commitment` (probes).

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
| `FEINT` | FEINT | `LEAD`/`REAR` optional (blank = shoulder/head/foot feint) | — | nothing is thrown; `target` = what it threatened; no outcome |
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
- `commitment` (`PROBE`/`FULL`): punches only; optional. `PROBE` = pawing, measuring, range-finding punch. A probe
  is still a thrown punch and counts as one; tokenizer `by_action_commitment` renders it as `JAB_PROBE`.
- `confidence`: `null` for human annotations (never a fake `1.0`).

**Feint vs probe.** A feint *fakes* an attack (no strike is thrown) and is its own action; a probe is a *real*
but uncommitted punch, so it is a punch with an attribute. Feints are in the default sequence stream (`FEINT →
CROSS` is a pattern) but never in punch counts.

Fighter-level enums (used by annotation `meta.toml` and, from M4, the database): `Corner` (`RED`/`BLUE`),
`Stance` (`ORTHODOX`/`SOUTHPAW`/`SWITCH`) and `SessionKind` (`FIGHT`/`SPARRING` with two annotated boxers;
`PADS`/`BAG`/`SHADOW` with one).

## Unobserved intervals (broadcast footage)
`REPLAY`, `CUTAWAY`, `UNSUPPORTED_ANGLE`, `UNCERTAIN`. Events may not lie inside these; they are hard breaks
between sequences and are subtracted when computing observed time.
