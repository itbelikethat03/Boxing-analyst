# Annotation format and protocol

Status: **implemented (M3)** — `src/boxing_ai/annotations/`. Worked example: `data/annotations/sample-synthetic/`.
Rationale: `docs/plan.md` §5.

## Folder layout

One folder per video: `data/annotations/<video-slug>/meta.toml` + `events.csv`. Annotation folders are the
ground-truth dataset and are committed to git; the video file itself is not.

```python
from boxing_ai.annotations import load_annotation

# Raises AnnotationError listing every problem.
a = load_annotation("data/annotations/sample-synthetic")
a.events, a.rounds, a.unobserved, a.context
```

## `meta.toml`

```toml
schema_version = 1
ontology_version = "0.2"

[annotator]
name = "brin"
kind = "HUMAN"          # HUMAN (default) | MODEL
version = "r1"          # annotation revision, optional

[video]
slug = "fightX-full"
file = "raw/fightX-full.mp4"   # optional, informational
fps = 29.97                    # optional
duration = "38:12.000"         # optional; enables the "event beyond end of video" check
# sha256 = "…"                 # optional; the M4 import computes it from the file

[fight]
slug = "fighter-a-vs-fighter-b-2024-05-04"
kind = "FIGHT"                 # FIGHT (default) | SPARRING | PADS | BAG | SHADOW
date = 2024-05-04              # optional

[[fighters]]                   # 2 for FIGHT/SPARRING, 1 for PADS/BAG/SHADOW; corners and slugs differ
corner = "RED"
slug = "fighter-a"
name = "Fighter A"
stance = "ORTHODOX"            # ORTHODOX | SOUTHPAW | SWITCH, optional

[[rounds]]                     # optional; bell-to-bell spans in VIDEO time, must not overlap
number = 1
start = "03:12.400"
end = "06:12.500"

[[unobserved]]                 # optional; REPLAY | CUTAWAY | UNSUPPORTED_ANGLE | UNCERTAIN
start = "04:01.000"
end = "04:09.500"
kind = "REPLAY"
```

- Times are **quoted** timecode strings. Unknown keys are errors (a typo must not silently drop data).
- Spans are half-open `[start, end)`.

## `events.csv`

```
start,end,fighter,action,side,target,direction,outcome,commitment,round,notes
03:25.900,03:26.050,RED,FEINT,REAR,BODY,,,,,
03:26.320,03:26.500,RED,JAB,,HEAD,,BLOCKED,PROBE,,
03:26.710,03:26.910,RED,CROSS,,HEAD,,MISSED,,,
03:27.420,03:27.700,BLUE,SLIP,,,LEFT,,,,"slipped the cross, countered"
```

| Column | Required | Values |
|---|---|---|
| `start`, `end` | yes | `SS.mmm`, `MM:SS.mmm` or `HH:MM:SS.mmm` (≤ 3 decimals); `end ≥ start` |
| `fighter` | yes | corner: `RED` / `BLUE` (must exist in `meta.toml`) |
| `action` | yes | ontology code (`docs/ontology.md`) |
| `side` | punches | `LEAD` / `REAR`; blank for `JAB` (LEAD) / `CROSS` (REAR); optional on `FEINT` |
| `target` | no | `HEAD` / `BODY`, punches and feints; blank = unknown |
| `direction` | per action | `LEFT`/`RIGHT`/`FORWARD`/`BACK`, fighter's own frame; non-punches only |
| `outcome` | punches in FIGHT/SPARRING | `LANDED` / `BLOCKED` / `MISSED`, or `UNKNOWN` if it cannot be seen |
| `commitment` | no | `PROBE` / `FULL`, punches only; blank = not judged |
| `round` | no | derived from the round spans; if filled in it must agree |
| `notes` | no | free text, kept in `Event.metadata["notes"]` |

- Column order is free; only `start,end,fighter,action` must be present. Values are case-insensitive.
- Blank lines are skipped. UTF-8 with or without BOM; comma or semicolon delimiter (Excel in Slovenian
  locale saves with `;`).
- When `[[rounds]]` are defined, every event must **start** inside a round.
- **Outcome is mandatory** for every punch by a HUMAN annotator in a FIGHT or SPARRING session: blocked and
  missed punches are what make "successful pattern" analysis possible. A blank cell is an error, so a forgotten
  outcome can't pass as "unknown"; write `UNKNOWN` when the impact is genuinely hidden (stored as no outcome).
  Solo sessions (PADS/BAG/SHADOW) and MODEL sources don't require it.
- No event may overlap an `[[unobserved]]` interval or end after `video.duration`.

**Errors.** The loader reports *every* problem at once as `file:line: message`, e.g.
`events.csv:12: JAB is always LEAD, got side=REAR`. Nothing is returned unless the folder is fully valid.

**Writing.** `write_events_csv(path, events, meta)` writes events back in the same format (implied sides and
`round` left blank); loading the result reproduces identical events. It refuses events with `confidence` or
non-`notes` metadata instead of silently dropping them.

## Protocol rules

1. **Exhaustive-or-excluded:** inside any observed stretch *every* punch is annotated. If you cannot be
   exhaustive (flurry, clinch, blur), mark that stretch `UNCERTAIN`. Missing real punches corrupt both the
   statistics and later ML training/evaluation.
2. **Mark `REPLAY` / `CUTAWAY` / `UNSUPPORTED_ANGLE`** (tight shots, corner/crowd cameras). A replay that is not
   marked double-counts punches.
3. Annotate **thrown** punches (a visible attempt) with their `outcome` (`LANDED`/`BLOCKED`/`MISSED`).
   - **Probe** (`commitment=PROBE`): a real but uncommitted punch — pawing, measuring, range-finding. Mark it only
     when clearly so; blank means a normal punch.
   - **Feint** (`FEINT`): a faked attack that throws nothing — a hand twitch, shoulder dip, level change or step
     meant to draw a reaction. `side` = the hand that faked (blank for shoulder/head/foot), `target` = what it
     threatened. Annotate a feint only when it is deliberate and visible; when unsure, leave it out.
   - Guard adjustments, arm swings, clinch pushing and referee separations are neither punches nor feints.
4. Identify fighters by corner; `LEAD`/`REAR` is relative to that fighter's stance *at that moment* (switch
   hitters change lead hand mid-fight).
5. Record `ontology_version`; the protocol and ontology are versioned.
6. Double-annotate ~5–10 % of clips (another annotator, or yourself weeks later) to measure agreement.

**Time convention (proposed, not yet confirmed — `docs/plan.md` §12 Q1):** `start` = first frame the hand /
defensive motion visibly begins; `end` = full extension or return to guard (punch) or motion completion.
Boundaries are fuzzy by ±3–5 frames; evaluation will match on the event midpoint with a tolerance.

## Still open

- Q1: confirm the time convention above after annotating a first real 30-second clip.

Resolved: Q2 — outcomes (incl. blocked/missed) are annotated from day one; Q3 — pad/bag/shadow sessions are in
scope as `kind = PADS/BAG/SHADOW` for comparison with fight footage. Feints and probes were added in ontology 0.2.
