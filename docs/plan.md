# Boxing AI — Architecture & Implementation Plan

## Context

**Goal.** Build a boxing analytics system in which *individual boxing actions are the fundamental stored unit*; combinations are derived at analysis time; and human annotation and (later) ML predictions produce the *same* event representation, so everything downstream is source-agnostic.

**Repository state (Step 1 — Explore).** `D:\Boxing-analyst` is **empty** and **not a git repo**. Nothing to reuse, no conventions to preserve, no API to break. Greenfield.

**Environment findings that change the plan**

| Finding | Consequence |
|---|---|
| Default `python` is **3.9** (Anaconda). **3.11** exists at `E:\Sola\Prog\Python` | Project pins `requires-python >=3.11` (stdlib `tomllib`, `StrEnum`, modern typing); use a `.venv` from 3.11 |
| `psql` absent; **Docker Desktop installed but engine not running** | You chose Docker → `docker-compose.yml` (Postgres only). Docker Desktop must be started before M4 |
| `ffmpeg` absent | Not needed for MVP. Required at Phase 3 start (CFR normalisation, frame extraction) |
| **GTX 970, 4 GB (Maxwell, sm_52)** | Fine for pose inference + small temporal models. Not for fine-tuning video transformers. Recent PyTorch wheels are progressively dropping old GPU architectures → **verify `torch.cuda.get_arch_list()` before committing to a Phase-3 stack**; CPU fallback / occasional rented GPU are contingency |

**Your answers.** Footage = **broadcast pro fights** (hardest CV case). Postgres = **Docker Desktop + compose** (later: portable PostgreSQL, no Docker on this PC — `docs/database.md`). Annotation = **CSV first, web annotator later**.

**What broadcast footage adds that your spec doesn't cover.** Broadcast video has camera cuts, replays/slow-mo (which would *double-count* punches), cutaways, tight shots that hide the opponent, and many viewpoints. So the annotation format and schema must express **"which parts of the video were observed and exhaustively annotated"** (`unobserved_intervals`, §3/§5). Without it: replays inflate counts, punches across a camera cut become fake adjacent pairs, punches/min is wrong, and unannotated real punches poison future ML training as false "negatives". This is the largest addition to your proposal and it's much cheaper to build in now than to retrofit into annotations.

---

## 0. Challenges to your assumptions (verdicts)

| Your assumption | Verdict | Why (short) |
|---|---|---|
| **PostgreSQL** | ✅ Keep — but not because of scale | Volume is tiny (~1–2.5k events/fight → ~2M rows for 1,000 fights). SQLite would cope. Postgres wins on expressiveness (composite FKs, CHECKs, window functions for cross-checking analytics, JSONB), concurrent writers (annotator + dashboard + ML worker), and growth path. Cost = a running server (Docker). Rejected: TimescaleDB (time is video-relative, volume tiny), MongoDB (relations/aggregations are the core), DuckDB (great later for offline research on exported Parquet, not system of record). **Bulk signals (video, per-frame pose) live on disk as Parquet/npz, not in Postgres — DB holds only the semantic layer (events).** |
| **Event = `action` + `punch_type`/`defense_type`** | ⚠️ Simplify | One flat `action_type` (JAB, CROSS, … SLIP, STEP) + derived `category`. Nested discriminators give sparse columns and make n-gram tokens awkward. |
| **Ontology lists `SLIP_LEFT`, `STEP_BACK` … as actions** | ⚠️ Contradicts your own principle | Direction is an *attribute* (`SLIP` + `direction=LEFT`). Whether an n-gram token is `SLIP` or `SLIP_LEFT` becomes an **analysis-time tokenizer choice**, not a data decision. Same trick for `JAB/HEAD`, etc. |
| **`side` for every punch** | ⚠️ Partly redundant | JAB ≡ LEAD, CROSS ≡ REAR by definition. Store `side` always for punches, *validate consistency* (JAB+REAR is rejected), let the annotation loader fill it for JAB/CROSS. LEAD/REAR is stance-relative → `stance` stored on fighter/participant. |
| **`target` HEAD/BODY** | ⚠️ Make optional | Likely the hardest attribute for ML; `NULL` = unknown must be valid so weak predictions don't invalidate events. |
| **Hierarchy Fighter→Fight→Round→Video→Event** | ⚠️ Adjust | A Video belongs to a **Fight** (many videos/angles possible). A **Round** is a *time span within a Video* (bell times are video-relative). Events reference video + fighter; round is a time-derived label. |
| **`timestamp` (float seconds) + `duration`** | ⚠️ Change representation | Integer **milliseconds** `start_ms`/`end_ms` (exact, deterministic, indexable; no `Decimal(14.32)` float bugs). Seconds/`MM:SS.mmm` only at the I/O boundary. |
| **Missing: provenance** | ➕ Add `event_sources` | Human ground truth and model predictions for the *same video* must coexist (that's how you evaluate). Provenance lives on the *run*, not in the event's analytical fields, so downstream code stays source-agnostic. |
| **Missing: thrown vs landed** | ➕ Add optional `outcome` | LANDED/BLOCKED/MISSED is the standard boxing metric; annotating it later means re-watching everything. Nullable, costs nothing now. |
| **Missing: what counts as a "combination"/"exchange"** | ➕ Define | Naïve n-grams across a 30 s pause invent `JAB→CROSS`. Sequences are computed within **bursts** split by a gap threshold and by hard breaks (unobserved intervals, round boundaries). Derived, never stored. |
| **Pose estimation** | ✅ Baseline, not gospel | Compact, data-efficient, small temporal model fits a 4 GB GPU. Risks: wrists blurred/occluded by gloves & opponent, boxers only ~150–300 px tall in wide shots, no glove/impact cues, 3–6 frames per punch at 25–30 fps. Decide by *measured* ablation; keep the classifier interface swappable. |
| **YOLO** | ✅ but as **YOLO-pose + tracker**, not a separate detector | A separate person-detector adds little (two boxers + referee is easy); the hard parts are tracking, identity, clinches. Note Ultralytics is **AGPL-3.0** (fine privately; matters if you distribute). Alternatives if wrists are bad: RTMPose/ViTPose. |
| **Streamlit** | ❌ Dropped (2026-09-25) | Rerun-the-script model can't keep a video player in sync with an event timeline, no frame stepping or hotkeys; custom JS components would be needed anyway. See **Decision: web stack** below. |
| **FastAPI** | ✅ From M6 | Thin HTTP layer over `reporting`/`database` (like `cli.py`) serving the React front-end; pydantic models already exist. |
| **Proposed repo tree** | ⚠️ Trim | No empty `detection/ tracking/ classification/` packages until Phase 3. Add `annotations/`, `evaluation/`, `ontology`. |

---

## 1. Architecture

```
                     ┌──────────── core (pure Python; deps: pydantic only) ────────────┐
 annotation files ─► │ annotations ─► events (ontology, Event, validation, ordering)   │
 (CSV + TOML)        │                    │                                            │
                     │                    ▼                                            │
                     │ sequences (tokenize, segment, n-grams, transitions)             │
                     │                    │                                            │
                     │                    ▼                                            │
                     │ analytics (profile, followers, entry/exit, per-round)           │
                     │ evaluation (match, hierarchical metrics)  ◄── used by ML later  │
                     └────────────────────┬────────────────────────────────────────────┘
                                          │ list[Event] / dataclasses
        database (psycopg3, SQL migrations, repository) ── only place that knows SQL
                                          │
                     reporting (thin composition: repo → analytics → Report dataclass)
                          ┌───────────────┴──────────────┐
                    cli (argparse)                 api (FastAPI) ──► web/ (React + TypeScript)
                                                   (serialise/render only; no logic)

 Phase 3 (separate optional extra, created later):  vision/  ──produces──► list[Event]
   ffmpeg/shots ► pose+track (YOLO-pose+ByteTrack) ► identity ► temporal model ► post-process ► Events
   (imports core.events; core NEVER imports vision)
```

**Dependency rules (enforced by an AST-based test, no extra dependency):** `events`, `sequences`, `analytics`, `evaluation` may not import `psycopg`, `fastapi`, `cv2`, `torch`, or `database`. `database` may not import `analytics`. UI may not import `psycopg`.

**Key architectural decisions**
1. **Events are immutable facts scoped to a *source run*.** Corrections = re-import the whole source in one transaction (`ON DELETE CASCADE`). No in-place row editing in MVP.
2. **Analytics take `Sequence[Event]`, not a DB handle.** Deterministic, unit-testable with plain lists. The repository fetches ordered streams; n-gram logic stays in Python (tokenizers + gap segmentation don't map cleanly to SQL). SQL window functions (`LEAD`) are used only as an *independent cross-check* in tests.
3. **Ontology is defined once in code** (`ontology.py`: action → category, required/allowed attributes, allowed directions) and synced into `action_types` by the migration runner. A test asserts DB rows == code. Adding SLIP/STEP later = data, not DDL.
4. **Tokenization is a projection** (`Event → str`), chosen per query.
5. **Optional dependency extras:** core = `pydantic`; `[db]` = `psycopg[binary]`; `[api]` = `fastapi` + `uvicorn`; `[cv]` = torch/ultralytics/opencv (later); `[dev]` = pytest, ruff.
6. **Determinism:** integer times, total ordering `(start_ms, end_ms, id)`, stable sorts with explicit tie-breaks, no unseeded randomness.

---

## 2. Event data model

| Field | Type | Rule |
|---|---|---|
| `fighter` | slug | must be a participant of `fight` |
| `fight`, `video` | slug | natural keys (repository maps to ids); keeps analytics DB-free |
| `start_ms`, `end_ms` | int ≥ 0 | `end ≥ start`; ≤ video duration when known; must not fall in an unobserved interval |
| `action_type` | str ∈ ontology | JAB, CROSS, HOOK, UPPERCUT (+ SLIP, ROLL, PULL, BLOCK, PARRY, STEP registered but low-priority to annotate) |
| `category` | derived | PUNCH / DEFENSE / MOVEMENT from ontology |
| `side` | LEAD/REAR | **required iff PUNCH**; JAB⇒LEAD, CROSS⇒REAR |
| `target` | HEAD/BODY \| null | punches only; null = unknown |
| `direction` | LEFT/RIGHT/FORWARD/BACK | non-punches only; SLIP/ROLL∈{L,R}, STEP required (any) — per ontology; direction = fighter's own frame |
| `outcome` | LANDED/BLOCKED/MISSED \| null | punches only; null = unknown |
| `confidence` | [0,1] \| null | null for humans (not a fake 1.0) |
| `round_number` | int \| null | derived from round spans at import; optional override |
| `metadata` | JSON | ML extras only (track id, class probs); never filtered on in core analytics |

**Time convention (must be fixed before annotating):** `start_ms` = first frame the striking hand/defensive motion visibly begins; `end_ms` = frame of full extension/return-to-guard (punch) or motion completion. Human boundaries are fuzzy (±3–5 frames), so **evaluation and matching use the event *midpoint* with a tolerance**, never boundary IoU. Ordering key `(start_ms, end_ms, id)`.

**Two-fighter reality:** defenses are reactions to the *opponent's* punch → "defensive reaction" queries need the interleaved **fight stream** (both fighters), tokens carry actor (`SELF:JAB`, `OPP:SLIP`). Reactions are inferred from temporal proximity, never annotated.

---

## 3. PostgreSQL schema (`migrations/0001_core.sql`)

```sql
CREATE TABLE fighters (
  id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  slug TEXT NOT NULL UNIQUE, name TEXT NOT NULL,
  stance TEXT CHECK (stance IN ('ORTHODOX','SOUTHPAW','SWITCH')),
  metadata JSONB NOT NULL DEFAULT '{}');

CREATE TABLE fights (
  id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  slug TEXT NOT NULL UNIQUE, fought_on DATE, weight_class TEXT,
  scheduled_rounds SMALLINT CHECK (scheduled_rounds > 0),
  metadata JSONB NOT NULL DEFAULT '{}');

CREATE TABLE fight_participants (              -- 1..2 fighters per fight
  fight_id BIGINT NOT NULL REFERENCES fights(id) ON DELETE CASCADE,
  fighter_id BIGINT NOT NULL REFERENCES fighters(id),
  corner TEXT NOT NULL CHECK (corner IN ('RED','BLUE')),
  stance TEXT CHECK (stance IN ('ORTHODOX','SOUTHPAW','SWITCH')),  -- overrides fighter default
  PRIMARY KEY (fight_id, fighter_id), UNIQUE (fight_id, corner));

CREATE TABLE videos (                          -- Video hangs off Fight, not Round
  id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  fight_id BIGINT NOT NULL REFERENCES fights(id),
  slug TEXT NOT NULL UNIQUE, path TEXT NOT NULL,
  sha256 TEXT,                                  -- ties annotations to the exact file
  fps DOUBLE PRECISION CHECK (fps > 0), duration_ms INTEGER CHECK (duration_ms > 0),
  width INTEGER, height INTEGER, metadata JSONB NOT NULL DEFAULT '{}',
  UNIQUE (id, fight_id));

CREATE TABLE rounds (                          -- bell-to-bell span in *video* time
  video_id BIGINT NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
  round_number SMALLINT NOT NULL CHECK (round_number > 0),
  start_ms INTEGER NOT NULL CHECK (start_ms >= 0), end_ms INTEGER NOT NULL,
  PRIMARY KEY (video_id, round_number), CHECK (end_ms > start_ms));

CREATE TABLE action_types (                    -- seeded from ontology.py
  code TEXT PRIMARY KEY,
  category TEXT NOT NULL CHECK (category IN ('PUNCH','DEFENSE','MOVEMENT')),
  UNIQUE (code, category));

CREATE TABLE event_sources (                   -- one annotation pass OR one model run, per video
  id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  video_id BIGINT NOT NULL REFERENCES videos(id),
  kind TEXT NOT NULL CHECK (kind IN ('HUMAN','MODEL')),
  name TEXT NOT NULL,                           -- annotator id | model name
  version TEXT NOT NULL DEFAULT '',             -- annotation revision | checkpoint
  ontology_version TEXT NOT NULL,
  content_sha256 TEXT,                          -- idempotent re-import
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  params JSONB NOT NULL DEFAULT '{}',           -- thresholds, git commit, model config
  UNIQUE (video_id, kind, name, version));

CREATE TABLE unobserved_intervals (            -- coverage: where events must NOT exist / are not exhaustive
  source_id BIGINT NOT NULL REFERENCES event_sources(id) ON DELETE CASCADE,
  start_ms INTEGER NOT NULL CHECK (start_ms >= 0), end_ms INTEGER NOT NULL,
  kind TEXT NOT NULL CHECK (kind IN ('REPLAY','CUTAWAY','UNSUPPORTED_ANGLE','UNCERTAIN')),
  PRIMARY KEY (source_id, start_ms), CHECK (end_ms > start_ms));

CREATE TABLE events (
  id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  source_id BIGINT NOT NULL REFERENCES event_sources(id) ON DELETE CASCADE,
  video_id  BIGINT NOT NULL REFERENCES videos(id),   -- denormalised, derived from source on insert
  fight_id  BIGINT NOT NULL REFERENCES fights(id),   -- denormalised, derived on insert
  fighter_id BIGINT NOT NULL REFERENCES fighters(id),
  round_number SMALLINT,
  start_ms INTEGER NOT NULL CHECK (start_ms >= 0), end_ms INTEGER NOT NULL,
  action_type TEXT NOT NULL, category TEXT NOT NULL,
  side TEXT CHECK (side IN ('LEAD','REAR')),
  target TEXT CHECK (target IN ('HEAD','BODY')),
  direction TEXT CHECK (direction IN ('LEFT','RIGHT','FORWARD','BACK')),
  outcome TEXT CHECK (outcome IN ('LANDED','BLOCKED','MISSED')),
  confidence REAL CHECK (confidence BETWEEN 0 AND 1),
  metadata JSONB NOT NULL DEFAULT '{}',
  CHECK (end_ms >= start_ms),
  FOREIGN KEY (action_type, category) REFERENCES action_types (code, category),
  FOREIGN KEY (fight_id, fighter_id) REFERENCES fight_participants (fight_id, fighter_id),
  FOREIGN KEY (video_id, round_number) REFERENCES rounds (video_id, round_number), -- NULL round ⇒ unchecked (MATCH SIMPLE)
  CHECK ((category = 'PUNCH') = (side IS NOT NULL)),
  CHECK (target  IS NULL OR category = 'PUNCH'),
  CHECK (outcome IS NULL OR category = 'PUNCH'),
  CHECK (direction IS NULL OR category <> 'PUNCH'),
  CHECK (action_type <> 'JAB'   OR side = 'LEAD'),
  CHECK (action_type <> 'CROSS' OR side = 'REAR'));

CREATE INDEX events_source_time ON events (source_id, start_ms, id);      -- timeline reads (primary path)
CREATE INDEX events_fighter     ON events (fighter_id, fight_id, start_ms);
CREATE INDEX events_video       ON events (video_id);                      -- FK / cascade support
```

**Design notes**
- **Integrity split.** DB enforces PK/FK, value domains, temporal sanity, category-level attribute rules, fighter-in-fight and round existence. Per-action rules (SLIP needs LEFT/RIGHT, etc.) live in `ontology.py` + Pydantic and are enforced on every write path. `video_id`/`fight_id` are *derived from `source_id` inside the INSERT* (`INSERT … SELECT`) so callers can't supply inconsistent values; a `db check` audit command verifies it.
- **"Unobserved" is enforced at import** (events inside REPLAY/CUTAWAY/UNSUPPORTED_ANGLE/UNCERTAIN are rejected) and used at analysis time as **hard sequence breaks** and to compute *observed* minutes for rates.
- **JSONB policy:** ML/annotator extras only (`events.metadata`, `event_sources.params`, `videos.metadata`). Anything grouped/filtered regularly gets promoted to a real column via migration. No GIN until `EXPLAIN` demands it.
- **Indexes:** start with the three above; add only from measured query plans. **Verification target (M4):** synthetic 2M-row dataset; one fighter/fight timeline ≲ tens of ms; multi-fight per-fighter aggregation ≲ 1 s. Not assumed — measured.
- **Migrations:** numbered `.sql` files + ~40-line runner recording applied versions in `schema_migrations`. No Alembic/SQLAlchemy/ORM (the schema is the reviewed artifact; ~8 tables).

---

## 4. Repository structure (root = `D:\Boxing-analyst`)

```
AGENTS.md            # source of truth for conventions (CLAUDE.md contains only "@AGENTS.md")
README.md  pyproject.toml  docker-compose.yml  .gitignore
docs/  requirements.md  architecture.md  database.md  annotation.md  ontology.md  plan.md
migrations/          0001_core.sql …
src/boxing_ai/
  ontology.py                 # actions, categories, attribute rules (single source of truth)
  events/                     # Event model, validation, ordering
  annotations/                # CSV+TOML loader/writer → validated Events, line-numbered errors
  sequences/                  # tokenizers, segmentation, n-grams, transition tables
  analytics/                  # profile, followers, entry/exit patterns, per-round rates
  evaluation/                 # event matching + hierarchical metrics (built BEFORE any model)
  database/                   # connection, migrate, repository (only SQL in the codebase)
  reporting.py                # thin: repository → analytics → Report dataclass
  cli.py                      # argparse: init-db | import | report | db-check
  # vision/ (detection, tracking, pose, classification) is created in Phase 3, not before
src/boxing_ai/api.py         # FastAPI; serialises Report/Event objects only
web/                         # React + TypeScript (Vite): viewer + annotator; talks to the API only
tests/  unit/  integration/(db)  fixtures/(hand-calculated golden data)
scripts/  data/{raw(gitignored), annotations(tracked), processed(gitignored)}
```
**Changes vs your proposal & why:** (1) no empty CV packages (premature scaffolding); (2) added `annotations/`, `evaluation/`, `ontology.py`, `reporting.py`, `migrations/`; (3) `git init` + `.gitignore` for `data/raw`, `data/processed`, `.env`, `.venv` — **annotation files are the ground-truth dataset and are versioned in git; raw video is not** (identity by `sha256` in `videos`); (4) AGENTS.md primary, CLAUDE.md a one-line import.

---

## 5. Annotation format & protocol

**One folder per video:** `data/annotations/<video-slug>/{meta.toml, events.csv}`. CSV for the ~hundreds of rows you'll hand-edit (diff-able, spreadsheet-friendly); TOML for the small header (comments allowed, stdlib parser).

`meta.toml`
```toml
schema_version = 1
ontology_version = "0.1"
[annotator]  name = "brin"  kind = "HUMAN"  version = "r1"
[video]      slug = "fightX-full"  file = "raw/fightX-full.mp4"  fps = 29.97   # sha256 auto-computed
[fight]      slug = "fighterA-vs-fighterB-2024-05-04"  date = 2024-05-04
[[fighters]] corner = "RED"  slug = "fighter-a"  name = "Fighter A"  stance = "ORTHODOX"
[[fighters]] corner = "BLUE" slug = "fighter-b"  name = "Fighter B"  stance = "SOUTHPAW"
[[rounds]]   number = 1  start = "03:12.400"  end = "06:12.500"
[[unobserved]] start = "04:01.000" end = "04:09.500" kind = "REPLAY"
```
`events.csv` (times accept `SS.mmm`, `MM:SS.mmm`, `HH:MM:SS.mmm`; `side` optional for JAB/CROSS; `round` optional)
```
start,end,fighter,action,side,target,direction,outcome,notes
03:26.320,03:26.500,RED,JAB,,HEAD,,LANDED,
03:26.710,03:26.910,RED,CROSS,,HEAD,,MISSED,
03:27.420,03:27.700,BLUE,SLIP,,,LEFT,,
```
Loader: parse → normalise (fill JAB/CROSS side, derive round from spans, ms conversion) → validate (ontology, fighter-in-fight, bounds, not in unobserved) → **collect all errors with file+line**, don't stop at the first.

**Protocol rules (`docs/annotation.md`) — these matter more than the file format:**
1. **Exhaustive-or-excluded:** inside any observed interval *every* punch is annotated. If you can't be exhaustive (flurry, clinch, blur) mark that interval `UNCERTAIN`. Otherwise unannotated real punches become false negatives that corrupt ML training and FP-rate evaluation.
2. **Mark REPLAY / CUTAWAY / UNSUPPORTED_ANGLE** (tight shots, corner/crowd cameras). Phase-3 v1 targets the **wide main-camera live shots** only.
3. Annotate **thrown** punches (visible attempt), plus `outcome` when discernible; feints, guard adjustments, arm swings, clinch pushing, referee separations are *not* punches.
4. Fighter identity by corner; LEAD/REAR relative to that fighter's stance.
5. Version the ontology + protocol (`ontology_version`).
6. **Double-annotate ~5–10 % of clips (different annotator or same annotator weeks apart) to measure inter-annotator agreement** — the ceiling any model can be judged against, and the source of the matching tolerance.

---

## 6. Sequence-analysis design

**Pipeline:** `events → filter(source, fighter, fight, round, time) → order → StreamSpec(categories, tokenizer, gap_ms) → segment into bursts → count/transition tables → report`.

- **Tokenizers:** `by_action` (`JAB`), `by_action_direction` (`SLIP_LEFT`), `by_action_target` (`HOOK_BODY`), custom callables.
- **Segmentation (bursts):** split when `max(0, next.start − prev.end) > gap_ms`, **and always** at unobserved-interval boundaries and round boundaries. Defaults are *starting guesses* (`combo_gap_ms≈1000`, `exchange_gap_ms≈2500`) to be **tuned from the real inter-punch gap histogram** (expected bimodal), with a sensitivity report.
- **n-grams** never cross segment boundaries. Counts + rates (per observed minute / per round — raw counts are confounded by exposure).
- **Conditional probability** uses explicit `START`/`END` sentinels so distributions sum to 1 and entry/exit patterns fall out (`P(JAB|START)`, `P(END|CROSS)`). Always report counts alongside p, apply `min_count`; higher orders (k=2) supported, no smoothing/back-off/HMMs until the simple approach is shown insufficient.
- **Queries:** followers after X, entry patterns (first-k of bursts), exit patterns (last-k), defensive reactions (fight stream with actor-tagged tokens), per-round distributions.
- **Fighter-specific ("distinctive") sequences:** lift vs. baseline with min-support — post-MVP (needs ≥ several fighters).

**Hand-calculated golden example** (your stream, one burst; tokens by action+direction):
`JAB JAB CROSS SLIP_LEFT JAB CROSS`
- Unigrams: JAB 3, CROSS 2, SLIP_LEFT 1.
- Bigrams (5 = n−1): (JAB,JAB)1 · (JAB,CROSS)**2** · (CROSS,SLIP_LEFT)1 · (SLIP_LEFT,JAB)1.
- Trigrams (4): (JAB,JAB,CROSS)1 · (JAB,CROSS,SLIP_LEFT)1 · (CROSS,SLIP_LEFT,JAB)1 · (SLIP_LEFT,JAB,CROSS)1.
- With sentinels: P(JAB|START)=1 · P(JAB|JAB)=1/3 · **P(CROSS|JAB)=2/3** · P(SLIP_LEFT|CROSS)=1/2 · **P(END|CROSS)=1/2** · P(JAB|SLIP_LEFT)=1.
- *Why `END` matters:* without it P(SLIP_LEFT|CROSS)=1/1=100 %, silently ignoring the CROSS that ended the burst.
- Add a JAB 30 s later → separate burst; no (CROSS,JAB) bigram across the pause. Also user's minimal case `JAB,CROSS,HOOK` → bigrams {(JAB,CROSS),(CROSS,HOOK)}, trigram {(JAB,CROSS,HOOK)}.

These fixtures live in `tests/fixtures/` and in `docs/`, and are additionally cross-checked against a `LEAD()` SQL implementation in the DB integration tests.

---

## 7. Computer-vision / ML strategy (Phase 3 — gated, after MVP)

**It is temporal action *detection*, not image classification:** find start/end of ~0.1–0.3 s events in a continuous stream where background dominates, then classify. Single-frame classifiers are insufficient (a cross and a retraction look alike in one frame).

**Stage pipeline (each stage caches artifacts on disk so failures are debuggable at the right stage):**
1. **Ingest:** ffmpeg → constant-frame-rate normalisation; store fps/sha256 (VFR + re-encodes shift timestamps, silently misaligning annotations).
2. **Shot/replay/angle gate (broadcast-specific):** shot-boundary detection (e.g. PySceneDetect), replay/slow-mo detection, camera-type classification → auto-generate `unobserved_intervals`. v1 restricts inference to the **wide main-camera live shots**.
3. **Pose + tracking:** one top-down model (YOLO-pose family) + ByteTrack/BoT-SORT; keep the 2 boxers, drop referee/corner. Bulk output → Parquet/npz under `data/processed/`.
4. **Identity:** track → corner via torso/shorts colour + human-seeded anchors; ID swaps in clinches are expected → human correction file in early phases.
5. **Temporal model on per-fighter pose sequences:** normalised (hip-centred, torso-scaled) keypoints, velocities, elbow angles, wrist–shoulder extension, **opponent-relative features** (needed for target HEAD/BODY). Small TCN/BiGRU with dense per-frame outputs incl. an explicit **BACKGROUND** class; **event-centred targets** (per-frame heatmap around event midpoint) rather than boundary regression, since human boundaries are noisy.
6. **Two-stage decomposition:** (a) punch/no-punch detector tuned for **precision** with hard negatives (guard shifts, feints, arm swings, clinch pushes) → (b) type/side/target classifier on detected segments. This isolates false positives and matches your requirement to measure error types separately. `side` is largely computable from which wrist extends + stance.
7. **Post-processing:** peak detection/hysteresis/NMS → `Event(confidence=…)` → same importer path as human annotations → `event_sources(kind='MODEL', params={git commit, thresholds})`.

**Order of work (each gated by the evaluation harness, M8):** feasibility spike on a small annotated broadcast sample (is wrist-keypoint quality adequate at wide-shot scale? measure) → **heuristic baseline** (wrist-velocity/extension threshold; the floor to beat) → learned temporal model → decide (from ablations, not hunch) whether to add appearance/optical-flow features or a video model. Model-assisted pre-annotation later (model proposes, human corrects — 3–5× faster; the schema already supports it).

**Realistic expectation:** automatic punch *detection* on broadcast wide shots is plausible; reliable *type* (hook vs cross vs uppercut), *target*, and *defensive* classification from a distant single camera is hard research-grade work. The MVP is valuable regardless because it runs on human annotation. Rule-of-thumb data need for a first prototype: **a few thousand annotated punches across several fights/camera setups** (a guess, revised after the spike).

**Integration contract:** `vision.predict_events(video_meta) → ModelRun(events, unobserved, params)` persisted via the *same* `repository.import_source(...)` as annotations. A test proves identical analytics on a HUMAN vs a MODEL source holding identical events.

**Evaluation (designed now, built at M8 and tested on synthetic predictions before any model exists):**
- **One-to-one matching** (Hungarian, cost = |Δmidpoint|) gated at τ ∈ {50, 100, 150, 250} ms, **class-agnostic first**. Unmatched preds = FP, unmatched GT = FN.
- **Nested, separately reported levels:** (1) *detection* P/R/F1 + **FP per observed minute** (excludes unobserved/UNCERTAIN/inter-round time); (2) *fighter attribution* accuracy among matches; (3) *timing error* (signed median/p90 ms); (4) *type* accuracy + confusion matrix + macro-F1; (5) *side*; (6) *target*; (7) *strict end-to-end* P/R/F1 (all correct).
- **Downstream-fidelity metrics** (the ones that matter for this product): distance between transition tables from predicted vs. gold events, top-k n-gram overlap/rank correlation. *A missed punch and a false positive both corrupt adjacency*, so an FP rate tolerable for counting can be fatal for n-grams.
- **Splits by fight (grouped CV), never by frame/clip** (adjacent frames leak; frame-level accuracy is meaningless here). Baselines: heuristic detector + class-prior. Operating threshold chosen on validation PR curve.

---

## 8. Testing strategy

| Area | Tests |
|---|---|
| Event creation | valid punch/defense builds; defaults; JAB/CROSS side fill |
| Event validation | reject: `end<start`, unknown action, JAB+REAR, punch w/o side, SLIP w/ target/side, SLIP w/o direction, confidence ∉[0,1], negative time, outside video, inside unobserved interval, fighter not in fight |
| Chronological ordering | out-of-order input sorted; ties broken by `(end_ms,id)`; overlaps; stable/deterministic |
| Sequence generation | `JAB,CROSS,HOOK` → expected 2-/3-grams; no cross-burst n-grams; gap threshold boundary (==gap vs gap+1 ms); unobserved break; round break |
| Statistics | golden example (§6) exact; probabilities sum to 1 with sentinels; min_count filter; rates use observed minutes |
| Annotation I/O | valid folder loads; every invalid class yields file+line message; CSV→Event→CSV round trip; time-format parsing |
| Database (real Postgres, `integration` marker) | round-trip equality; **each CHECK/FK rejects bad raw SQL** (e.g. JAB+REAR, non-participant fighter, unknown round); idempotent re-import (hash); source replace is atomic; cascade; ontology-in-code == `action_types`; SQL `LEAD()` bigram counts == Python counts; HUMAN vs MODEL source → identical analytics |
| Performance | synthetic 2M-row dataset + `EXPLAIN` on the primary queries (M4) |
| Architecture | AST import-boundary test (core never imports psycopg/fastapi/cv2/torch) |
| CLI/API/UI | end-to-end import→report equals hand-calculated numbers (CLI and API `TestClient`); front-end component tests + a manual browser pass |
| Evaluation harness (M8) | synthetic predictions: jitter times → timing metrics; drop events → recall; inject events → FP/min; flip classes → confusion matrix; wrong fighter → attribution — metrics match hand math |
| ML (Phase 3) | §7 metrics on grouped-CV; FP/min; downstream fidelity |

Tooling: `pytest` (`-m "not integration"` fast loop; integration runs when a test DSN is configured), `ruff`. Types throughout; no other tooling until needed.

---

## 9. MVP definition

> Given manually annotated broadcast footage (CSV+TOML), import individual actions into PostgreSQL and generate deterministic statistical analyses of boxing sequences, viewable in a basic interface.

Proves your 7 criteria: event model works (M1) · DB works (M4) · efficient queries (M4 EXPLAIN) · chronological sequences (M2) · n-grams (M2) · fighter stats (M2/M5) · basic interface (M5 CLI, M6 API + web viewer). **Out of scope:** any CV/ML, multi-user auth, sophisticated sequence mining. Defensive actions are **in the ontology and test fixtures** (to prove extensibility; punch-only assumptions would otherwise leak into the schema) but annotating them is optional/low-priority.

---

## 10. Milestones (each independently testable; proceed only when the previous passes)

| # | Milestone | Exit criteria | Size |
|---|---|---|---|
| **M0** | Scaffold: `git init`, 3.11 `.venv`, `pyproject` (extras), `.gitignore`, AGENTS.md/CLAUDE.md, docs skeleton, pytest+ruff | `pytest` and `ruff` run clean | S |
| **M1** | Ontology + Event model + validation + ordering (no DB/IO) | All creation/validation/ordering tests green | S |
| **M2** | Sequence engine + analytics (pure) with golden fixtures | §6 numbers reproduced exactly; boundary/gap/break tests green | M |
| **M3** | Annotation format + loader/writer + `docs/annotation.md` + synthetic sample folder | Valid loads; invalid → precise errors; round-trip | M |
| **M4** | Postgres: compose, migrations, schema, repository, import, queries | Integration suite green on real DB incl. constraint rejection, LEAD() cross-check, EXPLAIN on 2M rows | L |
| **M5** | `reporting` + CLI (`init-db/import/report/db-check`) — **first usable MVP**: you annotate a real clip and run it | E2E test matches hand-calculated report; you try it on a real fight round | M |
| **M6** | FastAPI layer (sources, fighter report, events per video, video streaming with seek) + React/TypeScript viewer: filters, totals, outcome table, top sequences with landed %, "after X", video player synced to an event timeline, click-a-pattern → jump to each occurrence | API `TestClient` suite; no logic in API/UI. **MVP complete.** (Use the `dataviz` skill for charts.) | L |
| **M7** | Web annotator in the same app: frame stepping, hotkeys per action/outcome, events drawn on the timeline, writes the same `events.csv`/import path | Replaces the OpenCV tool idea; build once real annotation shows where the time goes | M |
| **M8** | Evaluation harness (§7) on synthetic predictions + inter-annotator agreement tool | Metric self-tests match hand math | M |
| **M9** | Distinctive-sequence analysis (lift), between-round/fight change, gap-threshold sensitivity report | Needs ≥ several fights of annotations | M |
| **V0–V5** | Phase 3: feasibility spike → ingest/shots/replays → pose+track → heuristic baseline → temporal model → MODEL-source integration | Gated by M8 metrics; V0 go/no-go decision | XL |

**After you approve this plan** I will implement **M0 → M1 → M2**, run and report the tests, then pause for your go-ahead: M3 needs your input on the annotation protocol, and M4 needs Docker Desktop started.

---

### Decision: web stack (2026-09-25, replaces Streamlit)

The interface has two hard requirements Streamlit cannot meet: statistics linked to footage (click a pattern → jump to
every occurrence, event timeline synced with playback) and, later, a frame-accurate annotation tool — annotation
throughput is the project's real bottleneck. Chosen: **FastAPI** (thin, reuses the pydantic models; Python stays
server-side for analytics and CV) + **React + TypeScript (Vite)** (native `<video>` with precise seeking and
`requestVideoFrameCallback`; largest charting/table/video ecosystem). Considered: Dash/NiceGUI/Reflex (video
interaction still needs custom JS, smaller ecosystems), Django (ORM would compete with the hand-written schema),
SvelteKit (valid, leaner; React chosen for ecosystem). Cost: a second language and a JS build step. Hosting broadcast
footage publicly raises rights questions (risk 11); annotations are shareable.

---

## 11. Major technical risks

1. **Annotation cost/quality is the true bottleneck** (0.1–0.3 s events, flurries, fuzzy boundaries). Mitigate: exhaustive-or-`UNCERTAIN` rule, written protocol, double-annotation, web annotator with hotkeys (M7), later model pre-annotation.
2. **Broadcast domain difficulty:** cuts, replays (double counting), many angles, referee, overlays, fighters small in wide shots (~150–300 px), motion blur, 25–30 fps ⇒ 3–6 frames/punch. Mitigate: `unobserved_intervals`, main-wide-camera-only v1, V0 feasibility spike before any big investment, possible crop-and-upscale second pass.
3. **Pose failure modes:** wrists occluded by gloves/opponent, overlapping bodies, no glove/impact cues. Mitigate: measure keypoint quality first; alternatives RTMPose/ViTPose; add appearance features only if ablations justify.
4. **Fighter attribution/identity** in clinches, ID swaps, camera cuts. Human-seeded anchors + corrections; measured separately.
5. **False positives and misses corrupt n-grams** (spurious adjacencies). Precision-first operating point; downstream-fidelity metric; hard-negative mining.
6. **Sparse statistics:** small counts, 4ⁿ token growth, conditional probs on n=1. Always show counts, `min_count`, intervals later; cap n at 3–4.
7. **Definition drift:** "combination/exchange" gap thresholds; ontology growth. Parameterised + versioned (`ontology_version`, `params`); annotations survive ontology growth.
8. **Timestamp integrity:** VFR/re-encoded video misaligning annotations. `sha256` + CFR normalisation + stored fps.
9. **Hardware:** GTX 970 (sm_52, 4 GB): PyTorch wheel/CUDA compatibility unverified, no big-model fine-tuning. Verify at V0; CPU fallback; occasional rented GPU.
10. **Environment:** Docker engine stopped, Python 3.9 default, no ffmpeg/psql. Addressed in M0/M4.
11. **Licensing/rights:** Ultralytics AGPL-3.0; broadcast-footage rights. Keep raw video local & un-versioned; annotations (facts) are the shareable dataset. Fine for personal use; revisit if you distribute.
12. **Scope creep** (dashboard polish or CV before data). Milestone gating above.

---

## 12. Assumptions to resolve

**Resolved by you:** broadcast footage · Docker Postgres · CSV-first annotation.

**My decisions (tell me if you disagree — cheap now, expensive after annotating):**
- `SLIP`+`direction` instead of `SLIP_LEFT` as stored class (tokenizer decides).
- Integer-millisecond times; midpoint-with-tolerance for matching.
- Optional `outcome`; optional `target`; `side` required for punches & consistency-validated.
- Video ← Fight, Round = video-time span; `event_sources` + `unobserved_intervals` added.
- Python 3.11, psycopg3 + SQL migrations (no ORM/Alembic), Pydantic v2 for validation, argparse CLI.
- Defensive actions in ontology/fixtures now; annotation of them optional.

**Still open (don't block M0–M2; need answers by the noted milestone):**
1. *(M3)* Exact time convention for `start_ms/end_ms` (start of motion vs impact) — I proposed one; confirm after annotating a real 30-second clip.
2. ~~*(M3)* Annotate `outcome` from day one?~~ **Resolved: yes** — required for punches in fights/sparring (`UNKNOWN` allowed when hidden).
3. ~~*(M3)* Single-fighter footage in scope?~~ **Resolved: yes**, for comparison — `fight.kind` = FIGHT/SPARRING/PADS/BAG/SHADOW. Long-term goal: find the most *successful* patterns of elite fighters (Inoue, Durán, …), which is why outcome is mandatory. Also added (ontology 0.2): `FEINT` action and punch `commitment` (PROBE/FULL). **M4 schema must follow:** category CHECK gains `'FEINT'`, events gain `commitment`, fights gain `kind`, and the side/target CHECKs become per-category (feints take optional side/target).
4. ~~*(M4/M5)* Default source rule~~ **Resolved (M5):** per video the single HUMAN source, else the single MODEL source; otherwise the report refuses and lists candidates (`--source name[@version]`). See `src/boxing_ai/reporting.py`.
5. *(M2/M9)* Gap-threshold defaults — will be tuned from your real data.
6. *(V0)* GPU stack viability on GTX 970; willingness to use occasional cloud GPU/Colab (data-rights aware).
7. *(Later)* Personal use only, or eventually distributed? (AGPL/footage rights.)

---

## Verification (how each step is proven)

- **Every milestone:** `ruff` + `pytest` (unit; integration from M4). Nothing advances with red tests.
- **Sequence analytics (M2):** compare against the hand-computed tables in §6 stored as fixtures, plus the independent SQL `LEAD()` cross-check in M4.
- **DB design (M4):** run constraint-rejection tests via raw SQL, inspect the schema with `\d+`, run `EXPLAIN (ANALYZE)` on the synthetic 2M-row dataset.
- **End to end (M5):** import the synthetic sample folder, run `boxing-ai report`, assert output equals hand-calculated numbers; then run it on one real annotated round of your footage.
- **API + web viewer (M6):** API tests against the real DB via `TestClient`; front-end tests; a manual browser pass (click-to-seek, timeline sync).
- **Evaluation harness (M8):** perturb gold events synthetically and confirm each metric moves exactly as expected.
