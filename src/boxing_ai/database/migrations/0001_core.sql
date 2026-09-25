-- Core schema. Design and rationale: docs/plan.md §3, as built: docs/database.md.
-- The database enforces keys, value domains, temporal sanity, category-level attribute rules, fighter-in-fight and
-- round existence. Per-action rules (SLIP needs LEFT/RIGHT, JAB is LEAD, ...) live in ontology.py and are enforced
-- on every write path by the Event model; the JAB/CROSS side rules are repeated here as a cheap backstop.

CREATE TABLE fighters (
    id       BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    slug     TEXT NOT NULL UNIQUE,
    name     TEXT NOT NULL,
    stance   TEXT CHECK (stance IN ('ORTHODOX', 'SOUTHPAW', 'SWITCH')),
    metadata JSONB NOT NULL DEFAULT '{}'
);

-- A bout or a training session (kind); "fight" is kept as the name for both.
CREATE TABLE fights (
    id               BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    slug             TEXT NOT NULL UNIQUE,
    kind             TEXT NOT NULL DEFAULT 'FIGHT'
                     CHECK (kind IN ('FIGHT', 'SPARRING', 'PADS', 'BAG', 'SHADOW')),
    fought_on        DATE,
    weight_class     TEXT,
    scheduled_rounds SMALLINT CHECK (scheduled_rounds > 0),
    metadata         JSONB NOT NULL DEFAULT '{}'
);

CREATE TABLE fight_participants (
    fight_id   BIGINT NOT NULL REFERENCES fights (id) ON DELETE CASCADE,
    fighter_id BIGINT NOT NULL REFERENCES fighters (id),
    corner     TEXT NOT NULL CHECK (corner IN ('RED', 'BLUE')),
    stance     TEXT CHECK (stance IN ('ORTHODOX', 'SOUTHPAW', 'SWITCH')),  -- stance in this fight
    PRIMARY KEY (fight_id, fighter_id),
    UNIQUE (fight_id, corner)
);

CREATE TABLE videos (
    id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    fight_id    BIGINT NOT NULL REFERENCES fights (id),
    slug        TEXT NOT NULL UNIQUE,
    path        TEXT,
    sha256      TEXT,
    fps         DOUBLE PRECISION CHECK (fps > 0),
    duration_ms INTEGER CHECK (duration_ms > 0),
    metadata    JSONB NOT NULL DEFAULT '{}'
);

-- Bell-to-bell spans in *video* time.
CREATE TABLE rounds (
    video_id     BIGINT NOT NULL REFERENCES videos (id) ON DELETE CASCADE,
    round_number SMALLINT NOT NULL CHECK (round_number > 0),
    start_ms     INTEGER NOT NULL CHECK (start_ms >= 0),
    end_ms       INTEGER NOT NULL,
    PRIMARY KEY (video_id, round_number),
    CHECK (end_ms > start_ms)
);

-- Synced from ontology.py by the migration runner; never edited by hand.
CREATE TABLE action_types (
    code     TEXT PRIMARY KEY,
    category TEXT NOT NULL CHECK (category IN ('PUNCH', 'FEINT', 'DEFENSE', 'MOVEMENT')),
    UNIQUE (code, category)
);

-- One annotation pass OR one model run over one video.
CREATE TABLE event_sources (
    id               BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    video_id         BIGINT NOT NULL REFERENCES videos (id),
    kind             TEXT NOT NULL CHECK (kind IN ('HUMAN', 'MODEL')),
    name             TEXT NOT NULL,
    version          TEXT NOT NULL DEFAULT '',
    ontology_version TEXT NOT NULL,
    content_sha256   TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    params           JSONB NOT NULL DEFAULT '{}',
    UNIQUE (video_id, kind, name, version)
);

-- Where events must not exist / are not exhaustive. Hard sequence breaks; excluded from observed time.
CREATE TABLE unobserved_intervals (
    source_id BIGINT NOT NULL REFERENCES event_sources (id) ON DELETE CASCADE,
    start_ms  INTEGER NOT NULL CHECK (start_ms >= 0),
    end_ms    INTEGER NOT NULL,
    kind      TEXT NOT NULL CHECK (kind IN ('REPLAY', 'CUTAWAY', 'UNSUPPORTED_ANGLE', 'UNCERTAIN')),
    PRIMARY KEY (source_id, start_ms, end_ms),
    CHECK (end_ms > start_ms)
);

CREATE TABLE events (
    id           BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_id    BIGINT NOT NULL REFERENCES event_sources (id) ON DELETE CASCADE,
    video_id     BIGINT NOT NULL REFERENCES videos (id),   -- denormalised; derived from source on insert
    fight_id     BIGINT NOT NULL REFERENCES fights (id),   -- denormalised; derived from video on insert
    fighter_id   BIGINT NOT NULL REFERENCES fighters (id),
    round_number SMALLINT,
    start_ms     INTEGER NOT NULL CHECK (start_ms >= 0),
    end_ms       INTEGER NOT NULL,
    action_type  TEXT NOT NULL,
    category     TEXT NOT NULL,
    side         TEXT CHECK (side IN ('LEAD', 'REAR')),
    target       TEXT CHECK (target IN ('HEAD', 'BODY')),
    direction    TEXT CHECK (direction IN ('LEFT', 'RIGHT', 'FORWARD', 'BACK')),
    outcome      TEXT CHECK (outcome IN ('LANDED', 'BLOCKED', 'MISSED')),
    commitment   TEXT CHECK (commitment IN ('PROBE', 'FULL')),
    confidence   DOUBLE PRECISION CHECK (confidence BETWEEN 0 AND 1),
    metadata     JSONB NOT NULL DEFAULT '{}',
    CHECK (end_ms >= start_ms),
    FOREIGN KEY (action_type, category) REFERENCES action_types (code, category),
    FOREIGN KEY (fight_id, fighter_id) REFERENCES fight_participants (fight_id, fighter_id),
    -- NULL round => unchecked (MATCH SIMPLE)
    FOREIGN KEY (video_id, round_number) REFERENCES rounds (video_id, round_number),
    -- category-level attribute rules (mirror ontology.ActionSpec flags)
    CHECK (category <> 'PUNCH' OR side IS NOT NULL),
    CHECK (side IS NULL OR category IN ('PUNCH', 'FEINT')),
    CHECK (target IS NULL OR category IN ('PUNCH', 'FEINT')),
    CHECK (outcome IS NULL OR category = 'PUNCH'),
    CHECK (commitment IS NULL OR category = 'PUNCH'),
    CHECK (direction IS NULL OR category IN ('DEFENSE', 'MOVEMENT')),
    CHECK (action_type <> 'JAB' OR side = 'LEAD'),
    CHECK (action_type <> 'CROSS' OR side = 'REAR')
);

CREATE INDEX events_source_time ON events (source_id, start_ms, id);   -- timeline reads (primary path)
CREATE INDEX events_fighter ON events (fighter_id, fight_id, start_ms);  -- per-fighter aggregation
CREATE INDEX events_video ON events (video_id);                          -- FK / cascade support
