"""PostgreSQL: migrations, import round trip, idempotency, atomic replace, constraints, SQL cross-checks."""

from __future__ import annotations

from pathlib import Path

import psycopg
import pytest

from boxing_ai.analytics import round_stats, top_ngrams, transition_probabilities
from boxing_ai.annotations import SourceKind, load_annotation
from boxing_ai.database import (
    ImportConflict,
    action_counts,
    import_annotation,
    import_source,
    list_sources,
    load_events,
    load_rounds,
    load_unobserved,
    migrate,
    repository,
)
from boxing_ai.ontology import ACTIONS, Corner, SessionKind
from boxing_ai.sequences import StreamSpec, count_ngrams, token_segments

SAMPLE = Path(__file__).resolve().parents[2] / "data" / "annotations" / "sample-synthetic"


@pytest.fixture(scope="module")
def sample():
    return load_annotation(SAMPLE)


def without_ids(events):
    return [e.model_copy(update={"id": None}) for e in events]


def as_model_run(annotation):
    annotator = annotation.meta.annotator.model_copy(
        update={"kind": SourceKind.MODEL, "name": "fake-model", "version": "v1"}
    )
    return annotation.meta.model_copy(update={"annotator": annotator})


# --- schema ----------------------------------------------------------------------------------------


def test_migrate_is_idempotent_and_action_types_mirror_the_ontology(conn):
    assert migrate(conn) == []
    rows = dict(conn.execute("SELECT code, category FROM action_types").fetchall())
    assert rows == {code: spec.category.value for code, spec in ACTIONS.items()}
    assert conn.execute("SELECT version FROM schema_migrations").fetchall() == [("0001_core",)]


# --- import / read round trip ----------------------------------------------------------------------


def test_import_round_trip(conn, sample):
    result = import_annotation(conn, sample)
    assert (result.status, result.events) == ("created", 14)

    events = load_events(conn, [result.source_id])
    assert without_ids(events) == list(sample.events)
    assert all(e.id is not None for e in events)
    assert load_rounds(conn, [result.source_id]) == list(sample.rounds)
    assert load_unobserved(conn, [result.source_id]) == list(sample.unobserved)

    (info,) = list_sources(conn)
    assert (info.fight, info.video, info.kind, info.name, info.events) == (
        "fighter-a-vs-fighter-b-sample",
        "sample-synthetic",
        "HUMAN",
        "synthetic",
        14,
    )


def test_fighter_filter_and_sql_aggregation(conn, sample):
    source = import_annotation(conn, sample).source_id
    red = load_events(conn, [source], fighter="fighter-a")
    assert {e.fighter for e in red} == {"fighter-a"} and len(red) == 10
    assert action_counts(conn, "fighter-a", [source]) == {
        "JAB": 5,
        "CROSS": 2,
        "FEINT": 1,
        "HOOK": 1,
        "SLIP": 1,
    }


def test_reimport_is_idempotent_and_changed_content_replaces(conn, sample):
    first = import_annotation(conn, sample)
    assert import_annotation(conn, sample).status == "unchanged"

    fewer = sample.events[:-1]
    changed = sample.__class__(sample.folder, sample.meta, fewer, "different-hash")
    second = import_annotation(conn, changed)
    assert second.status == "replaced" and second.source_id != first.source_id
    assert without_ids(load_events(conn, [second.source_id])) == list(fewer)
    assert conn.execute("SELECT count(*) FROM events").fetchone() == (13,)
    assert conn.execute("SELECT count(*) FROM event_sources").fetchone() == (1,)


def test_failed_replace_leaves_the_old_source_intact(conn, sample, monkeypatch):
    first = import_annotation(conn, sample)
    changed = sample.__class__(sample.folder, sample.meta, sample.events, "different-hash")

    def boom(*args):
        raise RuntimeError("crash while inserting events")

    monkeypatch.setattr(repository, "_event_row", boom)
    with pytest.raises(RuntimeError):
        import_annotation(conn, changed)
    assert without_ids(load_events(conn, [first.source_id])) == list(sample.events)


def test_invalid_events_are_rejected_before_touching_the_database(conn, sample):
    bad = [sample.events[0].model_copy(update={"fighter": "nobody"})]
    with pytest.raises(ValueError, match="not in this fight"):
        import_source(conn, sample.meta, bad)
    assert conn.execute("SELECT count(*) FROM fights").fetchone() == (0,)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda m: {"rounds": m.rounds[:1]}, "round spans"),
        (
            lambda m: {"fight": m.fight.model_copy(update={"kind": SessionKind.SPARRING})},
            "stored as FIGHT",
        ),
        (
            lambda m: {
                "fighters": [
                    m.fighters[0].model_copy(update={"corner": Corner.BLUE}),
                    m.fighters[1].model_copy(update={"corner": Corner.RED}),
                ]
            },
            "corner",
        ),
    ],
)
def test_conflicting_second_source_is_refused(conn, sample, change, message):
    import_annotation(conn, sample)
    meta = as_model_run(sample).model_copy(update=change(sample.meta))
    with pytest.raises(ImportConflict, match=message):
        import_source(conn, meta, [])
    assert len(list_sources(conn)) == 1


# --- constraints reject bad rows written with raw SQL ----------------------------------------------


def _insert_raw(conn, source_id, **overrides):
    row = conn.execute(
        "SELECT source_id, video_id, fight_id, fighter_id, round_number, start_ms, end_ms,"
        " action_type, category, side, target, direction, outcome, commitment, confidence"
        " FROM events WHERE source_id = %s AND action_type = 'JAB' ORDER BY id LIMIT 1",
        (source_id,),
    ).fetchone()
    names = (
        "source_id video_id fight_id fighter_id round_number start_ms end_ms action_type category"
        " side target direction outcome commitment confidence"
    ).split()
    values = dict(zip(names, row, strict=True)) | overrides
    cols = ", ".join(values)
    marks = ", ".join(["%s"] * len(values))
    with conn.transaction():
        conn.execute(f"INSERT INTO events ({cols}) VALUES ({marks})", list(values.values()))


def test_raw_insert_of_a_valid_row_succeeds(conn, sample):
    source = import_annotation(conn, sample).source_id
    _insert_raw(conn, source)


@pytest.mark.parametrize(
    "overrides",
    [
        {"side": "REAR"},  # JAB is always LEAD
        {"action_type": "HOOK", "side": None},  # punch without side
        {"action_type": "CROSS", "side": "LEAD"},
        {"action_type": "FEINT", "category": "FEINT", "side": None, "outcome": "LANDED"},
        {"action_type": "SLIP", "category": "DEFENSE", "side": None, "commitment": "PROBE"},
        {"direction": "LEFT"},  # punch with direction
        {"category": "DEFENSE"},  # category disagrees with action_types
        {"action_type": "JAB_CROSS"},  # combinations are not actions
        {"round_number": 9},  # round not defined for the video
        {"start_ms": 5000, "end_ms": 4000},
        {"start_ms": -1},
        {"confidence": 1.5},
        {"side": "MIDDLE"},
        {"outcome": "KNOCKDOWN"},
    ],
)
def test_constraints_reject_invalid_rows(conn, sample, overrides):
    source = import_annotation(conn, sample).source_id
    with pytest.raises(psycopg.errors.IntegrityError):
        _insert_raw(conn, source, **overrides)


def test_fighter_must_take_part_in_the_fight(conn, sample):
    source = import_annotation(conn, sample).source_id
    (outsider,) = conn.execute(
        "INSERT INTO fighters (slug, name) VALUES ('outsider', 'Out Sider') RETURNING id"
    ).fetchone()
    with pytest.raises(psycopg.errors.ForeignKeyViolation):
        _insert_raw(conn, source, fighter_id=outsider)


# --- SQL cross-checks and source-agnostic analytics --------------------------------------------------

LEAD_BIGRAMS = """
WITH stream AS (
    SELECT e.action_type, e.round_number, e.start_ms, e.end_ms,
           LEAD(e.action_type) OVER w AS next_action,
           LEAD(e.start_ms) OVER w AS next_start,
           LEAD(e.round_number) OVER w AS next_round
    FROM events e JOIN fighters f ON f.id = e.fighter_id
    WHERE e.source_id = %(source)s AND f.slug = %(fighter)s
      AND e.category IN ('PUNCH', 'FEINT', 'DEFENSE')
    WINDOW w AS (ORDER BY e.start_ms, e.end_ms, e.id)
)
SELECT action_type, next_action, count(*)
FROM stream
WHERE next_action IS NOT NULL
  AND next_round IS NOT DISTINCT FROM round_number
  AND next_start - end_ms <= %(gap)s
  AND NOT EXISTS (
      SELECT 1 FROM unobserved_intervals u
      WHERE u.source_id = %(source)s AND u.start_ms < next_start AND u.end_ms > end_ms)
GROUP BY action_type, next_action
"""


@pytest.mark.parametrize("fighter", ["fighter-a", "fighter-b"])
def test_sql_lead_bigrams_match_python(conn, sample, fighter):
    source = import_annotation(conn, sample).source_id
    spec = StreamSpec()
    sql = conn.execute(LEAD_BIGRAMS, {"source": source, "fighter": fighter, "gap": spec.gap_ms})
    sql_counts = {(a, b): n for a, b, n in sql.fetchall()}

    events = load_events(conn, [source], fighter=fighter)
    unobserved = load_unobserved(conn, [source])
    python_counts = count_ngrams(token_segments(events, spec, unobserved), 2)
    assert sql_counts == dict(python_counts)
    if fighter == "fighter-a":  # the golden burst (docs/plan.md §6) plus round 2's FEINT JAB HOOK
        assert sql_counts == {
            ("JAB", "JAB"): 1,
            ("JAB", "CROSS"): 2,
            ("CROSS", "SLIP"): 1,
            ("SLIP", "JAB"): 1,
            ("FEINT", "JAB"): 1,
            ("JAB", "HOOK"): 1,
        }


def test_human_and_model_sources_give_identical_analytics(conn, sample):
    human = import_annotation(conn, sample).source_id
    model = import_source(conn, as_model_run(sample), sample.events).source_id

    def analytics(source):
        events = load_events(conn, [source])
        unobserved = load_unobserved(conn, [source])
        rounds = load_rounds(conn, [source])
        return (
            without_ids(events),
            top_ngrams(events, 2, unobserved=unobserved),
            transition_probabilities(events, unobserved=unobserved),
            round_stats(events, rounds, unobserved),
        )

    assert analytics(human) == analytics(model)
    assert {s.kind for s in list_sources(conn)} == {"HUMAN", "MODEL"}
