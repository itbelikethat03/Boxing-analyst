"""Annotation folders: valid loads, precise file+line errors, CSV round trip, and the synthetic sample."""

import shutil
from pathlib import Path

import pytest

from boxing_ai.analytics import round_stats, select_events
from boxing_ai.annotations import AnnotationError, load_annotation, write_events_csv
from boxing_ai.ontology import (
    Category,
    Commitment,
    Direction,
    Outcome,
    Side,
    Target,
    UnobservedKind,
)
from boxing_ai.sequences import (
    StreamSpec,
    by_action_commitment,
    by_action_direction,
    token_segments,
)
from tests.factories import ev

SAMPLE = Path(__file__).resolve().parents[2] / "data" / "annotations" / "sample-synthetic"

META = """\
schema_version = 1
ontology_version = "0.2"
[annotator]
name = "tester"
[video]
slug = "v1"
duration = "10:00.000"
[fight]
slug = "f1"
[[fighters]]
corner = "RED"
slug = "red-guy"
name = "Red Guy"
stance = "ORTHODOX"
[[fighters]]
corner = "BLUE"
slug = "blue-guy"
name = "Blue Guy"
[[rounds]]
number = 1
start = "00:10.000"
end = "03:10.000"
[[rounds]]
number = 2
start = "04:10.000"
end = "07:10.000"
[[unobserved]]
start = "01:00.000"
end = "01:10.000"
kind = "REPLAY"
"""
SOLO_META = META.replace(
    '[[fighters]]\ncorner = "BLUE"\nslug = "blue-guy"\nname = "Blue Guy"\n', ""
)
HEADER = "start,end,fighter,action,side,target,direction,outcome,round,notes\n"


def make_folder(tmp_path: Path, csv_body: str, meta: str = META, header: str = HEADER) -> Path:
    (tmp_path / "meta.toml").write_text(meta, encoding="utf-8")
    (tmp_path / "events.csv").write_text(header + csv_body, encoding="utf-8")
    return tmp_path


def problems_of(folder: Path) -> list[str]:
    with pytest.raises(AnnotationError) as info:
        load_annotation(folder)
    return [str(p) for p in info.value.problems]


# --- valid input ---------------------------------------------------------------------------------


def test_valid_folder_normalises_events(tmp_path):
    folder = make_folder(
        tmp_path,
        "00:20.500,00:20.700,BLUE,cross,,body,,landed,,\n"
        "00:20.000,00:20.180,RED,JAB,,HEAD,,MISSED,1,first jab\n"
        "\n"
        "04:20.000,04:20.300,RED,SLIP,,,RIGHT,,,\n"
        "04:21.000,04:21.250,RED,HOOK,REAR,,,unknown,2,\n",
    )
    a = load_annotation(folder)

    assert [(e.start_ms, e.fighter, e.action_type) for e in a.events] == [
        (20_000, "red-guy", "JAB"),
        (20_500, "blue-guy", "CROSS"),
        (260_000, "red-guy", "SLIP"),
        (261_000, "red-guy", "HOOK"),
    ]
    jab, cross, slip, hook = a.events
    assert (jab.side, jab.target, jab.outcome, jab.round_number) == (
        Side.LEAD,
        Target.HEAD,
        Outcome.MISSED,
        1,
    )
    assert jab.metadata == {"notes": "first jab"} and jab.confidence is None
    assert (cross.side, cross.target, cross.outcome) == (Side.REAR, Target.BODY, Outcome.LANDED)
    assert (slip.direction, slip.round_number) == (Direction.RIGHT, 2)
    assert (hook.side, hook.outcome) == (Side.REAR, None)  # UNKNOWN is stored as None
    assert {(e.fight, e.video) for e in a.events} == {("f1", "v1")}
    assert [(r.number, r.start_ms, r.end_ms) for r in a.rounds] == [
        (1, 10_000, 190_000),
        (2, 250_000, 430_000),
    ]
    assert [(u.kind, u.start_ms, u.end_ms) for u in a.unobserved] == [
        (UnobservedKind.REPLAY, 60_000, 70_000)
    ]
    assert a.context.fighters == {"red-guy", "blue-guy"}


def test_semicolon_delimiter_bom_and_column_order(tmp_path):
    header = "\ufeffAction;Fighter;Start;End;Outcome\n"
    folder = make_folder(tmp_path, "JAB;RED;00:20.000;00:20.180;LANDED\n", header=header)
    (event,) = load_annotation(folder).events
    assert (event.action_type, event.start_ms, event.side) == ("JAB", 20_000, Side.LEAD)


def test_no_rounds_means_no_round_numbers(tmp_path):
    meta = META.split("[[rounds]]")[0]
    folder = make_folder(tmp_path, "00:20.000,00:20.180,RED,JAB,,,,LANDED,,\n", meta=meta)
    assert load_annotation(folder).events[0].round_number is None


def test_solo_training_session(tmp_path):
    meta = SOLO_META.replace('slug = "f1"', 'slug = "f1"\nkind = "BAG"')
    folder = make_folder(tmp_path, "00:20.000,00:20.180,RED,JAB,,,,,,\n", meta=meta)
    (event,) = load_annotation(folder).events  # no opponent -> outcome not required
    assert event.outcome is None


def test_empty_events_file_is_valid(tmp_path):
    assert load_annotation(make_folder(tmp_path, "")).events == ()


# --- outcomes, feints and probes ------------------------------------------------------------------


def test_outcome_required_for_punches_in_a_fight(tmp_path):
    rows = [
        "00:20.000,00:20.180,RED,JAB,,,,,,",  # 2 missing outcome
        "00:21.000,00:21.180,RED,JAB,,,,UNKNOWN,,",  # 3 explicitly unknown: fine
        "00:22.000,00:22.180,RED,SLIP,,,LEFT,,,",  # 4 not a punch: no outcome needed
        "00:23.000,00:23.180,RED,FEINT,,,,,,",  # 5 feint: no outcome needed
    ]
    folder = make_folder(tmp_path, "\n".join(rows) + "\n")
    assert problems_of(folder) == [
        "events.csv:2: outcome: required for every JAB in a FIGHT (write UNKNOWN if it cannot be seen)"
    ]


def test_outcome_not_required_from_a_model_source(tmp_path):
    meta = META.replace('name = "tester"', 'name = "detector"\nkind = "MODEL"')
    folder = make_folder(tmp_path, "00:20.000,00:20.180,RED,JAB,,,,,,\n", meta=meta)
    assert load_annotation(folder).events[0].outcome is None


FEINT_HEADER = "start,end,fighter,action,side,target,outcome,commitment\n"


def test_feints_and_probes(tmp_path):
    rows = [
        "00:20.000,00:20.150,RED,FEINT,REAR,BODY,,",
        "00:20.300,00:20.450,RED,FEINT,,,,",  # shoulder/head feint: no hand
        "00:20.600,00:20.780,RED,JAB,,HEAD,MISSED,probe",
        "00:21.000,00:21.250,RED,HOOK,LEAD,HEAD,LANDED,FULL",
    ]
    folder = make_folder(tmp_path, "\n".join(rows) + "\n", header=FEINT_HEADER)
    feint, head_feint, probe, hook = load_annotation(folder).events
    assert (feint.category, feint.side, feint.target) == (Category.FEINT, Side.REAR, Target.BODY)
    assert (head_feint.side, head_feint.target) == (None, None)
    assert (probe.commitment, hook.commitment) == (Commitment.PROBE, Commitment.FULL)


@pytest.mark.parametrize(
    ("row", "fragment"),
    [
        ("FEINT,,,LANDED,", "FEINT is FEINT and takes no outcome"),
        ("FEINT,,,,PROBE", "FEINT is FEINT and takes no commitment"),
        ("SLIP,,,,PROBE", "SLIP is DEFENSE and takes no commitment"),
        ("JAB,,,LANDED,HALF", "commitment: 'HALF' is not one of PROBE/FULL"),
    ],
)
def test_feint_and_probe_rules(tmp_path, row, fragment):
    folder = make_folder(tmp_path, f"00:20.000,00:20.150,RED,{row}\n", header=FEINT_HEADER)
    problems = problems_of(folder)
    assert any(fragment in p for p in problems), problems


@pytest.mark.parametrize(
    ("kind", "solo"), [("FIGHT", True), ("SPARRING", True), ("PADS", False), ("BAG", False)]
)
def test_session_kind_fixes_the_number_of_fighters(tmp_path, kind, solo):
    meta = (SOLO_META if solo else META).replace('slug = "f1"', f'slug = "f1"\nkind = "{kind}"')
    problems = problems_of(make_folder(tmp_path, "", meta=meta))
    assert any(f"a {kind} session has" in p for p in problems), problems


# --- invalid events: every problem reported, with its line --------------------------------------


def test_every_invalid_row_is_reported_with_its_line(tmp_path):
    rows = [
        "00:20.000,00:20.180,RED,JAB,,,,LANDED,,",  # 2 ok
        "00:2x.000,00:20.180,RED,JAB,,,,LANDED,,",  # 3 bad time
        "00:21.000,00:21.180,RED,JAB,REAR,,,LANDED,,",  # 4 jab with rear
        "00:22.000,00:22.180,RED,HOOK,,,,LANDED,,",  # 5 hook without side
        "00:23.000,00:23.180,RED,SPINNING_BACKFIST,,,,,,",  # 6 unknown action
        "00:24.000,00:24.180,RED,SLIP,,HEAD,LEFT,,,",  # 7 slip with target
        "00:25.000,00:24.000,RED,JAB,,,,LANDED,,",  # 8 end before start
        "00:26.000,00:26.180,GREEN,JAB,,,,LANDED,,",  # 9 unknown corner
        "01:05.000,01:05.180,RED,JAB,,,,LANDED,,",  # 10 inside the replay
        "03:30.000,03:30.180,RED,JAB,,,,LANDED,,",  # 11 between rounds
        "00:27.000,00:27.180,RED,JAB,,,,LANDED,2,",  # 12 round column contradicts time
        "00:28.000,00:28.180,RED,JAB,,UPPER,,LANDED,,",  # 13 bad target
        "00:29.000,00:29.180,RED,JAB,,,,LANDED,,,surplus",  # 14 too many cells
        ",00:30.180,RED,JAB,,,,LANDED,,",  # 15 missing start
        "00:31.000,00:31.180,RED,SLIP,,,FORWARD,,,",  # 16 slip direction not allowed
        "00:32.000,00:32.180,RED,JAB,,,,LANDED,0,",  # 17 round not positive
        "00:33.000,00:33.180,RED,FEINT,LEAD,,,LANDED,,",  # 18 feint with an outcome
    ]
    folder = make_folder(tmp_path, "\n".join(rows) + "\n")
    problems = problems_of(folder)
    expected = {
        3: "start: invalid time",
        4: "JAB is always LEAD",
        5: "HOOK requires a side",
        6: "unknown action 'SPINNING_BACKFIST'",
        7: "SLIP is DEFENSE and takes no target",
        8: "end_ms (24000) is before start_ms (25000)",
        9: "fighter: 'GREEN' is not one of RED/BLUE",
        10: "inside an unobserved REPLAY interval",
        11: "outside every round",
        12: "round: column says 2 but the start time lies in round 1",
        13: "target: 'UPPER' is not one of HEAD/BODY",
        14: "more cells than header columns",
        15: "start: missing",
        16: "SLIP direction must be one of LEFT/RIGHT",
        17: "round: '0' is not a positive whole number",
        18: "FEINT is FEINT and takes no outcome",
    }
    for line, fragment in expected.items():
        assert any(p.startswith(f"events.csv:{line}: ") and fragment in p for p in problems), (
            line,
            fragment,
            problems,
        )
    assert not any(p.startswith("events.csv:2:") for p in problems)


def test_corner_missing_from_meta(tmp_path):
    meta = SOLO_META.replace('slug = "f1"', 'slug = "f1"\nkind = "SHADOW"')
    folder = make_folder(tmp_path, "00:20.000,00:20.180,BLUE,JAB,,,,,,\n", meta=meta)
    assert problems_of(folder) == ["events.csv:2: fighter: no fighter in corner BLUE in meta.toml"]


def test_event_beyond_video_duration(tmp_path):
    meta = META.split("[[rounds]]")[0]
    folder = make_folder(tmp_path, "09:59.900,10:00.100,RED,JAB,,,,LANDED,,\n", meta=meta)
    (problem,) = problems_of(folder)
    assert problem.startswith("events.csv:2: ") and "beyond the video duration" in problem


@pytest.mark.parametrize(
    ("header", "fragment"),
    [
        ("start,end,fighter,action,colour\n", "unknown column(s) ['colour']"),
        ("start,end,action\n", "missing column(s) ['fighter']"),
    ],
)
def test_header_problems(tmp_path, header, fragment):
    folder = make_folder(tmp_path, "", header=header)
    (problem,) = problems_of(folder)
    assert problem.startswith("events.csv:1: ") and fragment in problem


# --- invalid meta.toml ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("meta", "fragment"),
    [
        ("schema_version = [", "invalid TOML"),
        (META.replace('corner = "BLUE"', 'corner = "RED"'), "different corners"),
        (META.replace('end = "03:10.000"', 'end = "05:00.000"'), "rounds 1 and 2 overlap"),
        (
            META.replace('start = "00:10.000"', "start = 10.0"),
            "rounds.#1.start: write times as quoted strings",
        ),
        (META.replace("[[unobserved]]", "[[unobservd]]"), "unobservd: Extra inputs"),
        (META.replace("schema_version = 1", "schema_version = 2"), "schema_version"),
        (META.replace('end = "01:10.000"', 'end = "00:50.000"'), "must be after start"),
        (META.replace('duration = "10:00.000"', 'duration = "05:00.000"'), "after the video"),
    ],
)
def test_meta_problems(tmp_path, meta, fragment):
    folder = make_folder(tmp_path, "", meta=meta)
    problems = problems_of(folder)
    assert all(p.startswith("meta.toml: ") for p in problems)
    assert any(fragment in p for p in problems), problems


def test_missing_files(tmp_path):
    assert problems_of(tmp_path) == ["meta.toml: file not found"]
    (tmp_path / "meta.toml").write_text(META, encoding="utf-8")
    assert problems_of(tmp_path) == ["events.csv: file not found"]


# --- writer / round trip -------------------------------------------------------------------------


def test_csv_round_trip_reproduces_identical_events(tmp_path):
    original = load_annotation(SAMPLE)
    copy = tmp_path / "copy"
    copy.mkdir()
    shutil.copy(SAMPLE / "meta.toml", copy / "meta.toml")
    write_events_csv(copy / "events.csv", reversed(original.events), original.meta)

    assert load_annotation(copy).events == original.events
    written = (copy / "events.csv").read_text(encoding="utf-8").splitlines()
    assert written[1] == "00:14.320,00:14.500,RED,JAB,,HEAD,,LANDED,,,"  # implied side left blank
    assert "04:20.000,04:20.180,RED,JAB,,HEAD,,UNKNOWN,PROBE,," in written


def test_writer_refuses_data_the_csv_cannot_hold(tmp_path):
    meta = load_annotation(SAMPLE).meta
    kw = {"fight": meta.fight.slug, "video": meta.video.slug}
    with pytest.raises(ValueError, match="confidence"):
        write_events_csv(tmp_path / "x.csv", [ev(confidence=0.9, **kw)], meta)
    with pytest.raises(ValueError, match="not in"):
        write_events_csv(tmp_path / "x.csv", [ev(fighter="nobody", **kw)], meta)


# --- the synthetic sample reproduces the hand-calculated golden example ---------------------------


def test_sample_folder_reproduces_golden_analytics():
    a = load_annotation(SAMPLE)
    red = select_events(a.events, fighter="fighter-a")
    spec = StreamSpec(tokenizer=by_action_direction)

    assert token_segments(red, spec, a.unobserved) == [
        ["JAB", "JAB", "CROSS", "SLIP_LEFT", "JAB", "CROSS"],  # golden burst (§6)
        ["JAB"],  # the lone jab 30 s later
        ["FEINT", "JAB", "HOOK"],  # round 2: feints are part of the stream
    ]
    assert token_segments(red, StreamSpec(tokenizer=by_action_commitment), a.unobserved)[1:] == [
        ["JAB_PROBE"],
        ["FEINT", "JAB_PROBE", "HOOK"],
    ]
    r1, r2 = round_stats(red, a.rounds, a.unobserved)
    assert (r1.count, r1.observed_ms) == (6, 170_000)  # 180 s round minus a 10 s replay
    assert (r2.count, r2.observed_ms) == (2, 180_000)  # the feint is not a punch
