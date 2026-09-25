"""Report composition (pure part) and the default source rule."""

from pathlib import Path

import pytest

pytest.importorskip("psycopg")  # reporting imports the database layer

from boxing_ai.annotations import load_annotation  # noqa: E402
from boxing_ai.database import SourceInfo  # noqa: E402
from boxing_ai.reporting import AmbiguousSource, build_report, select_sources  # noqa: E402

SAMPLE = Path(__file__).resolve().parents[2] / "data" / "annotations" / "sample-synthetic"


def src(id, video="v1", kind="HUMAN", name="ann", version="r1"):
    return SourceInfo(id, "f1", video, kind, name, version, "0.2", 0)


def test_report_numbers_by_hand():
    a = load_annotation(SAMPLE)
    r = build_report("fighter-a", a.events, a.rounds, a.unobserved)
    assert (r.events, r.punches, r.observed_ms) == (10, 8, 350_000)  # 170 s + 180 s observed
    assert r.punches_per_observed_minute == pytest.approx(8 / (350 / 60))
    assert [(x.action_type, x.count) for x in r.actions] == [("JAB", 5), ("CROSS", 2), ("HOOK", 1)]
    assert [(s.round_number, s.count) for s in r.rounds] == [(1, 6), (2, 2)]
    assert sorted(r.combos) == [2, 3]
    top = r.combos[2][0]
    assert (top.ngram, top.count, top.landed_rate) == (("JAB", "CROSS"), 2, 0.5)
    assert [(e.ngram, e.count) for e in r.entries] == [(("JAB",), 2), (("FEINT",), 1)]


def test_report_ignores_other_fighters_events():
    a = load_annotation(SAMPLE)
    r = build_report("fighter-b", a.events, a.rounds, a.unobserved)
    assert (r.events, r.punches) == (4, 4)


def test_single_human_source_per_video_is_chosen():
    chosen = select_sources([src(1), src(2, kind="MODEL", name="m"), src(3, video="v2")])
    assert [s.id for s in chosen] == [1, 3]


def test_model_source_is_used_when_there_is_no_human_one():
    assert [s.id for s in select_sources([src(5, kind="MODEL", name="m")])] == [5]


def test_several_human_sources_are_ambiguous_until_one_is_named():
    options = [src(1, name="alice"), src(2, name="bob"), src(3, name="bob", version="r2")]
    with pytest.raises(AmbiguousSource, match="alice@r1.*bob@r1"):
        select_sources(options)
    assert [s.id for s in select_sources(options, "alice")] == [1]
    assert [s.id for s in select_sources(options, "bob@r2")] == [3]
    with pytest.raises(AmbiguousSource):
        select_sources(options, "bob")
