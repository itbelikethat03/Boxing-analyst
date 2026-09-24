import pytest

from boxing_ai.analytics import action_distribution, round_stats, select_events
from boxing_ai.events import Round, UnobservedInterval
from boxing_ai.ontology import Category
from tests.factories import ev


def replay(start, end, video="video-1"):
    return UnobservedInterval(video=video, start_ms=start, end_ms=end, kind="REPLAY")


# --- distribution ------------------------------------------------------------------------------------


def distribution_events():
    # 3 JAB, 2 CROSS, 1 HOOK, 1 UPPERCUT (7 punches) + 1 SLIP
    actions = ["JAB", "JAB", "JAB", "CROSS", "CROSS", "HOOK", "UPPERCUT", "SLIP"]
    return [ev(a, i * 1000) for i, a in enumerate(actions)]


def test_punch_distribution_by_hand():
    got = action_distribution(distribution_events())
    assert [(a.action_type, a.count) for a in got] == [
        ("JAB", 3),
        ("CROSS", 2),
        ("HOOK", 1),  # 1-count tie broken alphabetically
        ("UPPERCUT", 1),
    ]
    assert got[0].share == pytest.approx(3 / 7)
    assert got[1].share == pytest.approx(2 / 7)
    assert sum(a.share for a in got) == pytest.approx(1.0)


def test_distribution_can_include_defense():
    got = action_distribution(
        distribution_events(), categories=frozenset({Category.PUNCH, Category.DEFENSE})
    )
    assert [(a.action_type, a.count) for a in got] == [
        ("JAB", 3),
        ("CROSS", 2),
        ("HOOK", 1),
        ("SLIP", 1),
        ("UPPERCUT", 1),
    ]
    assert got[0].share == pytest.approx(3 / 8)


def test_distribution_of_nothing_is_empty():
    assert action_distribution([]) == []


# --- per-round rates -----------------------------------------------------------------------------------


ROUNDS = [
    Round(video="video-1", number=1, start_ms=0, end_ms=180_000),
    Round(video="video-1", number=2, start_ms=180_000, end_ms=360_000),
    Round(video="video-1", number=3, start_ms=360_000, end_ms=420_000),
]


def test_round_stats_use_observed_minutes():
    events = [ev("JAB", 1_000 * i, round_number=1) for i in range(5)]  # 5 punches in round 1
    events += [ev("CROSS", 200_000, round_number=2)]  # 1 punch in round 2
    events += [ev("HOOK", 250_000)]  # no round: not attributed
    events += [ev("SLIP", 2_000, round_number=1)]  # not a punch
    stats = round_stats(events, ROUNDS, [replay(60_000, 90_000), replay(360_000, 420_000)])

    r1, r2, r3 = stats
    # round 1: 180 s minus a 30 s replay = 150 s = 2.5 min observed -> 5 / 2.5 = 2 punches per minute
    assert (r1.round_number, r1.count, r1.observed_ms) == (1, 5, 150_000)
    assert r1.per_observed_minute == pytest.approx(2.0)
    # round 2: full 3 minutes observed -> 1 / 3
    assert (r2.round_number, r2.count, r2.observed_ms) == (2, 1, 180_000)
    assert r2.per_observed_minute == pytest.approx(1 / 3)
    # round 3 is entirely a replay: the rate is undefined, not zero
    assert (r3.count, r3.observed_ms) == (0, 0)
    assert r3.per_observed_minute is None


def test_round_stats_ignore_other_videos_events_and_gaps():
    events = [ev("JAB", 1_000, round_number=1, video="video-2")]
    stats = round_stats(events, ROUNDS[:1], [replay(0, 90_000, video="video-2")])
    assert (stats[0].count, stats[0].observed_ms) == (0, 180_000)


def test_round_stats_are_ordered_by_video_then_round():
    rounds = [
        Round(video="b", number=1, start_ms=0, end_ms=10),
        Round(video="a", number=2, start_ms=10, end_ms=20),
        Round(video="a", number=1, start_ms=0, end_ms=10),
    ]
    assert [(s.video, s.round_number) for s in round_stats([], rounds)] == [
        ("a", 1),
        ("a", 2),
        ("b", 1),
    ]


# --- select_events -------------------------------------------------------------------------------------


def test_select_events_filters_and_orders_chronologically():
    events = [
        ev("CROSS", 2_000, fighter="fighter-b", round_number=1),
        ev("JAB", 1_000, fighter="fighter-a", round_number=1),
        ev("HOOK", 3_000, fighter="fighter-a", round_number=2),
        ev("JAB", 500, fighter="fighter-a", fight="fight-2", video="video-2"),
    ]
    # Order leads with the video: video-1's JAB(1000) and HOOK(3000) come before video-2's JAB(500).
    got = select_events(events, fighter="fighter-a")
    assert [(e.video, e.start_ms, e.action_type) for e in got] == [
        ("video-1", 1_000, "JAB"),
        ("video-1", 3_000, "HOOK"),
        ("video-2", 500, "JAB"),
    ]
    got = select_events(events, fighter="fighter-a", fight="fight-1")
    assert [e.action_type for e in got] == ["JAB", "HOOK"]
    assert [e.action_type for e in select_events(events, round_number=1)] == ["JAB", "CROSS"]
    assert [e.action_type for e in select_events(events, video="video-2")] == ["JAB"]


def test_select_events_time_range_is_half_open_on_start():
    events = [ev("JAB", t) for t in (1_000, 2_000, 3_000)]
    got = select_events(events, start_ms=1_000, end_ms=3_000)
    assert [e.start_ms for e in got] == [1_000, 2_000]
