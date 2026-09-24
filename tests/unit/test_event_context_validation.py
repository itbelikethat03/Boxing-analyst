from boxing_ai.events import EventContext, Round, UnobservedInterval, check_events
from tests.factories import ev

CTX = EventContext(
    fight="fight-1",
    video="video-1",
    fighters=frozenset({"fighter-a", "fighter-b"}),
    video_duration_ms=600_000,
    unobserved=(
        UnobservedInterval(video="video-1", start_ms=10_000, end_ms=15_000, kind="REPLAY"),
    ),
    rounds=(Round(video="video-1", number=1, start_ms=0, end_ms=180_000),),
)


def messages(events, ctx=CTX):
    return [p.message for p in check_events(events, ctx)]


def test_valid_events_produce_no_problems():
    events = [ev("JAB", 1_000), ev("CROSS", 1_300, fighter="fighter-b", round_number=1)]
    assert check_events(events, CTX) == []


def test_fighter_must_be_in_the_fight():
    (msg,) = messages([ev("JAB", 1_000, fighter="ringside-guy")])
    assert "'ringside-guy' is not in this fight" in msg


def test_end_must_be_within_video_duration():
    (msg,) = messages([ev("JAB", 599_950)])
    assert "beyond the video duration 600000" in msg


def test_no_duration_check_when_duration_is_unknown():
    ctx = EventContext(fight="fight-1", video="video-1", fighters=frozenset({"fighter-a"}))
    assert check_events([ev("JAB", 9_999_999)], ctx) == []


def test_event_inside_unobserved_interval_is_rejected():
    (msg,) = messages([ev("JAB", 12_000)])
    assert "unobserved REPLAY interval [10000, 15000)" in msg


def test_event_partially_overlapping_unobserved_interval_is_rejected():
    assert len(messages([ev("JAB", 9_900)])) == 1  # 9900..10080 crosses into the replay


def test_event_touching_but_not_overlapping_unobserved_interval_is_fine():
    assert messages([ev("JAB", 9_820)]) == []  # ends exactly at 10000
    assert messages([ev("JAB", 15_000)]) == []  # starts exactly at the exclusive end


def test_round_must_exist_for_the_video():
    (msg,) = messages([ev("JAB", 1_000, round_number=2)])
    assert "round 2 is not defined" in msg


def test_fight_and_video_must_match_the_context():
    got = messages([ev("JAB", 1_000, fight="other-fight", video="other-video")])
    assert any("fight 'other-fight'" in m for m in got)
    assert any("video 'other-video'" in m for m in got)


def test_all_problems_are_collected_with_their_index():
    events = [
        ev("JAB", 1_000),  # ok
        ev("JAB", 12_000, fighter="nobody"),  # 2 problems
        ev("CROSS", 2_000, round_number=9),  # 1 problem
    ]
    problems = check_events(events, CTX)
    assert [p.index for p in problems] == [1, 1, 2]
