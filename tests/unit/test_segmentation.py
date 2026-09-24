import itertools

import pytest

from boxing_ai.events import UnobservedInterval
from boxing_ai.ontology import Category
from boxing_ai.sequences import StreamSpec, segment, token_segments
from tests.factories import ev


def actions(bursts):
    return [[e.action_type for e in burst] for burst in bursts]


def gap_iv(start, end, video="video-1", kind="REPLAY"):
    return UnobservedInterval(video=video, start_ms=start, end_ms=end, kind=kind)


# --- gap threshold ------------------------------------------------------------------------------------


def test_one_burst_when_all_gaps_are_small():
    events = [ev("JAB", 0), ev("CROSS", 400), ev("HOOK", 800)]
    assert actions(segment(events)) == [["JAB", "CROSS", "HOOK"]]


def test_gap_equal_to_threshold_stays_together_but_one_ms_more_splits():
    # JAB ends at 180; gap = start - 180.
    at_threshold = [ev("JAB", 0), ev("CROSS", 180 + 1000)]
    just_over = [ev("JAB", 0), ev("CROSS", 180 + 1001)]
    assert actions(segment(at_threshold)) == [["JAB", "CROSS"]]
    assert actions(segment(just_over)) == [["JAB"], ["CROSS"]]


def test_gap_is_measured_from_the_previous_end_not_the_previous_start():
    # start-to-start is 1400 ms, but the jab lasted until 500 so the silence is only 900 ms.
    events = [ev("JAB", 0, end_ms=500), ev("CROSS", 1400)]
    assert actions(segment(events)) == [["JAB", "CROSS"]]


def test_overlapping_events_have_zero_gap():
    events = [ev("JAB", 0, end_ms=300), ev("CROSS", 200, end_ms=450)]
    assert actions(segment(events)) == [["JAB", "CROSS"]]


def test_silence_is_measured_from_the_burst_latest_end_not_just_the_last_event():
    # A long block spans 0..5000; the jab inside it ends at 280. The cross at 5900 follows only 900 ms after
    # the block ends, so the burst is unbroken even though 5620 ms passed since the jab ended.
    events = [ev("BLOCK", 0, end_ms=5000), ev("JAB", 100), ev("CROSS", 5900)]
    assert actions(segment(events)) == [["BLOCK", "JAB", "CROSS"]]


def test_custom_gap_threshold():
    events = [ev("JAB", 0), ev("CROSS", 700)]  # 520 ms of silence
    assert len(segment(events, StreamSpec(gap_ms=500))) == 2
    assert len(segment(events, StreamSpec(gap_ms=520))) == 1


def test_a_long_pause_does_not_create_a_pair_across_it():
    events = [ev("JAB", 0), ev("CROSS", 400), ev("JAB", 30_000)]
    assert actions(segment(events)) == [["JAB", "CROSS"], ["JAB"]]


# --- category filtering happens before segmentation ---------------------------------------------------


def test_default_stream_excludes_footwork_and_measures_gaps_between_remaining_events():
    events = [ev("JAB", 0), ev("STEP", 900), ev("CROSS", 1900)]
    # STEP dropped -> the jab-to-cross silence is 1720 ms > 1000 -> two bursts.
    assert actions(segment(events)) == [["JAB"], ["CROSS"]]
    everything = StreamSpec(categories=frozenset(Category))
    # STEP kept -> 720 ms and 820 ms silences -> one burst.
    assert actions(segment(events, everything)) == [["JAB", "STEP", "CROSS"]]


def test_punch_only_stream_drops_defensive_events():
    events = [ev("JAB", 0), ev("SLIP", 300), ev("CROSS", 600)]
    punches_only = StreamSpec(categories=frozenset({Category.PUNCH}))
    assert actions(segment(events, punches_only)) == [["JAB", "CROSS"]]
    assert actions(segment(events)) == [["JAB", "SLIP", "CROSS"]]  # default keeps DEFENSE


# --- hard breaks --------------------------------------------------------------------------------------


def test_unobserved_interval_between_events_breaks_the_burst_even_with_a_small_gap():
    events = [ev("JAB", 0), ev("CROSS", 800)]  # only 620 ms apart
    assert actions(segment(events)) == [["JAB", "CROSS"]]
    assert actions(segment(events, unobserved=[gap_iv(300, 600)])) == [["JAB"], ["CROSS"]]


def test_unobserved_interval_exactly_filling_the_gap_still_breaks():
    events = [ev("JAB", 0), ev("CROSS", 800)]
    assert len(segment(events, unobserved=[gap_iv(180, 800)])) == 2


def test_unrelated_unobserved_intervals_do_not_break_bursts():
    events = [ev("JAB", 0), ev("CROSS", 800)]
    elsewhere = [gap_iv(5_000, 6_000), gap_iv(300, 600, video="another-video")]
    assert len(segment(events, unobserved=elsewhere)) == 1


def test_round_change_breaks_the_burst():
    events = [ev("JAB", 1000, round_number=1), ev("CROSS", 1200, round_number=2)]
    assert actions(segment(events)) == [["JAB"], ["CROSS"]]


def test_round_unset_versus_set_breaks_the_burst():
    events = [ev("JAB", 1000), ev("CROSS", 1200, round_number=1)]
    assert len(segment(events)) == 2


def test_video_change_breaks_the_burst_and_videos_never_interleave():
    events = [
        ev("JAB", 100, video="v-b"),
        ev("CROSS", 200, video="v-a"),
        ev("HOOK", 300, video="v-a"),
    ]
    assert actions(segment(events)) == [["CROSS", "HOOK"], ["JAB"]]


# --- shape and determinism ----------------------------------------------------------------------------


def test_empty_input_gives_no_bursts():
    assert segment([]) == []


def test_unsorted_input_is_ordered_first():
    events = [ev("CROSS", 400), ev("HOOK", 800), ev("JAB", 0)]
    assert actions(segment(events)) == [["JAB", "CROSS", "HOOK"]]


def test_result_does_not_depend_on_input_order():
    events = [
        ev("JAB", 0),
        ev("CROSS", 300),
        ev("HOOK", 300, fighter="fighter-b"),
        ev("JAB", 9_000),
    ]
    results = {
        tuple(tuple((e.fighter, e.action_type) for e in b) for b in segment(list(p)))
        for p in itertools.permutations(events)
    }
    assert len(results) == 1


def test_token_segments_applies_the_tokenizer():
    from boxing_ai.sequences import by_action_direction

    events = [ev("JAB", 0), ev("SLIP", 300, direction="RIGHT"), ev("CROSS", 600)]
    spec = StreamSpec(tokenizer=by_action_direction)
    assert token_segments(events, spec) == [["JAB", "SLIP_RIGHT", "CROSS"]]


@pytest.mark.parametrize("kwargs", [{"gap_ms": -1}, {"categories": frozenset()}])
def test_stream_spec_rejects_nonsense(kwargs):
    with pytest.raises(ValueError):
        StreamSpec(**kwargs)
