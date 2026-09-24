import itertools

from boxing_ai.events import chronological
from tests.factories import ev


def test_out_of_order_input_is_sorted_chronologically():
    events = [ev("CROSS", 500), ev("JAB", 100), ev("HOOK", 900)]
    assert [e.action_type for e in chronological(events)] == ["JAB", "CROSS", "HOOK"]


def test_input_is_not_mutated():
    events = [ev("CROSS", 500), ev("JAB", 100)]
    chronological(events)
    assert [e.action_type for e in events] == ["CROSS", "JAB"]


def test_ties_on_start_are_broken_by_end():
    long_jab = ev("JAB", 100, end_ms=400)
    short_cross = ev("CROSS", 100, end_ms=200)
    assert chronological([long_jab, short_cross]) == [short_cross, long_jab]


def test_overlapping_events_order_by_start():
    # The cross starts before the jab has finished retracting.
    jab = ev("JAB", 0, end_ms=300)
    cross = ev("CROSS", 200, end_ms=450)
    assert chronological([cross, jab]) == [jab, cross]


def test_persisted_events_precede_unpersisted_at_a_tie_then_order_by_id():
    unsaved = ev("JAB", 100)
    saved_2 = ev("JAB", 100, id=2)
    saved_1 = ev("JAB", 100, id=1)
    assert chronological([unsaved, saved_2, saved_1]) == [saved_1, saved_2, unsaved]


def test_order_is_independent_of_input_order_even_for_full_ties():
    # Same times, unpersisted, different fighters/actions: input order must not matter.
    events = [
        ev("JAB", 100, fighter="fighter-b"),
        ev("CROSS", 100, fighter="fighter-a"),
        ev("HOOK", 100, fighter="fighter-a"),
    ]
    results = {tuple(map(id, chronological(list(p)))) for p in itertools.permutations(events)}
    assert len(results) == 1


def test_videos_never_interleave():
    a1, a2 = ev("JAB", 5000, video="v-a"), ev("CROSS", 6000, video="v-a")
    b1 = ev("HOOK", 100, video="v-b")
    assert chronological([b1, a2, a1]) == [a1, a2, b1]


def test_empty_input():
    assert chronological([]) == []
