import pytest

from boxing_ai.sequences import (
    END,
    START,
    Transition,
    followers,
    transition_counts,
    transitions,
)
from tests.fixtures.golden import GOLDEN_TOKENS, TRANSITIONS_ORDER_1, TRANSITIONS_ORDER_2


def as_plain(table):
    return {ctx: dict(counter) for ctx, counter in table.items()}


def test_order_1_counts_match_the_hand_calculation():
    assert as_plain(transition_counts([GOLDEN_TOKENS])) == TRANSITIONS_ORDER_1


def test_order_2_counts_match_the_hand_calculation():
    assert as_plain(transition_counts([GOLDEN_TOKENS], order=2)) == TRANSITIONS_ORDER_2


def test_hand_calculated_probabilities():
    rows = {(t.context, t.next): t for t in transitions(transition_counts([GOLDEN_TOKENS]))}
    assert rows[((START,), "JAB")].probability == 1.0
    assert rows[(("JAB",), "CROSS")].probability == pytest.approx(2 / 3)
    assert rows[(("JAB",), "JAB")].probability == pytest.approx(1 / 3)
    assert rows[(("CROSS",), "SLIP_LEFT")].probability == pytest.approx(1 / 2)
    assert rows[(("CROSS",), END)].probability == pytest.approx(1 / 2)
    assert rows[(("SLIP_LEFT",), "JAB")].probability == 1.0


def test_count_and_total_are_exposed_for_small_sample_honesty():
    rows = {(t.context, t.next): t for t in transitions(transition_counts([GOLDEN_TOKENS]))}
    t = rows[(("JAB",), "CROSS")]
    assert (t.count, t.total) == (2, 3)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_probabilities_sum_to_one_for_every_context(order):
    segments = [GOLDEN_TOKENS, ["JAB"], ["CROSS", "HOOK", "JAB", "CROSS"]]
    sums: dict = {}
    for t in transitions(transition_counts(segments, order)):
        sums[t.context] = sums.get(t.context, 0.0) + t.probability
    assert sums and all(s == pytest.approx(1.0) for s in sums.values())


def test_end_sentinel_is_what_keeps_a_burst_final_token_in_the_denominator():
    # Without END, the CROSS that ended the burst would vanish and P(SLIP_LEFT | CROSS) would read 1/1.
    row = {t.next: t for t in followers(transition_counts([GOLDEN_TOKENS]), "CROSS")}
    assert row["SLIP_LEFT"].probability == pytest.approx(0.5)
    assert row[END].count == 1


def test_lone_token_burst_contributes_start_and_end_transitions():
    table = transition_counts([["JAB"]])
    assert as_plain(table) == {(START,): {"JAB": 1}, ("JAB",): {END: 1}}


def test_empty_segments_are_skipped():
    assert transition_counts([[], ["JAB"], []]) == transition_counts([["JAB"]])


def test_transitions_are_ordered_deterministically():
    rows = transitions(transition_counts([GOLDEN_TOKENS]))
    assert [(t.context, t.next) for t in rows] == [
        ((START,), "JAB"),
        (("CROSS",), END),  # count tie 1/1 -> token order; '<' sorts before 'S'
        (("CROSS",), "SLIP_LEFT"),
        (("JAB",), "CROSS"),  # 2 before 1
        (("JAB",), "JAB"),
        (("SLIP_LEFT",), "JAB"),
    ]


def test_min_count_hides_rows_but_keeps_true_totals():
    rows = transitions(transition_counts([GOLDEN_TOKENS]), min_count=2)
    assert rows == [Transition(("JAB",), "CROSS", 2, 3)]  # total still 3, so p is still 2/3


def test_followers_of_a_token_and_of_a_context_tuple():
    table1 = transition_counts([GOLDEN_TOKENS])
    assert [(t.next, t.count) for t in followers(table1, "JAB")] == [("CROSS", 2), ("JAB", 1)]
    table2 = transition_counts([GOLDEN_TOKENS], order=2)
    assert [(t.next, t.count) for t in followers(table2, ("JAB", "JAB"))] == [("CROSS", 1)]


def test_followers_of_an_unseen_context_is_empty():
    assert followers(transition_counts([GOLDEN_TOKENS]), "HOOK") == []


def test_order_must_be_positive():
    with pytest.raises(ValueError):
        transition_counts([GOLDEN_TOKENS], order=0)
