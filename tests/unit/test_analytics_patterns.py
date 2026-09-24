"""Sequence analytics checked against hand-calculated values (see tests/fixtures/golden.py)."""

import itertools

import pytest

from boxing_ai.analytics import (
    NgramCount,
    entry_patterns,
    exit_patterns,
    next_after,
    top_ngrams,
    transition_probabilities,
)
from boxing_ai.events import UnobservedInterval
from boxing_ai.ontology import Category
from boxing_ai.sequences import (
    END,
    START,
    StreamSpec,
    actor_tagged,
    by_action,
    by_action_direction,
)
from tests.factories import ev
from tests.fixtures.golden import BIGRAMS, TRIGRAMS, UNIGRAMS, golden_events

SPEC = StreamSpec(tokenizer=by_action_direction)


def as_dict(counts: list[NgramCount]) -> dict:
    return {c.ngram: c.count for c in counts}


# --- frequency ---------------------------------------------------------------------------------------


def test_golden_unigram_bigram_trigram_counts():
    events = golden_events()
    assert as_dict(top_ngrams(events, 1, SPEC)) == UNIGRAMS
    assert as_dict(top_ngrams(events, 2, SPEC)) == BIGRAMS
    assert as_dict(top_ngrams(events, 3, SPEC)) == TRIGRAMS


def test_golden_ngram_totals_are_length_minus_n_plus_one():
    events = golden_events()
    assert sum(c.count for c in top_ngrams(events, 2, SPEC)) == 5
    assert sum(c.count for c in top_ngrams(events, 3, SPEC)) == 4


def test_top_ngrams_ranking_is_deterministic():
    assert top_ngrams(golden_events(), 2, SPEC) == [
        NgramCount(("JAB", "CROSS"), 2),
        NgramCount(("CROSS", "SLIP_LEFT"), 1),
        NgramCount(("JAB", "JAB"), 1),
        NgramCount(("SLIP_LEFT", "JAB"), 1),
    ]


def test_top_ngrams_min_count_and_limit():
    events = golden_events()
    assert top_ngrams(events, 2, SPEC, min_count=2) == [NgramCount(("JAB", "CROSS"), 2)]
    assert len(top_ngrams(events, 2, SPEC, limit=2)) == 2


def test_default_tokenizer_drops_the_direction():
    ngrams_ = as_dict(top_ngrams(golden_events(), 2, StreamSpec(tokenizer=by_action)))
    assert ngrams_[("CROSS", "SLIP")] == 1
    assert ("CROSS", "SLIP_LEFT") not in ngrams_


def test_brief_example_jab_cross_hook():
    events = [ev("JAB", 0), ev("CROSS", 300), ev("HOOK", 600)]
    assert as_dict(top_ngrams(events, 2)) == {("JAB", "CROSS"): 1, ("CROSS", "HOOK"): 1}
    assert as_dict(top_ngrams(events, 3)) == {("JAB", "CROSS", "HOOK"): 1}


def test_a_long_pause_creates_no_ngram_across_it():
    events = golden_events(with_lone_jab=True)
    bigrams = as_dict(top_ngrams(events, 2, SPEC))
    assert bigrams == BIGRAMS  # unchanged by the extra jab
    assert ("CROSS", "JAB") not in bigrams


# --- conditional probability ------------------------------------------------------------------------


def probability(rows, context, nxt):
    (row,) = [t for t in rows if t.context == context and t.next == nxt]
    return row


def test_golden_conditional_probabilities():
    rows = transition_probabilities(golden_events(), SPEC)
    assert probability(rows, (START,), "JAB").probability == 1.0
    assert probability(rows, ("JAB",), "JAB").probability == pytest.approx(1 / 3)
    assert probability(rows, ("JAB",), "CROSS").probability == pytest.approx(2 / 3)
    assert probability(rows, ("CROSS",), "SLIP_LEFT").probability == pytest.approx(1 / 2)
    assert probability(rows, ("CROSS",), END).probability == pytest.approx(1 / 2)
    assert probability(rows, ("SLIP_LEFT",), "JAB").probability == 1.0


def test_lone_jab_burst_changes_the_conditionals_exactly_as_hand_calculated():
    rows = transition_probabilities(golden_events(with_lone_jab=True), SPEC)
    assert probability(rows, (START,), "JAB").probability == 1.0  # both bursts open with a jab
    assert probability(rows, (START,), "JAB").total == 2
    cross = probability(rows, ("JAB",), "CROSS")
    assert (cross.count, cross.total) == (2, 4)  # JAB now followed by JAB, CROSS, CROSS, END
    assert cross.probability == pytest.approx(1 / 2)
    assert probability(rows, ("JAB",), END).count == 1


def test_next_after_lists_followers_most_common_first():
    got = next_after(golden_events(), "JAB", SPEC)
    assert [(t.next, t.count, t.total) for t in got] == [("CROSS", 2, 3), ("JAB", 1, 3)]


def test_next_after_with_a_two_token_context():
    got = next_after(golden_events(), ("JAB", "CROSS"), SPEC)
    assert sorted((t.next, t.count) for t in got) == [(END, 1), ("SLIP_LEFT", 1)]


def test_next_after_an_action_that_never_occurs_is_empty():
    assert next_after(golden_events(), "UPPERCUT", SPEC) == []


# --- entry / exit patterns --------------------------------------------------------------------------


def test_entry_and_exit_patterns_of_the_single_burst():
    events = golden_events()
    assert entry_patterns(events, 1, SPEC) == [NgramCount(("JAB",), 1)]
    assert entry_patterns(events, 2, SPEC) == [NgramCount(("JAB", "JAB"), 1)]
    assert exit_patterns(events, 1, SPEC) == [NgramCount(("CROSS",), 1)]
    assert exit_patterns(events, 2, SPEC) == [NgramCount(("JAB", "CROSS"), 1)]


def test_entry_and_exit_patterns_across_two_bursts():
    events = golden_events(with_lone_jab=True)
    assert entry_patterns(events, 1, SPEC) == [NgramCount(("JAB",), 2)]
    # exits: burst 1 ends on CROSS, burst 2 on JAB -> tie, broken alphabetically
    assert exit_patterns(events, 1, SPEC) == [NgramCount(("CROSS",), 1), NgramCount(("JAB",), 1)]
    # bursts shorter than k are not counted: the lone JAB has no length-2 exit
    assert exit_patterns(events, 2, SPEC) == [NgramCount(("JAB", "CROSS"), 1)]


def test_edge_patterns_reject_k_below_one():
    with pytest.raises(ValueError):
        entry_patterns(golden_events(), 0)
    with pytest.raises(ValueError):
        exit_patterns(golden_events(), 0)


# --- context and breaks -----------------------------------------------------------------------------


def test_unobserved_replay_prevents_a_fake_adjacency():
    # jab .. (replay of an earlier exchange) .. cross: only 620 ms apart, but the replay sits between them.
    events = [ev("JAB", 0), ev("CROSS", 800)]
    replay = UnobservedInterval(video="video-1", start_ms=300, end_ms=600, kind="REPLAY")
    assert as_dict(top_ngrams(events, 2)) == {("JAB", "CROSS"): 1}
    assert top_ngrams(events, 2, unobserved=[replay]) == []


def test_movement_is_excluded_by_default_and_included_on_request():
    events = [ev("JAB", 0), ev("STEP", 300, direction="BACK"), ev("CROSS", 600)]
    assert as_dict(top_ngrams(events, 2)) == {("JAB", "CROSS"): 1}
    with_steps = StreamSpec(categories=frozenset(Category), tokenizer=by_action_direction)
    assert as_dict(top_ngrams(events, 2, with_steps)) == {
        ("JAB", "STEP_BACK"): 1,
        ("STEP_BACK", "CROSS"): 1,
    }


# --- reactions: a two-fighter stream with actor-tagged tokens ---------------------------------------


def exchange_events():
    """Opponent (fighter-b) jabs three times; fighter-a reacts differently each time. Gaps between the
    exchanges (~9 s) exceed the exchange threshold, so each is its own burst."""
    return [
        ev("JAB", 0, fighter="fighter-b"),
        ev("SLIP", 250, fighter="fighter-a", direction="LEFT"),
        ev("CROSS", 600, fighter="fighter-a"),
        ev("JAB", 10_000, fighter="fighter-b"),
        ev("BLOCK", 10_250, fighter="fighter-a"),
        ev("JAB", 20_000, fighter="fighter-b"),
        ev("SLIP", 20_250, fighter="fighter-a", direction="RIGHT"),
    ]


def test_defensive_reactions_to_the_opponents_jab():
    spec = StreamSpec(tokenizer=actor_tagged(by_action, self_fighter="fighter-a"), gap_ms=2_500)
    got = next_after(exchange_events(), "OPP:JAB", spec)
    assert [(t.next, t.count, t.total) for t in got] == [
        ("SELF:SLIP", 2, 3),
        ("SELF:BLOCK", 1, 3),
    ]


def test_exchange_entry_patterns_show_who_starts_and_with_what():
    spec = StreamSpec(tokenizer=actor_tagged(by_action, self_fighter="fighter-a"), gap_ms=2_500)
    # Exchanges open (OPP:JAB, SELF:SLIP), (OPP:JAB, SELF:BLOCK), (OPP:JAB, SELF:SLIP).
    assert entry_patterns(exchange_events(), 2, spec) == [
        NgramCount(("OPP:JAB", "SELF:SLIP"), 2),
        NgramCount(("OPP:JAB", "SELF:BLOCK"), 1),
    ]


# --- determinism ------------------------------------------------------------------------------------


def test_results_do_not_depend_on_input_order():
    events = golden_events(with_lone_jab=True)
    baseline = top_ngrams(events, 2, SPEC)
    for perm in itertools.islice(itertools.permutations(events), 50):
        assert top_ngrams(list(perm), 2, SPEC) == baseline
