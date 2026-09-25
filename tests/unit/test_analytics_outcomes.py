"""Outcome statistics, checked by hand against the synthetic sample (data/annotations/sample-synthetic).

fighter-a (RED), punches with outcomes:
    JAB   x5: LANDED, MISSED, BLOCKED, MISSED, UNKNOWN   -> known 4, landed 1
    CROSS x2: BLOCKED, LANDED                            -> known 2, landed 1
    HOOK  x1: LANDED                                     -> known 1, landed 1
Bursts (default stream, 1 s gap): [JAB JAB CROSS SLIP JAB CROSS]  [JAB]  [FEINT JAB HOOK]
"""

from pathlib import Path

import pytest

from boxing_ai.analytics import (
    OutcomeStat,
    PatternOutcome,
    outcome_breakdown,
    pattern_outcomes,
    select_events,
)
from boxing_ai.annotations import load_annotation
from tests.factories import ev

SAMPLE = Path(__file__).resolve().parents[2] / "data" / "annotations" / "sample-synthetic"


@pytest.fixture(scope="module")
def red():
    a = load_annotation(SAMPLE)
    return select_events(a.events, fighter="fighter-a"), a.unobserved


def test_outcome_breakdown_by_hand(red):
    events, _ = red
    stats = outcome_breakdown(events)
    assert stats == [
        OutcomeStat("JAB", thrown=5, landed=1, blocked=1, missed=2),
        OutcomeStat("CROSS", thrown=2, landed=1, blocked=1, missed=0),
        OutcomeStat("HOOK", thrown=1, landed=1, blocked=0, missed=0),
    ]
    jab = stats[0]
    # The UNKNOWN jab is not in the denominator.
    assert jab.unknown == 1 and jab.landed_rate == 1 / 4


def test_outcome_breakdown_ignores_non_punches_and_handles_all_unknown():
    stats = outcome_breakdown([ev("FEINT"), ev("SLIP"), ev("JAB")])
    assert stats == [OutcomeStat("JAB", thrown=1, landed=0, blocked=0, missed=0)]
    assert stats[0].landed_rate is None


def test_bigram_outcomes_by_hand(red):
    events, unobserved = red
    assert pattern_outcomes(events, 2, unobserved=unobserved) == [
        PatternOutcome(("JAB", "CROSS"), count=2, known=2, landed=1),  # blocked once, landed once
        PatternOutcome(("CROSS", "SLIP"), count=1, known=0, landed=0),  # ends in a defence
        PatternOutcome(("FEINT", "JAB"), count=1, known=0, landed=0),  # jab outcome UNKNOWN
        PatternOutcome(("JAB", "HOOK"), count=1, known=1, landed=1),
        PatternOutcome(("JAB", "JAB"), count=1, known=1, landed=0),
        PatternOutcome(("SLIP", "JAB"), count=1, known=1, landed=0),
    ]


def test_trigram_outcomes_by_hand(red):
    events, unobserved = red
    by_gram = {p.ngram: p for p in pattern_outcomes(events, 3, unobserved=unobserved)}
    assert by_gram[("FEINT", "JAB", "HOOK")].landed_rate == 1.0
    assert by_gram[("SLIP", "JAB", "CROSS")].landed_rate == 1.0
    assert by_gram[("JAB", "JAB", "CROSS")].landed_rate == 0.0
    assert by_gram[("JAB", "CROSS", "SLIP")].landed_rate is None
    assert len(by_gram) == 5


def test_pattern_counts_agree_with_top_ngrams(red):
    from boxing_ai.analytics import top_ngrams

    events, unobserved = red
    for n in (1, 2, 3):
        expected = [(c.ngram, c.count) for c in top_ngrams(events, n, unobserved=unobserved)]
        got = [(p.ngram, p.count) for p in pattern_outcomes(events, n, unobserved=unobserved)]
        assert got == expected


def test_pattern_outcomes_limit_min_count_and_bad_n(red):
    events, unobserved = red
    assert [p.ngram for p in pattern_outcomes(events, 2, min_count=2, unobserved=unobserved)] == [
        ("JAB", "CROSS")
    ]
    assert len(pattern_outcomes(events, 2, limit=3, unobserved=unobserved)) == 3
    with pytest.raises(ValueError):
        pattern_outcomes(events, 0)


def test_pattern_occurrences_point_at_the_events(red):
    from boxing_ai.analytics import pattern_occurrences

    events, unobserved = red
    occ = pattern_occurrences(events, ["JAB", "CROSS"], unobserved=unobserved)
    assert [(o[0].start_ms, o[-1].outcome) for o in occ] == [
        (14_710, "BLOCKED"),
        (16_010, "LANDED"),
    ]
    assert pattern_occurrences(events, ["UPPERCUT"], unobserved=unobserved) == []
    jabs = [ev("JAB", t) for t in (0, 300, 600)]
    assert len(pattern_occurrences(jabs, ["JAB", "JAB"])) == 2  # overlapping occurrences both count
    with pytest.raises(ValueError):
        pattern_occurrences(events, [])
