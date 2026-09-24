import pytest

from boxing_ai.sequences import count_ngrams, ngrams, ranked


def test_jab_cross_hook_ngrams():
    tokens = ["JAB", "CROSS", "HOOK"]
    assert list(ngrams(tokens, 1)) == [("JAB",), ("CROSS",), ("HOOK",)]
    assert list(ngrams(tokens, 2)) == [("JAB", "CROSS"), ("CROSS", "HOOK")]
    assert list(ngrams(tokens, 3)) == [("JAB", "CROSS", "HOOK")]
    assert list(ngrams(tokens, 4)) == []


def test_n_below_one_is_rejected():
    with pytest.raises(ValueError):
        list(ngrams(["JAB"], 0))


def test_empty_sequence_has_no_ngrams():
    assert list(ngrams([], 1)) == []


def test_ngrams_never_span_segments():
    counts = count_ngrams([["JAB", "CROSS"], ["HOOK", "JAB"]], 2)
    assert counts == {("JAB", "CROSS"): 1, ("HOOK", "JAB"): 1}
    assert ("CROSS", "HOOK") not in counts


def test_count_ngrams_accumulates_across_segments():
    counts = count_ngrams([["JAB", "CROSS"], ["JAB", "CROSS", "HOOK"]], 2)
    assert counts[("JAB", "CROSS")] == 2
    assert counts[("CROSS", "HOOK")] == 1


def test_ranked_orders_by_count_then_key_and_is_deterministic():
    counts = {("JAB", "JAB"): 1, ("JAB", "CROSS"): 2, ("CROSS", "HOOK"): 1, ("HOOK", "JAB"): 2}
    assert ranked(counts) == [
        (("HOOK", "JAB"), 2),  # 2-count ties broken alphabetically ...
        (("JAB", "CROSS"), 2),  # ... HOOK < JAB
        (("CROSS", "HOOK"), 1),
        (("JAB", "JAB"), 1),
    ]


def test_ranked_min_count_and_limit():
    counts = {"a": 5, "b": 3, "c": 1}
    assert ranked(counts, min_count=3) == [("a", 5), ("b", 3)]
    assert ranked(counts, limit=1) == [("a", 5)]
