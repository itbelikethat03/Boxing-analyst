import pytest
from pydantic import ValidationError

from boxing_ai.events import Round, UnobservedInterval, observed_ms, spans_overlap


def gap(start, end, video="v", kind="REPLAY"):
    return UnobservedInterval(video=video, start_ms=start, end_ms=end, kind=kind)


# --- spans_overlap: event [s, e] vs half-open interval [a, b) ---------------------------------------


@pytest.mark.parametrize(
    ("event", "interval", "expected"),
    [
        ((10, 20), (30, 40), False),
        ((10, 35), (30, 40), True),  # partial overlap
        ((32, 38), (30, 40), True),  # contained
        ((10, 50), (30, 40), True),  # contains the interval
        ((10, 30), (30, 40), False),  # merely touches the start
        ((40, 60), (30, 40), False),  # merely touches the end (half-open)
        ((30, 30), (30, 40), True),  # point event at the interval start is inside
        ((40, 40), (30, 40), False),  # point event at the (exclusive) end is outside
        ((29, 29), (30, 40), False),
    ],
)
def test_spans_overlap(event, interval, expected):
    assert spans_overlap(*event, *interval) is expected


# --- observed_ms --------------------------------------------------------------------------------------


def test_observed_ms_without_gaps_is_the_full_span():
    assert observed_ms(1000, 4000, []) == 3000


def test_observed_ms_subtracts_a_contained_gap():
    assert observed_ms(0, 10_000, [gap(2_000, 3_500)]) == 8_500


def test_observed_ms_clips_gaps_to_the_span():
    # gap starts before the span and ends inside it; another starts inside and runs past the end
    assert observed_ms(1_000, 5_000, [gap(0, 2_000), gap(4_500, 9_000)]) == 2_500


def test_observed_ms_merges_overlapping_and_duplicate_gaps():
    gaps = [gap(1_000, 3_000), gap(2_000, 4_000), gap(1_000, 3_000), gap(6_000, 7_000)]
    # union = [1000,4000) + [6000,7000) = 4000 ms unobserved
    assert observed_ms(0, 10_000, gaps) == 6_000


def test_observed_ms_span_fully_covered_is_zero():
    assert observed_ms(2_000, 3_000, [gap(0, 10_000)]) == 0


def test_observed_ms_ignores_gaps_outside_the_span():
    assert observed_ms(0, 1_000, [gap(5_000, 6_000)]) == 1_000


def test_observed_ms_rejects_reversed_span():
    with pytest.raises(ValueError):
        observed_ms(5, 4, [])


# --- models -------------------------------------------------------------------------------------------


def test_unobserved_interval_requires_positive_length_and_known_kind():
    with pytest.raises(ValidationError, match="must be greater than start_ms"):
        gap(100, 100)
    with pytest.raises(ValidationError):
        UnobservedInterval(video="v", start_ms=0, end_ms=10, kind="COMMERCIAL")


def test_round_requires_positive_number_and_length():
    Round(video="v", number=1, start_ms=0, end_ms=180_000)
    with pytest.raises(ValidationError):
        Round(video="v", number=0, start_ms=0, end_ms=10)
    with pytest.raises(ValidationError):
        Round(video="v", number=1, start_ms=10, end_ms=5)
