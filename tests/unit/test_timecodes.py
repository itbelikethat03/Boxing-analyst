"""Timecode parsing/formatting at the annotation I/O boundary."""

import pytest

from boxing_ai.annotations import format_timecode, parse_timecode


@pytest.mark.parametrize(
    ("text", "ms"),
    [
        ("0", 0),
        ("14.32", 14_320),
        ("14.320", 14_320),
        ("125.3", 125_300),  # seconds-only may exceed 59
        ("03:26.320", 206_320),
        ("3:26.3", 206_300),
        ("03:26", 206_000),
        ("01:02:03.450", 3_723_450),
        ("  00:01.001 ", 1_001),
    ],
)
def test_parse_accepted_formats(text, ms):
    assert parse_timecode(text) == ms


@pytest.mark.parametrize(
    "text",
    ["", "abc", "-1.000", "1.2345", "03:60.000", "01:60:00.000", "1:2:3:4", "03,26.320", "1."],
)
def test_parse_rejects(text):
    with pytest.raises(ValueError, match="invalid time"):
        parse_timecode(text)


@pytest.mark.parametrize(
    ("ms", "text"),
    [(0, "00:00.000"), (206_320, "03:26.320"), (3_723_450, "01:02:03.450"), (59_999, "00:59.999")],
)
def test_format(ms, text):
    assert format_timecode(ms) == text


@pytest.mark.parametrize("ms", [0, 1, 999, 60_000, 3_599_999, 3_600_000, 36_000_001])
def test_format_parse_round_trip(ms):
    assert parse_timecode(format_timecode(ms)) == ms


def test_format_rejects_negative():
    with pytest.raises(ValueError):
        format_timecode(-1)
