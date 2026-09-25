"""Human timecodes <-> integer milliseconds. Timecodes exist only at this I/O boundary.

Accepted: ``SS.mmm`` (any number of seconds, e.g. ``125.3``), ``MM:SS.mmm`` and ``HH:MM:SS.mmm``. The fraction is
optional and has at most 3 digits (``.4`` = 400 ms); more digits would silently lose precision, so they are
rejected.
"""

from __future__ import annotations

import re

_TIMECODE = re.compile(r"^(?:(?:(\d+):)?(\d{1,2}):)?(\d+)(?:\.(\d{1,3}))?$")


def parse_timecode(text: str) -> int:
    """Return the timecode as integer milliseconds, or raise ``ValueError`` with a readable message."""
    value = text.strip()
    m = _TIMECODE.match(value)
    if not m:
        raise ValueError(f"invalid time {text!r}; expected SS.mmm, MM:SS.mmm or HH:MM:SS.mmm")
    hours, minutes, seconds, fraction = m.groups()
    if minutes is not None and int(seconds) >= 60:
        raise ValueError(f"invalid time {text!r}: seconds must be < 60")
    if hours is not None and int(minutes) >= 60:
        raise ValueError(f"invalid time {text!r}: minutes must be < 60")
    total_s = int(hours or 0) * 3600 + int(minutes or 0) * 60 + int(seconds)
    return total_s * 1000 + int((fraction or "").ljust(3, "0"))


def format_timecode(ms: int) -> str:
    """``MM:SS.mmm`` below one hour, ``HH:MM:SS.mmm`` from one hour on (both parse back exactly)."""
    if ms < 0:
        raise ValueError(f"cannot format a negative time ({ms} ms)")
    total_s, millis = divmod(ms, 1000)
    hours, rest = divmod(total_s, 3600)
    minutes, seconds = divmod(rest, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"
    return f"{minutes:02d}:{seconds:02d}.{millis:03d}"
