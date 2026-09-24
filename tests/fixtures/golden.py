"""Hand-calculated golden data for sequence analytics (also reused by the DB cross-check in M4).

The stream is the example from the project brief. One fighter, one burst (all gaps well under 1 s):

    t(ms)   action
    14320   JAB
    14710   JAB
    15050   CROSS
    15420   SLIP (LEFT)
    16010   JAB
    16400   CROSS

Tokens (action + direction): JAB JAB CROSS SLIP_LEFT JAB CROSS.   Every number below was derived by hand:

Unigrams    JAB 3, CROSS 2, SLIP_LEFT 1
Bigrams (5) JAB>JAB 1, JAB>CROSS 2, CROSS>SLIP_LEFT 1, SLIP_LEFT>JAB 1
Trigrams (4) each of JAB>JAB>CROSS, JAB>CROSS>SLIP_LEFT, CROSS>SLIP_LEFT>JAB, SLIP_LEFT>JAB>CROSS once

Padded START JAB JAB CROSS SLIP_LEFT JAB CROSS END, what follows each token:
    START      -> JAB 1                       (total 1)
    JAB        -> JAB 1, CROSS 2              (total 3)   P(CROSS|JAB) = 2/3
    CROSS      -> SLIP_LEFT 1, END 1          (total 2)   P(END|CROSS) = 1/2
    SLIP_LEFT  -> JAB 1                       (total 1)

Adding one lone JAB 30 s later creates a second burst [JAB]: the bigrams are unchanged (no CROSS>JAB across
the pause), START>JAB becomes 2/2, and JAB -> {JAB 1, CROSS 2, END 1} (total 4) so P(CROSS|JAB) drops to 1/2.
"""

from __future__ import annotations

from boxing_ai.events import Event
from tests.factories import ev

TIMELINE = [
    ("JAB", 14_320),
    ("JAB", 14_710),
    ("CROSS", 15_050),
    ("SLIP", 15_420),
    ("JAB", 16_010),
    ("CROSS", 16_400),
]
LONE_JAB_MS = 46_400  # 30 s after the last cross ended (16_580)


def golden_events(*, with_lone_jab: bool = False) -> list[Event]:
    events = [ev(action, t) for action, t in TIMELINE]
    if with_lone_jab:
        events.append(ev("JAB", LONE_JAB_MS))
    return events


GOLDEN_TOKENS = ["JAB", "JAB", "CROSS", "SLIP_LEFT", "JAB", "CROSS"]

UNIGRAMS = {("JAB",): 3, ("CROSS",): 2, ("SLIP_LEFT",): 1}

BIGRAMS = {
    ("JAB", "JAB"): 1,
    ("JAB", "CROSS"): 2,
    ("CROSS", "SLIP_LEFT"): 1,
    ("SLIP_LEFT", "JAB"): 1,
}

TRIGRAMS = {
    ("JAB", "JAB", "CROSS"): 1,
    ("JAB", "CROSS", "SLIP_LEFT"): 1,
    ("CROSS", "SLIP_LEFT", "JAB"): 1,
    ("SLIP_LEFT", "JAB", "CROSS"): 1,
}

# (context) -> {next: count}; totals are the sums.
TRANSITIONS_ORDER_1 = {
    ("<START>",): {"JAB": 1},
    ("JAB",): {"JAB": 1, "CROSS": 2},
    ("CROSS",): {"SLIP_LEFT": 1, "<END>": 1},
    ("SLIP_LEFT",): {"JAB": 1},
}

TRANSITIONS_ORDER_2 = {
    ("<START>", "<START>"): {"JAB": 1},
    ("<START>", "JAB"): {"JAB": 1},
    ("JAB", "JAB"): {"CROSS": 1},
    ("JAB", "CROSS"): {"SLIP_LEFT": 1, "<END>": 1},
    ("CROSS", "SLIP_LEFT"): {"JAB": 1},
    ("SLIP_LEFT", "JAB"): {"CROSS": 1},
}
