"""Transition statistics: P(next token | previous ``order`` tokens), with explicit START/END sentinels.

Every segment is padded ``START ... END``. Consequences:

* ``P(JAB | START)`` is how often bursts open with a jab (entry patterns).
* ``P(END | CROSS)`` is how often a burst ends on a cross (exit patterns).
* For each context the probabilities sum to exactly 1.

Without ``END`` the last token of each burst would silently vanish from the denominator: a cross that ended the
burst would be ignored and ``P(SLIP | CROSS)`` would be overstated.

Small counts are the norm in annotated data, so every ``Transition`` carries its raw ``count`` and ``total``;
report them next to the probability and use ``min_count`` to hide noise.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

START = "<START>"
END = "<END>"

Context = tuple[str, ...]


@dataclass(frozen=True)
class Transition:
    context: Context
    next: str
    count: int  # times `next` followed `context`
    total: int  # times `context` occurred followed by anything (including END)

    @property
    def probability(self) -> float:
        return self.count / self.total


def transition_counts(
    segments: Iterable[Sequence[str]], order: int = 1
) -> dict[Context, Counter[str]]:
    """Count what follows every ``order``-token context, with START/END padding. Empty segments are skipped."""
    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")
    table: dict[Context, Counter[str]] = {}
    for tokens in segments:
        if not tokens:
            continue
        padded = [START] * order + list(tokens) + [END]
        for i in range(order, len(padded)):
            context = tuple(padded[i - order : i])
            table.setdefault(context, Counter())[padded[i]] += 1
    return table


def transitions(table: dict[Context, Counter[str]], *, min_count: int = 1) -> list[Transition]:
    """Flatten a count table into ``Transition`` rows: context ascending, then count descending, then token."""
    rows: list[Transition] = []
    for context in sorted(table):
        following = table[context]
        total = sum(following.values())
        for token, count in sorted(following.items(), key=lambda tc: (-tc[1], tc[0])):
            if count >= min_count:
                rows.append(Transition(context, token, count, total))
    return rows


def followers(
    table: dict[Context, Counter[str]], context: str | Context, *, min_count: int = 1
) -> list[Transition]:
    """What follows one specific context, most frequent first. A bare string means a 1-token context."""
    key: Context = (context,) if isinstance(context, str) else tuple(context)
    if key not in table:
        return []
    return transitions({key: table[key]}, min_count=min_count)
