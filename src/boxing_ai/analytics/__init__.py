"""Analytics over event streams: pure functions of events (no database, no UI)."""

from boxing_ai.analytics.filters import select_events
from boxing_ai.analytics.patterns import (
    NgramCount,
    entry_patterns,
    exit_patterns,
    next_after,
    top_ngrams,
    transition_probabilities,
)
from boxing_ai.analytics.profile import (
    ActionCount,
    RoundStat,
    action_distribution,
    round_stats,
)

__all__ = [
    "ActionCount",
    "NgramCount",
    "RoundStat",
    "action_distribution",
    "entry_patterns",
    "exit_patterns",
    "next_after",
    "round_stats",
    "select_events",
    "top_ngrams",
    "transition_probabilities",
]
