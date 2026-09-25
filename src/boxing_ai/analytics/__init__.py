"""Analytics over event streams: pure functions of events (no database, no UI)."""

from boxing_ai.analytics.filters import select_events
from boxing_ai.analytics.patterns import (
    NgramCount,
    PatternOutcome,
    entry_patterns,
    exit_patterns,
    next_after,
    pattern_occurrences,
    pattern_outcomes,
    top_ngrams,
    transition_probabilities,
)
from boxing_ai.analytics.profile import (
    ActionCount,
    OutcomeStat,
    RoundStat,
    action_distribution,
    outcome_breakdown,
    round_stats,
)

__all__ = [
    "ActionCount",
    "NgramCount",
    "OutcomeStat",
    "PatternOutcome",
    "RoundStat",
    "action_distribution",
    "entry_patterns",
    "exit_patterns",
    "next_after",
    "outcome_breakdown",
    "pattern_occurrences",
    "pattern_outcomes",
    "round_stats",
    "select_events",
    "top_ngrams",
    "transition_probabilities",
]
