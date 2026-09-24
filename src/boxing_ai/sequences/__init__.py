"""Sequence primitives: tokenize -> segment into bursts -> n-grams / transition tables."""

from boxing_ai.sequences.ngrams import Ngram, count_ngrams, ngrams, ranked
from boxing_ai.sequences.segmentation import (
    DEFAULT_CATEGORIES,
    DEFAULT_COMBO_GAP_MS,
    DEFAULT_EXCHANGE_GAP_MS,
    DEFAULT_SPEC,
    StreamSpec,
    segment,
    token_segments,
)
from boxing_ai.sequences.tokens import (
    OPPONENT,
    SELF,
    Tokenizer,
    actor_tagged,
    by_action,
    by_action_direction,
    by_action_target,
)
from boxing_ai.sequences.transitions import (
    END,
    START,
    Context,
    Transition,
    followers,
    transition_counts,
    transitions,
)

__all__ = [
    "DEFAULT_CATEGORIES",
    "DEFAULT_COMBO_GAP_MS",
    "DEFAULT_EXCHANGE_GAP_MS",
    "DEFAULT_SPEC",
    "END",
    "OPPONENT",
    "SELF",
    "START",
    "Context",
    "Ngram",
    "StreamSpec",
    "Tokenizer",
    "Transition",
    "actor_tagged",
    "by_action",
    "by_action_direction",
    "by_action_target",
    "count_ngrams",
    "followers",
    "ngrams",
    "ranked",
    "segment",
    "token_segments",
    "transition_counts",
    "transitions",
]
