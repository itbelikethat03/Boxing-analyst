"""Tokenizers: projections ``Event -> str`` used as the alphabet of sequence analysis.

The stored event keeps ``action_type`` and its attributes separate. Which of them are folded into the token is
decided *per analysis*, so ``SLIP`` vs ``SLIP_LEFT`` (or ``HOOK`` vs ``HOOK_BODY``) never needs re-annotation
or retraining.
"""

from __future__ import annotations

from collections.abc import Callable

from boxing_ai.events import Event
from boxing_ai.ontology import Commitment

Tokenizer = Callable[[Event], str]

SELF = "SELF"
OPPONENT = "OPP"


def by_action(event: Event) -> str:
    """``JAB``, ``SLIP``, … — the default alphabet."""
    return event.action_type


def by_action_direction(event: Event) -> str:
    """``SLIP_LEFT``, ``STEP_BACK``; actions without a direction keep their plain code."""
    return f"{event.action_type}_{event.direction}" if event.direction else event.action_type


def by_action_target(event: Event) -> str:
    """``HOOK_BODY``, ``JAB_HEAD``; events with an unknown target keep their plain code."""
    return f"{event.action_type}_{event.target}" if event.target else event.action_type


def by_action_commitment(event: Event) -> str:
    """``JAB_PROBE`` for probing punches; everything else keeps its plain code (``JAB``, ``FEINT``)."""
    return (
        f"{event.action_type}_{event.commitment}"
        if event.commitment is Commitment.PROBE
        else event.action_type
    )


def actor_tagged(tokenizer: Tokenizer, *, self_fighter: str) -> Tokenizer:
    """Prefix tokens with who acted, for two-fighter streams: ``SELF:JAB``, ``OPP:SLIP_LEFT``.

    Needed for reaction questions ("what does he do after the *opponent's* jab?"), which cannot be answered
    from one fighter's own events alone.
    """

    def tokenize(event: Event) -> str:
        actor = SELF if event.fighter == self_fighter else OPPONENT
        return f"{actor}:{tokenizer(event)}"

    return tokenize


# Named tokenizers for user interfaces (CLI flags, API parameters).
TOKENIZERS: dict[str, Tokenizer] = {
    "action": by_action,
    "direction": by_action_direction,
    "target": by_action_target,
    "commitment": by_action_commitment,
}
