"""The boxing action vocabulary — the single source of truth.

Everything else (Event validation, the DB ``action_types`` table, annotation parsing) derives from here.
Design rules:

* Actions are *individual* actions. Combinations (``JAB_JAB_CROSS``) are never entries in this file.
* Variants are **attributes**, not classes: a slip to the left is ``SLIP`` + ``direction=LEFT``.
  Whether an n-gram token reads ``SLIP`` or ``SLIP_LEFT`` is an analysis-time tokenizer choice.
* Adding an action later means adding one ``ActionSpec`` row here (and bumping ``ONTOLOGY_VERSION``);
  no schema change is needed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

ONTOLOGY_VERSION = "0.1"


class Category(StrEnum):
    PUNCH = "PUNCH"
    DEFENSE = "DEFENSE"
    MOVEMENT = "MOVEMENT"


class Side(StrEnum):
    """Which hand, relative to the fighter's stance (lead = the hand nearer the opponent)."""

    LEAD = "LEAD"
    REAR = "REAR"


class Target(StrEnum):
    HEAD = "HEAD"
    BODY = "BODY"


class Direction(StrEnum):
    """Direction of a defensive/movement action, in the *fighter's own* frame of reference."""

    LEFT = "LEFT"
    RIGHT = "RIGHT"
    FORWARD = "FORWARD"
    BACK = "BACK"


class Outcome(StrEnum):
    LANDED = "LANDED"
    BLOCKED = "BLOCKED"
    MISSED = "MISSED"


class UnobservedKind(StrEnum):
    """Why a stretch of video is not (exhaustively) annotated. Events may not fall inside these."""

    REPLAY = "REPLAY"
    CUTAWAY = "CUTAWAY"
    UNSUPPORTED_ANGLE = "UNSUPPORTED_ANGLE"
    UNCERTAIN = "UNCERTAIN"


@dataclass(frozen=True)
class ActionSpec:
    """Rules for one action code.

    ``implied_side``: side that is true *by definition* (jab = lead hand, cross = rear hand).
    ``directions``: allowed directions for non-punch actions (empty = the action takes no direction).
    ``direction_required``: whether a direction must be supplied.
    """

    code: str
    category: Category
    description: str
    implied_side: Side | None = None
    directions: frozenset[Direction] = frozenset()
    direction_required: bool = False


_LR = frozenset({Direction.LEFT, Direction.RIGHT})
_ALL_DIRECTIONS = frozenset(Direction)

_SPECS: tuple[ActionSpec, ...] = (
    # --- Punches -------------------------------------------------------------------------------
    ActionSpec("JAB", Category.PUNCH, "Straight punch with the lead hand", implied_side=Side.LEAD),
    ActionSpec(
        "CROSS", Category.PUNCH, "Straight punch with the rear hand", implied_side=Side.REAR
    ),
    ActionSpec("HOOK", Category.PUNCH, "Circular punch; either hand"),
    ActionSpec("UPPERCUT", Category.PUNCH, "Rising punch; either hand"),
    # --- Defensive actions (registered now so the model is proven generic; annotating them is optional)
    ActionSpec(
        "SLIP",
        Category.DEFENSE,
        "Head movement off the punch line",
        directions=_LR,
        direction_required=True,
    ),
    ActionSpec(
        "ROLL",
        Category.DEFENSE,
        "Rolling the head/shoulders under a punch",
        directions=_LR,
        direction_required=True,
    ),
    ActionSpec("PULL", Category.DEFENSE, "Pulling the head straight back from a punch"),
    ActionSpec("BLOCK", Category.DEFENSE, "Absorbing a punch on the gloves/arms"),
    ActionSpec("PARRY", Category.DEFENSE, "Redirecting a punch with the hand"),
    # --- Movement ------------------------------------------------------------------------------
    ActionSpec(
        "STEP",
        Category.MOVEMENT,
        "Footwork step",
        directions=_ALL_DIRECTIONS,
        direction_required=True,
    ),
)

ACTIONS: Mapping[str, ActionSpec] = MappingProxyType({s.code: s for s in _SPECS})


def action_spec(code: str) -> ActionSpec:
    """Return the spec for ``code`` or raise ``ValueError`` listing the known codes."""
    try:
        return ACTIONS[code]
    except KeyError:
        known = ", ".join(sorted(ACTIONS))
        raise ValueError(f"unknown action {code!r}; known actions: {known}") from None


def category_of(code: str) -> Category:
    return action_spec(code).category


def actions_in(category: Category) -> tuple[str, ...]:
    return tuple(s.code for s in _SPECS if s.category is category)


def resolve_side(action_type: str, side: Side | str | None) -> Side | None:
    """Fill in a side that is implied by the action, and reject a contradicting one.

    Used at input boundaries (annotation loader) so that stored events always carry an explicit side.
    Actions without an implied side (HOOK, UPPERCUT, non-punches) are returned as given.
    """
    spec = action_spec(action_type)
    given = Side(side) if side is not None else None
    if spec.implied_side is None:
        return given
    if given is not None and given is not spec.implied_side:
        raise ValueError(f"{action_type} is always {spec.implied_side}, got side={given}")
    return spec.implied_side
