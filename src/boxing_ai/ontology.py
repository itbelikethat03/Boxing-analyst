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

ONTOLOGY_VERSION = "0.2"  # 0.2: FEINT category/action, punch `commitment`


class Category(StrEnum):
    PUNCH = "PUNCH"
    FEINT = "FEINT"  # a faked attack: no strike is thrown, but it belongs in sequences
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


class Commitment(StrEnum):
    """How committed a punch is. ``PROBE`` = pawing/range-finding/measuring; ``FULL`` = a committed shot.

    A probe is still a thrown punch (it counts as one); a *feint* throws nothing and is the ``FEINT`` action.
    """

    PROBE = "PROBE"
    FULL = "FULL"


class Corner(StrEnum):
    """How annotators identify a fighter within a fight."""

    RED = "RED"
    BLUE = "BLUE"


class Stance(StrEnum):
    """Which hand is LEAD: orthodox = left, southpaw = right."""

    ORTHODOX = "ORTHODOX"
    SOUTHPAW = "SOUTHPAW"
    SWITCH = "SWITCH"


class SessionKind(StrEnum):
    """What kind of footage: a real bout, or training used for comparison."""

    FIGHT = "FIGHT"
    SPARRING = "SPARRING"
    PADS = "PADS"
    BAG = "BAG"
    SHADOW = "SHADOW"

    @property
    def participants(self) -> int:
        """Annotated people: two boxers, or one (pad holders, bags and mirrors are not annotated)."""
        return 2 if self in (SessionKind.FIGHT, SessionKind.SPARRING) else 1


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
    ``directions``: allowed directions (empty = the action takes no direction).
    ``direction_required``: whether a direction must be supplied.
    The ``takes_*`` flags say which optional attributes the action accepts; ``side_required`` whether the side
    must be known. Defaults are those of a defensive/movement action (no side/target/outcome/commitment).
    """

    code: str
    category: Category
    description: str
    implied_side: Side | None = None
    directions: frozenset[Direction] = frozenset()
    direction_required: bool = False
    takes_side: bool = False
    side_required: bool = False
    takes_target: bool = False
    takes_outcome: bool = False
    takes_commitment: bool = False


def _punch(code: str, description: str, implied_side: Side | None = None) -> ActionSpec:
    return ActionSpec(
        code,
        Category.PUNCH,
        description,
        implied_side=implied_side,
        takes_side=True,
        side_required=True,
        takes_target=True,
        takes_outcome=True,
        takes_commitment=True,
    )


_LR = frozenset({Direction.LEFT, Direction.RIGHT})
_ALL_DIRECTIONS = frozenset(Direction)

_SPECS: tuple[ActionSpec, ...] = (
    # --- Punches -------------------------------------------------------------------------------
    _punch("JAB", "Straight punch with the lead hand", implied_side=Side.LEAD),
    _punch("CROSS", "Straight punch with the rear hand", implied_side=Side.REAR),
    _punch("HOOK", "Circular punch; either hand"),
    _punch("UPPERCUT", "Rising punch; either hand"),
    # --- Feints: side = the hand that faked (blank for shoulder/head/foot feints), target = what it threatened
    ActionSpec(
        "FEINT",
        Category.FEINT,
        "Faked attack meant to draw a reaction; nothing is thrown",
        takes_side=True,
        takes_target=True,
    ),
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
