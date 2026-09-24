import pytest

from boxing_ai.ontology import (
    ACTIONS,
    Category,
    Side,
    action_spec,
    actions_in,
    category_of,
    resolve_side,
)


def test_initial_punch_ontology():
    assert actions_in(Category.PUNCH) == ("JAB", "CROSS", "HOOK", "UPPERCUT")


def test_defensive_and_movement_actions_are_registered_for_extensibility():
    assert set(actions_in(Category.DEFENSE)) == {"SLIP", "ROLL", "PULL", "BLOCK", "PARRY"}
    assert actions_in(Category.MOVEMENT) == ("STEP",)


def test_combinations_and_variants_are_never_action_codes():
    # Core principle: individual actions only. Variants are attributes (direction/side/target), and
    # combinations are derived — so no code may be a compound like JAB_CROSS or SLIP_LEFT.
    assert all("_" not in code for code in ACTIONS)


def test_category_lookup():
    assert category_of("HOOK") is Category.PUNCH
    assert category_of("SLIP") is Category.DEFENSE


def test_unknown_action_lists_known_codes():
    with pytest.raises(ValueError, match="unknown action 'JAB_JAB_CROSS'.*JAB"):
        action_spec("JAB_JAB_CROSS")


@pytest.mark.parametrize(
    ("action", "given", "expected"),
    [
        ("JAB", None, Side.LEAD),
        ("CROSS", None, Side.REAR),
        ("JAB", "LEAD", Side.LEAD),
        ("HOOK", None, None),  # not implied: left for the caller/validator to require
        ("HOOK", "REAR", Side.REAR),
        ("UPPERCUT", Side.LEAD, Side.LEAD),
        ("SLIP", None, None),
    ],
)
def test_resolve_side(action, given, expected):
    assert resolve_side(action, given) == expected


@pytest.mark.parametrize(("action", "given"), [("JAB", "REAR"), ("CROSS", "LEAD")])
def test_resolve_side_rejects_contradiction(action, given):
    with pytest.raises(ValueError, match="always"):
        resolve_side(action, given)
