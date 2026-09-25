import pytest

from boxing_ai.ontology import (
    ACTIONS,
    Category,
    SessionKind,
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


def test_feint_is_its_own_category_and_not_a_punch():
    assert actions_in(Category.FEINT) == ("FEINT",)
    spec = action_spec("FEINT")
    assert spec.takes_side and not spec.side_required and spec.takes_target
    assert not spec.takes_outcome and not spec.takes_commitment


def test_every_punch_takes_side_target_outcome_and_commitment():
    for code in actions_in(Category.PUNCH):
        spec = action_spec(code)
        assert spec.side_required and spec.takes_target and spec.takes_outcome
        assert spec.takes_commitment


def test_session_kinds_know_how_many_people_are_annotated():
    assert {k: k.participants for k in SessionKind} == {
        "FIGHT": 2,
        "SPARRING": 2,
        "PADS": 1,
        "BAG": 1,
        "SHADOW": 1,
    }


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
