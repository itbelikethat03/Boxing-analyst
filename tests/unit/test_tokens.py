from boxing_ai.sequences import (
    actor_tagged,
    by_action,
    by_action_direction,
    by_action_target,
)
from tests.factories import ev


def test_by_action_is_the_plain_code():
    assert by_action(ev("HOOK", target="BODY")) == "HOOK"
    assert by_action(ev("SLIP", direction="LEFT")) == "SLIP"


def test_by_action_direction_folds_direction_in_only_when_present():
    assert by_action_direction(ev("SLIP", direction="LEFT")) == "SLIP_LEFT"
    assert by_action_direction(ev("STEP", direction="BACK")) == "STEP_BACK"
    assert by_action_direction(ev("JAB")) == "JAB"
    assert by_action_direction(ev("PULL")) == "PULL"


def test_by_action_target_keeps_plain_code_for_unknown_target():
    assert by_action_target(ev("HOOK", target="BODY")) == "HOOK_BODY"
    assert by_action_target(ev("JAB", target="HEAD")) == "JAB_HEAD"
    assert by_action_target(ev("HOOK")) == "HOOK"
    assert by_action_target(ev("SLIP", direction="LEFT")) == "SLIP"


def test_actor_tagged_marks_self_and_opponent():
    tokenize = actor_tagged(by_action_direction, self_fighter="fighter-a")
    assert tokenize(ev("JAB", fighter="fighter-a")) == "SELF:JAB"
    assert tokenize(ev("SLIP", fighter="fighter-b", direction="RIGHT")) == "OPP:SLIP_RIGHT"
