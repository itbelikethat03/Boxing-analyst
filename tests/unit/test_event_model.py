import pytest
from pydantic import ValidationError

from boxing_ai.events import Event
from boxing_ai.ontology import Category, Commitment, Direction, Outcome, Side, Target
from tests.factories import ev

BASE = {"fight": "f", "video": "v", "fighter": "a", "start_ms": 1000, "end_ms": 1180}


# --- Creation ---------------------------------------------------------------------------------------


def test_valid_punch():
    e = Event(
        **BASE, action_type="HOOK", side="LEAD", target="BODY", outcome="LANDED", confidence=0.94
    )
    assert e.action_type == "HOOK"
    assert e.side is Side.LEAD
    assert e.target is Target.BODY
    assert e.outcome is Outcome.LANDED
    assert e.category is Category.PUNCH
    assert e.duration_ms == 180
    assert e.round_number is None and e.id is None and e.metadata == {}


def test_valid_defense():
    e = Event(**BASE, action_type="SLIP", direction="LEFT", confidence=0.87)
    assert e.category is Category.DEFENSE
    assert e.direction is Direction.LEFT
    assert e.side is None and e.target is None


def test_human_events_carry_no_confidence_rather_than_a_fake_one():
    assert ev("JAB").confidence is None


def test_target_and_outcome_are_optional_unknowns_for_punches():
    e = Event(**BASE, action_type="HOOK", side="REAR")
    assert e.target is None and e.outcome is None


def test_zero_duration_event_is_allowed():
    assert Event(**BASE | {"end_ms": 1000}, action_type="JAB", side="LEAD").duration_ms == 0


def test_event_is_immutable():
    e = ev("JAB")
    with pytest.raises(ValidationError):
        e.start_ms = 5


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        Event(**BASE, action_type="JAB", side="LEAD", punch_type="JAB")


def test_whitespace_is_stripped_from_keys():
    e = Event(**BASE | {"fighter": "  a  "}, action_type="JAB", side="LEAD")
    assert e.fighter == "a"


def test_metadata_holds_ml_extras():
    e = ev("JAB", confidence=0.6, metadata={"track_id": 7, "probs": {"JAB": 0.6}})
    assert e.metadata["track_id"] == 7


# --- Validation: every rejection --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"start_ms": 2000, "end_ms": 1000}, "before start_ms"),
        ({"start_ms": -1}, "greater than or equal to 0"),
        ({"action_type": "FLURRY", "side": None}, "unknown action 'FLURRY'"),
        ({"action_type": "JAB_JAB_CROSS", "side": None}, "unknown action"),  # combos are not events
        ({"action_type": "JAB", "side": "REAR"}, "JAB is always LEAD"),
        ({"action_type": "CROSS", "side": "LEAD"}, "CROSS is always REAR"),
        ({"action_type": "JAB", "side": None}, "JAB requires a side"),
        ({"action_type": "HOOK", "side": None}, "HOOK requires a side"),
        ({"action_type": "HOOK", "side": "LEAD", "direction": "LEFT"}, "takes no direction"),
        ({"confidence": -0.1}, "greater than or equal to 0"),
        ({"confidence": 1.01}, "less than or equal to 1"),
        ({"round_number": 0}, "greater than or equal to 1"),
        ({"fighter": ""}, "at least 1 character"),
        ({"fight": ""}, "at least 1 character"),
        ({"side": "LEFT"}, "Input should be 'LEAD' or 'REAR'"),
        ({"target": "LEGS"}, "Input should be 'HEAD' or 'BODY'"),
    ],
)
def test_invalid_punch_rejected(overrides, message):
    fields = BASE | {"action_type": "HOOK", "side": "LEAD"} | overrides
    with pytest.raises(ValidationError, match=message):
        Event(**fields)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"target": "HEAD"}, "SLIP is DEFENSE and takes no target"),
        ({"side": "LEAD"}, "SLIP is DEFENSE and takes no side"),
        ({"outcome": "LANDED"}, "SLIP is DEFENSE and takes no outcome"),
        ({"direction": None}, "SLIP requires a direction"),
        ({"direction": "FORWARD"}, "direction must be one of LEFT/RIGHT"),
        ({"direction": "BACK"}, "direction must be one of LEFT/RIGHT"),
    ],
)
def test_invalid_defense_rejected(overrides, message):
    fields = BASE | {"action_type": "SLIP", "direction": "LEFT"} | overrides
    with pytest.raises(ValidationError, match=message):
        Event(**fields)


def test_actions_that_take_no_direction_reject_one():
    with pytest.raises(ValidationError, match="PULL takes no direction"):
        Event(**BASE, action_type="PULL", direction="BACK")
    assert Event(**BASE, action_type="PULL").direction is None


def test_step_requires_a_direction_and_accepts_any():
    with pytest.raises(ValidationError, match="STEP requires a direction"):
        Event(**BASE, action_type="STEP")
    for d in Direction:
        assert Event(**BASE, action_type="STEP", direction=d).direction is d


def test_all_problems_in_one_event_are_reported_together():
    with pytest.raises(ValidationError) as exc:
        Event(**BASE | {"end_ms": 10}, action_type="JAB", side="REAR", direction="LEFT")
    text = str(exc.value)
    assert "before start_ms" in text
    assert "JAB is always LEAD" in text
    assert "takes no direction" in text


# --- Feints and probes ------------------------------------------------------------------------------


def test_feint_takes_optional_side_and_target():
    hand = Event(**BASE, action_type="FEINT", side="REAR", target="BODY")
    assert (hand.category, hand.side, hand.target) == (Category.FEINT, Side.REAR, Target.BODY)
    shoulder = Event(**BASE, action_type="FEINT")
    assert shoulder.side is None and shoulder.target is None


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"outcome": "LANDED"}, "FEINT is FEINT and takes no outcome"),
        ({"commitment": "PROBE"}, "FEINT is FEINT and takes no commitment"),
        ({"direction": "LEFT"}, "FEINT takes no direction"),
    ],
)
def test_invalid_feint_rejected(overrides, message):
    with pytest.raises(ValidationError, match=message):
        Event(**BASE, action_type="FEINT", **overrides)


def test_probe_is_a_punch_attribute():
    e = Event(**BASE, action_type="JAB", side="LEAD", commitment="PROBE")
    assert e.commitment is Commitment.PROBE and e.category is Category.PUNCH
    with pytest.raises(ValidationError, match="SLIP is DEFENSE and takes no commitment"):
        Event(**BASE, action_type="SLIP", direction="LEFT", commitment="PROBE")
