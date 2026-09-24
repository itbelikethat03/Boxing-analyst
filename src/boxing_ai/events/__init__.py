"""Event domain model: the Event itself, ordering, coverage (rounds/unobserved) and context validation."""

from boxing_ai.events.coverage import Round, UnobservedInterval, observed_ms, spans_overlap
from boxing_ai.events.model import Event
from boxing_ai.events.ordering import chronological, sort_key
from boxing_ai.events.validation import EventContext, Problem, check_events

__all__ = [
    "Event",
    "EventContext",
    "Problem",
    "Round",
    "UnobservedInterval",
    "check_events",
    "chronological",
    "observed_ms",
    "sort_key",
    "spans_overlap",
]
