"""The Event: the fundamental stored unit. One individual boxing action by one fighter."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from boxing_ai.ontology import (
    Category,
    Direction,
    Outcome,
    Side,
    Target,
    action_spec,
)


class Event(BaseModel):
    """A single boxing action.

    Identical for human annotations and ML predictions; provenance (who/what produced it) lives on the
    *source run*, not here. ``fight``/``video``/``fighter`` are natural keys (slugs) so that analytics never
    need a database handle. Times are integer milliseconds of video time.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)

    fight: str = Field(min_length=1)
    video: str = Field(min_length=1)
    fighter: str = Field(min_length=1)

    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)

    action_type: str
    side: Side | None = None
    target: Target | None = None
    direction: Direction | None = None
    outcome: Outcome | None = None

    # None for human annotations — never a fake 1.0.
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    round_number: int | None = Field(default=None, ge=1)
    # ML extras only (track id, class probabilities); never used by core analytics.
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Assigned by the database.
    id: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _check_consistency(self) -> Event:
        spec = action_spec(self.action_type)  # raises ValueError for unknown actions
        problems: list[str] = []

        if self.end_ms < self.start_ms:
            problems.append(f"end_ms ({self.end_ms}) is before start_ms ({self.start_ms})")

        if spec.category is Category.PUNCH:
            if self.side is None:
                hint = f" (always {spec.implied_side})" if spec.implied_side else ""
                problems.append(f"{self.action_type} requires a side{hint}")
            elif spec.implied_side is not None and self.side is not spec.implied_side:
                problems.append(
                    f"{self.action_type} is always {spec.implied_side}, got side={self.side}"
                )
            if self.direction is not None:
                problems.append(f"{self.action_type} is a punch and takes no direction")
        else:
            for name in ("side", "target", "outcome"):
                if getattr(self, name) is not None:
                    problems.append(f"{self.action_type} is {spec.category} and takes no {name}")
            if self.direction is None:
                if spec.direction_required:
                    problems.append(f"{self.action_type} requires a direction")
            elif not spec.directions:
                problems.append(f"{self.action_type} takes no direction")
            elif self.direction not in spec.directions:
                allowed = "/".join(sorted(d.value for d in spec.directions))
                problems.append(
                    f"{self.action_type} direction must be one of {allowed}, got {self.direction}"
                )

        if problems:
            raise ValueError("; ".join(problems))
        return self

    @property
    def category(self) -> Category:
        return action_spec(self.action_type).category

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms
