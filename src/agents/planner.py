"""Goal planning for healthcare assistant requests."""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Goal:
    """A single executable goal produced from a user request."""

    name: str
    input_text: str


class Planner:
    """Convert user requests into goals for the execution service."""

    def plan(self, request: str) -> List[Goal]:
        """Return the goals required to handle a healthcare request."""
        request = request.strip()
        if not request:
            raise ValueError("A request is required")

        goals = []
        booking_terms = ("book", "appointment", "schedule", "doctor")
        if any(term in request.lower() for term in booking_terms):
            goals.append(Goal("appointment", request))
        else:
            goals.append(Goal("medical_question", request))
        return goals
