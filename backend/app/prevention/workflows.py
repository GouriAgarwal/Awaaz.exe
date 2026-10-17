"""
Prevention Workflow State Machine.
Enforces strictly validated state transitions, prevents automatic unblocking of blocked actions,
and logs every state change into the incident timeline.
"""
from typing import Set, Dict, List, Optional
from datetime import datetime, timezone

from app.prevention.schemas import (
    PreventionStatus,
    PreventionAction,
    TimelineEvent,
)
from app.prevention.timeline import IncidentTimeline


class InvalidStateTransitionError(ValueError):
    """Raised when an illegal or unauthorized workflow transition is attempted."""
    pass


class PreventionWorkflowStateMachine:
    """
    State machine governing the lifecycle of a prevention workflow.
    Ensures blocked actions cannot be silently or automatically cleared without audit authorization.
    """

    # Valid transitions mapping
    VALID_TRANSITIONS: Dict[PreventionStatus, Set[PreventionStatus]] = {
        PreventionStatus.ALLOWED: {
            PreventionStatus.MONITORING,
            PreventionStatus.WARNING,
            PreventionStatus.VERIFICATION_REQUIRED,
            PreventionStatus.PAUSED,
            PreventionStatus.ESCALATED,
            PreventionStatus.BLOCKED,
            PreventionStatus.RESOLVED,
        },
        PreventionStatus.MONITORING: {
            PreventionStatus.ALLOWED,
            PreventionStatus.WARNING,
            PreventionStatus.VERIFICATION_REQUIRED,
            PreventionStatus.PAUSED,
            PreventionStatus.ESCALATED,
            PreventionStatus.BLOCKED,
            PreventionStatus.RESOLVED,
        },
        PreventionStatus.WARNING: {
            PreventionStatus.MONITORING,
            PreventionStatus.ALLOWED,
            PreventionStatus.VERIFICATION_REQUIRED,
            PreventionStatus.PAUSED,
            PreventionStatus.ESCALATED,
            PreventionStatus.BLOCKED,
            PreventionStatus.RESOLVED,
        },
        PreventionStatus.VERIFICATION_REQUIRED: {
            PreventionStatus.PAUSED,
            PreventionStatus.ESCALATED,
            PreventionStatus.BLOCKED,
            PreventionStatus.RESOLVED,
            PreventionStatus.ALLOWED,
        },
        PreventionStatus.PAUSED: {
            PreventionStatus.VERIFICATION_REQUIRED,
            PreventionStatus.ESCALATED,
            PreventionStatus.BLOCKED,
            PreventionStatus.RESOLVED,
            PreventionStatus.ALLOWED,
        },
        PreventionStatus.ESCALATED: {
            PreventionStatus.PAUSED,
            PreventionStatus.BLOCKED,
            PreventionStatus.RESOLVED,
            PreventionStatus.ALLOWED,
        },
        PreventionStatus.BLOCKED: {
            # STRICT GUARD: A blocked workflow CANNOT automatically transition to ALLOWED or MONITORING.
            # It can only be ESCALATED to higher tier or explicitly RESOLVED with supervisor sign-off.
            PreventionStatus.ESCALATED,
            PreventionStatus.RESOLVED,
        },
        PreventionStatus.RESOLVED: {
            # Can re-open only if a new threat event occurs
            PreventionStatus.WARNING,
            PreventionStatus.PAUSED,
            PreventionStatus.ESCALATED,
            PreventionStatus.BLOCKED,
            PreventionStatus.MONITORING,
        },
    }

    @classmethod
    def can_transition(cls, current_status: PreventionStatus, target_status: PreventionStatus) -> bool:
        """Checks if a transition between two states is allowed."""
        if current_status == target_status:
            return True
        allowed = cls.VALID_TRANSITIONS.get(current_status, set())
        return target_status in allowed

    @classmethod
    def validate_transition(cls, current_status: PreventionStatus, target_status: PreventionStatus) -> None:
        """Raises InvalidStateTransitionError if the transition is prohibited."""
        if not cls.can_transition(current_status, target_status):
            raise InvalidStateTransitionError(
                f"Prohibited workflow transition: Cannot move from '{current_status.value}' to '{target_status.value}'. "
                f"Blocked or high-risk actions require explicit supervisor resolution or escalation."
            )
