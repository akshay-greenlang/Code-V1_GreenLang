# -*- coding: utf-8 -*-
"""
Feature Flag Lifecycle Manager - INFRA-008

Manages feature flag lifecycle transitions using a strict state machine.
Enforces valid transitions between flag statuses and provides recommendations
for flags that should transition based on age and rollout criteria.

State Machine:
    DRAFT -> ACTIVE
    ACTIVE -> ROLLED_OUT
    ACTIVE -> KILLED
    ACTIVE -> ARCHIVED
    ROLLED_OUT -> PERMANENT
    ROLLED_OUT -> ARCHIVED
    PERMANENT -> ARCHIVED
    KILLED -> ACTIVE (restore)
    KILLED -> ARCHIVED
    Any -> ARCHIVED (universal archive)

Recommendations:
    - Flags at 100% rollout for >14 days -> suggest ROLLED_OUT
    - Flags in ACTIVE state for >90 days -> suggest review
    - Killed flags for >7 days -> suggest ARCHIVE

Example:
    >>> from greenlang.infrastructure.feature_flags.lifecycle.manager import FlagLifecycleManager
    >>> manager = FlagLifecycleManager(service)
    >>> await manager.transition_state("my-flag", FlagStatus.ACTIVE, "engineer")
    >>> recommendations = await manager.recommend_transitions()
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from greenlang.infrastructure.feature_flags.models import (
    AuditLogEntry,
    FeatureFlag,
    FlagStatus,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State Machine Definition
# ---------------------------------------------------------------------------

# Mapping: current_status -> set of allowed next statuses
VALID_TRANSITIONS: Dict[FlagStatus, Set[FlagStatus]] = {
    FlagStatus.DRAFT: {
        FlagStatus.ACTIVE,
        FlagStatus.ARCHIVED,
    },
    FlagStatus.ACTIVE: {
        FlagStatus.ROLLED_OUT,
        FlagStatus.KILLED,
        FlagStatus.ARCHIVED,
    },
    FlagStatus.ROLLED_OUT: {
        FlagStatus.PERMANENT,
        FlagStatus.ARCHIVED,
    },
    FlagStatus.PERMANENT: {
        FlagStatus.ARCHIVED,
    },
    FlagStatus.ARCHIVED: set(),  # Terminal state - no transitions out
    FlagStatus.KILLED: {
        FlagStatus.ACTIVE,   # Restore from kill
        FlagStatus.ARCHIVED,
    },
}

# Recommendation thresholds (in days)
ROLLOUT_PROMOTION_DAYS = 14
ACTIVE_REVIEW_DAYS = 90
KILLED_ARCHIVE_DAYS = 7


class FlagLifecycleManager:
    """Manages feature flag lifecycle transitions with state machine enforcement.

    Ensures all flag status changes follow the defined state machine,
    records audit log entries for every transition, and provides
    intelligent recommendations for flags that should transition.

    Attributes:
        _service: The FeatureFlagService for flag access and mutation.
    """

    def __init__(self, service: "FeatureFlagService") -> None:  # noqa: F821
        """Initialize the lifecycle manager.

        Args:
            service: The FeatureFlagService instance.
        """
        # Import here to avoid circular imports at module level
        from greenlang.infrastructure.feature_flags.service import FeatureFlagService

        self._service: FeatureFlagService = service
        logger.info("FlagLifecycleManager initialized")

    async def transition_state(
        self,
        flag_key: str,
        new_status: FlagStatus,
        changed_by: str,
        reason: str = "",
    ) -> FeatureFlag:
        """Transition a flag to a new lifecycle state.

        Validates the transition against the state machine, updates the
        flag, and records an audit log entry.

        Args:
            flag_key: The flag key to transition.
            new_status: The target lifecycle status.
            changed_by: Identity of who is making the change.
            reason: Explanation for the transition.

        Returns:
            The updated FeatureFlag.

        Raises:
            ValueError: If the flag does not exist.
            InvalidTransitionError: If the transition is not allowed.
        """
        flag = await self._service.get_flag(flag_key)
        if flag is None:
            raise ValueError(f"Flag '{flag_key}' not found")

        current_status = flag.status

        # Validate the transition
        if not self.is_valid_transition(current_status, new_status):
            raise InvalidTransitionError(
                flag_key=flag_key,
                current_status=current_status,
                target_status=new_status,
            )

        # Perform the transition
        updated = await self._service.update_flag(
            flag_key,
            {"status": new_status},
            updated_by=changed_by,
        )

        logger.info(
            "Flag '%s' transitioned: %s -> %s (by %s, reason: %s)",
            flag_key, current_status.value, new_status.value,
            changed_by, reason or "none",
        )

        return updated

    @staticmethod
    def is_valid_transition(
        current: FlagStatus,
        target: FlagStatus,
    ) -> bool:
        """Check if a status transition is valid.

        Args:
            current: Current flag status.
            target: Target flag status.

        Returns:
            True if the transition is allowed by the state machine.
        """
        if current == target:
            return True  # No-op transition is always valid

        allowed = VALID_TRANSITIONS.get(current, set())
        return target in allowed

    @staticmethod
    def get_valid_transitions(current: FlagStatus) -> List[FlagStatus]:
        """Get all valid target statuses for the current status.

        Args:
            current: Current flag status.

        Returns:
            List of valid target statuses.
        """
        return sorted(
            VALID_TRANSITIONS.get(current, set()),
            key=lambda s: s.value,
        )

    async def recommend_transitions(self) -> List[Dict[str, Any]]:
        """Analyze all flags and recommend lifecycle transitions.

        Recommendations are based on:
            - Flags at 100% rollout for >14 days -> ROLLED_OUT
            - Flags in ACTIVE state for >90 days -> review needed
            - Killed flags for >7 days -> ARCHIVE

        Returns:
            List of recommendation dictionaries with flag_key,
            current_status, recommended_status, reason, and age_days.
        """
        flags = await self._service.list_flags(offset=0, limit=10000)
        now = datetime.now(timezone.utc)
        recommendations: List[Dict[str, Any]] = []

        for flag in flags:
            recommendation = self._evaluate_flag_for_recommendation(flag, now)
            if recommendation is not None:
                recommendations.append(recommendation)

        recommendations.sort(key=lambda r: r.get("priority", 100))

        logger.info(
            "Generated %d lifecycle transition recommendations",
            len(recommendations),
        )
        return recommendations

    @staticmethod
    def _evaluate_flag_for_recommendation(
        flag: FeatureFlag,
        now: datetime,
    ) -> Optional[Dict[str, Any]]:
        """Evaluate a single flag for transition recommendations.

        Args:
            flag: The flag to evaluate.
            now: Current UTC datetime.

        Returns:
            Recommendation dict if applicable, None otherwise.
        """
        age_days = (now - flag.updated_at).days

        # Active flag at 100% rollout for >14 days -> suggest ROLLED_OUT
        if (
            flag.status == FlagStatus.ACTIVE
            and flag.rollout_percentage >= 100.0
            and age_days >= ROLLOUT_PROMOTION_DAYS
        ):
            return {
                "flag_key": flag.key,
                "current_status": flag.status.value,
                "recommended_status": FlagStatus.ROLLED_OUT.value,
                "reason": (
                    f"Flag has been at 100% rollout for {age_days} days "
                    f"(threshold: {ROLLOUT_PROMOTION_DAYS} days). "
                    f"Consider promoting to ROLLED_OUT."
                ),
                "age_days": age_days,
                "priority": 1,
            }

        # Active flag for >90 days -> suggest review
        if (
            flag.status == FlagStatus.ACTIVE
            and age_days >= ACTIVE_REVIEW_DAYS
        ):
            return {
                "flag_key": flag.key,
                "current_status": flag.status.value,
                "recommended_status": "review",
                "reason": (
                    f"Flag has been ACTIVE for {age_days} days "
                    f"(threshold: {ACTIVE_REVIEW_DAYS} days). "
                    f"Consider promoting, archiving, or making permanent."
                ),
                "age_days": age_days,
                "priority": 2,
            }

        # Killed flag for >7 days -> suggest ARCHIVE
        if (
            flag.status == FlagStatus.KILLED
            and age_days >= KILLED_ARCHIVE_DAYS
        ):
            return {
                "flag_key": flag.key,
                "current_status": flag.status.value,
                "recommended_status": FlagStatus.ARCHIVED.value,
                "reason": (
                    f"Flag has been KILLED for {age_days} days "
                    f"(threshold: {KILLED_ARCHIVE_DAYS} days). "
                    f"Consider archiving."
                ),
                "age_days": age_days,
                "priority": 3,
            }

        # Rolled-out flag for >30 days -> suggest PERMANENT or ARCHIVE
        if (
            flag.status == FlagStatus.ROLLED_OUT
            and age_days >= 30
        ):
            return {
                "flag_key": flag.key,
                "current_status": flag.status.value,
                "recommended_status": FlagStatus.PERMANENT.value,
                "reason": (
                    f"Flag has been ROLLED_OUT for {age_days} days. "
                    f"Consider making permanent or archiving."
                ),
                "age_days": age_days,
                "priority": 4,
            }

        return None

    async def get_lifecycle_summary(self) -> Dict[str, Any]:
        """Get a summary of the flag lifecycle state distribution.

        Returns:
            Dictionary with status counts, recommendations count,
            and transition graph.
        """
        flags = await self._service.list_flags(offset=0, limit=10000)
        recommendations = await self.recommend_transitions()

        status_counts: Dict[str, int] = {}
        for flag in flags:
            status_counts[flag.status.value] = (
                status_counts.get(flag.status.value, 0) + 1
            )

        transition_graph = {
            status.value: [t.value for t in targets]
            for status, targets in VALID_TRANSITIONS.items()
        }

        return {
            "total_flags": len(flags),
            "by_status": status_counts,
            "recommendations_count": len(recommendations),
            "transition_graph": transition_graph,
        }


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class InvalidTransitionError(Exception):
    """Raised when a flag lifecycle transition is not valid.

    Attributes:
        flag_key: The flag key.
        current_status: Current status.
        target_status: Attempted target status.
        allowed_targets: Valid target statuses.
    """

    def __init__(
        self,
        flag_key: str,
        current_status: FlagStatus,
        target_status: FlagStatus,
    ) -> None:
        """Initialize the error.

        Args:
            flag_key: The flag key.
            current_status: Current lifecycle status.
            target_status: Attempted target status.
        """
        self.flag_key = flag_key
        self.current_status = current_status
        self.target_status = target_status
        self.allowed_targets = VALID_TRANSITIONS.get(current_status, set())

        allowed_str = ", ".join(
            sorted(s.value for s in self.allowed_targets)
        ) or "none"

        super().__init__(
            f"Invalid lifecycle transition for flag '{flag_key}': "
            f"{current_status.value} -> {target_status.value}. "
            f"Allowed transitions from {current_status.value}: {allowed_str}."
        )


__all__ = [
    "FlagLifecycleManager",
    "InvalidTransitionError",
    "VALID_TRANSITIONS",
]
