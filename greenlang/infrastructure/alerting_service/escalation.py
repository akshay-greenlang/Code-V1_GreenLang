# -*- coding: utf-8 -*-
"""
Escalation Engine - OBS-004: Unified Alerting Service

Automatically escalates unacknowledged alerts through a configurable
policy of notification steps. Each severity level has a default policy
and custom policies can be registered at runtime.

Example:
    >>> engine = EscalationEngine(config, lifecycle, router, oncall)
    >>> await engine.check_escalations()

Author: GreenLang Platform Team
Date: February 2026
PRD: OBS-004 Unified Alerting Service
Status: Production Ready
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from greenlang.infrastructure.alerting_service.config import AlertingConfig
from greenlang.infrastructure.alerting_service.lifecycle import AlertLifecycle
from greenlang.infrastructure.alerting_service.metrics import record_escalation
from greenlang.infrastructure.alerting_service.models import (
    Alert,
    AlertSeverity,
    AlertStatus,
    EscalationPolicy,
    EscalationStep,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default Policies
# ---------------------------------------------------------------------------

_CRITICAL_DEFAULT = EscalationPolicy(
    name="critical_default",
    steps=[
        EscalationStep(
            delay_minutes=0,
            channels=["pagerduty", "opsgenie", "slack"],
            repeat=1,
        ),
        EscalationStep(
            delay_minutes=15,
            channels=["pagerduty", "opsgenie", "slack", "email"],
            repeat=1,
        ),
        EscalationStep(
            delay_minutes=30,
            channels=["pagerduty", "opsgenie", "slack", "email", "teams"],
            repeat=2,
        ),
        EscalationStep(
            delay_minutes=60,
            channels=["pagerduty", "opsgenie", "slack", "email", "teams"],
            oncall_schedule_id="management",
            repeat=1,
        ),
    ],
)

_WARNING_DEFAULT = EscalationPolicy(
    name="warning_default",
    steps=[
        EscalationStep(
            delay_minutes=0,
            channels=["slack", "email"],
            repeat=1,
        ),
        EscalationStep(
            delay_minutes=60,
            channels=["slack", "email", "pagerduty"],
            repeat=1,
        ),
    ],
)


# ---------------------------------------------------------------------------
# EscalationEngine
# ---------------------------------------------------------------------------


class EscalationEngine:
    """Automatic alert escalation engine.

    Scans active FIRING alerts and escalates unacknowledged ones
    through the appropriate escalation policy.

    Attributes:
        config: AlertingConfig instance.
        lifecycle: AlertLifecycle for state queries.
        router: AlertRouter for notification dispatch.
        oncall: OnCallManager for responder lookups.
    """

    def __init__(
        self,
        config: AlertingConfig,
        lifecycle: AlertLifecycle,
        router: Any,  # AlertRouter (avoid circular import)
        oncall: Any,  # OnCallManager (avoid circular import)
    ) -> None:
        self.config = config
        self.lifecycle = lifecycle
        self.router = router
        self.oncall = oncall
        self._policies: Dict[str, EscalationPolicy] = {
            "critical_default": _CRITICAL_DEFAULT,
            "warning_default": _WARNING_DEFAULT,
        }
        logger.info(
            "EscalationEngine initialized: ack_timeout=%dm, policies=%d",
            config.escalation_ack_timeout_minutes,
            len(self._policies),
        )

    # ------------------------------------------------------------------
    # Policy management
    # ------------------------------------------------------------------

    def add_policy(self, policy: EscalationPolicy) -> None:
        """Register a custom escalation policy.

        Args:
            policy: EscalationPolicy to register.
        """
        self._policies[policy.name] = policy
        logger.info("Escalation policy registered: %s", policy.name)

    def get_policy(self, name: str) -> Optional[EscalationPolicy]:
        """Retrieve a policy by name.

        Args:
            name: Policy name.

        Returns:
            EscalationPolicy or None.
        """
        return self._policies.get(name)

    # ------------------------------------------------------------------
    # Escalation check
    # ------------------------------------------------------------------

    async def check_escalations(self) -> List[Alert]:
        """Scan active alerts and escalate those that are overdue.

        Iterates all FIRING alerts and checks if they have exceeded the
        acknowledgement timeout at their current escalation level.

        Returns:
            List of alerts that were escalated.
        """
        if not self.config.escalation_enabled:
            return []

        escalated: List[Alert] = []
        firing = self.lifecycle.list_alerts(status=AlertStatus.FIRING, limit=1000)

        for alert in firing:
            policy = self._get_policy_for_alert(alert)
            if policy is None:
                continue

            should, step = self._should_escalate(alert, policy)
            if should and step is not None:
                try:
                    result = await self.escalate(
                        alert.alert_id,
                        reason=f"Unacknowledged for >{step.delay_minutes}m",
                        step=step,
                    )
                    if result is not None:
                        escalated.append(result)
                except Exception as exc:
                    logger.error(
                        "Escalation failed: alert=%s, error=%s",
                        alert.alert_id[:8], exc,
                    )

        if escalated:
            logger.info("Escalation check: %d alerts escalated", len(escalated))
        return escalated

    async def escalate(
        self,
        alert_id: str,
        reason: str = "",
        step: Optional[EscalationStep] = None,
    ) -> Optional[Alert]:
        """Manually or automatically escalate an alert.

        Increments the escalation level and dispatches notifications to
        the channels defined in the next escalation step.

        Args:
            alert_id: Alert to escalate.
            reason: Free-text reason.
            step: Explicit step (if from check_escalations).

        Returns:
            Updated Alert or None on failure.
        """
        alert = self.lifecycle.get_alert(alert_id)
        if alert is None:
            logger.warning("Escalation target not found: %s", alert_id)
            return None

        alert.escalation_level += 1
        level_str = str(alert.escalation_level)

        # Determine channels from step or policy
        if step is None:
            policy = self._get_policy_for_alert(alert)
            if policy and alert.escalation_level - 1 < len(policy.steps):
                step = policy.steps[alert.escalation_level - 1]

        channels = step.channels if step else ["pagerduty", "slack", "email"]

        # Dispatch notifications
        results = await self.router.notify(alert, channels)

        # Look up on-call if step specifies a schedule
        if step and step.oncall_schedule_id and self.oncall:
            try:
                oncall_user = await self.oncall.get_current_oncall(
                    step.oncall_schedule_id,
                )
                if oncall_user:
                    alert.annotations["escalated_to"] = oncall_user.name
            except Exception as exc:
                logger.warning("On-call lookup during escalation failed: %s", exc)

        policy_name = self._get_policy_for_alert(alert)
        record_escalation(
            level=level_str,
            policy=policy_name.name if policy_name else "unknown",
        )

        logger.info(
            "Alert escalated: alert=%s, level=%d, reason=%s, channels=%s",
            alert_id[:8], alert.escalation_level, reason, channels,
        )
        return alert

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_policy_for_alert(self, alert: Alert) -> Optional[EscalationPolicy]:
        """Select the escalation policy for an alert.

        Custom team/service policies override defaults.

        Args:
            alert: Alert to look up policy for.

        Returns:
            EscalationPolicy or None.
        """
        # Check for team-specific or service-specific policy
        team_policy = self._policies.get(f"{alert.team}_policy")
        if team_policy:
            return team_policy

        service_policy = self._policies.get(f"{alert.service}_policy")
        if service_policy:
            return service_policy

        # Default by severity
        if alert.severity == AlertSeverity.CRITICAL:
            return self._policies.get("critical_default")
        if alert.severity == AlertSeverity.WARNING:
            return self._policies.get("warning_default")

        return None

    def _should_escalate(
        self,
        alert: Alert,
        policy: EscalationPolicy,
    ) -> Tuple[bool, Optional[EscalationStep]]:
        """Determine if an alert should be escalated.

        Args:
            alert: The alert to check.
            policy: The applicable escalation policy.

        Returns:
            Tuple of (should_escalate, next_step).
        """
        if alert.escalation_level >= len(policy.steps):
            return False, None  # Already at max level

        next_step = policy.steps[alert.escalation_level]
        elapsed_minutes = 0.0

        if alert.fired_at:
            elapsed = datetime.now(timezone.utc) - alert.fired_at
            elapsed_minutes = elapsed.total_seconds() / 60.0

        if elapsed_minutes >= next_step.delay_minutes:
            return True, next_step

        return False, None
