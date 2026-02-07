# -*- coding: utf-8 -*-
"""
Compliance Alerts Module - SEC-009 Phase 9

Provides compliance alert condition checking and notification for SOC 2
audit preparation. Monitors readiness scores, SLA compliance, finding status,
attestation deadlines, and control test results.

Alert Conditions:
    - readiness_score_dropped: >5% drop in readiness score
    - sla_breach_imminent: <4h to SLA breach
    - material_finding_unaddressed: >24h material finding open
    - attestation_overdue: >72h pending signature
    - control_test_failed: Critical control test failure

Classes:
    - AlertSeverity: Alert severity levels
    - AlertCondition: Types of alert conditions
    - Alert: Alert data model
    - AlertConfig: Alert configuration thresholds
    - ComplianceAlerts: Main alert checking class

Example:
    >>> alerts = ComplianceAlerts(metrics_collector)
    >>> active_alerts = await alerts.check_alert_conditions()
    >>> for alert in active_alerts:
    ...     await alerts.send_alert(alert)

Author: GreenLang Security Team
Date: February 2026
PRD: SEC-009 SOC 2 Type II Preparation
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    CRITICAL = "critical"
    """Requires immediate action - audit at risk."""

    HIGH = "high"
    """Urgent attention needed - significant impact."""

    MEDIUM = "medium"
    """Should be addressed soon - moderate impact."""

    LOW = "low"
    """Informational - minor impact."""

    INFO = "info"
    """Advisory only - no action required."""


class AlertCondition(str, Enum):
    """Types of compliance alert conditions."""

    READINESS_SCORE_DROPPED = "readiness_score_dropped"
    """Readiness score dropped more than threshold."""

    SLA_BREACH_IMMINENT = "sla_breach_imminent"
    """Auditor request approaching SLA deadline."""

    MATERIAL_FINDING_UNADDRESSED = "material_finding_unaddressed"
    """Critical/high finding not addressed within threshold."""

    ATTESTATION_OVERDUE = "attestation_overdue"
    """Attestation pending signature beyond threshold."""

    CONTROL_TEST_FAILED = "control_test_failed"
    """Control test failure detected."""

    EVIDENCE_COLLECTION_FAILED = "evidence_collection_failed"
    """Evidence collection job failure."""

    ASSESSMENT_OVERDUE = "assessment_overdue"
    """Scheduled assessment not completed on time."""

    PROJECT_MILESTONE_DELAYED = "project_milestone_delayed"
    """Project milestone is past due date."""

    PORTAL_ACCESS_ANOMALY = "portal_access_anomaly"
    """Unusual auditor portal access pattern."""

    REMEDIATION_OVERDUE = "remediation_overdue"
    """Finding remediation past deadline."""

    READINESS_SCORE_LOW = "readiness_score_low"
    """Readiness score below threshold."""


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


class Alert(BaseModel):
    """Alert data model.

    Attributes:
        alert_id: Unique identifier for the alert.
        condition: Type of alert condition.
        severity: Alert severity level.
        title: Short alert title.
        description: Detailed alert description.
        source: Source system or component.
        resource_id: ID of the affected resource.
        resource_type: Type of affected resource.
        threshold: Threshold that was exceeded.
        current_value: Current value that triggered the alert.
        triggered_at: When the alert was triggered.
        acknowledged_at: When the alert was acknowledged.
        resolved_at: When the alert was resolved.
        metadata: Additional alert metadata.
    """

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    alert_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for the alert.",
    )
    condition: AlertCondition = Field(
        ...,
        description="Type of alert condition.",
    )
    severity: AlertSeverity = Field(
        ...,
        description="Alert severity level.",
    )
    title: str = Field(
        ...,
        min_length=1,
        max_length=256,
        description="Short alert title.",
    )
    description: str = Field(
        default="",
        max_length=4096,
        description="Detailed alert description.",
    )
    source: str = Field(
        default="soc2_preparation",
        max_length=128,
        description="Source system or component.",
    )
    resource_id: Optional[str] = Field(
        default=None,
        max_length=256,
        description="ID of the affected resource.",
    )
    resource_type: Optional[str] = Field(
        default=None,
        max_length=128,
        description="Type of affected resource.",
    )
    threshold: Optional[float] = Field(
        default=None,
        description="Threshold that was exceeded.",
    )
    current_value: Optional[float] = Field(
        default=None,
        description="Current value that triggered the alert.",
    )
    triggered_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the alert was triggered.",
    )
    acknowledged_at: Optional[datetime] = Field(
        default=None,
        description="When the alert was acknowledged.",
    )
    resolved_at: Optional[datetime] = Field(
        default=None,
        description="When the alert was resolved.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional alert metadata.",
    )

    @property
    def is_active(self) -> bool:
        """Check if the alert is still active (not resolved)."""
        return self.resolved_at is None


class AlertConfig(BaseModel):
    """Alert configuration thresholds.

    Attributes:
        readiness_drop_threshold: Percentage drop to trigger alert (default 5%).
        sla_warning_hours: Hours before SLA breach to warn (default 4h).
        material_finding_hours: Hours for material finding alert (default 24h).
        attestation_pending_hours: Hours for attestation alert (default 72h).
        readiness_warning_threshold: Readiness score warning level (default 90%).
        readiness_critical_threshold: Readiness score critical level (default 80%).
        remediation_overdue_days: Days before remediation overdue (default 7).
    """

    model_config = ConfigDict(extra="forbid")

    readiness_drop_threshold: float = Field(
        default=5.0,
        ge=1.0,
        le=50.0,
        description="Percentage drop to trigger readiness alert.",
    )
    sla_warning_hours: float = Field(
        default=4.0,
        ge=0.5,
        le=24.0,
        description="Hours before SLA breach to warn.",
    )
    material_finding_hours: float = Field(
        default=24.0,
        ge=1.0,
        le=168.0,
        description="Hours for material finding alert.",
    )
    attestation_pending_hours: float = Field(
        default=72.0,
        ge=24.0,
        le=336.0,
        description="Hours for attestation pending alert.",
    )
    readiness_warning_threshold: float = Field(
        default=90.0,
        ge=50.0,
        le=100.0,
        description="Readiness score warning level.",
    )
    readiness_critical_threshold: float = Field(
        default=80.0,
        ge=30.0,
        le=95.0,
        description="Readiness score critical level.",
    )
    remediation_overdue_days: int = Field(
        default=7,
        ge=1,
        le=30,
        description="Days before remediation overdue.",
    )


# ---------------------------------------------------------------------------
# Compliance Alerts
# ---------------------------------------------------------------------------


class ComplianceAlerts:
    """Compliance alert checking and notification.

    Monitors compliance metrics and generates alerts when conditions
    exceed configured thresholds.

    Attributes:
        metrics_collector: ComplianceMetrics instance for data.
        config: AlertConfig with thresholds.
        _active_alerts: Currently active alerts.
        _notification_handlers: Registered notification handlers.

    Example:
        >>> alerts = ComplianceAlerts(metrics)
        >>> active = await alerts.check_alert_conditions()
        >>> for alert in active:
        ...     await alerts.send_alert(alert)
    """

    def __init__(
        self,
        metrics_collector: Any = None,
        config: Optional[AlertConfig] = None,
    ) -> None:
        """Initialize ComplianceAlerts.

        Args:
            metrics_collector: ComplianceMetrics instance.
            config: AlertConfig with thresholds.
        """
        self.metrics_collector = metrics_collector
        self.config = config or AlertConfig()
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_history: List[Alert] = []
        self._notification_handlers: List[Callable[[Alert], None]] = []

        # State tracking for trend-based alerts
        self._previous_readiness: Optional[float] = None

        logger.info("ComplianceAlerts initialized with config: %s", self.config)

    async def check_alert_conditions(self) -> List[Alert]:
        """Check all alert conditions and return triggered alerts.

        Returns:
            List of Alert objects for conditions that were triggered.
        """
        start_time = datetime.now(timezone.utc)
        triggered: List[Alert] = []

        # Check each condition
        triggered.extend(await self._check_readiness_score())
        triggered.extend(await self._check_sla_breach_imminent())
        triggered.extend(await self._check_material_findings())
        triggered.extend(await self._check_attestation_overdue())
        triggered.extend(await self._check_control_test_failures())

        # Store active alerts
        for alert in triggered:
            self._active_alerts[alert.alert_id] = alert
            self._alert_history.append(alert)

        elapsed_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        logger.info(
            "Alert conditions checked: triggered=%d, active=%d, elapsed=%.2fms",
            len(triggered),
            len(self._active_alerts),
            elapsed_ms,
        )

        return triggered

    async def send_alert(self, alert: Alert) -> None:
        """Send an alert through registered notification handlers.

        Args:
            alert: Alert to send.
        """
        logger.warning(
            "ALERT [%s] %s: %s",
            alert.severity.value.upper(),
            alert.condition.value,
            alert.title,
        )

        # Call registered handlers
        for handler in self._notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error("Alert handler failed: %s", str(e))

    def configure_thresholds(self, config: AlertConfig) -> None:
        """Update alert configuration thresholds.

        Args:
            config: New AlertConfig.
        """
        self.config = config
        logger.info("Alert thresholds updated: %s", config)

    def register_handler(self, handler: Callable[[Alert], None]) -> None:
        """Register a notification handler for alerts.

        Args:
            handler: Callable that accepts an Alert.
        """
        self._notification_handlers.append(handler)
        logger.debug("Alert handler registered: %s", handler.__name__)

    async def acknowledge_alert(self, alert_id: str) -> None:
        """Acknowledge an active alert.

        Args:
            alert_id: ID of the alert to acknowledge.

        Raises:
            ValueError: If alert not found.
        """
        alert = self._active_alerts.get(alert_id)
        if alert is None:
            raise ValueError(f"Alert '{alert_id}' not found.")

        alert.acknowledged_at = datetime.now(timezone.utc)
        logger.info("Alert acknowledged: id=%s", alert_id)

    async def resolve_alert(self, alert_id: str) -> None:
        """Resolve an active alert.

        Args:
            alert_id: ID of the alert to resolve.

        Raises:
            ValueError: If alert not found.
        """
        alert = self._active_alerts.get(alert_id)
        if alert is None:
            raise ValueError(f"Alert '{alert_id}' not found.")

        alert.resolved_at = datetime.now(timezone.utc)
        del self._active_alerts[alert_id]
        logger.info("Alert resolved: id=%s", alert_id)

    def get_active_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
    ) -> List[Alert]:
        """Get currently active alerts.

        Args:
            severity: Filter by severity level.

        Returns:
            List of active Alert objects.
        """
        alerts = list(self._active_alerts.values())

        if severity is not None:
            alerts = [a for a in alerts if a.severity == severity]

        # Sort by severity then triggered time
        severity_order = {
            AlertSeverity.CRITICAL: 0,
            AlertSeverity.HIGH: 1,
            AlertSeverity.MEDIUM: 2,
            AlertSeverity.LOW: 3,
            AlertSeverity.INFO: 4,
        }
        alerts.sort(
            key=lambda a: (severity_order.get(a.severity, 99), a.triggered_at)
        )

        return alerts

    def get_alert_history(
        self,
        days: int = 30,
        condition: Optional[AlertCondition] = None,
    ) -> List[Alert]:
        """Get alert history.

        Args:
            days: Number of days to include.
            condition: Filter by alert condition.

        Returns:
            List of historical Alert objects.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        history = [a for a in self._alert_history if a.triggered_at >= cutoff]

        if condition is not None:
            history = [a for a in history if a.condition == condition]

        history.sort(key=lambda a: a.triggered_at, reverse=True)
        return history

    # -----------------------------------------------------------------------
    # Condition Checkers
    # -----------------------------------------------------------------------

    async def _check_readiness_score(self) -> List[Alert]:
        """Check readiness score conditions."""
        alerts = []

        if not self.metrics_collector:
            return alerts

        current = await self.metrics_collector.calculate_readiness_score()

        # Check for drop
        if self._previous_readiness is not None:
            drop = self._previous_readiness - current
            if drop >= self.config.readiness_drop_threshold:
                alerts.append(
                    Alert(
                        condition=AlertCondition.READINESS_SCORE_DROPPED,
                        severity=AlertSeverity.HIGH,
                        title=f"Readiness score dropped {drop:.1f}%",
                        description=(
                            f"SOC 2 readiness score dropped from "
                            f"{self._previous_readiness:.1f}% to {current:.1f}% "
                            f"(threshold: {self.config.readiness_drop_threshold}%)"
                        ),
                        threshold=self.config.readiness_drop_threshold,
                        current_value=drop,
                        metadata={"previous": self._previous_readiness, "current": current},
                    )
                )

        # Check absolute thresholds
        if current < self.config.readiness_critical_threshold:
            alerts.append(
                Alert(
                    condition=AlertCondition.READINESS_SCORE_LOW,
                    severity=AlertSeverity.CRITICAL,
                    title=f"Readiness score critical: {current:.1f}%",
                    description=(
                        f"SOC 2 readiness score is {current:.1f}%, below the "
                        f"critical threshold of {self.config.readiness_critical_threshold}%"
                    ),
                    threshold=self.config.readiness_critical_threshold,
                    current_value=current,
                )
            )
        elif current < self.config.readiness_warning_threshold:
            alerts.append(
                Alert(
                    condition=AlertCondition.READINESS_SCORE_LOW,
                    severity=AlertSeverity.MEDIUM,
                    title=f"Readiness score below target: {current:.1f}%",
                    description=(
                        f"SOC 2 readiness score is {current:.1f}%, below the "
                        f"warning threshold of {self.config.readiness_warning_threshold}%"
                    ),
                    threshold=self.config.readiness_warning_threshold,
                    current_value=current,
                )
            )

        self._previous_readiness = current
        return alerts

    async def _check_sla_breach_imminent(self) -> List[Alert]:
        """Check for auditor requests approaching SLA deadline."""
        alerts = []

        if not self.metrics_collector:
            return alerts

        # Check for requests close to SLA
        for request in self.metrics_collector._requests:
            if request.get("status") == "pending":
                created = request.get("created_at")
                priority = request.get("priority", "normal")

                if created:
                    if isinstance(created, str):
                        created = datetime.fromisoformat(created.replace("Z", "+00:00"))

                    # Get SLA threshold
                    sla_hours = {
                        "critical": 4,
                        "high": 24,
                        "normal": 48,
                        "low": 72,
                    }.get(priority, 48)

                    deadline = created + timedelta(hours=sla_hours)
                    now = datetime.now(timezone.utc)
                    hours_remaining = (deadline - now).total_seconds() / 3600

                    if 0 < hours_remaining <= self.config.sla_warning_hours:
                        alerts.append(
                            Alert(
                                condition=AlertCondition.SLA_BREACH_IMMINENT,
                                severity=AlertSeverity.HIGH,
                                title=f"SLA breach in {hours_remaining:.1f}h",
                                description=(
                                    f"Auditor request '{request.get('title', 'Unknown')}' "
                                    f"will breach SLA in {hours_remaining:.1f} hours"
                                ),
                                resource_id=request.get("request_id"),
                                resource_type="auditor_request",
                                threshold=self.config.sla_warning_hours,
                                current_value=hours_remaining,
                            )
                        )

        return alerts

    async def _check_material_findings(self) -> List[Alert]:
        """Check for material findings not addressed within threshold."""
        alerts = []

        if not self.metrics_collector:
            return alerts

        now = datetime.now(timezone.utc)
        threshold_hours = self.config.material_finding_hours

        for finding in self.metrics_collector._findings:
            severity = finding.get("severity")
            status = finding.get("status")

            if severity in ("critical", "high") and status in ("open", "in_remediation"):
                created = finding.get("created_at")

                if created:
                    if isinstance(created, str):
                        created = datetime.fromisoformat(created.replace("Z", "+00:00"))

                    hours_open = (now - created).total_seconds() / 3600

                    if hours_open > threshold_hours:
                        alerts.append(
                            Alert(
                                condition=AlertCondition.MATERIAL_FINDING_UNADDRESSED,
                                severity=AlertSeverity.CRITICAL
                                if severity == "critical"
                                else AlertSeverity.HIGH,
                                title=f"{severity.title()} finding open {hours_open:.0f}h",
                                description=(
                                    f"{severity.title()} finding "
                                    f"'{finding.get('title', 'Unknown')}' has been open "
                                    f"for {hours_open:.0f} hours (threshold: {threshold_hours}h)"
                                ),
                                resource_id=finding.get("finding_id"),
                                resource_type="finding",
                                threshold=threshold_hours,
                                current_value=hours_open,
                            )
                        )

        return alerts

    async def _check_attestation_overdue(self) -> List[Alert]:
        """Check for attestations pending signature beyond threshold."""
        alerts = []

        if not self.metrics_collector:
            return alerts

        now = datetime.now(timezone.utc)
        threshold_hours = self.config.attestation_pending_hours

        for att in self.metrics_collector._attestations:
            status = att.get("status")

            if status == "pending_signature":
                sent_at = att.get("sent_at")

                if sent_at:
                    if isinstance(sent_at, str):
                        sent_at = datetime.fromisoformat(sent_at.replace("Z", "+00:00"))

                    hours_pending = (now - sent_at).total_seconds() / 3600

                    if hours_pending > threshold_hours:
                        alerts.append(
                            Alert(
                                condition=AlertCondition.ATTESTATION_OVERDUE,
                                severity=AlertSeverity.HIGH,
                                title=f"Attestation pending {hours_pending:.0f}h",
                                description=(
                                    f"Attestation '{att.get('name', 'Unknown')}' "
                                    f"has been pending signature for {hours_pending:.0f} hours "
                                    f"(threshold: {threshold_hours}h)"
                                ),
                                resource_id=att.get("attestation_id"),
                                resource_type="attestation",
                                threshold=threshold_hours,
                                current_value=hours_pending,
                            )
                        )

        return alerts

    async def _check_control_test_failures(self) -> List[Alert]:
        """Check for control test failures."""
        alerts = []

        if not self.metrics_collector:
            return alerts

        for test in self.metrics_collector._test_results:
            if test.get("result") == "fail":
                control_id = test.get("control_id", "Unknown")
                criterion = test.get("criterion", "Unknown")

                # Determine severity based on control
                is_critical = control_id.startswith("CC6") or control_id.startswith("CC7")

                alerts.append(
                    Alert(
                        condition=AlertCondition.CONTROL_TEST_FAILED,
                        severity=AlertSeverity.CRITICAL if is_critical else AlertSeverity.HIGH,
                        title=f"Control test failed: {control_id}",
                        description=(
                            f"Control test for {control_id} ({criterion}) failed. "
                            f"Reason: {test.get('failure_reason', 'Unknown')}"
                        ),
                        resource_id=test.get("test_id"),
                        resource_type="control_test",
                        metadata={"control_id": control_id, "criterion": criterion},
                    )
                )

        return alerts


__all__ = [
    "AlertSeverity",
    "AlertCondition",
    "Alert",
    "AlertConfig",
    "ComplianceAlerts",
]
