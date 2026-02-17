# -*- coding: utf-8 -*-
"""
Alert Manager Engine - AGENT-DATA-016: Data Freshness Monitor (GL-DATA-X-019)

Engine 6 of the Data Freshness Monitor agent. Generates and manages alerts
for SLA breaches, including throttling, deduplication, escalation, and
lifecycle management (create, send, acknowledge, resolve, suppress).

Manages two primary entities:
    - **FreshnessAlert**: Notifications sent through channels (email, Slack,
      PagerDuty, webhook, Teams, SMS) when SLA breaches are detected.
    - **SLABreach**: Records of SLA violations including severity, detection
      time, acknowledgement, and resolution tracking.

Zero-Hallucination Guarantees:
    - All alert decisions are deterministic (time-based throttling/dedup)
    - Escalation levels are resolved by comparing elapsed minutes to policy
    - No LLM or ML model calls in the alerting path
    - SHA-256 provenance hashes on every alert and breach mutation
    - Thread-safe in-memory storage with Lock

Alert Lifecycle:
    pending -> sent -> acknowledged -> resolved
                   \-> suppressed

Breach Lifecycle:
    active -> acknowledged -> resolved

Example:
    >>> from greenlang.data_freshness_monitor.alert_manager import AlertManagerEngine
    >>> engine = AlertManagerEngine()
    >>> breach = engine.record_breach("ds-001", "sla-001", "critical", 96.5)
    >>> alert = engine.create_and_send_alert(
    ...     breach_id=breach.breach_id,
    ...     dataset_id="ds-001",
    ...     alert_severity="critical",
    ...     channel="email",
    ...     message="Dataset ds-001 is 96.5h old (SLA: 72h)",
    ...     recipients=["oncall@example.com"],
    ... )
    >>> acknowledged = engine.acknowledge_alert(alert.alert_id, "ops-engineer")
    >>> resolved = engine.resolve_alert(alert.alert_id, "Pipeline restarted")

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-016 Data Freshness Monitor (GL-DATA-X-019)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from greenlang.data_freshness_monitor.config import get_config
from greenlang.data_freshness_monitor.models import (
    AlertChannel,
    AlertSeverity,
    AlertStatus,
    BreachSeverity,
    BreachStatus,
    EscalationPolicy,
    FreshnessAlert,
    SLABreach,
)

logger = logging.getLogger(__name__)

__all__ = [
    "AlertManagerEngine",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _generate_id(prefix: str = "ALT") -> str:
    """Generate a unique identifier with the given prefix.

    Args:
        prefix: ID prefix string.

    Returns:
        String of the form ``{prefix}-{hex12}``.
    """
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _compute_provenance(operation: str, data_repr: str) -> str:
    """Compute SHA-256 provenance hash for an alert manager operation.

    Args:
        operation: Name of the operation (e.g. ``create_alert``).
        data_repr: Serialised representation of the data involved.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    payload = f"{operation}:{data_repr}:{_utcnow().isoformat()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _safe_inc(metric: Any, **labels: str) -> None:
    """Safely increment a Prometheus counter if available.

    Args:
        metric: Counter metric or None.
        **labels: Label key-value pairs.
    """
    if metric is None:
        return
    try:
        metric.labels(**labels).inc()
    except Exception:  # noqa: BLE001
        pass


def _safe_observe(metric: Any, value: float, **labels: str) -> None:
    """Safely observe a Prometheus histogram value if available.

    Args:
        metric: Histogram metric or None.
        value: Value to observe.
        **labels: Label key-value pairs.
    """
    if metric is None:
        return
    try:
        if labels:
            metric.labels(**labels).observe(value)
        else:
            metric.observe(value)
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# Prometheus metrics (graceful import)
# ---------------------------------------------------------------------------

try:
    from prometheus_client import Counter, Gauge, Histogram

    _PROM_AVAILABLE = True
except ImportError:
    _PROM_AVAILABLE = False
    Counter = None  # type: ignore[assignment,misc]
    Gauge = None  # type: ignore[assignment,misc]
    Histogram = None  # type: ignore[assignment,misc]
    logger.info(
        "prometheus_client not installed; alert manager metrics disabled"
    )

# Import shared metrics from centralized metrics.py to avoid duplicate
# Prometheus registrations.  Only create *new* metrics unique to this engine.
try:
    from greenlang.data_freshness_monitor.metrics import (
        dfm_alerts_sent_total as _alerts_sent_total,
        dfm_active_breaches as _active_breaches_gauge,
        dfm_sla_breaches_total as _breaches_recorded_total,
        dfm_processing_duration_seconds as _alert_processing_duration,
    )
except ImportError:
    _alerts_sent_total = None  # type: ignore[assignment]
    _active_breaches_gauge = None  # type: ignore[assignment]
    _breaches_recorded_total = None  # type: ignore[assignment]
    _alert_processing_duration = None  # type: ignore[assignment]

# Alert-specific metrics (unique names, no conflict with metrics.py)
if _PROM_AVAILABLE:
    _alerts_created_total = Counter(
        "gl_dfm_am_alerts_created_total",
        "Total freshness alerts created",
        labelnames=["severity", "channel"],
    )
    _alerts_acknowledged_total = Counter(
        "gl_dfm_am_alerts_acknowledged_total",
        "Total freshness alerts acknowledged",
        labelnames=["severity"],
    )
    _alerts_resolved_total = Counter(
        "gl_dfm_am_alerts_resolved_total",
        "Total freshness alerts resolved",
        labelnames=["severity"],
    )
    _alerts_suppressed_total = Counter(
        "gl_dfm_am_alerts_suppressed_total",
        "Total freshness alerts suppressed",
        labelnames=["severity"],
    )
    _alerts_throttled_total = Counter(
        "gl_dfm_am_alerts_throttled_total",
        "Total freshness alerts throttled",
        labelnames=["channel"],
    )
    _alerts_deduplicated_total = Counter(
        "gl_dfm_am_alerts_deduplicated_total",
        "Total freshness alerts deduplicated",
        labelnames=["severity"],
    )
    _escalations_total = Counter(
        "gl_dfm_am_escalations_total",
        "Total alert escalations performed",
        labelnames=["channel"],
    )
    _active_alerts_gauge = Gauge(
        "gl_dfm_am_active_alerts",
        "Number of currently active (non-resolved) alerts",
    )
else:
    _alerts_created_total = None  # type: ignore[assignment]
    _alerts_acknowledged_total = None  # type: ignore[assignment]
    _alerts_resolved_total = None  # type: ignore[assignment]
    _alerts_suppressed_total = None  # type: ignore[assignment]
    _alerts_throttled_total = None  # type: ignore[assignment]
    _alerts_deduplicated_total = None  # type: ignore[assignment]
    _escalations_total = None  # type: ignore[assignment]
    _active_alerts_gauge = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Default alert template
# ---------------------------------------------------------------------------

_DEFAULT_ALERT_TEMPLATE: str = (
    "[{severity}] Dataset '{dataset_name}' SLA breach: "
    "data is {age_hours:.1f}h old (SLA: {sla_hours}h)"
)


# ---------------------------------------------------------------------------
# AlertManagerEngine
# ---------------------------------------------------------------------------


class AlertManagerEngine:
    """Generates and manages alerts for SLA breaches.

    Provides alert creation, sending, acknowledgement, resolution, and
    suppression. Implements throttling per (dataset_id, channel) and
    deduplication per (dataset_id, severity, channel) to prevent alert
    fatigue. Supports multi-level escalation policies that promote
    unresolved breaches through progressively higher-urgency channels.

    Internal Storage:
        _alerts: Dict mapping alert_id to FreshnessAlert.
        _breaches: Dict mapping breach_id to SLABreach.
        _last_alert_times: Dict mapping (dataset_id, channel) to the
            last datetime an alert was sent for that combination.
        _recent_alerts: Dict mapping (dataset_id, severity, channel) to
            the last datetime an alert was sent for dedup tracking.
        _escalation_levels: Dict mapping breach_id to the current
            escalation level that has been reached.
        _lock: Threading lock for all mutation operations.

    Attributes:
        _config: DataFreshnessMonitorConfig for throttle/dedup settings.

    Example:
        >>> engine = AlertManagerEngine()
        >>> breach = engine.record_breach("ds-001", "sla-001", "critical", 96.5)
        >>> alert = engine.create_and_send_alert(
        ...     breach_id=breach.breach_id,
        ...     dataset_id="ds-001",
        ...     alert_severity="critical",
        ...     channel="email",
        ...     message="SLA breached",
        ...     recipients=["oncall@acme.com"],
        ... )
        >>> engine.acknowledge_alert(alert.alert_id, "ops-engineer")
    """

    def __init__(
        self,
        config: Optional[Any] = None,
    ) -> None:
        """Initialize AlertManagerEngine.

        Args:
            config: Optional DataFreshnessMonitorConfig instance. If None,
                uses the module-level singleton from ``get_config()``.
        """
        self._config = config or get_config()

        # Primary storage
        self._alerts: Dict[str, FreshnessAlert] = {}
        self._breaches: Dict[str, SLABreach] = {}

        # Throttle / dedup tracking
        self._last_alert_times: Dict[Tuple[str, str], datetime] = {}
        self._recent_alerts: Dict[Tuple[str, str, str], datetime] = {}

        # Escalation tracking
        self._escalation_levels: Dict[str, int] = {}

        # Thread safety
        self._lock = threading.Lock()

        logger.info(
            "AlertManagerEngine initialized (throttle=%dmin, dedup=%dh)",
            self._config.alert_throttle_minutes,
            self._config.alert_dedup_window_hours,
        )

    # ------------------------------------------------------------------
    # Alert CRUD
    # ------------------------------------------------------------------

    def create_alert(
        self,
        breach_id: str,
        dataset_id: str,
        alert_severity: str,
        channel: str,
        message: str,
        recipients: Optional[List[str]] = None,
    ) -> FreshnessAlert:
        """Create a new freshness alert without sending it.

        Args:
            breach_id: Identifier of the associated SLA breach.
            dataset_id: Identifier of the affected dataset.
            alert_severity: Alert severity level (info, warning, critical,
                emergency). Must be a valid ``AlertSeverity`` value.
            channel: Delivery channel (email, slack, pagerduty, webhook,
                teams, sms). Must be a valid ``AlertChannel`` value.
            message: Human-readable alert message body.
            recipients: Optional list of recipient addresses/identifiers.

        Returns:
            Newly created FreshnessAlert with status ``pending``.

        Raises:
            ValueError: If severity or channel is invalid.
        """
        start = time.monotonic()
        self._validate_severity(alert_severity)
        self._validate_channel(channel)

        alert_id = _generate_id("ALT")
        now = _utcnow()
        provenance_hash = _compute_provenance(
            "create_alert",
            json.dumps({
                "alert_id": alert_id,
                "breach_id": breach_id,
                "dataset_id": dataset_id,
                "severity": alert_severity,
                "channel": channel,
            }, sort_keys=True),
        )

        alert = FreshnessAlert(
            alert_id=alert_id,
            breach_id=breach_id,
            dataset_id=dataset_id,
            severity=AlertSeverity(alert_severity),
            channel=AlertChannel(channel),
            status=AlertStatus.PENDING,
            message=message,
            recipients=recipients or [],
            created_at=now,
            sent_at=None,
            acknowledged_at=None,
            acknowledged_by=None,
            resolved_at=None,
            resolution_notes=None,
            suppressed_at=None,
            suppression_reason=None,
            escalation_level=0,
            provenance_hash=provenance_hash,
        )

        with self._lock:
            self._alerts[alert_id] = alert

        _safe_inc(
            _alerts_created_total,
            severity=alert_severity,
            channel=channel,
        )
        elapsed = time.monotonic() - start
        _safe_observe(
            _alert_processing_duration, elapsed, operation="create_alert",
        )

        logger.info(
            "Created alert %s for breach %s dataset %s [%s/%s]",
            alert_id, breach_id, dataset_id, alert_severity, channel,
        )
        return alert

    def send_alert(self, alert_id: str) -> bool:
        """Simulate sending an alert and mark it as sent.

        In production this would dispatch to the appropriate channel
        integration (SMTP, Slack API, PagerDuty Events API, etc.).
        This implementation marks the alert as ``sent`` and updates
        throttle/dedup tracking timestamps.

        Args:
            alert_id: Identifier of the alert to send.

        Returns:
            True if the alert was sent, False if not found or already
            in a terminal state (resolved, suppressed).

        Raises:
            ValueError: If alert_id is not found.
        """
        start = time.monotonic()

        with self._lock:
            alert = self._alerts.get(alert_id)
            if alert is None:
                raise ValueError(f"Alert not found: {alert_id}")

            if alert.status in (AlertStatus.RESOLVED, AlertStatus.SUPPRESSED):
                logger.warning(
                    "Cannot send alert %s in status %s",
                    alert_id, alert.status.value,
                )
                return False

            now = _utcnow()
            alert.status = AlertStatus.SENT
            alert.sent_at = now
            alert.provenance_hash = _compute_provenance(
                "send_alert",
                json.dumps({
                    "alert_id": alert_id,
                    "sent_at": now.isoformat(),
                }, sort_keys=True),
            )

            # Update throttle tracking
            throttle_key = (alert.dataset_id, alert.channel.value)
            self._last_alert_times[throttle_key] = now

            # Update dedup tracking
            dedup_key = (
                alert.dataset_id,
                alert.severity.value,
                alert.channel.value,
            )
            self._recent_alerts[dedup_key] = now

        _safe_inc(_alerts_sent_total, channel=alert.channel.value)
        self._update_active_alerts_gauge()

        elapsed = time.monotonic() - start
        _safe_observe(
            _alert_processing_duration, elapsed, operation="send_alert",
        )

        logger.info("Sent alert %s via %s", alert_id, alert.channel.value)
        return True

    def create_and_send_alert(
        self,
        breach_id: str,
        dataset_id: str,
        alert_severity: str,
        channel: str,
        message: str,
        recipients: Optional[List[str]] = None,
    ) -> FreshnessAlert:
        """Create a new alert and immediately send it.

        Combines ``create_alert`` and ``send_alert`` into a single
        convenience method. Respects throttling and deduplication:
        if the alert would be throttled or deduplicated, the alert is
        still created but its status remains ``pending`` (not sent).

        Args:
            breach_id: Identifier of the associated SLA breach.
            dataset_id: Identifier of the affected dataset.
            alert_severity: Severity level string.
            channel: Delivery channel string.
            message: Alert message body.
            recipients: Optional recipient list.

        Returns:
            FreshnessAlert in ``sent`` status (or ``pending`` if throttled
            or deduplicated).
        """
        alert = self.create_alert(
            breach_id=breach_id,
            dataset_id=dataset_id,
            alert_severity=alert_severity,
            channel=channel,
            message=message,
            recipients=recipients,
        )

        # Check throttle and dedup before sending
        throttled = self.should_throttle(dataset_id, channel)
        deduped = self.should_deduplicate(dataset_id, alert_severity, channel)

        if throttled:
            _safe_inc(_alerts_throttled_total, channel=channel)
            logger.info(
                "Alert %s throttled for dataset %s channel %s",
                alert.alert_id, dataset_id, channel,
            )
            return alert

        if deduped:
            _safe_inc(
                _alerts_deduplicated_total, severity=alert_severity,
            )
            logger.info(
                "Alert %s deduplicated for dataset %s severity %s channel %s",
                alert.alert_id, dataset_id, alert_severity, channel,
            )
            return alert

        self.send_alert(alert.alert_id)
        # Re-fetch the alert to get the updated state
        with self._lock:
            return self._alerts[alert.alert_id]

    def acknowledge_alert(
        self,
        alert_id: str,
        acknowledged_by: str,
    ) -> FreshnessAlert:
        """Acknowledge an alert.

        Transitions alert from ``sent`` (or ``pending``) to
        ``acknowledged`` state, recording who acknowledged it and when.

        Args:
            alert_id: Identifier of the alert to acknowledge.
            acknowledged_by: Name or ID of the acknowledging user.

        Returns:
            Updated FreshnessAlert with ``acknowledged`` status.

        Raises:
            ValueError: If alert not found or already resolved/suppressed.
        """
        start = time.monotonic()

        with self._lock:
            alert = self._alerts.get(alert_id)
            if alert is None:
                raise ValueError(f"Alert not found: {alert_id}")

            if alert.status in (AlertStatus.RESOLVED, AlertStatus.SUPPRESSED):
                raise ValueError(
                    f"Cannot acknowledge alert {alert_id} "
                    f"in status {alert.status.value}"
                )

            now = _utcnow()
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = now
            alert.acknowledged_by = acknowledged_by
            alert.provenance_hash = _compute_provenance(
                "acknowledge_alert",
                json.dumps({
                    "alert_id": alert_id,
                    "acknowledged_by": acknowledged_by,
                    "acknowledged_at": now.isoformat(),
                }, sort_keys=True),
            )

        _safe_inc(
            _alerts_acknowledged_total, severity=alert.severity.value,
        )

        elapsed = time.monotonic() - start
        _safe_observe(
            _alert_processing_duration, elapsed,
            operation="acknowledge_alert",
        )

        logger.info(
            "Alert %s acknowledged by %s", alert_id, acknowledged_by,
        )
        return alert

    def resolve_alert(
        self,
        alert_id: str,
        resolution_notes: str,
    ) -> FreshnessAlert:
        """Resolve an alert.

        Transitions alert to ``resolved`` state with resolution notes.
        Can be called from any non-suppressed state.

        Args:
            alert_id: Identifier of the alert to resolve.
            resolution_notes: Free-text description of how the issue
                was resolved.

        Returns:
            Updated FreshnessAlert with ``resolved`` status.

        Raises:
            ValueError: If alert not found or already suppressed.
        """
        start = time.monotonic()

        with self._lock:
            alert = self._alerts.get(alert_id)
            if alert is None:
                raise ValueError(f"Alert not found: {alert_id}")

            if alert.status == AlertStatus.SUPPRESSED:
                raise ValueError(
                    f"Cannot resolve suppressed alert {alert_id}"
                )

            now = _utcnow()
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = now
            alert.resolution_notes = resolution_notes
            alert.provenance_hash = _compute_provenance(
                "resolve_alert",
                json.dumps({
                    "alert_id": alert_id,
                    "resolution_notes": resolution_notes,
                    "resolved_at": now.isoformat(),
                }, sort_keys=True),
            )

        _safe_inc(
            _alerts_resolved_total, severity=alert.severity.value,
        )
        self._update_active_alerts_gauge()

        elapsed = time.monotonic() - start
        _safe_observe(
            _alert_processing_duration, elapsed, operation="resolve_alert",
        )

        logger.info("Alert %s resolved: %s", alert_id, resolution_notes)
        return alert

    def suppress_alert(
        self,
        alert_id: str,
        reason: str,
    ) -> FreshnessAlert:
        """Suppress an alert.

        Marks the alert as suppressed with a reason. Suppressed alerts
        are excluded from active alert counts and escalation checks.

        Args:
            alert_id: Identifier of the alert to suppress.
            reason: Reason for suppression (e.g. maintenance window,
                known issue, false positive).

        Returns:
            Updated FreshnessAlert with ``suppressed`` status.

        Raises:
            ValueError: If alert not found or already resolved.
        """
        start = time.monotonic()

        with self._lock:
            alert = self._alerts.get(alert_id)
            if alert is None:
                raise ValueError(f"Alert not found: {alert_id}")

            if alert.status == AlertStatus.RESOLVED:
                raise ValueError(
                    f"Cannot suppress resolved alert {alert_id}"
                )

            now = _utcnow()
            alert.status = AlertStatus.SUPPRESSED
            alert.suppressed_at = now
            alert.suppression_reason = reason
            alert.provenance_hash = _compute_provenance(
                "suppress_alert",
                json.dumps({
                    "alert_id": alert_id,
                    "reason": reason,
                    "suppressed_at": now.isoformat(),
                }, sort_keys=True),
            )

        _safe_inc(
            _alerts_suppressed_total, severity=alert.severity.value,
        )
        self._update_active_alerts_gauge()

        elapsed = time.monotonic() - start
        _safe_observe(
            _alert_processing_duration, elapsed, operation="suppress_alert",
        )

        logger.info("Alert %s suppressed: %s", alert_id, reason)
        return alert

    # ------------------------------------------------------------------
    # Throttling and deduplication
    # ------------------------------------------------------------------

    def should_throttle(
        self,
        dataset_id: str,
        channel: str,
    ) -> bool:
        """Check whether an alert should be throttled.

        An alert is throttled if a previous alert was sent for the same
        (dataset_id, channel) combination within the configured
        ``alert_throttle_minutes`` window.

        Args:
            dataset_id: Dataset identifier.
            channel: Alert channel name.

        Returns:
            True if the alert should be throttled (suppressed), False
            if it is safe to send.
        """
        throttle_key = (dataset_id, channel)
        now = _utcnow()
        throttle_window = timedelta(
            minutes=self._config.alert_throttle_minutes,
        )

        with self._lock:
            last_sent = self._last_alert_times.get(throttle_key)

        if last_sent is None:
            return False

        return (now - last_sent) < throttle_window

    def should_deduplicate(
        self,
        dataset_id: str,
        alert_severity: str,
        channel: str,
    ) -> bool:
        """Check whether an alert should be deduplicated.

        An alert is deduplicated if a previous alert with the same
        (dataset_id, severity, channel) combination was sent within
        the configured ``alert_dedup_window_hours`` window.

        Args:
            dataset_id: Dataset identifier.
            alert_severity: Alert severity level string.
            channel: Alert channel name.

        Returns:
            True if the alert should be deduplicated (suppressed),
            False if it is safe to send.
        """
        dedup_key = (dataset_id, alert_severity, channel)
        now = _utcnow()
        dedup_window = timedelta(
            hours=self._config.alert_dedup_window_hours,
        )

        with self._lock:
            last_sent = self._recent_alerts.get(dedup_key)

        if last_sent is None:
            return False

        return (now - last_sent) < dedup_window

    # ------------------------------------------------------------------
    # Escalation
    # ------------------------------------------------------------------

    def escalate(
        self,
        breach_id: str,
        escalation_policy: Dict[str, Any],
        current_level: int,
    ) -> Optional[FreshnessAlert]:
        """Create an escalated alert for an active breach.

        Evaluates the escalation policy to determine if a higher-level
        alert should be created based on the time elapsed since the
        breach was detected.

        Each escalation policy contains a list of levels, each with:
            - ``delay_minutes``: Minutes after breach detection to trigger.
            - ``channel``: Channel to send the escalated alert on.
            - ``message_template``: Template string with ``{variable}``
              placeholders for the alert message.
            - ``severity``: Optional severity override.
            - ``recipients``: Optional recipient list override.

        Args:
            breach_id: Identifier of the active breach to escalate.
            escalation_policy: Escalation policy dictionary containing
                a ``levels`` list.
            current_level: Current escalation level (0-based index).

        Returns:
            FreshnessAlert for the escalation, or None if no escalation
            is needed at this time.

        Raises:
            ValueError: If breach is not found.
        """
        start = time.monotonic()

        with self._lock:
            breach = self._breaches.get(breach_id)
            if breach is None:
                raise ValueError(f"Breach not found: {breach_id}")

        levels = escalation_policy.get("levels", [])
        if not levels:
            return None

        now = _utcnow()
        time_since_detection = (now - breach.detected_at).total_seconds() / 60.0

        # Find the highest applicable escalation level
        target_level = self._find_escalation_level(
            levels, time_since_detection, current_level,
        )
        if target_level is None:
            return None

        level_config = levels[target_level]
        channel = level_config.get("channel", "email")
        severity = level_config.get("severity", breach.severity.value)
        recipients = level_config.get("recipients", [])
        message_template = level_config.get(
            "message_template", _DEFAULT_ALERT_TEMPLATE,
        )

        # Build context for template rendering
        context = {
            "severity": severity,
            "dataset_name": breach.dataset_id,
            "dataset_id": breach.dataset_id,
            "age_hours": breach.age_at_breach_hours,
            "sla_hours": "N/A",
            "breach_id": breach_id,
            "escalation_level": target_level,
        }
        message = self.format_alert_message(message_template, context)

        alert = self.create_alert(
            breach_id=breach_id,
            dataset_id=breach.dataset_id,
            alert_severity=severity,
            channel=channel,
            message=message,
            recipients=recipients,
        )

        with self._lock:
            alert.escalation_level = target_level
            self._escalation_levels[breach_id] = target_level

        self.send_alert(alert.alert_id)

        _safe_inc(_escalations_total, channel=channel)

        elapsed = time.monotonic() - start
        _safe_observe(
            _alert_processing_duration, elapsed, operation="escalate",
        )

        logger.info(
            "Escalated breach %s to level %d via %s",
            breach_id, target_level, channel,
        )

        # Re-fetch the updated alert
        with self._lock:
            return self._alerts[alert.alert_id]

    def run_escalation_check(
        self,
        active_breaches: List[Dict[str, Any]],
        escalation_policies: Dict[str, Dict[str, Any]],
    ) -> List[FreshnessAlert]:
        """Run escalation checks for a set of active breaches.

        For each active breach, determines whether it should be escalated
        based on the matching escalation policy and the time elapsed since
        detection.

        Args:
            active_breaches: List of dicts, each containing at least:
                - ``breach_id``: str
                - ``dataset_id``: str
                - ``detected_at``: datetime
                - ``severity``: str
                - ``policy_name``: str (key into escalation_policies)
            escalation_policies: Dict mapping policy name to escalation
                policy dict. Each policy has a ``levels`` list.

        Returns:
            List of newly created FreshnessAlert objects from escalations.
        """
        start = time.monotonic()
        escalated_alerts: List[FreshnessAlert] = []

        for breach_info in active_breaches:
            breach_id = breach_info.get("breach_id", "")
            policy_name = breach_info.get("policy_name", "default")
            policy = escalation_policies.get(policy_name)

            if not policy:
                logger.debug(
                    "No escalation policy '%s' for breach %s",
                    policy_name, breach_id,
                )
                continue

            with self._lock:
                current_level = self._escalation_levels.get(breach_id, -1)

            try:
                alert = self.escalate(breach_id, policy, current_level)
                if alert is not None:
                    escalated_alerts.append(alert)
            except ValueError as exc:
                logger.warning(
                    "Escalation failed for breach %s: %s",
                    breach_id, str(exc),
                )

        elapsed = time.monotonic() - start
        _safe_observe(
            _alert_processing_duration, elapsed,
            operation="run_escalation_check",
        )

        logger.info(
            "Escalation check: %d breaches evaluated, %d escalated",
            len(active_breaches), len(escalated_alerts),
        )
        return escalated_alerts

    # ------------------------------------------------------------------
    # Alert message formatting
    # ------------------------------------------------------------------

    def format_alert_message(
        self,
        template: str,
        context: Dict[str, Any],
    ) -> str:
        """Render an alert message from a template and context dict.

        Uses simple ``{variable}`` substitution. Format specifiers
        (e.g. ``{age_hours:.1f}``) are supported via Python's
        ``str.format_map``.

        Args:
            template: Message template string with ``{variable}``
                placeholders.
            context: Dictionary of variable names to values.

        Returns:
            Rendered message string. If rendering fails, returns the
            raw template with a warning prefix.
        """
        try:
            return template.format_map(context)
        except (KeyError, ValueError, IndexError) as exc:
            logger.warning(
                "Alert template rendering failed: %s (template=%s)",
                str(exc), template,
            )
            return f"[template error] {template}"

    # ------------------------------------------------------------------
    # Alert queries
    # ------------------------------------------------------------------

    def get_alert(self, alert_id: str) -> Optional[FreshnessAlert]:
        """Retrieve a single alert by ID.

        Args:
            alert_id: Alert identifier.

        Returns:
            FreshnessAlert if found, None otherwise.
        """
        with self._lock:
            return self._alerts.get(alert_id)

    def list_alerts(
        self,
        dataset_id: Optional[str] = None,
        status: Optional[str] = None,
        severity: Optional[str] = None,
        channel: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[FreshnessAlert]:
        """List alerts with optional filtering.

        Args:
            dataset_id: Filter by dataset ID.
            status: Filter by alert status (pending, sent, acknowledged,
                resolved, suppressed).
            severity: Filter by alert severity (info, warning, critical,
                emergency).
            channel: Filter by alert channel (email, slack, pagerduty,
                webhook, teams, sms).
            limit: Maximum number of results.
            offset: Number of results to skip.

        Returns:
            List of matching FreshnessAlert objects, ordered by
            creation time descending.
        """
        with self._lock:
            alerts = list(self._alerts.values())

        # Apply filters
        if dataset_id is not None:
            alerts = [a for a in alerts if a.dataset_id == dataset_id]
        if status is not None:
            alerts = [a for a in alerts if a.status.value == status]
        if severity is not None:
            alerts = [a for a in alerts if a.severity.value == severity]
        if channel is not None:
            alerts = [a for a in alerts if a.channel.value == channel]

        # Sort by created_at descending (newest first)
        alerts.sort(key=lambda a: a.created_at, reverse=True)

        # Apply pagination
        return alerts[offset: offset + limit]

    def get_active_alerts(self) -> List[FreshnessAlert]:
        """Return all alerts that are not resolved or suppressed.

        Returns:
            List of FreshnessAlert objects with status in
            (pending, sent, acknowledged).
        """
        terminal_statuses = {AlertStatus.RESOLVED, AlertStatus.SUPPRESSED}
        with self._lock:
            return [
                alert for alert in self._alerts.values()
                if alert.status not in terminal_statuses
            ]

    def get_alert_statistics(self) -> Dict[str, Any]:
        """Compute aggregated statistics for all managed alerts.

        Returns:
            Dictionary containing:
                - ``total``: Total alert count.
                - ``by_channel``: Count per channel.
                - ``by_severity``: Count per severity.
                - ``by_status``: Count per status.
                - ``avg_time_to_ack_seconds``: Average seconds from
                  creation to acknowledgement (for acknowledged/resolved).
                - ``avg_time_to_resolve_seconds``: Average seconds from
                  creation to resolution (for resolved alerts).
        """
        with self._lock:
            alerts = list(self._alerts.values())

        by_channel: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}
        by_status: Dict[str, int] = {}
        ack_times: List[float] = []
        resolve_times: List[float] = []

        for alert in alerts:
            channel_val = alert.channel.value
            severity_val = alert.severity.value
            status_val = alert.status.value

            by_channel[channel_val] = by_channel.get(channel_val, 0) + 1
            by_severity[severity_val] = by_severity.get(severity_val, 0) + 1
            by_status[status_val] = by_status.get(status_val, 0) + 1

            if alert.acknowledged_at is not None:
                delta = (alert.acknowledged_at - alert.created_at).total_seconds()
                ack_times.append(delta)

            if alert.resolved_at is not None:
                delta = (alert.resolved_at - alert.created_at).total_seconds()
                resolve_times.append(delta)

        avg_ack = (
            sum(ack_times) / len(ack_times) if ack_times else 0.0
        )
        avg_resolve = (
            sum(resolve_times) / len(resolve_times) if resolve_times else 0.0
        )

        return {
            "total": len(alerts),
            "by_channel": by_channel,
            "by_severity": by_severity,
            "by_status": by_status,
            "avg_time_to_ack_seconds": round(avg_ack, 2),
            "avg_time_to_resolve_seconds": round(avg_resolve, 2),
        }

    # ------------------------------------------------------------------
    # Breach management
    # ------------------------------------------------------------------

    def record_breach(
        self,
        dataset_id: str,
        sla_id: str,
        breach_severity: str,
        age_at_breach_hours: float,
    ) -> SLABreach:
        """Record a new SLA breach.

        Creates a new SLABreach record with ``active`` status and
        records the dataset age at the time of breach detection.

        Args:
            dataset_id: Identifier of the dataset that breached SLA.
            sla_id: Identifier of the SLA definition that was breached.
            breach_severity: Breach severity level (warning, critical,
                emergency). Must be a valid ``BreachSeverity`` value.
            age_at_breach_hours: Age of the dataset in hours at the
                time the breach was detected.

        Returns:
            Newly created SLABreach with ``active`` status.

        Raises:
            ValueError: If severity is invalid.
        """
        start = time.monotonic()
        self._validate_breach_severity(breach_severity)

        breach_id = _generate_id("BRC")
        now = _utcnow()
        provenance_hash = _compute_provenance(
            "record_breach",
            json.dumps({
                "breach_id": breach_id,
                "dataset_id": dataset_id,
                "sla_id": sla_id,
                "severity": breach_severity,
                "age_at_breach_hours": age_at_breach_hours,
            }, sort_keys=True),
        )

        breach = SLABreach(
            breach_id=breach_id,
            dataset_id=dataset_id,
            sla_id=sla_id,
            severity=BreachSeverity(breach_severity),
            status=BreachStatus.DETECTED,
            age_at_breach_hours=age_at_breach_hours,
            detected_at=now,
            acknowledged_at=None,
            acknowledged_by=None,
            resolved_at=None,
            resolution_notes=None,
            provenance_hash=provenance_hash,
        )

        with self._lock:
            self._breaches[breach_id] = breach

        _safe_inc(_breaches_recorded_total, severity=breach_severity)
        self._update_active_breaches_gauge()

        elapsed = time.monotonic() - start
        _safe_observe(
            _alert_processing_duration, elapsed, operation="record_breach",
        )

        logger.info(
            "Recorded breach %s for dataset %s SLA %s [%s] age=%.1fh",
            breach_id, dataset_id, sla_id, breach_severity,
            age_at_breach_hours,
        )
        return breach

    def acknowledge_breach(
        self,
        breach_id: str,
        acknowledged_by: str,
    ) -> SLABreach:
        """Acknowledge an SLA breach.

        Transitions breach from ``active`` to ``acknowledged`` state.

        Args:
            breach_id: Identifier of the breach to acknowledge.
            acknowledged_by: Name or ID of the acknowledging user.

        Returns:
            Updated SLABreach with ``acknowledged`` status.

        Raises:
            ValueError: If breach not found or already resolved.
        """
        start = time.monotonic()

        with self._lock:
            breach = self._breaches.get(breach_id)
            if breach is None:
                raise ValueError(f"Breach not found: {breach_id}")

            if breach.status == BreachStatus.RESOLVED:
                raise ValueError(
                    f"Cannot acknowledge resolved breach {breach_id}"
                )

            now = _utcnow()
            breach.status = BreachStatus.ACKNOWLEDGED
            breach.acknowledged_at = now
            breach.acknowledged_by = acknowledged_by
            breach.provenance_hash = _compute_provenance(
                "acknowledge_breach",
                json.dumps({
                    "breach_id": breach_id,
                    "acknowledged_by": acknowledged_by,
                    "acknowledged_at": now.isoformat(),
                }, sort_keys=True),
            )

        elapsed = time.monotonic() - start
        _safe_observe(
            _alert_processing_duration, elapsed,
            operation="acknowledge_breach",
        )

        logger.info(
            "Breach %s acknowledged by %s", breach_id, acknowledged_by,
        )
        return breach

    def resolve_breach(
        self,
        breach_id: str,
        resolution_notes: str,
    ) -> SLABreach:
        """Resolve an SLA breach.

        Transitions breach to ``resolved`` state with resolution notes.

        Args:
            breach_id: Identifier of the breach to resolve.
            resolution_notes: Free-text description of how the breach
                was resolved.

        Returns:
            Updated SLABreach with ``resolved`` status.

        Raises:
            ValueError: If breach not found.
        """
        start = time.monotonic()

        with self._lock:
            breach = self._breaches.get(breach_id)
            if breach is None:
                raise ValueError(f"Breach not found: {breach_id}")

            now = _utcnow()
            breach.status = BreachStatus.RESOLVED
            breach.resolved_at = now
            breach.resolution_notes = resolution_notes
            breach.provenance_hash = _compute_provenance(
                "resolve_breach",
                json.dumps({
                    "breach_id": breach_id,
                    "resolution_notes": resolution_notes,
                    "resolved_at": now.isoformat(),
                }, sort_keys=True),
            )

        self._update_active_breaches_gauge()

        elapsed = time.monotonic() - start
        _safe_observe(
            _alert_processing_duration, elapsed,
            operation="resolve_breach",
        )

        logger.info("Breach %s resolved: %s", breach_id, resolution_notes)
        return breach

    def get_breach(self, breach_id: str) -> Optional[SLABreach]:
        """Retrieve a single breach by ID.

        Args:
            breach_id: Breach identifier.

        Returns:
            SLABreach if found, None otherwise.
        """
        with self._lock:
            return self._breaches.get(breach_id)

    def list_breaches(
        self,
        dataset_id: Optional[str] = None,
        status: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[SLABreach]:
        """List breaches with optional filtering.

        Args:
            dataset_id: Filter by dataset ID.
            status: Filter by breach status (active, acknowledged,
                resolved).
            severity: Filter by breach severity (warning, critical,
                emergency).
            limit: Maximum number of results.
            offset: Number of results to skip.

        Returns:
            List of matching SLABreach objects, ordered by detection
            time descending.
        """
        with self._lock:
            breaches = list(self._breaches.values())

        # Apply filters
        if dataset_id is not None:
            breaches = [b for b in breaches if b.dataset_id == dataset_id]
        if status is not None:
            breaches = [b for b in breaches if b.status.value == status]
        if severity is not None:
            breaches = [b for b in breaches if b.severity.value == severity]

        # Sort by detected_at descending (newest first)
        breaches.sort(key=lambda b: b.detected_at, reverse=True)

        # Apply pagination
        return breaches[offset: offset + limit]

    def get_active_breaches(self) -> List[SLABreach]:
        """Return all breaches that are not resolved.

        Returns:
            List of SLABreach objects with status in
            (active, acknowledged).
        """
        with self._lock:
            return [
                breach for breach in self._breaches.values()
                if breach.status != BreachStatus.RESOLVED
            ]

    # ------------------------------------------------------------------
    # Combined statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Compute aggregated statistics for alerts and breaches.

        Returns:
            Dictionary containing:
                - ``alerts``: Alert statistics from ``get_alert_statistics()``.
                - ``breaches``: Breach counts by status and severity.
                - ``escalation_levels``: Current escalation level per breach.
                - ``throttle_entries``: Number of active throttle entries.
                - ``dedup_entries``: Number of active dedup entries.
                - ``timestamp``: ISO-8601 timestamp of computation.
        """
        alert_stats = self.get_alert_statistics()

        with self._lock:
            breaches = list(self._breaches.values())
            escalation_levels = dict(self._escalation_levels)
            throttle_count = len(self._last_alert_times)
            dedup_count = len(self._recent_alerts)

        breach_by_status: Dict[str, int] = {}
        breach_by_severity: Dict[str, int] = {}
        for breach in breaches:
            s_val = breach.status.value
            breach_by_status[s_val] = breach_by_status.get(s_val, 0) + 1
            sev_val = breach.severity.value
            breach_by_severity[sev_val] = breach_by_severity.get(sev_val, 0) + 1

        return {
            "alerts": alert_stats,
            "breaches": {
                "total": len(breaches),
                "by_status": breach_by_status,
                "by_severity": breach_by_severity,
            },
            "escalation_levels": escalation_levels,
            "throttle_entries": throttle_count,
            "dedup_entries": dedup_count,
            "timestamp": _utcnow().isoformat(),
        }

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all internal state.

        Removes all alerts, breaches, throttle/dedup tracking, and
        escalation levels. Primarily used for testing.
        """
        with self._lock:
            self._alerts.clear()
            self._breaches.clear()
            self._last_alert_times.clear()
            self._recent_alerts.clear()
            self._escalation_levels.clear()

        self._update_active_alerts_gauge()
        self._update_active_breaches_gauge()
        logger.info("AlertManagerEngine reset: all state cleared")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_severity(self, severity: str) -> None:
        """Validate that a severity string is a valid AlertSeverity value.

        Args:
            severity: Severity string to validate.

        Raises:
            ValueError: If the severity is not a valid AlertSeverity.
        """
        valid = {s.value for s in AlertSeverity}
        if severity not in valid:
            raise ValueError(
                f"Invalid alert severity '{severity}'. "
                f"Must be one of: {sorted(valid)}"
            )

    def _validate_channel(self, channel: str) -> None:
        """Validate that a channel string is a valid AlertChannel value.

        Args:
            channel: Channel string to validate.

        Raises:
            ValueError: If the channel is not a valid AlertChannel.
        """
        valid = {c.value for c in AlertChannel}
        if channel not in valid:
            raise ValueError(
                f"Invalid alert channel '{channel}'. "
                f"Must be one of: {sorted(valid)}"
            )

    def _validate_breach_severity(self, severity: str) -> None:
        """Validate that a severity string is a valid BreachSeverity value.

        Args:
            severity: Severity string to validate.

        Raises:
            ValueError: If the severity is not a valid BreachSeverity.
        """
        valid = {s.value for s in BreachSeverity}
        if severity not in valid:
            raise ValueError(
                f"Invalid breach severity '{severity}'. "
                f"Must be one of: {sorted(valid)}"
            )

    def _find_escalation_level(
        self,
        levels: List[Dict[str, Any]],
        time_since_detection_minutes: float,
        current_level: int,
    ) -> Optional[int]:
        """Find the highest applicable escalation level.

        Scans all levels to find the highest index whose
        ``delay_minutes`` is less than or equal to the elapsed time
        since breach detection, and that is strictly greater than the
        current level.

        Args:
            levels: List of escalation level configuration dicts.
            time_since_detection_minutes: Minutes since breach detection.
            current_level: Current escalation level (index, -1 = none).

        Returns:
            Index of the target escalation level, or None if no
            escalation is needed.
        """
        target: Optional[int] = None

        for i, level in enumerate(levels):
            delay = level.get("delay_minutes", 0)
            if i > current_level and delay <= time_since_detection_minutes:
                target = i

        return target

    def _update_active_alerts_gauge(self) -> None:
        """Update the active alerts Prometheus gauge."""
        if _active_alerts_gauge is None:
            return
        try:
            active_count = len(self.get_active_alerts())
            _active_alerts_gauge.set(active_count)
        except Exception:  # noqa: BLE001
            pass

    def _update_active_breaches_gauge(self) -> None:
        """Update the active breaches Prometheus gauge."""
        if _active_breaches_gauge is None:
            return
        try:
            active_count = len(self.get_active_breaches())
            _active_breaches_gauge.set(active_count)
        except Exception:  # noqa: BLE001
            pass
