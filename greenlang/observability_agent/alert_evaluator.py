# -*- coding: utf-8 -*-
"""
Alert Rule Evaluation Engine - AGENT-FOUND-010: Observability & Telemetry Agent

Provides rule-based alert evaluation against metric data with support for
threshold conditions, duration-based firing, acknowledgment, silencing,
and full alert lifecycle management. All alert transitions include SHA-256
provenance hashes for audit trails.

Zero-Hallucination Guarantees:
    - All condition evaluations use deterministic comparison operators
    - Duration checks use pure arithmetic on UTC timestamps
    - No probabilistic or ML-based anomaly detection
    - Alert state transitions follow a deterministic state machine

Example:
    >>> from greenlang.observability_agent.alert_evaluator import AlertEvaluator
    >>> from greenlang.observability_agent.metrics_collector import MetricsCollector
    >>> from greenlang.observability_agent.config import ObservabilityConfig
    >>> config = ObservabilityConfig()
    >>> collector = MetricsCollector(config)
    >>> evaluator = AlertEvaluator(config, collector)
    >>> evaluator.add_rule("high_latency", "request_latency", "gt", 1.0, 60, "warning")
    >>> result = evaluator.evaluate_all(collector)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-FOUND-010 Observability & Telemetry Agent
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_CONDITIONS: Tuple[str, ...] = ("gt", "lt", "eq", "gte", "lte", "ne")
VALID_SEVERITIES: Tuple[str, ...] = ("info", "warning", "critical", "page")
VALID_ALERT_STATES: Tuple[str, ...] = ("pending", "firing", "resolved", "acknowledged", "silenced")


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class AlertRule:
    """Definition of an alert evaluation rule.

    Attributes:
        rule_id: Unique identifier for this rule.
        name: Human-readable rule name (must be unique).
        metric_name: Name of the metric to evaluate.
        condition: Comparison operator (gt, lt, eq, gte, lte, ne).
        threshold: Threshold value for the condition.
        duration_seconds: How long the condition must hold before firing.
        severity: Alert severity level.
        labels: Additional labels for the alert.
        annotations: Human-readable annotations (summary, description).
        enabled: Whether this rule is currently active.
        silenced_until: If set, the rule is silenced until this timestamp.
        created_at: Rule creation timestamp.
        updated_at: Last update timestamp.
    """

    rule_id: str = ""
    name: str = ""
    metric_name: str = ""
    condition: str = "gt"
    threshold: float = 0.0
    duration_seconds: int = 0
    severity: str = "warning"
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    silenced_until: Optional[datetime] = None
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)

    def __post_init__(self) -> None:
        """Generate rule_id if not provided."""
        if not self.rule_id:
            self.rule_id = str(uuid.uuid4())


@dataclass
class AlertInstance:
    """Instance of a fired or pending alert.

    Attributes:
        alert_id: Unique identifier for this alert instance.
        rule_id: ID of the rule that produced this alert.
        rule_name: Name of the rule (denormalized for convenience).
        state: Current alert state.
        severity: Alert severity.
        metric_name: Name of the evaluated metric.
        metric_value: Metric value that triggered the alert.
        threshold: Threshold value from the rule.
        condition: Condition from the rule.
        labels: Alert labels.
        annotations: Alert annotations.
        started_at: Timestamp when the condition first became true.
        fired_at: Timestamp when the alert actually fired (after duration).
        resolved_at: Timestamp when the alert resolved.
        acknowledged_at: Timestamp when the alert was acknowledged.
        acknowledged_by: Identifier of who acknowledged the alert.
        provenance_hash: SHA-256 hash for audit trail.
    """

    alert_id: str = ""
    rule_id: str = ""
    rule_name: str = ""
    state: str = "pending"
    severity: str = "warning"
    metric_name: str = ""
    metric_value: float = 0.0
    threshold: float = 0.0
    condition: str = "gt"
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    started_at: datetime = field(default_factory=_utcnow)
    fired_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: str = ""
    provenance_hash: str = ""

    def __post_init__(self) -> None:
        """Generate alert_id if not provided."""
        if not self.alert_id:
            self.alert_id = str(uuid.uuid4())


# =============================================================================
# AlertEvaluator
# =============================================================================


class AlertEvaluator:
    """Alert rule evaluation engine.

    Evaluates alert rules against current metric values, manages alert
    lifecycle (pending, firing, resolved, acknowledged, silenced), and
    tracks alert history for analysis.

    Thread-safe via a reentrant lock on all mutating operations.

    Attributes:
        _config: Observability configuration.
        _metrics_collector: Metrics collector for metric lookups.
        _rules: Registered alert rules keyed by name.
        _active_alerts: Currently active alert instances keyed by rule_name.
        _alert_history: Historical alert instances (most recent first).
        _pending_since: Tracks when a rule first started being true.
        _lock: Thread lock for concurrent access.

    Example:
        >>> evaluator = AlertEvaluator(config, collector)
        >>> evaluator.add_rule("high_error_rate", "errors_total", "gt", 100, 30, "critical")
        >>> result = evaluator.evaluate_all(collector)
        >>> print(result["new_alerts"])
    """

    def __init__(self, config: Any, metrics_collector: Any) -> None:
        """Initialize AlertEvaluator.

        Args:
            config: Observability configuration.
            metrics_collector: MetricsCollector instance for metric lookups.
        """
        self._config = config
        self._metrics_collector = metrics_collector
        self._rules: Dict[str, AlertRule] = {}
        self._active_alerts: Dict[str, AlertInstance] = {}
        self._alert_history: List[AlertInstance] = []
        self._pending_since: Dict[str, datetime] = {}
        self._total_evaluations: int = 0
        self._total_alerts_fired: int = 0
        self._total_alerts_resolved: int = 0
        self._lock = threading.RLock()

        self._max_history: int = getattr(config, "alert_history_limit", 10000)

        logger.info(
            "AlertEvaluator initialized: max_history=%d",
            self._max_history,
        )

    # ------------------------------------------------------------------
    # Rule management
    # ------------------------------------------------------------------

    def add_rule(
        self,
        name: str,
        metric_name: str,
        condition: str,
        threshold: float,
        duration_seconds: int = 0,
        severity: str = "warning",
        labels: Optional[Dict[str, str]] = None,
        annotations: Optional[Dict[str, str]] = None,
    ) -> AlertRule:
        """Add a new alert evaluation rule.

        Args:
            name: Unique rule name.
            metric_name: Metric to evaluate.
            condition: Comparison operator (gt, lt, eq, gte, lte, ne).
            threshold: Threshold value.
            duration_seconds: How long condition must hold before firing.
            severity: Alert severity (info, warning, critical, page).
            labels: Additional alert labels.
            annotations: Alert annotations (summary, description, etc.).

        Returns:
            AlertRule that was created.

        Raises:
            ValueError: If name is empty, condition/severity is invalid,
                        or rule already exists.
        """
        if not name or not name.strip():
            raise ValueError("Rule name must be non-empty")

        if condition not in VALID_CONDITIONS:
            raise ValueError(
                f"Invalid condition '{condition}'; must be one of {VALID_CONDITIONS}"
            )

        if severity not in VALID_SEVERITIES:
            raise ValueError(
                f"Invalid severity '{severity}'; must be one of {VALID_SEVERITIES}"
            )

        if duration_seconds < 0:
            raise ValueError("duration_seconds must be non-negative")

        with self._lock:
            if name in self._rules:
                raise ValueError(f"Alert rule '{name}' already exists")

            rule = AlertRule(
                name=name,
                metric_name=metric_name,
                condition=condition,
                threshold=threshold,
                duration_seconds=duration_seconds,
                severity=severity,
                labels=labels or {},
                annotations=annotations or {},
            )
            self._rules[name] = rule

        logger.info(
            "Added alert rule: name=%s, metric=%s, condition=%s %s, severity=%s",
            name, metric_name, condition, threshold, severity,
        )
        return rule

    def update_rule(
        self,
        name: str,
        threshold: Optional[float] = None,
        condition: Optional[str] = None,
        duration_seconds: Optional[int] = None,
        severity: Optional[str] = None,
        enabled: Optional[bool] = None,
    ) -> AlertRule:
        """Update an existing alert rule.

        Args:
            name: Rule name to update.
            threshold: New threshold value.
            condition: New condition operator.
            duration_seconds: New duration requirement.
            severity: New severity level.
            enabled: Enable or disable the rule.

        Returns:
            Updated AlertRule.

        Raises:
            ValueError: If rule not found or invalid parameters.
        """
        with self._lock:
            rule = self._rules.get(name)
            if rule is None:
                raise ValueError(f"Alert rule '{name}' not found")

            if condition is not None:
                if condition not in VALID_CONDITIONS:
                    raise ValueError(f"Invalid condition '{condition}'")
                rule.condition = condition

            if severity is not None:
                if severity not in VALID_SEVERITIES:
                    raise ValueError(f"Invalid severity '{severity}'")
                rule.severity = severity

            if threshold is not None:
                rule.threshold = threshold

            if duration_seconds is not None:
                if duration_seconds < 0:
                    raise ValueError("duration_seconds must be non-negative")
                rule.duration_seconds = duration_seconds

            if enabled is not None:
                rule.enabled = enabled

            rule.updated_at = _utcnow()

        logger.info("Updated alert rule: name=%s", name)
        return rule

    def remove_rule(self, name: str) -> bool:
        """Remove an alert rule and resolve any active alert from it.

        Args:
            name: Rule name to remove.

        Returns:
            True if rule was found and removed, False otherwise.
        """
        with self._lock:
            if name not in self._rules:
                return False

            del self._rules[name]
            self._pending_since.pop(name, None)

            # Resolve active alert if any
            active = self._active_alerts.pop(name, None)
            if active is not None:
                active.state = "resolved"
                active.resolved_at = _utcnow()
                self._append_history(active)
                self._total_alerts_resolved += 1

        logger.info("Removed alert rule: name=%s", name)
        return True

    def list_rules(self) -> List[AlertRule]:
        """List all registered alert rules.

        Returns:
            List of AlertRule objects sorted by name.
        """
        with self._lock:
            rules = list(self._rules.values())
        rules.sort(key=lambda r: r.name)
        return rules

    def get_rule(self, name: str) -> Optional[AlertRule]:
        """Get a specific alert rule by name.

        Args:
            name: Rule name.

        Returns:
            AlertRule or None if not found.
        """
        with self._lock:
            return self._rules.get(name)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_all(self, metrics_collector: Any) -> Dict[str, Any]:
        """Evaluate all enabled rules against current metric values.

        Args:
            metrics_collector: MetricsCollector to query for current values.

        Returns:
            Dictionary with new_alerts, resolved_alerts, still_firing, and
            evaluation_count.
        """
        now = _utcnow()
        new_alerts: List[AlertInstance] = []
        resolved_alerts: List[AlertInstance] = []
        still_firing: List[AlertInstance] = []

        with self._lock:
            for name, rule in self._rules.items():
                if not rule.enabled:
                    continue

                if rule.silenced_until and now < rule.silenced_until:
                    continue

                # Clear expired silence
                if rule.silenced_until and now >= rule.silenced_until:
                    rule.silenced_until = None

                metric_value = self._get_metric_value(metrics_collector, rule.metric_name)
                if metric_value is None:
                    # Metric not found; resolve if previously firing
                    self._handle_condition_false(name, resolved_alerts, now)
                    continue

                is_true = self._evaluate_condition(metric_value, rule.condition, rule.threshold)

                if is_true:
                    should_fire = self._check_duration(name, True, rule.duration_seconds, now)
                    if should_fire:
                        alert = self._fire_alert(rule, metric_value, now)
                        if alert is not None:
                            new_alerts.append(alert)
                        elif name in self._active_alerts:
                            still_firing.append(self._active_alerts[name])
                else:
                    self._handle_condition_false(name, resolved_alerts, now)

            self._total_evaluations += 1

        logger.info(
            "Alert evaluation: new=%d, resolved=%d, firing=%d",
            len(new_alerts), len(resolved_alerts), len(still_firing),
        )

        return {
            "new_alerts": new_alerts,
            "resolved_alerts": resolved_alerts,
            "still_firing": still_firing,
            "evaluation_count": self._total_evaluations,
            "evaluated_at": now.isoformat(),
        }

    def evaluate_rule(
        self,
        name: str,
        metrics_collector: Any,
    ) -> Optional[AlertInstance]:
        """Evaluate a single rule against current metric values.

        Args:
            name: Rule name to evaluate.
            metrics_collector: MetricsCollector for metric lookup.

        Returns:
            AlertInstance if an alert is active or newly fired, None otherwise.

        Raises:
            ValueError: If rule not found.
        """
        with self._lock:
            rule = self._rules.get(name)
            if rule is None:
                raise ValueError(f"Alert rule '{name}' not found")

            metric_value = self._get_metric_value(metrics_collector, rule.metric_name)
            if metric_value is None:
                return None

            now = _utcnow()
            is_true = self._evaluate_condition(metric_value, rule.condition, rule.threshold)

            if is_true:
                should_fire = self._check_duration(name, True, rule.duration_seconds, now)
                if should_fire:
                    alert = self._fire_alert(rule, metric_value, now)
                    if alert is not None:
                        return alert
                    return self._active_alerts.get(name)

            return self._active_alerts.get(name)

    # ------------------------------------------------------------------
    # Alert lifecycle
    # ------------------------------------------------------------------

    def get_active_alerts(self) -> List[AlertInstance]:
        """Get all currently active (firing) alerts.

        Returns:
            List of active AlertInstance objects.
        """
        with self._lock:
            alerts = list(self._active_alerts.values())
        alerts.sort(key=lambda a: a.started_at, reverse=True)
        return alerts

    def get_alert_history(self, limit: int = 100) -> List[AlertInstance]:
        """Get historical alerts (resolved and acknowledged).

        Args:
            limit: Maximum number of history entries to return.

        Returns:
            List of AlertInstance objects, most recent first.
        """
        with self._lock:
            return list(self._alert_history[:limit])

    def acknowledge_alert(
        self,
        alert_id: str,
        acknowledged_by: str = "",
    ) -> AlertInstance:
        """Mark an active alert as acknowledged.

        Args:
            alert_id: Alert identifier to acknowledge.
            acknowledged_by: Identifier of who acknowledged.

        Returns:
            Updated AlertInstance.

        Raises:
            ValueError: If alert not found or not in firing state.
        """
        with self._lock:
            for rule_name, alert in self._active_alerts.items():
                if alert.alert_id == alert_id:
                    if alert.state not in ("firing", "pending"):
                        raise ValueError(
                            f"Alert {alert_id[:8]} is in state '{alert.state}', "
                            "cannot acknowledge"
                        )
                    alert.state = "acknowledged"
                    alert.acknowledged_at = _utcnow()
                    alert.acknowledged_by = acknowledged_by
                    alert.provenance_hash = self._compute_alert_hash(alert)

                    logger.info(
                        "Acknowledged alert: id=%s, rule=%s, by=%s",
                        alert_id[:8], rule_name, acknowledged_by,
                    )
                    return alert

            raise ValueError(f"Active alert '{alert_id[:8]}' not found")

    def silence_rule(
        self,
        name: str,
        duration_seconds: int,
    ) -> AlertRule:
        """Temporarily silence an alert rule.

        Args:
            name: Rule name to silence.
            duration_seconds: Duration of silence in seconds.

        Returns:
            Updated AlertRule.

        Raises:
            ValueError: If rule not found or duration is non-positive.
        """
        if duration_seconds <= 0:
            raise ValueError("Silence duration must be positive")

        with self._lock:
            rule = self._rules.get(name)
            if rule is None:
                raise ValueError(f"Alert rule '{name}' not found")

            now = _utcnow()
            silenced_until = datetime.fromtimestamp(
                now.timestamp() + duration_seconds, tz=timezone.utc,
            ).replace(microsecond=0)

            rule.silenced_until = silenced_until
            rule.updated_at = now

        logger.info(
            "Silenced rule '%s' until %s (%ds)",
            name, silenced_until.isoformat(), duration_seconds,
        )
        return rule

    def unsilence_rule(self, name: str) -> AlertRule:
        """Remove silence from an alert rule.

        Args:
            name: Rule name to unsilence.

        Returns:
            Updated AlertRule.

        Raises:
            ValueError: If rule not found.
        """
        with self._lock:
            rule = self._rules.get(name)
            if rule is None:
                raise ValueError(f"Alert rule '{name}' not found")

            rule.silenced_until = None
            rule.updated_at = _utcnow()

        logger.info("Unsilenced rule '%s'", name)
        return rule

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Get alert evaluator statistics.

        Returns:
            Dictionary with total_rules, active_alerts, total_evaluations,
            total_fired, total_resolved, and history_size.
        """
        with self._lock:
            severity_counts: Dict[str, int] = {}
            for alert in self._active_alerts.values():
                severity_counts[alert.severity] = (
                    severity_counts.get(alert.severity, 0) + 1
                )

            return {
                "total_rules": len(self._rules),
                "enabled_rules": sum(1 for r in self._rules.values() if r.enabled),
                "silenced_rules": sum(
                    1 for r in self._rules.values()
                    if r.silenced_until is not None
                ),
                "active_alerts": len(self._active_alerts),
                "active_by_severity": severity_counts,
                "total_evaluations": self._total_evaluations,
                "total_alerts_fired": self._total_alerts_fired,
                "total_alerts_resolved": self._total_alerts_resolved,
                "history_size": len(self._alert_history),
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate_condition(
        self,
        value: float,
        condition: str,
        threshold: float,
    ) -> bool:
        """Evaluate a numeric condition deterministically.

        Args:
            value: Current metric value.
            condition: Comparison operator.
            threshold: Threshold value.

        Returns:
            True if the condition is met.
        """
        if condition == "gt":
            return value > threshold
        if condition == "lt":
            return value < threshold
        if condition == "eq":
            return value == threshold
        if condition == "gte":
            return value >= threshold
        if condition == "lte":
            return value <= threshold
        if condition == "ne":
            return value != threshold
        return False

    def _check_duration(
        self,
        rule_name: str,
        is_firing: bool,
        duration_seconds: int,
        now: datetime,
    ) -> bool:
        """Check if the condition has been true long enough to fire.

        Must be called within the lock context.

        Args:
            rule_name: Rule name.
            is_firing: Whether the condition is currently true.
            duration_seconds: Required duration in seconds.
            now: Current timestamp.

        Returns:
            True if the alert should fire.
        """
        if not is_firing:
            self._pending_since.pop(rule_name, None)
            return False

        if duration_seconds == 0:
            return True

        if rule_name not in self._pending_since:
            self._pending_since[rule_name] = now
            return False

        elapsed = (now - self._pending_since[rule_name]).total_seconds()
        return elapsed >= duration_seconds

    def _fire_alert(
        self,
        rule: AlertRule,
        metric_value: float,
        now: datetime,
    ) -> Optional[AlertInstance]:
        """Fire a new alert or return None if already firing.

        Must be called within the lock context.

        Args:
            rule: AlertRule that triggered.
            metric_value: Current metric value.
            now: Current timestamp.

        Returns:
            New AlertInstance or None if already active.
        """
        if rule.name in self._active_alerts:
            # Update metric value on existing alert
            self._active_alerts[rule.name].metric_value = metric_value
            return None

        alert = AlertInstance(
            rule_id=rule.rule_id,
            rule_name=rule.name,
            state="firing",
            severity=rule.severity,
            metric_name=rule.metric_name,
            metric_value=metric_value,
            threshold=rule.threshold,
            condition=rule.condition,
            labels=dict(rule.labels),
            annotations=dict(rule.annotations),
            started_at=self._pending_since.get(rule.name, now),
            fired_at=now,
        )
        alert.provenance_hash = self._compute_alert_hash(alert)

        self._active_alerts[rule.name] = alert
        self._total_alerts_fired += 1

        logger.warning(
            "ALERT FIRED: rule=%s, severity=%s, metric=%s, value=%s, threshold=%s",
            rule.name, rule.severity, rule.metric_name, metric_value, rule.threshold,
        )
        return alert

    def _handle_condition_false(
        self,
        rule_name: str,
        resolved_list: List[AlertInstance],
        now: datetime,
    ) -> None:
        """Handle a rule whose condition is no longer true.

        Must be called within the lock context.

        Args:
            rule_name: Rule name.
            resolved_list: List to append resolved alerts to.
            now: Current timestamp.
        """
        self._pending_since.pop(rule_name, None)

        active = self._active_alerts.pop(rule_name, None)
        if active is not None:
            active.state = "resolved"
            active.resolved_at = now
            active.provenance_hash = self._compute_alert_hash(active)
            self._append_history(active)
            self._total_alerts_resolved += 1
            resolved_list.append(active)

            logger.info(
                "Alert resolved: rule=%s, alert_id=%s",
                rule_name, active.alert_id[:8],
            )

    def _append_history(self, alert: AlertInstance) -> None:
        """Append an alert to history, trimming if needed.

        Must be called within the lock context.

        Args:
            alert: AlertInstance to archive.
        """
        self._alert_history.insert(0, alert)
        if len(self._alert_history) > self._max_history:
            self._alert_history = self._alert_history[: self._max_history]

    def _get_metric_value(
        self,
        metrics_collector: Any,
        metric_name: str,
    ) -> Optional[float]:
        """Get the current value of a metric from the collector.

        Tries to get the first available series value for the metric.

        Args:
            metrics_collector: MetricsCollector instance.
            metric_name: Name of the metric to query.

        Returns:
            Current metric value or None if not available.
        """
        try:
            series_list = metrics_collector.get_metric_series(metric_name)
            if not series_list:
                return None
            # Return value from the most recently updated series
            latest = max(series_list, key=lambda s: s.last_updated)
            return latest.value
        except (AttributeError, ValueError):
            return None

    def _compute_alert_hash(self, alert: AlertInstance) -> str:
        """Compute SHA-256 provenance hash for an alert instance.

        Args:
            alert: AlertInstance to hash.

        Returns:
            Hex-encoded SHA-256 hash.
        """
        payload = json.dumps(
            {
                "alert_id": alert.alert_id,
                "rule_id": alert.rule_id,
                "rule_name": alert.rule_name,
                "state": alert.state,
                "severity": alert.severity,
                "metric_name": alert.metric_name,
                "metric_value": alert.metric_value,
                "threshold": alert.threshold,
                "condition": alert.condition,
                "started_at": alert.started_at.isoformat(),
                "fired_at": alert.fired_at.isoformat() if alert.fired_at else None,
                "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None,
            },
            sort_keys=True,
            ensure_ascii=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "AlertEvaluator",
    "AlertRule",
    "AlertInstance",
    "VALID_CONDITIONS",
    "VALID_SEVERITIES",
    "VALID_ALERT_STATES",
]
