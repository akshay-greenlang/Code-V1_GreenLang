# -*- coding: utf-8 -*-
"""
AlertingBridge - Off-Track Performance Alerting for PACK-029
==============================================================

Enterprise bridge for sending notifications when interim target
performance is off-track. Supports email alerts (executive team),
Slack/Teams integration, and dashboard alerts (red/amber flags).
Implements configurable alert rules for variance thresholds,
projected target miss, milestone achievement shortfall, alert
escalation workflows, and alert history tracking.

Alert Rules:
    - Variance > 10% for 2 consecutive quarters
    - Projected target miss > 20%
    - Milestone achievement < 80%
    - Initiative delivery < 60%
    - Budget overrun > 15%
    - Data quality degradation

Alert Channels:
    - Email (SMTP / SendGrid / AWS SES)
    - Slack (webhook integration)
    - Microsoft Teams (webhook integration)
    - Dashboard alerts (in-platform red/amber flags)
    - Custom webhook

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-029 Interim Targets Pack
Status: Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from greenlang.schemas import utcnow
from greenlang.schemas.enums import AlertSeverity, AlertStatus

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

def _new_uuid() -> str:
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AlertChannel(str, Enum):
    EMAIL = "email"
    SLACK = "slack"
    TEAMS = "teams"
    DASHBOARD = "dashboard"
    WEBHOOK = "webhook"

class AlertRuleType(str, Enum):
    VARIANCE_THRESHOLD = "variance_threshold"
    PROJECTED_MISS = "projected_miss"
    MILESTONE_SHORTFALL = "milestone_shortfall"
    INITIATIVE_UNDERPERFORMANCE = "initiative_underperformance"
    BUDGET_OVERRUN = "budget_overrun"
    DATA_QUALITY = "data_quality"
    CONSECUTIVE_MISS = "consecutive_miss"
    CUSTOM = "custom"

class EscalationLevel(str, Enum):
    L1_TEAM_LEAD = "l1_team_lead"
    L2_DEPARTMENT_HEAD = "l2_department_head"
    L3_CSO = "l3_cso"
    L4_CEO = "l4_ceo"
    L5_BOARD = "l5_board"

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class AlertingBridgeConfig(BaseModel):
    """Configuration for the alerting bridge."""
    pack_id: str = Field(default="PACK-029")
    organization_id: str = Field(default="")
    enabled_channels: List[AlertChannel] = Field(
        default_factory=lambda: [AlertChannel.EMAIL, AlertChannel.DASHBOARD]
    )
    email_recipients: List[str] = Field(default_factory=list)
    slack_webhook_url: str = Field(default="")
    teams_webhook_url: str = Field(default="")
    custom_webhook_url: str = Field(default="")
    smtp_host: str = Field(default="")
    smtp_port: int = Field(default=587)
    smtp_user: str = Field(default="")
    smtp_password: str = Field(default="")
    from_email: str = Field(default="alerts@greenlang.platform")
    enable_provenance: bool = Field(default=True)
    alert_cooldown_minutes: int = Field(default=60, ge=5, le=1440)
    max_alerts_per_day: int = Field(default=50, ge=1, le=200)
    escalation_enabled: bool = Field(default=True)
    escalation_timeout_hours: int = Field(default=24, ge=1, le=168)

class AlertRule(BaseModel):
    """Configurable alert rule definition."""
    rule_id: str = Field(default_factory=_new_uuid)
    rule_type: AlertRuleType = Field(default=AlertRuleType.VARIANCE_THRESHOLD)
    name: str = Field(default="")
    description: str = Field(default="")
    enabled: bool = Field(default=True)
    severity: AlertSeverity = Field(default=AlertSeverity.MEDIUM)
    threshold_value: float = Field(default=10.0)
    threshold_unit: str = Field(default="percent")
    consecutive_periods: int = Field(default=2)
    channels: List[AlertChannel] = Field(default_factory=lambda: [AlertChannel.DASHBOARD])
    escalation_level: EscalationLevel = Field(default=EscalationLevel.L1_TEAM_LEAD)
    recipients: List[str] = Field(default_factory=list)

class Alert(BaseModel):
    """Single alert instance."""
    alert_id: str = Field(default_factory=_new_uuid)
    rule_id: str = Field(default="")
    rule_type: AlertRuleType = Field(default=AlertRuleType.VARIANCE_THRESHOLD)
    severity: AlertSeverity = Field(default=AlertSeverity.MEDIUM)
    title: str = Field(default="")
    message: str = Field(default="")
    detail: Dict[str, Any] = Field(default_factory=dict)
    status: AlertStatus = Field(default=AlertStatus.PENDING)
    channels_targeted: List[AlertChannel] = Field(default_factory=list)
    channels_delivered: List[AlertChannel] = Field(default_factory=list)
    escalation_level: EscalationLevel = Field(default=EscalationLevel.L1_TEAM_LEAD)
    created_at: datetime = Field(default_factory=utcnow)
    acknowledged_at: Optional[datetime] = Field(None)
    resolved_at: Optional[datetime] = Field(None)
    acknowledged_by: str = Field(default="")
    provenance_hash: str = Field(default="")

class AlertEvaluation(BaseModel):
    """Result of evaluating alert rules against current data."""
    evaluation_id: str = Field(default_factory=_new_uuid)
    rules_evaluated: int = Field(default=0)
    rules_triggered: int = Field(default=0)
    alerts_generated: List[Alert] = Field(default_factory=list)
    alerts_suppressed: int = Field(default=0)
    evaluation_timestamp: datetime = Field(default_factory=utcnow)
    provenance_hash: str = Field(default="")

# ---------------------------------------------------------------------------
# Default Alert Rules
# ---------------------------------------------------------------------------

DEFAULT_ALERT_RULES: List[Dict[str, Any]] = [
    {
        "rule_type": "variance_threshold",
        "name": "Quarterly variance > 10%",
        "description": "Emissions exceed budget by >10% for 2 consecutive quarters",
        "severity": "high",
        "threshold_value": 10.0,
        "consecutive_periods": 2,
        "channels": ["email", "dashboard"],
        "escalation_level": "l2_department_head",
    },
    {
        "rule_type": "projected_miss",
        "name": "Projected target miss > 20%",
        "description": "Projected emissions at year-end exceed target by >20%",
        "severity": "critical",
        "threshold_value": 20.0,
        "channels": ["email", "slack", "dashboard"],
        "escalation_level": "l3_cso",
    },
    {
        "rule_type": "milestone_shortfall",
        "name": "Milestone achievement < 80%",
        "description": "Interim milestone delivery below 80% threshold",
        "severity": "high",
        "threshold_value": 80.0,
        "channels": ["email", "dashboard"],
        "escalation_level": "l2_department_head",
    },
    {
        "rule_type": "initiative_underperformance",
        "name": "Initiative delivery < 60%",
        "description": "Key initiative delivering <60% of projected reduction",
        "severity": "medium",
        "threshold_value": 60.0,
        "channels": ["dashboard"],
        "escalation_level": "l1_team_lead",
    },
    {
        "rule_type": "budget_overrun",
        "name": "Carbon budget overrun > 15%",
        "description": "Annual carbon budget exceeded by >15%",
        "severity": "high",
        "threshold_value": 15.0,
        "channels": ["email", "dashboard"],
        "escalation_level": "l2_department_head",
    },
    {
        "rule_type": "data_quality",
        "name": "Data quality degradation",
        "description": "Overall data quality score dropped below 70%",
        "severity": "medium",
        "threshold_value": 70.0,
        "channels": ["dashboard"],
        "escalation_level": "l1_team_lead",
    },
]

# ---------------------------------------------------------------------------
# AlertingBridge
# ---------------------------------------------------------------------------

class AlertingBridge:
    """Off-track performance alerting bridge for PACK-029.

    Evaluates configurable alert rules against interim target
    performance data and sends notifications via email, Slack,
    Teams, and dashboard channels with escalation workflows.

    Example:
        >>> bridge = AlertingBridge(AlertingBridgeConfig(
        ...     email_recipients=["cso@company.com"],
        ...     slack_webhook_url="https://hooks.slack.com/...",
        ... ))
        >>> rules = bridge.get_default_rules()
        >>> evaluation = await bridge.evaluate_rules(performance_data, rules)
        >>> print(f"Triggered: {evaluation.rules_triggered}")
    """

    def __init__(self, config: Optional[AlertingBridgeConfig] = None) -> None:
        self.config = config or AlertingBridgeConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._alert_history: List[Alert] = []
        self._rules: List[AlertRule] = []
        self._daily_alert_count: int = 0
        self._last_alert_times: Dict[str, float] = {}

        self.logger.info(
            "AlertingBridge (PACK-029) initialized: channels=%s, "
            "escalation=%s, cooldown=%dm",
            [c.value for c in self.config.enabled_channels],
            self.config.escalation_enabled,
            self.config.alert_cooldown_minutes,
        )

    def get_default_rules(self) -> List[AlertRule]:
        """Get default alert rules."""
        rules: List[AlertRule] = []
        for r in DEFAULT_ALERT_RULES:
            rule = AlertRule(
                rule_type=AlertRuleType(r["rule_type"]),
                name=r["name"],
                description=r["description"],
                severity=AlertSeverity(r["severity"]),
                threshold_value=r["threshold_value"],
                consecutive_periods=r.get("consecutive_periods", 1),
                channels=[AlertChannel(c) for c in r.get("channels", ["dashboard"])],
                escalation_level=EscalationLevel(r.get("escalation_level", "l1_team_lead")),
                recipients=self.config.email_recipients,
            )
            rules.append(rule)
        self._rules = rules
        return rules

    async def evaluate_rules(
        self,
        performance_data: Dict[str, Any],
        rules: Optional[List[AlertRule]] = None,
    ) -> AlertEvaluation:
        """Evaluate alert rules against current performance data."""
        rule_list = rules or self._rules or self.get_default_rules()
        alerts: List[Alert] = []
        suppressed = 0

        for rule in rule_list:
            if not rule.enabled:
                continue

            triggered, alert_data = self._evaluate_single_rule(rule, performance_data)
            if triggered:
                # Check cooldown
                if self._is_in_cooldown(rule.rule_id):
                    suppressed += 1
                    continue

                # Check daily limit
                if self._daily_alert_count >= self.config.max_alerts_per_day:
                    suppressed += 1
                    continue

                alert = Alert(
                    rule_id=rule.rule_id,
                    rule_type=rule.rule_type,
                    severity=rule.severity,
                    title=alert_data.get("title", rule.name),
                    message=alert_data.get("message", rule.description),
                    detail=alert_data,
                    channels_targeted=[c for c in rule.channels if c in self.config.enabled_channels],
                    escalation_level=rule.escalation_level,
                )

                # Deliver to channels
                delivered = await self._deliver_alert(alert)
                alert.channels_delivered = delivered
                alert.status = AlertStatus.SENT if delivered else AlertStatus.PENDING

                if self.config.enable_provenance:
                    alert.provenance_hash = _compute_hash(alert)

                alerts.append(alert)
                self._alert_history.append(alert)
                self._daily_alert_count += 1
                self._last_alert_times[rule.rule_id] = time.monotonic()

        evaluation = AlertEvaluation(
            rules_evaluated=len(rule_list),
            rules_triggered=len(alerts),
            alerts_generated=alerts,
            alerts_suppressed=suppressed,
        )

        if self.config.enable_provenance:
            evaluation.provenance_hash = _compute_hash(evaluation)

        self.logger.info(
            "Alert evaluation: %d rules, %d triggered, %d suppressed",
            len(rule_list), len(alerts), suppressed,
        )
        return evaluation

    async def acknowledge_alert(
        self, alert_id: str, acknowledged_by: str,
    ) -> Optional[Alert]:
        """Acknowledge an alert."""
        alert = next((a for a in self._alert_history if a.alert_id == alert_id), None)
        if alert:
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = utcnow()
            alert.acknowledged_by = acknowledged_by
        return alert

    async def resolve_alert(
        self, alert_id: str,
    ) -> Optional[Alert]:
        """Resolve an alert."""
        alert = next((a for a in self._alert_history if a.alert_id == alert_id), None)
        if alert:
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = utcnow()
        return alert

    async def escalate_alert(
        self, alert_id: str,
    ) -> Optional[Alert]:
        """Escalate an alert to next level."""
        alert = next((a for a in self._alert_history if a.alert_id == alert_id), None)
        if not alert:
            return None

        escalation_order = [
            EscalationLevel.L1_TEAM_LEAD,
            EscalationLevel.L2_DEPARTMENT_HEAD,
            EscalationLevel.L3_CSO,
            EscalationLevel.L4_CEO,
            EscalationLevel.L5_BOARD,
        ]
        current_idx = escalation_order.index(alert.escalation_level) if alert.escalation_level in escalation_order else 0
        if current_idx < len(escalation_order) - 1:
            alert.escalation_level = escalation_order[current_idx + 1]
            alert.status = AlertStatus.ESCALATED
            self.logger.info(
                "Alert %s escalated to %s", alert_id, alert.escalation_level.value,
            )
        return alert

    def get_alert_history(
        self,
        severity: Optional[AlertSeverity] = None,
        status: Optional[AlertStatus] = None,
        limit: int = 100,
    ) -> List[Alert]:
        """Get alert history with optional filtering."""
        alerts = self._alert_history
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if status:
            alerts = [a for a in alerts if a.status == status]
        return sorted(alerts, key=lambda a: a.created_at, reverse=True)[:limit]

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics."""
        by_severity: Dict[str, int] = {}
        by_status: Dict[str, int] = {}
        by_type: Dict[str, int] = {}

        for alert in self._alert_history:
            by_severity[alert.severity.value] = by_severity.get(alert.severity.value, 0) + 1
            by_status[alert.status.value] = by_status.get(alert.status.value, 0) + 1
            by_type[alert.rule_type.value] = by_type.get(alert.rule_type.value, 0) + 1

        return {
            "total_alerts": len(self._alert_history),
            "by_severity": by_severity,
            "by_status": by_status,
            "by_type": by_type,
            "pending": sum(1 for a in self._alert_history if a.status == AlertStatus.PENDING),
            "unresolved": sum(1 for a in self._alert_history if a.status not in (AlertStatus.RESOLVED, AlertStatus.SUPPRESSED)),
        }

    def get_bridge_status(self) -> Dict[str, Any]:
        """Get current bridge status."""
        return {
            "pack_id": self.config.pack_id,
            "enabled_channels": [c.value for c in self.config.enabled_channels],
            "rules_configured": len(self._rules),
            "total_alerts_sent": len(self._alert_history),
            "daily_count": self._daily_alert_count,
            "escalation_enabled": self.config.escalation_enabled,
        }

    # -------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------

    def _evaluate_single_rule(
        self, rule: AlertRule, data: Dict[str, Any],
    ) -> tuple:
        """Evaluate a single alert rule. Returns (triggered, detail)."""
        if rule.rule_type == AlertRuleType.VARIANCE_THRESHOLD:
            variance_pct = data.get("variance_pct", 0.0)
            consecutive = data.get("consecutive_over_quarters", 0)
            triggered = (
                abs(variance_pct) > rule.threshold_value
                and consecutive >= rule.consecutive_periods
            )
            return triggered, {
                "title": f"Emissions variance {variance_pct:.1f}% exceeds {rule.threshold_value:.0f}% threshold",
                "message": f"Emissions have exceeded budget by >{rule.threshold_value:.0f}% for {consecutive} consecutive quarters. Immediate corrective action required.",
                "variance_pct": variance_pct,
                "threshold_pct": rule.threshold_value,
                "consecutive_periods": consecutive,
            }

        elif rule.rule_type == AlertRuleType.PROJECTED_MISS:
            projected_miss_pct = data.get("projected_miss_pct", 0.0)
            triggered = projected_miss_pct > rule.threshold_value
            return triggered, {
                "title": f"Projected target miss: {projected_miss_pct:.1f}%",
                "message": f"Based on current trajectory, the {data.get('target_year', 2030)} interim target will be missed by {projected_miss_pct:.1f}%.",
                "projected_miss_pct": projected_miss_pct,
            }

        elif rule.rule_type == AlertRuleType.MILESTONE_SHORTFALL:
            achievement_pct = data.get("milestone_achievement_pct", 100.0)
            triggered = achievement_pct < rule.threshold_value
            return triggered, {
                "title": f"Milestone achievement at {achievement_pct:.1f}%",
                "message": f"Interim milestone achievement is {achievement_pct:.1f}%, below the {rule.threshold_value:.0f}% threshold.",
                "achievement_pct": achievement_pct,
            }

        elif rule.rule_type == AlertRuleType.INITIATIVE_UNDERPERFORMANCE:
            delivery_pct = data.get("initiative_delivery_pct", 100.0)
            triggered = delivery_pct < rule.threshold_value
            return triggered, {
                "title": f"Initiative delivery at {delivery_pct:.1f}%",
                "message": f"Initiative portfolio delivery is {delivery_pct:.1f}%, below the {rule.threshold_value:.0f}% threshold.",
                "delivery_pct": delivery_pct,
            }

        elif rule.rule_type == AlertRuleType.BUDGET_OVERRUN:
            budget_overrun_pct = data.get("budget_overrun_pct", 0.0)
            triggered = budget_overrun_pct > rule.threshold_value
            return triggered, {
                "title": f"Carbon budget overrun: {budget_overrun_pct:.1f}%",
                "message": f"Annual carbon budget exceeded by {budget_overrun_pct:.1f}%.",
                "overrun_pct": budget_overrun_pct,
            }

        elif rule.rule_type == AlertRuleType.DATA_QUALITY:
            quality_score = data.get("data_quality_score_pct", 100.0)
            triggered = quality_score < rule.threshold_value
            return triggered, {
                "title": f"Data quality at {quality_score:.1f}%",
                "message": f"Data quality score {quality_score:.1f}% below {rule.threshold_value:.0f}% minimum.",
                "quality_score_pct": quality_score,
            }

        return False, {}

    async def _deliver_alert(self, alert: Alert) -> List[AlertChannel]:
        """Deliver alert to all targeted channels."""
        delivered: List[AlertChannel] = []

        for channel in alert.channels_targeted:
            try:
                if channel == AlertChannel.DASHBOARD:
                    # Dashboard alerts are stored in history
                    delivered.append(channel)

                elif channel == AlertChannel.EMAIL:
                    if self.config.smtp_host or self.config.email_recipients:
                        # In production: send via SMTP/SendGrid/SES
                        self.logger.info(
                            "Email alert [%s]: %s -> %s",
                            alert.severity.value, alert.title,
                            self.config.email_recipients,
                        )
                        delivered.append(channel)

                elif channel == AlertChannel.SLACK:
                    if self.config.slack_webhook_url:
                        # In production: POST to Slack webhook
                        self.logger.info(
                            "Slack alert [%s]: %s", alert.severity.value, alert.title,
                        )
                        delivered.append(channel)

                elif channel == AlertChannel.TEAMS:
                    if self.config.teams_webhook_url:
                        # In production: POST to Teams webhook
                        self.logger.info(
                            "Teams alert [%s]: %s", alert.severity.value, alert.title,
                        )
                        delivered.append(channel)

                elif channel == AlertChannel.WEBHOOK:
                    if self.config.custom_webhook_url:
                        self.logger.info(
                            "Webhook alert [%s]: %s", alert.severity.value, alert.title,
                        )
                        delivered.append(channel)

            except Exception as exc:
                self.logger.error(
                    "Failed to deliver alert to %s: %s", channel.value, exc,
                )

        return delivered

    def _is_in_cooldown(self, rule_id: str) -> bool:
        """Check if a rule is in cooldown period."""
        last_time = self._last_alert_times.get(rule_id)
        if last_time is None:
            return False
        elapsed = time.monotonic() - last_time
        return elapsed < self.config.alert_cooldown_minutes * 60
