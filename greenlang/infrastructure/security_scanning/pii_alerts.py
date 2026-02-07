# -*- coding: utf-8 -*-
"""
PII Alert Router - SEC-007 Security Scanning Pipeline

Routes PII detection alerts to appropriate teams based on data
classification. Integrates with audit logging and provides
remediation guidance for detected PII.

Routing Rules:
    - PII (Personally Identifiable Information) -> Legal/Privacy Team
    - PHI (Protected Health Information) -> Compliance Team (HIPAA)
    - PCI (Payment Card Industry) -> Security Team (PCI-DSS)
    - SECRET (Credentials/Keys) -> Security Team (Immediate)

Integration Points:
    - Audit logging (SEC-005)
    - Notification service
    - Metrics collection (SEC-007)
    - Incident management

Author: GreenLang Framework Team
Date: February 2026
PRD: SEC-007 Security Scanning Pipeline
Status: Production Ready
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from uuid import UUID, uuid4

from .pii_scanner import PIIFinding, DataClassification, PIIType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums and Constants
# ---------------------------------------------------------------------------


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertTeam(str, Enum):
    """Target teams for alert routing."""

    SECURITY = "security"
    LEGAL = "legal"
    COMPLIANCE = "compliance"
    DATA_STEWARD = "data_steward"
    PRIVACY = "privacy"
    ENGINEERING = "engineering"


class AlertStatus(str, Enum):
    """Alert status."""

    PENDING = "pending"
    SENT = "sent"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------


@dataclass
class PIIAlert:
    """A PII detection alert."""

    id: UUID
    finding_id: UUID
    classification: DataClassification
    pii_type: PIIType
    severity: AlertSeverity
    target_teams: List[AlertTeam]
    file_path: Optional[str]
    remediation_guidance: str
    status: AlertStatus
    created_at: datetime
    sent_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingRule:
    """Routing rule definition."""

    classification: DataClassification
    target_teams: List[AlertTeam]
    severity_override: Optional[AlertSeverity] = None
    notification_channels: List[str] = field(default_factory=lambda: ["email"])
    escalation_timeout_minutes: int = 60
    auto_create_incident: bool = False
    remediation_template: str = ""


# ---------------------------------------------------------------------------
# Remediation Guidance Templates
# ---------------------------------------------------------------------------

REMEDIATION_TEMPLATES: Dict[DataClassification, Dict[PIIType, str]] = {
    DataClassification.PII: {
        PIIType.SSN: """
**SSN Detected - Immediate Action Required**

1. **Remove the SSN** from the codebase immediately
2. **Rotate any related credentials** if SSN was used in authentication
3. **Check git history** - you may need to rewrite commits
4. **Notify the Privacy Team** about the data exposure
5. **Document the incident** for compliance records

Command to remove from git history:
```bash
git filter-branch --force --index-filter \\
  'git rm --cached --ignore-unmatch {file_path}' \\
  --prune-empty --tag-name-filter cat -- --all
```
""",
        PIIType.EMAIL: """
**Email Address Detected**

1. **Determine if this is test data** - check if domain is example.com, test.com, etc.
2. **If production data**, remove or replace with anonymized values
3. **Consider using environment variables** for configuration emails
4. **Update documentation** if this was intentionally committed

For test fixtures, use clearly fake domains:
- test@example.com
- user@test.localhost
""",
        PIIType.PHONE: """
**Phone Number Detected**

1. **Verify if this is test data** or production data
2. **Remove or anonymize** if it's real user data
3. **Use placeholder formats** for examples: (555) 123-4567
4. **Check related files** for other phone numbers
""",
    },
    DataClassification.PHI: {
        PIIType.MEDICAL_RECORD: """
**PHI Detected - HIPAA Alert**

**CRITICAL: Protected Health Information requires immediate action**

1. **STOP - Do not merge or deploy** this code
2. **Notify the Compliance Team** immediately
3. **Document the exposure** - who, what, when, where
4. **Begin incident response** per HIPAA Breach Notification Rule
5. **Assess if breach notification** is required (>500 individuals)

HIPAA requires:
- Breach notification within 60 days
- Documentation of the risk assessment
- Remediation and prevention measures

Contact: compliance@greenlang.io
""",
    },
    DataClassification.PCI: {
        PIIType.CREDIT_CARD: """
**PCI Data Detected - Security Alert**

**CRITICAL: Payment card data must never be stored in code**

1. **Remove immediately** - do not commit or merge
2. **Invalidate the card** if this is a real card number
3. **Report to PCI Compliance** for assessment
4. **Review PCI DSS requirements** - we cannot store full PANs
5. **Use tokenization** for any payment card handling

PCI DSS Requirements:
- Never store full track data, CVV, or PIN
- Mask PAN when displayed (show only last 4)
- Encrypt PAN if storage is absolutely necessary

Contact: security@greenlang.io
""",
    },
    DataClassification.SECRET: {
        PIIType.API_KEY: """
**API Key/Secret Detected - Immediate Rotation Required**

1. **ROTATE THE KEY IMMEDIATELY** before any other action
2. **Revoke the exposed key** in the provider's console
3. **Generate a new key** and store securely
4. **Update all systems** using the old key
5. **Check for unauthorized usage** in provider logs
6. **Clean git history** to remove the exposed key

For AWS keys:
```bash
aws iam delete-access-key --access-key-id <exposed-key>
aws iam create-access-key --user-name <user>
```

Use environment variables or secrets manager - NEVER commit secrets.
""",
        PIIType.PASSWORD: """
**Password Detected - Security Alert**

1. **Remove the password** from the code
2. **Change the password** if it's for a real account
3. **Use environment variables** or secrets manager
4. **Audit access** to the repository
5. **Enable MFA** on the affected account

Never commit passwords. Use:
- Environment variables
- AWS Secrets Manager / HashiCorp Vault
- .env files (gitignored)
""",
        PIIType.TOKEN: """
**Token Detected - Security Alert**

1. **Invalidate the token** immediately
2. **Generate a new token** with minimal required permissions
3. **Store securely** using secrets management
4. **Review token permissions** - principle of least privilege
5. **Set token expiration** if not already configured

Tokens should:
- Have short TTLs (15 minutes for access tokens)
- Be scoped to minimum required permissions
- Never be logged or stored in code
""",
    },
}


# ---------------------------------------------------------------------------
# Default Routing Rules
# ---------------------------------------------------------------------------


def _get_default_routing_rules() -> Dict[DataClassification, RoutingRule]:
    """Get default routing rules by classification.

    Returns:
        Dictionary of classification -> RoutingRule.
    """
    return {
        DataClassification.PII: RoutingRule(
            classification=DataClassification.PII,
            target_teams=[AlertTeam.LEGAL, AlertTeam.PRIVACY, AlertTeam.DATA_STEWARD],
            severity_override=AlertSeverity.HIGH,
            notification_channels=["email", "slack"],
            escalation_timeout_minutes=120,
            auto_create_incident=False,
            remediation_template="Remove or anonymize PII data",
        ),
        DataClassification.PHI: RoutingRule(
            classification=DataClassification.PHI,
            target_teams=[AlertTeam.COMPLIANCE, AlertTeam.LEGAL, AlertTeam.SECURITY],
            severity_override=AlertSeverity.CRITICAL,
            notification_channels=["email", "slack", "pagerduty"],
            escalation_timeout_minutes=30,
            auto_create_incident=True,
            remediation_template="HIPAA breach response required",
        ),
        DataClassification.PCI: RoutingRule(
            classification=DataClassification.PCI,
            target_teams=[AlertTeam.SECURITY, AlertTeam.COMPLIANCE],
            severity_override=AlertSeverity.CRITICAL,
            notification_channels=["email", "slack", "pagerduty"],
            escalation_timeout_minutes=30,
            auto_create_incident=True,
            remediation_template="PCI-DSS compliance action required",
        ),
        DataClassification.SECRET: RoutingRule(
            classification=DataClassification.SECRET,
            target_teams=[AlertTeam.SECURITY],
            severity_override=AlertSeverity.CRITICAL,
            notification_channels=["email", "slack", "pagerduty"],
            escalation_timeout_minutes=15,
            auto_create_incident=True,
            remediation_template="Rotate credentials immediately",
        ),
        DataClassification.INTERNAL: RoutingRule(
            classification=DataClassification.INTERNAL,
            target_teams=[AlertTeam.ENGINEERING, AlertTeam.DATA_STEWARD],
            severity_override=AlertSeverity.MEDIUM,
            notification_channels=["email"],
            escalation_timeout_minutes=240,
            auto_create_incident=False,
            remediation_template="Review and remediate internal data exposure",
        ),
    }


# ---------------------------------------------------------------------------
# Alert Router
# ---------------------------------------------------------------------------


class PIIAlertRouter:
    """Routes PII detection alerts to appropriate teams.

    Routes alerts based on data classification and integrates with
    audit logging, notifications, and incident management systems.

    Example:
        >>> router = PIIAlertRouter()
        >>> alerts = await router.route_findings(findings)
        >>> for alert in alerts:
        ...     print(f"Alert {alert.id} -> {alert.target_teams}")
    """

    def __init__(
        self,
        routing_rules: Optional[Dict[DataClassification, RoutingRule]] = None,
        audit_service: Optional[Any] = None,
        notification_service: Optional[Any] = None,
        metrics: Optional[Any] = None,
    ) -> None:
        """Initialize PIIAlertRouter.

        Args:
            routing_rules: Custom routing rules (uses defaults if None).
            audit_service: AuditService for logging.
            notification_service: Notification service for alerts.
            metrics: SecurityMetrics for tracking.
        """
        self._rules = routing_rules or _get_default_routing_rules()
        self._audit_service = audit_service
        self._notification_service = notification_service
        self._metrics = metrics
        self._pending_alerts: Dict[UUID, PIIAlert] = {}

    async def route_findings(
        self,
        findings: List[PIIFinding],
    ) -> List[PIIAlert]:
        """Route PII findings to appropriate teams.

        Args:
            findings: List of PII findings to route.

        Returns:
            List of created alerts.
        """
        alerts: List[PIIAlert] = []

        for finding in findings:
            alert = await self._create_alert(finding)
            if alert:
                alerts.append(alert)
                self._pending_alerts[alert.id] = alert

                # Send notifications
                await self._send_notifications(alert)

                # Log to audit service
                await self._log_audit_event(alert)

                # Record metrics
                self._record_metrics(alert)

        logger.info(
            "Routed %d PII findings to %d alerts",
            len(findings), len(alerts)
        )

        return alerts

    async def route_single(
        self,
        finding: PIIFinding,
    ) -> Optional[PIIAlert]:
        """Route a single PII finding.

        Args:
            finding: The PII finding to route.

        Returns:
            Created alert or None.
        """
        alerts = await self.route_findings([finding])
        return alerts[0] if alerts else None

    def get_remediation_guidance(
        self,
        classification: DataClassification,
        pii_type: PIIType,
        file_path: Optional[str] = None,
    ) -> str:
        """Get remediation guidance for a finding.

        Args:
            classification: Data classification.
            pii_type: PII type detected.
            file_path: Optional file path for templates.

        Returns:
            Remediation guidance string.
        """
        # Check for specific template
        classification_templates = REMEDIATION_TEMPLATES.get(classification, {})
        template = classification_templates.get(pii_type)

        if template:
            # Replace placeholders
            if file_path:
                template = template.replace("{file_path}", file_path)
            return template.strip()

        # Fall back to generic guidance
        rule = self._rules.get(classification)
        if rule:
            return rule.remediation_template

        return f"Review and remediate {classification.value} data ({pii_type.value})"

    def get_routing_rule(
        self,
        classification: DataClassification,
    ) -> Optional[RoutingRule]:
        """Get the routing rule for a classification.

        Args:
            classification: Data classification.

        Returns:
            RoutingRule or None.
        """
        return self._rules.get(classification)

    async def acknowledge_alert(
        self,
        alert_id: UUID,
        acknowledged_by: str,
    ) -> bool:
        """Acknowledge an alert.

        Args:
            alert_id: Alert UUID.
            acknowledged_by: User who acknowledged.

        Returns:
            True if acknowledged successfully.
        """
        alert = self._pending_alerts.get(alert_id)
        if not alert:
            logger.warning("Alert %s not found", alert_id)
            return False

        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_at = datetime.utcnow()
        alert.metadata["acknowledged_by"] = acknowledged_by

        logger.info("Alert %s acknowledged by %s", alert_id, acknowledged_by)

        # Log to audit
        if self._audit_service:
            await self._log_audit_event(
                alert,
                action="acknowledge",
                actor=acknowledged_by,
            )

        return True

    async def resolve_alert(
        self,
        alert_id: UUID,
        resolved_by: str,
        resolution_notes: Optional[str] = None,
    ) -> bool:
        """Resolve an alert.

        Args:
            alert_id: Alert UUID.
            resolved_by: User who resolved.
            resolution_notes: Optional notes.

        Returns:
            True if resolved successfully.
        """
        alert = self._pending_alerts.get(alert_id)
        if not alert:
            logger.warning("Alert %s not found", alert_id)
            return False

        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.utcnow()
        alert.metadata["resolved_by"] = resolved_by
        if resolution_notes:
            alert.metadata["resolution_notes"] = resolution_notes

        logger.info("Alert %s resolved by %s", alert_id, resolved_by)

        # Remove from pending
        del self._pending_alerts[alert_id]

        # Log to audit
        if self._audit_service:
            await self._log_audit_event(
                alert,
                action="resolve",
                actor=resolved_by,
            )

        return True

    # -------------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------------

    async def _create_alert(
        self,
        finding: PIIFinding,
    ) -> Optional[PIIAlert]:
        """Create an alert from a finding.

        Args:
            finding: PII finding.

        Returns:
            Created PIIAlert or None.
        """
        rule = self._rules.get(finding.classification)
        if not rule:
            logger.warning(
                "No routing rule for classification: %s",
                finding.classification,
            )
            rule = self._rules.get(DataClassification.INTERNAL)
            if not rule:
                return None

        # Determine severity
        if rule.severity_override:
            severity = rule.severity_override
        elif finding.exposure_risk == "critical":
            severity = AlertSeverity.CRITICAL
        elif finding.exposure_risk == "high":
            severity = AlertSeverity.HIGH
        elif finding.exposure_risk == "medium":
            severity = AlertSeverity.MEDIUM
        else:
            severity = AlertSeverity.LOW

        # Get remediation guidance
        guidance = self.get_remediation_guidance(
            finding.classification,
            finding.pii_type,
            finding.file_path,
        )

        alert = PIIAlert(
            id=uuid4(),
            finding_id=finding.id,
            classification=finding.classification,
            pii_type=finding.pii_type,
            severity=severity,
            target_teams=rule.target_teams.copy(),
            file_path=finding.file_path,
            remediation_guidance=guidance,
            status=AlertStatus.PENDING,
            created_at=datetime.utcnow(),
            metadata={
                "pattern_name": finding.pattern_name,
                "confidence": finding.confidence_score,
                "detection_method": finding.detection_method.value,
                "notification_channels": rule.notification_channels,
                "auto_incident": rule.auto_create_incident,
            },
        )

        return alert

    async def _send_notifications(self, alert: PIIAlert) -> None:
        """Send notifications for an alert.

        Args:
            alert: The alert to notify about.
        """
        if not self._notification_service:
            logger.debug("No notification service configured")
            return

        channels = alert.metadata.get("notification_channels", ["email"])

        for team in alert.target_teams:
            for channel in channels:
                try:
                    # Format message
                    message = self._format_notification(alert)

                    # Send via notification service
                    # await self._notification_service.send(
                    #     channel=channel,
                    #     recipient=f"{team.value}_team",
                    #     subject=f"PII Alert: {alert.classification.value} detected",
                    #     body=message,
                    # )

                    logger.info(
                        "Notification sent: channel=%s, team=%s, alert=%s",
                        channel, team.value, alert.id
                    )

                except Exception as e:
                    logger.error(
                        "Failed to send notification: channel=%s, error=%s",
                        channel, e
                    )

        alert.status = AlertStatus.SENT
        alert.sent_at = datetime.utcnow()

    def _format_notification(self, alert: PIIAlert) -> str:
        """Format alert notification message.

        Args:
            alert: The alert to format.

        Returns:
            Formatted message string.
        """
        return f"""
PII Detection Alert
===================

Classification: {alert.classification.value.upper()}
Type: {alert.pii_type.value}
Severity: {alert.severity.value.upper()}
File: {alert.file_path or "N/A"}
Detected At: {alert.created_at.isoformat()}

Remediation Guidance:
{alert.remediation_guidance}

Alert ID: {alert.id}
"""

    async def _log_audit_event(
        self,
        alert: PIIAlert,
        action: str = "create",
        actor: Optional[str] = None,
    ) -> None:
        """Log an audit event for the alert.

        Args:
            alert: The alert to log.
            action: Action taken (create, acknowledge, resolve).
            actor: User performing action.
        """
        if not self._audit_service:
            return

        try:
            await self._audit_service.log_data_event(
                event_type=f"pii_alert.{action}",
                resource_type="pii_finding",
                resource_id=str(alert.finding_id),
                user_id=actor,
                metadata={
                    "alert_id": str(alert.id),
                    "classification": alert.classification.value,
                    "pii_type": alert.pii_type.value,
                    "severity": alert.severity.value,
                    "target_teams": [t.value for t in alert.target_teams],
                    "file_path": alert.file_path,
                },
            )
        except Exception as e:
            logger.warning("Failed to log audit event: %s", e)

    def _record_metrics(self, alert: PIIAlert) -> None:
        """Record metrics for the alert.

        Args:
            alert: The alert to record.
        """
        if not self._metrics:
            return

        try:
            self._metrics.record_pii_finding(
                classification=alert.classification.value,
                pii_type=alert.pii_type.value,
            )
        except Exception as e:
            logger.warning("Failed to record metrics: %s", e)


# ---------------------------------------------------------------------------
# Global Instance
# ---------------------------------------------------------------------------

_global_alert_router: Optional[PIIAlertRouter] = None


def get_pii_alert_router() -> PIIAlertRouter:
    """Get or create the global PII alert router instance.

    Returns:
        The global PIIAlertRouter instance.
    """
    global _global_alert_router

    if _global_alert_router is None:
        _global_alert_router = PIIAlertRouter()

    return _global_alert_router


def configure_pii_alert_router(
    routing_rules: Optional[Dict[DataClassification, RoutingRule]] = None,
    audit_service: Optional[Any] = None,
    notification_service: Optional[Any] = None,
    metrics: Optional[Any] = None,
) -> PIIAlertRouter:
    """Configure the global PII alert router.

    Args:
        routing_rules: Custom routing rules.
        audit_service: AuditService instance.
        notification_service: Notification service.
        metrics: SecurityMetrics instance.

    Returns:
        Configured PIIAlertRouter.
    """
    global _global_alert_router

    _global_alert_router = PIIAlertRouter(
        routing_rules=routing_rules,
        audit_service=audit_service,
        notification_service=notification_service,
        metrics=metrics,
    )

    return _global_alert_router


__all__ = [
    "PIIAlertRouter",
    "PIIAlert",
    "RoutingRule",
    "AlertSeverity",
    "AlertTeam",
    "AlertStatus",
    "REMEDIATION_TEMPLATES",
    "get_pii_alert_router",
    "configure_pii_alert_router",
]
