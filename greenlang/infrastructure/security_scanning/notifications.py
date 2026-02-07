# -*- coding: utf-8 -*-
"""
Notification Service - SEC-007

Multi-channel notification service for security findings and events.
Supports Slack, Email (SES/SMTP), PagerDuty, and Microsoft Teams webhooks.

Example:
    >>> config = NotificationConfig(
    ...     slack_webhook_url="https://hooks.slack.com/...",
    ...     pagerduty_routing_key="xxx",
    ... )
    >>> service = NotificationService(config)
    >>> await service.send_slack("#security", "Critical vulnerability found", "CRITICAL")
    >>> await service.send_pagerduty("CRITICAL", "CVE-2024-1234", details)

Author: GreenLang Security Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import json
import logging
import smtplib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Try to import httpx for async HTTP
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None

# Try to import boto3 for SES
try:
    import boto3
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    boto3 = None


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class NotificationChannel(str, Enum):
    """Supported notification channels."""

    SLACK = "slack"
    EMAIL = "email"
    PAGERDUTY = "pagerduty"
    TEAMS = "teams"
    WEBHOOK = "webhook"


class NotificationPriority(str, Enum):
    """Notification priority levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class NotificationConfig:
    """Configuration for the notification service.

    Attributes:
        slack_webhook_url: Slack incoming webhook URL.
        slack_default_channel: Default Slack channel.
        email_provider: Email provider (ses or smtp).
        email_from: From address for emails.
        smtp_host: SMTP server host.
        smtp_port: SMTP server port.
        smtp_username: SMTP authentication username.
        smtp_password: SMTP authentication password.
        smtp_use_tls: Use TLS for SMTP.
        ses_region: AWS region for SES.
        pagerduty_routing_key: PagerDuty Events API routing key.
        pagerduty_api_url: PagerDuty Events API URL.
        teams_webhook_url: Microsoft Teams webhook URL.
        enabled_channels: Set of enabled notification channels.
        severity_thresholds: Map channel to minimum severity.
        rate_limit_per_minute: Maximum notifications per minute.
        dry_run: If True, log notifications but don't send.
    """

    slack_webhook_url: Optional[str] = None
    slack_default_channel: str = "#security-alerts"
    email_provider: str = "smtp"
    email_from: str = "security@greenlang.io"
    smtp_host: str = "localhost"
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None
    smtp_use_tls: bool = True
    ses_region: str = "us-east-1"
    pagerduty_routing_key: Optional[str] = None
    pagerduty_api_url: str = "https://events.pagerduty.com/v2/enqueue"
    teams_webhook_url: Optional[str] = None
    enabled_channels: set = field(default_factory=lambda: {
        NotificationChannel.SLACK,
        NotificationChannel.EMAIL,
    })
    severity_thresholds: Dict[str, str] = field(default_factory=lambda: {
        "slack": "MEDIUM",
        "email": "HIGH",
        "pagerduty": "CRITICAL",
        "teams": "HIGH",
    })
    rate_limit_per_minute: int = 60
    dry_run: bool = False


# ---------------------------------------------------------------------------
# Notification Result
# ---------------------------------------------------------------------------


@dataclass
class NotificationResult:
    """Result of a notification attempt.

    Attributes:
        success: Whether the notification was sent successfully.
        channel: The notification channel used.
        message_id: Identifier for the sent message.
        error: Error message if failed.
        sent_at: Timestamp of the notification.
    """

    success: bool
    channel: NotificationChannel
    message_id: Optional[str] = None
    error: Optional[str] = None
    sent_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "channel": self.channel.value,
            "message_id": self.message_id,
            "error": self.error,
            "sent_at": self.sent_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# Notification Service
# ---------------------------------------------------------------------------


class NotificationService:
    """Multi-channel notification service for security events.

    Sends notifications via Slack, Email, PagerDuty, and Teams based on
    severity and configuration. Supports rate limiting and dry-run mode.

    Attributes:
        config: Notification configuration.
        _notification_count: Counter for rate limiting.
        _last_reset: Last rate limit reset time.

    Example:
        >>> service = NotificationService(config)
        >>> await service.send_slack("#security", "Vulnerability detected", "HIGH")
        >>> await service.notify_finding(finding)
    """

    def __init__(self, config: Optional[NotificationConfig] = None) -> None:
        """Initialize the notification service.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or NotificationConfig()
        self._notification_count = 0
        self._last_reset = datetime.now(timezone.utc)

    def _check_rate_limit(self) -> bool:
        """Check if we're within the rate limit.

        Returns:
            True if we can send, False if rate limited.
        """
        now = datetime.now(timezone.utc)
        if (now - self._last_reset).total_seconds() >= 60:
            self._notification_count = 0
            self._last_reset = now

        if self._notification_count >= self.config.rate_limit_per_minute:
            logger.warning("Rate limit exceeded for notifications")
            return False

        self._notification_count += 1
        return True

    def _should_notify(
        self, channel: NotificationChannel, severity: str
    ) -> bool:
        """Check if a notification should be sent based on severity threshold.

        Args:
            channel: The notification channel.
            severity: The severity level.

        Returns:
            True if notification should be sent.
        """
        severity_order = ["INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
        threshold = self.config.severity_thresholds.get(
            channel.value, "MEDIUM"
        ).upper()

        try:
            severity_idx = severity_order.index(severity.upper())
            threshold_idx = severity_order.index(threshold)
            return severity_idx >= threshold_idx
        except ValueError:
            return True

    # -------------------------------------------------------------------------
    # Slack Notifications
    # -------------------------------------------------------------------------

    async def send_slack(
        self,
        channel: str,
        message: str,
        severity: str = "INFO",
        title: Optional[str] = None,
        fields: Optional[Dict[str, str]] = None,
        thread_ts: Optional[str] = None,
    ) -> NotificationResult:
        """Send a Slack notification.

        Args:
            channel: Slack channel (e.g., "#security-alerts").
            message: Main message text.
            severity: Severity level for color coding.
            title: Optional title for the message.
            fields: Optional key-value fields to display.
            thread_ts: Optional thread timestamp for threading.

        Returns:
            NotificationResult with success status.

        Example:
            >>> await service.send_slack(
            ...     "#security",
            ...     "Critical vulnerability CVE-2024-1234 detected",
            ...     severity="CRITICAL",
            ...     fields={"Package": "requests", "Version": "2.31.0"},
            ... )
        """
        if not self.config.slack_webhook_url:
            return NotificationResult(
                success=False,
                channel=NotificationChannel.SLACK,
                error="Slack webhook URL not configured",
            )

        if not self._should_notify(NotificationChannel.SLACK, severity):
            return NotificationResult(
                success=True,
                channel=NotificationChannel.SLACK,
                message_id="skipped",
            )

        if not self._check_rate_limit():
            return NotificationResult(
                success=False,
                channel=NotificationChannel.SLACK,
                error="Rate limit exceeded",
            )

        # Build Slack message
        color = self._severity_to_slack_color(severity)
        attachment = {
            "color": color,
            "text": message,
            "footer": "GreenLang Security Scanner",
            "ts": int(datetime.now(timezone.utc).timestamp()),
        }

        if title:
            attachment["title"] = title

        if fields:
            attachment["fields"] = [
                {"title": k, "value": v, "short": True}
                for k, v in fields.items()
            ]

        payload = {
            "channel": channel or self.config.slack_default_channel,
            "attachments": [attachment],
        }

        if thread_ts:
            payload["thread_ts"] = thread_ts

        if self.config.dry_run:
            logger.info("DRY RUN - Slack notification: %s", json.dumps(payload))
            return NotificationResult(
                success=True,
                channel=NotificationChannel.SLACK,
                message_id="dry_run",
            )

        if not HTTPX_AVAILABLE:
            return NotificationResult(
                success=False,
                channel=NotificationChannel.SLACK,
                error="httpx not available",
            )

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.config.slack_webhook_url,
                    json=payload,
                    timeout=10.0,
                )

                if response.status_code == 200:
                    logger.info("Slack notification sent to %s", channel)
                    return NotificationResult(
                        success=True,
                        channel=NotificationChannel.SLACK,
                        message_id="sent",
                    )
                else:
                    return NotificationResult(
                        success=False,
                        channel=NotificationChannel.SLACK,
                        error=f"HTTP {response.status_code}: {response.text}",
                    )

        except Exception as e:
            logger.error("Failed to send Slack notification: %s", e)
            return NotificationResult(
                success=False,
                channel=NotificationChannel.SLACK,
                error=str(e),
            )

    def _severity_to_slack_color(self, severity: str) -> str:
        """Convert severity to Slack attachment color.

        Args:
            severity: Severity level.

        Returns:
            Hex color code.
        """
        colors = {
            "CRITICAL": "#ff0000",  # Red
            "HIGH": "#ff6600",  # Orange
            "MEDIUM": "#ffcc00",  # Yellow
            "LOW": "#00cc00",  # Green
            "INFO": "#0066ff",  # Blue
        }
        return colors.get(severity.upper(), "#808080")

    # -------------------------------------------------------------------------
    # Email Notifications
    # -------------------------------------------------------------------------

    async def send_email(
        self,
        to: Union[str, List[str]],
        subject: str,
        body: str,
        html_body: Optional[str] = None,
        severity: str = "INFO",
    ) -> NotificationResult:
        """Send an email notification.

        Args:
            to: Recipient email address(es).
            subject: Email subject line.
            body: Plain text email body.
            html_body: Optional HTML email body.
            severity: Severity for filtering.

        Returns:
            NotificationResult with success status.

        Example:
            >>> await service.send_email(
            ...     "security-team@example.com",
            ...     "Critical Vulnerability Detected",
            ...     "CVE-2024-1234 found in production",
            ...     severity="CRITICAL",
            ... )
        """
        if not self._should_notify(NotificationChannel.EMAIL, severity):
            return NotificationResult(
                success=True,
                channel=NotificationChannel.EMAIL,
                message_id="skipped",
            )

        if not self._check_rate_limit():
            return NotificationResult(
                success=False,
                channel=NotificationChannel.EMAIL,
                error="Rate limit exceeded",
            )

        recipients = [to] if isinstance(to, str) else to

        if self.config.dry_run:
            logger.info(
                "DRY RUN - Email notification: to=%s subject=%s",
                recipients,
                subject,
            )
            return NotificationResult(
                success=True,
                channel=NotificationChannel.EMAIL,
                message_id="dry_run",
            )

        if self.config.email_provider == "ses":
            return await self._send_email_ses(recipients, subject, body, html_body)
        else:
            return await self._send_email_smtp(recipients, subject, body, html_body)

    async def _send_email_smtp(
        self,
        recipients: List[str],
        subject: str,
        body: str,
        html_body: Optional[str],
    ) -> NotificationResult:
        """Send email via SMTP.

        Args:
            recipients: List of recipient addresses.
            subject: Email subject.
            body: Plain text body.
            html_body: Optional HTML body.

        Returns:
            NotificationResult.
        """
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = self.config.email_from
            msg["To"] = ", ".join(recipients)

            msg.attach(MIMEText(body, "plain"))
            if html_body:
                msg.attach(MIMEText(html_body, "html"))

            with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port) as server:
                if self.config.smtp_use_tls:
                    server.starttls()
                if self.config.smtp_username and self.config.smtp_password:
                    server.login(self.config.smtp_username, self.config.smtp_password)
                server.sendmail(
                    self.config.email_from,
                    recipients,
                    msg.as_string(),
                )

            logger.info("Email sent to %s", recipients)
            return NotificationResult(
                success=True,
                channel=NotificationChannel.EMAIL,
                message_id="smtp_sent",
            )

        except Exception as e:
            logger.error("Failed to send SMTP email: %s", e)
            return NotificationResult(
                success=False,
                channel=NotificationChannel.EMAIL,
                error=str(e),
            )

    async def _send_email_ses(
        self,
        recipients: List[str],
        subject: str,
        body: str,
        html_body: Optional[str],
    ) -> NotificationResult:
        """Send email via AWS SES.

        Args:
            recipients: List of recipient addresses.
            subject: Email subject.
            body: Plain text body.
            html_body: Optional HTML body.

        Returns:
            NotificationResult.
        """
        if not BOTO3_AVAILABLE:
            return NotificationResult(
                success=False,
                channel=NotificationChannel.EMAIL,
                error="boto3 not available for SES",
            )

        try:
            ses = boto3.client("ses", region_name=self.config.ses_region)

            email_body: Dict[str, Dict[str, str]] = {
                "Text": {"Data": body},
            }
            if html_body:
                email_body["Html"] = {"Data": html_body}

            response = ses.send_email(
                Source=self.config.email_from,
                Destination={"ToAddresses": recipients},
                Message={
                    "Subject": {"Data": subject},
                    "Body": email_body,
                },
            )

            message_id = response.get("MessageId", "ses_sent")
            logger.info("SES email sent: %s", message_id)
            return NotificationResult(
                success=True,
                channel=NotificationChannel.EMAIL,
                message_id=message_id,
            )

        except Exception as e:
            logger.error("Failed to send SES email: %s", e)
            return NotificationResult(
                success=False,
                channel=NotificationChannel.EMAIL,
                error=str(e),
            )

    # -------------------------------------------------------------------------
    # PagerDuty Notifications
    # -------------------------------------------------------------------------

    async def send_pagerduty(
        self,
        severity: str,
        title: str,
        details: Dict[str, Any],
        dedup_key: Optional[str] = None,
        links: Optional[List[Dict[str, str]]] = None,
    ) -> NotificationResult:
        """Send a PagerDuty alert.

        Args:
            severity: Severity level (critical, error, warning, info).
            title: Alert title/summary.
            details: Alert details (custom_details).
            dedup_key: Deduplication key for alert correlation.
            links: List of link objects with href and text.

        Returns:
            NotificationResult with success status.

        Example:
            >>> await service.send_pagerduty(
            ...     "critical",
            ...     "Critical vulnerability CVE-2024-1234",
            ...     {"package": "requests", "cve": "CVE-2024-1234"},
            ...     dedup_key="CVE-2024-1234",
            ... )
        """
        if not self.config.pagerduty_routing_key:
            return NotificationResult(
                success=False,
                channel=NotificationChannel.PAGERDUTY,
                error="PagerDuty routing key not configured",
            )

        if not self._should_notify(NotificationChannel.PAGERDUTY, severity):
            return NotificationResult(
                success=True,
                channel=NotificationChannel.PAGERDUTY,
                message_id="skipped",
            )

        if not self._check_rate_limit():
            return NotificationResult(
                success=False,
                channel=NotificationChannel.PAGERDUTY,
                error="Rate limit exceeded",
            )

        # Map severity to PagerDuty severity
        pd_severity = self._severity_to_pagerduty(severity)

        payload = {
            "routing_key": self.config.pagerduty_routing_key,
            "event_action": "trigger",
            "payload": {
                "summary": title[:1024],  # PD has 1024 char limit
                "severity": pd_severity,
                "source": "GreenLang Security Scanner",
                "component": "security-scanning",
                "group": "security",
                "class": "vulnerability",
                "custom_details": details,
            },
        }

        if dedup_key:
            payload["dedup_key"] = dedup_key

        if links:
            payload["links"] = links

        if self.config.dry_run:
            logger.info("DRY RUN - PagerDuty alert: %s", json.dumps(payload))
            return NotificationResult(
                success=True,
                channel=NotificationChannel.PAGERDUTY,
                message_id="dry_run",
            )

        if not HTTPX_AVAILABLE:
            return NotificationResult(
                success=False,
                channel=NotificationChannel.PAGERDUTY,
                error="httpx not available",
            )

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.config.pagerduty_api_url,
                    json=payload,
                    timeout=10.0,
                )

                if response.status_code in (200, 202):
                    result = response.json()
                    dedup_key = result.get("dedup_key", "")
                    logger.info(
                        "PagerDuty alert triggered: %s",
                        dedup_key,
                    )
                    return NotificationResult(
                        success=True,
                        channel=NotificationChannel.PAGERDUTY,
                        message_id=dedup_key,
                    )
                else:
                    return NotificationResult(
                        success=False,
                        channel=NotificationChannel.PAGERDUTY,
                        error=f"HTTP {response.status_code}: {response.text}",
                    )

        except Exception as e:
            logger.error("Failed to send PagerDuty alert: %s", e)
            return NotificationResult(
                success=False,
                channel=NotificationChannel.PAGERDUTY,
                error=str(e),
            )

    def _severity_to_pagerduty(self, severity: str) -> str:
        """Convert severity to PagerDuty severity.

        Args:
            severity: Our severity level.

        Returns:
            PagerDuty severity string.
        """
        mapping = {
            "CRITICAL": "critical",
            "HIGH": "error",
            "MEDIUM": "warning",
            "LOW": "info",
            "INFO": "info",
        }
        return mapping.get(severity.upper(), "warning")

    # -------------------------------------------------------------------------
    # Microsoft Teams Notifications
    # -------------------------------------------------------------------------

    async def send_teams(
        self,
        webhook_url: Optional[str],
        message: str,
        title: Optional[str] = None,
        severity: str = "INFO",
        facts: Optional[Dict[str, str]] = None,
        actions: Optional[List[Dict[str, str]]] = None,
    ) -> NotificationResult:
        """Send a Microsoft Teams notification.

        Args:
            webhook_url: Teams webhook URL (or uses config default).
            message: Main message text.
            title: Optional card title.
            severity: Severity level for theming.
            facts: Optional key-value facts to display.
            actions: Optional action buttons.

        Returns:
            NotificationResult with success status.

        Example:
            >>> await service.send_teams(
            ...     webhook_url,
            ...     "Critical vulnerability detected in production",
            ...     title="Security Alert",
            ...     severity="CRITICAL",
            ...     facts={"CVE": "CVE-2024-1234", "Package": "requests"},
            ... )
        """
        url = webhook_url or self.config.teams_webhook_url
        if not url:
            return NotificationResult(
                success=False,
                channel=NotificationChannel.TEAMS,
                error="Teams webhook URL not configured",
            )

        if not self._should_notify(NotificationChannel.TEAMS, severity):
            return NotificationResult(
                success=True,
                channel=NotificationChannel.TEAMS,
                message_id="skipped",
            )

        if not self._check_rate_limit():
            return NotificationResult(
                success=False,
                channel=NotificationChannel.TEAMS,
                error="Rate limit exceeded",
            )

        # Build Teams Adaptive Card
        color = self._severity_to_teams_color(severity)

        card = {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": color,
            "summary": title or "Security Alert",
            "sections": [
                {
                    "activityTitle": title or "Security Alert",
                    "activitySubtitle": f"Severity: {severity}",
                    "text": message,
                    "markdown": True,
                }
            ],
        }

        if facts:
            card["sections"][0]["facts"] = [
                {"name": k, "value": v} for k, v in facts.items()
            ]

        if actions:
            card["potentialAction"] = [
                {
                    "@type": "OpenUri",
                    "name": action.get("name", "View"),
                    "targets": [{"os": "default", "uri": action.get("url", "")}],
                }
                for action in actions
            ]

        if self.config.dry_run:
            logger.info("DRY RUN - Teams notification: %s", json.dumps(card))
            return NotificationResult(
                success=True,
                channel=NotificationChannel.TEAMS,
                message_id="dry_run",
            )

        if not HTTPX_AVAILABLE:
            return NotificationResult(
                success=False,
                channel=NotificationChannel.TEAMS,
                error="httpx not available",
            )

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url,
                    json=card,
                    timeout=10.0,
                )

                if response.status_code == 200:
                    logger.info("Teams notification sent")
                    return NotificationResult(
                        success=True,
                        channel=NotificationChannel.TEAMS,
                        message_id="sent",
                    )
                else:
                    return NotificationResult(
                        success=False,
                        channel=NotificationChannel.TEAMS,
                        error=f"HTTP {response.status_code}: {response.text}",
                    )

        except Exception as e:
            logger.error("Failed to send Teams notification: %s", e)
            return NotificationResult(
                success=False,
                channel=NotificationChannel.TEAMS,
                error=str(e),
            )

    def _severity_to_teams_color(self, severity: str) -> str:
        """Convert severity to Teams theme color.

        Args:
            severity: Severity level.

        Returns:
            Hex color code (without #).
        """
        colors = {
            "CRITICAL": "FF0000",
            "HIGH": "FF6600",
            "MEDIUM": "FFCC00",
            "LOW": "00CC00",
            "INFO": "0066FF",
        }
        return colors.get(severity.upper(), "808080")

    # -------------------------------------------------------------------------
    # High-Level Notification Methods
    # -------------------------------------------------------------------------

    async def notify_finding(
        self,
        finding: Dict[str, Any],
        channels: Optional[List[NotificationChannel]] = None,
    ) -> List[NotificationResult]:
        """Send notifications for a security finding to multiple channels.

        Args:
            finding: Security finding data.
            channels: Optional list of channels. Uses enabled channels if not provided.

        Returns:
            List of NotificationResults for each channel.

        Example:
            >>> finding = {
            ...     "id": "VULN-001",
            ...     "cve_id": "CVE-2024-1234",
            ...     "severity": "CRITICAL",
            ...     "description": "Remote code execution vulnerability",
            ...     "package_name": "requests",
            ... }
            >>> results = await service.notify_finding(finding)
        """
        results: List[NotificationResult] = []
        severity = finding.get("severity", "MEDIUM")
        channels = channels or list(self.config.enabled_channels)

        title = f"Security Finding: {finding.get('cve_id', finding.get('id', 'Unknown'))}"
        message = self._format_finding_message(finding)

        fields = {
            "Severity": severity,
            "Type": finding.get("type", "vulnerability"),
            "Scanner": finding.get("scanner", "unknown"),
        }

        if finding.get("package_name"):
            fields["Package"] = finding.get("package_name")
        if finding.get("file_path"):
            fields["File"] = finding.get("file_path")

        for channel in channels:
            if channel == NotificationChannel.SLACK:
                result = await self.send_slack(
                    self.config.slack_default_channel,
                    message,
                    severity=severity,
                    title=title,
                    fields=fields,
                )
                results.append(result)

            elif channel == NotificationChannel.TEAMS:
                result = await self.send_teams(
                    None,
                    message,
                    title=title,
                    severity=severity,
                    facts=fields,
                )
                results.append(result)

            elif channel == NotificationChannel.PAGERDUTY:
                result = await self.send_pagerduty(
                    severity,
                    title,
                    finding,
                    dedup_key=finding.get("cve_id") or finding.get("id"),
                )
                results.append(result)

            elif channel == NotificationChannel.EMAIL:
                result = await self.send_email(
                    self.config.email_from,  # Would need to configure recipients
                    f"[{severity}] {title}",
                    message,
                    severity=severity,
                )
                results.append(result)

        return results

    def _format_finding_message(self, finding: Dict[str, Any]) -> str:
        """Format a finding into a human-readable message.

        Args:
            finding: Security finding data.

        Returns:
            Formatted message string.
        """
        parts = []

        if finding.get("description"):
            parts.append(finding["description"])

        if finding.get("cve_id"):
            parts.append(f"\n*CVE:* {finding['cve_id']}")

        if finding.get("cvss_score"):
            parts.append(f"*CVSS:* {finding['cvss_score']}")

        if finding.get("package_name") and finding.get("current_version"):
            parts.append(
                f"*Package:* {finding['package_name']} @ {finding['current_version']}"
            )

        if finding.get("fixed_version"):
            parts.append(f"*Fixed in:* {finding['fixed_version']}")

        if finding.get("file_path"):
            parts.append(f"*File:* `{finding['file_path']}`")

        if finding.get("remediation"):
            parts.append(f"\n*Remediation:* {finding['remediation']}")

        return "\n".join(parts)

    async def notify_scan_complete(
        self,
        scan_summary: Dict[str, Any],
        channels: Optional[List[NotificationChannel]] = None,
    ) -> List[NotificationResult]:
        """Send notification that a security scan is complete.

        Args:
            scan_summary: Summary of the scan results.
            channels: Optional list of channels.

        Returns:
            List of NotificationResults.
        """
        results: List[NotificationResult] = []
        channels = channels or [NotificationChannel.SLACK]

        total = scan_summary.get("total_findings", 0)
        critical = scan_summary.get("critical_count", 0)
        high = scan_summary.get("high_count", 0)

        severity = "CRITICAL" if critical > 0 else ("HIGH" if high > 0 else "INFO")

        title = "Security Scan Complete"
        message = (
            f"Scan completed with {total} finding(s).\n"
            f"Critical: {critical}, High: {high}, "
            f"Medium: {scan_summary.get('medium_count', 0)}, "
            f"Low: {scan_summary.get('low_count', 0)}"
        )

        fields = {
            "Total Findings": str(total),
            "Critical": str(critical),
            "High": str(high),
            "Duration": scan_summary.get("duration", "N/A"),
        }

        for channel in channels:
            if channel == NotificationChannel.SLACK:
                result = await self.send_slack(
                    self.config.slack_default_channel,
                    message,
                    severity=severity,
                    title=title,
                    fields=fields,
                )
                results.append(result)

        return results
