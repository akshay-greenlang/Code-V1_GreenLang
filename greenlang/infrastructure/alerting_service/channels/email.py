# -*- coding: utf-8 -*-
"""
Email Notification Channel - OBS-004: Unified Alerting Service

Delivers alert notifications via e-mail using either AWS SES (default)
or plain SMTP with STARTTLS. Emails are sent as HTML with inline CSS
for broad client compatibility.

Example:
    >>> channel = EmailChannel(
    ...     from_addr="alerts@greenlang.io",
    ...     use_ses=True,
    ...     ses_region="eu-west-1",
    ... )
    >>> result = await channel.send(alert, "CPU above 90%")

Author: GreenLang Platform Team
Date: February 2026
PRD: OBS-004 Unified Alerting Service
Status: Production Ready
"""

from __future__ import annotations

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, Optional

from greenlang.infrastructure.alerting_service.channels.base import (
    BaseNotificationChannel,
)
from greenlang.infrastructure.alerting_service.models import (
    Alert,
    AlertSeverity,
    NotificationResult,
    NotificationStatus,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional boto3 import
# ---------------------------------------------------------------------------

try:
    import boto3

    BOTO3_AVAILABLE = True
except ImportError:
    boto3 = None  # type: ignore[assignment]
    BOTO3_AVAILABLE = False
    logger.debug("boto3 not installed; SES email delivery unavailable")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEVERITY_COLOR: Dict[AlertSeverity, str] = {
    AlertSeverity.CRITICAL: "#DC3545",
    AlertSeverity.WARNING: "#FFC107",
    AlertSeverity.INFO: "#17A2B8",
}


# ---------------------------------------------------------------------------
# EmailChannel
# ---------------------------------------------------------------------------


class EmailChannel(BaseNotificationChannel):
    """Email notification channel via AWS SES or SMTP.

    Attributes:
        from_addr: Sender address.
        smtp_host: SMTP server hostname.
        smtp_port: SMTP server port.
        use_ses: If True, use AWS SES instead of SMTP.
        ses_region: AWS region for SES.
    """

    name = "email"

    def __init__(
        self,
        from_addr: str = "alerts@greenlang.io",
        smtp_host: str = "",
        smtp_port: int = 587,
        use_ses: bool = True,
        ses_region: str = "eu-west-1",
    ) -> None:
        self.from_addr = from_addr
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.use_ses = use_ses
        self.ses_region = ses_region
        self.enabled = (
            (use_ses and BOTO3_AVAILABLE) or bool(smtp_host)
        )

    # ------------------------------------------------------------------
    # BaseNotificationChannel interface
    # ------------------------------------------------------------------

    async def send(
        self,
        alert: Alert,
        rendered_message: str,
    ) -> NotificationResult:
        """Send an alert notification e-mail.

        The recipient is determined from ``alert.annotations["email"]``
        or ``alert.labels.get("email")``.  If neither is present, the
        notification is skipped.

        Args:
            alert: Alert to notify about.
            rendered_message: Pre-rendered description text.

        Returns:
            NotificationResult.
        """
        to_addr = (
            alert.annotations.get("email")
            or alert.labels.get("email")
            or ""
        )
        if not to_addr:
            return self._make_result(
                NotificationStatus.SKIPPED,
                error_message="No recipient email in alert annotations/labels",
            )

        subject = f"[{alert.severity.value.upper()}] {alert.title}"
        html_body = self._build_html(alert, rendered_message)

        start = self._timed()
        try:
            if self.use_ses and BOTO3_AVAILABLE:
                self._send_ses(to_addr, subject, html_body)
            else:
                self._send_smtp(to_addr, subject, html_body)

            duration_ms = (self._timed() - start) * 1000
            logger.info(
                "Email sent: to=%s, alert=%s", to_addr, alert.alert_id[:8],
            )
            return self._make_result(
                NotificationStatus.SENT,
                recipient=to_addr,
                duration_ms=duration_ms,
                response_code=200,
            )

        except Exception as exc:
            duration_ms = (self._timed() - start) * 1000
            logger.error("Email send error: %s", exc)
            return self._make_result(
                NotificationStatus.FAILED,
                recipient=to_addr,
                duration_ms=duration_ms,
                error_message=str(exc),
            )

    async def health_check(self) -> bool:
        """Verify email delivery capability.

        Returns:
            True if SES is available or SMTP is reachable.
        """
        if self.use_ses and BOTO3_AVAILABLE:
            try:
                ses = boto3.client("ses", region_name=self.ses_region)
                ses.get_send_quota()
                return True
            except Exception as exc:
                logger.warning("SES health check failed: %s", exc)
                return False

        if self.smtp_host:
            try:
                with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=5) as server:
                    server.ehlo()
                return True
            except Exception as exc:
                logger.warning("SMTP health check failed: %s", exc)
                return False

        return False

    # ------------------------------------------------------------------
    # SES delivery
    # ------------------------------------------------------------------

    def _send_ses(self, to: str, subject: str, html_body: str) -> None:
        """Send via AWS SES.

        Args:
            to: Recipient address.
            subject: Email subject.
            html_body: HTML message body.
        """
        ses = boto3.client("ses", region_name=self.ses_region)
        ses.send_email(
            Source=self.from_addr,
            Destination={"ToAddresses": [to]},
            Message={
                "Subject": {"Data": subject, "Charset": "UTF-8"},
                "Body": {
                    "Html": {"Data": html_body, "Charset": "UTF-8"},
                },
            },
        )

    # ------------------------------------------------------------------
    # SMTP delivery
    # ------------------------------------------------------------------

    def _send_smtp(self, to: str, subject: str, html_body: str) -> None:
        """Send via SMTP with STARTTLS.

        Args:
            to: Recipient address.
            subject: Email subject.
            html_body: HTML message body.
        """
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.from_addr
        msg["To"] = to
        msg.attach(MIMEText(html_body, "html", "utf-8"))

        with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=15) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.send_message(msg)

    # ------------------------------------------------------------------
    # HTML builder
    # ------------------------------------------------------------------

    def _build_html(self, alert: Alert, message: str) -> str:
        """Build an HTML email body with inline CSS.

        Args:
            alert: Source alert.
            message: Pre-rendered description.

        Returns:
            HTML string.
        """
        color = SEVERITY_COLOR.get(alert.severity, "#6C757D")
        links = ""
        if alert.runbook_url:
            links += (
                f'<a href="{alert.runbook_url}" '
                f'style="margin-right:12px;color:#0d6efd;">Runbook</a>'
            )
        if alert.dashboard_url:
            links += (
                f'<a href="{alert.dashboard_url}" '
                f'style="color:#0d6efd;">Dashboard</a>'
            )

        fired_at_str = ""
        if alert.fired_at:
            fired_at_str = alert.fired_at.strftime("%Y-%m-%d %H:%M:%S UTC")

        return f"""\
<!DOCTYPE html>
<html>
<head><meta charset="utf-8"/></head>
<body style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;">
  <div style="background:{color};color:white;padding:16px 20px;border-radius:4px 4px 0 0;">
    <h2 style="margin:0;">[{alert.severity.value.upper()}] {alert.name}</h2>
  </div>
  <div style="border:1px solid #dee2e6;border-top:none;padding:20px;border-radius:0 0 4px 4px;">
    <h3 style="margin-top:0;">{alert.title}</h3>
    <p style="color:#495057;">{message}</p>
    <table style="width:100%;border-collapse:collapse;margin:16px 0;">
      <tr><td style="padding:4px 8px;font-weight:bold;">Status</td><td>{alert.status.value}</td></tr>
      <tr><td style="padding:4px 8px;font-weight:bold;">Service</td><td>{alert.service}</td></tr>
      <tr><td style="padding:4px 8px;font-weight:bold;">Team</td><td>{alert.team}</td></tr>
      <tr><td style="padding:4px 8px;font-weight:bold;">Environment</td><td>{alert.environment}</td></tr>
      <tr><td style="padding:4px 8px;font-weight:bold;">Fired At</td><td>{fired_at_str}</td></tr>
    </table>
    <div style="margin-top:16px;">{links}</div>
    <hr style="border:none;border-top:1px solid #dee2e6;margin:20px 0;"/>
    <p style="font-size:12px;color:#6c757d;">
      GreenLang Unified Alerting Service - {alert.environment}
    </p>
  </div>
</body>
</html>"""
