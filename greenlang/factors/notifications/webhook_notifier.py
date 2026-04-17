# -*- coding: utf-8 -*-
"""
Webhook notification service for factors watch & release events.

Sends Slack and email notifications when the watch scheduler detects
source changes or errors, and when releases are prepared or published.

Integrates with the ``run_watch(notify=...)`` callback interface.
"""

from __future__ import annotations

import json
import logging
import smtplib
import urllib.error
import urllib.request
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Callable, Dict, List, Optional

from greenlang.factors.notifications.config import (
    NotificationChannel,
    NotificationConfig,
    NotificationEventType,
)

logger = logging.getLogger(__name__)


def send_slack_notification(
    webhook_url: str,
    message: str,
    *,
    channel: Optional[str] = None,
    attachments: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    """
    Send a Slack notification via incoming webhook.

    Returns True on success, False on failure (never raises).
    """
    payload: Dict[str, Any] = {"text": message}
    if channel:
        payload["channel"] = channel
    if attachments:
        payload["attachments"] = attachments

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:  # nosec B310
            if resp.status == 200:
                logger.info("Slack notification sent successfully")
                return True
            logger.warning("Slack webhook returned status=%d", resp.status)
            return False
    except (urllib.error.URLError, OSError) as exc:
        logger.error("Slack notification failed: %s", exc)
        return False


def send_email_notification(
    smtp_host: str,
    smtp_port: int,
    smtp_user: Optional[str],
    smtp_password: Optional[str],
    from_addr: str,
    to_addrs: List[str],
    subject: str,
    body: str,
) -> bool:
    """
    Send an email notification via SMTP.

    Returns True on success, False on failure (never raises).
    """
    if not to_addrs:
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = ", ".join(to_addrs)
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as server:
            server.ehlo()
            server.starttls()
            if smtp_user and smtp_password:
                server.login(smtp_user, smtp_password)
            server.sendmail(from_addr, to_addrs, msg.as_string())
        logger.info("Email notification sent to %s", to_addrs)
        return True
    except (smtplib.SMTPException, OSError) as exc:
        logger.error("Email notification failed: %s", exc)
        return False


class WebhookNotifier:
    """
    Central notification dispatcher for factors events.

    Routes events to Slack and/or email based on NotificationConfig routing.
    Tracks delivery status for audit logging.
    """

    def __init__(self, config: Optional[NotificationConfig] = None) -> None:
        self.config = config or NotificationConfig.from_env()
        self._delivery_log: List[Dict[str, Any]] = []

    @property
    def delivery_log(self) -> List[Dict[str, Any]]:
        return list(self._delivery_log)

    def notify(
        self,
        event_type: NotificationEventType,
        subject: str,
        body: str,
        *,
        attachments: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        Send notifications for an event.

        Returns the number of successfully delivered notifications.
        """
        if not self.config.enabled:
            logger.debug("Notifications disabled, skipping event=%s", event_type.value)
            return 0

        routes = self.config.routes_for_event(event_type)
        if not routes:
            logger.debug("No routes for event=%s", event_type.value)
            return 0

        sent = 0
        timestamp = datetime.now(timezone.utc).isoformat()

        for route in routes:
            for channel in route.channels:
                success = False
                error_msg = None

                if channel == NotificationChannel.SLACK and self.config.slack_configured:
                    slack_text = f"*{subject}*\n{body}"
                    success = send_slack_notification(
                        self.config.slack_webhook_url,  # type: ignore[arg-type]
                        slack_text,
                        channel=route.slack_channel,
                        attachments=attachments,
                    )
                elif channel == NotificationChannel.EMAIL and self.config.email_configured:
                    recipients = route.email_recipients or self.config.email_to
                    success = send_email_notification(
                        self.config.smtp_host,  # type: ignore[arg-type]
                        self.config.smtp_port,
                        self.config.smtp_user,
                        self.config.smtp_password,
                        self.config.email_from,  # type: ignore[arg-type]
                        recipients,
                        f"[GreenLang Factors] {subject}",
                        body,
                    )

                if success:
                    sent += 1

                self._delivery_log.append({
                    "timestamp": timestamp,
                    "event_type": event_type.value,
                    "channel": channel.value,
                    "subject": subject,
                    "success": success,
                    "error": error_msg,
                    "metadata": metadata or {},
                })

        return sent

    def notify_watch_change(self, source_id: str, change_type: str, details: str = "") -> int:
        return self.notify(
            NotificationEventType.WATCH_CHANGE,
            f"Source change detected: {source_id}",
            f"Change type: {change_type}\n{details}".strip(),
            metadata={"source_id": source_id, "change_type": change_type},
        )

    def notify_watch_error(self, source_id: str, error: str) -> int:
        return self.notify(
            NotificationEventType.WATCH_ERROR,
            f"Watch error: {source_id}",
            f"Error: {error}",
            metadata={"source_id": source_id, "error": error},
        )

    def notify_release_ready(self, edition_id: str, factor_count: int) -> int:
        return self.notify(
            NotificationEventType.RELEASE_READY,
            f"Release ready: {edition_id}",
            f"Edition {edition_id} with {factor_count} factors is ready for approval.",
            metadata={"edition_id": edition_id, "factor_count": factor_count},
        )

    def notify_release_published(self, edition_id: str, approved_by: str) -> int:
        return self.notify(
            NotificationEventType.RELEASE_PUBLISHED,
            f"Release published: {edition_id}",
            f"Edition {edition_id} promoted to stable by {approved_by}.",
            metadata={"edition_id": edition_id, "approved_by": approved_by},
        )

    def notify_policy_change(self, source_id: str, details: str) -> int:
        return self.notify(
            NotificationEventType.POLICY_CHANGE,
            f"Policy change: {source_id}",
            f"Methodology lead review required.\n{details}",
            metadata={"source_id": source_id},
        )


# ---------------------------------------------------------------------------
# Factory: build a notify callback compatible with run_watch()
# ---------------------------------------------------------------------------

NotifyCallback = Callable[[str, Any], None]


def build_watch_notify_callback(
    config: Optional[NotificationConfig] = None,
) -> NotifyCallback:
    """
    Build a notification callback compatible with ``run_watch(notify=...)``.

    The returned callable dispatches watch results to Slack/email based on
    the notification config.
    """
    notifier = WebhookNotifier(config)

    def _callback(message: str, watch_result: Any) -> None:
        source_id = getattr(watch_result, "source_id", "unknown")
        change_detected = getattr(watch_result, "change_detected", False)
        error_message = getattr(watch_result, "error_message", None)
        change_type = getattr(watch_result, "change_type", "unknown")

        if error_message:
            notifier.notify_watch_error(source_id, error_message)
        if change_detected:
            notifier.notify_watch_change(source_id, change_type, message)

    return _callback
