# -*- coding: utf-8 -*-
"""
Incident Notifier - SEC-010

Sends notifications through multiple channels (PagerDuty, Slack, Email, SMS)
for incident alerts, escalations, and status updates. Uses rich formatting
and templates for clear, actionable notifications.

Example:
    >>> from greenlang.infrastructure.incident_response.notifier import (
    ...     Notifier,
    ... )
    >>> notifier = Notifier(config)
    >>> await notifier.notify_slack(incident)

Author: GreenLang Security Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

import httpx

from greenlang.infrastructure.incident_response.config import (
    IncidentResponseConfig,
    get_config,
)
from greenlang.infrastructure.incident_response.models import (
    Incident,
    IncidentStatus,
    EscalationLevel,
    PlaybookExecution,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Notification Templates
# ---------------------------------------------------------------------------

TEMPLATES: Dict[str, Dict[str, str]] = {
    "incident_created": {
        "subject": "[{severity}] Security Incident: {title}",
        "body": """
Security Incident Detected

Incident: {incident_number}
Title: {title}
Severity: {severity}
Type: {incident_type}
Status: {status}
Detected: {detected_at}

Description:
{description}

Affected Systems:
{affected_systems}

Please acknowledge and begin investigation immediately.

Dashboard: {dashboard_url}
""",
        "slack": """
:rotating_light: *Security Incident Detected*

*Incident:* `{incident_number}`
*Title:* {title}
*Severity:* {severity_emoji} `{severity}`
*Type:* {incident_type}
*Status:* {status}
*Detected:* {detected_at}

{description}

*Affected Systems:* {affected_systems}

<{dashboard_url}|View in Dashboard>
""",
    },
    "incident_acknowledged": {
        "subject": "[{severity}] Incident Acknowledged: {title}",
        "body": """
Incident Acknowledged

Incident: {incident_number}
Acknowledged by: {assignee_name}
Time to Acknowledge: {mttd}

The incident is now being investigated.

Dashboard: {dashboard_url}
""",
        "slack": """
:white_check_mark: *Incident Acknowledged*

*Incident:* `{incident_number}`
*Acknowledged by:* {assignee_name}
*Time to Acknowledge:* {mttd}

The incident is now being investigated.
""",
    },
    "incident_resolved": {
        "subject": "[{severity}] Incident Resolved: {title}",
        "body": """
Incident Resolved

Incident: {incident_number}
Title: {title}
Severity: {severity}
Time to Resolve: {mttr}

The incident has been resolved. A post-mortem will follow.

Dashboard: {dashboard_url}
""",
        "slack": """
:white_check_mark: *Incident Resolved*

*Incident:* `{incident_number}`
*Title:* {title}
*Severity:* {severity}
*Time to Resolve:* {mttr}

The incident has been resolved.
""",
    },
    "escalation": {
        "subject": "[ESCALATION] {severity} Incident: {title}",
        "body": """
INCIDENT ESCALATION

Incident: {incident_number}
Title: {title}
Severity: {severity}
Reason: {escalation_reason}

This incident requires immediate attention.

Dashboard: {dashboard_url}
""",
        "slack": """
:warning: *INCIDENT ESCALATION*

*Incident:* `{incident_number}`
*Title:* {title}
*Severity:* {severity_emoji} `{severity}`
*Reason:* {escalation_reason}

<!here> This incident requires immediate attention.

<{dashboard_url}|View in Dashboard>
""",
    },
    "playbook_started": {
        "subject": "Playbook Started: {playbook_name}",
        "body": """
Automated Playbook Started

Incident: {incident_number}
Playbook: {playbook_name}
Steps: {steps_total}

The automated response playbook has been initiated.
""",
        "slack": """
:robot_face: *Playbook Started*

*Incident:* `{incident_number}`
*Playbook:* {playbook_name}
*Steps:* {steps_total}

Automated remediation in progress...
""",
    },
    "playbook_completed": {
        "subject": "Playbook Completed: {playbook_name}",
        "body": """
Automated Playbook Completed

Incident: {incident_number}
Playbook: {playbook_name}
Status: {playbook_status}
Duration: {duration}

{execution_summary}
""",
        "slack": """
:white_check_mark: *Playbook Completed*

*Incident:* `{incident_number}`
*Playbook:* {playbook_name}
*Status:* {playbook_status}
*Duration:* {duration}
""",
    },
    "playbook_failed": {
        "subject": "[ALERT] Playbook Failed: {playbook_name}",
        "body": """
PLAYBOOK EXECUTION FAILED

Incident: {incident_number}
Playbook: {playbook_name}
Failed at Step: {failed_step}
Error: {error_message}

Manual intervention required.
""",
        "slack": """
:x: *Playbook Failed*

*Incident:* `{incident_number}`
*Playbook:* {playbook_name}
*Failed at Step:* {failed_step}
*Error:* {error_message}

<!channel> Manual intervention required.
""",
    },
}

SEVERITY_EMOJI: Dict[str, str] = {
    "P0": ":red_circle:",
    "P1": ":large_orange_circle:",
    "P2": ":large_yellow_circle:",
    "P3": ":white_circle:",
}


# ---------------------------------------------------------------------------
# Notifier
# ---------------------------------------------------------------------------


class Notifier:
    """Sends incident notifications through multiple channels.

    Supports PagerDuty Events API v2, Slack webhooks, email via SES,
    and SMS via SNS. Uses templates for consistent, rich formatting.

    Attributes:
        config: Incident response configuration.
        http_client: Async HTTP client for API calls.

    Example:
        >>> notifier = Notifier(config)
        >>> await notifier.notify_slack(incident)
        >>> await notifier.notify_all(incident, "incident_created")
    """

    def __init__(
        self,
        config: Optional[IncidentResponseConfig] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        """Initialize the notifier.

        Args:
            config: Incident response configuration.
            http_client: Optional HTTP client (created if not provided).
        """
        self.config = config or get_config()
        self._http_client = http_client
        self._owns_client = http_client is None

        logger.info("Notifier initialized")

    async def _get_http_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client.

        Returns:
            Async HTTP client instance.
        """
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    async def close(self) -> None:
        """Close HTTP client if owned by this instance."""
        if self._owns_client and self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    async def notify_pagerduty(
        self,
        incident: Incident,
        action: str = "trigger",
        custom_details: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Send notification to PagerDuty.

        Uses PagerDuty Events API v2 to create, acknowledge, or resolve
        incidents in PagerDuty.

        Args:
            incident: Incident to notify about.
            action: PagerDuty action (trigger, acknowledge, resolve).
            custom_details: Additional details to include.

        Returns:
            True if notification was sent successfully.
        """
        if not self.config.pagerduty.enabled:
            logger.debug("PagerDuty notifications disabled")
            return False

        routing_key = self.config.pagerduty.get_routing_key()
        if not routing_key:
            logger.warning("PagerDuty routing key not configured")
            return False

        try:
            client = await self._get_http_client()

            # Build dedup key
            dedup_key = f"{self.config.pagerduty.dedup_key_prefix}-{incident.incident_number}"

            # Map severity to PagerDuty severity
            pd_severity_map = {
                EscalationLevel.P0: "critical",
                EscalationLevel.P1: "error",
                EscalationLevel.P2: "warning",
                EscalationLevel.P3: "info",
            }
            pd_severity = pd_severity_map.get(incident.severity, "error")

            # Build payload
            payload: Dict[str, Any] = {
                "routing_key": routing_key,
                "event_action": action,
                "dedup_key": dedup_key,
            }

            if action == "trigger":
                payload["payload"] = {
                    "summary": f"[{incident.severity.value}] {incident.title}",
                    "severity": pd_severity,
                    "source": "GreenLang Security",
                    "component": "incident-response",
                    "group": incident.incident_type.value,
                    "class": incident.severity.value,
                    "custom_details": {
                        "incident_number": incident.incident_number,
                        "incident_type": incident.incident_type.value,
                        "description": incident.description,
                        "affected_systems": incident.affected_systems,
                        "detected_at": incident.detected_at.isoformat(),
                        **(custom_details or {}),
                    },
                }

                # Add links
                payload["links"] = [
                    {
                        "href": self._get_dashboard_url(incident),
                        "text": "View in Dashboard",
                    },
                ]

            response = await client.post(
                self.config.pagerduty.base_url,
                json=payload,
            )
            response.raise_for_status()

            logger.info(
                "PagerDuty notification sent for %s (action=%s)",
                incident.incident_number,
                action,
            )
            return True

        except httpx.HTTPError as e:
            logger.error("Failed to send PagerDuty notification: %s", e)
            return False
        except Exception as e:
            logger.error("Unexpected error sending PagerDuty notification: %s", e)
            return False

    async def notify_slack(
        self,
        incident: Incident,
        template_name: str = "incident_created",
        channel_override: Optional[str] = None,
    ) -> bool:
        """Send notification to Slack.

        Uses Slack Incoming Webhook for rich-formatted messages
        with severity-based channel routing.

        Args:
            incident: Incident to notify about.
            template_name: Template to use for formatting.
            channel_override: Override default channel.

        Returns:
            True if notification was sent successfully.
        """
        if not self.config.slack.enabled:
            logger.debug("Slack notifications disabled")
            return False

        webhook_url = self.config.slack.get_webhook_url()
        if not webhook_url:
            logger.warning("Slack webhook URL not configured")
            return False

        try:
            client = await self._get_http_client()

            # Get template
            template = TEMPLATES.get(template_name, TEMPLATES["incident_created"])
            slack_template = template.get("slack", template["body"])

            # Format message
            message = self._format_template(slack_template, incident)

            # Determine channel
            channel = channel_override
            if not channel:
                channel = self.config.slack.channels_by_severity.get(
                    incident.severity.value,
                    self.config.slack.default_channel,
                )

            # Build payload
            payload: Dict[str, Any] = {
                "channel": channel,
                "text": message,
                "unfurl_links": False,
                "unfurl_media": False,
            }

            # Add mentions for high-severity incidents
            if incident.severity in (EscalationLevel.P0, EscalationLevel.P1):
                mentions = self.config.slack.mention_users.get(
                    incident.severity.value, []
                )
                if mentions:
                    mention_str = " ".join(f"<@{uid}>" for uid in mentions)
                    payload["text"] = f"{mention_str}\n\n{message}"

            response = await client.post(webhook_url, json=payload)
            response.raise_for_status()

            logger.info(
                "Slack notification sent for %s to %s",
                incident.incident_number,
                channel,
            )
            return True

        except httpx.HTTPError as e:
            logger.error("Failed to send Slack notification: %s", e)
            return False
        except Exception as e:
            logger.error("Unexpected error sending Slack notification: %s", e)
            return False

    async def notify_email(
        self,
        incident: Incident,
        template_name: str = "incident_created",
        recipients_override: Optional[List[str]] = None,
    ) -> bool:
        """Send notification via email (AWS SES).

        Args:
            incident: Incident to notify about.
            template_name: Template to use for formatting.
            recipients_override: Override default recipients.

        Returns:
            True if notification was sent successfully.
        """
        if not self.config.email.enabled:
            logger.debug("Email notifications disabled")
            return False

        try:
            import boto3
            from botocore.exceptions import ClientError
        except ImportError:
            logger.warning("boto3 not available - skipping email notification")
            return False

        try:
            # Get template
            template = TEMPLATES.get(template_name, TEMPLATES["incident_created"])
            subject = self._format_template(template["subject"], incident)
            body = self._format_template(template["body"], incident)

            # Get recipients
            recipients = recipients_override
            if not recipients:
                recipients = self.config.email.recipients_by_severity.get(
                    incident.severity.value,
                    self.config.email.recipients_by_severity.get("P3", []),
                )

            if not recipients:
                logger.warning("No email recipients configured")
                return False

            # Send via SES
            ses_client = boto3.client("ses", region_name="us-east-1")

            response = ses_client.send_email(
                Source=self.config.email.from_address,
                Destination={
                    "ToAddresses": recipients,
                },
                Message={
                    "Subject": {"Data": subject, "Charset": "UTF-8"},
                    "Body": {
                        "Text": {"Data": body, "Charset": "UTF-8"},
                    },
                },
                ReplyToAddresses=[self.config.email.reply_to],
            )

            logger.info(
                "Email notification sent for %s to %d recipients",
                incident.incident_number,
                len(recipients),
            )
            return True

        except ClientError as e:
            logger.error("Failed to send email via SES: %s", e)
            return False
        except Exception as e:
            logger.error("Unexpected error sending email: %s", e)
            return False

    async def notify_sms(
        self,
        incident: Incident,
        phone_numbers_override: Optional[List[str]] = None,
    ) -> bool:
        """Send SMS notification via AWS SNS.

        Only used for P0 incidents by default.

        Args:
            incident: Incident to notify about.
            phone_numbers_override: Override default phone numbers.

        Returns:
            True if notification was sent successfully.
        """
        if not self.config.sms.enabled:
            logger.debug("SMS notifications disabled")
            return False

        # Only send SMS for P0 by default
        if incident.severity != EscalationLevel.P0:
            logger.debug("SMS only for P0 incidents")
            return False

        try:
            import boto3
            from botocore.exceptions import ClientError
        except ImportError:
            logger.warning("boto3 not available - skipping SMS notification")
            return False

        try:
            # Get phone numbers
            phone_numbers = phone_numbers_override
            if not phone_numbers:
                phone_numbers = self.config.sms.phone_numbers_by_severity.get(
                    incident.severity.value, []
                )

            if not phone_numbers:
                logger.debug("No SMS recipients configured for %s", incident.severity.value)
                return False

            # Compose short message
            message = (
                f"[{incident.severity.value}] {incident.incident_number}: "
                f"{incident.title[:100]}"
            )

            sns_client = boto3.client("sns", region_name=self.config.sms.region)

            sent_count = 0
            for phone in phone_numbers:
                try:
                    sns_client.publish(
                        PhoneNumber=phone,
                        Message=message,
                        MessageAttributes={
                            "AWS.SNS.SMS.SMSType": {
                                "DataType": "String",
                                "StringValue": "Transactional",
                            },
                        },
                    )
                    sent_count += 1
                except ClientError as e:
                    logger.warning("Failed to send SMS to %s: %s", phone, e)

            logger.info(
                "SMS notification sent for %s to %d numbers",
                incident.incident_number,
                sent_count,
            )
            return sent_count > 0

        except Exception as e:
            logger.error("Unexpected error sending SMS: %s", e)
            return False

    async def notify_all(
        self,
        incident: Incident,
        template_name: str = "incident_created",
    ) -> Dict[str, bool]:
        """Send notifications through all appropriate channels.

        Determines which channels to use based on incident severity.

        Args:
            incident: Incident to notify about.
            template_name: Template to use for formatting.

        Returns:
            Dictionary mapping channel names to success status.
        """
        results: Dict[str, bool] = {}

        # Determine channels based on severity
        channels = self._get_channels_for_severity(incident.severity)

        # Send to each channel
        if "pagerduty" in channels:
            results["pagerduty"] = await self.notify_pagerduty(incident)

        if "slack" in channels:
            results["slack"] = await self.notify_slack(incident, template_name)

        if "email" in channels:
            results["email"] = await self.notify_email(incident, template_name)

        if "sms" in channels:
            results["sms"] = await self.notify_sms(incident)

        success_count = sum(1 for v in results.values() if v)
        logger.info(
            "Sent notifications for %s: %d/%d successful",
            incident.incident_number,
            success_count,
            len(results),
        )

        return results

    async def notify_playbook_status(
        self,
        incident: Incident,
        execution: PlaybookExecution,
    ) -> Dict[str, bool]:
        """Send playbook status notification.

        Args:
            incident: Associated incident.
            execution: Playbook execution details.

        Returns:
            Dictionary mapping channel names to success status.
        """
        # Determine template based on status
        from greenlang.infrastructure.incident_response.models import PlaybookStatus

        if execution.status == PlaybookStatus.RUNNING:
            template_name = "playbook_started"
        elif execution.status == PlaybookStatus.COMPLETED:
            template_name = "playbook_completed"
        elif execution.status == PlaybookStatus.FAILED:
            template_name = "playbook_failed"
        else:
            return {}

        # Create combined context
        context = self._build_template_context(incident)
        context.update({
            "playbook_name": execution.playbook_name,
            "playbook_status": execution.status.value,
            "steps_total": execution.steps_total,
            "steps_completed": execution.steps_completed,
            "duration": self._format_duration(execution.get_duration_seconds()),
        })

        if execution.status == PlaybookStatus.FAILED:
            context["failed_step"] = execution.current_step or "unknown"
            context["error_message"] = (
                execution.execution_log[-1].get("message", "Unknown error")
                if execution.execution_log
                else "Unknown error"
            )

        results: Dict[str, bool] = {}

        # Only Slack for playbook updates
        if self.config.slack.enabled:
            template = TEMPLATES.get(template_name, {})
            slack_template = template.get("slack", "")
            message = self._format_dict_template(slack_template, context)

            results["slack"] = await self._send_slack_message(
                message,
                self.config.slack.default_channel,
            )

        return results

    def _get_channels_for_severity(
        self,
        severity: EscalationLevel,
    ) -> List[str]:
        """Get notification channels for severity level.

        Args:
            severity: Incident severity.

        Returns:
            List of channel names.
        """
        if severity == EscalationLevel.P0:
            return ["pagerduty", "slack", "sms", "email"]
        elif severity == EscalationLevel.P1:
            return ["pagerduty", "slack", "email"]
        elif severity == EscalationLevel.P2:
            return ["slack", "email"]
        else:
            return ["email"]

    def _format_template(
        self,
        template: str,
        incident: Incident,
    ) -> str:
        """Format a template with incident data.

        Args:
            template: Template string.
            incident: Incident to use for formatting.

        Returns:
            Formatted string.
        """
        context = self._build_template_context(incident)
        return self._format_dict_template(template, context)

    def _format_dict_template(
        self,
        template: str,
        context: Dict[str, Any],
    ) -> str:
        """Format a template with dictionary context.

        Args:
            template: Template string.
            context: Context dictionary.

        Returns:
            Formatted string.
        """
        try:
            return template.format(**context)
        except KeyError as e:
            logger.warning("Missing template key: %s", e)
            # Return partially formatted template
            return template

    def _build_template_context(self, incident: Incident) -> Dict[str, Any]:
        """Build template context from incident.

        Args:
            incident: Incident to extract context from.

        Returns:
            Context dictionary.
        """
        mttd = incident.get_mttd_seconds()
        mttr = incident.get_mttr_seconds()

        return {
            "incident_number": incident.incident_number,
            "title": incident.title,
            "description": incident.description or "No description provided",
            "severity": incident.severity.value,
            "severity_emoji": SEVERITY_EMOJI.get(incident.severity.value, ":white_circle:"),
            "status": incident.status.value,
            "incident_type": incident.incident_type.value.replace("_", " ").title(),
            "detected_at": incident.detected_at.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "affected_systems": ", ".join(incident.affected_systems) or "None specified",
            "assignee_name": incident.assignee_name or "Unassigned",
            "mttd": self._format_duration(mttd),
            "mttr": self._format_duration(mttr),
            "dashboard_url": self._get_dashboard_url(incident),
            "escalation_reason": incident.metadata.get("escalation_reason", ""),
        }

    def _get_dashboard_url(self, incident: Incident) -> str:
        """Get dashboard URL for incident.

        Args:
            incident: Incident to link to.

        Returns:
            Dashboard URL string.
        """
        # In production, this would be configurable
        return f"https://dashboard.greenlang.io/incidents/{incident.id}"

    def _format_duration(self, seconds: Optional[float]) -> str:
        """Format duration in seconds to human-readable string.

        Args:
            seconds: Duration in seconds.

        Returns:
            Formatted duration string.
        """
        if seconds is None:
            return "N/A"

        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.1f}m"
        else:
            return f"{seconds / 3600:.1f}h"

    async def _send_slack_message(
        self,
        message: str,
        channel: str,
    ) -> bool:
        """Send a raw Slack message.

        Args:
            message: Message text.
            channel: Target channel.

        Returns:
            True if successful.
        """
        webhook_url = self.config.slack.get_webhook_url()
        if not webhook_url:
            return False

        try:
            client = await self._get_http_client()
            response = await client.post(
                webhook_url,
                json={"channel": channel, "text": message},
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error("Failed to send Slack message: %s", e)
            return False


# ---------------------------------------------------------------------------
# Global Notifier Instance
# ---------------------------------------------------------------------------

_global_notifier: Optional[Notifier] = None


def get_notifier(
    config: Optional[IncidentResponseConfig] = None,
) -> Notifier:
    """Get or create the global notifier.

    Args:
        config: Optional configuration override.

    Returns:
        The global Notifier instance.
    """
    global _global_notifier

    if _global_notifier is None:
        _global_notifier = Notifier(config)

    return _global_notifier


async def reset_notifier() -> None:
    """Reset and close the global notifier."""
    global _global_notifier

    if _global_notifier is not None:
        await _global_notifier.close()
        _global_notifier = None


__all__ = [
    "Notifier",
    "TEMPLATES",
    "SEVERITY_EMOJI",
    "get_notifier",
    "reset_notifier",
]
