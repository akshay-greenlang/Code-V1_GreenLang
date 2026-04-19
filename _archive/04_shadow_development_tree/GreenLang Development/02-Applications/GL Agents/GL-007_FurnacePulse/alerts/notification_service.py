"""
NotificationService - Multi-Channel Alert Notification for FurnacePulse

This module implements the NotificationService for delivering alerts through
multiple channels including email, SMS, webhooks, and collaboration platforms
(Teams/Slack). It handles priority-based routing, acknowledgement tracking,
and escalation timers.

Example:
    >>> config = NotificationConfig(...)
    >>> service = NotificationService(config)
    >>> result = service.send_notification(alert, [NotificationChannel.EMAIL])
    >>> print(result.status)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Type
from uuid import uuid4

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class NotificationChannel(str, Enum):
    """Supported notification channels."""

    EMAIL = "EMAIL"
    SMS = "SMS"
    WEBHOOK = "WEBHOOK"
    TEAMS = "TEAMS"
    SLACK = "SLACK"
    PUSH = "PUSH"
    PAGER = "PAGER"


class NotificationPriority(str, Enum):
    """Notification priority levels affecting delivery behavior."""

    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    URGENT = "URGENT"
    EMERGENCY = "EMERGENCY"


class NotificationStatus(str, Enum):
    """Notification delivery status."""

    PENDING = "PENDING"
    QUEUED = "QUEUED"
    SENDING = "SENDING"
    SENT = "SENT"
    DELIVERED = "DELIVERED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    FAILED = "FAILED"
    EXPIRED = "EXPIRED"
    RETRYING = "RETRYING"


class AcknowledgementStatus(str, Enum):
    """Acknowledgement tracking status."""

    PENDING = "PENDING"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    ESCALATED = "ESCALATED"
    EXPIRED = "EXPIRED"


@dataclass
class ChannelConfig:
    """Configuration for a notification channel."""

    channel: NotificationChannel
    enabled: bool = True
    endpoint: str = ""
    api_key: Optional[str] = None
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay_seconds: int = 5
    rate_limit_per_minute: int = 60
    priority_threshold: NotificationPriority = NotificationPriority.LOW
    batch_enabled: bool = False
    batch_size: int = 10
    batch_window_seconds: int = 60


class NotificationRecipient(BaseModel):
    """Recipient information for notifications."""

    recipient_id: str = Field(..., description="Unique recipient identifier")
    name: str = Field(..., description="Display name")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number for SMS")
    teams_id: Optional[str] = Field(None, description="Microsoft Teams user ID")
    slack_id: Optional[str] = Field(None, description="Slack user ID")
    webhook_url: Optional[str] = Field(None, description="Personal webhook URL")
    preferred_channels: List[NotificationChannel] = Field(
        default_factory=lambda: [NotificationChannel.EMAIL]
    )
    do_not_disturb_start: Optional[str] = Field(None, description="DND start time HH:MM")
    do_not_disturb_end: Optional[str] = Field(None, description="DND end time HH:MM")
    escalation_contact: Optional[str] = Field(None, description="Escalation recipient ID")


class NotificationPayload(BaseModel):
    """Payload for a notification message."""

    notification_id: str = Field(default_factory=lambda: str(uuid4()))
    alert_id: str = Field(..., description="Associated alert ID")
    alert_code: str = Field(..., description="Alert taxonomy code")
    severity: str = Field(..., description="Alert severity")
    title: str = Field(..., description="Notification title")
    message: str = Field(..., description="Notification body")
    priority: NotificationPriority = Field(default=NotificationPriority.NORMAL)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = Field(None, description="Expiration time")
    acknowledgement_required: bool = Field(default=False)
    acknowledgement_url: Optional[str] = Field(None, description="URL for acknowledgement")
    action_buttons: List[Dict[str, str]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="", description="SHA-256 hash for audit")


class NotificationResult(BaseModel):
    """Result of a notification delivery attempt."""

    notification_id: str = Field(..., description="Notification identifier")
    channel: NotificationChannel = Field(..., description="Delivery channel")
    recipient_id: str = Field(..., description="Recipient identifier")
    status: NotificationStatus = Field(..., description="Delivery status")
    sent_at: Optional[datetime] = Field(None, description="When sent")
    delivered_at: Optional[datetime] = Field(None, description="When delivered")
    error_message: Optional[str] = Field(None, description="Error if failed")
    retry_count: int = Field(default=0, description="Number of retries")
    response_data: Dict[str, Any] = Field(default_factory=dict)


class AcknowledgementRecord(BaseModel):
    """Record of notification acknowledgement."""

    acknowledgement_id: str = Field(default_factory=lambda: str(uuid4()))
    notification_id: str = Field(..., description="Notification identifier")
    alert_id: str = Field(..., description="Associated alert ID")
    recipient_id: str = Field(..., description="Recipient who acknowledged")
    status: AcknowledgementStatus = Field(default=AcknowledgementStatus.PENDING)
    required_by: datetime = Field(..., description="Acknowledgement deadline")
    acknowledged_at: Optional[datetime] = Field(None)
    escalated_at: Optional[datetime] = Field(None)
    escalated_to: Optional[str] = Field(None, description="Escalation recipient ID")
    notes: Optional[str] = Field(None)


class NotificationConfig(BaseModel):
    """Configuration for NotificationService."""

    channels: Dict[NotificationChannel, ChannelConfig] = Field(default_factory=dict)
    default_expiry_minutes: int = Field(default=60, ge=1)
    acknowledgement_timeout_minutes: int = Field(default=15, ge=1)
    max_escalation_levels: int = Field(default=3, ge=1)
    enable_rate_limiting: bool = Field(default=True)
    enable_deduplication: bool = Field(default=True)
    deduplication_window_seconds: int = Field(default=60)
    async_delivery: bool = Field(default=True)
    log_all_notifications: bool = Field(default=True)

    @validator("channels", pre=True, always=True)
    def set_default_channels(cls, v):
        """Set default channel configurations if not provided."""
        if not v:
            return {
                NotificationChannel.EMAIL: ChannelConfig(
                    channel=NotificationChannel.EMAIL,
                    enabled=True,
                    endpoint="smtp://localhost:25",
                ),
                NotificationChannel.WEBHOOK: ChannelConfig(
                    channel=NotificationChannel.WEBHOOK,
                    enabled=True,
                ),
            }
        return v


class ChannelProvider(ABC):
    """Abstract base class for channel-specific notification providers."""

    def __init__(self, config: ChannelConfig):
        """Initialize the channel provider."""
        self.config = config
        self.last_send_times: List[datetime] = []

    @abstractmethod
    async def send(
        self, payload: NotificationPayload, recipient: NotificationRecipient
    ) -> NotificationResult:
        """Send a notification through this channel."""
        pass

    @abstractmethod
    def format_message(
        self, payload: NotificationPayload, recipient: NotificationRecipient
    ) -> Dict[str, Any]:
        """Format the notification for this channel."""
        pass

    def check_rate_limit(self) -> bool:
        """Check if rate limit allows sending."""
        now = datetime.utcnow()
        cutoff = now - timedelta(minutes=1)
        self.last_send_times = [t for t in self.last_send_times if t > cutoff]
        return len(self.last_send_times) < self.config.rate_limit_per_minute

    def record_send(self) -> None:
        """Record a send for rate limiting."""
        self.last_send_times.append(datetime.utcnow())


class EmailProvider(ChannelProvider):
    """Email notification provider."""

    async def send(
        self, payload: NotificationPayload, recipient: NotificationRecipient
    ) -> NotificationResult:
        """Send email notification."""
        notification_id = payload.notification_id
        result = NotificationResult(
            notification_id=notification_id,
            channel=NotificationChannel.EMAIL,
            recipient_id=recipient.recipient_id,
            status=NotificationStatus.PENDING,
        )

        try:
            if not recipient.email:
                result.status = NotificationStatus.FAILED
                result.error_message = "Recipient has no email address"
                return result

            if not self.check_rate_limit():
                result.status = NotificationStatus.RETRYING
                result.error_message = "Rate limit exceeded"
                return result

            formatted = self.format_message(payload, recipient)

            # In production, this would use aiosmtplib or similar
            # For now, we simulate the send
            logger.info(
                "Sending email to %s: %s", recipient.email, formatted["subject"]
            )

            # Simulate async send
            await asyncio.sleep(0.1)

            result.status = NotificationStatus.SENT
            result.sent_at = datetime.utcnow()
            self.record_send()

            logger.info("Email sent successfully: %s", notification_id)

        except Exception as e:
            result.status = NotificationStatus.FAILED
            result.error_message = str(e)
            logger.error("Email send failed: %s - %s", notification_id, e)

        return result

    def format_message(
        self, payload: NotificationPayload, recipient: NotificationRecipient
    ) -> Dict[str, Any]:
        """Format email message."""
        priority_prefix = ""
        if payload.priority in (NotificationPriority.URGENT, NotificationPriority.EMERGENCY):
            priority_prefix = "[URGENT] "

        body = f"""
{payload.message}

Alert Details:
- Alert ID: {payload.alert_id}
- Alert Code: {payload.alert_code}
- Severity: {payload.severity}
- Time: {payload.created_at.isoformat()}

"""
        if payload.acknowledgement_required and payload.acknowledgement_url:
            body += f"\nAcknowledge this alert: {payload.acknowledgement_url}\n"

        if payload.action_buttons:
            body += "\nActions:\n"
            for button in payload.action_buttons:
                body += f"- {button.get('label', 'Action')}: {button.get('url', 'N/A')}\n"

        return {
            "to": recipient.email,
            "subject": f"{priority_prefix}{payload.title}",
            "body": body,
            "priority": "high" if payload.priority.value in ("URGENT", "EMERGENCY") else "normal",
        }


class SMSProvider(ChannelProvider):
    """SMS notification provider."""

    async def send(
        self, payload: NotificationPayload, recipient: NotificationRecipient
    ) -> NotificationResult:
        """Send SMS notification."""
        notification_id = payload.notification_id
        result = NotificationResult(
            notification_id=notification_id,
            channel=NotificationChannel.SMS,
            recipient_id=recipient.recipient_id,
            status=NotificationStatus.PENDING,
        )

        try:
            if not recipient.phone:
                result.status = NotificationStatus.FAILED
                result.error_message = "Recipient has no phone number"
                return result

            if not self.check_rate_limit():
                result.status = NotificationStatus.RETRYING
                result.error_message = "Rate limit exceeded"
                return result

            formatted = self.format_message(payload, recipient)

            # In production, use Twilio, AWS SNS, etc.
            logger.info("Sending SMS to %s: %s", recipient.phone, formatted["message"][:50])

            await asyncio.sleep(0.1)

            result.status = NotificationStatus.SENT
            result.sent_at = datetime.utcnow()
            self.record_send()

        except Exception as e:
            result.status = NotificationStatus.FAILED
            result.error_message = str(e)
            logger.error("SMS send failed: %s - %s", notification_id, e)

        return result

    def format_message(
        self, payload: NotificationPayload, recipient: NotificationRecipient
    ) -> Dict[str, Any]:
        """Format SMS message (160 char limit)."""
        # Truncate for SMS limits
        prefix = f"[{payload.severity}] "
        max_len = 160 - len(prefix)
        message = payload.title[:max_len]

        return {
            "to": recipient.phone,
            "message": f"{prefix}{message}",
        }


class WebhookProvider(ChannelProvider):
    """Webhook notification provider."""

    async def send(
        self, payload: NotificationPayload, recipient: NotificationRecipient
    ) -> NotificationResult:
        """Send webhook notification."""
        notification_id = payload.notification_id
        result = NotificationResult(
            notification_id=notification_id,
            channel=NotificationChannel.WEBHOOK,
            recipient_id=recipient.recipient_id,
            status=NotificationStatus.PENDING,
        )

        try:
            webhook_url = recipient.webhook_url or self.config.endpoint
            if not webhook_url:
                result.status = NotificationStatus.FAILED
                result.error_message = "No webhook URL configured"
                return result

            formatted = self.format_message(payload, recipient)

            # In production, use aiohttp or httpx
            logger.info("Sending webhook to %s", webhook_url)

            await asyncio.sleep(0.1)

            result.status = NotificationStatus.SENT
            result.sent_at = datetime.utcnow()
            result.response_data = {"url": webhook_url}
            self.record_send()

        except Exception as e:
            result.status = NotificationStatus.FAILED
            result.error_message = str(e)
            logger.error("Webhook send failed: %s - %s", notification_id, e)

        return result

    def format_message(
        self, payload: NotificationPayload, recipient: NotificationRecipient
    ) -> Dict[str, Any]:
        """Format webhook JSON payload."""
        return {
            "notification_id": payload.notification_id,
            "alert_id": payload.alert_id,
            "alert_code": payload.alert_code,
            "severity": payload.severity,
            "priority": payload.priority.value,
            "title": payload.title,
            "message": payload.message,
            "timestamp": payload.created_at.isoformat(),
            "acknowledgement_required": payload.acknowledgement_required,
            "acknowledgement_url": payload.acknowledgement_url,
            "metadata": payload.metadata,
        }


class TeamsProvider(ChannelProvider):
    """Microsoft Teams notification provider."""

    async def send(
        self, payload: NotificationPayload, recipient: NotificationRecipient
    ) -> NotificationResult:
        """Send Teams notification via Incoming Webhook."""
        notification_id = payload.notification_id
        result = NotificationResult(
            notification_id=notification_id,
            channel=NotificationChannel.TEAMS,
            recipient_id=recipient.recipient_id,
            status=NotificationStatus.PENDING,
        )

        try:
            webhook_url = self.config.endpoint
            if not webhook_url:
                result.status = NotificationStatus.FAILED
                result.error_message = "No Teams webhook URL configured"
                return result

            formatted = self.format_message(payload, recipient)

            logger.info("Sending Teams notification: %s", payload.title)

            await asyncio.sleep(0.1)

            result.status = NotificationStatus.SENT
            result.sent_at = datetime.utcnow()
            self.record_send()

        except Exception as e:
            result.status = NotificationStatus.FAILED
            result.error_message = str(e)
            logger.error("Teams send failed: %s - %s", notification_id, e)

        return result

    def format_message(
        self, payload: NotificationPayload, recipient: NotificationRecipient
    ) -> Dict[str, Any]:
        """Format Teams Adaptive Card message."""
        # Determine theme color based on severity
        theme_colors = {
            "CRITICAL": "FF0000",
            "HIGH": "FF6600",
            "MEDIUM": "FFCC00",
            "LOW": "00CC00",
            "INFO": "0066FF",
        }
        theme_color = theme_colors.get(payload.severity, "0066FF")

        # Build Adaptive Card
        card = {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": theme_color,
            "summary": payload.title,
            "sections": [
                {
                    "activityTitle": payload.title,
                    "activitySubtitle": f"Alert {payload.alert_code} - {payload.severity}",
                    "facts": [
                        {"name": "Alert ID", "value": payload.alert_id},
                        {"name": "Time", "value": payload.created_at.strftime("%Y-%m-%d %H:%M:%S UTC")},
                    ],
                    "text": payload.message,
                }
            ],
        }

        # Add action buttons
        if payload.action_buttons or payload.acknowledgement_url:
            actions = []
            if payload.acknowledgement_url:
                actions.append({
                    "@type": "OpenUri",
                    "name": "Acknowledge",
                    "targets": [{"os": "default", "uri": payload.acknowledgement_url}],
                })
            for button in payload.action_buttons:
                actions.append({
                    "@type": "OpenUri",
                    "name": button.get("label", "Action"),
                    "targets": [{"os": "default", "uri": button.get("url", "#")}],
                })
            card["potentialAction"] = actions

        return card


class SlackProvider(ChannelProvider):
    """Slack notification provider."""

    async def send(
        self, payload: NotificationPayload, recipient: NotificationRecipient
    ) -> NotificationResult:
        """Send Slack notification via Incoming Webhook."""
        notification_id = payload.notification_id
        result = NotificationResult(
            notification_id=notification_id,
            channel=NotificationChannel.SLACK,
            recipient_id=recipient.recipient_id,
            status=NotificationStatus.PENDING,
        )

        try:
            webhook_url = self.config.endpoint
            if not webhook_url:
                result.status = NotificationStatus.FAILED
                result.error_message = "No Slack webhook URL configured"
                return result

            formatted = self.format_message(payload, recipient)

            logger.info("Sending Slack notification: %s", payload.title)

            await asyncio.sleep(0.1)

            result.status = NotificationStatus.SENT
            result.sent_at = datetime.utcnow()
            self.record_send()

        except Exception as e:
            result.status = NotificationStatus.FAILED
            result.error_message = str(e)
            logger.error("Slack send failed: %s - %s", notification_id, e)

        return result

    def format_message(
        self, payload: NotificationPayload, recipient: NotificationRecipient
    ) -> Dict[str, Any]:
        """Format Slack Block Kit message."""
        # Determine emoji based on severity
        severity_emoji = {
            "CRITICAL": ":rotating_light:",
            "HIGH": ":warning:",
            "MEDIUM": ":large_yellow_circle:",
            "LOW": ":large_green_circle:",
            "INFO": ":information_source:",
        }
        emoji = severity_emoji.get(payload.severity, ":bell:")

        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"{emoji} {payload.title}"},
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Alert Code:*\n{payload.alert_code}"},
                    {"type": "mrkdwn", "text": f"*Severity:*\n{payload.severity}"},
                    {"type": "mrkdwn", "text": f"*Alert ID:*\n{payload.alert_id}"},
                    {"type": "mrkdwn", "text": f"*Time:*\n{payload.created_at.strftime('%H:%M:%S UTC')}"},
                ],
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": payload.message},
            },
        ]

        # Add action buttons
        if payload.acknowledgement_url or payload.action_buttons:
            elements = []
            if payload.acknowledgement_url:
                elements.append({
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Acknowledge"},
                    "url": payload.acknowledgement_url,
                    "style": "primary",
                })
            for button in payload.action_buttons:
                elements.append({
                    "type": "button",
                    "text": {"type": "plain_text", "text": button.get("label", "Action")},
                    "url": button.get("url", "#"),
                })
            blocks.append({"type": "actions", "elements": elements})

        return {"blocks": blocks}


# Channel provider registry
CHANNEL_PROVIDERS: Dict[NotificationChannel, Type[ChannelProvider]] = {
    NotificationChannel.EMAIL: EmailProvider,
    NotificationChannel.SMS: SMSProvider,
    NotificationChannel.WEBHOOK: WebhookProvider,
    NotificationChannel.TEAMS: TeamsProvider,
    NotificationChannel.SLACK: SlackProvider,
}


class NotificationService:
    """
    Multi-channel notification service for FurnacePulse alerts.

    This service handles notification delivery across multiple channels,
    priority-based routing, acknowledgement tracking, and escalation timers.

    Attributes:
        config: Service configuration
        providers: Channel-specific notification providers
        pending_acknowledgements: Acknowledgements awaiting response
        notification_history: Recent notification history for deduplication

    Example:
        >>> config = NotificationConfig()
        >>> service = NotificationService(config)
        >>> payload = NotificationPayload(
        ...     alert_id="alert-123",
        ...     alert_code="A-001",
        ...     severity="MEDIUM",
        ...     title="Hotspot Advisory",
        ...     message="TMT-101 approaching threshold"
        ... )
        >>> recipient = NotificationRecipient(
        ...     recipient_id="user-1",
        ...     name="John Operator",
        ...     email="john@example.com"
        ... )
        >>> results = await service.send_notification(payload, [recipient])
    """

    def __init__(self, config: NotificationConfig):
        """
        Initialize NotificationService.

        Args:
            config: Service configuration
        """
        self.config = config
        self.providers: Dict[NotificationChannel, ChannelProvider] = {}
        self.pending_acknowledgements: Dict[str, AcknowledgementRecord] = {}
        self.notification_history: Dict[str, datetime] = {}
        self._escalation_callbacks: List[Callable[[AcknowledgementRecord], None]] = []

        # Initialize providers for enabled channels
        for channel, channel_config in config.channels.items():
            if channel_config.enabled and channel in CHANNEL_PROVIDERS:
                provider_class = CHANNEL_PROVIDERS[channel]
                self.providers[channel] = provider_class(channel_config)
                logger.info("Initialized %s provider", channel.value)

        logger.info(
            "NotificationService initialized with %d channels",
            len(self.providers)
        )

    async def send_notification(
        self,
        payload: NotificationPayload,
        recipients: List[NotificationRecipient],
        channels: Optional[List[NotificationChannel]] = None,
    ) -> List[NotificationResult]:
        """
        Send notification to recipients through specified channels.

        Args:
            payload: Notification payload
            recipients: List of recipients
            channels: Specific channels to use (or None for recipient preferences)

        Returns:
            List of NotificationResult for each delivery attempt
        """
        start_time = datetime.utcnow()
        results: List[NotificationResult] = []

        # Calculate provenance hash
        payload.provenance_hash = self._calculate_provenance_hash(payload)

        # Check for duplicate
        if self.config.enable_deduplication:
            dedup_key = f"{payload.alert_id}:{payload.alert_code}"
            if self._is_duplicate(dedup_key):
                logger.debug("Suppressing duplicate notification: %s", dedup_key)
                return results
            self.notification_history[dedup_key] = datetime.utcnow()

        # Set expiration if not set
        if not payload.expires_at:
            payload.expires_at = start_time + timedelta(
                minutes=self.config.default_expiry_minutes
            )

        # Process each recipient
        for recipient in recipients:
            recipient_channels = channels or self._get_channels_for_recipient(
                recipient, payload.priority
            )

            for channel in recipient_channels:
                if channel not in self.providers:
                    logger.warning("No provider for channel: %s", channel.value)
                    continue

                # Check priority threshold for channel
                channel_config = self.config.channels.get(channel)
                if channel_config:
                    priority_order = list(NotificationPriority)
                    if (priority_order.index(payload.priority) <
                            priority_order.index(channel_config.priority_threshold)):
                        logger.debug(
                            "Skipping %s for low priority notification",
                            channel.value
                        )
                        continue

                # Send notification
                try:
                    result = await self.providers[channel].send(payload, recipient)
                    results.append(result)

                    # Create acknowledgement record if required
                    if payload.acknowledgement_required and result.status == NotificationStatus.SENT:
                        self._create_acknowledgement_record(payload, recipient)

                except Exception as e:
                    logger.error(
                        "Error sending %s notification: %s",
                        channel.value, str(e)
                    )
                    results.append(NotificationResult(
                        notification_id=payload.notification_id,
                        channel=channel,
                        recipient_id=recipient.recipient_id,
                        status=NotificationStatus.FAILED,
                        error_message=str(e),
                    ))

        processing_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        success_count = sum(1 for r in results if r.status == NotificationStatus.SENT)
        logger.info(
            "Notification %s sent to %d/%d recipients in %.2fms",
            payload.notification_id, success_count, len(results), processing_time_ms
        )

        return results

    def _get_channels_for_recipient(
        self,
        recipient: NotificationRecipient,
        priority: NotificationPriority,
    ) -> List[NotificationChannel]:
        """Determine channels to use for a recipient based on priority and preferences."""
        channels = []

        # For emergency, use all available channels
        if priority == NotificationPriority.EMERGENCY:
            for channel in recipient.preferred_channels:
                if channel in self.providers:
                    channels.append(channel)
            # Also add SMS and pager if available
            for channel in (NotificationChannel.SMS, NotificationChannel.PAGER):
                if channel in self.providers and channel not in channels:
                    channels.append(channel)
            return channels

        # For urgent, use preferred channels plus SMS
        if priority == NotificationPriority.URGENT:
            channels = [c for c in recipient.preferred_channels if c in self.providers]
            if NotificationChannel.SMS in self.providers and NotificationChannel.SMS not in channels:
                channels.append(NotificationChannel.SMS)
            return channels

        # For normal/low, use preferred channels only
        return [c for c in recipient.preferred_channels if c in self.providers]

    def _is_duplicate(self, dedup_key: str) -> bool:
        """Check if notification is a duplicate within the deduplication window."""
        if dedup_key not in self.notification_history:
            return False

        last_sent = self.notification_history[dedup_key]
        window = timedelta(seconds=self.config.deduplication_window_seconds)
        return datetime.utcnow() - last_sent < window

    def _calculate_provenance_hash(self, payload: NotificationPayload) -> str:
        """Calculate SHA-256 hash for audit trail."""
        provenance_str = (
            f"{payload.notification_id}|{payload.alert_id}|{payload.alert_code}|"
            f"{payload.title}|{payload.created_at.isoformat()}"
        )
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    def _create_acknowledgement_record(
        self,
        payload: NotificationPayload,
        recipient: NotificationRecipient,
    ) -> AcknowledgementRecord:
        """Create an acknowledgement tracking record."""
        deadline = datetime.utcnow() + timedelta(
            minutes=self.config.acknowledgement_timeout_minutes
        )

        record = AcknowledgementRecord(
            notification_id=payload.notification_id,
            alert_id=payload.alert_id,
            recipient_id=recipient.recipient_id,
            required_by=deadline,
        )

        self.pending_acknowledgements[record.acknowledgement_id] = record
        logger.debug(
            "Created acknowledgement record %s for %s",
            record.acknowledgement_id, payload.notification_id
        )

        return record

    def acknowledge(
        self,
        acknowledgement_id: str,
        recipient_id: str,
        notes: Optional[str] = None,
    ) -> bool:
        """
        Record an acknowledgement.

        Args:
            acknowledgement_id: Acknowledgement record ID
            recipient_id: ID of recipient acknowledging
            notes: Optional notes

        Returns:
            True if acknowledgement recorded successfully
        """
        if acknowledgement_id not in self.pending_acknowledgements:
            logger.warning("Unknown acknowledgement ID: %s", acknowledgement_id)
            return False

        record = self.pending_acknowledgements[acknowledgement_id]

        if record.recipient_id != recipient_id:
            logger.warning(
                "Acknowledgement recipient mismatch: expected %s, got %s",
                record.recipient_id, recipient_id
            )
            return False

        if record.status != AcknowledgementStatus.PENDING:
            logger.warning(
                "Acknowledgement already processed: %s",
                acknowledgement_id
            )
            return False

        record.status = AcknowledgementStatus.ACKNOWLEDGED
        record.acknowledged_at = datetime.utcnow()
        record.notes = notes

        logger.info(
            "Acknowledgement recorded: %s by %s",
            acknowledgement_id, recipient_id
        )

        return True

    def process_escalation_timers(self) -> List[AcknowledgementRecord]:
        """
        Process acknowledgement timeouts and trigger escalations.

        Returns:
            List of records that were escalated
        """
        now = datetime.utcnow()
        escalated = []

        for ack_id, record in list(self.pending_acknowledgements.items()):
            if record.status != AcknowledgementStatus.PENDING:
                continue

            if now >= record.required_by:
                # Timeout - trigger escalation
                record.status = AcknowledgementStatus.ESCALATED
                record.escalated_at = now

                logger.warning(
                    "Acknowledgement timeout - escalating: %s",
                    ack_id
                )

                # Trigger escalation callbacks
                for callback in self._escalation_callbacks:
                    try:
                        callback(record)
                    except Exception as e:
                        logger.error("Escalation callback failed: %s", e)

                escalated.append(record)

        return escalated

    def register_escalation_callback(
        self,
        callback: Callable[[AcknowledgementRecord], None],
    ) -> None:
        """Register a callback for escalation events."""
        self._escalation_callbacks.append(callback)

    def get_pending_acknowledgements(
        self,
        recipient_id: Optional[str] = None,
    ) -> List[AcknowledgementRecord]:
        """
        Get pending acknowledgements.

        Args:
            recipient_id: Optional filter by recipient

        Returns:
            List of pending acknowledgement records
        """
        records = [
            r for r in self.pending_acknowledgements.values()
            if r.status == AcknowledgementStatus.PENDING
        ]

        if recipient_id:
            records = [r for r in records if r.recipient_id == recipient_id]

        return sorted(records, key=lambda r: r.required_by)

    def get_notification_statistics(self) -> Dict[str, Any]:
        """
        Get notification statistics for monitoring.

        Returns:
            Dictionary with notification statistics
        """
        now = datetime.utcnow()

        # Clean up old history entries
        cutoff = now - timedelta(hours=24)
        self.notification_history = {
            k: v for k, v in self.notification_history.items() if v > cutoff
        }

        pending_acks = [
            r for r in self.pending_acknowledgements.values()
            if r.status == AcknowledgementStatus.PENDING
        ]

        return {
            "active_channels": list(self.providers.keys()),
            "notifications_last_24h": len(self.notification_history),
            "pending_acknowledgements": len(pending_acks),
            "overdue_acknowledgements": sum(
                1 for r in pending_acks if now >= r.required_by
            ),
            "channel_stats": {
                channel.value: {
                    "enabled": channel in self.providers,
                    "rate_limit_remaining": (
                        self.providers[channel].config.rate_limit_per_minute
                        - len(self.providers[channel].last_send_times)
                        if channel in self.providers else 0
                    ),
                }
                for channel in NotificationChannel
            },
        }

    def cleanup_old_records(self) -> int:
        """
        Clean up old acknowledgement records.

        Returns:
            Number of records cleaned up
        """
        cutoff = datetime.utcnow() - timedelta(hours=24)

        old_ids = [
            ack_id for ack_id, record in self.pending_acknowledgements.items()
            if record.status in (AcknowledgementStatus.ACKNOWLEDGED, AcknowledgementStatus.EXPIRED)
            and (record.acknowledged_at or record.escalated_at or datetime.utcnow()) < cutoff
        ]

        for ack_id in old_ids:
            del self.pending_acknowledgements[ack_id]

        if old_ids:
            logger.info("Cleaned up %d old acknowledgement records", len(old_ids))

        return len(old_ids)
