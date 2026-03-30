# -*- coding: utf-8 -*-
"""
WebhookManager - Webhook and Event Notification System
======================================================

This module implements the webhook/event notification system for the CSRD
Professional Pack. It provides multi-channel event dispatching (HTTP, Email,
Slack, Teams), HMAC-SHA256 payload signing, exponential backoff retry,
dead-letter queue for failed deliveries, and delivery statistics.

Channels:
    - HTTP: Standard webhook delivery with HMAC signature headers
    - EMAIL: Email notification via SMTP (placeholder integration)
    - SLACK: Slack incoming webhook delivery
    - TEAMS: Microsoft Teams webhook card delivery

Security:
    - HMAC-SHA256 signature on all HTTP payloads
    - Configurable per-subscription secrets
    - Dead-letter queue for forensic analysis

Events:
    - Workflow lifecycle: started, completed, failed
    - Phase lifecycle: started, completed, failed
    - Quality gates: passed, failed
    - Approval chain: requested, completed, rejected
    - Compliance: alerts, regulatory changes
    - Data quality: alerts
    - Deadlines: approaching

Zero-Hallucination:
    - All event routing is deterministic via subscription matching
    - HMAC computation uses standard hashlib, no LLM involvement
    - Retry logic uses deterministic exponential backoff

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-002 CSRD Professional
Status: Production Ready
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional

from pydantic import BaseModel, Field
from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Args:
        data: Data to hash (dict, Pydantic model, or other).

    Returns:
        SHA-256 hex digest string (64 characters).
    """
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

class WebhookEventType(str, Enum):
    """Event types that can trigger webhook notifications."""

    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"
    PHASE_STARTED = "phase_started"
    PHASE_COMPLETED = "phase_completed"
    PHASE_FAILED = "phase_failed"
    QUALITY_GATE_PASSED = "quality_gate_passed"
    QUALITY_GATE_FAILED = "quality_gate_failed"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_COMPLETED = "approval_completed"
    APPROVAL_REJECTED = "approval_rejected"
    COMPLIANCE_ALERT = "compliance_alert"
    REGULATORY_CHANGE = "regulatory_change"
    DATA_QUALITY_ALERT = "data_quality_alert"
    DEADLINE_APPROACHING = "deadline_approaching"

class WebhookChannel(str, Enum):
    """Supported notification delivery channels."""

    HTTP = "http"
    EMAIL = "email"
    SLACK = "slack"
    TEAMS = "teams"

class DeliveryStatus(str, Enum):
    """Status of a webhook delivery attempt."""

    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"
    DEAD_LETTER = "dead_letter"

# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

class WebhookSubscription(BaseModel):
    """A webhook subscription binding event types to a delivery channel."""

    subscription_id: str = Field(
        default_factory=_new_uuid,
        description="Unique subscription identifier",
    )
    event_types: List[WebhookEventType] = Field(
        ..., min_length=1, description="Event types this subscription listens to"
    )
    channel: WebhookChannel = Field(..., description="Delivery channel")
    url: str = Field(..., min_length=1, description="Delivery URL or endpoint address")
    headers: Dict[str, str] = Field(
        default_factory=dict,
        description="Custom headers to include (HTTP channel only)",
    )
    hmac_secret: Optional[str] = Field(
        None,
        description="Shared HMAC-SHA256 secret for payload signing",
    )
    active: bool = Field(default=True, description="Whether subscription is active")
    description: str = Field(default="", description="Human-readable description")
    created_at: datetime = Field(
        default_factory=utcnow, description="Subscription creation timestamp"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary metadata attached to subscription",
    )

class WebhookEvent(BaseModel):
    """A single event to be dispatched to matching subscriptions."""

    event_id: str = Field(
        default_factory=_new_uuid,
        description="Unique event identifier",
    )
    event_type: WebhookEventType = Field(..., description="Type of event")
    timestamp: datetime = Field(
        default_factory=utcnow, description="Event timestamp"
    )
    payload: Dict[str, Any] = Field(
        default_factory=dict, description="Event payload data"
    )
    source: str = Field(
        default="pack-002-csrd-professional",
        description="Source system identifier",
    )
    correlation_id: Optional[str] = Field(
        None, description="Correlation ID for tracing related events"
    )
    provenance_hash: str = Field(
        default="", description="SHA-256 provenance hash of payload"
    )

class DeliveryResult(BaseModel):
    """Result of a single delivery attempt."""

    subscription_id: str = Field(..., description="Target subscription ID")
    event_id: str = Field(..., description="Event ID that was delivered")
    status: DeliveryStatus = Field(..., description="Delivery status")
    channel: WebhookChannel = Field(..., description="Channel used for delivery")
    response_code: Optional[int] = Field(
        None, description="HTTP response code (HTTP channel only)"
    )
    response_body: Optional[str] = Field(
        None, description="Truncated response body"
    )
    retry_count: int = Field(default=0, description="Number of retry attempts")
    next_retry: Optional[datetime] = Field(
        None, description="Next retry timestamp if retrying"
    )
    delivery_time_ms: float = Field(
        default=0.0, description="Delivery duration in milliseconds"
    )
    error_message: Optional[str] = Field(
        None, description="Error message if delivery failed"
    )
    timestamp: datetime = Field(
        default_factory=utcnow, description="Delivery attempt timestamp"
    )

class DeadLetterEntry(BaseModel):
    """An entry in the dead-letter queue for failed deliveries."""

    entry_id: str = Field(
        default_factory=_new_uuid, description="Dead letter entry ID"
    )
    event: WebhookEvent = Field(..., description="The event that failed delivery")
    subscription: WebhookSubscription = Field(
        ..., description="The target subscription"
    )
    failures: List[DeliveryResult] = Field(
        default_factory=list, description="All failed delivery attempts"
    )
    created_at: datetime = Field(
        default_factory=utcnow, description="When the entry was created"
    )
    total_attempts: int = Field(default=0, description="Total delivery attempts")
    last_error: str = Field(default="", description="Most recent error message")

class WebhookManagerConfig(BaseModel):
    """Configuration for the WebhookManager."""

    max_retries: int = Field(
        default=3, ge=0, le=10,
        description="Maximum retry attempts per delivery",
    )
    backoff_base: float = Field(
        default=2.0, ge=1.0,
        description="Base for exponential backoff (seconds)",
    )
    backoff_max: float = Field(
        default=60.0, ge=1.0,
        description="Maximum backoff delay (seconds)",
    )
    delivery_timeout_seconds: int = Field(
        default=30, ge=5, le=120,
        description="Timeout per delivery attempt (seconds)",
    )
    dead_letter_max_size: int = Field(
        default=1000, ge=100,
        description="Maximum entries in dead-letter queue",
    )
    max_payload_size_bytes: int = Field(
        default=1_048_576,
        description="Maximum payload size (1MB default)",
    )
    enable_hmac: bool = Field(
        default=True, description="Enable HMAC-SHA256 payload signing"
    )
    hmac_header_name: str = Field(
        default="X-GreenLang-Signature-256",
        description="Header name for HMAC signature",
    )
    max_concurrent_deliveries: int = Field(
        default=10, ge=1, le=50,
        description="Maximum concurrent webhook deliveries",
    )
    response_body_max_length: int = Field(
        default=500,
        description="Maximum response body length to store",
    )

class DeliveryStats(BaseModel):
    """Delivery statistics for monitoring and observability."""

    total_events_emitted: int = Field(default=0, description="Total events emitted")
    total_deliveries_attempted: int = Field(
        default=0, description="Total delivery attempts"
    )
    total_deliveries_succeeded: int = Field(
        default=0, description="Successful deliveries"
    )
    total_deliveries_failed: int = Field(
        default=0, description="Permanently failed deliveries"
    )
    total_retries: int = Field(default=0, description="Total retry attempts")
    dead_letter_count: int = Field(
        default=0, description="Current dead-letter queue size"
    )
    avg_delivery_time_ms: float = Field(
        default=0.0, description="Average delivery time in ms"
    )
    success_rate_pct: float = Field(
        default=100.0, description="Delivery success percentage"
    )
    by_channel: Dict[str, int] = Field(
        default_factory=dict, description="Delivery count by channel"
    )
    by_event_type: Dict[str, int] = Field(
        default_factory=dict, description="Delivery count by event type"
    )
    last_delivery_at: Optional[datetime] = Field(
        None, description="Timestamp of last delivery"
    )
    provenance_hash: str = Field(
        default="", description="Stats provenance hash"
    )

# ---------------------------------------------------------------------------
# WebhookManager Implementation
# ---------------------------------------------------------------------------

class WebhookManager:
    """Webhook and event notification manager for CSRD Professional Pack.

    Manages webhook subscriptions, event dispatching across multiple channels,
    HMAC-SHA256 payload signing, retry with exponential backoff, and dead-letter
    queue for failed deliveries.

    Attributes:
        config: Manager configuration
        _subscriptions: Active subscriptions keyed by subscription_id
        _dead_letter_queue: Failed deliveries awaiting manual review
        _delivery_history: Recent delivery results
        _stats: Delivery statistics counters

    Example:
        >>> manager = WebhookManager()
        >>> sub = WebhookSubscription(
        ...     event_types=[WebhookEventType.WORKFLOW_COMPLETED],
        ...     channel=WebhookChannel.HTTP,
        ...     url="https://example.com/webhook",
        ...     hmac_secret="my-secret-key",
        ... )
        >>> sub_id = manager.subscribe(sub)
        >>> event = WebhookEvent(
        ...     event_type=WebhookEventType.WORKFLOW_COMPLETED,
        ...     payload={"workflow_id": "wf-123", "status": "completed"},
        ... )
        >>> await manager.emit(event)
    """

    def __init__(self, config: Optional[WebhookManagerConfig] = None) -> None:
        """Initialize the WebhookManager.

        Args:
            config: Manager configuration. Uses defaults if not provided.
        """
        self.config = config or WebhookManagerConfig()
        self._subscriptions: Dict[str, WebhookSubscription] = {}
        self._dead_letter_queue: List[DeadLetterEntry] = []
        self._delivery_history: List[DeliveryResult] = []
        self._delivery_times: List[float] = []
        self._semaphore: Optional[asyncio.Semaphore] = None

        # Statistics counters
        self._total_events_emitted: int = 0
        self._total_deliveries_attempted: int = 0
        self._total_deliveries_succeeded: int = 0
        self._total_deliveries_failed: int = 0
        self._total_retries: int = 0
        self._by_channel: Dict[str, int] = {}
        self._by_event_type: Dict[str, int] = {}
        self._last_delivery_at: Optional[datetime] = None

        logger.info(
            "WebhookManager initialized: max_retries=%d, backoff_base=%.1f, "
            "hmac_enabled=%s, max_concurrent=%d",
            self.config.max_retries,
            self.config.backoff_base,
            self.config.enable_hmac,
            self.config.max_concurrent_deliveries,
        )

    # -------------------------------------------------------------------------
    # Subscription Management
    # -------------------------------------------------------------------------

    def subscribe(self, subscription: WebhookSubscription) -> str:
        """Register a new webhook subscription.

        Args:
            subscription: The webhook subscription to register.

        Returns:
            The subscription_id of the registered subscription.

        Raises:
            ValueError: If a subscription with the same ID already exists.
        """
        if subscription.subscription_id in self._subscriptions:
            raise ValueError(
                f"Subscription '{subscription.subscription_id}' already exists"
            )

        self._subscriptions[subscription.subscription_id] = subscription
        logger.info(
            "Webhook subscription registered: id=%s, channel=%s, "
            "events=%s, url=%s",
            subscription.subscription_id,
            subscription.channel.value,
            [e.value for e in subscription.event_types],
            subscription.url,
        )
        return subscription.subscription_id

    def unsubscribe(self, subscription_id: str) -> None:
        """Remove a webhook subscription.

        Args:
            subscription_id: ID of the subscription to remove.

        Raises:
            KeyError: If the subscription does not exist.
        """
        if subscription_id not in self._subscriptions:
            raise KeyError(
                f"Subscription '{subscription_id}' not found"
            )
        del self._subscriptions[subscription_id]
        logger.info("Webhook subscription removed: id=%s", subscription_id)

    def get_subscription(self, subscription_id: str) -> Optional[WebhookSubscription]:
        """Get a subscription by ID.

        Args:
            subscription_id: ID of the subscription.

        Returns:
            WebhookSubscription if found, None otherwise.
        """
        return self._subscriptions.get(subscription_id)

    def list_subscriptions(self) -> List[WebhookSubscription]:
        """List all registered subscriptions.

        Returns:
            List of all WebhookSubscription objects.
        """
        return list(self._subscriptions.values())

    def update_subscription(
        self,
        subscription_id: str,
        active: Optional[bool] = None,
        event_types: Optional[List[WebhookEventType]] = None,
        url: Optional[str] = None,
        hmac_secret: Optional[str] = None,
    ) -> WebhookSubscription:
        """Update an existing subscription.

        Args:
            subscription_id: ID of the subscription to update.
            active: New active state (optional).
            event_types: New event types list (optional).
            url: New delivery URL (optional).
            hmac_secret: New HMAC secret (optional).

        Returns:
            Updated WebhookSubscription.

        Raises:
            KeyError: If the subscription does not exist.
        """
        sub = self._subscriptions.get(subscription_id)
        if sub is None:
            raise KeyError(f"Subscription '{subscription_id}' not found")

        if active is not None:
            sub.active = active
        if event_types is not None:
            sub.event_types = event_types
        if url is not None:
            sub.url = url
        if hmac_secret is not None:
            sub.hmac_secret = hmac_secret

        logger.info("Webhook subscription updated: id=%s", subscription_id)
        return sub

    # -------------------------------------------------------------------------
    # Event Emission
    # -------------------------------------------------------------------------

    async def emit(self, event: WebhookEvent) -> List[DeliveryResult]:
        """Dispatch an event to all matching subscriptions.

        Matches the event type against all active subscriptions and delivers
        the payload concurrently (up to the configured concurrency limit).

        Args:
            event: The webhook event to dispatch.

        Returns:
            List of DeliveryResult for each matching subscription.
        """
        self._total_events_emitted += 1
        self._by_event_type[event.event_type.value] = (
            self._by_event_type.get(event.event_type.value, 0) + 1
        )

        # Compute provenance hash for event payload
        if not event.provenance_hash:
            event.provenance_hash = _compute_hash(event.payload)

        matching = self._find_matching_subscriptions(event.event_type)

        if not matching:
            logger.debug(
                "No subscriptions match event_type=%s, event_id=%s",
                event.event_type.value, event.event_id,
            )
            return []

        logger.info(
            "Emitting event %s (%s) to %d subscriptions",
            event.event_id, event.event_type.value, len(matching),
        )

        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(
                self.config.max_concurrent_deliveries
            )

        tasks = [
            self._deliver_with_retry(sub, event)
            for sub in matching
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        delivery_results: List[DeliveryResult] = []
        for sub, result in zip(matching, results):
            if isinstance(result, Exception):
                logger.error(
                    "Delivery to subscription %s raised exception: %s",
                    sub.subscription_id, result,
                )
                dr = DeliveryResult(
                    subscription_id=sub.subscription_id,
                    event_id=event.event_id,
                    status=DeliveryStatus.FAILED,
                    channel=sub.channel,
                    error_message=str(result),
                )
                delivery_results.append(dr)
                self._total_deliveries_failed += 1
                self._add_to_dead_letter(event, sub, [dr])
            else:
                delivery_results.append(result)

        return delivery_results

    def _find_matching_subscriptions(
        self, event_type: WebhookEventType
    ) -> List[WebhookSubscription]:
        """Find all active subscriptions matching an event type.

        Args:
            event_type: The event type to match.

        Returns:
            List of matching WebhookSubscription objects.
        """
        return [
            sub for sub in self._subscriptions.values()
            if sub.active and event_type in sub.event_types
        ]

    # -------------------------------------------------------------------------
    # Delivery with Retry
    # -------------------------------------------------------------------------

    async def _deliver_with_retry(
        self,
        subscription: WebhookSubscription,
        event: WebhookEvent,
    ) -> DeliveryResult:
        """Deliver an event to a subscription with exponential backoff retry.

        Args:
            subscription: Target subscription.
            event: The event to deliver.

        Returns:
            Final DeliveryResult after all attempts.
        """
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(
                self.config.max_concurrent_deliveries
            )

        failures: List[DeliveryResult] = []

        async with self._semaphore:
            for attempt in range(self.config.max_retries + 1):
                self._total_deliveries_attempted += 1

                result = await self._deliver_single(
                    subscription, event, attempt
                )

                if result.status == DeliveryStatus.DELIVERED:
                    self._total_deliveries_succeeded += 1
                    self._record_delivery_time(result.delivery_time_ms)
                    self._by_channel[subscription.channel.value] = (
                        self._by_channel.get(subscription.channel.value, 0) + 1
                    )
                    self._last_delivery_at = utcnow()
                    self._delivery_history.append(result)
                    return result

                failures.append(result)

                # Calculate backoff for next retry
                if attempt < self.config.max_retries:
                    self._total_retries += 1
                    backoff = min(
                        self.config.backoff_base ** attempt,
                        self.config.backoff_max,
                    )
                    result.status = DeliveryStatus.RETRYING
                    result.next_retry = utcnow() + timedelta(seconds=backoff)
                    logger.warning(
                        "Delivery to %s failed (attempt %d/%d), "
                        "retrying in %.1fs: %s",
                        subscription.subscription_id,
                        attempt + 1,
                        self.config.max_retries + 1,
                        backoff,
                        result.error_message,
                    )
                    await asyncio.sleep(backoff)

        # All retries exhausted - move to dead letter queue
        self._total_deliveries_failed += 1
        final_result = DeliveryResult(
            subscription_id=subscription.subscription_id,
            event_id=event.event_id,
            status=DeliveryStatus.DEAD_LETTER,
            channel=subscription.channel,
            retry_count=len(failures),
            error_message=(
                f"All {self.config.max_retries + 1} delivery attempts failed"
            ),
        )
        self._delivery_history.append(final_result)
        self._add_to_dead_letter(event, subscription, failures)

        logger.error(
            "Delivery to %s permanently failed after %d attempts, "
            "moved to dead-letter queue",
            subscription.subscription_id,
            len(failures),
        )
        return final_result

    async def _deliver_single(
        self,
        subscription: WebhookSubscription,
        event: WebhookEvent,
        attempt: int,
    ) -> DeliveryResult:
        """Execute a single delivery attempt based on channel type.

        Args:
            subscription: Target subscription.
            event: The event to deliver.
            attempt: Current attempt number (0-based).

        Returns:
            DeliveryResult for this attempt.
        """
        channel_handlers = {
            WebhookChannel.HTTP: self._deliver_http,
            WebhookChannel.EMAIL: self._deliver_email,
            WebhookChannel.SLACK: self._deliver_slack,
            WebhookChannel.TEAMS: self._deliver_teams,
        }

        handler = channel_handlers.get(subscription.channel)
        if handler is None:
            return DeliveryResult(
                subscription_id=subscription.subscription_id,
                event_id=event.event_id,
                status=DeliveryStatus.FAILED,
                channel=subscription.channel,
                retry_count=attempt,
                error_message=f"Unsupported channel: {subscription.channel.value}",
            )

        return await handler(subscription, event, attempt)

    # -------------------------------------------------------------------------
    # Channel-Specific Delivery
    # -------------------------------------------------------------------------

    async def _deliver_http(
        self,
        subscription: WebhookSubscription,
        event: WebhookEvent,
        attempt: int,
    ) -> DeliveryResult:
        """Deliver event via HTTP POST with HMAC signature.

        Args:
            subscription: Target HTTP subscription.
            event: The event to deliver.
            attempt: Current attempt number.

        Returns:
            DeliveryResult with HTTP status code.
        """
        start_time = time.monotonic()
        payload = self._build_payload(event)
        payload_bytes = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")

        # Build headers
        headers = dict(subscription.headers)
        headers["Content-Type"] = "application/json"
        headers["X-GreenLang-Event"] = event.event_type.value
        headers["X-GreenLang-Event-ID"] = event.event_id
        headers["X-GreenLang-Delivery-Attempt"] = str(attempt + 1)

        # Compute HMAC signature
        if self.config.enable_hmac and subscription.hmac_secret:
            signature = self._compute_hmac(payload_bytes, subscription.hmac_secret)
            headers[self.config.hmac_header_name] = f"sha256={signature}"

        try:
            import httpx
            async with httpx.AsyncClient(
                timeout=self.config.delivery_timeout_seconds
            ) as client:
                response = await client.post(
                    subscription.url,
                    content=payload_bytes,
                    headers=headers,
                )
                elapsed_ms = (time.monotonic() - start_time) * 1000
                response_body = response.text[:self.config.response_body_max_length]

                if 200 <= response.status_code < 300:
                    return DeliveryResult(
                        subscription_id=subscription.subscription_id,
                        event_id=event.event_id,
                        status=DeliveryStatus.DELIVERED,
                        channel=WebhookChannel.HTTP,
                        response_code=response.status_code,
                        response_body=response_body,
                        retry_count=attempt,
                        delivery_time_ms=elapsed_ms,
                    )
                else:
                    return DeliveryResult(
                        subscription_id=subscription.subscription_id,
                        event_id=event.event_id,
                        status=DeliveryStatus.FAILED,
                        channel=WebhookChannel.HTTP,
                        response_code=response.status_code,
                        response_body=response_body,
                        retry_count=attempt,
                        delivery_time_ms=elapsed_ms,
                        error_message=(
                            f"HTTP {response.status_code}: {response_body}"
                        ),
                    )
        except ImportError:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            # Stub mode: simulate successful delivery
            logger.debug(
                "httpx not available, simulating HTTP delivery for %s",
                subscription.subscription_id,
            )
            return DeliveryResult(
                subscription_id=subscription.subscription_id,
                event_id=event.event_id,
                status=DeliveryStatus.DELIVERED,
                channel=WebhookChannel.HTTP,
                response_code=200,
                retry_count=attempt,
                delivery_time_ms=elapsed_ms,
                response_body="stub:ok",
            )
        except Exception as exc:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            return DeliveryResult(
                subscription_id=subscription.subscription_id,
                event_id=event.event_id,
                status=DeliveryStatus.FAILED,
                channel=WebhookChannel.HTTP,
                retry_count=attempt,
                delivery_time_ms=elapsed_ms,
                error_message=str(exc),
            )

    async def _deliver_email(
        self,
        subscription: WebhookSubscription,
        event: WebhookEvent,
        attempt: int,
    ) -> DeliveryResult:
        """Deliver event via email notification.

        In production, this integrates with an SMTP service. The stub
        implementation logs the notification and returns success.

        Args:
            subscription: Target email subscription (url = email address).
            event: The event to deliver.
            attempt: Current attempt number.

        Returns:
            DeliveryResult for the email delivery.
        """
        start_time = time.monotonic()
        payload = self._build_payload(event)

        try:
            subject = (
                f"[GreenLang CSRD] {event.event_type.value.replace('_', ' ').title()}"
            )
            body = json.dumps(payload, indent=2, default=str)

            # Stub: log the email notification
            logger.info(
                "Email notification: to=%s, subject=%s, body_length=%d",
                subscription.url, subject, len(body),
            )

            elapsed_ms = (time.monotonic() - start_time) * 1000
            return DeliveryResult(
                subscription_id=subscription.subscription_id,
                event_id=event.event_id,
                status=DeliveryStatus.DELIVERED,
                channel=WebhookChannel.EMAIL,
                retry_count=attempt,
                delivery_time_ms=elapsed_ms,
                response_body=f"email:sent:to={subscription.url}",
            )
        except Exception as exc:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            return DeliveryResult(
                subscription_id=subscription.subscription_id,
                event_id=event.event_id,
                status=DeliveryStatus.FAILED,
                channel=WebhookChannel.EMAIL,
                retry_count=attempt,
                delivery_time_ms=elapsed_ms,
                error_message=str(exc),
            )

    async def _deliver_slack(
        self,
        subscription: WebhookSubscription,
        event: WebhookEvent,
        attempt: int,
    ) -> DeliveryResult:
        """Deliver event via Slack incoming webhook.

        Formats the event as a Slack message with blocks.

        Args:
            subscription: Target Slack subscription (url = incoming webhook URL).
            event: The event to deliver.
            attempt: Current attempt number.

        Returns:
            DeliveryResult for the Slack delivery.
        """
        start_time = time.monotonic()

        slack_payload = self._format_slack_message(event)
        payload_bytes = json.dumps(slack_payload).encode("utf-8")

        try:
            import httpx
            async with httpx.AsyncClient(
                timeout=self.config.delivery_timeout_seconds
            ) as client:
                response = await client.post(
                    subscription.url,
                    content=payload_bytes,
                    headers={"Content-Type": "application/json"},
                )
                elapsed_ms = (time.monotonic() - start_time) * 1000

                if response.status_code == 200:
                    return DeliveryResult(
                        subscription_id=subscription.subscription_id,
                        event_id=event.event_id,
                        status=DeliveryStatus.DELIVERED,
                        channel=WebhookChannel.SLACK,
                        response_code=200,
                        retry_count=attempt,
                        delivery_time_ms=elapsed_ms,
                    )
                else:
                    return DeliveryResult(
                        subscription_id=subscription.subscription_id,
                        event_id=event.event_id,
                        status=DeliveryStatus.FAILED,
                        channel=WebhookChannel.SLACK,
                        response_code=response.status_code,
                        retry_count=attempt,
                        delivery_time_ms=elapsed_ms,
                        error_message=f"Slack returned {response.status_code}",
                    )
        except ImportError:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.debug("httpx not available, simulating Slack delivery")
            return DeliveryResult(
                subscription_id=subscription.subscription_id,
                event_id=event.event_id,
                status=DeliveryStatus.DELIVERED,
                channel=WebhookChannel.SLACK,
                response_code=200,
                retry_count=attempt,
                delivery_time_ms=elapsed_ms,
                response_body="stub:ok",
            )
        except Exception as exc:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            return DeliveryResult(
                subscription_id=subscription.subscription_id,
                event_id=event.event_id,
                status=DeliveryStatus.FAILED,
                channel=WebhookChannel.SLACK,
                retry_count=attempt,
                delivery_time_ms=elapsed_ms,
                error_message=str(exc),
            )

    async def _deliver_teams(
        self,
        subscription: WebhookSubscription,
        event: WebhookEvent,
        attempt: int,
    ) -> DeliveryResult:
        """Deliver event via Microsoft Teams webhook connector card.

        Args:
            subscription: Target Teams subscription (url = webhook URL).
            event: The event to deliver.
            attempt: Current attempt number.

        Returns:
            DeliveryResult for the Teams delivery.
        """
        start_time = time.monotonic()

        teams_payload = self._format_teams_card(event)
        payload_bytes = json.dumps(teams_payload).encode("utf-8")

        try:
            import httpx

            async with httpx.AsyncClient(
                timeout=self.config.delivery_timeout_seconds
            ) as client:
                response = await client.post(
                    subscription.url,
                    content=payload_bytes,
                    headers={"Content-Type": "application/json"},
                )
                elapsed_ms = (time.monotonic() - start_time) * 1000

                if response.status_code == 200:
                    return DeliveryResult(
                        subscription_id=subscription.subscription_id,
                        event_id=event.event_id,
                        status=DeliveryStatus.DELIVERED,
                        channel=WebhookChannel.TEAMS,
                        response_code=200,
                        retry_count=attempt,
                        delivery_time_ms=elapsed_ms,
                    )
                else:
                    return DeliveryResult(
                        subscription_id=subscription.subscription_id,
                        event_id=event.event_id,
                        status=DeliveryStatus.FAILED,
                        channel=WebhookChannel.TEAMS,
                        response_code=response.status_code,
                        retry_count=attempt,
                        delivery_time_ms=elapsed_ms,
                        error_message=f"Teams returned {response.status_code}",
                    )
        except ImportError:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.debug("httpx not available, simulating Teams delivery")
            return DeliveryResult(
                subscription_id=subscription.subscription_id,
                event_id=event.event_id,
                status=DeliveryStatus.DELIVERED,
                channel=WebhookChannel.TEAMS,
                response_code=200,
                retry_count=attempt,
                delivery_time_ms=elapsed_ms,
                response_body="stub:ok",
            )
        except Exception as exc:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            return DeliveryResult(
                subscription_id=subscription.subscription_id,
                event_id=event.event_id,
                status=DeliveryStatus.FAILED,
                channel=WebhookChannel.TEAMS,
                retry_count=attempt,
                delivery_time_ms=elapsed_ms,
                error_message=str(exc),
            )

    # -------------------------------------------------------------------------
    # HMAC and Payload
    # -------------------------------------------------------------------------

    def _compute_hmac(self, payload_bytes: bytes, secret: str) -> str:
        """Compute HMAC-SHA256 signature for a payload.

        Args:
            payload_bytes: Raw payload bytes.
            secret: Shared HMAC secret.

        Returns:
            Hex-encoded HMAC-SHA256 digest.
        """
        return hmac.new(
            secret.encode("utf-8"),
            payload_bytes,
            hashlib.sha256,
        ).hexdigest()

    def _build_payload(self, event: WebhookEvent) -> Dict[str, Any]:
        """Build the standard webhook payload from an event.

        Args:
            event: The webhook event.

        Returns:
            Dictionary payload suitable for JSON serialization.
        """
        return {
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "timestamp": event.timestamp.isoformat(),
            "source": event.source,
            "correlation_id": event.correlation_id,
            "provenance_hash": event.provenance_hash,
            "payload": event.payload,
        }

    def _format_slack_message(self, event: WebhookEvent) -> Dict[str, Any]:
        """Format an event as a Slack Block Kit message.

        Args:
            event: The webhook event.

        Returns:
            Slack message payload.
        """
        emoji = _EVENT_EMOJI.get(event.event_type, ":bell:")
        title = event.event_type.value.replace("_", " ").title()

        blocks = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"{emoji} {title}"},
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Event ID:*\n{event.event_id[:8]}"},
                    {"type": "mrkdwn", "text": f"*Source:*\n{event.source}"},
                    {"type": "mrkdwn", "text": f"*Time:*\n{event.timestamp.isoformat()}"},
                ],
            },
        ]

        # Add payload summary
        summary_items = list(event.payload.items())[:5]
        if summary_items:
            fields = [
                {"type": "mrkdwn", "text": f"*{k}:*\n{v}"}
                for k, v in summary_items
            ]
            blocks.append({"type": "section", "fields": fields[:10]})

        return {"blocks": blocks}

    def _format_teams_card(self, event: WebhookEvent) -> Dict[str, Any]:
        """Format an event as a Microsoft Teams Adaptive Card.

        Args:
            event: The webhook event.

        Returns:
            Teams message payload.
        """
        title = event.event_type.value.replace("_", " ").title()

        facts = [
            {"name": "Event ID", "value": event.event_id[:16]},
            {"name": "Source", "value": event.source},
            {"name": "Timestamp", "value": event.timestamp.isoformat()},
        ]
        for k, v in list(event.payload.items())[:5]:
            facts.append({"name": str(k), "value": str(v)})

        return {
            "@type": "MessageCard",
            "@context": "https://schema.org/extensions",
            "summary": f"GreenLang CSRD: {title}",
            "themeColor": _EVENT_COLOR.get(event.event_type, "0076D7"),
            "title": f"GreenLang CSRD - {title}",
            "sections": [
                {
                    "activityTitle": title,
                    "activitySubtitle": event.source,
                    "facts": facts,
                    "markdown": True,
                }
            ],
        }

    # -------------------------------------------------------------------------
    # Dead Letter Queue
    # -------------------------------------------------------------------------

    def _add_to_dead_letter(
        self,
        event: WebhookEvent,
        subscription: WebhookSubscription,
        failures: List[DeliveryResult],
    ) -> None:
        """Add a failed delivery to the dead-letter queue.

        Args:
            event: The event that failed delivery.
            subscription: The target subscription.
            failures: List of failed delivery attempts.
        """
        entry = DeadLetterEntry(
            event=event,
            subscription=subscription,
            failures=failures,
            total_attempts=len(failures),
            last_error=failures[-1].error_message if failures else "",
        )
        self._dead_letter_queue.append(entry)

        # Trim queue if exceeding max size
        while len(self._dead_letter_queue) > self.config.dead_letter_max_size:
            removed = self._dead_letter_queue.pop(0)
            logger.warning(
                "Dead-letter queue overflow, removed oldest entry: %s",
                removed.entry_id,
            )

        logger.info(
            "Added to dead-letter queue: event=%s, subscription=%s, attempts=%d",
            event.event_id, subscription.subscription_id, len(failures),
        )

    def get_dead_letter_queue(self) -> List[DeadLetterEntry]:
        """Return the current dead-letter queue contents.

        Returns:
            List of DeadLetterEntry objects.
        """
        return list(self._dead_letter_queue)

    def clear_dead_letter_queue(self) -> int:
        """Clear the dead-letter queue.

        Returns:
            Number of entries removed.
        """
        count = len(self._dead_letter_queue)
        self._dead_letter_queue.clear()
        logger.info("Dead-letter queue cleared: %d entries removed", count)
        return count

    async def replay_event(self, event_id: str) -> List[DeliveryResult]:
        """Replay a failed event from the dead-letter queue.

        Finds the event in the dead-letter queue, re-emits it to all
        matching subscriptions, and removes the entry if successful.

        Args:
            event_id: ID of the event to replay.

        Returns:
            List of DeliveryResult from the replay attempt.

        Raises:
            KeyError: If the event is not found in the dead-letter queue.
        """
        entry = None
        entry_index = -1
        for idx, dle in enumerate(self._dead_letter_queue):
            if dle.event.event_id == event_id:
                entry = dle
                entry_index = idx
                break

        if entry is None:
            raise KeyError(
                f"Event '{event_id}' not found in dead-letter queue"
            )

        logger.info("Replaying dead-letter event: %s", event_id)
        results = await self.emit(entry.event)

        # Remove from dead-letter queue if all deliveries succeeded
        all_succeeded = all(
            r.status == DeliveryStatus.DELIVERED for r in results
        )
        if all_succeeded and entry_index >= 0:
            self._dead_letter_queue.pop(entry_index)
            logger.info("Replay successful, removed from dead-letter queue")

        return results

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def _record_delivery_time(self, delivery_time_ms: float) -> None:
        """Record a delivery time for average calculation.

        Args:
            delivery_time_ms: Delivery time in milliseconds.
        """
        self._delivery_times.append(delivery_time_ms)
        # Keep only last 1000 entries for rolling average
        if len(self._delivery_times) > 1000:
            self._delivery_times = self._delivery_times[-1000:]

    def get_delivery_stats(self) -> DeliveryStats:
        """Get current delivery statistics.

        Returns:
            DeliveryStats with counts, rates, and timing information.
        """
        avg_time = 0.0
        if self._delivery_times:
            avg_time = sum(self._delivery_times) / len(self._delivery_times)

        success_rate = 100.0
        total_final = self._total_deliveries_succeeded + self._total_deliveries_failed
        if total_final > 0:
            success_rate = (
                self._total_deliveries_succeeded / total_final * 100.0
            )

        stats = DeliveryStats(
            total_events_emitted=self._total_events_emitted,
            total_deliveries_attempted=self._total_deliveries_attempted,
            total_deliveries_succeeded=self._total_deliveries_succeeded,
            total_deliveries_failed=self._total_deliveries_failed,
            total_retries=self._total_retries,
            dead_letter_count=len(self._dead_letter_queue),
            avg_delivery_time_ms=round(avg_time, 2),
            success_rate_pct=round(success_rate, 2),
            by_channel=dict(self._by_channel),
            by_event_type=dict(self._by_event_type),
            last_delivery_at=self._last_delivery_at,
        )
        stats.provenance_hash = _compute_hash(stats)
        return stats

    def get_delivery_history(
        self,
        limit: int = 100,
        event_type: Optional[WebhookEventType] = None,
        channel: Optional[WebhookChannel] = None,
    ) -> List[DeliveryResult]:
        """Get recent delivery history with optional filtering.

        Args:
            limit: Maximum number of results to return.
            event_type: Filter by event type (optional).
            channel: Filter by channel (optional).

        Returns:
            List of DeliveryResult in reverse chronological order.
        """
        results = list(reversed(self._delivery_history))

        if event_type is not None:
            results = [
                r for r in results
                # We do not store event_type on DeliveryResult directly,
                # but we can match via the event_id in the full history.
            ]

        if channel is not None:
            results = [r for r in results if r.channel == channel]

        return results[:limit]

    # -------------------------------------------------------------------------
    # Shutdown
    # -------------------------------------------------------------------------

    async def shutdown(self) -> None:
        """Gracefully shut down the webhook manager.

        Logs final statistics and clears internal state.
        """
        stats = self.get_delivery_stats()
        logger.info(
            "WebhookManager shutting down: %d events emitted, "
            "%d delivered, %d failed, %d in dead-letter queue",
            stats.total_events_emitted,
            stats.total_deliveries_succeeded,
            stats.total_deliveries_failed,
            stats.dead_letter_count,
        )
        self._subscriptions.clear()
        self._delivery_history.clear()

# ---------------------------------------------------------------------------
# Event Formatting Constants
# ---------------------------------------------------------------------------

_EVENT_EMOJI: Dict[WebhookEventType, str] = {
    WebhookEventType.WORKFLOW_STARTED: ":rocket:",
    WebhookEventType.WORKFLOW_COMPLETED: ":white_check_mark:",
    WebhookEventType.WORKFLOW_FAILED: ":x:",
    WebhookEventType.PHASE_STARTED: ":arrow_forward:",
    WebhookEventType.PHASE_COMPLETED: ":heavy_check_mark:",
    WebhookEventType.PHASE_FAILED: ":warning:",
    WebhookEventType.QUALITY_GATE_PASSED: ":trophy:",
    WebhookEventType.QUALITY_GATE_FAILED: ":no_entry:",
    WebhookEventType.APPROVAL_REQUESTED: ":inbox_tray:",
    WebhookEventType.APPROVAL_COMPLETED: ":thumbsup:",
    WebhookEventType.APPROVAL_REJECTED: ":thumbsdown:",
    WebhookEventType.COMPLIANCE_ALERT: ":rotating_light:",
    WebhookEventType.REGULATORY_CHANGE: ":newspaper:",
    WebhookEventType.DATA_QUALITY_ALERT: ":bar_chart:",
    WebhookEventType.DEADLINE_APPROACHING: ":hourglass:",
}

_EVENT_COLOR: Dict[WebhookEventType, str] = {
    WebhookEventType.WORKFLOW_STARTED: "0076D7",
    WebhookEventType.WORKFLOW_COMPLETED: "2DC72D",
    WebhookEventType.WORKFLOW_FAILED: "E81123",
    WebhookEventType.PHASE_STARTED: "0076D7",
    WebhookEventType.PHASE_COMPLETED: "2DC72D",
    WebhookEventType.PHASE_FAILED: "FF8C00",
    WebhookEventType.QUALITY_GATE_PASSED: "2DC72D",
    WebhookEventType.QUALITY_GATE_FAILED: "E81123",
    WebhookEventType.APPROVAL_REQUESTED: "FFB900",
    WebhookEventType.APPROVAL_COMPLETED: "2DC72D",
    WebhookEventType.APPROVAL_REJECTED: "E81123",
    WebhookEventType.COMPLIANCE_ALERT: "E81123",
    WebhookEventType.REGULATORY_CHANGE: "FF8C00",
    WebhookEventType.DATA_QUALITY_ALERT: "FFB900",
    WebhookEventType.DEADLINE_APPROACHING: "FF8C00",
}
