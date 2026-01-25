"""
Webhook Manager for GreenLang

Event-driven webhook delivery system providing:
- Webhook registration and management
- Event delivery (calculation complete, alert triggered, etc.)
- Retry logic with exponential backoff
- HMAC-SHA256 signature verification
- Delivery logging and monitoring

Example:
    >>> from app.webhooks import create_webhook_manager
    >>> webhook_manager = create_webhook_manager()
    >>>
    >>> # Register a webhook
    >>> await webhook_manager.register_webhook(
    ...     tenant_id="tenant-123",
    ...     url="https://example.com/webhook",
    ...     events=["execution.completed", "alert.triggered"],
    ...     secret="my-webhook-secret",
    ... )
    >>>
    >>> # Deliver an event
    >>> await webhook_manager.deliver_event(
    ...     event_type="execution.completed",
    ...     tenant_id="tenant-123",
    ...     payload={"execution_id": "exec-123", "status": "completed"},
    ... )
"""

import asyncio
import hashlib
import hmac
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field, HttpUrl

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================


class WebhookEventType(str, Enum):
    """Types of events that can trigger webhooks."""

    # Execution events
    EXECUTION_STARTED = "execution.started"
    EXECUTION_PROGRESS = "execution.progress"
    EXECUTION_COMPLETED = "execution.completed"
    EXECUTION_FAILED = "execution.failed"

    # Agent events
    AGENT_CREATED = "agent.created"
    AGENT_UPDATED = "agent.updated"
    AGENT_CERTIFIED = "agent.certified"
    AGENT_DEPRECATED = "agent.deprecated"

    # Batch events
    BATCH_STARTED = "batch.started"
    BATCH_COMPLETED = "batch.completed"
    BATCH_FAILED = "batch.failed"

    # Alert events
    ALERT_TRIGGERED = "alert.triggered"
    ALERT_RESOLVED = "alert.resolved"

    # Calculation events
    CALCULATION_RESULT = "calculation.result"
    THRESHOLD_EXCEEDED = "threshold.exceeded"

    # System events
    QUOTA_WARNING = "quota.warning"
    QUOTA_EXCEEDED = "quota.exceeded"

    # Wildcard
    ALL = "*"


class DeliveryStatus(str, Enum):
    """Webhook delivery status."""

    PENDING = "pending"
    DELIVERING = "delivering"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"


# Default configuration
DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_MAX_RETRIES = 5
DEFAULT_RETRY_DELAYS = [60, 300, 900, 3600, 7200]  # 1m, 5m, 15m, 1h, 2h


# =============================================================================
# Models
# =============================================================================


class WebhookConfig(BaseModel):
    """Configuration for a webhook endpoint."""

    id: str = Field(default_factory=lambda: f"wh_{uuid.uuid4().hex[:12]}")
    tenant_id: str
    name: str = Field(..., description="Friendly name for the webhook")
    url: HttpUrl = Field(..., description="Webhook endpoint URL")
    events: List[str] = Field(..., description="List of event types to subscribe to")
    secret: str = Field(..., description="Secret for HMAC signature")
    headers: Dict[str, str] = Field(default_factory=dict, description="Custom headers")
    is_active: bool = Field(True, description="Whether webhook is active")
    timeout_seconds: int = Field(DEFAULT_TIMEOUT_SECONDS, ge=1, le=60)
    max_retries: int = Field(DEFAULT_MAX_RETRIES, ge=0, le=10)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_triggered_at: Optional[datetime] = None
    failure_count: int = Field(0, description="Consecutive failure count")

    class Config:
        use_enum_values = True


class WebhookEvent(BaseModel):
    """Event to be delivered via webhook."""

    id: str = Field(default_factory=lambda: f"evt_{uuid.uuid4().hex[:12]}")
    event_type: str
    tenant_id: str
    payload: Dict[str, Any]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = Field("greenlang", description="Event source")
    version: str = Field("1.0", description="Event schema version")

    def to_delivery_payload(self) -> Dict[str, Any]:
        """Convert to webhook delivery payload."""
        return {
            "id": self.id,
            "type": self.event_type,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "version": self.version,
            "data": self.payload,
        }


class WebhookDelivery(BaseModel):
    """Record of a webhook delivery attempt."""

    id: str = Field(default_factory=lambda: f"del_{uuid.uuid4().hex[:12]}")
    webhook_id: str
    event_id: str
    event_type: str
    url: str
    status: DeliveryStatus = DeliveryStatus.PENDING
    attempt: int = 1
    response_status_code: Optional[int] = None
    response_body: Optional[str] = None
    error_message: Optional[str] = None
    duration_ms: Optional[float] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    delivered_at: Optional[datetime] = None
    next_retry_at: Optional[datetime] = None

    class Config:
        use_enum_values = True


class WebhookCreateRequest(BaseModel):
    """Request to create a webhook."""

    name: str = Field(..., min_length=1, max_length=100)
    url: HttpUrl
    events: List[str] = Field(..., min_items=1)
    secret: Optional[str] = Field(None, description="Secret for signing (auto-generated if not provided)")
    headers: Dict[str, str] = Field(default_factory=dict)
    timeout_seconds: int = Field(DEFAULT_TIMEOUT_SECONDS, ge=1, le=60)
    max_retries: int = Field(DEFAULT_MAX_RETRIES, ge=0, le=10)


class WebhookUpdateRequest(BaseModel):
    """Request to update a webhook."""

    name: Optional[str] = Field(None, min_length=1, max_length=100)
    url: Optional[HttpUrl] = None
    events: Optional[List[str]] = Field(None, min_items=1)
    headers: Optional[Dict[str, str]] = None
    is_active: Optional[bool] = None
    timeout_seconds: Optional[int] = Field(None, ge=1, le=60)
    max_retries: Optional[int] = Field(None, ge=0, le=10)


class WebhookResponse(BaseModel):
    """Response model for webhook."""

    id: str
    name: str
    url: str
    events: List[str]
    is_active: bool
    timeout_seconds: int
    max_retries: int
    created_at: datetime
    updated_at: datetime
    last_triggered_at: Optional[datetime]
    failure_count: int

    # Don't expose secret in response


class WebhookListResponse(BaseModel):
    """Paginated list of webhooks."""

    data: List[WebhookResponse]
    meta: Dict[str, Any]


class DeliveryListResponse(BaseModel):
    """Paginated list of deliveries."""

    data: List[WebhookDelivery]
    meta: Dict[str, Any]


# =============================================================================
# Signature Generation and Verification
# =============================================================================


def generate_signature(
    payload: bytes,
    secret: str,
    timestamp: str,
) -> str:
    """
    Generate HMAC-SHA256 signature for webhook payload.

    Args:
        payload: The webhook payload as bytes
        secret: The webhook secret
        timestamp: ISO timestamp string

    Returns:
        Hex-encoded signature
    """
    # Create signature base string: timestamp.payload
    signature_base = f"{timestamp}.{payload.decode('utf-8')}"
    signature = hmac.new(
        secret.encode("utf-8"),
        signature_base.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return signature


def verify_signature(
    payload: bytes,
    secret: str,
    timestamp: str,
    provided_signature: str,
    tolerance_seconds: int = 300,
) -> bool:
    """
    Verify webhook signature.

    Args:
        payload: The webhook payload as bytes
        secret: The webhook secret
        timestamp: ISO timestamp from header
        provided_signature: Signature from header
        tolerance_seconds: Maximum age of signature

    Returns:
        True if signature is valid
    """
    # Check timestamp freshness
    try:
        event_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - event_time).total_seconds()
        if abs(age) > tolerance_seconds:
            logger.warning(f"Webhook signature too old: {age}s")
            return False
    except ValueError:
        logger.warning(f"Invalid timestamp format: {timestamp}")
        return False

    # Compute expected signature
    expected_signature = generate_signature(payload, secret, timestamp)

    # Constant-time comparison
    return hmac.compare_digest(expected_signature, provided_signature)


# =============================================================================
# Webhook Store
# =============================================================================


class WebhookStore:
    """
    In-memory store for webhooks and deliveries.

    In production, this would be backed by a database.
    """

    def __init__(self):
        self._webhooks: Dict[str, WebhookConfig] = {}
        self._deliveries: Dict[str, WebhookDelivery] = {}
        self._tenant_webhooks: Dict[str, Set[str]] = {}

    async def create_webhook(self, config: WebhookConfig) -> WebhookConfig:
        """Create a new webhook."""
        self._webhooks[config.id] = config

        if config.tenant_id not in self._tenant_webhooks:
            self._tenant_webhooks[config.tenant_id] = set()
        self._tenant_webhooks[config.tenant_id].add(config.id)

        return config

    async def get_webhook(self, webhook_id: str, tenant_id: str) -> Optional[WebhookConfig]:
        """Get a webhook by ID."""
        webhook = self._webhooks.get(webhook_id)
        if webhook and webhook.tenant_id == tenant_id:
            return webhook
        return None

    async def update_webhook(
        self,
        webhook_id: str,
        tenant_id: str,
        updates: Dict[str, Any],
    ) -> Optional[WebhookConfig]:
        """Update a webhook."""
        webhook = await self.get_webhook(webhook_id, tenant_id)
        if not webhook:
            return None

        for key, value in updates.items():
            if hasattr(webhook, key) and value is not None:
                setattr(webhook, key, value)

        webhook.updated_at = datetime.now(timezone.utc)
        self._webhooks[webhook_id] = webhook
        return webhook

    async def delete_webhook(self, webhook_id: str, tenant_id: str) -> bool:
        """Delete a webhook."""
        webhook = await self.get_webhook(webhook_id, tenant_id)
        if not webhook:
            return False

        del self._webhooks[webhook_id]
        if tenant_id in self._tenant_webhooks:
            self._tenant_webhooks[tenant_id].discard(webhook_id)
        return True

    async def list_webhooks(
        self,
        tenant_id: str,
        is_active: Optional[bool] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[List[WebhookConfig], int]:
        """List webhooks for a tenant."""
        webhook_ids = self._tenant_webhooks.get(tenant_id, set())
        webhooks = [
            self._webhooks[wh_id]
            for wh_id in webhook_ids
            if wh_id in self._webhooks
        ]

        if is_active is not None:
            webhooks = [w for w in webhooks if w.is_active == is_active]

        webhooks.sort(key=lambda w: w.created_at, reverse=True)

        total = len(webhooks)
        return webhooks[offset:offset + limit], total

    async def get_webhooks_for_event(
        self,
        tenant_id: str,
        event_type: str,
    ) -> List[WebhookConfig]:
        """Get all active webhooks subscribed to an event."""
        webhook_ids = self._tenant_webhooks.get(tenant_id, set())
        webhooks = []

        for wh_id in webhook_ids:
            webhook = self._webhooks.get(wh_id)
            if not webhook or not webhook.is_active:
                continue

            # Check if subscribed to this event or wildcard
            if event_type in webhook.events or "*" in webhook.events:
                webhooks.append(webhook)

        return webhooks

    async def create_delivery(self, delivery: WebhookDelivery) -> WebhookDelivery:
        """Create a delivery record."""
        self._deliveries[delivery.id] = delivery
        return delivery

    async def update_delivery(
        self,
        delivery_id: str,
        updates: Dict[str, Any],
    ) -> Optional[WebhookDelivery]:
        """Update a delivery record."""
        delivery = self._deliveries.get(delivery_id)
        if not delivery:
            return None

        for key, value in updates.items():
            if hasattr(delivery, key) and value is not None:
                setattr(delivery, key, value)

        self._deliveries[delivery_id] = delivery
        return delivery

    async def list_deliveries(
        self,
        webhook_id: str,
        status_filter: Optional[DeliveryStatus] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[List[WebhookDelivery], int]:
        """List deliveries for a webhook."""
        deliveries = [
            d for d in self._deliveries.values()
            if d.webhook_id == webhook_id
        ]

        if status_filter:
            deliveries = [d for d in deliveries if d.status == status_filter]

        deliveries.sort(key=lambda d: d.created_at, reverse=True)

        total = len(deliveries)
        return deliveries[offset:offset + limit], total

    async def get_pending_retries(self) -> List[WebhookDelivery]:
        """Get deliveries pending retry."""
        now = datetime.now(timezone.utc)
        return [
            d for d in self._deliveries.values()
            if d.status == DeliveryStatus.RETRYING
            and d.next_retry_at
            and d.next_retry_at <= now
        ]


# Global store instance
_webhook_store = WebhookStore()


# =============================================================================
# Webhook Manager
# =============================================================================


class WebhookManager:
    """
    Manages webhook registration, event delivery, and retries.

    Features:
    - Webhook CRUD operations
    - Event delivery with HMAC signatures
    - Exponential backoff retry logic
    - Delivery logging and monitoring
    - Circuit breaker for failing endpoints

    Example:
        >>> manager = WebhookManager()
        >>> await manager.start()
        >>>
        >>> # Register webhook
        >>> webhook = await manager.register_webhook(
        ...     tenant_id="tenant-123",
        ...     name="My Webhook",
        ...     url="https://example.com/webhook",
        ...     events=["execution.completed"],
        ...     secret="my-secret",
        ... )
        >>>
        >>> # Deliver event
        >>> await manager.deliver_event(
        ...     event_type="execution.completed",
        ...     tenant_id="tenant-123",
        ...     payload={"execution_id": "exec-123"},
        ... )
    """

    def __init__(
        self,
        store: Optional[WebhookStore] = None,
        http_client: Optional[httpx.AsyncClient] = None,
        retry_delays: Optional[List[int]] = None,
    ):
        """
        Initialize the webhook manager.

        Args:
            store: Webhook store (uses global if not provided)
            http_client: HTTP client for deliveries
            retry_delays: List of retry delays in seconds
        """
        self.store = store or _webhook_store
        self.http_client = http_client
        self.retry_delays = retry_delays or DEFAULT_RETRY_DELAYS

        self._retry_task: Optional[asyncio.Task] = None
        self._event_handlers: Dict[str, List[Callable]] = {}

    async def start(self) -> None:
        """Start the webhook manager and retry worker."""
        if not self.http_client:
            self.http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(DEFAULT_TIMEOUT_SECONDS),
                follow_redirects=True,
            )

        self._retry_task = asyncio.create_task(self._retry_worker())
        logger.info("Webhook manager started")

    async def stop(self) -> None:
        """Stop the webhook manager."""
        if self._retry_task:
            self._retry_task.cancel()
            try:
                await self._retry_task
            except asyncio.CancelledError:
                pass

        if self.http_client:
            await self.http_client.aclose()

        logger.info("Webhook manager stopped")

    # =========================================================================
    # Webhook CRUD
    # =========================================================================

    async def register_webhook(
        self,
        tenant_id: str,
        name: str,
        url: str,
        events: List[str],
        secret: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> WebhookConfig:
        """
        Register a new webhook.

        Args:
            tenant_id: Tenant ID
            name: Friendly name
            url: Webhook endpoint URL
            events: List of event types to subscribe to
            secret: Signing secret (auto-generated if not provided)
            headers: Custom headers to include
            timeout_seconds: Request timeout
            max_retries: Maximum retry attempts

        Returns:
            Created webhook configuration
        """
        # Generate secret if not provided
        if not secret:
            secret = f"whsec_{uuid.uuid4().hex}"

        config = WebhookConfig(
            tenant_id=tenant_id,
            name=name,
            url=url,
            events=events,
            secret=secret,
            headers=headers or {},
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )

        return await self.store.create_webhook(config)

    async def get_webhook(
        self,
        webhook_id: str,
        tenant_id: str,
    ) -> Optional[WebhookConfig]:
        """Get a webhook by ID."""
        return await self.store.get_webhook(webhook_id, tenant_id)

    async def update_webhook(
        self,
        webhook_id: str,
        tenant_id: str,
        updates: Dict[str, Any],
    ) -> Optional[WebhookConfig]:
        """Update a webhook."""
        return await self.store.update_webhook(webhook_id, tenant_id, updates)

    async def delete_webhook(
        self,
        webhook_id: str,
        tenant_id: str,
    ) -> bool:
        """Delete a webhook."""
        return await self.store.delete_webhook(webhook_id, tenant_id)

    async def list_webhooks(
        self,
        tenant_id: str,
        is_active: Optional[bool] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[List[WebhookConfig], int]:
        """List webhooks for a tenant."""
        return await self.store.list_webhooks(tenant_id, is_active, limit, offset)

    async def rotate_secret(
        self,
        webhook_id: str,
        tenant_id: str,
    ) -> Optional[str]:
        """
        Rotate webhook secret.

        Returns:
            New secret, or None if webhook not found
        """
        new_secret = f"whsec_{uuid.uuid4().hex}"
        webhook = await self.store.update_webhook(
            webhook_id,
            tenant_id,
            {"secret": new_secret},
        )
        return new_secret if webhook else None

    # =========================================================================
    # Event Delivery
    # =========================================================================

    async def deliver_event(
        self,
        event_type: str,
        tenant_id: str,
        payload: Dict[str, Any],
    ) -> List[WebhookDelivery]:
        """
        Deliver an event to all subscribed webhooks.

        Args:
            event_type: Type of event
            tenant_id: Tenant ID
            payload: Event payload

        Returns:
            List of delivery records
        """
        # Create event
        event = WebhookEvent(
            event_type=event_type,
            tenant_id=tenant_id,
            payload=payload,
        )

        # Get subscribed webhooks
        webhooks = await self.store.get_webhooks_for_event(tenant_id, event_type)

        if not webhooks:
            logger.debug(f"No webhooks subscribed to {event_type} for tenant {tenant_id}")
            return []

        # Deliver to each webhook
        deliveries = []
        for webhook in webhooks:
            delivery = await self._deliver_to_webhook(webhook, event)
            deliveries.append(delivery)

        return deliveries

    async def _deliver_to_webhook(
        self,
        webhook: WebhookConfig,
        event: WebhookEvent,
    ) -> WebhookDelivery:
        """Deliver an event to a single webhook."""
        # Create delivery record
        delivery = WebhookDelivery(
            webhook_id=webhook.id,
            event_id=event.id,
            event_type=event.event_type,
            url=str(webhook.url),
        )
        delivery = await self.store.create_delivery(delivery)

        # Attempt delivery
        await self._attempt_delivery(webhook, event, delivery)

        return delivery

    async def _attempt_delivery(
        self,
        webhook: WebhookConfig,
        event: WebhookEvent,
        delivery: WebhookDelivery,
    ) -> None:
        """Attempt to deliver a webhook."""
        if not self.http_client:
            raise RuntimeError("HTTP client not initialized")

        # Update status
        await self.store.update_delivery(delivery.id, {
            "status": DeliveryStatus.DELIVERING,
        })

        # Prepare payload
        payload = event.to_delivery_payload()
        payload_bytes = json.dumps(payload).encode("utf-8")

        # Generate signature
        timestamp = event.timestamp.isoformat()
        signature = generate_signature(payload_bytes, webhook.secret, timestamp)

        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "X-Webhook-Signature": signature,
            "X-Webhook-Timestamp": timestamp,
            "X-Webhook-Event": event.event_type,
            "X-Webhook-ID": webhook.id,
            "X-Delivery-ID": delivery.id,
            "User-Agent": "GreenLang-Webhook/1.0",
            **webhook.headers,
        }

        start_time = time.time()

        try:
            response = await self.http_client.post(
                str(webhook.url),
                content=payload_bytes,
                headers=headers,
                timeout=webhook.timeout_seconds,
            )

            duration_ms = (time.time() - start_time) * 1000

            # Check for success (2xx status codes)
            if 200 <= response.status_code < 300:
                await self.store.update_delivery(delivery.id, {
                    "status": DeliveryStatus.DELIVERED,
                    "response_status_code": response.status_code,
                    "response_body": response.text[:1000] if response.text else None,
                    "duration_ms": duration_ms,
                    "delivered_at": datetime.now(timezone.utc),
                })

                # Reset failure count
                await self.store.update_webhook(webhook.id, webhook.tenant_id, {
                    "failure_count": 0,
                    "last_triggered_at": datetime.now(timezone.utc),
                })

                logger.info(
                    f"Webhook delivered: {delivery.id} to {webhook.url} "
                    f"(status={response.status_code}, duration={duration_ms:.0f}ms)"
                )

            else:
                await self._handle_delivery_failure(
                    webhook,
                    delivery,
                    f"HTTP {response.status_code}",
                    response.status_code,
                    response.text[:1000] if response.text else None,
                    duration_ms,
                )

        except httpx.TimeoutException:
            duration_ms = (time.time() - start_time) * 1000
            await self._handle_delivery_failure(
                webhook,
                delivery,
                "Request timeout",
                None,
                None,
                duration_ms,
            )

        except httpx.RequestError as e:
            duration_ms = (time.time() - start_time) * 1000
            await self._handle_delivery_failure(
                webhook,
                delivery,
                f"Request error: {e}",
                None,
                None,
                duration_ms,
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Unexpected error delivering webhook: {e}")
            await self._handle_delivery_failure(
                webhook,
                delivery,
                f"Unexpected error: {e}",
                None,
                None,
                duration_ms,
            )

    async def _handle_delivery_failure(
        self,
        webhook: WebhookConfig,
        delivery: WebhookDelivery,
        error_message: str,
        status_code: Optional[int],
        response_body: Optional[str],
        duration_ms: float,
    ) -> None:
        """Handle a failed delivery attempt."""
        # Update webhook failure count
        new_failure_count = webhook.failure_count + 1
        await self.store.update_webhook(webhook.id, webhook.tenant_id, {
            "failure_count": new_failure_count,
            "last_triggered_at": datetime.now(timezone.utc),
        })

        # Check if we should retry
        if delivery.attempt < webhook.max_retries:
            # Calculate next retry time
            retry_index = min(delivery.attempt - 1, len(self.retry_delays) - 1)
            retry_delay = self.retry_delays[retry_index]
            next_retry_at = datetime.now(timezone.utc) + timedelta(seconds=retry_delay)

            await self.store.update_delivery(delivery.id, {
                "status": DeliveryStatus.RETRYING,
                "attempt": delivery.attempt + 1,
                "response_status_code": status_code,
                "response_body": response_body,
                "error_message": error_message,
                "duration_ms": duration_ms,
                "next_retry_at": next_retry_at,
            })

            logger.warning(
                f"Webhook delivery failed, will retry: {delivery.id} "
                f"(attempt={delivery.attempt}, next_retry={next_retry_at})"
            )

        else:
            await self.store.update_delivery(delivery.id, {
                "status": DeliveryStatus.FAILED,
                "response_status_code": status_code,
                "response_body": response_body,
                "error_message": error_message,
                "duration_ms": duration_ms,
            })

            logger.error(
                f"Webhook delivery failed permanently: {delivery.id} "
                f"(attempts={delivery.attempt}, error={error_message})"
            )

            # Disable webhook if too many consecutive failures
            if new_failure_count >= 10:
                await self.store.update_webhook(webhook.id, webhook.tenant_id, {
                    "is_active": False,
                })
                logger.warning(f"Webhook disabled due to failures: {webhook.id}")

    async def _retry_worker(self) -> None:
        """Background worker to process pending retries."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute

                pending = await self.store.get_pending_retries()

                for delivery in pending:
                    webhook = self._webhook_store._webhooks.get(delivery.webhook_id)
                    if not webhook:
                        continue

                    # Reconstruct event (simplified - in production, store the event)
                    event = WebhookEvent(
                        id=delivery.event_id,
                        event_type=delivery.event_type,
                        tenant_id=webhook.tenant_id,
                        payload={},  # Would need to retrieve from storage
                    )

                    await self._attempt_delivery(webhook, event, delivery)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Retry worker error: {e}")

    # =========================================================================
    # Delivery Management
    # =========================================================================

    async def list_deliveries(
        self,
        webhook_id: str,
        tenant_id: str,
        status_filter: Optional[DeliveryStatus] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[List[WebhookDelivery], int]:
        """List deliveries for a webhook."""
        # Verify webhook ownership
        webhook = await self.store.get_webhook(webhook_id, tenant_id)
        if not webhook:
            raise ValueError(f"Webhook {webhook_id} not found")

        return await self.store.list_deliveries(webhook_id, status_filter, limit, offset)

    async def retry_delivery(
        self,
        delivery_id: str,
        tenant_id: str,
    ) -> Optional[WebhookDelivery]:
        """Manually retry a failed delivery."""
        delivery = self.store._deliveries.get(delivery_id)
        if not delivery:
            return None

        webhook = await self.store.get_webhook(delivery.webhook_id, tenant_id)
        if not webhook:
            return None

        # Reset for retry
        await self.store.update_delivery(delivery_id, {
            "status": DeliveryStatus.PENDING,
            "attempt": 1,
            "next_retry_at": None,
        })

        # Trigger delivery
        event = WebhookEvent(
            id=delivery.event_id,
            event_type=delivery.event_type,
            tenant_id=tenant_id,
            payload={},
        )
        await self._attempt_delivery(webhook, event, delivery)

        return self.store._deliveries.get(delivery_id)

    # =========================================================================
    # Event Handlers (Internal)
    # =========================================================================

    def on_event(self, event_type: str) -> Callable:
        """
        Decorator to register internal event handlers.

        These are called before webhook delivery.
        """
        def decorator(func: Callable) -> Callable:
            if event_type not in self._event_handlers:
                self._event_handlers[event_type] = []
            self._event_handlers[event_type].append(func)
            return func
        return decorator


# =============================================================================
# API Router
# =============================================================================


router = APIRouter(prefix="/webhooks", tags=["Webhooks"])


def get_webhook_manager() -> WebhookManager:
    """Get or create webhook manager."""
    # In production, this would be managed by the application lifecycle
    return WebhookManager()


@router.post(
    "",
    response_model=WebhookResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create webhook",
    description="Register a new webhook endpoint",
)
async def create_webhook(
    request: Request,
    body: WebhookCreateRequest,
) -> WebhookResponse:
    """Create a new webhook."""
    tenant_id = getattr(request.state, "tenant_id", "dev_tenant")
    manager = get_webhook_manager()

    webhook = await manager.register_webhook(
        tenant_id=tenant_id,
        name=body.name,
        url=str(body.url),
        events=body.events,
        secret=body.secret,
        headers=body.headers,
        timeout_seconds=body.timeout_seconds,
        max_retries=body.max_retries,
    )

    return WebhookResponse(
        id=webhook.id,
        name=webhook.name,
        url=str(webhook.url),
        events=webhook.events,
        is_active=webhook.is_active,
        timeout_seconds=webhook.timeout_seconds,
        max_retries=webhook.max_retries,
        created_at=webhook.created_at,
        updated_at=webhook.updated_at,
        last_triggered_at=webhook.last_triggered_at,
        failure_count=webhook.failure_count,
    )


@router.get(
    "",
    response_model=WebhookListResponse,
    summary="List webhooks",
    description="Get paginated list of webhooks",
)
async def list_webhooks(
    request: Request,
    is_active: Optional[bool] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
) -> WebhookListResponse:
    """List webhooks for the current tenant."""
    tenant_id = getattr(request.state, "tenant_id", "dev_tenant")
    manager = get_webhook_manager()

    webhooks, total = await manager.list_webhooks(
        tenant_id=tenant_id,
        is_active=is_active,
        limit=limit,
        offset=offset,
    )

    return WebhookListResponse(
        data=[
            WebhookResponse(
                id=w.id,
                name=w.name,
                url=str(w.url),
                events=w.events,
                is_active=w.is_active,
                timeout_seconds=w.timeout_seconds,
                max_retries=w.max_retries,
                created_at=w.created_at,
                updated_at=w.updated_at,
                last_triggered_at=w.last_triggered_at,
                failure_count=w.failure_count,
            )
            for w in webhooks
        ],
        meta={
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total,
        },
    )


@router.get(
    "/{webhook_id}",
    response_model=WebhookResponse,
    summary="Get webhook",
    description="Get webhook details",
)
async def get_webhook(
    request: Request,
    webhook_id: str,
) -> WebhookResponse:
    """Get webhook details."""
    tenant_id = getattr(request.state, "tenant_id", "dev_tenant")
    manager = get_webhook_manager()

    webhook = await manager.get_webhook(webhook_id, tenant_id)
    if not webhook:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook {webhook_id} not found",
        )

    return WebhookResponse(
        id=webhook.id,
        name=webhook.name,
        url=str(webhook.url),
        events=webhook.events,
        is_active=webhook.is_active,
        timeout_seconds=webhook.timeout_seconds,
        max_retries=webhook.max_retries,
        created_at=webhook.created_at,
        updated_at=webhook.updated_at,
        last_triggered_at=webhook.last_triggered_at,
        failure_count=webhook.failure_count,
    )


@router.patch(
    "/{webhook_id}",
    response_model=WebhookResponse,
    summary="Update webhook",
    description="Update webhook configuration",
)
async def update_webhook(
    request: Request,
    webhook_id: str,
    body: WebhookUpdateRequest,
) -> WebhookResponse:
    """Update webhook configuration."""
    tenant_id = getattr(request.state, "tenant_id", "dev_tenant")
    manager = get_webhook_manager()

    updates = body.dict(exclude_unset=True)
    if "url" in updates:
        updates["url"] = str(updates["url"])

    webhook = await manager.update_webhook(webhook_id, tenant_id, updates)
    if not webhook:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook {webhook_id} not found",
        )

    return WebhookResponse(
        id=webhook.id,
        name=webhook.name,
        url=str(webhook.url),
        events=webhook.events,
        is_active=webhook.is_active,
        timeout_seconds=webhook.timeout_seconds,
        max_retries=webhook.max_retries,
        created_at=webhook.created_at,
        updated_at=webhook.updated_at,
        last_triggered_at=webhook.last_triggered_at,
        failure_count=webhook.failure_count,
    )


@router.delete(
    "/{webhook_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete webhook",
    description="Delete a webhook",
)
async def delete_webhook(
    request: Request,
    webhook_id: str,
) -> None:
    """Delete a webhook."""
    tenant_id = getattr(request.state, "tenant_id", "dev_tenant")
    manager = get_webhook_manager()

    deleted = await manager.delete_webhook(webhook_id, tenant_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook {webhook_id} not found",
        )


@router.post(
    "/{webhook_id}/rotate-secret",
    summary="Rotate secret",
    description="Rotate webhook signing secret",
)
async def rotate_webhook_secret(
    request: Request,
    webhook_id: str,
) -> Dict[str, str]:
    """Rotate webhook signing secret."""
    tenant_id = getattr(request.state, "tenant_id", "dev_tenant")
    manager = get_webhook_manager()

    new_secret = await manager.rotate_secret(webhook_id, tenant_id)
    if not new_secret:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook {webhook_id} not found",
        )

    return {"secret": new_secret}


@router.get(
    "/{webhook_id}/deliveries",
    response_model=DeliveryListResponse,
    summary="List deliveries",
    description="Get delivery history for a webhook",
)
async def list_webhook_deliveries(
    request: Request,
    webhook_id: str,
    status_filter: Optional[DeliveryStatus] = Query(None, alias="status"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> DeliveryListResponse:
    """List deliveries for a webhook."""
    tenant_id = getattr(request.state, "tenant_id", "dev_tenant")
    manager = get_webhook_manager()

    try:
        deliveries, total = await manager.list_deliveries(
            webhook_id=webhook_id,
            tenant_id=tenant_id,
            status_filter=status_filter,
            limit=limit,
            offset=offset,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )

    return DeliveryListResponse(
        data=deliveries,
        meta={
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": offset + limit < total,
        },
    )


@router.post(
    "/{webhook_id}/test",
    summary="Test webhook",
    description="Send a test event to verify webhook configuration",
)
async def test_webhook(
    request: Request,
    webhook_id: str,
) -> Dict[str, Any]:
    """Send a test event to a webhook."""
    tenant_id = getattr(request.state, "tenant_id", "dev_tenant")
    manager = get_webhook_manager()

    webhook = await manager.get_webhook(webhook_id, tenant_id)
    if not webhook:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Webhook {webhook_id} not found",
        )

    # Send test event
    deliveries = await manager.deliver_event(
        event_type="test.ping",
        tenant_id=tenant_id,
        payload={
            "webhook_id": webhook_id,
            "message": "This is a test event from GreenLang",
        },
    )

    if not deliveries:
        return {"success": False, "error": "No delivery created"}

    delivery = deliveries[0]

    return {
        "success": delivery.status == DeliveryStatus.DELIVERED,
        "delivery_id": delivery.id,
        "status": delivery.status.value,
        "response_status_code": delivery.response_status_code,
        "duration_ms": delivery.duration_ms,
        "error": delivery.error_message,
    }


# =============================================================================
# Factory Function
# =============================================================================


def create_webhook_manager(
    store: Optional[WebhookStore] = None,
    http_client: Optional[httpx.AsyncClient] = None,
) -> WebhookManager:
    """
    Create a webhook manager.

    Args:
        store: Optional webhook store
        http_client: Optional HTTP client

    Returns:
        Configured WebhookManager

    Example:
        >>> manager = create_webhook_manager()
        >>> await manager.start()
    """
    return WebhookManager(store=store, http_client=http_client)
