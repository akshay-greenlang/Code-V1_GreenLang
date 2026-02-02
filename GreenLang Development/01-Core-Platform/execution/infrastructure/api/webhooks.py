"""Webhook Framework for GreenLang Process Heat Agents.

Provides webhook subscriber management with event-driven delivery.
Supports pluggable storage backends for persistence.

Storage Backends:
    - InMemory (default): For testing, non-persistent
    - SQLite: File-based persistence for single-node deployments
    - PostgreSQL: Production-grade persistence for distributed systems

Example:
    >>> from greenlang.infrastructure.api.storage import StorageFactory
    >>> store = StorageFactory.get_webhook_store({"backend": "sqlite"})
    >>> manager = WebhookSubscriberManager(store=store)
"""

import asyncio
import hashlib
import hmac
import httpx
import json
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from uuid import uuid4

from pydantic import BaseModel, Field, validator

try:
    from fastapi import APIRouter, HTTPException, status
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = object
    HTTPException = Exception
    status = None

if TYPE_CHECKING:
    from greenlang.infrastructure.api.storage.webhook_store import BaseWebhookStore

logger = logging.getLogger(__name__)

PROCESS_HEAT_EVENTS = {
    "calculation.completed",
    "calculation.failed",
    "alarm.triggered",
    "alarm.cleared",
    "model.deployed",
    "model.degraded",
    "compliance.violation",
    "compliance.report_ready",
}


class WebhookStatus(str, Enum):
    """Webhook delivery status."""
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"
    RETRYING = "retrying"


class WebhookModel(BaseModel):
    """Registered webhook subscriber."""
    webhook_id: str = Field(default_factory=lambda: str(uuid4()))
    url: str = Field(..., description="Callback URL")
    events: List[str] = Field(..., description="Subscribed event types")
    secret: str = Field(..., description="HMAC signing secret")
    is_active: bool = Field(default=True)
    health_status: str = Field(default="healthy")
    consecutive_failures: int = Field(default=0)
    last_triggered_at: Optional[datetime] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @validator("events")
    def validate_events(cls, v):
        """Validate events are supported."""
        for event in v:
            if event not in PROCESS_HEAT_EVENTS:
                raise ValueError(f"Unsupported event type: {event}")
        return v


class WebhookDelivery(BaseModel):
    """Webhook delivery attempt record."""
    delivery_id: str = Field(default_factory=lambda: str(uuid4()))
    webhook_id: str = Field(...)
    event_type: str = Field(...)
    payload: Dict[str, Any] = Field(...)
    signature: str = Field(description="HMAC-SHA256 signature")
    status: WebhookStatus = Field(default=WebhookStatus.PENDING)
    attempt: int = Field(default=1)
    http_status: Optional[int] = Field(default=None)
    error_message: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    sent_at: Optional[datetime] = Field(default=None)
    provenance_hash: str = Field(default="")


class RegisterWebhookRequest(BaseModel):
    """Register webhook request."""
    url: str = Field(..., description="Callback URL")
    events: List[str] = Field(..., description="Event subscriptions")
    secret: str = Field(..., description="HMAC secret")
    metadata: Optional[Dict[str, Any]] = Field(default=None)


class WebhookResponse(BaseModel):
    """Webhook registration response."""
    webhook_id: str
    message: str


class WebhookSubscriberManager:
    """
    Webhook subscriber manager for Process Heat agents.

    Manages external webhook registration, event delivery with
    HMAC-SHA256 signatures, exponential backoff retries, and
    health monitoring.

    Supports pluggable storage backends for webhook and delivery persistence.
    When no store is provided, uses in-memory storage (non-persistent).

    Attributes:
        store: Optional storage backend for persistence
        webhooks: In-memory webhook cache (synced with store if provided)
        deliveries: In-memory delivery cache (synced with store if provided)

    Methods:
        register_webhook(url, events, secret) - Register subscriber
        unregister_webhook(webhook_id) - Unregister subscriber
        list_webhooks() - List all webhooks
        trigger_webhook(event_type, payload) - Send event
        verify_signature(payload, signature, secret) - Verify HMAC

    Example:
        >>> # Without persistence (in-memory only)
        >>> manager = WebhookSubscriberManager()
        >>>
        >>> # With SQLite persistence
        >>> from greenlang.infrastructure.api.storage import StorageFactory
        >>> store = StorageFactory.get_webhook_store({"backend": "sqlite"})
        >>> manager = WebhookSubscriberManager(store=store)
        >>>
        >>> webhook_id = await manager.register_webhook(
        ...     url="https://api.example.com/events",
        ...     events=["calculation.completed"],
        ...     secret="my-secret"
        ... )
        >>> await manager.trigger_webhook(
        ...     "calculation.completed",
        ...     {"result": 123.45}
        ... )
    """

    def __init__(
        self,
        base_url: str = "/api/v1/webhooks",
        store: Optional["BaseWebhookStore"] = None
    ):
        """
        Initialize webhook subscriber manager.

        Args:
            base_url: Base URL path for FastAPI routes
            store: Optional storage backend for persistence.
                   If None, uses in-memory storage only.
        """
        self._store: Optional["BaseWebhookStore"] = store
        self.webhooks: Dict[str, WebhookModel] = {}
        self.deliveries: Dict[str, WebhookDelivery] = {}
        self.router = APIRouter(prefix=base_url, tags=["webhooks"]) if FASTAPI_AVAILABLE else None
        self._http_client: Optional[httpx.AsyncClient] = None
        self._retry_queue: asyncio.Queue = asyncio.Queue()
        self._initialized = False
        self._setup_routes()
        logger.info(
            f"WebhookSubscriberManager initialized "
            f"(store={'provided' if store else 'in-memory'})"
        )

    def _setup_routes(self) -> None:
        """Setup FastAPI routes."""
        if not FASTAPI_AVAILABLE or not self.router:
            return

        @self.router.post(
            "",
            response_model=WebhookResponse,
            status_code=201,
            summary="Register webhook subscriber"
        )
        async def register(req: RegisterWebhookRequest) -> WebhookResponse:
            webhook_id = await self.register_webhook_async(
                url=req.url,
                events=req.events,
                secret=req.secret,
                metadata=req.metadata
            )
            return WebhookResponse(
                webhook_id=webhook_id,
                message="Webhook registered successfully"
            )

        @self.router.delete(
            "/{webhook_id}",
            response_model=WebhookResponse,
            summary="Unregister webhook"
        )
        async def unregister(webhook_id: str) -> WebhookResponse:
            if await self.unregister_webhook_async(webhook_id):
                return WebhookResponse(
                    webhook_id=webhook_id,
                    message="Webhook unregistered successfully"
                )
            raise HTTPException(status_code=404, detail="Webhook not found")

        @self.router.get(
            "",
            response_model=List[WebhookModel],
            summary="List all webhooks"
        )
        async def list_hooks() -> List[WebhookModel]:
            return await self.list_webhooks_async()

        @self.router.post(
            "/{webhook_id}/test",
            response_model=dict,
            summary="Test webhook delivery"
        )
        async def test_delivery(webhook_id: str) -> dict:
            webhook = await self.get_webhook_async(webhook_id)
            if not webhook:
                raise HTTPException(status_code=404, detail="Webhook not found")

            test_payload = {
                "event_type": "test",
                "timestamp": datetime.utcnow().isoformat(),
                "message": "Test delivery"
            }

            delivery_id = await self.trigger_webhook("test", test_payload, [webhook_id])
            delivery = self.deliveries.get(delivery_id)

            return {
                "delivery_id": delivery_id,
                "webhook_id": webhook_id,
                "status": delivery.status.value if delivery else "unknown",
                "message": "Test webhook sent"
            }

        @self.router.get(
            "/{webhook_id}/deliveries",
            response_model=List[WebhookDelivery],
            summary="Get webhook delivery history"
        )
        async def get_deliveries(
            webhook_id: str,
            limit: int = 100,
            offset: int = 0
        ) -> List[WebhookDelivery]:
            webhook = await self.get_webhook_async(webhook_id)
            if not webhook:
                raise HTTPException(status_code=404, detail="Webhook not found")
            return await self.get_deliveries_async(webhook_id, limit, offset)

    async def _ensure_initialized(self) -> None:
        """Ensure storage backend is initialized and cache is populated."""
        if self._initialized:
            return

        if self._store and hasattr(self._store, 'initialize'):
            await self._store.initialize()

        # Load existing webhooks from store into cache
        if self._store:
            stored_webhooks = await self._store.list_webhooks()
            for webhook in stored_webhooks:
                self.webhooks[webhook.webhook_id] = webhook
            logger.info(f"Loaded {len(stored_webhooks)} webhooks from storage")

        self._initialized = True

    def register_webhook(
        self,
        url: str,
        events: List[str],
        secret: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Register external webhook subscriber (synchronous version).

        Note: This is a synchronous wrapper. For async code, use register_webhook_async().
        Storage persistence will be done asynchronously if a store is configured.

        Args:
            url: Callback URL for webhook delivery
            events: List of event types to subscribe to
            secret: HMAC signing secret for payload verification
            metadata: Optional additional metadata

        Returns:
            The webhook_id of the registered webhook
        """
        webhook_id = str(uuid4())
        webhook = WebhookModel(
            webhook_id=webhook_id,
            url=url,
            events=events,
            secret=secret,
            metadata=metadata or {}
        )
        self.webhooks[webhook_id] = webhook

        # Schedule async storage if store is available
        if self._store:
            asyncio.create_task(self._persist_webhook(webhook))

        logger.info(f"Registered webhook {webhook_id} for {len(events)} events to {url}")
        return webhook_id

    async def register_webhook_async(
        self,
        url: str,
        events: List[str],
        secret: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Register external webhook subscriber (async version).

        Persists to storage backend if configured.

        Args:
            url: Callback URL for webhook delivery
            events: List of event types to subscribe to
            secret: HMAC signing secret for payload verification
            metadata: Optional additional metadata

        Returns:
            The webhook_id of the registered webhook
        """
        await self._ensure_initialized()

        webhook_id = str(uuid4())
        webhook = WebhookModel(
            webhook_id=webhook_id,
            url=url,
            events=events,
            secret=secret,
            metadata=metadata or {}
        )

        # Persist to storage first
        if self._store:
            await self._store.save_webhook(webhook)

        # Update cache
        self.webhooks[webhook_id] = webhook

        logger.info(f"Registered webhook {webhook_id} for {len(events)} events to {url}")
        return webhook_id

    async def _persist_webhook(self, webhook: WebhookModel) -> None:
        """Persist webhook to storage backend."""
        if self._store:
            try:
                await self._store.save_webhook(webhook)
            except Exception as e:
                logger.error(f"Failed to persist webhook {webhook.webhook_id}: {e}")

    def unregister_webhook(self, webhook_id: str) -> bool:
        """
        Unregister webhook subscriber (synchronous version).

        Note: This is a synchronous wrapper. For async code, use unregister_webhook_async().

        Args:
            webhook_id: The webhook ID to unregister

        Returns:
            True if webhook was found and deleted, False otherwise
        """
        if webhook_id in self.webhooks:
            del self.webhooks[webhook_id]

            # Schedule async storage deletion if store is available
            if self._store:
                asyncio.create_task(self._delete_webhook_from_store(webhook_id))

            logger.info(f"Unregistered webhook {webhook_id}")
            return True
        return False

    async def unregister_webhook_async(self, webhook_id: str) -> bool:
        """
        Unregister webhook subscriber (async version).

        Removes from storage backend if configured.

        Args:
            webhook_id: The webhook ID to unregister

        Returns:
            True if webhook was found and deleted, False otherwise
        """
        await self._ensure_initialized()

        # Delete from storage first
        if self._store:
            deleted = await self._store.delete_webhook(webhook_id)
            if not deleted and webhook_id not in self.webhooks:
                return False

        # Update cache
        if webhook_id in self.webhooks:
            del self.webhooks[webhook_id]
            logger.info(f"Unregistered webhook {webhook_id}")
            return True

        return False

    async def _delete_webhook_from_store(self, webhook_id: str) -> None:
        """Delete webhook from storage backend."""
        if self._store:
            try:
                await self._store.delete_webhook(webhook_id)
            except Exception as e:
                logger.error(f"Failed to delete webhook {webhook_id} from store: {e}")

    def list_webhooks(self) -> List[WebhookModel]:
        """
        List all registered webhooks (synchronous version).

        Note: Returns cached webhooks only. For storage-backed listing,
        use list_webhooks_async().

        Returns:
            List of all registered webhook models
        """
        return list(self.webhooks.values())

    async def list_webhooks_async(self) -> List[WebhookModel]:
        """
        List all registered webhooks (async version).

        Fetches from storage backend if configured, otherwise returns cached webhooks.

        Returns:
            List of all registered webhook models
        """
        await self._ensure_initialized()

        if self._store:
            webhooks = await self._store.list_webhooks()
            # Update cache
            self.webhooks = {w.webhook_id: w for w in webhooks}
            return webhooks

        return list(self.webhooks.values())

    async def get_webhook_async(self, webhook_id: str) -> Optional[WebhookModel]:
        """
        Get a specific webhook by ID.

        Args:
            webhook_id: The webhook ID to retrieve

        Returns:
            The webhook model if found, None otherwise
        """
        await self._ensure_initialized()

        # Check cache first
        if webhook_id in self.webhooks:
            return self.webhooks[webhook_id]

        # Check storage
        if self._store:
            webhook = await self._store.get_webhook(webhook_id)
            if webhook:
                self.webhooks[webhook_id] = webhook
            return webhook

        return None

    async def get_deliveries_async(
        self,
        webhook_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[WebhookDelivery]:
        """
        Get delivery history for a webhook.

        Args:
            webhook_id: The webhook ID
            limit: Maximum number of deliveries to return
            offset: Number of deliveries to skip

        Returns:
            List of delivery records
        """
        if self._store:
            return await self._store.get_deliveries(webhook_id, limit, offset)

        # Return from in-memory cache
        deliveries = [
            d for d in self.deliveries.values()
            if d.webhook_id == webhook_id
        ]
        deliveries.sort(key=lambda d: d.created_at, reverse=True)
        return deliveries[offset:offset + limit]

    def _create_signature(self, payload: Dict[str, Any], secret: str) -> str:
        """Create HMAC-SHA256 signature for payload."""
        payload_str = json.dumps(payload, sort_keys=True, default=str)
        signature = hmac.new(
            secret.encode(),
            payload_str.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature

    def verify_signature(self, payload: Dict[str, Any], signature: str, secret: str) -> bool:
        """Verify HMAC-SHA256 signature."""
        expected = self._create_signature(payload, secret)
        return hmac.compare_digest(expected, signature)

    async def trigger_webhook(
        self,
        event_type: str,
        payload: Dict[str, Any],
        webhook_ids: Optional[List[str]] = None
    ) -> str:
        """
        Trigger webhook delivery to subscribers.

        Uses exponential backoff for retries (3 attempts: 2s, 4s, 8s).
        Tracks delivery status and webhook health.
        Disables webhook after 5+ consecutive failures.
        Persists delivery records to storage if configured.

        Args:
            event_type: The event type to trigger
            payload: The event payload to send
            webhook_ids: Optional list of specific webhook IDs to trigger

        Returns:
            The first delivery_id if any webhooks were triggered, empty string otherwise
        """
        await self._ensure_initialized()

        if event_type not in PROCESS_HEAT_EVENTS and event_type != "test":
            logger.warning(f"Unknown event type: {event_type}")
            return ""

        delivery_ids = []

        if webhook_ids:
            targets = [self.webhooks[wid] for wid in webhook_ids if wid in self.webhooks]
        else:
            targets = [w for w in self.webhooks.values()
                      if event_type in w.events and w.is_active]

        if not targets:
            logger.debug(f"No active webhooks subscribed to {event_type}")
            return ""

        for webhook in targets:
            signature = self._create_signature(payload, webhook.secret)
            delivery = WebhookDelivery(
                webhook_id=webhook.webhook_id,
                event_type=event_type,
                payload=payload,
                signature=signature,
                provenance_hash=hashlib.sha256(
                    f"{event_type}:{json.dumps(payload, sort_keys=True, default=str)}".encode()
                ).hexdigest()
            )

            # Persist delivery to storage
            if self._store:
                await self._store.save_delivery(delivery)

            self.deliveries[delivery.delivery_id] = delivery
            delivery_ids.append(delivery.delivery_id)

            await self._retry_queue.put((delivery.delivery_id, webhook))

        if delivery_ids:
            asyncio.create_task(self._process_delivery_queue())

        return delivery_ids[0] if delivery_ids else ""

    async def _process_delivery_queue(self) -> None:
        """Process queued webhook deliveries with exponential backoff."""
        if not self._http_client:
            self._http_client = httpx.AsyncClient(timeout=10)

        while not self._retry_queue.empty():
            try:
                delivery_id, webhook = await asyncio.wait_for(
                    self._retry_queue.get(),
                    timeout=1
                )
                delivery = self.deliveries.get(delivery_id)
                if not delivery:
                    continue

                success = await self._send_webhook(delivery, webhook)

                if not success and delivery.attempt < 3:
                    delay = 2 ** delivery.attempt
                    await asyncio.sleep(delay)
                    delivery.attempt += 1
                    delivery.status = WebhookStatus.RETRYING

                    # Persist delivery status update
                    if self._store:
                        await self._store.save_delivery(delivery)

                    await self._retry_queue.put((delivery_id, webhook))
                else:
                    if success:
                        webhook.consecutive_failures = 0
                    else:
                        webhook.consecutive_failures += 1
                        if webhook.consecutive_failures >= 5:
                            webhook.is_active = False
                            webhook.health_status = "unhealthy"
                            logger.warning(
                                f"Webhook {webhook.webhook_id} marked unhealthy "
                                f"({webhook.consecutive_failures} failures)"
                            )

                    # Persist webhook health status update
                    if self._store:
                        await self._store.save_webhook(webhook)

            except asyncio.TimeoutError:
                break
            except Exception as e:
                logger.error(f"Error processing delivery: {e}")

    async def _send_webhook(self, delivery: WebhookDelivery, webhook: WebhookModel) -> bool:
        """
        Send webhook delivery to the target URL.

        Updates delivery status and persists to storage if configured.

        Args:
            delivery: The delivery record to send
            webhook: The webhook configuration

        Returns:
            True if delivery succeeded, False otherwise
        """
        if not self._http_client:
            return False

        try:
            headers = {
                "X-Webhook-Signature": f"sha256={delivery.signature}",
                "X-Webhook-ID": delivery.delivery_id,
                "X-Event-Type": delivery.event_type,
                "Content-Type": "application/json"
            }

            response = await self._http_client.post(
                webhook.url,
                json=delivery.payload,
                headers=headers
            )

            delivery.sent_at = datetime.utcnow()
            delivery.http_status = response.status_code

            if 200 <= response.status_code < 300:
                delivery.status = WebhookStatus.SENT
                webhook.last_triggered_at = datetime.utcnow()
                logger.info(
                    f"Webhook {webhook.webhook_id} delivered ({response.status_code})"
                )

                # Persist successful delivery
                if self._store:
                    await self._store.save_delivery(delivery)

                return True
            else:
                delivery.status = WebhookStatus.FAILED
                delivery.error_message = f"HTTP {response.status_code}"
                logger.warning(
                    f"Webhook delivery failed to {webhook.url}: {response.status_code}"
                )

                # Persist failed delivery
                if self._store:
                    await self._store.save_delivery(delivery)

                return False

        except Exception as e:
            delivery.status = WebhookStatus.FAILED
            delivery.error_message = str(e)
            logger.error(f"Failed to send webhook to {webhook.url}: {e}")

            # Persist failed delivery
            if self._store:
                try:
                    await self._store.save_delivery(delivery)
                except Exception as persist_error:
                    logger.error(f"Failed to persist delivery error: {persist_error}")

            return False

    async def cleanup(self) -> None:
        """
        Cleanup resources.

        Closes HTTP client and storage backend connections.
        """
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        if self._store:
            await self._store.close()

        logger.info("WebhookSubscriberManager cleaned up")


# Backward compatibility alias
WebhookManager = WebhookSubscriberManager
