"""
Webhook Endpoint Framework for GreenLang

This module provides webhook endpoint management for
receiving and processing external system notifications.

Features:
- Webhook registration and validation
- Signature verification (HMAC-SHA256)
- Retry handling
- Event routing
- Dead letter queue
- Rate limiting

Example:
    >>> manager = WebhookManager(config)
    >>> manager.register_endpoint("/webhooks/github", github_handler)
    >>> await manager.start()
"""

import asyncio
import hashlib
import hmac
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

try:
    from fastapi import APIRouter, HTTPException, Request, Response
    from fastapi.responses import JSONResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    APIRouter = object
    HTTPException = Exception
    Request = None
    Response = None

logger = logging.getLogger(__name__)


class WebhookStatus(str, Enum):
    """Webhook delivery status."""
    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"


class SignatureAlgorithm(str, Enum):
    """Signature verification algorithms."""
    HMAC_SHA256 = "hmac-sha256"
    HMAC_SHA512 = "hmac-sha512"
    RSA_SHA256 = "rsa-sha256"


@dataclass
class WebhookManagerConfig:
    """Configuration for webhook manager."""
    prefix: str = "/webhooks"
    default_timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay_seconds: int = 60
    enable_signature_verification: bool = True
    signature_header: str = "X-Webhook-Signature"
    timestamp_header: str = "X-Webhook-Timestamp"
    timestamp_tolerance_seconds: int = 300
    enable_idempotency: bool = True
    idempotency_header: str = "X-Webhook-ID"
    enable_rate_limiting: bool = True
    rate_limit_per_minute: int = 100


class WebhookEndpoint(BaseModel):
    """Webhook endpoint definition."""
    endpoint_id: str = Field(default_factory=lambda: str(uuid4()))
    path: str = Field(..., description="Endpoint path")
    source: str = Field(..., description="Source system")
    secret: Optional[str] = Field(default=None, description="Signing secret")
    signature_algorithm: SignatureAlgorithm = Field(
        default=SignatureAlgorithm.HMAC_SHA256
    )
    event_types: List[str] = Field(default_factory=list)
    is_active: bool = Field(default=True)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WebhookDelivery(BaseModel):
    """Webhook delivery record."""
    delivery_id: str = Field(default_factory=lambda: str(uuid4()))
    endpoint_id: str = Field(..., description="Endpoint ID")
    event_type: Optional[str] = Field(default=None)
    payload: Dict[str, Any] = Field(..., description="Webhook payload")
    headers: Dict[str, str] = Field(default_factory=dict)
    status: WebhookStatus = Field(default=WebhookStatus.PENDING)
    attempts: int = Field(default=0)
    last_attempt_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)
    response_status: Optional[int] = Field(default=None)
    response_body: Optional[str] = Field(default=None)
    error_message: Optional[str] = Field(default=None)
    received_at: datetime = Field(default_factory=datetime.utcnow)
    provenance_hash: str = Field(default="")


class WebhookResponse(BaseModel):
    """Standard webhook response."""
    success: bool = Field(..., description="Processing success")
    message: str = Field(default="OK", description="Response message")
    delivery_id: str = Field(..., description="Delivery ID")


class WebhookHandler:
    """
    Base class for webhook handlers.

    Implement custom handlers by extending this class
    and overriding the handle method.
    """

    async def validate(
        self,
        request: Request,
        payload: Dict[str, Any]
    ) -> bool:
        """
        Validate the webhook request.

        Args:
            request: FastAPI request
            payload: Parsed payload

        Returns:
            True if valid
        """
        return True

    async def handle(
        self,
        event_type: str,
        payload: Dict[str, Any],
        headers: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Handle the webhook event.

        Args:
            event_type: Event type
            payload: Event payload
            headers: Request headers

        Returns:
            Handler result
        """
        raise NotImplementedError("Subclasses must implement handle()")


class WebhookManager:
    """
    Webhook endpoint manager.

    Manages webhook registration, signature verification,
    and event routing.

    Attributes:
        config: Manager configuration
        router: FastAPI router

    Example:
        >>> config = WebhookManagerConfig()
        >>> manager = WebhookManager(config)
        >>> manager.register_endpoint(
        ...     path="/github",
        ...     source="github",
        ...     handler=GitHubHandler(),
        ...     secret="webhook-secret"
        ... )
        >>> app.include_router(manager.router)
    """

    def __init__(self, config: Optional[WebhookManagerConfig] = None):
        """
        Initialize webhook manager.

        Args:
            config: Manager configuration
        """
        if not FASTAPI_AVAILABLE:
            raise ImportError(
                "FastAPI is required for webhook support. "
                "Install with: pip install fastapi"
            )

        self.config = config or WebhookManagerConfig()
        self.router = APIRouter(prefix=self.config.prefix)
        self._endpoints: Dict[str, WebhookEndpoint] = {}
        self._handlers: Dict[str, WebhookHandler] = {}
        self._deliveries: Dict[str, WebhookDelivery] = {}
        self._processed_ids: Dict[str, datetime] = {}
        self._rate_limits: Dict[str, List[datetime]] = {}

        logger.info("WebhookManager initialized")

    def register_endpoint(
        self,
        path: str,
        source: str,
        handler: WebhookHandler,
        secret: Optional[str] = None,
        event_types: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> WebhookEndpoint:
        """
        Register a webhook endpoint.

        Args:
            path: Endpoint path
            source: Source system name
            handler: Webhook handler
            secret: Signing secret for verification
            event_types: Supported event types
            metadata: Additional metadata

        Returns:
            Registered endpoint
        """
        endpoint = WebhookEndpoint(
            path=path,
            source=source,
            secret=secret,
            event_types=event_types or [],
            metadata=metadata or {},
        )

        self._endpoints[endpoint.endpoint_id] = endpoint
        self._handlers[endpoint.endpoint_id] = handler

        # Register route
        self._create_route(endpoint)

        logger.info(f"Registered webhook endpoint: {path} for {source}")
        return endpoint

    def _create_route(self, endpoint: WebhookEndpoint) -> None:
        """Create FastAPI route for endpoint."""
        @self.router.post(
            endpoint.path,
            response_model=WebhookResponse,
            summary=f"Webhook endpoint for {endpoint.source}"
        )
        async def webhook_handler(request: Request) -> WebhookResponse:
            return await self._process_webhook(request, endpoint.endpoint_id)

    async def _process_webhook(
        self,
        request: Request,
        endpoint_id: str
    ) -> WebhookResponse:
        """Process incoming webhook."""
        endpoint = self._endpoints.get(endpoint_id)
        if not endpoint or not endpoint.is_active:
            raise HTTPException(status_code=404, detail="Endpoint not found")

        handler = self._handlers.get(endpoint_id)
        if not handler:
            raise HTTPException(status_code=500, detail="Handler not configured")

        # Get headers
        headers = dict(request.headers)

        # Check rate limit
        if self.config.enable_rate_limiting:
            if not self._check_rate_limit(endpoint_id):
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded"
                )

        # Check idempotency
        if self.config.enable_idempotency:
            webhook_id = headers.get(self.config.idempotency_header.lower())
            if webhook_id and self._is_duplicate(webhook_id):
                return WebhookResponse(
                    success=True,
                    message="Already processed",
                    delivery_id=webhook_id
                )

        # Read body
        body = await request.body()
        payload = json.loads(body.decode())

        # Verify signature
        if self.config.enable_signature_verification and endpoint.secret:
            if not self._verify_signature(
                body,
                headers.get(self.config.signature_header.lower(), ""),
                endpoint.secret,
                endpoint.signature_algorithm
            ):
                raise HTTPException(
                    status_code=401,
                    detail="Invalid signature"
                )

        # Check timestamp
        timestamp_str = headers.get(self.config.timestamp_header.lower())
        if timestamp_str:
            if not self._check_timestamp(timestamp_str):
                raise HTTPException(
                    status_code=400,
                    detail="Request too old"
                )

        # Create delivery record
        provenance_str = f"{endpoint_id}:{json.dumps(payload)}:{datetime.utcnow().isoformat()}"
        provenance_hash = hashlib.sha256(provenance_str.encode()).hexdigest()

        delivery = WebhookDelivery(
            endpoint_id=endpoint_id,
            event_type=payload.get("event_type") or payload.get("type"),
            payload=payload,
            headers=headers,
            provenance_hash=provenance_hash,
        )

        self._deliveries[delivery.delivery_id] = delivery

        # Validate
        try:
            if not await handler.validate(request, payload):
                delivery.status = WebhookStatus.FAILED
                delivery.error_message = "Validation failed"
                raise HTTPException(status_code=400, detail="Validation failed")

        except HTTPException:
            raise
        except Exception as e:
            delivery.status = WebhookStatus.FAILED
            delivery.error_message = str(e)
            raise HTTPException(status_code=400, detail=str(e))

        # Process
        try:
            delivery.status = WebhookStatus.PENDING
            delivery.attempts += 1
            delivery.last_attempt_at = datetime.utcnow()

            result = await handler.handle(
                delivery.event_type or "",
                payload,
                headers
            )

            delivery.status = WebhookStatus.DELIVERED
            delivery.completed_at = datetime.utcnow()
            delivery.response_status = 200
            delivery.response_body = json.dumps(result)

            # Mark as processed for idempotency
            webhook_id = headers.get(self.config.idempotency_header.lower())
            if webhook_id:
                self._processed_ids[webhook_id] = datetime.utcnow()

            logger.info(f"Webhook processed: {delivery.delivery_id}")

            return WebhookResponse(
                success=True,
                message="Processed",
                delivery_id=delivery.delivery_id
            )

        except Exception as e:
            delivery.status = WebhookStatus.FAILED
            delivery.error_message = str(e)

            logger.error(f"Webhook processing failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def _verify_signature(
        self,
        body: bytes,
        signature: str,
        secret: str,
        algorithm: SignatureAlgorithm
    ) -> bool:
        """Verify webhook signature."""
        if algorithm == SignatureAlgorithm.HMAC_SHA256:
            expected = hmac.new(
                secret.encode(),
                body,
                hashlib.sha256
            ).hexdigest()

            # Handle different signature formats
            if signature.startswith("sha256="):
                signature = signature[7:]

            return hmac.compare_digest(expected, signature)

        elif algorithm == SignatureAlgorithm.HMAC_SHA512:
            expected = hmac.new(
                secret.encode(),
                body,
                hashlib.sha512
            ).hexdigest()

            if signature.startswith("sha512="):
                signature = signature[7:]

            return hmac.compare_digest(expected, signature)

        return False

    def _check_timestamp(self, timestamp_str: str) -> bool:
        """Check if timestamp is within tolerance."""
        try:
            # Try parsing as Unix timestamp
            if timestamp_str.isdigit():
                timestamp = datetime.utcfromtimestamp(int(timestamp_str))
            else:
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))

            age = (datetime.utcnow() - timestamp).total_seconds()
            return abs(age) <= self.config.timestamp_tolerance_seconds

        except Exception:
            return False

    def _is_duplicate(self, webhook_id: str) -> bool:
        """Check if webhook was already processed."""
        if webhook_id in self._processed_ids:
            processed_at = self._processed_ids[webhook_id]
            # Consider duplicates within 24 hours
            if (datetime.utcnow() - processed_at).total_seconds() < 86400:
                return True
        return False

    def _check_rate_limit(self, endpoint_id: str) -> bool:
        """Check rate limit for endpoint."""
        now = datetime.utcnow()
        minute_ago = now - timedelta(minutes=1)

        if endpoint_id not in self._rate_limits:
            self._rate_limits[endpoint_id] = []

        # Clean old entries
        self._rate_limits[endpoint_id] = [
            t for t in self._rate_limits[endpoint_id]
            if t > minute_ago
        ]

        if len(self._rate_limits[endpoint_id]) >= self.config.rate_limit_per_minute:
            return False

        self._rate_limits[endpoint_id].append(now)
        return True

    def get_endpoint(self, endpoint_id: str) -> Optional[WebhookEndpoint]:
        """Get endpoint by ID."""
        return self._endpoints.get(endpoint_id)

    def get_delivery(self, delivery_id: str) -> Optional[WebhookDelivery]:
        """Get delivery by ID."""
        return self._deliveries.get(delivery_id)

    def list_endpoints(self) -> List[WebhookEndpoint]:
        """List all registered endpoints."""
        return list(self._endpoints.values())

    def list_deliveries(
        self,
        endpoint_id: Optional[str] = None,
        status: Optional[WebhookStatus] = None,
        limit: int = 100
    ) -> List[WebhookDelivery]:
        """
        List deliveries with optional filtering.

        Args:
            endpoint_id: Filter by endpoint
            status: Filter by status
            limit: Maximum results

        Returns:
            List of deliveries
        """
        deliveries = list(self._deliveries.values())

        if endpoint_id:
            deliveries = [d for d in deliveries if d.endpoint_id == endpoint_id]

        if status:
            deliveries = [d for d in deliveries if d.status == status]

        # Sort by received_at descending
        deliveries.sort(key=lambda d: d.received_at, reverse=True)

        return deliveries[:limit]

    def deactivate_endpoint(self, endpoint_id: str) -> bool:
        """
        Deactivate an endpoint.

        Args:
            endpoint_id: Endpoint to deactivate

        Returns:
            True if deactivated
        """
        if endpoint_id in self._endpoints:
            self._endpoints[endpoint_id].is_active = False
            logger.info(f"Deactivated webhook endpoint: {endpoint_id}")
            return True
        return False

    def cleanup_old_deliveries(self, days: int = 7) -> int:
        """
        Clean up old delivery records.

        Args:
            days: Delete records older than this

        Returns:
            Number of records deleted
        """
        cutoff = datetime.utcnow() - timedelta(days=days)
        old_ids = [
            did for did, delivery in self._deliveries.items()
            if delivery.received_at < cutoff
        ]

        for did in old_ids:
            del self._deliveries[did]

        logger.info(f"Cleaned up {len(old_ids)} old delivery records")
        return len(old_ids)
