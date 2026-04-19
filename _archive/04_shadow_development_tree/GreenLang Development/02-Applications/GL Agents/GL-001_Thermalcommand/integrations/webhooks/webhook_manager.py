"""
GL-001 ThermalCommand - Webhook Manager

This module provides the core webhook management functionality including:
- Webhook registration and lifecycle management
- Idempotent delivery with retry logic
- HMAC-SHA256 signature generation and verification
- Rate limiting per endpoint
- Dead letter queue for failed deliveries
- Delivery tracking and reporting

The manager coordinates between the configuration registry and the
async dispatcher to ensure reliable event delivery.

Example:
    >>> manager = WebhookManager(config)
    >>> await manager.start()
    >>> result = await manager.deliver_event(event, endpoint_id)
    >>> await manager.shutdown()

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple
import hashlib
import hmac
import json
import logging
import time
import uuid

from pydantic import BaseModel, Field

from .webhook_events import WebhookEvent, WebhookEventType
from .webhook_config import (
    WebhookConfig,
    WebhookEndpoint,
    EndpointRegistry,
    EndpointStatus,
    AuthenticationType,
)


logger = logging.getLogger(__name__)


class DeliveryStatus(str, Enum):
    """Status of a webhook delivery attempt."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DELIVERED = "delivered"
    RETRYING = "retrying"
    FAILED = "failed"
    RATE_LIMITED = "rate_limited"
    CIRCUIT_OPEN = "circuit_open"
    DLQ = "dead_letter_queue"


class DeliveryResult(BaseModel):
    """
    Result of a webhook delivery attempt.

    Attributes:
        delivery_id: Unique delivery identifier
        event_id: Event identifier
        endpoint_id: Target endpoint identifier
        status: Delivery status
        attempt_count: Number of delivery attempts
        http_status_code: HTTP response status code
        response_body: Response body (truncated if large)
        error_message: Error message if failed
        started_at: Delivery start timestamp
        completed_at: Delivery completion timestamp
        duration_ms: Total delivery duration
        signature: HMAC signature used
    """

    delivery_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique delivery identifier"
    )
    event_id: str = Field(..., description="Event identifier")
    endpoint_id: str = Field(..., description="Target endpoint ID")
    status: DeliveryStatus = Field(..., description="Delivery status")
    attempt_count: int = Field(default=1, ge=1, description="Attempt count")
    http_status_code: Optional[int] = Field(
        default=None,
        description="HTTP response status code"
    )
    response_body: Optional[str] = Field(
        default=None,
        max_length=4096,
        description="Response body (truncated)"
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if failed"
    )
    started_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Delivery start time"
    )
    completed_at: Optional[datetime] = Field(
        default=None,
        description="Delivery completion time"
    )
    duration_ms: float = Field(default=0.0, ge=0.0, description="Duration in ms")
    signature: Optional[str] = Field(default=None, description="HMAC signature")
    next_retry_at: Optional[datetime] = Field(
        default=None,
        description="Next retry timestamp if retrying"
    )

    class Config:
        use_enum_values = True


@dataclass
class DeadLetterEntry:
    """Entry in the dead letter queue."""

    entry_id: str
    event: WebhookEvent
    endpoint_id: str
    delivery_result: DeliveryResult
    created_at: datetime
    retry_count: int
    last_error: str
    expires_at: datetime


@dataclass
class RateLimitState:
    """State for rate limiting an endpoint."""

    endpoint_id: str
    window_start: float = field(default_factory=time.time)
    request_count: int = 0
    tokens: float = 0.0
    last_update: float = field(default_factory=time.time)


class SignatureGenerator:
    """
    HMAC-SHA256 signature generator for webhook payloads.

    Generates cryptographically secure signatures for webhook payloads
    to ensure authenticity and integrity of delivered events.

    Example:
        >>> generator = SignatureGenerator()
        >>> signature = generator.generate("secret", "payload", timestamp)
        >>> is_valid = generator.verify("secret", "payload", timestamp, signature)
    """

    ALGORITHM = "sha256"
    VERSION = "v1"

    def generate(
        self,
        secret: str,
        payload: str,
        timestamp: int
    ) -> str:
        """
        Generate HMAC-SHA256 signature.

        The signature is computed over: timestamp.payload
        This prevents replay attacks by including the timestamp.

        Args:
            secret: Shared secret key
            payload: JSON payload string
            timestamp: Unix timestamp

        Returns:
            Signature string in format: v1=<hex_digest>
        """
        message = f"{timestamp}.{payload}"
        signature = hmac.new(
            secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

        return f"{self.VERSION}={signature}"

    def verify(
        self,
        secret: str,
        payload: str,
        timestamp: int,
        signature: str,
        tolerance_seconds: int = 300
    ) -> Tuple[bool, Optional[str]]:
        """
        Verify HMAC-SHA256 signature.

        Args:
            secret: Shared secret key
            payload: JSON payload string
            timestamp: Unix timestamp from request
            signature: Signature from request header
            tolerance_seconds: Maximum age of request in seconds

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check timestamp freshness
        current_time = int(time.time())
        if abs(current_time - timestamp) > tolerance_seconds:
            return False, f"Timestamp too old: {abs(current_time - timestamp)}s"

        # Parse signature version
        if not signature.startswith(f"{self.VERSION}="):
            return False, f"Invalid signature version: expected {self.VERSION}"

        # Generate expected signature
        expected = self.generate(secret, payload, timestamp)

        # Constant-time comparison to prevent timing attacks
        if hmac.compare_digest(expected, signature):
            return True, None

        return False, "Signature mismatch"


class RateLimiter:
    """
    Token bucket rate limiter for webhook endpoints.

    Implements a token bucket algorithm with configurable rate and burst.
    Each endpoint has its own rate limit state.

    Example:
        >>> limiter = RateLimiter()
        >>> can_proceed, wait_time = limiter.check_rate_limit(endpoint)
    """

    def __init__(self):
        """Initialize rate limiter."""
        self._states: Dict[str, RateLimitState] = {}

    def _get_state(self, endpoint: WebhookEndpoint) -> RateLimitState:
        """Get or create rate limit state for endpoint."""
        if endpoint.endpoint_id not in self._states:
            state = RateLimitState(
                endpoint_id=endpoint.endpoint_id,
                tokens=float(endpoint.rate_limit_config.burst_size)
            )
            self._states[endpoint.endpoint_id] = state
        return self._states[endpoint.endpoint_id]

    def check_rate_limit(
        self,
        endpoint: WebhookEndpoint
    ) -> Tuple[bool, float]:
        """
        Check if request is allowed under rate limit.

        Uses token bucket algorithm:
        - Tokens are added at rate of requests_per_second
        - Maximum tokens equals burst_size
        - Each request consumes one token

        Args:
            endpoint: Target endpoint

        Returns:
            Tuple of (can_proceed, wait_time_seconds)
        """
        if not endpoint.rate_limit_config.enabled:
            return True, 0.0

        state = self._get_state(endpoint)
        config = endpoint.rate_limit_config
        current_time = time.time()

        # Add tokens based on elapsed time
        elapsed = current_time - state.last_update
        tokens_to_add = elapsed * config.requests_per_second
        state.tokens = min(
            config.burst_size,
            state.tokens + tokens_to_add
        )
        state.last_update = current_time

        # Check window-based limit
        if current_time - state.window_start >= config.window_seconds:
            state.window_start = current_time
            state.request_count = 0

        if state.request_count >= config.max_requests_per_window:
            wait_time = config.window_seconds - (current_time - state.window_start)
            return False, wait_time

        # Check token bucket
        if state.tokens >= 1.0:
            state.tokens -= 1.0
            state.request_count += 1
            return True, 0.0

        # Calculate wait time for next token
        wait_time = (1.0 - state.tokens) / config.requests_per_second
        return False, wait_time

    def record_request(self, endpoint_id: str) -> None:
        """Record a completed request."""
        state = self._states.get(endpoint_id)
        if state:
            state.request_count += 1

    def reset(self, endpoint_id: str) -> None:
        """Reset rate limit state for an endpoint."""
        self._states.pop(endpoint_id, None)


class DeadLetterQueue:
    """
    Dead letter queue for failed webhook deliveries.

    Stores failed events for later inspection, retry, or purging.
    Supports persistence to disk for durability.

    Example:
        >>> dlq = DeadLetterQueue(config)
        >>> dlq.add(event, endpoint_id, delivery_result)
        >>> entries = dlq.get_entries(endpoint_id)
    """

    def __init__(self, config: WebhookConfig):
        """
        Initialize dead letter queue.

        Args:
            config: Webhook configuration
        """
        self._config = config.dlq_config
        self._entries: Dict[str, DeadLetterEntry] = {}
        self._by_endpoint: Dict[str, Set[str]] = {}
        self._by_event: Dict[str, Set[str]] = {}

    def add(
        self,
        event: WebhookEvent,
        endpoint_id: str,
        delivery_result: DeliveryResult,
        error: str
    ) -> str:
        """
        Add a failed delivery to the queue.

        Args:
            event: Failed event
            endpoint_id: Target endpoint
            delivery_result: Delivery result
            error: Error message

        Returns:
            Entry ID
        """
        if not self._config.enabled:
            return ""

        # Check size limit
        if len(self._entries) >= self._config.max_size:
            self._evict_oldest()

        entry_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(hours=self._config.retention_hours)

        entry = DeadLetterEntry(
            entry_id=entry_id,
            event=event,
            endpoint_id=endpoint_id,
            delivery_result=delivery_result,
            created_at=now,
            retry_count=delivery_result.attempt_count,
            last_error=error,
            expires_at=expires_at
        )

        self._entries[entry_id] = entry

        # Index by endpoint
        if endpoint_id not in self._by_endpoint:
            self._by_endpoint[endpoint_id] = set()
        self._by_endpoint[endpoint_id].add(entry_id)

        # Index by event
        if event.event_id not in self._by_event:
            self._by_event[event.event_id] = set()
        self._by_event[event.event_id].add(entry_id)

        logger.warning(
            f"Added event {event.event_id} to DLQ for endpoint {endpoint_id}: {error}"
        )

        # Check alert threshold
        if len(self._entries) >= self._config.alert_threshold:
            logger.error(
                f"DLQ size ({len(self._entries)}) exceeds alert threshold "
                f"({self._config.alert_threshold})"
            )

        return entry_id

    def get_entry(self, entry_id: str) -> Optional[DeadLetterEntry]:
        """Get an entry by ID."""
        return self._entries.get(entry_id)

    def get_entries_for_endpoint(
        self,
        endpoint_id: str,
        limit: int = 100
    ) -> List[DeadLetterEntry]:
        """Get entries for a specific endpoint."""
        entry_ids = self._by_endpoint.get(endpoint_id, set())
        entries = [
            self._entries[eid]
            for eid in list(entry_ids)[:limit]
            if eid in self._entries
        ]
        return sorted(entries, key=lambda e: e.created_at, reverse=True)

    def remove_entry(self, entry_id: str) -> bool:
        """Remove an entry from the queue."""
        entry = self._entries.pop(entry_id, None)
        if entry is None:
            return False

        # Remove from indices
        self._by_endpoint.get(entry.endpoint_id, set()).discard(entry_id)
        self._by_event.get(entry.event.event_id, set()).discard(entry_id)

        return True

    def _evict_oldest(self) -> None:
        """Evict oldest entries when queue is full."""
        if not self._entries:
            return

        # Sort by creation time and remove oldest 10%
        sorted_entries = sorted(
            self._entries.values(),
            key=lambda e: e.created_at
        )
        num_to_evict = max(1, len(sorted_entries) // 10)

        for entry in sorted_entries[:num_to_evict]:
            self.remove_entry(entry.entry_id)

        logger.info(f"Evicted {num_to_evict} oldest entries from DLQ")

    def cleanup_expired(self) -> int:
        """
        Remove expired entries.

        Returns:
            Number of entries removed
        """
        now = datetime.now(timezone.utc)
        expired_ids = [
            eid for eid, entry in self._entries.items()
            if entry.expires_at <= now
        ]

        for entry_id in expired_ids:
            self.remove_entry(entry_id)

        if expired_ids:
            logger.info(f"Cleaned up {len(expired_ids)} expired DLQ entries")

        return len(expired_ids)

    def size(self) -> int:
        """Get current queue size."""
        return len(self._entries)

    def get_statistics(self) -> Dict[str, Any]:
        """Get queue statistics."""
        by_endpoint_counts = {
            ep: len(ids) for ep, ids in self._by_endpoint.items()
        }
        return {
            "total_entries": len(self._entries),
            "by_endpoint": by_endpoint_counts,
            "max_size": self._config.max_size,
            "retention_hours": self._config.retention_hours,
        }


class IdempotencyTracker:
    """
    Tracks delivery attempts for idempotency.

    Ensures events are not delivered multiple times to the same endpoint
    by tracking delivery IDs and their results.

    Example:
        >>> tracker = IdempotencyTracker()
        >>> if tracker.is_duplicate(event_id, endpoint_id):
        ...     return cached_result
    """

    def __init__(self, ttl_seconds: int = 86400):
        """
        Initialize idempotency tracker.

        Args:
            ttl_seconds: Time-to-live for idempotency records
        """
        self._ttl_seconds = ttl_seconds
        self._records: Dict[str, Tuple[DeliveryResult, float]] = {}

    def _make_key(self, event_id: str, endpoint_id: str) -> str:
        """Create unique key for event-endpoint pair."""
        return f"{event_id}:{endpoint_id}"

    def is_duplicate(self, event_id: str, endpoint_id: str) -> bool:
        """
        Check if event was already delivered to endpoint.

        Args:
            event_id: Event identifier
            endpoint_id: Endpoint identifier

        Returns:
            True if already delivered successfully
        """
        key = self._make_key(event_id, endpoint_id)
        record = self._records.get(key)

        if record is None:
            return False

        result, timestamp = record

        # Check if expired
        if time.time() - timestamp > self._ttl_seconds:
            del self._records[key]
            return False

        # Only consider successful deliveries as duplicates
        return result.status == DeliveryStatus.DELIVERED

    def get_cached_result(
        self,
        event_id: str,
        endpoint_id: str
    ) -> Optional[DeliveryResult]:
        """
        Get cached delivery result.

        Args:
            event_id: Event identifier
            endpoint_id: Endpoint identifier

        Returns:
            Cached result if available
        """
        key = self._make_key(event_id, endpoint_id)
        record = self._records.get(key)

        if record is None:
            return None

        result, timestamp = record

        if time.time() - timestamp > self._ttl_seconds:
            del self._records[key]
            return None

        return result

    def record_delivery(
        self,
        event_id: str,
        endpoint_id: str,
        result: DeliveryResult
    ) -> None:
        """
        Record a delivery attempt.

        Args:
            event_id: Event identifier
            endpoint_id: Endpoint identifier
            result: Delivery result
        """
        key = self._make_key(event_id, endpoint_id)
        self._records[key] = (result, time.time())

    def cleanup_expired(self) -> int:
        """
        Remove expired records.

        Returns:
            Number of records removed
        """
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self._records.items()
            if current_time - timestamp > self._ttl_seconds
        ]

        for key in expired_keys:
            del self._records[key]

        return len(expired_keys)


class WebhookManager:
    """
    Central webhook management system.

    Coordinates all webhook operations including registration, delivery,
    rate limiting, retries, and dead letter queue management.

    Attributes:
        config: Webhook configuration
        registry: Endpoint registry
        signature_generator: HMAC signature generator
        rate_limiter: Rate limiter
        dlq: Dead letter queue
        idempotency_tracker: Idempotency tracker

    Example:
        >>> config = WebhookConfig(...)
        >>> manager = WebhookManager(config)
        >>> await manager.start()
        >>> result = await manager.deliver_event(event)
        >>> await manager.shutdown()
    """

    def __init__(self, config: WebhookConfig):
        """
        Initialize webhook manager.

        Args:
            config: Webhook configuration
        """
        self._config = config
        self._registry = EndpointRegistry(config)
        self._signature_generator = SignatureGenerator()
        self._rate_limiter = RateLimiter()
        self._dlq = DeadLetterQueue(config)
        self._idempotency_tracker = IdempotencyTracker()
        self._running = False
        self._cleanup_task: Optional[asyncio.Task] = None
        self._delivery_stats: Dict[str, int] = {
            "total_delivered": 0,
            "total_failed": 0,
            "total_retried": 0,
            "total_rate_limited": 0,
        }

    @property
    def registry(self) -> EndpointRegistry:
        """Get endpoint registry."""
        return self._registry

    @property
    def dlq(self) -> DeadLetterQueue:
        """Get dead letter queue."""
        return self._dlq

    @property
    def config(self) -> WebhookConfig:
        """Get webhook configuration."""
        return self._config

    async def start(self) -> None:
        """
        Start the webhook manager.

        Starts background tasks for cleanup and monitoring.
        """
        if self._running:
            return

        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Webhook manager started")

    async def shutdown(self) -> None:
        """
        Shutdown the webhook manager.

        Stops background tasks and cleans up resources.
        """
        if not self._running:
            return

        self._running = False

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        logger.info("Webhook manager shutdown complete")

    async def _cleanup_loop(self) -> None:
        """Background task for periodic cleanup."""
        while self._running:
            try:
                await asyncio.sleep(3600)  # Run every hour
                self._dlq.cleanup_expired()
                self._idempotency_tracker.cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")

    def generate_signature(
        self,
        endpoint: WebhookEndpoint,
        payload: str,
        timestamp: int
    ) -> Optional[str]:
        """
        Generate signature for webhook payload.

        Args:
            endpoint: Target endpoint
            payload: JSON payload string
            timestamp: Unix timestamp

        Returns:
            Signature string or None if not using HMAC auth
        """
        if endpoint.authentication_type != AuthenticationType.HMAC_SHA256:
            return None

        secret = endpoint.get_secret_value()
        if not secret:
            logger.error(f"No secret configured for endpoint {endpoint.endpoint_id}")
            return None

        return self._signature_generator.generate(secret, payload, timestamp)

    def verify_signature(
        self,
        endpoint: WebhookEndpoint,
        payload: str,
        timestamp: int,
        signature: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Verify signature from incoming webhook request.

        Args:
            endpoint: Endpoint configuration
            payload: JSON payload string
            timestamp: Unix timestamp from request
            signature: Signature from request header

        Returns:
            Tuple of (is_valid, error_message)
        """
        secret = endpoint.get_secret_value()
        if not secret:
            return False, "No secret configured"

        return self._signature_generator.verify(
            secret,
            payload,
            timestamp,
            signature,
            self._config.signature_tolerance_seconds
        )

    def check_rate_limit(
        self,
        endpoint: WebhookEndpoint
    ) -> Tuple[bool, float]:
        """
        Check if delivery is allowed under rate limit.

        Args:
            endpoint: Target endpoint

        Returns:
            Tuple of (can_proceed, wait_time_seconds)
        """
        return self._rate_limiter.check_rate_limit(endpoint)

    def prepare_delivery(
        self,
        event: WebhookEvent,
        endpoint: WebhookEndpoint
    ) -> Tuple[Dict[str, str], str, str]:
        """
        Prepare headers and payload for delivery.

        Args:
            event: Event to deliver
            endpoint: Target endpoint

        Returns:
            Tuple of (headers, payload_json, delivery_id)
        """
        delivery_id = str(uuid.uuid4())
        timestamp = int(time.time())

        # Ensure provenance hash is set
        if event.provenance_hash is None:
            event = event.with_provenance()

        payload = event.to_json()

        # Build headers
        headers = {
            "Content-Type": self._config.content_type,
            "User-Agent": self._config.user_agent,
            self._config.timestamp_header: str(timestamp),
            self._config.idempotency_header: event.event_id,
            self._config.event_type_header: str(event.event_type),
            self._config.delivery_id_header: delivery_id,
        }

        # Add custom headers from endpoint
        headers.update(endpoint.headers)

        # Add authentication
        if endpoint.authentication_type == AuthenticationType.HMAC_SHA256:
            signature = self.generate_signature(endpoint, payload, timestamp)
            if signature:
                headers[self._config.signature_header] = signature

        elif endpoint.authentication_type == AuthenticationType.BEARER_TOKEN:
            if endpoint.bearer_token:
                headers["Authorization"] = f"Bearer {endpoint.bearer_token.get_secret_value()}"

        elif endpoint.authentication_type == AuthenticationType.API_KEY:
            if endpoint.api_key:
                headers[endpoint.api_key_header] = endpoint.api_key.get_secret_value()

        return headers, payload, delivery_id

    def is_duplicate(self, event: WebhookEvent, endpoint: WebhookEndpoint) -> bool:
        """
        Check if event was already delivered to endpoint.

        Args:
            event: Event to check
            endpoint: Target endpoint

        Returns:
            True if already successfully delivered
        """
        return self._idempotency_tracker.is_duplicate(
            event.event_id,
            endpoint.endpoint_id
        )

    def get_cached_result(
        self,
        event: WebhookEvent,
        endpoint: WebhookEndpoint
    ) -> Optional[DeliveryResult]:
        """
        Get cached delivery result for idempotency.

        Args:
            event: Event
            endpoint: Endpoint

        Returns:
            Cached result if available
        """
        return self._idempotency_tracker.get_cached_result(
            event.event_id,
            endpoint.endpoint_id
        )

    def record_delivery_result(
        self,
        event: WebhookEvent,
        endpoint: WebhookEndpoint,
        result: DeliveryResult
    ) -> None:
        """
        Record delivery result for tracking and idempotency.

        Args:
            event: Delivered event
            endpoint: Target endpoint
            result: Delivery result
        """
        self._idempotency_tracker.record_delivery(
            event.event_id,
            endpoint.endpoint_id,
            result
        )

        # Update statistics
        if result.status == DeliveryStatus.DELIVERED:
            self._delivery_stats["total_delivered"] += 1
        elif result.status == DeliveryStatus.FAILED:
            self._delivery_stats["total_failed"] += 1
        elif result.status == DeliveryStatus.RETRYING:
            self._delivery_stats["total_retried"] += 1
        elif result.status == DeliveryStatus.RATE_LIMITED:
            self._delivery_stats["total_rate_limited"] += 1

    def add_to_dlq(
        self,
        event: WebhookEvent,
        endpoint: WebhookEndpoint,
        result: DeliveryResult,
        error: str
    ) -> str:
        """
        Add failed delivery to dead letter queue.

        Args:
            event: Failed event
            endpoint: Target endpoint
            result: Delivery result
            error: Error message

        Returns:
            DLQ entry ID
        """
        return self._dlq.add(event, endpoint.endpoint_id, result, error)

    def get_endpoints_for_event(
        self,
        event_type: WebhookEventType
    ) -> List[WebhookEndpoint]:
        """
        Get all active endpoints for an event type.

        Args:
            event_type: Event type

        Returns:
            List of endpoints
        """
        return self._registry.get_endpoints_for_event(event_type, active_only=True)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get webhook system statistics.

        Returns:
            Dictionary of statistics
        """
        return {
            "delivery_stats": self._delivery_stats.copy(),
            "endpoint_count": self._registry.endpoint_count(),
            "active_endpoint_count": self._registry.active_endpoint_count(),
            "dlq_stats": self._dlq.get_statistics(),
            "running": self._running,
        }

    def update_endpoint_status(
        self,
        endpoint_id: str,
        status: EndpointStatus
    ) -> bool:
        """
        Update endpoint status based on delivery results.

        Args:
            endpoint_id: Endpoint ID
            status: New status

        Returns:
            True if updated
        """
        return self._registry.update_endpoint_status(endpoint_id, status)
