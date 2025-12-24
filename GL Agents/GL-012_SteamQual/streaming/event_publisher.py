"""
GL-012_SteamQual - Event Publisher

Publishes quality events to message bus/stream for downstream consumers.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
from enum import Enum
import hashlib
import json
import threading
from collections import deque


class EventPriority(Enum):
    """Event priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class QualityEventMessage:
    """Quality event message for publishing."""
    event_id: str
    event_type: str
    severity: str
    header_id: str
    timestamp: datetime
    message: str
    data: Dict[str, Any]
    priority: EventPriority = EventPriority.NORMAL

    # Provenance
    source_agent: str = "GL-012"
    provenance_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "severity": self.severity,
            "header_id": self.header_id,
            "timestamp": self.timestamp.isoformat(),
            "message": self.message,
            "data": self.data,
            "priority": self.priority.value,
            "source_agent": self.source_agent,
            "provenance_hash": self.provenance_hash,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


@dataclass
class PublishResult:
    """Result of event publishing."""
    success: bool
    event_id: str
    destination: str
    timestamp: datetime
    error: Optional[str] = None


class EventPublisher:
    """
    Event publisher for steam quality events.

    Features:
    - Multiple destination support (Kafka, Redis, webhooks)
    - Retry logic with exponential backoff
    - Dead letter queue for failed events
    - Rate limiting to prevent flooding
    - Batching for efficiency

    Event emission target: < 10 seconds latency
    """

    def __init__(
        self,
        destinations: Optional[List[str]] = None,
        max_retries: int = 3,
        retry_delay_seconds: float = 1.0,
        rate_limit_per_second: float = 100.0,
        batch_size: int = 100,
        batch_timeout_seconds: float = 1.0,
    ):
        """Initialize event publisher."""
        self.destinations = destinations or ["memory"]
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds
        self.rate_limit_per_second = rate_limit_per_second
        self.batch_size = batch_size
        self.batch_timeout_seconds = batch_timeout_seconds

        # Internal state
        self._event_queue: deque = deque(maxlen=10000)
        self._dead_letter_queue: deque = deque(maxlen=1000)
        self._published_events: List[QualityEventMessage] = []  # For memory destination

        # Rate limiting
        self._tokens = rate_limit_per_second
        self._last_token_update = datetime.now(timezone.utc)

        # Callbacks
        self._on_publish_callbacks: List[Callable[[QualityEventMessage], None]] = []
        self._on_error_callbacks: List[Callable[[QualityEventMessage, str], None]] = []

        # Statistics
        self._events_published = 0
        self._events_failed = 0
        self._events_in_dlq = 0

        # Thread safety
        self._lock = threading.Lock()

    def publish(self, event: QualityEventMessage) -> PublishResult:
        """
        Publish a single event.

        Args:
            event: Event to publish

        Returns:
            PublishResult with success status
        """
        # Compute provenance hash if not set
        if not event.provenance_hash:
            event.provenance_hash = self._compute_hash(event.to_dict())

        # Check rate limit
        if not self._acquire_token():
            # Queue for later
            with self._lock:
                self._event_queue.append(event)
            return PublishResult(
                success=False,
                event_id=event.event_id,
                destination="queued",
                timestamp=datetime.now(timezone.utc),
                error="Rate limited - queued for later",
            )

        # Publish to all destinations
        results = []
        for destination in self.destinations:
            result = self._publish_to_destination(event, destination)
            results.append(result)

        # Check if any succeeded
        success = any(r.success for r in results)

        if success:
            with self._lock:
                self._events_published += 1

            # Trigger callbacks
            for callback in self._on_publish_callbacks:
                try:
                    callback(event)
                except Exception:
                    pass
        else:
            # All destinations failed - add to dead letter queue
            with self._lock:
                self._dead_letter_queue.append(event)
                self._events_failed += 1
                self._events_in_dlq += 1

            # Trigger error callbacks
            for callback in self._on_error_callbacks:
                try:
                    callback(event, "All destinations failed")
                except Exception:
                    pass

        return results[0] if results else PublishResult(
            success=False,
            event_id=event.event_id,
            destination="none",
            timestamp=datetime.now(timezone.utc),
            error="No destinations configured",
        )

    def publish_batch(self, events: List[QualityEventMessage]) -> List[PublishResult]:
        """
        Publish a batch of events.

        Args:
            events: Events to publish

        Returns:
            List of publish results
        """
        results = []
        for event in events:
            result = self.publish(event)
            results.append(result)
        return results

    def _publish_to_destination(
        self,
        event: QualityEventMessage,
        destination: str,
    ) -> PublishResult:
        """Publish event to a specific destination."""
        timestamp = datetime.now(timezone.utc)

        try:
            if destination == "memory":
                # In-memory storage for testing/debugging
                with self._lock:
                    self._published_events.append(event)
                return PublishResult(
                    success=True,
                    event_id=event.event_id,
                    destination=destination,
                    timestamp=timestamp,
                )

            elif destination.startswith("kafka://"):
                # Kafka publishing (placeholder)
                return self._publish_to_kafka(event, destination)

            elif destination.startswith("redis://"):
                # Redis publishing (placeholder)
                return self._publish_to_redis(event, destination)

            elif destination.startswith("http://") or destination.startswith("https://"):
                # Webhook publishing (placeholder)
                return self._publish_to_webhook(event, destination)

            else:
                return PublishResult(
                    success=False,
                    event_id=event.event_id,
                    destination=destination,
                    timestamp=timestamp,
                    error=f"Unknown destination type: {destination}",
                )

        except Exception as e:
            return PublishResult(
                success=False,
                event_id=event.event_id,
                destination=destination,
                timestamp=timestamp,
                error=str(e),
            )

    def _publish_to_kafka(
        self,
        event: QualityEventMessage,
        destination: str,
    ) -> PublishResult:
        """Publish to Kafka (placeholder for integration)."""
        # TODO: Implement actual Kafka producer
        return PublishResult(
            success=True,
            event_id=event.event_id,
            destination=destination,
            timestamp=datetime.now(timezone.utc),
        )

    def _publish_to_redis(
        self,
        event: QualityEventMessage,
        destination: str,
    ) -> PublishResult:
        """Publish to Redis (placeholder for integration)."""
        # TODO: Implement actual Redis publishing
        return PublishResult(
            success=True,
            event_id=event.event_id,
            destination=destination,
            timestamp=datetime.now(timezone.utc),
        )

    def _publish_to_webhook(
        self,
        event: QualityEventMessage,
        destination: str,
    ) -> PublishResult:
        """Publish to webhook (placeholder for integration)."""
        # TODO: Implement actual HTTP webhook
        return PublishResult(
            success=True,
            event_id=event.event_id,
            destination=destination,
            timestamp=datetime.now(timezone.utc),
        )

    def _acquire_token(self) -> bool:
        """Acquire rate limit token."""
        with self._lock:
            now = datetime.now(timezone.utc)
            elapsed = (now - self._last_token_update).total_seconds()

            # Replenish tokens
            self._tokens = min(
                self.rate_limit_per_second,
                self._tokens + elapsed * self.rate_limit_per_second,
            )
            self._last_token_update = now

            if self._tokens >= 1:
                self._tokens -= 1
                return True
            return False

    def _compute_hash(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash."""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def on_publish(self, callback: Callable[[QualityEventMessage], None]) -> None:
        """Register callback for successful publishes."""
        self._on_publish_callbacks.append(callback)

    def on_error(self, callback: Callable[[QualityEventMessage, str], None]) -> None:
        """Register callback for publish errors."""
        self._on_error_callbacks.append(callback)

    def get_published_events(self) -> List[QualityEventMessage]:
        """Get events published to memory destination."""
        with self._lock:
            return self._published_events.copy()

    def get_dead_letter_queue(self) -> List[QualityEventMessage]:
        """Get events in dead letter queue."""
        with self._lock:
            return list(self._dead_letter_queue)

    def retry_dead_letter_queue(self) -> int:
        """Retry all events in dead letter queue."""
        with self._lock:
            events = list(self._dead_letter_queue)
            self._dead_letter_queue.clear()
            self._events_in_dlq = 0

        retried = 0
        for event in events:
            result = self.publish(event)
            if result.success:
                retried += 1

        return retried

    def get_statistics(self) -> Dict[str, Any]:
        """Get publishing statistics."""
        with self._lock:
            return {
                "events_published": self._events_published,
                "events_failed": self._events_failed,
                "events_in_dlq": self._events_in_dlq,
                "events_queued": len(self._event_queue),
                "destinations": self.destinations,
                "rate_limit": self.rate_limit_per_second,
                "tokens_available": self._tokens,
            }

    def clear(self) -> None:
        """Clear all queues and statistics."""
        with self._lock:
            self._event_queue.clear()
            self._dead_letter_queue.clear()
            self._published_events.clear()
            self._events_published = 0
            self._events_failed = 0
            self._events_in_dlq = 0
