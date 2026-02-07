# -*- coding: utf-8 -*-
"""
Audit Event Collector - SEC-005: Centralized Audit Logging Service

Provides async queue-based event collection with backpressure handling.
Events are buffered in an asyncio.Queue and processed by background workers
for efficient, non-blocking audit logging.

**Design Principles:**
- Non-blocking event collection (fire-and-forget)
- Backpressure when queue exceeds threshold (10,000 events)
- Queue depth metrics for monitoring
- Graceful shutdown with drain support

Author: GreenLang Framework Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, List, Optional

from .event_model import UnifiedAuditEvent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Collector Configuration
# ---------------------------------------------------------------------------


@dataclass
class CollectorConfig:
    """Configuration for the AuditEventCollector.

    Attributes:
        max_queue_size: Maximum queue depth before backpressure (default 10000).
        backpressure_threshold: Queue depth that triggers warnings (default 8000).
        batch_size: Number of events to collect before flushing (default 100).
        flush_interval_seconds: Max time between flushes (default 5.0).
        enable_metrics: Whether to emit queue depth metrics (default True).
    """

    max_queue_size: int = 10000
    backpressure_threshold: int = 8000
    batch_size: int = 100
    flush_interval_seconds: float = 5.0
    enable_metrics: bool = True


# ---------------------------------------------------------------------------
# Queue Metrics
# ---------------------------------------------------------------------------


@dataclass
class CollectorMetrics:
    """Metrics for monitoring the event collector.

    Attributes:
        events_collected: Total events collected.
        events_dropped: Events dropped due to backpressure.
        batches_flushed: Number of batches flushed.
        current_queue_depth: Current number of events in queue.
        peak_queue_depth: Maximum queue depth observed.
        backpressure_events: Number of times backpressure was applied.
        last_flush_time: Timestamp of last flush.
    """

    events_collected: int = 0
    events_dropped: int = 0
    batches_flushed: int = 0
    current_queue_depth: int = 0
    peak_queue_depth: int = 0
    backpressure_events: int = 0
    last_flush_time: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Audit Event Collector
# ---------------------------------------------------------------------------


class AuditEventCollector:
    """Async queue-based audit event collector with backpressure.

    Collects events into an asyncio.Queue and processes them in batches
    via background workers. Supports both single-event and batch collection.

    Example:
        >>> collector = AuditEventCollector()
        >>> await collector.start()
        >>> await collector.collect(event)
        >>> await collector.collect_batch([event1, event2])
        >>> await collector.stop()
    """

    def __init__(
        self,
        config: Optional[CollectorConfig] = None,
        on_batch_ready: Optional[
            Callable[[List[UnifiedAuditEvent]], Coroutine[Any, Any, None]]
        ] = None,
    ) -> None:
        """Initialize the event collector.

        Args:
            config: Collector configuration.
            on_batch_ready: Async callback invoked when a batch is ready.
        """
        self._config = config or CollectorConfig()
        self._on_batch_ready = on_batch_ready
        self._queue: asyncio.Queue[UnifiedAuditEvent] = asyncio.Queue(
            maxsize=self._config.max_queue_size
        )
        self._metrics = CollectorMetrics()
        self._running = False
        self._worker_task: Optional[asyncio.Task[None]] = None
        self._lock = asyncio.Lock()

    @property
    def metrics(self) -> CollectorMetrics:
        """Get current collector metrics.

        Returns:
            Current metrics snapshot.
        """
        self._metrics.current_queue_depth = self._queue.qsize()
        return self._metrics

    @property
    def is_running(self) -> bool:
        """Check if the collector is running.

        Returns:
            True if the collector is running.
        """
        return self._running

    async def start(self) -> None:
        """Start the background worker for batch processing.

        Spawns an asyncio task that collects events and flushes batches.
        """
        if self._running:
            logger.warning("AuditEventCollector is already running")
            return

        self._running = True
        self._worker_task = asyncio.create_task(
            self._worker_loop(), name="audit_collector_worker"
        )
        logger.info(
            "AuditEventCollector started with max_queue_size=%d, batch_size=%d",
            self._config.max_queue_size,
            self._config.batch_size,
        )

    async def stop(self, drain: bool = True, timeout: float = 10.0) -> None:
        """Stop the collector and optionally drain remaining events.

        Args:
            drain: Whether to flush remaining events before stopping.
            timeout: Maximum time to wait for drain (seconds).
        """
        if not self._running:
            return

        self._running = False

        if drain and not self._queue.empty():
            logger.info(
                "Draining %d remaining events from queue",
                self._queue.qsize(),
            )
            try:
                await asyncio.wait_for(self._drain_queue(), timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning(
                    "Drain timeout after %.1fs, %d events remaining",
                    timeout,
                    self._queue.qsize(),
                )

        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

        logger.info(
            "AuditEventCollector stopped. Total collected: %d, dropped: %d",
            self._metrics.events_collected,
            self._metrics.events_dropped,
        )

    async def collect(self, event: UnifiedAuditEvent) -> bool:
        """Collect a single audit event.

        Non-blocking if queue has capacity. Applies backpressure if queue
        exceeds threshold.

        Args:
            event: The audit event to collect.

        Returns:
            True if event was collected, False if dropped due to backpressure.
        """
        queue_size = self._queue.qsize()

        # Update peak queue depth
        if queue_size > self._metrics.peak_queue_depth:
            self._metrics.peak_queue_depth = queue_size

        # Check backpressure threshold
        if queue_size >= self._config.backpressure_threshold:
            self._metrics.backpressure_events += 1
            if queue_size >= self._config.max_queue_size:
                # Queue is full, drop the event
                self._metrics.events_dropped += 1
                logger.warning(
                    "Audit event dropped due to queue overflow: event_id=%s, "
                    "queue_size=%d, max=%d",
                    event.event_id,
                    queue_size,
                    self._config.max_queue_size,
                )
                return False
            else:
                logger.warning(
                    "Audit queue backpressure: queue_size=%d, threshold=%d",
                    queue_size,
                    self._config.backpressure_threshold,
                )

        try:
            self._queue.put_nowait(event)
            self._metrics.events_collected += 1
            return True
        except asyncio.QueueFull:
            self._metrics.events_dropped += 1
            logger.warning(
                "Audit event dropped due to QueueFull: event_id=%s",
                event.event_id,
            )
            return False

    async def collect_batch(self, events: List[UnifiedAuditEvent]) -> int:
        """Collect multiple audit events.

        Attempts to collect all events, dropping those that exceed capacity.

        Args:
            events: List of audit events to collect.

        Returns:
            Number of events successfully collected.
        """
        collected = 0
        for event in events:
            if await self.collect(event):
                collected += 1
        return collected

    async def _worker_loop(self) -> None:
        """Background worker that batches and flushes events."""
        batch: List[UnifiedAuditEvent] = []
        last_flush = time.time()

        while self._running:
            try:
                # Wait for events with timeout
                timeout = self._config.flush_interval_seconds - (
                    time.time() - last_flush
                )
                timeout = max(0.1, timeout)

                try:
                    event = await asyncio.wait_for(
                        self._queue.get(), timeout=timeout
                    )
                    batch.append(event)
                except asyncio.TimeoutError:
                    pass

                # Flush if batch is full or interval elapsed
                should_flush = (
                    len(batch) >= self._config.batch_size
                    or (
                        batch
                        and time.time() - last_flush
                        >= self._config.flush_interval_seconds
                    )
                )

                if should_flush and batch:
                    await self._flush_batch(batch)
                    batch = []
                    last_flush = time.time()

            except asyncio.CancelledError:
                # Flush remaining batch on cancellation
                if batch:
                    await self._flush_batch(batch)
                raise
            except Exception:
                logger.exception("Error in audit collector worker loop")
                await asyncio.sleep(1.0)  # Brief pause on error

    async def _flush_batch(self, batch: List[UnifiedAuditEvent]) -> None:
        """Flush a batch of events via callback.

        Args:
            batch: List of events to flush.
        """
        if not batch:
            return

        self._metrics.batches_flushed += 1
        self._metrics.last_flush_time = time.time()

        if self._on_batch_ready:
            try:
                await self._on_batch_ready(batch)
                logger.debug(
                    "Flushed batch of %d audit events", len(batch)
                )
            except Exception:
                logger.exception(
                    "Error flushing audit event batch of %d events",
                    len(batch),
                )
        else:
            logger.debug(
                "No batch callback configured, discarding %d events",
                len(batch),
            )

    async def _drain_queue(self) -> None:
        """Drain all remaining events from the queue."""
        batch: List[UnifiedAuditEvent] = []

        while not self._queue.empty():
            try:
                event = self._queue.get_nowait()
                batch.append(event)

                if len(batch) >= self._config.batch_size:
                    await self._flush_batch(batch)
                    batch = []
            except asyncio.QueueEmpty:
                break

        # Flush any remaining events
        if batch:
            await self._flush_batch(batch)

    def get_queue_depth(self) -> int:
        """Get current queue depth.

        Returns:
            Number of events currently in queue.
        """
        return self._queue.qsize()

    def is_backpressure_active(self) -> bool:
        """Check if backpressure is currently active.

        Returns:
            True if queue depth exceeds backpressure threshold.
        """
        return self._queue.qsize() >= self._config.backpressure_threshold


__all__ = [
    "AuditEventCollector",
    "CollectorConfig",
    "CollectorMetrics",
]
