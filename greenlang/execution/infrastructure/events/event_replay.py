"""
Event Replay Mechanism for GreenLang
=====================================

TASK-125: Event Replay Implementation

This module provides event replay capabilities for event-sourced systems,
enabling reconstruction of state, debugging, and event reprocessing.

Features:
- Event store replay from timestamp
- Event ID-based replay
- Filtered replay by event type
- Replay speed control (1x, 10x, real-time)
- Idempotency handling for replayed events
- Replay progress tracking
- Integration with event_sourcing.py

Example:
    >>> from greenlang.infrastructure.events import EventReplayManager, ReplayConfig
    >>> config = ReplayConfig(speed_multiplier=10.0)
    >>> replay_manager = EventReplayManager(event_store, config)
    >>> async for event in replay_manager.replay_from_timestamp(start_time):
    ...     await process_event(event)

Author: GreenLang Infrastructure Team
Created: 2025-12-07
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)
from uuid import uuid4

from pydantic import BaseModel, Field, validator

from greenlang.infrastructure.events.event_schema import BaseEvent, EventMetadata
from greenlang.infrastructure.events.event_sourcing import (
    EventStore,
    StoredEvent,
    StreamInfo,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================


class ReplaySpeed(str, Enum):
    """Predefined replay speed options."""
    REAL_TIME = "real_time"  # 1x - Original timing
    FAST = "fast"            # 10x - 10 times faster
    FASTER = "faster"        # 100x - 100 times faster
    INSTANT = "instant"      # No delay between events
    CUSTOM = "custom"        # Use speed_multiplier


class ReplayMode(str, Enum):
    """Replay mode options."""
    SEQUENTIAL = "sequential"      # Replay events in order
    PARALLEL = "parallel"          # Replay events in parallel batches
    STREAM_BASED = "stream_based"  # Replay by stream


class ReplayStatus(str, Enum):
    """Replay job status."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class IdempotencyStrategy(str, Enum):
    """Strategy for handling idempotency."""
    SKIP_PROCESSED = "skip_processed"    # Skip already processed events
    REPROCESS_ALL = "reprocess_all"      # Reprocess all events
    CHECK_HANDLER = "check_handler"      # Let handler decide


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class ReplayConfig:
    """Configuration for event replay."""
    # Speed control
    speed: ReplaySpeed = ReplaySpeed.FAST
    speed_multiplier: float = 10.0  # Used when speed is CUSTOM

    # Batch settings
    batch_size: int = 100
    max_concurrent_events: int = 10

    # Idempotency
    idempotency_strategy: IdempotencyStrategy = IdempotencyStrategy.SKIP_PROCESSED
    idempotency_store: str = "memory"  # memory, redis, postgres

    # Filtering
    event_types: Optional[List[str]] = None  # Filter by event types
    stream_ids: Optional[List[str]] = None   # Filter by stream IDs

    # Error handling
    stop_on_error: bool = False
    max_errors: int = 100
    error_delay_seconds: float = 1.0

    # Progress tracking
    checkpoint_interval: int = 100  # Events between checkpoints
    enable_progress_events: bool = True

    # Audit
    enable_audit_trail: bool = True
    audit_log_path: str = "/var/log/greenlang/event-replay.log"


# =============================================================================
# Models
# =============================================================================


class ReplayFilter(BaseModel):
    """Filter criteria for event replay."""
    event_types: Optional[List[str]] = Field(
        default=None,
        description="Event types to include"
    )
    exclude_event_types: Optional[List[str]] = Field(
        default=None,
        description="Event types to exclude"
    )
    stream_ids: Optional[List[str]] = Field(
        default=None,
        description="Stream IDs to include"
    )
    start_timestamp: Optional[datetime] = Field(
        default=None,
        description="Start timestamp (inclusive)"
    )
    end_timestamp: Optional[datetime] = Field(
        default=None,
        description="End timestamp (inclusive)"
    )
    start_event_id: Optional[str] = Field(
        default=None,
        description="Start from specific event ID"
    )
    start_version: Optional[int] = Field(
        default=None,
        description="Start from specific version"
    )
    metadata_filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Filter by metadata fields"
    )

    def matches(self, event: StoredEvent) -> bool:
        """Check if an event matches this filter."""
        # Event type filter
        if self.event_types and event.event_type not in self.event_types:
            return False

        if self.exclude_event_types and event.event_type in self.exclude_event_types:
            return False

        # Stream ID filter
        if self.stream_ids and event.stream_id not in self.stream_ids:
            return False

        # Timestamp filters
        if self.start_timestamp and event.timestamp < self.start_timestamp:
            return False

        if self.end_timestamp and event.timestamp > self.end_timestamp:
            return False

        # Metadata filters
        if self.metadata_filters:
            for key, value in self.metadata_filters.items():
                if event.metadata.get(key) != value:
                    return False

        return True


class ReplayProgress(BaseModel):
    """Progress information for a replay job."""
    job_id: str = Field(..., description="Replay job identifier")
    status: ReplayStatus = Field(default=ReplayStatus.PENDING)
    total_events: int = Field(default=0, description="Total events to replay")
    processed_events: int = Field(default=0, description="Events processed")
    skipped_events: int = Field(default=0, description="Events skipped (idempotency)")
    failed_events: int = Field(default=0, description="Events that failed")
    current_position: int = Field(default=0, description="Current position in stream")
    current_timestamp: Optional[datetime] = Field(default=None)
    current_event_id: Optional[str] = Field(default=None)
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)
    elapsed_seconds: float = Field(default=0.0)
    events_per_second: float = Field(default=0.0)
    estimated_remaining_seconds: float = Field(default=0.0)
    last_checkpoint: Optional[datetime] = Field(default=None)
    last_error: Optional[str] = Field(default=None)
    provenance_hash: str = Field(default="", description="Hash for audit trail")

    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage."""
        if self.total_events == 0:
            return 0.0
        return (self.processed_events / self.total_events) * 100

    def calculate_provenance_hash(self) -> str:
        """Calculate provenance hash for audit."""
        data = f"{self.job_id}:{self.processed_events}:{self.current_event_id}"
        return hashlib.sha256(data.encode()).hexdigest()


class ReplayJob(BaseModel):
    """Represents a replay job."""
    job_id: str = Field(default_factory=lambda: str(uuid4()))
    name: Optional[str] = Field(default=None, description="Job name")
    description: Optional[str] = Field(default=None)
    filter: ReplayFilter = Field(default_factory=ReplayFilter)
    config: Dict[str, Any] = Field(default_factory=dict)
    progress: ReplayProgress = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = Field(default=None)

    def __init__(self, **data):
        super().__init__(**data)
        if self.progress is None:
            self.progress = ReplayProgress(job_id=self.job_id)


class ReplayCheckpoint(BaseModel):
    """Checkpoint for resumable replay."""
    job_id: str = Field(..., description="Replay job ID")
    position: int = Field(..., description="Last processed position")
    event_id: str = Field(..., description="Last processed event ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    processed_count: int = Field(default=0)
    checkpoint_data: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")


class ReplayEvent(BaseModel):
    """Wrapper for replayed events with metadata."""
    original_event: StoredEvent = Field(..., description="Original stored event")
    replay_job_id: str = Field(..., description="Replay job ID")
    replay_sequence: int = Field(..., description="Sequence in replay")
    original_timestamp: datetime = Field(..., description="Original event timestamp")
    replay_timestamp: datetime = Field(default_factory=datetime.utcnow)
    is_reprocessed: bool = Field(default=False, description="Whether this is a reprocessed event")
    idempotency_key: str = Field(default="", description="Idempotency key")

    def __init__(self, **data):
        super().__init__(**data)
        if not self.idempotency_key:
            self.idempotency_key = self._generate_idempotency_key()

    def _generate_idempotency_key(self) -> str:
        """Generate idempotency key for this event."""
        key_data = f"{self.replay_job_id}:{self.original_event.event_id}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]


# =============================================================================
# Idempotency Store
# =============================================================================


class IdempotencyStore:
    """Base class for idempotency stores."""

    async def is_processed(self, key: str) -> bool:
        """Check if event with key has been processed."""
        raise NotImplementedError

    async def mark_processed(self, key: str, result: Any = None) -> None:
        """Mark event as processed."""
        raise NotImplementedError

    async def get_result(self, key: str) -> Optional[Any]:
        """Get stored result for processed event."""
        raise NotImplementedError

    async def clear(self) -> None:
        """Clear all stored idempotency data."""
        raise NotImplementedError


class MemoryIdempotencyStore(IdempotencyStore):
    """In-memory idempotency store for testing."""

    def __init__(self):
        self._processed: Dict[str, Tuple[datetime, Any]] = {}
        self._lock = asyncio.Lock()

    async def is_processed(self, key: str) -> bool:
        """Check if event has been processed."""
        async with self._lock:
            return key in self._processed

    async def mark_processed(self, key: str, result: Any = None) -> None:
        """Mark event as processed."""
        async with self._lock:
            self._processed[key] = (datetime.utcnow(), result)

    async def get_result(self, key: str) -> Optional[Any]:
        """Get stored result."""
        async with self._lock:
            if key in self._processed:
                return self._processed[key][1]
            return None

    async def clear(self) -> None:
        """Clear all data."""
        async with self._lock:
            self._processed.clear()


# =============================================================================
# Checkpoint Store
# =============================================================================


class CheckpointStore:
    """Base class for checkpoint stores."""

    async def save_checkpoint(self, checkpoint: ReplayCheckpoint) -> None:
        """Save a checkpoint."""
        raise NotImplementedError

    async def get_checkpoint(self, job_id: str) -> Optional[ReplayCheckpoint]:
        """Get the latest checkpoint for a job."""
        raise NotImplementedError

    async def delete_checkpoint(self, job_id: str) -> None:
        """Delete checkpoint for a job."""
        raise NotImplementedError


class MemoryCheckpointStore(CheckpointStore):
    """In-memory checkpoint store."""

    def __init__(self):
        self._checkpoints: Dict[str, ReplayCheckpoint] = {}

    async def save_checkpoint(self, checkpoint: ReplayCheckpoint) -> None:
        """Save checkpoint."""
        self._checkpoints[checkpoint.job_id] = checkpoint

    async def get_checkpoint(self, job_id: str) -> Optional[ReplayCheckpoint]:
        """Get checkpoint."""
        return self._checkpoints.get(job_id)

    async def delete_checkpoint(self, job_id: str) -> None:
        """Delete checkpoint."""
        self._checkpoints.pop(job_id, None)


# =============================================================================
# Event Replay Manager
# =============================================================================


class EventReplayManager:
    """
    Event Replay Manager for GreenLang.

    Provides comprehensive event replay capabilities with support for:
    - Timestamp-based replay
    - Event ID-based replay
    - Filtered replay by event type
    - Speed control (real-time to instant)
    - Idempotency handling
    - Progress tracking and checkpointing

    Attributes:
        event_store: Event store for reading events
        config: Replay configuration

    Example:
        >>> store = EventStore(EventStoreConfig())
        >>> await store.start()
        >>>
        >>> replay_manager = EventReplayManager(store)
        >>>
        >>> # Replay from timestamp
        >>> async for event in replay_manager.replay_from_timestamp(
        ...     start_time=datetime(2025, 1, 1)
        ... ):
        ...     await process_event(event)
        >>>
        >>> # Replay with filters
        >>> filter = ReplayFilter(event_types=["emission.calculated"])
        >>> async for event in replay_manager.replay_with_filter(filter):
        ...     await process_event(event)
    """

    def __init__(
        self,
        event_store: EventStore,
        config: Optional[ReplayConfig] = None
    ):
        """
        Initialize the replay manager.

        Args:
            event_store: Event store to replay from
            config: Replay configuration
        """
        self.event_store = event_store
        self.config = config or ReplayConfig()

        # Stores
        self._idempotency_store: IdempotencyStore = MemoryIdempotencyStore()
        self._checkpoint_store: CheckpointStore = MemoryCheckpointStore()

        # Active jobs
        self._active_jobs: Dict[str, ReplayJob] = {}
        self._job_tasks: Dict[str, asyncio.Task] = {}

        # Event handlers
        self._event_handlers: List[Callable] = []
        self._progress_handlers: List[Callable] = []

        # Metrics
        self._total_replayed = 0
        self._total_skipped = 0
        self._total_failed = 0

        logger.info("EventReplayManager initialized")

    def set_idempotency_store(self, store: IdempotencyStore) -> None:
        """Set the idempotency store."""
        self._idempotency_store = store

    def set_checkpoint_store(self, store: CheckpointStore) -> None:
        """Set the checkpoint store."""
        self._checkpoint_store = store

    def add_event_handler(self, handler: Callable) -> None:
        """
        Add an event handler for replayed events.

        Args:
            handler: Async function that processes replayed events
        """
        self._event_handlers.append(handler)

    def add_progress_handler(self, handler: Callable) -> None:
        """
        Add a progress handler for replay progress updates.

        Args:
            handler: Function called on progress updates
        """
        self._progress_handlers.append(handler)

    async def replay_from_timestamp(
        self,
        start_time: datetime,
        end_time: Optional[datetime] = None,
        filter: Optional[ReplayFilter] = None
    ) -> AsyncIterator[ReplayEvent]:
        """
        Replay events starting from a timestamp.

        Args:
            start_time: Start timestamp (inclusive)
            end_time: End timestamp (inclusive)
            filter: Additional filter criteria

        Yields:
            ReplayEvent for each matching event
        """
        if filter is None:
            filter = ReplayFilter()

        filter.start_timestamp = start_time
        if end_time:
            filter.end_timestamp = end_time

        async for event in self._replay_with_filter(filter):
            yield event

    async def replay_from_event_id(
        self,
        event_id: str,
        filter: Optional[ReplayFilter] = None
    ) -> AsyncIterator[ReplayEvent]:
        """
        Replay events starting from a specific event ID.

        Args:
            event_id: Event ID to start from
            filter: Additional filter criteria

        Yields:
            ReplayEvent for each matching event
        """
        if filter is None:
            filter = ReplayFilter()

        filter.start_event_id = event_id

        async for event in self._replay_with_filter(filter):
            yield event

    async def replay_with_filter(
        self,
        filter: ReplayFilter
    ) -> AsyncIterator[ReplayEvent]:
        """
        Replay events matching a filter.

        Args:
            filter: Filter criteria

        Yields:
            ReplayEvent for each matching event
        """
        async for event in self._replay_with_filter(filter):
            yield event

    async def replay_stream(
        self,
        stream_id: str,
        start_version: int = 0,
        end_version: Optional[int] = None
    ) -> AsyncIterator[ReplayEvent]:
        """
        Replay events from a specific stream.

        Args:
            stream_id: Stream identifier
            start_version: Starting version
            end_version: Ending version (optional)

        Yields:
            ReplayEvent for each event in stream
        """
        filter = ReplayFilter(
            stream_ids=[stream_id],
            start_version=start_version
        )

        async for event in self._replay_with_filter(filter):
            if end_version and event.original_event.version > end_version:
                break
            yield event

    async def _replay_with_filter(
        self,
        filter: ReplayFilter
    ) -> AsyncIterator[ReplayEvent]:
        """
        Internal method to replay events with a filter.

        Args:
            filter: Filter criteria

        Yields:
            ReplayEvent for each matching event
        """
        job = ReplayJob(filter=filter)
        job.progress.status = ReplayStatus.RUNNING
        job.progress.started_at = datetime.utcnow()

        self._active_jobs[job.job_id] = job

        try:
            # Get events from store
            events = await self._load_events(filter)
            job.progress.total_events = len(events)

            sequence = 0
            last_event_time: Optional[datetime] = None
            start_time = time.time()

            for stored_event in events:
                # Check if job was cancelled
                if job.progress.status == ReplayStatus.CANCELLED:
                    break

                # Apply filter
                if not filter.matches(stored_event):
                    continue

                # Handle idempotency
                idempotency_key = self._generate_idempotency_key(
                    job.job_id, stored_event.event_id
                )

                if self.config.idempotency_strategy == IdempotencyStrategy.SKIP_PROCESSED:
                    if await self._idempotency_store.is_processed(idempotency_key):
                        job.progress.skipped_events += 1
                        self._total_skipped += 1
                        continue

                # Apply speed control
                if last_event_time and self.config.speed != ReplaySpeed.INSTANT:
                    delay = await self._calculate_delay(
                        last_event_time, stored_event.timestamp
                    )
                    if delay > 0:
                        await asyncio.sleep(delay)

                last_event_time = stored_event.timestamp

                # Create replay event
                sequence += 1
                replay_event = ReplayEvent(
                    original_event=stored_event,
                    replay_job_id=job.job_id,
                    replay_sequence=sequence,
                    original_timestamp=stored_event.timestamp,
                    idempotency_key=idempotency_key
                )

                # Update progress
                job.progress.processed_events += 1
                job.progress.current_position = sequence
                job.progress.current_event_id = stored_event.event_id
                job.progress.current_timestamp = stored_event.timestamp

                # Calculate metrics
                elapsed = time.time() - start_time
                job.progress.elapsed_seconds = elapsed
                if elapsed > 0:
                    job.progress.events_per_second = job.progress.processed_events / elapsed
                    remaining = job.progress.total_events - job.progress.processed_events
                    job.progress.estimated_remaining_seconds = remaining / job.progress.events_per_second

                # Mark as processed
                await self._idempotency_store.mark_processed(idempotency_key)
                self._total_replayed += 1

                # Checkpoint
                if sequence % self.config.checkpoint_interval == 0:
                    await self._save_checkpoint(job, stored_event)

                # Notify progress handlers
                await self._notify_progress(job.progress)

                yield replay_event

            # Complete
            job.progress.status = ReplayStatus.COMPLETED
            job.progress.completed_at = datetime.utcnow()
            job.progress.provenance_hash = job.progress.calculate_provenance_hash()

            await self._notify_progress(job.progress)
            logger.info(
                f"Replay completed: job={job.job_id} "
                f"processed={job.progress.processed_events} "
                f"skipped={job.progress.skipped_events}"
            )

        except Exception as e:
            job.progress.status = ReplayStatus.FAILED
            job.progress.last_error = str(e)
            logger.error(f"Replay failed: {e}", exc_info=True)
            raise

        finally:
            self._active_jobs.pop(job.job_id, None)

    async def _load_events(self, filter: ReplayFilter) -> List[StoredEvent]:
        """Load events from store based on filter."""
        events = []

        # If specific streams are requested
        if filter.stream_ids:
            for stream_id in filter.stream_ids:
                start_version = filter.start_version or 0
                stream_events = await self.event_store.read_stream(
                    stream_id, start_version
                )
                events.extend(stream_events)
        else:
            # Read all events
            position = 0
            while True:
                batch = await self.event_store.read_all(position, self.config.batch_size)
                if not batch:
                    break
                events.extend(batch)
                position += len(batch)

        # Sort by timestamp
        events.sort(key=lambda e: e.timestamp)

        # Apply timestamp filter
        if filter.start_timestamp:
            events = [e for e in events if e.timestamp >= filter.start_timestamp]
        if filter.end_timestamp:
            events = [e for e in events if e.timestamp <= filter.end_timestamp]

        # Find start event if specified
        if filter.start_event_id:
            start_index = None
            for i, event in enumerate(events):
                if event.event_id == filter.start_event_id:
                    start_index = i
                    break
            if start_index is not None:
                events = events[start_index:]

        return events

    async def _calculate_delay(
        self,
        last_time: datetime,
        current_time: datetime
    ) -> float:
        """Calculate delay between events based on speed settings."""
        if self.config.speed == ReplaySpeed.INSTANT:
            return 0.0

        # Calculate original time difference
        time_diff = (current_time - last_time).total_seconds()
        if time_diff <= 0:
            return 0.0

        # Apply speed multiplier
        if self.config.speed == ReplaySpeed.REAL_TIME:
            return time_diff
        elif self.config.speed == ReplaySpeed.FAST:
            return time_diff / 10.0
        elif self.config.speed == ReplaySpeed.FASTER:
            return time_diff / 100.0
        elif self.config.speed == ReplaySpeed.CUSTOM:
            return time_diff / self.config.speed_multiplier

        return 0.0

    def _generate_idempotency_key(self, job_id: str, event_id: str) -> str:
        """Generate idempotency key for an event."""
        key_data = f"{job_id}:{event_id}"
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]

    async def _save_checkpoint(
        self,
        job: ReplayJob,
        event: StoredEvent
    ) -> None:
        """Save a checkpoint for resumable replay."""
        checkpoint = ReplayCheckpoint(
            job_id=job.job_id,
            position=job.progress.current_position,
            event_id=event.event_id,
            processed_count=job.progress.processed_events,
            checkpoint_data={
                "skipped_count": job.progress.skipped_events,
                "failed_count": job.progress.failed_events,
            }
        )
        checkpoint.provenance_hash = hashlib.sha256(
            f"{checkpoint.job_id}:{checkpoint.position}:{checkpoint.event_id}".encode()
        ).hexdigest()

        await self._checkpoint_store.save_checkpoint(checkpoint)
        job.progress.last_checkpoint = datetime.utcnow()

    async def _notify_progress(self, progress: ReplayProgress) -> None:
        """Notify progress handlers."""
        for handler in self._progress_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(progress)
                else:
                    handler(progress)
            except Exception as e:
                logger.error(f"Progress handler error: {e}")

    # ==========================================================================
    # Job Management
    # ==========================================================================

    async def create_replay_job(
        self,
        filter: ReplayFilter,
        name: Optional[str] = None,
        description: Optional[str] = None,
        created_by: Optional[str] = None
    ) -> ReplayJob:
        """
        Create a new replay job without starting it.

        Args:
            filter: Filter criteria
            name: Job name
            description: Job description
            created_by: User who created the job

        Returns:
            Created ReplayJob
        """
        job = ReplayJob(
            name=name,
            description=description,
            filter=filter,
            created_by=created_by
        )
        self._active_jobs[job.job_id] = job
        logger.info(f"Created replay job: {job.job_id}")
        return job

    async def start_job(
        self,
        job_id: str,
        handler: Callable[[ReplayEvent], Any]
    ) -> None:
        """
        Start a replay job.

        Args:
            job_id: Job identifier
            handler: Function to process each replayed event
        """
        job = self._active_jobs.get(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")

        if job.progress.status == ReplayStatus.RUNNING:
            raise ValueError(f"Job already running: {job_id}")

        async def run_job():
            async for event in self._replay_with_filter(job.filter):
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)
                except Exception as e:
                    job.progress.failed_events += 1
                    self._total_failed += 1
                    job.progress.last_error = str(e)
                    logger.error(f"Event handler error: {e}")

                    if self.config.stop_on_error:
                        raise

                    if job.progress.failed_events >= self.config.max_errors:
                        raise RuntimeError("Max errors exceeded")

        task = asyncio.create_task(run_job())
        self._job_tasks[job_id] = task

    async def pause_job(self, job_id: str) -> None:
        """
        Pause a running replay job.

        Args:
            job_id: Job identifier
        """
        job = self._active_jobs.get(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")

        job.progress.status = ReplayStatus.PAUSED
        logger.info(f"Paused replay job: {job_id}")

    async def resume_job(
        self,
        job_id: str,
        handler: Callable[[ReplayEvent], Any]
    ) -> None:
        """
        Resume a paused replay job.

        Args:
            job_id: Job identifier
            handler: Function to process events
        """
        job = self._active_jobs.get(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")

        # Get checkpoint
        checkpoint = await self._checkpoint_store.get_checkpoint(job_id)
        if checkpoint:
            job.filter.start_event_id = checkpoint.event_id

        await self.start_job(job_id, handler)

    async def cancel_job(self, job_id: str) -> None:
        """
        Cancel a replay job.

        Args:
            job_id: Job identifier
        """
        job = self._active_jobs.get(job_id)
        if job:
            job.progress.status = ReplayStatus.CANCELLED

        task = self._job_tasks.get(job_id)
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._job_tasks.pop(job_id, None)
        logger.info(f"Cancelled replay job: {job_id}")

    async def get_job_progress(self, job_id: str) -> Optional[ReplayProgress]:
        """
        Get progress for a replay job.

        Args:
            job_id: Job identifier

        Returns:
            ReplayProgress or None if job not found
        """
        job = self._active_jobs.get(job_id)
        return job.progress if job else None

    async def list_active_jobs(self) -> List[ReplayJob]:
        """
        List all active replay jobs.

        Returns:
            List of active ReplayJob instances
        """
        return list(self._active_jobs.values())

    # ==========================================================================
    # Metrics and Status
    # ==========================================================================

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get replay manager metrics.

        Returns:
            Dictionary of metrics
        """
        return {
            "total_replayed": self._total_replayed,
            "total_skipped": self._total_skipped,
            "total_failed": self._total_failed,
            "active_jobs": len(self._active_jobs),
            "running_tasks": len(self._job_tasks),
        }

    async def health_check(self) -> Dict[str, Any]:
        """
        Check health of the replay manager.

        Returns:
            Health status
        """
        return {
            "healthy": True,
            "event_store_connected": self.event_store._started,
            "active_jobs": len(self._active_jobs),
            "metrics": self.get_metrics(),
        }


# =============================================================================
# FastAPI Router for Replay Management
# =============================================================================


def create_replay_router(replay_manager: EventReplayManager):
    """
    Create FastAPI router for replay management.

    Args:
        replay_manager: EventReplayManager instance

    Returns:
        FastAPI APIRouter
    """
    try:
        from fastapi import APIRouter, HTTPException, Query, status
        from fastapi.responses import JSONResponse
    except ImportError:
        logger.warning("FastAPI not available, skipping router creation")
        return None

    router = APIRouter(prefix="/api/v1/replay", tags=["Event Replay"])

    @router.post("/jobs", status_code=status.HTTP_201_CREATED)
    async def create_job(
        filter: ReplayFilter,
        name: Optional[str] = Query(None),
        description: Optional[str] = Query(None)
    ):
        """Create a new replay job."""
        job = await replay_manager.create_replay_job(
            filter=filter,
            name=name,
            description=description
        )
        return {
            "job_id": job.job_id,
            "status": job.progress.status.value,
            "created_at": job.created_at.isoformat()
        }

    @router.get("/jobs")
    async def list_jobs():
        """List all active replay jobs."""
        jobs = await replay_manager.list_active_jobs()
        return {
            "jobs": [
                {
                    "job_id": job.job_id,
                    "name": job.name,
                    "status": job.progress.status.value,
                    "progress_percent": job.progress.progress_percent,
                    "processed_events": job.progress.processed_events,
                    "total_events": job.progress.total_events,
                }
                for job in jobs
            ]
        }

    @router.get("/jobs/{job_id}")
    async def get_job(job_id: str):
        """Get replay job details."""
        progress = await replay_manager.get_job_progress(job_id)
        if not progress:
            raise HTTPException(status_code=404, detail="Job not found")

        return {
            "job_id": progress.job_id,
            "status": progress.status.value,
            "total_events": progress.total_events,
            "processed_events": progress.processed_events,
            "skipped_events": progress.skipped_events,
            "failed_events": progress.failed_events,
            "progress_percent": progress.progress_percent,
            "events_per_second": progress.events_per_second,
            "elapsed_seconds": progress.elapsed_seconds,
            "estimated_remaining_seconds": progress.estimated_remaining_seconds,
            "started_at": progress.started_at.isoformat() if progress.started_at else None,
            "completed_at": progress.completed_at.isoformat() if progress.completed_at else None,
            "last_error": progress.last_error,
        }

    @router.post("/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str):
        """Cancel a replay job."""
        try:
            await replay_manager.cancel_job(job_id)
            return {"message": "Job cancelled", "job_id": job_id}
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

    @router.get("/metrics")
    async def get_metrics():
        """Get replay manager metrics."""
        return replay_manager.get_metrics()

    @router.get("/health")
    async def health_check():
        """Check replay manager health."""
        return await replay_manager.health_check()

    return router
