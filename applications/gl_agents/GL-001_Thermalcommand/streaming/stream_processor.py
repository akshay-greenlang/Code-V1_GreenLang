"""
Stream Processor Module - GL-001 ThermalCommand

This module provides real-time stream processing capabilities for
ThermalCommand events including windowing, aggregation, and
exactly-once processing semantics.

Key Features:
    - Tumbling windows (fixed, non-overlapping)
    - Sliding windows (overlapping)
    - Session windows (gap-based)
    - Hopping windows (fixed with hop interval)
    - Real-time aggregations (count, sum, avg, min, max, percentiles)
    - State management with checkpointing
    - Exactly-once processing guarantees
    - Late arrival handling with watermarks

Example:
    >>> processor = StreamProcessor(config)
    >>> processor.add_window(
    ...     WindowConfig(
    ...         window_type=WindowType.TUMBLING,
    ...         window_size_seconds=60,
    ...         aggregations=[AggregationType.AVG, AggregationType.MAX]
    ...     )
    ... )
    >>> await processor.process(telemetry_stream)

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import bisect
import hashlib
import heapq
import json
import logging
import statistics
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import (
    Any,
    AsyncGenerator,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from pydantic import BaseModel, Field, field_validator

from .event_envelope import EventEnvelope
from .kafka_schemas import (
    TelemetryNormalizedEvent,
    TelemetryPoint,
    QualityCode,
    UnitOfMeasure,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# ENUMS AND TYPES
# =============================================================================


class WindowType(str, Enum):
    """Window types for stream processing."""

    TUMBLING = "tumbling"  # Fixed, non-overlapping windows
    SLIDING = "sliding"  # Overlapping windows
    SESSION = "session"  # Gap-based windows
    HOPPING = "hopping"  # Fixed windows with hop interval


class AggregationType(str, Enum):
    """Aggregation types for window operations."""

    COUNT = "count"
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    FIRST = "first"
    LAST = "last"
    STDDEV = "stddev"
    VARIANCE = "variance"
    MEDIAN = "median"
    PERCENTILE_90 = "p90"
    PERCENTILE_95 = "p95"
    PERCENTILE_99 = "p99"
    RATE = "rate"  # Events per second
    DISTINCT_COUNT = "distinct_count"


class WatermarkStrategy(str, Enum):
    """Watermark strategies for handling late arrivals."""

    STRICT = "strict"  # Discard late arrivals
    ALLOW_LATE = "allow_late"  # Process late arrivals with separate output
    REPROCESS = "reprocess"  # Reprocess window on late arrival


# =============================================================================
# CONFIGURATION MODELS
# =============================================================================


class WindowConfig(BaseModel):
    """
    Configuration for a processing window.

    Attributes:
        window_id: Unique window identifier
        window_type: Type of window
        window_size_seconds: Window duration in seconds
        slide_seconds: Slide interval for sliding/hopping windows
        gap_seconds: Gap threshold for session windows
        aggregations: List of aggregation types to compute
        group_by_fields: Fields to group by within window
        watermark_seconds: Watermark delay for late arrivals
        watermark_strategy: How to handle late arrivals
        emit_early: Whether to emit partial results
        emit_interval_seconds: Interval for early emissions
    """

    window_id: str = Field(
        default_factory=lambda: f"win-{uuid.uuid4().hex[:8]}",
        description="Unique window identifier",
    )
    window_type: WindowType = Field(
        WindowType.TUMBLING,
        description="Type of window",
    )
    window_size_seconds: int = Field(
        60,
        ge=1,
        le=86400,
        description="Window duration in seconds",
    )
    slide_seconds: Optional[int] = Field(
        None,
        ge=1,
        description="Slide interval for sliding/hopping windows",
    )
    gap_seconds: Optional[int] = Field(
        None,
        ge=1,
        description="Gap threshold for session windows",
    )
    aggregations: List[AggregationType] = Field(
        default_factory=lambda: [AggregationType.AVG],
        description="Aggregation types to compute",
    )
    group_by_fields: List[str] = Field(
        default_factory=list,
        description="Fields to group by",
    )
    watermark_seconds: int = Field(
        30,
        ge=0,
        description="Watermark delay for late arrivals",
    )
    watermark_strategy: WatermarkStrategy = Field(
        WatermarkStrategy.ALLOW_LATE,
        description="Strategy for handling late arrivals",
    )
    emit_early: bool = Field(
        False,
        description="Whether to emit partial results",
    )
    emit_interval_seconds: int = Field(
        10,
        ge=1,
        description="Interval for early emissions",
    )
    allowed_lateness_seconds: int = Field(
        300,  # 5 minutes
        ge=0,
        description="Maximum lateness before discarding",
    )
    state_backend: str = Field(
        "memory",
        description="State backend: memory, rocksdb, redis",
    )

    @field_validator("slide_seconds")
    @classmethod
    def validate_slide(cls, v: Optional[int], info) -> Optional[int]:
        """Validate slide is set for sliding/hopping windows."""
        window_type = info.data.get("window_type")
        if window_type in (WindowType.SLIDING, WindowType.HOPPING):
            if v is None:
                raise ValueError(
                    f"slide_seconds required for {window_type.value} windows"
                )
        return v

    @field_validator("gap_seconds")
    @classmethod
    def validate_gap(cls, v: Optional[int], info) -> Optional[int]:
        """Validate gap is set for session windows."""
        window_type = info.data.get("window_type")
        if window_type == WindowType.SESSION:
            if v is None:
                raise ValueError("gap_seconds required for session windows")
        return v


class StreamProcessorConfig(BaseModel):
    """
    Configuration for the stream processor.

    Attributes:
        processor_id: Unique processor identifier
        parallelism: Number of parallel processing tasks
        checkpoint_interval_seconds: Checkpoint interval
        max_buffer_size: Maximum buffer size per window
        emit_on_close: Emit results when window closes
    """

    processor_id: str = Field(
        default_factory=lambda: f"proc-{uuid.uuid4().hex[:8]}",
        description="Unique processor identifier",
    )
    parallelism: int = Field(
        4,
        ge=1,
        le=64,
        description="Number of parallel processing tasks",
    )
    checkpoint_interval_seconds: int = Field(
        30,
        ge=1,
        description="Checkpoint interval in seconds",
    )
    max_buffer_size: int = Field(
        100000,
        ge=1000,
        description="Maximum buffer size per window",
    )
    emit_on_close: bool = Field(
        True,
        description="Emit results when window closes",
    )
    enable_metrics: bool = Field(
        True,
        description="Enable processing metrics",
    )
    dedup_enabled: bool = Field(
        True,
        description="Enable deduplication by envelope ID",
    )
    dedup_window_seconds: int = Field(
        300,
        ge=60,
        description="Deduplication window duration",
    )


# =============================================================================
# AGGREGATION RESULT MODELS
# =============================================================================


class AggregationValue(BaseModel):
    """Single aggregation result value."""

    aggregation_type: AggregationType
    value: float
    count: int
    unit: Optional[UnitOfMeasure] = None


class WindowBounds(BaseModel):
    """Window time bounds."""

    start: datetime
    end: datetime
    is_closed: bool = False

    @property
    def duration_seconds(self) -> float:
        """Return window duration in seconds."""
        return (self.end - self.start).total_seconds()

    def contains(self, timestamp: datetime) -> bool:
        """Check if timestamp is within window bounds."""
        return self.start <= timestamp < self.end


class GroupKey(BaseModel):
    """Key for grouped aggregations."""

    fields: Dict[str, Any] = Field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(json.dumps(self.fields, sort_keys=True, default=str))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, GroupKey):
            return False
        return self.fields == other.fields


class AggregationResult(BaseModel):
    """
    Result of window aggregation.

    Attributes:
        window_id: Window configuration ID
        window_bounds: Window time bounds
        group_key: Group key for grouped aggregations
        aggregations: Computed aggregation values
        event_count: Number of events in window
        late_event_count: Number of late arrivals
        processing_time_ms: Processing time in milliseconds
        is_partial: Whether this is a partial (early) result
        provenance_hash: Hash for audit trail
    """

    window_id: str
    window_bounds: WindowBounds
    group_key: Optional[GroupKey] = None
    aggregations: Dict[str, AggregationValue] = Field(default_factory=dict)
    event_count: int = 0
    late_event_count: int = 0
    quality_distribution: Dict[str, int] = Field(default_factory=dict)
    processing_time_ms: float = 0.0
    is_partial: bool = False
    emitted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    provenance_hash: str = ""

    def model_post_init(self, __context: Any) -> None:
        """Compute provenance hash after initialization."""
        if not self.provenance_hash:
            content = json.dumps(
                {
                    "window_id": self.window_id,
                    "bounds": {
                        "start": self.window_bounds.start.isoformat(),
                        "end": self.window_bounds.end.isoformat(),
                    },
                    "aggregations": {
                        k: v.value for k, v in self.aggregations.items()
                    },
                    "event_count": self.event_count,
                },
                sort_keys=True,
            )
            object.__setattr__(
                self,
                "provenance_hash",
                hashlib.sha256(content.encode()).hexdigest(),
            )


# =============================================================================
# WINDOW STATE MANAGEMENT
# =============================================================================


@dataclass
class WindowState:
    """State for a single window instance."""

    window_id: str
    bounds: WindowBounds
    group_key: Optional[GroupKey]
    values: List[float] = field(default_factory=list)
    timestamps: List[datetime] = field(default_factory=list)
    qualities: List[QualityCode] = field(default_factory=list)
    event_ids: Set[str] = field(default_factory=set)
    late_count: int = 0
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    last_updated: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def add_value(
        self,
        value: float,
        timestamp: datetime,
        quality: QualityCode,
        event_id: str,
    ) -> bool:
        """
        Add a value to the window state.

        Returns:
            True if value was added, False if duplicate
        """
        if event_id in self.event_ids:
            return False  # Duplicate

        self.values.append(value)
        self.timestamps.append(timestamp)
        self.qualities.append(quality)
        self.event_ids.add(event_id)
        self.last_updated = datetime.now(timezone.utc)
        return True

    @property
    def count(self) -> int:
        """Return number of values in window."""
        return len(self.values)

    def clear(self) -> None:
        """Clear window state."""
        self.values.clear()
        self.timestamps.clear()
        self.qualities.clear()
        self.event_ids.clear()
        self.late_count = 0


class StateStore(ABC):
    """Abstract base class for window state storage."""

    @abstractmethod
    async def get_window(
        self,
        window_id: str,
        group_key: Optional[GroupKey],
    ) -> Optional[WindowState]:
        """Get window state."""
        pass

    @abstractmethod
    async def put_window(self, state: WindowState) -> None:
        """Store window state."""
        pass

    @abstractmethod
    async def delete_window(
        self,
        window_id: str,
        group_key: Optional[GroupKey],
    ) -> None:
        """Delete window state."""
        pass

    @abstractmethod
    async def get_expired_windows(
        self,
        cutoff: datetime,
    ) -> List[WindowState]:
        """Get windows that have expired."""
        pass

    @abstractmethod
    async def checkpoint(self) -> str:
        """Create checkpoint and return checkpoint ID."""
        pass


class InMemoryStateStore(StateStore):
    """In-memory state store for window processing."""

    def __init__(self) -> None:
        self._windows: Dict[str, WindowState] = {}
        self._checkpoints: Dict[str, Dict[str, WindowState]] = {}

    def _make_key(
        self,
        window_id: str,
        group_key: Optional[GroupKey],
    ) -> str:
        """Create composite key for window lookup."""
        if group_key:
            return f"{window_id}:{hash(group_key)}"
        return window_id

    async def get_window(
        self,
        window_id: str,
        group_key: Optional[GroupKey],
    ) -> Optional[WindowState]:
        """Get window state from memory."""
        key = self._make_key(window_id, group_key)
        return self._windows.get(key)

    async def put_window(self, state: WindowState) -> None:
        """Store window state in memory."""
        key = self._make_key(state.window_id, state.group_key)
        self._windows[key] = state

    async def delete_window(
        self,
        window_id: str,
        group_key: Optional[GroupKey],
    ) -> None:
        """Delete window state from memory."""
        key = self._make_key(window_id, group_key)
        self._windows.pop(key, None)

    async def get_expired_windows(
        self,
        cutoff: datetime,
    ) -> List[WindowState]:
        """Get windows with end time before cutoff."""
        expired = []
        for state in self._windows.values():
            if state.bounds.end < cutoff:
                expired.append(state)
        return expired

    async def checkpoint(self) -> str:
        """Create checkpoint of current state."""
        checkpoint_id = f"ckpt-{uuid.uuid4().hex[:8]}"
        self._checkpoints[checkpoint_id] = {
            k: WindowState(
                window_id=v.window_id,
                bounds=v.bounds,
                group_key=v.group_key,
                values=v.values.copy(),
                timestamps=v.timestamps.copy(),
                qualities=v.qualities.copy(),
                event_ids=v.event_ids.copy(),
                late_count=v.late_count,
                created_at=v.created_at,
                last_updated=v.last_updated,
            )
            for k, v in self._windows.items()
        }
        logger.debug(f"Created checkpoint {checkpoint_id} with {len(self._windows)} windows")
        return checkpoint_id

    async def restore(self, checkpoint_id: str) -> bool:
        """Restore state from checkpoint."""
        if checkpoint_id not in self._checkpoints:
            return False
        self._windows = self._checkpoints[checkpoint_id].copy()
        logger.info(f"Restored from checkpoint {checkpoint_id}")
        return True


# =============================================================================
# AGGREGATION ENGINE
# =============================================================================


class AggregationEngine:
    """
    Engine for computing aggregations on window data.

    Supports various aggregation types including statistical
    measures and percentiles.
    """

    @staticmethod
    def compute(
        values: List[float],
        aggregation_type: AggregationType,
        unit: Optional[UnitOfMeasure] = None,
        window_seconds: float = 60.0,
    ) -> AggregationValue:
        """
        Compute aggregation for a list of values.

        Args:
            values: List of numeric values
            aggregation_type: Type of aggregation
            unit: Unit of measure
            window_seconds: Window duration for rate calculation

        Returns:
            AggregationValue with computed result
        """
        if not values:
            return AggregationValue(
                aggregation_type=aggregation_type,
                value=0.0,
                count=0,
                unit=unit,
            )

        count = len(values)

        if aggregation_type == AggregationType.COUNT:
            result = float(count)
        elif aggregation_type == AggregationType.SUM:
            result = sum(values)
        elif aggregation_type == AggregationType.AVG:
            result = statistics.mean(values)
        elif aggregation_type == AggregationType.MIN:
            result = min(values)
        elif aggregation_type == AggregationType.MAX:
            result = max(values)
        elif aggregation_type == AggregationType.FIRST:
            result = values[0]
        elif aggregation_type == AggregationType.LAST:
            result = values[-1]
        elif aggregation_type == AggregationType.STDDEV:
            result = statistics.stdev(values) if count > 1 else 0.0
        elif aggregation_type == AggregationType.VARIANCE:
            result = statistics.variance(values) if count > 1 else 0.0
        elif aggregation_type == AggregationType.MEDIAN:
            result = statistics.median(values)
        elif aggregation_type == AggregationType.PERCENTILE_90:
            result = AggregationEngine._percentile(values, 90)
        elif aggregation_type == AggregationType.PERCENTILE_95:
            result = AggregationEngine._percentile(values, 95)
        elif aggregation_type == AggregationType.PERCENTILE_99:
            result = AggregationEngine._percentile(values, 99)
        elif aggregation_type == AggregationType.RATE:
            result = count / window_seconds if window_seconds > 0 else 0.0
        elif aggregation_type == AggregationType.DISTINCT_COUNT:
            result = float(len(set(values)))
        else:
            result = 0.0

        return AggregationValue(
            aggregation_type=aggregation_type,
            value=result,
            count=count,
            unit=unit,
        )

    @staticmethod
    def _percentile(values: List[float], percentile: int) -> float:
        """Compute percentile of values."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * percentile / 100
        f = int(k)
        c = f + 1 if f < len(sorted_values) - 1 else f
        return sorted_values[f] + (k - f) * (sorted_values[c] - sorted_values[f])

    @staticmethod
    def compute_all(
        state: WindowState,
        aggregation_types: List[AggregationType],
        unit: Optional[UnitOfMeasure] = None,
    ) -> Dict[str, AggregationValue]:
        """
        Compute all requested aggregations for a window state.

        Args:
            state: Window state with values
            aggregation_types: List of aggregation types
            unit: Unit of measure

        Returns:
            Dictionary of aggregation name to value
        """
        window_seconds = state.bounds.duration_seconds
        results = {}

        for agg_type in aggregation_types:
            results[agg_type.value] = AggregationEngine.compute(
                state.values,
                agg_type,
                unit,
                window_seconds,
            )

        return results


# =============================================================================
# WINDOW ASSIGNERS
# =============================================================================


class WindowAssigner(ABC):
    """Abstract base class for window assignment."""

    @abstractmethod
    def assign_windows(
        self,
        timestamp: datetime,
        config: WindowConfig,
    ) -> List[WindowBounds]:
        """
        Assign event to windows based on timestamp.

        Args:
            timestamp: Event timestamp
            config: Window configuration

        Returns:
            List of window bounds the event belongs to
        """
        pass


class TumblingWindowAssigner(WindowAssigner):
    """Assigner for tumbling (fixed, non-overlapping) windows."""

    def assign_windows(
        self,
        timestamp: datetime,
        config: WindowConfig,
    ) -> List[WindowBounds]:
        """Assign to single tumbling window."""
        # Calculate window start aligned to epoch
        ts_seconds = timestamp.timestamp()
        window_start_ts = (
            ts_seconds // config.window_size_seconds
        ) * config.window_size_seconds

        window_start = datetime.fromtimestamp(window_start_ts, tz=timezone.utc)
        window_end = window_start + timedelta(seconds=config.window_size_seconds)

        return [WindowBounds(start=window_start, end=window_end)]


class SlidingWindowAssigner(WindowAssigner):
    """Assigner for sliding (overlapping) windows."""

    def assign_windows(
        self,
        timestamp: datetime,
        config: WindowConfig,
    ) -> List[WindowBounds]:
        """Assign to multiple overlapping windows."""
        if not config.slide_seconds:
            raise ValueError("slide_seconds required for sliding windows")

        windows = []
        ts_seconds = timestamp.timestamp()

        # Find all windows that contain this timestamp
        # Start from the earliest possible window
        earliest_window_start = (
            (ts_seconds - config.window_size_seconds) // config.slide_seconds + 1
        ) * config.slide_seconds

        window_start_ts = earliest_window_start
        while window_start_ts <= ts_seconds:
            window_end_ts = window_start_ts + config.window_size_seconds
            if window_end_ts > ts_seconds:
                windows.append(
                    WindowBounds(
                        start=datetime.fromtimestamp(window_start_ts, tz=timezone.utc),
                        end=datetime.fromtimestamp(window_end_ts, tz=timezone.utc),
                    )
                )
            window_start_ts += config.slide_seconds

        return windows


class HoppingWindowAssigner(WindowAssigner):
    """Assigner for hopping (fixed with hop interval) windows."""

    def assign_windows(
        self,
        timestamp: datetime,
        config: WindowConfig,
    ) -> List[WindowBounds]:
        """Assign to hopping windows (similar to sliding but typically non-overlapping)."""
        # Hopping windows are essentially sliding windows with hop = slide
        return SlidingWindowAssigner().assign_windows(timestamp, config)


class SessionWindowAssigner(WindowAssigner):
    """Assigner for session (gap-based) windows."""

    def assign_windows(
        self,
        timestamp: datetime,
        config: WindowConfig,
    ) -> List[WindowBounds]:
        """
        Assign to session window.

        Note: Session windows are dynamically merged, so initial assignment
        creates a single-event window that may be merged later.
        """
        if not config.gap_seconds:
            raise ValueError("gap_seconds required for session windows")

        return [
            WindowBounds(
                start=timestamp,
                end=timestamp + timedelta(seconds=config.gap_seconds),
            )
        ]


def get_window_assigner(window_type: WindowType) -> WindowAssigner:
    """Get appropriate window assigner for window type."""
    assigners = {
        WindowType.TUMBLING: TumblingWindowAssigner(),
        WindowType.SLIDING: SlidingWindowAssigner(),
        WindowType.HOPPING: HoppingWindowAssigner(),
        WindowType.SESSION: SessionWindowAssigner(),
    }
    return assigners[window_type]


# =============================================================================
# STREAM PROCESSOR
# =============================================================================


@dataclass
class ProcessorMetrics:
    """Metrics for stream processor monitoring."""

    events_processed: int = 0
    events_late: int = 0
    events_dropped: int = 0
    events_deduplicated: int = 0
    windows_created: int = 0
    windows_closed: int = 0
    aggregations_emitted: int = 0
    checkpoints_created: int = 0
    processing_errors: int = 0
    avg_processing_time_ms: float = 0.0


class StreamProcessor:
    """
    Stream processor for ThermalCommand events.

    Provides real-time aggregation with windowing and
    exactly-once processing semantics.

    Example:
        >>> config = StreamProcessorConfig()
        >>> processor = StreamProcessor(config)
        >>> window_config = WindowConfig(
        ...     window_type=WindowType.TUMBLING,
        ...     window_size_seconds=60,
        ...     aggregations=[AggregationType.AVG, AggregationType.MAX]
        ... )
        >>> processor.add_window(window_config)
        >>> async for result in processor.process_stream(event_stream):
        ...     print(f"Window result: {result}")
    """

    def __init__(
        self,
        config: Optional[StreamProcessorConfig] = None,
        state_store: Optional[StateStore] = None,
    ) -> None:
        """
        Initialize stream processor.

        Args:
            config: Processor configuration
            state_store: State store for window management
        """
        self.config = config or StreamProcessorConfig()
        self.state_store = state_store or InMemoryStateStore()

        self._window_configs: Dict[str, WindowConfig] = {}
        self._window_assigners: Dict[str, WindowAssigner] = {}
        self._running = False
        self._watermark: datetime = datetime.min.replace(tzinfo=timezone.utc)
        self._dedup_cache: Dict[str, datetime] = {}
        self._checkpoint_task: Optional[asyncio.Task] = None
        self._emit_task: Optional[asyncio.Task] = None

        self.metrics = ProcessorMetrics()

        logger.info(
            f"StreamProcessor initialized with id={self.config.processor_id}"
        )

    def add_window(self, config: WindowConfig) -> None:
        """
        Add a window configuration.

        Args:
            config: Window configuration
        """
        self._window_configs[config.window_id] = config
        self._window_assigners[config.window_id] = get_window_assigner(
            config.window_type
        )
        logger.info(
            f"Added window {config.window_id} type={config.window_type.value} "
            f"size={config.window_size_seconds}s"
        )

    def remove_window(self, window_id: str) -> bool:
        """
        Remove a window configuration.

        Args:
            window_id: Window ID to remove

        Returns:
            True if window was removed
        """
        if window_id in self._window_configs:
            del self._window_configs[window_id]
            del self._window_assigners[window_id]
            logger.info(f"Removed window {window_id}")
            return True
        return False

    async def start(self) -> None:
        """Start the stream processor."""
        if self._running:
            return

        self._running = True

        # Start background tasks
        self._checkpoint_task = asyncio.create_task(self._periodic_checkpoint())
        self._emit_task = asyncio.create_task(self._periodic_emit())

        logger.info("StreamProcessor started")

    async def stop(self) -> None:
        """Stop the stream processor."""
        if not self._running:
            return

        self._running = False

        # Cancel background tasks
        for task in [self._checkpoint_task, self._emit_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Final checkpoint
        await self.state_store.checkpoint()

        logger.info("StreamProcessor stopped")

    async def _periodic_checkpoint(self) -> None:
        """Background task for periodic checkpointing."""
        while self._running:
            try:
                await asyncio.sleep(self.config.checkpoint_interval_seconds)
                checkpoint_id = await self.state_store.checkpoint()
                self.metrics.checkpoints_created += 1
                logger.debug(f"Checkpoint created: {checkpoint_id}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Checkpoint error: {e}")

    async def _periodic_emit(self) -> None:
        """Background task for emitting window results."""
        while self._running:
            try:
                await asyncio.sleep(1.0)  # Check every second
                await self._emit_closed_windows()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Emit error: {e}")

    async def _emit_closed_windows(self) -> AsyncGenerator[AggregationResult, None]:
        """Emit results for closed windows."""
        # Calculate watermark cutoff
        cutoff = self._watermark - timedelta(
            seconds=max(
                wc.watermark_seconds + wc.allowed_lateness_seconds
                for wc in self._window_configs.values()
            )
            if self._window_configs
            else 0
        )

        expired = await self.state_store.get_expired_windows(cutoff)

        for state in expired:
            config = self._window_configs.get(state.window_id)
            if not config:
                continue

            result = self._compute_result(state, config, is_partial=False)
            await self.state_store.delete_window(state.window_id, state.group_key)
            self.metrics.windows_closed += 1
            self.metrics.aggregations_emitted += 1

            yield result

    def _is_duplicate(self, event_id: str) -> bool:
        """Check if event is a duplicate."""
        if not self.config.dedup_enabled:
            return False

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(seconds=self.config.dedup_window_seconds)

        # Clean old entries
        self._dedup_cache = {
            k: v for k, v in self._dedup_cache.items() if v > cutoff
        }

        if event_id in self._dedup_cache:
            self.metrics.events_deduplicated += 1
            return True

        self._dedup_cache[event_id] = now
        return False

    def _extract_group_key(
        self,
        point: TelemetryPoint,
        group_by_fields: List[str],
    ) -> Optional[GroupKey]:
        """Extract group key from telemetry point."""
        if not group_by_fields:
            return None

        fields = {}
        point_dict = point.model_dump()

        for field_name in group_by_fields:
            if field_name in point_dict:
                fields[field_name] = point_dict[field_name]

        return GroupKey(fields=fields) if fields else None

    async def process_telemetry(
        self,
        envelope: EventEnvelope,
    ) -> AsyncGenerator[AggregationResult, None]:
        """
        Process a telemetry envelope through all windows.

        Args:
            envelope: Event envelope containing telemetry

        Yields:
            Aggregation results as windows close or emit early
        """
        start_time = time.monotonic()

        # Check for duplicate
        if self._is_duplicate(envelope.metadata.envelope_id):
            return

        # Extract telemetry event
        if isinstance(envelope.payload, dict):
            event = TelemetryNormalizedEvent.model_validate(envelope.payload)
        else:
            event = envelope.payload

        # Update watermark
        self._watermark = max(self._watermark, event.collection_timestamp)

        # Process each telemetry point
        for point in event.points:
            # Skip bad quality points
            if point.quality in (
                QualityCode.BAD,
                QualityCode.BAD_SENSOR_FAILURE,
                QualityCode.BAD_COMM_FAILURE,
            ):
                continue

            # Process through each window configuration
            for window_id, config in self._window_configs.items():
                assigner = self._window_assigners[window_id]

                # Assign to windows
                windows = assigner.assign_windows(point.timestamp, config)

                for window_bounds in windows:
                    # Check for late arrival
                    is_late = point.timestamp < (
                        self._watermark
                        - timedelta(seconds=config.watermark_seconds)
                    )

                    if is_late:
                        if config.watermark_strategy == WatermarkStrategy.STRICT:
                            self.metrics.events_dropped += 1
                            continue
                        self.metrics.events_late += 1

                    # Extract group key
                    group_key = self._extract_group_key(
                        point, config.group_by_fields
                    )

                    # Get or create window state
                    state_key = f"{window_id}:{window_bounds.start.isoformat()}"
                    state = await self.state_store.get_window(state_key, group_key)

                    if state is None:
                        state = WindowState(
                            window_id=state_key,
                            bounds=window_bounds,
                            group_key=group_key,
                        )
                        self.metrics.windows_created += 1

                    # Add value to window
                    added = state.add_value(
                        point.value,
                        point.timestamp,
                        point.quality,
                        envelope.metadata.envelope_id,
                    )

                    if added and is_late:
                        state.late_count += 1

                    await self.state_store.put_window(state)

        self.metrics.events_processed += 1

        # Update average processing time
        processing_time = (time.monotonic() - start_time) * 1000
        n = self.metrics.events_processed
        self.metrics.avg_processing_time_ms = (
            (self.metrics.avg_processing_time_ms * (n - 1) + processing_time) / n
        )

        # Emit any closed windows
        async for result in self._emit_closed_windows():
            yield result

    def _compute_result(
        self,
        state: WindowState,
        config: WindowConfig,
        is_partial: bool = False,
    ) -> AggregationResult:
        """Compute aggregation result for a window state."""
        start_time = time.monotonic()

        # Compute all requested aggregations
        aggregations = AggregationEngine.compute_all(
            state,
            config.aggregations,
        )

        # Count quality distribution
        quality_dist = {}
        for q in state.qualities:
            quality_dist[q.value] = quality_dist.get(q.value, 0) + 1

        processing_time = (time.monotonic() - start_time) * 1000

        return AggregationResult(
            window_id=config.window_id,
            window_bounds=state.bounds,
            group_key=state.group_key,
            aggregations=aggregations,
            event_count=state.count,
            late_event_count=state.late_count,
            quality_distribution=quality_dist,
            processing_time_ms=processing_time,
            is_partial=is_partial,
        )

    async def process_stream(
        self,
        event_stream: AsyncGenerator[EventEnvelope, None],
    ) -> AsyncGenerator[AggregationResult, None]:
        """
        Process a stream of events.

        Args:
            event_stream: Async generator of event envelopes

        Yields:
            Aggregation results
        """
        async for envelope in event_stream:
            try:
                async for result in self.process_telemetry(envelope):
                    yield result
            except Exception as e:
                logger.error(f"Processing error: {e}")
                self.metrics.processing_errors += 1

    async def get_window_state(
        self,
        window_id: str,
        group_key: Optional[GroupKey] = None,
    ) -> Optional[WindowState]:
        """
        Get current state for a window.

        Args:
            window_id: Window ID
            group_key: Optional group key

        Returns:
            Window state if exists
        """
        return await self.state_store.get_window(window_id, group_key)

    def get_metrics(self) -> ProcessorMetrics:
        """Return current processor metrics."""
        return self.metrics

    async def checkpoint(self) -> str:
        """Create a manual checkpoint."""
        checkpoint_id = await self.state_store.checkpoint()
        self.metrics.checkpoints_created += 1
        return checkpoint_id


# =============================================================================
# STREAM PROCESSOR BUILDER
# =============================================================================


class StreamProcessorBuilder:
    """
    Builder for creating configured StreamProcessor instances.

    Example:
        >>> processor = (
        ...     StreamProcessorBuilder()
        ...     .with_tumbling_window(60, [AggregationType.AVG, AggregationType.MAX])
        ...     .with_sliding_window(300, 60, [AggregationType.PERCENTILE_95])
        ...     .with_group_by(["tag_id", "equipment_id"])
        ...     .with_watermark(30)
        ...     .build()
        ... )
    """

    def __init__(self) -> None:
        self._config = StreamProcessorConfig()
        self._windows: List[WindowConfig] = []
        self._state_store: Optional[StateStore] = None

    def with_processor_id(self, processor_id: str) -> StreamProcessorBuilder:
        """Set processor ID."""
        self._config = self._config.model_copy(
            update={"processor_id": processor_id}
        )
        return self

    def with_parallelism(self, parallelism: int) -> StreamProcessorBuilder:
        """Set parallelism level."""
        self._config = self._config.model_copy(
            update={"parallelism": parallelism}
        )
        return self

    def with_tumbling_window(
        self,
        size_seconds: int,
        aggregations: List[AggregationType],
        group_by: Optional[List[str]] = None,
    ) -> StreamProcessorBuilder:
        """Add a tumbling window."""
        self._windows.append(
            WindowConfig(
                window_type=WindowType.TUMBLING,
                window_size_seconds=size_seconds,
                aggregations=aggregations,
                group_by_fields=group_by or [],
            )
        )
        return self

    def with_sliding_window(
        self,
        size_seconds: int,
        slide_seconds: int,
        aggregations: List[AggregationType],
        group_by: Optional[List[str]] = None,
    ) -> StreamProcessorBuilder:
        """Add a sliding window."""
        self._windows.append(
            WindowConfig(
                window_type=WindowType.SLIDING,
                window_size_seconds=size_seconds,
                slide_seconds=slide_seconds,
                aggregations=aggregations,
                group_by_fields=group_by or [],
            )
        )
        return self

    def with_session_window(
        self,
        gap_seconds: int,
        aggregations: List[AggregationType],
        group_by: Optional[List[str]] = None,
    ) -> StreamProcessorBuilder:
        """Add a session window."""
        self._windows.append(
            WindowConfig(
                window_type=WindowType.SESSION,
                window_size_seconds=gap_seconds,  # Session uses gap as effective size
                gap_seconds=gap_seconds,
                aggregations=aggregations,
                group_by_fields=group_by or [],
            )
        )
        return self

    def with_watermark(
        self,
        watermark_seconds: int,
        strategy: WatermarkStrategy = WatermarkStrategy.ALLOW_LATE,
    ) -> StreamProcessorBuilder:
        """Set watermark configuration for all windows."""
        for window in self._windows:
            window.watermark_seconds = watermark_seconds
            window.watermark_strategy = strategy
        return self

    def with_state_store(self, store: StateStore) -> StreamProcessorBuilder:
        """Set custom state store."""
        self._state_store = store
        return self

    def with_deduplication(
        self,
        enabled: bool = True,
        window_seconds: int = 300,
    ) -> StreamProcessorBuilder:
        """Configure deduplication."""
        self._config = self._config.model_copy(
            update={
                "dedup_enabled": enabled,
                "dedup_window_seconds": window_seconds,
            }
        )
        return self

    def build(self) -> StreamProcessor:
        """Build the configured StreamProcessor."""
        processor = StreamProcessor(
            config=self._config,
            state_store=self._state_store,
        )

        for window_config in self._windows:
            processor.add_window(window_config)

        return processor
