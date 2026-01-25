"""
GL-012_SteamQual - Stream Processor

Real-time stream processing for steam quality data.
Handles high-rate sensor data with buffering and aggregation.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Dict, List, Optional
from collections import deque
import hashlib
import json
import threading
import time


@dataclass
class StreamConfig:
    """Configuration for stream processor."""
    buffer_size: int = 1000
    aggregation_window_seconds: float = 5.0
    max_latency_seconds: float = 5.0
    enable_buffering: bool = True
    enable_deduplication: bool = True


@dataclass
class StreamMessage:
    """Message in the data stream."""
    tag_id: str
    value: float
    timestamp: datetime
    quality: int = 192  # Good quality
    source: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tag_id": self.tag_id,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "quality": self.quality,
            "source": self.source,
        }

    def compute_hash(self) -> str:
        """Compute message hash for deduplication."""
        data = f"{self.tag_id}:{self.value}:{self.timestamp.isoformat()}"
        return hashlib.md5(data.encode()).hexdigest()


@dataclass
class AggregatedData:
    """Aggregated data over a time window."""
    tag_id: str
    window_start: datetime
    window_end: datetime
    count: int
    mean: float
    min_value: float
    max_value: float
    std_dev: float
    quality_score: float  # Fraction of good quality readings

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tag_id": self.tag_id,
            "window_start": self.window_start.isoformat(),
            "window_end": self.window_end.isoformat(),
            "count": self.count,
            "mean": self.mean,
            "min": self.min_value,
            "max": self.max_value,
            "std_dev": self.std_dev,
            "quality_score": self.quality_score,
        }


class StreamProcessor:
    """
    Real-time stream processor for steam quality data.

    Features:
    - Buffering during connectivity outages
    - Time alignment for mixed-rate signals
    - Aggregation over configurable windows
    - Deduplication of messages
    - Quality filtering

    Latency Target: sensor-to-metric < 5 seconds
    """

    GOOD_QUALITY_THRESHOLD = 192  # OPC UA good quality

    def __init__(self, config: Optional[StreamConfig] = None):
        """Initialize stream processor."""
        self.config = config or StreamConfig()

        # Message buffers per tag
        self._buffers: Dict[str, deque] = {}
        self._seen_hashes: deque = deque(maxlen=10000)  # For deduplication

        # Callbacks
        self._on_message_callbacks: List[Callable[[StreamMessage], None]] = []
        self._on_aggregation_callbacks: List[Callable[[AggregatedData], None]] = []

        # Statistics
        self._messages_received = 0
        self._messages_processed = 0
        self._messages_dropped = 0
        self._duplicates_filtered = 0

        # Thread safety
        self._lock = threading.Lock()

        # Aggregation state
        self._aggregation_buffers: Dict[str, List[StreamMessage]] = {}
        self._last_aggregation: Dict[str, datetime] = {}

    def process(self, message: StreamMessage) -> bool:
        """
        Process an incoming message.

        Args:
            message: Stream message to process

        Returns:
            True if message was processed, False if filtered/dropped
        """
        with self._lock:
            self._messages_received += 1

            # Deduplication
            if self.config.enable_deduplication:
                msg_hash = message.compute_hash()
                if msg_hash in self._seen_hashes:
                    self._duplicates_filtered += 1
                    return False
                self._seen_hashes.append(msg_hash)

            # Quality check
            if message.quality < self.GOOD_QUALITY_THRESHOLD:
                # Still process but mark as low quality
                pass

            # Latency check
            latency = (datetime.now(timezone.utc) - message.timestamp).total_seconds()
            if latency > self.config.max_latency_seconds * 10:  # Allow 10x for buffered data
                self._messages_dropped += 1
                return False

            # Add to buffer
            if self.config.enable_buffering:
                if message.tag_id not in self._buffers:
                    self._buffers[message.tag_id] = deque(maxlen=self.config.buffer_size)
                self._buffers[message.tag_id].append(message)

            # Add to aggregation buffer
            if message.tag_id not in self._aggregation_buffers:
                self._aggregation_buffers[message.tag_id] = []
                self._last_aggregation[message.tag_id] = message.timestamp
            self._aggregation_buffers[message.tag_id].append(message)

            # Check if aggregation window complete
            self._check_aggregation(message.tag_id)

            # Trigger callbacks
            for callback in self._on_message_callbacks:
                try:
                    callback(message)
                except Exception:
                    pass  # Don't let callback errors break processing

            self._messages_processed += 1
            return True

    def process_batch(self, messages: List[StreamMessage]) -> int:
        """
        Process a batch of messages.

        Args:
            messages: List of messages to process

        Returns:
            Number of messages successfully processed
        """
        processed = 0
        for message in messages:
            if self.process(message):
                processed += 1
        return processed

    def _check_aggregation(self, tag_id: str) -> None:
        """Check if aggregation window is complete and emit if so."""
        if tag_id not in self._aggregation_buffers:
            return

        buffer = self._aggregation_buffers[tag_id]
        if not buffer:
            return

        last_agg = self._last_aggregation.get(tag_id)
        if last_agg is None:
            return

        now = datetime.now(timezone.utc)
        window_elapsed = (now - last_agg).total_seconds()

        if window_elapsed >= self.config.aggregation_window_seconds:
            # Perform aggregation
            aggregated = self._aggregate(tag_id, buffer)

            # Clear buffer and update last aggregation time
            self._aggregation_buffers[tag_id] = []
            self._last_aggregation[tag_id] = now

            # Trigger callbacks
            for callback in self._on_aggregation_callbacks:
                try:
                    callback(aggregated)
                except Exception:
                    pass

    def _aggregate(self, tag_id: str, messages: List[StreamMessage]) -> AggregatedData:
        """Aggregate messages over the window."""
        if not messages:
            now = datetime.now(timezone.utc)
            return AggregatedData(
                tag_id=tag_id,
                window_start=now,
                window_end=now,
                count=0,
                mean=0.0,
                min_value=0.0,
                max_value=0.0,
                std_dev=0.0,
                quality_score=0.0,
            )

        values = [m.value for m in messages]
        timestamps = [m.timestamp for m in messages]
        qualities = [m.quality for m in messages]

        import numpy as np

        values_arr = np.array(values)
        good_quality_count = sum(1 for q in qualities if q >= self.GOOD_QUALITY_THRESHOLD)

        return AggregatedData(
            tag_id=tag_id,
            window_start=min(timestamps),
            window_end=max(timestamps),
            count=len(messages),
            mean=float(np.mean(values_arr)),
            min_value=float(np.min(values_arr)),
            max_value=float(np.max(values_arr)),
            std_dev=float(np.std(values_arr)) if len(values) > 1 else 0.0,
            quality_score=good_quality_count / len(messages),
        )

    def on_message(self, callback: Callable[[StreamMessage], None]) -> None:
        """Register callback for new messages."""
        self._on_message_callbacks.append(callback)

    def on_aggregation(self, callback: Callable[[AggregatedData], None]) -> None:
        """Register callback for aggregated data."""
        self._on_aggregation_callbacks.append(callback)

    def get_buffer(self, tag_id: str) -> List[StreamMessage]:
        """Get buffered messages for a tag."""
        with self._lock:
            if tag_id in self._buffers:
                return list(self._buffers[tag_id])
            return []

    def get_latest(self, tag_id: str) -> Optional[StreamMessage]:
        """Get latest message for a tag."""
        with self._lock:
            if tag_id in self._buffers and self._buffers[tag_id]:
                return self._buffers[tag_id][-1]
            return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        with self._lock:
            return {
                "messages_received": self._messages_received,
                "messages_processed": self._messages_processed,
                "messages_dropped": self._messages_dropped,
                "duplicates_filtered": self._duplicates_filtered,
                "active_tags": len(self._buffers),
                "buffer_sizes": {tag: len(buf) for tag, buf in self._buffers.items()},
            }

    def flush(self) -> None:
        """Flush all buffers and emit remaining aggregations."""
        with self._lock:
            for tag_id in list(self._aggregation_buffers.keys()):
                buffer = self._aggregation_buffers.get(tag_id, [])
                if buffer:
                    aggregated = self._aggregate(tag_id, buffer)
                    for callback in self._on_aggregation_callbacks:
                        try:
                            callback(aggregated)
                        except Exception:
                            pass

            self._aggregation_buffers.clear()
            self._last_aggregation.clear()

    def clear(self) -> None:
        """Clear all buffers and state."""
        with self._lock:
            self._buffers.clear()
            self._aggregation_buffers.clear()
            self._last_aggregation.clear()
            self._seen_hashes.clear()
            self._messages_received = 0
            self._messages_processed = 0
            self._messages_dropped = 0
            self._duplicates_filtered = 0
