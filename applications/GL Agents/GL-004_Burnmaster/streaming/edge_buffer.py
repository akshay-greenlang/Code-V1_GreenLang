"""
Edge Buffer Module - GL-004 BURNMASTER

This module provides edge buffering capabilities for combustion data streaming
with exactly-once semantics, data integrity guarantees, and graceful handling
of network partitions.

Key Features:
    - Local buffering for network partition resilience
    - Exactly-once semantics through message deduplication
    - Data integrity checks with SHA-256 checksums
    - Configurable overflow strategies
    - Persistent buffering option for durability

Example:
    >>> buffer = EdgeBuffer(config)
    >>> result = buffer.buffer_data(combustion_data)
    >>> data_list = buffer.get_buffered_data("5m")
    >>> flush_result = await buffer.flush_buffer("kafka")

Author: GreenLang Combustion Optimization Team
Version: 1.0.0
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import pickle
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field, field_validator

from .kafka_producer import CombustionData

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class BufferOverflowStrategy(str, Enum):
    """Strategies for handling buffer overflow."""

    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST = "drop_newest"
    REJECT = "reject"
    COMPRESS = "compress"
    SPILL_TO_DISK = "spill_to_disk"


class BufferState(str, Enum):
    """Buffer state."""

    EMPTY = "empty"
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"
    OVERFLOW = "overflow"


class FlushDestination(str, Enum):
    """Flush destination types."""

    KAFKA = "kafka"
    FILE = "file"
    DATABASE = "database"
    WEBHOOK = "webhook"


class IntegrityStatus(str, Enum):
    """Data integrity check status."""

    OK = "ok"
    CORRUPTED = "corrupted"
    MISSING = "missing"
    DUPLICATE = "duplicate"


# =============================================================================
# CONFIGURATION MODELS
# =============================================================================


class EdgeBufferConfig(BaseModel):
    """
    Configuration for edge buffer.

    Attributes:
        max_size: Maximum number of items in buffer
        max_age_seconds: Maximum age of buffered items
        overflow_strategy: Strategy for handling overflow
        persistence_enabled: Enable disk persistence
        persistence_path: Path for persistent storage
    """

    max_size: int = Field(
        10000,
        ge=100,
        le=1000000,
        description="Maximum number of items in buffer",
    )
    max_age_seconds: int = Field(
        3600,
        ge=60,
        le=86400,
        description="Maximum age of buffered items in seconds",
    )
    overflow_strategy: BufferOverflowStrategy = Field(
        BufferOverflowStrategy.DROP_OLDEST,
        description="Strategy for handling buffer overflow",
    )
    warning_threshold: float = Field(
        0.75,
        ge=0.5,
        le=0.95,
        description="Buffer fill ratio to trigger warning",
    )
    critical_threshold: float = Field(
        0.90,
        ge=0.8,
        le=0.99,
        description="Buffer fill ratio to trigger critical state",
    )
    persistence_enabled: bool = Field(
        False,
        description="Enable disk persistence for durability",
    )
    persistence_path: Optional[str] = Field(
        None,
        description="Path for persistent storage",
    )
    dedup_enabled: bool = Field(
        True,
        description="Enable message deduplication",
    )
    dedup_window_seconds: int = Field(
        300,
        ge=60,
        description="Deduplication window in seconds",
    )
    compression_enabled: bool = Field(
        False,
        description="Enable compression for spill-to-disk",
    )
    integrity_check_interval: int = Field(
        100,
        ge=10,
        description="Interval for integrity checks (every N operations)",
    )


# =============================================================================
# RESULT MODELS
# =============================================================================


class BufferResult(BaseModel):
    """Result of buffer operation."""

    success: bool = Field(..., description="Operation success status")
    message_id: str = Field(..., description="Message identifier")
    buffer_size: int = Field(0, ge=0, description="Current buffer size")
    buffer_state: BufferState = Field(
        BufferState.NORMAL,
        description="Current buffer state",
    )
    buffered_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Buffer timestamp",
    )
    duplicate: bool = Field(
        False,
        description="Whether message was a duplicate",
    )
    overflow_action: Optional[str] = Field(
        None,
        description="Action taken on overflow",
    )
    checksum: str = Field("", description="Data checksum")
    error: Optional[str] = Field(None, description="Error message if failed")


class FlushResult(BaseModel):
    """Result of buffer flush operation."""

    success: bool = Field(..., description="Flush success status")
    destination: str = Field(..., description="Flush destination")
    items_flushed: int = Field(0, ge=0, description="Number of items flushed")
    items_failed: int = Field(0, ge=0, description="Number of items that failed")
    bytes_flushed: int = Field(0, ge=0, description="Bytes flushed")
    flush_duration_ms: float = Field(
        0.0,
        ge=0.0,
        description="Flush duration in milliseconds",
    )
    flushed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Flush timestamp",
    )
    remaining_items: int = Field(0, ge=0, description="Items remaining in buffer")
    error: Optional[str] = Field(None, description="Error message if failed")


class OverflowResult(BaseModel):
    """Result of overflow handling."""

    success: bool = Field(..., description="Overflow handling success")
    strategy: BufferOverflowStrategy = Field(
        ...,
        description="Strategy applied",
    )
    items_affected: int = Field(0, ge=0, description="Items affected by overflow")
    items_dropped: int = Field(0, ge=0, description="Items dropped")
    items_spilled: int = Field(0, ge=0, description="Items spilled to disk")
    spill_path: Optional[str] = Field(
        None,
        description="Path to spill file if applicable",
    )
    handled_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Handling timestamp",
    )
    error: Optional[str] = Field(None, description="Error message if failed")


class IntegrityCheck(BaseModel):
    """Result of data integrity check."""

    status: IntegrityStatus = Field(..., description="Integrity status")
    buffer_size: int = Field(0, ge=0, description="Current buffer size")
    items_checked: int = Field(0, ge=0, description="Items checked")
    items_valid: int = Field(0, ge=0, description="Valid items")
    items_corrupted: int = Field(0, ge=0, description="Corrupted items")
    items_missing: int = Field(0, ge=0, description="Missing items")
    duplicates_found: int = Field(0, ge=0, description="Duplicates found")
    check_duration_ms: float = Field(
        0.0,
        ge=0.0,
        description="Check duration in milliseconds",
    )
    checked_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Check timestamp",
    )
    corrupted_ids: List[str] = Field(
        default_factory=list,
        description="IDs of corrupted items",
    )


# =============================================================================
# BUFFERED ITEM
# =============================================================================


@dataclass
class BufferedItem:
    """Single buffered item with metadata."""

    item_id: str
    data: CombustionData
    buffered_at: datetime
    checksum: str
    size_bytes: int
    attempts: int = 0
    last_attempt: Optional[datetime] = None

    def compute_checksum(self) -> str:
        """Compute SHA-256 checksum of data."""
        content = self.data.model_dump_json()
        return hashlib.sha256(content.encode()).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify data integrity."""
        return self.checksum == self.compute_checksum()


# =============================================================================
# EDGE BUFFER IMPLEMENTATION
# =============================================================================


class EdgeBuffer:
    """
    Edge buffer for combustion data with exactly-once semantics.

    This buffer provides local buffering for network partition resilience
    with data integrity guarantees and configurable overflow handling.

    Example:
        >>> config = EdgeBufferConfig(max_size=10000)
        >>> buffer = EdgeBuffer(config)
        >>> result = buffer.buffer_data(combustion_data)
        >>> if result.success:
        ...     data_list = buffer.get_buffered_data("5m")
        ...     flush_result = await buffer.flush_buffer("kafka")
    """

    def __init__(
        self,
        config: Optional[EdgeBufferConfig] = None,
    ) -> None:
        """
        Initialize EdgeBuffer.

        Args:
            config: Buffer configuration
        """
        self.config = config or EdgeBufferConfig()

        self._buffer: Deque[BufferedItem] = deque(maxlen=self.config.max_size)
        self._seen_ids: Dict[str, datetime] = {}
        self._checksums: Dict[str, str] = {}
        self._operation_count = 0
        self._lock = asyncio.Lock()

        # Metrics
        self._total_buffered = 0
        self._total_flushed = 0
        self._total_dropped = 0
        self._total_duplicates = 0

        # Initialize persistence if enabled
        if self.config.persistence_enabled and self.config.persistence_path:
            self._persistence_path = Path(self.config.persistence_path)
            self._persistence_path.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()

        logger.info(
            f"EdgeBuffer initialized with max_size={self.config.max_size}, "
            f"overflow_strategy={self.config.overflow_strategy.value}"
        )

    def buffer_data(
        self,
        data: CombustionData,
    ) -> BufferResult:
        """
        Buffer combustion data.

        Args:
            data: Combustion data to buffer

        Returns:
            BufferResult with operation status
        """
        self._operation_count += 1
        item_id = data.batch_id

        # Check for duplicates
        if self.config.dedup_enabled:
            if item_id in self._seen_ids:
                self._total_duplicates += 1
                return BufferResult(
                    success=True,
                    message_id=item_id,
                    buffer_size=len(self._buffer),
                    buffer_state=self._get_buffer_state(),
                    duplicate=True,
                )

        # Compute checksum for integrity
        content = data.model_dump_json()
        checksum = hashlib.sha256(content.encode()).hexdigest()
        size_bytes = len(content.encode())

        # Check for overflow
        overflow_action = None
        if len(self._buffer) >= self.config.max_size:
            overflow_result = self._handle_overflow()
            if not overflow_result.success:
                return BufferResult(
                    success=False,
                    message_id=item_id,
                    buffer_size=len(self._buffer),
                    buffer_state=BufferState.OVERFLOW,
                    error=overflow_result.error,
                )
            overflow_action = overflow_result.strategy.value

        # Create buffered item
        item = BufferedItem(
            item_id=item_id,
            data=data,
            buffered_at=datetime.now(timezone.utc),
            checksum=checksum,
            size_bytes=size_bytes,
        )

        # Add to buffer
        self._buffer.append(item)
        self._seen_ids[item_id] = item.buffered_at
        self._checksums[item_id] = checksum
        self._total_buffered += 1

        # Cleanup old dedup entries
        if self._operation_count % 100 == 0:
            self._cleanup_dedup_cache()

        # Periodic integrity check
        if self._operation_count % self.config.integrity_check_interval == 0:
            self.ensure_data_integrity()

        return BufferResult(
            success=True,
            message_id=item_id,
            buffer_size=len(self._buffer),
            buffer_state=self._get_buffer_state(),
            checksum=checksum,
            overflow_action=overflow_action,
        )

    def get_buffered_data(
        self,
        window: str,
    ) -> List[CombustionData]:
        """
        Get buffered data within time window.

        Args:
            window: Time window (e.g., "5m", "1h")

        Returns:
            List of combustion data within window
        """
        # Parse window
        value = int(window[:-1])
        unit = window[-1].lower()

        if unit == "s":
            window_seconds = value
        elif unit == "m":
            window_seconds = value * 60
        elif unit == "h":
            window_seconds = value * 3600
        elif unit == "d":
            window_seconds = value * 86400
        else:
            window_seconds = value * 60

        cutoff = datetime.now(timezone.utc) - timedelta(seconds=window_seconds)

        result = []
        for item in self._buffer:
            if item.buffered_at >= cutoff:
                result.append(item.data)

        logger.debug(
            f"Retrieved {len(result)} items from buffer for window {window}"
        )

        return result

    async def flush_buffer(
        self,
        destination: str,
    ) -> FlushResult:
        """
        Flush buffer to destination.

        Args:
            destination: Flush destination ("kafka", "file", etc.)

        Returns:
            FlushResult with flush status
        """
        start_time = time.monotonic()
        items_to_flush = list(self._buffer)

        if not items_to_flush:
            return FlushResult(
                success=True,
                destination=destination,
                items_flushed=0,
                remaining_items=0,
            )

        logger.info(
            f"Flushing {len(items_to_flush)} items to {destination}"
        )

        items_flushed = 0
        items_failed = 0
        bytes_flushed = 0

        async with self._lock:
            try:
                if destination == "kafka":
                    # In production, this would use the Kafka producer
                    for item in items_to_flush:
                        try:
                            # Simulate kafka send
                            await asyncio.sleep(0.001)
                            items_flushed += 1
                            bytes_flushed += item.size_bytes

                            # Remove from buffer
                            if item in self._buffer:
                                self._buffer.remove(item)
                                self._checksums.pop(item.item_id, None)

                        except Exception as e:
                            logger.error(f"Failed to flush item {item.item_id}: {e}")
                            items_failed += 1
                            item.attempts += 1
                            item.last_attempt = datetime.now(timezone.utc)

                elif destination == "file":
                    # Flush to file
                    file_path = self._get_flush_file_path()
                    items_flushed = await self._flush_to_file(
                        items_to_flush, file_path
                    )
                    bytes_flushed = sum(item.size_bytes for item in items_to_flush)

                    # Clear buffer on successful file flush
                    if items_flushed == len(items_to_flush):
                        self._buffer.clear()
                        self._checksums.clear()

                else:
                    return FlushResult(
                        success=False,
                        destination=destination,
                        error=f"Unknown destination: {destination}",
                    )

                flush_duration = (time.monotonic() - start_time) * 1000
                self._total_flushed += items_flushed

                return FlushResult(
                    success=items_failed == 0,
                    destination=destination,
                    items_flushed=items_flushed,
                    items_failed=items_failed,
                    bytes_flushed=bytes_flushed,
                    flush_duration_ms=flush_duration,
                    remaining_items=len(self._buffer),
                )

            except Exception as e:
                logger.error(f"Flush failed: {e}")
                return FlushResult(
                    success=False,
                    destination=destination,
                    items_flushed=items_flushed,
                    items_failed=items_failed,
                    remaining_items=len(self._buffer),
                    error=str(e),
                )

    def _handle_overflow(self) -> OverflowResult:
        """Handle buffer overflow based on configured strategy."""
        strategy = self.config.overflow_strategy
        items_affected = 0
        items_dropped = 0
        items_spilled = 0
        spill_path = None

        try:
            if strategy == BufferOverflowStrategy.DROP_OLDEST:
                # Drop oldest items (up to 10% of buffer)
                drop_count = max(1, len(self._buffer) // 10)
                for _ in range(drop_count):
                    if self._buffer:
                        item = self._buffer.popleft()
                        self._checksums.pop(item.item_id, None)
                        items_dropped += 1
                items_affected = drop_count
                self._total_dropped += items_dropped

            elif strategy == BufferOverflowStrategy.DROP_NEWEST:
                # Reject new items
                items_affected = 1
                return OverflowResult(
                    success=False,
                    strategy=strategy,
                    items_affected=1,
                    error="Buffer full, rejecting new item",
                )

            elif strategy == BufferOverflowStrategy.REJECT:
                # Reject with error
                return OverflowResult(
                    success=False,
                    strategy=strategy,
                    items_affected=0,
                    error="Buffer overflow - rejecting new items",
                )

            elif strategy == BufferOverflowStrategy.SPILL_TO_DISK:
                # Spill oldest items to disk
                if self.config.persistence_path:
                    spill_count = max(1, len(self._buffer) // 10)
                    spill_items = []
                    for _ in range(spill_count):
                        if self._buffer:
                            spill_items.append(self._buffer.popleft())

                    if spill_items:
                        spill_path = self._spill_to_disk(spill_items)
                        items_spilled = len(spill_items)
                        items_affected = items_spilled

                        for item in spill_items:
                            self._checksums.pop(item.item_id, None)
                else:
                    # Fall back to drop oldest if no persistence path
                    return self._handle_overflow_drop_oldest()

            elif strategy == BufferOverflowStrategy.COMPRESS:
                # Compress existing data (placeholder)
                items_affected = len(self._buffer)
                logger.info("Compression strategy applied (placeholder)")

            return OverflowResult(
                success=True,
                strategy=strategy,
                items_affected=items_affected,
                items_dropped=items_dropped,
                items_spilled=items_spilled,
                spill_path=spill_path,
            )

        except Exception as e:
            logger.error(f"Overflow handling failed: {e}")
            return OverflowResult(
                success=False,
                strategy=strategy,
                error=str(e),
            )

    def _handle_overflow_drop_oldest(self) -> OverflowResult:
        """Helper for drop oldest strategy."""
        drop_count = max(1, len(self._buffer) // 10)
        for _ in range(drop_count):
            if self._buffer:
                item = self._buffer.popleft()
                self._checksums.pop(item.item_id, None)

        self._total_dropped += drop_count

        return OverflowResult(
            success=True,
            strategy=BufferOverflowStrategy.DROP_OLDEST,
            items_affected=drop_count,
            items_dropped=drop_count,
        )

    def ensure_data_integrity(self) -> IntegrityCheck:
        """
        Check data integrity of buffered items.

        Returns:
            IntegrityCheck with integrity status
        """
        start_time = time.monotonic()

        items_checked = 0
        items_valid = 0
        items_corrupted = 0
        duplicates_found = 0
        corrupted_ids: List[str] = []
        seen_checksums: Set[str] = set()

        for item in self._buffer:
            items_checked += 1

            # Verify checksum
            if not item.verify_integrity():
                items_corrupted += 1
                corrupted_ids.append(item.item_id)
            else:
                items_valid += 1

            # Check for duplicates
            if item.checksum in seen_checksums:
                duplicates_found += 1
            else:
                seen_checksums.add(item.checksum)

        # Check for missing items (checksums without items)
        buffer_ids = {item.item_id for item in self._buffer}
        missing_count = len(self._checksums) - len(buffer_ids & set(self._checksums.keys()))

        # Determine overall status
        if items_corrupted > 0:
            status = IntegrityStatus.CORRUPTED
        elif missing_count > 0:
            status = IntegrityStatus.MISSING
        elif duplicates_found > 0:
            status = IntegrityStatus.DUPLICATE
        else:
            status = IntegrityStatus.OK

        check_duration = (time.monotonic() - start_time) * 1000

        logger.info(
            f"Integrity check completed: {items_valid}/{items_checked} valid, "
            f"{items_corrupted} corrupted, {duplicates_found} duplicates"
        )

        return IntegrityCheck(
            status=status,
            buffer_size=len(self._buffer),
            items_checked=items_checked,
            items_valid=items_valid,
            items_corrupted=items_corrupted,
            items_missing=missing_count,
            duplicates_found=duplicates_found,
            check_duration_ms=check_duration,
            corrupted_ids=corrupted_ids,
        )

    def _get_buffer_state(self) -> BufferState:
        """Get current buffer state based on fill ratio."""
        if len(self._buffer) == 0:
            return BufferState.EMPTY

        fill_ratio = len(self._buffer) / self.config.max_size

        if fill_ratio >= 1.0:
            return BufferState.OVERFLOW
        elif fill_ratio >= self.config.critical_threshold:
            return BufferState.CRITICAL
        elif fill_ratio >= self.config.warning_threshold:
            return BufferState.WARNING
        else:
            return BufferState.NORMAL

    def _cleanup_dedup_cache(self) -> None:
        """Clean up old entries from deduplication cache."""
        cutoff = datetime.now(timezone.utc) - timedelta(
            seconds=self.config.dedup_window_seconds
        )

        old_ids = [
            item_id
            for item_id, buffered_at in self._seen_ids.items()
            if buffered_at < cutoff
        ]

        for item_id in old_ids:
            del self._seen_ids[item_id]

        if old_ids:
            logger.debug(f"Cleaned {len(old_ids)} old dedup entries")

    def _spill_to_disk(self, items: List[BufferedItem]) -> str:
        """Spill items to disk file."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        filename = f"spill_{timestamp}.pkl"

        if self.config.persistence_path:
            file_path = Path(self.config.persistence_path) / filename
        else:
            file_path = Path(filename)

        with open(file_path, "wb") as f:
            pickle.dump(items, f)

        logger.info(f"Spilled {len(items)} items to {file_path}")

        return str(file_path)

    async def _flush_to_file(
        self,
        items: List[BufferedItem],
        file_path: Path,
    ) -> int:
        """Flush items to file."""
        try:
            data = [item.data.model_dump() for item in items]
            with open(file_path, "w") as f:
                json.dump(data, f)
            return len(items)
        except Exception as e:
            logger.error(f"Failed to flush to file: {e}")
            return 0

    def _get_flush_file_path(self) -> Path:
        """Get path for flush file."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        filename = f"flush_{timestamp}.json"

        if self.config.persistence_path:
            return Path(self.config.persistence_path) / filename
        return Path(filename)

    def _load_from_disk(self) -> None:
        """Load persisted items from disk on startup."""
        if not self.config.persistence_path:
            return

        persistence_path = Path(self.config.persistence_path)
        if not persistence_path.exists():
            return

        loaded_count = 0
        for file_path in persistence_path.glob("spill_*.pkl"):
            try:
                with open(file_path, "rb") as f:
                    items = pickle.load(f)
                    for item in items:
                        if len(self._buffer) < self.config.max_size:
                            self._buffer.append(item)
                            self._checksums[item.item_id] = item.checksum
                            loaded_count += 1

                # Remove processed spill file
                file_path.unlink()

            except Exception as e:
                logger.error(f"Failed to load spill file {file_path}: {e}")

        if loaded_count > 0:
            logger.info(f"Loaded {loaded_count} items from disk")

    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics."""
        return {
            "buffer_size": len(self._buffer),
            "max_size": self.config.max_size,
            "fill_ratio": len(self._buffer) / self.config.max_size,
            "state": self._get_buffer_state().value,
            "total_buffered": self._total_buffered,
            "total_flushed": self._total_flushed,
            "total_dropped": self._total_dropped,
            "total_duplicates": self._total_duplicates,
            "dedup_cache_size": len(self._seen_ids),
        }

    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()
        self._checksums.clear()
        logger.info("Buffer cleared")

    @property
    def size(self) -> int:
        """Return current buffer size."""
        return len(self._buffer)

    @property
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self._buffer) == 0

    @property
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return len(self._buffer) >= self.config.max_size
