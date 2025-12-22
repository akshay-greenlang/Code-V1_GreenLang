"""
Time Synchronizer Module - GL-004 BURNMASTER

This module provides timestamp synchronization capabilities for multi-source
combustion data streams, including clock drift detection, time alignment,
and out-of-order event handling.

Key Features:
    - Multi-source timestamp synchronization
    - Clock drift detection and correction
    - Reference time alignment
    - Out-of-order event handling
    - Sync issue logging and monitoring

Example:
    >>> synchronizer = TimeSynchronizer(config)
    >>> sync_result = synchronizer.synchronize_timestamps(sources)
    >>> drift = synchronizer.detect_clock_drift("source_a")
    >>> aligned = synchronizer.align_to_reference(timestamp, "ntp")

Author: GreenLang Combustion Optimization Team
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import statistics
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================


class SyncStatus(str, Enum):
    """Synchronization status."""

    SYNCHRONIZED = "synchronized"
    DRIFTING = "drifting"
    OUT_OF_SYNC = "out_of_sync"
    UNKNOWN = "unknown"


class DriftSeverity(str, Enum):
    """Clock drift severity levels."""

    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class SyncIssueType(str, Enum):
    """Types of synchronization issues."""

    CLOCK_DRIFT = "clock_drift"
    OUT_OF_ORDER = "out_of_order"
    MISSING_TIMESTAMP = "missing_timestamp"
    FUTURE_TIMESTAMP = "future_timestamp"
    STALE_TIMESTAMP = "stale_timestamp"
    SOURCE_UNAVAILABLE = "source_unavailable"


class ReferenceType(str, Enum):
    """Reference time source types."""

    NTP = "ntp"
    GPS = "gps"
    LOCAL = "local"
    DCS = "dcs"
    HISTORIAN = "historian"


# =============================================================================
# CONFIGURATION MODELS
# =============================================================================


class TimeSyncConfig(BaseModel):
    """Configuration for time synchronizer."""

    sync_id: str = Field(
        default_factory=lambda: f"sync-{uuid.uuid4().hex[:8]}",
        description="Synchronizer identifier",
    )
    reference_source: ReferenceType = Field(
        ReferenceType.NTP,
        description="Reference time source",
    )
    max_drift_ms: float = Field(
        1000.0,
        ge=1.0,
        description="Maximum acceptable drift in milliseconds",
    )
    minor_drift_threshold_ms: float = Field(
        100.0,
        ge=1.0,
        description="Minor drift threshold in milliseconds",
    )
    moderate_drift_threshold_ms: float = Field(
        500.0,
        ge=1.0,
        description="Moderate drift threshold in milliseconds",
    )
    severe_drift_threshold_ms: float = Field(
        2000.0,
        ge=1.0,
        description="Severe drift threshold in milliseconds",
    )
    history_window_size: int = Field(
        100,
        ge=10,
        description="Window size for drift calculation",
    )
    max_future_tolerance_ms: float = Field(
        5000.0,
        ge=0.0,
        description="Maximum tolerance for future timestamps",
    )
    stale_threshold_seconds: int = Field(
        300,
        ge=1,
        description="Threshold for stale timestamps",
    )
    reorder_buffer_size: int = Field(
        1000,
        ge=10,
        description="Buffer size for reordering events",
    )
    reorder_window_ms: float = Field(
        5000.0,
        ge=100.0,
        description="Window for reordering out-of-order events",
    )


# =============================================================================
# DATA MODELS
# =============================================================================


class Event(BaseModel):
    """Generic event with timestamp."""

    event_id: str = Field(
        default_factory=lambda: f"evt-{uuid.uuid4().hex[:8]}",
        description="Event identifier",
    )
    source: str = Field(..., description="Event source")
    timestamp: datetime = Field(..., description="Event timestamp")
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="Event payload",
    )
    sequence_number: Optional[int] = Field(
        None,
        description="Sequence number for ordering",
    )
    received_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Reception timestamp",
    )


class SyncResult(BaseModel):
    """Result of timestamp synchronization."""

    success: bool = Field(..., description="Synchronization success")
    status: SyncStatus = Field(..., description="Synchronization status")
    sources: Dict[str, SyncStatus] = Field(
        default_factory=dict,
        description="Status per source",
    )
    offsets: Dict[str, float] = Field(
        default_factory=dict,
        description="Offset from reference per source (ms)",
    )
    reference_time: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Reference timestamp",
    )
    reference_source: str = Field("", description="Reference source used")
    sync_duration_ms: float = Field(
        0.0,
        ge=0.0,
        description="Synchronization duration in milliseconds",
    )
    issues: List["SyncIssue"] = Field(
        default_factory=list,
        description="Detected sync issues",
    )
    synchronized_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Synchronization timestamp",
    )


class DriftDetection(BaseModel):
    """Result of clock drift detection."""

    source: str = Field(..., description="Source identifier")
    drift_ms: float = Field(..., description="Current drift in milliseconds")
    drift_rate_ms_per_second: float = Field(
        0.0,
        description="Drift rate in ms per second",
    )
    severity: DriftSeverity = Field(..., description="Drift severity")
    samples: int = Field(0, ge=0, description="Number of samples used")
    min_offset_ms: float = Field(0.0, description="Minimum observed offset")
    max_offset_ms: float = Field(0.0, description="Maximum observed offset")
    avg_offset_ms: float = Field(0.0, description="Average offset")
    std_offset_ms: float = Field(0.0, description="Standard deviation of offset")
    is_stable: bool = Field(True, description="Whether drift is stable")
    trend: str = Field("stable", description="Drift trend: increasing, decreasing, stable")
    detected_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Detection timestamp",
    )
    recommended_correction_ms: float = Field(
        0.0,
        description="Recommended correction value",
    )


class OrderedEvents(BaseModel):
    """Result of event reordering."""

    events: List[Event] = Field(..., description="Ordered events")
    original_order: List[str] = Field(
        default_factory=list,
        description="Original event ID order",
    )
    reordered_count: int = Field(
        0,
        ge=0,
        description="Number of events reordered",
    )
    dropped_count: int = Field(
        0,
        ge=0,
        description="Number of events dropped (too late)",
    )
    processing_time_ms: float = Field(
        0.0,
        ge=0.0,
        description="Processing time in milliseconds",
    )


class SyncIssue(BaseModel):
    """Synchronization issue record."""

    issue_id: str = Field(
        default_factory=lambda: f"issue-{uuid.uuid4().hex[:8]}",
        description="Issue identifier",
    )
    issue_type: SyncIssueType = Field(..., description="Type of issue")
    source: str = Field(..., description="Affected source")
    description: str = Field(..., description="Issue description")
    severity: DriftSeverity = Field(
        DriftSeverity.MINOR,
        description="Issue severity",
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Issue timestamp",
    )
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context",
    )
    resolved: bool = Field(False, description="Whether issue is resolved")
    resolved_at: Optional[datetime] = Field(None, description="Resolution timestamp")


# =============================================================================
# TIME SYNCHRONIZER IMPLEMENTATION
# =============================================================================


@dataclass
class SourceState:
    """State tracking for a time source."""

    source_id: str
    last_timestamp: Optional[datetime] = None
    last_received: Optional[datetime] = None
    offsets: Deque[float] = field(default_factory=lambda: deque(maxlen=100))
    status: SyncStatus = SyncStatus.UNKNOWN
    event_buffer: List[Event] = field(default_factory=list)
    last_sequence: Optional[int] = None

    def add_offset(self, offset_ms: float) -> None:
        """Add an offset observation."""
        self.offsets.append(offset_ms)

    def get_avg_offset(self) -> float:
        """Get average offset."""
        if not self.offsets:
            return 0.0
        return statistics.mean(self.offsets)

    def get_std_offset(self) -> float:
        """Get standard deviation of offset."""
        if len(self.offsets) < 2:
            return 0.0
        return statistics.stdev(self.offsets)


class TimeSynchronizer:
    """
    Time synchronizer for multi-source combustion data streams.

    This synchronizer provides clock drift detection, time alignment,
    and out-of-order event handling using DETERMINISTIC calculations.

    Example:
        >>> config = TimeSyncConfig()
        >>> synchronizer = TimeSynchronizer(config)
        >>> result = synchronizer.synchronize_timestamps({
        ...     "dcs": datetime.now(timezone.utc),
        ...     "plc": datetime.now(timezone.utc) + timedelta(milliseconds=50)
        ... })
    """

    def __init__(self, config: Optional[TimeSyncConfig] = None) -> None:
        """
        Initialize TimeSynchronizer.

        Args:
            config: Synchronizer configuration
        """
        self.config = config or TimeSyncConfig()
        self._source_states: Dict[str, SourceState] = {}
        self._reference_offset_ms: float = 0.0
        self._issues: List[SyncIssue] = []
        self._reorder_buffer: List[Event] = []
        self._last_sync: Optional[datetime] = None
        self._sync_count = 0

        logger.info(
            f"TimeSynchronizer initialized: "
            f"sync_id={self.config.sync_id}, "
            f"reference={self.config.reference_source.value}, "
            f"max_drift={self.config.max_drift_ms}ms"
        )

    def synchronize_timestamps(
        self,
        sources: Dict[str, datetime],
    ) -> SyncResult:
        """
        Synchronize timestamps from multiple sources.

        Args:
            sources: Dictionary of source ID to timestamp

        Returns:
            SyncResult with synchronization status
        """
        start_time = time.monotonic()
        self._sync_count += 1

        reference_time = datetime.now(timezone.utc)
        source_statuses: Dict[str, SyncStatus] = {}
        source_offsets: Dict[str, float] = {}
        issues: List[SyncIssue] = []

        for source_id, timestamp in sources.items():
            # Initialize source state if needed
            if source_id not in self._source_states:
                self._source_states[source_id] = SourceState(source_id=source_id)

            state = self._source_states[source_id]

            # Calculate offset from reference
            offset_ms = (timestamp - reference_time).total_seconds() * 1000
            state.add_offset(offset_ms)
            state.last_timestamp = timestamp
            state.last_received = reference_time
            source_offsets[source_id] = offset_ms

            # Determine sync status
            abs_offset = abs(offset_ms)

            if abs_offset <= self.config.minor_drift_threshold_ms:
                status = SyncStatus.SYNCHRONIZED
            elif abs_offset <= self.config.max_drift_ms:
                status = SyncStatus.DRIFTING
                issues.append(
                    SyncIssue(
                        issue_type=SyncIssueType.CLOCK_DRIFT,
                        source=source_id,
                        description=f"Clock drift detected: {offset_ms:.2f}ms",
                        severity=self._offset_to_severity(abs_offset),
                        context={"offset_ms": offset_ms},
                    )
                )
            else:
                status = SyncStatus.OUT_OF_SYNC
                issues.append(
                    SyncIssue(
                        issue_type=SyncIssueType.CLOCK_DRIFT,
                        source=source_id,
                        description=f"Source out of sync: {offset_ms:.2f}ms",
                        severity=DriftSeverity.CRITICAL,
                        context={"offset_ms": offset_ms},
                    )
                )

            state.status = status
            source_statuses[source_id] = status

            # Check for future timestamp
            if offset_ms > self.config.max_future_tolerance_ms:
                issues.append(
                    SyncIssue(
                        issue_type=SyncIssueType.FUTURE_TIMESTAMP,
                        source=source_id,
                        description=f"Future timestamp detected: {offset_ms:.2f}ms ahead",
                        severity=DriftSeverity.MODERATE,
                        context={"offset_ms": offset_ms},
                    )
                )

        # Determine overall status
        if all(s == SyncStatus.SYNCHRONIZED for s in source_statuses.values()):
            overall_status = SyncStatus.SYNCHRONIZED
        elif any(s == SyncStatus.OUT_OF_SYNC for s in source_statuses.values()):
            overall_status = SyncStatus.OUT_OF_SYNC
        elif any(s == SyncStatus.DRIFTING for s in source_statuses.values()):
            overall_status = SyncStatus.DRIFTING
        else:
            overall_status = SyncStatus.UNKNOWN

        sync_duration = (time.monotonic() - start_time) * 1000
        self._last_sync = reference_time
        self._issues.extend(issues)

        logger.debug(
            f"Synchronized {len(sources)} sources: status={overall_status.value}, "
            f"issues={len(issues)}, duration={sync_duration:.2f}ms"
        )

        return SyncResult(
            success=overall_status != SyncStatus.OUT_OF_SYNC,
            status=overall_status,
            sources=source_statuses,
            offsets=source_offsets,
            reference_time=reference_time,
            reference_source=self.config.reference_source.value,
            sync_duration_ms=sync_duration,
            issues=issues,
        )

    def detect_clock_drift(
        self,
        source: str,
    ) -> DriftDetection:
        """
        Detect clock drift for a specific source.

        Uses DETERMINISTIC statistical analysis - no ML involved.

        Args:
            source: Source identifier

        Returns:
            DriftDetection with drift analysis
        """
        state = self._source_states.get(source)

        if not state or len(state.offsets) < 2:
            return DriftDetection(
                source=source,
                drift_ms=0.0,
                severity=DriftSeverity.NONE,
                samples=0 if not state else len(state.offsets),
                is_stable=True,
            )

        offsets = list(state.offsets)
        avg_offset = statistics.mean(offsets)
        std_offset = statistics.stdev(offsets) if len(offsets) > 1 else 0.0

        # Calculate drift rate (ms per second) using linear regression
        drift_rate = self._calculate_drift_rate(offsets)

        # Determine severity
        abs_offset = abs(avg_offset)
        severity = self._offset_to_severity(abs_offset)

        # Determine trend
        if len(offsets) >= 10:
            recent = statistics.mean(offsets[-5:])
            older = statistics.mean(offsets[-10:-5])
            if recent > older + std_offset:
                trend = "increasing"
            elif recent < older - std_offset:
                trend = "decreasing"
            else:
                trend = "stable"
        else:
            trend = "stable"

        # Check stability
        is_stable = std_offset < self.config.minor_drift_threshold_ms

        # Calculate recommended correction
        recommended_correction = -avg_offset if abs_offset > self.config.minor_drift_threshold_ms else 0.0

        logger.debug(
            f"Drift detection for {source}: "
            f"drift={avg_offset:.2f}ms, rate={drift_rate:.4f}ms/s, "
            f"severity={severity.value}"
        )

        return DriftDetection(
            source=source,
            drift_ms=avg_offset,
            drift_rate_ms_per_second=drift_rate,
            severity=severity,
            samples=len(offsets),
            min_offset_ms=min(offsets),
            max_offset_ms=max(offsets),
            avg_offset_ms=avg_offset,
            std_offset_ms=std_offset,
            is_stable=is_stable,
            trend=trend,
            recommended_correction_ms=recommended_correction,
        )

    def align_to_reference(
        self,
        timestamp: datetime,
        reference: str,
    ) -> datetime:
        """
        Align a timestamp to the reference time source.

        Args:
            timestamp: Timestamp to align
            reference: Reference source type

        Returns:
            Aligned timestamp
        """
        # Get correction offset based on reference
        correction_ms = 0.0

        if reference == "ntp" or reference == ReferenceType.NTP.value:
            correction_ms = self._reference_offset_ms
        elif reference in self._source_states:
            state = self._source_states[reference]
            if state.offsets:
                correction_ms = -state.get_avg_offset()

        # Apply correction
        corrected = timestamp + timedelta(milliseconds=correction_ms)

        logger.debug(
            f"Aligned timestamp: original={timestamp.isoformat()}, "
            f"corrected={corrected.isoformat()}, correction={correction_ms:.2f}ms"
        )

        return corrected

    def handle_out_of_order(
        self,
        events: List[Event],
    ) -> OrderedEvents:
        """
        Handle out-of-order events by reordering within a time window.

        Args:
            events: List of potentially out-of-order events

        Returns:
            OrderedEvents with reordered events
        """
        start_time = time.monotonic()

        if not events:
            return OrderedEvents(
                events=[],
                original_order=[],
                reordered_count=0,
            )

        original_order = [e.event_id for e in events]

        # Add to reorder buffer
        self._reorder_buffer.extend(events)

        # Sort by timestamp
        self._reorder_buffer.sort(key=lambda e: e.timestamp)

        # Calculate cutoff for releasing events
        now = datetime.now(timezone.utc)
        release_cutoff = now - timedelta(milliseconds=self.config.reorder_window_ms)

        # Partition events
        ready_events: List[Event] = []
        buffered_events: List[Event] = []
        dropped_events: List[Event] = []

        stale_cutoff = now - timedelta(seconds=self.config.stale_threshold_seconds)

        for event in self._reorder_buffer:
            if event.timestamp < stale_cutoff:
                dropped_events.append(event)
            elif event.received_at < release_cutoff:
                ready_events.append(event)
            else:
                buffered_events.append(event)

        # Update buffer
        self._reorder_buffer = buffered_events

        # Log drops
        for event in dropped_events:
            self._issues.append(
                SyncIssue(
                    issue_type=SyncIssueType.STALE_TIMESTAMP,
                    source=event.source,
                    description=f"Dropped stale event {event.event_id}",
                    severity=DriftSeverity.MINOR,
                    context={"event_id": event.event_id},
                )
            )

        # Check for out-of-order
        reordered_ids = [e.event_id for e in ready_events]
        original_subset = [eid for eid in original_order if eid in reordered_ids]
        reordered_count = sum(
            1 for i, eid in enumerate(original_subset)
            if i < len(reordered_ids) and eid != reordered_ids[i]
        )

        processing_time = (time.monotonic() - start_time) * 1000

        if reordered_count > 0:
            logger.info(
                f"Reordered {reordered_count} events, "
                f"dropped {len(dropped_events)}"
            )

        return OrderedEvents(
            events=ready_events,
            original_order=original_order,
            reordered_count=reordered_count,
            dropped_count=len(dropped_events),
            processing_time_ms=processing_time,
        )

    def log_sync_issues(
        self,
        issue: SyncIssue,
    ) -> None:
        """
        Log a synchronization issue.

        Args:
            issue: Sync issue to log
        """
        self._issues.append(issue)

        # Log based on severity
        if issue.severity in (DriftSeverity.CRITICAL, DriftSeverity.SEVERE):
            logger.error(
                f"Sync issue [{issue.issue_type.value}] {issue.source}: "
                f"{issue.description}"
            )
        elif issue.severity == DriftSeverity.MODERATE:
            logger.warning(
                f"Sync issue [{issue.issue_type.value}] {issue.source}: "
                f"{issue.description}"
            )
        else:
            logger.info(
                f"Sync issue [{issue.issue_type.value}] {issue.source}: "
                f"{issue.description}"
            )

    def _offset_to_severity(self, abs_offset_ms: float) -> DriftSeverity:
        """Convert offset to severity level."""
        if abs_offset_ms <= self.config.minor_drift_threshold_ms:
            return DriftSeverity.NONE
        elif abs_offset_ms <= self.config.moderate_drift_threshold_ms:
            return DriftSeverity.MINOR
        elif abs_offset_ms <= self.config.severe_drift_threshold_ms:
            return DriftSeverity.MODERATE
        elif abs_offset_ms <= self.config.max_drift_ms * 2:
            return DriftSeverity.SEVERE
        else:
            return DriftSeverity.CRITICAL

    def _calculate_drift_rate(self, offsets: List[float]) -> float:
        """Calculate drift rate using simple linear regression."""
        if len(offsets) < 2:
            return 0.0

        n = len(offsets)
        x = list(range(n))
        y = offsets

        x_mean = statistics.mean(x)
        y_mean = statistics.mean(y)

        numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
        denominator = sum((xi - x_mean) ** 2 for xi in x)

        if denominator == 0:
            return 0.0

        # Slope represents drift per sample; convert to drift per second
        # Assuming 1 sample per second on average
        slope = numerator / denominator
        return slope

    def get_source_status(self, source: str) -> Optional[SyncStatus]:
        """Get synchronization status for a source."""
        state = self._source_states.get(source)
        return state.status if state else None

    def get_all_issues(
        self,
        since: Optional[datetime] = None,
        issue_type: Optional[SyncIssueType] = None,
    ) -> List[SyncIssue]:
        """
        Get logged sync issues with optional filtering.

        Args:
            since: Only return issues after this time
            issue_type: Filter by issue type

        Returns:
            List of sync issues
        """
        issues = self._issues

        if since:
            issues = [i for i in issues if i.timestamp >= since]

        if issue_type:
            issues = [i for i in issues if i.issue_type == issue_type]

        return issues

    def clear_issues(self) -> int:
        """Clear all logged issues. Returns count cleared."""
        count = len(self._issues)
        self._issues.clear()
        return count

    def get_stats(self) -> Dict[str, Any]:
        """Get synchronizer statistics."""
        return {
            "sync_id": self.config.sync_id,
            "sync_count": self._sync_count,
            "last_sync": self._last_sync.isoformat() if self._last_sync else None,
            "source_count": len(self._source_states),
            "issue_count": len(self._issues),
            "reorder_buffer_size": len(self._reorder_buffer),
            "sources": {
                source: {
                    "status": state.status.value,
                    "samples": len(state.offsets),
                    "avg_offset_ms": state.get_avg_offset(),
                }
                for source, state in self._source_states.items()
            },
        }
