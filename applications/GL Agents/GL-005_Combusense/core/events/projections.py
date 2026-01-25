# -*- coding: utf-8 -*-
"""
GL-005 CombustionControlAgent - Event Projections

This module implements read-model projections from the event stream.
Projections transform events into optimized views for specific query patterns.

Design Principles:
    - Projections are derived from events (CQRS pattern)
    - Projections can be rebuilt at any time
    - Multiple projections can exist for the same events
    - Projections are eventually consistent

Projections Included:
    - ControlHistoryProjection: Historical setpoint changes
    - StabilityMetricsProjection: Stability trending over time
    - AlarmHistoryProjection: Alarm occurrence tracking
    - PerformanceProjection: Control loop performance metrics

Example:
    >>> projector = ControlHistoryProjection()
    >>> async for event in event_store.subscribe():
    ...     projector.apply(event)
    >>> history = projector.get_setpoint_history(last_n=100)
"""

from __future__ import annotations

import logging
import statistics
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from core.events.base_event import DomainEvent
from core.events.domain_events import (
    AlarmTriggered,
    ControlSetpointChanged,
    OptimizationCompleted,
    SafetyInterventionTriggered,
    SensorReadingReceived,
    SystemStateChanged,
    AlarmSeverity,
    SystemMode,
)

logger = logging.getLogger(__name__)


class Projection(ABC):
    """
    Base class for event projections.

    Projections subscribe to events and build read-optimized views.
    They can be rebuilt from scratch by replaying the event stream.
    """

    def __init__(self, name: str):
        """Initialize projection."""
        self.name = name
        self.events_processed = 0
        self.last_event_timestamp: Optional[datetime] = None
        self.started_at = datetime.utcnow()

    @abstractmethod
    def apply(self, event: DomainEvent) -> None:
        """
        Apply an event to update the projection.

        Args:
            event: Event to apply
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the projection to initial state."""
        pass

    def on_event(self, event: DomainEvent) -> None:
        """
        Handle an incoming event.

        Args:
            event: Event to process
        """
        self.apply(event)
        self.events_processed += 1
        self.last_event_timestamp = event.metadata.timestamp

    def get_stats(self) -> Dict[str, Any]:
        """Get projection statistics."""
        return {
            "name": self.name,
            "events_processed": self.events_processed,
            "last_event_timestamp": self.last_event_timestamp,
            "started_at": self.started_at,
            "uptime_seconds": (datetime.utcnow() - self.started_at).total_seconds()
        }


@dataclass
class SetpointRecord:
    """Record of a setpoint change."""
    timestamp: datetime
    fuel_flow: float
    air_flow: float
    fuel_valve_position: float
    air_damper_position: float
    reason: str
    aggregate_id: str


class ControlHistoryProjection(Projection):
    """
    Projection for control setpoint history.

    Provides:
        - Chronological setpoint history
        - Change frequency analysis
        - Delta tracking between changes
    """

    def __init__(self, max_history: int = 10000):
        """Initialize control history projection."""
        super().__init__("ControlHistory")
        self.max_history = max_history
        self._history: Deque[SetpointRecord] = deque(maxlen=max_history)
        self._by_aggregate: Dict[str, Deque[SetpointRecord]] = {}
        self._change_counts: Dict[str, int] = {}

    def apply(self, event: DomainEvent) -> None:
        """Apply setpoint change events."""
        if not isinstance(event, ControlSetpointChanged):
            return

        record = SetpointRecord(
            timestamp=event.metadata.timestamp,
            fuel_flow=event.fuel_flow_setpoint,
            air_flow=event.air_flow_setpoint,
            fuel_valve_position=event.fuel_valve_position,
            air_damper_position=event.air_damper_position,
            reason=event.reason.value,
            aggregate_id=event.aggregate_id
        )

        self._history.append(record)

        # Track by aggregate
        if event.aggregate_id not in self._by_aggregate:
            self._by_aggregate[event.aggregate_id] = deque(maxlen=1000)
        self._by_aggregate[event.aggregate_id].append(record)

        # Count changes
        self._change_counts[event.aggregate_id] = (
            self._change_counts.get(event.aggregate_id, 0) + 1
        )

    def reset(self) -> None:
        """Reset projection."""
        self._history.clear()
        self._by_aggregate.clear()
        self._change_counts.clear()

    def get_history(
        self,
        aggregate_id: Optional[str] = None,
        last_n: Optional[int] = None,
        since: Optional[datetime] = None
    ) -> List[SetpointRecord]:
        """
        Get setpoint history.

        Args:
            aggregate_id: Filter by aggregate
            last_n: Return only last N records
            since: Return records after this time

        Returns:
            List of setpoint records
        """
        if aggregate_id:
            records = list(self._by_aggregate.get(aggregate_id, []))
        else:
            records = list(self._history)

        if since:
            records = [r for r in records if r.timestamp >= since]

        if last_n:
            records = records[-last_n:]

        return records

    def get_change_rate(
        self,
        aggregate_id: str,
        window_hours: float = 1.0
    ) -> float:
        """
        Get rate of setpoint changes per hour.

        Args:
            aggregate_id: Aggregate to analyze
            window_hours: Time window in hours

        Returns:
            Changes per hour
        """
        records = self._by_aggregate.get(aggregate_id, [])
        if not records:
            return 0.0

        cutoff = datetime.utcnow() - timedelta(hours=window_hours)
        recent = [r for r in records if r.timestamp >= cutoff]

        return len(recent) / window_hours

    def get_average_delta(
        self,
        aggregate_id: str,
        last_n: int = 100
    ) -> Dict[str, float]:
        """
        Get average setpoint deltas.

        Args:
            aggregate_id: Aggregate to analyze
            last_n: Number of recent changes

        Returns:
            Dict with average fuel and air flow deltas
        """
        records = list(self._by_aggregate.get(aggregate_id, []))[-last_n:]
        if len(records) < 2:
            return {"fuel_flow_delta": 0.0, "air_flow_delta": 0.0}

        fuel_deltas = []
        air_deltas = []

        for i in range(1, len(records)):
            fuel_deltas.append(
                abs(records[i].fuel_flow - records[i-1].fuel_flow)
            )
            air_deltas.append(
                abs(records[i].air_flow - records[i-1].air_flow)
            )

        return {
            "fuel_flow_delta": statistics.mean(fuel_deltas) if fuel_deltas else 0.0,
            "air_flow_delta": statistics.mean(air_deltas) if air_deltas else 0.0
        }


@dataclass
class StabilityRecord:
    """Record of stability metrics."""
    timestamp: datetime
    heat_output: float
    stability_score: float
    aggregate_id: str


class StabilityMetricsProjection(Projection):
    """
    Projection for stability metrics over time.

    Provides:
        - Stability score trending
        - Heat output variance tracking
        - Stability improvement analysis
    """

    def __init__(self, window_size: int = 1000):
        """Initialize stability projection."""
        super().__init__("StabilityMetrics")
        self.window_size = window_size
        self._readings: Dict[str, Deque[float]] = {}
        self._stability_history: Dict[str, Deque[StabilityRecord]] = {}

    def apply(self, event: DomainEvent) -> None:
        """Apply sensor reading events."""
        if not isinstance(event, SensorReadingReceived):
            return

        if event.reading_type != "heat_output":
            return

        agg_id = event.aggregate_id

        # Track readings
        if agg_id not in self._readings:
            self._readings[agg_id] = deque(maxlen=self.window_size)
        self._readings[agg_id].append(event.value)

        # Calculate stability
        readings = list(self._readings[agg_id])
        if len(readings) >= 10:
            mean = statistics.mean(readings)
            stdev = statistics.stdev(readings) if len(readings) > 1 else 0
            cv = stdev / mean if mean > 0 else 0
            stability_score = max(0, min(100, (1 - cv) * 100))

            record = StabilityRecord(
                timestamp=event.metadata.timestamp,
                heat_output=event.value,
                stability_score=stability_score,
                aggregate_id=agg_id
            )

            if agg_id not in self._stability_history:
                self._stability_history[agg_id] = deque(maxlen=1000)
            self._stability_history[agg_id].append(record)

    def reset(self) -> None:
        """Reset projection."""
        self._readings.clear()
        self._stability_history.clear()

    def get_current_stability(self, aggregate_id: str) -> float:
        """Get current stability score."""
        history = self._stability_history.get(aggregate_id, [])
        if not history:
            return 50.0
        return history[-1].stability_score

    def get_stability_trend(
        self,
        aggregate_id: str,
        window_size: int = 100
    ) -> str:
        """
        Determine if stability is improving or degrading.

        Returns:
            "improving", "degrading", or "stable"
        """
        history = list(self._stability_history.get(aggregate_id, []))[-window_size:]
        if len(history) < 10:
            return "stable"

        first_half = [r.stability_score for r in history[:len(history)//2]]
        second_half = [r.stability_score for r in history[len(history)//2:]]

        first_avg = statistics.mean(first_half)
        second_avg = statistics.mean(second_half)

        threshold = 2.0  # 2% change threshold
        if second_avg > first_avg + threshold:
            return "improving"
        elif second_avg < first_avg - threshold:
            return "degrading"
        else:
            return "stable"

    def get_variance_history(
        self,
        aggregate_id: str,
        last_n: int = 100
    ) -> List[Dict[str, Any]]:
        """Get heat output variance history."""
        history = list(self._stability_history.get(aggregate_id, []))[-last_n:]
        return [
            {
                "timestamp": r.timestamp.isoformat(),
                "heat_output": r.heat_output,
                "stability_score": r.stability_score
            }
            for r in history
        ]


@dataclass
class AlarmRecord:
    """Record of an alarm occurrence."""
    timestamp: datetime
    alarm_id: str
    alarm_name: str
    severity: str
    trigger_value: float
    setpoint: float
    aggregate_id: str
    acknowledged: bool = False
    duration_seconds: Optional[float] = None


class AlarmHistoryProjection(Projection):
    """
    Projection for alarm history.

    Provides:
        - Alarm occurrence tracking
        - Alarm frequency analysis
        - Mean time between alarms (MTBA)
        - Alarm acknowledgment tracking
    """

    def __init__(self, max_history: int = 5000):
        """Initialize alarm history projection."""
        super().__init__("AlarmHistory")
        self.max_history = max_history
        self._history: Deque[AlarmRecord] = deque(maxlen=max_history)
        self._by_aggregate: Dict[str, Deque[AlarmRecord]] = {}
        self._by_alarm_id: Dict[str, List[AlarmRecord]] = {}
        self._active_alarms: Dict[str, AlarmRecord] = {}

    def apply(self, event: DomainEvent) -> None:
        """Apply alarm events."""
        if not isinstance(event, AlarmTriggered):
            return

        record = AlarmRecord(
            timestamp=event.metadata.timestamp,
            alarm_id=event.alarm_id,
            alarm_name=event.alarm_name,
            severity=event.severity.value,
            trigger_value=event.trigger_value,
            setpoint=event.setpoint,
            aggregate_id=event.aggregate_id
        )

        self._history.append(record)

        # Track by aggregate
        if event.aggregate_id not in self._by_aggregate:
            self._by_aggregate[event.aggregate_id] = deque(maxlen=1000)
        self._by_aggregate[event.aggregate_id].append(record)

        # Track by alarm ID
        if event.alarm_id not in self._by_alarm_id:
            self._by_alarm_id[event.alarm_id] = []
        self._by_alarm_id[event.alarm_id].append(record)

        # Track active alarms
        key = f"{event.aggregate_id}:{event.alarm_id}"
        self._active_alarms[key] = record

    def reset(self) -> None:
        """Reset projection."""
        self._history.clear()
        self._by_aggregate.clear()
        self._by_alarm_id.clear()
        self._active_alarms.clear()

    def get_alarm_history(
        self,
        aggregate_id: Optional[str] = None,
        alarm_id: Optional[str] = None,
        severity: Optional[str] = None,
        since: Optional[datetime] = None,
        last_n: Optional[int] = None
    ) -> List[AlarmRecord]:
        """
        Get alarm history with optional filters.

        Args:
            aggregate_id: Filter by aggregate
            alarm_id: Filter by alarm ID
            severity: Filter by severity
            since: Filter by time
            last_n: Return only last N records

        Returns:
            Filtered list of alarm records
        """
        if alarm_id:
            records = self._by_alarm_id.get(alarm_id, [])
        elif aggregate_id:
            records = list(self._by_aggregate.get(aggregate_id, []))
        else:
            records = list(self._history)

        if severity:
            records = [r for r in records if r.severity == severity]

        if since:
            records = [r for r in records if r.timestamp >= since]

        if last_n:
            records = records[-last_n:]

        return records

    def get_alarm_frequency(
        self,
        aggregate_id: str,
        window_hours: float = 24.0
    ) -> Dict[str, int]:
        """
        Get alarm frequency by severity.

        Args:
            aggregate_id: Aggregate to analyze
            window_hours: Time window

        Returns:
            Dict mapping severity to count
        """
        cutoff = datetime.utcnow() - timedelta(hours=window_hours)
        records = [
            r for r in self._by_aggregate.get(aggregate_id, [])
            if r.timestamp >= cutoff
        ]

        frequency: Dict[str, int] = {}
        for record in records:
            frequency[record.severity] = frequency.get(record.severity, 0) + 1

        return frequency

    def get_mtba(
        self,
        aggregate_id: str,
        alarm_id: Optional[str] = None
    ) -> Optional[float]:
        """
        Calculate Mean Time Between Alarms (MTBA).

        Args:
            aggregate_id: Aggregate to analyze
            alarm_id: Optional specific alarm ID

        Returns:
            MTBA in seconds, or None if insufficient data
        """
        if alarm_id:
            records = [
                r for r in self._by_alarm_id.get(alarm_id, [])
                if r.aggregate_id == aggregate_id
            ]
        else:
            records = list(self._by_aggregate.get(aggregate_id, []))

        if len(records) < 2:
            return None

        # Sort by timestamp
        records = sorted(records, key=lambda r: r.timestamp)

        # Calculate time between alarms
        intervals = []
        for i in range(1, len(records)):
            delta = (records[i].timestamp - records[i-1].timestamp).total_seconds()
            intervals.append(delta)

        return statistics.mean(intervals) if intervals else None

    def get_active_alarm_count(self, aggregate_id: str) -> int:
        """Get count of active alarms for an aggregate."""
        return sum(
            1 for key in self._active_alarms
            if key.startswith(f"{aggregate_id}:")
        )


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot."""
    timestamp: datetime
    aggregate_id: str
    control_cycles: int
    avg_cycle_time_ms: float
    max_cycle_time_ms: float
    safety_interventions: int
    optimizations: int


class PerformanceProjection(Projection):
    """
    Projection for control loop performance.

    Provides:
        - Cycle time tracking
        - Safety intervention rate
        - Optimization effectiveness
    """

    def __init__(self):
        """Initialize performance projection."""
        super().__init__("Performance")
        self._metrics: Dict[str, PerformanceMetrics] = {}
        self._cycle_times: Dict[str, Deque[float]] = {}
        self._safety_counts: Dict[str, int] = {}
        self._optimization_counts: Dict[str, int] = {}
        self._setpoint_counts: Dict[str, int] = {}

    def apply(self, event: DomainEvent) -> None:
        """Apply various events to track performance."""
        agg_id = event.aggregate_id

        if isinstance(event, ControlSetpointChanged):
            self._setpoint_counts[agg_id] = (
                self._setpoint_counts.get(agg_id, 0) + 1
            )

        elif isinstance(event, SafetyInterventionTriggered):
            self._safety_counts[agg_id] = (
                self._safety_counts.get(agg_id, 0) + 1
            )

        elif isinstance(event, OptimizationCompleted):
            self._optimization_counts[agg_id] = (
                self._optimization_counts.get(agg_id, 0) + 1
            )
            # Track optimization execution time
            if agg_id not in self._cycle_times:
                self._cycle_times[agg_id] = deque(maxlen=1000)
            self._cycle_times[agg_id].append(event.execution_time_ms)

    def reset(self) -> None:
        """Reset projection."""
        self._metrics.clear()
        self._cycle_times.clear()
        self._safety_counts.clear()
        self._optimization_counts.clear()
        self._setpoint_counts.clear()

    def get_performance_summary(
        self,
        aggregate_id: str
    ) -> Dict[str, Any]:
        """
        Get performance summary for an aggregate.

        Args:
            aggregate_id: Aggregate to summarize

        Returns:
            Performance summary dict
        """
        cycle_times = list(self._cycle_times.get(aggregate_id, []))

        return {
            "aggregate_id": aggregate_id,
            "total_setpoint_changes": self._setpoint_counts.get(aggregate_id, 0),
            "total_safety_interventions": self._safety_counts.get(aggregate_id, 0),
            "total_optimizations": self._optimization_counts.get(aggregate_id, 0),
            "avg_cycle_time_ms": statistics.mean(cycle_times) if cycle_times else 0,
            "max_cycle_time_ms": max(cycle_times) if cycle_times else 0,
            "safety_rate_percent": (
                self._safety_counts.get(aggregate_id, 0) /
                max(self._setpoint_counts.get(aggregate_id, 1), 1) * 100
            )
        }


class ProjectionManager:
    """
    Manager for multiple projections.

    Handles:
        - Projection registration
        - Event distribution
        - Projection lifecycle
    """

    def __init__(self):
        """Initialize projection manager."""
        self._projections: Dict[str, Projection] = {}

    def register(self, projection: Projection) -> None:
        """Register a projection."""
        self._projections[projection.name] = projection
        logger.info(f"Registered projection: {projection.name}")

    def unregister(self, name: str) -> bool:
        """Unregister a projection."""
        if name in self._projections:
            del self._projections[name]
            return True
        return False

    def apply_event(self, event: DomainEvent) -> None:
        """Apply an event to all projections."""
        for projection in self._projections.values():
            try:
                projection.on_event(event)
            except Exception as e:
                logger.error(
                    f"Projection {projection.name} failed on event: {e}"
                )

    def get_projection(self, name: str) -> Optional[Projection]:
        """Get a projection by name."""
        return self._projections.get(name)

    def reset_all(self) -> None:
        """Reset all projections."""
        for projection in self._projections.values():
            projection.reset()

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get stats for all projections."""
        return {
            name: proj.get_stats()
            for name, proj in self._projections.items()
        }


# Default projection instances
control_history = ControlHistoryProjection()
stability_metrics = StabilityMetricsProjection()
alarm_history = AlarmHistoryProjection()
performance = PerformanceProjection()

# Default manager with all projections
default_manager = ProjectionManager()
default_manager.register(control_history)
default_manager.register(stability_metrics)
default_manager.register(alarm_history)
default_manager.register(performance)
