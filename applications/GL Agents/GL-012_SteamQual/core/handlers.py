# -*- coding: utf-8 -*-
"""
GL-012 STEAMQUAL SteamQualityController - Event Handlers

This module provides event handlers for steam quality events including:
- Quality threshold violations
- Carryover risk alerts
- Separator performance events
- Control action events
- Safety and emergency events
- Audit and provenance events

All handlers support async processing and maintain audit trails
for regulatory compliance and provenance tracking.

Standards Compliance:
    - ASME PTC 19.11 (Steam Quality)
    - IEC 61511 (Functional Safety)
    - GreenLang Zero-Hallucination Principle

Example:
    >>> handler = QualityEventHandler()
    >>> handler.register_callback("quality_low", my_callback)
    >>> await handler.handle(quality_event)

Author: GL-BackendDeveloper
Version: 1.0.0
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Awaitable
from enum import Enum
import asyncio
import hashlib
import json
import logging
import uuid

from pydantic import BaseModel, Field

from .config import (
    AlertSeverity,
    CarryoverRiskLevel,
    QualityControlMode,
    SteamPhase,
)

logger = logging.getLogger(__name__)


# =============================================================================
# EVENT SCHEMAS
# =============================================================================

class EventType(str, Enum):
    """Steam quality event types."""
    # Quality events
    QUALITY_MEASUREMENT = "quality_measurement"
    QUALITY_LOW = "quality_low"
    QUALITY_CRITICAL = "quality_critical"
    QUALITY_RECOVERED = "quality_recovered"

    # Superheat events
    SUPERHEAT_LOW = "superheat_low"
    SUPERHEAT_HIGH = "superheat_high"
    SUPERHEAT_NORMAL = "superheat_normal"

    # Carryover events
    CARRYOVER_RISK_ELEVATED = "carryover_risk_elevated"
    CARRYOVER_DETECTED = "carryover_detected"
    CARRYOVER_CLEARED = "carryover_cleared"

    # Separator events
    SEPARATOR_EFFICIENCY_LOW = "separator_efficiency_low"
    SEPARATOR_PRESSURE_DROP_HIGH = "separator_pressure_drop_high"
    SEPARATOR_MAINTENANCE_NEEDED = "separator_maintenance_needed"

    # Control events
    CONTROL_ACTION_RECOMMENDED = "control_action_recommended"
    CONTROL_ACTION_EXECUTED = "control_action_executed"
    CONTROL_ACTION_REJECTED = "control_action_rejected"
    SETPOINT_CHANGED = "setpoint_changed"

    # Safety events
    SAFETY_INTERLOCK_ACTIVATED = "safety_interlock_activated"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"
    WATCHDOG_TIMEOUT = "watchdog_timeout"

    # Calculation events
    CALCULATION_COMPLETED = "calculation_completed"
    CALCULATION_FAILED = "calculation_failed"
    VALIDATION_ERROR = "validation_error"

    # Audit events
    PROVENANCE_RECORDED = "provenance_recorded"
    AUDIT_CHECKPOINT = "audit_checkpoint"


class SteamQualityEvent(BaseModel):
    """
    Base event model for steam quality events.

    Attributes:
        event_id: Unique event identifier
        event_type: Type of event
        timestamp: Event timestamp (UTC)
        system_id: Steam system identifier
        severity: Event severity level
        payload: Event-specific data
        provenance_hash: SHA-256 hash for audit trail
    """

    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique event identifier"
    )
    event_type: EventType = Field(
        ...,
        description="Type of event"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Event timestamp (UTC)"
    )
    system_id: Optional[str] = Field(
        default=None,
        description="Steam system identifier"
    )
    severity: AlertSeverity = Field(
        default=AlertSeverity.INFO,
        description="Event severity"
    )
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="Event-specific data"
    )
    provenance_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 hash for audit"
    )
    source_component: Optional[str] = Field(
        default=None,
        description="Component that generated the event"
    )

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of event."""
        data = {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "system_id": self.system_id,
            "payload": self.payload,
        }
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]


class QualityAlertEvent(SteamQualityEvent):
    """Alert event for quality threshold violations."""

    measured_value: float = Field(
        ...,
        description="Measured quality value"
    )
    threshold_value: float = Field(
        ...,
        description="Threshold that was violated"
    )
    deviation: float = Field(
        ...,
        description="Amount by which threshold was exceeded"
    )
    phase: Optional[SteamPhase] = Field(
        default=None,
        description="Steam phase at time of alert"
    )


class CarryoverAlertEvent(SteamQualityEvent):
    """Alert event for carryover risk."""

    risk_level: CarryoverRiskLevel = Field(
        ...,
        description="Carryover risk classification"
    )
    risk_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Estimated risk probability"
    )
    contributing_factors: List[str] = Field(
        default_factory=list,
        description="Factors contributing to risk"
    )


class ControlActionEvent(SteamQualityEvent):
    """Event for control actions and recommendations."""

    action_type: str = Field(
        ...,
        description="Type of control action"
    )
    target_parameter: str = Field(
        ...,
        description="Parameter being controlled"
    )
    current_value: float = Field(
        ...,
        description="Current value"
    )
    recommended_value: float = Field(
        ...,
        description="Recommended new value"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence in recommendation"
    )
    requires_approval: bool = Field(
        default=True,
        description="Whether action requires operator approval"
    )


class CalculationEvent(SteamQualityEvent):
    """Event for calculation audit trail."""

    calculation_type: str = Field(
        ...,
        description="Type of calculation"
    )
    inputs_hash: str = Field(
        ...,
        description="Hash of input data"
    )
    outputs_hash: str = Field(
        ...,
        description="Hash of output data"
    )
    formula_id: str = Field(
        ...,
        description="Formula identifier"
    )
    formula_version: str = Field(
        default="1.0.0",
        description="Formula version"
    )
    deterministic: bool = Field(
        default=True,
        description="Whether calculation was deterministic"
    )
    execution_time_ms: float = Field(
        default=0.0,
        description="Execution time in milliseconds"
    )


# =============================================================================
# BASE EVENT HANDLER
# =============================================================================

class EventHandler(ABC):
    """
    Base class for steam quality event handlers.

    Event handlers process events and trigger appropriate actions.
    All handlers support both synchronous and asynchronous callbacks.

    Attributes:
        name: Handler identifier
        _callbacks: Registered synchronous callbacks
        _async_callbacks: Registered asynchronous callbacks
    """

    def __init__(self, name: str) -> None:
        """
        Initialize the event handler.

        Args:
            name: Handler name/identifier
        """
        self.name = name
        self._callbacks: Dict[str, List[Callable]] = {}
        self._async_callbacks: Dict[str, List[Callable[..., Awaitable[None]]]] = {}
        self._event_count = 0
        self._error_count = 0
        self._last_event_time: Optional[datetime] = None
        self._event_history: List[SteamQualityEvent] = []

    @abstractmethod
    async def handle(self, event: SteamQualityEvent) -> None:
        """
        Handle an event.

        Args:
            event: Event to process

        Must be implemented by subclasses.
        """
        pass

    def register_callback(
        self,
        event_type: str,
        callback: Callable[[SteamQualityEvent], None],
    ) -> None:
        """
        Register a synchronous callback for an event type.

        Args:
            event_type: Event type to listen for (use "*" for all)
            callback: Callback function to invoke
        """
        if event_type not in self._callbacks:
            self._callbacks[event_type] = []
        self._callbacks[event_type].append(callback)
        logger.debug(f"Registered callback for {event_type} on {self.name}")

    def register_async_callback(
        self,
        event_type: str,
        callback: Callable[..., Awaitable[None]],
    ) -> None:
        """
        Register an asynchronous callback for an event type.

        Args:
            event_type: Event type to listen for (use "*" for all)
            callback: Async callback function to invoke
        """
        if event_type not in self._async_callbacks:
            self._async_callbacks[event_type] = []
        self._async_callbacks[event_type].append(callback)
        logger.debug(f"Registered async callback for {event_type} on {self.name}")

    def unregister_callback(
        self,
        event_type: str,
        callback: Callable,
    ) -> bool:
        """
        Unregister a callback.

        Args:
            event_type: Event type
            callback: Callback to remove

        Returns:
            True if callback was found and removed
        """
        if event_type in self._callbacks and callback in self._callbacks[event_type]:
            self._callbacks[event_type].remove(callback)
            return True
        if event_type in self._async_callbacks and callback in self._async_callbacks[event_type]:
            self._async_callbacks[event_type].remove(callback)
            return True
        return False

    def _invoke_callbacks(self, event: SteamQualityEvent) -> None:
        """Invoke registered synchronous callbacks for an event."""
        event_type = event.event_type.value if isinstance(event.event_type, Enum) else event.event_type
        callbacks = self._callbacks.get(event_type, [])
        callbacks.extend(self._callbacks.get("*", []))  # Wildcard callbacks

        for callback in callbacks:
            try:
                callback(event)
            except Exception as e:
                self._error_count += 1
                logger.error(f"Callback error in {self.name}: {e}", exc_info=True)

    async def _invoke_async_callbacks(self, event: SteamQualityEvent) -> None:
        """Invoke registered asynchronous callbacks for an event."""
        event_type = event.event_type.value if isinstance(event.event_type, Enum) else event.event_type
        callbacks = self._async_callbacks.get(event_type, [])
        callbacks.extend(self._async_callbacks.get("*", []))

        for callback in callbacks:
            try:
                await callback(event)
            except Exception as e:
                self._error_count += 1
                logger.error(f"Async callback error in {self.name}: {e}", exc_info=True)

    def _record_event(self, event: SteamQualityEvent) -> None:
        """Record event in history."""
        self._event_count += 1
        self._last_event_time = datetime.now(timezone.utc)
        self._event_history.append(event)

        # Keep bounded
        if len(self._event_history) > 1000:
            self._event_history = self._event_history[-1000:]

    @property
    def event_count(self) -> int:
        """Get total events handled."""
        return self._event_count

    @property
    def error_count(self) -> int:
        """Get total errors during handling."""
        return self._error_count

    def get_stats(self) -> Dict[str, Any]:
        """Get handler statistics."""
        return {
            "name": self.name,
            "event_count": self._event_count,
            "error_count": self._error_count,
            "last_event_time": (
                self._last_event_time.isoformat() if self._last_event_time else None
            ),
            "callback_count": sum(len(v) for v in self._callbacks.values()),
            "async_callback_count": sum(len(v) for v in self._async_callbacks.values()),
            "history_size": len(self._event_history),
        }

    def get_recent_events(
        self,
        count: int = 10,
        event_type: Optional[EventType] = None,
    ) -> List[SteamQualityEvent]:
        """
        Get recent events from history.

        Args:
            count: Maximum number of events to return
            event_type: Optional filter by event type

        Returns:
            List of recent events
        """
        events = self._event_history
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        return events[-count:]


# =============================================================================
# QUALITY EVENT HANDLER
# =============================================================================

class QualityEventHandler(EventHandler):
    """
    Handler for steam quality measurement and threshold events.

    Tracks quality measurements, detects threshold violations,
    and manages quality state transitions.

    Attributes:
        quality_history: History of quality measurements
        active_alerts: Currently active quality alerts
    """

    def __init__(self, name: str = "QualityEventHandler") -> None:
        """Initialize quality event handler."""
        super().__init__(name)
        self._quality_history: Dict[str, List[Dict]] = {}  # system_id -> measurements
        self._active_alerts: Dict[str, List[QualityAlertEvent]] = {}  # system_id -> alerts
        self._last_quality: Dict[str, float] = {}  # system_id -> last quality value

    async def handle(self, event: SteamQualityEvent) -> None:
        """
        Handle a quality event.

        Args:
            event: Quality event to process
        """
        self._record_event(event)
        system_id = event.system_id or "DEFAULT"

        # Initialize tracking for system
        if system_id not in self._quality_history:
            self._quality_history[system_id] = []
        if system_id not in self._active_alerts:
            self._active_alerts[system_id] = []

        event_type = event.event_type

        if event_type == EventType.QUALITY_MEASUREMENT:
            await self._handle_measurement(event, system_id)

        elif event_type in [EventType.QUALITY_LOW, EventType.QUALITY_CRITICAL]:
            await self._handle_quality_alert(event, system_id)

        elif event_type == EventType.QUALITY_RECOVERED:
            await self._handle_quality_recovery(event, system_id)

        elif event_type in [EventType.SUPERHEAT_LOW, EventType.SUPERHEAT_HIGH]:
            await self._handle_superheat_alert(event, system_id)

        # Invoke registered callbacks
        self._invoke_callbacks(event)
        await self._invoke_async_callbacks(event)

    async def _handle_measurement(
        self,
        event: SteamQualityEvent,
        system_id: str,
    ) -> None:
        """Handle quality measurement event."""
        measurement = {
            "timestamp": event.timestamp,
            "quality": event.payload.get("quality", 0.0),
            "superheat_c": event.payload.get("superheat_c"),
            "pressure_kpa": event.payload.get("pressure_kpa"),
            "temperature_c": event.payload.get("temperature_c"),
            "phase": event.payload.get("phase"),
        }

        self._quality_history[system_id].append(measurement)
        self._last_quality[system_id] = measurement["quality"]

        # Keep history bounded
        if len(self._quality_history[system_id]) > 1000:
            self._quality_history[system_id] = self._quality_history[system_id][-1000:]

        logger.debug(
            f"Quality measurement [{system_id}]: x={measurement['quality']:.4f}"
        )

    async def _handle_quality_alert(
        self,
        event: SteamQualityEvent,
        system_id: str,
    ) -> None:
        """Handle quality threshold violation."""
        if isinstance(event, QualityAlertEvent):
            alert = event
        else:
            alert = QualityAlertEvent(
                event_id=event.event_id,
                event_type=event.event_type,
                timestamp=event.timestamp,
                system_id=system_id,
                severity=event.severity,
                payload=event.payload,
                measured_value=event.payload.get("quality", 0.0),
                threshold_value=event.payload.get("threshold", 0.95),
                deviation=event.payload.get("deviation", 0.0),
            )

        self._active_alerts[system_id].append(alert)

        severity_str = event.severity.value if isinstance(event.severity, Enum) else event.severity
        logger.warning(
            f"Quality alert [{system_id}]: {event.event_type.value} - "
            f"x={alert.measured_value:.4f} (threshold: {alert.threshold_value:.4f}), "
            f"severity: {severity_str}"
        )

    async def _handle_quality_recovery(
        self,
        event: SteamQualityEvent,
        system_id: str,
    ) -> None:
        """Handle quality recovery event."""
        # Clear active alerts for this system
        cleared_count = len(self._active_alerts.get(system_id, []))
        self._active_alerts[system_id] = []

        logger.info(
            f"Quality recovered [{system_id}]: cleared {cleared_count} alerts, "
            f"current x={event.payload.get('quality', 'N/A')}"
        )

    async def _handle_superheat_alert(
        self,
        event: SteamQualityEvent,
        system_id: str,
    ) -> None:
        """Handle superheat margin alert."""
        superheat = event.payload.get("superheat_c", 0.0)
        threshold = event.payload.get("threshold_c", 3.0)

        logger.warning(
            f"Superheat alert [{system_id}]: {event.event_type.value} - "
            f"{superheat:.1f}C (threshold: {threshold:.1f}C)"
        )

    def get_current_quality(self, system_id: str) -> Optional[float]:
        """Get current quality value for a system."""
        return self._last_quality.get(system_id)

    def get_quality_trend(
        self,
        system_id: str,
        periods: int = 10,
    ) -> Dict[str, Any]:
        """
        Get quality trend analysis for a system.

        Args:
            system_id: System identifier
            periods: Number of recent measurements to analyze

        Returns:
            Trend analysis dictionary
        """
        history = self._quality_history.get(system_id, [])
        if not history:
            return {"system_id": system_id, "data_points": 0, "trend": "unknown"}

        recent = history[-periods:] if len(history) >= periods else history
        qualities = [m["quality"] for m in recent if m.get("quality") is not None]

        if len(qualities) < 2:
            return {
                "system_id": system_id,
                "data_points": len(qualities),
                "trend": "insufficient_data",
            }

        # Simple trend analysis
        first_half = qualities[:len(qualities)//2]
        second_half = qualities[len(qualities)//2:]
        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)

        if avg_second > avg_first + 0.01:
            trend = "improving"
        elif avg_second < avg_first - 0.01:
            trend = "declining"
        else:
            trend = "stable"

        return {
            "system_id": system_id,
            "data_points": len(qualities),
            "trend": trend,
            "current": qualities[-1] if qualities else None,
            "average": sum(qualities) / len(qualities),
            "min": min(qualities),
            "max": max(qualities),
        }

    def get_active_alerts(
        self,
        system_id: Optional[str] = None,
    ) -> List[QualityAlertEvent]:
        """Get active quality alerts."""
        if system_id:
            return list(self._active_alerts.get(system_id, []))

        all_alerts = []
        for alerts in self._active_alerts.values():
            all_alerts.extend(alerts)
        return all_alerts

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of all active alerts."""
        all_alerts = self.get_active_alerts()

        severity_counts = {s.value: 0 for s in AlertSeverity}
        for alert in all_alerts:
            sev = alert.severity.value if isinstance(alert.severity, Enum) else str(alert.severity)
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        return {
            "total_active": len(all_alerts),
            "by_severity": severity_counts,
            "systems_affected": list(self._active_alerts.keys()),
        }


# =============================================================================
# CARRYOVER EVENT HANDLER
# =============================================================================

class CarryoverEventHandler(EventHandler):
    """
    Handler for carryover risk and detection events.

    Tracks carryover risk levels, detects carryover conditions,
    and manages mitigation actions.
    """

    def __init__(self, name: str = "CarryoverEventHandler") -> None:
        """Initialize carryover event handler."""
        super().__init__(name)
        self._risk_history: Dict[str, List[Dict]] = {}
        self._active_risks: Dict[str, CarryoverRiskLevel] = {}
        self._carryover_incidents: List[Dict] = []

    async def handle(self, event: SteamQualityEvent) -> None:
        """Handle a carryover event."""
        self._record_event(event)
        system_id = event.system_id or "DEFAULT"

        if system_id not in self._risk_history:
            self._risk_history[system_id] = []

        event_type = event.event_type

        if event_type == EventType.CARRYOVER_RISK_ELEVATED:
            await self._handle_risk_elevated(event, system_id)

        elif event_type == EventType.CARRYOVER_DETECTED:
            await self._handle_carryover_detected(event, system_id)

        elif event_type == EventType.CARRYOVER_CLEARED:
            await self._handle_carryover_cleared(event, system_id)

        self._invoke_callbacks(event)
        await self._invoke_async_callbacks(event)

    async def _handle_risk_elevated(
        self,
        event: SteamQualityEvent,
        system_id: str,
    ) -> None:
        """Handle elevated carryover risk."""
        risk_level = CarryoverRiskLevel(
            event.payload.get("risk_level", CarryoverRiskLevel.LOW.value)
        )
        probability = event.payload.get("probability", 0.0)

        self._active_risks[system_id] = risk_level
        self._risk_history[system_id].append({
            "timestamp": event.timestamp,
            "risk_level": risk_level.value,
            "probability": probability,
            "factors": event.payload.get("factors", []),
        })

        logger.warning(
            f"Carryover risk elevated [{system_id}]: {risk_level.value} "
            f"(probability: {probability:.1%})"
        )

    async def _handle_carryover_detected(
        self,
        event: SteamQualityEvent,
        system_id: str,
    ) -> None:
        """Handle carryover detection."""
        incident = {
            "timestamp": event.timestamp,
            "system_id": system_id,
            "severity": event.severity.value if isinstance(event.severity, Enum) else event.severity,
            "tds_ppm": event.payload.get("tds_ppm"),
            "silica_ppb": event.payload.get("silica_ppb"),
            "estimated_loss_kg_hr": event.payload.get("estimated_loss_kg_hr"),
        }
        self._carryover_incidents.append(incident)

        logger.error(
            f"Carryover detected [{system_id}]: TDS={incident.get('tds_ppm')}ppm, "
            f"Silica={incident.get('silica_ppb')}ppb"
        )

    async def _handle_carryover_cleared(
        self,
        event: SteamQualityEvent,
        system_id: str,
    ) -> None:
        """Handle carryover cleared condition."""
        self._active_risks[system_id] = CarryoverRiskLevel.NEGLIGIBLE

        logger.info(f"Carryover cleared [{system_id}]")

    def get_current_risk(self, system_id: str) -> CarryoverRiskLevel:
        """Get current carryover risk level for a system."""
        return self._active_risks.get(system_id, CarryoverRiskLevel.NEGLIGIBLE)

    def get_risk_summary(self) -> Dict[str, Any]:
        """Get summary of carryover risks across all systems."""
        risk_counts = {level.value: 0 for level in CarryoverRiskLevel}
        for risk in self._active_risks.values():
            risk_counts[risk.value] += 1

        return {
            "systems_monitored": len(self._active_risks),
            "by_risk_level": risk_counts,
            "total_incidents": len(self._carryover_incidents),
            "recent_incidents": self._carryover_incidents[-5:],
        }


# =============================================================================
# SEPARATOR EVENT HANDLER
# =============================================================================

class SeparatorEventHandler(EventHandler):
    """
    Handler for steam separator performance events.

    Tracks separator efficiency, pressure drops, and maintenance needs.
    """

    def __init__(self, name: str = "SeparatorEventHandler") -> None:
        """Initialize separator event handler."""
        super().__init__(name)
        self._separator_status: Dict[str, Dict] = {}  # separator_id -> status
        self._efficiency_history: Dict[str, List[float]] = {}
        self._maintenance_queue: List[Dict] = []

    async def handle(self, event: SteamQualityEvent) -> None:
        """Handle a separator event."""
        self._record_event(event)
        separator_id = event.payload.get("separator_id", event.system_id or "DEFAULT")

        event_type = event.event_type

        if event_type == EventType.SEPARATOR_EFFICIENCY_LOW:
            await self._handle_low_efficiency(event, separator_id)

        elif event_type == EventType.SEPARATOR_PRESSURE_DROP_HIGH:
            await self._handle_high_pressure_drop(event, separator_id)

        elif event_type == EventType.SEPARATOR_MAINTENANCE_NEEDED:
            await self._handle_maintenance_needed(event, separator_id)

        self._invoke_callbacks(event)
        await self._invoke_async_callbacks(event)

    async def _handle_low_efficiency(
        self,
        event: SteamQualityEvent,
        separator_id: str,
    ) -> None:
        """Handle low separator efficiency."""
        efficiency = event.payload.get("efficiency", 0.0)
        threshold = event.payload.get("threshold", 0.85)

        self._separator_status[separator_id] = {
            "status": "degraded",
            "efficiency": efficiency,
            "last_update": event.timestamp,
        }

        if separator_id not in self._efficiency_history:
            self._efficiency_history[separator_id] = []
        self._efficiency_history[separator_id].append(efficiency)

        logger.warning(
            f"Separator efficiency low [{separator_id}]: {efficiency:.1%} "
            f"(threshold: {threshold:.1%})"
        )

    async def _handle_high_pressure_drop(
        self,
        event: SteamQualityEvent,
        separator_id: str,
    ) -> None:
        """Handle high pressure drop across separator."""
        pressure_drop = event.payload.get("pressure_drop_kpa", 0.0)
        threshold = event.payload.get("threshold_kpa", 20.0)

        logger.warning(
            f"Separator pressure drop high [{separator_id}]: {pressure_drop:.1f} kPa "
            f"(threshold: {threshold:.1f} kPa)"
        )

    async def _handle_maintenance_needed(
        self,
        event: SteamQualityEvent,
        separator_id: str,
    ) -> None:
        """Handle maintenance requirement."""
        maintenance = {
            "separator_id": separator_id,
            "timestamp": event.timestamp,
            "reason": event.payload.get("reason", "Scheduled maintenance"),
            "priority": event.payload.get("priority", "routine"),
            "estimated_downtime_hours": event.payload.get("downtime_hours", 4),
        }
        self._maintenance_queue.append(maintenance)

        logger.info(
            f"Separator maintenance needed [{separator_id}]: {maintenance['reason']}"
        )

    def get_separator_status(self, separator_id: str) -> Optional[Dict]:
        """Get current status of a separator."""
        return self._separator_status.get(separator_id)

    def get_maintenance_queue(self) -> List[Dict]:
        """Get pending maintenance items."""
        return self._maintenance_queue.copy()


# =============================================================================
# CONTROL ACTION EVENT HANDLER
# =============================================================================

class ControlActionEventHandler(EventHandler):
    """
    Handler for control action events.

    Tracks control recommendations, executions, and outcomes.
    """

    def __init__(self, name: str = "ControlActionEventHandler") -> None:
        """Initialize control action event handler."""
        super().__init__(name)
        self._pending_actions: Dict[str, ControlActionEvent] = {}  # action_id -> event
        self._executed_actions: List[Dict] = []
        self._rejected_actions: List[Dict] = []

    async def handle(self, event: SteamQualityEvent) -> None:
        """Handle a control action event."""
        self._record_event(event)

        event_type = event.event_type

        if event_type == EventType.CONTROL_ACTION_RECOMMENDED:
            await self._handle_recommendation(event)

        elif event_type == EventType.CONTROL_ACTION_EXECUTED:
            await self._handle_execution(event)

        elif event_type == EventType.CONTROL_ACTION_REJECTED:
            await self._handle_rejection(event)

        elif event_type == EventType.SETPOINT_CHANGED:
            await self._handle_setpoint_change(event)

        self._invoke_callbacks(event)
        await self._invoke_async_callbacks(event)

    async def _handle_recommendation(self, event: SteamQualityEvent) -> None:
        """Handle control action recommendation."""
        action_id = event.payload.get("action_id", event.event_id)

        if isinstance(event, ControlActionEvent):
            self._pending_actions[action_id] = event
        else:
            control_event = ControlActionEvent(
                event_id=event.event_id,
                event_type=event.event_type,
                timestamp=event.timestamp,
                system_id=event.system_id,
                payload=event.payload,
                action_type=event.payload.get("action_type", "adjustment"),
                target_parameter=event.payload.get("parameter", "unknown"),
                current_value=event.payload.get("current_value", 0.0),
                recommended_value=event.payload.get("recommended_value", 0.0),
                confidence=event.payload.get("confidence", 1.0),
            )
            self._pending_actions[action_id] = control_event

        logger.info(
            f"Control action recommended [{event.system_id}]: "
            f"{event.payload.get('parameter')} -> {event.payload.get('recommended_value')}"
        )

    async def _handle_execution(self, event: SteamQualityEvent) -> None:
        """Handle control action execution."""
        action_id = event.payload.get("action_id")

        executed = {
            "action_id": action_id,
            "timestamp": event.timestamp,
            "system_id": event.system_id,
            "parameter": event.payload.get("parameter"),
            "old_value": event.payload.get("old_value"),
            "new_value": event.payload.get("new_value"),
            "executed_by": event.payload.get("executed_by", "system"),
        }
        self._executed_actions.append(executed)

        # Remove from pending
        if action_id in self._pending_actions:
            del self._pending_actions[action_id]

        logger.info(
            f"Control action executed [{event.system_id}]: "
            f"{executed['parameter']} = {executed['new_value']}"
        )

    async def _handle_rejection(self, event: SteamQualityEvent) -> None:
        """Handle control action rejection."""
        action_id = event.payload.get("action_id")

        rejected = {
            "action_id": action_id,
            "timestamp": event.timestamp,
            "system_id": event.system_id,
            "reason": event.payload.get("reason", "Operator rejected"),
            "rejected_by": event.payload.get("rejected_by", "operator"),
        }
        self._rejected_actions.append(rejected)

        # Remove from pending
        if action_id in self._pending_actions:
            del self._pending_actions[action_id]

        logger.info(
            f"Control action rejected [{event.system_id}]: {rejected['reason']}"
        )

    async def _handle_setpoint_change(self, event: SteamQualityEvent) -> None:
        """Handle setpoint change event."""
        logger.info(
            f"Setpoint changed [{event.system_id}]: "
            f"{event.payload.get('parameter')} = {event.payload.get('new_value')} "
            f"(was: {event.payload.get('old_value')})"
        )

    def get_pending_actions(
        self,
        system_id: Optional[str] = None,
    ) -> List[ControlActionEvent]:
        """Get pending control actions."""
        actions = list(self._pending_actions.values())
        if system_id:
            actions = [a for a in actions if a.system_id == system_id]
        return actions

    def get_action_stats(self) -> Dict[str, Any]:
        """Get control action statistics."""
        return {
            "pending": len(self._pending_actions),
            "executed": len(self._executed_actions),
            "rejected": len(self._rejected_actions),
            "success_rate": (
                len(self._executed_actions) /
                (len(self._executed_actions) + len(self._rejected_actions))
                if (self._executed_actions or self._rejected_actions) else 1.0
            ),
        }


# =============================================================================
# SAFETY EVENT HANDLER
# =============================================================================

class SafetyEventHandler(EventHandler):
    """
    Handler for safety-critical events.

    Manages safety interlocks, emergency shutdowns, and watchdog events
    per IEC 61511 requirements.
    """

    def __init__(
        self,
        name: str = "SafetyEventHandler",
        shutdown_callback: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        """
        Initialize safety event handler.

        Args:
            name: Handler name
            shutdown_callback: Callback for emergency shutdown (system_id, reason)
        """
        super().__init__(name)
        self._shutdown_callback = shutdown_callback
        self._active_interlocks: Dict[str, List[Dict]] = {}
        self._shutdown_history: List[Dict] = []
        self._watchdog_failures: List[Dict] = []

    async def handle(self, event: SteamQualityEvent) -> None:
        """Handle a safety event."""
        self._record_event(event)
        system_id = event.system_id or "DEFAULT"

        event_type = event.event_type

        if event_type == EventType.SAFETY_INTERLOCK_ACTIVATED:
            await self._handle_interlock(event, system_id)

        elif event_type == EventType.EMERGENCY_SHUTDOWN:
            await self._handle_emergency_shutdown(event, system_id)

        elif event_type == EventType.WATCHDOG_TIMEOUT:
            await self._handle_watchdog_timeout(event, system_id)

        self._invoke_callbacks(event)
        await self._invoke_async_callbacks(event)

    async def _handle_interlock(
        self,
        event: SteamQualityEvent,
        system_id: str,
    ) -> None:
        """Handle safety interlock activation."""
        if system_id not in self._active_interlocks:
            self._active_interlocks[system_id] = []

        interlock = {
            "timestamp": event.timestamp,
            "interlock_id": event.payload.get("interlock_id"),
            "reason": event.payload.get("reason"),
            "parameter": event.payload.get("parameter"),
            "value": event.payload.get("value"),
            "threshold": event.payload.get("threshold"),
        }
        self._active_interlocks[system_id].append(interlock)

        logger.critical(
            f"SAFETY INTERLOCK ACTIVATED [{system_id}]: {interlock['reason']}"
        )

    async def _handle_emergency_shutdown(
        self,
        event: SteamQualityEvent,
        system_id: str,
    ) -> None:
        """Handle emergency shutdown event."""
        shutdown = {
            "timestamp": event.timestamp,
            "system_id": system_id,
            "reason": event.payload.get("reason", "Unknown"),
            "initiated_by": event.payload.get("initiated_by", "system"),
            "severity": event.severity.value if isinstance(event.severity, Enum) else str(event.severity),
        }
        self._shutdown_history.append(shutdown)

        logger.critical(
            f"EMERGENCY SHUTDOWN [{system_id}]: {shutdown['reason']}"
        )

        if self._shutdown_callback:
            try:
                self._shutdown_callback(system_id, shutdown["reason"])
            except Exception as e:
                logger.error(f"Shutdown callback failed: {e}")

    async def _handle_watchdog_timeout(
        self,
        event: SteamQualityEvent,
        system_id: str,
    ) -> None:
        """Handle watchdog timeout event."""
        failure = {
            "timestamp": event.timestamp,
            "system_id": system_id,
            "component": event.payload.get("component"),
            "last_heartbeat": event.payload.get("last_heartbeat"),
        }
        self._watchdog_failures.append(failure)

        logger.error(
            f"WATCHDOG TIMEOUT [{system_id}]: Component {failure['component']}"
        )

    def get_active_interlocks(
        self,
        system_id: Optional[str] = None,
    ) -> List[Dict]:
        """Get active safety interlocks."""
        if system_id:
            return self._active_interlocks.get(system_id, [])

        all_interlocks = []
        for interlocks in self._active_interlocks.values():
            all_interlocks.extend(interlocks)
        return all_interlocks

    def clear_interlock(
        self,
        system_id: str,
        interlock_id: str,
        user: str,
    ) -> bool:
        """
        Clear a safety interlock (requires authorization).

        Args:
            system_id: System identifier
            interlock_id: Interlock to clear
            user: User performing clearance

        Returns:
            True if cleared successfully
        """
        if system_id not in self._active_interlocks:
            return False

        original_count = len(self._active_interlocks[system_id])
        self._active_interlocks[system_id] = [
            i for i in self._active_interlocks[system_id]
            if i.get("interlock_id") != interlock_id
        ]

        if len(self._active_interlocks[system_id]) < original_count:
            logger.info(f"Interlock {interlock_id} cleared by {user}")
            return True
        return False

    def get_safety_summary(self) -> Dict[str, Any]:
        """Get safety system summary."""
        return {
            "active_interlocks": sum(
                len(i) for i in self._active_interlocks.values()
            ),
            "total_shutdowns": len(self._shutdown_history),
            "watchdog_failures": len(self._watchdog_failures),
            "systems_with_interlocks": list(self._active_interlocks.keys()),
            "recent_shutdowns": self._shutdown_history[-5:],
        }


# =============================================================================
# AUDIT EVENT HANDLER
# =============================================================================

class AuditEventHandler(EventHandler):
    """
    Handler for audit and provenance events.

    Maintains complete audit trail for regulatory compliance
    and zero-hallucination verification.
    """

    def __init__(
        self,
        name: str = "AuditEventHandler",
        persist_callback: Optional[Callable[[Dict], None]] = None,
    ) -> None:
        """
        Initialize audit event handler.

        Args:
            name: Handler name
            persist_callback: Optional callback to persist records
        """
        super().__init__(name)
        self._persist_callback = persist_callback
        self._calculation_records: List[CalculationEvent] = []
        self._provenance_records: List[Dict] = []

    async def handle(self, event: SteamQualityEvent) -> None:
        """Handle an audit event."""
        self._record_event(event)

        event_type = event.event_type

        if event_type == EventType.CALCULATION_COMPLETED:
            await self._handle_calculation(event)

        elif event_type == EventType.PROVENANCE_RECORDED:
            await self._handle_provenance(event)

        elif event_type == EventType.AUDIT_CHECKPOINT:
            await self._handle_checkpoint(event)

        self._invoke_callbacks(event)
        await self._invoke_async_callbacks(event)

    async def _handle_calculation(self, event: SteamQualityEvent) -> None:
        """Handle calculation audit event."""
        if isinstance(event, CalculationEvent):
            calc = event
        else:
            calc = CalculationEvent(
                event_id=event.event_id,
                event_type=event.event_type,
                timestamp=event.timestamp,
                system_id=event.system_id,
                payload=event.payload,
                calculation_type=event.payload.get("calculation_type", "unknown"),
                inputs_hash=event.payload.get("inputs_hash", ""),
                outputs_hash=event.payload.get("outputs_hash", ""),
                formula_id=event.payload.get("formula_id", ""),
                formula_version=event.payload.get("formula_version", "1.0.0"),
                deterministic=event.payload.get("deterministic", True),
                execution_time_ms=event.payload.get("execution_time_ms", 0.0),
            )

        self._calculation_records.append(calc)

        # Persist if callback provided
        if self._persist_callback:
            try:
                self._persist_callback(calc.model_dump())
            except Exception as e:
                logger.error(f"Failed to persist calculation: {e}")

        logger.debug(
            f"Audit: Calculation {calc.formula_id} recorded, "
            f"hash={calc.inputs_hash[:8]}..."
        )

    async def _handle_provenance(self, event: SteamQualityEvent) -> None:
        """Handle provenance record event."""
        record = {
            "timestamp": event.timestamp,
            "event_id": event.event_id,
            "provenance_hash": event.provenance_hash or event.compute_hash(),
            "data": event.payload,
        }
        self._provenance_records.append(record)

        if self._persist_callback:
            try:
                self._persist_callback(record)
            except Exception as e:
                logger.error(f"Failed to persist provenance: {e}")

    async def _handle_checkpoint(self, event: SteamQualityEvent) -> None:
        """Handle audit checkpoint event."""
        logger.info(
            f"Audit checkpoint [{event.system_id}]: "
            f"{len(self._calculation_records)} calculations, "
            f"{len(self._provenance_records)} provenance records"
        )

    def get_audit_trail(
        self,
        calculation_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[CalculationEvent]:
        """
        Get calculation audit trail with filters.

        Args:
            calculation_type: Filter by calculation type
            start_time: Filter by start time
            end_time: Filter by end time
            limit: Maximum records to return

        Returns:
            List of matching calculation events
        """
        records = self._calculation_records

        if calculation_type:
            records = [r for r in records if r.calculation_type == calculation_type]
        if start_time:
            records = [r for r in records if r.timestamp >= start_time]
        if end_time:
            records = [r for r in records if r.timestamp <= end_time]

        return records[-limit:]

    def verify_provenance(self, provenance_hash: str) -> Optional[Dict]:
        """
        Find a record by provenance hash.

        Args:
            provenance_hash: Hash to search for

        Returns:
            Record if found, None otherwise
        """
        for record in self._provenance_records:
            if record.get("provenance_hash") == provenance_hash:
                return record
        return None

    def get_audit_stats(self) -> Dict[str, Any]:
        """Get audit statistics."""
        calc_types: Dict[str, int] = {}
        for calc in self._calculation_records:
            calc_types[calc.calculation_type] = calc_types.get(calc.calculation_type, 0) + 1

        return {
            "total_calculations": len(self._calculation_records),
            "total_provenance_records": len(self._provenance_records),
            "calculations_by_type": calc_types,
            "deterministic_count": sum(
                1 for c in self._calculation_records if c.deterministic
            ),
        }


# =============================================================================
# EVENT DISPATCHER
# =============================================================================

class EventDispatcher:
    """
    Central dispatcher for routing events to appropriate handlers.

    Manages multiple handlers and routes events based on type.

    Example:
        >>> dispatcher = EventDispatcher()
        >>> dispatcher.register_handler(QualityEventHandler())
        >>> dispatcher.register_handler(SafetyEventHandler())
        >>> await dispatcher.dispatch(event)
    """

    def __init__(self) -> None:
        """Initialize the event dispatcher."""
        self._handlers: Dict[str, List[EventHandler]] = {}
        self._global_handlers: List[EventHandler] = []

    def register_handler(
        self,
        handler: EventHandler,
        event_types: Optional[List[EventType]] = None,
    ) -> None:
        """
        Register an event handler.

        Args:
            handler: Handler to register
            event_types: Event types to handle (None for all)
        """
        if event_types is None:
            self._global_handlers.append(handler)
        else:
            for event_type in event_types:
                type_key = event_type.value
                if type_key not in self._handlers:
                    self._handlers[type_key] = []
                self._handlers[type_key].append(handler)

        logger.debug(f"Registered handler: {handler.name}")

    async def dispatch(self, event: SteamQualityEvent) -> None:
        """
        Dispatch an event to registered handlers.

        Args:
            event: Event to dispatch
        """
        event_type = event.event_type.value if isinstance(event.event_type, Enum) else event.event_type

        # Get type-specific handlers
        handlers = self._handlers.get(event_type, [])

        # Add global handlers
        handlers.extend(self._global_handlers)

        # Dispatch to all handlers
        for handler in handlers:
            try:
                await handler.handle(event)
            except Exception as e:
                logger.error(
                    f"Handler {handler.name} failed for {event_type}: {e}",
                    exc_info=True
                )

    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics from all handlers."""
        stats = {}
        all_handlers = list(self._global_handlers)
        for handlers in self._handlers.values():
            all_handlers.extend(handlers)

        for handler in set(all_handlers):
            stats[handler.name] = handler.get_stats()

        return stats
