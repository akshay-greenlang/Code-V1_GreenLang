"""
GL-002 FLAMEGUARD BoilerEfficiencyOptimizer - Event Handlers

This module provides event handlers for various FLAMEGUARD events
including safety events, optimization events, combustion events,
and compliance/audit events.

All handlers support async processing and maintain audit trails.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Awaitable
import asyncio
import logging

from pydantic import BaseModel, Field

from .schemas import (
    FlameguardEvent,
    SafetyEvent,
    CalculationEvent,
    BoilerProcessData,
    OptimizationResult,
    EfficiencyCalculation,
    EmissionsCalculation,
    SeverityLevel,
    CalculationType,
    TripType,
)

logger = logging.getLogger(__name__)


# =============================================================================
# BASE EVENT HANDLER
# =============================================================================

class EventHandler(ABC):
    """
    Base class for event handlers.

    Event handlers process events from the FLAMEGUARD agent and
    trigger appropriate actions.
    """

    def __init__(self, name: str) -> None:
        """
        Initialize the event handler.

        Args:
            name: Handler name
        """
        self.name = name
        self._callbacks: Dict[str, List[Callable]] = {}
        self._async_callbacks: Dict[str, List[Callable[..., Awaitable[None]]]] = {}
        self._event_count = 0
        self._error_count = 0
        self._last_event_time: Optional[datetime] = None

    @abstractmethod
    async def handle(self, event: FlameguardEvent) -> None:
        """
        Handle an event.

        Args:
            event: Event to handle
        """
        pass

    def register_callback(
        self,
        event_type: str,
        callback: Callable[[FlameguardEvent], None],
    ) -> None:
        """Register a synchronous callback for an event type."""
        if event_type not in self._callbacks:
            self._callbacks[event_type] = []
        self._callbacks[event_type].append(callback)

    def register_async_callback(
        self,
        event_type: str,
        callback: Callable[..., Awaitable[None]],
    ) -> None:
        """Register an async callback for an event type."""
        if event_type not in self._async_callbacks:
            self._async_callbacks[event_type] = []
        self._async_callbacks[event_type].append(callback)

    def _invoke_callbacks(self, event: FlameguardEvent) -> None:
        """Invoke registered synchronous callbacks for an event."""
        callbacks = self._callbacks.get(event.event_type, [])
        callbacks.extend(self._callbacks.get("*", []))  # Wildcard callbacks

        for callback in callbacks:
            try:
                callback(event)
            except Exception as e:
                self._error_count += 1
                logger.error(f"Callback error in {self.name}: {e}")

    async def _invoke_async_callbacks(self, event: FlameguardEvent) -> None:
        """Invoke registered async callbacks for an event."""
        callbacks = self._async_callbacks.get(event.event_type, [])
        callbacks.extend(self._async_callbacks.get("*", []))

        for callback in callbacks:
            try:
                await callback(event)
            except Exception as e:
                self._error_count += 1
                logger.error(f"Async callback error in {self.name}: {e}")

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
            "last_event_time": self._last_event_time.isoformat() if self._last_event_time else None,
            "callback_count": sum(len(v) for v in self._callbacks.values()),
            "async_callback_count": sum(len(v) for v in self._async_callbacks.values()),
        }


# =============================================================================
# SAFETY EVENT HANDLER
# =============================================================================

class BoilerSafetyEventHandler(EventHandler):
    """
    Handler for boiler safety-related events.

    This handler processes safety events and can trigger
    emergency actions when necessary per NFPA 85.
    """

    def __init__(
        self,
        name: str = "BoilerSafetyHandler",
        trip_callback: Optional[Callable[[str, str], None]] = None,
        alarm_callback: Optional[Callable[[SafetyEvent], None]] = None,
    ) -> None:
        """
        Initialize the boiler safety event handler.

        Args:
            name: Handler name
            trip_callback: Callback to trigger boiler trip (boiler_id, reason)
            alarm_callback: Callback for alarm notification
        """
        super().__init__(name)
        self._trip_callback = trip_callback
        self._alarm_callback = alarm_callback
        self._active_alarms: Dict[str, List[SafetyEvent]] = {}  # boiler_id -> alarms
        self._alarm_history: List[SafetyEvent] = []
        self._trip_history: List[Dict[str, Any]] = []

    async def handle(self, event: FlameguardEvent) -> None:
        """
        Handle a safety event.

        Args:
            event: Event to handle
        """
        self._event_count += 1
        self._last_event_time = datetime.now(timezone.utc)

        boiler_id = event.boiler_id or "UNKNOWN"

        # Convert to SafetyEvent if needed
        safety_event = SafetyEvent(
            event_type=event.event_type,
            boiler_id=boiler_id,
            severity=event.severity if hasattr(event, 'severity') else SeverityLevel.WARNING,
            description=event.payload.get("description", str(event.payload)),
            source_tag=event.payload.get("source_tag"),
            measured_value=event.payload.get("measured_value"),
            threshold_value=event.payload.get("threshold_value"),
            unit=event.payload.get("unit"),
        )

        logger.warning(
            f"Boiler safety event [{boiler_id}]: {safety_event.event_type} - "
            f"{safety_event.description}"
        )

        # Track alarm
        if boiler_id not in self._active_alarms:
            self._active_alarms[boiler_id] = []
        self._active_alarms[boiler_id].append(safety_event)
        self._alarm_history.append(safety_event)

        # Notify alarm callback
        if self._alarm_callback:
            try:
                self._alarm_callback(safety_event)
            except Exception as e:
                logger.error(f"Alarm callback error: {e}")

        # Check for trip conditions
        severity = event.payload.get("severity", "warning")
        if severity == "critical" or safety_event.severity == SeverityLevel.CRITICAL:
            await self._handle_critical_event(safety_event)

        # Check for emergency conditions
        if safety_event.severity == SeverityLevel.EMERGENCY:
            await self._handle_emergency_event(safety_event)

        # Invoke callbacks
        self._invoke_callbacks(event)
        await self._invoke_async_callbacks(event)

    async def _handle_critical_event(self, event: SafetyEvent) -> None:
        """Handle a critical safety event."""
        logger.critical(f"CRITICAL SAFETY EVENT [{event.boiler_id}]: {event.description}")

        # Trip triggers per NFPA 85
        trip_triggers = [
            "FLAME_FAILURE",
            "LOW_WATER",
            "HIGH_PRESSURE",
            "HIGH_TEMPERATURE",
            "FUEL_LEAK",
            "FAN_FAILURE",
            "PURGE_FAILURE",
        ]

        if any(trigger in event.event_type.upper() for trigger in trip_triggers):
            if self._trip_callback:
                logger.critical(f"Triggering boiler trip for {event.boiler_id}")
                self._trip_callback(event.boiler_id, event.description)

            self._trip_history.append({
                "boiler_id": event.boiler_id,
                "timestamp": datetime.now(timezone.utc),
                "reason": event.description,
                "event_type": event.event_type,
            })

    async def _handle_emergency_event(self, event: SafetyEvent) -> None:
        """Handle an emergency safety event - immediate action required."""
        logger.critical(f"EMERGENCY [{event.boiler_id}]: {event.description}")

        # Always trigger trip for emergency events
        if self._trip_callback:
            self._trip_callback(event.boiler_id, f"EMERGENCY: {event.description}")

    def acknowledge_alarm(
        self,
        boiler_id: str,
        event_id: str,
        user: str,
    ) -> bool:
        """
        Acknowledge an alarm.

        Args:
            boiler_id: Boiler ID
            event_id: Event ID to acknowledge
            user: User acknowledging

        Returns:
            True if acknowledged successfully
        """
        alarms = self._active_alarms.get(boiler_id, [])
        for alarm in alarms:
            if alarm.event_id == event_id:
                alarm.acknowledged = True
                alarm.acknowledged_by = user
                alarm.acknowledged_at = datetime.now(timezone.utc)
                logger.info(f"Alarm {event_id} acknowledged by {user}")
                return True
        return False

    def clear_alarm(self, boiler_id: str, event_id: str) -> bool:
        """
        Clear an alarm.

        Args:
            boiler_id: Boiler ID
            event_id: Event ID to clear

        Returns:
            True if cleared successfully
        """
        if boiler_id in self._active_alarms:
            self._active_alarms[boiler_id] = [
                a for a in self._active_alarms[boiler_id]
                if a.event_id != event_id
            ]
            return True
        return False

    def get_active_alarms(self, boiler_id: Optional[str] = None) -> List[SafetyEvent]:
        """Get active alarms, optionally filtered by boiler."""
        if boiler_id:
            return list(self._active_alarms.get(boiler_id, []))

        all_alarms = []
        for alarms in self._active_alarms.values():
            all_alarms.extend(alarms)
        return all_alarms

    def get_alarm_summary(self, boiler_id: Optional[str] = None) -> Dict[str, Any]:
        """Get alarm summary."""
        alarms = self.get_active_alarms(boiler_id)

        severity_counts = {"info": 0, "warning": 0, "critical": 0, "emergency": 0}
        for alarm in alarms:
            sev = alarm.severity.value if hasattr(alarm.severity, 'value') else alarm.severity
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        return {
            "total_active": len(alarms),
            "by_severity": severity_counts,
            "total_historical": len(self._alarm_history),
            "unacknowledged": sum(1 for a in alarms if not a.acknowledged),
            "total_trips": len(self._trip_history),
        }


# =============================================================================
# COMBUSTION EVENT HANDLER
# =============================================================================

class CombustionEventHandler(EventHandler):
    """
    Handler for combustion-related events.

    Tracks combustion quality, O2 deviations, CO breakthroughs,
    and flame events.
    """

    def __init__(self, name: str = "CombustionHandler") -> None:
        """Initialize the combustion event handler."""
        super().__init__(name)
        self._o2_deviations: Dict[str, List[Dict]] = {}  # boiler_id -> deviations
        self._co_breakthroughs: Dict[str, List[Dict]] = {}
        self._flame_events: Dict[str, List[Dict]] = {}
        self._combustion_quality_scores: Dict[str, List[float]] = {}

    async def handle(self, event: FlameguardEvent) -> None:
        """
        Handle a combustion event.

        Args:
            event: Event to handle
        """
        self._event_count += 1
        self._last_event_time = datetime.now(timezone.utc)

        boiler_id = event.boiler_id or "UNKNOWN"
        event_type = event.event_type.upper()

        # Initialize boiler tracking if needed
        for tracking_dict in [
            self._o2_deviations,
            self._co_breakthroughs,
            self._flame_events,
            self._combustion_quality_scores,
        ]:
            if boiler_id not in tracking_dict:
                tracking_dict[boiler_id] = []

        # Handle specific combustion event types
        if "O2_DEVIATION" in event_type:
            self._o2_deviations[boiler_id].append({
                "timestamp": event.timestamp,
                "deviation": event.payload.get("deviation", 0),
                "measured": event.payload.get("measured_o2"),
                "target": event.payload.get("target_o2"),
            })
            logger.info(f"O2 deviation recorded for {boiler_id}")

        elif "CO_BREAKTHROUGH" in event_type:
            self._co_breakthroughs[boiler_id].append({
                "timestamp": event.timestamp,
                "co_ppm": event.payload.get("co_ppm"),
                "duration_s": event.payload.get("duration_s"),
            })
            logger.warning(f"CO breakthrough detected for {boiler_id}")

        elif "FLAME" in event_type:
            self._flame_events[boiler_id].append({
                "timestamp": event.timestamp,
                "event_type": event_type,
                "flame_signal": event.payload.get("flame_signal"),
                "description": event.payload.get("description"),
            })
            if "FAILURE" in event_type or "LOSS" in event_type:
                logger.critical(f"Flame event for {boiler_id}: {event_type}")
            else:
                logger.info(f"Flame event for {boiler_id}: {event_type}")

        elif "COMBUSTION_QUALITY" in event_type:
            score = event.payload.get("quality_score", 100.0)
            self._combustion_quality_scores[boiler_id].append(score)
            logger.debug(f"Combustion quality for {boiler_id}: {score}")

        # Invoke callbacks
        self._invoke_callbacks(event)
        await self._invoke_async_callbacks(event)

    def get_combustion_summary(self, boiler_id: str) -> Dict[str, Any]:
        """Get combustion summary for a boiler."""
        quality_scores = self._combustion_quality_scores.get(boiler_id, [])
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 100.0

        return {
            "boiler_id": boiler_id,
            "o2_deviation_count": len(self._o2_deviations.get(boiler_id, [])),
            "co_breakthrough_count": len(self._co_breakthroughs.get(boiler_id, [])),
            "flame_event_count": len(self._flame_events.get(boiler_id, [])),
            "avg_combustion_quality": round(avg_quality, 1),
            "quality_data_points": len(quality_scores),
        }


# =============================================================================
# OPTIMIZATION EVENT HANDLER
# =============================================================================

class OptimizationEventHandler(EventHandler):
    """
    Handler for optimization-related events.

    Tracks optimization requests, results, and recommendations.
    """

    def __init__(self, name: str = "OptimizationHandler") -> None:
        """Initialize the optimization event handler."""
        super().__init__(name)
        self._optimization_history: Dict[str, List[Dict]] = {}  # boiler_id -> results
        self._recommendation_history: Dict[str, List[Dict]] = {}
        self._active_optimizations: Dict[str, str] = {}  # boiler_id -> optimization_id

    async def handle(self, event: FlameguardEvent) -> None:
        """
        Handle an optimization event.

        Args:
            event: Event to handle
        """
        self._event_count += 1
        self._last_event_time = datetime.now(timezone.utc)

        boiler_id = event.boiler_id or "UNKNOWN"
        event_type = event.event_type.upper()

        # Initialize tracking
        if boiler_id not in self._optimization_history:
            self._optimization_history[boiler_id] = []
        if boiler_id not in self._recommendation_history:
            self._recommendation_history[boiler_id] = []

        if "OPTIMIZATION_STARTED" in event_type:
            optimization_id = event.payload.get("optimization_id", str(event.event_id))
            self._active_optimizations[boiler_id] = optimization_id
            logger.info(f"Optimization started for {boiler_id}: {optimization_id}")

        elif "OPTIMIZATION_COMPLETED" in event_type:
            result = {
                "timestamp": event.timestamp,
                "optimization_id": event.payload.get("optimization_id"),
                "efficiency_improvement": event.payload.get("efficiency_improvement", 0),
                "cost_savings": event.payload.get("cost_savings", 0),
                "emissions_reduction": event.payload.get("emissions_reduction", 0),
                "status": "completed",
            }
            self._optimization_history[boiler_id].append(result)
            self._active_optimizations.pop(boiler_id, None)
            logger.info(
                f"Optimization completed for {boiler_id}: "
                f"+{result['efficiency_improvement']:.2f}% efficiency"
            )

        elif "OPTIMIZATION_FAILED" in event_type:
            result = {
                "timestamp": event.timestamp,
                "optimization_id": event.payload.get("optimization_id"),
                "error": event.payload.get("error"),
                "status": "failed",
            }
            self._optimization_history[boiler_id].append(result)
            self._active_optimizations.pop(boiler_id, None)
            logger.error(f"Optimization failed for {boiler_id}: {result['error']}")

        elif "RECOMMENDATION" in event_type:
            recommendation = {
                "timestamp": event.timestamp,
                "setpoint_changes": event.payload.get("setpoint_changes", []),
                "expected_benefit": event.payload.get("expected_benefit"),
                "confidence": event.payload.get("confidence", 95),
                "implemented": event.payload.get("implemented", False),
            }
            self._recommendation_history[boiler_id].append(recommendation)
            logger.info(f"Recommendation for {boiler_id}: {len(recommendation['setpoint_changes'])} changes")

        # Invoke callbacks
        self._invoke_callbacks(event)
        await self._invoke_async_callbacks(event)

    def get_optimization_stats(self, boiler_id: str) -> Dict[str, Any]:
        """Get optimization statistics for a boiler."""
        history = self._optimization_history.get(boiler_id, [])
        completed = [h for h in history if h.get("status") == "completed"]
        failed = [h for h in history if h.get("status") == "failed"]

        total_efficiency_gain = sum(h.get("efficiency_improvement", 0) for h in completed)
        total_cost_savings = sum(h.get("cost_savings", 0) for h in completed)
        total_emissions_reduction = sum(h.get("emissions_reduction", 0) for h in completed)

        return {
            "boiler_id": boiler_id,
            "total_optimizations": len(history),
            "successful": len(completed),
            "failed": len(failed),
            "success_rate": len(completed) / len(history) * 100 if history else 0,
            "total_efficiency_gain_percent": round(total_efficiency_gain, 2),
            "total_cost_savings_usd": round(total_cost_savings, 2),
            "total_emissions_reduction_kg": round(total_emissions_reduction, 2),
            "active_optimization": self._active_optimizations.get(boiler_id),
            "recommendations_count": len(self._recommendation_history.get(boiler_id, [])),
        }


# =============================================================================
# EFFICIENCY EVENT HANDLER
# =============================================================================

class EfficiencyEventHandler(EventHandler):
    """
    Handler for efficiency calculation events.

    Tracks efficiency calculations and trends.
    """

    def __init__(self, name: str = "EfficiencyHandler") -> None:
        """Initialize the efficiency event handler."""
        super().__init__(name)
        self._efficiency_history: Dict[str, List[Dict]] = {}  # boiler_id -> calculations
        self._efficiency_alerts: Dict[str, List[Dict]] = {}

    async def handle(self, event: FlameguardEvent) -> None:
        """
        Handle an efficiency event.

        Args:
            event: Event to handle
        """
        self._event_count += 1
        self._last_event_time = datetime.now(timezone.utc)

        boiler_id = event.boiler_id or "UNKNOWN"

        # Initialize tracking
        if boiler_id not in self._efficiency_history:
            self._efficiency_history[boiler_id] = []
        if boiler_id not in self._efficiency_alerts:
            self._efficiency_alerts[boiler_id] = []

        if "EFFICIENCY_CALCULATED" in event.event_type.upper():
            calculation = {
                "timestamp": event.timestamp,
                "efficiency_percent": event.payload.get("efficiency_percent"),
                "total_losses_percent": event.payload.get("total_losses_percent"),
                "fuel_input_mmbtu_hr": event.payload.get("fuel_input_mmbtu_hr"),
                "steam_output_mmbtu_hr": event.payload.get("steam_output_mmbtu_hr"),
            }
            self._efficiency_history[boiler_id].append(calculation)

            # Keep only last 1000 calculations per boiler
            if len(self._efficiency_history[boiler_id]) > 1000:
                self._efficiency_history[boiler_id] = self._efficiency_history[boiler_id][-1000:]

            logger.debug(f"Efficiency calculated for {boiler_id}: {calculation['efficiency_percent']:.1f}%")

        elif "EFFICIENCY_ALERT" in event.event_type.upper():
            alert = {
                "timestamp": event.timestamp,
                "current_efficiency": event.payload.get("current_efficiency"),
                "threshold": event.payload.get("threshold"),
                "deviation": event.payload.get("deviation"),
                "severity": event.payload.get("severity", "warning"),
            }
            self._efficiency_alerts[boiler_id].append(alert)
            logger.warning(
                f"Efficiency alert for {boiler_id}: "
                f"{alert['current_efficiency']:.1f}% (threshold: {alert['threshold']:.1f}%)"
            )

        # Invoke callbacks
        self._invoke_callbacks(event)
        await self._invoke_async_callbacks(event)

    def get_efficiency_trend(
        self,
        boiler_id: str,
        periods: int = 10,
    ) -> Dict[str, Any]:
        """Get efficiency trend for a boiler."""
        history = self._efficiency_history.get(boiler_id, [])

        if not history:
            return {"boiler_id": boiler_id, "data_points": 0, "trend": "unknown"}

        recent = history[-periods:] if len(history) >= periods else history
        efficiencies = [h["efficiency_percent"] for h in recent if h.get("efficiency_percent")]

        if len(efficiencies) < 2:
            trend = "insufficient_data"
        else:
            first_half = efficiencies[:len(efficiencies)//2]
            second_half = efficiencies[len(efficiencies)//2:]
            avg_first = sum(first_half) / len(first_half)
            avg_second = sum(second_half) / len(second_half)

            if avg_second > avg_first + 0.5:
                trend = "improving"
            elif avg_second < avg_first - 0.5:
                trend = "declining"
            else:
                trend = "stable"

        return {
            "boiler_id": boiler_id,
            "data_points": len(efficiencies),
            "trend": trend,
            "current_efficiency": efficiencies[-1] if efficiencies else None,
            "avg_efficiency": sum(efficiencies) / len(efficiencies) if efficiencies else None,
            "min_efficiency": min(efficiencies) if efficiencies else None,
            "max_efficiency": max(efficiencies) if efficiencies else None,
            "alert_count": len(self._efficiency_alerts.get(boiler_id, [])),
        }


# =============================================================================
# EMISSIONS EVENT HANDLER
# =============================================================================

class EmissionsEventHandler(EventHandler):
    """
    Handler for emissions-related events.

    Tracks emissions calculations and compliance events.
    """

    def __init__(self, name: str = "EmissionsHandler") -> None:
        """Initialize the emissions event handler."""
        super().__init__(name)
        self._emissions_history: Dict[str, List[Dict]] = {}
        self._compliance_violations: Dict[str, List[Dict]] = {}
        self._ghg_totals: Dict[str, float] = {}  # boiler_id -> cumulative CO2e

    async def handle(self, event: FlameguardEvent) -> None:
        """
        Handle an emissions event.

        Args:
            event: Event to handle
        """
        self._event_count += 1
        self._last_event_time = datetime.now(timezone.utc)

        boiler_id = event.boiler_id or "UNKNOWN"
        event_type = event.event_type.upper()

        # Initialize tracking
        if boiler_id not in self._emissions_history:
            self._emissions_history[boiler_id] = []
        if boiler_id not in self._compliance_violations:
            self._compliance_violations[boiler_id] = []
        if boiler_id not in self._ghg_totals:
            self._ghg_totals[boiler_id] = 0.0

        if "EMISSIONS_CALCULATED" in event_type:
            calculation = {
                "timestamp": event.timestamp,
                "co2_lb_hr": event.payload.get("co2_lb_hr"),
                "nox_lb_hr": event.payload.get("nox_lb_hr"),
                "co_lb_hr": event.payload.get("co_lb_hr"),
                "so2_lb_hr": event.payload.get("so2_lb_hr"),
                "co2e_metric_tons_hr": event.payload.get("co2e_metric_tons_hr", 0),
            }
            self._emissions_history[boiler_id].append(calculation)

            # Update cumulative GHG
            self._ghg_totals[boiler_id] += calculation.get("co2e_metric_tons_hr", 0)

            # Keep only last 1000
            if len(self._emissions_history[boiler_id]) > 1000:
                self._emissions_history[boiler_id] = self._emissions_history[boiler_id][-1000:]

        elif "COMPLIANCE_VIOLATION" in event_type:
            violation = {
                "timestamp": event.timestamp,
                "pollutant": event.payload.get("pollutant"),
                "limit": event.payload.get("limit"),
                "measured": event.payload.get("measured"),
                "exceedance_percent": event.payload.get("exceedance_percent"),
                "standard": event.payload.get("standard"),
            }
            self._compliance_violations[boiler_id].append(violation)
            logger.warning(
                f"Emissions compliance violation for {boiler_id}: "
                f"{violation['pollutant']} exceeded by {violation['exceedance_percent']:.1f}%"
            )

        # Invoke callbacks
        self._invoke_callbacks(event)
        await self._invoke_async_callbacks(event)

    def get_emissions_summary(self, boiler_id: str) -> Dict[str, Any]:
        """Get emissions summary for a boiler."""
        history = self._emissions_history.get(boiler_id, [])
        violations = self._compliance_violations.get(boiler_id, [])

        if not history:
            return {"boiler_id": boiler_id, "data_points": 0}

        # Calculate averages from recent data
        recent = history[-100:]  # Last 100 readings
        avg_co2 = sum(h.get("co2_lb_hr", 0) for h in recent) / len(recent)
        avg_nox = sum(h.get("nox_lb_hr", 0) for h in recent) / len(recent)
        avg_co = sum(h.get("co_lb_hr", 0) for h in recent) / len(recent)

        return {
            "boiler_id": boiler_id,
            "data_points": len(history),
            "avg_co2_lb_hr": round(avg_co2, 2),
            "avg_nox_lb_hr": round(avg_nox, 4),
            "avg_co_lb_hr": round(avg_co, 4),
            "cumulative_co2e_metric_tons": round(self._ghg_totals.get(boiler_id, 0), 2),
            "compliance_violation_count": len(violations),
            "recent_violations": violations[-5:] if violations else [],
        }


# =============================================================================
# AUDIT EVENT HANDLER
# =============================================================================

class AuditEventHandler(EventHandler):
    """
    Handler for audit and provenance events.

    Maintains complete audit trail of all calculations and actions.
    """

    def __init__(
        self,
        name: str = "AuditHandler",
        persist_callback: Optional[Callable[[CalculationEvent], None]] = None,
    ) -> None:
        """Initialize the audit event handler."""
        super().__init__(name)
        self._calculation_events: List[CalculationEvent] = []
        self._action_events: List[Dict] = []
        self._persist_callback = persist_callback

    async def handle(self, event: FlameguardEvent) -> None:
        """
        Handle an audit event.

        Args:
            event: Event to handle
        """
        self._event_count += 1
        self._last_event_time = datetime.now(timezone.utc)

        event_type = event.event_type.upper()

        if "CALCULATION" in event_type:
            # Parse calculation event
            calc_type_str = event.payload.get("calculation_type", "efficiency")
            try:
                calc_type = CalculationType(calc_type_str)
            except ValueError:
                calc_type = CalculationType.EFFICIENCY

            calc_event = CalculationEvent(
                calculation_type=calc_type,
                boiler_id=event.boiler_id or "UNKNOWN",
                input_summary=event.payload.get("input_summary", {}),
                input_hash=event.payload.get("input_hash", ""),
                output_summary=event.payload.get("output_summary", {}),
                output_hash=event.payload.get("output_hash", ""),
                formula_id=event.payload.get("formula_id", ""),
                formula_version=event.payload.get("formula_version", "1.0.0"),
                deterministic=event.payload.get("deterministic", True),
                calculation_time_ms=event.payload.get("calculation_time_ms", 0),
            )
            self._calculation_events.append(calc_event)

            # Persist if callback provided
            if self._persist_callback:
                try:
                    self._persist_callback(calc_event)
                except Exception as e:
                    logger.error(f"Failed to persist calculation event: {e}")

            logger.debug(f"Audit: Calculation {calc_event.formula_id} recorded")

        elif "ACTION" in event_type or "SETPOINT" in event_type:
            action_event = {
                "timestamp": event.timestamp,
                "event_type": event.event_type,
                "boiler_id": event.boiler_id,
                "action": event.payload.get("action"),
                "user": event.payload.get("user"),
                "details": event.payload,
            }
            self._action_events.append(action_event)
            logger.info(f"Audit: Action {event.event_type} recorded")

        # Keep audit trail bounded
        if len(self._calculation_events) > 10000:
            self._calculation_events = self._calculation_events[-10000:]
        if len(self._action_events) > 10000:
            self._action_events = self._action_events[-10000:]

        # Invoke callbacks
        self._invoke_callbacks(event)
        await self._invoke_async_callbacks(event)

    def get_audit_trail(
        self,
        boiler_id: Optional[str] = None,
        calculation_type: Optional[CalculationType] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[CalculationEvent]:
        """Get calculation audit trail with filters."""
        events = self._calculation_events

        if boiler_id:
            events = [e for e in events if e.boiler_id == boiler_id]
        if calculation_type:
            events = [e for e in events if e.calculation_type == calculation_type]
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]

        return events[-limit:]

    def get_audit_stats(self) -> Dict[str, Any]:
        """Get audit statistics."""
        calc_types = {}
        for event in self._calculation_events:
            ct = event.calculation_type.value if hasattr(event.calculation_type, 'value') else str(event.calculation_type)
            calc_types[ct] = calc_types.get(ct, 0) + 1

        return {
            "total_calculations": len(self._calculation_events),
            "total_actions": len(self._action_events),
            "calculations_by_type": calc_types,
            "deterministic_count": sum(1 for e in self._calculation_events if e.deterministic),
        }


# =============================================================================
# METRICS EVENT HANDLER
# =============================================================================

class MetricsEventHandler(EventHandler):
    """
    Handler for metrics events.

    Collects and aggregates metrics for Prometheus export.
    """

    def __init__(self, name: str = "MetricsHandler") -> None:
        """Initialize the metrics event handler."""
        super().__init__(name)
        self._gauges: Dict[str, float] = {}
        self._counters: Dict[str, int] = {}
        self._histograms: Dict[str, List[float]] = {}

    async def handle(self, event: FlameguardEvent) -> None:
        """
        Handle a metrics event.

        Args:
            event: Event to handle
        """
        self._event_count += 1
        self._last_event_time = datetime.now(timezone.utc)

        # Extract metrics from payload
        gauges = event.payload.get("gauges", {})
        counters = event.payload.get("counters", {})
        histograms = event.payload.get("histograms", {})

        # Update gauges (overwrite)
        for name, value in gauges.items():
            if isinstance(value, (int, float)):
                self._gauges[name] = float(value)

        # Update counters (increment)
        for name, delta in counters.items():
            if isinstance(delta, (int, float)):
                self._counters[name] = self._counters.get(name, 0) + int(delta)

        # Update histograms (append)
        for name, value in histograms.items():
            if isinstance(value, (int, float)):
                if name not in self._histograms:
                    self._histograms[name] = []
                self._histograms[name].append(float(value))
                # Keep bounded
                if len(self._histograms[name]) > 1000:
                    self._histograms[name] = self._histograms[name][-1000:]

        # Invoke callbacks
        self._invoke_callbacks(event)
        await self._invoke_async_callbacks(event)

    def get_gauge(self, name: str) -> Optional[float]:
        """Get gauge value."""
        return self._gauges.get(name)

    def get_counter(self, name: str) -> int:
        """Get counter value."""
        return self._counters.get(name, 0)

    def get_histogram_stats(self, name: str) -> Dict[str, float]:
        """Get histogram statistics."""
        values = self._histograms.get(name, [])
        if not values:
            return {}

        sorted_values = sorted(values)
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "p50": sorted_values[len(values) // 2],
            "p95": sorted_values[int(len(values) * 0.95)] if len(values) >= 20 else sorted_values[-1],
            "p99": sorted_values[int(len(values) * 0.99)] if len(values) >= 100 else sorted_values[-1],
        }

    def export_prometheus(self, prefix: str = "flameguard") -> str:
        """Export metrics in Prometheus format."""
        lines = []

        # Export gauges
        for name, value in self._gauges.items():
            metric_name = f"{prefix}_{name}".replace(".", "_").replace("-", "_")
            lines.append(f"# TYPE {metric_name} gauge")
            lines.append(f"{metric_name} {value}")

        # Export counters
        for name, value in self._counters.items():
            metric_name = f"{prefix}_{name}".replace(".", "_").replace("-", "_")
            lines.append(f"# TYPE {metric_name} counter")
            lines.append(f"{metric_name} {value}")

        return "\n".join(lines)

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics."""
        return {
            "gauges": dict(self._gauges),
            "counters": dict(self._counters),
            "histograms": {k: self.get_histogram_stats(k) for k in self._histograms},
        }
