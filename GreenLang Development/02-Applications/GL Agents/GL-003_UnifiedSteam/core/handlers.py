"""
GL-003 UNIFIEDSTEAM SteamSystemOptimizer - Event Handlers

This module provides event handlers for various UNIFIEDSTEAM events
including safety events, optimization events, thermodynamic events,
trap diagnostics events, condensate events, and audit events.

All handlers support async processing and maintain audit trails
for regulatory compliance and provenance tracking.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Awaitable
import asyncio
import logging

from pydantic import BaseModel, Field

from .schemas import (
    SteamSystemEvent,
    AlarmEvent,
    OptimizationEvent,
    SteamProcessData,
    OptimizationResult,
    EnthalpyBalanceResult,
    TrapDiagnosticsResult,
    TrapHealthAssessment,
    DesuperheaterRecommendation,
    CondensateRecoveryResult,
    SteamProperties,
    SeverityLevel,
    AlarmState,
    CalculationType,
    OptimizationType,
    OptimizationStatus,
    TrapFailureMode,
    MaintenancePriority,
)

logger = logging.getLogger(__name__)


# =============================================================================
# BASE EVENT HANDLER
# =============================================================================

class EventHandler(ABC):
    """
    Base class for event handlers.

    Event handlers process events from the UNIFIEDSTEAM agent and
    trigger appropriate actions. All handlers support both sync and
    async callbacks.
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
    async def handle(self, event: SteamSystemEvent) -> None:
        """
        Handle an event.

        Args:
            event: Event to handle
        """
        pass

    def register_callback(
        self,
        event_type: str,
        callback: Callable[[SteamSystemEvent], None],
    ) -> None:
        """
        Register a synchronous callback for an event type.

        Args:
            event_type: Event type to listen for (use "*" for all)
            callback: Callback function
        """
        if event_type not in self._callbacks:
            self._callbacks[event_type] = []
        self._callbacks[event_type].append(callback)

    def register_async_callback(
        self,
        event_type: str,
        callback: Callable[..., Awaitable[None]],
    ) -> None:
        """
        Register an async callback for an event type.

        Args:
            event_type: Event type to listen for
            callback: Async callback function
        """
        if event_type not in self._async_callbacks:
            self._async_callbacks[event_type] = []
        self._async_callbacks[event_type].append(callback)

    def _invoke_callbacks(self, event: SteamSystemEvent) -> None:
        """Invoke registered synchronous callbacks for an event."""
        callbacks = self._callbacks.get(event.event_type, [])
        callbacks.extend(self._callbacks.get("*", []))  # Wildcard callbacks

        for callback in callbacks:
            try:
                callback(event)
            except Exception as e:
                self._error_count += 1
                logger.error(f"Callback error in {self.name}: {e}")

    async def _invoke_async_callbacks(self, event: SteamSystemEvent) -> None:
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
            "last_event_time": (
                self._last_event_time.isoformat() if self._last_event_time else None
            ),
            "callback_count": sum(len(v) for v in self._callbacks.values()),
            "async_callback_count": sum(len(v) for v in self._async_callbacks.values()),
        }


# =============================================================================
# STEAM SAFETY EVENT HANDLER
# =============================================================================

class SteamSafetyEventHandler(EventHandler):
    """
    Handler for steam system safety-related events.

    This handler processes safety events including pressure/temperature
    alarms, equipment trips, and emergency conditions.
    """

    def __init__(
        self,
        name: str = "SteamSafetyHandler",
        trip_callback: Optional[Callable[[str, str], None]] = None,
        alarm_callback: Optional[Callable[[AlarmEvent], None]] = None,
    ) -> None:
        """
        Initialize the steam safety event handler.

        Args:
            name: Handler name
            trip_callback: Callback for trip conditions (system_id, reason)
            alarm_callback: Callback for alarm notifications
        """
        super().__init__(name)
        self._trip_callback = trip_callback
        self._alarm_callback = alarm_callback
        self._active_alarms: Dict[str, List[AlarmEvent]] = {}  # system_id -> alarms
        self._alarm_history: List[AlarmEvent] = []
        self._trip_history: List[Dict[str, Any]] = []

    async def handle(self, event: SteamSystemEvent) -> None:
        """
        Handle a safety event.

        Args:
            event: Event to handle
        """
        self._event_count += 1
        self._last_event_time = datetime.now(timezone.utc)

        system_id = event.system_id or "UNKNOWN"

        # Create alarm event
        alarm_event = AlarmEvent(
            event_type=event.event_type,
            system_id=system_id,
            alarm_type=event.event_type,
            severity=event.severity,
            description=event.payload.get("description", str(event.payload)),
            source_tag=event.payload.get("source_tag"),
            measured_value=event.payload.get("measured_value"),
            threshold_value=event.payload.get("threshold_value"),
            unit=event.payload.get("unit"),
        )

        logger.warning(
            f"Steam safety event [{system_id}]: {alarm_event.alarm_type} - "
            f"{alarm_event.description}"
        )

        # Track alarm
        if system_id not in self._active_alarms:
            self._active_alarms[system_id] = []
        self._active_alarms[system_id].append(alarm_event)
        self._alarm_history.append(alarm_event)

        # Notify alarm callback
        if self._alarm_callback:
            try:
                self._alarm_callback(alarm_event)
            except Exception as e:
                logger.error(f"Alarm callback error: {e}")

        # Check for trip conditions
        if event.severity in [SeverityLevel.CRITICAL, SeverityLevel.EMERGENCY]:
            await self._handle_critical_event(alarm_event)

        # Invoke registered callbacks
        self._invoke_callbacks(event)
        await self._invoke_async_callbacks(event)

    async def _handle_critical_event(self, event: AlarmEvent) -> None:
        """Handle a critical safety event."""
        logger.critical(f"CRITICAL SAFETY EVENT [{event.system_id}]: {event.description}")

        # Trip triggers for steam systems
        trip_triggers = [
            "HIGH_PRESSURE",
            "LOW_PRESSURE",
            "HIGH_TEMPERATURE",
            "LOW_TEMPERATURE",
            "OVERHEAT",
            "WATER_HAMMER",
            "RELIEF_VALVE",
            "EMERGENCY_SHUTDOWN",
        ]

        if any(trigger in event.alarm_type.upper() for trigger in trip_triggers):
            if self._trip_callback:
                logger.critical(f"Triggering system trip for {event.system_id}")
                self._trip_callback(event.system_id, event.description)

            self._trip_history.append({
                "system_id": event.system_id,
                "timestamp": datetime.now(timezone.utc),
                "reason": event.description,
                "event_type": event.alarm_type,
            })

    def acknowledge_alarm(
        self,
        system_id: str,
        event_id: str,
        user: str,
    ) -> bool:
        """
        Acknowledge an alarm.

        Args:
            system_id: System ID
            event_id: Event ID to acknowledge
            user: User acknowledging

        Returns:
            True if acknowledged successfully
        """
        alarms = self._active_alarms.get(system_id, [])
        for alarm in alarms:
            if alarm.event_id == event_id:
                alarm.acknowledged = True
                alarm.acknowledged_by = user
                alarm.acknowledged_at = datetime.now(timezone.utc)
                alarm.state = AlarmState.ACKNOWLEDGED
                logger.info(f"Alarm {event_id} acknowledged by {user}")
                return True
        return False

    def clear_alarm(self, system_id: str, event_id: str) -> bool:
        """
        Clear an alarm.

        Args:
            system_id: System ID
            event_id: Event ID to clear

        Returns:
            True if cleared successfully
        """
        if system_id in self._active_alarms:
            original_count = len(self._active_alarms[system_id])
            self._active_alarms[system_id] = [
                a for a in self._active_alarms[system_id]
                if a.event_id != event_id
            ]
            if len(self._active_alarms[system_id]) < original_count:
                logger.info(f"Alarm {event_id} cleared for {system_id}")
                return True
        return False

    def get_active_alarms(self, system_id: Optional[str] = None) -> List[AlarmEvent]:
        """Get active alarms, optionally filtered by system."""
        if system_id:
            return list(self._active_alarms.get(system_id, []))

        all_alarms = []
        for alarms in self._active_alarms.values():
            all_alarms.extend(alarms)
        return all_alarms

    def get_alarm_summary(self, system_id: Optional[str] = None) -> Dict[str, Any]:
        """Get alarm summary."""
        alarms = self.get_active_alarms(system_id)

        severity_counts = {
            "info": 0,
            "warning": 0,
            "critical": 0,
            "emergency": 0
        }
        for alarm in alarms:
            sev = alarm.severity.value if hasattr(alarm.severity, 'value') else str(alarm.severity)
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        return {
            "total_active": len(alarms),
            "by_severity": severity_counts,
            "total_historical": len(self._alarm_history),
            "unacknowledged": sum(1 for a in alarms if not a.acknowledged),
            "total_trips": len(self._trip_history),
        }


# =============================================================================
# TRAP DIAGNOSTICS EVENT HANDLER
# =============================================================================

class TrapDiagnosticsEventHandler(EventHandler):
    """
    Handler for steam trap diagnostics events.

    Tracks trap health assessments, failures, and maintenance
    recommendations.
    """

    def __init__(self, name: str = "TrapDiagnosticsHandler") -> None:
        """Initialize the trap diagnostics event handler."""
        super().__init__(name)
        self._trap_status: Dict[str, TrapHealthAssessment] = {}  # trap_id -> status
        self._failure_history: List[Dict[str, Any]] = []
        self._diagnostics_history: Dict[str, List[Dict]] = {}  # system_id -> results

    async def handle(self, event: SteamSystemEvent) -> None:
        """
        Handle a trap diagnostics event.

        Args:
            event: Event to handle
        """
        self._event_count += 1
        self._last_event_time = datetime.now(timezone.utc)

        system_id = event.system_id or "UNKNOWN"
        event_type = event.event_type.upper()

        # Initialize tracking
        if system_id not in self._diagnostics_history:
            self._diagnostics_history[system_id] = []

        if "TRAP_ASSESSMENT" in event_type or "TRAP_STATUS" in event_type:
            trap_id = event.payload.get("trap_id", "UNKNOWN")
            status = event.payload.get("status", "healthy")

            # Convert status string to enum
            try:
                failure_mode = TrapFailureMode(status)
            except ValueError:
                failure_mode = TrapFailureMode.UNKNOWN

            assessment = TrapHealthAssessment(
                trap_id=trap_id,
                location=event.payload.get("location", ""),
                status=failure_mode,
                failure_probability=event.payload.get("failure_probability", 0.0),
                temperature_c=event.payload.get("temperature_c", 0.0),
                estimated_loss_rate_kg_hr=event.payload.get("loss_rate_kg_hr", 0.0),
                maintenance_priority=MaintenancePriority(
                    event.payload.get("maintenance_priority", "routine")
                ) if event.payload.get("maintenance_priority") else MaintenancePriority.ROUTINE,
            )

            self._trap_status[trap_id] = assessment

            if failure_mode != TrapFailureMode.HEALTHY:
                logger.warning(
                    f"Trap {trap_id} status: {failure_mode.value} "
                    f"(probability: {assessment.failure_probability:.1%})"
                )

        elif "TRAP_FAILURE" in event_type:
            failure_record = {
                "timestamp": event.timestamp,
                "trap_id": event.payload.get("trap_id"),
                "system_id": system_id,
                "failure_mode": event.payload.get("failure_mode"),
                "estimated_loss_kg_hr": event.payload.get("estimated_loss_kg_hr", 0),
            }
            self._failure_history.append(failure_record)
            logger.error(f"Trap failure detected: {failure_record}")

        elif "DIAGNOSTICS_COMPLETE" in event_type:
            result = {
                "timestamp": event.timestamp,
                "total_traps": event.payload.get("total_traps", 0),
                "failed_traps": event.payload.get("failed_traps", 0),
                "total_loss_kg_hr": event.payload.get("total_loss_kg_hr", 0),
            }
            self._diagnostics_history[system_id].append(result)
            logger.info(
                f"Trap diagnostics complete for {system_id}: "
                f"{result['failed_traps']}/{result['total_traps']} failed"
            )

        # Invoke callbacks
        self._invoke_callbacks(event)
        await self._invoke_async_callbacks(event)

    def get_trap_status(self, trap_id: str) -> Optional[TrapHealthAssessment]:
        """Get current status of a specific trap."""
        return self._trap_status.get(trap_id)

    def get_all_trap_statuses(self) -> Dict[str, TrapHealthAssessment]:
        """Get status of all tracked traps."""
        return dict(self._trap_status)

    def get_failed_traps(self) -> List[TrapHealthAssessment]:
        """Get list of failed traps."""
        return [
            t for t in self._trap_status.values()
            if t.status != TrapFailureMode.HEALTHY
        ]

    def get_diagnostics_summary(self, system_id: str) -> Dict[str, Any]:
        """Get diagnostics summary for a system."""
        traps_in_system = [
            t for t in self._trap_status.values()
            # In production, filter by system
        ]

        healthy = sum(1 for t in traps_in_system if t.status == TrapFailureMode.HEALTHY)
        failed = len(traps_in_system) - healthy
        total_loss = sum(t.estimated_loss_rate_kg_hr for t in traps_in_system)

        return {
            "system_id": system_id,
            "total_traps": len(traps_in_system),
            "healthy_traps": healthy,
            "failed_traps": failed,
            "failure_rate_percent": (failed / len(traps_in_system) * 100) if traps_in_system else 0,
            "total_estimated_loss_kg_hr": round(total_loss, 2),
            "critical_traps": [
                t.trap_id for t in traps_in_system
                if t.maintenance_priority == MaintenancePriority.CRITICAL
            ],
        }


# =============================================================================
# OPTIMIZATION EVENT HANDLER
# =============================================================================

class OptimizationEventHandler(EventHandler):
    """
    Handler for optimization-related events.

    Tracks optimization requests, results, recommendations,
    and implementation status.
    """

    def __init__(self, name: str = "OptimizationHandler") -> None:
        """Initialize the optimization event handler."""
        super().__init__(name)
        self._optimization_history: Dict[str, List[Dict]] = {}  # system_id -> results
        self._recommendation_history: Dict[str, List[Dict]] = {}
        self._active_optimizations: Dict[str, str] = {}  # system_id -> optimization_id
        self._implemented_recommendations: List[Dict] = []

    async def handle(self, event: SteamSystemEvent) -> None:
        """
        Handle an optimization event.

        Args:
            event: Event to handle
        """
        self._event_count += 1
        self._last_event_time = datetime.now(timezone.utc)

        system_id = event.system_id or "UNKNOWN"
        event_type = event.event_type.upper()

        # Initialize tracking
        if system_id not in self._optimization_history:
            self._optimization_history[system_id] = []
        if system_id not in self._recommendation_history:
            self._recommendation_history[system_id] = []

        if "OPTIMIZATION_STARTED" in event_type:
            optimization_id = event.payload.get("optimization_id", str(event.event_id))
            optimization_type = event.payload.get("optimization_type", "combined")
            self._active_optimizations[system_id] = optimization_id
            logger.info(
                f"Optimization started for {system_id}: {optimization_id} "
                f"(type: {optimization_type})"
            )

        elif "OPTIMIZATION_COMPLETED" in event_type:
            result = {
                "timestamp": event.timestamp,
                "optimization_id": event.payload.get("optimization_id"),
                "optimization_type": event.payload.get("optimization_type"),
                "efficiency_improvement": event.payload.get("efficiency_improvement", 0),
                "cost_savings_usd_hr": event.payload.get("cost_savings_usd_hr", 0),
                "co2_reduction_kg_hr": event.payload.get("co2_reduction_kg_hr", 0),
                "status": "completed",
            }
            self._optimization_history[system_id].append(result)
            self._active_optimizations.pop(system_id, None)
            logger.info(
                f"Optimization completed for {system_id}: "
                f"+{result['efficiency_improvement']:.2f}% efficiency, "
                f"${result['cost_savings_usd_hr']:.2f}/hr savings"
            )

        elif "OPTIMIZATION_FAILED" in event_type:
            result = {
                "timestamp": event.timestamp,
                "optimization_id": event.payload.get("optimization_id"),
                "error": event.payload.get("error"),
                "status": "failed",
            }
            self._optimization_history[system_id].append(result)
            self._active_optimizations.pop(system_id, None)
            logger.error(f"Optimization failed for {system_id}: {result['error']}")

        elif "RECOMMENDATION" in event_type:
            recommendation = {
                "timestamp": event.timestamp,
                "recommendation_id": event.payload.get("recommendation_id"),
                "optimization_type": event.payload.get("optimization_type"),
                "setpoint_changes": event.payload.get("setpoint_changes", []),
                "expected_benefit": event.payload.get("expected_benefit"),
                "confidence_percent": event.payload.get("confidence_percent", 95),
                "requires_approval": event.payload.get("requires_approval", True),
                "implemented": False,
            }
            self._recommendation_history[system_id].append(recommendation)
            logger.info(
                f"Recommendation for {system_id}: "
                f"{len(recommendation['setpoint_changes'])} setpoint changes"
            )

        elif "RECOMMENDATION_IMPLEMENTED" in event_type:
            rec_id = event.payload.get("recommendation_id")
            implemented = {
                "timestamp": event.timestamp,
                "recommendation_id": rec_id,
                "system_id": system_id,
                "implemented_by": event.payload.get("implemented_by", "auto"),
                "actual_improvement": event.payload.get("actual_improvement"),
            }
            self._implemented_recommendations.append(implemented)
            logger.info(f"Recommendation {rec_id} implemented for {system_id}")

        # Invoke callbacks
        self._invoke_callbacks(event)
        await self._invoke_async_callbacks(event)

    def get_optimization_stats(self, system_id: str) -> Dict[str, Any]:
        """Get optimization statistics for a system."""
        history = self._optimization_history.get(system_id, [])
        completed = [h for h in history if h.get("status") == "completed"]
        failed = [h for h in history if h.get("status") == "failed"]

        total_efficiency_gain = sum(h.get("efficiency_improvement", 0) for h in completed)
        total_cost_savings = sum(h.get("cost_savings_usd_hr", 0) for h in completed)
        total_co2_reduction = sum(h.get("co2_reduction_kg_hr", 0) for h in completed)

        return {
            "system_id": system_id,
            "total_optimizations": len(history),
            "successful": len(completed),
            "failed": len(failed),
            "success_rate_percent": (
                len(completed) / len(history) * 100 if history else 0
            ),
            "total_efficiency_gain_percent": round(total_efficiency_gain, 2),
            "total_cost_savings_usd_hr": round(total_cost_savings, 2),
            "total_co2_reduction_kg_hr": round(total_co2_reduction, 2),
            "active_optimization": self._active_optimizations.get(system_id),
            "pending_recommendations": len([
                r for r in self._recommendation_history.get(system_id, [])
                if not r.get("implemented")
            ]),
        }

    def get_pending_recommendations(
        self, system_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get pending (not implemented) recommendations."""
        if system_id:
            recs = self._recommendation_history.get(system_id, [])
            return [r for r in recs if not r.get("implemented")]

        all_pending = []
        for recs in self._recommendation_history.values():
            all_pending.extend([r for r in recs if not r.get("implemented")])
        return all_pending


# =============================================================================
# THERMODYNAMICS EVENT HANDLER
# =============================================================================

class ThermodynamicsEventHandler(EventHandler):
    """
    Handler for thermodynamic calculation events.

    Tracks IAPWS-IF97 calculations, enthalpy balances,
    and property lookups.
    """

    def __init__(self, name: str = "ThermodynamicsHandler") -> None:
        """Initialize the thermodynamics event handler."""
        super().__init__(name)
        self._calculation_history: Dict[str, List[Dict]] = {}  # system_id -> calcs
        self._enthalpy_balance_history: Dict[str, List[Dict]] = {}
        self._property_cache_hits = 0
        self._property_cache_misses = 0

    async def handle(self, event: SteamSystemEvent) -> None:
        """
        Handle a thermodynamics event.

        Args:
            event: Event to handle
        """
        self._event_count += 1
        self._last_event_time = datetime.now(timezone.utc)

        system_id = event.system_id or "UNKNOWN"
        event_type = event.event_type.upper()

        # Initialize tracking
        if system_id not in self._calculation_history:
            self._calculation_history[system_id] = []
        if system_id not in self._enthalpy_balance_history:
            self._enthalpy_balance_history[system_id] = []

        if "PROPERTY_CALCULATION" in event_type:
            calc = {
                "timestamp": event.timestamp,
                "pressure_kpa": event.payload.get("pressure_kpa"),
                "temperature_c": event.payload.get("temperature_c"),
                "enthalpy_kj_kg": event.payload.get("enthalpy_kj_kg"),
                "entropy_kj_kg_k": event.payload.get("entropy_kj_kg_k"),
                "iapws_region": event.payload.get("iapws_region"),
                "calculation_time_ms": event.payload.get("calculation_time_ms", 0),
            }
            self._calculation_history[system_id].append(calc)

            # Track cache performance
            if event.payload.get("cache_hit"):
                self._property_cache_hits += 1
            else:
                self._property_cache_misses += 1

            # Keep bounded
            if len(self._calculation_history[system_id]) > 1000:
                self._calculation_history[system_id] = \
                    self._calculation_history[system_id][-1000:]

        elif "ENTHALPY_BALANCE" in event_type:
            balance = {
                "timestamp": event.timestamp,
                "energy_input_kw": event.payload.get("energy_input_kw"),
                "energy_output_kw": event.payload.get("energy_output_kw"),
                "total_losses_kw": event.payload.get("total_losses_kw"),
                "efficiency_percent": event.payload.get("efficiency_percent"),
                "mass_balance_error_percent": event.payload.get("mass_balance_error_percent"),
                "energy_balance_error_percent": event.payload.get("energy_balance_error_percent"),
            }
            self._enthalpy_balance_history[system_id].append(balance)
            logger.debug(
                f"Enthalpy balance for {system_id}: "
                f"{balance['efficiency_percent']:.1f}% efficiency"
            )

            # Keep bounded
            if len(self._enthalpy_balance_history[system_id]) > 500:
                self._enthalpy_balance_history[system_id] = \
                    self._enthalpy_balance_history[system_id][-500:]

        # Invoke callbacks
        self._invoke_callbacks(event)
        await self._invoke_async_callbacks(event)

    def get_efficiency_trend(
        self,
        system_id: str,
        periods: int = 10,
    ) -> Dict[str, Any]:
        """Get efficiency trend for a system."""
        history = self._enthalpy_balance_history.get(system_id, [])

        if not history:
            return {"system_id": system_id, "data_points": 0, "trend": "unknown"}

        recent = history[-periods:] if len(history) >= periods else history
        efficiencies = [
            h["efficiency_percent"]
            for h in recent if h.get("efficiency_percent") is not None
        ]

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
            "system_id": system_id,
            "data_points": len(efficiencies),
            "trend": trend,
            "current_efficiency": efficiencies[-1] if efficiencies else None,
            "avg_efficiency": (
                sum(efficiencies) / len(efficiencies) if efficiencies else None
            ),
            "min_efficiency": min(efficiencies) if efficiencies else None,
            "max_efficiency": max(efficiencies) if efficiencies else None,
        }

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get property cache statistics."""
        total = self._property_cache_hits + self._property_cache_misses
        hit_rate = (
            self._property_cache_hits / total * 100 if total > 0 else 0
        )
        return {
            "cache_hits": self._property_cache_hits,
            "cache_misses": self._property_cache_misses,
            "hit_rate_percent": round(hit_rate, 1),
        }


# =============================================================================
# CONDENSATE EVENT HANDLER
# =============================================================================

class CondensateEventHandler(EventHandler):
    """
    Handler for condensate recovery events.

    Tracks condensate return rates, flash losses, and
    recovery optimization results.
    """

    def __init__(self, name: str = "CondensateHandler") -> None:
        """Initialize the condensate event handler."""
        super().__init__(name)
        self._return_rate_history: Dict[str, List[Dict]] = {}  # system_id -> history
        self._flash_loss_history: Dict[str, List[Dict]] = {}
        self._recovery_recommendations: Dict[str, List[Dict]] = {}

    async def handle(self, event: SteamSystemEvent) -> None:
        """
        Handle a condensate event.

        Args:
            event: Event to handle
        """
        self._event_count += 1
        self._last_event_time = datetime.now(timezone.utc)

        system_id = event.system_id or "UNKNOWN"
        event_type = event.event_type.upper()

        # Initialize tracking
        if system_id not in self._return_rate_history:
            self._return_rate_history[system_id] = []
        if system_id not in self._flash_loss_history:
            self._flash_loss_history[system_id] = []
        if system_id not in self._recovery_recommendations:
            self._recovery_recommendations[system_id] = []

        if "RETURN_RATE" in event_type or "CONDENSATE_FLOW" in event_type:
            data = {
                "timestamp": event.timestamp,
                "return_ratio_percent": event.payload.get("return_ratio_percent"),
                "return_flow_kg_s": event.payload.get("return_flow_kg_s"),
                "return_temperature_c": event.payload.get("return_temperature_c"),
            }
            self._return_rate_history[system_id].append(data)

            # Keep bounded
            if len(self._return_rate_history[system_id]) > 1000:
                self._return_rate_history[system_id] = \
                    self._return_rate_history[system_id][-1000:]

        elif "FLASH_LOSS" in event_type:
            data = {
                "timestamp": event.timestamp,
                "source": event.payload.get("source"),
                "flash_percent": event.payload.get("flash_percent"),
                "energy_loss_kw": event.payload.get("energy_loss_kw"),
            }
            self._flash_loss_history[system_id].append(data)
            logger.info(
                f"Flash loss detected at {data['source']}: "
                f"{data['flash_percent']:.1f}%"
            )

        elif "RECOVERY_RECOMMENDATION" in event_type:
            rec = {
                "timestamp": event.timestamp,
                "recommendation": event.payload.get("recommendation"),
                "target_return_percent": event.payload.get("target_return_percent"),
                "estimated_savings_usd_year": event.payload.get(
                    "estimated_savings_usd_year", 0
                ),
                "payback_months": event.payload.get("payback_months"),
            }
            self._recovery_recommendations[system_id].append(rec)
            logger.info(
                f"Condensate recovery recommendation for {system_id}: "
                f"${rec['estimated_savings_usd_year']}/year potential"
            )

        # Invoke callbacks
        self._invoke_callbacks(event)
        await self._invoke_async_callbacks(event)

    def get_recovery_summary(self, system_id: str) -> Dict[str, Any]:
        """Get condensate recovery summary for a system."""
        history = self._return_rate_history.get(system_id, [])

        if not history:
            return {"system_id": system_id, "data_points": 0}

        recent = history[-100:]
        return_rates = [
            h["return_ratio_percent"]
            for h in recent if h.get("return_ratio_percent") is not None
        ]

        flash_history = self._flash_loss_history.get(system_id, [])
        recent_flash = flash_history[-20:] if flash_history else []
        total_flash_loss = sum(
            f.get("energy_loss_kw", 0) for f in recent_flash
        )

        return {
            "system_id": system_id,
            "data_points": len(return_rates),
            "current_return_percent": return_rates[-1] if return_rates else None,
            "avg_return_percent": (
                sum(return_rates) / len(return_rates) if return_rates else None
            ),
            "flash_loss_events": len(flash_history),
            "recent_flash_loss_kw": round(total_flash_loss, 2),
            "pending_recommendations": len(
                self._recovery_recommendations.get(system_id, [])
            ),
        }


# =============================================================================
# AUDIT EVENT HANDLER
# =============================================================================

class AuditEventHandler(EventHandler):
    """
    Handler for audit and provenance events.

    Maintains complete audit trail of all calculations and actions
    for regulatory compliance and zero-hallucination verification.
    """

    def __init__(
        self,
        name: str = "AuditHandler",
        persist_callback: Optional[Callable[[Dict], None]] = None,
    ) -> None:
        """
        Initialize the audit event handler.

        Args:
            name: Handler name
            persist_callback: Callback to persist audit records
        """
        super().__init__(name)
        self._calculation_events: List[Dict] = []
        self._action_events: List[Dict] = []
        self._persist_callback = persist_callback

    async def handle(self, event: SteamSystemEvent) -> None:
        """
        Handle an audit event.

        Args:
            event: Event to handle
        """
        self._event_count += 1
        self._last_event_time = datetime.now(timezone.utc)

        event_type = event.event_type.upper()

        if "CALCULATION" in event_type:
            calc_event = {
                "event_id": event.event_id,
                "timestamp": event.timestamp,
                "system_id": event.system_id,
                "calculation_type": event.payload.get("calculation_type"),
                "input_hash": event.payload.get("input_hash", ""),
                "output_hash": event.payload.get("output_hash", ""),
                "provenance_hash": event.payload.get("provenance_hash", ""),
                "formula_id": event.payload.get("formula_id", ""),
                "formula_version": event.payload.get("formula_version", "1.0.0"),
                "deterministic": event.payload.get("deterministic", True),
                "calculation_time_ms": event.payload.get("calculation_time_ms", 0),
            }
            self._calculation_events.append(calc_event)

            # Persist if callback provided
            if self._persist_callback:
                try:
                    self._persist_callback(calc_event)
                except Exception as e:
                    logger.error(f"Failed to persist calculation event: {e}")

            logger.debug(f"Audit: Calculation {calc_event['formula_id']} recorded")

        elif "ACTION" in event_type or "SETPOINT" in event_type or "IMPLEMENT" in event_type:
            action_event = {
                "event_id": event.event_id,
                "timestamp": event.timestamp,
                "system_id": event.system_id,
                "event_type": event.event_type,
                "action": event.payload.get("action"),
                "user": event.payload.get("user", "system"),
                "details": event.payload,
            }
            self._action_events.append(action_event)

            if self._persist_callback:
                try:
                    self._persist_callback(action_event)
                except Exception as e:
                    logger.error(f"Failed to persist action event: {e}")

            logger.info(f"Audit: Action {event.event_type} recorded")

        # Keep audit trail bounded (in production, persist to database)
        if len(self._calculation_events) > 10000:
            self._calculation_events = self._calculation_events[-10000:]
        if len(self._action_events) > 10000:
            self._action_events = self._action_events[-10000:]

        # Invoke callbacks
        self._invoke_callbacks(event)
        await self._invoke_async_callbacks(event)

    def get_audit_trail(
        self,
        system_id: Optional[str] = None,
        calculation_type: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """Get calculation audit trail with filters."""
        events = self._calculation_events

        if system_id:
            events = [e for e in events if e.get("system_id") == system_id]
        if calculation_type:
            events = [e for e in events if e.get("calculation_type") == calculation_type]
        if start_time:
            events = [e for e in events if e.get("timestamp") >= start_time]
        if end_time:
            events = [e for e in events if e.get("timestamp") <= end_time]

        return events[-limit:]

    def get_action_history(
        self,
        system_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """Get action audit trail."""
        events = self._action_events

        if system_id:
            events = [e for e in events if e.get("system_id") == system_id]

        return events[-limit:]

    def get_audit_stats(self) -> Dict[str, Any]:
        """Get audit statistics."""
        calc_types: Dict[str, int] = {}
        for event in self._calculation_events:
            ct = event.get("calculation_type", "unknown")
            calc_types[ct] = calc_types.get(ct, 0) + 1

        return {
            "total_calculations": len(self._calculation_events),
            "total_actions": len(self._action_events),
            "calculations_by_type": calc_types,
            "deterministic_count": sum(
                1 for e in self._calculation_events if e.get("deterministic", True)
            ),
        }

    def verify_provenance(self, provenance_hash: str) -> Optional[Dict]:
        """
        Find a calculation by its provenance hash.

        Args:
            provenance_hash: SHA-256 hash to search for

        Returns:
            Calculation event if found, None otherwise
        """
        for event in self._calculation_events:
            if event.get("provenance_hash") == provenance_hash:
                return event
        return None


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

    async def handle(self, event: SteamSystemEvent) -> None:
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

    def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge value directly."""
        self._gauges[name] = value

    def increment_counter(self, name: str, delta: int = 1) -> None:
        """Increment a counter directly."""
        self._counters[name] = self._counters.get(name, 0) + delta

    def record_histogram(self, name: str, value: float) -> None:
        """Record a histogram observation directly."""
        if name not in self._histograms:
            self._histograms[name] = []
        self._histograms[name].append(value)

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
        n = len(values)
        return {
            "count": n,
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / n,
            "p50": sorted_values[n // 2],
            "p95": sorted_values[int(n * 0.95)] if n >= 20 else sorted_values[-1],
            "p99": sorted_values[int(n * 0.99)] if n >= 100 else sorted_values[-1],
        }

    def export_prometheus(self, prefix: str = "unifiedsteam") -> str:
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
