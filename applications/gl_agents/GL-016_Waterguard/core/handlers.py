"""
GL-016 WATERGUARD Boiler Water Treatment Agent - Event Handlers

This module provides event handlers for various WATERGUARD events
including chemistry events, safety events, compliance events,
anomaly events, dosing events, and blowdown events.

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
    WaterguardEvent,
    ChemistryEvent,
    SafetyEvent,
    AnomalyEvent,
    SeverityLevel,
    ComplianceViolation,
    ComplianceWarning,
    BlowdownRecommendation,
    DosingRecommendation,
)
from .config import QualityFlag, ComplianceStatus

logger = logging.getLogger(__name__)


# =============================================================================
# BASE EVENT HANDLER
# =============================================================================

class ChemistryEventHandler(ABC):
    """
    Base class for chemistry event handlers.
    
    Event handlers process events from the WATERGUARD agent and
    trigger appropriate actions. All handlers support both sync and
    async callbacks.
    """
    
    def __init__(self, name: str) -> None:
        self.name = name
        self._callbacks: Dict[str, List[Callable]] = {}
        self._async_callbacks: Dict[str, List[Callable[..., Awaitable[None]]]] = {}
        self._event_count = 0
        self._error_count = 0
        self._last_event_time: Optional[datetime] = None
    
    @abstractmethod
    async def handle(self, event: WaterguardEvent) -> None:
        """Handle an event."""
        pass
    
    def register_callback(self, event_type: str, callback: Callable[[WaterguardEvent], None]) -> None:
        """Register a synchronous callback for an event type."""
        if event_type not in self._callbacks:
            self._callbacks[event_type] = []
        self._callbacks[event_type].append(callback)
    
    def register_async_callback(self, event_type: str, callback: Callable[..., Awaitable[None]]) -> None:
        """Register an async callback for an event type."""
        if event_type not in self._async_callbacks:
            self._async_callbacks[event_type] = []
        self._async_callbacks[event_type].append(callback)
    
    def _invoke_callbacks(self, event: WaterguardEvent) -> None:
        """Invoke registered synchronous callbacks for an event."""
        callbacks = self._callbacks.get(event.event_type, [])
        callbacks.extend(self._callbacks.get("*", []))
        for callback in callbacks:
            try:
                callback(event)
            except Exception as e:
                self._error_count += 1
                logger.error(f"Callback error in {self.name}: {e}")
    
    async def _invoke_async_callbacks(self, event: WaterguardEvent) -> None:
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
        return self._event_count
    
    @property
    def error_count(self) -> int:
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

class SafetyEventHandler(ChemistryEventHandler):
    """
    Handler for safety gate violations and interlocks.
    
    Tracks safety events including high/low limit violations,
    interlock activations, and emergency conditions.
    """
    
    def __init__(
        self,
        name: str = "SafetyEventHandler",
        interlock_callback: Optional[Callable[[str, str], None]] = None,
    ) -> None:
        super().__init__(name)
        self._interlock_callback = interlock_callback
        self._active_interlocks: Dict[str, Dict[str, Any]] = {}
        self._interlock_history: List[Dict[str, Any]] = []
        self._safety_violation_count = 0
    
    async def handle(self, event: WaterguardEvent) -> None:
        """Handle a safety event."""
        self._event_count += 1
        self._last_event_time = datetime.now(timezone.utc)
        
        system_id = event.system_id or "UNKNOWN"
        event_type = event.event_type.upper()
        
        if "INTERLOCK" in event_type or "SAFETY" in event_type:
            self._safety_violation_count += 1
            
            interlock_record = {
                "event_id": event.event_id,
                "timestamp": event.timestamp,
                "system_id": system_id,
                "event_type": event.event_type,
                "severity": event.severity.value if hasattr(event.severity, "value") else str(event.severity),
                "payload": event.payload,
            }
            
            self._active_interlocks[event.event_id] = interlock_record
            self._interlock_history.append(interlock_record)
            
            logger.warning(f"Safety event [{system_id}]: {event.event_type} - {event.payload}")
            
            if self._interlock_callback:
                try:
                    self._interlock_callback(system_id, str(event.payload))
                except Exception as e:
                    logger.error(f"Interlock callback error: {e}")
        
        if event.severity in [SeverityLevel.CRITICAL, SeverityLevel.EMERGENCY]:
            logger.critical(f"CRITICAL SAFETY EVENT [{system_id}]: {event.payload}")
        
        self._invoke_callbacks(event)
        await self._invoke_async_callbacks(event)
    
    def acknowledge_interlock(self, event_id: str, user: str) -> bool:
        """Acknowledge an interlock."""
        if event_id in self._active_interlocks:
            self._active_interlocks[event_id]["acknowledged"] = True
            self._active_interlocks[event_id]["acknowledged_by"] = user
            self._active_interlocks[event_id]["acknowledged_at"] = datetime.now(timezone.utc)
            logger.info(f"Interlock {event_id} acknowledged by {user}")
            return True
        return False
    
    def clear_interlock(self, event_id: str) -> bool:
        """Clear an interlock."""
        if event_id in self._active_interlocks:
            del self._active_interlocks[event_id]
            logger.info(f"Interlock {event_id} cleared")
            return True
        return False
    
    def get_active_interlocks(self) -> List[Dict[str, Any]]:
        """Get all active interlocks."""
        return list(self._active_interlocks.values())
    
    def get_safety_summary(self) -> Dict[str, Any]:
        """Get safety summary statistics."""
        return {
            "total_safety_events": self._safety_violation_count,
            "active_interlocks": len(self._active_interlocks),
            "total_interlocks_triggered": len(self._interlock_history),
            "last_event_time": self._last_event_time.isoformat() if self._last_event_time else None,
        }


# =============================================================================
# COMPLIANCE EVENT HANDLER
# =============================================================================

class ComplianceEventHandler(ChemistryEventHandler):
    """
    Handler for constraint breaches and compliance events.
    
    Tracks violations, warnings, and compliance status changes.
    """
    
    def __init__(self, name: str = "ComplianceEventHandler") -> None:
        super().__init__(name)
        self._active_violations: Dict[str, ComplianceViolation] = {}
        self._active_warnings: Dict[str, ComplianceWarning] = {}
        self._violation_history: List[Dict[str, Any]] = []
        self._current_status: ComplianceStatus = ComplianceStatus.COMPLIANT
    
    async def handle(self, event: WaterguardEvent) -> None:
        """Handle a compliance event."""
        self._event_count += 1
        self._last_event_time = datetime.now(timezone.utc)
        
        system_id = event.system_id or "UNKNOWN"
        event_type = event.event_type.upper()
        
        if "VIOLATION" in event_type:
            violation_data = event.payload
            parameter = violation_data.get("parameter", "unknown")
            
            self._active_violations[parameter] = violation_data
            self._violation_history.append({
                "event_id": event.event_id,
                "timestamp": event.timestamp,
                "system_id": system_id,
                "parameter": parameter,
                "violation": violation_data,
            })
            
            self._current_status = ComplianceStatus.VIOLATION
            logger.warning(f"Compliance violation [{system_id}]: {parameter} - {violation_data}")
        
        elif "WARNING" in event_type:
            warning_data = event.payload
            parameter = warning_data.get("parameter", "unknown")
            self._active_warnings[parameter] = warning_data
            
            if self._current_status == ComplianceStatus.COMPLIANT:
                self._current_status = ComplianceStatus.WARNING
            
            logger.info(f"Compliance warning [{system_id}]: {parameter} approaching limit")
        
        elif "CLEAR" in event_type or "RESOLVED" in event_type:
            parameter = event.payload.get("parameter", "unknown")
            if parameter in self._active_violations:
                del self._active_violations[parameter]
            if parameter in self._active_warnings:
                del self._active_warnings[parameter]
            
            self._update_status()
            logger.info(f"Compliance issue cleared [{system_id}]: {parameter}")
        
        self._invoke_callbacks(event)
        await self._invoke_async_callbacks(event)
    
    def _update_status(self) -> None:
        """Update overall compliance status."""
        if self._active_violations:
            self._current_status = ComplianceStatus.VIOLATION
        elif self._active_warnings:
            self._current_status = ComplianceStatus.WARNING
        else:
            self._current_status = ComplianceStatus.COMPLIANT
    
    def get_compliance_summary(self) -> Dict[str, Any]:
        """Get compliance summary."""
        return {
            "current_status": self._current_status.value,
            "active_violations": len(self._active_violations),
            "active_warnings": len(self._active_warnings),
            "violation_parameters": list(self._active_violations.keys()),
            "warning_parameters": list(self._active_warnings.keys()),
            "total_historical_violations": len(self._violation_history),
        }


# =============================================================================
# ANOMALY EVENT HANDLER
# =============================================================================

class AnomalyEventHandler(ChemistryEventHandler):
    """
    Handler for sensor and analyzer anomalies.
    
    Tracks data quality issues, sensor failures, and anomalous readings.
    """
    
    def __init__(self, name: str = "AnomalyEventHandler") -> None:
        super().__init__(name)
        self._active_anomalies: Dict[str, Dict[str, Any]] = {}
        self._anomaly_history: List[Dict[str, Any]] = []
        self._sensor_status: Dict[str, QualityFlag] = {}
    
    async def handle(self, event: WaterguardEvent) -> None:
        """Handle an anomaly event."""
        self._event_count += 1
        self._last_event_time = datetime.now(timezone.utc)
        
        system_id = event.system_id or "UNKNOWN"
        event_type = event.event_type.upper()
        
        if isinstance(event, AnomalyEvent):
            anomaly_record = {
                "event_id": event.event_id,
                "timestamp": event.timestamp,
                "system_id": system_id,
                "anomaly_type": event.anomaly_type,
                "affected_sensors": event.affected_sensors,
                "confidence": event.confidence,
                "payload": event.payload,
            }
            
            for sensor in event.affected_sensors:
                self._active_anomalies[sensor] = anomaly_record
                self._sensor_status[sensor] = QualityFlag.SUSPECT
            
            self._anomaly_history.append(anomaly_record)
            logger.warning(f"Anomaly detected [{system_id}]: {event.anomaly_type} affecting {event.affected_sensors}")
        
        elif "SENSOR_FAILURE" in event_type or "BAD_DATA" in event_type:
            sensor_id = event.payload.get("sensor_id", "unknown")
            self._sensor_status[sensor_id] = QualityFlag.BAD
            logger.error(f"Sensor failure [{system_id}]: {sensor_id}")
        
        elif "SENSOR_RECOVERED" in event_type:
            sensor_id = event.payload.get("sensor_id", "unknown")
            if sensor_id in self._active_anomalies:
                del self._active_anomalies[sensor_id]
            self._sensor_status[sensor_id] = QualityFlag.GOOD
            logger.info(f"Sensor recovered [{system_id}]: {sensor_id}")
        
        self._invoke_callbacks(event)
        await self._invoke_async_callbacks(event)
    
    def get_sensor_status(self, sensor_id: str) -> QualityFlag:
        """Get current status of a sensor."""
        return self._sensor_status.get(sensor_id, QualityFlag.GOOD)
    
    def get_anomaly_summary(self) -> Dict[str, Any]:
        """Get anomaly summary."""
        bad_sensors = [s for s, q in self._sensor_status.items() if q == QualityFlag.BAD]
        suspect_sensors = [s for s, q in self._sensor_status.items() if q == QualityFlag.SUSPECT]
        
        return {
            "active_anomalies": len(self._active_anomalies),
            "bad_sensors": bad_sensors,
            "suspect_sensors": suspect_sensors,
            "total_historical_anomalies": len(self._anomaly_history),
        }


# =============================================================================
# DOSING EVENT HANDLER
# =============================================================================

class DosingEventHandler(ChemistryEventHandler):
    def __init__(self, name: str = "DosingEventHandler") -> None:
        super().__init__(name)
        self._dosing_history: List[Dict[str, Any]] = []
        self._active_dosing: Dict[str, Dict[str, Any]] = {}
        self._daily_consumption: Dict[str, float] = {}
    
    async def handle(self, event: WaterguardEvent) -> None:
        self._event_count += 1
        self._last_event_time = datetime.now(timezone.utc)
        system_id = event.system_id or "UNKNOWN"
        event_type = event.event_type.upper()
        
        if "DOSING_STARTED" in event_type:
            chemical = event.payload.get("chemical_type", "unknown")
            self._active_dosing[chemical] = {
                "start_time": event.timestamp,
                "rate_ml_min": event.payload.get("rate_ml_min", 0),
                "system_id": system_id,
            }
            logger.info(f"Dosing started [{system_id}]: {chemical}")
        
        elif "DOSING_STOPPED" in event_type:
            chemical = event.payload.get("chemical_type", "unknown")
            if chemical in self._active_dosing:
                start_info = self._active_dosing.pop(chemical)
                duration_min = (event.timestamp - start_info["start_time"]).total_seconds() / 60
                volume_ml = duration_min * start_info["rate_ml_min"]
                self._daily_consumption[chemical] = self._daily_consumption.get(chemical, 0) + volume_ml
                self._dosing_history.append({
                    "chemical": chemical, "start_time": start_info["start_time"],
                    "end_time": event.timestamp, "duration_min": duration_min,
                    "volume_ml": volume_ml, "system_id": system_id,
                })
        
        self._invoke_callbacks(event)
        await self._invoke_async_callbacks(event)
    
    def get_dosing_summary(self) -> Dict[str, Any]:
        return {
            "active_dosing_chemicals": list(self._active_dosing.keys()),
            "daily_consumption_ml": dict(self._daily_consumption),
            "total_dosing_events": len(self._dosing_history),
        }


class BlowdownEventHandler(ChemistryEventHandler):
    def __init__(self, name: str = "BlowdownEventHandler") -> None:
        super().__init__(name)
        self._blowdown_history: List[Dict[str, Any]] = []
        self._current_setpoint: float = 0.0
        self._total_blowdown_volume_m3: float = 0.0
        self._heat_recovered_kwh: float = 0.0
    
    async def handle(self, event: WaterguardEvent) -> None:
        self._event_count += 1
        self._last_event_time = datetime.now(timezone.utc)
        system_id = event.system_id or "UNKNOWN"
        event_type = event.event_type.upper()
        
        if "SETPOINT_CHANGE" in event_type:
            old_setpoint = self._current_setpoint
            self._current_setpoint = event.payload.get("new_setpoint_percent", 0.0)
            self._blowdown_history.append({
                "type": "setpoint_change", "timestamp": event.timestamp,
                "old_setpoint": old_setpoint, "new_setpoint": self._current_setpoint,
            })
        
        elif "INTERMITTENT_BLOWDOWN" in event_type:
            volume_m3 = event.payload.get("volume_m3", 0)
            self._total_blowdown_volume_m3 += volume_m3
            self._blowdown_history.append({
                "type": "intermittent", "timestamp": event.timestamp,
                "volume_m3": volume_m3, "system_id": system_id,
            })
        
        self._invoke_callbacks(event)
        await self._invoke_async_callbacks(event)
    
    def get_blowdown_summary(self) -> Dict[str, Any]:
        return {
            "current_setpoint_percent": self._current_setpoint,
            "total_blowdown_volume_m3": round(self._total_blowdown_volume_m3, 3),
            "total_heat_recovered_kwh": round(self._heat_recovered_kwh, 2),
        }


# =============================================================================
# AUDIT EVENT HANDLER
# =============================================================================

class AuditEventHandler(ChemistryEventHandler):
    def __init__(self, name: str = "AuditEventHandler", persist_callback: Optional[Callable[[Dict], None]] = None) -> None:
        super().__init__(name)
        self._calculation_events: List[Dict] = []
        self._action_events: List[Dict] = []
        self._persist_callback = persist_callback
    
    async def handle(self, event: WaterguardEvent) -> None:
        self._event_count += 1
        self._last_event_time = datetime.now(timezone.utc)
        event_type = event.event_type.upper()
        
        if "CALCULATION" in event_type:
            calc_event = {
                "event_id": event.event_id, "timestamp": event.timestamp,
                "system_id": event.system_id, "calculation_type": event.payload.get("calculation_type"),
                "input_hash": event.payload.get("input_hash", ""),
                "output_hash": event.payload.get("output_hash", ""),
                "provenance_hash": event.payload.get("provenance_hash", ""),
                "deterministic": event.payload.get("deterministic", True),
            }
            self._calculation_events.append(calc_event)
            if self._persist_callback:
                try: self._persist_callback(calc_event)
                except Exception as e: logger.error(f"Failed to persist calculation event: {e}")
        
        elif "ACTION" in event_type or "SETPOINT" in event_type:
            action_event = {
                "event_id": event.event_id, "timestamp": event.timestamp,
                "system_id": event.system_id, "event_type": event.event_type,
                "action": event.payload.get("action"),
                "user": event.payload.get("user", "system"),
                "details": event.payload,
            }
            self._action_events.append(action_event)
            if self._persist_callback:
                try: self._persist_callback(action_event)
                except Exception as e: logger.error(f"Failed to persist action event: {e}")
        
        if len(self._calculation_events) > 10000:
            self._calculation_events = self._calculation_events[-10000:]
        if len(self._action_events) > 10000:
            self._action_events = self._action_events[-10000:]
        
        self._invoke_callbacks(event)
        await self._invoke_async_callbacks(event)
    
    def get_audit_trail(self, system_id: Optional[str] = None, limit: int = 100) -> List[Dict]:
        events = self._calculation_events
        if system_id:
            events = [e for e in events if e.get("system_id") == system_id]
        return events[-limit:]
    
    def get_audit_stats(self) -> Dict[str, Any]:
        return {
            "total_calculations": len(self._calculation_events),
            "total_actions": len(self._action_events),
            "deterministic_count": sum(1 for e in self._calculation_events if e.get("deterministic", True)),
        }
    
    def verify_provenance(self, provenance_hash: str) -> Optional[Dict]:
        for event in self._calculation_events:
            if event.get("provenance_hash") == provenance_hash:
                return event
        return None


# =============================================================================
# METRICS EVENT HANDLER
# =============================================================================

class MetricsEventHandler(ChemistryEventHandler):
    def __init__(self, name: str = "MetricsEventHandler") -> None:
        super().__init__(name)
        self._gauges: Dict[str, float] = {}
        self._counters: Dict[str, int] = {}
        self._histograms: Dict[str, List[float]] = {}
    
    async def handle(self, event: WaterguardEvent) -> None:
        self._event_count += 1
        self._last_event_time = datetime.now(timezone.utc)
        
        gauges = event.payload.get("gauges", {})
        counters = event.payload.get("counters", {})
        histograms = event.payload.get("histograms", {})
        
        for name, value in gauges.items():
            if isinstance(value, (int, float)):
                self._gauges[name] = float(value)
        
        for name, delta in counters.items():
            if isinstance(delta, (int, float)):
                self._counters[name] = self._counters.get(name, 0) + int(delta)
        
        for name, value in histograms.items():
            if isinstance(value, (int, float)):
                if name not in self._histograms:
                    self._histograms[name] = []
                self._histograms[name].append(float(value))
                if len(self._histograms[name]) > 1000:
                    self._histograms[name] = self._histograms[name][-1000:]
        
        self._invoke_callbacks(event)
        await self._invoke_async_callbacks(event)
    
    def set_gauge(self, name: str, value: float) -> None:
        self._gauges[name] = value
    
    def increment_counter(self, name: str, delta: int = 1) -> None:
        self._counters[name] = self._counters.get(name, 0) + delta
    
    def record_histogram(self, name: str, value: float) -> None:
        if name not in self._histograms:
            self._histograms[name] = []
        self._histograms[name].append(value)
    
    def get_all_metrics(self) -> Dict[str, Any]:
        return {
            "gauges": dict(self._gauges),
            "counters": dict(self._counters),
            "histograms": {k: len(v) for k, v in self._histograms.items()},
        }
    
    def export_prometheus(self, prefix: str = "waterguard") -> str:
        lines = []
        for name, value in self._gauges.items():
            metric_name = f"{prefix}_{name}".replace(".", "_").replace("-", "_")
            lines.append(f"# TYPE {metric_name} gauge")
            lines.append(f"{metric_name} {value}")
        for name, value in self._counters.items():
            metric_name = f"{prefix}_{name}".replace(".", "_").replace("-", "_")
            lines.append(f"# TYPE {metric_name} counter")
            lines.append(f"{metric_name} {value}")
        return chr(10).join(lines)
