# -*- coding: utf-8 -*-
"""
GL-001 THERMALCOMMAND - Emergency Shutdown (ESD) Documentation and Implementation

This module documents and implements Emergency Shutdown procedures for
industrial thermal systems per IEC 61511, ISA 84, and NFPA 85/86.

CRITICAL: This module defines ESD levels and procedures. GL-001 monitors
and advises but NEVER directly actuates ESD functions. All ESD actuation
is performed by independent Safety Instrumented Systems (SIS).

Reference Standards:
- IEC 61511: Safety Instrumented Systems for Process Industry
- IEC 61508: Functional Safety of E/E/PE Systems
- ISA 84.00.01: Safety Instrumented Functions
- NFPA 85: Boiler and Combustion Systems
- NFPA 86: Ovens and Furnaces
- API 14C: Offshore Safety Systems

Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0

SAFETY DISCLAIMER:
==================
GL-001 ThermalCommand is an ADVISORY system only. It:
1. MONITORS thermal system state
2. PREDICTS potential safety issues
3. RECOMMENDS operator actions
4. LOGS all safety-relevant events

GL-001 NEVER:
1. Directly actuates ESD systems
2. Modifies SIS setpoints
3. Bypasses safety interlocks
4. Overrides operator decisions on safety matters

All emergency shutdown actuation is performed by certified, independent
Safety Instrumented Systems (SIS) that are completely separate from GL-001.
"""

from __future__ import annotations
import hashlib, logging, uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from decimal import Decimal

logger = logging.getLogger(__name__)


class ESDLevel(str, Enum):
    """
    Emergency Shutdown Levels per API/ISA standards.

    Higher numbers = more severe shutdown.
    """
    LEVEL_0 = "ESD_0"  # Total plant shutdown
    LEVEL_1 = "ESD_1"  # Process unit shutdown
    LEVEL_2 = "ESD_2"  # Equipment shutdown
    LEVEL_3 = "ESD_3"  # Local equipment isolation
    LEVEL_4 = "ESD_4"  # Single loop shutdown


class ShutdownTrigger(str, Enum):
    """Categories of shutdown triggers."""
    HIGH_TEMPERATURE = "high_temperature"
    LOW_TEMPERATURE = "low_temperature"
    HIGH_PRESSURE = "high_pressure"
    LOW_PRESSURE = "low_pressure"
    HIGH_LEVEL = "high_level"
    LOW_LEVEL = "low_level"
    HIGH_FLOW = "high_flow"
    LOW_FLOW = "low_flow"
    FLAME_FAILURE = "flame_failure"
    COMBUSTION_AIR_LOSS = "combustion_air_loss"
    FUEL_LEAK = "fuel_leak"
    TOXIC_GAS = "toxic_gas"
    OPERATOR_INITIATED = "operator_initiated"
    EQUIPMENT_FAILURE = "equipment_failure"
    COMMUNICATION_LOSS = "communication_loss"


class ShutdownPhase(str, Enum):
    """Phases of controlled shutdown sequence."""
    NORMAL_OPERATION = "normal_operation"
    PRE_SHUTDOWN = "pre_shutdown"
    LOAD_REDUCTION = "load_reduction"
    ISOLATION = "isolation"
    DEPRESSURIZATION = "depressurization"
    COOLDOWN = "cooldown"
    SECURED = "secured"
    VERIFICATION = "verification"


class SafeState(str, Enum):
    """Defined safe states for thermal equipment."""
    FUEL_ISOLATED = "fuel_isolated"
    POWER_ISOLATED = "power_isolated"
    PRESSURE_RELIEVED = "pressure_relieved"
    TEMPERATURE_BELOW_LIMIT = "temperature_below_limit"
    ATMOSPHERE_PURGED = "atmosphere_purged"
    FULLY_SECURED = "fully_secured"


@dataclass
class ESDAction:
    """Individual action in an ESD sequence."""
    action_id: str
    sequence_order: int
    description: str
    actuator_tag: str
    target_state: str
    max_time_seconds: int
    verification_tag: str
    is_critical: bool = True
    operator_confirmation: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "sequence_order": self.sequence_order,
            "description": self.description,
            "actuator_tag": self.actuator_tag,
            "target_state": self.target_state,
            "max_time_seconds": self.max_time_seconds,
            "verification_tag": self.verification_tag,
            "is_critical": self.is_critical,
        }


@dataclass
class ESDSequence:
    """Complete ESD sequence definition."""
    sequence_id: str
    name: str
    esd_level: ESDLevel
    description: str
    trigger_conditions: List[Dict[str, Any]]
    actions: List[ESDAction]
    safe_state: SafeState
    estimated_duration_seconds: int
    requires_operator: bool = False
    restart_procedure: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "name": self.name,
            "esd_level": self.esd_level.value,
            "description": self.description,
            "trigger_conditions": self.trigger_conditions,
            "actions": [a.to_dict() for a in self.actions],
            "safe_state": self.safe_state.value,
            "estimated_duration_seconds": self.estimated_duration_seconds,
            "requires_operator": self.requires_operator,
        }


@dataclass
class ESDEvent:
    """Record of an ESD event (for audit purposes)."""
    event_id: str
    timestamp: datetime
    sequence_id: str
    trigger: ShutdownTrigger
    trigger_details: Dict[str, Any]
    initiator: str  # "SIS", "OPERATOR", "PLC"
    current_phase: ShutdownPhase
    actions_completed: List[str]
    actions_pending: List[str]
    elapsed_seconds: float
    safe_state_achieved: bool
    notes: str = ""
    provenance_hash: str = ""

    def __post_init__(self):
        if not self.provenance_hash:
            content = f"{self.event_id}|{self.sequence_id}|{self.timestamp.isoformat()}"
            self.provenance_hash = hashlib.sha256(content.encode()).hexdigest()


# =============================================================================
# STANDARD ESD SEQUENCES FOR THERMAL SYSTEMS
# =============================================================================

THERMAL_ESD_SEQUENCES: Dict[str, ESDSequence] = {
    "ESD-BOILER-001": ESDSequence(
        sequence_id="ESD-BOILER-001",
        name="Boiler Emergency Shutdown",
        esd_level=ESDLevel.LEVEL_2,
        description="Emergency shutdown sequence for industrial boiler per NFPA 85",
        trigger_conditions=[
            {"tag": "TIC-101", "condition": ">", "value": 450, "unit": "°F"},
            {"tag": "PIC-101", "condition": ">", "value": 250, "unit": "psig"},
            {"tag": "LIC-101", "condition": "<", "value": 20, "unit": "%"},
            {"tag": "FLAME-DET", "condition": "==", "value": 0, "unit": "bool"},
        ],
        actions=[
            ESDAction("A1", 1, "Close main fuel valve", "XV-101", "CLOSED", 5, "ZSC-101", True),
            ESDAction("A2", 2, "Close pilot fuel valve", "XV-102", "CLOSED", 3, "ZSC-102", True),
            ESDAction("A3", 3, "Open air dampers full", "FCV-101", "100%", 10, "ZT-101", True),
            ESDAction("A4", 4, "Start purge sequence", "SOL-PURGE", "ON", 2, "LS-PURGE", True),
            ESDAction("A5", 5, "Verify 4 volume changes", "PURGE-CNT", ">=4", 300, "AI-PURGE", True),
            ESDAction("A6", 6, "Close stack damper", "XV-105", "CLOSED", 30, "ZSC-105", False),
            ESDAction("A7", 7, "Secure combustion air fan", "FAN-101", "OFF", 5, "XI-FAN", False),
        ],
        safe_state=SafeState.FUEL_ISOLATED,
        estimated_duration_seconds=360,
        requires_operator=False,
        restart_procedure="Refer to SOP-BOILER-RESTART-001",
    ),
    "ESD-FURNACE-001": ESDSequence(
        sequence_id="ESD-FURNACE-001",
        name="Industrial Furnace Emergency Shutdown",
        esd_level=ESDLevel.LEVEL_2,
        description="Emergency shutdown for industrial furnace per NFPA 86",
        trigger_conditions=[
            {"tag": "TIC-201", "condition": ">", "value": 2200, "unit": "°F"},
            {"tag": "O2-ANAL", "condition": "<", "value": 1.0, "unit": "%"},
            {"tag": "FLAME-UV", "condition": "==", "value": 0, "unit": "bool"},
            {"tag": "DOOR-SW", "condition": "==", "value": 1, "unit": "bool"},
        ],
        actions=[
            ESDAction("A1", 1, "Close all fuel zone valves", "XV-201-209", "CLOSED", 5, "ZSC-201", True),
            ESDAction("A2", 2, "De-energize heating elements", "HTR-GROUP", "OFF", 2, "XI-HTR", True),
            ESDAction("A3", 3, "Close atmosphere supply", "XV-210", "CLOSED", 5, "ZSC-210", True),
            ESDAction("A4", 4, "Open emergency vent", "XV-211", "OPEN", 3, "ZSO-211", True),
            ESDAction("A5", 5, "Start N2 purge", "SOL-N2", "ON", 2, "LS-N2", True),
            ESDAction("A6", 6, "Verify atmosphere safe", "O2-ANAL", "<21%", 600, "AI-O2", True),
        ],
        safe_state=SafeState.ATMOSPHERE_PURGED,
        estimated_duration_seconds=620,
        requires_operator=True,
        restart_procedure="Refer to SOP-FURNACE-RESTART-001",
    ),
    "ESD-HEATER-001": ESDSequence(
        sequence_id="ESD-HEATER-001",
        name="Process Heater Emergency Shutdown",
        esd_level=ESDLevel.LEVEL_2,
        description="Emergency shutdown for fired process heater",
        trigger_conditions=[
            {"tag": "TIC-301", "condition": ">", "value": 1000, "unit": "°F"},
            {"tag": "TIC-TUBE", "condition": ">", "value": 1200, "unit": "°F"},
            {"tag": "FIC-FUEL", "condition": "<", "value": 10, "unit": "%"},
            {"tag": "FLOW-PROC", "condition": "<", "value": 50, "unit": "%"},
        ],
        actions=[
            ESDAction("A1", 1, "Close main fuel block", "XV-301", "CLOSED", 3, "ZSC-301", True),
            ESDAction("A2", 2, "Close fuel control valve", "FCV-301", "CLOSED", 2, "ZT-301", True),
            ESDAction("A3", 3, "Open stack damper", "FCV-302", "100%", 5, "ZT-302", True),
            ESDAction("A4", 4, "Maintain process flow", "PROC-PUMP", "RUNNING", 0, "XI-PUMP", True),
            ESDAction("A5", 5, "Start snuffing steam", "XV-SNUFF", "OPEN", 3, "ZSO-SNUFF", False),
        ],
        safe_state=SafeState.FUEL_ISOLATED,
        estimated_duration_seconds=180,
        requires_operator=False,
        restart_procedure="Refer to SOP-HEATER-RESTART-001",
    ),
}


# =============================================================================
# ESD MONITORING AND ADVISORY SYSTEM
# =============================================================================

class ThermalESDMonitor:
    """
    Emergency Shutdown Monitoring and Advisory System for GL-001.

    IMPORTANT: This class MONITORS and ADVISES only. It does NOT actuate
    any ESD functions. All ESD actuation is performed by independent SIS.

    Functions:
    1. Monitor conditions that could lead to ESD
    2. Predict ESD likelihood based on trends
    3. Log all safety-relevant observations
    4. Provide operator advisories
    5. Record ESD events for audit
    """

    VERSION = "1.0.0"

    def __init__(self, agent_id: str = "GL-001"):
        self.agent_id = agent_id
        self._sequences = dict(THERMAL_ESD_SEQUENCES)
        self._events: List[ESDEvent] = []
        self._advisories: List[Dict[str, Any]] = []
        logger.info(f"ThermalESDMonitor initialized for {agent_id} (ADVISORY ONLY)")

    def evaluate_esd_conditions(
        self,
        current_values: Dict[str, float],
        sequence_id: str
    ) -> Dict[str, Any]:
        """
        Evaluate if ESD conditions are approaching or met.

        IMPORTANT: This is advisory only. GL-001 does not trigger ESD.

        Args:
            current_values: Dict of tag_id -> current_value
            sequence_id: ESD sequence to evaluate

        Returns:
            Advisory result with condition status
        """
        sequence = self._sequences.get(sequence_id)
        if not sequence:
            return {"error": f"Unknown sequence: {sequence_id}"}

        condition_results = []
        approaching_trip = []
        at_trip = []

        for condition in sequence.trigger_conditions:
            tag = condition["tag"]
            operator = condition["condition"]
            limit = condition["value"]
            current = current_values.get(tag)

            if current is None:
                condition_results.append({
                    "tag": tag,
                    "status": "UNKNOWN",
                    "message": "No data available",
                })
                continue

            # Evaluate condition
            if operator == ">":
                at_limit = current > limit
                margin = (limit - current) / limit * 100 if limit != 0 else 0
                approaching = 0 < margin < 10
            elif operator == "<":
                at_limit = current < limit
                margin = (current - limit) / limit * 100 if limit != 0 else 0
                approaching = 0 < margin < 10
            elif operator == "==":
                at_limit = current == limit
                approaching = False
                margin = None
            else:
                at_limit = False
                approaching = False
                margin = None

            status = "AT_LIMIT" if at_limit else ("APPROACHING" if approaching else "NORMAL")

            result = {
                "tag": tag,
                "current_value": current,
                "limit": limit,
                "operator": operator,
                "unit": condition.get("unit", ""),
                "status": status,
                "margin_percent": margin,
            }
            condition_results.append(result)

            if at_limit:
                at_trip.append(tag)
            elif approaching:
                approaching_trip.append(tag)

        # Generate advisory
        if at_trip:
            advisory_level = "CRITICAL"
            message = f"ESD condition met for tags: {', '.join(at_trip)}"
        elif approaching_trip:
            advisory_level = "WARNING"
            message = f"Approaching ESD limits: {', '.join(approaching_trip)}"
        else:
            advisory_level = "NORMAL"
            message = "All conditions within normal limits"

        return {
            "sequence_id": sequence_id,
            "sequence_name": sequence.name,
            "esd_level": sequence.esd_level.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "advisory_level": advisory_level,
            "message": message,
            "conditions": condition_results,
            "note": "GL-001 ADVISORY ONLY - ESD actuation by independent SIS",
        }

    def predict_esd_likelihood(
        self,
        trend_data: Dict[str, List[Tuple[datetime, float]]],
        sequence_id: str,
        horizon_minutes: int = 30
    ) -> Dict[str, Any]:
        """
        Predict likelihood of ESD based on parameter trends.

        Args:
            trend_data: Dict of tag_id -> list of (timestamp, value) tuples
            sequence_id: ESD sequence to evaluate
            horizon_minutes: Prediction horizon

        Returns:
            Prediction result with estimated time to trip
        """
        sequence = self._sequences.get(sequence_id)
        if not sequence:
            return {"error": f"Unknown sequence: {sequence_id}"}

        predictions = []

        for condition in sequence.trigger_conditions:
            tag = condition["tag"]
            limit = condition["value"]
            operator = condition["condition"]
            data = trend_data.get(tag, [])

            if len(data) < 2:
                predictions.append({
                    "tag": tag,
                    "prediction": "INSUFFICIENT_DATA",
                })
                continue

            # Simple linear regression for trend
            times = [(t - data[0][0]).total_seconds() / 60 for t, _ in data]
            values = [v for _, v in data]

            n = len(times)
            sum_x = sum(times)
            sum_y = sum(values)
            sum_xy = sum(t * v for t, v in zip(times, values))
            sum_x2 = sum(t * t for t in times)

            denominator = n * sum_x2 - sum_x ** 2
            if abs(denominator) < 1e-10:
                slope = 0
            else:
                slope = (n * sum_xy - sum_x * sum_y) / denominator

            current_value = values[-1]
            current_time = times[-1]

            # Estimate time to limit
            if operator == ">" and slope > 0:
                time_to_limit = (limit - current_value) / slope
            elif operator == "<" and slope < 0:
                time_to_limit = (limit - current_value) / slope
            else:
                time_to_limit = float('inf')

            if time_to_limit <= 0:
                eta_status = "AT_LIMIT"
                eta_minutes = 0
            elif time_to_limit <= horizon_minutes:
                eta_status = "WITHIN_HORIZON"
                eta_minutes = time_to_limit
            else:
                eta_status = "BEYOND_HORIZON"
                eta_minutes = time_to_limit

            predictions.append({
                "tag": tag,
                "current_value": current_value,
                "limit": limit,
                "trend_slope_per_minute": slope,
                "eta_status": eta_status,
                "eta_minutes": eta_minutes if eta_minutes != float('inf') else None,
            })

        # Overall prediction
        critical_predictions = [p for p in predictions if p.get("eta_status") == "WITHIN_HORIZON"]

        if critical_predictions:
            min_eta = min(p["eta_minutes"] for p in critical_predictions)
            overall_status = "LIKELY"
            message = f"ESD likely within {min_eta:.1f} minutes"
        else:
            overall_status = "UNLIKELY"
            message = "No immediate ESD risk based on current trends"

        return {
            "sequence_id": sequence_id,
            "horizon_minutes": horizon_minutes,
            "overall_status": overall_status,
            "message": message,
            "predictions": predictions,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "note": "PREDICTION ONLY - Not a guarantee of future behavior",
        }

    def log_esd_event(self, event: ESDEvent) -> str:
        """Log an ESD event for audit purposes."""
        self._events.append(event)
        logger.warning(
            f"ESD Event logged: {event.sequence_id} triggered by {event.trigger.value} "
            f"at {event.timestamp.isoformat()}"
        )
        return event.event_id

    def get_esd_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[ESDEvent]:
        """Get ESD event history for audit."""
        events = self._events
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]
        return events

    def generate_safety_report(self) -> Dict[str, Any]:
        """Generate safety status report."""
        return {
            "agent_id": self.agent_id,
            "report_timestamp": datetime.now(timezone.utc).isoformat(),
            "esd_sequences_defined": len(self._sequences),
            "esd_events_logged": len(self._events),
            "sequences": [s.to_dict() for s in self._sequences.values()],
            "recent_events": [
                {
                    "event_id": e.event_id,
                    "timestamp": e.timestamp.isoformat(),
                    "sequence_id": e.sequence_id,
                    "trigger": e.trigger.value,
                    "safe_state_achieved": e.safe_state_achieved,
                }
                for e in self._events[-10:]
            ],
            "disclaimer": (
                "GL-001 ThermalCommand provides ADVISORY information only. "
                "All ESD actuation is performed by independent Safety Instrumented Systems."
            ),
        }


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "ESDLevel",
    "ShutdownTrigger",
    "ShutdownPhase",
    "SafeState",
    "ESDAction",
    "ESDSequence",
    "ESDEvent",
    "THERMAL_ESD_SEQUENCES",
    "ThermalESDMonitor",
]
