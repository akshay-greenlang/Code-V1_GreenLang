# -*- coding: utf-8 -*-
"""
GL-016 WATERGUARD Safety Gate System
IEC 61511 SIL-3 Compliant Safety Functions

Safety gates provide hard limits and interlocks for autonomous control.
All gates must pass before control actions are permitted.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class GateStatus(str, Enum):
    """Safety gate status."""
    PASS = "PASS"
    FAIL = "FAIL"
    BYPASS = "BYPASS"
    UNKNOWN = "UNKNOWN"


class TripAction(str, Enum):
    """Action to take on safety trip."""
    ALARM_ONLY = "ALARM_ONLY"
    INHIBIT_INCREASE = "INHIBIT_INCREASE"
    INHIBIT_DECREASE = "INHIBIT_DECREASE"
    INHIBIT_ALL = "INHIBIT_ALL"
    TRIP_TO_SAFE = "TRIP_TO_SAFE"
    EMERGENCY_STOP = "EMERGENCY_STOP"


class SafetyLevel(str, Enum):
    """IEC 61511 Safety Integrity Level."""
    SIL_0 = "SIL_0"
    SIL_1 = "SIL_1"
    SIL_2 = "SIL_2"
    SIL_3 = "SIL_3"
    SIL_4 = "SIL_4"


@dataclass
class GateCheckResult:
    """Result of a safety gate check."""
    gate_id: str
    gate_name: str
    status: GateStatus
    message: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    current_value: Optional[float] = None
    limit_value: Optional[float] = None
    deviation_percent: Optional[float] = None
    trip_action: Optional[TripAction] = None

    def to_dict(self) -> dict:
        return {
            "gate_id": self.gate_id,
            "gate_name": self.gate_name,
            "status": self.status.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "current_value": self.current_value,
            "limit_value": self.limit_value,
            "deviation_percent": self.deviation_percent,
            "trip_action": self.trip_action.value if self.trip_action else None,
        }


@dataclass
class SafetyGateConfig:
    """Configuration for a safety gate."""
    gate_id: str
    name: str
    description: str
    parameter: str
    safety_level: SafetyLevel
    enabled: bool = True

    # Limits
    low_limit: Optional[float] = None
    high_limit: Optional[float] = None
    alarm_low: Optional[float] = None
    alarm_high: Optional[float] = None
    trip_low: Optional[float] = None
    trip_high: Optional[float] = None

    # Rate limits
    rate_limit_per_sec: Optional[float] = None
    rate_limit_per_hour: Optional[float] = None

    # Actions
    alarm_action: TripAction = TripAction.ALARM_ONLY
    trip_action: TripAction = TripAction.TRIP_TO_SAFE

    # Deadband
    deadband_percent: float = 2.0

    # Bypass
    bypass_permitted: bool = False
    bypass_requires_auth: bool = True
    max_bypass_duration_min: int = 60


class SafetyGate:
    """
    Individual safety gate with limit checking.

    Provides hard limits and interlocks for a single parameter.
    """

    def __init__(self, config: SafetyGateConfig):
        self.config = config
        self._last_value: Optional[float] = None
        self._last_check_time: Optional[datetime] = None
        self._bypass_active: bool = False
        self._bypass_expires: Optional[datetime] = None
        self._bypass_reason: Optional[str] = None
        self._trip_active: bool = False
        self._trip_time: Optional[datetime] = None

    @property
    def gate_id(self) -> str:
        return self.config.gate_id

    @property
    def is_bypassed(self) -> bool:
        if not self._bypass_active:
            return False
        if self._bypass_expires and datetime.utcnow() > self._bypass_expires:
            self._bypass_active = False
            return False
        return True

    @property
    def is_tripped(self) -> bool:
        return self._trip_active

    def check(self, value: float) -> GateCheckResult:
        """
        Check if value passes the safety gate.

        Args:
            value: Current value of the parameter

        Returns:
            GateCheckResult with pass/fail status
        """
        now = datetime.utcnow()
        self._last_value = value
        self._last_check_time = now

        # Check if gate is bypassed
        if self.is_bypassed:
            return GateCheckResult(
                gate_id=self.gate_id,
                gate_name=self.config.name,
                status=GateStatus.BYPASS,
                message=f"Gate bypassed: {self._bypass_reason}",
                current_value=value,
            )

        # Check if gate is disabled
        if not self.config.enabled:
            return GateCheckResult(
                gate_id=self.gate_id,
                gate_name=self.config.name,
                status=GateStatus.PASS,
                message="Gate disabled",
                current_value=value,
            )

        # Check trip limits (most severe)
        if self.config.trip_low is not None and value < self.config.trip_low:
            self._activate_trip(value, "low")
            return GateCheckResult(
                gate_id=self.gate_id,
                gate_name=self.config.name,
                status=GateStatus.FAIL,
                message=f"TRIP: {self.config.parameter} below trip limit",
                current_value=value,
                limit_value=self.config.trip_low,
                deviation_percent=self._calc_deviation(value, self.config.trip_low),
                trip_action=self.config.trip_action,
            )

        if self.config.trip_high is not None and value > self.config.trip_high:
            self._activate_trip(value, "high")
            return GateCheckResult(
                gate_id=self.gate_id,
                gate_name=self.config.name,
                status=GateStatus.FAIL,
                message=f"TRIP: {self.config.parameter} above trip limit",
                current_value=value,
                limit_value=self.config.trip_high,
                deviation_percent=self._calc_deviation(value, self.config.trip_high),
                trip_action=self.config.trip_action,
            )

        # Check alarm limits
        if self.config.alarm_low is not None and value < self.config.alarm_low:
            return GateCheckResult(
                gate_id=self.gate_id,
                gate_name=self.config.name,
                status=GateStatus.FAIL,
                message=f"ALARM: {self.config.parameter} below alarm limit",
                current_value=value,
                limit_value=self.config.alarm_low,
                deviation_percent=self._calc_deviation(value, self.config.alarm_low),
                trip_action=self.config.alarm_action,
            )

        if self.config.alarm_high is not None and value > self.config.alarm_high:
            return GateCheckResult(
                gate_id=self.gate_id,
                gate_name=self.config.name,
                status=GateStatus.FAIL,
                message=f"ALARM: {self.config.parameter} above alarm limit",
                current_value=value,
                limit_value=self.config.alarm_high,
                deviation_percent=self._calc_deviation(value, self.config.alarm_high),
                trip_action=self.config.alarm_action,
            )

        # All checks passed
        self._trip_active = False
        return GateCheckResult(
            gate_id=self.gate_id,
            gate_name=self.config.name,
            status=GateStatus.PASS,
            message=f"{self.config.parameter} within limits",
            current_value=value,
        )

    def check_rate(self, value: float, previous_value: float, time_delta_sec: float) -> GateCheckResult:
        """
        Check if rate of change is within limits.

        Args:
            value: Current value
            previous_value: Previous value
            time_delta_sec: Time between readings (seconds)

        Returns:
            GateCheckResult with pass/fail status
        """
        if time_delta_sec <= 0:
            return GateCheckResult(
                gate_id=self.gate_id,
                gate_name=self.config.name,
                status=GateStatus.UNKNOWN,
                message="Invalid time delta",
            )

        rate_per_sec = abs(value - previous_value) / time_delta_sec

        if self.config.rate_limit_per_sec is not None:
            if rate_per_sec > self.config.rate_limit_per_sec:
                return GateCheckResult(
                    gate_id=self.gate_id,
                    gate_name=self.config.name,
                    status=GateStatus.FAIL,
                    message=f"Rate of change exceeds limit: {rate_per_sec:.4f}/sec > {self.config.rate_limit_per_sec}/sec",
                    current_value=rate_per_sec,
                    limit_value=self.config.rate_limit_per_sec,
                    trip_action=TripAction.INHIBIT_ALL,
                )

        return GateCheckResult(
            gate_id=self.gate_id,
            gate_name=self.config.name,
            status=GateStatus.PASS,
            message="Rate within limits",
            current_value=rate_per_sec,
        )

    def activate_bypass(
        self,
        reason: str,
        duration_min: int,
        operator_id: str,
    ) -> bool:
        """
        Activate bypass for this gate.

        Args:
            reason: Reason for bypass
            duration_min: Duration in minutes
            operator_id: ID of operator activating bypass

        Returns:
            True if bypass was activated
        """
        if not self.config.bypass_permitted:
            return False

        if duration_min > self.config.max_bypass_duration_min:
            duration_min = self.config.max_bypass_duration_min

        self._bypass_active = True
        self._bypass_expires = datetime.utcnow() + timedelta(minutes=duration_min)
        self._bypass_reason = f"{reason} (by {operator_id})"
        return True

    def deactivate_bypass(self) -> None:
        """Deactivate bypass."""
        self._bypass_active = False
        self._bypass_expires = None
        self._bypass_reason = None

    def reset_trip(self) -> bool:
        """
        Reset trip condition.

        Returns:
            True if trip was reset
        """
        if not self._trip_active:
            return False

        # Can only reset if value is within limits
        if self._last_value is not None:
            result = self.check(self._last_value)
            if result.status == GateStatus.PASS:
                self._trip_active = False
                self._trip_time = None
                return True

        return False

    def _activate_trip(self, value: float, direction: str) -> None:
        """Activate trip condition."""
        self._trip_active = True
        self._trip_time = datetime.utcnow()

    def _calc_deviation(self, value: float, limit: float) -> float:
        """Calculate deviation percentage from limit."""
        if limit == 0:
            return 0.0
        return ((value - limit) / limit) * 100


# =============================================================================
# Pre-configured Safety Gates for Boiler Water Treatment
# =============================================================================

def create_chemistry_gates() -> list[SafetyGate]:
    """Create standard chemistry safety gates."""
    gates = []

    # pH Gate
    gates.append(SafetyGate(SafetyGateConfig(
        gate_id="GATE_PH",
        name="pH Safety Gate",
        description="Boiler water pH limits for corrosion prevention",
        parameter="ph",
        safety_level=SafetyLevel.SIL_3,
        low_limit=10.5,
        high_limit=11.5,
        alarm_low=10.3,
        alarm_high=11.7,
        trip_low=10.0,
        trip_high=12.0,
        alarm_action=TripAction.ALARM_ONLY,
        trip_action=TripAction.TRIP_TO_SAFE,
    )))

    # Conductivity Gate
    gates.append(SafetyGate(SafetyGateConfig(
        gate_id="GATE_COND",
        name="Conductivity Safety Gate",
        description="Boiler water conductivity limits",
        parameter="conductivity_umho",
        safety_level=SafetyLevel.SIL_3,
        high_limit=5000.0,
        alarm_high=4500.0,
        trip_high=5500.0,
        alarm_action=TripAction.ALARM_ONLY,
        trip_action=TripAction.INHIBIT_DECREASE,  # Force blowdown
    )))

    # Silica Gate
    gates.append(SafetyGate(SafetyGateConfig(
        gate_id="GATE_SILICA",
        name="Silica Safety Gate",
        description="Boiler water silica limits for carryover prevention",
        parameter="silica_ppm",
        safety_level=SafetyLevel.SIL_3,
        high_limit=150.0,
        alarm_high=120.0,
        trip_high=180.0,
        alarm_action=TripAction.ALARM_ONLY,
        trip_action=TripAction.INHIBIT_DECREASE,  # Force blowdown
    )))

    # Dissolved Oxygen Gate
    gates.append(SafetyGate(SafetyGateConfig(
        gate_id="GATE_DO",
        name="Dissolved Oxygen Safety Gate",
        description="Feedwater dissolved oxygen limits for corrosion prevention",
        parameter="dissolved_oxygen_ppb",
        safety_level=SafetyLevel.SIL_3,
        high_limit=7.0,
        alarm_high=5.0,
        trip_high=10.0,
        alarm_action=TripAction.ALARM_ONLY,
        trip_action=TripAction.INHIBIT_ALL,  # Increase scavenger dosing
    )))

    # Phosphate Gate
    gates.append(SafetyGate(SafetyGateConfig(
        gate_id="GATE_PHOSPHATE",
        name="Phosphate Residual Safety Gate",
        description="Boiler water phosphate residual limits",
        parameter="phosphate_ppm",
        safety_level=SafetyLevel.SIL_2,
        low_limit=20.0,
        high_limit=60.0,
        alarm_low=15.0,
        alarm_high=70.0,
        alarm_action=TripAction.ALARM_ONLY,
        trip_action=TripAction.ALARM_ONLY,
    )))

    return gates


def create_rate_limit_gates() -> list[SafetyGate]:
    """Create rate-of-change safety gates."""
    gates = []

    # Blowdown Valve Rate Gate
    gates.append(SafetyGate(SafetyGateConfig(
        gate_id="GATE_BD_RATE",
        name="Blowdown Valve Rate Gate",
        description="Blowdown valve position rate limit",
        parameter="blowdown_valve_percent",
        safety_level=SafetyLevel.SIL_2,
        rate_limit_per_sec=10.0,  # Max 10%/sec
        alarm_action=TripAction.INHIBIT_ALL,
        trip_action=TripAction.INHIBIT_ALL,
    )))

    # Dosing Pump Rate Gate
    gates.append(SafetyGate(SafetyGateConfig(
        gate_id="GATE_DOSE_RATE",
        name="Dosing Pump Rate Gate",
        description="Chemical dosing pump rate limit",
        parameter="dosing_pump_percent",
        safety_level=SafetyLevel.SIL_2,
        rate_limit_per_sec=5.0,  # Max 5%/sec
        alarm_action=TripAction.INHIBIT_ALL,
        trip_action=TripAction.INHIBIT_ALL,
    )))

    # CoC Rate Gate
    gates.append(SafetyGate(SafetyGateConfig(
        gate_id="GATE_COC_RATE",
        name="Cycles of Concentration Rate Gate",
        description="CoC setpoint change rate limit",
        parameter="coc_setpoint",
        safety_level=SafetyLevel.SIL_2,
        rate_limit_per_hour=0.5,  # Max 0.5 CoC units/hour
        alarm_action=TripAction.INHIBIT_ALL,
        trip_action=TripAction.INHIBIT_ALL,
    )))

    return gates


# =============================================================================
# Safety Gate Manager
# =============================================================================

class SafetyGateManager:
    """
    Manages all safety gates for the agent.

    Provides coordinated checking and enforcement of all safety limits.
    """

    def __init__(self, boiler_id: str):
        self.boiler_id = boiler_id
        self._gates: dict[str, SafetyGate] = {}
        self._check_history: list[dict] = []

        # Initialize default gates
        for gate in create_chemistry_gates():
            self._gates[gate.gate_id] = gate
        for gate in create_rate_limit_gates():
            self._gates[gate.gate_id] = gate

    def add_gate(self, gate: SafetyGate) -> None:
        """Add a safety gate."""
        self._gates[gate.gate_id] = gate

    def get_gate(self, gate_id: str) -> Optional[SafetyGate]:
        """Get a safety gate by ID."""
        return self._gates.get(gate_id)

    def check_all_gates(self, readings: dict[str, float]) -> dict[str, GateCheckResult]:
        """
        Check all gates against current readings.

        Args:
            readings: Dict mapping parameter names to current values

        Returns:
            Dict mapping gate IDs to check results
        """
        results = {}
        timestamp = datetime.utcnow()

        for gate_id, gate in self._gates.items():
            param = gate.config.parameter
            if param in readings:
                result = gate.check(readings[param])
                results[gate_id] = result

        # Log check
        self._check_history.append({
            "timestamp": timestamp.isoformat(),
            "readings": readings,
            "results": {k: v.to_dict() for k, v in results.items()},
        })

        return results

    def is_safe_to_proceed(self, readings: dict[str, float]) -> Tuple[bool, list[GateCheckResult]]:
        """
        Check if it's safe to proceed with control actions.

        Args:
            readings: Current sensor readings

        Returns:
            Tuple of (is_safe, list of failed gate results)
        """
        results = self.check_all_gates(readings)
        failed = [r for r in results.values() if r.status == GateStatus.FAIL]
        return len(failed) == 0, failed

    def get_tripped_gates(self) -> list[SafetyGate]:
        """Get all gates currently in trip state."""
        return [g for g in self._gates.values() if g.is_tripped]

    def get_bypassed_gates(self) -> list[SafetyGate]:
        """Get all gates currently bypassed."""
        return [g for g in self._gates.values() if g.is_bypassed]

    def reset_all_trips(self) -> dict[str, bool]:
        """
        Attempt to reset all tripped gates.

        Returns:
            Dict mapping gate IDs to reset success
        """
        results = {}
        for gate_id, gate in self._gates.items():
            if gate.is_tripped:
                results[gate_id] = gate.reset_trip()
        return results

    def get_status_summary(self) -> dict:
        """Get summary of all gate statuses."""
        return {
            "boiler_id": self.boiler_id,
            "timestamp": datetime.utcnow().isoformat(),
            "total_gates": len(self._gates),
            "tripped_gates": len(self.get_tripped_gates()),
            "bypassed_gates": len(self.get_bypassed_gates()),
            "gates": {
                gate_id: {
                    "name": gate.config.name,
                    "enabled": gate.config.enabled,
                    "is_tripped": gate.is_tripped,
                    "is_bypassed": gate.is_bypassed,
                    "safety_level": gate.config.safety_level.value,
                }
                for gate_id, gate in self._gates.items()
            },
        }


# Import needed for bypass duration
from datetime import timedelta
import logging
import threading
from typing import Callable, List, Dict, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# ADDITIONAL GATE RESULT MODEL
# =============================================================================


class GateResult(BaseModel):
    """Result of a gate check with provenance."""

    gate_id: str = Field(..., description="Gate identifier")
    gate_name: str = Field(..., description="Gate name")
    passed: bool = Field(..., description="Gate passed")
    status: GateStatus = Field(..., description="Gate status")
    message: str = Field(default="", description="Result message")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Check time")
    current_value: Optional[float] = Field(default=None, description="Current value")
    limit_value: Optional[float] = Field(default=None, description="Limit value")
    provenance_hash: str = Field(default="", description="SHA-256 hash")

    def model_post_init(self, __context: Any) -> None:
        """Calculate provenance hash."""
        if not self.provenance_hash:
            hash_str = f"{self.gate_id}|{self.passed}|{self.timestamp.isoformat()}"
            self.provenance_hash = hashlib.sha256(hash_str.encode()).hexdigest()[:16]


# =============================================================================
# ANALYZER STATUS
# =============================================================================


class AnalyzerStatus(str, Enum):
    """Status of an analyzer."""
    ONLINE = "online"
    OFFLINE = "offline"
    FAULT = "fault"
    CALIBRATING = "calibrating"
    MAINTENANCE = "maintenance"
    UNKNOWN = "unknown"


class CommunicationStatus(str, Enum):
    """Status of communication links."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    DEGRADED = "degraded"
    TIMEOUT = "timeout"
    ERROR = "error"


class OperatorMode(str, Enum):
    """Operator mode."""
    AUTO = "auto"
    MANUAL = "manual"
    SEMI_AUTO = "semi_auto"
    MAINTENANCE = "maintenance"


# =============================================================================
# SPECIALIZED SAFETY GATES
# =============================================================================


class AnalyzerHealthGate:
    """
    Safety gate that checks analyzer health status.

    WATERGUARD relies on analyzer readings for control decisions.
    If analyzers are unhealthy, control actions must be inhibited.

    Checks:
        - Analyzer communication status
        - Calibration status
        - Fault conditions
        - Reading age (stale data detection)

    Example:
        >>> gate = AnalyzerHealthGate("CT-001", max_reading_age_sec=60)
        >>> result = gate.check(analyzer_status)
        >>> if not result.passed:
        ...     waterguard.enter_safe_mode()
    """

    def __init__(
        self,
        gate_id: str,
        analyzer_tags: List[str],
        max_reading_age_sec: float = 60.0,
        min_healthy_count: int = 1,
    ) -> None:
        """
        Initialize AnalyzerHealthGate.

        Args:
            gate_id: Gate identifier
            analyzer_tags: List of analyzer tags to monitor
            max_reading_age_sec: Maximum age for readings
            min_healthy_count: Minimum healthy analyzers required
        """
        self.gate_id = gate_id
        self.name = "Analyzer Health Gate"
        self.analyzer_tags = analyzer_tags
        self.max_reading_age_sec = max_reading_age_sec
        self.min_healthy_count = min_healthy_count

        self._analyzer_statuses: Dict[str, AnalyzerStatus] = {}
        self._last_reading_times: Dict[str, datetime] = {}
        self._lock = threading.Lock()

    def update_status(
        self,
        tag: str,
        status: AnalyzerStatus,
        reading_time: Optional[datetime] = None
    ) -> None:
        """Update analyzer status."""
        with self._lock:
            self._analyzer_statuses[tag] = status
            if reading_time:
                self._last_reading_times[tag] = reading_time

    def check(self) -> GateResult:
        """Check analyzer health gate."""
        now = datetime.utcnow()
        healthy_count = 0
        failed_tags = []

        with self._lock:
            for tag in self.analyzer_tags:
                status = self._analyzer_statuses.get(tag, AnalyzerStatus.UNKNOWN)
                last_reading = self._last_reading_times.get(tag)

                is_healthy = (
                    status == AnalyzerStatus.ONLINE and
                    last_reading is not None and
                    (now - last_reading).total_seconds() <= self.max_reading_age_sec
                )

                if is_healthy:
                    healthy_count += 1
                else:
                    failed_tags.append(tag)

        passed = healthy_count >= self.min_healthy_count

        if passed:
            message = f"{healthy_count}/{len(self.analyzer_tags)} analyzers healthy"
        else:
            message = f"Insufficient healthy analyzers: {healthy_count}/{self.min_healthy_count} required. Failed: {failed_tags}"

        return GateResult(
            gate_id=self.gate_id,
            gate_name=self.name,
            passed=passed,
            status=GateStatus.PASS if passed else GateStatus.FAIL,
            message=message,
        )


class CommunicationsGate:
    """
    Safety gate that checks communication links.

    Verifies that critical communication links are healthy:
        - OPC-UA to DCS/PLC
        - Historian connection
        - Alarm system link
        - Network connectivity

    Example:
        >>> gate = CommunicationsGate("COMM-001")
        >>> gate.update_link_status("OPC_UA", CommunicationStatus.CONNECTED)
        >>> result = gate.check()
    """

    def __init__(
        self,
        gate_id: str,
        critical_links: List[str] = None,
        timeout_sec: float = 30.0,
    ) -> None:
        """
        Initialize CommunicationsGate.

        Args:
            gate_id: Gate identifier
            critical_links: List of critical link names
            timeout_sec: Communication timeout
        """
        self.gate_id = gate_id
        self.name = "Communications Gate"
        self.critical_links = critical_links or ["OPC_UA", "DCS"]
        self.timeout_sec = timeout_sec

        self._link_statuses: Dict[str, CommunicationStatus] = {}
        self._last_heartbeats: Dict[str, datetime] = {}
        self._lock = threading.Lock()

    def update_link_status(
        self,
        link_name: str,
        status: CommunicationStatus
    ) -> None:
        """Update link status."""
        with self._lock:
            self._link_statuses[link_name] = status
            if status == CommunicationStatus.CONNECTED:
                self._last_heartbeats[link_name] = datetime.utcnow()

    def record_heartbeat(self, link_name: str) -> None:
        """Record a heartbeat from a link."""
        with self._lock:
            self._last_heartbeats[link_name] = datetime.utcnow()

    def check(self) -> GateResult:
        """Check communications gate."""
        now = datetime.utcnow()
        failed_links = []

        with self._lock:
            for link in self.critical_links:
                status = self._link_statuses.get(link, CommunicationStatus.DISCONNECTED)
                last_hb = self._last_heartbeats.get(link)

                is_healthy = (
                    status == CommunicationStatus.CONNECTED and
                    last_hb is not None and
                    (now - last_hb).total_seconds() <= self.timeout_sec
                )

                if not is_healthy:
                    failed_links.append(link)

        passed = len(failed_links) == 0

        if passed:
            message = "All critical communication links healthy"
        else:
            message = f"Communication failures: {failed_links}"

        return GateResult(
            gate_id=self.gate_id,
            gate_name=self.name,
            passed=passed,
            status=GateStatus.PASS if passed else GateStatus.FAIL,
            message=message,
        )


class ManualOverrideGate:
    """
    Safety gate that checks for manual overrides.

    When operator is in manual mode, WATERGUARD must respect
    operator control and inhibit automatic actions.

    Example:
        >>> gate = ManualOverrideGate("MANUAL-001")
        >>> gate.set_operator_mode(OperatorMode.MANUAL)
        >>> result = gate.check()  # Will fail - manual mode active
    """

    def __init__(
        self,
        gate_id: str,
        allow_recommendations: bool = True,
    ) -> None:
        """
        Initialize ManualOverrideGate.

        Args:
            gate_id: Gate identifier
            allow_recommendations: Allow recommendations in manual mode
        """
        self.gate_id = gate_id
        self.name = "Manual Override Gate"
        self.allow_recommendations = allow_recommendations

        self._operator_mode = OperatorMode.AUTO
        self._override_tags: Dict[str, datetime] = {}
        self._lock = threading.Lock()

    def set_operator_mode(self, mode: OperatorMode) -> None:
        """Set operator mode."""
        with self._lock:
            old_mode = self._operator_mode
            self._operator_mode = mode
            if old_mode != mode:
                logger.info("Operator mode changed: %s -> %s", old_mode.value, mode.value)

    def add_override(self, tag: str) -> None:
        """Add manual override for a tag."""
        with self._lock:
            self._override_tags[tag] = datetime.utcnow()

    def remove_override(self, tag: str) -> None:
        """Remove manual override for a tag."""
        with self._lock:
            self._override_tags.pop(tag, None)

    def is_tag_overridden(self, tag: str) -> bool:
        """Check if a tag is under manual override."""
        with self._lock:
            return tag in self._override_tags

    def check(self) -> GateResult:
        """Check manual override gate."""
        with self._lock:
            mode = self._operator_mode
            override_count = len(self._override_tags)

        passed = mode in (OperatorMode.AUTO, OperatorMode.SEMI_AUTO)

        if passed:
            message = f"Automatic control allowed (mode: {mode.value})"
        else:
            message = f"Manual mode active - automatic control inhibited (overrides: {override_count})"

        return GateResult(
            gate_id=self.gate_id,
            gate_name=self.name,
            passed=passed,
            status=GateStatus.PASS if passed else GateStatus.FAIL,
            message=message,
        )


class RateLimitGate:
    """
    Safety gate that enforces rate limits on control actions.

    Prevents too-frequent changes that could destabilize the process.

    Example:
        >>> gate = RateLimitGate("RATE-001", max_changes_per_hour=10)
        >>> result = gate.check_rate_limit()
        >>> if result.passed:
        ...     execute_control_action()
    """

    def __init__(
        self,
        gate_id: str,
        max_changes_per_minute: int = 2,
        max_changes_per_hour: int = 20,
    ) -> None:
        """
        Initialize RateLimitGate.

        Args:
            gate_id: Gate identifier
            max_changes_per_minute: Max changes per minute
            max_changes_per_hour: Max changes per hour
        """
        self.gate_id = gate_id
        self.name = "Rate Limit Gate"
        self.max_per_minute = max_changes_per_minute
        self.max_per_hour = max_changes_per_hour

        self._change_timestamps: List[datetime] = []
        self._lock = threading.Lock()

    def record_change(self) -> None:
        """Record a control action change."""
        with self._lock:
            self._change_timestamps.append(datetime.utcnow())
            # Keep only last hour
            cutoff = datetime.utcnow() - timedelta(hours=1)
            self._change_timestamps = [
                t for t in self._change_timestamps if t > cutoff
            ]

    def check(self) -> GateResult:
        """Check rate limit gate."""
        now = datetime.utcnow()
        minute_ago = now - timedelta(minutes=1)
        hour_ago = now - timedelta(hours=1)

        with self._lock:
            changes_per_minute = sum(1 for t in self._change_timestamps if t > minute_ago)
            changes_per_hour = sum(1 for t in self._change_timestamps if t > hour_ago)

        minute_ok = changes_per_minute < self.max_per_minute
        hour_ok = changes_per_hour < self.max_per_hour
        passed = minute_ok and hour_ok

        if passed:
            message = f"Rate OK: {changes_per_minute}/min ({self.max_per_minute} max), {changes_per_hour}/hr ({self.max_per_hour} max)"
        else:
            if not minute_ok:
                message = f"Rate limit exceeded: {changes_per_minute}/min > {self.max_per_minute}/min"
            else:
                message = f"Rate limit exceeded: {changes_per_hour}/hr > {self.max_per_hour}/hr"

        return GateResult(
            gate_id=self.gate_id,
            gate_name=self.name,
            passed=passed,
            status=GateStatus.PASS if passed else GateStatus.FAIL,
            message=message,
            current_value=float(changes_per_hour),
            limit_value=float(self.max_per_hour),
        )


class ConstraintGate:
    """
    Safety gate that checks boundary constraints.

    Validates that proposed actions are within defined constraints.

    Example:
        >>> gate = ConstraintGate("CONSTRAINT-001", boundary_engine)
        >>> result = gate.check_action(proposed_action)
    """

    def __init__(
        self,
        gate_id: str,
        boundary_engine: Any = None,
    ) -> None:
        """
        Initialize ConstraintGate.

        Args:
            gate_id: Gate identifier
            boundary_engine: WaterguardBoundaryEngine instance
        """
        self.gate_id = gate_id
        self.name = "Constraint Gate"
        self._boundary_engine = boundary_engine

    def set_boundary_engine(self, engine: Any) -> None:
        """Set the boundary engine."""
        self._boundary_engine = engine

    def check_action(
        self,
        action_type: str,
        target_tag: str,
        target_value: float
    ) -> GateResult:
        """Check if action is within constraints."""
        if not self._boundary_engine:
            return GateResult(
                gate_id=self.gate_id,
                gate_name=self.name,
                passed=False,
                status=GateStatus.FAIL,
                message="Boundary engine not configured",
            )

        try:
            from .boundary_engine import ProposedAction

            proposed = ProposedAction(
                action_type=action_type,
                target_tag=target_tag,
                target_value=target_value,
            )

            permitted, reason = self._boundary_engine.validate_action(proposed)

            return GateResult(
                gate_id=self.gate_id,
                gate_name=self.name,
                passed=permitted,
                status=GateStatus.PASS if permitted else GateStatus.FAIL,
                message=reason,
                current_value=target_value,
            )

        except Exception as e:
            return GateResult(
                gate_id=self.gate_id,
                gate_name=self.name,
                passed=False,
                status=GateStatus.FAIL,
                message=f"Constraint check error: {str(e)}",
            )


class ChangeManagementGate:
    """
    Safety gate that enforces change management procedures.

    Ensures that significant changes follow proper authorization.

    Example:
        >>> gate = ChangeManagementGate("CM-001")
        >>> gate.approve_change("CHG-001", "shift_supervisor")
        >>> result = gate.check_change("CHG-001")
    """

    def __init__(
        self,
        gate_id: str,
        require_approval: bool = True,
        approval_timeout_hours: int = 24,
    ) -> None:
        """
        Initialize ChangeManagementGate.

        Args:
            gate_id: Gate identifier
            require_approval: Require explicit approval
            approval_timeout_hours: Approval timeout in hours
        """
        self.gate_id = gate_id
        self.name = "Change Management Gate"
        self.require_approval = require_approval
        self.approval_timeout_hours = approval_timeout_hours

        self._approved_changes: Dict[str, Tuple[str, datetime]] = {}
        self._lock = threading.Lock()

    def request_change(self, change_id: str, description: str) -> str:
        """Request a change (returns change ID)."""
        logger.info("Change requested: %s - %s", change_id, description)
        return change_id

    def approve_change(self, change_id: str, approved_by: str) -> None:
        """Approve a change."""
        with self._lock:
            self._approved_changes[change_id] = (approved_by, datetime.utcnow())
        logger.info("Change approved: %s by %s", change_id, approved_by)

    def revoke_approval(self, change_id: str) -> None:
        """Revoke a change approval."""
        with self._lock:
            self._approved_changes.pop(change_id, None)
        logger.info("Change approval revoked: %s", change_id)

    def check_change(self, change_id: str) -> GateResult:
        """Check if a change is approved."""
        if not self.require_approval:
            return GateResult(
                gate_id=self.gate_id,
                gate_name=self.name,
                passed=True,
                status=GateStatus.PASS,
                message="Approval not required",
            )

        with self._lock:
            approval = self._approved_changes.get(change_id)

        if not approval:
            return GateResult(
                gate_id=self.gate_id,
                gate_name=self.name,
                passed=False,
                status=GateStatus.FAIL,
                message=f"Change {change_id} not approved",
            )

        approved_by, approved_at = approval
        age_hours = (datetime.utcnow() - approved_at).total_seconds() / 3600

        if age_hours > self.approval_timeout_hours:
            return GateResult(
                gate_id=self.gate_id,
                gate_name=self.name,
                passed=False,
                status=GateStatus.FAIL,
                message=f"Change {change_id} approval expired ({age_hours:.1f} hours old)",
            )

        return GateResult(
            gate_id=self.gate_id,
            gate_name=self.name,
            passed=True,
            status=GateStatus.PASS,
            message=f"Change {change_id} approved by {approved_by}",
        )


# =============================================================================
# SAFETY GATE COORDINATOR
# =============================================================================


class SafetyGateCoordinator:
    """
    Coordinates all safety gates for WATERGUARD.

    The coordinator ensures that ALL gates must pass before any
    control action is permitted. It provides:
        - Centralized gate management
        - Combined gate checking
        - Audit trail of all gate checks
        - Integration with emergency shutdown

    Example:
        >>> coordinator = SafetyGateCoordinator()
        >>> coordinator.register_gate("analyzer", analyzer_gate)
        >>> coordinator.register_gate("comms", comms_gate)
        >>> result = coordinator.check_all_gates()
        >>> if result.all_passed:
        ...     execute_control_action()
    """

    def __init__(
        self,
        emergency_handler: Any = None,
    ) -> None:
        """
        Initialize SafetyGateCoordinator.

        Args:
            emergency_handler: EmergencyShutdownHandler instance
        """
        self._emergency_handler = emergency_handler

        self._gates: Dict[str, Any] = {}
        self._lock = threading.Lock()

        # Check history
        self._check_history: List[Dict] = []
        self._max_history = 1000

        # Statistics
        self._stats = {
            "checks_total": 0,
            "checks_passed": 0,
            "checks_failed": 0,
            "gates_failed_by_id": {},
        }

        logger.info("SafetyGateCoordinator initialized")

    def register_gate(self, gate_id: str, gate: Any) -> None:
        """Register a safety gate."""
        with self._lock:
            self._gates[gate_id] = gate
        logger.info("Registered gate: %s", gate_id)

    def unregister_gate(self, gate_id: str) -> None:
        """Unregister a safety gate."""
        with self._lock:
            self._gates.pop(gate_id, None)

    def get_gate(self, gate_id: str) -> Optional[Any]:
        """Get a gate by ID."""
        with self._lock:
            return self._gates.get(gate_id)

    def check_all_gates(self) -> Dict[str, Any]:
        """
        Check all registered gates.

        Returns:
            Dict with:
                - all_passed: bool
                - results: Dict[gate_id, GateResult]
                - failed_gates: List[str]
        """
        now = datetime.utcnow()
        results = {}
        failed_gates = []

        with self._lock:
            gates_to_check = dict(self._gates)

        for gate_id, gate in gates_to_check.items():
            try:
                if hasattr(gate, 'check'):
                    result = gate.check()
                else:
                    result = GateResult(
                        gate_id=gate_id,
                        gate_name=str(gate),
                        passed=True,
                        status=GateStatus.UNKNOWN,
                        message="Gate does not support check()",
                    )

                results[gate_id] = result
                if not result.passed:
                    failed_gates.append(gate_id)

            except Exception as e:
                logger.error("Gate %s check failed: %s", gate_id, e)
                results[gate_id] = GateResult(
                    gate_id=gate_id,
                    gate_name=gate_id,
                    passed=False,
                    status=GateStatus.FAIL,
                    message=f"Check error: {str(e)}",
                )
                failed_gates.append(gate_id)

        all_passed = len(failed_gates) == 0

        # Update statistics
        with self._lock:
            self._stats["checks_total"] += 1
            if all_passed:
                self._stats["checks_passed"] += 1
            else:
                self._stats["checks_failed"] += 1
                for gate_id in failed_gates:
                    self._stats["gates_failed_by_id"][gate_id] = (
                        self._stats["gates_failed_by_id"].get(gate_id, 0) + 1
                    )

            # Add to history
            self._check_history.append({
                "timestamp": now.isoformat(),
                "all_passed": all_passed,
                "failed_gates": failed_gates,
                "results": {k: v.dict() if hasattr(v, 'dict') else str(v) for k, v in results.items()},
            })
            if len(self._check_history) > self._max_history:
                self._check_history = self._check_history[-self._max_history:]

        # Log result
        if all_passed:
            logger.debug("All %d gates passed", len(results))
        else:
            logger.warning("Gate check FAILED: %s", failed_gates)

        return {
            "all_passed": all_passed,
            "results": results,
            "failed_gates": failed_gates,
            "timestamp": now.isoformat(),
        }

    def is_safe_to_proceed(self) -> Tuple[bool, List[str]]:
        """
        Check if it's safe to proceed with control actions.

        Returns:
            Tuple of (is_safe, list of failed gate IDs)
        """
        result = self.check_all_gates()
        return result["all_passed"], result["failed_gates"]

    def get_statistics(self) -> Dict[str, Any]:
        """Get coordinator statistics."""
        with self._lock:
            return {
                **self._stats,
                "registered_gates": len(self._gates),
            }

    def get_check_history(self, limit: int = 100) -> List[Dict]:
        """Get check history."""
        with self._lock:
            return list(reversed(self._check_history[-limit:]))
