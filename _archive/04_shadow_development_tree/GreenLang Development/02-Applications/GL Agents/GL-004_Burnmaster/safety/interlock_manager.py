"""
InterlockManager - Read-only interface to BMS/SIS interlock status.

This module provides READ-ONLY access to Burner Management System (BMS) and
Safety Instrumented System (SIS) status signals. The optimizer NEVER bypasses
or controls interlocks - it only consumes status signals.

CRITICAL: This module is READ-ONLY. No write operations to BMS/SIS are allowed.

Example:
    >>> manager = InterlockManager(unit_id="BLR-001")
    >>> bms_status = manager.read_bms_status("BLR-001")
    >>> if not bms_status.flame_proven:
    ...     # BLOCK all optimization, observe only
    ...     pass
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from enum import Enum
import hashlib
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class InterlockState(str, Enum):
    """State of an interlock."""
    NORMAL = "normal"  # Interlock not active, safe to operate
    ACTIVE = "active"  # Interlock tripped, operation blocked
    BYPASSED = "bypassed"  # Interlock bypassed (by BMS, not optimizer)
    FAULT = "fault"  # Interlock sensor/logic fault
    UNKNOWN = "unknown"  # Status unknown - treat as ACTIVE


class BMSState(str, Enum):
    """Overall BMS state."""
    RUN = "run"  # Normal operation
    STANDBY = "standby"  # Ready to fire
    PURGE = "purge"  # Purge cycle active
    PILOT = "pilot"  # Pilot flame lit
    LOCKOUT = "lockout"  # Safety lockout
    FAULT = "fault"  # BMS fault


class SISState(str, Enum):
    """Overall SIS state."""
    NORMAL = "normal"  # Normal operation
    ALERT = "alert"  # Alert condition
    TRIP = "trip"  # Trip condition active
    FAULT = "fault"  # SIS fault


class BMSStatus(BaseModel):
    """Burner Management System status - READ ONLY."""
    unit_id: str = Field(..., description="Unit identifier")
    state: BMSState = Field(..., description="Overall BMS state")
    flame_proven: bool = Field(..., description="Flame detected and proven")
    purge_complete: bool = Field(..., description="Purge cycle complete")
    pilot_proven: bool = Field(..., description="Pilot flame proven")
    main_fuel_valve_open: bool = Field(..., description="Main fuel valve status")
    air_damper_proven: bool = Field(..., description="Air damper in position")
    low_fire_hold: bool = Field(default=False, description="Low fire hold active")
    high_fire_hold: bool = Field(default=False, description="High fire hold active")
    interlock_trip: bool = Field(default=False, description="Any interlock tripped")
    lockout_active: bool = Field(default=False, description="BMS lockout active")
    fault_codes: List[str] = Field(default_factory=list, description="Active fault codes")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    provenance_hash: str = Field(..., description="SHA-256 hash for audit")


class SISStatus(BaseModel):
    """Safety Instrumented System status - READ ONLY."""
    unit_id: str = Field(..., description="Unit identifier")
    state: SISState = Field(..., description="Overall SIS state")
    high_pressure_trip: bool = Field(default=False, description="High pressure trip")
    low_pressure_trip: bool = Field(default=False, description="Low pressure trip")
    high_temp_trip: bool = Field(default=False, description="High temperature trip")
    low_water_trip: bool = Field(default=False, description="Low water trip")
    flame_failure_trip: bool = Field(default=False, description="Flame failure trip")
    emergency_stop: bool = Field(default=False, description="Emergency stop active")
    manual_trip: bool = Field(default=False, description="Manual trip activated")
    sif_demands: int = Field(default=0, description="SIF demand count")
    proof_test_overdue: bool = Field(default=False, description="Proof test overdue")
    fault_codes: List[str] = Field(default_factory=list, description="Active fault codes")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    provenance_hash: str = Field(..., description="SHA-256 hash for audit")


class PermissiveStatus(BaseModel):
    """Permissive conditions for burner operation."""
    unit_id: str = Field(..., description="Unit identifier")
    all_permissives_met: bool = Field(..., description="All permissives satisfied")
    combustion_air_available: bool = Field(..., description="Combustion air available")
    fuel_pressure_ok: bool = Field(..., description="Fuel pressure in range")
    atomizing_steam_ok: bool = Field(default=True, description="Atomizing steam OK")
    draft_available: bool = Field(..., description="Furnace draft available")
    ignition_transformer_ok: bool = Field(..., description="Ignition transformer OK")
    pilot_gas_available: bool = Field(..., description="Pilot gas available")
    safety_valve_closed: bool = Field(..., description="Safety valve closed")
    vent_valve_closed: bool = Field(..., description="Vent valve closed")
    missing_permissives: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    provenance_hash: str = Field(..., description="SHA-256 hash for audit")


class Interlock(BaseModel):
    """Individual interlock definition."""
    interlock_id: str = Field(..., description="Interlock identifier")
    name: str = Field(..., description="Interlock name")
    state: InterlockState = Field(..., description="Current state")
    trip_point: Optional[float] = Field(None, description="Trip setpoint")
    actual_value: Optional[float] = Field(None, description="Actual value")
    description: str = Field(default="", description="Description")
    last_trip_time: Optional[datetime] = Field(None, description="Last trip time")


class InterlockEvent(BaseModel):
    """Interlock event for logging."""
    event_id: str = Field(..., description="Unique event ID")
    unit_id: str = Field(..., description="Unit identifier")
    interlock_id: str = Field(..., description="Interlock identifier")
    event_type: str = Field(..., description="Event type: trip, reset, bypass, fault")
    previous_state: InterlockState = Field(..., description="Previous state")
    new_state: InterlockState = Field(..., description="New state")
    trigger_value: Optional[float] = Field(None, description="Value that triggered event")
    operator_id: Optional[str] = Field(None, description="Operator if manual action")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    provenance_hash: str = Field(..., description="SHA-256 hash for audit")


class BlockResult(BaseModel):
    """Result of blocking on an interlock."""
    blocked: bool = Field(..., description="Whether operation is blocked")
    interlock_id: str = Field(..., description="Blocking interlock ID")
    interlock_state: InterlockState = Field(..., description="Interlock state")
    reason: str = Field(..., description="Reason for block")
    recommended_action: str = Field(..., description="Recommended action")
    can_proceed_observe_only: bool = Field(default=True, description="Can proceed in observe mode")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    provenance_hash: str = Field(..., description="SHA-256 hash for audit")


class InterlockManager:
    """
    InterlockManager provides READ-ONLY access to BMS/SIS status.

    CRITICAL SAFETY INVARIANT:
    - This class ONLY READS interlock status
    - NEVER attempts to bypass, reset, or control interlocks
    - NEVER writes to BMS/SIS systems
    - The optimizer can only OBSERVE interlock status

    Attributes:
        unit_id: Identifier for the combustion unit
        event_log: Log of interlock events

    Example:
        >>> manager = InterlockManager(unit_id="BLR-001")
        >>> bms_status = manager.read_bms_status("BLR-001")
        >>> if bms_status.lockout_active:
        ...     # Cannot optimize, observe only
        ...     pass
    """

    def __init__(self, unit_id: str, bms_interface: Any = None, sis_interface: Any = None):
        """
        Initialize InterlockManager.

        Args:
            unit_id: Unit identifier
            bms_interface: Optional BMS read interface (for production)
            sis_interface: Optional SIS read interface (for production)
        """
        self.unit_id = unit_id
        self._bms_interface = bms_interface
        self._sis_interface = sis_interface
        self.event_log: List[InterlockEvent] = []
        self._creation_time = datetime.utcnow()
        logger.info(f"InterlockManager initialized for unit {unit_id} (READ-ONLY)")

    def read_bms_status(self, unit_id: str) -> BMSStatus:
        """
        Read current BMS status (READ-ONLY operation).

        Args:
            unit_id: Unit identifier

        Returns:
            Current BMS status

        Note:
            This method ONLY READS status. It cannot modify BMS state.
        """
        if unit_id != self.unit_id:
            logger.warning(f"Unit ID mismatch: expected {self.unit_id}, got {unit_id}")

        # In production, read from actual BMS interface
        if self._bms_interface is not None:
            try:
                raw_status = self._bms_interface.read_status()
                return self._parse_bms_status(raw_status, unit_id)
            except Exception as e:
                logger.error(f"BMS read failed: {e}")
                return self._create_fault_bms_status(unit_id, str(e))

        # Return simulated status for testing
        return self._create_default_bms_status(unit_id)

    def read_sis_status(self, unit_id: str) -> SISStatus:
        """
        Read current SIS status (READ-ONLY operation).

        Args:
            unit_id: Unit identifier

        Returns:
            Current SIS status

        Note:
            This method ONLY READS status. It cannot modify SIS state.
        """
        if unit_id != self.unit_id:
            logger.warning(f"Unit ID mismatch: expected {self.unit_id}, got {unit_id}")

        # In production, read from actual SIS interface
        if self._sis_interface is not None:
            try:
                raw_status = self._sis_interface.read_status()
                return self._parse_sis_status(raw_status, unit_id)
            except Exception as e:
                logger.error(f"SIS read failed: {e}")
                return self._create_fault_sis_status(unit_id, str(e))

        # Return simulated status for testing
        return self._create_default_sis_status(unit_id)

    def check_permissives(self, unit_id: str) -> PermissiveStatus:
        """
        Check all permissive conditions (READ-ONLY).

        Args:
            unit_id: Unit identifier

        Returns:
            Current permissive status
        """
        # Read BMS status to determine permissives
        bms_status = self.read_bms_status(unit_id)

        # Determine permissive states from BMS
        missing_permissives = []

        combustion_air = bms_status.air_damper_proven
        if not combustion_air:
            missing_permissives.append("combustion_air")

        fuel_pressure = not bms_status.lockout_active
        if not fuel_pressure:
            missing_permissives.append("fuel_pressure")

        draft_available = bms_status.air_damper_proven
        if not draft_available:
            missing_permissives.append("draft")

        ignition_ok = not bms_status.lockout_active
        if not ignition_ok:
            missing_permissives.append("ignition_transformer")

        pilot_gas = bms_status.pilot_proven or bms_status.purge_complete
        if not pilot_gas:
            missing_permissives.append("pilot_gas")

        all_permissives = len(missing_permissives) == 0

        provenance_hash = hashlib.sha256(
            f"permissives_{unit_id}_{all_permissives}_{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()

        return PermissiveStatus(
            unit_id=unit_id,
            all_permissives_met=all_permissives,
            combustion_air_available=combustion_air,
            fuel_pressure_ok=fuel_pressure,
            draft_available=draft_available,
            ignition_transformer_ok=ignition_ok,
            pilot_gas_available=pilot_gas,
            safety_valve_closed=True,
            vent_valve_closed=True,
            missing_permissives=missing_permissives,
            provenance_hash=provenance_hash
        )

    def block_on_interlock(self, interlock: Interlock) -> BlockResult:
        """
        Determine if operation should be blocked by interlock.

        This method does NOT bypass the interlock - it determines if the
        optimizer should block its own actions due to interlock state.

        Args:
            interlock: Interlock to check

        Returns:
            BlockResult indicating if optimization is blocked
        """
        blocked = interlock.state in [
            InterlockState.ACTIVE,
            InterlockState.FAULT,
            InterlockState.UNKNOWN
        ]

        if blocked:
            reason = f"Interlock {interlock.interlock_id} in state {interlock.state.value}"
            if interlock.state == InterlockState.ACTIVE:
                recommended_action = "Wait for interlock to clear, do not bypass"
            elif interlock.state == InterlockState.FAULT:
                recommended_action = "Report fault to maintenance, do not bypass"
            else:
                recommended_action = "Investigate unknown state, assume unsafe"
        else:
            reason = "Interlock not blocking"
            recommended_action = "Proceed with caution"

        provenance_hash = hashlib.sha256(
            f"block_{interlock.interlock_id}_{blocked}_{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()

        result = BlockResult(
            blocked=blocked,
            interlock_id=interlock.interlock_id,
            interlock_state=interlock.state,
            reason=reason,
            recommended_action=recommended_action,
            can_proceed_observe_only=True,  # Can always observe
            provenance_hash=provenance_hash
        )

        if blocked:
            logger.warning(f"Optimization BLOCKED by interlock: {interlock.interlock_id}")

        return result

    def log_interlock_event(self, event: InterlockEvent) -> None:
        """
        Log an interlock event for audit trail.

        Note: This only LOGS events observed by the optimizer.
        It does not generate or control interlock events.

        Args:
            event: Interlock event to log
        """
        # Validate event
        if not event.event_id:
            event.event_id = hashlib.sha256(
                f"{event.interlock_id}_{event.timestamp.isoformat()}".encode()
            ).hexdigest()[:16]

        if not event.provenance_hash:
            event.provenance_hash = hashlib.sha256(
                f"{event.json()}".encode()
            ).hexdigest()

        self.event_log.append(event)

        logger.info(
            f"Interlock event logged: {event.interlock_id} "
            f"{event.previous_state.value} -> {event.new_state.value}"
        )

    def _create_default_bms_status(self, unit_id: str) -> BMSStatus:
        """Create default BMS status for testing."""
        provenance_hash = hashlib.sha256(
            f"bms_default_{unit_id}_{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()

        return BMSStatus(
            unit_id=unit_id,
            state=BMSState.RUN,
            flame_proven=True,
            purge_complete=True,
            pilot_proven=True,
            main_fuel_valve_open=True,
            air_damper_proven=True,
            provenance_hash=provenance_hash
        )

    def _create_fault_bms_status(self, unit_id: str, error: str) -> BMSStatus:
        """Create fault BMS status when read fails."""
        provenance_hash = hashlib.sha256(
            f"bms_fault_{unit_id}_{error}_{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()

        return BMSStatus(
            unit_id=unit_id,
            state=BMSState.FAULT,
            flame_proven=False,
            purge_complete=False,
            pilot_proven=False,
            main_fuel_valve_open=False,
            air_damper_proven=False,
            lockout_active=True,
            fault_codes=[f"READ_FAIL: {error}"],
            provenance_hash=provenance_hash
        )

    def _create_default_sis_status(self, unit_id: str) -> SISStatus:
        """Create default SIS status for testing."""
        provenance_hash = hashlib.sha256(
            f"sis_default_{unit_id}_{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()

        return SISStatus(
            unit_id=unit_id,
            state=SISState.NORMAL,
            provenance_hash=provenance_hash
        )

    def _create_fault_sis_status(self, unit_id: str, error: str) -> SISStatus:
        """Create fault SIS status when read fails."""
        provenance_hash = hashlib.sha256(
            f"sis_fault_{unit_id}_{error}_{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()

        return SISStatus(
            unit_id=unit_id,
            state=SISState.FAULT,
            fault_codes=[f"READ_FAIL: {error}"],
            provenance_hash=provenance_hash
        )

    def _parse_bms_status(self, raw_status: Dict[str, Any], unit_id: str) -> BMSStatus:
        """Parse raw BMS status from interface."""
        provenance_hash = hashlib.sha256(
            f"bms_{unit_id}_{str(raw_status)}_{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()

        return BMSStatus(
            unit_id=unit_id,
            state=BMSState(raw_status.get('state', 'fault')),
            flame_proven=raw_status.get('flame_proven', False),
            purge_complete=raw_status.get('purge_complete', False),
            pilot_proven=raw_status.get('pilot_proven', False),
            main_fuel_valve_open=raw_status.get('main_fuel_valve_open', False),
            air_damper_proven=raw_status.get('air_damper_proven', False),
            low_fire_hold=raw_status.get('low_fire_hold', False),
            high_fire_hold=raw_status.get('high_fire_hold', False),
            interlock_trip=raw_status.get('interlock_trip', False),
            lockout_active=raw_status.get('lockout_active', False),
            fault_codes=raw_status.get('fault_codes', []),
            provenance_hash=provenance_hash
        )

    def _parse_sis_status(self, raw_status: Dict[str, Any], unit_id: str) -> SISStatus:
        """Parse raw SIS status from interface."""
        provenance_hash = hashlib.sha256(
            f"sis_{unit_id}_{str(raw_status)}_{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()

        return SISStatus(
            unit_id=unit_id,
            state=SISState(raw_status.get('state', 'fault')),
            high_pressure_trip=raw_status.get('high_pressure_trip', False),
            low_pressure_trip=raw_status.get('low_pressure_trip', False),
            high_temp_trip=raw_status.get('high_temp_trip', False),
            low_water_trip=raw_status.get('low_water_trip', False),
            flame_failure_trip=raw_status.get('flame_failure_trip', False),
            emergency_stop=raw_status.get('emergency_stop', False),
            manual_trip=raw_status.get('manual_trip', False),
            sif_demands=raw_status.get('sif_demands', 0),
            proof_test_overdue=raw_status.get('proof_test_overdue', False),
            fault_codes=raw_status.get('fault_codes', []),
            provenance_hash=provenance_hash
        )
