"""
GL-012 STEAMQUAL - Safety Interlock Manager

Read-only monitoring of Safety Instrumented System (SIS) interlocks.
This module provides ADVISORY ONLY status tracking with NO override capability.

CRITICAL SAFETY NOTICE:
This module is ADVISORY ONLY. It monitors SIS interlock status but has
NO capability to override, bypass, or modify any safety interlocks.
All safety control remains with the certified SIS per IEC 61511.

Interlock Monitoring:
- High pressure shutdown (PSH)
- Low water level (LSL)
- High temperature (TSH)
- Flow rate limits (FSL/FSH)
- Drum level protection
- Emergency isolation

Standards Compliance:
    - IEC 61508 (Functional Safety of E/E/PE Safety-Related Systems)
    - IEC 61511 (Functional Safety - Safety Instrumented Systems)
    - NFPA 85 (Boiler and Combustion Systems Hazards Code)
    - API 556 (Instrumentation and Control Systems)

FAIL-SAFE Design:
When interlock status cannot be read, the manager assumes the MOST
RESTRICTIVE state (assumes interlock is tripped). This prevents
optimization actions that could conflict with active interlocks.

Example:
    >>> from safety.interlock_manager import InterlockManager
    >>> manager = InterlockManager(header_id="STEAM-HDR-001")
    >>>
    >>> # Read-only status check
    >>> status = manager.get_status()
    >>> if status.any_tripped:
    ...     logger.warning(f"Interlocks tripped: {status.tripped_tags}")
    >>>
    >>> # Check specific interlock
    >>> if manager.is_interlock_tripped("PSH-001"):
    ...     logger.critical("High pressure shutdown active!")

Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================


class InterlockStatus(str, Enum):
    """
    Interlock status states.

    IMPORTANT: This is READ-ONLY status. The manager cannot change
    the actual interlock state in the SIS.
    """

    NORMAL = "normal"           # Interlock not activated - safe to operate
    ALARM = "alarm"             # Pre-trip alarm - warning condition
    TRIPPED = "tripped"         # Interlock activated - shutdown/isolation active
    BYPASSED = "bypassed"       # Interlock bypassed (by SIS operators, not this system)
    UNKNOWN = "unknown"         # Status cannot be determined (FAIL-SAFE: assume tripped)
    FAULT = "fault"             # Sensor/system fault detected


class InterlockType(str, Enum):
    """Types of safety interlocks for steam systems."""

    # Pressure interlocks
    HIGH_PRESSURE = "high_pressure"           # PSH - High pressure shutdown
    LOW_PRESSURE = "low_pressure"             # PSL - Low pressure

    # Temperature interlocks
    HIGH_TEMPERATURE = "high_temperature"     # TSH - High temperature shutdown
    LOW_TEMPERATURE = "low_temperature"       # TSL - Low temperature

    # Level interlocks
    HIGH_LEVEL = "high_level"                 # LSH - High level
    LOW_LEVEL = "low_level"                   # LSL - Low level (critical for boilers)
    HIGH_HIGH_LEVEL = "high_high_level"       # LSHH - Very high level
    LOW_LOW_LEVEL = "low_low_level"           # LSLL - Very low level (emergency)

    # Flow interlocks
    HIGH_FLOW = "high_flow"                   # FSH - High flow
    LOW_FLOW = "low_flow"                     # FSL - Low flow (loss of steam)

    # Quality interlocks
    LOW_QUALITY = "low_quality"               # QSL - Low steam quality

    # Emergency
    EMERGENCY_SHUTDOWN = "emergency_shutdown"  # ESD - Emergency shutdown
    MANUAL_TRIP = "manual_trip"               # Manual operator trip


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass(frozen=True)
class SafetyInterlock:
    """
    Immutable specification of a safety interlock.

    This is the DEFINITION of an interlock, not its current state.
    Interlock definitions come from the SIS configuration and
    are read-only to this system.

    Attributes:
        tag: Unique interlock tag (e.g., "PSH-001")
        description: Human-readable description
        interlock_type: Type of interlock (pressure, temp, level, etc.)
        trip_setpoint: Value at which interlock trips
        alarm_setpoint: Value at which pre-trip alarm activates
        unit: Unit of measurement
        sil_level: Safety Integrity Level (1-4) per IEC 61511
        process_action: What happens when tripped (shutdown, isolation, etc.)
        reset_type: Manual or automatic reset
    """
    tag: str
    description: str
    interlock_type: InterlockType
    trip_setpoint: Optional[float] = None
    alarm_setpoint: Optional[float] = None
    unit: str = ""
    sil_level: int = 2  # Default SIL 2 per IEC 61511
    process_action: str = "shutdown"
    reset_type: str = "manual"  # "manual" or "auto"


@dataclass
class InterlockReading:
    """
    Current reading of an interlock status.

    This is a POINT-IN-TIME observation of the interlock state.
    The actual interlock state is controlled by the SIS, not this system.

    Attributes:
        tag: Interlock tag
        status: Current status (normal, alarm, tripped, etc.)
        current_value: Current process value (if available)
        timestamp: When this reading was taken
        data_quality: Quality of the reading (good, bad, uncertain)
        source: Where this reading came from (OPC-UA node, SCADA tag, etc.)
    """
    tag: str
    status: InterlockStatus
    current_value: Optional[float] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    data_quality: str = "good"
    source: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "tag": self.tag,
            "status": self.status.value,
            "current_value": self.current_value,
            "timestamp": self.timestamp.isoformat(),
            "data_quality": self.data_quality,
            "source": self.source,
        }


# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class InterlockStatusSummary(BaseModel):
    """
    Summary of all interlock statuses for a steam system.

    Provides a complete snapshot of interlock states for
    advisory display and optimization constraint consideration.
    """

    header_id: str = Field(..., description="Steam header identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When this summary was generated"
    )

    # Counts
    total_interlocks: int = Field(0, ge=0, description="Total interlocks monitored")
    normal_count: int = Field(0, ge=0, description="Interlocks in normal state")
    alarm_count: int = Field(0, ge=0, description="Interlocks in alarm state")
    tripped_count: int = Field(0, ge=0, description="Interlocks in tripped state")
    bypassed_count: int = Field(0, ge=0, description="Interlocks in bypassed state")
    unknown_count: int = Field(0, ge=0, description="Interlocks with unknown status")
    fault_count: int = Field(0, ge=0, description="Interlocks with faults")

    # Status flags
    any_tripped: bool = Field(False, description="True if any interlock is tripped")
    any_alarm: bool = Field(False, description="True if any interlock in alarm")
    any_unknown: bool = Field(False, description="True if any status unknown (FAIL-SAFE)")
    system_safe: bool = Field(True, description="True if safe for optimization")

    # Lists
    tripped_tags: List[str] = Field(default_factory=list, description="Tags of tripped interlocks")
    alarm_tags: List[str] = Field(default_factory=list, description="Tags of alarmed interlocks")
    bypassed_tags: List[str] = Field(default_factory=list, description="Tags of bypassed interlocks")

    # Detailed readings
    interlock_readings: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="All interlock readings"
    )

    # Provenance
    provenance_hash: str = Field("", description="SHA-256 hash for audit trail")

    def get_summary_message(self) -> str:
        """Get human-readable summary message."""
        if self.any_tripped:
            return (
                f"INTERLOCKS ACTIVE at {self.header_id}: "
                f"{self.tripped_count} tripped ({', '.join(self.tripped_tags)})"
            )
        elif self.any_unknown:
            return (
                f"INTERLOCK STATUS UNKNOWN at {self.header_id}: "
                f"{self.unknown_count} unknown - ASSUMING SAFE=FALSE"
            )
        elif self.any_alarm:
            return (
                f"INTERLOCK ALARMS at {self.header_id}: "
                f"{self.alarm_count} in alarm ({', '.join(self.alarm_tags)})"
            )
        else:
            return f"All {self.total_interlocks} interlocks NORMAL at {self.header_id}"


# =============================================================================
# INTERLOCK MANAGER
# =============================================================================


class InterlockManager:
    """
    Read-only Safety Interlock Manager for GL-012 STEAMQUAL.

    CRITICAL: This manager is ADVISORY ONLY. It provides read-only
    monitoring of SIS interlocks with NO capability to override,
    bypass, or modify any safety interlock.

    Purpose:
    1. Monitor interlock status for optimization decisions
    2. Prevent optimization that conflicts with active interlocks
    3. Provide status display for operator awareness
    4. Log interlock events for audit trail

    FAIL-SAFE Behavior:
    - If interlock status cannot be read: assume TRIPPED
    - If data quality is bad: assume TRIPPED
    - If communication fails: assume ALL TRIPPED

    This ensures the optimization system never acts against
    potentially active safety interlocks.

    Standards:
    - IEC 61508/61511: Functional Safety
    - NFPA 85: Boiler Safety
    - API 556: Control Systems

    Example:
        >>> manager = InterlockManager(header_id="STEAM-HDR-001")
        >>>
        >>> # Check if safe to optimize
        >>> status = manager.get_status()
        >>> if not status.system_safe:
        ...     logger.warning("Optimization blocked: interlocks active")
        ...     return  # Do not proceed with optimization
        >>>
        >>> # Safe to proceed
        >>> optimization_result = run_optimization()
    """

    VERSION = "1.0.0"
    STANDARDS = ["IEC 61508", "IEC 61511", "NFPA 85", "API 556"]

    def __init__(
        self,
        header_id: str,
        fail_safe: bool = True,
        status_callback: Optional[Callable[[InterlockReading], None]] = None,
    ):
        """
        Initialize interlock manager.

        Args:
            header_id: Steam header or system identifier
            fail_safe: If True, assume tripped when status unknown
            status_callback: Optional callback when status changes
        """
        self.header_id = header_id
        self.fail_safe = fail_safe
        self._status_callback = status_callback

        # Interlock definitions (read from SIS configuration)
        self._interlocks: Dict[str, SafetyInterlock] = {}

        # Current readings (updated from SIS/SCADA)
        self._readings: Dict[str, InterlockReading] = {}
        self._readings_lock = threading.Lock()

        # Event log
        self._event_log: List[Dict[str, Any]] = []
        self._event_log_lock = threading.Lock()

        # Initialize standard steam system interlocks
        self._init_standard_interlocks()

        logger.info(
            f"InterlockManager initialized for {header_id}: "
            f"fail_safe={fail_safe}, interlocks={len(self._interlocks)}"
        )

    def _init_standard_interlocks(self) -> None:
        """Initialize standard steam system safety interlocks."""
        standard_interlocks = [
            SafetyInterlock(
                tag="PSH-001",
                description="High steam pressure shutdown",
                interlock_type=InterlockType.HIGH_PRESSURE,
                trip_setpoint=15.0,  # bar
                alarm_setpoint=14.0,
                unit="bar",
                sil_level=2,
                process_action="header_isolation",
                reset_type="manual",
            ),
            SafetyInterlock(
                tag="PSL-001",
                description="Low steam pressure alarm",
                interlock_type=InterlockType.LOW_PRESSURE,
                trip_setpoint=3.0,  # bar
                alarm_setpoint=4.0,
                unit="bar",
                sil_level=1,
                process_action="alarm_only",
                reset_type="auto",
            ),
            SafetyInterlock(
                tag="TSH-001",
                description="High temperature shutdown",
                interlock_type=InterlockType.HIGH_TEMPERATURE,
                trip_setpoint=250.0,  # C
                alarm_setpoint=240.0,
                unit="C",
                sil_level=2,
                process_action="desuperheat_increase",
                reset_type="manual",
            ),
            SafetyInterlock(
                tag="LSLL-001",
                description="Very low drum level emergency",
                interlock_type=InterlockType.LOW_LOW_LEVEL,
                trip_setpoint=-6.0,  # inches from normal
                alarm_setpoint=-4.0,
                unit="inches",
                sil_level=3,  # Higher SIL for critical
                process_action="emergency_shutdown",
                reset_type="manual",
            ),
            SafetyInterlock(
                tag="LSL-001",
                description="Low drum level alarm",
                interlock_type=InterlockType.LOW_LEVEL,
                trip_setpoint=-4.0,
                alarm_setpoint=-2.0,
                unit="inches",
                sil_level=2,
                process_action="feedwater_increase",
                reset_type="auto",
            ),
            SafetyInterlock(
                tag="LSHH-001",
                description="Very high drum level",
                interlock_type=InterlockType.HIGH_HIGH_LEVEL,
                trip_setpoint=8.0,
                alarm_setpoint=6.0,
                unit="inches",
                sil_level=2,
                process_action="feedwater_shutoff",
                reset_type="manual",
            ),
            SafetyInterlock(
                tag="FSL-001",
                description="Low steam flow",
                interlock_type=InterlockType.LOW_FLOW,
                trip_setpoint=1000.0,  # kg/hr
                alarm_setpoint=2000.0,
                unit="kg/hr",
                sil_level=1,
                process_action="alarm_only",
                reset_type="auto",
            ),
            SafetyInterlock(
                tag="QSL-001",
                description="Low steam quality",
                interlock_type=InterlockType.LOW_QUALITY,
                trip_setpoint=0.90,  # 90% quality
                alarm_setpoint=0.93,
                unit="fraction",
                sil_level=1,
                process_action="separator_blowdown",
                reset_type="auto",
            ),
            SafetyInterlock(
                tag="ESD-001",
                description="Emergency shutdown",
                interlock_type=InterlockType.EMERGENCY_SHUTDOWN,
                sil_level=3,
                process_action="full_shutdown",
                reset_type="manual",
            ),
        ]

        for interlock in standard_interlocks:
            self._interlocks[interlock.tag] = interlock
            # Initialize with UNKNOWN status (FAIL-SAFE)
            self._readings[interlock.tag] = InterlockReading(
                tag=interlock.tag,
                status=InterlockStatus.UNKNOWN if self.fail_safe else InterlockStatus.NORMAL,
                data_quality="unknown",
            )

    # =========================================================================
    # READ-ONLY STATUS ACCESS
    # =========================================================================

    def update_reading(
        self,
        tag: str,
        status: InterlockStatus,
        current_value: Optional[float] = None,
        data_quality: str = "good",
        source: str = "",
    ) -> None:
        """
        Update interlock reading from external source (SCADA/OPC-UA).

        This method RECEIVES status updates from the SIS.
        It does NOT send commands to the SIS.

        Args:
            tag: Interlock tag
            status: Current status from SIS
            current_value: Current process value
            data_quality: Quality indicator (good, bad, uncertain)
            source: Data source identifier
        """
        if tag not in self._interlocks:
            logger.warning(f"Unknown interlock tag: {tag}")
            return

        with self._readings_lock:
            old_reading = self._readings.get(tag)
            old_status = old_reading.status if old_reading else InterlockStatus.UNKNOWN

            # FAIL-SAFE: Bad data quality = assume tripped
            if self.fail_safe and data_quality in ["bad", "uncertain"]:
                effective_status = InterlockStatus.UNKNOWN
                logger.warning(
                    f"Interlock {tag}: bad data quality ({data_quality}), "
                    f"FAIL-SAFE assuming UNKNOWN"
                )
            else:
                effective_status = status

            new_reading = InterlockReading(
                tag=tag,
                status=effective_status,
                current_value=current_value,
                data_quality=data_quality,
                source=source,
            )
            self._readings[tag] = new_reading

            # Log state changes
            if old_status != effective_status:
                self._log_event(tag, old_status, effective_status, current_value)

                # Invoke callback
                if self._status_callback:
                    try:
                        self._status_callback(new_reading)
                    except Exception as e:
                        logger.error(f"Status callback error: {e}")

    def get_reading(self, tag: str) -> Optional[InterlockReading]:
        """
        Get current reading for an interlock.

        Args:
            tag: Interlock tag

        Returns:
            Current reading or None if tag unknown
        """
        with self._readings_lock:
            return self._readings.get(tag)

    def is_interlock_tripped(self, tag: str) -> bool:
        """
        Check if specific interlock is tripped.

        FAIL-SAFE: Returns True if status is TRIPPED or UNKNOWN.

        Args:
            tag: Interlock tag

        Returns:
            True if interlock is tripped or status unknown
        """
        with self._readings_lock:
            reading = self._readings.get(tag)
            if reading is None:
                return self.fail_safe  # Unknown tag = assume tripped if fail_safe

            return reading.status in [
                InterlockStatus.TRIPPED,
                InterlockStatus.UNKNOWN,
                InterlockStatus.FAULT,
            ]

    def is_interlock_normal(self, tag: str) -> bool:
        """
        Check if specific interlock is in normal state.

        Args:
            tag: Interlock tag

        Returns:
            True only if interlock is definitely NORMAL
        """
        with self._readings_lock:
            reading = self._readings.get(tag)
            if reading is None:
                return False

            return reading.status == InterlockStatus.NORMAL

    def is_system_safe(self) -> bool:
        """
        Check if system is safe for optimization.

        Returns True only if ALL of:
        - No interlocks tripped
        - No interlocks in unknown state
        - No interlocks with faults

        This is the primary gate for optimization decisions.

        Returns:
            True if safe to proceed with optimization
        """
        with self._readings_lock:
            for reading in self._readings.values():
                if reading.status in [
                    InterlockStatus.TRIPPED,
                    InterlockStatus.UNKNOWN,
                    InterlockStatus.FAULT,
                ]:
                    return False
            return True

    def get_status(self) -> InterlockStatusSummary:
        """
        Get complete status summary of all interlocks.

        Returns comprehensive status for display and logging.

        Returns:
            InterlockStatusSummary with all interlock states
        """
        with self._readings_lock:
            readings_list = list(self._readings.values())

        # Count by status
        normal_count = sum(1 for r in readings_list if r.status == InterlockStatus.NORMAL)
        alarm_count = sum(1 for r in readings_list if r.status == InterlockStatus.ALARM)
        tripped_count = sum(1 for r in readings_list if r.status == InterlockStatus.TRIPPED)
        bypassed_count = sum(1 for r in readings_list if r.status == InterlockStatus.BYPASSED)
        unknown_count = sum(1 for r in readings_list if r.status == InterlockStatus.UNKNOWN)
        fault_count = sum(1 for r in readings_list if r.status == InterlockStatus.FAULT)

        # Get tags by status
        tripped_tags = [r.tag for r in readings_list if r.status == InterlockStatus.TRIPPED]
        alarm_tags = [r.tag for r in readings_list if r.status == InterlockStatus.ALARM]
        bypassed_tags = [r.tag for r in readings_list if r.status == InterlockStatus.BYPASSED]

        # Compute safety flag
        any_tripped = tripped_count > 0
        any_unknown = unknown_count > 0
        any_alarm = alarm_count > 0

        # System is safe only if no trips, unknowns, or faults
        system_safe = not any_tripped and not any_unknown and fault_count == 0

        # Compute provenance hash
        provenance_data = {
            "header_id": self.header_id,
            "total": len(readings_list),
            "tripped_tags": tripped_tags,
            "system_safe": system_safe,
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()

        return InterlockStatusSummary(
            header_id=self.header_id,
            total_interlocks=len(readings_list),
            normal_count=normal_count,
            alarm_count=alarm_count,
            tripped_count=tripped_count,
            bypassed_count=bypassed_count,
            unknown_count=unknown_count,
            fault_count=fault_count,
            any_tripped=any_tripped,
            any_alarm=any_alarm,
            any_unknown=any_unknown,
            system_safe=system_safe,
            tripped_tags=tripped_tags,
            alarm_tags=alarm_tags,
            bypassed_tags=bypassed_tags,
            interlock_readings=[r.to_dict() for r in readings_list],
            provenance_hash=provenance_hash,
        )

    # =========================================================================
    # INTERLOCK CONFIGURATION (READ-ONLY)
    # =========================================================================

    def get_interlock_definition(self, tag: str) -> Optional[SafetyInterlock]:
        """Get interlock definition (read-only)."""
        return self._interlocks.get(tag)

    def get_all_interlock_tags(self) -> List[str]:
        """Get list of all monitored interlock tags."""
        return list(self._interlocks.keys())

    def get_interlocks_by_type(
        self,
        interlock_type: InterlockType,
    ) -> List[SafetyInterlock]:
        """Get all interlocks of a specific type."""
        return [
            interlock
            for interlock in self._interlocks.values()
            if interlock.interlock_type == interlock_type
        ]

    def add_interlock_definition(self, interlock: SafetyInterlock) -> None:
        """
        Add a new interlock definition to monitor.

        This adds to the monitoring list, it does NOT create
        an interlock in the SIS.

        Args:
            interlock: Interlock definition to add
        """
        self._interlocks[interlock.tag] = interlock
        with self._readings_lock:
            if interlock.tag not in self._readings:
                self._readings[interlock.tag] = InterlockReading(
                    tag=interlock.tag,
                    status=InterlockStatus.UNKNOWN if self.fail_safe else InterlockStatus.NORMAL,
                    data_quality="unknown",
                )
        logger.info(f"Added interlock definition: {interlock.tag}")

    # =========================================================================
    # EVENT LOGGING
    # =========================================================================

    def _log_event(
        self,
        tag: str,
        old_status: InterlockStatus,
        new_status: InterlockStatus,
        value: Optional[float] = None,
    ) -> None:
        """Log interlock status change event."""
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "header_id": self.header_id,
            "tag": tag,
            "old_status": old_status.value,
            "new_status": new_status.value,
            "value": value,
        }

        with self._event_log_lock:
            self._event_log.append(event)
            # Keep last 1000 events
            if len(self._event_log) > 1000:
                self._event_log = self._event_log[-1000:]

        # Log appropriately
        if new_status == InterlockStatus.TRIPPED:
            logger.critical(
                f"INTERLOCK TRIPPED [{self.header_id}] {tag}: "
                f"{old_status.value} -> {new_status.value} (value={value})"
            )
        elif new_status == InterlockStatus.ALARM:
            logger.warning(
                f"INTERLOCK ALARM [{self.header_id}] {tag}: "
                f"{old_status.value} -> {new_status.value} (value={value})"
            )
        elif new_status == InterlockStatus.NORMAL and old_status != InterlockStatus.NORMAL:
            logger.info(
                f"Interlock returned to NORMAL [{self.header_id}] {tag}"
            )
        else:
            logger.debug(
                f"Interlock status change [{self.header_id}] {tag}: "
                f"{old_status.value} -> {new_status.value}"
            )

    def get_event_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent interlock events."""
        with self._event_log_lock:
            return list(reversed(self._event_log[-limit:]))

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def reset_to_unknown(self) -> None:
        """
        Reset all interlock readings to UNKNOWN (FAIL-SAFE state).

        Used when communication with SIS is lost.
        """
        with self._readings_lock:
            for tag in self._readings:
                self._readings[tag] = InterlockReading(
                    tag=tag,
                    status=InterlockStatus.UNKNOWN,
                    data_quality="lost_communication",
                )

        logger.warning(
            f"All interlocks reset to UNKNOWN (FAIL-SAFE) for {self.header_id}"
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get manager statistics."""
        status = self.get_status()
        return {
            "version": self.VERSION,
            "header_id": self.header_id,
            "fail_safe": self.fail_safe,
            "total_interlocks": status.total_interlocks,
            "system_safe": status.system_safe,
            "any_tripped": status.any_tripped,
            "any_unknown": status.any_unknown,
            "standards": self.STANDARDS,
        }

    def __repr__(self) -> str:
        """String representation."""
        status = self.get_status()
        return (
            f"InterlockManager(header_id={self.header_id!r}, "
            f"interlocks={status.total_interlocks}, "
            f"tripped={status.tripped_count}, "
            f"safe={status.system_safe})"
        )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "InterlockStatus",
    "InterlockType",
    # Data classes
    "SafetyInterlock",
    "InterlockReading",
    # Models
    "InterlockStatusSummary",
    # Main class
    "InterlockManager",
]
