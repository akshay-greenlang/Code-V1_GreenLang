"""
GL-016 Waterguard SIS Integration - IEC 61511 SIL-3 Compliant

This module provides READ-ONLY access to the Safety Instrumented System (SIS).
WATERGUARD is SUPERVISORY ONLY and CANNOT override SIS, BMS, or safety valves.

CRITICAL SAFETY PRINCIPLE:
    WATERGUARD reads SIS status but NEVER writes to SIS.
    All safety trips are executed by the independent SIS.

Protected Systems (READ-ONLY):
    - Boiler Management System (BMS)
    - Low-water cutoff (LWCO)
    - High-pressure safety valves
    - Flame safeguard systems
    - Emergency blowdown interlocks

Reference Standards:
    - IEC 61511-1:2016 Functional Safety
    - NFPA 85 Boiler and Combustion Systems
    - ASME CSD-1 Controls and Safety Devices

Author: GreenLang Safety Engineering Team
Version: 1.0.0
SIL Level: 3
"""

from __future__ import annotations

import hashlib
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import uuid

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================


class SISHealthStatus(str, Enum):
    """Health status of the SIS."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAULT = "fault"
    OFFLINE = "offline"
    UNKNOWN = "unknown"


class TripType(str, Enum):
    """Types of SIS trips."""
    LOW_WATER = "low_water"
    HIGH_WATER = "high_water"
    HIGH_PRESSURE = "high_pressure"
    LOW_PRESSURE = "low_pressure"
    HIGH_TEMPERATURE = "high_temperature"
    FLAME_FAILURE = "flame_failure"
    HIGH_CONDUCTIVITY = "high_conductivity"
    HIGH_SILICA = "high_silica"
    FUEL_SYSTEM = "fuel_system"
    COMBUSTION_AIR = "combustion_air"
    EMERGENCY_STOP = "emergency_stop"
    MANUAL_TRIP = "manual_trip"


class InterlockStatus(str, Enum):
    """Status of an interlock."""
    ARMED = "armed"
    TRIPPED = "tripped"
    BYPASSED = "bypassed"
    FAULT = "fault"
    UNKNOWN = "unknown"


# =============================================================================
# DATA MODELS
# =============================================================================


class ActiveTrip(BaseModel):
    """An active SIS trip condition."""

    trip_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Unique trip identifier"
    )
    trip_type: TripType = Field(
        ...,
        description="Type of trip"
    )
    trip_time: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Time trip occurred"
    )
    trip_value: Optional[float] = Field(
        default=None,
        description="Value that caused trip"
    )
    setpoint: Optional[float] = Field(
        default=None,
        description="Trip setpoint"
    )
    source_tag: str = Field(
        default="",
        description="Tag that triggered trip"
    )
    message: str = Field(
        default="",
        description="Trip message"
    )
    acknowledged: bool = Field(
        default=False,
        description="Trip acknowledged by operator"
    )
    acknowledged_by: Optional[str] = Field(
        default=None,
        description="Operator who acknowledged"
    )
    reset_time: Optional[datetime] = Field(
        default=None,
        description="Time trip was reset"
    )


class SISStatus(BaseModel):
    """Status of the Safety Instrumented System."""

    status_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())[:8],
        description="Status record ID"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Status timestamp"
    )
    overall_health: SISHealthStatus = Field(
        default=SISHealthStatus.UNKNOWN,
        description="Overall SIS health"
    )

    # BMS Status
    bms_healthy: bool = Field(
        default=True,
        description="Boiler Management System healthy"
    )
    bms_in_control: bool = Field(
        default=True,
        description="BMS is in control"
    )

    # Low-water cutoff
    lwco_healthy: bool = Field(
        default=True,
        description="Low-water cutoff healthy"
    )
    lwco_tripped: bool = Field(
        default=False,
        description="Low-water cutoff tripped"
    )

    # Safety valve status
    safety_valves_healthy: bool = Field(
        default=True,
        description="Safety valves healthy"
    )

    # Active trips
    active_trip_count: int = Field(
        default=0,
        ge=0,
        description="Number of active trips"
    )

    # Bypass status
    active_bypasses: int = Field(
        default=0,
        ge=0,
        description="Number of active bypasses"
    )

    # Diagnostics
    diagnostics_ok: bool = Field(
        default=True,
        description="Diagnostics passing"
    )
    last_proof_test: Optional[datetime] = Field(
        default=None,
        description="Last proof test date"
    )
    proof_test_overdue: bool = Field(
        default=False,
        description="Proof test overdue"
    )


class BoilerSafetyStatus(BaseModel):
    """Safety status specific to boiler systems."""

    boiler_id: str = Field(
        ...,
        description="Boiler identifier"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Status timestamp"
    )

    # Level protection
    drum_level_percent: Optional[float] = Field(
        default=None,
        description="Drum level percent"
    )
    low_water_trip_armed: bool = Field(
        default=True,
        description="Low water trip is armed"
    )
    high_water_trip_armed: bool = Field(
        default=True,
        description="High water trip is armed"
    )

    # Pressure protection
    drum_pressure_psig: Optional[float] = Field(
        default=None,
        description="Drum pressure psig"
    )
    high_pressure_trip_armed: bool = Field(
        default=True,
        description="High pressure trip armed"
    )
    safety_valve_set_pressure: Optional[float] = Field(
        default=None,
        description="Safety valve set pressure"
    )

    # Flame safety
    flame_present: bool = Field(
        default=True,
        description="Flame is present"
    )
    flame_safeguard_healthy: bool = Field(
        default=True,
        description="Flame safeguard healthy"
    )

    # Feedwater
    feedwater_flow_available: bool = Field(
        default=True,
        description="Feedwater flow available"
    )

    # Overall
    safe_to_operate: bool = Field(
        default=True,
        description="Safe to operate"
    )


# =============================================================================
# SIS INTERFACE
# =============================================================================


class SISInterface:
    """
    Read-Only Interface to Safety Instrumented System.

    CRITICAL: This interface is READ-ONLY.
    WATERGUARD CANNOT write to or override SIS.

    The SIS operates independently and takes precedence over all
    optimization recommendations. WATERGUARD must respect all SIS
    states and never attempt to interfere with safety functions.

    Protected Systems:
        - Boiler Management System (BMS)
        - Low-water cutoff (LWCO)
        - High-pressure safety valves
        - Flame safeguard systems
        - Emergency blowdown interlocks

    Example:
        >>> sis = SISInterface()
        >>> status = sis.get_sis_status()
        >>> if not sis.is_sis_healthy():
        ...     waterguard.enter_fail_safe("SIS unhealthy")
        >>> trips = sis.get_active_trips()
    """

    # Tags that WATERGUARD is FORBIDDEN to write to
    PROTECTED_TAG_PATTERNS = [
        "SIS_*",
        "BMS_*",
        "LWCO_*",
        "FSG_*",        # Flame safeguard
        "PSV_*",        # Pressure safety valve
        "ESV_*",        # Emergency shutoff valve
        "ESD_*",        # Emergency shutdown
        "TRIP_*",
        "INTERLOCK_*",
        "SAFETY_*",
    ]

    def __init__(
        self,
        status_callback: Optional[Callable[[SISStatus], None]] = None,
        trip_callback: Optional[Callable[[ActiveTrip], None]] = None,
    ) -> None:
        """
        Initialize SIS Interface.

        Args:
            status_callback: Called when SIS status changes
            trip_callback: Called when new trip occurs
        """
        self._status_callback = status_callback
        self._trip_callback = trip_callback

        # Current status (READ-ONLY from SIS)
        self._sis_status = SISStatus()
        self._boiler_statuses: Dict[str, BoilerSafetyStatus] = {}
        self._active_trips: Dict[str, ActiveTrip] = {}

        self._lock = threading.Lock()

        # Statistics
        self._stats = {
            "status_reads": 0,
            "trips_observed": 0,
            "health_checks": 0,
        }

        logger.info(
            "SISInterface initialized - READ-ONLY access to Safety Instrumented System"
        )

    def get_sis_status(self) -> SISStatus:
        """
        Get current SIS status (READ-ONLY).

        Returns:
            Current SIS status

        Note:
            This is a READ-ONLY operation.
            WATERGUARD cannot modify SIS status.
        """
        with self._lock:
            self._stats["status_reads"] += 1
            return self._sis_status

    def is_sis_healthy(self) -> bool:
        """
        Check if SIS is healthy (READ-ONLY).

        Returns:
            True if SIS is healthy

        Note:
            If SIS is not healthy, WATERGUARD should enter fail-safe mode.
        """
        with self._lock:
            self._stats["health_checks"] += 1
            return (
                self._sis_status.overall_health == SISHealthStatus.HEALTHY and
                self._sis_status.bms_healthy and
                self._sis_status.lwco_healthy and
                self._sis_status.safety_valves_healthy and
                self._sis_status.diagnostics_ok
            )

    def get_active_trips(self) -> List[ActiveTrip]:
        """
        Get all active SIS trips (READ-ONLY).

        Returns:
            List of active trips

        Note:
            WATERGUARD should suspend optimization during active trips.
        """
        with self._lock:
            return list(self._active_trips.values())

    def get_boiler_safety_status(
        self,
        boiler_id: str
    ) -> Optional[BoilerSafetyStatus]:
        """
        Get safety status for a specific boiler (READ-ONLY).

        Args:
            boiler_id: Boiler identifier

        Returns:
            Boiler safety status or None
        """
        with self._lock:
            return self._boiler_statuses.get(boiler_id)

    def is_safe_to_operate(self, boiler_id: str) -> bool:
        """
        Check if it's safe to operate a boiler (READ-ONLY).

        Args:
            boiler_id: Boiler identifier

        Returns:
            True if safe to operate

        Note:
            WATERGUARD must NEVER override this determination.
        """
        status = self.get_boiler_safety_status(boiler_id)
        if status is None:
            logger.warning("No safety status for boiler %s - assuming unsafe", boiler_id)
            return False
        return status.safe_to_operate

    def is_tag_protected(self, tag: str) -> bool:
        """
        Check if a tag is protected from WATERGUARD writes.

        Args:
            tag: Tag to check

        Returns:
            True if tag is protected (WATERGUARD cannot write)

        Note:
            WATERGUARD must NEVER attempt to write to protected tags.
        """
        tag_upper = tag.upper()
        for pattern in self.PROTECTED_TAG_PATTERNS:
            if pattern.endswith("*"):
                if tag_upper.startswith(pattern[:-1]):
                    return True
            elif tag_upper == pattern:
                return True
        return False

    def has_active_trips(self) -> bool:
        """
        Check if there are any active trips (READ-ONLY).

        Returns:
            True if any trips are active
        """
        with self._lock:
            return len(self._active_trips) > 0

    def is_lwco_tripped(self) -> bool:
        """
        Check if low-water cutoff is tripped (READ-ONLY).

        Returns:
            True if LWCO is tripped

        Note:
            If LWCO is tripped, all water treatment optimization
            must be suspended immediately.
        """
        with self._lock:
            return self._sis_status.lwco_tripped

    def get_bypass_count(self) -> int:
        """
        Get number of active bypasses (READ-ONLY).

        Returns:
            Number of active bypasses

        Note:
            WATERGUARD should log warnings if bypasses are active.
        """
        with self._lock:
            return self._sis_status.active_bypasses

    # =========================================================================
    # STATUS UPDATE METHODS (Called by SIS data provider, not by WATERGUARD)
    # =========================================================================

    def update_sis_status(self, status: SISStatus) -> None:
        """
        Update SIS status from SIS data provider.

        This method is called by the SIS data provider (OPC-UA subscription),
        NOT by WATERGUARD. WATERGUARD is READ-ONLY.

        Args:
            status: New SIS status
        """
        with self._lock:
            old_status = self._sis_status
            self._sis_status = status

            # Check for health change
            if old_status.overall_health != status.overall_health:
                logger.warning(
                    "SIS health changed: %s -> %s",
                    old_status.overall_health.value,
                    status.overall_health.value
                )

            # Callback
            if self._status_callback:
                try:
                    self._status_callback(status)
                except Exception as e:
                    logger.error("SIS status callback failed: %s", e)

    def update_boiler_safety_status(self, status: BoilerSafetyStatus) -> None:
        """
        Update boiler safety status from SIS data provider.

        Args:
            status: New boiler safety status
        """
        with self._lock:
            self._boiler_statuses[status.boiler_id] = status

    def add_active_trip(self, trip: ActiveTrip) -> None:
        """
        Add an active trip (called by SIS data provider).

        Args:
            trip: New active trip
        """
        with self._lock:
            self._active_trips[trip.trip_id] = trip
            self._stats["trips_observed"] += 1

            logger.critical(
                "SIS TRIP ACTIVE: %s - %s",
                trip.trip_type.value, trip.message
            )

            if self._trip_callback:
                try:
                    self._trip_callback(trip)
                except Exception as e:
                    logger.error("Trip callback failed: %s", e)

    def clear_trip(self, trip_id: str) -> None:
        """
        Clear a trip (called by SIS data provider when trip is reset).

        Args:
            trip_id: Trip to clear
        """
        with self._lock:
            if trip_id in self._active_trips:
                trip = self._active_trips.pop(trip_id)
                logger.info("SIS trip cleared: %s", trip.trip_type.value)

    def get_statistics(self) -> Dict[str, int]:
        """Get interface statistics."""
        with self._lock:
            return dict(self._stats)


# =============================================================================
# MODULE EXPORTS
# =============================================================================


__all__ = [
    # Enums
    "SISHealthStatus",
    "TripType",
    "InterlockStatus",
    # Models
    "ActiveTrip",
    "SISStatus",
    "BoilerSafetyStatus",
    # Classes
    "SISInterface",
]
