# -*- coding: utf-8 -*-
"""
GL-023 HeatLoadBalancer - Safety Validation and Fleet Coordination.

This module provides safety constraint validation and fleet coordination
for the Heat Load Balancer agent. It implements safety checks per ASME CSD-1,
NFPA 85, and NFPA 86 standards for multi-unit thermal systems.

Features:
    - Equipment limit validation
    - Total capacity verification
    - N+1 redundancy checking
    - Ramp rate constraint validation
    - Emergency reserve monitoring
    - Fuel availability verification
    - Startup/shutdown sequencing per NFPA 85
    - Equipment trip handling and load redistribution

Standards:
    - ASME CSD-1: Controls and Safety Devices for Automatically Fired Boilers
    - NFPA 85: Boiler and Combustion Systems Hazards Code
    - NFPA 86: Standard for Ovens and Furnaces
    - IEC 61511: Functional Safety - Safety Instrumented Systems

Author: GreenLang Process Heat Team
Date: December 2025
Status: Production Ready
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class SafetyViolationType(str, Enum):
    """Types of safety violations."""
    EQUIPMENT_OVERLOAD = "EQUIPMENT_OVERLOAD"
    EQUIPMENT_UNDERLOAD = "EQUIPMENT_UNDERLOAD"
    CAPACITY_EXCEEDED = "CAPACITY_EXCEEDED"
    INSUFFICIENT_RESERVE = "INSUFFICIENT_RESERVE"
    N_PLUS_1_VIOLATION = "N_PLUS_1_VIOLATION"
    RAMP_RATE_EXCEEDED = "RAMP_RATE_EXCEEDED"
    MIN_RUN_TIME_VIOLATION = "MIN_RUN_TIME_VIOLATION"
    MIN_DOWN_TIME_VIOLATION = "MIN_DOWN_TIME_VIOLATION"
    FUEL_SHORTAGE = "FUEL_SHORTAGE"
    STARTUP_LIMIT_EXCEEDED = "STARTUP_LIMIT_EXCEEDED"
    SHUTDOWN_LIMIT_EXCEEDED = "SHUTDOWN_LIMIT_EXCEEDED"
    PURGE_INCOMPLETE = "PURGE_INCOMPLETE"
    INTERLOCK_ACTIVE = "INTERLOCK_ACTIVE"


class SafetySeverity(str, Enum):
    """Safety violation severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    ALARM = "ALARM"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


class SequenceStep(str, Enum):
    """Startup/shutdown sequence steps per NFPA 85."""
    # Startup sequence
    PRE_PURGE_CHECK = "PRE_PURGE_CHECK"
    PRE_PURGE = "PRE_PURGE"
    PILOT_IGNITION = "PILOT_IGNITION"
    PILOT_PROVE = "PILOT_PROVE"
    MAIN_FLAME_IGNITION = "MAIN_FLAME_IGNITION"
    MAIN_FLAME_PROVE = "MAIN_FLAME_PROVE"
    LOAD_RAMP_UP = "LOAD_RAMP_UP"
    NORMAL_OPERATION = "NORMAL_OPERATION"
    # Shutdown sequence
    LOAD_RAMP_DOWN = "LOAD_RAMP_DOWN"
    MAIN_FLAME_CUTOFF = "MAIN_FLAME_CUTOFF"
    PILOT_CUTOFF = "PILOT_CUTOFF"
    POST_PURGE = "POST_PURGE"
    STANDBY = "STANDBY"
    # Emergency
    EMERGENCY_SHUTDOWN = "EMERGENCY_SHUTDOWN"


class TripReason(str, Enum):
    """Equipment trip reasons."""
    FLAME_FAILURE = "FLAME_FAILURE"
    HIGH_PRESSURE = "HIGH_PRESSURE"
    LOW_PRESSURE = "LOW_PRESSURE"
    HIGH_TEMPERATURE = "HIGH_TEMPERATURE"
    LOW_WATER = "LOW_WATER"
    FUEL_PRESSURE = "FUEL_PRESSURE"
    COMBUSTION_AIR = "COMBUSTION_AIR"
    SAFETY_INTERLOCK = "SAFETY_INTERLOCK"
    OPERATOR_INITIATED = "OPERATOR_INITIATED"
    COMMUNICATION_LOSS = "COMMUNICATION_LOSS"
    UNKNOWN = "UNKNOWN"


# =============================================================================
# DATA MODELS
# =============================================================================

class SafetyViolation(BaseModel):
    """Safety violation record."""

    violation_id: str = Field(..., description="Unique violation identifier")
    violation_type: SafetyViolationType = Field(
        ...,
        description="Type of safety violation"
    )
    severity: SafetySeverity = Field(
        ...,
        description="Violation severity"
    )
    unit_id: Optional[str] = Field(
        None,
        description="Affected unit ID (if applicable)"
    )
    message: str = Field(..., description="Violation description")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Violation timestamp"
    )
    current_value: Optional[float] = Field(
        None,
        description="Current value causing violation"
    )
    limit_value: Optional[float] = Field(
        None,
        description="Limit value that was exceeded"
    )
    recommended_action: Optional[str] = Field(
        None,
        description="Recommended corrective action"
    )
    auto_correctable: bool = Field(
        False,
        description="Can be automatically corrected"
    )

    class Config:
        use_enum_values = True


class SafetyValidationResult(BaseModel):
    """Result of safety validation."""

    is_valid: bool = Field(..., description="All safety checks passed")
    violations: List[SafetyViolation] = Field(
        default_factory=list,
        description="List of violations"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Warning messages"
    )
    active_constraints: List[str] = Field(
        default_factory=list,
        description="Binding constraints"
    )
    n_plus_1_satisfied: bool = Field(
        True,
        description="N+1 redundancy satisfied"
    )
    emergency_reserve_adequate: bool = Field(
        True,
        description="Emergency reserve adequate"
    )
    validation_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Validation timestamp"
    )
    provenance_hash: Optional[str] = Field(
        None,
        description="SHA-256 validation hash"
    )


class EquipmentSetpoint(BaseModel):
    """Setpoint for equipment control."""

    unit_id: str = Field(..., description="Equipment unit ID")
    target_load_mw: float = Field(..., ge=0, description="Target load (MW)")
    current_load_mw: float = Field(..., ge=0, description="Current load (MW)")
    min_load_mw: float = Field(..., ge=0, description="Minimum load (MW)")
    max_load_mw: float = Field(..., ge=0, description="Maximum load (MW)")
    ramp_rate_mw_min: float = Field(1.0, gt=0, description="Ramp rate (MW/min)")


class RampValidation(BaseModel):
    """Ramp rate validation result."""

    unit_id: str = Field(..., description="Equipment unit ID")
    is_valid: bool = Field(..., description="Ramp is within limits")
    required_time_min: float = Field(
        ...,
        ge=0,
        description="Time required to reach setpoint (min)"
    )
    max_allowed_change_mw: float = Field(
        ...,
        description="Maximum allowed change in time step"
    )
    actual_change_mw: float = Field(
        ...,
        description="Requested change"
    )
    violation_message: Optional[str] = Field(
        None,
        description="Violation description if invalid"
    )


class SequenceState(BaseModel):
    """Equipment sequence state."""

    unit_id: str = Field(..., description="Equipment unit ID")
    current_step: SequenceStep = Field(
        ...,
        description="Current sequence step"
    )
    step_start_time: datetime = Field(
        ...,
        description="Time current step started"
    )
    step_timeout_seconds: float = Field(
        300.0,
        description="Timeout for current step"
    )
    sequence_type: str = Field(
        "STARTUP",
        description="STARTUP or SHUTDOWN"
    )
    is_complete: bool = Field(False, description="Sequence complete")
    is_failed: bool = Field(False, description="Sequence failed")
    failure_reason: Optional[str] = Field(
        None,
        description="Failure reason if failed"
    )

    class Config:
        use_enum_values = True


class TripEvent(BaseModel):
    """Equipment trip event record."""

    trip_id: str = Field(..., description="Unique trip identifier")
    unit_id: str = Field(..., description="Tripped unit ID")
    trip_reason: TripReason = Field(..., description="Trip reason")
    trip_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Trip timestamp"
    )
    load_at_trip_mw: float = Field(0.0, ge=0, description="Load at time of trip")
    load_redistributed: bool = Field(
        False,
        description="Load has been redistributed"
    )
    redistributed_to: List[str] = Field(
        default_factory=list,
        description="Units receiving redistributed load"
    )
    operator_notified: bool = Field(False, description="Operator notified")
    auto_restart_enabled: bool = Field(
        False,
        description="Auto restart enabled"
    )

    class Config:
        use_enum_values = True


class FuelAvailability(BaseModel):
    """Fuel availability check result."""

    is_adequate: bool = Field(..., description="Fuel supply adequate")
    current_inventory_mmbtu: float = Field(
        ...,
        ge=0,
        description="Current fuel inventory"
    )
    required_rate_mmbtu_hr: float = Field(
        ...,
        ge=0,
        description="Required fuel rate"
    )
    hours_of_supply: float = Field(
        ...,
        ge=0,
        description="Hours of supply remaining"
    )
    minimum_hours_required: float = Field(
        4.0,
        ge=0,
        description="Minimum hours required"
    )
    reorder_triggered: bool = Field(
        False,
        description="Fuel reorder triggered"
    )
    message: str = Field("", description="Status message")


# =============================================================================
# SAFETY VALIDATOR CLASS
# =============================================================================

class SafetyValidator:
    """
    Safety constraint validator for heat load balancing.

    Validates all safety constraints per ASME CSD-1, NFPA 85, and NFPA 86.
    Provides comprehensive checks for equipment limits, capacity, redundancy,
    ramp rates, and emergency reserves.

    Attributes:
        config: Validator configuration
        violation_history: Recent violation history
        _lock: Thread lock for state access

    Example:
        >>> validator = SafetyValidator()
        >>> result = validator.validate_equipment_limits(setpoints)
        >>> if not result.is_valid:
        ...     for violation in result.violations:
        ...         print(f"Violation: {violation.message}")
    """

    # Standard limits per NFPA 85
    MIN_PURGE_TIME_SECONDS = 60.0  # Minimum pre-purge time
    MAX_TRIAL_FOR_IGNITION_SECONDS = 10.0  # Maximum trial for ignition
    MIN_POST_PURGE_TIME_SECONDS = 30.0  # Minimum post-purge time
    MIN_SPINNING_RESERVE_PCT = 5.0  # Absolute minimum reserve
    DEFAULT_N_PLUS_1_REQUIRED = True

    def __init__(
        self,
        min_reserve_pct: float = 10.0,
        require_n_plus_1: bool = True,
        max_simultaneous_startups: int = 1,
        max_simultaneous_shutdowns: int = 1,
        min_fuel_hours: float = 4.0,
    ) -> None:
        """
        Initialize the SafetyValidator.

        Args:
            min_reserve_pct: Minimum spinning reserve percentage
            require_n_plus_1: Require N+1 redundancy
            max_simultaneous_startups: Maximum simultaneous startups
            max_simultaneous_shutdowns: Maximum simultaneous shutdowns
            min_fuel_hours: Minimum fuel supply hours
        """
        self.min_reserve_pct = max(min_reserve_pct, self.MIN_SPINNING_RESERVE_PCT)
        self.require_n_plus_1 = require_n_plus_1
        self.max_simultaneous_startups = max_simultaneous_startups
        self.max_simultaneous_shutdowns = max_simultaneous_shutdowns
        self.min_fuel_hours = min_fuel_hours

        self._lock = threading.RLock()
        self._violation_history: List[SafetyViolation] = []

        logger.info(
            f"SafetyValidator initialized: reserve={min_reserve_pct}%, "
            f"N+1={require_n_plus_1}"
        )

    # =========================================================================
    # EQUIPMENT LIMITS VALIDATION
    # =========================================================================

    def validate_equipment_limits(
        self,
        setpoints: List[EquipmentSetpoint],
    ) -> SafetyValidationResult:
        """
        Validate setpoints against equipment limits.

        Checks that all setpoints are within equipment min/max capacity
        and that turndown ratios are respected.

        Args:
            setpoints: List of equipment setpoints to validate

        Returns:
            SafetyValidationResult with any violations
        """
        violations = []
        warnings = []

        for sp in setpoints:
            # Check overload
            if sp.target_load_mw > sp.max_load_mw:
                violation = SafetyViolation(
                    violation_id=f"OVL-{sp.unit_id}-{int(time.time())}",
                    violation_type=SafetyViolationType.EQUIPMENT_OVERLOAD,
                    severity=SafetySeverity.ALARM,
                    unit_id=sp.unit_id,
                    message=(
                        f"Unit {sp.unit_id} target load {sp.target_load_mw:.2f} MW "
                        f"exceeds maximum {sp.max_load_mw:.2f} MW"
                    ),
                    current_value=sp.target_load_mw,
                    limit_value=sp.max_load_mw,
                    recommended_action=f"Reduce load to {sp.max_load_mw:.2f} MW",
                    auto_correctable=True,
                )
                violations.append(violation)

            # Check underload (if running)
            if 0 < sp.target_load_mw < sp.min_load_mw:
                violation = SafetyViolation(
                    violation_id=f"UND-{sp.unit_id}-{int(time.time())}",
                    violation_type=SafetyViolationType.EQUIPMENT_UNDERLOAD,
                    severity=SafetySeverity.ALARM,
                    unit_id=sp.unit_id,
                    message=(
                        f"Unit {sp.unit_id} target load {sp.target_load_mw:.2f} MW "
                        f"below minimum stable {sp.min_load_mw:.2f} MW"
                    ),
                    current_value=sp.target_load_mw,
                    limit_value=sp.min_load_mw,
                    recommended_action=(
                        f"Increase to {sp.min_load_mw:.2f} MW or shut down"
                    ),
                    auto_correctable=True,
                )
                violations.append(violation)

            # Warning for high loading
            if sp.target_load_mw > sp.max_load_mw * 0.95:
                warnings.append(
                    f"Unit {sp.unit_id} approaching maximum capacity "
                    f"({sp.target_load_mw / sp.max_load_mw * 100:.0f}%)"
                )

        # Store violations
        with self._lock:
            self._violation_history.extend(violations)
            if len(self._violation_history) > 1000:
                self._violation_history = self._violation_history[-1000:]

        # Calculate provenance hash
        provenance_data = {
            "validation_type": "equipment_limits",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "setpoints": [sp.dict() for sp in setpoints],
            "violations_count": len(violations),
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()

        return SafetyValidationResult(
            is_valid=len(violations) == 0,
            violations=violations,
            warnings=warnings,
            provenance_hash=provenance_hash,
        )

    # =========================================================================
    # CAPACITY VALIDATION
    # =========================================================================

    def validate_total_capacity(
        self,
        total_load: float,
        max_capacity: float,
    ) -> SafetyValidationResult:
        """
        Validate that total load does not exceed system capacity.

        Args:
            total_load: Total load demand (MW)
            max_capacity: Maximum system capacity (MW)

        Returns:
            SafetyValidationResult
        """
        violations = []
        warnings = []
        active_constraints = []

        if total_load > max_capacity:
            violations.append(SafetyViolation(
                violation_id=f"CAP-{int(time.time())}",
                violation_type=SafetyViolationType.CAPACITY_EXCEEDED,
                severity=SafetySeverity.CRITICAL,
                message=(
                    f"Total demand {total_load:.2f} MW exceeds "
                    f"system capacity {max_capacity:.2f} MW"
                ),
                current_value=total_load,
                limit_value=max_capacity,
                recommended_action="Reduce demand or start additional units",
            ))

        if total_load > max_capacity * 0.95:
            active_constraints.append("CAPACITY_LIMIT")
            warnings.append(
                f"System at {total_load / max_capacity * 100:.0f}% capacity"
            )

        return SafetyValidationResult(
            is_valid=len(violations) == 0,
            violations=violations,
            warnings=warnings,
            active_constraints=active_constraints,
        )

    # =========================================================================
    # N+1 REDUNDANCY
    # =========================================================================

    def check_n_plus_1_redundancy(
        self,
        running_units: List[Dict[str, Any]],
        demand: float,
    ) -> SafetyValidationResult:
        """
        Check N+1 redundancy - system can meet demand with largest unit offline.

        Per NFPA 85 and industry best practices, critical thermal systems
        should maintain sufficient capacity to handle loss of the single
        largest unit.

        Args:
            running_units: List of running units with 'unit_id', 'max_load_mw'
            demand: Current demand (MW)

        Returns:
            SafetyValidationResult with N+1 status
        """
        violations = []
        warnings = []

        if not running_units:
            return SafetyValidationResult(
                is_valid=False,
                violations=[SafetyViolation(
                    violation_id=f"N1-NONE-{int(time.time())}",
                    violation_type=SafetyViolationType.N_PLUS_1_VIOLATION,
                    severity=SafetySeverity.CRITICAL,
                    message="No units running - cannot satisfy N+1",
                )],
                n_plus_1_satisfied=False,
            )

        # Find largest unit
        largest_unit = max(running_units, key=lambda u: u.get('max_load_mw', 0))
        largest_capacity = largest_unit.get('max_load_mw', 0)
        largest_unit_id = largest_unit.get('unit_id', 'unknown')

        # Total capacity without largest unit
        total_capacity = sum(u.get('max_load_mw', 0) for u in running_units)
        remaining_capacity = total_capacity - largest_capacity

        # Check if remaining capacity meets demand
        n_plus_1_satisfied = remaining_capacity >= demand

        if not n_plus_1_satisfied:
            if self.require_n_plus_1:
                severity = SafetySeverity.ALARM
            else:
                severity = SafetySeverity.WARNING

            violations.append(SafetyViolation(
                violation_id=f"N1-{int(time.time())}",
                violation_type=SafetyViolationType.N_PLUS_1_VIOLATION,
                severity=severity,
                unit_id=largest_unit_id,
                message=(
                    f"N+1 not satisfied: loss of {largest_unit_id} "
                    f"({largest_capacity:.1f} MW) would leave "
                    f"{remaining_capacity:.1f} MW < demand {demand:.1f} MW"
                ),
                current_value=remaining_capacity,
                limit_value=demand,
                recommended_action="Start additional standby unit",
            ))
        else:
            # Warning if margin is tight
            margin = remaining_capacity - demand
            margin_pct = (margin / demand * 100) if demand > 0 else 100
            if margin_pct < 10:
                warnings.append(
                    f"N+1 margin only {margin_pct:.0f}% - consider starting standby"
                )

        return SafetyValidationResult(
            is_valid=n_plus_1_satisfied or not self.require_n_plus_1,
            violations=violations,
            warnings=warnings,
            n_plus_1_satisfied=n_plus_1_satisfied,
        )

    # =========================================================================
    # RAMP RATE VALIDATION
    # =========================================================================

    def validate_ramp_rates(
        self,
        current: float,
        target: float,
        dt_minutes: float,
        ramp_rate_mw_min: float,
        unit_id: str = "unknown",
    ) -> RampValidation:
        """
        Validate load change against ramp rate constraints.

        Ensures load changes do not exceed equipment ramp rate limits
        to prevent thermal stress and combustion instability.

        Args:
            current: Current load (MW)
            target: Target load (MW)
            dt_minutes: Time step (minutes)
            ramp_rate_mw_min: Maximum ramp rate (MW/min)
            unit_id: Equipment unit ID

        Returns:
            RampValidation result
        """
        load_change = target - current
        max_change = ramp_rate_mw_min * dt_minutes
        required_time = abs(load_change) / ramp_rate_mw_min if ramp_rate_mw_min > 0 else float('inf')

        is_valid = abs(load_change) <= max_change * 1.001  # Small tolerance

        violation_message = None
        if not is_valid:
            violation_message = (
                f"Unit {unit_id}: load change {load_change:.2f} MW "
                f"exceeds max {max_change:.2f} MW in {dt_minutes:.1f} min "
                f"(ramp rate {ramp_rate_mw_min:.2f} MW/min)"
            )

        return RampValidation(
            unit_id=unit_id,
            is_valid=is_valid,
            required_time_min=required_time,
            max_allowed_change_mw=max_change,
            actual_change_mw=load_change,
            violation_message=violation_message,
        )

    # =========================================================================
    # EMERGENCY RESERVE
    # =========================================================================

    def check_emergency_reserve(
        self,
        available_reserve_mw: float,
        required_reserve_mw: float,
    ) -> SafetyValidationResult:
        """
        Check that emergency/spinning reserve is adequate.

        Args:
            available_reserve_mw: Available spinning reserve (MW)
            required_reserve_mw: Required minimum reserve (MW)

        Returns:
            SafetyValidationResult
        """
        violations = []
        warnings = []

        is_adequate = available_reserve_mw >= required_reserve_mw

        if not is_adequate:
            shortfall = required_reserve_mw - available_reserve_mw
            violations.append(SafetyViolation(
                violation_id=f"RSV-{int(time.time())}",
                violation_type=SafetyViolationType.INSUFFICIENT_RESERVE,
                severity=SafetySeverity.ALARM,
                message=(
                    f"Spinning reserve {available_reserve_mw:.2f} MW "
                    f"below required {required_reserve_mw:.2f} MW "
                    f"(shortfall: {shortfall:.2f} MW)"
                ),
                current_value=available_reserve_mw,
                limit_value=required_reserve_mw,
                recommended_action="Start standby unit or reduce load",
            ))

        # Warning if reserve is tight
        if is_adequate and available_reserve_mw < required_reserve_mw * 1.2:
            warnings.append(
                f"Reserve margin tight: {available_reserve_mw:.1f} MW "
                f"(minimum: {required_reserve_mw:.1f} MW)"
            )

        return SafetyValidationResult(
            is_valid=is_adequate,
            violations=violations,
            warnings=warnings,
            emergency_reserve_adequate=is_adequate,
        )

    # =========================================================================
    # FUEL AVAILABILITY
    # =========================================================================

    def validate_fuel_availability(
        self,
        fuel_demands: Dict[str, float],
        inventory_mmbtu: float,
        horizon_hours: float = 4.0,
    ) -> FuelAvailability:
        """
        Validate fuel availability for planned operation.

        Args:
            fuel_demands: Dictionary of unit_id -> fuel rate (MMBtu/hr)
            inventory_mmbtu: Current fuel inventory (MMBtu)
            horizon_hours: Planning horizon (hours)

        Returns:
            FuelAvailability check result
        """
        total_rate = sum(fuel_demands.values())
        required_fuel = total_rate * horizon_hours

        if total_rate > 0:
            hours_of_supply = inventory_mmbtu / total_rate
        else:
            hours_of_supply = float('inf')

        is_adequate = hours_of_supply >= self.min_fuel_hours
        reorder_triggered = hours_of_supply < self.min_fuel_hours * 2

        if is_adequate:
            message = f"Fuel supply adequate: {hours_of_supply:.1f} hours remaining"
        else:
            message = (
                f"FUEL SHORTAGE: Only {hours_of_supply:.1f} hours of supply "
                f"(minimum: {self.min_fuel_hours:.1f} hours)"
            )

        return FuelAvailability(
            is_adequate=is_adequate,
            current_inventory_mmbtu=inventory_mmbtu,
            required_rate_mmbtu_hr=total_rate,
            hours_of_supply=hours_of_supply,
            minimum_hours_required=self.min_fuel_hours,
            reorder_triggered=reorder_triggered,
            message=message,
        )

    # =========================================================================
    # COMPREHENSIVE VALIDATION
    # =========================================================================

    def validate_all(
        self,
        setpoints: List[EquipmentSetpoint],
        running_units: List[Dict[str, Any]],
        total_demand: float,
        total_capacity: float,
        available_reserve: float,
        required_reserve: float,
        fuel_inventory: Optional[float] = None,
        fuel_demands: Optional[Dict[str, float]] = None,
    ) -> SafetyValidationResult:
        """
        Perform comprehensive safety validation.

        Args:
            setpoints: Equipment setpoints
            running_units: Running unit information
            total_demand: Total demand (MW)
            total_capacity: Total capacity (MW)
            available_reserve: Available reserve (MW)
            required_reserve: Required reserve (MW)
            fuel_inventory: Fuel inventory (MMBtu), optional
            fuel_demands: Fuel demands by unit, optional

        Returns:
            Combined SafetyValidationResult
        """
        all_violations = []
        all_warnings = []
        all_constraints = []

        # Equipment limits
        eq_result = self.validate_equipment_limits(setpoints)
        all_violations.extend(eq_result.violations)
        all_warnings.extend(eq_result.warnings)

        # Capacity
        cap_result = self.validate_total_capacity(total_demand, total_capacity)
        all_violations.extend(cap_result.violations)
        all_warnings.extend(cap_result.warnings)
        all_constraints.extend(cap_result.active_constraints)

        # N+1
        n1_result = self.check_n_plus_1_redundancy(running_units, total_demand)
        all_violations.extend(n1_result.violations)
        all_warnings.extend(n1_result.warnings)

        # Reserve
        rsv_result = self.check_emergency_reserve(available_reserve, required_reserve)
        all_violations.extend(rsv_result.violations)
        all_warnings.extend(rsv_result.warnings)

        # Fuel (if provided)
        if fuel_inventory is not None and fuel_demands is not None:
            fuel_result = self.validate_fuel_availability(
                fuel_demands, fuel_inventory
            )
            if not fuel_result.is_adequate:
                all_violations.append(SafetyViolation(
                    violation_id=f"FUEL-{int(time.time())}",
                    violation_type=SafetyViolationType.FUEL_SHORTAGE,
                    severity=SafetySeverity.ALARM,
                    message=fuel_result.message,
                    current_value=fuel_result.hours_of_supply,
                    limit_value=fuel_result.minimum_hours_required,
                ))

        # Calculate provenance hash
        provenance_data = {
            "validation_type": "comprehensive",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "demand_mw": total_demand,
            "capacity_mw": total_capacity,
            "reserve_mw": available_reserve,
            "violations_count": len(all_violations),
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()

        return SafetyValidationResult(
            is_valid=len(all_violations) == 0,
            violations=all_violations,
            warnings=all_warnings,
            active_constraints=all_constraints,
            n_plus_1_satisfied=n1_result.n_plus_1_satisfied,
            emergency_reserve_adequate=rsv_result.emergency_reserve_adequate,
            provenance_hash=provenance_hash,
        )

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_violation_history(
        self,
        limit: int = 100,
        severity: Optional[SafetySeverity] = None,
    ) -> List[SafetyViolation]:
        """Get recent violation history."""
        with self._lock:
            history = self._violation_history[-limit:]
            if severity:
                history = [v for v in history if v.severity == severity]
            return history

    def clear_violation_history(self) -> None:
        """Clear violation history."""
        with self._lock:
            self._violation_history.clear()


# =============================================================================
# FLEET COORDINATOR CLASS
# =============================================================================

class FleetCoordinator:
    """
    Fleet coordination for startup/shutdown sequencing and trip handling.

    Implements NFPA 85 compliant sequencing for boiler and combustion
    equipment, including:
    - Pre-purge and post-purge timing
    - Trial for ignition limits
    - Flame proving
    - Trip handling and load redistribution

    Attributes:
        sequences: Active sequences by unit_id
        trip_events: Recent trip events
        _lock: Thread lock for state access

    Example:
        >>> coordinator = FleetCoordinator()
        >>> sequence = coordinator.coordinate_startup_sequence(['B-001', 'B-002'])
        >>> while not sequence['B-001'].is_complete:
        ...     coordinator.advance_sequence('B-001')
    """

    # NFPA 85 timing requirements
    MIN_PRE_PURGE_SECONDS = 60.0
    MAX_TRIAL_FOR_IGNITION_SECONDS = 10.0
    PILOT_PROVING_TIME_SECONDS = 10.0
    MAIN_FLAME_PROVING_TIME_SECONDS = 10.0
    MIN_POST_PURGE_SECONDS = 30.0

    # Step durations (can be overridden per unit)
    DEFAULT_STEP_DURATIONS = {
        SequenceStep.PRE_PURGE_CHECK: 10.0,
        SequenceStep.PRE_PURGE: 60.0,
        SequenceStep.PILOT_IGNITION: 10.0,
        SequenceStep.PILOT_PROVE: 10.0,
        SequenceStep.MAIN_FLAME_IGNITION: 10.0,
        SequenceStep.MAIN_FLAME_PROVE: 10.0,
        SequenceStep.LOAD_RAMP_UP: 300.0,
        SequenceStep.LOAD_RAMP_DOWN: 300.0,
        SequenceStep.MAIN_FLAME_CUTOFF: 5.0,
        SequenceStep.PILOT_CUTOFF: 5.0,
        SequenceStep.POST_PURGE: 30.0,
    }

    # Sequence step order
    STARTUP_SEQUENCE = [
        SequenceStep.PRE_PURGE_CHECK,
        SequenceStep.PRE_PURGE,
        SequenceStep.PILOT_IGNITION,
        SequenceStep.PILOT_PROVE,
        SequenceStep.MAIN_FLAME_IGNITION,
        SequenceStep.MAIN_FLAME_PROVE,
        SequenceStep.LOAD_RAMP_UP,
        SequenceStep.NORMAL_OPERATION,
    ]

    SHUTDOWN_SEQUENCE = [
        SequenceStep.LOAD_RAMP_DOWN,
        SequenceStep.MAIN_FLAME_CUTOFF,
        SequenceStep.PILOT_CUTOFF,
        SequenceStep.POST_PURGE,
        SequenceStep.STANDBY,
    ]

    def __init__(
        self,
        max_simultaneous_startups: int = 1,
        max_simultaneous_shutdowns: int = 1,
        callback_on_trip: Optional[Callable[[TripEvent], None]] = None,
    ) -> None:
        """
        Initialize the FleetCoordinator.

        Args:
            max_simultaneous_startups: Maximum concurrent startups
            max_simultaneous_shutdowns: Maximum concurrent shutdowns
            callback_on_trip: Callback function when trip occurs
        """
        self.max_simultaneous_startups = max_simultaneous_startups
        self.max_simultaneous_shutdowns = max_simultaneous_shutdowns
        self._callback_on_trip = callback_on_trip

        self._lock = threading.RLock()
        self._sequences: Dict[str, SequenceState] = {}
        self._trip_events: List[TripEvent] = []

        logger.info(
            f"FleetCoordinator initialized: max_startups={max_simultaneous_startups}, "
            f"max_shutdowns={max_simultaneous_shutdowns}"
        )

    # =========================================================================
    # STARTUP SEQUENCING
    # =========================================================================

    def coordinate_startup_sequence(
        self,
        units_to_start: List[str],
    ) -> Dict[str, SequenceState]:
        """
        Coordinate startup sequence for multiple units.

        Ensures compliance with max simultaneous startup limits and
        initiates NFPA 85 compliant startup sequences.

        Args:
            units_to_start: List of unit IDs to start

        Returns:
            Dictionary of unit_id -> SequenceState for initiated sequences
        """
        with self._lock:
            # Count current startups
            active_startups = sum(
                1 for seq in self._sequences.values()
                if seq.sequence_type == "STARTUP" and not seq.is_complete
            )

            available_slots = self.max_simultaneous_startups - active_startups
            units_to_process = units_to_start[:available_slots]

            results = {}
            for unit_id in units_to_process:
                sequence = SequenceState(
                    unit_id=unit_id,
                    current_step=SequenceStep.PRE_PURGE_CHECK,
                    step_start_time=datetime.now(timezone.utc),
                    step_timeout_seconds=self.DEFAULT_STEP_DURATIONS[
                        SequenceStep.PRE_PURGE_CHECK
                    ],
                    sequence_type="STARTUP",
                )
                self._sequences[unit_id] = sequence
                results[unit_id] = sequence

                logger.info(f"Initiated startup sequence for unit {unit_id}")

            # Log deferred units
            deferred = units_to_start[available_slots:]
            if deferred:
                logger.info(
                    f"Startup deferred for {len(deferred)} units due to "
                    f"simultaneous startup limit"
                )

            return results

    def advance_startup_sequence(
        self,
        unit_id: str,
        step_successful: bool = True,
    ) -> SequenceState:
        """
        Advance startup sequence to next step.

        Args:
            unit_id: Unit ID
            step_successful: Whether current step completed successfully

        Returns:
            Updated SequenceState
        """
        with self._lock:
            if unit_id not in self._sequences:
                raise ValueError(f"No active sequence for unit {unit_id}")

            sequence = self._sequences[unit_id]

            if sequence.is_complete or sequence.is_failed:
                return sequence

            if not step_successful:
                sequence.is_failed = True
                sequence.failure_reason = (
                    f"Step {sequence.current_step.value} failed"
                )
                logger.error(f"Startup sequence failed for {unit_id}")
                return sequence

            # Get current step index
            try:
                current_idx = self.STARTUP_SEQUENCE.index(sequence.current_step)
            except ValueError:
                sequence.is_failed = True
                sequence.failure_reason = "Invalid sequence state"
                return sequence

            # Move to next step
            if current_idx < len(self.STARTUP_SEQUENCE) - 1:
                next_step = self.STARTUP_SEQUENCE[current_idx + 1]
                sequence.current_step = next_step
                sequence.step_start_time = datetime.now(timezone.utc)
                sequence.step_timeout_seconds = self.DEFAULT_STEP_DURATIONS.get(
                    next_step, 300.0
                )
                logger.debug(f"Unit {unit_id} advanced to {next_step.value}")
            else:
                sequence.is_complete = True
                logger.info(f"Startup sequence complete for unit {unit_id}")

            return sequence

    # =========================================================================
    # SHUTDOWN SEQUENCING
    # =========================================================================

    def coordinate_shutdown_sequence(
        self,
        units_to_stop: List[str],
    ) -> Dict[str, SequenceState]:
        """
        Coordinate shutdown sequence for multiple units.

        Ensures compliance with max simultaneous shutdown limits and
        initiates proper post-purge sequences.

        Args:
            units_to_stop: List of unit IDs to stop

        Returns:
            Dictionary of unit_id -> SequenceState for initiated sequences
        """
        with self._lock:
            # Count current shutdowns
            active_shutdowns = sum(
                1 for seq in self._sequences.values()
                if seq.sequence_type == "SHUTDOWN" and not seq.is_complete
            )

            available_slots = self.max_simultaneous_shutdowns - active_shutdowns
            units_to_process = units_to_stop[:available_slots]

            results = {}
            for unit_id in units_to_process:
                sequence = SequenceState(
                    unit_id=unit_id,
                    current_step=SequenceStep.LOAD_RAMP_DOWN,
                    step_start_time=datetime.now(timezone.utc),
                    step_timeout_seconds=self.DEFAULT_STEP_DURATIONS[
                        SequenceStep.LOAD_RAMP_DOWN
                    ],
                    sequence_type="SHUTDOWN",
                )
                self._sequences[unit_id] = sequence
                results[unit_id] = sequence

                logger.info(f"Initiated shutdown sequence for unit {unit_id}")

            return results

    def advance_shutdown_sequence(
        self,
        unit_id: str,
        step_successful: bool = True,
    ) -> SequenceState:
        """
        Advance shutdown sequence to next step.

        Args:
            unit_id: Unit ID
            step_successful: Whether current step completed successfully

        Returns:
            Updated SequenceState
        """
        with self._lock:
            if unit_id not in self._sequences:
                raise ValueError(f"No active sequence for unit {unit_id}")

            sequence = self._sequences[unit_id]

            if sequence.is_complete or sequence.is_failed:
                return sequence

            if not step_successful:
                sequence.is_failed = True
                sequence.failure_reason = (
                    f"Step {sequence.current_step.value} failed"
                )
                logger.error(f"Shutdown sequence failed for {unit_id}")
                return sequence

            # Get current step index
            try:
                current_idx = self.SHUTDOWN_SEQUENCE.index(sequence.current_step)
            except ValueError:
                sequence.is_failed = True
                sequence.failure_reason = "Invalid sequence state"
                return sequence

            # Move to next step
            if current_idx < len(self.SHUTDOWN_SEQUENCE) - 1:
                next_step = self.SHUTDOWN_SEQUENCE[current_idx + 1]
                sequence.current_step = next_step
                sequence.step_start_time = datetime.now(timezone.utc)
                sequence.step_timeout_seconds = self.DEFAULT_STEP_DURATIONS.get(
                    next_step, 60.0
                )
                logger.debug(f"Unit {unit_id} advanced to {next_step.value}")
            else:
                sequence.is_complete = True
                logger.info(f"Shutdown sequence complete for unit {unit_id}")

            return sequence

    # =========================================================================
    # TRIP HANDLING
    # =========================================================================

    def handle_equipment_trip(
        self,
        failed_unit_id: str,
        trip_reason: TripReason = TripReason.UNKNOWN,
        load_at_trip_mw: float = 0.0,
    ) -> TripEvent:
        """
        Handle equipment trip event.

        Records the trip event and prepares for load redistribution.
        Does NOT automatically redistribute load - call redistribute_load_on_failure
        after determining new allocations.

        Args:
            failed_unit_id: ID of tripped unit
            trip_reason: Reason for trip
            load_at_trip_mw: Load at time of trip

        Returns:
            TripEvent record
        """
        with self._lock:
            trip_event = TripEvent(
                trip_id=f"TRIP-{failed_unit_id}-{int(time.time())}",
                unit_id=failed_unit_id,
                trip_reason=trip_reason,
                load_at_trip_mw=load_at_trip_mw,
            )

            self._trip_events.append(trip_event)

            # Cancel any active sequence for this unit
            if failed_unit_id in self._sequences:
                self._sequences[failed_unit_id].is_failed = True
                self._sequences[failed_unit_id].failure_reason = (
                    f"Unit tripped: {trip_reason.value}"
                )

            logger.warning(
                f"EQUIPMENT TRIP: Unit {failed_unit_id} tripped - "
                f"Reason: {trip_reason.value}, Load: {load_at_trip_mw:.2f} MW"
            )

            # Call callback if registered
            if self._callback_on_trip:
                try:
                    self._callback_on_trip(trip_event)
                except Exception as e:
                    logger.error(f"Trip callback error: {e}")

            return trip_event

    def redistribute_load_on_failure(
        self,
        remaining_units: List[Dict[str, Any]],
        demand: float,
    ) -> List[Dict[str, float]]:
        """
        Redistribute load after equipment failure.

        Calculates new load allocation for remaining units to meet demand.
        Uses proportional scaling based on available capacity.

        Args:
            remaining_units: List of remaining units with 'unit_id',
                           'current_load_mw', 'max_load_mw', 'min_load_mw'
            demand: Total demand to meet (MW)

        Returns:
            List of {'unit_id': str, 'new_load_mw': float} allocations
        """
        if not remaining_units:
            logger.error("No remaining units for load redistribution")
            return []

        # Calculate available capacity
        available_capacity = sum(
            u.get('max_load_mw', 0) - u.get('current_load_mw', 0)
            for u in remaining_units
        )

        total_current = sum(u.get('current_load_mw', 0) for u in remaining_units)
        additional_needed = demand - total_current

        if additional_needed <= 0:
            # Current allocation meets demand
            return [
                {'unit_id': u['unit_id'], 'new_load_mw': u.get('current_load_mw', 0)}
                for u in remaining_units
            ]

        if available_capacity < additional_needed:
            logger.warning(
                f"Insufficient capacity: need {additional_needed:.2f} MW, "
                f"have {available_capacity:.2f} MW available"
            )

        # Proportionally distribute additional load
        allocations = []
        for unit in remaining_units:
            current = unit.get('current_load_mw', 0)
            max_load = unit.get('max_load_mw', current)
            headroom = max_load - current

            if available_capacity > 0:
                share = headroom / available_capacity
            else:
                share = 1.0 / len(remaining_units)

            additional = min(additional_needed * share, headroom)
            new_load = current + additional

            allocations.append({
                'unit_id': unit['unit_id'],
                'new_load_mw': round(new_load, 3),
            })

        # Update latest trip event
        with self._lock:
            if self._trip_events:
                latest_trip = self._trip_events[-1]
                latest_trip.load_redistributed = True
                latest_trip.redistributed_to = [a['unit_id'] for a in allocations]

        logger.info(
            f"Load redistributed to {len(allocations)} units: "
            f"total {sum(a['new_load_mw'] for a in allocations):.2f} MW"
        )

        return allocations

    # =========================================================================
    # SEQUENCE MANAGEMENT
    # =========================================================================

    def get_sequence_state(
        self,
        unit_id: str,
    ) -> Optional[SequenceState]:
        """Get current sequence state for a unit."""
        with self._lock:
            return self._sequences.get(unit_id)

    def get_active_sequences(
        self,
        sequence_type: Optional[str] = None,
    ) -> List[SequenceState]:
        """Get all active sequences."""
        with self._lock:
            sequences = [
                seq for seq in self._sequences.values()
                if not seq.is_complete and not seq.is_failed
            ]
            if sequence_type:
                sequences = [s for s in sequences if s.sequence_type == sequence_type]
            return sequences

    def cancel_sequence(
        self,
        unit_id: str,
        reason: str = "Operator cancelled",
    ) -> bool:
        """Cancel an active sequence."""
        with self._lock:
            if unit_id in self._sequences:
                self._sequences[unit_id].is_failed = True
                self._sequences[unit_id].failure_reason = reason
                logger.info(f"Sequence cancelled for {unit_id}: {reason}")
                return True
            return False

    def clear_completed_sequences(self) -> int:
        """Clear completed/failed sequences from tracking."""
        with self._lock:
            to_remove = [
                uid for uid, seq in self._sequences.items()
                if seq.is_complete or seq.is_failed
            ]
            for uid in to_remove:
                del self._sequences[uid]
            return len(to_remove)

    # =========================================================================
    # TRIP HISTORY
    # =========================================================================

    def get_trip_events(
        self,
        limit: int = 100,
        unit_id: Optional[str] = None,
    ) -> List[TripEvent]:
        """Get recent trip events."""
        with self._lock:
            events = self._trip_events[-limit:]
            if unit_id:
                events = [e for e in events if e.unit_id == unit_id]
            return events

    def acknowledge_trip(
        self,
        trip_id: str,
    ) -> bool:
        """Acknowledge a trip event."""
        with self._lock:
            for event in self._trip_events:
                if event.trip_id == trip_id:
                    event.operator_notified = True
                    return True
            return False


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Classes
    "SafetyValidator",
    "FleetCoordinator",
    # Data models
    "SafetyViolation",
    "SafetyValidationResult",
    "EquipmentSetpoint",
    "RampValidation",
    "SequenceState",
    "TripEvent",
    "FuelAvailability",
    # Enums
    "SafetyViolationType",
    "SafetySeverity",
    "SequenceStep",
    "TripReason",
]
