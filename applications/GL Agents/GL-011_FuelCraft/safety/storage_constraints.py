# -*- coding: utf-8 -*-
"""
StorageConstraints - Tank safety validators for GL-011 FuelCraft.

This module implements storage safety validators including flash point verification,
vapor pressure limits, tank heel and fill constraints, overfill protection, and
temperature range validation.

Reference Standards:
    - NFPA 30: Flammable and Combustible Liquids Code
    - API 650: Welded Tanks for Oil Storage
    - API 2000: Venting Atmospheric and Low-Pressure Storage Tanks
    - API 2610: Design, Construction, Operation, Maintenance of Terminal Facilities

Author: GL-BackendDeveloper
Date: 2025-01-01
Version: 1.0.0
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from enum import Enum
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


class SafetyAction(str, Enum):
    """Safety action to take on constraint violation."""
    BLOCK = "block"
    WARN = "warn"
    ALARM = "alarm"
    SHUTDOWN = "shutdown"


class StorageViolation(BaseModel):
    """Record of a storage constraint violation."""
    constraint_id: str = Field(...)
    constraint_name: str = Field(...)
    tank_id: str = Field(...)
    parameter: str = Field(...)
    actual_value: float = Field(...)
    limit_value: float = Field(...)
    limit_type: str = Field(...)
    action: SafetyAction = Field(...)
    message: str = Field(...)
    reference_standard: Optional[str] = Field(None)


class StorageValidationResult(BaseModel):
    """Result of storage constraint validation."""
    validation_id: str = Field(...)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tank_id: str = Field(...)
    is_safe: bool = Field(...)
    violations: List[StorageViolation] = Field(default_factory=list)
    actions_required: List[SafetyAction] = Field(default_factory=list)
    provenance_hash: str = Field(...)


class TankConfig(BaseModel):
    """Tank configuration for storage constraints."""
    tank_id: str = Field(...)
    capacity_m3: float = Field(..., ge=0)
    min_heel_pct: float = Field(5.0, ge=0, le=100)
    max_fill_pct: float = Field(95.0, ge=0, le=100)
    flash_point_min_c: float = Field(60.0)
    max_temp_c: float = Field(90.0)
    min_temp_c: float = Field(10.0)
    has_overfill_protection: bool = Field(True)
    rvp_max_psi: Optional[float] = Field(None)


class FlashPointValidator:
    """Flash point constraint validator per NFPA 30."""

    COMBUSTIBLE_THRESHOLD_C = 37.8  # 100F - Class II
    FLAMMABLE_THRESHOLD_C = 22.8   # 73F - Class I

    def __init__(self, min_flash_point_c: float = 60.0):
        """Initialize with minimum flash point requirement."""
        self._min_flash_point = min_flash_point_c
        logger.info(f"FlashPointValidator initialized: min_flash_point={min_flash_point_c}C")

    def validate(self, flash_point_c: float, tank_id: str) -> StorageValidationResult:
        """Validate flash point meets safety requirements."""
        validation_id = hashlib.sha256(
            f"flash|{flash_point_c}|{tank_id}|{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()[:16]

        violations = []
        actions = []

        if flash_point_c < self._min_flash_point:
            severity = SafetyAction.SHUTDOWN if flash_point_c < self.FLAMMABLE_THRESHOLD_C else SafetyAction.BLOCK
            violations.append(StorageViolation(
                constraint_id="FP_MIN", constraint_name="Flash Point Minimum",
                tank_id=tank_id, parameter="flash_point_c",
                actual_value=flash_point_c, limit_value=self._min_flash_point,
                limit_type="min", action=severity,
                message=f"Flash point {flash_point_c}C below minimum {self._min_flash_point}C",
                reference_standard="NFPA 30 Chapter 4"
            ))
            actions.append(severity)

        return StorageValidationResult(
            validation_id=validation_id, tank_id=tank_id,
            is_safe=len(violations) == 0, violations=violations,
            actions_required=actions,
            provenance_hash=hashlib.sha256(
                json.dumps({"id": validation_id, "fp": flash_point_c, "tank": tank_id}, sort_keys=True).encode()
            ).hexdigest()
        )


class VaporPressureValidator:
    """Reid Vapor Pressure constraint validator per API 2000."""

    def __init__(self, max_rvp_psi: float = 11.0):
        """Initialize with maximum RVP."""
        self._max_rvp = max_rvp_psi
        logger.info(f"VaporPressureValidator initialized: max_rvp={max_rvp_psi} psi")

    def validate(self, rvp_psi: float, tank_id: str, ambient_temp_c: float = 25.0) -> StorageValidationResult:
        """Validate vapor pressure for safe storage."""
        validation_id = hashlib.sha256(
            f"rvp|{rvp_psi}|{tank_id}|{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()[:16]

        violations = []
        actions = []

        temp_factor = 1.0 + (ambient_temp_c - 25.0) * 0.01
        effective_limit = self._max_rvp / temp_factor

        if rvp_psi > effective_limit:
            violations.append(StorageViolation(
                constraint_id="RVP_MAX", constraint_name="Vapor Pressure Maximum",
                tank_id=tank_id, parameter="rvp_psi",
                actual_value=rvp_psi, limit_value=effective_limit,
                limit_type="max", action=SafetyAction.BLOCK,
                message=f"RVP {rvp_psi} psi exceeds temp-adjusted limit {effective_limit:.1f} psi",
                reference_standard="API 2000"
            ))
            actions.append(SafetyAction.BLOCK)

        return StorageValidationResult(
            validation_id=validation_id, tank_id=tank_id,
            is_safe=len(violations) == 0, violations=violations,
            actions_required=actions,
            provenance_hash=hashlib.sha256(
                json.dumps({"id": validation_id, "rvp": rvp_psi, "tank": tank_id}, sort_keys=True).encode()
            ).hexdigest()
        )


class MinHeelConstraint:
    """Minimum tank heel constraint to prevent pump cavitation."""

    def __init__(self, min_heel_pct: float = 5.0):
        """Initialize with minimum heel percentage."""
        self._min_heel_pct = min_heel_pct
        logger.info(f"MinHeelConstraint initialized: min_heel={min_heel_pct}%")

    def validate(
        self, current_level_pct: float, tank_id: str, is_transfer_active: bool = False
    ) -> StorageValidationResult:
        """Validate tank level maintains minimum heel."""
        validation_id = hashlib.sha256(
            f"heel|{current_level_pct}|{tank_id}|{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()[:16]

        violations = []
        actions = []

        warning_threshold = self._min_heel_pct * 1.5

        if current_level_pct < self._min_heel_pct:
            action = SafetyAction.SHUTDOWN if is_transfer_active else SafetyAction.BLOCK
            violations.append(StorageViolation(
                constraint_id="HEEL_MIN", constraint_name="Minimum Heel",
                tank_id=tank_id, parameter="level_pct",
                actual_value=current_level_pct, limit_value=self._min_heel_pct,
                limit_type="min", action=action,
                message=f"Tank level {current_level_pct}% below minimum heel {self._min_heel_pct}%",
                reference_standard="API 2610 Section 8"
            ))
            actions.append(action)
        elif current_level_pct < warning_threshold:
            violations.append(StorageViolation(
                constraint_id="HEEL_WARN", constraint_name="Heel Warning",
                tank_id=tank_id, parameter="level_pct",
                actual_value=current_level_pct, limit_value=warning_threshold,
                limit_type="min", action=SafetyAction.WARN,
                message=f"Tank level {current_level_pct}% approaching minimum heel",
                reference_standard="API 2610"
            ))
            actions.append(SafetyAction.WARN)

        is_safe = not any(v.action in [SafetyAction.BLOCK, SafetyAction.SHUTDOWN] for v in violations)

        return StorageValidationResult(
            validation_id=validation_id, tank_id=tank_id,
            is_safe=is_safe, violations=violations, actions_required=actions,
            provenance_hash=hashlib.sha256(
                json.dumps({"id": validation_id, "level": current_level_pct, "tank": tank_id}, sort_keys=True).encode()
            ).hexdigest()
        )


class MaxFillConstraint:
    """Maximum fill constraint to prevent tank overfill."""

    def __init__(self, max_fill_pct: float = 95.0, high_high_pct: float = 98.0):
        """Initialize with maximum fill percentages."""
        self._max_fill_pct = max_fill_pct
        self._high_high_pct = high_high_pct
        logger.info(f"MaxFillConstraint initialized: max_fill={max_fill_pct}%, HH={high_high_pct}%")

    def validate(
        self, current_level_pct: float, tank_id: str, incoming_volume_m3: float = 0.0,
        tank_capacity_m3: float = 1000.0
    ) -> StorageValidationResult:
        """Validate tank level with incoming volume does not exceed max fill."""
        validation_id = hashlib.sha256(
            f"fill|{current_level_pct}|{incoming_volume_m3}|{tank_id}|{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()[:16]

        violations = []
        actions = []

        projected_level = current_level_pct + (incoming_volume_m3 / tank_capacity_m3 * 100.0)

        if current_level_pct >= self._high_high_pct:
            violations.append(StorageViolation(
                constraint_id="FILL_HH", constraint_name="High-High Level",
                tank_id=tank_id, parameter="level_pct",
                actual_value=current_level_pct, limit_value=self._high_high_pct,
                limit_type="max", action=SafetyAction.SHUTDOWN,
                message=f"CRITICAL: Tank at {current_level_pct}% - HIGH-HIGH limit reached",
                reference_standard="API 2350"
            ))
            actions.append(SafetyAction.SHUTDOWN)
        elif projected_level > self._max_fill_pct:
            violations.append(StorageViolation(
                constraint_id="FILL_MAX", constraint_name="Maximum Fill",
                tank_id=tank_id, parameter="projected_level_pct",
                actual_value=projected_level, limit_value=self._max_fill_pct,
                limit_type="max", action=SafetyAction.BLOCK,
                message=f"Projected level {projected_level:.1f}% would exceed max fill {self._max_fill_pct}%",
                reference_standard="API 2350"
            ))
            actions.append(SafetyAction.BLOCK)
        elif current_level_pct > self._max_fill_pct * 0.95:
            violations.append(StorageViolation(
                constraint_id="FILL_WARN", constraint_name="High Level Warning",
                tank_id=tank_id, parameter="level_pct",
                actual_value=current_level_pct, limit_value=self._max_fill_pct,
                limit_type="max", action=SafetyAction.WARN,
                message=f"Tank level {current_level_pct}% approaching maximum",
                reference_standard="API 2350"
            ))
            actions.append(SafetyAction.WARN)

        is_safe = not any(v.action in [SafetyAction.BLOCK, SafetyAction.SHUTDOWN] for v in violations)

        return StorageValidationResult(
            validation_id=validation_id, tank_id=tank_id,
            is_safe=is_safe, violations=violations, actions_required=actions,
            provenance_hash=hashlib.sha256(
                json.dumps({"id": validation_id, "level": current_level_pct, "projected": projected_level}, sort_keys=True).encode()
            ).hexdigest()
        )


class OverfillProtectionPolicy:
    """Overfill protection policy per API 2350."""

    def __init__(self, requires_independent_hh: bool = True, requires_sis_tested: bool = True):
        """Initialize overfill protection policy."""
        self._requires_independent_hh = requires_independent_hh
        self._requires_sis_tested = requires_sis_tested
        logger.info("OverfillProtectionPolicy initialized per API 2350")

    def validate_protection_status(
        self,
        tank_id: str,
        has_independent_hh: bool,
        hh_last_tested: Optional[datetime],
        sis_functional: bool,
        test_interval_days: int = 30
    ) -> StorageValidationResult:
        """Validate overfill protection system status."""
        validation_id = hashlib.sha256(
            f"overfill|{tank_id}|{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()[:16]

        violations = []
        actions = []
        now = datetime.now(timezone.utc)

        if self._requires_independent_hh and not has_independent_hh:
            violations.append(StorageViolation(
                constraint_id="OFP_HH", constraint_name="Independent High-High",
                tank_id=tank_id, parameter="has_independent_hh",
                actual_value=0.0, limit_value=1.0, limit_type="min",
                action=SafetyAction.BLOCK,
                message="Tank lacks independent high-high level protection",
                reference_standard="API 2350 Category 3"
            ))
            actions.append(SafetyAction.BLOCK)

        if hh_last_tested is not None:
            days_since_test = (now - hh_last_tested).days
            if days_since_test > test_interval_days:
                violations.append(StorageViolation(
                    constraint_id="OFP_TEST", constraint_name="HH Test Overdue",
                    tank_id=tank_id, parameter="days_since_test",
                    actual_value=float(days_since_test), limit_value=float(test_interval_days),
                    limit_type="max", action=SafetyAction.WARN,
                    message=f"High-high test overdue by {days_since_test - test_interval_days} days",
                    reference_standard="API 2350"
                ))
                actions.append(SafetyAction.WARN)

        if self._requires_sis_tested and not sis_functional:
            violations.append(StorageViolation(
                constraint_id="OFP_SIS", constraint_name="SIS Functional",
                tank_id=tank_id, parameter="sis_functional",
                actual_value=0.0, limit_value=1.0, limit_type="min",
                action=SafetyAction.SHUTDOWN,
                message="Safety Instrumented System not functional",
                reference_standard="IEC 61511"
            ))
            actions.append(SafetyAction.SHUTDOWN)

        is_safe = not any(v.action in [SafetyAction.BLOCK, SafetyAction.SHUTDOWN] for v in violations)

        return StorageValidationResult(
            validation_id=validation_id, tank_id=tank_id,
            is_safe=is_safe, violations=violations, actions_required=actions,
            provenance_hash=hashlib.sha256(
                json.dumps({"id": validation_id, "tank": tank_id, "sis": sis_functional}, sort_keys=True).encode()
            ).hexdigest()
        )


class TemperatureRangeValidator:
    """Temperature range validator for fuel storage."""

    def __init__(self, min_temp_c: float = 10.0, max_temp_c: float = 90.0):
        """Initialize with temperature range."""
        self._min_temp = min_temp_c
        self._max_temp = max_temp_c
        logger.info(f"TemperatureRangeValidator initialized: range={min_temp_c}C to {max_temp_c}C")

    def validate(
        self, current_temp_c: float, tank_id: str, fuel_pour_point_c: Optional[float] = None
    ) -> StorageValidationResult:
        """Validate storage temperature within safe range."""
        validation_id = hashlib.sha256(
            f"temp|{current_temp_c}|{tank_id}|{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()[:16]

        violations = []
        actions = []

        effective_min = max(self._min_temp, (fuel_pour_point_c or -100) + 10)

        if current_temp_c < effective_min:
            violations.append(StorageViolation(
                constraint_id="TEMP_LOW", constraint_name="Low Temperature",
                tank_id=tank_id, parameter="temperature_c",
                actual_value=current_temp_c, limit_value=effective_min,
                limit_type="min", action=SafetyAction.ALARM,
                message=f"Temperature {current_temp_c}C below minimum {effective_min}C",
                reference_standard="ISO 8217 Storage Guidelines"
            ))
            actions.append(SafetyAction.ALARM)

        if current_temp_c > self._max_temp:
            action = SafetyAction.SHUTDOWN if current_temp_c > self._max_temp + 10 else SafetyAction.ALARM
            violations.append(StorageViolation(
                constraint_id="TEMP_HIGH", constraint_name="High Temperature",
                tank_id=tank_id, parameter="temperature_c",
                actual_value=current_temp_c, limit_value=self._max_temp,
                limit_type="max", action=action,
                message=f"Temperature {current_temp_c}C exceeds maximum {self._max_temp}C",
                reference_standard="API 650"
            ))
            actions.append(action)

        is_safe = not any(v.action in [SafetyAction.BLOCK, SafetyAction.SHUTDOWN] for v in violations)

        return StorageValidationResult(
            validation_id=validation_id, tank_id=tank_id,
            is_safe=is_safe, violations=violations, actions_required=actions,
            provenance_hash=hashlib.sha256(
                json.dumps({"id": validation_id, "temp": current_temp_c, "tank": tank_id}, sort_keys=True).encode()
            ).hexdigest()
        )


class StorageConstraintValidator:
    """Unified storage constraint validator combining all storage safety checks."""

    def __init__(self, tank_configs: Optional[Dict[str, TankConfig]] = None):
        """Initialize with tank configurations."""
        self._tank_configs = tank_configs or {}
        self.flash_point = FlashPointValidator()
        self.vapor_pressure = VaporPressureValidator()
        self.min_heel = MinHeelConstraint()
        self.max_fill = MaxFillConstraint()
        self.overfill_policy = OverfillProtectionPolicy()
        self.temperature = TemperatureRangeValidator()
        logger.info(f"StorageConstraintValidator initialized with {len(self._tank_configs)} tank configs")

    def register_tank(self, config: TankConfig) -> None:
        """Register a tank configuration."""
        self._tank_configs[config.tank_id] = config
        logger.info(f"Registered tank: {config.tank_id}")

    def validate_storage_safety(
        self,
        tank_id: str,
        current_level_pct: float,
        current_temp_c: float,
        flash_point_c: Optional[float] = None,
        rvp_psi: Optional[float] = None,
        incoming_volume_m3: float = 0.0,
        is_transfer_active: bool = False
    ) -> StorageValidationResult:
        """
        Validate all storage constraints for a tank.

        FAIL-CLOSED: Any BLOCK or SHUTDOWN action fails entire validation.
        """
        validation_id = hashlib.sha256(
            f"storage|{tank_id}|{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()[:16]

        config = self._tank_configs.get(tank_id)
        capacity = config.capacity_m3 if config else 1000.0

        all_violations: List[StorageViolation] = []
        all_actions: List[SafetyAction] = []

        heel_result = self.min_heel.validate(current_level_pct, tank_id, is_transfer_active)
        all_violations.extend(heel_result.violations)
        all_actions.extend(heel_result.actions_required)

        fill_result = self.max_fill.validate(current_level_pct, tank_id, incoming_volume_m3, capacity)
        all_violations.extend(fill_result.violations)
        all_actions.extend(fill_result.actions_required)

        temp_result = self.temperature.validate(current_temp_c, tank_id)
        all_violations.extend(temp_result.violations)
        all_actions.extend(temp_result.actions_required)

        if flash_point_c is not None:
            fp_result = self.flash_point.validate(flash_point_c, tank_id)
            all_violations.extend(fp_result.violations)
            all_actions.extend(fp_result.actions_required)

        if rvp_psi is not None:
            vp_result = self.vapor_pressure.validate(rvp_psi, tank_id, current_temp_c)
            all_violations.extend(vp_result.violations)
            all_actions.extend(vp_result.actions_required)

        is_safe = not any(a in [SafetyAction.BLOCK, SafetyAction.SHUTDOWN] for a in all_actions)

        return StorageValidationResult(
            validation_id=validation_id, tank_id=tank_id,
            is_safe=is_safe, violations=all_violations, actions_required=list(set(all_actions)),
            provenance_hash=hashlib.sha256(
                json.dumps({
                    "id": validation_id, "tank": tank_id,
                    "violations": len(all_violations), "is_safe": is_safe
                }, sort_keys=True).encode()
            ).hexdigest()
        )
