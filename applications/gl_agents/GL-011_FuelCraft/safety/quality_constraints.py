# -*- coding: utf-8 -*-
"""
QualityConstraints - Fuel quality limits and validation for GL-011 FuelCraft.

This module implements quality constraint validators for fuel properties including
sulfur content, ash content, water content, viscosity, and metals concentrations.
Equipment-specific and vendor-specific limits are supported.

Reference Standards:
    - ISO 8217:2017 Petroleum products - Fuels (class F)
    - ASTM D975 Standard Specification for Diesel Fuel
    - ASTM D396 Standard Specification for Fuel Oils
    - IMO MARPOL Annex VI (Sulfur limits)

Author: GL-BackendDeveloper
Date: 2025-01-01
Version: 1.0.0
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, field_validator
from datetime import datetime, timezone
from enum import Enum
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


class ConstraintSeverity(str, Enum):
    """Severity level for constraint violations."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class ValidationStatus(str, Enum):
    """Validation result status."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"


class ConstraintViolation(BaseModel):
    """Record of a constraint violation."""
    constraint_id: str = Field(..., description="Constraint identifier")
    constraint_name: str = Field(..., description="Constraint name")
    parameter: str = Field(..., description="Parameter that violated constraint")
    actual_value: float = Field(..., description="Actual value")
    limit_value: float = Field(..., description="Limit value")
    limit_type: str = Field(..., description="max, min, or range")
    severity: ConstraintSeverity = Field(...)
    message: str = Field(..., description="Violation message")
    reference_standard: Optional[str] = Field(None)


class QualityValidationResult(BaseModel):
    """Result of quality constraint validation."""
    validation_id: str = Field(...)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: ValidationStatus = Field(...)
    sample_id: Optional[str] = Field(None)
    fuel_type: str = Field(...)
    equipment_id: Optional[str] = Field(None)
    violations: List[ConstraintViolation] = Field(default_factory=list)
    warnings: List[ConstraintViolation] = Field(default_factory=list)
    passed_constraints: int = Field(0)
    failed_constraints: int = Field(0)
    provenance_hash: str = Field(...)


class QualityLimit(BaseModel):
    """Quality limit definition."""
    limit_id: str = Field(...)
    parameter: str = Field(...)
    min_value: Optional[float] = Field(None)
    max_value: Optional[float] = Field(None)
    unit: str = Field(...)
    severity: ConstraintSeverity = Field(ConstraintSeverity.CRITICAL)
    reference_standard: Optional[str] = Field(None)
    applies_to_equipment: Optional[List[str]] = Field(None)
    applies_to_vendors: Optional[List[str]] = Field(None)


class SulfurConstraint:
    """Sulfur content constraint validator per MARPOL/ISO requirements."""

    def __init__(self, limits: Optional[Dict[str, QualityLimit]] = None):
        """Initialize with default MARPOL sulfur limits."""
        self._limits = limits or {
            "ECA": QualityLimit(
                limit_id="S_ECA", parameter="sulfur_pct", max_value=0.10,
                unit="%m/m", severity=ConstraintSeverity.CRITICAL,
                reference_standard="MARPOL Annex VI Reg. 14.4 (ECA)"
            ),
            "GLOBAL": QualityLimit(
                limit_id="S_GLOBAL", parameter="sulfur_pct", max_value=0.50,
                unit="%m/m", severity=ConstraintSeverity.CRITICAL,
                reference_standard="MARPOL Annex VI Reg. 14.1 (Global 2020)"
            ),
            "HSFO": QualityLimit(
                limit_id="S_HSFO", parameter="sulfur_pct", max_value=3.50,
                unit="%m/m", severity=ConstraintSeverity.WARNING,
                reference_standard="ISO 8217:2017 (with scrubber)"
            ),
        }
        logger.info(f"SulfurConstraint initialized with {len(self._limits)} limits")

    def validate(self, sulfur_pct: float, zone: str = "GLOBAL") -> QualityValidationResult:
        """Validate sulfur content against zone-specific limit."""
        limit = self._limits.get(zone)
        if limit is None:
            limit = self._limits["GLOBAL"]

        validation_id = hashlib.sha256(
            f"sulfur|{sulfur_pct}|{zone}|{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()[:16]

        violations = []
        if limit.max_value is not None and sulfur_pct > limit.max_value:
            violations.append(ConstraintViolation(
                constraint_id=limit.limit_id, constraint_name="Sulfur Content",
                parameter="sulfur_pct", actual_value=sulfur_pct,
                limit_value=limit.max_value, limit_type="max",
                severity=limit.severity,
                message=f"Sulfur {sulfur_pct}% exceeds {zone} limit of {limit.max_value}%",
                reference_standard=limit.reference_standard
            ))

        status = ValidationStatus.FAIL if violations else ValidationStatus.PASS
        return QualityValidationResult(
            validation_id=validation_id, status=status, fuel_type="unknown",
            violations=violations if status == ValidationStatus.FAIL else [],
            warnings=violations if status == ValidationStatus.WARNING else [],
            passed_constraints=0 if violations else 1,
            failed_constraints=len(violations),
            provenance_hash=hashlib.sha256(
                json.dumps({"id": validation_id, "sulfur": sulfur_pct, "zone": zone}, sort_keys=True).encode()
            ).hexdigest()
        )


class AshConstraint:
    """Ash content constraint validator."""

    def __init__(self, limits: Optional[Dict[str, QualityLimit]] = None):
        """Initialize with default ash limits per ISO 8217."""
        self._limits = limits or {
            "RMA": QualityLimit(
                limit_id="ASH_RMA", parameter="ash_pct", max_value=0.04,
                unit="%m/m", severity=ConstraintSeverity.CRITICAL,
                reference_standard="ISO 8217:2017 RMA 10"
            ),
            "RMD": QualityLimit(
                limit_id="ASH_RMD", parameter="ash_pct", max_value=0.10,
                unit="%m/m", severity=ConstraintSeverity.CRITICAL,
                reference_standard="ISO 8217:2017 RMD 80"
            ),
            "RMK": QualityLimit(
                limit_id="ASH_RMK", parameter="ash_pct", max_value=0.15,
                unit="%m/m", severity=ConstraintSeverity.CRITICAL,
                reference_standard="ISO 8217:2017 RMK 700"
            ),
        }

    def validate(self, ash_pct: float, fuel_grade: str = "RMD") -> QualityValidationResult:
        """Validate ash content against fuel grade limit."""
        limit = self._limits.get(fuel_grade, self._limits["RMD"])
        validation_id = hashlib.sha256(
            f"ash|{ash_pct}|{fuel_grade}|{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()[:16]

        violations = []
        if limit.max_value is not None and ash_pct > limit.max_value:
            violations.append(ConstraintViolation(
                constraint_id=limit.limit_id, constraint_name="Ash Content",
                parameter="ash_pct", actual_value=ash_pct, limit_value=limit.max_value,
                limit_type="max", severity=limit.severity,
                message=f"Ash {ash_pct}% exceeds limit of {limit.max_value}%",
                reference_standard=limit.reference_standard
            ))

        return QualityValidationResult(
            validation_id=validation_id,
            status=ValidationStatus.FAIL if violations else ValidationStatus.PASS,
            fuel_type=fuel_grade, violations=violations,
            passed_constraints=0 if violations else 1, failed_constraints=len(violations),
            provenance_hash=hashlib.sha256(
                json.dumps({"id": validation_id, "ash": ash_pct}, sort_keys=True).encode()
            ).hexdigest()
        )


class WaterContentConstraint:
    """Water content constraint validator."""

    def __init__(self, limits: Optional[Dict[str, QualityLimit]] = None):
        """Initialize with default water limits per ISO 8217."""
        self._limits = limits or {
            "DISTILLATE": QualityLimit(
                limit_id="WATER_DIST", parameter="water_pct", max_value=0.30,
                unit="%v/v", severity=ConstraintSeverity.CRITICAL,
                reference_standard="ISO 8217:2017 DMA/DMB"
            ),
            "RESIDUAL": QualityLimit(
                limit_id="WATER_RES", parameter="water_pct", max_value=0.50,
                unit="%v/v", severity=ConstraintSeverity.CRITICAL,
                reference_standard="ISO 8217:2017 RMx"
            ),
        }

    def validate(self, water_pct: float, fuel_category: str = "RESIDUAL") -> QualityValidationResult:
        """Validate water content against fuel category limit."""
        limit = self._limits.get(fuel_category, self._limits["RESIDUAL"])
        validation_id = hashlib.sha256(
            f"water|{water_pct}|{fuel_category}|{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()[:16]

        violations = []
        if limit.max_value is not None and water_pct > limit.max_value:
            violations.append(ConstraintViolation(
                constraint_id=limit.limit_id, constraint_name="Water Content",
                parameter="water_pct", actual_value=water_pct, limit_value=limit.max_value,
                limit_type="max", severity=limit.severity,
                message=f"Water {water_pct}% exceeds limit of {limit.max_value}%",
                reference_standard=limit.reference_standard
            ))

        return QualityValidationResult(
            validation_id=validation_id,
            status=ValidationStatus.FAIL if violations else ValidationStatus.PASS,
            fuel_type=fuel_category, violations=violations,
            passed_constraints=0 if violations else 1, failed_constraints=len(violations),
            provenance_hash=hashlib.sha256(
                json.dumps({"id": validation_id, "water": water_pct}, sort_keys=True).encode()
            ).hexdigest()
        )


class ViscosityBandConstraint:
    """Viscosity band constraint validator for equipment compatibility."""

    def __init__(self, limits: Optional[Dict[str, QualityLimit]] = None):
        """Initialize with default viscosity limits."""
        self._limits = limits or {
            "BOILER_STANDARD": QualityLimit(
                limit_id="VISC_BOILER", parameter="viscosity_cst",
                min_value=10.0, max_value=380.0, unit="cSt@50C",
                severity=ConstraintSeverity.CRITICAL,
                reference_standard="Equipment manufacturer spec"
            ),
            "BOILER_WIDE_RANGE": QualityLimit(
                limit_id="VISC_BOILER_WR", parameter="viscosity_cst",
                min_value=5.0, max_value=700.0, unit="cSt@50C",
                severity=ConstraintSeverity.WARNING,
                reference_standard="ISO 8217:2017 RMK 700"
            ),
            "DISTILLATE": QualityLimit(
                limit_id="VISC_DIST", parameter="viscosity_cst",
                min_value=2.0, max_value=11.0, unit="cSt@40C",
                severity=ConstraintSeverity.CRITICAL,
                reference_standard="ISO 8217:2017 DMB"
            ),
        }

    def validate(
        self, viscosity_cst: float, equipment_type: str = "BOILER_STANDARD"
    ) -> QualityValidationResult:
        """Validate viscosity against equipment-specific band."""
        limit = self._limits.get(equipment_type, self._limits["BOILER_STANDARD"])
        validation_id = hashlib.sha256(
            f"visc|{viscosity_cst}|{equipment_type}|{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()[:16]

        violations = []
        if limit.min_value is not None and viscosity_cst < limit.min_value:
            violations.append(ConstraintViolation(
                constraint_id=limit.limit_id, constraint_name="Viscosity (Low)",
                parameter="viscosity_cst", actual_value=viscosity_cst,
                limit_value=limit.min_value, limit_type="min", severity=limit.severity,
                message=f"Viscosity {viscosity_cst} cSt below minimum {limit.min_value} cSt",
                reference_standard=limit.reference_standard
            ))
        if limit.max_value is not None and viscosity_cst > limit.max_value:
            violations.append(ConstraintViolation(
                constraint_id=limit.limit_id, constraint_name="Viscosity (High)",
                parameter="viscosity_cst", actual_value=viscosity_cst,
                limit_value=limit.max_value, limit_type="max", severity=limit.severity,
                message=f"Viscosity {viscosity_cst} cSt exceeds maximum {limit.max_value} cSt",
                reference_standard=limit.reference_standard
            ))

        return QualityValidationResult(
            validation_id=validation_id,
            status=ValidationStatus.FAIL if violations else ValidationStatus.PASS,
            fuel_type=equipment_type, violations=violations,
            passed_constraints=0 if violations else 1, failed_constraints=len(violations),
            provenance_hash=hashlib.sha256(
                json.dumps({"id": validation_id, "viscosity": viscosity_cst}, sort_keys=True).encode()
            ).hexdigest()
        )


class MetalsConstraint:
    """Metals content constraint validator (vanadium, sodium, aluminum+silicon)."""

    def __init__(self, limits: Optional[Dict[str, Dict[str, QualityLimit]]] = None):
        """Initialize with default metals limits per ISO 8217."""
        self._limits = limits or {
            "vanadium_ppm": {
                "RMA": QualityLimit(
                    limit_id="V_RMA", parameter="vanadium_ppm", max_value=50.0,
                    unit="mg/kg", severity=ConstraintSeverity.CRITICAL,
                    reference_standard="ISO 8217:2017 RMA"
                ),
                "RMK": QualityLimit(
                    limit_id="V_RMK", parameter="vanadium_ppm", max_value=450.0,
                    unit="mg/kg", severity=ConstraintSeverity.CRITICAL,
                    reference_standard="ISO 8217:2017 RMK"
                ),
            },
            "sodium_ppm": {
                "DEFAULT": QualityLimit(
                    limit_id="NA_DEFAULT", parameter="sodium_ppm", max_value=100.0,
                    unit="mg/kg", severity=ConstraintSeverity.CRITICAL,
                    reference_standard="ISO 8217:2017"
                ),
            },
            "aluminum_silicon_ppm": {
                "RMA": QualityLimit(
                    limit_id="ALSI_RMA", parameter="al_si_ppm", max_value=25.0,
                    unit="mg/kg", severity=ConstraintSeverity.CRITICAL,
                    reference_standard="ISO 8217:2017 RMA"
                ),
                "RMK": QualityLimit(
                    limit_id="ALSI_RMK", parameter="al_si_ppm", max_value=60.0,
                    unit="mg/kg", severity=ConstraintSeverity.CRITICAL,
                    reference_standard="ISO 8217:2017 RMK"
                ),
            },
        }

    def validate(
        self,
        vanadium_ppm: Optional[float] = None,
        sodium_ppm: Optional[float] = None,
        aluminum_silicon_ppm: Optional[float] = None,
        fuel_grade: str = "RMA"
    ) -> QualityValidationResult:
        """Validate metals content against fuel grade limits."""
        validation_id = hashlib.sha256(
            f"metals|{vanadium_ppm}|{sodium_ppm}|{aluminum_silicon_ppm}|{fuel_grade}|{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()[:16]

        violations = []

        if vanadium_ppm is not None:
            v_limits = self._limits.get("vanadium_ppm", {})
            v_limit = v_limits.get(fuel_grade, v_limits.get("RMA"))
            if v_limit and v_limit.max_value is not None and vanadium_ppm > v_limit.max_value:
                violations.append(ConstraintViolation(
                    constraint_id=v_limit.limit_id, constraint_name="Vanadium",
                    parameter="vanadium_ppm", actual_value=vanadium_ppm,
                    limit_value=v_limit.max_value, limit_type="max", severity=v_limit.severity,
                    message=f"Vanadium {vanadium_ppm} ppm exceeds limit of {v_limit.max_value} ppm",
                    reference_standard=v_limit.reference_standard
                ))

        if sodium_ppm is not None:
            na_limits = self._limits.get("sodium_ppm", {})
            na_limit = na_limits.get(fuel_grade, na_limits.get("DEFAULT"))
            if na_limit and na_limit.max_value is not None and sodium_ppm > na_limit.max_value:
                violations.append(ConstraintViolation(
                    constraint_id=na_limit.limit_id, constraint_name="Sodium",
                    parameter="sodium_ppm", actual_value=sodium_ppm,
                    limit_value=na_limit.max_value, limit_type="max", severity=na_limit.severity,
                    message=f"Sodium {sodium_ppm} ppm exceeds limit of {na_limit.max_value} ppm",
                    reference_standard=na_limit.reference_standard
                ))

        if aluminum_silicon_ppm is not None:
            alsi_limits = self._limits.get("aluminum_silicon_ppm", {})
            alsi_limit = alsi_limits.get(fuel_grade, alsi_limits.get("RMA"))
            if alsi_limit and alsi_limit.max_value is not None and aluminum_silicon_ppm > alsi_limit.max_value:
                violations.append(ConstraintViolation(
                    constraint_id=alsi_limit.limit_id, constraint_name="Aluminum+Silicon",
                    parameter="al_si_ppm", actual_value=aluminum_silicon_ppm,
                    limit_value=alsi_limit.max_value, limit_type="max", severity=alsi_limit.severity,
                    message=f"Al+Si {aluminum_silicon_ppm} ppm exceeds limit of {alsi_limit.max_value} ppm",
                    reference_standard=alsi_limit.reference_standard
                ))

        tested = sum(1 for x in [vanadium_ppm, sodium_ppm, aluminum_silicon_ppm] if x is not None)
        return QualityValidationResult(
            validation_id=validation_id,
            status=ValidationStatus.FAIL if violations else ValidationStatus.PASS,
            fuel_type=fuel_grade, violations=violations,
            passed_constraints=tested - len(violations), failed_constraints=len(violations),
            provenance_hash=hashlib.sha256(
                json.dumps({"id": validation_id, "v": vanadium_ppm, "na": sodium_ppm, "alsi": aluminum_silicon_ppm}, sort_keys=True).encode()
            ).hexdigest()
        )


class EquipmentQualityLimits:
    """Equipment-specific quality limit lookup."""

    def __init__(self):
        """Initialize with default equipment limits."""
        self._equipment_limits: Dict[str, Dict[str, QualityLimit]] = {
            "BOILER_001": {
                "sulfur": QualityLimit(
                    limit_id="B001_S", parameter="sulfur_pct", max_value=0.5,
                    unit="%", severity=ConstraintSeverity.CRITICAL,
                    applies_to_equipment=["BOILER_001"]
                ),
                "viscosity": QualityLimit(
                    limit_id="B001_V", parameter="viscosity_cst",
                    min_value=20.0, max_value=380.0, unit="cSt@50C",
                    severity=ConstraintSeverity.CRITICAL,
                    applies_to_equipment=["BOILER_001"]
                ),
            },
            "ENGINE_001": {
                "sulfur": QualityLimit(
                    limit_id="E001_S", parameter="sulfur_pct", max_value=0.1,
                    unit="%", severity=ConstraintSeverity.CRITICAL,
                    applies_to_equipment=["ENGINE_001"]
                ),
                "viscosity": QualityLimit(
                    limit_id="E001_V", parameter="viscosity_cst",
                    min_value=2.0, max_value=14.0, unit="cSt@40C",
                    severity=ConstraintSeverity.CRITICAL,
                    applies_to_equipment=["ENGINE_001"]
                ),
            },
        }

    def get_limits(self, equipment_id: str) -> Dict[str, QualityLimit]:
        """Get quality limits for specific equipment."""
        return self._equipment_limits.get(equipment_id, {})

    def register_equipment(self, equipment_id: str, limits: Dict[str, QualityLimit]) -> None:
        """Register equipment-specific limits."""
        self._equipment_limits[equipment_id] = limits
        logger.info(f"Registered limits for equipment {equipment_id}")


class QualityConstraintValidator:
    """Unified quality constraint validator combining all constraint types."""

    def __init__(self):
        """Initialize all constraint validators."""
        self.sulfur = SulfurConstraint()
        self.ash = AshConstraint()
        self.water = WaterContentConstraint()
        self.viscosity = ViscosityBandConstraint()
        self.metals = MetalsConstraint()
        self.equipment_limits = EquipmentQualityLimits()
        logger.info("QualityConstraintValidator initialized with all constraint types")

    def validate_fuel_quality(
        self,
        sulfur_pct: Optional[float] = None,
        ash_pct: Optional[float] = None,
        water_pct: Optional[float] = None,
        viscosity_cst: Optional[float] = None,
        vanadium_ppm: Optional[float] = None,
        sodium_ppm: Optional[float] = None,
        aluminum_silicon_ppm: Optional[float] = None,
        fuel_grade: str = "RMD",
        sulfur_zone: str = "GLOBAL",
        equipment_id: Optional[str] = None
    ) -> QualityValidationResult:
        """
        Validate all fuel quality parameters.

        FAIL-CLOSED: Any critical violation fails entire validation.
        """
        validation_id = hashlib.sha256(
            f"quality|{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()[:16]

        all_violations: List[ConstraintViolation] = []
        all_warnings: List[ConstraintViolation] = []
        passed = 0
        failed = 0

        if sulfur_pct is not None:
            result = self.sulfur.validate(sulfur_pct, sulfur_zone)
            if result.violations:
                for v in result.violations:
                    if v.severity == ConstraintSeverity.CRITICAL:
                        all_violations.append(v)
                    else:
                        all_warnings.append(v)
                failed += 1
            else:
                passed += 1

        if ash_pct is not None:
            result = self.ash.validate(ash_pct, fuel_grade)
            if result.violations:
                all_violations.extend(result.violations)
                failed += 1
            else:
                passed += 1

        if water_pct is not None:
            fuel_cat = "DISTILLATE" if fuel_grade.startswith("DM") else "RESIDUAL"
            result = self.water.validate(water_pct, fuel_cat)
            if result.violations:
                all_violations.extend(result.violations)
                failed += 1
            else:
                passed += 1

        if viscosity_cst is not None:
            equip_type = "DISTILLATE" if fuel_grade.startswith("DM") else "BOILER_STANDARD"
            result = self.viscosity.validate(viscosity_cst, equip_type)
            if result.violations:
                all_violations.extend(result.violations)
                failed += 1
            else:
                passed += 1

        if any([vanadium_ppm, sodium_ppm, aluminum_silicon_ppm]):
            result = self.metals.validate(
                vanadium_ppm, sodium_ppm, aluminum_silicon_ppm, fuel_grade
            )
            if result.violations:
                all_violations.extend(result.violations)
                failed += result.failed_constraints
            passed += result.passed_constraints

        status = ValidationStatus.FAIL if all_violations else (
            ValidationStatus.WARNING if all_warnings else ValidationStatus.PASS
        )

        return QualityValidationResult(
            validation_id=validation_id, status=status, fuel_type=fuel_grade,
            equipment_id=equipment_id, violations=all_violations, warnings=all_warnings,
            passed_constraints=passed, failed_constraints=failed,
            provenance_hash=hashlib.sha256(
                json.dumps({
                    "id": validation_id, "violations": len(all_violations),
                    "passed": passed, "failed": failed
                }, sort_keys=True).encode()
            ).hexdigest()
        )
