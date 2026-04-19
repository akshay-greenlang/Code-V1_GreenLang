# -*- coding: utf-8 -*-
"""
GL-015 Insulscan - Constraint Validator

Production-grade physical constraint validation for insulation assessment
and thermal measurement systems. Enforces:

1. Temperature Range Validation:
   - Operating temperature must be > ambient
   - Surface temperature must be < operating temperature
   - Temperature differentials within physical bounds

2. Insulation Thickness Sanity Checks:
   - Thickness > 0 and < max design thickness
   - Thickness appropriate for temperature class

3. Surface Area Bounds:
   - Non-negative surface area
   - Plausible for asset type

4. Heat Loss Plausibility Checks:
   - Heat loss > 0 for hot equipment
   - Heat loss within theoretical maximum
   - Efficiency calculations consistent

Safety Principles:
- Fail-closed on any violation
- All checks use deterministic arithmetic
- SHA-256 provenance for all validations
- No LLM inference for safety decisions

Standards Compliance:
- ASTM C680: Heat Loss Calculations for Insulation
- CINI Manual: Insulation Inspection Guidelines
- ISO 12241: Thermal Insulation for Building Equipment
- BS 5422: Thermal Insulating Materials

Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from .exceptions import (
    InsulscanSafetyError,
    ConstraintViolationError,
    ThermalMeasurementError,
    InsulationThicknessError,
    HeatLossImplausibleError,
    SurfaceBelowAmbientError,
    ViolationDetails,
    ViolationSeverity,
    ViolationContext,
    SafetyDomain,
)

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================


class ConstraintType(str, Enum):
    """Types of physical constraints validated."""

    TEMPERATURE_RANGE = "temperature_range"
    TEMPERATURE_DIFFERENTIAL = "temperature_differential"
    SURFACE_VS_AMBIENT = "surface_vs_ambient"
    SURFACE_VS_OPERATING = "surface_vs_operating"
    INSULATION_THICKNESS = "insulation_thickness"
    SURFACE_AREA = "surface_area"
    HEAT_LOSS_PLAUSIBILITY = "heat_loss_plausibility"
    HEAT_LOSS_MAXIMUM = "heat_loss_maximum"
    EFFICIENCY_BOUNDS = "efficiency_bounds"
    CONDITION_SCORE = "condition_score"


class ConstraintSeverity(str, Enum):
    """Severity level for constraint violations."""

    INFO = "info"  # Informational, no action required
    WARNING = "warning"  # Near limit, log and continue
    ERROR = "error"  # Violation, reject with option to override
    CRITICAL = "critical"  # Hard violation, no override allowed


class InsulationType(str, Enum):
    """Classification of insulation type."""

    MINERAL_WOOL = "mineral_wool"
    CALCIUM_SILICATE = "calcium_silicate"
    FOAM_GLASS = "foam_glass"
    PERLITE = "perlite"
    AEROGEL = "aerogel"
    CERAMIC_FIBER = "ceramic_fiber"
    POLYURETHANE = "polyurethane"
    UNKNOWN = "unknown"


class AssetType(str, Enum):
    """Types of insulated assets."""

    PIPE = "pipe"
    VESSEL = "vessel"
    TANK = "tank"
    VALVE = "valve"
    EQUIPMENT = "equipment"
    DUCT = "duct"
    BOILER = "boiler"
    TURBINE = "turbine"


# =============================================================================
# CONFIGURATION
# =============================================================================


class InsulationConstraintLimits(BaseModel):
    """
    Physical constraint limits configuration for insulation.

    Default values based on industry standards and best practices.
    All values can be customized for specific applications.
    """

    # Temperature bounds (Celsius)
    ambient_temp_min_C: float = Field(
        default=-40.0,
        ge=-60.0,
        le=50.0,
        description="Minimum ambient temperature"
    )
    ambient_temp_max_C: float = Field(
        default=50.0,
        ge=0.0,
        le=80.0,
        description="Maximum ambient temperature"
    )
    operating_temp_min_C: float = Field(
        default=-200.0,
        ge=-273.15,
        le=100.0,
        description="Minimum operating temperature (cryogenic)"
    )
    operating_temp_max_C: float = Field(
        default=800.0,
        ge=100.0,
        le=1200.0,
        description="Maximum operating temperature"
    )
    min_temp_differential_C: float = Field(
        default=5.0,
        ge=1.0,
        le=50.0,
        description="Minimum meaningful temperature differential"
    )

    # Insulation thickness bounds (mm)
    thickness_min_mm: float = Field(
        default=10.0,
        ge=1.0,
        le=50.0,
        description="Minimum insulation thickness"
    )
    thickness_max_mm: float = Field(
        default=500.0,
        ge=100.0,
        le=1000.0,
        description="Maximum insulation thickness"
    )

    # Surface area bounds (m2)
    surface_area_min_m2: float = Field(
        default=0.001,
        ge=0.0001,
        le=1.0,
        description="Minimum surface area"
    )
    surface_area_max_m2: float = Field(
        default=100000.0,
        ge=1000.0,
        le=1000000.0,
        description="Maximum surface area"
    )

    # Heat loss bounds (W/m2)
    heat_loss_max_W_m2: float = Field(
        default=5000.0,
        ge=500.0,
        le=20000.0,
        description="Maximum plausible heat loss"
    )
    heat_loss_min_positive_W_m2: float = Field(
        default=1.0,
        ge=0.1,
        le=50.0,
        description="Minimum detectable heat loss"
    )

    # Efficiency bounds
    efficiency_min: float = Field(
        default=0.0,
        ge=0.0,
        le=0.5,
        description="Minimum efficiency (0% = no insulation)"
    )
    efficiency_max: float = Field(
        default=1.0,
        ge=0.5,
        le=1.0,
        description="Maximum efficiency (100% = perfect insulation)"
    )

    # Condition score bounds
    condition_score_min: float = Field(
        default=0.0,
        ge=0.0,
        le=0.5,
        description="Minimum condition score (failed)"
    )
    condition_score_max: float = Field(
        default=1.0,
        ge=0.5,
        le=1.0,
        description="Maximum condition score (excellent)"
    )


class InsulationConstraintValidatorConfig(BaseModel):
    """
    Configuration for the insulation constraint validator.

    Attributes:
        limits: Physical constraint limits
        fail_closed: If True, raise exception on violations
        log_warnings: If True, log warning-level issues
        strict_temperature_checks: If True, treat temperature issues as errors
    """

    limits: InsulationConstraintLimits = Field(
        default_factory=InsulationConstraintLimits,
        description="Physical constraint limits"
    )
    fail_closed: bool = Field(
        default=True,
        description="Raise exception on critical violations"
    )
    log_warnings: bool = Field(
        default=True,
        description="Log warning-level issues"
    )
    strict_temperature_checks: bool = Field(
        default=True,
        description="Treat temperature inconsistencies as errors"
    )


# =============================================================================
# DATA MODELS
# =============================================================================


@dataclass(frozen=True)
class ConstraintCheckResult:
    """
    Result of a single constraint check.

    Immutable for audit trail integrity.

    Attributes:
        constraint_type: Type of constraint checked
        asset_id: Insulation asset identifier
        parameter_name: Name of the checked parameter
        actual_value: Measured/calculated value
        limit_value: Constraint limit
        unit: Engineering unit
        is_valid: Whether constraint is satisfied
        severity: Violation severity
        margin: Distance from limit (positive = within limit)
        margin_percent: Margin as percentage of limit
        message: Human-readable description
        standard_reference: Applicable standard
        recommended_action: Suggested remediation
        provenance_hash: SHA-256 hash for audit trail
    """

    constraint_type: ConstraintType
    asset_id: str
    parameter_name: str
    actual_value: float
    limit_value: float
    unit: str
    is_valid: bool
    severity: ConstraintSeverity
    margin: float
    margin_percent: float
    message: str
    standard_reference: str
    recommended_action: str
    provenance_hash: str = ""

    def __post_init__(self) -> None:
        """Calculate provenance hash."""
        if not self.provenance_hash:
            content = (
                f"{self.constraint_type.value}|{self.asset_id}|"
                f"{self.parameter_name}|{self.actual_value:.8f}|"
                f"{self.limit_value:.8f}|{self.is_valid}"
            )
            object.__setattr__(
                self,
                'provenance_hash',
                hashlib.sha256(content.encode()).hexdigest()
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "constraint_type": self.constraint_type.value,
            "asset_id": self.asset_id,
            "parameter_name": self.parameter_name,
            "actual_value": self.actual_value,
            "limit_value": self.limit_value,
            "unit": self.unit,
            "is_valid": self.is_valid,
            "severity": self.severity.value,
            "margin": self.margin,
            "margin_percent": self.margin_percent,
            "message": self.message,
            "standard_reference": self.standard_reference,
            "recommended_action": self.recommended_action,
            "provenance_hash": self.provenance_hash,
        }


class InsulationConstraintValidationSummary(BaseModel):
    """
    Summary of all constraint validation results.

    Attributes:
        asset_id: Insulation asset identifier
        timestamp: Validation timestamp
        is_valid: Overall validation status
        total_checks: Number of checks performed
        violations_count: Number of violations
        warnings_count: Number of warnings
        critical_violations: List of critical violations
        errors: List of error-level violations
        warnings: List of warnings
        all_results: All check results
        config_hash: Hash of configuration used
        data_hash: Hash of input data
        result_hash: Hash of this result
    """

    asset_id: str = Field(..., description="Insulation asset identifier")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Validation timestamp"
    )
    is_valid: bool = Field(..., description="Overall validation passed")
    total_checks: int = Field(default=0, ge=0)
    violations_count: int = Field(default=0, ge=0)
    warnings_count: int = Field(default=0, ge=0)

    critical_violations: List[Dict[str, Any]] = Field(default_factory=list)
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[Dict[str, Any]] = Field(default_factory=list)
    all_results: List[Dict[str, Any]] = Field(default_factory=list)

    config_hash: str = Field(default="")
    data_hash: str = Field(default="")
    result_hash: str = Field(default="")

    def get_rejection_reasons(self) -> List[str]:
        """Get list of rejection reasons."""
        reasons = []
        for v in self.critical_violations + self.errors:
            reasons.append(
                f"{v['constraint_type']}: {v['parameter_name']} = "
                f"{v['actual_value']:.4f} {v['unit']} "
                f"(limit: {v['limit_value']:.4f})"
            )
        return reasons

    def model_post_init(self, __context: Any) -> None:
        """Calculate result hash after initialization."""
        if not self.result_hash:
            content = (
                f"{self.asset_id}|{self.timestamp.isoformat()}|"
                f"{self.is_valid}|{self.violations_count}|{self.warnings_count}"
            )
            self.result_hash = hashlib.sha256(content.encode()).hexdigest()


@dataclass
class InsulationData:
    """
    Input data for insulation constraint validation.

    Attributes:
        asset_id: Insulation asset identifier
        operating_temp_C: Process/operating temperature
        ambient_temp_C: Ambient temperature
        surface_temp_C: Measured surface temperature
        insulation_thickness_mm: Insulation thickness
        surface_area_m2: Surface area
        heat_loss_W_m2: Calculated/measured heat loss
        efficiency: Insulation efficiency [0, 1]
        condition_score: Condition assessment score [0, 1]
        insulation_type: Type of insulation material
        asset_type: Type of asset
        location: Physical location
    """

    asset_id: str
    operating_temp_C: float
    ambient_temp_C: float
    surface_temp_C: Optional[float] = None
    insulation_thickness_mm: Optional[float] = None
    surface_area_m2: Optional[float] = None
    heat_loss_W_m2: Optional[float] = None
    efficiency: Optional[float] = None
    condition_score: Optional[float] = None
    insulation_type: InsulationType = InsulationType.UNKNOWN
    asset_type: AssetType = AssetType.PIPE
    location: str = "unknown"


# =============================================================================
# CONSTRAINT VALIDATOR
# =============================================================================


class InsulationConstraintValidator:
    """
    Production-grade physical constraint validator for insulation assessments.

    Validates all physical bounds, temperature relationships, and operational
    constraints with deterministic checks and full provenance tracking.

    Safety Principles:
    - All checks use deterministic arithmetic (no LLM)
    - Fail-closed on critical violations
    - SHA-256 provenance for audit trail
    - Standards-compliant validation

    Example:
        >>> config = InsulationConstraintValidatorConfig()
        >>> validator = InsulationConstraintValidator(config)
        >>>
        >>> data = InsulationData(
        ...     asset_id="INS-PIPE-101",
        ...     operating_temp_C=150.0,
        ...     ambient_temp_C=25.0,
        ...     surface_temp_C=45.0,
        ...     insulation_thickness_mm=50.0,
        ... )
        >>>
        >>> summary = validator.validate(data)
        >>> if not summary.is_valid:
        ...     raise ConstraintViolationError(summary.get_rejection_reasons())

    Author: GL-BackendDeveloper
    Version: 1.0.0
    """

    VERSION = "1.0.0"

    def __init__(self, config: Optional[InsulationConstraintValidatorConfig] = None) -> None:
        """
        Initialize constraint validator.

        Args:
            config: Validator configuration
        """
        self.config = config or InsulationConstraintValidatorConfig()
        self._config_hash = self._compute_config_hash()

        logger.info(
            f"InsulationConstraintValidator initialized: "
            f"fail_closed={self.config.fail_closed}, "
            f"strict_temperature_checks={self.config.strict_temperature_checks}"
        )

    def _compute_config_hash(self) -> str:
        """Compute SHA-256 hash of configuration."""
        config_dict = self.config.model_dump()
        json_str = json.dumps(config_dict, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def _compute_data_hash(self, data: InsulationData) -> str:
        """Compute SHA-256 hash of input data."""
        data_dict = {
            "asset_id": data.asset_id,
            "operating_temp_C": data.operating_temp_C,
            "ambient_temp_C": data.ambient_temp_C,
            "surface_temp_C": data.surface_temp_C,
            "insulation_thickness_mm": data.insulation_thickness_mm,
            "surface_area_m2": data.surface_area_m2,
        }
        json_str = json.dumps(data_dict, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()

    # =========================================================================
    # MAIN VALIDATION METHOD
    # =========================================================================

    def validate(self, data: InsulationData) -> InsulationConstraintValidationSummary:
        """
        Validate all physical constraints for insulation data.

        Performs comprehensive validation:
        1. Temperature range checks
        2. Surface vs ambient temperature
        3. Surface vs operating temperature
        4. Insulation thickness bounds
        5. Surface area bounds
        6. Heat loss plausibility
        7. Efficiency bounds
        8. Condition score bounds

        Args:
            data: Insulation data to validate

        Returns:
            InsulationConstraintValidationSummary with all results

        Raises:
            ConstraintViolationError: If fail_closed and critical violation
        """
        start_time = datetime.now(timezone.utc)
        results: List[ConstraintCheckResult] = []

        logger.debug(f"Validating constraints for {data.asset_id}")

        # 1. Temperature range checks
        results.extend(self._check_temperature_range(data))

        # 2. Surface vs ambient temperature
        if data.surface_temp_C is not None:
            results.append(self._check_surface_vs_ambient(data))

        # 3. Surface vs operating temperature
        if data.surface_temp_C is not None:
            results.append(self._check_surface_vs_operating(data))

        # 4. Insulation thickness bounds
        if data.insulation_thickness_mm is not None:
            results.append(self._check_insulation_thickness(data))

        # 5. Surface area bounds
        if data.surface_area_m2 is not None:
            results.append(self._check_surface_area(data))

        # 6. Heat loss plausibility
        if data.heat_loss_W_m2 is not None:
            results.extend(self._check_heat_loss_plausibility(data))

        # 7. Efficiency bounds
        if data.efficiency is not None:
            results.append(self._check_efficiency_bounds(data))

        # 8. Condition score bounds
        if data.condition_score is not None:
            results.append(self._check_condition_score_bounds(data))

        # Build summary
        summary = self._build_summary(data, results)

        # Log results
        if summary.violations_count > 0:
            logger.warning(
                f"Constraint validation for {data.asset_id}: "
                f"{summary.violations_count} violations, "
                f"{summary.warnings_count} warnings"
            )
        else:
            logger.debug(
                f"Constraint validation for {data.asset_id}: "
                f"PASSED ({summary.total_checks} checks)"
            )

        # Fail-closed behavior
        if self.config.fail_closed and not summary.is_valid:
            self._raise_violation_error(data, summary)

        return summary

    def validate_batch(
        self,
        assets: List[InsulationData],
    ) -> Dict[str, InsulationConstraintValidationSummary]:
        """
        Validate multiple insulation assets.

        Args:
            assets: List of insulation data to validate

        Returns:
            Dict mapping asset_id to validation summary
        """
        results = {}
        for data in assets:
            try:
                results[data.asset_id] = self.validate(data)
            except InsulscanSafetyError as e:
                # Create failed summary
                results[data.asset_id] = InsulationConstraintValidationSummary(
                    asset_id=data.asset_id,
                    is_valid=False,
                    violations_count=len(e.violations),
                    critical_violations=[v.to_dict() for v in e.violations],
                )
        return results

    # =========================================================================
    # INDIVIDUAL CONSTRAINT CHECKS
    # =========================================================================

    def _check_temperature_range(
        self,
        data: InsulationData,
    ) -> List[ConstraintCheckResult]:
        """Check all temperatures are in valid range."""
        results = []
        limits = self.config.limits

        temps = [
            ("operating_temp", data.operating_temp_C, limits.operating_temp_min_C, limits.operating_temp_max_C),
            ("ambient_temp", data.ambient_temp_C, limits.ambient_temp_min_C, limits.ambient_temp_max_C),
        ]

        if data.surface_temp_C is not None:
            temps.append(("surface_temp", data.surface_temp_C, limits.ambient_temp_min_C, limits.operating_temp_max_C))

        for name, temp, min_temp, max_temp in temps:
            is_valid = min_temp <= temp <= max_temp
            margin = min(temp - min_temp, max_temp - temp)
            margin_percent = (margin / (max_temp - min_temp)) * 100 if (max_temp - min_temp) > 0 else 0

            if temp < min_temp:
                severity = ConstraintSeverity.ERROR
                message = f"{name} {temp:.1f}C below minimum {min_temp:.1f}C"
                action = "Check temperature sensor calibration"
            elif temp > max_temp:
                severity = ConstraintSeverity.ERROR
                message = f"{name} {temp:.1f}C exceeds maximum {max_temp:.1f}C"
                action = "Verify equipment and insulation rating"
            else:
                severity = ConstraintSeverity.INFO
                message = f"{name} {temp:.1f}C is valid"
                action = "No action required"

            results.append(ConstraintCheckResult(
                constraint_type=ConstraintType.TEMPERATURE_RANGE,
                asset_id=data.asset_id,
                parameter_name=name,
                actual_value=temp,
                limit_value=max_temp,
                unit="C",
                is_valid=is_valid,
                severity=severity,
                margin=margin,
                margin_percent=margin_percent,
                message=message,
                standard_reference="ISO 12241, ASTM C680",
                recommended_action=action,
            ))

        return results

    def _check_surface_vs_ambient(self, data: InsulationData) -> ConstraintCheckResult:
        """Check surface temperature is above ambient for hot equipment."""
        limits = self.config.limits

        surface_temp = data.surface_temp_C
        ambient_temp = data.ambient_temp_C
        operating_temp = data.operating_temp_C

        # For hot equipment, surface must be > ambient
        is_hot_equipment = operating_temp > ambient_temp

        if is_hot_equipment:
            is_valid = surface_temp >= ambient_temp
            margin = surface_temp - ambient_temp
        else:
            # Cold equipment - surface must be < ambient
            is_valid = surface_temp <= ambient_temp
            margin = ambient_temp - surface_temp

        margin_percent = (margin / abs(operating_temp - ambient_temp)) * 100 if abs(operating_temp - ambient_temp) > 0 else 0

        if not is_valid:
            severity = ConstraintSeverity.CRITICAL
            message = (
                f"Surface temp {surface_temp:.1f}C vs ambient {ambient_temp:.1f}C "
                f"is physically impossible for {'hot' if is_hot_equipment else 'cold'} equipment"
            )
            action = "Check IR scanner calibration and measurement location"
        else:
            severity = ConstraintSeverity.INFO
            message = f"Surface temp {surface_temp:.1f}C is valid relative to ambient {ambient_temp:.1f}C"
            action = "No action required"

        return ConstraintCheckResult(
            constraint_type=ConstraintType.SURFACE_VS_AMBIENT,
            asset_id=data.asset_id,
            parameter_name="surface_vs_ambient",
            actual_value=surface_temp,
            limit_value=ambient_temp,
            unit="C",
            is_valid=is_valid,
            severity=severity,
            margin=margin,
            margin_percent=margin_percent,
            message=message,
            standard_reference="ASTM C680, Thermodynamics",
            recommended_action=action,
        )

    def _check_surface_vs_operating(self, data: InsulationData) -> ConstraintCheckResult:
        """Check surface temperature is between ambient and operating."""
        limits = self.config.limits

        surface_temp = data.surface_temp_C
        ambient_temp = data.ambient_temp_C
        operating_temp = data.operating_temp_C

        is_hot_equipment = operating_temp > ambient_temp

        if is_hot_equipment:
            # Surface must be between ambient and operating
            is_valid = ambient_temp <= surface_temp <= operating_temp
            if surface_temp > operating_temp:
                margin = surface_temp - operating_temp
            elif surface_temp < ambient_temp:
                margin = ambient_temp - surface_temp
            else:
                margin = min(surface_temp - ambient_temp, operating_temp - surface_temp)
        else:
            # Cold equipment - surface must be between operating and ambient
            is_valid = operating_temp <= surface_temp <= ambient_temp
            margin = min(abs(surface_temp - operating_temp), abs(ambient_temp - surface_temp))

        margin_percent = (margin / abs(operating_temp - ambient_temp)) * 100 if abs(operating_temp - ambient_temp) > 0 else 0

        if not is_valid:
            if is_hot_equipment and surface_temp > operating_temp:
                severity = ConstraintSeverity.CRITICAL
                message = f"Surface temp {surface_temp:.1f}C exceeds operating temp {operating_temp:.1f}C - impossible"
            else:
                severity = ConstraintSeverity.ERROR
                message = f"Surface temp {surface_temp:.1f}C outside valid range"
            action = "Verify all temperature measurements"
        else:
            severity = ConstraintSeverity.INFO
            message = f"Surface temp {surface_temp:.1f}C is valid (between ambient {ambient_temp:.1f}C and operating {operating_temp:.1f}C)"
            action = "No action required"

        return ConstraintCheckResult(
            constraint_type=ConstraintType.SURFACE_VS_OPERATING,
            asset_id=data.asset_id,
            parameter_name="surface_vs_operating",
            actual_value=surface_temp,
            limit_value=operating_temp,
            unit="C",
            is_valid=is_valid,
            severity=severity,
            margin=margin,
            margin_percent=margin_percent,
            message=message,
            standard_reference="ASTM C680, Heat Transfer",
            recommended_action=action,
        )

    def _check_insulation_thickness(self, data: InsulationData) -> ConstraintCheckResult:
        """Check insulation thickness is within bounds."""
        limits = self.config.limits

        thickness = data.insulation_thickness_mm
        is_valid = limits.thickness_min_mm <= thickness <= limits.thickness_max_mm
        margin = min(thickness - limits.thickness_min_mm, limits.thickness_max_mm - thickness)
        margin_percent = (margin / (limits.thickness_max_mm - limits.thickness_min_mm)) * 100

        if thickness < limits.thickness_min_mm:
            severity = ConstraintSeverity.ERROR
            message = f"Thickness {thickness:.1f}mm below minimum {limits.thickness_min_mm:.1f}mm"
            action = "May indicate severe damage or measurement error"
        elif thickness > limits.thickness_max_mm:
            severity = ConstraintSeverity.WARNING
            message = f"Thickness {thickness:.1f}mm exceeds maximum {limits.thickness_max_mm:.1f}mm"
            action = "Verify measurement - unusually thick insulation"
        else:
            severity = ConstraintSeverity.INFO
            message = f"Thickness {thickness:.1f}mm is valid"
            action = "No action required"

        return ConstraintCheckResult(
            constraint_type=ConstraintType.INSULATION_THICKNESS,
            asset_id=data.asset_id,
            parameter_name="insulation_thickness",
            actual_value=thickness,
            limit_value=limits.thickness_max_mm,
            unit="mm",
            is_valid=is_valid,
            severity=severity,
            margin=margin,
            margin_percent=margin_percent,
            message=message,
            standard_reference="CINI Manual, ISO 12241",
            recommended_action=action,
        )

    def _check_surface_area(self, data: InsulationData) -> ConstraintCheckResult:
        """Check surface area is within bounds."""
        limits = self.config.limits

        area = data.surface_area_m2
        is_valid = limits.surface_area_min_m2 <= area <= limits.surface_area_max_m2
        margin = min(area - limits.surface_area_min_m2, limits.surface_area_max_m2 - area)
        margin_percent = (margin / limits.surface_area_max_m2) * 100 if limits.surface_area_max_m2 > 0 else 0

        if area < limits.surface_area_min_m2:
            severity = ConstraintSeverity.ERROR
            message = f"Surface area {area:.4f} m2 below minimum"
            action = "Check measurement or calculation"
        elif area > limits.surface_area_max_m2:
            severity = ConstraintSeverity.WARNING
            message = f"Surface area {area:.1f} m2 unusually large"
            action = "Verify measurement or asset scope"
        else:
            severity = ConstraintSeverity.INFO
            message = f"Surface area {area:.2f} m2 is valid"
            action = "No action required"

        return ConstraintCheckResult(
            constraint_type=ConstraintType.SURFACE_AREA,
            asset_id=data.asset_id,
            parameter_name="surface_area",
            actual_value=area,
            limit_value=limits.surface_area_max_m2,
            unit="m2",
            is_valid=is_valid,
            severity=severity,
            margin=margin,
            margin_percent=margin_percent,
            message=message,
            standard_reference="Engineering standards",
            recommended_action=action,
        )

    def _check_heat_loss_plausibility(self, data: InsulationData) -> List[ConstraintCheckResult]:
        """Check heat loss is plausible."""
        results = []
        limits = self.config.limits

        heat_loss = data.heat_loss_W_m2
        operating_temp = data.operating_temp_C
        ambient_temp = data.ambient_temp_C
        is_hot_equipment = operating_temp > ambient_temp

        # Check heat loss is positive for hot equipment
        if is_hot_equipment:
            is_positive_valid = heat_loss > 0
            if not is_positive_valid:
                severity = ConstraintSeverity.ERROR
                message = f"Heat loss {heat_loss:.1f} W/m2 should be positive for hot equipment"
                action = "Check calculation inputs"
            else:
                severity = ConstraintSeverity.INFO
                message = f"Heat loss {heat_loss:.1f} W/m2 has correct sign"
                action = "No action required"

            results.append(ConstraintCheckResult(
                constraint_type=ConstraintType.HEAT_LOSS_PLAUSIBILITY,
                asset_id=data.asset_id,
                parameter_name="heat_loss_sign",
                actual_value=heat_loss,
                limit_value=0.0,
                unit="W/m2",
                is_valid=is_positive_valid,
                severity=severity,
                margin=heat_loss if heat_loss > 0 else 0,
                margin_percent=100 if heat_loss > 0 else 0,
                message=message,
                standard_reference="ASTM C680, Heat Transfer",
                recommended_action=action,
            ))

        # Check heat loss within maximum
        is_max_valid = heat_loss <= limits.heat_loss_max_W_m2
        margin = limits.heat_loss_max_W_m2 - heat_loss
        margin_percent = (margin / limits.heat_loss_max_W_m2) * 100 if limits.heat_loss_max_W_m2 > 0 else 0

        if not is_max_valid:
            severity = ConstraintSeverity.ERROR
            message = f"Heat loss {heat_loss:.1f} W/m2 exceeds maximum {limits.heat_loss_max_W_m2:.1f}"
            action = "Verify measurement - may indicate bare surface"
        elif heat_loss > limits.heat_loss_max_W_m2 * 0.8:
            severity = ConstraintSeverity.WARNING
            message = f"Heat loss {heat_loss:.1f} W/m2 is very high"
            action = "Investigate for damaged insulation"
        else:
            severity = ConstraintSeverity.INFO
            message = f"Heat loss {heat_loss:.1f} W/m2 is within limits"
            action = "No action required"

        results.append(ConstraintCheckResult(
            constraint_type=ConstraintType.HEAT_LOSS_MAXIMUM,
            asset_id=data.asset_id,
            parameter_name="heat_loss_max",
            actual_value=heat_loss,
            limit_value=limits.heat_loss_max_W_m2,
            unit="W/m2",
            is_valid=is_max_valid,
            severity=severity,
            margin=margin,
            margin_percent=margin_percent,
            message=message,
            standard_reference="ASTM C680",
            recommended_action=action,
        ))

        return results

    def _check_efficiency_bounds(self, data: InsulationData) -> ConstraintCheckResult:
        """Check efficiency is in valid range [0, 1]."""
        limits = self.config.limits

        efficiency = data.efficiency
        is_valid = limits.efficiency_min <= efficiency <= limits.efficiency_max
        margin = min(efficiency - limits.efficiency_min, limits.efficiency_max - efficiency)
        margin_percent = margin * 100

        if efficiency < limits.efficiency_min:
            severity = ConstraintSeverity.ERROR
            message = f"Efficiency {efficiency:.2f} below minimum"
            action = "Check calculation - negative efficiency not physical"
        elif efficiency > limits.efficiency_max:
            severity = ConstraintSeverity.CRITICAL
            message = f"Efficiency {efficiency:.2f} exceeds 1.0 - impossible"
            action = "Verify calculation - efficiency cannot exceed 100%"
        elif efficiency < 0.1:
            severity = ConstraintSeverity.WARNING
            message = f"Efficiency {efficiency:.2f} is very low"
            action = "Investigate - may indicate severe degradation"
        else:
            severity = ConstraintSeverity.INFO
            message = f"Efficiency {efficiency:.2f} is valid"
            action = "No action required"

        return ConstraintCheckResult(
            constraint_type=ConstraintType.EFFICIENCY_BOUNDS,
            asset_id=data.asset_id,
            parameter_name="efficiency",
            actual_value=efficiency,
            limit_value=limits.efficiency_max,
            unit="dimensionless",
            is_valid=is_valid,
            severity=severity,
            margin=margin,
            margin_percent=margin_percent,
            message=message,
            standard_reference="ASTM C680, Heat Transfer",
            recommended_action=action,
        )

    def _check_condition_score_bounds(self, data: InsulationData) -> ConstraintCheckResult:
        """Check condition score is in valid range [0, 1]."""
        limits = self.config.limits

        score = data.condition_score
        is_valid = limits.condition_score_min <= score <= limits.condition_score_max
        margin = min(score - limits.condition_score_min, limits.condition_score_max - score)
        margin_percent = margin * 100

        if score < limits.condition_score_min:
            severity = ConstraintSeverity.ERROR
            message = f"Condition score {score:.2f} below minimum"
            action = "Check assessment calculation"
        elif score > limits.condition_score_max:
            severity = ConstraintSeverity.ERROR
            message = f"Condition score {score:.2f} exceeds maximum"
            action = "Check assessment calculation"
        else:
            severity = ConstraintSeverity.INFO
            message = f"Condition score {score:.2f} is valid"
            action = "No action required"

        return ConstraintCheckResult(
            constraint_type=ConstraintType.CONDITION_SCORE,
            asset_id=data.asset_id,
            parameter_name="condition_score",
            actual_value=score,
            limit_value=limits.condition_score_max,
            unit="dimensionless",
            is_valid=is_valid,
            severity=severity,
            margin=margin,
            margin_percent=margin_percent,
            message=message,
            standard_reference="Internal assessment standard",
            recommended_action=action,
        )

    # =========================================================================
    # SUMMARY AND ERROR HANDLING
    # =========================================================================

    def _build_summary(
        self,
        data: InsulationData,
        results: List[ConstraintCheckResult],
    ) -> InsulationConstraintValidationSummary:
        """Build validation summary from individual results."""
        # Categorize by severity
        critical_violations = [
            r for r in results
            if r.severity == ConstraintSeverity.CRITICAL and not r.is_valid
        ]
        errors = [
            r for r in results
            if r.severity == ConstraintSeverity.ERROR and not r.is_valid
        ]
        warnings = [
            r for r in results
            if r.severity == ConstraintSeverity.WARNING
        ]

        # Overall validity
        is_valid = len(critical_violations) == 0 and len(errors) == 0

        summary = InsulationConstraintValidationSummary(
            asset_id=data.asset_id,
            is_valid=is_valid,
            total_checks=len(results),
            violations_count=len(critical_violations) + len(errors),
            warnings_count=len(warnings),
            critical_violations=[r.to_dict() for r in critical_violations],
            errors=[r.to_dict() for r in errors],
            warnings=[r.to_dict() for r in warnings],
            all_results=[r.to_dict() for r in results],
            config_hash=self._config_hash,
            data_hash=self._compute_data_hash(data),
        )

        return summary

    def _raise_violation_error(
        self,
        data: InsulationData,
        summary: InsulationConstraintValidationSummary,
    ) -> None:
        """Raise appropriate violation error."""
        context = ViolationContext(
            asset_id=data.asset_id,
            location=data.location,
            sensor_readings={
                "operating_temp": data.operating_temp_C,
                "ambient_temp": data.ambient_temp_C,
                "surface_temp": data.surface_temp_C or 0.0,
            },
        )

        violations = []
        for v in summary.critical_violations + summary.errors:
            violations.append(ViolationDetails(
                constraint_tag=v["constraint_type"],
                constraint_description=v["message"],
                actual_value=v["actual_value"],
                limit_value=v["limit_value"],
                unit=v["unit"],
                severity=ViolationSeverity.ERROR,
                location=f"{data.asset_id}: {v['parameter_name']}",
                standard_reference=v["standard_reference"],
                recommended_action=v["recommended_action"],
            ))

        # Check for specific error types
        has_surface_ambient_violation = any(
            v["constraint_type"] == ConstraintType.SURFACE_VS_AMBIENT.value
            for v in summary.critical_violations
        )

        if has_surface_ambient_violation:
            raise SurfaceBelowAmbientError(
                surface_temp=data.surface_temp_C or 0.0,
                ambient_temp=data.ambient_temp_C,
                asset_id=data.asset_id,
                context=context,
            )
        else:
            raise ConstraintViolationError(
                message=f"Constraint violation for {data.asset_id}",
                constraint_name="multiple",
                actual_value=0.0,
                limit_value=0.0,
                unit="",
                violations=violations,
                context=context,
            )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def validate_temperature_relationship(
    operating_temp: float,
    ambient_temp: float,
    surface_temp: float,
) -> Tuple[bool, str]:
    """
    Quick check if temperature relationship is physically valid.

    Args:
        operating_temp: Operating/process temperature
        ambient_temp: Ambient temperature
        surface_temp: Measured surface temperature

    Returns:
        Tuple of (is_valid, message)
    """
    is_hot = operating_temp > ambient_temp

    if is_hot:
        if surface_temp < ambient_temp:
            return False, f"Surface {surface_temp}C < ambient {ambient_temp}C (impossible for hot equipment)"
        if surface_temp > operating_temp:
            return False, f"Surface {surface_temp}C > operating {operating_temp}C (impossible)"
    else:
        if surface_temp > ambient_temp:
            return False, f"Surface {surface_temp}C > ambient {ambient_temp}C (impossible for cold equipment)"
        if surface_temp < operating_temp:
            return False, f"Surface {surface_temp}C < operating {operating_temp}C (impossible)"

    return True, "Temperature relationship is valid"


def validate_heat_loss(
    heat_loss_W_m2: float,
    operating_temp: float,
    ambient_temp: float,
    max_heat_loss: float = 5000.0,
) -> Tuple[bool, str]:
    """
    Quick check if heat loss is plausible.

    Args:
        heat_loss_W_m2: Heat loss in W/m2
        operating_temp: Operating temperature
        ambient_temp: Ambient temperature
        max_heat_loss: Maximum plausible heat loss

    Returns:
        Tuple of (is_valid, message)
    """
    is_hot = operating_temp > ambient_temp

    if is_hot and heat_loss_W_m2 < 0:
        return False, "Heat loss should be positive for hot equipment"

    if heat_loss_W_m2 > max_heat_loss:
        return False, f"Heat loss {heat_loss_W_m2} W/m2 exceeds maximum {max_heat_loss}"

    return True, f"Heat loss {heat_loss_W_m2} W/m2 is plausible"


# =============================================================================
# MODULE EXPORTS
# =============================================================================


__all__ = [
    # Enums
    "ConstraintType",
    "ConstraintSeverity",
    "InsulationType",
    "AssetType",
    # Config
    "InsulationConstraintLimits",
    "InsulationConstraintValidatorConfig",
    # Data models
    "ConstraintCheckResult",
    "InsulationConstraintValidationSummary",
    "InsulationData",
    # Main class
    "InsulationConstraintValidator",
    # Convenience functions
    "validate_temperature_relationship",
    "validate_heat_loss",
]
