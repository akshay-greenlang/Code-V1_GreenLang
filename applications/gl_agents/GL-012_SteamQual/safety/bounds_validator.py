"""
GL-012 STEAMQUAL - Physical Bounds Validator

Production-grade physical bounds validation for steam quality parameters.
Ensures all sensor inputs and calculated values fall within physically
meaningful and safe ranges per ASME PTC 19.11 and related standards.

Physical Bounds Validated:
1. Pressure: 0.1 - 50 bar (absolute) for industrial steam systems
2. Temperature: 0 - 600 C (superheated steam range)
3. Quality: 0.0 - 1.0 (dryness fraction bounded)
4. Flow rate: 0 - 500,000 kg/hr (industrial capacity range)
5. Superheat: -50 to +200 C (allows for subcooled to highly superheated)

Standards Compliance:
    - ASME PTC 19.11 (Steam and Water Properties)
    - ASME B31.1 (Power Piping)
    - API 560 (Fired Heaters)
    - IAPWS-IF97 (Steam Properties)

Zero-Hallucination Guarantee:
All validation logic uses deterministic rules from published standards.
No LLM or AI inference in validation path.
SHA-256 provenance hashing for complete audit trail.

FAIL-SAFE Design:
When data quality is poor or values are at physical extremes,
validation FAILS to the safe side (flags as invalid) to prevent
control actions based on potentially erroneous data.

Example:
    >>> from safety.bounds_validator import SteamQualBoundsValidator
    >>> validator = SteamQualBoundsValidator()
    >>> result = validator.validate_steam_conditions(
    ...     pressure_bar=10.0,
    ...     temperature_c=185.0,
    ...     quality=0.95
    ... )
    >>> assert result.is_valid
    >>> print(result.provenance_hash)

Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================


class BoundsViolationSeverity(str, Enum):
    """Severity classification for bounds violations per ASME standards."""

    WARNING = "warning"      # Value near boundary - may affect accuracy
    ERROR = "error"          # Value outside valid range - invalid data
    CRITICAL = "critical"    # Value physically impossible - sensor failure


class ValidationStatus(str, Enum):
    """Overall validation status."""

    PASS = "pass"
    PASS_WITH_WARNINGS = "pass_with_warnings"
    FAIL = "fail"
    CRITICAL_FAIL = "critical_fail"


class ParameterCategory(str, Enum):
    """Parameter categories for grouping validations."""

    THERMAL = "thermal"         # Temperature-related
    PRESSURE = "pressure"       # Pressure-related
    QUALITY = "quality"         # Steam quality (dryness)
    FLOW = "flow"               # Flow rates
    DERIVED = "derived"         # Calculated parameters (superheat, enthalpy)
    TIMING = "timing"           # Time-related (ramp rates)


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass(frozen=True)
class PhysicalBounds:
    """
    Immutable physical bounds specification for a parameter.

    Defines the valid range for a measurement based on physical
    constraints and regulatory standards.

    Attributes:
        min_value: Minimum allowable value
        max_value: Maximum allowable value
        unit: Unit of measurement
        warning_margin: Fraction of range for boundary warnings (0.0-0.5)
        standard_reference: Regulatory standard reference
        category: Parameter category for grouping
        description: Human-readable description
    """
    min_value: float
    max_value: float
    unit: str
    warning_margin: float = 0.1
    standard_reference: str = ""
    category: ParameterCategory = ParameterCategory.DERIVED
    description: str = ""

    def get_range(self) -> float:
        """Calculate the valid range size."""
        return self.max_value - self.min_value

    def get_warning_threshold_low(self) -> float:
        """Get lower warning threshold."""
        return self.min_value + (self.get_range() * self.warning_margin)

    def get_warning_threshold_high(self) -> float:
        """Get upper warning threshold."""
        return self.max_value - (self.get_range() * self.warning_margin)


@dataclass(frozen=True)
class BoundsViolation:
    """
    Immutable record of a bounds violation.

    Contains complete information about the violation for audit trail
    and regulatory compliance documentation.
    """
    parameter: str
    value: float
    min_bound: float
    max_bound: float
    unit: str
    severity: BoundsViolationSeverity
    message: str
    standard_reference: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    category: ParameterCategory = ParameterCategory.DERIVED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "parameter": self.parameter,
            "value": self.value,
            "min_bound": self.min_bound,
            "max_bound": self.max_bound,
            "unit": self.unit,
            "severity": self.severity.value,
            "message": self.message,
            "standard_reference": self.standard_reference,
            "timestamp": self.timestamp.isoformat(),
            "category": self.category.value,
        }


# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class SteamQualDiagnosticInput(BaseModel):
    """
    Pydantic model for steam quality diagnostic input validation.

    Provides type-safe input validation with custom validators for
    physical bounds checking per ASME PTC 19.11.
    """

    # Pressure parameters (0.1-50 bar)
    pressure_bar: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Operating pressure in bar absolute (ASME PTC 19.11)"
    )

    # Temperature parameters (0-600 C)
    temperature_c: Optional[float] = Field(
        None,
        ge=-50.0,
        le=700.0,
        description="Temperature in Celsius (ASME PTC 19.11)"
    )

    saturation_temp_c: Optional[float] = Field(
        None,
        ge=-50.0,
        le=400.0,
        description="Saturation temperature at operating pressure"
    )

    # Quality parameters (0-1)
    quality: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Steam quality (dryness fraction)"
    )

    # Superheat parameters
    superheat_c: Optional[float] = Field(
        None,
        ge=-100.0,
        le=300.0,
        description="Superheat above saturation temperature"
    )

    # Flow parameters
    flow_rate_kg_hr: Optional[float] = Field(
        None,
        ge=0.0,
        le=1000000.0,
        description="Mass flow rate in kg/hr"
    )

    condensate_flow_kg_hr: Optional[float] = Field(
        None,
        ge=0.0,
        le=100000.0,
        description="Condensate flow rate in kg/hr"
    )

    # Derived parameters
    specific_enthalpy_kj_kg: Optional[float] = Field(
        None,
        ge=0.0,
        le=4000.0,
        description="Specific enthalpy in kJ/kg"
    )

    specific_entropy_kj_kg_k: Optional[float] = Field(
        None,
        ge=0.0,
        le=15.0,
        description="Specific entropy in kJ/(kg*K)"
    )

    density_kg_m3: Optional[float] = Field(
        None,
        ge=0.0,
        le=1500.0,
        description="Density in kg/m3"
    )

    velocity_m_s: Optional[float] = Field(
        None,
        ge=0.0,
        le=100.0,
        description="Steam velocity in m/s"
    )

    @field_validator('pressure_bar')
    @classmethod
    def validate_pressure(cls, v: Optional[float]) -> Optional[float]:
        """Validate pressure is within physical limits."""
        if v is not None and v < 0:
            raise ValueError("Pressure cannot be negative")
        return v

    @field_validator('quality')
    @classmethod
    def validate_quality(cls, v: Optional[float]) -> Optional[float]:
        """Validate quality is a valid fraction."""
        if v is not None and (v < 0 or v > 1):
            raise ValueError("Quality must be between 0 and 1")
        return v

    @model_validator(mode='after')
    def validate_thermodynamic_consistency(self) -> 'SteamQualDiagnosticInput':
        """Validate thermodynamic consistency between parameters."""
        # Superheat consistency check
        if (self.temperature_c is not None and
            self.saturation_temp_c is not None and
            self.superheat_c is not None):
            expected_superheat = self.temperature_c - self.saturation_temp_c
            if abs(expected_superheat - self.superheat_c) > 5.0:
                logger.warning(
                    f"Superheat inconsistency: expected {expected_superheat:.1f}C, "
                    f"provided {self.superheat_c:.1f}C"
                )
        return self


class BoundsValidationResult(BaseModel):
    """
    Pydantic model for validation result with provenance tracking.

    Provides complete audit trail for regulatory compliance.
    """

    is_valid: bool = Field(
        True,
        description="Whether all validations passed (no errors/critical)"
    )

    status: ValidationStatus = Field(
        ValidationStatus.PASS,
        description="Overall validation status"
    )

    validated_parameters: List[str] = Field(
        default_factory=list,
        description="List of parameters that were validated"
    )

    violations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of bounds violations"
    )

    error_count: int = Field(0, ge=0, description="Count of error-level violations")
    warning_count: int = Field(0, ge=0, description="Count of warning-level violations")
    critical_count: int = Field(0, ge=0, description="Count of critical-level violations")

    provenance_hash: str = Field(
        "",
        description="SHA-256 hash for audit trail"
    )

    validation_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When validation occurred"
    )

    validator_version: str = Field(
        "1.0.0",
        description="Bounds validator version"
    )

    standards_applied: List[str] = Field(
        default_factory=lambda: ["ASME PTC 19.11", "ASME B31.1"],
        description="Standards used for validation"
    )


# =============================================================================
# BOUNDS DEFINITIONS
# =============================================================================


# Complete bounds definitions per ASME PTC 19.11 and related standards
STEAM_QUALITY_BOUNDS: Dict[str, PhysicalBounds] = {
    # Pressure bounds (0.1-50 bar for industrial steam)
    "pressure_bar": PhysicalBounds(
        min_value=0.1,
        max_value=50.0,
        unit="bar",
        warning_margin=0.05,
        standard_reference="ASME PTC 19.11 Table 1",
        category=ParameterCategory.PRESSURE,
        description="Operating pressure in bar absolute"
    ),

    # Temperature bounds (0-600 C for superheated steam)
    "temperature_c": PhysicalBounds(
        min_value=0.0,
        max_value=600.0,
        unit="C",
        warning_margin=0.05,
        standard_reference="ASME PTC 19.11 Table 1",
        category=ParameterCategory.THERMAL,
        description="Temperature in Celsius"
    ),

    # Saturation temperature bounds
    "saturation_temp_c": PhysicalBounds(
        min_value=0.0,
        max_value=374.0,  # Critical point
        unit="C",
        warning_margin=0.05,
        standard_reference="IAPWS-IF97",
        category=ParameterCategory.THERMAL,
        description="Saturation temperature in Celsius"
    ),

    # Steam quality bounds (dryness fraction)
    "quality": PhysicalBounds(
        min_value=0.0,
        max_value=1.0,
        unit="fraction",
        warning_margin=0.02,  # Tighter margin for quality
        standard_reference="ASME PTC 19.11 Section 4.2",
        category=ParameterCategory.QUALITY,
        description="Steam quality (dryness fraction, 0-1)"
    ),

    # Superheat bounds
    "superheat_c": PhysicalBounds(
        min_value=-50.0,  # Allow subcooled conditions
        max_value=300.0,
        unit="C",
        warning_margin=0.10,
        standard_reference="ASME PTC 19.11",
        category=ParameterCategory.THERMAL,
        description="Superheat above saturation temperature"
    ),

    # Mass flow rate bounds
    "flow_rate_kg_hr": PhysicalBounds(
        min_value=0.0,
        max_value=500000.0,  # 500 t/hr for large industrial systems
        unit="kg/hr",
        warning_margin=0.10,
        standard_reference="ASME B31.1 Section 101",
        category=ParameterCategory.FLOW,
        description="Mass flow rate in kg/hr"
    ),

    # Condensate flow bounds
    "condensate_flow_kg_hr": PhysicalBounds(
        min_value=0.0,
        max_value=50000.0,
        unit="kg/hr",
        warning_margin=0.10,
        standard_reference="ASHRAE Handbook",
        category=ParameterCategory.FLOW,
        description="Condensate flow rate in kg/hr"
    ),

    # Specific enthalpy bounds (IAPWS-IF97)
    "specific_enthalpy_kj_kg": PhysicalBounds(
        min_value=0.0,
        max_value=4000.0,  # Well above superheated steam max
        unit="kJ/kg",
        warning_margin=0.10,
        standard_reference="IAPWS-IF97",
        category=ParameterCategory.DERIVED,
        description="Specific enthalpy in kJ/kg"
    ),

    # Specific entropy bounds
    "specific_entropy_kj_kg_k": PhysicalBounds(
        min_value=0.0,
        max_value=12.0,  # Practical limit for steam
        unit="kJ/(kg*K)",
        warning_margin=0.10,
        standard_reference="IAPWS-IF97",
        category=ParameterCategory.DERIVED,
        description="Specific entropy in kJ/(kg*K)"
    ),

    # Density bounds
    "density_kg_m3": PhysicalBounds(
        min_value=0.01,  # Very low for superheated steam
        max_value=1100.0,  # Compressed liquid water
        unit="kg/m3",
        warning_margin=0.10,
        standard_reference="IAPWS-IF97",
        category=ParameterCategory.DERIVED,
        description="Density in kg/m3"
    ),

    # Velocity bounds per ASME B31.1
    "velocity_m_s": PhysicalBounds(
        min_value=0.0,
        max_value=80.0,  # Max recommended for steam
        unit="m/s",
        warning_margin=0.15,
        standard_reference="ASME B31.1 Table 102.3.1",
        category=ParameterCategory.FLOW,
        description="Steam velocity in m/s"
    ),

    # Pressure ramp rate
    "pressure_ramp_bar_min": PhysicalBounds(
        min_value=-5.0,  # Allow depressurization
        max_value=5.0,
        unit="bar/min",
        warning_margin=0.10,
        standard_reference="API 530 Section 4.4",
        category=ParameterCategory.TIMING,
        description="Pressure ramp rate in bar/min"
    ),

    # Temperature ramp rate
    "temp_ramp_c_min": PhysicalBounds(
        min_value=-20.0,  # Allow cooldown
        max_value=20.0,
        unit="C/min",
        warning_margin=0.10,
        standard_reference="API 530 Section 4.4",
        category=ParameterCategory.TIMING,
        description="Temperature ramp rate in C/min"
    ),
}


# =============================================================================
# BOUNDS VALIDATOR
# =============================================================================


class SteamQualBoundsValidator:
    """
    Production-grade bounds validator for GL-012 STEAMQUAL.

    Provides comprehensive validation against ASME PTC 19.11 and
    related standards with SHA-256 provenance tracking for audit compliance.

    ZERO-HALLUCINATION GUARANTEE:
    - All validation uses deterministic rules from published standards
    - No LLM or ML inference in validation path
    - Same inputs always produce identical validation results
    - Complete provenance tracking with SHA-256 hashes

    FAIL-SAFE Design:
    - Values near physical limits trigger warnings
    - Values outside limits are flagged as errors
    - Physically impossible values (negative quality, etc.) are critical failures
    - When in doubt, validation fails to the safe side

    Features:
    - Pressure range validation (0.1-50 bar)
    - Temperature bounds (0-600 C)
    - Quality bounds (0-1)
    - Flow rate limits
    - Thermodynamic consistency checks
    - SHA-256 provenance for audit trail

    Example:
        >>> validator = SteamQualBoundsValidator()
        >>> result = validator.validate_steam_conditions(
        ...     pressure_bar=10.0,
        ...     temperature_c=185.0,
        ...     quality=0.95
        ... )
        >>> print(f"Valid: {result.is_valid}")
        >>> print(f"Hash: {result.provenance_hash}")
    """

    VERSION = "1.0.0"
    STANDARDS = ["ASME PTC 19.11", "ASME B31.1", "IAPWS-IF97"]

    def __init__(
        self,
        strict_mode: bool = True,
        custom_bounds: Optional[Dict[str, PhysicalBounds]] = None,
    ):
        """
        Initialize the bounds validator.

        Args:
            strict_mode: If True, fails on any warning or error
            custom_bounds: Optional custom bounds to override defaults
        """
        self.strict_mode = strict_mode
        self.bounds = dict(STEAM_QUALITY_BOUNDS)

        # Merge custom bounds if provided
        if custom_bounds:
            self.bounds.update(custom_bounds)

        self._validation_count = 0
        self._lock = threading.Lock()

        logger.info(
            f"SteamQualBoundsValidator v{self.VERSION} initialized "
            f"(strict_mode={strict_mode}, bounds_count={len(self.bounds)})"
        )

    # =========================================================================
    # SINGLE VALUE VALIDATION
    # =========================================================================

    def validate_value(
        self,
        parameter: str,
        value: float,
        bounds: Optional[PhysicalBounds] = None,
    ) -> Tuple[bool, Optional[BoundsViolation]]:
        """
        Validate a single parameter value against bounds.

        Args:
            parameter: Parameter name
            value: Value to validate
            bounds: Optional custom bounds (uses defaults if None)

        Returns:
            Tuple of (is_valid, violation or None)

        Example:
            >>> validator = SteamQualBoundsValidator()
            >>> is_valid, violation = validator.validate_value("pressure_bar", 10.0)
            >>> assert is_valid is True
        """
        if bounds is None:
            bounds = self.bounds.get(parameter)
            if bounds is None:
                logger.debug(f"No bounds defined for parameter: {parameter}")
                return True, None

        timestamp = datetime.now(timezone.utc)

        # FAIL-SAFE: NaN values are critical violations
        if value is None or math.isnan(value):
            return False, BoundsViolation(
                parameter=parameter,
                value=0.0,
                min_bound=bounds.min_value,
                max_bound=bounds.max_value,
                unit=bounds.unit,
                severity=BoundsViolationSeverity.CRITICAL,
                message=f"{parameter} is NaN or None - sensor failure suspected",
                standard_reference=bounds.standard_reference,
                timestamp=timestamp,
                category=bounds.category,
            )

        # FAIL-SAFE: Infinite values are critical violations
        if math.isinf(value):
            return False, BoundsViolation(
                parameter=parameter,
                value=value,
                min_bound=bounds.min_value,
                max_bound=bounds.max_value,
                unit=bounds.unit,
                severity=BoundsViolationSeverity.CRITICAL,
                message=f"{parameter} is infinite - calculation error",
                standard_reference=bounds.standard_reference,
                timestamp=timestamp,
                category=bounds.category,
            )

        # Check if value is below minimum
        if value < bounds.min_value:
            # Determine severity based on how far below
            margin = bounds.min_value - value
            relative_margin = margin / bounds.get_range() if bounds.get_range() > 0 else 1.0

            if relative_margin > 0.2 or value < 0:  # More than 20% below or negative
                severity = BoundsViolationSeverity.CRITICAL
            else:
                severity = BoundsViolationSeverity.ERROR

            return False, BoundsViolation(
                parameter=parameter,
                value=value,
                min_bound=bounds.min_value,
                max_bound=bounds.max_value,
                unit=bounds.unit,
                severity=severity,
                message=(
                    f"{parameter}={value:.4f} {bounds.unit} is below minimum "
                    f"{bounds.min_value} {bounds.unit}"
                ),
                standard_reference=bounds.standard_reference,
                timestamp=timestamp,
                category=bounds.category,
            )

        # Check if value exceeds maximum
        if value > bounds.max_value:
            margin = value - bounds.max_value
            relative_margin = margin / bounds.get_range() if bounds.get_range() > 0 else 1.0

            if relative_margin > 0.2:  # More than 20% above
                severity = BoundsViolationSeverity.CRITICAL
            else:
                severity = BoundsViolationSeverity.ERROR

            return False, BoundsViolation(
                parameter=parameter,
                value=value,
                min_bound=bounds.min_value,
                max_bound=bounds.max_value,
                unit=bounds.unit,
                severity=severity,
                message=(
                    f"{parameter}={value:.4f} {bounds.unit} exceeds maximum "
                    f"{bounds.max_value} {bounds.unit}"
                ),
                standard_reference=bounds.standard_reference,
                timestamp=timestamp,
                category=bounds.category,
            )

        # Check for warning zone (near boundaries)
        if bounds.warning_margin > 0:
            low_threshold = bounds.get_warning_threshold_low()
            high_threshold = bounds.get_warning_threshold_high()

            if value < low_threshold:
                return True, BoundsViolation(
                    parameter=parameter,
                    value=value,
                    min_bound=bounds.min_value,
                    max_bound=bounds.max_value,
                    unit=bounds.unit,
                    severity=BoundsViolationSeverity.WARNING,
                    message=(
                        f"{parameter}={value:.4f} {bounds.unit} is near "
                        f"minimum boundary ({bounds.min_value})"
                    ),
                    standard_reference=bounds.standard_reference,
                    timestamp=timestamp,
                    category=bounds.category,
                )

            if value > high_threshold:
                return True, BoundsViolation(
                    parameter=parameter,
                    value=value,
                    min_bound=bounds.min_value,
                    max_bound=bounds.max_value,
                    unit=bounds.unit,
                    severity=BoundsViolationSeverity.WARNING,
                    message=(
                        f"{parameter}={value:.4f} {bounds.unit} is near "
                        f"maximum boundary ({bounds.max_value})"
                    ),
                    standard_reference=bounds.standard_reference,
                    timestamp=timestamp,
                    category=bounds.category,
                )

        return True, None

    # =========================================================================
    # COMPREHENSIVE VALIDATION
    # =========================================================================

    def validate_steam_conditions(
        self,
        pressure_bar: Optional[float] = None,
        temperature_c: Optional[float] = None,
        quality: Optional[float] = None,
        superheat_c: Optional[float] = None,
        flow_rate_kg_hr: Optional[float] = None,
        velocity_m_s: Optional[float] = None,
        **kwargs: Any,
    ) -> BoundsValidationResult:
        """
        Validate complete steam conditions.

        Args:
            pressure_bar: Operating pressure in bar
            temperature_c: Temperature in Celsius
            quality: Steam quality (0-1)
            superheat_c: Superheat above saturation
            flow_rate_kg_hr: Mass flow rate in kg/hr
            velocity_m_s: Steam velocity in m/s
            **kwargs: Additional parameters to validate

        Returns:
            BoundsValidationResult with complete validation details

        Example:
            >>> result = validator.validate_steam_conditions(
            ...     pressure_bar=10.0,
            ...     temperature_c=185.0,
            ...     quality=0.95
            ... )
            >>> assert result.is_valid
        """
        with self._lock:
            self._validation_count += 1

        violations: List[BoundsViolation] = []
        validated: List[str] = []

        # Build parameter dictionary
        params = {
            "pressure_bar": pressure_bar,
            "temperature_c": temperature_c,
            "quality": quality,
            "superheat_c": superheat_c,
            "flow_rate_kg_hr": flow_rate_kg_hr,
            "velocity_m_s": velocity_m_s,
        }
        params.update(kwargs)

        # Validate each provided parameter
        for param, value in params.items():
            if value is not None:
                is_valid, violation = self.validate_value(param, value)
                validated.append(param)
                if violation:
                    violations.append(violation)
                    logger.debug(
                        f"Bounds validation: {param}={value} -> "
                        f"{violation.severity.value}: {violation.message}"
                    )

        return self._build_result(validated, violations, params)

    def validate_diagnostic_input(
        self,
        input_data: SteamQualDiagnosticInput,
    ) -> BoundsValidationResult:
        """
        Validate Pydantic diagnostic input model.

        Args:
            input_data: SteamQualDiagnosticInput model

        Returns:
            BoundsValidationResult
        """
        # Extract values from model
        params = {
            k: v for k, v in input_data.model_dump().items()
            if v is not None
        }

        return self.validate_steam_conditions(**params)

    def validate_flow_parameters(
        self,
        flow_rate_kg_hr: Optional[float] = None,
        condensate_flow_kg_hr: Optional[float] = None,
        velocity_m_s: Optional[float] = None,
        location_id: str = "unknown",
    ) -> BoundsValidationResult:
        """
        Validate flow-related parameters.

        Args:
            flow_rate_kg_hr: Steam mass flow rate
            condensate_flow_kg_hr: Condensate flow rate
            velocity_m_s: Steam velocity
            location_id: Location identifier

        Returns:
            BoundsValidationResult
        """
        return self.validate_steam_conditions(
            flow_rate_kg_hr=flow_rate_kg_hr,
            condensate_flow_kg_hr=condensate_flow_kg_hr,
            velocity_m_s=velocity_m_s,
        )

    def validate_thermodynamic_properties(
        self,
        pressure_bar: float,
        temperature_c: float,
        quality: Optional[float] = None,
        specific_enthalpy_kj_kg: Optional[float] = None,
        specific_entropy_kj_kg_k: Optional[float] = None,
        density_kg_m3: Optional[float] = None,
    ) -> BoundsValidationResult:
        """
        Validate thermodynamic properties with consistency checks.

        Args:
            pressure_bar: Operating pressure
            temperature_c: Temperature
            quality: Steam quality (if two-phase)
            specific_enthalpy_kj_kg: Specific enthalpy
            specific_entropy_kj_kg_k: Specific entropy
            density_kg_m3: Density

        Returns:
            BoundsValidationResult
        """
        with self._lock:
            self._validation_count += 1

        violations: List[BoundsViolation] = []
        validated: List[str] = []

        # Validate individual parameters
        params = {
            "pressure_bar": pressure_bar,
            "temperature_c": temperature_c,
            "quality": quality,
            "specific_enthalpy_kj_kg": specific_enthalpy_kj_kg,
            "specific_entropy_kj_kg_k": specific_entropy_kj_kg_k,
            "density_kg_m3": density_kg_m3,
        }

        for param, value in params.items():
            if value is not None:
                is_valid, violation = self.validate_value(param, value)
                validated.append(param)
                if violation:
                    violations.append(violation)

        # Thermodynamic consistency checks
        # Quality can only exist below critical point
        if quality is not None and pressure_bar is not None:
            critical_pressure = 220.64  # bar
            if pressure_bar >= critical_pressure and quality < 1.0:
                violations.append(BoundsViolation(
                    parameter="quality_consistency",
                    value=quality,
                    min_bound=1.0,
                    max_bound=1.0,
                    unit="fraction",
                    severity=BoundsViolationSeverity.ERROR,
                    message=(
                        f"Quality={quality:.2%} is invalid at supercritical pressure "
                        f"({pressure_bar:.1f} bar >= {critical_pressure} bar)"
                    ),
                    standard_reference="IAPWS-IF97",
                    category=ParameterCategory.QUALITY,
                ))
                validated.append("quality_consistency")

        return self._build_result(validated, violations, params)

    # =========================================================================
    # RESULT BUILDING
    # =========================================================================

    def _build_result(
        self,
        validated: List[str],
        violations: List[BoundsViolation],
        input_params: Dict[str, Any],
    ) -> BoundsValidationResult:
        """Build complete validation result with provenance hash."""
        # Count violations by severity
        error_count = sum(
            1 for v in violations
            if v.severity in [BoundsViolationSeverity.ERROR, BoundsViolationSeverity.CRITICAL]
        )
        warning_count = sum(
            1 for v in violations
            if v.severity == BoundsViolationSeverity.WARNING
        )
        critical_count = sum(
            1 for v in violations
            if v.severity == BoundsViolationSeverity.CRITICAL
        )

        # Determine overall validity and status
        if critical_count > 0:
            is_valid = False
            status = ValidationStatus.CRITICAL_FAIL
        elif error_count > 0:
            is_valid = False
            status = ValidationStatus.FAIL
        elif warning_count > 0 and self.strict_mode:
            is_valid = False
            status = ValidationStatus.FAIL
        elif warning_count > 0:
            is_valid = True
            status = ValidationStatus.PASS_WITH_WARNINGS
        else:
            is_valid = True
            status = ValidationStatus.PASS

        # Calculate provenance hash
        provenance_data = {
            "input_params": {k: v for k, v in input_params.items() if v is not None},
            "validated_parameters": validated,
            "violation_count": len(violations),
            "validator_version": self.VERSION,
            "standards": self.STANDARDS,
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True, default=str).encode()
        ).hexdigest()

        return BoundsValidationResult(
            is_valid=is_valid,
            status=status,
            validated_parameters=validated,
            violations=[v.to_dict() for v in violations],
            error_count=error_count,
            warning_count=warning_count,
            critical_count=critical_count,
            provenance_hash=provenance_hash,
            validator_version=self.VERSION,
            standards_applied=self.STANDARDS,
        )

    # =========================================================================
    # UTILITIES
    # =========================================================================

    def get_bounds(self, parameter: str) -> Optional[PhysicalBounds]:
        """Get bounds specification for a parameter."""
        return self.bounds.get(parameter)

    def get_all_bounds(self) -> Dict[str, PhysicalBounds]:
        """Get all bounds specifications."""
        return dict(self.bounds)

    def get_statistics(self) -> Dict[str, Any]:
        """Get validator statistics."""
        with self._lock:
            return {
                "version": self.VERSION,
                "validation_count": self._validation_count,
                "bounds_count": len(self.bounds),
                "strict_mode": self.strict_mode,
                "standards": self.STANDARDS,
                "categories": list(set(b.category.value for b in self.bounds.values())),
            }

    # =========================================================================
    # UNIT CONVERSION UTILITIES
    # =========================================================================

    @staticmethod
    def celsius_to_kelvin(celsius: float) -> float:
        """Convert Celsius to Kelvin."""
        return celsius + 273.15

    @staticmethod
    def kelvin_to_celsius(kelvin: float) -> float:
        """Convert Kelvin to Celsius."""
        return kelvin - 273.15

    @staticmethod
    def bar_to_kpa(bar: float) -> float:
        """Convert bar to kPa."""
        return bar * 100.0

    @staticmethod
    def kpa_to_bar(kpa: float) -> float:
        """Convert kPa to bar."""
        return kpa / 100.0

    @staticmethod
    def bar_to_psi(bar: float) -> float:
        """Convert bar to PSI."""
        return bar * 14.5038

    @staticmethod
    def psi_to_bar(psi: float) -> float:
        """Convert PSI to bar."""
        return psi / 14.5038

    @staticmethod
    def kg_hr_to_kg_s(kg_hr: float) -> float:
        """Convert kg/hr to kg/s."""
        return kg_hr / 3600.0

    @staticmethod
    def kg_s_to_kg_hr(kg_s: float) -> float:
        """Convert kg/s to kg/hr."""
        return kg_s * 3600.0


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Main validator
    "SteamQualBoundsValidator",

    # Data classes
    "PhysicalBounds",
    "BoundsViolation",
    "BoundsValidationResult",

    # Pydantic models
    "SteamQualDiagnosticInput",

    # Enums
    "BoundsViolationSeverity",
    "ValidationStatus",
    "ParameterCategory",

    # Constants
    "STEAM_QUALITY_BOUNDS",
]
