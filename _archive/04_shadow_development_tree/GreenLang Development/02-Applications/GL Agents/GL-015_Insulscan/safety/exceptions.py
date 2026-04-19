# -*- coding: utf-8 -*-
"""
GL-015 Insulscan - Safety Exceptions Module

Exception hierarchy for safety constraint violations and operational safety
in insulation scanning and thermal assessment systems.

All exceptions follow fail-safe design principles:
- Violations trigger immediate rejection of unsafe recommendations
- Clear error messages with actionable remediation guidance
- Full provenance tracking for audit trails
- Severity classification for appropriate escalation

Safety Principles:
- Never present predictions as certainties
- Fail safe on poor data quality
- Request engineering review when outside training distribution
- Personnel safety limits for high-temperature surfaces

References:
- ASTM C680: Heat Loss Calculations for Insulation
- CINI Manual: Insulation Inspection Guidelines
- OSHA 1910.147: Burn Prevention Standards
- ISO 12241: Thermal Insulation for Building Equipment

Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class ViolationSeverity(str, Enum):
    """
    Severity levels for safety violations.

    Levels follow industrial safety guidelines:
    - INFO: Logged for awareness, no action required
    - WARNING: Log and continue with penalty/adjustment
    - ERROR: Reject recommendation, may be overridden with authorization
    - CRITICAL: Reject recommendation, cannot be overridden (fail-closed)
    """

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class SafetyDomain(str, Enum):
    """
    Domain classification for safety violations.

    Helps route exceptions to appropriate handling systems.
    """

    THERMAL_MEASUREMENT = "thermal_measurement"
    INSULATION_PHYSICAL = "insulation_physical"
    PERSONNEL_SAFETY = "personnel_safety"
    MODEL_SERVICE = "model_service"
    OPERATIONAL = "operational"
    DATA_QUALITY = "data_quality"
    CONSTRAINT_VIOLATION = "constraint_violation"


@dataclass(frozen=True)
class ViolationContext:
    """
    Immutable context information for a safety violation.

    Provides complete traceability for audit and root cause analysis.

    Attributes:
        asset_id: Insulation asset identifier
        timestamp: When the violation occurred
        sensor_readings: Relevant sensor data at time of violation
        location: Physical location in plant
        provenance_hash: SHA-256 hash for integrity verification
    """

    asset_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    sensor_readings: Dict[str, float] = field(default_factory=dict)
    location: str = "unknown"
    additional_context: Dict[str, Any] = field(default_factory=dict)
    provenance_hash: str = ""

    def __post_init__(self) -> None:
        """Calculate provenance hash after initialization."""
        if not self.provenance_hash:
            content = (
                f"{self.asset_id}|{self.timestamp.isoformat()}|"
                f"{self.location}|{str(sorted(self.sensor_readings.items()))}"
            )
            object.__setattr__(
                self,
                'provenance_hash',
                hashlib.sha256(content.encode()).hexdigest()
            )


@dataclass(frozen=True)
class ViolationDetails:
    """
    Immutable details about a specific safety violation.

    Provides actionable information for remediation.

    Attributes:
        constraint_tag: Unique identifier for the constraint
        constraint_description: Human-readable description
        actual_value: Measured/calculated value that violated constraint
        limit_value: The constraint limit that was violated
        unit: Engineering unit for the values
        severity: Violation severity level
        location: Where the violation occurred
        standard_reference: Applicable standard
        recommended_action: Suggested remediation steps
    """

    constraint_tag: str
    constraint_description: str
    actual_value: float
    limit_value: float
    unit: str
    severity: ViolationSeverity
    location: str
    standard_reference: str
    recommended_action: str
    domain: SafetyDomain = SafetyDomain.THERMAL_MEASUREMENT

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "constraint_tag": self.constraint_tag,
            "constraint_description": self.constraint_description,
            "actual_value": self.actual_value,
            "limit_value": self.limit_value,
            "unit": self.unit,
            "severity": self.severity.value,
            "location": self.location,
            "standard_reference": self.standard_reference,
            "recommended_action": self.recommended_action,
            "domain": self.domain.value,
        }


# =============================================================================
# BASE EXCEPTION CLASSES
# =============================================================================


class InsulscanSafetyError(Exception):
    """
    Base exception for all GL-015 Insulscan safety violations.

    This exception and its subclasses implement fail-closed behavior:
    any unhandled safety exception results in rejection of the
    recommendation or action.

    Attributes:
        message: Human-readable error description
        violations: List of specific violation details
        context: Context information for the violation
        severity: Overall severity (highest of all violations)
        domain: Primary safety domain
    """

    def __init__(
        self,
        message: str,
        violations: Optional[List[ViolationDetails]] = None,
        context: Optional[ViolationContext] = None,
        severity: ViolationSeverity = ViolationSeverity.ERROR,
        domain: SafetyDomain = SafetyDomain.THERMAL_MEASUREMENT,
    ) -> None:
        """
        Initialize safety exception.

        Args:
            message: Human-readable error description
            violations: List of specific violation details
            context: Context information for the violation
            severity: Overall severity level
            domain: Primary safety domain
        """
        super().__init__(message)
        self.message = message
        self.violations = violations or []
        self.context = context
        self.severity = severity
        self.domain = domain
        self.timestamp = datetime.now(timezone.utc)
        self._provenance_hash = self._calculate_provenance()

    def _calculate_provenance(self) -> str:
        """Calculate SHA-256 provenance hash for audit trail."""
        content = (
            f"{self.message}|{self.severity.value}|{self.domain.value}|"
            f"{self.timestamp.isoformat()}|{len(self.violations)}"
        )
        return hashlib.sha256(content.encode()).hexdigest()

    @property
    def provenance_hash(self) -> str:
        """Get provenance hash for audit trail."""
        return self._provenance_hash

    def get_violation_summary(self) -> str:
        """
        Generate a formatted summary of all violations.

        Returns:
            Multi-line string with all violation details.
        """
        if not self.violations:
            return f"[{self.severity.value.upper()}] {self.message}"

        lines = [
            f"Safety violations detected ({len(self.violations)} total):",
            f"  Domain: {self.domain.value}",
            f"  Severity: {self.severity.value.upper()}",
            "",
        ]

        for i, v in enumerate(self.violations, 1):
            lines.append(
                f"  {i}. [{v.severity.value.upper()}] {v.constraint_tag}: "
                f"{v.actual_value:.4f} {v.unit} (limit: {v.limit_value:.4f} {v.unit})"
            )
            lines.append(f"     Location: {v.location}")
            lines.append(f"     Standard: {v.standard_reference}")
            lines.append(f"     Action: {v.recommended_action}")
            lines.append("")

        if self.context:
            lines.append(f"  Context: Asset={self.context.asset_id}, "
                        f"Location={self.context.location}")

        lines.append(f"  Provenance: {self._provenance_hash[:16]}...")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "severity": self.severity.value,
            "domain": self.domain.value,
            "timestamp": self.timestamp.isoformat(),
            "violations": [v.to_dict() for v in self.violations],
            "context": {
                "asset_id": self.context.asset_id if self.context else None,
                "location": self.context.location if self.context else None,
                "provenance_hash": self.context.provenance_hash if self.context else None,
            },
            "provenance_hash": self._provenance_hash,
        }


# =============================================================================
# INSULATION VALIDATION ERRORS
# =============================================================================


class InsulationValidationError(InsulscanSafetyError):
    """
    Validation error for insulation assessment data.

    Raised when input data fails validation checks:
    - Invalid temperature readings
    - Missing required measurements
    - Inconsistent data
    """

    def __init__(
        self,
        message: str,
        parameter_name: str,
        actual_value: Optional[float] = None,
        expected_range: Optional[str] = None,
        context: Optional[ViolationContext] = None,
    ) -> None:
        """
        Initialize insulation validation error.

        Args:
            message: Human-readable error description
            parameter_name: Name of the invalid parameter
            actual_value: The invalid value
            expected_range: Expected valid range
            context: Context information
        """
        self.parameter_name = parameter_name
        self.actual_value = actual_value
        self.expected_range = expected_range

        violations = [
            ViolationDetails(
                constraint_tag=f"VALIDATION_{parameter_name.upper()}",
                constraint_description=f"Validation error for {parameter_name}",
                actual_value=actual_value or 0.0,
                limit_value=0.0,
                unit="",
                severity=ViolationSeverity.ERROR,
                location=f"Parameter: {parameter_name}",
                standard_reference="Internal validation rules",
                recommended_action=f"Verify input data. Expected: {expected_range or 'valid value'}",
                domain=SafetyDomain.DATA_QUALITY,
            )
        ]

        super().__init__(
            message=message,
            violations=violations,
            context=context,
            severity=ViolationSeverity.ERROR,
            domain=SafetyDomain.DATA_QUALITY,
        )


# =============================================================================
# THERMAL MEASUREMENT ERRORS
# =============================================================================


class ThermalMeasurementError(InsulscanSafetyError):
    """
    Error in thermal measurement data.

    Raised when thermal measurements are invalid or inconsistent:
    - Surface temperature below ambient (impossible)
    - Extreme temperature gradients
    - IR scanner calibration issues
    """

    def __init__(
        self,
        message: str,
        measurement_type: str,
        measured_value: float,
        expected_range: str,
        asset_id: str,
        context: Optional[ViolationContext] = None,
    ) -> None:
        """
        Initialize thermal measurement error.

        Args:
            message: Human-readable error description
            measurement_type: Type of measurement (surface_temp, ambient_temp, etc.)
            measured_value: The problematic measurement
            expected_range: Expected valid range
            asset_id: Asset identifier
            context: Context information
        """
        self.measurement_type = measurement_type
        self.measured_value = measured_value
        self.expected_range = expected_range
        self.asset_id = asset_id

        if context is None:
            context = ViolationContext(asset_id=asset_id)

        violations = [
            ViolationDetails(
                constraint_tag=f"THERMAL_{measurement_type.upper()}",
                constraint_description=f"Invalid thermal measurement: {measurement_type}",
                actual_value=measured_value,
                limit_value=0.0,
                unit="C",
                severity=ViolationSeverity.ERROR,
                location=f"Asset {asset_id}",
                standard_reference="ASTM C680, IR Thermography Standards",
                recommended_action=f"Verify IR scanner calibration. Expected: {expected_range}",
                domain=SafetyDomain.THERMAL_MEASUREMENT,
            )
        ]

        super().__init__(
            message=message,
            violations=violations,
            context=context,
            severity=ViolationSeverity.ERROR,
            domain=SafetyDomain.THERMAL_MEASUREMENT,
        )


class SurfaceBelowAmbientError(ThermalMeasurementError):
    """Surface temperature measured below ambient - physically impossible."""

    def __init__(
        self,
        surface_temp: float,
        ambient_temp: float,
        asset_id: str,
        context: Optional[ViolationContext] = None,
    ) -> None:
        """
        Initialize surface below ambient error.

        Args:
            surface_temp: Measured surface temperature
            ambient_temp: Ambient temperature
            asset_id: Asset identifier
            context: Context information
        """
        message = (
            f"Asset {asset_id}: Surface temperature {surface_temp:.1f}C is below "
            f"ambient {ambient_temp:.1f}C. This is physically impossible for "
            f"insulated hot equipment. Check IR scanner calibration."
        )
        super().__init__(
            message=message,
            measurement_type="surface_vs_ambient",
            measured_value=surface_temp,
            expected_range=f"> {ambient_temp:.1f}C (ambient)",
            asset_id=asset_id,
            context=context,
        )


# =============================================================================
# CONSTRAINT VIOLATION ERRORS
# =============================================================================


class ConstraintViolationError(InsulscanSafetyError):
    """
    Physical constraint violation detected.

    Raised when calculated or measured values violate physical constraints:
    - Temperature ranges
    - Insulation thickness bounds
    - Surface area limits
    - Heat loss plausibility
    """

    def __init__(
        self,
        message: str,
        constraint_name: str,
        actual_value: float,
        limit_value: float,
        unit: str,
        violations: Optional[List[ViolationDetails]] = None,
        context: Optional[ViolationContext] = None,
    ) -> None:
        """
        Initialize constraint violation error.

        Args:
            message: Human-readable error description
            constraint_name: Name of violated constraint
            actual_value: The violating value
            limit_value: The constraint limit
            unit: Engineering unit
            violations: Additional violation details
            context: Context information
        """
        self.constraint_name = constraint_name
        self.actual_value = actual_value
        self.limit_value = limit_value
        self.unit = unit

        if not violations:
            violations = [
                ViolationDetails(
                    constraint_tag=f"CONSTRAINT_{constraint_name.upper()}",
                    constraint_description=f"Physical constraint: {constraint_name}",
                    actual_value=actual_value,
                    limit_value=limit_value,
                    unit=unit,
                    severity=ViolationSeverity.ERROR,
                    location="Constraint validation",
                    standard_reference="ASTM C680, ISO 12241",
                    recommended_action="Verify input data and recalculate",
                    domain=SafetyDomain.CONSTRAINT_VIOLATION,
                )
            ]

        super().__init__(
            message=message,
            violations=violations,
            context=context,
            severity=ViolationSeverity.ERROR,
            domain=SafetyDomain.CONSTRAINT_VIOLATION,
        )


class InsulationThicknessError(ConstraintViolationError):
    """Insulation thickness outside valid range."""

    def __init__(
        self,
        thickness_mm: float,
        min_mm: float,
        max_mm: float,
        asset_id: str,
        context: Optional[ViolationContext] = None,
    ) -> None:
        """
        Initialize insulation thickness error.

        Args:
            thickness_mm: Measured/specified thickness
            min_mm: Minimum valid thickness
            max_mm: Maximum valid thickness
            asset_id: Asset identifier
            context: Context information
        """
        if thickness_mm < min_mm:
            message = (
                f"Asset {asset_id}: Insulation thickness {thickness_mm:.1f}mm "
                f"below minimum {min_mm:.1f}mm. May indicate severe damage."
            )
        else:
            message = (
                f"Asset {asset_id}: Insulation thickness {thickness_mm:.1f}mm "
                f"exceeds maximum {max_mm:.1f}mm. Verify measurement."
            )

        if context is None:
            context = ViolationContext(asset_id=asset_id)

        super().__init__(
            message=message,
            constraint_name="insulation_thickness",
            actual_value=thickness_mm,
            limit_value=max_mm if thickness_mm > max_mm else min_mm,
            unit="mm",
            context=context,
        )


class HeatLossImplausibleError(ConstraintViolationError):
    """Calculated heat loss is physically implausible."""

    def __init__(
        self,
        heat_loss_W_m2: float,
        max_plausible: float,
        asset_id: str,
        context: Optional[ViolationContext] = None,
    ) -> None:
        """
        Initialize heat loss implausible error.

        Args:
            heat_loss_W_m2: Calculated heat loss
            max_plausible: Maximum plausible heat loss
            asset_id: Asset identifier
            context: Context information
        """
        message = (
            f"Asset {asset_id}: Calculated heat loss {heat_loss_W_m2:.1f} W/m2 "
            f"exceeds maximum plausible {max_plausible:.1f} W/m2. "
            f"Check temperature differential and insulation properties."
        )

        if context is None:
            context = ViolationContext(asset_id=asset_id)

        super().__init__(
            message=message,
            constraint_name="heat_loss_plausibility",
            actual_value=heat_loss_W_m2,
            limit_value=max_plausible,
            unit="W/m2",
            context=context,
        )


# =============================================================================
# SAFETY LIMIT ERRORS
# =============================================================================


class SafetyLimitExceededError(InsulscanSafetyError):
    """
    Safety limit exceeded.

    Raised when safety-critical limits are exceeded:
    - Personnel burn risk temperatures
    - Maximum investment thresholds
    - Priority override conditions
    """

    def __init__(
        self,
        message: str,
        safety_type: str,
        actual_value: float,
        safety_limit: float,
        unit: str,
        context: Optional[ViolationContext] = None,
    ) -> None:
        """
        Initialize safety limit exceeded error.

        Args:
            message: Human-readable error description
            safety_type: Type of safety limit
            actual_value: The value exceeding the limit
            safety_limit: The safety limit
            unit: Engineering unit
            context: Context information
        """
        self.safety_type = safety_type
        self.actual_value = actual_value
        self.safety_limit = safety_limit
        self.unit = unit

        violations = [
            ViolationDetails(
                constraint_tag=f"SAFETY_{safety_type.upper()}",
                constraint_description=f"Safety limit: {safety_type}",
                actual_value=actual_value,
                limit_value=safety_limit,
                unit=unit,
                severity=ViolationSeverity.CRITICAL,
                location="Safety system",
                standard_reference="OSHA 1910.147, Plant Safety Standards",
                recommended_action="Immediate attention required",
                domain=SafetyDomain.PERSONNEL_SAFETY,
            )
        ]

        super().__init__(
            message=message,
            violations=violations,
            context=context,
            severity=ViolationSeverity.CRITICAL,
            domain=SafetyDomain.PERSONNEL_SAFETY,
        )


class BurnRiskError(SafetyLimitExceededError):
    """Surface temperature poses burn risk to personnel."""

    def __init__(
        self,
        surface_temp: float,
        burn_threshold: float,
        asset_id: str,
        context: Optional[ViolationContext] = None,
    ) -> None:
        """
        Initialize burn risk error.

        Args:
            surface_temp: Measured surface temperature
            burn_threshold: Temperature threshold for burn risk
            asset_id: Asset identifier
            context: Context information
        """
        message = (
            f"BURN HAZARD: Asset {asset_id} surface temperature {surface_temp:.1f}C "
            f"exceeds personnel safety limit {burn_threshold:.1f}C. "
            f"IMMEDIATE action required to prevent injury."
        )

        if context is None:
            context = ViolationContext(asset_id=asset_id)

        super().__init__(
            message=message,
            safety_type="burn_risk",
            actual_value=surface_temp,
            safety_limit=burn_threshold,
            unit="C",
            context=context,
        )


class InvestmentLimitExceededError(SafetyLimitExceededError):
    """Recommended repair investment exceeds authorization limit."""

    def __init__(
        self,
        estimated_cost: float,
        max_authorized: float,
        asset_id: str,
        context: Optional[ViolationContext] = None,
    ) -> None:
        """
        Initialize investment limit exceeded error.

        Args:
            estimated_cost: Estimated repair cost
            max_authorized: Maximum authorized investment
            asset_id: Asset identifier
            context: Context information
        """
        message = (
            f"Asset {asset_id}: Estimated repair cost ${estimated_cost:,.2f} "
            f"exceeds authorization limit ${max_authorized:,.2f}. "
            f"Management approval required."
        )

        if context is None:
            context = ViolationContext(asset_id=asset_id)

        super().__init__(
            message=message,
            safety_type="investment_limit",
            actual_value=estimated_cost,
            safety_limit=max_authorized,
            unit="USD",
            context=context,
        )


# =============================================================================
# MODEL SERVICE ERRORS
# =============================================================================


class ModelUnavailableError(InsulscanSafetyError):
    """
    ML model service is unavailable.

    Raised when the ML prediction service cannot be reached or
    returns errors. The system should fall back to deterministic
    calculations only.
    """

    def __init__(
        self,
        message: str,
        service_name: str,
        retry_after_seconds: Optional[float] = None,
        fallback_available: bool = True,
        context: Optional[ViolationContext] = None,
    ) -> None:
        """
        Initialize model unavailable error.

        Args:
            message: Human-readable error description
            service_name: Name of the unavailable service
            retry_after_seconds: Suggested retry delay
            fallback_available: Whether fallback mode is available
            context: Context information
        """
        self.service_name = service_name
        self.retry_after_seconds = retry_after_seconds
        self.fallback_available = fallback_available

        severity = ViolationSeverity.WARNING if fallback_available else ViolationSeverity.ERROR

        violations = [
            ViolationDetails(
                constraint_tag="MODEL_UNAVAILABLE",
                constraint_description=f"ML service {service_name} unavailable",
                actual_value=0.0,
                limit_value=0.0,
                unit="",
                severity=severity,
                location=f"Service: {service_name}",
                standard_reference="Internal SLA",
                recommended_action=(
                    f"Falling back to deterministic mode. "
                    f"Retry after {retry_after_seconds or 'unknown'} seconds."
                    if fallback_available else
                    f"No fallback available. Manual intervention required."
                ),
                domain=SafetyDomain.MODEL_SERVICE,
            )
        ]

        super().__init__(
            message=message,
            violations=violations,
            context=context,
            severity=severity,
            domain=SafetyDomain.MODEL_SERVICE,
        )


# =============================================================================
# MODULE EXPORTS
# =============================================================================


__all__ = [
    # Enums
    "ViolationSeverity",
    "SafetyDomain",
    # Context and details
    "ViolationContext",
    "ViolationDetails",
    # Base exception
    "InsulscanSafetyError",
    # Validation errors
    "InsulationValidationError",
    # Thermal measurement errors
    "ThermalMeasurementError",
    "SurfaceBelowAmbientError",
    # Constraint violations
    "ConstraintViolationError",
    "InsulationThicknessError",
    "HeatLossImplausibleError",
    # Safety limit errors
    "SafetyLimitExceededError",
    "BurnRiskError",
    "InvestmentLimitExceededError",
    # Model service errors
    "ModelUnavailableError",
]
