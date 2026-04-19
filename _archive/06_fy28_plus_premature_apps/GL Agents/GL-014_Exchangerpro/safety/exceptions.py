# -*- coding: utf-8 -*-
"""
GL-014 Exchangerpro - Safety Exceptions Module

Exception hierarchy for safety constraint violations and operational safety
in heat exchanger pro optimization and cleaning recommendation systems.

All exceptions follow fail-safe design principles:
- Violations trigger immediate rejection of unsafe recommendations
- Clear error messages with actionable remediation guidance
- Full provenance tracking for audit trails
- Severity classification for appropriate escalation

Safety Principles:
- Never present predictions as certainties
- Fail safe on poor data quality
- Request engineering review when outside training distribution
- No sensitive OT data export without authorization

References:
- ASME PTC 4.3: Air Heater Performance
- ASME PTC 4.4: HRSG Performance
- API 660: Shell and Tube Heat Exchangers
- IEC 61511: Safety Instrumented Systems for Process Industries
- ISO 14414: Pump System Energy Assessment

Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class ViolationSeverity(str, Enum):
    """
    Severity levels for safety violations.

    Levels follow IEC 61511 safety integrity guidelines:
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

    PHYSICAL_BOUNDS = "physical_bounds"
    ENERGY_BALANCE = "energy_balance"
    INSTRUMENTATION = "instrumentation"
    MODEL_SERVICE = "model_service"
    OPERATIONAL = "operational"
    DATA_QUALITY = "data_quality"
    AUTHORIZATION = "authorization"


@dataclass(frozen=True)
class ViolationContext:
    """
    Immutable context information for a safety violation.

    Provides complete traceability for audit and root cause analysis.

    Attributes:
        exchanger_id: Heat exchanger identifier
        timestamp: When the violation occurred
        sensor_readings: Relevant sensor data at time of violation
        operating_mode: Current operating mode (normal, startup, shutdown)
        provenance_hash: SHA-256 hash for integrity verification
    """

    exchanger_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    sensor_readings: Dict[str, float] = field(default_factory=dict)
    operating_mode: str = "normal"
    additional_context: Dict[str, Any] = field(default_factory=dict)
    provenance_hash: str = ""

    def __post_init__(self) -> None:
        """Calculate provenance hash after initialization."""
        if not self.provenance_hash:
            content = (
                f"{self.exchanger_id}|{self.timestamp.isoformat()}|"
                f"{self.operating_mode}|{str(sorted(self.sensor_readings.items()))}"
            )
            # Using object.__setattr__ because dataclass is frozen
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
        standard_reference: Applicable standard (e.g., "ASME PTC 4.4 Section 5.3")
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
    domain: SafetyDomain = SafetyDomain.PHYSICAL_BOUNDS

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


class ExchangerproSafetyError(Exception):
    """
    Base exception for all GL-014 Exchangerpro safety violations.

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
        domain: SafetyDomain = SafetyDomain.PHYSICAL_BOUNDS,
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
            lines.append(f"  Context: Exchanger={self.context.exchanger_id}, "
                        f"Mode={self.context.operating_mode}")

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
                "exchanger_id": self.context.exchanger_id if self.context else None,
                "operating_mode": self.context.operating_mode if self.context else None,
                "provenance_hash": self.context.provenance_hash if self.context else None,
            },
            "provenance_hash": self._provenance_hash,
        }


# =============================================================================
# PHYSICAL BOUNDS VIOLATIONS
# =============================================================================


class PhysicalBoundsViolation(ExchangerproSafetyError):
    """
    Violation of physical bounds constraints.

    Raised when calculated or measured values fall outside physically
    valid or safe operating ranges. This includes:
    - Effectiveness outside [0, 1] range
    - Negative flow rates
    - Temperatures outside operating envelope
    - Pressure drops exceeding limits

    These violations indicate either:
    1. Sensor/measurement errors
    2. Calculation errors
    3. Truly unsafe operating conditions

    Reference: API 660, ASME PTC 4.3/4.4
    """

    def __init__(
        self,
        message: str,
        parameter_name: str,
        actual_value: float,
        min_bound: Optional[float] = None,
        max_bound: Optional[float] = None,
        unit: str = "",
        violations: Optional[List[ViolationDetails]] = None,
        context: Optional[ViolationContext] = None,
    ) -> None:
        """
        Initialize physical bounds violation.

        Args:
            message: Human-readable error description
            parameter_name: Name of the parameter that violated bounds
            actual_value: The value that violated bounds
            min_bound: Minimum allowed value (if applicable)
            max_bound: Maximum allowed value (if applicable)
            unit: Engineering unit
            violations: Additional violation details
            context: Context information
        """
        self.parameter_name = parameter_name
        self.actual_value = actual_value
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.unit = unit

        # Create violation detail if not provided
        if not violations:
            bound_str = ""
            if min_bound is not None and max_bound is not None:
                bound_str = f"[{min_bound}, {max_bound}]"
            elif min_bound is not None:
                bound_str = f">= {min_bound}"
            elif max_bound is not None:
                bound_str = f"<= {max_bound}"

            violations = [
                ViolationDetails(
                    constraint_tag=f"PHYSICAL_BOUNDS_{parameter_name.upper()}",
                    constraint_description=f"Physical bounds constraint for {parameter_name}",
                    actual_value=actual_value,
                    limit_value=min_bound if min_bound is not None else (max_bound or 0.0),
                    unit=unit,
                    severity=ViolationSeverity.ERROR,
                    location=f"Parameter: {parameter_name}",
                    standard_reference="API 660, ASME PTC 4.3/4.4",
                    recommended_action=f"Verify sensor calibration and data quality. "
                                      f"Expected range: {bound_str} {unit}",
                    domain=SafetyDomain.PHYSICAL_BOUNDS,
                )
            ]

        super().__init__(
            message=message,
            violations=violations,
            context=context,
            severity=ViolationSeverity.ERROR,
            domain=SafetyDomain.PHYSICAL_BOUNDS,
        )


class EffectivenessOutOfBoundsError(PhysicalBoundsViolation):
    """
    Heat exchanger effectiveness outside valid [0, 1] range.

    Effectiveness (epsilon) must be between 0 and 1:
    - epsilon = 0: No heat transfer
    - epsilon = 1: Maximum theoretical heat transfer (NTU -> infinity)
    - epsilon < 0 or > 1: Physically impossible, indicates error

    Common causes:
    - Sensor drift or failure
    - Incorrect flow rate measurements
    - Temperature sensor placement errors
    - Calculation bugs
    """

    def __init__(
        self,
        effectiveness: float,
        exchanger_id: str,
        context: Optional[ViolationContext] = None,
    ) -> None:
        """
        Initialize effectiveness bounds error.

        Args:
            effectiveness: Calculated effectiveness value
            exchanger_id: Heat exchanger identifier
            context: Context information
        """
        message = (
            f"Heat exchanger {exchanger_id} effectiveness {effectiveness:.4f} "
            f"is outside valid range [0, 1]. This indicates measurement or "
            f"calculation error. Verify sensor calibration and data quality."
        )

        if context is None:
            context = ViolationContext(exchanger_id=exchanger_id)

        super().__init__(
            message=message,
            parameter_name="effectiveness",
            actual_value=effectiveness,
            min_bound=0.0,
            max_bound=1.0,
            unit="dimensionless",
            context=context,
        )


class NegativeFlowError(PhysicalBoundsViolation):
    """
    Negative flow rate detected.

    Flow rates must be non-negative. Negative values indicate:
    - Sensor failure or misconfiguration
    - Reverse flow (may be valid in some systems, requires verification)
    - Data transmission errors
    """

    def __init__(
        self,
        flow_rate: float,
        stream_id: str,
        flow_type: str = "mass",
        context: Optional[ViolationContext] = None,
    ) -> None:
        """
        Initialize negative flow error.

        Args:
            flow_rate: Measured/calculated flow rate (negative)
            stream_id: Stream identifier
            flow_type: Type of flow ("mass" or "volumetric")
            context: Context information
        """
        unit = "kg/s" if flow_type == "mass" else "m3/s"
        message = (
            f"Stream {stream_id} has negative {flow_type} flow rate: "
            f"{flow_rate:.4f} {unit}. Verify flow sensor calibration."
        )

        super().__init__(
            message=message,
            parameter_name=f"{flow_type}_flow_rate",
            actual_value=flow_rate,
            min_bound=0.0,
            max_bound=None,
            unit=unit,
            context=context,
        )


class TemperatureOutOfRangeError(PhysicalBoundsViolation):
    """
    Temperature outside allowed operating range.

    Temperature constraints protect against:
    - Coking at high temperatures
    - Acid dew point corrosion at low temperatures
    - Metallurgical limits
    - Process quality requirements
    """

    def __init__(
        self,
        temperature: float,
        location: str,
        min_temp: Optional[float] = None,
        max_temp: Optional[float] = None,
        context: Optional[ViolationContext] = None,
    ) -> None:
        """
        Initialize temperature range error.

        Args:
            temperature: Measured/calculated temperature
            location: Where the temperature was measured
            min_temp: Minimum allowed temperature
            max_temp: Maximum allowed temperature
            context: Context information
        """
        if min_temp is not None and temperature < min_temp:
            message = (
                f"Temperature at {location} is {temperature:.1f}C, "
                f"below minimum {min_temp:.1f}C. Risk of acid dew point corrosion."
            )
        elif max_temp is not None and temperature > max_temp:
            message = (
                f"Temperature at {location} is {temperature:.1f}C, "
                f"above maximum {max_temp:.1f}C. Risk of coking or metallurgy failure."
            )
        else:
            message = f"Temperature at {location} is {temperature:.1f}C, outside valid range."

        super().__init__(
            message=message,
            parameter_name="temperature",
            actual_value=temperature,
            min_bound=min_temp,
            max_bound=max_temp,
            unit="C",
            context=context,
        )


class PressureDropExceededError(PhysicalBoundsViolation):
    """
    Pressure drop exceeds allowable limit.

    Excessive pressure drop indicates:
    - Severe fouling
    - Flow restriction
    - Pump/compressor capacity issues

    Reference: API 660 Section 6.3, ISO 14414
    """

    def __init__(
        self,
        pressure_drop: float,
        max_allowed: float,
        side: str,
        exchanger_id: str,
        context: Optional[ViolationContext] = None,
    ) -> None:
        """
        Initialize pressure drop exceeded error.

        Args:
            pressure_drop: Measured pressure drop
            max_allowed: Maximum allowed pressure drop
            side: "hot" or "cold" side
            exchanger_id: Heat exchanger identifier
            context: Context information
        """
        message = (
            f"Exchanger {exchanger_id} {side} side pressure drop "
            f"{pressure_drop:.2f} kPa exceeds limit {max_allowed:.2f} kPa. "
            f"Possible severe fouling or flow restriction."
        )

        if context is None:
            context = ViolationContext(exchanger_id=exchanger_id)

        violations = [
            ViolationDetails(
                constraint_tag="MAX_PRESSURE_DROP",
                constraint_description="Maximum pressure drop limit",
                actual_value=pressure_drop,
                limit_value=max_allowed,
                unit="kPa",
                severity=ViolationSeverity.ERROR,
                location=f"{exchanger_id} {side} side",
                standard_reference="API 660 Section 6.3, ISO 14414",
                recommended_action="Schedule cleaning or inspect for flow restrictions.",
                domain=SafetyDomain.PHYSICAL_BOUNDS,
            )
        ]

        super().__init__(
            message=message,
            parameter_name="pressure_drop",
            actual_value=pressure_drop,
            min_bound=None,
            max_bound=max_allowed,
            unit="kPa",
            violations=violations,
            context=context,
        )


# =============================================================================
# ENERGY BALANCE VIOLATIONS
# =============================================================================


class EnergyBalanceError(ExchangerproSafetyError):
    """
    Energy balance consistency check failure.

    Raised when the energy transferred between hot and cold streams
    does not balance within acceptable tolerance. This indicates:
    - Measurement errors
    - Heat losses not accounted for
    - Incorrect flow rate or temperature data
    - Phase change effects not properly modeled

    Energy balance equation:
    Q_hot = m_hot * Cp_hot * (T_hot_in - T_hot_out)
    Q_cold = m_cold * Cp_cold * (T_cold_out - T_cold_in)
    |Q_hot - Q_cold| / max(Q_hot, Q_cold) should be < tolerance

    Reference: ASME PTC 4.4 Section 5.2
    """

    def __init__(
        self,
        message: str,
        q_hot: float,
        q_cold: float,
        tolerance: float,
        exchanger_id: str,
        context: Optional[ViolationContext] = None,
    ) -> None:
        """
        Initialize energy balance error.

        Args:
            message: Human-readable error description
            q_hot: Heat duty calculated from hot side
            q_cold: Heat duty calculated from cold side
            tolerance: Allowed relative difference
            exchanger_id: Heat exchanger identifier
            context: Context information
        """
        self.q_hot = q_hot
        self.q_cold = q_cold
        self.tolerance = tolerance
        self.exchanger_id = exchanger_id

        # Calculate actual imbalance
        q_max = max(abs(q_hot), abs(q_cold))
        if q_max > 0:
            imbalance = abs(q_hot - q_cold) / q_max
        else:
            imbalance = 0.0

        self.imbalance = imbalance

        if context is None:
            context = ViolationContext(exchanger_id=exchanger_id)

        violations = [
            ViolationDetails(
                constraint_tag="ENERGY_BALANCE",
                constraint_description="Energy balance between hot and cold sides",
                actual_value=imbalance,
                limit_value=tolerance,
                unit="dimensionless",
                severity=ViolationSeverity.ERROR,
                location=f"Exchanger {exchanger_id}",
                standard_reference="ASME PTC 4.4 Section 5.2",
                recommended_action=(
                    f"Verify temperature and flow sensors. "
                    f"Q_hot={q_hot:.2f} kW, Q_cold={q_cold:.2f} kW, "
                    f"Imbalance={imbalance:.1%} > tolerance={tolerance:.1%}"
                ),
                domain=SafetyDomain.ENERGY_BALANCE,
            )
        ]

        super().__init__(
            message=message,
            violations=violations,
            context=context,
            severity=ViolationSeverity.ERROR,
            domain=SafetyDomain.ENERGY_BALANCE,
        )


# =============================================================================
# INSTRUMENTATION FAULTS
# =============================================================================


class InstrumentationFault(ExchangerproSafetyError):
    """
    Instrumentation system fault detected.

    Raised when sensor or measurement system issues are detected:
    - Sensor stuck at constant value
    - Sensor reading out of calibration range
    - Missing or stale data
    - Communication failures with OT systems

    Safety principle: Fail safe on poor data quality.

    Reference: IEC 61511, ISA-84
    """

    def __init__(
        self,
        message: str,
        sensor_id: str,
        fault_type: str,
        last_valid_reading: Optional[float] = None,
        last_valid_time: Optional[datetime] = None,
        context: Optional[ViolationContext] = None,
    ) -> None:
        """
        Initialize instrumentation fault.

        Args:
            message: Human-readable error description
            sensor_id: Sensor identifier
            fault_type: Type of fault detected
            last_valid_reading: Last known good value
            last_valid_time: When last valid reading was received
            context: Context information
        """
        self.sensor_id = sensor_id
        self.fault_type = fault_type
        self.last_valid_reading = last_valid_reading
        self.last_valid_time = last_valid_time

        violations = [
            ViolationDetails(
                constraint_tag=f"INSTRUMENTATION_{fault_type.upper()}",
                constraint_description=f"Instrumentation fault: {fault_type}",
                actual_value=last_valid_reading or 0.0,
                limit_value=0.0,
                unit="",
                severity=ViolationSeverity.WARNING,
                location=f"Sensor {sensor_id}",
                standard_reference="IEC 61511, ISA-84",
                recommended_action=(
                    f"Check sensor {sensor_id} for {fault_type}. "
                    f"Recommendations will use conservative assumptions."
                ),
                domain=SafetyDomain.INSTRUMENTATION,
            )
        ]

        super().__init__(
            message=message,
            violations=violations,
            context=context,
            severity=ViolationSeverity.WARNING,
            domain=SafetyDomain.INSTRUMENTATION,
        )


class SensorStuckFault(InstrumentationFault):
    """Sensor value stuck at constant reading."""

    def __init__(
        self,
        sensor_id: str,
        stuck_value: float,
        stuck_duration_seconds: float,
        context: Optional[ViolationContext] = None,
    ) -> None:
        """
        Initialize sensor stuck fault.

        Args:
            sensor_id: Sensor identifier
            stuck_value: The value sensor is stuck at
            stuck_duration_seconds: How long sensor has been stuck
            context: Context information
        """
        message = (
            f"Sensor {sensor_id} appears stuck at {stuck_value:.2f} "
            f"for {stuck_duration_seconds:.0f} seconds. "
            f"Verify sensor operation."
        )
        super().__init__(
            message=message,
            sensor_id=sensor_id,
            fault_type="stuck",
            last_valid_reading=stuck_value,
            context=context,
        )


class SensorStaleDataFault(InstrumentationFault):
    """Sensor data is stale (no recent updates)."""

    def __init__(
        self,
        sensor_id: str,
        stale_duration_seconds: float,
        last_value: Optional[float] = None,
        context: Optional[ViolationContext] = None,
    ) -> None:
        """
        Initialize stale data fault.

        Args:
            sensor_id: Sensor identifier
            stale_duration_seconds: How long since last update
            last_value: Last known value
            context: Context information
        """
        message = (
            f"Sensor {sensor_id} has not updated in {stale_duration_seconds:.0f} seconds. "
            f"Check OT connectivity."
        )
        super().__init__(
            message=message,
            sensor_id=sensor_id,
            fault_type="stale_data",
            last_valid_reading=last_value,
            context=context,
        )


# =============================================================================
# MODEL SERVICE ERRORS
# =============================================================================


class ModelUnavailableError(ExchangerproSafetyError):
    """
    ML model service is unavailable.

    Raised when the ML prediction service cannot be reached or
    returns errors. The system should fall back to deterministic
    calculations only.

    This triggers graceful degradation:
    - Use physics-based models only
    - Apply conservative safety margins
    - Log for monitoring
    """

    def __init__(
        self,
        message: str,
        service_name: str,
        error_code: Optional[str] = None,
        retry_after_seconds: Optional[float] = None,
        fallback_available: bool = True,
        context: Optional[ViolationContext] = None,
    ) -> None:
        """
        Initialize model unavailable error.

        Args:
            message: Human-readable error description
            service_name: Name of the unavailable service
            error_code: Error code from the service
            retry_after_seconds: Suggested retry delay
            fallback_available: Whether fallback mode is available
            context: Context information
        """
        self.service_name = service_name
        self.error_code = error_code
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


class ModelPredictionError(ExchangerproSafetyError):
    """
    ML model returned invalid or uncertain prediction.

    Raised when:
    - Model returns NaN or infinite values
    - Prediction uncertainty exceeds threshold
    - Input is outside training distribution

    Safety principle: Never present predictions as certainties.
    """

    def __init__(
        self,
        message: str,
        model_name: str,
        prediction_value: Optional[float] = None,
        uncertainty: Optional[float] = None,
        is_out_of_distribution: bool = False,
        context: Optional[ViolationContext] = None,
    ) -> None:
        """
        Initialize model prediction error.

        Args:
            message: Human-readable error description
            model_name: Name of the model
            prediction_value: The prediction value (if any)
            uncertainty: Uncertainty estimate
            is_out_of_distribution: Whether input is OOD
            context: Context information
        """
        self.model_name = model_name
        self.prediction_value = prediction_value
        self.uncertainty = uncertainty
        self.is_out_of_distribution = is_out_of_distribution

        severity = ViolationSeverity.WARNING
        action = "Using conservative estimate."

        if is_out_of_distribution:
            severity = ViolationSeverity.ERROR
            action = "Request engineering review. Input outside training distribution."

        violations = [
            ViolationDetails(
                constraint_tag="MODEL_PREDICTION_INVALID",
                constraint_description=f"Model {model_name} prediction issue",
                actual_value=uncertainty or 0.0,
                limit_value=0.0,
                unit="",
                severity=severity,
                location=f"Model: {model_name}",
                standard_reference="Internal ML guidelines",
                recommended_action=action,
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
# OPERATIONAL SAFETY VIOLATIONS
# =============================================================================


class OperationalSafetyViolation(ExchangerproSafetyError):
    """
    Operational safety rule violation.

    Base class for violations of operational safety guardrails:
    - SIS bypass attempts
    - Control loop manipulation
    - Unauthorized autonomous actions
    """

    def __init__(
        self,
        message: str,
        rule_name: str,
        violations: Optional[List[ViolationDetails]] = None,
        context: Optional[ViolationContext] = None,
    ) -> None:
        """
        Initialize operational safety violation.

        Args:
            message: Human-readable error description
            rule_name: Name of the rule that was violated
            violations: Violation details
            context: Context information
        """
        self.rule_name = rule_name

        super().__init__(
            message=message,
            violations=violations,
            context=context,
            severity=ViolationSeverity.CRITICAL,
            domain=SafetyDomain.OPERATIONAL,
        )


class SISBypassAttemptError(OperationalSafetyViolation):
    """
    Attempt to bypass Safety Instrumented System detected.

    CRITICAL: Recommendations must NEVER bypass SIS.
    This is a hard safety constraint with no override.
    """

    def __init__(
        self,
        sis_function: str,
        attempted_action: str,
        context: Optional[ViolationContext] = None,
    ) -> None:
        """
        Initialize SIS bypass attempt error.

        Args:
            sis_function: The SIS function that would be bypassed
            attempted_action: The action that was attempted
            context: Context information
        """
        message = (
            f"BLOCKED: Attempted action '{attempted_action}' would bypass "
            f"SIS function '{sis_function}'. SIS bypass is NEVER allowed."
        )

        violations = [
            ViolationDetails(
                constraint_tag="SIS_BYPASS_BLOCKED",
                constraint_description="Safety Instrumented System bypass prevention",
                actual_value=1.0,
                limit_value=0.0,
                unit="attempt",
                severity=ViolationSeverity.CRITICAL,
                location=f"SIS: {sis_function}",
                standard_reference="IEC 61511",
                recommended_action="Action blocked. No override available.",
                domain=SafetyDomain.OPERATIONAL,
            )
        ]

        super().__init__(
            message=message,
            rule_name="NO_SIS_BYPASS",
            violations=violations,
            context=context,
        )


class ControlLoopManipulationError(OperationalSafetyViolation):
    """
    Attempt to directly manipulate control loop detected.

    CRITICAL: System provides recommendations only, not direct control.
    """

    def __init__(
        self,
        control_loop: str,
        attempted_setpoint: Optional[float] = None,
        context: Optional[ViolationContext] = None,
    ) -> None:
        """
        Initialize control loop manipulation error.

        Args:
            control_loop: The control loop that would be manipulated
            attempted_setpoint: The setpoint that was attempted
            context: Context information
        """
        message = (
            f"BLOCKED: Direct manipulation of control loop '{control_loop}' "
            f"is not allowed. System provides recommendations only."
        )

        violations = [
            ViolationDetails(
                constraint_tag="CONTROL_LOOP_BLOCKED",
                constraint_description="Direct control loop manipulation prevention",
                actual_value=attempted_setpoint or 0.0,
                limit_value=0.0,
                unit="",
                severity=ViolationSeverity.CRITICAL,
                location=f"Control loop: {control_loop}",
                standard_reference="Internal safety policy",
                recommended_action="Use recommendation interface only.",
                domain=SafetyDomain.OPERATIONAL,
            )
        ]

        super().__init__(
            message=message,
            rule_name="NO_DIRECT_CONTROL",
            violations=violations,
            context=context,
        )


class UnauthorizedDataExportError(OperationalSafetyViolation):
    """
    Attempt to export sensitive OT data without authorization.

    Safety principle: No sensitive OT data export without authorization.
    """

    def __init__(
        self,
        data_type: str,
        destination: str,
        context: Optional[ViolationContext] = None,
    ) -> None:
        """
        Initialize unauthorized data export error.

        Args:
            data_type: Type of data being exported
            destination: Export destination
            context: Context information
        """
        message = (
            f"BLOCKED: Export of '{data_type}' to '{destination}' requires "
            f"authorization. OT data export is restricted."
        )

        violations = [
            ViolationDetails(
                constraint_tag="DATA_EXPORT_BLOCKED",
                constraint_description="Sensitive OT data export authorization required",
                actual_value=0.0,
                limit_value=0.0,
                unit="",
                severity=ViolationSeverity.ERROR,
                location=f"Destination: {destination}",
                standard_reference="Internal data governance policy",
                recommended_action="Obtain authorization before data export.",
                domain=SafetyDomain.AUTHORIZATION,
            )
        ]

        super().__init__(
            message=message,
            rule_name="OT_DATA_EXPORT_RESTRICTED",
            violations=violations,
            context=context,
        )


# =============================================================================
# DATA QUALITY ERRORS
# =============================================================================


class DataQualityError(ExchangerproSafetyError):
    """
    Data quality issue detected.

    Safety principle: Fail safe on poor data quality.
    """

    def __init__(
        self,
        message: str,
        quality_score: float,
        minimum_required: float,
        affected_parameters: Optional[List[str]] = None,
        context: Optional[ViolationContext] = None,
    ) -> None:
        """
        Initialize data quality error.

        Args:
            message: Human-readable error description
            quality_score: Calculated data quality score
            minimum_required: Minimum required quality score
            affected_parameters: List of affected parameters
            context: Context information
        """
        self.quality_score = quality_score
        self.minimum_required = minimum_required
        self.affected_parameters = affected_parameters or []

        violations = [
            ViolationDetails(
                constraint_tag="DATA_QUALITY",
                constraint_description="Minimum data quality requirement",
                actual_value=quality_score,
                limit_value=minimum_required,
                unit="score",
                severity=ViolationSeverity.WARNING,
                location=f"Parameters: {', '.join(affected_parameters) if affected_parameters else 'multiple'}",
                standard_reference="Internal data quality standards",
                recommended_action="Using conservative estimates due to data quality issues.",
                domain=SafetyDomain.DATA_QUALITY,
            )
        ]

        super().__init__(
            message=message,
            violations=violations,
            context=context,
            severity=ViolationSeverity.WARNING,
            domain=SafetyDomain.DATA_QUALITY,
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
    "ExchangerproSafetyError",
    # Physical bounds
    "PhysicalBoundsViolation",
    "EffectivenessOutOfBoundsError",
    "NegativeFlowError",
    "TemperatureOutOfRangeError",
    "PressureDropExceededError",
    # Energy balance
    "EnergyBalanceError",
    # Instrumentation
    "InstrumentationFault",
    "SensorStuckFault",
    "SensorStaleDataFault",
    # Model service
    "ModelUnavailableError",
    "ModelPredictionError",
    # Operational safety
    "OperationalSafetyViolation",
    "SISBypassAttemptError",
    "ControlLoopManipulationError",
    "UnauthorizedDataExportError",
    # Data quality
    "DataQualityError",
]
