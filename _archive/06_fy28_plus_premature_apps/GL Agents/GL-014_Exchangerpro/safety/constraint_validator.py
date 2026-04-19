# -*- coding: utf-8 -*-
"""
GL-014 Exchangerpro - Constraint Validator

Production-grade physical constraint validation for heat exchanger
optimization and cleaning recommendations. Enforces:

1. Physical Bounds Validation:
   - Effectiveness in [0, 1]
   - Flow rates >= 0
   - Temperatures in operating range
   - Pressure drops within limits

2. Energy Balance Consistency:
   - Q_hot ~= Q_cold within tolerance
   - Heat capacity rate ratios valid

3. Delta-P Limit Enforcement:
   - Per-side pressure drop limits
   - Phase-specific limits (liquid vs gas)

4. Outlet Temperature Constraints:
   - Acid dew point protection
   - Process quality requirements
   - Metallurgical limits

Safety Principles:
- Fail-closed on any violation
- All checks use deterministic arithmetic
- SHA-256 provenance for all validations
- No LLM inference for safety decisions

Standards Compliance:
- ASME PTC 4.3: Air Heater Performance
- ASME PTC 4.4: HRSG Performance
- API 660: Shell and Tube Heat Exchangers
- ISO 14414: Pump System Energy Assessment
- TEMA: Tubular Exchanger Manufacturers Association

Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator

from .exceptions import (
    ExchangerproSafetyError,
    PhysicalBoundsViolation,
    EffectivenessOutOfBoundsError,
    NegativeFlowError,
    TemperatureOutOfRangeError,
    PressureDropExceededError,
    EnergyBalanceError,
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

    EFFECTIVENESS_BOUNDS = "effectiveness_bounds"
    FLOW_RATE_NON_NEGATIVE = "flow_rate_non_negative"
    TEMPERATURE_RANGE = "temperature_range"
    PRESSURE_DROP_LIMIT = "pressure_drop_limit"
    ENERGY_BALANCE = "energy_balance"
    APPROACH_TEMPERATURE = "approach_temperature"
    ACID_DEW_POINT = "acid_dew_point"
    FOULING_RESISTANCE = "fouling_resistance"
    NTU_VALIDITY = "ntu_validity"


class ConstraintSeverity(str, Enum):
    """Severity level for constraint violations."""

    INFO = "info"  # Informational, no action required
    WARNING = "warning"  # Near limit, log and continue
    ERROR = "error"  # Violation, reject with option to override
    CRITICAL = "critical"  # Hard violation, no override allowed


class FluidPhase(str, Enum):
    """Fluid phase classification."""

    LIQUID = "liquid"
    GAS = "gas"
    TWO_PHASE = "two_phase"
    SUPERCRITICAL = "supercritical"


# =============================================================================
# CONFIGURATION
# =============================================================================


class ConstraintLimits(BaseModel):
    """
    Physical constraint limits configuration.

    Default values based on industry standards and best practices.
    All values can be customized for specific applications.
    """

    # Effectiveness bounds
    effectiveness_min: float = Field(
        default=0.0,
        ge=0.0,
        le=0.5,
        description="Minimum valid effectiveness"
    )
    effectiveness_max: float = Field(
        default=1.0,
        ge=0.5,
        le=1.0,
        description="Maximum valid effectiveness"
    )

    # Temperature bounds (Celsius)
    temperature_min_C: float = Field(
        default=-50.0,
        ge=-273.15,
        le=100.0,
        description="Minimum valid temperature"
    )
    temperature_max_C: float = Field(
        default=600.0,
        ge=100.0,
        le=1500.0,
        description="Maximum valid temperature"
    )
    acid_dew_point_C: float = Field(
        default=120.0,
        ge=80.0,
        le=180.0,
        description="Acid dew point temperature"
    )
    min_approach_temp_C: float = Field(
        default=5.0,
        ge=1.0,
        le=30.0,
        description="Minimum approach temperature"
    )

    # Pressure drop limits (kPa)
    max_pressure_drop_liquid_kPa: float = Field(
        default=50.0,
        ge=10.0,
        le=200.0,
        description="Maximum pressure drop for liquids"
    )
    max_pressure_drop_gas_kPa: float = Field(
        default=5.0,
        ge=1.0,
        le=20.0,
        description="Maximum pressure drop for gases"
    )

    # Energy balance tolerance
    energy_balance_tolerance: float = Field(
        default=0.05,
        ge=0.01,
        le=0.20,
        description="Relative tolerance for energy balance (5%)"
    )

    # Fouling limits
    max_fouling_resistance_m2K_W: float = Field(
        default=0.002,
        ge=0.0001,
        le=0.01,
        description="Maximum allowed fouling resistance"
    )

    # NTU limits
    max_ntu: float = Field(
        default=10.0,
        ge=1.0,
        le=50.0,
        description="Maximum practical NTU"
    )

    @field_validator("effectiveness_max")
    @classmethod
    def validate_effectiveness_max(cls, v: float, info) -> float:
        """Ensure max > min."""
        # Using info.data for Pydantic v2 compatibility
        if hasattr(info, 'data') and 'effectiveness_min' in info.data:
            if v <= info.data['effectiveness_min']:
                raise ValueError("effectiveness_max must be > effectiveness_min")
        return v


class ConstraintValidatorConfig(BaseModel):
    """
    Configuration for the constraint validator.

    Attributes:
        limits: Physical constraint limits
        fail_closed: If True, raise exception on violations
        log_warnings: If True, log warning-level issues
        calculate_penalties: If True, compute penalty costs
        strict_energy_balance: If True, treat energy imbalance as error
    """

    limits: ConstraintLimits = Field(
        default_factory=ConstraintLimits,
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
    calculate_penalties: bool = Field(
        default=True,
        description="Calculate penalty costs for near-violations"
    )
    strict_energy_balance: bool = Field(
        default=True,
        description="Treat energy imbalance as error"
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
        exchanger_id: Heat exchanger identifier
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
    exchanger_id: str
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
                f"{self.constraint_type.value}|{self.exchanger_id}|"
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
            "exchanger_id": self.exchanger_id,
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


class ConstraintValidationSummary(BaseModel):
    """
    Summary of all constraint validation results.

    Attributes:
        exchanger_id: Heat exchanger identifier
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

    exchanger_id: str = Field(..., description="Heat exchanger identifier")
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
                f"{self.exchanger_id}|{self.timestamp.isoformat()}|"
                f"{self.is_valid}|{self.violations_count}|{self.warnings_count}"
            )
            self.result_hash = hashlib.sha256(content.encode()).hexdigest()


@dataclass
class ExchangerData:
    """
    Input data for heat exchanger constraint validation.

    Attributes:
        exchanger_id: Heat exchanger identifier
        hot_inlet_T_C: Hot stream inlet temperature
        hot_outlet_T_C: Hot stream outlet temperature
        cold_inlet_T_C: Cold stream inlet temperature
        cold_outlet_T_C: Cold stream outlet temperature
        hot_flow_rate_kg_s: Hot stream mass flow rate
        cold_flow_rate_kg_s: Cold stream mass flow rate
        hot_cp_kJ_kgK: Hot stream specific heat
        cold_cp_kJ_kgK: Cold stream specific heat
        hot_side_dp_kPa: Hot side pressure drop
        cold_side_dp_kPa: Cold side pressure drop
        hot_phase: Hot stream phase
        cold_phase: Cold stream phase
        effectiveness: Heat exchanger effectiveness
        ntu: Number of transfer units
        fouling_resistance_m2K_W: Fouling resistance
        is_flue_gas: Whether hot stream is flue gas
    """

    exchanger_id: str
    hot_inlet_T_C: float
    hot_outlet_T_C: float
    cold_inlet_T_C: float
    cold_outlet_T_C: float
    hot_flow_rate_kg_s: float = 0.0
    cold_flow_rate_kg_s: float = 0.0
    hot_cp_kJ_kgK: float = 4.18  # Water default
    cold_cp_kJ_kgK: float = 4.18  # Water default
    hot_side_dp_kPa: float = 0.0
    cold_side_dp_kPa: float = 0.0
    hot_phase: FluidPhase = FluidPhase.LIQUID
    cold_phase: FluidPhase = FluidPhase.LIQUID
    effectiveness: Optional[float] = None
    ntu: Optional[float] = None
    fouling_resistance_m2K_W: Optional[float] = None
    is_flue_gas: bool = False


# =============================================================================
# CONSTRAINT VALIDATOR
# =============================================================================


class ConstraintValidator:
    """
    Production-grade physical constraint validator for heat exchangers.

    Validates all physical bounds, energy balance, and operational
    constraints with deterministic checks and full provenance tracking.

    Safety Principles:
    - All checks use deterministic arithmetic (no LLM)
    - Fail-closed on critical violations
    - SHA-256 provenance for audit trail
    - Standards-compliant validation

    Example:
        >>> config = ConstraintValidatorConfig()
        >>> validator = ConstraintValidator(config)
        >>>
        >>> data = ExchangerData(
        ...     exchanger_id="HX-101",
        ...     hot_inlet_T_C=150.0,
        ...     hot_outlet_T_C=80.0,
        ...     cold_inlet_T_C=20.0,
        ...     cold_outlet_T_C=70.0,
        ...     effectiveness=0.65,
        ... )
        >>>
        >>> summary = validator.validate(data)
        >>> if not summary.is_valid:
        ...     raise SafetyError(summary.get_rejection_reasons())

    Author: GL-BackendDeveloper
    Version: 1.0.0
    """

    VERSION = "1.0.0"

    def __init__(self, config: Optional[ConstraintValidatorConfig] = None) -> None:
        """
        Initialize constraint validator.

        Args:
            config: Validator configuration
        """
        self.config = config or ConstraintValidatorConfig()
        self._config_hash = self._compute_config_hash()

        logger.info(
            f"ConstraintValidator initialized: "
            f"fail_closed={self.config.fail_closed}, "
            f"strict_energy_balance={self.config.strict_energy_balance}"
        )

    def _compute_config_hash(self) -> str:
        """Compute SHA-256 hash of configuration."""
        config_dict = self.config.model_dump()
        json_str = json.dumps(config_dict, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def _compute_data_hash(self, data: ExchangerData) -> str:
        """Compute SHA-256 hash of input data."""
        data_dict = {
            "exchanger_id": data.exchanger_id,
            "hot_inlet_T_C": data.hot_inlet_T_C,
            "hot_outlet_T_C": data.hot_outlet_T_C,
            "cold_inlet_T_C": data.cold_inlet_T_C,
            "cold_outlet_T_C": data.cold_outlet_T_C,
            "hot_flow_rate_kg_s": data.hot_flow_rate_kg_s,
            "cold_flow_rate_kg_s": data.cold_flow_rate_kg_s,
        }
        json_str = json.dumps(data_dict, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()

    # =========================================================================
    # MAIN VALIDATION METHOD
    # =========================================================================

    def validate(self, data: ExchangerData) -> ConstraintValidationSummary:
        """
        Validate all physical constraints for a heat exchanger.

        Performs comprehensive validation:
        1. Effectiveness bounds [0, 1]
        2. Flow rates non-negative
        3. Temperature ranges
        4. Pressure drop limits
        5. Energy balance consistency
        6. Approach temperature
        7. Acid dew point (for flue gas)

        Args:
            data: Heat exchanger data to validate

        Returns:
            ConstraintValidationSummary with all results

        Raises:
            PhysicalBoundsViolation: If fail_closed and critical violation
            EnergyBalanceError: If fail_closed and energy imbalance
        """
        start_time = datetime.now(timezone.utc)
        results: List[ConstraintCheckResult] = []

        logger.debug(f"Validating constraints for {data.exchanger_id}")

        # 1. Effectiveness bounds
        if data.effectiveness is not None:
            results.append(self._check_effectiveness_bounds(data))

        # 2. Flow rate non-negative
        results.extend(self._check_flow_rates(data))

        # 3. Temperature range
        results.extend(self._check_temperature_range(data))

        # 4. Pressure drop limits
        results.extend(self._check_pressure_drops(data))

        # 5. Energy balance
        if data.hot_flow_rate_kg_s > 0 and data.cold_flow_rate_kg_s > 0:
            results.append(self._check_energy_balance(data))

        # 6. Approach temperature
        results.extend(self._check_approach_temperature(data))

        # 7. Acid dew point
        if data.is_flue_gas:
            results.append(self._check_acid_dew_point(data))

        # 8. Fouling resistance
        if data.fouling_resistance_m2K_W is not None:
            results.append(self._check_fouling_resistance(data))

        # 9. NTU validity
        if data.ntu is not None:
            results.append(self._check_ntu_validity(data))

        # Build summary
        summary = self._build_summary(data, results)

        # Log results
        if summary.violations_count > 0:
            logger.warning(
                f"Constraint validation for {data.exchanger_id}: "
                f"{summary.violations_count} violations, "
                f"{summary.warnings_count} warnings"
            )
        else:
            logger.debug(
                f"Constraint validation for {data.exchanger_id}: "
                f"PASSED ({summary.total_checks} checks)"
            )

        # Fail-closed behavior
        if self.config.fail_closed and not summary.is_valid:
            self._raise_violation_error(data, summary)

        return summary

    def validate_batch(
        self,
        exchangers: List[ExchangerData],
    ) -> Dict[str, ConstraintValidationSummary]:
        """
        Validate multiple heat exchangers.

        Args:
            exchangers: List of exchanger data to validate

        Returns:
            Dict mapping exchanger_id to validation summary
        """
        results = {}
        for data in exchangers:
            # Catch exceptions to continue processing all exchangers
            try:
                results[data.exchanger_id] = self.validate(data)
            except ExchangerproSafetyError as e:
                # Create failed summary
                results[data.exchanger_id] = ConstraintValidationSummary(
                    exchanger_id=data.exchanger_id,
                    is_valid=False,
                    violations_count=len(e.violations),
                    critical_violations=[v.to_dict() for v in e.violations],
                )
        return results

    # =========================================================================
    # INDIVIDUAL CONSTRAINT CHECKS
    # =========================================================================

    def _check_effectiveness_bounds(
        self,
        data: ExchangerData,
    ) -> ConstraintCheckResult:
        """Check effectiveness is in [0, 1]."""
        limits = self.config.limits
        epsilon = data.effectiveness

        is_valid = limits.effectiveness_min <= epsilon <= limits.effectiveness_max
        margin = min(
            epsilon - limits.effectiveness_min,
            limits.effectiveness_max - epsilon
        )
        margin_percent = margin * 100

        if epsilon < limits.effectiveness_min:
            severity = ConstraintSeverity.ERROR
            message = f"Effectiveness {epsilon:.4f} below minimum {limits.effectiveness_min}"
            action = "Check sensor calibration and flow measurements"
        elif epsilon > limits.effectiveness_max:
            severity = ConstraintSeverity.CRITICAL
            message = f"Effectiveness {epsilon:.4f} exceeds maximum {limits.effectiveness_max}"
            action = "This is physically impossible - verify all temperature/flow sensors"
        elif epsilon > 0.95:
            severity = ConstraintSeverity.WARNING
            message = f"Effectiveness {epsilon:.4f} is very high, verify measurements"
            action = "Double-check temperature sensor readings"
            is_valid = True
        else:
            severity = ConstraintSeverity.INFO
            message = f"Effectiveness {epsilon:.4f} is valid"
            action = "No action required"

        return ConstraintCheckResult(
            constraint_type=ConstraintType.EFFECTIVENESS_BOUNDS,
            exchanger_id=data.exchanger_id,
            parameter_name="effectiveness",
            actual_value=epsilon,
            limit_value=limits.effectiveness_max,
            unit="dimensionless",
            is_valid=is_valid,
            severity=severity,
            margin=margin,
            margin_percent=margin_percent,
            message=message,
            standard_reference="Heat Transfer Theory",
            recommended_action=action,
        )

    def _check_flow_rates(self, data: ExchangerData) -> List[ConstraintCheckResult]:
        """Check flow rates are non-negative."""
        results = []

        for side, flow_rate in [
            ("hot", data.hot_flow_rate_kg_s),
            ("cold", data.cold_flow_rate_kg_s),
        ]:
            is_valid = flow_rate >= 0
            margin = flow_rate
            margin_percent = 100.0 if flow_rate > 0 else 0.0

            if flow_rate < 0:
                severity = ConstraintSeverity.ERROR
                message = f"{side.capitalize()} flow rate {flow_rate:.4f} kg/s is negative"
                action = "Check flow sensor calibration and direction"
            elif flow_rate == 0:
                severity = ConstraintSeverity.WARNING
                message = f"{side.capitalize()} flow rate is zero"
                action = "Verify flow is present or exchanger is bypassed"
            else:
                severity = ConstraintSeverity.INFO
                message = f"{side.capitalize()} flow rate {flow_rate:.4f} kg/s is valid"
                action = "No action required"

            results.append(ConstraintCheckResult(
                constraint_type=ConstraintType.FLOW_RATE_NON_NEGATIVE,
                exchanger_id=data.exchanger_id,
                parameter_name=f"{side}_flow_rate",
                actual_value=flow_rate,
                limit_value=0.0,
                unit="kg/s",
                is_valid=is_valid,
                severity=severity,
                margin=margin,
                margin_percent=margin_percent,
                message=message,
                standard_reference="Mass Conservation",
                recommended_action=action,
            ))

        return results

    def _check_temperature_range(
        self,
        data: ExchangerData,
    ) -> List[ConstraintCheckResult]:
        """Check all temperatures are in valid range."""
        results = []
        limits = self.config.limits

        temps = [
            ("hot_inlet", data.hot_inlet_T_C),
            ("hot_outlet", data.hot_outlet_T_C),
            ("cold_inlet", data.cold_inlet_T_C),
            ("cold_outlet", data.cold_outlet_T_C),
        ]

        for name, temp in temps:
            is_valid = limits.temperature_min_C <= temp <= limits.temperature_max_C
            margin = min(
                temp - limits.temperature_min_C,
                limits.temperature_max_C - temp
            )
            margin_percent = (margin / (limits.temperature_max_C - limits.temperature_min_C)) * 100

            if temp < limits.temperature_min_C:
                severity = ConstraintSeverity.ERROR
                message = f"{name} temperature {temp:.1f}C below minimum"
                action = "Check for freezing risk or sensor error"
            elif temp > limits.temperature_max_C:
                severity = ConstraintSeverity.ERROR
                message = f"{name} temperature {temp:.1f}C exceeds maximum"
                action = "Check for metallurgical limits and sensor calibration"
            else:
                severity = ConstraintSeverity.INFO
                message = f"{name} temperature {temp:.1f}C is valid"
                action = "No action required"

            results.append(ConstraintCheckResult(
                constraint_type=ConstraintType.TEMPERATURE_RANGE,
                exchanger_id=data.exchanger_id,
                parameter_name=name,
                actual_value=temp,
                limit_value=limits.temperature_max_C,
                unit="C",
                is_valid=is_valid,
                severity=severity,
                margin=margin,
                margin_percent=margin_percent,
                message=message,
                standard_reference="API 660, ASME PTC 4.4",
                recommended_action=action,
            ))

        return results

    def _check_pressure_drops(
        self,
        data: ExchangerData,
    ) -> List[ConstraintCheckResult]:
        """Check pressure drops are within limits."""
        results = []
        limits = self.config.limits

        checks = [
            ("hot", data.hot_side_dp_kPa, data.hot_phase),
            ("cold", data.cold_side_dp_kPa, data.cold_phase),
        ]

        for side, dp, phase in checks:
            # Select limit based on phase
            if phase == FluidPhase.GAS:
                limit = limits.max_pressure_drop_gas_kPa
            else:
                limit = limits.max_pressure_drop_liquid_kPa

            is_valid = dp <= limit
            margin = limit - dp
            margin_percent = (margin / limit) * 100 if limit > 0 else 0

            if dp > limit:
                severity = ConstraintSeverity.ERROR
                message = f"{side.capitalize()} side dP {dp:.2f} kPa exceeds limit {limit:.2f} kPa"
                action = "Schedule cleaning or check for flow restrictions"
            elif dp > limit * 0.9:
                severity = ConstraintSeverity.WARNING
                message = f"{side.capitalize()} side dP {dp:.2f} kPa approaching limit"
                action = "Monitor closely, plan preventive cleaning"
            else:
                severity = ConstraintSeverity.INFO
                message = f"{side.capitalize()} side dP {dp:.2f} kPa is acceptable"
                action = "No action required"

            results.append(ConstraintCheckResult(
                constraint_type=ConstraintType.PRESSURE_DROP_LIMIT,
                exchanger_id=data.exchanger_id,
                parameter_name=f"{side}_side_pressure_drop",
                actual_value=dp,
                limit_value=limit,
                unit="kPa",
                is_valid=is_valid,
                severity=severity,
                margin=margin,
                margin_percent=margin_percent,
                message=message,
                standard_reference="API 660 Section 6.3, ISO 14414",
                recommended_action=action,
            ))

        return results

    def _check_energy_balance(self, data: ExchangerData) -> ConstraintCheckResult:
        """Check energy balance between hot and cold sides."""
        limits = self.config.limits

        # Calculate heat duties (kW = kg/s * kJ/(kg.K) * K)
        q_hot = (
            data.hot_flow_rate_kg_s *
            data.hot_cp_kJ_kgK *
            (data.hot_inlet_T_C - data.hot_outlet_T_C)
        )
        q_cold = (
            data.cold_flow_rate_kg_s *
            data.cold_cp_kJ_kgK *
            (data.cold_outlet_T_C - data.cold_inlet_T_C)
        )

        # Calculate imbalance
        q_max = max(abs(q_hot), abs(q_cold))
        if q_max > 0:
            imbalance = abs(q_hot - q_cold) / q_max
        else:
            imbalance = 0.0

        is_valid = imbalance <= limits.energy_balance_tolerance
        margin = limits.energy_balance_tolerance - imbalance
        margin_percent = (margin / limits.energy_balance_tolerance) * 100 if limits.energy_balance_tolerance > 0 else 0

        if imbalance > limits.energy_balance_tolerance:
            if self.config.strict_energy_balance:
                severity = ConstraintSeverity.ERROR
            else:
                severity = ConstraintSeverity.WARNING
            message = (
                f"Energy imbalance {imbalance:.1%} exceeds tolerance "
                f"{limits.energy_balance_tolerance:.1%}. "
                f"Q_hot={q_hot:.2f} kW, Q_cold={q_cold:.2f} kW"
            )
            action = "Verify all temperature and flow sensors"
        else:
            severity = ConstraintSeverity.INFO
            message = f"Energy balance OK: imbalance {imbalance:.1%}"
            action = "No action required"

        return ConstraintCheckResult(
            constraint_type=ConstraintType.ENERGY_BALANCE,
            exchanger_id=data.exchanger_id,
            parameter_name="energy_imbalance",
            actual_value=imbalance,
            limit_value=limits.energy_balance_tolerance,
            unit="ratio",
            is_valid=is_valid,
            severity=severity,
            margin=margin,
            margin_percent=margin_percent,
            message=message,
            standard_reference="ASME PTC 4.4 Section 5.2",
            recommended_action=action,
        )

    def _check_approach_temperature(
        self,
        data: ExchangerData,
    ) -> List[ConstraintCheckResult]:
        """Check approach temperatures at both ends."""
        results = []
        limits = self.config.limits

        # Hot end approach: T_hot_in - T_cold_out
        hot_end_approach = data.hot_inlet_T_C - data.cold_outlet_T_C

        # Cold end approach: T_hot_out - T_cold_in
        cold_end_approach = data.hot_outlet_T_C - data.cold_inlet_T_C

        for name, approach in [
            ("hot_end_approach", hot_end_approach),
            ("cold_end_approach", cold_end_approach),
        ]:
            is_valid = approach >= limits.min_approach_temp_C
            margin = approach - limits.min_approach_temp_C
            margin_percent = (margin / limits.min_approach_temp_C) * 100 if limits.min_approach_temp_C > 0 else 0

            if approach < 0:
                severity = ConstraintSeverity.CRITICAL
                message = f"Temperature crossover! {name} = {approach:.1f}C"
                action = "This is thermodynamically impossible - check sensor wiring"
            elif approach < limits.min_approach_temp_C:
                severity = ConstraintSeverity.ERROR
                message = f"{name} {approach:.1f}C below minimum {limits.min_approach_temp_C}C"
                action = "Reduce heat recovery target or check sensors"
            elif approach < limits.min_approach_temp_C * 1.5:
                severity = ConstraintSeverity.WARNING
                message = f"{name} {approach:.1f}C is marginal"
                action = "Monitor for approach temperature pinch"
            else:
                severity = ConstraintSeverity.INFO
                message = f"{name} {approach:.1f}C is acceptable"
                action = "No action required"

            results.append(ConstraintCheckResult(
                constraint_type=ConstraintType.APPROACH_TEMPERATURE,
                exchanger_id=data.exchanger_id,
                parameter_name=name,
                actual_value=approach,
                limit_value=limits.min_approach_temp_C,
                unit="C",
                is_valid=is_valid,
                severity=severity,
                margin=margin,
                margin_percent=margin_percent,
                message=message,
                standard_reference="Linnhoff Pinch Design Method",
                recommended_action=action,
            ))

        return results

    def _check_acid_dew_point(self, data: ExchangerData) -> ConstraintCheckResult:
        """Check flue gas outlet temperature vs acid dew point."""
        limits = self.config.limits

        outlet_temp = data.hot_outlet_T_C
        acid_dew_point = limits.acid_dew_point_C

        is_valid = outlet_temp >= acid_dew_point
        margin = outlet_temp - acid_dew_point
        margin_percent = (margin / acid_dew_point) * 100 if acid_dew_point > 0 else 0

        if outlet_temp < acid_dew_point:
            severity = ConstraintSeverity.CRITICAL
            message = (
                f"Flue gas outlet {outlet_temp:.1f}C below acid dew point "
                f"{acid_dew_point:.1f}C - CORROSION RISK"
            )
            action = "Immediately reduce heat recovery to raise outlet temperature"
        elif outlet_temp < acid_dew_point + 10:
            severity = ConstraintSeverity.WARNING
            message = f"Flue gas outlet {outlet_temp:.1f}C approaching acid dew point"
            action = "Monitor closely, consider reducing heat recovery"
        else:
            severity = ConstraintSeverity.INFO
            message = f"Flue gas outlet {outlet_temp:.1f}C safely above acid dew point"
            action = "No action required"

        return ConstraintCheckResult(
            constraint_type=ConstraintType.ACID_DEW_POINT,
            exchanger_id=data.exchanger_id,
            parameter_name="flue_gas_outlet_temperature",
            actual_value=outlet_temp,
            limit_value=acid_dew_point,
            unit="C",
            is_valid=is_valid,
            severity=severity,
            margin=margin,
            margin_percent=margin_percent,
            message=message,
            standard_reference="ASME PTC 4.3 Section 5.4.2",
            recommended_action=action,
        )

    def _check_fouling_resistance(
        self,
        data: ExchangerData,
    ) -> ConstraintCheckResult:
        """Check fouling resistance is within allowable range."""
        limits = self.config.limits

        rf = data.fouling_resistance_m2K_W
        rf_max = limits.max_fouling_resistance_m2K_W

        is_valid = rf <= rf_max
        margin = rf_max - rf
        margin_percent = (margin / rf_max) * 100 if rf_max > 0 else 0

        if rf > rf_max:
            severity = ConstraintSeverity.ERROR
            message = f"Fouling resistance {rf:.6f} m2K/W exceeds design {rf_max:.6f}"
            action = "Schedule cleaning - performance significantly degraded"
        elif rf > rf_max * 0.8:
            severity = ConstraintSeverity.WARNING
            message = f"Fouling resistance {rf:.6f} m2K/W approaching limit"
            action = "Plan preventive cleaning within next maintenance window"
        else:
            severity = ConstraintSeverity.INFO
            message = f"Fouling resistance {rf:.6f} m2K/W is acceptable"
            action = "No action required"

        return ConstraintCheckResult(
            constraint_type=ConstraintType.FOULING_RESISTANCE,
            exchanger_id=data.exchanger_id,
            parameter_name="fouling_resistance",
            actual_value=rf,
            limit_value=rf_max,
            unit="m2K/W",
            is_valid=is_valid,
            severity=severity,
            margin=margin,
            margin_percent=margin_percent,
            message=message,
            standard_reference="TEMA Standards Table 10",
            recommended_action=action,
        )

    def _check_ntu_validity(self, data: ExchangerData) -> ConstraintCheckResult:
        """Check NTU is in practical range."""
        limits = self.config.limits

        ntu = data.ntu
        ntu_max = limits.max_ntu

        is_valid = 0 < ntu <= ntu_max
        margin = ntu_max - ntu
        margin_percent = (margin / ntu_max) * 100 if ntu_max > 0 else 0

        if ntu <= 0:
            severity = ConstraintSeverity.ERROR
            message = f"NTU {ntu:.2f} is non-positive - invalid"
            action = "Check heat transfer calculation"
        elif ntu > ntu_max:
            severity = ConstraintSeverity.WARNING
            message = f"NTU {ntu:.2f} exceeds practical limit {ntu_max}"
            action = "Design may be impractical - verify assumptions"
        else:
            severity = ConstraintSeverity.INFO
            message = f"NTU {ntu:.2f} is in practical range"
            action = "No action required"

        return ConstraintCheckResult(
            constraint_type=ConstraintType.NTU_VALIDITY,
            exchanger_id=data.exchanger_id,
            parameter_name="ntu",
            actual_value=ntu,
            limit_value=ntu_max,
            unit="dimensionless",
            is_valid=is_valid,
            severity=severity,
            margin=margin,
            margin_percent=margin_percent,
            message=message,
            standard_reference="Heat Exchanger Design Theory",
            recommended_action=action,
        )

    # =========================================================================
    # SUMMARY AND ERROR HANDLING
    # =========================================================================

    def _build_summary(
        self,
        data: ExchangerData,
        results: List[ConstraintCheckResult],
    ) -> ConstraintValidationSummary:
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

        summary = ConstraintValidationSummary(
            exchanger_id=data.exchanger_id,
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
        data: ExchangerData,
        summary: ConstraintValidationSummary,
    ) -> None:
        """Raise appropriate violation error."""
        context = ViolationContext(
            exchanger_id=data.exchanger_id,
            sensor_readings={
                "hot_inlet_T": data.hot_inlet_T_C,
                "hot_outlet_T": data.hot_outlet_T_C,
                "cold_inlet_T": data.cold_inlet_T_C,
                "cold_outlet_T": data.cold_outlet_T_C,
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
                location=f"{data.exchanger_id}: {v['parameter_name']}",
                standard_reference=v["standard_reference"],
                recommended_action=v["recommended_action"],
            ))

        # Check for specific error types
        has_effectiveness_violation = any(
            v["constraint_type"] == ConstraintType.EFFECTIVENESS_BOUNDS.value
            for v in summary.critical_violations + summary.errors
        )
        has_energy_balance_violation = any(
            v["constraint_type"] == ConstraintType.ENERGY_BALANCE.value
            for v in summary.errors
        )

        if has_effectiveness_violation:
            eff = data.effectiveness or 0.0
            raise EffectivenessOutOfBoundsError(
                effectiveness=eff,
                exchanger_id=data.exchanger_id,
                context=context,
            )
        elif has_energy_balance_violation:
            q_hot = (
                data.hot_flow_rate_kg_s *
                data.hot_cp_kJ_kgK *
                (data.hot_inlet_T_C - data.hot_outlet_T_C)
            )
            q_cold = (
                data.cold_flow_rate_kg_s *
                data.cold_cp_kJ_kgK *
                (data.cold_outlet_T_C - data.cold_inlet_T_C)
            )
            raise EnergyBalanceError(
                message=f"Energy balance violation for {data.exchanger_id}",
                q_hot=q_hot,
                q_cold=q_cold,
                tolerance=self.config.limits.energy_balance_tolerance,
                exchanger_id=data.exchanger_id,
                context=context,
            )
        else:
            raise PhysicalBoundsViolation(
                message=f"Physical constraint violation for {data.exchanger_id}",
                parameter_name="multiple",
                actual_value=0.0,
                violations=violations,
                context=context,
            )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def validate_effectiveness(
    effectiveness: float,
    exchanger_id: str = "unknown",
) -> Tuple[bool, str]:
    """
    Quick check if effectiveness is in valid range.

    Args:
        effectiveness: Effectiveness value to check
        exchanger_id: Exchanger identifier for error message

    Returns:
        Tuple of (is_valid, message)
    """
    if effectiveness < 0:
        return False, f"{exchanger_id}: Effectiveness {effectiveness} is negative"
    elif effectiveness > 1:
        return False, f"{exchanger_id}: Effectiveness {effectiveness} exceeds 1.0"
    else:
        return True, f"{exchanger_id}: Effectiveness {effectiveness} is valid"


def validate_energy_balance(
    q_hot: float,
    q_cold: float,
    tolerance: float = 0.05,
) -> Tuple[bool, float]:
    """
    Quick check of energy balance.

    Args:
        q_hot: Hot side heat duty (kW)
        q_cold: Cold side heat duty (kW)
        tolerance: Allowed relative imbalance

    Returns:
        Tuple of (is_balanced, imbalance_ratio)
    """
    q_max = max(abs(q_hot), abs(q_cold))
    if q_max == 0:
        return True, 0.0

    imbalance = abs(q_hot - q_cold) / q_max
    return imbalance <= tolerance, imbalance


# =============================================================================
# MODULE EXPORTS
# =============================================================================


__all__ = [
    # Enums
    "ConstraintType",
    "ConstraintSeverity",
    "FluidPhase",
    # Config
    "ConstraintLimits",
    "ConstraintValidatorConfig",
    # Data models
    "ConstraintCheckResult",
    "ConstraintValidationSummary",
    "ExchangerData",
    # Main class
    "ConstraintValidator",
    # Convenience functions
    "validate_effectiveness",
    "validate_energy_balance",
]
