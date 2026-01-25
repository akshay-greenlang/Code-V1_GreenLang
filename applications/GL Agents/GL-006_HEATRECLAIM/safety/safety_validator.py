"""
GL-006 HEATRECLAIM - Safety Validator

Comprehensive safety constraint enforcement for heat exchanger network (HEN)
designs. Implements fail-closed validation against industrial safety standards.

Standards Compliance:
- ASME PTC 4.3: Air Heater Performance Test Code
- ASME PTC 4.4: Heat Recovery Steam Generator Performance
- API 660: Shell and Tube Heat Exchangers for General Refinery Service
- ISO 14414: Pump System Energy Assessment
- TEMA Standards: Heat Exchanger Design

Zero-Hallucination Guarantee:
All safety checks use deterministic arithmetic with SHA-256 provenance tracking.
No LLM inference is used for safety-critical decisions.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import logging

from ..core.config import (
    ThermalConstraints,
    Phase,
    StreamType,
)
from ..core.schemas import (
    HeatExchanger,
    HeatStream,
    HENDesign,
)
from .exceptions import (
    SafetyViolationError,
    ApproachTemperatureViolation,
    FilmTemperatureViolation,
    AcidDewPointViolation,
    PressureDropViolation,
    ThermalStressViolation,
    ViolationDetails,
    ViolationSeverity,
)


logger = logging.getLogger(__name__)


class ValidationMode(Enum):
    """Safety validation strictness modes."""
    STRICT = "strict"      # All violations are errors (production)
    NORMAL = "normal"      # Critical violations are errors, others warnings
    RELAXED = "relaxed"    # Log violations but allow (testing only)


@dataclass(frozen=True)
class SafetyValidationResult:
    """
    Immutable result of safety validation.

    This dataclass captures all safety checks performed on a HEN design
    with full provenance for audit trails.
    """

    # Validation status
    is_safe: bool
    validation_passed: bool
    validation_mode: str

    # Violation counts
    total_violations: int
    critical_violations: int
    error_violations: int
    warning_violations: int

    # Detailed violations
    violations: Tuple[ViolationDetails, ...]

    # Constraint checks performed
    checks_performed: Tuple[str, ...]
    checks_passed: int
    checks_failed: int

    # Design being validated
    design_id: str

    # Provenance
    validation_timestamp: str
    validator_version: str
    constraints_hash: str
    design_hash: str
    result_hash: str

    def get_violations_by_severity(
        self, severity: ViolationSeverity
    ) -> List[ViolationDetails]:
        """Filter violations by severity level."""
        return [v for v in self.violations if v.severity == severity]

    def get_violations_by_constraint(
        self, constraint_tag: str
    ) -> List[ViolationDetails]:
        """Filter violations by constraint type."""
        return [v for v in self.violations if v.constraint_tag == constraint_tag]

    def to_audit_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for audit logging."""
        return {
            "is_safe": self.is_safe,
            "validation_passed": self.validation_passed,
            "validation_mode": self.validation_mode,
            "total_violations": self.total_violations,
            "critical_violations": self.critical_violations,
            "design_id": self.design_id,
            "validation_timestamp": self.validation_timestamp,
            "result_hash": self.result_hash,
        }


class SafetyValidator:
    """
    Safety constraint validator for HEN designs.

    Implements comprehensive safety checks against thermal constraints
    defined in ASME, API, and ISO standards. All validations are
    deterministic with full provenance tracking.

    Usage:
        validator = SafetyValidator(constraints, mode=ValidationMode.STRICT)
        result = validator.validate_design(hen_design, hot_streams, cold_streams)

        if not result.is_safe:
            raise result.violations[0]  # Or handle appropriately
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        constraints: ThermalConstraints,
        mode: ValidationMode = ValidationMode.STRICT,
        fail_closed: bool = True,
    ):
        """
        Initialize safety validator.

        Args:
            constraints: Thermal safety constraints to enforce
            mode: Validation strictness mode
            fail_closed: If True, reject designs with any errors
        """
        self.constraints = constraints
        self.mode = mode
        self.fail_closed = fail_closed
        self._constraints_hash = self._compute_constraints_hash()

    def _compute_constraints_hash(self) -> str:
        """Compute SHA-256 hash of constraint configuration."""
        constraint_dict = {
            "delta_t_min_default": self.constraints.delta_t_min_default,
            "delta_t_min_gas_gas": self.constraints.delta_t_min_gas_gas,
            "delta_t_min_gas_liquid": self.constraints.delta_t_min_gas_liquid,
            "delta_t_min_liquid_liquid": self.constraints.delta_t_min_liquid_liquid,
            "delta_t_min_phase_change": self.constraints.delta_t_min_phase_change,
            "max_film_temperature": self.constraints.max_film_temperature,
            "acid_dew_point": self.constraints.acid_dew_point,
            "max_pressure_drop_liquid": self.constraints.max_pressure_drop_liquid,
            "max_pressure_drop_gas": self.constraints.max_pressure_drop_gas,
            "max_thermal_stress_rate": self.constraints.max_thermal_stress_rate,
        }
        json_str = json.dumps(constraint_dict, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def _compute_hash(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash of arbitrary data."""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def validate_design(
        self,
        design: HENDesign,
        hot_streams: List[HeatStream],
        cold_streams: List[HeatStream],
    ) -> SafetyValidationResult:
        """
        Validate complete HEN design against all safety constraints.

        Args:
            design: The HEN design to validate
            hot_streams: List of hot process streams
            cold_streams: List of cold process streams

        Returns:
            SafetyValidationResult with all validation details

        Raises:
            SafetyViolationError: If fail_closed=True and violations found
        """
        violations: List[ViolationDetails] = []
        checks_performed: List[str] = []
        stream_map = self._build_stream_map(hot_streams, cold_streams)

        # Validate each exchanger
        for exchanger in design.exchangers:
            hot_stream = stream_map.get(exchanger.hot_stream_id)
            cold_stream = stream_map.get(exchanger.cold_stream_id)

            # Check approach temperatures
            checks_performed.append(f"DELTA_T_MIN:{exchanger.exchanger_id}")
            delta_t_violations = self._check_approach_temperature(
                exchanger, hot_stream, cold_stream
            )
            violations.extend(delta_t_violations)

            # Check film temperature
            checks_performed.append(f"FILM_TEMP:{exchanger.exchanger_id}")
            film_violations = self._check_film_temperature(
                exchanger, hot_stream
            )
            violations.extend(film_violations)

            # Check acid dew point
            checks_performed.append(f"ACID_DEW_POINT:{exchanger.exchanger_id}")
            acid_violations = self._check_acid_dew_point(
                exchanger, hot_stream
            )
            violations.extend(acid_violations)

            # Check pressure drops
            checks_performed.append(f"PRESSURE_DROP:{exchanger.exchanger_id}")
            pressure_violations = self._check_pressure_drop(
                exchanger, hot_stream, cold_stream
            )
            violations.extend(pressure_violations)

        # Count violations by severity
        critical = sum(1 for v in violations if v.severity == ViolationSeverity.CRITICAL)
        errors = sum(1 for v in violations if v.severity == ViolationSeverity.ERROR)
        warnings = sum(1 for v in violations if v.severity == ViolationSeverity.WARNING)

        # Determine if validation passed
        if self.mode == ValidationMode.STRICT:
            validation_passed = len(violations) == 0
        elif self.mode == ValidationMode.NORMAL:
            validation_passed = critical == 0 and errors == 0
        else:  # RELAXED
            validation_passed = critical == 0

        is_safe = critical == 0

        # Compute design hash
        design_dict = {
            "design_id": design.design_id,
            "exchanger_count": len(design.exchangers),
            "exchangers": [
                {
                    "id": e.exchanger_id,
                    "hot_inlet": e.hot_inlet_T_C,
                    "hot_outlet": e.hot_outlet_T_C,
                    "cold_inlet": e.cold_inlet_T_C,
                    "cold_outlet": e.cold_outlet_T_C,
                }
                for e in design.exchangers
            ],
        }
        design_hash = self._compute_hash(design_dict)

        # Compute result hash
        result_dict = {
            "is_safe": is_safe,
            "validation_passed": validation_passed,
            "total_violations": len(violations),
            "design_hash": design_hash,
            "constraints_hash": self._constraints_hash,
        }
        result_hash = self._compute_hash(result_dict)

        result = SafetyValidationResult(
            is_safe=is_safe,
            validation_passed=validation_passed,
            validation_mode=self.mode.value,
            total_violations=len(violations),
            critical_violations=critical,
            error_violations=errors,
            warning_violations=warnings,
            violations=tuple(violations),
            checks_performed=tuple(checks_performed),
            checks_passed=len(checks_performed) - len(violations),
            checks_failed=len(violations),
            design_id=design.design_id,
            validation_timestamp=datetime.now(timezone.utc).isoformat(),
            validator_version=self.VERSION,
            constraints_hash=self._constraints_hash,
            design_hash=design_hash,
            result_hash=result_hash,
        )

        # Log result
        if violations:
            logger.warning(
                f"Safety validation found {len(violations)} violations "
                f"({critical} critical, {errors} errors, {warnings} warnings) "
                f"for design {design.design_id}"
            )
        else:
            logger.info(
                f"Safety validation passed for design {design.design_id}"
            )

        # Fail-closed behavior
        if self.fail_closed and not validation_passed:
            violation_error = self._create_violation_error(violations, design.design_id)
            raise violation_error

        return result

    def _build_stream_map(
        self,
        hot_streams: List[HeatStream],
        cold_streams: List[HeatStream],
    ) -> Dict[str, HeatStream]:
        """Build lookup map for streams by ID."""
        stream_map = {}
        for stream in hot_streams:
            stream_map[stream.stream_id] = stream
        for stream in cold_streams:
            stream_map[stream.stream_id] = stream
        return stream_map

    def _check_approach_temperature(
        self,
        exchanger: HeatExchanger,
        hot_stream: Optional[HeatStream],
        cold_stream: Optional[HeatStream],
    ) -> List[ViolationDetails]:
        """
        Check minimum approach temperature constraint.

        The approach temperature is the temperature difference between
        the hot and cold streams at each end of the exchanger.
        """
        violations = []

        # Calculate approach temperatures
        hot_end_approach = exchanger.hot_inlet_T_C - exchanger.cold_outlet_T_C
        cold_end_approach = exchanger.hot_outlet_T_C - exchanger.cold_inlet_T_C

        # Determine required minimum based on stream phases
        delta_t_min = self._get_required_delta_t_min(hot_stream, cold_stream)

        # Check hot end approach
        if hot_end_approach < delta_t_min:
            severity = (
                ViolationSeverity.CRITICAL if hot_end_approach < 0
                else ViolationSeverity.ERROR if hot_end_approach < delta_t_min * 0.5
                else ViolationSeverity.WARNING
            )
            violations.append(ViolationDetails(
                constraint_tag="DELTA_T_MIN",
                constraint_description="Minimum approach temperature at hot end",
                actual_value=round(hot_end_approach, 2),
                limit_value=delta_t_min,
                unit="C",
                severity=severity,
                location=f"Exchanger {exchanger.exchanger_id}, hot end",
                standard_reference="Linnhoff & Hindmarsh (1983), Pinch Design Method",
                recommended_action=(
                    "Increase hot stream inlet temperature or decrease cold "
                    "stream outlet temperature to meet minimum approach"
                ),
            ))

        # Check cold end approach
        if cold_end_approach < delta_t_min:
            severity = (
                ViolationSeverity.CRITICAL if cold_end_approach < 0
                else ViolationSeverity.ERROR if cold_end_approach < delta_t_min * 0.5
                else ViolationSeverity.WARNING
            )
            violations.append(ViolationDetails(
                constraint_tag="DELTA_T_MIN",
                constraint_description="Minimum approach temperature at cold end",
                actual_value=round(cold_end_approach, 2),
                limit_value=delta_t_min,
                unit="C",
                severity=severity,
                location=f"Exchanger {exchanger.exchanger_id}, cold end",
                standard_reference="Linnhoff & Hindmarsh (1983), Pinch Design Method",
                recommended_action=(
                    "Increase hot stream outlet temperature or decrease cold "
                    "stream inlet temperature to meet minimum approach"
                ),
            ))

        return violations

    def _get_required_delta_t_min(
        self,
        hot_stream: Optional[HeatStream],
        cold_stream: Optional[HeatStream],
    ) -> float:
        """Determine required delta_t_min based on stream phases."""
        if hot_stream is None or cold_stream is None:
            return self.constraints.delta_t_min_default

        hot_phase = hot_stream.phase if hasattr(hot_stream, 'phase') else Phase.LIQUID
        cold_phase = cold_stream.phase if hasattr(cold_stream, 'phase') else Phase.LIQUID

        # Phase change has lowest delta_t_min
        if hot_phase == Phase.TWO_PHASE or cold_phase == Phase.TWO_PHASE:
            return self.constraints.delta_t_min_phase_change

        # Gas-gas has highest delta_t_min
        if hot_phase == Phase.GAS and cold_phase == Phase.GAS:
            return self.constraints.delta_t_min_gas_gas

        # Gas-liquid intermediate
        if hot_phase == Phase.GAS or cold_phase == Phase.GAS:
            return self.constraints.delta_t_min_gas_liquid

        # Liquid-liquid default
        return self.constraints.delta_t_min_liquid_liquid

    def _check_film_temperature(
        self,
        exchanger: HeatExchanger,
        hot_stream: Optional[HeatStream],
    ) -> List[ViolationDetails]:
        """
        Check maximum film temperature constraint.

        High film temperatures can cause:
        - Coking in hydrocarbon services
        - Tube metallurgy degradation
        - Accelerated fouling
        """
        violations = []
        max_film_temp = self.constraints.max_film_temperature

        # Estimate film temperature (simplified: average of bulk and wall)
        # In reality, this requires heat transfer coefficient calculations
        # Here we use hot inlet as conservative upper bound for film temp
        estimated_film_temp = exchanger.hot_inlet_T_C

        if estimated_film_temp > max_film_temp:
            severity = (
                ViolationSeverity.CRITICAL if estimated_film_temp > max_film_temp * 1.1
                else ViolationSeverity.ERROR
            )
            violations.append(ViolationDetails(
                constraint_tag="MAX_FILM_TEMPERATURE",
                constraint_description="Maximum film temperature to prevent coking",
                actual_value=round(estimated_film_temp, 2),
                limit_value=max_film_temp,
                unit="C",
                severity=severity,
                location=f"Exchanger {exchanger.exchanger_id}, hot side inlet",
                standard_reference="API 660 Section 7.2.5, TEMA Standards",
                recommended_action=(
                    "Reduce hot stream inlet temperature, increase heat transfer "
                    "area, or use different exchanger type to reduce film temperature"
                ),
            ))

        return violations

    def _check_acid_dew_point(
        self,
        exchanger: HeatExchanger,
        hot_stream: Optional[HeatStream],
    ) -> List[ViolationDetails]:
        """
        Check acid dew point constraint for flue gas streams.

        Outlet temperatures below acid dew point cause:
        - Sulfuric acid condensation
        - Nitric acid condensation
        - Severe cold-end corrosion
        """
        violations = []
        acid_dew_point = self.constraints.acid_dew_point

        # Only check for gas streams (flue gas typically)
        if hot_stream is None:
            return violations

        is_gas_stream = (
            hasattr(hot_stream, 'phase') and hot_stream.phase == Phase.GAS
        ) or (
            hot_stream.fluid_name.lower() in ['flue_gas', 'exhaust', 'combustion_gas']
        )

        if not is_gas_stream:
            return violations

        # Check hot outlet temperature against acid dew point
        if exchanger.hot_outlet_T_C < acid_dew_point:
            margin = acid_dew_point - exchanger.hot_outlet_T_C
            severity = (
                ViolationSeverity.CRITICAL if margin > 20
                else ViolationSeverity.ERROR if margin > 10
                else ViolationSeverity.WARNING
            )
            violations.append(ViolationDetails(
                constraint_tag="ACID_DEW_POINT",
                constraint_description="Minimum outlet temperature to prevent acid condensation",
                actual_value=round(exchanger.hot_outlet_T_C, 2),
                limit_value=acid_dew_point,
                unit="C",
                severity=severity,
                location=f"Exchanger {exchanger.exchanger_id}, hot side outlet",
                standard_reference="ASME PTC 4.3 Section 5.4.2",
                recommended_action=(
                    f"Increase hot stream outlet temperature to at least "
                    f"{acid_dew_point}C to prevent acid condensation and corrosion"
                ),
            ))

        return violations

    def _check_pressure_drop(
        self,
        exchanger: HeatExchanger,
        hot_stream: Optional[HeatStream],
        cold_stream: Optional[HeatStream],
    ) -> List[ViolationDetails]:
        """
        Check pressure drop constraints for both sides.

        Excessive pressure drop causes:
        - Pump/compressor capacity exceeded
        - Increased operating cost
        - Process bottleneck
        """
        violations = []

        # Check hot side pressure drop
        hot_is_gas = self._is_gas_phase(hot_stream)
        hot_limit = (
            self.constraints.max_pressure_drop_gas if hot_is_gas
            else self.constraints.max_pressure_drop_liquid
        )

        if exchanger.hot_side_dp_kPa > hot_limit:
            violations.append(ViolationDetails(
                constraint_tag="MAX_PRESSURE_DROP",
                constraint_description=f"Maximum pressure drop for {'gas' if hot_is_gas else 'liquid'}",
                actual_value=round(exchanger.hot_side_dp_kPa, 2),
                limit_value=hot_limit,
                unit="kPa",
                severity=ViolationSeverity.ERROR,
                location=f"Exchanger {exchanger.exchanger_id}, hot side",
                standard_reference="API 660 Section 6.3, ISO 14414",
                recommended_action=(
                    "Reduce flow velocity by increasing flow area or using "
                    "different exchanger geometry"
                ),
            ))

        # Check cold side pressure drop
        cold_is_gas = self._is_gas_phase(cold_stream)
        cold_limit = (
            self.constraints.max_pressure_drop_gas if cold_is_gas
            else self.constraints.max_pressure_drop_liquid
        )

        if exchanger.cold_side_dp_kPa > cold_limit:
            violations.append(ViolationDetails(
                constraint_tag="MAX_PRESSURE_DROP",
                constraint_description=f"Maximum pressure drop for {'gas' if cold_is_gas else 'liquid'}",
                actual_value=round(exchanger.cold_side_dp_kPa, 2),
                limit_value=cold_limit,
                unit="kPa",
                severity=ViolationSeverity.ERROR,
                location=f"Exchanger {exchanger.exchanger_id}, cold side",
                standard_reference="API 660 Section 6.3, ISO 14414",
                recommended_action=(
                    "Reduce flow velocity by increasing flow area or using "
                    "different exchanger geometry"
                ),
            ))

        return violations

    def _is_gas_phase(self, stream: Optional[HeatStream]) -> bool:
        """Check if stream is in gas phase."""
        if stream is None:
            return False
        if hasattr(stream, 'phase'):
            return stream.phase in [Phase.GAS, Phase.SUPERCRITICAL]
        return False

    def _create_violation_error(
        self,
        violations: List[ViolationDetails],
        design_id: str,
    ) -> SafetyViolationError:
        """Create appropriate exception based on violation types."""
        # Find the most severe violation type
        has_approach = any(v.constraint_tag == "DELTA_T_MIN" for v in violations)
        has_film = any(v.constraint_tag == "MAX_FILM_TEMPERATURE" for v in violations)
        has_acid = any(v.constraint_tag == "ACID_DEW_POINT" for v in violations)
        has_pressure = any(v.constraint_tag == "MAX_PRESSURE_DROP" for v in violations)

        # Prioritize violation type for exception
        if any(v.severity == ViolationSeverity.CRITICAL for v in violations):
            if has_acid:
                return AcidDewPointViolation(
                    f"Critical acid dew point violation in design {design_id}",
                    violations=violations,
                    design_id=design_id,
                )
            if has_film:
                return FilmTemperatureViolation(
                    f"Critical film temperature violation in design {design_id}",
                    violations=violations,
                    design_id=design_id,
                )
            if has_approach:
                return ApproachTemperatureViolation(
                    f"Critical approach temperature violation in design {design_id}",
                    violations=violations,
                    design_id=design_id,
                )

        # Return generic error for non-critical violations
        return SafetyViolationError(
            f"Safety constraint violations found in design {design_id}",
            violations=violations,
            design_id=design_id,
        )

    def validate_single_exchanger(
        self,
        exchanger: HeatExchanger,
        hot_stream: Optional[HeatStream] = None,
        cold_stream: Optional[HeatStream] = None,
    ) -> SafetyValidationResult:
        """
        Validate a single heat exchanger against safety constraints.

        Useful for quick validation during design iteration.
        """
        # Create minimal design wrapper
        from ..core.schemas import HENDesign

        temp_design = HENDesign(
            design_id=f"temp_{exchanger.exchanger_id}",
            exchangers=[exchanger],
            total_heat_recovered_kW=exchanger.duty_kW,
            hot_utility_required_kW=0.0,
            cold_utility_required_kW=0.0,
        )

        hot_streams = [hot_stream] if hot_stream else []
        cold_streams = [cold_stream] if cold_stream else []

        # Use non-fail-closed mode for single exchanger validation
        original_fail_closed = self.fail_closed
        self.fail_closed = False

        try:
            result = self.validate_design(temp_design, hot_streams, cold_streams)
        finally:
            self.fail_closed = original_fail_closed

        return result
