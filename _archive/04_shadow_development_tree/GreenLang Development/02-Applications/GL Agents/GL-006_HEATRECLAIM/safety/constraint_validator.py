"""
GL-006 HEATRECLAIM - Safety Constraint Validator

Production-grade safety constraint enforcement for heat exchanger network (HEN)
designs. Implements all safety constraints from pack.yaml with penalty costs
for near-violations and strict rejection of violating designs.

Safety Constraints Enforced:
1. DELTA_T_MIN: 5C minimum approach temperature (per phase)
2. MAX_FILM_TEMPERATURE: 400C coking prevention limit
3. ACID_DEW_POINT: 120C minimum outlet for flue gas
4. MAX_PRESSURE_DROP: 50 kPa liquids, 5 kPa gases
5. THERMAL_STRESS_RATE: 5C/min maximum temperature change rate

Standards Compliance:
- ASME PTC 4.3: Air Heater Performance
- ASME PTC 4.4: HRSG Performance
- API 660: Shell and Tube Heat Exchangers
- ISO 14414: Pump System Energy Assessment
- TEMA: Tubular Exchanger Manufacturers Association

Zero-Hallucination Guarantee:
All constraint checks use deterministic arithmetic with SHA-256 provenance.
No LLM inference is used for any safety-critical decisions.

Example:
    >>> from safety.constraint_validator import ConstraintValidator
    >>> validator = ConstraintValidator(config)
    >>> result = validator.validate_hen_design(design, streams)
    >>> if not result.is_acceptable:
    ...     raise SafetyViolationError(result.get_rejection_reason())
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib
import json
import logging
import math

from pydantic import BaseModel, Field, validator

from ..core.config import ThermalConstraints, Phase, StreamType
from ..core.schemas import HeatExchanger, HeatStream, HENDesign
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


class ConstraintType(str, Enum):
    """Safety constraint types from pack.yaml."""

    DELTA_T_MIN = "DELTA_T_MIN"
    MAX_FILM_TEMPERATURE = "MAX_FILM_TEMPERATURE"
    ACID_DEW_POINT = "ACID_DEW_POINT"
    MAX_PRESSURE_DROP = "MAX_PRESSURE_DROP"
    THERMAL_STRESS_RATE = "THERMAL_STRESS_RATE"


class PenaltyLevel(str, Enum):
    """Penalty severity for near-violations."""

    NONE = "none"            # Within safe operating range
    LOW = "low"              # 80-90% of limit (warning zone)
    MEDIUM = "medium"        # 90-95% of limit (caution zone)
    HIGH = "high"            # 95-100% of limit (danger zone)
    VIOLATION = "violation"  # Exceeds limit (reject)


@dataclass(frozen=True)
class ConstraintLimit:
    """Immutable constraint limit specification."""

    constraint_type: ConstraintType
    limit_value: float
    unit: str
    is_minimum: bool  # True for minimum limits (e.g., acid dew point)
    standard_reference: str
    description: str

    # Penalty thresholds (fractions of limit)
    warning_threshold: float = 0.80   # 80% of limit = low penalty
    caution_threshold: float = 0.90   # 90% of limit = medium penalty
    danger_threshold: float = 0.95    # 95% of limit = high penalty

    # Penalty costs per unit violation
    penalty_cost_per_unit: float = 1000.0  # $/unit


@dataclass(frozen=True)
class ConstraintCheckResult:
    """Result of a single constraint check."""

    constraint_type: ConstraintType
    exchanger_id: str
    location: str  # hot_end, cold_end, hot_side, cold_side
    actual_value: float
    limit_value: float
    unit: str
    margin_percent: float  # (actual - limit) / limit * 100
    penalty_level: PenaltyLevel
    penalty_cost_usd: float
    is_violation: bool
    severity: ViolationSeverity
    standard_reference: str
    recommended_action: str
    calculation_hash: str


class ConstraintCheckSummary(BaseModel):
    """Summary of all constraint checks for a HEN design."""

    design_id: str = Field(..., description="Design being validated")
    timestamp: str = Field(..., description="Validation timestamp ISO format")

    # Overall status
    is_acceptable: bool = Field(..., description="Design passes all constraints")
    has_violations: bool = Field(..., description="Any hard violations")
    has_near_violations: bool = Field(..., description="Any near-violation warnings")

    # Counts by constraint type
    checks_performed: int = Field(default=0, ge=0)
    violations_count: int = Field(default=0, ge=0)
    warnings_count: int = Field(default=0, ge=0)

    # Counts by severity
    critical_count: int = Field(default=0, ge=0)
    error_count: int = Field(default=0, ge=0)
    warning_count: int = Field(default=0, ge=0)

    # Penalty costs
    total_penalty_cost_usd: float = Field(default=0.0, ge=0.0)
    penalty_cost_by_constraint: Dict[str, float] = Field(default_factory=dict)

    # Detailed results
    check_results: List[Dict[str, Any]] = Field(default_factory=list)
    violations: List[Dict[str, Any]] = Field(default_factory=list)
    near_violations: List[Dict[str, Any]] = Field(default_factory=list)

    # Provenance
    constraints_hash: str = Field(default="", description="SHA-256 of constraints")
    design_hash: str = Field(default="", description="SHA-256 of design inputs")
    result_hash: str = Field(default="", description="SHA-256 of this result")

    def get_rejection_reason(self) -> str:
        """Get human-readable rejection reason."""
        if not self.violations:
            return ""

        reasons = []
        for v in self.violations:
            reasons.append(
                f"- {v['constraint_type']}: {v['actual_value']:.2f} {v['unit']} "
                f"(limit: {v['limit_value']:.2f} {v['unit']}) at {v['location']}"
            )
        return f"Design {self.design_id} rejected due to safety violations:\n" + "\n".join(reasons)


class ConstraintValidator:
    """
    Production-grade safety constraint validator for HEN designs.

    Enforces all five safety constraints from pack.yaml with:
    - Hard rejection of violating designs (fail-closed)
    - Penalty costs for near-violations
    - SHA-256 provenance for all checks
    - Detailed audit trail

    Standards:
    - ASME PTC 4.3/4.4: Heat recovery performance
    - API 660: Shell and tube exchangers
    - ISO 14414: Pump system assessment
    - TEMA: Exchanger design standards

    Usage:
        >>> validator = ConstraintValidator(thermal_constraints)
        >>> summary = validator.validate_hen_design(design, hot_streams, cold_streams)
        >>> if not summary.is_acceptable:
        ...     raise SafetyViolationError(summary.get_rejection_reason())
        >>> # Apply penalty costs to economic analysis
        >>> adjusted_cost = base_cost + summary.total_penalty_cost_usd
    """

    VERSION = "1.0.0"

    # Default constraint limits from pack.yaml
    DEFAULT_LIMITS = {
        ConstraintType.DELTA_T_MIN: ConstraintLimit(
            constraint_type=ConstraintType.DELTA_T_MIN,
            limit_value=5.0,
            unit="C",
            is_minimum=True,
            standard_reference="Linnhoff & Hindmarsh (1983), Pinch Design Method",
            description="Minimum approach temperature to ensure heat transfer driving force",
            penalty_cost_per_unit=5000.0,  # $/C below limit
        ),
        ConstraintType.MAX_FILM_TEMPERATURE: ConstraintLimit(
            constraint_type=ConstraintType.MAX_FILM_TEMPERATURE,
            limit_value=400.0,
            unit="C",
            is_minimum=False,
            standard_reference="API 660 Section 7.2.5, TEMA Standards",
            description="Maximum film temperature to prevent coking and metallurgy degradation",
            penalty_cost_per_unit=10000.0,  # $/C above limit
        ),
        ConstraintType.ACID_DEW_POINT: ConstraintLimit(
            constraint_type=ConstraintType.ACID_DEW_POINT,
            limit_value=120.0,
            unit="C",
            is_minimum=True,
            standard_reference="ASME PTC 4.3 Section 5.4.2",
            description="Minimum outlet temperature to prevent acid condensation and corrosion",
            penalty_cost_per_unit=8000.0,  # $/C below limit
        ),
        ConstraintType.MAX_PRESSURE_DROP: ConstraintLimit(
            constraint_type=ConstraintType.MAX_PRESSURE_DROP,
            limit_value=50.0,  # Default for liquids
            unit="kPa",
            is_minimum=False,
            standard_reference="API 660 Section 6.3, ISO 14414",
            description="Maximum pressure drop to avoid pump/compressor constraints",
            penalty_cost_per_unit=2000.0,  # $/kPa above limit
        ),
        ConstraintType.THERMAL_STRESS_RATE: ConstraintLimit(
            constraint_type=ConstraintType.THERMAL_STRESS_RATE,
            limit_value=5.0,
            unit="C/min",
            is_minimum=False,
            standard_reference="ASME PTC 4.4 Section 5.5",
            description="Maximum temperature change rate to prevent thermal shock",
            penalty_cost_per_unit=15000.0,  # High cost for thermal stress risk
        ),
    }

    def __init__(
        self,
        constraints: ThermalConstraints,
        fail_closed: bool = True,
        apply_penalties: bool = True,
        custom_limits: Optional[Dict[ConstraintType, ConstraintLimit]] = None,
    ):
        """
        Initialize constraint validator.

        Args:
            constraints: Thermal constraints configuration
            fail_closed: If True, raise exception on any violation
            apply_penalties: If True, calculate penalty costs for near-violations
            custom_limits: Optional custom constraint limits
        """
        self.constraints = constraints
        self.fail_closed = fail_closed
        self.apply_penalties = apply_penalties

        # Merge custom limits with defaults
        self.limits = dict(self.DEFAULT_LIMITS)
        if custom_limits:
            self.limits.update(custom_limits)

        # Update limits from ThermalConstraints
        self._update_limits_from_config()

        # Compute constraints hash for provenance
        self._constraints_hash = self._compute_constraints_hash()

        logger.info(
            f"ConstraintValidator initialized: fail_closed={fail_closed}, "
            f"apply_penalties={apply_penalties}, hash={self._constraints_hash[:16]}..."
        )

    def _update_limits_from_config(self) -> None:
        """Update constraint limits from ThermalConstraints config."""
        # Update delta_t_min based on phase-specific values
        # Use most restrictive (minimum of minimums) as the hard limit
        min_delta_t = min(
            self.constraints.delta_t_min_phase_change,  # 5C
            self.constraints.delta_t_min_liquid_liquid,  # 10C
            self.constraints.delta_t_min_gas_liquid,     # 15C
            self.constraints.delta_t_min_gas_gas,        # 20C
        )
        self.limits[ConstraintType.DELTA_T_MIN] = ConstraintLimit(
            constraint_type=ConstraintType.DELTA_T_MIN,
            limit_value=min_delta_t,
            unit="C",
            is_minimum=True,
            standard_reference="Linnhoff & Hindmarsh (1983), Pinch Design Method",
            description="Minimum approach temperature to ensure heat transfer driving force",
            penalty_cost_per_unit=5000.0,
        )

        # Update max film temperature
        self.limits[ConstraintType.MAX_FILM_TEMPERATURE] = ConstraintLimit(
            constraint_type=ConstraintType.MAX_FILM_TEMPERATURE,
            limit_value=self.constraints.max_film_temperature,
            unit="C",
            is_minimum=False,
            standard_reference="API 660 Section 7.2.5, TEMA Standards",
            description="Maximum film temperature to prevent coking and metallurgy degradation",
            penalty_cost_per_unit=10000.0,
        )

        # Update acid dew point
        self.limits[ConstraintType.ACID_DEW_POINT] = ConstraintLimit(
            constraint_type=ConstraintType.ACID_DEW_POINT,
            limit_value=self.constraints.acid_dew_point,
            unit="C",
            is_minimum=True,
            standard_reference="ASME PTC 4.3 Section 5.4.2",
            description="Minimum outlet temperature to prevent acid condensation and corrosion",
            penalty_cost_per_unit=8000.0,
        )

        # Update thermal stress rate
        self.limits[ConstraintType.THERMAL_STRESS_RATE] = ConstraintLimit(
            constraint_type=ConstraintType.THERMAL_STRESS_RATE,
            limit_value=self.constraints.max_thermal_stress_rate,
            unit="C/min",
            is_minimum=False,
            standard_reference="ASME PTC 4.4 Section 5.5",
            description="Maximum temperature change rate to prevent thermal shock",
            penalty_cost_per_unit=15000.0,
        )

    def _compute_constraints_hash(self) -> str:
        """Compute SHA-256 hash of constraint configuration."""
        config_dict = {
            "limits": {
                k.value: {
                    "limit_value": v.limit_value,
                    "unit": v.unit,
                    "is_minimum": v.is_minimum,
                }
                for k, v in self.limits.items()
            },
            "fail_closed": self.fail_closed,
            "apply_penalties": self.apply_penalties,
            "version": self.VERSION,
        }
        json_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def _compute_hash(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash of arbitrary data."""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def validate_hen_design(
        self,
        design: HENDesign,
        hot_streams: List[HeatStream],
        cold_streams: List[HeatStream],
        startup_time_minutes: Optional[float] = None,
    ) -> ConstraintCheckSummary:
        """
        Validate complete HEN design against all safety constraints.

        Args:
            design: The HEN design to validate
            hot_streams: List of hot process streams
            cold_streams: List of cold process streams
            startup_time_minutes: Optional startup time for thermal stress check

        Returns:
            ConstraintCheckSummary with all validation details

        Raises:
            SafetyViolationError: If fail_closed=True and violations found
        """
        start_time = datetime.now(timezone.utc)
        check_results: List[ConstraintCheckResult] = []
        stream_map = self._build_stream_map(hot_streams, cold_streams)

        logger.info(f"Validating HEN design {design.design_id} with {len(design.exchangers)} exchangers")

        # Validate each exchanger against all constraints
        for exchanger in design.exchangers:
            hot_stream = stream_map.get(exchanger.hot_stream_id)
            cold_stream = stream_map.get(exchanger.cold_stream_id)

            # Constraint 1: Delta T Min (approach temperature)
            check_results.extend(
                self._check_delta_t_min(exchanger, hot_stream, cold_stream)
            )

            # Constraint 2: Max Film Temperature
            check_results.extend(
                self._check_max_film_temperature(exchanger, hot_stream)
            )

            # Constraint 3: Acid Dew Point
            check_results.extend(
                self._check_acid_dew_point(exchanger, hot_stream)
            )

            # Constraint 4: Max Pressure Drop
            check_results.extend(
                self._check_max_pressure_drop(exchanger, hot_stream, cold_stream)
            )

            # Constraint 5: Thermal Stress Rate (if startup time provided)
            if startup_time_minutes is not None:
                check_results.extend(
                    self._check_thermal_stress_rate(exchanger, startup_time_minutes)
                )

        # Build summary
        summary = self._build_summary(design, check_results)

        # Log results
        if summary.has_violations:
            logger.warning(
                f"Design {design.design_id} has {summary.violations_count} safety violations, "
                f"{summary.warnings_count} warnings. Total penalty: ${summary.total_penalty_cost_usd:,.2f}"
            )
        else:
            logger.info(
                f"Design {design.design_id} passed safety validation. "
                f"Warnings: {summary.warnings_count}, Penalty: ${summary.total_penalty_cost_usd:,.2f}"
            )

        # Fail-closed behavior
        if self.fail_closed and summary.has_violations:
            raise SafetyViolationError(
                summary.get_rejection_reason(),
                violations=[
                    ViolationDetails(
                        constraint_tag=v["constraint_type"],
                        constraint_description=v.get("description", ""),
                        actual_value=v["actual_value"],
                        limit_value=v["limit_value"],
                        unit=v["unit"],
                        severity=ViolationSeverity(v["severity"]),
                        location=v["location"],
                        standard_reference=v["standard_reference"],
                        recommended_action=v["recommended_action"],
                    )
                    for v in summary.violations
                ],
                design_id=design.design_id,
            )

        return summary

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

    def _get_phase_specific_delta_t_min(
        self,
        hot_stream: Optional[HeatStream],
        cold_stream: Optional[HeatStream],
    ) -> float:
        """Get phase-specific minimum approach temperature."""
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

    def _calculate_penalty(
        self,
        actual_value: float,
        limit_value: float,
        is_minimum: bool,
        limit_spec: ConstraintLimit,
    ) -> Tuple[PenaltyLevel, float, float]:
        """
        Calculate penalty level and cost for a constraint check.

        Returns:
            Tuple of (penalty_level, margin_percent, penalty_cost_usd)
        """
        if is_minimum:
            # For minimum limits (e.g., delta_t_min, acid_dew_point)
            # margin > 0 means above limit (good), margin < 0 means below limit (bad)
            margin = actual_value - limit_value
            margin_percent = (margin / limit_value) * 100 if limit_value != 0 else 0

            if margin < 0:
                # Violation: below minimum
                penalty_cost = abs(margin) * limit_spec.penalty_cost_per_unit
                return PenaltyLevel.VIOLATION, margin_percent, penalty_cost

            # Calculate proximity to limit (how close to minimum)
            proximity = actual_value / limit_value if limit_value != 0 else float('inf')

            if proximity < 1.0 + (1.0 - limit_spec.danger_threshold):  # < 1.05
                penalty_cost = (limit_value * 0.05) * limit_spec.penalty_cost_per_unit * 0.5
                return PenaltyLevel.HIGH, margin_percent, penalty_cost
            elif proximity < 1.0 + (1.0 - limit_spec.caution_threshold):  # < 1.10
                penalty_cost = (limit_value * 0.10) * limit_spec.penalty_cost_per_unit * 0.25
                return PenaltyLevel.MEDIUM, margin_percent, penalty_cost
            elif proximity < 1.0 + (1.0 - limit_spec.warning_threshold):  # < 1.20
                penalty_cost = (limit_value * 0.20) * limit_spec.penalty_cost_per_unit * 0.1
                return PenaltyLevel.LOW, margin_percent, penalty_cost
            else:
                return PenaltyLevel.NONE, margin_percent, 0.0

        else:
            # For maximum limits (e.g., max_film_temp, max_pressure_drop)
            # margin > 0 means above limit (bad), margin < 0 means below limit (good)
            margin = actual_value - limit_value
            margin_percent = (margin / limit_value) * 100 if limit_value != 0 else 0

            if margin > 0:
                # Violation: above maximum
                penalty_cost = margin * limit_spec.penalty_cost_per_unit
                return PenaltyLevel.VIOLATION, margin_percent, penalty_cost

            # Calculate proximity to limit (what fraction of limit is actual)
            proximity = actual_value / limit_value if limit_value != 0 else 0

            if proximity >= limit_spec.danger_threshold:  # >= 95%
                penalty_cost = (limit_value * 0.05) * limit_spec.penalty_cost_per_unit * 0.5
                return PenaltyLevel.HIGH, margin_percent, penalty_cost
            elif proximity >= limit_spec.caution_threshold:  # >= 90%
                penalty_cost = (limit_value * 0.10) * limit_spec.penalty_cost_per_unit * 0.25
                return PenaltyLevel.MEDIUM, margin_percent, penalty_cost
            elif proximity >= limit_spec.warning_threshold:  # >= 80%
                penalty_cost = (limit_value * 0.20) * limit_spec.penalty_cost_per_unit * 0.1
                return PenaltyLevel.LOW, margin_percent, penalty_cost
            else:
                return PenaltyLevel.NONE, margin_percent, 0.0

    def _create_check_result(
        self,
        constraint_type: ConstraintType,
        exchanger_id: str,
        location: str,
        actual_value: float,
        limit_value: float,
        limit_spec: ConstraintLimit,
        recommended_action: str,
    ) -> ConstraintCheckResult:
        """Create a constraint check result with penalty calculation."""
        penalty_level, margin_percent, penalty_cost = self._calculate_penalty(
            actual_value, limit_value, limit_spec.is_minimum, limit_spec
        )

        is_violation = penalty_level == PenaltyLevel.VIOLATION

        # Determine severity
        if is_violation:
            # Check if it's a critical violation (temperature crossover or extreme)
            if constraint_type == ConstraintType.DELTA_T_MIN and actual_value < 0:
                severity = ViolationSeverity.CRITICAL
            elif abs(margin_percent) > 20:
                severity = ViolationSeverity.CRITICAL
            else:
                severity = ViolationSeverity.ERROR
        elif penalty_level == PenaltyLevel.HIGH:
            severity = ViolationSeverity.WARNING
        else:
            severity = ViolationSeverity.WARNING if penalty_level != PenaltyLevel.NONE else ViolationSeverity.WARNING

        # Compute calculation hash for provenance
        calc_data = {
            "constraint_type": constraint_type.value,
            "exchanger_id": exchanger_id,
            "location": location,
            "actual_value": actual_value,
            "limit_value": limit_value,
            "is_violation": is_violation,
        }
        calc_hash = self._compute_hash(calc_data)

        return ConstraintCheckResult(
            constraint_type=constraint_type,
            exchanger_id=exchanger_id,
            location=location,
            actual_value=round(actual_value, 4),
            limit_value=limit_value,
            unit=limit_spec.unit,
            margin_percent=round(margin_percent, 2),
            penalty_level=penalty_level,
            penalty_cost_usd=round(penalty_cost, 2) if self.apply_penalties else 0.0,
            is_violation=is_violation,
            severity=severity,
            standard_reference=limit_spec.standard_reference,
            recommended_action=recommended_action,
            calculation_hash=calc_hash,
        )

    def _check_delta_t_min(
        self,
        exchanger: HeatExchanger,
        hot_stream: Optional[HeatStream],
        cold_stream: Optional[HeatStream],
    ) -> List[ConstraintCheckResult]:
        """
        Check minimum approach temperature constraint (DELTA_T_MIN).

        The approach temperature is the temperature difference between
        the hot and cold streams at each end of the exchanger.
        A negative approach means temperature crossover (infeasible).
        """
        results = []
        limit_spec = self.limits[ConstraintType.DELTA_T_MIN]

        # Get phase-specific minimum
        phase_delta_t_min = self._get_phase_specific_delta_t_min(hot_stream, cold_stream)

        # Hot end approach: T_hot_in - T_cold_out
        hot_end_approach = exchanger.hot_inlet_T_C - exchanger.cold_outlet_T_C
        results.append(self._create_check_result(
            constraint_type=ConstraintType.DELTA_T_MIN,
            exchanger_id=exchanger.exchanger_id,
            location="hot_end",
            actual_value=hot_end_approach,
            limit_value=phase_delta_t_min,
            limit_spec=limit_spec,
            recommended_action=(
                f"Increase hot stream inlet temperature or decrease cold stream "
                f"outlet temperature by at least {max(0, phase_delta_t_min - hot_end_approach):.1f}C"
            ),
        ))

        # Cold end approach: T_hot_out - T_cold_in
        cold_end_approach = exchanger.hot_outlet_T_C - exchanger.cold_inlet_T_C
        results.append(self._create_check_result(
            constraint_type=ConstraintType.DELTA_T_MIN,
            exchanger_id=exchanger.exchanger_id,
            location="cold_end",
            actual_value=cold_end_approach,
            limit_value=phase_delta_t_min,
            limit_spec=limit_spec,
            recommended_action=(
                f"Increase hot stream outlet temperature or decrease cold stream "
                f"inlet temperature by at least {max(0, phase_delta_t_min - cold_end_approach):.1f}C"
            ),
        ))

        return results

    def _check_max_film_temperature(
        self,
        exchanger: HeatExchanger,
        hot_stream: Optional[HeatStream],
    ) -> List[ConstraintCheckResult]:
        """
        Check maximum film temperature constraint (MAX_FILM_TEMPERATURE).

        High film temperatures can cause coking, metallurgy degradation,
        and accelerated fouling.
        """
        results = []
        limit_spec = self.limits[ConstraintType.MAX_FILM_TEMPERATURE]
        max_film_temp = limit_spec.limit_value

        # Estimate film temperature (conservative: use hot inlet as upper bound)
        # More accurate calculation would require heat transfer coefficients
        estimated_film_temp = exchanger.hot_inlet_T_C

        results.append(self._create_check_result(
            constraint_type=ConstraintType.MAX_FILM_TEMPERATURE,
            exchanger_id=exchanger.exchanger_id,
            location="hot_side_inlet",
            actual_value=estimated_film_temp,
            limit_value=max_film_temp,
            limit_spec=limit_spec,
            recommended_action=(
                f"Reduce hot stream inlet temperature by {max(0, estimated_film_temp - max_film_temp):.1f}C, "
                f"or increase heat transfer area to reduce film temperature"
            ),
        ))

        return results

    def _check_acid_dew_point(
        self,
        exchanger: HeatExchanger,
        hot_stream: Optional[HeatStream],
    ) -> List[ConstraintCheckResult]:
        """
        Check acid dew point constraint (ACID_DEW_POINT).

        Only applies to flue gas / combustion gas streams.
        Outlet temperature below acid dew point causes acid condensation
        and severe corrosion.
        """
        results = []

        # Only check for gas streams (flue gas typically)
        if hot_stream is None:
            return results

        is_flue_gas = (
            (hasattr(hot_stream, 'phase') and hot_stream.phase == Phase.GAS) and
            hot_stream.fluid_name.lower() in ['flue_gas', 'exhaust', 'combustion_gas', 'stack_gas']
        )

        if not is_flue_gas:
            return results

        limit_spec = self.limits[ConstraintType.ACID_DEW_POINT]
        acid_dew_point = limit_spec.limit_value

        results.append(self._create_check_result(
            constraint_type=ConstraintType.ACID_DEW_POINT,
            exchanger_id=exchanger.exchanger_id,
            location="hot_side_outlet",
            actual_value=exchanger.hot_outlet_T_C,
            limit_value=acid_dew_point,
            limit_spec=limit_spec,
            recommended_action=(
                f"Increase hot stream outlet temperature to at least {acid_dew_point}C "
                f"to prevent acid condensation and cold-end corrosion"
            ),
        ))

        return results

    def _check_max_pressure_drop(
        self,
        exchanger: HeatExchanger,
        hot_stream: Optional[HeatStream],
        cold_stream: Optional[HeatStream],
    ) -> List[ConstraintCheckResult]:
        """
        Check maximum pressure drop constraint (MAX_PRESSURE_DROP).

        Different limits for liquids (50 kPa) and gases (5 kPa).
        """
        results = []
        limit_spec = self.limits[ConstraintType.MAX_PRESSURE_DROP]

        # Hot side pressure drop
        hot_is_gas = self._is_gas_phase(hot_stream)
        hot_limit = (
            self.constraints.max_pressure_drop_gas if hot_is_gas
            else self.constraints.max_pressure_drop_liquid
        )

        # Update limit spec for hot side
        hot_limit_spec = ConstraintLimit(
            constraint_type=ConstraintType.MAX_PRESSURE_DROP,
            limit_value=hot_limit,
            unit="kPa",
            is_minimum=False,
            standard_reference="API 660 Section 6.3, ISO 14414",
            description=f"Maximum pressure drop for {'gas' if hot_is_gas else 'liquid'}",
            penalty_cost_per_unit=limit_spec.penalty_cost_per_unit,
        )

        results.append(self._create_check_result(
            constraint_type=ConstraintType.MAX_PRESSURE_DROP,
            exchanger_id=exchanger.exchanger_id,
            location="hot_side",
            actual_value=exchanger.hot_side_dp_kPa,
            limit_value=hot_limit,
            limit_spec=hot_limit_spec,
            recommended_action=(
                f"Reduce hot side pressure drop by increasing flow area, "
                f"reducing flow velocity, or changing exchanger geometry"
            ),
        ))

        # Cold side pressure drop
        cold_is_gas = self._is_gas_phase(cold_stream)
        cold_limit = (
            self.constraints.max_pressure_drop_gas if cold_is_gas
            else self.constraints.max_pressure_drop_liquid
        )

        cold_limit_spec = ConstraintLimit(
            constraint_type=ConstraintType.MAX_PRESSURE_DROP,
            limit_value=cold_limit,
            unit="kPa",
            is_minimum=False,
            standard_reference="API 660 Section 6.3, ISO 14414",
            description=f"Maximum pressure drop for {'gas' if cold_is_gas else 'liquid'}",
            penalty_cost_per_unit=limit_spec.penalty_cost_per_unit,
        )

        results.append(self._create_check_result(
            constraint_type=ConstraintType.MAX_PRESSURE_DROP,
            exchanger_id=exchanger.exchanger_id,
            location="cold_side",
            actual_value=exchanger.cold_side_dp_kPa,
            limit_value=cold_limit,
            limit_spec=cold_limit_spec,
            recommended_action=(
                f"Reduce cold side pressure drop by increasing flow area, "
                f"reducing flow velocity, or changing exchanger geometry"
            ),
        ))

        return results

    def _check_thermal_stress_rate(
        self,
        exchanger: HeatExchanger,
        startup_time_minutes: float,
    ) -> List[ConstraintCheckResult]:
        """
        Check thermal stress rate constraint (THERMAL_STRESS_RATE).

        Maximum allowable temperature change rate during startup/shutdown
        to prevent thermal shock and equipment damage.
        """
        results = []
        limit_spec = self.limits[ConstraintType.THERMAL_STRESS_RATE]

        if startup_time_minutes <= 0:
            return results

        # Calculate temperature change rate for hot side
        hot_delta_t = abs(exchanger.hot_inlet_T_C - exchanger.hot_outlet_T_C)
        hot_stress_rate = hot_delta_t / startup_time_minutes

        results.append(self._create_check_result(
            constraint_type=ConstraintType.THERMAL_STRESS_RATE,
            exchanger_id=exchanger.exchanger_id,
            location="hot_side",
            actual_value=hot_stress_rate,
            limit_value=limit_spec.limit_value,
            limit_spec=limit_spec,
            recommended_action=(
                f"Increase startup time to at least {hot_delta_t / limit_spec.limit_value:.1f} minutes "
                f"or implement gradual temperature ramping"
            ),
        ))

        # Calculate temperature change rate for cold side
        cold_delta_t = abs(exchanger.cold_outlet_T_C - exchanger.cold_inlet_T_C)
        cold_stress_rate = cold_delta_t / startup_time_minutes

        results.append(self._create_check_result(
            constraint_type=ConstraintType.THERMAL_STRESS_RATE,
            exchanger_id=exchanger.exchanger_id,
            location="cold_side",
            actual_value=cold_stress_rate,
            limit_value=limit_spec.limit_value,
            limit_spec=limit_spec,
            recommended_action=(
                f"Increase startup time to at least {cold_delta_t / limit_spec.limit_value:.1f} minutes "
                f"or implement gradual temperature ramping"
            ),
        ))

        return results

    def _is_gas_phase(self, stream: Optional[HeatStream]) -> bool:
        """Check if stream is in gas phase."""
        if stream is None:
            return False
        if hasattr(stream, 'phase'):
            return stream.phase in [Phase.GAS, Phase.SUPERCRITICAL]
        return False

    def _build_summary(
        self,
        design: HENDesign,
        check_results: List[ConstraintCheckResult],
    ) -> ConstraintCheckSummary:
        """Build constraint check summary from individual results."""
        # Separate violations and near-violations
        violations = [r for r in check_results if r.is_violation]
        near_violations = [
            r for r in check_results
            if not r.is_violation and r.penalty_level != PenaltyLevel.NONE
        ]

        # Count by severity
        critical_count = sum(1 for r in check_results if r.severity == ViolationSeverity.CRITICAL)
        error_count = sum(1 for r in check_results if r.severity == ViolationSeverity.ERROR)
        warning_count = sum(1 for r in check_results if r.severity == ViolationSeverity.WARNING)

        # Calculate total penalty costs
        total_penalty = sum(r.penalty_cost_usd for r in check_results)

        # Calculate penalty by constraint type
        penalty_by_constraint: Dict[str, float] = {}
        for r in check_results:
            key = r.constraint_type.value
            penalty_by_constraint[key] = penalty_by_constraint.get(key, 0) + r.penalty_cost_usd

        # Compute design hash
        design_data = {
            "design_id": design.design_id,
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
        design_hash = self._compute_hash(design_data)

        # Convert results to dicts for summary
        check_result_dicts = [
            {
                "constraint_type": r.constraint_type.value,
                "exchanger_id": r.exchanger_id,
                "location": r.location,
                "actual_value": r.actual_value,
                "limit_value": r.limit_value,
                "unit": r.unit,
                "margin_percent": r.margin_percent,
                "penalty_level": r.penalty_level.value,
                "penalty_cost_usd": r.penalty_cost_usd,
                "is_violation": r.is_violation,
                "severity": r.severity.value,
                "standard_reference": r.standard_reference,
                "recommended_action": r.recommended_action,
                "calculation_hash": r.calculation_hash,
            }
            for r in check_results
        ]

        violation_dicts = [
            d for d in check_result_dicts if d["is_violation"]
        ]
        near_violation_dicts = [
            d for d in check_result_dicts
            if not d["is_violation"] and d["penalty_level"] != "none"
        ]

        # Build summary
        summary = ConstraintCheckSummary(
            design_id=design.design_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            is_acceptable=len(violations) == 0,
            has_violations=len(violations) > 0,
            has_near_violations=len(near_violations) > 0,
            checks_performed=len(check_results),
            violations_count=len(violations),
            warnings_count=len(near_violations),
            critical_count=critical_count,
            error_count=error_count,
            warning_count=warning_count,
            total_penalty_cost_usd=round(total_penalty, 2),
            penalty_cost_by_constraint=penalty_by_constraint,
            check_results=check_result_dicts,
            violations=violation_dicts,
            near_violations=near_violation_dicts,
            constraints_hash=self._constraints_hash,
            design_hash=design_hash,
        )

        # Compute result hash
        result_data = {
            "design_id": summary.design_id,
            "is_acceptable": summary.is_acceptable,
            "violations_count": summary.violations_count,
            "total_penalty": summary.total_penalty_cost_usd,
            "design_hash": design_hash,
            "constraints_hash": self._constraints_hash,
        }
        summary.result_hash = self._compute_hash(result_data)

        return summary

    def validate_single_exchanger(
        self,
        exchanger: HeatExchanger,
        hot_stream: Optional[HeatStream] = None,
        cold_stream: Optional[HeatStream] = None,
        startup_time_minutes: Optional[float] = None,
    ) -> ConstraintCheckSummary:
        """
        Validate a single heat exchanger against all constraints.

        Convenience method for quick validation during design iteration.

        Args:
            exchanger: Heat exchanger to validate
            hot_stream: Optional hot stream data
            cold_stream: Optional cold stream data
            startup_time_minutes: Optional startup time for thermal stress

        Returns:
            ConstraintCheckSummary for the single exchanger
        """
        # Create temporary design wrapper
        temp_design = HENDesign(
            design_id=f"single_{exchanger.exchanger_id}",
            exchangers=[exchanger],
            total_heat_recovered_kW=exchanger.duty_kW,
            hot_utility_required_kW=0.0,
            cold_utility_required_kW=0.0,
        )

        hot_streams = [hot_stream] if hot_stream else []
        cold_streams = [cold_stream] if cold_stream else []

        # Temporarily disable fail-closed for single exchanger validation
        original_fail_closed = self.fail_closed
        self.fail_closed = False

        try:
            return self.validate_hen_design(
                temp_design,
                hot_streams,
                cold_streams,
                startup_time_minutes=startup_time_minutes,
            )
        finally:
            self.fail_closed = original_fail_closed

    def get_constraint_limits(self) -> Dict[str, Dict[str, Any]]:
        """Get all constraint limits for transparency."""
        return {
            constraint_type.value: {
                "limit_value": limit.limit_value,
                "unit": limit.unit,
                "is_minimum": limit.is_minimum,
                "standard_reference": limit.standard_reference,
                "description": limit.description,
                "penalty_cost_per_unit": limit.penalty_cost_per_unit,
            }
            for constraint_type, limit in self.limits.items()
        }
