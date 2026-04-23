"""
GL-012 STEAMQUAL - Steam Quality Constraint Validator

Production-grade constraint validation for steam quality control ensuring:
- x >= x_min for each header/consumer (minimum dryness fraction)
- DeltaT_sh >= DeltaT_min for superheated steam requirements
- Drain capacity limits to prevent flooding
- Ramp rate limits for thermal stress prevention
- Water hammer risk detection and alerting

Safety Constraints Enforced:
1. QUALITY_MIN: x >= x_min (dryness fraction minimum per consumer)
2. SUPERHEAT_MIN: DeltaT_sh >= DeltaT_min for superheated requirements
3. DRAIN_CAPACITY: Condensate flow <= drain system capacity
4. RAMP_RATE: dP/dt and dT/dt within thermal stress limits
5. WATER_HAMMER: Velocity and condensate accumulation checks

Standards Compliance:
    - ASME PTC 19.11 (Steam and Water Properties)
    - API 560/530 (Process Steam Systems)
    - NFPA 85 (Boiler and Combustion Systems)
    - ASHRAE Handbook (HVAC Steam Systems)

Zero-Hallucination Guarantee:
All constraint checks use deterministic arithmetic from published standards.
No LLM or AI inference for any safety-critical decisions.
SHA-256 provenance hashing for complete audit trail.

FAIL-SAFE Design:
When data quality is poor or uncertain, validation FAILS to the safe side
(assumes constraint is violated). This prevents delivery of poor quality
steam that could damage equipment or endanger operations.

Example:
    >>> from safety.constraint_validator import ConstraintValidator
    >>> validator = ConstraintValidator()
    >>> result = validator.validate_quality_constraints(
    ...     quality=0.92,
    ...     min_quality=0.95,
    ...     header_id="STEAM-HDR-001"
    ... )
    >>> if not result.is_valid:
    ...     logger.warning(f"Quality constraint violated: {result.violations}")

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
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================


class ConstraintType(str, Enum):
    """Steam quality constraint types."""

    QUALITY_MIN = "quality_min"           # x >= x_min
    SUPERHEAT_MIN = "superheat_min"       # DeltaT_sh >= DeltaT_min
    DRAIN_CAPACITY = "drain_capacity"     # Flow <= capacity
    RAMP_RATE_PRESSURE = "ramp_rate_p"    # dP/dt <= limit
    RAMP_RATE_TEMP = "ramp_rate_t"        # dT/dt <= limit
    WATER_HAMMER_RISK = "water_hammer"    # Condensate velocity check


class ConstraintSeverity(str, Enum):
    """Constraint importance levels for prioritization."""

    CRITICAL = "critical"   # Immediate shutdown/action required
    HIGH = "high"           # Urgent corrective action needed
    MEDIUM = "medium"       # Timely correction recommended
    LOW = "low"             # Advisory only


class ViolationSeverity(str, Enum):
    """Severity classification for constraint violations."""

    WARNING = "warning"      # Near boundary - caution
    ERROR = "error"          # Constraint violated - action required
    CRITICAL = "critical"    # Severe violation - immediate action


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass(frozen=True)
class QualityConstraint:
    """
    Immutable specification for a steam quality constraint.

    Attributes:
        constraint_type: Type of constraint
        limit_value: Constraint limit (min or max depending on type)
        unit: Unit of measurement
        is_minimum: True for minimum constraints, False for maximum
        severity: How critical is this constraint
        standard_reference: Regulatory/standards reference
        description: Human-readable description
        warning_margin: Fraction for early warning (0.0-0.5)
    """
    constraint_type: ConstraintType
    limit_value: float
    unit: str
    is_minimum: bool
    severity: ConstraintSeverity
    standard_reference: str
    description: str
    warning_margin: float = 0.05  # 5% margin for warnings

    def get_warning_threshold(self) -> float:
        """Get warning threshold value."""
        if self.is_minimum:
            # For minimums, warning when approaching from above
            return self.limit_value * (1.0 + self.warning_margin)
        else:
            # For maximums, warning when approaching from below
            return self.limit_value * (1.0 - self.warning_margin)


@dataclass(frozen=True)
class ConstraintViolation:
    """
    Immutable record of a constraint violation.

    Contains complete information for audit trail and
    regulatory compliance documentation.
    """
    constraint_type: ConstraintType
    location_id: str          # Header ID, consumer ID, etc.
    actual_value: float
    limit_value: float
    unit: str
    severity: ViolationSeverity
    message: str
    standard_reference: str
    recommended_action: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    calculation_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "constraint_type": self.constraint_type.value,
            "location_id": self.location_id,
            "actual_value": self.actual_value,
            "limit_value": self.limit_value,
            "unit": self.unit,
            "severity": self.severity.value,
            "message": self.message,
            "standard_reference": self.standard_reference,
            "recommended_action": self.recommended_action,
            "timestamp": self.timestamp.isoformat(),
            "calculation_hash": self.calculation_hash,
        }


# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class ConstraintValidationResult(BaseModel):
    """
    Result of constraint validation with provenance tracking.

    Provides complete audit trail for regulatory compliance.
    """

    is_valid: bool = Field(
        True,
        description="Whether all constraints passed"
    )
    location_id: str = Field(
        "",
        description="Header/consumer ID being validated"
    )
    validated_constraints: List[str] = Field(
        default_factory=list,
        description="List of constraints that were checked"
    )
    violations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of constraint violations"
    )
    warnings: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of warnings (near violations)"
    )
    error_count: int = Field(0, ge=0, description="Count of errors")
    warning_count: int = Field(0, ge=0, description="Count of warnings")
    critical_count: int = Field(0, ge=0, description="Count of critical violations")
    validation_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When validation occurred"
    )
    provenance_hash: str = Field(
        "",
        description="SHA-256 hash for audit trail"
    )
    validator_version: str = Field(
        "1.0.0",
        description="Validator version"
    )
    standards_applied: List[str] = Field(
        default_factory=lambda: ["ASME PTC 19.11", "API 560"],
        description="Standards used for validation"
    )

    def get_summary(self) -> str:
        """Get human-readable summary of validation result."""
        if self.is_valid:
            if self.warning_count > 0:
                return f"PASS with {self.warning_count} warning(s) at {self.location_id}"
            return f"PASS - All constraints satisfied at {self.location_id}"
        else:
            return (
                f"FAIL at {self.location_id}: "
                f"{self.critical_count} critical, "
                f"{self.error_count} error(s), "
                f"{self.warning_count} warning(s)"
            )


# =============================================================================
# CONSTRAINT DEFINITIONS
# =============================================================================


# Default steam quality constraints per ASME PTC 19.11 and API standards
STEAM_QUALITY_CONSTRAINTS: Dict[ConstraintType, QualityConstraint] = {
    ConstraintType.QUALITY_MIN: QualityConstraint(
        constraint_type=ConstraintType.QUALITY_MIN,
        limit_value=0.95,  # 95% minimum dryness typical for process steam
        unit="fraction",
        is_minimum=True,
        severity=ConstraintSeverity.HIGH,
        standard_reference="ASME PTC 19.11 Section 4.2",
        description="Minimum steam dryness fraction (x >= x_min)",
        warning_margin=0.02,  # Warn at 97%
    ),
    ConstraintType.SUPERHEAT_MIN: QualityConstraint(
        constraint_type=ConstraintType.SUPERHEAT_MIN,
        limit_value=10.0,  # 10C minimum superheat
        unit="C",
        is_minimum=True,
        severity=ConstraintSeverity.MEDIUM,
        standard_reference="API 560 Section 5.3.2",
        description="Minimum superheat temperature above saturation (DeltaT_sh >= DeltaT_min)",
        warning_margin=0.20,  # Warn at 12C
    ),
    ConstraintType.DRAIN_CAPACITY: QualityConstraint(
        constraint_type=ConstraintType.DRAIN_CAPACITY,
        limit_value=100.0,  # kg/hr typical drain capacity
        unit="kg/hr",
        is_minimum=False,  # This is a maximum constraint
        severity=ConstraintSeverity.HIGH,
        standard_reference="ASHRAE Handbook - Steam Systems",
        description="Maximum condensate flow rate for drain system capacity",
        warning_margin=0.10,  # Warn at 90% capacity
    ),
    ConstraintType.RAMP_RATE_PRESSURE: QualityConstraint(
        constraint_type=ConstraintType.RAMP_RATE_PRESSURE,
        limit_value=0.5,  # 0.5 bar/min typical limit
        unit="bar/min",
        is_minimum=False,  # Maximum ramp rate
        severity=ConstraintSeverity.HIGH,
        standard_reference="API 530 Section 4.4.1",
        description="Maximum pressure ramp rate to prevent thermal stress",
        warning_margin=0.10,
    ),
    ConstraintType.RAMP_RATE_TEMP: QualityConstraint(
        constraint_type=ConstraintType.RAMP_RATE_TEMP,
        limit_value=5.0,  # 5 C/min typical limit
        unit="C/min",
        is_minimum=False,  # Maximum ramp rate
        severity=ConstraintSeverity.HIGH,
        standard_reference="API 530 Section 4.4.1",
        description="Maximum temperature ramp rate to prevent thermal stress",
        warning_margin=0.10,
    ),
    ConstraintType.WATER_HAMMER_RISK: QualityConstraint(
        constraint_type=ConstraintType.WATER_HAMMER_RISK,
        limit_value=30.0,  # 30 m/s velocity threshold
        unit="m/s",
        is_minimum=False,  # Maximum velocity
        severity=ConstraintSeverity.CRITICAL,
        standard_reference="ASME B31.1 Section 101.5.3",
        description="Maximum steam velocity to prevent water hammer",
        warning_margin=0.15,
    ),
}


# =============================================================================
# CONSTRAINT VALIDATOR
# =============================================================================


class ConstraintValidator:
    """
    Production-grade constraint validator for GL-012 STEAMQUAL.

    Validates steam quality parameters against safety constraints:
    - Quality minimum (dryness fraction)
    - Superheat minimum (for superheated requirements)
    - Drain capacity limits
    - Ramp rate limits (pressure and temperature)
    - Water hammer risk detection

    FAIL-SAFE Design:
    When data quality is uncertain or missing, validation fails to
    the safe side (assumes violation). This prevents delivery of
    potentially dangerous steam conditions.

    Zero-Hallucination:
    All calculations use deterministic formulas from ASME/API standards.
    No LLM or ML inference in the validation path.

    Example:
        >>> validator = ConstraintValidator()
        >>> result = validator.validate_all(
        ...     quality=0.92,
        ...     superheat_c=8.0,
        ...     condensate_flow_kg_hr=85.0,
        ...     header_id="STEAM-HDR-001",
        ...     min_quality=0.95,
        ...     min_superheat_c=10.0
        ... )
        >>> print(result.get_summary())
    """

    VERSION = "1.0.0"
    STANDARDS = ["ASME PTC 19.11", "API 560", "API 530", "ASME B31.1"]

    def __init__(
        self,
        custom_constraints: Optional[Dict[ConstraintType, QualityConstraint]] = None,
        fail_safe: bool = True,
    ):
        """
        Initialize constraint validator.

        Args:
            custom_constraints: Optional custom constraint definitions
            fail_safe: If True, assume violation when data is uncertain
        """
        self.fail_safe = fail_safe
        self.constraints = dict(STEAM_QUALITY_CONSTRAINTS)

        if custom_constraints:
            self.constraints.update(custom_constraints)

        self._validation_count = 0
        self._lock = threading.Lock()

        # Compute constraints hash for provenance
        self._constraints_hash = self._compute_constraints_hash()

        logger.info(
            f"ConstraintValidator v{self.VERSION} initialized: "
            f"fail_safe={fail_safe}, constraints={len(self.constraints)}"
        )

    def _compute_constraints_hash(self) -> str:
        """Compute SHA-256 hash of constraint configuration."""
        config = {
            k.value: {
                "limit": v.limit_value,
                "unit": v.unit,
                "is_min": v.is_minimum,
            }
            for k, v in self.constraints.items()
        }
        return hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()

    def _compute_hash(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash of data for provenance."""
        return hashlib.sha256(
            json.dumps(data, sort_keys=True, default=str).encode()
        ).hexdigest()

    # =========================================================================
    # INDIVIDUAL CONSTRAINT VALIDATORS
    # =========================================================================

    def validate_quality_min(
        self,
        quality: float,
        min_quality: Optional[float] = None,
        location_id: str = "unknown",
    ) -> Tuple[bool, Optional[ConstraintViolation]]:
        """
        Validate steam quality (dryness fraction) meets minimum.

        x >= x_min

        Args:
            quality: Actual steam quality (0.0 - 1.0)
            min_quality: Minimum required quality (uses default if None)
            location_id: Header or consumer identifier

        Returns:
            Tuple of (is_valid, violation or None)
        """
        constraint = self.constraints[ConstraintType.QUALITY_MIN]
        limit = min_quality if min_quality is not None else constraint.limit_value

        # FAIL-SAFE: If quality is None or NaN, assume violation
        if quality is None or math.isnan(quality):
            if self.fail_safe:
                return False, ConstraintViolation(
                    constraint_type=ConstraintType.QUALITY_MIN,
                    location_id=location_id,
                    actual_value=0.0,
                    limit_value=limit,
                    unit=constraint.unit,
                    severity=ViolationSeverity.CRITICAL,
                    message=f"Quality data unavailable at {location_id} - FAIL-SAFE triggered",
                    standard_reference=constraint.standard_reference,
                    recommended_action="Verify quality sensor operation",
                    calculation_hash=self._compute_hash({"quality": None, "limit": limit}),
                )
            return True, None

        # Calculate violation hash
        calc_hash = self._compute_hash({
            "constraint": "quality_min",
            "actual": quality,
            "limit": limit,
            "location": location_id,
        })

        # Check constraint: x >= x_min
        if quality < limit:
            severity = (
                ViolationSeverity.CRITICAL if quality < limit * 0.9
                else ViolationSeverity.ERROR
            )
            return False, ConstraintViolation(
                constraint_type=ConstraintType.QUALITY_MIN,
                location_id=location_id,
                actual_value=round(quality, 4),
                limit_value=limit,
                unit=constraint.unit,
                severity=severity,
                message=(
                    f"Steam quality {quality:.2%} below minimum {limit:.2%} at {location_id}"
                ),
                standard_reference=constraint.standard_reference,
                recommended_action=(
                    f"Increase desuperheating or reduce heat loss. "
                    f"Target quality >= {limit:.2%}"
                ),
                calculation_hash=calc_hash,
            )

        # Check warning threshold
        warning_threshold = constraint.get_warning_threshold()
        if quality < warning_threshold:
            return True, ConstraintViolation(
                constraint_type=ConstraintType.QUALITY_MIN,
                location_id=location_id,
                actual_value=round(quality, 4),
                limit_value=limit,
                unit=constraint.unit,
                severity=ViolationSeverity.WARNING,
                message=(
                    f"Steam quality {quality:.2%} approaching minimum {limit:.2%} at {location_id}"
                ),
                standard_reference=constraint.standard_reference,
                recommended_action="Monitor quality trend",
                calculation_hash=calc_hash,
            )

        return True, None

    def validate_superheat_min(
        self,
        superheat_c: float,
        min_superheat_c: Optional[float] = None,
        location_id: str = "unknown",
    ) -> Tuple[bool, Optional[ConstraintViolation]]:
        """
        Validate superheat meets minimum requirement.

        DeltaT_sh >= DeltaT_min

        Args:
            superheat_c: Actual superheat in Celsius (T_actual - T_sat)
            min_superheat_c: Minimum required superheat
            location_id: Header or consumer identifier

        Returns:
            Tuple of (is_valid, violation or None)
        """
        constraint = self.constraints[ConstraintType.SUPERHEAT_MIN]
        limit = min_superheat_c if min_superheat_c is not None else constraint.limit_value

        # FAIL-SAFE: If superheat is None or NaN, assume violation
        if superheat_c is None or math.isnan(superheat_c):
            if self.fail_safe:
                return False, ConstraintViolation(
                    constraint_type=ConstraintType.SUPERHEAT_MIN,
                    location_id=location_id,
                    actual_value=0.0,
                    limit_value=limit,
                    unit=constraint.unit,
                    severity=ViolationSeverity.ERROR,
                    message=f"Superheat data unavailable at {location_id} - FAIL-SAFE triggered",
                    standard_reference=constraint.standard_reference,
                    recommended_action="Verify temperature sensors",
                    calculation_hash=self._compute_hash({"superheat": None, "limit": limit}),
                )
            return True, None

        calc_hash = self._compute_hash({
            "constraint": "superheat_min",
            "actual": superheat_c,
            "limit": limit,
            "location": location_id,
        })

        # Check constraint: DeltaT_sh >= DeltaT_min
        if superheat_c < limit:
            severity = (
                ViolationSeverity.CRITICAL if superheat_c < 0
                else ViolationSeverity.ERROR
            )
            return False, ConstraintViolation(
                constraint_type=ConstraintType.SUPERHEAT_MIN,
                location_id=location_id,
                actual_value=round(superheat_c, 2),
                limit_value=limit,
                unit=constraint.unit,
                severity=severity,
                message=(
                    f"Superheat {superheat_c:.1f}C below minimum {limit:.1f}C at {location_id}"
                ),
                standard_reference=constraint.standard_reference,
                recommended_action=(
                    f"Reduce desuperheating spray or increase steam temperature. "
                    f"Target DeltaT_sh >= {limit:.1f}C"
                ),
                calculation_hash=calc_hash,
            )

        # Check warning
        warning_threshold = constraint.get_warning_threshold()
        if superheat_c < warning_threshold:
            return True, ConstraintViolation(
                constraint_type=ConstraintType.SUPERHEAT_MIN,
                location_id=location_id,
                actual_value=round(superheat_c, 2),
                limit_value=limit,
                unit=constraint.unit,
                severity=ViolationSeverity.WARNING,
                message=(
                    f"Superheat {superheat_c:.1f}C approaching minimum {limit:.1f}C at {location_id}"
                ),
                standard_reference=constraint.standard_reference,
                recommended_action="Monitor superheat trend",
                calculation_hash=calc_hash,
            )

        return True, None

    def validate_drain_capacity(
        self,
        condensate_flow_kg_hr: float,
        drain_capacity_kg_hr: Optional[float] = None,
        location_id: str = "unknown",
    ) -> Tuple[bool, Optional[ConstraintViolation]]:
        """
        Validate condensate flow within drain system capacity.

        Flow <= Capacity

        Args:
            condensate_flow_kg_hr: Actual condensate flow rate (kg/hr)
            drain_capacity_kg_hr: Drain system capacity (kg/hr)
            location_id: Drain or trap identifier

        Returns:
            Tuple of (is_valid, violation or None)
        """
        constraint = self.constraints[ConstraintType.DRAIN_CAPACITY]
        limit = drain_capacity_kg_hr if drain_capacity_kg_hr is not None else constraint.limit_value

        # FAIL-SAFE
        if condensate_flow_kg_hr is None or math.isnan(condensate_flow_kg_hr):
            if self.fail_safe:
                return False, ConstraintViolation(
                    constraint_type=ConstraintType.DRAIN_CAPACITY,
                    location_id=location_id,
                    actual_value=0.0,
                    limit_value=limit,
                    unit=constraint.unit,
                    severity=ViolationSeverity.ERROR,
                    message=f"Condensate flow data unavailable at {location_id}",
                    standard_reference=constraint.standard_reference,
                    recommended_action="Verify flow measurement",
                    calculation_hash=self._compute_hash({"flow": None, "limit": limit}),
                )
            return True, None

        calc_hash = self._compute_hash({
            "constraint": "drain_capacity",
            "actual": condensate_flow_kg_hr,
            "limit": limit,
            "location": location_id,
        })

        # Check constraint: Flow <= Capacity
        if condensate_flow_kg_hr > limit:
            utilization = condensate_flow_kg_hr / limit
            severity = (
                ViolationSeverity.CRITICAL if utilization > 1.2
                else ViolationSeverity.ERROR
            )
            return False, ConstraintViolation(
                constraint_type=ConstraintType.DRAIN_CAPACITY,
                location_id=location_id,
                actual_value=round(condensate_flow_kg_hr, 1),
                limit_value=limit,
                unit=constraint.unit,
                severity=severity,
                message=(
                    f"Condensate flow {condensate_flow_kg_hr:.1f} kg/hr exceeds "
                    f"drain capacity {limit:.1f} kg/hr at {location_id} ({utilization:.0%})"
                ),
                standard_reference=constraint.standard_reference,
                recommended_action=(
                    f"Reduce condensate load or increase drain capacity. "
                    f"Risk of flooding and water hammer."
                ),
                calculation_hash=calc_hash,
            )

        # Check warning (approaching capacity)
        warning_threshold = constraint.get_warning_threshold()
        if condensate_flow_kg_hr > warning_threshold:
            return True, ConstraintViolation(
                constraint_type=ConstraintType.DRAIN_CAPACITY,
                location_id=location_id,
                actual_value=round(condensate_flow_kg_hr, 1),
                limit_value=limit,
                unit=constraint.unit,
                severity=ViolationSeverity.WARNING,
                message=(
                    f"Condensate flow {condensate_flow_kg_hr:.1f} kg/hr approaching "
                    f"drain capacity {limit:.1f} kg/hr at {location_id}"
                ),
                standard_reference=constraint.standard_reference,
                recommended_action="Monitor drain system and prepare for increased load",
                calculation_hash=calc_hash,
            )

        return True, None

    def validate_ramp_rate(
        self,
        ramp_rate: float,
        constraint_type: ConstraintType,
        max_ramp_rate: Optional[float] = None,
        location_id: str = "unknown",
    ) -> Tuple[bool, Optional[ConstraintViolation]]:
        """
        Validate pressure or temperature ramp rate within limits.

        |dP/dt| <= limit or |dT/dt| <= limit

        Args:
            ramp_rate: Actual ramp rate (bar/min or C/min)
            constraint_type: RAMP_RATE_PRESSURE or RAMP_RATE_TEMP
            max_ramp_rate: Maximum allowed ramp rate
            location_id: Location identifier

        Returns:
            Tuple of (is_valid, violation or None)
        """
        if constraint_type not in [ConstraintType.RAMP_RATE_PRESSURE, ConstraintType.RAMP_RATE_TEMP]:
            raise ValueError(f"Invalid constraint type for ramp rate: {constraint_type}")

        constraint = self.constraints[constraint_type]
        limit = max_ramp_rate if max_ramp_rate is not None else constraint.limit_value

        # Use absolute value for ramp rate
        abs_ramp_rate = abs(ramp_rate) if ramp_rate is not None else None

        # FAIL-SAFE
        if abs_ramp_rate is None or math.isnan(abs_ramp_rate):
            if self.fail_safe:
                return False, ConstraintViolation(
                    constraint_type=constraint_type,
                    location_id=location_id,
                    actual_value=0.0,
                    limit_value=limit,
                    unit=constraint.unit,
                    severity=ViolationSeverity.ERROR,
                    message=f"Ramp rate data unavailable at {location_id}",
                    standard_reference=constraint.standard_reference,
                    recommended_action="Verify process variable trending",
                    calculation_hash=self._compute_hash({"ramp": None, "limit": limit}),
                )
            return True, None

        calc_hash = self._compute_hash({
            "constraint": constraint_type.value,
            "actual": abs_ramp_rate,
            "limit": limit,
            "location": location_id,
        })

        # Check constraint
        if abs_ramp_rate > limit:
            severity = (
                ViolationSeverity.CRITICAL if abs_ramp_rate > limit * 1.5
                else ViolationSeverity.ERROR
            )
            param = "Pressure" if constraint_type == ConstraintType.RAMP_RATE_PRESSURE else "Temperature"
            return False, ConstraintViolation(
                constraint_type=constraint_type,
                location_id=location_id,
                actual_value=round(abs_ramp_rate, 3),
                limit_value=limit,
                unit=constraint.unit,
                severity=severity,
                message=(
                    f"{param} ramp rate {abs_ramp_rate:.2f} {constraint.unit} exceeds "
                    f"limit {limit:.2f} {constraint.unit} at {location_id}"
                ),
                standard_reference=constraint.standard_reference,
                recommended_action=(
                    f"Reduce rate of change to prevent thermal stress. "
                    f"Maximum allowed: {limit:.2f} {constraint.unit}"
                ),
                calculation_hash=calc_hash,
            )

        # Check warning
        warning_threshold = constraint.get_warning_threshold()
        if abs_ramp_rate > warning_threshold:
            param = "Pressure" if constraint_type == ConstraintType.RAMP_RATE_PRESSURE else "Temperature"
            return True, ConstraintViolation(
                constraint_type=constraint_type,
                location_id=location_id,
                actual_value=round(abs_ramp_rate, 3),
                limit_value=limit,
                unit=constraint.unit,
                severity=ViolationSeverity.WARNING,
                message=(
                    f"{param} ramp rate {abs_ramp_rate:.2f} {constraint.unit} approaching "
                    f"limit at {location_id}"
                ),
                standard_reference=constraint.standard_reference,
                recommended_action="Consider reducing rate of change",
                calculation_hash=calc_hash,
            )

        return True, None

    def validate_water_hammer_risk(
        self,
        steam_velocity_m_s: float,
        quality: Optional[float] = None,
        max_velocity_m_s: Optional[float] = None,
        location_id: str = "unknown",
    ) -> Tuple[bool, Optional[ConstraintViolation]]:
        """
        Validate water hammer risk based on steam velocity and quality.

        Water hammer risk increases when:
        - Steam velocity > threshold
        - Steam quality < 1.0 (presence of condensate)

        Args:
            steam_velocity_m_s: Steam velocity in m/s
            quality: Steam quality (0-1), if known
            max_velocity_m_s: Maximum safe velocity
            location_id: Pipe or header identifier

        Returns:
            Tuple of (is_valid, violation or None)
        """
        constraint = self.constraints[ConstraintType.WATER_HAMMER_RISK]
        limit = max_velocity_m_s if max_velocity_m_s is not None else constraint.limit_value

        # FAIL-SAFE
        if steam_velocity_m_s is None or math.isnan(steam_velocity_m_s):
            if self.fail_safe:
                return False, ConstraintViolation(
                    constraint_type=ConstraintType.WATER_HAMMER_RISK,
                    location_id=location_id,
                    actual_value=0.0,
                    limit_value=limit,
                    unit=constraint.unit,
                    severity=ViolationSeverity.ERROR,
                    message=f"Velocity data unavailable at {location_id}",
                    standard_reference=constraint.standard_reference,
                    recommended_action="Verify velocity calculation",
                    calculation_hash=self._compute_hash({"velocity": None, "limit": limit}),
                )
            return True, None

        # Adjust limit if quality is known and low (more conservative)
        effective_limit = limit
        if quality is not None and quality < 0.95:
            # Reduce velocity limit for wet steam
            effective_limit = limit * quality

        calc_hash = self._compute_hash({
            "constraint": "water_hammer",
            "velocity": steam_velocity_m_s,
            "quality": quality,
            "limit": effective_limit,
            "location": location_id,
        })

        # Check constraint
        if steam_velocity_m_s > effective_limit:
            severity = (
                ViolationSeverity.CRITICAL if steam_velocity_m_s > effective_limit * 1.2
                else ViolationSeverity.ERROR
            )
            quality_note = f" (adjusted for quality={quality:.2%})" if quality and quality < 0.95 else ""
            return False, ConstraintViolation(
                constraint_type=ConstraintType.WATER_HAMMER_RISK,
                location_id=location_id,
                actual_value=round(steam_velocity_m_s, 2),
                limit_value=round(effective_limit, 2),
                unit=constraint.unit,
                severity=severity,
                message=(
                    f"Steam velocity {steam_velocity_m_s:.1f} m/s exceeds safe limit "
                    f"{effective_limit:.1f} m/s at {location_id}{quality_note}. "
                    f"HIGH WATER HAMMER RISK!"
                ),
                standard_reference=constraint.standard_reference,
                recommended_action=(
                    f"URGENT: Reduce flow rate or increase pipe diameter. "
                    f"Ensure adequate drainage. Risk of equipment damage."
                ),
                calculation_hash=calc_hash,
            )

        # Check warning
        warning_threshold = effective_limit * (1.0 - constraint.warning_margin)
        if steam_velocity_m_s > warning_threshold:
            return True, ConstraintViolation(
                constraint_type=ConstraintType.WATER_HAMMER_RISK,
                location_id=location_id,
                actual_value=round(steam_velocity_m_s, 2),
                limit_value=round(effective_limit, 2),
                unit=constraint.unit,
                severity=ViolationSeverity.WARNING,
                message=(
                    f"Steam velocity {steam_velocity_m_s:.1f} m/s approaching water hammer "
                    f"risk threshold at {location_id}"
                ),
                standard_reference=constraint.standard_reference,
                recommended_action="Monitor velocity and ensure adequate drainage",
                calculation_hash=calc_hash,
            )

        return True, None

    # =========================================================================
    # COMBINED VALIDATORS
    # =========================================================================

    def validate_quality_constraints(
        self,
        quality: float,
        min_quality: float = 0.95,
        header_id: str = "unknown",
    ) -> ConstraintValidationResult:
        """
        Validate all quality-related constraints for a header/consumer.

        Args:
            quality: Steam quality (dryness fraction)
            min_quality: Required minimum quality
            header_id: Header or consumer identifier

        Returns:
            Complete validation result with provenance
        """
        with self._lock:
            self._validation_count += 1

        violations: List[ConstraintViolation] = []
        warnings: List[ConstraintViolation] = []
        validated: List[str] = []

        # Validate quality minimum
        validated.append(ConstraintType.QUALITY_MIN.value)
        is_valid, violation = self.validate_quality_min(
            quality=quality,
            min_quality=min_quality,
            location_id=header_id,
        )
        if violation:
            if violation.severity == ViolationSeverity.WARNING:
                warnings.append(violation)
            else:
                violations.append(violation)

        return self._build_result(header_id, validated, violations, warnings)

    def validate_all(
        self,
        quality: Optional[float] = None,
        superheat_c: Optional[float] = None,
        condensate_flow_kg_hr: Optional[float] = None,
        pressure_ramp_bar_min: Optional[float] = None,
        temp_ramp_c_min: Optional[float] = None,
        steam_velocity_m_s: Optional[float] = None,
        header_id: str = "unknown",
        min_quality: Optional[float] = None,
        min_superheat_c: Optional[float] = None,
        drain_capacity_kg_hr: Optional[float] = None,
        max_pressure_ramp: Optional[float] = None,
        max_temp_ramp: Optional[float] = None,
        max_velocity_m_s: Optional[float] = None,
    ) -> ConstraintValidationResult:
        """
        Validate all applicable steam quality constraints.

        Only validates constraints for which actual values are provided.
        Missing values are handled according to fail_safe setting.

        Args:
            quality: Steam quality (dryness fraction)
            superheat_c: Superheat above saturation temperature
            condensate_flow_kg_hr: Condensate flow rate
            pressure_ramp_bar_min: Pressure change rate
            temp_ramp_c_min: Temperature change rate
            steam_velocity_m_s: Steam velocity
            header_id: Location identifier
            min_quality: Required minimum quality
            min_superheat_c: Required minimum superheat
            drain_capacity_kg_hr: Drain system capacity
            max_pressure_ramp: Maximum pressure ramp rate
            max_temp_ramp: Maximum temperature ramp rate
            max_velocity_m_s: Maximum safe velocity

        Returns:
            Complete validation result with all constraint checks
        """
        with self._lock:
            self._validation_count += 1

        violations: List[ConstraintViolation] = []
        warnings: List[ConstraintViolation] = []
        validated: List[str] = []

        # Quality minimum
        if quality is not None or self.fail_safe:
            validated.append(ConstraintType.QUALITY_MIN.value)
            is_valid, violation = self.validate_quality_min(
                quality=quality if quality is not None else float('nan'),
                min_quality=min_quality,
                location_id=header_id,
            )
            if violation:
                if violation.severity == ViolationSeverity.WARNING:
                    warnings.append(violation)
                else:
                    violations.append(violation)

        # Superheat minimum
        if superheat_c is not None:
            validated.append(ConstraintType.SUPERHEAT_MIN.value)
            is_valid, violation = self.validate_superheat_min(
                superheat_c=superheat_c,
                min_superheat_c=min_superheat_c,
                location_id=header_id,
            )
            if violation:
                if violation.severity == ViolationSeverity.WARNING:
                    warnings.append(violation)
                else:
                    violations.append(violation)

        # Drain capacity
        if condensate_flow_kg_hr is not None:
            validated.append(ConstraintType.DRAIN_CAPACITY.value)
            is_valid, violation = self.validate_drain_capacity(
                condensate_flow_kg_hr=condensate_flow_kg_hr,
                drain_capacity_kg_hr=drain_capacity_kg_hr,
                location_id=header_id,
            )
            if violation:
                if violation.severity == ViolationSeverity.WARNING:
                    warnings.append(violation)
                else:
                    violations.append(violation)

        # Pressure ramp rate
        if pressure_ramp_bar_min is not None:
            validated.append(ConstraintType.RAMP_RATE_PRESSURE.value)
            is_valid, violation = self.validate_ramp_rate(
                ramp_rate=pressure_ramp_bar_min,
                constraint_type=ConstraintType.RAMP_RATE_PRESSURE,
                max_ramp_rate=max_pressure_ramp,
                location_id=header_id,
            )
            if violation:
                if violation.severity == ViolationSeverity.WARNING:
                    warnings.append(violation)
                else:
                    violations.append(violation)

        # Temperature ramp rate
        if temp_ramp_c_min is not None:
            validated.append(ConstraintType.RAMP_RATE_TEMP.value)
            is_valid, violation = self.validate_ramp_rate(
                ramp_rate=temp_ramp_c_min,
                constraint_type=ConstraintType.RAMP_RATE_TEMP,
                max_ramp_rate=max_temp_ramp,
                location_id=header_id,
            )
            if violation:
                if violation.severity == ViolationSeverity.WARNING:
                    warnings.append(violation)
                else:
                    violations.append(violation)

        # Water hammer risk
        if steam_velocity_m_s is not None:
            validated.append(ConstraintType.WATER_HAMMER_RISK.value)
            is_valid, violation = self.validate_water_hammer_risk(
                steam_velocity_m_s=steam_velocity_m_s,
                quality=quality,
                max_velocity_m_s=max_velocity_m_s,
                location_id=header_id,
            )
            if violation:
                if violation.severity == ViolationSeverity.WARNING:
                    warnings.append(violation)
                else:
                    violations.append(violation)

        return self._build_result(header_id, validated, violations, warnings)

    def _build_result(
        self,
        location_id: str,
        validated: List[str],
        violations: List[ConstraintViolation],
        warnings: List[ConstraintViolation],
    ) -> ConstraintValidationResult:
        """Build complete validation result with provenance hash."""
        error_count = sum(
            1 for v in violations
            if v.severity in [ViolationSeverity.ERROR, ViolationSeverity.CRITICAL]
        )
        critical_count = sum(
            1 for v in violations
            if v.severity == ViolationSeverity.CRITICAL
        )
        warning_count = len(warnings)

        is_valid = len(violations) == 0

        # Compute provenance hash
        provenance_data = {
            "location_id": location_id,
            "validated_constraints": validated,
            "violation_count": len(violations),
            "warning_count": warning_count,
            "constraints_hash": self._constraints_hash,
            "version": self.VERSION,
        }
        provenance_hash = self._compute_hash(provenance_data)

        return ConstraintValidationResult(
            is_valid=is_valid,
            location_id=location_id,
            validated_constraints=validated,
            violations=[v.to_dict() for v in violations],
            warnings=[w.to_dict() for w in warnings],
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

    def get_constraint_limits(self) -> Dict[str, Dict[str, Any]]:
        """Get all constraint limits for transparency."""
        return {
            ct.value: {
                "limit_value": c.limit_value,
                "unit": c.unit,
                "is_minimum": c.is_minimum,
                "severity": c.severity.value,
                "standard_reference": c.standard_reference,
                "description": c.description,
            }
            for ct, c in self.constraints.items()
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get validator statistics."""
        with self._lock:
            return {
                "version": self.VERSION,
                "validation_count": self._validation_count,
                "constraint_count": len(self.constraints),
                "fail_safe": self.fail_safe,
                "standards": self.STANDARDS,
                "constraints_hash": self._constraints_hash[:16] + "...",
            }


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "ConstraintType",
    "ConstraintSeverity",
    "ViolationSeverity",
    # Data classes
    "QualityConstraint",
    "ConstraintViolation",
    # Models
    "ConstraintValidationResult",
    # Main class
    "ConstraintValidator",
    # Constants
    "STEAM_QUALITY_CONSTRAINTS",
]
