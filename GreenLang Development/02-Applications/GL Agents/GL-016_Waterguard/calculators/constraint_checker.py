"""
Constraint Validation Engine for Water Chemistry

This module provides deterministic constraint checking for water chemistry
parameters. It distinguishes between hard constraints (absolute limits that
must never be exceeded) and soft constraints (preferred operating ranges).

Zero-Hallucination Guarantee:
- All constraint checks are deterministic comparisons
- No probabilistic or ML-based decisions
- Complete audit trail with SHA-256 hashes
- Distance-to-limit and time-to-violation calculations
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Union, Optional, Dict, List, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from datetime import datetime


class ConstraintSeverity(Enum):
    """Severity levels for constraint violations."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class ConstraintStatus(Enum):
    """Status of a constraint check."""
    COMPLIANT = "compliant"
    WARNING = "warning"
    VIOLATED = "violated"
    UNKNOWN = "unknown"


@dataclass
class ConstraintResult:
    """Result of a single constraint check."""
    constraint_id: str
    constraint_name: str
    status: ConstraintStatus
    current_value: Decimal
    limit_value: Decimal
    unit: str
    distance_to_limit: Decimal
    distance_percent: Decimal
    message: str
    severity: ConstraintSeverity
    provenance_hash: str


@dataclass
class ComplianceResult:
    """Result of checking all constraints."""
    overall_status: ConstraintStatus
    total_constraints: int
    compliant_count: int
    warning_count: int
    violated_count: int
    constraint_results: List[ConstraintResult]
    timestamp: str
    provenance_hash: str
    config_version: str
    code_version: str


@dataclass
class TimeToViolation:
    """Estimated time to constraint violation."""
    constraint_id: str
    current_value: Decimal
    limit_value: Decimal
    trend_rate: Decimal
    trend_unit: str
    time_to_violation: Optional[Decimal]
    time_unit: str
    direction: str
    provenance_hash: str


@dataclass
class HardConstraint:
    """
    Hard constraint that must never be exceeded.

    Hard constraints represent absolute limits that, if exceeded,
    could result in equipment damage, safety hazards, or regulatory
    violations.
    """
    constraint_id: str
    name: str
    parameter: str
    limit_type: str  # "max", "min", or "range"
    limit_value: Optional[Decimal] = None
    min_value: Optional[Decimal] = None
    max_value: Optional[Decimal] = None
    unit: str = ""
    description: str = ""
    regulatory_reference: str = ""
    severity: ConstraintSeverity = ConstraintSeverity.CRITICAL

    def check(self, value: Union[float, Decimal]) -> Tuple[ConstraintStatus, Decimal]:
        """
        Check if value complies with this constraint.

        Returns:
            Tuple of (status, distance_to_limit)
        """
        val = Decimal(str(value))

        if self.limit_type == "max":
            distance = self.limit_value - val
            if val > self.limit_value:
                return ConstraintStatus.VIOLATED, distance
            return ConstraintStatus.COMPLIANT, distance

        elif self.limit_type == "min":
            distance = val - self.limit_value
            if val < self.limit_value:
                return ConstraintStatus.VIOLATED, distance
            return ConstraintStatus.COMPLIANT, distance

        elif self.limit_type == "range":
            if val < self.min_value:
                distance = self.min_value - val
                return ConstraintStatus.VIOLATED, -distance
            elif val > self.max_value:
                distance = val - self.max_value
                return ConstraintStatus.VIOLATED, distance
            else:
                # Distance to nearest limit
                dist_to_min = val - self.min_value
                dist_to_max = self.max_value - val
                distance = min(dist_to_min, dist_to_max)
                return ConstraintStatus.COMPLIANT, distance

        return ConstraintStatus.UNKNOWN, Decimal("0")


@dataclass
class SoftConstraint:
    """
    Soft constraint representing preferred operating range.

    Soft constraints define optimal operating conditions. Values
    outside these ranges are not violations but may indicate
    suboptimal operation or early warning of potential issues.
    """
    constraint_id: str
    name: str
    parameter: str
    target_value: Optional[Decimal] = None
    preferred_min: Optional[Decimal] = None
    preferred_max: Optional[Decimal] = None
    warning_min: Optional[Decimal] = None
    warning_max: Optional[Decimal] = None
    unit: str = ""
    description: str = ""
    optimization_goal: str = ""

    def check(self, value: Union[float, Decimal]) -> Tuple[ConstraintStatus, Decimal]:
        """
        Check if value is within preferred operating range.

        Returns:
            Tuple of (status, distance_from_target_or_range)
        """
        val = Decimal(str(value))

        # Check warning limits first
        if self.warning_min is not None and val < self.warning_min:
            distance = self.warning_min - val
            return ConstraintStatus.WARNING, -distance

        if self.warning_max is not None and val > self.warning_max:
            distance = val - self.warning_max
            return ConstraintStatus.WARNING, distance

        # Check preferred range
        if self.preferred_min is not None and val < self.preferred_min:
            distance = self.preferred_min - val
            return ConstraintStatus.WARNING, -distance

        if self.preferred_max is not None and val > self.preferred_max:
            distance = val - self.preferred_max
            return ConstraintStatus.WARNING, distance

        # Calculate distance from target or center of range
        if self.target_value is not None:
            distance = abs(val - self.target_value)
        elif self.preferred_min is not None and self.preferred_max is not None:
            center = (self.preferred_min + self.preferred_max) / 2
            distance = abs(val - center)
        else:
            distance = Decimal("0")

        return ConstraintStatus.COMPLIANT, distance


class ConstraintChecker:
    """
    Constraint validation engine for water chemistry.

    Provides:
    - Hard constraint checking (absolute limits)
    - Soft constraint checking (preferred ranges)
    - Distance-to-limit calculations
    - Time-to-violation estimation
    - Complete audit trails
    """

    # Standard hard constraints for boiler water chemistry
    DEFAULT_HARD_CONSTRAINTS = {
        "max_boiler_conductivity": HardConstraint(
            constraint_id="HC001",
            name="Maximum Boiler Conductivity",
            parameter="boiler_conductivity",
            limit_type="max",
            limit_value=Decimal("5000"),  # uS/cm typical for medium pressure
            unit="uS/cm",
            description="Maximum allowable conductivity in boiler water",
            regulatory_reference="ASME Guidelines",
            severity=ConstraintSeverity.CRITICAL,
        ),
        "max_silica": HardConstraint(
            constraint_id="HC002",
            name="Maximum Silica",
            parameter="silica",
            limit_type="max",
            limit_value=Decimal("150"),  # mg/L as SiO2
            unit="mg/L",
            description="Maximum silica to prevent scaling and carryover",
            regulatory_reference="ASME Guidelines",
            severity=ConstraintSeverity.CRITICAL,
        ),
        "pH_range": HardConstraint(
            constraint_id="HC003",
            name="Boiler pH Range",
            parameter="pH",
            limit_type="range",
            min_value=Decimal("10.5"),
            max_value=Decimal("11.5"),
            unit="pH",
            description="Acceptable pH range for boiler water",
            regulatory_reference="ASME Guidelines",
            severity=ConstraintSeverity.CRITICAL,
        ),
        "max_dissolved_O2": HardConstraint(
            constraint_id="HC004",
            name="Maximum Dissolved Oxygen",
            parameter="dissolved_oxygen",
            limit_type="max",
            limit_value=Decimal("7"),  # ppb
            unit="ppb",
            description="Maximum dissolved oxygen to prevent corrosion",
            regulatory_reference="ASME Guidelines",
            severity=ConstraintSeverity.CRITICAL,
        ),
        "max_iron": HardConstraint(
            constraint_id="HC005",
            name="Maximum Iron",
            parameter="iron",
            limit_type="max",
            limit_value=Decimal("100"),  # ppb
            unit="ppb",
            description="Maximum iron indicating corrosion",
            regulatory_reference="Industry Best Practice",
            severity=ConstraintSeverity.WARNING,
        ),
    }

    # Standard soft constraints for optimal operation
    DEFAULT_SOFT_CONSTRAINTS = {
        "preferred_alkalinity": SoftConstraint(
            constraint_id="SC001",
            name="Preferred Alkalinity Range",
            parameter="alkalinity",
            preferred_min=Decimal("100"),
            preferred_max=Decimal("700"),
            warning_min=Decimal("50"),
            warning_max=Decimal("800"),
            unit="mg/L as CaCO3",
            description="Optimal alkalinity range for corrosion protection",
            optimization_goal="Maintain protective alkalinity without excess",
        ),
        "preferred_hardness": SoftConstraint(
            constraint_id="SC002",
            name="Preferred Hardness",
            parameter="hardness",
            preferred_max=Decimal("0.5"),
            warning_max=Decimal("1.0"),
            unit="mg/L as CaCO3",
            description="Target hardness for scale prevention",
            optimization_goal="Minimize hardness to prevent scaling",
        ),
        "preferred_phosphate": SoftConstraint(
            constraint_id="SC003",
            name="Preferred Phosphate Range",
            parameter="phosphate",
            target_value=Decimal("20"),
            preferred_min=Decimal("10"),
            preferred_max=Decimal("30"),
            warning_min=Decimal("5"),
            warning_max=Decimal("50"),
            unit="mg/L as PO4",
            description="Optimal phosphate for treatment program",
            optimization_goal="Maintain phosphate residual for scale control",
        ),
        "preferred_sulfite": SoftConstraint(
            constraint_id="SC004",
            name="Preferred Sulfite Residual",
            parameter="sulfite",
            target_value=Decimal("20"),
            preferred_min=Decimal("10"),
            preferred_max=Decimal("40"),
            warning_min=Decimal("5"),
            warning_max=Decimal("60"),
            unit="mg/L as Na2SO3",
            description="Optimal sulfite residual for oxygen scavenging",
            optimization_goal="Maintain sulfite residual for corrosion protection",
        ),
    }

    def __init__(
        self,
        config_version: str = "1.0.0",
        code_version: str = "1.0.0",
        hard_constraints: Optional[Dict[str, HardConstraint]] = None,
        soft_constraints: Optional[Dict[str, SoftConstraint]] = None
    ):
        """
        Initialize the constraint checker.

        Args:
            config_version: Version of the configuration
            code_version: Version of the calculation code
            hard_constraints: Custom hard constraints (uses defaults if None)
            soft_constraints: Custom soft constraints (uses defaults if None)
        """
        self.config_version = config_version
        self.code_version = code_version
        self.hard_constraints = hard_constraints or self.DEFAULT_HARD_CONSTRAINTS.copy()
        self.soft_constraints = soft_constraints or self.DEFAULT_SOFT_CONSTRAINTS.copy()

    def check_all_constraints(
        self,
        chemistry_state: Dict[str, Union[float, Decimal]],
        input_event_ids: Optional[list] = None
    ) -> ComplianceResult:
        """
        Check all constraints against current chemistry state.

        Args:
            chemistry_state: Dictionary of parameter names to values
            input_event_ids: List of input event IDs for provenance

        Returns:
            ComplianceResult with all constraint check results
        """
        timestamp = datetime.utcnow().isoformat() + "Z"
        constraint_results = []
        compliant_count = 0
        warning_count = 0
        violated_count = 0

        # Check hard constraints
        for constraint_name, constraint in self.hard_constraints.items():
            if constraint.parameter in chemistry_state:
                value = Decimal(str(chemistry_state[constraint.parameter]))
                status, distance = constraint.check(value)

                # Calculate distance percent
                if constraint.limit_type == "max" and constraint.limit_value:
                    distance_percent = (distance / constraint.limit_value * 100).quantize(
                        Decimal("0.01"), rounding=ROUND_HALF_UP
                    )
                    limit_val = constraint.limit_value
                elif constraint.limit_type == "min" and constraint.limit_value:
                    distance_percent = (distance / constraint.limit_value * 100).quantize(
                        Decimal("0.01"), rounding=ROUND_HALF_UP
                    )
                    limit_val = constraint.limit_value
                elif constraint.limit_type == "range":
                    range_size = constraint.max_value - constraint.min_value
                    distance_percent = (distance / range_size * 100).quantize(
                        Decimal("0.01"), rounding=ROUND_HALF_UP
                    )
                    limit_val = constraint.max_value if distance > 0 else constraint.min_value
                else:
                    distance_percent = Decimal("0")
                    limit_val = Decimal("0")

                # Build message
                if status == ConstraintStatus.VIOLATED:
                    message = f"{constraint.name} VIOLATED: {value} {constraint.unit}"
                    violated_count += 1
                else:
                    message = f"{constraint.name} compliant: {value} {constraint.unit}"
                    compliant_count += 1

                # Calculate provenance hash for this constraint
                result_data = {
                    "constraint_id": constraint.constraint_id,
                    "value": str(value),
                    "limit": str(limit_val),
                    "status": status.value,
                    "distance": str(distance),
                }
                result_str = json.dumps(result_data, sort_keys=True)
                result_hash = hashlib.sha256(result_str.encode("utf-8")).hexdigest()

                constraint_results.append(ConstraintResult(
                    constraint_id=constraint.constraint_id,
                    constraint_name=constraint.name,
                    status=status,
                    current_value=value,
                    limit_value=limit_val,
                    unit=constraint.unit,
                    distance_to_limit=distance,
                    distance_percent=distance_percent,
                    message=message,
                    severity=constraint.severity,
                    provenance_hash=result_hash,
                ))

        # Check soft constraints
        for constraint_name, constraint in self.soft_constraints.items():
            if constraint.parameter in chemistry_state:
                value = Decimal(str(chemistry_state[constraint.parameter]))
                status, distance = constraint.check(value)

                # Calculate reference for distance percent
                if constraint.target_value:
                    reference = constraint.target_value
                elif constraint.preferred_max:
                    reference = constraint.preferred_max
                else:
                    reference = Decimal("1")

                distance_percent = (abs(distance) / reference * 100).quantize(
                    Decimal("0.01"), rounding=ROUND_HALF_UP
                )

                if status == ConstraintStatus.WARNING:
                    message = f"{constraint.name} outside preferred range: {value} {constraint.unit}"
                    warning_count += 1
                else:
                    message = f"{constraint.name} within range: {value} {constraint.unit}"
                    compliant_count += 1

                # Calculate provenance hash
                result_data = {
                    "constraint_id": constraint.constraint_id,
                    "value": str(value),
                    "status": status.value,
                    "distance": str(distance),
                }
                result_str = json.dumps(result_data, sort_keys=True)
                result_hash = hashlib.sha256(result_str.encode("utf-8")).hexdigest()

                constraint_results.append(ConstraintResult(
                    constraint_id=constraint.constraint_id,
                    constraint_name=constraint.name,
                    status=status,
                    current_value=value,
                    limit_value=reference,
                    unit=constraint.unit,
                    distance_to_limit=distance,
                    distance_percent=distance_percent,
                    message=message,
                    severity=ConstraintSeverity.WARNING,
                    provenance_hash=result_hash,
                ))

        # Determine overall status
        if violated_count > 0:
            overall_status = ConstraintStatus.VIOLATED
        elif warning_count > 0:
            overall_status = ConstraintStatus.WARNING
        else:
            overall_status = ConstraintStatus.COMPLIANT

        # Calculate overall provenance hash
        provenance_data = {
            "operation": "check_all_constraints",
            "config_version": self.config_version,
            "code_version": self.code_version,
            "chemistry_state": {k: str(v) for k, v in chemistry_state.items()},
            "input_event_ids": input_event_ids or [],
            "constraint_hashes": [r.provenance_hash for r in constraint_results],
            "overall_status": overall_status.value,
            "timestamp": timestamp,
        }
        provenance_str = json.dumps(provenance_data, sort_keys=True)
        provenance_hash = hashlib.sha256(provenance_str.encode("utf-8")).hexdigest()

        return ComplianceResult(
            overall_status=overall_status,
            total_constraints=len(constraint_results),
            compliant_count=compliant_count,
            warning_count=warning_count,
            violated_count=violated_count,
            constraint_results=constraint_results,
            timestamp=timestamp,
            provenance_hash=provenance_hash,
            config_version=self.config_version,
            code_version=self.code_version,
        )

    def calculate_distance_to_limits(
        self,
        chemistry_state: Dict[str, Union[float, Decimal]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate distance to limits for all parameters.

        Args:
            chemistry_state: Dictionary of parameter names to values

        Returns:
            Dictionary with distance information for each parameter
        """
        results = {}

        for constraint_name, constraint in self.hard_constraints.items():
            if constraint.parameter in chemistry_state:
                value = Decimal(str(chemistry_state[constraint.parameter]))
                status, distance = constraint.check(value)

                if constraint.limit_type == "max":
                    percent_of_limit = (value / constraint.limit_value * 100).quantize(
                        Decimal("0.01"), rounding=ROUND_HALF_UP
                    )
                    headroom_percent = (Decimal("100") - percent_of_limit)
                elif constraint.limit_type == "min":
                    percent_of_limit = (value / constraint.limit_value * 100).quantize(
                        Decimal("0.01"), rounding=ROUND_HALF_UP
                    )
                    headroom_percent = percent_of_limit - Decimal("100")
                else:
                    range_size = constraint.max_value - constraint.min_value
                    position = value - constraint.min_value
                    percent_of_limit = (position / range_size * 100).quantize(
                        Decimal("0.01"), rounding=ROUND_HALF_UP
                    )
                    headroom_percent = min(percent_of_limit, Decimal("100") - percent_of_limit)

                results[constraint.parameter] = {
                    "constraint_id": constraint.constraint_id,
                    "current_value": value,
                    "limit_value": constraint.limit_value or constraint.max_value,
                    "distance_absolute": distance,
                    "percent_of_limit": percent_of_limit,
                    "headroom_percent": headroom_percent,
                    "status": status.value,
                    "unit": constraint.unit,
                }

        return results

    def estimate_time_to_violation(
        self,
        current_value: Union[float, Decimal],
        trend_rate: Union[float, Decimal],
        limit_value: Union[float, Decimal],
        trend_unit: str = "per_hour"
    ) -> TimeToViolation:
        """
        Estimate time until a constraint limit will be reached.

        Args:
            current_value: Current parameter value
            trend_rate: Rate of change (positive = increasing)
            limit_value: Constraint limit value
            trend_unit: Unit of trend rate (per_hour, per_minute, per_day)

        Returns:
            TimeToViolation with estimated time and direction
        """
        current = Decimal(str(current_value))
        rate = Decimal(str(trend_rate))
        limit = Decimal(str(limit_value))

        # Determine direction
        if rate > 0:
            direction = "increasing"
            if current >= limit:
                time_to_violation = Decimal("0")
            else:
                time_to_violation = (limit - current) / rate
        elif rate < 0:
            direction = "decreasing"
            if current <= limit:
                time_to_violation = Decimal("0")
            else:
                time_to_violation = (current - limit) / abs(rate)
        else:
            direction = "stable"
            time_to_violation = None

        # Determine time unit based on trend_unit
        if trend_unit == "per_hour":
            time_unit = "hours"
        elif trend_unit == "per_minute":
            time_unit = "minutes"
        elif trend_unit == "per_day":
            time_unit = "days"
        else:
            time_unit = "units"

        if time_to_violation is not None:
            time_to_violation = time_to_violation.quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )

        # Calculate provenance hash
        provenance_data = {
            "operation": "estimate_time_to_violation",
            "current_value": str(current),
            "trend_rate": str(rate),
            "limit_value": str(limit),
            "trend_unit": trend_unit,
            "time_to_violation": str(time_to_violation) if time_to_violation else None,
            "direction": direction,
        }
        provenance_str = json.dumps(provenance_data, sort_keys=True)
        provenance_hash = hashlib.sha256(provenance_str.encode("utf-8")).hexdigest()

        return TimeToViolation(
            constraint_id="",
            current_value=current,
            limit_value=limit,
            trend_rate=rate,
            trend_unit=trend_unit,
            time_to_violation=time_to_violation,
            time_unit=time_unit,
            direction=direction,
            provenance_hash=provenance_hash,
        )
