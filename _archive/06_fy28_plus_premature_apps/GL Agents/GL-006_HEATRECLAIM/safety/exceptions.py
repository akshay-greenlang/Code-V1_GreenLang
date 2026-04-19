"""
GL-006 HEATRECLAIM - Safety Exceptions

Exception hierarchy for safety constraint violations in heat exchanger
network designs. All violations are fail-closed to prevent unsafe designs.

References:
- ASME PTC 4.3: Air Heater Performance
- ASME PTC 4.4: HRSG Performance
- API 660: Shell and Tube Heat Exchangers
- ISO 14414: Pump System Energy Assessment
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, List


class ViolationSeverity(Enum):
    """Severity levels for safety violations."""
    WARNING = "warning"      # Log and continue with penalty
    ERROR = "error"          # Reject design, can be overridden
    CRITICAL = "critical"    # Reject design, cannot be overridden


@dataclass(frozen=True)
class ViolationDetails:
    """Immutable details about a safety violation."""
    constraint_tag: str
    constraint_description: str
    actual_value: float
    limit_value: float
    unit: str
    severity: ViolationSeverity
    location: str  # e.g., "Exchanger E-101, hot side outlet"
    standard_reference: str  # e.g., "ASME PTC 4.4 Section 5.3"
    recommended_action: str


class SafetyViolationError(Exception):
    """
    Base exception for all safety violations in HEN design.

    This exception triggers fail-closed behavior - designs with
    unhandled safety violations are rejected.
    """

    def __init__(
        self,
        message: str,
        violations: Optional[List[ViolationDetails]] = None,
        design_id: Optional[str] = None,
    ):
        super().__init__(message)
        self.violations = violations or []
        self.design_id = design_id

    def get_violation_summary(self) -> str:
        """Generate a summary of all violations."""
        if not self.violations:
            return str(self)

        lines = [f"Safety violations in design {self.design_id}:"]
        for i, v in enumerate(self.violations, 1):
            lines.append(
                f"  {i}. [{v.severity.value.upper()}] {v.constraint_tag}: "
                f"{v.actual_value:.2f} {v.unit} (limit: {v.limit_value:.2f} {v.unit})"
            )
            lines.append(f"     Location: {v.location}")
            lines.append(f"     Standard: {v.standard_reference}")
            lines.append(f"     Action: {v.recommended_action}")
        return "\n".join(lines)


class ApproachTemperatureViolation(SafetyViolationError):
    """
    Violation of minimum approach temperature (delta_t_min).

    Caused by: Heat exchanger design with insufficient temperature
    driving force, which can lead to:
    - Uneconomically large heat transfer area
    - Temperature crossover (thermodynamically infeasible)
    - Process instability

    Reference: Linnhoff & Hindmarsh (1983), Pinch Design Method
    """
    pass


class FilmTemperatureViolation(SafetyViolationError):
    """
    Violation of maximum film temperature limit.

    Caused by: Tube wall/film temperature exceeding safe limits,
    which can lead to:
    - Coking in hydrocarbon services
    - Tube metallurgy failure
    - Fouling acceleration

    Reference: API 660 Section 7.2.5, TEMA Standards
    """
    pass


class AcidDewPointViolation(SafetyViolationError):
    """
    Violation of acid dew point constraint.

    Caused by: Flue gas outlet temperature dropping below acid
    dew point, which can lead to:
    - Sulfuric/nitric acid condensation
    - Severe corrosion (cold-end corrosion)
    - Equipment failure

    Reference: ASME PTC 4.3 Section 5.4.2
    """
    pass


class PressureDropViolation(SafetyViolationError):
    """
    Violation of maximum pressure drop limits.

    Caused by: Excessive pressure drop across exchanger,
    which can lead to:
    - Pump/compressor capacity exceeded
    - Process bottleneck
    - Increased operating cost

    Reference: API 660 Section 6.3, ISO 14414
    """
    pass


class ThermalStressViolation(SafetyViolationError):
    """
    Violation of thermal stress rate limit.

    Caused by: Temperature change rate exceeding material limits,
    which can lead to:
    - Thermal shock
    - Tube/shell cracking
    - Weld failure
    - Equipment damage during startup/shutdown

    Reference: ASME PTC 4.4 Section 5.5
    """
    pass


class FoulingExceedanceViolation(SafetyViolationError):
    """
    Violation of fouling factor assumptions.

    Caused by: Actual fouling exceeding design allowance,
    which can lead to:
    - Reduced heat transfer
    - Increased pressure drop
    - Process constraint

    Reference: TEMA Standards, Table 10
    """
    pass
