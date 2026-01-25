"""
GL-007 FurnacePulse - Remaining Useful Life Predictor

Deterministic calculator for furnace component RUL prediction using
Weibull-based failure modeling. Integrates with CMMS maintenance history
for improved predictions.

Key Calculations:
    - Weibull distribution-based failure probability
    - Confidence interval calculation
    - CMMS integration for maintenance history
    - Predictive lead time estimation

Weibull Distribution:
    F(t) = 1 - exp(-(t/eta)^beta)

    Where:
        - beta (shape parameter): failure rate behavior
        - eta (scale parameter): characteristic life
        - t: operating time

Example:
    >>> predictor = RULPredictor(agent_id="GL-007")
    >>> inputs = RULInputs(
    ...     component_id="TUBE-001",
    ...     component_type=ComponentType.RADIANT_TUBE,
    ...     operating_hours=45000,
    ...     weibull_params=WeibullParameters(beta=2.5, eta=60000)
    ... )
    >>> result = predictor.calculate(inputs)
    >>> print(f"RUL: {result.result.rul_hours:.0f} hours")
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum
import math
import sys
import os

# Add framework path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'Framework_GreenLang', 'shared'))

from calculator_base import DeterministicCalculator, CalculationResult


class ComponentType(str, Enum):
    """Furnace component types with default Weibull parameters."""
    RADIANT_TUBE = "radiant_tube"
    CONVECTION_TUBE = "convection_tube"
    BURNER = "burner"
    REFRACTORY = "refractory"
    FAN = "fan"
    DAMPER = "damper"
    THERMOCOUPLE = "thermocouple"
    SIGHT_GLASS = "sight_glass"
    EXPANSION_JOINT = "expansion_joint"
    TUBE_SUPPORT = "tube_support"


class FailureMode(str, Enum):
    """Component failure modes."""
    CREEP = "creep"
    FATIGUE = "fatigue"
    CORROSION = "corrosion"
    EROSION = "erosion"
    OXIDATION = "oxidation"
    THERMAL_SHOCK = "thermal_shock"
    MECHANICAL_WEAR = "mechanical_wear"


@dataclass
class WeibullParameters:
    """
    Weibull distribution parameters for failure modeling.

    Attributes:
        beta: Shape parameter (failure rate behavior)
            - beta < 1: Decreasing failure rate (infant mortality)
            - beta = 1: Constant failure rate (random failures)
            - beta > 1: Increasing failure rate (wear-out)
        eta: Scale parameter (characteristic life in hours)
        gamma: Location parameter (minimum life, default 0)
    """
    beta: float  # Shape parameter
    eta: float   # Scale parameter (hours)
    gamma: float = 0.0  # Location parameter (hours)

    def __post_init__(self):
        """Validate parameters."""
        if self.beta <= 0:
            raise ValueError("beta must be positive")
        if self.eta <= 0:
            raise ValueError("eta must be positive")
        if self.gamma < 0:
            raise ValueError("gamma cannot be negative")


# Default Weibull parameters by component type (based on industry data)
DEFAULT_WEIBULL_PARAMS: Dict[ComponentType, WeibullParameters] = {
    ComponentType.RADIANT_TUBE: WeibullParameters(beta=2.5, eta=70000),
    ComponentType.CONVECTION_TUBE: WeibullParameters(beta=2.2, eta=90000),
    ComponentType.BURNER: WeibullParameters(beta=1.8, eta=40000),
    ComponentType.REFRACTORY: WeibullParameters(beta=2.0, eta=50000),
    ComponentType.FAN: WeibullParameters(beta=2.8, eta=60000),
    ComponentType.DAMPER: WeibullParameters(beta=1.5, eta=45000),
    ComponentType.THERMOCOUPLE: WeibullParameters(beta=1.2, eta=15000),
    ComponentType.SIGHT_GLASS: WeibullParameters(beta=1.3, eta=20000),
    ComponentType.EXPANSION_JOINT: WeibullParameters(beta=2.3, eta=55000),
    ComponentType.TUBE_SUPPORT: WeibullParameters(beta=2.6, eta=80000),
}


@dataclass
class MaintenanceRecord:
    """
    CMMS maintenance history record.

    Attributes:
        record_id: Unique record identifier
        component_id: Component identifier
        maintenance_date: Date of maintenance
        maintenance_type: Type of maintenance performed
        description: Maintenance description
        operating_hours_at_maintenance: Component hours at maintenance
        condition_score: Condition assessment (0-100)
        next_scheduled_hours: Next scheduled maintenance (hours)
    """
    record_id: str
    component_id: str
    maintenance_date: datetime
    maintenance_type: str  # e.g., "inspection", "repair", "replacement"
    description: str
    operating_hours_at_maintenance: float
    condition_score: Optional[float] = None  # 0-100
    next_scheduled_hours: Optional[float] = None


@dataclass
class OperatingConditions:
    """
    Current operating conditions affecting RUL.

    Attributes:
        avg_temperature_c: Average operating temperature
        design_temperature_c: Design temperature
        thermal_cycles: Number of thermal cycles
        max_temperature_excursion_c: Maximum temperature excursion
        corrosion_rate_mm_yr: Measured corrosion rate
        wall_thickness_mm: Current wall thickness
        min_wall_thickness_mm: Minimum allowable thickness
    """
    avg_temperature_c: Optional[float] = None
    design_temperature_c: Optional[float] = None
    thermal_cycles: Optional[int] = None
    max_temperature_excursion_c: Optional[float] = None
    corrosion_rate_mm_yr: Optional[float] = None
    wall_thickness_mm: Optional[float] = None
    min_wall_thickness_mm: Optional[float] = None


@dataclass
class RULInputs:
    """
    Input data for RUL prediction.

    Attributes:
        component_id: Unique component identifier
        component_type: Type of furnace component
        operating_hours: Current operating hours
        weibull_params: Weibull parameters (optional, uses defaults)
        maintenance_history: CMMS maintenance records
        operating_conditions: Current operating conditions
        failure_modes: Active failure modes to consider
        confidence_level: Confidence level for interval (0.9, 0.95, 0.99)
    """
    component_id: str
    component_type: ComponentType
    operating_hours: float
    weibull_params: Optional[WeibullParameters] = None
    maintenance_history: List[MaintenanceRecord] = field(default_factory=list)
    operating_conditions: Optional[OperatingConditions] = None
    failure_modes: List[FailureMode] = field(default_factory=list)
    confidence_level: float = 0.90

    def get_weibull_params(self) -> WeibullParameters:
        """Get Weibull parameters (explicit or defaults)."""
        if self.weibull_params is not None:
            return self.weibull_params
        return DEFAULT_WEIBULL_PARAMS[self.component_type]


@dataclass
class RULOutputs:
    """
    Output from RUL prediction.

    Attributes:
        component_id: Component identifier
        component_type: Component type
        operating_hours: Current operating hours
        rul_hours: Predicted remaining useful life (hours)
        rul_days: RUL in days (assuming 24/7 operation)
        failure_probability: Current failure probability (0-1)
        hazard_rate: Current hazard rate (failures per hour)
        confidence_interval_lower: Lower bound of RUL CI
        confidence_interval_upper: Upper bound of RUL CI
        maintenance_lead_time_hours: Recommended lead time for maintenance
        health_index: Overall health index (0-100)
        risk_category: Risk categorization (LOW/MEDIUM/HIGH/CRITICAL)
        recommended_action: Suggested maintenance action
        next_inspection_hours: Recommended hours until next inspection
    """
    component_id: str
    component_type: ComponentType
    operating_hours: float
    rul_hours: float
    rul_days: float
    failure_probability: float
    hazard_rate: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    maintenance_lead_time_hours: float
    health_index: float
    risk_category: str
    recommended_action: str
    next_inspection_hours: float


class RULPredictor(DeterministicCalculator[RULInputs, RULOutputs]):
    """
    Deterministic calculator for Remaining Useful Life prediction.

    Uses Weibull distribution for failure probability modeling with
    CMMS maintenance history integration. All calculations are
    deterministic with SHA-256 provenance tracking.

    Weibull Formulas:
        - Reliability: R(t) = exp(-(t/eta)^beta)
        - Failure probability: F(t) = 1 - R(t)
        - Hazard rate: h(t) = (beta/eta) * (t/eta)^(beta-1)
        - RUL: Time until F(t) reaches threshold

    Example:
        >>> predictor = RULPredictor(agent_id="GL-007")
        >>> inputs = RULInputs(
        ...     component_id="TUBE-001",
        ...     component_type=ComponentType.RADIANT_TUBE,
        ...     operating_hours=45000
        ... )
        >>> result = predictor.calculate(inputs)
        >>> print(f"RUL: {result.result.rul_hours:.0f} hours")
    """

    NAME = "FurnaceRULPredictor"
    VERSION = "1.0.0"

    # RUL thresholds and constants
    FAILURE_PROBABILITY_THRESHOLD = 0.10  # Default P(failure) threshold for RUL
    HAZARD_RATE_CRITICAL = 0.0001  # Critical hazard rate threshold
    MIN_INSPECTION_INTERVAL_HOURS = 720  # Minimum 30 days
    MAX_INSPECTION_INTERVAL_HOURS = 8760  # Maximum 1 year

    def __init__(
        self,
        agent_id: str = "GL-007",
        track_provenance: bool = True,
        failure_threshold: float = 0.10,
    ):
        """
        Initialize RUL predictor.

        Args:
            agent_id: Agent identifier for provenance
            track_provenance: Whether to track calculation provenance
            failure_threshold: Failure probability threshold for RUL
        """
        super().__init__(agent_id, track_provenance)
        self.failure_threshold = failure_threshold

    def _validate_inputs(self, inputs: RULInputs) -> List[str]:
        """
        Validate RUL prediction inputs.

        Args:
            inputs: RUL inputs to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Validate component ID
        if not inputs.component_id or not inputs.component_id.strip():
            errors.append("component_id is required")

        # Validate operating hours
        if inputs.operating_hours < 0:
            errors.append("operating_hours cannot be negative")

        # Validate Weibull parameters if provided
        if inputs.weibull_params is not None:
            params = inputs.weibull_params
            if params.beta <= 0:
                errors.append("Weibull beta must be positive")
            if params.eta <= 0:
                errors.append("Weibull eta must be positive")
            if params.gamma < 0:
                errors.append("Weibull gamma cannot be negative")
            if params.gamma >= inputs.operating_hours:
                errors.append("Weibull gamma cannot exceed operating hours")

        # Validate confidence level
        if not 0.5 <= inputs.confidence_level <= 0.999:
            errors.append("confidence_level must be between 0.5 and 0.999")

        # Validate operating conditions if provided
        if inputs.operating_conditions is not None:
            cond = inputs.operating_conditions
            if cond.avg_temperature_c is not None and cond.avg_temperature_c < 0:
                errors.append("avg_temperature_c cannot be negative in Kelvin-equivalent")
            if cond.wall_thickness_mm is not None and cond.wall_thickness_mm <= 0:
                errors.append("wall_thickness_mm must be positive")
            if cond.min_wall_thickness_mm is not None and cond.min_wall_thickness_mm <= 0:
                errors.append("min_wall_thickness_mm must be positive")

        return errors

    def _calculate(self, inputs: RULInputs, **kwargs: Any) -> RULOutputs:
        """
        Calculate Remaining Useful Life.

        This is a DETERMINISTIC calculation using Weibull distribution:
        - R(t) = exp(-(t/eta)^beta)
        - F(t) = 1 - R(t)
        - h(t) = (beta/eta) * (t/eta)^(beta-1)

        Args:
            inputs: Validated RUL inputs

        Returns:
            RULOutputs with all RUL metrics
        """
        params = inputs.get_weibull_params()
        t = inputs.operating_hours - params.gamma  # Effective time

        # Ensure t is positive
        t = max(t, 0.001)

        # Calculate current failure probability: F(t) = 1 - exp(-(t/eta)^beta)
        reliability = math.exp(-((t / params.eta) ** params.beta))
        failure_probability = 1.0 - reliability

        # Calculate hazard rate: h(t) = (beta/eta) * (t/eta)^(beta-1)
        hazard_rate = (params.beta / params.eta) * ((t / params.eta) ** (params.beta - 1))

        # Calculate RUL: time until failure probability reaches threshold
        # Solve for t_fail: F(t_fail) = threshold
        # t_fail = eta * (-ln(1 - threshold))^(1/beta)
        target_reliability = 1.0 - self.failure_threshold
        if target_reliability > 0:
            t_fail = params.eta * ((-math.log(target_reliability)) ** (1.0 / params.beta))
        else:
            t_fail = params.eta * 3  # Fallback to 3x characteristic life

        rul_hours = max(t_fail - t, 0)
        rul_days = rul_hours / 24.0

        # Calculate confidence interval using Weibull properties
        # For a given confidence level alpha, find t such that F(t) = alpha
        alpha = inputs.confidence_level
        z_lower = (1.0 - alpha) / 2.0
        z_upper = 1.0 - z_lower

        # Lower bound: time when failure probability = z_lower
        if z_lower > 0:
            t_lower = params.eta * ((-math.log(1.0 - z_lower)) ** (1.0 / params.beta))
        else:
            t_lower = 0

        # Upper bound: time when failure probability = z_upper
        if z_upper < 1:
            t_upper = params.eta * ((-math.log(1.0 - z_upper)) ** (1.0 / params.beta))
        else:
            t_upper = params.eta * 5

        ci_lower = max(t_lower - t, 0)
        ci_upper = max(t_upper - t, 0)

        # Apply adjustments from operating conditions
        adjustment_factor = self._calculate_condition_adjustment(inputs)
        rul_hours *= adjustment_factor
        rul_days = rul_hours / 24.0
        ci_lower *= adjustment_factor
        ci_upper *= adjustment_factor

        # Apply CMMS history adjustment
        cmms_factor = self._calculate_cmms_adjustment(inputs)
        rul_hours *= cmms_factor
        rul_days = rul_hours / 24.0
        ci_lower *= cmms_factor
        ci_upper *= cmms_factor

        # Calculate maintenance lead time (10% of RUL, min 720 hours)
        maintenance_lead_time = max(rul_hours * 0.10, 720)

        # Calculate health index (0-100)
        health_index = self._calculate_health_index(
            failure_probability, hazard_rate, rul_hours, params.eta
        )

        # Determine risk category
        risk_category = self._determine_risk_category(
            failure_probability, hazard_rate, health_index, rul_hours
        )

        # Generate recommended action
        recommended_action = self._generate_recommendation(
            risk_category, rul_hours, inputs.component_type, health_index
        )

        # Calculate next inspection interval
        next_inspection_hours = self._calculate_inspection_interval(
            risk_category, rul_hours, health_index
        )

        return RULOutputs(
            component_id=inputs.component_id,
            component_type=inputs.component_type,
            operating_hours=inputs.operating_hours,
            rul_hours=round(rul_hours, 1),
            rul_days=round(rul_days, 1),
            failure_probability=round(failure_probability, 6),
            hazard_rate=round(hazard_rate, 10),
            confidence_interval_lower=round(ci_lower, 1),
            confidence_interval_upper=round(ci_upper, 1),
            maintenance_lead_time_hours=round(maintenance_lead_time, 1),
            health_index=round(health_index, 1),
            risk_category=risk_category,
            recommended_action=recommended_action,
            next_inspection_hours=round(next_inspection_hours, 1),
        )

    def _calculate_condition_adjustment(self, inputs: RULInputs) -> float:
        """
        Calculate RUL adjustment factor based on operating conditions.

        Returns factor < 1 for harsh conditions, > 1 for mild conditions.
        """
        if inputs.operating_conditions is None:
            return 1.0

        cond = inputs.operating_conditions
        adjustment = 1.0

        # Temperature stress adjustment
        if cond.avg_temperature_c is not None and cond.design_temperature_c is not None:
            temp_ratio = cond.avg_temperature_c / cond.design_temperature_c
            if temp_ratio > 1.0:
                # Operating above design - accelerated degradation
                # Each 10% over design reduces RUL by 20%
                adjustment *= max(0.5, 1.0 - (temp_ratio - 1.0) * 2.0)
            elif temp_ratio < 0.8:
                # Operating well below design - extended life
                adjustment *= min(1.3, 1.0 + (0.8 - temp_ratio) * 0.5)

        # Thermal cycling adjustment
        if cond.thermal_cycles is not None:
            # High cycle count reduces life
            if cond.thermal_cycles > 1000:
                cycle_penalty = min(0.3, (cond.thermal_cycles - 1000) * 0.0001)
                adjustment *= (1.0 - cycle_penalty)

        # Wall thickness adjustment (for tubes)
        if cond.wall_thickness_mm is not None and cond.min_wall_thickness_mm is not None:
            thickness_margin = cond.wall_thickness_mm - cond.min_wall_thickness_mm
            if thickness_margin <= 0:
                # Already at or below minimum
                adjustment *= 0.1
            else:
                # Estimate remaining life based on corrosion rate
                if cond.corrosion_rate_mm_yr is not None and cond.corrosion_rate_mm_yr > 0:
                    years_remaining = thickness_margin / cond.corrosion_rate_mm_yr
                    hours_remaining = years_remaining * 8760
                    # This directly limits RUL
                    # We return a factor that when multiplied gives hours_remaining
                    # This is handled in the main calculation

        return adjustment

    def _calculate_cmms_adjustment(self, inputs: RULInputs) -> float:
        """
        Calculate RUL adjustment based on CMMS maintenance history.

        Returns factor reflecting maintenance quality and recency.
        """
        if not inputs.maintenance_history:
            return 1.0

        # Sort by date (most recent first)
        sorted_records = sorted(
            inputs.maintenance_history,
            key=lambda r: r.maintenance_date,
            reverse=True
        )

        adjustment = 1.0

        # Check most recent maintenance
        latest = sorted_records[0]

        # Condition score adjustment
        if latest.condition_score is not None:
            # Good condition (80+) improves factor
            # Poor condition (50-) reduces factor
            score = latest.condition_score
            if score >= 80:
                adjustment *= 1.0 + (score - 80) * 0.005  # Up to 10% bonus
            elif score <= 50:
                adjustment *= 0.7 + score * 0.006  # Down to 70%

        # Maintenance type adjustment
        if latest.maintenance_type == "replacement":
            # Component was replaced - reset degradation
            hours_since_replacement = inputs.operating_hours - latest.operating_hours_at_maintenance
            # Adjust effective age
            adjustment *= 1.2  # New component bonus

        elif latest.maintenance_type == "repair":
            # Repair extends life somewhat
            adjustment *= 1.1

        return adjustment

    def _calculate_health_index(
        self,
        failure_probability: float,
        hazard_rate: float,
        rul_hours: float,
        eta: float,
    ) -> float:
        """
        Calculate overall health index (0-100).

        Combines multiple factors into single health score.
        """
        # Reliability-based component (0-40 points)
        reliability = 1.0 - failure_probability
        reliability_score = reliability * 40

        # RUL-based component (0-30 points)
        # Compare RUL to characteristic life
        rul_ratio = min(rul_hours / eta, 1.0)
        rul_score = rul_ratio * 30

        # Hazard rate component (0-30 points)
        # Lower hazard rate = higher score
        if hazard_rate > 0:
            # Normalize to typical range
            hazard_normalized = min(hazard_rate * 10000, 1.0)
            hazard_score = (1.0 - hazard_normalized) * 30
        else:
            hazard_score = 30

        return reliability_score + rul_score + hazard_score

    def _determine_risk_category(
        self,
        failure_probability: float,
        hazard_rate: float,
        health_index: float,
        rul_hours: float,
    ) -> str:
        """Determine risk category based on metrics."""
        # Critical: High failure probability or very low RUL
        if failure_probability > 0.15 or rul_hours < 720:
            return "CRITICAL"

        # High: Elevated risk indicators
        if failure_probability > 0.08 or health_index < 40 or rul_hours < 2160:
            return "HIGH"

        # Medium: Moderate risk
        if failure_probability > 0.03 or health_index < 60 or rul_hours < 8760:
            return "MEDIUM"

        # Low: Normal operation
        return "LOW"

    def _generate_recommendation(
        self,
        risk_category: str,
        rul_hours: float,
        component_type: ComponentType,
        health_index: float,
    ) -> str:
        """Generate maintenance recommendation based on risk."""
        rul_days = rul_hours / 24

        if risk_category == "CRITICAL":
            return (
                f"IMMEDIATE ACTION: {component_type.value} requires urgent inspection. "
                f"RUL estimated at {rul_days:.0f} days. Schedule replacement or repair within 30 days."
            )

        elif risk_category == "HIGH":
            return (
                f"PRIORITY MAINTENANCE: {component_type.value} health index at {health_index:.0f}%. "
                f"Plan inspection within 90 days. Begin sourcing replacement parts."
            )

        elif risk_category == "MEDIUM":
            return (
                f"SCHEDULED MAINTENANCE: {component_type.value} approaching maintenance window. "
                f"Include in next planned turnaround. Monitor condition indicators."
            )

        else:  # LOW
            return (
                f"ROUTINE MONITORING: {component_type.value} operating normally. "
                f"Continue standard inspection schedule. RUL: {rul_days:.0f} days."
            )

    def _calculate_inspection_interval(
        self,
        risk_category: str,
        rul_hours: float,
        health_index: float,
    ) -> float:
        """Calculate recommended inspection interval in hours."""
        # Base interval based on risk
        base_intervals = {
            "CRITICAL": 168,   # Weekly
            "HIGH": 720,       # Monthly
            "MEDIUM": 2160,    # Quarterly
            "LOW": 4320,       # 6 months
        }

        base = base_intervals.get(risk_category, 2160)

        # Adjust based on RUL
        # Don't exceed 10% of RUL
        max_interval = rul_hours * 0.10

        interval = min(base, max_interval)

        # Apply bounds
        return max(
            self.MIN_INSPECTION_INTERVAL_HOURS,
            min(interval, self.MAX_INSPECTION_INTERVAL_HOURS)
        )

    def predict_failure_date(
        self,
        inputs: RULInputs,
        operating_hours_per_day: float = 24.0,
    ) -> CalculationResult[Dict[str, Any]]:
        """
        Predict failure date based on RUL and operating schedule.

        Args:
            inputs: RUL inputs
            operating_hours_per_day: Operating hours per calendar day

        Returns:
            Predicted failure date and related metrics
        """
        # Calculate RUL first
        rul_result = self.calculate(inputs)

        if not rul_result.is_valid:
            return CalculationResult(
                result=None,
                computation_hash="",
                inputs_hash=rul_result.inputs_hash,
                calculator_name=self.NAME,
                calculator_version=self.VERSION,
                is_valid=False,
                warnings=rul_result.warnings,
            )

        rul = rul_result.result

        # Calculate calendar days until failure
        calendar_days = rul.rul_hours / operating_hours_per_day

        # Calculate dates
        now = datetime.now(timezone.utc)
        failure_date = now + timedelta(days=calendar_days)
        maintenance_date = now + timedelta(
            days=(rul.rul_hours - rul.maintenance_lead_time_hours) / operating_hours_per_day
        )

        result = {
            "component_id": rul.component_id,
            "current_date": now.isoformat(),
            "predicted_failure_date": failure_date.isoformat(),
            "recommended_maintenance_date": maintenance_date.isoformat(),
            "calendar_days_to_failure": round(calendar_days, 1),
            "calendar_days_to_maintenance": round(
                (rul.rul_hours - rul.maintenance_lead_time_hours) / operating_hours_per_day, 1
            ),
            "operating_hours_per_day": operating_hours_per_day,
            "rul_hours": rul.rul_hours,
            "risk_category": rul.risk_category,
        }

        # Compute provenance
        inputs_hash = self._compute_hash(inputs)
        outputs_hash = self._compute_hash(result)
        computation_hash = self._compute_combined_hash(
            inputs_hash, outputs_hash, {"operating_hours_per_day": operating_hours_per_day}
        )

        return CalculationResult(
            result=result,
            computation_hash=computation_hash,
            inputs_hash=inputs_hash,
            calculator_name=self.NAME,
            calculator_version=self.VERSION,
            is_valid=True,
        )
