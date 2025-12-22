"""
Uncertainty Data Models for GL-003 UNIFIEDSTEAM SteamSystemOptimizer.

This module defines immutable, strongly-typed data models for uncertainty
quantification. All uncertainty values carry complete provenance information
for audit trails and regulatory compliance.

Zero-Hallucination Guarantee:
- All uncertainty calculations are deterministic
- No LLM inference in uncertainty computation paths
- Complete provenance tracking with SHA-256 hashes
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Dict, List, Optional, Tuple
import hashlib
import json


class DistributionType(Enum):
    """Supported probability distribution types for uncertainty modeling."""
    NORMAL = "normal"
    UNIFORM = "uniform"
    TRIANGULAR = "triangular"
    LOGNORMAL = "lognormal"
    BETA = "beta"
    STUDENT_T = "student_t"
    EMPIRICAL = "empirical"


class DriftClass(Enum):
    """
    Sensor drift classification based on ISO 17025 and industrial standards.

    Drift classes define how sensor uncertainty degrades over time since
    last calibration.
    """
    CLASS_A = "A"  # Low drift: <0.1% per month (precision instruments)
    CLASS_B = "B"  # Medium drift: 0.1-0.5% per month (standard industrial)
    CLASS_C = "C"  # High drift: 0.5-2% per month (harsh environment)
    CLASS_D = "D"  # Very high drift: >2% per month (requires frequent calibration)


class ConfidenceLevel(Enum):
    """Standard confidence levels for uncertainty intervals."""
    CI_68 = 68.27  # 1-sigma
    CI_90 = 90.0
    CI_95 = 95.0   # 2-sigma (regulatory default)
    CI_99 = 99.0
    CI_99_7 = 99.73  # 3-sigma


@dataclass(frozen=True)
class UncertainValue:
    """
    Immutable representation of a value with associated uncertainty.

    This is the fundamental building block for uncertainty propagation.
    All values include complete statistical characterization and provenance.

    Attributes:
        mean: Central estimate (expected value)
        std: Standard deviation (1-sigma uncertainty)
        lower_95: Lower bound of 95% confidence interval
        upper_95: Upper bound of 95% confidence interval
        distribution_type: Assumed probability distribution
        unit: Physical unit of the value
        source_id: Identifier of the measurement source
        timestamp: When the value was recorded/computed
        provenance_hash: SHA-256 hash for audit trail
    """
    mean: float
    std: float
    lower_95: float
    upper_95: float
    distribution_type: DistributionType = DistributionType.NORMAL
    unit: str = ""
    source_id: str = ""
    timestamp: Optional[datetime] = None
    provenance_hash: str = ""

    def __post_init__(self):
        """Validate uncertainty bounds and compute provenance hash."""
        # Validation (frozen dataclass requires object.__setattr__)
        if self.std < 0:
            raise ValueError(f"Standard deviation cannot be negative: {self.std}")
        if self.lower_95 > self.mean:
            raise ValueError(f"Lower 95% CI ({self.lower_95}) exceeds mean ({self.mean})")
        if self.upper_95 < self.mean:
            raise ValueError(f"Upper 95% CI ({self.upper_95}) below mean ({self.mean})")

        # Compute provenance hash if not provided
        if not self.provenance_hash:
            hash_data = {
                "mean": self.mean,
                "std": self.std,
                "lower_95": self.lower_95,
                "upper_95": self.upper_95,
                "distribution": self.distribution_type.value,
                "source_id": self.source_id
            }
            hash_str = json.dumps(hash_data, sort_keys=True)
            computed_hash = hashlib.sha256(hash_str.encode()).hexdigest()
            object.__setattr__(self, 'provenance_hash', computed_hash)

    @classmethod
    def from_measurement(
        cls,
        value: float,
        uncertainty_percent: float,
        unit: str = "",
        source_id: str = "",
        distribution: DistributionType = DistributionType.NORMAL
    ) -> "UncertainValue":
        """
        Create UncertainValue from a measurement with percent uncertainty.

        Args:
            value: Measured value
            uncertainty_percent: Uncertainty as percentage of value
            unit: Physical unit
            source_id: Sensor or source identifier
            distribution: Assumed distribution type

        Returns:
            UncertainValue with computed bounds
        """
        std = abs(value) * (uncertainty_percent / 100.0)
        # 95% CI = mean +/- 1.96*sigma for normal distribution
        z_95 = 1.96
        lower_95 = value - z_95 * std
        upper_95 = value + z_95 * std

        return cls(
            mean=value,
            std=std,
            lower_95=lower_95,
            upper_95=upper_95,
            distribution_type=distribution,
            unit=unit,
            source_id=source_id,
            timestamp=datetime.utcnow()
        )

    @classmethod
    def from_bounds(
        cls,
        lower: float,
        upper: float,
        confidence_level: float = 95.0,
        distribution: DistributionType = DistributionType.NORMAL
    ) -> "UncertainValue":
        """
        Create UncertainValue from confidence interval bounds.

        Args:
            lower: Lower bound of confidence interval
            upper: Upper bound of confidence interval
            confidence_level: Confidence level percentage
            distribution: Assumed distribution type

        Returns:
            UncertainValue with computed mean and std
        """
        mean = (lower + upper) / 2.0

        # Z-score lookup for common confidence levels
        z_scores = {
            68.27: 1.0,
            90.0: 1.645,
            95.0: 1.96,
            99.0: 2.576,
            99.73: 3.0
        }
        z = z_scores.get(confidence_level, 1.96)

        half_width = (upper - lower) / 2.0
        std = half_width / z

        # Recompute 95% bounds if different confidence level provided
        z_95 = 1.96
        lower_95 = mean - z_95 * std
        upper_95 = mean + z_95 * std

        return cls(
            mean=mean,
            std=std,
            lower_95=lower_95,
            upper_95=upper_95,
            distribution_type=distribution,
            timestamp=datetime.utcnow()
        )

    def relative_uncertainty(self) -> float:
        """Return uncertainty as percentage of mean value."""
        if abs(self.mean) < 1e-10:
            return float('inf') if self.std > 0 else 0.0
        return (self.std / abs(self.mean)) * 100.0

    def get_bounds(self, confidence_level: ConfidenceLevel = ConfidenceLevel.CI_95) -> Tuple[float, float]:
        """
        Get confidence interval bounds at specified level.

        Args:
            confidence_level: Desired confidence level

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        z_scores = {
            ConfidenceLevel.CI_68: 1.0,
            ConfidenceLevel.CI_90: 1.645,
            ConfidenceLevel.CI_95: 1.96,
            ConfidenceLevel.CI_99: 2.576,
            ConfidenceLevel.CI_99_7: 3.0
        }
        z = z_scores.get(confidence_level, 1.96)

        return (self.mean - z * self.std, self.mean + z * self.std)


@dataclass(frozen=True)
class SensorUncertainty:
    """
    Sensor-specific uncertainty metadata including calibration and drift.

    This model tracks the complete uncertainty profile of a sensor,
    including time-dependent degradation since last calibration.

    Attributes:
        sensor_id: Unique sensor identifier
        base_accuracy: Manufacturer-specified accuracy (% of reading)
        drift_rate: Uncertainty increase rate (% per month)
        drift_class: Categorized drift behavior
        last_calibration: Date of most recent calibration
        calibration_accuracy: Accuracy achieved at calibration
        current_uncertainty: Time-degraded uncertainty estimate
        measurement_range_min: Minimum measurement range
        measurement_range_max: Maximum measurement range
        operating_temperature_min: Minimum operating temperature (C)
        operating_temperature_max: Maximum operating temperature (C)
    """
    sensor_id: str
    base_accuracy: float  # Percent
    drift_rate: float  # Percent per month
    drift_class: DriftClass
    last_calibration: datetime
    calibration_accuracy: float  # Percent achieved at calibration
    current_uncertainty: float  # Time-degraded uncertainty
    measurement_range_min: float = 0.0
    measurement_range_max: float = float('inf')
    operating_temperature_min: float = -40.0
    operating_temperature_max: float = 85.0
    calibration_certificate_id: str = ""

    def days_since_calibration(self) -> int:
        """Calculate days elapsed since last calibration."""
        delta = datetime.utcnow() - self.last_calibration
        return delta.days

    def months_since_calibration(self) -> float:
        """Calculate months elapsed since last calibration."""
        return self.days_since_calibration() / 30.44  # Average days per month

    def is_calibration_due(self, interval_months: int = 12) -> bool:
        """Check if calibration is due based on interval."""
        return self.months_since_calibration() >= interval_months

    def compute_degraded_uncertainty(self) -> float:
        """
        Compute current uncertainty including time degradation.

        Uses linear drift model: U(t) = U_0 + drift_rate * t
        where t is time since calibration in months.
        """
        months = self.months_since_calibration()
        degraded = self.calibration_accuracy + (self.drift_rate * months)
        return degraded


@dataclass(frozen=True)
class SensorRegistration:
    """
    Registration record for a sensor in the uncertainty tracking system.

    Attributes:
        sensor_id: Unique sensor identifier
        sensor_type: Type of sensor (pressure, temperature, flow, etc.)
        manufacturer: Sensor manufacturer
        model: Sensor model number
        serial_number: Unique serial number
        installation_date: When sensor was installed
        location: Physical location identifier
        uncertainty: Current uncertainty profile
        registration_timestamp: When registered in system
    """
    sensor_id: str
    sensor_type: str
    manufacturer: str
    model: str
    serial_number: str
    installation_date: datetime
    location: str
    uncertainty: SensorUncertainty
    registration_timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SensorFlag:
    """
    Flag indicating a sensor with high uncertainty requiring attention.

    Attributes:
        sensor_id: Flagged sensor identifier
        current_uncertainty: Current uncertainty level (%)
        threshold_exceeded: Which threshold was exceeded
        days_since_calibration: Days since last calibration
        recommended_action: Suggested corrective action
        priority: Flag priority (1=critical, 2=high, 3=medium)
        timestamp: When flag was raised
    """
    sensor_id: str
    current_uncertainty: float
    threshold_exceeded: str
    days_since_calibration: int
    recommended_action: str
    priority: int
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True)
class PropertyUncertainty:
    """
    Uncertainty attached to a computed thermodynamic property.

    Attributes:
        property_name: Name of the property (e.g., "enthalpy", "entropy")
        value: Computed property value
        uncertainty: Absolute uncertainty (same units as value)
        relative_uncertainty: Uncertainty as percentage
        confidence_level: Confidence level for uncertainty bounds
        unit: Physical unit of the property
        contributing_sensors: Sensors contributing to this uncertainty
        computation_method: How property was computed
        provenance_hash: Audit trail hash
    """
    property_name: str
    value: float
    uncertainty: float
    relative_uncertainty: float
    confidence_level: ConfidenceLevel
    unit: str
    contributing_sensors: List[str]
    computation_method: str
    provenance_hash: str = ""

    def __post_init__(self):
        """Compute provenance hash if not provided."""
        if not self.provenance_hash:
            hash_data = {
                "property_name": self.property_name,
                "value": self.value,
                "uncertainty": self.uncertainty,
                "confidence_level": self.confidence_level.value,
                "contributing_sensors": sorted(self.contributing_sensors),
                "computation_method": self.computation_method
            }
            hash_str = json.dumps(hash_data, sort_keys=True)
            computed_hash = hashlib.sha256(hash_str.encode()).hexdigest()
            object.__setattr__(self, 'provenance_hash', computed_hash)

    def to_uncertain_value(self) -> UncertainValue:
        """Convert to UncertainValue for propagation."""
        z_95 = 1.96
        # Adjust uncertainty based on confidence level
        if self.confidence_level == ConfidenceLevel.CI_95:
            std = self.uncertainty / z_95
        else:
            # Scale appropriately
            z_scores = {
                ConfidenceLevel.CI_68: 1.0,
                ConfidenceLevel.CI_90: 1.645,
                ConfidenceLevel.CI_95: 1.96,
                ConfidenceLevel.CI_99: 2.576,
                ConfidenceLevel.CI_99_7: 3.0
            }
            z = z_scores.get(self.confidence_level, 1.96)
            std = self.uncertainty / z

        return UncertainValue(
            mean=self.value,
            std=std,
            lower_95=self.value - z_95 * std,
            upper_95=self.value + z_95 * std,
            unit=self.unit,
            source_id=self.property_name
        )


@dataclass(frozen=True)
class PropagatedUncertainty:
    """
    Result of uncertainty propagation through a calculation.

    Contains complete information about how input uncertainties
    contribute to output uncertainty for transparency and debugging.

    Attributes:
        output_name: Name of the computed output
        value: Computed output value
        uncertainty: Propagated uncertainty (1-sigma)
        confidence_interval_95: 95% confidence interval (lower, upper)
        contributing_inputs: Input uncertainties and their contributions
        sensitivity_coefficients: Partial derivatives for each input
        dominant_contributor: Input contributing most to uncertainty
        propagation_method: Method used (jacobian, monte_carlo, etc.)
        computation_time_ms: Time taken for propagation
        provenance_hash: Audit trail hash
    """
    output_name: str
    value: float
    uncertainty: float
    confidence_interval_95: Tuple[float, float]
    contributing_inputs: Dict[str, UncertainValue]
    sensitivity_coefficients: Dict[str, float]
    dominant_contributor: str
    propagation_method: str
    computation_time_ms: float
    provenance_hash: str = ""

    def __post_init__(self):
        """Compute provenance hash if not provided."""
        if not self.provenance_hash:
            hash_data = {
                "output_name": self.output_name,
                "value": self.value,
                "uncertainty": self.uncertainty,
                "confidence_interval_95": self.confidence_interval_95,
                "contributing_inputs": {
                    k: {"mean": v.mean, "std": v.std}
                    for k, v in self.contributing_inputs.items()
                },
                "sensitivity_coefficients": self.sensitivity_coefficients,
                "propagation_method": self.propagation_method
            }
            hash_str = json.dumps(hash_data, sort_keys=True, default=str)
            computed_hash = hashlib.sha256(hash_str.encode()).hexdigest()
            object.__setattr__(self, 'provenance_hash', computed_hash)

    def relative_uncertainty_percent(self) -> float:
        """Return uncertainty as percentage of value."""
        if abs(self.value) < 1e-10:
            return float('inf') if self.uncertainty > 0 else 0.0
        return (self.uncertainty / abs(self.value)) * 100.0

    def get_contribution_breakdown(self) -> Dict[str, float]:
        """
        Get percentage contribution of each input to total uncertainty.

        Uses variance decomposition: contribution_i = (dF/dx_i * sigma_i)^2 / sigma_total^2
        """
        if self.uncertainty < 1e-10:
            return {k: 0.0 for k in self.contributing_inputs}

        total_variance = self.uncertainty ** 2
        contributions = {}

        for input_name, input_val in self.contributing_inputs.items():
            sensitivity = self.sensitivity_coefficients.get(input_name, 0.0)
            variance_contribution = (sensitivity * input_val.std) ** 2
            contributions[input_name] = (variance_contribution / total_variance) * 100.0

        return contributions


@dataclass(frozen=True)
class MonteCarloResult:
    """
    Result from Monte Carlo uncertainty propagation.

    Attributes:
        output_name: Name of the computed output
        mean: Sample mean
        std: Sample standard deviation
        median: Sample median
        percentiles: Dictionary of percentile values
        samples: Raw samples (optional, for debugging)
        n_samples: Number of Monte Carlo samples used
        convergence_achieved: Whether convergence criteria met
        seed: Random seed for reproducibility
        provenance_hash: Audit trail hash
    """
    output_name: str
    mean: float
    std: float
    median: float
    percentiles: Dict[float, float]  # e.g., {2.5: value, 97.5: value}
    n_samples: int
    convergence_achieved: bool
    seed: int
    samples: Optional[List[float]] = None
    provenance_hash: str = ""

    def __post_init__(self):
        """Compute provenance hash if not provided."""
        if not self.provenance_hash:
            hash_data = {
                "output_name": self.output_name,
                "mean": self.mean,
                "std": self.std,
                "median": self.median,
                "n_samples": self.n_samples,
                "seed": self.seed
            }
            hash_str = json.dumps(hash_data, sort_keys=True)
            computed_hash = hashlib.sha256(hash_str.encode()).hexdigest()
            object.__setattr__(self, 'provenance_hash', computed_hash)

    @property
    def confidence_interval_95(self) -> Tuple[float, float]:
        """Get 95% confidence interval from percentiles."""
        return (
            self.percentiles.get(2.5, self.mean - 1.96 * self.std),
            self.percentiles.get(97.5, self.mean + 1.96 * self.std)
        )

    def to_uncertain_value(self) -> UncertainValue:
        """Convert to UncertainValue for further propagation."""
        lower_95, upper_95 = self.confidence_interval_95
        return UncertainValue(
            mean=self.mean,
            std=self.std,
            lower_95=lower_95,
            upper_95=upper_95,
            distribution_type=DistributionType.EMPIRICAL,
            source_id=self.output_name
        )


@dataclass(frozen=True)
class Distribution:
    """
    Probability distribution specification for Monte Carlo sampling.

    Attributes:
        distribution_type: Type of distribution
        parameters: Distribution-specific parameters
        bounds: Optional truncation bounds
    """
    distribution_type: DistributionType
    parameters: Dict[str, float]  # e.g., {"mean": 100, "std": 5}
    bounds: Optional[Tuple[float, float]] = None

    @classmethod
    def normal(cls, mean: float, std: float) -> "Distribution":
        """Create normal distribution."""
        return cls(
            distribution_type=DistributionType.NORMAL,
            parameters={"mean": mean, "std": std}
        )

    @classmethod
    def uniform(cls, low: float, high: float) -> "Distribution":
        """Create uniform distribution."""
        return cls(
            distribution_type=DistributionType.UNIFORM,
            parameters={"low": low, "high": high}
        )

    @classmethod
    def triangular(cls, low: float, mode: float, high: float) -> "Distribution":
        """Create triangular distribution."""
        return cls(
            distribution_type=DistributionType.TRIANGULAR,
            parameters={"low": low, "mode": mode, "high": high}
        )

    @classmethod
    def from_uncertain_value(cls, uv: UncertainValue) -> "Distribution":
        """Create distribution from UncertainValue."""
        if uv.distribution_type == DistributionType.NORMAL:
            return cls.normal(uv.mean, uv.std)
        elif uv.distribution_type == DistributionType.UNIFORM:
            return cls.uniform(uv.lower_95, uv.upper_95)
        else:
            # Default to normal
            return cls.normal(uv.mean, uv.std)


@dataclass(frozen=True)
class UncertaintyBreakdown:
    """
    Detailed breakdown of uncertainty contributions for reporting.

    Attributes:
        total_uncertainty: Combined uncertainty value
        contributions: Individual contributions by source
        dominant_sources: Top contributors to uncertainty
        recommendations: Suggested improvements
        visualization_data: Data for uncertainty visualization
    """
    total_uncertainty: float
    total_uncertainty_percent: float
    contributions: Dict[str, float]  # Source -> contribution (%)
    dominant_sources: List[str]  # Ordered by contribution
    recommendations: List[str]
    visualization_data: Dict[str, any] = field(default_factory=dict)

    def get_pareto_sources(self, cumulative_threshold: float = 80.0) -> List[str]:
        """
        Get sources contributing to specified cumulative percentage.

        Args:
            cumulative_threshold: Cumulative contribution threshold (%)

        Returns:
            List of sources contributing to threshold
        """
        sorted_contributions = sorted(
            self.contributions.items(),
            key=lambda x: x[1],
            reverse=True
        )

        cumulative = 0.0
        pareto_sources = []

        for source, contribution in sorted_contributions:
            pareto_sources.append(source)
            cumulative += contribution
            if cumulative >= cumulative_threshold:
                break

        return pareto_sources


@dataclass(frozen=True)
class UncertaintySource:
    """
    Individual uncertainty source with contribution analysis.

    Attributes:
        source_id: Identifier of the uncertainty source
        source_type: Type (sensor, model, parameter, etc.)
        contribution_percent: Contribution to total uncertainty
        current_uncertainty: Current uncertainty level
        reducibility: How much uncertainty could be reduced
        improvement_cost: Estimated cost to improve
        improvement_benefit: Expected uncertainty reduction
    """
    source_id: str
    source_type: str
    contribution_percent: float
    current_uncertainty: float
    reducibility: str  # "high", "medium", "low", "fixed"
    improvement_cost: str  # "low", "medium", "high"
    improvement_benefit: float  # Expected reduction in total uncertainty (%)
