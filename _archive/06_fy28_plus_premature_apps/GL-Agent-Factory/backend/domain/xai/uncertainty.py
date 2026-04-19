# -*- coding: utf-8 -*-
"""
Uncertainty Quantification Module for GreenLang Agents
======================================================

Provides uncertainty quantification, propagation, and bounds calculation
for all agent outputs to ensure reliable confidence intervals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel, Field


class UncertaintyType(str, Enum):
    """Types of uncertainty in measurements and calculations."""
    ALEATORY = "aleatory"  # Inherent randomness (irreducible)
    EPISTEMIC = "epistemic"  # Lack of knowledge (reducible)
    MODEL = "model"  # Model uncertainty
    MEASUREMENT = "measurement"  # Sensor/measurement uncertainty
    PARAMETER = "parameter"  # Parameter estimation uncertainty
    NUMERICAL = "numerical"  # Numerical/computational uncertainty
    COMBINED = "combined"  # Combined uncertainty


class DistributionType(str, Enum):
    """Probability distribution types for uncertainty."""
    NORMAL = "normal"
    UNIFORM = "uniform"
    TRIANGULAR = "triangular"
    LOGNORMAL = "lognormal"
    BETA = "beta"
    STUDENT_T = "student_t"
    RECTANGULAR = "rectangular"


class CoverageLevel(str, Enum):
    """Coverage probability levels for confidence intervals."""
    P68 = "68%"  # 1 sigma
    P90 = "90%"
    P95 = "95%"  # 2 sigma (commonly used)
    P99 = "99%"
    P99_7 = "99.7%"  # 3 sigma


# Coverage factors for normal distribution
COVERAGE_FACTORS = {
    CoverageLevel.P68: 1.0,
    CoverageLevel.P90: 1.645,
    CoverageLevel.P95: 1.96,
    CoverageLevel.P99: 2.576,
    CoverageLevel.P99_7: 3.0,
}


@dataclass
class UncertaintySource:
    """Single source of uncertainty."""
    name: str
    uncertainty_type: UncertaintyType
    distribution: DistributionType = DistributionType.NORMAL
    standard_uncertainty: float = 0.0  # Standard uncertainty (1 sigma)
    degrees_of_freedom: int = 100  # Effective DoF for t-distribution
    sensitivity_coefficient: float = 1.0
    correlation_id: Optional[str] = None  # For correlated uncertainties
    source_reference: Optional[str] = None  # Standard/calibration source


@dataclass
class UncertaintyBound:
    """Uncertainty bounds for a value."""
    value: float
    standard_uncertainty: float
    lower_bound: float
    upper_bound: float
    coverage_level: CoverageLevel = CoverageLevel.P95
    coverage_factor: float = 1.96
    unit: str = ""

    @property
    def relative_uncertainty(self) -> float:
        """Calculate relative uncertainty as percentage."""
        if abs(self.value) < 1e-10:
            return 0.0
        return (self.standard_uncertainty / abs(self.value)) * 100

    @property
    def expanded_uncertainty(self) -> float:
        """Calculate expanded uncertainty."""
        return self.standard_uncertainty * self.coverage_factor

    def to_string(self, precision: int = 3) -> str:
        """Format as value ± uncertainty string."""
        return f"{self.value:.{precision}f} ± {self.expanded_uncertainty:.{precision}f} {self.unit}".strip()


class PropagatedUncertainty(BaseModel):
    """
    Complete uncertainty analysis result with propagation details.

    Implements GUM (Guide to Uncertainty in Measurement) methodology.
    """
    result_value: float
    standard_uncertainty: float
    expanded_uncertainty: float
    coverage_level: str = "95%"
    coverage_factor: float = 1.96

    # Bounds
    lower_bound: float
    upper_bound: float

    # Sources
    uncertainty_sources: List[Dict[str, Any]] = Field(default_factory=list)
    dominant_source: Optional[str] = None

    # Sensitivity analysis
    sensitivity_coefficients: Dict[str, float] = Field(default_factory=dict)
    contribution_percentages: Dict[str, float] = Field(default_factory=dict)

    # Metadata
    unit: str = ""
    method: str = "GUM"  # GUM or Monte Carlo
    effective_dof: int = 100

    def relative_uncertainty_percent(self) -> float:
        """Get relative uncertainty as percentage."""
        if abs(self.result_value) < 1e-10:
            return 0.0
        return (self.standard_uncertainty / abs(self.result_value)) * 100


class UncertaintyQuantifier:
    """
    Engine for quantifying and propagating uncertainty through calculations.

    Implements ISO GUM (Guide to Uncertainty in Measurement) methodology
    with support for both analytical and Monte Carlo propagation.
    """

    def __init__(self, default_coverage: CoverageLevel = CoverageLevel.P95):
        self.default_coverage = default_coverage
        self.correlation_matrix: Dict[Tuple[str, str], float] = {}

    def type_a_uncertainty(
        self,
        measurements: List[float],
        confidence_in_mean: bool = True
    ) -> UncertaintySource:
        """
        Calculate Type A uncertainty from repeated measurements.

        Type A: Statistical analysis of series of observations.

        Args:
            measurements: List of measurement values
            confidence_in_mean: If True, return uncertainty of mean (s/√n)

        Returns:
            UncertaintySource with calculated standard uncertainty
        """
        n = len(measurements)
        if n < 2:
            return UncertaintySource(
                name="type_a_measurement",
                uncertainty_type=UncertaintyType.MEASUREMENT,
                standard_uncertainty=0.0,
                degrees_of_freedom=0,
            )

        mean = sum(measurements) / n
        variance = sum((x - mean) ** 2 for x in measurements) / (n - 1)
        std_dev = math.sqrt(variance)

        if confidence_in_mean:
            # Standard uncertainty of the mean
            standard_uncertainty = std_dev / math.sqrt(n)
        else:
            # Standard uncertainty of individual measurement
            standard_uncertainty = std_dev

        return UncertaintySource(
            name="type_a_measurement",
            uncertainty_type=UncertaintyType.MEASUREMENT,
            distribution=DistributionType.STUDENT_T,
            standard_uncertainty=standard_uncertainty,
            degrees_of_freedom=n - 1,
        )

    def type_b_uncertainty(
        self,
        name: str,
        half_width: float,
        distribution: DistributionType = DistributionType.RECTANGULAR,
        confidence_level: Optional[float] = None,
        source_reference: Optional[str] = None,
    ) -> UncertaintySource:
        """
        Calculate Type B uncertainty from non-statistical information.

        Type B: Evaluation by scientific judgment using all available information.

        Args:
            name: Uncertainty source name
            half_width: Half-width of uncertainty interval (±a)
            distribution: Assumed distribution type
            confidence_level: Confidence level if from spec sheet
            source_reference: Reference to source document

        Returns:
            UncertaintySource with calculated standard uncertainty
        """
        # Calculate standard uncertainty based on distribution
        if distribution == DistributionType.RECTANGULAR:
            # u = a / √3 for rectangular distribution
            standard_uncertainty = half_width / math.sqrt(3)
        elif distribution == DistributionType.TRIANGULAR:
            # u = a / √6 for triangular distribution
            standard_uncertainty = half_width / math.sqrt(6)
        elif distribution == DistributionType.NORMAL:
            # If confidence level given, divide by coverage factor
            if confidence_level is not None:
                if confidence_level >= 0.99:
                    k = 2.576
                elif confidence_level >= 0.95:
                    k = 1.96
                elif confidence_level >= 0.90:
                    k = 1.645
                else:
                    k = 1.0
                standard_uncertainty = half_width / k
            else:
                standard_uncertainty = half_width
        else:
            # Default to rectangular
            standard_uncertainty = half_width / math.sqrt(3)

        return UncertaintySource(
            name=name,
            uncertainty_type=UncertaintyType.PARAMETER,
            distribution=distribution,
            standard_uncertainty=standard_uncertainty,
            degrees_of_freedom=50,  # Type B typically has large DoF
            source_reference=source_reference,
        )

    def instrument_uncertainty(
        self,
        name: str,
        accuracy_spec: float,
        resolution: float,
        stability: float = 0.0,
        calibration_uncertainty: float = 0.0,
        source_reference: Optional[str] = None,
    ) -> List[UncertaintySource]:
        """
        Calculate combined instrument uncertainty from specifications.

        Args:
            name: Instrument name
            accuracy_spec: Manufacturer accuracy (±)
            resolution: Resolution/readability
            stability: Drift/stability uncertainty
            calibration_uncertainty: Calibration certificate uncertainty
            source_reference: Instrument model/calibration cert

        Returns:
            List of UncertaintySource components
        """
        sources = []

        # Accuracy (usually rectangular)
        sources.append(UncertaintySource(
            name=f"{name}_accuracy",
            uncertainty_type=UncertaintyType.MEASUREMENT,
            distribution=DistributionType.RECTANGULAR,
            standard_uncertainty=accuracy_spec / math.sqrt(3),
            source_reference=source_reference,
        ))

        # Resolution (rectangular, half resolution)
        sources.append(UncertaintySource(
            name=f"{name}_resolution",
            uncertainty_type=UncertaintyType.MEASUREMENT,
            distribution=DistributionType.RECTANGULAR,
            standard_uncertainty=(resolution / 2) / math.sqrt(3),
        ))

        # Stability if provided
        if stability > 0:
            sources.append(UncertaintySource(
                name=f"{name}_stability",
                uncertainty_type=UncertaintyType.MEASUREMENT,
                distribution=DistributionType.RECTANGULAR,
                standard_uncertainty=stability / math.sqrt(3),
            ))

        # Calibration uncertainty if provided
        if calibration_uncertainty > 0:
            sources.append(UncertaintySource(
                name=f"{name}_calibration",
                uncertainty_type=UncertaintyType.MEASUREMENT,
                distribution=DistributionType.NORMAL,
                standard_uncertainty=calibration_uncertainty,
            ))

        return sources

    def combine_uncertainties(
        self,
        sources: List[UncertaintySource],
        coverage: CoverageLevel = CoverageLevel.P95,
    ) -> UncertaintyBound:
        """
        Combine multiple uncertainty sources (RSS for uncorrelated).

        Uses root-sum-square for uncorrelated uncertainties as per GUM.

        Args:
            sources: List of uncertainty sources
            coverage: Coverage level for expanded uncertainty

        Returns:
            Combined UncertaintyBound
        """
        if not sources:
            return UncertaintyBound(
                value=0.0,
                standard_uncertainty=0.0,
                lower_bound=0.0,
                upper_bound=0.0,
            )

        # RSS combination for uncorrelated sources
        combined_variance = sum(
            (s.sensitivity_coefficient * s.standard_uncertainty) ** 2
            for s in sources
        )
        combined_std = math.sqrt(combined_variance)

        # Coverage factor
        k = COVERAGE_FACTORS.get(coverage, 1.96)
        expanded = combined_std * k

        return UncertaintyBound(
            value=0.0,  # Value must be set separately
            standard_uncertainty=combined_std,
            lower_bound=-expanded,
            upper_bound=expanded,
            coverage_level=coverage,
            coverage_factor=k,
        )

    def propagate_uncertainty(
        self,
        func: Callable[..., float],
        input_values: Dict[str, float],
        input_uncertainties: Dict[str, UncertaintySource],
        coverage: CoverageLevel = CoverageLevel.P95,
        delta: float = 1e-6,
    ) -> PropagatedUncertainty:
        """
        Propagate uncertainty through a calculation using GUM linear method.

        Uses numerical differentiation to calculate sensitivity coefficients.

        Args:
            func: Calculation function f(x1, x2, ...)
            input_values: Dictionary of input values
            input_uncertainties: Dictionary of input uncertainties
            coverage: Coverage level for expanded uncertainty
            delta: Step size for numerical differentiation

        Returns:
            Complete PropagatedUncertainty result
        """
        # Calculate result value
        result = func(**input_values)

        # Calculate sensitivity coefficients via numerical differentiation
        sensitivity_coefficients = {}
        for param, value in input_values.items():
            if param in input_uncertainties:
                # Perturb parameter
                perturbed_inputs = input_values.copy()
                h = max(abs(value) * delta, delta)
                perturbed_inputs[param] = value + h

                try:
                    result_plus = func(**perturbed_inputs)
                    sensitivity_coefficients[param] = (result_plus - result) / h
                except Exception:
                    sensitivity_coefficients[param] = 0.0

        # Calculate uncertainty contributions
        contributions = {}
        total_variance = 0.0

        for param, unc_source in input_uncertainties.items():
            c = sensitivity_coefficients.get(param, 1.0)
            u = unc_source.standard_uncertainty
            contribution = (c * u) ** 2
            contributions[param] = contribution
            total_variance += contribution

        # Combined standard uncertainty
        combined_std = math.sqrt(total_variance) if total_variance > 0 else 0.0

        # Coverage factor
        k = COVERAGE_FACTORS.get(coverage, 1.96)
        expanded = combined_std * k

        # Calculate contribution percentages
        contribution_pcts = {}
        dominant = None
        max_contrib = 0.0

        for param, contrib in contributions.items():
            pct = (contrib / total_variance * 100) if total_variance > 0 else 0.0
            contribution_pcts[param] = pct
            if pct > max_contrib:
                max_contrib = pct
                dominant = param

        return PropagatedUncertainty(
            result_value=result,
            standard_uncertainty=combined_std,
            expanded_uncertainty=expanded,
            coverage_level=coverage.value,
            coverage_factor=k,
            lower_bound=result - expanded,
            upper_bound=result + expanded,
            uncertainty_sources=[
                {
                    "name": s.name,
                    "type": s.uncertainty_type.value,
                    "standard_uncertainty": s.standard_uncertainty,
                }
                for s in input_uncertainties.values()
            ],
            dominant_source=dominant,
            sensitivity_coefficients={k: round(v, 6) for k, v in sensitivity_coefficients.items()},
            contribution_percentages={k: round(v, 2) for k, v in contribution_pcts.items()},
            method="GUM",
        )

    def monte_carlo_propagation(
        self,
        func: Callable[..., float],
        input_values: Dict[str, float],
        input_uncertainties: Dict[str, UncertaintySource],
        n_samples: int = 10000,
        coverage: CoverageLevel = CoverageLevel.P95,
        seed: Optional[int] = None,
    ) -> PropagatedUncertainty:
        """
        Propagate uncertainty using Monte Carlo simulation.

        More accurate than GUM for non-linear functions.

        Args:
            func: Calculation function
            input_values: Dictionary of input values
            input_uncertainties: Dictionary of input uncertainties
            n_samples: Number of Monte Carlo samples
            coverage: Coverage level for intervals
            seed: Random seed for reproducibility

        Returns:
            Complete PropagatedUncertainty result
        """
        if seed is not None:
            np.random.seed(seed)

        # Generate samples for each input
        samples = {}
        for param, value in input_values.items():
            if param in input_uncertainties:
                unc = input_uncertainties[param]
                if unc.distribution == DistributionType.NORMAL:
                    samples[param] = np.random.normal(value, unc.standard_uncertainty, n_samples)
                elif unc.distribution == DistributionType.UNIFORM:
                    half_width = unc.standard_uncertainty * math.sqrt(3)
                    samples[param] = np.random.uniform(value - half_width, value + half_width, n_samples)
                elif unc.distribution == DistributionType.TRIANGULAR:
                    half_width = unc.standard_uncertainty * math.sqrt(6)
                    samples[param] = np.random.triangular(value - half_width, value, value + half_width, n_samples)
                else:
                    samples[param] = np.random.normal(value, unc.standard_uncertainty, n_samples)
            else:
                samples[param] = np.full(n_samples, value)

        # Evaluate function for all samples
        results = np.zeros(n_samples)
        for i in range(n_samples):
            try:
                sample_inputs = {k: v[i] for k, v in samples.items()}
                results[i] = func(**sample_inputs)
            except Exception:
                results[i] = np.nan

        # Remove NaN values
        results = results[~np.isnan(results)]

        # Calculate statistics
        mean_result = np.mean(results)
        std_result = np.std(results, ddof=1)

        # Calculate percentile bounds based on coverage
        coverage_pct = float(coverage.value.replace("%", ""))
        alpha = (100 - coverage_pct) / 2
        lower_percentile = np.percentile(results, alpha)
        upper_percentile = np.percentile(results, 100 - alpha)

        # Estimate coverage factor
        k_estimated = (upper_percentile - lower_percentile) / (2 * std_result) if std_result > 0 else 1.96

        return PropagatedUncertainty(
            result_value=mean_result,
            standard_uncertainty=std_result,
            expanded_uncertainty=std_result * k_estimated,
            coverage_level=coverage.value,
            coverage_factor=k_estimated,
            lower_bound=lower_percentile,
            upper_bound=upper_percentile,
            uncertainty_sources=[
                {
                    "name": s.name,
                    "type": s.uncertainty_type.value,
                    "standard_uncertainty": s.standard_uncertainty,
                }
                for s in input_uncertainties.values()
            ],
            method="Monte Carlo",
            effective_dof=len(results) - 1,
        )

    def emission_factor_uncertainty(
        self,
        emission_factor: float,
        uncertainty_percent: float,
        source: str,
        distribution: DistributionType = DistributionType.LOGNORMAL,
    ) -> UncertaintySource:
        """
        Create uncertainty source for emission factors (commonly lognormal).

        Args:
            emission_factor: Emission factor value
            uncertainty_percent: Uncertainty as percentage (±%)
            source: Source reference (e.g., "IPCC 2006 Vol 2 Ch 2")
            distribution: Distribution type (lognormal common for EFs)

        Returns:
            UncertaintySource for the emission factor
        """
        # Convert percentage to standard uncertainty
        # For lognormal, use geometric standard deviation
        relative_unc = uncertainty_percent / 100

        if distribution == DistributionType.LOGNORMAL:
            # For lognormal at 95% CI, k ≈ 2
            standard_uncertainty = emission_factor * (relative_unc / 2)
        else:
            standard_uncertainty = emission_factor * (relative_unc / 1.96)

        return UncertaintySource(
            name=f"emission_factor_{source.replace(' ', '_')}",
            uncertainty_type=UncertaintyType.PARAMETER,
            distribution=distribution,
            standard_uncertainty=standard_uncertainty,
            source_reference=source,
        )


# Module-level singleton
_quantifier_instance: Optional[UncertaintyQuantifier] = None


def get_uncertainty_quantifier() -> UncertaintyQuantifier:
    """Get or create the global uncertainty quantifier instance."""
    global _quantifier_instance
    if _quantifier_instance is None:
        _quantifier_instance = UncertaintyQuantifier()
    return _quantifier_instance
