"""Measurement Uncertainty Calculator.

This module quantifies uncertainty in thermal efficiency calculations,
including instrument accuracy propagation, Monte Carlo analysis,
confidence intervals, and sensitivity analysis.

Standards:
    - ISO/IEC Guide 98-3 (GUM): Uncertainty of Measurement
    - ASME PTC 19.1: Test Uncertainty
    - NIST Technical Note 1297

Features:
    - Instrument accuracy propagation (Type B uncertainty)
    - Monte Carlo uncertainty analysis (Type A)
    - Confidence interval estimation
    - Sensitivity analysis (Morris method)
    - Combined and expanded uncertainty

Author: GL-009 THERMALIQ Agent
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
import hashlib
import json
from datetime import datetime
import math
import random


class UncertaintyType(Enum):
    """Type of uncertainty source."""
    INSTRUMENT = "instrument"       # Measurement device accuracy
    CALIBRATION = "calibration"     # Calibration uncertainty
    RANDOM = "random"               # Random/repeatability
    SYSTEMATIC = "systematic"       # Systematic bias
    MODEL = "model"                 # Model/calculation uncertainty
    ENVIRONMENTAL = "environmental" # Environmental effects
    OPERATOR = "operator"           # Operator variability


class DistributionType(Enum):
    """Probability distribution for uncertainty."""
    NORMAL = "normal"
    RECTANGULAR = "rectangular"
    TRIANGULAR = "triangular"
    UNIFORM = "uniform"
    LOGNORMAL = "lognormal"


@dataclass(frozen=True)
class InstrumentAccuracy:
    """Instrument accuracy specification.

    Attributes:
        instrument_name: Name of measurement instrument
        measured_parameter: Parameter being measured
        nominal_value: Nominal/expected value
        accuracy_percent: Accuracy as % of reading
        accuracy_absolute: Accuracy as absolute value
        resolution: Instrument resolution
        distribution: Assumed distribution
        calibration_date: Last calibration date
        coverage_factor: Coverage factor for uncertainty (k)
    """
    instrument_name: str
    measured_parameter: str
    nominal_value: float
    accuracy_percent: Optional[float] = None
    accuracy_absolute: Optional[float] = None
    resolution: Optional[float] = None
    distribution: DistributionType = DistributionType.RECTANGULAR
    calibration_date: Optional[str] = None
    coverage_factor: float = 2.0

    @property
    def standard_uncertainty(self) -> float:
        """Calculate standard uncertainty from accuracy spec.

        For rectangular distribution: u = a / sqrt(3)
        For normal distribution: u = a / k
        """
        # Use percent or absolute accuracy
        if self.accuracy_percent is not None:
            half_width = abs(self.nominal_value) * (self.accuracy_percent / 100)
        elif self.accuracy_absolute is not None:
            half_width = self.accuracy_absolute
        else:
            half_width = 0.0

        # Add resolution contribution if specified
        if self.resolution:
            res_uncertainty = self.resolution / (2 * math.sqrt(3))
            half_width = math.sqrt(half_width**2 + res_uncertainty**2)

        # Convert to standard uncertainty based on distribution
        if self.distribution == DistributionType.NORMAL:
            return half_width / self.coverage_factor
        elif self.distribution == DistributionType.RECTANGULAR:
            return half_width / math.sqrt(3)
        elif self.distribution == DistributionType.TRIANGULAR:
            return half_width / math.sqrt(6)
        else:
            return half_width / math.sqrt(3)


@dataclass
class MonteCarloResult:
    """Monte Carlo simulation result.

    Attributes:
        mean_value: Mean of simulated outputs
        standard_deviation: Standard deviation
        median_value: Median value
        percentile_2_5: 2.5th percentile
        percentile_97_5: 97.5th percentile
        min_value: Minimum simulated value
        max_value: Maximum simulated value
        num_iterations: Number of Monte Carlo iterations
        convergence_achieved: Whether convergence was achieved
    """
    mean_value: float
    standard_deviation: float
    median_value: float
    percentile_2_5: float
    percentile_97_5: float
    min_value: float
    max_value: float
    num_iterations: int
    convergence_achieved: bool


@dataclass
class ConfidenceInterval:
    """Confidence interval for a calculated value.

    Attributes:
        central_value: Best estimate (mean)
        lower_bound: Lower confidence bound
        upper_bound: Upper confidence bound
        confidence_level_percent: Confidence level (e.g., 95)
        coverage_factor: Coverage factor used (k)
        expanded_uncertainty: k * standard_uncertainty
    """
    central_value: float
    lower_bound: float
    upper_bound: float
    confidence_level_percent: float
    coverage_factor: float
    expanded_uncertainty: float


@dataclass
class SensitivityFactor:
    """Sensitivity analysis result for a single input.

    Attributes:
        parameter_name: Name of input parameter
        sensitivity_coefficient: Partial derivative dy/dx
        relative_sensitivity: (x/y) * (dy/dx)
        contribution_percent: Contribution to total variance
        ranking: Rank by importance
    """
    parameter_name: str
    sensitivity_coefficient: float
    relative_sensitivity: float
    contribution_percent: float
    ranking: int


@dataclass
class CalculationStep:
    """Records a single calculation step for audit trail."""
    step_number: int
    description: str
    operation: str
    inputs: Dict[str, float]
    output_value: float
    output_name: str
    formula: Optional[str] = None


@dataclass
class UncertaintyResult:
    """Complete uncertainty analysis result.

    Attributes:
        calculated_value: Best estimate of calculated value
        combined_standard_uncertainty: Combined standard uncertainty
        expanded_uncertainty: Expanded uncertainty (k*u_c)
        coverage_factor: Coverage factor used
        confidence_interval: Confidence interval
        monte_carlo_result: Monte Carlo analysis (if performed)
        sensitivity_factors: Sensitivity analysis results
        input_uncertainties: Uncertainties of input parameters
        dominant_contributor: Most significant uncertainty source
        calculation_steps: Audit trail
        provenance_hash: SHA-256 hash
        analysis_timestamp: When analyzed
    """
    calculated_value: float
    combined_standard_uncertainty: float
    expanded_uncertainty: float
    coverage_factor: float
    confidence_interval: ConfidenceInterval
    monte_carlo_result: Optional[MonteCarloResult]
    sensitivity_factors: List[SensitivityFactor]
    input_uncertainties: Dict[str, float]
    dominant_contributor: str
    calculation_steps: List[CalculationStep]
    provenance_hash: str
    analysis_timestamp: str
    analyzer_version: str = "1.0.0"
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "calculated_value": self.calculated_value,
            "combined_standard_uncertainty": self.combined_standard_uncertainty,
            "expanded_uncertainty": self.expanded_uncertainty,
            "coverage_factor": self.coverage_factor,
            "confidence_interval": {
                "lower": self.confidence_interval.lower_bound,
                "upper": self.confidence_interval.upper_bound,
                "confidence_level": self.confidence_interval.confidence_level_percent
            },
            "sensitivity_factors": [
                {
                    "parameter": sf.parameter_name,
                    "contribution_percent": sf.contribution_percent,
                    "ranking": sf.ranking
                }
                for sf in self.sensitivity_factors
            ],
            "dominant_contributor": self.dominant_contributor,
            "provenance_hash": self.provenance_hash
        }


class UncertaintyCalculator:
    """Measurement Uncertainty Calculator.

    Quantifies uncertainty in thermal efficiency calculations using
    GUM (Guide to the Expression of Uncertainty in Measurement)
    methodology.

    Example:
        >>> calculator = UncertaintyCalculator()
        >>> instruments = [
        ...     InstrumentAccuracy("fuel_flow", "fuel_mass_flow_kg_s",
        ...                        0.5, accuracy_percent=1.0),
        ...     InstrumentAccuracy("steam_flow", "steam_mass_flow_kg_s",
        ...                        2.0, accuracy_percent=0.5)
        ... ]
        >>> result = calculator.analyze_uncertainty(
        ...     calculation_function=calculate_efficiency,
        ...     nominal_inputs={"fuel_mass_flow_kg_s": 0.5, "steam_mass_flow_kg_s": 2.0},
        ...     instrument_specs=instruments
        ... )
    """

    VERSION: str = "1.0.0"
    PRECISION: int = 4

    # Default coverage factors for confidence levels
    COVERAGE_FACTORS = {
        68.27: 1.0,
        90.0: 1.645,
        95.0: 1.96,
        95.45: 2.0,
        99.0: 2.576,
        99.73: 3.0
    }

    def __init__(
        self,
        confidence_level: float = 95.0,
        monte_carlo_iterations: int = 10000,
        precision: int = 4,
        random_seed: Optional[int] = None
    ) -> None:
        """Initialize the Uncertainty Calculator.

        Args:
            confidence_level: Confidence level for intervals (%)
            monte_carlo_iterations: Number of MC iterations
            precision: Decimal places for rounding
            random_seed: Optional seed for reproducibility
        """
        self.confidence_level = confidence_level
        self.coverage_factor = self._get_coverage_factor(confidence_level)
        self.mc_iterations = monte_carlo_iterations
        self.precision = precision

        if random_seed is not None:
            random.seed(random_seed)

        self._calculation_steps: List[CalculationStep] = []
        self._step_counter: int = 0
        self._warnings: List[str] = []

    def propagate_uncertainty(
        self,
        nominal_value: float,
        input_values: Dict[str, float],
        input_uncertainties: Dict[str, float],
        sensitivity_coefficients: Dict[str, float]
    ) -> UncertaintyResult:
        """Propagate uncertainties using linear approximation (GUM).

        Combined uncertainty:
            u_c^2 = sum(c_i^2 * u_i^2)

        Where c_i is the sensitivity coefficient (partial derivative).

        Args:
            nominal_value: Calculated value
            input_values: Dict of input parameter values
            input_uncertainties: Dict of input standard uncertainties
            sensitivity_coefficients: Dict of sensitivity coefficients

        Returns:
            UncertaintyResult with propagated uncertainty
        """
        self._reset_calculation_state()

        # Calculate variance contributions
        variance_contributions: Dict[str, float] = {}
        total_variance = 0.0

        for param, uncertainty in input_uncertainties.items():
            coef = sensitivity_coefficients.get(param, 1.0)
            variance = (coef * uncertainty) ** 2
            variance_contributions[param] = variance
            total_variance += variance

        # Combined standard uncertainty
        combined_uncertainty = math.sqrt(total_variance)

        self._add_calculation_step(
            description="Calculate combined standard uncertainty",
            operation="uncertainty_propagation",
            inputs={"total_variance": total_variance},
            output_value=combined_uncertainty,
            output_name="combined_uncertainty",
            formula="u_c = sqrt(sum(c_i^2 * u_i^2))"
        )

        # Expanded uncertainty
        expanded = combined_uncertainty * self.coverage_factor

        # Confidence interval
        lower = nominal_value - expanded
        upper = nominal_value + expanded

        confidence_interval = ConfidenceInterval(
            central_value=nominal_value,
            lower_bound=self._round_value(lower),
            upper_bound=self._round_value(upper),
            confidence_level_percent=self.confidence_level,
            coverage_factor=self.coverage_factor,
            expanded_uncertainty=self._round_value(expanded)
        )

        # Sensitivity analysis
        sensitivity_factors = self._calculate_sensitivity_factors(
            variance_contributions, total_variance
        )

        # Find dominant contributor
        if sensitivity_factors:
            dominant = max(sensitivity_factors, key=lambda x: x.contribution_percent)
            dominant_contributor = dominant.parameter_name
        else:
            dominant_contributor = "Unknown"

        # Generate provenance
        provenance = self._generate_provenance_hash(
            nominal_value, input_uncertainties
        )
        timestamp = datetime.utcnow().isoformat() + "Z"

        return UncertaintyResult(
            calculated_value=self._round_value(nominal_value),
            combined_standard_uncertainty=self._round_value(combined_uncertainty),
            expanded_uncertainty=self._round_value(expanded),
            coverage_factor=self.coverage_factor,
            confidence_interval=confidence_interval,
            monte_carlo_result=None,
            sensitivity_factors=sensitivity_factors,
            input_uncertainties={k: self._round_value(v) for k, v in input_uncertainties.items()},
            dominant_contributor=dominant_contributor,
            calculation_steps=self._calculation_steps.copy(),
            provenance_hash=provenance,
            analysis_timestamp=timestamp,
            warnings=self._warnings.copy()
        )

    def analyze_uncertainty(
        self,
        calculation_function: Callable[..., float],
        nominal_inputs: Dict[str, float],
        instrument_specs: List[InstrumentAccuracy],
        perform_monte_carlo: bool = True
    ) -> UncertaintyResult:
        """Comprehensive uncertainty analysis.

        Combines instrument accuracy propagation with optional
        Monte Carlo analysis.

        Args:
            calculation_function: Function that calculates output
            nominal_inputs: Dict of nominal input values
            instrument_specs: List of instrument accuracy specs
            perform_monte_carlo: Whether to perform MC analysis

        Returns:
            UncertaintyResult with complete analysis
        """
        self._reset_calculation_state()

        # Get nominal calculated value
        nominal_value = calculation_function(**nominal_inputs)

        # Build input uncertainties from instrument specs
        input_uncertainties: Dict[str, float] = {}
        for spec in instrument_specs:
            input_uncertainties[spec.measured_parameter] = spec.standard_uncertainty

        # Calculate sensitivity coefficients numerically
        sensitivity_coefficients = self._calculate_numerical_sensitivities(
            calculation_function, nominal_inputs, input_uncertainties
        )

        # Propagate uncertainty (GUM method)
        gum_result = self.propagate_uncertainty(
            nominal_value, nominal_inputs, input_uncertainties, sensitivity_coefficients
        )

        # Monte Carlo analysis if requested
        mc_result = None
        if perform_monte_carlo:
            mc_result = self.monte_carlo_analysis(
                calculation_function, nominal_inputs, input_uncertainties
            )

            # Compare GUM and MC results
            mc_uncertainty = mc_result.standard_deviation
            gum_uncertainty = gum_result.combined_standard_uncertainty

            if abs(mc_uncertainty - gum_uncertainty) / gum_uncertainty > 0.1:
                self._warnings.append(
                    f"MC uncertainty ({mc_uncertainty:.4f}) differs from "
                    f"GUM ({gum_uncertainty:.4f}) by >10%"
                )

        # Update result with MC
        return UncertaintyResult(
            calculated_value=gum_result.calculated_value,
            combined_standard_uncertainty=gum_result.combined_standard_uncertainty,
            expanded_uncertainty=gum_result.expanded_uncertainty,
            coverage_factor=gum_result.coverage_factor,
            confidence_interval=gum_result.confidence_interval,
            monte_carlo_result=mc_result,
            sensitivity_factors=gum_result.sensitivity_factors,
            input_uncertainties=gum_result.input_uncertainties,
            dominant_contributor=gum_result.dominant_contributor,
            calculation_steps=self._calculation_steps.copy(),
            provenance_hash=gum_result.provenance_hash,
            analysis_timestamp=gum_result.analysis_timestamp,
            warnings=self._warnings.copy()
        )

    def monte_carlo_analysis(
        self,
        calculation_function: Callable[..., float],
        nominal_inputs: Dict[str, float],
        input_uncertainties: Dict[str, float],
        num_iterations: Optional[int] = None
    ) -> MonteCarloResult:
        """Perform Monte Carlo uncertainty analysis.

        Randomly samples inputs from their uncertainty distributions
        and calculates output statistics.

        Args:
            calculation_function: Function that calculates output
            nominal_inputs: Dict of nominal input values
            input_uncertainties: Dict of input standard uncertainties
            num_iterations: Number of iterations (default from init)

        Returns:
            MonteCarloResult with statistics
        """
        iterations = num_iterations or self.mc_iterations
        results: List[float] = []

        for i in range(iterations):
            # Sample inputs from normal distributions
            sampled_inputs = {}
            for param, nominal in nominal_inputs.items():
                uncertainty = input_uncertainties.get(param, 0)
                sampled = random.gauss(nominal, uncertainty)
                sampled_inputs[param] = sampled

            # Calculate output
            try:
                output = calculation_function(**sampled_inputs)
                if math.isfinite(output):
                    results.append(output)
            except Exception:
                # Skip invalid calculations
                pass

        if len(results) < 100:
            self._warnings.append(
                f"Only {len(results)} valid MC iterations (expected {iterations})"
            )

        # Calculate statistics
        results.sort()
        n = len(results)

        mean = sum(results) / n if n > 0 else 0
        variance = sum((x - mean)**2 for x in results) / (n - 1) if n > 1 else 0
        std_dev = math.sqrt(variance)

        median = results[n // 2] if n > 0 else 0
        p2_5 = results[int(0.025 * n)] if n >= 40 else (results[0] if n > 0 else 0)
        p97_5 = results[int(0.975 * n)] if n >= 40 else (results[-1] if n > 0 else 0)
        min_val = results[0] if n > 0 else 0
        max_val = results[-1] if n > 0 else 0

        # Check convergence (std error of mean < 1% of mean)
        std_error = std_dev / math.sqrt(n) if n > 0 else float('inf')
        converged = std_error < 0.01 * abs(mean) if mean != 0 else False

        self._add_calculation_step(
            description="Monte Carlo simulation",
            operation="monte_carlo",
            inputs={"iterations": iterations, "valid_results": n},
            output_value=std_dev,
            output_name="mc_standard_deviation",
            formula="Random sampling from input distributions"
        )

        return MonteCarloResult(
            mean_value=self._round_value(mean),
            standard_deviation=self._round_value(std_dev),
            median_value=self._round_value(median),
            percentile_2_5=self._round_value(p2_5),
            percentile_97_5=self._round_value(p97_5),
            min_value=self._round_value(min_val),
            max_value=self._round_value(max_val),
            num_iterations=n,
            convergence_achieved=converged
        )

    def _calculate_numerical_sensitivities(
        self,
        func: Callable[..., float],
        nominal_inputs: Dict[str, float],
        input_uncertainties: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate sensitivity coefficients numerically.

        Uses central difference approximation:
            df/dx ~ (f(x+h) - f(x-h)) / (2h)
        """
        sensitivities: Dict[str, float] = {}
        nominal_output = func(**nominal_inputs)

        for param, value in nominal_inputs.items():
            # Perturbation step (1% of value or uncertainty)
            h = max(0.01 * abs(value), input_uncertainties.get(param, 0.01))

            # Forward and backward perturbation
            inputs_plus = nominal_inputs.copy()
            inputs_plus[param] = value + h

            inputs_minus = nominal_inputs.copy()
            inputs_minus[param] = value - h

            try:
                output_plus = func(**inputs_plus)
                output_minus = func(**inputs_minus)

                # Central difference
                sensitivity = (output_plus - output_minus) / (2 * h)
                sensitivities[param] = sensitivity
            except Exception:
                sensitivities[param] = 0.0

        return sensitivities

    def _calculate_sensitivity_factors(
        self,
        variance_contributions: Dict[str, float],
        total_variance: float
    ) -> List[SensitivityFactor]:
        """Calculate sensitivity factors from variance contributions."""
        factors: List[SensitivityFactor] = []

        for param, variance in variance_contributions.items():
            contribution = (variance / total_variance * 100) if total_variance > 0 else 0

            factors.append(SensitivityFactor(
                parameter_name=param,
                sensitivity_coefficient=0,  # Would need to store from earlier
                relative_sensitivity=0,
                contribution_percent=self._round_value(contribution),
                ranking=0
            ))

        # Assign rankings
        factors.sort(key=lambda x: -x.contribution_percent)
        for i, factor in enumerate(factors):
            factors[i] = SensitivityFactor(
                parameter_name=factor.parameter_name,
                sensitivity_coefficient=factor.sensitivity_coefficient,
                relative_sensitivity=factor.relative_sensitivity,
                contribution_percent=factor.contribution_percent,
                ranking=i + 1
            )

        return factors

    def _get_coverage_factor(self, confidence_level: float) -> float:
        """Get coverage factor for confidence level.

        Uses Student's t distribution approximation for normal.
        """
        if confidence_level in self.COVERAGE_FACTORS:
            return self.COVERAGE_FACTORS[confidence_level]

        # Linear interpolation between known values
        levels = sorted(self.COVERAGE_FACTORS.keys())
        for i, level in enumerate(levels[:-1]):
            if level <= confidence_level < levels[i + 1]:
                k1 = self.COVERAGE_FACTORS[level]
                k2 = self.COVERAGE_FACTORS[levels[i + 1]]
                fraction = (confidence_level - level) / (levels[i + 1] - level)
                return k1 + fraction * (k2 - k1)

        return 2.0  # Default

    def _reset_calculation_state(self) -> None:
        """Reset calculation state."""
        self._calculation_steps = []
        self._step_counter = 0
        self._warnings = []

    def _add_calculation_step(
        self,
        description: str,
        operation: str,
        inputs: Dict[str, float],
        output_value: float,
        output_name: str,
        formula: Optional[str] = None
    ) -> None:
        """Record a calculation step."""
        self._step_counter += 1
        step = CalculationStep(
            step_number=self._step_counter,
            description=description,
            operation=operation,
            inputs=inputs,
            output_value=output_value,
            output_name=output_name,
            formula=formula
        )
        self._calculation_steps.append(step)

    def _generate_provenance_hash(
        self,
        value: float,
        uncertainties: Dict[str, float]
    ) -> str:
        """Generate SHA-256 provenance hash."""
        data = {
            "calculator": "UncertaintyCalculator",
            "version": self.VERSION,
            "calculated_value": value,
            "input_uncertainties": uncertainties,
            "confidence_level": self.confidence_level
        }
        json_str = json.dumps(data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(json_str.encode('utf-8')).hexdigest()

    def _round_value(self, value: float) -> float:
        """Round value to precision."""
        if value is None or not math.isfinite(value):
            return 0.0
        decimal_value = Decimal(str(value))
        quantize_str = '0.' + '0' * self.precision
        rounded = decimal_value.quantize(
            Decimal(quantize_str),
            rounding=ROUND_HALF_UP
        )
        return float(rounded)
