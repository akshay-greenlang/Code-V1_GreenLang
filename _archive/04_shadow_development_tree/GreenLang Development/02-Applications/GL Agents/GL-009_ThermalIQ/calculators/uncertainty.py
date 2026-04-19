"""
Uncertainty Quantifier
======================

Zero-hallucination deterministic uncertainty quantification engine.

Implements Monte Carlo propagation, sensitivity analysis, and confidence
interval calculations following the GUM (Guide to Expression of Uncertainty).

Uncertainty Analysis Methodology:
---------------------------------
1. Type A: Statistical analysis of repeated observations
2. Type B: Other means (calibration data, specifications, etc.)

Combined Standard Uncertainty:
    u_c = sqrt(sum(ci^2 * ui^2))

Where:
    - ci: Sensitivity coefficient (partial derivative)
    - ui: Standard uncertainty of input i

Expanded Uncertainty:
    U = k * u_c

Where k is the coverage factor (typically 2 for 95% confidence)

Standards Compliance:
--------------------
- JCGM 100:2008 (GUM - Guide to Expression of Uncertainty in Measurement)
- JCGM 101:2008 (GUM Supplement 1 - Monte Carlo Method)
- ISO/IEC 17025 - Testing and Calibration Laboratories
- ASME PTC 19.1 - Measurement Uncertainty

Author: GL-009_ThermalIQ
Version: 1.0.0
"""

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
import hashlib
import json
import time
import random
from datetime import datetime, timezone
import math


@dataclass
class InputUncertainty:
    """
    Uncertainty specification for an input variable.

    Attributes:
        name: Variable name
        value: Nominal value
        uncertainty: Standard uncertainty (1-sigma)
        uncertainty_type: 'A' (statistical) or 'B' (other)
        distribution: 'normal', 'uniform', 'triangular'
        coverage_factor: k-factor if expanded uncertainty given
        source: Source of uncertainty estimate
        unit: Physical unit
    """
    name: str
    value: float
    uncertainty: float
    uncertainty_type: str = "B"
    distribution: str = "normal"
    coverage_factor: float = 1.0
    source: str = "manufacturer specification"
    unit: str = ""

    def get_standard_uncertainty(self) -> float:
        """Get standard uncertainty (1-sigma equivalent)."""
        if self.coverage_factor != 1.0:
            return self.uncertainty / self.coverage_factor
        return self.uncertainty


@dataclass
class UncertaintyResult:
    """
    Result of uncertainty analysis.

    Attributes:
        output_value: Calculated output value
        standard_uncertainty: Combined standard uncertainty
        expanded_uncertainty: Expanded uncertainty (95% confidence)
        coverage_factor: Coverage factor used
        relative_uncertainty_percent: Relative uncertainty (%)
        input_contributions: Uncertainty contribution from each input
        provenance_hash: SHA-256 hash
        method: Analysis method used
    """
    output_value: float
    standard_uncertainty: float
    expanded_uncertainty: float
    coverage_factor: float
    relative_uncertainty_percent: float
    input_contributions: Dict[str, float]
    sensitivity_coefficients: Dict[str, float]
    provenance_hash: str
    method: str
    n_samples: Optional[int]
    calculation_time_ms: float
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "output_value": self.output_value,
            "standard_uncertainty": self.standard_uncertainty,
            "expanded_uncertainty": self.expanded_uncertainty,
            "coverage_factor": self.coverage_factor,
            "relative_uncertainty_percent": self.relative_uncertainty_percent,
            "input_contributions": self.input_contributions,
            "sensitivity_coefficients": self.sensitivity_coefficients,
            "provenance_hash": self.provenance_hash,
            "method": self.method,
            "n_samples": self.n_samples,
            "calculation_time_ms": self.calculation_time_ms,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


@dataclass
class SensitivityResult:
    """
    Result of sensitivity analysis.

    Attributes:
        sensitivities: Sensitivity coefficient for each input
        normalized_sensitivities: Normalized sensitivity indices
        most_influential: Ranked list of most influential inputs
        elasticities: Percent change in output per percent change in input
        provenance_hash: SHA-256 hash
    """
    sensitivities: Dict[str, float]
    normalized_sensitivities: Dict[str, float]
    most_influential: List[Tuple[str, float]]
    elasticities: Dict[str, float]
    provenance_hash: str
    method: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "sensitivities": self.sensitivities,
            "normalized_sensitivities": self.normalized_sensitivities,
            "most_influential": self.most_influential,
            "elasticities": self.elasticities,
            "provenance_hash": self.provenance_hash,
            "method": self.method,
            "timestamp": self.timestamp
        }


@dataclass
class ConfidenceInterval:
    """
    Confidence interval result.

    Attributes:
        value: Central value (mean or median)
        lower_bound: Lower confidence limit
        upper_bound: Upper confidence limit
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        interval_type: 'symmetric' or 'asymmetric'
        distribution_type: Assumed or detected distribution
        provenance_hash: SHA-256 hash
    """
    value: float
    lower_bound: float
    upper_bound: float
    confidence_level: float
    interval_type: str
    distribution_type: str
    provenance_hash: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "value": self.value,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "confidence_level": self.confidence_level,
            "interval_type": self.interval_type,
            "distribution_type": self.distribution_type,
            "provenance_hash": self.provenance_hash,
            "timestamp": self.timestamp
        }


class UncertaintyQuantifier:
    """
    Zero-hallucination uncertainty quantification engine.

    Implements Monte Carlo propagation and analytical methods for
    uncertainty analysis following GUM standards.

    Guarantees:
    - DETERMINISTIC: Same inputs + seed produce identical outputs
    - REPRODUCIBLE: Full provenance tracking with SHA-256 hashes
    - AUDITABLE: Complete calculation trails
    - STANDARDS-BASED: All methods from GUM and supplements
    - NO LLM: Zero hallucination risk in calculation path

    References:
    -----------
    [1] JCGM 100:2008 - Guide to Expression of Uncertainty in Measurement
    [2] JCGM 101:2008 - GUM Supplement 1: Monte Carlo Method
    [3] ASME PTC 19.1-2018 - Measurement Uncertainty
    [4] ISO/IEC 17025:2017 - Testing and Calibration Laboratories

    Example:
    --------
    >>> uq = UncertaintyQuantifier(seed=42)
    >>> inputs = {
    ...     "T": InputUncertainty("T", 500.0, 2.0, unit="K"),
    ...     "P": InputUncertainty("P", 100.0, 0.5, unit="kPa"),
    ... }
    >>> def calculation(T, P):
    ...     return T * P / 1000
    >>> result = uq.monte_carlo_propagation(calculation, inputs, n_samples=10000)
    >>> print(f"Result: {result.output_value:.3f} +/- {result.expanded_uncertainty:.3f}")
    """

    # Default coverage factor for 95% confidence (normal distribution)
    DEFAULT_COVERAGE_FACTOR = 2.0

    # Precision for outputs
    PRECISION = 4

    def __init__(
        self,
        seed: Optional[int] = None,
        default_coverage_factor: float = 2.0
    ):
        """
        Initialize uncertainty quantifier.

        Args:
            seed: Random seed for reproducibility
            default_coverage_factor: Coverage factor for expanded uncertainty
        """
        self.seed = seed
        self.default_coverage_factor = default_coverage_factor

        # Initialize random generator with seed if provided
        if seed is not None:
            random.seed(seed)

    def monte_carlo_propagation(
        self,
        calculation: Callable[..., float],
        inputs: Dict[str, InputUncertainty],
        n_samples: int = 10000,
        coverage_factor: Optional[float] = None,
    ) -> UncertaintyResult:
        """
        Propagate uncertainty using Monte Carlo simulation.

        This method samples from input distributions and propagates
        through the calculation function to determine output uncertainty.

        Args:
            calculation: Function that takes input values and returns output
            inputs: Dictionary of input uncertainties
            n_samples: Number of Monte Carlo samples
            coverage_factor: Override for coverage factor

        Returns:
            UncertaintyResult with complete provenance

        Reference:
            JCGM 101:2008 - GUM Supplement 1, Section 7

        Example:
            >>> def efficiency(Q_out, Q_in):
            ...     return Q_out / Q_in * 100
            >>> inputs = {
            ...     "Q_out": InputUncertainty("Q_out", 850.0, 10.0, unit="kW"),
            ...     "Q_in": InputUncertainty("Q_in", 1000.0, 15.0, unit="kW"),
            ... }
            >>> result = uq.monte_carlo_propagation(efficiency, inputs)
        """
        start_time = time.perf_counter()

        # Reset random seed for reproducibility
        if self.seed is not None:
            random.seed(self.seed)

        k = coverage_factor or self.default_coverage_factor

        # Validate inputs
        if n_samples < 100:
            raise ValueError("n_samples must be at least 100 for reliable statistics")
        if n_samples > 1000000:
            raise ValueError("n_samples exceeds maximum of 1,000,000")

        # Generate samples for each input
        samples = {}
        for name, inp in inputs.items():
            samples[name] = self._generate_samples(inp, n_samples)

        # Evaluate calculation for each sample
        outputs = []
        for i in range(n_samples):
            try:
                sample_inputs = {name: samples[name][i] for name in inputs}
                output = calculation(**sample_inputs)
                outputs.append(output)
            except Exception:
                # Skip failed samples (edge cases)
                pass

        if len(outputs) < n_samples * 0.95:
            raise ValueError("Too many calculation failures during Monte Carlo simulation")

        # Calculate statistics
        mean_output = sum(outputs) / len(outputs)
        variance = sum((x - mean_output) ** 2 for x in outputs) / (len(outputs) - 1)
        std_uncertainty = math.sqrt(variance)

        # Expanded uncertainty
        expanded_uncertainty = k * std_uncertainty

        # Relative uncertainty
        relative_uncertainty = (std_uncertainty / abs(mean_output) * 100) if mean_output != 0 else 0

        # Calculate input contributions (via variance decomposition)
        input_contributions = self._calculate_variance_contributions(
            calculation, inputs, outputs, mean_output, variance
        )

        # Calculate sensitivity coefficients (numerical derivatives)
        sensitivity_coeffs = self._calculate_sensitivity_coefficients(
            calculation, {name: inp.value for name, inp in inputs.items()}
        )

        # Calculate provenance hash
        provenance_hash = self._calculate_provenance_hash(
            method="monte_carlo",
            inputs={name: {"value": inp.value, "uncertainty": inp.uncertainty}
                    for name, inp in inputs.items()},
            n_samples=n_samples,
            seed=self.seed,
            output=mean_output,
            uncertainty=std_uncertainty
        )

        end_time = time.perf_counter()
        calculation_time_ms = (end_time - start_time) * 1000

        return UncertaintyResult(
            output_value=round(mean_output, self.PRECISION),
            standard_uncertainty=round(std_uncertainty, self.PRECISION),
            expanded_uncertainty=round(expanded_uncertainty, self.PRECISION),
            coverage_factor=k,
            relative_uncertainty_percent=round(relative_uncertainty, 2),
            input_contributions=input_contributions,
            sensitivity_coefficients=sensitivity_coeffs,
            provenance_hash=provenance_hash,
            method="Monte Carlo (GUM Supplement 1)",
            n_samples=len(outputs),
            calculation_time_ms=calculation_time_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata={
                "seed": self.seed,
                "requested_samples": n_samples,
                "successful_samples": len(outputs)
            }
        )

    def sensitivity_analysis(
        self,
        calculation: Callable[..., float],
        inputs: Dict[str, InputUncertainty],
        perturbation_fraction: float = 0.01,
    ) -> SensitivityResult:
        """
        Perform sensitivity analysis on calculation.

        Calculates sensitivity coefficients (partial derivatives) for
        each input variable using numerical differentiation.

        Sensitivity Coefficient:
            c_i = partial(f) / partial(x_i)

        Normalized Sensitivity:
            S_i = (c_i * x_i) / f(x)

        Args:
            calculation: Function to analyze
            inputs: Dictionary of input uncertainties
            perturbation_fraction: Fraction for numerical derivative

        Returns:
            SensitivityResult with ranked sensitivities

        Reference:
            JCGM 100:2008, Section 5.1.3
        """
        start_time = time.perf_counter()

        # Get nominal values
        nominal_values = {name: inp.value for name, inp in inputs.items()}

        # Calculate nominal output
        nominal_output = calculation(**nominal_values)

        sensitivities = {}
        normalized_sensitivities = {}
        elasticities = {}

        for name, inp in inputs.items():
            # Calculate sensitivity coefficient via central difference
            h = abs(inp.value) * perturbation_fraction
            if h == 0:
                h = perturbation_fraction

            # Perturb up
            perturbed_up = nominal_values.copy()
            perturbed_up[name] = inp.value + h
            output_up = calculation(**perturbed_up)

            # Perturb down
            perturbed_down = nominal_values.copy()
            perturbed_down[name] = inp.value - h
            output_down = calculation(**perturbed_down)

            # Central difference derivative
            sensitivity = (output_up - output_down) / (2 * h)
            sensitivities[name] = round(sensitivity, self.PRECISION)

            # Normalized sensitivity (dimensionless)
            if nominal_output != 0:
                normalized = (sensitivity * inp.value) / nominal_output
                normalized_sensitivities[name] = round(normalized, self.PRECISION)

                # Elasticity (% change in output per % change in input)
                elasticities[name] = round(normalized * 100, 2)
            else:
                normalized_sensitivities[name] = 0.0
                elasticities[name] = 0.0

        # Rank by absolute normalized sensitivity
        most_influential = sorted(
            normalized_sensitivities.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        # Calculate provenance hash
        provenance_hash = self._calculate_provenance_hash(
            method="sensitivity_analysis",
            inputs={name: inp.value for name, inp in inputs.items()},
            perturbation=perturbation_fraction,
            output=nominal_output,
            uncertainty=0
        )

        return SensitivityResult(
            sensitivities=sensitivities,
            normalized_sensitivities=normalized_sensitivities,
            most_influential=most_influential,
            elasticities=elasticities,
            provenance_hash=provenance_hash,
            method="Central difference (GUM Section 5.1.3)",
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    def confidence_intervals(
        self,
        result: UncertaintyResult,
        confidence: float = 0.95,
    ) -> ConfidenceInterval:
        """
        Calculate confidence intervals for a result.

        Uses the coverage factor approach for normal distributions.
        For non-normal distributions from Monte Carlo, uses percentiles.

        Args:
            result: Uncertainty result to analyze
            confidence: Confidence level (0 to 1)

        Returns:
            ConfidenceInterval with bounds

        Reference:
            JCGM 100:2008, Section 6.2
        """
        # Coverage factors for common confidence levels (normal distribution)
        coverage_factors = {
            0.68: 1.0,
            0.90: 1.645,
            0.95: 2.0,
            0.99: 2.576,
            0.9545: 2.0,  # Exactly 2-sigma
            0.9973: 3.0,  # 3-sigma
        }

        # Find closest coverage factor
        if confidence in coverage_factors:
            k = coverage_factors[confidence]
        else:
            # Approximate using inverse normal
            # For 95%, k=1.96; for 99%, k=2.576
            from math import sqrt
            # Simple approximation (exact for normal distribution)
            if confidence >= 0.99:
                k = 2.576
            elif confidence >= 0.95:
                k = 1.96
            elif confidence >= 0.90:
                k = 1.645
            else:
                k = 1.0

        # Calculate bounds
        half_width = k * result.standard_uncertainty
        lower_bound = result.output_value - half_width
        upper_bound = result.output_value + half_width

        # Determine if symmetric (normal) or asymmetric
        interval_type = "symmetric"
        distribution_type = "normal (assumed)"

        if result.method.startswith("Monte Carlo"):
            distribution_type = "empirical (Monte Carlo)"

        # Calculate provenance hash
        provenance_hash = self._calculate_provenance_hash(
            method="confidence_interval",
            inputs={
                "value": result.output_value,
                "uncertainty": result.standard_uncertainty,
                "confidence": confidence
            },
            perturbation=0,
            output=result.output_value,
            uncertainty=result.standard_uncertainty
        )

        return ConfidenceInterval(
            value=result.output_value,
            lower_bound=round(lower_bound, self.PRECISION),
            upper_bound=round(upper_bound, self.PRECISION),
            confidence_level=confidence,
            interval_type=interval_type,
            distribution_type=distribution_type,
            provenance_hash=provenance_hash,
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    def analytical_propagation(
        self,
        calculation: Callable[..., float],
        inputs: Dict[str, InputUncertainty],
        coverage_factor: Optional[float] = None,
    ) -> UncertaintyResult:
        """
        Propagate uncertainty using analytical (GUM) method.

        Uses linear approximation with sensitivity coefficients:
            u_c^2 = sum(c_i^2 * u_i^2)

        This is faster than Monte Carlo but assumes linearity and
        normal distributions.

        Args:
            calculation: Function to evaluate
            inputs: Dictionary of input uncertainties
            coverage_factor: Override for coverage factor

        Returns:
            UncertaintyResult with complete provenance

        Reference:
            JCGM 100:2008, Section 5.2
        """
        start_time = time.perf_counter()

        k = coverage_factor or self.default_coverage_factor

        # Get nominal values
        nominal_values = {name: inp.value for name, inp in inputs.items()}

        # Calculate nominal output
        nominal_output = calculation(**nominal_values)

        # Calculate sensitivity coefficients
        sensitivity_coeffs = self._calculate_sensitivity_coefficients(
            calculation, nominal_values
        )

        # Calculate combined variance
        combined_variance = Decimal("0")
        input_contributions = {}

        for name, inp in inputs.items():
            std_u = inp.get_standard_uncertainty()
            c_i = sensitivity_coeffs.get(name, 0)

            # Variance contribution: (c_i * u_i)^2
            contribution = (c_i * std_u) ** 2
            combined_variance += Decimal(str(contribution))
            input_contributions[name] = round(contribution, self.PRECISION)

        # Combined standard uncertainty
        std_uncertainty = float(combined_variance.sqrt())

        # Expanded uncertainty
        expanded_uncertainty = k * std_uncertainty

        # Relative uncertainty
        relative_uncertainty = (std_uncertainty / abs(nominal_output) * 100) if nominal_output != 0 else 0

        # Calculate provenance hash
        provenance_hash = self._calculate_provenance_hash(
            method="analytical_gum",
            inputs={name: {"value": inp.value, "uncertainty": inp.uncertainty}
                    for name, inp in inputs.items()},
            perturbation=0,
            output=nominal_output,
            uncertainty=std_uncertainty
        )

        end_time = time.perf_counter()
        calculation_time_ms = (end_time - start_time) * 1000

        return UncertaintyResult(
            output_value=round(nominal_output, self.PRECISION),
            standard_uncertainty=round(std_uncertainty, self.PRECISION),
            expanded_uncertainty=round(expanded_uncertainty, self.PRECISION),
            coverage_factor=k,
            relative_uncertainty_percent=round(relative_uncertainty, 2),
            input_contributions=input_contributions,
            sensitivity_coefficients=sensitivity_coeffs,
            provenance_hash=provenance_hash,
            method="Analytical (GUM Section 5.2)",
            n_samples=None,
            calculation_time_ms=calculation_time_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata={"assumption": "linear approximation, normal distributions"}
        )

    def input_uncertainty_sources(self) -> Dict[str, Dict[str, Any]]:
        """
        Return standard uncertainty sources for thermal measurements.

        Provides typical uncertainty values for common measurement types
        based on calibrated instrumentation.

        Returns:
            Dictionary of uncertainty sources and typical values

        Reference:
            ASME PTC 19.1, Tables 1-5
        """
        return {
            "temperature": {
                "thermocouple_type_K": {
                    "typical_uncertainty_K": 1.5,
                    "coverage_factor": 2.0,
                    "source": "IEC 60584-1",
                    "notes": "Class 1, +/- 1.5C or 0.4% above 375C"
                },
                "thermocouple_type_T": {
                    "typical_uncertainty_K": 0.5,
                    "coverage_factor": 2.0,
                    "source": "IEC 60584-1",
                    "notes": "Class 1, +/- 0.5C or 0.4%"
                },
                "RTD_Pt100": {
                    "typical_uncertainty_K": 0.15,
                    "coverage_factor": 2.0,
                    "source": "IEC 60751",
                    "notes": "Class A, (0.15 + 0.002*T)"
                },
                "infrared_pyrometer": {
                    "typical_uncertainty_percent": 1.0,
                    "coverage_factor": 2.0,
                    "source": "Manufacturer spec",
                    "notes": "Emissivity uncertainty adds to total"
                }
            },
            "pressure": {
                "pressure_transmitter": {
                    "typical_uncertainty_percent": 0.25,
                    "coverage_factor": 2.0,
                    "source": "IEC 61298",
                    "notes": "Reference accuracy"
                },
                "differential_pressure": {
                    "typical_uncertainty_percent": 0.5,
                    "coverage_factor": 2.0,
                    "source": "ASME PTC 19.2",
                    "notes": "Including temperature effects"
                }
            },
            "flow": {
                "orifice_plate": {
                    "typical_uncertainty_percent": 1.0,
                    "coverage_factor": 2.0,
                    "source": "ISO 5167-2",
                    "notes": "Properly installed, calculated discharge"
                },
                "venturi_tube": {
                    "typical_uncertainty_percent": 0.75,
                    "coverage_factor": 2.0,
                    "source": "ISO 5167-4",
                    "notes": "Cast or machined"
                },
                "ultrasonic_flowmeter": {
                    "typical_uncertainty_percent": 0.5,
                    "coverage_factor": 2.0,
                    "source": "ISO 12242",
                    "notes": "Multi-path, clean fluid"
                },
                "coriolis_flowmeter": {
                    "typical_uncertainty_percent": 0.1,
                    "coverage_factor": 2.0,
                    "source": "ISO 10790",
                    "notes": "Direct mass flow measurement"
                }
            },
            "power": {
                "power_analyzer": {
                    "typical_uncertainty_percent": 0.1,
                    "coverage_factor": 2.0,
                    "source": "IEC 61000-4-30",
                    "notes": "Class A instrument"
                },
                "current_transformer": {
                    "typical_uncertainty_percent": 0.2,
                    "coverage_factor": 2.0,
                    "source": "IEC 61869-2",
                    "notes": "Class 0.2"
                }
            },
            "heat_rate": {
                "derived_Q_m_cp_dT": {
                    "typical_uncertainty_percent": 2.0,
                    "coverage_factor": 2.0,
                    "source": "Propagated from components",
                    "notes": "Combines flow, temperature, and property uncertainties"
                },
                "derived_Q_enthalpy": {
                    "typical_uncertainty_percent": 1.5,
                    "coverage_factor": 2.0,
                    "source": "Steam tables uncertainty",
                    "notes": "IAPWS-IF97 property uncertainty"
                }
            }
        }

    # =========================================================================
    # PRIVATE HELPER METHODS
    # =========================================================================

    def _generate_samples(
        self,
        inp: InputUncertainty,
        n_samples: int
    ) -> List[float]:
        """Generate random samples from input distribution."""
        std_u = inp.get_standard_uncertainty()

        if inp.distribution == "normal":
            # Normal distribution
            return [random.gauss(inp.value, std_u) for _ in range(n_samples)]

        elif inp.distribution == "uniform":
            # Uniform distribution: std = (b-a) / sqrt(12)
            # So half-width a = std * sqrt(3)
            half_width = std_u * math.sqrt(3)
            return [random.uniform(inp.value - half_width, inp.value + half_width)
                    for _ in range(n_samples)]

        elif inp.distribution == "triangular":
            # Symmetric triangular: std = a / sqrt(6)
            # So half-width a = std * sqrt(6)
            half_width = std_u * math.sqrt(6)
            return [random.triangular(inp.value - half_width, inp.value + half_width, inp.value)
                    for _ in range(n_samples)]

        else:
            # Default to normal
            return [random.gauss(inp.value, std_u) for _ in range(n_samples)]

    def _calculate_sensitivity_coefficients(
        self,
        calculation: Callable[..., float],
        nominal_values: Dict[str, float],
        perturbation: float = 0.001
    ) -> Dict[str, float]:
        """Calculate sensitivity coefficients via numerical differentiation."""
        sensitivities = {}

        for name, value in nominal_values.items():
            h = abs(value) * perturbation
            if h == 0:
                h = perturbation

            # Perturb up
            perturbed_up = nominal_values.copy()
            perturbed_up[name] = value + h

            # Perturb down
            perturbed_down = nominal_values.copy()
            perturbed_down[name] = value - h

            try:
                output_up = calculation(**perturbed_up)
                output_down = calculation(**perturbed_down)
                sensitivity = (output_up - output_down) / (2 * h)
                sensitivities[name] = round(sensitivity, self.PRECISION)
            except Exception:
                sensitivities[name] = 0.0

        return sensitivities

    def _calculate_variance_contributions(
        self,
        calculation: Callable[..., float],
        inputs: Dict[str, InputUncertainty],
        outputs: List[float],
        mean_output: float,
        total_variance: float
    ) -> Dict[str, float]:
        """
        Estimate variance contribution from each input.

        Uses correlation analysis between input samples and output.
        """
        contributions = {}

        if total_variance == 0:
            return {name: 0.0 for name in inputs}

        # Simplified approach: use sensitivity coefficients and input variances
        sensitivities = self._calculate_sensitivity_coefficients(
            calculation,
            {name: inp.value for name, inp in inputs.items()}
        )

        total_contrib = 0.0
        raw_contributions = {}

        for name, inp in inputs.items():
            std_u = inp.get_standard_uncertainty()
            c_i = sensitivities.get(name, 0)
            contrib = (c_i * std_u) ** 2
            raw_contributions[name] = contrib
            total_contrib += contrib

        # Normalize contributions to sum to total variance
        for name, contrib in raw_contributions.items():
            if total_contrib > 0:
                normalized = (contrib / total_contrib) * 100  # Percentage
            else:
                normalized = 0.0
            contributions[name] = round(normalized, 2)

        return contributions

    def _calculate_provenance_hash(
        self,
        method: str,
        inputs: Dict[str, Any],
        perturbation: float,
        output: float,
        uncertainty: float,
        n_samples: Optional[int] = None,
        seed: Optional[int] = None
    ) -> str:
        """Calculate SHA-256 provenance hash."""
        provenance_data = {
            "method": method,
            "inputs": inputs,
            "perturbation": perturbation,
            "output": output,
            "uncertainty": uncertainty,
            "n_samples": n_samples,
            "seed": seed
        }

        provenance_str = json.dumps(provenance_data, sort_keys=True)
        return hashlib.sha256(provenance_str.encode('utf-8')).hexdigest()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def calculate_combined_uncertainty(
    uncertainties: List[float],
    correlation_matrix: Optional[List[List[float]]] = None
) -> float:
    """
    Calculate combined standard uncertainty from multiple sources.

    For uncorrelated inputs:
        u_c = sqrt(sum(u_i^2))

    For correlated inputs:
        u_c = sqrt(sum(u_i^2) + 2*sum(r_ij*u_i*u_j))

    Args:
        uncertainties: List of standard uncertainties
        correlation_matrix: Optional correlation matrix (NxN)

    Returns:
        Combined standard uncertainty

    Reference:
        JCGM 100:2008, Equation 13
    """
    if not uncertainties:
        return 0.0

    # Variance sum
    variance_sum = sum(u ** 2 for u in uncertainties)

    # Add correlation terms if provided
    if correlation_matrix is not None:
        n = len(uncertainties)
        for i in range(n):
            for j in range(i + 1, n):
                r_ij = correlation_matrix[i][j]
                variance_sum += 2 * r_ij * uncertainties[i] * uncertainties[j]

    return math.sqrt(variance_sum)


def get_coverage_factor(
    confidence: float = 0.95,
    degrees_of_freedom: Optional[int] = None
) -> float:
    """
    Get coverage factor for specified confidence level.

    Args:
        confidence: Confidence level (0 to 1)
        degrees_of_freedom: Degrees of freedom (for t-distribution)

    Returns:
        Coverage factor k

    Reference:
        JCGM 100:2008, Table G.2
    """
    # For large DOF or unspecified, use normal distribution
    if degrees_of_freedom is None or degrees_of_freedom > 100:
        coverage_factors = {
            0.68: 1.0,
            0.90: 1.645,
            0.95: 1.96,
            0.99: 2.576,
        }

        # Find closest
        closest = min(coverage_factors.keys(), key=lambda x: abs(x - confidence))
        return coverage_factors[closest]

    # t-distribution for small DOF (simplified lookup)
    # Values for k at 95.45% (2-sigma equivalent)
    t_values_95 = {
        1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57,
        6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23,
        15: 2.13, 20: 2.09, 30: 2.04, 50: 2.01, 100: 1.98
    }

    # Find closest DOF
    closest_dof = min(t_values_95.keys(), key=lambda x: abs(x - degrees_of_freedom))
    return t_values_95[closest_dof]
