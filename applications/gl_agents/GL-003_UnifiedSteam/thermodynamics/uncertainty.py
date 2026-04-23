"""
Uncertainty Propagation Module - Measurement and Calculation Uncertainty

This module provides deterministic uncertainty propagation for steam property
calculations, supporting both analytical (Taylor series) and numerical
(Monte Carlo) methods.

Key Concepts:
- Measurement uncertainty: Inherent in sensor readings (P, T, flow)
- Property uncertainty: Propagated through thermodynamic calculations
- Combined uncertainty: Total uncertainty in derived quantities
- Coverage factor: For confidence intervals (k=2 for 95%)

Standards Reference:
- GUM (Guide to the Expression of Uncertainty in Measurement)
- ISO/IEC Guide 98-3:2008

Author: GL-CalculatorEngineer
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any, Tuple
from decimal import Decimal
import hashlib
import json
import math
import random


@dataclass
class UncertaintyInput:
    """
    Input parameter with uncertainty for propagation.
    """
    name: str                       # Parameter name
    value: float                    # Central value
    uncertainty: float              # Standard uncertainty (1-sigma)
    unit: str = ""                  # Unit of measurement
    distribution: str = "normal"    # Assumed distribution
    correlation: Optional[Dict[str, float]] = None  # Correlations with other inputs


@dataclass
class UncertaintyResult:
    """
    Result of uncertainty propagation (analytical method).
    """
    # Output value and uncertainty
    value: float                    # Calculated central value
    standard_uncertainty: float     # Standard uncertainty (1-sigma)
    expanded_uncertainty_95: float  # Expanded uncertainty (k=2, 95% CI)
    relative_uncertainty_percent: float  # Relative uncertainty as percentage

    # Sensitivity analysis
    sensitivity_coefficients: Dict[str, float]  # Partial derivatives
    uncertainty_contributions: Dict[str, float]  # Contribution from each input

    # Dominant uncertainty source
    dominant_source: str            # Input with largest contribution
    dominant_fraction: float        # Fraction of total variance

    # Correlation effect
    correlation_contribution: float  # Additional uncertainty from correlations

    # Provenance
    provenance_hash: str = ""


@dataclass
class MonteCarloResult:
    """
    Result of Monte Carlo uncertainty propagation.
    """
    # Output statistics
    mean: float                     # Sample mean
    standard_deviation: float       # Sample standard deviation
    median: float                   # Sample median
    percentile_2_5: float           # 2.5th percentile (lower 95% CI)
    percentile_97_5: float          # 97.5th percentile (upper 95% CI)
    min_value: float                # Minimum sampled value
    max_value: float                # Maximum sampled value

    # Distribution shape
    skewness: float                 # Skewness of output distribution
    kurtosis: float                 # Excess kurtosis

    # Sample information
    n_samples: int                  # Number of Monte Carlo samples
    n_valid: int                    # Number of valid (non-NaN) samples
    convergence_achieved: bool      # Whether results converged

    # Histogram data (for plotting)
    histogram_bins: List[float] = field(default_factory=list)
    histogram_counts: List[int] = field(default_factory=list)

    # Provenance
    provenance_hash: str = ""
    random_seed: Optional[int] = None


@dataclass
class PropertyUncertainties:
    """
    Uncertainties for all steam properties at a state point.
    """
    # State point
    pressure_kpa: float
    temperature_c: float

    # Input uncertainties
    pressure_uncertainty_kpa: float
    temperature_uncertainty_c: float
    flow_uncertainty_kg_s: float

    # Property uncertainties (all in standard uncertainty, 1-sigma)
    enthalpy_uncertainty_kj_kg: float
    entropy_uncertainty_kj_kgk: float
    specific_volume_uncertainty_m3_kg: float
    density_uncertainty_kg_m3: float

    # Energy flow uncertainty
    energy_rate_uncertainty_kw: Optional[float] = None

    # Quality uncertainty (for two-phase)
    quality_uncertainty: Optional[float] = None

    # Provenance
    provenance_hash: str = ""


# =============================================================================
# ANALYTICAL UNCERTAINTY PROPAGATION
# =============================================================================

def propagate_uncertainty(
    inputs: List[UncertaintyInput],
    function: Callable[..., float],
    sensitivities: Optional[Dict[str, float]] = None,
    delta_fraction: float = 0.01,
) -> UncertaintyResult:
    """
    Propagate uncertainties through a function using Taylor series expansion.

    DETERMINISTIC: Same inputs always produce same output.

    Uses the law of propagation of uncertainty (GUM method):
    u_y^2 = sum_i (df/dx_i)^2 * u_xi^2 + 2 * sum_i sum_j (df/dx_i)(df/dx_j) * u_xi * u_xj * r_ij

    Args:
        inputs: List of input parameters with uncertainties
        function: Function to propagate uncertainty through
        sensitivities: Pre-computed sensitivity coefficients (optional)
        delta_fraction: Fractional step for numerical differentiation

    Returns:
        UncertaintyResult with propagated uncertainty

    Raises:
        ValueError: If inputs are invalid
    """
    if not inputs:
        raise ValueError("At least one input required")

    # Calculate central value
    input_values = {inp.name: inp.value for inp in inputs}
    central_value = function(**input_values)

    # Calculate sensitivity coefficients if not provided
    if sensitivities is None:
        sensitivities = compute_sensitivity_coefficients(
            inputs, function, delta_fraction
        )

    # Calculate variance contributions
    variance = 0.0
    uncertainty_contributions = {}

    for inp in inputs:
        sens = sensitivities.get(inp.name, 0.0)
        var_contribution = (sens ** 2) * (inp.uncertainty ** 2)
        variance += var_contribution
        uncertainty_contributions[inp.name] = var_contribution

    # Add correlation terms
    correlation_contribution = 0.0

    for i, inp_i in enumerate(inputs):
        if inp_i.correlation is None:
            continue

        for j, inp_j in enumerate(inputs):
            if i >= j:  # Only upper triangle
                continue

            corr = inp_i.correlation.get(inp_j.name, 0.0)
            if corr != 0.0:
                sens_i = sensitivities.get(inp_i.name, 0.0)
                sens_j = sensitivities.get(inp_j.name, 0.0)
                cov_term = 2 * sens_i * sens_j * inp_i.uncertainty * inp_j.uncertainty * corr
                variance += cov_term
                correlation_contribution += cov_term

    # Standard uncertainty
    standard_uncertainty = math.sqrt(max(0, variance))

    # Expanded uncertainty (k=2 for approximately 95% coverage)
    expanded_uncertainty = 2.0 * standard_uncertainty

    # Relative uncertainty
    if abs(central_value) > 1e-10:
        relative_uncertainty = standard_uncertainty / abs(central_value) * 100
    else:
        relative_uncertainty = float('inf') if standard_uncertainty > 0 else 0.0

    # Find dominant source
    if uncertainty_contributions:
        dominant_source = max(
            uncertainty_contributions.keys(),
            key=lambda k: uncertainty_contributions[k]
        )
        total_variance = sum(uncertainty_contributions.values())
        dominant_fraction = (
            uncertainty_contributions[dominant_source] / total_variance
            if total_variance > 0 else 0.0
        )
    else:
        dominant_source = ""
        dominant_fraction = 0.0

    # Create provenance hash
    provenance_hash = _compute_provenance({
        "inputs": [{"name": i.name, "value": i.value, "unc": i.uncertainty} for i in inputs],
        "central_value": central_value,
        "standard_uncertainty": standard_uncertainty,
    })

    return UncertaintyResult(
        value=central_value,
        standard_uncertainty=standard_uncertainty,
        expanded_uncertainty_95=expanded_uncertainty,
        relative_uncertainty_percent=relative_uncertainty,
        sensitivity_coefficients=sensitivities,
        uncertainty_contributions=uncertainty_contributions,
        dominant_source=dominant_source,
        dominant_fraction=dominant_fraction,
        correlation_contribution=correlation_contribution,
        provenance_hash=provenance_hash,
    )


def compute_sensitivity_coefficients(
    inputs: List[UncertaintyInput],
    function: Callable[..., float],
    delta_fraction: float = 0.01,
) -> Dict[str, float]:
    """
    Compute sensitivity coefficients (partial derivatives) numerically.

    DETERMINISTIC: Same inputs always produce same output.

    Uses central difference: df/dx ~ (f(x+h) - f(x-h)) / (2h)

    Args:
        inputs: List of input parameters
        function: Function to differentiate
        delta_fraction: Fractional step size for differentiation

    Returns:
        Dictionary mapping input names to sensitivity coefficients
    """
    sensitivities = {}
    input_values = {inp.name: inp.value for inp in inputs}

    for inp in inputs:
        # Determine step size
        if abs(inp.value) > 1e-10:
            delta = abs(inp.value) * delta_fraction
        else:
            delta = 1e-6

        # Forward value
        input_values_plus = input_values.copy()
        input_values_plus[inp.name] = inp.value + delta

        # Backward value
        input_values_minus = input_values.copy()
        input_values_minus[inp.name] = inp.value - delta

        try:
            f_plus = function(**input_values_plus)
            f_minus = function(**input_values_minus)

            # Central difference
            sensitivity = (f_plus - f_minus) / (2 * delta)

        except (ValueError, ZeroDivisionError):
            # One-sided difference if central fails
            try:
                f_center = function(**input_values)
                f_plus = function(**input_values_plus)
                sensitivity = (f_plus - f_center) / delta
            except (ValueError, ZeroDivisionError):
                sensitivity = 0.0

        sensitivities[inp.name] = sensitivity

    return sensitivities


# =============================================================================
# MONTE CARLO UNCERTAINTY PROPAGATION
# =============================================================================

def monte_carlo_propagation(
    inputs: List[UncertaintyInput],
    function: Callable[..., float],
    n_samples: int = 10000,
    random_seed: Optional[int] = None,
    n_histogram_bins: int = 50,
) -> MonteCarloResult:
    """
    Propagate uncertainties using Monte Carlo simulation.

    DETERMINISTIC when random_seed is provided: Same seed produces same output.

    Samples inputs from their probability distributions and evaluates the
    function to build an output distribution.

    Args:
        inputs: List of input parameters with uncertainties
        function: Function to propagate uncertainty through
        n_samples: Number of Monte Carlo samples
        random_seed: Random seed for reproducibility
        n_histogram_bins: Number of bins for histogram

    Returns:
        MonteCarloResult with output distribution statistics
    """
    if random_seed is not None:
        random.seed(random_seed)

    # Generate correlated samples using Cholesky decomposition
    samples = _generate_correlated_samples(inputs, n_samples)

    # Evaluate function for each sample
    outputs = []
    n_valid = 0

    for i in range(n_samples):
        sample_dict = {inp.name: samples[inp.name][i] for inp in inputs}

        try:
            output = function(**sample_dict)
            if not math.isnan(output) and not math.isinf(output):
                outputs.append(output)
                n_valid += 1
        except (ValueError, ZeroDivisionError):
            # Skip invalid samples
            pass

    if n_valid < 10:
        raise ValueError(
            f"Monte Carlo simulation failed: only {n_valid} valid samples out of {n_samples}"
        )

    # Sort outputs for percentile calculations
    outputs_sorted = sorted(outputs)

    # Calculate statistics
    n = len(outputs)
    mean = sum(outputs) / n

    variance = sum((x - mean) ** 2 for x in outputs) / (n - 1)
    std_dev = math.sqrt(variance)

    # Percentiles
    median = outputs_sorted[n // 2]
    idx_2_5 = int(0.025 * n)
    idx_97_5 = int(0.975 * n)
    percentile_2_5 = outputs_sorted[idx_2_5]
    percentile_97_5 = outputs_sorted[idx_97_5]

    # Skewness and kurtosis
    if std_dev > 0:
        skewness = sum((x - mean) ** 3 for x in outputs) / (n * std_dev ** 3)
        kurtosis = sum((x - mean) ** 4 for x in outputs) / (n * std_dev ** 4) - 3
    else:
        skewness = 0.0
        kurtosis = 0.0

    # Check convergence (standard error of mean < 1% of std dev)
    se_mean = std_dev / math.sqrt(n)
    convergence_achieved = se_mean < 0.01 * std_dev if std_dev > 0 else True

    # Build histogram
    min_val = min(outputs)
    max_val = max(outputs)
    bin_width = (max_val - min_val) / n_histogram_bins if max_val > min_val else 1.0

    histogram_bins = [min_val + i * bin_width for i in range(n_histogram_bins + 1)]
    histogram_counts = [0] * n_histogram_bins

    for x in outputs:
        bin_idx = int((x - min_val) / bin_width)
        bin_idx = min(bin_idx, n_histogram_bins - 1)  # Handle edge case
        histogram_counts[bin_idx] += 1

    # Create provenance hash
    provenance_hash = _compute_provenance({
        "inputs": [{"name": i.name, "value": i.value, "unc": i.uncertainty} for i in inputs],
        "n_samples": n_samples,
        "random_seed": random_seed,
        "mean": mean,
        "std_dev": std_dev,
    })

    return MonteCarloResult(
        mean=mean,
        standard_deviation=std_dev,
        median=median,
        percentile_2_5=percentile_2_5,
        percentile_97_5=percentile_97_5,
        min_value=min_val,
        max_value=max_val,
        skewness=skewness,
        kurtosis=kurtosis,
        n_samples=n_samples,
        n_valid=n_valid,
        convergence_achieved=convergence_achieved,
        histogram_bins=histogram_bins,
        histogram_counts=histogram_counts,
        provenance_hash=provenance_hash,
        random_seed=random_seed,
    )


def _generate_correlated_samples(
    inputs: List[UncertaintyInput],
    n_samples: int,
) -> Dict[str, List[float]]:
    """
    Generate correlated random samples using Cholesky decomposition.

    DETERMINISTIC given the random state: Same random state produces same samples.

    For simplicity, this implementation assumes inputs are independent
    unless correlations are explicitly specified.

    Args:
        inputs: List of input parameters
        n_samples: Number of samples to generate

    Returns:
        Dictionary mapping input names to lists of samples
    """
    n_inputs = len(inputs)
    samples = {}

    # Build correlation matrix
    corr_matrix = [[0.0] * n_inputs for _ in range(n_inputs)]

    for i, inp_i in enumerate(inputs):
        corr_matrix[i][i] = 1.0  # Diagonal

        if inp_i.correlation is not None:
            for j, inp_j in enumerate(inputs):
                if i != j and inp_j.name in inp_i.correlation:
                    corr_matrix[i][j] = inp_i.correlation[inp_j.name]
                    corr_matrix[j][i] = inp_i.correlation[inp_j.name]

    # Cholesky decomposition (for correlated sampling)
    try:
        L = _cholesky_decomposition(corr_matrix)
    except ValueError:
        # If correlation matrix is not positive definite, assume independence
        L = [[1.0 if i == j else 0.0 for j in range(n_inputs)] for i in range(n_inputs)]

    # Generate uncorrelated standard normal samples
    uncorrelated = [
        [random.gauss(0, 1) for _ in range(n_samples)]
        for _ in range(n_inputs)
    ]

    # Apply Cholesky factor to introduce correlations
    for i, inp in enumerate(inputs):
        correlated_samples = []

        for s in range(n_samples):
            # y = L * z (z is uncorrelated standard normal)
            y = sum(L[i][j] * uncorrelated[j][s] for j in range(i + 1))

            # Transform to actual distribution
            if inp.distribution == "normal":
                sample = inp.value + y * inp.uncertainty
            elif inp.distribution == "uniform":
                # Uniform over [value - sqrt(3)*u, value + sqrt(3)*u]
                half_width = math.sqrt(3) * inp.uncertainty
                u = random.random()  # Uniform [0, 1]
                sample = inp.value + (2 * u - 1) * half_width
            elif inp.distribution == "lognormal":
                # Lognormal with mean=value and CV=uncertainty/value
                if inp.value > 0:
                    cv = inp.uncertainty / inp.value
                    sigma = math.sqrt(math.log(1 + cv ** 2))
                    mu = math.log(inp.value) - sigma ** 2 / 2
                    sample = math.exp(mu + sigma * y)
                else:
                    sample = inp.value + y * inp.uncertainty
            else:
                # Default to normal
                sample = inp.value + y * inp.uncertainty

            correlated_samples.append(sample)

        samples[inp.name] = correlated_samples

    return samples


def _cholesky_decomposition(matrix: List[List[float]]) -> List[List[float]]:
    """
    Compute Cholesky decomposition L such that A = L * L^T.

    DETERMINISTIC: Same matrix produces same decomposition.

    Args:
        matrix: Symmetric positive definite matrix

    Returns:
        Lower triangular Cholesky factor

    Raises:
        ValueError: If matrix is not positive definite
    """
    n = len(matrix)
    L = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))

            if i == j:
                val = matrix[i][i] - s
                if val <= 0:
                    raise ValueError("Matrix is not positive definite")
                L[i][j] = math.sqrt(val)
            else:
                if L[j][j] == 0:
                    raise ValueError("Zero diagonal in Cholesky decomposition")
                L[i][j] = (matrix[i][j] - s) / L[j][j]

    return L


# =============================================================================
# STEAM PROPERTY UNCERTAINTY
# =============================================================================

def compute_property_uncertainty(
    pressure_kpa: float,
    temperature_c: float,
    mass_flow_kg_s: float,
    pressure_uncertainty_kpa: float,
    temperature_uncertainty_c: float,
    flow_uncertainty_kg_s: float,
) -> PropertyUncertainties:
    """
    Compute uncertainties for all steam properties at a state point.

    DETERMINISTIC: Same inputs always produce same output.

    Uses analytical propagation through IAPWS-IF97 correlations.

    Args:
        pressure_kpa: Pressure in kPa
        temperature_c: Temperature in Celsius
        mass_flow_kg_s: Mass flow rate in kg/s
        pressure_uncertainty_kpa: Pressure uncertainty (1-sigma) in kPa
        temperature_uncertainty_c: Temperature uncertainty (1-sigma) in C
        flow_uncertainty_kg_s: Flow uncertainty (1-sigma) in kg/s

    Returns:
        PropertyUncertainties with all property uncertainties
    """
    from .steam_properties import compute_properties

    # Define input uncertainties
    inputs = [
        UncertaintyInput(
            name="pressure_kpa",
            value=pressure_kpa,
            uncertainty=pressure_uncertainty_kpa,
            unit="kPa",
        ),
        UncertaintyInput(
            name="temperature_c",
            value=temperature_c,
            uncertainty=temperature_uncertainty_c,
            unit="C",
        ),
    ]

    # ==========================================================================
    # Enthalpy uncertainty
    # ==========================================================================
    def enthalpy_func(pressure_kpa: float, temperature_c: float) -> float:
        props = compute_properties(pressure_kpa, temperature_c)
        return props.specific_enthalpy_kj_kg

    h_result = propagate_uncertainty(inputs, enthalpy_func)
    enthalpy_unc = h_result.standard_uncertainty

    # ==========================================================================
    # Entropy uncertainty
    # ==========================================================================
    def entropy_func(pressure_kpa: float, temperature_c: float) -> float:
        props = compute_properties(pressure_kpa, temperature_c)
        return props.specific_entropy_kj_kgk

    s_result = propagate_uncertainty(inputs, entropy_func)
    entropy_unc = s_result.standard_uncertainty

    # ==========================================================================
    # Specific volume uncertainty
    # ==========================================================================
    def volume_func(pressure_kpa: float, temperature_c: float) -> float:
        props = compute_properties(pressure_kpa, temperature_c)
        return props.specific_volume_m3_kg

    v_result = propagate_uncertainty(inputs, volume_func)
    volume_unc = v_result.standard_uncertainty

    # ==========================================================================
    # Density uncertainty
    # ==========================================================================
    def density_func(pressure_kpa: float, temperature_c: float) -> float:
        props = compute_properties(pressure_kpa, temperature_c)
        return props.density_kg_m3

    rho_result = propagate_uncertainty(inputs, density_func)
    density_unc = rho_result.standard_uncertainty

    # ==========================================================================
    # Energy rate uncertainty (includes flow uncertainty)
    # ==========================================================================
    inputs_with_flow = inputs + [
        UncertaintyInput(
            name="mass_flow_kg_s",
            value=mass_flow_kg_s,
            uncertainty=flow_uncertainty_kg_s,
            unit="kg/s",
        ),
    ]

    def energy_rate_func(
        pressure_kpa: float,
        temperature_c: float,
        mass_flow_kg_s: float,
    ) -> float:
        props = compute_properties(pressure_kpa, temperature_c)
        return mass_flow_kg_s * props.specific_enthalpy_kj_kg

    energy_result = propagate_uncertainty(inputs_with_flow, energy_rate_func)
    energy_rate_unc = energy_result.standard_uncertainty

    # ==========================================================================
    # Quality uncertainty (for two-phase only)
    # ==========================================================================
    try:
        props = compute_properties(pressure_kpa, temperature_c)
        if props.quality_x is not None:
            # For wet steam, quality uncertainty depends on enthalpy uncertainty
            # x = (h - hf) / hfg
            # u_x = u_h / hfg (approximately)
            from .steam_properties import get_saturation_properties
            sat = get_saturation_properties(pressure_kpa=pressure_kpa)
            quality_unc = enthalpy_unc / sat.hfg_kj_kg if sat.hfg_kj_kg > 0 else 0.0
        else:
            quality_unc = None
    except (ValueError, ImportError):
        quality_unc = None

    # Create provenance hash
    provenance_hash = _compute_provenance({
        "pressure_kpa": pressure_kpa,
        "temperature_c": temperature_c,
        "mass_flow_kg_s": mass_flow_kg_s,
        "pressure_unc": pressure_uncertainty_kpa,
        "temperature_unc": temperature_uncertainty_c,
        "flow_unc": flow_uncertainty_kg_s,
    })

    return PropertyUncertainties(
        pressure_kpa=pressure_kpa,
        temperature_c=temperature_c,
        pressure_uncertainty_kpa=pressure_uncertainty_kpa,
        temperature_uncertainty_c=temperature_uncertainty_c,
        flow_uncertainty_kg_s=flow_uncertainty_kg_s,
        enthalpy_uncertainty_kj_kg=enthalpy_unc,
        entropy_uncertainty_kj_kgk=entropy_unc,
        specific_volume_uncertainty_m3_kg=volume_unc,
        density_uncertainty_kg_m3=density_unc,
        energy_rate_uncertainty_kw=energy_rate_unc,
        quality_uncertainty=quality_unc,
        provenance_hash=provenance_hash,
    )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def combined_uncertainty(
    uncertainties: List[float],
    correlations: Optional[List[List[float]]] = None,
) -> float:
    """
    Calculate combined standard uncertainty from multiple sources.

    DETERMINISTIC: Same inputs always produce same output.

    For uncorrelated: u_c = sqrt(sum(u_i^2))
    For correlated: u_c = sqrt(u^T * C * u)

    Args:
        uncertainties: List of standard uncertainties
        correlations: Optional correlation matrix

    Returns:
        Combined standard uncertainty
    """
    n = len(uncertainties)

    if correlations is None:
        # Assume uncorrelated
        variance = sum(u ** 2 for u in uncertainties)
    else:
        # With correlations
        if len(correlations) != n or any(len(row) != n for row in correlations):
            raise ValueError("Correlation matrix dimensions must match uncertainties")

        variance = 0.0
        for i in range(n):
            for j in range(n):
                variance += uncertainties[i] * uncertainties[j] * correlations[i][j]

    return math.sqrt(max(0, variance))


def relative_to_absolute(relative_percent: float, value: float) -> float:
    """
    Convert relative uncertainty (%) to absolute uncertainty.

    DETERMINISTIC: Same inputs always produce same output.

    Args:
        relative_percent: Relative uncertainty in percent
        value: Central value

    Returns:
        Absolute uncertainty in same units as value
    """
    return abs(value) * relative_percent / 100


def absolute_to_relative(absolute: float, value: float) -> float:
    """
    Convert absolute uncertainty to relative uncertainty (%).

    DETERMINISTIC: Same inputs always produce same output.

    Args:
        absolute: Absolute uncertainty
        value: Central value

    Returns:
        Relative uncertainty in percent
    """
    if abs(value) < 1e-10:
        return float('inf') if absolute > 0 else 0.0
    return abs(absolute / value) * 100


def expanded_uncertainty(standard_uncertainty: float, coverage_factor: float = 2.0) -> float:
    """
    Calculate expanded uncertainty for a given coverage factor.

    DETERMINISTIC: Same inputs always produce same output.

    Common coverage factors:
    - k=1: 68% confidence (normal distribution)
    - k=2: 95% confidence (normal distribution)
    - k=3: 99.7% confidence (normal distribution)

    Args:
        standard_uncertainty: Standard (1-sigma) uncertainty
        coverage_factor: Coverage factor k

    Returns:
        Expanded uncertainty
    """
    return standard_uncertainty * coverage_factor


def _compute_provenance(data: Dict[str, Any]) -> str:
    """
    Compute SHA-256 provenance hash.

    DETERMINISTIC: Same inputs always produce same hash.
    """
    data_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(data_str.encode()).hexdigest()
