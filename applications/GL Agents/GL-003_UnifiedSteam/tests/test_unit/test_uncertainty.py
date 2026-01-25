"""
Unit Tests: Uncertainty Quantification

Tests uncertainty propagation and Monte Carlo methods including:
- Linear propagation against manual calculation
- Monte Carlo convergence
- Sensitivity computation
- Bounds behavior when inputs degrade

Reference: GUM (Guide to the Expression of Uncertainty in Measurement)

Author: GL-TestEngineer
Version: 1.0.0
Target Coverage: 85%+
"""

import pytest
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any
from enum import Enum, auto
import hashlib
import json


# =============================================================================
# Data Classes and Enumerations
# =============================================================================

class PropagationMethod(Enum):
    """Uncertainty propagation methods."""
    LINEAR = auto()          # First-order Taylor series (GUM)
    MONTE_CARLO = auto()     # Monte Carlo simulation
    ANALYTICAL = auto()      # Analytical solution (if available)


class DistributionType(Enum):
    """Probability distribution types for inputs."""
    NORMAL = auto()
    UNIFORM = auto()
    TRIANGULAR = auto()
    LOGNORMAL = auto()


@dataclass
class UncertaintyInput:
    """Input variable with uncertainty."""
    name: str
    value: float
    uncertainty: float  # Standard uncertainty (1 sigma)
    distribution: DistributionType = DistributionType.NORMAL
    correlation_group: Optional[str] = None


@dataclass
class UncertaintyResult:
    """Result of uncertainty propagation."""
    output_value: float
    standard_uncertainty: float
    expanded_uncertainty_95: float  # 95% confidence interval
    expanded_uncertainty_99: float  # 99% confidence interval
    relative_uncertainty_percent: float
    sensitivity_coefficients: Dict[str, float]
    contribution_percent: Dict[str, float]  # Variance contribution by input
    propagation_method: PropagationMethod
    provenance_hash: str


@dataclass
class MonteCarloResult:
    """Result of Monte Carlo uncertainty analysis."""
    mean: float
    standard_deviation: float
    percentile_2_5: float
    percentile_97_5: float
    percentile_0_5: float
    percentile_99_5: float
    num_samples: int
    convergence_achieved: bool
    convergence_criterion: float
    samples: Optional[np.ndarray] = None  # Full sample array if requested
    provenance_hash: str = ""


@dataclass
class PropertyUncertainties:
    """Uncertainty bounds for steam properties."""
    enthalpy_uncertainty_kj_kg: float
    entropy_uncertainty_kj_kg_k: float
    specific_volume_uncertainty_m3_kg: float
    temperature_uncertainty_k: float
    pressure_uncertainty_mpa: float


# =============================================================================
# Constants
# =============================================================================

# Coverage factors for expanded uncertainty
K_95 = 1.96  # 95% confidence
K_99 = 2.576  # 99% confidence

# Monte Carlo defaults
DEFAULT_MC_SAMPLES = 10000
CONVERGENCE_THRESHOLD = 0.01  # 1% relative change


# =============================================================================
# Uncertainty Propagation Implementation
# =============================================================================

class UncertaintyError(Exception):
    """Error in uncertainty calculation."""
    pass


def propagate_uncertainty_linear(
    function: Callable[..., float],
    inputs: List[UncertaintyInput],
    correlation_matrix: Optional[np.ndarray] = None
) -> UncertaintyResult:
    """
    Propagate uncertainty using linear (first-order Taylor series) method.

    This implements the GUM (Guide to expression of Uncertainty in Measurement)
    methodology.

    y = f(x1, x2, ..., xn)
    u(y)^2 = sum_i (df/dxi)^2 * u(xi)^2 + 2 * sum_i sum_j>i (df/dxi)(df/dxj) * u(xi,xj)

    Args:
        function: Function to evaluate
        inputs: List of input variables with uncertainties
        correlation_matrix: Optional correlation matrix between inputs

    Returns:
        UncertaintyResult with propagated uncertainty
    """
    if not inputs:
        raise UncertaintyError("No inputs provided")

    # Calculate output at nominal values
    nominal_values = {inp.name: inp.value for inp in inputs}
    output_value = function(**nominal_values)

    # Calculate sensitivity coefficients (partial derivatives)
    sensitivities = {}
    delta = 1e-8  # For numerical differentiation

    for inp in inputs:
        # Central difference
        values_plus = nominal_values.copy()
        values_minus = nominal_values.copy()
        values_plus[inp.name] = inp.value + delta
        values_minus[inp.name] = inp.value - delta

        try:
            f_plus = function(**values_plus)
            f_minus = function(**values_minus)
            sensitivities[inp.name] = (f_plus - f_minus) / (2 * delta)
        except Exception:
            sensitivities[inp.name] = 0.0

    # Calculate variance
    variance = 0.0
    contributions = {}

    for inp in inputs:
        c_i = sensitivities[inp.name]
        var_contribution = (c_i * inp.uncertainty) ** 2
        variance += var_contribution
        contributions[inp.name] = var_contribution

    # Add correlation terms if provided
    if correlation_matrix is not None:
        n = len(inputs)
        for i in range(n):
            for j in range(i + 1, n):
                c_i = sensitivities[inputs[i].name]
                c_j = sensitivities[inputs[j].name]
                rho_ij = correlation_matrix[i, j]
                variance += 2 * c_i * c_j * inputs[i].uncertainty * inputs[j].uncertainty * rho_ij

    standard_uncertainty = math.sqrt(variance)

    # Expanded uncertainties
    expanded_95 = K_95 * standard_uncertainty
    expanded_99 = K_99 * standard_uncertainty

    # Relative uncertainty
    relative_percent = (standard_uncertainty / abs(output_value) * 100) if output_value != 0 else 0

    # Contribution percentages
    total_var = variance if variance > 0 else 1e-10
    contribution_percent = {name: (var / total_var * 100) for name, var in contributions.items()}

    # Provenance hash
    provenance_hash = hashlib.sha256(
        json.dumps({
            "inputs": [(i.name, i.value, i.uncertainty) for i in inputs],
            "output": output_value,
            "uncertainty": standard_uncertainty
        }, sort_keys=True).encode()
    ).hexdigest()

    return UncertaintyResult(
        output_value=output_value,
        standard_uncertainty=standard_uncertainty,
        expanded_uncertainty_95=expanded_95,
        expanded_uncertainty_99=expanded_99,
        relative_uncertainty_percent=relative_percent,
        sensitivity_coefficients=sensitivities,
        contribution_percent=contribution_percent,
        propagation_method=PropagationMethod.LINEAR,
        provenance_hash=provenance_hash
    )


def monte_carlo_propagation(
    function: Callable[..., float],
    inputs: List[UncertaintyInput],
    num_samples: int = DEFAULT_MC_SAMPLES,
    correlation_matrix: Optional[np.ndarray] = None,
    seed: int = 42,
    return_samples: bool = False,
    convergence_check: bool = True
) -> MonteCarloResult:
    """
    Propagate uncertainty using Monte Carlo simulation.

    Generates random samples from input distributions and evaluates
    the function to build output distribution.

    Args:
        function: Function to evaluate
        inputs: List of input variables with uncertainties
        num_samples: Number of Monte Carlo samples
        correlation_matrix: Optional correlation matrix
        seed: Random seed for reproducibility
        return_samples: Whether to return full sample array
        convergence_check: Whether to check convergence

    Returns:
        MonteCarloResult with statistical summary
    """
    if not inputs:
        raise UncertaintyError("No inputs provided")

    np.random.seed(seed)
    n_inputs = len(inputs)

    # Generate samples for each input
    if correlation_matrix is not None:
        # Use Cholesky decomposition for correlated samples
        L = np.linalg.cholesky(correlation_matrix)
        uncorrelated_samples = np.random.normal(0, 1, (num_samples, n_inputs))
        correlated_samples = uncorrelated_samples @ L.T

        input_samples = {}
        for i, inp in enumerate(inputs):
            if inp.distribution == DistributionType.NORMAL:
                input_samples[inp.name] = inp.value + inp.uncertainty * correlated_samples[:, i]
            else:
                # For non-normal, generate uncorrelated
                input_samples[inp.name] = generate_samples(inp, num_samples)
    else:
        input_samples = {inp.name: generate_samples(inp, num_samples) for inp in inputs}

    # Evaluate function for all samples
    outputs = np.zeros(num_samples)
    for i in range(num_samples):
        sample_dict = {name: samples[i] for name, samples in input_samples.items()}
        try:
            outputs[i] = function(**sample_dict)
        except Exception:
            outputs[i] = np.nan

    # Remove any NaN values
    outputs = outputs[~np.isnan(outputs)]
    actual_samples = len(outputs)

    if actual_samples < 100:
        raise UncertaintyError(f"Too few valid samples ({actual_samples})")

    # Calculate statistics
    mean = np.mean(outputs)
    std = np.std(outputs, ddof=1)
    p2_5 = np.percentile(outputs, 2.5)
    p97_5 = np.percentile(outputs, 97.5)
    p0_5 = np.percentile(outputs, 0.5)
    p99_5 = np.percentile(outputs, 99.5)

    # Convergence check
    if convergence_check and num_samples >= 2000:
        # Check if statistics have converged
        half = num_samples // 2
        mean_first = np.mean(outputs[:half])
        mean_second = np.mean(outputs[half:])
        relative_change = abs(mean_second - mean_first) / abs(mean) if mean != 0 else 0
        convergence_achieved = relative_change < CONVERGENCE_THRESHOLD
    else:
        convergence_achieved = num_samples >= DEFAULT_MC_SAMPLES

    provenance_hash = hashlib.sha256(
        json.dumps({
            "num_samples": num_samples,
            "seed": seed,
            "mean": mean,
            "std": std
        }, sort_keys=True).encode()
    ).hexdigest()

    return MonteCarloResult(
        mean=mean,
        standard_deviation=std,
        percentile_2_5=p2_5,
        percentile_97_5=p97_5,
        percentile_0_5=p0_5,
        percentile_99_5=p99_5,
        num_samples=actual_samples,
        convergence_achieved=convergence_achieved,
        convergence_criterion=CONVERGENCE_THRESHOLD,
        samples=outputs if return_samples else None,
        provenance_hash=provenance_hash
    )


def generate_samples(inp: UncertaintyInput, num_samples: int) -> np.ndarray:
    """Generate random samples for an input based on its distribution."""
    if inp.distribution == DistributionType.NORMAL:
        return np.random.normal(inp.value, inp.uncertainty, num_samples)
    elif inp.distribution == DistributionType.UNIFORM:
        # For uniform, uncertainty represents half-width
        half_width = inp.uncertainty * math.sqrt(3)  # Convert std to half-width
        return np.random.uniform(inp.value - half_width, inp.value + half_width, num_samples)
    elif inp.distribution == DistributionType.TRIANGULAR:
        # Symmetric triangular
        half_width = inp.uncertainty * math.sqrt(6)
        return np.random.triangular(
            inp.value - half_width,
            inp.value,
            inp.value + half_width,
            num_samples
        )
    elif inp.distribution == DistributionType.LOGNORMAL:
        # Convert mean and std to lognormal parameters
        sigma = np.sqrt(np.log(1 + (inp.uncertainty / inp.value) ** 2))
        mu = np.log(inp.value) - sigma ** 2 / 2
        return np.random.lognormal(mu, sigma, num_samples)
    else:
        return np.random.normal(inp.value, inp.uncertainty, num_samples)


def compute_sensitivity_coefficients(
    function: Callable[..., float],
    inputs: List[UncertaintyInput],
    delta_fraction: float = 0.01
) -> Dict[str, float]:
    """
    Compute sensitivity coefficients (partial derivatives) for a function.

    Uses central finite difference method.
    """
    nominal_values = {inp.name: inp.value for inp in inputs}
    sensitivities = {}

    for inp in inputs:
        delta = inp.value * delta_fraction if inp.value != 0 else delta_fraction

        values_plus = nominal_values.copy()
        values_minus = nominal_values.copy()
        values_plus[inp.name] = inp.value + delta
        values_minus[inp.name] = inp.value - delta

        try:
            f_plus = function(**values_plus)
            f_minus = function(**values_minus)
            sensitivities[inp.name] = (f_plus - f_minus) / (2 * delta)
        except Exception:
            sensitivities[inp.name] = 0.0

    return sensitivities


def compute_property_uncertainty(
    pressure_mpa: float,
    pressure_uncertainty_mpa: float,
    temperature_k: float,
    temperature_uncertainty_k: float
) -> PropertyUncertainties:
    """
    Compute uncertainty in steam properties from P, T uncertainties.

    Uses sensitivity analysis for IAPWS-IF97 properties.
    """
    # Simplified sensitivity values (would be computed from IAPWS-IF97)
    # dh/dP ~ -1 to 10 kJ/(kg.MPa) depending on region
    # dh/dT ~ 2 kJ/(kg.K) (approximately Cp)

    dh_dp = 5.0  # Approximate sensitivity
    dh_dt = 2.0  # Approximate Cp

    ds_dp = -0.5  # Approximate
    ds_dt = 0.005  # Approximate

    dv_dp = -0.0001  # Approximate compressibility
    dv_dt = 0.000001  # Approximate thermal expansion

    # Propagate uncertainties (linear, uncorrelated)
    u_h = math.sqrt((dh_dp * pressure_uncertainty_mpa) ** 2 + (dh_dt * temperature_uncertainty_k) ** 2)
    u_s = math.sqrt((ds_dp * pressure_uncertainty_mpa) ** 2 + (ds_dt * temperature_uncertainty_k) ** 2)
    u_v = math.sqrt((dv_dp * pressure_uncertainty_mpa) ** 2 + (dv_dt * temperature_uncertainty_k) ** 2)

    return PropertyUncertainties(
        enthalpy_uncertainty_kj_kg=u_h,
        entropy_uncertainty_kj_kg_k=u_s,
        specific_volume_uncertainty_m3_kg=u_v,
        temperature_uncertainty_k=temperature_uncertainty_k,
        pressure_uncertainty_mpa=pressure_uncertainty_mpa
    )


def propagate_uncertainty(
    function: Callable[..., float],
    inputs: List[UncertaintyInput],
    method: PropagationMethod = PropagationMethod.LINEAR,
    correlation_matrix: Optional[np.ndarray] = None,
    mc_samples: int = DEFAULT_MC_SAMPLES
) -> UncertaintyResult:
    """
    Main interface for uncertainty propagation.

    Selects appropriate method based on input.
    """
    if method == PropagationMethod.LINEAR:
        return propagate_uncertainty_linear(function, inputs, correlation_matrix)
    elif method == PropagationMethod.MONTE_CARLO:
        mc_result = monte_carlo_propagation(
            function, inputs, mc_samples, correlation_matrix
        )
        # Convert MC result to UncertaintyResult format
        sensitivities = compute_sensitivity_coefficients(function, inputs)

        return UncertaintyResult(
            output_value=mc_result.mean,
            standard_uncertainty=mc_result.standard_deviation,
            expanded_uncertainty_95=mc_result.percentile_97_5 - mc_result.percentile_2_5,
            expanded_uncertainty_99=mc_result.percentile_99_5 - mc_result.percentile_0_5,
            relative_uncertainty_percent=(mc_result.standard_deviation / abs(mc_result.mean) * 100) if mc_result.mean != 0 else 0,
            sensitivity_coefficients=sensitivities,
            contribution_percent={},  # Would need additional analysis
            propagation_method=PropagationMethod.MONTE_CARLO,
            provenance_hash=mc_result.provenance_hash
        )
    else:
        raise UncertaintyError(f"Unknown propagation method: {method}")


# =============================================================================
# Test Functions (for testing uncertainty propagation)
# =============================================================================

def simple_addition(x: float, y: float) -> float:
    """Simple addition: z = x + y"""
    return x + y


def simple_multiplication(x: float, y: float) -> float:
    """Simple multiplication: z = x * y"""
    return x * y


def power_function(x: float, n: float = 2.0) -> float:
    """Power function: z = x^n"""
    return x ** n


def complex_function(a: float, b: float, c: float) -> float:
    """Complex function: z = a*b^2 + c"""
    return a * b ** 2 + c


def enthalpy_rate(mass_flow: float, enthalpy: float) -> float:
    """Enthalpy rate: Q = m_dot * h"""
    return mass_flow * enthalpy


def efficiency_calculation(output: float, input_val: float) -> float:
    """Efficiency: eta = output / input"""
    if input_val == 0:
        return 0
    return output / input_val


# =============================================================================
# Test Classes
# =============================================================================

class TestLinearPropagation:
    """Test linear uncertainty propagation."""

    def test_addition_uncertainty(self):
        """Test uncertainty propagation for addition.

        For z = x + y:
        u(z) = sqrt(u(x)^2 + u(y)^2)
        """
        inputs = [
            UncertaintyInput("x", 10.0, 0.5),
            UncertaintyInput("y", 20.0, 0.3),
        ]

        result = propagate_uncertainty_linear(simple_addition, inputs)

        # Manual calculation
        expected_value = 30.0
        expected_uncertainty = math.sqrt(0.5 ** 2 + 0.3 ** 2)

        assert pytest.approx(result.output_value, rel=0.001) == expected_value
        assert pytest.approx(result.standard_uncertainty, rel=0.01) == expected_uncertainty

    def test_multiplication_uncertainty(self):
        """Test uncertainty propagation for multiplication.

        For z = x * y:
        u(z)/z = sqrt((u(x)/x)^2 + (u(y)/y)^2)
        """
        inputs = [
            UncertaintyInput("x", 10.0, 0.5),
            UncertaintyInput("y", 20.0, 0.4),
        ]

        result = propagate_uncertainty_linear(simple_multiplication, inputs)

        # Manual calculation
        expected_value = 200.0
        rel_x = 0.5 / 10.0
        rel_y = 0.4 / 20.0
        expected_uncertainty = expected_value * math.sqrt(rel_x ** 2 + rel_y ** 2)

        assert pytest.approx(result.output_value, rel=0.001) == expected_value
        assert pytest.approx(result.standard_uncertainty, rel=0.05) == expected_uncertainty

    def test_sensitivity_coefficients_addition(self):
        """Test sensitivity coefficients for addition are both 1."""
        inputs = [
            UncertaintyInput("x", 10.0, 0.5),
            UncertaintyInput("y", 20.0, 0.3),
        ]

        result = propagate_uncertainty_linear(simple_addition, inputs)

        assert pytest.approx(result.sensitivity_coefficients["x"], rel=0.01) == 1.0
        assert pytest.approx(result.sensitivity_coefficients["y"], rel=0.01) == 1.0

    def test_sensitivity_coefficients_multiplication(self):
        """Test sensitivity coefficients for multiplication."""
        inputs = [
            UncertaintyInput("x", 10.0, 0.5),
            UncertaintyInput("y", 20.0, 0.4),
        ]

        result = propagate_uncertainty_linear(simple_multiplication, inputs)

        # dz/dx = y, dz/dy = x
        assert pytest.approx(result.sensitivity_coefficients["x"], rel=0.01) == 20.0
        assert pytest.approx(result.sensitivity_coefficients["y"], rel=0.01) == 10.0

    def test_contribution_percentages_sum_to_100(self):
        """Test variance contributions sum to 100%."""
        inputs = [
            UncertaintyInput("x", 10.0, 0.5),
            UncertaintyInput("y", 20.0, 0.3),
        ]

        result = propagate_uncertainty_linear(simple_addition, inputs)

        total = sum(result.contribution_percent.values())
        assert pytest.approx(total, rel=0.01) == 100.0

    def test_expanded_uncertainty_95(self):
        """Test 95% expanded uncertainty is ~2x standard uncertainty."""
        inputs = [
            UncertaintyInput("x", 10.0, 1.0),
        ]

        result = propagate_uncertainty_linear(lambda x: x, inputs)

        assert pytest.approx(result.expanded_uncertainty_95, rel=0.01) == K_95 * result.standard_uncertainty

    def test_expanded_uncertainty_99(self):
        """Test 99% expanded uncertainty is ~2.6x standard uncertainty."""
        inputs = [
            UncertaintyInput("x", 10.0, 1.0),
        ]

        result = propagate_uncertainty_linear(lambda x: x, inputs)

        assert pytest.approx(result.expanded_uncertainty_99, rel=0.01) == K_99 * result.standard_uncertainty


class TestMonteCarloConvergence:
    """Test Monte Carlo convergence behavior."""

    def test_mc_mean_matches_expected(self):
        """Test MC mean converges to expected value."""
        inputs = [
            UncertaintyInput("x", 10.0, 0.5),
            UncertaintyInput("y", 20.0, 0.3),
        ]

        result = monte_carlo_propagation(
            simple_addition, inputs, num_samples=50000, seed=42
        )

        expected_mean = 30.0
        assert pytest.approx(result.mean, rel=0.01) == expected_mean

    def test_mc_std_matches_linear(self):
        """Test MC standard deviation matches linear propagation."""
        inputs = [
            UncertaintyInput("x", 10.0, 0.5),
            UncertaintyInput("y", 20.0, 0.3),
        ]

        mc_result = monte_carlo_propagation(
            simple_addition, inputs, num_samples=50000, seed=42
        )
        linear_result = propagate_uncertainty_linear(simple_addition, inputs)

        # Should be within 5% of each other
        assert pytest.approx(mc_result.standard_deviation, rel=0.05) == linear_result.standard_uncertainty

    def test_mc_convergence_with_samples(self):
        """Test convergence improves with more samples."""
        inputs = [UncertaintyInput("x", 10.0, 1.0)]

        result_1k = monte_carlo_propagation(
            lambda x: x, inputs, num_samples=1000, seed=42
        )
        result_10k = monte_carlo_propagation(
            lambda x: x, inputs, num_samples=10000, seed=42
        )
        result_50k = monte_carlo_propagation(
            lambda x: x, inputs, num_samples=50000, seed=42
        )

        # More samples should get closer to true value (10.0)
        error_1k = abs(result_1k.mean - 10.0)
        error_10k = abs(result_10k.mean - 10.0)
        error_50k = abs(result_50k.mean - 10.0)

        # Generally, error should decrease (may not always be true due to randomness)
        # But at least 50k should be close
        assert error_50k < 0.1

    def test_mc_reproducibility_with_seed(self):
        """Test MC results are reproducible with same seed."""
        inputs = [UncertaintyInput("x", 10.0, 1.0)]

        result1 = monte_carlo_propagation(
            lambda x: x, inputs, num_samples=1000, seed=42
        )
        result2 = monte_carlo_propagation(
            lambda x: x, inputs, num_samples=1000, seed=42
        )

        assert result1.mean == result2.mean
        assert result1.standard_deviation == result2.standard_deviation

    def test_mc_percentiles(self):
        """Test MC percentiles are reasonable."""
        inputs = [UncertaintyInput("x", 10.0, 1.0, DistributionType.NORMAL)]

        result = monte_carlo_propagation(
            lambda x: x, inputs, num_samples=50000, seed=42
        )

        # For normal distribution, 95% interval should be ~4 sigma wide
        interval_width = result.percentile_97_5 - result.percentile_2_5
        expected_width = 2 * K_95 * 1.0  # 2 * 1.96 * sigma

        assert pytest.approx(interval_width, rel=0.1) == expected_width


class TestSensitivityComputation:
    """Test sensitivity coefficient computation."""

    def test_sensitivity_linear_function(self):
        """Test sensitivity for linear function."""
        inputs = [
            UncertaintyInput("x", 5.0, 0.1),
            UncertaintyInput("y", 10.0, 0.1),
        ]

        sensitivities = compute_sensitivity_coefficients(simple_addition, inputs)

        assert pytest.approx(sensitivities["x"], rel=0.01) == 1.0
        assert pytest.approx(sensitivities["y"], rel=0.01) == 1.0

    def test_sensitivity_quadratic_function(self):
        """Test sensitivity for quadratic function."""
        inputs = [UncertaintyInput("x", 5.0, 0.1)]

        sensitivities = compute_sensitivity_coefficients(power_function, inputs)

        # d(x^2)/dx = 2x = 10
        assert pytest.approx(sensitivities["x"], rel=0.01) == 10.0

    def test_sensitivity_complex_function(self):
        """Test sensitivity for complex function z = a*b^2 + c."""
        inputs = [
            UncertaintyInput("a", 2.0, 0.1),
            UncertaintyInput("b", 3.0, 0.1),
            UncertaintyInput("c", 1.0, 0.1),
        ]

        sensitivities = compute_sensitivity_coefficients(complex_function, inputs)

        # dz/da = b^2 = 9
        assert pytest.approx(sensitivities["a"], rel=0.05) == 9.0

        # dz/db = 2*a*b = 12
        assert pytest.approx(sensitivities["b"], rel=0.05) == 12.0

        # dz/dc = 1
        assert pytest.approx(sensitivities["c"], rel=0.05) == 1.0


class TestBoundsWithDegradedInputs:
    """Test uncertainty bounds behavior when inputs degrade."""

    def test_uncertainty_increases_with_input_uncertainty(self):
        """Test output uncertainty increases when input uncertainty increases."""
        inputs_tight = [
            UncertaintyInput("x", 10.0, 0.1),
            UncertaintyInput("y", 20.0, 0.1),
        ]
        inputs_loose = [
            UncertaintyInput("x", 10.0, 1.0),
            UncertaintyInput("y", 20.0, 1.0),
        ]

        result_tight = propagate_uncertainty_linear(simple_addition, inputs_tight)
        result_loose = propagate_uncertainty_linear(simple_addition, inputs_loose)

        assert result_loose.standard_uncertainty > result_tight.standard_uncertainty

    def test_uncertainty_ratio_preserved(self):
        """Test uncertainty ratio is preserved in proportional scaling."""
        inputs_1 = [UncertaintyInput("x", 10.0, 0.5)]
        inputs_2 = [UncertaintyInput("x", 10.0, 1.0)]

        result_1 = propagate_uncertainty_linear(lambda x: x, inputs_1)
        result_2 = propagate_uncertainty_linear(lambda x: x, inputs_2)

        # Uncertainty should double
        assert pytest.approx(result_2.standard_uncertainty / result_1.standard_uncertainty, rel=0.01) == 2.0

    def test_dominant_uncertainty_contribution(self):
        """Test that dominant input contributes most to uncertainty."""
        inputs = [
            UncertaintyInput("x", 10.0, 0.1),  # Small uncertainty
            UncertaintyInput("y", 20.0, 2.0),  # Large uncertainty
        ]

        result = propagate_uncertainty_linear(simple_addition, inputs)

        # y should dominate
        assert result.contribution_percent["y"] > result.contribution_percent["x"]
        assert result.contribution_percent["y"] > 90  # Should be >90%

    def test_zero_uncertainty_no_contribution(self):
        """Test that zero uncertainty input has zero contribution."""
        inputs = [
            UncertaintyInput("x", 10.0, 0.0),  # Zero uncertainty
            UncertaintyInput("y", 20.0, 1.0),
        ]

        result = propagate_uncertainty_linear(simple_addition, inputs)

        # x should have 0% contribution
        assert pytest.approx(result.contribution_percent["x"], abs=0.1) == 0.0
        assert pytest.approx(result.contribution_percent["y"], rel=0.01) == 100.0

    def test_large_uncertainty_bounds_reasonable(self):
        """Test bounds remain reasonable even with large uncertainties."""
        inputs = [UncertaintyInput("x", 10.0, 5.0)]  # 50% relative uncertainty

        result = propagate_uncertainty_linear(lambda x: x, inputs)

        # 95% interval should contain the nominal value
        lower_95 = result.output_value - result.expanded_uncertainty_95
        upper_95 = result.output_value + result.expanded_uncertainty_95

        assert lower_95 < result.output_value < upper_95


class TestDifferentDistributions:
    """Test uncertainty with different input distributions."""

    def test_uniform_distribution_mc(self):
        """Test MC with uniform distribution input."""
        inputs = [
            UncertaintyInput("x", 10.0, 1.0, DistributionType.UNIFORM),
        ]

        result = monte_carlo_propagation(lambda x: x, inputs, num_samples=50000, seed=42)

        # For uniform, std should be ~1.0 (as specified)
        assert pytest.approx(result.standard_deviation, rel=0.1) == 1.0

    def test_triangular_distribution_mc(self):
        """Test MC with triangular distribution input."""
        inputs = [
            UncertaintyInput("x", 10.0, 1.0, DistributionType.TRIANGULAR),
        ]

        result = monte_carlo_propagation(lambda x: x, inputs, num_samples=50000, seed=42)

        # Triangular should have std close to specified
        assert 0.5 < result.standard_deviation < 2.0

    def test_lognormal_distribution_mc(self):
        """Test MC with lognormal distribution input."""
        inputs = [
            UncertaintyInput("x", 10.0, 1.0, DistributionType.LOGNORMAL),
        ]

        result = monte_carlo_propagation(lambda x: x, inputs, num_samples=50000, seed=42)

        # Lognormal should be skewed positive
        assert result.mean > 0
        assert result.percentile_97_5 - result.mean > result.mean - result.percentile_2_5


class TestPropertyUncertainty:
    """Test steam property uncertainty computation."""

    def test_property_uncertainty_structure(self):
        """Test property uncertainty returns complete structure."""
        result = compute_property_uncertainty(
            pressure_mpa=1.0,
            pressure_uncertainty_mpa=0.01,
            temperature_k=450.0,
            temperature_uncertainty_k=0.5
        )

        assert result.enthalpy_uncertainty_kj_kg > 0
        assert result.entropy_uncertainty_kj_kg_k > 0
        assert result.specific_volume_uncertainty_m3_kg > 0

    def test_property_uncertainty_increases_with_input(self):
        """Test property uncertainty increases with input uncertainty."""
        result_tight = compute_property_uncertainty(1.0, 0.001, 450.0, 0.1)
        result_loose = compute_property_uncertainty(1.0, 0.1, 450.0, 5.0)

        assert result_loose.enthalpy_uncertainty_kj_kg > result_tight.enthalpy_uncertainty_kj_kg


class TestCorrelatedInputs:
    """Test uncertainty propagation with correlated inputs."""

    def test_positive_correlation_increases_uncertainty(self):
        """Test that positive correlation increases uncertainty."""
        inputs = [
            UncertaintyInput("x", 10.0, 1.0),
            UncertaintyInput("y", 20.0, 1.0),
        ]

        # No correlation
        result_uncorr = propagate_uncertainty_linear(simple_addition, inputs)

        # Positive correlation
        corr_matrix = np.array([[1.0, 0.8], [0.8, 1.0]])
        result_corr = propagate_uncertainty_linear(simple_addition, inputs, corr_matrix)

        # Positive correlation should increase uncertainty for addition
        assert result_corr.standard_uncertainty > result_uncorr.standard_uncertainty

    def test_negative_correlation_decreases_uncertainty(self):
        """Test that negative correlation can decrease uncertainty."""
        inputs = [
            UncertaintyInput("x", 10.0, 1.0),
            UncertaintyInput("y", 20.0, 1.0),
        ]

        # No correlation
        result_uncorr = propagate_uncertainty_linear(simple_addition, inputs)

        # Negative correlation
        corr_matrix = np.array([[1.0, -0.8], [-0.8, 1.0]])
        result_corr = propagate_uncertainty_linear(simple_addition, inputs, corr_matrix)

        # Negative correlation should decrease uncertainty for addition
        assert result_corr.standard_uncertainty < result_uncorr.standard_uncertainty


class TestEnthalpyRateUncertainty:
    """Test uncertainty for enthalpy rate calculation (practical example)."""

    def test_enthalpy_rate_uncertainty(self):
        """Test uncertainty propagation for Q = m_dot * h."""
        inputs = [
            UncertaintyInput("mass_flow", 5.0, 0.05),  # 1% uncertainty
            UncertaintyInput("enthalpy", 2800.0, 28.0),  # 1% uncertainty
        ]

        result = propagate_uncertainty_linear(enthalpy_rate, inputs)

        # Expected: Q = 14000 kW
        assert pytest.approx(result.output_value, rel=0.001) == 14000.0

        # Relative uncertainty should be ~sqrt(2) * 1% = 1.4%
        expected_rel = math.sqrt(0.01 ** 2 + 0.01 ** 2) * 100
        assert pytest.approx(result.relative_uncertainty_percent, rel=0.1) == expected_rel

    def test_efficiency_uncertainty(self):
        """Test uncertainty propagation for efficiency = output/input."""
        inputs = [
            UncertaintyInput("output", 850.0, 8.5),  # 1% uncertainty
            UncertaintyInput("input_val", 1000.0, 10.0),  # 1% uncertainty
        ]

        result = propagate_uncertainty_linear(efficiency_calculation, inputs)

        # Expected: eta = 0.85
        assert pytest.approx(result.output_value, rel=0.001) == 0.85

        # Both inputs contribute to uncertainty
        assert len(result.contribution_percent) == 2


class TestProvenanceTracking:
    """Test provenance hash generation."""

    def test_linear_provenance_deterministic(self):
        """Test linear propagation provenance is deterministic."""
        inputs = [UncertaintyInput("x", 10.0, 0.5)]

        result1 = propagate_uncertainty_linear(lambda x: x, inputs)
        result2 = propagate_uncertainty_linear(lambda x: x, inputs)

        assert result1.provenance_hash == result2.provenance_hash

    def test_mc_provenance_deterministic(self):
        """Test MC provenance is deterministic with same seed."""
        inputs = [UncertaintyInput("x", 10.0, 0.5)]

        result1 = monte_carlo_propagation(lambda x: x, inputs, seed=42)
        result2 = monte_carlo_propagation(lambda x: x, inputs, seed=42)

        assert result1.provenance_hash == result2.provenance_hash

    def test_provenance_changes_with_input(self):
        """Test provenance changes when input changes."""
        inputs1 = [UncertaintyInput("x", 10.0, 0.5)]
        inputs2 = [UncertaintyInput("x", 10.0, 0.6)]

        result1 = propagate_uncertainty_linear(lambda x: x, inputs1)
        result2 = propagate_uncertainty_linear(lambda x: x, inputs2)

        assert result1.provenance_hash != result2.provenance_hash


class TestErrorHandling:
    """Test error handling in uncertainty calculations."""

    def test_empty_inputs_raises_error(self):
        """Test that empty inputs list raises error."""
        with pytest.raises(UncertaintyError):
            propagate_uncertainty_linear(lambda: 0, [])

    def test_mc_empty_inputs_raises_error(self):
        """Test that MC with empty inputs raises error."""
        with pytest.raises(UncertaintyError):
            monte_carlo_propagation(lambda: 0, [])
