"""
Unit Tests for GL-003 UnifiedSteam - Uncertainty Propagation Module

Tests for:
- UncertaintyPropagator (linear, Jacobian, Monte Carlo)
- Correlation matrix handling
- Monte Carlo convergence
- Sobol sensitivity analysis
- Quality gates integration

Target Coverage: 90%+
"""

import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest
import numpy as np

# Import application modules
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from uncertainty.propagation import (
    CorrelationMatrix,
    UncertaintyPropagator,
    combine_uncertainties,
    DEFAULT_MC_SAMPLES,
    MC_CONVERGENCE_THRESHOLD,
)
from uncertainty.uncertainty_models import (
    Distribution,
    DistributionType,
    MonteCarloResult,
    PropagatedUncertainty,
    UncertainValue,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def propagator() -> UncertaintyPropagator:
    """Create an UncertaintyPropagator instance."""
    return UncertaintyPropagator(default_seed=42)


@pytest.fixture
def propagator_custom() -> UncertaintyPropagator:
    """Create a propagator with custom settings."""
    return UncertaintyPropagator(
        finite_diff_step=1e-8,
        default_mc_samples=5000,
        default_seed=123,
    )


@pytest.fixture
def sample_uncertain_inputs() -> Dict[str, UncertainValue]:
    """Create sample uncertain inputs."""
    return {
        "x": UncertainValue(
            mean=100.0,
            std=2.0,
            lower_95=96.08,
            upper_95=103.92,
            distribution_type=DistributionType.NORMAL,
            timestamp=datetime.now(timezone.utc),
        ),
        "y": UncertainValue(
            mean=50.0,
            std=1.5,
            lower_95=47.06,
            upper_95=52.94,
            distribution_type=DistributionType.NORMAL,
            timestamp=datetime.now(timezone.utc),
        ),
    }


@pytest.fixture
def sample_distributions() -> Dict[str, Distribution]:
    """Create sample distributions for Monte Carlo."""
    return {
        "x": Distribution(
            distribution_type=DistributionType.NORMAL,
            parameters={"mean": 100.0, "std": 2.0},
        ),
        "y": Distribution(
            distribution_type=DistributionType.NORMAL,
            parameters={"mean": 50.0, "std": 1.5},
        ),
    }


@pytest.fixture
def identity_correlation() -> CorrelationMatrix:
    """Create an identity correlation matrix."""
    return CorrelationMatrix.identity(["x", "y"])


@pytest.fixture
def positive_correlation() -> CorrelationMatrix:
    """Create a positive correlation matrix."""
    return CorrelationMatrix(
        variable_names=["x", "y"],
        matrix=np.array([[1.0, 0.8], [0.8, 1.0]]),
    )


@pytest.fixture
def negative_correlation() -> CorrelationMatrix:
    """Create a negative correlation matrix."""
    return CorrelationMatrix(
        variable_names=["x", "y"],
        matrix=np.array([[1.0, -0.6], [-0.6, 1.0]]),
    )


# =============================================================================
# Test CorrelationMatrix
# =============================================================================

class TestCorrelationMatrix:
    """Tests for CorrelationMatrix class."""

    def test_identity_matrix_creation(self):
        """Test creating identity correlation matrix."""
        corr = CorrelationMatrix.identity(["a", "b", "c"])

        assert len(corr.variable_names) == 3
        assert corr.matrix.shape == (3, 3)
        np.testing.assert_array_equal(corr.matrix, np.eye(3))

    def test_get_correlation(self, positive_correlation):
        """Test getting correlation between variables."""
        rho = positive_correlation.get_correlation("x", "y")
        assert abs(rho - 0.8) < 1e-10

        rho_same = positive_correlation.get_correlation("x", "x")
        assert rho_same == 1.0

    def test_symmetric_validation(self):
        """Test that non-symmetric matrix raises error."""
        with pytest.raises(ValueError, match="symmetric"):
            CorrelationMatrix(
                variable_names=["x", "y"],
                matrix=np.array([[1.0, 0.5], [0.3, 1.0]]),  # Not symmetric
            )

    def test_diagonal_validation(self):
        """Test that diagonal != 1 raises error."""
        with pytest.raises(ValueError, match="diagonal"):
            CorrelationMatrix(
                variable_names=["x", "y"],
                matrix=np.array([[0.9, 0.5], [0.5, 0.9]]),  # Diagonal != 1
            )

    def test_bounds_validation(self):
        """Test that correlation > 1 raises error."""
        with pytest.raises(ValueError):
            CorrelationMatrix(
                variable_names=["x", "y"],
                matrix=np.array([[1.0, 1.5], [1.5, 1.0]]),  # > 1
            )

    def test_shape_validation(self):
        """Test that mismatched shape raises error."""
        with pytest.raises(ValueError, match="shape"):
            CorrelationMatrix(
                variable_names=["x", "y", "z"],  # 3 variables
                matrix=np.array([[1.0, 0.5], [0.5, 1.0]]),  # 2x2 matrix
            )


# =============================================================================
# Test Linear Propagation
# =============================================================================

class TestLinearPropagation:
    """Tests for linear uncertainty propagation."""

    def test_simple_sum(self, propagator, sample_uncertain_inputs):
        """Test propagation for simple sum z = x + y."""
        coefficients = {"x": 1.0, "y": 1.0}

        result = propagator.propagate_linear(
            inputs=sample_uncertain_inputs,
            coefficients=coefficients,
            output_name="z",
        )

        assert isinstance(result, PropagatedUncertainty)
        assert result.output_name == "z"

        # z = 100 + 50 = 150
        assert abs(result.value - 150.0) < 0.01

        # Var(z) = Var(x) + Var(y) = 4 + 2.25 = 6.25, std = 2.5
        assert abs(result.uncertainty - 2.5) < 0.01

    def test_weighted_sum(self, propagator, sample_uncertain_inputs):
        """Test propagation for weighted sum z = 2*x + 3*y."""
        coefficients = {"x": 2.0, "y": 3.0}

        result = propagator.propagate_linear(
            inputs=sample_uncertain_inputs,
            coefficients=coefficients,
        )

        # z = 2*100 + 3*50 = 350
        assert abs(result.value - 350.0) < 0.01

        # Var(z) = 4*Var(x) + 9*Var(y) = 4*4 + 9*2.25 = 16 + 20.25 = 36.25
        expected_std = math.sqrt(36.25)
        assert abs(result.uncertainty - expected_std) < 0.01

    def test_difference(self, propagator, sample_uncertain_inputs):
        """Test propagation for difference z = x - y."""
        coefficients = {"x": 1.0, "y": -1.0}

        result = propagator.propagate_linear(
            inputs=sample_uncertain_inputs,
            coefficients=coefficients,
        )

        # z = 100 - 50 = 50
        assert abs(result.value - 50.0) < 0.01

        # Var(z) = Var(x) + Var(y) = 6.25 (same as sum for uncorrelated)
        assert abs(result.uncertainty - 2.5) < 0.01

    def test_with_positive_correlation(
        self, propagator, sample_uncertain_inputs, positive_correlation
    ):
        """Test propagation with positive correlation."""
        coefficients = {"x": 1.0, "y": -1.0}

        # Uncorrelated result for comparison
        uncorr_result = propagator.propagate_linear(
            inputs=sample_uncertain_inputs,
            coefficients=coefficients,
        )

        # Correlated result
        corr_result = propagator.propagate_linear(
            inputs=sample_uncertain_inputs,
            coefficients=coefficients,
            correlation=positive_correlation,
        )

        # Positive correlation should REDUCE variance for x - y
        # Because x and y move together, their difference is more stable
        assert corr_result.uncertainty < uncorr_result.uncertainty

    def test_with_negative_correlation(
        self, propagator, sample_uncertain_inputs, negative_correlation
    ):
        """Test propagation with negative correlation."""
        coefficients = {"x": 1.0, "y": 1.0}

        # Uncorrelated result for comparison
        uncorr_result = propagator.propagate_linear(
            inputs=sample_uncertain_inputs,
            coefficients=coefficients,
        )

        # Correlated result
        corr_result = propagator.propagate_linear(
            inputs=sample_uncertain_inputs,
            coefficients=coefficients,
            correlation=negative_correlation,
        )

        # Negative correlation should REDUCE variance for x + y
        # Because when x goes up, y tends to go down
        assert corr_result.uncertainty < uncorr_result.uncertainty

    def test_confidence_interval(self, propagator, sample_uncertain_inputs):
        """Test 95% confidence interval calculation."""
        coefficients = {"x": 1.0, "y": 1.0}

        result = propagator.propagate_linear(
            inputs=sample_uncertain_inputs,
            coefficients=coefficients,
        )

        # CI should be mean +/- 1.96*std
        expected_lower = result.value - 1.96 * result.uncertainty
        expected_upper = result.value + 1.96 * result.uncertainty

        assert abs(result.confidence_interval_95[0] - expected_lower) < 0.01
        assert abs(result.confidence_interval_95[1] - expected_upper) < 0.01

    def test_dominant_contributor(self, propagator, sample_uncertain_inputs):
        """Test identification of dominant uncertainty contributor."""
        # x has larger std (2.0 vs 1.5)
        coefficients = {"x": 1.0, "y": 1.0}

        result = propagator.propagate_linear(
            inputs=sample_uncertain_inputs,
            coefficients=coefficients,
        )

        # x contributes more (4.0 vs 2.25)
        assert result.dominant_contributor == "x"

    def test_missing_coefficient_raises(self, propagator, sample_uncertain_inputs):
        """Test that missing coefficient raises error."""
        coefficients = {"x": 1.0, "z": 2.0}  # z not in inputs

        with pytest.raises(ValueError, match="z"):
            propagator.propagate_linear(
                inputs=sample_uncertain_inputs,
                coefficients=coefficients,
            )

    def test_propagation_method_recorded(self, propagator, sample_uncertain_inputs):
        """Test that propagation method is recorded."""
        result = propagator.propagate_linear(
            inputs=sample_uncertain_inputs,
            coefficients={"x": 1.0},
        )

        assert result.propagation_method == "linear"

    def test_computation_time_recorded(self, propagator, sample_uncertain_inputs):
        """Test that computation time is recorded."""
        result = propagator.propagate_linear(
            inputs=sample_uncertain_inputs,
            coefficients={"x": 1.0},
        )

        assert result.computation_time_ms >= 0


# =============================================================================
# Test Nonlinear (Jacobian) Propagation
# =============================================================================

class TestNonlinearPropagation:
    """Tests for nonlinear Jacobian-based propagation."""

    def test_multiplication(self, propagator, sample_uncertain_inputs):
        """Test propagation for z = x * y."""
        def f(vals):
            return vals["x"] * vals["y"]

        def jacobian(vals):
            return {"x": vals["y"], "y": vals["x"]}

        result = propagator.propagate_nonlinear(
            inputs=sample_uncertain_inputs,
            function=f,
            jacobian=jacobian,
            output_name="z",
        )

        # z = 100 * 50 = 5000
        assert abs(result.value - 5000.0) < 0.1

        # Using first-order Taylor:
        # dz/dx = y = 50, dz/dy = x = 100
        # Var(z) = (50)^2 * 4 + (100)^2 * 2.25 = 10000 + 22500 = 32500
        expected_std = math.sqrt(32500)
        assert abs(result.uncertainty - expected_std) < 5.0  # Some tolerance

    def test_division(self, propagator, sample_uncertain_inputs):
        """Test propagation for z = x / y."""
        def f(vals):
            return vals["x"] / vals["y"]

        def jacobian(vals):
            return {"x": 1/vals["y"], "y": -vals["x"]/(vals["y"]**2)}

        result = propagator.propagate_nonlinear(
            inputs=sample_uncertain_inputs,
            function=f,
            jacobian=jacobian,
        )

        # z = 100 / 50 = 2.0
        assert abs(result.value - 2.0) < 0.01

    def test_power_function(self, propagator, sample_uncertain_inputs):
        """Test propagation for z = x^2."""
        def f(vals):
            return vals["x"] ** 2

        def jacobian(vals):
            return {"x": 2 * vals["x"]}

        result = propagator.propagate_nonlinear(
            inputs=sample_uncertain_inputs,
            function=f,
            jacobian=jacobian,
        )

        # z = 100^2 = 10000
        assert abs(result.value - 10000.0) < 0.1

        # dz/dx = 200
        # Var(z) = (200)^2 * 4 = 160000, std = 400
        assert abs(result.uncertainty - 400.0) < 5.0

    def test_numerical_jacobian(self, propagator, sample_uncertain_inputs):
        """Test propagation with numerical Jacobian."""
        def f(vals):
            return vals["x"] * vals["y"]

        # No Jacobian provided - should use numerical differentiation
        result = propagator.propagate_nonlinear(
            inputs=sample_uncertain_inputs,
            function=f,
            jacobian=None,  # Will compute numerically
        )

        # z = 100 * 50 = 5000
        assert abs(result.value - 5000.0) < 0.1

        # Should still get reasonable uncertainty
        assert result.uncertainty > 0

    def test_with_correlation(
        self, propagator, sample_uncertain_inputs, positive_correlation
    ):
        """Test nonlinear propagation with correlation."""
        def f(vals):
            return vals["x"] * vals["y"]

        def jacobian(vals):
            return {"x": vals["y"], "y": vals["x"]}

        result = propagator.propagate_nonlinear(
            inputs=sample_uncertain_inputs,
            function=f,
            jacobian=jacobian,
            correlation=positive_correlation,
        )

        # Should have different uncertainty than uncorrelated case
        assert result.uncertainty > 0

    def test_propagation_method_recorded(self, propagator, sample_uncertain_inputs):
        """Test that propagation method is recorded as jacobian."""
        def f(vals):
            return vals["x"]

        result = propagator.propagate_nonlinear(
            inputs=sample_uncertain_inputs,
            function=f,
        )

        assert result.propagation_method == "jacobian"


# =============================================================================
# Test Monte Carlo Propagation
# =============================================================================

class TestMonteCarloProgation:
    """Tests for Monte Carlo uncertainty propagation."""

    def test_simple_sum_monte_carlo(self, propagator, sample_distributions):
        """Test Monte Carlo for simple sum."""
        def f(vals):
            return vals["x"] + vals["y"]

        result = propagator.propagate_monte_carlo(
            inputs=sample_distributions,
            function=f,
            n_samples=10000,
        )

        assert isinstance(result, MonteCarloResult)

        # Mean should be close to 150
        assert abs(result.mean - 150.0) < 1.0

        # Std should be close to sqrt(4 + 2.25) = 2.5
        assert abs(result.std - 2.5) < 0.2

    def test_multiplication_monte_carlo(self, propagator, sample_distributions):
        """Test Monte Carlo for multiplication."""
        def f(vals):
            return vals["x"] * vals["y"]

        result = propagator.propagate_monte_carlo(
            inputs=sample_distributions,
            function=f,
            n_samples=10000,
        )

        # Mean should be close to 100 * 50 = 5000
        assert abs(result.mean - 5000.0) < 50.0

    def test_nonlinear_monte_carlo(self, propagator, sample_distributions):
        """Test Monte Carlo for complex nonlinear function."""
        def f(vals):
            return math.exp(vals["x"] / 100) + vals["y"] ** 0.5

        result = propagator.propagate_monte_carlo(
            inputs=sample_distributions,
            function=f,
            n_samples=10000,
        )

        # Should produce valid result
        assert math.isfinite(result.mean)
        assert result.std > 0

    def test_percentiles(self, propagator, sample_distributions):
        """Test percentile calculations."""
        def f(vals):
            return vals["x"]

        result = propagator.propagate_monte_carlo(
            inputs=sample_distributions,
            function=f,
            n_samples=10000,
        )

        # 50th percentile should be close to mean for normal distribution
        assert abs(result.percentiles[50.0] - result.mean) < 0.5

        # 97.5 > 50 > 2.5
        assert result.percentiles[97.5] > result.percentiles[50.0]
        assert result.percentiles[50.0] > result.percentiles[2.5]

    def test_convergence_check(self, propagator, sample_distributions):
        """Test convergence checking."""
        def f(vals):
            return vals["x"]

        result = propagator.propagate_monte_carlo(
            inputs=sample_distributions,
            function=f,
            n_samples=10000,
            convergence_check=True,
        )

        # Should achieve convergence with enough samples
        assert result.convergence_achieved

    def test_reproducibility_with_seed(self, propagator, sample_distributions):
        """Test reproducibility with fixed seed."""
        def f(vals):
            return vals["x"] + vals["y"]

        result1 = propagator.propagate_monte_carlo(
            inputs=sample_distributions,
            function=f,
            n_samples=1000,
            seed=42,
        )

        result2 = propagator.propagate_monte_carlo(
            inputs=sample_distributions,
            function=f,
            n_samples=1000,
            seed=42,
        )

        # Same seed should give same results
        assert result1.mean == result2.mean
        assert result1.std == result2.std

    def test_different_seeds_different_results(self, propagator, sample_distributions):
        """Test that different seeds give different results."""
        def f(vals):
            return vals["x"] + vals["y"]

        result1 = propagator.propagate_monte_carlo(
            inputs=sample_distributions,
            function=f,
            n_samples=1000,
            seed=42,
        )

        result2 = propagator.propagate_monte_carlo(
            inputs=sample_distributions,
            function=f,
            n_samples=1000,
            seed=123,
        )

        # Different seeds should give different results (with high probability)
        assert result1.mean != result2.mean

    def test_seed_recorded(self, propagator, sample_distributions):
        """Test that seed is recorded in result."""
        def f(vals):
            return vals["x"]

        result = propagator.propagate_monte_carlo(
            inputs=sample_distributions,
            function=f,
            seed=42,
        )

        assert result.seed == 42

    def test_return_samples(self, propagator, sample_distributions):
        """Test returning raw samples."""
        def f(vals):
            return vals["x"]

        result = propagator.propagate_monte_carlo(
            inputs=sample_distributions,
            function=f,
            n_samples=1000,
            return_samples=True,
        )

        assert result.samples is not None
        assert len(result.samples) == 1000

    def test_samples_not_returned_by_default(self, propagator, sample_distributions):
        """Test that samples are not returned by default."""
        def f(vals):
            return vals["x"]

        result = propagator.propagate_monte_carlo(
            inputs=sample_distributions,
            function=f,
        )

        assert result.samples is None

    def test_correlated_monte_carlo(
        self, propagator, sample_distributions, positive_correlation
    ):
        """Test Monte Carlo with correlated inputs."""
        def f(vals):
            return vals["x"] - vals["y"]

        # Uncorrelated
        uncorr_result = propagator.propagate_monte_carlo(
            inputs=sample_distributions,
            function=f,
            n_samples=10000,
        )

        # Correlated
        corr_result = propagator.propagate_monte_carlo(
            inputs=sample_distributions,
            function=f,
            n_samples=10000,
            correlation=positive_correlation,
        )

        # Positive correlation should reduce variance of x - y
        assert corr_result.std < uncorr_result.std * 1.1  # Allow some tolerance

    def test_uniform_distribution(self, propagator):
        """Test Monte Carlo with uniform distribution."""
        distributions = {
            "x": Distribution(
                distribution_type=DistributionType.UNIFORM,
                parameters={"low": 0.0, "high": 100.0},
            )
        }

        def f(vals):
            return vals["x"]

        result = propagator.propagate_monte_carlo(
            inputs=distributions,
            function=f,
            n_samples=10000,
        )

        # Mean should be ~50 for uniform(0, 100)
        assert abs(result.mean - 50.0) < 2.0

        # Std should be ~sqrt(100^2/12) = 28.9
        assert abs(result.std - 28.9) < 2.0

    def test_triangular_distribution(self, propagator):
        """Test Monte Carlo with triangular distribution."""
        distributions = {
            "x": Distribution(
                distribution_type=DistributionType.TRIANGULAR,
                parameters={"low": 0.0, "mode": 50.0, "high": 100.0},
            )
        }

        def f(vals):
            return vals["x"]

        result = propagator.propagate_monte_carlo(
            inputs=distributions,
            function=f,
            n_samples=10000,
        )

        # Mean should be (0 + 50 + 100)/3 = 50
        assert abs(result.mean - 50.0) < 2.0

    def test_lognormal_distribution(self, propagator):
        """Test Monte Carlo with lognormal distribution."""
        distributions = {
            "x": Distribution(
                distribution_type=DistributionType.LOGNORMAL,
                parameters={"mu": 0.0, "sigma": 0.5},
            )
        }

        def f(vals):
            return vals["x"]

        result = propagator.propagate_monte_carlo(
            inputs=distributions,
            function=f,
            n_samples=10000,
        )

        # Should produce valid positive values
        assert result.mean > 0
        assert result.percentiles[2.5] > 0


# =============================================================================
# Test Sensitivity Analysis
# =============================================================================

class TestSensitivityAnalysis:
    """Tests for sensitivity coefficient computation."""

    def test_compute_sensitivity_linear(self, propagator):
        """Test sensitivity computation for linear function."""
        def f(vals):
            return 2 * vals["x"] + 3 * vals["y"]

        inputs = {"x": 10.0, "y": 5.0}

        sens_x = propagator.compute_sensitivity(f, inputs, "x")
        sens_y = propagator.compute_sensitivity(f, inputs, "y")

        assert abs(sens_x - 2.0) < 0.01
        assert abs(sens_y - 3.0) < 0.01

    def test_compute_sensitivity_nonlinear(self, propagator):
        """Test sensitivity computation for nonlinear function."""
        def f(vals):
            return vals["x"] ** 2

        inputs = {"x": 10.0}

        # d/dx(x^2) = 2x = 20 at x=10
        sens = propagator.compute_sensitivity(f, inputs, "x")
        assert abs(sens - 20.0) < 0.1

    def test_compute_sensitivity_missing_parameter(self, propagator):
        """Test that missing parameter raises error."""
        def f(vals):
            return vals["x"]

        with pytest.raises(ValueError):
            propagator.compute_sensitivity(f, {"x": 10.0}, "y")

    def test_numerical_jacobian(self, propagator):
        """Test numerical Jacobian computation."""
        def f(vals):
            return vals["x"] * vals["y"]

        inputs = {"x": 10.0, "y": 5.0}

        jacobian = propagator._numerical_jacobian(f, inputs)

        # df/dx = y = 5, df/dy = x = 10
        assert abs(jacobian["x"] - 5.0) < 0.01
        assert abs(jacobian["y"] - 10.0) < 0.01


# =============================================================================
# Test Sobol Sensitivity Analysis
# =============================================================================

class TestSobolSensitivity:
    """Tests for Sobol sensitivity indices."""

    def test_sobol_additive_model(self, propagator):
        """Test Sobol indices for additive model."""
        distributions = {
            "x": Distribution(
                distribution_type=DistributionType.UNIFORM,
                parameters={"low": 0.0, "high": 1.0},
            ),
            "y": Distribution(
                distribution_type=DistributionType.UNIFORM,
                parameters={"low": 0.0, "high": 1.0},
            ),
        }

        def f(vals):
            return 3 * vals["x"] + vals["y"]

        result = propagator.sobol_sensitivity(
            inputs=distributions,
            function=f,
            n_samples=5000,
            seed=42,
        )

        # x contributes more (coefficient 3 vs 1)
        # S1 for x should be > S1 for y
        assert "x" in result and "y" in result
        assert "S1" in result["x"] and "ST" in result["x"]

    def test_sobol_multiplicative_model(self, propagator):
        """Test Sobol indices for multiplicative model."""
        distributions = {
            "x": Distribution(
                distribution_type=DistributionType.UNIFORM,
                parameters={"low": 0.5, "high": 1.5},
            ),
            "y": Distribution(
                distribution_type=DistributionType.UNIFORM,
                parameters={"low": 0.5, "high": 1.5},
            ),
        }

        def f(vals):
            return vals["x"] * vals["y"]

        result = propagator.sobol_sensitivity(
            inputs=distributions,
            function=f,
            n_samples=5000,
            seed=42,
        )

        # For multiplicative model, both have similar first-order effects
        # but total effects should be higher due to interaction
        assert result["x"]["ST"] >= result["x"]["S1"] * 0.9
        assert result["y"]["ST"] >= result["y"]["S1"] * 0.9

    def test_sobol_indices_bounded(self, propagator):
        """Test that Sobol indices are in [0, 1]."""
        distributions = {
            "x": Distribution(
                distribution_type=DistributionType.UNIFORM,
                parameters={"low": 0.0, "high": 1.0},
            ),
        }

        def f(vals):
            return vals["x"] ** 2

        result = propagator.sobol_sensitivity(
            inputs=distributions,
            function=f,
            n_samples=2000,
        )

        assert 0 <= result["x"]["S1"] <= 1
        assert 0 <= result["x"]["ST"] <= 1


# =============================================================================
# Test Combine Uncertainties
# =============================================================================

class TestCombineUncertainties:
    """Tests for combining multiple uncertainties."""

    def test_quadrature_combination(self):
        """Test quadrature (RSS) combination."""
        uncertainties = [
            UncertainValue(mean=100.0, std=3.0, lower_95=94.12, upper_95=105.88,
                           distribution_type=DistributionType.NORMAL, timestamp=datetime.now(timezone.utc)),
            UncertainValue(mean=50.0, std=4.0, lower_95=42.16, upper_95=57.84,
                           distribution_type=DistributionType.NORMAL, timestamp=datetime.now(timezone.utc)),
        ]

        result = combine_uncertainties(uncertainties, method="quadrature")

        # Mean = 100 + 50 = 150
        assert abs(result.mean - 150.0) < 0.01

        # Std = sqrt(9 + 16) = 5
        assert abs(result.std - 5.0) < 0.01

    def test_linear_combination(self):
        """Test linear (conservative) combination."""
        uncertainties = [
            UncertainValue(mean=100.0, std=3.0, lower_95=94.12, upper_95=105.88,
                           distribution_type=DistributionType.NORMAL, timestamp=datetime.now(timezone.utc)),
            UncertainValue(mean=50.0, std=4.0, lower_95=42.16, upper_95=57.84,
                           distribution_type=DistributionType.NORMAL, timestamp=datetime.now(timezone.utc)),
        ]

        result = combine_uncertainties(uncertainties, method="linear")

        # Mean = 150
        assert abs(result.mean - 150.0) < 0.01

        # Std = 3 + 4 = 7 (conservative)
        assert abs(result.std - 7.0) < 0.01

    def test_max_combination(self):
        """Test maximum uncertainty combination."""
        uncertainties = [
            UncertainValue(mean=100.0, std=3.0, lower_95=94.12, upper_95=105.88,
                           distribution_type=DistributionType.NORMAL, timestamp=datetime.now(timezone.utc)),
            UncertainValue(mean=50.0, std=4.0, lower_95=42.16, upper_95=57.84,
                           distribution_type=DistributionType.NORMAL, timestamp=datetime.now(timezone.utc)),
        ]

        result = combine_uncertainties(uncertainties, method="max")

        # Std = max(3, 4) = 4
        assert abs(result.std - 4.0) < 0.01

    def test_single_uncertainty(self):
        """Test combining single uncertainty returns same."""
        uncertainty = UncertainValue(
            mean=100.0, std=3.0, lower_95=94.12, upper_95=105.88,
            distribution_type=DistributionType.NORMAL, timestamp=datetime.now(timezone.utc)
        )

        result = combine_uncertainties([uncertainty])

        assert result.mean == uncertainty.mean
        assert result.std == uncertainty.std

    def test_empty_list_raises(self):
        """Test that empty list raises error."""
        with pytest.raises(ValueError):
            combine_uncertainties([])

    def test_unknown_method_raises(self):
        """Test that unknown method raises error."""
        uncertainty = UncertainValue(
            mean=100.0, std=3.0, lower_95=94.12, upper_95=105.88,
            distribution_type=DistributionType.NORMAL, timestamp=datetime.now(timezone.utc)
        )

        with pytest.raises(ValueError):
            combine_uncertainties([uncertainty], method="unknown")


# =============================================================================
# Test Monte Carlo Convergence
# =============================================================================

class TestMonteCarloConvergence:
    """Tests for Monte Carlo convergence behavior."""

    def test_convergence_with_more_samples(self, propagator, sample_distributions):
        """Test that convergence improves with more samples."""
        def f(vals):
            return vals["x"]

        results = []
        for n_samples in [100, 1000, 10000]:
            result = propagator.propagate_monte_carlo(
                inputs=sample_distributions,
                function=f,
                n_samples=n_samples,
                seed=42,
            )
            results.append(result)

        # Error in mean should decrease with more samples
        # (comparing to true mean of 100)
        errors = [abs(r.mean - 100.0) for r in results]
        # Generally errors should decrease (allowing for randomness)
        assert errors[2] < errors[0] + 1.0  # 10000 samples better than 100

    def test_convergence_threshold(self):
        """Test convergence threshold constant."""
        assert MC_CONVERGENCE_THRESHOLD == 0.01  # 1%

    def test_insufficient_samples_no_convergence_check(self, propagator):
        """Test that convergence check is skipped for small samples."""
        distributions = {
            "x": Distribution(
                distribution_type=DistributionType.NORMAL,
                parameters={"mean": 0.0, "std": 1.0},
            ),
        }

        def f(vals):
            return vals["x"]

        # With < 1000 samples, convergence check is skipped
        result = propagator.propagate_monte_carlo(
            inputs=distributions,
            function=f,
            n_samples=500,
            convergence_check=True,
        )

        # Should still work
        assert math.isfinite(result.mean)


# =============================================================================
# Performance Tests
# =============================================================================

@pytest.mark.slow
class TestPerformance:
    """Performance tests for uncertainty propagation."""

    def test_linear_propagation_performance(
        self, propagator, sample_uncertain_inputs, benchmark
    ):
        """Benchmark linear propagation."""
        coefficients = {"x": 1.0, "y": 1.0}

        def propagate():
            return propagator.propagate_linear(
                inputs=sample_uncertain_inputs,
                coefficients=coefficients,
            )

        benchmark(propagate)

    def test_monte_carlo_performance(
        self, propagator, sample_distributions, benchmark
    ):
        """Benchmark Monte Carlo propagation."""
        def f(vals):
            return vals["x"] + vals["y"]

        def propagate():
            return propagator.propagate_monte_carlo(
                inputs=sample_distributions,
                function=f,
                n_samples=1000,
            )

        benchmark(propagate)

    def test_large_monte_carlo(self, propagator, sample_distributions):
        """Test Monte Carlo with large sample size."""
        def f(vals):
            return vals["x"] * vals["y"]

        start_time = time.perf_counter()
        result = propagator.propagate_monte_carlo(
            inputs=sample_distributions,
            function=f,
            n_samples=100000,
        )
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Should complete in reasonable time
        assert elapsed_ms < 5000, f"Took {elapsed_ms:.1f}ms for 100000 samples"
        assert result.convergence_achieved


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases."""

    def test_zero_uncertainty(self, propagator):
        """Test propagation with zero uncertainty."""
        inputs = {
            "x": UncertainValue(
                mean=100.0, std=0.0, lower_95=100.0, upper_95=100.0,
                distribution_type=DistributionType.NORMAL, timestamp=datetime.now(timezone.utc)
            ),
        }

        result = propagator.propagate_linear(
            inputs=inputs,
            coefficients={"x": 1.0},
        )

        assert result.uncertainty == 0.0

    def test_single_input(self, propagator):
        """Test propagation with single input."""
        inputs = {
            "x": UncertainValue(
                mean=100.0, std=5.0, lower_95=90.2, upper_95=109.8,
                distribution_type=DistributionType.NORMAL, timestamp=datetime.now(timezone.utc)
            ),
        }

        result = propagator.propagate_linear(
            inputs=inputs,
            coefficients={"x": 2.0},
        )

        assert abs(result.value - 200.0) < 0.01
        assert abs(result.uncertainty - 10.0) < 0.01  # 2 * 5

    def test_many_inputs(self, propagator):
        """Test propagation with many inputs."""
        inputs = {
            f"x{i}": UncertainValue(
                mean=10.0, std=1.0, lower_95=8.04, upper_95=11.96,
                distribution_type=DistributionType.NORMAL, timestamp=datetime.now(timezone.utc)
            )
            for i in range(20)
        }
        coefficients = {f"x{i}": 1.0 for i in range(20)}

        result = propagator.propagate_linear(
            inputs=inputs,
            coefficients=coefficients,
        )

        # Sum of 20 values of mean 10 = 200
        assert abs(result.value - 200.0) < 0.01

        # Std = sqrt(20 * 1) = 4.47
        assert abs(result.uncertainty - 4.47) < 0.1

    def test_negative_values(self, propagator):
        """Test propagation with negative mean values."""
        inputs = {
            "x": UncertainValue(
                mean=-100.0, std=5.0, lower_95=-109.8, upper_95=-90.2,
                distribution_type=DistributionType.NORMAL, timestamp=datetime.now(timezone.utc)
            ),
        }

        result = propagator.propagate_linear(
            inputs=inputs,
            coefficients={"x": 1.0},
        )

        assert abs(result.value - (-100.0)) < 0.01

    def test_monte_carlo_function_failure_handling(self, propagator):
        """Test Monte Carlo handles function evaluation failures."""
        distributions = {
            "x": Distribution(
                distribution_type=DistributionType.NORMAL,
                parameters={"mean": 0.0, "std": 1.0},
            ),
        }

        def f(vals):
            if vals["x"] < -2:  # Occasional failure
                raise ValueError("Negative value")
            return vals["x"]

        # Should handle failures gracefully
        result = propagator.propagate_monte_carlo(
            inputs=distributions,
            function=f,
            n_samples=1000,
        )

        # Result should still be computed from successful samples
        assert math.isfinite(result.mean)
        assert result.n_samples < 1000  # Some samples failed
