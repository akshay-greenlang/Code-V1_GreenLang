# -*- coding: utf-8 -*-
"""
Tests for Monte Carlo Simulation Engine

Tests cover:
- Sample generation (normal, lognormal, uniform, triangular)
- Statistical calculation
- Monte Carlo simulation execution
- Analytical propagation
- Sensitivity analysis
- Performance benchmarks

Version: 1.0.0
Date: 2025-10-30
"""

import pytest
import numpy as np
import time
from typing import Dict
from services.methodologies import (
    MonteCarloSimulator,
    MonteCarloInput,
    MonteCarloResult,
    AnalyticalPropagator,
    run_monte_carlo,
    DistributionType,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def simulator():
    """Create a MonteCarloSimulator with fixed seed."""
    return MonteCarloSimulator(seed=42)


@pytest.fixture
def simple_parameters():
    """Create simple test parameters."""
    return {
        "activity": MonteCarloInput(
            name="activity",
            mean=1000.0,
            std_dev=100.0,
            distribution=DistributionType.NORMAL,
        ),
        "factor": MonteCarloInput(
            name="factor",
            mean=2.5,
            std_dev=0.25,
            distribution=DistributionType.NORMAL,
        ),
    }


# ============================================================================
# SAMPLE GENERATION TESTS
# ============================================================================

def test_generate_normal_samples(simulator):
    """Test normal distribution sample generation."""
    param = MonteCarloInput(
        name="test",
        mean=100.0,
        std_dev=10.0,
        distribution=DistributionType.NORMAL,
    )

    samples = simulator.generate_samples(param, 10000)

    assert len(samples) == 10000
    assert np.mean(samples) == pytest.approx(100.0, rel=0.05)
    assert np.std(samples) == pytest.approx(10.0, rel=0.05)


def test_generate_lognormal_samples(simulator):
    """Test lognormal distribution sample generation."""
    param = MonteCarloInput(
        name="test",
        mean=100.0,
        std_dev=20.0,
        distribution=DistributionType.LOGNORMAL,
    )

    samples = simulator.generate_samples(param, 10000)

    assert len(samples) == 10000
    assert np.mean(samples) == pytest.approx(100.0, rel=0.1)
    # All samples should be positive for lognormal
    assert np.all(samples > 0)


def test_generate_uniform_samples(simulator):
    """Test uniform distribution sample generation."""
    param = MonteCarloInput(
        name="test",
        mean=100.0,
        std_dev=10.0,
        distribution=DistributionType.UNIFORM,
        min_value=50.0,
        max_value=150.0,
    )

    samples = simulator.generate_samples(param, 10000)

    assert len(samples) == 10000
    assert np.min(samples) >= 50.0
    assert np.max(samples) <= 150.0
    assert np.mean(samples) == pytest.approx(100.0, rel=0.05)


def test_generate_triangular_samples(simulator):
    """Test triangular distribution sample generation."""
    param = MonteCarloInput(
        name="test",
        mean=100.0,  # Mode
        std_dev=10.0,
        distribution=DistributionType.TRIANGULAR,
        min_value=80.0,
        max_value=120.0,
    )

    samples = simulator.generate_samples(param, 10000)

    assert len(samples) == 10000
    assert np.min(samples) >= 80.0
    assert np.max(samples) <= 120.0


def test_invalid_distribution():
    """Test handling of invalid distribution type."""
    simulator = MonteCarloSimulator(seed=42)

    # Create parameter with invalid distribution (using string)
    param = MonteCarloInput(
        name="test",
        mean=100.0,
        std_dev=10.0,
        distribution="invalid_dist",  # type: ignore
    )

    with pytest.raises(ValueError):
        simulator.generate_samples(param, 1000)


# ============================================================================
# STATISTICAL CALCULATION TESTS
# ============================================================================

def test_calculate_statistics(simulator):
    """Test statistical calculation."""
    samples = np.random.normal(100, 10, 10000)
    stats = simulator.calculate_statistics(samples)

    assert "mean" in stats
    assert "median" in stats
    assert "std_dev" in stats
    assert "variance" in stats
    assert "min" in stats
    assert "max" in stats
    assert "skewness" in stats
    assert "kurtosis" in stats

    # Check percentiles
    assert "p5" in stats
    assert "p50" in stats
    assert "p95" in stats

    # Validate values
    assert stats["mean"] == pytest.approx(100, rel=0.05)
    assert stats["std_dev"] == pytest.approx(10, rel=0.1)
    assert stats["p50"] == pytest.approx(stats["median"], rel=0.01)


# ============================================================================
# SIMULATION EXECUTION TESTS
# ============================================================================

def test_run_simple_simulation(simulator, simple_parameters):
    """Test running a simple Monte Carlo simulation."""

    def calc(params: Dict[str, float]) -> float:
        return params["activity"] * params["factor"]

    result = simulator.run_simulation(calc, simple_parameters, iterations=1000)

    assert isinstance(result, MonteCarloResult)
    assert result.iterations == 1000
    assert result.seed == 42
    assert result.mean > 0
    assert result.std_dev > 0
    assert result.p5 < result.p50 < result.p95


def test_simulation_result_statistics(simulator, simple_parameters):
    """Test statistical properties of simulation results."""

    def calc(params: Dict[str, float]) -> float:
        return params["activity"] * params["factor"]

    result = simulator.run_simulation(calc, simple_parameters, iterations=10000)

    # Expected mean: 1000 × 2.5 = 2500
    assert result.mean == pytest.approx(2500, rel=0.05)

    # Check percentiles are ordered
    assert result.p5 < result.p25 < result.p50 < result.p75 < result.p95

    # Check min/max bounds
    assert result.min_value <= result.p5
    assert result.max_value >= result.p95


def test_simulation_reproducibility():
    """Test that simulations with same seed produce same results."""
    params = {
        "x": MonteCarloInput(name="x", mean=100, std_dev=10),
    }

    def calc(p: Dict[str, float]) -> float:
        return p["x"] * 2

    sim1 = MonteCarloSimulator(seed=123)
    result1 = sim1.run_simulation(calc, params, iterations=1000)

    sim2 = MonteCarloSimulator(seed=123)
    result2 = sim2.run_simulation(calc, params, iterations=1000)

    assert result1.mean == result2.mean
    assert result1.std_dev == result2.std_dev


def test_simulation_different_seeds():
    """Test that simulations with different seeds produce different results."""
    params = {
        "x": MonteCarloInput(name="x", mean=100, std_dev=10),
    }

    def calc(p: Dict[str, float]) -> float:
        return p["x"] * 2

    sim1 = MonteCarloSimulator(seed=123)
    result1 = sim1.run_simulation(calc, params, iterations=1000)

    sim2 = MonteCarloSimulator(seed=456)
    result2 = sim2.run_simulation(calc, params, iterations=1000)

    # Results should be close but not identical
    assert abs(result1.mean - result2.mean) < 50  # Close
    assert result1.mean != result2.mean  # But different


# ============================================================================
# UNCERTAINTY PROPAGATION TESTS
# ============================================================================

def test_propagate_uncertainty(simulator):
    """Test uncertainty propagation."""

    def calc(p: Dict[str, float]) -> float:
        return p["a"] * p["b"] + p["c"]

    means = {"a": 100, "b": 2.5, "c": 50}
    uncert = {"a": 0.1, "b": 0.15, "c": 0.2}

    result = simulator.propagate_uncertainty(
        means, uncert, calc, iterations=1000
    )

    # Expected: 100 × 2.5 + 50 = 300
    assert result.mean == pytest.approx(300, rel=0.1)
    assert result.std_dev > 0


def test_simple_propagation(simulator):
    """Test simple emission calculation propagation."""
    result = simulator.simple_propagation(
        activity_data=1000.0,
        activity_uncertainty=0.1,
        emission_factor=2.5,
        factor_uncertainty=0.15,
        iterations=10000,
    )

    # Expected: 1000 × 2.5 = 2500
    assert result.mean == pytest.approx(2500, rel=0.05)

    # Check uncertainty propagated correctly
    # CV² = (0.1)² + (0.15)² = 0.01 + 0.0225 = 0.0325
    # CV = sqrt(0.0325) ≈ 0.18
    expected_cv = np.sqrt(0.1**2 + 0.15**2)
    assert result.coefficient_of_variation == pytest.approx(expected_cv, rel=0.1)


# ============================================================================
# SENSITIVITY ANALYSIS TESTS
# ============================================================================

def test_sensitivity_indices(simulator, simple_parameters):
    """Test sensitivity index calculation."""

    def calc(params: Dict[str, float]) -> float:
        return params["activity"] * params["factor"]

    result = simulator.run_simulation(calc, simple_parameters, iterations=5000)

    assert result.sensitivity_indices is not None
    assert "activity" in result.sensitivity_indices
    assert "factor" in result.sensitivity_indices

    # Both parameters should have significant sensitivity
    assert abs(result.sensitivity_indices["activity"]) > 0.5
    assert abs(result.sensitivity_indices["factor"]) > 0.5


def test_top_contributors(simulator):
    """Test identification of top contributors."""
    params = {
        "dominant": MonteCarloInput(name="dominant", mean=1000, std_dev=200),
        "minor": MonteCarloInput(name="minor", mean=10, std_dev=1),
    }

    def calc(p: Dict[str, float]) -> float:
        return p["dominant"] + p["minor"]

    result = simulator.run_simulation(calc, params, iterations=5000)

    assert result.top_contributors is not None
    # Dominant parameter should be identified as top contributor
    assert "dominant" in result.top_contributors


# ============================================================================
# ANALYTICAL PROPAGATION TESTS
# ============================================================================

def test_analytical_multiply():
    """Test analytical multiplication propagation."""
    mean_result, std_result = AnalyticalPropagator.multiply(
        mean1=100, std1=10,
        mean2=2.5, std2=0.25
    )

    # Expected mean: 100 × 2.5 = 250
    assert mean_result == 250

    # Expected CV: sqrt((10/100)² + (0.25/2.5)²) = sqrt(0.01 + 0.01) ≈ 0.141
    expected_cv = np.sqrt((10/100)**2 + (0.25/2.5)**2)
    calculated_cv = std_result / mean_result
    assert calculated_cv == pytest.approx(expected_cv, rel=0.01)


def test_analytical_add():
    """Test analytical addition propagation."""
    mean_result, std_result = AnalyticalPropagator.add(
        mean1=100, std1=10,
        mean2=50, std2=5
    )

    # Expected mean: 100 + 50 = 150
    assert mean_result == 150

    # Expected std: sqrt(10² + 5²) = sqrt(125) ≈ 11.18
    expected_std = np.sqrt(10**2 + 5**2)
    assert std_result == pytest.approx(expected_std, rel=0.01)


def test_analytical_simple_emission():
    """Test analytical emission calculation."""
    mean, std = AnalyticalPropagator.simple_emission(
        activity_mean=1000,
        activity_std=100,
        factor_mean=2.5,
        factor_std=0.25,
    )

    # Expected mean: 1000 × 2.5 = 2500
    assert mean == 2500

    # Check uncertainty is reasonable
    cv = std / mean
    assert 0.1 < cv < 0.2


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

def test_simulation_performance(simulator, simple_parameters):
    """Test that 10,000 iterations complete in <1 second."""

    def calc(params: Dict[str, float]) -> float:
        return params["activity"] * params["factor"]

    start = time.time()
    result = simulator.run_simulation(calc, simple_parameters, iterations=10000)
    elapsed = time.time() - start

    assert elapsed < 1.0  # Should complete in less than 1 second
    assert result.computation_time is not None
    assert result.computation_time < 1.0


@pytest.mark.slow
def test_large_simulation_performance(simulator, simple_parameters):
    """Test performance with large iteration count."""

    def calc(params: Dict[str, float]) -> float:
        return params["activity"] * params["factor"]

    start = time.time()
    result = simulator.run_simulation(calc, simple_parameters, iterations=100000)
    elapsed = time.time() - start

    assert elapsed < 5.0  # Should complete in less than 5 seconds
    assert result.iterations == 100000


# ============================================================================
# CONVENIENCE FUNCTION TESTS
# ============================================================================

def test_run_monte_carlo_convenience():
    """Test convenience function for Monte Carlo."""
    result = run_monte_carlo(
        activity_data=1000.0,
        activity_uncertainty=0.1,
        emission_factor=2.5,
        factor_uncertainty=0.15,
        iterations=1000,
        seed=42,
    )

    assert isinstance(result, MonteCarloResult)
    assert result.mean == pytest.approx(2500, rel=0.1)


# ============================================================================
# EDGE CASE TESTS
# ============================================================================

def test_zero_uncertainty(simulator):
    """Test simulation with zero uncertainty."""
    params = {
        "x": MonteCarloInput(name="x", mean=100, std_dev=0),
    }

    def calc(p: Dict[str, float]) -> float:
        return p["x"] * 2

    result = simulator.run_simulation(calc, params, iterations=1000)

    # All samples should be identical
    assert result.std_dev == pytest.approx(0, abs=1e-6)
    assert result.p5 == pytest.approx(result.p95, abs=1e-6)


def test_negative_values(simulator):
    """Test simulation with negative means (edge case)."""
    params = {
        "x": MonteCarloInput(name="x", mean=-100, std_dev=10),
    }

    def calc(p: Dict[str, float]) -> float:
        return abs(p["x"])

    # Should handle gracefully
    result = simulator.run_simulation(calc, params, iterations=1000)
    assert result.mean > 0  # Result is absolute value


def test_iteration_limits(simulator, simple_parameters):
    """Test iteration limits validation."""

    def calc(p: Dict[str, float]) -> float:
        return p["activity"] * p["factor"]

    # Too few iterations
    with pytest.raises(ValueError):
        simulator.run_simulation(calc, simple_parameters, iterations=100)

    # Valid minimum
    result = simulator.run_simulation(calc, simple_parameters, iterations=1000)
    assert result.iterations == 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
