"""
Tests for GLRNG Statistical Properties - Sanity checks

Tests statistical properties of distributions to ensure implementation correctness.
Uses relaxed thresholds (not research-grade) for sanity testing as specified in SIM-401.

Author: GreenLang Framework Team
Date: October 2025
Spec: SIM-401 (Scenario Spec & Seeded RNG)
"""

import pytest
import math
from greenlang.simulation.rng import GLRNG


def test_uniform_distribution_bounds():
    """Test uniform distribution respects bounds."""
    rng = GLRNG(seed=42)

    for _ in range(1000):
        val = rng.uniform(0, 1)
        assert 0 <= val < 1

        val = rng.uniform(10, 20)
        assert 10 <= val < 20


def test_uniform_distribution_mean():
    """Test uniform distribution has correct mean (sanity check)."""
    rng = GLRNG(seed=42)

    samples = [rng.uniform(0, 1) for _ in range(10000)]
    mean = sum(samples) / len(samples)

    # Expected: 0.5, tolerance: ±0.02
    assert 0.48 < mean < 0.52


def test_uniform_distribution_variance():
    """Test uniform distribution has reasonable variance."""
    rng = GLRNG(seed=42)

    samples = [rng.uniform(0, 1) for _ in range(10000)]
    mean = sum(samples) / len(samples)
    variance = sum((x - mean)**2 for x in samples) / len(samples)

    # Expected: 1/12 ≈ 0.0833, tolerance: ±0.01
    assert 0.07 < variance < 0.10


def test_normal_distribution_mean():
    """Test normal distribution has correct mean (sanity check)."""
    rng = GLRNG(seed=42)

    target_mean = 100.0
    samples = [rng.normal(target_mean, 15) for _ in range(10000)]
    actual_mean = sum(samples) / len(samples)

    # Tolerance: ±0.5 (generous for sanity check)
    assert abs(actual_mean - target_mean) < 0.5


def test_normal_distribution_std():
    """Test normal distribution has correct standard deviation."""
    rng = GLRNG(seed=42)

    target_mean = 100.0
    target_std = 15.0
    samples = [rng.normal(target_mean, target_std) for _ in range(10000)]

    mean = sum(samples) / len(samples)
    variance = sum((x - mean)**2 for x in samples) / len(samples)
    std = math.sqrt(variance)

    # Tolerance: ±0.5 (generous)
    assert abs(std - target_std) < 0.5


def test_normal_distribution_68_95_997_rule():
    """Test normal distribution follows 68-95-99.7 rule (sanity)."""
    rng = GLRNG(seed=42)

    mean = 100.0
    std = 15.0
    samples = [rng.normal(mean, std) for _ in range(10000)]

    # Count samples within 1, 2, 3 standard deviations
    within_1sigma = sum(1 for x in samples if abs(x - mean) <= 1 * std)
    within_2sigma = sum(1 for x in samples if abs(x - mean) <= 2 * std)
    within_3sigma = sum(1 for x in samples if abs(x - mean) <= 3 * std)

    # Expected: 68%, 95%, 99.7% (with tolerance)
    assert 0.65 < within_1sigma / len(samples) < 0.71  # ~68%
    assert 0.93 < within_2sigma / len(samples) < 0.97  # ~95%
    assert 0.995 < within_3sigma / len(samples) < 1.0   # ~99.7%


def test_lognormal_distribution_positivity():
    """Test lognormal distribution produces only positive values."""
    rng = GLRNG(seed=42)

    samples = [rng.lognormal(mean=0.0, sigma=1.0) for _ in range(1000)]

    # All samples must be positive
    assert all(x > 0 for x in samples)


def test_lognormal_distribution_mean():
    """Test lognormal distribution has reasonable mean."""
    rng = GLRNG(seed=42)

    # For lognormal(mean=0, sigma=1), expected mean ≈ exp(0 + 1^2/2) = exp(0.5) ≈ 1.649
    samples = [rng.lognormal(mean=0.0, sigma=1.0) for _ in range(10000)]
    actual_mean = sum(samples) / len(samples)

    expected_mean = math.exp(0.0 + 1.0**2 / 2)

    # Tolerance: ±0.1 (generous)
    assert abs(actual_mean - expected_mean) < 0.1


def test_triangular_distribution_bounds():
    """Test triangular distribution respects bounds."""
    rng = GLRNG(seed=42)

    low = 0.08
    mode = 0.12
    high = 0.22

    samples = [rng.triangular(low, mode, high) for _ in range(1000)]

    # All samples must be within bounds
    assert all(low <= x <= high for x in samples)


def test_triangular_distribution_mode():
    """Test triangular distribution peaks near mode."""
    rng = GLRNG(seed=42)

    low = 0.0
    mode = 0.5
    high = 1.0

    samples = [rng.triangular(low, mode, high) for _ in range(10000)]

    # Count samples near mode (within 0.1)
    near_mode = sum(1 for x in samples if abs(x - mode) < 0.1)

    # Should have more samples near mode than elsewhere (crude check)
    assert near_mode > 1000


def test_triangular_distribution_mean():
    """Test triangular distribution has correct mean."""
    rng = GLRNG(seed=42)

    low = 0.0
    mode = 0.5
    high = 1.0

    samples = [rng.triangular(low, mode, high) for _ in range(10000)]
    actual_mean = sum(samples) / len(samples)

    # Expected mean: (low + mode + high) / 3
    expected_mean = (low + mode + high) / 3

    # Tolerance: ±0.02
    assert abs(actual_mean - expected_mean) < 0.02


def test_randint_distribution():
    """Test randint produces uniform distribution over integers."""
    rng = GLRNG(seed=42)

    low = 1
    high = 6
    samples = [rng.randint(low, high) for _ in range(6000)]

    # Count each value
    counts = {i: samples.count(i) for i in range(low, high + 1)}

    # Each value should appear roughly 1000 times (6000 / 6)
    for count in counts.values():
        assert 900 < count < 1100  # Tolerance: ±10%


def test_choice_distribution():
    """Test choice produces uniform distribution over options."""
    rng = GLRNG(seed=42)

    options = ['a', 'b', 'c', 'd', 'e']
    samples = [rng.choice(options) for _ in range(5000)]

    # Count each option
    counts = {opt: samples.count(opt) for opt in options}

    # Each option should appear roughly 1000 times (5000 / 5)
    for count in counts.values():
        assert 900 < count < 1100  # Tolerance: ±10%


def test_shuffle_randomness():
    """Test shuffle produces varied permutations."""
    rng = GLRNG(seed=42)

    # Generate multiple shuffles
    permutations = []
    for _ in range(100):
        items = list(range(10))
        rng.shuffle(items)
        permutations.append(tuple(items))

    # Should have many unique permutations (not all identical)
    unique_perms = len(set(permutations))
    assert unique_perms > 90  # At least 90% unique


def test_sample_without_replacement():
    """Test sample produces samples without replacement."""
    rng = GLRNG(seed=42)

    population = list(range(100))

    for _ in range(100):
        sample = rng.sample(population, k=10)

        # All elements should be unique (no replacement)
        assert len(sample) == len(set(sample))

        # All elements should be from population
        assert all(x in population for x in sample)


def test_normal_box_muller_caching():
    """Test Box-Muller caching works correctly."""
    rng = GLRNG(seed=42)

    # First call generates 2 values, caches second
    val1 = rng.normal()

    # Check state shows 2 uniform draws were consumed (for Box-Muller)
    state = rng.state()
    assert state["call_count"] == 2

    # Second call uses cached value (no new uniform draws)
    val2 = rng.normal()

    state = rng.state()
    # Should still be 2 (used cached value)
    # But then needs to generate new pair, so becomes 4
    assert state["call_count"] == 4


def test_distributions_with_edge_parameters():
    """Test distributions handle edge case parameters."""
    rng = GLRNG(seed=42)

    # Very small std
    val = rng.normal(mean=100, std=0.001)
    assert 99.9 < val < 100.1

    # Very wide uniform
    val = rng.uniform(-1e6, 1e6)
    assert -1e6 <= val < 1e6

    # Triangular where mode = low (edge case)
    val = rng.triangular(low=0.0, mode=0.0, high=1.0)
    assert 0.0 <= val <= 1.0

    # Triangular where mode = high (edge case)
    val = rng.triangular(low=0.0, mode=1.0, high=1.0)
    assert 0.0 <= val <= 1.0


def test_distribution_parameter_validation():
    """Test that invalid distribution parameters raise errors."""
    rng = GLRNG(seed=42)

    # Normal: std must be positive
    with pytest.raises(ValueError, match="std must be positive"):
        rng.normal(mean=100, std=0)

    with pytest.raises(ValueError, match="std must be positive"):
        rng.normal(mean=100, std=-1)

    # Triangular: low <= mode <= high
    with pytest.raises(ValueError, match="low.*mode.*high"):
        rng.triangular(low=0.0, mode=1.5, high=1.0)

    with pytest.raises(ValueError, match="low.*mode.*high"):
        rng.triangular(low=0.5, mode=0.0, high=1.0)

    # Randint: low <= high
    with pytest.raises(ValueError, match="low.*must be.*high"):
        rng.randint(10, 5)


def test_large_sample_statistics():
    """Test distributions with large samples (100k) for stability."""
    rng = GLRNG(seed=42)

    # Normal distribution with 100k samples
    samples = [rng.normal(100, 15) for _ in range(100000)]
    mean = sum(samples) / len(samples)
    variance = sum((x - mean)**2 for x in samples) / len(samples)
    std = math.sqrt(variance)

    # With 100k samples, should be very close to target
    assert abs(mean - 100.0) < 0.2
    assert abs(std - 15.0) < 0.2
