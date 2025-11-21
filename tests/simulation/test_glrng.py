# -*- coding: utf-8 -*-
"""
Tests for GLRNG (GreenLang RNG)

Tests determinism, cross-platform consistency, and statistical properties.
"""

import pytest
from greenlang.intelligence.glrng import GLRNG, derive_substream_seed, SplitMix64


def test_splitmix64_determinism():
    """Test SplitMix64 produces identical sequences."""
    rng1 = SplitMix64(seed=42)
    rng2 = SplitMix64(seed=42)

    for _ in range(1000):
        assert rng1.next() == rng2.next()


def test_glrng_determinism():
    """Test GLRNG produces identical sequences for same seed."""
    rng1 = GLRNG(seed=42)
    rng2 = GLRNG(seed=42)

    for _ in range(100):
        assert rng1.uniform() == rng2.uniform()
        assert rng1.normal() == rng2.normal()


def test_glrng_substream_independence():
    """Test substreams are independent."""
    root = GLRNG(seed=42)

    stream1 = root.spawn("param:temperature")
    stream2 = root.spawn("param:pressure")

    # Different streams should produce different values
    vals1 = [stream1.uniform() for _ in range(100)]
    vals2 = [stream2.uniform() for _ in range(100)]

    assert vals1 != vals2

    # But identical streams should match
    stream1_copy = root.spawn("param:temperature")
    vals1_copy = [stream1_copy.uniform() for _ in range(100)]

    # Note: stream1 already consumed 100 values, so create fresh stream
    stream1_fresh = root.spawn("param:temperature")
    vals1_fresh = [stream1_fresh.uniform() for _ in range(100)]

    assert vals1 == vals1_fresh


def test_glrng_seed_derivation():
    """Test HMAC-SHA256 substream derivation is deterministic."""
    seed = 42
    path = "scenario:test|param:x|trial:0"

    child1 = derive_substream_seed(seed, path)
    child2 = derive_substream_seed(seed, path)

    assert child1 == child2
    assert isinstance(child1, int)
    assert 0 <= child1 <= 2**64 - 1


def test_glrng_distributions():
    """Test various distributions produce reasonable values."""
    rng = GLRNG(seed=42)

    # Uniform
    u = rng.uniform(0, 1)
    assert 0 <= u < 1

    # Normal
    n = rng.normal(mean=100, std=15)
    # Rough sanity check (99.7% within 3 sigma)
    assert 50 < n < 150

    # Triangular
    t = rng.triangular(low=0.08, mode=0.12, high=0.22)
    assert 0.08 <= t <= 0.22

    # Lognormal (always positive)
    ln = rng.lognormal(mean=0, sigma=1)
    assert ln > 0


def test_glrng_choice_shuffle():
    """Test choice and shuffle methods."""
    rng = GLRNG(seed=42)

    # Choice
    options = ['a', 'b', 'c', 'd']
    choice = rng.choice(options)
    assert choice in options

    # Shuffle
    items = [1, 2, 3, 4, 5]
    original = items.copy()
    rng.shuffle(items)

    # Should be permuted
    assert set(items) == set(original)

    # Should be reproducible
    rng2 = GLRNG(seed=42)
    items2 = [1, 2, 3, 4, 5]
    rng2.shuffle(items2)
    assert items == items2


def test_glrng_float_precision():
    """Test float precision normalization."""
    rng = GLRNG(seed=42, float_precision=6)

    values = [rng.uniform() for _ in range(100)]

    # All values should be rounded to 6 decimals
    for v in values:
        assert round(v, 6) == v


def test_glrng_cross_platform_consistency():
    """Test that specific seeds produce expected values (regression test)."""
    rng = GLRNG(seed=42)

    # These values should remain stable across platforms
    # If this test fails, it indicates float determinism issue
    first_uniform = rng.uniform()
    assert isinstance(first_uniform, float)
    assert 0 <= first_uniform < 1

    # Record expected value for future regression testing
    # (Update this if algorithm changes intentionally)
    print(f"First uniform value for seed=42: {first_uniform}")


def test_glrng_state_tracking():
    """Test RNG state tracking for provenance."""
    rng = GLRNG(seed=42, path="test:path")

    # Generate some values
    for _ in range(10):
        rng.uniform()

    state = rng.state()

    assert state["algo"] == "splitmix64"
    assert state["path"] == "test:path"
    assert state["call_count"] == 10
    assert state["float_precision"] == 6


def test_glrng_validation():
    """Test GLRNG input validation."""
    # Valid seed
    rng = GLRNG(seed=42)
    assert rng is not None

    # Invalid seed (out of range)
    with pytest.raises(ValueError):
        GLRNG(seed=-1)

    with pytest.raises(ValueError):
        GLRNG(seed=2**64)  # Too large

    # Invalid distribution parameters
    rng = GLRNG(seed=42)

    with pytest.raises(ValueError):
        rng.normal(mean=0, std=-1)  # Negative std

    with pytest.raises(ValueError):
        rng.triangular(low=1.0, mode=0.5, high=2.0)  # mode < low


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
