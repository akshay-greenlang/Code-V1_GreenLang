"""
Integration test for SIM-401: Scenario spec + Seeded RNG + Provenance

Tests the complete end-to-end workflow:
1. Load scenario spec from YAML
2. Initialize scenario runner with seeded RNG
3. Generate parameter samples (sweep + Monte Carlo)
4. Verify seed is recorded in provenance
5. Verify reproducibility (round-trip)
"""

import json
import sys
from pathlib import Path

# Ensure UTF-8 output for Windows console
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from greenlang.specs.scenariospec_v1 import from_yaml, ScenarioSpecV1
from greenlang.simulation.runner import ScenarioRunner
from greenlang.intelligence.glrng import GLRNG


def test_scenario_spec_loading():
    """Test scenario spec can be loaded from YAML."""
    print("\n[TEST 1] Loading scenario spec from YAML...")

    # Load baseline sweep scenario
    spec = from_yaml("docs/scenarios/examples/baseline_sweep.yaml")

    assert spec.name == "building_baseline_sweep"
    assert spec.seed == 42
    assert len(spec.parameters) == 2
    assert spec.parameters[0].id == "retrofit_level"
    assert spec.parameters[1].id == "chiller_cop"

    print("✓ Scenario spec loaded successfully")
    print(f"  - Name: {spec.name}")
    print(f"  - Seed: {spec.seed}")
    print(f"  - Parameters: {len(spec.parameters)}")

    return spec


def test_glrng_determinism():
    """Test GLRNG produces deterministic results."""
    print("\n[TEST 2] Testing GLRNG determinism...")

    # Create two RNGs with same seed
    rng1 = GLRNG(seed=42)
    rng2 = GLRNG(seed=42)

    # Generate 100 samples from each
    samples1 = [rng1.uniform() for _ in range(100)]
    samples2 = [rng2.uniform() for _ in range(100)]

    # Should be identical
    assert samples1 == samples2

    print("✓ GLRNG is deterministic")
    print(f"  - First 5 samples: {samples1[:5]}")
    print(f"  - All 100 samples matched")

    return samples1


def test_glrng_substreams():
    """Test GLRNG substream independence."""
    print("\n[TEST 3] Testing GLRNG substream derivation...")

    root = GLRNG(seed=42)

    # Create two independent substreams
    stream1 = root.spawn("param:temperature|trial:0")
    stream2 = root.spawn("param:pressure|trial:0")

    # Generate samples
    vals1 = [stream1.uniform() for _ in range(10)]
    vals2 = [stream2.uniform() for _ in range(10)]

    # Should be different (independent)
    assert vals1 != vals2

    # But reproducible
    stream1_copy = root.spawn("param:temperature|trial:0")
    vals1_copy = [stream1_copy.uniform() for _ in range(10)]
    assert vals1 == vals1_copy

    print("✓ GLRNG substreams are independent and reproducible")
    print(f"  - Stream 1 first value: {vals1[0]}")
    print(f"  - Stream 2 first value: {vals2[0]}")
    print(f"  - Reproducibility verified")

    return vals1, vals2


def test_scenario_runner_grid_sweep():
    """Test scenario runner with pure grid sweep."""
    print("\n[TEST 4] Testing scenario runner with grid sweep...")

    spec = from_yaml("docs/scenarios/examples/baseline_sweep.yaml")
    runner = ScenarioRunner(spec=spec)

    # Generate all samples
    samples = list(runner.generate_samples())

    # Should have 3 retrofit levels × 4 COP values = 12 scenarios
    assert len(samples) == 12

    # Verify parameter coverage
    retrofit_levels = {s["retrofit_level"] for s in samples}
    cop_values = {s["chiller_cop"] for s in samples}

    assert retrofit_levels == {"none", "light", "deep"}
    assert cop_values == {3.2, 3.6, 4.0, 4.4}

    print("✓ Grid sweep generated correctly")
    print(f"  - Total scenarios: {len(samples)}")
    print(f"  - Retrofit levels: {retrofit_levels}")
    print(f"  - COP values: {cop_values}")

    return samples


def test_scenario_runner_monte_carlo():
    """Test scenario runner with Monte Carlo sampling."""
    print("\n[TEST 5] Testing scenario runner with Monte Carlo...")

    spec = from_yaml("docs/scenarios/examples/monte_carlo.yaml")
    runner = ScenarioRunner(spec=spec)

    # Generate first 100 samples (out of 18,000 total)
    samples = []
    for i, sample in enumerate(runner.generate_samples()):
        samples.append(sample)
        if i >= 99:
            break

    assert len(samples) == 100

    # Verify all samples have all parameters
    for sample in samples:
        assert "retrofit_level" in sample
        assert "chiller_cop" in sample
        assert "electricity_price_usd_per_kwh" in sample

    # Verify stochastic parameter is sampled
    prices = [s["electricity_price_usd_per_kwh"] for s in samples]
    assert all(0.08 <= p <= 0.22 for p in prices)
    assert len(set(prices)) > 10  # Should have variety (not all identical)

    print("✓ Monte Carlo sampling works correctly")
    print(f"  - Samples generated: {len(samples)}")
    print(f"  - Price range: [{min(prices):.4f}, {max(prices):.4f}]")
    print(f"  - Unique prices: {len(set(prices))}")

    return samples


def test_provenance_seed_recording():
    """Test that seed is recorded in provenance."""
    print("\n[TEST 6] Testing seed recording in provenance...")

    spec = from_yaml("docs/scenarios/examples/baseline_sweep.yaml")
    runner = ScenarioRunner(spec=spec)

    # Check provenance context has seed info
    assert hasattr(runner.provenance_ctx, "metadata")
    assert "seed_tracking" in runner.provenance_ctx.metadata

    seed_info = runner.provenance_ctx.metadata["seed_tracking"]

    assert seed_info["seed_root"] == 42
    assert seed_info["spec_type"] == "scenario"
    assert "spec_hash_scenario" in seed_info
    assert "recorded_at" in seed_info

    print("✓ Seed recorded in provenance")
    print(f"  - Seed root: {seed_info['seed_root']}")
    print(f"  - Spec hash: {seed_info['spec_hash_scenario'][:16]}...")
    print(f"  - Recorded at: {seed_info['recorded_at']}")

    return seed_info


def test_reproducibility_round_trip():
    """Test that scenarios can be reproduced from same seed."""
    print("\n[TEST 7] Testing reproducibility (round-trip)...")

    spec = from_yaml("docs/scenarios/examples/monte_carlo.yaml")

    # Run 1
    runner1 = ScenarioRunner(spec=spec)
    samples1 = []
    for i, sample in enumerate(runner1.generate_samples()):
        samples1.append(sample)
        if i >= 99:
            break

    # Run 2 (same spec, same seed)
    runner2 = ScenarioRunner(spec=spec)
    samples2 = []
    for i, sample in enumerate(runner2.generate_samples()):
        samples2.append(sample)
        if i >= 99:
            break

    # Should be identical
    assert len(samples1) == len(samples2)

    for s1, s2 in zip(samples1, samples2):
        assert s1["retrofit_level"] == s2["retrofit_level"]
        assert s1["chiller_cop"] == s2["chiller_cop"]
        assert s1["electricity_price_usd_per_kwh"] == s2["electricity_price_usd_per_kwh"]

    print("✓ Reproducibility verified (round-trip successful)")
    print(f"  - Run 1 first price: {samples1[0]['electricity_price_usd_per_kwh']:.6f}")
    print(f"  - Run 2 first price: {samples2[0]['electricity_price_usd_per_kwh']:.6f}")
    print(f"  - All 100 samples matched exactly")

    return samples1, samples2


def main():
    """Run all integration tests."""
    print("="*70)
    print("SIM-401 INTEGRATION TEST SUITE")
    print("Testing: Scenario Spec + Seeded RNG + Provenance Round-Trip")
    print("="*70)

    try:
        # Test 1: Scenario spec loading
        spec = test_scenario_spec_loading()

        # Test 2: GLRNG determinism
        samples = test_glrng_determinism()

        # Test 3: GLRNG substreams
        streams = test_glrng_substreams()

        # Test 4: Grid sweep
        grid_samples = test_scenario_runner_grid_sweep()

        # Test 5: Monte Carlo
        mc_samples = test_scenario_runner_monte_carlo()

        # Test 6: Provenance seed recording
        seed_info = test_provenance_seed_recording()

        # Test 7: Reproducibility round-trip
        round_trip = test_reproducibility_round_trip()

        print("\n" + "="*70)
        print("ALL TESTS PASSED ✓")
        print("="*70)
        print("\nSIM-401 Acceptance Criteria VERIFIED:")
        print("  ✓ Scenario spec outline implemented")
        print("  ✓ Seeded RNG helper functional")
        print("  ✓ Seed stored in provenance")
        print("  ✓ Round-trip reproducibility confirmed")
        print("\nStatus: COMPLETE - READY FOR INTEGRATION")
        print("="*70)

        return 0

    except Exception as e:
        print("\n" + "="*70)
        print(f"TEST FAILED: {e}")
        print("="*70)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
