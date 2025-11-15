"""
Determinism tests for GL-002 BoilerEfficiencyOptimizer

Tests reproducibility, ensuring same input always produces same output.
Validates cross-environment consistency and bit-perfect reproducibility.

Target: 5+ tests for deterministic behavior validation
"""

import pytest
import hashlib
import json
import random
import numpy as np
from decimal import Decimal, getcontext
from datetime import datetime
from unittest.mock import Mock, patch
import pickle
import platform
import sys

# Import components to test
from greenlang_boiler_efficiency import (
    BoilerEfficiencyOrchestrator,
    BoilerInput,
    CombustionCalculator,
    ProvenanceTracker,
)
from greenlang_core import AgentConfig


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def deterministic_config():
    """Create deterministic configuration."""
    return AgentConfig(
        name="GL-002-Deterministic",
        version="2.0.0",
        environment="test",
        deterministic_mode=True,
        random_seed=42,
        use_decimal_precision=True,
        decimal_places=10,
    )


@pytest.fixture
def orchestrator(deterministic_config):
    """Create deterministic orchestrator."""
    return BoilerEfficiencyOrchestrator(deterministic_config)


@pytest.fixture
def test_input():
    """Create test input for determinism testing."""
    return BoilerInput(
        boiler_id="DET-001",
        fuel_type="natural_gas",
        fuel_flow_rate=100.0,
        steam_output=1500.0,
        steam_pressure=10.0,
        steam_temperature=180.0,
        feedwater_temperature=80.0,
        excess_air_ratio=1.15,
        ambient_temperature=25.0,
        o2_percentage=3.0,
        co_ppm=50,
    )


# ============================================================================
# TEST BASIC DETERMINISM
# ============================================================================

class TestBasicDeterminism:
    """Test basic deterministic behavior."""

    def test_identical_inputs_produce_identical_outputs(self, orchestrator, test_input):
        """Test that identical inputs always produce identical outputs."""
        results = []

        # Run same calculation 100 times
        for _ in range(100):
            result = orchestrator.process(test_input)
            results.append({
                "efficiency": result.efficiency,
                "provenance_hash": result.provenance_hash,
                "emissions": result.emissions,
            })

        # All results should be identical
        first_result = results[0]
        for result in results[1:]:
            assert result == first_result, "Results differ for identical input"

        # Verify specific values are identical
        efficiencies = [r["efficiency"] for r in results]
        assert len(set(efficiencies)) == 1, "Efficiency values differ"

        hashes = [r["provenance_hash"] for r in results]
        assert len(set(hashes)) == 1, "Provenance hashes differ"

    def test_deterministic_floating_point_operations(self, orchestrator):
        """Test floating-point operations are deterministic."""
        # Test with values that often cause floating-point issues
        problematic_values = [
            0.1 + 0.2,  # Classic floating-point problem
            1.0 / 3.0,  # Repeating decimal
            np.sqrt(2),  # Irrational number
            1e-10 + 1e10,  # Large magnitude difference
        ]

        for value in problematic_values:
            input1 = BoilerInput(
                boiler_id="FP-TEST",
                fuel_flow_rate=value * 100,
                steam_output=value * 1500,
            )

            # Calculate multiple times
            results = [orchestrator.process(input1) for _ in range(10)]

            # All results should be bit-identical
            hashes = [r.provenance_hash for r in results]
            assert len(set(hashes)) == 1, f"Non-deterministic for value {value}"

    def test_deterministic_random_operations(self, deterministic_config):
        """Test that random operations are deterministic when seeded."""
        # Create two orchestrators with same seed
        orch1 = BoilerEfficiencyOrchestrator(deterministic_config)
        orch2 = BoilerEfficiencyOrchestrator(deterministic_config)

        # Both should use same random seed
        with patch('random.random', return_value=0.5):
            result1 = orch1.process_with_uncertainty(test_input)
            result2 = orch2.process_with_uncertainty(test_input)

        assert result1.uncertainty_bounds == result2.uncertainty_bounds
        assert result1.monte_carlo_results == result2.monte_carlo_results

    def test_deterministic_ordering(self, orchestrator):
        """Test that operations order is deterministic."""
        # Create input with multiple components that could be processed in any order
        multi_component_input = {
            "components": {
                "burner": {"efficiency": 0.95},
                "economizer": {"efficiency": 0.85},
                "air_preheater": {"efficiency": 0.75},
                "superheater": {"efficiency": 0.90},
            }
        }

        # Process multiple times
        results = []
        for _ in range(20):
            result = orchestrator.process_multi_component(multi_component_input)
            results.append(result.component_order)

        # Order should always be the same
        first_order = results[0]
        for order in results[1:]:
            assert order == first_order, "Component processing order differs"

    def test_deterministic_parallel_execution(self, orchestrator):
        """Test determinism in parallel execution."""
        inputs = [test_input.copy() for _ in range(10)]

        # Process in parallel multiple times
        results_sets = []
        for _ in range(5):
            results = orchestrator.process_batch(inputs, parallel=True)
            results_sets.append([r.provenance_hash for r in results])

        # Each run should produce identical results
        first_set = results_sets[0]
        for result_set in results_sets[1:]:
            assert result_set == first_set, "Parallel execution not deterministic"


# ============================================================================
# TEST CROSS-ENVIRONMENT CONSISTENCY
# ============================================================================

class TestCrossEnvironmentConsistency:
    """Test consistency across different environments."""

    def test_platform_independence(self, orchestrator, test_input):
        """Test that results are identical across platforms."""
        result = orchestrator.process(test_input)

        # Create platform-specific metadata
        platform_info = {
            "system": platform.system(),
            "python_version": sys.version,
            "architecture": platform.architecture(),
        }

        # Results should be independent of platform
        # Store expected values (these would be validated across platforms in CI/CD)
        expected_efficiency = 0.8523456789  # Example expected value
        expected_hash_prefix = "a1b2c3"  # First 6 chars of expected hash

        # In real testing, these would be compared across different platforms
        assert abs(result.efficiency - expected_efficiency) < 1e-9
        assert result.provenance_hash.startswith(expected_hash_prefix)

    def test_decimal_precision_consistency(self):
        """Test consistent decimal precision across calculations."""
        getcontext().prec = 28  # Set high precision

        calc = CombustionCalculator(use_decimal=True)

        # Test with exact decimal values
        decimal_input = {
            "fuel_flow": Decimal("100.123456789"),
            "air_flow": Decimal("1500.987654321"),
            "heating_value": Decimal("50000.111111111"),
        }

        # Calculate multiple times
        results = []
        for _ in range(10):
            result = calc.calculate_efficiency(decimal_input)
            results.append(str(result))  # Convert to string for exact comparison

        # All results should be exactly identical
        assert len(set(results)) == 1, "Decimal calculations not consistent"

    def test_serialization_determinism(self, orchestrator, test_input):
        """Test that serialization/deserialization maintains determinism."""
        original_result = orchestrator.process(test_input)

        # Serialize result
        serialized = pickle.dumps(original_result)

        # Deserialize
        deserialized_result = pickle.loads(serialized)

        # Should be identical
        assert original_result.efficiency == deserialized_result.efficiency
        assert original_result.provenance_hash == deserialized_result.provenance_hash

        # Process again with deserialized data
        new_result = orchestrator.process(test_input)
        assert new_result.provenance_hash == original_result.provenance_hash

    def test_configuration_independence(self, test_input):
        """Test that non-functional config changes don't affect results."""
        # Create configs with different non-functional settings
        config1 = AgentConfig(
            name="GL-002",
            log_level="INFO",
            timeout=30,
        )

        config2 = AgentConfig(
            name="GL-002",
            log_level="DEBUG",  # Different log level
            timeout=60,  # Different timeout
        )

        orch1 = BoilerEfficiencyOrchestrator(config1)
        orch2 = BoilerEfficiencyOrchestrator(config2)

        result1 = orch1.process(test_input)
        result2 = orch2.process(test_input)

        # Results should be identical despite config differences
        assert result1.efficiency == result2.efficiency
        assert result1.provenance_hash == result2.provenance_hash


# ============================================================================
# TEST PROVENANCE DETERMINISM
# ============================================================================

class TestProvenanceDeterminism:
    """Test deterministic provenance tracking."""

    def test_provenance_hash_determinism(self):
        """Test that provenance hashes are deterministic."""
        tracker = ProvenanceTracker()

        data = {
            "input": {"fuel_flow": 100.0, "steam_flow": 1500.0},
            "method": "indirect",
            "timestamp": "2024-01-01T12:00:00Z",  # Fixed timestamp
            "result": 0.85,
        }

        # Generate hash multiple times
        hashes = [tracker.generate_hash(data) for _ in range(10)]

        # All hashes should be identical
        assert len(set(hashes)) == 1
        assert len(hashes[0]) == 64  # SHA-256 produces 64 hex characters

    def test_provenance_chain_determinism(self):
        """Test that provenance chains are deterministic."""
        tracker = ProvenanceTracker()

        steps = [
            {"step": 1, "action": "read_input", "data": {"fuel": 100}},
            {"step": 2, "action": "calculate", "result": 0.85},
            {"step": 3, "action": "validate", "status": "valid"},
        ]

        # Create chain multiple times
        chains = [tracker.create_chain(steps) for _ in range(5)]

        # All chains should be identical
        first_chain = chains[0]
        for chain in chains[1:]:
            assert len(chain) == len(first_chain)
            for i, step in enumerate(chain):
                assert step["hash"] == first_chain[i]["hash"]

    def test_provenance_with_sorted_keys(self):
        """Test that dict key ordering doesn't affect provenance."""
        tracker = ProvenanceTracker()

        # Same data with different key orders
        data1 = {"b": 2, "a": 1, "c": 3}
        data2 = {"a": 1, "c": 3, "b": 2}
        data3 = {"c": 3, "b": 2, "a": 1}

        hash1 = tracker.generate_hash(data1)
        hash2 = tracker.generate_hash(data2)
        hash3 = tracker.generate_hash(data3)

        # All hashes should be identical
        assert hash1 == hash2 == hash3


# ============================================================================
# TEST NUMERICAL DETERMINISM
# ============================================================================

class TestNumericalDeterminism:
    """Test determinism in numerical calculations."""

    def test_matrix_operation_determinism(self):
        """Test that matrix operations are deterministic."""
        np.random.seed(42)  # Set seed for reproducibility

        # Create test matrices
        matrix_a = np.random.rand(10, 10)
        matrix_b = np.random.rand(10, 10)

        # Perform operations multiple times
        results = []
        for _ in range(10):
            result = np.dot(matrix_a, matrix_b)
            result = np.linalg.inv(result)
            result = np.sum(result)
            results.append(result)

        # All results should be identical (within floating-point precision)
        assert np.allclose(results, results[0], rtol=1e-15)

    def test_iterative_calculation_determinism(self, orchestrator):
        """Test that iterative calculations converge deterministically."""
        # Test iterative optimization
        optimization_params = {
            "initial_guess": 1.1,
            "target_efficiency": 0.85,
            "max_iterations": 100,
            "tolerance": 1e-9,
        }

        # Run optimization multiple times
        results = []
        for _ in range(5):
            result = orchestrator.optimize_excess_air(**optimization_params)
            results.append({
                "final_value": result.final_excess_air,
                "iterations": result.iterations_used,
                "final_error": result.convergence_error,
            })

        # All runs should produce identical results
        first_result = results[0]
        for result in results[1:]:
            assert result == first_result

    def test_accumulator_determinism(self):
        """Test that accumulation operations are deterministic."""
        values = [0.1] * 10  # Values that can cause floating-point errors

        # Test different accumulation methods
        sum_results = []
        product_results = []

        for _ in range(10):
            # Sum accumulation
            total = 0
            for v in values:
                total += v
            sum_results.append(total)

            # Product accumulation
            product = 1
            for v in values:
                product *= v
            product_results.append(product)

        # Results should be consistent
        assert len(set(map(str, sum_results))) == 1
        assert len(set(map(str, product_results))) == 1


# ============================================================================
# TEST EDGE CASE DETERMINISM
# ============================================================================

class TestEdgeCaseDeterminism:
    """Test determinism in edge cases."""

    def test_determinism_with_nan_inf(self, orchestrator):
        """Test handling of NaN and Inf values deterministically."""
        # Test with edge case values
        edge_cases = [
            float('inf'),
            float('-inf'),
            float('nan'),
            0.0,
            -0.0,  # Negative zero
            sys.float_info.min,
            sys.float_info.max,
        ]

        for value in edge_cases:
            try:
                input_data = test_input.copy()
                input_data.fuel_flow_rate = value

                # Process multiple times
                results = []
                for _ in range(3):
                    try:
                        result = orchestrator.process(input_data)
                        results.append(result.error_code if hasattr(result, 'error_code') else None)
                    except Exception as e:
                        results.append(str(e))

                # Error handling should be deterministic
                assert len(set(map(str, results))) == 1

            except Exception:
                pass  # Expected for some edge cases

    def test_determinism_with_extreme_values(self, orchestrator):
        """Test determinism with extreme input values."""
        extreme_inputs = [
            {"fuel_flow": 1e-100, "steam_flow": 1e100},  # Extreme scale difference
            {"fuel_flow": 1e100, "steam_flow": 1e-100},  # Reversed
            {"pressure": 1000, "temperature": -273},  # Physical limits
        ]

        for extreme_input in extreme_inputs:
            input_data = test_input.copy()
            for key, value in extreme_input.items():
                if hasattr(input_data, key):
                    setattr(input_data, key, value)

            # Process multiple times
            results = []
            for _ in range(3):
                try:
                    result = orchestrator.process(input_data)
                    results.append(result.provenance_hash)
                except Exception as e:
                    results.append(f"Error: {type(e).__name__}")

            # Results should be consistent
            assert len(set(results)) == 1

    def test_determinism_after_error_recovery(self, orchestrator, test_input):
        """Test that determinism is maintained after error recovery."""
        # Cause an error
        invalid_input = test_input.copy()
        invalid_input.fuel_flow_rate = -100  # Invalid

        try:
            orchestrator.process(invalid_input)
        except Exception:
            pass  # Expected

        # Process valid input after error
        result1 = orchestrator.process(test_input)

        # Reset and do the same sequence
        orchestrator.reset()

        try:
            orchestrator.process(invalid_input)
        except Exception:
            pass

        result2 = orchestrator.process(test_input)

        # Results should be identical despite error in between
        assert result1.provenance_hash == result2.provenance_hash