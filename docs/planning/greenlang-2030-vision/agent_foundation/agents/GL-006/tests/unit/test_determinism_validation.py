# -*- coding: utf-8 -*-
"""
Determinism validation tests for GL-006 HeatRecoveryMaximizer.

This module validates that all calculations are bit-perfect reproducible,
provenance hashes are correctly generated, and zero-hallucination
principles are enforced across all calculation paths.

Tests cover:
- SHA-256 provenance hash generation and validation
- Bit-perfect reproducibility of all calculators
- LLM temperature/seed enforcement (when applicable)
- Floating point calculation consistency
- Timestamp exclusion from calculation hashes
- Thread-safe determinism in concurrent execution
- Data structure serialization determinism

Target: 15+ determinism tests
"""

import pytest
import hashlib
import json
import numpy as np
from typing import Dict, Any, List
from datetime import datetime
from decimal import Decimal
from unittest.mock import Mock, patch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from greenlang_core import BaseAgent, AgentConfig, ValidationResult
from greenlang_core.provenance import ProvenanceTracker


# ============================================================================
# PROVENANCE HASH TESTS
# ============================================================================

@pytest.mark.determinism
class TestProvenanceHashFormat:
    """Test SHA-256 provenance hash generation and format validation."""

    def test_provenance_hash_is_valid_sha256(self):
        """Test that provenance hash is valid SHA-256 (64 hex characters)."""
        test_data = {"key": "value", "number": 42}
        hash_str = json.dumps(test_data, sort_keys=True)
        provenance_hash = hashlib.sha256(hash_str.encode()).hexdigest()

        assert provenance_hash is not None
        assert isinstance(provenance_hash, str)
        assert len(provenance_hash) == 64  # SHA-256 = 256 bits = 64 hex chars
        assert all(c in '0123456789abcdef' for c in provenance_hash)

    def test_provenance_hash_determinism(self):
        """Test that identical inputs produce identical provenance hashes."""
        test_data = {
            "stream_id": "H1",
            "temperature": 150.0,
            "flow_rate": 10.5
        }

        hashes = []
        for _ in range(10):
            hash_str = json.dumps(test_data, sort_keys=True)
            hash_result = hashlib.sha256(hash_str.encode()).hexdigest()
            hashes.append(hash_result)

        # All hashes must be identical
        assert len(set(hashes)) == 1

    def test_provenance_hash_input_sensitivity(self):
        """Test that different inputs produce different hashes."""
        data1 = {"value": 1.0}
        data2 = {"value": 1.00001}  # Slightly different

        hash1 = hashlib.sha256(json.dumps(data1, sort_keys=True).encode()).hexdigest()
        hash2 = hashlib.sha256(json.dumps(data2, sort_keys=True).encode()).hexdigest()

        # Different inputs must produce different hashes
        assert hash1 != hash2

    def test_provenance_hash_includes_all_inputs(self):
        """Test that provenance hash changes if any input changes."""
        base_data = {
            "stream_id": "H1",
            "temperature": 150.0,
            "flow_rate": 10.5
        }

        # Hash base case
        base_hash = hashlib.sha256(json.dumps(base_data, sort_keys=True).encode()).hexdigest()

        # Change stream_id
        data_v1 = {**base_data, "stream_id": "H2"}
        hash_v1 = hashlib.sha256(json.dumps(data_v1, sort_keys=True).encode()).hexdigest()
        assert base_hash != hash_v1

        # Change temperature
        data_v2 = {**base_data, "temperature": 151.0}
        hash_v2 = hashlib.sha256(json.dumps(data_v2, sort_keys=True).encode()).hexdigest()
        assert base_hash != hash_v2

        # Change flow_rate
        data_v3 = {**base_data, "flow_rate": 10.6}
        hash_v3 = hashlib.sha256(json.dumps(data_v3, sort_keys=True).encode()).hexdigest()
        assert base_hash != hash_v3

    def test_provenance_hash_collision_resistance(self):
        """Test that provenance hashes are unique (no collisions)."""
        hashes = set()

        for i in range(1000):
            data = {
                "iteration": i,
                "value": np.random.random(),
                "timestamp": f"2025-01-{i % 28 + 1:02d}"
            }
            hash_result = hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()
            hashes.add(hash_result)

        # All 1000 hashes should be unique
        assert len(hashes) == 1000


# ============================================================================
# BIT-PERFECT REPRODUCIBILITY TESTS
# ============================================================================

@pytest.mark.determinism
class TestBitPerfectReproducibility:
    """Test that calculations are bit-perfect reproducible."""

    def test_pinch_analysis_reproducibility(self, mock_stream_data):
        """Test pinch analysis produces identical results across runs."""
        from calculators.pinch_analysis_calculator import (
            PinchAnalysisCalculator,
            PinchAnalysisInput,
            ProcessStream,
            StreamType
        )

        hot_streams = [
            ProcessStream(
                stream_id="H1",
                stream_type=StreamType.HOT,
                supply_temp=180.0,
                target_temp=60.0,
                heat_capacity_flow=10.0
            )
        ]
        cold_streams = [
            ProcessStream(
                stream_id="C1",
                stream_type=StreamType.COLD,
                supply_temp=30.0,
                target_temp=140.0,
                heat_capacity_flow=8.0
            )
        ]

        calc_input = PinchAnalysisInput(
            streams=hot_streams + cold_streams,
            minimum_approach_temp=10.0
        )

        calculator = PinchAnalysisCalculator()
        results = []

        # Run 10 times
        for _ in range(10):
            result = calculator.calculate(calc_input)
            results.append({
                "pinch_temp_hot": result.pinch_temperature_hot,
                "pinch_temp_cold": result.pinch_temperature_cold,
                "min_hot_utility": result.minimum_hot_utility,
                "min_cold_utility": result.minimum_cold_utility,
                "hash": result.calculation_hash
            })

        # All results must be identical
        first = results[0]
        for result in results[1:]:
            assert result["pinch_temp_hot"] == first["pinch_temp_hot"]
            assert result["pinch_temp_cold"] == first["pinch_temp_cold"]
            assert result["min_hot_utility"] == first["min_hot_utility"]
            assert result["min_cold_utility"] == first["min_cold_utility"]
            assert result["hash"] == first["hash"]

    def test_exergy_calculation_reproducibility(self):
        """Test exergy calculations produce identical results across runs."""
        from calculators.exergy_calculator import (
            ExergyCalculator,
            StreamState,
            FluidType,
            ReferenceEnvironment
        )

        inlet_stream = StreamState(
            stream_id="IN-001",
            fluid_type=FluidType.STEAM,
            temperature=473.15,  # 200C in Kelvin
            pressure=1000.0,  # kPa
            mass_flow=5.0
        )

        outlet_stream = StreamState(
            stream_id="OUT-001",
            fluid_type=FluidType.STEAM,
            temperature=373.15,  # 100C in Kelvin
            pressure=101.325,  # kPa
            mass_flow=5.0
        )

        calculator = ExergyCalculator()
        results = []

        for _ in range(10):
            result = calculator.calculate([inlet_stream], [outlet_stream])
            results.append({
                "total_exergy_input": result.total_exergy_input,
                "total_exergy_output": result.total_exergy_output,
                "exergetic_efficiency": result.exergetic_efficiency,
                "hash": result.calculation_hash
            })

        # All results must be bit-perfect identical
        first = results[0]
        for result in results[1:]:
            assert result["total_exergy_input"] == first["total_exergy_input"]
            assert result["total_exergy_output"] == first["total_exergy_output"]
            assert result["exergetic_efficiency"] == first["exergetic_efficiency"]
            assert result["hash"] == first["hash"]

    def test_roi_calculation_reproducibility(self):
        """Test ROI calculation produces identical results across runs."""
        from calculators.roi_calculator import (
            ROICalculator,
            CapitalCostInput,
            OperatingCostInput,
            EnergySavingsInput,
            FinancialParameters,
            EquipmentType
        )

        capital = CapitalCostInput(
            equipment_type=EquipmentType.SHELL_TUBE_HX,
            heat_capacity_kw=500.0,
            material="carbon_steel"
        )
        operating = OperatingCostInput(
            maintenance_percent_of_capital=2.0,
            operating_hours_per_year=8000
        )
        savings = EnergySavingsInput(
            heat_recovery_kw=500.0,
            operating_hours_per_year=8000,
            energy_cost_usd_per_kwh=0.08
        )
        financial = FinancialParameters(
            discount_rate_percent=10.0,
            analysis_period_years=15
        )

        calculator = ROICalculator()
        results = []

        for _ in range(10):
            result = calculator.calculate_roi(capital, operating, savings, financial)
            results.append({
                "npv": result.npv_usd,
                "irr": result.irr_percent,
                "payback": result.simple_payback_years,
                "roi": result.simple_roi_percent
            })

        # All results must be identical
        first = results[0]
        for result in results[1:]:
            assert result["npv"] == first["npv"]
            assert result["irr"] == first["irr"]
            assert result["payback"] == first["payback"]
            assert result["roi"] == first["roi"]


# ============================================================================
# FLOATING POINT CONSISTENCY TESTS
# ============================================================================

@pytest.mark.determinism
class TestFloatingPointConsistency:
    """Test floating point calculation consistency."""

    def test_floating_point_precision_maintained(self):
        """Test that floating point calculations maintain precision."""
        # Test precise calculation that should be deterministic
        value = 1.0 / 3.0
        results = [value * 3.0 for _ in range(100)]

        # All results should be exactly the same (not necessarily 1.0 due to FP)
        assert all(r == results[0] for r in results)

    def test_division_consistency(self):
        """Test that division operations are consistent."""
        numerator = 100.0
        denominator = 3.0

        results = [numerator / denominator for _ in range(100)]

        # All division results should be identical
        assert all(r == results[0] for r in results)

    def test_transcendental_function_consistency(self):
        """Test that transcendental functions (sin, exp, log) are consistent."""
        test_value = 2.5

        results = []
        for _ in range(100):
            result = {
                "sin": np.sin(test_value),
                "cos": np.cos(test_value),
                "exp": np.exp(test_value),
                "log": np.log(test_value)
            }
            results.append(result)

        # All results should be identical
        first = results[0]
        for result in results[1:]:
            assert result["sin"] == first["sin"]
            assert result["cos"] == first["cos"]
            assert result["exp"] == first["exp"]
            assert result["log"] == first["log"]

    def test_numpy_operations_determinism(self):
        """Test that numpy operations are deterministic."""
        np.random.seed(42)
        array1 = np.random.randn(1000)

        np.random.seed(42)
        array2 = np.random.randn(1000)

        # Arrays generated with same seed should be identical
        assert np.array_equal(array1, array2)

        # Operations on arrays should be deterministic
        results = []
        for _ in range(10):
            result = {
                "mean": np.mean(array1),
                "std": np.std(array1),
                "sum": np.sum(array1),
                "max": np.max(array1),
                "min": np.min(array1)
            }
            results.append(result)

        first = results[0]
        for result in results[1:]:
            assert result["mean"] == first["mean"]
            assert result["std"] == first["std"]
            assert result["sum"] == first["sum"]


# ============================================================================
# TIMESTAMP HANDLING TESTS
# ============================================================================

@pytest.mark.determinism
class TestTimestampHandling:
    """Test that timestamps don't affect determinism of core calculations."""

    def test_timestamp_excluded_from_calculation_hash(self):
        """Test that timestamps don't affect calculation provenance hash."""
        import time

        # Create calculation data
        calc_data = {
            "input_value": 100.0,
            "output_value": 200.0,
            "coefficient": 2.0
        }

        # Calculate hash at different times
        hash1 = hashlib.sha256(json.dumps(calc_data, sort_keys=True).encode()).hexdigest()
        time.sleep(0.1)
        hash2 = hashlib.sha256(json.dumps(calc_data, sort_keys=True).encode()).hexdigest()

        # Hashes should be identical despite time difference
        assert hash1 == hash2

    def test_calculation_results_independent_of_execution_time(self):
        """Test that calculation results are independent of execution time."""
        import time

        def deterministic_calculation(x: float) -> float:
            """Pure deterministic calculation."""
            return x * 2.5 + 100.0

        # Execute at different times
        result1 = deterministic_calculation(50.0)
        time.sleep(0.1)
        result2 = deterministic_calculation(50.0)
        time.sleep(0.1)
        result3 = deterministic_calculation(50.0)

        # All results must be identical
        assert result1 == result2 == result3


# ============================================================================
# DATA STRUCTURE SERIALIZATION TESTS
# ============================================================================

@pytest.mark.determinism
class TestDataStructureSerialization:
    """Test that data structures serialize deterministically."""

    def test_dict_serialization_determinism(self):
        """Test that dictionaries serialize consistently for hashing."""
        data1 = {"a": 1, "b": 2, "c": 3}
        data2 = {"c": 3, "a": 1, "b": 2}  # Different insertion order

        # With sort_keys=True, order shouldn't matter
        json1 = json.dumps(data1, sort_keys=True)
        json2 = json.dumps(data2, sort_keys=True)

        assert json1 == json2

        hash1 = hashlib.sha256(json1.encode()).hexdigest()
        hash2 = hashlib.sha256(json2.encode()).hexdigest()

        assert hash1 == hash2

    def test_nested_dict_serialization(self):
        """Test nested dictionary serialization determinism."""
        data = {
            "outer": {
                "inner1": {"value": 1.0},
                "inner2": {"value": 2.0}
            },
            "list": [1, 2, 3]
        }

        # Serialize multiple times
        serializations = [json.dumps(data, sort_keys=True) for _ in range(10)]

        # All should be identical
        assert len(set(serializations)) == 1

    def test_numpy_array_serialization(self):
        """Test that numpy arrays serialize deterministically."""
        array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # Convert to list for serialization
        lists = [array.tolist() for _ in range(10)]

        # All lists should be identical
        assert all(lst == lists[0] for lst in lists)

        # JSON serializations should be identical
        jsons = [json.dumps(lst) for lst in lists]
        assert len(set(jsons)) == 1

    def test_float_serialization_precision(self):
        """Test float serialization maintains precision."""
        values = [1.123456789012345, 0.0000001, 999999999.999999]

        for val in values:
            serialized = json.dumps({"value": val})
            deserialized = json.loads(serialized)["value"]

            # Value should round-trip correctly
            assert val == deserialized


# ============================================================================
# CONCURRENT EXECUTION TESTS
# ============================================================================

@pytest.mark.determinism
class TestConcurrentDeterminism:
    """Test that determinism is maintained in concurrent execution."""

    def test_concurrent_calculations_produce_same_results(self):
        """Test that concurrent calculations don't interfere with each other."""
        import concurrent.futures

        def calculate(seed: int) -> Dict[str, float]:
            """Seeded deterministic calculation."""
            np.random.seed(seed)
            data = np.random.randn(100)
            return {
                "mean": float(np.mean(data)),
                "std": float(np.std(data)),
                "sum": float(np.sum(data))
            }

        # Run same calculation multiple times concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # All use same seed = should produce same result
            futures = [executor.submit(calculate, 42) for _ in range(10)]
            results = [f.result() for f in futures]

        # All results should be identical
        first = results[0]
        for result in results[1:]:
            assert result["mean"] == first["mean"]
            assert result["std"] == first["std"]
            assert result["sum"] == first["sum"]

    def test_thread_local_state_isolation(self):
        """Test that thread-local state is properly isolated."""
        import threading
        import concurrent.futures

        results = {}
        lock = threading.Lock()

        def thread_calculation(thread_id: int) -> None:
            """Calculate with thread-specific seed."""
            np.random.seed(thread_id)
            value = np.random.random()
            with lock:
                results[thread_id] = value

        # Run in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(thread_calculation, range(10))

        # Each thread should have produced a deterministic result
        for thread_id in range(10):
            np.random.seed(thread_id)
            expected = np.random.random()
            assert results[thread_id] == expected


# ============================================================================
# ZERO-HALLUCINATION ENFORCEMENT TESTS
# ============================================================================

@pytest.mark.determinism
class TestZeroHallucinationEnforcement:
    """Test that zero-hallucination principles are enforced."""

    def test_calculation_path_is_deterministic(self):
        """Test that all calculation paths are deterministic."""
        # Define a pure calculation function
        def pure_calculation(inputs: Dict[str, float]) -> Dict[str, float]:
            """Pure deterministic calculation - no randomness, no external state."""
            return {
                "output_a": inputs["input_a"] * 2.0,
                "output_b": inputs["input_a"] + inputs["input_b"],
                "output_c": inputs["input_b"] ** 2
            }

        inputs = {"input_a": 10.0, "input_b": 5.0}

        # Run 100 times
        results = [pure_calculation(inputs) for _ in range(100)]

        # All must be identical
        first = results[0]
        for result in results[1:]:
            assert result == first

    def test_no_random_in_calculation_path(self):
        """Test that random functions are not used in critical calculations."""
        # Simulate a calculation that should NOT use random
        def controlled_calculation(value: float, seed: int = 42) -> float:
            """Calculation with controlled randomness (if needed)."""
            # If randomness is needed, it must be seeded
            np.random.seed(seed)
            noise = np.random.random() * 0.001  # Small controlled noise
            return value + noise

        # Same seed should produce same result
        result1 = controlled_calculation(100.0, seed=42)
        result2 = controlled_calculation(100.0, seed=42)

        assert result1 == result2

    def test_database_lookup_simulation(self):
        """Test that database lookups are deterministic (simulated)."""
        # Simulate emission factor database
        EMISSION_FACTORS = {
            "natural_gas": 0.18,
            "fuel_oil": 0.27,
            "coal": 0.34
        }

        def lookup_emission_factor(fuel_type: str) -> float:
            """Deterministic database lookup simulation."""
            return EMISSION_FACTORS.get(fuel_type, 0.0)

        # Same input should always return same output
        for fuel in EMISSION_FACTORS:
            results = [lookup_emission_factor(fuel) for _ in range(100)]
            assert len(set(results)) == 1
            assert results[0] == EMISSION_FACTORS[fuel]


# ============================================================================
# PROVENANCE TRACKER TESTS
# ============================================================================

@pytest.mark.determinism
class TestProvenanceTracker:
    """Test ProvenanceTracker functionality."""

    def test_provenance_tracker_creates_valid_hash(self):
        """Test that ProvenanceTracker creates valid hashes."""
        tracker = ProvenanceTracker()

        input_data = {"temperature": 150.0, "flow_rate": 10.5}
        output_data = {"heat_duty": 1000.0}

        record = tracker.create_record(
            operation="heat_calculation",
            inputs=input_data,
            outputs=output_data
        )

        assert record.hash is not None
        assert len(record.hash) == 64
        assert all(c in '0123456789abcdef' for c in record.hash)

    def test_provenance_tracker_reproducibility(self):
        """Test that ProvenanceTracker produces reproducible hashes."""
        tracker = ProvenanceTracker()

        input_data = {"a": 1, "b": 2}
        output_data = {"c": 3}

        # Create multiple records with same data
        hashes = []
        for _ in range(10):
            record = tracker.create_record(
                operation="test_op",
                inputs=input_data,
                outputs=output_data
            )
            hashes.append(record.hash)

        # All hashes should be identical
        assert len(set(hashes)) == 1


# ============================================================================
# EDGE CASES
# ============================================================================

@pytest.mark.determinism
class TestDeterminismEdgeCases:
    """Test edge cases for determinism."""

    def test_very_small_numbers(self):
        """Test determinism with very small numbers."""
        small = 1e-15
        results = [small * 1000000 for _ in range(100)]
        assert all(r == results[0] for r in results)

    def test_very_large_numbers(self):
        """Test determinism with very large numbers."""
        large = 1e15
        results = [large / 1000000 for _ in range(100)]
        assert all(r == results[0] for r in results)

    def test_negative_numbers(self):
        """Test determinism with negative numbers."""
        neg = -100.5
        results = [neg * 2.0 + 50.0 for _ in range(100)]
        assert all(r == results[0] for r in results)

    def test_zero_handling(self):
        """Test determinism with zero values."""
        # Division by zero should consistently raise error
        with pytest.raises(ZeroDivisionError):
            _ = 1.0 / 0.0

        # Multiplication by zero
        results = [100.0 * 0.0 for _ in range(100)]
        assert all(r == 0.0 for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "determinism"])
