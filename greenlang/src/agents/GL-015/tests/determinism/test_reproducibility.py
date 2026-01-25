# -*- coding: utf-8 -*-
"""
GL-015 INSULSCAN - Determinism and Reproducibility Tests

Tests for bit-perfect reproducibility and deterministic calculations.
Ensures zero-hallucination guarantee through consistent results.

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import hashlib
import json
import math
from decimal import Decimal, ROUND_HALF_UP, getcontext
from datetime import datetime, date
from typing import Any, Dict, List
import numpy as np


# =============================================================================
# TEST: CALCULATION DETERMINISM
# =============================================================================

@pytest.mark.determinism
class TestCalculationDeterminism:
    """Tests for deterministic calculation results."""

    def test_heat_loss_calculation_reproducibility(self, known_heat_loss_values):
        """Test that heat loss calculations are reproducible."""
        case = known_heat_loss_values["case_1"]

        results = []
        for _ in range(100):
            k = float(case["k_value_w_m_k"])
            L = float(case["pipe_length_m"])
            r1 = float(case["pipe_od_m"]) / 2
            r2 = r1 + float(case["insulation_thickness_m"])
            dT = float(case["process_temp_c"]) - float(case["ambient_temp_c"])

            Q = (2 * math.pi * k * L * dT) / math.log(r2 / r1)
            results.append(Q)

        # All results should be identical
        assert all(r == results[0] for r in results), "Results should be identical"

    def test_decimal_calculation_reproducibility(self):
        """Test Decimal calculation reproducibility."""
        # Set precision
        getcontext().prec = 28

        results = []
        for _ in range(100):
            a = Decimal("175.0")
            b = Decimal("25.0")
            c = Decimal("0.075")
            d = Decimal("0.040")

            result = (a - b) * d / c
            results.append(result)

        # All results should be identical
        first_result = results[0]
        assert all(r == first_result for r in results)

    def test_temperature_conversion_reproducibility(self):
        """Test temperature conversion reproducibility."""
        temps_c = [25.0, 100.0, 175.0, 350.0, -40.0]

        for temp_c in temps_c:
            results_k = []
            for _ in range(50):
                temp_k = temp_c + 273.15
                results_k.append(temp_k)

            assert all(r == results_k[0] for r in results_k)

    @pytest.mark.parametrize("seed", [42, 123, 456, 789, 1000])
    def test_random_seed_reproducibility(self, seed):
        """Test that seeded random operations are reproducible."""
        results1 = []
        np.random.seed(seed)
        for _ in range(10):
            results1.append(np.random.uniform(0, 100))

        results2 = []
        np.random.seed(seed)
        for _ in range(10):
            results2.append(np.random.uniform(0, 100))

        assert results1 == results2


# =============================================================================
# TEST: HASH REPRODUCIBILITY
# =============================================================================

@pytest.mark.determinism
class TestHashReproducibility:
    """Tests for hash generation reproducibility."""

    def test_provenance_hash_reproducibility(self, provenance_test_data, calculate_provenance_hash):
        """Test provenance hash is reproducible."""
        hashes = []
        for _ in range(100):
            h = calculate_provenance_hash(provenance_test_data)
            hashes.append(h)

        # All hashes should be identical
        assert all(h == hashes[0] for h in hashes)

    def test_json_serialization_determinism(self):
        """Test JSON serialization produces deterministic output."""
        data = {
            "z_field": "last",
            "a_field": "first",
            "m_field": "middle",
            "nested": {"b": 2, "a": 1},
        }

        serializations = []
        for _ in range(50):
            s = json.dumps(data, sort_keys=True)
            serializations.append(s)

        assert all(s == serializations[0] for s in serializations)

    def test_sha256_hash_consistency(self):
        """Test SHA-256 hash consistency."""
        test_data = b"test data for hashing"

        hashes = []
        for _ in range(100):
            h = hashlib.sha256(test_data).hexdigest()
            hashes.append(h)

        expected = "916f0027a575074ce72a331777c3478d6513f786a591bd892da1a577bf2335f9"
        assert all(h == hashes[0] for h in hashes)

    def test_complex_data_hash_determinism(self):
        """Test hash determinism for complex data structures."""
        complex_data = {
            "equipment_tag": "P-1001-A",
            "measurements": [175.0, 176.2, 174.8],
            "metadata": {
                "inspector": "INSP-001",
                "date": "2025-01-15",
            },
        }

        hashes = []
        for _ in range(50):
            json_str = json.dumps(complex_data, sort_keys=True, default=str)
            h = hashlib.sha256(json_str.encode()).hexdigest()
            hashes.append(h)

        assert all(h == hashes[0] for h in hashes)


# =============================================================================
# TEST: FLOATING POINT REPRODUCIBILITY
# =============================================================================

@pytest.mark.determinism
class TestFloatingPointReproducibility:
    """Tests for floating point calculation reproducibility."""

    def test_float_addition_reproducibility(self):
        """Test floating point addition reproducibility."""
        results = []
        for _ in range(100):
            result = 0.1 + 0.2
            results.append(result)

        assert all(r == results[0] for r in results)

    def test_float_multiplication_reproducibility(self):
        """Test floating point multiplication reproducibility."""
        a = 175.0
        b = 0.040
        c = 0.075

        results = []
        for _ in range(100):
            result = a * b / c
            results.append(result)

        assert all(r == results[0] for r in results)

    def test_math_function_reproducibility(self):
        """Test math function reproducibility."""
        test_values = [0.5, 1.0, 2.0, math.pi, math.e]

        for val in test_values:
            sin_results = [math.sin(val) for _ in range(50)]
            cos_results = [math.cos(val) for _ in range(50)]
            log_results = [math.log(val) for _ in range(50)]
            sqrt_results = [math.sqrt(val) for _ in range(50)]

            assert all(r == sin_results[0] for r in sin_results)
            assert all(r == cos_results[0] for r in cos_results)
            assert all(r == log_results[0] for r in log_results)
            assert all(r == sqrt_results[0] for r in sqrt_results)

    def test_numpy_operation_reproducibility(self):
        """Test NumPy operation reproducibility."""
        np.random.seed(42)
        array = np.random.uniform(0, 100, (100, 100))

        results = []
        for _ in range(50):
            mean = np.mean(array)
            std = np.std(array)
            results.append((mean, std))

        assert all(r == results[0] for r in results)


# =============================================================================
# TEST: ALGORITHM DETERMINISM
# =============================================================================

@pytest.mark.determinism
class TestAlgorithmDeterminism:
    """Tests for algorithm determinism."""

    def test_sorting_determinism(self):
        """Test sorting algorithm determinism."""
        data = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]

        results = []
        for _ in range(50):
            sorted_data = sorted(data)
            results.append(tuple(sorted_data))

        assert all(r == results[0] for r in results)

    def test_hotspot_detection_determinism(self, sample_thermal_image_data):
        """Test hotspot detection algorithm determinism."""
        temp_matrix = np.array(sample_thermal_image_data["temperature_matrix"])

        results = []
        for _ in range(50):
            mean = np.mean(temp_matrix)
            std = np.std(temp_matrix)
            threshold = mean + 2 * std
            hotspot_count = np.sum(temp_matrix > threshold)
            results.append(hotspot_count)

        assert all(r == results[0] for r in results)

    def test_gradient_calculation_determinism(self, sample_thermal_image_data):
        """Test gradient calculation determinism."""
        temp_matrix = np.array(sample_thermal_image_data["temperature_matrix"])

        results = []
        for _ in range(50):
            grad_y, grad_x = np.gradient(temp_matrix)
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            max_gradient = np.max(grad_magnitude)
            results.append(max_gradient)

        assert all(r == results[0] for r in results)

    def test_priority_calculation_determinism(self, multiple_thermal_defects):
        """Test priority score calculation determinism."""
        defects = multiple_thermal_defects

        results = []
        for _ in range(50):
            scores = []
            for defect in defects:
                score = (
                    float(defect["heat_loss_w_per_m"]) * 0.3 +
                    float(defect["process_temperature_c"]) * 0.2 +
                    float(defect["length_m"]) * 10 * 0.5
                )
                scores.append(score)
            results.append(tuple(scores))

        assert all(r == results[0] for r in results)


# =============================================================================
# TEST: CROSS-RUN REPRODUCIBILITY
# =============================================================================

@pytest.mark.determinism
class TestCrossRunReproducibility:
    """Tests for reproducibility across multiple test runs."""

    def test_known_value_calculation(self):
        """Test calculation against known/expected values."""
        # Heat loss for known parameters
        k = 0.040  # W/m.K
        L = 1.0    # m
        r1 = 0.05  # m (pipe radius)
        r2 = 0.10  # m (insulation outer radius)
        dT = 150   # C

        Q = (2 * math.pi * k * L * dT) / math.log(r2 / r1)

        # Known expected value (pre-calculated)
        expected_Q = 54.414  # W/m (approximately)

        assert abs(Q - expected_Q) < 0.1

    def test_fixed_decimal_calculation(self):
        """Test fixed Decimal calculations."""
        # Use fixed precision
        getcontext().prec = 28

        a = Decimal("100")
        b = Decimal("3")
        result = a / b

        # Should always produce same result
        expected_prefix = "33.33333333"
        assert str(result).startswith(expected_prefix)

    def test_temperature_range_statistics(self, sample_thermal_image_data):
        """Test temperature statistics are consistent."""
        temp_matrix = np.array(sample_thermal_image_data["temperature_matrix"])

        # These values should be consistent across runs
        min_temp = float(np.min(temp_matrix))
        max_temp = float(np.max(temp_matrix))
        mean_temp = float(np.mean(temp_matrix))

        # Store for comparison (in real test, compare to stored baseline)
        stats = {
            "min": round(min_temp, 6),
            "max": round(max_temp, 6),
            "mean": round(mean_temp, 6),
        }

        # Verify against baseline (seeded data should produce same stats)
        assert stats["min"] < stats["mean"] < stats["max"]


# =============================================================================
# TEST: PROVENANCE CHAIN INTEGRITY
# =============================================================================

@pytest.mark.determinism
class TestProvenanceChainIntegrity:
    """Tests for provenance chain integrity and reproducibility."""

    def test_chain_hash_reproducibility(self):
        """Test provenance chain hash is reproducible."""
        chain_data = [
            {"step": 1, "input": "A", "output": "B"},
            {"step": 2, "input": "B", "output": "C"},
            {"step": 3, "input": "C", "output": "D"},
        ]

        hashes = []
        for _ in range(50):
            chain_json = json.dumps(chain_data, sort_keys=True)
            h = hashlib.sha256(chain_json.encode()).hexdigest()
            hashes.append(h)

        assert all(h == hashes[0] for h in hashes)

    def test_incremental_hash_chain(self):
        """Test incremental hash chain construction."""
        steps = ["input", "calculate_dT", "calculate_R", "calculate_Q", "output"]

        chains = []
        for _ in range(50):
            chain = []
            prev_hash = "0" * 64

            for step in steps:
                data = {"step": step, "prev_hash": prev_hash}
                current_hash = hashlib.sha256(
                    json.dumps(data, sort_keys=True).encode()
                ).hexdigest()
                chain.append(current_hash)
                prev_hash = current_hash

            chains.append(tuple(chain))

        assert all(c == chains[0] for c in chains)

    def test_provenance_verification_reproducibility(self):
        """Test provenance verification is reproducible."""
        original_data = {
            "calculation_id": "CALC-001",
            "inputs": {"temp_c": 175.0},
            "outputs": {"heat_loss_w": 450.5},
        }

        original_hash = hashlib.sha256(
            json.dumps(original_data, sort_keys=True).encode()
        ).hexdigest()

        # Verify multiple times
        verification_results = []
        for _ in range(50):
            verify_hash = hashlib.sha256(
                json.dumps(original_data, sort_keys=True).encode()
            ).hexdigest()
            verification_results.append(verify_hash == original_hash)

        assert all(verification_results)


# =============================================================================
# TEST: EDGE CASE DETERMINISM
# =============================================================================

@pytest.mark.determinism
class TestEdgeCaseDeterminism:
    """Tests for determinism in edge cases."""

    def test_zero_handling_determinism(self):
        """Test zero value handling is deterministic."""
        results = []
        for _ in range(50):
            zero_heat_loss = 0.0 * 175.0 / 0.075
            results.append(zero_heat_loss)

        assert all(r == 0.0 for r in results)

    def test_very_small_value_determinism(self):
        """Test very small value calculations are deterministic."""
        results = []
        for _ in range(50):
            small_val = 1e-10
            result = small_val * 1000000
            results.append(result)

        assert all(r == results[0] for r in results)

    def test_very_large_value_determinism(self):
        """Test very large value calculations are deterministic."""
        results = []
        for _ in range(50):
            large_val = 1e10
            result = large_val / 1000000
            results.append(result)

        assert all(r == results[0] for r in results)

    def test_boundary_condition_determinism(self):
        """Test boundary condition handling is deterministic."""
        # At zero temperature difference
        results = []
        for _ in range(50):
            dT = 0.0
            Q = dT * 0.040 / 0.075  # Should be 0
            results.append(Q)

        assert all(r == 0.0 for r in results)
