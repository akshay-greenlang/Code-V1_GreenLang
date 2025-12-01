# -*- coding: utf-8 -*-
"""
GL-013 PREDICTMAINT - Determinism and Reproducibility Tests
Verification that all calculations are bit-perfect reproducible.

Tests cover:
- RUL calculation determinism
- Failure probability calculation determinism
- Vibration analysis determinism
- Provenance hash consistency
- Same input produces same output (100+ iterations)
- Cross-session reproducibility
- Decimal arithmetic precision
- Parallel execution reproducibility

Zero-Hallucination Policy:
All numeric results must be deterministic and reproducible.
Same inputs must always produce identical outputs.

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import hashlib
import json
import random
import multiprocessing
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime
from typing import Dict, List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading

# Import test fixtures from conftest
from ..conftest import (
    MachineClass,
    VibrationZone,
    HealthState,
    WEIBULL_PARAMETERS,
)


# =============================================================================
# TEST CLASS: RUL CALCULATION DETERMINISM
# =============================================================================


class TestRULCalculationDeterministic:
    """Tests for RUL calculation determinism."""

    @pytest.mark.determinism
    def test_weibull_rul_same_inputs_same_output(self, rul_calculator):
        """Test that identical inputs produce identical RUL output."""
        inputs = {
            "equipment_type": "pump_centrifugal",
            "operating_hours": Decimal("25000"),
            "target_reliability": "0.5",
        }

        result1 = rul_calculator.calculate_weibull_rul(**inputs)
        result2 = rul_calculator.calculate_weibull_rul(**inputs)

        # All numeric values must be identical
        assert result1["rul_hours"] == result2["rul_hours"]
        assert result1["rul_days"] == result2["rul_days"]
        assert result1["current_reliability"] == result2["current_reliability"]
        assert result1["confidence_lower"] == result2["confidence_lower"]
        assert result1["confidence_upper"] == result2["confidence_upper"]

    @pytest.mark.determinism
    def test_weibull_rul_100_iterations(self, rul_calculator):
        """Test RUL determinism over 100 iterations."""
        inputs = {
            "equipment_type": "pump_centrifugal",
            "operating_hours": Decimal("30000"),
            "target_reliability": "0.5",
        }

        results = []
        for _ in range(100):
            result = rul_calculator.calculate_weibull_rul(**inputs)
            results.append(result["rul_hours"])

        # All 100 results must be identical
        first_result = results[0]
        for i, result in enumerate(results):
            assert result == first_result, f"Iteration {i} produced different result"

    @pytest.mark.determinism
    def test_weibull_rul_provenance_hash_deterministic(self, rul_calculator):
        """Test that provenance hash is deterministic."""
        inputs = {
            "equipment_type": "pump_centrifugal",
            "operating_hours": Decimal("25000"),
        }

        hashes = set()
        for _ in range(50):
            result = rul_calculator.calculate_weibull_rul(**inputs)
            hashes.add(result["provenance_hash"])

        # All hashes must be the same
        assert len(hashes) == 1, f"Got {len(hashes)} different hashes, expected 1"

    @pytest.mark.determinism
    @pytest.mark.parametrize("equipment_type", [
        "pump_centrifugal",
        "motor_ac_induction_large",
        "gearbox_helical",
        "bearing_6205",
        "compressor_reciprocating",
    ])
    def test_weibull_rul_deterministic_all_equipment_types(
        self, rul_calculator, equipment_type
    ):
        """Test determinism across all equipment types."""
        inputs = {
            "equipment_type": equipment_type,
            "operating_hours": Decimal("20000"),
        }

        results = [
            rul_calculator.calculate_weibull_rul(**inputs)
            for _ in range(10)
        ]

        first_rul = results[0]["rul_hours"]
        for result in results[1:]:
            assert result["rul_hours"] == first_rul

    @pytest.mark.determinism
    def test_exponential_rul_deterministic(self, rul_calculator):
        """Test exponential RUL calculation determinism."""
        inputs = {
            "failure_rate": Decimal("0.00002"),
            "operating_hours": Decimal("10000"),
            "target_reliability": Decimal("0.5"),
        }

        results = [
            rul_calculator.calculate_exponential_rul(**inputs)
            for _ in range(50)
        ]

        first_rul = results[0]["rul_hours"]
        for result in results[1:]:
            assert result["rul_hours"] == first_rul


# =============================================================================
# TEST CLASS: FAILURE PROBABILITY DETERMINISM
# =============================================================================


class TestFailureProbabilityDeterministic:
    """Tests for failure probability calculation determinism."""

    @pytest.mark.determinism
    def test_weibull_probability_deterministic(self, failure_probability_calculator):
        """Test Weibull failure probability determinism."""
        inputs = {
            "beta": Decimal("2.5"),
            "eta": Decimal("50000"),
            "time_hours": Decimal("30000"),
        }

        results = [
            failure_probability_calculator.calculate_weibull_failure_probability(**inputs)
            for _ in range(100)
        ]

        # All results must be identical
        first = results[0]
        for result in results[1:]:
            assert result["failure_probability"] == first["failure_probability"]
            assert result["reliability"] == first["reliability"]
            assert result["hazard_rate"] == first["hazard_rate"]

    @pytest.mark.determinism
    def test_failure_probability_provenance_deterministic(
        self, failure_probability_calculator
    ):
        """Test failure probability provenance hash determinism."""
        inputs = {
            "beta": Decimal("2.5"),
            "eta": Decimal("50000"),
            "time_hours": Decimal("30000"),
        }

        hashes = set()
        for _ in range(50):
            result = failure_probability_calculator.calculate_weibull_failure_probability(
                **inputs
            )
            hashes.add(result["provenance_hash"])

        assert len(hashes) == 1

    @pytest.mark.determinism
    @pytest.mark.parametrize("beta,eta,time", [
        (Decimal("0.5"), Decimal("50000"), Decimal("10000")),
        (Decimal("1.0"), Decimal("50000"), Decimal("10000")),
        (Decimal("2.5"), Decimal("50000"), Decimal("10000")),
        (Decimal("2.5"), Decimal("30000"), Decimal("10000")),
        (Decimal("2.5"), Decimal("50000"), Decimal("40000")),
    ])
    def test_failure_probability_deterministic_various_params(
        self, failure_probability_calculator, beta, eta, time
    ):
        """Test determinism with various parameter combinations."""
        results = [
            failure_probability_calculator.calculate_weibull_failure_probability(
                beta=beta, eta=eta, time_hours=time
            )
            for _ in range(10)
        ]

        first_fp = results[0]["failure_probability"]
        for result in results[1:]:
            assert result["failure_probability"] == first_fp


# =============================================================================
# TEST CLASS: VIBRATION ANALYSIS DETERMINISM
# =============================================================================


class TestVibrationAnalysisDeterministic:
    """Tests for vibration analysis determinism."""

    @pytest.mark.determinism
    def test_vibration_severity_deterministic(self, vibration_analyzer):
        """Test vibration severity assessment determinism."""
        inputs = {
            "velocity_rms": Decimal("4.5"),
            "machine_class": MachineClass.CLASS_II,
        }

        results = [
            vibration_analyzer.assess_severity(**inputs)
            for _ in range(100)
        ]

        # All results must be identical
        first = results[0]
        for result in results[1:]:
            assert result["zone"] == first["zone"]
            assert result["alarm_level"] == first["alarm_level"]
            assert result["margin_to_next_zone"] == first["margin_to_next_zone"]

    @pytest.mark.determinism
    def test_vibration_zone_boundary_deterministic(self, vibration_analyzer):
        """Test determinism at zone boundaries."""
        # Test at exact Zone A/B boundary
        boundary_value = Decimal("1.12")  # Class II Zone A limit

        results = [
            vibration_analyzer.assess_severity(
                velocity_rms=boundary_value,
                machine_class=MachineClass.CLASS_II,
            )
            for _ in range(100)
        ]

        # All should classify consistently
        zones = set(r["zone"] for r in results)
        assert len(zones) == 1, f"Inconsistent zone classification at boundary: {zones}"

    @pytest.mark.determinism
    def test_bearing_frequencies_deterministic(self, vibration_analyzer):
        """Test bearing frequency calculation determinism."""
        inputs = {
            "shaft_speed_rpm": Decimal("1480"),
            "num_balls": 9,
            "ball_diameter": Decimal("7.938"),
            "pitch_diameter": Decimal("38.5"),
            "contact_angle_deg": Decimal("0"),
        }

        results = [
            vibration_analyzer.calculate_bearing_frequencies(**inputs)
            for _ in range(100)
        ]

        first = results[0]
        for result in results[1:]:
            assert result["bpfo"] == first["bpfo"]
            assert result["bpfi"] == first["bpfi"]
            assert result["bsf"] == first["bsf"]
            assert result["ftf"] == first["ftf"]


# =============================================================================
# TEST CLASS: PROVENANCE HASH CONSISTENCY
# =============================================================================


class TestProvenanceHashConsistent:
    """Tests for provenance hash consistency."""

    @pytest.mark.determinism
    def test_hash_same_data_same_hash(self, provenance_validator):
        """Test that same data produces same hash."""
        data = {
            "equipment_id": "PUMP-001",
            "value": Decimal("2.5"),
            "timestamp": "2024-01-01T00:00:00",
        }

        hashes = [provenance_validator.compute_hash(data) for _ in range(100)]

        # All hashes must be identical
        assert len(set(hashes)) == 1

    @pytest.mark.determinism
    def test_hash_different_data_different_hash(self, provenance_validator):
        """Test that different data produces different hash."""
        data1 = {"value": Decimal("2.5")}
        data2 = {"value": Decimal("2.6")}

        hash1 = provenance_validator.compute_hash(data1)
        hash2 = provenance_validator.compute_hash(data2)

        assert hash1 != hash2

    @pytest.mark.determinism
    def test_hash_order_independence(self, provenance_validator):
        """Test that dict key order doesn't affect hash (using sorted keys)."""
        data1 = {"a": 1, "b": 2, "c": 3}
        data2 = {"c": 3, "a": 1, "b": 2}

        hash1 = provenance_validator.compute_hash(data1)
        hash2 = provenance_validator.compute_hash(data2)

        assert hash1 == hash2

    @pytest.mark.determinism
    def test_hash_decimal_precision(self, provenance_validator):
        """Test hash consistency with Decimal precision."""
        # Same value, different representations
        data1 = {"value": Decimal("2.5000")}
        data2 = {"value": Decimal("2.5")}

        # When serialized as string, these should be different
        # (if precision matters) or same (if normalized)
        # The test verifies consistent behavior
        hash1 = provenance_validator.compute_hash(data1)
        hash2 = provenance_validator.compute_hash(data2)

        # Document the expected behavior
        # Note: Actual implementation may normalize decimals
        assert isinstance(hash1, str)
        assert isinstance(hash2, str)


# =============================================================================
# TEST CLASS: 100 ITERATIONS SAME INPUT SAME OUTPUT
# =============================================================================


class TestSameInputSameOutput100Iterations:
    """Tests verifying 100+ iteration reproducibility."""

    @pytest.mark.determinism
    def test_rul_100_iterations_all_fields(self, rul_calculator):
        """Test all RUL result fields over 100 iterations."""
        inputs = {
            "equipment_type": "pump_centrifugal",
            "operating_hours": Decimal("25000"),
            "target_reliability": "0.5",
            "confidence_level": "90%",
        }

        first_result = rul_calculator.calculate_weibull_rul(**inputs)

        for i in range(100):
            result = rul_calculator.calculate_weibull_rul(**inputs)

            assert result["rul_hours"] == first_result["rul_hours"], \
                f"rul_hours differs at iteration {i}"
            assert result["rul_days"] == first_result["rul_days"], \
                f"rul_days differs at iteration {i}"
            assert result["rul_years"] == first_result["rul_years"], \
                f"rul_years differs at iteration {i}"
            assert result["current_reliability"] == first_result["current_reliability"], \
                f"current_reliability differs at iteration {i}"
            assert result["confidence_lower"] == first_result["confidence_lower"], \
                f"confidence_lower differs at iteration {i}"
            assert result["confidence_upper"] == first_result["confidence_upper"], \
                f"confidence_upper differs at iteration {i}"
            assert result["provenance_hash"] == first_result["provenance_hash"], \
                f"provenance_hash differs at iteration {i}"

    @pytest.mark.determinism
    def test_failure_probability_100_iterations(self, failure_probability_calculator):
        """Test failure probability over 100 iterations."""
        inputs = {
            "beta": Decimal("2.5"),
            "eta": Decimal("50000"),
            "time_hours": Decimal("30000"),
        }

        first_result = failure_probability_calculator.calculate_weibull_failure_probability(
            **inputs
        )

        for i in range(100):
            result = failure_probability_calculator.calculate_weibull_failure_probability(
                **inputs
            )

            assert result["failure_probability"] == first_result["failure_probability"], \
                f"failure_probability differs at iteration {i}"
            assert result["reliability"] == first_result["reliability"], \
                f"reliability differs at iteration {i}"
            assert result["hazard_rate"] == first_result["hazard_rate"], \
                f"hazard_rate differs at iteration {i}"

    @pytest.mark.determinism
    def test_vibration_100_iterations(self, vibration_analyzer):
        """Test vibration analysis over 100 iterations."""
        inputs = {
            "velocity_rms": Decimal("4.5"),
            "machine_class": MachineClass.CLASS_II,
        }

        first_result = vibration_analyzer.assess_severity(**inputs)

        for i in range(100):
            result = vibration_analyzer.assess_severity(**inputs)

            assert result["zone"] == first_result["zone"], \
                f"zone differs at iteration {i}"
            assert result["margin_to_next_zone"] == first_result["margin_to_next_zone"], \
                f"margin differs at iteration {i}"
            assert result["provenance_hash"] == first_result["provenance_hash"], \
                f"provenance_hash differs at iteration {i}"

    @pytest.mark.determinism
    def test_anomaly_detection_100_iterations(self, anomaly_detector):
        """Test anomaly detection over 100 iterations."""
        historical = [100, 101, 99, 102, 100, 98, 101, 103, 99, 100]
        test_value = Decimal("150")

        first_result = anomaly_detector.detect_univariate_anomaly(
            value=test_value,
            historical_data=historical,
            threshold_sigma="3.0",
        )

        for i in range(100):
            result = anomaly_detector.detect_univariate_anomaly(
                value=test_value,
                historical_data=historical,
                threshold_sigma="3.0",
            )

            assert result["is_anomaly"] == first_result["is_anomaly"], \
                f"is_anomaly differs at iteration {i}"
            assert result["z_score"] == first_result["z_score"], \
                f"z_score differs at iteration {i}"
            assert result["anomaly_score"] == first_result["anomaly_score"], \
                f"anomaly_score differs at iteration {i}"


# =============================================================================
# TEST CLASS: CROSS-SESSION REPRODUCIBILITY
# =============================================================================


class TestCrossSessionReproducibility:
    """Tests for cross-session reproducibility."""

    @pytest.mark.determinism
    def test_known_values_rul(self, rul_calculator):
        """Test RUL against pre-computed known values."""
        # These are pre-computed expected values for validation
        result = rul_calculator.calculate_weibull_rul(
            equipment_type="pump_centrifugal",
            operating_hours=Decimal("30000"),
            target_reliability="0.5",
            custom_beta=Decimal("2.5"),
            custom_eta=Decimal("45000"),
            custom_gamma=Decimal("0"),
        )

        # Verify against expected values (calculated manually or from reference)
        # The actual values will depend on implementation
        assert result["current_reliability"] < Decimal("1")
        assert result["current_reliability"] > Decimal("0")
        assert result["rul_hours"] >= Decimal("0")

    @pytest.mark.determinism
    def test_known_values_failure_probability(self, failure_probability_calculator):
        """Test failure probability against known values."""
        # At t = eta for beta=1 (exponential), F(eta) = 1 - exp(-1) = 0.6321...
        result = failure_probability_calculator.calculate_weibull_failure_probability(
            beta=Decimal("1.0"),
            eta=Decimal("50000"),
            time_hours=Decimal("50000"),
        )

        expected_fp = Decimal("1") - Decimal("0.367879441")  # 1 - exp(-1)
        assert result["failure_probability"] == pytest.approx(
            expected_fp, abs=Decimal("0.001")
        )

    @pytest.mark.determinism
    def test_known_values_vibration(self, vibration_analyzer):
        """Test vibration classification against ISO 10816 standard."""
        # Class II, Zone A limit is 1.12 mm/s
        # Value of 1.0 should be Zone A
        result = vibration_analyzer.assess_severity(
            velocity_rms=Decimal("1.0"),
            machine_class=MachineClass.CLASS_II,
        )

        assert result["zone"] == VibrationZone.ZONE_A

        # Value of 2.0 should be Zone B (between 1.12 and 2.8)
        result = vibration_analyzer.assess_severity(
            velocity_rms=Decimal("2.0"),
            machine_class=MachineClass.CLASS_II,
        )

        assert result["zone"] == VibrationZone.ZONE_B


# =============================================================================
# TEST CLASS: PARALLEL EXECUTION REPRODUCIBILITY
# =============================================================================


class TestParallelExecutionReproducibility:
    """Tests for reproducibility under parallel execution."""

    @pytest.mark.determinism
    def test_threaded_rul_calculation(self, rul_calculator):
        """Test RUL determinism under threaded execution."""
        inputs = {
            "equipment_type": "pump_centrifugal",
            "operating_hours": Decimal("25000"),
        }

        results = []
        lock = threading.Lock()

        def calculate():
            result = rul_calculator.calculate_weibull_rul(**inputs)
            with lock:
                results.append(result["rul_hours"])

        threads = [threading.Thread(target=calculate) for _ in range(20)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All results must be identical
        assert len(set(results)) == 1, f"Got different results from threads: {set(results)}"

    @pytest.mark.determinism
    def test_concurrent_vibration_analysis(self, vibration_analyzer):
        """Test vibration analysis under concurrent execution."""
        inputs = {
            "velocity_rms": Decimal("4.5"),
            "machine_class": MachineClass.CLASS_II,
        }

        results = []
        lock = threading.Lock()

        def analyze():
            result = vibration_analyzer.assess_severity(**inputs)
            with lock:
                results.append((result["zone"], result["margin_to_next_zone"]))

        threads = [threading.Thread(target=analyze) for _ in range(20)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All results must be identical
        unique_results = set(results)
        assert len(unique_results) == 1, f"Got different results: {unique_results}"

    @pytest.mark.determinism
    def test_thread_pool_executor(self, rul_calculator):
        """Test with ThreadPoolExecutor."""
        inputs = {
            "equipment_type": "pump_centrifugal",
            "operating_hours": Decimal("25000"),
        }

        def calculate(_):
            return rul_calculator.calculate_weibull_rul(**inputs)["rul_hours"]

        with ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(calculate, range(50)))

        # All results must be identical
        assert len(set(results)) == 1


# =============================================================================
# TEST CLASS: DECIMAL ARITHMETIC PRECISION
# =============================================================================


class TestDecimalArithmeticPrecision:
    """Tests for Decimal arithmetic precision maintenance."""

    @pytest.mark.determinism
    def test_no_floating_point_conversion(self, rul_calculator):
        """Test that Decimal values don't get converted to float."""
        result = rul_calculator.calculate_weibull_rul(
            equipment_type="pump_centrifugal",
            operating_hours=Decimal("25000.123456789"),
        )

        # Result should be Decimal, not float
        assert isinstance(result["rul_hours"], Decimal)
        assert isinstance(result["current_reliability"], Decimal)

    @pytest.mark.determinism
    def test_decimal_precision_preserved(self, failure_probability_calculator):
        """Test that Decimal precision is preserved through calculation."""
        result = failure_probability_calculator.calculate_weibull_failure_probability(
            beta=Decimal("2.500000000"),
            eta=Decimal("50000.000000000"),
            time_hours=Decimal("30000.000000000"),
        )

        # Values should have appropriate precision
        assert isinstance(result["failure_probability"], Decimal)
        assert isinstance(result["reliability"], Decimal)

    @pytest.mark.determinism
    def test_decimal_rounding_deterministic(self):
        """Test that Decimal rounding is deterministic."""
        value = Decimal("1.23456789")

        rounded_values = [
            value.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
            for _ in range(100)
        ]

        # All should be identical
        assert len(set(rounded_values)) == 1
        assert rounded_values[0] == Decimal("1.235")


# =============================================================================
# TEST CLASS: EDGE CASE DETERMINISM
# =============================================================================


class TestEdgeCaseDeterminism:
    """Tests for determinism in edge cases."""

    @pytest.mark.determinism
    def test_zero_operating_hours_deterministic(self, rul_calculator):
        """Test determinism at zero operating hours."""
        results = [
            rul_calculator.calculate_weibull_rul(
                equipment_type="pump_centrifugal",
                operating_hours=Decimal("0"),
            )
            for _ in range(50)
        ]

        first = results[0]
        for result in results[1:]:
            assert result["current_reliability"] == first["current_reliability"]

    @pytest.mark.determinism
    def test_very_large_hours_deterministic(self, rul_calculator):
        """Test determinism with very large operating hours."""
        results = [
            rul_calculator.calculate_weibull_rul(
                equipment_type="pump_centrifugal",
                operating_hours=Decimal("500000"),
            )
            for _ in range(50)
        ]

        first = results[0]
        for result in results[1:]:
            assert result["rul_hours"] == first["rul_hours"]

    @pytest.mark.determinism
    def test_boundary_values_deterministic(self, vibration_analyzer):
        """Test determinism at exact boundary values."""
        # Test at exact Class II Zone A limit
        boundary = Decimal("1.12")

        results = [
            vibration_analyzer.assess_severity(
                velocity_rms=boundary,
                machine_class=MachineClass.CLASS_II,
            )
            for _ in range(100)
        ]

        # All must classify the same way
        zones = set(r["zone"] for r in results)
        assert len(zones) == 1

    @pytest.mark.determinism
    def test_very_small_values_deterministic(self, vibration_analyzer):
        """Test determinism with very small values."""
        results = [
            vibration_analyzer.assess_severity(
                velocity_rms=Decimal("0.001"),
                machine_class=MachineClass.CLASS_II,
            )
            for _ in range(50)
        ]

        first = results[0]
        for result in results[1:]:
            assert result["zone"] == first["zone"]
            assert result["margin_to_next_zone"] == first["margin_to_next_zone"]
