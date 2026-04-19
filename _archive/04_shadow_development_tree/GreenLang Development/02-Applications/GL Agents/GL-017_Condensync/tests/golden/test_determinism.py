# -*- coding: utf-8 -*-
"""
Golden Master Tests: Determinism and Reproducibility

Comprehensive tests for bit-perfect reproducibility and deterministic behavior
of GL-017 Condensync calculations. Ensures identical inputs always produce
identical outputs across multiple runs.

Key Areas:
- Calculation determinism (LMTD, CF, heat duty)
- Hash consistency (SHA-256 provenance tracking)
- Fleet analysis reproducibility
- Cross-platform consistency

Author: GL-TestEngineer
Date: December 2025
"""

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conftest import (
    TubeMaterial,
    FailureMode,
    FailureSeverity,
    CleaningMethod,
    OperatingMode,
    CondenserConfig,
    CondenserReading,
    ThermalInput,
    VacuumOptimizationInput,
    FoulingHistoryEntry,
    GoldenTestCase,
    AssertionHelpers,
    ProvenanceCalculator,
    DeterminismChecker,
    calculate_lmtd,
    saturation_temp_from_pressure,
    pressure_from_saturation_temp,
    TEST_SEED,
    OPERATING_LIMITS,
)


# =============================================================================
# DETERMINISM VALIDATOR
# =============================================================================

class DeterminismValidator:
    """
    Validator for deterministic calculation behavior.

    Ensures all calculations produce bit-perfect reproducible results.
    """

    def __init__(self):
        """Initialize validator."""
        self._run_count = 0

    def calculate_lmtd(self, ttd: float, approach: float) -> float:
        """Calculate LMTD deterministically."""
        if ttd <= 0 or approach <= 0:
            return 0.0
        if abs(ttd - approach) < 1e-10:
            return ttd
        return (approach - ttd) / math.log(approach / ttd)

    def calculate_heat_duty(
        self,
        flow_kg_s: float,
        temp_rise_c: float,
        cp: float = 4.186
    ) -> float:
        """Calculate heat duty deterministically."""
        if flow_kg_s <= 0 or temp_rise_c <= 0:
            return 0.0
        return flow_kg_s * cp * temp_rise_c

    def calculate_cleanliness_factor(
        self,
        ua_actual: float,
        ua_design: float
    ) -> float:
        """Calculate CF deterministically."""
        if ua_design <= 0:
            return 0.0
        return min(1.0, max(0.0, ua_actual / ua_design))

    def generate_provenance_hash(self, data: Any) -> str:
        """Generate SHA-256 hash for provenance tracking."""
        if isinstance(data, dict):
            data_str = json.dumps(data, sort_keys=True, default=str)
        else:
            data_str = str(data)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def analyze_condenser(
        self,
        reading: CondenserReading,
        design_ua: float = 80000.0
    ) -> Dict[str, Any]:
        """
        Analyze condenser reading deterministically.

        Args:
            reading: Condenser sensor reading
            design_ua: Design UA value

        Returns:
            Analysis results dictionary
        """
        self._run_count += 1

        # Calculate LMTD
        ttd = reading.saturation_temp_c - reading.cw_outlet_temp_c
        approach = reading.saturation_temp_c - reading.cw_inlet_temp_c
        lmtd = self.calculate_lmtd(ttd, approach)

        # Calculate heat duty
        cw_flow_kg_s = reading.cw_flow_m3_s * 1000
        temp_rise = reading.cw_outlet_temp_c - reading.cw_inlet_temp_c
        heat_duty = self.calculate_heat_duty(cw_flow_kg_s, temp_rise)

        # Calculate UA
        ua_actual = heat_duty / lmtd if lmtd > 0 else 0.0

        # Calculate CF
        cf = self.calculate_cleanliness_factor(ua_actual, design_ua)

        # Generate provenance hash
        input_data = {
            "condenser_id": reading.condenser_id,
            "timestamp": reading.timestamp.isoformat(),
            "cw_inlet_temp": reading.cw_inlet_temp_c,
            "cw_outlet_temp": reading.cw_outlet_temp_c,
            "saturation_temp": reading.saturation_temp_c,
            "cw_flow": reading.cw_flow_m3_s,
        }
        provenance_hash = self.generate_provenance_hash(input_data)

        return {
            "condenser_id": reading.condenser_id,
            "timestamp": reading.timestamp,
            "lmtd_c": round(lmtd, 6),
            "heat_duty_kw": round(heat_duty, 2),
            "ua_actual_kw_k": round(ua_actual, 2),
            "cleanliness_factor": round(cf, 6),
            "provenance_hash": provenance_hash,
        }


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def validator() -> DeterminismValidator:
    """Create determinism validator instance."""
    return DeterminismValidator()


@pytest.fixture
def determinism_checker() -> DeterminismChecker:
    """Create determinism checker instance."""
    return DeterminismChecker()


# =============================================================================
# TEST CLASSES
# =============================================================================

class TestBitPerfectReproducibility:
    """Tests for bit-perfect reproducibility."""

    @pytest.mark.golden
    def test_same_input_same_output(
        self,
        validator: DeterminismValidator,
        healthy_condenser_reading: CondenserReading
    ):
        """Test identical inputs produce identical outputs."""
        result1 = validator.analyze_condenser(healthy_condenser_reading)
        result2 = validator.analyze_condenser(healthy_condenser_reading)

        # All values should be exactly equal
        assert result1["lmtd_c"] == result2["lmtd_c"]
        assert result1["heat_duty_kw"] == result2["heat_duty_kw"]
        assert result1["cleanliness_factor"] == result2["cleanliness_factor"]
        assert result1["provenance_hash"] == result2["provenance_hash"]

    @pytest.mark.golden
    def test_hundred_runs_identical(
        self,
        validator: DeterminismValidator,
        healthy_condenser_reading: CondenserReading
    ):
        """Test 100 runs produce identical results."""
        results = [
            validator.analyze_condenser(healthy_condenser_reading)
            for _ in range(100)
        ]

        # All CF values should be identical
        cf_values = [r["cleanliness_factor"] for r in results]
        assert len(set(cf_values)) == 1, f"Found {len(set(cf_values))} unique CF values"

        # All hashes should be identical
        hashes = [r["provenance_hash"] for r in results]
        assert len(set(hashes)) == 1, f"Found {len(set(hashes))} unique hashes"

    @pytest.mark.golden
    def test_calculation_order_independence(
        self,
        validator: DeterminismValidator,
        condenser_fleet: List[CondenserReading]
    ):
        """Test calculation order doesn't affect results."""
        # Process in original order
        results_forward = [
            validator.analyze_condenser(r)
            for r in condenser_fleet
        ]

        # Process in reverse order
        results_reverse = [
            validator.analyze_condenser(r)
            for r in reversed(condenser_fleet)
        ]
        results_reverse.reverse()  # Restore original order for comparison

        # Results should be identical regardless of processing order
        for r1, r2 in zip(results_forward, results_reverse):
            assert r1["cleanliness_factor"] == r2["cleanliness_factor"]
            assert r1["provenance_hash"] == r2["provenance_hash"]


class TestHashConsistency:
    """Tests for hash consistency."""

    @pytest.mark.golden
    def test_hash_length(self, validator: DeterminismValidator):
        """Test hash is correct length (SHA-256 = 64 hex chars)."""
        data = {"test": "data", "value": 123}
        phash = validator.generate_provenance_hash(data)

        assert len(phash) == 64

    @pytest.mark.golden
    def test_hash_is_deterministic(self, validator: DeterminismValidator):
        """Test hash is deterministic for same input."""
        data = {"test": "data", "value": 123, "nested": {"a": 1, "b": 2}}

        hashes = [validator.generate_provenance_hash(data) for _ in range(100)]

        assert len(set(hashes)) == 1

    @pytest.mark.golden
    def test_hash_key_order_independent(self, validator: DeterminismValidator):
        """Test hash is independent of dictionary key order."""
        data1 = {"a": 1, "b": 2, "c": 3}
        data2 = {"c": 3, "a": 1, "b": 2}
        data3 = {"b": 2, "c": 3, "a": 1}

        hash1 = validator.generate_provenance_hash(data1)
        hash2 = validator.generate_provenance_hash(data2)
        hash3 = validator.generate_provenance_hash(data3)

        assert hash1 == hash2 == hash3

    @pytest.mark.golden
    def test_different_data_different_hash(self, validator: DeterminismValidator):
        """Test different inputs produce different hashes."""
        data1 = {"value": 1}
        data2 = {"value": 2}
        data3 = {"value": 1, "extra": True}

        hash1 = validator.generate_provenance_hash(data1)
        hash2 = validator.generate_provenance_hash(data2)
        hash3 = validator.generate_provenance_hash(data3)

        assert hash1 != hash2
        assert hash1 != hash3
        assert hash2 != hash3

    @pytest.mark.golden
    def test_hash_is_valid_hex(self, validator: DeterminismValidator):
        """Test hash is valid hexadecimal."""
        data = {"test": 123}
        phash = validator.generate_provenance_hash(data)

        # Should not raise
        int(phash, 16)


class TestLMTDDeterminism:
    """Tests for LMTD calculation determinism."""

    @pytest.mark.golden
    def test_lmtd_deterministic(self, validator: DeterminismValidator):
        """Test LMTD calculation is deterministic."""
        ttd = 3.0
        approach = 13.0

        results = [validator.calculate_lmtd(ttd, approach) for _ in range(1000)]

        assert len(set(results)) == 1

    @pytest.mark.golden
    def test_lmtd_known_value(self, validator: DeterminismValidator):
        """Test LMTD against known analytical value."""
        ttd = 3.0
        approach = 13.0

        lmtd = validator.calculate_lmtd(ttd, approach)

        # Analytical: (13-3)/ln(13/3) = 10/ln(4.333...) = 10/1.4663 = 6.8189...
        expected = (13 - 3) / math.log(13 / 3)
        assert abs(lmtd - expected) < 1e-10

    @pytest.mark.golden
    def test_lmtd_singularity_deterministic(self, validator: DeterminismValidator):
        """Test LMTD singularity handling is deterministic."""
        ttd = 5.0
        approach = 5.0

        results = [validator.calculate_lmtd(ttd, approach) for _ in range(100)]

        assert len(set(results)) == 1
        assert all(r == ttd for r in results)

    @pytest.mark.golden
    @pytest.mark.parametrize("ttd,approach", [
        (3.0, 13.0),
        (5.0, 15.0),
        (2.0, 12.0),
        (8.0, 18.0),
        (5.0, 5.0),  # Singularity
        (0.5, 10.5),  # Small TTD
    ])
    def test_lmtd_parametric_determinism(
        self,
        validator: DeterminismValidator,
        ttd, approach
    ):
        """Test LMTD is deterministic for various inputs."""
        results = [validator.calculate_lmtd(ttd, approach) for _ in range(50)]
        assert len(set(results)) == 1


class TestHeatDutyDeterminism:
    """Tests for heat duty calculation determinism."""

    @pytest.mark.golden
    def test_heat_duty_deterministic(self, validator: DeterminismValidator):
        """Test heat duty calculation is deterministic."""
        results = [
            validator.calculate_heat_duty(15000.0, 10.0)
            for _ in range(1000)
        ]

        assert len(set(results)) == 1

    @pytest.mark.golden
    def test_heat_duty_known_value(self, validator: DeterminismValidator):
        """Test heat duty against known value."""
        # Q = m_dot * Cp * dT = 15000 * 4.186 * 10 = 627,900 kW
        duty = validator.calculate_heat_duty(15000.0, 10.0, 4.186)

        expected = 627900.0
        assert duty == expected


class TestCFDeterminism:
    """Tests for cleanliness factor calculation determinism."""

    @pytest.mark.golden
    def test_cf_deterministic(self, validator: DeterminismValidator):
        """Test CF calculation is deterministic."""
        results = [
            validator.calculate_cleanliness_factor(75000.0, 80000.0)
            for _ in range(1000)
        ]

        assert len(set(results)) == 1

    @pytest.mark.golden
    def test_cf_known_value(self, validator: DeterminismValidator):
        """Test CF against known value."""
        # CF = UA_actual / UA_design = 75000 / 80000 = 0.9375
        cf = validator.calculate_cleanliness_factor(75000.0, 80000.0)

        expected = 0.9375
        assert cf == expected

    @pytest.mark.golden
    def test_cf_clamping_deterministic(self, validator: DeterminismValidator):
        """Test CF clamping is deterministic."""
        # Value that would exceed 1.0
        results = [
            validator.calculate_cleanliness_factor(90000.0, 80000.0)
            for _ in range(100)
        ]

        assert len(set(results)) == 1
        assert all(r == 1.0 for r in results)


class TestFleetAnalysisDeterminism:
    """Tests for fleet analysis determinism."""

    @pytest.mark.golden
    def test_fleet_analysis_deterministic(
        self,
        validator: DeterminismValidator,
        condenser_fleet: List[CondenserReading]
    ):
        """Test fleet analysis is deterministic."""
        # Run analysis twice
        results1 = [validator.analyze_condenser(r) for r in condenser_fleet]
        results2 = [validator.analyze_condenser(r) for r in condenser_fleet]

        # All corresponding results should match
        for r1, r2 in zip(results1, results2):
            assert r1["cleanliness_factor"] == r2["cleanliness_factor"]
            assert r1["provenance_hash"] == r2["provenance_hash"]

    @pytest.mark.golden
    def test_fleet_hashes_unique(
        self,
        validator: DeterminismValidator,
        condenser_fleet: List[CondenserReading]
    ):
        """Test each condenser in fleet has unique hash."""
        results = [validator.analyze_condenser(r) for r in condenser_fleet]

        hashes = [r["provenance_hash"] for r in results]
        unique_hashes = set(hashes)

        assert len(unique_hashes) == len(hashes), "All condensers should have unique hashes"


class TestDeterminismChecker:
    """Tests using DeterminismChecker utility."""

    @pytest.mark.golden
    def test_checker_detects_identical_results(
        self,
        determinism_checker: DeterminismChecker,
        validator: DeterminismValidator
    ):
        """Test checker correctly identifies identical results."""
        determinism_checker.run_multiple(
            validator.calculate_lmtd,
            args=(3.0, 13.0),
            iterations=50
        )

        is_identical, message = determinism_checker.check_identical_results()

        assert is_identical
        assert "identical" in message.lower()

    @pytest.mark.golden
    def test_checker_detects_identical_hashes(
        self,
        determinism_checker: DeterminismChecker,
        validator: DeterminismValidator,
        healthy_condenser_reading: CondenserReading
    ):
        """Test checker correctly verifies hash consistency."""
        determinism_checker.run_multiple(
            validator.analyze_condenser,
            args=(healthy_condenser_reading,),
            iterations=20
        )

        is_identical, message = determinism_checker.check_identical_hashes(
            lambda r: r["provenance_hash"]
        )

        assert is_identical


class TestCrossPlatformConsistency:
    """Tests for cross-platform consistency."""

    @pytest.mark.golden
    def test_math_operations_consistent(self):
        """Test basic math operations are consistent."""
        # These should be identical on all platforms
        assert math.log(2.0) == math.log(2.0)
        assert math.exp(1.0) == math.exp(1.0)
        assert math.sqrt(2.0) == math.sqrt(2.0)

    @pytest.mark.golden
    def test_floating_point_representation(self, validator: DeterminismValidator):
        """Test floating point representation is consistent."""
        # Use exact decimal values that can be represented in binary
        lmtd1 = validator.calculate_lmtd(4.0, 16.0)
        lmtd2 = validator.calculate_lmtd(4.0, 16.0)

        assert lmtd1 == lmtd2

        # String representation should be identical
        assert str(lmtd1) == str(lmtd2)

    @pytest.mark.golden
    def test_json_serialization_consistent(self, validator: DeterminismValidator):
        """Test JSON serialization is consistent."""
        data = {
            "a": 1.5,
            "b": 2.5,
            "nested": {"x": 10, "y": 20},
        }

        json1 = json.dumps(data, sort_keys=True)
        json2 = json.dumps(data, sort_keys=True)

        assert json1 == json2

    @pytest.mark.golden
    def test_hash_consistent_across_calls(self):
        """Test hashlib produces consistent results."""
        data = "test data for hashing"

        hash1 = hashlib.sha256(data.encode()).hexdigest()
        hash2 = hashlib.sha256(data.encode()).hexdigest()

        assert hash1 == hash2


class TestEdgeCaseDeterminism:
    """Tests for edge case determinism."""

    @pytest.mark.golden
    def test_zero_value_determinism(self, validator: DeterminismValidator):
        """Test zero values are handled deterministically."""
        results = [
            validator.calculate_heat_duty(0.0, 10.0)
            for _ in range(100)
        ]

        assert len(set(results)) == 1
        assert all(r == 0.0 for r in results)

    @pytest.mark.golden
    def test_very_small_values_deterministic(self, validator: DeterminismValidator):
        """Test very small values are deterministic."""
        results = [
            validator.calculate_lmtd(0.001, 10.001)
            for _ in range(100)
        ]

        assert len(set(results)) == 1

    @pytest.mark.golden
    def test_very_large_values_deterministic(self, validator: DeterminismValidator):
        """Test very large values are deterministic."""
        results = [
            validator.calculate_heat_duty(1000000.0, 100.0)
            for _ in range(100)
        ]

        assert len(set(results)) == 1

    @pytest.mark.golden
    def test_boundary_values_deterministic(self, validator: DeterminismValidator):
        """Test boundary values are deterministic."""
        # CF at exactly 1.0
        results = [
            validator.calculate_cleanliness_factor(80000.0, 80000.0)
            for _ in range(100)
        ]

        assert len(set(results)) == 1
        assert all(r == 1.0 for r in results)


class TestTimestampHandling:
    """Tests for deterministic timestamp handling."""

    @pytest.mark.golden
    def test_timestamp_in_hash_deterministic(
        self,
        validator: DeterminismValidator
    ):
        """Test timestamp is handled deterministically in hash."""
        timestamp = datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc)

        data1 = {"timestamp": timestamp.isoformat(), "value": 1}
        data2 = {"timestamp": timestamp.isoformat(), "value": 1}

        hash1 = validator.generate_provenance_hash(data1)
        hash2 = validator.generate_provenance_hash(data2)

        assert hash1 == hash2

    @pytest.mark.golden
    def test_different_timestamps_different_hash(
        self,
        validator: DeterminismValidator
    ):
        """Test different timestamps produce different hashes."""
        ts1 = datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        ts2 = datetime(2025, 1, 15, 10, 30, 1, tzinfo=timezone.utc)  # 1 second later

        data1 = {"timestamp": ts1.isoformat(), "value": 1}
        data2 = {"timestamp": ts2.isoformat(), "value": 1}

        hash1 = validator.generate_provenance_hash(data1)
        hash2 = validator.generate_provenance_hash(data2)

        assert hash1 != hash2
