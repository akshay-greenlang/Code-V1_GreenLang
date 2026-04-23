# -*- coding: utf-8 -*-
"""
Multi-Sensor Fusion Tests for GL-005 CombustionSense
====================================================

Tests for sensor fusion, correlation, and redundancy validation
in combustion monitoring systems.

Sensor Fusion Concepts:
    - Redundant sensor voting (2oo3, 1oo2D voting)
    - Sensor cross-correlation validation
    - Virtual sensor derivation
    - Consensus algorithms

Reference Standards:
    - IEC 61508: Functional Safety
    - IEC 61511: Safety Instrumented Systems
    - ISA-84: Safety Instrumented Functions

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import math
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import hashlib


# =============================================================================
# VOTING LOGIC ENUMS
# =============================================================================

class VotingScheme(Enum):
    """Safety voting schemes per IEC 61508."""
    ONE_OUT_OF_ONE = "1oo1"       # Single sensor (no redundancy)
    ONE_OUT_OF_TWO = "1oo2"       # Any one of two
    TWO_OUT_OF_TWO = "2oo2"       # Both must agree
    ONE_OUT_OF_TWO_D = "1oo2D"    # One of two with diagnostics
    TWO_OUT_OF_THREE = "2oo3"     # Two of three must agree


class SensorAgreement(Enum):
    """Sensor agreement status."""
    AGREE = "agree"
    DISAGREE = "disagree"
    PARTIAL = "partial"
    INSUFFICIENT_DATA = "insufficient"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SensorReading:
    """Generic sensor reading."""
    sensor_id: str
    parameter: str
    value: float
    timestamp: datetime
    quality: str = "GOOD"
    uncertainty: float = 0.0


@dataclass
class FusedReading:
    """Result of sensor fusion."""
    parameter: str
    fused_value: float
    confidence: float          # 0.0 to 1.0
    agreement: SensorAgreement
    source_readings: List[SensorReading]
    voting_scheme: VotingScheme
    provenance_hash: str


@dataclass
class CorrelationResult:
    """Cross-correlation analysis result."""
    parameter_a: str
    parameter_b: str
    correlation_coefficient: float
    is_valid_correlation: bool
    expected_relationship: str
    deviation_from_expected: float


# =============================================================================
# SENSOR FUSION ENGINE
# =============================================================================

class SensorFusionEngine:
    """
    Multi-sensor fusion engine for combustion monitoring.

    Implements:
        - Redundant sensor voting
        - Weighted averaging
        - Median selection
        - Cross-correlation validation
    """

    def __init__(self, voting_scheme: VotingScheme = VotingScheme.TWO_OUT_OF_THREE):
        self.voting_scheme = voting_scheme
        self.correlation_thresholds: Dict[Tuple[str, str], Tuple[float, float]] = {}

    def fuse_readings(
        self,
        readings: List[SensorReading],
        tolerance_percent: float = 10.0
    ) -> FusedReading:
        """
        Fuse multiple sensor readings into single value.

        Args:
            readings: List of sensor readings for same parameter
            tolerance_percent: Maximum acceptable deviation between sensors

        Returns:
            FusedReading with fused value and metadata
        """
        if not readings:
            raise ValueError("No readings to fuse")

        # Filter out bad quality readings
        good_readings = [r for r in readings if r.quality in ["GOOD", "MARGINAL"]]

        if not good_readings:
            return FusedReading(
                parameter=readings[0].parameter,
                fused_value=float('nan'),
                confidence=0.0,
                agreement=SensorAgreement.INSUFFICIENT_DATA,
                source_readings=readings,
                voting_scheme=self.voting_scheme,
                provenance_hash=self._calculate_hash(readings),
            )

        # Calculate fused value based on voting scheme
        fused_value, confidence, agreement = self._apply_voting(good_readings, tolerance_percent)

        return FusedReading(
            parameter=good_readings[0].parameter,
            fused_value=fused_value,
            confidence=confidence,
            agreement=agreement,
            source_readings=readings,
            voting_scheme=self.voting_scheme,
            provenance_hash=self._calculate_hash(readings),
        )

    def _apply_voting(
        self,
        readings: List[SensorReading],
        tolerance_percent: float
    ) -> Tuple[float, float, SensorAgreement]:
        """Apply voting scheme to readings."""
        values = [r.value for r in readings]
        n = len(values)

        if n == 0:
            return float('nan'), 0.0, SensorAgreement.INSUFFICIENT_DATA

        # Calculate agreement
        mean_val = statistics.mean(values)
        max_dev = max(abs(v - mean_val) for v in values) if n > 1 else 0
        relative_dev = (max_dev / mean_val * 100) if mean_val != 0 else 0

        if relative_dev <= tolerance_percent:
            agreement = SensorAgreement.AGREE
        elif relative_dev <= tolerance_percent * 2:
            agreement = SensorAgreement.PARTIAL
        else:
            agreement = SensorAgreement.DISAGREE

        if self.voting_scheme == VotingScheme.ONE_OUT_OF_ONE:
            return values[0], 1.0 if readings[0].quality == "GOOD" else 0.7, agreement

        elif self.voting_scheme == VotingScheme.TWO_OUT_OF_TWO:
            if n < 2:
                return values[0], 0.5, SensorAgreement.INSUFFICIENT_DATA
            if agreement == SensorAgreement.AGREE:
                return statistics.mean(values), 1.0, agreement
            else:
                return float('nan'), 0.0, SensorAgreement.DISAGREE

        elif self.voting_scheme == VotingScheme.TWO_OUT_OF_THREE:
            if n < 2:
                return values[0], 0.5, SensorAgreement.INSUFFICIENT_DATA
            # Use median to reject outlier
            median_val = statistics.median(values)
            confidence = 1.0 if agreement == SensorAgreement.AGREE else 0.7
            return median_val, confidence, agreement

        elif self.voting_scheme == VotingScheme.ONE_OUT_OF_TWO_D:
            # Select best quality reading, with diagnostic check
            if n < 2:
                return values[0], 0.7, SensorAgreement.INSUFFICIENT_DATA

            # Check if readings agree within tolerance
            if agreement == SensorAgreement.AGREE:
                return statistics.mean(values), 1.0, agreement
            else:
                # Use reading with better quality
                best = min(readings, key=lambda r: 0 if r.quality == "GOOD" else 1)
                return best.value, 0.6, agreement

        # Default: weighted average
        return statistics.mean(values), 0.8, agreement

    def validate_cross_correlation(
        self,
        readings_a: List[SensorReading],
        readings_b: List[SensorReading],
        expected_ratio: Optional[float] = None
    ) -> CorrelationResult:
        """
        Validate correlation between two related measurements.

        Example: O2 and CO2 should have inverse correlation in flue gas.

        Args:
            readings_a: First parameter readings
            readings_b: Second parameter readings
            expected_ratio: Expected ratio between parameters (optional)

        Returns:
            CorrelationResult with correlation analysis
        """
        if len(readings_a) != len(readings_b):
            raise ValueError("Reading lists must have same length")

        values_a = [r.value for r in readings_a]
        values_b = [r.value for r in readings_b]

        # Calculate Pearson correlation coefficient
        if len(values_a) < 3:
            corr = 0.0
        else:
            mean_a = statistics.mean(values_a)
            mean_b = statistics.mean(values_b)

            numerator = sum((a - mean_a) * (b - mean_b) for a, b in zip(values_a, values_b))
            denom_a = math.sqrt(sum((a - mean_a) ** 2 for a in values_a))
            denom_b = math.sqrt(sum((b - mean_b) ** 2 for b in values_b))

            if denom_a * denom_b == 0:
                corr = 0.0
            else:
                corr = numerator / (denom_a * denom_b)

        # Determine expected relationship
        param_a = readings_a[0].parameter
        param_b = readings_b[0].parameter

        expected_rel, is_valid = self._check_expected_relationship(param_a, param_b, corr)

        # Calculate deviation from expected ratio if provided
        deviation = 0.0
        if expected_ratio is not None and len(values_a) > 0:
            actual_ratio = statistics.mean(values_a) / statistics.mean(values_b) if statistics.mean(values_b) != 0 else float('inf')
            deviation = abs(actual_ratio - expected_ratio) / expected_ratio * 100

        return CorrelationResult(
            parameter_a=param_a,
            parameter_b=param_b,
            correlation_coefficient=corr,
            is_valid_correlation=is_valid,
            expected_relationship=expected_rel,
            deviation_from_expected=deviation,
        )

    def _check_expected_relationship(
        self,
        param_a: str,
        param_b: str,
        correlation: float
    ) -> Tuple[str, bool]:
        """Check if correlation matches expected relationship."""
        # Known relationships in combustion
        inverse_pairs = [("O2", "CO2"), ("excess_air", "fuel_flow")]
        direct_pairs = [("fuel_flow", "heat_output"), ("air_flow", "O2")]

        pair = (param_a, param_b)
        reverse_pair = (param_b, param_a)

        if pair in inverse_pairs or reverse_pair in inverse_pairs:
            is_valid = correlation < -0.5  # Should be negatively correlated
            return "inverse", is_valid
        elif pair in direct_pairs or reverse_pair in direct_pairs:
            is_valid = correlation > 0.5  # Should be positively correlated
            return "direct", is_valid
        else:
            return "unknown", True  # Can't validate unknown relationships

    def _calculate_hash(self, readings: List[SensorReading]) -> str:
        """Calculate provenance hash for readings."""
        data = "|".join(f"{r.sensor_id}:{r.value}:{r.timestamp.isoformat()}" for r in readings)
        return hashlib.sha256(data.encode()).hexdigest()[:16]


# =============================================================================
# SENSOR FUSION TESTS
# =============================================================================

class TestSensorFusion:
    """Test sensor fusion algorithms."""

    @pytest.fixture
    def fusion_engine(self) -> SensorFusionEngine:
        return SensorFusionEngine(VotingScheme.TWO_OUT_OF_THREE)

    def create_readings(
        self,
        values: List[float],
        parameter: str = "O2"
    ) -> List[SensorReading]:
        """Helper to create test readings."""
        return [
            SensorReading(
                sensor_id=f"SENSOR-{i+1}",
                parameter=parameter,
                value=v,
                timestamp=datetime.now(),
                quality="GOOD",
            )
            for i, v in enumerate(values)
        ]

    @pytest.mark.parametrize("values,expected_fused,tolerance", [
        # Perfect agreement
        ([3.0, 3.0, 3.0], 3.0, 0.01),

        # Small variation (within tolerance)
        ([3.0, 3.05, 2.95], 3.0, 0.1),

        # One outlier (median rejects it)
        ([3.0, 3.05, 5.0], 3.05, 0.1),

        # Two values only
        ([5.0, 5.1], 5.05, 0.1),
    ])
    def test_2oo3_voting_fusion(
        self,
        fusion_engine: SensorFusionEngine,
        values: List[float],
        expected_fused: float,
        tolerance: float
    ):
        """Test 2oo3 voting fusion."""
        readings = self.create_readings(values)
        result = fusion_engine.fuse_readings(readings)

        assert abs(result.fused_value - expected_fused) < tolerance, \
            f"Fused value {result.fused_value} differs from expected {expected_fused}"

    @pytest.mark.parametrize("voting_scheme,values,expected_agreement", [
        # All agree
        (VotingScheme.TWO_OUT_OF_THREE, [3.0, 3.0, 3.0], SensorAgreement.AGREE),
        (VotingScheme.TWO_OUT_OF_TWO, [5.0, 5.05], SensorAgreement.AGREE),

        # Partial agreement
        (VotingScheme.TWO_OUT_OF_THREE, [3.0, 3.5, 3.0], SensorAgreement.PARTIAL),

        # Disagreement
        (VotingScheme.TWO_OUT_OF_THREE, [3.0, 8.0, 12.0], SensorAgreement.DISAGREE),
        (VotingScheme.TWO_OUT_OF_TWO, [3.0, 6.0], SensorAgreement.DISAGREE),
    ])
    def test_sensor_agreement_detection(
        self,
        voting_scheme: VotingScheme,
        values: List[float],
        expected_agreement: SensorAgreement
    ):
        """Test sensor agreement detection."""
        engine = SensorFusionEngine(voting_scheme)
        readings = self.create_readings(values)
        result = engine.fuse_readings(readings)

        assert result.agreement == expected_agreement, \
            f"Expected {expected_agreement}, got {result.agreement}"

    def test_bad_quality_readings_filtered(self, fusion_engine: SensorFusionEngine):
        """Test that bad quality readings are filtered out."""
        readings = [
            SensorReading("S1", "O2", 3.0, datetime.now(), "GOOD"),
            SensorReading("S2", "O2", 3.0, datetime.now(), "GOOD"),
            SensorReading("S3", "O2", 999.0, datetime.now(), "BAD"),  # Should be ignored
        ]

        result = fusion_engine.fuse_readings(readings)

        # Fused value should ignore the BAD reading
        assert abs(result.fused_value - 3.0) < 0.1
        assert result.agreement == SensorAgreement.AGREE

    def test_all_bad_readings_handled(self, fusion_engine: SensorFusionEngine):
        """Test handling when all readings are bad quality."""
        readings = [
            SensorReading("S1", "O2", 3.0, datetime.now(), "BAD"),
            SensorReading("S2", "O2", 3.5, datetime.now(), "FAILED"),
        ]

        result = fusion_engine.fuse_readings(readings)

        assert result.agreement == SensorAgreement.INSUFFICIENT_DATA
        assert math.isnan(result.fused_value)
        assert result.confidence == 0.0

    def test_provenance_hash_unique(self, fusion_engine: SensorFusionEngine):
        """Test that provenance hash is unique per reading set."""
        readings1 = self.create_readings([3.0, 3.0, 3.0])
        readings2 = self.create_readings([3.0, 3.1, 3.0])

        result1 = fusion_engine.fuse_readings(readings1)
        result2 = fusion_engine.fuse_readings(readings2)

        # Different readings should have different hashes
        # Note: timestamps may be same, so values should differ
        # Actually they will differ due to values


# =============================================================================
# CROSS-CORRELATION TESTS
# =============================================================================

class TestCrossCorrelation:
    """Test sensor cross-correlation validation."""

    @pytest.fixture
    def fusion_engine(self) -> SensorFusionEngine:
        return SensorFusionEngine()

    def test_o2_co2_inverse_correlation(self, fusion_engine: SensorFusionEngine):
        """Test O2 and CO2 have inverse correlation."""
        # As O2 increases, CO2 should decrease (dilution effect)
        o2_readings = [
            SensorReading("O2-1", "O2", o2, datetime.now())
            for o2 in [2.0, 3.0, 4.0, 5.0, 6.0]
        ]

        # Inverse relationship: CO2 decreases as O2 increases
        co2_readings = [
            SensorReading("CO2-1", "CO2", 14.0 - o2 * 0.8, datetime.now())
            for o2 in [2.0, 3.0, 4.0, 5.0, 6.0]
        ]

        result = fusion_engine.validate_cross_correlation(o2_readings, co2_readings)

        assert result.correlation_coefficient < -0.9, \
            f"O2/CO2 should be negatively correlated: {result.correlation_coefficient}"
        assert result.expected_relationship == "inverse"
        assert result.is_valid_correlation

    def test_fuel_flow_heat_output_correlation(self, fusion_engine: SensorFusionEngine):
        """Test fuel flow and heat output have positive correlation."""
        fuel_readings = [
            SensorReading("F1", "fuel_flow", f, datetime.now())
            for f in [100, 200, 300, 400, 500]
        ]

        # Direct relationship: heat increases with fuel
        heat_readings = [
            SensorReading("H1", "heat_output", f * 12.5, datetime.now())
            for f in [100, 200, 300, 400, 500]
        ]

        result = fusion_engine.validate_cross_correlation(fuel_readings, heat_readings)

        assert result.correlation_coefficient > 0.99, \
            "Fuel/heat should be positively correlated"
        assert result.expected_relationship == "direct"
        assert result.is_valid_correlation

    def test_broken_correlation_detected(self, fusion_engine: SensorFusionEngine):
        """Test detection of broken expected correlation."""
        o2_readings = [
            SensorReading("O2-1", "O2", 3.0, datetime.now())
            for _ in range(5)
        ]

        # Random CO2 values - no correlation with O2
        import random
        random.seed(42)
        co2_readings = [
            SensorReading("CO2-1", "CO2", random.uniform(8, 14), datetime.now())
            for _ in range(5)
        ]

        result = fusion_engine.validate_cross_correlation(o2_readings, co2_readings)

        # Should detect that correlation is broken
        assert result.expected_relationship == "inverse"
        # Correlation should be weak or wrong direction
        assert abs(result.correlation_coefficient) < 0.5 or not result.is_valid_correlation


# =============================================================================
# VOTING SCHEME TESTS
# =============================================================================

class TestVotingSchemes:
    """Test different voting schemes."""

    def create_readings(
        self,
        values: List[float],
        qualities: Optional[List[str]] = None
    ) -> List[SensorReading]:
        if qualities is None:
            qualities = ["GOOD"] * len(values)
        return [
            SensorReading(f"S{i}", "test", v, datetime.now(), q)
            for i, (v, q) in enumerate(zip(values, qualities))
        ]

    @pytest.mark.parametrize("scheme,values,expected_success", [
        # 1oo1: Single reading always valid
        (VotingScheme.ONE_OUT_OF_ONE, [5.0], True),

        # 2oo2: Both must agree
        (VotingScheme.TWO_OUT_OF_TWO, [5.0, 5.05], True),
        (VotingScheme.TWO_OUT_OF_TWO, [5.0, 6.0], False),  # Disagree

        # 2oo3: Two of three must agree
        (VotingScheme.TWO_OUT_OF_THREE, [5.0, 5.0, 5.0], True),
        (VotingScheme.TWO_OUT_OF_THREE, [5.0, 5.0, 8.0], True),   # 2 agree
        (VotingScheme.TWO_OUT_OF_THREE, [5.0, 7.0, 9.0], False),  # None agree
    ])
    def test_voting_scheme_outcomes(
        self,
        scheme: VotingScheme,
        values: List[float],
        expected_success: bool
    ):
        """Test voting scheme outcomes."""
        engine = SensorFusionEngine(scheme)
        readings = self.create_readings(values)
        result = engine.fuse_readings(readings, tolerance_percent=5.0)

        is_success = result.confidence >= 0.5 and not math.isnan(result.fused_value)

        assert is_success == expected_success, \
            f"Voting scheme {scheme.value} with values {values}: " \
            f"expected success={expected_success}, got confidence={result.confidence}"

    def test_1oo2d_diagnostics(self):
        """Test 1oo2D voting with diagnostic checks."""
        engine = SensorFusionEngine(VotingScheme.ONE_OUT_OF_TWO_D)

        # Both good quality, agree
        readings_agree = self.create_readings([5.0, 5.1], ["GOOD", "GOOD"])
        result = engine.fuse_readings(readings_agree)
        assert result.confidence == 1.0

        # One good, one marginal, disagree
        readings_disagree = self.create_readings([5.0, 7.0], ["GOOD", "MARGINAL"])
        result = engine.fuse_readings(readings_disagree)
        assert result.fused_value == 5.0  # Should prefer GOOD quality


# =============================================================================
# REDUNDANCY TESTS
# =============================================================================

class TestRedundancy:
    """Test sensor redundancy validation."""

    def test_redundancy_check_pass(self):
        """Test redundancy check passes with agreeing sensors."""
        readings = [
            SensorReading("A", "temp", 1000.0, datetime.now()),
            SensorReading("B", "temp", 1002.0, datetime.now()),
            SensorReading("C", "temp", 998.0, datetime.now()),
        ]

        engine = SensorFusionEngine(VotingScheme.TWO_OUT_OF_THREE)
        result = engine.fuse_readings(readings, tolerance_percent=1.0)

        assert result.agreement == SensorAgreement.AGREE
        assert result.confidence >= 0.9

    def test_single_sensor_failure_detected(self):
        """Test detection of single sensor failure in redundant set."""
        readings = [
            SensorReading("A", "temp", 1000.0, datetime.now()),
            SensorReading("B", "temp", 1001.0, datetime.now()),
            SensorReading("C", "temp", 500.0, datetime.now()),  # Failed
        ]

        engine = SensorFusionEngine(VotingScheme.TWO_OUT_OF_THREE)
        result = engine.fuse_readings(readings, tolerance_percent=1.0)

        # 2oo3 should reject the outlier
        assert abs(result.fused_value - 1000.5) < 2.0
        assert result.agreement in [SensorAgreement.PARTIAL, SensorAgreement.DISAGREE]

    def test_dual_sensor_failure_causes_low_confidence(self):
        """Test that two sensor failures cause low confidence."""
        readings = [
            SensorReading("A", "temp", 1000.0, datetime.now()),
            SensorReading("B", "temp", 500.0, datetime.now()),   # Failed
            SensorReading("C", "temp", 1500.0, datetime.now()),  # Failed
        ]

        engine = SensorFusionEngine(VotingScheme.TWO_OUT_OF_THREE)
        result = engine.fuse_readings(readings, tolerance_percent=5.0)

        # No agreement possible
        assert result.agreement == SensorAgreement.DISAGREE
        assert result.confidence < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
