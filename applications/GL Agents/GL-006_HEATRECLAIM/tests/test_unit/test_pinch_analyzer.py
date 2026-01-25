"""
GL-006 HEATRECLAIM - Pinch Analysis Calculator Unit Tests

Comprehensive unit tests for the pinch analysis calculator ensuring
deterministic, reproducible calculations with SHA-256 provenance tracking.

Reference:
    - Linnhoff & Hindmarsh, "The Pinch Design Method", 1983
    - ASME PTC 4.4 for heat recovery system testing standards

Test Coverage:
    - Basic pinch point calculation
    - Minimum utility targeting
    - Composite curve generation
    - Grand composite curve
    - Energy balance validation
    - Edge cases and error handling
    - Provenance tracking
    - Golden test cases from literature
"""

import hashlib
import json
import math
import pytest
from datetime import datetime, timezone
from typing import List
from unittest.mock import Mock, patch

# Import modules under test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.schemas import HeatStream, PinchAnalysisResult, CompositePoint
from core.config import StreamType, Phase
from calculators.pinch_analysis import (
    PinchAnalysisCalculator,
    TemperatureInterval,
    calculate_minimum_utilities,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def calculator():
    """Create a pinch analysis calculator instance."""
    return PinchAnalysisCalculator(delta_t_min=10.0, tolerance=1e-6)


@pytest.fixture
def simple_hot_streams() -> List[HeatStream]:
    """Create simple hot stream test data."""
    return [
        HeatStream(
            stream_id="H1",
            stream_name="Hot Stream 1",
            stream_type=StreamType.HOT,
            T_supply_C=180.0,
            T_target_C=80.0,
            m_dot_kg_s=2.0,
            Cp_kJ_kgK=2.5,
        ),
        HeatStream(
            stream_id="H2",
            stream_name="Hot Stream 2",
            stream_type=StreamType.HOT,
            T_supply_C=150.0,
            T_target_C=50.0,
            m_dot_kg_s=3.0,
            Cp_kJ_kgK=2.0,
        ),
    ]


@pytest.fixture
def simple_cold_streams() -> List[HeatStream]:
    """Create simple cold stream test data."""
    return [
        HeatStream(
            stream_id="C1",
            stream_name="Cold Stream 1",
            stream_type=StreamType.COLD,
            T_supply_C=40.0,
            T_target_C=140.0,
            m_dot_kg_s=2.5,
            Cp_kJ_kgK=2.0,
        ),
        HeatStream(
            stream_id="C2",
            stream_name="Cold Stream 2",
            stream_type=StreamType.COLD,
            T_supply_C=60.0,
            T_target_C=160.0,
            m_dot_kg_s=1.5,
            Cp_kJ_kgK=3.0,
        ),
    ]


@pytest.fixture
def balanced_streams() -> tuple:
    """Create balanced hot and cold streams (equal total duty)."""
    hot = [
        HeatStream(
            stream_id="H1",
            stream_type=StreamType.HOT,
            T_supply_C=200.0,
            T_target_C=100.0,
            m_dot_kg_s=2.0,
            Cp_kJ_kgK=2.0,  # Duty = 400 kW
        ),
    ]
    cold = [
        HeatStream(
            stream_id="C1",
            stream_type=StreamType.COLD,
            T_supply_C=50.0,
            T_target_C=150.0,
            m_dot_kg_s=2.0,
            Cp_kJ_kgK=2.0,  # Duty = 400 kW
        ),
    ]
    return hot, cold


@pytest.fixture
def linnhoff_example_streams() -> tuple:
    """
    Classic Linnhoff 4-stream example from literature.

    Reference: Linnhoff & Hindmarsh, "The Pinch Design Method", 1983
    """
    hot_streams = [
        HeatStream(
            stream_id="H1",
            stream_type=StreamType.HOT,
            T_supply_C=175.0,
            T_target_C=45.0,
            m_dot_kg_s=1.0,
            Cp_kJ_kgK=10.0,  # FCp = 10 kW/K
        ),
        HeatStream(
            stream_id="H2",
            stream_type=StreamType.HOT,
            T_supply_C=125.0,
            T_target_C=65.0,
            m_dot_kg_s=2.0,
            Cp_kJ_kgK=20.0,  # FCp = 40 kW/K
        ),
    ]
    cold_streams = [
        HeatStream(
            stream_id="C1",
            stream_type=StreamType.COLD,
            T_supply_C=20.0,
            T_target_C=155.0,
            m_dot_kg_s=1.5,
            Cp_kJ_kgK=13.33,  # FCp = 20 kW/K
        ),
        HeatStream(
            stream_id="C2",
            stream_type=StreamType.COLD,
            T_supply_C=40.0,
            T_target_C=112.0,
            m_dot_kg_s=2.0,
            Cp_kJ_kgK=7.5,  # FCp = 15 kW/K
        ),
    ]
    return hot_streams, cold_streams


# =============================================================================
# BASIC FUNCTIONALITY TESTS
# =============================================================================

class TestPinchAnalysisCalculator:
    """Test suite for PinchAnalysisCalculator basic functionality."""

    @pytest.mark.unit
    @pytest.mark.pinch
    def test_initialization(self, calculator):
        """Test calculator initialization with default parameters."""
        assert calculator.delta_t_min == 10.0
        assert calculator.tolerance == 1e-6
        assert calculator.VERSION == "1.0.0"
        assert calculator.FORMULA_VERSION == "PINCH_PROBLEM_TABLE_v1.0"

    @pytest.mark.unit
    @pytest.mark.pinch
    def test_initialization_custom_params(self):
        """Test calculator initialization with custom parameters."""
        calc = PinchAnalysisCalculator(delta_t_min=5.0, tolerance=1e-8)
        assert calc.delta_t_min == 5.0
        assert calc.tolerance == 1e-8

    @pytest.mark.unit
    @pytest.mark.pinch
    def test_basic_calculation(self, calculator, simple_hot_streams, simple_cold_streams):
        """Test basic pinch analysis calculation."""
        result = calculator.calculate(simple_hot_streams, simple_cold_streams)

        assert isinstance(result, PinchAnalysisResult)
        assert result.pinch_temperature_C > 0
        assert result.minimum_hot_utility_kW >= 0
        assert result.minimum_cold_utility_kW >= 0
        assert result.maximum_heat_recovery_kW >= 0
        assert result.is_valid is True

    @pytest.mark.unit
    @pytest.mark.pinch
    def test_delta_t_min_override(self, calculator, simple_hot_streams, simple_cold_streams):
        """Test that delta_t_min can be overridden per calculation."""
        result_default = calculator.calculate(
            simple_hot_streams, simple_cold_streams
        )
        result_custom = calculator.calculate(
            simple_hot_streams, simple_cold_streams, delta_t_min=20.0
        )

        # Higher delta_t_min should generally result in higher utilities
        assert result_custom.delta_t_min_C == 20.0
        assert result_default.delta_t_min_C == 10.0
        # With larger approach temp, less heat recovery is possible
        assert result_custom.maximum_heat_recovery_kW <= result_default.maximum_heat_recovery_kW

    @pytest.mark.unit
    @pytest.mark.pinch
    def test_stream_ids_recorded(self, calculator, simple_hot_streams, simple_cold_streams):
        """Test that stream IDs are recorded in result."""
        result = calculator.calculate(simple_hot_streams, simple_cold_streams)

        assert "H1" in result.hot_streams
        assert "H2" in result.hot_streams
        assert "C1" in result.cold_streams
        assert "C2" in result.cold_streams


# =============================================================================
# ENERGY BALANCE TESTS
# =============================================================================

class TestEnergyBalance:
    """Test suite for energy balance validation."""

    @pytest.mark.unit
    @pytest.mark.pinch
    def test_energy_balance_satisfied(self, calculator, simple_hot_streams, simple_cold_streams):
        """Test that energy balance is satisfied in the result."""
        result = calculator.calculate(simple_hot_streams, simple_cold_streams)

        # Calculate total duties
        total_hot = sum(s.duty_kW for s in simple_hot_streams)
        total_cold = sum(s.duty_kW for s in simple_cold_streams)

        # Energy balance: Hot + QH = Cold + QC
        lhs = total_hot + result.minimum_hot_utility_kW
        rhs = total_cold + result.minimum_cold_utility_kW

        assert abs(lhs - rhs) < 1.0, f"Energy balance violated: {lhs} != {rhs}"

    @pytest.mark.unit
    @pytest.mark.pinch
    def test_balanced_streams_minimum_utility(self, calculator, balanced_streams):
        """Test that balanced streams minimize utility requirements."""
        hot, cold = balanced_streams
        result = calculator.calculate(hot, cold)

        # With balanced streams and temperature overlap, utilities should be minimal
        # (may not be zero due to delta_t_min constraint)
        total_duty = hot[0].duty_kW

        # Recovery should be significant
        assert result.maximum_heat_recovery_kW > 0
        # At least some recovery should occur
        recovery_fraction = result.maximum_heat_recovery_kW / total_duty
        assert recovery_fraction > 0.5, "Recovery fraction should be significant"


# =============================================================================
# COMPOSITE CURVE TESTS
# =============================================================================

class TestCompositeCurves:
    """Test suite for composite curve generation."""

    @pytest.mark.unit
    @pytest.mark.pinch
    def test_hot_composite_generated(self, calculator, simple_hot_streams, simple_cold_streams):
        """Test that hot composite curve is generated."""
        result = calculator.calculate(simple_hot_streams, simple_cold_streams)

        assert len(result.hot_composite) > 0
        for point in result.hot_composite:
            assert isinstance(point, CompositePoint)
            assert hasattr(point, 'temperature_C')
            assert hasattr(point, 'enthalpy_kW')

    @pytest.mark.unit
    @pytest.mark.pinch
    def test_cold_composite_generated(self, calculator, simple_hot_streams, simple_cold_streams):
        """Test that cold composite curve is generated."""
        result = calculator.calculate(simple_hot_streams, simple_cold_streams)

        assert len(result.cold_composite) > 0
        for point in result.cold_composite:
            assert isinstance(point, CompositePoint)

    @pytest.mark.unit
    @pytest.mark.pinch
    def test_grand_composite_generated(self, calculator, simple_hot_streams, simple_cold_streams):
        """Test that grand composite curve is generated."""
        result = calculator.calculate(simple_hot_streams, simple_cold_streams)

        assert len(result.grand_composite) > 0
        for point in result.grand_composite:
            assert isinstance(point, CompositePoint)

    @pytest.mark.unit
    @pytest.mark.pinch
    def test_composite_enthalpy_monotonic(self, calculator, simple_hot_streams, simple_cold_streams):
        """Test that composite curve enthalpy is monotonically increasing."""
        result = calculator.calculate(simple_hot_streams, simple_cold_streams)

        # Hot composite should have increasing enthalpy
        if len(result.hot_composite) > 1:
            for i in range(len(result.hot_composite) - 1):
                assert result.hot_composite[i].enthalpy_kW <= result.hot_composite[i+1].enthalpy_kW

    @pytest.mark.unit
    @pytest.mark.pinch
    def test_heat_cascade_generated(self, calculator, simple_hot_streams, simple_cold_streams):
        """Test that heat cascade is generated."""
        result = calculator.calculate(simple_hot_streams, simple_cold_streams)

        assert len(result.heat_cascade) > 0
        for entry in result.heat_cascade:
            assert "interval" in entry
            assert "T_hot" in entry
            assert "T_cold" in entry
            assert "heat_deficit" in entry
            assert "cumulative" in entry


# =============================================================================
# VALIDATION TESTS
# =============================================================================

class TestValidation:
    """Test suite for input validation."""

    @pytest.mark.unit
    @pytest.mark.pinch
    def test_empty_hot_streams_raises(self, calculator, simple_cold_streams):
        """Test that empty hot streams raises ValueError."""
        with pytest.raises(ValueError, match="At least one hot stream required"):
            calculator.calculate([], simple_cold_streams)

    @pytest.mark.unit
    @pytest.mark.pinch
    def test_empty_cold_streams_raises(self, calculator, simple_hot_streams):
        """Test that empty cold streams raises ValueError."""
        with pytest.raises(ValueError, match="At least one cold stream required"):
            calculator.calculate(simple_hot_streams, [])

    @pytest.mark.unit
    @pytest.mark.pinch
    def test_invalid_hot_stream_direction(self, calculator, simple_cold_streams):
        """Test that hot stream with wrong temperature direction raises error."""
        invalid_hot = [
            HeatStream(
                stream_id="H_invalid",
                stream_type=StreamType.HOT,
                T_supply_C=80.0,  # Supply < Target for hot stream
                T_target_C=180.0,
                m_dot_kg_s=2.0,
                Cp_kJ_kgK=2.5,
            )
        ]
        with pytest.raises(ValueError, match="Hot stream.*must cool down"):
            calculator.calculate(invalid_hot, simple_cold_streams)

    @pytest.mark.unit
    @pytest.mark.pinch
    def test_invalid_cold_stream_direction(self, calculator, simple_hot_streams):
        """Test that cold stream with wrong temperature direction raises error."""
        invalid_cold = [
            HeatStream(
                stream_id="C_invalid",
                stream_type=StreamType.COLD,
                T_supply_C=140.0,  # Supply > Target for cold stream
                T_target_C=40.0,
                m_dot_kg_s=2.5,
                Cp_kJ_kgK=2.0,
            )
        ]
        with pytest.raises(ValueError, match="Cold stream.*must heat up"):
            calculator.calculate(simple_hot_streams, invalid_cold)


# =============================================================================
# PROVENANCE TRACKING TESTS
# =============================================================================

class TestProvenanceTracking:
    """Test suite for SHA-256 provenance tracking."""

    @pytest.mark.unit
    @pytest.mark.pinch
    def test_input_hash_generated(self, calculator, simple_hot_streams, simple_cold_streams):
        """Test that input hash is generated."""
        result = calculator.calculate(simple_hot_streams, simple_cold_streams)

        assert result.input_hash is not None
        assert len(result.input_hash) == 16  # Truncated hash

    @pytest.mark.unit
    @pytest.mark.pinch
    def test_output_hash_generated(self, calculator, simple_hot_streams, simple_cold_streams):
        """Test that output hash is generated."""
        result = calculator.calculate(simple_hot_streams, simple_cold_streams)

        assert result.output_hash is not None
        assert len(result.output_hash) == 16

    @pytest.mark.unit
    @pytest.mark.pinch
    def test_formula_version_recorded(self, calculator, simple_hot_streams, simple_cold_streams):
        """Test that formula version is recorded."""
        result = calculator.calculate(simple_hot_streams, simple_cold_streams)

        assert result.formula_version == "PINCH_PROBLEM_TABLE_v1.0"

    @pytest.mark.unit
    @pytest.mark.pinch
    def test_deterministic_hash(self, calculator, simple_hot_streams, simple_cold_streams):
        """Test that same inputs produce same hash."""
        result1 = calculator.calculate(simple_hot_streams, simple_cold_streams)
        result2 = calculator.calculate(simple_hot_streams, simple_cold_streams)

        assert result1.input_hash == result2.input_hash
        assert result1.output_hash == result2.output_hash

    @pytest.mark.unit
    @pytest.mark.pinch
    def test_different_inputs_different_hash(self, calculator, simple_hot_streams, simple_cold_streams):
        """Test that different inputs produce different hash."""
        result1 = calculator.calculate(
            simple_hot_streams, simple_cold_streams, delta_t_min=10.0
        )
        result2 = calculator.calculate(
            simple_hot_streams, simple_cold_streams, delta_t_min=20.0
        )

        assert result1.input_hash != result2.input_hash


# =============================================================================
# GOLDEN TEST CASES
# =============================================================================

class TestGoldenCases:
    """Test suite with known expected results from literature."""

    @pytest.mark.unit
    @pytest.mark.pinch
    @pytest.mark.golden
    def test_linnhoff_example(self, linnhoff_example_streams):
        """
        Test against classic Linnhoff 4-stream example.

        Expected results (delta_t_min = 10 C):
        - Pinch temperature around 85-90 C
        - Some hot and cold utility required
        """
        hot, cold = linnhoff_example_streams
        calculator = PinchAnalysisCalculator(delta_t_min=10.0)
        result = calculator.calculate(hot, cold)

        # Verify reasonable pinch temperature
        assert 70.0 <= result.pinch_temperature_C <= 110.0, \
            f"Pinch temperature {result.pinch_temperature_C} out of expected range"

        # Verify energy balance
        total_hot = sum(s.duty_kW for s in hot)
        total_cold = sum(s.duty_kW for s in cold)
        balance = abs(
            (total_hot + result.minimum_hot_utility_kW) -
            (total_cold + result.minimum_cold_utility_kW)
        )
        assert balance < 1.0, f"Energy balance error: {balance} kW"

    @pytest.mark.unit
    @pytest.mark.pinch
    @pytest.mark.golden
    def test_single_stream_pair(self):
        """Test simple single hot-cold stream pair."""
        calculator = PinchAnalysisCalculator(delta_t_min=10.0)

        hot = [
            HeatStream(
                stream_id="H1",
                stream_type=StreamType.HOT,
                T_supply_C=100.0,
                T_target_C=40.0,
                m_dot_kg_s=1.0,
                Cp_kJ_kgK=4.0,  # Duty = 240 kW
            )
        ]
        cold = [
            HeatStream(
                stream_id="C1",
                stream_type=StreamType.COLD,
                T_supply_C=20.0,
                T_target_C=80.0,
                m_dot_kg_s=1.0,
                Cp_kJ_kgK=4.0,  # Duty = 240 kW
            )
        ]

        result = calculator.calculate(hot, cold)

        # With delta_t_min = 10, some utility is needed
        # Hot outlet (40) - cold inlet (20) = 20 > 10, OK at cold end
        # Hot inlet (100) - cold outlet (80) = 20 > 10, OK at hot end
        # Full recovery should be possible with some utility
        assert result.maximum_heat_recovery_kW > 0
        assert result.is_valid is True


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    @pytest.mark.unit
    @pytest.mark.pinch
    def test_very_small_delta_t_min(self, simple_hot_streams, simple_cold_streams):
        """Test with very small delta_t_min."""
        calculator = PinchAnalysisCalculator(delta_t_min=1.0)
        result = calculator.calculate(simple_hot_streams, simple_cold_streams)

        # Smaller delta_t_min allows more heat recovery
        assert result.is_valid is True

    @pytest.mark.unit
    @pytest.mark.pinch
    def test_large_delta_t_min(self, simple_hot_streams, simple_cold_streams):
        """Test with large delta_t_min."""
        calculator = PinchAnalysisCalculator(delta_t_min=50.0)
        result = calculator.calculate(simple_hot_streams, simple_cold_streams)

        # Large delta_t_min limits heat recovery
        assert result.is_valid is True
        # May require more utilities

    @pytest.mark.unit
    @pytest.mark.pinch
    def test_single_hot_stream(self, calculator, simple_cold_streams):
        """Test with single hot stream."""
        single_hot = [
            HeatStream(
                stream_id="H1",
                stream_type=StreamType.HOT,
                T_supply_C=180.0,
                T_target_C=80.0,
                m_dot_kg_s=2.0,
                Cp_kJ_kgK=2.5,
            )
        ]
        result = calculator.calculate(single_hot, simple_cold_streams)
        assert result.is_valid is True

    @pytest.mark.unit
    @pytest.mark.pinch
    def test_single_cold_stream(self, calculator, simple_hot_streams):
        """Test with single cold stream."""
        single_cold = [
            HeatStream(
                stream_id="C1",
                stream_type=StreamType.COLD,
                T_supply_C=40.0,
                T_target_C=140.0,
                m_dot_kg_s=2.5,
                Cp_kJ_kgK=2.0,
            )
        ]
        result = calculator.calculate(simple_hot_streams, single_cold)
        assert result.is_valid is True

    @pytest.mark.unit
    @pytest.mark.pinch
    def test_equal_supply_and_target_temps(self, calculator):
        """Test handling of streams with very small temperature change."""
        hot = [
            HeatStream(
                stream_id="H1",
                stream_type=StreamType.HOT,
                T_supply_C=100.1,  # Very small change
                T_target_C=100.0,
                m_dot_kg_s=2.0,
                Cp_kJ_kgK=2.5,
            )
        ]
        cold = [
            HeatStream(
                stream_id="C1",
                stream_type=StreamType.COLD,
                T_supply_C=40.0,
                T_target_C=40.1,  # Very small change
                m_dot_kg_s=2.5,
                Cp_kJ_kgK=2.0,
            )
        ]

        result = calculator.calculate(hot, cold)
        # Should handle gracefully
        assert result.is_valid is True


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================

class TestConvenienceFunctions:
    """Test suite for convenience functions."""

    @pytest.mark.unit
    @pytest.mark.pinch
    def test_calculate_minimum_utilities(self, simple_hot_streams, simple_cold_streams):
        """Test the convenience function for minimum utility targeting."""
        qh_min, qc_min, pinch_temp = calculate_minimum_utilities(
            simple_hot_streams, simple_cold_streams, delta_t_min=10.0
        )

        assert qh_min >= 0
        assert qc_min >= 0
        assert pinch_temp > 0

    @pytest.mark.unit
    @pytest.mark.pinch
    def test_convenience_function_matches_class(self, simple_hot_streams, simple_cold_streams):
        """Test that convenience function matches class method results."""
        calculator = PinchAnalysisCalculator(delta_t_min=10.0)
        result = calculator.calculate(simple_hot_streams, simple_cold_streams)

        qh_min, qc_min, pinch_temp = calculate_minimum_utilities(
            simple_hot_streams, simple_cold_streams, delta_t_min=10.0
        )

        assert abs(qh_min - result.minimum_hot_utility_kW) < 0.01
        assert abs(qc_min - result.minimum_cold_utility_kW) < 0.01
        assert abs(pinch_temp - result.pinch_temperature_C) < 0.01


# =============================================================================
# TEMPERATURE INTERVAL TESTS
# =============================================================================

class TestTemperatureIntervals:
    """Test suite for temperature interval creation."""

    @pytest.mark.unit
    @pytest.mark.pinch
    def test_intervals_created(self, calculator, simple_hot_streams, simple_cold_streams):
        """Test that temperature intervals are created correctly."""
        intervals = calculator._create_temperature_intervals(
            simple_hot_streams, simple_cold_streams, 10.0
        )

        assert len(intervals) > 0
        for interval in intervals:
            assert isinstance(interval, TemperatureInterval)
            assert interval.delta_T > 0

    @pytest.mark.unit
    @pytest.mark.pinch
    def test_interval_temperatures_descending(self, calculator, simple_hot_streams, simple_cold_streams):
        """Test that interval temperatures are in descending order."""
        intervals = calculator._create_temperature_intervals(
            simple_hot_streams, simple_cold_streams, 10.0
        )

        for i in range(len(intervals) - 1):
            assert intervals[i].T_hot >= intervals[i+1].T_hot


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Test suite for performance benchmarks."""

    @pytest.mark.unit
    @pytest.mark.pinch
    @pytest.mark.benchmark
    def test_calculation_time(self, calculator, simple_hot_streams, simple_cold_streams):
        """Test that calculation completes in reasonable time."""
        import time

        start = time.time()
        for _ in range(100):
            calculator.calculate(simple_hot_streams, simple_cold_streams)
        elapsed = time.time() - start

        # Should complete 100 calculations in under 5 seconds
        assert elapsed < 5.0, f"Too slow: {elapsed:.2f}s for 100 calculations"

    @pytest.mark.unit
    @pytest.mark.pinch
    @pytest.mark.benchmark
    def test_many_streams_performance(self, calculator):
        """Test performance with many streams."""
        import time

        # Create 10 hot and 10 cold streams
        hot_streams = []
        cold_streams = []

        for i in range(10):
            hot_streams.append(HeatStream(
                stream_id=f"H{i}",
                stream_type=StreamType.HOT,
                T_supply_C=200 - i*10,
                T_target_C=100 - i*5,
                m_dot_kg_s=2.0,
                Cp_kJ_kgK=2.5,
            ))
            cold_streams.append(HeatStream(
                stream_id=f"C{i}",
                stream_type=StreamType.COLD,
                T_supply_C=30 + i*5,
                T_target_C=130 + i*10,
                m_dot_kg_s=2.5,
                Cp_kJ_kgK=2.0,
            ))

        start = time.time()
        result = calculator.calculate(hot_streams, cold_streams)
        elapsed = time.time() - start

        # Should complete in under 1 second
        assert elapsed < 1.0, f"Too slow: {elapsed:.2f}s for 20 streams"
        assert result.is_valid is True
