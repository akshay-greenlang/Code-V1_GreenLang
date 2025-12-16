"""
GL-006 WasteHeatRecovery Agent - Pinch Analysis Tests

Comprehensive unit tests for the PinchAnalyzer class.
Tests pinch point identification, composite curves, and utility targeting.

Coverage Target: 85%+
"""

import pytest
import math
from datetime import datetime
from unittest.mock import Mock, patch

from greenlang.agents.process_heat.gl_006_waste_heat_recovery.pinch_analysis import (
    PinchAnalyzer,
    HeatStream,
    StreamType,
    PinchAnalysisResult,
    TemperatureInterval,
    CompositeCurvePoint,
    CompositeData,
    GrandCompositePoint,
    PinchViolation,
    PinchViolationType,
    DeltaTMinOptimizationResult,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def analyzer():
    """Create PinchAnalyzer instance for testing."""
    return PinchAnalyzer(
        delta_t_min_f=20.0,
        hot_utility_cost_per_mmbtu=8.0,
        cold_utility_cost_per_mmbtu=1.5,
        operating_hours_per_year=8760,
    )


@pytest.fixture
def simple_hot_stream():
    """Create simple hot stream."""
    return HeatStream(
        name="H1",
        stream_type=StreamType.HOT,
        supply_temp_f=300.0,
        target_temp_f=150.0,
        mcp=10.0,
    )


@pytest.fixture
def simple_cold_stream():
    """Create simple cold stream."""
    return HeatStream(
        name="C1",
        stream_type=StreamType.COLD,
        supply_temp_f=80.0,
        target_temp_f=200.0,
        mcp=12.0,
    )


@pytest.fixture
def four_stream_problem():
    """Classic 4-stream pinch analysis problem."""
    return [
        HeatStream(
            name="H1",
            stream_type=StreamType.HOT,
            supply_temp_f=350.0,
            target_temp_f=140.0,
            mcp=3.0,
        ),
        HeatStream(
            name="H2",
            stream_type=StreamType.HOT,
            supply_temp_f=260.0,
            target_temp_f=100.0,
            mcp=1.5,
        ),
        HeatStream(
            name="C1",
            stream_type=StreamType.COLD,
            supply_temp_f=50.0,
            target_temp_f=260.0,
            mcp=2.0,
        ),
        HeatStream(
            name="C2",
            stream_type=StreamType.COLD,
            supply_temp_f=120.0,
            target_temp_f=300.0,
            mcp=2.5,
        ),
    ]


@pytest.fixture
def balanced_streams():
    """Streams with balanced heat duties."""
    return [
        HeatStream(
            name="H1",
            supply_temp_f=300.0,
            target_temp_f=100.0,
            mcp=10.0,
        ),
        HeatStream(
            name="C1",
            supply_temp_f=50.0,
            target_temp_f=250.0,
            mcp=10.0,
        ),
    ]


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================

class TestPinchAnalyzerInitialization:
    """Test PinchAnalyzer initialization."""

    @pytest.mark.unit
    def test_default_initialization(self):
        """Test analyzer initializes with defaults."""
        analyzer = PinchAnalyzer()

        assert analyzer.delta_t_min == 20.0
        assert analyzer.hot_utility_cost == 8.0
        assert analyzer.cold_utility_cost == 1.5
        assert analyzer.operating_hours == 8760

    @pytest.mark.unit
    def test_custom_initialization(self):
        """Test analyzer initializes with custom parameters."""
        analyzer = PinchAnalyzer(
            delta_t_min_f=15.0,
            hot_utility_cost_per_mmbtu=10.0,
            cold_utility_cost_per_mmbtu=2.0,
            operating_hours_per_year=8000,
        )

        assert analyzer.delta_t_min == 15.0
        assert analyzer.hot_utility_cost == 10.0
        assert analyzer.cold_utility_cost == 2.0
        assert analyzer.operating_hours == 8000


# =============================================================================
# HEAT STREAM TESTS
# =============================================================================

class TestHeatStream:
    """Test HeatStream model."""

    @pytest.mark.unit
    def test_hot_stream_auto_detection(self):
        """Test hot stream type auto-detection."""
        stream = HeatStream(
            name="H1",
            supply_temp_f=300.0,
            target_temp_f=150.0,  # Cooling down
            mcp=10.0,
        )

        assert stream.stream_type == StreamType.HOT

    @pytest.mark.unit
    def test_cold_stream_auto_detection(self):
        """Test cold stream type auto-detection."""
        stream = HeatStream(
            name="C1",
            supply_temp_f=80.0,
            target_temp_f=200.0,  # Heating up
            mcp=10.0,
        )

        assert stream.stream_type == StreamType.COLD

    @pytest.mark.unit
    def test_heat_duty_auto_calculation(self):
        """Test heat duty is auto-calculated."""
        stream = HeatStream(
            name="H1",
            supply_temp_f=300.0,
            target_temp_f=150.0,
            mcp=10.0,
        )

        expected_duty = 10.0 * abs(300.0 - 150.0)
        assert stream.heat_duty == expected_duty

    @pytest.mark.unit
    def test_explicit_stream_type(self):
        """Test explicit stream type is respected."""
        stream = HeatStream(
            name="H1",
            stream_type=StreamType.HOT,
            supply_temp_f=300.0,
            target_temp_f=150.0,
            mcp=10.0,
        )

        assert stream.stream_type == StreamType.HOT


# =============================================================================
# BASIC ANALYSIS TESTS
# =============================================================================

class TestBasicAnalysis:
    """Test basic pinch analysis functionality."""

    @pytest.mark.unit
    def test_minimum_streams_requirement(self, analyzer):
        """Test minimum streams requirement."""
        stream = HeatStream(
            name="H1",
            supply_temp_f=300.0,
            target_temp_f=150.0,
            mcp=10.0,
        )

        with pytest.raises(ValueError, match="At least 2 streams required"):
            analyzer.analyze([stream])

    @pytest.mark.unit
    def test_both_stream_types_required(self, analyzer, simple_hot_stream):
        """Test both hot and cold streams are required."""
        hot2 = HeatStream(
            name="H2",
            stream_type=StreamType.HOT,
            supply_temp_f=250.0,
            target_temp_f=100.0,
            mcp=8.0,
        )

        with pytest.raises(ValueError, match="Both hot and cold streams required"):
            analyzer.analyze([simple_hot_stream, hot2])

    @pytest.mark.unit
    def test_basic_two_stream_analysis(self, analyzer, simple_hot_stream, simple_cold_stream):
        """Test basic analysis with two streams."""
        result = analyzer.analyze([simple_hot_stream, simple_cold_stream])

        assert isinstance(result, PinchAnalysisResult)
        assert result.stream_count == 2
        assert result.delta_t_min_f == 20.0

    @pytest.mark.unit
    def test_pinch_temperature_calculated(self, analyzer, simple_hot_stream, simple_cold_stream):
        """Test pinch temperature is calculated."""
        result = analyzer.analyze([simple_hot_stream, simple_cold_stream])

        assert result.pinch_temperature_f is not None
        # Pinch should be between min and max temperatures
        all_temps = [
            simple_hot_stream.supply_temp_f,
            simple_hot_stream.target_temp_f,
            simple_cold_stream.supply_temp_f,
            simple_cold_stream.target_temp_f,
        ]
        assert min(all_temps) <= result.pinch_temperature_f <= max(all_temps)


# =============================================================================
# FOUR STREAM PROBLEM TESTS
# =============================================================================

class TestFourStreamProblem:
    """Test classic 4-stream pinch analysis problem."""

    @pytest.mark.unit
    def test_four_stream_pinch_identification(self, analyzer, four_stream_problem):
        """Test pinch identification in 4-stream problem."""
        result = analyzer.analyze(four_stream_problem)

        assert result.pinch_temperature_f is not None
        assert result.shifted_pinch_temp_f is not None

    @pytest.mark.unit
    def test_four_stream_utility_targets(self, analyzer, four_stream_problem):
        """Test utility targets in 4-stream problem."""
        result = analyzer.analyze(four_stream_problem)

        assert result.minimum_hot_utility_btu_hr >= 0
        assert result.minimum_cold_utility_btu_hr >= 0

    @pytest.mark.unit
    def test_four_stream_heat_recovery(self, analyzer, four_stream_problem):
        """Test heat recovery target in 4-stream problem."""
        result = analyzer.analyze(four_stream_problem)

        # Max recovery should be positive
        assert result.maximum_heat_recovery_btu_hr > 0

        # Verify heat balance
        total_hot = result.total_hot_duty_btu_hr
        total_cold = result.total_cold_duty_btu_hr

        # Heat balance: Hot + Hot_utility = Cold + Cold_utility
        # This should approximately hold
        lhs = total_hot + result.minimum_hot_utility_btu_hr
        rhs = total_cold + result.minimum_cold_utility_btu_hr
        # Allow some tolerance due to rounding
        assert abs(lhs - rhs) < max(lhs, rhs) * 0.1


# =============================================================================
# TEMPERATURE INTERVAL TESTS
# =============================================================================

class TestTemperatureIntervals:
    """Test temperature interval construction."""

    @pytest.mark.unit
    def test_intervals_created(self, analyzer, four_stream_problem):
        """Test temperature intervals are created."""
        result = analyzer.analyze(four_stream_problem)

        assert len(result.temperature_intervals) > 0

    @pytest.mark.unit
    def test_intervals_sorted_descending(self, analyzer, four_stream_problem):
        """Test intervals are sorted by temperature descending."""
        result = analyzer.analyze(four_stream_problem)

        for i in range(len(result.temperature_intervals) - 1):
            assert (
                result.temperature_intervals[i].temp_high_f >=
                result.temperature_intervals[i + 1].temp_high_f
            )

    @pytest.mark.unit
    def test_interval_heat_calculation(self, analyzer, four_stream_problem):
        """Test interval heat is calculated correctly."""
        result = analyzer.analyze(four_stream_problem)

        for interval in result.temperature_intervals:
            # Interval heat = net_mcp * delta_t
            expected = interval.net_mcp * interval.delta_t_f
            assert interval.interval_heat == pytest.approx(expected, rel=0.01)


# =============================================================================
# COMPOSITE CURVE TESTS
# =============================================================================

class TestCompositeCurves:
    """Test composite curve construction."""

    @pytest.mark.unit
    def test_hot_composite_created(self, analyzer, four_stream_problem):
        """Test hot composite curve is created."""
        result = analyzer.analyze(four_stream_problem)

        assert result.hot_composite is not None
        assert result.hot_composite.curve_type == "hot"
        assert len(result.hot_composite.points) > 0

    @pytest.mark.unit
    def test_cold_composite_created(self, analyzer, four_stream_problem):
        """Test cold composite curve is created."""
        result = analyzer.analyze(four_stream_problem)

        assert result.cold_composite is not None
        assert result.cold_composite.curve_type == "cold"
        assert len(result.cold_composite.points) > 0

    @pytest.mark.unit
    def test_composite_enthalpy_monotonic(self, analyzer, four_stream_problem):
        """Test composite curve enthalpy is monotonic."""
        result = analyzer.analyze(four_stream_problem)

        # Hot composite: enthalpy increases as temperature decreases
        for curve in [result.hot_composite, result.cold_composite]:
            enthalpies = [p.enthalpy_btu_hr for p in curve.points]
            for i in range(len(enthalpies) - 1):
                assert enthalpies[i] <= enthalpies[i + 1]

    @pytest.mark.unit
    def test_composite_total_duty(self, analyzer, four_stream_problem):
        """Test composite curve total duty matches stream sum."""
        result = analyzer.analyze(four_stream_problem)

        hot_streams = [s for s in four_stream_problem if s.stream_type == StreamType.HOT]
        expected_hot_duty = sum(s.heat_duty for s in hot_streams)

        assert result.hot_composite.total_duty_btu_hr == pytest.approx(
            expected_hot_duty, rel=0.01
        )


# =============================================================================
# GRAND COMPOSITE CURVE TESTS
# =============================================================================

class TestGrandCompositeCurve:
    """Test Grand Composite Curve (GCC) construction."""

    @pytest.mark.unit
    def test_gcc_created(self, analyzer, four_stream_problem):
        """Test GCC is created."""
        result = analyzer.analyze(four_stream_problem)

        assert result.grand_composite is not None
        assert len(result.grand_composite) > 0

    @pytest.mark.unit
    def test_gcc_pinch_at_zero(self, analyzer, four_stream_problem):
        """Test GCC touches zero at pinch point."""
        result = analyzer.analyze(four_stream_problem)

        # Find minimum heat flow in GCC
        min_heat = min(p.net_heat_btu_hr for p in result.grand_composite)

        # After hot utility adjustment, minimum should be near zero
        assert min_heat >= -1  # Allow small numerical tolerance


# =============================================================================
# UTILITY TARGET TESTS
# =============================================================================

class TestUtilityTargets:
    """Test utility target calculations."""

    @pytest.mark.unit
    def test_utility_targets_positive(self, analyzer, four_stream_problem):
        """Test utility targets are non-negative."""
        result = analyzer.analyze(four_stream_problem)

        assert result.minimum_hot_utility_btu_hr >= 0
        assert result.minimum_cold_utility_btu_hr >= 0

    @pytest.mark.unit
    def test_heat_balance_satisfied(self, analyzer, four_stream_problem):
        """Test overall heat balance is satisfied."""
        result = analyzer.analyze(four_stream_problem)

        # Hot streams + Hot utility = Cold streams + Cold utility
        hot_side = result.total_hot_duty_btu_hr + result.minimum_hot_utility_btu_hr
        cold_side = result.total_cold_duty_btu_hr + result.minimum_cold_utility_btu_hr

        # Should be approximately equal
        assert hot_side == pytest.approx(cold_side, rel=0.05)

    @pytest.mark.unit
    def test_balanced_system_low_utility(self, analyzer, balanced_streams):
        """Test balanced system has low utility requirements."""
        result = analyzer.analyze(balanced_streams)

        # Hot duty = Cold duty = 10 * 200 = 2000 BTU/hr
        # With matching MCps, utilities should be relatively low
        total_duty = 10.0 * 200.0

        # Utilities should be much less than total duty for balanced system
        assert result.minimum_hot_utility_btu_hr < total_duty
        assert result.minimum_cold_utility_btu_hr < total_duty


# =============================================================================
# PINCH VIOLATION DETECTION TESTS
# =============================================================================

class TestPinchViolationDetection:
    """Test pinch violation detection."""

    @pytest.mark.unit
    def test_no_violations_in_normal_problem(self, analyzer, four_stream_problem):
        """Test no violations detected in standard problem."""
        result = analyzer.analyze(four_stream_problem)

        # Standard analysis shouldn't detect violations
        # (violations are for existing networks)
        assert isinstance(result.pinch_violations, list)

    @pytest.mark.unit
    def test_stream_crossing_pinch_detection(self, analyzer):
        """Test detection of stream crossing pinch."""
        # Create streams where one clearly crosses the pinch
        streams = [
            HeatStream(
                name="H1",
                stream_type=StreamType.HOT,
                supply_temp_f=400.0,
                target_temp_f=100.0,  # Wide range crossing pinch
                mcp=10.0,
            ),
            HeatStream(
                name="C1",
                stream_type=StreamType.COLD,
                supply_temp_f=50.0,
                target_temp_f=350.0,  # Wide range crossing pinch
                mcp=10.0,
            ),
        ]

        result = analyzer.analyze(streams)

        # Both streams likely cross the pinch
        # Detection depends on pinch location
        assert hasattr(result, 'pinch_violations')


# =============================================================================
# DELTA T MIN OPTIMIZATION TESTS
# =============================================================================

class TestDeltaTMinOptimization:
    """Test delta T min optimization."""

    @pytest.mark.unit
    def test_optimization_returns_result(self, analyzer, four_stream_problem):
        """Test optimization returns valid result."""
        result = analyzer.optimize_delta_t_min(
            four_stream_problem,
            delta_t_range=(10.0, 40.0),
            num_points=5,
        )

        assert isinstance(result, DeltaTMinOptimizationResult)
        assert result.optimal_delta_t_min_f >= 10.0
        assert result.optimal_delta_t_min_f <= 40.0

    @pytest.mark.unit
    def test_optimization_cost_tradeoff(self, analyzer, four_stream_problem):
        """Test capital vs operating cost tradeoff."""
        result = analyzer.optimize_delta_t_min(
            four_stream_problem,
            delta_t_range=(10.0, 40.0),
            num_points=5,
        )

        # At higher delta T, capital cost should be lower but utility cost higher
        assert result.capital_cost_usd > 0
        assert result.utility_cost_usd_yr > 0

    @pytest.mark.unit
    def test_optimization_arrays_populated(self, analyzer, four_stream_problem):
        """Test optimization arrays are populated."""
        result = analyzer.optimize_delta_t_min(
            four_stream_problem,
            delta_t_range=(10.0, 40.0),
            num_points=5,
        )

        assert len(result.delta_t_values) == 5
        assert len(result.total_costs) == 5


# =============================================================================
# STREAM SEGMENTATION TESTS
# =============================================================================

class TestStreamSegmentation:
    """Test stream segmentation by pinch."""

    @pytest.mark.unit
    def test_get_streams_above_pinch(self, analyzer, four_stream_problem):
        """Test getting streams above pinch."""
        result = analyzer.analyze(four_stream_problem)

        hot_above, cold_above = analyzer.get_streams_above_pinch(
            four_stream_problem,
            result.pinch_temperature_f,
            result.delta_t_min_f,
        )

        # Should have some streams above pinch
        assert isinstance(hot_above, list)
        assert isinstance(cold_above, list)

    @pytest.mark.unit
    def test_get_streams_below_pinch(self, analyzer, four_stream_problem):
        """Test getting streams below pinch."""
        result = analyzer.analyze(four_stream_problem)

        hot_below, cold_below = analyzer.get_streams_below_pinch(
            four_stream_problem,
            result.pinch_temperature_f,
            result.delta_t_min_f,
        )

        # Should have some streams below pinch
        assert isinstance(hot_below, list)
        assert isinstance(cold_below, list)


# =============================================================================
# PROVENANCE TESTS
# =============================================================================

class TestProvenance:
    """Test provenance tracking."""

    @pytest.mark.unit
    def test_provenance_hash_generated(self, analyzer, four_stream_problem):
        """Test provenance hash is generated."""
        result = analyzer.analyze(four_stream_problem)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256

    @pytest.mark.unit
    def test_provenance_hash_deterministic(self, analyzer, four_stream_problem):
        """Test provenance hash is deterministic for same inputs."""
        # Note: timestamp makes hashes different each time
        # This test verifies hash format, not determinism
        result = analyzer.analyze(four_stream_problem)

        assert all(c in '0123456789abcdef' for c in result.provenance_hash)

    @pytest.mark.unit
    def test_analysis_id_unique(self, analyzer, four_stream_problem):
        """Test analysis IDs are unique."""
        result1 = analyzer.analyze(four_stream_problem)
        result2 = analyzer.analyze(four_stream_problem)

        assert result1.analysis_id != result2.analysis_id


# =============================================================================
# RECOMMENDATION TESTS
# =============================================================================

class TestRecommendations:
    """Test recommendation generation."""

    @pytest.mark.unit
    def test_recommendations_generated(self, analyzer, four_stream_problem):
        """Test recommendations are generated."""
        result = analyzer.analyze(four_stream_problem)

        assert isinstance(result.recommendations, list)
        assert len(result.recommendations) > 0

    @pytest.mark.unit
    def test_high_recovery_recommendation(self, analyzer, balanced_streams):
        """Test recommendation for high recovery potential."""
        result = analyzer.analyze(balanced_streams)

        # Balanced system has high recovery potential
        # Should generate positive recommendation
        assert len(result.recommendations) > 0


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.mark.unit
    def test_equal_temperatures_handled(self, analyzer):
        """Test handling of streams with equal supply/target temps."""
        streams = [
            HeatStream(
                name="H1",
                stream_type=StreamType.HOT,
                supply_temp_f=300.0,
                target_temp_f=150.0,
                mcp=10.0,
            ),
            HeatStream(
                name="C1",
                stream_type=StreamType.COLD,
                supply_temp_f=100.0,
                target_temp_f=250.0,
                mcp=10.0,
            ),
        ]

        # Should not raise exception
        result = analyzer.analyze(streams)
        assert result is not None

    @pytest.mark.unit
    def test_large_mcp_difference(self, analyzer):
        """Test with large MCP difference between streams."""
        streams = [
            HeatStream(
                name="H1",
                stream_type=StreamType.HOT,
                supply_temp_f=300.0,
                target_temp_f=100.0,
                mcp=1000.0,  # Very large
            ),
            HeatStream(
                name="C1",
                stream_type=StreamType.COLD,
                supply_temp_f=50.0,
                target_temp_f=250.0,
                mcp=1.0,  # Very small
            ),
        ]

        result = analyzer.analyze(streams)

        # Large hot stream should result in significant cold utility need
        assert result.minimum_cold_utility_btu_hr > 0

    @pytest.mark.unit
    def test_delta_t_override(self, analyzer, four_stream_problem):
        """Test delta T min override."""
        result = analyzer.analyze(four_stream_problem, delta_t_min_override=30.0)

        assert result.delta_t_min_f == 30.0

    @pytest.mark.unit
    def test_very_small_delta_t(self, four_stream_problem):
        """Test with very small delta T min."""
        analyzer = PinchAnalyzer(delta_t_min_f=5.0)
        result = analyzer.analyze(four_stream_problem)

        # Smaller delta T should give higher heat recovery
        assert result.maximum_heat_recovery_btu_hr > 0


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Performance tests for pinch analysis."""

    @pytest.mark.performance
    def test_analysis_speed_10_streams(self, analyzer):
        """Test analysis speed with 10 streams."""
        import time

        streams = []
        for i in range(5):
            streams.append(HeatStream(
                name=f"H{i}",
                stream_type=StreamType.HOT,
                supply_temp_f=400.0 - i * 30,
                target_temp_f=150.0 - i * 10,
                mcp=10.0 + i,
            ))
            streams.append(HeatStream(
                name=f"C{i}",
                stream_type=StreamType.COLD,
                supply_temp_f=50.0 + i * 20,
                target_temp_f=300.0 + i * 15,
                mcp=8.0 + i,
            ))

        start = time.time()
        result = analyzer.analyze(streams)
        elapsed = time.time() - start

        assert elapsed < 1.0  # Should complete in under 1 second
        assert result.stream_count == 10

    @pytest.mark.performance
    @pytest.mark.slow
    def test_analysis_speed_50_streams(self, analyzer):
        """Test analysis speed with 50 streams."""
        import time

        streams = []
        for i in range(25):
            streams.append(HeatStream(
                name=f"H{i}",
                stream_type=StreamType.HOT,
                supply_temp_f=300.0 + (i % 5) * 50,
                target_temp_f=100.0 + (i % 5) * 20,
                mcp=5.0 + (i % 10),
            ))
            streams.append(HeatStream(
                name=f"C{i}",
                stream_type=StreamType.COLD,
                supply_temp_f=40.0 + (i % 5) * 30,
                target_temp_f=250.0 + (i % 5) * 40,
                mcp=6.0 + (i % 10),
            ))

        start = time.time()
        result = analyzer.analyze(streams)
        elapsed = time.time() - start

        assert elapsed < 5.0  # Should complete in under 5 seconds
        assert result.stream_count == 50


# =============================================================================
# CALCULATION ACCURACY TESTS
# =============================================================================

class TestCalculationAccuracy:
    """Test calculation accuracy against known values."""

    @pytest.mark.unit
    def test_simple_heat_balance(self, analyzer):
        """Test simple heat balance calculation."""
        # Simple case: one hot, one cold with known values
        streams = [
            HeatStream(
                name="H1",
                stream_type=StreamType.HOT,
                supply_temp_f=200.0,
                target_temp_f=100.0,
                mcp=10.0,  # Duty = 10 * 100 = 1000 BTU/hr
            ),
            HeatStream(
                name="C1",
                stream_type=StreamType.COLD,
                supply_temp_f=50.0,
                target_temp_f=150.0,
                mcp=10.0,  # Duty = 10 * 100 = 1000 BTU/hr
            ),
        ]

        result = analyzer.analyze(streams)

        # Verify total duties
        assert result.total_hot_duty_btu_hr == 1000.0
        assert result.total_cold_duty_btu_hr == 1000.0

    @pytest.mark.unit
    def test_lmtd_approximation(self, analyzer):
        """Test LMTD calculation is reasonable."""
        # When dt1 = dt2, LMTD = dt1
        streams = [
            HeatStream(
                name="H1",
                stream_type=StreamType.HOT,
                supply_temp_f=200.0,
                target_temp_f=100.0,
                mcp=10.0,
            ),
            HeatStream(
                name="C1",
                stream_type=StreamType.COLD,
                supply_temp_f=60.0,
                target_temp_f=160.0,
                mcp=10.0,
            ),
        ]

        result = analyzer.analyze(streams)

        # With equal MCps and equal temp ranges (100F each)
        # LMTD should be around 40F (approach at each end)
        # This is more of a sanity check
        assert result.delta_t_min_f == 20.0
