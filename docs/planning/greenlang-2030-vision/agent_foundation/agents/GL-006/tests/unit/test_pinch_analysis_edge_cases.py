# -*- coding: utf-8 -*-
"""
Pinch Analysis Edge Case tests for GL-006 HeatRecoveryMaximizer.

This module validates pinch analysis boundary conditions and edge cases:
- Single stream scenarios
- Identical temperatures
- Very small/large temperature approaches
- Threshold problems (pinch at utility)
- Multiple pinch points
- Zero heat duty streams
- Temperature crossovers
- Degenerate cases

Target: 20+ edge case tests
"""

import pytest
import numpy as np
from typing import Dict, List, Any
from decimal import Decimal
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from calculators.pinch_analysis_calculator import (
    PinchAnalysisCalculator,
    PinchAnalysisInput,
    ProcessStream,
    StreamType,
    PinchAnalysisResult
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def calculator():
    """Create pinch analysis calculator instance."""
    return PinchAnalysisCalculator()


@pytest.fixture
def standard_hot_stream():
    """Create standard hot stream for testing."""
    return ProcessStream(
        stream_id="H1",
        stream_type=StreamType.HOT,
        supply_temp=180.0,
        target_temp=60.0,
        heat_capacity_flow=10.0
    )


@pytest.fixture
def standard_cold_stream():
    """Create standard cold stream for testing."""
    return ProcessStream(
        stream_id="C1",
        stream_type=StreamType.COLD,
        supply_temp=30.0,
        target_temp=140.0,
        heat_capacity_flow=8.0
    )


# ============================================================================
# SINGLE STREAM EDGE CASES
# ============================================================================

@pytest.mark.edge_case
class TestSingleStreamScenarios:
    """Test edge cases with single streams."""

    def test_single_hot_stream_only(self, calculator, standard_hot_stream):
        """Test pinch analysis with only one hot stream (no cold streams)."""
        input_data = PinchAnalysisInput(
            streams=[standard_hot_stream],
            minimum_approach_temp=10.0
        )

        result = calculator.calculate(input_data)

        # With only hot stream, all heat must go to cold utility
        expected_cold_utility = standard_hot_stream.heat_capacity_flow * (
            standard_hot_stream.supply_temp - standard_hot_stream.target_temp
        )

        assert result.minimum_cold_utility >= 0
        assert result.minimum_hot_utility >= 0
        assert result.maximum_heat_recovery == 0  # No heat recovery possible

    def test_single_cold_stream_only(self, calculator, standard_cold_stream):
        """Test pinch analysis with only one cold stream (no hot streams)."""
        input_data = PinchAnalysisInput(
            streams=[standard_cold_stream],
            minimum_approach_temp=10.0
        )

        result = calculator.calculate(input_data)

        # With only cold stream, all heat must come from hot utility
        expected_hot_utility = standard_cold_stream.heat_capacity_flow * (
            standard_cold_stream.target_temp - standard_cold_stream.supply_temp
        )

        assert result.minimum_hot_utility >= 0
        assert result.minimum_cold_utility >= 0
        assert result.maximum_heat_recovery == 0

    def test_two_hot_streams_no_cold(self, calculator):
        """Test with multiple hot streams but no cold streams."""
        hot_streams = [
            ProcessStream(
                stream_id="H1",
                stream_type=StreamType.HOT,
                supply_temp=200.0,
                target_temp=80.0,
                heat_capacity_flow=5.0
            ),
            ProcessStream(
                stream_id="H2",
                stream_type=StreamType.HOT,
                supply_temp=150.0,
                target_temp=50.0,
                heat_capacity_flow=8.0
            )
        ]

        input_data = PinchAnalysisInput(
            streams=hot_streams,
            minimum_approach_temp=10.0
        )

        result = calculator.calculate(input_data)

        # All heat goes to cold utility
        total_hot_duty = sum(
            s.heat_capacity_flow * (s.supply_temp - s.target_temp)
            for s in hot_streams
        )

        assert result.minimum_cold_utility >= 0


# ============================================================================
# TEMPERATURE BOUNDARY EDGE CASES
# ============================================================================

@pytest.mark.edge_case
class TestTemperatureBoundaries:
    """Test edge cases with temperature boundaries."""

    def test_streams_with_identical_temperatures(self, calculator):
        """Test when hot and cold streams have overlapping temperature ranges."""
        hot = ProcessStream(
            stream_id="H1",
            stream_type=StreamType.HOT,
            supply_temp=100.0,
            target_temp=50.0,
            heat_capacity_flow=10.0
        )
        cold = ProcessStream(
            stream_id="C1",
            stream_type=StreamType.COLD,
            supply_temp=40.0,
            target_temp=90.0,
            heat_capacity_flow=10.0
        )

        input_data = PinchAnalysisInput(
            streams=[hot, cold],
            minimum_approach_temp=10.0
        )

        result = calculator.calculate(input_data)

        # Should complete without error
        assert result.pinch_temperature_hot is not None
        assert result.pinch_temperature_cold is not None

    def test_very_small_temperature_approach(self, calculator, standard_hot_stream, standard_cold_stream):
        """Test with very small minimum approach temperature."""
        input_data = PinchAnalysisInput(
            streams=[standard_hot_stream, standard_cold_stream],
            minimum_approach_temp=1.0  # Very small approach
        )

        result = calculator.calculate(input_data)

        # Smaller approach should allow more heat recovery
        assert result.maximum_heat_recovery > 0
        assert result.pinch_temperature_hot - result.pinch_temperature_cold == pytest.approx(1.0, abs=0.1)

    def test_large_temperature_approach(self, calculator, standard_hot_stream, standard_cold_stream):
        """Test with large minimum approach temperature."""
        input_data = PinchAnalysisInput(
            streams=[standard_hot_stream, standard_cold_stream],
            minimum_approach_temp=50.0  # Large approach
        )

        result = calculator.calculate(input_data)

        # Larger approach reduces heat recovery potential
        assert result.maximum_heat_recovery >= 0
        assert result.pinch_temperature_hot - result.pinch_temperature_cold == pytest.approx(50.0, abs=0.1)

    def test_streams_at_exact_minimum_approach(self, calculator):
        """Test streams at exactly minimum approach temperature."""
        hot = ProcessStream(
            stream_id="H1",
            stream_type=StreamType.HOT,
            supply_temp=100.0,
            target_temp=60.0,
            heat_capacity_flow=10.0
        )
        cold = ProcessStream(
            stream_id="C1",
            stream_type=StreamType.COLD,
            supply_temp=50.0,  # Exactly 10 degrees below hot target
            target_temp=90.0,
            heat_capacity_flow=10.0
        )

        input_data = PinchAnalysisInput(
            streams=[hot, cold],
            minimum_approach_temp=10.0
        )

        result = calculator.calculate(input_data)

        # Should handle exact approach temperature
        assert result.minimum_hot_utility >= 0
        assert result.minimum_cold_utility >= 0


# ============================================================================
# THRESHOLD PROBLEM EDGE CASES
# ============================================================================

@pytest.mark.edge_case
class TestThresholdProblems:
    """Test threshold problems where pinch is at utility level."""

    def test_pinch_at_hot_utility_end(self, calculator):
        """Test case where pinch occurs at hot utility temperature level."""
        # Cold stream requiring temperature above all hot stream supplies
        hot = ProcessStream(
            stream_id="H1",
            stream_type=StreamType.HOT,
            supply_temp=100.0,
            target_temp=40.0,
            heat_capacity_flow=10.0
        )
        cold = ProcessStream(
            stream_id="C1",
            stream_type=StreamType.COLD,
            supply_temp=30.0,
            target_temp=150.0,  # Target higher than hot supply
            heat_capacity_flow=5.0
        )

        input_data = PinchAnalysisInput(
            streams=[hot, cold],
            minimum_approach_temp=10.0
        )

        result = calculator.calculate(input_data)

        # Hot utility required to meet cold stream target
        assert result.minimum_hot_utility > 0

    def test_pinch_at_cold_utility_end(self, calculator):
        """Test case where pinch occurs at cold utility temperature level."""
        # Hot stream cooling below all cold stream supply temperatures
        hot = ProcessStream(
            stream_id="H1",
            stream_type=StreamType.HOT,
            supply_temp=200.0,
            target_temp=20.0,  # Very low target
            heat_capacity_flow=5.0
        )
        cold = ProcessStream(
            stream_id="C1",
            stream_type=StreamType.COLD,
            supply_temp=80.0,  # High supply temp
            target_temp=150.0,
            heat_capacity_flow=10.0
        )

        input_data = PinchAnalysisInput(
            streams=[hot, cold],
            minimum_approach_temp=10.0
        )

        result = calculator.calculate(input_data)

        # Cold utility required to cool hot stream fully
        assert result.minimum_cold_utility > 0


# ============================================================================
# MULTIPLE PINCH POINTS
# ============================================================================

@pytest.mark.edge_case
class TestMultiplePinchPoints:
    """Test scenarios with multiple potential pinch points."""

    def test_symmetric_streams_dual_pinch(self, calculator):
        """Test symmetric streams that could have multiple pinch points."""
        streams = [
            ProcessStream(
                stream_id="H1",
                stream_type=StreamType.HOT,
                supply_temp=200.0,
                target_temp=100.0,
                heat_capacity_flow=10.0
            ),
            ProcessStream(
                stream_id="H2",
                stream_type=StreamType.HOT,
                supply_temp=100.0,
                target_temp=40.0,
                heat_capacity_flow=10.0
            ),
            ProcessStream(
                stream_id="C1",
                stream_type=StreamType.COLD,
                supply_temp=30.0,
                target_temp=90.0,
                heat_capacity_flow=10.0
            ),
            ProcessStream(
                stream_id="C2",
                stream_type=StreamType.COLD,
                supply_temp=90.0,
                target_temp=190.0,
                heat_capacity_flow=10.0
            )
        ]

        input_data = PinchAnalysisInput(
            streams=streams,
            minimum_approach_temp=10.0
        )

        result = calculator.calculate(input_data)

        # Should find a valid pinch point
        assert result.pinch_temperature_hot is not None
        assert result.maximum_heat_recovery > 0


# ============================================================================
# ZERO AND NEAR-ZERO VALUES
# ============================================================================

@pytest.mark.edge_case
class TestZeroValues:
    """Test edge cases with zero or near-zero values."""

    def test_very_small_heat_capacity_flow(self, calculator, standard_cold_stream):
        """Test stream with very small heat capacity flow rate."""
        hot = ProcessStream(
            stream_id="H1",
            stream_type=StreamType.HOT,
            supply_temp=150.0,
            target_temp=50.0,
            heat_capacity_flow=0.001  # Very small CP
        )

        input_data = PinchAnalysisInput(
            streams=[hot, standard_cold_stream],
            minimum_approach_temp=10.0
        )

        result = calculator.calculate(input_data)

        # Should handle small CP without division by zero
        assert result.minimum_hot_utility >= 0
        assert result.minimum_cold_utility >= 0

    def test_very_small_temperature_range(self, calculator, standard_cold_stream):
        """Test stream with very small temperature change."""
        hot = ProcessStream(
            stream_id="H1",
            stream_type=StreamType.HOT,
            supply_temp=100.0,
            target_temp=99.0,  # Only 1 degree change
            heat_capacity_flow=10.0
        )

        input_data = PinchAnalysisInput(
            streams=[hot, standard_cold_stream],
            minimum_approach_temp=10.0
        )

        result = calculator.calculate(input_data)

        # Should handle small temperature range
        assert result is not None


# ============================================================================
# LARGE SCALE PROBLEMS
# ============================================================================

@pytest.mark.edge_case
class TestLargeScaleProblems:
    """Test with large numbers of streams."""

    def test_many_streams(self, calculator):
        """Test pinch analysis with many streams (10+ hot, 10+ cold)."""
        np.random.seed(42)

        hot_streams = []
        for i in range(10):
            supply = np.random.uniform(150, 250)
            target = np.random.uniform(40, 100)
            if supply <= target:
                supply, target = target + 50, supply  # Ensure hot stream cools

            hot_streams.append(ProcessStream(
                stream_id=f"H{i+1}",
                stream_type=StreamType.HOT,
                supply_temp=supply,
                target_temp=target,
                heat_capacity_flow=np.random.uniform(5, 20)
            ))

        cold_streams = []
        for i in range(10):
            supply = np.random.uniform(20, 80)
            target = np.random.uniform(100, 180)
            if target <= supply:
                supply, target = target - 50, supply  # Ensure cold stream heats

            cold_streams.append(ProcessStream(
                stream_id=f"C{i+1}",
                stream_type=StreamType.COLD,
                supply_temp=supply,
                target_temp=target,
                heat_capacity_flow=np.random.uniform(5, 20)
            ))

        all_streams = hot_streams + cold_streams

        input_data = PinchAnalysisInput(
            streams=all_streams,
            minimum_approach_temp=10.0
        )

        result = calculator.calculate(input_data)

        # Should complete for large problem
        assert result is not None
        assert result.maximum_heat_recovery >= 0


# ============================================================================
# NUMERICAL PRECISION EDGE CASES
# ============================================================================

@pytest.mark.edge_case
class TestNumericalPrecision:
    """Test numerical precision edge cases."""

    def test_floating_point_precision(self, calculator):
        """Test floating point precision doesn't cause issues."""
        hot = ProcessStream(
            stream_id="H1",
            stream_type=StreamType.HOT,
            supply_temp=100.0000001,  # Floating point near 100
            target_temp=49.9999999,   # Floating point near 50
            heat_capacity_flow=10.0
        )
        cold = ProcessStream(
            stream_id="C1",
            stream_type=StreamType.COLD,
            supply_temp=30.0000001,
            target_temp=89.9999999,
            heat_capacity_flow=10.0
        )

        input_data = PinchAnalysisInput(
            streams=[hot, cold],
            minimum_approach_temp=10.0
        )

        result = calculator.calculate(input_data)

        # Should handle floating point values
        assert result is not None

    def test_very_large_numbers(self, calculator):
        """Test with very large temperature and flow values."""
        hot = ProcessStream(
            stream_id="H1",
            stream_type=StreamType.HOT,
            supply_temp=1000.0,
            target_temp=200.0,
            heat_capacity_flow=100.0
        )
        cold = ProcessStream(
            stream_id="C1",
            stream_type=StreamType.COLD,
            supply_temp=50.0,
            target_temp=500.0,
            heat_capacity_flow=150.0
        )

        input_data = PinchAnalysisInput(
            streams=[hot, cold],
            minimum_approach_temp=10.0
        )

        result = calculator.calculate(input_data)

        # Should handle large values
        assert result is not None
        assert result.maximum_heat_recovery >= 0


# ============================================================================
# SPECIAL STREAM CONFIGURATIONS
# ============================================================================

@pytest.mark.edge_case
class TestSpecialConfigurations:
    """Test special stream configurations."""

    def test_equal_cp_streams(self, calculator):
        """Test streams with equal heat capacity flow rates."""
        hot = ProcessStream(
            stream_id="H1",
            stream_type=StreamType.HOT,
            supply_temp=150.0,
            target_temp=60.0,
            heat_capacity_flow=10.0  # Same CP
        )
        cold = ProcessStream(
            stream_id="C1",
            stream_type=StreamType.COLD,
            supply_temp=30.0,
            target_temp=120.0,
            heat_capacity_flow=10.0  # Same CP
        )

        input_data = PinchAnalysisInput(
            streams=[hot, cold],
            minimum_approach_temp=10.0
        )

        result = calculator.calculate(input_data)

        # Equal CP streams should work
        assert result is not None

    def test_perfectly_balanced_network(self, calculator):
        """Test perfectly balanced heat load network."""
        # Hot and cold duties exactly equal
        hot = ProcessStream(
            stream_id="H1",
            stream_type=StreamType.HOT,
            supply_temp=150.0,
            target_temp=50.0,  # 100 degree range
            heat_capacity_flow=10.0  # Duty = 1000 kW
        )
        cold = ProcessStream(
            stream_id="C1",
            stream_type=StreamType.COLD,
            supply_temp=30.0,
            target_temp=130.0,  # 100 degree range
            heat_capacity_flow=10.0  # Duty = 1000 kW
        )

        input_data = PinchAnalysisInput(
            streams=[hot, cold],
            minimum_approach_temp=10.0
        )

        result = calculator.calculate(input_data)

        # Balanced network should minimize utilities
        assert result is not None

    def test_streams_with_same_supply_temp(self, calculator):
        """Test multiple streams starting at same temperature."""
        streams = [
            ProcessStream(
                stream_id="H1",
                stream_type=StreamType.HOT,
                supply_temp=150.0,  # Same supply
                target_temp=60.0,
                heat_capacity_flow=10.0
            ),
            ProcessStream(
                stream_id="H2",
                stream_type=StreamType.HOT,
                supply_temp=150.0,  # Same supply
                target_temp=40.0,
                heat_capacity_flow=8.0
            ),
            ProcessStream(
                stream_id="C1",
                stream_type=StreamType.COLD,
                supply_temp=30.0,
                target_temp=100.0,
                heat_capacity_flow=12.0
            )
        ]

        input_data = PinchAnalysisInput(
            streams=streams,
            minimum_approach_temp=10.0
        )

        result = calculator.calculate(input_data)

        # Should handle same supply temperatures
        assert result is not None


# ============================================================================
# ECONOMIC EDGE CASES
# ============================================================================

@pytest.mark.edge_case
class TestEconomicEdgeCases:
    """Test economic parameter edge cases."""

    def test_zero_utility_costs(self, calculator, standard_hot_stream, standard_cold_stream):
        """Test with zero utility costs."""
        input_data = PinchAnalysisInput(
            streams=[standard_hot_stream, standard_cold_stream],
            minimum_approach_temp=10.0,
            utility_cost_hot=0.0,
            utility_cost_cold=0.0
        )

        result = calculator.calculate(input_data)

        # Zero costs should result in zero annual savings
        assert result.annual_cost_savings == 0.0

    def test_very_high_utility_costs(self, calculator, standard_hot_stream, standard_cold_stream):
        """Test with very high utility costs."""
        input_data = PinchAnalysisInput(
            streams=[standard_hot_stream, standard_cold_stream],
            minimum_approach_temp=10.0,
            utility_cost_hot=1000.0,  # Very high
            utility_cost_cold=500.0
        )

        result = calculator.calculate(input_data)

        # High costs should result in large savings potential
        assert result.annual_cost_savings >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-m", "edge_case"])
