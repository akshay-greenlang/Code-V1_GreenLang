"""
GL-006 HEATRECLAIM - Pinch Analysis Tests

Unit tests for pinch analysis calculator.
"""

import pytest
import math
from typing import List

from ..calculators.pinch_analysis import PinchAnalysisCalculator
from ..core.schemas import HeatStream, PinchAnalysisResult
from ..core.config import StreamType, Phase


class TestPinchAnalysisCalculator:
    """Tests for PinchAnalysisCalculator."""

    def test_init_default(self):
        """Test default initialization."""
        calc = PinchAnalysisCalculator()
        assert calc.delta_t_min == 10.0
        assert calc.T_ref == 25.0

    def test_init_custom(self):
        """Test custom initialization."""
        calc = PinchAnalysisCalculator(delta_t_min=15.0, T_ref=20.0)
        assert calc.delta_t_min == 15.0
        assert calc.T_ref == 20.0

    def test_analyze_simple_case(
        self,
        simple_hot_streams,
        simple_cold_streams,
    ):
        """Test analysis on simple 2x2 problem."""
        calc = PinchAnalysisCalculator(delta_t_min=10.0)
        result = calc.analyze(simple_hot_streams, simple_cold_streams)

        assert isinstance(result, PinchAnalysisResult)
        assert result.pinch_temperature_C > 0
        assert result.minimum_hot_utility_kW >= 0
        assert result.minimum_cold_utility_kW >= 0
        assert result.maximum_heat_recovery_kW >= 0

    def test_analyze_textbook_case(
        self,
        textbook_hot_streams,
        textbook_cold_streams,
    ):
        """Test against textbook example."""
        calc = PinchAnalysisCalculator(delta_t_min=10.0)
        result = calc.analyze(textbook_hot_streams, textbook_cold_streams)

        # Verify pinch temperature (textbook value ~90°C for hot side)
        assert 85 <= result.pinch_temperature_C <= 95

        # Verify energy balance
        total_hot_duty = sum(s.duty_kW for s in textbook_hot_streams)
        total_cold_duty = sum(s.duty_kW for s in textbook_cold_streams)

        # Q_H - Q_C = cold_duty - hot_duty (energy balance)
        expected_imbalance = total_cold_duty - total_hot_duty
        actual_imbalance = result.minimum_hot_utility_kW - result.minimum_cold_utility_kW
        assert abs(expected_imbalance - actual_imbalance) < 1.0  # Within 1 kW

    def test_composite_curves_monotonic(
        self,
        simple_hot_streams,
        simple_cold_streams,
    ):
        """Test that composite curves are monotonically increasing."""
        calc = PinchAnalysisCalculator()
        result = calc.analyze(simple_hot_streams, simple_cold_streams)

        # Hot composite should decrease in temperature as enthalpy increases
        hot_T = result.hot_composite_T_C
        hot_H = result.hot_composite_H_kW

        for i in range(1, len(hot_H)):
            assert hot_H[i] >= hot_H[i-1], "Hot enthalpy should increase"

        # Cold composite should increase in temperature as enthalpy increases
        cold_T = result.cold_composite_T_C
        cold_H = result.cold_composite_H_kW

        for i in range(1, len(cold_H)):
            assert cold_H[i] >= cold_H[i-1], "Cold enthalpy should increase"

    def test_energy_balance(
        self,
        simple_hot_streams,
        simple_cold_streams,
    ):
        """Test overall energy balance."""
        calc = PinchAnalysisCalculator()
        result = calc.analyze(simple_hot_streams, simple_cold_streams)

        total_hot = sum(s.duty_kW for s in simple_hot_streams)
        total_cold = sum(s.duty_kW for s in simple_cold_streams)

        # Heat recovery + utilities = total duties
        # Hot side: total_hot = Q_rec + Q_Cmin
        # Cold side: total_cold = Q_rec + Q_Hmin
        Q_rec = result.maximum_heat_recovery_kW
        Q_Hmin = result.minimum_hot_utility_kW
        Q_Cmin = result.minimum_cold_utility_kW

        assert abs(total_hot - Q_rec - Q_Cmin) < 1.0
        assert abs(total_cold - Q_rec - Q_Hmin) < 1.0

    def test_threshold_problem_detection(self):
        """Test detection of threshold problems."""
        calc = PinchAnalysisCalculator(delta_t_min=10.0)

        # Create threshold problem: all hot > all cold temperatures
        hot_streams = [
            HeatStream(
                stream_id="H1",
                stream_name="Hot 1",
                stream_type=StreamType.HOT,
                fluid_name="Process",
                phase=Phase.LIQUID,
                T_supply_C=200.0,
                T_target_C=150.0,
                m_dot_kg_s=1.0,
                Cp_kJ_kgK=4.0,
            ),
        ]
        cold_streams = [
            HeatStream(
                stream_id="C1",
                stream_name="Cold 1",
                stream_type=StreamType.COLD,
                fluid_name="Process",
                phase=Phase.LIQUID,
                T_supply_C=20.0,
                T_target_C=80.0,
                m_dot_kg_s=1.0,
                Cp_kJ_kgK=4.0,
            ),
        ]

        result = calc.analyze(hot_streams, cold_streams)

        # Should be threshold problem (no pinch within temperature range)
        # Either Q_Hmin or Q_Cmin should be zero
        assert result.minimum_hot_utility_kW == 0 or result.minimum_cold_utility_kW == 0

    def test_provenance_hash_deterministic(
        self,
        simple_hot_streams,
        simple_cold_streams,
    ):
        """Test that computation hash is deterministic."""
        calc = PinchAnalysisCalculator()

        result1 = calc.analyze(simple_hot_streams, simple_cold_streams)
        result2 = calc.analyze(simple_hot_streams, simple_cold_streams)

        assert result1.computation_hash == result2.computation_hash

    def test_different_inputs_different_hash(
        self,
        simple_hot_streams,
        simple_cold_streams,
        textbook_hot_streams,
        textbook_cold_streams,
    ):
        """Test that different inputs produce different hashes."""
        calc = PinchAnalysisCalculator()

        result1 = calc.analyze(simple_hot_streams, simple_cold_streams)
        result2 = calc.analyze(textbook_hot_streams, textbook_cold_streams)

        assert result1.computation_hash != result2.computation_hash

    def test_delta_t_min_effect(
        self,
        simple_hot_streams,
        simple_cold_streams,
    ):
        """Test effect of varying delta T min."""
        results = []
        for dt_min in [5.0, 10.0, 20.0, 30.0]:
            calc = PinchAnalysisCalculator(delta_t_min=dt_min)
            result = calc.analyze(simple_hot_streams, simple_cold_streams)
            results.append((dt_min, result))

        # Higher delta T min -> lower heat recovery
        for i in range(1, len(results)):
            prev_dt, prev_result = results[i-1]
            curr_dt, curr_result = results[i]

            assert curr_result.maximum_heat_recovery_kW <= prev_result.maximum_heat_recovery_kW

    def test_empty_streams_error(self):
        """Test that empty stream lists raise error."""
        calc = PinchAnalysisCalculator()

        with pytest.raises(ValueError):
            calc.analyze([], [])

    def test_single_stream_each(self):
        """Test with single hot and cold stream."""
        calc = PinchAnalysisCalculator(delta_t_min=10.0)

        hot = [
            HeatStream(
                stream_id="H1",
                stream_name="Hot 1",
                stream_type=StreamType.HOT,
                fluid_name="Water",
                phase=Phase.LIQUID,
                T_supply_C=100.0,
                T_target_C=50.0,
                m_dot_kg_s=1.0,
                Cp_kJ_kgK=4.18,
            ),
        ]
        cold = [
            HeatStream(
                stream_id="C1",
                stream_name="Cold 1",
                stream_type=StreamType.COLD,
                fluid_name="Water",
                phase=Phase.LIQUID,
                T_supply_C=20.0,
                T_target_C=80.0,
                m_dot_kg_s=1.0,
                Cp_kJ_kgK=4.18,
            ),
        ]

        result = calc.analyze(hot, cold)

        assert result.pinch_temperature_C > 0
        assert result.maximum_heat_recovery_kW > 0


class TestGrandCompositeCurve:
    """Tests for grand composite curve generation."""

    def test_gcc_generation(
        self,
        simple_hot_streams,
        simple_cold_streams,
    ):
        """Test GCC is generated correctly."""
        calc = PinchAnalysisCalculator()
        result = calc.analyze(simple_hot_streams, simple_cold_streams)

        assert len(result.grand_composite_T_C) > 0
        assert len(result.grand_composite_H_kW) > 0
        assert len(result.grand_composite_T_C) == len(result.grand_composite_H_kW)

    def test_gcc_pinch_at_zero(
        self,
        textbook_hot_streams,
        textbook_cold_streams,
    ):
        """Test that GCC touches zero at pinch."""
        calc = PinchAnalysisCalculator(delta_t_min=10.0)
        result = calc.analyze(textbook_hot_streams, textbook_cold_streams)

        # Find minimum heat flow in GCC (should be at or near zero at pinch)
        min_heat_flow = min(result.grand_composite_H_kW)
        assert abs(min_heat_flow) < 1.0  # Within 1 kW of zero


class TestProblemTableAlgorithm:
    """Tests for problem table algorithm internals."""

    def test_temperature_intervals(
        self,
        simple_hot_streams,
        simple_cold_streams,
    ):
        """Test temperature interval identification."""
        calc = PinchAnalysisCalculator(delta_t_min=10.0)

        # Get unique temperatures (shifted)
        temps = set()
        for s in simple_hot_streams:
            temps.add(s.T_supply_C)
            temps.add(s.T_target_C)
        for s in simple_cold_streams:
            temps.add(s.T_supply_C + calc.delta_t_min)
            temps.add(s.T_target_C + calc.delta_t_min)

        # Should have proper number of interval boundaries
        assert len(temps) >= 4  # At least 4 unique temperatures


class TestIndustrialScale:
    """Tests for industrial-scale problems."""

    def test_large_problem(
        self,
        industrial_hot_streams,
        industrial_cold_streams,
    ):
        """Test analysis of industrial-scale problem."""
        calc = PinchAnalysisCalculator(delta_t_min=20.0)
        result = calc.analyze(industrial_hot_streams, industrial_cold_streams)

        # Should complete without error
        assert result is not None
        assert result.pinch_temperature_C > 0

        # Industrial problems have significant energy flows
        assert result.maximum_heat_recovery_kW > 1000  # MW scale

    def test_performance_large_problem(
        self,
        industrial_hot_streams,
        industrial_cold_streams,
    ):
        """Test performance on larger problem."""
        import time

        calc = PinchAnalysisCalculator()

        start = time.time()
        result = calc.analyze(industrial_hot_streams, industrial_cold_streams)
        elapsed = time.time() - start

        # Should complete in reasonable time
        assert elapsed < 1.0  # Less than 1 second
