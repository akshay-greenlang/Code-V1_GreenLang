"""
GL-006 HEATRECLAIM - Exergy Calculator Tests

Comprehensive test suite for second-law (exergy) analysis calculations.
Validates thermodynamic correctness against textbook examples and
verifies reproducibility with SHA-256 hash verification.

Reference: Bejan, Tsatsaronis, Moran, "Thermal Design and Optimization"
"""

import math
import pytest
from unittest.mock import MagicMock

from ..core.schemas import HeatStream, HeatExchanger
from ..core.config import StreamType, Phase, ExchangerType, FlowArrangement
from ..calculators.exergy_calculator import (
    ExergyCalculator,
    StreamExergy,
    ExchangerExergy,
    calculate_carnot_efficiency,
    calculate_exergy_factor,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def exergy_calculator():
    """Create exergy calculator with standard reference conditions."""
    return ExergyCalculator(T0_K=298.15, p0_kPa=101.325)


@pytest.fixture
def hot_stream():
    """Create a sample hot stream."""
    return HeatStream(
        stream_id="H1",
        stream_name="Hot Process Stream",
        stream_type=StreamType.HOT,
        fluid_name="Water",
        phase=Phase.LIQUID,
        T_supply_C=150.0,
        T_target_C=60.0,
        m_dot_kg_s=2.0,
        Cp_kJ_kgK=4.186,
    )


@pytest.fixture
def cold_stream():
    """Create a sample cold stream."""
    return HeatStream(
        stream_id="C1",
        stream_name="Cold Process Stream",
        stream_type=StreamType.COLD,
        fluid_name="Water",
        phase=Phase.LIQUID,
        T_supply_C=30.0,
        T_target_C=100.0,
        m_dot_kg_s=3.0,
        Cp_kJ_kgK=4.186,
    )


@pytest.fixture
def heat_exchanger():
    """Create a sample heat exchanger."""
    return HeatExchanger(
        exchanger_id="HX-001",
        exchanger_name="Test HX",
        exchanger_type=ExchangerType.SHELL_AND_TUBE,
        hot_stream_id="H1",
        cold_stream_id="C1",
        duty_kW=500.0,
        hot_inlet_T_C=150.0,
        hot_outlet_T_C=90.0,
        cold_inlet_T_C=30.0,
        cold_outlet_T_C=70.0,
        area_m2=25.0,
    )


# =============================================================================
# TEST: INITIALIZATION
# =============================================================================

class TestExergyCalculatorInit:
    """Test exergy calculator initialization."""

    def test_default_initialization(self):
        """Test default reference conditions."""
        calc = ExergyCalculator()
        assert calc.T0_K == pytest.approx(298.15, rel=0.001)
        assert calc.p0_kPa == pytest.approx(101.325, rel=0.001)

    def test_custom_reference_temperature(self):
        """Test custom reference temperature."""
        calc = ExergyCalculator(T0_K=288.15)  # 15°C
        assert calc.T0_K == 288.15
        assert calc.T0_C == pytest.approx(15.0, rel=0.01)

    def test_custom_reference_pressure(self):
        """Test custom reference pressure."""
        calc = ExergyCalculator(p0_kPa=100.0)
        assert calc.p0_kPa == 100.0


# =============================================================================
# TEST: SPECIFIC EXERGY CALCULATIONS
# =============================================================================

class TestSpecificExergy:
    """Test specific exergy calculations."""

    def test_exergy_at_reference_temperature(self, exergy_calculator):
        """Exergy should be zero at reference temperature."""
        T_K = 298.15  # Same as T0
        Cp = 4.186
        ex = exergy_calculator._specific_exergy(T_K, Cp)
        assert ex == pytest.approx(0.0, abs=0.001)

    def test_exergy_above_reference_temperature(self, exergy_calculator):
        """Exergy should be positive above reference temperature."""
        T_K = 373.15  # 100°C
        Cp = 4.186
        ex = exergy_calculator._specific_exergy(T_K, Cp)
        assert ex > 0
        # Expected: Cp * [(T - T0) - T0 * ln(T/T0)]
        # = 4.186 * [(373.15 - 298.15) - 298.15 * ln(373.15/298.15)]
        # = 4.186 * [75 - 298.15 * 0.2244]
        # = 4.186 * [75 - 66.91]
        # = 4.186 * 8.09 ≈ 33.86 kJ/kg
        assert ex == pytest.approx(33.86, rel=0.05)

    def test_exergy_below_reference_temperature(self, exergy_calculator):
        """Exergy should be positive below reference temperature (cooling potential)."""
        T_K = 268.15  # -5°C (below T0)
        Cp = 4.186
        ex = exergy_calculator._specific_exergy(T_K, Cp)
        # Below T0, there's cooling exergy
        assert ex > 0

    def test_exergy_zero_temperature_returns_zero(self, exergy_calculator):
        """Should return zero for invalid temperature."""
        ex = exergy_calculator._specific_exergy(0.0, 4.186)
        assert ex == 0.0

    def test_exergy_negative_temperature_returns_zero(self, exergy_calculator):
        """Should return zero for negative temperature."""
        ex = exergy_calculator._specific_exergy(-100.0, 4.186)
        assert ex == 0.0


# =============================================================================
# TEST: STREAM EXERGY CALCULATIONS
# =============================================================================

class TestStreamExergy:
    """Test stream exergy rate calculations."""

    def test_hot_stream_exergy_rate(self, exergy_calculator, hot_stream):
        """Test exergy calculation for hot stream."""
        result = exergy_calculator.calculate_stream_exergy_rate(hot_stream)

        assert isinstance(result, StreamExergy)
        assert result.stream_id == "H1"
        assert result.exergy_rate_kW > 0  # Hot stream has positive exergy

    def test_cold_stream_exergy_rate(self, exergy_calculator, cold_stream):
        """Test exergy calculation for cold stream."""
        result = exergy_calculator.calculate_stream_exergy_rate(cold_stream)

        assert isinstance(result, StreamExergy)
        assert result.stream_id == "C1"
        # Cold stream at 30°C is close to reference (25°C), so low exergy

    def test_exergy_change_hot_stream(self, exergy_calculator, hot_stream):
        """Hot stream losing heat should have negative exergy change."""
        result = exergy_calculator.calculate_stream_exergy_rate(hot_stream)
        # Hot stream goes from 150°C to 60°C, losing exergy
        assert result.exergy_change_kW < 0

    def test_exergy_change_cold_stream(self, exergy_calculator, cold_stream):
        """Cold stream gaining heat should have positive exergy change."""
        result = exergy_calculator.calculate_stream_exergy_rate(cold_stream)
        # Cold stream goes from 30°C to 100°C, gaining exergy
        assert result.exergy_change_kW > 0

    def test_specific_exergy_units(self, exergy_calculator, hot_stream):
        """Verify specific exergy is in kJ/kg."""
        result = exergy_calculator.calculate_stream_exergy_rate(hot_stream)
        # At 150°C, specific exergy should be order of 50-100 kJ/kg
        assert 10 < result.specific_exergy_kJ_kg < 200


# =============================================================================
# TEST: HEAT EXCHANGER EXERGY DESTRUCTION
# =============================================================================

class TestExchangerExergyDestruction:
    """Test exergy destruction in heat exchangers."""

    def test_exchanger_exergy_destruction_positive(
        self, exergy_calculator, heat_exchanger, hot_stream, cold_stream
    ):
        """Exergy destruction should always be positive (2nd law)."""
        result = exergy_calculator.calculate_exchanger_exergy(
            heat_exchanger, hot_stream, cold_stream
        )

        assert isinstance(result, ExchangerExergy)
        assert result.exergy_destruction_kW >= 0

    def test_exchanger_exergy_efficiency_bounded(
        self, exergy_calculator, heat_exchanger, hot_stream, cold_stream
    ):
        """Exergy efficiency should be between 0 and 1."""
        result = exergy_calculator.calculate_exchanger_exergy(
            heat_exchanger, hot_stream, cold_stream
        )

        assert 0 <= result.exergy_efficiency <= 1

    def test_entropy_generation_positive(
        self, exergy_calculator, heat_exchanger, hot_stream, cold_stream
    ):
        """Entropy generation should be positive for irreversible process."""
        result = exergy_calculator.calculate_exchanger_exergy(
            heat_exchanger, hot_stream, cold_stream
        )

        assert result.entropy_generation_kW_K >= 0

    def test_exergy_balance(
        self, exergy_calculator, heat_exchanger, hot_stream, cold_stream
    ):
        """Verify exergy balance: in = out + destruction."""
        result = exergy_calculator.calculate_exchanger_exergy(
            heat_exchanger, hot_stream, cold_stream
        )

        exergy_in = result.hot_exergy_in_kW + result.cold_exergy_in_kW
        exergy_out = result.hot_exergy_out_kW + result.cold_exergy_out_kW
        destruction = result.exergy_destruction_kW

        # Balance should hold (with small tolerance)
        assert exergy_in == pytest.approx(exergy_out + destruction, rel=0.01)

    def test_exchanger_without_stream_info(self, exergy_calculator, heat_exchanger):
        """Test exchanger exergy calculation without stream objects."""
        result = exergy_calculator.calculate_exchanger_exergy(
            heat_exchanger, None, None
        )

        # Should still calculate using estimated properties
        assert result.exergy_destruction_kW >= 0


# =============================================================================
# TEST: NETWORK EXERGY ANALYSIS
# =============================================================================

class TestNetworkExergyAnalysis:
    """Test complete network exergy analysis."""

    def test_network_analysis_basic(self, exergy_calculator, hot_stream, cold_stream, heat_exchanger):
        """Test basic network analysis."""
        result = exergy_calculator.analyze_network(
            hot_streams=[hot_stream],
            cold_streams=[cold_stream],
            exchangers=[heat_exchanger],
        )

        assert result is not None
        assert result.reference_temperature_K == exergy_calculator.T0_K
        assert result.total_exergy_input_kW > 0
        assert result.total_exergy_destruction_kW >= 0
        assert 0 <= result.exergy_efficiency <= 1

    def test_network_with_utilities(self, exergy_calculator, hot_stream, cold_stream, heat_exchanger):
        """Test network analysis with utility consumption."""
        result = exergy_calculator.analyze_network(
            hot_streams=[hot_stream],
            cold_streams=[cold_stream],
            exchangers=[heat_exchanger],
            hot_utility_duty_kW=100.0,
            cold_utility_duty_kW=50.0,
            hot_utility_T_C=200.0,
            cold_utility_T_C=20.0,
        )

        # Utilities should add to exergy destruction
        assert result.total_exergy_destruction_kW > 0
        assert "hot_utility" in result.exergy_by_utility or "cold_utility" in result.exergy_by_utility

    def test_improvement_potential(self, exergy_calculator, hot_stream, cold_stream, heat_exchanger):
        """Test improvement potential calculation."""
        result = exergy_calculator.analyze_network(
            hot_streams=[hot_stream],
            cold_streams=[cold_stream],
            exchangers=[heat_exchanger],
        )

        # Improvement potential equals exergy destruction
        assert result.improvement_potential_kW == pytest.approx(
            result.total_exergy_destruction_kW, rel=0.01
        )

    def test_provenance_hashes(self, exergy_calculator, hot_stream, cold_stream, heat_exchanger):
        """Verify provenance hash generation."""
        result = exergy_calculator.analyze_network(
            hot_streams=[hot_stream],
            cold_streams=[cold_stream],
            exchangers=[heat_exchanger],
        )

        assert result.input_hash is not None
        assert result.output_hash is not None
        assert len(result.input_hash) == 16  # SHA-256 truncated
        assert result.formula_version == "EXERGY_v1.0"

    def test_reproducibility(self, exergy_calculator, hot_stream, cold_stream, heat_exchanger):
        """Same inputs should produce identical results."""
        result1 = exergy_calculator.analyze_network(
            hot_streams=[hot_stream],
            cold_streams=[cold_stream],
            exchangers=[heat_exchanger],
        )

        result2 = exergy_calculator.analyze_network(
            hot_streams=[hot_stream],
            cold_streams=[cold_stream],
            exchangers=[heat_exchanger],
        )

        assert result1.total_exergy_destruction_kW == result2.total_exergy_destruction_kW
        assert result1.exergy_efficiency == result2.exergy_efficiency
        assert result1.input_hash == result2.input_hash


# =============================================================================
# TEST: UTILITY FUNCTIONS
# =============================================================================

class TestUtilityFunctions:
    """Test utility exergy functions."""

    def test_carnot_efficiency_valid(self):
        """Test Carnot efficiency with valid temperatures."""
        # At T_hot=400K, T_cold=300K: η = 1 - 300/400 = 0.25
        eta = calculate_carnot_efficiency(400.0, 300.0)
        assert eta == pytest.approx(0.25, rel=0.001)

    def test_carnot_efficiency_high_temp_ratio(self):
        """Test Carnot efficiency with high temperature ratio."""
        # At T_hot=600K, T_cold=300K: η = 1 - 300/600 = 0.5
        eta = calculate_carnot_efficiency(600.0, 300.0)
        assert eta == pytest.approx(0.5, rel=0.001)

    def test_carnot_efficiency_invalid_temps(self):
        """Carnot efficiency should be zero for invalid temperatures."""
        assert calculate_carnot_efficiency(0, 300) == 0.0
        assert calculate_carnot_efficiency(300, 0) == 0.0
        assert calculate_carnot_efficiency(300, 400) == 0.0  # Cold > Hot

    def test_exergy_factor_above_reference(self):
        """Test exergy factor above reference temperature."""
        # At T=400K, T0=300K: τ = 1 - 300/400 = 0.25
        tau = calculate_exergy_factor(400.0, 300.0)
        assert tau == pytest.approx(0.25, rel=0.001)

    def test_exergy_factor_below_reference(self):
        """Test exergy factor below reference temperature."""
        # At T=250K, T0=300K: τ = 300/250 - 1 = 0.2
        tau = calculate_exergy_factor(250.0, 300.0)
        assert tau == pytest.approx(0.2, rel=0.001)

    def test_exergy_factor_at_reference(self):
        """Exergy factor at reference temperature should be zero."""
        tau = calculate_exergy_factor(300.0, 300.0)
        assert tau == pytest.approx(0.0, abs=0.001)

    def test_exergy_factor_invalid_temps(self):
        """Exergy factor should be zero for invalid temperatures."""
        assert calculate_exergy_factor(0, 300) == 0.0
        assert calculate_exergy_factor(300, 0) == 0.0


# =============================================================================
# TEST: EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_duty_exchanger(self, exergy_calculator):
        """Test exchanger with zero duty."""
        hx = HeatExchanger(
            exchanger_id="HX-ZERO",
            exchanger_type=ExchangerType.SHELL_AND_TUBE,
            hot_stream_id="H1",
            cold_stream_id="C1",
            duty_kW=0.0,
            hot_inlet_T_C=100.0,
            hot_outlet_T_C=100.0,
            cold_inlet_T_C=50.0,
            cold_outlet_T_C=50.0,
            area_m2=0.0,
        )

        result = exergy_calculator.calculate_exchanger_exergy(hx)
        assert result.exergy_destruction_kW >= 0

    def test_stream_at_reference_temperature(self, exergy_calculator):
        """Test stream at reference temperature has minimal exergy."""
        stream = HeatStream(
            stream_id="REF",
            stream_type=StreamType.HOT,
            T_supply_C=25.0,  # ~298K
            T_target_C=25.0,
            m_dot_kg_s=1.0,
            Cp_kJ_kgK=4.186,
        )

        result = exergy_calculator.calculate_stream_exergy_rate(stream)
        # Exergy should be very close to zero
        assert abs(result.exergy_rate_kW) < 5.0

    def test_empty_network(self, exergy_calculator):
        """Test analysis of empty network."""
        result = exergy_calculator.analyze_network(
            hot_streams=[],
            cold_streams=[],
            exchangers=[],
        )

        assert result.total_exergy_input_kW == 0
        assert result.total_exergy_destruction_kW == 0


# =============================================================================
# TEST: THERMODYNAMIC CONSISTENCY
# =============================================================================

class TestThermodynamicConsistency:
    """Test thermodynamic consistency of calculations."""

    def test_second_law_not_violated(self, exergy_calculator, hot_stream, cold_stream, heat_exchanger):
        """Exergy output should never exceed input (2nd law)."""
        result = exergy_calculator.analyze_network(
            hot_streams=[hot_stream],
            cold_streams=[cold_stream],
            exchangers=[heat_exchanger],
        )

        assert result.total_exergy_output_kW <= result.total_exergy_input_kW

    def test_exergy_efficiency_decreases_with_delta_t(self, exergy_calculator):
        """Larger temperature differences should reduce exergy efficiency."""
        # Small ΔT exchanger
        hx_small_dt = HeatExchanger(
            exchanger_id="HX-SMALL",
            exchanger_type=ExchangerType.SHELL_AND_TUBE,
            hot_stream_id="H1",
            cold_stream_id="C1",
            duty_kW=100.0,
            hot_inlet_T_C=100.0,
            hot_outlet_T_C=90.0,
            cold_inlet_T_C=80.0,  # Small ΔT
            cold_outlet_T_C=85.0,
            area_m2=10.0,
        )

        # Large ΔT exchanger
        hx_large_dt = HeatExchanger(
            exchanger_id="HX-LARGE",
            exchanger_type=ExchangerType.SHELL_AND_TUBE,
            hot_stream_id="H1",
            cold_stream_id="C1",
            duty_kW=100.0,
            hot_inlet_T_C=200.0,
            hot_outlet_T_C=150.0,
            cold_inlet_T_C=30.0,  # Large ΔT
            cold_outlet_T_C=50.0,
            area_m2=10.0,
        )

        result_small = exergy_calculator.calculate_exchanger_exergy(hx_small_dt)
        result_large = exergy_calculator.calculate_exchanger_exergy(hx_large_dt)

        # Large ΔT should have lower efficiency
        assert result_large.exergy_efficiency < result_small.exergy_efficiency
