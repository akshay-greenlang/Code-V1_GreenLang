"""
GL-006 HEATRECLAIM - LMTD Calculator Tests

Unit tests for LMTD and NTU heat exchanger calculations.
"""

import pytest
import math

from ..calculators.lmtd_calculator import LMTDCalculator
from ..core.config import ExchangerType, FlowArrangement


class TestLMTDCalculation:
    """Tests for LMTD calculation."""

    def test_counterflow_lmtd(self):
        """Test LMTD for counterflow arrangement."""
        calc = LMTDCalculator()

        # Classic counterflow: hot in 100°C out 60°C, cold in 20°C out 80°C
        result = calc.calculate_lmtd(
            T_hot_in=100.0,
            T_hot_out=60.0,
            T_cold_in=20.0,
            T_cold_out=80.0,
            flow_arrangement=FlowArrangement.COUNTERFLOW,
        )

        # ΔT1 = 100-80 = 20, ΔT2 = 60-20 = 40
        # LMTD = (40-20)/ln(40/20) = 20/0.693 = 28.85°C
        expected_lmtd = (40 - 20) / math.log(40 / 20)
        assert abs(result.lmtd_C - expected_lmtd) < 0.1

    def test_parallel_flow_lmtd(self):
        """Test LMTD for parallel flow arrangement."""
        calc = LMTDCalculator()

        # Parallel flow: hot in 100°C out 60°C, cold in 20°C out 50°C
        result = calc.calculate_lmtd(
            T_hot_in=100.0,
            T_hot_out=60.0,
            T_cold_in=20.0,
            T_cold_out=50.0,
            flow_arrangement=FlowArrangement.PARALLEL,
        )

        # ΔT1 = 100-20 = 80, ΔT2 = 60-50 = 10
        # LMTD = (80-10)/ln(80/10) = 70/2.08 = 33.7°C
        expected_lmtd = (80 - 10) / math.log(80 / 10)
        assert abs(result.lmtd_C - expected_lmtd) < 0.1

    def test_equal_terminal_differences(self):
        """Test LMTD when terminal differences are equal."""
        calc = LMTDCalculator()

        # Equal ΔTs: LMTD should equal the terminal difference
        result = calc.calculate_lmtd(
            T_hot_in=100.0,
            T_hot_out=50.0,
            T_cold_in=20.0,
            T_cold_out=70.0,
            flow_arrangement=FlowArrangement.COUNTERFLOW,
        )

        # ΔT1 = 100-70 = 30, ΔT2 = 50-20 = 30
        # When equal, LMTD = ΔT
        assert abs(result.lmtd_C - 30.0) < 0.1

    def test_f_correction_factor(self):
        """Test F correction factor for shell-and-tube."""
        calc = LMTDCalculator()

        result = calc.calculate_lmtd(
            T_hot_in=150.0,
            T_hot_out=90.0,
            T_cold_in=30.0,
            T_cold_out=100.0,
            flow_arrangement=FlowArrangement.SHELL_AND_TUBE_1_2,
        )

        # F factor should be between 0 and 1
        assert 0.5 <= result.F_correction <= 1.0

        # Corrected LMTD should be less than pure counterflow
        pure_counterflow = calc.calculate_lmtd(
            T_hot_in=150.0,
            T_hot_out=90.0,
            T_cold_in=30.0,
            T_cold_out=100.0,
            flow_arrangement=FlowArrangement.COUNTERFLOW,
        )
        assert result.lmtd_C <= pure_counterflow.lmtd_C

    def test_temperature_cross_warning(self):
        """Test warning for temperature cross condition."""
        calc = LMTDCalculator()

        # Temperature cross: cold out > hot out
        result = calc.calculate_lmtd(
            T_hot_in=100.0,
            T_hot_out=50.0,
            T_cold_in=20.0,
            T_cold_out=80.0,  # Cold outlet > hot outlet
            flow_arrangement=FlowArrangement.PARALLEL,
        )

        # Should still calculate but flag issue
        assert result.lmtd_C > 0


class TestNTUMethod:
    """Tests for NTU-effectiveness method."""

    def test_ntu_counterflow(self):
        """Test NTU calculation for counterflow."""
        calc = LMTDCalculator()

        result = calc.calculate_ntu_effectiveness(
            C_min=2000.0,  # W/K
            C_max=4000.0,  # W/K
            UA=5000.0,  # W/K
            flow_arrangement=FlowArrangement.COUNTERFLOW,
        )

        assert result.ntu > 0
        assert 0 < result.effectiveness <= 1.0
        assert result.C_ratio == 0.5

    def test_ntu_parallel_flow(self):
        """Test NTU calculation for parallel flow."""
        calc = LMTDCalculator()

        result = calc.calculate_ntu_effectiveness(
            C_min=2000.0,
            C_max=4000.0,
            UA=5000.0,
            flow_arrangement=FlowArrangement.PARALLEL,
        )

        # Parallel flow always has lower effectiveness than counterflow
        counterflow_result = calc.calculate_ntu_effectiveness(
            C_min=2000.0,
            C_max=4000.0,
            UA=5000.0,
            flow_arrangement=FlowArrangement.COUNTERFLOW,
        )

        assert result.effectiveness <= counterflow_result.effectiveness

    def test_ntu_high_value(self):
        """Test behavior at high NTU."""
        calc = LMTDCalculator()

        result = calc.calculate_ntu_effectiveness(
            C_min=1000.0,
            C_max=2000.0,
            UA=50000.0,  # Very high UA
            flow_arrangement=FlowArrangement.COUNTERFLOW,
        )

        # At very high NTU, effectiveness approaches maximum
        assert result.effectiveness > 0.9
        assert result.ntu > 10

    def test_balanced_exchanger(self):
        """Test balanced exchanger (C_ratio = 1)."""
        calc = LMTDCalculator()

        result = calc.calculate_ntu_effectiveness(
            C_min=2000.0,
            C_max=2000.0,  # Balanced
            UA=4000.0,
            flow_arrangement=FlowArrangement.COUNTERFLOW,
        )

        assert result.C_ratio == 1.0
        # For balanced counterflow: ε = NTU/(1+NTU)
        expected_eff = result.ntu / (1 + result.ntu)
        assert abs(result.effectiveness - expected_eff) < 0.01


class TestExchangerSizing:
    """Tests for exchanger sizing calculations."""

    def test_area_calculation(self):
        """Test heat exchanger area calculation."""
        calc = LMTDCalculator()

        result = calc.size_exchanger(
            duty_kW=500.0,
            T_hot_in=150.0,
            T_hot_out=90.0,
            T_cold_in=30.0,
            T_cold_out=100.0,
            U_W_m2K=500.0,  # Typical U for liquid-liquid
        )

        # Q = U * A * LMTD → A = Q / (U * LMTD)
        assert result.area_m2 > 0
        assert result.lmtd_C > 0

        # Verify: Area should be reasonable for 500 kW duty
        # A = 500,000 W / (500 W/m²K * ~40°C) ≈ 25 m²
        assert 10 < result.area_m2 < 100

    def test_overall_u_estimation(self):
        """Test overall U coefficient estimation."""
        calc = LMTDCalculator()

        # Liquid-liquid in shell and tube
        U = calc.estimate_overall_U(
            exchanger_type=ExchangerType.SHELL_AND_TUBE,
            hot_fluid="Water",
            cold_fluid="Water",
            hot_pressure_kPa=500.0,
            cold_pressure_kPa=300.0,
        )

        # Liquid-liquid U typically 300-1000 W/m²K
        assert 200 < U < 1500

        # Gas-gas should have much lower U
        U_gas = calc.estimate_overall_U(
            exchanger_type=ExchangerType.SHELL_AND_TUBE,
            hot_fluid="Air",
            cold_fluid="Air",
            hot_pressure_kPa=200.0,
            cold_pressure_kPa=101.0,
        )

        assert U_gas < U  # Gas U should be lower

    def test_fouling_factor_effect(self):
        """Test effect of fouling factors."""
        calc = LMTDCalculator()

        # Without fouling
        result_clean = calc.size_exchanger(
            duty_kW=500.0,
            T_hot_in=150.0,
            T_hot_out=90.0,
            T_cold_in=30.0,
            T_cold_out=100.0,
            U_W_m2K=500.0,
            fouling_factor_hot=0.0,
            fouling_factor_cold=0.0,
        )

        # With fouling
        result_fouled = calc.size_exchanger(
            duty_kW=500.0,
            T_hot_in=150.0,
            T_hot_out=90.0,
            T_cold_in=30.0,
            T_cold_out=100.0,
            U_W_m2K=500.0,
            fouling_factor_hot=0.0002,
            fouling_factor_cold=0.0002,
        )

        # Fouled exchanger needs more area
        assert result_fouled.area_m2 > result_clean.area_m2


class TestDeterminism:
    """Tests for calculation determinism."""

    def test_lmtd_deterministic(self):
        """Test LMTD calculation is deterministic."""
        calc = LMTDCalculator()

        results = []
        for _ in range(5):
            result = calc.calculate_lmtd(
                T_hot_in=120.0,
                T_hot_out=70.0,
                T_cold_in=25.0,
                T_cold_out=85.0,
            )
            results.append(result.lmtd_C)

        # All results should be identical
        assert all(r == results[0] for r in results)

    def test_provenance_hash(self):
        """Test provenance hash is generated."""
        calc = LMTDCalculator()

        result = calc.size_exchanger(
            duty_kW=500.0,
            T_hot_in=150.0,
            T_hot_out=90.0,
            T_cold_in=30.0,
            T_cold_out=100.0,
            U_W_m2K=500.0,
        )

        assert result.computation_hash is not None
        assert len(result.computation_hash) == 64  # SHA-256
