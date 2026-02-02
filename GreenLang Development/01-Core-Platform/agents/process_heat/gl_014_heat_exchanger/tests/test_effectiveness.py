# -*- coding: utf-8 -*-
"""
GL-014 EXCHANGER-PRO Agent - Effectiveness Tests

Comprehensive tests for thermal effectiveness calculations including:
- LMTD (Log Mean Temperature Difference) calculations
- Epsilon-NTU method for all flow arrangements
- F-factor corrections for multi-pass exchangers
- Thermal effectiveness validation against known values

Coverage Target: 90%+

References:
    - Kays & London, "Compact Heat Exchangers" (3rd Ed.)
    - HEDH Heat Exchanger Design Handbook
    - Incropera & DeWitt, "Fundamentals of Heat Transfer"
"""

import pytest
import math
from unittest.mock import Mock, patch

from greenlang.agents.process_heat.gl_014_heat_exchanger.effectiveness import (
    EffectivenessNTUCalculator,
    ThermalAnalysisInput,
)
from greenlang.agents.process_heat.gl_014_heat_exchanger.config import (
    FlowArrangement,
)


class TestLMTDCalculations:
    """Tests for Log Mean Temperature Difference calculations."""

    @pytest.fixture
    def calculator(self):
        """Create EffectivenessNTUCalculator instance."""
        return EffectivenessNTUCalculator()

    def test_lmtd_counter_flow(self, calculator):
        """Test LMTD calculation for counter flow."""
        # Counter flow: Hot in=150C, out=90C; Cold in=30C, out=80C
        # dT1 = 150 - 80 = 70, dT2 = 90 - 30 = 60
        # LMTD = (70 - 60) / ln(70/60) = 10 / ln(1.167) = 64.9C
        lmtd = calculator.calculate_lmtd(
            hot_inlet_c=150.0,
            hot_outlet_c=90.0,
            cold_inlet_c=30.0,
            cold_outlet_c=80.0,
            flow_arrangement=FlowArrangement.COUNTER_FLOW,
        )
        expected = (70 - 60) / math.log(70 / 60)
        assert lmtd == pytest.approx(expected, rel=0.01)

    def test_lmtd_parallel_flow(self, calculator):
        """Test LMTD calculation for parallel flow."""
        # Parallel flow: Hot in=150C, out=90C; Cold in=30C, out=70C
        # dT1 = 150 - 30 = 120, dT2 = 90 - 70 = 20
        # LMTD = (120 - 20) / ln(120/20) = 100 / ln(6) = 55.8C
        lmtd = calculator.calculate_lmtd(
            hot_inlet_c=150.0,
            hot_outlet_c=90.0,
            cold_inlet_c=30.0,
            cold_outlet_c=70.0,
            flow_arrangement=FlowArrangement.PARALLEL_FLOW,
        )
        expected = (120 - 20) / math.log(120 / 20)
        assert lmtd == pytest.approx(expected, rel=0.01)

    def test_lmtd_equal_delta_t(self, calculator):
        """Test LMTD when dT1 equals dT2 (special case)."""
        # When dT1 = dT2, LMTD = dT1 = dT2
        lmtd = calculator.calculate_lmtd(
            hot_inlet_c=100.0,
            hot_outlet_c=60.0,
            cold_inlet_c=30.0,
            cold_outlet_c=70.0,  # dT1 = 100-70=30, dT2 = 60-30=30
            flow_arrangement=FlowArrangement.PARALLEL_FLOW,
        )
        assert lmtd == pytest.approx(30.0, rel=0.01)

    def test_lmtd_near_equal_delta_t(self, calculator):
        """Test LMTD when dT1 is nearly equal to dT2."""
        # Should handle numerical stability
        lmtd = calculator.calculate_lmtd(
            hot_inlet_c=100.0,
            hot_outlet_c=60.0,
            cold_inlet_c=30.0,
            cold_outlet_c=69.9,  # dT1 = 30.1, dT2 = 30
            flow_arrangement=FlowArrangement.PARALLEL_FLOW,
        )
        assert lmtd == pytest.approx(30.0, rel=0.02)

    def test_lmtd_temperature_cross_error(self, calculator):
        """Test LMTD raises error for temperature cross."""
        # Temperature cross: cold outlet > hot outlet is allowed
        # but cold outlet > hot inlet is invalid
        with pytest.raises(ValueError):
            calculator.calculate_lmtd(
                hot_inlet_c=80.0,
                hot_outlet_c=60.0,
                cold_inlet_c=50.0,
                cold_outlet_c=90.0,  # Invalid: cold out > hot in
                flow_arrangement=FlowArrangement.COUNTER_FLOW,
            )

    def test_lmtd_negative_delta_t_error(self, calculator):
        """Test LMTD raises error for negative temperature difference."""
        with pytest.raises(ValueError):
            calculator.calculate_lmtd(
                hot_inlet_c=50.0,  # Hot is colder than cold
                hot_outlet_c=40.0,
                cold_inlet_c=60.0,
                cold_outlet_c=70.0,
                flow_arrangement=FlowArrangement.COUNTER_FLOW,
            )


class TestFFactorCorrection:
    """Tests for F-factor (LMTD correction factor) calculations."""

    @pytest.fixture
    def calculator(self):
        """Create EffectivenessNTUCalculator instance."""
        return EffectivenessNTUCalculator()

    def test_f_factor_counter_flow(self, calculator):
        """Test F-factor is 1.0 for pure counter flow."""
        f = calculator.calculate_f_factor(
            hot_inlet_c=150.0,
            hot_outlet_c=90.0,
            cold_inlet_c=30.0,
            cold_outlet_c=80.0,
            flow_arrangement=FlowArrangement.COUNTER_FLOW,
        )
        assert f == pytest.approx(1.0, rel=0.001)

    def test_f_factor_parallel_flow(self, calculator):
        """Test F-factor is 1.0 for pure parallel flow."""
        f = calculator.calculate_f_factor(
            hot_inlet_c=150.0,
            hot_outlet_c=90.0,
            cold_inlet_c=30.0,
            cold_outlet_c=70.0,
            flow_arrangement=FlowArrangement.PARALLEL_FLOW,
        )
        assert f == pytest.approx(1.0, rel=0.001)

    def test_f_factor_one_shell_two_tube(self, calculator):
        """Test F-factor for 1 shell pass, 2 tube passes."""
        # R = (Th,in - Th,out) / (Tc,out - Tc,in)
        # P = (Tc,out - Tc,in) / (Th,in - Tc,in)
        f = calculator.calculate_f_factor(
            hot_inlet_c=150.0,
            hot_outlet_c=90.0,
            cold_inlet_c=30.0,
            cold_outlet_c=80.0,
            flow_arrangement=FlowArrangement.ONE_SHELL_EVEN_TUBE,
        )
        # F should be less than 1.0 for multi-pass
        assert 0.7 < f < 1.0

    def test_f_factor_shell_tube_typical(self, calculator):
        """Test F-factor for typical shell-tube arrangement."""
        f = calculator.calculate_f_factor(
            hot_inlet_c=200.0,
            hot_outlet_c=120.0,
            cold_inlet_c=50.0,
            cold_outlet_c=100.0,
            flow_arrangement=FlowArrangement.ONE_SHELL_EVEN_TUBE,
        )
        # Typical F factors are 0.75-0.95
        assert 0.75 < f < 1.0

    def test_f_factor_bounds_check(self, calculator):
        """Test F-factor is always between 0 and 1."""
        test_cases = [
            (200, 150, 30, 80),
            (300, 200, 50, 120),
            (150, 100, 20, 60),
            (180, 90, 40, 100),
        ]
        for h_in, h_out, c_in, c_out in test_cases:
            f = calculator.calculate_f_factor(
                hot_inlet_c=h_in,
                hot_outlet_c=h_out,
                cold_inlet_c=c_in,
                cold_outlet_c=c_out,
                flow_arrangement=FlowArrangement.ONE_SHELL_EVEN_TUBE,
            )
            assert 0 < f <= 1.0, f"F-factor {f} out of bounds for {h_in}/{h_out}/{c_in}/{c_out}"


class TestNTUCalculations:
    """Tests for NTU (Number of Transfer Units) calculations."""

    @pytest.fixture
    def calculator(self):
        """Create EffectivenessNTUCalculator instance."""
        return EffectivenessNTUCalculator()

    def test_ntu_from_u_area(self, calculator):
        """Test NTU calculation from U*A."""
        # NTU = U*A / C_min
        # U = 500 W/m2K, A = 100 m2, C_min = 10000 W/K
        ntu = calculator.calculate_ntu(
            u_w_m2k=500.0,
            area_m2=100.0,
            c_min_w_k=10000.0,
        )
        expected = 500 * 100 / 10000
        assert ntu == pytest.approx(expected, rel=0.001)

    def test_ntu_high_value(self, calculator):
        """Test NTU calculation for high NTU (large exchanger)."""
        ntu = calculator.calculate_ntu(
            u_w_m2k=500.0,
            area_m2=500.0,
            c_min_w_k=10000.0,
        )
        assert ntu == 25.0

    def test_ntu_low_value(self, calculator):
        """Test NTU calculation for low NTU (small exchanger)."""
        ntu = calculator.calculate_ntu(
            u_w_m2k=200.0,
            area_m2=20.0,
            c_min_w_k=10000.0,
        )
        assert ntu == pytest.approx(0.4, rel=0.001)


class TestHeatCapacityRatio:
    """Tests for heat capacity ratio calculations."""

    @pytest.fixture
    def calculator(self):
        """Create EffectivenessNTUCalculator instance."""
        return EffectivenessNTUCalculator()

    def test_heat_capacity_ratio_balanced(self, calculator):
        """Test Cr = 1 for balanced flows."""
        cr = calculator.calculate_capacity_ratio(
            hot_flow_kg_s=10.0,
            hot_cp_j_kgk=4180.0,
            cold_flow_kg_s=10.0,
            cold_cp_j_kgk=4180.0,
        )
        assert cr == pytest.approx(1.0, rel=0.001)

    def test_heat_capacity_ratio_unbalanced(self, calculator):
        """Test Cr < 1 for unbalanced flows."""
        # Hot side: 10 kg/s * 2000 J/kgK = 20000 W/K
        # Cold side: 10 kg/s * 4180 J/kgK = 41800 W/K
        # Cr = Cmin/Cmax = 20000/41800 = 0.478
        cr = calculator.calculate_capacity_ratio(
            hot_flow_kg_s=10.0,
            hot_cp_j_kgk=2000.0,
            cold_flow_kg_s=10.0,
            cold_cp_j_kgk=4180.0,
        )
        expected = 20000 / 41800
        assert cr == pytest.approx(expected, rel=0.01)

    def test_heat_capacity_ratio_phase_change(self, calculator):
        """Test Cr = 0 for phase change (infinite capacity)."""
        # When one side has phase change, treat as Cr = 0
        cr = calculator.calculate_capacity_ratio(
            hot_flow_kg_s=10.0,
            hot_cp_j_kgk=4180.0,
            cold_flow_kg_s=10.0,
            cold_cp_j_kgk=float('inf'),  # Phase change
        )
        assert cr == pytest.approx(0.0, abs=0.01)


class TestEffectivenessCalculations:
    """Tests for thermal effectiveness calculations using e-NTU method."""

    @pytest.fixture
    def calculator(self):
        """Create EffectivenessNTUCalculator instance."""
        return EffectivenessNTUCalculator()

    @pytest.mark.parametrize("ntu,cr,expected_eff", [
        (0.5, 0.5, 0.3934),
        (1.0, 0.5, 0.5934),
        (2.0, 0.5, 0.7754),
        (3.0, 0.5, 0.8647),
    ])
    def test_effectiveness_counter_flow(self, calculator, ntu, cr, expected_eff):
        """Test effectiveness for counter flow (Kays & London values)."""
        eff = calculator.calculate_effectiveness(
            ntu=ntu,
            cr=cr,
            flow_arrangement=FlowArrangement.COUNTER_FLOW,
        )
        assert eff == pytest.approx(expected_eff, rel=0.02)

    def test_effectiveness_counter_flow_cr_1(self, calculator):
        """Test effectiveness for counter flow with Cr=1."""
        # epsilon = NTU / (1 + NTU) when Cr = 1
        ntu = 2.0
        expected = ntu / (1 + ntu)
        eff = calculator.calculate_effectiveness(
            ntu=ntu,
            cr=1.0,
            flow_arrangement=FlowArrangement.COUNTER_FLOW,
        )
        assert eff == pytest.approx(expected, rel=0.01)

    def test_effectiveness_parallel_flow(self, calculator):
        """Test effectiveness for parallel flow."""
        # epsilon = [1 - exp(-NTU*(1+Cr))] / (1 + Cr)
        ntu = 1.0
        cr = 0.5
        expected = (1 - math.exp(-ntu * (1 + cr))) / (1 + cr)
        eff = calculator.calculate_effectiveness(
            ntu=ntu,
            cr=cr,
            flow_arrangement=FlowArrangement.PARALLEL_FLOW,
        )
        assert eff == pytest.approx(expected, rel=0.02)

    def test_effectiveness_phase_change_cr_0(self, calculator):
        """Test effectiveness for Cr=0 (phase change)."""
        # epsilon = 1 - exp(-NTU) when Cr = 0
        ntu = 1.0
        expected = 1 - math.exp(-ntu)  # 0.6321
        eff = calculator.calculate_effectiveness(
            ntu=ntu,
            cr=0.0,
            flow_arrangement=FlowArrangement.COUNTER_FLOW,
        )
        assert eff == pytest.approx(expected, rel=0.01)

    def test_effectiveness_high_ntu(self, calculator):
        """Test effectiveness approaches maximum at high NTU."""
        # For counter flow with Cr < 1, max effectiveness -> 1.0 as NTU -> inf
        eff = calculator.calculate_effectiveness(
            ntu=100.0,
            cr=0.5,
            flow_arrangement=FlowArrangement.COUNTER_FLOW,
        )
        assert eff > 0.99

    def test_effectiveness_shell_tube_1_2(self, calculator):
        """Test effectiveness for 1-2 shell-tube exchanger."""
        # Standard 1 shell, 2 tube pass correlation
        ntu = 2.0
        cr = 0.5
        eff = calculator.calculate_effectiveness(
            ntu=ntu,
            cr=cr,
            flow_arrangement=FlowArrangement.ONE_SHELL_EVEN_TUBE,
        )
        # Should be between parallel and counter flow
        eff_parallel = calculator.calculate_effectiveness(
            ntu=ntu, cr=cr, flow_arrangement=FlowArrangement.PARALLEL_FLOW
        )
        eff_counter = calculator.calculate_effectiveness(
            ntu=ntu, cr=cr, flow_arrangement=FlowArrangement.COUNTER_FLOW
        )
        assert eff_parallel < eff < eff_counter

    def test_effectiveness_cross_flow_both_unmixed(self, calculator):
        """Test effectiveness for cross flow with both fluids unmixed."""
        ntu = 2.0
        cr = 0.5
        eff = calculator.calculate_effectiveness(
            ntu=ntu,
            cr=cr,
            flow_arrangement=FlowArrangement.CROSS_FLOW_UNMIXED,
        )
        # Cross flow unmixed is between parallel and counter
        assert 0.5 < eff < 0.9

    def test_effectiveness_bounds(self, calculator):
        """Test effectiveness is always between 0 and 1."""
        test_cases = [
            (0.1, 0.1),
            (0.5, 0.5),
            (1.0, 1.0),
            (5.0, 0.5),
            (10.0, 0.8),
        ]
        for ntu, cr in test_cases:
            for arrangement in [
                FlowArrangement.COUNTER_FLOW,
                FlowArrangement.PARALLEL_FLOW,
                FlowArrangement.CROSS_FLOW_UNMIXED,
            ]:
                eff = calculator.calculate_effectiveness(
                    ntu=ntu, cr=cr, flow_arrangement=arrangement
                )
                assert 0 <= eff <= 1.0, f"eff={eff} for NTU={ntu}, Cr={cr}"


class TestThermalAnalysis:
    """Tests for complete thermal analysis."""

    @pytest.fixture
    def calculator(self):
        """Create EffectivenessNTUCalculator instance."""
        return EffectivenessNTUCalculator()

    def test_thermal_analysis_complete(self, calculator):
        """Test complete thermal analysis workflow."""
        thermal_input = ThermalAnalysisInput(
            hot_inlet_temp_c=150.0,
            hot_outlet_temp_c=100.0,
            hot_mass_flow_kg_s=10.0,
            hot_cp_kj_kgk=2.1,  # Oil
            cold_inlet_temp_c=30.0,
            cold_outlet_temp_c=70.0,
            cold_mass_flow_kg_s=15.0,
            cold_cp_kj_kgk=4.18,  # Water
            heat_transfer_area_m2=100.0,
            flow_arrangement=FlowArrangement.COUNTER_FLOW,
        )

        result = calculator.analyze_thermal_performance(
            thermal_input,
            design_u_w_m2k=500.0,
        )

        # Verify results
        assert result.q_actual_kw > 0
        assert result.lmtd_c > 0
        assert 0 < result.effectiveness < 1
        assert result.u_required_w_m2k > 0

    def test_thermal_analysis_heat_balance(self, calculator):
        """Test heat balance in thermal analysis."""
        thermal_input = ThermalAnalysisInput(
            hot_inlet_temp_c=150.0,
            hot_outlet_temp_c=100.0,
            hot_mass_flow_kg_s=10.0,
            hot_cp_kj_kgk=4.0,
            cold_inlet_temp_c=30.0,
            cold_outlet_temp_c=80.0,
            cold_mass_flow_kg_s=10.0,
            cold_cp_kj_kgk=4.0,
            heat_transfer_area_m2=100.0,
            flow_arrangement=FlowArrangement.COUNTER_FLOW,
        )

        result = calculator.analyze_thermal_performance(
            thermal_input,
            design_u_w_m2k=500.0,
        )

        # Q_hot = m_h * Cp_h * (Th_in - Th_out)
        q_hot = 10.0 * 4.0 * (150.0 - 100.0)  # 2000 kW
        # Q_cold = m_c * Cp_c * (Tc_out - Tc_in)
        q_cold = 10.0 * 4.0 * (80.0 - 30.0)  # 2000 kW

        # Heat balance should be close
        assert result.q_actual_kw == pytest.approx(q_hot, rel=0.01)
        assert abs(q_hot - q_cold) < 1.0  # Perfect balance in this case

    def test_thermal_analysis_u_calculation(self, calculator):
        """Test U value calculation from thermal analysis."""
        thermal_input = ThermalAnalysisInput(
            hot_inlet_temp_c=150.0,
            hot_outlet_temp_c=100.0,
            hot_mass_flow_kg_s=10.0,
            hot_cp_kj_kgk=4.18,
            cold_inlet_temp_c=30.0,
            cold_outlet_temp_c=70.0,
            cold_mass_flow_kg_s=10.0,
            cold_cp_kj_kgk=4.18,
            heat_transfer_area_m2=100.0,
            flow_arrangement=FlowArrangement.COUNTER_FLOW,
        )

        result = calculator.analyze_thermal_performance(
            thermal_input,
            design_u_w_m2k=500.0,
        )

        # U = Q / (A * LMTD)
        expected_u = result.q_actual_kw * 1000 / (100.0 * result.lmtd_c)
        assert result.u_required_w_m2k == pytest.approx(expected_u, rel=0.01)


class TestFoulingFromU:
    """Tests for fouling factor calculation from U values."""

    @pytest.fixture
    def calculator(self):
        """Create EffectivenessNTUCalculator instance."""
        return EffectivenessNTUCalculator()

    def test_fouling_from_u_values(self, calculator):
        """Test fouling factor calculation from clean and current U."""
        # Rf = 1/U_fouled - 1/U_clean
        u_clean = 500.0
        u_fouled = 400.0
        rf = calculator.calculate_fouling_from_u(u_clean, u_fouled)

        expected_rf = 1/400.0 - 1/500.0  # 0.0005 m2K/W
        assert rf == pytest.approx(expected_rf, rel=0.01)

    def test_fouling_from_u_clean(self, calculator):
        """Test fouling is zero when U is at clean value."""
        u_clean = 500.0
        rf = calculator.calculate_fouling_from_u(u_clean, u_clean)
        assert rf == pytest.approx(0.0, abs=1e-10)

    def test_fouling_from_u_heavily_fouled(self, calculator):
        """Test fouling calculation for heavily fouled exchanger."""
        u_clean = 500.0
        u_fouled = 250.0  # 50% reduction
        rf = calculator.calculate_fouling_from_u(u_clean, u_fouled)

        expected_rf = 1/250.0 - 1/500.0  # 0.002 m2K/W
        assert rf == pytest.approx(expected_rf, rel=0.01)


class TestEffectivenessFromMeasurements:
    """Tests for effectiveness calculated from temperature measurements."""

    @pytest.fixture
    def calculator(self):
        """Create EffectivenessNTUCalculator instance."""
        return EffectivenessNTUCalculator()

    def test_effectiveness_from_temperatures(self, calculator):
        """Test effectiveness from temperature measurements."""
        # epsilon = Q_actual / Q_max
        # Q_actual = Cmin * (T_hot,in - T_cold,in) * epsilon

        hot_in = 150.0
        hot_out = 100.0
        cold_in = 30.0
        cold_out = 70.0

        # C_hot = m * Cp; C_cold = m * Cp
        # Assume same capacity rates: Cmin = C_hot = C_cold
        # epsilon = (Th_in - Th_out) / (Th_in - Tc_in) if C_hot = Cmin
        # epsilon = (Tc_out - Tc_in) / (Th_in - Tc_in) if C_cold = Cmin

        eff = calculator.calculate_effectiveness_from_temps(
            hot_inlet_c=hot_in,
            hot_outlet_c=hot_out,
            cold_inlet_c=cold_in,
            cold_outlet_c=cold_out,
            c_hot_w_k=10000.0,
            c_cold_w_k=15000.0,
        )

        # C_min = 10000 (hot side)
        # epsilon = (150-100)/(150-30) = 50/120 = 0.417
        expected = (hot_in - hot_out) / (hot_in - cold_in)
        assert eff == pytest.approx(expected, rel=0.01)

    def test_effectiveness_from_temps_cold_limited(self, calculator):
        """Test effectiveness when cold side is capacity limited."""
        hot_in = 150.0
        hot_out = 80.0
        cold_in = 30.0
        cold_out = 100.0

        eff = calculator.calculate_effectiveness_from_temps(
            hot_inlet_c=hot_in,
            hot_outlet_c=hot_out,
            cold_inlet_c=cold_in,
            cold_outlet_c=cold_out,
            c_hot_w_k=20000.0,  # Hot has higher capacity
            c_cold_w_k=10000.0,  # Cold is Cmin
        )

        # C_min = 10000 (cold side)
        # epsilon = (Tc_out - Tc_in) / (Th_in - Tc_in)
        expected = (cold_out - cold_in) / (hot_in - cold_in)
        assert eff == pytest.approx(expected, rel=0.01)


class TestNTUFromEffectiveness:
    """Tests for NTU calculation from known effectiveness."""

    @pytest.fixture
    def calculator(self):
        """Create EffectivenessNTUCalculator instance."""
        return EffectivenessNTUCalculator()

    def test_ntu_from_effectiveness_counter_flow(self, calculator):
        """Test inverse NTU calculation for counter flow."""
        # Calculate effectiveness for known NTU, then reverse
        ntu_original = 2.0
        cr = 0.5

        eff = calculator.calculate_effectiveness(
            ntu=ntu_original,
            cr=cr,
            flow_arrangement=FlowArrangement.COUNTER_FLOW,
        )

        ntu_calculated = calculator.calculate_ntu_from_effectiveness(
            effectiveness=eff,
            cr=cr,
            flow_arrangement=FlowArrangement.COUNTER_FLOW,
        )

        assert ntu_calculated == pytest.approx(ntu_original, rel=0.02)

    def test_ntu_from_effectiveness_cr_0(self, calculator):
        """Test inverse NTU for Cr=0 (phase change)."""
        # epsilon = 1 - exp(-NTU)
        # NTU = -ln(1 - epsilon)
        eff = 0.6321
        ntu = calculator.calculate_ntu_from_effectiveness(
            effectiveness=eff,
            cr=0.0,
            flow_arrangement=FlowArrangement.COUNTER_FLOW,
        )
        expected = -math.log(1 - eff)
        assert ntu == pytest.approx(expected, rel=0.02)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.fixture
    def calculator(self):
        """Create EffectivenessNTUCalculator instance."""
        return EffectivenessNTUCalculator()

    def test_zero_ntu(self, calculator):
        """Test effectiveness is 0 when NTU is 0."""
        eff = calculator.calculate_effectiveness(
            ntu=0.0,
            cr=0.5,
            flow_arrangement=FlowArrangement.COUNTER_FLOW,
        )
        assert eff == pytest.approx(0.0, abs=0.001)

    def test_very_small_delta_t(self, calculator):
        """Test LMTD handles very small temperature differences."""
        lmtd = calculator.calculate_lmtd(
            hot_inlet_c=100.0,
            hot_outlet_c=99.5,
            cold_inlet_c=30.0,
            cold_outlet_c=30.3,
            flow_arrangement=FlowArrangement.COUNTER_FLOW,
        )
        assert lmtd > 0
        assert lmtd < 100

    def test_large_temperature_range(self, calculator):
        """Test calculations with large temperature range."""
        thermal_input = ThermalAnalysisInput(
            hot_inlet_temp_c=500.0,
            hot_outlet_temp_c=200.0,
            hot_mass_flow_kg_s=5.0,
            hot_cp_kj_kgk=1.0,  # Gas
            cold_inlet_temp_c=50.0,
            cold_outlet_temp_c=250.0,
            cold_mass_flow_kg_s=5.0,
            cold_cp_kj_kgk=1.5,
            heat_transfer_area_m2=200.0,
            flow_arrangement=FlowArrangement.COUNTER_FLOW,
        )

        result = calculator.analyze_thermal_performance(
            thermal_input,
            design_u_w_m2k=100.0,
        )

        assert result.q_actual_kw > 0
        assert result.effectiveness > 0
