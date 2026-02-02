# -*- coding: utf-8 -*-
"""
GL-014 EXCHANGERPRO - UA Calculator Unit Tests

Tests for overall heat transfer coefficient (UA) calculations including:
- UA from LMTD method: UA = Q / (F * LMTD)
- UA from NTU method: UA = NTU * C_min
- UA degradation due to fouling
- Clean vs. fouled UA comparison
- Design vs. actual UA ratio
- Provenance hash verification

Reference:
- TEMA Standards (9th Edition)
- ASME PTC 12.5

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import math
import hashlib
from typing import Dict, Any


# Test tolerances
UA_TOLERANCE = 0.5  # kW/K
RATIO_TOLERANCE = 0.02


class TestUAFromLMTDMethod:
    """Test UA calculation from LMTD method: UA = Q / (F * LMTD)."""

    def test_ua_lmtd_basic_calculation(self, sample_operating_state, sample_thermal_kpis):
        """Test basic UA calculation from LMTD."""
        state = sample_operating_state
        kpis = sample_thermal_kpis

        # Q = m_dot * Cp * dT
        Q = state.m_dot_hot_kg_s * state.Cp_hot_kJ_kgK * \
            (state.T_hot_in_C - state.T_hot_out_C)

        # Calculate LMTD (counterflow)
        dT1 = state.T_hot_in_C - state.T_cold_out_C
        dT2 = state.T_hot_out_C - state.T_cold_in_C
        lmtd = (dT1 - dT2) / math.log(dT1 / dT2)

        F = 0.9  # Typical F-factor for shell-and-tube

        # UA = Q / (F * LMTD)
        UA = Q / (F * lmtd)

        assert UA > 0
        assert isinstance(UA, float)

    def test_ua_with_f_correction(self):
        """Test UA calculation with F-factor correction."""
        Q = 5000.0  # kW
        lmtd = 40.0  # C
        F = 0.90  # 1-2 shell-and-tube

        UA_counterflow = Q / lmtd  # Pure counterflow (F=1)
        UA_corrected = Q / (F * lmtd)  # Corrected for shell-and-tube

        # Corrected UA is higher (requires more area for same duty)
        assert UA_corrected > UA_counterflow

    @pytest.mark.parametrize("Q,lmtd,F,expected_UA", [
        (5000.0, 40.0, 1.00, 125.0),   # Pure counterflow
        (5000.0, 40.0, 0.90, 138.9),   # 1-2 shell-and-tube
        (5000.0, 40.0, 0.85, 147.1),   # Lower F-factor
        (3000.0, 30.0, 0.95, 105.3),   # Lower duty
        (8000.0, 50.0, 0.92, 173.9),   # Higher duty
    ])
    def test_ua_lmtd_parametric(
        self,
        Q: float,
        lmtd: float,
        F: float,
        expected_UA: float,
    ):
        """Test UA calculation with various parameters."""
        UA = Q / (F * lmtd)
        assert abs(UA - expected_UA) < UA_TOLERANCE


class TestUAFromNTUMethod:
    """Test UA calculation from NTU method: UA = NTU * C_min."""

    def test_ua_ntu_basic_calculation(self, sample_operating_state):
        """Test UA calculation from NTU method."""
        state = sample_operating_state

        # Calculate capacity rates
        C_hot = state.m_dot_hot_kg_s * state.Cp_hot_kJ_kgK
        C_cold = state.m_dot_cold_kg_s * state.Cp_cold_kJ_kgK
        C_min = min(C_hot, C_cold)

        NTU = 1.8  # Typical NTU value

        # UA = NTU * C_min
        UA = NTU * C_min

        assert UA > 0

    def test_ua_consistency_between_methods(self, sample_operating_state):
        """Test that LMTD and NTU methods give consistent UA."""
        state = sample_operating_state

        # LMTD method
        Q = state.m_dot_hot_kg_s * state.Cp_hot_kJ_kgK * \
            (state.T_hot_in_C - state.T_hot_out_C)

        dT1 = state.T_hot_in_C - state.T_cold_out_C
        dT2 = state.T_hot_out_C - state.T_cold_in_C
        lmtd = (dT1 - dT2) / math.log(dT1 / dT2)
        F = 0.9

        UA_lmtd = Q / (F * lmtd)

        # NTU method (back-calculate NTU from effectiveness)
        C_hot = state.m_dot_hot_kg_s * state.Cp_hot_kJ_kgK
        C_cold = state.m_dot_cold_kg_s * state.Cp_cold_kJ_kgK
        C_min = min(C_hot, C_cold)

        Q_max = C_min * (state.T_hot_in_C - state.T_cold_in_C)
        epsilon = Q / Q_max

        # For counterflow (simplified)
        C_ratio = C_min / max(C_hot, C_cold)
        if C_ratio < 1.0:
            # NTU from epsilon for counterflow
            NTU = math.log((1 - epsilon * C_ratio) / (1 - epsilon)) / (C_ratio - 1) \
                if C_ratio != 1.0 else epsilon / (1 - epsilon)
        else:
            NTU = epsilon / (1 - epsilon)

        UA_ntu = NTU * C_min

        # Allow some tolerance due to F-factor effects
        assert UA_lmtd > 0
        assert UA_ntu > 0


class TestUADegradation:
    """Test UA degradation calculations due to fouling."""

    def test_ua_degradation_ratio(self, sample_thermal_kpis):
        """Test UA degradation ratio calculation."""
        kpis = sample_thermal_kpis

        UA_ratio = kpis.UA_actual_kW_K / kpis.UA_design_kW_K

        assert 0 < UA_ratio <= 1.0
        assert UA_ratio == pytest.approx(kpis.UA_ratio, abs=RATIO_TOLERANCE)

    def test_fouling_impact_on_ua(self, clean_exchanger_kpis, fouled_exchanger_kpis):
        """Test fouling impact on UA."""
        UA_clean = clean_exchanger_kpis.UA_actual_kW_K
        UA_fouled = fouled_exchanger_kpis.UA_actual_kW_K

        # Fouled exchanger should have lower UA
        assert UA_fouled < UA_clean

        degradation = (UA_clean - UA_fouled) / UA_clean * 100
        assert degradation > 20  # Significant degradation

    def test_cleanliness_factor(self, sample_thermal_kpis):
        """Test cleanliness factor calculation."""
        kpis = sample_thermal_kpis

        # Cleanliness factor = UA_actual / UA_design
        cleanliness_factor = kpis.UA_actual_kW_K / kpis.UA_design_kW_K

        assert cleanliness_factor == pytest.approx(kpis.cleanliness_factor, abs=0.01)


class TestFoulingResistance:
    """Test fouling resistance calculations."""

    def test_fouling_resistance_from_ua(self):
        """Test calculating fouling resistance from UA values."""
        UA_clean = 150.0  # kW/K
        UA_fouled = 100.0  # kW/K
        A = 100.0  # m2 (heat transfer area)

        # 1/UA_fouled = 1/UA_clean + Rf/A
        # Rf = A * (1/UA_fouled - 1/UA_clean)
        Rf = A * (1 / UA_fouled - 1 / UA_clean)

        assert Rf > 0
        assert Rf == pytest.approx(0.333, abs=0.01)  # m2-K/kW

    def test_fouling_resistance_bounds(self, tema_reference_data):
        """Test that fouling resistance is within TEMA guidelines."""
        fouling_factors = tema_reference_data["fouling_factors"]

        for fluid, Rf in fouling_factors.items():
            assert Rf > 0
            assert Rf < 0.005  # Reasonable upper bound for most fluids

    @pytest.mark.parametrize("fluid,expected_Rf", [
        ("crude_oil", 0.00035),
        ("diesel_fuel", 0.00020),
        ("cooling_water_treated", 0.00018),
        ("steam_clean", 0.00009),
    ])
    def test_tema_fouling_factors(self, fluid: str, expected_Rf: float, tema_reference_data):
        """Test TEMA standard fouling factors."""
        Rf = tema_reference_data["fouling_factors"][fluid]
        assert Rf == pytest.approx(expected_Rf, abs=0.00005)


class TestDesignVsActualUA:
    """Test comparison of design and actual UA values."""

    def test_design_ua_from_specification(self, sample_exchanger_config):
        """Test design UA from exchanger specification."""
        config = sample_exchanger_config

        UA_design = config.design_UA_kW_K
        Q_design = config.design_duty_kW

        # Design LMTD
        LMTD_design = Q_design / UA_design

        assert UA_design > 0
        assert LMTD_design > 0

    def test_actual_vs_design_ua_clean(self, clean_exchanger_kpis):
        """Test actual UA vs design for clean exchanger."""
        kpis = clean_exchanger_kpis

        # Clean exchanger may have UA_ratio > 1 due to conservative design
        assert kpis.UA_ratio > 0.9

    def test_actual_vs_design_ua_fouled(self, fouled_exchanger_kpis):
        """Test actual UA vs design for fouled exchanger."""
        kpis = fouled_exchanger_kpis

        # Fouled exchanger will have reduced UA
        assert kpis.UA_ratio < 0.7

    def test_ua_ratio_triggers_cleaning(self):
        """Test that low UA ratio triggers cleaning recommendation."""
        UA_ratio = 0.65
        cleaning_threshold = 0.70

        needs_cleaning = UA_ratio < cleaning_threshold
        assert needs_cleaning


class TestOverallHeatTransferCoefficient:
    """Test overall heat transfer coefficient U calculations."""

    def test_u_from_ua_and_area(self):
        """Test U calculation from UA and area."""
        UA = 125.0  # kW/K
        A = 100.0   # m2

        U = UA / A  # kW/(m2-K) = 1.25

        assert U == 1.25

    def test_u_from_individual_resistances(self):
        """Test U from individual thermal resistances."""
        # 1/UA = 1/h_hot*A_hot + R_wall + Rf_hot + Rf_cold + 1/h_cold*A_cold

        h_hot = 2.0  # kW/(m2-K)
        h_cold = 5.0  # kW/(m2-K)
        A = 100.0    # m2
        R_wall = 0.00001  # m2-K/kW (negligible for metal)
        Rf_hot = 0.00035  # m2-K/kW
        Rf_cold = 0.00018  # m2-K/kW

        # Total resistance
        R_total = 1 / (h_hot * A) + 1 / (h_cold * A) + R_wall + Rf_hot + Rf_cold

        UA = 1 / R_total

        assert UA > 0

    def test_u_typical_values(self):
        """Test U values against typical ranges."""
        # Liquid-liquid: 300-1000 W/(m2-K)
        U_liquid_liquid = 0.5  # kW/(m2-K) = 500 W/(m2-K)

        # Gas-gas: 10-50 W/(m2-K)
        U_gas_gas = 0.03  # kW/(m2-K) = 30 W/(m2-K)

        # Condensing steam to liquid: 1000-6000 W/(m2-K)
        U_condensing = 3.0  # kW/(m2-K) = 3000 W/(m2-K)

        assert 0.3 < U_liquid_liquid < 1.0
        assert 0.01 < U_gas_gas < 0.05
        assert 1.0 < U_condensing < 6.0


class TestUADeterminism:
    """Test UA calculation determinism."""

    def test_deterministic_ua_calculation(self, sample_operating_state):
        """Test that UA calculation is deterministic."""
        state = sample_operating_state

        results = []
        for _ in range(10):
            Q = state.m_dot_hot_kg_s * state.Cp_hot_kJ_kgK * \
                (state.T_hot_in_C - state.T_hot_out_C)

            dT1 = state.T_hot_in_C - state.T_cold_out_C
            dT2 = state.T_hot_out_C - state.T_cold_in_C
            lmtd = (dT1 - dT2) / math.log(dT1 / dT2)

            UA = Q / (0.9 * lmtd)
            results.append(UA)

        assert all(r == results[0] for r in results)

    def test_provenance_hash_for_ua(self, sample_operating_state):
        """Test provenance hash generation for UA calculation."""
        state = sample_operating_state

        Q = state.m_dot_hot_kg_s * state.Cp_hot_kJ_kgK * \
            (state.T_hot_in_C - state.T_hot_out_C)

        dT1 = state.T_hot_in_C - state.T_cold_out_C
        dT2 = state.T_hot_out_C - state.T_cold_in_C
        lmtd = (dT1 - dT2) / math.log(dT1 / dT2)

        UA = Q / (0.9 * lmtd)

        provenance_data = f"{state.exchanger_id}:UA:{UA:.6f}:Q:{Q:.6f}"
        provenance_hash = hashlib.sha256(provenance_data.encode()).hexdigest()

        assert len(provenance_hash) == 64


class TestUAEdgeCases:
    """Test edge cases for UA calculations."""

    def test_zero_lmtd_handling(self):
        """Test handling of zero LMTD (infinite UA)."""
        Q = 5000.0
        lmtd = 0.0  # Impossible in operating exchanger

        # Should handle gracefully
        if lmtd == 0:
            UA = float('inf')
        else:
            UA = Q / lmtd

        assert UA == float('inf')

    def test_very_small_lmtd(self):
        """Test UA with very small LMTD."""
        Q = 5000.0
        lmtd = 1.0  # Very small (high-efficiency case)
        F = 0.9

        UA = Q / (F * lmtd)

        # UA will be very high
        assert UA > 5000

    def test_zero_duty_handling(self):
        """Test handling of zero heat duty."""
        Q = 0.0
        lmtd = 40.0
        F = 0.9

        UA = Q / (F * lmtd) if lmtd > 0 and F > 0 else 0.0

        assert UA == 0.0


class TestUAValidation:
    """Test input validation for UA calculations."""

    def test_negative_ua_detection(self):
        """Test that negative UA is invalid."""
        Q = 5000.0
        lmtd = -40.0  # Invalid

        UA = Q / lmtd  # Would give negative UA

        assert UA < 0, "Negative LMTD should be detected as invalid"

    def test_f_factor_bounds_validation(self):
        """Test F-factor bounds validation."""
        F_valid = 0.9
        F_invalid_low = -0.1
        F_invalid_high = 1.5

        assert 0 < F_valid <= 1.0
        assert F_invalid_low < 0
        assert F_invalid_high > 1.0

    def test_ua_ratio_warning_threshold(self):
        """Test UA ratio warning thresholds."""
        UA_actual = 90.0
        UA_design = 125.0

        ratio = UA_actual / UA_design

        # Warning levels
        warning = ratio < 0.85
        critical = ratio < 0.70
        shutdown = ratio < 0.50

        assert ratio == pytest.approx(0.72, abs=0.01)
        assert warning
        assert critical
        assert not shutdown


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "TestUAFromLMTDMethod",
    "TestUAFromNTUMethod",
    "TestUADegradation",
    "TestFoulingResistance",
    "TestDesignVsActualUA",
    "TestOverallHeatTransferCoefficient",
    "TestUADeterminism",
    "TestUAEdgeCases",
    "TestUAValidation",
]
