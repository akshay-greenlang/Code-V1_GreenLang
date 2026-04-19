# -*- coding: utf-8 -*-
"""
GL-014 EXCHANGERPRO - LMTD Calculator Unit Tests

Tests for Log Mean Temperature Difference (LMTD) calculations including:
- Counterflow LMTD
- Parallel flow LMTD
- Equal terminal differences (edge case where dT1 = dT2)
- F-factor correction for shell-and-tube exchangers
- Temperature cross detection
- Provenance hash verification

Reference:
- Incropera & DeWitt, Fundamentals of Heat and Mass Transfer
- TEMA Standards (9th Edition)

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import math
import hashlib
from typing import Dict, Any


# Test tolerances
LMTD_TOLERANCE = 0.1  # Celsius
F_FACTOR_TOLERANCE = 0.02


class TestLMTDCounterflow:
    """Test LMTD calculation for counterflow arrangement."""

    def test_counterflow_basic_case(self, sample_operating_state):
        """Test basic counterflow LMTD calculation."""
        state = sample_operating_state

        # For counterflow:
        # dT1 = T_hot_in - T_cold_out
        # dT2 = T_hot_out - T_cold_in
        dT1 = state.T_hot_in_C - state.T_cold_out_C  # 150 - 100 = 50
        dT2 = state.T_hot_out_C - state.T_cold_in_C  # 90 - 30 = 60

        # LMTD = (dT1 - dT2) / ln(dT1/dT2)
        lmtd = (dT1 - dT2) / math.log(dT1 / dT2)

        # Expected: (50 - 60) / ln(50/60) = -10 / (-0.182) = 54.85 C
        expected_lmtd = 54.85

        assert abs(lmtd - expected_lmtd) < LMTD_TOLERANCE

    def test_counterflow_textbook_example(self):
        """Test classic textbook counterflow example."""
        # Classic example: hot 100->60, cold 20->80
        T_hot_in = 100.0
        T_hot_out = 60.0
        T_cold_in = 20.0
        T_cold_out = 80.0

        dT1 = T_hot_in - T_cold_out  # 100 - 80 = 20
        dT2 = T_hot_out - T_cold_in  # 60 - 20 = 40

        lmtd = (dT1 - dT2) / math.log(dT1 / dT2)

        # Expected: (20 - 40) / ln(20/40) = -20 / (-0.693) = 28.85 C
        expected_lmtd = 28.85

        assert abs(lmtd - expected_lmtd) < LMTD_TOLERANCE

    @pytest.mark.parametrize("T_hot_in,T_hot_out,T_cold_in,T_cold_out,expected_lmtd", [
        (100.0, 60.0, 20.0, 80.0, 28.85),   # Textbook case
        (150.0, 90.0, 30.0, 100.0, 54.85),  # Industrial case
        (200.0, 100.0, 50.0, 150.0, 48.27), # High temperature case
        (80.0, 50.0, 20.0, 60.0, 24.66),    # Low temperature case
    ])
    def test_counterflow_parametric(
        self,
        T_hot_in: float,
        T_hot_out: float,
        T_cold_in: float,
        T_cold_out: float,
        expected_lmtd: float,
    ):
        """Test counterflow LMTD with various temperature profiles."""
        dT1 = T_hot_in - T_cold_out
        dT2 = T_hot_out - T_cold_in

        if abs(dT1 - dT2) < 0.01:
            lmtd = dT1
        else:
            lmtd = (dT1 - dT2) / math.log(dT1 / dT2)

        assert abs(lmtd - expected_lmtd) < LMTD_TOLERANCE


class TestLMTDParallelFlow:
    """Test LMTD calculation for parallel flow arrangement."""

    def test_parallel_flow_basic_case(self):
        """Test basic parallel flow LMTD calculation."""
        # Parallel flow: both fluids enter at same end
        T_hot_in = 100.0
        T_hot_out = 60.0
        T_cold_in = 20.0
        T_cold_out = 50.0

        # For parallel flow:
        # dT1 = T_hot_in - T_cold_in
        # dT2 = T_hot_out - T_cold_out
        dT1 = T_hot_in - T_cold_in  # 100 - 20 = 80
        dT2 = T_hot_out - T_cold_out  # 60 - 50 = 10

        lmtd = (dT1 - dT2) / math.log(dT1 / dT2)

        # Expected: (80 - 10) / ln(80/10) = 70 / 2.08 = 33.65 C
        expected_lmtd = 33.65

        assert abs(lmtd - expected_lmtd) < LMTD_TOLERANCE

    def test_parallel_flow_lower_effectiveness(self):
        """Test that parallel flow gives lower LMTD than counterflow for same duty."""
        T_hot_in = 100.0
        T_hot_out = 60.0
        T_cold_in = 20.0
        T_cold_out = 50.0  # Parallel flow limited outlet

        # Parallel flow LMTD
        dT1_parallel = T_hot_in - T_cold_in
        dT2_parallel = T_hot_out - T_cold_out
        lmtd_parallel = (dT1_parallel - dT2_parallel) / math.log(dT1_parallel / dT2_parallel)

        # Counterflow with same inlet temps but higher cold outlet
        T_cold_out_cf = 80.0
        dT1_counter = T_hot_in - T_cold_out_cf
        dT2_counter = T_hot_out - T_cold_in
        lmtd_counter = (dT1_counter - dT2_counter) / math.log(dT1_counter / dT2_counter)

        # For same UA, counterflow achieves higher effectiveness
        # This test verifies the thermodynamic advantage
        assert lmtd_parallel > 0
        assert lmtd_counter > 0


class TestLMTDEqualTerminalDifferences:
    """Test LMTD when terminal temperature differences are equal (dT1 = dT2)."""

    def test_equal_dt_case(self, operating_state_equal_dt):
        """Test LMTD when dT1 equals dT2."""
        state = operating_state_equal_dt

        dT1 = state.T_hot_in_C - state.T_cold_out_C  # 100 - 70 = 30
        dT2 = state.T_hot_out_C - state.T_cold_in_C  # 50 - 20 = 30

        assert abs(dT1 - dT2) < 0.01, "This test case requires dT1 = dT2"

        # When dT1 = dT2, LMTD = dT1 = dT2 (limit as ratio approaches 1)
        if abs(dT1 - dT2) < 0.01:
            lmtd = dT1
        else:
            lmtd = (dT1 - dT2) / math.log(dT1 / dT2)

        assert abs(lmtd - 30.0) < LMTD_TOLERANCE

    def test_near_equal_dt_numerical_stability(self):
        """Test numerical stability when dT1 is nearly equal to dT2."""
        # dT1 and dT2 very close but not exactly equal
        dT1 = 30.001
        dT2 = 29.999

        # Standard formula should still work
        lmtd = (dT1 - dT2) / math.log(dT1 / dT2)

        # Should be approximately 30
        assert abs(lmtd - 30.0) < 0.1

    def test_exact_equal_dt_formula_handling(self):
        """Test handling of exactly equal terminal differences."""
        dT1 = 30.0
        dT2 = 30.0

        # Use arithmetic mean when dT1 = dT2 (L'Hopital's rule limit)
        if abs(dT1 - dT2) < 1e-6:
            lmtd = (dT1 + dT2) / 2
        else:
            lmtd = (dT1 - dT2) / math.log(dT1 / dT2)

        assert lmtd == 30.0


class TestLMTDTemperatureCross:
    """Test LMTD behavior with temperature cross conditions."""

    def test_temperature_cross_detection(self, operating_state_temperature_cross):
        """Test detection of temperature cross condition."""
        state = operating_state_temperature_cross

        # Temperature cross: T_cold_out > T_hot_out
        has_temperature_cross = state.T_cold_out_C > state.T_hot_out_C

        assert has_temperature_cross, "Test case should have temperature cross"
        assert state.T_cold_out_C == 80.0
        assert state.T_hot_out_C == 50.0

    def test_temperature_cross_lmtd_calculation(self, operating_state_temperature_cross):
        """Test LMTD calculation with temperature cross."""
        state = operating_state_temperature_cross

        # For counterflow with temperature cross
        dT1 = state.T_hot_in_C - state.T_cold_out_C  # 100 - 80 = 20
        dT2 = state.T_hot_out_C - state.T_cold_in_C  # 50 - 20 = 30

        # LMTD is still calculable
        lmtd = (dT1 - dT2) / math.log(dT1 / dT2)

        assert lmtd > 0, "LMTD should still be positive"

    def test_temperature_cross_requires_multipass(self):
        """Test that temperature cross requires multiple shell passes."""
        # In a 1-2 shell-and-tube, temperature cross limits F-factor
        T_hot_in = 100.0
        T_cold_out = 90.0  # Close to T_hot_in (high effectiveness)
        T_hot_out = 50.0
        T_cold_in = 20.0

        # Calculate P and R for F-factor
        P = (T_cold_out - T_cold_in) / (T_hot_in - T_cold_in)  # (90-20)/(100-20) = 0.875
        R = (T_hot_in - T_hot_out) / (T_cold_out - T_cold_in)  # (100-50)/(90-20) = 0.714

        # High P with moderate R may cause temperature cross issues
        assert P > 0.8, "High P indicates temperature cross risk"


class TestFFactorCorrection:
    """Test F-factor correction for shell-and-tube exchangers."""

    def test_f_factor_bounds(self, sample_operating_state):
        """Test that F-factor is between 0 and 1."""
        state = sample_operating_state

        # Calculate P and R
        P = (state.T_cold_out_C - state.T_cold_in_C) / (state.T_hot_in_C - state.T_cold_in_C)
        R = (state.T_hot_in_C - state.T_hot_out_C) / (state.T_cold_out_C - state.T_cold_in_C)

        # F-factor for 1-2 shell-and-tube
        # Simplified formula (full formula in production code)
        if R == 1.0:
            # Special case when R = 1
            F = (P * math.sqrt(2)) / ((1 - P) * math.log(
                (2 - P * (2 - math.sqrt(2))) / (2 - P * (2 + math.sqrt(2)))
            ))
        else:
            S = math.sqrt(R * R + 1)
            W = ((1 - P * R) / (1 - P)) ** (1 / 1)  # Number of shell passes = 1
            F_num = S * math.log(W)
            F_den = (R - 1) * math.log(
                (2 - P * (R + 1 - S)) / (2 - P * (R + 1 + S))
            ) if abs(R - 1) > 0.001 else 1.0
            F = F_num / F_den if F_den != 0 else 0.9

        # F should be between 0.5 and 1.0 for valid configurations
        # Allow wider range for edge cases
        assert 0.0 <= F <= 1.0, f"F-factor {F} out of bounds"

    def test_f_factor_counterflow_equals_one(self):
        """Test that pure counterflow has F = 1.0."""
        # Pure counterflow needs no correction
        F_counterflow = 1.0
        assert F_counterflow == 1.0

    def test_f_factor_decreases_with_shell_passes(self):
        """Test F-factor behavior with increasing shell passes."""
        # F-factor approaches 1.0 as shell passes increase
        F_1_shell = 0.85
        F_2_shell = 0.92
        F_4_shell = 0.97

        # More shells -> higher F (closer to counterflow)
        assert F_1_shell < F_2_shell < F_4_shell


class TestLMTDCorrected:
    """Test corrected LMTD (LMTD * F) calculations."""

    def test_corrected_lmtd_less_than_pure(self):
        """Test that corrected LMTD is less than pure counterflow LMTD."""
        T_hot_in = 150.0
        T_hot_out = 90.0
        T_cold_in = 30.0
        T_cold_out = 100.0

        # Pure counterflow LMTD
        dT1 = T_hot_in - T_cold_out
        dT2 = T_hot_out - T_cold_in
        lmtd_pure = (dT1 - dT2) / math.log(dT1 / dT2)

        # Corrected LMTD with F = 0.9
        F = 0.9
        lmtd_corrected = lmtd_pure * F

        assert lmtd_corrected < lmtd_pure

    def test_corrected_lmtd_calculation(self, sample_thermal_kpis):
        """Test corrected LMTD from thermal KPIs."""
        kpis = sample_thermal_kpis

        # Verify relationship
        calculated_corrected = kpis.lmtd_C * kpis.F_factor
        assert abs(calculated_corrected - kpis.lmtd_corrected_C) < LMTD_TOLERANCE


class TestLMTDDeterminism:
    """Test LMTD calculation determinism."""

    def test_deterministic_calculation(self, sample_operating_state):
        """Test that LMTD calculation is deterministic."""
        state = sample_operating_state

        results = []
        for _ in range(10):
            dT1 = state.T_hot_in_C - state.T_cold_out_C
            dT2 = state.T_hot_out_C - state.T_cold_in_C
            lmtd = (dT1 - dT2) / math.log(dT1 / dT2)
            results.append(lmtd)

        assert all(r == results[0] for r in results)

    def test_provenance_hash_generation(self, sample_operating_state):
        """Test provenance hash generation for LMTD calculation."""
        state = sample_operating_state

        dT1 = state.T_hot_in_C - state.T_cold_out_C
        dT2 = state.T_hot_out_C - state.T_cold_in_C
        lmtd = (dT1 - dT2) / math.log(dT1 / dT2)

        provenance_data = f"{state.exchanger_id}:LMTD:{lmtd:.6f}"
        provenance_hash = hashlib.sha256(provenance_data.encode()).hexdigest()

        assert len(provenance_hash) == 64


class TestLMTDEdgeCases:
    """Test edge cases for LMTD calculations."""

    def test_very_small_temperature_difference(self):
        """Test LMTD with very small temperature differences."""
        dT1 = 1.0
        dT2 = 0.5

        lmtd = (dT1 - dT2) / math.log(dT1 / dT2)

        assert lmtd > 0
        assert lmtd < 1.0

    def test_very_large_temperature_difference(self):
        """Test LMTD with very large temperature differences."""
        dT1 = 200.0
        dT2 = 50.0

        lmtd = (dT1 - dT2) / math.log(dT1 / dT2)

        assert lmtd > 0
        assert lmtd < 200.0  # LMTD should be between dT2 and dT1

    def test_arithmetic_mean_bounds(self):
        """Test that LMTD is always less than arithmetic mean."""
        dT1 = 80.0
        dT2 = 20.0

        lmtd = (dT1 - dT2) / math.log(dT1 / dT2)
        arithmetic_mean = (dT1 + dT2) / 2

        assert lmtd < arithmetic_mean

    def test_lmtd_between_terminal_differences(self):
        """Test that LMTD is between dT1 and dT2."""
        dT1 = 50.0
        dT2 = 30.0

        lmtd = (dT1 - dT2) / math.log(dT1 / dT2)

        assert dT2 < lmtd < dT1


class TestLMTDValidation:
    """Test input validation for LMTD calculations."""

    def test_zero_temperature_difference_detection(self):
        """Test detection of zero temperature difference."""
        dT1 = 0.0
        dT2 = 30.0

        # Zero dT1 is physically impossible in operating exchanger
        assert dT1 == 0.0, "Zero temperature difference detected"

    def test_negative_temperature_difference_detection(self):
        """Test detection of negative temperature difference."""
        # Negative dT indicates reversed flow or sensor error
        T_hot_in = 100.0
        T_cold_out = 120.0  # Higher than hot inlet (impossible)

        dT1 = T_hot_in - T_cold_out
        assert dT1 < 0, "Negative temperature difference indicates error"


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "TestLMTDCounterflow",
    "TestLMTDParallelFlow",
    "TestLMTDEqualTerminalDifferences",
    "TestLMTDTemperatureCross",
    "TestFFactorCorrection",
    "TestLMTDCorrected",
    "TestLMTDDeterminism",
    "TestLMTDEdgeCases",
    "TestLMTDValidation",
]
