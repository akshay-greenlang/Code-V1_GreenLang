"""
GL-006 HEATRECLAIM - LMTD Calculator Unit Tests

Comprehensive unit tests for the Log Mean Temperature Difference (LMTD)
and NTU heat exchanger calculators with SHA-256 provenance tracking.

Reference:
    - Kays & London, "Compact Heat Exchangers", 1984
    - ASME PTC 4.4 for heat exchanger testing standards

Test Coverage:
    - LMTD calculation
    - F-correction factors
    - NTU-effectiveness relationships
    - Heat exchanger sizing
    - Flow arrangement handling
    - Edge cases and error handling
    - Provenance tracking
    - Golden test cases
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

from core.config import FlowArrangement, ExchangerType
from calculators.lmtd_calculator import (
    LMTDCalculator,
    NTUCalculator,
    LMTDResult,
    NTUResult,
    SizingResult,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def lmtd_calculator():
    """Create an LMTD calculator instance."""
    return LMTDCalculator(tolerance=0.1)


@pytest.fixture
def ntu_calculator():
    """Create an NTU calculator instance."""
    return NTUCalculator()


@pytest.fixture
def counter_current_case():
    """Standard counter-current heat exchanger test case."""
    return {
        "T_hot_in": 150.0,
        "T_hot_out": 90.0,
        "T_cold_in": 30.0,
        "T_cold_out": 80.0,
        "flow_arrangement": FlowArrangement.COUNTER_CURRENT,
    }


@pytest.fixture
def co_current_case():
    """Standard co-current (parallel) heat exchanger test case."""
    return {
        "T_hot_in": 150.0,
        "T_hot_out": 100.0,
        "T_cold_in": 30.0,
        "T_cold_out": 70.0,
        "flow_arrangement": FlowArrangement.CO_CURRENT,
    }


@pytest.fixture
def shell_tube_case():
    """Shell-and-tube 1-2 heat exchanger test case."""
    return {
        "T_hot_in": 150.0,
        "T_hot_out": 90.0,
        "T_cold_in": 30.0,
        "T_cold_out": 80.0,
        "flow_arrangement": FlowArrangement.SHELL_PASS_1_TUBE_2,
    }


@pytest.fixture
def sizing_parameters():
    """Parameters for heat exchanger sizing."""
    return {
        "duty_kW": 100.0,
        "T_hot_in": 150.0,
        "T_hot_out": 90.0,
        "T_cold_in": 30.0,
        "T_cold_out": 80.0,
        "m_dot_hot_kg_s": 1.0,
        "Cp_hot_kJ_kgK": 4.2,
        "m_dot_cold_kg_s": 1.2,
        "Cp_cold_kJ_kgK": 4.2,
        "U_assumed_W_m2K": 500.0,
    }


# =============================================================================
# LMTD CALCULATION TESTS
# =============================================================================

class TestLMTDCalculation:
    """Test suite for LMTD calculation."""

    @pytest.mark.unit
    @pytest.mark.lmtd
    def test_initialization(self, lmtd_calculator):
        """Test calculator initialization."""
        assert lmtd_calculator.tolerance == 0.1
        assert lmtd_calculator.VERSION == "1.0.0"

    @pytest.mark.unit
    @pytest.mark.lmtd
    def test_counter_current_lmtd(self, lmtd_calculator, counter_current_case):
        """Test LMTD calculation for counter-current flow."""
        result = lmtd_calculator.calculate(**counter_current_case)

        assert isinstance(result, LMTDResult)
        assert result.LMTD_C > 0
        assert result.is_valid is True
        assert result.F_correction == 1.0  # Counter-current has F=1

        # Verify LMTD calculation manually
        # Counter-current: dT1 = T_hot_in - T_cold_out, dT2 = T_hot_out - T_cold_in
        dT1 = 150.0 - 80.0  # = 70
        dT2 = 90.0 - 30.0   # = 60
        expected_lmtd = (dT1 - dT2) / math.log(dT1 / dT2)
        assert abs(result.LMTD_C - expected_lmtd) < 0.01

    @pytest.mark.unit
    @pytest.mark.lmtd
    def test_co_current_lmtd(self, lmtd_calculator, co_current_case):
        """Test LMTD calculation for co-current flow."""
        result = lmtd_calculator.calculate(**co_current_case)

        assert result.is_valid is True
        assert result.F_correction == 1.0  # Co-current has F=1

        # Verify LMTD calculation manually
        # Co-current: dT1 = T_hot_in - T_cold_in, dT2 = T_hot_out - T_cold_out
        dT1 = 150.0 - 30.0   # = 120
        dT2 = 100.0 - 70.0   # = 30
        expected_lmtd = (dT1 - dT2) / math.log(dT1 / dT2)
        assert abs(result.LMTD_C - expected_lmtd) < 0.01

    @pytest.mark.unit
    @pytest.mark.lmtd
    def test_equal_temperature_differences(self, lmtd_calculator):
        """Test LMTD when temperature differences are equal."""
        result = lmtd_calculator.calculate(
            T_hot_in=100.0,
            T_hot_out=60.0,
            T_cold_in=20.0,
            T_cold_out=60.0,
            flow_arrangement=FlowArrangement.COUNTER_CURRENT,
        )

        # When dT1 = dT2, LMTD = dT1 (or dT2)
        assert result.is_valid is True
        assert abs(result.LMTD_C - 40.0) < 0.1

    @pytest.mark.unit
    @pytest.mark.lmtd
    def test_temperature_cross_invalid(self, lmtd_calculator):
        """Test that temperature cross is detected as invalid."""
        result = lmtd_calculator.calculate(
            T_hot_in=80.0,   # Hot inlet
            T_hot_out=60.0,  # Hot outlet
            T_cold_in=50.0,  # Cold inlet
            T_cold_out=90.0, # Cold outlet > Hot inlet (cross!)
            flow_arrangement=FlowArrangement.COUNTER_CURRENT,
        )

        assert result.is_valid is False
        assert "Temperature cross detected" in result.warnings[0]

    @pytest.mark.unit
    @pytest.mark.lmtd
    def test_effective_lmtd(self, lmtd_calculator, shell_tube_case):
        """Test effective LMTD calculation with F-factor."""
        result = lmtd_calculator.calculate(**shell_tube_case)

        assert result.is_valid is True
        assert result.F_correction <= 1.0  # F <= 1 for shell-tube
        assert result.effective_LMTD_C == result.LMTD_C * result.F_correction


# =============================================================================
# F-CORRECTION FACTOR TESTS
# =============================================================================

class TestFCorrectionFactor:
    """Test suite for F-correction factor calculation."""

    @pytest.mark.unit
    @pytest.mark.lmtd
    def test_counter_current_f_equals_one(self, lmtd_calculator):
        """Test F=1 for counter-current flow."""
        result = lmtd_calculator.calculate(
            T_hot_in=150.0,
            T_hot_out=90.0,
            T_cold_in=30.0,
            T_cold_out=80.0,
            flow_arrangement=FlowArrangement.COUNTER_CURRENT,
        )
        assert result.F_correction == 1.0

    @pytest.mark.unit
    @pytest.mark.lmtd
    def test_co_current_f_equals_one(self, lmtd_calculator):
        """Test F=1 for co-current flow."""
        result = lmtd_calculator.calculate(
            T_hot_in=150.0,
            T_hot_out=100.0,
            T_cold_in=30.0,
            T_cold_out=70.0,
            flow_arrangement=FlowArrangement.CO_CURRENT,
        )
        assert result.F_correction == 1.0

    @pytest.mark.unit
    @pytest.mark.lmtd
    def test_shell_tube_1_2_f_less_than_one(self, lmtd_calculator):
        """Test F<1 for 1-2 shell-tube exchanger."""
        result = lmtd_calculator.calculate(
            T_hot_in=150.0,
            T_hot_out=90.0,
            T_cold_in=30.0,
            T_cold_out=80.0,
            flow_arrangement=FlowArrangement.SHELL_PASS_1_TUBE_2,
        )

        assert result.F_correction <= 1.0
        assert result.F_correction >= 0.5  # Should not be too low

    @pytest.mark.unit
    @pytest.mark.lmtd
    def test_low_f_factor_warning(self, lmtd_calculator):
        """Test warning when F-factor is low."""
        # Create a case that would give low F
        result = lmtd_calculator.calculate(
            T_hot_in=100.0,
            T_hot_out=50.0,
            T_cold_in=40.0,
            T_cold_out=90.0,
            flow_arrangement=FlowArrangement.SHELL_PASS_1_TUBE_2,
        )

        # Check if warning generated for low F
        if result.F_correction < 0.75:
            assert any("Low F-factor" in w for w in result.warnings)

    @pytest.mark.unit
    @pytest.mark.lmtd
    def test_crossflow_f_factor(self, lmtd_calculator):
        """Test F-factor for cross-flow arrangement."""
        result = lmtd_calculator.calculate(
            T_hot_in=150.0,
            T_hot_out=90.0,
            T_cold_in=30.0,
            T_cold_out=80.0,
            flow_arrangement=FlowArrangement.CROSS_FLOW,
        )

        assert result.is_valid is True
        assert 0.5 <= result.F_correction <= 1.0


# =============================================================================
# NTU-EFFECTIVENESS TESTS
# =============================================================================

class TestNTUEffectiveness:
    """Test suite for NTU-effectiveness calculations."""

    @pytest.mark.unit
    @pytest.mark.lmtd
    def test_effectiveness_counter_current(self, ntu_calculator):
        """Test effectiveness calculation for counter-current flow."""
        effectiveness = ntu_calculator.calculate_effectiveness(
            NTU=2.0,
            C_ratio=0.5,
            flow_arrangement=FlowArrangement.COUNTER_CURRENT,
        )

        assert 0.0 <= effectiveness <= 1.0
        # Higher NTU should give higher effectiveness
        eff_low = ntu_calculator.calculate_effectiveness(
            NTU=0.5, C_ratio=0.5,
            flow_arrangement=FlowArrangement.COUNTER_CURRENT,
        )
        assert effectiveness > eff_low

    @pytest.mark.unit
    @pytest.mark.lmtd
    def test_effectiveness_co_current(self, ntu_calculator):
        """Test effectiveness calculation for co-current flow."""
        effectiveness = ntu_calculator.calculate_effectiveness(
            NTU=2.0,
            C_ratio=0.5,
            flow_arrangement=FlowArrangement.CO_CURRENT,
        )

        assert 0.0 <= effectiveness <= 1.0
        # Co-current is limited to (1 + C_ratio)^-1 max
        max_eff = 1.0 / (1.0 + 0.5)
        assert effectiveness <= max_eff + 0.01

    @pytest.mark.unit
    @pytest.mark.lmtd
    def test_effectiveness_c_ratio_one(self, ntu_calculator):
        """Test effectiveness when C_ratio = 1."""
        effectiveness = ntu_calculator.calculate_effectiveness(
            NTU=2.0,
            C_ratio=1.0,
            flow_arrangement=FlowArrangement.COUNTER_CURRENT,
        )

        # Special case: epsilon = NTU / (1 + NTU)
        expected = 2.0 / (1.0 + 2.0)
        assert abs(effectiveness - expected) < 0.01

    @pytest.mark.unit
    @pytest.mark.lmtd
    def test_ntu_from_effectiveness(self, ntu_calculator):
        """Test NTU calculation from effectiveness."""
        # Calculate effectiveness first
        eff = ntu_calculator.calculate_effectiveness(
            NTU=2.0,
            C_ratio=0.5,
            flow_arrangement=FlowArrangement.COUNTER_CURRENT,
        )

        # Then calculate NTU back
        ntu = ntu_calculator.calculate_NTU(
            effectiveness=eff,
            C_ratio=0.5,
            flow_arrangement=FlowArrangement.COUNTER_CURRENT,
        )

        assert abs(ntu - 2.0) < 0.1  # Should recover original NTU

    @pytest.mark.unit
    @pytest.mark.lmtd
    def test_ntu_zero_gives_zero_effectiveness(self, ntu_calculator):
        """Test that NTU=0 gives effectiveness=0."""
        effectiveness = ntu_calculator.calculate_effectiveness(
            NTU=0.0,
            C_ratio=0.5,
            flow_arrangement=FlowArrangement.COUNTER_CURRENT,
        )
        assert effectiveness == 0.0

    @pytest.mark.unit
    @pytest.mark.lmtd
    def test_high_ntu_approaches_limit(self, ntu_calculator):
        """Test that high NTU approaches effectiveness limit."""
        effectiveness = ntu_calculator.calculate_effectiveness(
            NTU=10.0,
            C_ratio=0.5,
            flow_arrangement=FlowArrangement.COUNTER_CURRENT,
        )

        # For counter-current, max = 1 when C_ratio < 1
        assert effectiveness > 0.95


# =============================================================================
# HEAT EXCHANGER SIZING TESTS
# =============================================================================

class TestSizing:
    """Test suite for heat exchanger sizing."""

    @pytest.mark.unit
    @pytest.mark.lmtd
    def test_basic_sizing(self, ntu_calculator, sizing_parameters):
        """Test basic heat exchanger sizing."""
        result = ntu_calculator.size_exchanger(**sizing_parameters)

        assert isinstance(result, SizingResult)
        assert result.is_feasible is True
        assert result.duty_kW == 100.0
        assert result.UA_kW_K > 0
        assert result.area_m2 > 0
        assert result.LMTD_C > 0

    @pytest.mark.unit
    @pytest.mark.lmtd
    def test_sizing_ua_calculation(self, ntu_calculator, sizing_parameters):
        """Test UA calculation in sizing."""
        result = ntu_calculator.size_exchanger(**sizing_parameters)

        # UA = Q / LMTD
        expected_ua = sizing_parameters["duty_kW"] / result.LMTD_C
        assert abs(result.UA_kW_K - expected_ua) < 0.01

    @pytest.mark.unit
    @pytest.mark.lmtd
    def test_sizing_area_calculation(self, ntu_calculator, sizing_parameters):
        """Test area calculation in sizing."""
        result = ntu_calculator.size_exchanger(**sizing_parameters)

        # A = UA / U
        U_kW = sizing_parameters["U_assumed_W_m2K"] / 1000.0
        expected_area = result.UA_kW_K / U_kW
        assert abs(result.area_m2 - expected_area) < 0.01

    @pytest.mark.unit
    @pytest.mark.lmtd
    def test_sizing_provenance_hash(self, ntu_calculator, sizing_parameters):
        """Test provenance hash in sizing result."""
        result = ntu_calculator.size_exchanger(**sizing_parameters)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 16

    @pytest.mark.unit
    @pytest.mark.lmtd
    def test_sizing_with_temperature_cross(self, ntu_calculator):
        """Test sizing with temperature cross (infeasible)."""
        result = ntu_calculator.size_exchanger(
            duty_kW=100.0,
            T_hot_in=80.0,
            T_hot_out=60.0,
            T_cold_in=50.0,
            T_cold_out=90.0,  # Temperature cross
            m_dot_hot_kg_s=1.0,
            Cp_hot_kJ_kgK=4.2,
            m_dot_cold_kg_s=1.0,
            Cp_cold_kJ_kgK=4.2,
            U_assumed_W_m2K=500.0,
        )

        assert result.is_feasible is False


# =============================================================================
# FLOW ARRANGEMENT TESTS
# =============================================================================

class TestFlowArrangements:
    """Test suite for different flow arrangements."""

    @pytest.mark.unit
    @pytest.mark.lmtd
    def test_all_flow_arrangements_valid(self, lmtd_calculator):
        """Test that all flow arrangements give valid results."""
        for arrangement in FlowArrangement:
            result = lmtd_calculator.calculate(
                T_hot_in=150.0,
                T_hot_out=90.0,
                T_cold_in=30.0,
                T_cold_out=80.0,
                flow_arrangement=arrangement,
            )
            assert result.is_valid is True

    @pytest.mark.unit
    @pytest.mark.lmtd
    def test_counter_current_best_for_same_temps(self, lmtd_calculator):
        """Test counter-current gives best effective LMTD."""
        temps = {
            "T_hot_in": 150.0,
            "T_hot_out": 90.0,
            "T_cold_in": 30.0,
            "T_cold_out": 80.0,
        }

        counter = lmtd_calculator.calculate(
            **temps, flow_arrangement=FlowArrangement.COUNTER_CURRENT
        )
        shell_tube = lmtd_calculator.calculate(
            **temps, flow_arrangement=FlowArrangement.SHELL_PASS_1_TUBE_2
        )

        # Counter-current should have F=1, so better effective LMTD
        assert counter.effective_LMTD_C >= shell_tube.effective_LMTD_C


# =============================================================================
# EDGE CASES TESTS
# =============================================================================

class TestEdgeCases:
    """Test suite for edge cases."""

    @pytest.mark.unit
    @pytest.mark.lmtd
    def test_very_small_temperature_difference(self, lmtd_calculator):
        """Test with very small temperature differences."""
        result = lmtd_calculator.calculate(
            T_hot_in=100.0,
            T_hot_out=99.5,
            T_cold_in=30.0,
            T_cold_out=30.3,
            flow_arrangement=FlowArrangement.COUNTER_CURRENT,
        )

        assert result.is_valid is True
        # LMTD should be small but positive
        assert 0 < result.LMTD_C < 100

    @pytest.mark.unit
    @pytest.mark.lmtd
    def test_large_temperature_range(self, lmtd_calculator):
        """Test with large temperature range."""
        result = lmtd_calculator.calculate(
            T_hot_in=500.0,
            T_hot_out=100.0,
            T_cold_in=20.0,
            T_cold_out=300.0,
            flow_arrangement=FlowArrangement.COUNTER_CURRENT,
        )

        assert result.is_valid is True
        assert result.LMTD_C > 0

    @pytest.mark.unit
    @pytest.mark.lmtd
    def test_ntu_high_c_ratio(self, ntu_calculator):
        """Test NTU calculations with C_ratio approaching 1."""
        effectiveness = ntu_calculator.calculate_effectiveness(
            NTU=2.0,
            C_ratio=0.99,
            flow_arrangement=FlowArrangement.COUNTER_CURRENT,
        )

        assert 0.0 <= effectiveness <= 1.0

    @pytest.mark.unit
    @pytest.mark.lmtd
    def test_ntu_zero_c_ratio(self, ntu_calculator):
        """Test NTU calculations with C_ratio = 0."""
        effectiveness = ntu_calculator.calculate_effectiveness(
            NTU=2.0,
            C_ratio=0.0,
            flow_arrangement=FlowArrangement.COUNTER_CURRENT,
        )

        assert 0.0 <= effectiveness <= 1.0


# =============================================================================
# GOLDEN TEST CASES
# =============================================================================

class TestGoldenCases:
    """Test suite with known expected results."""

    @pytest.mark.unit
    @pytest.mark.lmtd
    @pytest.mark.golden
    def test_textbook_counter_current_example(self, lmtd_calculator):
        """
        Test against textbook counter-current example.

        Hot: 150 -> 90 C
        Cold: 30 -> 80 C

        dT1 = 150 - 80 = 70 C
        dT2 = 90 - 30 = 60 C
        LMTD = (70 - 60) / ln(70/60) = 64.87 C
        """
        result = lmtd_calculator.calculate(
            T_hot_in=150.0,
            T_hot_out=90.0,
            T_cold_in=30.0,
            T_cold_out=80.0,
            flow_arrangement=FlowArrangement.COUNTER_CURRENT,
        )

        expected_lmtd = (70 - 60) / math.log(70 / 60)
        assert abs(result.LMTD_C - expected_lmtd) < 0.1
        assert abs(result.LMTD_C - 64.87) < 0.1

    @pytest.mark.unit
    @pytest.mark.lmtd
    @pytest.mark.golden
    def test_textbook_co_current_example(self, lmtd_calculator):
        """
        Test against textbook co-current example.

        Hot: 150 -> 100 C
        Cold: 30 -> 70 C

        dT1 = 150 - 30 = 120 C
        dT2 = 100 - 70 = 30 C
        LMTD = (120 - 30) / ln(120/30) = 64.94 C
        """
        result = lmtd_calculator.calculate(
            T_hot_in=150.0,
            T_hot_out=100.0,
            T_cold_in=30.0,
            T_cold_out=70.0,
            flow_arrangement=FlowArrangement.CO_CURRENT,
        )

        expected_lmtd = (120 - 30) / math.log(120 / 30)
        assert abs(result.LMTD_C - expected_lmtd) < 0.1

    @pytest.mark.unit
    @pytest.mark.lmtd
    @pytest.mark.golden
    def test_ntu_effectiveness_verification(self, ntu_calculator):
        """
        Verify NTU-effectiveness relationship.

        For counter-current with C_ratio = 0.5 and NTU = 2:
        epsilon = (1 - exp(-NTU*(1-Cr))) / (1 - Cr*exp(-NTU*(1-Cr)))
        """
        NTU = 2.0
        C_r = 0.5

        effectiveness = ntu_calculator.calculate_effectiveness(
            NTU=NTU,
            C_ratio=C_r,
            flow_arrangement=FlowArrangement.COUNTER_CURRENT,
        )

        # Manual calculation
        exp_term = math.exp(-NTU * (1 - C_r))
        expected = (1 - exp_term) / (1 - C_r * exp_term)

        assert abs(effectiveness - expected) < 0.001


# =============================================================================
# DETERMINISM TESTS
# =============================================================================

class TestDeterminism:
    """Test suite for deterministic behavior."""

    @pytest.mark.unit
    @pytest.mark.lmtd
    def test_lmtd_deterministic(self, lmtd_calculator):
        """Test that LMTD calculation is deterministic."""
        params = {
            "T_hot_in": 150.0,
            "T_hot_out": 90.0,
            "T_cold_in": 30.0,
            "T_cold_out": 80.0,
            "flow_arrangement": FlowArrangement.COUNTER_CURRENT,
        }

        results = [lmtd_calculator.calculate(**params) for _ in range(10)]

        # All results should be identical
        for r in results[1:]:
            assert r.LMTD_C == results[0].LMTD_C
            assert r.F_correction == results[0].F_correction
            assert r.effective_LMTD_C == results[0].effective_LMTD_C

    @pytest.mark.unit
    @pytest.mark.lmtd
    def test_ntu_deterministic(self, ntu_calculator):
        """Test that NTU calculation is deterministic."""
        params = {"NTU": 2.0, "C_ratio": 0.5}

        results = [
            ntu_calculator.calculate_effectiveness(
                **params, flow_arrangement=FlowArrangement.COUNTER_CURRENT
            )
            for _ in range(10)
        ]

        # All results should be identical
        for r in results[1:]:
            assert r == results[0]


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Test suite for performance benchmarks."""

    @pytest.mark.unit
    @pytest.mark.lmtd
    @pytest.mark.benchmark
    def test_lmtd_calculation_time(self, lmtd_calculator):
        """Test LMTD calculation performance."""
        import time

        params = {
            "T_hot_in": 150.0,
            "T_hot_out": 90.0,
            "T_cold_in": 30.0,
            "T_cold_out": 80.0,
            "flow_arrangement": FlowArrangement.COUNTER_CURRENT,
        }

        start = time.time()
        for _ in range(1000):
            lmtd_calculator.calculate(**params)
        elapsed = time.time() - start

        # Should complete 1000 calculations in under 1 second
        assert elapsed < 1.0, f"Too slow: {elapsed:.3f}s for 1000 calculations"

    @pytest.mark.unit
    @pytest.mark.lmtd
    @pytest.mark.benchmark
    def test_sizing_calculation_time(self, ntu_calculator, sizing_parameters):
        """Test sizing calculation performance."""
        import time

        start = time.time()
        for _ in range(1000):
            ntu_calculator.size_exchanger(**sizing_parameters)
        elapsed = time.time() - start

        # Should complete 1000 calculations in under 2 seconds
        assert elapsed < 2.0, f"Too slow: {elapsed:.3f}s for 1000 calculations"
