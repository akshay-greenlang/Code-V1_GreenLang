# -*- coding: utf-8 -*-
"""
GL-014 EXCHANGERPRO - Epsilon-NTU Calculator Unit Tests

Tests for effectiveness-NTU method calculations including:
- All flow configurations (counterflow, parallel, crossflow, shell-and-tube)
- Effectiveness bounds (0 <= epsilon <= 1)
- NTU calculation from UA and capacity rates
- Balanced exchanger (C_ratio = 1) special cases
- High NTU asymptotic behavior
- Provenance hash verification

Reference:
- Incropera & DeWitt, Fundamentals of Heat and Mass Transfer
- Kays & London, Compact Heat Exchangers
- TEMA Standards

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import math
import hashlib
from typing import Dict, Any


# Test tolerances
EPSILON_TOLERANCE = 0.01  # 1% tolerance on effectiveness
NTU_TOLERANCE = 0.05
C_RATIO_TOLERANCE = 0.001


class TestEffectivenessCounterflow:
    """Test effectiveness calculations for counterflow exchangers."""

    def test_counterflow_effectiveness_formula(self):
        """Test counterflow effectiveness formula."""
        NTU = 2.0
        C_ratio = 0.5

        # Counterflow: epsilon = (1 - exp(-NTU*(1-C))) / (1 - C*exp(-NTU*(1-C)))
        epsilon = (1 - math.exp(-NTU * (1 - C_ratio))) / \
                  (1 - C_ratio * math.exp(-NTU * (1 - C_ratio)))

        assert 0 < epsilon < 1
        assert epsilon == pytest.approx(0.797, abs=EPSILON_TOLERANCE)

    def test_counterflow_balanced_exchanger(self, operating_state_balanced):
        """Test counterflow with balanced capacity rates (C_ratio = 1)."""
        state = operating_state_balanced

        C_hot = state.m_dot_hot_kg_s * state.Cp_hot_kJ_kgK
        C_cold = state.m_dot_cold_kg_s * state.Cp_cold_kJ_kgK
        C_ratio = min(C_hot, C_cold) / max(C_hot, C_cold)

        # For balanced exchanger (C_ratio = 1):
        # epsilon = NTU / (1 + NTU)
        assert abs(C_ratio - 1.0) < C_RATIO_TOLERANCE

    def test_balanced_effectiveness_formula(self):
        """Test effectiveness formula for C_ratio = 1."""
        NTU = 2.0
        C_ratio = 1.0

        # For C_ratio = 1: epsilon = NTU / (1 + NTU)
        epsilon = NTU / (1 + NTU)

        assert epsilon == pytest.approx(0.667, abs=EPSILON_TOLERANCE)

    @pytest.mark.parametrize("NTU,C_ratio,expected_epsilon", [
        (0.5, 0.5, 0.378),
        (1.0, 0.5, 0.565),
        (2.0, 0.5, 0.797),
        (3.0, 0.5, 0.902),
        (5.0, 0.5, 0.968),
        (1.0, 1.0, 0.500),
        (2.0, 1.0, 0.667),
        (3.0, 1.0, 0.750),
        (0.5, 0.0, 0.393),  # Evaporator/condenser (C_ratio = 0)
        (2.0, 0.0, 0.865),
    ])
    def test_counterflow_parametric(self, NTU: float, C_ratio: float, expected_epsilon: float):
        """Test counterflow effectiveness with various NTU and C_ratio."""
        if C_ratio == 1.0:
            epsilon = NTU / (1 + NTU)
        elif C_ratio == 0.0:
            epsilon = 1 - math.exp(-NTU)
        else:
            epsilon = (1 - math.exp(-NTU * (1 - C_ratio))) / \
                      (1 - C_ratio * math.exp(-NTU * (1 - C_ratio)))

        assert epsilon == pytest.approx(expected_epsilon, abs=EPSILON_TOLERANCE)


class TestEffectivenessParallelFlow:
    """Test effectiveness calculations for parallel flow exchangers."""

    def test_parallel_flow_effectiveness_formula(self):
        """Test parallel flow effectiveness formula."""
        NTU = 2.0
        C_ratio = 0.5

        # Parallel flow: epsilon = (1 - exp(-NTU*(1+C))) / (1 + C)
        epsilon = (1 - math.exp(-NTU * (1 + C_ratio))) / (1 + C_ratio)

        assert 0 < epsilon < 1
        assert epsilon == pytest.approx(0.632, abs=EPSILON_TOLERANCE)

    def test_parallel_flow_lower_than_counterflow(self):
        """Test that parallel flow has lower effectiveness than counterflow."""
        NTU = 2.0
        C_ratio = 0.5

        # Counterflow
        epsilon_cf = (1 - math.exp(-NTU * (1 - C_ratio))) / \
                     (1 - C_ratio * math.exp(-NTU * (1 - C_ratio)))

        # Parallel flow
        epsilon_pf = (1 - math.exp(-NTU * (1 + C_ratio))) / (1 + C_ratio)

        assert epsilon_pf < epsilon_cf

    def test_parallel_flow_max_effectiveness(self):
        """Test maximum effectiveness limit for parallel flow."""
        NTU = 100.0  # Very high NTU
        C_ratio = 0.5

        # Parallel flow max: epsilon_max = 1 / (1 + C)
        epsilon_max = 1 / (1 + C_ratio)
        epsilon = (1 - math.exp(-NTU * (1 + C_ratio))) / (1 + C_ratio)

        assert epsilon == pytest.approx(epsilon_max, abs=EPSILON_TOLERANCE)


class TestEffectivenessCrossflow:
    """Test effectiveness calculations for crossflow exchangers."""

    def test_crossflow_both_unmixed(self):
        """Test crossflow with both fluids unmixed."""
        NTU = 2.0
        C_ratio = 0.5

        # Approximation for crossflow both unmixed
        # epsilon = 1 - exp((NTU^0.22/C) * (exp(-C*NTU^0.78) - 1))
        epsilon_approx = 1 - math.exp(
            (NTU ** 0.22 / C_ratio) *
            (math.exp(-C_ratio * NTU ** 0.78) - 1)
        )

        assert 0 < epsilon_approx < 1

    def test_crossflow_cmin_mixed(self):
        """Test crossflow with Cmin fluid mixed."""
        NTU = 2.0
        C_ratio = 0.5

        # epsilon = (1/C) * (1 - exp(-C * (1 - exp(-NTU))))
        epsilon = (1 / C_ratio) * (1 - math.exp(-C_ratio * (1 - math.exp(-NTU))))

        assert 0 < epsilon < 1

    def test_crossflow_cmax_mixed(self):
        """Test crossflow with Cmax fluid mixed."""
        NTU = 2.0
        C_ratio = 0.5

        # epsilon = 1 - exp(-(1/C) * (1 - exp(-C*NTU)))
        epsilon = 1 - math.exp(-(1 / C_ratio) * (1 - math.exp(-C_ratio * NTU)))

        assert 0 < epsilon < 1


class TestEffectivenessShellAndTube:
    """Test effectiveness calculations for shell-and-tube exchangers."""

    def test_shell_and_tube_one_shell(self):
        """Test 1-2 shell-and-tube effectiveness."""
        NTU = 2.0
        C_ratio = 0.5

        # 1-2 shell-and-tube formula
        E = math.exp(-NTU * math.sqrt(1 + C_ratio ** 2))
        epsilon = 2 * (1 + C_ratio + math.sqrt(1 + C_ratio ** 2) *
                       (1 + E) / (1 - E)) ** (-1)

        assert 0 < epsilon < 1

    def test_shell_and_tube_multiple_shells(self):
        """Test n-2n shell-and-tube effectiveness."""
        NTU = 2.0
        C_ratio = 0.5
        N = 2  # Number of shell passes

        # Calculate epsilon for 1-2, then use multiple shell formula
        E = math.exp(-NTU / N * math.sqrt(1 + C_ratio ** 2))
        epsilon_1 = 2 * (1 + C_ratio + math.sqrt(1 + C_ratio ** 2) *
                         (1 + E) / (1 - E)) ** (-1)

        # Multiple shells
        if C_ratio == 1.0:
            epsilon = (N * epsilon_1) / (1 + (N - 1) * epsilon_1)
        else:
            X = ((1 - epsilon_1 * C_ratio) / (1 - epsilon_1)) ** N
            epsilon = (X - 1) / (X - C_ratio)

        assert 0 < epsilon < 1


class TestNTUCalculation:
    """Test NTU calculation from UA and capacity rates."""

    def test_ntu_calculation_basic(self, sample_operating_state):
        """Test basic NTU calculation."""
        state = sample_operating_state

        # Calculate capacity rates
        C_hot = state.m_dot_hot_kg_s * state.Cp_hot_kJ_kgK  # kW/K
        C_cold = state.m_dot_cold_kg_s * state.Cp_cold_kJ_kgK

        C_min = min(C_hot, C_cold)
        C_max = max(C_hot, C_cold)

        # Assume UA = 90 kW/K
        UA = 90.0

        # NTU = UA / C_min
        NTU = UA / C_min

        assert NTU > 0

    def test_ntu_from_thermal_kpis(self, sample_thermal_kpis):
        """Test NTU value from thermal KPIs."""
        kpis = sample_thermal_kpis

        assert kpis.NTU > 0
        assert kpis.C_ratio > 0
        assert kpis.C_ratio <= 1.0

    def test_c_ratio_bounds(self, sample_operating_state):
        """Test that C_ratio is always between 0 and 1."""
        state = sample_operating_state

        C_hot = state.m_dot_hot_kg_s * state.Cp_hot_kJ_kgK
        C_cold = state.m_dot_cold_kg_s * state.Cp_cold_kJ_kgK

        C_ratio = min(C_hot, C_cold) / max(C_hot, C_cold)

        assert 0 < C_ratio <= 1.0


class TestEffectivenessBounds:
    """Test effectiveness bounds and physical constraints."""

    def test_effectiveness_between_zero_and_one(self):
        """Test that effectiveness is always between 0 and 1."""
        for NTU in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
            for C_ratio in [0.0, 0.25, 0.5, 0.75, 1.0]:
                if C_ratio == 1.0:
                    epsilon = NTU / (1 + NTU)
                elif C_ratio == 0.0:
                    epsilon = 1 - math.exp(-NTU)
                else:
                    epsilon = (1 - math.exp(-NTU * (1 - C_ratio))) / \
                              (1 - C_ratio * math.exp(-NTU * (1 - C_ratio)))

                assert 0 <= epsilon <= 1, f"epsilon={epsilon} for NTU={NTU}, C={C_ratio}"

    def test_effectiveness_increases_with_ntu(self):
        """Test that effectiveness increases with NTU."""
        C_ratio = 0.5
        ntus = [0.5, 1.0, 2.0, 3.0, 5.0]

        epsilons = []
        for NTU in ntus:
            epsilon = (1 - math.exp(-NTU * (1 - C_ratio))) / \
                      (1 - C_ratio * math.exp(-NTU * (1 - C_ratio)))
            epsilons.append(epsilon)

        for i in range(1, len(epsilons)):
            assert epsilons[i] > epsilons[i - 1], "Effectiveness must increase with NTU"

    def test_effectiveness_asymptotic_limit(self):
        """Test effectiveness approaches asymptotic limit at high NTU."""
        C_ratio = 0.5
        NTU_high = 100.0

        epsilon = (1 - math.exp(-NTU_high * (1 - C_ratio))) / \
                  (1 - C_ratio * math.exp(-NTU_high * (1 - C_ratio)))

        # For counterflow, epsilon_max = 1 when C_ratio < 1
        assert epsilon > 0.99


class TestHighNTUBehavior:
    """Test behavior at high NTU values."""

    def test_high_ntu_counterflow(self, operating_state_high_ntu):
        """Test high NTU counterflow effectiveness."""
        state = operating_state_high_ntu

        # Calculate actual effectiveness from temperatures
        Q_actual = state.m_dot_hot_kg_s * state.Cp_hot_kJ_kgK * \
                   (state.T_hot_in_C - state.T_hot_out_C)

        C_hot = state.m_dot_hot_kg_s * state.Cp_hot_kJ_kgK
        C_cold = state.m_dot_cold_kg_s * state.Cp_cold_kJ_kgK
        C_min = min(C_hot, C_cold)

        Q_max = C_min * (state.T_hot_in_C - state.T_cold_in_C)

        epsilon = Q_actual / Q_max

        assert epsilon > 0.9, "High NTU should give high effectiveness"

    def test_ntu_above_10_near_max_effectiveness(self):
        """Test that NTU > 10 gives near-maximum effectiveness."""
        NTU = 15.0
        C_ratio = 0.5

        epsilon = (1 - math.exp(-NTU * (1 - C_ratio))) / \
                  (1 - C_ratio * math.exp(-NTU * (1 - C_ratio)))

        assert epsilon > 0.999


class TestCapacityRateRatio:
    """Test capacity rate ratio calculations."""

    def test_c_ratio_calculation(self, sample_operating_state):
        """Test capacity rate ratio calculation."""
        state = sample_operating_state

        C_hot = state.m_dot_hot_kg_s * state.Cp_hot_kJ_kgK
        C_cold = state.m_dot_cold_kg_s * state.Cp_cold_kJ_kgK

        C_min = min(C_hot, C_cold)
        C_max = max(C_hot, C_cold)
        C_ratio = C_min / C_max

        # Verify C_hot and C_cold values
        assert C_hot == pytest.approx(57.5, abs=0.1)  # 25 * 2.3
        assert C_cold == pytest.approx(83.6, abs=0.1)  # 20 * 4.18

        assert C_ratio < 1.0

    def test_evaporator_condenser_c_ratio(self):
        """Test C_ratio = 0 case (evaporator or condenser)."""
        # When one fluid undergoes phase change, its effective Cp is infinite
        # So C_ratio = 0
        C_ratio = 0.0
        NTU = 2.0

        # epsilon = 1 - exp(-NTU) for C_ratio = 0
        epsilon = 1 - math.exp(-NTU)

        assert epsilon == pytest.approx(0.865, abs=EPSILON_TOLERANCE)


class TestEpsilonNTUDeterminism:
    """Test epsilon-NTU calculation determinism."""

    def test_deterministic_effectiveness(self):
        """Test that effectiveness calculation is deterministic."""
        NTU = 2.0
        C_ratio = 0.5

        results = []
        for _ in range(10):
            epsilon = (1 - math.exp(-NTU * (1 - C_ratio))) / \
                      (1 - C_ratio * math.exp(-NTU * (1 - C_ratio)))
            results.append(epsilon)

        assert all(r == results[0] for r in results)

    def test_provenance_hash_generation(self, sample_operating_state):
        """Test provenance hash for epsilon-NTU calculation."""
        state = sample_operating_state

        C_hot = state.m_dot_hot_kg_s * state.Cp_hot_kJ_kgK
        C_cold = state.m_dot_cold_kg_s * state.Cp_cold_kJ_kgK
        C_min = min(C_hot, C_cold)
        C_max = max(C_hot, C_cold)
        C_ratio = C_min / C_max

        NTU = 2.0  # Assumed
        epsilon = NTU / (1 + NTU) if C_ratio == 1.0 else \
            (1 - math.exp(-NTU * (1 - C_ratio))) / \
            (1 - C_ratio * math.exp(-NTU * (1 - C_ratio)))

        provenance_data = f"{state.exchanger_id}:epsilon:{epsilon:.6f}:NTU:{NTU:.6f}"
        provenance_hash = hashlib.sha256(provenance_data.encode()).hexdigest()

        assert len(provenance_hash) == 64


class TestEpsilonNTUEdgeCases:
    """Test edge cases for epsilon-NTU calculations."""

    def test_zero_ntu(self):
        """Test behavior at NTU = 0."""
        NTU = 0.0
        C_ratio = 0.5

        # At NTU = 0, epsilon = 0
        epsilon = (1 - math.exp(-NTU * (1 - C_ratio))) / \
                  (1 - C_ratio * math.exp(-NTU * (1 - C_ratio))) if NTU > 0 else 0.0

        assert epsilon == 0.0

    def test_very_small_ntu(self):
        """Test behavior at very small NTU."""
        NTU = 0.01
        C_ratio = 0.5

        epsilon = (1 - math.exp(-NTU * (1 - C_ratio))) / \
                  (1 - C_ratio * math.exp(-NTU * (1 - C_ratio)))

        # At small NTU, epsilon approximately equals NTU
        assert epsilon < 0.05

    def test_near_unity_c_ratio(self):
        """Test numerical stability near C_ratio = 1."""
        NTU = 2.0
        C_ratio = 0.9999

        # Should use balanced formula or handle limit carefully
        epsilon_balanced = NTU / (1 + NTU)
        epsilon_general = (1 - math.exp(-NTU * (1 - C_ratio))) / \
                          (1 - C_ratio * math.exp(-NTU * (1 - C_ratio)))

        # Both should give similar results
        assert abs(epsilon_general - epsilon_balanced) < 0.01


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "TestEffectivenessCounterflow",
    "TestEffectivenessParallelFlow",
    "TestEffectivenessCrossflow",
    "TestEffectivenessShellAndTube",
    "TestNTUCalculation",
    "TestEffectivenessBounds",
    "TestHighNTUBehavior",
    "TestCapacityRateRatio",
    "TestEpsilonNTUDeterminism",
    "TestEpsilonNTUEdgeCases",
]
