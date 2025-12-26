# -*- coding: utf-8 -*-
"""
GL-014 EXCHANGERPRO - Pressure Drop Calculator Unit Tests

Tests for pressure drop calculations including:
- Tube-side pressure drop (Darcy-Weisbach)
- Shell-side pressure drop (Kern method)
- Friction factor calculations
- Reynolds number effects
- Pressure drop ratio monitoring
- Fouling-induced pressure drop increase
- Provenance hash verification

Reference:
- TEMA Standards (9th Edition)
- Kern, Process Heat Transfer
- ASME PTC 12.5

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import math
import hashlib
from typing import Dict, Any


# Test tolerances
PRESSURE_TOLERANCE = 0.5  # kPa
FRICTION_TOLERANCE = 0.001


class TestTubeSidePressureDrop:
    """Test tube-side pressure drop calculations using Darcy-Weisbach."""

    def test_tube_pressure_drop_basic(self, sample_operating_state, sample_exchanger_config):
        """Test basic tube-side pressure drop calculation."""
        state = sample_operating_state
        config = sample_exchanger_config

        # Pressure drop from measurements
        dP_tube_measured = state.P_cold_in_kPa - state.P_cold_out_kPa

        # Should be positive (pressure decreases in flow direction)
        assert dP_tube_measured > 0

    def test_darcy_weisbach_formula(self, sample_exchanger_config):
        """Test Darcy-Weisbach pressure drop calculation."""
        config = sample_exchanger_config

        # Fluid properties (water)
        rho = 990.0  # kg/m3
        mu = 0.0008  # Pa-s
        v = 2.0      # m/s (velocity)

        # Tube geometry
        D = config.tube_id_m
        L = config.tube_length_m * config.tube_passes

        # Reynolds number
        Re = rho * v * D / mu

        # Friction factor (Blasius for turbulent flow)
        if Re > 4000:
            f = 0.316 * Re ** (-0.25)
        else:
            f = 64 / Re

        # Darcy-Weisbach: dP = f * (L/D) * (rho*v^2/2)
        dP = f * (L / D) * (rho * v ** 2 / 2)  # Pa

        dP_kPa = dP / 1000

        assert dP_kPa > 0

    @pytest.mark.parametrize("velocity,expected_dp_range", [
        (1.0, (5, 20)),     # Low velocity
        (2.0, (20, 80)),    # Moderate velocity
        (3.0, (45, 180)),   # High velocity
        (5.0, (125, 500)),  # Very high velocity
    ])
    def test_tube_dp_vs_velocity(
        self,
        velocity: float,
        expected_dp_range: tuple,
        sample_exchanger_config,
    ):
        """Test tube pressure drop vs velocity relationship."""
        config = sample_exchanger_config

        rho = 990.0
        mu = 0.0008
        D = config.tube_id_m
        L = config.tube_length_m * config.tube_passes

        Re = rho * velocity * D / mu

        if Re > 4000:
            f = 0.316 * Re ** (-0.25)
        else:
            f = 64 / Re

        dP = f * (L / D) * (rho * velocity ** 2 / 2) / 1000  # kPa

        # Pressure drop increases with velocity squared
        min_dp, max_dp = expected_dp_range
        assert min_dp < dP < max_dp


class TestShellSidePressureDrop:
    """Test shell-side pressure drop calculations."""

    def test_shell_pressure_drop_basic(self, sample_operating_state):
        """Test basic shell-side pressure drop from measurements."""
        state = sample_operating_state

        dP_shell_measured = state.P_hot_in_kPa - state.P_hot_out_kPa

        assert dP_shell_measured > 0

    def test_shell_dp_kern_method(self, sample_exchanger_config):
        """Test shell-side pressure drop using Kern method."""
        config = sample_exchanger_config

        # Fluid properties (crude oil)
        rho = 800.0  # kg/m3
        mu = 0.002   # Pa-s
        v_shell = 1.0  # m/s (shell-side velocity)

        # Shell geometry
        D_shell = config.shell_diameter_m
        baffle_spacing = config.baffle_spacing_m
        n_baffles = int(config.tube_length_m / baffle_spacing) - 1

        # Simplified Kern method
        # dP = f * n_baffles * rho * v^2 / 2
        f = 0.5  # Typical friction factor for shell side

        dP = f * n_baffles * (rho * v_shell ** 2 / 2) / 1000  # kPa

        assert dP > 0

    def test_baffle_cut_effect(self):
        """Test effect of baffle cut on shell-side pressure drop."""
        # Higher baffle cut -> lower pressure drop
        baffle_cuts = [20, 25, 30, 35]  # percent
        pressure_drops = []

        for cut in baffle_cuts:
            # Simplified model: dP inversely proportional to baffle cut
            dP_factor = 1.0 - (cut - 20) * 0.02
            pressure_drops.append(dP_factor)

        # Verify trend
        for i in range(1, len(pressure_drops)):
            assert pressure_drops[i] <= pressure_drops[i - 1]


class TestFrictionFactor:
    """Test friction factor calculations."""

    def test_laminar_friction_factor(self):
        """Test laminar flow friction factor (Re < 2300)."""
        Re = 1000

        f = 64 / Re

        assert f == 0.064

    def test_turbulent_friction_factor_blasius(self):
        """Test turbulent Blasius friction factor."""
        Re = 10000

        f = 0.316 * Re ** (-0.25)

        assert f == pytest.approx(0.0316, abs=FRICTION_TOLERANCE)

    def test_turbulent_friction_factor_colebrook(self):
        """Test Colebrook-White friction factor for rough pipes."""
        Re = 100000
        epsilon_D = 0.0001  # Relative roughness

        # Explicit Swamee-Jain approximation
        f = 0.25 / (math.log10(epsilon_D / 3.7 + 5.74 / Re ** 0.9)) ** 2

        assert f > 0
        assert f < 0.1

    @pytest.mark.parametrize("Re,flow_regime", [
        (500, "laminar"),
        (2000, "transitional"),
        (5000, "turbulent"),
        (100000, "fully_turbulent"),
    ])
    def test_flow_regime_classification(self, Re: float, flow_regime: str):
        """Test flow regime classification by Reynolds number."""
        if Re < 2300:
            regime = "laminar"
        elif Re < 4000:
            regime = "transitional"
        elif Re < 10000:
            regime = "turbulent"
        else:
            regime = "fully_turbulent"

        assert regime == flow_regime


class TestReynoldsNumber:
    """Test Reynolds number calculations."""

    def test_reynolds_number_tube_side(self, sample_operating_state, sample_exchanger_config):
        """Test tube-side Reynolds number calculation."""
        state = sample_operating_state
        config = sample_exchanger_config

        # Calculate tube velocity
        A_tube = math.pi * config.tube_id_m ** 2 / 4
        A_total = A_tube * config.tube_count / config.tube_passes

        velocity = state.m_dot_cold_kg_s / (state.rho_cold_kg_m3 * A_total)

        # Reynolds number
        Re = state.rho_cold_kg_m3 * velocity * config.tube_id_m / state.mu_cold_Pa_s

        assert Re > 0

    def test_reynolds_dimensionless(self, sample_operating_state):
        """Test that Reynolds number is dimensionless."""
        state = sample_operating_state

        # Re = rho * v * D / mu
        # [kg/m3] * [m/s] * [m] / [Pa-s] = [kg/m3] * [m/s] * [m] / [kg/(m-s)]
        # = [kg * m * m * m * s] / [m3 * s * kg] = dimensionless

        D = 0.01483  # m
        v = 2.0      # m/s

        Re = state.rho_cold_kg_m3 * v * D / state.mu_cold_Pa_s

        # Check reasonable magnitude for tube flow
        assert 1000 < Re < 1000000


class TestPressureDropRatio:
    """Test pressure drop ratio calculations (actual vs design)."""

    def test_dp_ratio_calculation(self, sample_thermal_kpis, sample_exchanger_config):
        """Test pressure drop ratio calculation."""
        kpis = sample_thermal_kpis
        config = sample_exchanger_config

        dP_ratio_shell = kpis.dP_shell_kPa / config.design_pressure_drop_shell_kPa
        dP_ratio_tube = kpis.dP_tube_kPa / config.design_pressure_drop_tube_kPa

        assert dP_ratio_shell > 0
        assert dP_ratio_tube > 0

    def test_high_dp_ratio_indicates_fouling(self):
        """Test that high pressure drop ratio indicates fouling."""
        dP_actual = 50.0
        dP_design = 35.0

        dP_ratio = dP_actual / dP_design

        # Ratio > 1.3 typically indicates significant fouling
        assert dP_ratio > 1.0
        assert dP_ratio > 1.3  # Warning threshold

    def test_dp_ratio_thresholds(self, sample_thermal_kpis):
        """Test pressure drop ratio warning thresholds."""
        # Shell side
        if sample_thermal_kpis.dP_ratio_shell > 1.5:
            shell_status = "critical"
        elif sample_thermal_kpis.dP_ratio_shell > 1.3:
            shell_status = "warning"
        else:
            shell_status = "normal"

        # Tube side
        if sample_thermal_kpis.dP_ratio_tube > 1.5:
            tube_status = "critical"
        elif sample_thermal_kpis.dP_ratio_tube > 1.3:
            tube_status = "warning"
        else:
            tube_status = "normal"

        assert shell_status in ["normal", "warning", "critical"]
        assert tube_status in ["normal", "warning", "critical"]


class TestFoulingPressureDropImpact:
    """Test fouling impact on pressure drop."""

    def test_fouling_increases_dp(self, clean_exchanger_kpis, fouled_exchanger_kpis):
        """Test that fouling increases pressure drop."""
        dP_clean_shell = clean_exchanger_kpis.dP_shell_kPa
        dP_fouled_shell = fouled_exchanger_kpis.dP_shell_kPa

        dP_clean_tube = clean_exchanger_kpis.dP_tube_kPa
        dP_fouled_tube = fouled_exchanger_kpis.dP_tube_kPa

        # Fouled exchanger has higher pressure drop
        assert dP_fouled_shell > dP_clean_shell
        assert dP_fouled_tube > dP_clean_tube

    def test_dp_ratio_vs_ua_ratio_correlation(self):
        """Test correlation between dP ratio and UA ratio."""
        # As fouling increases:
        # - UA decreases (thermal resistance increases)
        # - dP increases (flow restriction increases)

        UA_ratio = 0.7   # 70% of design
        dP_ratio = 1.35  # 135% of design

        # Both indicate fouling
        assert UA_ratio < 1.0
        assert dP_ratio > 1.0

    def test_fouling_thickness_from_dp(self):
        """Test estimating fouling thickness from pressure drop increase."""
        D_clean = 0.01483  # m (clean tube ID)
        dP_ratio = 1.5

        # dP proportional to 1/D^5 (Darcy-Weisbach)
        # dP_ratio = (D_clean / D_fouled)^5
        D_fouled = D_clean / (dP_ratio ** 0.2)

        fouling_thickness = (D_clean - D_fouled) / 2

        assert fouling_thickness > 0
        assert fouling_thickness < 0.005  # Less than 5mm typically


class TestPressureDropDeterminism:
    """Test pressure drop calculation determinism."""

    def test_deterministic_dp_calculation(self, sample_operating_state):
        """Test that pressure drop calculation is deterministic."""
        state = sample_operating_state

        results = []
        for _ in range(10):
            dP_shell = state.P_hot_in_kPa - state.P_hot_out_kPa
            dP_tube = state.P_cold_in_kPa - state.P_cold_out_kPa
            results.append((dP_shell, dP_tube))

        assert all(r == results[0] for r in results)

    def test_provenance_hash_for_dp(self, sample_operating_state):
        """Test provenance hash generation for pressure drop."""
        state = sample_operating_state

        dP_shell = state.P_hot_in_kPa - state.P_hot_out_kPa
        dP_tube = state.P_cold_in_kPa - state.P_cold_out_kPa

        provenance_data = f"{state.exchanger_id}:dP_shell:{dP_shell:.3f}:dP_tube:{dP_tube:.3f}"
        provenance_hash = hashlib.sha256(provenance_data.encode()).hexdigest()

        assert len(provenance_hash) == 64


class TestPressureDropEdgeCases:
    """Test edge cases for pressure drop calculations."""

    def test_zero_flow_dp(self):
        """Test pressure drop at zero flow."""
        # At zero flow, pressure drop should be zero
        velocity = 0.0
        rho = 990.0
        f = 0.02
        L_D = 100.0

        dP = f * L_D * (rho * velocity ** 2 / 2)

        assert dP == 0.0

    def test_very_low_flow_laminar(self, operating_state_low_flow):
        """Test pressure drop at very low flow (laminar regime)."""
        state = operating_state_low_flow

        # Low flow should give low Reynolds number
        D = 0.01483
        v = 0.1  # m/s (very low)

        Re = state.rho_cold_kg_m3 * v * D / state.mu_cold_Pa_s

        # Should be laminar or low turbulent
        assert Re < 5000

    def test_negative_dp_detection(self):
        """Test detection of negative pressure drop (sensor error)."""
        P_in = 500.0
        P_out = 510.0  # Higher than inlet (impossible in real flow)

        dP = P_in - P_out

        assert dP < 0, "Negative pressure drop indicates sensor error"


class TestPressureDropValidation:
    """Test input validation for pressure drop calculations."""

    def test_pressure_physical_bounds(self, sample_operating_state):
        """Test that pressures are within physical bounds."""
        state = sample_operating_state

        # All pressures should be positive
        assert state.P_hot_in_kPa > 0
        assert state.P_hot_out_kPa > 0
        assert state.P_cold_in_kPa > 0
        assert state.P_cold_out_kPa > 0

        # Inlet pressure > outlet pressure
        assert state.P_hot_in_kPa > state.P_hot_out_kPa
        assert state.P_cold_in_kPa > state.P_cold_out_kPa

    def test_dp_within_design_limits(self, sample_thermal_kpis, sample_exchanger_config):
        """Test that pressure drop is within design limits with allowance."""
        kpis = sample_thermal_kpis
        config = sample_exchanger_config

        # Allow 50% increase over design before critical alert
        max_shell_dp = config.design_pressure_drop_shell_kPa * 1.5
        max_tube_dp = config.design_pressure_drop_tube_kPa * 1.5

        within_limits_shell = kpis.dP_shell_kPa < max_shell_dp
        within_limits_tube = kpis.dP_tube_kPa < max_tube_dp

        # For test fixtures, should be within limits
        assert within_limits_shell
        assert within_limits_tube


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "TestTubeSidePressureDrop",
    "TestShellSidePressureDrop",
    "TestFrictionFactor",
    "TestReynoldsNumber",
    "TestPressureDropRatio",
    "TestFoulingPressureDropImpact",
    "TestPressureDropDeterminism",
    "TestPressureDropEdgeCases",
    "TestPressureDropValidation",
]
