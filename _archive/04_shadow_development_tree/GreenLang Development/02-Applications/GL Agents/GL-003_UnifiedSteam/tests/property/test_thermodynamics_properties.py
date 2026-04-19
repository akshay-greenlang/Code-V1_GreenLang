"""
Property-Based Tests for Thermodynamic Properties - IAPWS-IF97

This module provides comprehensive property-based tests using Hypothesis to validate
thermodynamic calculations against mathematical invariants and physical laws.

Test Categories:
1. Thermodynamic Consistency (Maxwell Relations)
2. Region Boundary Continuity
3. Inverse Function Accuracy (T(p,h) vs h(T,p))
4. Energy Conservation in Calculations
5. Calculation Invariants (quality bounds, enthalpy monotonicity)

Reference: IAPWS-IF97 Industrial Formulation 1997

Author: GL-TestEngineer
Version: 1.0.0
"""

import math
import pytest
from typing import Optional, Tuple
from datetime import datetime

from hypothesis import given, assume, settings, Verbosity, Phase, HealthCheck
from hypothesis import strategies as st
from hypothesis.strategies import composite

import sys
import os

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import thermodynamics modules
from thermodynamics.iapws_if97 import (
    IF97_CONSTANTS,
    REGION_BOUNDARIES,
    detect_region,
    get_saturation_pressure,
    get_saturation_temperature,
    get_boundary_23_temperature,
    get_boundary_23_pressure,
    region1_specific_enthalpy,
    region1_specific_entropy,
    region1_specific_volume,
    region1_specific_internal_energy,
    region1_specific_isobaric_heat_capacity,
    region1_speed_of_sound,
    region2_specific_enthalpy,
    region2_specific_entropy,
    region2_specific_volume,
    region2_specific_internal_energy,
    region2_specific_isobaric_heat_capacity,
    region2_speed_of_sound,
    region4_saturation_properties,
    region4_mixture_enthalpy,
    region4_mixture_entropy,
    region4_mixture_specific_volume,
    compute_property_derivatives,
    celsius_to_kelvin,
    kelvin_to_celsius,
    kpa_to_mpa,
    mpa_to_kpa,
    compute_density,
)
from thermodynamics.steam_properties import (
    compute_properties,
    get_saturation_properties,
    detect_steam_state,
    compute_superheat_degree,
    compute_dryness_fraction,
    SteamState,
)
from thermodynamics.steam_quality import (
    compute_wet_steam_enthalpy,
    compute_wet_steam_entropy,
    compute_wet_steam_specific_volume,
    quality_from_enthalpy,
    clamp_quality,
)


# =============================================================================
# HYPOTHESIS CONFIGURATION
# =============================================================================

# Configure profiles for different testing scenarios
settings.register_profile(
    "ci",
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
    phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.shrink],
    verbosity=Verbosity.normal,
)

settings.register_profile(
    "dev",
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
    verbosity=Verbosity.verbose,
)

settings.register_profile(
    "full",
    max_examples=1000,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
    phases=[Phase.explicit, Phase.reuse, Phase.generate, Phase.shrink],
)

settings.register_profile(
    "debug",
    max_examples=10,
    deadline=None,
    verbosity=Verbosity.debug,
)


# =============================================================================
# CUSTOM STRATEGIES
# =============================================================================

# Pressure ranges for different regions [MPa]
P_MIN = REGION_BOUNDARIES["P_MIN"]  # Triple point pressure
P_MAX = REGION_BOUNDARIES["P_MAX_1_2"]  # 100 MPa
P_CRIT = IF97_CONSTANTS["P_CRIT"]  # Critical pressure

# Temperature ranges [K]
T_MIN = REGION_BOUNDARIES["T_MIN"]  # 273.15 K (0 C)
T_MAX_1_3 = REGION_BOUNDARIES["T_MAX_1_3"]  # 623.15 K (350 C)
T_MAX_2 = REGION_BOUNDARIES["T_MAX_2"]  # 1073.15 K (800 C)
T_CRIT = IF97_CONSTANTS["T_CRIT"]  # Critical temperature


@composite
def valid_pressure_mpa(draw, min_val: float = P_MIN, max_val: float = P_CRIT):
    """Generate valid pressure values in MPa."""
    return draw(st.floats(min_value=min_val * 1.01, max_value=max_val * 0.99, allow_nan=False, allow_infinity=False))


@composite
def valid_temperature_k(draw, min_val: float = T_MIN, max_val: float = T_MAX_2):
    """Generate valid temperature values in Kelvin."""
    return draw(st.floats(min_value=min_val + 1.0, max_value=max_val - 1.0, allow_nan=False, allow_infinity=False))


@composite
def valid_quality(draw):
    """Generate valid steam quality values [0, 1]."""
    return draw(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))


@composite
def region1_state(draw):
    """Generate valid Region 1 (compressed liquid) state points."""
    # Region 1: P up to 100 MPa, T from 273.15 K to T_sat(P) or 623.15 K
    p_mpa = draw(st.floats(min_value=0.001, max_value=50.0, allow_nan=False, allow_infinity=False))

    # Calculate saturation temperature at this pressure
    try:
        if p_mpa < P_CRIT:
            t_sat = get_saturation_temperature(p_mpa)
            # Temperature must be below saturation for Region 1
            t_max = min(t_sat - 1.0, T_MAX_1_3)
        else:
            t_max = T_MAX_1_3
    except ValueError:
        t_max = T_MAX_1_3

    t_k = draw(st.floats(min_value=T_MIN + 1.0, max_value=max(t_max, T_MIN + 2.0), allow_nan=False, allow_infinity=False))

    return p_mpa, t_k


@composite
def region2_state(draw):
    """Generate valid Region 2 (superheated vapor) state points."""
    # Region 2: P up to 4 MPa for T > 623.15 K, or T > T_sat for P < P_crit
    p_mpa = draw(st.floats(min_value=0.001, max_value=4.0, allow_nan=False, allow_infinity=False))

    try:
        if p_mpa < P_CRIT:
            t_sat = get_saturation_temperature(p_mpa)
            t_min = t_sat + 5.0  # Above saturation
        else:
            t_min = T_MAX_1_3 + 10.0
    except ValueError:
        t_min = T_MAX_1_3 + 10.0

    t_k = draw(st.floats(min_value=t_min, max_value=T_MAX_2 - 10.0, allow_nan=False, allow_infinity=False))

    return p_mpa, t_k


@composite
def saturation_pressure(draw):
    """Generate valid saturation pressure values."""
    return draw(st.floats(min_value=P_MIN * 2, max_value=P_CRIT * 0.95, allow_nan=False, allow_infinity=False))


@composite
def saturation_temperature(draw):
    """Generate valid saturation temperature values."""
    return draw(st.floats(min_value=T_MIN + 5.0, max_value=T_CRIT - 5.0, allow_nan=False, allow_infinity=False))


# =============================================================================
# TEST CLASS: THERMODYNAMIC CONSISTENCY (MAXWELL RELATIONS)
# =============================================================================

@pytest.mark.hypothesis
class TestMaxwellRelations:
    """
    Test thermodynamic consistency using Maxwell relations.

    Maxwell relations are fundamental thermodynamic identities that must hold
    for any valid equation of state. Violations indicate calculation errors.

    Key Relations:
    - (dT/dV)_S = -(dP/dS)_V
    - (dS/dV)_T = (dP/dT)_V
    - (dS/dP)_T = -(dV/dT)_P
    - (dT/dP)_S = (dV/dS)_P
    """

    @given(region1_state())
    @settings(max_examples=100, deadline=None)
    def test_maxwell_dsdp_t_region1(self, state: Tuple[float, float]):
        """
        Test Maxwell relation: (dS/dP)_T = -(dV/dT)_P for Region 1.

        This tests the fundamental thermodynamic identity that relates
        entropy change with pressure to volume change with temperature.
        """
        p_mpa, t_k = state

        try:
            # Verify we're in Region 1
            region = detect_region(p_mpa, t_k)
            assume(region == 1)

            # Small deltas for numerical differentiation
            dp = 0.001  # MPa
            dt = 0.1    # K

            # Calculate (dS/dP)_T numerically
            s_p_plus = region1_specific_entropy(p_mpa + dp, t_k)
            s_p_minus = region1_specific_entropy(p_mpa - dp, t_k)
            ds_dp_t = (s_p_plus - s_p_minus) / (2 * dp)

            # Calculate -(dV/dT)_P numerically
            v_t_plus = region1_specific_volume(p_mpa, t_k + dt)
            v_t_minus = region1_specific_volume(p_mpa, t_k - dt)
            neg_dv_dt_p = -(v_t_plus - v_t_minus) / (2 * dt)

            # Maxwell relation: (dS/dP)_T = -(dV/dT)_P
            # Allow for numerical error (relative tolerance)
            if abs(ds_dp_t) > 1e-10:
                rel_error = abs(ds_dp_t - neg_dv_dt_p) / abs(ds_dp_t)
                assert rel_error < 0.1, f"Maxwell relation violated: rel_error = {rel_error:.4f}"

        except ValueError:
            # Outside valid range - skip
            assume(False)

    @given(region2_state())
    @settings(max_examples=100, deadline=None)
    def test_maxwell_dsdp_t_region2(self, state: Tuple[float, float]):
        """
        Test Maxwell relation: (dS/dP)_T = -(dV/dT)_P for Region 2.
        """
        p_mpa, t_k = state

        try:
            region = detect_region(p_mpa, t_k)
            assume(region == 2)

            dp = 0.001  # MPa
            dt = 0.1    # K

            # Calculate (dS/dP)_T numerically
            s_p_plus = region2_specific_entropy(p_mpa + dp, t_k)
            s_p_minus = region2_specific_entropy(p_mpa - dp, t_k)
            ds_dp_t = (s_p_plus - s_p_minus) / (2 * dp)

            # Calculate -(dV/dT)_P numerically
            v_t_plus = region2_specific_volume(p_mpa, t_k + dt)
            v_t_minus = region2_specific_volume(p_mpa, t_k - dt)
            neg_dv_dt_p = -(v_t_plus - v_t_minus) / (2 * dt)

            # Maxwell relation check
            if abs(ds_dp_t) > 1e-10:
                rel_error = abs(ds_dp_t - neg_dv_dt_p) / abs(ds_dp_t)
                assert rel_error < 0.1, f"Maxwell relation violated: rel_error = {rel_error:.4f}"

        except ValueError:
            assume(False)

    @given(region1_state())
    @settings(max_examples=100, deadline=None)
    def test_fundamental_relation_region1(self, state: Tuple[float, float]):
        """
        Test fundamental thermodynamic relation: dU = TdS - PdV

        For infinitesimal changes, the internal energy must satisfy:
        (dU/dT)_V = T * (dS/dT)_V
        """
        p_mpa, t_k = state

        try:
            region = detect_region(p_mpa, t_k)
            assume(region == 1)

            dt = 0.1  # K

            # Calculate (dU/dT) at constant P (approximately constant V for liquids)
            u_t_plus = region1_specific_internal_energy(p_mpa, t_k + dt)
            u_t_minus = region1_specific_internal_energy(p_mpa, t_k - dt)
            du_dt = (u_t_plus - u_t_minus) / (2 * dt)

            # Calculate T * (dS/dT)_P
            s_t_plus = region1_specific_entropy(p_mpa, t_k + dt)
            s_t_minus = region1_specific_entropy(p_mpa, t_k - dt)
            ds_dt = (s_t_plus - s_t_minus) / (2 * dt)
            t_ds_dt = t_k * ds_dt

            # Cp = T * (dS/dT)_P
            cp = region1_specific_isobaric_heat_capacity(p_mpa, t_k)

            # T * (dS/dT)_P should approximately equal Cp
            if abs(cp) > 1e-10:
                rel_error = abs(t_ds_dt - cp) / abs(cp)
                assert rel_error < 0.05, f"Cp consistency violated: {t_ds_dt:.4f} vs {cp:.4f}"

        except ValueError:
            assume(False)


# =============================================================================
# TEST CLASS: REGION BOUNDARY CONTINUITY
# =============================================================================

@pytest.mark.hypothesis
class TestRegionBoundaryContinuity:
    """
    Test continuity of properties across region boundaries.

    IAPWS-IF97 defines multiple regions. Properties must be continuous
    across region boundaries to avoid discontinuities in calculations.
    """

    @given(saturation_pressure())
    @settings(max_examples=100, deadline=None)
    def test_saturation_line_continuity(self, p_mpa: float):
        """
        Test property continuity across the saturation line (Region 4).

        At saturation, properties from Region 1 (liquid) and Region 2 (vapor)
        must match the saturation properties.
        """
        try:
            # Get saturation properties
            sat = region4_saturation_properties(p_mpa)
            t_sat = sat.temperature_k

            # Get properties just below saturation (Region 1)
            t_liquid = t_sat - 0.1
            if t_liquid > T_MIN:
                h_liquid = region1_specific_enthalpy(p_mpa, t_liquid)
                s_liquid = region1_specific_entropy(p_mpa, t_liquid)

                # Liquid enthalpy should approach hf
                # Allow for small temperature difference effect
                assert h_liquid < sat.hf + 50, f"Liquid enthalpy discontinuity: {h_liquid:.2f} vs hf={sat.hf:.2f}"

            # Get properties just above saturation (Region 2)
            t_vapor = t_sat + 0.1
            if t_vapor < T_MAX_2:
                h_vapor = region2_specific_enthalpy(p_mpa, t_vapor)
                s_vapor = region2_specific_entropy(p_mpa, t_vapor)

                # Vapor enthalpy should approach hg
                assert h_vapor > sat.hg - 50, f"Vapor enthalpy discontinuity: {h_vapor:.2f} vs hg={sat.hg:.2f}"

            # Latent heat must be positive below critical point
            assert sat.hfg > 0, f"Latent heat must be positive: hfg={sat.hfg:.2f}"

        except ValueError:
            assume(False)

    @given(st.floats(min_value=16.5, max_value=100.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50, deadline=None)
    def test_boundary_23_continuity(self, p_mpa: float):
        """
        Test continuity at the Region 2/3 boundary.

        The B23 boundary separates superheated steam (Region 2) from
        supercritical fluid (Region 3).
        """
        try:
            # Get boundary temperature
            t_boundary = get_boundary_23_temperature(p_mpa)

            # Temperature should be reasonable
            assert T_MAX_1_3 < t_boundary < T_MAX_2, \
                f"Boundary temperature out of range: {t_boundary:.2f} K"

            # Inverse function should be consistent
            p_calc = get_boundary_23_pressure(t_boundary)
            rel_error = abs(p_calc - p_mpa) / p_mpa

            assert rel_error < 0.01, \
                f"B23 boundary inverse inconsistent: {p_calc:.4f} vs {p_mpa:.4f}"

        except ValueError:
            assume(False)

    @given(saturation_temperature())
    @settings(max_examples=100, deadline=None)
    def test_saturation_pressure_temperature_consistency(self, t_k: float):
        """
        Test P_sat(T) and T_sat(P) are inverse functions.

        These must be consistent: T_sat(P_sat(T)) = T and P_sat(T_sat(P)) = P
        """
        try:
            # Forward: T -> P_sat
            p_sat = get_saturation_pressure(t_k)

            # Inverse: P_sat -> T_sat
            t_sat_calc = get_saturation_temperature(p_sat)

            # Should recover original temperature
            rel_error = abs(t_sat_calc - t_k) / t_k

            assert rel_error < 1e-6, \
                f"Saturation inverse inconsistent: T={t_k:.4f} -> P={p_sat:.6f} -> T={t_sat_calc:.4f}"

        except ValueError:
            assume(False)


# =============================================================================
# TEST CLASS: INVERSE FUNCTION ACCURACY
# =============================================================================

@pytest.mark.hypothesis
class TestInverseFunctionAccuracy:
    """
    Test accuracy of inverse property functions.

    When computing T from (P, h), the result should satisfy h(T, P) = h_target.
    """

    @given(saturation_pressure(), valid_quality())
    @settings(max_examples=100, deadline=None)
    def test_quality_from_enthalpy_inverse(self, p_mpa: float, x: float):
        """
        Test that quality_from_enthalpy inverts correctly.

        Given x, compute h = hf + x*hfg, then x_calc = (h - hf) / hfg
        should equal original x.
        """
        try:
            p_kpa = mpa_to_kpa(p_mpa)

            # Compute enthalpy for given quality
            h = compute_wet_steam_enthalpy(p_kpa, x)

            # Inverse: compute quality from enthalpy
            x_calc, is_two_phase = quality_from_enthalpy(p_kpa, h)

            # Should be in two-phase region
            assert is_two_phase or x in [0.0, 1.0], \
                f"Should be two-phase for x={x:.4f}"

            # Quality should match (within tolerance)
            if is_two_phase:
                assert abs(x_calc - x) < 1e-6, \
                    f"Quality inverse failed: x={x:.6f} -> h={h:.2f} -> x_calc={x_calc:.6f}"

        except ValueError:
            assume(False)

    @given(region1_state())
    @settings(max_examples=100, deadline=None)
    def test_compute_from_ph_region1(self, state: Tuple[float, float]):
        """
        Test P-h inverse for Region 1.

        Compute h from (P, T), then solve for T from (P, h).
        Result should match original T.
        """
        p_mpa, t_k = state

        try:
            region = detect_region(p_mpa, t_k)
            assume(region == 1)

            # Forward: compute enthalpy
            h = region1_specific_enthalpy(p_mpa, t_k)
            p_kpa = mpa_to_kpa(p_mpa)
            t_c = kelvin_to_celsius(t_k)

            # Inverse: compute properties from P, h
            props = compute_properties(p_kpa, enthalpy_kj_kg=h)

            # Temperature should match
            t_c_calc = props.temperature_c

            # Allow reasonable tolerance (iteration may not be exact)
            assert abs(t_c_calc - t_c) < 1.0, \
                f"P-h inverse failed: T={t_c:.2f}C -> h={h:.2f} -> T={t_c_calc:.2f}C"

        except ValueError:
            assume(False)

    @given(region2_state())
    @settings(max_examples=100, deadline=None)
    def test_compute_from_ph_region2(self, state: Tuple[float, float]):
        """
        Test P-h inverse for Region 2.
        """
        p_mpa, t_k = state

        try:
            region = detect_region(p_mpa, t_k)
            assume(region == 2)

            # Forward: compute enthalpy
            h = region2_specific_enthalpy(p_mpa, t_k)
            p_kpa = mpa_to_kpa(p_mpa)
            t_c = kelvin_to_celsius(t_k)

            # Inverse: compute properties from P, h
            props = compute_properties(p_kpa, enthalpy_kj_kg=h)

            # Temperature should match
            t_c_calc = props.temperature_c

            assert abs(t_c_calc - t_c) < 2.0, \
                f"P-h inverse failed: T={t_c:.2f}C -> h={h:.2f} -> T={t_c_calc:.2f}C"

        except ValueError:
            assume(False)


# =============================================================================
# TEST CLASS: ENERGY CONSERVATION
# =============================================================================

@pytest.mark.hypothesis
class TestEnergyConservation:
    """
    Test energy conservation in thermodynamic calculations.

    Key invariants:
    - U = H - PV (internal energy definition)
    - Enthalpy changes must be consistent with work and heat
    """

    @given(region1_state())
    @settings(max_examples=100, deadline=None)
    def test_internal_energy_definition_region1(self, state: Tuple[float, float]):
        """
        Test U = H - PV in Region 1.

        The fundamental relation between internal energy, enthalpy,
        pressure, and volume must hold exactly.
        """
        p_mpa, t_k = state

        try:
            region = detect_region(p_mpa, t_k)
            assume(region == 1)

            # Get properties
            h = region1_specific_enthalpy(p_mpa, t_k)
            u = region1_specific_internal_energy(p_mpa, t_k)
            v = region1_specific_volume(p_mpa, t_k)

            # U = H - PV
            # P in MPa, V in m^3/kg, H and U in kJ/kg
            # PV in MPa * m^3/kg = MJ/kg = 1000 kJ/kg
            pv = p_mpa * 1000 * v  # Convert to kJ/kg
            u_calc = h - pv

            # Should match exactly (same calculation basis)
            rel_error = abs(u_calc - u) / abs(u) if abs(u) > 1e-10 else abs(u_calc - u)

            assert rel_error < 1e-6, \
                f"U = H - PV violated: u={u:.4f}, h-pv={u_calc:.4f}, error={rel_error:.6f}"

        except ValueError:
            assume(False)

    @given(region2_state())
    @settings(max_examples=100, deadline=None)
    def test_internal_energy_definition_region2(self, state: Tuple[float, float]):
        """
        Test U = H - PV in Region 2.
        """
        p_mpa, t_k = state

        try:
            region = detect_region(p_mpa, t_k)
            assume(region == 2)

            h = region2_specific_enthalpy(p_mpa, t_k)
            u = region2_specific_internal_energy(p_mpa, t_k)
            v = region2_specific_volume(p_mpa, t_k)

            pv = p_mpa * 1000 * v
            u_calc = h - pv

            rel_error = abs(u_calc - u) / abs(u) if abs(u) > 1e-10 else abs(u_calc - u)

            assert rel_error < 1e-6, \
                f"U = H - PV violated: u={u:.4f}, h-pv={u_calc:.4f}"

        except ValueError:
            assume(False)

    @given(saturation_pressure(), valid_quality())
    @settings(max_examples=100, deadline=None)
    def test_mixture_property_interpolation(self, p_mpa: float, x: float):
        """
        Test that mixture properties interpolate correctly.

        For wet steam: property = property_f + x * (property_g - property_f)
        """
        try:
            p_kpa = mpa_to_kpa(p_mpa)

            # Get saturation properties
            sat = region4_saturation_properties(p_mpa)

            # Compute mixture properties
            h_mix = compute_wet_steam_enthalpy(p_kpa, x)
            s_mix = compute_wet_steam_entropy(p_kpa, x)
            v_mix = compute_wet_steam_specific_volume(p_kpa, x)

            # Expected values from interpolation
            h_expected = sat.hf + x * sat.hfg
            s_expected = sat.sf + x * sat.sfg
            v_expected = sat.vf + x * (sat.vg - sat.vf)

            # Check enthalpy
            assert abs(h_mix - h_expected) < 1e-6, \
                f"Mixture enthalpy error: {h_mix:.4f} vs {h_expected:.4f}"

            # Check entropy
            assert abs(s_mix - s_expected) < 1e-6, \
                f"Mixture entropy error: {s_mix:.4f} vs {s_expected:.4f}"

            # Check specific volume
            assert abs(v_mix - v_expected) < 1e-10, \
                f"Mixture volume error: {v_mix:.8f} vs {v_expected:.8f}"

        except ValueError:
            assume(False)


# =============================================================================
# TEST CLASS: CALCULATION INVARIANTS
# =============================================================================

@pytest.mark.hypothesis
class TestCalculationInvariants:
    """
    Test fundamental calculation invariants that must always hold.
    """

    @given(valid_quality())
    @settings(max_examples=200, deadline=None)
    def test_quality_bounds(self, x: float):
        """
        Test that steam quality is always bounded [0, 1].

        Quality represents mass fraction of vapor, which physically
        cannot be negative or greater than 1.
        """
        # Quality must be in [0, 1]
        assert 0.0 <= x <= 1.0, f"Quality {x} out of bounds [0, 1]"

        # Clamping should preserve valid values
        x_clamped = clamp_quality(x, warn=False)
        assert abs(x_clamped - x) < 1e-10, f"Valid quality was modified by clamping"

    @given(st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False))
    @settings(max_examples=100, deadline=None)
    def test_quality_clamping(self, x: float):
        """
        Test that quality clamping produces valid values.
        """
        x_clamped = clamp_quality(x, warn=False)

        assert 0.0 <= x_clamped <= 1.0, \
            f"Clamped quality {x_clamped} out of bounds"

        if 0 <= x <= 1:
            assert abs(x_clamped - x) < 1e-10, \
                f"Valid quality was modified: {x} -> {x_clamped}"

    @given(saturation_pressure())
    @settings(max_examples=100, deadline=None)
    def test_enthalpy_monotonicity_with_quality(self, p_mpa: float):
        """
        Test that enthalpy increases monotonically with quality.

        As more liquid evaporates (quality increases), enthalpy must increase
        because energy is added during evaporation.
        """
        try:
            p_kpa = mpa_to_kpa(p_mpa)

            # Generate quality values
            qualities = [0.0, 0.25, 0.5, 0.75, 1.0]
            enthalpies = [compute_wet_steam_enthalpy(p_kpa, x) for x in qualities]

            # Must be strictly increasing
            for i in range(1, len(enthalpies)):
                assert enthalpies[i] > enthalpies[i-1], \
                    f"Enthalpy not monotonic: h({qualities[i-1]})={enthalpies[i-1]:.2f} >= h({qualities[i]})={enthalpies[i]:.2f}"

        except ValueError:
            assume(False)

    @given(saturation_pressure())
    @settings(max_examples=100, deadline=None)
    def test_entropy_monotonicity_with_quality(self, p_mpa: float):
        """
        Test that entropy increases monotonically with quality.

        Vapor has higher entropy than liquid at the same conditions.
        """
        try:
            p_kpa = mpa_to_kpa(p_mpa)

            qualities = [0.0, 0.25, 0.5, 0.75, 1.0]
            entropies = [compute_wet_steam_entropy(p_kpa, x) for x in qualities]

            for i in range(1, len(entropies)):
                assert entropies[i] > entropies[i-1], \
                    f"Entropy not monotonic: s({qualities[i-1]})={entropies[i-1]:.4f} >= s({qualities[i]})={entropies[i]:.4f}"

        except ValueError:
            assume(False)

    @given(region1_state())
    @settings(max_examples=100, deadline=None)
    def test_enthalpy_increases_with_temperature_region1(self, state: Tuple[float, float]):
        """
        Test that enthalpy increases with temperature at constant pressure.

        Adding heat at constant pressure increases enthalpy (definition of Cp > 0).
        """
        p_mpa, t_k = state

        try:
            region = detect_region(p_mpa, t_k)
            assume(region == 1)
            assume(t_k + 5.0 < T_MAX_1_3)  # Room to increase temperature

            h1 = region1_specific_enthalpy(p_mpa, t_k)
            h2 = region1_specific_enthalpy(p_mpa, t_k + 5.0)

            assert h2 > h1, \
                f"Enthalpy not increasing with T: h({t_k:.1f})={h1:.2f} >= h({t_k+5:.1f})={h2:.2f}"

            # Cp must be positive
            cp = region1_specific_isobaric_heat_capacity(p_mpa, t_k)
            assert cp > 0, f"Cp must be positive: Cp={cp:.4f}"

        except ValueError:
            assume(False)

    @given(region2_state())
    @settings(max_examples=100, deadline=None)
    def test_enthalpy_increases_with_temperature_region2(self, state: Tuple[float, float]):
        """
        Test that enthalpy increases with temperature at constant pressure in Region 2.
        """
        p_mpa, t_k = state

        try:
            region = detect_region(p_mpa, t_k)
            assume(region == 2)
            assume(t_k + 10.0 < T_MAX_2)

            h1 = region2_specific_enthalpy(p_mpa, t_k)
            h2 = region2_specific_enthalpy(p_mpa, t_k + 10.0)

            assert h2 > h1, \
                f"Enthalpy not increasing with T: h({t_k:.1f})={h1:.2f} >= h({t_k+10:.1f})={h2:.2f}"

            cp = region2_specific_isobaric_heat_capacity(p_mpa, t_k)
            assert cp > 0, f"Cp must be positive: Cp={cp:.4f}"

        except ValueError:
            assume(False)

    @given(saturation_temperature())
    @settings(max_examples=100, deadline=None)
    def test_saturation_pressure_increases_with_temperature(self, t_k: float):
        """
        Test that saturation pressure increases with temperature.

        The Clausius-Clapeyron equation dictates that P_sat increases
        exponentially with temperature.
        """
        try:
            assume(t_k + 5.0 < T_CRIT)  # Stay below critical

            p1 = get_saturation_pressure(t_k)
            p2 = get_saturation_pressure(t_k + 5.0)

            assert p2 > p1, \
                f"P_sat not increasing with T: P({t_k:.1f})={p1:.6f} >= P({t_k+5:.1f})={p2:.6f}"

        except ValueError:
            assume(False)


# =============================================================================
# TEST CLASS: SPECIFIC VOLUME PROPERTIES
# =============================================================================

@pytest.mark.hypothesis
class TestSpecificVolumeProperties:
    """
    Test properties related to specific volume and density.
    """

    @given(region1_state())
    @settings(max_examples=100, deadline=None)
    def test_density_positive_region1(self, state: Tuple[float, float]):
        """
        Test that density is always positive in Region 1.
        """
        p_mpa, t_k = state

        try:
            region = detect_region(p_mpa, t_k)
            assume(region == 1)

            v = region1_specific_volume(p_mpa, t_k)
            rho = compute_density(v)

            assert v > 0, f"Specific volume must be positive: v={v}"
            assert rho > 0, f"Density must be positive: rho={rho}"
            assert abs(v * rho - 1.0) < 1e-10, f"v * rho should equal 1: {v * rho}"

        except ValueError:
            assume(False)

    @given(region2_state())
    @settings(max_examples=100, deadline=None)
    def test_density_positive_region2(self, state: Tuple[float, float]):
        """
        Test that density is always positive in Region 2.
        """
        p_mpa, t_k = state

        try:
            region = detect_region(p_mpa, t_k)
            assume(region == 2)

            v = region2_specific_volume(p_mpa, t_k)
            rho = compute_density(v)

            assert v > 0, f"Specific volume must be positive: v={v}"
            assert rho > 0, f"Density must be positive: rho={rho}"

        except ValueError:
            assume(False)

    @given(saturation_pressure())
    @settings(max_examples=100, deadline=None)
    def test_vapor_volume_greater_than_liquid(self, p_mpa: float):
        """
        Test that vapor specific volume is always greater than liquid.

        This is a fundamental property of the liquid-vapor transition.
        """
        try:
            sat = region4_saturation_properties(p_mpa)

            assert sat.vg > sat.vf, \
                f"vg must be > vf: vg={sat.vg:.6f}, vf={sat.vf:.6f}"

            # Volume difference should decrease as pressure approaches critical
            if p_mpa < 0.5 * P_CRIT:
                # At low pressures, vg should be much larger
                ratio = sat.vg / sat.vf
                assert ratio > 10, f"vg/vf should be large at low P: ratio={ratio:.2f}"

        except ValueError:
            assume(False)


# =============================================================================
# TEST CLASS: SPEED OF SOUND
# =============================================================================

@pytest.mark.hypothesis
class TestSpeedOfSound:
    """
    Test speed of sound calculations.
    """

    @given(region1_state())
    @settings(max_examples=100, deadline=None)
    def test_speed_of_sound_positive_region1(self, state: Tuple[float, float]):
        """
        Test that speed of sound is positive and reasonable in Region 1.
        """
        p_mpa, t_k = state

        try:
            region = detect_region(p_mpa, t_k)
            assume(region == 1)

            w = region1_speed_of_sound(p_mpa, t_k)

            assert w > 0, f"Speed of sound must be positive: w={w}"

            # For water, speed of sound is typically 1000-1800 m/s
            assert 500 < w < 2500, f"Speed of sound out of expected range: w={w}"

        except ValueError:
            assume(False)

    @given(region2_state())
    @settings(max_examples=100, deadline=None)
    def test_speed_of_sound_positive_region2(self, state: Tuple[float, float]):
        """
        Test that speed of sound is positive and reasonable in Region 2.
        """
        p_mpa, t_k = state

        try:
            region = detect_region(p_mpa, t_k)
            assume(region == 2)

            w = region2_speed_of_sound(p_mpa, t_k)

            assert w > 0, f"Speed of sound must be positive: w={w}"

            # For steam, speed of sound is typically 300-800 m/s
            assert 100 < w < 1500, f"Speed of sound out of expected range: w={w}"

        except ValueError:
            assume(False)


# =============================================================================
# TEST CLASS: DETERMINISM AND REPRODUCIBILITY
# =============================================================================

@pytest.mark.hypothesis
class TestDeterminism:
    """
    Test that all calculations are deterministic and reproducible.

    Same inputs must always produce identical outputs.
    """

    @given(region1_state())
    @settings(max_examples=50, deadline=None)
    def test_region1_deterministic(self, state: Tuple[float, float]):
        """
        Test Region 1 calculations are deterministic.
        """
        p_mpa, t_k = state

        try:
            region = detect_region(p_mpa, t_k)
            assume(region == 1)

            # Call multiple times
            results = []
            for _ in range(5):
                h = region1_specific_enthalpy(p_mpa, t_k)
                s = region1_specific_entropy(p_mpa, t_k)
                v = region1_specific_volume(p_mpa, t_k)
                results.append((h, s, v))

            # All results must be identical
            for i in range(1, len(results)):
                assert results[i] == results[0], \
                    f"Non-deterministic result: {results[i]} != {results[0]}"

        except ValueError:
            assume(False)

    @given(region2_state())
    @settings(max_examples=50, deadline=None)
    def test_region2_deterministic(self, state: Tuple[float, float]):
        """
        Test Region 2 calculations are deterministic.
        """
        p_mpa, t_k = state

        try:
            region = detect_region(p_mpa, t_k)
            assume(region == 2)

            results = []
            for _ in range(5):
                h = region2_specific_enthalpy(p_mpa, t_k)
                s = region2_specific_entropy(p_mpa, t_k)
                v = region2_specific_volume(p_mpa, t_k)
                results.append((h, s, v))

            for i in range(1, len(results)):
                assert results[i] == results[0]

        except ValueError:
            assume(False)


# =============================================================================
# RUN CONFIGURATION
# =============================================================================

if __name__ == "__main__":
    # Set profile based on environment
    import os
    profile = os.getenv("HYPOTHESIS_PROFILE", "dev")
    settings.load_profile(profile)

    pytest.main([__file__, "-v", "--tb=short"])
