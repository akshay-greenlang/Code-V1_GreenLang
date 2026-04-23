"""
Unit Tests: Steam Thermodynamic Properties

Tests the thermodynamic property functions for steam:
1. Saturation properties (temperature, pressure)
2. Enthalpy calculations
3. Entropy calculations
4. Specific volume calculations
5. State determination (liquid, vapor, two-phase, supercritical)

Reference: IAPWS-IF97 Industrial Formulation
Author: GL-TestEngineer
Version: 1.0.0
Target Coverage: 85%+
"""

import pytest
import math
import hashlib
import json
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple
from decimal import Decimal


# =============================================================================
# Constants from IAPWS-IF97
# =============================================================================

# Critical point constants
CRITICAL_TEMPERATURE_K = 647.096
CRITICAL_PRESSURE_MPA = 22.064
CRITICAL_DENSITY_KG_M3 = 322.0

# Triple point constants
TRIPLE_TEMPERATURE_K = 273.16
TRIPLE_PRESSURE_MPA = 0.000611657

# Specific gas constant for water
R = 0.461526  # kJ/(kg.K)

# Tolerances for verification
TEMPERATURE_TOLERANCE_K = 0.1
PRESSURE_TOLERANCE_MPA = 0.001
ENTHALPY_TOLERANCE_KJ_KG = 0.5
ENTROPY_TOLERANCE_KJ_KG_K = 0.001
VOLUME_TOLERANCE_PERCENT = 0.5


# =============================================================================
# Enumerations
# =============================================================================

class SteamRegion(Enum):
    """IAPWS-IF97 regions."""
    REGION_1 = 1  # Subcooled liquid
    REGION_2 = 2  # Superheated vapor
    REGION_3 = 3  # Near-critical
    REGION_4 = 4  # Two-phase
    REGION_5 = 5  # High-temperature steam
    UNKNOWN = 0


class SteamState(Enum):
    """Steam thermodynamic state."""
    SUBCOOLED_LIQUID = auto()
    SATURATED_LIQUID = auto()
    WET_STEAM = auto()
    SATURATED_VAPOR = auto()
    SUPERHEATED_VAPOR = auto()
    SUPERCRITICAL = auto()
    UNKNOWN = auto()


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SteamProperties:
    """Complete steam properties at a state point."""
    pressure_mpa: float
    temperature_k: float
    specific_volume_m3_kg: float
    specific_enthalpy_kj_kg: float
    specific_entropy_kj_kg_k: float
    specific_internal_energy_kj_kg: float
    region: SteamRegion
    state: SteamState
    quality: Optional[float] = None
    cp_kj_kg_k: Optional[float] = None
    cv_kj_kg_k: Optional[float] = None
    speed_of_sound_m_s: Optional[float] = None


@dataclass
class SaturationPoint:
    """Saturation properties at a given pressure or temperature."""
    pressure_mpa: float
    temperature_k: float
    # Liquid properties
    vf_m3_kg: float  # Specific volume
    hf_kj_kg: float  # Enthalpy
    sf_kj_kg_k: float  # Entropy
    uf_kj_kg: float  # Internal energy
    # Vapor properties
    vg_m3_kg: float
    hg_kj_kg: float
    sg_kj_kg_k: float
    ug_kj_kg: float
    # Latent properties
    hfg_kj_kg: float
    sfg_kj_kg_k: float


# =============================================================================
# IAPWS-IF97 Reference Verification Points
# =============================================================================

# Table 5: Verification values for Region 1
REGION1_VERIFICATION = [
    # (T [K], P [MPa], v [m3/kg], h [kJ/kg], s [kJ/(kg.K)], u [kJ/kg])
    (300.0, 3.0, 0.00100215168e-2, 115.331273, 0.392294792, 112.324818),
    (300.0, 80.0, 0.000971180894e-2, 184.142828, 0.368563852, 106.448356),
    (500.0, 80.0, 0.00120241800e-2, 975.542239, 2.58041912, 971.934985),
]

# Table 15: Verification values for Region 2
REGION2_VERIFICATION = [
    # (T [K], P [MPa], v [m3/kg], h [kJ/kg], s [kJ/(kg.K)])
    (300.0, 0.001, 0.394913866e2, 2549.91145, 9.15546786),
    (700.0, 0.001, 0.923015898e2, 3335.68375, 10.1749996),
    (700.0, 30.0, 0.00542946619e-1, 2631.49474, 5.17540298),
]

# Table 33: Saturation temperature verification
SATURATION_T_VERIFICATION = [
    # (P [MPa], T [K])
    (0.1, 372.7559186),
    (1.0, 453.0356),
    (10.0, 584.1494),
]

# Table 35: Saturation pressure verification
SATURATION_P_VERIFICATION = [
    # (T [K], P [MPa])
    (300.0, 0.00353658941),
    (500.0, 2.63889776),
    (600.0, 12.3443146),
]


# =============================================================================
# Property Calculation Functions (Simplified for Testing)
# =============================================================================

def calculate_saturation_temperature(pressure_mpa: float) -> float:
    """
    Calculate saturation temperature from pressure.

    Uses backward equation from IAPWS-IF97 Region 4.
    """
    if pressure_mpa < TRIPLE_PRESSURE_MPA:
        raise ValueError(f"Pressure {pressure_mpa} below triple point")
    if pressure_mpa > CRITICAL_PRESSURE_MPA:
        raise ValueError(f"Pressure {pressure_mpa} above critical point")

    # Backward equation coefficients
    n = [
        0.11670521452767e4, -0.72421316703206e6, -0.17073846940092e2,
        0.12020824702470e5, -0.32325550322333e7, 0.14915108613530e2,
        -0.48232657361591e4, 0.40511340542057e6, -0.23855557567849,
        0.65017534844798e3
    ]

    beta = pressure_mpa ** 0.25
    E = beta**2 + n[2] * beta + n[5]
    F = n[0] * beta**2 + n[3] * beta + n[6]
    G = n[1] * beta**2 + n[4] * beta + n[7]
    D = 2 * G / (-F - math.sqrt(F**2 - 4 * E * G))

    T = (n[9] + D - math.sqrt((n[9] + D)**2 - 4 * (n[8] + n[9] * D))) / 2

    return T


def calculate_saturation_pressure(temperature_k: float) -> float:
    """
    Calculate saturation pressure from temperature.

    Uses equation from IAPWS-IF97 Region 4.
    """
    if temperature_k < TRIPLE_TEMPERATURE_K:
        raise ValueError(f"Temperature {temperature_k} below triple point")
    if temperature_k > CRITICAL_TEMPERATURE_K:
        raise ValueError(f"Temperature {temperature_k} above critical point")

    # Saturation equation coefficients
    n = [
        0.11670521452767e4, -0.72421316703206e6, -0.17073846940092e2,
        0.12020824702470e5, -0.32325550322333e7, 0.14915108613530e2,
        -0.48232657361591e4, 0.40511340542057e6, -0.23855557567849,
        0.65017534844798e3
    ]

    theta = temperature_k + n[8] / (temperature_k - n[9])
    A = theta**2 + n[0] * theta + n[1]
    B = n[2] * theta**2 + n[3] * theta + n[4]
    C = n[5] * theta**2 + n[6] * theta + n[7]

    P = (2 * C / (-B + math.sqrt(B**2 - 4 * A * C)))**4

    return P


def determine_region(pressure_mpa: float, temperature_k: float) -> SteamRegion:
    """
    Determine IAPWS-IF97 region for given P and T.
    """
    if pressure_mpa <= 0 or temperature_k <= 0:
        return SteamRegion.UNKNOWN

    if pressure_mpa > CRITICAL_PRESSURE_MPA and temperature_k > CRITICAL_TEMPERATURE_K:
        return SteamRegion.REGION_3  # Supercritical

    if temperature_k > 1073.15:
        if pressure_mpa <= 50:
            return SteamRegion.REGION_5
        return SteamRegion.UNKNOWN

    try:
        t_sat = calculate_saturation_temperature(pressure_mpa)
    except ValueError:
        # Above critical pressure
        return SteamRegion.REGION_3

    if abs(temperature_k - t_sat) < 0.1:
        return SteamRegion.REGION_4  # On saturation line

    if temperature_k < t_sat:
        return SteamRegion.REGION_1  # Subcooled liquid
    else:
        return SteamRegion.REGION_2  # Superheated vapor


def determine_state(
    pressure_mpa: float,
    temperature_k: float,
    quality: Optional[float] = None
) -> SteamState:
    """
    Determine thermodynamic state from P, T, and optional quality.
    """
    region = determine_region(pressure_mpa, temperature_k)

    if region == SteamRegion.REGION_1:
        return SteamState.SUBCOOLED_LIQUID
    elif region == SteamRegion.REGION_2:
        return SteamState.SUPERHEATED_VAPOR
    elif region == SteamRegion.REGION_3:
        return SteamState.SUPERCRITICAL
    elif region == SteamRegion.REGION_4:
        if quality is not None:
            if quality <= 0.001:
                return SteamState.SATURATED_LIQUID
            elif quality >= 0.999:
                return SteamState.SATURATED_VAPOR
            else:
                return SteamState.WET_STEAM
        return SteamState.WET_STEAM
    else:
        return SteamState.UNKNOWN


def get_saturation_properties(pressure_mpa: float) -> SaturationPoint:
    """
    Get saturation properties at given pressure.

    Uses simplified correlations based on IAPWS-IF97.
    """
    T_sat = calculate_saturation_temperature(pressure_mpa)

    # Simplified correlations (for testing purposes)
    # In production, use full IAPWS-IF97 equations

    # Liquid properties
    vf = 0.001 * (1 + 0.0001 * (T_sat - 273.15))  # Approximate
    hf = 4.186 * (T_sat - 273.15)  # Approximate
    sf = 4.186 * math.log(T_sat / 273.15)  # Approximate

    # Vapor properties (simplified)
    vg = R * T_sat / (pressure_mpa * 1000) * 0.95  # With compressibility factor
    hg = 2500 + 2.0 * (T_sat - 373.15)  # Approximate
    sg = 7.0 + 0.001 * (T_sat - 373.15)  # Approximate

    # Internal energy
    uf = hf - pressure_mpa * 1000 * vf
    ug = hg - pressure_mpa * 1000 * vg

    return SaturationPoint(
        pressure_mpa=pressure_mpa,
        temperature_k=T_sat,
        vf_m3_kg=vf,
        hf_kj_kg=hf,
        sf_kj_kg_k=sf,
        uf_kj_kg=uf,
        vg_m3_kg=vg,
        hg_kj_kg=hg,
        sg_kj_kg_k=sg,
        ug_kj_kg=ug,
        hfg_kj_kg=hg - hf,
        sfg_kj_kg_k=sg - sf,
    )


def calculate_wet_steam_properties(
    pressure_mpa: float,
    quality: float
) -> SteamProperties:
    """
    Calculate properties of wet steam at given pressure and quality.
    """
    if quality < 0 or quality > 1:
        raise ValueError(f"Quality must be between 0 and 1: {quality}")

    sat = get_saturation_properties(pressure_mpa)

    v = sat.vf_m3_kg + quality * (sat.vg_m3_kg - sat.vf_m3_kg)
    h = sat.hf_kj_kg + quality * sat.hfg_kj_kg
    s = sat.sf_kj_kg_k + quality * sat.sfg_kj_kg_k
    u = h - pressure_mpa * 1000 * v

    return SteamProperties(
        pressure_mpa=pressure_mpa,
        temperature_k=sat.temperature_k,
        specific_volume_m3_kg=v,
        specific_enthalpy_kj_kg=h,
        specific_entropy_kj_kg_k=s,
        specific_internal_energy_kj_kg=u,
        region=SteamRegion.REGION_4,
        state=SteamState.WET_STEAM if 0 < quality < 1 else (
            SteamState.SATURATED_LIQUID if quality == 0 else SteamState.SATURATED_VAPOR
        ),
        quality=quality,
    )


# =============================================================================
# Test Classes
# =============================================================================

class TestSaturationTemperature:
    """Tests for saturation temperature calculation."""

    @pytest.mark.parametrize("P,T_expected", SATURATION_T_VERIFICATION)
    def test_verification_points(self, P, T_expected):
        """Verify against IAPWS-IF97 Table 33."""
        T_calc = calculate_saturation_temperature(P)
        assert T_calc == pytest.approx(T_expected, rel=0.001)

    def test_triple_point(self):
        """Test saturation temperature at triple point."""
        T = calculate_saturation_temperature(TRIPLE_PRESSURE_MPA)
        assert T == pytest.approx(TRIPLE_TEMPERATURE_K, rel=0.01)

    def test_critical_point(self):
        """Test saturation temperature at critical point."""
        T = calculate_saturation_temperature(CRITICAL_PRESSURE_MPA - 0.001)
        assert T < CRITICAL_TEMPERATURE_K
        assert T > 640  # Close to critical

    def test_below_triple_point_raises(self):
        """Test that pressure below triple point raises error."""
        with pytest.raises(ValueError):
            calculate_saturation_temperature(TRIPLE_PRESSURE_MPA / 2)

    def test_above_critical_raises(self):
        """Test that pressure above critical raises error."""
        with pytest.raises(ValueError):
            calculate_saturation_temperature(CRITICAL_PRESSURE_MPA * 1.1)

    def test_monotonic_increase(self):
        """Test that saturation temperature increases with pressure."""
        pressures = [0.01, 0.1, 1.0, 5.0, 10.0, 20.0]
        temperatures = [calculate_saturation_temperature(P) for P in pressures]

        for i in range(1, len(temperatures)):
            assert temperatures[i] > temperatures[i-1]


class TestSaturationPressure:
    """Tests for saturation pressure calculation."""

    @pytest.mark.parametrize("T,P_expected", SATURATION_P_VERIFICATION)
    def test_verification_points(self, T, P_expected):
        """Verify against IAPWS-IF97 Table 35."""
        P_calc = calculate_saturation_pressure(T)
        assert P_calc == pytest.approx(P_expected, rel=0.001)

    def test_triple_point(self):
        """Test saturation pressure at triple point."""
        P = calculate_saturation_pressure(TRIPLE_TEMPERATURE_K)
        assert P == pytest.approx(TRIPLE_PRESSURE_MPA, rel=0.01)

    def test_below_triple_raises(self):
        """Test that temperature below triple point raises error."""
        with pytest.raises(ValueError):
            calculate_saturation_pressure(TRIPLE_TEMPERATURE_K - 10)

    def test_above_critical_raises(self):
        """Test that temperature above critical raises error."""
        with pytest.raises(ValueError):
            calculate_saturation_pressure(CRITICAL_TEMPERATURE_K + 10)

    def test_monotonic_increase(self):
        """Test that saturation pressure increases with temperature."""
        temperatures = [300, 350, 400, 450, 500, 550, 600]
        pressures = [calculate_saturation_pressure(T) for T in temperatures]

        for i in range(1, len(pressures)):
            assert pressures[i] > pressures[i-1]

    def test_inverse_consistency(self):
        """Test that P(T(P)) = P."""
        for P in [0.1, 1.0, 5.0, 10.0]:
            T = calculate_saturation_temperature(P)
            P_back = calculate_saturation_pressure(T)
            assert P_back == pytest.approx(P, rel=0.001)


class TestRegionDetermination:
    """Tests for region determination."""

    def test_subcooled_liquid_region1(self):
        """Test subcooled liquid is Region 1."""
        region = determine_region(1.0, 350.0)  # Below Tsat at 1 MPa
        assert region == SteamRegion.REGION_1

    def test_superheated_vapor_region2(self):
        """Test superheated vapor is Region 2."""
        region = determine_region(1.0, 500.0)  # Above Tsat at 1 MPa
        assert region == SteamRegion.REGION_2

    def test_saturation_region4(self):
        """Test saturation line is Region 4."""
        T_sat = calculate_saturation_temperature(1.0)
        region = determine_region(1.0, T_sat)
        assert region == SteamRegion.REGION_4

    def test_supercritical_region3(self):
        """Test supercritical is Region 3."""
        region = determine_region(25.0, 700.0)
        assert region == SteamRegion.REGION_3

    def test_high_temperature_region5(self):
        """Test high temperature is Region 5."""
        region = determine_region(1.0, 1100.0)
        assert region == SteamRegion.REGION_5

    def test_invalid_conditions(self):
        """Test invalid conditions return UNKNOWN."""
        assert determine_region(-1.0, 400.0) == SteamRegion.UNKNOWN
        assert determine_region(1.0, -100.0) == SteamRegion.UNKNOWN


class TestStateDetermination:
    """Tests for state determination."""

    def test_subcooled_liquid(self):
        """Test subcooled liquid state."""
        state = determine_state(1.0, 350.0)
        assert state == SteamState.SUBCOOLED_LIQUID

    def test_superheated_vapor(self):
        """Test superheated vapor state."""
        state = determine_state(1.0, 500.0)
        assert state == SteamState.SUPERHEATED_VAPOR

    def test_saturated_liquid(self):
        """Test saturated liquid state (x=0)."""
        T_sat = calculate_saturation_temperature(1.0)
        state = determine_state(1.0, T_sat, quality=0.0)
        assert state == SteamState.SATURATED_LIQUID

    def test_saturated_vapor(self):
        """Test saturated vapor state (x=1)."""
        T_sat = calculate_saturation_temperature(1.0)
        state = determine_state(1.0, T_sat, quality=1.0)
        assert state == SteamState.SATURATED_VAPOR

    def test_wet_steam(self):
        """Test wet steam state (0 < x < 1)."""
        T_sat = calculate_saturation_temperature(1.0)
        state = determine_state(1.0, T_sat, quality=0.5)
        assert state == SteamState.WET_STEAM

    def test_supercritical(self):
        """Test supercritical state."""
        state = determine_state(25.0, 700.0)
        assert state == SteamState.SUPERCRITICAL


class TestSaturationProperties:
    """Tests for saturation property calculation."""

    def test_liquid_vapor_ordering(self):
        """Test that vapor properties > liquid properties."""
        sat = get_saturation_properties(1.0)

        assert sat.vg_m3_kg > sat.vf_m3_kg  # Vapor volume > liquid
        assert sat.hg_kj_kg > sat.hf_kj_kg  # Vapor enthalpy > liquid
        assert sat.sg_kj_kg_k > sat.sf_kj_kg_k  # Vapor entropy > liquid

    def test_latent_heat_positive(self):
        """Test that latent heat is positive."""
        sat = get_saturation_properties(1.0)

        assert sat.hfg_kj_kg > 0
        assert sat.hfg_kj_kg == sat.hg_kj_kg - sat.hf_kj_kg

    def test_latent_heat_decreases_with_pressure(self):
        """Test that latent heat decreases with increasing pressure."""
        sat_low = get_saturation_properties(0.1)
        sat_high = get_saturation_properties(10.0)

        assert sat_high.hfg_kj_kg < sat_low.hfg_kj_kg

    def test_properties_at_various_pressures(self):
        """Test properties at various pressures are reasonable."""
        for P in [0.1, 1.0, 5.0, 10.0]:
            sat = get_saturation_properties(P)

            # All properties should be positive
            assert sat.vf_m3_kg > 0
            assert sat.vg_m3_kg > 0
            assert sat.hf_kj_kg > 0
            assert sat.hg_kj_kg > 0
            assert sat.sf_kj_kg_k > 0
            assert sat.sg_kj_kg_k > 0


class TestWetSteamProperties:
    """Tests for wet steam property calculation."""

    def test_quality_zero_matches_liquid(self):
        """Test that quality=0 gives liquid properties."""
        sat = get_saturation_properties(1.0)
        wet = calculate_wet_steam_properties(1.0, 0.0)

        assert wet.specific_volume_m3_kg == pytest.approx(sat.vf_m3_kg)
        assert wet.specific_enthalpy_kj_kg == pytest.approx(sat.hf_kj_kg)
        assert wet.specific_entropy_kj_kg_k == pytest.approx(sat.sf_kj_kg_k)

    def test_quality_one_matches_vapor(self):
        """Test that quality=1 gives vapor properties."""
        sat = get_saturation_properties(1.0)
        wet = calculate_wet_steam_properties(1.0, 1.0)

        assert wet.specific_volume_m3_kg == pytest.approx(sat.vg_m3_kg)
        assert wet.specific_enthalpy_kj_kg == pytest.approx(sat.hg_kj_kg)
        assert wet.specific_entropy_kj_kg_k == pytest.approx(sat.sg_kj_kg_k)

    def test_quality_half_interpolates(self):
        """Test that quality=0.5 interpolates properties."""
        sat = get_saturation_properties(1.0)
        wet = calculate_wet_steam_properties(1.0, 0.5)

        expected_h = sat.hf_kj_kg + 0.5 * sat.hfg_kj_kg
        assert wet.specific_enthalpy_kj_kg == pytest.approx(expected_h)

    def test_invalid_quality_raises(self):
        """Test that invalid quality raises error."""
        with pytest.raises(ValueError):
            calculate_wet_steam_properties(1.0, -0.1)

        with pytest.raises(ValueError):
            calculate_wet_steam_properties(1.0, 1.1)

    def test_properties_linear_with_quality(self):
        """Test that properties vary linearly with quality."""
        sat = get_saturation_properties(1.0)
        qualities = [0.0, 0.25, 0.5, 0.75, 1.0]
        enthalpies = [calculate_wet_steam_properties(1.0, x).specific_enthalpy_kj_kg for x in qualities]

        # Check linearity
        for i in range(1, len(enthalpies)):
            expected_diff = sat.hfg_kj_kg * (qualities[i] - qualities[i-1])
            actual_diff = enthalpies[i] - enthalpies[i-1]
            assert actual_diff == pytest.approx(expected_diff, rel=0.01)


class TestThermodynamicConsistency:
    """Tests for thermodynamic consistency of properties."""

    def test_internal_energy_relation(self):
        """Test that u = h - Pv holds."""
        wet = calculate_wet_steam_properties(1.0, 0.5)

        u_calc = wet.specific_enthalpy_kj_kg - wet.pressure_mpa * 1000 * wet.specific_volume_m3_kg
        assert wet.specific_internal_energy_kj_kg == pytest.approx(u_calc, rel=0.01)

    def test_entropy_increases_with_quality(self):
        """Test that entropy increases with quality."""
        entropies = [calculate_wet_steam_properties(1.0, x).specific_entropy_kj_kg_k
                     for x in [0.0, 0.25, 0.5, 0.75, 1.0]]

        for i in range(1, len(entropies)):
            assert entropies[i] > entropies[i-1]

    def test_volume_increases_with_quality(self):
        """Test that specific volume increases with quality."""
        volumes = [calculate_wet_steam_properties(1.0, x).specific_volume_m3_kg
                   for x in [0.0, 0.25, 0.5, 0.75, 1.0]]

        for i in range(1, len(volumes)):
            assert volumes[i] > volumes[i-1]


class TestPhysicalReasonableness:
    """Tests for physical reasonableness of property values."""

    def test_liquid_specific_volume_reasonable(self):
        """Test liquid specific volume is about 0.001 m3/kg."""
        sat = get_saturation_properties(1.0)

        assert 0.0005 < sat.vf_m3_kg < 0.002  # Close to liquid water

    def test_vapor_specific_volume_reasonable(self):
        """Test vapor specific volume is reasonable."""
        sat = get_saturation_properties(1.0)

        # At 1 MPa, vapor should have much larger volume than liquid
        assert sat.vg_m3_kg > 100 * sat.vf_m3_kg

    def test_latent_heat_reasonable(self):
        """Test latent heat is in reasonable range."""
        sat = get_saturation_properties(1.0)

        # Latent heat at atmospheric pressure is about 2260 kJ/kg
        # At higher pressure, it should be less
        assert 1000 < sat.hfg_kj_kg < 2500

    def test_entropy_reasonable(self):
        """Test entropy values are reasonable."""
        sat = get_saturation_properties(1.0)

        # Entropy should be positive and in reasonable range
        assert 0 < sat.sf_kj_kg_k < 5  # Liquid entropy
        assert 5 < sat.sg_kj_kg_k < 10  # Vapor entropy


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_saturation_temperature_deterministic(self):
        """Test that saturation temperature is deterministic."""
        results = [calculate_saturation_temperature(1.0) for _ in range(10)]

        for r in results[1:]:
            assert r == results[0]

    def test_saturation_pressure_deterministic(self):
        """Test that saturation pressure is deterministic."""
        results = [calculate_saturation_pressure(400.0) for _ in range(10)]

        for r in results[1:]:
            assert r == results[0]

    def test_wet_steam_properties_deterministic(self):
        """Test that wet steam properties are deterministic."""
        results = [calculate_wet_steam_properties(1.0, 0.5) for _ in range(10)]

        first = results[0]
        for r in results[1:]:
            assert r.specific_enthalpy_kj_kg == first.specific_enthalpy_kj_kg
            assert r.specific_entropy_kj_kg_k == first.specific_entropy_kj_kg_k


class TestEdgeCases:
    """Tests for edge cases."""

    def test_near_triple_point(self):
        """Test calculations near triple point."""
        T = calculate_saturation_temperature(TRIPLE_PRESSURE_MPA * 1.1)
        assert T > TRIPLE_TEMPERATURE_K

    def test_near_critical_point(self):
        """Test calculations near critical point."""
        T = calculate_saturation_temperature(CRITICAL_PRESSURE_MPA * 0.99)
        assert T < CRITICAL_TEMPERATURE_K

    def test_very_low_pressure(self):
        """Test at very low (but valid) pressure."""
        sat = get_saturation_properties(0.001)

        assert sat.temperature_k > TRIPLE_TEMPERATURE_K
        assert sat.vg_m3_kg > 100  # Very large vapor volume at low pressure


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
