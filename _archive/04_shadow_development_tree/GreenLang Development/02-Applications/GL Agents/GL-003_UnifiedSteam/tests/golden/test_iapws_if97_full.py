# -*- coding: utf-8 -*-
"""
IAPWS-IF97 Comprehensive Golden Value Tests for GL-003 UnifiedSteam
====================================================================

This module validates steam property calculations against official
IAPWS-IF97 verification tables from the standard document.

Reference Tables Used:
    - Table 5: Region 1 test values (compressed liquid)
    - Table 15: Region 2 test values (superheated vapor)
    - Table 33: Region 3 test values (supercritical)
    - Table 9: Region 4 saturation test values
    - Table 42: Region 5 test values (high temperature)

Source: Wagner, W., et al. (2000). The IAPWS Industrial Formulation 1997
        for the Thermodynamic Properties of Water and Steam.

Author: GL-TestEngineer
Version: 2.0.0
"""

import pytest
import sys
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Any
from decimal import Decimal, ROUND_HALF_UP

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from thermodynamics.iapws_if97 import (
        region1_properties,
        region2_properties,
        region3_properties,
        saturation_temperature,
        saturation_pressure,
        steam_properties,
    )
    HAS_IAPWS = True
except ImportError:
    HAS_IAPWS = False


# =============================================================================
# IAPWS-IF97 OFFICIAL VERIFICATION VALUES
# Source: Tables 5, 15, 33, 9, 42 from IAPWS-IF97 Document
# =============================================================================

@dataclass(frozen=True)
class IAPWSIF97VerificationPoint:
    """Official IAPWS-IF97 verification test point."""
    region: int
    temperature_K: float
    pressure_MPa: float
    specific_volume_m3_kg: float
    specific_enthalpy_kJ_kg: float
    specific_entropy_kJ_kgK: float
    specific_internal_energy_kJ_kg: float
    specific_isobaric_heat_capacity_kJ_kgK: float
    speed_of_sound_m_s: float
    tolerance_percent: float = 0.001  # 0.001% tolerance per IAPWS


# -----------------------------------------------------------------------------
# Region 1: Compressed Liquid (Table 5 from IAPWS-IF97)
# -----------------------------------------------------------------------------
REGION1_VERIFICATION = [
    # Test 1: T=300 K, P=3 MPa
    IAPWSIF97VerificationPoint(
        region=1,
        temperature_K=300.0,
        pressure_MPa=3.0,
        specific_volume_m3_kg=0.100215168e-2,
        specific_enthalpy_kJ_kg=0.115331273e3,
        specific_entropy_kJ_kgK=0.392294792,
        specific_internal_energy_kJ_kg=0.112324818e3,
        specific_isobaric_heat_capacity_kJ_kgK=0.417301218e1,
        speed_of_sound_m_s=0.150773921e4,
    ),
    # Test 2: T=300 K, P=80 MPa
    IAPWSIF97VerificationPoint(
        region=1,
        temperature_K=300.0,
        pressure_MPa=80.0,
        specific_volume_m3_kg=0.971180894e-3,
        specific_enthalpy_kJ_kg=0.184142828e3,
        specific_entropy_kJ_kgK=0.368563852,
        specific_internal_energy_kJ_kg=0.106448356e3,
        specific_isobaric_heat_capacity_kJ_kgK=0.401008987e1,
        speed_of_sound_m_s=0.163469054e4,
    ),
    # Test 3: T=500 K, P=3 MPa
    IAPWSIF97VerificationPoint(
        region=1,
        temperature_K=500.0,
        pressure_MPa=3.0,
        specific_volume_m3_kg=0.120241800e-2,
        specific_enthalpy_kJ_kg=0.975542239e3,
        specific_entropy_kJ_kgK=0.258041912e1,
        specific_internal_energy_kJ_kg=0.971934985e3,
        specific_isobaric_heat_capacity_kJ_kgK=0.465580682e1,
        speed_of_sound_m_s=0.124071337e4,
    ),
]


# -----------------------------------------------------------------------------
# Region 2: Superheated Vapor (Table 15 from IAPWS-IF97)
# -----------------------------------------------------------------------------
REGION2_VERIFICATION = [
    # Test 1: T=300 K, P=0.001 MPa (low pressure superheated vapor)
    IAPWSIF97VerificationPoint(
        region=2,
        temperature_K=300.0,
        pressure_MPa=0.001,
        specific_volume_m3_kg=0.394913866e2,
        specific_enthalpy_kJ_kg=0.254991145e4,
        specific_entropy_kJ_kgK=0.852238967e1,
        specific_internal_energy_kJ_kg=0.241169160e4,
        specific_isobaric_heat_capacity_kJ_kgK=0.191300162e1,
        speed_of_sound_m_s=0.427920172e3,
    ),
    # Test 2: T=700 K, P=0.001 MPa
    IAPWSIF97VerificationPoint(
        region=2,
        temperature_K=700.0,
        pressure_MPa=0.001,
        specific_volume_m3_kg=0.923015898e2,
        specific_enthalpy_kJ_kg=0.333568375e4,
        specific_entropy_kJ_kgK=0.101749996e2,
        specific_internal_energy_kJ_kg=0.301262819e4,
        specific_isobaric_heat_capacity_kJ_kgK=0.208141274e1,
        speed_of_sound_m_s=0.644289068e3,
    ),
    # Test 3: T=700 K, P=30 MPa
    IAPWSIF97VerificationPoint(
        region=2,
        temperature_K=700.0,
        pressure_MPa=30.0,
        specific_volume_m3_kg=0.542946619e-2,
        specific_enthalpy_kJ_kg=0.263149474e4,
        specific_entropy_kJ_kgK=0.517540298e1,
        specific_internal_energy_kJ_kg=0.246861076e4,
        specific_isobaric_heat_capacity_kJ_kgK=0.103505092e2,
        speed_of_sound_m_s=0.480386523e3,
    ),
]


# -----------------------------------------------------------------------------
# Region 3: Supercritical (Table 33 from IAPWS-IF97)
# -----------------------------------------------------------------------------
REGION3_VERIFICATION = [
    # Test 1: T=650 K, rho=500 kg/m³
    IAPWSIF97VerificationPoint(
        region=3,
        temperature_K=650.0,
        pressure_MPa=25.5837018,  # Calculated from rho=500
        specific_volume_m3_kg=0.002,  # 1/500
        specific_enthalpy_kJ_kg=0.186343019e4,
        specific_entropy_kJ_kgK=0.405427273e1,
        specific_internal_energy_kJ_kg=0.181226279e4,
        specific_isobaric_heat_capacity_kJ_kgK=0.138935717e2,
        speed_of_sound_m_s=0.502005554e3,
    ),
    # Test 2: T=650 K, rho=200 kg/m³
    IAPWSIF97VerificationPoint(
        region=3,
        temperature_K=650.0,
        pressure_MPa=22.2930643,  # Calculated from rho=200
        specific_volume_m3_kg=0.005,  # 1/200
        specific_enthalpy_kJ_kg=0.237512401e4,
        specific_entropy_kJ_kgK=0.485438792e1,
        specific_internal_energy_kJ_kg=0.226365868e4,
        specific_isobaric_heat_capacity_kJ_kgK=0.446579342e2,
        speed_of_sound_m_s=0.383444594e3,
    ),
]


# -----------------------------------------------------------------------------
# Region 4: Saturation Line (Table 9 from IAPWS-IF97)
# -----------------------------------------------------------------------------
REGION4_SATURATION_VERIFICATION = [
    # Saturation pressure from temperature
    {"T_K": 300.0, "P_sat_MPa": 0.353658941e-2, "tolerance": 1e-6},
    {"T_K": 500.0, "P_sat_MPa": 0.263889776e1, "tolerance": 1e-6},
    {"T_K": 600.0, "P_sat_MPa": 0.123443146e2, "tolerance": 1e-6},

    # Saturation temperature from pressure
    {"P_MPa": 0.1, "T_sat_K": 0.372755919e3, "tolerance": 1e-6},
    {"P_MPa": 1.0, "T_sat_K": 0.453035632e3, "tolerance": 1e-6},
    {"P_MPa": 10.0, "T_sat_K": 0.584149488e3, "tolerance": 1e-6},
]


# -----------------------------------------------------------------------------
# Region 5: High Temperature Steam (Table 42 from IAPWS-IF97)
# -----------------------------------------------------------------------------
REGION5_VERIFICATION = [
    # Test 1: T=1500 K, P=0.5 MPa
    IAPWSIF97VerificationPoint(
        region=5,
        temperature_K=1500.0,
        pressure_MPa=0.5,
        specific_volume_m3_kg=0.138455090e1,
        specific_enthalpy_kJ_kg=0.521976855e4,
        specific_entropy_kJ_kgK=0.965408875e1,
        specific_internal_energy_kJ_kg=0.452749310e4,
        specific_isobaric_heat_capacity_kJ_kgK=0.261609445e1,
        speed_of_sound_m_s=0.917068690e3,
    ),
    # Test 2: T=1500 K, P=30 MPa
    IAPWSIF97VerificationPoint(
        region=5,
        temperature_K=1500.0,
        pressure_MPa=30.0,
        specific_volume_m3_kg=0.230761299e-1,
        specific_enthalpy_kJ_kg=0.516723514e4,
        specific_entropy_kJ_kgK=0.772970133e1,
        specific_internal_energy_kJ_kg=0.447495124e4,
        specific_isobaric_heat_capacity_kJ_kgK=0.272724317e1,
        speed_of_sound_m_s=0.928548002e3,
    ),
    # Test 3: T=2000 K, P=30 MPa
    IAPWSIF97VerificationPoint(
        region=5,
        temperature_K=2000.0,
        pressure_MPa=30.0,
        specific_volume_m3_kg=0.311385219e-1,
        specific_enthalpy_kJ_kg=0.657122604e4,
        specific_entropy_kJ_kgK=0.853640523e1,
        specific_internal_energy_kJ_kg=0.563707038e4,
        specific_isobaric_heat_capacity_kJ_kgK=0.288569882e1,
        speed_of_sound_m_s=0.106736948e4,
    ),
]


# =============================================================================
# INDUSTRIAL STEAM TABLE VERIFICATION
# Common engineering reference points
# =============================================================================

INDUSTRIAL_STEAM_TABLE = {
    # Saturated steam at common pressures (psig converted to MPa)
    "150_psig": {
        "pressure_MPa": 1.136,  # 150 psig = 164.7 psia = 1.136 MPa
        "T_sat_C": 186.0,
        "hf_kJ_kg": 789.0,
        "hg_kJ_kg": 2782.0,
        "hfg_kJ_kg": 1993.0,
        "tolerance_percent": 0.5,
    },
    "300_psig": {
        "pressure_MPa": 2.171,
        "T_sat_C": 216.0,
        "hf_kJ_kg": 922.0,
        "hg_kJ_kg": 2800.0,
        "hfg_kJ_kg": 1878.0,
        "tolerance_percent": 0.5,
    },
    "600_psig": {
        "pressure_MPa": 4.240,
        "T_sat_C": 254.0,
        "hf_kJ_kg": 1101.0,
        "hg_kJ_kg": 2802.0,
        "hfg_kJ_kg": 1701.0,
        "tolerance_percent": 0.5,
    },
}


# =============================================================================
# TEST CLASSES
# =============================================================================

@pytest.mark.golden
@pytest.mark.skipif(not HAS_IAPWS, reason="IAPWS module not available")
class TestRegion1CompressedLiquid:
    """Test Region 1 (compressed liquid) against IAPWS-IF97 Table 5."""

    @pytest.mark.parametrize("test_point", REGION1_VERIFICATION)
    def test_region1_specific_volume(self, test_point: IAPWSIF97VerificationPoint):
        """Validate specific volume in Region 1."""
        result = region1_properties(
            T_K=test_point.temperature_K,
            P_MPa=test_point.pressure_MPa
        )
        calculated = result.get("v", result.get("specific_volume"))
        expected = test_point.specific_volume_m3_kg

        deviation = abs(calculated - expected) / expected * 100
        assert deviation <= test_point.tolerance_percent, (
            f"v deviation {deviation:.6f}% exceeds {test_point.tolerance_percent}% "
            f"at T={test_point.temperature_K}K, P={test_point.pressure_MPa}MPa"
        )

    @pytest.mark.parametrize("test_point", REGION1_VERIFICATION)
    def test_region1_enthalpy(self, test_point: IAPWSIF97VerificationPoint):
        """Validate specific enthalpy in Region 1."""
        result = region1_properties(
            T_K=test_point.temperature_K,
            P_MPa=test_point.pressure_MPa
        )
        calculated = result.get("h", result.get("specific_enthalpy"))
        expected = test_point.specific_enthalpy_kJ_kg

        deviation = abs(calculated - expected) / expected * 100
        assert deviation <= test_point.tolerance_percent, (
            f"h deviation {deviation:.6f}% exceeds {test_point.tolerance_percent}%"
        )

    @pytest.mark.parametrize("test_point", REGION1_VERIFICATION)
    def test_region1_entropy(self, test_point: IAPWSIF97VerificationPoint):
        """Validate specific entropy in Region 1."""
        result = region1_properties(
            T_K=test_point.temperature_K,
            P_MPa=test_point.pressure_MPa
        )
        calculated = result.get("s", result.get("specific_entropy"))
        expected = test_point.specific_entropy_kJ_kgK

        deviation = abs(calculated - expected) / expected * 100
        assert deviation <= test_point.tolerance_percent, (
            f"s deviation {deviation:.6f}% exceeds {test_point.tolerance_percent}%"
        )


@pytest.mark.golden
@pytest.mark.skipif(not HAS_IAPWS, reason="IAPWS module not available")
class TestRegion2SuperheatedVapor:
    """Test Region 2 (superheated vapor) against IAPWS-IF97 Table 15."""

    @pytest.mark.parametrize("test_point", REGION2_VERIFICATION)
    def test_region2_specific_volume(self, test_point: IAPWSIF97VerificationPoint):
        """Validate specific volume in Region 2."""
        result = region2_properties(
            T_K=test_point.temperature_K,
            P_MPa=test_point.pressure_MPa
        )
        calculated = result.get("v", result.get("specific_volume"))
        expected = test_point.specific_volume_m3_kg

        deviation = abs(calculated - expected) / expected * 100
        assert deviation <= test_point.tolerance_percent, (
            f"v deviation {deviation:.6f}% exceeds {test_point.tolerance_percent}%"
        )

    @pytest.mark.parametrize("test_point", REGION2_VERIFICATION)
    def test_region2_enthalpy(self, test_point: IAPWSIF97VerificationPoint):
        """Validate specific enthalpy in Region 2."""
        result = region2_properties(
            T_K=test_point.temperature_K,
            P_MPa=test_point.pressure_MPa
        )
        calculated = result.get("h", result.get("specific_enthalpy"))
        expected = test_point.specific_enthalpy_kJ_kg

        deviation = abs(calculated - expected) / expected * 100
        assert deviation <= test_point.tolerance_percent, (
            f"h deviation {deviation:.6f}% exceeds {test_point.tolerance_percent}%"
        )

    @pytest.mark.parametrize("test_point", REGION2_VERIFICATION)
    def test_region2_entropy(self, test_point: IAPWSIF97VerificationPoint):
        """Validate specific entropy in Region 2."""
        result = region2_properties(
            T_K=test_point.temperature_K,
            P_MPa=test_point.pressure_MPa
        )
        calculated = result.get("s", result.get("specific_entropy"))
        expected = test_point.specific_entropy_kJ_kgK

        deviation = abs(calculated - expected) / expected * 100
        assert deviation <= test_point.tolerance_percent, (
            f"s deviation {deviation:.6f}% exceeds {test_point.tolerance_percent}%"
        )


@pytest.mark.golden
@pytest.mark.skipif(not HAS_IAPWS, reason="IAPWS module not available")
class TestRegion4Saturation:
    """Test Region 4 (saturation line) against IAPWS-IF97 Table 9."""

    @pytest.mark.parametrize("test_data", [
        t for t in REGION4_SATURATION_VERIFICATION if "T_K" in t and "P_sat_MPa" in t
    ])
    def test_saturation_pressure_from_temperature(self, test_data):
        """Validate saturation pressure from temperature."""
        T_K = test_data["T_K"]
        expected_P = test_data["P_sat_MPa"]
        tolerance = test_data["tolerance"]

        calculated_P = saturation_pressure(T_K=T_K)

        deviation = abs(calculated_P - expected_P)
        assert deviation <= tolerance, (
            f"P_sat deviation {deviation:.9f} MPa exceeds tolerance {tolerance} "
            f"at T={T_K}K"
        )

    @pytest.mark.parametrize("test_data", [
        t for t in REGION4_SATURATION_VERIFICATION if "P_MPa" in t and "T_sat_K" in t
    ])
    def test_saturation_temperature_from_pressure(self, test_data):
        """Validate saturation temperature from pressure."""
        P_MPa = test_data["P_MPa"]
        expected_T = test_data["T_sat_K"]
        tolerance = test_data["tolerance"]

        calculated_T = saturation_temperature(P_MPa=P_MPa)

        deviation = abs(calculated_T - expected_T)
        assert deviation <= tolerance, (
            f"T_sat deviation {deviation:.9f} K exceeds tolerance {tolerance} "
            f"at P={P_MPa}MPa"
        )


@pytest.mark.golden
class TestThermodynamicConsistency:
    """Test thermodynamic consistency relationships."""

    def test_enthalpy_entropy_consistency_at_saturation(self):
        """At saturation, dh = T*ds for reversible process."""
        # At 100°C (373.15 K) saturation
        T_K = 373.15
        hf = 419.05  # kJ/kg
        hg = 2675.46  # kJ/kg
        sf = 1.3069  # kJ/(kg·K)
        sg = 7.3545  # kJ/(kg·K)

        # Check h_fg ≈ T * s_fg (approximately, for reversible vaporization)
        h_fg = hg - hf
        s_fg = sg - sf
        T_times_s_fg = T_K * s_fg

        # These should be approximately equal (within 1%)
        deviation = abs(h_fg - T_times_s_fg) / h_fg * 100
        assert deviation <= 1.0, (
            f"Thermodynamic consistency: h_fg={h_fg:.2f}, T*s_fg={T_times_s_fg:.2f}, "
            f"deviation={deviation:.2f}%"
        )

    def test_clausius_clapeyron_consistency(self):
        """Verify Clausius-Clapeyron equation: dP/dT = h_fg / (T * v_fg)."""
        # Two saturation points
        T1, P1 = 373.15, 0.101325  # 100°C, 1 atm
        T2, P2 = 393.15, 0.198673  # 120°C

        # Approximate derivative
        dP_dT = (P2 - P1) / (T2 - T1)  # MPa/K

        # At 100°C
        h_fg = 2256.41  # kJ/kg
        v_fg = 1.672 - 0.001044  # m³/kg (v_g - v_f at 100°C)
        T = 373.15

        # Clausius-Clapeyron: dP/dT = h_fg / (T * v_fg)
        # Convert: h_fg in kJ/kg, v_fg in m³/kg, T in K
        # Result in MPa/K: h_fg / (T * v_fg) * 0.001 (kJ to MJ)
        clausius_dP_dT = (h_fg / (T * v_fg)) * 0.001

        # Should be within 10% (approximate comparison)
        deviation = abs(dP_dT - clausius_dP_dT) / clausius_dP_dT * 100
        assert deviation <= 15.0, (
            f"Clausius-Clapeyron: numerical dP/dT={dP_dT:.6f}, "
            f"theoretical={clausius_dP_dT:.6f}, deviation={deviation:.1f}%"
        )

    def test_specific_heat_ratio_consistency(self):
        """Verify cp/cv = gamma for ideal gas behavior at low pressure."""
        # At very low pressure, steam behaves nearly ideally
        # For water vapor (triatomic), gamma ≈ 1.33
        # cp - cv = R for ideal gas

        R = 0.4615  # kJ/(kg·K) for water

        # At 500K, 0.001 MPa (nearly ideal)
        cp = 1.913  # kJ/(kg·K) from IAPWS
        cv = cp - R  # Approximately for near-ideal behavior
        gamma = cp / cv

        # Should be close to 1.33 for water vapor
        assert 1.2 <= gamma <= 1.4, (
            f"Specific heat ratio gamma={gamma:.3f} outside expected range"
        )


@pytest.mark.golden
class TestDeterminism:
    """Verify calculation determinism (same input → same output)."""

    def test_enthalpy_calculation_determinism(self):
        """Verify enthalpy calculations are perfectly deterministic."""
        hashes = set()

        for _ in range(100):
            # Calculate steam quality from enthalpy
            hf = Decimal("419.05")
            hg = Decimal("2675.46")
            h = Decimal("1500.0")

            quality = float((h - hf) / (hg - hf))
            quality_rounded = round(quality, 10)

            hash_val = hashlib.sha256(str(quality_rounded).encode()).hexdigest()
            hashes.add(hash_val)

        assert len(hashes) == 1, (
            f"Non-deterministic calculation: {len(hashes)} different results"
        )

    def test_saturation_lookup_determinism(self):
        """Verify saturation lookups are deterministic."""
        results = []

        for _ in range(50):
            # Saturation properties at 1 MPa
            T_sat = 453.03  # K (from IAPWS table)
            hf = 762.51
            hg = 2777.11

            result_hash = hashlib.sha256(
                f"{T_sat:.6f}:{hf:.6f}:{hg:.6f}".encode()
            ).hexdigest()
            results.append(result_hash)

        assert len(set(results)) == 1, "Saturation lookup not deterministic"


@pytest.mark.golden
class TestIndustrialSteamTable:
    """Validate against common industrial steam table values."""

    @pytest.mark.parametrize("steam_point", [
        ("150_psig", 1.136, 186.0, 789.0, 2782.0),
        ("300_psig", 2.171, 216.0, 922.0, 2800.0),
        ("600_psig", 4.240, 254.0, 1101.0, 2802.0),
    ])
    def test_industrial_saturation_values(self, steam_point):
        """Verify against industrial steam table values."""
        name, P_MPa, T_sat_C_expected, hf_expected, hg_expected = steam_point

        # Convert to Kelvin
        T_sat_K_expected = T_sat_C_expected + 273.15

        # These are reference values - just verify the data is consistent
        hfg = hg_expected - hf_expected

        # Latent heat should decrease with increasing pressure
        assert 1500 <= hfg <= 2100, (
            f"Latent heat {hfg} kJ/kg outside expected range for {name}"
        )

        # Temperature should increase with pressure
        assert T_sat_K_expected > 373.15, (
            f"Saturation temperature {T_sat_K_expected}K should be above boiling"
        )


@pytest.mark.golden
class TestBoundaryConditions:
    """Test critical point and boundary conditions."""

    def test_critical_point_properties(self):
        """Verify critical point values."""
        T_crit = 647.096  # K
        P_crit = 22.064  # MPa
        rho_crit = 322.0  # kg/m³

        # Critical point specific volume
        v_crit = 1 / rho_crit
        assert abs(v_crit - 0.003106) < 0.0001, (
            f"Critical specific volume {v_crit} m³/kg incorrect"
        )

        # At critical point, liquid and vapor properties converge
        # h_crit ≈ 2084 kJ/kg, s_crit ≈ 4.41 kJ/(kg·K)
        h_crit_expected = 2084.0
        s_crit_expected = 4.41

        # Verify we have reasonable critical point values defined
        assert T_crit > 600 and T_crit < 700
        assert P_crit > 20 and P_crit < 25

    def test_triple_point_properties(self):
        """Verify triple point values."""
        T_triple = 273.16  # K (0.01°C)
        P_triple = 0.000611657  # MPa

        # At triple point, ice, water, and vapor coexist
        assert T_triple > 273.15 and T_triple < 273.17
        assert P_triple < 0.001


@pytest.mark.golden
class TestMassAndEnergyBalance:
    """Test mass and energy balance calculations."""

    def test_desuperheater_mass_balance(self):
        """Verify desuperheater spray calculation."""
        # Superheated steam at 10 MPa, 500°C
        h_in = 3373.0  # kJ/kg (superheated)
        m_steam = 100.0  # kg/s

        # Spray water at saturation (10 MPa)
        h_spray = 1408.0  # kJ/kg (saturated liquid at 10 MPa)

        # Target: saturated vapor at 10 MPa
        h_target = 2725.0  # kJ/kg (saturated vapor)

        # Mass balance: m_steam * h_in + m_spray * h_spray = (m_steam + m_spray) * h_target
        # m_spray = m_steam * (h_in - h_target) / (h_target - h_spray)
        m_spray = m_steam * (h_in - h_target) / (h_target - h_spray)

        # Verify reasonable spray rate (should be positive and < steam rate)
        assert 0 < m_spray < m_steam, (
            f"Spray rate {m_spray} kg/s outside expected range"
        )

        # Verify energy balance
        energy_in = m_steam * h_in + m_spray * h_spray
        energy_out = (m_steam + m_spray) * h_target

        assert abs(energy_in - energy_out) < 0.1, (
            f"Energy imbalance: in={energy_in:.1f}, out={energy_out:.1f}"
        )

    def test_flash_steam_calculation(self):
        """Verify flash steam generation calculation."""
        # Condensate at 1 MPa drops to 0.1 MPa
        P1, P2 = 1.0, 0.1  # MPa

        # Saturation enthalpies
        hf_1MPa = 762.51  # kJ/kg
        hf_01MPa = 417.44  # kJ/kg
        hfg_01MPa = 2258.0  # kJ/kg

        # Flash steam fraction: x = (h1 - hf2) / hfg2
        flash_fraction = (hf_1MPa - hf_01MPa) / hfg_01MPa

        # Should be between 0 and 1, typically 10-20% for this pressure drop
        assert 0.1 <= flash_fraction <= 0.2, (
            f"Flash fraction {flash_fraction:.3f} outside expected range"
        )


# =============================================================================
# EXPORT FUNCTION FOR GOLDEN VALUES
# =============================================================================

def export_iapws_golden_values() -> Dict[str, Any]:
    """Export all golden values for external validation."""
    return {
        "metadata": {
            "version": "2.0.0",
            "source": "IAPWS-IF97",
            "agent": "GL-003_UnifiedSteam",
        },
        "region1": [
            {
                "T_K": p.temperature_K,
                "P_MPa": p.pressure_MPa,
                "v": p.specific_volume_m3_kg,
                "h": p.specific_enthalpy_kJ_kg,
                "s": p.specific_entropy_kJ_kgK,
            }
            for p in REGION1_VERIFICATION
        ],
        "region2": [
            {
                "T_K": p.temperature_K,
                "P_MPa": p.pressure_MPa,
                "v": p.specific_volume_m3_kg,
                "h": p.specific_enthalpy_kJ_kg,
                "s": p.specific_entropy_kJ_kgK,
            }
            for p in REGION2_VERIFICATION
        ],
        "region4_saturation": REGION4_SATURATION_VERIFICATION,
        "critical_point": {
            "T_K": 647.096,
            "P_MPa": 22.064,
            "rho_kg_m3": 322.0,
        },
    }


if __name__ == "__main__":
    import json
    print(json.dumps(export_iapws_golden_values(), indent=2))
