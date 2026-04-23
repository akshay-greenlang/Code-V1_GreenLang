# -*- coding: utf-8 -*-
"""
Property-Based Tests for Steam Calculation Determinism

This module provides comprehensive property-based tests to validate that
all steam property calculations in GL-003_UnifiedSteam are deterministic
and produce bit-perfect reproducible results.

Determinism Requirements:
- Same inputs ALWAYS produce identical outputs (byte-level reproducibility)
- SHA-256 hashes of outputs must match across multiple invocations
- Decimal precision must remain consistent
- Region detection must be deterministic

IAPWS-IF97 Valid Ranges:
- Region 1 (Compressed Liquid): 273.15 K <= T <= 623.15 K, P <= 100 MPa
- Region 2 (Superheated Vapor): 273.15 K <= T <= 1073.15 K, P <= 100 MPa
- Region 4 (Saturation): 273.15 K <= T <= 647.096 K (critical point)

Author: GL-TestEngineer
"""

import hashlib
import json
import sys
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Tuple

import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck, Phase

# Add parent path for imports
sys.path.insert(0, str(__file__).replace('\\', '/').rsplit('/tests/', 1)[0])

from thermodynamics.iapws_if97 import (
    IF97_CONSTANTS,
    REGION_BOUNDARIES,
    detect_region,
    get_saturation_pressure,
    get_saturation_temperature,
    region1_specific_volume,
    region1_specific_enthalpy,
    region1_specific_entropy,
    region1_specific_internal_energy,
    region1_specific_isobaric_heat_capacity,
    region1_speed_of_sound,
    region2_specific_volume,
    region2_specific_enthalpy,
    region2_specific_entropy,
    region2_specific_internal_energy,
    region2_specific_isobaric_heat_capacity,
    region2_speed_of_sound,
    region4_saturation_properties,
    region4_mixture_enthalpy,
    region4_mixture_entropy,
    region4_mixture_specific_volume,
    compute_calculation_provenance,
    celsius_to_kelvin,
    kelvin_to_celsius,
    kpa_to_mpa,
    mpa_to_kpa,
)

from thermodynamics.enthalpy_balance import (
    StreamData,
    compute_mass_balance,
    compute_energy_balance,
    compute_enthalpy_rate,
    estimate_distribution_losses,
)


# =============================================================================
# IAPWS-IF97 VALID RANGE STRATEGIES
# =============================================================================

# Region 1: Compressed Liquid (subcooled water)
# Valid: 273.15 K <= T <= 623.15 K, P_sat(T) < P <= 100 MPa
region1_temperature_k = st.floats(
    min_value=273.15 + 1.0,  # Slightly above freezing
    max_value=623.15 - 1.0,  # Below Region 1/3 boundary
    allow_nan=False,
    allow_infinity=False,
)

region1_pressure_mpa = st.floats(
    min_value=0.001,  # Above triple point
    max_value=100.0,  # IAPWS-IF97 max for Region 1
    allow_nan=False,
    allow_infinity=False,
)

# Region 2: Superheated Vapor
# Valid: T > T_sat(P) and T <= 800 C (1073.15 K)
region2_temperature_k = st.floats(
    min_value=373.15 + 10.0,  # Safely above 100 C saturation
    max_value=1073.15 - 10.0,  # Below max with margin
    allow_nan=False,
    allow_infinity=False,
)

region2_pressure_mpa = st.floats(
    min_value=0.001,  # Low pressure
    max_value=4.0,  # Keep in Region 2
    allow_nan=False,
    allow_infinity=False,
)

# Region 4: Saturation
# Valid: Triple point to critical point
saturation_temperature_k = st.floats(
    min_value=273.16 + 1.0,  # Just above triple point
    max_value=647.096 - 1.0,  # Just below critical point
    allow_nan=False,
    allow_infinity=False,
)

saturation_pressure_mpa = st.floats(
    min_value=0.001,  # Above triple point pressure
    max_value=22.0,  # Below critical pressure
    allow_nan=False,
    allow_infinity=False,
)

# Steam quality (dryness fraction)
quality_fraction = st.floats(
    min_value=0.0,
    max_value=1.0,
    allow_nan=False,
    allow_infinity=False,
)

# Mass flow rates for balance calculations
mass_flow_kg_s = st.floats(
    min_value=0.001,
    max_value=1000.0,
    allow_nan=False,
    allow_infinity=False,
)

# Enthalpy values
enthalpy_kj_kg = st.floats(
    min_value=0.0,
    max_value=4000.0,
    allow_nan=False,
    allow_infinity=False,
)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_output_hash(output_dict: Dict[str, Any]) -> str:
    """
    Compute SHA-256 hash of calculation output for determinism verification.

    Uses sorted keys and consistent serialization for reproducibility.
    """
    # Convert floats to fixed precision strings for bit-perfect comparison
    def serialize_value(v: Any) -> str:
        if isinstance(v, float):
            # Use high precision but consistent formatting
            return f"{v:.15e}"
        elif isinstance(v, Decimal):
            return str(v)
        elif isinstance(v, dict):
            return json.dumps({k: serialize_value(val) for k, val in sorted(v.items())})
        elif isinstance(v, (list, tuple)):
            return json.dumps([serialize_value(item) for item in v])
        else:
            return str(v)

    serialized = json.dumps(
        {k: serialize_value(v) for k, v in sorted(output_dict.items())},
        sort_keys=True
    )
    return hashlib.sha256(serialized.encode()).hexdigest()


def is_in_region1(P_mpa: float, T_k: float) -> bool:
    """Check if point is in valid Region 1."""
    try:
        if T_k < 273.15 or T_k > 623.15:
            return False
        if P_mpa < REGION_BOUNDARIES["P_MIN"] or P_mpa > 100.0:
            return False
        # Must be subcooled (below saturation)
        T_sat = get_saturation_temperature(min(P_mpa, IF97_CONSTANTS["P_CRIT"]))
        return T_k <= T_sat
    except (ValueError, ZeroDivisionError):
        return False


def is_in_region2(P_mpa: float, T_k: float) -> bool:
    """Check if point is in valid Region 2."""
    try:
        if T_k < 273.15 or T_k > 1073.15:
            return False
        if P_mpa < REGION_BOUNDARIES["P_MIN"] or P_mpa > 100.0:
            return False
        # Must be superheated (above saturation)
        if P_mpa > IF97_CONSTANTS["P_CRIT"]:
            return False  # Supercritical
        T_sat = get_saturation_temperature(P_mpa)
        return T_k > T_sat
    except (ValueError, ZeroDivisionError):
        return False


def is_valid_saturation_pressure(P_mpa: float) -> bool:
    """Check if pressure is valid for saturation calculations."""
    return REGION_BOUNDARIES["P_MIN"] < P_mpa < IF97_CONSTANTS["P_CRIT"]


def is_valid_saturation_temperature(T_k: float) -> bool:
    """Check if temperature is valid for saturation calculations."""
    return 273.16 < T_k < IF97_CONSTANTS["T_CRIT"]


# =============================================================================
# REGION 1 DETERMINISM TESTS
# =============================================================================

@pytest.mark.property
class TestRegion1PropertiesDeterminism:
    """
    Property-based tests for Region 1 (compressed liquid) determinism.

    All calculations must produce identical results when called multiple times
    with the same inputs. SHA-256 hashes verify byte-level reproducibility.
    """

    @given(
        P_mpa=st.floats(min_value=0.5, max_value=50.0, allow_nan=False, allow_infinity=False),
        T_k=st.floats(min_value=280.0, max_value=500.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_region1_enthalpy_determinism(self, P_mpa: float, T_k: float):
        """Test that Region 1 enthalpy calculation is deterministic."""
        # Filter to valid Region 1 points
        assume(is_in_region1(P_mpa, T_k))

        # Calculate enthalpy multiple times
        h1 = region1_specific_enthalpy(P_mpa, T_k)
        h2 = region1_specific_enthalpy(P_mpa, T_k)
        h3 = region1_specific_enthalpy(P_mpa, T_k)

        # All results must be identical
        assert h1 == h2, f"Enthalpy mismatch: {h1} != {h2}"
        assert h2 == h3, f"Enthalpy mismatch: {h2} != {h3}"

        # Verify hash consistency
        hash1 = compute_output_hash({"h": h1})
        hash2 = compute_output_hash({"h": h2})
        hash3 = compute_output_hash({"h": h3})

        assert hash1 == hash2 == hash3, "SHA-256 hash mismatch for enthalpy"

    @given(
        P_mpa=st.floats(min_value=0.5, max_value=50.0, allow_nan=False, allow_infinity=False),
        T_k=st.floats(min_value=280.0, max_value=500.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_region1_entropy_determinism(self, P_mpa: float, T_k: float):
        """Test that Region 1 entropy calculation is deterministic."""
        assume(is_in_region1(P_mpa, T_k))

        s1 = region1_specific_entropy(P_mpa, T_k)
        s2 = region1_specific_entropy(P_mpa, T_k)
        s3 = region1_specific_entropy(P_mpa, T_k)

        assert s1 == s2 == s3, f"Entropy mismatch: {s1}, {s2}, {s3}"

        hash1 = compute_output_hash({"s": s1})
        hash2 = compute_output_hash({"s": s2})
        assert hash1 == hash2, "SHA-256 hash mismatch for entropy"

    @given(
        P_mpa=st.floats(min_value=0.5, max_value=50.0, allow_nan=False, allow_infinity=False),
        T_k=st.floats(min_value=280.0, max_value=500.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_region1_specific_volume_determinism(self, P_mpa: float, T_k: float):
        """Test that Region 1 specific volume calculation is deterministic."""
        assume(is_in_region1(P_mpa, T_k))

        v1 = region1_specific_volume(P_mpa, T_k)
        v2 = region1_specific_volume(P_mpa, T_k)
        v3 = region1_specific_volume(P_mpa, T_k)

        assert v1 == v2 == v3, f"Specific volume mismatch: {v1}, {v2}, {v3}"

    @given(
        P_mpa=st.floats(min_value=0.5, max_value=50.0, allow_nan=False, allow_infinity=False),
        T_k=st.floats(min_value=280.0, max_value=500.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_region1_internal_energy_determinism(self, P_mpa: float, T_k: float):
        """Test that Region 1 internal energy calculation is deterministic."""
        assume(is_in_region1(P_mpa, T_k))

        u1 = region1_specific_internal_energy(P_mpa, T_k)
        u2 = region1_specific_internal_energy(P_mpa, T_k)

        assert u1 == u2, f"Internal energy mismatch: {u1} != {u2}"

    @given(
        P_mpa=st.floats(min_value=0.5, max_value=50.0, allow_nan=False, allow_infinity=False),
        T_k=st.floats(min_value=280.0, max_value=500.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_region1_heat_capacity_determinism(self, P_mpa: float, T_k: float):
        """Test that Region 1 heat capacity calculation is deterministic."""
        assume(is_in_region1(P_mpa, T_k))

        cp1 = region1_specific_isobaric_heat_capacity(P_mpa, T_k)
        cp2 = region1_specific_isobaric_heat_capacity(P_mpa, T_k)

        assert cp1 == cp2, f"Heat capacity mismatch: {cp1} != {cp2}"

    @given(
        P_mpa=st.floats(min_value=0.5, max_value=50.0, allow_nan=False, allow_infinity=False),
        T_k=st.floats(min_value=280.0, max_value=500.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_region1_speed_of_sound_determinism(self, P_mpa: float, T_k: float):
        """Test that Region 1 speed of sound calculation is deterministic."""
        assume(is_in_region1(P_mpa, T_k))

        w1 = region1_speed_of_sound(P_mpa, T_k)
        w2 = region1_speed_of_sound(P_mpa, T_k)

        assert w1 == w2, f"Speed of sound mismatch: {w1} != {w2}"

    @given(
        P_mpa=st.floats(min_value=0.5, max_value=50.0, allow_nan=False, allow_infinity=False),
        T_k=st.floats(min_value=280.0, max_value=500.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_region1_all_properties_hash_consistency(self, P_mpa: float, T_k: float):
        """
        Test that all Region 1 properties together produce consistent hash.

        This is the key determinism test - all properties calculated together
        must produce identical SHA-256 hash on every invocation.
        """
        assume(is_in_region1(P_mpa, T_k))

        # Calculate all properties - first run
        result1 = {
            "P_mpa": P_mpa,
            "T_k": T_k,
            "h": region1_specific_enthalpy(P_mpa, T_k),
            "s": region1_specific_entropy(P_mpa, T_k),
            "v": region1_specific_volume(P_mpa, T_k),
            "u": region1_specific_internal_energy(P_mpa, T_k),
            "cp": region1_specific_isobaric_heat_capacity(P_mpa, T_k),
            "w": region1_speed_of_sound(P_mpa, T_k),
        }

        # Calculate all properties - second run
        result2 = {
            "P_mpa": P_mpa,
            "T_k": T_k,
            "h": region1_specific_enthalpy(P_mpa, T_k),
            "s": region1_specific_entropy(P_mpa, T_k),
            "v": region1_specific_volume(P_mpa, T_k),
            "u": region1_specific_internal_energy(P_mpa, T_k),
            "cp": region1_specific_isobaric_heat_capacity(P_mpa, T_k),
            "w": region1_speed_of_sound(P_mpa, T_k),
        }

        hash1 = compute_output_hash(result1)
        hash2 = compute_output_hash(result2)

        assert hash1 == hash2, (
            f"Region 1 properties hash mismatch at P={P_mpa} MPa, T={T_k} K:\n"
            f"  Hash 1: {hash1}\n"
            f"  Hash 2: {hash2}"
        )

        # Verify hash is 64 characters (SHA-256)
        assert len(hash1) == 64, f"Invalid hash length: {len(hash1)}"


# =============================================================================
# REGION 2 DETERMINISM TESTS
# =============================================================================

@pytest.mark.property
class TestRegion2PropertiesDeterminism:
    """
    Property-based tests for Region 2 (superheated vapor) determinism.
    """

    @given(
        P_mpa=st.floats(min_value=0.01, max_value=4.0, allow_nan=False, allow_infinity=False),
        T_k=st.floats(min_value=400.0, max_value=800.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_region2_enthalpy_determinism(self, P_mpa: float, T_k: float):
        """Test that Region 2 enthalpy calculation is deterministic."""
        assume(is_in_region2(P_mpa, T_k))

        h1 = region2_specific_enthalpy(P_mpa, T_k)
        h2 = region2_specific_enthalpy(P_mpa, T_k)
        h3 = region2_specific_enthalpy(P_mpa, T_k)

        assert h1 == h2 == h3, f"Region 2 enthalpy mismatch: {h1}, {h2}, {h3}"

        hash1 = compute_output_hash({"h": h1})
        hash2 = compute_output_hash({"h": h2})
        assert hash1 == hash2, "SHA-256 hash mismatch for Region 2 enthalpy"

    @given(
        P_mpa=st.floats(min_value=0.01, max_value=4.0, allow_nan=False, allow_infinity=False),
        T_k=st.floats(min_value=400.0, max_value=800.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_region2_entropy_determinism(self, P_mpa: float, T_k: float):
        """Test that Region 2 entropy calculation is deterministic."""
        assume(is_in_region2(P_mpa, T_k))

        s1 = region2_specific_entropy(P_mpa, T_k)
        s2 = region2_specific_entropy(P_mpa, T_k)

        assert s1 == s2, f"Region 2 entropy mismatch: {s1} != {s2}"

    @given(
        P_mpa=st.floats(min_value=0.01, max_value=4.0, allow_nan=False, allow_infinity=False),
        T_k=st.floats(min_value=400.0, max_value=800.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_region2_specific_volume_determinism(self, P_mpa: float, T_k: float):
        """Test that Region 2 specific volume calculation is deterministic."""
        assume(is_in_region2(P_mpa, T_k))

        v1 = region2_specific_volume(P_mpa, T_k)
        v2 = region2_specific_volume(P_mpa, T_k)

        assert v1 == v2, f"Region 2 specific volume mismatch: {v1} != {v2}"

    @given(
        P_mpa=st.floats(min_value=0.01, max_value=4.0, allow_nan=False, allow_infinity=False),
        T_k=st.floats(min_value=400.0, max_value=800.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_region2_internal_energy_determinism(self, P_mpa: float, T_k: float):
        """Test that Region 2 internal energy calculation is deterministic."""
        assume(is_in_region2(P_mpa, T_k))

        u1 = region2_specific_internal_energy(P_mpa, T_k)
        u2 = region2_specific_internal_energy(P_mpa, T_k)

        assert u1 == u2, f"Region 2 internal energy mismatch: {u1} != {u2}"

    @given(
        P_mpa=st.floats(min_value=0.01, max_value=4.0, allow_nan=False, allow_infinity=False),
        T_k=st.floats(min_value=400.0, max_value=800.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_region2_heat_capacity_determinism(self, P_mpa: float, T_k: float):
        """Test that Region 2 heat capacity calculation is deterministic."""
        assume(is_in_region2(P_mpa, T_k))

        cp1 = region2_specific_isobaric_heat_capacity(P_mpa, T_k)
        cp2 = region2_specific_isobaric_heat_capacity(P_mpa, T_k)

        assert cp1 == cp2, f"Region 2 heat capacity mismatch: {cp1} != {cp2}"

    @given(
        P_mpa=st.floats(min_value=0.01, max_value=4.0, allow_nan=False, allow_infinity=False),
        T_k=st.floats(min_value=400.0, max_value=800.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_region2_speed_of_sound_determinism(self, P_mpa: float, T_k: float):
        """Test that Region 2 speed of sound calculation is deterministic."""
        assume(is_in_region2(P_mpa, T_k))

        w1 = region2_speed_of_sound(P_mpa, T_k)
        w2 = region2_speed_of_sound(P_mpa, T_k)

        assert w1 == w2, f"Region 2 speed of sound mismatch: {w1} != {w2}"

    @given(
        P_mpa=st.floats(min_value=0.01, max_value=4.0, allow_nan=False, allow_infinity=False),
        T_k=st.floats(min_value=400.0, max_value=800.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_region2_all_properties_hash_consistency(self, P_mpa: float, T_k: float):
        """
        Test that all Region 2 properties together produce consistent hash.
        """
        assume(is_in_region2(P_mpa, T_k))

        result1 = {
            "P_mpa": P_mpa,
            "T_k": T_k,
            "h": region2_specific_enthalpy(P_mpa, T_k),
            "s": region2_specific_entropy(P_mpa, T_k),
            "v": region2_specific_volume(P_mpa, T_k),
            "u": region2_specific_internal_energy(P_mpa, T_k),
            "cp": region2_specific_isobaric_heat_capacity(P_mpa, T_k),
            "w": region2_speed_of_sound(P_mpa, T_k),
        }

        result2 = {
            "P_mpa": P_mpa,
            "T_k": T_k,
            "h": region2_specific_enthalpy(P_mpa, T_k),
            "s": region2_specific_entropy(P_mpa, T_k),
            "v": region2_specific_volume(P_mpa, T_k),
            "u": region2_specific_internal_energy(P_mpa, T_k),
            "cp": region2_specific_isobaric_heat_capacity(P_mpa, T_k),
            "w": region2_speed_of_sound(P_mpa, T_k),
        }

        hash1 = compute_output_hash(result1)
        hash2 = compute_output_hash(result2)

        assert hash1 == hash2, (
            f"Region 2 properties hash mismatch at P={P_mpa} MPa, T={T_k} K"
        )


# =============================================================================
# SATURATION PRESSURE/TEMPERATURE DETERMINISM TESTS
# =============================================================================

@pytest.mark.property
class TestSaturationDeterminism:
    """
    Property-based tests for saturation pressure and temperature determinism.
    """

    @given(T_k=st.floats(min_value=275.0, max_value=645.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_saturation_pressure_determinism(self, T_k: float):
        """Test that saturation pressure calculation is deterministic."""
        assume(is_valid_saturation_temperature(T_k))

        P1 = get_saturation_pressure(T_k)
        P2 = get_saturation_pressure(T_k)
        P3 = get_saturation_pressure(T_k)

        assert P1 == P2 == P3, f"Saturation pressure mismatch at T={T_k} K: {P1}, {P2}, {P3}"

        hash1 = compute_output_hash({"P_sat": P1, "T_k": T_k})
        hash2 = compute_output_hash({"P_sat": P2, "T_k": T_k})

        assert hash1 == hash2, "SHA-256 hash mismatch for saturation pressure"

    @given(P_mpa=st.floats(min_value=0.001, max_value=22.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_saturation_temperature_determinism(self, P_mpa: float):
        """Test that saturation temperature calculation is deterministic."""
        assume(is_valid_saturation_pressure(P_mpa))

        T1 = get_saturation_temperature(P_mpa)
        T2 = get_saturation_temperature(P_mpa)
        T3 = get_saturation_temperature(P_mpa)

        assert T1 == T2 == T3, f"Saturation temperature mismatch at P={P_mpa} MPa: {T1}, {T2}, {T3}"

        hash1 = compute_output_hash({"T_sat": T1, "P_mpa": P_mpa})
        hash2 = compute_output_hash({"T_sat": T2, "P_mpa": P_mpa})

        assert hash1 == hash2, "SHA-256 hash mismatch for saturation temperature"

    @given(P_mpa=st.floats(min_value=0.001, max_value=22.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_saturation_properties_determinism(self, P_mpa: float):
        """Test that all saturation properties are deterministic."""
        assume(is_valid_saturation_pressure(P_mpa))

        sat1 = region4_saturation_properties(P_mpa)
        sat2 = region4_saturation_properties(P_mpa)

        # All properties must match exactly
        assert sat1.temperature_k == sat2.temperature_k
        assert sat1.hf == sat2.hf
        assert sat1.hg == sat2.hg
        assert sat1.hfg == sat2.hfg
        assert sat1.sf == sat2.sf
        assert sat1.sg == sat2.sg
        assert sat1.sfg == sat2.sfg
        assert sat1.vf == sat2.vf
        assert sat1.vg == sat2.vg

    @given(
        P_mpa=st.floats(min_value=0.001, max_value=22.0, allow_nan=False, allow_infinity=False),
        x=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_mixture_properties_determinism(self, P_mpa: float, x: float):
        """Test that two-phase mixture properties are deterministic."""
        assume(is_valid_saturation_pressure(P_mpa))

        h1 = region4_mixture_enthalpy(P_mpa, x)
        h2 = region4_mixture_enthalpy(P_mpa, x)

        s1 = region4_mixture_entropy(P_mpa, x)
        s2 = region4_mixture_entropy(P_mpa, x)

        v1 = region4_mixture_specific_volume(P_mpa, x)
        v2 = region4_mixture_specific_volume(P_mpa, x)

        assert h1 == h2, f"Mixture enthalpy mismatch: {h1} != {h2}"
        assert s1 == s2, f"Mixture entropy mismatch: {s1} != {s2}"
        assert v1 == v2, f"Mixture specific volume mismatch: {v1} != {v2}"

        # Hash check for full determinism
        result1 = {"P": P_mpa, "x": x, "h": h1, "s": s1, "v": v1}
        result2 = {"P": P_mpa, "x": x, "h": h2, "s": s2, "v": v2}

        assert compute_output_hash(result1) == compute_output_hash(result2)

    @given(P_mpa=st.floats(min_value=0.001, max_value=22.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_saturation_roundtrip_determinism(self, P_mpa: float):
        """
        Test that P -> T_sat -> P_sat roundtrip is deterministic.

        This validates the inverse relationship is consistent.
        """
        assume(is_valid_saturation_pressure(P_mpa))

        # Forward: P -> T_sat
        T_sat1 = get_saturation_temperature(P_mpa)
        T_sat2 = get_saturation_temperature(P_mpa)

        assert T_sat1 == T_sat2, "Forward calculation not deterministic"

        # Backward: T_sat -> P_sat (should recover original P)
        P_recovered1 = get_saturation_pressure(T_sat1)
        P_recovered2 = get_saturation_pressure(T_sat2)

        assert P_recovered1 == P_recovered2, "Backward calculation not deterministic"


# =============================================================================
# STEAM BALANCE CALCULATIONS DETERMINISM TESTS
# =============================================================================

@pytest.mark.property
class TestSteamBalanceDeterminism:
    """
    Property-based tests for steam balance calculation determinism.
    """

    @given(
        m1=st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False),
        m2=st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False),
        m3=st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_mass_balance_determinism(self, m1: float, m2: float, m3: float):
        """Test that mass balance calculation is deterministic."""
        # Create input streams
        input_stream = StreamData(
            stream_id="input_1",
            name="Main Input Supply",
            mass_flow_kg_s=m1 + m2,
            pressure_kpa=500.0,
            temperature_c=150.0,
            specific_enthalpy_kj_kg=2750.0,
        )

        # Create output streams
        output_stream1 = StreamData(
            stream_id="output_1",
            name="Output Process 1 Return",
            mass_flow_kg_s=m1,
            pressure_kpa=500.0,
            temperature_c=150.0,
            specific_enthalpy_kj_kg=2750.0,
        )

        output_stream2 = StreamData(
            stream_id="output_2",
            name="Output Process 2 Return",
            mass_flow_kg_s=m2,
            pressure_kpa=500.0,
            temperature_c=150.0,
            specific_enthalpy_kj_kg=2750.0,
        )

        # Calculate mass balance multiple times
        result1 = compute_mass_balance([input_stream], [output_stream1, output_stream2])
        result2 = compute_mass_balance([input_stream], [output_stream1, output_stream2])

        # Results must be deterministic
        assert result1.total_input_kg_s == result2.total_input_kg_s
        assert result1.total_output_kg_s == result2.total_output_kg_s
        assert result1.imbalance_kg_s == result2.imbalance_kg_s
        assert result1.imbalance_percent == result2.imbalance_percent
        assert result1.is_balanced == result2.is_balanced

        # Provenance hash must be consistent (excluding timestamp)
        assert result1.provenance_hash == result2.provenance_hash

    @given(
        m=st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False),
        h=st.floats(min_value=100.0, max_value=3500.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_enthalpy_rate_determinism(self, m: float, h: float):
        """Test that enthalpy rate calculation is deterministic."""
        rate1 = compute_enthalpy_rate(m, h)
        rate2 = compute_enthalpy_rate(m, h)
        rate3 = compute_enthalpy_rate(m, h)

        assert rate1 == rate2 == rate3, f"Enthalpy rate mismatch: {rate1}, {rate2}, {rate3}"

        # Verify mathematical correctness
        expected = m * h
        assert rate1 == expected, f"Enthalpy rate incorrect: {rate1} != {expected}"

    @given(
        m_in=st.floats(min_value=1.0, max_value=50.0, allow_nan=False, allow_infinity=False),
        m_out=st.floats(min_value=1.0, max_value=50.0, allow_nan=False, allow_infinity=False),
        h_in=st.floats(min_value=500.0, max_value=3500.0, allow_nan=False, allow_infinity=False),
        h_out=st.floats(min_value=500.0, max_value=3500.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_energy_balance_determinism(self, m_in: float, m_out: float, h_in: float, h_out: float):
        """Test that energy balance calculation is deterministic."""
        input_stream = StreamData(
            stream_id="input_1",
            name="Main Input",
            mass_flow_kg_s=m_in,
            pressure_kpa=500.0,
            temperature_c=150.0,
            specific_enthalpy_kj_kg=h_in,
        )

        output_stream = StreamData(
            stream_id="output_1",
            name="Main Output",
            mass_flow_kg_s=m_out,
            pressure_kpa=500.0,
            temperature_c=150.0,
            specific_enthalpy_kj_kg=h_out,
        )

        result1 = compute_energy_balance([input_stream], [output_stream])
        result2 = compute_energy_balance([input_stream], [output_stream])

        assert result1.total_input_kw == result2.total_input_kw
        assert result1.total_output_kw == result2.total_output_kw
        assert result1.imbalance_kw == result2.imbalance_kw
        assert result1.provenance_hash == result2.provenance_hash

    @given(
        ambient_temp=st.floats(min_value=-20.0, max_value=50.0, allow_nan=False, allow_infinity=False),
        steam_temp=st.floats(min_value=100.0, max_value=300.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_distribution_losses_determinism(self, ambient_temp: float, steam_temp: float):
        """Test that distribution loss estimation is deterministic."""
        assume(steam_temp > ambient_temp + 50)  # Reasonable temperature difference

        network_data = {
            "pipes": [
                {"length_m": 100, "outer_diameter_m": 0.1, "insulation_thickness_m": 0.05},
                {"length_m": 50, "outer_diameter_m": 0.05, "insulation_thickness_m": 0.03},
            ],
            "valves": [
                {"type": "gate", "insulated": False, "condition": "good"},
                {"type": "globe", "insulated": True, "condition": "fair"},
            ],
            "traps": [
                {"type": "thermodynamic", "condition": "good"},
                {"type": "float", "condition": "poor"},
            ],
            "steam_temperature_c": steam_temp,
            "steam_pressure_kpa": 500.0,
            "total_input_kw": 1000.0,
        }

        result1 = estimate_distribution_losses(network_data, ambient_temp)
        result2 = estimate_distribution_losses(network_data, ambient_temp)

        assert result1.total_loss_kw == result2.total_loss_kw
        assert result1.pipe_conduction_kw == result2.pipe_conduction_kw
        assert result1.valve_leakage_kw == result2.valve_leakage_kw
        assert result1.trap_losses_kw == result2.trap_losses_kw
        assert result1.flange_losses_kw == result2.flange_losses_kw
        assert result1.provenance_hash == result2.provenance_hash


# =============================================================================
# DECIMAL PRECISION CONSISTENCY TESTS
# =============================================================================

@pytest.mark.property
class TestDecimalPrecisionConsistency:
    """
    Property-based tests for decimal precision consistency.

    Validates that calculations maintain consistent precision across runs.
    """

    @given(
        P_mpa=st.floats(min_value=0.5, max_value=50.0, allow_nan=False, allow_infinity=False),
        T_k=st.floats(min_value=280.0, max_value=500.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_region1_decimal_precision(self, P_mpa: float, T_k: float):
        """Test Region 1 maintains consistent decimal precision."""
        assume(is_in_region1(P_mpa, T_k))

        # Get enthalpy with full precision
        h = region1_specific_enthalpy(P_mpa, T_k)

        # Convert to Decimal for precision analysis
        h_decimal = Decimal(str(h))

        # Calculate again
        h2 = region1_specific_enthalpy(P_mpa, T_k)
        h2_decimal = Decimal(str(h2))

        # Decimal representations must be identical
        assert h_decimal == h2_decimal, (
            f"Decimal precision mismatch: {h_decimal} != {h2_decimal}"
        )

        # Sign, digits, and exponent must match
        assert h_decimal.as_tuple() == h2_decimal.as_tuple()

    @given(
        P_mpa=st.floats(min_value=0.01, max_value=4.0, allow_nan=False, allow_infinity=False),
        T_k=st.floats(min_value=400.0, max_value=800.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_region2_decimal_precision(self, P_mpa: float, T_k: float):
        """Test Region 2 maintains consistent decimal precision."""
        assume(is_in_region2(P_mpa, T_k))

        s = region2_specific_entropy(P_mpa, T_k)
        s_decimal = Decimal(str(s))

        s2 = region2_specific_entropy(P_mpa, T_k)
        s2_decimal = Decimal(str(s2))

        assert s_decimal == s2_decimal

    @given(
        m1=st.floats(min_value=0.001, max_value=1000.0, allow_nan=False, allow_infinity=False),
        m2=st.floats(min_value=0.001, max_value=1000.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_mass_balance_decimal_precision(self, m1: float, m2: float):
        """Test mass balance uses Decimal internally for precision."""
        input_stream = StreamData(
            stream_id="input_1",
            name="Input Supply",
            mass_flow_kg_s=m1 + m2,
        )

        output1 = StreamData(stream_id="out_1", name="Output 1 Return", mass_flow_kg_s=m1)
        output2 = StreamData(stream_id="out_2", name="Output 2 Return", mass_flow_kg_s=m2)

        result = compute_mass_balance([input_stream], [output1, output2])

        # Convert to Decimal for verification
        total_in = Decimal(str(result.total_input_kg_s))
        total_out = Decimal(str(result.total_output_kg_s))
        imbalance = Decimal(str(result.imbalance_kg_s))

        # Verify conservation
        expected_imbalance = total_in - total_out

        # Must be precisely consistent
        assert imbalance == expected_imbalance, (
            f"Precision loss in mass balance: {imbalance} != {expected_imbalance}"
        )


# =============================================================================
# PROVENANCE HASH DETERMINISM TESTS
# =============================================================================

@pytest.mark.property
class TestProvenanceHashDeterminism:
    """
    Property-based tests for SHA-256 provenance hash determinism.
    """

    @given(
        inputs=st.fixed_dictionaries({
            "pressure_kpa": st.floats(min_value=100, max_value=10000, allow_nan=False),
            "temperature_c": st.floats(min_value=50, max_value=500, allow_nan=False),
        }),
        outputs=st.fixed_dictionaries({
            "h": st.floats(min_value=100, max_value=4000, allow_nan=False),
            "s": st.floats(min_value=0, max_value=10, allow_nan=False),
        }),
    )
    @settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_provenance_hash_determinism(self, inputs: Dict, outputs: Dict):
        """Test that provenance hash is deterministic."""
        hash1 = compute_calculation_provenance(inputs, outputs)
        hash2 = compute_calculation_provenance(inputs, outputs)
        hash3 = compute_calculation_provenance(inputs, outputs)

        assert hash1 == hash2 == hash3, (
            f"Provenance hash not deterministic:\n"
            f"  Hash 1: {hash1}\n"
            f"  Hash 2: {hash2}\n"
            f"  Hash 3: {hash3}"
        )

        # Verify it's a valid SHA-256 hash (64 hex characters)
        assert len(hash1) == 64
        assert all(c in '0123456789abcdef' for c in hash1)

    @given(
        P=st.floats(min_value=100, max_value=10000, allow_nan=False),
        T=st.floats(min_value=50, max_value=500, allow_nan=False),
        delta=st.floats(min_value=0.001, max_value=0.01, allow_nan=False),
    )
    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_provenance_hash_sensitivity(self, P: float, T: float, delta: float):
        """Test that provenance hash changes with input changes."""
        inputs1 = {"pressure_kpa": P, "temperature_c": T}
        inputs2 = {"pressure_kpa": P + delta, "temperature_c": T}
        outputs = {"h": 2500.0, "s": 6.5}

        hash1 = compute_calculation_provenance(inputs1, outputs)
        hash2 = compute_calculation_provenance(inputs2, outputs)

        # Different inputs should produce different hashes
        assert hash1 != hash2, (
            f"Provenance hash collision detected for different inputs"
        )


# =============================================================================
# OPTIMIZATION OUTPUT REPRODUCIBILITY TESTS
# =============================================================================

@pytest.mark.property
class TestOptimizationReproducibility:
    """
    Property-based tests for optimization output reproducibility.

    Note: These tests import the optimizer module and verify that given
    the same inputs, optimization produces identical outputs.
    """

    @given(
        demand=st.floats(min_value=10.0, max_value=500.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_load_allocation_reproducibility(self, demand: float):
        """Test that load allocation is reproducible given same inputs."""
        try:
            from optimization.steam_network_optimizer import (
                SteamNetworkOptimizer,
                BoilerState,
            )
        except ImportError:
            pytest.skip("Optimizer module not available")

        # Create deterministic boiler configuration
        boilers = [
            BoilerState(
                boiler_id="B1",
                is_online=True,
                current_load_percent=50.0,
                rated_capacity_klb_hr=100.0,
                current_efficiency_percent=85.0,
            ),
            BoilerState(
                boiler_id="B2",
                is_online=True,
                current_load_percent=60.0,
                rated_capacity_klb_hr=80.0,
                current_efficiency_percent=83.0,
            ),
        ]

        optimizer = SteamNetworkOptimizer()

        # Ensure demand is within capacity
        total_capacity = sum(b.rated_capacity_klb_hr for b in boilers)
        actual_demand = min(demand, total_capacity * 0.9)

        # Run optimization multiple times
        result1 = optimizer.optimize_load_allocation(boilers, actual_demand, "cost")
        result2 = optimizer.optimize_load_allocation(boilers, actual_demand, "cost")

        # Results must be identical (excluding timestamp)
        assert result1.total_demand_klb_hr == result2.total_demand_klb_hr
        assert result1.total_cost_per_hr == result2.total_cost_per_hr
        assert result1.total_co2_lb_hr == result2.total_co2_lb_hr
        assert result1.weighted_efficiency == result2.weighted_efficiency

        # Allocations must match
        for a1, a2 in zip(result1.allocations, result2.allocations):
            assert a1.boiler_id == a2.boiler_id
            assert a1.recommended_load_percent == a2.recommended_load_percent
            assert a1.recommended_output_klb_hr == a2.recommended_output_klb_hr
            assert a1.efficiency_at_load == a2.efficiency_at_load
            assert a1.cost_at_load == a2.cost_at_load


# =============================================================================
# MULTI-RUN HASH CONSISTENCY TESTS
# =============================================================================

@pytest.mark.property
class TestMultiRunHashConsistency:
    """
    Tests that verify calculations produce identical SHA-256 hashes
    across multiple runs, validating byte-level reproducibility.
    """

    NUM_RUNS = 10  # Number of runs for hash consistency check

    @given(
        P_mpa=st.floats(min_value=0.5, max_value=50.0, allow_nan=False, allow_infinity=False),
        T_k=st.floats(min_value=280.0, max_value=500.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_region1_multi_run_hash_consistency(self, P_mpa: float, T_k: float):
        """
        Test that Region 1 calculations produce identical hash across 10 runs.
        """
        assume(is_in_region1(P_mpa, T_k))

        hashes = []
        for _ in range(self.NUM_RUNS):
            result = {
                "h": region1_specific_enthalpy(P_mpa, T_k),
                "s": region1_specific_entropy(P_mpa, T_k),
                "v": region1_specific_volume(P_mpa, T_k),
                "u": region1_specific_internal_energy(P_mpa, T_k),
            }
            hashes.append(compute_output_hash(result))

        # All hashes must be identical
        assert len(set(hashes)) == 1, (
            f"Hash inconsistency across {self.NUM_RUNS} runs:\n"
            f"Unique hashes: {set(hashes)}"
        )

    @given(
        P_mpa=st.floats(min_value=0.01, max_value=4.0, allow_nan=False, allow_infinity=False),
        T_k=st.floats(min_value=400.0, max_value=800.0, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_region2_multi_run_hash_consistency(self, P_mpa: float, T_k: float):
        """
        Test that Region 2 calculations produce identical hash across 10 runs.
        """
        assume(is_in_region2(P_mpa, T_k))

        hashes = []
        for _ in range(self.NUM_RUNS):
            result = {
                "h": region2_specific_enthalpy(P_mpa, T_k),
                "s": region2_specific_entropy(P_mpa, T_k),
                "v": region2_specific_volume(P_mpa, T_k),
                "u": region2_specific_internal_energy(P_mpa, T_k),
            }
            hashes.append(compute_output_hash(result))

        assert len(set(hashes)) == 1, (
            f"Hash inconsistency across {self.NUM_RUNS} runs"
        )

    @given(P_mpa=st.floats(min_value=0.001, max_value=22.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_saturation_multi_run_hash_consistency(self, P_mpa: float):
        """
        Test that saturation calculations produce identical hash across 10 runs.
        """
        assume(is_valid_saturation_pressure(P_mpa))

        hashes = []
        for _ in range(self.NUM_RUNS):
            sat = region4_saturation_properties(P_mpa)
            result = {
                "T": sat.temperature_k,
                "hf": sat.hf,
                "hg": sat.hg,
                "sf": sat.sf,
                "sg": sat.sg,
                "vf": sat.vf,
                "vg": sat.vg,
            }
            hashes.append(compute_output_hash(result))

        assert len(set(hashes)) == 1, (
            f"Saturation hash inconsistency across {self.NUM_RUNS} runs"
        )


# =============================================================================
# UNIT CONVERSION DETERMINISM TESTS
# =============================================================================

@pytest.mark.property
class TestUnitConversionDeterminism:
    """
    Tests for unit conversion function determinism.
    """

    @given(T_c=st.floats(min_value=-273.15, max_value=2000.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=200)
    def test_celsius_kelvin_roundtrip(self, T_c: float):
        """Test Celsius <-> Kelvin roundtrip is deterministic and lossless."""
        T_k = celsius_to_kelvin(T_c)
        T_c_recovered = kelvin_to_celsius(T_k)

        # Must recover exact original value
        assert T_c == T_c_recovered, (
            f"Roundtrip loss: {T_c} -> {T_k} -> {T_c_recovered}"
        )

    @given(P_kpa=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False))
    @settings(max_examples=200)
    def test_kpa_mpa_roundtrip(self, P_kpa: float):
        """Test kPa <-> MPa roundtrip is deterministic and lossless."""
        P_mpa = kpa_to_mpa(P_kpa)
        P_kpa_recovered = mpa_to_kpa(P_mpa)

        # Check for precision (may have small floating point errors)
        assert abs(P_kpa - P_kpa_recovered) < 1e-10 * max(abs(P_kpa), 1.0), (
            f"Roundtrip loss: {P_kpa} -> {P_mpa} -> {P_kpa_recovered}"
        )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "-m", "property",
        "--hypothesis-show-statistics",
        "--tb=short",
    ])
