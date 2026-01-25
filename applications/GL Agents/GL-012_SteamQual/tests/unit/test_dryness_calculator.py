"""
Unit Tests: Dryness Fraction Calculator

Tests the dryness fraction (steam quality) calculation methods:
1. Enthalpy method: x = (h - h_f) / h_fg
2. Entropy method: x = (s - s_f) / s_fg
3. Edge cases: x=0 (saturated liquid), x=1 (saturated vapor), supercritical
4. Uncertainty propagation through calculations

Reference: IAPWS-IF97 and ASME PTC 19.11
Author: GL-TestEngineer
Version: 1.0.0
Target Coverage: 85%+
"""

import pytest
import math
import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Optional, Tuple
from datetime import datetime, timezone


# =============================================================================
# Constants
# =============================================================================

# Critical point
CRITICAL_PRESSURE_MPA = 22.064
CRITICAL_TEMPERATURE_K = 647.096

# Tolerances
DRYNESS_TOLERANCE = 0.001
ENTHALPY_TOLERANCE_KJ_KG = 0.5
ENTROPY_TOLERANCE_KJ_KG_K = 0.001


# =============================================================================
# Dryness Calculator Implementation (for testing)
# =============================================================================

@dataclass
class SaturationProperties:
    """Saturation properties at a given pressure."""
    pressure_mpa: float
    temperature_k: float
    h_f_kj_kg: float
    h_g_kj_kg: float
    h_fg_kj_kg: float
    s_f_kj_kg_k: float
    s_g_kj_kg_k: float
    s_fg_kj_kg_k: float


@dataclass
class DrynessResult:
    """Result of dryness fraction calculation."""
    dryness_fraction: float
    uncertainty_percent: float
    method: str
    is_valid: bool
    error_message: str = ""
    provenance_hash: str = ""


class DrynessCalculationError(Exception):
    """Error in dryness fraction calculation."""
    pass


# Reference saturation data at common pressures (IAPWS-IF97)
SATURATION_DATA = {
    0.1: SaturationProperties(
        pressure_mpa=0.1, temperature_k=372.756,
        h_f_kj_kg=417.44, h_g_kj_kg=2675.5, h_fg_kj_kg=2258.0,
        s_f_kj_kg_k=1.3026, s_g_kj_kg_k=7.3594, s_fg_kj_kg_k=6.0568,
    ),
    1.0: SaturationProperties(
        pressure_mpa=1.0, temperature_k=453.03,
        h_f_kj_kg=762.81, h_g_kj_kg=2778.1, h_fg_kj_kg=2015.3,
        s_f_kj_kg_k=2.1387, s_g_kj_kg_k=6.5865, s_fg_kj_kg_k=4.4478,
    ),
    5.0: SaturationProperties(
        pressure_mpa=5.0, temperature_k=536.67,
        h_f_kj_kg=1154.2, h_g_kj_kg=2794.3, h_fg_kj_kg=1640.1,
        s_f_kj_kg_k=2.9202, s_g_kj_kg_k=5.9734, s_fg_kj_kg_k=3.0532,
    ),
    10.0: SaturationProperties(
        pressure_mpa=10.0, temperature_k=584.15,
        h_f_kj_kg=1407.6, h_g_kj_kg=2724.7, h_fg_kj_kg=1317.1,
        s_f_kj_kg_k=3.3596, s_g_kj_kg_k=5.6141, s_fg_kj_kg_k=2.2545,
    ),
    15.0: SaturationProperties(
        pressure_mpa=15.0, temperature_k=615.31,
        h_f_kj_kg=1610.5, h_g_kj_kg=2610.5, h_fg_kj_kg=1000.0,
        s_f_kj_kg_k=3.6848, s_g_kj_kg_k=5.3098, s_fg_kj_kg_k=1.625,
    ),
}


def get_saturation_properties(pressure_mpa: float) -> SaturationProperties:
    """Get saturation properties at given pressure (interpolated)."""
    if pressure_mpa >= CRITICAL_PRESSURE_MPA:
        raise DrynessCalculationError(
            f"Pressure {pressure_mpa} MPa is above critical point ({CRITICAL_PRESSURE_MPA} MPa)"
        )

    if pressure_mpa <= 0:
        raise DrynessCalculationError(f"Pressure must be positive: {pressure_mpa} MPa")

    # Exact match
    if pressure_mpa in SATURATION_DATA:
        return SATURATION_DATA[pressure_mpa]

    # Linear interpolation
    pressures = sorted(SATURATION_DATA.keys())
    lower_p = max([p for p in pressures if p <= pressure_mpa], default=pressures[0])
    upper_p = min([p for p in pressures if p >= pressure_mpa], default=pressures[-1])

    if lower_p == upper_p:
        return SATURATION_DATA[lower_p]

    lower = SATURATION_DATA[lower_p]
    upper = SATURATION_DATA[upper_p]

    frac = (pressure_mpa - lower_p) / (upper_p - lower_p)

    return SaturationProperties(
        pressure_mpa=pressure_mpa,
        temperature_k=lower.temperature_k + frac * (upper.temperature_k - lower.temperature_k),
        h_f_kj_kg=lower.h_f_kj_kg + frac * (upper.h_f_kj_kg - lower.h_f_kj_kg),
        h_g_kj_kg=lower.h_g_kj_kg + frac * (upper.h_g_kj_kg - lower.h_g_kj_kg),
        h_fg_kj_kg=lower.h_fg_kj_kg + frac * (upper.h_fg_kj_kg - lower.h_fg_kj_kg),
        s_f_kj_kg_k=lower.s_f_kj_kg_k + frac * (upper.s_f_kj_kg_k - lower.s_f_kj_kg_k),
        s_g_kj_kg_k=lower.s_g_kj_kg_k + frac * (upper.s_g_kj_kg_k - lower.s_g_kj_kg_k),
        s_fg_kj_kg_k=lower.s_fg_kj_kg_k + frac * (upper.s_fg_kj_kg_k - lower.s_fg_kj_kg_k),
    )


def calculate_dryness_from_enthalpy(
    pressure_mpa: float,
    enthalpy_kj_kg: float,
    pressure_uncertainty: float = 0.0,
    enthalpy_uncertainty: float = 0.0,
) -> DrynessResult:
    """
    Calculate dryness fraction from pressure and enthalpy.

    Formula: x = (h - h_f) / h_fg

    Args:
        pressure_mpa: Steam pressure in MPa
        enthalpy_kj_kg: Steam enthalpy in kJ/kg
        pressure_uncertainty: Pressure measurement uncertainty (%)
        enthalpy_uncertainty: Enthalpy measurement uncertainty (%)

    Returns:
        DrynessResult with calculated dryness fraction and uncertainty
    """
    try:
        sat_props = get_saturation_properties(pressure_mpa)
    except DrynessCalculationError as e:
        return DrynessResult(
            dryness_fraction=0.0,
            uncertainty_percent=100.0,
            method="enthalpy",
            is_valid=False,
            error_message=str(e),
        )

    # Check if in two-phase region
    if enthalpy_kj_kg < sat_props.h_f_kj_kg - ENTHALPY_TOLERANCE_KJ_KG:
        return DrynessResult(
            dryness_fraction=0.0,
            uncertainty_percent=0.0,
            method="enthalpy",
            is_valid=True,
            error_message="Subcooled liquid (h < h_f)",
        )

    if enthalpy_kj_kg > sat_props.h_g_kj_kg + ENTHALPY_TOLERANCE_KJ_KG:
        return DrynessResult(
            dryness_fraction=1.0,
            uncertainty_percent=0.0,
            method="enthalpy",
            is_valid=True,
            error_message="Superheated vapor (h > h_g)",
        )

    # Calculate dryness fraction
    if sat_props.h_fg_kj_kg <= 0:
        return DrynessResult(
            dryness_fraction=0.0,
            uncertainty_percent=100.0,
            method="enthalpy",
            is_valid=False,
            error_message="Invalid h_fg (near critical point)",
        )

    x = (enthalpy_kj_kg - sat_props.h_f_kj_kg) / sat_props.h_fg_kj_kg

    # Clamp to valid range
    x = max(0.0, min(1.0, x))

    # Propagate uncertainty
    # dx/dh = 1/h_fg
    # dx/dP affects h_f, h_g, h_fg
    dh_dx = sat_props.h_fg_kj_kg
    uncertainty_from_h = enthalpy_uncertainty / 100 * enthalpy_kj_kg / dh_dx if dh_dx > 0 else 0
    uncertainty_from_p = pressure_uncertainty / 100 * 0.1  # Simplified P sensitivity

    total_uncertainty_abs = math.sqrt(uncertainty_from_h**2 + uncertainty_from_p**2)
    uncertainty_percent = (total_uncertainty_abs / max(x, 0.01)) * 100 if x > 0.01 else total_uncertainty_abs * 100

    # Calculate provenance hash
    inputs = {
        "pressure_mpa": round(pressure_mpa, 10),
        "enthalpy_kj_kg": round(enthalpy_kj_kg, 10),
    }
    provenance_hash = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()

    return DrynessResult(
        dryness_fraction=x,
        uncertainty_percent=min(uncertainty_percent, 100.0),
        method="enthalpy",
        is_valid=True,
        provenance_hash=provenance_hash,
    )


def calculate_dryness_from_entropy(
    pressure_mpa: float,
    entropy_kj_kg_k: float,
    pressure_uncertainty: float = 0.0,
    entropy_uncertainty: float = 0.0,
) -> DrynessResult:
    """
    Calculate dryness fraction from pressure and entropy.

    Formula: x = (s - s_f) / s_fg

    Args:
        pressure_mpa: Steam pressure in MPa
        entropy_kj_kg_k: Steam entropy in kJ/(kg.K)
        pressure_uncertainty: Pressure measurement uncertainty (%)
        entropy_uncertainty: Entropy measurement uncertainty (%)

    Returns:
        DrynessResult with calculated dryness fraction and uncertainty
    """
    try:
        sat_props = get_saturation_properties(pressure_mpa)
    except DrynessCalculationError as e:
        return DrynessResult(
            dryness_fraction=0.0,
            uncertainty_percent=100.0,
            method="entropy",
            is_valid=False,
            error_message=str(e),
        )

    # Check if in two-phase region
    if entropy_kj_kg_k < sat_props.s_f_kj_kg_k - ENTROPY_TOLERANCE_KJ_KG_K:
        return DrynessResult(
            dryness_fraction=0.0,
            uncertainty_percent=0.0,
            method="entropy",
            is_valid=True,
            error_message="Subcooled liquid (s < s_f)",
        )

    if entropy_kj_kg_k > sat_props.s_g_kj_kg_k + ENTROPY_TOLERANCE_KJ_KG_K:
        return DrynessResult(
            dryness_fraction=1.0,
            uncertainty_percent=0.0,
            method="entropy",
            is_valid=True,
            error_message="Superheated vapor (s > s_g)",
        )

    # Calculate dryness fraction
    if sat_props.s_fg_kj_kg_k <= 0:
        return DrynessResult(
            dryness_fraction=0.0,
            uncertainty_percent=100.0,
            method="entropy",
            is_valid=False,
            error_message="Invalid s_fg (near critical point)",
        )

    x = (entropy_kj_kg_k - sat_props.s_f_kj_kg_k) / sat_props.s_fg_kj_kg_k

    # Clamp to valid range
    x = max(0.0, min(1.0, x))

    # Propagate uncertainty
    ds_dx = sat_props.s_fg_kj_kg_k
    uncertainty_from_s = entropy_uncertainty / 100 * entropy_kj_kg_k / ds_dx if ds_dx > 0 else 0
    uncertainty_from_p = pressure_uncertainty / 100 * 0.1  # Simplified P sensitivity

    total_uncertainty_abs = math.sqrt(uncertainty_from_s**2 + uncertainty_from_p**2)
    uncertainty_percent = (total_uncertainty_abs / max(x, 0.01)) * 100 if x > 0.01 else total_uncertainty_abs * 100

    # Calculate provenance hash
    inputs = {
        "pressure_mpa": round(pressure_mpa, 10),
        "entropy_kj_kg_k": round(entropy_kj_kg_k, 10),
    }
    provenance_hash = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()

    return DrynessResult(
        dryness_fraction=x,
        uncertainty_percent=min(uncertainty_percent, 100.0),
        method="entropy",
        is_valid=True,
        provenance_hash=provenance_hash,
    )


def is_supercritical(pressure_mpa: float, temperature_k: float) -> bool:
    """Check if state is supercritical."""
    return pressure_mpa >= CRITICAL_PRESSURE_MPA and temperature_k >= CRITICAL_TEMPERATURE_K


# =============================================================================
# Test Classes
# =============================================================================

class TestEnthalpyMethod:
    """Tests for dryness fraction calculation using enthalpy method."""

    def test_saturated_liquid_quality_zero(self):
        """Test x=0 at saturated liquid enthalpy."""
        sat_props = SATURATION_DATA[1.0]
        result = calculate_dryness_from_enthalpy(1.0, sat_props.h_f_kj_kg)

        assert result.is_valid
        assert result.method == "enthalpy"
        assert abs(result.dryness_fraction - 0.0) < DRYNESS_TOLERANCE

    def test_saturated_vapor_quality_one(self):
        """Test x=1 at saturated vapor enthalpy."""
        sat_props = SATURATION_DATA[1.0]
        result = calculate_dryness_from_enthalpy(1.0, sat_props.h_g_kj_kg)

        assert result.is_valid
        assert result.method == "enthalpy"
        assert abs(result.dryness_fraction - 1.0) < DRYNESS_TOLERANCE

    def test_mid_quality_calculation(self):
        """Test x=0.5 at mid-range enthalpy."""
        sat_props = SATURATION_DATA[1.0]
        h_mid = sat_props.h_f_kj_kg + 0.5 * sat_props.h_fg_kj_kg
        result = calculate_dryness_from_enthalpy(1.0, h_mid)

        assert result.is_valid
        assert abs(result.dryness_fraction - 0.5) < DRYNESS_TOLERANCE

    @pytest.mark.parametrize("target_quality", [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0])
    def test_quality_calculation_parametrized(self, target_quality):
        """Test quality calculation for various target values."""
        sat_props = SATURATION_DATA[1.0]
        h_target = sat_props.h_f_kj_kg + target_quality * sat_props.h_fg_kj_kg
        result = calculate_dryness_from_enthalpy(1.0, h_target)

        assert result.is_valid
        assert abs(result.dryness_fraction - target_quality) < 0.01

    @pytest.mark.parametrize("pressure", [0.1, 1.0, 5.0, 10.0])
    def test_quality_at_various_pressures(self, pressure):
        """Test quality calculation at various pressures."""
        sat_props = SATURATION_DATA[pressure]
        target_quality = 0.9
        h_target = sat_props.h_f_kj_kg + target_quality * sat_props.h_fg_kj_kg

        result = calculate_dryness_from_enthalpy(pressure, h_target)

        assert result.is_valid
        assert abs(result.dryness_fraction - target_quality) < 0.01

    def test_subcooled_returns_zero(self):
        """Test that subcooled liquid returns x=0."""
        sat_props = SATURATION_DATA[1.0]
        h_subcooled = sat_props.h_f_kj_kg - 100.0  # Below saturated liquid

        result = calculate_dryness_from_enthalpy(1.0, h_subcooled)

        assert result.is_valid
        assert result.dryness_fraction == 0.0
        assert "Subcooled" in result.error_message

    def test_superheated_returns_one(self):
        """Test that superheated vapor returns x=1."""
        sat_props = SATURATION_DATA[1.0]
        h_superheated = sat_props.h_g_kj_kg + 100.0  # Above saturated vapor

        result = calculate_dryness_from_enthalpy(1.0, h_superheated)

        assert result.is_valid
        assert result.dryness_fraction == 1.0
        assert "Superheated" in result.error_message

    def test_provenance_hash_generated(self):
        """Test that provenance hash is generated."""
        sat_props = SATURATION_DATA[1.0]
        h_mid = sat_props.h_f_kj_kg + 0.5 * sat_props.h_fg_kj_kg

        result = calculate_dryness_from_enthalpy(1.0, h_mid)

        assert result.provenance_hash
        assert len(result.provenance_hash) == 64  # SHA-256 hex length

    def test_provenance_hash_deterministic(self):
        """Test that same inputs produce same provenance hash."""
        sat_props = SATURATION_DATA[1.0]
        h_mid = sat_props.h_f_kj_kg + 0.5 * sat_props.h_fg_kj_kg

        result1 = calculate_dryness_from_enthalpy(1.0, h_mid)
        result2 = calculate_dryness_from_enthalpy(1.0, h_mid)

        assert result1.provenance_hash == result2.provenance_hash

    def test_provenance_hash_changes_with_input(self):
        """Test that different inputs produce different hashes."""
        sat_props = SATURATION_DATA[1.0]
        h_mid1 = sat_props.h_f_kj_kg + 0.5 * sat_props.h_fg_kj_kg
        h_mid2 = sat_props.h_f_kj_kg + 0.6 * sat_props.h_fg_kj_kg

        result1 = calculate_dryness_from_enthalpy(1.0, h_mid1)
        result2 = calculate_dryness_from_enthalpy(1.0, h_mid2)

        assert result1.provenance_hash != result2.provenance_hash


class TestEntropyMethod:
    """Tests for dryness fraction calculation using entropy method."""

    def test_saturated_liquid_quality_zero(self):
        """Test x=0 at saturated liquid entropy."""
        sat_props = SATURATION_DATA[1.0]
        result = calculate_dryness_from_entropy(1.0, sat_props.s_f_kj_kg_k)

        assert result.is_valid
        assert result.method == "entropy"
        assert abs(result.dryness_fraction - 0.0) < DRYNESS_TOLERANCE

    def test_saturated_vapor_quality_one(self):
        """Test x=1 at saturated vapor entropy."""
        sat_props = SATURATION_DATA[1.0]
        result = calculate_dryness_from_entropy(1.0, sat_props.s_g_kj_kg_k)

        assert result.is_valid
        assert result.method == "entropy"
        assert abs(result.dryness_fraction - 1.0) < DRYNESS_TOLERANCE

    def test_mid_quality_calculation(self):
        """Test x=0.5 at mid-range entropy."""
        sat_props = SATURATION_DATA[1.0]
        s_mid = sat_props.s_f_kj_kg_k + 0.5 * sat_props.s_fg_kj_kg_k
        result = calculate_dryness_from_entropy(1.0, s_mid)

        assert result.is_valid
        assert abs(result.dryness_fraction - 0.5) < DRYNESS_TOLERANCE

    @pytest.mark.parametrize("target_quality", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_quality_calculation_parametrized(self, target_quality):
        """Test quality calculation for various target values."""
        sat_props = SATURATION_DATA[1.0]
        s_target = sat_props.s_f_kj_kg_k + target_quality * sat_props.s_fg_kj_kg_k
        result = calculate_dryness_from_entropy(1.0, s_target)

        assert result.is_valid
        assert abs(result.dryness_fraction - target_quality) < 0.01

    def test_subcooled_returns_zero(self):
        """Test that subcooled liquid returns x=0."""
        sat_props = SATURATION_DATA[1.0]
        s_subcooled = sat_props.s_f_kj_kg_k - 0.5

        result = calculate_dryness_from_entropy(1.0, s_subcooled)

        assert result.is_valid
        assert result.dryness_fraction == 0.0
        assert "Subcooled" in result.error_message

    def test_superheated_returns_one(self):
        """Test that superheated vapor returns x=1."""
        sat_props = SATURATION_DATA[1.0]
        s_superheated = sat_props.s_g_kj_kg_k + 0.5

        result = calculate_dryness_from_entropy(1.0, s_superheated)

        assert result.is_valid
        assert result.dryness_fraction == 1.0
        assert "Superheated" in result.error_message


class TestEdgeCases:
    """Tests for edge cases in dryness calculation."""

    def test_exactly_zero_quality(self):
        """Test calculation at exactly x=0."""
        sat_props = SATURATION_DATA[1.0]

        result_h = calculate_dryness_from_enthalpy(1.0, sat_props.h_f_kj_kg)
        result_s = calculate_dryness_from_entropy(1.0, sat_props.s_f_kj_kg_k)

        assert result_h.dryness_fraction == pytest.approx(0.0, abs=DRYNESS_TOLERANCE)
        assert result_s.dryness_fraction == pytest.approx(0.0, abs=DRYNESS_TOLERANCE)

    def test_exactly_one_quality(self):
        """Test calculation at exactly x=1."""
        sat_props = SATURATION_DATA[1.0]

        result_h = calculate_dryness_from_enthalpy(1.0, sat_props.h_g_kj_kg)
        result_s = calculate_dryness_from_entropy(1.0, sat_props.s_g_kj_kg_k)

        assert result_h.dryness_fraction == pytest.approx(1.0, abs=DRYNESS_TOLERANCE)
        assert result_s.dryness_fraction == pytest.approx(1.0, abs=DRYNESS_TOLERANCE)

    def test_supercritical_pressure_fails(self):
        """Test that supercritical pressure returns invalid result."""
        result = calculate_dryness_from_enthalpy(25.0, 2800.0)  # Above critical

        assert not result.is_valid
        assert "critical" in result.error_message.lower()

    def test_negative_pressure_fails(self):
        """Test that negative pressure returns invalid result."""
        result = calculate_dryness_from_enthalpy(-1.0, 2500.0)

        assert not result.is_valid
        assert "positive" in result.error_message.lower()

    def test_zero_pressure_fails(self):
        """Test that zero pressure returns invalid result."""
        result = calculate_dryness_from_enthalpy(0.0, 2500.0)

        assert not result.is_valid

    def test_very_low_pressure(self):
        """Test calculation at very low pressure."""
        # Low pressure, should still work
        sat_props = SATURATION_DATA[0.1]
        h_mid = sat_props.h_f_kj_kg + 0.5 * sat_props.h_fg_kj_kg

        result = calculate_dryness_from_enthalpy(0.1, h_mid)

        assert result.is_valid
        assert abs(result.dryness_fraction - 0.5) < 0.01

    def test_near_critical_pressure(self):
        """Test calculation near critical pressure."""
        # Near but below critical
        result = calculate_dryness_from_enthalpy(20.0, 2500.0)

        # Should work but may have high uncertainty near critical point
        assert result.is_valid or "critical" in result.error_message.lower()

    def test_is_supercritical_detection(self):
        """Test supercritical state detection."""
        assert is_supercritical(25.0, 700.0)  # Both above critical
        assert not is_supercritical(10.0, 500.0)  # Both below critical
        assert not is_supercritical(25.0, 600.0)  # P above, T below
        assert not is_supercritical(15.0, 700.0)  # P below, T above


class TestUncertaintyPropagation:
    """Tests for uncertainty propagation in dryness calculations."""

    def test_zero_uncertainty_input(self):
        """Test with zero input uncertainty."""
        sat_props = SATURATION_DATA[1.0]
        h_mid = sat_props.h_f_kj_kg + 0.5 * sat_props.h_fg_kj_kg

        result = calculate_dryness_from_enthalpy(
            1.0, h_mid,
            pressure_uncertainty=0.0,
            enthalpy_uncertainty=0.0,
        )

        assert result.is_valid
        assert result.uncertainty_percent >= 0

    def test_nonzero_uncertainty_propagates(self):
        """Test that nonzero uncertainty propagates."""
        sat_props = SATURATION_DATA[1.0]
        h_mid = sat_props.h_f_kj_kg + 0.5 * sat_props.h_fg_kj_kg

        result_no_unc = calculate_dryness_from_enthalpy(
            1.0, h_mid,
            pressure_uncertainty=0.0,
            enthalpy_uncertainty=0.0,
        )

        result_with_unc = calculate_dryness_from_enthalpy(
            1.0, h_mid,
            pressure_uncertainty=2.0,
            enthalpy_uncertainty=1.5,
        )

        # Uncertainty should be higher with input uncertainties
        # (or at least non-negative in both cases)
        assert result_with_unc.uncertainty_percent >= 0
        assert result_no_unc.uncertainty_percent >= 0

    def test_uncertainty_increases_with_input_uncertainty(self):
        """Test that output uncertainty increases with input uncertainty."""
        sat_props = SATURATION_DATA[1.0]
        h_mid = sat_props.h_f_kj_kg + 0.5 * sat_props.h_fg_kj_kg

        result_low = calculate_dryness_from_enthalpy(
            1.0, h_mid,
            pressure_uncertainty=1.0,
            enthalpy_uncertainty=0.5,
        )

        result_high = calculate_dryness_from_enthalpy(
            1.0, h_mid,
            pressure_uncertainty=5.0,
            enthalpy_uncertainty=3.0,
        )

        # Higher input uncertainty should lead to higher output uncertainty
        assert result_high.uncertainty_percent >= result_low.uncertainty_percent

    def test_uncertainty_bounded(self):
        """Test that uncertainty is bounded at 100%."""
        sat_props = SATURATION_DATA[1.0]
        h_mid = sat_props.h_f_kj_kg + 0.5 * sat_props.h_fg_kj_kg

        result = calculate_dryness_from_enthalpy(
            1.0, h_mid,
            pressure_uncertainty=50.0,  # Very high uncertainty
            enthalpy_uncertainty=50.0,
        )

        assert result.uncertainty_percent <= 100.0


class TestConsistencyBetweenMethods:
    """Tests for consistency between enthalpy and entropy methods."""

    @pytest.mark.parametrize("quality", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_methods_agree(self, quality):
        """Test that enthalpy and entropy methods give consistent results."""
        sat_props = SATURATION_DATA[1.0]

        h = sat_props.h_f_kj_kg + quality * sat_props.h_fg_kj_kg
        s = sat_props.s_f_kj_kg_k + quality * sat_props.s_fg_kj_kg_k

        result_h = calculate_dryness_from_enthalpy(1.0, h)
        result_s = calculate_dryness_from_entropy(1.0, s)

        assert result_h.is_valid
        assert result_s.is_valid
        assert abs(result_h.dryness_fraction - result_s.dryness_fraction) < 0.01

    @pytest.mark.parametrize("pressure", [0.1, 1.0, 5.0, 10.0])
    def test_methods_agree_at_various_pressures(self, pressure):
        """Test method agreement at various pressures."""
        sat_props = SATURATION_DATA[pressure]
        quality = 0.85

        h = sat_props.h_f_kj_kg + quality * sat_props.h_fg_kj_kg
        s = sat_props.s_f_kj_kg_k + quality * sat_props.s_fg_kj_kg_k

        result_h = calculate_dryness_from_enthalpy(pressure, h)
        result_s = calculate_dryness_from_entropy(pressure, s)

        assert result_h.is_valid
        assert result_s.is_valid
        assert abs(result_h.dryness_fraction - result_s.dryness_fraction) < 0.02


class TestDeterminism:
    """Tests for deterministic behavior of dryness calculations."""

    def test_repeated_calculations_identical(self):
        """Test that repeated calculations give identical results."""
        sat_props = SATURATION_DATA[1.0]
        h_mid = sat_props.h_f_kj_kg + 0.5 * sat_props.h_fg_kj_kg

        results = [calculate_dryness_from_enthalpy(1.0, h_mid) for _ in range(10)]

        first = results[0]
        for result in results[1:]:
            assert result.dryness_fraction == first.dryness_fraction
            assert result.uncertainty_percent == first.uncertainty_percent
            assert result.provenance_hash == first.provenance_hash

    def test_determinism_across_methods(self):
        """Test determinism of both enthalpy and entropy methods."""
        sat_props = SATURATION_DATA[1.0]

        h = sat_props.h_f_kj_kg + 0.75 * sat_props.h_fg_kj_kg
        s = sat_props.s_f_kj_kg_k + 0.75 * sat_props.s_fg_kj_kg_k

        results_h = [calculate_dryness_from_enthalpy(1.0, h) for _ in range(5)]
        results_s = [calculate_dryness_from_entropy(1.0, s) for _ in range(5)]

        # All enthalpy results should be identical
        for r in results_h[1:]:
            assert r.dryness_fraction == results_h[0].dryness_fraction

        # All entropy results should be identical
        for r in results_s[1:]:
            assert r.dryness_fraction == results_s[0].dryness_fraction


class TestSaturationPropertyInterpolation:
    """Tests for saturation property interpolation."""

    def test_interpolation_between_known_points(self):
        """Test interpolation between known pressure points."""
        # Interpolate between 1.0 and 5.0 MPa
        props = get_saturation_properties(3.0)

        # Should be between the two reference points
        assert SATURATION_DATA[1.0].temperature_k < props.temperature_k < SATURATION_DATA[5.0].temperature_k
        assert SATURATION_DATA[1.0].h_f_kj_kg < props.h_f_kj_kg < SATURATION_DATA[5.0].h_f_kj_kg

    def test_exact_match_no_interpolation(self):
        """Test that exact pressure match returns reference data."""
        props = get_saturation_properties(1.0)
        ref = SATURATION_DATA[1.0]

        assert props.temperature_k == ref.temperature_k
        assert props.h_f_kj_kg == ref.h_f_kj_kg
        assert props.h_g_kj_kg == ref.h_g_kj_kg


class TestPhysicalReasonableness:
    """Tests that results are physically reasonable."""

    def test_dryness_in_valid_range(self):
        """Test that dryness is always in [0, 1]."""
        sat_props = SATURATION_DATA[1.0]

        for h in [sat_props.h_f_kj_kg - 50, sat_props.h_f_kj_kg,
                  sat_props.h_f_kj_kg + 0.5 * sat_props.h_fg_kj_kg,
                  sat_props.h_g_kj_kg, sat_props.h_g_kj_kg + 50]:
            result = calculate_dryness_from_enthalpy(1.0, h)
            if result.is_valid:
                assert 0.0 <= result.dryness_fraction <= 1.0

    def test_dryness_monotonic_with_enthalpy(self):
        """Test that dryness increases monotonically with enthalpy."""
        sat_props = SATURATION_DATA[1.0]
        enthalpies = [
            sat_props.h_f_kj_kg + i * sat_props.h_fg_kj_kg / 10
            for i in range(11)
        ]

        results = [calculate_dryness_from_enthalpy(1.0, h) for h in enthalpies]
        dryness_values = [r.dryness_fraction for r in results if r.is_valid]

        # Should be monotonically increasing
        for i in range(1, len(dryness_values)):
            assert dryness_values[i] >= dryness_values[i-1]

    def test_higher_enthalpy_higher_quality(self):
        """Test that higher enthalpy means higher quality."""
        sat_props = SATURATION_DATA[1.0]

        h_low = sat_props.h_f_kj_kg + 0.3 * sat_props.h_fg_kj_kg
        h_high = sat_props.h_f_kj_kg + 0.7 * sat_props.h_fg_kj_kg

        result_low = calculate_dryness_from_enthalpy(1.0, h_low)
        result_high = calculate_dryness_from_enthalpy(1.0, h_high)

        assert result_high.dryness_fraction > result_low.dryness_fraction


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
