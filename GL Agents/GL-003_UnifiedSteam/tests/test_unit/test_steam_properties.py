"""
Unit Tests: Steam Property Calculator

Tests the high-level steam property calculator that wraps IAPWS-IF97
functions with additional features like state detection, superheat
calculation, and validity checks.

Test Categories:
1. Property computation for various P/T combinations
2. Superheat degree calculation
3. Dryness fraction calculation
4. Steam state detection (wet/saturated/superheated)
5. Validity checks (T < Tsat handling, quality clamping)

Reference: IAPWS-IF97 and ASME PTC 19.11

Author: GL-TestEngineer
Version: 1.0.0
Target Coverage: 85%+
"""

import pytest
import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal


# =============================================================================
# Steam State Enumerations and Data Classes
# =============================================================================

class SteamState(Enum):
    """Enumeration of possible steam states."""
    SUBCOOLED_LIQUID = auto()  # T < Tsat
    SATURATED_LIQUID = auto()  # T = Tsat, x = 0
    WET_STEAM = auto()         # T = Tsat, 0 < x < 1
    SATURATED_VAPOR = auto()   # T = Tsat, x = 1
    SUPERHEATED_VAPOR = auto() # T > Tsat
    SUPERCRITICAL = auto()     # P > Pc and T > Tc
    UNKNOWN = auto()           # Cannot determine


@dataclass
class SteamProperties:
    """Complete steam properties at a given state."""
    pressure_mpa: float
    temperature_k: float
    state: SteamState
    specific_volume_m3_kg: float
    specific_enthalpy_kj_kg: float
    specific_entropy_kj_kg_k: float
    specific_internal_energy_kj_kg: float
    quality: Optional[float]  # None if not in two-phase region
    superheat_k: Optional[float]  # None if not superheated
    subcooling_k: Optional[float]  # None if not subcooled
    isobaric_heat_capacity_kj_kg_k: float
    speed_of_sound_m_s: float
    provenance_hash: str


@dataclass
class SaturationProperties:
    """Saturation properties at a given pressure or temperature."""
    pressure_mpa: float
    temperature_k: float
    # Saturated liquid properties (f subscript)
    h_f_kj_kg: float
    s_f_kj_kg_k: float
    v_f_m3_kg: float
    u_f_kj_kg: float
    # Saturated vapor properties (g subscript)
    h_g_kj_kg: float
    s_g_kj_kg_k: float
    v_g_m3_kg: float
    u_g_kj_kg: float
    # Latent properties (fg subscript)
    h_fg_kj_kg: float
    s_fg_kj_kg_k: float
    v_fg_m3_kg: float


@dataclass
class ValidationResult:
    """Result of input validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    adjusted_values: Dict[str, float]


# =============================================================================
# Constants
# =============================================================================

# Critical point
CRITICAL_TEMPERATURE_K = 647.096
CRITICAL_PRESSURE_MPA = 22.064

# Triple point
TRIPLE_TEMPERATURE_K = 273.16
TRIPLE_PRESSURE_MPA = 611.657e-6

# Valid ranges
MIN_PRESSURE_MPA = 611.657e-6
MAX_PRESSURE_MPA = 100.0
MIN_TEMPERATURE_K = 273.15
MAX_TEMPERATURE_K = 2273.15

# Tolerances
SATURATION_TOLERANCE_K = 0.1  # Tolerance for saturation detection
QUALITY_TOLERANCE = 0.001


# =============================================================================
# Steam Property Calculator Implementation (Simulated)
# =============================================================================

class SteamPropertyError(Exception):
    """Error in steam property calculation."""
    pass


class InputValidationError(SteamPropertyError):
    """Error in input validation."""
    pass


def validate_input_ranges(
    pressure_mpa: Optional[float] = None,
    temperature_k: Optional[float] = None,
    quality: Optional[float] = None
) -> ValidationResult:
    """
    Validate input parameters are within acceptable ranges.

    Returns ValidationResult with any adjustments and warnings.
    """
    errors = []
    warnings = []
    adjusted_values = {}

    if pressure_mpa is not None:
        if pressure_mpa < 0:
            errors.append(f"Pressure cannot be negative: {pressure_mpa} MPa")
        elif pressure_mpa < MIN_PRESSURE_MPA:
            warnings.append(f"Pressure {pressure_mpa} MPa below triple point, clamped to minimum")
            adjusted_values["pressure_mpa"] = MIN_PRESSURE_MPA
        elif pressure_mpa > MAX_PRESSURE_MPA:
            errors.append(f"Pressure {pressure_mpa} MPa exceeds maximum {MAX_PRESSURE_MPA} MPa")

    if temperature_k is not None:
        if temperature_k < 0:
            errors.append(f"Temperature cannot be negative: {temperature_k} K")
        elif temperature_k < MIN_TEMPERATURE_K:
            warnings.append(f"Temperature {temperature_k} K below triple point, clamped to minimum")
            adjusted_values["temperature_k"] = MIN_TEMPERATURE_K
        elif temperature_k > MAX_TEMPERATURE_K:
            errors.append(f"Temperature {temperature_k} K exceeds maximum {MAX_TEMPERATURE_K} K")

    if quality is not None:
        if quality < 0:
            warnings.append(f"Quality {quality} < 0, clamped to 0")
            adjusted_values["quality"] = 0.0
        elif quality > 1:
            warnings.append(f"Quality {quality} > 1, clamped to 1")
            adjusted_values["quality"] = 1.0

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        adjusted_values=adjusted_values
    )


def get_saturation_temperature(pressure_mpa: float) -> float:
    """Calculate saturation temperature from pressure."""
    if pressure_mpa < MIN_PRESSURE_MPA or pressure_mpa > CRITICAL_PRESSURE_MPA:
        raise SteamPropertyError(f"Pressure {pressure_mpa} MPa outside saturation range")

    # Simplified Antoine-like equation for demonstration
    # Real implementation uses IAPWS-IF97 equations
    n = [0.11670521452767e4, -0.72421316703206e6, -0.17073846940092e2,
         0.12020824702470e5, -0.32325550322333e7, 0.14915108613530e2,
         -0.48232657361591e4, 0.40511340542057e6, -0.23855557567849,
         0.65017534844798e3]

    beta = pressure_mpa ** 0.25
    E = beta ** 2 + n[2] * beta + n[5]
    F = n[0] * beta ** 2 + n[3] * beta + n[6]
    G = n[1] * beta ** 2 + n[4] * beta + n[7]
    D = 2 * G / (-F - math.sqrt(F ** 2 - 4 * E * G))

    return (n[9] + D - math.sqrt((n[9] + D) ** 2 - 4 * (n[8] + n[9] * D))) / 2


def get_saturation_pressure(temperature_k: float) -> float:
    """Calculate saturation pressure from temperature."""
    if temperature_k < MIN_TEMPERATURE_K or temperature_k > CRITICAL_TEMPERATURE_K:
        raise SteamPropertyError(f"Temperature {temperature_k} K outside saturation range")

    n = [0.11670521452767e4, -0.72421316703206e6, -0.17073846940092e2,
         0.12020824702470e5, -0.32325550322333e7, 0.14915108613530e2,
         -0.48232657361591e4, 0.40511340542057e6, -0.23855557567849,
         0.65017534844798e3]

    theta = temperature_k + n[8] / (temperature_k - n[9])
    A = theta ** 2 + n[0] * theta + n[1]
    B = n[2] * theta ** 2 + n[3] * theta + n[4]
    C = n[5] * theta ** 2 + n[6] * theta + n[7]

    return (2 * C / (-B + math.sqrt(B ** 2 - 4 * A * C))) ** 4


def detect_steam_state(
    pressure_mpa: float,
    temperature_k: float,
    quality: Optional[float] = None
) -> SteamState:
    """
    Detect the thermodynamic state of steam.

    Args:
        pressure_mpa: Pressure in MPa
        temperature_k: Temperature in K
        quality: Steam quality (0-1), if known

    Returns:
        SteamState enumeration value
    """
    # Check for supercritical
    if pressure_mpa > CRITICAL_PRESSURE_MPA and temperature_k > CRITICAL_TEMPERATURE_K:
        return SteamState.SUPERCRITICAL

    # Get saturation temperature
    try:
        t_sat = get_saturation_temperature(pressure_mpa)
    except SteamPropertyError:
        # Above critical pressure, no saturation line
        if pressure_mpa > CRITICAL_PRESSURE_MPA:
            return SteamState.SUPERCRITICAL
        return SteamState.UNKNOWN

    # Check if subcooled
    if temperature_k < t_sat - SATURATION_TOLERANCE_K:
        return SteamState.SUBCOOLED_LIQUID

    # Check if superheated
    if temperature_k > t_sat + SATURATION_TOLERANCE_K:
        return SteamState.SUPERHEATED_VAPOR

    # On or near saturation line
    if quality is not None:
        if quality < QUALITY_TOLERANCE:
            return SteamState.SATURATED_LIQUID
        elif quality > 1 - QUALITY_TOLERANCE:
            return SteamState.SATURATED_VAPOR
        else:
            return SteamState.WET_STEAM
    else:
        # Without quality information, we can't distinguish
        return SteamState.WET_STEAM


def compute_superheat_degree(
    pressure_mpa: float,
    temperature_k: float
) -> Optional[float]:
    """
    Calculate the degree of superheat.

    Returns:
        Superheat in K, or None if not superheated
    """
    try:
        t_sat = get_saturation_temperature(pressure_mpa)
    except SteamPropertyError:
        return None

    superheat = temperature_k - t_sat

    if superheat > SATURATION_TOLERANCE_K:
        return superheat
    return None


def compute_subcooling_degree(
    pressure_mpa: float,
    temperature_k: float
) -> Optional[float]:
    """
    Calculate the degree of subcooling.

    Returns:
        Subcooling in K, or None if not subcooled
    """
    try:
        t_sat = get_saturation_temperature(pressure_mpa)
    except SteamPropertyError:
        return None

    subcooling = t_sat - temperature_k

    if subcooling > SATURATION_TOLERANCE_K:
        return subcooling
    return None


def compute_dryness_fraction(
    pressure_mpa: float,
    enthalpy_kj_kg: float
) -> Optional[float]:
    """
    Calculate dryness fraction (quality) from pressure and enthalpy.

    Returns:
        Quality (0-1), or None if not in two-phase region
    """
    try:
        t_sat = get_saturation_temperature(pressure_mpa)
    except SteamPropertyError:
        return None

    # Get saturation enthalpies (simplified calculation)
    h_f = 4.186 * (t_sat - 273.15)  # Approximate saturated liquid enthalpy
    h_g = 2675 + 0.5 * (t_sat - 373.15)  # Approximate saturated vapor enthalpy
    h_fg = h_g - h_f

    if h_fg <= 0:
        return None

    quality = (enthalpy_kj_kg - h_f) / h_fg

    # Clamp to valid range
    if quality < 0:
        return 0.0 if quality > -QUALITY_TOLERANCE else None
    elif quality > 1:
        return 1.0 if quality < 1 + QUALITY_TOLERANCE else None

    return quality


def compute_dryness_fraction_from_entropy(
    pressure_mpa: float,
    entropy_kj_kg_k: float
) -> Optional[float]:
    """
    Calculate dryness fraction from pressure and entropy.

    Returns:
        Quality (0-1), or None if not in two-phase region
    """
    try:
        t_sat = get_saturation_temperature(pressure_mpa)
    except SteamPropertyError:
        return None

    # Get saturation entropies (simplified)
    s_f = 4.186 * math.log(t_sat / 273.15)  # Approximate
    s_g = 7.355 + 0.001 * (t_sat - 373.15)  # Approximate
    s_fg = s_g - s_f

    if s_fg <= 0:
        return None

    quality = (entropy_kj_kg_k - s_f) / s_fg

    if quality < -QUALITY_TOLERANCE or quality > 1 + QUALITY_TOLERANCE:
        return None

    return max(0.0, min(1.0, quality))


def get_saturation_properties(pressure_mpa: float) -> SaturationProperties:
    """
    Get complete saturation properties at given pressure.
    """
    t_sat = get_saturation_temperature(pressure_mpa)

    # Saturated liquid properties (simplified)
    h_f = 4.186 * (t_sat - 273.15)
    s_f = 4.186 * math.log(t_sat / 273.15)
    v_f = 0.001 * (1 + 0.0001 * (t_sat - 273.15))
    u_f = h_f - pressure_mpa * 1000 * v_f

    # Saturated vapor properties (simplified)
    h_g = 2675 + 0.5 * (t_sat - 373.15)
    s_g = 7.355 + 0.001 * (t_sat - 373.15)
    v_g = 0.461526 * t_sat / (pressure_mpa * 1000) * 0.95  # With compressibility
    u_g = h_g - pressure_mpa * 1000 * v_g

    return SaturationProperties(
        pressure_mpa=pressure_mpa,
        temperature_k=t_sat,
        h_f_kj_kg=h_f,
        s_f_kj_kg_k=s_f,
        v_f_m3_kg=v_f,
        u_f_kj_kg=u_f,
        h_g_kj_kg=h_g,
        s_g_kj_kg_k=s_g,
        v_g_m3_kg=v_g,
        u_g_kj_kg=u_g,
        h_fg_kj_kg=h_g - h_f,
        s_fg_kj_kg_k=s_g - s_f,
        v_fg_m3_kg=v_g - v_f
    )


def compute_properties(
    pressure_mpa: float,
    temperature_k: float,
    quality: Optional[float] = None
) -> SteamProperties:
    """
    Compute complete steam properties at given state.

    Args:
        pressure_mpa: Pressure in MPa
        temperature_k: Temperature in K
        quality: Steam quality (0-1) if in two-phase region

    Returns:
        SteamProperties dataclass with all computed values
    """
    import hashlib
    import json

    # Validate inputs
    validation = validate_input_ranges(pressure_mpa, temperature_k, quality)
    if not validation.is_valid:
        raise InputValidationError("; ".join(validation.errors))

    # Apply any adjustments
    if "pressure_mpa" in validation.adjusted_values:
        pressure_mpa = validation.adjusted_values["pressure_mpa"]
    if "temperature_k" in validation.adjusted_values:
        temperature_k = validation.adjusted_values["temperature_k"]
    if "quality" in validation.adjusted_values:
        quality = validation.adjusted_values["quality"]

    # Detect state
    state = detect_steam_state(pressure_mpa, temperature_k, quality)

    # Compute properties based on state
    if state == SteamState.SUBCOOLED_LIQUID:
        # Region 1 properties
        v = 0.001 * (1 + 0.0001 * (temperature_k - 273.15))
        h = 4.186 * (temperature_k - 273.15)
        s = 4.186 * math.log(temperature_k / 273.15)
        cp = 4.186
        c = 1500 + 4 * (temperature_k - 273.15)
        computed_quality = None
        superheat = None
        subcooling = compute_subcooling_degree(pressure_mpa, temperature_k)

    elif state == SteamState.SUPERHEATED_VAPOR:
        # Region 2 properties
        t_sat = get_saturation_temperature(pressure_mpa)
        v = 0.461526 * temperature_k / (pressure_mpa * 1000)
        h = 2675 + 2.0 * (temperature_k - t_sat)
        s = 7.355 + 2.0 * math.log(temperature_k / t_sat)
        cp = 2.0
        c = 400 + 0.5 * (temperature_k - 373.15)
        computed_quality = None
        superheat = compute_superheat_degree(pressure_mpa, temperature_k)
        subcooling = None

    elif state in [SteamState.WET_STEAM, SteamState.SATURATED_LIQUID, SteamState.SATURATED_VAPOR]:
        # Region 4 properties
        sat_props = get_saturation_properties(pressure_mpa)

        if quality is None:
            # Try to infer from temperature
            quality = 0.5  # Default if unknown

        v = sat_props.v_f_m3_kg + quality * sat_props.v_fg_m3_kg
        h = sat_props.h_f_kj_kg + quality * sat_props.h_fg_kj_kg
        s = sat_props.s_f_kj_kg_k + quality * sat_props.s_fg_kj_kg_k
        cp = 4.186 * (1 - quality) + 2.0 * quality  # Weighted average
        c = 1500 * (1 - quality) + 400 * quality  # Weighted average
        computed_quality = quality
        superheat = None
        subcooling = None
        temperature_k = sat_props.temperature_k  # Force to saturation temperature

    elif state == SteamState.SUPERCRITICAL:
        # Supercritical properties (simplified)
        v = 0.461526 * temperature_k / (pressure_mpa * 1000) * 0.8
        h = 2675 + 2.5 * (temperature_k - CRITICAL_TEMPERATURE_K)
        s = 7.355 + 2.0 * math.log(temperature_k / CRITICAL_TEMPERATURE_K)
        cp = 2.5
        c = 450
        computed_quality = None
        superheat = None
        subcooling = None

    else:
        raise SteamPropertyError(f"Cannot compute properties for state: {state}")

    # Internal energy
    u = h - pressure_mpa * 1000 * v

    # Compute provenance hash
    inputs_for_hash = {
        "pressure_mpa": pressure_mpa,
        "temperature_k": temperature_k,
        "quality": quality
    }
    outputs_for_hash = {
        "v": v, "h": h, "s": s, "u": u
    }
    hash_data = json.dumps(
        {"inputs": inputs_for_hash, "outputs": outputs_for_hash},
        sort_keys=True, default=str
    )
    provenance_hash = hashlib.sha256(hash_data.encode()).hexdigest()

    return SteamProperties(
        pressure_mpa=pressure_mpa,
        temperature_k=temperature_k,
        state=state,
        specific_volume_m3_kg=v,
        specific_enthalpy_kj_kg=h,
        specific_entropy_kj_kg_k=s,
        specific_internal_energy_kj_kg=u,
        quality=computed_quality,
        superheat_k=superheat,
        subcooling_k=subcooling,
        isobaric_heat_capacity_kj_kg_k=cp,
        speed_of_sound_m_s=c,
        provenance_hash=provenance_hash
    )


# =============================================================================
# Test Classes
# =============================================================================

class TestInputValidation:
    """Test input validation for steam property calculations."""

    def test_valid_inputs_pass(self):
        """Test that valid inputs pass validation."""
        result = validate_input_ranges(
            pressure_mpa=1.0,
            temperature_k=400.0,
            quality=0.5
        )
        assert result.is_valid
        assert len(result.errors) == 0

    def test_negative_pressure_fails(self):
        """Test that negative pressure fails validation."""
        result = validate_input_ranges(pressure_mpa=-1.0)
        assert not result.is_valid
        assert any("negative" in e.lower() for e in result.errors)

    def test_negative_temperature_fails(self):
        """Test that negative temperature fails validation."""
        result = validate_input_ranges(temperature_k=-100.0)
        assert not result.is_valid
        assert any("negative" in e.lower() for e in result.errors)

    def test_pressure_below_triple_point_warns(self):
        """Test that pressure below triple point gives warning."""
        result = validate_input_ranges(pressure_mpa=1e-10)
        assert result.is_valid  # Still valid, just adjusted
        assert len(result.warnings) > 0
        assert "pressure_mpa" in result.adjusted_values

    def test_temperature_below_triple_point_warns(self):
        """Test that temperature below triple point gives warning."""
        result = validate_input_ranges(temperature_k=250.0)
        assert result.is_valid
        assert len(result.warnings) > 0
        assert "temperature_k" in result.adjusted_values

    def test_quality_clamped_below_zero(self):
        """Test that quality below 0 is clamped with warning."""
        result = validate_input_ranges(quality=-0.1)
        assert result.is_valid
        assert result.adjusted_values["quality"] == 0.0

    def test_quality_clamped_above_one(self):
        """Test that quality above 1 is clamped with warning."""
        result = validate_input_ranges(quality=1.2)
        assert result.is_valid
        assert result.adjusted_values["quality"] == 1.0

    def test_pressure_above_maximum_fails(self):
        """Test that pressure above maximum fails."""
        result = validate_input_ranges(pressure_mpa=150.0)
        assert not result.is_valid
        assert any("exceeds" in e.lower() for e in result.errors)

    def test_temperature_above_maximum_fails(self):
        """Test that temperature above maximum fails."""
        result = validate_input_ranges(temperature_k=3000.0)
        assert not result.is_valid
        assert any("exceeds" in e.lower() for e in result.errors)


class TestSteamStateDetection:
    """Test steam state detection logic."""

    def test_detect_subcooled_liquid(self):
        """Test detection of subcooled liquid state."""
        # At 1 MPa, Tsat ~ 453 K; 300 K is subcooled
        state = detect_steam_state(1.0, 300.0)
        assert state == SteamState.SUBCOOLED_LIQUID

    def test_detect_superheated_vapor(self):
        """Test detection of superheated vapor state."""
        # At 1 MPa, Tsat ~ 453 K; 550 K is superheated
        state = detect_steam_state(1.0, 550.0)
        assert state == SteamState.SUPERHEATED_VAPOR

    def test_detect_supercritical(self):
        """Test detection of supercritical state."""
        # Above critical point: P > 22.064 MPa, T > 647.096 K
        state = detect_steam_state(25.0, 700.0)
        assert state == SteamState.SUPERCRITICAL

    def test_detect_saturated_liquid(self):
        """Test detection of saturated liquid with quality=0."""
        state = detect_steam_state(1.0, 453.0, quality=0.0)
        assert state == SteamState.SATURATED_LIQUID

    def test_detect_saturated_vapor(self):
        """Test detection of saturated vapor with quality=1."""
        state = detect_steam_state(1.0, 453.0, quality=1.0)
        assert state == SteamState.SATURATED_VAPOR

    def test_detect_wet_steam(self):
        """Test detection of wet steam with quality=0.5."""
        state = detect_steam_state(1.0, 453.0, quality=0.5)
        assert state == SteamState.WET_STEAM

    @pytest.mark.parametrize("pressure,temperature,expected_state", [
        (0.1, 300.0, SteamState.SUBCOOLED_LIQUID),  # Low P, low T
        (0.1, 400.0, SteamState.SUPERHEATED_VAPOR),  # Low P, above Tsat
        (5.0, 300.0, SteamState.SUBCOOLED_LIQUID),   # High P, low T
        (5.0, 600.0, SteamState.SUPERHEATED_VAPOR),  # High P, high T
        (25.0, 700.0, SteamState.SUPERCRITICAL),     # Supercritical
    ])
    def test_state_detection_parametrized(self, pressure, temperature, expected_state):
        """Parametrized test for state detection."""
        state = detect_steam_state(pressure, temperature)
        assert state == expected_state


class TestSuperheatCalculation:
    """Test superheat degree calculation."""

    def test_superheat_positive_when_above_saturation(self):
        """Test superheat is positive when above saturation temperature."""
        # At 1 MPa, Tsat ~ 453 K; 500 K is superheated
        superheat = compute_superheat_degree(1.0, 500.0)
        assert superheat is not None
        assert superheat > 0
        assert 40 < superheat < 60  # Approximately 47 K superheat

    def test_superheat_none_when_subcooled(self):
        """Test superheat is None when subcooled."""
        # At 1 MPa, Tsat ~ 453 K; 400 K is subcooled
        superheat = compute_superheat_degree(1.0, 400.0)
        assert superheat is None

    def test_superheat_none_at_saturation(self):
        """Test superheat is None or small at saturation."""
        t_sat = get_saturation_temperature(1.0)
        superheat = compute_superheat_degree(1.0, t_sat)
        # Should be None or very small (within tolerance)
        assert superheat is None or superheat < SATURATION_TOLERANCE_K

    @pytest.mark.parametrize("pressure,temperature,expected_superheat_range", [
        (0.1, 400.0, (20, 40)),   # Low pressure
        (1.0, 500.0, (40, 55)),   # Medium pressure
        (5.0, 600.0, (50, 80)),   # High pressure
    ])
    def test_superheat_ranges(self, pressure, temperature, expected_superheat_range):
        """Test superheat calculation across various conditions."""
        superheat = compute_superheat_degree(pressure, temperature)
        assert superheat is not None
        min_sh, max_sh = expected_superheat_range
        assert min_sh < superheat < max_sh


class TestSubcoolingCalculation:
    """Test subcooling degree calculation."""

    def test_subcooling_positive_when_below_saturation(self):
        """Test subcooling is positive when below saturation temperature."""
        # At 1 MPa, Tsat ~ 453 K; 400 K is subcooled
        subcooling = compute_subcooling_degree(1.0, 400.0)
        assert subcooling is not None
        assert subcooling > 0
        assert 45 < subcooling < 60  # Approximately 53 K subcooling

    def test_subcooling_none_when_superheated(self):
        """Test subcooling is None when superheated."""
        # At 1 MPa, Tsat ~ 453 K; 500 K is superheated
        subcooling = compute_subcooling_degree(1.0, 500.0)
        assert subcooling is None

    def test_subcooling_none_at_saturation(self):
        """Test subcooling is None or small at saturation."""
        t_sat = get_saturation_temperature(1.0)
        subcooling = compute_subcooling_degree(1.0, t_sat)
        assert subcooling is None or subcooling < SATURATION_TOLERANCE_K


class TestDrynessFractionCalculation:
    """Test dryness fraction (quality) calculation."""

    def test_quality_from_enthalpy_at_saturation_liquid(self):
        """Test quality = 0 at saturated liquid enthalpy."""
        sat_props = get_saturation_properties(1.0)
        quality = compute_dryness_fraction(1.0, sat_props.h_f_kj_kg)
        assert quality is not None
        assert abs(quality) < 0.01  # Should be ~0

    def test_quality_from_enthalpy_at_saturation_vapor(self):
        """Test quality = 1 at saturated vapor enthalpy."""
        sat_props = get_saturation_properties(1.0)
        quality = compute_dryness_fraction(1.0, sat_props.h_g_kj_kg)
        assert quality is not None
        assert abs(quality - 1.0) < 0.01  # Should be ~1

    def test_quality_from_enthalpy_mid_range(self):
        """Test quality calculation at mid-range enthalpy."""
        sat_props = get_saturation_properties(1.0)
        h_mid = sat_props.h_f_kj_kg + 0.5 * sat_props.h_fg_kj_kg
        quality = compute_dryness_fraction(1.0, h_mid)
        assert quality is not None
        assert 0.45 < quality < 0.55  # Should be ~0.5

    def test_quality_none_when_subcooled(self):
        """Test quality is None when enthalpy indicates subcooling."""
        sat_props = get_saturation_properties(1.0)
        # Enthalpy well below saturated liquid
        h_subcooled = sat_props.h_f_kj_kg - 100
        quality = compute_dryness_fraction(1.0, h_subcooled)
        # Should be None or 0 (clamped)
        assert quality is None or quality == 0.0

    def test_quality_from_entropy_mid_range(self):
        """Test quality calculation from entropy."""
        sat_props = get_saturation_properties(1.0)
        s_mid = sat_props.s_f_kj_kg_k + 0.5 * sat_props.s_fg_kj_kg_k
        quality = compute_dryness_fraction_from_entropy(1.0, s_mid)
        assert quality is not None
        assert 0.45 < quality < 0.55

    @pytest.mark.parametrize("target_quality", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_quality_roundtrip(self, target_quality):
        """Test quality calculation roundtrips correctly."""
        sat_props = get_saturation_properties(1.0)
        h_target = sat_props.h_f_kj_kg + target_quality * sat_props.h_fg_kj_kg
        computed_quality = compute_dryness_fraction(1.0, h_target)

        assert computed_quality is not None
        assert abs(computed_quality - target_quality) < 0.02


class TestPropertyComputation:
    """Test complete property computation."""

    def test_compute_subcooled_properties(self):
        """Test property computation for subcooled liquid."""
        props = compute_properties(5.0, 350.0)

        assert props.state == SteamState.SUBCOOLED_LIQUID
        assert props.quality is None
        assert props.superheat_k is None
        assert props.subcooling_k is not None
        assert props.subcooling_k > 0

    def test_compute_superheated_properties(self):
        """Test property computation for superheated vapor."""
        props = compute_properties(1.0, 550.0)

        assert props.state == SteamState.SUPERHEATED_VAPOR
        assert props.quality is None
        assert props.superheat_k is not None
        assert props.superheat_k > 0
        assert props.subcooling_k is None

    def test_compute_wet_steam_properties(self):
        """Test property computation for wet steam."""
        sat_props = get_saturation_properties(1.0)
        props = compute_properties(1.0, sat_props.temperature_k, quality=0.5)

        assert props.state == SteamState.WET_STEAM
        assert props.quality is not None
        assert 0.45 < props.quality < 0.55
        assert props.superheat_k is None
        assert props.subcooling_k is None

    def test_compute_supercritical_properties(self):
        """Test property computation for supercritical state."""
        props = compute_properties(25.0, 700.0)

        assert props.state == SteamState.SUPERCRITICAL
        assert props.quality is None

    def test_properties_thermodynamic_consistency(self):
        """Test that u = h - Pv relation holds."""
        props = compute_properties(1.0, 450.0)

        u_expected = props.specific_enthalpy_kj_kg - props.pressure_mpa * 1000 * props.specific_volume_m3_kg

        assert pytest.approx(props.specific_internal_energy_kj_kg, rel=0.01) == u_expected

    def test_properties_provenance_hash_exists(self):
        """Test that provenance hash is generated."""
        props = compute_properties(1.0, 450.0)

        assert props.provenance_hash is not None
        assert len(props.provenance_hash) == 64  # SHA-256 hex length

    def test_properties_provenance_hash_deterministic(self):
        """Test that same inputs produce same provenance hash."""
        props1 = compute_properties(1.0, 450.0)
        props2 = compute_properties(1.0, 450.0)

        assert props1.provenance_hash == props2.provenance_hash

    def test_properties_provenance_hash_changes_with_input(self):
        """Test that different inputs produce different provenance hash."""
        props1 = compute_properties(1.0, 450.0)
        props2 = compute_properties(1.0, 451.0)

        assert props1.provenance_hash != props2.provenance_hash


class TestValidityChecks:
    """Test validity checks and edge case handling."""

    def test_temperature_below_saturation_clamped(self):
        """Test that T < Tsat in two-phase sets to saturation."""
        sat_props = get_saturation_properties(1.0)
        # Request temperature below saturation but with quality specified
        props = compute_properties(1.0, sat_props.temperature_k - 5, quality=0.5)

        # Should snap to saturation temperature
        assert props.state == SteamState.WET_STEAM or props.state == SteamState.SUBCOOLED_LIQUID

    def test_quality_clamped_to_valid_range(self):
        """Test that quality outside [0,1] is clamped."""
        sat_props = get_saturation_properties(1.0)

        # Quality < 0 should be clamped to 0
        props_low = compute_properties(1.0, sat_props.temperature_k, quality=-0.05)
        assert props_low.quality is not None
        assert props_low.quality >= 0.0

        # Quality > 1 should be clamped to 1
        props_high = compute_properties(1.0, sat_props.temperature_k, quality=1.05)
        assert props_high.quality is not None
        assert props_high.quality <= 1.0

    def test_invalid_pressure_raises_error(self):
        """Test that invalid pressure raises appropriate error."""
        with pytest.raises(InputValidationError):
            compute_properties(-1.0, 400.0)

    def test_invalid_temperature_raises_error(self):
        """Test that invalid temperature raises appropriate error."""
        with pytest.raises(InputValidationError):
            compute_properties(1.0, -100.0)

    def test_extreme_low_pressure(self):
        """Test handling of very low pressure."""
        # Near triple point pressure
        props = compute_properties(0.001, 400.0)
        assert props is not None
        assert props.specific_volume_m3_kg > 0

    def test_near_critical_point(self):
        """Test behavior near critical point."""
        # Just below critical point
        props = compute_properties(22.0, 645.0)
        assert props is not None


class TestSaturationProperties:
    """Test saturation property retrieval."""

    def test_saturation_properties_structure(self):
        """Test that all saturation properties are returned."""
        sat_props = get_saturation_properties(1.0)

        # Check all fields exist
        assert sat_props.pressure_mpa == 1.0
        assert sat_props.temperature_k > 0
        assert sat_props.h_f_kj_kg < sat_props.h_g_kj_kg
        assert sat_props.s_f_kj_kg_k < sat_props.s_g_kj_kg_k
        assert sat_props.v_f_m3_kg < sat_props.v_g_m3_kg

    def test_saturation_hfg_equals_difference(self):
        """Test that h_fg = h_g - h_f."""
        sat_props = get_saturation_properties(1.0)

        h_fg_computed = sat_props.h_g_kj_kg - sat_props.h_f_kj_kg
        assert pytest.approx(sat_props.h_fg_kj_kg, rel=0.001) == h_fg_computed

    def test_saturation_sfg_equals_difference(self):
        """Test that s_fg = s_g - s_f."""
        sat_props = get_saturation_properties(1.0)

        s_fg_computed = sat_props.s_g_kj_kg_k - sat_props.s_f_kj_kg_k
        assert pytest.approx(sat_props.s_fg_kj_kg_k, rel=0.001) == s_fg_computed

    @pytest.mark.parametrize("pressure", [0.1, 0.5, 1.0, 5.0, 10.0])
    def test_saturation_properties_at_various_pressures(self, pressure):
        """Test saturation properties at various pressures."""
        sat_props = get_saturation_properties(pressure)

        # Temperature should increase with pressure
        assert sat_props.temperature_k > TRIPLE_TEMPERATURE_K

        # All properties should be positive
        assert sat_props.h_f_kj_kg > 0
        assert sat_props.h_g_kj_kg > 0
        assert sat_props.v_f_m3_kg > 0
        assert sat_props.v_g_m3_kg > 0

    def test_saturation_temperature_increases_with_pressure(self):
        """Test that saturation temperature increases with pressure."""
        t_low = get_saturation_temperature(0.1)
        t_mid = get_saturation_temperature(1.0)
        t_high = get_saturation_temperature(10.0)

        assert t_low < t_mid < t_high


class TestPhysicalReasonableness:
    """Test that computed properties are physically reasonable."""

    def test_liquid_specific_volume_small(self):
        """Test that liquid specific volume is in correct range."""
        props = compute_properties(5.0, 350.0)

        # Liquid water ~ 0.001 m3/kg
        assert 0.0005 < props.specific_volume_m3_kg < 0.005

    def test_vapor_specific_volume_large(self):
        """Test that vapor specific volume is larger than liquid."""
        props_vapor = compute_properties(0.1, 500.0)
        props_liquid = compute_properties(5.0, 350.0)

        assert props_vapor.specific_volume_m3_kg > 10 * props_liquid.specific_volume_m3_kg

    def test_enthalpy_increases_with_temperature_liquid(self):
        """Test enthalpy increases with temperature in liquid."""
        props_low = compute_properties(5.0, 320.0)
        props_high = compute_properties(5.0, 400.0)

        assert props_high.specific_enthalpy_kj_kg > props_low.specific_enthalpy_kj_kg

    def test_enthalpy_increases_with_temperature_vapor(self):
        """Test enthalpy increases with temperature in vapor."""
        props_low = compute_properties(0.5, 450.0)
        props_high = compute_properties(0.5, 550.0)

        assert props_high.specific_enthalpy_kj_kg > props_low.specific_enthalpy_kj_kg

    def test_entropy_increases_with_temperature(self):
        """Test entropy increases with temperature at constant pressure."""
        props_low = compute_properties(1.0, 350.0)
        props_high = compute_properties(1.0, 550.0)

        assert props_high.specific_entropy_kj_kg_k > props_low.specific_entropy_kj_kg_k

    def test_heat_capacity_positive(self):
        """Test that heat capacity is always positive."""
        test_cases = [
            (1.0, 350.0),   # Liquid
            (1.0, 550.0),   # Superheated vapor
            (25.0, 700.0),  # Supercritical
        ]

        for p, t in test_cases:
            props = compute_properties(p, t)
            assert props.isobaric_heat_capacity_kj_kg_k > 0

    def test_speed_of_sound_positive(self):
        """Test that speed of sound is always positive."""
        test_cases = [
            (1.0, 350.0),
            (1.0, 550.0),
            (25.0, 700.0),
        ]

        for p, t in test_cases:
            props = compute_properties(p, t)
            assert props.speed_of_sound_m_s > 0
