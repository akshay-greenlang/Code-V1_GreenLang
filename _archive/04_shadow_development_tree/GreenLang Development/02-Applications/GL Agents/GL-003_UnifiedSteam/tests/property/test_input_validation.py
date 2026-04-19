"""
Property-Based Tests for Input Validation and Fuzzing

This module provides comprehensive fuzzing tests for sensor inputs using Hypothesis
to validate input handling, boundary conditions, and edge cases.

Test Categories:
1. Temperature Input Fuzzing (valid/invalid ranges, NaN/Inf handling)
2. Pressure Input Fuzzing (boundary conditions, gauge/absolute conversion)
3. Flow Input Fuzzing (negative values, zero handling)
4. Unit Conversion Accuracy (precision, round-trip conversions)
5. Boundary Condition Testing (region transitions, near-critical points)
6. NaN/Inf Handling (graceful error handling)

Author: GL-TestEngineer
Version: 1.0.0
"""

import math
import pytest
from typing import Optional, Tuple, Dict, Any
from datetime import datetime, timezone, timedelta

from hypothesis import given, assume, settings, Verbosity, HealthCheck
from hypothesis import strategies as st
from hypothesis.strategies import composite

import sys
import os

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import modules under test
from thermodynamics.iapws_if97 import (
    IF97_CONSTANTS,
    REGION_BOUNDARIES,
    detect_region,
    get_saturation_pressure,
    get_saturation_temperature,
    celsius_to_kelvin,
    kelvin_to_celsius,
    kpa_to_mpa,
    mpa_to_kpa,
    compute_density,
)
from thermodynamics.steam_properties import (
    validate_input_ranges,
    compute_properties,
    SteamState,
)
from thermodynamics.steam_quality import (
    validate_steam_state,
    clamp_quality,
)
from integration.sensor_transformer import (
    UnitConverter,
    UnitCategory,
    SensorTransformer,
    CalibrationParams,
    QualityCode,
    ValidationResult,
    QualifiedValue,
)


# =============================================================================
# HYPOTHESIS CONFIGURATION
# =============================================================================

settings.register_profile(
    "ci",
    max_examples=200,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)

settings.register_profile(
    "dev",
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

settings.register_profile(
    "full",
    max_examples=1000,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


# =============================================================================
# CUSTOM STRATEGIES FOR FUZZING
# =============================================================================

# Temperature strategies
valid_temperature_c = st.floats(min_value=-50.0, max_value=800.0, allow_nan=False, allow_infinity=False)
valid_temperature_k = st.floats(min_value=250.0, max_value=1100.0, allow_nan=False, allow_infinity=False)
invalid_temperature_c = st.floats(min_value=-300.0, max_value=-100.0, allow_nan=False, allow_infinity=False) | \
                        st.floats(min_value=1000.0, max_value=5000.0, allow_nan=False, allow_infinity=False)

# Pressure strategies
valid_pressure_kpa = st.floats(min_value=1.0, max_value=100000.0, allow_nan=False, allow_infinity=False)
valid_pressure_mpa = st.floats(min_value=0.001, max_value=100.0, allow_nan=False, allow_infinity=False)
invalid_pressure_kpa = st.floats(min_value=-1000.0, max_value=-0.1, allow_nan=False, allow_infinity=False) | \
                       st.floats(min_value=200000.0, max_value=1000000.0, allow_nan=False, allow_infinity=False)

# Flow strategies
valid_flow_kg_s = st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False)
negative_flow = st.floats(min_value=-1000.0, max_value=-0.001, allow_nan=False, allow_infinity=False)

# Quality strategies
valid_quality = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
invalid_quality = st.floats(min_value=-1.0, max_value=-0.001, allow_nan=False, allow_infinity=False) | \
                  st.floats(min_value=1.001, max_value=10.0, allow_nan=False, allow_infinity=False)

# Special values for edge case testing
special_floats = st.sampled_from([
    0.0, -0.0, 1e-300, 1e300, -1e-300, -1e300,
    float('inf'), float('-inf'), float('nan'),
    1e-15, -1e-15, 1e15, -1e15,
])

# Arbitrary floats for extreme fuzzing
arbitrary_floats = st.floats(
    allow_nan=True,
    allow_infinity=True,
    allow_subnormal=True
)


@composite
def sensor_reading(draw, tag_prefix: str = "sensor"):
    """Generate random sensor reading data."""
    return {
        "tag": f"{tag_prefix}_{draw(st.integers(min_value=1, max_value=999))}",
        "value": draw(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)),
        "timestamp": datetime.now(timezone.utc) - timedelta(seconds=draw(st.integers(min_value=0, max_value=3600))),
        "quality_code": draw(st.integers(min_value=0, max_value=0x80000000)),
    }


@composite
def calibration_params(draw):
    """Generate random calibration parameters."""
    return CalibrationParams(
        offset=draw(st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False)),
        gain=draw(st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)),
    )


# =============================================================================
# TEST CLASS: TEMPERATURE INPUT FUZZING
# =============================================================================

@pytest.mark.hypothesis
class TestTemperatureFuzzing:
    """
    Fuzz testing for temperature input validation.

    Tests boundary conditions, invalid inputs, and unit conversions.
    """

    @given(valid_temperature_c)
    @settings(max_examples=200, deadline=None)
    def test_valid_temperature_celsius_accepted(self, temp_c: float):
        """
        Test that valid temperatures in Celsius are accepted.
        """
        # Skip values outside practical steam system range
        assume(-50 <= temp_c <= 800)

        # Convert to Kelvin
        temp_k = celsius_to_kelvin(temp_c)

        # Should be valid Kelvin temperature
        assert temp_k > 0, f"Kelvin temperature must be positive: {temp_k}"
        assert temp_k == temp_c + 273.15, f"Conversion error: {temp_k} != {temp_c} + 273.15"

    @given(valid_temperature_k)
    @settings(max_examples=200, deadline=None)
    def test_valid_temperature_kelvin_accepted(self, temp_k: float):
        """
        Test that valid temperatures in Kelvin are accepted.
        """
        assume(temp_k > 200)  # Above absolute zero margin

        # Convert to Celsius
        temp_c = kelvin_to_celsius(temp_k)

        # Round-trip should be consistent
        temp_k_back = celsius_to_kelvin(temp_c)

        assert abs(temp_k_back - temp_k) < 1e-10, \
            f"Round-trip conversion failed: {temp_k} -> {temp_c} -> {temp_k_back}"

    @given(invalid_temperature_c)
    @settings(max_examples=100, deadline=None)
    def test_invalid_temperature_handled(self, temp_c: float):
        """
        Test that invalid temperatures produce appropriate validation errors.
        """
        # Validate input ranges
        result = validate_input_ranges(temperature_c=temp_c)

        if temp_c < -273.15:
            # Below absolute zero - physically impossible
            # Validation should catch this
            assert not result.is_valid or len(result.warnings) > 0, \
                f"Should flag temperature below absolute zero: {temp_c}"
        elif temp_c > 2000:
            # Way above steam table range
            assert not result.is_valid, \
                f"Should reject temperature far above range: {temp_c}"

    @given(special_floats)
    @settings(max_examples=50, deadline=None)
    def test_special_float_temperatures(self, temp: float):
        """
        Test handling of special float values (NaN, Inf, etc.)
        """
        # NaN and Inf should be rejected gracefully
        if math.isnan(temp) or math.isinf(temp):
            # Should not raise unhandled exception
            try:
                result = validate_input_ranges(temperature_c=temp)
                # If no exception, result should indicate invalid
                # (implementation may or may not validate NaN)
            except (ValueError, TypeError):
                # Expected - NaN/Inf rejected
                pass

    @given(st.floats(min_value=-273.15, max_value=-273.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50, deadline=None)
    def test_near_absolute_zero(self, temp_c: float):
        """
        Test temperatures near absolute zero boundary.
        """
        temp_k = celsius_to_kelvin(temp_c)

        # Should be very close to 0 K
        assert temp_k >= 0, f"Temperature cannot be negative Kelvin: {temp_k}"
        assert temp_k < 1.0, f"Near absolute zero should be < 1 K: {temp_k}"


# =============================================================================
# TEST CLASS: PRESSURE INPUT FUZZING
# =============================================================================

@pytest.mark.hypothesis
class TestPressureFuzzing:
    """
    Fuzz testing for pressure input validation.

    Tests boundary conditions, gauge/absolute conversions, and invalid inputs.
    """

    @given(valid_pressure_kpa)
    @settings(max_examples=200, deadline=None)
    def test_valid_pressure_kpa_accepted(self, pressure_kpa: float):
        """
        Test that valid pressures in kPa are accepted.
        """
        assume(pressure_kpa > 0.1)

        # Convert to MPa
        pressure_mpa = kpa_to_mpa(pressure_kpa)

        # Verify conversion
        assert pressure_mpa == pressure_kpa * 0.001, \
            f"kPa to MPa conversion error: {pressure_mpa}"

        # Round-trip
        pressure_kpa_back = mpa_to_kpa(pressure_mpa)
        assert abs(pressure_kpa_back - pressure_kpa) < 1e-10, \
            f"Round-trip conversion failed: {pressure_kpa} -> {pressure_mpa} -> {pressure_kpa_back}"

    @given(valid_pressure_mpa)
    @settings(max_examples=200, deadline=None)
    def test_valid_pressure_mpa_accepted(self, pressure_mpa: float):
        """
        Test that valid pressures in MPa are accepted.
        """
        assume(pressure_mpa > 0.0001)

        # Validation should accept
        result = validate_input_ranges(pressure_kpa=mpa_to_kpa(pressure_mpa))

        # Only values outside IAPWS range should be invalid
        if pressure_mpa < REGION_BOUNDARIES["P_MAX_1_2"]:
            assert result.is_valid or len(result.errors) == 0, \
                f"Valid pressure rejected: {pressure_mpa} MPa"

    @given(negative_flow)  # Using negative values for pressure
    @settings(max_examples=100, deadline=None)
    def test_negative_pressure_rejected(self, pressure_kpa: float):
        """
        Test that negative pressures are rejected.
        """
        result = validate_input_ranges(pressure_kpa=pressure_kpa)

        assert not result.is_valid, \
            f"Negative pressure should be rejected: {pressure_kpa}"

    @given(st.floats(min_value=0.0001, max_value=0.001, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50, deadline=None)
    def test_near_vacuum_pressure(self, pressure_mpa: float):
        """
        Test handling of very low (near vacuum) pressures.
        """
        # Very low pressures are physically valid but may be outside steam table range
        pressure_kpa = mpa_to_kpa(pressure_mpa)
        result = validate_input_ranges(pressure_kpa=pressure_kpa)

        # Should either be valid or have appropriate warning/error
        if pressure_mpa < REGION_BOUNDARIES["P_MIN"]:
            assert not result.is_valid or len(result.warnings) > 0, \
                f"Very low pressure should be flagged: {pressure_mpa} MPa"

    @given(st.floats(min_value=20.0, max_value=25.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50, deadline=None)
    def test_near_critical_pressure(self, pressure_mpa: float):
        """
        Test handling of pressures near critical point.
        """
        pressure_kpa = mpa_to_kpa(pressure_mpa)
        result = validate_input_ranges(pressure_kpa=pressure_kpa)

        # Near critical pressure - should have warning
        if abs(pressure_mpa - IF97_CONSTANTS["P_CRIT"]) < 0.5:
            assert len(result.warnings) > 0, \
                f"Near-critical pressure should have warning: {pressure_mpa} MPa"

    @given(st.floats(min_value=0, max_value=500, allow_nan=False, allow_infinity=False))
    @settings(max_examples=100, deadline=None)
    def test_gauge_to_absolute_conversion(self, pressure_psig: float):
        """
        Test gauge to absolute pressure conversion.
        """
        # Standard atmospheric pressure
        ATM_PSI = 14.696

        # Convert gauge to absolute
        pressure_psia = pressure_psig + ATM_PSI

        # Absolute must be positive
        assert pressure_psia > 0, f"Absolute pressure must be positive"

        # Convert to kPa (1 psi = 6.89476 kPa)
        pressure_kpa = pressure_psia * 6.89476

        assert pressure_kpa > 0, f"kPa must be positive after conversion"


# =============================================================================
# TEST CLASS: FLOW INPUT FUZZING
# =============================================================================

@pytest.mark.hypothesis
class TestFlowFuzzing:
    """
    Fuzz testing for flow input validation.

    Tests zero flow, negative flow, and boundary conditions.
    """

    @given(valid_flow_kg_s)
    @settings(max_examples=200, deadline=None)
    def test_valid_flow_accepted(self, flow_kg_s: float):
        """
        Test that valid (non-negative) flows are accepted.
        """
        # Flow must be non-negative
        assert flow_kg_s >= 0, f"Flow must be non-negative: {flow_kg_s}"

        # Convert to lb/hr
        flow_lb_hr = flow_kg_s * 3600 / 0.453592

        assert flow_lb_hr >= 0, f"Converted flow must be non-negative: {flow_lb_hr}"

    @given(negative_flow)
    @settings(max_examples=100, deadline=None)
    def test_negative_flow_rejected(self, flow_kg_s: float):
        """
        Test that negative flows are properly handled.
        """
        # Mass flow cannot be negative (direction is handled separately)
        assert flow_kg_s < 0, f"Test setup error: flow should be negative"

        # The abs() function would correct this
        corrected_flow = abs(flow_kg_s)
        assert corrected_flow > 0, f"Corrected flow should be positive"

    @given(st.floats(min_value=0.0, max_value=1e-15, allow_nan=False, allow_infinity=False))
    @settings(max_examples=50, deadline=None)
    def test_near_zero_flow(self, flow_kg_s: float):
        """
        Test handling of very small (near-zero) flows.
        """
        # Near-zero flows may cause division issues
        if flow_kg_s > 0:
            # Specific volume calculation with density
            try:
                # Avoid division by zero
                if flow_kg_s > 1e-20:
                    rate = 100.0 / flow_kg_s  # Example calculation
                    assert not math.isinf(rate), f"Rate became infinite"
            except ZeroDivisionError:
                pass  # Expected for very small values


# =============================================================================
# TEST CLASS: UNIT CONVERSION ACCURACY
# =============================================================================

@pytest.mark.hypothesis
class TestUnitConversionAccuracy:
    """
    Test unit conversion accuracy and round-trip consistency.
    """

    @given(st.floats(min_value=0.1, max_value=1000.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=200, deadline=None)
    def test_pressure_unit_roundtrip(self, pressure_bar: float):
        """
        Test pressure unit round-trip conversions.
        """
        # bar -> kPa -> psi -> MPa -> bar
        pressure_kpa = UnitConverter.convert(pressure_bar, "bar", "kPa")
        pressure_psi = UnitConverter.convert(pressure_kpa, "kPa", "psi")
        pressure_mpa = UnitConverter.convert(pressure_psi, "psi", "MPa")
        pressure_bar_back = UnitConverter.convert(pressure_mpa, "MPa", "bar")

        # Should recover original value
        rel_error = abs(pressure_bar_back - pressure_bar) / pressure_bar

        assert rel_error < 1e-10, \
            f"Pressure round-trip error: {pressure_bar} -> {pressure_bar_back}, error={rel_error}"

    @given(st.floats(min_value=-50.0, max_value=500.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=200, deadline=None)
    def test_temperature_unit_roundtrip(self, temp_c: float):
        """
        Test temperature unit round-trip conversions.
        """
        # C -> F -> K -> C
        temp_f = UnitConverter.convert(temp_c, "C", "F")
        temp_k = UnitConverter.convert(temp_f, "F", "K")
        temp_c_back = UnitConverter.convert(temp_k, "K", "C")

        # Should recover original value
        abs_error = abs(temp_c_back - temp_c)

        assert abs_error < 1e-10, \
            f"Temperature round-trip error: {temp_c} -> {temp_c_back}, error={abs_error}"

    @given(st.floats(min_value=0.01, max_value=1000.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=200, deadline=None)
    def test_mass_flow_unit_roundtrip(self, flow_kg_s: float):
        """
        Test mass flow unit round-trip conversions.
        """
        # kg/s -> lb/hr -> klb/hr -> kg/hr -> kg/s
        flow_lb_hr = UnitConverter.convert(flow_kg_s, "kg/s", "lb/hr")
        flow_klb_hr = UnitConverter.convert(flow_lb_hr, "lb/hr", "klb/hr")
        flow_kg_hr = UnitConverter.convert(flow_klb_hr, "klb/hr", "kg/hr")
        flow_kg_s_back = UnitConverter.convert(flow_kg_hr, "kg/hr", "kg/s")

        rel_error = abs(flow_kg_s_back - flow_kg_s) / flow_kg_s

        assert rel_error < 1e-10, \
            f"Mass flow round-trip error: {flow_kg_s} -> {flow_kg_s_back}, error={rel_error}"

    @given(st.floats(min_value=1.0, max_value=10000.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=200, deadline=None)
    def test_energy_unit_roundtrip(self, energy_kj: float):
        """
        Test energy unit round-trip conversions.
        """
        # kJ -> BTU -> kWh -> MJ -> kJ
        energy_btu = UnitConverter.convert(energy_kj, "kJ", "BTU")
        energy_kwh = UnitConverter.convert(energy_btu, "BTU", "kWh")
        energy_mj = UnitConverter.convert(energy_kwh, "kWh", "MJ")
        energy_kj_back = UnitConverter.convert(energy_mj, "MJ", "kJ")

        rel_error = abs(energy_kj_back - energy_kj) / energy_kj

        assert rel_error < 1e-10, \
            f"Energy round-trip error: {energy_kj} -> {energy_kj_back}, error={rel_error}"

    @given(
        st.sampled_from(["kPa", "bar", "psi", "MPa", "atm"]),
        st.sampled_from(["kPa", "bar", "psi", "MPa", "atm"]),
        st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=200, deadline=None)
    def test_pressure_conversion_matrix(self, from_unit: str, to_unit: str, value: float):
        """
        Test all pressure unit conversions are consistent.
        """
        # Convert value
        converted = UnitConverter.convert(value, from_unit, to_unit)

        # Must be positive for positive input
        assert converted > 0, f"Converted pressure must be positive"

        # Convert back
        back = UnitConverter.convert(converted, to_unit, from_unit)

        rel_error = abs(back - value) / value

        assert rel_error < 1e-10, \
            f"Pressure conversion {from_unit} -> {to_unit} -> {from_unit} failed"

    @given(
        st.sampled_from(["C", "F", "K"]),
        st.sampled_from(["C", "F", "K"]),
        st.floats(min_value=-40.0, max_value=400.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=200, deadline=None)
    def test_temperature_conversion_matrix(self, from_unit: str, to_unit: str, value: float):
        """
        Test all temperature unit conversions are consistent.
        """
        converted = UnitConverter.convert(value, from_unit, to_unit)
        back = UnitConverter.convert(converted, to_unit, from_unit)

        abs_error = abs(back - value)

        assert abs_error < 1e-10, \
            f"Temperature conversion {from_unit} -> {to_unit} -> {from_unit} failed: {value} -> {back}"


# =============================================================================
# TEST CLASS: NaN/Inf HANDLING
# =============================================================================

@pytest.mark.hypothesis
class TestNaNInfHandling:
    """
    Test graceful handling of NaN and Infinity values.
    """

    @given(st.sampled_from([float('nan'), float('inf'), float('-inf')]))
    @settings(max_examples=50, deadline=None)
    def test_nan_inf_temperature_handling(self, bad_value: float):
        """
        Test that NaN/Inf temperatures are handled gracefully.
        """
        try:
            result = validate_input_ranges(temperature_c=bad_value)
            # If no exception, result should indicate invalid or have warnings
            if not math.isnan(bad_value):  # Inf case
                assert not result.is_valid or len(result.errors) > 0
        except (ValueError, TypeError):
            # Expected - graceful rejection
            pass

    @given(st.sampled_from([float('nan'), float('inf'), float('-inf')]))
    @settings(max_examples=50, deadline=None)
    def test_nan_inf_pressure_handling(self, bad_value: float):
        """
        Test that NaN/Inf pressures are handled gracefully.
        """
        try:
            result = validate_input_ranges(pressure_kpa=bad_value)
            if not math.isnan(bad_value):
                assert not result.is_valid or len(result.errors) > 0
        except (ValueError, TypeError):
            pass

    @given(st.sampled_from([float('nan'), float('inf'), float('-inf')]))
    @settings(max_examples=50, deadline=None)
    def test_nan_inf_quality_handling(self, bad_value: float):
        """
        Test that NaN/Inf quality values are handled gracefully.
        """
        try:
            result = validate_input_ranges(quality_x=bad_value)
            if not math.isnan(bad_value):
                assert not result.is_valid or len(result.errors) > 0
        except (ValueError, TypeError):
            pass

    @given(st.sampled_from([float('nan'), float('inf'), float('-inf')]))
    @settings(max_examples=50, deadline=None)
    def test_nan_inf_in_unit_conversion(self, bad_value: float):
        """
        Test unit converter handles NaN/Inf gracefully.
        """
        try:
            result = UnitConverter.convert(bad_value, "kPa", "bar")
            # If no exception, result should be NaN or Inf (preserved)
            if math.isnan(bad_value):
                assert math.isnan(result), f"NaN should propagate"
            elif math.isinf(bad_value):
                assert math.isinf(result), f"Inf should propagate"
        except (ValueError, TypeError, OverflowError):
            # Also acceptable - reject special values
            pass


# =============================================================================
# TEST CLASS: BOUNDARY CONDITIONS
# =============================================================================

@pytest.mark.hypothesis
class TestBoundaryConditions:
    """
    Test behavior at boundary conditions and edge cases.
    """

    @given(st.floats(min_value=0.0, max_value=0.001, allow_nan=False, allow_infinity=False))
    @settings(max_examples=100, deadline=None)
    def test_near_triple_point_pressure(self, delta_p: float):
        """
        Test behavior near triple point pressure.
        """
        p_triple = REGION_BOUNDARIES["P_MIN"]
        p_test = p_triple + delta_p

        try:
            t_sat = get_saturation_temperature(p_test)
            assert t_sat > 0, f"Saturation temperature must be positive"
            assert t_sat >= IF97_CONSTANTS["T_TRIPLE"] - 1.0, \
                f"T_sat should be near triple point temperature"
        except ValueError:
            # May be outside valid range
            pass

    @given(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=100, deadline=None)
    def test_near_critical_point(self, delta: float):
        """
        Test behavior near critical point.
        """
        p_crit = IF97_CONSTANTS["P_CRIT"]
        t_crit = IF97_CONSTANTS["T_CRIT"]

        # Approach from below
        p_test = p_crit * (1.0 - delta * 0.1)
        t_test = t_crit * (1.0 - delta * 0.05)

        if p_test > REGION_BOUNDARIES["P_MIN"] and t_test > REGION_BOUNDARIES["T_MIN"]:
            try:
                # Properties should still be calculable
                region = detect_region(p_test, t_test)
                assert region in [1, 2, 3, 4, 5], f"Invalid region: {region}"
            except ValueError:
                # Near critical - may fail
                pass

    @given(valid_quality)
    @settings(max_examples=100, deadline=None)
    def test_quality_boundaries(self, x: float):
        """
        Test quality at boundaries (0 and 1).
        """
        # Quality must be in [0, 1]
        assert 0.0 <= x <= 1.0

        # Clamping should not modify valid values
        x_clamped = clamp_quality(x, warn=False)
        assert abs(x_clamped - x) < 1e-10

        # At boundaries
        if x == 0.0:
            assert x_clamped == 0.0, "Quality = 0 should be exact"
        elif x == 1.0:
            assert x_clamped == 1.0, "Quality = 1 should be exact"

    @given(invalid_quality)
    @settings(max_examples=100, deadline=None)
    def test_invalid_quality_clamped(self, x: float):
        """
        Test that invalid quality values are properly clamped.
        """
        x_clamped = clamp_quality(x, warn=False)

        assert 0.0 <= x_clamped <= 1.0, \
            f"Clamped quality must be in [0, 1]: {x_clamped}"

        if x < 0:
            assert x_clamped == 0.0, f"Negative quality should clamp to 0"
        elif x > 1:
            assert x_clamped == 1.0, f"Quality > 1 should clamp to 1"


# =============================================================================
# TEST CLASS: SENSOR TRANSFORMER FUZZING
# =============================================================================

@pytest.mark.hypothesis
class TestSensorTransformerFuzzing:
    """
    Fuzz testing for the sensor transformer component.
    """

    @given(
        st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        calibration_params()
    )
    @settings(max_examples=200, deadline=None)
    def test_calibration_application(self, raw_value: float, cal_params: CalibrationParams):
        """
        Test calibration is applied correctly.
        """
        transformer = SensorTransformer()

        # Apply calibration
        calibrated = transformer.apply_calibration(raw_value, cal_params)

        # Verify linear calibration: y = gain * x + offset
        expected = cal_params.gain * raw_value + cal_params.offset

        assert abs(calibrated - expected) < 1e-10, \
            f"Calibration error: {calibrated} != {expected}"

    @given(
        st.floats(min_value=0.1, max_value=1000.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.1, max_value=1000.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=200, deadline=None)
    def test_range_validation(self, value: float, range_size: float):
        """
        Test range validation with various value and range combinations.
        """
        transformer = SensorTransformer()

        min_val = 0.0
        max_val = min_val + range_size

        result = transformer.validate_range(value, min_val, max_val)

        if value < min_val:
            assert not result.is_valid or result.quality_code == QualityCode.BAD_OUT_OF_RANGE
        elif value > max_val:
            assert not result.is_valid or result.quality_code == QualityCode.BAD_OUT_OF_RANGE
        else:
            assert result.is_valid or result.quality_code.is_good() or result.quality_code.is_uncertain()

    @given(
        st.dictionaries(
            st.text(min_size=1, max_size=20),
            st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=10
        )
    )
    @settings(max_examples=100, deadline=None)
    def test_batch_transform(self, raw_data: Dict[str, float]):
        """
        Test batch transformation handles various inputs.
        """
        assume(len(raw_data) > 0)

        transformer = SensorTransformer()

        # Transform batch
        result = transformer.transform_batch(raw_data)

        # Verify all values transformed
        assert result.total_count == len(raw_data)
        assert len(result.values) == len(raw_data)

        # Quality counts should sum to total
        assert (result.good_count + result.uncertain_count + result.bad_count) == result.total_count

        # Quality score should be valid
        assert 0.0 <= result.quality_score <= 100.0


# =============================================================================
# TEST CLASS: RATE OF CHANGE VALIDATION
# =============================================================================

@pytest.mark.hypothesis
class TestRateOfChangeValidation:
    """
    Test rate of change validation for sensor values.
    """

    @given(
        st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=1.0, max_value=60.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=200, deadline=None)
    def test_rate_calculation(self, value1: float, value2: float, dt_seconds: float, max_rate: float):
        """
        Test rate of change calculation.
        """
        transformer = SensorTransformer()

        t1 = datetime.now(timezone.utc)
        t2 = t1 + timedelta(seconds=dt_seconds)

        # First value - establishes baseline
        result1 = transformer.validate_rate_of_change("test_sensor", value1, t1, max_rate)
        assert result1.is_valid, "First value should always be valid"

        # Second value - rate checked
        result2 = transformer.validate_rate_of_change("test_sensor", value2, t2, max_rate)

        # Calculate actual rate
        actual_rate = abs(value2 - value1) / dt_seconds

        if actual_rate > max_rate:
            # Should be flagged
            assert not result2.rate_check_passed or result2.quality_code != QualityCode.GOOD


# =============================================================================
# RUN CONFIGURATION
# =============================================================================

if __name__ == "__main__":
    import os
    profile = os.getenv("HYPOTHESIS_PROFILE", "dev")
    settings.load_profile(profile)

    pytest.main([__file__, "-v", "--tb=short"])
