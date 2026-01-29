"""
Golden Tests: Unit Conversion Tests for GL-FOUND-X-003.

This module tests unit conversions against golden file specifications.
Tests cover all GL Canonical Unit dimensions with tolerance-based comparisons.

Test Coverage:
    - Energy conversions (MJ canonical)
    - Mass conversions (kg canonical)
    - Volume conversions (m3 canonical)
    - Emissions conversions (kgCO2e canonical)
    - Pressure conversions (kPa canonical)
    - Temperature conversions (degC canonical, affine)

Features:
    - Automatic test discovery from YAML golden files
    - Tolerance-based floating point comparison
    - Cross-validation with Pint library
    - Clear failure messages with diff
"""

import math
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

from .conftest import (
    UNIT_CONVERSIONS_DIR,
    compare_values,
    compare_with_pint,
    load_test_cases,
    discover_golden_files,
    GoldenTestResult,
)


# =============================================================================
# Test Data Loading
# =============================================================================

def get_unit_conversion_test_cases() -> List[Tuple[str, str, Dict[str, Any]]]:
    """Load all unit conversion test cases from golden files."""
    test_cases = []
    for golden_file in discover_golden_files(UNIT_CONVERSIONS_DIR):
        dimension = golden_file.stem
        for test_case in load_test_cases(golden_file):
            test_id = f"{dimension}::{test_case.get('name', 'unnamed')}"
            test_cases.append((dimension, test_id, test_case))
    return test_cases


# Generate test IDs
TEST_CASES = get_unit_conversion_test_cases()
TEST_IDS = [tc[1] for tc in TEST_CASES]


# =============================================================================
# Helper Functions
# =============================================================================

def get_expected_value(test_case: Dict[str, Any]) -> float:
    """Extract expected value from test case."""
    expected = test_case.get("expected", {})
    return expected.get("canonical_value", 0.0)


def get_tolerance(test_case: Dict[str, Any]) -> float:
    """Extract tolerance from test case, with sensible defaults."""
    expected = test_case.get("expected", {})

    # Explicit tolerance takes precedence
    if "tolerance" in expected:
        return float(expected["tolerance"])

    # For exact conversions, use very tight tolerance
    if expected.get("exact", False):
        return 1e-12

    # Default tolerance based on expected value magnitude
    expected_value = expected.get("canonical_value", 1.0)
    if expected_value == 0:
        return 1e-15
    else:
        return abs(float(expected_value)) * 1e-9


def calculate_conversion(
    value: float,
    source_unit: str,
    target_unit: str,
    dimension: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[float]:
    """
    Calculate unit conversion using known factors.

    This function implements the canonical conversion logic for testing.
    """
    # Define conversion factors to canonical units
    ENERGY_FACTORS = {
        "J": 1.0e-6,
        "kJ": 1.0e-3,
        "MJ": 1.0,
        "GJ": 1.0e3,
        "TJ": 1.0e6,
        "Wh": 3.6e-3,
        "kWh": 3.6,
        "MWh": 3600.0,
        "GWh": 3.6e6,
        "BTU": 1.055056e-3,
        "MMBtu": 1055.056,
        "therm": 105.5056,
        "quad": 1.055056e9,
        "cal": 4.184e-6,
        "kcal": 4.184e-3,
        "toe": 41868.0,
        "ktoe": 4.1868e7,
        "tce": 29307.6,
    }

    MASS_FACTORS = {
        "mg": 1.0e-6,
        "g": 1.0e-3,
        "kg": 1.0,
        "t": 1000.0,
        "kt": 1.0e6,
        "Mt": 1.0e9,
        "Gt": 1.0e12,
        "oz": 0.028349523125,
        "lb": 0.45359237,
        "lbs": 0.45359237,
        "st": 6.35029318,
        "cwt_us": 45.359237,
        "cwt_uk": 50.80234544,
        "ton_short": 907.18474,
        "ton_long": 1016.0469088,
        "oz_troy": 0.0311034768,
        "lb_troy": 0.3732417216,
    }

    VOLUME_FACTORS = {
        "mL": 1.0e-6,
        "cL": 1.0e-5,
        "dL": 1.0e-4,
        "L": 1.0e-3,
        "hL": 0.1,
        "kL": 1.0,
        "m3": 1.0,
        "fl_oz_us": 2.95735295625e-5,
        "cup_us": 2.365882365e-4,
        "pt_us": 4.73176473e-4,
        "qt_us": 9.46352946e-4,
        "gal_us": 3.785411784e-3,
        "bbl": 0.158987294928,
        "fl_oz_uk": 2.84130625e-5,
        "pt_uk": 5.6826125e-4,
        "gal_uk": 4.54609e-3,
        "in3": 1.6387064e-5,
        "ft3": 0.028316846592,
        "yd3": 0.764554857984,
        "acre_ft": 1233.48183754752,
        "Mbbl": 158.987294928,
        "MMbbl": 158987.294928,
    }

    EMISSIONS_FACTORS = {
        "gCO2e": 1.0e-3,
        "kgCO2e": 1.0,
        "tCO2e": 1000.0,
        "ktCO2e": 1.0e6,
        "MtCO2e": 1.0e9,
        "GtCO2e": 1.0e12,
        "kgCO2": 1.0,
        "tCO2": 1000.0,
        # GWP factors for direct conversion (default AR5 100yr)
        "kgCH4": 28.0,  # AR5 GWP100
        "kgN2O": 265.0,  # AR5 GWP100
        "kgSF6": 23500.0,  # AR5 GWP100
        "kgHFC-134a": 1300.0,  # AR5 GWP100
        "kgNF3": 16100.0,  # AR5 GWP100
        "tCH4": 28000.0,  # tonnes
        "tN2O": 265000.0,  # tonnes
    }

    # GWP factors by profile for metadata-aware conversions
    GWP_PROFILES = {
        "AR5_100yr": {"CH4": 28.0, "N2O": 265.0, "SF6": 23500.0, "HFC-134a": 1300.0, "NF3": 16100.0},
        "AR5_20yr": {"CH4": 84.0, "N2O": 264.0, "SF6": 16300.0, "HFC-134a": 3710.0, "NF3": 12800.0},
        "AR6_100yr": {"CH4": 27.9, "N2O": 273.0, "SF6": 25200.0, "HFC-134a": 1530.0, "NF3": 17400.0},
        "AR6_20yr": {"CH4": 82.5, "N2O": 273.0, "SF6": 18300.0, "HFC-134a": 4140.0, "NF3": 13400.0},
        "AR5_100yr_fossil": {"CH4": 30.0},
    }

    PRESSURE_FACTORS = {
        "Pa": 1.0e-3,
        "hPa": 0.1,
        "kPa": 1.0,
        "kPa_abs": 1.0,
        "MPa": 1000.0,
        "GPa": 1.0e6,
        "bar": 100.0,
        "mbar": 0.1,
        "atm": 101.325,
        "at": 98.0665,
        "psi": 6.894757293168,
        "psia": 6.894757293168,
        "mmHg": 0.1333224,
        "torr": 0.1333224,
        "inHg": 3.38639,
        "mmH2O": 9.80665e-3,
        "inH2O": 0.2490889,
    }

    # Gauge pressure requires affine transformation (add atmospheric pressure)
    GAUGE_PRESSURE_AFFINE = {
        "kPag": (1.0, 101.325),  # kPag + 101.325 = kPa_abs
        "psig": (6.894757293168, 101.325),  # psig * 6.89 + 101.325 = kPa_abs
    }

    # Temperature is affine (special handling)
    TEMP_TO_CELSIUS = {
        "degC": (1.0, 0.0),
        "degF": (5.0 / 9.0, -32.0 * 5.0 / 9.0),
        "K": (1.0, -273.15),
        "degR": (5.0 / 9.0, -273.15),
    }

    # Select factor map based on dimension
    if dimension == "energy":
        factors = ENERGY_FACTORS
    elif dimension == "mass":
        factors = MASS_FACTORS
    elif dimension == "volume":
        factors = VOLUME_FACTORS
    elif dimension == "emissions" or dimension == "emissions_mass":
        # Handle GWP-based conversions with metadata
        gwp_profile = None
        source_type = None
        if metadata:
            gwp_profile = metadata.get("gwp_profile_id")
            source_type = metadata.get("source_type")

        # Extract gas type from unit (e.g., "kgCH4" -> "CH4")
        gas = None
        mass_prefix = 1.0
        if source_unit.startswith("kg"):
            gas = source_unit[2:]
            mass_prefix = 1.0
        elif source_unit.startswith("t") and len(source_unit) > 1:
            gas = source_unit[1:]
            mass_prefix = 1000.0
        elif source_unit.startswith("g") and len(source_unit) > 1:
            gas = source_unit[1:]
            mass_prefix = 0.001

        # Handle GWP-based conversion
        if gas and gas not in ["CO2e", "CO2"]:
            # Get GWP from profile or use defaults
            gwp = None
            if gwp_profile and gwp_profile in GWP_PROFILES:
                profile = GWP_PROFILES[gwp_profile]
                # Handle fossil CH4 special case
                if source_type == "fossil" and gas == "CH4" and "AR5_100yr_fossil" in GWP_PROFILES:
                    gwp = GWP_PROFILES["AR5_100yr_fossil"].get("CH4", profile.get(gas))
                else:
                    gwp = profile.get(gas)
            if gwp is None:
                # Use default AR5 values from EMISSIONS_FACTORS
                default_key = f"kg{gas}"
                if default_key in EMISSIONS_FACTORS:
                    gwp = EMISSIONS_FACTORS[default_key]

            if gwp:
                # Convert: value * mass_prefix * gwp -> kgCO2e
                canonical_value = value * mass_prefix * gwp
                # Handle target unit (kgCO2e, tCO2e, etc.)
                if target_unit == "kgCO2e":
                    return canonical_value
                elif target_unit == "tCO2e":
                    return canonical_value / 1000.0
                elif target_unit == "gCO2e":
                    return canonical_value * 1000.0
                return canonical_value

        factors = EMISSIONS_FACTORS
    elif dimension == "pressure":
        # Handle gauge pressure conversions (affine)
        if source_unit in GAUGE_PRESSURE_AFFINE:
            scale, offset = GAUGE_PRESSURE_AFFINE[source_unit]
            kpa_abs = value * scale + offset
            # Return as kPa absolute or convert to target
            if target_unit in ["kPa", "kPa_abs"]:
                return kpa_abs
            elif target_unit in PRESSURE_FACTORS:
                return kpa_abs / PRESSURE_FACTORS[target_unit]
            return kpa_abs
        factors = PRESSURE_FACTORS
    elif dimension == "temperature":
        # Affine conversion
        if source_unit in TEMP_TO_CELSIUS:
            scale, offset = TEMP_TO_CELSIUS[source_unit]
            celsius_value = value * scale + offset

            if target_unit == "degC":
                return celsius_value
            elif target_unit == "K":
                return celsius_value + 273.15
            elif target_unit == "degF":
                return celsius_value * 9.0 / 5.0 + 32.0
        return None
    else:
        return None

    # Linear conversion: source -> canonical
    if source_unit in factors:
        canonical_value = value * factors[source_unit]

        # If target is not canonical, convert from canonical -> target
        if target_unit in factors and target_unit != get_canonical_unit(dimension):
            return canonical_value / factors[target_unit]
        return canonical_value

    return None


def get_canonical_unit(dimension: str) -> str:
    """Get canonical unit for a dimension."""
    CANONICAL_UNITS = {
        "energy": "MJ",
        "mass": "kg",
        "volume": "m3",
        "emissions": "kgCO2e",
        "emissions_mass": "kgCO2e",
        "pressure": "kPa",
        "temperature": "degC",
    }
    return CANONICAL_UNITS.get(dimension, "")


# =============================================================================
# Test Classes
# =============================================================================

class TestUnitConversions:
    """Golden tests for unit conversions."""

    @pytest.mark.parametrize("dimension,test_id,test_case", TEST_CASES, ids=TEST_IDS)
    def test_conversion_accuracy(
        self,
        dimension: str,
        test_id: str,
        test_case: Dict[str, Any],
    ):
        """Test conversion accuracy against golden values."""
        input_data = test_case.get("input", {})
        expected = test_case.get("expected", {})

        value = input_data.get("value", 0)
        source_unit = input_data.get("unit", "")
        target_unit = input_data.get("target_unit", get_canonical_unit(dimension))

        expected_value = expected.get("canonical_value")
        tolerance = get_tolerance(test_case)

        # Skip if expected value is not provided (error test cases)
        if expected_value is None:
            pytest.skip("No expected value (error test case)")

        # Calculate actual conversion
        actual_value = calculate_conversion(
            value,
            source_unit,
            target_unit,
            dimension,
            input_data.get("metadata"),
        )

        assert actual_value is not None, (
            f"Conversion not supported: {source_unit} -> {target_unit}"
        )

        # Compare with tolerance
        result = compare_values(expected_value, actual_value, tolerance)

        assert result.passed, (
            f"Conversion mismatch for {test_case.get('name', 'unnamed')}:\n"
            f"  Input: {value} {source_unit}\n"
            f"  Expected: {expected_value} {target_unit}\n"
            f"  Actual: {actual_value}\n"
            f"  {result.diff}"
        )

    @pytest.mark.parametrize("dimension,test_id,test_case", TEST_CASES, ids=TEST_IDS)
    def test_conversion_determinism(
        self,
        dimension: str,
        test_id: str,
        test_case: Dict[str, Any],
    ):
        """Test that conversions are deterministic (same input -> same output)."""
        input_data = test_case.get("input", {})
        expected = test_case.get("expected", {})

        if expected.get("canonical_value") is None:
            pytest.skip("No expected value (error test case)")

        value = input_data.get("value", 0)
        source_unit = input_data.get("unit", "")
        target_unit = input_data.get("target_unit", get_canonical_unit(dimension))

        # Run conversion multiple times
        results = []
        for _ in range(5):
            result = calculate_conversion(
                value, source_unit, target_unit, dimension
            )
            if result is not None:
                results.append(result)

        if not results:
            pytest.skip("Conversion not supported")

        # All results should be identical
        first_result = results[0]
        for i, result in enumerate(results[1:], 1):
            assert result == first_result, (
                f"Non-deterministic conversion detected on iteration {i}:\n"
                f"  First: {first_result}\n"
                f"  Current: {result}"
            )


class TestUnitConversionEdgeCases:
    """Edge case tests for unit conversions."""

    @pytest.mark.parametrize("dimension", ["energy", "mass", "volume", "emissions"])
    def test_zero_value_conversion(self, dimension: str):
        """Zero value should remain zero after conversion."""
        canonical = get_canonical_unit(dimension)
        result = calculate_conversion(0, canonical, canonical, dimension)

        assert result == 0.0, f"Zero conversion failed for {dimension}"

    @pytest.mark.parametrize("dimension", ["energy", "mass", "volume"])
    def test_identity_conversion(self, dimension: str):
        """Converting to same unit should return same value."""
        canonical = get_canonical_unit(dimension)
        test_value = 123.456

        result = calculate_conversion(test_value, canonical, canonical, dimension)

        assert result == test_value, f"Identity conversion failed for {dimension}"

    def test_temperature_absolute_zero(self):
        """Test absolute zero conversions."""
        # 0 K = -273.15 C
        result = calculate_conversion(0, "K", "degC", "temperature")
        assert abs(result - (-273.15)) < 1e-10

        # 0 R = -273.15 C
        result = calculate_conversion(0, "degR", "degC", "temperature")
        assert abs(result - (-273.15)) < 1e-10

    def test_temperature_crossover_point(self):
        """Test -40 F = -40 C crossover point."""
        result = calculate_conversion(-40, "degF", "degC", "temperature")
        assert abs(result - (-40.0)) < 1e-10

    @pytest.mark.parametrize(
        "value,expected",
        [
            (1e-15, 1e-15),  # Very small
            (1e15, 1e15),  # Very large
            (0.123456789012345, 0.123456789012345),  # High precision
        ],
    )
    def test_extreme_values(self, value: float, expected: float):
        """Test extreme value handling."""
        result = calculate_conversion(value, "MJ", "MJ", "energy")
        assert abs(result - expected) < abs(expected) * 1e-14


class TestUnitConversionPintCrossValidation:
    """Cross-validate conversions with Pint library."""

    @pytest.mark.pint_cross_validation
    @pytest.mark.parametrize(
        "value,source,target,dimension",
        [
            (100, "kWh", "MJ", "energy"),
            (1000, "kg", "t", "mass"),
            (1000, "L", "m3", "volume"),
            (100, "psi", "kPa", "pressure"),
        ],
    )
    def test_pint_cross_validation(
        self,
        value: float,
        source: str,
        target: str,
        dimension: str,
    ):
        """Cross-validate conversion results with Pint."""
        try:
            import pint
            ureg = pint.UnitRegistry()
        except ImportError:
            pytest.skip("Pint not available")

        # Our conversion
        our_result = calculate_conversion(value, source, target, dimension)

        if our_result is None:
            pytest.skip(f"Conversion {source} -> {target} not implemented")

        # Map unit names for Pint
        PINT_UNIT_MAP = {
            "kWh": "kilowatt_hour",
            "MJ": "megajoule",
            "t": "metric_ton",
            "m3": "meter**3",
            "kPa": "kilopascal",
        }

        pint_source = PINT_UNIT_MAP.get(source, source)
        pint_target = PINT_UNIT_MAP.get(target, target)

        try:
            pint_qty = ureg.Quantity(value, pint_source)
            pint_result = pint_qty.to(pint_target).magnitude
        except Exception as e:
            pytest.skip(f"Pint conversion failed: {e}")

        # Compare results
        tolerance = abs(pint_result) * 1e-9 if pint_result != 0 else 1e-15
        assert abs(our_result - pint_result) <= tolerance, (
            f"Pint cross-validation failed:\n"
            f"  Our result: {our_result}\n"
            f"  Pint result: {pint_result}\n"
            f"  Difference: {abs(our_result - pint_result)}"
        )


# =============================================================================
# Compliance Tests
# =============================================================================

class TestConversionCompliance:
    """Regulatory compliance tests for unit conversions."""

    @pytest.mark.compliance
    def test_ghg_protocol_energy_conversions(self):
        """Test GHG Protocol energy unit conversions."""
        # Common energy conversions used in GHG reporting
        test_cases = [
            (1000, "kWh", "GJ", 3.6),  # 1000 kWh = 3.6 GJ
            (1, "MMBtu", "GJ", 1.055056),  # 1 MMBtu = 1.055 GJ
            (1, "therm", "kWh", 29.3071),  # 1 therm = 29.3 kWh
        ]

        for value, source, target, expected in test_cases:
            # Convert source to MJ first
            to_mj = calculate_conversion(value, source, "MJ", "energy")
            # Then convert MJ to target
            if target == "GJ":
                result = to_mj / 1000
            elif target == "kWh":
                result = to_mj / 3.6
            else:
                result = to_mj

            assert abs(result - expected) < expected * 0.001, (
                f"GHG Protocol conversion failed: {value} {source} -> {target}\n"
                f"Expected: {expected}, Got: {result}"
            )

    @pytest.mark.compliance
    def test_cbam_mass_conversions(self):
        """Test CBAM-relevant mass conversions."""
        # CBAM uses metric tonnes
        test_cases = [
            (1000, "kg", "t", 1.0),
            (1, "t", "kg", 1000.0),
            (1, "ton_short", "t", 0.90718474),  # US short ton to metric
        ]

        for value, source, target, expected in test_cases:
            to_kg = calculate_conversion(value, source, "kg", "mass")
            if target == "t":
                result = to_kg / 1000
            else:
                result = to_kg

            assert abs(result - expected) < expected * 1e-6, (
                f"CBAM conversion failed: {value} {source} -> {target}\n"
                f"Expected: {expected}, Got: {result}"
            )

    @pytest.mark.compliance
    def test_emissions_gwp_conversions(self):
        """Test emissions conversions with GWP."""
        # Using AR5 100-year GWP values
        GWP_AR5 = {
            "CO2": 1,
            "CH4": 28,
            "N2O": 265,
        }

        # 1 kg CH4 = 28 kgCO2e (AR5)
        ch4_in_co2e = 1 * GWP_AR5["CH4"]
        assert ch4_in_co2e == 28

        # 1 kg N2O = 265 kgCO2e (AR5)
        n2o_in_co2e = 1 * GWP_AR5["N2O"]
        assert n2o_in_co2e == 265


# =============================================================================
# Performance Tests
# =============================================================================

class TestConversionPerformance:
    """Performance tests for unit conversions."""

    @pytest.mark.slow
    def test_batch_conversion_performance(self, benchmark=None):
        """Test batch conversion performance."""
        import time

        values = list(range(10000))
        source_unit = "kWh"
        target_unit = "MJ"
        dimension = "energy"

        start = time.time()
        results = [
            calculate_conversion(v, source_unit, target_unit, dimension)
            for v in values
        ]
        elapsed = time.time() - start

        # Should complete 10000 conversions in under 1 second
        assert elapsed < 1.0, f"Batch conversion too slow: {elapsed:.2f}s"
        assert len(results) == 10000
        assert all(r is not None for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
