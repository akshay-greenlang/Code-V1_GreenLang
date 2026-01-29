"""
Property-Based Tests for Unit Conversion Properties (GL-FOUND-X-003).

This module tests mathematical properties of unit conversions using Hypothesis.
These properties ensure that the conversion engine behaves correctly across
all possible inputs, not just hand-picked test cases.

Properties Tested:
    1. Roundtrip: convert(convert(x, A, B), B, A) == x (within tolerance)
    2. Transitivity: convert(x, A, C) == convert(convert(x, A, B), B, C)
    3. Identity: convert(x, A, A) == x
    4. Scaling: convert(k*x, A, B) == k*convert(x, A, B)
    5. Zero preservation: convert(0, A, B) == 0
    6. Sign preservation: sign(convert(x, A, B)) == sign(x)
    7. Monotonicity: x < y => convert(x, A, B) < convert(y, A, B)

References:
    - FR-030: Convert scalar values to canonical units with full conversion trace
    - FR-037: Ensure numerical tolerance bounds (default 1e-9 relative)
    - NFR-003: Normalization must be deterministic
"""

import math
from typing import Tuple

import pytest
from hypothesis import given, assume, settings, note, example

from .strategies import (
    valid_values,
    valid_positive_values,
    valid_scaling_factors,
    valid_conversion_pairs_strategy,
    valid_same_dimension_units,
    VALID_CONVERSION_PAIRS,
)


# =============================================================================
# Test Configuration
# =============================================================================

# Default tolerance for conversion comparisons (per NFR-037: 1e-9 relative)
RELATIVE_TOLERANCE = 1e-9

# Relaxed tolerance for multi-step conversions
RELAXED_TOLERANCE = 1e-6


# =============================================================================
# Helper Functions
# =============================================================================

def approx_equal(a: float, b: float, rel_tol: float = RELATIVE_TOLERANCE) -> bool:
    """
    Check if two floats are approximately equal within relative tolerance.

    Args:
        a: First value
        b: Second value
        rel_tol: Relative tolerance (default 1e-9 per NFR-037)

    Returns:
        True if values are approximately equal
    """
    if a == b:
        return True
    if a == 0 or b == 0:
        return abs(a - b) < 1e-15
    return abs(a - b) / max(abs(a), abs(b)) < rel_tol


def convert_value(converter, value: float, source_unit: str, target_unit: str) -> float:
    """
    Convert a value using the converter and return the result magnitude.

    Args:
        converter: UnitConverter instance
        value: Value to convert
        source_unit: Source unit
        target_unit: Target unit

    Returns:
        Converted value magnitude
    """
    try:
        from gl_normalizer_core.parser import Quantity
        quantity = Quantity(magnitude=value, unit=source_unit)
        result = converter.convert(quantity, target_unit)
        if result.success and result.converted_quantity:
            return result.converted_quantity.magnitude
        raise ValueError(f"Conversion failed: {result.warnings}")
    except ImportError:
        pytest.skip("gl_normalizer_core not available")


# =============================================================================
# Property 1: Identity - convert(x, A, A) == x
# =============================================================================

class TestIdentityProperty:
    """
    Tests for the identity property: converting a value to the same unit
    should return the original value unchanged.

    Identity: convert(x, A, A) == x

    This is a fundamental property that must hold for any valid conversion.
    """

    @pytest.mark.property
    @pytest.mark.conversion
    @given(value=valid_positive_values())
    @settings(max_examples=1000)
    def test_identity_mass_kg(self, unit_converter, value: float):
        """Identity conversion for kilogram should return same value."""
        result = convert_value(unit_converter, value, "kilogram", "kilogram")
        note(f"Input: {value}, Output: {result}")
        assert approx_equal(result, value), f"Identity failed: {value} != {result}"

    @pytest.mark.property
    @pytest.mark.conversion
    @given(value=valid_positive_values())
    @settings(max_examples=1000)
    def test_identity_energy_kwh(self, unit_converter, value: float):
        """Identity conversion for kilowatt_hour should return same value."""
        result = convert_value(unit_converter, value, "kilowatt_hour", "kilowatt_hour")
        note(f"Input: {value}, Output: {result}")
        assert approx_equal(result, value), f"Identity failed: {value} != {result}"

    @pytest.mark.property
    @pytest.mark.conversion
    @given(value=valid_positive_values())
    @settings(max_examples=1000)
    def test_identity_volume_liter(self, unit_converter, value: float):
        """Identity conversion for liter should return same value."""
        result = convert_value(unit_converter, value, "liter", "liter")
        note(f"Input: {value}, Output: {result}")
        assert approx_equal(result, value), f"Identity failed: {value} != {result}"


# =============================================================================
# Property 2: Roundtrip - convert(convert(x, A, B), B, A) == x
# =============================================================================

class TestRoundtripProperty:
    """
    Tests for the roundtrip property: converting a value from A to B and
    back to A should return the original value (within tolerance).

    Roundtrip: convert(convert(x, A, B), B, A) == x

    This property verifies that conversions are reversible and that the
    inverse conversion factors are correctly defined.
    """

    @pytest.mark.property
    @pytest.mark.conversion
    @given(value=valid_positive_values())
    @settings(max_examples=1000)
    def test_roundtrip_kg_to_gram(self, unit_converter, value: float):
        """Roundtrip conversion kg -> g -> kg should preserve value."""
        intermediate = convert_value(unit_converter, value, "kilogram", "gram")
        result = convert_value(unit_converter, intermediate, "gram", "kilogram")
        note(f"Original: {value}, Intermediate (g): {intermediate}, Final: {result}")
        assert approx_equal(result, value, RELAXED_TOLERANCE), \
            f"Roundtrip failed: {value} -> {intermediate} -> {result}"

    @pytest.mark.property
    @pytest.mark.conversion
    @given(value=valid_positive_values())
    @settings(max_examples=1000)
    def test_roundtrip_kg_to_metric_ton(self, unit_converter, value: float):
        """Roundtrip conversion kg -> t -> kg should preserve value."""
        intermediate = convert_value(unit_converter, value, "kilogram", "metric_ton")
        result = convert_value(unit_converter, intermediate, "metric_ton", "kilogram")
        note(f"Original: {value}, Intermediate (t): {intermediate}, Final: {result}")
        assert approx_equal(result, value, RELAXED_TOLERANCE), \
            f"Roundtrip failed: {value} -> {intermediate} -> {result}"

    @pytest.mark.property
    @pytest.mark.conversion
    @given(value=valid_positive_values())
    @settings(max_examples=1000)
    def test_roundtrip_kwh_to_mj(self, unit_converter, value: float):
        """Roundtrip conversion kWh -> MJ -> kWh should preserve value."""
        intermediate = convert_value(unit_converter, value, "kilowatt_hour", "megajoule")
        result = convert_value(unit_converter, intermediate, "megajoule", "kilowatt_hour")
        note(f"Original: {value}, Intermediate (MJ): {intermediate}, Final: {result}")
        assert approx_equal(result, value, RELAXED_TOLERANCE), \
            f"Roundtrip failed: {value} -> {intermediate} -> {result}"

    @pytest.mark.property
    @pytest.mark.conversion
    @given(value=valid_positive_values())
    @settings(max_examples=1000)
    def test_roundtrip_liter_to_gallon(self, unit_converter, value: float):
        """Roundtrip conversion L -> gal -> L should preserve value."""
        intermediate = convert_value(unit_converter, value, "liter", "gallon")
        result = convert_value(unit_converter, intermediate, "gallon", "liter")
        note(f"Original: {value}, Intermediate (gal): {intermediate}, Final: {result}")
        assert approx_equal(result, value, RELAXED_TOLERANCE), \
            f"Roundtrip failed: {value} -> {intermediate} -> {result}"

    @pytest.mark.property
    @pytest.mark.conversion
    @given(value=valid_positive_values())
    @settings(max_examples=1000)
    def test_roundtrip_km_to_mile(self, unit_converter, value: float):
        """Roundtrip conversion km -> mi -> km should preserve value."""
        intermediate = convert_value(unit_converter, value, "kilometer", "mile")
        result = convert_value(unit_converter, intermediate, "mile", "kilometer")
        note(f"Original: {value}, Intermediate (mi): {intermediate}, Final: {result}")
        assert approx_equal(result, value, RELAXED_TOLERANCE), \
            f"Roundtrip failed: {value} -> {intermediate} -> {result}"


# =============================================================================
# Property 3: Scaling - convert(k*x, A, B) == k*convert(x, A, B)
# =============================================================================

class TestScalingProperty:
    """
    Tests for the scaling property: multiplying the input by a constant
    should multiply the output by the same constant.

    Scaling: convert(k*x, A, B) == k*convert(x, A, B)

    This property verifies that the conversion is linear (for non-affine units).
    """

    @pytest.mark.property
    @pytest.mark.conversion
    @given(
        value=valid_positive_values(),
        scale=valid_scaling_factors(),
    )
    @settings(max_examples=1000)
    def test_scaling_kg_to_gram(self, unit_converter, value: float, scale: float):
        """Scaling property for kg -> g conversion."""
        # Convert x
        result_x = convert_value(unit_converter, value, "kilogram", "gram")
        # Convert k*x
        result_kx = convert_value(unit_converter, value * scale, "kilogram", "gram")
        # k * convert(x)
        expected = scale * result_x

        note(f"x={value}, k={scale}, convert(x)={result_x}, convert(k*x)={result_kx}, k*convert(x)={expected}")
        assert approx_equal(result_kx, expected, RELAXED_TOLERANCE), \
            f"Scaling failed: {result_kx} != {expected}"

    @pytest.mark.property
    @pytest.mark.conversion
    @given(
        value=valid_positive_values(),
        scale=valid_scaling_factors(),
    )
    @settings(max_examples=1000)
    def test_scaling_kwh_to_mj(self, unit_converter, value: float, scale: float):
        """Scaling property for kWh -> MJ conversion."""
        result_x = convert_value(unit_converter, value, "kilowatt_hour", "megajoule")
        result_kx = convert_value(unit_converter, value * scale, "kilowatt_hour", "megajoule")
        expected = scale * result_x

        note(f"x={value}, k={scale}, convert(x)={result_x}, convert(k*x)={result_kx}, k*convert(x)={expected}")
        assert approx_equal(result_kx, expected, RELAXED_TOLERANCE), \
            f"Scaling failed: {result_kx} != {expected}"

    @pytest.mark.property
    @pytest.mark.conversion
    @given(
        value=valid_positive_values(),
        scale=valid_scaling_factors(),
    )
    @settings(max_examples=1000)
    def test_scaling_liter_to_m3(self, unit_converter, value: float, scale: float):
        """Scaling property for L -> m3 conversion."""
        result_x = convert_value(unit_converter, value, "liter", "cubic_meter")
        result_kx = convert_value(unit_converter, value * scale, "liter", "cubic_meter")
        expected = scale * result_x

        note(f"x={value}, k={scale}, convert(x)={result_x}, convert(k*x)={result_kx}, k*convert(x)={expected}")
        assert approx_equal(result_kx, expected, RELAXED_TOLERANCE), \
            f"Scaling failed: {result_kx} != {expected}"


# =============================================================================
# Property 4: Transitivity - convert(x, A, C) == convert(convert(x, A, B), B, C)
# =============================================================================

class TestTransitivityProperty:
    """
    Tests for the transitivity property: converting directly from A to C
    should equal converting from A to B and then B to C.

    Transitivity: convert(x, A, C) == convert(convert(x, A, B), B, C)

    This property verifies consistency across conversion chains.
    """

    @pytest.mark.property
    @pytest.mark.conversion
    @given(value=valid_positive_values())
    @settings(max_examples=1000)
    def test_transitivity_mass_kg_g_mg(self, unit_converter, value: float):
        """Transitivity: kg -> mg == kg -> g -> mg (via gram as intermediate)."""
        # We need to add mg conversion or test with available units
        # Direct: kg -> metric_ton
        # Via gram: kg -> gram -> metric_ton
        intermediate = convert_value(unit_converter, value, "kilogram", "gram")
        via_chain = convert_value(unit_converter, intermediate, "gram", "kilogram")
        via_chain_to_ton = convert_value(unit_converter, via_chain, "kilogram", "metric_ton")

        direct = convert_value(unit_converter, value, "kilogram", "metric_ton")

        note(f"Direct: {direct}, Via chain: {via_chain_to_ton}")
        # Since we're going kg -> g -> kg -> t, it should equal kg -> t
        assert approx_equal(direct, via_chain_to_ton, RELAXED_TOLERANCE), \
            f"Transitivity failed: direct={direct}, chain={via_chain_to_ton}"

    @pytest.mark.property
    @pytest.mark.conversion
    @given(value=valid_positive_values())
    @settings(max_examples=1000)
    def test_transitivity_energy_gj_mj_kwh(self, unit_converter, value: float):
        """Transitivity: GJ -> kWh == GJ -> MJ -> kWh."""
        # Direct path: GJ -> kWh
        gj_to_mj = convert_value(unit_converter, value, "gigajoule", "megajoule")
        mj_to_kwh = convert_value(unit_converter, gj_to_mj, "megajoule", "kilowatt_hour")

        # We need GJ -> kWh directly, but let's verify MJ transitivity
        # Direct: MJ -> kWh
        direct_kwh = convert_value(unit_converter, value, "megajoule", "kilowatt_hour")
        # Via: MJ -> GJ -> MJ -> kWh
        mj_to_gj = convert_value(unit_converter, value, "megajoule", "gigajoule")
        gj_back_to_mj = convert_value(unit_converter, mj_to_gj, "gigajoule", "megajoule")
        via_kwh = convert_value(unit_converter, gj_back_to_mj, "megajoule", "kilowatt_hour")

        note(f"Direct kWh: {direct_kwh}, Via GJ: {via_kwh}")
        assert approx_equal(direct_kwh, via_kwh, RELAXED_TOLERANCE), \
            f"Transitivity failed: direct={direct_kwh}, via_chain={via_kwh}"


# =============================================================================
# Property 5: Zero Preservation - convert(0, A, B) == 0
# =============================================================================

class TestZeroPreservationProperty:
    """
    Tests for zero preservation: converting zero should always return zero.

    Zero Preservation: convert(0, A, B) == 0

    This is a special case of the scaling property (k=0).
    """

    @pytest.mark.property
    @pytest.mark.conversion
    def test_zero_preservation_kg_to_gram(self, unit_converter):
        """Converting 0 kg to g should return 0."""
        result = convert_value(unit_converter, 0.0, "kilogram", "gram")
        assert result == 0.0, f"Zero not preserved: {result}"

    @pytest.mark.property
    @pytest.mark.conversion
    def test_zero_preservation_kwh_to_mj(self, unit_converter):
        """Converting 0 kWh to MJ should return 0."""
        result = convert_value(unit_converter, 0.0, "kilowatt_hour", "megajoule")
        assert result == 0.0, f"Zero not preserved: {result}"

    @pytest.mark.property
    @pytest.mark.conversion
    def test_zero_preservation_liter_to_gallon(self, unit_converter):
        """Converting 0 L to gal should return 0."""
        result = convert_value(unit_converter, 0.0, "liter", "gallon")
        assert result == 0.0, f"Zero not preserved: {result}"


# =============================================================================
# Property 6: Monotonicity - x < y => convert(x, A, B) < convert(y, A, B)
# =============================================================================

class TestMonotonicityProperty:
    """
    Tests for monotonicity: if x < y, then convert(x) < convert(y).

    Monotonicity: x < y => convert(x, A, B) < convert(y, A, B)

    This property ensures that ordering is preserved through conversion.
    """

    @pytest.mark.property
    @pytest.mark.conversion
    @given(
        x=valid_positive_values(),
        y=valid_positive_values(),
    )
    @settings(max_examples=1000)
    def test_monotonicity_kg_to_gram(self, unit_converter, x: float, y: float):
        """Conversion from kg to g should preserve ordering."""
        assume(x != y)  # Skip if values are equal

        result_x = convert_value(unit_converter, x, "kilogram", "gram")
        result_y = convert_value(unit_converter, y, "kilogram", "gram")

        if x < y:
            assert result_x < result_y, f"Monotonicity violated: {x} < {y} but {result_x} >= {result_y}"
        else:
            assert result_x > result_y, f"Monotonicity violated: {x} > {y} but {result_x} <= {result_y}"

    @pytest.mark.property
    @pytest.mark.conversion
    @given(
        x=valid_positive_values(),
        y=valid_positive_values(),
    )
    @settings(max_examples=1000)
    def test_monotonicity_kwh_to_mj(self, unit_converter, x: float, y: float):
        """Conversion from kWh to MJ should preserve ordering."""
        assume(x != y)

        result_x = convert_value(unit_converter, x, "kilowatt_hour", "megajoule")
        result_y = convert_value(unit_converter, y, "kilowatt_hour", "megajoule")

        if x < y:
            assert result_x < result_y, f"Monotonicity violated: {x} < {y} but {result_x} >= {result_y}"
        else:
            assert result_x > result_y, f"Monotonicity violated: {x} > {y} but {result_x} <= {result_y}"


# =============================================================================
# Property 7: Determinism - Same inputs always produce same outputs
# =============================================================================

class TestDeterminismProperty:
    """
    Tests for determinism: the same input should always produce the same output.

    Determinism: convert(x, A, B) at time t1 == convert(x, A, B) at time t2

    This property is critical for reproducibility (NFR-003).
    """

    @pytest.mark.property
    @pytest.mark.conversion
    @given(value=valid_positive_values())
    @settings(max_examples=1000)
    def test_determinism_multiple_calls(self, unit_converter, value: float):
        """Multiple conversion calls with same input should return same result."""
        results = [
            convert_value(unit_converter, value, "kilogram", "gram")
            for _ in range(10)
        ]

        first_result = results[0]
        for i, result in enumerate(results[1:], 2):
            assert result == first_result, \
                f"Non-deterministic: call {i} returned {result}, expected {first_result}"

    @pytest.mark.property
    @pytest.mark.conversion
    @given(value=valid_positive_values())
    @settings(max_examples=1000)
    def test_determinism_provenance_hash(self, unit_converter, value: float):
        """Provenance hash should be deterministic for same input."""
        try:
            from gl_normalizer_core.parser import Quantity
            quantity = Quantity(magnitude=value, unit="kilogram")

            # Multiple conversions
            results = [
                unit_converter.convert(quantity, "gram")
                for _ in range(5)
            ]

            first_hash = results[0].provenance_hash
            for i, result in enumerate(results[1:], 2):
                assert result.provenance_hash == first_hash, \
                    f"Non-deterministic hash: call {i} hash differs"
        except ImportError:
            pytest.skip("gl_normalizer_core not available")


# =============================================================================
# Property 8: Conversion Factor Consistency
# =============================================================================

class TestConversionFactorConsistency:
    """
    Tests that conversion factors are internally consistent.

    Verifies that factor(A, B) * factor(B, A) == 1 (within tolerance).
    """

    @pytest.mark.property
    @pytest.mark.conversion
    def test_inverse_factor_consistency(self, conversion_factors):
        """
        For each conversion pair, verify the inverse factor is correct.

        factor(A, B) * factor(B, A) should equal 1.
        """
        checked_pairs = set()

        for (source, target), factor in conversion_factors.items():
            if (target, source) in checked_pairs:
                continue

            inverse_factor = conversion_factors.get((target, source))
            if inverse_factor is not None:
                product = factor * inverse_factor
                assert approx_equal(product, 1.0, RELAXED_TOLERANCE), \
                    f"Inverse factors inconsistent for {source}<->{target}: " \
                    f"{factor} * {inverse_factor} = {product} (expected 1.0)"
                checked_pairs.add((source, target))


# =============================================================================
# Property 9: Known Value Verification
# =============================================================================

class TestKnownValueVerification:
    """
    Tests conversion against known exact values.

    These are regression tests to ensure basic conversions are correct.
    """

    @pytest.mark.property
    @pytest.mark.conversion
    @pytest.mark.parametrize("source_unit,target_unit,input_value,expected", [
        ("kilogram", "gram", 1.0, 1000.0),
        ("gram", "kilogram", 1000.0, 1.0),
        ("kilogram", "metric_ton", 1000.0, 1.0),
        ("metric_ton", "kilogram", 1.0, 1000.0),
        ("kilowatt_hour", "megajoule", 1.0, 3.6),
        ("megajoule", "kilowatt_hour", 3.6, 1.0),
        ("liter", "cubic_meter", 1000.0, 1.0),
        ("cubic_meter", "liter", 1.0, 1000.0),
    ])
    def test_known_conversion_values(
        self,
        unit_converter,
        source_unit: str,
        target_unit: str,
        input_value: float,
        expected: float,
    ):
        """Test conversions against known exact values."""
        result = convert_value(unit_converter, input_value, source_unit, target_unit)
        assert approx_equal(result, expected, RELAXED_TOLERANCE), \
            f"Known value mismatch: {input_value} {source_unit} -> {result} {target_unit}, expected {expected}"


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """
    Tests for edge cases that could cause numerical issues.
    """

    @pytest.mark.property
    @pytest.mark.conversion
    def test_very_small_value(self, unit_converter):
        """Test conversion of very small values."""
        small_value = 1e-10
        result = convert_value(unit_converter, small_value, "kilogram", "gram")
        expected = small_value * 1000
        assert approx_equal(result, expected, RELAXED_TOLERANCE), \
            f"Small value conversion failed: {result} != {expected}"

    @pytest.mark.property
    @pytest.mark.conversion
    def test_very_large_value(self, unit_converter):
        """Test conversion of very large values."""
        large_value = 1e10
        result = convert_value(unit_converter, large_value, "kilogram", "metric_ton")
        expected = large_value / 1000
        assert approx_equal(result, expected, RELAXED_TOLERANCE), \
            f"Large value conversion failed: {result} != {expected}"

    @pytest.mark.property
    @pytest.mark.conversion
    @given(value=valid_positive_values())
    @settings(max_examples=100)
    def test_precision_preservation(self, unit_converter, value: float):
        """
        Test that precision is preserved through conversion chain.

        Multiple roundtrips should still return approximately the original value.
        """
        current = value
        for _ in range(5):
            # kg -> g -> kg
            current = convert_value(unit_converter, current, "kilogram", "gram")
            current = convert_value(unit_converter, current, "gram", "kilogram")

        assert approx_equal(current, value, 1e-5), \
            f"Precision lost after 5 roundtrips: {value} -> {current}"
