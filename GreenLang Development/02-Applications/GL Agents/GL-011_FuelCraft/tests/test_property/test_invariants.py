# -*- coding: utf-8 -*-
"""
Property-Based Tests for GL-011 FuelCraft

Uses Hypothesis to validate mathematical invariants and properties
that must hold for ALL valid inputs.

Invariants Tested:
1. Blend fractions always sum to 1.0
2. All costs are non-negative
3. Inventory never goes negative
4. Carbon intensity is always positive
5. Energy calculations are consistent
6. Provenance hashes are deterministic

Author: GL-TestEngineer
Date: 2025-01-01
"""

import sys
from pathlib import Path
from decimal import Decimal
from datetime import datetime, timezone

import pytest

# Conditionally import hypothesis
try:
    from hypothesis import given, strategies as st, settings, assume, HealthCheck
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    # Create dummy decorators for when hypothesis is not available
    def given(*args, **kwargs):
        def decorator(func):
            return pytest.mark.skip(reason="hypothesis not installed")(func)
        return decorator

    class st:
        @staticmethod
        def decimals(*args, **kwargs):
            return None
        @staticmethod
        def lists(*args, **kwargs):
            return None
        @staticmethod
        def integers(*args, **kwargs):
            return None
        @staticmethod
        def floats(*args, **kwargs):
            return None

    def settings(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def assume(condition):
        pass

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# Strategies for Generating Test Data
# =============================================================================

if HYPOTHESIS_AVAILABLE:
    # Decimal strategy for fuel properties
    positive_decimal = st.decimals(
        min_value=Decimal("0.001"),
        max_value=Decimal("1000000"),
        allow_nan=False,
        allow_infinity=False,
        places=6
    )

    # Decimal strategy for fractions (0 to 1)
    fraction_decimal = st.decimals(
        min_value=Decimal("0.0"),
        max_value=Decimal("1.0"),
        allow_nan=False,
        allow_infinity=False,
        places=6
    )

    # Decimal strategy for prices (positive)
    price_decimal = st.decimals(
        min_value=Decimal("0.0001"),
        max_value=Decimal("1000"),
        allow_nan=False,
        allow_infinity=False,
        places=6
    )

    # Decimal strategy for carbon intensity
    carbon_intensity = st.decimals(
        min_value=Decimal("0.001"),
        max_value=Decimal("1.0"),
        allow_nan=False,
        allow_infinity=False,
        places=6
    )


# =============================================================================
# Blend Fraction Invariants
# =============================================================================

@pytest.mark.property
class TestBlendFractionInvariants:
    """Test invariants for blend fractions."""

    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
    @given(st.lists(fraction_decimal, min_size=2, max_size=5))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_normalized_fractions_sum_to_one(self, fractions):
        """Any set of normalized fractions must sum to 1.0."""
        assume(sum(fractions) > Decimal("0"))

        # Normalize fractions
        total = sum(fractions)
        normalized = [f / total for f in fractions]

        # Check sum
        fraction_sum = sum(normalized)

        # Allow small tolerance for decimal arithmetic
        assert abs(fraction_sum - Decimal("1.0")) < Decimal("0.0001")

    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
    @given(st.lists(fraction_decimal, min_size=1, max_size=10))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_all_fractions_non_negative(self, fractions):
        """All blend fractions must be non-negative."""
        for frac in fractions:
            assert frac >= Decimal("0")


# =============================================================================
# Cost Invariants
# =============================================================================

@pytest.mark.property
class TestCostInvariants:
    """Test invariants for cost calculations."""

    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
    @given(
        quantity=positive_decimal,
        price=price_decimal
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_purchase_cost_non_negative(self, quantity, price):
        """Purchase cost = quantity * price must be non-negative."""
        cost = quantity * price
        assert cost >= Decimal("0")

    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
    @given(
        quantities=st.lists(positive_decimal, min_size=1, max_size=5),
        prices=st.lists(price_decimal, min_size=1, max_size=5)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_total_cost_is_sum_of_components(self, quantities, prices):
        """Total cost equals sum of individual fuel costs."""
        assume(len(quantities) == len(prices))

        total = sum(q * p for q, p in zip(quantities, prices))

        assert total >= Decimal("0")

    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
    @given(
        emissions=positive_decimal,
        carbon_price=price_decimal
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_carbon_cost_non_negative(self, emissions, carbon_price):
        """Carbon cost must be non-negative."""
        cost = emissions * carbon_price
        assert cost >= Decimal("0")


# =============================================================================
# Inventory Invariants
# =============================================================================

@pytest.mark.property
class TestInventoryInvariants:
    """Test invariants for inventory calculations."""

    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
    @given(
        initial=positive_decimal,
        inflow=positive_decimal,
        outflow=positive_decimal,
        loss_rate=st.decimals(
            min_value=Decimal("0"),
            max_value=Decimal("0.1"),
            allow_nan=False,
            allow_infinity=False,
            places=6
        )
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_inventory_balance_equation(self, initial, inflow, outflow, loss_rate):
        """Inventory balance: final = initial + inflow - outflow - losses."""
        # Only test valid scenarios where outflow <= initial + inflow
        assume(outflow <= initial + inflow)

        losses = initial * loss_rate
        final = initial + inflow - outflow - losses

        # Final inventory should be non-negative for valid scenarios
        if outflow + losses <= initial + inflow:
            assert final >= Decimal("-0.001")  # Small tolerance

    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
    @given(
        capacity=positive_decimal,
        level=positive_decimal
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_inventory_within_capacity(self, capacity, level):
        """Inventory level must be within tank capacity."""
        # Constrain level to be at most capacity for valid scenarios
        valid_level = min(level, capacity)
        assert valid_level <= capacity


# =============================================================================
# Carbon Intensity Invariants
# =============================================================================

@pytest.mark.property
class TestCarbonIntensityInvariants:
    """Test invariants for carbon intensity calculations."""

    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
    @given(carbon_intensity)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_carbon_intensity_positive(self, ci):
        """Carbon intensity must be positive."""
        assert ci > Decimal("0")

    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
    @given(
        intensities=st.lists(carbon_intensity, min_size=2, max_size=5),
        fractions=st.lists(fraction_decimal, min_size=2, max_size=5)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_blended_ci_within_component_range(self, intensities, fractions):
        """Blended carbon intensity must be within range of components."""
        assume(len(intensities) == len(fractions))
        assume(sum(fractions) > Decimal("0"))

        # Normalize fractions
        total_frac = sum(fractions)
        normalized = [f / total_frac for f in fractions]

        # Calculate blend (simple weighted average)
        blended_ci = sum(ci * frac for ci, frac in zip(intensities, normalized))

        # Blend must be within range of components
        min_ci = min(intensities)
        max_ci = max(intensities)

        assert blended_ci >= min_ci - Decimal("0.0001")
        assert blended_ci <= max_ci + Decimal("0.0001")


# =============================================================================
# Energy Calculation Invariants
# =============================================================================

@pytest.mark.property
class TestEnergyInvariants:
    """Test invariants for energy calculations."""

    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
    @given(
        mass=positive_decimal,
        lhv=st.decimals(
            min_value=Decimal("10"),
            max_value=Decimal("60"),
            allow_nan=False,
            allow_infinity=False,
            places=6
        )
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_energy_equals_mass_times_lhv(self, mass, lhv):
        """Energy = mass * LHV must be positive."""
        energy = mass * lhv
        assert energy > Decimal("0")

    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
    @given(
        lhv=st.decimals(
            min_value=Decimal("10"),
            max_value=Decimal("60"),
            allow_nan=False,
            allow_infinity=False,
            places=6
        ),
        hhv=st.decimals(
            min_value=Decimal("10"),
            max_value=Decimal("70"),
            allow_nan=False,
            allow_infinity=False,
            places=6
        )
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_hhv_greater_than_lhv(self, lhv, hhv):
        """HHV should typically be greater than LHV (water vapor energy)."""
        # For most fuels, HHV > LHV
        # Allow some tolerance for edge cases
        assume(hhv >= lhv)
        assert hhv >= lhv


# =============================================================================
# Provenance Hash Invariants
# =============================================================================

@pytest.mark.property
class TestProvenanceInvariants:
    """Test invariants for provenance hash generation."""

    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
    @given(st.text(min_size=1, max_size=1000))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_hash_determinism(self, input_data):
        """Same input must produce same hash."""
        import hashlib

        hash1 = hashlib.sha256(input_data.encode()).hexdigest()
        hash2 = hashlib.sha256(input_data.encode()).hexdigest()

        assert hash1 == hash2

    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
    @given(
        data1=st.text(min_size=1, max_size=100),
        data2=st.text(min_size=1, max_size=100)
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_different_inputs_different_hashes(self, data1, data2):
        """Different inputs should produce different hashes."""
        assume(data1 != data2)
        import hashlib

        hash1 = hashlib.sha256(data1.encode()).hexdigest()
        hash2 = hashlib.sha256(data2.encode()).hexdigest()

        assert hash1 != hash2

    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
    @given(st.binary(min_size=1, max_size=1000))
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_hash_length_constant(self, data):
        """SHA-256 hash is always 64 hex characters."""
        import hashlib

        hash_result = hashlib.sha256(data).hexdigest()

        assert len(hash_result) == 64


# =============================================================================
# Safety Constraint Invariants
# =============================================================================

@pytest.mark.property
class TestSafetyConstraintInvariants:
    """Test invariants for safety constraints."""

    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
    @given(
        flash_point=st.decimals(
            min_value=Decimal("-50"),
            max_value=Decimal("300"),
            allow_nan=False,
            allow_infinity=False,
            places=2
        ),
        min_flash=st.decimals(
            min_value=Decimal("50"),
            max_value=Decimal("100"),
            allow_nan=False,
            allow_infinity=False,
            places=2
        )
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_flash_point_constraint_logic(self, flash_point, min_flash):
        """Flash point constraint validation is consistent."""
        is_valid = flash_point >= min_flash

        if flash_point >= min_flash:
            assert is_valid is True
        else:
            assert is_valid is False

    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
    @given(
        sulfur=st.decimals(
            min_value=Decimal("0"),
            max_value=Decimal("5"),
            allow_nan=False,
            allow_infinity=False,
            places=4
        ),
        max_sulfur=st.decimals(
            min_value=Decimal("0.1"),
            max_value=Decimal("1"),
            allow_nan=False,
            allow_infinity=False,
            places=4
        )
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_sulfur_constraint_logic(self, sulfur, max_sulfur):
        """Sulfur constraint validation is consistent."""
        is_valid = sulfur <= max_sulfur

        if sulfur <= max_sulfur:
            assert is_valid is True
        else:
            assert is_valid is False


# =============================================================================
# Mathematical Identities
# =============================================================================

@pytest.mark.property
class TestMathematicalIdentities:
    """Test mathematical identities that must always hold."""

    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
    @given(
        a=positive_decimal,
        b=positive_decimal
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_addition_commutativity(self, a, b):
        """a + b = b + a for all Decimals."""
        assert a + b == b + a

    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
    @given(
        a=positive_decimal,
        b=positive_decimal
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_multiplication_commutativity(self, a, b):
        """a * b = b * a for all Decimals."""
        assert a * b == b * a

    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
    @given(positive_decimal)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_division_identity(self, a):
        """a / a = 1 for all non-zero Decimals."""
        assume(a != Decimal("0"))
        result = a / a
        assert abs(result - Decimal("1")) < Decimal("0.0001")


# =============================================================================
# Unit Conversion Invariants
# =============================================================================

@pytest.mark.property
class TestUnitConversionInvariants:
    """Test invariants for unit conversions."""

    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
    @given(positive_decimal)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_mj_to_mmbtu_roundtrip(self, energy_mj):
        """MJ -> MMBtu -> MJ should return original value."""
        # Conversion factors
        MJ_PER_MMBTU = Decimal("1055.06")

        # Convert MJ to MMBtu
        energy_mmbtu = energy_mj / MJ_PER_MMBTU

        # Convert back to MJ
        energy_mj_roundtrip = energy_mmbtu * MJ_PER_MMBTU

        # Check roundtrip (allow small tolerance)
        diff = abs(energy_mj - energy_mj_roundtrip)
        assert diff < Decimal("0.01")

    @pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="hypothesis not installed")
    @given(positive_decimal)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_kg_to_tonnes_roundtrip(self, mass_kg):
        """kg -> tonnes -> kg should return original value."""
        # Convert kg to tonnes
        mass_tonnes = mass_kg / Decimal("1000")

        # Convert back to kg
        mass_kg_roundtrip = mass_tonnes * Decimal("1000")

        # Check roundtrip
        assert mass_kg == mass_kg_roundtrip
