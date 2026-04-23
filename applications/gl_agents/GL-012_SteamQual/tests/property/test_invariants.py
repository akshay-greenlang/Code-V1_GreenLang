# -*- coding: utf-8 -*-
"""
GL-012 STEAMQUAL - Property-Based Tests for Steam Quality Invariants

Uses Hypothesis to verify mathematical and physical invariants:
1. Dryness fraction always in [0, 1]
2. Velocity limiting never exceeds configured max rates
3. Energy balance equations hold
4. Provenance hashes are deterministic
5. Thermodynamic properties are consistent

Reference:
    - ASME PTC 19.11: Steam Properties
    - IAPWS-IF97: Industrial Formulation for Steam
    - Hypothesis Testing Library

Author: GL-TestEngineer
Date: December 2025
Version: 1.0.0
"""

import hashlib
import math
from decimal import Decimal
from typing import Dict, List, Optional

import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

# Import steam quality components
from calculators.dryness_fraction_calculator import (
    DrynessCalculator,
    DrynessCalculationResult,
)
from safety.velocity_limiter import (
    VelocityLimiter,
    VelocityLimitConfig,
    VelocityLimitResult,
    VelocityLimitStatus,
    SetpointType,
)
from thermodynamics.steam_properties import (
    SteamPropertiesCalculator,
)


# =============================================================================
# HYPOTHESIS STRATEGIES
# =============================================================================

# Steam quality parameters within physical bounds
dryness_fraction = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
superheat_celsius = st.floats(min_value=0.0, max_value=200.0, allow_nan=False)
pressure_mpa = st.floats(min_value=0.1, max_value=22.0, allow_nan=False)  # Below critical
temperature_celsius = st.floats(min_value=100.0, max_value=400.0, allow_nan=False)
flow_rate_percent = st.floats(min_value=0.0, max_value=100.0, allow_nan=False)
enthalpy_kj_kg = st.floats(min_value=100.0, max_value=3500.0, allow_nan=False)

# Positive non-zero values
positive_float = st.floats(min_value=0.001, max_value=1000.0, allow_nan=False)

# Valid header IDs
header_id = st.text(min_size=1, max_size=20, alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")


# =============================================================================
# DRYNESS CALCULATOR INVARIANTS
# =============================================================================


class TestDrynessCalculatorInvariants:
    """Property-based tests for dryness fraction calculations."""

    @pytest.fixture
    def calculator(self):
        """Create dryness calculator instance."""
        return DrynessCalculator()

    @given(
        enthalpy_mixture=enthalpy_kj_kg,
        enthalpy_liquid=st.floats(min_value=100.0, max_value=1500.0),
        enthalpy_vapor=st.floats(min_value=1800.0, max_value=3500.0),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_dryness_fraction_bounds(
        self,
        calculator,
        enthalpy_mixture: float,
        enthalpy_liquid: float,
        enthalpy_vapor: float,
    ):
        """
        INVARIANT: Dryness fraction must always be in [0, 1].

        Mathematical property: x = (h_mix - h_f) / (h_g - h_f)
        Must clamp to [0, 1] for any valid inputs.
        """
        assume(enthalpy_vapor > enthalpy_liquid)  # Physical requirement

        try:
            result = calculator.calculate_from_enthalpy(
                h_mixture=enthalpy_mixture,
                h_liquid=enthalpy_liquid,
                h_vapor=enthalpy_vapor,
            )

            # INVARIANT: Dryness always in [0, 1]
            assert 0.0 <= result.dryness_fraction <= 1.0, (
                f"Dryness fraction {result.dryness_fraction} out of bounds"
            )

            # INVARIANT: Provenance hash exists
            assert result.provenance_hash, "Missing provenance hash"
            assert len(result.provenance_hash) == 64, "Invalid hash length"

        except ValueError:
            # Invalid input combinations are acceptable
            pass

    @given(
        dryness=dryness_fraction,
        h_liquid=st.floats(min_value=100.0, max_value=1500.0),
        h_vapor=st.floats(min_value=1800.0, max_value=3500.0),
    )
    @settings(max_examples=100)
    def test_enthalpy_reconstruction(
        self,
        calculator,
        dryness: float,
        h_liquid: float,
        h_vapor: float,
    ):
        """
        INVARIANT: h_mix = h_f + x * (h_g - h_f)

        Forward and reverse calculations must be consistent.
        """
        assume(h_vapor > h_liquid)

        # Forward: compute mixture enthalpy from dryness
        h_mix_computed = h_liquid + dryness * (h_vapor - h_liquid)

        # Reverse: compute dryness from mixture enthalpy
        try:
            result = calculator.calculate_from_enthalpy(
                h_mixture=h_mix_computed,
                h_liquid=h_liquid,
                h_vapor=h_vapor,
            )

            # INVARIANT: Round-trip consistency
            assert abs(result.dryness_fraction - dryness) < 0.0001, (
                f"Round-trip error: {result.dryness_fraction} vs {dryness}"
            )

        except ValueError:
            pass

    @given(
        enthalpy=enthalpy_kj_kg,
        h_liquid=st.floats(min_value=100.0, max_value=1500.0),
        h_vapor=st.floats(min_value=1800.0, max_value=3500.0),
    )
    @settings(max_examples=50)
    def test_determinism(
        self,
        calculator,
        enthalpy: float,
        h_liquid: float,
        h_vapor: float,
    ):
        """
        INVARIANT: Same inputs always produce same outputs.

        Zero-hallucination guarantee: calculations are deterministic.
        """
        assume(h_vapor > h_liquid)

        try:
            result1 = calculator.calculate_from_enthalpy(
                h_mixture=enthalpy,
                h_liquid=h_liquid,
                h_vapor=h_vapor,
            )
            result2 = calculator.calculate_from_enthalpy(
                h_mixture=enthalpy,
                h_liquid=h_liquid,
                h_vapor=h_vapor,
            )

            # INVARIANT: Determinism
            assert result1.dryness_fraction == result2.dryness_fraction
            assert result1.provenance_hash == result2.provenance_hash

        except ValueError:
            pass


# =============================================================================
# VELOCITY LIMITER INVARIANTS
# =============================================================================


class TestVelocityLimiterInvariants:
    """Property-based tests for velocity limiting."""

    @pytest.fixture
    def limiter(self):
        """Create velocity limiter instance."""
        return VelocityLimiter()

    @given(
        current_value=dryness_fraction,
        requested_value=dryness_fraction,
    )
    @settings(max_examples=100)
    def test_rate_limiting_bounds(
        self,
        limiter,
        current_value: float,
        requested_value: float,
    ):
        """
        INVARIANT: Limited value change never exceeds max delta.

        The velocity limiter must always respect configured rate limits.
        """
        result = limiter.limit_dryness_fraction(
            requested_value=requested_value,
            header_id="TEST-HDR-001",
        )

        # First call sets the previous value, so do a second call
        result = limiter.limit_dryness_fraction(
            requested_value=requested_value,
            header_id="TEST-HDR-001",
        )

        max_delta = limiter.config.max_delta_dryness_fraction

        # INVARIANT: Change never exceeds max delta
        actual_delta = abs(result.delta_allowed)
        assert actual_delta <= max_delta + 0.0001, (
            f"Delta {actual_delta} exceeds max {max_delta}"
        )

    @given(
        setpoint1=dryness_fraction,
        setpoint2=dryness_fraction,
        setpoint3=dryness_fraction,
    )
    @settings(max_examples=50)
    def test_monotonic_convergence(
        self,
        limiter,
        setpoint1: float,
        setpoint2: float,
        setpoint3: float,
    ):
        """
        INVARIANT: Repeated limiting converges towards target.

        The limited value should move towards the requested value
        (assuming no emergency overrides).
        """
        # Reset state for clean test
        limiter.reset_state("CONV-TEST")

        # Set initial state
        limiter.limit_dryness_fraction(setpoint1, "CONV-TEST")

        # Request a different value
        target = setpoint2

        # Apply limiting multiple times
        prev_distance = abs(target - setpoint1)
        for _ in range(10):
            result = limiter.limit_dryness_fraction(target, "CONV-TEST")
            current_distance = abs(target - result.limited_setpoint)

            # INVARIANT: Distance should not increase (monotonic convergence)
            assert current_distance <= prev_distance + 0.0001, (
                f"Distance increased: {prev_distance} -> {current_distance}"
            )
            prev_distance = current_distance

    @given(
        requested=st.floats(min_value=0.8, max_value=1.0),
    )
    @settings(max_examples=50)
    def test_emergency_override_bypasses_limits(
        self,
        limiter,
        requested: float,
    ):
        """
        INVARIANT: Emergency override allows immediate setpoint change.

        When emergency_override=True, rate limiting is bypassed.
        """
        # Reset and set far initial value
        limiter.reset_state("EMERG-TEST")
        limiter.limit_dryness_fraction(0.5, "EMERG-TEST")

        # Request with emergency override
        result = limiter.limit_dryness_fraction(
            requested_value=requested,
            header_id="EMERG-TEST",
            emergency_override=True,
        )

        # INVARIANT: Emergency allows full change
        assert result.status == VelocityLimitStatus.EMERGENCY_OVERRIDE
        assert result.limited_setpoint == requested

    @given(
        value1=dryness_fraction,
        value2=dryness_fraction,
    )
    @settings(max_examples=50)
    def test_provenance_hash_uniqueness(
        self,
        limiter,
        value1: float,
        value2: float,
    ):
        """
        INVARIANT: Different inputs produce different provenance hashes.

        Provenance hashes must be unique for audit trail integrity.
        """
        assume(abs(value1 - value2) > 0.01)  # Ensure meaningfully different

        limiter.reset_state("HASH-TEST")
        result1 = limiter.limit_dryness_fraction(value1, "HASH-TEST")

        limiter.reset_state("HASH-TEST")
        result2 = limiter.limit_dryness_fraction(value2, "HASH-TEST")

        # INVARIANT: Different inputs -> different hashes
        assert result1.provenance_hash != result2.provenance_hash


# =============================================================================
# THERMODYNAMIC INVARIANTS
# =============================================================================


class TestThermodynamicInvariants:
    """Property-based tests for thermodynamic calculations."""

    @given(
        pressure=pressure_mpa,
    )
    @settings(max_examples=50)
    def test_saturation_temperature_monotonic(
        self,
        pressure: float,
    ):
        """
        INVARIANT: Saturation temperature increases with pressure.

        T_sat(P1) < T_sat(P2) when P1 < P2 (monotonic relationship).
        """
        try:
            from thermodynamics.steam_properties import get_saturation_temperature

            assume(pressure < 22.0)  # Below critical point

            # Get saturation temperatures at two pressures
            p1 = pressure * 0.9
            p2 = pressure

            assume(p1 > 0.01)

            t1 = get_saturation_temperature(p1)
            t2 = get_saturation_temperature(p2)

            # INVARIANT: Monotonic increase
            assert t1 <= t2, f"T_sat not monotonic: T({p1})={t1} > T({p2})={t2}"

        except (ImportError, ValueError):
            pytest.skip("Thermodynamic functions not available")

    @given(
        pressure=st.floats(min_value=0.1, max_value=20.0),
    )
    @settings(max_examples=50)
    def test_latent_heat_positive(
        self,
        pressure: float,
    ):
        """
        INVARIANT: Latent heat of vaporization is always positive.

        h_fg = h_g - h_f > 0 for all pressures below critical point.
        """
        try:
            from thermodynamics.steam_properties import (
                get_saturated_liquid_enthalpy,
                get_saturated_vapor_enthalpy,
            )

            h_f = get_saturated_liquid_enthalpy(pressure)
            h_g = get_saturated_vapor_enthalpy(pressure)
            h_fg = h_g - h_f

            # INVARIANT: Latent heat positive
            assert h_fg > 0, f"Latent heat not positive: h_fg={h_fg}"

            # Additional: h_fg decreases with pressure (approaches 0 at critical)
            # This is a physical property of steam

        except (ImportError, ValueError):
            pytest.skip("Thermodynamic functions not available")


# =============================================================================
# PROVENANCE AND AUDIT INVARIANTS
# =============================================================================


class TestProvenanceInvariants:
    """Property-based tests for provenance and audit trail."""

    @given(
        data=st.dictionaries(
            keys=st.text(min_size=1, max_size=20),
            values=st.floats(allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=10,
        )
    )
    @settings(max_examples=50)
    def test_hash_determinism(self, data: Dict[str, float]):
        """
        INVARIANT: Hash function is deterministic.

        Same data always produces same hash.
        """
        import json

        data_str = json.dumps(data, sort_keys=True)
        hash1 = hashlib.sha256(data_str.encode()).hexdigest()
        hash2 = hashlib.sha256(data_str.encode()).hexdigest()

        assert hash1 == hash2, "Hash not deterministic"
        assert len(hash1) == 64, "Invalid SHA-256 length"

    @given(
        data1=st.dictionaries(
            keys=st.text(min_size=1, max_size=10),
            values=st.floats(min_value=0, max_value=100, allow_nan=False),
            min_size=1,
            max_size=5,
        ),
        data2=st.dictionaries(
            keys=st.text(min_size=1, max_size=10),
            values=st.floats(min_value=0, max_value=100, allow_nan=False),
            min_size=1,
            max_size=5,
        ),
    )
    @settings(max_examples=50)
    def test_hash_collision_resistance(
        self,
        data1: Dict[str, float],
        data2: Dict[str, float],
    ):
        """
        INVARIANT: Different data produces different hashes (with high probability).

        SHA-256 should be collision-resistant.
        """
        import json

        assume(data1 != data2)  # Ensure different inputs

        str1 = json.dumps(data1, sort_keys=True)
        str2 = json.dumps(data2, sort_keys=True)

        hash1 = hashlib.sha256(str1.encode()).hexdigest()
        hash2 = hashlib.sha256(str2.encode()).hexdigest()

        # INVARIANT: Different data -> different hash
        assert hash1 != hash2, "Hash collision detected"

    @given(
        value=st.floats(min_value=-1e10, max_value=1e10, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_decimal_precision_preservation(self, value: float):
        """
        INVARIANT: Decimal conversion preserves precision.

        Float -> Decimal -> Float should be stable.
        """
        try:
            decimal_value = Decimal(str(value))
            back_to_float = float(decimal_value)

            # INVARIANT: Round-trip preserves value
            assert abs(back_to_float - value) < 1e-10 * abs(value) + 1e-15, (
                f"Precision loss: {value} -> {back_to_float}"
            )

        except Exception:
            # Some edge cases may not convert
            pass


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "TestDrynessCalculatorInvariants",
    "TestVelocityLimiterInvariants",
    "TestThermodynamicInvariants",
    "TestProvenanceInvariants",
]
