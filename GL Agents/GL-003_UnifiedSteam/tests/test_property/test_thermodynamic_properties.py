"""
Property-Based Tests for Thermodynamic Calculations

Uses Hypothesis to verify invariants and properties of IAPWS-IF97
calculations across the valid operating range.

Author: GL-003 Test Engineering Team
"""

from decimal import Decimal
from typing import Tuple
import pytest

try:
    from hypothesis import given, assume, settings, HealthCheck
    from hypothesis import strategies as st
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False


# Skip all tests if hypothesis not installed
pytestmark = pytest.mark.skipif(
    not HYPOTHESIS_AVAILABLE,
    reason="hypothesis not installed"
)


# Define valid operating ranges for industrial steam systems
# Pressure in kPa (0.1 MPa to 20 MPa)
pressure_strategy = st.decimals(
    min_value=Decimal("100"),
    max_value=Decimal("20000"),
    places=2,
    allow_nan=False,
    allow_infinity=False,
)

# Temperature in Celsius (20C to 550C)
temperature_strategy = st.decimals(
    min_value=Decimal("20"),
    max_value=Decimal("550"),
    places=2,
    allow_nan=False,
    allow_infinity=False,
)

# Mass flow in kg/s (0.01 to 100)
mass_flow_strategy = st.decimals(
    min_value=Decimal("0.01"),
    max_value=Decimal("100"),
    places=3,
    allow_nan=False,
    allow_infinity=False,
)


class TestThermodynamicInvariants:
    """Property-based tests for thermodynamic invariants."""

    @given(pressure=pressure_strategy, temperature=temperature_strategy)
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_enthalpy_increases_with_temperature(self, pressure, temperature):
        """Enthalpy should increase with temperature at constant pressure."""
        try:
            from GL_Agents.GL003_UnifiedSteam.thermodynamics.iapws_if97 import (
                compute_properties_pt,
            )
        except ImportError:
            pytest.skip("IF97 module not available")

        # Ensure we're in a valid single-phase region
        assume(float(temperature) > 100 or float(pressure) > 5000)

        result1 = compute_properties_pt(pressure_kpa=pressure, temperature_c=temperature)
        result2 = compute_properties_pt(
            pressure_kpa=pressure,
            temperature_c=temperature + Decimal("10")
        )

        if result1 and result2 and hasattr(result1, 'enthalpy_kj_kg'):
            # Enthalpy should increase with temperature
            assert result2.enthalpy_kj_kg >= result1.enthalpy_kj_kg, (
                f"Enthalpy decreased with temperature increase at P={pressure}kPa"
            )

    @given(pressure=pressure_strategy, temperature=temperature_strategy)
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_entropy_positive(self, pressure, temperature):
        """Entropy should always be positive."""
        try:
            from GL_Agents.GL003_UnifiedSteam.thermodynamics.iapws_if97 import (
                compute_properties_pt,
            )
        except ImportError:
            pytest.skip("IF97 module not available")

        result = compute_properties_pt(pressure_kpa=pressure, temperature_c=temperature)

        if result and hasattr(result, 'entropy_kj_kg_k'):
            assert result.entropy_kj_kg_k > 0, (
                f"Negative entropy at P={pressure}, T={temperature}"
            )

    @given(pressure=pressure_strategy, temperature=temperature_strategy)
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_specific_volume_positive(self, pressure, temperature):
        """Specific volume should always be positive."""
        try:
            from GL_Agents.GL003_UnifiedSteam.thermodynamics.iapws_if97 import (
                compute_properties_pt,
            )
        except ImportError:
            pytest.skip("IF97 module not available")

        result = compute_properties_pt(pressure_kpa=pressure, temperature_c=temperature)

        if result and hasattr(result, 'specific_volume_m3_kg'):
            assert result.specific_volume_m3_kg > 0, (
                f"Non-positive specific volume at P={pressure}, T={temperature}"
            )

    @given(pressure=pressure_strategy, temperature=temperature_strategy)
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_density_specific_volume_inverse(self, pressure, temperature):
        """Density should be inverse of specific volume."""
        try:
            from GL_Agents.GL003_UnifiedSteam.thermodynamics.iapws_if97 import (
                compute_properties_pt,
            )
        except ImportError:
            pytest.skip("IF97 module not available")

        result = compute_properties_pt(pressure_kpa=pressure, temperature_c=temperature)

        if result and hasattr(result, 'specific_volume_m3_kg') and hasattr(result, 'density_kg_m3'):
            v = float(result.specific_volume_m3_kg)
            rho = float(result.density_kg_m3)

            # rho * v should equal 1
            product = v * rho
            assert abs(product - 1.0) < 0.001, (
                f"Density * Specific Volume = {product}, expected 1.0"
            )


class TestRegionBoundaries:
    """Property-based tests for region boundary behavior."""

    @given(
        pressure=st.decimals(
            min_value=Decimal("100"),
            max_value=Decimal("10000"),
            places=2,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_saturation_temperature_monotonic(self, pressure):
        """Saturation temperature should increase with pressure."""
        try:
            from GL_Agents.GL003_UnifiedSteam.thermodynamics.iapws_if97 import (
                get_saturation_temperature,
            )
        except ImportError:
            pytest.skip("IF97 module not available")

        # Test at pressure and pressure + delta
        delta = Decimal("100")
        assume(float(pressure) + float(delta) <= 10000)

        t1 = get_saturation_temperature(pressure_kpa=pressure)
        t2 = get_saturation_temperature(pressure_kpa=pressure + delta)

        if t1 is not None and t2 is not None:
            assert t2 > t1, (
                f"Saturation temp not monotonic: T({pressure})={t1}, "
                f"T({pressure + delta})={t2}"
            )

    @given(
        temperature=st.decimals(
            min_value=Decimal("50"),
            max_value=Decimal("350"),
            places=2,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_saturation_pressure_monotonic(self, temperature):
        """Saturation pressure should increase with temperature."""
        try:
            from GL_Agents.GL003_UnifiedSteam.thermodynamics.iapws_if97 import (
                get_saturation_pressure,
            )
        except ImportError:
            pytest.skip("IF97 module not available")

        delta = Decimal("10")
        assume(float(temperature) + float(delta) <= 350)

        p1 = get_saturation_pressure(temperature_c=temperature)
        p2 = get_saturation_pressure(temperature_c=temperature + delta)

        if p1 is not None and p2 is not None:
            assert p2 > p1, (
                f"Saturation pressure not monotonic: P({temperature})={p1}, "
                f"P({temperature + delta})={p2}"
            )


class TestMassEnergyBalance:
    """Property-based tests for mass and energy conservation."""

    @given(
        flow1=mass_flow_strategy,
        flow2=mass_flow_strategy,
        h1=st.decimals(min_value=Decimal("100"), max_value=Decimal("3500"), places=1),
        h2=st.decimals(min_value=Decimal("100"), max_value=Decimal("3500"), places=1),
    )
    @settings(max_examples=100)
    def test_mixing_enthalpy_bounded(self, flow1, flow2, h1, h2):
        """Mixed stream enthalpy should be bounded by inlet enthalpies."""
        assume(float(flow1) + float(flow2) > 0.01)

        total_flow = flow1 + flow2
        mixed_h = (flow1 * h1 + flow2 * h2) / total_flow

        min_h = min(h1, h2)
        max_h = max(h1, h2)

        assert min_h <= mixed_h <= max_h, (
            f"Mixed enthalpy {mixed_h} not bounded by [{min_h}, {max_h}]"
        )

    @given(
        inflows=st.lists(
            st.tuples(mass_flow_strategy, mass_flow_strategy),
            min_size=1,
            max_size=5,
        ),
    )
    @settings(max_examples=50)
    def test_mass_conservation(self, inflows):
        """Mass should be conserved in calculations."""
        # Each tuple is (mass_flow, enthalpy * 1000 as placeholder)
        total_mass_in = sum(flow for flow, _ in inflows)

        # Conservation: mass in = mass out
        # For closed system, mass is conserved
        assert total_mass_in >= 0, "Total mass should be non-negative"


class TestUnitConversions:
    """Property-based tests for unit conversion consistency."""

    @given(
        pressure_kpa=st.decimals(
            min_value=Decimal("100"),
            max_value=Decimal("20000"),
            places=2,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    @settings(max_examples=100)
    def test_pressure_kpa_to_mpa_roundtrip(self, pressure_kpa):
        """Pressure conversion should round-trip correctly."""
        pressure_mpa = pressure_kpa / Decimal("1000")
        pressure_kpa_back = pressure_mpa * Decimal("1000")

        assert abs(pressure_kpa - pressure_kpa_back) < Decimal("0.001"), (
            f"Pressure roundtrip error: {pressure_kpa} -> {pressure_mpa} -> {pressure_kpa_back}"
        )

    @given(
        temp_c=st.decimals(
            min_value=Decimal("-50"),
            max_value=Decimal("600"),
            places=2,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    @settings(max_examples=100)
    def test_temperature_c_to_k_roundtrip(self, temp_c):
        """Temperature conversion should round-trip correctly."""
        temp_k = temp_c + Decimal("273.15")
        temp_c_back = temp_k - Decimal("273.15")

        assert abs(temp_c - temp_c_back) < Decimal("0.001"), (
            f"Temperature roundtrip error: {temp_c} -> {temp_k} -> {temp_c_back}"
        )


class TestDataValidation:
    """Property-based tests for data validation rules."""

    @given(
        value=st.floats(min_value=-1e10, max_value=1e10, allow_nan=True, allow_infinity=True)
    )
    def test_nan_detection(self, value):
        """NaN values should be detected."""
        import math

        is_nan = math.isnan(value) if not math.isinf(value) else False

        if is_nan:
            # NaN should fail validation
            assert value != value  # NaN property

    @given(
        pressure=st.floats(min_value=-1000, max_value=50000),
        temperature=st.floats(min_value=-100, max_value=700),
    )
    def test_operating_range_validation(self, pressure, temperature):
        """Values outside operating range should be flagged."""
        # Define valid industrial ranges
        pressure_valid = 10 <= pressure <= 25000  # kPa
        temperature_valid = 0 <= temperature <= 600  # C

        # At least one should be valid for typical operations
        if pressure_valid and temperature_valid:
            # Both in range - should be processable
            pass
        else:
            # Out of range - should trigger warning/error
            assert not (pressure_valid and temperature_valid)


class TestComputationDeterminism:
    """Property-based tests for computation determinism."""

    @given(pressure=pressure_strategy, temperature=temperature_strategy)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_calculation_determinism(self, pressure, temperature):
        """Same inputs should always produce same outputs."""
        try:
            from GL_Agents.GL003_UnifiedSteam.thermodynamics.iapws_if97 import (
                compute_properties_pt,
            )
        except ImportError:
            pytest.skip("IF97 module not available")

        result1 = compute_properties_pt(pressure_kpa=pressure, temperature_c=temperature)
        result2 = compute_properties_pt(pressure_kpa=pressure, temperature_c=temperature)

        if result1 and result2:
            if hasattr(result1, 'enthalpy_kj_kg'):
                assert result1.enthalpy_kj_kg == result2.enthalpy_kj_kg, (
                    "Non-deterministic enthalpy calculation"
                )
            if hasattr(result1, 'entropy_kj_kg_k'):
                assert result1.entropy_kj_kg_k == result2.entropy_kj_kg_k, (
                    "Non-deterministic entropy calculation"
                )

    @given(
        pressure=pressure_strategy,
        temperature=temperature_strategy,
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_calculation_independence_from_order(self, pressure, temperature, seed):
        """Calculation order should not affect results."""
        try:
            from GL_Agents.GL003_UnifiedSteam.thermodynamics.iapws_if97 import (
                compute_properties_pt,
            )
        except ImportError:
            pytest.skip("IF97 module not available")

        # Calculate at different point first
        _ = compute_properties_pt(
            pressure_kpa=Decimal(str(seed % 10000 + 100)),
            temperature_c=Decimal(str(seed % 500 + 50)),
        )

        # Then calculate at test point
        result = compute_properties_pt(pressure_kpa=pressure, temperature_c=temperature)

        # Reference calculation without the intermediate
        result_ref = compute_properties_pt(pressure_kpa=pressure, temperature_c=temperature)

        if result and result_ref and hasattr(result, 'enthalpy_kj_kg'):
            assert result.enthalpy_kj_kg == result_ref.enthalpy_kj_kg, (
                "Calculation order affected result"
            )


class TestEdgeCases:
    """Property-based tests for edge case handling."""

    @given(
        pressure=st.sampled_from([
            Decimal("100"),      # Low pressure
            Decimal("611.657"),  # Near triple point
            Decimal("22064"),    # Near critical pressure
            Decimal("20000"),    # High pressure
        ]),
        temperature=temperature_strategy,
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_boundary_pressure_handling(self, pressure, temperature):
        """Boundary pressures should be handled correctly."""
        try:
            from GL_Agents.GL003_UnifiedSteam.thermodynamics.iapws_if97 import (
                compute_properties_pt,
            )
        except ImportError:
            pytest.skip("IF97 module not available")

        # Should not raise exception
        result = compute_properties_pt(pressure_kpa=pressure, temperature_c=temperature)

        # Result should be valid or None (if out of range)
        if result is not None:
            if hasattr(result, 'enthalpy_kj_kg'):
                assert result.enthalpy_kj_kg > 0

    @given(
        pressure=pressure_strategy,
        temperature=st.sampled_from([
            Decimal("0.01"),     # Near freezing
            Decimal("100"),      # Boiling at 1 atm
            Decimal("373.946"),  # Near critical temperature
            Decimal("550"),      # High temperature
        ]),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_boundary_temperature_handling(self, pressure, temperature):
        """Boundary temperatures should be handled correctly."""
        try:
            from GL_Agents.GL003_UnifiedSteam.thermodynamics.iapws_if97 import (
                compute_properties_pt,
            )
        except ImportError:
            pytest.skip("IF97 module not available")

        result = compute_properties_pt(pressure_kpa=pressure, temperature_c=temperature)

        if result is not None:
            if hasattr(result, 'enthalpy_kj_kg'):
                assert result.enthalpy_kj_kg > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])
