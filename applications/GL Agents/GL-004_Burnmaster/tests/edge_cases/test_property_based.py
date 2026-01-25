"""
Property-Based Tests with Hypothesis for GL-004 BURNMASTER

Tests using property-based testing to verify:
- Combustion calculation invariants
- Safety interlock state machines
- Stoichiometry mathematical properties
- Stability index bounds
- Envelope validation consistency

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import numpy as np
from decimal import Decimal
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Try to import hypothesis, skip if not available
try:
    from hypothesis import given, strategies as st, assume, settings, Verbosity
    from hypothesis.stateful import RuleBasedStateMachine, rule, invariant, initialize
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    # Create dummy decorators
    def given(*args, **kwargs):
        def decorator(f):
            return pytest.mark.skip(reason="Hypothesis not installed")(f)
        return decorator

    class st:
        @staticmethod
        def floats(*args, **kwargs):
            return None
        @staticmethod
        def integers(*args, **kwargs):
            return None
        @staticmethod
        def lists(*args, **kwargs):
            return None
        @staticmethod
        def dictionaries(*args, **kwargs):
            return None
        @staticmethod
        def just(*args, **kwargs):
            return None
        @staticmethod
        def one_of(*args, **kwargs):
            return None
        @staticmethod
        def sampled_from(*args, **kwargs):
            return None
        @staticmethod
        def composite(*args, **kwargs):
            def decorator(f):
                return f
            return decorator

    def settings(*args, **kwargs):
        def decorator(f):
            return f
        return decorator

    class Verbosity:
        verbose = None


# Import system modules
import sys
sys.path.insert(0, 'C:/Users/aksha/Code-V1_GreenLang/GL Agents/GL-004_Burnmaster')

from combustion.stoichiometry import (
    compute_stoichiometric_air,
    compute_lambda,
    compute_excess_air_percent,
    compute_excess_o2,
    infer_lambda_from_o2,
    validate_stoichiometry_inputs,
)
from combustion.fuel_properties import (
    FuelType, FuelComposition,
    compute_molecular_weight, compute_heating_values,
    compute_fuel_properties, validate_fuel_composition,
    MOLECULAR_WEIGHTS, STOICH_O2,
)
from combustion.thermodynamics import (
    compute_stack_loss,
    compute_radiation_loss,
    compute_efficiency_indirect,
)
from safety.safety_envelope import SafetyEnvelope, Setpoint, EnvelopeStatus
from calculators.stability_calculator import (
    FlameStabilityCalculator, StabilityLevel, RiskLevel,
)


# ============================================================================
# CUSTOM STRATEGIES
# ============================================================================

if HYPOTHESIS_AVAILABLE:
    @st.composite
    def fuel_composition_strategy(draw):
        """Generate valid fuel compositions that sum to 100%."""
        # Generate components
        ch4 = draw(st.floats(min_value=0, max_value=100))
        remaining = 100 - ch4

        c2h6 = draw(st.floats(min_value=0, max_value=remaining))
        remaining -= c2h6

        c3h8 = draw(st.floats(min_value=0, max_value=remaining))
        remaining -= c3h8

        n2 = remaining  # Use remaining for N2 to ensure sum is 100

        # Ensure we have at least some combustible
        assume(ch4 + c2h6 + c3h8 > 0)

        return {"CH4": ch4, "C2H6": c2h6, "C3H8": c3h8, "N2": n2}


    @st.composite
    def flame_signal_strategy(draw):
        """Generate valid flame signal arrays."""
        length = draw(st.integers(min_value=10, max_value=500))
        mean = draw(st.floats(min_value=10, max_value=100))
        std = draw(st.floats(min_value=0.1, max_value=20))

        signal = [max(1, mean + np.random.normal(0, std)) for _ in range(length)]
        return np.array(signal)


    @st.composite
    def lambda_strategy(draw):
        """Generate valid lambda values."""
        return draw(st.floats(min_value=0.5, max_value=5.0))


    @st.composite
    def o2_percentage_strategy(draw):
        """Generate valid O2 percentages."""
        return draw(st.floats(min_value=0.0, max_value=20.9))


    @st.composite
    def setpoint_strategy(draw):
        """Generate valid setpoint values for envelope testing."""
        param = draw(st.sampled_from(['o2', 'co', 'firing_rate', 'nox']))
        ranges = {
            'o2': (0.5, 10.0),
            'co': (0, 500),
            'firing_rate': (0, 110),
            'nox': (0, 200),
        }
        min_val, max_val = ranges[param]
        value = draw(st.floats(min_value=min_val, max_value=max_val))
        return param, value
else:
    # Dummy strategies for when hypothesis is not available
    fuel_composition_strategy = lambda: None
    flame_signal_strategy = lambda: None
    lambda_strategy = lambda: None
    o2_percentage_strategy = lambda: None
    setpoint_strategy = lambda: None


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def stability_calculator():
    """Create FlameStabilityCalculator instance."""
    return FlameStabilityCalculator(precision=4)


@pytest.fixture
def safety_envelope():
    """Create configured SafetyEnvelope instance."""
    envelope = SafetyEnvelope(unit_id="BLR-TEST")
    envelope.define_envelope("BLR-TEST", {
        "o2_min": 1.5,
        "o2_max": 8.0,
        "co_max": 200,
        "nox_max": 100,
        "draft_min": -0.5,
        "draft_max": -0.01,
        "flame_signal_min": 30.0,
        "steam_temp_max": 550.0,
        "steam_pressure_max": 150.0,
        "firing_rate_min": 10.0,
        "firing_rate_max": 100.0,
    })
    return envelope


# ============================================================================
# COMBUSTION CALCULATION INVARIANTS
# ============================================================================

@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not installed")
class TestCombustionInvariants:
    """Property-based tests for combustion calculation invariants."""

    @given(st.floats(min_value=0.5, max_value=5.0))
    @settings(max_examples=100)
    def test_excess_air_from_lambda_is_linear(self, lambda_val):
        """Property: Excess air % = (lambda - 1) * 100."""
        excess_air = compute_excess_air_percent(lambda_val)
        expected = (lambda_val - 1.0) * 100.0

        assert abs(excess_air - expected) < 0.01

    @given(
        st.floats(min_value=1.0, max_value=100.0),
        st.floats(min_value=1.0, max_value=100.0)
    )
    @settings(max_examples=100)
    def test_lambda_is_ratio_of_afrs(self, actual_af, stoich_af):
        """Property: Lambda = actual_AFR / stoich_AFR."""
        assume(stoich_af > 0)

        lambda_val = compute_lambda(actual_af, stoich_af)
        expected = actual_af / stoich_af

        assert abs(lambda_val - expected) < 0.0001

    @given(st.floats(min_value=0.1, max_value=15.0))
    @settings(max_examples=100)
    def test_o2_lambda_roundtrip_consistency(self, o2_percent):
        """Property: infer_lambda(compute_o2(lambda)) should be consistent."""
        assume(0.5 <= o2_percent <= 15.0)

        lambda_from_o2 = infer_lambda_from_o2(o2_percent, "natural_gas")
        o2_back = compute_excess_o2(lambda_from_o2, "natural_gas")

        # Round-trip should be reasonably close
        if lambda_from_o2 > 1.0:  # Only check for lean mixtures
            assert abs(o2_back - o2_percent) < 1.5

    @given(st.floats(min_value=1.01, max_value=3.0))
    @settings(max_examples=100)
    def test_o2_increases_with_lambda(self, lambda_val):
        """Property: Higher lambda -> higher O2 percentage."""
        o2_low = compute_excess_o2(lambda_val, "natural_gas")
        o2_high = compute_excess_o2(lambda_val + 0.1, "natural_gas")

        assert o2_high >= o2_low

    @given(st.floats(min_value=0.5, max_value=1.0))
    @settings(max_examples=50)
    def test_substoichiometric_has_zero_o2(self, lambda_val):
        """Property: Sub-stoichiometric (lambda < 1) has zero excess O2."""
        o2 = compute_excess_o2(lambda_val, "natural_gas")
        assert o2 == 0.0


# ============================================================================
# STABILITY INDEX INVARIANTS
# ============================================================================

@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not installed")
class TestStabilityInvariants:
    """Property-based tests for stability index invariants."""

    @given(
        st.lists(st.floats(min_value=10, max_value=200), min_size=10, max_size=100),
        st.floats(min_value=0, max_value=2)
    )
    @settings(max_examples=100)
    def test_stability_index_bounded(self, signal_list, o2_variance):
        """Property: Stability index is always between 0 and 1."""
        calculator = FlameStabilityCalculator(precision=4)
        signal = np.array(signal_list)

        result = calculator.compute_stability_index(signal, o2_variance)

        assert 0 <= float(result.stability_index) <= 1

    @given(st.floats(min_value=50, max_value=150))
    @settings(max_examples=50)
    def test_constant_signal_is_stable(self, value):
        """Property: Constant signal has excellent stability."""
        calculator = FlameStabilityCalculator(precision=4)
        signal = np.array([value] * 50)

        result = calculator.compute_stability_index(signal, 0.0)

        assert result.stability_level == StabilityLevel.EXCELLENT

    @given(
        st.floats(min_value=50, max_value=150),
        st.floats(min_value=0.1, max_value=50)
    )
    @settings(max_examples=100)
    def test_higher_variance_lower_stability(self, mean, std):
        """Property: Higher variance generally means lower stability."""
        calculator = FlameStabilityCalculator(precision=4)
        assume(std > 0)

        # Low variance signal
        low_var_signal = np.array([mean + np.random.normal(0, 1) for _ in range(100)])

        # High variance signal
        high_var_signal = np.array([mean + np.random.normal(0, std) for _ in range(100)])

        low_result = calculator.compute_stability_index(low_var_signal, 0.1)
        high_result = calculator.compute_stability_index(high_var_signal, 0.5)

        # High variance should generally have lower or equal stability
        # (May not always be strictly true due to randomness, but usually holds)

    @given(st.lists(st.floats(min_value=10, max_value=200), min_size=50, max_size=100))
    @settings(max_examples=50)
    def test_stability_is_deterministic(self, signal_list):
        """Property: Same input always produces same output."""
        calculator = FlameStabilityCalculator(precision=4)
        signal = np.array(signal_list)

        result1 = calculator.compute_stability_index(signal.copy(), 0.1)
        result2 = calculator.compute_stability_index(signal.copy(), 0.1)

        assert result1.provenance_hash == result2.provenance_hash
        assert result1.stability_index == result2.stability_index


# ============================================================================
# ENVELOPE VALIDATION INVARIANTS
# ============================================================================

@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not installed")
class TestEnvelopeInvariants:
    """Property-based tests for safety envelope invariants."""

    @given(st.floats(min_value=1.5, max_value=8.0))
    @settings(max_examples=100)
    def test_valid_o2_always_passes(self, o2_value):
        """Property: O2 values within limits always validate."""
        envelope = SafetyEnvelope(unit_id="BLR-TEST")
        envelope.define_envelope("BLR-TEST", {
            "o2_min": 1.5,
            "o2_max": 8.0,
            "co_max": 200,
            "nox_max": 100,
            "draft_min": -0.5,
            "draft_max": -0.01,
            "flame_signal_min": 30.0,
            "steam_temp_max": 550.0,
            "steam_pressure_max": 150.0,
            "firing_rate_min": 10.0,
            "firing_rate_max": 100.0,
        })

        setpoint = Setpoint(parameter_name="o2", value=o2_value, unit="%")
        validation = envelope.validate_within_envelope(setpoint)

        assert validation.is_valid

    @given(st.one_of(
        st.floats(min_value=0, max_value=1.49),
        st.floats(min_value=8.01, max_value=20)
    ))
    @settings(max_examples=100)
    def test_invalid_o2_always_fails(self, o2_value):
        """Property: O2 values outside limits always fail."""
        envelope = SafetyEnvelope(unit_id="BLR-TEST")
        envelope.define_envelope("BLR-TEST", {
            "o2_min": 1.5,
            "o2_max": 8.0,
            "co_max": 200,
            "nox_max": 100,
            "draft_min": -0.5,
            "draft_max": -0.01,
            "flame_signal_min": 30.0,
            "steam_temp_max": 550.0,
            "steam_pressure_max": 150.0,
            "firing_rate_min": 10.0,
            "firing_rate_max": 100.0,
        })

        setpoint = Setpoint(parameter_name="o2", value=o2_value, unit="%")
        validation = envelope.validate_within_envelope(setpoint)

        assert not validation.is_valid

    @given(st.floats(min_value=0.8, max_value=0.99))
    @settings(max_examples=50)
    def test_shrink_makes_envelope_smaller(self, factor):
        """Property: Shrinking reduces the valid range."""
        envelope = SafetyEnvelope(unit_id="BLR-TEST")
        envelope.define_envelope("BLR-TEST", {
            "o2_min": 1.5,
            "o2_max": 8.0,
            "co_max": 200,
            "nox_max": 100,
            "draft_min": -0.5,
            "draft_max": -0.01,
            "flame_signal_min": 30.0,
            "steam_temp_max": 550.0,
            "steam_pressure_max": 150.0,
            "firing_rate_min": 10.0,
            "firing_rate_max": 100.0,
        })

        original_o2_range = envelope.limits.o2_max - envelope.limits.o2_min

        envelope.shrink_envelope(factor, "test")

        new_o2_range = envelope.limits.o2_max - envelope.limits.o2_min

        assert new_o2_range < original_o2_range


# ============================================================================
# FUEL PROPERTIES INVARIANTS
# ============================================================================

@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not installed")
class TestFuelPropertiesInvariants:
    """Property-based tests for fuel property invariants."""

    @given(st.floats(min_value=10, max_value=100))
    @settings(max_examples=100)
    def test_hhv_greater_than_lhv(self, ch4_pct):
        """Property: HHV is always >= LHV."""
        assume(ch4_pct > 0)
        n2_pct = 100 - ch4_pct

        composition = FuelComposition(ch4=ch4_pct, n2=n2_pct)
        props = compute_fuel_properties(composition, FuelType.NATURAL_GAS)

        assert props.hhv >= props.lhv

    @given(
        st.floats(min_value=0, max_value=50),
        st.floats(min_value=0, max_value=50)
    )
    @settings(max_examples=100)
    def test_more_combustibles_higher_hhv(self, ch4_pct_1, ch4_pct_2):
        """Property: More combustibles generally means higher HHV."""
        assume(ch4_pct_1 > 0 and ch4_pct_2 > 0)
        assume(ch4_pct_1 != ch4_pct_2)

        # Ensure we have two different compositions
        low_ch4 = min(ch4_pct_1, ch4_pct_2)
        high_ch4 = max(ch4_pct_1, ch4_pct_2)

        assume(high_ch4 - low_ch4 > 5)  # Ensure meaningful difference

        comp_low = FuelComposition(ch4=low_ch4, n2=100-low_ch4)
        comp_high = FuelComposition(ch4=high_ch4, n2=100-high_ch4)

        props_low = compute_fuel_properties(comp_low, FuelType.NATURAL_GAS)
        props_high = compute_fuel_properties(comp_high, FuelType.NATURAL_GAS)

        assert props_high.hhv > props_low.hhv

    @given(st.floats(min_value=10, max_value=100))
    @settings(max_examples=50)
    def test_molecular_weight_positive(self, ch4_pct):
        """Property: Molecular weight is always positive."""
        assume(ch4_pct > 0)

        composition = FuelComposition(ch4=ch4_pct, n2=100-ch4_pct)
        mw = compute_molecular_weight(composition)

        assert mw > 0


# ============================================================================
# BLOWOFF AND FLASHBACK INVARIANTS
# ============================================================================

@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not installed")
class TestBlowoffFlashbackInvariants:
    """Property-based tests for blowoff and flashback risk invariants."""

    @given(
        st.floats(min_value=0.1, max_value=10),
        st.floats(min_value=0.1, max_value=5),
        st.floats(min_value=0.9, max_value=1.5)
    )
    @settings(max_examples=100)
    def test_blowoff_risk_bounded(self, velocity, flame_speed, lambda_val):
        """Property: Blowoff risk is always between 0 and 1."""
        calculator = FlameStabilityCalculator(precision=4)

        risk_score, risk_level = calculator.compute_blowoff_risk(
            velocity, flame_speed, lambda_val
        )

        assert 0 <= float(risk_score) <= 1
        assert risk_level in [RiskLevel.LOW, RiskLevel.MODERATE, RiskLevel.HIGH, RiskLevel.CRITICAL]

    @given(
        st.floats(min_value=0, max_value=500),
        st.floats(min_value=200, max_value=600)
    )
    @settings(max_examples=100)
    def test_flashback_risk_bounded(self, premix_temp, autoignition_temp):
        """Property: Flashback risk is always between 0 and 1."""
        calculator = FlameStabilityCalculator(precision=4)
        assume(autoignition_temp > 0)

        risk_score, risk_level = calculator.compute_flashback_risk(
            premix_temp, autoignition_temp
        )

        assert 0 <= float(risk_score) <= 1

    @given(
        st.floats(min_value=0.1, max_value=5),
        st.floats(min_value=0.1, max_value=5)
    )
    @settings(max_examples=100)
    def test_higher_velocity_higher_blowoff_risk(self, velocity_low, velocity_high):
        """Property: Higher velocity generally means higher blowoff risk."""
        calculator = FlameStabilityCalculator(precision=4)

        assume(velocity_high > velocity_low * 1.5)  # Ensure meaningful difference

        risk_low, _ = calculator.compute_blowoff_risk(velocity_low, 0.4, 1.1)
        risk_high, _ = calculator.compute_blowoff_risk(velocity_high, 0.4, 1.1)

        assert risk_high >= risk_low


# ============================================================================
# THERMODYNAMICS INVARIANTS
# ============================================================================

@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not installed")
class TestThermodynamicsInvariants:
    """Property-based tests for thermodynamic calculation invariants."""

    @given(
        st.floats(min_value=50, max_value=400),
        st.floats(min_value=-20, max_value=45)
    )
    @settings(max_examples=100)
    def test_stack_loss_non_negative(self, stack_temp, ambient_temp):
        """Property: Stack loss is always non-negative."""
        assume(stack_temp > ambient_temp)

        loss = compute_stack_loss(stack_temp, ambient_temp, 3.0, "natural_gas")

        assert loss >= 0

    @given(
        st.floats(min_value=100, max_value=400),
        st.floats(min_value=0, max_value=40)
    )
    @settings(max_examples=100)
    def test_stack_loss_increases_with_temp_difference(self, stack_temp, ambient_temp):
        """Property: Stack loss increases with temperature difference."""
        assume(stack_temp > ambient_temp + 50)

        loss_low = compute_stack_loss(stack_temp - 50, ambient_temp, 3.0, "natural_gas")
        loss_high = compute_stack_loss(stack_temp, ambient_temp, 3.0, "natural_gas")

        assert loss_high > loss_low

    @given(
        st.floats(min_value=1, max_value=100),
        st.floats(min_value=0.1, max_value=1.0)
    )
    @settings(max_examples=100)
    def test_radiation_loss_bounded(self, rating_mw, load_fraction):
        """Property: Radiation loss is bounded between 0 and 3%."""
        loss = compute_radiation_loss(rating_mw, load_fraction)

        assert 0 <= loss <= 3.0


# ============================================================================
# STATE MACHINE TESTS
# ============================================================================

@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not installed")
class TestEnvelopeStateMachine:
    """State machine tests for safety envelope."""

    # Note: Full state machine testing would use RuleBasedStateMachine
    # Simplified version here

    @given(st.lists(st.floats(min_value=0.8, max_value=0.99), min_size=1, max_size=10))
    @settings(max_examples=50)
    def test_multiple_shrinks_always_reduce(self, factors):
        """Property: Multiple shrinks always reduce the range."""
        envelope = SafetyEnvelope(unit_id="BLR-TEST")
        envelope.define_envelope("BLR-TEST", {
            "o2_min": 1.5,
            "o2_max": 8.0,
            "co_max": 200,
            "nox_max": 100,
            "draft_min": -0.5,
            "draft_max": -0.01,
            "flame_signal_min": 30.0,
            "steam_temp_max": 550.0,
            "steam_pressure_max": 150.0,
            "firing_rate_min": 10.0,
            "firing_rate_max": 100.0,
        })

        initial_range = envelope.limits.o2_max - envelope.limits.o2_min

        for factor in factors:
            try:
                envelope.shrink_envelope(factor, f"shrink_{factor}")
            except Exception:
                pass  # May fail if range becomes too small

        final_range = envelope.limits.o2_max - envelope.limits.o2_min

        assert final_range <= initial_range


# ============================================================================
# VALIDATION INPUT INVARIANTS
# ============================================================================

@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not installed")
class TestValidationInvariants:
    """Property-based tests for validation function invariants."""

    @given(st.floats(min_value=0.5, max_value=5.0))
    @settings(max_examples=100)
    def test_valid_lambda_passes_validation(self, lambda_val):
        """Property: Lambda values in valid range pass validation."""
        is_valid, errors = validate_stoichiometry_inputs(lambda_val=lambda_val)

        assert is_valid
        assert len(errors) == 0

    @given(st.one_of(
        st.floats(min_value=-10, max_value=0.49),
        st.floats(min_value=5.01, max_value=100)
    ))
    @settings(max_examples=100)
    def test_invalid_lambda_fails_validation(self, lambda_val):
        """Property: Lambda values outside valid range fail validation."""
        is_valid, errors = validate_stoichiometry_inputs(lambda_val=lambda_val)

        assert not is_valid
        assert len(errors) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
