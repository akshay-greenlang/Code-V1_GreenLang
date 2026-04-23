"""
Boundary Condition Edge Case Tests for GL-004 BURNMASTER

Tests system behavior at operational boundaries:
- Minimum/maximum firing rates
- Fuel switchover scenarios
- Hydrogen blend limits (0-30% and beyond)
- Near-flameout conditions
- Turndown ratio limits
- Regulatory emission limits

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import numpy as np
from decimal import Decimal
from datetime import datetime
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass

# Import system modules
import sys
sys.path.insert(0, 'C:/Users/aksha/Code-V1_GreenLang/GL Agents/GL-004_Burnmaster')

from combustion.stoichiometry import (
    compute_stoichiometric_air,
    compute_lambda,
    compute_excess_air_percent,
    compute_excess_o2,
    infer_lambda_from_o2,
    compute_air_flow_for_target_o2,
    compute_fuel_flow_for_target_duty,
    validate_stoichiometry_inputs,
)
from combustion.fuel_properties import (
    FuelType, FuelComposition, FuelProperties,
    compute_molecular_weight, compute_heating_values,
    compute_fuel_properties, get_fuel_properties,
    validate_fuel_composition, validate_fuel_properties,
    STANDARD_COMPOSITIONS,
)
from combustion.thermodynamics import (
    compute_stack_loss,
    compute_radiation_loss,
    compute_efficiency_indirect,
    compute_efficiency_direct,
    compute_heat_balance,
)
from safety.safety_envelope import SafetyEnvelope, Setpoint, EnvelopeStatus
from calculators.stability_calculator import (
    FlameStabilityCalculator, StabilityLevel, RiskLevel,
)


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
# FIRING RATE BOUNDARY TESTS
# ============================================================================

class TestFiringRateBoundaries:
    """Test suite for firing rate boundaries."""

    @pytest.mark.parametrize("firing_rate,expected_valid", [
        (0.0, False),     # Below minimum
        (5.0, False),     # Below minimum
        (10.0, True),     # At minimum (boundary)
        (10.1, True),     # Just above minimum
        (50.0, True),     # Mid-range
        (99.9, True),     # Just below maximum
        (100.0, True),    # At maximum (boundary)
        (100.1, False),   # Just above maximum
        (110.0, False),   # Above maximum (overfire)
    ])
    def test_firing_rate_boundary_validation(
        self,
        safety_envelope,
        firing_rate: float,
        expected_valid: bool
    ):
        """Test firing rate validation at boundaries."""
        setpoint = Setpoint(
            parameter_name="firing_rate",
            value=firing_rate,
            unit="%"
        )

        validation = safety_envelope.validate_within_envelope(setpoint)

        assert validation.is_valid == expected_valid, \
            f"Firing rate {firing_rate}%: expected valid={expected_valid}"

    def test_minimum_turndown_ratio(self, stability_calculator):
        """Test stability at minimum turndown (10:1 typical)."""
        # At 10% firing rate (10:1 turndown)
        weak_flame_signal = np.array([40.0 + np.random.normal(0, 5) for _ in range(50)])

        result = stability_calculator.compute_stability_index(weak_flame_signal, 0.5)

        # Should still be acceptable at minimum turndown
        assert result.stability_level in [
            StabilityLevel.EXCELLENT, StabilityLevel.GOOD, StabilityLevel.MARGINAL
        ]

    def test_extreme_turndown_instability(self, stability_calculator):
        """Test detection of instability at extreme turndown."""
        # At 5% (beyond typical turndown limit)
        very_weak_signal = np.array([20.0 + np.random.normal(0, 8) for _ in range(50)])
        very_weak_signal = np.maximum(very_weak_signal, 5.0)

        result = stability_calculator.compute_stability_index(very_weak_signal, 0.8)

        # Should detect poor stability
        assert result.stability_level in [StabilityLevel.MARGINAL, StabilityLevel.POOR, StabilityLevel.CRITICAL]
        assert any("low flame" in rec.lower() or "variability" in rec.lower()
                   for rec in result.recommendations)

    @pytest.mark.parametrize("load_pct,expected_radiation_loss_range", [
        (10, (2.0, 3.0)),    # Very low load - high relative radiation loss
        (25, (1.5, 2.5)),    # Low load
        (50, (1.0, 2.0)),    # Half load
        (75, (0.7, 1.5)),    # High load
        (100, (0.5, 1.2)),   # Full load - lowest relative radiation loss
    ])
    def test_radiation_loss_vs_load(
        self,
        load_pct: int,
        expected_radiation_loss_range: Tuple[float, float]
    ):
        """Test radiation loss percentage varies with load."""
        load_fraction = load_pct / 100.0

        radiation_loss = compute_radiation_loss(
            furnace_rating_mw=20.0,
            load_fraction=load_fraction
        )

        min_loss, max_loss = expected_radiation_loss_range
        assert min_loss <= radiation_loss <= max_loss, \
            f"At {load_pct}% load, radiation loss {radiation_loss}% outside expected range"


# ============================================================================
# FUEL SWITCHOVER TESTS
# ============================================================================

class TestFuelSwitchover:
    """Test suite for fuel switchover scenarios."""

    def test_natural_gas_to_propane_switchover(self):
        """Test calculations during natural gas to propane switchover."""
        ng_props = get_fuel_properties(FuelType.NATURAL_GAS)
        propane_props = get_fuel_properties(FuelType.PROPANE)

        # Key differences that must be accounted for:
        assert propane_props.hhv > ng_props.hhv, "Propane has higher HHV"
        assert propane_props.stoichiometric_afr_vol > ng_props.stoichiometric_afr_vol, \
            "Propane requires more air"

        # Wobbe index check (interchangeability)
        ng_wobbe = ng_props.hhv / np.sqrt(ng_props.specific_gravity)
        propane_wobbe = propane_props.hhv / np.sqrt(propane_props.specific_gravity)

        # Wobbe indices should be reasonably close for interchangeability
        wobbe_diff_pct = abs(ng_wobbe - propane_wobbe) / ng_wobbe * 100
        assert wobbe_diff_pct < 20, f"Wobbe index difference {wobbe_diff_pct:.1f}% may cause issues"

    def test_fuel_blend_transition(self):
        """Test gradual transition between fuel compositions."""
        # Start: 100% natural gas
        # End: Natural gas + 30% hydrogen blend

        blend_steps = [0, 10, 20, 30]  # H2 percentages
        properties = []

        for h2_pct in blend_steps:
            ch4_pct = 100 - h2_pct
            composition = FuelComposition(ch4=ch4_pct, h2=h2_pct)
            props = compute_fuel_properties(composition, FuelType.HYDROGEN_BLEND)
            properties.append(props)

        # Heating value should decrease with H2 addition (on volume basis)
        for i in range(len(properties) - 1):
            assert properties[i+1].hhv < properties[i].hhv, \
                "HHV should decrease with H2 addition"

        # Flame speed should increase with H2 addition
        for i in range(len(properties) - 1):
            assert properties[i+1].flame_speed > properties[i].flame_speed, \
                "Flame speed should increase with H2 addition"

    def test_dual_fuel_capability(self):
        """Test calculations for dual-fuel operation."""
        # Primary: Natural gas
        # Backup: Fuel Oil #2

        ng_props = get_fuel_properties(FuelType.NATURAL_GAS)

        # Simulate fuel oil composition (liquid fuel approximation)
        # Using propane as proxy for liquid fuel behavior
        backup_props = get_fuel_properties(FuelType.PROPANE)

        # Calculate required air flow adjustment for switchover
        air_ratio_change = backup_props.stoichiometric_afr_vol / ng_props.stoichiometric_afr_vol

        # Should need different air for same heat output
        assert air_ratio_change != 1.0

    def test_fuel_quality_variation_impact(self):
        """Test impact of fuel quality variations on combustion."""
        # Normal natural gas
        normal_ng = FuelComposition(ch4=94.0, c2h6=3.0, c3h8=1.0, n2=2.0)

        # Low quality (high inerts)
        low_quality = FuelComposition(ch4=88.0, c2h6=2.0, n2=8.0, co2=2.0)

        # High quality (more heavy hydrocarbons)
        high_quality = FuelComposition(ch4=90.0, c2h6=5.0, c3h8=3.0, n2=2.0)

        normal_props = compute_fuel_properties(normal_ng, FuelType.NATURAL_GAS)
        low_props = compute_fuel_properties(low_quality, FuelType.NATURAL_GAS)
        high_props = compute_fuel_properties(high_quality, FuelType.NATURAL_GAS)

        # Low quality should have lower HHV
        assert low_props.hhv < normal_props.hhv

        # High quality should have higher HHV
        assert high_props.hhv > normal_props.hhv


# ============================================================================
# HYDROGEN BLEND LIMIT TESTS
# ============================================================================

class TestHydrogenBlendLimits:
    """Test suite for hydrogen blend limit scenarios."""

    @pytest.mark.parametrize("h2_percent", [0, 5, 10, 15, 20, 25, 30])
    def test_hydrogen_blend_within_limit(self, h2_percent: int):
        """Test hydrogen blends within typical 0-30% limit."""
        ch4_percent = 100 - h2_percent

        composition = FuelComposition(ch4=float(ch4_percent), h2=float(h2_percent))
        props = compute_fuel_properties(composition, FuelType.HYDROGEN_BLEND)

        # All properties should be valid
        is_valid, errors = validate_fuel_properties(props)
        assert is_valid, f"H2={h2_percent}%: {errors}"

        # Key safety checks
        assert props.flame_speed < 2.0, f"Flame speed {props.flame_speed} too high at {h2_percent}% H2"
        assert props.adiabatic_flame_temp < 2600, "AFT within reasonable range"

    @pytest.mark.parametrize("h2_percent", [40, 50, 60, 70, 80, 90, 100])
    def test_hydrogen_blend_beyond_typical_limit(self, h2_percent: int):
        """Test hydrogen blends beyond typical 30% limit."""
        ch4_percent = 100 - h2_percent

        if h2_percent == 100:
            composition = FuelComposition(h2=100.0)
        else:
            composition = FuelComposition(ch4=float(ch4_percent), h2=float(h2_percent))

        props = compute_fuel_properties(composition, FuelType.HYDROGEN_BLEND)

        # Properties should still be calculable
        assert props.hhv > 0
        assert props.lhv > 0

        # Higher H2 = higher flame speed (flashback risk)
        if h2_percent >= 50:
            assert props.flame_speed > 1.0, "High H2 blend should have elevated flame speed"

    def test_hydrogen_flashback_risk_assessment(self, stability_calculator):
        """Test flashback risk assessment for hydrogen blends."""
        # Pure methane autoignition
        ch4_autoignition = 580.0  # Celsius

        # Pure hydrogen autoignition
        h2_autoignition = 500.0  # Celsius

        # Test at various preheat temperatures
        test_cases = [
            (200.0, ch4_autoignition, RiskLevel.LOW),
            (300.0, ch4_autoignition, RiskLevel.LOW),
            (400.0, ch4_autoignition, RiskLevel.MODERATE),
            (200.0, h2_autoignition, RiskLevel.LOW),
            (350.0, h2_autoignition, RiskLevel.MODERATE),
            (450.0, h2_autoignition, RiskLevel.HIGH),
        ]

        for premix_temp, autoignition_temp, expected_min_risk in test_cases:
            risk_score, risk_level = stability_calculator.compute_flashback_risk(
                premix_temp, autoignition_temp
            )

            # Risk should be at least at expected level
            risk_order = [RiskLevel.LOW, RiskLevel.MODERATE, RiskLevel.HIGH, RiskLevel.CRITICAL]
            assert risk_order.index(risk_level) >= risk_order.index(expected_min_risk), \
                f"Preheat {premix_temp}C, autoignition {autoignition_temp}C: " \
                f"expected at least {expected_min_risk}, got {risk_level}"

    def test_hydrogen_blowoff_risk_assessment(self, stability_calculator):
        """Test blowoff risk assessment for hydrogen-rich fuels."""
        # Hydrogen has much higher flame speed
        h2_flame_speed = 3.10  # m/s
        ng_flame_speed = 0.38  # m/s

        # Same velocity, different risks
        velocity = 1.0  # m/s
        lambda_val = 1.15

        h2_risk, h2_level = stability_calculator.compute_blowoff_risk(
            velocity, h2_flame_speed, lambda_val
        )
        ng_risk, ng_level = stability_calculator.compute_blowoff_risk(
            velocity, ng_flame_speed, lambda_val
        )

        # H2 should have lower blowoff risk (higher flame speed)
        assert h2_risk < ng_risk, "H2 should have lower blowoff risk at same velocity"


# ============================================================================
# NEAR-FLAMEOUT CONDITION TESTS
# ============================================================================

class TestNearFlameoutConditions:
    """Test suite for near-flameout conditions."""

    def test_flame_signal_below_minimum(self, safety_envelope):
        """Test detection of flame signal below minimum."""
        setpoint = Setpoint(
            parameter_name="flame_signal",
            value=25.0,  # Below minimum of 30.0
            unit=""
        )

        validation = safety_envelope.validate_within_envelope(setpoint)

        assert not validation.is_valid
        assert "outside limits" in validation.blocking_reason.lower()

    def test_stability_at_flameout_boundary(self, stability_calculator):
        """Test stability calculation at flameout boundary."""
        # Flame signal hovering around minimum
        borderline_signal = np.array([35.0 + np.random.normal(0, 8) for _ in range(100)])
        borderline_signal = np.maximum(borderline_signal, 5.0)  # Prevent negative

        result = stability_calculator.compute_stability_index(borderline_signal, 0.6)

        # Should be marginal or worse
        assert result.stability_level in [
            StabilityLevel.MARGINAL, StabilityLevel.POOR, StabilityLevel.CRITICAL
        ]

    def test_lambda_at_lean_blowout_limit(self):
        """Test calculations at lean blowout limit."""
        # Typical lean blowout occurs around lambda = 1.8-2.0
        lean_limit_lambdas = [1.6, 1.8, 2.0, 2.2]

        for lambda_val in lean_limit_lambdas:
            excess_air = compute_excess_air_percent(lambda_val)
            o2 = compute_excess_o2(lambda_val, "natural_gas")

            # Excess air and O2 should be high
            assert excess_air >= 60, f"Lambda {lambda_val} should have high excess air"
            assert o2 >= 8, f"Lambda {lambda_val} should have high O2"

    def test_rich_mixture_incomplete_combustion(self):
        """Test calculations for rich (sub-stoichiometric) mixtures."""
        rich_lambdas = [0.95, 0.90, 0.85, 0.80]

        for lambda_val in rich_lambdas:
            excess_air = compute_excess_air_percent(lambda_val)
            o2 = compute_excess_o2(lambda_val, "natural_gas")

            # Rich mixture should have negative excess air
            assert excess_air < 0, f"Lambda {lambda_val} should have negative excess air"

            # No excess O2 (all consumed)
            assert o2 == 0, f"Lambda {lambda_val} should have zero excess O2"


# ============================================================================
# O2 BOUNDARY TESTS
# ============================================================================

class TestO2Boundaries:
    """Test suite for O2 setpoint boundaries."""

    @pytest.mark.parametrize("o2_value,expected_valid", [
        (0.5, False),     # Too low (dangerous)
        (1.0, False),     # Below minimum
        (1.5, True),      # At minimum
        (2.0, True),      # Low but valid
        (3.0, True),      # Optimal range
        (5.0, True),      # Higher but valid
        (8.0, True),      # At maximum
        (8.5, False),     # Above maximum
        (10.0, False),    # Way too high
    ])
    def test_o2_boundary_validation(
        self,
        safety_envelope,
        o2_value: float,
        expected_valid: bool
    ):
        """Test O2 setpoint validation at boundaries."""
        setpoint = Setpoint(
            parameter_name="o2",
            value=o2_value,
            unit="%"
        )

        validation = safety_envelope.validate_within_envelope(setpoint)

        assert validation.is_valid == expected_valid, \
            f"O2 {o2_value}%: expected valid={expected_valid}"

    def test_o2_lambda_consistency_at_boundaries(self):
        """Test O2-lambda relationship at boundaries."""
        boundary_o2_values = [0.5, 1.0, 2.0, 5.0, 10.0, 15.0]

        for o2 in boundary_o2_values:
            try:
                lambda_val = infer_lambda_from_o2(o2, "natural_gas")

                # Lambda should be >= 1.0 for any positive O2
                if o2 > 0.1:
                    assert lambda_val >= 1.0

                # Verify round-trip consistency
                if 1.0 < lambda_val < 5.0:
                    o2_back = compute_excess_o2(lambda_val, "natural_gas")
                    assert abs(o2_back - o2) < 1.0, "Round-trip O2 should be close"
            except ValueError:
                # Expected for out-of-range values
                pass


# ============================================================================
# EMISSION LIMIT TESTS
# ============================================================================

class TestEmissionLimits:
    """Test suite for emission limit boundaries."""

    @pytest.mark.parametrize("co_ppm,expected_valid", [
        (0, True),        # Perfect combustion
        (50, True),       # Normal
        (100, True),      # Elevated
        (150, True),      # High but within limit
        (200, True),      # At limit
        (201, False),     # Just over limit
        (500, False),     # Way over limit
    ])
    def test_co_emission_limits(
        self,
        safety_envelope,
        co_ppm: int,
        expected_valid: bool
    ):
        """Test CO emission limit validation."""
        setpoint = Setpoint(
            parameter_name="co",
            value=float(co_ppm),
            unit="ppm"
        )

        validation = safety_envelope.validate_within_envelope(setpoint)

        assert validation.is_valid == expected_valid

    @pytest.mark.parametrize("nox_ppm,expected_valid", [
        (0, True),
        (30, True),
        (50, True),
        (80, True),
        (100, True),      # At limit
        (101, False),     # Over limit
        (200, False),
    ])
    def test_nox_emission_limits(
        self,
        safety_envelope,
        nox_ppm: int,
        expected_valid: bool
    ):
        """Test NOx emission limit validation."""
        setpoint = Setpoint(
            parameter_name="nox",
            value=float(nox_ppm),
            unit="ppm"
        )

        validation = safety_envelope.validate_within_envelope(setpoint)

        assert validation.is_valid == expected_valid

    def test_co_increases_at_low_o2(self):
        """Test that CO typically increases at very low O2."""
        # This is a characteristic behavior - incomplete combustion at low O2
        # Lower O2 -> higher CO (typically)

        # Calculate efficiency loss from unburned at various CO levels
        co_levels = [50, 100, 200, 500, 1000]

        from combustion.thermodynamics import compute_unburned_loss

        for co in co_levels:
            unburned_loss = compute_unburned_loss(co, 0.0, "natural_gas")

            # Loss should increase with CO
            assert unburned_loss >= 0
            if co >= 100:
                assert unburned_loss > 0.05, "High CO should cause measurable loss"


# ============================================================================
# EFFICIENCY BOUNDARY TESTS
# ============================================================================

class TestEfficiencyBoundaries:
    """Test suite for efficiency calculation boundaries."""

    @pytest.mark.parametrize("stack_temp,ambient_temp,expected_eff_range", [
        (120, 25, (92, 98)),     # Low stack temp - high efficiency
        (150, 25, (88, 95)),     # Normal operation
        (180, 25, (85, 92)),     # Typical operation
        (220, 30, (80, 88)),     # Higher stack temp
        (300, 35, (70, 82)),     # High stack temp - lower efficiency
        (400, 40, (60, 75)),     # Very high stack temp
    ])
    def test_efficiency_vs_stack_temperature(
        self,
        stack_temp: int,
        ambient_temp: int,
        expected_eff_range: Tuple[int, int]
    ):
        """Test efficiency calculation at various stack temperatures."""
        result = compute_efficiency_indirect(
            stack_temp_c=float(stack_temp),
            ambient_temp_c=float(ambient_temp),
            excess_o2_pct=3.0,
            co_ppm=50.0,
            furnace_rating_mw=10.0,
            load_fraction=0.8,
            fuel_type="natural_gas"
        )

        min_eff, max_eff = expected_eff_range
        assert min_eff <= result.gross_efficiency_pct <= max_eff, \
            f"Stack temp {stack_temp}C: efficiency {result.gross_efficiency_pct:.1f}% " \
            f"outside expected range [{min_eff}, {max_eff}]"

    def test_efficiency_at_zero_load(self):
        """Test efficiency behavior at zero load."""
        result = compute_heat_balance(
            fuel_flow_kg_s=0.0,
            useful_output_mw=0.0,
            stack_temp_c=180.0,
            ambient_temp_c=25.0,
            excess_o2_pct=3.0,
            furnace_rating_mw=10.0
        )

        # Zero input should give zero efficiency
        assert result.heat_input_mw == 0
        assert result.efficiency_pct == 0

    def test_efficiency_exceeds_100_check(self):
        """Test that efficiency cannot exceed 100%."""
        result = compute_efficiency_direct(
            useful_output_mw=15.0,  # More output than input (impossible)
            fuel_flow_kg_s=0.1,
            fuel_type="natural_gas",
            use_lhv=False
        )

        # Should be capped at 100%
        assert result <= 100


# ============================================================================
# STOICHIOMETRY BOUNDARY TESTS
# ============================================================================

class TestStoichiometryBoundaries:
    """Test suite for stoichiometry calculation boundaries."""

    def test_composition_sum_validation(self):
        """Test fuel composition sum validation."""
        # Valid composition (sums to 100%)
        valid_comp = {"CH4": 94.0, "C2H6": 3.0, "N2": 3.0}
        is_valid, errors = validate_fuel_composition(valid_comp)
        assert is_valid

        # Invalid composition (doesn't sum to 100%)
        invalid_comp = {"CH4": 80.0, "C2H6": 3.0, "N2": 3.0}
        is_valid, errors = validate_fuel_composition(invalid_comp)
        assert not is_valid
        assert any("100" in e for e in errors)

    def test_negative_composition_values(self):
        """Test rejection of negative composition values."""
        invalid_comp = {"CH4": 105.0, "N2": -5.0}
        is_valid, errors = validate_fuel_composition(invalid_comp)
        assert not is_valid
        assert any("negative" in e.lower() for e in errors)

    def test_air_flow_for_target_o2_boundaries(self):
        """Test air flow calculation at O2 boundaries."""
        fuel_flow = 100.0  # Nm3/h

        # Valid O2 targets
        for target_o2 in [1.0, 3.0, 5.0, 10.0]:
            air_flow = compute_air_flow_for_target_o2(
                fuel_flow, target_o2, FuelType.NATURAL_GAS
            )
            assert air_flow > 0
            assert air_flow > fuel_flow * 9  # Air flow > 9x fuel flow

        # Invalid O2 target (too high)
        with pytest.raises(ValueError):
            compute_air_flow_for_target_o2(fuel_flow, 16.0, FuelType.NATURAL_GAS)

    def test_fuel_flow_for_target_duty_boundaries(self):
        """Test fuel flow calculation for target duty."""
        # Valid duty targets
        for duty in [0.1, 1.0, 10.0, 100.0]:
            fuel_flow = compute_fuel_flow_for_target_duty(
                duty, FuelType.NATURAL_GAS, 0.90
            )
            assert fuel_flow > 0

        # Invalid duty (zero or negative)
        with pytest.raises(ValueError):
            compute_fuel_flow_for_target_duty(0.0, FuelType.NATURAL_GAS, 0.90)

        with pytest.raises(ValueError):
            compute_fuel_flow_for_target_duty(-10.0, FuelType.NATURAL_GAS, 0.90)

        # Invalid efficiency
        with pytest.raises(ValueError):
            compute_fuel_flow_for_target_duty(10.0, FuelType.NATURAL_GAS, 0.0)

        with pytest.raises(ValueError):
            compute_fuel_flow_for_target_duty(10.0, FuelType.NATURAL_GAS, 1.5)


# ============================================================================
# PRESSURE AND DRAFT BOUNDARY TESTS
# ============================================================================

class TestPressureDraftBoundaries:
    """Test suite for pressure and draft boundaries."""

    @pytest.mark.parametrize("draft_value,expected_valid", [
        (-0.6, False),    # Too negative (high vacuum)
        (-0.5, True),     # At minimum
        (-0.3, True),     # Normal
        (-0.1, True),     # Normal
        (-0.01, True),    # At maximum (slight negative)
        (0.0, False),     # Zero draft (blocked)
        (0.1, False),     # Positive pressure (dangerous)
    ])
    def test_draft_boundary_validation(
        self,
        safety_envelope,
        draft_value: float,
        expected_valid: bool
    ):
        """Test furnace draft validation at boundaries."""
        setpoint = Setpoint(
            parameter_name="draft",
            value=draft_value,
            unit="inwc"
        )

        validation = safety_envelope.validate_within_envelope(setpoint)

        assert validation.is_valid == expected_valid


# ============================================================================
# TEMPERATURE BOUNDARY TESTS
# ============================================================================

class TestTemperatureBoundaries:
    """Test suite for temperature boundaries."""

    @pytest.mark.parametrize("steam_temp,expected_valid", [
        (400.0, True),    # Normal
        (500.0, True),    # High normal
        (550.0, True),    # At limit
        (551.0, False),   # Over limit
        (600.0, False),   # Way over
    ])
    def test_steam_temperature_limits(
        self,
        safety_envelope,
        steam_temp: float,
        expected_valid: bool
    ):
        """Test steam temperature limit validation."""
        setpoint = Setpoint(
            parameter_name="steam_temp",
            value=steam_temp,
            unit="F"
        )

        validation = safety_envelope.validate_within_envelope(setpoint)

        assert validation.is_valid == expected_valid


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
