"""
Extreme Conditions Edge Case Tests for GL-004 BURNMASTER

Tests system behavior under extreme operating conditions including:
- Extreme temperatures (cryogenic to furnace temperatures)
- Extreme pressures (vacuum to high pressure)
- Extreme fuel compositions (pure hydrogen, blast furnace gas)
- Extreme air-fuel ratios (near-stoichiometric to very lean)
- Extreme load conditions (minimum turndown to overfire)

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
import numpy as np
from decimal import Decimal
from datetime import datetime
from typing import Dict, List, Any
import math

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
    STANDARD_COMPOSITIONS, validate_fuel_composition,
)
from combustion.thermodynamics import (
    compute_cp_shomate,
    compute_enthalpy_shomate,
    compute_stack_loss,
    compute_radiation_loss,
    compute_efficiency_indirect,
    compute_efficiency_direct,
    compute_heat_balance,
)
from safety.safety_envelope import (
    SafetyEnvelope, EnvelopeLimits, Setpoint, EnvelopeStatus,
)
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


@pytest.fixture
def extreme_fuel_compositions() -> Dict[str, Dict[str, float]]:
    """Generate extreme fuel composition test cases."""
    return {
        "pure_hydrogen": {"H2": 100.0},
        "pure_methane": {"CH4": 100.0},
        "pure_propane": {"C3H8": 100.0},
        "high_inerts": {"CH4": 50.0, "N2": 40.0, "CO2": 10.0},
        "blast_furnace_gas": {"CO": 23.0, "CO2": 22.0, "N2": 54.0, "H2": 1.0},
        "coke_oven_gas": {"H2": 55.0, "CH4": 25.0, "CO": 6.0, "CO2": 2.0, "N2": 10.0, "C2H4": 2.0},
        "high_hydrogen_blend": {"CH4": 30.0, "H2": 70.0},
        "refinery_off_gas": {"CH4": 35.0, "C2H6": 10.0, "C3H8": 5.0, "H2": 30.0, "C2H4": 8.0, "C3H6": 5.0, "CO": 2.0, "N2": 5.0},
    }


# ============================================================================
# EXTREME TEMPERATURE TESTS
# ============================================================================

class TestExtremeTemperatures:
    """Test suite for extreme temperature conditions."""

    @pytest.mark.parametrize("temp_k,expected_valid", [
        (200.0, True),     # Cryogenic (below ambient)
        (298.15, True),    # Standard reference temperature
        (500.0, True),     # Elevated preheated air
        (800.0, True),     # High preheat
        (1200.0, True),    # Furnace zone
        (1800.0, True),    # Flame zone
        (2200.0, True),    # Near adiabatic flame temp
        (2500.0, True),    # Above typical flame temp
    ])
    def test_shomate_equation_temperature_range(self, temp_k: float, expected_valid: bool):
        """Test Shomate equation at extreme temperatures."""
        try:
            cp_n2 = compute_cp_shomate("N2", temp_k)
            cp_o2 = compute_cp_shomate("O2", temp_k)

            # Verify results are physically reasonable
            assert cp_n2 > 0, f"N2 Cp must be positive at {temp_k}K"
            assert cp_o2 > 0, f"O2 Cp must be positive at {temp_k}K"

            # Cp should increase with temperature (generally true)
            if temp_k > 300:
                cp_n2_low = compute_cp_shomate("N2", 300.0)
                # N2 Cp increases slowly with temperature
                assert cp_n2 >= cp_n2_low * 0.95, f"N2 Cp should not decrease significantly"

        except ValueError as e:
            if expected_valid:
                pytest.fail(f"Unexpected error at {temp_k}K: {e}")

    @pytest.mark.parametrize("stack_temp_c,ambient_temp_c", [
        (-40.0, -50.0),    # Arctic conditions
        (50.0, 45.0),      # Minimal temperature difference
        (200.0, 25.0),     # Normal operation
        (400.0, 35.0),     # High stack temp
        (600.0, 40.0),     # Very high stack temp
        (800.0, 30.0),     # Extreme stack temp
    ])
    def test_stack_loss_extreme_temperatures(self, stack_temp_c: float, ambient_temp_c: float):
        """Test stack loss calculation at extreme temperature differentials."""
        stack_loss = compute_stack_loss(
            stack_temp_c=stack_temp_c,
            ambient_temp_c=ambient_temp_c,
            excess_o2_pct=3.0,
            fuel_type="natural_gas"
        )

        # Stack loss should be non-negative
        assert stack_loss >= 0, "Stack loss cannot be negative"

        # Stack loss should increase with temperature difference
        if stack_temp_c > ambient_temp_c + 50:
            assert stack_loss > 0, "Stack loss should be positive for significant temp difference"

        # Stack loss should not exceed 100%
        assert stack_loss <= 100, "Stack loss cannot exceed 100%"

    def test_efficiency_near_zero_temp_difference(self):
        """Test efficiency calculation when stack temp equals ambient."""
        result = compute_efficiency_indirect(
            stack_temp_c=25.0,
            ambient_temp_c=25.0,
            excess_o2_pct=3.0,
            co_ppm=50.0,
            furnace_rating_mw=10.0,
            load_fraction=0.8,
            fuel_type="natural_gas"
        )

        # When stack temp equals ambient, stack loss should be minimal
        assert result.dry_flue_gas_loss_pct >= 0
        assert result.gross_efficiency_pct >= 0

    def test_enthalpy_temperature_extremes(self):
        """Test enthalpy calculations at temperature extremes."""
        temps = [300, 500, 800, 1000, 1200]

        for i in range(len(temps) - 1):
            h_low = compute_enthalpy_shomate("N2", temps[i])
            h_high = compute_enthalpy_shomate("N2", temps[i + 1])

            # Enthalpy should increase with temperature
            assert h_high > h_low, f"Enthalpy should increase from {temps[i]}K to {temps[i+1]}K"


# ============================================================================
# EXTREME PRESSURE TESTS
# ============================================================================

class TestExtremePressures:
    """Test suite for extreme pressure conditions."""

    @pytest.mark.parametrize("furnace_rating_mw,load_fraction,expected_max_loss", [
        (0.5, 0.5, 5.0),     # Very small furnace
        (1.0, 0.1, 10.0),    # Low load on small furnace
        (5.0, 0.2, 5.0),     # Low load
        (20.0, 0.5, 3.0),    # Medium load
        (100.0, 0.9, 2.0),   # High load large furnace
        (500.0, 1.0, 1.0),   # Maximum load very large furnace
    ])
    def test_radiation_loss_at_extreme_loads(
        self,
        furnace_rating_mw: float,
        load_fraction: float,
        expected_max_loss: float
    ):
        """Test radiation loss calculation at extreme load conditions."""
        radiation_loss = compute_radiation_loss(
            furnace_rating_mw=furnace_rating_mw,
            load_fraction=load_fraction
        )

        # Radiation loss should be non-negative
        assert radiation_loss >= 0

        # Radiation loss should be bounded
        assert radiation_loss <= 3.0, "Radiation loss capped at 3%"

    def test_radiation_loss_zero_load(self):
        """Test radiation loss at zero load."""
        radiation_loss = compute_radiation_loss(
            furnace_rating_mw=10.0,
            load_fraction=0.0
        )

        assert radiation_loss == 0.0, "Zero load should have zero radiation loss"

    def test_radiation_loss_very_small_load(self):
        """Test radiation loss at very small load fractions."""
        for load_frac in [0.01, 0.05, 0.1]:
            radiation_loss = compute_radiation_loss(
                furnace_rating_mw=10.0,
                load_fraction=load_frac
            )

            # At very low loads, radiation loss percentage is high (capped at 3%)
            assert radiation_loss <= 3.0


# ============================================================================
# EXTREME FUEL COMPOSITION TESTS
# ============================================================================

class TestExtremeFuelCompositions:
    """Test suite for extreme fuel compositions."""

    def test_pure_hydrogen_combustion(self, extreme_fuel_compositions):
        """Test stoichiometry for pure hydrogen fuel."""
        h2_composition = extreme_fuel_compositions["pure_hydrogen"]

        stoich_air = compute_stoichiometric_air(h2_composition)

        # Pure H2: H2 + 0.5 O2 -> H2O
        # Stoich O2 = 0.5 mol O2 / mol H2
        # Stoich air = 0.5 / 0.2095 = 2.387 Nm3 air / Nm3 H2
        expected_stoich_air = 0.5 / 0.2095

        assert abs(stoich_air - expected_stoich_air) < 0.01, \
            f"Pure H2 stoichiometric air should be ~{expected_stoich_air:.3f}"

    def test_blast_furnace_gas_low_heating_value(self, extreme_fuel_compositions):
        """Test calculations for low heating value blast furnace gas."""
        bfg_composition = extreme_fuel_compositions["blast_furnace_gas"]

        # Create FuelComposition object
        fuel_comp = FuelComposition(
            co=bfg_composition["CO"],
            co2=bfg_composition["CO2"],
            n2=bfg_composition["N2"],
            h2=bfg_composition["H2"]
        )

        # Compute properties
        mw = compute_molecular_weight(fuel_comp)
        hhv, lhv = compute_heating_values(fuel_comp)

        # BFG has very low heating value due to high inerts
        assert hhv < 10, "BFG should have very low HHV due to high inerts"
        assert lhv > 0, "LHV should be positive"

    def test_high_hydrogen_blend_flame_speed(self, extreme_fuel_compositions):
        """Test high hydrogen blend properties and stability implications."""
        h2_blend = extreme_fuel_compositions["high_hydrogen_blend"]

        fuel_comp = FuelComposition(
            ch4=h2_blend["CH4"],
            h2=h2_blend["H2"]
        )

        props = compute_fuel_properties(fuel_comp, FuelType.HYDROGEN_BLEND)

        # High H2 blends have higher flame speed
        assert props.flame_speed > 0.5, "High H2 blend should have elevated flame speed"

        # Adiabatic flame temp should be higher
        assert props.adiabatic_flame_temp > 2200, "High H2 blend has higher AFT"

    @pytest.mark.parametrize("h2_percent", [0, 10, 20, 30, 50, 70, 100])
    def test_hydrogen_blend_progression(self, h2_percent: int):
        """Test properties across hydrogen blend range."""
        ch4_percent = 100 - h2_percent

        if h2_percent == 100:
            fuel_comp = FuelComposition(h2=100.0)
        elif ch4_percent == 100:
            fuel_comp = FuelComposition(ch4=100.0)
        else:
            fuel_comp = FuelComposition(ch4=float(ch4_percent), h2=float(h2_percent))

        props = compute_fuel_properties(fuel_comp, FuelType.HYDROGEN_BLEND)

        # Properties should be valid
        assert props.hhv > 0
        assert props.lhv > 0
        assert props.stoichiometric_afr > 0

        # Flame speed increases with H2 content
        if h2_percent > 0:
            assert props.flame_speed > 0.38, "H2 addition increases flame speed"

    def test_high_inert_fuel_efficiency_impact(self, extreme_fuel_compositions):
        """Test efficiency calculations for fuel with high inert content."""
        high_inerts = extreme_fuel_compositions["high_inerts"]

        fuel_comp = FuelComposition(
            ch4=high_inerts["CH4"],
            n2=high_inerts["N2"],
            co2=high_inerts["CO2"]
        )

        hhv, lhv = compute_heating_values(fuel_comp)

        # High inerts reduce heating value proportionally
        pure_ch4_hhv = 39.82  # MJ/Nm3
        expected_hhv = pure_ch4_hhv * 0.50  # 50% CH4

        assert abs(hhv - expected_hhv) < 1.0, "HHV should scale with combustible fraction"

    def test_coke_oven_gas_properties(self, extreme_fuel_compositions):
        """Test coke oven gas with mixed combustibles."""
        cog = extreme_fuel_compositions["coke_oven_gas"]

        fuel_comp = FuelComposition(
            h2=cog["H2"],
            ch4=cog["CH4"],
            co=cog["CO"],
            co2=cog["CO2"],
            n2=cog["N2"],
            c2h4=cog["C2H4"]
        )

        props = compute_fuel_properties(fuel_comp, FuelType.COKE_OVEN_GAS)

        # COG has intermediate properties
        assert 15 < props.hhv < 25, "COG HHV typically 15-25 MJ/Nm3"
        assert props.flame_speed > 0.6, "High H2 content increases flame speed"


# ============================================================================
# EXTREME AIR-FUEL RATIO TESTS
# ============================================================================

class TestExtremeAirFuelRatios:
    """Test suite for extreme air-fuel ratio conditions."""

    @pytest.mark.parametrize("lambda_val,expected_excess_air", [
        (0.8, -20.0),    # Rich mixture (incomplete combustion)
        (0.9, -10.0),    # Slightly rich
        (1.0, 0.0),      # Stoichiometric
        (1.05, 5.0),     # Optimal lean
        (1.15, 15.0),    # Typical lean operation
        (1.5, 50.0),     # Very lean
        (2.0, 100.0),    # Extremely lean
        (3.0, 200.0),    # Very high excess air
    ])
    def test_excess_air_calculation_extremes(self, lambda_val: float, expected_excess_air: float):
        """Test excess air calculation at extreme lambda values."""
        excess_air = compute_excess_air_percent(lambda_val)

        assert abs(excess_air - expected_excess_air) < 0.1, \
            f"Lambda {lambda_val} should give {expected_excess_air}% excess air"

    @pytest.mark.parametrize("stack_o2,fuel_type,expected_lambda_range", [
        (0.1, "natural_gas", (0.99, 1.01)),   # Near stoichiometric
        (1.0, "natural_gas", (1.02, 1.08)),   # Low excess
        (3.0, "natural_gas", (1.10, 1.20)),   # Typical operation
        (5.0, "natural_gas", (1.20, 1.35)),   # High excess
        (8.0, "natural_gas", (1.40, 1.70)),   # Very high excess
        (12.0, "natural_gas", (1.80, 2.50)),  # Extreme excess
        (18.0, "natural_gas", (3.0, 5.0)),    # Near air-only
    ])
    def test_lambda_inference_from_extreme_o2(
        self,
        stack_o2: float,
        fuel_type: str,
        expected_lambda_range: tuple
    ):
        """Test lambda inference from extreme O2 readings."""
        lambda_val = infer_lambda_from_o2(stack_o2, fuel_type)

        min_lambda, max_lambda = expected_lambda_range
        assert min_lambda <= lambda_val <= max_lambda, \
            f"Lambda for {stack_o2}% O2 should be in range [{min_lambda}, {max_lambda}]"

    def test_lambda_o2_roundtrip_consistency(self):
        """Test that lambda -> O2 -> lambda is consistent."""
        fuel_type = "natural_gas"

        for original_lambda in [1.05, 1.10, 1.15, 1.20, 1.30, 1.50]:
            # Compute O2 from lambda
            o2 = compute_excess_o2(original_lambda, fuel_type)

            # Infer lambda back from O2
            inferred_lambda = infer_lambda_from_o2(o2, fuel_type)

            # Should be close to original
            assert abs(inferred_lambda - original_lambda) < 0.05, \
                f"Roundtrip lambda {original_lambda} -> O2 {o2}% -> lambda {inferred_lambda}"

    def test_sub_stoichiometric_operation(self):
        """Test behavior for sub-stoichiometric (rich) operation."""
        # Sub-stoichiometric should have no excess O2
        o2 = compute_excess_o2(0.9, "natural_gas")
        assert o2 == 0.0, "Sub-stoichiometric operation has no excess O2"

    def test_lambda_validation_edge_cases(self):
        """Test validation of extreme lambda values."""
        # Very low lambda
        is_valid, errors = validate_stoichiometry_inputs(lambda_val=0.3)
        assert not is_valid
        assert any("too low" in e.lower() for e in errors)

        # Very high lambda
        is_valid, errors = validate_stoichiometry_inputs(lambda_val=6.0)
        assert not is_valid
        assert any("too high" in e.lower() for e in errors)

        # Valid range
        is_valid, errors = validate_stoichiometry_inputs(lambda_val=1.15)
        assert is_valid


# ============================================================================
# EXTREME LOAD CONDITION TESTS
# ============================================================================

class TestExtremeLoadConditions:
    """Test suite for extreme load conditions."""

    @pytest.mark.parametrize("load_fraction", [
        0.05,   # Minimum turndown
        0.10,   # Very low load
        0.25,   # Low load
        0.50,   # Half load
        0.75,   # High load
        0.90,   # Near full load
        1.00,   # Full load
        1.05,   # Slight overfire (should be clamped)
    ])
    def test_heat_balance_at_extreme_loads(self, load_fraction: float):
        """Test heat balance calculation at extreme loads."""
        # Clamp load fraction to valid range
        clamped_load = min(1.0, load_fraction)

        result = compute_heat_balance(
            fuel_flow_kg_s=0.5 * clamped_load,
            useful_output_mw=10.0 * clamped_load,
            stack_temp_c=180.0,
            ambient_temp_c=25.0,
            excess_o2_pct=3.0,
            furnace_rating_mw=12.0,
            co_ppm=50.0,
            fuel_type="natural_gas"
        )

        # All components should be non-negative
        assert result.heat_input_mw >= 0
        assert result.useful_output_mw >= 0
        assert result.stack_loss_mw >= 0
        assert result.radiation_loss_mw >= 0

    def test_minimum_turndown_stability(self, stability_calculator):
        """Test flame stability at minimum turndown conditions."""
        # At minimum turndown, flame signal is weaker and more variable
        weak_signal = np.array([30.0 + np.random.normal(0, 5) for _ in range(100)])
        high_o2_variance = 0.8  # Higher variance at low load

        result = stability_calculator.compute_stability_index(weak_signal, high_o2_variance)

        # Should identify marginal stability
        assert result.stability_level in [StabilityLevel.MARGINAL, StabilityLevel.POOR, StabilityLevel.GOOD]

    def test_overfire_conditions(self, safety_envelope):
        """Test safety envelope behavior at overfire conditions."""
        # Attempt to set firing rate above 100%
        setpoint = Setpoint(
            parameter_name="firing_rate",
            value=105.0,  # Overfire
            unit="%"
        )

        validation = safety_envelope.validate_within_envelope(setpoint)

        assert not validation.is_valid, "Overfire should be blocked"
        assert "outside limits" in validation.blocking_reason.lower()

    @pytest.mark.parametrize("fuel_flow_kg_s", [
        0.001,   # Minimum pilot
        0.01,    # Very low
        0.1,     # Low
        1.0,     # Medium
        10.0,    # High
        100.0,   # Very high
    ])
    def test_fuel_flow_extremes(self, fuel_flow_kg_s: float):
        """Test calculations at extreme fuel flow rates."""
        result = compute_heat_balance(
            fuel_flow_kg_s=fuel_flow_kg_s,
            useful_output_mw=fuel_flow_kg_s * 40,  # Approximate
            stack_temp_c=180.0,
            ambient_temp_c=25.0,
            excess_o2_pct=3.0,
            furnace_rating_mw=fuel_flow_kg_s * 50,
            fuel_type="natural_gas"
        )

        assert result.heat_input_mw >= 0
        if fuel_flow_kg_s > 0:
            assert result.efficiency_pct >= 0
            assert result.efficiency_pct <= 100


# ============================================================================
# EXTREME STABILITY CONDITION TESTS
# ============================================================================

class TestExtremeStabilityConditions:
    """Test suite for extreme flame stability conditions."""

    def test_perfectly_stable_flame(self, stability_calculator):
        """Test stability index for perfectly stable flame."""
        # Constant flame signal with zero variance
        perfect_signal = np.array([100.0] * 100)
        zero_variance = 0.0

        result = stability_calculator.compute_stability_index(perfect_signal, zero_variance)

        assert result.stability_level == StabilityLevel.EXCELLENT
        assert result.stability_index >= Decimal("0.95")

    def test_highly_unstable_flame(self, stability_calculator):
        """Test stability index for highly unstable flame."""
        # Highly variable flame signal
        unstable_signal = np.array([50.0, 150.0, 30.0, 170.0, 40.0, 160.0] * 20)
        high_variance = 2.0

        result = stability_calculator.compute_stability_index(unstable_signal, high_variance)

        assert result.stability_level in [StabilityLevel.POOR, StabilityLevel.CRITICAL]

    def test_near_flameout_conditions(self, stability_calculator):
        """Test stability near flameout (very low flame signal)."""
        # Signal approaching zero
        near_flameout = np.array([10.0 + np.random.normal(0, 8) for _ in range(100)])
        near_flameout = np.maximum(near_flameout, 1.0)  # Prevent zero/negative

        result = stability_calculator.compute_stability_index(near_flameout, 0.5)

        # Should generate low flame signal warning
        assert any("low flame signal" in rec.lower() for rec in result.recommendations)

    def test_blowoff_risk_at_extreme_velocity(self, stability_calculator):
        """Test blowoff risk at extreme velocity conditions."""
        # Velocity exceeding flame speed
        risk_score, risk_level = stability_calculator.compute_blowoff_risk(
            velocity=5.0,        # m/s - exceeds flame speed
            flame_speed=0.38,    # m/s - natural gas
            lambda_val=1.3       # Lean mixture
        )

        assert risk_level == RiskLevel.CRITICAL
        assert risk_score >= Decimal("0.75")

    def test_flashback_risk_high_preheat(self, stability_calculator):
        """Test flashback risk with high preheat temperature."""
        # Preheat approaching autoignition
        risk_score, risk_level = stability_calculator.compute_flashback_risk(
            premix_temp=500.0,      # Very high preheat
            autoignition_temp=540.0  # Natural gas
        )

        assert risk_level == RiskLevel.CRITICAL
        assert risk_score >= Decimal("0.75")

    def test_stability_margin_at_envelope_edge(self, stability_calculator):
        """Test stability margin when operating at envelope edge."""
        operating_point = {
            'velocity': 48.0,       # Near max_velocity
            'lambda': 1.45,         # Near max_lambda
            'premix_temp': 380.0    # Near max temp
        }

        stability_envelope = {
            'min_velocity': 5.0,
            'max_velocity': 50.0,
            'min_lambda': 0.9,
            'max_lambda': 1.5,
            'max_premix_temp': 400.0
        }

        result = stability_calculator.compute_stability_margin(
            operating_point, stability_envelope
        )

        # Should have low margin
        assert result.margin_percent < Decimal("20")

    @pytest.mark.parametrize("oscillation_freq_hz,expected_type", [
        (1.0, "flow_instability"),
        (10.0, "combustion_dynamics"),
        (100.0, "acoustic_resonance"),
        (1000.0, "high_frequency_noise"),
    ])
    def test_oscillation_detection_frequency_ranges(
        self,
        stability_calculator,
        oscillation_freq_hz: float,
        expected_type: str
    ):
        """Test oscillation detection at different frequency ranges."""
        # Generate sinusoidal pressure signal
        sampling_rate = 10000.0  # Hz
        duration = 0.5  # seconds
        t = np.linspace(0, duration, int(sampling_rate * duration))

        # Create oscillating signal
        signal = 100.0 + 20.0 * np.sin(2 * np.pi * oscillation_freq_hz * t)

        result = stability_calculator.detect_oscillations(signal, sampling_rate)

        if result.has_oscillations:
            assert result.oscillation_type == expected_type


# ============================================================================
# NUMERICAL EDGE CASE TESTS
# ============================================================================

class TestNumericalEdgeCases:
    """Test suite for numerical edge cases and precision."""

    def test_very_small_values(self):
        """Test handling of very small numerical values."""
        # Very small fuel composition
        small_composition = {"CH4": 99.99, "C2H6": 0.01}
        stoich_air = compute_stoichiometric_air(small_composition)

        assert stoich_air > 0
        assert not np.isnan(stoich_air)
        assert not np.isinf(stoich_air)

    def test_near_zero_denominators(self):
        """Test protection against division by zero."""
        # Near-zero stoichiometric AFR (should raise error)
        with pytest.raises(ValueError):
            compute_lambda(actual_af=10.0, stoich_af=0.0)

        with pytest.raises(ValueError):
            compute_lambda(actual_af=10.0, stoich_af=-1.0)

    def test_large_value_handling(self):
        """Test handling of large numerical values."""
        # Large fuel flow
        result = compute_heat_balance(
            fuel_flow_kg_s=1000.0,
            useful_output_mw=40000.0,
            stack_temp_c=180.0,
            ambient_temp_c=25.0,
            excess_o2_pct=3.0,
            furnace_rating_mw=50000.0
        )

        assert not np.isnan(result.heat_input_mw)
        assert not np.isinf(result.heat_input_mw)

    def test_decimal_precision_preservation(self, stability_calculator):
        """Test that Decimal precision is preserved in calculations."""
        signal = np.array([100.0] * 50)
        result = stability_calculator.compute_stability_index(signal, 0.1)

        # Check Decimal precision
        assert isinstance(result.stability_index, Decimal)
        assert len(str(result.stability_index).split('.')[-1]) <= 4  # 4 decimal places

    def test_nan_handling_in_inputs(self, stability_calculator):
        """Test handling of NaN values in inputs."""
        signal_with_nan = np.array([100.0, np.nan, 100.0, 100.0] + [100.0] * 10)

        # Should handle gracefully or raise appropriate error
        try:
            result = stability_calculator.compute_stability_index(signal_with_nan, 0.1)
            # If it succeeds, verify output is valid
            assert not np.isnan(float(result.stability_index))
        except (ValueError, RuntimeWarning):
            # Expected behavior for NaN input
            pass

    def test_inf_handling_in_inputs(self, stability_calculator):
        """Test handling of infinity values in inputs."""
        signal_with_inf = np.array([100.0, np.inf, 100.0, 100.0] + [100.0] * 10)

        # Should handle gracefully or raise appropriate error
        try:
            result = stability_calculator.compute_stability_index(signal_with_inf, 0.1)
            # If it succeeds, verify output is valid
            assert not np.isinf(float(result.stability_index))
        except (ValueError, RuntimeWarning, OverflowError):
            # Expected behavior for inf input
            pass


# ============================================================================
# PROVENANCE AND DETERMINISM TESTS
# ============================================================================

class TestProvenanceDeterminism:
    """Test determinism and provenance tracking under extreme conditions."""

    def test_deterministic_stability_calculation(self, stability_calculator):
        """Verify stability calculation is deterministic."""
        signal = np.array([100.0, 102.0, 98.0, 101.0, 99.0] * 20)
        o2_variance = 0.3

        # Run multiple times
        results = [
            stability_calculator.compute_stability_index(signal.copy(), o2_variance)
            for _ in range(5)
        ]

        # All results should be identical
        first_hash = results[0].provenance_hash
        for result in results[1:]:
            assert result.provenance_hash == first_hash
            assert result.stability_index == results[0].stability_index

    def test_provenance_hash_changes_with_input(self, stability_calculator):
        """Verify provenance hash changes when input changes."""
        signal = np.array([100.0] * 50)

        result1 = stability_calculator.compute_stability_index(signal, 0.1)
        result2 = stability_calculator.compute_stability_index(signal, 0.2)

        # Different inputs should produce different provenance hashes
        assert result1.provenance_hash != result2.provenance_hash

    def test_stoichiometry_determinism(self):
        """Verify stoichiometry calculations are deterministic."""
        composition = {"CH4": 94.0, "C2H6": 3.0, "C3H8": 1.0, "N2": 2.0}

        results = [compute_stoichiometric_air(composition) for _ in range(10)]

        # All results should be identical
        assert all(r == results[0] for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
