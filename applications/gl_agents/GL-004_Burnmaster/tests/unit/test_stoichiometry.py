"""
Unit Tests for GL-004 BURNMASTER Stoichiometry Module

Comprehensive test coverage for combustion stoichiometry calculations.
Tests validate:
- Stoichiometric air-fuel ratio calculations
- Lambda (equivalence ratio) calculations
- Excess air percentage calculations
- O2 inference from stack measurements
- Multi-fuel support
- Provenance hash determinism
- Edge cases and physics validation

Reference Standards:
- ASME PTC 4: Fired Steam Generators
- EPA Method 19: Determination of Sulfur Dioxide Removal Efficiency
"""

import pytest
import hashlib
import json
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from combustion.stoichiometry import (
    compute_stoichiometric_air,
    compute_lambda,
    compute_excess_air_percent,
    compute_excess_o2,
    infer_lambda_from_o2,
    compute_stoichiometry_from_fuel_type,
    compute_air_flow_for_target_o2,
    compute_fuel_flow_for_target_duty,
    compute_flue_gas_flow,
    validate_stoichiometry_inputs,
    check_stoichiometry_consistency,
    StoichiometryResult,
    FUEL_O2_CONSTANTS,
    MAX_DRY_O2_PERCENT,
)
from combustion.fuel_properties import FuelType, STOICH_O2


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def natural_gas_composition():
    """Standard natural gas composition for testing."""
    return {
        "CH4": 94.0,
        "C2H6": 3.0,
        "C3H8": 1.0,
        "N2": 2.0,
    }


@pytest.fixture
def refinery_gas_composition():
    """High-H2 refinery gas composition."""
    return {
        "H2": 45.0,
        "CH4": 30.0,
        "C2H6": 10.0,
        "CO": 5.0,
        "N2": 10.0,
    }


@pytest.fixture
def pure_hydrogen_composition():
    """Pure hydrogen fuel."""
    return {"H2": 100.0}


# ============================================================================
# Test Classes
# ============================================================================

class TestStoichiometricAFR:
    """Tests for stoichiometric air-fuel ratio calculations."""

    def test_natural_gas_stoich_air(self, natural_gas_composition):
        """Test stoichiometric air for natural gas."""
        stoich_air = compute_stoichiometric_air(natural_gas_composition)

        # Natural gas typically requires ~9.5-10.0 Nm3 air per Nm3 fuel
        assert 9.0 < stoich_air < 11.0, f"Unexpected stoich air: {stoich_air}"

    def test_pure_hydrogen_stoich_air(self, pure_hydrogen_composition):
        """Test stoichiometric air for pure hydrogen."""
        stoich_air = compute_stoichiometric_air(pure_hydrogen_composition)

        # H2 + 0.5 O2 -> H2O
        # O2 required = 0.5 mol per mol H2
        # Air = 0.5 / 0.2095 = 2.387 Nm3/Nm3
        assert 2.0 < stoich_air < 3.0, f"Unexpected H2 stoich air: {stoich_air}"

    def test_refinery_gas_stoich_air(self, refinery_gas_composition):
        """Test stoichiometric air for refinery gas."""
        stoich_air = compute_stoichiometric_air(refinery_gas_composition)

        # Refinery gas with high H2 has lower AFR than natural gas
        assert 4.0 < stoich_air < 8.0, f"Unexpected refinery gas stoich air: {stoich_air}"

    def test_empty_composition_raises(self):
        """Test that empty composition raises ValueError."""
        with pytest.raises(ValueError, match="Empty fuel composition"):
            compute_stoichiometric_air({})

    def test_invalid_composition_sum(self):
        """Test that composition not summing to 100% raises error."""
        invalid_composition = {"CH4": 50.0, "C2H6": 10.0}  # Sum = 60%
        with pytest.raises(ValueError, match="must sum to 100%"):
            compute_stoichiometric_air(invalid_composition)

    def test_determinism(self, natural_gas_composition):
        """Test calculation determinism - same input yields same output."""
        result1 = compute_stoichiometric_air(natural_gas_composition)
        result2 = compute_stoichiometric_air(natural_gas_composition)
        assert result1 == result2, "Calculation not deterministic"


class TestLambdaCalculations:
    """Tests for lambda (equivalence ratio) calculations."""

    def test_stoichiometric_lambda(self):
        """Test lambda = 1.0 at stoichiometric conditions."""
        lambda_val = compute_lambda(actual_af=10.0, stoich_af=10.0)
        assert lambda_val == 1.0

    def test_lean_condition(self):
        """Test lambda > 1.0 for excess air (lean) conditions."""
        lambda_val = compute_lambda(actual_af=11.5, stoich_af=10.0)
        assert lambda_val == 1.15

    def test_rich_condition(self):
        """Test lambda < 1.0 for fuel-rich conditions."""
        lambda_val = compute_lambda(actual_af=9.0, stoich_af=10.0)
        assert lambda_val == 0.9

    def test_zero_stoich_raises(self):
        """Test that zero stoichiometric AFR raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            compute_lambda(actual_af=10.0, stoich_af=0.0)

    def test_negative_stoich_raises(self):
        """Test that negative stoichiometric AFR raises error."""
        with pytest.raises(ValueError, match="must be positive"):
            compute_lambda(actual_af=10.0, stoich_af=-5.0)

    def test_negative_actual_raises(self):
        """Test that negative actual AFR raises error."""
        with pytest.raises(ValueError, match="must be non-negative"):
            compute_lambda(actual_af=-5.0, stoich_af=10.0)

    def test_precision(self):
        """Test lambda calculation precision."""
        lambda_val = compute_lambda(actual_af=11.523, stoich_af=10.0)
        assert lambda_val == 1.1523, "Precision lost in calculation"


class TestExcessAir:
    """Tests for excess air percentage calculations."""

    def test_stoichiometric_zero_excess(self):
        """Test 0% excess air at stoichiometric."""
        excess = compute_excess_air_percent(lambda_val=1.0)
        assert excess == 0.0

    def test_15_percent_excess(self):
        """Test 15% excess air calculation."""
        excess = compute_excess_air_percent(lambda_val=1.15)
        assert excess == 15.0

    def test_50_percent_excess(self):
        """Test 50% excess air calculation."""
        excess = compute_excess_air_percent(lambda_val=1.5)
        assert excess == 50.0

    def test_rich_negative_excess(self):
        """Test negative excess air for rich conditions."""
        excess = compute_excess_air_percent(lambda_val=0.9)
        assert excess == -10.0

    def test_negative_lambda_raises(self):
        """Test that negative lambda raises error."""
        with pytest.raises(ValueError, match="must be non-negative"):
            compute_excess_air_percent(lambda_val=-0.5)


class TestO2FromLambda:
    """Tests for excess O2 calculation from lambda."""

    def test_stoichiometric_zero_o2(self):
        """Test 0% O2 at stoichiometric conditions."""
        o2 = compute_excess_o2(lambda_val=1.0, fuel_type="natural_gas")
        assert o2 == 0.0

    def test_typical_lean_operation(self):
        """Test O2 at typical lean operation (lambda=1.15)."""
        o2 = compute_excess_o2(lambda_val=1.15, fuel_type="natural_gas")
        # Typical O2 at 15% excess air is ~2.5-3.5%
        assert 2.0 < o2 < 4.0, f"Unexpected O2: {o2}%"

    def test_high_excess_air(self):
        """Test O2 at high excess air (lambda=1.5)."""
        o2 = compute_excess_o2(lambda_val=1.5, fuel_type="natural_gas")
        # Higher excess air means higher O2
        assert 5.0 < o2 < 10.0, f"Unexpected O2 at high excess: {o2}%"

    def test_fuel_type_affects_o2(self):
        """Test that fuel type affects O2 calculation."""
        o2_ng = compute_excess_o2(lambda_val=1.15, fuel_type="natural_gas")
        o2_h2 = compute_excess_o2(lambda_val=1.15, fuel_type="pure_hydrogen")
        # Different fuels have different O2-lambda relationships
        assert o2_ng != o2_h2, "Fuel type should affect O2 calculation"

    def test_unknown_fuel_defaults_natural_gas(self):
        """Test unknown fuel type defaults to natural gas behavior."""
        o2_unknown = compute_excess_o2(lambda_val=1.15, fuel_type="unknown_fuel")
        o2_ng = compute_excess_o2(lambda_val=1.15, fuel_type="natural_gas")
        assert o2_unknown == o2_ng

    def test_sub_stoichiometric_zero_o2(self):
        """Test sub-stoichiometric conditions return 0% O2."""
        o2 = compute_excess_o2(lambda_val=0.95, fuel_type="natural_gas")
        assert o2 == 0.0

    def test_very_low_lambda_raises(self):
        """Test very low lambda raises error."""
        with pytest.raises(ValueError, match="Lambda too low"):
            compute_excess_o2(lambda_val=0.3, fuel_type="natural_gas")


class TestInferLambdaFromO2:
    """Tests for inferring lambda from stack O2 measurements."""

    def test_zero_o2_stoichiometric(self):
        """Test near-zero O2 implies stoichiometric."""
        lambda_val = infer_lambda_from_o2(stack_o2=0.05, fuel_type="natural_gas")
        assert lambda_val == 1.0

    def test_typical_o2_level(self):
        """Test inference at typical O2 levels (3%)."""
        lambda_val = infer_lambda_from_o2(stack_o2=3.0, fuel_type="natural_gas")
        # 3% O2 typically corresponds to ~15-20% excess air
        assert 1.1 < lambda_val < 1.3, f"Unexpected lambda: {lambda_val}"

    def test_high_o2_high_lambda(self):
        """Test high O2 implies high lambda."""
        lambda_val = infer_lambda_from_o2(stack_o2=8.0, fuel_type="natural_gas")
        assert lambda_val > 1.5, f"High O2 should give high lambda: {lambda_val}"

    def test_round_trip_consistency(self):
        """Test O2 -> lambda -> O2 round trip consistency."""
        original_o2 = 4.0
        lambda_val = infer_lambda_from_o2(original_o2, "natural_gas")
        calculated_o2 = compute_excess_o2(lambda_val, "natural_gas")
        assert abs(original_o2 - calculated_o2) < 0.5, "Round trip inconsistent"

    def test_negative_o2_raises(self):
        """Test negative O2 raises error."""
        with pytest.raises(ValueError, match="cannot be negative"):
            infer_lambda_from_o2(stack_o2=-1.0, fuel_type="natural_gas")

    def test_o2_above_21_raises(self):
        """Test O2 > 21% raises error."""
        with pytest.raises(ValueError, match="cannot exceed 21%"):
            infer_lambda_from_o2(stack_o2=22.0, fuel_type="natural_gas")


class TestMultiFuelSupport:
    """Tests for multi-fuel support in stoichiometry calculations."""

    @pytest.mark.parametrize("fuel_type", list(FUEL_O2_CONSTANTS.keys()))
    def test_all_fuel_types_have_constants(self, fuel_type):
        """Test all fuel types have required O2 constants."""
        constants = FUEL_O2_CONSTANTS[fuel_type]
        assert "k1" in constants
        assert "k2" in constants
        assert "max_o2" in constants

    @pytest.mark.parametrize("fuel_type", list(FUEL_O2_CONSTANTS.keys()))
    def test_o2_calculation_all_fuels(self, fuel_type):
        """Test O2 calculation works for all fuel types."""
        o2 = compute_excess_o2(lambda_val=1.2, fuel_type=fuel_type)
        assert 0 < o2 < 20, f"Invalid O2 for {fuel_type}: {o2}"

    @pytest.mark.parametrize("fuel_type", list(FUEL_O2_CONSTANTS.keys()))
    def test_lambda_inference_all_fuels(self, fuel_type):
        """Test lambda inference works for all fuel types."""
        lambda_val = infer_lambda_from_o2(stack_o2=4.0, fuel_type=fuel_type)
        assert 1.0 < lambda_val < 3.0, f"Invalid lambda for {fuel_type}: {lambda_val}"


class TestProvenanceTracking:
    """Tests for SHA-256 provenance hash tracking."""

    def test_stoichiometry_result_has_hash(self):
        """Test StoichiometryResult includes provenance hash."""
        result = StoichiometryResult(
            stoichiometric_afr=17.2,
            stoichiometric_afr_vol=9.6,
            lambda_value=1.15,
            excess_air_percent=15.0,
            excess_o2_percent=2.8,
            calculation_method="flow_rates",
            fuel_type="natural_gas"
        )
        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 16  # First 16 chars of SHA-256

    def test_hash_determinism(self):
        """Test same inputs produce same hash."""
        result1 = StoichiometryResult(
            stoichiometric_afr=17.2,
            stoichiometric_afr_vol=9.6,
            lambda_value=1.15,
            excess_air_percent=15.0,
            excess_o2_percent=2.8,
            calculation_method="flow_rates",
            fuel_type="natural_gas"
        )
        result2 = StoichiometryResult(
            stoichiometric_afr=17.2,
            stoichiometric_afr_vol=9.6,
            lambda_value=1.15,
            excess_air_percent=15.0,
            excess_o2_percent=2.8,
            calculation_method="flow_rates",
            fuel_type="natural_gas"
        )
        assert result1.provenance_hash == result2.provenance_hash

    def test_different_inputs_different_hash(self):
        """Test different inputs produce different hashes."""
        result1 = StoichiometryResult(
            stoichiometric_afr=17.2,
            stoichiometric_afr_vol=9.6,
            lambda_value=1.15,
            excess_air_percent=15.0,
            excess_o2_percent=2.8,
            calculation_method="flow_rates",
            fuel_type="natural_gas"
        )
        result2 = StoichiometryResult(
            stoichiometric_afr=17.2,
            stoichiometric_afr_vol=9.6,
            lambda_value=1.20,  # Different lambda
            excess_air_percent=20.0,
            excess_o2_percent=3.5,
            calculation_method="flow_rates",
            fuel_type="natural_gas"
        )
        assert result1.provenance_hash != result2.provenance_hash


class TestValidationFunctions:
    """Tests for input validation functions."""

    def test_valid_inputs_pass(self):
        """Test valid inputs pass validation."""
        is_valid, errors = validate_stoichiometry_inputs(
            lambda_val=1.15,
            excess_air=15.0,
            stack_o2=3.0
        )
        assert is_valid
        assert len(errors) == 0

    def test_lambda_too_low(self):
        """Test lambda too low fails validation."""
        is_valid, errors = validate_stoichiometry_inputs(lambda_val=0.3)
        assert not is_valid
        assert any("too low" in e for e in errors)

    def test_lambda_too_high(self):
        """Test lambda too high fails validation."""
        is_valid, errors = validate_stoichiometry_inputs(lambda_val=6.0)
        assert not is_valid
        assert any("too high" in e for e in errors)

    def test_excess_air_too_low(self):
        """Test excess air too low fails validation."""
        is_valid, errors = validate_stoichiometry_inputs(excess_air=-60.0)
        assert not is_valid
        assert any("too low" in e for e in errors)

    def test_negative_o2_fails(self):
        """Test negative O2 fails validation."""
        is_valid, errors = validate_stoichiometry_inputs(stack_o2=-1.0)
        assert not is_valid
        assert any("cannot be negative" in e for e in errors)

    def test_inconsistent_lambda_excess_air(self):
        """Test inconsistent lambda and excess air fails."""
        is_valid, errors = validate_stoichiometry_inputs(
            lambda_val=1.15,
            excess_air=50.0  # Should be 15% for lambda=1.15
        )
        assert not is_valid
        assert any("inconsistent" in e.lower() for e in errors)


class TestConsistencyCheck:
    """Tests for stoichiometry consistency checking."""

    def test_consistent_values(self):
        """Test consistent O2 and lambda pass check."""
        is_consistent, msg = check_stoichiometry_consistency(
            lambda_val=1.15,
            stack_o2=2.8,
            fuel_type="natural_gas",
            tolerance=0.5
        )
        assert is_consistent

    def test_inconsistent_values(self):
        """Test inconsistent O2 and lambda fail check."""
        is_consistent, msg = check_stoichiometry_consistency(
            lambda_val=1.15,
            stack_o2=8.0,  # Way too high for lambda=1.15
            fuel_type="natural_gas",
            tolerance=0.5
        )
        assert not is_consistent
        assert "Inconsistent" in msg


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_very_lean_operation(self):
        """Test very lean operation (high lambda)."""
        o2 = compute_excess_o2(lambda_val=3.0, fuel_type="natural_gas")
        assert o2 > 10.0, "Very lean should have high O2"

    def test_very_high_o2_caps_lambda(self):
        """Test very high O2 caps lambda at reasonable value."""
        lambda_val = infer_lambda_from_o2(stack_o2=19.0, fuel_type="natural_gas")
        assert lambda_val <= 5.0, "Lambda should be capped"

    def test_float_precision_handling(self):
        """Test float precision is handled correctly."""
        # Test with values that might cause float precision issues
        lambda_val = compute_lambda(actual_af=10.000001, stoich_af=10.0)
        assert lambda_val > 1.0
        assert lambda_val < 1.0001

    def test_composition_near_100(self, natural_gas_composition):
        """Test composition summing to nearly 100% is accepted."""
        # Modify composition to sum to 100.4% (within tolerance)
        modified = natural_gas_composition.copy()
        modified["CH4"] = 94.4
        stoich_air = compute_stoichiometric_air(modified)
        assert stoich_air > 0


class TestPhysicsValidation:
    """Tests validating physical correctness of calculations."""

    def test_higher_lambda_means_higher_o2(self):
        """Test O2 increases monotonically with lambda."""
        o2_1 = compute_excess_o2(1.1, "natural_gas")
        o2_2 = compute_excess_o2(1.2, "natural_gas")
        o2_3 = compute_excess_o2(1.3, "natural_gas")
        assert o2_1 < o2_2 < o2_3, "O2 should increase with lambda"

    def test_o2_approaches_max_asymptotically(self):
        """Test O2 approaches but doesn't exceed max."""
        o2 = compute_excess_o2(lambda_val=5.0, fuel_type="natural_gas")
        max_o2 = FUEL_O2_CONSTANTS["natural_gas"]["max_o2"]
        assert o2 < max_o2, "O2 should not exceed fuel-specific max"

    def test_hydrogen_lower_afr_than_methane(self):
        """Test hydrogen has lower AFR than methane (less air needed)."""
        h2_composition = {"H2": 100.0}
        ch4_composition = {"CH4": 100.0}

        afr_h2 = compute_stoichiometric_air(h2_composition)
        afr_ch4 = compute_stoichiometric_air(ch4_composition)

        assert afr_h2 < afr_ch4, "H2 should need less air than CH4"


class TestFuelO2Constants:
    """Tests for fuel O2 constants validity."""

    def test_all_k1_positive(self):
        """Test all k1 coefficients are positive."""
        for fuel, constants in FUEL_O2_CONSTANTS.items():
            assert constants["k1"] > 0, f"{fuel} has invalid k1"

    def test_all_k2_non_negative(self):
        """Test all k2 coefficients are non-negative."""
        for fuel, constants in FUEL_O2_CONSTANTS.items():
            assert constants["k2"] >= 0, f"{fuel} has invalid k2"

    def test_all_max_o2_reasonable(self):
        """Test all max_o2 values are reasonable (< 21%)."""
        for fuel, constants in FUEL_O2_CONSTANTS.items():
            assert 18.0 < constants["max_o2"] < 21.0, f"{fuel} has invalid max_o2"


class TestAdvancedFunctions:
    """Tests for advanced stoichiometry functions."""

    def test_air_flow_for_target_o2(self):
        """Test air flow calculation for target O2."""
        air_flow = compute_air_flow_for_target_o2(
            fuel_flow=100.0,
            target_o2=3.0,
            fuel_type=FuelType.NATURAL_GAS
        )
        assert air_flow > 0, "Air flow should be positive"
        # At 3% O2, air flow should be ~10-15% above stoichiometric
        assert 1000 < air_flow < 2000

    def test_fuel_flow_for_duty(self):
        """Test fuel flow calculation for thermal duty."""
        fuel_flow = compute_fuel_flow_for_target_duty(
            target_duty=10.0,  # 10 MW
            fuel_type=FuelType.NATURAL_GAS,
            combustion_efficiency=0.90
        )
        assert fuel_flow > 0, "Fuel flow should be positive"

    def test_flue_gas_flow(self):
        """Test flue gas flow calculation."""
        wet, dry = compute_flue_gas_flow(
            fuel_flow=100.0,
            lambda_val=1.15,
            fuel_type=FuelType.NATURAL_GAS
        )
        assert wet > dry, "Wet flue gas should exceed dry"
        assert dry > 0, "Dry flue gas should be positive"


# ============================================================================
# Golden Tests - Reference calculations for regression testing
# ============================================================================

class TestGoldenCalculations:
    """Golden tests with known reference values."""

    def test_golden_natural_gas_stoich(self):
        """Golden test: Natural gas stoichiometric air."""
        composition = {"CH4": 95.0, "C2H6": 3.0, "CO2": 1.0, "N2": 1.0}
        stoich_air = compute_stoichiometric_air(composition)
        # Known reference: ~9.6 Nm3/Nm3 for typical natural gas
        assert 9.4 < stoich_air < 9.9

    def test_golden_lambda_calculation(self):
        """Golden test: Lambda calculation."""
        lambda_val = compute_lambda(actual_af=11.04, stoich_af=9.6)
        assert abs(lambda_val - 1.15) < 0.001

    def test_golden_excess_air(self):
        """Golden test: Excess air from lambda."""
        excess = compute_excess_air_percent(lambda_val=1.15)
        assert excess == 15.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
