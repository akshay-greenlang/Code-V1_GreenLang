"""
Test Validation Hooks

Unit tests for validation hooks to ensure they correctly validate
climate data and catch errors.
"""

import sys
from pathlib import Path

try:
    from greenlang.core.greenlang.validation.hooks import (
        EmissionFactorValidator,
        UnitValidator,
        ThermodynamicValidator,
        GWPValidator,
        ValidationLevel
    )
except ImportError:
    # Try alternative import path
    from greenlang.data.validation import (
        EmissionFactorValidator,
        UnitValidator,
        ThermodynamicValidator,
        GWPValidator,
        ValidationLevel
    )


def test_emission_factor_validator():
    """Test emission factor validation."""
    print("\n" + "="*70)
    print("Testing Emission Factor Validator")
    print("="*70)

    validator = EmissionFactorValidator()

    # Test 1: Valid diesel factor
    result = validator.validate_factor('diesel', 2.687, 'UK', 'DEFRA')
    assert result.is_valid
    assert result.level == ValidationLevel.INFO
    print(f"[PASS] Test 1: {result.message}")

    # Test 2: Valid natural gas factor
    result = validator.validate_factor('natural_gas', 0.18385, 'UK', 'DEFRA')
    assert result.is_valid
    print(f"[PASS] Test 2 PASSED: {result.message}")

    # Test 3: Factor outside range (should fail)
    result = validator.validate_factor('diesel', 5.0, 'UK', 'DEFRA')
    assert not result.is_valid
    assert result.level == ValidationLevel.ERROR
    print(f"[PASS] Test 3 PASSED: {result.message}")

    # Test 4: Factor with deviation warning (>10% deviation but within range)
    result = validator.validate_factor('natural_gas', 0.202, 'UK', 'DEFRA')  # +9.8% deviation from typical 0.184 (close to 10%)
    # Actually, let's use coal which has wider range
    result = validator.validate_factor('coal', 2.05, 'UK', 'DEFRA')  # -9.6% deviation from 2.269, within [2.0, 2.5]
    # This won't trigger warning at 9.6%, so use value further from typical
    result = validator.validate_factor('electricity_uk', 0.17, 'UK', 'DEFRA')  # -11.9% from 0.193, within [0.15, 0.30]
    assert result.is_valid
    assert result.level == ValidationLevel.WARNING
    print(f"[PASS] Test 4 PASSED: {result.message}")

    # Test 5: Unknown fuel type (warning)
    result = validator.validate_factor('unknown_fuel', 1.0, 'UK', 'DEFRA')
    assert not result.is_valid
    assert result.level == ValidationLevel.WARNING
    print(f"[PASS] Test 5 PASSED: {result.message}")

    # Test 6: UK electricity
    result = validator.validate_factor('electricity', 0.193, 'UK', 'DEFRA')
    assert result.is_valid
    print(f"[PASS] Test 6 PASSED: {result.message}")

    # Test 7: US electricity
    result = validator.validate_factor('electricity', 0.417, 'US', 'EPA')
    assert result.is_valid
    print(f"[PASS] Test 7 PASSED: {result.message}")

    print("\n[PASS] All Emission Factor Validator tests PASSED\n")


def test_unit_validator():
    """Test unit validation."""
    print("\n" + "="*70)
    print("Testing Unit Validator")
    print("="*70)

    validator = UnitValidator()

    # Test 1: Valid energy unit
    result = validator.validate_unit('kWh')
    assert result.is_valid
    print(f"[PASS] Test 1: {result.message}")

    # Test 2: Valid emission unit
    result = validator.validate_unit('tCO2e')
    assert result.is_valid
    print(f"[PASS] Test 2 PASSED: {result.message}")

    # Test 3: Invalid unit
    result = validator.validate_unit('invalid_unit')
    assert not result.is_valid
    print(f"[PASS] Test 3 PASSED: {result.message}")

    # Test 4: Valid conversion (kWh to MWh)
    result = validator.validate_conversion(1000, 'kWh', 'MWh', 1.0)
    assert result.is_valid
    print(f"[PASS] Test 4 PASSED: {result.message}")

    # Test 5: Valid conversion (kg to tonnes)
    result = validator.validate_conversion(1000, 'kgCO2e', 'tCO2e', 1.0)
    assert result.is_valid
    print(f"[PASS] Test 5 PASSED: {result.message}")

    # Test 6: Invalid conversion (energy to mass)
    result = validator.validate_conversion(100, 'kWh', 'kg')
    assert not result.is_valid
    print(f"[PASS] Test 6 PASSED: Cannot convert between incompatible units")

    print("\n[PASS] All Unit Validator tests PASSED\n")


def test_thermodynamic_validator():
    """Test thermodynamic validation."""
    print("\n" + "="*70)
    print("Testing Thermodynamic Validator")
    print("="*70)

    validator = ThermodynamicValidator()

    # Test 1: Valid efficiency (80%)
    result = validator.validate_efficiency(0.80, 'boiler')
    assert result.is_valid
    print(f"[PASS] Test 1: {result.message}")

    # Test 2: Valid efficiency (percentage format)
    result = validator.validate_efficiency(85.0, 'boiler')  # 85%
    assert result.is_valid
    print(f"[PASS] Test 2 PASSED: {result.message}")

    # Test 3: Invalid efficiency (>100%)
    result = validator.validate_efficiency(120.0, 'boiler')
    assert not result.is_valid
    print(f"[PASS] Test 3 PASSED: {result.message}")

    # Test 4: Valid COP (heat pump)
    result = validator.validate_efficiency(3.5, 'heat_pump', is_cop=True)
    assert result.is_valid
    print(f"[PASS] Test 4 PASSED: {result.message}")

    # Test 5: Invalid COP (<1)
    result = validator.validate_efficiency(0.8, 'heat_pump', is_cop=True)
    assert not result.is_valid
    print(f"[PASS] Test 5 PASSED: {result.message}")

    # Test 6: Negative efficiency
    result = validator.validate_efficiency(-0.1, 'generic')
    assert not result.is_valid
    print(f"[PASS] Test 6 PASSED: {result.message}")

    # Test 7: Valid energy balance
    result = validator.validate_energy_balance(100, 80, 0.80)
    assert result.is_valid
    print(f"[PASS] Test 7 PASSED: {result.message}")

    # Test 8: Invalid energy balance
    result = validator.validate_energy_balance(100, 90, 0.80)  # Should be 80, not 90
    assert not result.is_valid
    print(f"[PASS] Test 8 PASSED: {result.message}")

    print("\n[PASS] All Thermodynamic Validator tests PASSED\n")


def test_gwp_validator():
    """Test GWP validation."""
    print("\n" + "="*70)
    print("Testing GWP Validator")
    print("="*70)

    validator = GWPValidator()

    # Test 1: CO2 (always 1)
    result = validator.validate_gwp('CO2', 1.0, 'AR6')
    assert result.is_valid
    print(f"[PASS] Test 1: {result.message}")

    # Test 2: CH4 (AR6)
    result = validator.validate_gwp('CH4', 29.8, 'AR6')
    assert result.is_valid
    print(f"[PASS] Test 2 PASSED: {result.message}")

    # Test 3: N2O (AR6)
    result = validator.validate_gwp('N2O', 273, 'AR6')
    assert result.is_valid
    print(f"[PASS] Test 3 PASSED: {result.message}")

    # Test 4: CH4 (AR5) - different value
    result = validator.validate_gwp('CH4', 28, 'AR5')
    assert result.is_valid
    print(f"[PASS] Test 4 PASSED: {result.message}")

    # Test 5: Invalid GWP value
    result = validator.validate_gwp('CH4', 50, 'AR6')  # Too high
    assert not result.is_valid
    print(f"[PASS] Test 5 PASSED: {result.message}")

    # Test 6: SF6 (very high GWP)
    result = validator.validate_gwp('SF6', 25200, 'AR6')
    assert result.is_valid
    print(f"[PASS] Test 6 PASSED: {result.message}")

    # Test 7: Unknown gas
    result = validator.validate_gwp('unknown_gas', 100, 'AR6')
    assert not result.is_valid
    print(f"[PASS] Test 7 PASSED: {result.message}")

    print("\n[PASS] All GWP Validator tests PASSED\n")


def main():
    """Run all validation hook tests."""
    print("\n" + "="*70)
    print("VALIDATION HOOK TEST SUITE")
    print("="*70)

    try:
        test_emission_factor_validator()
        test_unit_validator()
        test_thermodynamic_validator()
        test_gwp_validator()

        print("\n" + "="*70)
        print("ALL VALIDATION TESTS PASSED [PASS]")
        print("="*70 + "\n")
        return 0

    except AssertionError as e:
        print(f"\n[FAIL] TEST FAILED: {e}\n")
        return 1
    except Exception as e:
        print(f"\n[FAIL] ERROR: {e}\n")
        return 1


if __name__ == '__main__':
    sys.exit(main())
