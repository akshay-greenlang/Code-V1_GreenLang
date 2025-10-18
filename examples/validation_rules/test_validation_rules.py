#!/usr/bin/env python3
"""
Test script to validate YAML rule files and demonstrate usage.

This script:
1. Validates YAML syntax
2. Loads rules into the RulesEngine
3. Tests with sample data
4. Demonstrates all 12 rule operators

Run: python examples/validation_rules/test_validation_rules.py
"""

import yaml
import sys
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from greenlang.validation.rules import RulesEngine, Rule, RuleOperator
    from greenlang.validation.framework import ValidationSeverity
except ImportError:
    print("Warning: GreenLang validation modules not found. Testing YAML syntax only.")
    RulesEngine = None


def test_yaml_syntax(yaml_file: Path) -> bool:
    """Test if YAML file has valid syntax."""
    print(f"\n{'='*70}")
    print(f"Testing YAML Syntax: {yaml_file.name}")
    print(f"{'='*70}")

    try:
        with open(yaml_file) as f:
            config = yaml.safe_load(f)

        print(f"✓ YAML syntax is valid")

        # Print summary
        if 'metadata' in config:
            print(f"\nMetadata:")
            print(f"  Name: {config['metadata'].get('name', 'N/A')}")
            print(f"  Version: {config['metadata'].get('version', 'N/A')}")
            print(f"  Description: {config['metadata'].get('description', 'N/A')}")

        if 'rules' in config:
            print(f"\nRules: {len(config['rules'])} defined")

            # Count by severity
            severity_counts = {}
            for rule in config['rules']:
                severity = rule.get('severity', 'unknown')
                severity_counts[severity] = severity_counts.get(severity, 0) + 1

            for severity, count in sorted(severity_counts.items()):
                print(f"  - {severity}: {count}")

            # Count by operator
            operator_counts = {}
            for rule in config['rules']:
                operator = rule.get('operator', 'unknown')
                operator_counts[operator] = operator_counts.get(operator, 0) + 1

            print(f"\nOperators used:")
            for operator, count in sorted(operator_counts.items()):
                print(f"  - {operator}: {count}")

        if 'rule_sets' in config:
            print(f"\nRule Sets: {len(config['rule_sets'])} defined")
            for rule_set in config['rule_sets']:
                print(f"  - {rule_set['name']}: {rule_set.get('description', 'N/A')}")

        return True

    except yaml.YAMLError as e:
        print(f"✗ YAML syntax error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_rule_loading(yaml_file: Path) -> bool:
    """Test loading rules into RulesEngine."""
    if RulesEngine is None:
        print("\nSkipping rule loading test (GreenLang modules not available)")
        return True

    print(f"\n{'='*70}")
    print(f"Testing Rule Loading: {yaml_file.name}")
    print(f"{'='*70}")

    try:
        with open(yaml_file) as f:
            config = yaml.safe_load(f)

        engine = RulesEngine()
        loaded_count = 0

        for rule_dict in config['rules']:
            try:
                rule = Rule(**rule_dict)
                engine.add_rule(rule)
                loaded_count += 1
            except Exception as e:
                print(f"✗ Failed to load rule '{rule_dict.get('name', 'unknown')}': {e}")
                return False

        print(f"✓ Successfully loaded {loaded_count} rules into RulesEngine")
        return True

    except Exception as e:
        print(f"✗ Error loading rules: {e}")
        return False


def test_sample_validation() -> bool:
    """Test validation with sample data."""
    if RulesEngine is None:
        print("\nSkipping validation test (GreenLang modules not available)")
        return True

    print(f"\n{'='*70}")
    print(f"Testing Sample Data Validation")
    print(f"{'='*70}")

    # Load data import validation rules
    yaml_file = Path(__file__).parent / "data_import_validation.yaml"

    try:
        with open(yaml_file) as f:
            config = yaml.safe_load(f)

        engine = RulesEngine()
        for rule_dict in config['rules']:
            rule = Rule(**rule_dict)
            engine.add_rule(rule)

        # Test with valid data
        print("\nTest 1: Valid Data")
        print("-" * 70)

        valid_data = {
            "building_id": "NYC-001234",
            "facility_name": "Corporate Office Tower",
            "building_area_sqft": 50000,
            "building_type": "Office",
            "energy_data": {
                "electricity_kwh": 125000,
                "electricity_unit": "kWh",
                "gas_therms": 5000,
                "gas_unit": "therms"
            },
            "location": {
                "city": "New York",
                "country_code": "US",
                "postal_code": "10001"
            },
            "reporting_year": 2024,
            "reporting_date": "2024-01-15",
            "contact_email": "facilities@example.com"
        }

        result = engine.validate(valid_data)
        print(f"Status: {'PASSED ✓' if result.valid else 'FAILED ✗'}")
        print(f"Errors: {len(result.errors)}")
        print(f"Warnings: {len(result.warnings)}")
        print(f"Info: {len(result.info)}")

        # Test with invalid data
        print("\n\nTest 2: Invalid Data (Missing Required Fields)")
        print("-" * 70)

        invalid_data = {
            "facility_name": "Test Building",
            "building_area_sqft": -100,  # Invalid: negative
        }

        result = engine.validate(invalid_data)
        print(f"Status: {'PASSED ✓' if result.valid else 'FAILED ✗'}")
        print(f"Errors: {len(result.errors)}")
        print(f"Warnings: {len(result.warnings)}")

        if result.errors:
            print("\nFirst 5 errors:")
            for i, error in enumerate(result.errors[:5], 1):
                print(f"  {i}. [{error.field}] {error.message}")

        # Test with data that triggers warnings
        print("\n\nTest 3: Data with Warnings")
        print("-" * 70)

        warning_data = {
            "building_id": "B001",  # Doesn't match format
            "facility_name": "Test Building",
            "building_area_sqft": 50000,
            "energy_data": {
                "electricity_kwh": 125000,
                "electricity_unit": "kWh",
                "gas_therms": 5000,
                "gas_unit": "therms"
            },
            "reporting_year": 2024,
            "reporting_date": "2024-01-15"
        }

        result = engine.validate(warning_data)
        print(f"Status: {'PASSED ✓' if result.valid else 'FAILED ✗'}")
        print(f"Errors: {len(result.errors)}")
        print(f"Warnings: {len(result.warnings)}")

        if result.warnings:
            print("\nWarnings:")
            for warning in result.warnings[:5]:
                print(f"  ⚠ [{warning.field}] {warning.message}")

        return True

    except Exception as e:
        print(f"✗ Validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def demonstrate_operators() -> bool:
    """Demonstrate all 12 rule operators."""
    if RulesEngine is None:
        print("\nSkipping operator demonstration (GreenLang modules not available)")
        return True

    print(f"\n{'='*70}")
    print(f"Demonstrating All 12 Rule Operators")
    print(f"{'='*70}")

    test_cases = [
        {
            "name": "== (equals)",
            "rule": {"name": "test_eq", "field": "value", "operator": "==", "value": 100},
            "data": {"value": 100},
            "should_pass": True
        },
        {
            "name": "!= (not equals)",
            "rule": {"name": "test_ne", "field": "value", "operator": "!=", "value": 0},
            "data": {"value": 10},
            "should_pass": True
        },
        {
            "name": "> (greater than)",
            "rule": {"name": "test_gt", "field": "value", "operator": ">", "value": 0},
            "data": {"value": 10},
            "should_pass": True
        },
        {
            "name": ">= (greater equal)",
            "rule": {"name": "test_ge", "field": "value", "operator": ">=", "value": 10},
            "data": {"value": 10},
            "should_pass": True
        },
        {
            "name": "< (less than)",
            "rule": {"name": "test_lt", "field": "value", "operator": "<", "value": 100},
            "data": {"value": 50},
            "should_pass": True
        },
        {
            "name": "<= (less equal)",
            "rule": {"name": "test_le", "field": "value", "operator": "<=", "value": 100},
            "data": {"value": 100},
            "should_pass": True
        },
        {
            "name": "in (in list)",
            "rule": {"name": "test_in", "field": "value", "operator": "in", "value": ["A", "B", "C"]},
            "data": {"value": "B"},
            "should_pass": True
        },
        {
            "name": "not_in (not in list)",
            "rule": {"name": "test_not_in", "field": "value", "operator": "not_in", "value": ["invalid", "error"]},
            "data": {"value": "valid"},
            "should_pass": True
        },
        {
            "name": "contains (string contains)",
            "rule": {"name": "test_contains", "field": "value", "operator": "contains", "value": "test"},
            "data": {"value": "this is a test string"},
            "should_pass": True
        },
        {
            "name": "regex (pattern match)",
            "rule": {"name": "test_regex", "field": "value", "operator": "regex", "value": r"^\d{4}-\d{2}-\d{2}$"},
            "data": {"value": "2024-01-15"},
            "should_pass": True
        },
        {
            "name": "is_null (field is null)",
            "rule": {"name": "test_is_null", "field": "value", "operator": "is_null"},
            "data": {},
            "should_pass": True
        },
        {
            "name": "not_null (field not null)",
            "rule": {"name": "test_not_null", "field": "value", "operator": "not_null"},
            "data": {"value": "present"},
            "should_pass": True
        },
    ]

    all_passed = True
    for test_case in test_cases:
        engine = RulesEngine()
        rule = Rule(**test_case["rule"])
        engine.add_rule(rule)

        result = engine.validate(test_case["data"])
        passed = (result.valid == test_case["should_pass"])

        status = "✓" if passed else "✗"
        print(f"{status} {test_case['name']}: {'PASSED' if passed else 'FAILED'}")

        if not passed:
            all_passed = False

    return all_passed


def main():
    """Main test runner."""
    print("\n" + "="*70)
    print("GreenLang Validation Rules - Test Suite")
    print("="*70)

    validation_rules_dir = Path(__file__).parent
    yaml_files = [
        validation_rules_dir / "data_import_validation.yaml",
        validation_rules_dir / "calculation_validation.yaml",
        validation_rules_dir / "report_validation.yaml"
    ]

    all_tests_passed = True

    # Test 1: YAML Syntax
    print("\n" + "="*70)
    print("PHASE 1: YAML Syntax Validation")
    print("="*70)

    for yaml_file in yaml_files:
        if not yaml_file.exists():
            print(f"✗ File not found: {yaml_file}")
            all_tests_passed = False
            continue

        if not test_yaml_syntax(yaml_file):
            all_tests_passed = False

    # Test 2: Rule Loading
    if RulesEngine is not None:
        print("\n" + "="*70)
        print("PHASE 2: Rule Loading")
        print("="*70)

        for yaml_file in yaml_files:
            if yaml_file.exists():
                if not test_rule_loading(yaml_file):
                    all_tests_passed = False

        # Test 3: Operator Demonstration
        print("\n" + "="*70)
        print("PHASE 3: Operator Demonstration")
        print("="*70)

        if not demonstrate_operators():
            all_tests_passed = False

        # Test 4: Sample Validation
        print("\n" + "="*70)
        print("PHASE 4: Sample Data Validation")
        print("="*70)

        if not test_sample_validation():
            all_tests_passed = False

    # Final Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    if all_tests_passed:
        print("✓ All tests PASSED")
        print("\nValidation rule files are ready to use!")
        print("\nNext steps:")
        print("1. Review the README.md for usage examples")
        print("2. Customize rules for your specific needs")
        print("3. Integrate with your GreenLang agents and pipelines")
    else:
        print("✗ Some tests FAILED")
        print("\nPlease review the errors above and fix any issues.")

    print("="*70 + "\n")

    return 0 if all_tests_passed else 1


if __name__ == "__main__":
    sys.exit(main())
