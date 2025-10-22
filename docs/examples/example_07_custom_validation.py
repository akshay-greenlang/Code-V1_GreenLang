"""
Example 07: Custom Validation

This example demonstrates advanced input validation patterns.
You'll learn:
- How to implement custom validation logic
- How to provide detailed validation error messages
- How to validate complex business rules
- How to use Pydantic-style validation
"""

from greenlang.agents import BaseAgent, AgentConfig, AgentResult, BaseCalculator, CalculatorConfig
from typing import Dict, Any, List


class BuildingEmissionsValidator(BaseAgent):
    """
    Validate building emissions data with comprehensive checks.

    This agent demonstrates:
    - Multi-level validation
    - Detailed error messages
    - Business rule validation
    - Data quality checks
    """

    def __init__(self):
        config = AgentConfig(
            name="BuildingEmissionsValidator",
            description="Comprehensive building data validation",
            enable_metrics=True
        )
        super().__init__(config)

        # Define validation rules
        self.valid_fuel_types = ['electricity', 'natural_gas', 'diesel', 'propane', 'coal']
        self.valid_units = {
            'electricity': ['kWh', 'MWh'],
            'natural_gas': ['therms', 'cubic_feet', 'cubic_meters'],
            'diesel': ['liters', 'gallons'],
            'propane': ['liters', 'gallons'],
            'coal': ['kg', 'tons']
        }
        self.max_consumption_limits = {
            'electricity': 10000000,  # kWh
            'natural_gas': 100000,    # therms
            'diesel': 50000,          # liters
        }

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """
        Validate building data.

        Args:
            input_data: Building data to validate

        Returns:
            AgentResult with validation results and detailed errors
        """
        validation_errors = []
        warnings = []

        # Level 1: Required fields validation
        required_fields = ['building_id', 'name', 'area_sqft', 'fuels']
        for field in required_fields:
            if field not in input_data:
                validation_errors.append(f"Missing required field: {field}")

        if validation_errors:
            return AgentResult(
                success=False,
                error="Required field validation failed",
                metadata={
                    'validation_errors': validation_errors,
                    'validation_stage': 'required_fields'
                }
            )

        # Level 2: Data type validation
        if not isinstance(input_data['building_id'], str):
            validation_errors.append("building_id must be a string")

        if not isinstance(input_data['area_sqft'], (int, float)):
            validation_errors.append("area_sqft must be numeric")
        elif input_data['area_sqft'] <= 0:
            validation_errors.append("area_sqft must be positive")

        if not isinstance(input_data['fuels'], list):
            validation_errors.append("fuels must be a list")
        elif len(input_data['fuels']) == 0:
            validation_errors.append("fuels list cannot be empty")

        if validation_errors:
            return AgentResult(
                success=False,
                error="Data type validation failed",
                metadata={
                    'validation_errors': validation_errors,
                    'validation_stage': 'data_types'
                }
            )

        # Level 3: Fuel data validation
        for idx, fuel in enumerate(input_data['fuels']):
            prefix = f"fuels[{idx}]"

            # Check required fuel fields
            if 'fuel_type' not in fuel:
                validation_errors.append(f"{prefix}: Missing fuel_type")
                continue

            if 'consumption' not in fuel:
                validation_errors.append(f"{prefix}: Missing consumption")
                continue

            if 'unit' not in fuel:
                validation_errors.append(f"{prefix}: Missing unit")
                continue

            fuel_type = fuel['fuel_type']
            consumption = fuel['consumption']
            unit = fuel['unit']

            # Validate fuel type
            if fuel_type not in self.valid_fuel_types:
                validation_errors.append(
                    f"{prefix}: Invalid fuel_type '{fuel_type}'. "
                    f"Must be one of: {', '.join(self.valid_fuel_types)}"
                )

            # Validate consumption
            if not isinstance(consumption, (int, float)):
                validation_errors.append(f"{prefix}: consumption must be numeric")
            elif consumption < 0:
                validation_errors.append(f"{prefix}: consumption cannot be negative")
            elif consumption == 0:
                warnings.append(f"{prefix}: consumption is zero")

            # Validate unit matches fuel type
            if fuel_type in self.valid_units:
                if unit not in self.valid_units[fuel_type]:
                    validation_errors.append(
                        f"{prefix}: Invalid unit '{unit}' for {fuel_type}. "
                        f"Must be one of: {', '.join(self.valid_units[fuel_type])}"
                    )

            # Check consumption limits (data quality)
            if fuel_type in self.max_consumption_limits:
                max_limit = self.max_consumption_limits[fuel_type]
                if consumption > max_limit:
                    warnings.append(
                        f"{prefix}: Consumption {consumption} {unit} exceeds typical maximum "
                        f"({max_limit} {unit}). Please verify this value."
                    )

        if validation_errors:
            return AgentResult(
                success=False,
                error="Fuel data validation failed",
                metadata={
                    'validation_errors': validation_errors,
                    'warnings': warnings,
                    'validation_stage': 'fuel_data'
                }
            )

        # Level 4: Business rule validation
        total_area = input_data['area_sqft']

        # Check for unrealistic building sizes
        if total_area > 1000000:
            warnings.append(
                f"Building area {total_area:,} sqft is very large. "
                f"Ensure this is correct."
            )
        elif total_area < 100:
            warnings.append(
                f"Building area {total_area} sqft is very small. "
                f"Ensure this is correct."
            )

        # Check for duplicate fuel types
        fuel_types = [f['fuel_type'] for f in input_data['fuels']]
        duplicate_fuels = set([ft for ft in fuel_types if fuel_types.count(ft) > 1])
        if duplicate_fuels:
            warnings.append(
                f"Duplicate fuel types found: {', '.join(duplicate_fuels)}. "
                f"Consider aggregating these entries."
            )

        # Success with warnings
        return AgentResult(
            success=True,
            data={
                'validated': True,
                'warnings_count': len(warnings),
                'fuels_count': len(input_data['fuels'])
            },
            metadata={
                'warnings': warnings,
                'validation_passed': True
            }
        )

    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """Basic input validation (data structure)."""
        return isinstance(input_data, dict)


def main():
    """Run the example."""
    print("=" * 60)
    print("Example 07: Custom Validation")
    print("=" * 60)
    print()

    validator = BuildingEmissionsValidator()

    # Test 1: Valid data
    print("Test 1: Valid Building Data")
    print("-" * 40)

    valid_data = {
        'building_id': 'B001',
        'name': 'Office Building A',
        'area_sqft': 25000,
        'fuels': [
            {'fuel_type': 'electricity', 'consumption': 50000, 'unit': 'kWh'},
            {'fuel_type': 'natural_gas', 'consumption': 1000, 'unit': 'therms'}
        ]
    }

    result = validator.run(valid_data)

    if result.success:
        print(f"✓ Validation passed")
        if result.metadata.get('warnings'):
            print(f"  Warnings ({len(result.metadata['warnings'])}):")
            for warning in result.metadata['warnings']:
                print(f"    ⚠ {warning}")
    else:
        print(f"✗ Validation failed: {result.error}")
    print()

    # Test 2: Missing required fields
    print("Test 2: Missing Required Fields")
    print("-" * 40)

    invalid_data1 = {
        'building_id': 'B002',
        'name': 'Office Building B'
        # Missing area_sqft and fuels
    }

    result = validator.run(invalid_data1)

    if not result.success:
        print(f"✓ Correctly rejected invalid data")
        print(f"  Error: {result.error}")
        print(f"  Validation stage: {result.metadata.get('validation_stage')}")
        print(f"  Errors:")
        for error in result.metadata.get('validation_errors', []):
            print(f"    • {error}")
    print()

    # Test 3: Invalid fuel type
    print("Test 3: Invalid Fuel Type")
    print("-" * 40)

    invalid_data2 = {
        'building_id': 'B003',
        'name': 'Office Building C',
        'area_sqft': 30000,
        'fuels': [
            {'fuel_type': 'rocket_fuel', 'consumption': 1000, 'unit': 'liters'}  # Invalid
        ]
    }

    result = validator.run(invalid_data2)

    if not result.success:
        print(f"✓ Correctly rejected invalid fuel type")
        print(f"  Error: {result.error}")
        print(f"  Errors:")
        for error in result.metadata.get('validation_errors', []):
            print(f"    • {error}")
    print()

    # Test 4: Data quality warnings
    print("Test 4: Data Quality Warnings")
    print("-" * 40)

    warning_data = {
        'building_id': 'B004',
        'name': 'Huge Warehouse',
        'area_sqft': 1500000,  # Very large
        'fuels': [
            {'fuel_type': 'electricity', 'consumption': 5000000, 'unit': 'kWh'},  # High consumption
            {'fuel_type': 'electricity', 'consumption': 1000000, 'unit': 'kWh'}   # Duplicate
        ]
    }

    result = validator.run(warning_data)

    if result.success:
        print(f"✓ Validation passed with warnings")
        print(f"  Warnings ({len(result.metadata.get('warnings', []))}):")
        for warning in result.metadata.get('warnings', []):
            print(f"    ⚠ {warning}")
    print()

    # Test 5: Multiple validation errors
    print("Test 5: Multiple Validation Errors")
    print("-" * 40)

    invalid_data3 = {
        'building_id': 'B005',
        'name': 'Bad Data Building',
        'area_sqft': -1000,  # Negative
        'fuels': [
            {'fuel_type': 'electricity', 'consumption': 'not a number', 'unit': 'kWh'},  # Wrong type
            {'fuel_type': 'natural_gas', 'consumption': -500, 'unit': 'therms'},  # Negative
            {'consumption': 1000, 'unit': 'liters'}  # Missing fuel_type
        ]
    }

    result = validator.run(invalid_data3)

    if not result.success:
        print(f"✓ Correctly rejected data with multiple errors")
        print(f"  Total errors: {len(result.metadata.get('validation_errors', []))}")
        print(f"  Errors:")
        for error in result.metadata.get('validation_errors', []):
            print(f"    • {error}")
    print()

    # Statistics
    print("Validation Statistics:")
    print("-" * 40)
    stats = validator.get_stats()
    print(f"  Total validations: {stats['executions']}")
    print(f"  Success rate: {stats['success_rate']}%")
    print()

    print("=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
