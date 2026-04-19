"""
Example Golden Test Runner

This demonstrates how to use the golden test framework to validate
climate calculations against expert-validated answers.
"""

import sys
from pathlib import Path

try:
    from greenlang.testing.golden_test_runner import GoldenTestRunner
except ImportError:
    from greenlang.testing.golden_tests import GoldenTestRunner

try:
    from greenlang.validation.emission_factors import EmissionFactorDB
except ImportError:
    from greenlang.data.emission_factor_db import EmissionFactorDB


def calculate_emissions(inputs: dict) -> float:
    """
    Example calculation function that uses emission factor database.

    This is where you would implement your actual calculation logic.
    For this example, we'll implement basic Scope 1, 2, and 3 calculations.
    """
    db = EmissionFactorDB()

    # Scope 1 - Stationary Combustion
    if 'fuel_type' in inputs and 'fuel_quantity' in inputs:
        fuel_type = inputs['fuel_type']
        quantity = inputs['fuel_quantity']
        region = inputs.get('region', 'UK')

        # Get emission factor
        factor = db.get_factor(fuel_type, region=region)
        if not factor:
            raise ValueError(f"Emission factor not found for {fuel_type} in {region}")

        # Calculate emissions
        emissions_kg_co2e = quantity * factor.factor_value
        return emissions_kg_co2e

    # Scope 1 - Mixed Fuels
    if 'natural_gas_kwh' in inputs and 'diesel_liters' in inputs:
        region = inputs.get('region', 'UK')

        # Natural gas
        ng_factor = db.get_factor('natural_gas', region=region)
        ng_emissions = inputs['natural_gas_kwh'] * ng_factor.factor_value

        # Diesel
        diesel_factor = db.get_factor('diesel', region=region)
        diesel_emissions = inputs['diesel_liters'] * diesel_factor.factor_value

        return ng_emissions + diesel_emissions

    # Scope 2 - Electricity
    if 'electricity_kwh' in inputs:
        electricity_kwh = inputs['electricity_kwh']
        region = inputs.get('region', 'UK')

        # Get electricity factor
        factor = db.get_factor('electricity', region=region)
        if not factor:
            raise ValueError(f"Electricity factor not found for {region}")

        emissions_kg_co2e = electricity_kwh * factor.factor_value
        return emissions_kg_co2e

    # Scope 3 - Transport
    if 'transport_mode' in inputs:
        transport_mode = inputs['transport_mode']
        distance_km = inputs.get('distance_km', 0)
        region = inputs.get('region', 'UK')

        # For freight, multiply by cargo weight
        if 'cargo_tonnes' in inputs:
            cargo_tonnes = inputs['cargo_tonnes']

            # Look up transport factor
            factor = db.get_factor(transport_mode, region=region)
            if not factor:
                raise ValueError(f"Transport factor not found for {transport_mode}")

            # Calculate tonne-km emissions
            emissions_kg_co2e = cargo_tonnes * distance_km * factor.factor_value
            return emissions_kg_co2e

        else:
            # Distance-based (e.g., cars, HGV)
            factor = db.get_factor(transport_mode, region=region)
            if not factor:
                raise ValueError(f"Transport factor not found for {transport_mode}")

            emissions_kg_co2e = distance_km * factor.factor_value
            return emissions_kg_co2e

    # CBAM - Embedded Emissions
    if 'material' in inputs and 'quantity_kg' in inputs:
        material = inputs['material']
        quantity_kg = inputs['quantity_kg']
        embedded_factor = inputs.get('embedded_factor')

        if embedded_factor:
            # Use provided factor
            emissions_kg_co2e = quantity_kg * embedded_factor
            return emissions_kg_co2e

        else:
            # Look up factor
            factor = db.get_factor(material, region=inputs.get('production_region', 'GLOBAL'))
            if not factor:
                raise ValueError(f"Material factor not found for {material}")

            emissions_kg_co2e = quantity_kg * factor.factor_value
            return emissions_kg_co2e

    # CBAM - Mixed Materials
    if 'steel_kg' in inputs:
        steel_emissions = inputs['steel_kg'] * inputs['steel_factor']
        aluminium_emissions = inputs.get('aluminium_kg', 0) * inputs.get('aluminium_factor', 0)
        cement_emissions = inputs.get('cement_kg', 0) * inputs.get('cement_factor', 0)

        return steel_emissions + aluminium_emissions + cement_emissions

    raise ValueError(f"Unknown input format: {inputs}")


def main():
    """Run all golden tests."""
    # Initialize test runner
    runner = GoldenTestRunner(tolerance=0.01)

    # Load tests from YAML
    yaml_path = Path(__file__).parent / 'scenarios.yaml'
    num_tests = runner.load_tests_from_yaml(yaml_path)
    print(f"\nLoaded {num_tests} golden tests from {yaml_path}\n")

    # Run all tests
    results = runner.run_all_tests(calculate_emissions)

    # Print results
    all_passed = runner.print_results(results, verbose=False)

    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
