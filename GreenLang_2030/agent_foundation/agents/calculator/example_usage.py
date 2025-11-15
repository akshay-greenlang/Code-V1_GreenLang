"""
Example Usage - Zero-Hallucination Calculation Engine

Demonstrates all capabilities of the GreenLang calculation engine:
- Scope 1, 2, 3 emission calculations
- CBAM embedded emissions
- Financial calculations (NPV, IRR, ROI)
- Unit conversions
- Validation and compliance checking
"""

from datetime import date
from decimal import Decimal
from pathlib import Path

from formula_engine import FormulaLibrary
from emission_factors import EmissionFactorDatabase, EmissionFactor
from calculation_engine import CalculationEngine
from unit_converter import UnitConverter
from validators import CalculationValidator


def setup_sample_data():
    """Setup sample emission factors for demonstration."""
    db = EmissionFactorDatabase()

    # Sample emission factors
    factors = [
        # Scope 1: Diesel combustion
        EmissionFactor(
            factor_id="defra_2024_diesel_stationary",
            category="scope1",
            subcategory="stationary_combustion",
            activity_type="fuel_combustion",
            material_or_fuel="diesel",
            unit="kg_co2e_per_liter",
            factor_co2=Decimal("2.68"),
            factor_ch4=Decimal("0.0001"),
            factor_n2o=Decimal("0.0001"),
            factor_co2e=Decimal("2.69"),
            region="GB",
            valid_from=date(2024, 1, 1),
            valid_to=date(2024, 12, 31),
            source="DEFRA",
            source_year=2024,
            source_version="2024",
            source_url="https://www.gov.uk/ghg-conversion-factors",
            uncertainty_percentage=5.0,
            data_quality="high",
            notes="DEFRA 2024 conversion factor for diesel in stationary combustion"
        ),

        # Scope 2: Grid electricity (UK)
        EmissionFactor(
            factor_id="defra_2024_grid_electricity_gb",
            category="scope2",
            activity_type="electricity",
            material_or_fuel="grid_electricity",
            unit="kg_co2e_per_kwh",
            factor_co2=Decimal("0.21233"),
            factor_co2e=Decimal("0.21233"),
            region="GB",
            valid_from=date(2024, 1, 1),
            valid_to=date(2024, 12, 31),
            source="DEFRA",
            source_year=2024,
            source_version="2024",
            source_url="https://www.gov.uk/ghg-conversion-factors",
            uncertainty_percentage=8.0,
            data_quality="high",
            notes="UK grid electricity emission factor (location-based)"
        ),

        # Scope 3: Business travel - Short-haul flight
        EmissionFactor(
            factor_id="defra_2024_flight_short_haul",
            category="scope3",
            subcategory="business_travel",
            activity_type="business_travel",
            material_or_fuel="flight_short_haul",
            unit="kg_co2e_per_km",
            factor_co2=Decimal("0.15573"),
            factor_co2e=Decimal("0.15573"),
            region="GLOBAL",
            valid_from=date(2024, 1, 1),
            valid_to=date(2024, 12, 31),
            source="DEFRA",
            source_year=2024,
            source_version="2024",
            source_url="https://www.gov.uk/ghg-conversion-factors",
            uncertainty_percentage=15.0,
            data_quality="medium",
            notes="Short-haul flight (<500km) including radiative forcing"
        ),

        # CBAM: Cement production
        EmissionFactor(
            factor_id="cbam_2024_cement_eu",
            category="cbam",
            activity_type="cement_production",
            material_or_fuel="cement",
            unit="t_co2e_per_tonne",
            factor_co2=Decimal("0.766"),
            factor_co2e=Decimal("0.766"),
            region="EU",
            valid_from=date(2024, 1, 1),
            source="EU CBAM",
            source_year=2024,
            source_version="2024_transitional",
            uncertainty_percentage=10.0,
            data_quality="high",
            notes="EU default value for cement production embedded emissions"
        ),
    ]

    for factor in factors:
        db.insert_factor(factor)

    return db


def example_scope1_calculation():
    """Example: Calculate Scope 1 emissions from diesel combustion."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Scope 1 Stationary Combustion Emissions")
    print("="*80)

    # Setup
    formula_library = FormulaLibrary()
    emission_db = setup_sample_data()
    engine = CalculationEngine(formula_library, emission_db)

    # Load formulas
    formula_count = formula_library.load_formulas()
    print(f"\nLoaded {formula_count} formulas from library")

    # Calculate emissions for 1000 liters of diesel
    try:
        result = engine.calculate(
            formula_id="scope1_stationary_combustion",
            parameters={
                "fuel_quantity": 1000,  # liters
                "fuel_type": "diesel",
                "region": "GB"
            }
        )

        print(f"\nCalculation Result:")
        print(f"  Formula: {result.formula_id} v{result.formula_version}")
        print(f"  Input: 1000 liters of diesel")
        print(f"  Output: {result.output_value} {result.output_unit}")
        print(f"  Processing time: {result.calculation_time_ms:.2f}ms")
        print(f"  Provenance hash: {result.provenance_hash[:16]}...")

        print(f"\nCalculation Steps:")
        for step in result.calculation_steps:
            print(f"  Step {step.step_number}: {step.description}")
            print(f"    Operation: {step.operation}")
            print(f"    Output: {step.output_name} = {step.output_value}")

        print(f"\nEmission Factors Used:")
        for ef in result.emission_factors_used:
            print(f"  - {ef['material_or_fuel']}: {ef['factor_co2e']} {ef['unit']}")
            print(f"    Source: {ef['source']} {ef['source_year']}, Quality: {ef['data_quality']}")

        # Validate result
        validator = CalculationValidator()
        validation = validator.validate_result(result, standard="ghg_protocol")

        print(f"\nValidation:")
        print(f"  Valid: {validation.is_valid}")
        print(f"  Errors: {validation.error_count}")
        print(f"  Warnings: {validation.warning_count}")

        for warning in validation.warnings:
            print(f"  WARNING [{warning.code}]: {warning.message}")

    except Exception as e:
        print(f"ERROR: {e}")

    finally:
        emission_db.close()


def example_scope2_calculation():
    """Example: Calculate Scope 2 emissions from purchased electricity."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Scope 2 Purchased Electricity Emissions")
    print("="*80)

    # Setup
    formula_library = FormulaLibrary()
    emission_db = setup_sample_data()
    engine = CalculationEngine(formula_library, emission_db)

    formula_library.load_formulas()

    # Calculate emissions for 50,000 kWh of electricity
    try:
        result = engine.calculate(
            formula_id="scope2_purchased_electricity",
            parameters={
                "electricity_kwh": 50000,
                "region": "GB",
                "reporting_year": 2024
            }
        )

        print(f"\nCalculation Result:")
        print(f"  Input: 50,000 kWh of grid electricity")
        print(f"  Output: {result.output_value} {result.output_unit}")
        print(f"  Uncertainty: {result.uncertainty_percentage:.1f}%" if result.uncertainty_percentage else "  Uncertainty: N/A")

        # Convert to different unit
        converter = UnitConverter()
        kg_co2e = converter.convert(
            float(result.output_value),
            "t_co2e",
            "kg_co2e",
            precision=1
        )
        print(f"  (Equivalent: {kg_co2e} kg CO2e)")

    except Exception as e:
        print(f"ERROR: {e}")

    finally:
        emission_db.close()


def example_scope3_calculation():
    """Example: Calculate Scope 3 emissions from business travel."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Scope 3 Business Travel Emissions")
    print("="*80)

    # Setup
    formula_library = FormulaLibrary()
    emission_db = setup_sample_data()
    engine = CalculationEngine(formula_library, emission_db)

    formula_library.load_formulas()

    # Calculate emissions for flight
    try:
        result = engine.calculate(
            formula_id="scope3_business_travel",
            parameters={
                "distance_km": 450,  # London to Edinburgh
                "travel_mode": "flight_short_haul",
                "region": "GLOBAL"
            }
        )

        print(f"\nCalculation Result:")
        print(f"  Input: 450 km short-haul flight")
        print(f"  Output: {result.output_value} {result.output_unit}")
        print(f"  Category: Scope 3, Category 6 (Business Travel)")

    except Exception as e:
        print(f"ERROR: {e}")

    finally:
        emission_db.close()


def example_cbam_calculation():
    """Example: Calculate CBAM embedded emissions."""
    print("\n" + "="*80)
    print("EXAMPLE 4: CBAM Embedded Emissions Calculation")
    print("="*80)

    # Setup
    formula_library = FormulaLibrary()
    emission_db = setup_sample_data()
    engine = CalculationEngine(formula_library, emission_db)

    formula_library.load_formulas()

    # Calculate embedded emissions for cement import
    try:
        result = engine.calculate(
            formula_id="cbam_embedded_emissions",
            parameters={
                "production_quantity": 100,  # tonnes of cement
                "material_type": "cement",
                "production_country": "EU",
                "production_process": "cement_production"
            }
        )

        print(f"\nCalculation Result:")
        print(f"  Input: 100 tonnes of cement")
        print(f"  Output: {result.output_value} {result.output_unit}")
        print(f"  Standard: EU CBAM (Regulation 2023/956)")

        # Validate for CBAM compliance
        validator = CalculationValidator()
        validation = validator.validate_result(result, standard="cbam")

        print(f"\nCBAM Compliance Check:")
        print(f"  Valid: {validation.is_valid}")

        for error in validation.errors:
            print(f"  ERROR [{error.code}]: {error.message}")

        for info in validation.info:
            print(f"  INFO [{info.code}]: {info.message}")

    except Exception as e:
        print(f"ERROR: {e}")

    finally:
        emission_db.close()


def example_unit_conversions():
    """Example: Unit conversions."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Unit Conversions")
    print("="*80)

    converter = UnitConverter()

    print("\nEnergy Conversions:")
    print(f"  1000 kWh = {converter.convert(1000, 'kWh', 'MWh', precision=3)} MWh")
    print(f"  100 MWh = {converter.convert(100, 'MWh', 'GJ', precision=2)} GJ")
    print(f"  50 GJ = {converter.convert(50, 'GJ', 'kWh', precision=2)} kWh")

    print("\nMass Conversions:")
    print(f"  2500 kg = {converter.convert(2500, 'kg', 't', precision=3)} tonnes")
    print(f"  5 tonnes = {converter.convert(5, 't', 'kg', precision=0)} kg")

    print("\nEmissions Conversions:")
    print(f"  2690 kg CO2e = {converter.convert(2690, 'kg_co2e', 't_co2e', precision=3)} t CO2e")
    print(f"  1000 t CO2e = {converter.convert(1000, 't_co2e', 'kt_co2e', precision=3)} kt CO2e")

    print("\nVolume Conversions:")
    print(f"  100 liters = {converter.convert(100, 'L', 'gal_us', precision=2)} US gallons")
    print(f"  1000 L = {converter.convert(1000, 'L', 'm3', precision=3)} m³")


def example_reproducibility_verification():
    """Example: Verify calculation reproducibility."""
    print("\n" + "="*80)
    print("EXAMPLE 6: Reproducibility Verification")
    print("="*80)

    # Setup
    formula_library = FormulaLibrary()
    emission_db = setup_sample_data()
    engine = CalculationEngine(formula_library, emission_db)

    formula_library.load_formulas()

    params = {
        "fuel_quantity": 1000,
        "fuel_type": "diesel",
        "region": "GB"
    }

    try:
        # Run calculation 3 times
        print("\nRunning calculation 3 times with identical inputs...")
        results = []
        for i in range(3):
            result = engine.calculate("scope1_stationary_combustion", params)
            results.append(result)
            print(f"  Run {i+1}: {result.output_value} {result.output_unit}, hash: {result.provenance_hash[:16]}...")

        # Verify all results are identical
        all_values = [r.output_value for r in results]
        all_hashes = [r.provenance_hash for r in results]

        if len(set(all_values)) == 1 and len(set(all_hashes)) == 1:
            print(f"\n✓ BIT-PERFECT REPRODUCIBILITY VERIFIED")
            print(f"  All 3 calculations produced identical results")
            print(f"  Value: {results[0].output_value}")
            print(f"  Hash: {results[0].provenance_hash}")
        else:
            print(f"\n✗ REPRODUCIBILITY FAILED - Results differ!")

    except Exception as e:
        print(f"ERROR: {e}")

    finally:
        emission_db.close()


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("GreenLang Zero-Hallucination Calculation Engine - Examples")
    print("="*80)

    example_scope1_calculation()
    example_scope2_calculation()
    example_scope3_calculation()
    example_cbam_calculation()
    example_unit_conversions()
    example_reproducibility_verification()

    print("\n" + "="*80)
    print("All examples completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
