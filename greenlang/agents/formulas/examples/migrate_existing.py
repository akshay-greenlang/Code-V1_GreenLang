"""
Migration Example: Import Existing Formulas

This script demonstrates how to migrate formulas from existing sources:
- CSRD esrs_formulas.yaml
- CBAM emission_factors.py
- Custom formula definitions

Run this to populate your formula database with existing formulas.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from greenlang.formulas import FormulaManager
from greenlang.formulas.migration import FormulaMigrator


def migrate_csrd_formulas():
    """Migrate CSRD ESRS formulas from YAML."""
    print("\n" + "=" * 80)
    print("MIGRATING CSRD ESRS FORMULAS")
    print("=" * 80)

    # Initialize manager
    manager = FormulaManager("production_formulas.db")
    migrator = FormulaMigrator(manager)

    # Path to CSRD formulas YAML
    yaml_path = Path(__file__).parent.parent.parent.parent.parent / \
                "GL-CSRD-APP" / "CSRD-Reporting-Platform" / "data" / "esrs_formulas.yaml"

    if not yaml_path.exists():
        print(f"⚠️  CSRD formulas file not found at {yaml_path}")
        print("   Skipping CSRD migration")
        return

    print(f"Migrating from: {yaml_path}")

    # Migrate
    stats = migrator.migrate_from_yaml(
        yaml_path=str(yaml_path),
        created_by="migration_script",
        auto_activate=True
    )

    # Display results
    print(f"\nMigration Results:")
    print(f"  Total Formulas: {stats['total']}")
    print(f"  ✓ Success:      {stats['success']}")
    print(f"  ✗ Failed:       {stats['failed']}")
    print(f"  ⊘ Skipped:      {stats['skipped']}")

    summary = migrator.get_migration_summary()
    print(f"  Success Rate:   {summary['success_rate']:.1f}%")

    manager.close()


def migrate_cbam_emission_factors():
    """Migrate CBAM emission factors from Python module."""
    print("\n" + "=" * 80)
    print("MIGRATING CBAM EMISSION FACTORS")
    print("=" * 80)

    # Initialize manager
    manager = FormulaManager("production_formulas.db")
    migrator = FormulaMigrator(manager)

    # Path to CBAM emission factors
    python_path = Path(__file__).parent.parent.parent.parent.parent / \
                  "GL-CBAM-APP" / "CBAM-Importer-Copilot" / "data" / "emission_factors.py"

    if not python_path.exists():
        print(f"⚠️  CBAM emission factors file not found at {python_path}")
        print("   Skipping CBAM migration")
        return

    print(f"Migrating from: {python_path}")

    # Migrate
    stats = migrator.migrate_from_python(
        python_path=str(python_path),
        created_by="migration_script",
        auto_activate=True
    )

    # Display results
    print(f"\nMigration Results:")
    print(f"  Total Emission Factors: {stats['total']}")
    print(f"  ✓ Success:              {stats['success']}")
    print(f"  ✗ Failed:               {stats['failed']}")
    print(f"  ⊘ Skipped:              {stats['skipped']}")

    summary = migrator.get_migration_summary()
    print(f"  Success Rate:           {summary['success_rate']:.1f}%")

    manager.close()


def migrate_custom_formulas():
    """Migrate custom formula definitions."""
    print("\n" + "=" * 80)
    print("MIGRATING CUSTOM FORMULAS")
    print("=" * 80)

    # Initialize manager
    manager = FormulaManager("production_formulas.db")
    migrator = FormulaMigrator(manager)

    # Define custom formulas for GL-001 through GL-010
    custom_formulas = [
        # GL-001: ThermoSync - Boiler efficiency
        {
            'formula_code': 'GL001_BOILER_EFF',
            'formula_name': 'Boiler Thermal Efficiency',
            'category': 'efficiency',
            'description': 'Calculate boiler thermal efficiency per ASME PTC 4.1',
            'formula_expression': '(energy_output / energy_input) * 100',
            'calculation_type': 'percentage',
            'required_inputs': ['energy_output', 'energy_input'],
            'output_unit': '%',
            'standard_reference': 'ASME PTC 4.1',
            'deterministic': True,
            'zero_hallucination': True,
        },
        {
            'formula_code': 'GL001_FUEL_SAVINGS',
            'formula_name': 'Annual Fuel Savings',
            'category': 'cost',
            'description': 'Calculate annual fuel savings from efficiency improvement',
            'formula_expression': 'baseline_fuel - optimized_fuel',
            'calculation_type': 'subtraction',
            'required_inputs': ['baseline_fuel', 'optimized_fuel'],
            'output_unit': 'therms',
            'deterministic': True,
            'zero_hallucination': True,
        },

        # GL-002: CarbonTrack - Emissions calculations
        {
            'formula_code': 'GL002_SCOPE1_TOTAL',
            'formula_name': 'Total Scope 1 Emissions',
            'category': 'emissions',
            'description': 'Sum of all Scope 1 emission sources',
            'formula_expression': 'stationary + mobile + process + fugitive',
            'calculation_type': 'sum',
            'required_inputs': ['stationary', 'mobile', 'process', 'fugitive'],
            'output_unit': 'tCO2e',
            'standard_reference': 'GHG Protocol',
            'deterministic': True,
            'zero_hallucination': True,
        },
        {
            'formula_code': 'GL002_EMISSION_INTENSITY',
            'formula_name': 'Emission Intensity',
            'category': 'emissions',
            'description': 'Emissions per unit of revenue',
            'formula_expression': 'total_emissions / revenue',
            'calculation_type': 'division',
            'required_inputs': ['total_emissions', 'revenue'],
            'output_unit': 'tCO2e/EUR',
            'deterministic': True,
            'zero_hallucination': True,
        },

        # GL-003: WaterWatch - Water consumption
        {
            'formula_code': 'GL003_WATER_CONSUMPTION',
            'formula_name': 'Total Water Consumption',
            'category': 'water',
            'description': 'Water withdrawal minus discharge',
            'formula_expression': 'withdrawal - discharge',
            'calculation_type': 'subtraction',
            'required_inputs': ['withdrawal', 'discharge'],
            'output_unit': 'm3',
            'standard_reference': 'ESRS E3',
            'deterministic': True,
            'zero_hallucination': True,
        },
        {
            'formula_code': 'GL003_RECYCLING_RATE',
            'formula_name': 'Water Recycling Rate',
            'category': 'water',
            'description': 'Percentage of water recycled',
            'formula_expression': '(recycled / withdrawal) * 100',
            'calculation_type': 'percentage',
            'required_inputs': ['recycled', 'withdrawal'],
            'output_unit': '%',
            'deterministic': True,
            'zero_hallucination': True,
        },

        # GL-004: CircularFlow - Waste management
        {
            'formula_code': 'GL004_TOTAL_WASTE',
            'formula_name': 'Total Waste Generated',
            'category': 'waste',
            'description': 'Sum of hazardous and non-hazardous waste',
            'formula_expression': 'hazardous + non_hazardous',
            'calculation_type': 'sum',
            'required_inputs': ['hazardous', 'non_hazardous'],
            'output_unit': 'tonnes',
            'standard_reference': 'ESRS E5',
            'deterministic': True,
            'zero_hallucination': True,
        },
        {
            'formula_code': 'GL004_RECYCLING_RATE',
            'formula_name': 'Waste Recycling Rate',
            'category': 'waste',
            'description': 'Percentage of waste diverted from disposal',
            'formula_expression': '(diverted / total) * 100',
            'calculation_type': 'percentage',
            'required_inputs': ['diverted', 'total'],
            'output_unit': '%',
            'deterministic': True,
            'zero_hallucination': True,
        },

        # GL-005: EnergyOptimizer - Energy calculations
        {
            'formula_code': 'GL005_RENEWABLE_PCT',
            'formula_name': 'Renewable Energy Percentage',
            'category': 'energy',
            'description': 'Percentage of renewable energy in total',
            'formula_expression': '(renewable / total) * 100',
            'calculation_type': 'percentage',
            'required_inputs': ['renewable', 'total'],
            'output_unit': '%',
            'standard_reference': 'ESRS E1',
            'deterministic': True,
            'zero_hallucination': True,
        },
        {
            'formula_code': 'GL005_ENERGY_INTENSITY',
            'formula_name': 'Energy Intensity',
            'category': 'energy',
            'description': 'Energy consumption per unit of production',
            'formula_expression': 'energy / production',
            'calculation_type': 'division',
            'required_inputs': ['energy', 'production'],
            'output_unit': 'MWh/unit',
            'deterministic': True,
            'zero_hallucination': True,
        },

        # Additional utility formulas
        {
            'formula_code': 'UTIL_KG_TO_TONNES',
            'formula_name': 'Convert Kilograms to Tonnes',
            'category': 'utility',
            'description': 'Unit conversion: kg to tonnes',
            'formula_expression': 'kg / 1000',
            'calculation_type': 'division',
            'required_inputs': ['kg'],
            'output_unit': 'tonnes',
            'deterministic': True,
            'zero_hallucination': True,
        },
        {
            'formula_code': 'UTIL_KWH_TO_MWH',
            'formula_name': 'Convert kWh to MWh',
            'category': 'utility',
            'description': 'Unit conversion: kWh to MWh',
            'formula_expression': 'kwh / 1000',
            'calculation_type': 'division',
            'required_inputs': ['kwh'],
            'output_unit': 'MWh',
            'deterministic': True,
            'zero_hallucination': True,
        },
    ]

    print(f"Migrating {len(custom_formulas)} custom formulas...")

    # Migrate
    stats = migrator.migrate_custom_formulas(
        formulas=custom_formulas,
        created_by="migration_script",
        auto_activate=True
    )

    # Display results
    print(f"\nMigration Results:")
    print(f"  Total Custom Formulas: {stats['total']}")
    print(f"  ✓ Success:             {stats['success']}")
    print(f"  ✗ Failed:              {stats['failed']}")
    print(f"  ⊘ Skipped:             {stats['skipped']}")

    summary = migrator.get_migration_summary()
    print(f"  Success Rate:          {summary['success_rate']:.1f}%")

    manager.close()


def verify_migration():
    """Verify all formulas were migrated successfully."""
    print("\n" + "=" * 80)
    print("VERIFYING MIGRATION")
    print("=" * 80)

    manager = FormulaManager("production_formulas.db")

    # List all formulas
    formulas = manager.list_formulas()

    print(f"\nTotal formulas in database: {len(formulas)}")

    # Group by category
    by_category = {}
    for formula in formulas:
        category = formula.category
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(formula)

    print(f"\nFormulas by category:")
    for category, formulas_list in sorted(by_category.items()):
        print(f"  {category:<15} {len(formulas_list):>4} formulas")

    # Show sample formulas
    print(f"\nSample formulas:")
    for formula in formulas[:10]:
        active = manager.get_active_formula(formula.formula_code)
        status = "✓ ACTIVE" if active else "  INACTIVE"
        print(f"  {status} {formula.formula_code:<20} {formula.formula_name}")

    manager.close()


def main():
    """Run all migrations."""
    print("\n" + "=" * 80)
    print("GreenLang Formula Migration - Populate Database")
    print("=" * 80)

    try:
        # Run migrations
        migrate_csrd_formulas()
        migrate_cbam_emission_factors()
        migrate_custom_formulas()

        # Verify
        verify_migration()

        print("\n" + "=" * 80)
        print("Migration completed successfully!")
        print("=" * 80)
        print(f"\nProduction database created at: production_formulas.db")
        print(f"\nYou can now use this database in your applications:")
        print(f"  manager = FormulaManager('production_formulas.db')")
        print(f"\nOr inspect it via CLI:")
        print(f"  greenlang formula list --db production_formulas.db")
        print()

    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
