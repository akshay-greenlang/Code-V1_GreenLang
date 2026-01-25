"""
Basic Usage Examples for GreenLang Formula Versioning System

This script demonstrates the core functionality of the formula versioning system.
Run this to understand how to use the system in your applications.
"""

import sys
from pathlib import Path
from datetime import date

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from greenlang.formulas import FormulaManager
from greenlang.formulas.models import FormulaCategory
from greenlang.formulas.migration import FormulaMigrator


def example_1_create_formula():
    """Example 1: Create a new formula from scratch."""
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Creating a Formula")
    print("=" * 80)

    # Initialize manager with temporary database
    manager = FormulaManager("examples_formulas.db")

    # Create formula metadata
    formula_id = manager.create_formula(
        formula_code="E1-1",
        formula_name="Total Scope 1 GHG Emissions",
        category=FormulaCategory.EMISSIONS,
        description="Sum of all Scope 1 emission sources per ESRS E1",
        standard_reference="ESRS E1-1",
        created_by="demo_user"
    )

    print(f"✓ Created formula E1-1 (id={formula_id})")

    # Create initial version
    version_data = {
        'formula_expression': 'stationary + mobile + process + fugitive',
        'calculation_type': 'sum',
        'required_inputs': ['stationary', 'mobile', 'process', 'fugitive'],
        'output_unit': 'tCO2e',
        'deterministic': True,
        'zero_hallucination': True,
    }

    version_id = manager.create_new_version(
        formula_code="E1-1",
        formula_data=version_data,
        change_notes="Initial version based on ESRS E1 guidance",
        auto_activate=True
    )

    print(f"✓ Created version 1 (id={version_id})")
    print(f"✓ Formula E1-1 is now active and ready to use")

    manager.close()


def example_2_execute_formula():
    """Example 2: Execute a formula with input data."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Executing a Formula")
    print("=" * 80)

    manager = FormulaManager("examples_formulas.db")

    # Sample emission data for a company
    emission_data = {
        'stationary': 1500.5,  # tCO2e from stationary combustion
        'mobile': 750.2,       # tCO2e from mobile combustion
        'process': 300.8,      # tCO2e from process emissions
        'fugitive': 125.3      # tCO2e from fugitive emissions
    }

    print(f"Input data:")
    for source, value in emission_data.items():
        print(f"  {source}: {value} tCO2e")

    # Execute formula
    result = manager.execute_formula(
        formula_code="E1-1",
        input_data=emission_data,
        agent_name="EmissionsCalculatorAgent"
    )

    print(f"\n✓ Total Scope 1 Emissions: {result} tCO2e")

    manager.close()


def example_3_version_management():
    """Example 3: Create new version and manage versions."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Version Management")
    print("=" * 80)

    manager = FormulaManager("examples_formulas.db")

    # Create updated version with additional emission source
    print("Creating version 2 with biogenic emissions...")

    version_data = {
        'formula_expression': 'stationary + mobile + process + fugitive + biogenic',
        'calculation_type': 'sum',
        'required_inputs': ['stationary', 'mobile', 'process', 'fugitive', 'biogenic'],
        'output_unit': 'tCO2e',
        'deterministic': True,
        'zero_hallucination': True,
    }

    v2_id = manager.create_new_version(
        formula_code="E1-1",
        formula_data=version_data,
        change_notes="Added biogenic emissions per updated ESRS guidance 2025",
        auto_activate=False  # Don't activate yet
    )

    print(f"✓ Created version 2 (id={v2_id})")

    # Test new version before activating
    test_data = {
        'stationary': 1500.5,
        'mobile': 750.2,
        'process': 300.8,
        'fugitive': 125.3,
        'biogenic': 50.0  # New input
    }

    result = manager.execute_formula(
        formula_code="E1-1",
        input_data=test_data,
        version_number=2  # Test version 2 specifically
    )

    print(f"✓ Tested version 2: {result} tCO2e")

    # Activate version 2
    manager.activate_version("E1-1", version_number=2)
    print(f"✓ Activated version 2")

    # List all versions
    versions = manager.list_versions("E1-1")
    print(f"\nAll versions of E1-1:")
    for v in versions:
        status = "✓ ACTIVE" if v.version_status == "active" else f"  {v.version_status}"
        print(f"  v{v.version_number}: {status} - {v.change_notes}")

    manager.close()


def example_4_rollback():
    """Example 4: Rollback to previous version."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Rolling Back to Previous Version")
    print("=" * 80)

    manager = FormulaManager("examples_formulas.db")

    print("Rolling back E1-1 to version 1...")

    # Rollback to version 1
    new_version_id = manager.rollback_to_version(
        formula_code="E1-1",
        version_number=1
    )

    print(f"✓ Rolled back to version 1")
    print(f"✓ Created version 3 as copy of version 1 (id={new_version_id})")

    # Verify rollback
    active = manager.get_active_formula("E1-1")
    print(f"\nActive version details:")
    print(f"  Version: {active.version_number}")
    print(f"  Expression: {active.formula_expression}")
    print(f"  Inputs: {', '.join(active.required_inputs)}")

    manager.close()


def example_5_full_execution_result():
    """Example 5: Get full execution result with provenance."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Full Execution Result with Provenance")
    print("=" * 80)

    manager = FormulaManager("examples_formulas.db")

    emission_data = {
        'stationary': 1500.5,
        'mobile': 750.2,
        'process': 300.8,
        'fugitive': 125.3
    }

    # Execute with full result
    result = manager.execute_formula_full(
        formula_code="E1-1",
        input_data=emission_data,
        agent_name="EmissionsCalculatorAgent",
        calculation_id="CALC-2025-001",
        user_id="john.doe@company.com"
    )

    print(f"Execution Result:")
    print(f"  Output: {result.output_value} tCO2e")
    print(f"  Status: {result.execution_status}")
    print(f"  Execution Time: {result.execution_time_ms:.2f} ms")
    print(f"\nProvenance:")
    print(f"  Input Hash:  {result.input_hash}")
    print(f"  Output Hash: {result.output_hash}")
    print(f"\nAudit Trail:")
    print(f"  Agent: {result.agent_name}")
    print(f"  Calculation ID: {result.calculation_id}")
    print(f"  User: {result.user_id}")
    print(f"  Timestamp: {result.execution_timestamp}")

    manager.close()


def example_6_compare_versions():
    """Example 6: Compare two formula versions."""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Comparing Formula Versions")
    print("=" * 80)

    manager = FormulaManager("examples_formulas.db")

    # Compare version 1 and 2
    comparison = manager.compare_versions(
        formula_code="E1-1",
        version_a=1,
        version_b=2
    )

    print(f"Comparison: E1-1 v1 vs v2")
    print(f"\nExpression Changed: {comparison.expression_changed}")
    if comparison.expression_changed:
        print(f"{comparison.expression_diff}")

    print(f"\nInputs Changed: {comparison.inputs_changed}")
    if comparison.added_inputs:
        print(f"  Added: {', '.join(comparison.added_inputs)}")
    if comparison.removed_inputs:
        print(f"  Removed: {', '.join(comparison.removed_inputs)}")

    print(f"\nOutput Unit Changed: {comparison.output_unit_changed}")

    manager.close()


def example_7_custom_expression():
    """Example 7: Create formula with custom expression."""
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Custom Expression Formula")
    print("=" * 80)

    manager = FormulaManager("examples_formulas.db")

    # Create formula for renewable energy percentage
    manager.create_formula(
        formula_code="E1-7",
        formula_name="Renewable Energy Percentage",
        category=FormulaCategory.ENERGY,
        description="Percentage of renewable energy in total energy consumption",
        standard_reference="ESRS E1",
    )

    # Create version with percentage calculation
    version_data = {
        'formula_expression': '(renewable_energy / total_energy) * 100',
        'calculation_type': 'percentage',
        'required_inputs': ['renewable_energy', 'total_energy'],
        'output_unit': '%',
        'deterministic': True,
        'zero_hallucination': True,
    }

    manager.create_new_version(
        formula_code="E1-7",
        formula_data=version_data,
        change_notes="Initial version",
        auto_activate=True
    )

    print(f"✓ Created formula E1-7: Renewable Energy Percentage")

    # Execute
    result = manager.execute_formula(
        formula_code="E1-7",
        input_data={
            'renewable_energy': 7500,  # MWh
            'total_energy': 10000      # MWh
        }
    )

    print(f"✓ Renewable Energy Percentage: {result}%")

    manager.close()


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("GreenLang Formula Versioning System - Usage Examples")
    print("=" * 80)

    try:
        # Run examples in sequence
        example_1_create_formula()
        example_2_execute_formula()
        example_3_version_management()
        example_4_rollback()
        example_5_full_execution_result()
        example_6_compare_versions()
        example_7_custom_expression()

        print("\n" + "=" * 80)
        print("All examples completed successfully!")
        print("=" * 80)
        print(f"\nDatabase created at: examples_formulas.db")
        print(f"You can inspect it using SQLite tools or the CLI:")
        print(f"  greenlang formula list --db examples_formulas.db")
        print()

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
