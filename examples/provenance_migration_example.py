# -*- coding: utf-8 -*-
"""
Example: Migrating Existing Calculators to Use CalculationProvenance

This example shows how to update existing GL-001 through GL-010 calculators
to use the standardized CalculationProvenance model.

Author: GreenLang Team
Version: 1.0.0
"""

from greenlang.agents.calculator import BaseCalculator, CalculatorConfig
from greenlang.core.provenance import CalculationProvenance, OperationType
from greenlang.core.provenance.storage import SQLiteProvenanceStorage
from decimal import Decimal


# ============================================================================
# BEFORE: Old Calculator (No Provenance)
# ============================================================================

class OldEmissionsCalculator(BaseCalculator):
    """
    Old-style calculator without standardized provenance.
    Uses basic calculation steps only.
    """

    def __init__(self):
        config = CalculatorConfig(name="EmissionsCalculator")
        super().__init__(config)

    def calculate(self, inputs):
        fuel_kg = inputs["fuel_consumption_kg"]
        fuel_type = inputs["fuel_type"]

        # Lookup emission factor
        ef = 0.18414  # Hardcoded - no provenance of where this came from

        # Record in old format (no data source tracking)
        self.add_calculation_step(
            step_name="lookup_ef",
            formula="EF lookup",
            inputs={"fuel_type": fuel_type},
            result=ef
        )

        # Calculate
        emissions = fuel_kg * ef

        self.add_calculation_step(
            step_name="calculate_emissions",
            formula="emissions = fuel * ef",
            inputs={"fuel": fuel_kg, "ef": ef},
            result=emissions
        )

        return emissions


# ============================================================================
# AFTER: Migrated Calculator (With Provenance)
# ============================================================================

class NewEmissionsCalculator(BaseCalculator):
    """
    Migrated calculator with standardized provenance tracking.

    Migration changes:
    1. Enable provenance in config
    2. Use record_provenance_step() instead of add_calculation_step()
    3. Add data sources and standard references
    4. Add operation types
    5. Store provenance records
    """

    def __init__(self):
        config = CalculatorConfig(
            name="EmissionsCalculator",
            agent_version="2.0.0",  # ← Version tracking
            enable_provenance=True,  # ← Enable provenance
        )
        super().__init__(config)

        # Database of emission factors with provenance
        self.emission_factors_db = {
            "natural_gas": {
                "value": 0.18414,
                "source": "EPA eGRID 2023",
                "reference": "EPA AP-42 Table 1.4-1"
            },
            "diesel": {
                "value": 0.26760,
                "source": "DEFRA 2024",
                "reference": "DEFRA Emission Factors 2024"
            },
        }

    def calculate(self, inputs):
        fuel_kg = inputs["fuel_consumption_kg"]
        fuel_type = inputs["fuel_type"]

        # Lookup emission factor (with full provenance)
        ef_data = self.emission_factors_db.get(fuel_type)
        if not ef_data:
            raise ValueError(f"Unknown fuel type: {fuel_type}")

        ef = ef_data["value"]

        # ← NEW: Record with provenance (data source, standard reference)
        self.record_provenance_step(
            operation=OperationType.LOOKUP,  # ← Operation type
            description=f"Lookup emission factor for {fuel_type}",  # ← Clear description
            inputs={"fuel_type": fuel_type},
            output=ef,
            data_source=ef_data["source"],  # ← Data source tracked
            standard_reference=ef_data["reference"],  # ← Standard reference tracked
        )

        # Calculate emissions
        emissions = fuel_kg * ef

        # ← NEW: Record with formula and standard
        self.record_provenance_step(
            operation=OperationType.MULTIPLY,  # ← Operation type
            description="Calculate direct CO2 emissions",
            inputs={"fuel_kg": fuel_kg, "emission_factor": ef},
            output=emissions,
            formula="emissions = fuel_kg * emission_factor",  # ← Mathematical formula
            standard_reference="GHG Protocol Scope 1 Equation 3.1",  # ← Standard
        )

        # Convert to tonnes
        emissions_tonnes = emissions / 1000

        self.record_provenance_step(
            operation=OperationType.DIVIDE,
            description="Convert kg to tonnes CO2e",
            inputs={"emissions_kg": emissions},
            output=emissions_tonnes,
            formula="emissions_tonnes = emissions_kg / 1000"
        )

        return {
            "total_emissions_tonnes_co2e": emissions_tonnes,
            "emissions_kg_co2": emissions,
            "emission_factor_used": ef,
        }


# ============================================================================
# EXAMPLE: Side-by-Side Comparison
# ============================================================================

def compare_old_vs_new():
    """Compare old and new calculator outputs."""

    print("=" * 70)
    print("COMPARING OLD VS NEW CALCULATOR")
    print("=" * 70)

    inputs = {
        "fuel_consumption_kg": 1000,
        "fuel_type": "natural_gas"
    }

    # Old calculator
    print("\n1. OLD CALCULATOR (No Provenance)")
    print("-" * 70)
    old_calc = OldEmissionsCalculator()
    old_result = old_calc.execute({"inputs": inputs})

    print(f"Result: {old_result.result_value}")
    print(f"Steps recorded: {len(old_result.calculation_steps)}")
    print(f"Provenance: {old_result.provenance}")  # None

    # New calculator
    print("\n2. NEW CALCULATOR (With Provenance)")
    print("-" * 70)
    new_calc = NewEmissionsCalculator()
    new_result = new_calc.execute({
        "inputs": inputs,
        "calculation_type": "scope1_emissions"
    })

    print(f"Result: {new_result.result_value}")
    print(f"Steps recorded: {len(new_result.calculation_steps)}")
    print(f"Provenance available: {new_result.provenance is not None}")

    if new_result.provenance:
        print(f"\nProvenance Details:")
        print(f"  Calculation ID: {new_result.provenance['calculation_id']}")
        print(f"  Agent: {new_result.provenance['metadata']['agent_name']} v{new_result.provenance['metadata']['agent_version']}")
        print(f"  Steps: {len(new_result.provenance['steps'])}")
        print(f"  Duration: {new_result.provenance['duration_ms']:.2f}ms")
        print(f"  Standards: {new_result.provenance['metadata']['standards_applied']}")
        print(f"  Data Sources: {new_result.provenance['metadata']['data_sources']}")

        # Show steps
        print(f"\n  Calculation Steps:")
        for step in new_result.provenance['steps']:
            print(f"    {step['step_number']}. [{step['operation']}] {step['description']}")
            if step.get('data_source'):
                print(f"       Data Source: {step['data_source']}")
            if step.get('standard_reference'):
                print(f"       Standard: {step['standard_reference']}")


# ============================================================================
# EXAMPLE: Storing Provenance
# ============================================================================

def store_provenance_example():
    """Example of storing provenance records."""

    print("\n" + "=" * 70)
    print("STORING PROVENANCE RECORDS")
    print("=" * 70)

    # Create calculator
    calculator = NewEmissionsCalculator()

    # Create storage
    storage = SQLiteProvenanceStorage("emissions_provenance.db")

    # Perform multiple calculations
    test_cases = [
        {"fuel_consumption_kg": 1000, "fuel_type": "natural_gas"},
        {"fuel_consumption_kg": 2500, "fuel_type": "diesel"},
        {"fuel_consumption_kg": 500, "fuel_type": "natural_gas"},
    ]

    calc_ids = []

    for i, inputs in enumerate(test_cases, 1):
        print(f"\n{i}. Calculating for {inputs}")

        result = calculator.execute({
            "inputs": inputs,
            "calculation_type": "scope1_emissions"
        })

        if result.provenance:
            # Recreate provenance object from dict
            prov = CalculationProvenance.from_dict(result.provenance)

            # Store in database
            calc_id = storage.store(prov)
            calc_ids.append(calc_id)

            print(f"   Result: {result.result_value}")
            print(f"   Stored: {calc_id}")

    # Query stored records
    print(f"\n" + "-" * 70)
    print("QUERYING STORED RECORDS")
    print("-" * 70)

    # Get all emissions calculations
    all_records = storage.query(
        calculation_type="scope1_emissions",
        limit=100
    )
    print(f"\nTotal scope1_emissions calculations: {len(all_records)}")

    # Get statistics
    stats = storage.get_statistics()
    print(f"\nStorage Statistics:")
    print(f"  Total calculations: {stats['total_calculations']}")
    print(f"  Calculation types: {stats['calculation_types']}")
    print(f"  Average duration: {stats['average_duration_ms']:.2f}ms")

    # Find duplicates (same inputs)
    if len(all_records) >= 2:
        first_record = all_records[0]
        duplicates = storage.find_by_input_hash(first_record.input_hash)
        print(f"\nCalculations with same inputs as first record: {len(duplicates)}")

    # Find by data source
    epa_calcs = storage.find_by_data_source("EPA eGRID 2023")
    print(f"Calculations using EPA eGRID 2023: {len(epa_calcs)}")

    return calc_ids, storage


# ============================================================================
# EXAMPLE: Audit Trail Verification
# ============================================================================

def verify_audit_trail_example(calc_ids, storage):
    """Example of verifying audit trail integrity."""

    print("\n" + "=" * 70)
    print("VERIFYING AUDIT TRAIL INTEGRITY")
    print("=" * 70)

    for calc_id in calc_ids:
        # Retrieve record
        record = storage.retrieve(calc_id)

        if record:
            # Verify integrity
            integrity = record.verify_integrity()

            print(f"\nCalculation: {calc_id}")
            print(f"  Input hash valid: {integrity['input_hash_valid']}")
            print(f"  Output hash valid: {integrity['output_hash_valid']}")
            print(f"  Steps sequential: {integrity['steps_sequential']}")
            print(f"  Has steps: {integrity['has_steps']}")
            print(f"  Is finalized: {integrity['is_finalized']}")
            print(f"  No errors: {integrity['no_errors']}")

            if all(integrity.values()):
                print(f"  ✓ INTEGRITY VERIFIED")
            else:
                print(f"  ✗ INTEGRITY CHECK FAILED")

            # Show audit summary
            summary = record.get_audit_summary()
            print(f"\n  {summary['summary']}")


# ============================================================================
# EXAMPLE: Exporting Audit Reports
# ============================================================================

def export_audit_report_example(storage):
    """Example of exporting audit reports."""

    print("\n" + "=" * 70)
    print("EXPORTING AUDIT REPORT")
    print("=" * 70)

    # Export complete audit report
    report_path = storage.export_audit_report("emissions_audit_report.json")

    print(f"\nExported audit report to: {report_path}")

    # Read and display summary
    import json
    with open(report_path) as f:
        report = json.load(f)

    print(f"\nReport Summary:")
    print(f"  Generated at: {report['generated_at']}")
    print(f"  Total records: {len(report['records'])}")
    print(f"  Statistics: {report['statistics']}")


# ============================================================================
# MAIN EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("\n")
    print("=" * 70)
    print("GREENLANG CALCULATION PROVENANCE MIGRATION EXAMPLE")
    print("=" * 70)

    # 1. Compare old vs new
    compare_old_vs_new()

    # 2. Store provenance
    calc_ids, storage = store_provenance_example()

    # 3. Verify audit trail
    verify_audit_trail_example(calc_ids, storage)

    # 4. Export audit report
    export_audit_report_example(storage)

    print("\n" + "=" * 70)
    print("EXAMPLE COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("1. Review the generated provenance.db database")
    print("2. Check emissions_audit_report.json for full audit trail")
    print("3. Migrate your calculators using this pattern")
    print()
