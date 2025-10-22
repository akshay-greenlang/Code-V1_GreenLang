#!/usr/bin/env python3
"""
Example 5: Complete Provenance and Audit Trail
===============================================

This example demonstrates:
- Full provenance tracking for calculations
- Audit trail creation and verification
- Input/output hashing for reproducibility
- Run ledger management

Run: python examples/05_provenance_tracking.py
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
from greenlang.sdk.base import Agent, Result, Metadata
from greenlang.provenance.ledger import RunLedger, stable_hash


class ProvenanceTrackedCalculator(Agent[Dict[str, Any], Dict[str, Any]]):
    """
    Calculator with complete provenance tracking.

    Tracks all inputs, outputs, and execution metadata for audit trail.
    """

    def __init__(self):
        metadata = Metadata(
            id="provenance_calculator",
            name="Provenance Tracked Calculator",
            version="1.0.0",
            description="Calculator with full audit trail",
            author="GreenLang Examples"
        )
        super().__init__(metadata)

        # Initialize run ledger
        ledger_dir = Path(__file__).parent / "out" / "ledger"
        ledger_dir.mkdir(parents=True, exist_ok=True)
        self.ledger = RunLedger(ledger_dir / "calculations.jsonl")

        # Load emission factors
        data_dir = Path(__file__).parent / "data"
        with open(data_dir / "emission_factors.json") as f:
            self.factors = json.load(f)["factors"]

    def validate(self, input_data: Dict[str, Any]) -> bool:
        """Validate input data"""
        return "energy_data" in input_data

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process with full provenance tracking"""
        # Calculate input hash for provenance
        input_hash = stable_hash(input_data)

        # Start execution timestamp
        start_time = datetime.utcnow()

        # Perform calculation
        energy_data = input_data["energy_data"]
        country = input_data.get("country", "US")

        results = []
        total_emissions = 0.0

        for entry in energy_data:
            fuel_type = entry["fuel_type"]
            consumption = entry["consumption"]

            factor = self.factors[fuel_type][country]["value"]
            emissions_tons = (consumption * factor) / 1000

            total_emissions += emissions_tons

            results.append({
                "fuel_type": fuel_type,
                "consumption": consumption,
                "factor": factor,
                "factor_source": self.factors[fuel_type][country]["source"],
                "emissions_tons": round(emissions_tons, 4)
            })

        # Calculate output hash
        output_data = {
            "total_emissions_tons": round(total_emissions, 4),
            "breakdown": results,
            "country": country
        }
        output_hash = stable_hash(output_data)

        # Record execution metadata
        execution_metadata = {
            "input_hash": input_hash,
            "output_hash": output_hash,
            "start_time": start_time.isoformat(),
            "end_time": datetime.utcnow().isoformat(),
            "duration_ms": (datetime.utcnow() - start_time).total_seconds() * 1000,
            "agent_id": self.metadata.id,
            "agent_version": self.metadata.version,
            "factors_used": {
                fuel: self.factors[fuel][country]["source"]
                for fuel in set(e["fuel_type"] for e in energy_data)
            }
        }

        # Record in ledger
        run_id = self.ledger.record_run(
            pipeline="provenance_calculator",
            inputs=input_data,
            outputs=output_data,
            metadata=execution_metadata
        )

        # Add provenance to output
        output_data["provenance"] = {
            "run_id": run_id,
            "input_hash": input_hash,
            "output_hash": output_hash,
            "execution": execution_metadata
        }

        return output_data


def main():
    """Run the example"""
    print("\n" + "="*70)
    print("Example 5: Complete Provenance and Audit Trail")
    print("="*70 + "\n")

    calculator = ProvenanceTrackedCalculator()

    # Test 1: Single calculation with provenance
    print("Test 1: Calculation with Provenance Tracking")
    print("-" * 70)

    input_data = {
        "energy_data": [
            {"fuel_type": "electricity", "consumption": 50000},
            {"fuel_type": "natural_gas", "consumption": 1000}
        ],
        "country": "US"
    }

    result = calculator.run(input_data)

    if result.success:
        prov = result.data["provenance"]
        print(f"Run ID: {prov['run_id']}")
        print(f"Input Hash: {prov['input_hash']}")
        print(f"Output Hash: {prov['output_hash']}")
        print(f"Total Emissions: {result.data['total_emissions_tons']:.4f} tCO2e")
        print(f"Duration: {prov['execution']['duration_ms']:.2f} ms")

    # Test 2: Reproduce same calculation
    print("\n\nTest 2: Reproducibility Check")
    print("-" * 70)

    result2 = calculator.run(input_data)

    if result2.success:
        print(f"First run hash: {result.data['provenance']['output_hash']}")
        print(f"Second run hash: {result2.data['provenance']['output_hash']}")
        print(f"Hashes match: {result.data['provenance']['output_hash'] == result2.data['provenance']['output_hash']}")

    # Test 3: Multiple calculations
    print("\n\nTest 3: Multiple Calculations")
    print("-" * 70)

    test_cases = [
        {"energy_data": [{"fuel_type": "electricity", "consumption": 1000}], "country": "US"},
        {"energy_data": [{"fuel_type": "natural_gas", "consumption": 500}], "country": "UK"},
        {"energy_data": [{"fuel_type": "electricity", "consumption": 2000}], "country": "CA"}
    ]

    for i, test_input in enumerate(test_cases, 1):
        result = calculator.run(test_input)
        if result.success:
            print(f"{i}. {test_input['country']}: {result.data['total_emissions_tons']:.4f} tCO2e "
                  f"(Run ID: {result.data['provenance']['run_id'][:8]}...)")

    # Test 4: Ledger query
    print("\n\nTest 4: Ledger Query and Statistics")
    print("-" * 70)

    # List recent runs
    recent_runs = calculator.ledger.list_runs(limit=5)
    print(f"Recent runs: {len(recent_runs)}")

    for run in recent_runs[:3]:
        print(f"  - {run['id'][:8]}... | {run['timestamp']} | {run['pipeline']}")

    # Get statistics
    stats = calculator.ledger.get_statistics(pipeline="provenance_calculator", days=1)
    print(f"\nLedger Statistics:")
    print(f"  Total runs: {stats['total_runs']}")
    print(f"  Unique inputs: {stats['unique_inputs']}")
    print(f"  Unique outputs: {stats['unique_outputs']}")

    # Test 5: Export ledger
    print("\n\nTest 5: Export Ledger")
    print("-" * 70)

    export_path = Path(__file__).parent / "out" / "ledger_export.json"
    calculator.ledger.export_to_json(export_path, pipeline="provenance_calculator")
    print(f"Ledger exported to: {export_path}")

    # Test 6: Verify reproducibility from ledger
    print("\n\nTest 6: Reproducibility Verification")
    print("-" * 70)

    if recent_runs:
        first_run = recent_runs[0]
        is_reproducible = calculator.ledger.verify_reproducibility(
            first_run["input_hash"],
            first_run["output_hash"],
            "provenance_calculator"
        )
        print(f"Calculation is reproducible: {is_reproducible}")

    print("\n" + "="*70)
    print("Example completed successfully!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
