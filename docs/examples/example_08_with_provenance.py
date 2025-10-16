"""
Example 08: Agent with Provenance Tracking

This example demonstrates provenance tracking for audit trails.
You'll learn:
- How to use lifecycle hooks for provenance
- How to track inputs and outputs
- How to maintain audit trails
- How to export provenance data
"""

from greenlang.agents import BaseCalculator, CalculatorConfig, AgentResult
from typing import Dict, Any
from datetime import datetime
import json
from pathlib import Path


class ProvenanceTrackedCalculator(BaseCalculator):
    """
    Calculator with comprehensive provenance tracking.

    This agent demonstrates:
    - Automatic input/output tracking
    - Audit trail generation
    - Provenance export
    - Regulatory compliance features
    """

    def __init__(self):
        config = CalculatorConfig(
            name="ProvenanceTrackedCalculator",
            description="Calculate with full provenance tracking",
            precision=4,
            enable_provenance=True
        )
        super().__init__(config)

        # Initialize provenance store
        self.provenance_records = []

        # Add hooks for provenance tracking
        self.add_pre_hook(self._record_pre_execution)
        self.add_post_hook(self._record_post_execution)

    def _record_pre_execution(self, agent, input_data):
        """Hook to record before execution."""
        record = {
            'record_id': f"prov_{len(self.provenance_records) + 1:04d}",
            'agent_name': self.config.name,
            'timestamp_start': datetime.now().isoformat(),
            'input_data': self._sanitize_data(input_data),
            'input_hash': self._hash_data(input_data)
        }
        self.provenance_records.append(record)

    def _record_post_execution(self, agent, result: AgentResult):
        """Hook to record after execution."""
        if self.provenance_records:
            record = self.provenance_records[-1]
            record['timestamp_end'] = datetime.now().isoformat()
            record['success'] = result.success
            record['output_data'] = self._sanitize_data(result.data)
            record['output_hash'] = self._hash_data(result.data)

            if result.metrics:
                record['metrics'] = {
                    'execution_time_ms': result.metrics.execution_time_ms,
                    'cache_hit': getattr(result, 'cached', False)
                }

            if hasattr(result, 'calculation_steps'):
                record['calculation_steps'] = [
                    {
                        'step_name': step.step_name,
                        'formula': step.formula,
                        'result': str(step.result),
                        'units': step.units
                    }
                    for step in result.calculation_steps
                ]

            if not result.success:
                record['error'] = result.error

    @staticmethod
    def _sanitize_data(data):
        """Sanitize data for provenance (remove sensitive info)."""
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if not k.startswith('_')}
        return data

    @staticmethod
    def _hash_data(data):
        """Generate hash of data for integrity verification."""
        import hashlib
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]

    def calculate(self, inputs: Dict[str, Any]) -> float:
        """
        Calculate carbon intensity.

        Args:
            inputs: Must contain 'emissions_kg' and 'output_units'

        Returns:
            Carbon intensity value
        """
        emissions_kg = inputs['emissions_kg']
        output_units = inputs['output_units']

        # Calculate intensity
        intensity = emissions_kg / output_units

        self.add_calculation_step(
            step_name="Calculate Intensity",
            formula="emissions_kg ÷ output_units",
            inputs={'emissions_kg': emissions_kg, 'output_units': output_units},
            result=intensity,
            units="kg CO2 per unit"
        )

        return intensity

    def validate_calculation_inputs(self, inputs: Dict[str, Any]) -> bool:
        """Validate inputs."""
        required = ['emissions_kg', 'output_units']
        if not all(k in inputs for k in required):
            return False

        if inputs['output_units'] <= 0:
            return False

        return True

    def export_provenance(self, output_path: str):
        """
        Export provenance records to file.

        Args:
            output_path: Path to save provenance data
        """
        provenance_data = {
            'agent_name': self.config.name,
            'agent_version': self.config.version,
            'export_timestamp': datetime.now().isoformat(),
            'total_records': len(self.provenance_records),
            'records': self.provenance_records
        }

        with open(output_path, 'w') as f:
            json.dump(provenance_data, f, indent=2)

        self.logger.info(f"Provenance exported to {output_path}")

    def get_provenance_summary(self) -> Dict[str, Any]:
        """Get provenance summary statistics."""
        if not self.provenance_records:
            return {'total_records': 0}

        successful = sum(1 for r in self.provenance_records if r.get('success', False))

        return {
            'total_records': len(self.provenance_records),
            'successful_executions': successful,
            'failed_executions': len(self.provenance_records) - successful,
            'first_execution': self.provenance_records[0]['timestamp_start'],
            'last_execution': self.provenance_records[-1]['timestamp_start']
        }


def main():
    """Run the example."""
    print("=" * 60)
    print("Example 08: Agent with Provenance Tracking")
    print("=" * 60)
    print()

    calculator = ProvenanceTrackedCalculator()

    # Example 1: Single calculation with provenance
    print("Test 1: Calculate with Provenance Tracking")
    print("-" * 40)

    result = calculator.run({
        "inputs": {
            "emissions_kg": 5000,
            "output_units": 100
        }
    })

    if result.success:
        print(f"✓ Calculation successful")
        print(f"  Intensity: {result.result_value:.2f} kg CO2 per unit")
        print(f"  Execution time: {result.metrics.execution_time_ms:.2f}ms")

        # Show provenance record
        print(f"\n  Provenance Record:")
        prov_record = calculator.provenance_records[-1]
        print(f"    ID: {prov_record['record_id']}")
        print(f"    Input hash: {prov_record['input_hash']}")
        print(f"    Output hash: {prov_record['output_hash']}")
        print(f"    Timestamp: {prov_record['timestamp_start']}")
    else:
        print(f"✗ Calculation failed: {result.error}")
    print()

    # Example 2: Multiple calculations
    print("Test 2: Multiple Calculations")
    print("-" * 40)

    test_cases = [
        {'emissions_kg': 1000, 'output_units': 50},
        {'emissions_kg': 2500, 'output_units': 75},
        {'emissions_kg': 3000, 'output_units': 120},
    ]

    for i, inputs in enumerate(test_cases, 1):
        result = calculator.run({"inputs": inputs})
        if result.success:
            print(f"  Case {i}: Intensity = {result.result_value:.2f} kg CO2 per unit")

    print(f"\n  Total provenance records: {len(calculator.provenance_records)}")
    print()

    # Example 3: Failed calculation (provenance still recorded)
    print("Test 3: Failed Calculation (Provenance Still Tracked)")
    print("-" * 40)

    result_fail = calculator.run({
        "inputs": {
            "emissions_kg": 1000,
            "output_units": 0  # Invalid: division by zero
        }
    })

    if not result_fail.success:
        print(f"✓ Calculation correctly failed")
        print(f"  Error: {result_fail.error}")

        # Show provenance still recorded failure
        prov_record = calculator.provenance_records[-1]
        print(f"\n  Provenance Record (Failure):")
        print(f"    ID: {prov_record['record_id']}")
        print(f"    Success: {prov_record.get('success', 'N/A')}")
        print(f"    Error: {prov_record.get('error', 'N/A')}")
    print()

    # Example 4: Provenance summary
    print("Test 4: Provenance Summary")
    print("-" * 40)

    summary = calculator.get_provenance_summary()
    print(f"  Total records: {summary['total_records']}")
    print(f"  Successful: {summary['successful_executions']}")
    print(f"  Failed: {summary['failed_executions']}")
    print(f"  First execution: {summary['first_execution']}")
    print(f"  Last execution: {summary['last_execution']}")
    print()

    # Example 5: Export provenance
    print("Test 5: Export Provenance Data")
    print("-" * 40)

    # Create output directory
    output_dir = Path("provenance_output")
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "calculator_provenance.json"
    calculator.export_provenance(str(output_path))

    print(f"  ✓ Provenance exported to: {output_path}")
    print(f"  File size: {output_path.stat().st_size} bytes")

    # Show sample of exported data
    with open(output_path, 'r') as f:
        exported_data = json.load(f)

    print(f"\n  Exported Data Summary:")
    print(f"    Agent: {exported_data['agent_name']}")
    print(f"    Version: {exported_data['agent_version']}")
    print(f"    Total records: {exported_data['total_records']}")
    print(f"    Export timestamp: {exported_data['export_timestamp']}")

    # Show a sample record
    if exported_data['records']:
        sample_record = exported_data['records'][0]
        print(f"\n  Sample Record (Record 1):")
        print(f"    ID: {sample_record['record_id']}")
        print(f"    Input hash: {sample_record['input_hash']}")
        print(f"    Output hash: {sample_record.get('output_hash', 'N/A')}")
        print(f"    Execution time: {sample_record.get('metrics', {}).get('execution_time_ms', 'N/A')}ms")
    print()

    # Example 6: Provenance verification
    print("Test 6: Verify Provenance Integrity")
    print("-" * 40)

    # Verify all records have required fields
    required_fields = ['record_id', 'agent_name', 'timestamp_start', 'input_hash']
    all_valid = True

    for record in calculator.provenance_records:
        if not all(field in record for field in required_fields):
            all_valid = False
            break

    if all_valid:
        print(f"  ✓ All provenance records are valid")
        print(f"  ✓ All records have required fields")
        print(f"  ✓ Audit trail is complete")
    else:
        print(f"  ✗ Some provenance records are incomplete")

    print()

    print("=" * 60)
    print("Example complete!")
    print(f"Check '{output_path}' for exported provenance data")
    print("=" * 60)


if __name__ == "__main__":
    main()
