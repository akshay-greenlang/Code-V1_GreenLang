"""
Example 05: Batch Processing

This example demonstrates efficient batch processing of large datasets.
You'll learn:
- How to process large datasets efficiently
- How to configure batch sizes
- How to handle errors in batch processing
- How to track progress
"""

from greenlang.agents import BaseDataProcessor, DataProcessorConfig
from typing import Dict, Any
import random


class BuildingEmissionsProcessor(BaseDataProcessor):
    """
    Calculate emissions for multiple buildings in batches.

    This agent demonstrates:
    - Efficient batch processing
    - Error handling for individual records
    - Progress tracking
    - Performance optimization
    """

    def __init__(self, batch_size=100, parallel_workers=1):
        config = DataProcessorConfig(
            name="BuildingEmissionsProcessor",
            description="Process building emissions in batches",
            batch_size=batch_size,
            parallel_workers=parallel_workers,
            enable_progress=True,
            collect_errors=True,
            max_errors=50,
            validate_records=True
        )
        super().__init__(config)

        # Emission factors (kg CO2 per unit)
        self.emission_factors = {
            'electricity_kwh': 0.5,
            'natural_gas_therms': 5.3,
            'diesel_liters': 2.68
        }

    def process_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate emissions for a single building.

        Args:
            record: Building data with energy consumption

        Returns:
            Record with calculated emissions
        """
        building_id = record['building_id']

        # Calculate emissions for each energy type
        total_emissions_kg = 0

        if 'electricity_kwh' in record:
            elec_emissions = record['electricity_kwh'] * self.emission_factors['electricity_kwh']
            total_emissions_kg += elec_emissions
            record['electricity_emissions_kg'] = round(elec_emissions, 2)

        if 'natural_gas_therms' in record:
            gas_emissions = record['natural_gas_therms'] * self.emission_factors['natural_gas_therms']
            total_emissions_kg += gas_emissions
            record['gas_emissions_kg'] = round(gas_emissions, 2)

        if 'diesel_liters' in record:
            diesel_emissions = record['diesel_liters'] * self.emission_factors['diesel_liters']
            total_emissions_kg += diesel_emissions
            record['diesel_emissions_kg'] = round(diesel_emissions, 2)

        # Calculate intensity (if area provided)
        if 'area_sqft' in record and record['area_sqft'] > 0:
            intensity = total_emissions_kg / record['area_sqft']
            record['intensity_kg_per_sqft'] = round(intensity, 4)

        record['total_emissions_kg'] = round(total_emissions_kg, 2)
        record['total_emissions_tons'] = round(total_emissions_kg / 1000, 4)

        return record

    def validate_record(self, record: Dict[str, Any]) -> bool:
        """
        Validate building record.

        Args:
            record: Record to validate

        Returns:
            True if valid, False otherwise
        """
        # Must have building ID
        if 'building_id' not in record:
            return False

        # Must have at least one energy consumption field
        has_energy_data = any(
            field in record
            for field in ['electricity_kwh', 'natural_gas_therms', 'diesel_liters']
        )

        if not has_energy_data:
            return False

        # Validate numeric fields
        numeric_fields = ['electricity_kwh', 'natural_gas_therms', 'diesel_liters', 'area_sqft']
        for field in numeric_fields:
            if field in record:
                value = record[field]
                if not isinstance(value, (int, float)) or value < 0:
                    return False

        return True


def generate_sample_data(num_buildings=1000):
    """Generate sample building data."""
    buildings = []
    for i in range(1, num_buildings + 1):
        building = {
            'building_id': f'B{i:05d}',
            'area_sqft': random.randint(5000, 50000),
            'electricity_kwh': random.randint(10000, 100000),
        }

        # 80% of buildings have natural gas
        if random.random() < 0.8:
            building['natural_gas_therms'] = random.randint(500, 5000)

        # 20% of buildings have diesel backup generators
        if random.random() < 0.2:
            building['diesel_liters'] = random.randint(100, 1000)

        buildings.append(building)

    # Add a few invalid records
    buildings.append({'building_id': 'B99999'})  # Missing energy data
    buildings.append({'building_id': 'INVALID', 'electricity_kwh': -100})  # Negative value

    return buildings


def main():
    """Run the example."""
    print("=" * 60)
    print("Example 05: Batch Processing")
    print("=" * 60)
    print()

    # Example 1: Small batch with sequential processing
    print("Test 1: Small Dataset (Sequential)")
    print("-" * 40)

    small_data = generate_sample_data(50)
    processor = BuildingEmissionsProcessor(batch_size=10, parallel_workers=1)

    result = processor.run({"records": small_data})

    if result.success:
        print(f"\n✓ Processing completed")
        print(f"  Input records: {result.metadata['total_input_records']}")
        print(f"  Processed: {result.records_processed}")
        print(f"  Failed: {result.records_failed}")
        print(f"  Batches: {result.batches_processed}")
        print(f"  Execution time: {result.metrics.execution_time_ms:.2f}ms")

        # Show sample results
        print(f"\n  Sample Results (first 3 buildings):")
        for record in result.data['records'][:3]:
            print(f"    {record['building_id']}: {record['total_emissions_tons']:.4f} tons CO2e")

        # Show errors
        if result.errors:
            print(f"\n  Errors:")
            for error in result.errors:
                print(f"    Record {error.record_id}: {error.error_message}")
    else:
        print(f"✗ Processing failed: {result.error}")
    print()

    # Example 2: Medium batch with sequential processing
    print("Test 2: Medium Dataset (Sequential)")
    print("-" * 40)

    medium_data = generate_sample_data(500)
    processor_seq = BuildingEmissionsProcessor(batch_size=50, parallel_workers=1)

    result_seq = processor_seq.run({"records": medium_data})

    if result_seq.success:
        print(f"\n✓ Sequential processing completed")
        print(f"  Processed: {result_seq.records_processed}")
        print(f"  Execution time: {result_seq.metrics.execution_time_ms:.0f}ms")
    print()

    # Example 3: Medium batch with parallel processing
    print("Test 3: Medium Dataset (Parallel - 4 workers)")
    print("-" * 40)

    processor_par = BuildingEmissionsProcessor(batch_size=50, parallel_workers=4)

    result_par = processor_par.run({"records": medium_data})

    if result_par.success:
        print(f"\n✓ Parallel processing completed")
        print(f"  Processed: {result_par.records_processed}")
        print(f"  Execution time: {result_par.metrics.execution_time_ms:.0f}ms")

        # Calculate speedup
        if result_seq.metrics.execution_time_ms > 0:
            speedup = result_seq.metrics.execution_time_ms / result_par.metrics.execution_time_ms
            print(f"  Speedup: {speedup:.2f}x faster than sequential")
    print()

    # Example 4: Aggregated statistics
    print("Test 4: Aggregate Results")
    print("-" * 40)

    if result_par.success:
        total_emissions = sum(r['total_emissions_tons'] for r in result_par.data['records'])
        avg_emissions = total_emissions / len(result_par.data['records'])
        max_building = max(result_par.data['records'], key=lambda r: r['total_emissions_tons'])
        min_building = min(result_par.data['records'], key=lambda r: r['total_emissions_tons'])

        print(f"  Total emissions: {total_emissions:,.2f} tons CO2e")
        print(f"  Average emissions: {avg_emissions:.2f} tons CO2e per building")
        print(f"  Highest emitter: {max_building['building_id']} ({max_building['total_emissions_tons']:.2f} tons)")
        print(f"  Lowest emitter: {min_building['building_id']} ({min_building['total_emissions_tons']:.2f} tons)")
    print()

    # Example 5: Processing statistics
    print("Processing Statistics:")
    print("-" * 40)
    stats = processor_par.get_processing_stats()
    print(f"  Total executions: {stats['executions']}")
    print(f"  Success rate: {stats['success_rate']}%")
    print(f"  Average execution time: {stats['avg_time_ms']:.2f}ms")
    print(f"  Total records processed: {stats['processing']['records_processed']}")
    print(f"  Record success rate: {stats['processing']['success_rate']}%")
    print()

    print("=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
