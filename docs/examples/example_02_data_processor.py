# -*- coding: utf-8 -*-
"""
Example 02: Data Processor Agent

This example demonstrates batch data processing using BaseDataProcessor.
You'll learn:
- How to process records in batches
- How to implement record-level validation
- How to handle processing errors gracefully
- How to use parallel processing
"""

from greenlang.agents import BaseDataProcessor, DataProcessorConfig
from typing import Dict, Any


class TemperatureConverter(BaseDataProcessor):
    """
    Convert temperature readings from Fahrenheit to Celsius.

    This agent demonstrates:
    - Batch processing
    - Record validation
    - Error collection
    - Progress tracking
    """

    def __init__(self, parallel_workers=1, enable_progress=True):
        config = DataProcessorConfig(
            name="TemperatureConverter",
            description="Convert temperature data from F to C",
            batch_size=3,  # Small batch for demonstration
            parallel_workers=parallel_workers,
            enable_progress=enable_progress,
            collect_errors=True,
            max_errors=5
        )
        super().__init__(config)

    def process_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert a single temperature reading.

        Args:
            record: Must contain 'sensor_id', 'temperature_f', and 'timestamp'

        Returns:
            Record with temperature_c added
        """
        fahrenheit = record['temperature_f']
        celsius = (fahrenheit - 32) * 5 / 9

        return {
            'sensor_id': record['sensor_id'],
            'timestamp': record['timestamp'],
            'temperature_f': fahrenheit,
            'temperature_c': round(celsius, 2),
            'unit': 'Celsius'
        }

    def validate_record(self, record: Dict[str, Any]) -> bool:
        """
        Validate that record has required fields and sensible values.

        Args:
            record: Record to validate

        Returns:
            True if valid, False otherwise
        """
        # Check required fields
        required_fields = ['sensor_id', 'temperature_f', 'timestamp']
        if not all(field in record for field in required_fields):
            return False

        # Check temperature is realistic (-100F to 200F)
        temp_f = record['temperature_f']
        if not isinstance(temp_f, (int, float)):
            return False

        if temp_f < -100 or temp_f > 200:
            self.logger.warning(f"Unrealistic temperature: {temp_f}°F")
            return False

        return True


def main():
    """Run the example."""
    print("=" * 60)
    print("Example 02: Data Processor Agent")
    print("=" * 60)
    print()

    # Sample temperature data (mix of valid and invalid records)
    temperature_data = [
        # Valid records
        {'sensor_id': 'S001', 'timestamp': '2025-01-01T08:00:00', 'temperature_f': 32},
        {'sensor_id': 'S002', 'timestamp': '2025-01-01T08:00:00', 'temperature_f': 68},
        {'sensor_id': 'S003', 'timestamp': '2025-01-01T08:00:00', 'temperature_f': 98.6},
        {'sensor_id': 'S004', 'timestamp': '2025-01-01T08:00:00', 'temperature_f': 75},
        {'sensor_id': 'S005', 'timestamp': '2025-01-01T08:00:00', 'temperature_f': 50},

        # Invalid records (will be caught by validation)
        {'sensor_id': 'S006', 'timestamp': '2025-01-01T08:00:00', 'temperature_f': 250},  # Too high
        {'sensor_id': 'S007', 'timestamp': '2025-01-01T08:00:00'},  # Missing temperature_f
    ]

    # Example 1: Sequential processing
    print("Test 1: Sequential Processing")
    print("-" * 40)

    converter = TemperatureConverter(parallel_workers=1, enable_progress=True)
    result = converter.run({"records": temperature_data})

    if result.success:
        print(f"\n✓ Processing completed")
        print(f"  Records processed: {result.records_processed}")
        print(f"  Records failed: {result.records_failed}")
        print(f"  Batches processed: {result.batches_processed}")
        print()

        # Show converted temperatures
        print("Converted temperatures:")
        for record in result.data['records'][:5]:  # Show first 5
            print(f"  {record['sensor_id']}: {record['temperature_f']}°F = {record['temperature_c']}°C")

        # Show errors
        if result.errors:
            print(f"\nErrors encountered:")
            for error in result.errors:
                print(f"  Record {error.record_id}: {error.error_message}")
    else:
        print(f"✗ Processing failed: {result.error}")
    print()

    # Example 2: Parallel processing
    print("Test 2: Parallel Processing (4 workers)")
    print("-" * 40)

    converter_parallel = TemperatureConverter(parallel_workers=4, enable_progress=True)
    result_parallel = converter_parallel.run({"records": temperature_data})

    if result_parallel.success:
        print(f"\n✓ Parallel processing completed")
        print(f"  Records processed: {result_parallel.records_processed}")
        print(f"  Records failed: {result_parallel.records_failed}")
    else:
        print(f"✗ Processing failed: {result_parallel.error}")
    print()

    # Example 3: Check statistics
    print("Processing Statistics:")
    print("-" * 40)
    stats = converter.get_processing_stats()
    print(f"  Total executions: {stats['executions']}")
    print(f"  Average time: {stats['avg_time_ms']:.2f}ms")
    print(f"  Records processed: {stats['processing']['records_processed']}")
    print(f"  Record success rate: {stats['processing']['success_rate']}%")
    print()

    print("=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
