#!/usr/bin/env python3
"""
Example 8: Parallel Batch Processing
=====================================

This example demonstrates:
- Parallel processing of multiple buildings
- Thread pool execution for performance
- Progress tracking and error handling
- Performance comparison: serial vs parallel

Run: python examples/08_parallel_processing.py
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from greenlang.sdk.base import Agent, Result, Metadata


class ParallelBuildingProcessor(Agent[List[Dict[str, Any]], Dict[str, Any]]):
    """
    Parallel processor for multiple buildings.

    Uses ThreadPoolExecutor for concurrent processing.
    """

    def __init__(self, max_workers: int = 4):
        metadata = Metadata(
            id="parallel_processor",
            name="Parallel Building Processor",
            version="1.0.0",
            description="Process multiple buildings in parallel",
            author="GreenLang Examples"
        )
        super().__init__(metadata)

        self.max_workers = max_workers

        # Load emission factors
        data_dir = Path(__file__).parent / "data"
        with open(data_dir / "emission_factors.json") as f:
            self.factors = json.load(f)["factors"]

    def validate(self, input_data: List[Dict[str, Any]]) -> bool:
        """Validate input is a list of building records"""
        if not isinstance(input_data, list):
            return False
        return len(input_data) > 0

    def process(self, input_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process buildings in parallel"""
        start_time = time.time()

        results = []
        errors = []

        # Process in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_building = {
                executor.submit(self._process_single_building, building): building
                for building in input_data
            }

            # Collect results as they complete
            for future in as_completed(future_to_building):
                building = future_to_building[future]
                try:
                    result = future.result()
                    if result["success"]:
                        results.append(result)
                    else:
                        errors.append({
                            "building_id": building.get("building_id", "unknown"),
                            "error": result.get("error", "Unknown error")
                        })
                except Exception as e:
                    errors.append({
                        "building_id": building.get("building_id", "unknown"),
                        "error": str(e)
                    })

        elapsed_time = time.time() - start_time

        # Calculate aggregate statistics
        total_emissions = sum(r["emissions_tons"] for r in results)
        avg_emissions = total_emissions / len(results) if results else 0

        return {
            "processing_stats": {
                "total_buildings": len(input_data),
                "successful": len(results),
                "failed": len(errors),
                "elapsed_time_seconds": round(elapsed_time, 3),
                "buildings_per_second": round(len(input_data) / elapsed_time, 2),
                "workers_used": self.max_workers
            },
            "aggregate_stats": {
                "total_emissions_tons": round(total_emissions, 2),
                "average_emissions_tons": round(avg_emissions, 2)
            },
            "results": results,
            "errors": errors
        }

    def _process_single_building(self, building: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single building (can be run in parallel)"""
        try:
            # Simulate some processing time
            time.sleep(0.1)  # 100ms per building

            building_id = building["building_id"]
            electricity_kwh = building["electricity_kwh"]
            gas_therms = building["gas_therms"]
            country = building.get("country", "US")

            # Calculate emissions
            elec_factor = self.factors["electricity"][country]["value"]
            gas_factor = self.factors["natural_gas"][country]["value"]

            elec_emissions = (electricity_kwh * elec_factor) / 1000
            gas_emissions = (gas_therms * gas_factor) / 1000
            total_emissions = elec_emissions + gas_emissions

            return {
                "success": True,
                "building_id": building_id,
                "name": building.get("name", "Unknown"),
                "emissions_tons": round(total_emissions, 2),
                "electricity_tons": round(elec_emissions, 2),
                "gas_tons": round(gas_emissions, 2)
            }

        except Exception as e:
            return {
                "success": False,
                "building_id": building.get("building_id", "unknown"),
                "error": str(e)
            }

    def process_serial(self, input_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process buildings serially for comparison"""
        start_time = time.time()

        results = []
        errors = []

        for building in input_data:
            result = self._process_single_building(building)
            if result["success"]:
                results.append(result)
            else:
                errors.append({
                    "building_id": result["building_id"],
                    "error": result["error"]
                })

        elapsed_time = time.time() - start_time

        total_emissions = sum(r["emissions_tons"] for r in results)
        avg_emissions = total_emissions / len(results) if results else 0

        return {
            "processing_stats": {
                "total_buildings": len(input_data),
                "successful": len(results),
                "failed": len(errors),
                "elapsed_time_seconds": round(elapsed_time, 3),
                "buildings_per_second": round(len(input_data) / elapsed_time, 2),
                "workers_used": 1
            },
            "aggregate_stats": {
                "total_emissions_tons": round(total_emissions, 2),
                "average_emissions_tons": round(avg_emissions, 2)
            },
            "results": results,
            "errors": errors
        }


def generate_test_buildings(count: int) -> List[Dict[str, Any]]:
    """Generate test building data"""
    import random
    random.seed(42)  # For reproducibility

    buildings = []
    for i in range(count):
        buildings.append({
            "building_id": f"B{i+1:03d}",
            "name": f"Building {i+1}",
            "electricity_kwh": random.randint(10000, 100000),
            "gas_therms": random.randint(500, 5000),
            "country": "US"
        })

    return buildings


def main():
    """Run the example"""
    print("\n" + "="*70)
    print("Example 8: Parallel Batch Processing")
    print("="*70 + "\n")

    # Generate test data
    test_buildings = generate_test_buildings(20)

    # Test 1: Serial processing
    print("Test 1: Serial Processing (1 worker)")
    print("-" * 70)

    processor_serial = ParallelBuildingProcessor(max_workers=1)
    result_serial = processor_serial.process_serial(test_buildings)

    stats_serial = result_serial["processing_stats"]
    print(f"Buildings processed: {stats_serial['successful']}/{stats_serial['total_buildings']}")
    print(f"Time: {stats_serial['elapsed_time_seconds']:.3f} seconds")
    print(f"Throughput: {stats_serial['buildings_per_second']:.2f} buildings/sec")
    print(f"Total emissions: {result_serial['aggregate_stats']['total_emissions_tons']:.2f} tCO2e")

    # Test 2: Parallel processing with 4 workers
    print("\n\nTest 2: Parallel Processing (4 workers)")
    print("-" * 70)

    processor_parallel = ParallelBuildingProcessor(max_workers=4)
    result_parallel = processor_parallel.run(test_buildings)

    if result_parallel.success:
        stats_parallel = result_parallel.data["processing_stats"]
        print(f"Buildings processed: {stats_parallel['successful']}/{stats_parallel['total_buildings']}")
        print(f"Time: {stats_parallel['elapsed_time_seconds']:.3f} seconds")
        print(f"Throughput: {stats_parallel['buildings_per_second']:.2f} buildings/sec")
        print(f"Total emissions: {result_parallel.data['aggregate_stats']['total_emissions_tons']:.2f} tCO2e")

        # Calculate speedup
        speedup = stats_serial['elapsed_time_seconds'] / stats_parallel['elapsed_time_seconds']
        print(f"\nSpeedup: {speedup:.2f}x faster than serial")

    # Test 3: Compare different worker counts
    print("\n\nTest 3: Worker Count Comparison")
    print("-" * 70)

    worker_counts = [1, 2, 4, 8]
    results_comparison = []

    for workers in worker_counts:
        processor = ParallelBuildingProcessor(max_workers=workers)
        result = processor.run(test_buildings)

        if result.success:
            stats = result.data["processing_stats"]
            results_comparison.append({
                "workers": workers,
                "time": stats['elapsed_time_seconds'],
                "throughput": stats['buildings_per_second']
            })

    print(f"{'Workers':<10} {'Time (s)':<12} {'Throughput':<15} {'Speedup':<10}")
    print("-" * 70)

    baseline_time = results_comparison[0]['time']
    for item in results_comparison:
        speedup = baseline_time / item['time']
        print(f"{item['workers']:<10} {item['time']:<12.3f} {item['throughput']:<15.2f} {speedup:<10.2f}x")

    # Test 4: Large batch
    print("\n\nTest 4: Large Batch (100 buildings)")
    print("-" * 70)

    large_batch = generate_test_buildings(100)

    processor_large = ParallelBuildingProcessor(max_workers=8)
    result_large = processor_large.run(large_batch)

    if result_large.success:
        stats_large = result_large.data["processing_stats"]
        agg_large = result_large.data["aggregate_stats"]

        print(f"Buildings processed: {stats_large['successful']}/{stats_large['total_buildings']}")
        print(f"Time: {stats_large['elapsed_time_seconds']:.3f} seconds")
        print(f"Throughput: {stats_large['buildings_per_second']:.2f} buildings/sec")
        print(f"Total emissions: {agg_large['total_emissions_tons']:.2f} tCO2e")
        print(f"Average per building: {agg_large['average_emissions_tons']:.2f} tCO2e")

        # Show sample results
        print(f"\nSample Results (first 5):")
        for result in result_large.data['results'][:5]:
            print(f"  {result['building_id']} ({result['name']}): {result['emissions_tons']:.2f} tCO2e")

    print("\n" + "="*70)
    print("Example completed successfully!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
