#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example 2: CSV Batch Processor with Error Handling
===================================================

This example demonstrates batch processing of CSV data with comprehensive
error handling, validation, and progress tracking.

Run: python examples/02_data_processor.py
"""

import csv
import json
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass
from greenlang.sdk.base import Agent, Result, Metadata
from greenlang.determinism import FinancialDecimal


@dataclass
class ProcessingStats:
    """Statistics for batch processing"""
    total_records: int = 0
    successful: int = 0
    failed: int = 0
    errors: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage"""
        if self.total_records == 0:
            return 0.0
        return (self.successful / self.total_records) * 100


class BuildingBatchProcessor(Agent[str, Dict[str, Any]]):
    """
    Batch processor for building energy data from CSV files.

    Processes multiple building records with error handling and validation.
    """

    def __init__(self):
        metadata = Metadata(
            id="building_batch_processor",
            name="Building Batch Processor",
            version="1.0.0",
            description="Process building energy data in batches with error handling",
            author="GreenLang Examples"
        )
        super().__init__(metadata)

        # Load emission factors
        data_dir = Path(__file__).parent / "data"
        with open(data_dir / "emission_factors.json") as f:
            self.factors = json.load(f)["factors"]

    def validate(self, input_data: str) -> bool:
        """Validate CSV file path exists"""
        csv_path = Path(input_data)
        if not csv_path.exists():
            self.logger.error(f"CSV file not found: {csv_path}")
            return False
        return True

    def process(self, input_data: str) -> Dict[str, Any]:
        """Process all buildings from CSV file"""
        csv_path = Path(input_data)
        stats = ProcessingStats()
        results = []

        self.logger.info(f"Processing CSV file: {csv_path}")

        # Read and process CSV
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)

            for row_num, row in enumerate(reader, start=2):  # Start at 2 (after header)
                stats.total_records += 1

                try:
                    # Process individual building
                    building_result = self._process_building(row)

                    if building_result["success"]:
                        stats.successful += 1
                        results.append(building_result)
                        self.logger.info(
                            f"Row {row_num}: Processed {row['name']} - "
                            f"{building_result['emissions_tons']:.2f} tCO2e"
                        )
                    else:
                        stats.failed += 1
                        stats.errors.append({
                            "row": row_num,
                            "building_id": row.get("building_id", "unknown"),
                            "error": building_result["error"]
                        })

                except Exception as e:
                    stats.failed += 1
                    stats.errors.append({
                        "row": row_num,
                        "building_id": row.get("building_id", "unknown"),
                        "error": str(e)
                    })
                    self.logger.error(f"Row {row_num}: Error - {e}")

        # Calculate aggregate statistics
        total_emissions = sum(r["emissions_tons"] for r in results)
        avg_emissions = total_emissions / len(results) if results else 0

        return {
            "summary": {
                "total_records": stats.total_records,
                "successful": stats.successful,
                "failed": stats.failed,
                "success_rate": round(stats.success_rate, 2),
                "total_emissions_tons": round(total_emissions, 2),
                "average_emissions_tons": round(avg_emissions, 2)
            },
            "results": results,
            "errors": stats.errors
        }

    def _process_building(self, row: Dict[str, str]) -> Dict[str, Any]:
        """Process a single building record"""
        try:
            # Extract and convert data
            building_id = row["building_id"]
            name = row["name"]
            electricity_kwh = float(row["electricity_kwh"])
            gas_therms = float(row["gas_therms"])
            location = row.get("location", "US")

            # Validate values
            if electricity_kwh < 0 or gas_therms < 0:
                return {
                    "success": False,
                    "error": "Energy consumption cannot be negative"
                }

            # Calculate emissions (default to US factors)
            elec_factor = self.factors["electricity"]["US"]["value"]
            gas_factor = self.factors["natural_gas"]["US"]["value"]

            elec_emissions = (electricity_kwh * elec_factor) / 1000  # tons
            gas_emissions = (gas_therms * gas_factor) / 1000  # tons
            total_emissions = elec_emissions + gas_emissions

            return {
                "success": True,
                "building_id": building_id,
                "name": name,
                "location": location,
                "emissions_tons": round(total_emissions, 2),
                "electricity_tons": round(elec_emissions, 2),
                "gas_tons": round(gas_emissions, 2),
                "area_sqm": float(row.get("area_sqm", 0)),
                "intensity_kgco2e_sqm": round((total_emissions * 1000) / FinancialDecimal.from_string(row.get("area_sqm", 1)), 2)
            }

        except KeyError as e:
            return {
                "success": False,
                "error": f"Missing required field: {e}"
            }
        except ValueError as e:
            return {
                "success": False,
                "error": f"Invalid numeric value: {e}"
            }


def main():
    """Run the example"""
    print("\n" + "="*70)
    print("Example 2: CSV Batch Processor with Error Handling")
    print("="*70 + "\n")

    # Create processor instance
    processor = BuildingBatchProcessor()

    # Process the sample buildings CSV
    csv_file = Path(__file__).parent / "data" / "sample_buildings.csv"

    print(f"Processing file: {csv_file}")
    print("-" * 70)

    result = processor.run(str(csv_file))

    if result.success:
        summary = result.data["summary"]
        results = result.data["results"]
        errors = result.data["errors"]

        print("\nBatch Processing Summary:")
        print(f"  Total Records: {summary['total_records']}")
        print(f"  Successful: {summary['successful']}")
        print(f"  Failed: {summary['failed']}")
        print(f"  Success Rate: {summary['success_rate']}%")
        print(f"\n  Total Emissions: {summary['total_emissions_tons']:.2f} metric tons CO2e")
        print(f"  Average per Building: {summary['average_emissions_tons']:.2f} tons CO2e")

        print("\n\nDetailed Results:")
        print("-" * 70)
        for i, building in enumerate(results, 1):
            print(f"{i}. {building['name']} ({building['building_id']})")
            print(f"   Location: {building['location']}")
            print(f"   Emissions: {building['emissions_tons']:.2f} tons CO2e")
            print(f"   Intensity: {building['intensity_kgco2e_sqm']:.2f} kgCO2e/sqm")
            print(f"   Breakdown: {building['electricity_tons']:.2f} (elec) + "
                  f"{building['gas_tons']:.2f} (gas)")
            print()

        if errors:
            print("\nErrors Encountered:")
            print("-" * 70)
            for error in errors:
                print(f"  Row {error['row']} ({error['building_id']}): {error['error']}")

        # Save results to output file
        output_dir = Path(__file__).parent / "out"
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / "batch_processing_results.json"

        with open(output_file, 'w') as f:
            json.dump(result.data, f, indent=2)

        print(f"\n\nResults saved to: {output_file}")

    else:
        print(f"Error: {result.error}")

    print("\n" + "="*70)
    print("Example completed successfully!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
