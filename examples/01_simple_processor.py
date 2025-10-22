#!/usr/bin/env python3
"""
Example 1: Simple Data Processor

This example demonstrates a basic BaseDataProcessor that transforms
CSV data and calculates emissions for each record.

Key Concepts:
- Creating a custom data processor agent
- Processing records one by one
- Automatic validation with Pydantic
- Simple emissions calculation

Usage:
    python 01_simple_processor.py
"""

from greenlang.sdk import Agent, Result
from pydantic import BaseModel, Field
from typing import Dict
import csv
from pathlib import Path


class BuildingRecord(BaseModel):
    """Input record model with validation"""
    building_id: str = Field(..., description="Unique building identifier")
    electricity_kwh: float = Field(..., gt=0, description="Electricity consumption in kWh")
    gas_therms: float = Field(..., gt=0, description="Gas consumption in therms")


class EmissionsRecord(BaseModel):
    """Output record with emissions data"""
    building_id: str
    electricity_emissions_kg: float
    gas_emissions_kg: float
    total_emissions_tons: float


class SimpleBuildingProcessor(Agent[BuildingRecord, EmissionsRecord]):
    """
    Simple processor that calculates emissions for building records.

    Uses standard emission factors:
    - Electricity: 0.417 kgCO2e/kWh (US average)
    - Natural gas: 5.3 kgCO2e/therm
    """

    def __init__(self):
        super().__init__(
            metadata={
                "id": "simple_building_processor",
                "name": "Simple Building Emissions Processor",
                "version": "1.0.0"
            }
        )
        # Emission factors
        self.electricity_factor = 0.417  # kgCO2e/kWh
        self.gas_factor = 5.3  # kgCO2e/therm

    def validate(self, input_data: BuildingRecord) -> bool:
        """Validation handled by Pydantic"""
        return True

    def process(self, input_data: BuildingRecord) -> EmissionsRecord:
        """
        Process single record and calculate emissions.

        Args:
            input_data: Building record with consumption data

        Returns:
            Emissions record with calculated values
        """
        # Calculate emissions
        elec_emissions = input_data.electricity_kwh * self.electricity_factor
        gas_emissions = input_data.gas_therms * self.gas_factor

        # Convert to tons
        total_tons = (elec_emissions + gas_emissions) / 1000

        return EmissionsRecord(
            building_id=input_data.building_id,
            electricity_emissions_kg=elec_emissions,
            gas_emissions_kg=gas_emissions,
            total_emissions_tons=total_tons
        )


def create_sample_data(filepath: Path):
    """Create sample CSV data for demonstration"""
    data = [
        {"building_id": "B001", "electricity_kwh": "10000", "gas_therms": "500"},
        {"building_id": "B002", "electricity_kwh": "25000", "gas_therms": "1200"},
        {"building_id": "B003", "electricity_kwh": "15000", "gas_therms": "800"},
        {"building_id": "B004", "electricity_kwh": "8000", "gas_therms": "400"},
        {"building_id": "B005", "electricity_kwh": "30000", "gas_therms": "1500"},
    ]

    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["building_id", "electricity_kwh", "gas_therms"])
        writer.writeheader()
        writer.writerows(data)

    print(f"Created sample data: {filepath}")


def main():
    """Run the simple processor example"""
    print("=" * 60)
    print("Example 1: Simple Data Processor")
    print("=" * 60)

    # Create sample data
    sample_file = Path("sample_buildings.csv")
    if not sample_file.exists():
        create_sample_data(sample_file)

    # Initialize processor
    processor = SimpleBuildingProcessor()
    print(f"\nInitialized {processor.metadata['name']}")

    # Load and process data
    print(f"\nProcessing buildings from {sample_file}...\n")

    results = []
    with open(sample_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Create input record
            record = BuildingRecord(
                building_id=row["building_id"],
                electricity_kwh=float(row["electricity_kwh"]),
                gas_therms=float(row["gas_therms"])
            )

            # Process record
            result = processor.run(record)

            if result.success:
                emissions = result.data
                results.append(emissions)

                print(f"Building {emissions.building_id}:")
                print(f"  Electricity: {emissions.electricity_emissions_kg:.2f} kg CO2e")
                print(f"  Gas: {emissions.gas_emissions_kg:.2f} kg CO2e")
                print(f"  Total: {emissions.total_emissions_tons:.3f} tons CO2e")
                print()
            else:
                print(f"Error processing {record.building_id}: {result.error}")

    # Summary
    total_emissions = sum(r.total_emissions_tons for r in results)
    avg_emissions = total_emissions / len(results) if results else 0

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Buildings processed: {len(results)}")
    print(f"Total emissions: {total_emissions:.2f} tons CO2e")
    print(f"Average emissions per building: {avg_emissions:.2f} tons CO2e")
    print()
    print("Key Takeaways:")
    print("  - Simple agent-based processing")
    print("  - Automatic validation with Pydantic")
    print("  - Easy to extend with more complex logic")
    print("  - Type-safe and maintainable")


if __name__ == "__main__":
    main()
