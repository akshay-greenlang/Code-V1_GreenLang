#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example 10: Complete Multi-Agent Pipeline
==========================================

This example demonstrates a complete end-to-end pipeline:
- Data intake and validation
- Emissions calculation
- Report generation
- Multi-format output (JSON, Markdown, HTML)

Pipeline stages:
1. IntakeAgent: Load and validate input data
2. CalculatorAgent: Perform emissions calculations
3. ReportAgent: Generate comprehensive reports

Run: python examples/10_complete_pipeline.py
"""

import json
import csv
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
from greenlang.sdk.base import Agent, Result, Metadata, Report
from greenlang.sdk.pipeline import Pipeline as BasePipeline
from greenlang.sdk.context import Context
from greenlang.determinism import DeterministicClock


class IntakeAgent(Agent[str, Dict[str, Any]]):
    """
    Data intake agent - loads and validates CSV data.
    """

    def __init__(self):
        metadata = Metadata(
            id="intake_agent",
            name="Data Intake Agent",
            version="1.0.0",
            description="Load and validate building data from CSV",
            author="GreenLang Examples"
        )
        super().__init__(metadata)

    def validate(self, input_data: str) -> bool:
        """Validate CSV file exists"""
        return Path(input_data).exists()

    def process(self, input_data: str) -> Dict[str, Any]:
        """Load CSV data"""
        csv_path = Path(input_data)

        buildings = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                buildings.append({
                    "building_id": row["building_id"],
                    "name": row["name"],
                    "area_sqm": float(row["area_sqm"]),
                    "electricity_kwh": float(row["electricity_kwh"]),
                    "gas_therms": float(row["gas_therms"]),
                    "location": row["location"]
                })

        self.logger.info(f"Loaded {len(buildings)} buildings from {csv_path}")

        return {
            "buildings": buildings,
            "source_file": str(csv_path),
            "record_count": len(buildings),
            "loaded_at": DeterministicClock.utcnow().isoformat()
        }


class CalculatorAgent(Agent[Dict[str, Any], Dict[str, Any]]):
    """
    Emissions calculator agent.
    """

    def __init__(self):
        metadata = Metadata(
            id="calculator_agent",
            name="Emissions Calculator Agent",
            version="1.0.0",
            description="Calculate emissions for all buildings",
            author="GreenLang Examples"
        )
        super().__init__(metadata)

        # Load emission factors
        data_dir = Path(__file__).parent / "data"
        with open(data_dir / "emission_factors.json") as f:
            self.factors = json.load(f)["factors"]

    def validate(self, input_data: Dict[str, Any]) -> bool:
        """Validate input has buildings"""
        return "buildings" in input_data and len(input_data["buildings"]) > 0

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate emissions for all buildings"""
        buildings = input_data["buildings"]
        country = input_data.get("country", "US")

        results = []
        total_emissions = 0.0

        for building in buildings:
            # Calculate emissions
            elec_kwh = building["electricity_kwh"]
            gas_therms = building["gas_therms"]

            elec_factor = self.factors["electricity"][country]["value"]
            gas_factor = self.factors["natural_gas"][country]["value"]

            elec_emissions_tons = (elec_kwh * elec_factor) / 1000
            gas_emissions_tons = (gas_therms * gas_factor) / 1000
            building_total = elec_emissions_tons + gas_emissions_tons

            total_emissions += building_total

            # Calculate intensity
            intensity = (building_total * 1000) / building["area_sqm"]  # kgCO2e/sqm

            results.append({
                "building_id": building["building_id"],
                "name": building["name"],
                "area_sqm": building["area_sqm"],
                "location": building["location"],
                "electricity_tons": round(elec_emissions_tons, 2),
                "gas_tons": round(gas_emissions_tons, 2),
                "total_tons": round(building_total, 2),
                "intensity_kgco2e_sqm": round(intensity, 2)
            })

        self.logger.info(f"Calculated emissions for {len(results)} buildings")

        return {
            "buildings": results,
            "summary": {
                "total_buildings": len(results),
                "total_emissions_tons": round(total_emissions, 2),
                "average_emissions_tons": round(total_emissions / len(results), 2),
                "country": country
            },
            "calculated_at": DeterministicClock.utcnow().isoformat()
        }


class ReportGeneratorAgent(Agent[Dict[str, Any], Dict[str, Any]]):
    """
    Report generator agent - creates multi-format reports.
    """

    def __init__(self):
        metadata = Metadata(
            id="report_generator",
            name="Report Generator Agent",
            version="1.0.0",
            description="Generate comprehensive emissions reports",
            author="GreenLang Examples"
        )
        super().__init__(metadata)

    def validate(self, input_data: Dict[str, Any]) -> bool:
        """Validate input has calculation results"""
        return "buildings" in input_data and "summary" in input_data

    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate reports in multiple formats"""
        output_dir = Path(__file__).parent / "out" / "pipeline_reports"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate Markdown report
        markdown_content = self._generate_markdown(input_data)
        markdown_path = output_dir / "emissions_report.md"
        with open(markdown_path, 'w') as f:
            f.write(markdown_content)

        # Generate JSON report
        json_path = output_dir / "emissions_report.json"
        with open(json_path, 'w') as f:
            json.dump(input_data, f, indent=2)

        # Generate CSV summary
        csv_path = output_dir / "emissions_summary.csv"
        with open(csv_path, 'w', newline='') as f:
            if input_data["buildings"]:
                fieldnames = input_data["buildings"][0].keys()
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(input_data["buildings"])

        self.logger.info(f"Generated reports in {output_dir}")

        return {
            "reports_generated": {
                "markdown": str(markdown_path),
                "json": str(json_path),
                "csv": str(csv_path)
            },
            "summary": input_data["summary"],
            "generated_at": DeterministicClock.utcnow().isoformat()
        }

    def _generate_markdown(self, data: Dict[str, Any]) -> str:
        """Generate Markdown report"""
        lines = []

        # Header
        lines.append("# Building Emissions Report")
        lines.append("")
        lines.append(f"**Generated:** {DeterministicClock.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Summary
        summary = data["summary"]
        lines.append("## Executive Summary")
        lines.append("")
        lines.append(f"- **Total Buildings:** {summary['total_buildings']}")
        lines.append(f"- **Total Emissions:** {summary['total_emissions_tons']:.2f} metric tons CO2e")
        lines.append(f"- **Average per Building:** {summary['average_emissions_tons']:.2f} tons CO2e")
        lines.append(f"- **Country:** {summary['country']}")
        lines.append("")

        # Building details
        lines.append("## Building Details")
        lines.append("")
        lines.append("| Building | Location | Area (sqm) | Emissions (tCO2e) | Intensity (kgCO2e/sqm) |")
        lines.append("|----------|----------|------------|-------------------|------------------------|")

        for building in data["buildings"]:
            lines.append(
                f"| {building['name']} | {building['location']} | "
                f"{building['area_sqm']:,.0f} | {building['total_tons']:.2f} | "
                f"{building['intensity_kgco2e_sqm']:.2f} |"
            )

        lines.append("")

        # Top emitters
        sorted_buildings = sorted(data["buildings"], key=lambda x: x["total_tons"], reverse=True)
        lines.append("## Top 3 Emitters")
        lines.append("")

        for i, building in enumerate(sorted_buildings[:3], 1):
            lines.append(f"{i}. **{building['name']}** - {building['total_tons']:.2f} tCO2e")

        lines.append("")

        # Footer
        lines.append("---")
        lines.append("*Generated by GreenLang Pipeline v1.0.0*")

        return "\n".join(lines)


class EmissionsPipeline(BasePipeline):
    """
    Complete emissions calculation pipeline.
    """

    def __init__(self):
        metadata = Metadata(
            id="emissions_pipeline",
            name="Complete Emissions Pipeline",
            version="1.0.0",
            description="End-to-end pipeline: intake -> calculate -> report",
            author="GreenLang Examples"
        )
        super().__init__(metadata)

        # Add agents to pipeline
        self.intake = IntakeAgent()
        self.calculator = CalculatorAgent()
        self.reporter = ReportGeneratorAgent()

        self.add_agent(self.intake)
        self.add_agent(self.calculator)
        self.add_agent(self.reporter)

    def execute(self, input_data: str) -> Result:
        """Execute the complete pipeline"""
        try:
            # Stage 1: Intake
            self.logger.info("Stage 1: Data Intake")
            intake_result = self.intake.run(input_data)
            if not intake_result.success:
                return Result(success=False, error=f"Intake failed: {intake_result.error}")

            # Stage 2: Calculate
            self.logger.info("Stage 2: Emissions Calculation")
            calc_result = self.calculator.run(intake_result.data)
            if not calc_result.success:
                return Result(success=False, error=f"Calculation failed: {calc_result.error}")

            # Stage 3: Report
            self.logger.info("Stage 3: Report Generation")
            report_result = self.reporter.run(calc_result.data)
            if not report_result.success:
                return Result(success=False, error=f"Report generation failed: {report_result.error}")

            # Return combined results
            return Result(
                success=True,
                data={
                    "intake": intake_result.data,
                    "calculations": calc_result.data,
                    "reports": report_result.data
                },
                metadata={"pipeline": self.metadata.id, "stages_completed": 3}
            )

        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            return Result(success=False, error=str(e))


def main():
    """Run the example"""
    print("\n" + "="*70)
    print("Example 10: Complete Multi-Agent Pipeline")
    print("="*70 + "\n")

    # Input CSV file
    csv_file = Path(__file__).parent / "data" / "sample_buildings.csv"

    # Create and execute pipeline
    print("Executing Pipeline...")
    print("-" * 70)

    pipeline = EmissionsPipeline()
    result = pipeline.execute(str(csv_file))

    if result.success:
        print("\nPipeline executed successfully!")
        print("=" * 70)

        # Stage 1: Intake
        intake_data = result.data["intake"]
        print(f"\n1. Data Intake:")
        print(f"   - Source: {intake_data['source_file']}")
        print(f"   - Records loaded: {intake_data['record_count']}")

        # Stage 2: Calculations
        calc_data = result.data["calculations"]
        summary = calc_data["summary"]
        print(f"\n2. Emissions Calculations:")
        print(f"   - Buildings processed: {summary['total_buildings']}")
        print(f"   - Total emissions: {summary['total_emissions_tons']:.2f} tCO2e")
        print(f"   - Average per building: {summary['average_emissions_tons']:.2f} tCO2e")

        # Show top emitters
        sorted_buildings = sorted(calc_data["buildings"], key=lambda x: x["total_tons"], reverse=True)
        print(f"\n   Top 3 Emitters:")
        for i, building in enumerate(sorted_buildings[:3], 1):
            print(f"     {i}. {building['name']}: {building['total_tons']:.2f} tCO2e")

        # Stage 3: Reports
        report_data = result.data["reports"]
        print(f"\n3. Report Generation:")
        print(f"   Reports generated:")
        for format_type, path in report_data["reports_generated"].items():
            print(f"     - {format_type.upper()}: {path}")

        print("\n" + "="*70)
        print("View the reports in the out/pipeline_reports/ directory")
        print("="*70)

    else:
        print(f"\nPipeline failed: {result.error}")

    print("\n" + "="*70)
    print("Example completed successfully!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
