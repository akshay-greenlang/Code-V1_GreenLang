#!/usr/bin/env python3
"""
GreenLang SDK Basic Usage Example

This example demonstrates the fundamental usage of the GreenLang SDK for:
- Creating and configuring agents
- Building and executing workflows
- Processing carbon accounting data
- Handling results and outputs

This is an educational example showing basic SDK patterns.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# GreenLang SDK imports
import greenlang
from greenlang import (
    CarbonAgent,
    ReportAgent,
    InputValidatorAgent,
    Orchestrator,
    Workflow,
    WorkflowStep
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BasicGreenLangExample:
    """Demonstrates basic GreenLang SDK usage patterns"""

    def __init__(self):
        """Initialize the example with basic configuration"""
        self.orchestrator = Orchestrator()
        self.results: Dict[str, Any] = {}

    def setup_agents(self):
        """Set up and register agents for carbon accounting"""
        logger.info("Setting up GreenLang agents...")

        # Configure Carbon Agent for emissions calculations
        carbon_agent = CarbonAgent()
        carbon_agent.configure({
            "default_grid_intensity": 0.82,  # kgCO2e/kWh
            "natural_gas_factor": 53.06,     # kgCO2e/MMBtu
            "calculation_precision": 3
        })

        # Configure Input Validator for data quality
        validator = InputValidatorAgent()
        validator.configure({
            "required_fields": ["building_area_sqft", "electricity_kwh"],
            "validation_strictness": "standard"
        })

        # Configure Report Agent for output generation
        report_agent = ReportAgent()
        report_agent.configure({
            "output_format": "json",
            "include_metadata": True,
            "precision": 2
        })

        # Register agents with orchestrator
        self.orchestrator.register_agent("carbon_calc", carbon_agent)
        self.orchestrator.register_agent("validator", validator)
        self.orchestrator.register_agent("reporter", report_agent)

        logger.info("Agents configured and registered successfully")

    def create_basic_workflow(self) -> Workflow:
        """Create a simple carbon calculation workflow"""
        logger.info("Creating basic carbon accounting workflow...")

        # Define workflow steps
        steps = [
            WorkflowStep(
                name="validate_input",
                agent_id="validator",
                description="Validate input facility data",
                input_mapping={
                    "facility_data": "input.facility_data"
                },
                output_key="validated_data",
                on_failure="stop"
            ),
            WorkflowStep(
                name="calculate_scope1",
                agent_id="carbon_calc",
                description="Calculate Scope 1 emissions from natural gas",
                input_mapping={
                    "natural_gas_mmbtu": "validated_data.natural_gas_mmbtu",
                    "emission_factor": "53.06"
                },
                output_key="scope1_emissions",
                on_failure="continue"
            ),
            WorkflowStep(
                name="calculate_scope2",
                agent_id="carbon_calc",
                description="Calculate Scope 2 emissions from electricity",
                input_mapping={
                    "electricity_kwh": "validated_data.electricity_kwh",
                    "grid_intensity": "validated_data.grid_intensity"
                },
                output_key="scope2_emissions",
                on_failure="continue"
            ),
            WorkflowStep(
                name="generate_report",
                agent_id="reporter",
                description="Generate emissions summary report",
                input_mapping={
                    "scope1_data": "scope1_emissions",
                    "scope2_data": "scope2_emissions",
                    "facility_info": "validated_data"
                },
                output_key="emissions_report"
            )
        ]

        # Create workflow
        workflow = Workflow(
            name="basic_carbon_accounting",
            description="Basic facility carbon footprint calculation",
            version="1.0.0",
            steps=steps,
            output_mapping={
                "total_emissions": "scope1_emissions.total + scope2_emissions.total",
                "detailed_report": "emissions_report",
                "calculation_metadata": "system.execution_info"
            }
        )

        return workflow

    def prepare_sample_data(self) -> Dict[str, Any]:
        """Prepare sample facility data for the example"""
        return {
            "facility_data": {
                "name": "Sample Office Building",
                "location": "San Francisco, CA",
                "building_area_sqft": 50000,
                "electricity_kwh": 350000,  # Annual electricity consumption
                "natural_gas_mmbtu": 2500,   # Annual natural gas consumption
                "grid_intensity": 0.45,      # California grid intensity (kgCO2e/kWh)
                "reporting_year": 2024
            }
        }

    def execute_carbon_analysis(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the carbon analysis workflow"""
        logger.info("Executing carbon analysis workflow...")

        try:
            # Create workflow
            workflow = self.create_basic_workflow()

            # Register workflow with orchestrator
            self.orchestrator.register_workflow("carbon_analysis", workflow)

            # Execute workflow
            results = self.orchestrator.execute_workflow("carbon_analysis", input_data)

            logger.info("Workflow execution completed successfully")
            return results

        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            raise

    def process_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Process and format the analysis results"""
        logger.info("Processing analysis results...")

        # Extract key metrics
        summary = {
            "facility_name": results.get("detailed_report", {}).get("facility_name", "Unknown"),
            "total_emissions_tons": results.get("total_emissions", 0),
            "scope1_emissions_tons": results.get("detailed_report", {}).get("scope1_total", 0),
            "scope2_emissions_tons": results.get("detailed_report", {}).get("scope2_total", 0),
            "emissions_per_sqft": 0,
            "calculation_timestamp": results.get("calculation_metadata", {}).get("timestamp"),
            "data_quality": results.get("detailed_report", {}).get("quality_score", "unknown")
        }

        # Calculate intensity metrics
        building_area = results.get("detailed_report", {}).get("building_area_sqft", 1)
        if building_area > 0:
            summary["emissions_per_sqft"] = summary["total_emissions_tons"] / building_area * 2000  # lbs CO2e per sqft

        return summary

    def save_results(self, results: Dict[str, Any], output_path: str = "basic_example_results.json"):
        """Save results to a JSON file"""
        logger.info(f"Saving results to {output_path}")

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

    def run_example(self):
        """Run the complete basic example"""
        logger.info("=== Starting GreenLang SDK Basic Usage Example ===")

        try:
            # Step 1: Set up agents
            self.setup_agents()

            # Step 2: Prepare sample data
            input_data = self.prepare_sample_data()
            logger.info(f"Using sample facility: {input_data['facility_data']['name']}")

            # Step 3: Execute analysis
            raw_results = self.execute_carbon_analysis(input_data)

            # Step 4: Process results
            processed_results = self.process_results(raw_results)

            # Step 5: Save and display results
            self.save_results(processed_results)

            # Display summary
            self.display_results_summary(processed_results)

            logger.info("=== Basic example completed successfully ===")

            return processed_results

        except Exception as e:
            logger.error(f"Example execution failed: {str(e)}")
            raise

    def display_results_summary(self, results: Dict[str, Any]):
        """Display a formatted summary of the results"""
        print("\n" + "="*60)
        print("CARBON FOOTPRINT ANALYSIS RESULTS")
        print("="*60)
        print(f"Facility: {results.get('facility_name', 'Unknown')}")
        print(f"Total Emissions: {results.get('total_emissions_tons', 0):.2f} tons CO2e")
        print(f"  Scope 1 (Direct): {results.get('scope1_emissions_tons', 0):.2f} tons CO2e")
        print(f"  Scope 2 (Electricity): {results.get('scope2_emissions_tons', 0):.2f} tons CO2e")
        print(f"Emissions Intensity: {results.get('emissions_per_sqft', 0):.2f} lbs CO2e/sqft")
        print(f"Data Quality: {results.get('data_quality', 'Unknown')}")
        print(f"Analysis Timestamp: {results.get('calculation_timestamp', 'Unknown')}")
        print("="*60 + "\n")


class AdvancedWorkflowExample:
    """Demonstrates more advanced SDK patterns"""

    def __init__(self):
        self.orchestrator = Orchestrator()

    def create_multi_facility_workflow(self) -> Workflow:
        """Create a workflow that processes multiple facilities"""
        logger.info("Creating advanced multi-facility workflow...")

        steps = [
            WorkflowStep(
                name="validate_all_facilities",
                agent_id="validator",
                description="Validate all facility data",
                input_mapping={"facilities": "input.facilities"},
                output_key="validated_facilities"
            ),
            WorkflowStep(
                name="calculate_facility_emissions",
                agent_id="carbon_calc",
                description="Calculate emissions for each facility",
                input_mapping={"facilities": "validated_facilities"},
                output_key="facility_emissions"
            ),
            WorkflowStep(
                name="aggregate_portfolio",
                agent_id="carbon_calc",
                description="Aggregate portfolio-level emissions",
                input_mapping={"facility_data": "facility_emissions"},
                output_key="portfolio_totals"
            ),
            WorkflowStep(
                name="benchmark_performance",
                agent_id="carbon_calc",
                description="Compare against industry benchmarks",
                input_mapping={
                    "portfolio_data": "portfolio_totals",
                    "facility_types": "validated_facilities.types"
                },
                output_key="benchmark_results",
                condition="len(validated_facilities) > 1"
            ),
            WorkflowStep(
                name="generate_portfolio_report",
                agent_id="reporter",
                description="Generate comprehensive portfolio report",
                input_mapping={
                    "portfolio_data": "portfolio_totals",
                    "facility_details": "facility_emissions",
                    "benchmarks": "benchmark_results"
                },
                output_key="portfolio_report"
            )
        ]

        return Workflow(
            name="multi_facility_analysis",
            description="Multi-facility portfolio carbon analysis",
            version="2.0.0",
            steps=steps,
            output_mapping={
                "portfolio_emissions": "portfolio_totals.total",
                "facility_count": "len(validated_facilities)",
                "detailed_report": "portfolio_report",
                "benchmark_position": "benchmark_results.percentile"
            }
        )


def demonstrate_workflow_creation():
    """Demonstrate programmatic workflow creation"""
    logger.info("Demonstrating programmatic workflow creation...")

    # Create a workflow programmatically
    workflow = Workflow(
        name="custom_analysis",
        description="Programmatically created carbon analysis",
        version="1.0.0",
        steps=[]
    )

    # Add steps dynamically
    workflow.add_step(WorkflowStep(
        name="data_validation",
        agent_id="validator",
        description="Validate input data"
    ))

    workflow.add_step(WorkflowStep(
        name="emissions_calculation",
        agent_id="carbon_calc",
        description="Calculate carbon emissions"
    ))

    # Save workflow to file
    output_path = "custom_workflow.yaml"
    workflow.to_yaml(output_path)
    logger.info(f"Saved custom workflow to {output_path}")

    # Load workflow from file
    loaded_workflow = Workflow.from_yaml(output_path)
    logger.info(f"Loaded workflow: {loaded_workflow.name} v{loaded_workflow.version}")

    return loaded_workflow


def main():
    """Main function to run all examples"""
    logger.info("Starting GreenLang SDK Examples...")

    # Run basic example
    basic_example = BasicGreenLangExample()
    results = basic_example.run_example()

    # Demonstrate workflow creation
    custom_workflow = demonstrate_workflow_creation()

    logger.info("All SDK examples completed successfully!")

    return results


if __name__ == "__main__":
    main()