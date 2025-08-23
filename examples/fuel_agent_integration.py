#!/usr/bin/env python3
"""
FuelAgent Integration Examples

This module demonstrates various ways to integrate and use the FuelAgent
in GreenLang workflows and applications.
"""

import json
from typing import List, Dict, Any
from greenlang.agents import FuelAgent
from greenlang.core.orchestrator import Orchestrator
from greenlang.core.workflow import Workflow, WorkflowStep


def example_1_basic_usage():
    """Example 1: Basic single fuel calculation."""
    print("\n=== Example 1: Basic Fuel Calculation ===")
    
    # Initialize agent
    fuel_agent = FuelAgent()
    
    # Calculate emissions for electricity consumption
    result = fuel_agent.run({
        "fuel_type": "electricity",
        "consumption": {"value": 10000, "unit": "kWh"},
        "country": "US",
        "year": 2024
    })
    
    if result["success"]:
        data = result["data"]
        print(f"Fuel Type: {data['fuel_type']}")
        print(f"Consumption: {data['consumption_value']} {data['consumption_unit']}")
        print(f"Emissions: {data['co2e_emissions_kg']:.2f} kg CO2e")
        print(f"Emission Factor: {data['emission_factor']} {data['emission_factor_unit']}")
        if "recommendations" in data:
            print("\nRecommendations:")
            for rec in data["recommendations"][:3]:
                print(f"  - {rec['action']} ({rec['impact']})")
    else:
        print(f"Error: {result['error']['message']}")


def example_2_batch_processing():
    """Example 2: Batch processing multiple fuel sources."""
    print("\n=== Example 2: Batch Processing ===")
    
    fuel_agent = FuelAgent()
    
    # Multiple fuel sources for a building
    fuel_sources = [
        {
            "fuel_type": "electricity",
            "consumption": {"value": 150000, "unit": "kWh"},
            "country": "US"
        },
        {
            "fuel_type": "natural_gas",
            "consumption": {"value": 5000, "unit": "therms"},
            "country": "US"
        },
        {
            "fuel_type": "diesel",
            "consumption": {"value": 500, "unit": "gallons"},
            "country": "US"
        },
        {
            "fuel_type": "solar_pv",
            "consumption": {"value": -30000, "unit": "kWh"},  # Negative for generation
            "country": "US"
        }
    ]
    
    # Process all fuels
    results = fuel_agent.batch_process(fuel_sources)
    
    total_emissions = 0
    print("\nFuel Emissions Breakdown:")
    print("-" * 50)
    
    for i, result in enumerate(results):
        if result["success"]:
            fuel = fuel_sources[i]["fuel_type"]
            emissions = result["data"]["co2e_emissions_kg"]
            total_emissions += emissions
            
            if emissions < 0:
                print(f"{fuel:15} | {emissions:10.2f} kg CO2e (offset)")
            else:
                print(f"{fuel:15} | {emissions:10.2f} kg CO2e")
    
    print("-" * 50)
    print(f"{'Net Total':15} | {total_emissions:10.2f} kg CO2e")
    print(f"{'':15} | {total_emissions/1000:10.2f} metric tons CO2e")


def example_3_regional_comparison():
    """Example 3: Compare emissions across different regions."""
    print("\n=== Example 3: Regional Comparison ===")
    
    fuel_agent = FuelAgent()
    
    # Same electricity consumption in different countries
    countries = ["US", "IN", "FR", "CN", "BR", "NO"]
    consumption = {"value": 100000, "unit": "kWh"}
    
    print(f"\nEmissions for {consumption['value']} {consumption['unit']} electricity:")
    print("-" * 60)
    print(f"{'Country':<10} | {'Emissions (kg CO2e)':<20} | {'Grid Factor':<15}")
    print("-" * 60)
    
    results = []
    for country in countries:
        result = fuel_agent.run({
            "fuel_type": "electricity",
            "consumption": consumption,
            "country": country
        })
        
        if result["success"]:
            emissions = result["data"]["co2e_emissions_kg"]
            factor = result["data"]["emission_factor"]
            results.append((country, emissions, factor))
            print(f"{country:<10} | {emissions:<20.2f} | {factor:<15.3f}")
    
    # Find best and worst
    results.sort(key=lambda x: x[1])
    print("-" * 60)
    print(f"Cleanest Grid: {results[0][0]} ({results[0][1]:.2f} kg)")
    print(f"Most Carbon Intensive: {results[-1][0]} ({results[-1][1]:.2f} kg)")
    print(f"Variation: {results[-1][1]/results[0][1]:.1f}x difference")


def example_4_workflow_integration():
    """Example 4: Integration in a workflow with other agents."""
    print("\n=== Example 4: Workflow Integration ===")
    
    from greenlang.agents import CarbonAgent, BenchmarkAgent
    
    # Create workflow for comprehensive analysis
    orchestrator = Orchestrator()
    
    # Register agents
    fuel_agent = FuelAgent()
    carbon_agent = CarbonAgent()
    benchmark_agent = BenchmarkAgent()
    
    orchestrator.register_agent("fuel", fuel_agent)
    orchestrator.register_agent("carbon", carbon_agent)
    orchestrator.register_agent("benchmark", benchmark_agent)
    
    # Define workflow
    workflow = Workflow(
        name="fuel_emissions_analysis",
        description="Complete fuel emissions analysis with benchmarking",
        steps=[
            WorkflowStep(
                name="calculate_electricity",
                agent_id="fuel",
                description="Calculate electricity emissions"
            ),
            WorkflowStep(
                name="calculate_gas",
                agent_id="fuel",
                description="Calculate natural gas emissions"
            ),
            WorkflowStep(
                name="aggregate",
                agent_id="carbon",
                description="Aggregate total emissions",
                input_mapping={
                    "emissions": "results"
                }
            ),
            WorkflowStep(
                name="benchmark",
                agent_id="benchmark",
                description="Compare against benchmarks",
                input_mapping={
                    "emissions": "results.aggregate.total_co2e_kg",
                    "area": "input.building_area"
                }
            )
        ]
    )
    
    orchestrator.register_workflow("main", workflow)
    
    # Execute workflow
    input_data = {
        "building_area": 50000,  # sqft
        "fuels": [
            {
                "fuel_type": "electricity",
                "consumption": {"value": 200000, "unit": "kWh"},
                "country": "US"
            },
            {
                "fuel_type": "natural_gas",
                "consumption": {"value": 10000, "unit": "therms"},
                "country": "US"
            }
        ]
    }
    
    # Note: This is a simplified example. Actual workflow execution
    # would require proper input mapping configuration
    print("Workflow configured for fuel emissions analysis")
    print(f"Steps: {[step.name for step in workflow.steps]}")


def example_5_renewable_offset_calculation():
    """Example 5: Calculate renewable energy offsets."""
    print("\n=== Example 5: Renewable Energy Offsets ===")
    
    fuel_agent = FuelAgent()
    
    # Building with mixed energy sources including renewables
    energy_profile = {
        "grid_consumption": {
            "fuel_type": "electricity",
            "consumption": {"value": 500000, "unit": "kWh"},
            "country": "US"
        },
        "solar_generation": {
            "fuel_type": "solar_pv",
            "consumption": {"value": -150000, "unit": "kWh"},
            "country": "US"
        },
        "wind_generation": {
            "fuel_type": "wind",
            "consumption": {"value": -50000, "unit": "kWh"},
            "country": "US"
        }
    }
    
    print("Energy Profile Analysis:")
    print("-" * 60)
    
    total_emissions = 0
    total_offset = 0
    
    for source, data in energy_profile.items():
        result = fuel_agent.run(data)
        
        if result["success"]:
            emissions = result["data"]["co2e_emissions_kg"]
            
            if emissions < 0:
                print(f"{source:20} | Generation: {abs(data['consumption']['value']):,} kWh")
                print(f"{'':20} | Offset: {abs(emissions):,.2f} kg CO2e")
                total_offset += abs(emissions)
            else:
                print(f"{source:20} | Consumption: {data['consumption']['value']:,} kWh")
                print(f"{'':20} | Emissions: {emissions:,.2f} kg CO2e")
                total_emissions += emissions
            print()
    
    net_emissions = total_emissions - total_offset
    renewable_percentage = (total_offset / total_emissions * 100) if total_emissions > 0 else 0
    
    print("-" * 60)
    print(f"Total Grid Emissions: {total_emissions:,.2f} kg CO2e")
    print(f"Renewable Offset: {total_offset:,.2f} kg CO2e")
    print(f"Net Emissions: {net_emissions:,.2f} kg CO2e")
    print(f"Renewable Coverage: {renewable_percentage:.1f}%")


def example_6_fuel_switching_analysis():
    """Example 6: Analyze potential savings from fuel switching."""
    print("\n=== Example 6: Fuel Switching Analysis ===")
    
    fuel_agent = FuelAgent()
    
    # Current fuel mix (oil heating)
    current_heating = {
        "fuel_type": "fuel_oil",
        "consumption": {"value": 5000, "unit": "gallons"},
        "country": "US"
    }
    
    # Alternative options
    alternatives = [
        {
            "name": "Natural Gas",
            "fuel_type": "natural_gas",
            "consumption": {"value": 4600, "unit": "therms"},  # Equivalent heating
            "country": "US"
        },
        {
            "name": "Electric Heat Pump",
            "fuel_type": "electricity",
            "consumption": {"value": 42000, "unit": "kWh"},  # With COP of 3
            "country": "US"
        },
        {
            "name": "Biomass Pellets",
            "fuel_type": "biomass",
            "consumption": {"value": 20, "unit": "tons"},
            "country": "US"
        }
    ]
    
    # Calculate current emissions
    current_result = fuel_agent.run(current_heating)
    current_emissions = current_result["data"]["co2e_emissions_kg"] if current_result["success"] else 0
    
    print(f"Current System: Fuel Oil Heating")
    print(f"Annual Consumption: {current_heating['consumption']['value']} gallons")
    print(f"Annual Emissions: {current_emissions:,.2f} kg CO2e")
    print("\nFuel Switching Options:")
    print("-" * 70)
    print(f"{'Alternative':<20} | {'Emissions':<15} | {'Reduction':<15} | {'Savings':<10}")
    print("-" * 70)
    
    for alt in alternatives:
        result = fuel_agent.run(alt)
        
        if result["success"]:
            alt_emissions = result["data"]["co2e_emissions_kg"]
            reduction = current_emissions - alt_emissions
            savings_pct = (reduction / current_emissions * 100) if current_emissions > 0 else 0
            
            print(f"{alt['name']:<20} | {alt_emissions:<15,.2f} | {reduction:<15,.2f} | {savings_pct:<10.1f}%")
    
    print("-" * 70)
    print("\nRecommendation: Consider natural gas or heat pump for significant emissions reduction")


def example_7_advanced_caching():
    """Example 7: Demonstrate caching for performance."""
    print("\n=== Example 7: Caching Performance ===")
    
    import time
    
    fuel_agent = FuelAgent()
    
    # Test data
    test_input = {
        "fuel_type": "electricity",
        "consumption": {"value": 1000, "unit": "kWh"},
        "country": "US"
    }
    
    # First run (cache miss)
    start = time.time()
    for _ in range(100):
        fuel_agent.run(test_input)
    first_run = time.time() - start
    
    # Second run (cache hit)
    start = time.time()
    for _ in range(100):
        fuel_agent.run(test_input)
    second_run = time.time() - start
    
    # Clear cache and run again
    fuel_agent.clear_cache()
    start = time.time()
    for _ in range(100):
        fuel_agent.run(test_input)
    third_run = time.time() - start
    
    print(f"First run (no cache): {first_run:.3f} seconds")
    print(f"Second run (cached): {second_run:.3f} seconds")
    print(f"Third run (cache cleared): {third_run:.3f} seconds")
    print(f"Cache speedup: {first_run/second_run:.1f}x faster")


def main():
    """Run all examples."""
    print("=" * 70)
    print("FuelAgent Integration Examples")
    print("=" * 70)
    
    examples = [
        example_1_basic_usage,
        example_2_batch_processing,
        example_3_regional_comparison,
        example_4_workflow_integration,
        example_5_renewable_offset_calculation,
        example_6_fuel_switching_analysis,
        example_7_advanced_caching
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\nError in {example.__name__}: {e}")
    
    print("\n" + "=" * 70)
    print("Examples completed!")


if __name__ == "__main__":
    main()