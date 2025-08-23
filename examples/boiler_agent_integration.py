#!/usr/bin/env python3
"""
BoilerAgent Integration Examples

This module demonstrates various ways to integrate and use the BoilerAgent
in GreenLang workflows and applications.
"""

import json
import asyncio
from typing import List, Dict, Any
from greenlang.agents import BoilerAgent
from greenlang.core.orchestrator import Orchestrator
from greenlang.core.workflow import Workflow, WorkflowStep


def example_1_basic_boiler_calculation():
    """Example 1: Basic boiler emissions calculation."""
    print("\n=== Example 1: Basic Boiler Calculation ===")
    
    # Initialize agent
    boiler_agent = BoilerAgent()
    
    # Calculate emissions for a condensing natural gas boiler
    result = boiler_agent.run({
        "boiler_type": "condensing",
        "fuel_type": "natural_gas",
        "thermal_output": {"value": 100, "unit": "MMBtu"},
        "efficiency": 0.92,
        "country": "US",
        "age": "new"
    })
    
    if result["success"]:
        data = result["data"]
        print(f"Boiler Type: {data['boiler_type']}")
        print(f"Fuel Type: {data['fuel_type']}")
        print(f"Thermal Output: {data['thermal_output_value']} {data['thermal_output_unit']}")
        print(f"Fuel Consumption: {data['fuel_consumption_value']:.2f} {data['fuel_consumption_unit']}")
        print(f"Efficiency: {data['thermal_efficiency_percent']:.1f}%")
        print(f"Emissions: {data['co2e_emissions_kg']:.2f} kg CO2e")
        print(f"Performance Rating: {data['performance_rating']}")
        
        if data["recommendations"]:
            print("\nTop Recommendations:")
            for rec in data["recommendations"][:3]:
                print(f"  - {rec['action']} ({rec['impact']})")
    else:
        print(f"Error: {result['error']['message']}")


def example_2_batch_processing_multiple_boilers():
    """Example 2: Batch processing for multiple boilers in a facility."""
    print("\n=== Example 2: Batch Processing Multiple Boilers ===")
    
    boiler_agent = BoilerAgent()
    
    # Multiple boilers in a campus or facility
    boilers = [
        {
            "boiler_type": "condensing",
            "fuel_type": "natural_gas",
            "thermal_output": {"value": 500, "unit": "MMBtu"},
            "efficiency": 0.95,
            "age": "new",
            "country": "US"
        },
        {
            "boiler_type": "standard",
            "fuel_type": "natural_gas",
            "thermal_output": {"value": 300, "unit": "MMBtu"},
            "efficiency": 0.78,
            "age": "old",
            "country": "US"
        },
        {
            "boiler_type": "heat_pump",
            "fuel_type": "electricity",
            "thermal_output": {"value": 1000000, "unit": "kWh"},
            "efficiency": 3.2,  # COP
            "age": "new",
            "country": "US"
        },
        {
            "boiler_type": "standard",
            "fuel_type": "oil",
            "fuel_consumption": {"value": 2000, "unit": "gallons"},
            "efficiency": 0.75,
            "age": "medium",
            "country": "US"
        }
    ]
    
    # Process all boilers in batch
    results = boiler_agent.batch_process(boilers)
    
    total_emissions = 0
    total_thermal_output_mmbtu = 0
    
    print("\nFacility Boiler Analysis:")
    print("-" * 70)
    print(f"{'Boiler Type':<15} | {'Fuel':<12} | {'Efficiency':<10} | {'Emissions (kg)':<15}")
    print("-" * 70)
    
    for i, result in enumerate(results):
        if result["success"]:
            data = result["data"]
            total_emissions += data["co2e_emissions_kg"]
            
            # Convert thermal output to MMBtu for aggregation
            if data["thermal_output_unit"] == "kWh":
                total_thermal_output_mmbtu += data["thermal_output_value"] * 0.003412
            else:
                total_thermal_output_mmbtu += data["thermal_output_value"]
            
            print(f"{data['boiler_type']:<15} | {data['fuel_type']:<12} | "
                  f"{data['efficiency']*100:<10.1f} | {data['co2e_emissions_kg']:<15,.2f}")
    
    print("-" * 70)
    print(f"Total Facility Emissions: {total_emissions:,.2f} kg CO2e")
    print(f"Total Thermal Output: {total_thermal_output_mmbtu:,.2f} MMBtu")
    print(f"Average Emission Intensity: {total_emissions/total_thermal_output_mmbtu:.2f} kg CO2e/MMBtu")


def example_3_fuel_vs_thermal_comparison():
    """Example 3: Compare calculating from fuel vs thermal output."""
    print("\n=== Example 3: Fuel vs Thermal Output Comparison ===")
    
    boiler_agent = BoilerAgent()
    
    # Same boiler, two calculation methods
    efficiency = 0.85
    
    # Method 1: From thermal output
    result1 = boiler_agent.run({
        "boiler_type": "standard",
        "fuel_type": "natural_gas",
        "thermal_output": {"value": 100, "unit": "MMBtu"},
        "efficiency": efficiency,
        "country": "US"
    })
    
    # Method 2: From fuel consumption (117.65 therms ≈ 11.765 MMBtu input for 100 MMBtu output at 85% efficiency)
    result2 = boiler_agent.run({
        "boiler_type": "standard",
        "fuel_type": "natural_gas",
        "fuel_consumption": {"value": 1176.5, "unit": "therms"},
        "efficiency": efficiency,
        "country": "US"
    })
    
    print("Calculation Method Comparison:")
    print("-" * 50)
    
    if result1["success"] and result2["success"]:
        print(f"{'Method':<20} | {'Fuel (therms)':<15} | {'Emissions (kg)':<15}")
        print("-" * 50)
        print(f"{'From Thermal Output':<20} | {result1['data']['fuel_consumption_value']:<15.2f} | "
              f"{result1['data']['co2e_emissions_kg']:<15.2f}")
        print(f"{'From Fuel Input':<20} | {result2['data']['fuel_consumption_value']:<15.2f} | "
              f"{result2['data']['co2e_emissions_kg']:<15.2f}")
        
        # Check consistency
        diff = abs(result1['data']['co2e_emissions_kg'] - result2['data']['co2e_emissions_kg'])
        print(f"\nDifference: {diff:.2f} kg CO2e ({diff/result1['data']['co2e_emissions_kg']*100:.2f}%)")
        print("✓ Both methods produce consistent results" if diff < 1 else "⚠ Methods show discrepancy")


def example_4_heat_pump_analysis():
    """Example 4: Heat pump vs traditional boiler comparison."""
    print("\n=== Example 4: Heat Pump vs Traditional Boiler ===")
    
    boiler_agent = BoilerAgent()
    
    # Same heating requirement
    thermal_requirement = {"value": 1000, "unit": "MMBtu"}
    
    systems = [
        {
            "name": "Gas Boiler (Standard)",
            "boiler_type": "standard",
            "fuel_type": "natural_gas",
            "thermal_output": thermal_requirement,
            "efficiency": 0.80
        },
        {
            "name": "Gas Boiler (Condensing)",
            "boiler_type": "condensing",
            "fuel_type": "natural_gas",
            "thermal_output": thermal_requirement,
            "efficiency": 0.95
        },
        {
            "name": "Oil Boiler",
            "boiler_type": "standard",
            "fuel_type": "oil",
            "thermal_output": thermal_requirement,
            "efficiency": 0.78
        },
        {
            "name": "Electric Resistance",
            "boiler_type": "resistance",
            "fuel_type": "electricity",
            "thermal_output": thermal_requirement,
            "efficiency": 0.99
        },
        {
            "name": "Heat Pump (COP 3.0)",
            "boiler_type": "heat_pump",
            "fuel_type": "electricity",
            "thermal_output": thermal_requirement,
            "efficiency": 3.0  # COP
        },
        {
            "name": "Heat Pump (COP 4.0)",
            "boiler_type": "heat_pump",
            "fuel_type": "electricity",
            "thermal_output": thermal_requirement,
            "efficiency": 4.0  # COP
        }
    ]
    
    print(f"Comparison for {thermal_requirement['value']} {thermal_requirement['unit']} heating requirement:")
    print("-" * 80)
    print(f"{'System':<25} | {'Fuel Use':<20} | {'Emissions (kg)':<15} | {'Rating':<10}")
    print("-" * 80)
    
    results = []
    for system in systems:
        system["country"] = "US"
        result = boiler_agent.run(system)
        
        if result["success"]:
            data = result["data"]
            fuel_str = f"{data['fuel_consumption_value']:.1f} {data['fuel_consumption_unit']}"
            
            print(f"{system['name']:<25} | {fuel_str:<20} | "
                  f"{data['co2e_emissions_kg']:<15,.2f} | {data['performance_rating']:<10}")
            
            results.append((system['name'], data['co2e_emissions_kg']))
    
    # Find best option
    if results:
        results.sort(key=lambda x: x[1])
        print("-" * 80)
        print(f"Best Option: {results[0][0]} ({results[0][1]:,.2f} kg CO2e)")
        print(f"Worst Option: {results[-1][0]} ({results[-1][1]:,.2f} kg CO2e)")
        print(f"Potential Savings: {results[-1][1] - results[0][1]:,.2f} kg CO2e "
              f"({(results[-1][1] - results[0][1])/results[-1][1]*100:.1f}%)")


def example_5_workflow_integration():
    """Example 5: Integration in a workflow with other agents."""
    print("\n=== Example 5: Workflow Integration ===")
    
    from greenlang.agents import FuelAgent, CarbonAgent, BenchmarkAgent
    
    # Create workflow for building heating system analysis
    orchestrator = Orchestrator()
    
    # Register agents
    boiler_agent = BoilerAgent()
    fuel_agent = FuelAgent()
    carbon_agent = CarbonAgent()
    benchmark_agent = BenchmarkAgent()
    
    orchestrator.register_agent("boiler", boiler_agent)
    orchestrator.register_agent("fuel", fuel_agent)
    orchestrator.register_agent("carbon", carbon_agent)
    orchestrator.register_agent("benchmark", benchmark_agent)
    
    # Define workflow
    workflow = Workflow(
        name="heating_system_analysis",
        description="Complete heating system emissions analysis",
        steps=[
            WorkflowStep(
                name="calculate_boiler_emissions",
                agent_id="boiler",
                description="Calculate boiler emissions"
            ),
            WorkflowStep(
                name="calculate_auxiliary_fuel",
                agent_id="fuel",
                description="Calculate auxiliary equipment fuel use"
            ),
            WorkflowStep(
                name="aggregate_emissions",
                agent_id="carbon",
                description="Aggregate total system emissions"
            ),
            WorkflowStep(
                name="benchmark_performance",
                agent_id="benchmark",
                description="Compare against industry benchmarks"
            )
        ]
    )
    
    orchestrator.register_workflow("heating_analysis", workflow)
    
    print("Workflow: Heating System Analysis")
    print("Steps:")
    for i, step in enumerate(workflow.steps, 1):
        print(f"  {i}. {step.description} (Agent: {step.agent_id})")
    
    # Example input data
    input_data = {
        "boiler": {
            "boiler_type": "condensing",
            "fuel_type": "natural_gas",
            "thermal_output": {"value": 500, "unit": "MMBtu"},
            "efficiency": 0.92
        },
        "auxiliary": {
            "fuel_type": "electricity",
            "consumption": {"value": 50000, "unit": "kWh"}
        },
        "building_area": 100000  # sqft
    }
    
    print(f"\nWorkflow configured for heating system with:")
    print(f"  - Boiler: {input_data['boiler']['boiler_type']} ({input_data['boiler']['fuel_type']})")
    print(f"  - Auxiliary: {input_data['auxiliary']['consumption']['value']} kWh electricity")
    print(f"  - Building: {input_data['building_area']:,} sqft")


async def example_6_async_processing():
    """Example 6: Async processing for multiple boilers."""
    print("\n=== Example 6: Async Processing ===")
    
    boiler_agent = BoilerAgent()
    
    # Multiple boilers to process concurrently
    boilers = [
        {
            "boiler_type": "condensing",
            "fuel_type": "natural_gas",
            "thermal_output": {"value": 200, "unit": "MMBtu"},
            "country": "US"
        },
        {
            "boiler_type": "standard",
            "fuel_type": "oil",
            "thermal_output": {"value": 150, "unit": "MMBtu"},
            "country": "US"
        },
        {
            "boiler_type": "heat_pump",
            "fuel_type": "electricity",
            "thermal_output": {"value": 500000, "unit": "kWh"},
            "efficiency": 3.5,
            "country": "EU"
        }
    ]
    
    # Process asynchronously
    results = await boiler_agent.async_batch_process(boilers)
    
    print("Async Processing Results:")
    print("-" * 50)
    
    for i, result in enumerate(results):
        if result["success"]:
            data = result["data"]
            print(f"Boiler {i+1}: {data['co2e_emissions_kg']:.2f} kg CO2e "
                  f"({data['performance_rating']} performance)")
        else:
            print(f"Boiler {i+1}: Error - {result['error']['message']}")


def example_7_performance_monitoring():
    """Example 7: Performance monitoring and optimization."""
    print("\n=== Example 7: Performance Monitoring ===")
    
    import time
    
    boiler_agent = BoilerAgent()
    
    # Test data for performance benchmarking
    test_boiler = {
        "boiler_type": "standard",
        "fuel_type": "natural_gas",
        "thermal_output": {"value": 100, "unit": "MMBtu"},
        "country": "US"
    }
    
    # Run multiple calculations to test caching
    print("Testing cache performance...")
    
    # First run (cache miss)
    start = time.time()
    for _ in range(10):
        boiler_agent.run(test_boiler)
    first_run = time.time() - start
    
    # Second run (cache hit)
    start = time.time()
    for _ in range(10):
        boiler_agent.run(test_boiler)
    second_run = time.time() - start
    
    # Get performance metrics
    metrics = boiler_agent.get_performance_metrics()
    
    print(f"\nPerformance Metrics:")
    print(f"  First run (10 calculations): {first_run:.3f} seconds")
    print(f"  Second run (cached): {second_run:.3f} seconds")
    print(f"  Cache speedup: {first_run/second_run:.1f}x")
    print(f"  Average execution time: {metrics['average_execution_time_ms']:.2f} ms")
    print(f"  Cache hit rate: {metrics['cache_hit_rate']:.1%}")
    print(f"  Total calculations: {metrics['total_calculations']}")


def example_8_export_results():
    """Example 8: Export results to different formats."""
    print("\n=== Example 8: Export Results ===")
    
    boiler_agent = BoilerAgent()
    
    # Generate results for multiple boilers
    boilers = [
        {
            "boiler_type": "condensing",
            "fuel_type": "natural_gas",
            "thermal_output": {"value": 100, "unit": "MMBtu"},
            "efficiency": 0.95,
            "country": "US"
        },
        {
            "boiler_type": "standard",
            "fuel_type": "oil",
            "thermal_output": {"value": 80, "unit": "MMBtu"},
            "efficiency": 0.75,
            "country": "US"
        }
    ]
    
    results = []
    for boiler in boilers:
        result = boiler_agent.run(boiler)
        if result["success"]:
            results.append(result["data"])
    
    # Export to different formats
    formats = ["json", "csv"]
    
    for fmt in formats:
        try:
            file_path = boiler_agent.export_results(results, fmt)
            print(f"Results exported to {fmt.upper()}: {file_path}")
        except Exception as e:
            print(f"Failed to export to {fmt}: {e}")


def example_9_historical_tracking():
    """Example 9: Historical performance tracking."""
    print("\n=== Example 9: Historical Performance Tracking ===")
    
    boiler_agent = BoilerAgent()
    
    # Simulate monthly boiler performance over a year
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    
    # Winter months need more heating
    seasonal_factors = [1.5, 1.4, 1.2, 0.9, 0.5, 0.3, 
                       0.2, 0.2, 0.4, 0.8, 1.2, 1.5]
    
    base_thermal = 100  # MMBtu
    
    print("Monthly Boiler Performance:")
    print("-" * 60)
    print(f"{'Month':<10} | {'Thermal (MMBtu)':<15} | {'Emissions (kg)':<15}")
    print("-" * 60)
    
    total_thermal = 0
    total_emissions = 0
    
    for month, factor in zip(months, seasonal_factors):
        thermal_output = base_thermal * factor
        
        result = boiler_agent.run({
            "boiler_type": "condensing",
            "fuel_type": "natural_gas",
            "thermal_output": {"value": thermal_output, "unit": "MMBtu"},
            "efficiency": 0.90,
            "country": "US"
        })
        
        if result["success"]:
            emissions = result["data"]["co2e_emissions_kg"]
            total_thermal += thermal_output
            total_emissions += emissions
            
            print(f"{month:<10} | {thermal_output:<15.1f} | {emissions:<15,.2f}")
    
    print("-" * 60)
    print(f"{'Annual':<10} | {total_thermal:<15.1f} | {total_emissions:<15,.2f}")
    print(f"\nAverage emission intensity: {total_emissions/total_thermal:.2f} kg CO2e/MMBtu")


def example_10_recommendation_analysis():
    """Example 10: Analyze and prioritize recommendations."""
    print("\n=== Example 10: Recommendation Analysis ===")
    
    boiler_agent = BoilerAgent()
    
    # Different boiler scenarios
    scenarios = [
        {
            "name": "Old Oil Boiler",
            "config": {
                "boiler_type": "low_efficiency",
                "fuel_type": "oil",
                "thermal_output": {"value": 200, "unit": "MMBtu"},
                "efficiency": 0.65,
                "age": "old",
                "country": "US"
            }
        },
        {
            "name": "Medium Age Gas Boiler",
            "config": {
                "boiler_type": "standard",
                "fuel_type": "natural_gas",
                "thermal_output": {"value": 200, "unit": "MMBtu"},
                "efficiency": 0.78,
                "age": "medium",
                "country": "US"
            }
        },
        {
            "name": "New Condensing Boiler",
            "config": {
                "boiler_type": "condensing",
                "fuel_type": "natural_gas",
                "thermal_output": {"value": 200, "unit": "MMBtu"},
                "efficiency": 0.95,
                "age": "new",
                "country": "US"
            }
        }
    ]
    
    for scenario in scenarios:
        result = boiler_agent.run(scenario["config"])
        
        if result["success"]:
            data = result["data"]
            
            print(f"\n{scenario['name']}:")
            print(f"  Current Efficiency: {data['thermal_efficiency_percent']:.1f}%")
            print(f"  Current Emissions: {data['co2e_emissions_kg']:,.2f} kg CO2e")
            print(f"  Performance Rating: {data['performance_rating']}")
            
            if data["recommendations"]:
                print("\n  Recommendations:")
                
                # Group by priority
                high_priority = [r for r in data["recommendations"] if r["priority"] == "high"]
                medium_priority = [r for r in data["recommendations"] if r["priority"] == "medium"]
                low_priority = [r for r in data["recommendations"] if r["priority"] == "low"]
                
                if high_priority:
                    print("    High Priority:")
                    for rec in high_priority:
                        print(f"      • {rec['action']}")
                        print(f"        Impact: {rec['impact']}, Payback: {rec['payback']}")
                
                if medium_priority:
                    print("    Medium Priority:")
                    for rec in medium_priority:
                        print(f"      • {rec['action']}")
                        print(f"        Impact: {rec['impact']}, Payback: {rec['payback']}")
                
                if low_priority:
                    print("    Low Priority:")
                    for rec in low_priority:
                        print(f"      • {rec['action']}")
                        print(f"        Impact: {rec['impact']}, Payback: {rec['payback']}")


def main():
    """Run all examples."""
    print("=" * 80)
    print("BoilerAgent Integration Examples")
    print("=" * 80)
    
    examples = [
        example_1_basic_boiler_calculation,
        example_2_batch_processing_multiple_boilers,
        example_3_fuel_vs_thermal_comparison,
        example_4_heat_pump_analysis,
        example_5_workflow_integration,
        # example_6_async_processing,  # Requires async context
        example_7_performance_monitoring,
        example_8_export_results,
        example_9_historical_tracking,
        example_10_recommendation_analysis
    ]
    
    for example in examples:
        try:
            if asyncio.iscoroutinefunction(example):
                asyncio.run(example())
            else:
                example()
        except Exception as e:
            print(f"\nError in {example.__name__}: {e}")
    
    # Run async example separately
    print("\n=== Example 6: Async Processing ===")
    try:
        asyncio.run(example_6_async_processing())
    except Exception as e:
        print(f"Error in async example: {e}")
    
    print("\n" + "=" * 80)
    print("Examples completed!")


if __name__ == "__main__":
    main()