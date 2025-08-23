#!/usr/bin/env python3
"""
Example usage of GreenLang SDK for carbon footprint calculations
"""

from greenlang.sdk import GreenLangClient, WorkflowBuilder, AgentBuilder
from greenlang.agents.base import AgentResult
import json


def basic_example():
    """Basic usage example"""
    print("=" * 60)
    print("BASIC CARBON FOOTPRINT CALCULATION")
    print("=" * 60)
    
    client = GreenLangClient()
    
    fuels = [
        {"fuel_type": "electricity", "consumption": 10000, "unit": "kWh"},
        {"fuel_type": "natural_gas", "consumption": 500, "unit": "therms"}
    ]
    
    building_info = {
        "type": "commercial_office",
        "area": 15000,
        "occupancy": 75
    }
    
    result = client.calculate_carbon_footprint(fuels, building_info)
    
    if result["success"]:
        print(f"✅ Total Emissions: {result['data']['total_emissions_tons']:.3f} metric tons CO2e")
    else:
        print(f"❌ Error: {result['errors']}")
    
    print()


def advanced_workflow_example():
    """Advanced workflow with custom configuration"""
    print("=" * 60)
    print("ADVANCED WORKFLOW WITH BENCHMARKING")
    print("=" * 60)
    
    client = GreenLangClient()
    
    workflow = (WorkflowBuilder("advanced_carbon", "Advanced carbon calculation with benchmarking")
        .add_step("validate", "validator", "Validate input data")
        .add_step("calc_electricity", "fuel", "Calculate electricity emissions")
        .with_input_mapping(
            fuel_type="input.fuels[0].fuel_type",
            consumption="input.fuels[0].consumption",
            unit="input.fuels[0].unit"
        )
        .add_step("calc_gas", "fuel", "Calculate natural gas emissions")
        .with_input_mapping(
            fuel_type="input.fuels[1].fuel_type",
            consumption="input.fuels[1].consumption",
            unit="input.fuels[1].unit"
        )
        .add_step("aggregate", "carbon", "Aggregate emissions")
        .with_input_mapping(
            emissions=["results.calc_electricity.data", "results.calc_gas.data"]
        )
        .add_step("benchmark", "benchmark", "Compare to benchmarks")
        .with_input_mapping(
            total_emissions_kg="results.aggregate.data.total_co2e_kg",
            building_area="input.building_info.area",
            building_type="input.building_info.type",
            period_months="input.period.duration"
        )
        .add_step("report", "report", "Generate report")
        .with_input_mapping(
            carbon_data="results.aggregate.data",
            building_info="input.building_info",
            format="input.report_format"
        )
        .with_output_mapping(
            total_emissions="results.aggregate.data.total_co2e_tons",
            rating="results.benchmark.data.rating",
            report="results.report.data.report"
        )
        .build()
    )
    
    client.register_workflow("advanced", workflow)
    
    input_data = {
        "fuels": [
            {"fuel_type": "electricity", "consumption": 20000, "unit": "kWh"},
            {"fuel_type": "natural_gas", "consumption": 1000, "unit": "therms"}
        ],
        "building_info": {
            "type": "commercial_office",
            "area": 30000,
            "occupancy": 150
        },
        "period": {"duration": 1},
        "report_format": "text"
    }
    
    result = client.execute_workflow("advanced", input_data)
    
    if result["success"]:
        print(f"✅ Total Emissions: {result['data']['total_emissions']:.3f} metric tons CO2e")
        print(f"📊 Benchmark Rating: {result['data']['rating']}")
        print("\n📄 Report Preview:")
        print(result['data']['report'][:500] + "...")
    else:
        print(f"❌ Errors: {result['errors']}")
    
    print()


def custom_agent_example():
    """Example of creating and using a custom agent"""
    print("=" * 60)
    print("CUSTOM AGENT EXAMPLE")
    print("=" * 60)
    
    def calculate_renewable_offset(input_data):
        renewable_kwh = input_data.get("renewable_generation", 0)
        offset_kg = renewable_kwh * 0.385
        
        return AgentResult(
            success=True,
            data={
                "renewable_kwh": renewable_kwh,
                "offset_kg": offset_kg,
                "offset_tons": offset_kg / 1000,
                "message": f"Renewable energy offset: {offset_kg:.2f} kg CO2e"
            }
        )
    
    renewable_agent = (AgentBuilder("RenewableAgent", "Calculate renewable energy offsets")
        .with_execute(calculate_renewable_offset)
        .with_parameters(emission_factor=0.385)
        .build()
    )
    
    client = GreenLangClient()
    client.register_agent("renewable", renewable_agent)
    
    result = client.execute_agent("renewable", {"renewable_generation": 5000})
    
    if result["success"]:
        print(f"✅ {result['data']['message']}")
        print(f"   Offset: {result['data']['offset_tons']:.3f} metric tons CO2e")
    else:
        print(f"❌ Error: {result['error']}")
    
    print()


def batch_processing_example():
    """Example of processing multiple buildings"""
    print("=" * 60)
    print("BATCH PROCESSING MULTIPLE BUILDINGS")
    print("=" * 60)
    
    client = GreenLangClient()
    
    buildings = [
        {
            "name": "Office Tower A",
            "fuels": [
                {"fuel_type": "electricity", "consumption": 50000, "unit": "kWh"},
                {"fuel_type": "natural_gas", "consumption": 2000, "unit": "therms"}
            ],
            "area": 75000
        },
        {
            "name": "Retail Center B",
            "fuels": [
                {"fuel_type": "electricity", "consumption": 30000, "unit": "kWh"},
                {"fuel_type": "natural_gas", "consumption": 1000, "unit": "therms"}
            ],
            "area": 45000
        },
        {
            "name": "Warehouse C",
            "fuels": [
                {"fuel_type": "electricity", "consumption": 15000, "unit": "kWh"},
                {"fuel_type": "diesel", "consumption": 500, "unit": "gallons"}
            ],
            "area": 100000
        }
    ]
    
    total_emissions = 0
    
    for building in buildings:
        emissions_list = []
        
        for fuel in building["fuels"]:
            result = client.calculate_emissions(
                fuel["fuel_type"],
                fuel["consumption"],
                fuel["unit"]
            )
            if result["success"]:
                emissions_list.append(result["data"])
        
        agg_result = client.aggregate_emissions(emissions_list)
        
        if agg_result["success"]:
            building_emissions = agg_result["data"]["total_co2e_tons"]
            total_emissions += building_emissions
            intensity = (agg_result["data"]["total_co2e_kg"] / building["area"]) * 12
            
            print(f"📢 {building['name']}:")
            print(f"   Emissions: {building_emissions:.3f} tons CO2e")
            print(f"   Intensity: {intensity:.2f} kg CO2e/sqft/year")
    
    print(f"\n📊 Portfolio Total: {total_emissions:.3f} metric tons CO2e")
    print()


def main():
    """Run all examples"""
    print("\n🌍 GreenLang SDK Examples\n")
    
    basic_example()
    advanced_workflow_example()
    custom_agent_example()
    batch_processing_example()
    
    print("✨ Examples completed successfully!")


if __name__ == "__main__":
    main()