#!/usr/bin/env python3
"""
GreenLang Developer Guide - Comprehensive Examples
This file demonstrates all features of GreenLang for developers
"""

from greenlang.sdk import GreenLangClient, WorkflowBuilder, AgentBuilder
from greenlang.agents.base import AgentResult
from greenlang.core.workflow import Workflow
import json
from typing import Dict, Any, List


class GreenLangDeveloperGuide:
    """Complete developer guide with examples"""
    
    def __init__(self):
        self.client = GreenLangClient()
    
    def example_1_basic_calculation(self):
        """Example 1: Basic Emissions Calculation"""
        print("\n" + "="*60)
        print("EXAMPLE 1: Basic Emissions Calculation")
        print("="*60)
        
        # Single fuel calculation
        result = self.client.calculate_emissions(
            fuel_type="electricity",
            consumption=5000,  # 5000 kWh
            unit="kWh",
            region="US"
        )
        
        print(f"Electricity: 5000 kWh")
        print(f"Emissions: {result['data']['co2e_emissions_kg']:.2f} kg CO2e")
        print(f"          ({result['data']['co2e_emissions_tons']:.3f} metric tons)")
        
        return result
    
    def example_2_multiple_fuels(self):
        """Example 2: Multiple Fuel Sources"""
        print("\n" + "="*60)
        print("EXAMPLE 2: Multiple Fuel Sources")
        print("="*60)
        
        # Calculate for multiple fuels
        fuels = [
            ("electricity", 10000, "kWh"),
            ("natural_gas", 800, "therms"),
            ("diesel", 100, "gallons"),
            ("gasoline", 50, "gallons")
        ]
        
        emissions_list = []
        total = 0
        
        for fuel_type, consumption, unit in fuels:
            result = self.client.calculate_emissions(fuel_type, consumption, unit)
            if result["success"]:
                emissions = result["data"]["co2e_emissions_kg"]
                emissions_list.append(result["data"])
                total += emissions
                print(f"{fuel_type:15} {consumption:8.1f} {unit:10} = {emissions:10.2f} kg CO2e")
        
        print(f"\n{'TOTAL':15} {' ':8} {' ':10} = {total:10.2f} kg CO2e")
        print(f"{'':15} {' ':8} {' ':10} = {total/1000:10.3f} metric tons")
        
        return emissions_list
    
    def example_3_building_benchmark(self):
        """Example 3: Building Benchmark Analysis"""
        print("\n" + "="*60)
        print("EXAMPLE 3: Building Benchmark Analysis")
        print("="*60)
        
        # Building parameters
        building_data = {
            "type": "commercial_office",
            "area": 50000,  # 50,000 sqft
            "occupancy": 250,
            "location": "New York, NY"
        }
        
        # Annual consumption
        annual_electricity = 1500000  # kWh
        annual_gas = 10000  # therms
        
        print(f"Building: {building_data['type']}")
        print(f"Area: {building_data['area']:,} sqft")
        print(f"Occupancy: {building_data['occupancy']} people")
        print(f"Annual electricity: {annual_electricity:,} kWh")
        print(f"Annual natural gas: {annual_gas:,} therms")
        
        # Calculate emissions
        elec_result = self.client.calculate_emissions("electricity", annual_electricity, "kWh")
        gas_result = self.client.calculate_emissions("natural_gas", annual_gas, "therms")
        
        total_kg = elec_result["data"]["co2e_emissions_kg"] + gas_result["data"]["co2e_emissions_kg"]
        
        # Benchmark
        benchmark_result = self.client.benchmark_emissions(
            total_emissions_kg=total_kg,
            building_area=building_data["area"],
            building_type=building_data["type"],
            period_months=12
        )
        
        if benchmark_result["success"]:
            data = benchmark_result["data"]
            print(f"\nBenchmark Results:")
            print(f"  Rating: {data['rating']}")
            print(f"  Carbon Intensity: {data['carbon_intensity']:.2f} kg CO2e/sqft/year")
            print(f"  Percentile: Top {data['percentile']}%")
            print(f"\nTop Recommendations:")
            for i, rec in enumerate(data["recommendations"][:3], 1):
                print(f"  {i}. {rec}")
        
        return benchmark_result
    
    def example_4_custom_workflow(self):
        """Example 4: Custom Workflow Creation"""
        print("\n" + "="*60)
        print("EXAMPLE 4: Custom Workflow Creation")
        print("="*60)
        
        # Build a custom workflow
        workflow = (WorkflowBuilder("comprehensive_analysis", "Complete carbon analysis workflow")
            .add_step("validate", "validator", "Validate input data")
            .add_step("calc_electricity", "fuel", "Calculate electricity emissions")
            .with_input_mapping(
                fuel_type="'electricity'",
                consumption="input.electricity_kwh",
                unit="'kWh'"
            )
            .add_step("calc_gas", "fuel", "Calculate gas emissions")
            .with_input_mapping(
                fuel_type="'natural_gas'",
                consumption="input.gas_therms",
                unit="'therms'"
            )
            .add_step("aggregate", "carbon", "Aggregate all emissions")
            .add_step("benchmark", "benchmark", "Compare to industry standards")
            .with_condition("'building_area' in context['input']")
            .add_step("report", "report", "Generate final report")
            .with_output_mapping(
                total_emissions="results.aggregate.data.total_co2e_tons",
                rating="results.benchmark.data.rating",
                report="results.report.data.report"
            )
            .build()
        )
        
        # Register workflow
        self.client.register_workflow("comprehensive", workflow)
        
        # Execute workflow
        input_data = {
            "electricity_kwh": 5000,
            "gas_therms": 300,
            "building_area": 10000,
            "building_type": "commercial_office"
        }
        
        print("Executing workflow with:")
        print(f"  Electricity: {input_data['electricity_kwh']} kWh")
        print(f"  Natural Gas: {input_data['gas_therms']} therms")
        print(f"  Building Area: {input_data['building_area']} sqft")
        
        result = self.client.execute_workflow("comprehensive", input_data)
        
        if result["success"]:
            print(f"\nWorkflow Results:")
            if "data" in result:
                print(f"  Total Emissions: {result['data'].get('total_emissions', 'N/A')} tons")
                print(f"  Rating: {result['data'].get('rating', 'N/A')}")
        
        return result
    
    def example_5_custom_agent(self):
        """Example 5: Creating a Custom Agent"""
        print("\n" + "="*60)
        print("EXAMPLE 5: Creating a Custom Agent")
        print("="*60)
        
        # Define custom agent logic
        def renewable_offset_calculator(input_data: Dict[str, Any]) -> AgentResult:
            """Calculate carbon offset from renewable energy"""
            solar_kwh = input_data.get("solar_generation", 0)
            wind_kwh = input_data.get("wind_generation", 0)
            
            # Assume grid emission factor
            grid_factor = 0.385  # kg CO2e/kWh
            
            total_renewable = solar_kwh + wind_kwh
            offset_kg = total_renewable * grid_factor
            
            return AgentResult(
                success=True,
                data={
                    "solar_kwh": solar_kwh,
                    "wind_kwh": wind_kwh,
                    "total_renewable_kwh": total_renewable,
                    "offset_kg": offset_kg,
                    "offset_tons": offset_kg / 1000,
                    "equivalent_trees": int(offset_kg / 21.77),  # One tree absorbs ~21.77 kg CO2/year
                    "message": f"Renewable energy offset: {offset_kg:.2f} kg CO2e"
                }
            )
        
        # Build and register agent
        renewable_agent = (AgentBuilder("RenewableOffsetAgent", "Calculate renewable energy carbon offsets")
            .with_execute(renewable_offset_calculator)
            .with_parameters(
                grid_emission_factor=0.385,
                tree_absorption_rate=21.77
            )
            .build()
        )
        
        self.client.register_agent("renewable_offset", renewable_agent)
        
        # Test the agent
        test_data = {
            "solar_generation": 5000,  # kWh
            "wind_generation": 3000     # kWh
        }
        
        print("Testing Renewable Offset Agent:")
        print(f"  Solar: {test_data['solar_generation']} kWh")
        print(f"  Wind: {test_data['wind_generation']} kWh")
        
        result = self.client.execute_agent("renewable_offset", test_data)
        
        if result["success"]:
            data = result["data"]
            print(f"\nResults:")
            print(f"  Total Renewable: {data['total_renewable_kwh']} kWh")
            print(f"  Carbon Offset: {data['offset_tons']:.3f} metric tons CO2e")
            print(f"  Equivalent to planting {data['equivalent_trees']} trees")
        
        return result
    
    def example_6_portfolio_analysis(self):
        """Example 6: Portfolio-Level Analysis"""
        print("\n" + "="*60)
        print("EXAMPLE 6: Portfolio-Level Analysis")
        print("="*60)
        
        # Define portfolio of buildings
        portfolio = [
            {
                "name": "Corporate HQ",
                "type": "commercial_office",
                "area": 100000,
                "electricity_monthly": 150000,
                "gas_monthly": 5000
            },
            {
                "name": "Distribution Center",
                "type": "warehouse",
                "area": 200000,
                "electricity_monthly": 80000,
                "gas_monthly": 2000
            },
            {
                "name": "Retail Store A",
                "type": "retail",
                "area": 25000,
                "electricity_monthly": 30000,
                "gas_monthly": 1000
            },
            {
                "name": "Retail Store B",
                "type": "retail",
                "area": 20000,
                "electricity_monthly": 25000,
                "gas_monthly": 800
            }
        ]
        
        portfolio_results = []
        total_emissions = 0
        total_area = 0
        
        for building in portfolio:
            print(f"\nAnalyzing: {building['name']}")
            print(f"  Type: {building['type']}")
            print(f"  Area: {building['area']:,} sqft")
            
            # Calculate annual emissions
            annual_electricity = building['electricity_monthly'] * 12
            annual_gas = building['gas_monthly'] * 12
            
            elec_result = self.client.calculate_emissions("electricity", annual_electricity, "kWh")
            gas_result = self.client.calculate_emissions("natural_gas", annual_gas, "therms")
            
            building_emissions_kg = (
                elec_result["data"]["co2e_emissions_kg"] + 
                gas_result["data"]["co2e_emissions_kg"]
            )
            building_emissions_tons = building_emissions_kg / 1000
            
            # Benchmark
            benchmark = self.client.benchmark_emissions(
                building_emissions_kg,
                building["area"],
                building["type"],
                12
            )
            
            # Store results
            portfolio_results.append({
                "name": building["name"],
                "emissions_tons": building_emissions_tons,
                "area": building["area"],
                "intensity": building_emissions_kg / building["area"],
                "rating": benchmark["data"]["rating"] if benchmark["success"] else "N/A"
            })
            
            total_emissions += building_emissions_tons
            total_area += building["area"]
            
            print(f"  Annual Emissions: {building_emissions_tons:.2f} metric tons CO2e")
            print(f"  Carbon Intensity: {building_emissions_kg/building['area']:.2f} kg CO2e/sqft/year")
            print(f"  Rating: {benchmark['data']['rating'] if benchmark['success'] else 'N/A'}")
        
        # Portfolio summary
        print("\n" + "-"*60)
        print("PORTFOLIO SUMMARY")
        print("-"*60)
        print(f"Total Buildings: {len(portfolio)}")
        print(f"Total Area: {total_area:,} sqft")
        print(f"Total Emissions: {total_emissions:.2f} metric tons CO2e/year")
        print(f"Portfolio Intensity: {(total_emissions*1000)/total_area:.2f} kg CO2e/sqft/year")
        
        # Ranking
        print("\nBuildings by Performance:")
        sorted_buildings = sorted(portfolio_results, key=lambda x: x["intensity"])
        for i, building in enumerate(sorted_buildings, 1):
            print(f"  {i}. {building['name']:20} - {building['intensity']:.2f} kg/sqft - {building['rating']}")
        
        return portfolio_results
    
    def example_7_time_series_analysis(self):
        """Example 7: Time Series Analysis"""
        print("\n" + "="*60)
        print("EXAMPLE 7: Time Series Analysis")
        print("="*60)
        
        # Monthly consumption data
        monthly_data = [
            {"month": "Jan", "electricity": 12000, "gas": 1500},
            {"month": "Feb", "electricity": 11500, "gas": 1400},
            {"month": "Mar", "electricity": 10500, "gas": 1200},
            {"month": "Apr", "electricity": 9000, "gas": 800},
            {"month": "May", "electricity": 8500, "gas": 500},
            {"month": "Jun", "electricity": 11000, "gas": 300},
            {"month": "Jul", "electricity": 13000, "gas": 200},
            {"month": "Aug", "electricity": 13500, "gas": 200},
            {"month": "Sep", "electricity": 11500, "gas": 300},
            {"month": "Oct", "electricity": 9500, "gas": 600},
            {"month": "Nov", "electricity": 10500, "gas": 1000},
            {"month": "Dec", "electricity": 12500, "gas": 1300}
        ]
        
        print("Monthly Emissions Analysis:")
        print("-" * 60)
        print(f"{'Month':<8} {'Elec (kWh)':<12} {'Gas (therms)':<12} {'CO2e (tons)':<12}")
        print("-" * 60)
        
        monthly_emissions = []
        total_emissions = 0
        
        for data in monthly_data:
            # Calculate emissions
            elec_result = self.client.calculate_emissions("electricity", data["electricity"], "kWh")
            gas_result = self.client.calculate_emissions("natural_gas", data["gas"], "therms")
            
            month_emissions = (
                elec_result["data"]["co2e_emissions_tons"] + 
                gas_result["data"]["co2e_emissions_tons"]
            )
            
            monthly_emissions.append(month_emissions)
            total_emissions += month_emissions
            
            print(f"{data['month']:<8} {data['electricity']:<12,} {data['gas']:<12,} {month_emissions:<12.3f}")
        
        # Statistics
        avg_emissions = total_emissions / 12
        max_emissions = max(monthly_emissions)
        min_emissions = min(monthly_emissions)
        
        print("-" * 60)
        print(f"{'TOTAL':<8} {sum(d['electricity'] for d in monthly_data):<12,} "
              f"{sum(d['gas'] for d in monthly_data):<12,} {total_emissions:<12.3f}")
        
        print("\nStatistics:")
        print(f"  Average Monthly: {avg_emissions:.3f} tons")
        print(f"  Maximum Month: {max_emissions:.3f} tons")
        print(f"  Minimum Month: {min_emissions:.3f} tons")
        print(f"  Variation: {((max_emissions - min_emissions) / avg_emissions * 100):.1f}%")
        
        return monthly_emissions
    
    def example_8_complete_report(self):
        """Example 8: Complete Emissions Report"""
        print("\n" + "="*60)
        print("EXAMPLE 8: Complete Emissions Report")
        print("="*60)
        
        # Comprehensive building data
        building_info = {
            "name": "Tech Campus Building A",
            "type": "commercial_office",
            "area": 75000,
            "occupancy": 350,
            "location": "San Francisco, CA",
            "year_built": 2015
        }
        
        # Fuel consumption data
        fuels = [
            {"fuel_type": "electricity", "consumption": 1200000, "unit": "kWh"},
            {"fuel_type": "natural_gas", "consumption": 8000, "unit": "therms"},
            {"fuel_type": "diesel", "consumption": 500, "unit": "gallons"}  # Backup generator
        ]
        
        print(f"Building: {building_info['name']}")
        print(f"Location: {building_info['location']}")
        print(f"Type: {building_info['type']}")
        print(f"Area: {building_info['area']:,} sqft")
        print(f"Occupancy: {building_info['occupancy']} people")
        
        # Calculate emissions for each fuel
        emissions_list = []
        for fuel in fuels:
            result = self.client.calculate_emissions(
                fuel["fuel_type"],
                fuel["consumption"],
                fuel["unit"]
            )
            if result["success"]:
                emissions_list.append(result["data"])
        
        # Aggregate emissions
        agg_result = self.client.aggregate_emissions(emissions_list)
        
        # Add building info to aggregated data
        if agg_result["success"]:
            agg_result["data"]["building_info"] = building_info
        
        # Benchmark analysis
        benchmark_result = self.client.benchmark_emissions(
            agg_result["data"]["total_co2e_kg"],
            building_info["area"],
            building_info["type"],
            12
        )
        
        # Generate comprehensive report
        report_result = self.client.generate_report(
            carbon_data=agg_result["data"],
            format="text",
            building_info=building_info
        )
        
        # Display report
        if report_result["success"]:
            print("\n" + "="*60)
            print(report_result["data"]["report"])
            
            if benchmark_result["success"]:
                print("\n" + "="*60)
                print("BENCHMARK ANALYSIS")
                print("="*60)
                bench_data = benchmark_result["data"]
                print(f"Performance Rating: {bench_data['rating']}")
                print(f"Industry Percentile: Top {bench_data['percentile']}%")
                print(f"Carbon Intensity: {bench_data['carbon_intensity']:.2f} kg CO2e/sqft/year")
                
                if bench_data["comparison"]["improvement_to_good"] > 0:
                    print(f"\nTo achieve 'Good' rating:")
                    print(f"  Reduce emissions by: {bench_data['comparison']['improvement_to_good']:.2f} kg CO2e/sqft/year")
                    print(f"  Total reduction needed: {bench_data['comparison']['improvement_to_good'] * building_info['area'] / 1000:.2f} tons/year")
        
        return {
            "emissions": agg_result["data"] if agg_result["success"] else None,
            "benchmark": benchmark_result["data"] if benchmark_result["success"] else None,
            "report": report_result["data"]["report"] if report_result["success"] else None
        }
    
    def run_all_examples(self):
        """Run all examples"""
        examples = [
            self.example_1_basic_calculation,
            self.example_2_multiple_fuels,
            self.example_3_building_benchmark,
            self.example_4_custom_workflow,
            self.example_5_custom_agent,
            self.example_6_portfolio_analysis,
            self.example_7_time_series_analysis,
            self.example_8_complete_report
        ]
        
        print("\n" + "="*60)
        print("GREENLANG DEVELOPER GUIDE - RUNNING ALL EXAMPLES")
        print("="*60)
        
        for i, example in enumerate(examples, 1):
            try:
                example()
                print(f"\n✓ Example {i} completed successfully")
            except Exception as e:
                print(f"\n✗ Example {i} failed: {e}")
        
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED")
        print("="*60)


def main():
    """Main function to run the developer guide"""
    guide = GreenLangDeveloperGuide()
    
    print("""
    ╔════════════════════════════════════════════════════════╗
    ║           GreenLang Developer Guide                    ║
    ║                                                        ║
    ║  Complete examples for emissions calculations,        ║
    ║  benchmarking, workflows, and custom agents          ║
    ╚════════════════════════════════════════════════════════╝
    """)
    
    while True:
        print("\nSelect an example to run:")
        print("1. Basic Emissions Calculation")
        print("2. Multiple Fuel Sources")
        print("3. Building Benchmark Analysis")
        print("4. Custom Workflow Creation")
        print("5. Creating a Custom Agent")
        print("6. Portfolio-Level Analysis")
        print("7. Time Series Analysis")
        print("8. Complete Emissions Report")
        print("9. Run All Examples")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-9): ")
        
        if choice == "0":
            print("Exiting...")
            break
        elif choice == "1":
            guide.example_1_basic_calculation()
        elif choice == "2":
            guide.example_2_multiple_fuels()
        elif choice == "3":
            guide.example_3_building_benchmark()
        elif choice == "4":
            guide.example_4_custom_workflow()
        elif choice == "5":
            guide.example_5_custom_agent()
        elif choice == "6":
            guide.example_6_portfolio_analysis()
        elif choice == "7":
            guide.example_7_time_series_analysis()
        elif choice == "8":
            guide.example_8_complete_report()
        elif choice == "9":
            guide.run_all_examples()
        else:
            print("Invalid choice. Please try again.")
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()