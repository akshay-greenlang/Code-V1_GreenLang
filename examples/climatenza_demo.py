"""
Climatenza AI Demo - Solar Thermal Feasibility Analysis

This demo shows how to use the GreenLang SDK to run solar thermal
feasibility analysis for industrial facilities.

Requirements:
    pip install greenlang[analytics]
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from greenlang.sdk import GreenLangClient
import json


def run_basic_analysis():
    """Run basic solar thermal feasibility analysis"""
    print("=" * 60)
    print("CLIMATENZA AI - Basic Solar Thermal Analysis")
    print("=" * 60)
    
    # Initialize client
    client = GreenLangClient()
    
    # Run analysis with default dairy example
    site_path = "climatenza_app/examples/dairy_hotwater_site.yaml"
    
    print(f"\nAnalyzing site: {site_path}")
    print("Running 8760-hour simulation...")
    
    result = client.run_solar_feasibility(site_path)
    
    if result["success"]:
        data = result["data"]
        print("\n✅ Analysis Complete!\n")
        print("📊 RESULTS:")
        print("-" * 40)
        print(f"Solar Fraction:        {data.get('solar_fraction', 0):.1%}")
        print(f"Annual Demand:         {data.get('total_annual_demand_gwh', 0):.3f} GWh")
        print(f"Solar Yield:           {data.get('total_solar_yield_gwh', 0):.3f} GWh")
        print(f"Collectors Required:   {data.get('num_collectors', 0)}")
        print(f"Aperture Area:         {data.get('required_aperture_area_m2', 0):,.0f} m²")
        print(f"Land Area Required:    {data.get('required_land_area_m2', 0):,.0f} m²")
    else:
        print("\n❌ Analysis Failed!")
        for error in result.get("errors", []):
            print(f"  Error: {error}")


def run_custom_location():
    """Demonstrate solar resource analysis for custom location"""
    print("\n" + "=" * 60)
    print("SOLAR RESOURCE ANALYSIS - Custom Location")
    print("=" * 60)
    
    client = GreenLangClient()
    
    # Example locations
    locations = [
        {"name": "Mumbai, India", "lat": 19.076, "lon": 72.877},
        {"name": "Cairo, Egypt", "lat": 30.044, "lon": 31.235},
        {"name": "Phoenix, USA", "lat": 33.448, "lon": -112.074},
    ]
    
    for loc in locations:
        print(f"\n📍 {loc['name']} (Lat: {loc['lat']}, Lon: {loc['lon']})")
        
        result = client.get_solar_resource(loc['lat'], loc['lon'])
        
        if result.get("success"):
            # Parse the solar data
            try:
                import pandas as pd
            except ImportError:
                print("Error: pandas is required for this demo.")
                print("Install it with: pip install greenlang[analytics]")
                return
            solar_df = pd.read_json(result["data"]["solar_resource_df"], orient="split")
            
            # Calculate annual statistics
            annual_dni = solar_df["dni_w_per_m2"].sum() / 1000  # Convert to kWh/m²
            avg_temp = solar_df["temp_c"].mean()
            peak_dni = solar_df["dni_w_per_m2"].max()
            
            print(f"  Annual DNI:     {annual_dni:.0f} kWh/m²/year")
            print(f"  Peak DNI:       {peak_dni:.0f} W/m²")
            print(f"  Avg Temperature: {avg_temp:.1f}°C")


def run_field_sizing():
    """Demonstrate solar field sizing for different demands"""
    print("\n" + "=" * 60)
    print("SOLAR FIELD SIZING - Industrial Applications")
    print("=" * 60)
    
    client = GreenLangClient()
    
    # Different industrial scenarios
    scenarios = [
        {"name": "Small Dairy", "demand_gwh": 0.5},
        {"name": "Textile Factory", "demand_gwh": 2.0},
        {"name": "Food Processing", "demand_gwh": 5.0},
        {"name": "Large Industrial", "demand_gwh": 10.0},
    ]
    
    solar_config = {
        "tech": "ASC",
        "orientation": "N-S",
        "row_spacing_factor": 2.2,
        "tracking": "1-axis"
    }
    
    print("\nSolar Configuration:")
    print(f"  Technology: {solar_config['tech']}")
    print(f"  Orientation: {solar_config['orientation']}")
    print(f"  Tracking: {solar_config['tracking']}")
    print("\nField Sizing Results:")
    print("-" * 50)
    print(f"{'Application':<20} {'Demand':<10} {'Collectors':<12} {'Area (m²)':<12}")
    print("-" * 50)
    
    for scenario in scenarios:
        result = client.calculate_solar_field_size(
            scenario["demand_gwh"],
            solar_config
        )
        
        if result.get("success"):
            data = result["data"]
            print(f"{scenario['name']:<20} {scenario['demand_gwh']:<10.1f} "
                  f"{data['num_collectors']:<12} {data['required_aperture_area_m2']:<12,.0f}")


def create_custom_site_config():
    """Create a custom site configuration file"""
    print("\n" + "=" * 60)
    print("CREATING CUSTOM SITE CONFIGURATION")
    print("=" * 60)
    
    config = {
        "site": {
            "name": "Demo Industrial Facility",
            "country": "IN",
            "lat": 28.613,  # Delhi
            "lon": 77.209,
            "tz": "Asia/Kolkata",
            "land_area_m2": 100000,
            "roof_area_m2": 20000
        },
        "process_demand": {
            "medium": "steam",
            "temp_in_C": 80,
            "temp_out_C": 150,
            "flow_profile": "climatenza_app/examples/data/dairy_hourly_load_2024.csv",
            "schedule": {
                "workdays": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
            }
        },
        "boiler": {
            "type": "NG",
            "rated_steam_tph": 15,
            "efficiency_pct": 85
        },
        "solar_config": {
            "tech": "T160",
            "orientation": "N-S",
            "row_spacing_factor": 2.5,
            "tracking": "1-axis"
        },
        "finance": {
            "currency": "INR",
            "discount_rate_pct": 12,
            "capex_breakdown": {
                "collector": 35000,
                "bos": 20000,
                "epc": 10000
            },
            "opex_pct_of_capex": 2.5,
            "tariff_fuel_per_kWh": 8.0,
            "tariff_elec_per_kWh": 10.0,
            "escalation_fuel_pct": 5,
            "escalation_elec_pct": 4
        },
        "assumptions": {
            "cleaning_cycle_days": 10,
            "soiling_loss_pct": 4,
            "availability_pct": 95
        }
    }
    
    # Save configuration
    output_path = "examples/custom_industrial_site.yaml"
    import yaml
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"\n✅ Custom configuration saved to: {output_path}")
    print("\nConfiguration Summary:")
    print(f"  Location: {config['site']['name']} ({config['site']['lat']}, {config['site']['lon']})")
    print(f"  Process: {config['process_demand']['medium']} ({config['process_demand']['temp_in_C']}°C → {config['process_demand']['temp_out_C']}°C)")
    print(f"  Boiler: {config['boiler']['type']}, {config['boiler']['rated_steam_tph']} TPH")
    print(f"  Solar Tech: {config['solar_config']['tech']} with {config['solar_config']['tracking']} tracking")
    
    return output_path


def main():
    """Run all demos"""
    print("\n" + "🌟" * 30)
    print("     CLIMATENZA AI DEMO SUITE")
    print("     Solar Thermal Feasibility Analysis")
    print("🌟" * 30)
    
    # Run demos
    run_basic_analysis()
    run_custom_location()
    run_field_sizing()
    
    # Create and analyze custom configuration
    custom_path = create_custom_site_config()
    
    print("\n" + "=" * 60)
    print("ANALYZING CUSTOM CONFIGURATION")
    print("=" * 60)
    
    client = GreenLangClient()
    result = client.run_solar_feasibility(custom_path)
    
    if result["success"]:
        data = result["data"]
        print("\n✅ Custom Site Analysis Complete!")
        print(f"\nResults for Demo Industrial Facility:")
        print(f"  Solar Fraction: {data.get('solar_fraction', 0):.1%}")
        print(f"  Collectors: {data.get('num_collectors', 0)}")
        print(f"  Investment Indicator: {data.get('required_aperture_area_m2', 0) * 35000:,.0f} INR")
    
    print("\n" + "🎉" * 30)
    print("     DEMO COMPLETE!")
    print("🎉" * 30)


if __name__ == "__main__":
    main()