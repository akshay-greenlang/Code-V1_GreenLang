#!/usr/bin/env python3
"""
GreenLang Hello World Example

This is your first GreenLang calculation - a simple office building
carbon footprint analysis. Perfect for verifying your installation
and getting familiar with the GreenLang SDK.

Usage:
    python hello-world.py

Expected runtime: < 30 seconds
"""

import sys
from pathlib import Path

# Add the parent directory to the path so we can import greenlang
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from greenlang.sdk import GreenLangClient
    from greenlang.models import BuildingData, FuelConsumption
except ImportError as e:
    print("❌ GreenLang import failed!")
    print(f"Error: {e}")
    print("\n💡 Try installing GreenLang:")
    print("   pip install greenlang-cli==0.3.0")
    print("   # or")
    print("   pip install greenlang-cli[analytics]==0.3.0")
    sys.exit(1)

def main():
    """
    Calculate carbon emissions for a demo office building.

    This example demonstrates:
    1. Initializing the GreenLang client
    2. Creating building and energy consumption models
    3. Running a basic emissions calculation
    4. Displaying formatted results
    """

    print("🌍 GreenLang Hello World - Carbon Footprint Calculation")
    print("=" * 60)

    # Step 1: Initialize the GreenLang client
    print("\n🔧 Initializing GreenLang client...")
    try:
        client = GreenLangClient()
        print("✅ Client initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize client: {e}")
        return False

    # Step 2: Define building characteristics
    print("\n🏢 Setting up building data...")
    building = BuildingData(
        name="Demo Office Building",
        building_type="commercial_office",
        area_m2=2500,  # 26,910 sq ft
        location="San Francisco, CA",
        occupancy=150,
        year_built=2015
    )
    print(f"   Building: {building.name}")
    print(f"   Type: {building.building_type}")
    print(f"   Area: {building.area_m2:,} m² ({building.area_m2 * 10.764:.0f} sq ft)")
    print(f"   Location: {building.location}")
    print(f"   Occupancy: {building.occupancy} people")

    # Step 3: Define energy consumption
    print("\n⚡ Setting up energy consumption data...")
    energy_data = [
        FuelConsumption(
            fuel_type="electricity",
            consumption=50000,  # kWh/year
            unit="kWh",
            period="annual"
        ),
        FuelConsumption(
            fuel_type="natural_gas",
            consumption=1000,   # therms/year
            unit="therms",
            period="annual"
        )
    ]

    for fuel in energy_data:
        print(f"   {fuel.fuel_type.title()}: {fuel.consumption:,} {fuel.unit}/year")

    # Step 4: Calculate carbon footprint
    print("\n🧮 Calculating carbon emissions...")
    print("   This may take a few seconds...")

    try:
        result = client.calculate_building_emissions(
            building=building,
            energy_consumption=energy_data,
            include_scope3=False  # Start with Scope 1 & 2 only
        )

        if result.success:
            print("✅ Calculation completed successfully!")

            # Step 5: Display results
            print("\n📊 RESULTS")
            print("=" * 40)
            print(f"🏢 Building: {building.name}")
            print(f"📊 Total Annual Emissions: {result.total_emissions_tons:.2f} metric tons CO2e")
            print(f"📏 Emission Intensity: {result.intensity_per_sqft:.3f} kgCO2e/sqft")

            print(f"\n📋 Emissions Breakdown:")
            print(f"   ⚡ Electricity: {result.breakdown.electricity_emissions:.1f} kg CO2e")
            print(f"   🔥 Natural Gas: {result.breakdown.gas_emissions:.1f} kg CO2e")

            # Performance benchmarking (if available)
            if hasattr(result, 'benchmark') and result.benchmark:
                print(f"\n🎯 Performance Analysis:")
                print(f"   Performance Rating: {result.benchmark.rating}")
                print(f"   Compared to Similar Buildings: {result.benchmark.percentile}th percentile")

            # Cost estimates (if available)
            if hasattr(result, 'cost_analysis') and result.cost_analysis:
                print(f"\n💰 Cost Analysis:")
                print(f"   Annual Energy Cost: ${result.cost_analysis.annual_cost:,.2f}")
                print(f"   Carbon Tax Impact (@ $50/tCO2e): ${result.total_emissions_tons * 50:.2f}")

            print(f"\n🌱 Environmental Context:")
            equivalent_cars = result.total_emissions_tons / 4.6  # Average car emissions
            print(f"   Equivalent to {equivalent_cars:.1f} cars driven for a year")

            trees_needed = result.total_emissions_tons * 16  # Trees to offset
            print(f"   Would require {trees_needed:.0f} tree seedlings grown for 10 years to offset")

            print(f"\n✨ Next Steps:")
            print(f"   1. Try the data processing example: python process-data.py")
            print(f"   2. Modify this example with your building data")
            print(f"   3. Explore optimization recommendations")
            print(f"   4. Set up real-time monitoring")

            return True

        else:
            print("❌ Calculation failed!")
            print(f"Errors: {result.errors}")
            return False

    except Exception as e:
        print(f"❌ Calculation error: {e}")
        print("\n💡 Troubleshooting tips:")
        print("   1. Check your internet connection (for emission factors)")
        print("   2. Verify GreenLang installation: pip show greenlang-cli")
        print("   3. Try enabling debug mode: export GL_DEBUG=1")
        return False

def run_quick_test():
    """
    Run a quick test calculation using simplified data.
    """
    print("\n🧪 Running quick validation test...")

    try:
        from greenlang.sdk import GreenLangClient
        client = GreenLangClient()

        # Simple calculation
        fuels = [
            {'fuel_type': 'electricity', 'consumption': 1000, 'unit': 'kWh'},
        ]

        result = client.calculate_carbon_footprint(fuels)

        if result.get('success'):
            emissions = result['data']['total_emissions_tons']
            print(f"✅ Quick test passed: {emissions:.2f} tCO2e for 1000 kWh")
            return True
        else:
            print(f"❌ Quick test failed: {result.get('errors')}")
            return False

    except Exception as e:
        print(f"❌ Quick test error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting GreenLang Hello World Example")
    print("This example calculates carbon emissions for a demo office building.")
    print("Perfect for testing your GreenLang installation!\n")

    # Run the main calculation
    success = main()

    if not success:
        print("\n🔧 Trying simplified validation test...")
        test_success = run_quick_test()

        if not test_success:
            print("\n❌ Both tests failed. Please check your GreenLang installation.")
            sys.exit(1)

    print("\n🎉 Hello World example completed successfully!")
    print("You're ready to start building with GreenLang!")
    print("\n📚 What's next?")
    print("   • Run the data processing example: python process-data.py")
    print("   • Check out examples/tutorials/ for advanced scenarios")
    print("   • Read the docs: https://greenlang.io/docs")
    print("   • Join our community: https://discord.gg/greenlang")