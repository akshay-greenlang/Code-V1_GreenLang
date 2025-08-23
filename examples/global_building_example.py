"""
GreenLang Framework - Global Commercial Building Emissions Example
Demonstrates comprehensive building analysis across different countries
"""

from greenlang.sdk.enhanced_client import GreenLangClient
import json
from datetime import datetime


def analyze_us_office():
    """Analyze a US commercial office building"""
    print("\n" + "="*60)
    print("ANALYZING US COMMERCIAL OFFICE")
    print("="*60)
    
    client = GreenLangClient(region="US")
    
    # Define building data
    us_office = {
        "metadata": {
            "building_type": "commercial_office",
            "area": 75000,
            "area_unit": "sqft",
            "location": {
                "country": "US",
                "region": "New York",
                "city": "New York City"
            },
            "occupancy": 400,
            "floor_count": 20,
            "building_age": 25,
            "climate_zone": "4A"
        },
        "energy_consumption": {
            "electricity": {"value": 2500000, "unit": "kWh"},
            "natural_gas": {"value": 50000, "unit": "therms"},
            "diesel": {"value": 1000, "unit": "gallons"}
        }
    }
    
    # Analyze building
    results = client.analyze_building(us_office)
    
    if results["success"]:
        data = results["data"]
        
        # Display results
        print(f"\nTotal Annual Emissions: {data['emissions']['total_co2e_tons']:.2f} metric tons CO2e")
        print(f"Carbon Intensity: {data['intensity']['intensities']['per_sqft_year']:.2f} kgCO2e/sqft/year")
        print(f"Performance Rating: {data['intensity']['performance_rating']}")
        print(f"Benchmark Rating: {data['benchmark']['rating']}")
        
        # Show top recommendations
        print("\nTop Recommendations:")
        for i, rec in enumerate(data['recommendations']['quick_wins'][:3], 1):
            print(f"  {i}. {rec['action']}")
            print(f"     Impact: {rec['impact']}, Payback: {rec['payback']}")
    
    return results


def analyze_india_hospital():
    """Analyze an Indian hospital"""
    print("\n" + "="*60)
    print("ANALYZING INDIAN HOSPITAL")
    print("="*60)
    
    client = GreenLangClient(region="IN")
    
    # Define building data
    india_hospital = {
        "metadata": {
            "building_type": "hospital",
            "area": 100000,
            "area_unit": "sqft",
            "location": {
                "country": "IN",
                "region": "Maharashtra",
                "city": "Mumbai"
            },
            "occupancy": 500,
            "floor_count": 8,
            "building_age": 10,
            "climate_zone": "tropical"
        },
        "energy_consumption": {
            "electricity": {"value": 3500000, "unit": "kWh"},
            "diesel": {"value": 50000, "unit": "liters"},  # For backup generators
            "lpg_propane": {"value": 1000, "unit": "kg"}  # For kitchen/heating
        }
    }
    
    # Analyze building
    results = client.analyze_building(india_hospital)
    
    if results["success"]:
        data = results["data"]
        
        print(f"\nTotal Annual Emissions: {data['emissions']['total_co2e_tons']:.2f} metric tons CO2e")
        print(f"Carbon Intensity: {data['intensity']['intensities']['per_sqft_year']:.2f} kgCO2e/sqft/year")
        print(f"Performance Rating: {data['intensity']['performance_rating']}")
        
        # India-specific metrics
        if 'per_sqm_year' in data['intensity']['intensities']:
            print(f"EPI (Energy Performance Index): {data['intensity']['intensities']['per_sqm_year']:.2f} kgCO2e/sqm/year")
        
        print("\nTop Recommendations:")
        for i, rec in enumerate(data['recommendations']['recommendations'][:3], 1):
            print(f"  {i}. {rec['action']}")
    
    return results


def analyze_eu_data_center():
    """Analyze an EU data center"""
    print("\n" + "="*60)
    print("ANALYZING EU DATA CENTER")
    print("="*60)
    
    client = GreenLangClient(region="EU")
    
    # Define data center
    eu_datacenter = {
        "metadata": {
            "building_type": "data_center",
            "area": 50000,
            "area_unit": "sqft",
            "location": {
                "country": "DE",  # Germany
                "region": "Bavaria",
                "city": "Munich"
            },
            "occupancy": 50,
            "floor_count": 3,
            "building_age": 5,
            "climate_zone": "temperate"
        },
        "energy_consumption": {
            "electricity": {"value": 15000000, "unit": "kWh"},  # High consumption for data center
            "district_heating": {"value": 500000, "unit": "kWh"},
            "solar_pv_generation": {"value": 1000000, "unit": "kWh"}  # On-site solar
        }
    }
    
    # Analyze
    results = client.analyze_building(eu_datacenter)
    
    if results["success"]:
        data = results["data"]
        
        print(f"\nTotal Annual Emissions: {data['emissions']['total_co2e_tons']:.2f} metric tons CO2e")
        print(f"Carbon Intensity: {data['intensity']['intensities']['per_sqft_year']:.2f} kgCO2e/sqft/year")
        
        # Data center specific metrics
        if 'typical_pue' in data.get('profile', {}):
            print(f"Expected PUE Range: {data['profile']['typical_pue']}")
        
        print(f"Performance Rating: {data['intensity']['performance_rating']}")
        
        # Show renewable impact
        print("\nRenewable Energy Impact:")
        print(f"  Solar PV Generation: 1,000,000 kWh/year")
        print(f"  Grid emissions offset: ~230 tons CO2e/year")
    
    return results


def compare_global_offices():
    """Compare office buildings across different countries"""
    print("\n" + "="*60)
    print("GLOBAL OFFICE COMPARISON")
    print("="*60)
    
    # Standard office profile (50,000 sqft, similar consumption)
    base_consumption = {
        "electricity": {"value": 1500000, "unit": "kWh"},
        "natural_gas": {"value": 30000, "unit": "therms"}
    }
    
    countries = ["US", "IN", "EU", "CN", "JP", "BR"]
    results = {}
    
    for country in countries:
        client = GreenLangClient(region=country)
        
        building = {
            "metadata": {
                "building_type": "commercial_office",
                "area": 50000,
                "area_unit": "sqft",
                "location": {"country": country},
                "occupancy": 200,
                "floor_count": 10,
                "building_age": 10
            },
            "energy_consumption": base_consumption
        }
        
        result = client.analyze_building(building)
        if result["success"]:
            results[country] = {
                "emissions_tons": result["data"]["emissions"]["total_co2e_tons"],
                "intensity": result["data"]["intensity"]["intensities"]["per_sqft_year"],
                "rating": result["data"]["intensity"]["performance_rating"]
            }
    
    # Display comparison
    print("\nSame Building, Different Countries:")
    print("-" * 50)
    print(f"{'Country':<10} {'Emissions (tons)':<20} {'Intensity':<20} {'Rating':<15}")
    print("-" * 50)
    
    for country, data in sorted(results.items(), key=lambda x: x[1]["emissions_tons"]):
        print(f"{country:<10} {data['emissions_tons']:<20.1f} {data['intensity']:<20.2f} {data['rating']:<15}")
    
    # Analysis
    print("\nKey Insights:")
    cleanest = min(results.items(), key=lambda x: x[1]["emissions_tons"])[0]
    dirtiest = max(results.items(), key=lambda x: x[1]["emissions_tons"])[0]
    
    print(f"  • Cleanest grid: {cleanest} (lowest emissions)")
    print(f"  • Most carbon-intensive: {dirtiest}")
    print(f"  • Variation: {results[dirtiest]['emissions_tons']/results[cleanest]['emissions_tons']:.1f}x difference")


def portfolio_analysis():
    """Analyze a portfolio of buildings"""
    print("\n" + "="*60)
    print("PORTFOLIO ANALYSIS")
    print("="*60)
    
    client = GreenLangClient(region="US")
    
    # Define portfolio
    portfolio = [
        {
            "id": "NYC_Office_1",
            "metadata": {
                "building_type": "commercial_office",
                "area": 100000,
                "area_unit": "sqft",
                "location": {"country": "US", "city": "New York"},
                "occupancy": 500,
                "floor_count": 25,
                "building_age": 20
            },
            "energy_consumption": {
                "electricity": {"value": 3000000, "unit": "kWh"},
                "natural_gas": {"value": 60000, "unit": "therms"}
            }
        },
        {
            "id": "LA_Retail",
            "metadata": {
                "building_type": "retail",
                "area": 50000,
                "area_unit": "sqft",
                "location": {"country": "US", "city": "Los Angeles"},
                "occupancy": 100,
                "floor_count": 2,
                "building_age": 5
            },
            "energy_consumption": {
                "electricity": {"value": 800000, "unit": "kWh"},
                "natural_gas": {"value": 10000, "unit": "therms"}
            }
        },
        {
            "id": "Chicago_Warehouse",
            "metadata": {
                "building_type": "warehouse",
                "area": 200000,
                "area_unit": "sqft",
                "location": {"country": "US", "city": "Chicago"},
                "occupancy": 50,
                "floor_count": 1,
                "building_age": 15
            },
            "energy_consumption": {
                "electricity": {"value": 500000, "unit": "kWh"},
                "natural_gas": {"value": 20000, "unit": "therms"},
                "diesel": {"value": 5000, "unit": "gallons"}
            }
        }
    ]
    
    # Analyze portfolio
    results = client.analyze_portfolio(portfolio)
    
    if results["success"]:
        metrics = results["data"]["portfolio_metrics"]
        
        print(f"\nPortfolio Summary:")
        print(f"  Total Buildings: {metrics['total_buildings']}")
        print(f"  Total Area: {metrics['total_area_sqft']:,.0f} sqft")
        print(f"  Total Emissions: {metrics['total_emissions_tons']:.1f} metric tons CO2e")
        print(f"  Average Intensity: {metrics['average_intensity']:.2f} kgCO2e/sqft/year")
        
        print("\nBuilding Performance:")
        for building in results["data"]["buildings"]:
            emissions = building["analysis"]["emissions"]["total_co2e_tons"]
            rating = building["analysis"]["intensity"]["performance_rating"]
            print(f"  {building['building_id']}: {emissions:.1f} tons, Rating: {rating}")


def generate_detailed_report():
    """Generate a detailed report for a building"""
    print("\n" + "="*60)
    print("GENERATING DETAILED REPORT")
    print("="*60)
    
    client = GreenLangClient(region="US")
    
    # Analyze a building
    building = {
        "metadata": {
            "building_type": "hotel",
            "area": 80000,
            "area_unit": "sqft",
            "location": {
                "country": "US",
                "region": "Florida",
                "city": "Miami"
            },
            "occupancy": 200,
            "floor_count": 15,
            "building_age": 12,
            "climate_zone": "2A"
        },
        "energy_consumption": {
            "electricity": {"value": 2000000, "unit": "kWh"},
            "natural_gas": {"value": 40000, "unit": "therms"}
        }
    }
    
    # Get full analysis
    analysis = client.analyze_building(building)
    
    # Generate report
    report = client.generate_report(analysis, format="markdown")
    
    # Save report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"hotel_report_{timestamp}.md"
    
    if report["success"]:
        with open(report_file, 'w') as f:
            f.write(report["data"]["report"])
        print(f"\nDetailed report saved to: {report_file}")
    
    # Also export to Excel
    excel_file = f"hotel_analysis_{timestamp}.xlsx"
    client.export_analysis(analysis, excel_file, format="excel")
    print(f"Analysis exported to: {excel_file}")


def main():
    """Run all examples"""
    print("\n" + "#"*60)
    print("# GREENLANG GLOBAL COMMERCIAL BUILDING EMISSIONS SIMULATOR")
    print("#"*60)
    
    # Run different analyses
    analyze_us_office()
    analyze_india_hospital()
    analyze_eu_data_center()
    compare_global_offices()
    portfolio_analysis()
    
    # Generate report (commented out to avoid file creation)
    # generate_detailed_report()
    
    print("\n" + "#"*60)
    print("# ANALYSIS COMPLETE")
    print("#"*60)


if __name__ == "__main__":
    main()