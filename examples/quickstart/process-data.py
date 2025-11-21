#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GreenLang Data Processing Example

This example demonstrates a complete data processing pipeline for
analyzing multiple buildings in a portfolio. It showcases:

- Batch processing of multiple buildings
- Data validation and error handling
- Comprehensive reporting
- Optimization recommendations
- Export to multiple formats

Usage:
    python process-data.py

Expected runtime: 1-2 minutes
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from greenlang.determinism import DeterministicClock

# Add the parent directory to the path so we can import greenlang
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from greenlang.sdk import GreenLangClient
    from greenlang.models import BuildingData, FuelConsumption
    from greenlang.agents import BenchmarkAgent, OptimizationAgent
    from greenlang.utils import BatchProcessor, DataValidator
    from greenlang.reporting import ReportGenerator
except ImportError as e:
    print("❌ GreenLang import failed!")
    print(f"Error: {e}")
    print("\n💡 Try installing GreenLang with analytics:")
    print("   pip install greenlang-cli[analytics]==0.3.0")
    sys.exit(1)

class PortfolioAnalyzer:
    """
    Analyzes a portfolio of buildings for carbon emissions,
    performance benchmarking, and optimization opportunities.
    """

    def __init__(self):
        """Initialize the portfolio analyzer."""
        self.client = GreenLangClient()
        self.validator = DataValidator()
        self.batch_processor = BatchProcessor(batch_size=10)
        self.benchmark_agent = BenchmarkAgent()
        self.optimization_agent = OptimizationAgent()
        self.report_generator = ReportGenerator()

        self.results = {}
        self.portfolio_summary = {}

    def load_portfolio_data(self, data_file: str = "sample-portfolio.json") -> List[Dict]:
        """Load portfolio data from JSON file."""
        print(f"📂 Loading portfolio data from {data_file}...")

        data_path = Path(__file__).parent / data_file

        if not data_path.exists():
            print(f"❌ Data file not found: {data_path}")
            print("Creating sample data file...")
            self._create_sample_portfolio_data(data_path)

        try:
            with open(data_path, 'r') as f:
                portfolio_data = json.load(f)

            print(f"✅ Loaded {len(portfolio_data['buildings'])} buildings")
            return portfolio_data['buildings']

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return []

    def _create_sample_portfolio_data(self, file_path: Path):
        """Create sample portfolio data for demonstration."""
        sample_portfolio = {
            "portfolio_name": "Demo Building Portfolio",
            "description": "Sample data for GreenLang quickstart",
            "buildings": [
                {
                    "id": "building-001",
                    "name": "Corporate Headquarters",
                    "type": "commercial_office",
                    "area_m2": 5000,
                    "location": "San Francisco, CA",
                    "occupancy": 300,
                    "year_built": 2010,
                    "energy_consumption": [
                        {"fuel_type": "electricity", "consumption": 75000, "unit": "kWh", "period": "annual"},
                        {"fuel_type": "natural_gas", "consumption": 1500, "unit": "therms", "period": "annual"}
                    ]
                },
                {
                    "id": "building-002",
                    "name": "Retail Store Downtown",
                    "type": "retail",
                    "area_m2": 1200,
                    "location": "San Francisco, CA",
                    "occupancy": 25,
                    "year_built": 2005,
                    "energy_consumption": [
                        {"fuel_type": "electricity", "consumption": 45000, "unit": "kWh", "period": "annual"},
                        {"fuel_type": "natural_gas", "consumption": 300, "unit": "therms", "period": "annual"}
                    ]
                },
                {
                    "id": "building-003",
                    "name": "Warehouse Facility",
                    "type": "warehouse",
                    "area_m2": 8000,
                    "location": "Oakland, CA",
                    "occupancy": 50,
                    "year_built": 1995,
                    "energy_consumption": [
                        {"fuel_type": "electricity", "consumption": 120000, "unit": "kWh", "period": "annual"},
                        {"fuel_type": "natural_gas", "consumption": 800, "unit": "therms", "period": "annual"}
                    ]
                },
                {
                    "id": "building-004",
                    "name": "Medical Office",
                    "type": "healthcare",
                    "area_m2": 3000,
                    "location": "San Jose, CA",
                    "occupancy": 100,
                    "year_built": 2018,
                    "energy_consumption": [
                        {"fuel_type": "electricity", "consumption": 90000, "unit": "kWh", "period": "annual"},
                        {"fuel_type": "natural_gas", "consumption": 1200, "unit": "therms", "period": "annual"}
                    ]
                }
            ]
        }

        with open(file_path, 'w') as f:
            json.dump(sample_portfolio, f, indent=2)

        print(f"✅ Created sample portfolio data: {file_path}")

    def validate_data(self, buildings_data: List[Dict]) -> List[Dict]:
        """Validate building data before processing."""
        print("\n🔍 Validating portfolio data...")

        valid_buildings = []
        validation_errors = []

        for building_data in buildings_data:
            try:
                # Validate required fields
                required_fields = ['name', 'type', 'area_m2', 'location', 'energy_consumption']
                missing_fields = [field for field in required_fields if field not in building_data]

                if missing_fields:
                    validation_errors.append(f"Building {building_data.get('id', 'unknown')}: Missing fields {missing_fields}")
                    continue

                # Validate energy consumption data
                if not building_data['energy_consumption']:
                    validation_errors.append(f"Building {building_data['id']}: No energy consumption data")
                    continue

                # Additional validation
                if building_data['area_m2'] <= 0:
                    validation_errors.append(f"Building {building_data['id']}: Invalid area")
                    continue

                valid_buildings.append(building_data)

            except Exception as e:
                validation_errors.append(f"Building {building_data.get('id', 'unknown')}: Validation error - {e}")

        if validation_errors:
            print("⚠️  Validation warnings:")
            for error in validation_errors:
                print(f"   {error}")

        print(f"✅ Validated {len(valid_buildings)} of {len(buildings_data)} buildings")
        return valid_buildings

    def analyze_portfolio(self, buildings_data: List[Dict]) -> Dict:
        """Analyze the entire portfolio."""
        print("\n🧮 Analyzing portfolio emissions...")

        portfolio_results = {
            "timestamp": DeterministicClock.now().isoformat(),
            "total_buildings": len(buildings_data),
            "successful_calculations": 0,
            "total_emissions_tons": 0,
            "total_area_m2": 0,
            "buildings": {},
            "portfolio_summary": {},
            "recommendations": []
        }

        # Process each building
        for building_data in buildings_data:
            building_id = building_data.get('id', building_data['name'])
            print(f"   Processing: {building_data['name']}...")

            try:
                # Create building model
                building = BuildingData(
                    name=building_data['name'],
                    building_type=building_data['type'],
                    area_m2=building_data['area_m2'],
                    location=building_data['location'],
                    occupancy=building_data.get('occupancy', 0),
                    year_built=building_data.get('year_built', 2000)
                )

                # Create energy consumption models
                energy_data = []
                for fuel_data in building_data['energy_consumption']:
                    energy_data.append(FuelConsumption(
                        fuel_type=fuel_data['fuel_type'],
                        consumption=fuel_data['consumption'],
                        unit=fuel_data['unit'],
                        period=fuel_data.get('period', 'annual')
                    ))

                # Calculate emissions
                result = self.client.calculate_building_emissions(
                    building=building,
                    energy_consumption=energy_data,
                    include_scope3=False
                )

                if result.success:
                    # Store building results
                    portfolio_results["buildings"][building_id] = {
                        "name": building.name,
                        "type": building.building_type,
                        "area_m2": building.area_m2,
                        "location": building.location,
                        "total_emissions_tons": result.total_emissions_tons,
                        "intensity_per_sqft": result.intensity_per_sqft,
                        "breakdown": result.breakdown.__dict__ if hasattr(result, 'breakdown') else {},
                        "calculation_status": "success"
                    }

                    # Add to portfolio totals
                    portfolio_results["total_emissions_tons"] += result.total_emissions_tons
                    portfolio_results["total_area_m2"] += building.area_m2
                    portfolio_results["successful_calculations"] += 1

                    # Get benchmarking data
                    if hasattr(result, 'benchmark') and result.benchmark:
                        portfolio_results["buildings"][building_id]["benchmark"] = {
                            "rating": result.benchmark.rating,
                            "percentile": result.benchmark.percentile
                        }

                else:
                    print(f"      ❌ Calculation failed: {result.errors}")
                    portfolio_results["buildings"][building_id] = {
                        "name": building.name,
                        "calculation_status": "failed",
                        "errors": result.errors
                    }

            except Exception as e:
                print(f"      ❌ Processing error: {e}")
                portfolio_results["buildings"][building_id] = {
                    "name": building_data['name'],
                    "calculation_status": "error",
                    "errors": str(e)
                }

        # Calculate portfolio metrics
        if portfolio_results["total_area_m2"] > 0:
            portfolio_results["portfolio_summary"] = {
                "average_intensity_per_sqft": portfolio_results["total_emissions_tons"] * 1000 / (portfolio_results["total_area_m2"] * 10.764),
                "total_emissions_tons": portfolio_results["total_emissions_tons"],
                "total_area_sqft": portfolio_results["total_area_m2"] * 10.764,
                "success_rate": portfolio_results["successful_calculations"] / portfolio_results["total_buildings"]
            }

        print(f"✅ Portfolio analysis completed")
        print(f"   Successful calculations: {portfolio_results['successful_calculations']}/{portfolio_results['total_buildings']}")
        print(f"   Total emissions: {portfolio_results['total_emissions_tons']:.2f} tCO2e")

        return portfolio_results

    def generate_optimization_recommendations(self, portfolio_results: Dict) -> List[Dict]:
        """Generate optimization recommendations for the portfolio."""
        print("\n💡 Generating optimization recommendations...")

        recommendations = []

        try:
            for building_id, building_data in portfolio_results["buildings"].items():
                if building_data.get("calculation_status") == "success":
                    # Generate building-specific recommendations
                    building_recommendations = self.optimization_agent.get_recommendations(
                        building_type=building_data["type"],
                        emissions_tons=building_data["total_emissions_tons"],
                        area_m2=building_data["area_m2"],
                        intensity=building_data["intensity_per_sqft"]
                    )

                    if building_recommendations:
                        recommendations.extend([
                            {
                                "building_id": building_id,
                                "building_name": building_data["name"],
                                **rec
                            }
                            for rec in building_recommendations
                        ])

            print(f"✅ Generated {len(recommendations)} recommendations")

        except Exception as e:
            print(f"⚠️  Could not generate detailed recommendations: {e}")
            # Provide basic recommendations
            recommendations = [
                {
                    "category": "Energy Efficiency",
                    "title": "LED Lighting Upgrade",
                    "description": "Replace existing lighting with LED fixtures",
                    "estimated_savings_percent": 15,
                    "payback_years": 2.5
                },
                {
                    "category": "HVAC Optimization",
                    "title": "Smart Thermostat Installation",
                    "description": "Install programmable thermostats with occupancy sensors",
                    "estimated_savings_percent": 10,
                    "payback_years": 1.8
                }
            ]

        return recommendations

    def export_results(self, portfolio_results: Dict, recommendations: List[Dict]):
        """Export results to various formats."""
        print("\n📊 Exporting results...")

        timestamp = DeterministicClock.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parent / "results"
        output_dir.mkdir(exist_ok=True)

        # 1. Export detailed JSON
        json_file = output_dir / f"portfolio_analysis_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump({
                "portfolio_results": portfolio_results,
                "recommendations": recommendations
            }, f, indent=2)
        print(f"   📄 JSON report: {json_file}")

        # 2. Export summary CSV
        csv_file = output_dir / f"portfolio_summary_{timestamp}.csv"
        try:
            import csv
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Building ID', 'Name', 'Type', 'Area (m²)', 'Emissions (tCO2e)', 'Intensity (kgCO2e/sqft)'])

                for building_id, data in portfolio_results["buildings"].items():
                    if data.get("calculation_status") == "success":
                        writer.writerow([
                            building_id,
                            data["name"],
                            data["type"],
                            data["area_m2"],
                            data["total_emissions_tons"],
                            data["intensity_per_sqft"]
                        ])

            print(f"   📊 CSV summary: {csv_file}")

        except Exception as e:
            print(f"   ⚠️  Could not export CSV: {e}")

        # 3. Export executive summary
        summary_file = output_dir / f"executive_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("PORTFOLIO CARBON ANALYSIS - EXECUTIVE SUMMARY\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Analysis Date: {DeterministicClock.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Buildings Analyzed: {portfolio_results['total_buildings']}\n")
            f.write(f"Successful Calculations: {portfolio_results['successful_calculations']}\n")
            f.write(f"Total Portfolio Emissions: {portfolio_results['total_emissions_tons']:.2f} tCO2e\n")

            if portfolio_results['portfolio_summary']:
                f.write(f"Average Intensity: {portfolio_results['portfolio_summary']['average_intensity_per_sqft']:.3f} kgCO2e/sqft\n")

            f.write(f"\nTOP RECOMMENDATIONS:\n")
            for i, rec in enumerate(recommendations[:5], 1):
                f.write(f"{i}. {rec.get('title', 'Energy Efficiency Improvement')}\n")

        print(f"   📋 Executive summary: {summary_file}")

    def display_results(self, portfolio_results: Dict, recommendations: List[Dict]):
        """Display formatted results."""
        print("\n" + "=" * 60)
        print("📊 PORTFOLIO ANALYSIS RESULTS")
        print("=" * 60)

        # Portfolio summary
        print(f"\n🏢 Portfolio Overview:")
        print(f"   Total Buildings: {portfolio_results['total_buildings']}")
        print(f"   Successful Calculations: {portfolio_results['successful_calculations']}")
        print(f"   Total Emissions: {portfolio_results['total_emissions_tons']:.2f} metric tons CO2e")

        if portfolio_results['portfolio_summary']:
            summary = portfolio_results['portfolio_summary']
            print(f"   Total Area: {summary['total_area_sqft']:,.0f} sq ft")
            print(f"   Average Intensity: {summary['average_intensity_per_sqft']:.3f} kgCO2e/sqft")
            print(f"   Success Rate: {summary['success_rate']:.1%}")

        # Building-by-building results
        print(f"\n🏗️  Individual Building Results:")
        for building_id, data in portfolio_results["buildings"].items():
            if data.get("calculation_status") == "success":
                print(f"   📍 {data['name']} ({data['type']})")
                print(f"      Emissions: {data['total_emissions_tons']:.2f} tCO2e")
                print(f"      Intensity: {data['intensity_per_sqft']:.3f} kgCO2e/sqft")
                if 'benchmark' in data:
                    print(f"      Performance: {data['benchmark']['rating']} ({data['benchmark']['percentile']}th percentile)")
            else:
                print(f"   ❌ {data['name']}: {data.get('errors', 'Unknown error')}")

        # Top recommendations
        print(f"\n💡 Top Optimization Recommendations:")
        for i, rec in enumerate(recommendations[:5], 1):
            print(f"   {i}. {rec.get('title', 'Energy Efficiency Improvement')}")
            if 'estimated_savings_percent' in rec:
                print(f"      Potential savings: {rec['estimated_savings_percent']}%")
            if 'payback_years' in rec:
                print(f"      Payback period: {rec['payback_years']} years")

        # Environmental impact
        total_emissions = portfolio_results['total_emissions_tons']
        print(f"\n🌱 Environmental Impact:")
        print(f"   Equivalent to {total_emissions / 4.6:.1f} cars driven for a year")
        print(f"   Would require {total_emissions * 16:.0f} tree seedlings grown for 10 years to offset")

def main():
    """Run the complete portfolio analysis."""
    print("🌍 GreenLang Data Processing Example")
    print("=" * 60)
    print("This example analyzes a portfolio of buildings for carbon emissions,")
    print("performance benchmarking, and optimization opportunities.\n")

    try:
        # Initialize analyzer
        analyzer = PortfolioAnalyzer()

        # Load and validate data
        buildings_data = analyzer.load_portfolio_data()
        if not buildings_data:
            print("❌ No building data available")
            return False

        valid_buildings = analyzer.validate_data(buildings_data)
        if not valid_buildings:
            print("❌ No valid buildings to process")
            return False

        # Analyze portfolio
        portfolio_results = analyzer.analyze_portfolio(valid_buildings)

        # Generate recommendations
        recommendations = analyzer.generate_optimization_recommendations(portfolio_results)

        # Display results
        analyzer.display_results(portfolio_results, recommendations)

        # Export results
        analyzer.export_results(portfolio_results, recommendations)

        print("\n✨ Analysis complete! Check the 'results' directory for exported files.")
        return True

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return False

if __name__ == "__main__":
    success = main()

    if success:
        print("\n🎉 Data processing example completed successfully!")
        print("\n📚 What's next?")
        print("   • Explore the generated reports in the 'results' directory")
        print("   • Modify sample-portfolio.json with your own building data")
        print("   • Try the advanced examples in ../tutorials/")
        print("   • Set up real-time monitoring for your buildings")
    else:
        print("\n❌ Example failed. Please check your GreenLang installation.")
        sys.exit(1)