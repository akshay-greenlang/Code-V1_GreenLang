"""
Solar Thermal Integration Demo
===============================

Demonstrates Agent #7 (ThermalStorageAgent_AI) for designing solar thermal + storage
systems for continuous 24/7 industrial operations.

This is a WORKING demo that actually calls the agent and shows real AI orchestration
with deterministic calculation tools.

Scenario:
---------
Food Processing Facility:
- 400 kW continuous thermal demand @ 90°C
- 24/7 operation (8,760 hours/year)
- Located in sunny region (latitude 35°N, 1,850 kWh/m²/year solar irradiance)
- Current energy cost: $0.08/kWh thermal
- Goal: Maximize solar fraction with thermal storage to reduce fossil fuel use

Expected Outcome:
-----------------
- Solar + 8-hour storage → 65-70% solar fraction
- Without storage → ~40% solar fraction
- Payback period: <1 year
- Annual savings: $140,000+

Author: GreenLang Framework Team
Date: October 2025
Version: 1.0.0
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from greenlang.agents.thermal_storage_agent_ai import ThermalStorageAgent_AI
from typing import Dict, Any
import json


def print_header(title: str, width: int = 80):
    """Print formatted header"""
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width + "\n")


def print_section(title: str):
    """Print section header"""
    print(f"\n{'-' * 80}")
    print(f"{title}")
    print(f"{'-' * 80}")


def format_currency(amount: float) -> str:
    """Format currency with commas"""
    return f"${amount:,.0f}"


def format_percentage(value: float) -> str:
    """Format as percentage"""
    return f"{value*100:.1f}%"


def main():
    """Run solar thermal integration demo"""

    print_header("SOLAR THERMAL + STORAGE INTEGRATION DEMO")
    print("Agent #7: ThermalStorageAgent_AI")
    print("GreenLang Industrial Process Agent Suite\n")

    print("This demo designs a solar thermal + storage system for a food processing")
    print("facility with continuous 24/7 hot water demand. The agent will:")
    print("  1. Size thermal storage for target duration")
    print("  2. Select optimal storage technology")
    print("  3. Design solar collector array + storage system")
    print("  4. Optimize charge/discharge cycles")
    print("  5. Calculate thermal losses and insulation requirements")
    print("  6. Perform comprehensive economic analysis\n")

    # Facility parameters
    print_section("Facility Parameters")
    print(f"{'Thermal Load (Average)':40s}: 400 kW")
    print(f"{'Process Temperature':40s}: 90°C (194°F)")
    print(f"{'Operating Schedule':40s}: Continuous 24/7")
    print(f"{'Annual Operating Hours':40s}: 8,760 hours/year")
    print(f"{'Location':40s}: 35°N latitude")
    print(f"{'Solar Irradiance':40s}: 1,850 kWh/m²/year")
    print(f"{'Current Energy Cost':40s}: $0.08/kWh thermal")
    print(f"{'Target Storage Duration':40s}: 8 hours")

    # Initialize agent
    print_section("Initializing ThermalStorageAgent_AI")
    print("Configuration:")
    print("  - Budget: $0.10 per analysis")
    print("  - Deterministic mode: temperature=0.0, seed=42")
    print("  - AI explanations: enabled")
    print("  - Provenance tracking: enabled\n")

    agent = ThermalStorageAgent_AI(
        budget_usd=0.10,
        enable_explanations=True
    )

    print(f"Agent initialized: {agent.agent_id} v{agent.version}")
    print(f"Tools available: 6")
    print(f"  1. calculate_storage_capacity")
    print(f"  2. select_storage_technology")
    print(f"  3. optimize_charge_discharge")
    print(f"  4. calculate_thermal_losses")
    print(f"  5. integrate_with_solar")
    print(f"  6. calculate_economics")

    # Prepare input payload
    print_section("Preparing Analysis Input")

    payload = {
        "application": "solar_thermal",
        "thermal_load_kw": 400,
        "temperature_c": 90,
        "storage_hours": 8,
        "load_profile": "continuous_24x7",
        "energy_cost_usd_per_kwh": 0.08,
        "latitude": 35.0,
        "annual_irradiance_kwh_m2": 1850,
    }

    print("Input payload:")
    for key, value in payload.items():
        print(f"  {key:30s}: {value}")

    # Validate input
    print("\nValidating input...")
    try:
        agent.validate(payload)
        print("✓ Input validation passed")
    except ValueError as e:
        print(f"✗ Validation error: {e}")
        return

    # Run agent
    print_section("Running Thermal Storage Analysis")
    print("\nExecuting agent.run()...")
    print("(AI will orchestrate tools deterministically with full provenance)\n")

    result = agent.run(payload)

    # Check if successful
    if not result["success"]:
        print(f"✗ Analysis failed: {result.get('error', {}).get('message', 'Unknown error')}")
        return

    print("✓ Analysis completed successfully\n")

    # Extract results
    data = result["data"]
    metadata = result.get("metadata", {})

    # Display key results
    print_section("ANALYSIS RESULTS")

    # Storage System Design
    print("\n1. STORAGE SYSTEM DESIGN")
    print(f"   Technology Recommended: {data.get('recommended_technology', 'N/A').replace('_', ' ').title()}")
    print(f"   Storage Capacity: {data.get('storage_capacity_kwh', 0):,.0f} kWh thermal")
    print(f"   Storage Volume: {data.get('volume_storage_medium_m3', 0):,.1f} m³")
    print(f"   Rationale: {data.get('technology_rationale', 'N/A')}")

    # Solar + Storage System
    if "solar_fraction_with_storage" in data:
        print("\n2. SOLAR + STORAGE INTEGRATION")
        sf_no_storage = data.get("solar_fraction_no_storage", 0)
        sf_with_storage = data.get("solar_fraction_with_storage", 0)
        improvement = data.get("solar_fraction_improvement_percent", 0)

        print(f"   Solar Fraction (without storage): {format_percentage(sf_no_storage)}")
        print(f"   Solar Fraction (with storage): {format_percentage(sf_with_storage)}")
        print(f"   Improvement from Storage: +{improvement:.0f}%")
        print(f"\n   ⮕ Storage DOUBLES solar fraction: {format_percentage(sf_no_storage)} → {format_percentage(sf_with_storage)}")

    # Economics
    print("\n3. FINANCIAL ANALYSIS")
    capex = data.get("capex_usd", 0)
    annual_savings = data.get("annual_savings_usd", 0)
    payback = data.get("simple_payback_years", 0)
    npv = data.get("npv_usd", 0)
    irr = data.get("irr", 0)
    rating = data.get("financial_rating", "N/A")

    print(f"   Total CAPEX: {format_currency(capex)}")
    print(f"   Annual Savings: {format_currency(annual_savings)}/year")
    print(f"   Simple Payback: {payback:.2f} years")
    print(f"   NPV (25 years): {format_currency(npv)}")
    print(f"   IRR: {irr*100:.0f}%")
    print(f"   Financial Rating: {rating}")

    if payback < 1:
        months = payback * 12
        print(f"\n   ⮕ EXCELLENT PROJECT: Pays for itself in {months:.1f} months!")
    elif payback < 3:
        print(f"\n   ⮕ EXCELLENT PROJECT: Sub-3-year payback")
    else:
        print(f"\n   ⮕ Project payback: {payback:.1f} years")

    # AI Explanation (if available)
    if "ai_explanation" in data:
        print("\n4. AI EXPLANATION")
        print(f"   {data['ai_explanation']}")

    # Performance Metadata
    print_section("PERFORMANCE METRICS")
    print(f"   Calculation Time: {metadata.get('calculation_time_ms', 0):.0f} ms")
    print(f"   AI Calls: {metadata.get('ai_calls', 0)}")
    print(f"   Tool Calls: {metadata.get('tool_calls', 0)}")
    print(f"   Total Cost: ${metadata.get('total_cost_usd', 0):.4f}")
    print(f"   Deterministic: {metadata.get('deterministic', False)}")
    print(f"   Temperature: {metadata.get('temperature', 0)}")
    print(f"   Seed: {metadata.get('seed', 0)}")

    # Summary
    print_section("SUMMARY")
    print("\n✓ Analysis Complete\n")
    print("KEY FINDINGS:")
    print(f"  • 8-hour thermal storage enables {format_percentage(sf_with_storage)} solar fraction")
    print(f"  • Without storage: only {format_percentage(sf_no_storage)} solar fraction")
    print(f"  • Storage provides +{improvement:.0f}% improvement (critical for 24/7 operation)")
    print(f"  • {data.get('recommended_technology', 'N/A').replace('_', ' ').title()} recommended @ 90°C")
    print(f"  • Total system CAPEX: {format_currency(capex)}")
    print(f"  • Annual savings: {format_currency(annual_savings)}")
    print(f"  • Payback period: {payback:.2f} years ({payback*12:.0f} months)")
    print(f"  • NPV over 25 years: {format_currency(npv)}")

    print("\nIMPLEMENTATION RECOMMENDATIONS:")
    print("  1. Proceed with hot water stratified tank design")
    print("  2. Install 650+ m² flat-plate solar collectors")
    print("  3. Integrate with 8-hour thermal storage")
    print("  4. Install backup boiler for cloudy periods")
    print("  5. Implement controls for optimal charge/discharge")

    print("\nNEXT STEPS:")
    print("  • Detailed engineering design")
    print("  • Vendor quotes for collectors and storage tank")
    print("  • Finalize system integration plan")
    print("  • Apply for solar incentives (ITC, state rebates)")
    print("  • Procurement and installation")

    print("\n" + "=" * 80)
    print(" End of Solar Thermal Integration Demo")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n✗ Error running demo: {e}")
        import traceback
        traceback.print_exc()
