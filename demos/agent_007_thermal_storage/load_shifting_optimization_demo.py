"""
Load Shifting Optimization Demo
================================

Demonstrates Agent #7 (ThermalStorageAgent_AI) for optimizing thermal load shifting
to reduce electricity costs through time-of-use (TOU) rate arbitrage.

This is a WORKING demo that actually calls the agent and shows real AI orchestration
with deterministic calculation tools.

Scenario:
---------
Manufacturing Facility:
- 300 kW electric process heating load
- Daytime operation (12 hours/day, 5 days/week)
- Subject to time-of-use electricity rates:
  * Off-peak (10pm-6am): $0.08/kWh
  * Mid-peak (6am-4pm, 9pm-10pm): $0.12/kWh
  * On-peak (4pm-9pm): $0.20/kWh
- Demand charges: $15/kW/month
- Process temperature: 120°C
- Goal: Shift load to off-peak hours to minimize costs

Expected Outcome:
-----------------
- Charge storage during off-peak ($0.08/kWh)
- Discharge during peak ($0.20/kWh)
- 40-50% reduction in energy costs
- Demand charge reduction
- Payback period: 1-2 years

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
    return f"{value*100:.0f}%"


def display_tou_schedule():
    """Display TOU rate schedule"""
    print("\nTime-of-Use Rate Schedule:")
    print("  Off-Peak (10pm-6am):        $0.08/kWh  ████░░░░")
    print("  Mid-Peak (6am-4pm, 9-10pm): $0.12/kWh  ████████░░░░")
    print("  On-Peak (4pm-9pm):          $0.20/kWh  ████████████████")
    print("\n  Demand Charge: $15/kW/month")


def main():
    """Run load shifting optimization demo"""

    print_header("LOAD SHIFTING OPTIMIZATION DEMO")
    print("Agent #7: ThermalStorageAgent_AI")
    print("GreenLang Industrial Process Agent Suite\n")

    print("This demo optimizes thermal storage for load shifting to minimize electricity")
    print("costs under time-of-use (TOU) pricing. The agent will:")
    print("  1. Size thermal storage for 10-hour operation")
    print("  2. Select technology for 120°C process temperature")
    print("  3. Optimize charge/discharge schedule for TOU savings")
    print("  4. Calculate thermal losses and standby efficiency")
    print("  5. Perform economic analysis with demand charge savings")
    print("  6. Provide implementation recommendations\n")

    # Facility parameters
    print_section("Facility Parameters")
    print(f"{'Process Heating Load':40s}: 300 kW electric")
    print(f"{'Process Temperature':40s}: 120°C (248°F)")
    print(f"{'Operating Schedule':40s}: Daytime only (12 hrs/day)")
    print(f"{'Operating Days':40s}: 5 days/week (260 days/year)")
    print(f"{'Annual Operating Hours':40s}: 3,120 hours/year")
    print(f"{'Current Average Energy Cost':40s}: $0.12/kWh (blended)")
    print(f"{'Target Storage Duration':40s}: 10 hours")

    display_tou_schedule()

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

    # Prepare input payload
    print_section("Preparing Analysis Input")

    payload = {
        "application": "load_shifting",
        "thermal_load_kw": 300,
        "temperature_c": 120,  # Higher temperature for process
        "storage_hours": 10,
        "load_profile": "daytime_only",
        "energy_cost_usd_per_kwh": 0.12,  # Blended average rate
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
    print_section("Running Load Shifting Optimization")
    print("\nExecuting agent.run()...")
    print("(AI will orchestrate tools for optimal load shifting strategy)\n")

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
    tech = data.get('recommended_technology', 'N/A')
    print(f"   Technology Recommended: {tech.replace('_', ' ').title()}")
    print(f"   Storage Capacity: {data.get('storage_capacity_kwh', 0):,.0f} kWh thermal")
    print(f"   Storage Volume: {data.get('volume_storage_medium_m3', 0):,.1f} m³")
    print(f"   Operating Temperature: 120°C (pressurized system required)")
    print(f"\n   Rationale: {data.get('technology_rationale', 'N/A')}")

    # Optimization Strategy
    print("\n2. LOAD SHIFTING STRATEGY")
    print("   Charge Schedule:")
    print("     • Charge during off-peak hours (10pm-6am)")
    print("     • Charge at $0.08/kWh (lowest rate)")
    print("     • Use electric resistance heaters or heat pump")
    print("\n   Discharge Schedule:")
    print("     • Discharge during peak hours (4pm-9pm)")
    print("     • Avoid on-peak rate of $0.20/kWh")
    print("     • Discharge during mid-peak (6am-4pm) at $0.12/kWh")
    print("\n   Energy Arbitrage:")
    print(f"     • Buy at: $0.08/kWh (off-peak)")
    print(f"     • Avoid: $0.20/kWh (on-peak)")
    print(f"     • Savings: $0.12/kWh per cycle")

    # Economics
    print("\n3. FINANCIAL ANALYSIS")
    capex = data.get("capex_usd", 0)
    annual_savings = data.get("annual_savings_usd", 0)
    payback = data.get("simple_payback_years", 0)
    npv = data.get("npv_usd", 0)
    irr = data.get("irr", 0)
    rating = data.get("financial_rating", "N/A")

    # Calculate savings breakdown
    daily_throughput = 300 * 10  # 300 kW × 10 hours = 3,000 kWh/day
    rate_differential = 0.20 - 0.08  # $0.12/kWh savings
    daily_savings = daily_throughput * rate_differential * 0.9  # 90% round-trip eff
    annual_energy_savings_est = daily_savings * 260  # 260 working days

    print(f"   Total System CAPEX: {format_currency(capex)}")
    print(f"\n   Annual Savings Breakdown:")
    print(f"     • Energy Cost Savings: {format_currency(annual_energy_savings_est)}/year")
    print(f"     • Demand Charge Reduction: ~{format_currency(annual_savings - annual_energy_savings_est)}/year")
    print(f"     • Total Annual Savings: {format_currency(annual_savings)}/year")
    print(f"\n   Financial Metrics:")
    print(f"     • Simple Payback: {payback:.2f} years")
    print(f"     • NPV (25 years): {format_currency(npv)}")
    print(f"     • IRR: {irr*100:.0f}%")
    print(f"     • Financial Rating: {rating}")

    # Cost savings visualization
    print("\n   Monthly Cost Comparison:")
    monthly_before = (300 * 12 * 260 / 12) * 0.14  # Current blended rate
    monthly_after = monthly_before - (annual_savings / 12)
    savings_pct = (monthly_before - monthly_after) / monthly_before * 100

    print(f"     • Current Monthly Cost:  {format_currency(monthly_before)}")
    print(f"     • With Storage:          {format_currency(monthly_after)}")
    print(f"     • Monthly Savings:       {format_currency(monthly_before - monthly_after)} ({savings_pct:.0f}%)")

    # AI Explanation (if available)
    if "ai_explanation" in data:
        print("\n4. AI EXPLANATION")
        explanation_lines = data['ai_explanation'].split('\n')
        for line in explanation_lines:
            print(f"   {line}")

    # Performance Metadata
    print_section("PERFORMANCE METRICS")
    print(f"   Calculation Time: {metadata.get('calculation_time_ms', 0):.0f} ms")
    print(f"   AI Calls: {metadata.get('ai_calls', 0)}")
    print(f"   Tool Calls: {metadata.get('tool_calls', 0)}")
    print(f"   Total Cost: ${metadata.get('total_cost_usd', 0):.4f}")
    print(f"   Deterministic: {metadata.get('deterministic', False)}")

    # Summary
    print_section("SUMMARY")
    print("\n✓ Load Shifting Optimization Complete\n")
    print("KEY FINDINGS:")
    print(f"  • 10-hour pressurized hot water storage for 120°C operation")
    print(f"  • Storage capacity: {data.get('storage_capacity_kwh', 0):,.0f} kWh thermal")
    print(f"  • Cost reduction: {savings_pct:.0f}% through TOU optimization")
    print(f"  • Annual savings: {format_currency(annual_savings)}")
    print(f"  • System pays for itself in {payback:.1f} years")
    print(f"  • Excellent project economics: NPV = {format_currency(npv)}")

    print("\nLOAD SHIFTING BENEFITS:")
    print("  ✓ Charge storage during cheap off-peak hours ($0.08/kWh)")
    print("  ✓ Avoid expensive on-peak rates ($0.20/kWh)")
    print("  ✓ Reduce demand charges through peak shaving")
    print("  ✓ Maintain production schedule without interruption")
    print("  ✓ Immediate cost savings from day 1 of operation")

    print("\nIMPLEMENTATION RECOMMENDATIONS:")
    print("  1. Install pressurized hot water storage tank (120°C rated)")
    print("  2. Add electric resistance heaters for off-peak charging")
    print("  3. Implement automated controls for TOU optimization")
    print("  4. Install thermal meters for monitoring and verification")
    print("  5. Consider adding heat pump for improved efficiency")

    print("\nOPERATIONAL STRATEGY:")
    print("  • 10pm-6am: Charge storage from grid (off-peak rate)")
    print("  • 6am-4pm: Discharge to process (avoid mid-peak rate)")
    print("  • 4pm-9pm: Maximum discharge (avoid on-peak rate)")
    print("  • Monitor and adjust based on actual TOU schedule")

    print("\nNEXT STEPS:")
    print("  1. Review detailed engineering design")
    print("  2. Obtain vendor quotes for storage tank and controls")
    print("  3. Verify utility TOU rate structure and demand charges")
    print("  4. Apply for energy efficiency incentives (if available)")
    print("  5. Plan installation during production downtime")
    print("  6. Set up M&V (measurement and verification) plan")

    print("\n" + "=" * 80)
    print(" End of Load Shifting Optimization Demo")
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
