#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple test script to validate AI agents work correctly.

This runs outside pytest to avoid conftest network blocking issues.
"""

import sys
import time
from greenlang.agents.fuel_agent_ai import FuelAgentAI
from greenlang.agents.carbon_agent_ai import CarbonAgentAI
from greenlang.agents.grid_factor_agent_ai import GridFactorAgentAI
from greenlang.agents.recommendation_agent_ai import RecommendationAgentAI
from greenlang.agents.report_agent_ai import ReportAgentAI


def test_fuel_agent():
    """Test FuelAgentAI"""
    print("\n=== Testing FuelAgentAI ===")

    agent = FuelAgentAI(budget_usd=0.50)
    input_data = {
        "fuel_type": "natural_gas",
        "amount": 1000,
        "unit": "therms",
        "country": "US",
    }

    result = agent.run(input_data)

    print(f"Success: {result['success']}")
    if result['success']:
        print(f"Emissions: {result['data']['co2e_emissions_kg']} kg CO2e")
        print(f"Emission Factor: {result['data']['emission_factor']}")
    else:
        print(f"Error: {result['error']}")

    assert result['success'], "Fuel agent should succeed"
    assert result['data']['co2e_emissions_kg'] > 0
    print("✓ FuelAgentAI test passed")

    return result


def test_carbon_agent():
    """Test CarbonAgentAI"""
    print("\n=== Testing CarbonAgentAI ===")

    agent = CarbonAgentAI(budget_usd=0.50)
    input_data = {
        "emissions": [
            {"fuel_type": "electricity", "co2e_emissions_kg": 15000},
            {"fuel_type": "natural_gas", "co2e_emissions_kg": 5300},
        ],
        "building_area": 50000,
        "occupancy": 200,
    }

    result = agent.execute(input_data)

    print(f"Success: {result.success}")
    if result.success:
        print(f"Total: {result.data['total_co2e_tons']:.2f} tons CO2e")
        print(f"Breakdown: {len(result.data['emissions_breakdown'])} sources")
    else:
        print(f"Error: {result.error}")

    assert result.success, "Carbon agent should succeed"
    assert result.data['total_co2e_tons'] > 0
    print("✓ CarbonAgentAI test passed")

    return result


def test_grid_agent():
    """Test GridFactorAgentAI"""
    print("\n=== Testing GridFactorAgentAI ===")

    agent = GridFactorAgentAI(budget_usd=0.50)
    input_data = {
        "country": "US",
        "fuel_type": "electricity",
        "unit": "kWh",
    }

    result = agent.run(input_data)

    print(f"Success: {result['success']}")
    if result['success']:
        print(f"Emission Factor: {result['data']['emission_factor']}")
        print(f"Country: {result['data']['country']}")
    else:
        print(f"Error: {result['error']}")

    assert result['success'], "Grid agent should succeed"
    assert result['data']['emission_factor'] > 0
    print("✓ GridFactorAgentAI test passed")

    return result


def test_recommendation_agent():
    """Test RecommendationAgentAI"""
    print("\n=== Testing RecommendationAgentAI ===")

    agent = RecommendationAgentAI(budget_usd=0.50, max_recommendations=5)
    input_data = {
        "emissions_by_source": {
            "electricity": 20000,
            "natural_gas": 8000,
        },
        "building_type": "commercial_office",
        "building_age": 20,
        "performance_rating": "Below Average",
        "country": "US",
    }

    result = agent.execute(input_data)

    print(f"Success: {result.success}")
    if result.success:
        recs = result.data.get('recommendations', [])
        print(f"Recommendations: {len(recs)}")
        if recs:
            print(f"Top recommendation: {recs[0].get('action', 'N/A')}")
    else:
        print(f"Error: {result.error}")

    assert result.success, "Recommendation agent should succeed"
    print("✓ RecommendationAgentAI test passed")

    return result


def test_report_agent():
    """Test ReportAgentAI"""
    print("\n=== Testing ReportAgentAI ===")

    agent = ReportAgentAI(budget_usd=1.00)
    carbon_data = {
        "total_co2e_kg": 28000,
        "total_co2e_tons": 28.0,
        "emissions_breakdown": [
            {"source": "electricity", "co2e_kg": 20000, "co2e_tons": 20.0, "percentage": 71.4},
            {"source": "natural_gas", "co2e_kg": 8000, "co2e_tons": 8.0, "percentage": 28.6},
        ],
    }

    input_data = {
        "framework": "TCFD",
        "carbon_data": carbon_data,
        "building_info": {"type": "commercial_office", "area": 50000, "occupancy": 200},
        "period": {"start_date": "2024-01-01", "end_date": "2024-12-31"},
    }

    result = agent.execute(input_data)

    print(f"Success: {result.success}")
    if result.success:
        print(f"Framework: {result.data.get('framework')}")
        print(f"Total Emissions: {result.data.get('total_co2e_tons')} tons")
        print(f"Report length: {len(result.data.get('report', ''))} chars")
    else:
        print(f"Error: {result.error}")

    assert result.success, "Report agent should succeed"
    assert result.data.get('framework') == "TCFD"
    print("✓ ReportAgentAI test passed")

    return result


def test_complete_workflow():
    """Test complete workflow"""
    print("\n=== Testing Complete Workflow ===")

    start_time = time.time()

    # 1. Calculate fuel emissions
    fuel_agent = FuelAgentAI(budget_usd=0.50)
    fuel_result = fuel_agent.run({
        "fuel_type": "natural_gas",
        "amount": 1000,
        "unit": "therms",
        "country": "US",
    })

    assert fuel_result['success']
    fuel_emissions = fuel_result['data']['co2e_emissions_kg']

    # 2. Aggregate emissions
    carbon_agent = CarbonAgentAI(budget_usd=0.50)
    carbon_result = carbon_agent.execute({
        "emissions": [
            {"fuel_type": "natural_gas", "co2e_emissions_kg": fuel_emissions},
            {"fuel_type": "electricity", "co2e_emissions_kg": 15000},
        ],
        "building_area": 50000,
        "occupancy": 200,
    })

    assert carbon_result.success
    total_emissions = carbon_result.data['total_co2e_kg']

    # 3. Generate report
    report_agent = ReportAgentAI(budget_usd=1.00)
    report_result = report_agent.execute({
        "framework": "TCFD",
        "carbon_data": carbon_result.data,
        "period": {"start_date": "2024-01-01", "end_date": "2024-12-31"},
    })

    assert report_result.success

    duration = time.time() - start_time

    print(f"✓ Complete workflow passed in {duration:.2f}s")
    print(f"  - Fuel emissions: {fuel_emissions:.0f} kg")
    print(f"  - Total emissions: {total_emissions:.0f} kg")
    print(f"  - Report framework: {report_result.data['framework']}")

    return {
        "fuel": fuel_result,
        "carbon": carbon_result,
        "report": report_result,
        "duration": duration,
    }


def main():
    """Run all tests"""
    print("=" * 70)
    print("AI AGENTS INTEGRATION TEST SUITE")
    print("=" * 70)

    try:
        # Run individual agent tests
        test_fuel_agent()
        test_carbon_agent()
        test_grid_agent()
        test_recommendation_agent()
        test_report_agent()

        # Run complete workflow
        results = test_complete_workflow()

        print("\n" + "=" * 70)
        print("ALL TESTS PASSED ✓")
        print("=" * 70)
        print(f"\nTotal workflow time: {results['duration']:.2f}s")

        return 0

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
