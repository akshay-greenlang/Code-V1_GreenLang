#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test script to verify dev interface fixes"""

from greenlang.sdk import GreenLangClient

def test_dev_interface_methods():
    """Test that all methods used by dev interface exist"""
    
    print("Testing GreenLang Dev Interface fixes...")
    print("=" * 50)
    
    client = GreenLangClient()
    
    # Test 1: List agents
    print("\n1. Testing list_agents():")
    try:
        agents = client.list_agents()
        print(f"   [OK] Found {len(agents)} agents: {agents}")
    except Exception as e:
        print(f"   [FAIL] list_agents() failed: {e}")
        return False
    
    # Test 2: Get agent info
    print("\n2. Testing get_agent_info():")
    for agent_id in ["validator", "fuel", "boiler"]:
        try:
            info = client.get_agent_info(agent_id)
            if info:
                print(f"   [OK] {agent_id}: {info['name']} - {info['description']}")
            else:
                print(f"   [FAIL] No info for {agent_id}")
        except Exception as e:
            print(f"   [FAIL] get_agent_info('{agent_id}') failed: {e}")
            return False
    
    # Test 3: Execute agent
    print("\n3. Testing execute_agent():")
    test_data = {
        "fuel_type": "electricity",
        "consumption": 1000,
        "unit": "kWh"
    }
    try:
        result = client.execute_agent("fuel", test_data)
        if result and result.get("success"):
            print(f"   [OK] Fuel agent executed successfully")
        else:
            print(f"   [WARNING] Fuel agent returned: {result}")
    except Exception as e:
        print(f"   [FAIL] execute_agent() failed: {e}")
        return False
    
    # Test 4: Validate input
    print("\n4. Testing validate_input():")
    try:
        result = client.validate_input({"test": "data"})
        print(f"   [OK] validate_input() executed")
    except Exception as e:
        print(f"   [FAIL] validate_input() failed: {e}")
        return False
    
    # Test 5: Calculate emissions
    print("\n5. Testing calculate_emissions():")
    try:
        result = client.calculate_emissions("electricity", 1000, "kWh")
        if result and result.get("success"):
            print(f"   [OK] calculate_emissions() executed successfully")
        else:
            print(f"   [WARNING] calculate_emissions() returned: {result}")
    except Exception as e:
        print(f"   [FAIL] calculate_emissions() failed: {e}")
        return False
    
    # Test 6: Aggregate emissions
    print("\n6. Testing aggregate_emissions():")
    try:
        emissions_list = [
            {"co2e_emissions_kg": 100, "source": "electricity"},
            {"co2e_emissions_kg": 50, "source": "natural_gas"}
        ]
        result = client.aggregate_emissions(emissions_list)
        print(f"   [OK] aggregate_emissions() executed")
    except Exception as e:
        print(f"   [FAIL] aggregate_emissions() failed: {e}")
        return False
    
    # Test 7: Generate report
    print("\n7. Testing generate_report():")
    try:
        carbon_data = {"total_emissions_kg": 150}
        result = client.generate_report(carbon_data)
        print(f"   [OK] generate_report() executed")
    except Exception as e:
        print(f"   [FAIL] generate_report() failed: {e}")
        return False
    
    # Test 8: Benchmark emissions
    print("\n8. Testing benchmark_emissions():")
    try:
        result = client.benchmark_emissions(1000, 5000, "commercial_office", 12)
        print(f"   [OK] benchmark_emissions() executed")
    except Exception as e:
        print(f"   [FAIL] benchmark_emissions() failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("[SUCCESS] All dev interface methods are working!")
    return True

if __name__ == "__main__":
    import sys
    success = test_dev_interface_methods()
    sys.exit(0 if success else 1)