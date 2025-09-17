#!/usr/bin/env python3
"""Test the boiler agent with correct data format"""

from greenlang.sdk import GreenLangClient

def test_boiler_agent():
    """Test boiler agent with proper test data"""
    
    print("Testing BoilerAgent with correct data format...")
    print("=" * 60)
    
    client = GreenLangClient()
    
    # Test data for boiler agent
    test_data = {
        "boiler_type": "standard",
        "fuel_type": "natural_gas", 
        "thermal_output": {
            "value": 1000,
            "unit": "kWh"
        },
        "efficiency": 0.85,
        "country": "US"
    }
    
    print("Test data:")
    import json
    print(json.dumps(test_data, indent=2))
    print()
    
    # Execute agent
    result = client.execute_agent("boiler", test_data)
    
    print("Result:")
    if result.get("success"):
        print("[SUCCESS] Boiler agent executed successfully!")
        print("\nOutput:")
        print(json.dumps(result.get("data", {}), indent=2))
    else:
        print(f"[FAILED] Boiler agent failed!")
        print(f"Error: {result.get('error')}")
    
    return result.get("success", False)

if __name__ == "__main__":
    import sys
    success = test_boiler_agent()
    sys.exit(0 if success else 1)