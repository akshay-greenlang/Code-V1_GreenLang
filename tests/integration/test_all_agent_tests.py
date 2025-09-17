#!/usr/bin/env python3
"""Test all agents with proper test data"""

import json
from greenlang.cli.dev_interface import GreenLangDevInterface

def test_all_agents():
    """Test all agents with the new test data"""
    
    print("Testing all agents with proper test data...")
    print("=" * 60)
    
    # Create dev interface instance
    dev = GreenLangDevInterface()
    
    # List of all agents to test
    agents = [
        "validator",
        "fuel", 
        "boiler",
        "carbon",
        "report",
        "benchmark",
        "grid_factor",
        "building_profile",
        "intensity",
        "recommendation"
    ]
    
    results = {}
    
    for agent_id in agents:
        print(f"\nTesting {agent_id}...")
        print("-" * 40)
        
        # Get test data
        test_data = dev._get_agent_test_data(agent_id)
        
        if not test_data:
            print(f"  [SKIP] No test data for {agent_id}")
            results[agent_id] = "SKIPPED"
            continue
        
        print(f"  Test data: {json.dumps(test_data, indent=2)[:200]}...")
        
        try:
            # Execute agent
            result = dev.client.execute_agent(agent_id, test_data)
            
            if result.get("success"):
                print(f"  [SUCCESS] {agent_id} executed successfully")
                if "data" in result:
                    print(f"  Result preview: {str(result['data'])[:100]}...")
                results[agent_id] = "SUCCESS"
            else:
                print(f"  [FAILED] {agent_id} failed: {result.get('error')}")
                results[agent_id] = f"FAILED: {result.get('error')}"
        except Exception as e:
            print(f"  [ERROR] {agent_id} error: {e}")
            results[agent_id] = f"ERROR: {str(e)}"
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("-" * 60)
    
    success_count = sum(1 for r in results.values() if r == "SUCCESS")
    failed_count = sum(1 for r in results.values() if "FAILED" in str(r) or "ERROR" in str(r))
    skipped_count = sum(1 for r in results.values() if r == "SKIPPED")
    
    for agent_id, result in results.items():
        status = "[OK]" if result == "SUCCESS" else "[FAIL]" if "FAILED" in str(result) or "ERROR" in str(result) else "[SKIP]"
        print(f"  {status} {agent_id:20} - {result}")
    
    print("\n" + "-" * 60)
    print(f"Total: {len(agents)} agents")
    print(f"Success: {success_count}")
    print(f"Failed: {failed_count}")
    print(f"Skipped: {skipped_count}")
    
    if success_count == len(agents):
        print("\n[SUCCESS] All agents tested successfully!")
        return True
    else:
        print(f"\n[WARNING] {failed_count} agents failed testing")
        return False

if __name__ == "__main__":
    import sys
    success = test_all_agents()
    sys.exit(0 if success else 1)