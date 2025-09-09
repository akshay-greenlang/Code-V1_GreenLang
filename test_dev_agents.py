#!/usr/bin/env python3
"""Test the dev interface agents command"""

from greenlang.cli.dev_interface import GreenLangDevInterface

def test_agents_command():
    """Test the agents command functionality"""
    
    print("Testing 'gl dev' agents command...")
    print("=" * 50)
    
    # Create dev interface instance
    dev = GreenLangDevInterface()
    
    # Test list agents
    print("\n1. Testing list_agents():")
    try:
        dev.list_agents()
        print("   [OK] list_agents() executed successfully")
    except Exception as e:
        print(f"   [FAIL] list_agents() failed: {e}")
        return False
    
    # Test show agent info
    print("\n2. Testing show_agent_info():")
    for agent_id in ["validator", "fuel", "boiler"]:
        try:
            dev.show_agent_info(agent_id)
            print(f"   [OK] show_agent_info('{agent_id}') executed successfully")
        except Exception as e:
            print(f"   [FAIL] show_agent_info('{agent_id}') failed: {e}")
            return False
    
    # Test test_agent (simulated)
    print("\n3. Testing test_agent() with fuel agent:")
    try:
        # This would normally be interactive, so we'll just check it doesn't crash
        # when accessing the methods
        agent_id = "fuel"
        info = dev.client.get_agent_info(agent_id)
        if info:
            print(f"   [OK] Agent '{agent_id}' info retrieved")
            
            # Test execute_agent
            test_data = {
                "fuels": [
                    {
                        "type": "electricity",
                        "amount": 1000,
                        "unit": "kWh"
                    }
                ]
            }
            result = dev.client.execute_agent(agent_id, test_data)
            print(f"   [OK] execute_agent('{agent_id}') called successfully")
        else:
            print(f"   [FAIL] Could not get info for agent '{agent_id}'")
            return False
    except Exception as e:
        print(f"   [FAIL] test_agent() failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("[SUCCESS] All 'gl dev' agents commands are working!")
    return True

if __name__ == "__main__":
    import sys
    success = test_agents_command()
    sys.exit(0 if success else 1)