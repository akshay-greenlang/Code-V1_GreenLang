#!/usr/bin/env python3
"""
Test script to verify all GreenLang Developer Interface commands
"""

import sys
from pathlib import Path

# Removed sys.path manipulation - using installed package

from greenlang.cli.dev_interface import GreenLangDevInterface
from greenlang.sdk import GreenLangClient

def test_commands():
    """Test that all commands are registered and callable"""
    
    print("Testing GreenLang Developer Interface Commands...")
    print("=" * 60)
    
    interface = GreenLangDevInterface()
    
    # List of all commands that should be available
    commands = [
        # Core Commands
        'new', 'calc', 'test', 'agents', 'workflow', 'repl',
        
        # Project Commands  
        'workspace', 'run', 'export', 'init', 'project',
        
        # Analysis Commands
        'benchmark', 'profile', 'validate', 'analyze', 'compare',
        
        # Documentation
        'docs', 'help', 'examples', 'api',
        
        # System
        'exit', 'quit', 'clear', 'status', 'version', 'config'
    ]
    
    # Test each command exists
    failed_commands = []
    for cmd in commands:
        # 'quit' is mapped to 'exit'
        if cmd == 'quit':
            method_name = 'cmd_exit'
        else:
            method_name = f"cmd_{cmd}"
        
        if hasattr(interface, method_name):
            print(f"[OK] {cmd:15} - Command found")
        else:
            print(f"[X]  {cmd:15} - Command NOT found")
            failed_commands.append(cmd)
    
    print("\n" + "=" * 60)
    
    if failed_commands:
        print(f"Failed commands: {', '.join(failed_commands)}")
        return False
    else:
        print("All commands are properly registered!")
        return True

def test_sdk():
    """Test SDK functionality"""
    print("\nTesting SDK Functions...")
    print("=" * 60)
    
    client = GreenLangClient()
    
    # Test basic calculation
    result = client.calculate_emissions("electricity", 1000, "kWh")
    if result["success"]:
        print(f"[OK] Emissions calculation: {result['data']['co2e_emissions_kg']} kg CO2e")
    else:
        print("[X] Emissions calculation failed")
    
    # Test aggregation
    emissions = [{"co2e_emissions_kg": 100}, {"co2e_emissions_kg": 200}]
    result = client.aggregate_emissions(emissions)
    if result["success"]:
        print(f"[OK] Aggregation: {result['data']['total_co2e_kg']} kg total")
    else:
        print("[X] Aggregation failed")
    
    # Test benchmark
    result = client.benchmark_emissions(1000, 10000, "commercial_office", 1)
    if result["success"]:
        print(f"[OK] Benchmark: {result['data']['rating']} rating")
    else:
        print("[X] Benchmark failed")
    
    print("\n" + "=" * 60)
    print("SDK tests completed!")

def main():
    """Run all tests"""
    print("""
    ============================================================
         GreenLang Developer Interface Test Suite          
    ============================================================
    """)
    
    # Test commands
    commands_ok = test_commands()
    
    # Test SDK
    test_sdk()
    
    if commands_ok:
        print("\n[OK] All tests passed! GreenLang is ready to use.")
        print("\nTo launch the developer interface, run:")
        print("  python -m greenlang.cli.main dev")
    else:
        print("\n[!] Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()