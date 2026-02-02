#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test the create agent functionality"""

from greenlang.cli.dev_interface import GreenLangDevInterface
import tempfile
import os

def test_create_agent_templates():
    """Test that all agent templates can be generated"""
    
    print("Testing agent creation templates...")
    print("=" * 60)
    
    dev = GreenLangDevInterface()
    
    # Test data for different agent types
    test_cases = [
        ("custom", "TestCustom", "A custom test agent"),
        ("boiler", "TestBoiler", "A boiler emissions agent"),
        ("emissions", "TestEmissions", "An emissions calculation agent"),
        ("fuel", "TestFuel", "A fuel emissions agent"),
        ("validator", "TestValidator", "A validation agent"),
        ("benchmark", "TestBenchmark", "A benchmarking agent"),
        ("report", "TestReport", "A reporting agent"),
        ("intensity", "TestIntensity", "An intensity calculation agent"),
        ("recommendation", "TestRecommend", "A recommendation agent")
    ]
    
    results = {}
    
    for agent_type, name, description in test_cases:
        print(f"\nTesting {agent_type} agent template...")
        print("-" * 40)
        
        try:
            # Generate agent code
            code = dev._generate_agent_code(agent_type, name, description)
            
            # Check if code was generated
            if code and len(code) > 100:
                # Check for key components
                checks = {
                    "class_definition": f"class {name}Agent" in code,
                    "base_class": "BaseAgent" in code,
                    "execute_method": "def execute" in code,
                    "validate_method": "def validate" in code or "def validate_input" in code,
                    "agent_result": "AgentResult" in code
                }
                
                all_passed = all(checks.values())
                
                if all_passed:
                    print(f"  [SUCCESS] {agent_type} template generated correctly")
                    print(f"  - Generated {len(code)} characters of code")
                    print(f"  - All required components present")
                    results[agent_type] = "SUCCESS"
                else:
                    print(f"  [PARTIAL] {agent_type} template missing components:")
                    for check, passed in checks.items():
                        if not passed:
                            print(f"    - Missing: {check}")
                    results[agent_type] = "PARTIAL"
                    
                # Test if code is valid Python
                try:
                    compile(code, f"{name}_agent.py", 'exec')
                    print(f"  - Valid Python syntax")
                except SyntaxError as e:
                    print(f"  - [ERROR] Syntax error: {e}")
                    results[agent_type] = f"SYNTAX_ERROR: {e}"
                    
            else:
                print(f"  [FAIL] No code generated")
                results[agent_type] = "NO_CODE"
                
        except Exception as e:
            print(f"  [ERROR] Failed to generate template: {e}")
            results[agent_type] = f"ERROR: {e}"
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print("-" * 60)
    
    success_count = sum(1 for r in results.values() if r == "SUCCESS")
    partial_count = sum(1 for r in results.values() if r == "PARTIAL")
    failed_count = len(results) - success_count - partial_count
    
    for agent_type, result in results.items():
        status = "[OK]" if result == "SUCCESS" else "[PARTIAL]" if result == "PARTIAL" else "[FAIL]"
        print(f"  {status} {agent_type:15} - {result}")
    
    print("\n" + "-" * 60)
    print(f"Total: {len(test_cases)} agent types")
    print(f"Success: {success_count}")
    print(f"Partial: {partial_count}")
    print(f"Failed: {failed_count}")
    
    if success_count == len(test_cases):
        print("\n[SUCCESS] All agent templates generated successfully!")
        return True
    else:
        print(f"\n[WARNING] {failed_count + partial_count} agent templates have issues")
        return False

if __name__ == "__main__":
    import sys
    success = test_create_agent_templates()
    sys.exit(0 if success else 1)