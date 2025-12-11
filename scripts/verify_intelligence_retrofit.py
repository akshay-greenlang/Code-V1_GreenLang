# -*- coding: utf-8 -*-
"""
Intelligence Retrofit Verification Script

This script verifies that all agents have been properly retrofitted with
LLM intelligence capabilities via IntelligenceMixin.

Run with: python scripts/verify_intelligence_retrofit.py

Author: GreenLang Intelligence Framework
Date: December 2025
"""

import sys
import os
from typing import Dict, List, Tuple
from pathlib import Path

# Add greenlang to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_agent_intelligence(agent_class) -> Dict:
    """Check if an agent has intelligence capabilities."""
    result = {
        "has_intelligence_mixin": False,
        "has_get_intelligence_level": False,
        "has_get_capabilities": False,
        "has_generate_explanation": False,
        "has_generate_recommendations": False,
        "intelligence_level": None,
        "can_explain": False,
        "can_recommend": False,
    }

    # Check for intelligence mixin
    from greenlang.agents.intelligence_mixin import IntelligenceMixin
    result["has_intelligence_mixin"] = issubclass(agent_class, IntelligenceMixin)

    # Check for required methods
    result["has_get_intelligence_level"] = hasattr(agent_class, "get_intelligence_level")
    result["has_get_capabilities"] = hasattr(agent_class, "get_intelligence_capabilities")
    result["has_generate_explanation"] = hasattr(agent_class, "generate_explanation")
    result["has_generate_recommendations"] = hasattr(agent_class, "generate_recommendations")

    # Try to get intelligence level
    if result["has_get_intelligence_level"]:
        try:
            instance = agent_class()
            level = instance.get_intelligence_level()
            result["intelligence_level"] = level.value if level else None

            if result["has_get_capabilities"]:
                caps = instance.get_intelligence_capabilities()
                result["can_explain"] = caps.can_explain
                result["can_recommend"] = caps.can_recommend
        except Exception as e:
            result["error"] = str(e)

    return result


def main():
    """Verify all agents have been retrofitted."""
    print("=" * 80)
    print("GREENLANG INTELLIGENCE RETROFIT VERIFICATION")
    print("=" * 80)
    print()

    # Core agents to verify
    core_agents = [
        ("CarbonAgent", "greenlang.agents.carbon_agent"),
        ("BenchmarkAgent", "greenlang.agents.benchmark_agent"),
        ("IntensityAgent", "greenlang.agents.intensity_agent"),
        ("BuildingProfileAgent", "greenlang.agents.building_profile_agent"),
        ("EnergyBalanceAgent", "greenlang.agents.energy_balance_agent"),
        ("LoadProfileAgent", "greenlang.agents.load_profile_agent"),
        ("SiteInputAgent", "greenlang.agents.site_input_agent"),
        ("SolarResourceAgent", "greenlang.agents.solar_resource_agent"),
        ("FieldLayoutAgent", "greenlang.agents.field_layout_agent"),
    ]

    results = []
    total_passed = 0
    total_failed = 0

    print("CORE GREENLANG AGENTS:")
    print("-" * 80)

    for agent_name, module_path in core_agents:
        try:
            # Import the module
            module = __import__(module_path, fromlist=[agent_name])
            agent_class = getattr(module, agent_name)

            # Check intelligence
            check = check_agent_intelligence(agent_class)

            # Determine pass/fail
            passed = (
                check["has_intelligence_mixin"] and
                check["has_get_intelligence_level"] and
                check["has_get_capabilities"] and
                check["has_generate_explanation"]
            )

            status = "PASS" if passed else "FAIL"
            if passed:
                total_passed += 1
            else:
                total_failed += 1

            print(f"  {agent_name:30} [{status}]")
            print(f"    - IntelligenceMixin: {check['has_intelligence_mixin']}")
            print(f"    - Intelligence Level: {check['intelligence_level']}")
            print(f"    - Can Explain: {check['can_explain']}")
            print(f"    - Can Recommend: {check['can_recommend']}")
            if "error" in check:
                print(f"    - Error: {check['error']}")
            print()

            results.append((agent_name, passed, check))

        except Exception as e:
            print(f"  {agent_name:30} [ERROR]")
            print(f"    - Import Error: {str(e)}")
            print()
            total_failed += 1
            results.append((agent_name, False, {"error": str(e)}))

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  Total Agents Checked: {len(results)}")
    print(f"  Passed: {total_passed}")
    print(f"  Failed: {total_failed}")
    print(f"  Intelligence Retrofit: {total_passed}/{len(results)} ({100*total_passed/len(results):.0f}%)")
    print("=" * 80)

    # Return exit code
    return 0 if total_failed == 0 else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
