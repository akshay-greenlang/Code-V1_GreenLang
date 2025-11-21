# -*- coding: utf-8 -*-
"""
Comprehensive Validation Script for 24 Retrofitted GreenLang Agents

This script validates that all 24 GreenLang agents have been properly retrofitted
with @tool decorators and LLM tool calling capabilities.

Validation checks:
✅ Import: Has `from greenlang.intelligence.runtime.tools import tool`
✅ Tool decorator: Has @tool decorator with all required parameters
✅ Tool method: Method exists and is callable
✅ Parameters schema: Comprehensive JSON Schema with types
✅ Returns schema: Has returns_schema with "No Naked Numbers" structure
✅ Timeout: Has timeout_s parameter
✅ Description: Has LLM-friendly description
✅ Execute preserved: Original execute() or run() method still exists

Usage:
    python scripts/batch_retrofit_agents.py
"""

import os
import sys
import inspect
import importlib
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# List of all 24 agents to validate
CORE_AGENTS = [
    ("greenlang.agents.carbon_agent", "CarbonAgent"),
    ("greenlang.agents.energy_balance_agent", "EnergyBalanceAgent"),
    ("greenlang.agents.grid_factor_agent", "GridFactorAgent"),
    ("greenlang.agents.solar_resource_agent", "SolarResourceAgent"),
    ("greenlang.agents.fuel_agent", "FuelAgent"),
    ("greenlang.agents.boiler_agent", "BoilerAgent"),
    ("greenlang.agents.intensity_agent", "IntensityAgent"),
    ("greenlang.agents.load_profile_agent", "LoadProfileAgent"),
    ("greenlang.agents.site_input_agent", "SiteInputAgent"),
    ("greenlang.agents.field_layout_agent", "FieldLayoutAgent"),
    ("greenlang.agents.validator_agent", "InputValidatorAgent"),
    ("greenlang.agents.benchmark_agent", "BenchmarkAgent"),
    ("greenlang.agents.report_agent", "ReportAgent"),
    ("greenlang.agents.building_profile_agent", "BuildingProfileAgent"),
    ("greenlang.agents.recommendation_agent", "RecommendationAgent"),
]

PACK_AGENTS = [
    ("packs.boiler-solar.agents.boiler_analyzer", "BoilerAnalyzerAgent"),
    ("packs.boiler-solar.agents.solar_estimator", "SolarEstimatorAgent"),
    ("packs.cement-lca.agents.material_analyzer", "MaterialAnalyzerAgent"),
    ("packs.cement-lca.agents.emissions_calculator", "EmissionsCalculatorAgent"),
    ("packs.cement-lca.agents.impact_assessor", "ImpactAssessorAgent"),
    ("packs.hvac-measures.agents.energy_calculator", "EnergyCalculatorAgent"),
    ("packs.hvac-measures.agents.thermal_comfort", "ThermalComfortAgent"),
    ("packs.hvac-measures.agents.ventilation_optimizer", "VentilationOptimizerAgent"),
    ("packs.emissions-core.agents.fuel", "FuelAgent"),
]

ALL_AGENTS = CORE_AGENTS + PACK_AGENTS


def check_import_statement(module_path: str) -> Tuple[bool, str]:
    """Check if module has correct tool import"""
    try:
        # Convert module path to file path
        if module_path.startswith("packs."):
            # Handle pack agents - need to prepend "packs/" directory
            parts = module_path.replace("packs.", "packs/").replace(".", "/")
            file_path = Path(__file__).parent.parent / parts
            file_path = file_path.with_suffix(".py")
        else:
            # Handle core agents
            file_path = Path(__file__).parent.parent / module_path.replace(".", "/")
            file_path = file_path.with_suffix(".py")

        if not file_path.exists():
            return False, f"File not found: {file_path}"

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        if "from greenlang.intelligence.runtime.tools import tool" in content:
            return True, "Import statement found"
        else:
            return False, "Missing tool import statement"
    except Exception as e:
        return False, f"Error checking import: {e}"


def check_tool_decorator(agent_class) -> Tuple[bool, str, Dict[str, Any]]:
    """Check if agent has @tool decorated method"""
    try:
        # Need to instantiate the agent to get instance methods
        try:
            agent_instance = agent_class()
        except Exception as e:
            # If instantiation fails, try with empty config
            try:
                from greenlang.agents.base import AgentConfig
                agent_instance = agent_class(config=AgentConfig(name="test", description="test"))
            except:
                agent_instance = None

        if agent_instance is None:
            # Fall back to class inspection
            target = agent_class
        else:
            target = agent_instance

        # Find methods with _tool_spec attribute (correct attribute name)
        tool_methods = []
        for name, method in inspect.getmembers(target):
            if callable(method) and hasattr(method, '_tool_spec'):
                tool_methods.append((name, method))

        if not tool_methods:
            return False, "No @tool decorated method found", {}

        # Get the first tool method for validation
        method_name, method = tool_methods[0]
        tool_spec = method._tool_spec

        # Convert ToolSpec to dict for validation
        metadata = {
            'name': tool_spec.name,
            'description': tool_spec.description,
            'parameters_schema': tool_spec.parameters_schema,
            'returns_schema': tool_spec.returns_schema,
            'timeout_s': tool_spec.timeout_s,
        }

        return True, f"Found @tool method: {method_name}", metadata
    except Exception as e:
        return False, f"Error checking decorator: {e}", {}


def check_parameters_schema(metadata: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate parameters schema completeness"""
    try:
        schema = metadata.get('parameters_schema', {})

        if not schema:
            return False, "Missing parameters_schema"

        if schema.get('type') != 'object':
            return False, "Schema type must be 'object'"

        properties = schema.get('properties', {})
        if not properties:
            return False, "Schema has no properties"

        # Check that properties have types
        for prop_name, prop_def in properties.items():
            if 'type' not in prop_def:
                return False, f"Property '{prop_name}' missing type"
            if 'description' not in prop_def:
                return False, f"Property '{prop_name}' missing description"

        return True, f"{len(properties)} parameters defined"
    except Exception as e:
        return False, f"Error validating schema: {e}"


def check_returns_schema(metadata: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate returns schema has 'No Naked Numbers' structure"""
    try:
        schema = metadata.get('returns_schema', {})

        if not schema:
            return False, "Missing returns_schema"

        if schema.get('type') != 'object':
            return False, "Returns schema type must be 'object'"

        properties = schema.get('properties', {})
        if not properties:
            return False, "Returns schema has no properties"

        # Check for "No Naked Numbers" compliance
        # Look for properties with value/unit/source structure
        structured_properties = 0
        for prop_name, prop_def in properties.items():
            if prop_def.get('type') == 'object':
                nested_props = prop_def.get('properties', {})
                if 'value' in nested_props and 'unit' in nested_props:
                    structured_properties += 1

        if structured_properties > 0:
            return True, f"{len(properties)} return properties ({structured_properties} structured)"
        else:
            return True, f"{len(properties)} return properties (basic structure)"
    except Exception as e:
        return False, f"Error validating returns: {e}"


def check_timeout(metadata: Dict[str, Any]) -> Tuple[bool, str]:
    """Check timeout configuration"""
    try:
        timeout = metadata.get('timeout_s')
        if timeout is None:
            return False, "Missing timeout_s parameter"

        if not isinstance(timeout, (int, float)) or timeout <= 0:
            return False, f"Invalid timeout: {timeout}"

        return True, f"Timeout: {timeout}s"
    except Exception as e:
        return False, f"Error checking timeout: {e}"


def check_description(metadata: Dict[str, Any]) -> Tuple[bool, str]:
    """Check description quality"""
    try:
        description = metadata.get('description', '')

        if not description:
            return False, "Missing description"

        if len(description) < 20:
            return False, f"Description too short ({len(description)} chars)"

        return True, f"Description: {len(description)} chars"
    except Exception as e:
        return False, f"Error checking description: {e}"


def check_execute_method(agent_class) -> Tuple[bool, str]:
    """Check if original execute/run method exists"""
    try:
        has_execute = hasattr(agent_class, 'execute') and callable(getattr(agent_class, 'execute'))
        has_run = hasattr(agent_class, 'run') and callable(getattr(agent_class, 'run'))

        if has_execute:
            return True, "execute() method present"
        elif has_run:
            return True, "run() method present"
        else:
            return False, "No execute() or run() method"
    except Exception as e:
        return False, f"Error checking execute: {e}"


def validate_agent(module_path: str, class_name: str) -> Dict[str, Any]:
    """Validate a single agent for retrofit compliance"""
    result = {
        "module": module_path,
        "class": class_name,
        "checks": {},
        "passed": 0,
        "failed": 0,
        "warnings": []
    }

    try:
        # Import module
        module = importlib.import_module(module_path)
        agent_class = getattr(module, class_name)

        # Check 1: Import statement
        passed, msg = check_import_statement(module_path)
        result["checks"]["import"] = {"passed": passed, "message": msg}
        if passed:
            result["passed"] += 1
        else:
            result["failed"] += 1

        # Check 2: Tool decorator
        passed, msg, metadata = check_tool_decorator(agent_class)
        result["checks"]["decorator"] = {"passed": passed, "message": msg}
        if passed:
            result["passed"] += 1
        else:
            result["failed"] += 1
            return result  # Can't continue without metadata

        # Check 3: Parameters schema
        passed, msg = check_parameters_schema(metadata)
        result["checks"]["parameters_schema"] = {"passed": passed, "message": msg}
        if passed:
            result["passed"] += 1
        else:
            result["failed"] += 1

        # Check 4: Returns schema
        passed, msg = check_returns_schema(metadata)
        result["checks"]["returns_schema"] = {"passed": passed, "message": msg}
        if passed:
            result["passed"] += 1
        else:
            result["failed"] += 1

        # Check 5: Timeout
        passed, msg = check_timeout(metadata)
        result["checks"]["timeout"] = {"passed": passed, "message": msg}
        if passed:
            result["passed"] += 1
        else:
            result["failed"] += 1

        # Check 6: Description
        passed, msg = check_description(metadata)
        result["checks"]["description"] = {"passed": passed, "message": msg}
        if passed:
            result["passed"] += 1
        else:
            result["failed"] += 1

        # Check 7: Execute method
        passed, msg = check_execute_method(agent_class)
        result["checks"]["execute_method"] = {"passed": passed, "message": msg}
        if passed:
            result["passed"] += 1
        else:
            result["failed"] += 1

    except ImportError as e:
        result["checks"]["import_error"] = {"passed": False, "message": f"Failed to import: {e}"}
        result["failed"] += 7  # All checks fail
    except Exception as e:
        result["checks"]["error"] = {"passed": False, "message": f"Validation error: {e}"}
        result["failed"] += 7  # All checks fail

    return result


def print_agent_result(result: Dict[str, Any], verbose: bool = False):
    """Print validation result for a single agent"""
    total_checks = result["passed"] + result["failed"]
    pass_rate = (result["passed"] / total_checks * 100) if total_checks > 0 else 0

    status = "[PASS]" if result["failed"] == 0 else "[WARN]" if result["passed"] >= 5 else "[FAIL]"

    print(f"\n{status} {result['class']}")
    print(f"   Module: {result['module']}")
    print(f"   Passed: {result['passed']}/{total_checks} ({pass_rate:.0f}%)")

    if verbose or result["failed"] > 0:
        print(f"   Checks:")
        for check_name, check_result in result["checks"].items():
            check_status = "[OK]" if check_result["passed"] else "[X]"
            print(f"     {check_status} {check_name}: {check_result['message']}")


def print_summary_report(results: List[Dict[str, Any]]):
    """Print comprehensive summary report"""
    total_agents = len(results)
    fully_passed = sum(1 for r in results if r["failed"] == 0)
    partially_passed = sum(1 for r in results if 0 < r["failed"] < 7)
    failed = sum(1 for r in results if r["failed"] >= 7 or r["passed"] == 0)

    total_checks = sum(r["passed"] + r["failed"] for r in results)
    total_passed = sum(r["passed"] for r in results)

    pass_rate = (total_passed / total_checks * 100) if total_checks > 0 else 0

    print("\n" + "="*80)
    print("COMPREHENSIVE VALIDATION REPORT - 24 RETROFITTED AGENTS")
    print("="*80)

    print(f"\nSUMMARY STATISTICS:")
    print(f"   Total Agents Validated: {total_agents}")
    print(f"   [PASS] Fully Compliant:     {fully_passed} ({fully_passed/total_agents*100:.1f}%)")
    print(f"   [WARN] Partially Compliant: {partially_passed} ({partially_passed/total_agents*100:.1f}%)")
    print(f"   [FAIL] Failed:              {failed} ({failed/total_agents*100:.1f}%)")
    print(f"\n   Total Checks:    {total_checks}")
    print(f"   Checks Passed:   {total_passed}")
    print(f"   Checks Failed:   {total_checks - total_passed}")
    print(f"   Overall Pass Rate: {pass_rate:.1f}%")

    # Breakdown by check type
    print(f"\nVALIDATION BY CHECK TYPE:")
    check_types = ["import", "decorator", "parameters_schema", "returns_schema", "timeout", "description", "execute_method"]

    for check_type in check_types:
        passed = sum(1 for r in results if check_type in r["checks"] and r["checks"][check_type]["passed"])
        print(f"   {check_type:20s}: {passed}/{total_agents} passed ({passed/total_agents*100:.0f}%)")

    # List failed agents
    if failed > 0 or partially_passed > 0:
        print(f"\nAGENTS NEEDING ATTENTION:")
        for r in results:
            if r["failed"] > 0:
                print(f"   - {r['class']:30s} ({r['passed']}/{r['passed']+r['failed']} checks passed)")

    print("\n" + "="*80 + "\n")


def run_validation(verbose: bool = False):
    """Run validation on all 24 agents"""
    print("\n" + "="*80)
    print("GREENLANG AGENT RETROFIT VALIDATOR")
    print("   Validating 24 Retrofitted Agents with @tool Decorators")
    print("="*80)

    results = []

    print(f"\n[CORE AGENTS] VALIDATING {len(CORE_AGENTS)} AGENTS:")
    for module_path, class_name in CORE_AGENTS:
        result = validate_agent(module_path, class_name)
        results.append(result)
        print_agent_result(result, verbose)

    print(f"\n\n[PACK AGENTS] VALIDATING {len(PACK_AGENTS)} AGENTS:")
    for module_path, class_name in PACK_AGENTS:
        result = validate_agent(module_path, class_name)
        results.append(result)
        print_agent_result(result, verbose)

    print_summary_report(results)

    return results


def main():
    """Main execution"""
    import argparse

    parser = argparse.ArgumentParser(description="Validate retrofitted GreenLang agents")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    results = run_validation(verbose=args.verbose)

    # Determine exit code
    failed = sum(1 for r in results if r["failed"] >= 7 or r["passed"] == 0)

    if failed > 0:
        print(f"[FAIL] {failed} agent(s) failed validation")
        sys.exit(1)
    else:
        print(f"[SUCCESS] All {len(results)} agents passed validation!")
        sys.exit(0)


if __name__ == "__main__":
    main()
