#!/usr/bin/env python3
"""
Agent Compliance Verification Script
GL-VCCI Scope 3 Platform

Verifies that all agents inherit from greenlang.sdk.base.Agent
and implement required methods.

Usage:
    python verify_agent_compliance.py
"""

import sys
import inspect
from typing import Dict, List, Any


def verify_agent_compliance() -> Dict[str, Any]:
    """
    Verify that all agents are compliant with GreenLang SDK requirements.

    Returns:
        Dictionary with compliance results
    """
    results = {
        "total_agents": 5,
        "compliant_agents": 0,
        "agents": [],
        "overall_status": "FAIL"
    }

    agents = [
        {
            "name": "ValueChainIntakeAgent",
            "module": "services.agents.intake.agent",
            "class_name": "ValueChainIntakeAgent"
        },
        {
            "name": "Scope3CalculatorAgent",
            "module": "services.agents.calculator.agent",
            "class_name": "Scope3CalculatorAgent"
        },
        {
            "name": "HotspotAnalysisAgent",
            "module": "services.agents.hotspot.agent",
            "class_name": "HotspotAnalysisAgent"
        },
        {
            "name": "Scope3ReportingAgent",
            "module": "services.agents.reporting.agent",
            "class_name": "Scope3ReportingAgent"
        },
        {
            "name": "SupplierEngagementAgent",
            "module": "services.agents.engagement.agent",
            "class_name": "SupplierEngagementAgent"
        }
    ]

    for agent_info in agents:
        agent_result = {
            "name": agent_info["name"],
            "compliant": False,
            "checks": {
                "imports_successfully": False,
                "inherits_from_agent": False,
                "has_metadata": False,
                "has_validate_method": False,
                "has_process_method": False,
                "has_run_method": False,
                "has_cache_manager": False,
                "has_metrics_collector": False
            },
            "errors": []
        }

        try:
            # Import agent
            module = __import__(agent_info["module"], fromlist=[agent_info["class_name"]])
            agent_class = getattr(module, agent_info["class_name"])
            agent_result["checks"]["imports_successfully"] = True

            # Check inheritance
            from greenlang.sdk.base import Agent
            if issubclass(agent_class, Agent):
                agent_result["checks"]["inherits_from_agent"] = True

            # Check for required methods
            if hasattr(agent_class, "validate"):
                agent_result["checks"]["has_validate_method"] = True

            if hasattr(agent_class, "process"):
                agent_result["checks"]["has_process_method"] = True

            if hasattr(agent_class, "run"):
                agent_result["checks"]["has_run_method"] = True

            # Check metadata (need to instantiate with mock args)
            # This is tricky without knowing exact constructor args
            # So we'll check the __init__ signature instead
            init_signature = inspect.signature(agent_class.__init__)

            # Check if agent would have metadata attribute (from base class)
            if hasattr(Agent, "metadata"):
                agent_result["checks"]["has_metadata"] = True

            # For cache and metrics, check if mentioned in source
            import inspect
            source = inspect.getsource(agent_class)
            if "cache_manager" in source or "CacheManager" in source:
                agent_result["checks"]["has_cache_manager"] = True

            if "MetricsCollector" in source or "metrics" in source:
                agent_result["checks"]["has_metrics_collector"] = True

            # Determine compliance
            required_checks = [
                "imports_successfully",
                "inherits_from_agent",
                "has_validate_method",
                "has_process_method",
                "has_run_method"
            ]

            if all(agent_result["checks"][check] for check in required_checks):
                agent_result["compliant"] = True
                results["compliant_agents"] += 1

        except Exception as e:
            agent_result["errors"].append(str(e))

        results["agents"].append(agent_result)

    # Overall status
    if results["compliant_agents"] == results["total_agents"]:
        results["overall_status"] = "PASS"
        results["compliance_percentage"] = 100.0
    else:
        results["compliance_percentage"] = (results["compliant_agents"] / results["total_agents"]) * 100

    return results


def print_results(results: Dict[str, Any]):
    """Print compliance results in a formatted way."""
    print("=" * 80)
    print("AGENT COMPLIANCE VERIFICATION REPORT")
    print("=" * 80)
    print()

    print(f"Overall Status: {results['overall_status']}")
    print(f"Compliance: {results['compliant_agents']}/{results['total_agents']} agents ({results['compliance_percentage']:.1f}%)")
    print()

    print("-" * 80)
    print("AGENT DETAILS")
    print("-" * 80)

    for agent in results["agents"]:
        status_icon = "✅" if agent["compliant"] else "❌"
        print(f"\n{status_icon} {agent['name']}")
        print(f"   {'─' * 70}")

        for check, passed in agent["checks"].items():
            check_icon = "✓" if passed else "✗"
            check_label = check.replace("_", " ").title()
            print(f"   {check_icon} {check_label}")

        if agent["errors"]:
            print(f"\n   Errors:")
            for error in agent["errors"]:
                print(f"   - {error}")

    print()
    print("=" * 80)

    if results["overall_status"] == "PASS":
        print("🎉 ALL AGENTS ARE COMPLIANT! 🎉")
    else:
        print(f"⚠️  {results['total_agents'] - results['compliant_agents']} agent(s) need attention")

    print("=" * 80)


def main():
    """Main entry point."""
    print("\nRunning Agent Compliance Verification...\n")

    try:
        results = verify_agent_compliance()
        print_results(results)

        # Exit with appropriate code
        if results["overall_status"] == "PASS":
            sys.exit(0)
        else:
            sys.exit(1)

    except Exception as e:
        print(f"\n❌ Verification failed with error: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
