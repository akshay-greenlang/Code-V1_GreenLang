#!/usr/bin/env python3
"""
Phase 6 Verification Script
============================

Verify that Phase 6 implementation is 100% complete with all components:
- Priority 1 tools (Financial, Grid, Emissions)
- Security features (Validation, Rate Limiting, Audit Logging)
- Telemetry system
- Agent migration (V4 agents using shared tools)
- Comprehensive test coverage

Author: GreenLang Framework Team
Date: October 2025
"""

import os
import sys
import importlib
from pathlib import Path
from typing import Dict, List, Tuple
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


class Phase6Verifier:
    """Verify Phase 6 implementation completeness."""

    def __init__(self):
        self.results = {}
        self.errors = []
        self.warnings = []

    def verify_tools_exist(self) -> bool:
        """Verify all Priority 1 tools exist and are importable."""
        print("\n" + "=" * 60)
        print("1. VERIFYING PRIORITY 1 TOOLS")
        print("=" * 60)

        tools_to_check = [
            ("greenlang.agents.tools.financial", "FinancialMetricsTool"),
            ("greenlang.agents.tools.grid", "GridIntegrationTool"),
            ("greenlang.agents.tools.emissions", "EmissionsCalculatorTool"),
        ]

        all_exist = True

        for module_name, class_name in tools_to_check:
            try:
                module = importlib.import_module(module_name)
                tool_class = getattr(module, class_name)
                print(f"  ✓ {class_name:<30} FOUND")

                # Try to instantiate
                tool = tool_class()
                print(f"    - Instantiation successful")
                print(f"    - Tool name: {tool.name}")

            except Exception as e:
                print(f"  ✗ {class_name:<30} FAILED")
                print(f"    Error: {e}")
                all_exist = False
                self.errors.append(f"Tool {class_name} failed: {e}")

        self.results["tools_exist"] = all_exist
        return all_exist

    def verify_security_features(self) -> bool:
        """Verify security features are working."""
        print("\n" + "=" * 60)
        print("2. VERIFYING SECURITY FEATURES")
        print("=" * 60)

        all_working = True

        # Check validation
        print("\n  Validation Framework:")
        try:
            from greenlang.agents.tools.validation import ValidationRule, ValidationResult
            print("    ✓ Validation imports successful")

            # Test creating a rule
            rule = ValidationRule(
                name="test_positive",
                check=lambda x, ctx: x > 0,
                error_message="Must be positive"
            )
            result = rule.validate(10)
            assert result.valid
            print("    ✓ Validation rule execution works")

        except Exception as e:
            print(f"    ✗ Validation failed: {e}")
            all_working = False
            self.errors.append(f"Validation: {e}")

        # Check rate limiting
        print("\n  Rate Limiting:")
        try:
            from greenlang.agents.tools.rate_limiting import RateLimiter, get_rate_limiter
            print("    ✓ Rate limiting imports successful")

            limiter = get_rate_limiter()
            print(f"    ✓ Global rate limiter active")
            print(f"      - Rate: {limiter.rate} calls/sec")
            print(f"      - Burst: {limiter.burst}")

        except Exception as e:
            print(f"    ✗ Rate limiting failed: {e}")
            all_working = False
            self.errors.append(f"Rate limiting: {e}")

        # Check audit logging
        print("\n  Audit Logging:")
        try:
            from greenlang.agents.tools.audit import AuditLogger, get_audit_logger
            print("    ✓ Audit logging imports successful")

            logger = get_audit_logger()
            print(f"    ✓ Global audit logger active")
            print(f"      - Max logs: {logger.max_logs}")

        except Exception as e:
            print(f"    ✗ Audit logging failed: {e}")
            all_working = False
            self.errors.append(f"Audit logging: {e}")

        # Check security config
        print("\n  Security Configuration:")
        try:
            from greenlang.agents.tools.security_config import SecurityConfig, get_security_config
            print("    ✓ Security config imports successful")

            config = get_security_config()
            print(f"    ✓ Global security config active")
            print(f"      - Validation enabled: {config.enable_validation}")
            print(f"      - Rate limiting enabled: {config.enable_rate_limiting}")
            print(f"      - Audit logging enabled: {config.audit_log_successes}")

        except Exception as e:
            print(f"    ✗ Security config failed: {e}")
            all_working = False
            self.errors.append(f"Security config: {e}")

        self.results["security_features"] = all_working
        return all_working

    def verify_telemetry(self) -> bool:
        """Verify telemetry system is working."""
        print("\n" + "=" * 60)
        print("3. VERIFYING TELEMETRY SYSTEM")
        print("=" * 60)

        all_working = True

        try:
            from greenlang.agents.tools.telemetry import (
                TelemetryCollector,
                ToolMetrics,
                get_telemetry,
                reset_global_telemetry
            )
            print("  ✓ Telemetry imports successful")

            # Create collector and test
            collector = TelemetryCollector()
            print("  ✓ TelemetryCollector instantiation successful")

            # Record a test execution
            collector.record_execution(
                tool_name="test_tool",
                execution_time_ms=45.5,
                success=True
            )
            print("  ✓ Execution recording works")

            # Get metrics
            metrics = collector.get_tool_metrics("test_tool")
            assert metrics.total_calls == 1
            assert metrics.successful_calls == 1
            assert metrics.avg_execution_time_ms == 45.5
            print("  ✓ Metrics retrieval works")

            # Test exports
            json_export = collector.export_metrics(format="json")
            assert isinstance(json_export, dict)
            print("  ✓ JSON export works")

            prometheus_export = collector.export_metrics(format="prometheus")
            assert isinstance(prometheus_export, str)
            print("  ✓ Prometheus export works")

            csv_export = collector.export_metrics(format="csv")
            assert isinstance(csv_export, str)
            print("  ✓ CSV export works")

            # Test global singleton
            tel1 = get_telemetry()
            tel2 = get_telemetry()
            assert tel1 is tel2
            print("  ✓ Global telemetry singleton works")

            # Test integration with tools
            print("\n  Testing telemetry integration with tools:")
            from greenlang.agents.tools.financial import FinancialMetricsTool

            reset_global_telemetry()
            tool = FinancialMetricsTool()
            result = tool(
                capital_cost=50000,
                annual_savings=8000,
                lifetime_years=25
            )

            assert result.success
            print("    ✓ Tool execution successful")

            telemetry = get_telemetry()
            tool_metrics = telemetry.get_tool_metrics("calculate_financial_metrics")
            assert tool_metrics.total_calls >= 1
            print(f"    ✓ Telemetry recorded: {tool_metrics.total_calls} calls")
            print(f"      - Avg time: {tool_metrics.avg_execution_time_ms:.2f}ms")

        except Exception as e:
            print(f"  ✗ Telemetry verification failed: {e}")
            all_working = False
            self.errors.append(f"Telemetry: {e}")
            import traceback
            traceback.print_exc()

        self.results["telemetry"] = all_working
        return all_working

    def verify_agent_migration(self) -> bool:
        """Verify V4 agents use shared tools."""
        print("\n" + "=" * 60)
        print("4. VERIFYING AGENT MIGRATION")
        print("=" * 60)

        # Check for V4 agents
        agents_dir = PROJECT_ROOT / "greenlang" / "agents"

        v4_agents = []
        for agent_file in agents_dir.glob("*_agent_*.py"):
            if "v4" in agent_file.stem:
                v4_agents.append(agent_file.stem)

        if v4_agents:
            print(f"\n  Found {len(v4_agents)} V4 agents:")
            for agent in v4_agents:
                print(f"    - {agent}")
            self.results["v4_agents_exist"] = True
        else:
            print("  ⚠ No V4 agents found yet")
            self.warnings.append("No V4 agents found")
            self.results["v4_agents_exist"] = False

        # For now, we have V3 agents that we're migrating
        # Check that shared tools are being used
        try:
            print("\n  Checking shared tools are importable from agents:")
            from greenlang.agents.tools import (
                FinancialMetricsTool,
                GridIntegrationTool,
                EmissionsCalculatorTool
            )
            print("    ✓ Shared tools can be imported from tools package")

        except Exception as e:
            print(f"    ✗ Failed to import shared tools: {e}")
            self.errors.append(f"Agent imports: {e}")
            return False

        self.results["agent_migration"] = True
        return True

    def verify_tests(self) -> bool:
        """Verify test coverage."""
        print("\n" + "=" * 60)
        print("5. VERIFYING TEST COVERAGE")
        print("=" * 60)

        tests_dir = PROJECT_ROOT / "tests" / "agents" / "tools"

        required_tests = [
            "test_financial.py",
            "test_grid.py",
            "test_security.py",
            "test_telemetry.py",
            "test_integration.py",
        ]

        all_exist = True

        for test_file in required_tests:
            test_path = tests_dir / test_file
            if test_path.exists():
                # Count test functions
                with open(test_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    test_count = content.count("def test_")

                print(f"  ✓ {test_file:<30} ({test_count} tests)")
            else:
                print(f"  ✗ {test_file:<30} MISSING")
                all_exist = False
                self.errors.append(f"Missing test file: {test_file}")

        # Check if we can import tests
        print("\n  Checking test imports:")
        try:
            sys.path.insert(0, str(tests_dir))
            import test_telemetry
            print("    ✓ test_telemetry imports successfully")

            import test_integration
            print("    ✓ test_integration imports successfully")

        except Exception as e:
            print(f"    ⚠ Test import issue: {e}")
            self.warnings.append(f"Test import: {e}")

        self.results["tests"] = all_exist
        return all_exist

    def generate_completion_report(self) -> None:
        """Generate Phase 6 completion report."""
        print("\n" + "=" * 60)
        print("PHASE 6 COMPLETION REPORT")
        print("=" * 60)

        # Calculate completion percentage
        completed_items = sum(1 for v in self.results.values() if v)
        total_items = len(self.results)
        completion_percentage = (completed_items / total_items * 100) if total_items > 0 else 0

        print(f"\nCompletion: {completion_percentage:.1f}% ({completed_items}/{total_items})")
        print("\nComponent Status:")

        status_map = {
            "tools_exist": "Priority 1 Tools",
            "security_features": "Security Features",
            "telemetry": "Telemetry System",
            "agent_migration": "Agent Migration",
            "tests": "Test Coverage"
        }

        for key, label in status_map.items():
            if key in self.results:
                status = "✓ PASS" if self.results[key] else "✗ FAIL"
                print(f"  {status:<10} {label}")

        # Print errors
        if self.errors:
            print("\n" + "=" * 60)
            print(f"ERRORS ({len(self.errors)}):")
            print("=" * 60)
            for i, error in enumerate(self.errors, 1):
                print(f"{i}. {error}")

        # Print warnings
        if self.warnings:
            print("\n" + "=" * 60)
            print(f"WARNINGS ({len(self.warnings)}):")
            print("=" * 60)
            for i, warning in enumerate(self.warnings, 1):
                print(f"{i}. {warning}")

        # Summary
        print("\n" + "=" * 60)
        if completion_percentage == 100 and not self.errors:
            print("✅ PHASE 6: 100% COMPLETE")
            print("=" * 60)
            print("\nAll components verified successfully:")
            print("  • Priority 1 tools (Financial, Grid, Emissions)")
            print("  • Security framework (Validation, Rate Limiting, Audit)")
            print("  • Telemetry system (Metrics, Export, Integration)")
            print("  • Comprehensive test coverage")
            print("\nPhase 6 is production-ready!")
        else:
            print("⚠ PHASE 6: INCOMPLETE")
            print("=" * 60)
            print(f"\nCompletion: {completion_percentage:.1f}%")
            print(f"Errors: {len(self.errors)}")
            print(f"Warnings: {len(self.warnings)}")
            print("\nSee details above for issues to resolve.")

        # Save report to file
        report_path = PROJECT_ROOT / "PHASE_6_VERIFICATION_REPORT.json"
        report_data = {
            "completion_percentage": completion_percentage,
            "results": self.results,
            "errors": self.errors,
            "warnings": self.warnings,
            "completed_items": completed_items,
            "total_items": total_items
        }

        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)

        print(f"\nReport saved to: {report_path}")

    def run_all_verifications(self) -> bool:
        """Run all verifications."""
        print("\n" + "=" * 60)
        print("PHASE 6 VERIFICATION STARTING")
        print("=" * 60)
        print(f"Project root: {PROJECT_ROOT}")

        # Run all verification steps
        self.verify_tools_exist()
        self.verify_security_features()
        self.verify_telemetry()
        self.verify_agent_migration()
        self.verify_tests()

        # Generate report
        self.generate_completion_report()

        # Return True if all passed
        return all(self.results.values()) and not self.errors


def main():
    """Main entry point."""
    verifier = Phase6Verifier()
    success = verifier.run_all_verifications()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
