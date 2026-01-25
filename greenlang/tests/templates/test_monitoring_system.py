#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test suite for the operational monitoring system.

This script validates that all components of the monitoring system
are working correctly.

Usage:
    python templates/test_monitoring_system.py

Author: GreenLang Framework Team
Date: October 2025
"""

import sys
import os
from pathlib import Path
from greenlang.utilities.determinism import deterministic_random

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that all monitoring components can be imported."""
    print("\nTest 1: Import Validation")
    print("-" * 70)

    try:
        from greenlang.templates.agent_monitoring import (
            OperationalMonitoringMixin,
            HealthStatus,
            AlertSeverity,
            PerformanceMetrics,
            HealthCheckResult,
            Alert,
            MetricsCollector
        )
        print("✓ All monitoring components imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_mixin_integration():
    """Test mixin integration with a simple agent."""
    print("\nTest 2: Mixin Integration")
    print("-" * 70)

    try:
        from greenlang.templates.agent_monitoring import OperationalMonitoringMixin
        from greenlang.agents.base import BaseAgent, AgentResult

        class TestAgent(OperationalMonitoringMixin, BaseAgent):
            def __init__(self):
                super().__init__()
                self.setup_monitoring(agent_name="test_agent")

            def execute(self, input_data):
                with self.track_execution(input_data) as tracker:
                    tracker.set_cost(0.05)
                    tracker.set_tokens(1000)
                    return AgentResult(success=True, data={"result": "ok"})

        # Create and test agent
        agent = TestAgent()
        result = agent.execute({"test": "data"})

        assert result.success, "Agent execution failed"
        assert len(agent.get_execution_history()) == 1, "Execution not tracked"

        print("✓ Mixin integrates correctly with BaseAgent")
        print(f"✓ Execution tracked: {len(agent.get_execution_history())} executions")
        return True

    except Exception as e:
        print(f"✗ Mixin integration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_tracking():
    """Test performance tracking functionality."""
    print("\nTest 3: Performance Tracking")
    print("-" * 70)

    try:
        from greenlang.templates.agent_monitoring import OperationalMonitoringMixin
        from greenlang.agents.base import BaseAgent, AgentResult

        class PerfAgent(OperationalMonitoringMixin, BaseAgent):
            def __init__(self):
                super().__init__()
                self.setup_monitoring(agent_name="perf_agent")

            def execute(self, input_data):
                with self.track_execution(input_data) as tracker:
                    tracker.set_cost(0.08)
                    tracker.set_tokens(2500)
                    tracker.increment_ai_calls(1)
                    tracker.increment_tool_calls(3)
                    return AgentResult(success=True, data={})

        agent = PerfAgent()

        # Run multiple executions
        for i in range(10):
            agent.execute({"iteration": i})

        # Check history
        history = agent.get_execution_history()
        assert len(history) == 10, "Not all executions tracked"

        # Check metrics
        first = history[0]
        assert first['cost_usd'] == 0.08, "Cost not tracked"
        assert first['tokens_used'] == 2500, "Tokens not tracked"
        assert first['ai_calls'] == 1, "AI calls not tracked"
        assert first['tool_calls'] == 3, "Tool calls not tracked"

        print(f"✓ Tracked {len(history)} executions")
        print(f"✓ Cost tracking: ${first['cost_usd']:.3f}")
        print(f"✓ Token tracking: {first['tokens_used']}")
        print(f"✓ AI calls: {first['ai_calls']}")
        print(f"✓ Tool calls: {first['tool_calls']}")
        return True

    except Exception as e:
        print(f"✗ Performance tracking failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_health_checks():
    """Test health check functionality."""
    print("\nTest 4: Health Checks")
    print("-" * 70)

    try:
        from greenlang.templates.agent_monitoring import (
            OperationalMonitoringMixin,
            HealthStatus
        )
        from greenlang.agents.base import BaseAgent, AgentResult

        class HealthAgent(OperationalMonitoringMixin, BaseAgent):
            def __init__(self):
                super().__init__()
                self.setup_monitoring(agent_name="health_agent")

            def execute(self, input_data):
                with self.track_execution(input_data) as tracker:
                    if input_data.get("fail"):
                        raise ValueError("Intentional failure")
                    return AgentResult(success=True, data={})

        agent = HealthAgent()

        # Run successful executions
        for i in range(5):
            agent.execute({})

        # Check health
        health = agent.health_check()

        assert health.status == HealthStatus.HEALTHY, "Should be healthy"
        assert health.checks['monitoring_enabled'], "Monitoring not enabled"
        assert health.metrics['success_rate'] == 1.0, "Success rate incorrect"

        print(f"✓ Health status: {health.status.value}")
        print(f"✓ Success rate: {health.metrics['success_rate']:.1%}")
        print(f"✓ Uptime: {health.uptime_seconds:.1f}s")

        # Introduce errors
        for i in range(3):
            try:
                agent.execute({"fail": True})
            except ValueError:
                pass

        # Check degraded health
        health = agent.health_check()
        print(f"✓ Health after errors: {health.status.value}")
        print(f"✓ New success rate: {health.metrics['success_rate']:.1%}")

        return True

    except Exception as e:
        print(f"✗ Health checks failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_alerting():
    """Test alert generation."""
    print("\nTest 5: Alerting")
    print("-" * 70)

    try:
        from greenlang.templates.agent_monitoring import (
            OperationalMonitoringMixin,
            AlertSeverity
        )
        from greenlang.agents.base import BaseAgent, AgentResult
        import time

        alerts_received = []

        def alert_callback(alert):
            alerts_received.append(alert)

        class AlertAgent(OperationalMonitoringMixin, BaseAgent):
            def __init__(self):
                super().__init__()
                self.setup_monitoring(
                    agent_name="alert_agent",
                    alert_callback=alert_callback
                )
                # Set low thresholds to trigger alerts
                self.set_thresholds(
                    latency_ms=10,
                    error_rate=0.01,
                    cost_usd=0.01
                )

            def execute(self, input_data):
                with self.track_execution(input_data) as tracker:
                    if input_data.get("slow"):
                        time.sleep(0.05)  # Force high latency
                    tracker.set_cost(0.10)  # High cost
                    return AgentResult(success=True, data={})

        agent = AlertAgent()

        # Trigger alerts
        agent.execute({"slow": True})

        # Check alerts
        assert len(alerts_received) > 0, "No alerts generated"

        print(f"✓ Generated {len(alerts_received)} alerts")
        for alert in alerts_received:
            print(f"  - [{alert.severity.value}] {alert.message}")

        # Test alert retrieval
        all_alerts = agent.get_alerts(unresolved_only=False)
        assert len(all_alerts) > 0, "Alert retrieval failed"

        # Test alert resolution
        if all_alerts:
            agent.resolve_alert(all_alerts[0].alert_id)
            resolved = [a for a in agent.get_alerts(unresolved_only=False) if a.resolved]
            assert len(resolved) > 0, "Alert resolution failed"
            print(f"✓ Resolved {len(resolved)} alerts")

        return True

    except Exception as e:
        print(f"✗ Alerting failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_prometheus_metrics():
    """Test Prometheus metrics export."""
    print("\nTest 6: Prometheus Metrics Export")
    print("-" * 70)

    try:
        from greenlang.templates.agent_monitoring import OperationalMonitoringMixin
        from greenlang.agents.base import BaseAgent, AgentResult

        class MetricsAgent(OperationalMonitoringMixin, BaseAgent):
            def __init__(self):
                super().__init__()
                self.setup_monitoring(agent_name="metrics_agent")

            def execute(self, input_data):
                with self.track_execution(input_data) as tracker:
                    tracker.set_cost(0.05)
                    return AgentResult(success=True, data={})

        agent = MetricsAgent()

        # Run executions
        for i in range(5):
            agent.execute({"test": i})

        # Export metrics
        metrics_text = agent.export_metrics_prometheus()

        assert len(metrics_text) > 0, "No metrics exported"
        assert "executions_total" in metrics_text, "Missing counter metric"
        assert "execution_duration_ms" in metrics_text, "Missing histogram metric"

        print("✓ Prometheus metrics exported")
        print(f"✓ Metrics size: {len(metrics_text)} bytes")
        print("\nSample metrics:")
        lines = metrics_text.split('\n')[:10]
        for line in lines:
            if line.strip():
                print(f"  {line}")

        return True

    except Exception as e:
        print(f"✗ Prometheus export failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_summary():
    """Test performance summary generation."""
    print("\nTest 7: Performance Summary")
    print("-" * 70)

    try:
        from greenlang.templates.agent_monitoring import OperationalMonitoringMixin
        from greenlang.agents.base import BaseAgent, AgentResult
        import random

        class SummaryAgent(OperationalMonitoringMixin, BaseAgent):
            def __init__(self):
                super().__init__()
                self.setup_monitoring(agent_name="summary_agent")

            def execute(self, input_data):
                with self.track_execution(input_data) as tracker:
                    tracker.set_cost(random.uniform(0.05, 0.15))
                    tracker.set_tokens(deterministic_random().randint(1000, 3000))
                    return AgentResult(success=True, data={})

        agent = SummaryAgent()

        # Run executions
        for i in range(20):
            agent.execute({"iteration": i})

        # Get summary
        summary = agent.get_performance_summary(window_minutes=60)

        assert summary['total_executions'] == 20, "Execution count wrong"
        assert 'latency' in summary, "Missing latency metrics"
        assert 'cost' in summary, "Missing cost metrics"
        assert 'tokens' in summary, "Missing token metrics"

        print(f"✓ Total executions: {summary['total_executions']}")
        print(f"✓ Success rate: {summary['success_rate']:.1%}")
        print(f"✓ Avg latency: {summary['latency']['avg_ms']:.0f}ms")
        print(f"✓ P95 latency: {summary['latency']['p95_ms']:.0f}ms")
        print(f"✓ Total cost: ${summary['cost']['total_usd']:.2f}")
        print(f"✓ Avg tokens: {summary['tokens']['avg']:.0f}")

        return True

    except Exception as e:
        print(f"✗ Performance summary failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_file_existence():
    """Test that all required files exist."""
    print("\nTest 8: File Existence")
    print("-" * 70)

    workspace = Path(__file__).parent.parent
    required_files = [
        "templates/agent_monitoring.py",
        "templates/CHANGELOG_TEMPLATE.md",
        "templates/README_MONITORING.md",
        "templates/example_integration.py",
        "templates/MONITORING_SYSTEM_SUMMARY.md",
        "scripts/add_monitoring_and_changelog.py",
    ]

    all_exist = True
    for file_path in required_files:
        full_path = workspace / file_path
        if full_path.exists():
            size = full_path.stat().st_size
            print(f"✓ {file_path} ({size:,} bytes)")
        else:
            print(f"✗ {file_path} MISSING")
            all_exist = False

    return all_exist


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("GREENLANG OPERATIONAL MONITORING SYSTEM - TEST SUITE")
    print("=" * 70)

    tests = [
        ("Import Validation", test_imports),
        ("Mixin Integration", test_mixin_integration),
        ("Performance Tracking", test_performance_tracking),
        ("Health Checks", test_health_checks),
        ("Alerting", test_alerting),
        ("Prometheus Metrics", test_prometheus_metrics),
        ("Performance Summary", test_performance_summary),
        ("File Existence", test_file_existence),
    ]

    results = {}
    passed = 0
    failed = 0

    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n✗ Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
            failed += 1

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print("\n" + "-" * 70)
    print(f"Total: {len(tests)} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success Rate: {passed / len(tests) * 100:.0f}%")
    print("=" * 70)

    # Exit code
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
