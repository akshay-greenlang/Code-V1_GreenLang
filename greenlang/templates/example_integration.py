#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Example: Integrating Operational Monitoring into an Existing Agent.

This example demonstrates:
1. Before/after comparison of agent code
2. Full integration with all monitoring features
3. Testing the integration
4. Using monitoring data in production

Author: GreenLang Framework Team
Date: October 2025
"""

from typing import Dict, Any
from greenlang.agents.base import BaseAgent, AgentResult, AgentConfig
from greenlang.determinism import deterministic_random
from greenlang.templates.agent_monitoring import (
    OperationalMonitoringMixin,
    HealthStatus,
    AlertSeverity
)
import time
import random


# ============================================================================
# BEFORE: Agent without monitoring
# ============================================================================

class CarbonAgentBefore(BaseAgent):
    """Carbon emissions calculator (WITHOUT monitoring)."""

    def __init__(self, config: AgentConfig = None):
        if config is None:
            config = AgentConfig(
                name="CarbonAgent",
                description="Calculates carbon emissions",
            )
        super().__init__(config)

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Calculate emissions (basic version)."""
        # Validate
        if "emissions" not in input_data:
            return AgentResult(success=False, error="Missing emissions data")

        # Calculate
        emissions_list = input_data["emissions"]
        total_co2e = sum(e.get("co2e_kg", 0) for e in emissions_list)

        # Return
        return AgentResult(
            success=True,
            data={
                "total_co2e_kg": total_co2e,
                "total_co2e_tons": total_co2e / 1000
            }
        )


# ============================================================================
# AFTER: Agent with comprehensive monitoring
# ============================================================================

class CarbonAgentAfter(OperationalMonitoringMixin, BaseAgent):
    """Carbon emissions calculator (WITH monitoring)."""

    def __init__(self, config: AgentConfig = None):
        if config is None:
            config = AgentConfig(
                name="CarbonAgent",
                description="Calculates carbon emissions",
            )
        super().__init__(config)

        # Setup monitoring
        self.setup_monitoring(
            agent_name="carbon_agent",
            enable_metrics=True,
            enable_health_checks=True,
            enable_alerting=True,
            max_history=1000,
            alert_callback=self._handle_alert
        )

        # Set production thresholds
        self.set_thresholds(
            latency_ms=3000,    # 3 second SLA
            error_rate=0.05,    # 5% error tolerance
            cost_usd=0.50       # Cost limit
        )

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        """Calculate emissions (monitored version)."""
        # Track execution with monitoring
        with self.track_execution(input_data) as tracker:
            # Validate
            if "emissions" not in input_data:
                return AgentResult(success=False, error="Missing emissions data")

            # Simulate AI call for demonstration
            time.sleep(0.1)  # Simulate processing
            tracker.increment_ai_calls(1)
            tracker.set_tokens(2500)
            tracker.set_cost(0.08)

            # Calculate
            emissions_list = input_data["emissions"]
            total_co2e = sum(e.get("co2e_kg", 0) for e in emissions_list)

            # Track tool usage
            tracker.increment_tool_calls(len(emissions_list))

            # Return result
            return AgentResult(
                success=True,
                data={
                    "total_co2e_kg": total_co2e,
                    "total_co2e_tons": total_co2e / 1000
                }
            )

    def _handle_alert(self, alert):
        """Handle alerts."""
        print(f"\n[ALERT] {alert.severity.value.upper()}: {alert.message}")
        print(f"Context: {alert.context}")


# ============================================================================
# Integration Testing
# ============================================================================

def test_basic_execution():
    """Test basic execution with monitoring."""
    print("\n" + "=" * 70)
    print("TEST 1: Basic Execution")
    print("=" * 70)

    agent = CarbonAgentAfter()

    # Execute
    input_data = {
        "emissions": [
            {"fuel_type": "natural_gas", "co2e_kg": 1000},
            {"fuel_type": "electricity", "co2e_kg": 500},
        ]
    }

    result = agent.execute(input_data)

    print(f"\nExecution Result:")
    print(f"  Success: {result.success}")
    print(f"  Total CO2e: {result.data['total_co2e_tons']:.3f} tons")

    # Check execution history
    history = agent.get_execution_history(limit=1)
    if history:
        metrics = history[0]
        print(f"\nExecution Metrics:")
        print(f"  Duration: {metrics['duration_ms']:.0f}ms")
        print(f"  Cost: ${metrics['cost_usd']:.3f}")
        print(f"  Tokens: {metrics['tokens_used']}")
        print(f"  AI Calls: {metrics['ai_calls']}")
        print(f"  Tool Calls: {metrics['tool_calls']}")


def test_health_checks():
    """Test health check functionality."""
    print("\n" + "=" * 70)
    print("TEST 2: Health Checks")
    print("=" * 70)

    agent = CarbonAgentAfter()

    # Run some executions
    for i in range(5):
        agent.execute({
            "emissions": [{"co2e_kg": deterministic_random().randint(100, 1000)}]
        })

    # Perform health check
    health = agent.health_check()

    print(f"\nHealth Status: {health.status.value.upper()}")
    print(f"Uptime: {health.uptime_seconds:.1f} seconds")

    print(f"\nHealth Checks:")
    for check_name, passed in health.checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check_name.replace('_', ' ').title()}")

    print(f"\nHealth Metrics:")
    for metric_name, value in health.metrics.items():
        if isinstance(value, float):
            if 'rate' in metric_name:
                print(f"  {metric_name}: {value:.1%}")
            elif 'latency' in metric_name or 'duration' in metric_name:
                print(f"  {metric_name}: {value:.0f}ms")
            else:
                print(f"  {metric_name}: {value:.2f}")
        else:
            print(f"  {metric_name}: {value}")

    if health.degradation_reasons:
        print(f"\nDegradation Reasons:")
        for reason in health.degradation_reasons:
            print(f"  - {reason}")


def test_performance_summary():
    """Test performance summary."""
    print("\n" + "=" * 70)
    print("TEST 3: Performance Summary")
    print("=" * 70)

    agent = CarbonAgentAfter()

    # Run multiple executions
    print("\nRunning 20 executions...")
    for i in range(20):
        try:
            agent.execute({
                "emissions": [{"co2e_kg": deterministic_random().randint(100, 5000)}]
            })
        except Exception as e:
            pass  # Some might fail for testing

    # Get performance summary
    summary = agent.get_performance_summary(window_minutes=60)

    print(f"\nPerformance Summary (Last 60 minutes):")
    print(f"  Total Executions: {summary['total_executions']}")
    print(f"  Successful: {summary['successful_executions']}")
    print(f"  Failed: {summary['failed_executions']}")
    print(f"  Success Rate: {summary['success_rate']:.1%}")

    print(f"\nLatency:")
    print(f"  Average: {summary['latency']['avg_ms']:.0f}ms")
    print(f"  Min: {summary['latency']['min_ms']:.0f}ms")
    print(f"  Max: {summary['latency']['max_ms']:.0f}ms")
    print(f"  p50: {summary['latency']['p50_ms']:.0f}ms")
    print(f"  p95: {summary['latency']['p95_ms']:.0f}ms")
    print(f"  p99: {summary['latency']['p99_ms']:.0f}ms")

    print(f"\nCost:")
    print(f"  Total: ${summary['cost']['total_usd']:.2f}")
    print(f"  Average: ${summary['cost']['avg_usd']:.3f}")
    print(f"  Min: ${summary['cost']['min_usd']:.3f}")
    print(f"  Max: ${summary['cost']['max_usd']:.3f}")

    print(f"\nTokens:")
    print(f"  Total: {summary['tokens']['total']:,}")
    print(f"  Average: {summary['tokens']['avg']:.0f}")

    print(f"\nCache Hit Rate: {summary['cache_hit_rate']:.1%}")


def test_alerting():
    """Test alerting functionality."""
    print("\n" + "=" * 70)
    print("TEST 4: Alerting")
    print("=" * 70)

    agent = CarbonAgentAfter()

    # Set aggressive thresholds to trigger alerts
    agent.set_thresholds(
        latency_ms=50,      # Very low to trigger
        error_rate=0.01,    # Very low to trigger
        cost_usd=0.01       # Very low to trigger
    )

    print("\nRunning executions to trigger alerts...")

    # Run executions that will trigger alerts
    for i in range(10):
        try:
            # Some with high latency
            if i % 3 == 0:
                time.sleep(0.1)  # Force high latency

            agent.execute({
                "emissions": [{"co2e_kg": 1000}]
            })

            # Some failures
            if i % 5 == 0:
                agent.execute({})  # Will fail validation

        except Exception:
            pass

    # Get alerts
    all_alerts = agent.get_alerts(unresolved_only=False)
    unresolved_alerts = agent.get_alerts(unresolved_only=True)

    print(f"\nTotal Alerts: {len(all_alerts)}")
    print(f"Unresolved: {len(unresolved_alerts)}")

    # Show alerts by severity
    for severity in AlertSeverity:
        severity_alerts = agent.get_alerts(severity=severity, unresolved_only=True)
        if severity_alerts:
            print(f"\n{severity.value.upper()} Alerts ({len(severity_alerts)}):")
            for alert in severity_alerts[:3]:  # Show first 3
                print(f"  - {alert.message}")
                print(f"    Context: {alert.context}")

    # Resolve some alerts
    if unresolved_alerts:
        print(f"\nResolving first alert...")
        agent.resolve_alert(unresolved_alerts[0].alert_id)
        print(f"Alerts now: {len(agent.get_alerts(unresolved_only=True))} unresolved")


def test_prometheus_metrics():
    """Test Prometheus metrics export."""
    print("\n" + "=" * 70)
    print("TEST 5: Prometheus Metrics")
    print("=" * 70)

    agent = CarbonAgentAfter()

    # Run some executions
    for i in range(5):
        agent.execute({
            "emissions": [{"co2e_kg": deterministic_random().randint(100, 1000)}]
        })

    # Export metrics
    metrics = agent.export_metrics_prometheus()

    print("\nPrometheus Metrics:")
    print("-" * 70)
    print(metrics)
    print("-" * 70)


def test_error_tracking():
    """Test error tracking."""
    print("\n" + "=" * 70)
    print("TEST 6: Error Tracking")
    print("=" * 70)

    agent = CarbonAgentAfter()

    print("\nRunning executions with some errors...")

    # Successful executions
    for i in range(5):
        agent.execute({
            "emissions": [{"co2e_kg": 100}]
        })

    # Failed executions
    for i in range(3):
        try:
            agent.execute({})  # Missing emissions
        except Exception:
            pass

    # Check error tracking
    history = agent.get_execution_history()
    errors = [e for e in history if not e['success']]

    print(f"\nTotal Executions: {len(history)}")
    print(f"Errors: {len(errors)}")

    if errors:
        print(f"\nError Details:")
        for error in errors[:3]:  # Show first 3
            print(f"  - Type: {error.get('error_type', 'Unknown')}")
            print(f"    Message: {error.get('error_message', 'No message')}")
            print(f"    Timestamp: {error['timestamp']}")


def test_comparison():
    """Compare before/after performance."""
    print("\n" + "=" * 70)
    print("TEST 7: Before/After Comparison")
    print("=" * 70)

    # Test data
    test_data = {
        "emissions": [
            {"fuel_type": "natural_gas", "co2e_kg": 1000},
            {"fuel_type": "electricity", "co2e_kg": 500},
        ]
    }

    # Before (no monitoring)
    agent_before = CarbonAgentBefore()
    start = time.time()
    for _ in range(100):
        agent_before.execute(test_data)
    time_before = time.time() - start

    # After (with monitoring)
    agent_after = CarbonAgentAfter()
    start = time.time()
    for _ in range(100):
        agent_after.execute(test_data)
    time_after = time.time() - start

    # Results
    print(f"\nPerformance Impact:")
    print(f"  Without monitoring: {time_before:.3f}s for 100 executions")
    print(f"  With monitoring: {time_after:.3f}s for 100 executions")
    print(f"  Overhead: {((time_after - time_before) / time_before * 100):.1f}%")
    print(f"  Per-execution overhead: {((time_after - time_before) / 100 * 1000):.2f}ms")

    # Benefits
    print(f"\nMonitoring Benefits:")
    summary = agent_after.get_performance_summary()
    print(f"  ✓ {summary['total_executions']} executions tracked")
    print(f"  ✓ Performance metrics collected")
    print(f"  ✓ Health status available")
    print(f"  ✓ Alerts configured")
    print(f"  ✓ Prometheus metrics ready")


# ============================================================================
# Production Usage Example
# ============================================================================

def production_example():
    """Show production usage pattern."""
    print("\n" + "=" * 70)
    print("PRODUCTION EXAMPLE")
    print("=" * 70)

    # Initialize agent
    agent = CarbonAgentAfter()

    print("\nAgent initialized with monitoring enabled")
    print("Ready for production traffic...")

    # Simulate production workload
    print("\nSimulating 50 production requests...")
    for i in range(50):
        try:
            result = agent.execute({
                "emissions": [
                    {"fuel_type": f"source_{j}", "co2e_kg": deterministic_random().randint(100, 5000)}
                    for j in range(deterministic_random().randint(1, 5))
                ]
            })

            if i % 10 == 0:
                print(f"  Processed {i + 1} requests...")

        except Exception as e:
            print(f"  Error on request {i + 1}: {e}")

    # Production monitoring dashboard
    print("\n" + "=" * 70)
    print("PRODUCTION MONITORING DASHBOARD")
    print("=" * 70)

    # Health status
    health = agent.health_check()
    status_emoji = {
        HealthStatus.HEALTHY: "✓",
        HealthStatus.DEGRADED: "⚠",
        HealthStatus.UNHEALTHY: "✗"
    }
    print(f"\n{status_emoji.get(health.status, '?')} Agent Status: {health.status.value.upper()}")

    # Performance metrics
    summary = agent.get_performance_summary()
    print(f"\nPerformance (Last 60 min):")
    print(f"  Requests: {summary['total_executions']}")
    print(f"  Success Rate: {summary['success_rate']:.1%}")
    print(f"  Avg Latency: {summary['latency']['avg_ms']:.0f}ms")
    print(f"  P95 Latency: {summary['latency']['p95_ms']:.0f}ms")

    # Cost tracking
    print(f"\nCost Analysis:")
    print(f"  Total: ${summary['cost']['total_usd']:.2f}")
    print(f"  Per Request: ${summary['cost']['avg_usd']:.3f}")

    # Active alerts
    alerts = agent.get_alerts(unresolved_only=True)
    print(f"\nActive Alerts: {len(alerts)}")
    if alerts:
        for alert in alerts[:3]:
            print(f"  [{alert.severity.value}] {alert.message}")

    # Recommendations
    print(f"\nRecommendations:")
    if summary['success_rate'] < 0.95:
        print(f"  ⚠ Success rate below 95% - investigate errors")
    if summary['latency']['p95_ms'] > 3000:
        print(f"  ⚠ P95 latency above 3s - consider optimization")
    if summary['cost']['total_usd'] > 10:
        print(f"  ⚠ High costs - review usage patterns")
    if not alerts and summary['success_rate'] > 0.99:
        print(f"  ✓ All systems nominal - no action needed")


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("GREENLANG OPERATIONAL MONITORING - INTEGRATION EXAMPLE")
    print("=" * 70)

    # Run all tests
    test_basic_execution()
    test_health_checks()
    test_performance_summary()
    test_alerting()
    test_prometheus_metrics()
    test_error_tracking()
    test_comparison()
    production_example()

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Review the monitoring data collected")
    print("  2. Integrate monitoring into your agents")
    print("  3. Configure alerts for your production environment")
    print("  4. Setup Prometheus scraping of /metrics endpoint")
    print("  5. Create dashboards for operational visibility")


if __name__ == "__main__":
    main()
