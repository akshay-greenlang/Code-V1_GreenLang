#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Monitoring System
======================

Test the monitoring and health check system.
"""

import asyncio
import json
import time
from pathlib import Path

from .metrics import setup_metrics, get_metrics, MetricType, CustomMetric
from .health import HealthChecker, HealthStatus


async def test_metrics_collection():
    """Test metrics collection"""
    print("Testing metrics collection...")

    # Setup metrics without Prometheus for testing
    metrics = setup_metrics(
        enable_prometheus=False,
        enable_system_metrics=False  # Disable for cleaner test
    )

    # Record some test metrics
    metrics.record_metric("test_counter", 1, {"type": "test"})
    metrics.record_metric("test_gauge", 42.5)

    # Register and use custom metric
    custom_metric = CustomMetric(
        name="custom_operations",
        type=MetricType.COUNTER,
        description="Custom operations performed",
        labels=["operation"]
    )
    metrics.register_custom_metric(custom_metric)
    metrics.record_custom_metric("custom_operations", 1, {"operation": "test"})

    # Test timing operations
    with metrics.time_operation("test_operation", {"test": "true"}):
        await asyncio.sleep(0.1)  # 100ms operation

    # Test pipeline/agent metrics
    metrics.record_pipeline_execution("test_pipeline", 0.05, True)
    metrics.record_agent_execution("TestAgent", 0.02, True)
    metrics.record_agent_execution("TestAgent", 0.03, False)  # One failure

    # Get operation stats
    stats = metrics.get_operation_stats("test_operation")
    print(f"  Operation stats: count={stats.get('count', 0)}, mean={stats.get('mean', 0):.3f}s")

    # Get recent metrics
    recent = metrics.get_recent_metrics(60)
    print(f"  Recent metrics collected: {len(recent)}")

    # Export metrics
    json_export = metrics.export_metrics_json()
    export_data = json.loads(json_export)
    print(f"  Exported {len(export_data['metrics'])} metric types")

    print("  Metrics collection test passed!")


async def test_health_checks():
    """Test health check system"""
    print("Testing health checks...")

    health = HealthChecker(check_interval=5)

    # Add a custom component check
    def custom_component_check():
        return HealthStatus.HEALTHY, "Custom component is working", {"uptime": 100}

    health.add_component_check("custom_service", custom_component_check)

    # Check individual components
    components_checked = 0
    for component_name in health.component_checks:
        result = await health.check_component(component_name)
        print(f"    {component_name}: {result.status.value} - {result.message}")
        components_checked += 1

    # Get overall system health
    system_health = await health.get_system_health()

    print(f"  Overall status: {system_health.status.value}")
    print(f"  Components checked: {components_checked}")
    print(f"  Healthy components: {system_health.summary['healthy_components']}")
    print(f"  System uptime: {system_health.uptime_seconds:.1f}s")

    # Test health metrics
    health_metrics = get_metrics().get_health_metrics() if get_metrics() else {}
    if health_metrics:
        print(f"  Health metrics available: {len(health_metrics)} metrics")

    print("  Health checks test passed!")


async def test_integration():
    """Test metrics and health integration"""
    print("Testing metrics and health integration...")

    # Setup metrics with health monitoring
    metrics = get_metrics()
    if not metrics:
        metrics = setup_metrics(enable_system_metrics=False)

    health = HealthChecker()

    # Simulate some workload that generates metrics
    for i in range(5):
        with metrics.time_operation("integration_test", {"iteration": str(i)}):
            # Simulate varying work
            await asyncio.sleep(0.01 * (i + 1))

        metrics.record_pipeline_execution(f"test_pipeline_{i}", 0.01 * (i + 1), True)

    # Check health after workload
    system_health = await health.get_system_health()
    health_metrics = metrics.get_health_metrics()

    print(f"  System health after workload: {system_health.status.value}")
    print(f"  Metrics buffer size: {health_metrics.get('buffer_size', 0)}")
    print(f"  Operations completed: 5")

    # Test error recording
    metrics.record_error("integration_test", "TestError")

    print("  Integration test passed!")


async def main():
    """Run all monitoring tests"""
    print("GreenLang Monitoring System Test")
    print("=" * 40)

    start_time = time.time()

    try:
        await test_metrics_collection()
        print()

        await test_health_checks()
        print()

        await test_integration()
        print()

        total_time = time.time() - start_time
        print(f"All monitoring tests completed in {total_time:.2f}s")
        print("Monitoring system operational!")

    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)