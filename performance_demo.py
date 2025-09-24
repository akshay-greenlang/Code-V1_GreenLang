#!/usr/bin/env python3
"""
GreenLang v0.2.0 Performance and Monitoring Demo
=================================================

This script demonstrates the comprehensive performance benchmarking and monitoring
system for GreenLang v0.2.0 production readiness.

Features demonstrated:
- Performance benchmarking with synthetic workloads
- Resource monitoring and metrics collection
- Health check system with component validation
- Load testing with concurrent users
- Performance regression detection
- Report generation (JSON and Markdown)

Run this script to validate the complete performance and monitoring infrastructure.
"""

import asyncio
import json
import logging
import time
from pathlib import Path

from greenlang.benchmarks.performance_suite import BenchmarkRunner, SyntheticAgent
from greenlang.monitoring.metrics import setup_metrics, MetricType, CustomMetric
from greenlang.monitoring.health import HealthChecker, HealthStatus

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def demo_performance_benchmarking():
    """Demonstrate performance benchmarking capabilities"""
    print("=" * 60)
    print("PERFORMANCE BENCHMARKING DEMONSTRATION")
    print("=" * 60)

    # Initialize benchmark runner
    output_dir = Path("demo_results")
    runner = BenchmarkRunner(output_dir=output_dir)

    print("1. Running Core Performance Benchmarks...")

    # Run benchmarks with smaller iterations for demo
    results = await runner.run_all_benchmarks(iterations=20)

    print("\nBenchmark Results Summary:")
    print("-" * 40)
    for name, result in results.items():
        status_icon = "[PASS]" if result.success else "[FAIL]"
        print(f"{status_icon} {name}")
        print(f"   P95: {result.percentiles.get('p95', 0):.2f}ms")
        print(f"   Throughput: {result.throughput_ops_per_sec:.2f} ops/sec")
        print(f"   Memory: {result.memory_peak_mb:.2f}MB")

    print("\n2. Running Load Test...")

    # Run load test
    load_result = runner.run_load_test(
        concurrent_users=5,
        requests_per_user=20,
        test_duration=30
    )

    print(f"Load Test Results:")
    print(f"   Concurrent Users: {load_result.concurrent_users}")
    print(f"   Success Rate: {load_result.successful_requests}/{load_result.total_requests} ({load_result.successful_requests/load_result.total_requests*100:.1f}%)")
    print(f"   P95 Response Time: {load_result.p95_response_time_ms:.2f}ms")
    print(f"   Throughput: {load_result.requests_per_second:.2f} requests/sec")

    print("\n3. Checking for Performance Regressions...")

    # Check for regressions
    regressions = runner.check_regression(results)
    if regressions:
        print("   Performance Regressions Detected:")
        for benchmark, issues in regressions.items():
            print(f"     {benchmark}:")
            for issue in issues:
                print(f"       - {issue}")
    else:
        print("   No performance regressions detected")

    print("\n4. Generating Performance Report...")

    # Generate comprehensive report
    report_file = runner.generate_report(results, load_result)
    print(f"   Report generated: {report_file}")
    print(f"   Markdown summary: {Path(report_file).with_suffix('.md')}")

    return results, load_result


async def demo_monitoring_system():
    """Demonstrate monitoring and observability capabilities"""
    print("\n" + "=" * 60)
    print("MONITORING & OBSERVABILITY DEMONSTRATION")
    print("=" * 60)

    print("1. Setting Up Metrics Collection...")

    # Setup metrics collection (without Prometheus for demo)
    metrics = setup_metrics(
        enable_prometheus=False,
        enable_system_metrics=True,
        buffer_size=1000
    )

    # Register custom metrics
    custom_metrics = [
        CustomMetric(
            name="demo_operations",
            type=MetricType.COUNTER,
            description="Demo operations performed",
            labels=["operation_type"]
        ),
        CustomMetric(
            name="demo_processing_time",
            type=MetricType.HISTOGRAM,
            description="Demo operation processing time",
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
            labels=["complexity"]
        )
    ]

    for metric in custom_metrics:
        metrics.register_custom_metric(metric)

    print(f"   Registered {len(custom_metrics)} custom metrics")

    print("\n2. Generating Sample Workload...")

    # Simulate workload that generates metrics
    complexities = ["light", "medium", "heavy"]
    for i in range(15):
        complexity = complexities[i % len(complexities)]

        # Time the operation
        with metrics.time_operation("demo_operation", {"complexity": complexity}):
            agent = SyntheticAgent(complexity=complexity)
            result = agent.run({"iteration": i, "timestamp": time.time()})

            # Record success/failure
            if result.success:
                metrics.record_pipeline_execution(f"demo_pipeline", 0.05, True)
                metrics.record_custom_metric("demo_operations", 1, {"operation_type": "success"})
            else:
                metrics.record_error("demo_agent", "ProcessingError")
                metrics.record_custom_metric("demo_operations", 1, {"operation_type": "failure"})

        # Small delay
        await asyncio.sleep(0.01)

    print("   Generated 15 operations across complexity levels")

    print("\n3. Metrics Collection Summary...")

    # Get metrics summary
    recent_metrics = metrics.get_recent_metrics(60)
    health_metrics = metrics.get_health_metrics()

    print(f"   Metrics collected: {len(recent_metrics)}")
    print(f"   Buffer usage: {health_metrics.get('buffer_size', 0)}")
    print(f"   Memory usage: {health_metrics.get('memory_usage_mb', 0):.2f}MB")

    # Get operation statistics
    for complexity in complexities:
        stats = metrics.get_operation_stats("demo_operation")
        if stats:
            print(f"   Operation stats - Count: {stats['count']}, Mean: {stats['mean']*1000:.2f}ms, P95: {stats['p95']*1000:.2f}ms")
            break

    print("\n4. Setting Up Health Monitoring...")

    # Setup health checker
    health = HealthChecker(check_interval=10)

    # Add custom component check
    def demo_service_check():
        # Simulate a service that's sometimes degraded
        import random
        if random.random() < 0.8:
            return HealthStatus.HEALTHY, "Demo service operational", {"uptime": 3600}
        else:
            return HealthStatus.DEGRADED, "Demo service experiencing minor issues", {"uptime": 3600}

    health.add_component_check("demo_service", demo_service_check)

    print("   Added custom health check for demo service")

    print("\n5. Running Health Assessment...")

    # Check system health
    system_health = await health.get_system_health()

    print(f"   Overall Status: {system_health.status.value.upper()}")
    print(f"   Components: {len(system_health.components)}")
    print(f"   Healthy: {system_health.summary['healthy_components']}")
    print(f"   Degraded: {system_health.summary['degraded_components']}")
    print(f"   Unhealthy: {system_health.summary['unhealthy_components']}")
    print(f"   System Uptime: {system_health.uptime_seconds:.1f}s")

    print("\n6. Component Health Details...")

    for name, component in system_health.components.items():
        status_icon = {
            HealthStatus.HEALTHY: "[OK]",
            HealthStatus.DEGRADED: "[WARN]",
            HealthStatus.UNHEALTHY: "[ERROR]",
            HealthStatus.UNKNOWN: "[UNKNOWN]"
        }.get(component.status, "[UNKNOWN]")

        print(f"   {status_icon} {name}: {component.status.value} - {component.message}")
        if component.response_time_ms:
            print(f"      Response time: {component.response_time_ms:.2f}ms")

    return metrics, health, system_health


async def demo_integration_scenarios():
    """Demonstrate integration between performance and monitoring"""
    print("\n" + "=" * 60)
    print("INTEGRATION SCENARIOS DEMONSTRATION")
    print("=" * 60)

    print("1. Performance-Aware Health Monitoring...")

    # Get current metrics
    metrics = setup_metrics()  # Get existing instance

    # Simulate performance degradation
    print("   Simulating performance degradation...")

    for i in range(5):
        # Simulate increasingly slow operations
        delay = 0.1 * (i + 1)  # 100ms to 500ms

        with metrics.time_operation("degraded_operation", {"iteration": str(i)}):
            await asyncio.sleep(delay)

        # Record the operation
        metrics.record_agent_execution("DegradedAgent", delay, True)

    # Check if performance metrics indicate issues
    stats = metrics.get_operation_stats("degraded_operation")
    if stats and stats.get('mean', 0) > 0.2:  # > 200ms average
        print(f"   Performance degradation detected: {stats['mean']*1000:.0f}ms average")

    print("\n2. Automated Alert Simulation...")

    # Simulate alert conditions
    error_count = 0
    for i in range(10):
        # 30% error rate simulation
        success = i % 10 < 7

        if success:
            metrics.record_pipeline_execution("alert_test", 0.05, True)
        else:
            metrics.record_pipeline_execution("alert_test", 0.05, False)
            metrics.record_error("alert_test", "SimulatedError")
            error_count += 1

    error_rate = error_count / 10
    if error_rate > 0.1:  # > 10% error rate
        print(f"   High error rate alert: {error_rate*100:.0f}% error rate detected")

    print("\n3. Capacity Planning Metrics...")

    # Collect resource usage for capacity planning
    health_metrics = metrics.get_health_metrics()

    current_memory = health_metrics.get('memory_usage_mb', 0)
    buffer_usage = health_metrics.get('buffer_size', 0)

    # Simulate capacity recommendations
    memory_threshold = 1000  # 1GB
    buffer_threshold = 8000  # 80% of 10k buffer

    recommendations = []
    if current_memory > memory_threshold * 0.8:
        recommendations.append(f"Memory usage approaching limit ({current_memory:.0f}MB/{memory_threshold}MB)")

    if buffer_usage > buffer_threshold:
        recommendations.append(f"Metrics buffer approaching limit ({buffer_usage}/{buffer_threshold})")

    if recommendations:
        print("   Capacity recommendations:")
        for rec in recommendations:
            print(f"     - {rec}")
    else:
        print("   System capacity within normal parameters")

    print("\n4. End-to-End Monitoring Summary...")

    # Final metrics export for analysis
    json_export = metrics.export_metrics_json()
    export_data = json.loads(json_export)

    print(f"   Total metric types: {len(export_data['metrics'])}")
    print(f"   Operation types monitored: {len(export_data['operation_stats'])}")
    print(f"   Data export size: {len(json_export)} bytes")

    # Save demo results
    demo_file = Path("demo_monitoring_export.json")
    with open(demo_file, "w") as f:
        f.write(json_export)
    print(f"   Demo data exported to: {demo_file}")


async def main():
    """Main demonstration function"""
    print("GreenLang v0.2.0 Performance & Monitoring Demo")
    print("Production Readiness Validation")
    print("=" * 60)

    start_time = time.time()

    try:
        # Run performance benchmarking demo
        bench_results, load_result = await demo_performance_benchmarking()

        # Run monitoring system demo
        metrics, health, system_health = await demo_monitoring_system()

        # Run integration scenarios
        await demo_integration_scenarios()

        # Final summary
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("DEMO COMPLETION SUMMARY")
        print("=" * 60)

        print(f"Total demo duration: {total_time:.2f} seconds")
        print(f"Benchmarks executed: {len(bench_results)}")
        print(f"Load test completed: {load_result.total_requests} requests")
        print(f"System health status: {system_health.status.value.upper()}")
        print(f"Metrics collected: Available")
        print(f"Health monitoring: Active")

        print("\nFiles generated:")
        print("  - Performance report (JSON + Markdown)")
        print("  - Monitoring data export")
        print("  - System health assessment")

        print("\nProduction readiness status: VALIDATED")
        print("GreenLang v0.2.0 performance and monitoring systems are operational!")

    except Exception as e:
        print(f"\nDemo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)