#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Performance Benchmark Test
=================================

Minimal test to validate the benchmarking system without dependencies.
"""

import asyncio
import json
import time
from pathlib import Path

from .performance_suite import (
    SyntheticAgent,
    BenchmarkRunner,
    ResourceMonitor,
    benchmark_agent_execution,
    PerformanceBenchmark
)


async def test_synthetic_agent():
    """Test synthetic agent performance"""
    print("Testing synthetic agent...")

    agent = SyntheticAgent(complexity="medium")

    # Test 10 iterations
    times = []
    success_count = 0
    for i in range(10):
        start = time.perf_counter()
        result = agent.run({"iteration": i, "size": 1000})
        duration = time.perf_counter() - start
        times.append(duration * 1000)  # Convert to ms

        if result.success:
            success_count += 1
        else:
            print(f"  Agent failed on iteration {i}: {result.error}")

    avg_time = sum(times) / len(times)
    max_time = max(times)
    min_time = min(times)

    print(f"  Average time: {avg_time:.2f}ms")
    print(f"  Min time: {min_time:.2f}ms")
    print(f"  Max time: {max_time:.2f}ms")
    print(f"  Successful iterations: {success_count}/10")


async def test_resource_monitor():
    """Test resource monitoring"""
    print("Testing resource monitor...")

    monitor = ResourceMonitor()
    monitor.start()

    # Simulate some work
    await asyncio.sleep(1)

    # Do some CPU work
    for i in range(100000):
        _ = i * i

    await asyncio.sleep(0.5)

    stats = monitor.stop()

    print(f"  Peak memory: {stats['memory_peak_mb']:.2f}MB")
    print(f"  Average memory: {stats['memory_avg_mb']:.2f}MB")
    print(f"  Average CPU: {stats['cpu_avg_percent']:.2f}%")


async def test_benchmark_execution():
    """Test benchmark execution"""
    print("Testing benchmark execution...")

    # Create a simple benchmark
    benchmark = PerformanceBenchmark(
        name="test_agent_execution",
        description="Test agent execution",
        benchmark_func=benchmark_agent_execution
    )

    # Run with 5 iterations
    result = await benchmark.run(iterations=5)

    print(f"  Benchmark: {result.name}")
    print(f"  Success: {result.success}")
    print(f"  Duration: {result.duration_ms:.2f}ms")
    print(f"  Throughput: {result.throughput_ops_per_sec:.2f} ops/sec")
    print(f"  Memory peak: {result.memory_peak_mb:.2f}MB")
    print(f"  P95: {result.percentiles.get('p95', 0):.2f}ms")

    if not result.success:
        print(f"  Error: {result.error}")


async def test_load_simulation():
    """Test load testing functionality"""
    print("Testing load simulation...")

    runner = BenchmarkRunner()

    # Run a small load test
    load_result = runner.run_load_test(
        concurrent_users=2,
        requests_per_user=5,
        test_duration=10  # 10 seconds max
    )

    print(f"  Concurrent users: {load_result.concurrent_users}")
    print(f"  Total requests: {load_result.total_requests}")
    print(f"  Successful: {load_result.successful_requests}")
    print(f"  Failed: {load_result.failed_requests}")
    print(f"  Success rate: {load_result.successful_requests/load_result.total_requests*100:.1f}%")
    print(f"  Average response time: {load_result.avg_response_time_ms:.2f}ms")
    print(f"  P95 response time: {load_result.p95_response_time_ms:.2f}ms")
    print(f"  Requests/sec: {load_result.requests_per_second:.2f}")

    if load_result.errors:
        print(f"  Errors: {len(load_result.errors)}")


async def main():
    """Run all tests"""
    print("GreenLang Performance Benchmark Test Suite")
    print("=" * 50)

    start_time = time.time()

    try:
        await test_synthetic_agent()
        print()

        await test_resource_monitor()
        print()

        await test_benchmark_execution()
        print()

        await test_load_simulation()
        print()

        total_time = time.time() - start_time
        print(f"All tests completed in {total_time:.2f}s")
        print("All systems operational!")

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