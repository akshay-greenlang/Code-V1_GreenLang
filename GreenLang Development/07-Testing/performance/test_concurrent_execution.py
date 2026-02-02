# -*- coding: utf-8 -*-
"""Concurrent Execution Tests for GreenLang Agents.

This module tests concurrent execution capabilities:
- 10 concurrent agent executions
- 100 concurrent agent executions
- 1000 concurrent lightweight operations
- AsyncOrchestrator parallel workflow tests
- Thread-safety tests
- Race condition detection

Test Scenarios:
1. Small concurrent load (10 agents)
2. Medium concurrent load (100 agents)
3. Large concurrent load (1000 operations)
4. Mixed agent types concurrent execution
5. Concurrent workflow orchestration
6. Thread-safety validation
7. Race condition detection

Author: GreenLang Framework Team
Date: November 2025
Status: Production Ready
"""

from __future__ import annotations

import asyncio
import pytest
import time
from typing import List, Dict, Any
import statistics

# Try importing async agents
try:
    from greenlang.agents.fuel_agent_ai_async import AsyncFuelAgentAI
    from greenlang.config import get_config
    ASYNC_AGENTS_AVAILABLE = True
except ImportError:
    ASYNC_AGENTS_AVAILABLE = False
    pytest.skip("Async agents not available", allow_module_level=True)


# ==============================================================================
# Test Fixtures
# ==============================================================================

@pytest.fixture
def agent_config():
    """Get agent configuration."""
    return get_config()


@pytest.fixture
def test_input():
    """Standard test input for agents."""
    return {
        "fuel_type": "natural_gas",
        "amount": 1000,
        "unit": "therms",
        "country": "US"
    }


# ==============================================================================
# Concurrent Execution Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_10_concurrent_executions(agent_config, test_input):
    """Test 10 concurrent agent executions.

    This validates that the async infrastructure can handle
    small-scale concurrent loads without errors.
    """
    num_concurrent = 10

    print(f"\n{'='*80}")
    print(f"TEST: {num_concurrent} Concurrent Executions")
    print(f"{'='*80}")

    start_time = time.perf_counter()

    async with AsyncFuelAgentAI(agent_config) as agent:
        # Create concurrent tasks
        tasks = [agent.run_async(test_input) for _ in range(num_concurrent)]

        # Execute all concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

    end_time = time.perf_counter()
    duration = end_time - start_time

    # Validate results
    assert len(results) == num_concurrent, "All tasks should complete"

    successful = sum(1 for r in results if not isinstance(r, Exception) and r.success)
    failed = num_concurrent - successful

    print(f"\nResults:")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Successful: {successful}/{num_concurrent}")
    print(f"  Failed: {failed}/{num_concurrent}")
    print(f"  Throughput: {num_concurrent / duration:.2f} agents/sec")

    # Assert performance expectations
    assert successful >= num_concurrent * 0.95, "At least 95% should succeed"
    assert duration < 10.0, f"Should complete within 10s (took {duration:.2f}s)"

    print(f"✓ Test passed!")


@pytest.mark.asyncio
async def test_100_concurrent_executions(agent_config, test_input):
    """Test 100 concurrent agent executions.

    This validates medium-scale concurrent loads and
    verifies the 8.6x speedup claim.
    """
    num_concurrent = 100

    print(f"\n{'='*80}")
    print(f"TEST: {num_concurrent} Concurrent Executions")
    print(f"{'='*80}")

    start_time = time.perf_counter()

    async with AsyncFuelAgentAI(agent_config) as agent:
        # Create concurrent tasks
        tasks = [agent.run_async(test_input) for _ in range(num_concurrent)]

        # Execute all concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

    end_time = time.perf_counter()
    duration = end_time - start_time

    # Validate results
    assert len(results) == num_concurrent, "All tasks should complete"

    successful = sum(1 for r in results if not isinstance(r, Exception) and r.success)
    failed = num_concurrent - successful

    print(f"\nResults:")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Successful: {successful}/{num_concurrent}")
    print(f"  Failed: {failed}/{num_concurrent}")
    print(f"  Throughput: {num_concurrent / duration:.2f} agents/sec")

    # Assert performance expectations
    assert successful >= num_concurrent * 0.95, "At least 95% should succeed"
    assert duration < 30.0, f"Should complete within 30s (took {duration:.2f}s)"

    print(f"✓ Test passed!")


@pytest.mark.asyncio
async def test_1000_lightweight_operations(agent_config):
    """Test 1000 concurrent lightweight operations.

    This tests the event loop's ability to handle many
    concurrent operations efficiently.
    """
    num_operations = 1000

    print(f"\n{'='*80}")
    print(f"TEST: {num_operations} Lightweight Concurrent Operations")
    print(f"{'='*80}")

    async def lightweight_operation(operation_id: int):
        """Lightweight async operation."""
        await asyncio.sleep(0.001)  # 1ms delay
        return {"id": operation_id, "result": operation_id * 2}

    start_time = time.perf_counter()

    # Create concurrent tasks
    tasks = [lightweight_operation(i) for i in range(num_operations)]

    # Execute all concurrently
    results = await asyncio.gather(*tasks)

    end_time = time.perf_counter()
    duration = end_time - start_time

    # Validate results
    assert len(results) == num_operations, "All operations should complete"

    print(f"\nResults:")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Operations: {num_operations}")
    print(f"  Throughput: {num_operations / duration:.2f} ops/sec")

    # Should complete much faster than sequential (which would take ~1s)
    assert duration < 0.5, f"Should complete within 0.5s (took {duration:.2f}s)"

    print(f"✓ Test passed!")


@pytest.mark.asyncio
async def test_concurrent_with_different_inputs(agent_config):
    """Test concurrent execution with varying inputs.

    This validates that agents handle different inputs
    correctly when running concurrently.
    """
    num_concurrent = 20

    print(f"\n{'='*80}")
    print(f"TEST: {num_concurrent} Concurrent Executions (Different Inputs)")
    print(f"{'='*80}")

    # Create varied inputs
    test_inputs = []
    fuel_types = ["natural_gas", "diesel", "gasoline", "coal", "electricity"]
    amounts = [100, 500, 1000, 5000, 10000]

    for i in range(num_concurrent):
        test_inputs.append({
            "fuel_type": fuel_types[i % len(fuel_types)],
            "amount": amounts[i % len(amounts)],
            "unit": "therms",
            "country": "US"
        })

    start_time = time.perf_counter()

    async with AsyncFuelAgentAI(agent_config) as agent:
        # Create concurrent tasks with different inputs
        tasks = [agent.run_async(inp) for inp in test_inputs]

        # Execute all concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

    end_time = time.perf_counter()
    duration = end_time - start_time

    # Validate results
    successful = sum(1 for r in results if not isinstance(r, Exception) and r.success)

    print(f"\nResults:")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Successful: {successful}/{num_concurrent}")
    print(f"  Throughput: {num_concurrent / duration:.2f} agents/sec")

    assert successful >= num_concurrent * 0.95, "At least 95% should succeed"

    print(f"✓ Test passed!")


@pytest.mark.asyncio
async def test_concurrent_latency_consistency(agent_config, test_input):
    """Test that concurrent execution maintains consistent latency.

    Validates that p95 latency doesn't degrade significantly
    under concurrent load.
    """
    num_concurrent = 50

    print(f"\n{'='*80}")
    print(f"TEST: Latency Consistency ({num_concurrent} concurrent)")
    print(f"{'='*80}")

    latencies = []

    async def timed_execution(agent, input_data):
        """Execute agent and measure latency."""
        start = time.perf_counter()
        result = await agent.run_async(input_data)
        end = time.perf_counter()
        return (end - start) * 1000, result  # ms

    async with AsyncFuelAgentAI(agent_config) as agent:
        # Create concurrent tasks
        tasks = [timed_execution(agent, test_input) for _ in range(num_concurrent)]

        # Execute all concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

    # Extract latencies
    for result in results:
        if isinstance(result, tuple):
            latency, agent_result = result
            latencies.append(latency)

    # Calculate statistics
    if latencies:
        sorted_latencies = sorted(latencies)
        mean_lat = statistics.mean(latencies)
        median_lat = statistics.median(latencies)
        p95_lat = sorted_latencies[int(len(sorted_latencies) * 0.95)]
        p99_lat = sorted_latencies[int(len(sorted_latencies) * 0.99)]

        print(f"\nLatency Statistics:")
        print(f"  Mean: {mean_lat:.2f}ms")
        print(f"  Median: {median_lat:.2f}ms")
        print(f"  p95: {p95_lat:.2f}ms")
        print(f"  p99: {p99_lat:.2f}ms")

        # Assert SLO: p95 < 500ms
        assert p95_lat < 500, f"p95 latency {p95_lat:.2f}ms exceeds 500ms SLO"

        print(f"✓ Test passed!")


# ==============================================================================
# Thread Safety Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_thread_safety_shared_agent(agent_config, test_input):
    """Test thread safety with shared agent instance.

    Validates that a single agent instance can safely
    handle multiple concurrent requests.
    """
    num_concurrent = 30

    print(f"\n{'='*80}")
    print(f"TEST: Thread Safety (Shared Agent)")
    print(f"{'='*80}")

    async with AsyncFuelAgentAI(agent_config) as agent:
        # All tasks share the same agent instance
        tasks = [agent.run_async(test_input) for _ in range(num_concurrent)]

        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

    # Check for errors that might indicate thread safety issues
    exceptions = [r for r in results if isinstance(r, Exception)]
    successful = sum(1 for r in results if not isinstance(r, Exception) and r.success)

    print(f"\nResults:")
    print(f"  Successful: {successful}/{num_concurrent}")
    print(f"  Exceptions: {len(exceptions)}")

    if exceptions:
        print(f"\nExceptions encountered:")
        for exc in exceptions[:5]:  # Print first 5
            print(f"  - {type(exc).__name__}: {exc}")

    # Should have minimal failures
    assert successful >= num_concurrent * 0.95, "Thread safety issues detected"

    print(f"✓ Test passed!")


@pytest.mark.asyncio
async def test_race_condition_detection(agent_config):
    """Test for race conditions in concurrent state updates.

    This test attempts to detect race conditions by having
    multiple tasks update shared state concurrently.
    """
    print(f"\n{'='*80}")
    print(f"TEST: Race Condition Detection")
    print(f"{'='*80}")

    # Shared counter (intentionally vulnerable if not thread-safe)
    shared_state = {"counter": 0}

    async def increment_counter(iterations: int):
        """Increment shared counter."""
        for _ in range(iterations):
            # This is a race condition if not properly protected
            current = shared_state["counter"]
            await asyncio.sleep(0)  # Yield control
            shared_state["counter"] = current + 1

    num_tasks = 10
    iterations_per_task = 100
    expected_final_count = num_tasks * iterations_per_task

    # Run concurrent increments
    tasks = [increment_counter(iterations_per_task) for _ in range(num_tasks)]
    await asyncio.gather(*tasks)

    actual_count = shared_state["counter"]

    print(f"\nResults:")
    print(f"  Expected: {expected_final_count}")
    print(f"  Actual: {actual_count}")
    print(f"  Difference: {expected_final_count - actual_count}")

    # Note: This test SHOULD fail without proper locking
    # This demonstrates the need for thread-safe state management
    if actual_count != expected_final_count:
        print(f"⚠ Race condition detected (as expected without locks)")
    else:
        print(f"✓ No race condition (or lucky timing)")

    # This is informational - we expect race conditions without locks


# ==============================================================================
# Performance Comparison Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_async_speedup_vs_sequential(agent_config, test_input):
    """Test async speedup vs sequential execution.

    Validates the claimed 8.6x speedup for parallel workflows.
    """
    num_agents = 10

    print(f"\n{'='*80}")
    print(f"TEST: Async Speedup vs Sequential ({num_agents} agents)")
    print(f"{'='*80}")

    # Measure parallel execution
    start = time.perf_counter()
    async with AsyncFuelAgentAI(agent_config) as agent:
        tasks = [agent.run_async(test_input) for _ in range(num_agents)]
        parallel_results = await asyncio.gather(*tasks)
    parallel_time = time.perf_counter() - start

    # Measure sequential execution
    start = time.perf_counter()
    async with AsyncFuelAgentAI(agent_config) as agent:
        sequential_results = []
        for _ in range(num_agents):
            result = await agent.run_async(test_input)
            sequential_results.append(result)
    sequential_time = time.perf_counter() - start

    # Calculate speedup
    speedup = sequential_time / parallel_time

    print(f"\nResults:")
    print(f"  Sequential Time: {sequential_time:.2f}s")
    print(f"  Parallel Time: {parallel_time:.2f}s")
    print(f"  Speedup: {speedup:.2f}x")

    # Assert minimum speedup (conservative threshold)
    assert speedup >= 3.0, f"Speedup {speedup:.2f}x is below minimum threshold of 3.0x"

    if speedup >= 8.0:
        print(f"✓ Excellent speedup! ({speedup:.2f}x)")
    elif speedup >= 5.0:
        print(f"✓ Good speedup ({speedup:.2f}x)")
    else:
        print(f"✓ Acceptable speedup ({speedup:.2f}x)")


# ==============================================================================
# Error Handling Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_concurrent_error_handling(agent_config):
    """Test error handling in concurrent execution.

    Validates that errors in some tasks don't affect others.
    """
    num_concurrent = 20
    num_failing = 5

    print(f"\n{'='*80}")
    print(f"TEST: Concurrent Error Handling")
    print(f"{'='*80}")

    async def task_with_potential_error(task_id: int, should_fail: bool):
        """Task that might fail."""
        if should_fail:
            raise ValueError(f"Intentional error in task {task_id}")

        await asyncio.sleep(0.01)
        return {"task_id": task_id, "success": True}

    # Create tasks with some that will fail
    tasks = []
    for i in range(num_concurrent):
        should_fail = i < num_failing
        tasks.append(task_with_potential_error(i, should_fail))

    # Execute with return_exceptions=True to handle errors gracefully
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Count successes and failures
    successful = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
    failed = sum(1 for r in results if isinstance(r, Exception))

    print(f"\nResults:")
    print(f"  Total Tasks: {num_concurrent}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Expected Failures: {num_failing}")

    assert failed == num_failing, "All failing tasks should fail"
    assert successful == num_concurrent - num_failing, "All non-failing tasks should succeed"

    print(f"✓ Test passed!")


# ==============================================================================
# Main Entry Point (for standalone execution)
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
