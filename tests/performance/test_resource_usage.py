# -*- coding: utf-8 -*-
"""Resource Usage Tests for GreenLang Agents.

This module tests resource consumption under load:
- CPU usage measurement
- Memory usage (RSS, VMS, peak)
- Disk I/O measurement
- Network I/O measurement
- File descriptor tracking
- Thread/process count tracking
- Resource leak detection

Test Scenarios:
1. CPU usage under increasing load
2. Memory usage patterns and leak detection
3. Resource cleanup validation
4. File descriptor leak detection
5. Thread pool efficiency

Author: GreenLang Framework Team
Date: November 2025
Status: Production Ready
"""

from __future__ import annotations

import asyncio
import gc
import os
import pytest
import time
from typing import Dict, List, Any

# Resource monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    pytest.skip("psutil not available", allow_module_level=True)

# Try importing async agents
try:
    from greenlang.agents.fuel_agent_ai_async import AsyncFuelAgentAI
    from greenlang.config import get_config
    ASYNC_AGENTS_AVAILABLE = True
except ImportError:
    ASYNC_AGENTS_AVAILABLE = False
    pytest.skip("Async agents not available", allow_module_level=True)


# ==============================================================================
# Resource Monitor
# ==============================================================================

class ResourceMonitor:
    """Monitor system resources during test execution."""

    def __init__(self):
        """Initialize resource monitor."""
        self.process = psutil.Process(os.getpid())
        self.samples: List[Dict[str, Any]] = []

    def take_snapshot(self) -> Dict[str, Any]:
        """Take a snapshot of current resource usage."""
        try:
            with self.process.oneshot():
                snapshot = {
                    "timestamp": time.time(),
                    "cpu_percent": self.process.cpu_percent(),
                    "memory_rss_mb": self.process.memory_info().rss / 1024 / 1024,
                    "memory_vms_mb": self.process.memory_info().vms / 1024 / 1024,
                    "num_threads": self.process.num_threads(),
                    "num_fds": self.process.num_fds() if hasattr(self.process, "num_fds") else 0,
                    "io_counters": self.process.io_counters() if hasattr(self.process, "io_counters") else None,
                }
            self.samples.append(snapshot)
            return snapshot
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return {}

    def get_peak_memory_mb(self) -> float:
        """Get peak memory usage."""
        if not self.samples:
            return 0
        return max(s["memory_rss_mb"] for s in self.samples)

    def get_avg_cpu_percent(self) -> float:
        """Get average CPU usage."""
        if not self.samples:
            return 0
        cpu_samples = [s["cpu_percent"] for s in self.samples if s["cpu_percent"] > 0]
        return sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0

    def get_memory_growth_mb(self) -> float:
        """Get memory growth from first to last sample."""
        if len(self.samples) < 2:
            return 0
        return self.samples[-1]["memory_rss_mb"] - self.samples[0]["memory_rss_mb"]

    def print_summary(self):
        """Print resource usage summary."""
        if not self.samples:
            print("No samples collected")
            return

        print(f"\n{'='*80}")
        print("RESOURCE USAGE SUMMARY")
        print(f"{'='*80}")
        print(f"Samples Collected: {len(self.samples)}")
        print(f"\nMemory:")
        print(f"  Initial RSS: {self.samples[0]['memory_rss_mb']:.2f} MB")
        print(f"  Final RSS: {self.samples[-1]['memory_rss_mb']:.2f} MB")
        print(f"  Peak RSS: {self.get_peak_memory_mb():.2f} MB")
        print(f"  Growth: {self.get_memory_growth_mb():.2f} MB")
        print(f"\nCPU:")
        print(f"  Average: {self.get_avg_cpu_percent():.1f}%")
        print(f"\nThreads:")
        print(f"  Initial: {self.samples[0]['num_threads']}")
        print(f"  Final: {self.samples[-1]['num_threads']}")
        print(f"{'='*80}\n")


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


@pytest.fixture
def resource_monitor():
    """Create resource monitor."""
    return ResourceMonitor()


# ==============================================================================
# CPU Usage Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_cpu_usage_under_load(agent_config, test_input, resource_monitor):
    """Test CPU usage under increasing load.

    Validates that CPU usage scales appropriately with load.
    """
    print(f"\n{'='*80}")
    print("TEST: CPU Usage Under Load")
    print(f"{'='*80}")

    load_levels = [1, 10, 50]

    for num_concurrent in load_levels:
        print(f"\nTesting with {num_concurrent} concurrent agents...")

        # Take snapshot before
        resource_monitor.take_snapshot()

        # Execute load
        start = time.perf_counter()
        async with AsyncFuelAgentAI(agent_config) as agent:
            tasks = [agent.run_async(test_input) for _ in range(num_concurrent)]
            await asyncio.gather(*tasks, return_exceptions=True)
        duration = time.perf_counter() - start

        # Take snapshot after
        resource_monitor.take_snapshot()

        # Sample during execution
        for _ in range(5):
            await asyncio.sleep(0.1)
            resource_monitor.take_snapshot()

        print(f"  Duration: {duration:.2f}s")

    resource_monitor.print_summary()

    # Validate CPU usage is reasonable
    avg_cpu = resource_monitor.get_avg_cpu_percent()
    print(f"Average CPU usage: {avg_cpu:.1f}%")

    # Should use CPU but not peg it at 100%
    assert avg_cpu < 95.0, f"CPU usage too high: {avg_cpu:.1f}%"

    print(f"✓ Test passed!")


# ==============================================================================
# Memory Usage Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_memory_usage_patterns(agent_config, test_input, resource_monitor):
    """Test memory usage patterns under different loads.

    Validates that memory usage is reasonable and doesn't grow unbounded.
    """
    print(f"\n{'='*80}")
    print("TEST: Memory Usage Patterns")
    print(f"{'='*80}")

    # Baseline memory
    resource_monitor.take_snapshot()
    baseline_memory = resource_monitor.samples[0]["memory_rss_mb"]

    # Test different loads
    load_levels = [10, 50, 100]

    for num_concurrent in load_levels:
        print(f"\nTesting {num_concurrent} concurrent agents...")

        async with AsyncFuelAgentAI(agent_config) as agent:
            tasks = [agent.run_async(test_input) for _ in range(num_concurrent)]
            await asyncio.gather(*tasks, return_exceptions=True)

        resource_monitor.take_snapshot()

        current_memory = resource_monitor.samples[-1]["memory_rss_mb"]
        memory_increase = current_memory - baseline_memory

        print(f"  Memory: {current_memory:.2f} MB (+{memory_increase:.2f} MB)")

    resource_monitor.print_summary()

    # Validate memory usage
    peak_memory = resource_monitor.get_peak_memory_mb()
    memory_growth = resource_monitor.get_memory_growth_mb()

    print(f"\nPeak Memory: {peak_memory:.2f} MB")
    print(f"Total Growth: {memory_growth:.2f} MB")

    # Memory growth should be bounded
    assert memory_growth < 500, f"Excessive memory growth: {memory_growth:.2f} MB"

    print(f"✓ Test passed!")


@pytest.mark.asyncio
async def test_memory_leak_detection(agent_config, test_input, resource_monitor):
    """Test for memory leaks with repeated executions.

    Runs multiple iterations and checks if memory grows unbounded.
    """
    print(f"\n{'='*80}")
    print("TEST: Memory Leak Detection")
    print(f"{'='*80}")

    num_iterations = 5
    executions_per_iteration = 20

    memory_readings = []

    for iteration in range(num_iterations):
        print(f"\nIteration {iteration + 1}/{num_iterations}...")

        # Run executions
        async with AsyncFuelAgentAI(agent_config) as agent:
            for _ in range(executions_per_iteration):
                await agent.run_async(test_input)

        # Force garbage collection
        gc.collect()

        # Take snapshot
        snapshot = resource_monitor.take_snapshot()
        memory_readings.append(snapshot["memory_rss_mb"])

        print(f"  Memory: {snapshot['memory_rss_mb']:.2f} MB")

    # Check if memory is growing linearly (indicates leak)
    memory_trend = memory_readings[-1] - memory_readings[0]
    avg_per_iteration = memory_trend / (num_iterations - 1) if num_iterations > 1 else 0

    print(f"\nMemory Trend:")
    print(f"  Initial: {memory_readings[0]:.2f} MB")
    print(f"  Final: {memory_readings[-1]:.2f} MB")
    print(f"  Total Growth: {memory_trend:.2f} MB")
    print(f"  Avg Growth/Iteration: {avg_per_iteration:.2f} MB")

    # Memory should stabilize (< 10MB growth per iteration)
    assert avg_per_iteration < 10.0, f"Potential memory leak: {avg_per_iteration:.2f} MB/iteration"

    print(f"✓ No memory leak detected!")


@pytest.mark.asyncio
async def test_memory_cleanup_after_context_exit(agent_config, test_input):
    """Test that memory is cleaned up after agent context exit.

    Validates that async context manager properly releases resources.
    """
    print(f"\n{'='*80}")
    print("TEST: Memory Cleanup After Context Exit")
    print(f"{'='*80}")

    process = psutil.Process(os.getpid())

    # Measure before
    gc.collect()
    memory_before = process.memory_info().rss / 1024 / 1024

    print(f"Memory before: {memory_before:.2f} MB")

    # Use agent in context
    async with AsyncFuelAgentAI(agent_config) as agent:
        for _ in range(50):
            await agent.run_async(test_input)

    # Measure after (context exited)
    gc.collect()
    await asyncio.sleep(0.1)  # Allow cleanup
    memory_after = process.memory_info().rss / 1024 / 1024

    print(f"Memory after: {memory_after:.2f} MB")

    memory_delta = memory_after - memory_before

    print(f"Memory delta: {memory_delta:.2f} MB")

    # Memory should not grow excessively after cleanup
    assert memory_delta < 50, f"Memory not cleaned up properly: {memory_delta:.2f} MB leaked"

    print(f"✓ Test passed!")


# ==============================================================================
# Thread and File Descriptor Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_thread_count_stability(agent_config, test_input):
    """Test that thread count remains stable.

    Validates that async agents don't create excessive threads.
    """
    print(f"\n{'='*80}")
    print("TEST: Thread Count Stability")
    print(f"{'='*80}")

    process = psutil.Process(os.getpid())

    # Baseline thread count
    threads_before = process.num_threads()
    print(f"Threads before: {threads_before}")

    # Execute many concurrent operations
    num_iterations = 3
    concurrent_per_iteration = 50

    for iteration in range(num_iterations):
        async with AsyncFuelAgentAI(agent_config) as agent:
            tasks = [agent.run_async(test_input) for _ in range(concurrent_per_iteration)]
            await asyncio.gather(*tasks, return_exceptions=True)

        threads_after = process.num_threads()
        print(f"Threads after iteration {iteration + 1}: {threads_after}")

    # Final thread count
    threads_final = process.num_threads()

    print(f"\nThread count growth: {threads_final - threads_before}")

    # Thread count should remain stable (within reason)
    assert threads_final - threads_before < 10, "Thread count growing excessively"

    print(f"✓ Test passed!")


@pytest.mark.asyncio
async def test_file_descriptor_leaks(agent_config, test_input):
    """Test for file descriptor leaks.

    Validates that file descriptors are properly closed.
    Note: This test may not work on Windows.
    """
    if os.name == 'nt':
        pytest.skip("File descriptor tracking not available on Windows")

    print(f"\n{'='*80}")
    print("TEST: File Descriptor Leak Detection")
    print(f"{'='*80}")

    process = psutil.Process(os.getpid())

    # Baseline FD count
    fds_before = process.num_fds()
    print(f"File descriptors before: {fds_before}")

    # Execute operations
    for iteration in range(5):
        async with AsyncFuelAgentAI(agent_config) as agent:
            for _ in range(10):
                await agent.run_async(test_input)

        fds_after = process.num_fds()
        print(f"File descriptors after iteration {iteration + 1}: {fds_after}")

    # Final FD count
    fds_final = process.num_fds()

    print(f"\nFile descriptor growth: {fds_final - fds_before}")

    # FD count should remain stable
    assert fds_final - fds_before < 20, "File descriptor leak detected"

    print(f"✓ Test passed!")


# ==============================================================================
# I/O Performance Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_io_performance(agent_config, test_input):
    """Test I/O performance under load.

    Measures I/O operations during agent execution.
    Note: May not work on all platforms.
    """
    print(f"\n{'='*80}")
    print("TEST: I/O Performance")
    print(f"{'='*80}")

    process = psutil.Process(os.getpid())

    # Get baseline I/O counters
    try:
        io_before = process.io_counters()
        print(f"I/O before:")
        print(f"  Read: {io_before.read_bytes / 1024:.2f} KB")
        print(f"  Write: {io_before.write_bytes / 1024:.2f} KB")
    except (AttributeError, psutil.AccessDenied):
        pytest.skip("I/O counters not available on this platform")
        return

    # Execute load
    async with AsyncFuelAgentAI(agent_config) as agent:
        tasks = [agent.run_async(test_input) for _ in range(50)]
        await asyncio.gather(*tasks, return_exceptions=True)

    # Get final I/O counters
    io_after = process.io_counters()
    print(f"\nI/O after:")
    print(f"  Read: {io_after.read_bytes / 1024:.2f} KB")
    print(f"  Write: {io_after.write_bytes / 1024:.2f} KB")

    # Calculate I/O operations
    read_kb = (io_after.read_bytes - io_before.read_bytes) / 1024
    write_kb = (io_after.write_bytes - io_before.write_bytes) / 1024

    print(f"\nI/O during test:")
    print(f"  Read: {read_kb:.2f} KB")
    print(f"  Write: {write_kb:.2f} KB")

    # Just informational - no strict assertions
    print(f"✓ Test completed!")


# ==============================================================================
# Resource Limits Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_resource_usage_under_stress(agent_config, test_input, resource_monitor):
    """Test resource usage under stress conditions.

    Runs maximum concurrent load and validates resource usage.
    """
    print(f"\n{'='*80}")
    print("TEST: Resource Usage Under Stress")
    print(f"{'='*80}")

    num_concurrent = 200  # Stress test

    print(f"Running {num_concurrent} concurrent agents...")

    # Monitor resources during execution
    monitor_task = asyncio.create_task(_monitor_resources(resource_monitor, interval=0.2))

    try:
        start = time.perf_counter()

        async with AsyncFuelAgentAI(agent_config) as agent:
            tasks = [agent.run_async(test_input) for _ in range(num_concurrent)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        duration = time.perf_counter() - start

    finally:
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass

    # Count successful vs failed
    successful = sum(1 for r in results if not isinstance(r, Exception) and r.success)

    print(f"\nResults:")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Successful: {successful}/{num_concurrent}")
    print(f"  Success Rate: {successful/num_concurrent*100:.1f}%")

    resource_monitor.print_summary()

    # Validate resource usage
    peak_memory = resource_monitor.get_peak_memory_mb()
    print(f"\nPeak Memory: {peak_memory:.2f} MB")

    # Should handle stress without excessive memory
    assert peak_memory < 1000, f"Excessive memory under stress: {peak_memory:.2f} MB"

    print(f"✓ Test passed!")


async def _monitor_resources(monitor: ResourceMonitor, interval: float):
    """Background task to monitor resources."""
    while True:
        monitor.take_snapshot()
        await asyncio.sleep(interval)


# ==============================================================================
# Main Entry Point (for standalone execution)
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
