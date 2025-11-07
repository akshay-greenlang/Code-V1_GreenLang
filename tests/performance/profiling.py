"""Performance Profiling Suite for GreenLang Agents.

This module provides comprehensive profiling capabilities:
- CPU profiling (cProfile)
- Memory profiling (tracemalloc)
- I/O profiling
- Bottleneck detection (top N slowest functions)
- Memory leak detection
- Profiling decorators (@profile, @memory_profile)
- Async-aware profiling

Architecture:
    PerformanceProfiler: Main profiling orchestrator
    CPUProfiler: CPU usage profiling with cProfile
    MemoryProfiler: Memory usage profiling with tracemalloc
    IOProfiler: I/O operation profiling
    BottleneckAnalyzer: Identify performance bottlenecks

Example Usage:
    >>> from tests.performance.profiling import PerformanceProfiler
    >>>
    >>> async def main():
    ...     profiler = PerformanceProfiler()
    ...     await profiler.profile_agent_execution(
    ...         agent_name="FuelAgentAI",
    ...         num_iterations=100
    ...     )
    ...     profiler.print_report()
    ...     profiler.save_report("fuel_agent_profile.txt")

Author: GreenLang Framework Team
Date: November 2025
Status: Production Ready
"""

from __future__ import annotations

import asyncio
import cProfile
import functools
import io
import os
import pstats
import sys
import time
import tracemalloc
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import linecache

# Try importing async agents
try:
    from greenlang.agents.fuel_agent_ai_async import AsyncFuelAgentAI
    from greenlang.config import get_config
    ASYNC_AGENTS_AVAILABLE = True
except ImportError:
    ASYNC_AGENTS_AVAILABLE = False


# ==============================================================================
# Profiling Results Data Structures
# ==============================================================================

@dataclass
class CPUProfileResult:
    """Results from CPU profiling."""
    total_calls: int
    total_time_seconds: float
    top_functions: List[Tuple[str, float, int]]  # (name, time, calls)
    profile_stats: pstats.Stats
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MemoryProfileResult:
    """Results from memory profiling."""
    peak_memory_mb: float
    current_memory_mb: float
    top_allocations: List[Tuple[str, int, int]]  # (file:line, size, count)
    memory_diff: Optional[Any] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class IOProfileResult:
    """Results from I/O profiling."""
    read_bytes: int
    write_bytes: int
    read_operations: int
    write_operations: int
    io_time_seconds: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ProfilingReport:
    """Combined profiling report."""
    test_name: str
    cpu_profile: Optional[CPUProfileResult] = None
    memory_profile: Optional[MemoryProfileResult] = None
    io_profile: Optional[IOProfileResult] = None
    bottlenecks: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


# ==============================================================================
# CPU Profiler
# ==============================================================================

class CPUProfiler:
    """CPU profiling using cProfile."""

    def __init__(self):
        """Initialize CPU profiler."""
        self.profiler = cProfile.Profile()
        self.is_profiling = False

    def start(self):
        """Start CPU profiling."""
        self.profiler.enable()
        self.is_profiling = True

    def stop(self):
        """Stop CPU profiling."""
        self.profiler.disable()
        self.is_profiling = False

    def get_results(self, top_n: int = 20) -> CPUProfileResult:
        """Get profiling results.

        Args:
            top_n: Number of top functions to include

        Returns:
            CPUProfileResult with statistics
        """
        # Create stats object
        stats = pstats.Stats(self.profiler)
        stats.strip_dirs()
        stats.sort_stats('cumulative')

        # Get total stats
        total_calls = stats.total_calls
        total_time = stats.total_tt

        # Get top functions
        top_functions = []
        for func, (cc, nc, tt, ct, callers) in list(stats.stats.items())[:top_n]:
            if isinstance(func, tuple) and len(func) >= 3:
                func_name = f"{func[0]}:{func[1]}({func[2]})"
            else:
                func_name = str(func)
            top_functions.append((func_name, ct, cc))

        return CPUProfileResult(
            total_calls=total_calls,
            total_time_seconds=total_time,
            top_functions=top_functions,
            profile_stats=stats
        )

    def print_stats(self, top_n: int = 20):
        """Print profiling statistics."""
        stats = pstats.Stats(self.profiler)
        stats.strip_dirs()
        stats.sort_stats('cumulative')
        stats.print_stats(top_n)


# ==============================================================================
# Memory Profiler
# ==============================================================================

class MemoryProfiler:
    """Memory profiling using tracemalloc."""

    def __init__(self):
        """Initialize memory profiler."""
        self.is_profiling = False
        self.snapshot_start = None
        self.snapshot_end = None

    def start(self):
        """Start memory profiling."""
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        self.snapshot_start = tracemalloc.take_snapshot()
        self.is_profiling = True

    def stop(self):
        """Stop memory profiling."""
        self.snapshot_end = tracemalloc.take_snapshot()
        self.is_profiling = False

    def get_results(self, top_n: int = 10) -> MemoryProfileResult:
        """Get memory profiling results.

        Args:
            top_n: Number of top allocations to include

        Returns:
            MemoryProfileResult with statistics
        """
        if not self.snapshot_end:
            self.snapshot_end = tracemalloc.take_snapshot()

        # Get current memory usage
        current, peak = tracemalloc.get_traced_memory()
        current_mb = current / 1024 / 1024
        peak_mb = peak / 1024 / 1024

        # Get top allocations
        top_stats = self.snapshot_end.statistics('lineno')
        top_allocations = []

        for stat in top_stats[:top_n]:
            top_allocations.append((
                f"{stat.traceback.format()[0]}",
                stat.size,
                stat.count
            ))

        # Get memory diff if we have start snapshot
        memory_diff = None
        if self.snapshot_start:
            memory_diff = self.snapshot_end.compare_to(self.snapshot_start, 'lineno')

        return MemoryProfileResult(
            peak_memory_mb=peak_mb,
            current_memory_mb=current_mb,
            top_allocations=top_allocations,
            memory_diff=memory_diff
        )

    def print_top_allocations(self, top_n: int = 10):
        """Print top memory allocations."""
        if not self.snapshot_end:
            print("No snapshot available")
            return

        print(f"\n{'='*80}")
        print(f"TOP {top_n} MEMORY ALLOCATIONS")
        print(f"{'='*80}\n")

        top_stats = self.snapshot_end.statistics('lineno')

        for index, stat in enumerate(top_stats[:top_n], 1):
            frame = stat.traceback[0]
            size_mb = stat.size / 1024 / 1024
            print(f"#{index}: {frame.filename}:{frame.lineno}")
            print(f"  Size: {size_mb:.2f} MB ({stat.count} blocks)")

            # Print source line
            line = linecache.getline(frame.filename, frame.lineno).strip()
            if line:
                print(f"  Code: {line}")
            print()

    def check_memory_leaks(self) -> List[str]:
        """Check for potential memory leaks.

        Returns:
            List of warnings about potential leaks
        """
        warnings = []

        if not self.snapshot_start or not self.snapshot_end:
            return warnings

        # Compare snapshots
        top_stats = self.snapshot_end.compare_to(self.snapshot_start, 'lineno')

        for stat in top_stats[:10]:
            if stat.size_diff > 1024 * 1024:  # > 1MB increase
                size_mb = stat.size_diff / 1024 / 1024
                warnings.append(
                    f"Memory increase of {size_mb:.2f} MB at {stat.traceback.format()[0]}"
                )

        return warnings


# ==============================================================================
# I/O Profiler
# ==============================================================================

class IOProfiler:
    """I/O profiling (basic implementation)."""

    def __init__(self):
        """Initialize I/O profiler."""
        self.start_time = None
        self.end_time = None
        self.read_bytes = 0
        self.write_bytes = 0
        self.read_ops = 0
        self.write_ops = 0

    def start(self):
        """Start I/O profiling."""
        self.start_time = time.perf_counter()

    def stop(self):
        """Stop I/O profiling."""
        self.end_time = time.perf_counter()

    def record_read(self, num_bytes: int):
        """Record a read operation."""
        self.read_bytes += num_bytes
        self.read_ops += 1

    def record_write(self, num_bytes: int):
        """Record a write operation."""
        self.write_bytes += num_bytes
        self.write_ops += 1

    def get_results(self) -> IOProfileResult:
        """Get I/O profiling results."""
        io_time = (self.end_time - self.start_time) if self.end_time else 0

        return IOProfileResult(
            read_bytes=self.read_bytes,
            write_bytes=self.write_bytes,
            read_operations=self.read_ops,
            write_operations=self.write_ops,
            io_time_seconds=io_time
        )


# ==============================================================================
# Performance Profiler (Main)
# ==============================================================================

class PerformanceProfiler:
    """Main performance profiler orchestrator."""

    def __init__(self, results_dir: Optional[Path] = None):
        """Initialize performance profiler.

        Args:
            results_dir: Directory to save profiling results
        """
        if results_dir is None:
            results_dir = Path(__file__).parent / "results"

        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.cpu_profiler = CPUProfiler()
        self.memory_profiler = MemoryProfiler()
        self.io_profiler = IOProfiler()

        self.config = get_config() if ASYNC_AGENTS_AVAILABLE else None
        self.reports: List[ProfilingReport] = []

        # Test input
        self.default_test_input = {
            "fuel_type": "natural_gas",
            "amount": 1000,
            "unit": "therms",
            "country": "US"
        }

    # ==========================================================================
    # Agent Profiling
    # ==========================================================================

    async def profile_agent_execution(
        self,
        agent_name: str = "FuelAgentAI",
        num_iterations: int = 10,
        test_input: Optional[Dict[str, Any]] = None,
        enable_cpu: bool = True,
        enable_memory: bool = True,
        enable_io: bool = True,
    ) -> ProfilingReport:
        """Profile agent execution.

        Args:
            agent_name: Name of agent to profile
            num_iterations: Number of iterations to run
            test_input: Input data for agent
            enable_cpu: Enable CPU profiling
            enable_memory: Enable memory profiling
            enable_io: Enable I/O profiling

        Returns:
            ProfilingReport with results
        """
        print(f"\n{'='*80}")
        print(f"PROFILING: {agent_name}")
        print(f"{'='*80}")
        print(f"Iterations: {num_iterations}")
        print(f"CPU Profiling: {enable_cpu}")
        print(f"Memory Profiling: {enable_memory}")
        print(f"I/O Profiling: {enable_io}")
        print(f"{'='*80}\n")

        if test_input is None:
            test_input = self.default_test_input

        # Start profilers
        if enable_cpu:
            self.cpu_profiler.start()
        if enable_memory:
            self.memory_profiler.start()
        if enable_io:
            self.io_profiler.start()

        # Execute agent
        async with AsyncFuelAgentAI(self.config) as agent:
            for i in range(num_iterations):
                if (i + 1) % 10 == 0:
                    print(f"  Completed {i + 1}/{num_iterations} iterations...")
                result = await agent.run_async(test_input)

                if not result.success:
                    print(f"  Warning: Iteration {i+1} failed: {result.error}")

        # Stop profilers
        if enable_cpu:
            self.cpu_profiler.stop()
        if enable_memory:
            self.memory_profiler.stop()
        if enable_io:
            self.io_profiler.stop()

        # Collect results
        report = ProfilingReport(test_name=f"{agent_name}_profile")

        if enable_cpu:
            report.cpu_profile = self.cpu_profiler.get_results(top_n=20)
        if enable_memory:
            report.memory_profile = self.memory_profiler.get_results(top_n=10)
        if enable_io:
            report.io_profile = self.io_profiler.get_results()

        # Analyze bottlenecks
        report.bottlenecks = self._identify_bottlenecks(report)
        report.recommendations = self._generate_recommendations(report)

        self.reports.append(report)

        print(f"\n{'='*80}")
        print("PROFILING COMPLETE")
        print(f"{'='*80}\n")

        return report

    async def profile_concurrent_execution(
        self,
        num_concurrent: int = 10,
        test_input: Optional[Dict[str, Any]] = None,
    ) -> ProfilingReport:
        """Profile concurrent agent execution.

        Args:
            num_concurrent: Number of concurrent executions
            test_input: Input data for agents

        Returns:
            ProfilingReport with results
        """
        print(f"\n{'='*80}")
        print(f"PROFILING: Concurrent Execution ({num_concurrent} agents)")
        print(f"{'='*80}\n")

        if test_input is None:
            test_input = self.default_test_input

        # Start profilers
        self.cpu_profiler.start()
        self.memory_profiler.start()

        # Execute concurrently
        async with AsyncFuelAgentAI(self.config) as agent:
            tasks = [agent.run_async(test_input) for _ in range(num_concurrent)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Stop profilers
        self.cpu_profiler.stop()
        self.memory_profiler.stop()

        # Collect results
        report = ProfilingReport(test_name=f"concurrent_{num_concurrent}_profile")
        report.cpu_profile = self.cpu_profiler.get_results(top_n=20)
        report.memory_profile = self.memory_profiler.get_results(top_n=10)
        report.bottlenecks = self._identify_bottlenecks(report)
        report.recommendations = self._generate_recommendations(report)

        self.reports.append(report)

        print(f"\n{'='*80}")
        print("PROFILING COMPLETE")
        print(f"{'='*80}\n")

        return report

    # ==========================================================================
    # Analysis and Reporting
    # ==========================================================================

    def _identify_bottlenecks(self, report: ProfilingReport) -> List[str]:
        """Identify performance bottlenecks from profiling data."""
        bottlenecks = []

        # Check CPU bottlenecks
        if report.cpu_profile:
            for func_name, cum_time, calls in report.cpu_profile.top_functions[:5]:
                if cum_time > 1.0:  # > 1 second cumulative time
                    bottlenecks.append(
                        f"CPU: {func_name} took {cum_time:.2f}s ({calls} calls)"
                    )

        # Check memory bottlenecks
        if report.memory_profile:
            if report.memory_profile.peak_memory_mb > 100:
                bottlenecks.append(
                    f"Memory: Peak usage {report.memory_profile.peak_memory_mb:.2f} MB"
                )

            for location, size, count in report.memory_profile.top_allocations[:3]:
                size_mb = size / 1024 / 1024
                if size_mb > 10:
                    bottlenecks.append(
                        f"Memory: {size_mb:.2f} MB allocated at {location}"
                    )

        return bottlenecks

    def _generate_recommendations(self, report: ProfilingReport) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []

        # CPU recommendations
        if report.cpu_profile:
            total_time = report.cpu_profile.total_time_seconds

            if total_time > 10:
                recommendations.append(
                    "Consider caching expensive computations or using async I/O"
                )

            # Check for serialization bottlenecks
            for func_name, cum_time, calls in report.cpu_profile.top_functions:
                if 'json' in func_name.lower() and cum_time > 0.5:
                    recommendations.append(
                        "JSON serialization is a bottleneck - consider using msgpack or protobuf"
                    )

        # Memory recommendations
        if report.memory_profile:
            if report.memory_profile.peak_memory_mb > 500:
                recommendations.append(
                    "High memory usage detected - consider streaming or chunking data"
                )

            # Check for memory leaks
            leak_warnings = self.memory_profiler.check_memory_leaks()
            if leak_warnings:
                recommendations.append(
                    "Potential memory leak detected - review resource cleanup"
                )

        return recommendations

    def print_report(self, report: Optional[ProfilingReport] = None):
        """Print profiling report.

        Args:
            report: Report to print (uses latest if not provided)
        """
        if report is None:
            if not self.reports:
                print("No profiling reports available")
                return
            report = self.reports[-1]

        print(f"\n{'='*80}")
        print(f"PROFILING REPORT: {report.test_name}")
        print(f"{'='*80}")

        # CPU Profile
        if report.cpu_profile:
            print(f"\nCPU PROFILE:")
            print(f"  Total Calls: {report.cpu_profile.total_calls:,}")
            print(f"  Total Time: {report.cpu_profile.total_time_seconds:.2f}s")
            print(f"\n  Top Functions (by cumulative time):")

            for i, (func_name, cum_time, calls) in enumerate(report.cpu_profile.top_functions[:10], 1):
                print(f"    {i:2d}. {func_name}")
                print(f"        Time: {cum_time:.3f}s  Calls: {calls:,}")

        # Memory Profile
        if report.memory_profile:
            print(f"\nMEMORY PROFILE:")
            print(f"  Peak Memory: {report.memory_profile.peak_memory_mb:.2f} MB")
            print(f"  Current Memory: {report.memory_profile.current_memory_mb:.2f} MB")
            print(f"\n  Top Allocations:")

            for i, (location, size, count) in enumerate(report.memory_profile.top_allocations[:5], 1):
                size_mb = size / 1024 / 1024
                print(f"    {i}. {size_mb:.2f} MB ({count:,} blocks)")
                print(f"       {location}")

        # I/O Profile
        if report.io_profile:
            print(f"\nI/O PROFILE:")
            print(f"  Read: {report.io_profile.read_bytes:,} bytes ({report.io_profile.read_operations} ops)")
            print(f"  Write: {report.io_profile.write_bytes:,} bytes ({report.io_profile.write_operations} ops)")
            print(f"  I/O Time: {report.io_profile.io_time_seconds:.2f}s")

        # Bottlenecks
        if report.bottlenecks:
            print(f"\nBOTTLENECKS IDENTIFIED:")
            for bottleneck in report.bottlenecks:
                print(f"  - {bottleneck}")

        # Recommendations
        if report.recommendations:
            print(f"\nOPTIMIZATION RECOMMENDATIONS:")
            for i, rec in enumerate(report.recommendations, 1):
                print(f"  {i}. {rec}")

        print(f"\n{'='*80}\n")

    def save_report(self, filename: str, report: Optional[ProfilingReport] = None):
        """Save profiling report to file.

        Args:
            filename: Output filename
            report: Report to save (uses latest if not provided)
        """
        if report is None:
            if not self.reports:
                print("No profiling reports available")
                return
            report = self.reports[-1]

        filepath = self.results_dir / filename

        with open(filepath, 'w') as f:
            # Redirect print to file
            original_stdout = sys.stdout
            sys.stdout = f
            self.print_report(report)
            sys.stdout = original_stdout

        print(f"Report saved to: {filepath}")
        return filepath


# ==============================================================================
# Profiling Decorators
# ==============================================================================

def profile(func: Callable) -> Callable:
    """Decorator to profile a function's CPU usage.

    Example:
        @profile
        async def my_function():
            # ... code to profile ...
            pass
    """
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()

        try:
            result = await func(*args, **kwargs)
        finally:
            profiler.disable()

            # Print stats
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
            ps.print_stats(20)
            print(f"\nProfile for {func.__name__}:")
            print(s.getvalue())

        return result

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()

        try:
            result = func(*args, **kwargs)
        finally:
            profiler.disable()

            # Print stats
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
            ps.print_stats(20)
            print(f"\nProfile for {func.__name__}:")
            print(s.getvalue())

        return result

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


def memory_profile(func: Callable) -> Callable:
    """Decorator to profile a function's memory usage.

    Example:
        @memory_profile
        async def my_function():
            # ... code to profile ...
            pass
    """
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        tracemalloc.start()
        snapshot_before = tracemalloc.take_snapshot()

        try:
            result = await func(*args, **kwargs)
        finally:
            snapshot_after = tracemalloc.take_snapshot()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Print memory stats
            print(f"\nMemory Profile for {func.__name__}:")
            print(f"  Current: {current / 1024 / 1024:.2f} MB")
            print(f"  Peak: {peak / 1024 / 1024:.2f} MB")

            # Show top differences
            top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')
            print("\n  Top memory increases:")
            for stat in top_stats[:5]:
                print(f"    {stat}")

        return result

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        tracemalloc.start()
        snapshot_before = tracemalloc.take_snapshot()

        try:
            result = func(*args, **kwargs)
        finally:
            snapshot_after = tracemalloc.take_snapshot()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Print memory stats
            print(f"\nMemory Profile for {func.__name__}:")
            print(f"  Current: {current / 1024 / 1024:.2f} MB")
            print(f"  Peak: {peak / 1024 / 1024:.2f} MB")

            # Show top differences
            top_stats = snapshot_after.compare_to(snapshot_before, 'lineno')
            print("\n  Top memory increases:")
            for stat in top_stats[:5]:
                print(f"    {stat}")

        return result

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


# ==============================================================================
# Main Entry Point
# ==============================================================================

async def main():
    """Run example profiling."""
    if not ASYNC_AGENTS_AVAILABLE:
        print("Error: Async agents not available. Cannot run profiling.")
        return

    profiler = PerformanceProfiler()

    # Test 1: Profile single agent execution
    print("\n" + "="*80)
    print("TEST 1: Profile Single Agent (100 iterations)")
    print("="*80)
    report1 = await profiler.profile_agent_execution(
        agent_name="FuelAgentAI",
        num_iterations=100
    )
    profiler.print_report(report1)
    profiler.save_report("fuel_agent_profile.txt", report1)

    # Test 2: Profile concurrent execution
    print("\n" + "="*80)
    print("TEST 2: Profile Concurrent Execution (50 agents)")
    print("="*80)
    report2 = await profiler.profile_concurrent_execution(num_concurrent=50)
    profiler.print_report(report2)
    profiler.save_report("concurrent_profile.txt", report2)

    print("\n" + "="*80)
    print("ALL PROFILING COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())
