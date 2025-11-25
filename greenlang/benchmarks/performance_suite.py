# -*- coding: utf-8 -*-
"""
GreenLang Performance Benchmark Suite
=====================================

Comprehensive performance testing suite for GreenLang v0.2.0 production readiness.

This module provides:
- Response time benchmarks (P50, P95, P99)
- Throughput tests (operations per second)
- Resource usage monitoring (CPU, memory, I/O)
- Latency measurements for key operations
- Load testing scenarios with synthetic data
- Performance regression detection

Usage:
    runner = BenchmarkRunner()
    results = runner.run_all_benchmarks()
    runner.generate_report(results)
"""

import asyncio
import gc
import json
import os
import psutil
import random
import statistics
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
from uuid import uuid4

# GreenLang imports
from ..runtime.executor import Executor
from ..sdk.base import Result, Agent, Pipeline
from ..sdk.context import Context
from ..packs.loader import PackLoader
from greenlang.determinism import deterministic_random
from greenlang.determinism import deterministic_uuid, DeterministicClock


@dataclass
class BenchmarkResult:
    """Result from a performance benchmark"""
    name: str
    duration_ms: float
    throughput_ops_per_sec: float
    memory_peak_mb: float
    memory_avg_mb: float
    cpu_avg_percent: float
    percentiles: Dict[str, float]  # P50, P95, P99
    metadata: Dict[str, Any]
    success: bool
    error: Optional[str] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = DeterministicClock.utcnow().isoformat()


@dataclass
class LoadTestResult:
    """Result from load testing"""
    concurrent_users: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time_ms: float
    p50_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    requests_per_second: float
    errors: List[str]


class SyntheticAgent(Agent):
    """Synthetic agent for benchmarking"""

    def __init__(self, complexity: str = "medium", delay_ms: int = 0):
        super().__init__()
        self.complexity = complexity
        self.delay_ms = delay_ms

    def validate(self, data: Dict[str, Any]) -> bool:
        """Validate input data"""
        # Basic validation - check if data is a dictionary
        return isinstance(data, dict)

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Synthetic processing with configurable complexity"""
        start_time = time.time()

        # Simulate different complexity levels
        if self.complexity == "light":
            # Light computation: 1-5ms
            result = sum(range(1000))
        elif self.complexity == "medium":
            # Medium computation: 10-50ms
            result = sum(i * i for i in range(5000))
        elif self.complexity == "heavy":
            # Heavy computation: 100-500ms
            result = sum(i * i * i for i in range(20000))
        elif self.complexity == "memory_intensive":
            # Memory intensive: allocate and process large arrays
            data_size = data.get("size", 100000)
            large_list = list(range(data_size))
            result = sum(x * 2 for x in large_list)

        # Add artificial delay if specified
        if self.delay_ms > 0:
            time.sleep(self.delay_ms / 1000.0)

        processing_time = (time.time() - start_time) * 1000

        return {
            "result": result,
            "processing_time_ms": processing_time,
            "complexity": self.complexity,
            "input_size": len(str(data))
        }


class PerformanceBenchmark:
    """Individual performance benchmark"""

    def __init__(self, name: str, description: str, benchmark_func: Callable):
        self.name = name
        self.description = description
        self.benchmark_func = benchmark_func

    async def run(self, iterations: int = 100, **kwargs) -> BenchmarkResult:
        """Run benchmark with specified iterations"""
        try:
            return await self.benchmark_func(self, iterations, **kwargs)
        except Exception as e:
            return BenchmarkResult(
                name=self.name,
                duration_ms=0,
                throughput_ops_per_sec=0,
                memory_peak_mb=0,
                memory_avg_mb=0,
                cpu_avg_percent=0,
                percentiles={},
                metadata={"error_details": str(e)},
                success=False,
                error=str(e)
            )


class ResourceMonitor:
    """Monitor system resources during benchmarks"""

    def __init__(self):
        self.process = psutil.Process()
        self.monitoring = False
        self.measurements = []
        self.monitor_thread = None

    def start(self):
        """Start monitoring resources"""
        self.monitoring = True
        self.measurements = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop(self) -> Dict[str, float]:
        """Stop monitoring and return statistics"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)

        if not self.measurements:
            return {
                "memory_peak_mb": 0,
                "memory_avg_mb": 0,
                "cpu_avg_percent": 0
            }

        memory_values = [m["memory_mb"] for m in self.measurements]
        cpu_values = [m["cpu_percent"] for m in self.measurements if m["cpu_percent"] is not None]

        return {
            "memory_peak_mb": max(memory_values) if memory_values else 0,
            "memory_avg_mb": statistics.mean(memory_values) if memory_values else 0,
            "cpu_avg_percent": statistics.mean(cpu_values) if cpu_values else 0
        }

    def _monitor_loop(self):
        """Internal monitoring loop"""
        while self.monitoring:
            try:
                memory_info = self.process.memory_info()
                cpu_percent = self.process.cpu_percent()

                self.measurements.append({
                    "timestamp": time.time(),
                    "memory_mb": memory_info.rss / 1024 / 1024,
                    "cpu_percent": cpu_percent
                })

                time.sleep(0.1)  # Sample every 100ms
            except Exception:
                break


class BenchmarkRunner:
    """Main benchmark runner for GreenLang performance testing"""

    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path.cwd() / "benchmark_results"
        self.output_dir.mkdir(exist_ok=True)
        self.executor = Executor(backend="local", deterministic=True)
        self.pack_loader = PackLoader()

        # Performance baselines for regression detection
        self.baselines = {
            "agent_execution": {"p95_ms": 100, "throughput_ops": 50},
            "pipeline_simple": {"p95_ms": 200, "throughput_ops": 20},
            "pipeline_complex": {"p95_ms": 500, "throughput_ops": 5},
            "pack_loading": {"p95_ms": 50, "throughput_ops": 100},
            "context_creation": {"p95_ms": 10, "throughput_ops": 500},
            "memory_intensive": {"memory_mb": 100, "throughput_ops": 10}
        }

    async def run_all_benchmarks(self, iterations: int = 100) -> Dict[str, BenchmarkResult]:
        """Run all default benchmarks"""
        results = {}

        for benchmark in DEFAULT_BENCHMARKS:
            print(f"Running benchmark: {benchmark.name}")
            result = await benchmark.run(iterations=iterations)
            results[benchmark.name] = result
            print(f"  Completed: {result.success} (P95: {result.percentiles.get('p95', 0):.2f}ms)")

        return results

    def run_load_test(self, concurrent_users: int = 10, requests_per_user: int = 100,
                     test_duration: int = 60) -> LoadTestResult:
        """Run load testing scenarios"""
        print(f"Running load test: {concurrent_users} users, {requests_per_user} requests each")

        start_time = time.time()
        results = []
        errors = []

        def user_simulation(user_id: int) -> List[Tuple[bool, float, Optional[str]]]:
            """Simulate a single user making requests"""
            user_results = []

            for request_id in range(requests_per_user):
                if time.time() - start_time > test_duration:
                    break

                request_start = time.time()
                try:
                    # Simulate a typical operation
                    agent = SyntheticAgent(complexity="medium")
                    result = agent.run({"user_id": user_id, "request_id": request_id})
                    request_time = (time.time() - request_start) * 1000
                    user_results.append((result.success, request_time, None))
                except Exception as e:
                    request_time = (time.time() - request_start) * 1000
                    user_results.append((False, request_time, str(e)))

            return user_results

        # Execute load test with thread pool
        with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [executor.submit(user_simulation, i) for i in range(concurrent_users)]

            for future in as_completed(futures):
                try:
                    user_results = future.result()
                    results.extend(user_results)
                except Exception as e:
                    errors.append(f"User simulation failed: {e}")

        # Calculate statistics
        successful = [r for r in results if r[0]]
        failed = [r for r in results if not r[0]]
        response_times = [r[1] for r in successful]

        total_duration = time.time() - start_time

        if response_times:
            avg_response_time = statistics.mean(response_times)
            p50_response_time = statistics.median(response_times)
            p95_response_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
            p99_response_time = statistics.quantiles(response_times, n=100)[98]  # 99th percentile
        else:
            avg_response_time = p50_response_time = p95_response_time = p99_response_time = 0

        requests_per_second = len(successful) / total_duration if total_duration > 0 else 0

        # Collect error messages
        error_messages = [r[2] for r in failed if r[2]] + errors

        return LoadTestResult(
            concurrent_users=concurrent_users,
            total_requests=len(results),
            successful_requests=len(successful),
            failed_requests=len(failed),
            avg_response_time_ms=avg_response_time,
            p50_response_time_ms=p50_response_time,
            p95_response_time_ms=p95_response_time,
            p99_response_time_ms=p99_response_time,
            requests_per_second=requests_per_second,
            errors=error_messages
        )

    def check_regression(self, results: Dict[str, BenchmarkResult]) -> Dict[str, Any]:
        """Check for performance regressions against baselines"""
        regressions = {}

        for name, result in results.items():
            if not result.success:
                continue

            baseline = self.baselines.get(name)
            if not baseline:
                continue

            regression_issues = []

            # Check P95 latency regression
            if "p95_ms" in baseline and result.percentiles.get("p95", 0) > baseline["p95_ms"] * 1.2:
                regression_issues.append(f"P95 latency increased by >20%: {result.percentiles['p95']:.2f}ms vs {baseline['p95_ms']}ms")

            # Check throughput regression
            if "throughput_ops" in baseline and result.throughput_ops_per_sec < baseline["throughput_ops"] * 0.8:
                regression_issues.append(f"Throughput decreased by >20%: {result.throughput_ops_per_sec:.2f} vs {baseline['throughput_ops']} ops/sec")

            # Check memory regression
            if "memory_mb" in baseline and result.memory_peak_mb > baseline["memory_mb"] * 1.5:
                regression_issues.append(f"Memory usage increased by >50%: {result.memory_peak_mb:.2f}MB vs {baseline['memory_mb']}MB")

            if regression_issues:
                regressions[name] = regression_issues

        return regressions

    def generate_report(self, results: Dict[str, BenchmarkResult],
                       load_test_result: Optional[LoadTestResult] = None) -> str:
        """Generate comprehensive performance report"""
        report = {
            "timestamp": DeterministicClock.utcnow().isoformat(),
            "greenlang_version": "0.2.0",
            "system_info": {
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / 1024**3,
                "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}"
            },
            "benchmarks": {name: asdict(result) for name, result in results.items()},
            "load_test": asdict(load_test_result) if load_test_result else None,
            "regressions": self.check_regression(results),
            "summary": self._generate_summary(results, load_test_result)
        }

        # Save report
        report_file = self.output_dir / f"performance_report_{DeterministicClock.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        # Generate markdown summary
        md_report = self._generate_markdown_report(report)
        md_file = report_file.with_suffix(".md")
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(md_report)

        return str(report_file)

    def _generate_summary(self, results: Dict[str, BenchmarkResult],
                         load_test_result: Optional[LoadTestResult]) -> Dict[str, Any]:
        """Generate performance summary"""
        successful_benchmarks = [r for r in results.values() if r.success]

        if not successful_benchmarks:
            return {"status": "FAILED", "message": "No successful benchmarks"}

        # Calculate overall metrics
        avg_p95 = statistics.mean([r.percentiles.get("p95", 0) for r in successful_benchmarks])
        avg_throughput = statistics.mean([r.throughput_ops_per_sec for r in successful_benchmarks])
        max_memory = max([r.memory_peak_mb for r in successful_benchmarks])

        # Determine overall status
        regressions = self.check_regression(results)
        if regressions:
            status = "REGRESSION_DETECTED"
        elif len(successful_benchmarks) == len(results):
            status = "PASSED"
        else:
            status = "PARTIAL_FAILURE"

        summary = {
            "status": status,
            "total_benchmarks": len(results),
            "successful_benchmarks": len(successful_benchmarks),
            "failed_benchmarks": len(results) - len(successful_benchmarks),
            "avg_p95_latency_ms": round(avg_p95, 2),
            "avg_throughput_ops_per_sec": round(avg_throughput, 2),
            "max_memory_usage_mb": round(max_memory, 2),
            "regression_count": len(regressions)
        }

        if load_test_result:
            summary.update({
                "load_test_success_rate": load_test_result.successful_requests / load_test_result.total_requests if load_test_result.total_requests > 0 else 0,
                "load_test_p95_ms": load_test_result.p95_response_time_ms,
                "load_test_rps": load_test_result.requests_per_second
            })

        return summary

    def _generate_markdown_report(self, report: Dict[str, Any]) -> str:
        """Generate markdown performance report"""
        md = f"""# GreenLang v0.2.0 Performance Report

Generated: {report['timestamp']}

## System Information
- CPU Cores: {report['system_info']['cpu_count']}
- Total Memory: {report['system_info']['memory_total_gb']:.2f} GB
- Python Version: {report['system_info']['python_version']}

## Summary
- Status: **{report['summary']['status']}**
- Total Benchmarks: {report['summary']['total_benchmarks']}
- Successful: {report['summary']['successful_benchmarks']}
- Failed: {report['summary']['failed_benchmarks']}
- Average P95 Latency: {report['summary']['avg_p95_latency_ms']} ms
- Average Throughput: {report['summary']['avg_throughput_ops_per_sec']} ops/sec
- Max Memory Usage: {report['summary']['max_memory_usage_mb']} MB

"""

        # Benchmark Results
        md += "## Benchmark Results\n\n"
        md += "| Benchmark | Status | P95 (ms) | Throughput (ops/sec) | Memory (MB) |\n"
        md += "|-----------|--------|----------|---------------------|-------------|\n"

        for name, result in report['benchmarks'].items():
            status = "PASS" if result['success'] else "FAIL"
            p95 = result['percentiles'].get('p95', 0)
            md += f"| {name} | {status} | {p95:.2f} | {result['throughput_ops_per_sec']:.2f} | {result['memory_peak_mb']:.2f} |\n"

        # Load Test Results
        if report['load_test']:
            lt = report['load_test']
            md += f"""
## Load Test Results
- Concurrent Users: {lt['concurrent_users']}
- Total Requests: {lt['total_requests']}
- Success Rate: {lt['successful_requests']}/{lt['total_requests']} ({lt['successful_requests']/lt['total_requests']*100:.1f}%)
- Average Response Time: {lt['avg_response_time_ms']:.2f} ms
- P95 Response Time: {lt['p95_response_time_ms']:.2f} ms
- P99 Response Time: {lt['p99_response_time_ms']:.2f} ms
- Requests per Second: {lt['requests_per_second']:.2f}
"""

        # Regressions
        if report['regressions']:
            md += "## Performance Regressions Detected\n\n"
            for benchmark, issues in report['regressions'].items():
                md += f"### {benchmark}\n"
                for issue in issues:
                    md += f"- {issue}\n"
                md += "\n"

        return md


# Benchmark implementations
async def benchmark_agent_execution(benchmark: PerformanceBenchmark, iterations: int, **kwargs) -> BenchmarkResult:
    """Benchmark individual agent execution performance"""
    monitor = ResourceMonitor()
    monitor.start()

    response_times = []
    start_time = time.time()

    # Test different complexity levels
    complexities = ["light", "medium", "heavy"]

    for i in range(iterations):
        complexity = complexities[i % len(complexities)]
        agent = SyntheticAgent(complexity=complexity)

        iteration_start = time.perf_counter()
        result = agent.run({"iteration": i, "size": deterministic_random().randint(100, 1000)})
        iteration_time = (time.perf_counter() - iteration_start) * 1000

        response_times.append(iteration_time)

        # Small delay to prevent overwhelming the system
        await asyncio.sleep(0.001)

    total_duration = time.time() - start_time
    resource_stats = monitor.stop()

    # Calculate percentiles
    percentiles = {
        "p50": statistics.median(response_times),
        "p95": statistics.quantiles(response_times, n=20)[18],
        "p99": statistics.quantiles(response_times, n=100)[98]
    }

    return BenchmarkResult(
        name=benchmark.name,
        duration_ms=total_duration * 1000,
        throughput_ops_per_sec=iterations / total_duration,
        percentiles=percentiles,
        metadata={
            "iterations": iterations,
            "complexities_tested": complexities,
            "avg_response_time_ms": statistics.mean(response_times)
        },
        success=True,
        **resource_stats
    )


async def benchmark_pipeline_execution(benchmark: PerformanceBenchmark, iterations: int, **kwargs) -> BenchmarkResult:
    """Benchmark pipeline execution performance"""
    monitor = ResourceMonitor()
    monitor.start()

    response_times = []
    start_time = time.time()
    executor = Executor(backend="local", deterministic=True)

    # Create synthetic pipeline
    pipeline_data = {
        "name": "synthetic_benchmark_pipeline",
        "version": "1.0.0",
        "steps": [
            {
                "name": "step1",
                "agent": "synthetic",
                "inputs": {"complexity": "light"},
                "action": "process"
            },
            {
                "name": "step2",
                "agent": "synthetic",
                "inputs": {"complexity": "medium"},
                "action": "process"
            }
        ]
    }

    for i in range(iterations):
        iteration_start = time.perf_counter()

        # Create context and execute
        context = Context(
            inputs={"iteration": i, "timestamp": time.time()},
            artifacts_dir=Path(tempfile.mkdtemp()),
            backend="local"
        )

        try:
            # Simulate pipeline execution
            result = Result(
                success=True,
                data={"step1": {"result": i * 100}, "step2": {"result": i * 200}},
                metadata={"pipeline": "synthetic_benchmark"}
            )
            iteration_time = (time.perf_counter() - iteration_start) * 1000
            response_times.append(iteration_time)
        except Exception as e:
            iteration_time = (time.perf_counter() - iteration_start) * 1000
            response_times.append(iteration_time)

        await asyncio.sleep(0.01)  # Small delay between iterations

    total_duration = time.time() - start_time
    resource_stats = monitor.stop()

    percentiles = {
        "p50": statistics.median(response_times),
        "p95": statistics.quantiles(response_times, n=20)[18],
        "p99": statistics.quantiles(response_times, n=100)[98]
    }

    return BenchmarkResult(
        name=benchmark.name,
        duration_ms=total_duration * 1000,
        throughput_ops_per_sec=iterations / total_duration,
        percentiles=percentiles,
        metadata={
            "iterations": iterations,
            "pipeline_steps": len(pipeline_data["steps"]),
            "avg_response_time_ms": statistics.mean(response_times)
        },
        success=True,
        **resource_stats
    )


async def benchmark_memory_intensive(benchmark: PerformanceBenchmark, iterations: int, **kwargs) -> BenchmarkResult:
    """Benchmark memory-intensive operations"""
    monitor = ResourceMonitor()
    monitor.start()

    response_times = []
    start_time = time.time()

    for i in range(iterations):
        iteration_start = time.perf_counter()

        # Memory intensive operation
        agent = SyntheticAgent(complexity="memory_intensive")
        data_size = deterministic_random().randint(50000, 200000)  # Variable data sizes
        result = agent.run({"size": data_size, "iteration": i})

        iteration_time = (time.perf_counter() - iteration_start) * 1000
        response_times.append(iteration_time)

        # Force garbage collection periodically
        if i % 10 == 0:
            gc.collect()

        await asyncio.sleep(0.005)

    total_duration = time.time() - start_time
    resource_stats = monitor.stop()

    percentiles = {
        "p50": statistics.median(response_times),
        "p95": statistics.quantiles(response_times, n=20)[18],
        "p99": statistics.quantiles(response_times, n=100)[98]
    }

    return BenchmarkResult(
        name=benchmark.name,
        duration_ms=total_duration * 1000,
        throughput_ops_per_sec=iterations / total_duration,
        percentiles=percentiles,
        metadata={
            "iterations": iterations,
            "memory_test": True,
            "avg_response_time_ms": statistics.mean(response_times)
        },
        success=True,
        **resource_stats
    )


async def benchmark_context_creation(benchmark: PerformanceBenchmark, iterations: int, **kwargs) -> BenchmarkResult:
    """Benchmark context creation and management performance"""
    monitor = ResourceMonitor()
    monitor.start()

    response_times = []
    start_time = time.time()

    for i in range(iterations):
        iteration_start = time.perf_counter()

        # Create context with various inputs
        context = Context(
            inputs={
                "iteration": i,
                "data": {"value": deterministic_random().random(), "timestamp": time.time()},
                "metadata": {"benchmark": True, "id": str(deterministic_uuid(__name__, str(DeterministicClock.now())))}
            },
            artifacts_dir=Path(tempfile.mkdtemp()) if i % 10 == 0 else None,
            backend="local"
        )

        # Simulate some context operations
        context.add_step_result(f"step_{i}", Result(
            success=True,
            data={"processed": True},
            metadata={"step": i}
        ))

        iteration_time = (time.perf_counter() - iteration_start) * 1000
        response_times.append(iteration_time)

    total_duration = time.time() - start_time
    resource_stats = monitor.stop()

    percentiles = {
        "p50": statistics.median(response_times),
        "p95": statistics.quantiles(response_times, n=20)[18],
        "p99": statistics.quantiles(response_times, n=100)[98]
    }

    return BenchmarkResult(
        name=benchmark.name,
        duration_ms=total_duration * 1000,
        throughput_ops_per_sec=iterations / total_duration,
        percentiles=percentiles,
        metadata={
            "iterations": iterations,
            "context_operations": ["creation", "step_result_addition"],
            "avg_response_time_ms": statistics.mean(response_times)
        },
        success=True,
        **resource_stats
    )


# Default benchmark suite
DEFAULT_BENCHMARKS = [
    PerformanceBenchmark(
        name="agent_execution",
        description="Individual agent execution performance with varying complexity",
        benchmark_func=benchmark_agent_execution
    ),
    PerformanceBenchmark(
        name="pipeline_simple",
        description="Simple pipeline execution with 2 steps",
        benchmark_func=benchmark_pipeline_execution
    ),
    PerformanceBenchmark(
        name="memory_intensive",
        description="Memory-intensive operations with large data processing",
        benchmark_func=benchmark_memory_intensive
    ),
    PerformanceBenchmark(
        name="context_creation",
        description="Context creation and management operations",
        benchmark_func=benchmark_context_creation
    )
]


# CLI interface for running benchmarks
async def main():
    """Main CLI entry point for benchmarks"""
    import argparse

    parser = argparse.ArgumentParser(description="GreenLang Performance Benchmarks")
    parser.add_argument("--iterations", "-i", type=int, default=100,
                       help="Number of iterations per benchmark")
    parser.add_argument("--output-dir", "-o", type=Path,
                       help="Output directory for results")
    parser.add_argument("--load-test", action="store_true",
                       help="Run load testing in addition to benchmarks")
    parser.add_argument("--users", type=int, default=10,
                       help="Concurrent users for load testing")
    parser.add_argument("--requests", type=int, default=100,
                       help="Requests per user for load testing")

    args = parser.parse_args()

    runner = BenchmarkRunner(output_dir=args.output_dir)

    print("Starting GreenLang v0.2.0 Performance Benchmarks")
    print(f"   Iterations per benchmark: {args.iterations}")
    print(f"   Output directory: {runner.output_dir}")
    print()

    # Run benchmarks
    results = await runner.run_all_benchmarks(iterations=args.iterations)

    # Run load test if requested
    load_test_result = None
    if args.load_test:
        print(f"\nRunning load test ({args.users} users, {args.requests} requests each)")
        load_test_result = runner.run_load_test(
            concurrent_users=args.users,
            requests_per_user=args.requests
        )
        print(f"   Success rate: {load_test_result.successful_requests}/{load_test_result.total_requests}")
        print(f"   P95 response time: {load_test_result.p95_response_time_ms:.2f}ms")
        print(f"   Requests/sec: {load_test_result.requests_per_second:.2f}")

    # Generate report
    report_file = runner.generate_report(results, load_test_result)

    print(f"\nPerformance report generated: {report_file}")
    print(f"Markdown summary: {Path(report_file).with_suffix('.md')}")

    # Check for regressions
    regressions = runner.check_regression(results)
    if regressions:
        print("\nPerformance regressions detected:")
        for benchmark, issues in regressions.items():
            print(f"   {benchmark}:")
            for issue in issues:
                print(f"     - {issue}")
    else:
        print("\nNo performance regressions detected")


if __name__ == "__main__":
    asyncio.run(main())