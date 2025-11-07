"""Quick performance test runner.

Runs a subset of performance tests to validate the infrastructure.
"""

import asyncio
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from tests.performance.load_testing import LoadTester
from tests.performance.profiling import PerformanceProfiler
from tests.performance.regression_tests import RegressionTester


async def run_quick_tests():
    """Run quick performance tests."""
    print("="*80)
    print("GREENLANG PERFORMANCE TEST SUITE")
    print("="*80)
    print()

    # 1. Load Testing
    print("[1/3] Load Testing")
    print("-" * 80)
    tester = LoadTester()

    try:
        # Small concurrent test
        print("\nTest 1.1: 10 Concurrent Executions")
        results_10 = await tester.run_concurrent_load_test(num_concurrent=10)
        tester.save_results_json(results_10, "test_10_concurrent.json")
        print(f"[OK] Completed: {results_10.successful_requests}/{results_10.total_requests} successful")
        print(f"  p95 latency: {results_10.p95_latency_ms:.2f}ms")
        print(f"  Throughput: {results_10.actual_rps:.2f} RPS")

        print("\nTest 1.2: 50 Concurrent Executions")
        results_50 = await tester.run_concurrent_load_test(num_concurrent=50)
        tester.save_results_json(results_50, "test_50_concurrent.json")
        print(f"[OK] Completed: {results_50.successful_requests}/{results_50.total_requests} successful")
        print(f"  p95 latency: {results_50.p95_latency_ms:.2f}ms")
        print(f"  Throughput: {results_50.actual_rps:.2f} RPS")

    except Exception as e:
        print(f"[FAIL] Load testing failed: {e}")
        import traceback
        traceback.print_exc()

    # 2. Profiling
    print("\n[2/3] Performance Profiling")
    print("-" * 80)
    profiler = PerformanceProfiler()

    try:
        print("\nTest 2.1: Profile Agent Execution (50 iterations)")
        report = await profiler.profile_agent_execution(
            agent_name="FuelAgentAI",
            num_iterations=50,
            enable_cpu=True,
            enable_memory=True,
            enable_io=False
        )
        profiler.save_report("fuel_agent_profile.txt", report)
        print("[OK] Profiling completed")

        if report.cpu_profile:
            print(f"  Total time: {report.cpu_profile.total_time_seconds:.2f}s")
            print(f"  Total calls: {report.cpu_profile.total_calls:,}")

        if report.memory_profile:
            print(f"  Peak memory: {report.memory_profile.peak_memory_mb:.2f} MB")

        if report.bottlenecks:
            print("\n  Bottlenecks detected:")
            for bottleneck in report.bottlenecks[:3]:
                print(f"    - {bottleneck}")

    except Exception as e:
        print(f"[FAIL] Profiling failed: {e}")
        import traceback
        traceback.print_exc()

    # 3. Regression Testing
    print("\n[3/3] Regression Testing")
    print("-" * 80)
    regression_tester = RegressionTester()

    try:
        print("\nTest 3.1: Create Baseline")
        # Just run single agent test to create baseline
        result = await regression_tester.test_single_agent_performance(update_baseline=True)
        print(f"[OK] Baseline created")
        print(f"  p95 latency: {result.current.p95_latency_ms:.2f}ms")
        if hasattr(result.current, 'success_rate'):
            print(f"  Success rate: {result.current.success_rate*100:.1f}%")
        else:
            print(f"  Error rate: {result.current.error_rate*100:.1f}%")

        print("\nTest 3.2: Validate Against Baseline")
        result2 = await regression_tester.test_single_agent_performance(update_baseline=False)
        status = "PASS" if result2.passed else "FAIL"
        print(f"  Status: {status}")

        if result2.regression_detected:
            print("  [WARNING] Regression detected:")
            for detail in result2.regression_details:
                print(f"    - {detail}")

    except Exception as e:
        print(f"[FAIL] Regression testing failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*80)
    print("PERFORMANCE TESTS COMPLETE")
    print("="*80)
    print()
    print("Results saved to:")
    print(f"  - tests/performance/results/")
    print()


if __name__ == "__main__":
    asyncio.run(run_quick_tests())
