# -*- coding: utf-8 -*-
"""
Calculation Engine Performance Benchmarks

Performance Targets:
- Single calculation: <100ms
- Batch 100: <1 second
- Batch 1000: <5 seconds
- Batch 10000: <30 seconds

Run with: python -m benchmarks.calculation_performance
"""

import time
from statistics import mean, stdev
from greenlang.calculation import (
    EmissionCalculator,
    CalculationRequest,
    BatchCalculator,
)


def benchmark_single_calculation(n_runs: int = 1000):
    """
    Benchmark single calculation performance.

    Target: <100ms per calculation
    """
    print("\n" + "="*60)
    print("BENCHMARK: Single Calculation Performance")
    print("="*60)

    calculator = EmissionCalculator()
    durations = []

    for i in range(n_runs):
        request = CalculationRequest(
            factor_id='diesel',
            activity_amount=100 + i,  # Vary input
            activity_unit='gallons',
        )

        start = time.perf_counter()
        result = calculator.calculate(request)
        duration_ms = (time.perf_counter() - start) * 1000

        durations.append(duration_ms)

    avg_duration = mean(durations)
    std_duration = stdev(durations) if len(durations) > 1 else 0
    min_duration = min(durations)
    max_duration = max(durations)
    p95_duration = sorted(durations)[int(len(durations) * 0.95)]

    print(f"Runs: {n_runs}")
    print(f"Average: {avg_duration:.2f} ms")
    print(f"Std Dev: {std_duration:.2f} ms")
    print(f"Min: {min_duration:.2f} ms")
    print(f"Max: {max_duration:.2f} ms")
    print(f"P95: {p95_duration:.2f} ms")

    # Check target
    target = 100  # ms
    status = "✓ PASS" if avg_duration < target else "✗ FAIL"
    print(f"Target: <{target} ms ... {status}")

    return avg_duration < target


def benchmark_batch_100(n_runs: int = 10):
    """
    Benchmark batch of 100 calculations.

    Target: <1 second
    """
    print("\n" + "="*60)
    print("BENCHMARK: Batch 100 Calculations")
    print("="*60)

    batch_calc = BatchCalculator()
    durations = []

    for run in range(n_runs):
        # Create 100 requests
        requests = [
            CalculationRequest(
                factor_id='diesel' if i % 2 == 0 else 'natural_gas',
                activity_amount=100 + i,
                activity_unit='gallons' if i % 2 == 0 else 'therms',
            )
            for i in range(100)
        ]

        start = time.perf_counter()
        result = batch_calc.calculate_batch(requests)
        duration_s = time.perf_counter() - start

        durations.append(duration_s)

    avg_duration = mean(durations)
    std_duration = stdev(durations) if len(durations) > 1 else 0
    min_duration = min(durations)
    max_duration = max(durations)

    print(f"Runs: {n_runs}")
    print(f"Average: {avg_duration:.3f} s")
    print(f"Std Dev: {std_duration:.3f} s")
    print(f"Min: {min_duration:.3f} s")
    print(f"Max: {max_duration:.3f} s")
    print(f"Throughput: {100/avg_duration:.1f} calc/sec")

    # Check target
    target = 1.0  # seconds
    status = "✓ PASS" if avg_duration < target else "✗ FAIL"
    print(f"Target: <{target} s ... {status}")

    return avg_duration < target


def benchmark_batch_1000(n_runs: int = 5):
    """
    Benchmark batch of 1000 calculations.

    Target: <5 seconds
    """
    print("\n" + "="*60)
    print("BENCHMARK: Batch 1000 Calculations")
    print("="*60)

    batch_calc = BatchCalculator()
    durations = []

    for run in range(n_runs):
        # Create 1000 requests
        requests = [
            CalculationRequest(
                factor_id=['diesel', 'natural_gas', 'gasoline_motor'][i % 3],
                activity_amount=100 + i,
                activity_unit=['gallons', 'therms', 'gallons'][i % 3],
            )
            for i in range(1000)
        ]

        start = time.perf_counter()
        result = batch_calc.calculate_batch(requests)
        duration_s = time.perf_counter() - start

        durations.append(duration_s)

    avg_duration = mean(durations)
    std_duration = stdev(durations) if len(durations) > 1 else 0
    min_duration = min(durations)
    max_duration = max(durations)

    print(f"Runs: {n_runs}")
    print(f"Average: {avg_duration:.3f} s")
    print(f"Std Dev: {std_duration:.3f} s")
    print(f"Min: {min_duration:.3f} s")
    print(f"Max: {max_duration:.3f} s")
    print(f"Throughput: {1000/avg_duration:.1f} calc/sec")

    # Check target
    target = 5.0  # seconds
    status = "✓ PASS" if avg_duration < target else "✗ FAIL"
    print(f"Target: <{target} s ... {status}")

    return avg_duration < target


def benchmark_uncertainty_propagation(n_runs: int = 100):
    """
    Benchmark uncertainty propagation (Monte Carlo simulation).

    Target: <1 second for 10K simulations
    """
    print("\n" + "="*60)
    print("BENCHMARK: Uncertainty Propagation (10K simulations)")
    print("="*60)

    from greenlang.agents.calculation.emissions.uncertainty import UncertaintyCalculator

    unc_calc = UncertaintyCalculator()
    durations = []

    for i in range(n_runs):
        start = time.perf_counter()

        result = unc_calc.propagate_uncertainty(
            activity_data=100,
            activity_uncertainty_pct=5,
            emission_factor=10.21,
            factor_uncertainty_pct=10,
            n_simulations=10000,
        )

        duration_s = time.perf_counter() - start
        durations.append(duration_s)

    avg_duration = mean(durations)
    std_duration = stdev(durations) if len(durations) > 1 else 0
    min_duration = min(durations)
    max_duration = max(durations)

    print(f"Runs: {n_runs}")
    print(f"Average: {avg_duration:.3f} s")
    print(f"Std Dev: {std_duration:.3f} s")
    print(f"Min: {min_duration:.3f} s")
    print(f"Max: {max_duration:.3f} s")

    # Check target
    target = 1.0  # seconds
    status = "✓ PASS" if avg_duration < target else "✗ FAIL"
    print(f"Target: <{target} s ... {status}")

    return avg_duration < target


def benchmark_gas_decomposition(n_runs: int = 10000):
    """
    Benchmark gas decomposition performance.

    Target: <1ms per decomposition
    """
    print("\n" + "="*60)
    print("BENCHMARK: Gas Decomposition Performance")
    print("="*60)

    from greenlang.agents.calculation.emissions.gas_decomposition import MultiGasCalculator

    gas_calc = MultiGasCalculator()
    durations = []

    for i in range(n_runs):
        start = time.perf_counter()

        result = gas_calc.decompose(
            total_co2e_kg=1000 + i,
            fuel_type='natural_gas',
        )

        duration_ms = (time.perf_counter() - start) * 1000
        durations.append(duration_ms)

    avg_duration = mean(durations)
    std_duration = stdev(durations) if len(durations) > 1 else 0
    min_duration = min(durations)
    max_duration = max(durations)

    print(f"Runs: {n_runs}")
    print(f"Average: {avg_duration:.4f} ms")
    print(f"Std Dev: {std_duration:.4f} ms")
    print(f"Min: {min_duration:.4f} ms")
    print(f"Max: {max_duration:.4f} ms")

    # Check target
    target = 1.0  # ms
    status = "✓ PASS" if avg_duration < target else "✗ FAIL"
    print(f"Target: <{target} ms ... {status}")

    return avg_duration < target


def run_all_benchmarks():
    """Run all performance benchmarks"""
    print("\n" + "="*60)
    print("GREENLANG CALCULATION ENGINE PERFORMANCE BENCHMARKS")
    print("="*60)

    results = {
        'Single Calculation': benchmark_single_calculation(),
        'Batch 100': benchmark_batch_100(),
        'Batch 1000': benchmark_batch_1000(),
        'Uncertainty (10K MC)': benchmark_uncertainty_propagation(),
        'Gas Decomposition': benchmark_gas_decomposition(),
    }

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    for benchmark, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{benchmark:.<40} {status}")

    total_passed = sum(results.values())
    total_tests = len(results)

    print(f"\nTotal: {total_passed}/{total_tests} benchmarks passed")

    if total_passed == total_tests:
        print("\n🎉 ALL PERFORMANCE TARGETS MET!")
    else:
        print(f"\n⚠️  {total_tests - total_passed} benchmark(s) failed")


if __name__ == '__main__':
    run_all_benchmarks()
