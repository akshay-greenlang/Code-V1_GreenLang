#!/usr/bin/env python
"""
CBAM Importer Copilot - Performance Benchmarks

Comprehensive performance benchmarking script.

Measures:
- Agent execution time
- Throughput (records/second)
- Memory usage
- CPU utilization
- End-to-end pipeline performance

Version: 1.0.0
"""

import sys
import time
import json
import pandas as pd
import psutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from cbam_pipeline import CBAMPipeline
from agents.shipment_intake_agent import ShipmentIntakeAgent
from agents.emissions_calculator_agent import EmissionsCalculatorAgent
from agents.reporting_packager_agent import ReportingPackagerAgent


# ============================================================================
# Benchmark Configuration
# ============================================================================

BENCHMARK_CONFIGS = {
    'small': {
        'name': 'Small Dataset (100 records)',
        'num_records': 100,
        'expected_duration': 1.0  # seconds
    },
    'medium': {
        'name': 'Medium Dataset (1,000 records)',
        'num_records': 1000,
        'expected_duration': 5.0
    },
    'large': {
        'name': 'Large Dataset (10,000 records)',
        'num_records': 10000,
        'expected_duration': 30.0
    },
    'xlarge': {
        'name': 'Extra Large Dataset (100,000 records)',
        'num_records': 100000,
        'expected_duration': 300.0
    }
}


# ============================================================================
# Test Data Generation
# ============================================================================

def generate_test_dataset(num_records: int, output_path: str) -> str:
    """Generate test dataset with specified number of records."""
    print(f"Generating {num_records:,} test records...")

    records = []
    cn_codes = ['72071100', '72071200', '28182000', '27011100', '27011200']
    countries = ['CN', 'IN', 'RU', 'TR', 'BR']
    suppliers = [f'SUP-{country}-{i:03d}' for country in countries for i in range(1, 6)]

    for i in range(num_records):
        record = {
            'cn_code': cn_codes[i % len(cn_codes)],
            'country_of_origin': countries[i % len(countries)],
            'quantity_tons': round(10.0 + (i % 50), 2),
            'import_date': f'2025-09-{(i % 28) + 1:02d}',
            'supplier_id': suppliers[i % len(suppliers)],
            'invoice_number': f'INV-2025-{i:06d}'
        }
        records.append(record)

    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)

    print(f"✓ Generated dataset: {output_path}")
    return output_path


# ============================================================================
# Performance Measurement Utilities
# ============================================================================

class PerformanceMonitor:
    """Monitor CPU, memory, and timing during execution."""

    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.start_memory = None
        self.end_memory = None
        self.process = psutil.Process()

    def start(self):
        """Start monitoring."""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB

    def stop(self):
        """Stop monitoring."""
        self.end_time = time.time()
        self.end_memory = self.process.memory_info().rss / 1024 / 1024  # MB

    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics."""
        duration = self.end_time - self.start_time
        memory_delta = self.end_memory - self.start_memory

        return {
            'duration_seconds': round(duration, 3),
            'start_memory_mb': round(self.start_memory, 2),
            'end_memory_mb': round(self.end_memory, 2),
            'memory_delta_mb': round(memory_delta, 2),
            'cpu_percent': round(self.process.cpu_percent(), 2)
        }


def calculate_throughput(num_records: int, duration_seconds: float) -> float:
    """Calculate throughput in records per second."""
    return num_records / duration_seconds if duration_seconds > 0 else 0.0


# ============================================================================
# Agent Benchmarks
# ============================================================================

def benchmark_agent_1(test_file: str, num_records: int) -> Dict[str, Any]:
    """Benchmark Agent 1 (Shipment Intake)."""
    print("\n[Agent 1] Shipment Intake Agent")
    print("=" * 60)

    monitor = PerformanceMonitor()

    agent = ShipmentIntakeAgent(
        cn_codes_path="data/cn_codes.json",
        cbam_rules_path="rules/cbam_rules.yaml"
    )

    importer_info = {
        'country': 'NL',
        'name': 'Test Company',
        'eori_number': 'NL123456789'
    }

    monitor.start()
    result = agent.process(test_file, importer_info)
    monitor.stop()

    metrics = monitor.get_metrics()
    throughput = calculate_throughput(num_records, metrics['duration_seconds'])

    print(f"Duration:    {metrics['duration_seconds']:.3f}s")
    print(f"Throughput:  {throughput:,.0f} records/second")
    print(f"Memory:      {metrics['memory_delta_mb']:.2f} MB")

    return {
        'agent': 'ShipmentIntakeAgent',
        'records_processed': num_records,
        'throughput_rps': round(throughput, 2),
        **metrics
    }


def benchmark_agent_2(validated_data: Dict[str, Any], num_records: int) -> Dict[str, Any]:
    """Benchmark Agent 2 (Emissions Calculator)."""
    print("\n[Agent 2] Emissions Calculator Agent")
    print("=" * 60)

    monitor = PerformanceMonitor()

    agent = EmissionsCalculatorAgent(cn_codes_path="data/cn_codes.json")

    monitor.start()
    result = agent.calculate(validated_data)
    monitor.stop()

    metrics = monitor.get_metrics()
    throughput = calculate_throughput(num_records, metrics['duration_seconds'])
    ms_per_calculation = (metrics['duration_seconds'] * 1000) / num_records

    print(f"Duration:         {metrics['duration_seconds']:.3f}s")
    print(f"Throughput:       {throughput:,.0f} records/second")
    print(f"Per-calculation:  {ms_per_calculation:.3f}ms")
    print(f"Memory:           {metrics['memory_delta_mb']:.2f} MB")

    return {
        'agent': 'EmissionsCalculatorAgent',
        'records_processed': num_records,
        'throughput_rps': round(throughput, 2),
        'ms_per_calculation': round(ms_per_calculation, 3),
        **metrics
    }


def benchmark_agent_3(emissions_data: Dict[str, Any], num_records: int) -> Dict[str, Any]:
    """Benchmark Agent 3 (Reporting Packager)."""
    print("\n[Agent 3] Reporting Packager Agent")
    print("=" * 60)

    monitor = PerformanceMonitor()

    agent = ReportingPackagerAgent()

    importer_info = {
        'country': 'NL',
        'name': 'Test Company',
        'eori_number': 'NL123456789'
    }

    metadata = {
        'input_file': 'benchmark.csv',
        'processing_date': datetime.now().isoformat()
    }

    monitor.start()
    result = agent.generate_report(emissions_data, importer_info, metadata)
    monitor.stop()

    metrics = monitor.get_metrics()
    throughput = calculate_throughput(num_records, metrics['duration_seconds'])

    print(f"Duration:    {metrics['duration_seconds']:.3f}s")
    print(f"Throughput:  {throughput:,.0f} records/second")
    print(f"Memory:      {metrics['memory_delta_mb']:.2f} MB")

    return {
        'agent': 'ReportingPackagerAgent',
        'records_processed': num_records,
        'throughput_rps': round(throughput, 2),
        **metrics
    }


# ============================================================================
# End-to-End Pipeline Benchmark
# ============================================================================

def benchmark_pipeline(test_file: str, num_records: int) -> Dict[str, Any]:
    """Benchmark complete end-to-end pipeline."""
    print("\n[Pipeline] End-to-End CBAM Pipeline")
    print("=" * 60)

    monitor = PerformanceMonitor()

    pipeline = CBAMPipeline(
        cn_codes_path="data/cn_codes.json",
        cbam_rules_path="rules/cbam_rules.yaml",
        enable_provenance=True
    )

    importer_info = {
        'country': 'NL',
        'name': 'Test Company',
        'eori_number': 'NL123456789'
    }

    output_path = f"benchmark_output_{num_records}.json"

    monitor.start()
    result = pipeline.run(test_file, importer_info, output_path)
    monitor.stop()

    metrics = monitor.get_metrics()
    throughput = calculate_throughput(num_records, metrics['duration_seconds'])

    print(f"Duration:    {metrics['duration_seconds']:.3f}s")
    print(f"Throughput:  {throughput:,.0f} records/second")
    print(f"Memory:      {metrics['memory_delta_mb']:.2f} MB")
    print(f"Output:      {output_path}")

    # Clean up output file
    Path(output_path).unlink(missing_ok=True)

    return {
        'pipeline': 'Complete CBAM Pipeline',
        'records_processed': num_records,
        'throughput_rps': round(throughput, 2),
        **metrics
    }


# ============================================================================
# Reproducibility Benchmark
# ============================================================================

def benchmark_reproducibility(test_file: str, num_records: int, num_runs: int = 10) -> Dict[str, Any]:
    """Benchmark reproducibility (bit-perfect determinism)."""
    print("\n[Reproducibility] Zero Hallucination Verification")
    print("=" * 60)

    pipeline = CBAMPipeline(
        cn_codes_path="data/cn_codes.json",
        cbam_rules_path="rules/cbam_rules.yaml",
        enable_provenance=False  # Disable for speed
    )

    importer_info = {
        'country': 'NL',
        'name': 'Test Company',
        'eori_number': 'NL123456789'
    }

    print(f"Running {num_runs} identical executions...")

    emissions = []
    durations = []

    for i in range(num_runs):
        start = time.time()
        result = pipeline.run(test_file, importer_info, f"repro_{i}.json")
        duration = time.time() - start

        total_emissions = result['emissions_summary']['total_embedded_emissions_tco2']
        emissions.append(total_emissions)
        durations.append(duration)

        # Clean up
        Path(f"repro_{i}.json").unlink(missing_ok=True)

        print(f"  Run {i+1:2d}: {total_emissions:.6f} tCO2 ({duration:.3f}s)")

    # Check determinism
    unique_emissions = set(emissions)
    is_deterministic = len(unique_emissions) == 1

    avg_duration = sum(durations) / len(durations)
    std_duration = (sum((d - avg_duration) ** 2 for d in durations) / len(durations)) ** 0.5

    print(f"\nUnique emission values: {len(unique_emissions)}")
    print(f"Deterministic:          {is_deterministic}")
    print(f"Avg duration:           {avg_duration:.3f}s")
    print(f"Std deviation:          {std_duration:.3f}s")

    return {
        'test': 'Reproducibility',
        'num_runs': num_runs,
        'is_deterministic': is_deterministic,
        'unique_values': len(unique_emissions),
        'avg_duration_seconds': round(avg_duration, 3),
        'std_deviation_seconds': round(std_duration, 3)
    }


# ============================================================================
# Main Benchmark Suite
# ============================================================================

def run_benchmark_suite(config_name: str = 'medium') -> Dict[str, Any]:
    """Run complete benchmark suite."""
    config = BENCHMARK_CONFIGS[config_name]
    num_records = config['num_records']

    print("=" * 60)
    print(f"CBAM Importer Copilot - Performance Benchmarks")
    print(f"Configuration: {config['name']}")
    print("=" * 60)

    # Generate test data
    test_file = f"benchmark_data_{num_records}.csv"
    generate_test_dataset(num_records, test_file)

    results = {
        'benchmark_config': config,
        'timestamp': datetime.now().isoformat(),
        'system_info': {
            'cpu_count': psutil.cpu_count(),
            'total_memory_mb': round(psutil.virtual_memory().total / 1024 / 1024, 2),
            'platform': sys.platform
        },
        'benchmarks': {}
    }

    try:
        # Benchmark Agent 1
        agent1_results = benchmark_agent_1(test_file, num_records)
        results['benchmarks']['agent1'] = agent1_results

        # Need validated data for Agent 2
        agent = ShipmentIntakeAgent(
            cn_codes_path="data/cn_codes.json",
            cbam_rules_path="rules/cbam_rules.yaml"
        )
        validated_data = agent.process(test_file, {
            'country': 'NL',
            'name': 'Test Company',
            'eori_number': 'NL123456789'
        })

        # Benchmark Agent 2
        agent2_results = benchmark_agent_2(validated_data, num_records)
        results['benchmarks']['agent2'] = agent2_results

        # Need emissions data for Agent 3
        calc_agent = EmissionsCalculatorAgent(cn_codes_path="data/cn_codes.json")
        emissions_data = calc_agent.calculate(validated_data)

        # Benchmark Agent 3
        agent3_results = benchmark_agent_3(emissions_data, num_records)
        results['benchmarks']['agent3'] = agent3_results

        # Benchmark complete pipeline
        pipeline_results = benchmark_pipeline(test_file, num_records)
        results['benchmarks']['pipeline'] = pipeline_results

        # Benchmark reproducibility (only for small/medium datasets)
        if num_records <= 1000:
            repro_results = benchmark_reproducibility(test_file, num_records, num_runs=5)
            results['benchmarks']['reproducibility'] = repro_results

        # Performance targets check
        print("\n" + "=" * 60)
        print("Performance Targets")
        print("=" * 60)

        target_throughput = 1000  # records/second
        target_calc_time = 3.0     # ms per calculation

        agent2_throughput = agent2_results['throughput_rps']
        agent2_ms = agent2_results['ms_per_calculation']

        print(f"Target throughput:      {target_throughput:,} records/sec")
        print(f"Achieved throughput:    {agent2_throughput:,.0f} records/sec")
        print(f"Status:                 {'✓ PASS' if agent2_throughput >= target_throughput else '✗ FAIL'}")
        print()
        print(f"Target calc time:       {target_calc_time:.3f} ms/calculation")
        print(f"Achieved calc time:     {agent2_ms:.3f} ms/calculation")
        print(f"Status:                 {'✓ PASS' if agent2_ms <= target_calc_time else '✗ FAIL'}")

        results['performance_targets'] = {
            'throughput_target': target_throughput,
            'throughput_achieved': agent2_throughput,
            'throughput_pass': agent2_throughput >= target_throughput,
            'calc_time_target_ms': target_calc_time,
            'calc_time_achieved_ms': agent2_ms,
            'calc_time_pass': agent2_ms <= target_calc_time
        }

    finally:
        # Clean up test file
        Path(test_file).unlink(missing_ok=True)

    return results


def save_benchmark_results(results: Dict[str, Any], output_path: str = "benchmark_results.json"):
    """Save benchmark results to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Benchmark results saved to: {output_path}")


# ============================================================================
# CLI
# ============================================================================

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='CBAM Importer Copilot Performance Benchmarks')
    parser.add_argument(
        '--config',
        choices=['small', 'medium', 'large', 'xlarge'],
        default='medium',
        help='Benchmark configuration (default: medium)'
    )
    parser.add_argument(
        '--output',
        default='benchmark_results.json',
        help='Output file for results (default: benchmark_results.json)'
    )

    args = parser.parse_args()

    # Run benchmarks
    results = run_benchmark_suite(args.config)

    # Save results
    save_benchmark_results(results, args.output)

    print("\n" + "=" * 60)
    print("Benchmark suite complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
