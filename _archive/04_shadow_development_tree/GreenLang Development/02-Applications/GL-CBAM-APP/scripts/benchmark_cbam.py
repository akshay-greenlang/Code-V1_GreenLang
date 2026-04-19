#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GL-CBAM-APP - Performance Benchmark Script

Purpose: Validate the claimed 20× speedup and measure performance
         characteristics under different load scenarios.

Usage:
    python scripts/benchmark_cbam.py                    # Run all benchmarks
    python scripts/benchmark_cbam.py --quick            # Quick benchmarks only
    python scripts/benchmark_cbam.py --dataset 10k      # Specific dataset size
    python scripts/benchmark_cbam.py --compare          # Compare with baseline
    python scripts/benchmark_cbam.py --report           # Generate detailed report

Scenarios:
    - 1K shipments (baseline)
    - 10K shipments (typical quarterly reporting)
    - 100K shipments (large importer, stress test)

Metrics:
    - Throughput (shipments/second)
    - Latency (ms per shipment)
    - Memory usage (MB)
    - CPU utilization (%)

Version: 1.0.0
"""

import sys
import time
import json
import psutil
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
from decimal import Decimal
import pandas as pd
import traceback
from greenlang.determinism import DeterministicClock

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "CBAM-Importer-Copilot"))

# ============================================================================
# Configuration
# ============================================================================

BENCHMARK_SCENARIOS = {
    "1k": 1_000,
    "10k": 10_000,
    "100k": 100_000
}

# Performance targets
TARGETS = {
    "throughput_min_per_sec": 100,  # Minimum 100 shipments/second
    "latency_max_ms": 10,            # Maximum 10ms per shipment
    "memory_max_mb": 500,            # Maximum 500 MB for 100K shipments
    "speedup_target": 20             # 20× speedup vs traditional methods
}

# ============================================================================
# Benchmark Data Generation
# ============================================================================

def generate_benchmark_data(size: int) -> List[Dict[str, Any]]:
    """Generate synthetic shipment data for benchmarking."""
    print(f"Generating {size:,} shipments for benchmarking...")

    base_shipments = [
        {
            "cn_code": "72071100",
            "country_of_origin": "CN",
            "quantity_tons": 15.5,
            "supplier_id": "SUP-CN-001"
        },
        {
            "cn_code": "76011000",
            "country_of_origin": "RU",
            "quantity_tons": 12.0,
            "supplier_id": "SUP-RU-001"
        },
        {
            "cn_code": "25232900",
            "country_of_origin": "UA",
            "quantity_tons": 20.5,
            "supplier_id": "SUP-UA-001"
        },
        {
            "cn_code": "28112100",
            "country_of_origin": "IN",
            "quantity_tons": 8.0,
            "supplier_id": "SUP-IN-001"
        },
        {
            "cn_code": "28342100",
            "country_of_origin": "TR",
            "quantity_tons": 5.5,
            "supplier_id": "SUP-TR-001"
        }
    ]

    shipments = []
    num_iterations = size // len(base_shipments)

    for i in range(num_iterations):
        for j, base in enumerate(base_shipments):
            shipment = {
                **base,
                "shipment_id": f"BENCH-{size}-{i:06d}-{j:02d}",
                "import_date": f"2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
                "quarter": f"Q{((i % 12) // 3) + 1}-2024",
                "invoice_number": f"INV-BENCH-{i:06d}",
                "importer_country": "NL",
                "importer_eori": "NL123456789012"
            }
            shipments.append(shipment)

    print(f"✓ Generated {len(shipments):,} shipments")
    return shipments


# ============================================================================
# Performance Measurement
# ============================================================================

class PerformanceMonitor:
    """Monitor performance metrics during benchmark execution."""

    def __init__(self):
        self.process = psutil.Process()
        self.start_time = None
        self.start_memory = None
        self.start_cpu = None

    def start(self):
        """Start monitoring."""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / (1024 * 1024)  # MB
        self.start_cpu = self.process.cpu_percent(interval=0.1)

    def stop(self) -> Dict[str, Any]:
        """Stop monitoring and return metrics."""
        end_time = time.time()
        end_memory = self.process.memory_info().rss / (1024 * 1024)  # MB
        end_cpu = self.process.cpu_percent(interval=0.1)

        duration = end_time - self.start_time
        memory_used = end_memory - self.start_memory
        peak_memory = self.process.memory_info().rss / (1024 * 1024)  # MB

        return {
            "duration_seconds": round(duration, 3),
            "memory_used_mb": round(memory_used, 2),
            "peak_memory_mb": round(peak_memory, 2),
            "cpu_percent": round((self.start_cpu + end_cpu) / 2, 2)
        }


# ============================================================================
# Benchmark Execution
# ============================================================================

def run_pipeline_benchmark(shipments: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Run CBAM pipeline benchmark."""
    try:
        from cbam_pipeline import CBAMPipeline

        # Initialize pipeline
        pipeline = CBAMPipeline(
            cn_codes_path=str(PROJECT_ROOT / "CBAM-Importer-Copilot" / "data" / "cn_codes.json"),
            cbam_rules_path=str(PROJECT_ROOT / "CBAM-Importer-Copilot" / "rules" / "cbam_rules.yaml"),
            enable_provenance=False  # Disable for pure performance test
        )

        # Prepare importer info
        importer_info = {
            "name": "Benchmark Test Company",
            "country": "NL",
            "eori": "NL123456789012",
            "declarant_name": "Benchmark User",
            "declarant_position": "Test Manager"
        }

        # Start monitoring
        monitor = PerformanceMonitor()
        monitor.start()

        # Run pipeline
        result = pipeline.run(
            shipments_data=shipments,
            importer_info=importer_info
        )

        # Stop monitoring
        metrics = monitor.stop()

        # Extract results
        if isinstance(result, dict):
            total_emissions = result.get("emissions_summary", {}).get("total_embedded_emissions_tco2", 0)
            total_shipments = result.get("emissions_summary", {}).get("total_shipments", 0)
        else:
            total_emissions = 0
            total_shipments = len(shipments)

        return metrics, {
            "total_shipments": total_shipments,
            "total_emissions_tco2": total_emissions,
            "success": True
        }

    except Exception as e:
        print(f"✗ Benchmark failed: {e}")
        traceback.print_exc()
        return {}, {"success": False, "error": str(e)}


def calculate_benchmark_stats(shipment_count: int, metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate benchmark statistics."""
    duration = metrics.get("duration_seconds", 0)

    if duration > 0:
        throughput = shipment_count / duration
        latency_ms = (duration / shipment_count) * 1000
    else:
        throughput = 0
        latency_ms = 0

    return {
        "shipment_count": shipment_count,
        "duration_seconds": metrics.get("duration_seconds", 0),
        "throughput_per_second": round(throughput, 2),
        "latency_ms_per_shipment": round(latency_ms, 3),
        "memory_used_mb": metrics.get("memory_used_mb", 0),
        "peak_memory_mb": metrics.get("peak_memory_mb", 0),
        "cpu_percent": metrics.get("cpu_percent", 0)
    }


def evaluate_performance(stats: Dict[str, Any]) -> Dict[str, bool]:
    """Evaluate if performance meets targets."""
    return {
        "throughput_ok": stats["throughput_per_second"] >= TARGETS["throughput_min_per_sec"],
        "latency_ok": stats["latency_ms_per_shipment"] <= TARGETS["latency_max_ms"],
        "memory_ok": stats["peak_memory_mb"] <= TARGETS["memory_max_mb"],
        "overall_pass": (
            stats["throughput_per_second"] >= TARGETS["throughput_min_per_sec"] and
            stats["latency_ms_per_shipment"] <= TARGETS["latency_max_ms"] and
            stats["peak_memory_mb"] <= TARGETS["memory_max_mb"]
        )
    }


# ============================================================================
# Reporting
# ============================================================================

def print_benchmark_results(scenario: str, stats: Dict[str, Any], evaluation: Dict[str, bool]):
    """Print benchmark results to console."""
    print(f"\n{'='*70}")
    print(f"Benchmark Results: {scenario}")
    print(f"{'='*70}")

    print(f"\nDataset:")
    print(f"  Shipments:           {stats['shipment_count']:>10,}")

    print(f"\nPerformance:")
    print(f"  Duration:            {stats['duration_seconds']:>10.2f} seconds")
    print(f"  Throughput:          {stats['throughput_per_second']:>10.2f} shipments/sec")
    print(f"  Latency:             {stats['latency_ms_per_shipment']:>10.3f} ms/shipment")

    print(f"\nResource Usage:")
    print(f"  Memory Used:         {stats['memory_used_mb']:>10.2f} MB")
    print(f"  Peak Memory:         {stats['peak_memory_mb']:>10.2f} MB")
    print(f"  CPU Usage:           {stats['cpu_percent']:>10.2f} %")

    print(f"\nTargets:")
    status = "✓ PASS" if evaluation["throughput_ok"] else "✗ FAIL"
    print(f"  Throughput:          {status} (target: ≥{TARGETS['throughput_min_per_sec']} shipments/sec)")

    status = "✓ PASS" if evaluation["latency_ok"] else "✗ FAIL"
    print(f"  Latency:             {status} (target: ≤{TARGETS['latency_max_ms']} ms)")

    status = "✓ PASS" if evaluation["memory_ok"] else "✗ FAIL"
    print(f"  Memory:              {status} (target: ≤{TARGETS['memory_max_mb']} MB)")

    print(f"\nOverall:               ", end="")
    if evaluation["overall_pass"]:
        print("✓ PASS - All targets met!")
    else:
        print("✗ FAIL - Some targets not met")

    print(f"{'='*70}\n")


def generate_benchmark_report(results: Dict[str, Dict[str, Any]], output_path: Path):
    """Generate detailed benchmark report."""
    report = {
        "benchmark_metadata": {
            "timestamp": DeterministicClock.now().isoformat(),
            "system_info": {
                "cpu_count": psutil.cpu_count(),
                "total_memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "python_version": sys.version
            },
            "targets": TARGETS
        },
        "results": results,
        "summary": {
            "scenarios_run": len(results),
            "scenarios_passed": sum(1 for r in results.values() if r.get("evaluation", {}).get("overall_pass", False)),
            "all_passed": all(r.get("evaluation", {}).get("overall_pass", False) for r in results.values())
        }
    }

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"✓ Detailed report saved: {output_path}")


# ============================================================================
# Main Benchmark Runner
# ============================================================================

def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(description="GL-CBAM Performance Benchmark")
    parser.add_argument("--quick", action="store_true", help="Quick benchmark (1K only)")
    parser.add_argument("--dataset", choices=["1k", "10k", "100k"], help="Specific dataset size")
    parser.add_argument("--report", action="store_true", help="Generate detailed JSON report")
    parser.add_argument("--output", default="benchmark-results.json", help="Output file for report")

    args = parser.parse_args()

    print("\n" + "="*70)
    print("GL-CBAM Performance Benchmark Suite")
    print("="*70)
    print(f"Timestamp: {DeterministicClock.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"System: {psutil.cpu_count()} CPUs, {psutil.virtual_memory().total / (1024**3):.1f} GB RAM")
    print("="*70)

    # Determine which scenarios to run
    if args.quick or args.dataset == "1k":
        scenarios = {"1k": BENCHMARK_SCENARIOS["1k"]}
    elif args.dataset:
        scenarios = {args.dataset: BENCHMARK_SCENARIOS[args.dataset]}
    else:
        scenarios = BENCHMARK_SCENARIOS

    all_results = {}

    # Run benchmarks
    for scenario_name, size in scenarios.items():
        print(f"\n\nRunning benchmark: {scenario_name.upper()} ({size:,} shipments)")
        print("-" * 70)

        # Generate data
        shipments = generate_benchmark_data(size)

        # Run benchmark
        print(f"Executing CBAM pipeline...")
        metrics, result = run_pipeline_benchmark(shipments)

        if result.get("success"):
            # Calculate stats
            stats = calculate_benchmark_stats(size, metrics)
            evaluation = evaluate_performance(stats)

            # Print results
            print_benchmark_results(scenario_name.upper(), stats, evaluation)

            # Store results
            all_results[scenario_name] = {
                "stats": stats,
                "evaluation": evaluation,
                "result": result
            }
        else:
            print(f"✗ Benchmark {scenario_name} failed: {result.get('error')}")
            all_results[scenario_name] = {
                "success": False,
                "error": result.get("error")
            }

    # Generate report if requested
    if args.report:
        output_path = PROJECT_ROOT / args.output
        generate_benchmark_report(all_results, output_path)

    # Final summary
    print("\n" + "="*70)
    print("Benchmark Suite Summary")
    print("="*70)

    total_scenarios = len(all_results)
    passed_scenarios = sum(1 for r in all_results.values() if r.get("evaluation", {}).get("overall_pass", False))

    print(f"Scenarios Run:    {total_scenarios}")
    print(f"Scenarios Passed: {passed_scenarios}")
    print(f"Success Rate:     {(passed_scenarios/total_scenarios)*100:.1f}%")

    if passed_scenarios == total_scenarios:
        print("\n✓ ALL BENCHMARKS PASSED!")
        print(f"✓ GL-CBAM meets all performance targets")
        return 0
    else:
        print(f"\n✗ {total_scenarios - passed_scenarios} benchmark(s) failed")
        print("  Review results and optimize performance")
        return 1


if __name__ == "__main__":
    sys.exit(main())
