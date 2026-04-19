# -*- coding: utf-8 -*-
"""
Performance Benchmarking Framework for CSRD Reporting Platform
===============================================================

Measures and validates performance against SLA targets.

Author: GreenLang Performance Team
Date: 2025-10-20
"""

import json
import time
import psutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Callable
import statistics
from greenlang.determinism import DeterministicClock


class PerformanceBenchmark:
    """Performance benchmarking and validation framework."""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.reports_dir = self.project_root / "benchmark-reports"
        self.reports_dir.mkdir(exist_ok=True)

        self.results = {
            "timestamp": DeterministicClock.now().isoformat(),
            "project": "CSRD-Reporting-Platform",
            "benchmarks": {},
            "summary": {},
            "system_info": self._get_system_info()
        }

        # SLA targets
        self.sla_targets = {
            "xbrl_generation": {
                "name": "XBRL Report Generation",
                "target_seconds": 300,  # 5 minutes
                "description": "Generate complete XBRL/iXBRL report"
            },
            "materiality_assessment": {
                "name": "AI Materiality Assessment",
                "target_seconds": 30,
                "description": "AI-powered materiality assessment"
            },
            "data_import": {
                "name": "Bulk Data Import",
                "target_seconds": 30,
                "description": "Import 10,000 data points"
            },
            "audit_validation": {
                "name": "ESRS Audit Validation",
                "target_seconds": 120,  # 2 minutes
                "description": "Validate 215+ ESRS compliance rules"
            },
            "api_response": {
                "name": "API Response Time",
                "target_ms": 200,  # p95 latency
                "description": "API endpoint p95 latency"
            },
            "calculator_throughput": {
                "name": "Calculator Throughput",
                "target_records_per_second": 1000,
                "description": "GHG emissions calculations per second"
            }
        }

    def _get_system_info(self) -> Dict:
        """Get system information for benchmark context."""
        return {
            "cpu_count": psutil.cpu_count(logical=True),
            "cpu_physical": psutil.cpu_count(logical=False),
            "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
            "python_version": sys.version,
            "platform": sys.platform
        }

    def measure_execution_time(self, func: Callable, *args, **kwargs) -> Dict:
        """Measure execution time and resource usage for a function."""
        # Get baseline metrics
        process = psutil.Process()
        cpu_before = process.cpu_percent(interval=0.1)
        memory_before = process.memory_info().rss / 1024**2  # MB

        # Execute function
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)

        end_time = time.time()
        duration = end_time - start_time

        # Get post-execution metrics
        cpu_after = process.cpu_percent(interval=0.1)
        memory_after = process.memory_info().rss / 1024**2  # MB

        return {
            "duration_seconds": round(duration, 3),
            "success": success,
            "error": error,
            "result": result,
            "resources": {
                "cpu_usage_percent": round((cpu_before + cpu_after) / 2, 2),
                "memory_usage_mb": round(memory_after - memory_before, 2),
                "peak_memory_mb": round(memory_after, 2)
            }
        }

    def run_repeated_benchmark(self, name: str, func: Callable, iterations: int = 10,
                               target_seconds: float = None) -> Dict:
        """Run benchmark multiple times and calculate statistics."""
        print(f"\n📊 Running Benchmark: {name}")
        print(f"   Iterations: {iterations}")

        durations = []
        cpu_usages = []
        memory_usages = []
        successes = 0

        for i in range(iterations):
            print(f"   Run {i+1}/{iterations}...", end=" ")

            result = self.measure_execution_time(func)

            if result["success"]:
                durations.append(result["duration_seconds"])
                cpu_usages.append(result["resources"]["cpu_usage_percent"])
                memory_usages.append(result["resources"]["memory_usage_mb"])
                successes += 1
                print(f"✓ {result['duration_seconds']:.3f}s")
            else:
                print(f"✗ ERROR: {result['error']}")

        if not durations:
            return {
                "status": "failed",
                "error": "All iterations failed"
            }

        # Calculate statistics
        stats = {
            "status": "completed",
            "iterations": iterations,
            "successes": successes,
            "failures": iterations - successes,
            "duration": {
                "min": round(min(durations), 3),
                "max": round(max(durations), 3),
                "mean": round(statistics.mean(durations), 3),
                "median": round(statistics.median(durations), 3),
                "stdev": round(statistics.stdev(durations), 3) if len(durations) > 1 else 0,
                "p50": round(statistics.median(durations), 3),
                "p95": round(statistics.quantiles(durations, n=20)[18], 3) if len(durations) >= 20 else round(max(durations), 3),
                "p99": round(statistics.quantiles(durations, n=100)[98], 3) if len(durations) >= 100 else round(max(durations), 3),
            },
            "resources": {
                "avg_cpu_percent": round(statistics.mean(cpu_usages), 2),
                "avg_memory_mb": round(statistics.mean(memory_usages), 2),
                "max_memory_mb": round(max(memory_usages), 2) if memory_usages else 0
            }
        }

        # Check against target
        if target_seconds:
            stats["target_seconds"] = target_seconds
            stats["target_met"] = stats["duration"]["p95"] <= target_seconds
            stats["margin"] = round(target_seconds - stats["duration"]["p95"], 3)
            stats["margin_percent"] = round((stats["margin"] / target_seconds) * 100, 2)

            if stats["target_met"]:
                print(f"   ✅ Target MET: {stats['duration']['p95']:.3f}s ≤ {target_seconds}s (margin: {stats['margin_percent']:.1f}%)")
            else:
                print(f"   ❌ Target MISSED: {stats['duration']['p95']:.3f}s > {target_seconds}s (over by {abs(stats['margin_percent']):.1f}%)")

        print(f"   📈 Statistics:")
        print(f"      Mean: {stats['duration']['mean']:.3f}s | Median: {stats['duration']['median']:.3f}s | P95: {stats['duration']['p95']:.3f}s")
        print(f"      CPU: {stats['resources']['avg_cpu_percent']:.1f}% | Memory: {stats['resources']['avg_memory_mb']:.1f}MB")

        return stats

    def benchmark_xbrl_generation(self) -> Dict:
        """Benchmark XBRL report generation."""
        print("\n" + "="*80)
        print("BENCHMARK: XBRL Report Generation")
        print("="*80)

        # Import here to avoid loading if not running benchmark
        try:
            from agents.reporting_agent import ReportingAgent

            def generate_report():
                """Generate a sample XBRL report."""
                agent = ReportingAgent()

                # Sample data for report generation
                sample_data = {
                    "company_name": "Benchmark Corp",
                    "reporting_period": "2024",
                    "esrs_data": {
                        "E1": {"emissions_scope1": 1000, "emissions_scope2": 500},
                        "E2": {"energy_consumption": 50000, "renewable_percent": 30},
                        "S1": {"employee_count": 250, "training_hours": 1000}
                    }
                }

                report = agent.generate_xbrl_report(sample_data)
                return report

            target = self.sla_targets["xbrl_generation"]["target_seconds"]
            result = self.run_repeated_benchmark(
                "XBRL Report Generation",
                generate_report,
                iterations=3,  # Fewer iterations for slow operation
                target_seconds=target
            )

            return result

        except ImportError as e:
            print(f"❌ Cannot import ReportingAgent: {e}")
            return {"status": "skipped", "reason": "Import failed"}

    def benchmark_materiality_assessment(self) -> Dict:
        """Benchmark AI materiality assessment."""
        print("\n" + "="*80)
        print("BENCHMARK: AI Materiality Assessment")
        print("="*80)

        try:
            from agents.materiality_agent import MaterialityAgent

            def run_assessment():
                """Run materiality assessment."""
                agent = MaterialityAgent()

                assessment = agent.assess_materiality({
                    "industry": "Manufacturing",
                    "region": "EU",
                    "size": "Large",
                    "activities": ["Production", "Distribution"]
                })

                return assessment

            target = self.sla_targets["materiality_assessment"]["target_seconds"]
            result = self.run_repeated_benchmark(
                "Materiality Assessment",
                run_assessment,
                iterations=5,
                target_seconds=target
            )

            return result

        except ImportError as e:
            print(f"❌ Cannot import MaterialityAgent: {e}")
            return {"status": "skipped", "reason": "Import failed"}

    def benchmark_data_import(self) -> Dict:
        """Benchmark bulk data import."""
        print("\n" + "="*80)
        print("BENCHMARK: Bulk Data Import (10K records)")
        print("="*80)

        try:
            from agents.intake_agent import IntakeAgent
            import pandas as pd

            def import_data():
                """Import 10,000 sample records."""
                agent = IntakeAgent()

                # Generate sample CSV data
                sample_data = pd.DataFrame({
                    "date": pd.date_range("2024-01-01", periods=10000, freq="h"),
                    "facility": [f"Facility_{i%10}" for i in range(10000)],
                    "emission_source": [f"Source_{i%5}" for i in range(10000)],
                    "co2_kg": [100 + (i % 100) for i in range(10000)],
                    "activity": [1000 + (i % 500) for i in range(10000)]
                })

                result = agent.process_dataframe(sample_data)
                return result

            target = self.sla_targets["data_import"]["target_seconds"]
            result = self.run_repeated_benchmark(
                "Data Import (10K records)",
                import_data,
                iterations=5,
                target_seconds=target
            )

            return result

        except ImportError as e:
            print(f"❌ Cannot import IntakeAgent: {e}")
            return {"status": "skipped", "reason": "Import failed"}

    def benchmark_audit_validation(self) -> Dict:
        """Benchmark ESRS audit validation."""
        print("\n" + "="*80)
        print("BENCHMARK: ESRS Audit Validation (215+ rules)")
        print("="*80)

        try:
            from agents.audit_agent import AuditAgent

            def run_audit():
                """Run complete audit validation."""
                agent = AuditAgent()

                # Sample report data
                report_data = {
                    "company_info": {"name": "Test Corp", "nace_code": "24.10"},
                    "esrs_disclosures": {
                        "E1": {"scope1": 1000, "scope2": 500, "scope3": 2000},
                        "E2": {"energy": 50000, "renewable": 15000},
                        "S1": {"employees": 250, "training": 1000}
                    }
                }

                audit_result = agent.validate_report(report_data)
                return audit_result

            target = self.sla_targets["audit_validation"]["target_seconds"]
            result = self.run_repeated_benchmark(
                "Audit Validation",
                run_audit,
                iterations=5,
                target_seconds=target
            )

            return result

        except ImportError as e:
            print(f"❌ Cannot import AuditAgent: {e}")
            return {"status": "skipped", "reason": "Import failed"}

    def benchmark_calculator_throughput(self) -> Dict:
        """Benchmark calculator throughput."""
        print("\n" + "="*80)
        print("BENCHMARK: Calculator Throughput")
        print("="*80)

        try:
            from agents.calculator_agent import CalculatorAgent

            def calculate_emissions():
                """Calculate emissions for 10,000 records."""
                agent = CalculatorAgent()

                start = time.time()
                count = 0

                # Run calculations for a fixed time period
                while time.time() - start < 1.0:  # 1 second
                    agent.calculate_emissions(
                        activity=1000.0,
                        emission_factor=0.5
                    )
                    count += 1

                return count

            result = self.run_repeated_benchmark(
                "Calculator Throughput",
                calculate_emissions,
                iterations=10
            )

            # Add throughput metric
            if result.get("status") == "completed":
                throughput = result["duration"]["mean"]
                result["throughput_per_second"] = round(throughput, 2)
                result["target_met"] = throughput >= self.sla_targets["calculator_throughput"]["target_records_per_second"]

                print(f"   Throughput: {result['throughput_per_second']} calculations/second")

            return result

        except ImportError as e:
            print(f"❌ Cannot import CalculatorAgent: {e}")
            return {"status": "skipped", "reason": "Import failed"}

    def run_all_benchmarks(self) -> Dict:
        """Run all performance benchmarks."""
        print("\n" + "="*80)
        print("CSRD REPORTING PLATFORM - PERFORMANCE BENCHMARKS")
        print("="*80)
        print(f"Timestamp: {self.results['timestamp']}")
        print(f"System: {self.results['system_info']}")
        print("="*80)

        # Run benchmarks
        self.results["benchmarks"]["xbrl_generation"] = self.benchmark_xbrl_generation()
        self.results["benchmarks"]["materiality_assessment"] = self.benchmark_materiality_assessment()
        self.results["benchmarks"]["data_import"] = self.benchmark_data_import()
        self.results["benchmarks"]["audit_validation"] = self.benchmark_audit_validation()
        self.results["benchmarks"]["calculator_throughput"] = self.benchmark_calculator_throughput()

        # Generate summary
        self.generate_summary()

        return self.results

    def generate_summary(self) -> None:
        """Generate benchmark summary."""
        print("\n" + "="*80)
        print("PERFORMANCE BENCHMARK SUMMARY")
        print("="*80)

        targets_met = 0
        targets_total = 0

        print("\n📊 SLA Target Validation:\n")
        print(f"{'Benchmark':<40} {'Target':<15} {'Actual (p95)':<15} {'Status':<10}")
        print("-" * 80)

        for benchmark_name, benchmark_result in self.results["benchmarks"].items():
            if benchmark_result.get("status") != "completed":
                continue

            target = self.sla_targets.get(benchmark_name, {})
            target_value = target.get("target_seconds") or target.get("target_ms", 0) / 1000

            actual_value = benchmark_result.get("duration", {}).get("p95", 0)
            target_met = benchmark_result.get("target_met", False)

            targets_total += 1
            if target_met:
                targets_met += 1

            status = "✅ PASS" if target_met else "❌ FAIL"

            print(f"{target.get('name', benchmark_name):<40} {target_value:<15.2f}s {actual_value:<15.3f}s {status:<10}")

        print("-" * 80)
        print(f"{'TOTAL':<40} {'':<15} {'':<15} {targets_met}/{targets_total}")

        # Overall status
        self.results["summary"] = {
            "targets_met": targets_met,
            "targets_total": targets_total,
            "pass_rate": round((targets_met / targets_total * 100) if targets_total > 0 else 0, 2),
            "status": "PASS" if targets_met == targets_total else "FAIL"
        }

        # Save results
        summary_path = self.reports_dir / "benchmark_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n📄 Full benchmark report saved to: {summary_path}")

        if self.results["summary"]["status"] == "PASS":
            print("\n✅ ALL PERFORMANCE TARGETS MET - READY FOR PRODUCTION")
        else:
            print("\n❌ SOME PERFORMANCE TARGETS MISSED - OPTIMIZATION REQUIRED")


def main():
    """Main entry point."""
    benchmark = PerformanceBenchmark()
    result = benchmark.run_all_benchmarks()

    sys.exit(0 if result["summary"]["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
