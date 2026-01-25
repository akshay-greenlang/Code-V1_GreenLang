# -*- coding: utf-8 -*-
"""
Performance Benchmark Reporting
GL-VCCI Scope 3 Platform

Utilities for generating performance reports, tracking metrics,
and comparing connector performance.

Version: 1.0.0
Phase: 4 (Weeks 24-26)
Date: 2025-11-06
"""

import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from greenlang.determinism import DeterministicClock


@dataclass
class BenchmarkResult:
    """Single benchmark test result."""

    connector_name: str  # SAP, Oracle, Workday
    test_name: str
    timestamp: str
    duration_seconds: float
    records_extracted: int
    throughput_per_hour: float
    throughput_per_second: float
    api_calls: int
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    max_memory_mb: float
    errors: int
    success_rate: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class BenchmarkReporter:
    """
    Performance benchmark reporter.

    Generates HTML and JSON reports for performance test results.
    """

    def __init__(self, output_dir: str = "benchmark_reports"):
        """
        Initialize reporter.

        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[BenchmarkResult] = []

    def add_result(self, result: BenchmarkResult):
        """
        Add benchmark result.

        Args:
            result: BenchmarkResult to add
        """
        self.results.append(result)

    def generate_json_report(self, filename: Optional[str] = None) -> str:
        """
        Generate JSON report.

        Args:
            filename: Output filename (auto-generated if None)

        Returns:
            Path to generated report
        """
        if filename is None:
            timestamp = DeterministicClock.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_report_{timestamp}.json"

        filepath = self.output_dir / filename

        report_data = {
            "generated_at": DeterministicClock.now().isoformat(),
            "total_tests": len(self.results),
            "results": [r.to_dict() for r in self.results],
            "summary": self._generate_summary()
        }

        with open(filepath, 'w') as f:
            json.dump(report_data, f, indent=2)

        return str(filepath)

    def generate_html_report(self, filename: Optional[str] = None) -> str:
        """
        Generate HTML report.

        Args:
            filename: Output filename (auto-generated if None)

        Returns:
            Path to generated report
        """
        if filename is None:
            timestamp = DeterministicClock.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_report_{timestamp}.html"

        filepath = self.output_dir / filename

        html_content = self._generate_html()

        with open(filepath, 'w') as f:
            f.write(html_content)

        return str(filepath)

    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not self.results:
            return {}

        by_connector = {}
        for result in self.results:
            conn = result.connector_name
            if conn not in by_connector:
                by_connector[conn] = []
            by_connector[conn].append(result)

        summary = {
            "connectors": {}
        }

        for connector, results in by_connector.items():
            summary["connectors"][connector] = {
                "tests_run": len(results),
                "avg_throughput_per_hour": sum(r.throughput_per_hour for r in results) / len(results),
                "max_throughput_per_hour": max(r.throughput_per_hour for r in results),
                "avg_latency_ms": sum(r.avg_latency_ms for r in results) / len(results),
                "avg_p95_latency_ms": sum(r.p95_latency_ms for r in results) / len(results),
                "total_records": sum(r.records_extracted for r in results),
                "total_errors": sum(r.errors for r in results),
                "avg_success_rate": sum(r.success_rate for r in results) / len(results)
            }

        return summary

    def _generate_html(self) -> str:
        """Generate HTML report content."""
        summary = self._generate_summary()

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>ERP Connector Performance Benchmark Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .summary-card {{
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }}
        .summary-card h3 {{
            margin: 0 0 10px 0;
            color: #2c3e50;
        }}
        .metric {{
            display: flex;
            justify-content: space-between;
            margin: 8px 0;
            font-size: 14px;
        }}
        .metric-label {{
            color: #7f8c8d;
        }}
        .metric-value {{
            font-weight: bold;
            color: #2c3e50;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th {{
            background-color: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
        }}
        td {{
            padding: 10px;
            border-bottom: 1px solid #ecf0f1;
        }}
        tr:hover {{
            background-color: #f8f9fa;
        }}
        .success {{
            color: #27ae60;
            font-weight: bold;
        }}
        .warning {{
            color: #f39c12;
            font-weight: bold;
        }}
        .error {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .footer {{
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ecf0f1;
            text-align: center;
            color: #7f8c8d;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>ERP Connector Performance Benchmark Report</h1>
        <p>Generated: {DeterministicClock.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        <p>Total Tests: {len(self.results)}</p>

        <h2>Summary by Connector</h2>
        <div class="summary">
"""

        # Add summary cards
        for connector, stats in summary.get("connectors", {}).items():
            html += f"""
            <div class="summary-card">
                <h3>{connector}</h3>
                <div class="metric">
                    <span class="metric-label">Tests Run:</span>
                    <span class="metric-value">{stats['tests_run']}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Avg Throughput:</span>
                    <span class="metric-value">{stats['avg_throughput_per_hour']:,.0f} rec/hr</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Max Throughput:</span>
                    <span class="metric-value">{stats['max_throughput_per_hour']:,.0f} rec/hr</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Avg Latency:</span>
                    <span class="metric-value">{stats['avg_latency_ms']:.2f} ms</span>
                </div>
                <div class="metric">
                    <span class="metric-label">P95 Latency:</span>
                    <span class="metric-value">{stats['avg_p95_latency_ms']:.2f} ms</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Total Records:</span>
                    <span class="metric-value">{stats['total_records']:,}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Success Rate:</span>
                    <span class="metric-value">{stats['avg_success_rate']*100:.2f}%</span>
                </div>
            </div>
"""

        html += """
        </div>

        <h2>Detailed Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Connector</th>
                    <th>Test</th>
                    <th>Timestamp</th>
                    <th>Records</th>
                    <th>Throughput (rec/hr)</th>
                    <th>Avg Latency (ms)</th>
                    <th>P95 Latency (ms)</th>
                    <th>Success Rate</th>
                </tr>
            </thead>
            <tbody>
"""

        # Add detailed results
        for result in sorted(self.results, key=lambda r: r.timestamp, reverse=True):
            success_class = "success" if result.success_rate >= 0.95 else "warning" if result.success_rate >= 0.80 else "error"

            html += f"""
                <tr>
                    <td>{result.connector_name}</td>
                    <td>{result.test_name}</td>
                    <td>{result.timestamp}</td>
                    <td>{result.records_extracted:,}</td>
                    <td>{result.throughput_per_hour:,.0f}</td>
                    <td>{result.avg_latency_ms:.2f}</td>
                    <td>{result.p95_latency_ms:.2f}</td>
                    <td class="{success_class}">{result.success_rate*100:.2f}%</td>
                </tr>
"""

        html += """
            </tbody>
        </table>

        <div class="footer">
            <p>GL-VCCI Scope 3 Platform - ERP Connector Performance Benchmarks</p>
            <p>Phase 4 (Weeks 24-26) - Integration Testing Framework</p>
        </div>
    </div>
</body>
</html>
"""

        return html

    def print_summary(self):
        """Print summary to console."""
        summary = self._generate_summary()

        print("\n" + "="*60)
        print("Performance Benchmark Summary")
        print("="*60)
        print(f"Total Tests: {len(self.results)}")
        print(f"Generated: {DeterministicClock.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        for connector, stats in summary.get("connectors", {}).items():
            print(f"\n{connector}:")
            print(f"  Tests Run: {stats['tests_run']}")
            print(f"  Avg Throughput: {stats['avg_throughput_per_hour']:,.0f} records/hour")
            print(f"  Max Throughput: {stats['max_throughput_per_hour']:,.0f} records/hour")
            print(f"  Avg Latency: {stats['avg_latency_ms']:.2f} ms")
            print(f"  P95 Latency: {stats['avg_p95_latency_ms']:.2f} ms")
            print(f"  Total Records: {stats['total_records']:,}")
            print(f"  Total Errors: {stats['total_errors']}")
            print(f"  Avg Success Rate: {stats['avg_success_rate']*100:.2f}%")

        print("\n" + "="*60 + "\n")


def create_benchmark_result_from_metrics(
    connector_name: str,
    test_name: str,
    metrics
) -> BenchmarkResult:
    """
    Create BenchmarkResult from PerformanceMetrics.

    Args:
        connector_name: Name of connector (SAP, Oracle, Workday)
        test_name: Name of test
        metrics: PerformanceMetrics instance

    Returns:
        BenchmarkResult instance
    """
    success_rate = 1.0 - (metrics.errors / max(metrics.requests_made, 1))

    return BenchmarkResult(
        connector_name=connector_name,
        test_name=test_name,
        timestamp=DeterministicClock.now().isoformat(),
        duration_seconds=metrics.duration_seconds,
        records_extracted=metrics.total_records,
        throughput_per_hour=metrics.throughput_per_hour,
        throughput_per_second=metrics.throughput_per_second,
        api_calls=metrics.requests_made,
        avg_latency_ms=metrics.avg_latency_ms,
        p95_latency_ms=metrics.p95_latency_ms,
        p99_latency_ms=metrics.p99_latency_ms,
        max_memory_mb=metrics.max_memory_mb,
        errors=metrics.errors,
        success_rate=success_rate
    )


if __name__ == "__main__":
    # Example usage
    reporter = BenchmarkReporter()

    # Add sample results
    reporter.add_result(BenchmarkResult(
        connector_name="SAP",
        test_name="100K Throughput Test",
        timestamp=DeterministicClock.now().isoformat(),
        duration_seconds=3200,
        records_extracted=100000,
        throughput_per_hour=112500,
        throughput_per_second=31.25,
        api_calls=100,
        avg_latency_ms=450.5,
        p95_latency_ms=850.2,
        p99_latency_ms=1200.5,
        max_memory_mb=256.8,
        errors=2,
        success_rate=0.98
    ))

    reporter.add_result(BenchmarkResult(
        connector_name="Oracle",
        test_name="100K Throughput Test",
        timestamp=DeterministicClock.now().isoformat(),
        duration_seconds=3400,
        records_extracted=100000,
        throughput_per_hour=105882,
        throughput_per_second=29.41,
        api_calls=100,
        avg_latency_ms=520.3,
        p95_latency_ms=920.5,
        p99_latency_ms=1350.2,
        max_memory_mb=298.5,
        errors=3,
        success_rate=0.97
    ))

    # Generate reports
    reporter.print_summary()
    json_path = reporter.generate_json_report()
    html_path = reporter.generate_html_report()

    print(f"JSON report: {json_path}")
    print(f"HTML report: {html_path}")
