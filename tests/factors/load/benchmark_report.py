# -*- coding: utf-8 -*-
"""
Benchmark report generator for GreenLang Factors API load tests.

Parses Locust CSV or k6 JSON output and generates a markdown report
with per-endpoint latency percentiles, error rates, and SLA compliance.

Usage:
    # From Locust CSV output:
    python tests/factors/load/benchmark_report.py --source locust --input results_stats.csv

    # From k6 JSON output:
    python tests/factors/load/benchmark_report.py --source k6 --input results.json

    # Output defaults to tests/factors/load/BENCHMARK_REPORT.md
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── SLA thresholds (mirrored from conftest.py) ─────────────────────

SLA_P95_MS = 50.0
SLA_P99_MS = 100.0
SLA_ERROR_RATE_PCT = 1.0
SLA_TARGET_RPS = 1000


@dataclass
class EndpointMetrics:
    """Aggregated metrics for a single endpoint."""

    endpoint: str
    request_count: int = 0
    error_count: int = 0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    avg_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    rps: float = 0.0

    @property
    def error_rate_pct(self) -> float:
        """Calculate error rate as a percentage."""
        if self.request_count == 0:
            return 0.0
        return (self.error_count / self.request_count) * 100.0


@dataclass
class BenchmarkReport:
    """Complete benchmark report from a load test run."""

    source: str  # "locust" or "k6"
    generated_at: str = ""
    duration_seconds: float = 0.0
    total_requests: int = 0
    total_errors: int = 0
    overall_rps: float = 0.0
    endpoints: List[EndpointMetrics] = field(default_factory=list)
    sla_violations: List[str] = field(default_factory=list)

    @property
    def overall_error_rate_pct(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.total_errors / self.total_requests) * 100.0

    @property
    def sla_passed(self) -> bool:
        return len(self.sla_violations) == 0


# ── Locust CSV parser ──────────────────────────────────────────────


def parse_locust_csv(csv_path: Path) -> BenchmarkReport:
    """
    Parse Locust stats CSV into a BenchmarkReport.

    Locust generates a CSV with columns like:
      Type, Name, Request Count, Failure Count, Median Response Time,
      Average Response Time, Min Response Time, Max Response Time,
      Average Content Size, Requests/s, Failures/s,
      50%, 66%, 75%, 80%, 90%, 95%, 98%, 99%, 99.9%, 99.99%, 100%
    """
    report = BenchmarkReport(
        source="locust",
        generated_at=datetime.now(timezone.utc).isoformat(),
    )

    rows: List[Dict[str, str]] = []
    with open(csv_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)

    for row in rows:
        name = row.get("Name", "").strip()
        req_type = row.get("Type", "").strip()

        # Skip the "Aggregated" summary row for individual parsing
        if name == "Aggregated":
            report.total_requests = _safe_int(row.get("Request Count", "0"))
            report.total_errors = _safe_int(row.get("Failure Count", "0"))
            report.overall_rps = _safe_float(row.get("Requests/s", "0"))
            continue

        if not name:
            continue

        endpoint = EndpointMetrics(
            endpoint="%s %s" % (req_type, name) if req_type else name,
            request_count=_safe_int(row.get("Request Count", "0")),
            error_count=_safe_int(row.get("Failure Count", "0")),
            p50_ms=_safe_float(row.get("50%", row.get("Median Response Time", "0"))),
            p95_ms=_safe_float(row.get("95%", "0")),
            p99_ms=_safe_float(row.get("99%", "0")),
            avg_ms=_safe_float(row.get("Average Response Time", "0")),
            min_ms=_safe_float(row.get("Min Response Time", "0")),
            max_ms=_safe_float(row.get("Max Response Time", "0")),
            rps=_safe_float(row.get("Requests/s", "0")),
        )
        report.endpoints.append(endpoint)

    _check_sla_violations(report)
    return report


# ── k6 JSON parser ─────────────────────────────────────────────────


def parse_k6_json(json_path: Path) -> BenchmarkReport:
    """
    Parse k6 JSON summary output into a BenchmarkReport.

    k6 outputs line-delimited JSON. Each line has a "type" field:
      - "Point" for individual data points
      - "Metric" for metric summaries

    Alternatively, k6 --summary-export produces a summary JSON.
    """
    report = BenchmarkReport(
        source="k6",
        generated_at=datetime.now(timezone.utc).isoformat(),
    )

    raw_text = json_path.read_text(encoding="utf-8").strip()

    # Try parsing as k6 summary export (single JSON object)
    try:
        summary = json.loads(raw_text)
        if "metrics" in summary:
            return _parse_k6_summary(summary, report)
    except json.JSONDecodeError:
        pass

    # Parse as line-delimited JSON (k6 --out json)
    endpoint_durations: Dict[str, List[float]] = {}
    endpoint_errors: Dict[str, int] = {}

    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        entry_type = entry.get("type")
        if entry_type != "Point":
            continue

        metric = entry.get("metric", "")
        data = entry.get("data", {})
        tags = data.get("tags", {})
        value = data.get("value", 0)

        endpoint_name = tags.get("endpoint", tags.get("name", ""))
        if not endpoint_name:
            continue

        if metric == "http_req_duration":
            endpoint_durations.setdefault(endpoint_name, []).append(value)
        elif metric == "http_req_failed":
            if value:
                endpoint_errors[endpoint_name] = (
                    endpoint_errors.get(endpoint_name, 0) + 1
                )

    # Build endpoint metrics from collected data
    for endpoint_name, durations in sorted(endpoint_durations.items()):
        durations_sorted = sorted(durations)
        count = len(durations_sorted)
        errors = endpoint_errors.get(endpoint_name, 0)

        endpoint = EndpointMetrics(
            endpoint=endpoint_name,
            request_count=count,
            error_count=errors,
            p50_ms=_percentile(durations_sorted, 50),
            p95_ms=_percentile(durations_sorted, 95),
            p99_ms=_percentile(durations_sorted, 99),
            avg_ms=statistics.mean(durations_sorted) if durations_sorted else 0.0,
            min_ms=durations_sorted[0] if durations_sorted else 0.0,
            max_ms=durations_sorted[-1] if durations_sorted else 0.0,
            rps=count / max(report.duration_seconds, 1.0),
        )
        report.endpoints.append(endpoint)
        report.total_requests += count
        report.total_errors += errors

    if report.duration_seconds > 0:
        report.overall_rps = report.total_requests / report.duration_seconds

    _check_sla_violations(report)
    return report


def _parse_k6_summary(summary: Dict[str, Any], report: BenchmarkReport) -> BenchmarkReport:
    """Parse k6 --summary-export JSON format."""
    metrics = summary.get("metrics", {})

    # Extract http_req_duration
    http_dur = metrics.get("http_req_duration", {})
    if "values" in http_dur:
        vals = http_dur["values"]
        report.endpoints.append(
            EndpointMetrics(
                endpoint="(all requests)",
                request_count=int(vals.get("count", 0)),
                p50_ms=vals.get("med", 0.0),
                p95_ms=vals.get("p(95)", 0.0),
                p99_ms=vals.get("p(99)", 0.0),
                avg_ms=vals.get("avg", 0.0),
                min_ms=vals.get("min", 0.0),
                max_ms=vals.get("max", 0.0),
            )
        )
        report.total_requests = int(vals.get("count", 0))

    # Extract per-scenario custom metrics
    for metric_name in (
        "factors_search_latency",
        "factors_list_latency",
        "factors_get_latency",
        "factors_match_latency",
        "factors_edition_latency",
        "factors_export_latency",
    ):
        metric_data = metrics.get(metric_name, {})
        if "values" in metric_data:
            vals = metric_data["values"]
            endpoint_label = metric_name.replace("factors_", "").replace("_latency", "")
            report.endpoints.append(
                EndpointMetrics(
                    endpoint=endpoint_label,
                    request_count=int(vals.get("count", 0)),
                    p50_ms=vals.get("med", 0.0),
                    p95_ms=vals.get("p(95)", 0.0),
                    p99_ms=vals.get("p(99)", 0.0),
                    avg_ms=vals.get("avg", 0.0),
                    min_ms=vals.get("min", 0.0),
                    max_ms=vals.get("max", 0.0),
                )
            )

    # Error rate
    error_metric = metrics.get("factors_error_rate", {})
    if "values" in error_metric:
        rate = error_metric["values"].get("rate", 0.0)
        report.total_errors = int(rate * report.total_requests)

    _check_sla_violations(report)
    return report


# ── SLA check ──────────────────────────────────────────────────────


def _check_sla_violations(report: BenchmarkReport) -> None:
    """Check all endpoints against SLA thresholds and populate violations."""
    report.sla_violations = []

    for ep in report.endpoints:
        if ep.p95_ms > SLA_P95_MS:
            report.sla_violations.append(
                "[FAIL] %s: p95=%.1fms exceeds SLA threshold of %.1fms"
                % (ep.endpoint, ep.p95_ms, SLA_P95_MS)
            )
        if ep.p99_ms > SLA_P99_MS:
            report.sla_violations.append(
                "[FAIL] %s: p99=%.1fms exceeds SLA threshold of %.1fms"
                % (ep.endpoint, ep.p99_ms, SLA_P99_MS)
            )
        if ep.error_rate_pct > SLA_ERROR_RATE_PCT:
            report.sla_violations.append(
                "[FAIL] %s: error_rate=%.2f%% exceeds SLA threshold of %.2f%%"
                % (ep.endpoint, ep.error_rate_pct, SLA_ERROR_RATE_PCT)
            )

    if report.overall_rps > 0 and report.overall_rps < SLA_TARGET_RPS:
        report.sla_violations.append(
            "[WARN] Overall RPS %.1f below target %d req/s"
            % (report.overall_rps, SLA_TARGET_RPS)
        )


# ── Markdown generation ───────────────────────────────────────────


def generate_markdown(report: BenchmarkReport) -> str:
    """Generate a markdown benchmark report from parsed results."""
    lines: List[str] = []

    lines.append("# Factors API Load Test Benchmark Report")
    lines.append("")
    lines.append("**Generated:** %s" % report.generated_at)
    lines.append("**Source:** %s" % report.source)
    lines.append("")

    # Overall summary
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append("| Total Requests | %d |" % report.total_requests)
    lines.append("| Total Errors | %d |" % report.total_errors)
    lines.append("| Overall Error Rate | %.2f%% |" % report.overall_error_rate_pct)
    lines.append("| Overall RPS | %.1f |" % report.overall_rps)
    lines.append(
        "| SLA Status | %s |"
        % ("PASS" if report.sla_passed else "FAIL")
    )
    lines.append("")

    # Per-endpoint table
    lines.append("## Per-Endpoint Latency")
    lines.append("")
    lines.append(
        "| Endpoint | Requests | p50 (ms) | p95 (ms) | p99 (ms) | Avg (ms) | Error Rate | RPS | SLA |"
    )
    lines.append(
        "|----------|----------|----------|----------|----------|----------|------------|-----|-----|"
    )

    for ep in report.endpoints:
        sla_status = "PASS"
        if ep.p95_ms > SLA_P95_MS or ep.error_rate_pct > SLA_ERROR_RATE_PCT:
            sla_status = "FAIL"

        lines.append(
            "| %s | %d | %.1f | %.1f | %.1f | %.1f | %.2f%% | %.1f | %s |"
            % (
                ep.endpoint,
                ep.request_count,
                ep.p50_ms,
                ep.p95_ms,
                ep.p99_ms,
                ep.avg_ms,
                ep.error_rate_pct,
                ep.rps,
                sla_status,
            )
        )

    lines.append("")

    # SLA violations
    if report.sla_violations:
        lines.append("## SLA Violations")
        lines.append("")
        for v in report.sla_violations:
            lines.append("- %s" % v)
        lines.append("")
    else:
        lines.append("## SLA Compliance")
        lines.append("")
        lines.append("All endpoints meet SLA thresholds.")
        lines.append("")
        lines.append("- p95 < %.0fms" % SLA_P95_MS)
        lines.append("- p99 < %.0fms" % SLA_P99_MS)
        lines.append("- Error rate < %.1f%%" % SLA_ERROR_RATE_PCT)
        lines.append("- Target RPS: %d" % SLA_TARGET_RPS)
        lines.append("")

    # Thresholds reference
    lines.append("## SLA Thresholds Reference")
    lines.append("")
    lines.append("| Threshold | Value |")
    lines.append("|-----------|-------|")
    lines.append("| p95 latency | < %.0f ms |" % SLA_P95_MS)
    lines.append("| p99 latency | < %.0f ms |" % SLA_P99_MS)
    lines.append("| Error rate | < %.1f%% |" % SLA_ERROR_RATE_PCT)
    lines.append("| Target throughput | %d req/s |" % SLA_TARGET_RPS)
    lines.append("")

    return "\n".join(lines)


# ── Utility functions ──────────────────────────────────────────────


def _safe_int(value: str) -> int:
    """Safely parse an integer from a string."""
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return 0


def _safe_float(value: str) -> float:
    """Safely parse a float from a string."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def _percentile(sorted_values: List[float], pct: float) -> float:
    """Calculate percentile from a pre-sorted list."""
    if not sorted_values:
        return 0.0
    idx = int(len(sorted_values) * pct / 100.0)
    idx = min(idx, len(sorted_values) - 1)
    return sorted_values[idx]


# ── CLI entry point ────────────────────────────────────────────────


def main() -> None:
    """CLI entry point for benchmark report generation."""
    parser = argparse.ArgumentParser(
        description="Generate benchmark report from load test results"
    )
    parser.add_argument(
        "--source",
        choices=["locust", "k6"],
        required=True,
        help="Load test tool that produced the results",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to results file (CSV for Locust, JSON for k6)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "BENCHMARK_REPORT.md",
        help="Output path for the markdown report",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print("Error: input file not found: %s" % input_path, file=sys.stderr)
        sys.exit(1)

    if args.source == "locust":
        report = parse_locust_csv(input_path)
    else:
        report = parse_k6_json(input_path)

    markdown = generate_markdown(report)

    output_path = Path(args.output)
    output_path.write_text(markdown, encoding="utf-8")
    print("Benchmark report written to: %s" % output_path)

    if not report.sla_passed:
        print("WARNING: SLA violations detected!", file=sys.stderr)
        for v in report.sla_violations:
            print("  %s" % v, file=sys.stderr)
        sys.exit(2)

    print("All SLA thresholds met.")


if __name__ == "__main__":
    main()
