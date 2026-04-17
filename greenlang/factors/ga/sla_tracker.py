# -*- coding: utf-8 -*-
"""
SLA tracker and compliance monitor (F102).

Tracks uptime, latency, and error rate against defined SLAs.
Provides real-time SLA status, violation alerts, and compliance reports.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SLAMetric(str, Enum):
    UPTIME = "uptime"
    LATENCY_P95 = "latency_p95"
    LATENCY_P99 = "latency_p99"
    ERROR_RATE = "error_rate"
    THROUGHPUT = "throughput"


class SLAStatus(str, Enum):
    COMPLIANT = "compliant"
    AT_RISK = "at_risk"
    VIOLATED = "violated"
    UNKNOWN = "unknown"


@dataclass
class SLADefinition:
    """Definition of a single SLA target."""

    name: str
    metric: SLAMetric
    target_value: float
    warning_threshold: float  # "at risk" if breached
    unit: str = ""
    description: str = ""

    def evaluate(self, actual: float) -> SLAStatus:
        """Evaluate current metric value against SLA target."""
        if self.metric in (SLAMetric.UPTIME, SLAMetric.THROUGHPUT):
            # Higher is better
            if actual >= self.target_value:
                return SLAStatus.COMPLIANT
            elif actual >= self.warning_threshold:
                return SLAStatus.AT_RISK
            return SLAStatus.VIOLATED
        else:
            # Lower is better (latency, error rate)
            if actual <= self.target_value:
                return SLAStatus.COMPLIANT
            elif actual <= self.warning_threshold:
                return SLAStatus.AT_RISK
            return SLAStatus.VIOLATED


@dataclass
class SLAMeasurement:
    """A point-in-time measurement for an SLA metric."""

    metric: SLAMetric
    value: float
    timestamp: str
    window_minutes: int = 5


@dataclass
class SLAReport:
    """Comprehensive SLA compliance report."""

    period: str
    generated_at: str
    overall_status: SLAStatus = SLAStatus.UNKNOWN
    sla_results: List[Dict[str, Any]] = field(default_factory=list)
    violations: List[Dict[str, Any]] = field(default_factory=list)
    uptime_pct: float = 0.0
    avg_latency_ms: float = 0.0
    error_rate_pct: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "period": self.period,
            "generated_at": self.generated_at,
            "overall_status": self.overall_status.value,
            "uptime_pct": round(self.uptime_pct, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "error_rate_pct": round(self.error_rate_pct, 4),
            "sla_results": self.sla_results,
            "violations": self.violations,
        }


# Default SLA definitions for Factors catalog
DEFAULT_SLAS: List[SLADefinition] = [
    SLADefinition(
        name="API Uptime",
        metric=SLAMetric.UPTIME,
        target_value=99.9,
        warning_threshold=99.5,
        unit="%",
        description="API availability over rolling 30 days",
    ),
    SLADefinition(
        name="Search Latency p95",
        metric=SLAMetric.LATENCY_P95,
        target_value=500.0,
        warning_threshold=1000.0,
        unit="ms",
        description="95th percentile search response time",
    ),
    SLADefinition(
        name="Search Latency p99",
        metric=SLAMetric.LATENCY_P99,
        target_value=2000.0,
        warning_threshold=5000.0,
        unit="ms",
        description="99th percentile search response time",
    ),
    SLADefinition(
        name="Error Rate",
        metric=SLAMetric.ERROR_RATE,
        target_value=0.1,
        warning_threshold=1.0,
        unit="%",
        description="5xx error rate over rolling 24 hours",
    ),
    SLADefinition(
        name="Throughput",
        metric=SLAMetric.THROUGHPUT,
        target_value=100.0,
        warning_threshold=50.0,
        unit="req/s",
        description="Minimum sustained throughput capacity",
    ),
]


class SLATracker:
    """
    Tracks SLA compliance for the Factors catalog.

    Provides:
      - Real-time SLA evaluation
      - Historical measurement recording
      - Violation detection and alerting
      - Monthly/weekly compliance reports
    """

    def __init__(self, slas: Optional[List[SLADefinition]] = None) -> None:
        self._slas = {s.name: s for s in (slas or DEFAULT_SLAS)}
        self._measurements: Dict[str, List[SLAMeasurement]] = defaultdict(list)
        self._violations: List[Dict[str, Any]] = []
        self._max_measurements = 50000

    def record(self, metric: SLAMetric, value: float, window_minutes: int = 5) -> SLAStatus:
        """Record a measurement and evaluate against SLAs."""
        measurement = SLAMeasurement(
            metric=metric,
            value=value,
            timestamp=datetime.now(timezone.utc).isoformat(),
            window_minutes=window_minutes,
        )
        self._measurements[metric.value].append(measurement)

        # Bound memory
        if len(self._measurements[metric.value]) > self._max_measurements:
            self._measurements[metric.value] = self._measurements[metric.value][-self._max_measurements:]

        # Evaluate all SLAs matching this metric
        worst = SLAStatus.COMPLIANT
        for sla in self._slas.values():
            if sla.metric == metric:
                status = sla.evaluate(value)
                if status == SLAStatus.VIOLATED:
                    self._violations.append({
                        "sla_name": sla.name,
                        "metric": metric.value,
                        "target": sla.target_value,
                        "actual": value,
                        "timestamp": measurement.timestamp,
                    })
                    logger.warning(
                        "SLA violation: %s actual=%.2f target=%.2f",
                        sla.name, value, sla.target_value,
                    )
                    worst = SLAStatus.VIOLATED
                elif status == SLAStatus.AT_RISK and worst != SLAStatus.VIOLATED:
                    worst = SLAStatus.AT_RISK

        return worst

    def current_status(self) -> Dict[str, Dict[str, Any]]:
        """Get current status for all SLAs based on latest measurements."""
        result: Dict[str, Dict[str, Any]] = {}
        for sla_name, sla in self._slas.items():
            measurements = self._measurements.get(sla.metric.value, [])
            if measurements:
                latest = measurements[-1]
                status = sla.evaluate(latest.value)
                result[sla_name] = {
                    "status": status.value,
                    "current_value": latest.value,
                    "target": sla.target_value,
                    "unit": sla.unit,
                    "last_measured": latest.timestamp,
                }
            else:
                result[sla_name] = {
                    "status": SLAStatus.UNKNOWN.value,
                    "current_value": None,
                    "target": sla.target_value,
                    "unit": sla.unit,
                }
        return result

    def generate_report(self, period: str = "30d") -> SLAReport:
        """Generate a compliance report for the specified period."""
        report = SLAReport(
            period=period,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

        sla_results = []
        has_violation = False
        has_risk = False

        for sla_name, sla in self._slas.items():
            measurements = self._measurements.get(sla.metric.value, [])
            if not measurements:
                sla_results.append({
                    "name": sla_name,
                    "status": "unknown",
                    "target": sla.target_value,
                    "unit": sla.unit,
                })
                continue

            values = [m.value for m in measurements]
            avg = sum(values) / len(values)
            status = sla.evaluate(avg)

            if status == SLAStatus.VIOLATED:
                has_violation = True
            elif status == SLAStatus.AT_RISK:
                has_risk = True

            sla_results.append({
                "name": sla_name,
                "status": status.value,
                "target": sla.target_value,
                "actual_avg": round(avg, 4),
                "measurements": len(measurements),
                "unit": sla.unit,
            })

            # Populate report-level metrics
            if sla.metric == SLAMetric.UPTIME:
                report.uptime_pct = avg
            elif sla.metric == SLAMetric.LATENCY_P95:
                report.avg_latency_ms = avg
            elif sla.metric == SLAMetric.ERROR_RATE:
                report.error_rate_pct = avg

        report.sla_results = sla_results
        report.violations = self._violations[-50:]  # Last 50 violations

        if has_violation:
            report.overall_status = SLAStatus.VIOLATED
        elif has_risk:
            report.overall_status = SLAStatus.AT_RISK
        else:
            report.overall_status = SLAStatus.COMPLIANT

        return report

    @property
    def violations(self) -> List[Dict[str, Any]]:
        return list(self._violations)

    @property
    def violation_count(self) -> int:
        return len(self._violations)
