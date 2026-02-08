# -*- coding: utf-8 -*-
"""
Load and Performance Tests for Observability Agent Service (AGENT-FOUND-010)

Tests high-volume metric recording, concurrent span creation, log ingestion
throughput, alert evaluation under load, SLO calculation performance,
Prometheus export at scale, and memory usage under load.

All tests use in-memory engines. Performance targets are for the pure
Python layer (no network/database I/O).

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Lightweight inline engines (reused from integration tests)
# ---------------------------------------------------------------------------


class LoadMetricsCollector:
    """High-throughput metrics collector for load testing."""

    def __init__(self):
        self._series: Dict[str, float] = {}
        self._definitions: Dict[str, str] = {}
        self._total: int = 0

    def register_metric(self, name, metric_type):
        self._definitions[name] = metric_type

    def record(self, name, value, labels=None):
        key = f"{name}:{json.dumps(labels or {}, sort_keys=True)}"
        if name in self._definitions and self._definitions[name] == "counter":
            self._series[key] = self._series.get(key, 0.0) + value
        else:
            self._series[key] = value
        self._total += 1
        return {"metric_name": name, "value": value}

    def export_prometheus(self):
        lines = []
        for key, value in sorted(self._series.items()):
            name = key.split(":")[0]
            lines.append(f"{name} {value}")
        return "\n".join(lines)


class LoadTraceManager:
    """High-throughput trace manager for load testing."""

    def __init__(self):
        self._spans: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def start_span(self, name, trace_id=None):
        tid = trace_id or str(uuid.uuid4())
        sid = str(uuid.uuid4())
        span = {"trace_id": tid, "span_id": sid, "name": name, "is_active": True}
        with self._lock:
            self._spans[f"{tid}:{sid}"] = span
        return span

    def end_span(self, trace_id, span_id, status="OK"):
        key = f"{trace_id}:{span_id}"
        with self._lock:
            span = self._spans.get(key)
            if span:
                span["status"] = status
                span["is_active"] = False
        return span


class LoadLogAggregator:
    """High-throughput log aggregator for load testing."""

    def __init__(self, max_buffer=100000):
        self._buffer: List[Dict[str, Any]] = []
        self._max = max_buffer
        self._total: int = 0

    def ingest(self, message, level="info"):
        if len(self._buffer) >= self._max:
            self._buffer = self._buffer[self._max // 2:]
        self._buffer.append({"message": message, "level": level})
        self._total += 1


class LoadAlertEvaluator:
    """High-throughput alert evaluator for load testing."""

    def __init__(self):
        self._rules: List[Dict[str, Any]] = []

    def add_rule(self, name, metric_name, condition, threshold):
        self._rules.append({
            "name": name, "metric_name": metric_name,
            "condition": condition, "threshold": threshold,
        })

    def evaluate_all(self, metric_values):
        fired = []
        for rule in self._rules:
            val = metric_values.get(rule["metric_name"])
            if val is None:
                continue
            if rule["condition"] == "gt" and val > rule["threshold"]:
                fired.append(rule["name"])
        return fired


class LoadSLOTracker:
    """High-throughput SLO tracker for load testing."""

    def __init__(self):
        self._slos: Dict[str, Dict[str, Any]] = {}
        self._observations: Dict[str, List[bool]] = {}

    def create_slo(self, name, target=0.999):
        slo_id = str(uuid.uuid4())
        self._slos[slo_id] = {"slo_id": slo_id, "name": name, "target": target}
        self._observations[slo_id] = []
        return slo_id

    def record_observation(self, slo_id, is_good=True):
        self._observations[slo_id].append(is_good)

    def calculate_compliance(self, slo_id):
        obs = self._observations.get(slo_id, [])
        total = len(obs)
        good = sum(1 for o in obs if o)
        return good / total if total > 0 else 1.0

    def evaluate_all(self):
        results = {}
        for slo_id in self._slos:
            results[slo_id] = self.calculate_compliance(slo_id)
        return results


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def metrics():
    return LoadMetricsCollector()


@pytest.fixture
def traces():
    return LoadTraceManager()


@pytest.fixture
def logs():
    return LoadLogAggregator()


@pytest.fixture
def alerts():
    return LoadAlertEvaluator()


@pytest.fixture
def slo_tracker():
    return LoadSLOTracker()


# ==========================================================================
# High Volume Metric Recording
# ==========================================================================

class TestHighVolumeMetricRecording:
    """Tests high-volume metric recording throughput."""

    def test_high_volume_metric_recording(self, metrics):
        """Record 10,000 metrics in under 5 seconds."""
        metrics.register_metric("load_test_counter", "counter")
        num_records = 10000

        start = time.monotonic()
        for i in range(num_records):
            metrics.record("load_test_counter", 1.0, labels={"batch": str(i % 10)})
        elapsed = time.monotonic() - start

        assert elapsed < 5.0, f"Recording {num_records} metrics took {elapsed:.2f}s (target: <5s)"
        assert metrics._total == num_records

    def test_metric_recording_throughput(self, metrics):
        """Verify throughput exceeds 2000 records/second."""
        metrics.register_metric("throughput_test", "counter")
        num_records = 5000

        start = time.monotonic()
        for i in range(num_records):
            metrics.record("throughput_test", 1.0)
        elapsed = time.monotonic() - start

        throughput = num_records / elapsed
        assert throughput > 2000, f"Throughput: {throughput:.0f} rps (target: >2000)"


# ==========================================================================
# Concurrent Span Creation
# ==========================================================================

class TestConcurrentSpanCreation:
    """Tests concurrent span creation from multiple threads."""

    def test_concurrent_span_creation(self, traces):
        """Create 1000 spans concurrently from 10 threads."""
        num_spans_per_thread = 100
        num_threads = 10
        errors = []

        def create_spans():
            try:
                for _ in range(num_spans_per_thread):
                    span = traces.start_span("load_test_op")
                    traces.end_span(span["trace_id"], span["span_id"])
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=create_spans) for _ in range(num_threads)]
        start = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        elapsed = time.monotonic() - start

        assert not errors, f"Errors during concurrent span creation: {errors}"
        total_spans = num_spans_per_thread * num_threads
        assert len(traces._spans) == total_spans
        assert elapsed < 10.0, f"Concurrent span creation took {elapsed:.2f}s"


# ==========================================================================
# High Volume Log Ingestion
# ==========================================================================

class TestHighVolumeLogIngestion:
    """Tests high-volume log ingestion throughput."""

    def test_high_volume_log_ingestion(self, logs):
        """Ingest 10,000 logs in under 5 seconds."""
        num_records = 10000

        start = time.monotonic()
        for i in range(num_records):
            logs.ingest(f"Log message {i}", level="info")
        elapsed = time.monotonic() - start

        assert elapsed < 5.0, f"Ingesting {num_records} logs took {elapsed:.2f}s (target: <5s)"
        assert logs._total == num_records


# ==========================================================================
# Concurrent Alert Evaluation
# ==========================================================================

class TestConcurrentAlertEvaluation:
    """Tests concurrent alert evaluation."""

    def test_concurrent_alert_evaluation(self, alerts):
        """Evaluate 100 alert rules against 50 metric values simultaneously."""
        for i in range(100):
            alerts.add_rule(f"rule_{i}", f"metric_{i % 50}", "gt", 0.5)

        metric_values = {f"metric_{i}": 0.8 for i in range(50)}

        start = time.monotonic()
        for _ in range(100):
            fired = alerts.evaluate_all(metric_values)
        elapsed = time.monotonic() - start

        assert elapsed < 2.0, f"100 evaluations took {elapsed:.2f}s (target: <2s)"
        assert len(fired) > 0  # Some alerts should fire


# ==========================================================================
# SLO Calculation Performance
# ==========================================================================

class TestSLOCalculationPerformance:
    """Tests SLO calculation performance at scale."""

    def test_slo_calculation_performance(self, slo_tracker):
        """Evaluate 100 SLOs with 1000 observations each in under 2 seconds."""
        slo_ids = []
        for i in range(100):
            slo_id = slo_tracker.create_slo(f"SLO_{i}", target=0.99)
            slo_ids.append(slo_id)
            for j in range(1000):
                slo_tracker.record_observation(slo_id, is_good=(j % 100 != 0))

        start = time.monotonic()
        results = slo_tracker.evaluate_all()
        elapsed = time.monotonic() - start

        assert elapsed < 2.0, f"Evaluating 100 SLOs took {elapsed:.2f}s (target: <2s)"
        assert len(results) == 100


# ==========================================================================
# Prometheus Export at Scale
# ==========================================================================

class TestPrometheusExportLargeScale:
    """Tests Prometheus export with many metric series."""

    def test_prometheus_export_large(self, metrics):
        """Export 1000+ metric series in Prometheus format."""
        for i in range(100):
            metrics.register_metric(f"metric_{i}", "gauge")

        for i in range(100):
            for j in range(10):
                metrics.record(f"metric_{i}", float(j), labels={"instance": str(j)})

        start = time.monotonic()
        output = metrics.export_prometheus()
        elapsed = time.monotonic() - start

        assert elapsed < 2.0, f"Export took {elapsed:.2f}s (target: <2s)"
        assert len(output) > 0
        lines = output.strip().split("\n")
        assert len(lines) >= 100  # At least 100 metric lines


# ==========================================================================
# Memory Usage Under Load
# ==========================================================================

class TestMemoryUsageUnderLoad:
    """Tests memory usage stays bounded under load."""

    def test_memory_usage_under_load(self, metrics):
        """Verify memory stays reasonable after 50K recordings."""
        metrics.register_metric("mem_test", "counter")

        # Measure baseline
        baseline = sys.getsizeof(metrics._series)

        for i in range(50000):
            metrics.record("mem_test", 1.0, labels={"batch": str(i % 100)})

        final = sys.getsizeof(metrics._series)
        # With 100 unique label combos, series dict should stay bounded
        # The delta is in keys, not in accumulated values
        assert len(metrics._series) <= 100  # Only 100 unique label combos

    def test_log_buffer_stays_bounded(self):
        """Verify log buffer trimming keeps memory bounded."""
        logs = LoadLogAggregator(max_buffer=1000)
        for i in range(5000):
            logs.ingest(f"msg-{i}")
        assert len(logs._buffer) <= 1000
        assert logs._total == 5000
