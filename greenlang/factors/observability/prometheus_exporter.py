# -*- coding: utf-8 -*-
"""
Factors-specific Prometheus metrics (F070).

Exports 8 metrics for the factors API, ingestion pipeline, source watch,
and QA gates. Integrates with the existing OBS-001 Prometheus stack.

Metrics:
  greenlang_factors_api_requests_total          - API request counter
  greenlang_factors_api_latency_seconds         - Request latency histogram
  greenlang_factors_search_results_count        - Search result count histogram
  greenlang_factors_match_score_top1            - Top-1 match confidence histogram
  greenlang_factors_edition_factor_count        - Factor count per edition gauge
  greenlang_factors_ingestion_rows_total        - Rows ingested counter
  greenlang_factors_watch_source_changes_total  - Source changes detected counter
  greenlang_factors_qa_gate_failures_total      - QA gate failure counter
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Gauge, Histogram, Info
    _PROM = True
except ImportError:
    _PROM = False


@dataclass
class _FallbackStore:
    counters: Dict[str, float] = field(default_factory=dict)
    gauges: Dict[str, float] = field(default_factory=dict)
    histograms: Dict[str, List[float]] = field(default_factory=dict)


class FactorsMetrics:
    """
    Central metrics registry for the Factors product.

    Uses prometheus_client when available; degrades to in-memory counters.
    """

    def __init__(self) -> None:
        self._fallback = _FallbackStore()
        if _PROM:
            self._api_requests = Counter(
                "greenlang_factors_api_requests_total",
                "Total Factors API requests",
                ["method", "path", "status"],
            )
            self._api_latency = Histogram(
                "greenlang_factors_api_latency_seconds",
                "Factors API request latency",
                ["method", "path"],
                buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            )
            self._search_results = Histogram(
                "greenlang_factors_search_results_count",
                "Number of results per search request",
                buckets=(0, 1, 5, 10, 25, 50, 100, 250, 500, 1000),
            )
            self._match_score = Histogram(
                "greenlang_factors_match_score_top1",
                "Top-1 match confidence score",
                buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
            )
            self._edition_factors = Gauge(
                "greenlang_factors_edition_factor_count",
                "Total factors in current edition",
                ["edition_id", "status"],
            )
            self._ingestion_rows = Counter(
                "greenlang_factors_ingestion_rows_total",
                "Total rows ingested",
                ["source_id", "status"],
            )
            self._watch_changes = Counter(
                "greenlang_factors_watch_source_changes_total",
                "Source changes detected by watch",
                ["source_id", "change_type"],
            )
            self._qa_failures = Counter(
                "greenlang_factors_qa_gate_failures_total",
                "QA gate failures",
                ["gate_name"],
            )

    # ------------------------------------------------------------------
    # API metrics
    # ------------------------------------------------------------------

    def record_api_request(self, method: str, path: str, status: int, latency_sec: float) -> None:
        if _PROM:
            self._api_requests.labels(method=method, path=path, status=str(status)).inc()
            self._api_latency.labels(method=method, path=path).observe(latency_sec)
        else:
            k = f"api:{method}:{path}:{status}"
            self._fallback.counters[k] = self._fallback.counters.get(k, 0) + 1

    @contextmanager
    def track_api_call(self, method: str, path: str) -> Generator[None, None, None]:
        start = time.monotonic()
        status = 200
        try:
            yield
        except Exception:
            status = 500
            raise
        finally:
            self.record_api_request(method, path, status, time.monotonic() - start)

    # ------------------------------------------------------------------
    # Search / match metrics
    # ------------------------------------------------------------------

    def record_search_results(self, count: int) -> None:
        if _PROM:
            self._search_results.observe(count)
        else:
            self._fallback.histograms.setdefault("search_results", []).append(count)

    def record_match_score(self, score: float) -> None:
        if _PROM:
            self._match_score.observe(score)
        else:
            self._fallback.histograms.setdefault("match_score", []).append(score)

    # ------------------------------------------------------------------
    # Edition metrics
    # ------------------------------------------------------------------

    def set_edition_factor_count(self, edition_id: str, status: str, count: int) -> None:
        if _PROM:
            self._edition_factors.labels(edition_id=edition_id, status=status).set(count)
        else:
            self._fallback.gauges[f"edition:{edition_id}:{status}"] = count

    # ------------------------------------------------------------------
    # Ingestion metrics
    # ------------------------------------------------------------------

    def record_ingestion(self, source_id: str, count: int, status: str = "ok") -> None:
        if _PROM:
            self._ingestion_rows.labels(source_id=source_id, status=status).inc(count)
        else:
            k = f"ingest:{source_id}:{status}"
            self._fallback.counters[k] = self._fallback.counters.get(k, 0) + count

    # ------------------------------------------------------------------
    # Watch metrics
    # ------------------------------------------------------------------

    def record_watch_change(self, source_id: str, change_type: str) -> None:
        if _PROM:
            self._watch_changes.labels(source_id=source_id, change_type=change_type).inc()
        else:
            k = f"watch:{source_id}:{change_type}"
            self._fallback.counters[k] = self._fallback.counters.get(k, 0) + 1

    # ------------------------------------------------------------------
    # QA metrics
    # ------------------------------------------------------------------

    def record_qa_failure(self, gate_name: str) -> None:
        if _PROM:
            self._qa_failures.labels(gate_name=gate_name).inc()
        else:
            k = f"qa:{gate_name}"
            self._fallback.counters[k] = self._fallback.counters.get(k, 0) + 1

    @property
    def fallback_store(self) -> _FallbackStore:
        return self._fallback


# Singleton
_metrics: Optional[FactorsMetrics] = None


def get_factors_metrics() -> FactorsMetrics:
    global _metrics
    if _metrics is None:
        _metrics = FactorsMetrics()
    return _metrics
