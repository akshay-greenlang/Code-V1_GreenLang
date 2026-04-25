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
                buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
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
            # ---- DEP2: factors-specific resolve / entitlement / signing ----
            # Names here match the PrometheusRule expressions exactly. Do not
            # rename without updating deployment/k8s/factors/base/prometheusrule.yaml
            # and deployment/observability/grafana/dashboards/factors.json.
            self._resolve_requests = Counter(
                "factors_resolve_requests_total",
                "Resolve requests by family, method profile, fallback rank, outcome",
                ["family", "method_profile", "fallback_rank", "outcome"],
            )
            self._resolve_latency = Histogram(
                "factors_resolve_latency_seconds",
                "Resolve end-to-end latency by family",
                ["family"],
                buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0),
            )
            self._entitlement_denials = Counter(
                "factors_entitlement_denials_total",
                "Entitlement / licensing denials",
                ["class", "tier"],
            )
            self._signed_receipt_failures = Counter(
                "factors_signed_receipt_failures_total",
                "Signed-receipt generation / verification failures",
                ["reason"],
            )
            self._source_watch_failures = Counter(
                "factors_source_watch_failures_total",
                "Source-watch probe failures (per source_id)",
                ["source_id"],
            )
            self._cannot_resolve_safely = Counter(
                "factors_cannot_resolve_safely_total",
                "Cannot-resolve-safely events (N-safety gate rejection)",
                ["pack_id", "method_profile"],
            )
            # ---- v0.1 Alpha (WS9-T3) — minimal Grafana dashboard support ----
            # These metrics back the 8-panel factors-v0.1-alpha dashboard. Names
            # MUST match the PromQL in
            # deployment/observability/grafana/dashboards/factors-v0.1-alpha.json
            # and the alert rules in
            # deployment/observability/prometheus/factors-v0.1-alpha-alerts.yaml.
            self._schema_validation_failures = Counter(
                "factors_schema_validation_failures_total",
                "Factor record schema validation failures (alpha gate, layer 1)",
                ["schema", "source"],
            )
            self._alpha_provenance_rejections = Counter(
                "factors_alpha_provenance_gate_rejections_total",
                "AlphaProvenanceGate.assert_valid() rejections by source + first-failure reason",
                ["source", "reason"],
            )
            self._ingestion_runs = Counter(
                "factors_ingestion_runs_total",
                "Ingestion run completions (one per ingest_from_paths invocation) by status + source",
                ["status", "source"],
            )
            self._parser_errors = Counter(
                "factors_parser_errors_total",
                "Parser errors raised by ETL parsers (CBAM/DEFRA/IPCC/EPA/CEA), by source + error_type",
                ["source", "error_type"],
            )
            self._current_edition = Gauge(
                "factors_current_edition_id_info",
                "Currently-served edition id (1.0 per active edition; 0.0 otherwise)",
                ["edition"],
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

    # ------------------------------------------------------------------
    # DEP2: resolve / entitlement / signing / source-watch metrics
    # ------------------------------------------------------------------

    def record_resolve(
        self,
        *,
        family: str,
        method_profile: str,
        fallback_rank: int,
        outcome: str,
        latency_sec: float,
    ) -> None:
        """Record one /v1/resolve call's outcome + latency."""
        if _PROM:
            self._resolve_requests.labels(
                family=family,
                method_profile=method_profile,
                fallback_rank=str(fallback_rank),
                outcome=outcome,
            ).inc()
            self._resolve_latency.labels(family=family).observe(latency_sec)
        else:
            k = f"resolve:{family}:{method_profile}:{fallback_rank}:{outcome}"
            self._fallback.counters[k] = self._fallback.counters.get(k, 0) + 1
            self._fallback.histograms.setdefault(f"resolve_latency:{family}", []).append(latency_sec)

    def record_entitlement_denial(self, *, denial_class: str, tier: str) -> None:
        if _PROM:
            self._entitlement_denials.labels(**{"class": denial_class, "tier": tier}).inc()
        else:
            k = f"denial:{denial_class}:{tier}"
            self._fallback.counters[k] = self._fallback.counters.get(k, 0) + 1

    def record_signed_receipt_failure(self, *, reason: str) -> None:
        if _PROM:
            self._signed_receipt_failures.labels(reason=reason).inc()
        else:
            k = f"signed_receipt_fail:{reason}"
            self._fallback.counters[k] = self._fallback.counters.get(k, 0) + 1

    def record_source_watch_failure(self, *, source_id: str) -> None:
        if _PROM:
            self._source_watch_failures.labels(source_id=source_id).inc()
        else:
            k = f"source_watch_fail:{source_id}"
            self._fallback.counters[k] = self._fallback.counters.get(k, 0) + 1

    def record_cannot_resolve_safely(self, *, pack_id: str, method_profile: str) -> None:
        if _PROM:
            self._cannot_resolve_safely.labels(pack_id=pack_id, method_profile=method_profile).inc()
        else:
            k = f"cannot_resolve_safely:{pack_id}:{method_profile}"
            self._fallback.counters[k] = self._fallback.counters.get(k, 0) + 1

    # ------------------------------------------------------------------
    # v0.1 Alpha — WS9-T3 (8-panel Grafana dashboard support)
    # ------------------------------------------------------------------

    def record_schema_validation_failure(
        self, *, schema: str = "factor_record_v0_1", source: str = "unknown"
    ) -> None:
        """Increment when AlphaProvenanceGate.validate() reports any schema error."""
        if _PROM:
            self._schema_validation_failures.labels(schema=schema, source=source).inc()
        else:
            k = f"schema_fail:{schema}:{source}"
            self._fallback.counters[k] = self._fallback.counters.get(k, 0) + 1

    def record_alpha_provenance_rejection(
        self, *, source: str = "unknown", reason: str = "unknown"
    ) -> None:
        """Increment when AlphaProvenanceGate.assert_valid() raises (label = first failure reason)."""
        if _PROM:
            self._alpha_provenance_rejections.labels(source=source, reason=reason).inc()
        else:
            k = f"prov_reject:{source}:{reason}"
            self._fallback.counters[k] = self._fallback.counters.get(k, 0) + 1

    def record_ingestion_run(self, *, status: str, source: str) -> None:
        """Increment at start (status=started) and end (status=success|failed) of a run."""
        if _PROM:
            self._ingestion_runs.labels(status=status, source=source).inc()
        else:
            k = f"ingest_run:{status}:{source}"
            self._fallback.counters[k] = self._fallback.counters.get(k, 0) + 1

    def record_parser_error(self, *, source: str, error_type: str) -> None:
        """Increment when an ETL parser raises (caller catches and labels)."""
        if _PROM:
            self._parser_errors.labels(source=source, error_type=error_type).inc()
        else:
            k = f"parser_err:{source}:{error_type}"
            self._fallback.counters[k] = self._fallback.counters.get(k, 0) + 1

    def set_current_edition(self, *, edition: str, active: bool = True) -> None:
        """Set the currently-served edition gauge to 1.0 (active) or 0.0 (retired)."""
        if _PROM:
            self._current_edition.labels(edition=edition).set(1.0 if active else 0.0)
        else:
            self._fallback.gauges[f"current_edition:{edition}"] = 1.0 if active else 0.0

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
