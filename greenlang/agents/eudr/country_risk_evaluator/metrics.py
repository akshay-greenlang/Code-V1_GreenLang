# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-EUDR-016: Country Risk Evaluator

18 Prometheus metrics for country risk evaluator agent service monitoring
with graceful fallback when prometheus_client is not installed.

All metric names use the ``gl_eudr_cre_`` prefix (GreenLang EUDR Country
Risk Evaluator) for consistent identification in Prometheus queries,
Grafana dashboards, and alerting rules across the GreenLang platform.

Metrics (18 per PRD Section 7.3):
    Counters (8):
        1.  gl_eudr_cre_assessments_total              - Country risk assessments completed
        2.  gl_eudr_cre_commodity_analyses_total        - Commodity risk analyses completed
        3.  gl_eudr_cre_hotspots_detected_total         - Deforestation hotspots detected
        4.  gl_eudr_cre_classifications_total           - DD classifications completed
        5.  gl_eudr_cre_reports_generated_total         - Risk reports generated
        6.  gl_eudr_cre_trade_analyses_total            - Trade flow analyses completed
        7.  gl_eudr_cre_regulatory_updates_total        - Regulatory updates tracked
        8.  gl_eudr_cre_api_errors_total                - API errors by operation

    Histograms (5):
        9.  gl_eudr_cre_assessment_duration_seconds     - Country assessment latency
        10. gl_eudr_cre_commodity_analysis_duration_seconds - Commodity analysis latency
        11. gl_eudr_cre_hotspot_detection_duration_seconds - Hotspot detection latency
        12. gl_eudr_cre_classification_duration_seconds - DD classification latency
        13. gl_eudr_cre_report_generation_duration_seconds - Report generation latency

    Gauges (5):
        14. gl_eudr_cre_active_hotspots                 - Active deforestation hotspots
        15. gl_eudr_cre_countries_assessed               - Countries with active assessments
        16. gl_eudr_cre_high_risk_countries              - Countries classified as high risk
        17. gl_eudr_cre_pending_reclassifications        - Pending country reclassifications
        18. gl_eudr_cre_stale_assessments                - Assessments exceeding freshness limit

Label Values Reference:
    risk_level:
        low, standard, high.
    commodity:
        cattle, cocoa, coffee, oil_palm, rubber, soya, wood.
    dd_level:
        simplified, standard, enhanced.
    report_type:
        country_profile, commodity_matrix, comparative, trend,
        due_diligence, executive_summary.
    report_format:
        pdf, json, html, csv, excel.
    severity:
        low, medium, high, critical.
    operation:
        assess, analyze, detect, evaluate, classify, generate,
        track, compare, search, export.

Example:
    >>> from greenlang.agents.eudr.country_risk_evaluator.metrics import (
    ...     record_assessment_completed,
    ...     record_commodity_analysis,
    ...     record_hotspot_detected,
    ...     observe_assessment_duration,
    ...     set_countries_assessed,
    ... )
    >>> record_assessment_completed("high")
    >>> record_commodity_analysis("oil_palm")
    >>> record_hotspot_detected("critical")
    >>> observe_assessment_duration(0.045)
    >>> set_countries_assessed(205)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-016 Country Risk Evaluator (GL-EUDR-CRE-016)
Status: Production Ready
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Graceful prometheus_client import
# ---------------------------------------------------------------------------

try:
    from prometheus_client import Counter, Gauge, Histogram
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.info(
        "prometheus_client not installed; "
        "country risk evaluator metrics disabled"
    )

# ---------------------------------------------------------------------------
# Safe metric registration helpers (avoid collisions with other modules)
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    from prometheus_client import REGISTRY as _REGISTRY

    def _safe_counter(
        name: str, doc: str, labelnames: list = None,  # type: ignore[assignment]
    ):  # type: ignore[return]
        """Create a Counter or retrieve existing one to avoid registry collisions."""
        try:
            return Counter(name, doc, labelnames=labelnames or [])
        except ValueError:
            for collector in _REGISTRY._names_to_collectors.values():
                if hasattr(collector, "_name") and collector._name == name:
                    return collector
            from prometheus_client import CollectorRegistry
            return Counter(
                name, doc, labelnames=labelnames or [],
                registry=CollectorRegistry(),
            )

    def _safe_histogram(
        name: str, doc: str, labelnames: list = None,  # type: ignore[assignment]
        buckets: tuple = (),
    ):  # type: ignore[return]
        """Create a Histogram or retrieve existing one."""
        try:
            kw = {}
            if buckets:
                kw["buckets"] = buckets
            return Histogram(
                name, doc, labelnames=labelnames or [], **kw,
            )
        except ValueError:
            for collector in _REGISTRY._names_to_collectors.values():
                if hasattr(collector, "_name") and collector._name == name:
                    return collector
            from prometheus_client import CollectorRegistry
            kw = {}
            if buckets:
                kw["buckets"] = buckets
            return Histogram(
                name, doc, labelnames=labelnames or [],
                registry=CollectorRegistry(), **kw,
            )

    def _safe_gauge(
        name: str, doc: str, labelnames: list = None,  # type: ignore[assignment]
    ):  # type: ignore[return]
        """Create a Gauge or retrieve existing one."""
        try:
            return Gauge(name, doc, labelnames=labelnames or [])
        except ValueError:
            for collector in _REGISTRY._names_to_collectors.values():
                if hasattr(collector, "_name") and collector._name == name:
                    return collector
            from prometheus_client import CollectorRegistry
            return Gauge(
                name, doc, labelnames=labelnames or [],
                registry=CollectorRegistry(),
            )

# ---------------------------------------------------------------------------
# Metric definitions (18 metrics per PRD Section 7.3)
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    # -- Counters (8) --------------------------------------------------------

    # 1. Country risk assessments completed by risk level
    cre_assessments_total = _safe_counter(
        "gl_eudr_cre_assessments_total",
        "Total country risk assessments completed",
        labelnames=["risk_level"],
    )

    # 2. Commodity risk analyses completed by commodity type
    cre_commodity_analyses_total = _safe_counter(
        "gl_eudr_cre_commodity_analyses_total",
        "Total commodity risk analyses completed",
        labelnames=["commodity"],
    )

    # 3. Deforestation hotspots detected by severity
    cre_hotspots_detected_total = _safe_counter(
        "gl_eudr_cre_hotspots_detected_total",
        "Total deforestation hotspots detected",
        labelnames=["severity"],
    )

    # 4. Due diligence classifications completed by level
    cre_classifications_total = _safe_counter(
        "gl_eudr_cre_classifications_total",
        "Total due diligence classifications completed",
        labelnames=["dd_level"],
    )

    # 5. Risk reports generated by type and format
    cre_reports_generated_total = _safe_counter(
        "gl_eudr_cre_reports_generated_total",
        "Total risk reports generated",
        labelnames=["report_type", "report_format"],
    )

    # 6. Trade flow analyses completed by commodity
    cre_trade_analyses_total = _safe_counter(
        "gl_eudr_cre_trade_analyses_total",
        "Total trade flow analyses completed",
        labelnames=["commodity"],
    )

    # 7. Regulatory updates tracked
    cre_regulatory_updates_total = _safe_counter(
        "gl_eudr_cre_regulatory_updates_total",
        "Total regulatory updates tracked",
    )

    # 8. API errors by operation type
    cre_api_errors_total = _safe_counter(
        "gl_eudr_cre_api_errors_total",
        "Total API errors across all endpoints",
        labelnames=["operation"],
    )

    # -- Histograms (5) ------------------------------------------------------

    # 9. Country risk assessment latency
    cre_assessment_duration_seconds = _safe_histogram(
        "gl_eudr_cre_assessment_duration_seconds",
        "Country risk assessment processing latency",
        buckets=(
            0.005, 0.01, 0.025, 0.05, 0.1, 0.2,
            0.5, 1.0, 2.5, 5.0, 10.0, 30.0,
        ),
    )

    # 10. Commodity analysis latency
    cre_commodity_analysis_duration_seconds = _safe_histogram(
        "gl_eudr_cre_commodity_analysis_duration_seconds",
        "Commodity risk analysis processing latency",
        buckets=(
            0.005, 0.01, 0.025, 0.05, 0.1, 0.2,
            0.5, 1.0, 2.5, 5.0, 10.0,
        ),
    )

    # 11. Hotspot detection latency
    cre_hotspot_detection_duration_seconds = _safe_histogram(
        "gl_eudr_cre_hotspot_detection_duration_seconds",
        "Deforestation hotspot detection processing latency",
        buckets=(
            0.1, 0.25, 0.5, 1.0, 2.5, 5.0,
            10.0, 30.0, 60.0, 120.0,
        ),
    )

    # 12. DD classification latency
    cre_classification_duration_seconds = _safe_histogram(
        "gl_eudr_cre_classification_duration_seconds",
        "Due diligence classification processing latency",
        buckets=(
            0.001, 0.005, 0.01, 0.025, 0.05, 0.1,
            0.25, 0.5, 1.0, 2.5,
        ),
    )

    # 13. Report generation latency
    cre_report_generation_duration_seconds = _safe_histogram(
        "gl_eudr_cre_report_generation_duration_seconds",
        "Risk report generation processing latency",
        buckets=(
            0.5, 1.0, 2.5, 5.0, 10.0, 30.0,
            60.0, 120.0, 300.0, 600.0,
        ),
    )

    # -- Gauges (5) ----------------------------------------------------------

    # 14. Current count of active deforestation hotspots
    cre_active_hotspots = _safe_gauge(
        "gl_eudr_cre_active_hotspots",
        "Current count of active deforestation hotspots",
    )

    # 15. Countries with active risk assessments
    cre_countries_assessed = _safe_gauge(
        "gl_eudr_cre_countries_assessed",
        "Countries with active risk assessments in the database",
    )

    # 16. Countries classified as high risk
    cre_high_risk_countries = _safe_gauge(
        "gl_eudr_cre_high_risk_countries",
        "Countries currently classified as high risk",
    )

    # 17. Pending country reclassifications
    cre_pending_reclassifications = _safe_gauge(
        "gl_eudr_cre_pending_reclassifications",
        "Country reclassifications pending review or propagation",
    )

    # 18. Stale assessments exceeding data freshness limit
    cre_stale_assessments = _safe_gauge(
        "gl_eudr_cre_stale_assessments",
        "Risk assessments exceeding the configured data freshness limit",
    )

else:
    # No-op placeholders so callers never need to guard on PROMETHEUS_AVAILABLE
    cre_assessments_total = None              # type: ignore[assignment]
    cre_commodity_analyses_total = None        # type: ignore[assignment]
    cre_hotspots_detected_total = None         # type: ignore[assignment]
    cre_classifications_total = None           # type: ignore[assignment]
    cre_reports_generated_total = None         # type: ignore[assignment]
    cre_trade_analyses_total = None            # type: ignore[assignment]
    cre_regulatory_updates_total = None        # type: ignore[assignment]
    cre_api_errors_total = None               # type: ignore[assignment]
    cre_assessment_duration_seconds = None     # type: ignore[assignment]
    cre_commodity_analysis_duration_seconds = None  # type: ignore[assignment]
    cre_hotspot_detection_duration_seconds = None   # type: ignore[assignment]
    cre_classification_duration_seconds = None      # type: ignore[assignment]
    cre_report_generation_duration_seconds = None   # type: ignore[assignment]
    cre_active_hotspots = None                # type: ignore[assignment]
    cre_countries_assessed = None             # type: ignore[assignment]
    cre_high_risk_countries = None            # type: ignore[assignment]
    cre_pending_reclassifications = None       # type: ignore[assignment]
    cre_stale_assessments = None              # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def record_assessment_completed(risk_level: str) -> None:
    """Record a country risk assessment completion event.

    Args:
        risk_level: Risk classification result (low, standard, high).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    cre_assessments_total.labels(risk_level=risk_level).inc()


def record_commodity_analysis(commodity: str) -> None:
    """Record a commodity risk analysis completion event.

    Args:
        commodity: EUDR commodity type (cattle, cocoa, coffee,
            oil_palm, rubber, soya, wood).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    cre_commodity_analyses_total.labels(commodity=commodity).inc()


def record_hotspot_detected(severity: str) -> None:
    """Record a deforestation hotspot detection event.

    Args:
        severity: Hotspot severity (low, medium, high, critical).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    cre_hotspots_detected_total.labels(severity=severity).inc()


def record_classification_completed(dd_level: str) -> None:
    """Record a due diligence classification completion event.

    Args:
        dd_level: Due diligence level (simplified, standard, enhanced).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    cre_classifications_total.labels(dd_level=dd_level).inc()


def record_report_generated(report_type: str, report_format: str) -> None:
    """Record a risk report generation event.

    Args:
        report_type: Type of report (country_profile, commodity_matrix,
            comparative, trend, due_diligence, executive_summary).
        report_format: Output format (pdf, json, html, csv, excel).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    cre_reports_generated_total.labels(
        report_type=report_type, report_format=report_format,
    ).inc()


def record_trade_analysis(commodity: str) -> None:
    """Record a trade flow analysis completion event.

    Args:
        commodity: EUDR commodity type.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    cre_trade_analyses_total.labels(commodity=commodity).inc()


def record_regulatory_update() -> None:
    """Record a regulatory update tracking event."""
    if not PROMETHEUS_AVAILABLE:
        return
    cre_regulatory_updates_total.inc()


def record_api_error(operation: str) -> None:
    """Record an API error event by operation type.

    Args:
        operation: Type of operation that failed (assess, analyze,
            detect, evaluate, classify, generate, track, compare,
            search, export).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    cre_api_errors_total.labels(operation=operation).inc()


def observe_assessment_duration(seconds: float) -> None:
    """Record the duration of a country risk assessment operation.

    Args:
        seconds: Operation wall-clock time in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    cre_assessment_duration_seconds.observe(seconds)


def observe_commodity_analysis_duration(seconds: float) -> None:
    """Record the duration of a commodity risk analysis operation.

    Args:
        seconds: Operation wall-clock time in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    cre_commodity_analysis_duration_seconds.observe(seconds)


def observe_hotspot_detection_duration(seconds: float) -> None:
    """Record the duration of a hotspot detection operation.

    Args:
        seconds: Operation wall-clock time in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    cre_hotspot_detection_duration_seconds.observe(seconds)


def observe_classification_duration(seconds: float) -> None:
    """Record the duration of a due diligence classification operation.

    Args:
        seconds: Operation wall-clock time in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    cre_classification_duration_seconds.observe(seconds)


def observe_report_generation_duration(seconds: float) -> None:
    """Record the duration of a report generation operation.

    Args:
        seconds: Operation wall-clock time in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    cre_report_generation_duration_seconds.observe(seconds)


def set_active_hotspots(count: int) -> None:
    """Set the gauge for active deforestation hotspots.

    Args:
        count: Number of active hotspots. Must be >= 0.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    cre_active_hotspots.set(count)


def set_countries_assessed(count: int) -> None:
    """Set the gauge for countries with active risk assessments.

    Args:
        count: Number of assessed countries. Must be >= 0.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    cre_countries_assessed.set(count)


def set_high_risk_countries(count: int) -> None:
    """Set the gauge for countries classified as high risk.

    Args:
        count: Number of high-risk countries. Must be >= 0.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    cre_high_risk_countries.set(count)


def set_pending_reclassifications(count: int) -> None:
    """Set the gauge for pending country reclassifications.

    Args:
        count: Number of pending reclassifications. Must be >= 0.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    cre_pending_reclassifications.set(count)


def set_stale_assessments(count: int) -> None:
    """Set the gauge for stale risk assessments.

    Args:
        count: Number of stale assessments. Must be >= 0.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    cre_stale_assessments.set(count)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "PROMETHEUS_AVAILABLE",
    # Metric objects
    "cre_assessments_total",
    "cre_commodity_analyses_total",
    "cre_hotspots_detected_total",
    "cre_classifications_total",
    "cre_reports_generated_total",
    "cre_trade_analyses_total",
    "cre_regulatory_updates_total",
    "cre_api_errors_total",
    "cre_assessment_duration_seconds",
    "cre_commodity_analysis_duration_seconds",
    "cre_hotspot_detection_duration_seconds",
    "cre_classification_duration_seconds",
    "cre_report_generation_duration_seconds",
    "cre_active_hotspots",
    "cre_countries_assessed",
    "cre_high_risk_countries",
    "cre_pending_reclassifications",
    "cre_stale_assessments",
    # Helper functions
    "record_assessment_completed",
    "record_commodity_analysis",
    "record_hotspot_detected",
    "record_classification_completed",
    "record_report_generated",
    "record_trade_analysis",
    "record_regulatory_update",
    "record_api_error",
    "observe_assessment_duration",
    "observe_commodity_analysis_duration",
    "observe_hotspot_detection_duration",
    "observe_classification_duration",
    "observe_report_generation_duration",
    "set_active_hotspots",
    "set_countries_assessed",
    "set_high_risk_countries",
    "set_pending_reclassifications",
    "set_stale_assessments",
]
