# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-EUDR-005: Land Use Change Detector Agent

18 Prometheus metrics for land use change detector agent service monitoring
with graceful fallback when prometheus_client is not installed.

All metric names use the ``gl_eudr_luc_`` prefix (GreenLang EUDR
Land Use Change) for consistent identification in Prometheus queries,
Grafana dashboards, and alerting rules across the GreenLang platform.

Metrics:
    1.  gl_eudr_luc_classifications_total             (Counter, labels: method, status)
    2.  gl_eudr_luc_transitions_detected_total        (Counter, labels: transition_type, status)
    3.  gl_eudr_luc_trajectories_analyzed_total        (Counter, labels: trajectory_type, status)
    4.  gl_eudr_luc_cutoff_verifications_total          (Counter, labels: verdict, commodity)
    5.  gl_eudr_luc_conversions_detected_total          (Counter, labels: conversion_type, commodity)
    6.  gl_eudr_luc_risk_assessments_total              (Counter, labels: risk_tier, commodity)
    7.  gl_eudr_luc_urban_analyses_total                (Counter, labels: status)
    8.  gl_eudr_luc_reports_generated_total              (Counter, labels: format, report_type)
    9.  gl_eudr_luc_non_compliant_parcels_total          (Counter, labels: commodity, country)
    10. gl_eudr_luc_api_errors_total                     (Counter, labels: operation)
    11. gl_eudr_luc_analysis_duration_seconds             (Histogram, labels: operation)
    12. gl_eudr_luc_batch_duration_seconds                (Histogram)
    13. gl_eudr_luc_active_analyses                       (Gauge)
    14. gl_eudr_luc_avg_classification_confidence         (Gauge)
    15. gl_eudr_luc_avg_transition_magnitude              (Gauge)
    16. gl_eudr_luc_data_quality_score                    (Gauge, labels: source)
    17. gl_eudr_luc_total_analyzed_area_ha                (Gauge)
    18. gl_eudr_luc_conversion_risk_score                 (Gauge, labels: commodity)

Label Values Reference:
    method (classification):
        spectral, vegetation_index, phenology, texture, ensemble.
    status:
        success, error, cached, inconclusive.
    transition_type:
        deforestation, degradation, reforestation, natural_regrowth,
        cropland_expansion, urban_expansion, wetland_conversion,
        stable, seasonal_change, unknown.
    trajectory_type:
        stable, abrupt_change, gradual_change, oscillating, recovery.
    verdict:
        compliant, non_compliant, under_review, inconclusive, exempt.
    commodity:
        cattle, cocoa, coffee, palm_oil, rubber, soya, wood.
    conversion_type:
        cattle_ranching, cocoa_farming, coffee_cultivation,
        palm_oil_plantation, rubber_plantation, soya_cultivation,
        timber_harvesting.
    risk_tier:
        low, moderate, high, critical.
    country:
        ISO 3166-1 alpha-2 country codes.
    format:
        json, pdf, csv, eudr_xml.
    report_type:
        full, summary, transition_analysis, risk_assessment,
        regulatory_submission.
    operation:
        classification, transition_detection, trajectory_analysis,
        cutoff_verification, conversion_detection, risk_assessment,
        urban_analysis, report_generation, batch_analyze.

Example:
    >>> from greenlang.agents.eudr.land_use_change.metrics import (
    ...     record_classification,
    ...     record_transition,
    ...     record_verification,
    ...     record_error,
    ... )
    >>> record_classification("ensemble", "success")
    >>> record_transition("deforestation", "success")
    >>> record_verification("non_compliant", "soya")
    >>> record_error("classification")

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-005 Land Use Change Detector Agent (GL-EUDR-LUC-005)
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
        "land use change detector metrics disabled"
    )

# ---------------------------------------------------------------------------
# Safe metric registration helpers (avoid collisions with other modules)
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    from prometheus_client import REGISTRY as _REGISTRY

    def _safe_counter(
        name: str, doc: str, labelnames: list = None,  # type: ignore[assignment]
    ):
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
    ):
        """Create a Histogram or retrieve existing one."""
        try:
            kw = {}
            if buckets:
                kw["buckets"] = buckets
            return Histogram(name, doc, labelnames=labelnames or [], **kw)
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
    ):
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
# Metric definitions (18 metrics per PRD Section 7.6)
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    # 1. Land use classifications by method and status
    luc_classifications_total = _safe_counter(
        "gl_eudr_luc_classifications_total",
        "Total land use classifications performed",
        labelnames=["method", "status"],
    )

    # 2. Transitions detected by type and status
    luc_transitions_detected_total = _safe_counter(
        "gl_eudr_luc_transitions_detected_total",
        "Total land use transitions detected",
        labelnames=["transition_type", "status"],
    )

    # 3. Trajectories analyzed by type and status
    luc_trajectories_analyzed_total = _safe_counter(
        "gl_eudr_luc_trajectories_analyzed_total",
        "Total temporal trajectories analyzed",
        labelnames=["trajectory_type", "status"],
    )

    # 4. Cutoff verifications by verdict and commodity
    luc_cutoff_verifications_total = _safe_counter(
        "gl_eudr_luc_cutoff_verifications_total",
        "Total EUDR cutoff date verifications performed",
        labelnames=["verdict", "commodity"],
    )

    # 5. Conversions detected by type and commodity
    luc_conversions_detected_total = _safe_counter(
        "gl_eudr_luc_conversions_detected_total",
        "Total cropland/commodity conversions detected",
        labelnames=["conversion_type", "commodity"],
    )

    # 6. Risk assessments by tier and commodity
    luc_risk_assessments_total = _safe_counter(
        "gl_eudr_luc_risk_assessments_total",
        "Total conversion risk assessments performed",
        labelnames=["risk_tier", "commodity"],
    )

    # 7. Urban encroachment analyses by status
    luc_urban_analyses_total = _safe_counter(
        "gl_eudr_luc_urban_analyses_total",
        "Total urban encroachment analyses performed",
        labelnames=["status"],
    )

    # 8. Compliance reports generated by format and type
    luc_reports_generated_total = _safe_counter(
        "gl_eudr_luc_reports_generated_total",
        "Total compliance reports generated",
        labelnames=["format", "report_type"],
    )

    # 9. Non-compliant parcels detected by commodity and country
    luc_non_compliant_parcels_total = _safe_counter(
        "gl_eudr_luc_non_compliant_parcels_total",
        "Total parcels classified as non-compliant since EUDR cutoff",
        labelnames=["commodity", "country"],
    )

    # 10. API errors by operation type
    luc_api_errors_total = _safe_counter(
        "gl_eudr_luc_api_errors_total",
        "Total API errors encountered by operation type",
        labelnames=["operation"],
    )

    # 11. Analysis operation duration by operation type
    luc_analysis_duration_seconds = _safe_histogram(
        "gl_eudr_luc_analysis_duration_seconds",
        "Duration of land use change analysis operations in seconds",
        labelnames=["operation"],
        buckets=(
            0.1, 0.25, 0.5, 1.0, 2.5, 5.0,
            10.0, 30.0, 60.0, 120.0, 300.0,
        ),
    )

    # 12. Batch analysis total duration
    luc_batch_duration_seconds = _safe_histogram(
        "gl_eudr_luc_batch_duration_seconds",
        "Duration of complete batch analysis jobs in seconds",
        buckets=(
            1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
            600.0, 1800.0, 3600.0,
        ),
    )

    # 13. Currently active analyses
    luc_active_analyses = _safe_gauge(
        "gl_eudr_luc_active_analyses",
        "Number of land use change analyses currently in progress",
    )

    # 14. Average classification confidence across analyzed parcels
    luc_avg_classification_confidence = _safe_gauge(
        "gl_eudr_luc_avg_classification_confidence",
        "Average classification confidence across analyzed parcels",
    )

    # 15. Average transition magnitude across detected transitions
    luc_avg_transition_magnitude = _safe_gauge(
        "gl_eudr_luc_avg_transition_magnitude",
        "Average transition magnitude across detected transitions",
    )

    # 16. Data quality score by source
    luc_data_quality_score = _safe_gauge(
        "gl_eudr_luc_data_quality_score",
        "Data quality score by analysis source (0-100)",
        labelnames=["source"],
    )

    # 17. Total analyzed area in hectares
    luc_total_analyzed_area_ha = _safe_gauge(
        "gl_eudr_luc_total_analyzed_area_ha",
        "Total land area analyzed for land use change in hectares",
    )

    # 18. Average conversion risk score by commodity
    luc_conversion_risk_score = _safe_gauge(
        "gl_eudr_luc_conversion_risk_score",
        "Average conversion risk score by EUDR commodity (0-100)",
        labelnames=["commodity"],
    )

else:
    # No-op placeholders so callers never need to guard on PROMETHEUS_AVAILABLE
    luc_classifications_total = None              # type: ignore[assignment]
    luc_transitions_detected_total = None         # type: ignore[assignment]
    luc_trajectories_analyzed_total = None         # type: ignore[assignment]
    luc_cutoff_verifications_total = None          # type: ignore[assignment]
    luc_conversions_detected_total = None          # type: ignore[assignment]
    luc_risk_assessments_total = None              # type: ignore[assignment]
    luc_urban_analyses_total = None                # type: ignore[assignment]
    luc_reports_generated_total = None             # type: ignore[assignment]
    luc_non_compliant_parcels_total = None         # type: ignore[assignment]
    luc_api_errors_total = None                    # type: ignore[assignment]
    luc_analysis_duration_seconds = None           # type: ignore[assignment]
    luc_batch_duration_seconds = None              # type: ignore[assignment]
    luc_active_analyses = None                     # type: ignore[assignment]
    luc_avg_classification_confidence = None       # type: ignore[assignment]
    luc_avg_transition_magnitude = None            # type: ignore[assignment]
    luc_data_quality_score = None                  # type: ignore[assignment]
    luc_total_analyzed_area_ha = None              # type: ignore[assignment]
    luc_conversion_risk_score = None               # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def record_classification(method: str, status: str) -> None:
    """Record a land use classification event.

    Args:
        method: Classification method used (e.g., 'spectral',
            'vegetation_index', 'phenology', 'texture', 'ensemble').
        status: Classification status ('success', 'error', 'cached').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    luc_classifications_total.labels(method=method, status=status).inc()


def record_transition(transition_type: str, status: str) -> None:
    """Record a land use transition detection event.

    Args:
        transition_type: Type of transition detected (e.g.,
            'deforestation', 'cropland_expansion', 'urban_expansion',
            'stable', 'unknown').
        status: Detection status ('success', 'error', 'cached').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    luc_transitions_detected_total.labels(
        transition_type=transition_type, status=status,
    ).inc()


def record_trajectory(trajectory_type: str, status: str) -> None:
    """Record a temporal trajectory analysis event.

    Args:
        trajectory_type: Classified trajectory type (e.g., 'stable',
            'abrupt_change', 'gradual_change', 'oscillating',
            'recovery').
        status: Analysis status ('success', 'error', 'cached').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    luc_trajectories_analyzed_total.labels(
        trajectory_type=trajectory_type, status=status,
    ).inc()


def record_verification(verdict: str, commodity: str) -> None:
    """Record a cutoff date verification event.

    Args:
        verdict: Verification verdict ('compliant', 'non_compliant',
            'under_review', 'inconclusive', 'exempt').
        commodity: EUDR commodity (e.g., 'soya', 'palm_oil').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    luc_cutoff_verifications_total.labels(
        verdict=verdict, commodity=commodity,
    ).inc()


def record_conversion(conversion_type: str, commodity: str) -> None:
    """Record a cropland/commodity conversion detection event.

    Args:
        conversion_type: Type of conversion detected (e.g.,
            'cattle_ranching', 'palm_oil_plantation',
            'soya_cultivation').
        commodity: EUDR commodity associated with the conversion.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    luc_conversions_detected_total.labels(
        conversion_type=conversion_type, commodity=commodity,
    ).inc()


def record_risk_assessment(risk_tier: str, commodity: str) -> None:
    """Record a conversion risk assessment event.

    Args:
        risk_tier: Classified risk tier ('low', 'moderate', 'high',
            'critical').
        commodity: EUDR commodity assessed.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    luc_risk_assessments_total.labels(
        risk_tier=risk_tier, commodity=commodity,
    ).inc()


def record_urban_analysis(status: str) -> None:
    """Record an urban encroachment analysis event.

    Args:
        status: Analysis status ('success', 'error', 'cached').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    luc_urban_analyses_total.labels(status=status).inc()


def record_report(report_format: str, report_type: str) -> None:
    """Record a compliance report generation event.

    Args:
        report_format: Report format ('json', 'pdf', 'csv', 'eudr_xml').
        report_type: Type of report ('full', 'summary',
            'transition_analysis', 'risk_assessment',
            'regulatory_submission').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    luc_reports_generated_total.labels(
        format=report_format, report_type=report_type,
    ).inc()


def record_error(operation: str) -> None:
    """Record an API error event by operation type.

    Args:
        operation: Type of operation that failed (e.g.,
            'classification', 'transition_detection',
            'trajectory_analysis', 'cutoff_verification',
            'conversion_detection', 'risk_assessment',
            'urban_analysis', 'report_generation', 'batch_analyze').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    luc_api_errors_total.labels(operation=operation).inc()


def record_batch_job(seconds: float) -> None:
    """Record the duration of a complete batch analysis job.

    Args:
        seconds: Total batch job wall-clock time in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    luc_batch_duration_seconds.observe(seconds)


def observe_analysis_duration(operation: str, seconds: float) -> None:
    """Record the duration of a land use change analysis operation.

    Args:
        operation: Type of analysis operation (e.g.,
            'classification', 'transition_detection',
            'trajectory_analysis', 'cutoff_verification',
            'conversion_detection', 'risk_assessment',
            'urban_analysis', 'report_generation', 'batch_analyze').
        seconds: Operation wall-clock time in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    luc_analysis_duration_seconds.labels(operation=operation).observe(seconds)


def set_active_analyses(count: int) -> None:
    """Set the gauge for number of analyses currently in progress.

    Args:
        count: Number of active analyses. Must be >= 0.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    luc_active_analyses.set(count)


def set_avg_classification_confidence(confidence: float) -> None:
    """Set the gauge for average classification confidence.

    Args:
        confidence: Average classification confidence (0.0-1.0).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    luc_avg_classification_confidence.set(confidence)


def set_avg_transition_magnitude(magnitude: float) -> None:
    """Set the gauge for average transition magnitude.

    Args:
        magnitude: Average transition magnitude (0.0-1.0).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    luc_avg_transition_magnitude.set(magnitude)


def set_data_quality_score(source: str, score: float) -> None:
    """Set the data quality score gauge for an analysis source.

    Args:
        source: Analysis source identifier (e.g., 'sentinel_2',
            'landsat_8', 'hansen_gfc', 'copernicus_glc').
        score: Data quality score (0-100).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    luc_data_quality_score.labels(source=source).set(score)


def set_total_analyzed_area_ha(area_ha: float) -> None:
    """Set the gauge for total analyzed land area.

    Args:
        area_ha: Total analyzed area in hectares.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    luc_total_analyzed_area_ha.set(area_ha)


def set_conversion_risk_score(commodity: str, score: float) -> None:
    """Set the conversion risk score gauge for a commodity.

    Args:
        commodity: EUDR commodity identifier.
        score: Average conversion risk score (0-100).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    luc_conversion_risk_score.labels(commodity=commodity).set(score)


def record_non_compliant_parcel(commodity: str, country: str) -> None:
    """Record a parcel classified as non-compliant since EUDR cutoff.

    Args:
        commodity: EUDR commodity affected.
        country: ISO 3166-1 alpha-2 country code.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    luc_non_compliant_parcels_total.labels(
        commodity=commodity, country=country,
    ).inc()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "PROMETHEUS_AVAILABLE",
    # Metric objects
    "luc_classifications_total",
    "luc_transitions_detected_total",
    "luc_trajectories_analyzed_total",
    "luc_cutoff_verifications_total",
    "luc_conversions_detected_total",
    "luc_risk_assessments_total",
    "luc_urban_analyses_total",
    "luc_reports_generated_total",
    "luc_non_compliant_parcels_total",
    "luc_api_errors_total",
    "luc_analysis_duration_seconds",
    "luc_batch_duration_seconds",
    "luc_active_analyses",
    "luc_avg_classification_confidence",
    "luc_avg_transition_magnitude",
    "luc_data_quality_score",
    "luc_total_analyzed_area_ha",
    "luc_conversion_risk_score",
    # Helper functions
    "record_classification",
    "record_transition",
    "record_trajectory",
    "record_verification",
    "record_conversion",
    "record_risk_assessment",
    "record_urban_analysis",
    "record_report",
    "record_error",
    "record_batch_job",
    "observe_analysis_duration",
    "set_active_analyses",
    "set_avg_classification_confidence",
    "set_avg_transition_magnitude",
    "set_data_quality_score",
    "set_total_analyzed_area_ha",
    "set_conversion_risk_score",
    "record_non_compliant_parcel",
]
