# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-EUDR-004: Forest Cover Analysis Agent

18 Prometheus metrics for forest cover analysis agent service monitoring
with graceful fallback when prometheus_client is not installed.

All metric names use the ``gl_eudr_fca_`` prefix (GreenLang EUDR
Forest Cover Analysis) for consistent identification in Prometheus
queries, Grafana dashboards, and alerting rules across the GreenLang
platform.

Metrics:
    1.  gl_eudr_fca_density_analyses_total           (Counter, labels: method, status)
    2.  gl_eudr_fca_classifications_total            (Counter, labels: forest_type, method)
    3.  gl_eudr_fca_reconstructions_total            (Counter, labels: status)
    4.  gl_eudr_fca_verdicts_total                   (Counter, labels: verdict, commodity)
    5.  gl_eudr_fca_height_estimates_total            (Counter, labels: source, status)
    6.  gl_eudr_fca_fragmentation_analyses_total      (Counter, labels: level)
    7.  gl_eudr_fca_biomass_estimates_total            (Counter, labels: source, status)
    8.  gl_eudr_fca_reports_generated_total            (Counter, labels: format, report_type)
    9.  gl_eudr_fca_deforested_plots_total             (Counter, labels: commodity, country)
    10. gl_eudr_fca_degraded_plots_total               (Counter, labels: commodity, country)
    11. gl_eudr_fca_api_errors_total                   (Counter, labels: operation)
    12. gl_eudr_fca_analysis_duration_seconds           (Histogram, labels: operation)
    13. gl_eudr_fca_batch_duration_seconds              (Histogram)
    14. gl_eudr_fca_active_analyses                     (Gauge)
    15. gl_eudr_fca_avg_canopy_density                  (Gauge)
    16. gl_eudr_fca_avg_confidence_score                (Gauge)
    17. gl_eudr_fca_data_quality_score                  (Gauge, labels: source)
    18. gl_eudr_fca_forest_area_ha                      (Gauge)

Label Values Reference:
    method (density):
        spectral_unmixing, ndvi_regression, dimidiation,
        sub_pixel_detection.
    status:
        success, error, cached, inconclusive.
    forest_type:
        primary_tropical, secondary_tropical, tropical_dry,
        temperate_broadleaf, temperate_coniferous, boreal,
        mangrove, plantation, agroforestry, non_forest.
    method (classification):
        spectral_signature, phenological, structural,
        multi_temporal, ensemble.
    verdict:
        deforestation_free, deforested, degraded, inconclusive.
    commodity:
        cattle, cocoa, coffee, oil_palm, rubber, soya, wood.
    country:
        ISO 3166-1 alpha-2 country codes.
    source (height):
        gedi_l2a, icesat2_atl08, sentinel2_texture,
        global_map_eth, global_map_meta.
    level (fragmentation):
        intact, slightly_fragmented, moderately_fragmented,
        highly_fragmented, severely_fragmented.
    source (biomass):
        esa_cci, gedi_l4a, sar_regression, ndvi_allometric.
    format:
        json, pdf, csv, eudr_xml.
    report_type:
        full, summary, compliance, evidence.
    operation:
        density_analysis, classification, reconstruction,
        verdict, height_estimate, fragmentation_analysis,
        biomass_estimate, report_generation, batch_analyze.

Example:
    >>> from greenlang.agents.eudr.forest_cover_analysis.metrics import (
    ...     record_density_analysis,
    ...     record_verdict,
    ...     observe_analysis_duration,
    ...     set_avg_canopy_density,
    ... )
    >>> record_density_analysis("spectral_unmixing", "success")
    >>> record_verdict("deforestation_free", "soya")
    >>> observe_analysis_duration("density_analysis", 2.5)
    >>> set_avg_canopy_density(65.3)

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-004 Forest Cover Analysis Agent (GL-EUDR-FCA-004)
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
        "forest cover analysis metrics disabled"
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
    # 1. Canopy density analyses by method and status
    fca_density_analyses_total = _safe_counter(
        "gl_eudr_fca_density_analyses_total",
        "Total canopy density analyses performed",
        labelnames=["method", "status"],
    )

    # 2. Forest type classifications by type and method
    fca_classifications_total = _safe_counter(
        "gl_eudr_fca_classifications_total",
        "Total forest type classifications performed",
        labelnames=["forest_type", "method"],
    )

    # 3. Historical cover reconstructions by status
    fca_reconstructions_total = _safe_counter(
        "gl_eudr_fca_reconstructions_total",
        "Total historical cover reconstructions performed",
        labelnames=["status"],
    )

    # 4. Deforestation-free verdicts by verdict type and commodity
    fca_verdicts_total = _safe_counter(
        "gl_eudr_fca_verdicts_total",
        "Total deforestation-free verification verdicts issued",
        labelnames=["verdict", "commodity"],
    )

    # 5. Canopy height estimates by source and status
    fca_height_estimates_total = _safe_counter(
        "gl_eudr_fca_height_estimates_total",
        "Total canopy height estimates performed",
        labelnames=["source", "status"],
    )

    # 6. Fragmentation analyses by level
    fca_fragmentation_analyses_total = _safe_counter(
        "gl_eudr_fca_fragmentation_analyses_total",
        "Total landscape fragmentation analyses performed",
        labelnames=["level"],
    )

    # 7. Biomass estimates by source and status
    fca_biomass_estimates_total = _safe_counter(
        "gl_eudr_fca_biomass_estimates_total",
        "Total above-ground biomass estimates performed",
        labelnames=["source", "status"],
    )

    # 8. Compliance reports generated by format and report type
    fca_reports_generated_total = _safe_counter(
        "gl_eudr_fca_reports_generated_total",
        "Total compliance reports generated",
        labelnames=["format", "report_type"],
    )

    # 9. Deforested plots detected by commodity and country
    fca_deforested_plots_total = _safe_counter(
        "gl_eudr_fca_deforested_plots_total",
        "Total plots classified as deforested since EUDR cutoff",
        labelnames=["commodity", "country"],
    )

    # 10. Degraded plots detected by commodity and country
    fca_degraded_plots_total = _safe_counter(
        "gl_eudr_fca_degraded_plots_total",
        "Total plots classified as degraded since EUDR cutoff",
        labelnames=["commodity", "country"],
    )

    # 11. API errors by operation type
    fca_api_errors_total = _safe_counter(
        "gl_eudr_fca_api_errors_total",
        "Total API errors encountered by operation type",
        labelnames=["operation"],
    )

    # 12. Analysis operation duration by operation type
    fca_analysis_duration_seconds = _safe_histogram(
        "gl_eudr_fca_analysis_duration_seconds",
        "Duration of forest cover analysis operations in seconds",
        labelnames=["operation"],
        buckets=(
            0.1, 0.25, 0.5, 1.0, 2.5, 5.0,
            10.0, 30.0, 60.0, 120.0, 300.0,
        ),
    )

    # 13. Batch analysis total duration
    fca_batch_duration_seconds = _safe_histogram(
        "gl_eudr_fca_batch_duration_seconds",
        "Duration of complete batch analysis jobs in seconds",
        buckets=(
            1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0,
            600.0, 1800.0, 3600.0,
        ),
    )

    # 14. Currently active analyses
    fca_active_analyses = _safe_gauge(
        "gl_eudr_fca_active_analyses",
        "Number of forest cover analyses currently in progress",
    )

    # 15. Average canopy density across analyzed plots
    fca_avg_canopy_density = _safe_gauge(
        "gl_eudr_fca_avg_canopy_density",
        "Average canopy density percentage across analyzed plots",
    )

    # 16. Average confidence score across all determinations
    fca_avg_confidence_score = _safe_gauge(
        "gl_eudr_fca_avg_confidence_score",
        "Average confidence score across deforestation-free determinations",
    )

    # 17. Data quality score by source
    fca_data_quality_score = _safe_gauge(
        "gl_eudr_fca_data_quality_score",
        "Data quality score by analysis source (0-100)",
        labelnames=["source"],
    )

    # 18. Total forest area under analysis
    fca_forest_area_ha = _safe_gauge(
        "gl_eudr_fca_forest_area_ha",
        "Total forest area under analysis in hectares",
    )

else:
    # No-op placeholders so callers never need to guard on PROMETHEUS_AVAILABLE
    fca_density_analyses_total = None           # type: ignore[assignment]
    fca_classifications_total = None            # type: ignore[assignment]
    fca_reconstructions_total = None            # type: ignore[assignment]
    fca_verdicts_total = None                   # type: ignore[assignment]
    fca_height_estimates_total = None           # type: ignore[assignment]
    fca_fragmentation_analyses_total = None     # type: ignore[assignment]
    fca_biomass_estimates_total = None          # type: ignore[assignment]
    fca_reports_generated_total = None          # type: ignore[assignment]
    fca_deforested_plots_total = None           # type: ignore[assignment]
    fca_degraded_plots_total = None             # type: ignore[assignment]
    fca_api_errors_total = None                 # type: ignore[assignment]
    fca_analysis_duration_seconds = None        # type: ignore[assignment]
    fca_batch_duration_seconds = None           # type: ignore[assignment]
    fca_active_analyses = None                  # type: ignore[assignment]
    fca_avg_canopy_density = None               # type: ignore[assignment]
    fca_avg_confidence_score = None             # type: ignore[assignment]
    fca_data_quality_score = None               # type: ignore[assignment]
    fca_forest_area_ha = None                   # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def record_density_analysis(method: str, status: str) -> None:
    """Record a canopy density analysis event.

    Args:
        method: Density mapping method used (e.g.,
            'spectral_unmixing', 'ndvi_regression', 'dimidiation',
            'sub_pixel_detection').
        status: Analysis status ('success', 'error', 'cached').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    fca_density_analyses_total.labels(method=method, status=status).inc()


def record_classification(forest_type: str, method: str) -> None:
    """Record a forest type classification event.

    Args:
        forest_type: Classified forest type (e.g., 'primary_tropical',
            'mangrove', 'plantation', 'non_forest').
        method: Classification method used (e.g., 'spectral_signature',
            'phenological', 'ensemble').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    fca_classifications_total.labels(
        forest_type=forest_type, method=method,
    ).inc()


def record_reconstruction(status: str) -> None:
    """Record a historical cover reconstruction event.

    Args:
        status: Reconstruction status ('success', 'error',
            'partial', 'cached').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    fca_reconstructions_total.labels(status=status).inc()


def record_verdict(verdict: str, commodity: str) -> None:
    """Record a deforestation-free verification verdict event.

    Args:
        verdict: Verdict issued ('deforestation_free', 'deforested',
            'degraded', 'inconclusive').
        commodity: EUDR commodity (e.g., 'soya', 'oil_palm').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    fca_verdicts_total.labels(verdict=verdict, commodity=commodity).inc()


def record_height_estimate(source: str, status: str) -> None:
    """Record a canopy height estimation event.

    Args:
        source: Height data source (e.g., 'gedi_l2a',
            'icesat2_atl08', 'sentinel2_texture').
        status: Estimation status ('success', 'error', 'cached').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    fca_height_estimates_total.labels(source=source, status=status).inc()


def record_fragmentation_analysis(level: str) -> None:
    """Record a landscape fragmentation analysis event.

    Args:
        level: Fragmentation level result ('intact',
            'slightly_fragmented', 'moderately_fragmented',
            'highly_fragmented', 'severely_fragmented').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    fca_fragmentation_analyses_total.labels(level=level).inc()


def record_biomass_estimate(source: str, status: str) -> None:
    """Record an above-ground biomass estimation event.

    Args:
        source: Biomass data source (e.g., 'esa_cci', 'gedi_l4a',
            'sar_regression', 'ndvi_allometric').
        status: Estimation status ('success', 'error', 'cached').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    fca_biomass_estimates_total.labels(source=source, status=status).inc()


def record_report_generated(format: str, report_type: str) -> None:
    """Record a compliance report generation event.

    Args:
        format: Report format ('json', 'pdf', 'csv', 'eudr_xml').
        report_type: Type of report ('full', 'summary', 'compliance',
            'evidence').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    fca_reports_generated_total.labels(
        format=format, report_type=report_type,
    ).inc()


def record_deforested_plot(commodity: str, country: str) -> None:
    """Record a plot classified as deforested since EUDR cutoff.

    Args:
        commodity: EUDR commodity affected.
        country: ISO 3166-1 alpha-2 country code.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    fca_deforested_plots_total.labels(
        commodity=commodity, country=country,
    ).inc()


def record_degraded_plot(commodity: str, country: str) -> None:
    """Record a plot classified as degraded since EUDR cutoff.

    Args:
        commodity: EUDR commodity affected.
        country: ISO 3166-1 alpha-2 country code.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    fca_degraded_plots_total.labels(
        commodity=commodity, country=country,
    ).inc()


def observe_analysis_duration(operation: str, seconds: float) -> None:
    """Record the duration of a forest cover analysis operation.

    Args:
        operation: Type of analysis operation (e.g.,
            'density_analysis', 'classification', 'reconstruction',
            'verdict', 'height_estimate', 'fragmentation_analysis',
            'biomass_estimate', 'report_generation', 'batch_analyze').
        seconds: Operation wall-clock time in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    fca_analysis_duration_seconds.labels(operation=operation).observe(seconds)


def observe_batch_duration(seconds: float) -> None:
    """Record the duration of a complete batch analysis job.

    Args:
        seconds: Total batch job wall-clock time in seconds.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    fca_batch_duration_seconds.observe(seconds)


def set_active_analyses(count: int) -> None:
    """Set the gauge for number of analyses currently in progress.

    Args:
        count: Number of active analyses. Must be >= 0.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    fca_active_analyses.set(count)


def set_avg_canopy_density(density: float) -> None:
    """Set the gauge for average canopy density across analyzed plots.

    Args:
        density: Average canopy density percentage (0-100).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    fca_avg_canopy_density.set(density)


def set_avg_confidence_score(score: float) -> None:
    """Set the gauge for average confidence across determinations.

    Args:
        score: Average confidence score (0.0-1.0).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    fca_avg_confidence_score.set(score)


def record_api_error(operation: str) -> None:
    """Record an API error event by operation type.

    Args:
        operation: Type of operation that failed (e.g.,
            'density_analysis', 'classification', 'reconstruction',
            'verdict', 'height_estimate', 'fragmentation_analysis',
            'biomass_estimate', 'report_generation', 'batch_analyze').
    """
    if not PROMETHEUS_AVAILABLE:
        return
    fca_api_errors_total.labels(operation=operation).inc()


def set_data_quality_score(source: str, score: float) -> None:
    """Set the data quality score gauge for an analysis source.

    Args:
        source: Analysis source identifier (e.g., 'hansen_gfc',
            'gedi', 'esa_cci', 'sentinel_2').
        score: Data quality score (0-100).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    fca_data_quality_score.labels(source=source).set(score)


def set_forest_area_ha(area_ha: float) -> None:
    """Set the gauge for total forest area under analysis.

    Args:
        area_ha: Total forest area in hectares.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    fca_forest_area_ha.set(area_ha)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "PROMETHEUS_AVAILABLE",
    # Metric objects
    "fca_density_analyses_total",
    "fca_classifications_total",
    "fca_reconstructions_total",
    "fca_verdicts_total",
    "fca_height_estimates_total",
    "fca_fragmentation_analyses_total",
    "fca_biomass_estimates_total",
    "fca_reports_generated_total",
    "fca_deforested_plots_total",
    "fca_degraded_plots_total",
    "fca_api_errors_total",
    "fca_analysis_duration_seconds",
    "fca_batch_duration_seconds",
    "fca_active_analyses",
    "fca_avg_canopy_density",
    "fca_avg_confidence_score",
    "fca_data_quality_score",
    "fca_forest_area_ha",
    # Helper functions
    "record_density_analysis",
    "record_classification",
    "record_reconstruction",
    "record_verdict",
    "record_height_estimate",
    "record_fragmentation_analysis",
    "record_biomass_estimate",
    "record_report_generated",
    "record_deforested_plot",
    "record_degraded_plot",
    "observe_analysis_duration",
    "observe_batch_duration",
    "set_active_analyses",
    "set_avg_canopy_density",
    "set_avg_confidence_score",
    "record_api_error",
    "set_data_quality_score",
    "set_forest_area_ha",
]
