# -*- coding: utf-8 -*-
"""
Prometheus Metrics - AGENT-DATA-020: Climate Hazard Connector

12 Prometheus metrics for climate hazard connector service monitoring with
graceful fallback when prometheus_client is not installed.

All metric names use the ``gl_chc_`` prefix (GreenLang Climate Hazard
Connector) for consistent identification in Prometheus queries, Grafana
dashboards, and alerting rules across the GreenLang platform.

Metrics:
    1.  gl_chc_hazard_data_ingested_total        (Counter,   labels: hazard_type, source)
    2.  gl_chc_risk_indices_calculated_total      (Counter,   labels: hazard_type, scenario)
    3.  gl_chc_scenario_projections_total         (Counter,   labels: scenario, time_horizon)
    4.  gl_chc_exposure_assessments_total         (Counter,   labels: asset_type, hazard_type)
    5.  gl_chc_vulnerability_scores_total         (Counter,   labels: sector, hazard_type)
    6.  gl_chc_reports_generated_total            (Counter,   labels: report_type, format)
    7.  gl_chc_pipeline_runs_total                (Counter,   labels: pipeline_stage, status)
    8.  gl_chc_active_sources                     (Gauge)
    9.  gl_chc_active_assets                      (Gauge)
    10. gl_chc_high_risk_locations                (Gauge)
    11. gl_chc_ingestion_duration_seconds         (Histogram, labels: source, buckets: ingestion-scale)
    12. gl_chc_pipeline_duration_seconds          (Histogram, labels: pipeline_stage, buckets: pipeline-scale)

Label Values Reference:
    hazard_type:
        flood, drought, wildfire, heat_wave, cold_wave, storm,
        sea_level_rise, tropical_cyclone, landslide, water_stress,
        precipitation_change, temperature_change, compound.
    source:
        noaa, copernicus, world_bank, nasa, ipcc, national_agency,
        satellite, ground_station, model_output, custom.
    scenario:
        rcp26, rcp45, rcp60, rcp85, ssp126, ssp245, ssp370,
        ssp585, historical, baseline.
    time_horizon:
        2030, 2040, 2050, 2070, 2100, short_term, medium_term,
        long_term.
    asset_type:
        facility, warehouse, office, data_center, factory,
        supply_chain_node, transport_hub, port, mine, farm,
        renewable_installation, portfolio.
    sector:
        energy, manufacturing, agriculture, real_estate, transport,
        water, financial, healthcare, technology, mining, forestry.
    report_type:
        tcfd, csrd, eu_taxonomy, physical_risk, transition_risk,
        portfolio_summary, hotspot_analysis, compliance_summary.
    format:
        json, html, pdf, csv, markdown, xml.
    pipeline_stage:
        ingestion, risk_calculation, scenario_projection,
        exposure_assessment, vulnerability_scoring, reporting,
        full_pipeline.
    status:
        success, failure, partial, timeout.

Example:
    >>> from greenlang.climate_hazard.metrics import (
    ...     record_ingestion,
    ...     record_risk_calculation,
    ...     set_active_sources,
    ... )
    >>> record_ingestion("flood", "noaa")
    >>> record_risk_calculation("flood", "ssp245")
    >>> set_active_sources(12)

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-020 Climate Hazard Connector (GL-DATA-GEO-002)
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
        "prometheus_client not installed; climate hazard connector metrics disabled"
    )

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    # 1. Hazard data ingestion events by hazard type and data source
    chc_hazard_data_ingested_total = Counter(
        "gl_chc_hazard_data_ingested_total",
        "Total climate hazard data records ingested by the connector",
        labelnames=["hazard_type", "source"],
    )

    # 2. Risk index calculations by hazard type and scenario
    chc_risk_indices_calculated_total = Counter(
        "gl_chc_risk_indices_calculated_total",
        "Total climate risk indices calculated by the connector",
        labelnames=["hazard_type", "scenario"],
    )

    # 3. Scenario projections by scenario pathway and time horizon
    chc_scenario_projections_total = Counter(
        "gl_chc_scenario_projections_total",
        "Total climate scenario projections performed by the connector",
        labelnames=["scenario", "time_horizon"],
    )

    # 4. Exposure assessments by asset type and hazard type
    chc_exposure_assessments_total = Counter(
        "gl_chc_exposure_assessments_total",
        "Total exposure assessments performed by the connector",
        labelnames=["asset_type", "hazard_type"],
    )

    # 5. Vulnerability scores by sector and hazard type
    chc_vulnerability_scores_total = Counter(
        "gl_chc_vulnerability_scores_total",
        "Total vulnerability scores computed by the connector",
        labelnames=["sector", "hazard_type"],
    )

    # 6. Reports generated by report type and output format
    chc_reports_generated_total = Counter(
        "gl_chc_reports_generated_total",
        "Total compliance and analysis reports generated by the connector",
        labelnames=["report_type", "format"],
    )

    # 7. Pipeline runs by pipeline stage and completion status
    chc_pipeline_runs_total = Counter(
        "gl_chc_pipeline_runs_total",
        "Total pipeline stage executions by the climate hazard connector",
        labelnames=["pipeline_stage", "status"],
    )

    # 8. Current number of active hazard data sources
    chc_active_sources = Gauge(
        "gl_chc_active_sources",
        "Current number of active climate hazard data sources registered",
    )

    # 9. Current number of active assets under monitoring
    chc_active_assets = Gauge(
        "gl_chc_active_assets",
        "Current number of active assets registered for climate hazard monitoring",
    )

    # 10. Current number of high-risk locations identified
    chc_high_risk_locations = Gauge(
        "gl_chc_high_risk_locations",
        "Current number of locations classified as high climate risk",
    )

    # 11. Ingestion duration by data source
    chc_ingestion_duration_seconds = Histogram(
        "gl_chc_ingestion_duration_seconds",
        "Duration of climate hazard data ingestion operations in seconds",
        labelnames=["source"],
        buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60, 120),
    )

    # 12. Pipeline stage duration by pipeline stage
    chc_pipeline_duration_seconds = Histogram(
        "gl_chc_pipeline_duration_seconds",
        "Duration of climate hazard pipeline stage executions in seconds",
        labelnames=["pipeline_stage"],
        buckets=(0.1, 0.5, 1, 5, 10, 30, 60, 120, 300),
    )

else:
    # No-op placeholders so callers never need to guard on PROMETHEUS_AVAILABLE
    chc_hazard_data_ingested_total = None       # type: ignore[assignment]
    chc_risk_indices_calculated_total = None     # type: ignore[assignment]
    chc_scenario_projections_total = None        # type: ignore[assignment]
    chc_exposure_assessments_total = None        # type: ignore[assignment]
    chc_vulnerability_scores_total = None        # type: ignore[assignment]
    chc_reports_generated_total = None           # type: ignore[assignment]
    chc_pipeline_runs_total = None               # type: ignore[assignment]
    chc_active_sources = None                    # type: ignore[assignment]
    chc_active_assets = None                     # type: ignore[assignment]
    chc_high_risk_locations = None               # type: ignore[assignment]
    chc_ingestion_duration_seconds = None        # type: ignore[assignment]
    chc_pipeline_duration_seconds = None         # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Helper functions (safe to call even without prometheus_client)
# ---------------------------------------------------------------------------


def record_ingestion(hazard_type: str, source: str) -> None:
    """Record a climate hazard data ingestion event.

    Args:
        hazard_type: Type of climate hazard ingested
            (flood, drought, wildfire, heat_wave, cold_wave, storm,
            sea_level_rise, tropical_cyclone, landslide, water_stress,
            precipitation_change, temperature_change, compound).
        source: Data source provider
            (noaa, copernicus, world_bank, nasa, ipcc, national_agency,
            satellite, ground_station, model_output, custom).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    chc_hazard_data_ingested_total.labels(
        hazard_type=hazard_type,
        source=source,
    ).inc()


def record_risk_calculation(hazard_type: str, scenario: str) -> None:
    """Record a climate risk index calculation event.

    Args:
        hazard_type: Type of climate hazard for which risk was calculated
            (flood, drought, wildfire, heat_wave, cold_wave, storm,
            sea_level_rise, tropical_cyclone, landslide, water_stress,
            precipitation_change, temperature_change, compound).
        scenario: Climate scenario pathway used for calculation
            (rcp26, rcp45, rcp60, rcp85, ssp126, ssp245, ssp370,
            ssp585, historical, baseline).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    chc_risk_indices_calculated_total.labels(
        hazard_type=hazard_type,
        scenario=scenario,
    ).inc()


def record_projection(scenario: str, time_horizon: str) -> None:
    """Record a climate scenario projection event.

    Args:
        scenario: Climate scenario pathway used for projection
            (rcp26, rcp45, rcp60, rcp85, ssp126, ssp245, ssp370,
            ssp585, historical, baseline).
        time_horizon: Target time horizon of the projection
            (2030, 2040, 2050, 2070, 2100, short_term, medium_term,
            long_term).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    chc_scenario_projections_total.labels(
        scenario=scenario,
        time_horizon=time_horizon,
    ).inc()


def record_exposure(asset_type: str, hazard_type: str) -> None:
    """Record an exposure assessment event.

    Args:
        asset_type: Type of asset assessed for exposure
            (facility, warehouse, office, data_center, factory,
            supply_chain_node, transport_hub, port, mine, farm,
            renewable_installation, portfolio).
        hazard_type: Type of climate hazard assessed against
            (flood, drought, wildfire, heat_wave, cold_wave, storm,
            sea_level_rise, tropical_cyclone, landslide, water_stress,
            precipitation_change, temperature_change, compound).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    chc_exposure_assessments_total.labels(
        asset_type=asset_type,
        hazard_type=hazard_type,
    ).inc()


def record_vulnerability(sector: str, hazard_type: str) -> None:
    """Record a vulnerability scoring event.

    Args:
        sector: Economic sector for which vulnerability was scored
            (energy, manufacturing, agriculture, real_estate, transport,
            water, financial, healthcare, technology, mining, forestry).
        hazard_type: Type of climate hazard assessed
            (flood, drought, wildfire, heat_wave, cold_wave, storm,
            sea_level_rise, tropical_cyclone, landslide, water_stress,
            precipitation_change, temperature_change, compound).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    chc_vulnerability_scores_total.labels(
        sector=sector,
        hazard_type=hazard_type,
    ).inc()


def record_report(report_type: str, format: str) -> None:
    """Record a compliance or analysis report generation event.

    Args:
        report_type: Type of report generated
            (tcfd, csrd, eu_taxonomy, physical_risk, transition_risk,
            portfolio_summary, hotspot_analysis, compliance_summary).
        format: Output format of the generated report
            (json, html, pdf, csv, markdown, xml).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    chc_reports_generated_total.labels(
        report_type=report_type,
        format=format,
    ).inc()


def record_pipeline(pipeline_stage: str, status: str) -> None:
    """Record a pipeline stage execution event.

    Args:
        pipeline_stage: Stage of the pipeline that executed
            (ingestion, risk_calculation, scenario_projection,
            exposure_assessment, vulnerability_scoring, reporting,
            full_pipeline).
        status: Completion status of the pipeline stage
            (success, failure, partial, timeout).
    """
    if not PROMETHEUS_AVAILABLE:
        return
    chc_pipeline_runs_total.labels(
        pipeline_stage=pipeline_stage,
        status=status,
    ).inc()


def set_active_sources(count: int) -> None:
    """Set the gauge for current number of active hazard data sources.

    This is an absolute set (not an increment) so the caller is
    responsible for computing the correct current count.

    Args:
        count: Number of active climate hazard data sources currently
            registered. Must be >= 0.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    chc_active_sources.set(count)


def set_active_assets(count: int) -> None:
    """Set the gauge for current number of active monitored assets.

    This is an absolute set (not an increment) so the caller is
    responsible for computing the correct current count.

    Args:
        count: Number of active assets currently registered for
            climate hazard monitoring. Must be >= 0.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    chc_active_assets.set(count)


def set_high_risk(count: int) -> None:
    """Set the gauge for current number of high-risk locations.

    This is an absolute set (not an increment) so the caller is
    responsible for computing the correct current count.

    Args:
        count: Number of locations currently classified as high
            climate risk. Must be >= 0.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    chc_high_risk_locations.set(count)


def observe_ingestion_duration(source: str, seconds: float) -> None:
    """Record the duration of a hazard data ingestion operation.

    Args:
        source: Data source provider for the ingestion
            (noaa, copernicus, world_bank, nasa, ipcc, national_agency,
            satellite, ground_station, model_output, custom).
        seconds: Ingestion wall-clock time in seconds.
            Buckets: 0.1, 0.5, 1, 2, 5, 10, 30, 60, 120.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    chc_ingestion_duration_seconds.labels(
        source=source,
    ).observe(seconds)


def observe_pipeline_duration(pipeline_stage: str, seconds: float) -> None:
    """Record the duration of a pipeline stage execution.

    Args:
        pipeline_stage: Stage of the pipeline that was measured
            (ingestion, risk_calculation, scenario_projection,
            exposure_assessment, vulnerability_scoring, reporting,
            full_pipeline).
        seconds: Pipeline stage wall-clock time in seconds.
            Buckets: 0.1, 0.5, 1, 5, 10, 30, 60, 120, 300.
    """
    if not PROMETHEUS_AVAILABLE:
        return
    chc_pipeline_duration_seconds.labels(
        pipeline_stage=pipeline_stage,
    ).observe(seconds)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "PROMETHEUS_AVAILABLE",
    # Metric objects
    "chc_hazard_data_ingested_total",
    "chc_risk_indices_calculated_total",
    "chc_scenario_projections_total",
    "chc_exposure_assessments_total",
    "chc_vulnerability_scores_total",
    "chc_reports_generated_total",
    "chc_pipeline_runs_total",
    "chc_active_sources",
    "chc_active_assets",
    "chc_high_risk_locations",
    "chc_ingestion_duration_seconds",
    "chc_pipeline_duration_seconds",
    # Helper functions
    "record_ingestion",
    "record_risk_calculation",
    "record_projection",
    "record_exposure",
    "record_vulnerability",
    "record_report",
    "record_pipeline",
    "set_active_sources",
    "set_active_assets",
    "set_high_risk",
    "observe_ingestion_duration",
    "observe_pipeline_duration",
]
