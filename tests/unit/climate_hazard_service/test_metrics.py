# -*- coding: utf-8 -*-
"""
Unit tests for Climate Hazard Connector Prometheus metrics module.

Tests all 12 Prometheus metrics (gl_chc_ prefix), 12 helper functions,
graceful fallback when prometheus_client is not installed, and correct
metric types (Counter, Gauge, Histogram).

AGENT-DATA-020: Climate Hazard Connector
Target: 85%+ coverage of greenlang.climate_hazard.metrics
"""

from __future__ import annotations

import importlib
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# Import the module under test
# =============================================================================

from greenlang.climate_hazard import metrics as metrics_mod
from greenlang.climate_hazard.metrics import (
    PROMETHEUS_AVAILABLE,
    chc_active_assets,
    chc_active_sources,
    chc_exposure_assessments_total,
    chc_hazard_data_ingested_total,
    chc_high_risk_locations,
    chc_ingestion_duration_seconds,
    chc_pipeline_duration_seconds,
    chc_pipeline_runs_total,
    chc_reports_generated_total,
    chc_risk_indices_calculated_total,
    chc_scenario_projections_total,
    chc_vulnerability_scores_total,
    observe_ingestion_duration,
    observe_pipeline_duration,
    record_exposure,
    record_ingestion,
    record_pipeline,
    record_projection,
    record_report,
    record_risk_calculation,
    record_vulnerability,
    set_active_assets,
    set_active_sources,
    set_high_risk,
)


# =============================================================================
# PROMETHEUS_AVAILABLE flag
# =============================================================================


class TestPrometheusAvailable:
    """Test the PROMETHEUS_AVAILABLE flag and metric definitions."""

    def test_prometheus_available_is_bool(self) -> None:
        assert isinstance(PROMETHEUS_AVAILABLE, bool)

    def test_prometheus_available_true_when_installed(self) -> None:
        # Since we import prometheus_client in tests, it should be available
        assert PROMETHEUS_AVAILABLE is True


# =============================================================================
# Metric object types (when prometheus_client is available)
# =============================================================================


class TestMetricObjectTypes:
    """Verify that metric objects are the correct Prometheus types."""

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_hazard_data_ingested_is_counter(self) -> None:
        from prometheus_client import Counter
        assert isinstance(chc_hazard_data_ingested_total, Counter)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_risk_indices_calculated_is_counter(self) -> None:
        from prometheus_client import Counter
        assert isinstance(chc_risk_indices_calculated_total, Counter)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_scenario_projections_is_counter(self) -> None:
        from prometheus_client import Counter
        assert isinstance(chc_scenario_projections_total, Counter)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_exposure_assessments_is_counter(self) -> None:
        from prometheus_client import Counter
        assert isinstance(chc_exposure_assessments_total, Counter)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_vulnerability_scores_is_counter(self) -> None:
        from prometheus_client import Counter
        assert isinstance(chc_vulnerability_scores_total, Counter)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_reports_generated_is_counter(self) -> None:
        from prometheus_client import Counter
        assert isinstance(chc_reports_generated_total, Counter)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_pipeline_runs_is_counter(self) -> None:
        from prometheus_client import Counter
        assert isinstance(chc_pipeline_runs_total, Counter)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_active_sources_is_gauge(self) -> None:
        from prometheus_client import Gauge
        assert isinstance(chc_active_sources, Gauge)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_active_assets_is_gauge(self) -> None:
        from prometheus_client import Gauge
        assert isinstance(chc_active_assets, Gauge)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_high_risk_locations_is_gauge(self) -> None:
        from prometheus_client import Gauge
        assert isinstance(chc_high_risk_locations, Gauge)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_ingestion_duration_is_histogram(self) -> None:
        from prometheus_client import Histogram
        assert isinstance(chc_ingestion_duration_seconds, Histogram)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_pipeline_duration_is_histogram(self) -> None:
        from prometheus_client import Histogram
        assert isinstance(chc_pipeline_duration_seconds, Histogram)


# =============================================================================
# Metric naming
# =============================================================================


class TestMetricNaming:
    """Verify all metric names use the gl_chc_ prefix.

    Note: prometheus_client Counter._name strips the ``_total`` suffix
    internally, so we check that the name starts with ``gl_chc_`` and
    contains the expected base name.
    """

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_hazard_data_ingested_name(self) -> None:
        assert chc_hazard_data_ingested_total._name.startswith("gl_chc_")
        assert "hazard_data_ingested" in chc_hazard_data_ingested_total._name

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_risk_indices_calculated_name(self) -> None:
        assert chc_risk_indices_calculated_total._name.startswith("gl_chc_")
        assert "risk_indices_calculated" in chc_risk_indices_calculated_total._name

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_scenario_projections_name(self) -> None:
        assert chc_scenario_projections_total._name.startswith("gl_chc_")
        assert "scenario_projections" in chc_scenario_projections_total._name

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_exposure_assessments_name(self) -> None:
        assert chc_exposure_assessments_total._name.startswith("gl_chc_")
        assert "exposure_assessments" in chc_exposure_assessments_total._name

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_vulnerability_scores_name(self) -> None:
        assert chc_vulnerability_scores_total._name.startswith("gl_chc_")
        assert "vulnerability_scores" in chc_vulnerability_scores_total._name

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_reports_generated_name(self) -> None:
        assert chc_reports_generated_total._name.startswith("gl_chc_")
        assert "reports_generated" in chc_reports_generated_total._name

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_pipeline_runs_name(self) -> None:
        assert chc_pipeline_runs_total._name.startswith("gl_chc_")
        assert "pipeline_runs" in chc_pipeline_runs_total._name

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_active_sources_name(self) -> None:
        assert chc_active_sources._name == "gl_chc_active_sources"

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_active_assets_name(self) -> None:
        assert chc_active_assets._name == "gl_chc_active_assets"

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_high_risk_locations_name(self) -> None:
        assert chc_high_risk_locations._name == "gl_chc_high_risk_locations"

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_ingestion_duration_name(self) -> None:
        assert chc_ingestion_duration_seconds._name == "gl_chc_ingestion_duration_seconds"

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_pipeline_duration_name(self) -> None:
        assert chc_pipeline_duration_seconds._name == "gl_chc_pipeline_duration_seconds"


# =============================================================================
# Metric labels
# =============================================================================


class TestMetricLabels:
    """Verify metric label names are correct."""

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_hazard_data_ingested_labels(self) -> None:
        assert chc_hazard_data_ingested_total._labelnames == ("hazard_type", "source")

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_risk_indices_calculated_labels(self) -> None:
        assert chc_risk_indices_calculated_total._labelnames == ("hazard_type", "scenario")

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_scenario_projections_labels(self) -> None:
        assert chc_scenario_projections_total._labelnames == ("scenario", "time_horizon")

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_exposure_assessments_labels(self) -> None:
        assert chc_exposure_assessments_total._labelnames == ("asset_type", "hazard_type")

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_vulnerability_scores_labels(self) -> None:
        assert chc_vulnerability_scores_total._labelnames == ("sector", "hazard_type")

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_reports_generated_labels(self) -> None:
        assert chc_reports_generated_total._labelnames == ("report_type", "format")

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_pipeline_runs_labels(self) -> None:
        assert chc_pipeline_runs_total._labelnames == ("pipeline_stage", "status")

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_active_sources_no_labels(self) -> None:
        assert chc_active_sources._labelnames == ()

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_active_assets_no_labels(self) -> None:
        assert chc_active_assets._labelnames == ()

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_high_risk_locations_no_labels(self) -> None:
        assert chc_high_risk_locations._labelnames == ()

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_ingestion_duration_labels(self) -> None:
        assert chc_ingestion_duration_seconds._labelnames == ("source",)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_pipeline_duration_labels(self) -> None:
        assert chc_pipeline_duration_seconds._labelnames == ("pipeline_stage",)


# =============================================================================
# Helper functions (with prometheus_client)
# =============================================================================


class TestHelperFunctionsWithPrometheus:
    """Test all 12 helper functions when prometheus_client is available."""

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_record_ingestion_does_not_raise(self) -> None:
        record_ingestion("flood", "noaa")

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_record_ingestion_various_types(self) -> None:
        for hazard in ["drought", "wildfire", "heat_wave", "storm"]:
            record_ingestion(hazard, "copernicus")

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_record_risk_calculation_does_not_raise(self) -> None:
        record_risk_calculation("flood", "ssp245")

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_record_risk_calculation_various_scenarios(self) -> None:
        for scenario in ["rcp26", "rcp45", "ssp126", "ssp585"]:
            record_risk_calculation("drought", scenario)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_record_projection_does_not_raise(self) -> None:
        record_projection("ssp245", "2050")

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_record_projection_various_horizons(self) -> None:
        for horizon in ["2030", "2050", "2100", "short_term", "long_term"]:
            record_projection("ssp585", horizon)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_record_exposure_does_not_raise(self) -> None:
        record_exposure("facility", "flood")

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_record_exposure_various_assets(self) -> None:
        for asset_type in ["warehouse", "data_center", "port", "farm"]:
            record_exposure(asset_type, "wildfire")

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_record_vulnerability_does_not_raise(self) -> None:
        record_vulnerability("energy", "drought")

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_record_vulnerability_various_sectors(self) -> None:
        for sector in ["manufacturing", "agriculture", "real_estate", "transport"]:
            record_vulnerability(sector, "heat_wave")

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_record_report_does_not_raise(self) -> None:
        record_report("tcfd", "json")

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_record_report_various_formats(self) -> None:
        for fmt in ["json", "html", "pdf", "csv", "markdown"]:
            record_report("csrd", fmt)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_record_pipeline_does_not_raise(self) -> None:
        record_pipeline("ingestion", "success")

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_record_pipeline_various_stages(self) -> None:
        for stage in [
            "ingestion", "risk_calculation", "scenario_projection",
            "exposure_assessment", "vulnerability_scoring",
            "reporting", "full_pipeline",
        ]:
            record_pipeline(stage, "success")

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_record_pipeline_various_statuses(self) -> None:
        for status in ["success", "failure", "partial", "timeout"]:
            record_pipeline("full_pipeline", status)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_set_active_sources_does_not_raise(self) -> None:
        set_active_sources(12)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_set_active_sources_zero(self) -> None:
        set_active_sources(0)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_set_active_assets_does_not_raise(self) -> None:
        set_active_assets(500)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_set_active_assets_zero(self) -> None:
        set_active_assets(0)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_set_high_risk_does_not_raise(self) -> None:
        set_high_risk(42)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_set_high_risk_zero(self) -> None:
        set_high_risk(0)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_observe_ingestion_duration_does_not_raise(self) -> None:
        observe_ingestion_duration("noaa", 2.5)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_observe_ingestion_duration_various_sources(self) -> None:
        for source in ["copernicus", "world_bank", "nasa", "ipcc"]:
            observe_ingestion_duration(source, 1.0)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_observe_pipeline_duration_does_not_raise(self) -> None:
        observe_pipeline_duration("ingestion", 10.0)

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_observe_pipeline_duration_various_stages(self) -> None:
        for stage in [
            "risk_calculation", "scenario_projection",
            "exposure_assessment", "reporting",
        ]:
            observe_pipeline_duration(stage, 5.0)


# =============================================================================
# Helper functions without prometheus_client (mocked)
# =============================================================================


class TestHelperFunctionsWithoutPrometheus:
    """Test that all helpers are no-ops when PROMETHEUS_AVAILABLE is False."""

    def test_record_ingestion_noop(self) -> None:
        with patch.object(metrics_mod, "PROMETHEUS_AVAILABLE", False):
            # Should return None without error
            result = metrics_mod.record_ingestion("flood", "noaa")
            assert result is None

    def test_record_risk_calculation_noop(self) -> None:
        with patch.object(metrics_mod, "PROMETHEUS_AVAILABLE", False):
            result = metrics_mod.record_risk_calculation("flood", "ssp245")
            assert result is None

    def test_record_projection_noop(self) -> None:
        with patch.object(metrics_mod, "PROMETHEUS_AVAILABLE", False):
            result = metrics_mod.record_projection("ssp245", "2050")
            assert result is None

    def test_record_exposure_noop(self) -> None:
        with patch.object(metrics_mod, "PROMETHEUS_AVAILABLE", False):
            result = metrics_mod.record_exposure("facility", "flood")
            assert result is None

    def test_record_vulnerability_noop(self) -> None:
        with patch.object(metrics_mod, "PROMETHEUS_AVAILABLE", False):
            result = metrics_mod.record_vulnerability("energy", "drought")
            assert result is None

    def test_record_report_noop(self) -> None:
        with patch.object(metrics_mod, "PROMETHEUS_AVAILABLE", False):
            result = metrics_mod.record_report("tcfd", "json")
            assert result is None

    def test_record_pipeline_noop(self) -> None:
        with patch.object(metrics_mod, "PROMETHEUS_AVAILABLE", False):
            result = metrics_mod.record_pipeline("ingestion", "success")
            assert result is None

    def test_set_active_sources_noop(self) -> None:
        with patch.object(metrics_mod, "PROMETHEUS_AVAILABLE", False):
            result = metrics_mod.set_active_sources(10)
            assert result is None

    def test_set_active_assets_noop(self) -> None:
        with patch.object(metrics_mod, "PROMETHEUS_AVAILABLE", False):
            result = metrics_mod.set_active_assets(100)
            assert result is None

    def test_set_high_risk_noop(self) -> None:
        with patch.object(metrics_mod, "PROMETHEUS_AVAILABLE", False):
            result = metrics_mod.set_high_risk(5)
            assert result is None

    def test_observe_ingestion_duration_noop(self) -> None:
        with patch.object(metrics_mod, "PROMETHEUS_AVAILABLE", False):
            result = metrics_mod.observe_ingestion_duration("noaa", 1.0)
            assert result is None

    def test_observe_pipeline_duration_noop(self) -> None:
        with patch.object(metrics_mod, "PROMETHEUS_AVAILABLE", False):
            result = metrics_mod.observe_pipeline_duration("ingestion", 1.0)
            assert result is None


# =============================================================================
# __all__ exports
# =============================================================================


class TestMetricsExports:
    """Verify __all__ exports contain expected symbols."""

    def test_all_contains_prometheus_available(self) -> None:
        assert "PROMETHEUS_AVAILABLE" in metrics_mod.__all__

    def test_all_contains_all_metric_objects(self) -> None:
        metric_names = [
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
        ]
        for name in metric_names:
            assert name in metrics_mod.__all__, f"{name} missing from __all__"

    def test_all_contains_all_helper_functions(self) -> None:
        helper_names = [
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
        for name in helper_names:
            assert name in metrics_mod.__all__, f"{name} missing from __all__"

    def test_all_has_correct_count(self) -> None:
        # 1 flag + 12 metric objects + 12 helper functions = 25
        assert len(metrics_mod.__all__) == 25


# =============================================================================
# Histogram bucket configuration
# =============================================================================


class TestHistogramBuckets:
    """Verify histogram bucket configurations."""

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_ingestion_duration_buckets(self) -> None:
        expected = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0]
        # _upper_bounds includes +Inf as last element
        actual = list(chc_ingestion_duration_seconds._upper_bounds[:-1])
        assert actual == expected

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_pipeline_duration_buckets(self) -> None:
        expected = [0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0]
        # _upper_bounds includes +Inf as last element
        actual = list(chc_pipeline_duration_seconds._upper_bounds[:-1])
        assert actual == expected


# =============================================================================
# Metric descriptions
# =============================================================================


class TestMetricDescriptions:
    """Verify all metrics have non-empty descriptions."""

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_all_metrics_have_descriptions(self) -> None:
        metric_objects = [
            chc_hazard_data_ingested_total,
            chc_risk_indices_calculated_total,
            chc_scenario_projections_total,
            chc_exposure_assessments_total,
            chc_vulnerability_scores_total,
            chc_reports_generated_total,
            chc_pipeline_runs_total,
            chc_active_sources,
            chc_active_assets,
            chc_high_risk_locations,
            chc_ingestion_duration_seconds,
            chc_pipeline_duration_seconds,
        ]
        for m in metric_objects:
            assert m._documentation, f"Metric {m._name} has no description"

    @pytest.mark.skipif(
        not PROMETHEUS_AVAILABLE, reason="prometheus_client not installed"
    )
    def test_descriptions_mention_relevant_context(self) -> None:
        """All descriptions should reference climate, hazard, pipeline, or connector context."""
        metric_objects = [
            chc_hazard_data_ingested_total,
            chc_risk_indices_calculated_total,
            chc_scenario_projections_total,
            chc_exposure_assessments_total,
            chc_vulnerability_scores_total,
            chc_reports_generated_total,
            chc_pipeline_runs_total,
            chc_active_sources,
            chc_active_assets,
            chc_high_risk_locations,
            chc_ingestion_duration_seconds,
            chc_pipeline_duration_seconds,
        ]
        for m in metric_objects:
            doc = m._documentation.lower()
            assert any(
                kw in doc
                for kw in ("climate", "hazard", "pipeline", "connector", "exposure", "vulnerability")
            ), (
                f"Metric {m._name} description lacks domain context: {doc}"
            )
