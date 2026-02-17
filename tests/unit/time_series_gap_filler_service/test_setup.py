# -*- coding: utf-8 -*-
"""
Unit Tests for TimeSeriesGapFillerService (AGENT-DATA-014)
============================================================

Comprehensive test suite for ``greenlang.time_series_gap_filler.setup``
covering the ``TimeSeriesGapFillerService`` facade, all public methods,
lifecycle, statistics, provenance, configuration helpers, singleton
management, and full end-to-end workflows.

Target: 85+ tests with 85%+ coverage.

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-014 Time Series Gap Filler (GL-DATA-X-017)
"""

from __future__ import annotations

import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from greenlang.time_series_gap_filler.config import (
    TimeSeriesGapFillerConfig,
    get_config,
    set_config,
    reset_config,
)
from greenlang.time_series_gap_filler.setup import (
    TimeSeriesGapFillerService,
    configure_gap_filler,
    get_gap_filler,
    get_router,
    get_service,
    reset_service,
    _compute_hash,
)


# ===================================================================
# Helpers
# ===================================================================


def _make_config(**overrides: Any) -> TimeSeriesGapFillerConfig:
    """Create a TimeSeriesGapFillerConfig with optional overrides."""
    defaults = dict(
        database_url="",
        redis_url="",
    )
    defaults.update(overrides)
    return TimeSeriesGapFillerConfig(**defaults)


def _make_service(**config_overrides: Any) -> TimeSeriesGapFillerService:
    """Create a TimeSeriesGapFillerService with optional config overrides."""
    cfg = _make_config(**config_overrides)
    svc = TimeSeriesGapFillerService(config=cfg)
    svc.startup()
    return svc


def _series_with_gaps() -> List[Optional[float]]:
    """Return a series with interior gaps."""
    return [1.0, 2.0, None, None, 5.0, 6.0, 7.0, None, 9.0, 10.0]


def _series_no_gaps() -> List[float]:
    """Return a complete series (no gaps)."""
    return [float(i) for i in range(1, 11)]


def _series_all_gaps() -> List[Optional[float]]:
    """Return a series that is entirely gaps."""
    return [None] * 10


def _series_leading_gap() -> List[Optional[float]]:
    """Return a series with leading gaps."""
    return [None, None, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]


def _series_trailing_gap() -> List[Optional[float]]:
    """Return a series with trailing gaps."""
    return [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, None, None]


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the module-level singleton before each test."""
    import greenlang.time_series_gap_filler.setup as setup_mod
    setup_mod._service_instance = None
    reset_config()
    yield
    setup_mod._service_instance = None
    reset_config()


@pytest.fixture
def service() -> TimeSeriesGapFillerService:
    """Create a fresh TimeSeriesGapFillerService for each test."""
    return _make_service()


# ===================================================================
# 1. Service Lifecycle Tests
# ===================================================================


class TestServiceLifecycle:
    """Tests for service creation, startup, shutdown, and singleton."""

    def test_service_creates_successfully(self):
        """Service can be created with default config."""
        svc = TimeSeriesGapFillerService()
        assert svc is not None
        assert svc.config is not None
        assert svc._started is False

    def test_service_creates_with_custom_config(self):
        """Service accepts a custom TimeSeriesGapFillerConfig."""
        cfg = _make_config(batch_size=500, max_records=50000)
        svc = TimeSeriesGapFillerService(config=cfg)
        assert svc.config.batch_size == 500
        assert svc.config.max_records == 50000

    def test_startup_sets_started_flag(self):
        """startup() sets the _started flag to True."""
        svc = TimeSeriesGapFillerService()
        assert svc._started is False
        svc.startup()
        assert svc._started is True

    def test_shutdown_clears_started_flag(self, service):
        """shutdown() sets the _started flag to False."""
        assert service._started is True
        service.shutdown()
        assert service._started is False

    def test_get_service_singleton_returns_same_instance(self):
        """get_service() returns the same singleton on repeated calls."""
        svc1 = get_service()
        svc2 = get_service()
        assert svc1 is svc2

    def test_reset_service_creates_new_instance(self):
        """reset_service() replaces the singleton with a new instance."""
        svc1 = get_service()
        svc2 = reset_service()
        assert svc1 is not svc2
        assert svc2._started is True

    def test_service_has_gap_detector_engine(self, service):
        """Service initializes the gap detector engine."""
        assert service._gap_detector is not None

    def test_service_has_frequency_analyzer_engine(self, service):
        """Service initializes the frequency analyzer engine."""
        assert service._frequency_analyzer is not None

    def test_service_has_interpolation_engine(self, service):
        """Service initializes the interpolation engine."""
        assert service._interpolation_engine is not None

    def test_service_has_provenance_tracker(self, service):
        """Service has a provenance tracker."""
        assert service._provenance is not None

    def test_health_check_returns_valid_response(self, service):
        """health_check() returns a dict with required keys."""
        result = service.health_check()
        assert isinstance(result, dict)
        assert "status" in result
        assert "service" in result
        assert "engines" in result
        assert "timestamp" in result

    def test_health_check_status_healthy(self, service):
        """health_check() reports healthy when started."""
        result = service.health_check()
        assert result["status"] == "healthy"

    def test_health_check_status_starting_when_not_started(self):
        """health_check() reports starting when not started."""
        svc = TimeSeriesGapFillerService()
        result = svc.health_check()
        assert result["status"] == "starting"

    def test_get_stats_returns_statistics(self, service):
        """get_stats() returns a dict with counter keys."""
        result = service.get_stats()
        assert isinstance(result, dict)
        assert "total_detections" in result
        assert "total_fills" in result
        assert "total_validations" in result
        assert "total_pipelines" in result
        assert "total_jobs" in result
        assert "timestamp" in result

    def test_get_statistics_alias(self, service):
        """get_statistics() is an alias for get_stats()."""
        stats1 = service.get_stats()
        stats2 = service.get_statistics()
        # Same data (minus timestamp variance)
        assert stats1["total_detections"] == stats2["total_detections"]


# ===================================================================
# 2. Module-Level Function Tests
# ===================================================================


class TestModuleFunctions:
    """Tests for configure_gap_filler, get_gap_filler, get_router."""

    def test_configure_gap_filler_attaches_to_app_state(self):
        """configure_gap_filler(app) sets app.state.gap_filler_service."""
        app = MagicMock()
        app.state = MagicMock()
        app.include_router = MagicMock()
        svc = configure_gap_filler(app)
        assert app.state.gap_filler_service is svc

    def test_get_gap_filler_retrieves_service(self):
        """get_gap_filler(app) retrieves the service from app.state."""
        app = MagicMock()
        app.state = MagicMock()
        app.include_router = MagicMock()
        svc = configure_gap_filler(app)
        retrieved = get_gap_filler(app)
        assert retrieved is svc

    def test_get_gap_filler_returns_none_when_not_configured(self):
        """get_gap_filler() returns None when service not configured."""
        app = MagicMock()
        app.state = MagicMock(spec=[])
        result = get_gap_filler(app)
        assert result is None

    def test_get_router_returns_api_router(self):
        """get_router() returns a FastAPI APIRouter."""
        router = get_router()
        # Should not be None when FastAPI is available
        assert router is not None

    def test_compute_hash_returns_sha256(self):
        """_compute_hash returns a 64-char hex SHA-256 hash."""
        h = _compute_hash({"key": "value"})
        assert isinstance(h, str)
        assert len(h) == 64

    def test_compute_hash_deterministic(self):
        """_compute_hash returns the same hash for same input."""
        data = {"a": 1, "b": 2}
        h1 = _compute_hash(data)
        h2 = _compute_hash(data)
        assert h1 == h2

    def test_compute_hash_key_order_independent(self):
        """_compute_hash sorts keys, so insertion order does not matter."""
        h1 = _compute_hash({"a": 1, "b": 2})
        h2 = _compute_hash({"b": 2, "a": 1})
        assert h1 == h2


# ===================================================================
# 3. Gap Detection Tests
# ===================================================================


class TestGapDetection:
    """Tests for detect_gaps and related methods."""

    def test_detect_gaps_with_gaps(self, service):
        """detect_gaps returns a result dict with gap_count > 0."""
        result = service.detect_gaps(values=_series_with_gaps())
        assert isinstance(result, dict)
        assert "detection_id" in result
        assert result["gap_count"] >= 1
        assert result["total_missing"] >= 1
        assert result["series_length"] == 10

    def test_detect_gaps_no_gaps(self, service):
        """detect_gaps with no gaps returns zero gap_count."""
        result = service.detect_gaps(values=_series_no_gaps())
        assert result["gap_count"] == 0
        assert result["total_missing"] == 0
        assert result["gap_pct"] == 0.0

    def test_detect_gaps_all_gaps(self, service):
        """detect_gaps with all gaps returns full gap count."""
        result = service.detect_gaps(values=_series_all_gaps())
        assert result["total_missing"] == 10

    def test_detect_gaps_leading_gap(self, service):
        """detect_gaps detects leading gaps correctly."""
        result = service.detect_gaps(values=_series_leading_gap())
        assert result["total_missing"] >= 2

    def test_detect_gaps_trailing_gap(self, service):
        """detect_gaps detects trailing gaps correctly."""
        result = service.detect_gaps(values=_series_trailing_gap())
        assert result["total_missing"] >= 2

    def test_detect_gaps_stores_result(self, service):
        """detect_gaps stores the result in the detections store."""
        result = service.detect_gaps(values=_series_with_gaps())
        stored = service.get_detection(result["detection_id"])
        assert stored is not None
        assert stored["detection_id"] == result["detection_id"]

    def test_detect_gaps_increments_stats(self, service):
        """detect_gaps increments total_detections counter."""
        assert service._stats["total_detections"] == 0
        service.detect_gaps(values=_series_with_gaps())
        assert service._stats["total_detections"] == 1

    def test_detect_gaps_has_provenance_hash(self, service):
        """detect_gaps result includes a provenance_hash."""
        result = service.detect_gaps(values=_series_with_gaps())
        assert "provenance_hash" in result
        assert isinstance(result["provenance_hash"], str)
        assert len(result["provenance_hash"]) == 64

    def test_detect_gaps_batch_multiple_series(self, service):
        """detect_gaps_batch processes multiple series."""
        batch = service.detect_gaps_batch(series_list=[
            {"values": _series_with_gaps(), "series_id": "s1"},
            {"values": _series_no_gaps(), "series_id": "s2"},
        ])
        assert batch["count"] == 2
        assert len(batch["results"]) == 2

    def test_detect_batch_router_compatible(self, service):
        """detect_batch (router alias) processes series dicts."""
        result = service.detect_batch(
            series_list=[
                {"name": "ts1", "values": [1, None, 3]},
                {"name": "ts2", "values": [4, 5, 6]},
            ],
        )
        assert result["count"] == 2

    def test_list_detections_returns_list(self, service):
        """list_detections returns a list of stored detections."""
        service.detect_gaps(values=_series_with_gaps())
        service.detect_gaps(values=_series_no_gaps())
        items = service.list_detections()
        assert isinstance(items, list)
        assert len(items) == 2

    def test_list_detections_supports_pagination(self, service):
        """list_detections supports limit and offset."""
        for _ in range(5):
            service.detect_gaps(values=_series_with_gaps())
        page = service.list_detections(limit=2, offset=1)
        assert len(page) == 2

    def test_get_detection_returns_detail(self, service):
        """get_detection returns the detection dict by ID."""
        result = service.detect_gaps(values=_series_with_gaps())
        fetched = service.get_detection(result["detection_id"])
        assert fetched is not None
        assert fetched["detection_id"] == result["detection_id"]

    def test_get_detection_unknown_id_returns_none(self, service):
        """get_detection returns None for an unknown ID."""
        result = service.get_detection("nonexistent-id")
        assert result is None

    def test_get_detections_returns_all(self, service):
        """get_detections returns all stored detections."""
        service.detect_gaps(values=_series_with_gaps())
        service.detect_gaps(values=_series_no_gaps())
        all_dets = service.get_detections()
        assert len(all_dets) == 2

    def test_detect_gaps_single_element(self, service):
        """detect_gaps works with a single-element series."""
        result = service.detect_gaps(values=[1.0])
        assert result["series_length"] == 1

    def test_detect_gaps_empty_series(self, service):
        """detect_gaps works with an empty series."""
        result = service.detect_gaps(values=[])
        assert result["series_length"] == 0


# ===================================================================
# 4. Frequency Analysis Tests
# ===================================================================


class TestFrequencyAnalysis:
    """Tests for analyze_frequency and get_frequency."""

    def test_analyze_frequency_returns_result(self, service):
        """analyze_frequency returns a result dict."""
        ts = [datetime(2025, 1, d, tzinfo=timezone.utc) for d in range(1, 20)]
        result = service.analyze_frequency(timestamps=ts)
        assert isinstance(result, dict)
        assert "frequency_id" in result

    def test_analyze_frequency_stores_result(self, service):
        """analyze_frequency stores the result."""
        ts = [datetime(2025, 1, d, tzinfo=timezone.utc) for d in range(1, 20)]
        result = service.analyze_frequency(timestamps=ts)
        stored = service.get_frequency(result["frequency_id"])
        assert stored is not None

    def test_get_frequency_returns_stored_result(self, service):
        """get_frequency returns the frequency result by ID."""
        ts = [datetime(2025, 1, d, tzinfo=timezone.utc) for d in range(1, 20)]
        result = service.analyze_frequency(timestamps=ts)
        fetched = service.get_frequency(result["frequency_id"])
        assert fetched["frequency_id"] == result["frequency_id"]

    def test_get_frequency_unknown_returns_none(self, service):
        """get_frequency returns None for an unknown ID."""
        result = service.get_frequency("nonexistent-id")
        assert result is None

    def test_analyze_frequency_has_provenance(self, service):
        """analyze_frequency result includes provenance_hash."""
        ts = [datetime(2025, 1, d, tzinfo=timezone.utc) for d in range(1, 20)]
        result = service.analyze_frequency(timestamps=ts)
        assert "provenance_hash" in result


# ===================================================================
# 5. Gap Filling Tests
# ===================================================================


class TestGapFilling:
    """Tests for fill_gaps, get_fill, and undo_fill."""

    def test_fill_gaps_linear_returns_result(self, service):
        """fill_gaps with linear method returns a fill result."""
        result = service.fill_gaps(
            values=_series_with_gaps(),
            method="linear",
        )
        assert isinstance(result, dict)
        assert "fill_id" in result
        assert "filled_values" in result
        assert "method" in result

    def test_fill_gaps_fills_none_values(self, service):
        """fill_gaps replaces None values in the output."""
        values = [1.0, None, 3.0]
        result = service.fill_gaps(values=values, method="linear")
        filled = result["filled_values"]
        assert None not in filled
        assert len(filled) == 3

    def test_fill_gaps_cubic_spline(self, service):
        """fill_gaps with cubic_spline method works."""
        result = service.fill_gaps(
            values=_series_with_gaps(),
            method="cubic_spline",
        )
        assert result["method"] == "cubic_spline"
        assert None not in result["filled_values"]

    def test_fill_gaps_nearest(self, service):
        """fill_gaps with nearest method works."""
        result = service.fill_gaps(
            values=_series_with_gaps(),
            method="nearest",
        )
        assert result["method"] == "nearest"

    def test_fill_gaps_stores_result(self, service):
        """fill_gaps stores the result."""
        result = service.fill_gaps(values=_series_with_gaps())
        stored = service.get_fill(result["fill_id"])
        assert stored is not None

    def test_fill_gaps_increments_stats(self, service):
        """fill_gaps increments total_fills."""
        assert service._stats["total_fills"] == 0
        service.fill_gaps(values=_series_with_gaps())
        assert service._stats["total_fills"] == 1

    def test_fill_gaps_has_provenance(self, service):
        """fill_gaps result includes provenance_hash."""
        result = service.fill_gaps(values=_series_with_gaps())
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_fill_gaps_preserves_original_values(self, service):
        """fill_gaps stores original_values for undo support."""
        values = _series_with_gaps()
        result = service.fill_gaps(values=values)
        assert "original_values" in result
        assert result["original_values"] == values

    def test_fill_gaps_no_gaps(self, service):
        """fill_gaps on a complete series returns it unchanged."""
        values = _series_no_gaps()
        result = service.fill_gaps(values=values)
        assert result["total_missing"] == 0

    def test_get_fill_returns_stored_result(self, service):
        """get_fill returns the fill result by ID."""
        result = service.fill_gaps(values=_series_with_gaps())
        fetched = service.get_fill(result["fill_id"])
        assert fetched is not None
        assert fetched["fill_id"] == result["fill_id"]

    def test_get_fill_unknown_returns_none(self, service):
        """get_fill returns None for an unknown ID."""
        result = service.get_fill("nonexistent-id")
        assert result is None

    def test_undo_fill_restores_original(self, service):
        """undo_fill restores original values."""
        values = [1.0, None, 3.0, 4.0, 5.0]
        result = service.fill_gaps(values=values)
        fill_id = result["fill_id"]

        # Values should be filled
        assert None not in result["filled_values"]

        # Undo
        undone = service.undo_fill(fill_id)
        assert undone is not None
        assert undone.get("undone") is True
        assert undone["filled_values"] == values

    def test_undo_fill_unknown_returns_none(self, service):
        """undo_fill returns None for an unknown fill ID."""
        result = service.undo_fill("nonexistent-id")
        assert result is None

    def test_fill_gaps_with_confidence(self, service):
        """fill_gaps result includes confidence metrics."""
        result = service.fill_gaps(values=_series_with_gaps())
        assert "mean_confidence" in result
        assert "min_confidence" in result
        assert 0.0 <= result["mean_confidence"] <= 1.0

    def test_fill_gaps_processing_time(self, service):
        """fill_gaps result includes processing_time_ms."""
        result = service.fill_gaps(values=_series_with_gaps())
        assert "processing_time_ms" in result
        assert result["processing_time_ms"] >= 0.0


# ===================================================================
# 6. Validation Tests
# ===================================================================


class TestValidation:
    """Tests for validate_fills and get_validation."""

    def test_validate_fills_returns_report(self, service):
        """validate_fills returns a validation result dict."""
        result = service.validate_fills(
            filled_values=[1.0, 2.0, 3.0, 4.0, 5.0],
        )
        assert isinstance(result, dict)
        assert "validation_id" in result
        assert "level" in result

    def test_validate_fills_pass_on_complete(self, service):
        """validate_fills returns pass when no gaps remain."""
        result = service.validate_fills(
            filled_values=[1.0, 2.0, 3.0, 4.0, 5.0],
        )
        assert result["level"] == "pass"
        assert result["continuity_check"] is True

    def test_validate_fills_fail_on_remaining_gaps(self, service):
        """validate_fills returns fail when gaps remain."""
        result = service.validate_fills(
            filled_values=[1.0, None, 3.0, None, 5.0],
        )
        assert result["level"] == "fail"
        assert result["remaining_gaps"] == 2

    def test_validate_fills_stores_result(self, service):
        """validate_fills stores the result."""
        result = service.validate_fills(
            filled_values=[1.0, 2.0, 3.0],
        )
        stored = service.get_validation(result["validation_id"])
        assert stored is not None

    def test_validate_fills_increments_stats(self, service):
        """validate_fills increments total_validations."""
        assert service._stats["total_validations"] == 0
        service.validate_fills(filled_values=[1.0, 2.0])
        assert service._stats["total_validations"] == 1

    def test_get_validation_returns_stored(self, service):
        """get_validation returns the validation by ID."""
        result = service.validate_fills(filled_values=[1.0])
        fetched = service.get_validation(result["validation_id"])
        assert fetched is not None

    def test_get_validation_unknown_returns_none(self, service):
        """get_validation returns None for unknown ID."""
        result = service.get_validation("nonexistent-id")
        assert result is None

    def test_validate_fills_router_calling_convention(self, service):
        """validate_fills works with the router's keyword arguments."""
        result = service.validate_fills(
            original=[1.0, None, 3.0],
            filled=[1.0, 2.0, 3.0],
            fill_indices=[1],
        )
        assert result["level"] == "pass"

    def test_validate_fills_has_provenance(self, service):
        """validate_fills result includes provenance_hash."""
        result = service.validate_fills(filled_values=[1.0, 2.0])
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64


# ===================================================================
# 7. Calendar Tests
# ===================================================================


class TestCalendar:
    """Tests for create_calendar and list_calendars."""

    def test_create_calendar_returns_definition(self, service):
        """create_calendar returns a calendar dict."""
        cal = service.create_calendar(name="test_cal")
        assert isinstance(cal, dict)
        assert "calendar_id" in cal
        assert cal["name"] == "test_cal"

    def test_create_calendar_default_business_days(self, service):
        """create_calendar defaults to Mon-Fri business days."""
        cal = service.create_calendar()
        assert cal["business_days"] == [0, 1, 2, 3, 4]

    def test_create_calendar_custom_holidays(self, service):
        """create_calendar accepts custom holidays."""
        cal = service.create_calendar(
            name="with_holidays",
            holidays=["2026-01-01", "2026-12-25"],
        )
        assert len(cal["holidays"]) == 2

    def test_list_calendars_returns_list(self, service):
        """list_calendars returns a list."""
        service.create_calendar(name="cal_1")
        service.create_calendar(name="cal_2")
        cals = service.list_calendars()
        assert isinstance(cals, list)
        assert len(cals) == 2

    def test_list_calendars_empty_initially(self, service):
        """list_calendars returns empty list when no calendars exist."""
        cals = service.list_calendars()
        assert cals == []


# ===================================================================
# 8. Pipeline Tests
# ===================================================================


class TestPipeline:
    """Tests for run_pipeline."""

    def test_run_pipeline_returns_result(self, service):
        """run_pipeline returns a complete pipeline result."""
        result = service.run_pipeline(values=_series_with_gaps())
        assert isinstance(result, dict)
        assert "pipeline_id" in result
        assert "detection" in result
        assert "fill" in result
        assert "status" in result

    def test_run_pipeline_status_completed(self, service):
        """run_pipeline returns status completed."""
        result = service.run_pipeline(values=_series_with_gaps())
        assert result["status"] == "completed"

    def test_run_pipeline_includes_validation(self, service):
        """run_pipeline includes validation when enabled."""
        result = service.run_pipeline(values=_series_with_gaps())
        assert result["validation"] is not None

    def test_run_pipeline_no_validation_when_disabled(self, service):
        """run_pipeline skips validation when disabled."""
        result = service.run_pipeline(
            values=_series_with_gaps(),
            validate=False,
        )
        assert result["validation"] is None

    def test_run_pipeline_has_provenance(self, service):
        """run_pipeline result includes provenance_hash."""
        result = service.run_pipeline(values=_series_with_gaps())
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_run_pipeline_increments_stats(self, service):
        """run_pipeline increments total_pipelines."""
        assert service._stats["total_pipelines"] == 0
        service.run_pipeline(values=_series_with_gaps())
        assert service._stats["total_pipelines"] == 1

    def test_run_pipeline_end_to_end(self, service):
        """run_pipeline end-to-end detects and fills gaps."""
        values = [1.0, None, 3.0, None, 5.0]
        result = service.run_pipeline(values=values)

        detection = result["detection"]
        fill = result["fill"]
        validation = result["validation"]

        assert detection["total_missing"] >= 2
        assert None not in fill["filled_values"]
        assert validation["level"] == "pass"

    def test_run_pipeline_no_gaps(self, service):
        """run_pipeline on a series with no gaps runs cleanly."""
        result = service.run_pipeline(values=_series_no_gaps())
        assert result["detection"]["total_missing"] == 0
        assert result["status"] == "completed"

    def test_run_pipeline_processing_time(self, service):
        """run_pipeline result includes total_processing_time_ms."""
        result = service.run_pipeline(values=_series_with_gaps())
        assert "total_processing_time_ms" in result
        assert result["total_processing_time_ms"] >= 0.0


# ===================================================================
# 9. Job Management Tests
# ===================================================================


class TestJobs:
    """Tests for create_job, list_jobs, get_job, delete_job."""

    def test_create_job_returns_dict_with_id(self, service):
        """create_job returns a dict containing job_id."""
        job = service.create_job(series_id="ts_001")
        assert isinstance(job, dict)
        assert "job_id" in job
        assert job["status"] == "pending"

    def test_create_job_with_request_dict(self, service):
        """create_job supports the router's request dict convention."""
        job = service.create_job(request={
            "series_name": "my_series",
            "strategy": "linear",
        })
        assert job["series_id"] == "my_series"
        assert job["strategy"] == "linear"

    def test_create_job_increments_stats(self, service):
        """create_job increments total_jobs."""
        assert service._stats["total_jobs"] == 0
        service.create_job()
        assert service._stats["total_jobs"] == 1

    def test_list_jobs_returns_list(self, service):
        """list_jobs returns a list of all jobs."""
        service.create_job(series_id="s1")
        service.create_job(series_id="s2")
        jobs = service.list_jobs()
        assert isinstance(jobs, list)
        assert len(jobs) == 2

    def test_list_jobs_empty_initially(self, service):
        """list_jobs returns empty list when no jobs exist."""
        jobs = service.list_jobs()
        assert jobs == []

    def test_list_jobs_with_pagination(self, service):
        """list_jobs supports limit and offset."""
        for i in range(5):
            service.create_job(series_id=f"s{i}")
        page = service.list_jobs(limit=2, offset=1)
        assert len(page) == 2

    def test_list_jobs_with_status_filter(self, service):
        """list_jobs filters by status."""
        service.create_job(series_id="s1")
        jobs = service.list_jobs(status="pending")
        assert len(jobs) == 1
        no_jobs = service.list_jobs(status="completed")
        assert len(no_jobs) == 0

    def test_get_job_returns_dict(self, service):
        """get_job returns the job by ID."""
        job = service.create_job(series_id="ts_001")
        fetched = service.get_job(job["job_id"])
        assert fetched is not None
        assert fetched["job_id"] == job["job_id"]

    def test_get_job_unknown_returns_none(self, service):
        """get_job returns None for unknown ID."""
        result = service.get_job("nonexistent-id")
        assert result is None

    def test_delete_job_removes_job(self, service):
        """delete_job removes the job from the store."""
        job = service.create_job()
        assert service.delete_job(job["job_id"]) is True
        assert service.get_job(job["job_id"]) is None

    def test_delete_job_unknown_returns_false(self, service):
        """delete_job returns False for unknown ID."""
        result = service.delete_job("nonexistent-id")
        assert result is False


# ===================================================================
# 10. Edge Cases and Robustness
# ===================================================================


class TestEdgeCases:
    """Edge case and robustness tests."""

    def test_detect_gaps_single_none(self, service):
        """detect_gaps handles a single None value."""
        result = service.detect_gaps(values=[None])
        assert result["total_missing"] == 1

    def test_fill_gaps_single_none(self, service):
        """fill_gaps handles a single None value."""
        result = service.fill_gaps(values=[None])
        assert len(result["filled_values"]) == 1

    def test_fill_gaps_large_series(self, service):
        """fill_gaps handles a large series efficiently."""
        values = [float(i) if i % 10 != 0 else None for i in range(1000)]
        result = service.fill_gaps(values=values)
        assert len(result["filled_values"]) == 1000

    def test_multiple_operations_sequential(self, service):
        """Multiple operations run sequentially without interference."""
        d1 = service.detect_gaps(values=[1.0, None, 3.0])
        d2 = service.detect_gaps(values=[4.0, 5.0, None])
        f1 = service.fill_gaps(values=[1.0, None, 3.0])
        assert d1["detection_id"] != d2["detection_id"]
        assert f1["fill_id"] not in [d1["detection_id"], d2["detection_id"]]

    def test_stats_accumulate_correctly(self, service):
        """Statistics accumulate across multiple operations."""
        service.detect_gaps(values=_series_with_gaps())
        service.detect_gaps(values=_series_no_gaps())
        service.fill_gaps(values=_series_with_gaps())
        service.validate_fills(filled_values=[1.0, 2.0, 3.0])
        service.create_job()
        service.run_pipeline(values=_series_with_gaps())

        stats = service.get_stats()
        # 2 direct + 1 from pipeline = 3 detections
        assert stats["total_detections"] >= 3
        # 1 direct + 1 from pipeline = 2 fills
        assert stats["total_fills"] >= 2
        # 1 direct + 1 from pipeline = 2 validations
        assert stats["total_validations"] >= 2
        assert stats["total_jobs"] == 1
        assert stats["total_pipelines"] == 1

    def test_health_check_engine_status_dict(self, service):
        """health_check engines dict reports True for initialized engines."""
        result = service.health_check()
        engines = result["engines"]
        assert isinstance(engines, dict)
        # At minimum, gap_detector should be initialized
        assert engines["gap_detector"] is True

    def test_pipeline_with_strategy_param(self, service):
        """run_pipeline accepts strategy parameter."""
        result = service.run_pipeline(
            values=_series_with_gaps(),
            strategy="linear",
        )
        assert result["status"] == "completed"

    def test_pipeline_enable_validation_false(self, service):
        """run_pipeline with enable_validation=False skips validation."""
        result = service.run_pipeline(
            values=_series_with_gaps(),
            enable_validation=False,
        )
        assert result["validation"] is None
