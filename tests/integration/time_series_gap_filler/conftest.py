# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for AGENT-DATA-014 Time Series Gap Filler integration tests.

Provides reusable test fixtures for:
- Override of parent conftest autouse fixtures (mock_agents, block_network)
- Configuration reset between tests (fresh_config)
- Sample time series data with gaps
- Timestamp generation
- Calendar definitions
- Reference series for cross-series testing
- FastAPI test app factory

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-014 Time Series Gap Filler (GL-DATA-X-017)
"""

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from greenlang.time_series_gap_filler.config import reset_config


# ---------------------------------------------------------------------------
# Override parent conftest autouse fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def mock_agents():
    """Override parent conftest mock_agents fixture.

    The parent tests/integration/conftest.py defines an autouse fixture
    that patches greenlang.agents.registry.get_agent, which does not
    apply to TSGF integration tests.
    """
    return {}


@pytest.fixture(scope="session", autouse=True)
def block_network():
    """Override parent conftest block_network fixture.

    The parent tests/integration/conftest.py blocks all socket access,
    which can interfere with asyncio event loop creation. We disable it
    for TSGF integration tests since our tests are fully self-contained.
    """
    pass


# ---------------------------------------------------------------------------
# Configuration reset
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def fresh_config():
    """Reset the singleton config before and after each test.

    Ensures each test starts with a clean default configuration
    and does not leak state to subsequent tests.
    """
    reset_config()
    yield
    reset_config()


# ---------------------------------------------------------------------------
# Sample series with gaps
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_series_with_gaps():
    """Return a 12-element numeric series with 4 gaps (None values).

    Pattern: [10, 12, None, 14, 16, None, None, 22, 24, 26, None, 30]
    Gap positions: index 2 (length 1), indices 5-6 (length 2), index 10 (length 1)
    """
    return [10.0, 12.0, None, 14.0, 16.0, None, None, 22.0, 24.0, 26.0, None, 30.0]


@pytest.fixture
def sample_timestamps():
    """Return 12 ISO-8601 hourly timestamps starting 2025-01-01T00:00:00."""
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    return [(base + timedelta(hours=i)).isoformat() for i in range(12)]


@pytest.fixture
def sample_datetime_timestamps():
    """Return 12 datetime objects (hourly) starting 2025-01-01T00:00:00 UTC."""
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    return [base + timedelta(hours=i) for i in range(12)]


# ---------------------------------------------------------------------------
# Longer series for seasonal / frequency testing
# ---------------------------------------------------------------------------


@pytest.fixture
def long_series_with_gaps():
    """Return a 48-element series with periodic gaps every 12th element.

    Useful for seasonal decomposition testing.
    """
    series = []
    for i in range(48):
        if i % 12 == 5 or i % 12 == 11:
            series.append(None)
        else:
            series.append(100.0 + i * 2.0 + 10.0 * (i % 12))
    return series


@pytest.fixture
def daily_timestamps_48():
    """Return 48 daily datetime objects starting 2025-01-01."""
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    return [base + timedelta(days=i) for i in range(48)]


# ---------------------------------------------------------------------------
# Reference series for cross-series filling
# ---------------------------------------------------------------------------


@pytest.fixture
def reference_series():
    """Return a complete (no gaps) 12-element reference series.

    Correlated with sample_series_with_gaps via a linear relationship.
    """
    return [11.0, 13.0, 15.0, 15.5, 17.0, 19.0, 21.0, 23.0, 25.0, 27.0, 29.0, 31.0]


# ---------------------------------------------------------------------------
# Calendar definitions
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_calendar():
    """Return a CalendarDefinition for calendar-aware filling tests."""
    from greenlang.time_series_gap_filler.seasonal_filler import CalendarDefinition

    return CalendarDefinition(
        business_days=[0, 1, 2, 3, 4],
        holidays=["2025-01-01", "2025-12-25"],
        fiscal_periods={"Q1": (1, 3), "Q2": (4, 6), "Q3": (7, 9), "Q4": (10, 12)},
    )


# ---------------------------------------------------------------------------
# Series with no gaps (control fixture)
# ---------------------------------------------------------------------------


@pytest.fixture
def complete_series():
    """Return a complete 12-element series with no gaps."""
    return [10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0]


# ---------------------------------------------------------------------------
# FastAPI test app factory
# ---------------------------------------------------------------------------


@pytest.fixture
def gap_filler_app():
    """Create a minimal FastAPI test app with TSGF service attached.

    Mounts the real router from greenlang.time_series_gap_filler.api.router
    and attaches a lightweight mock service to
    app.state.time_series_gap_filler_service (matching the attribute name
    that _get_service() reads from the request).
    """
    try:
        from fastapi import FastAPI
    except ImportError:
        pytest.skip("FastAPI not installed; skipping API integration tests")

    from greenlang.time_series_gap_filler.api.router import router, FASTAPI_AVAILABLE

    if not FASTAPI_AVAILABLE:
        pytest.skip("FastAPI not available in router module")

    app = FastAPI(title="TSGF Integration Test")
    app.include_router(router)

    # Attach a mock service to app.state using the name the router expects
    svc = _build_mock_service()
    app.state.time_series_gap_filler_service = svc

    return app


@pytest.fixture
def test_client(gap_filler_app):
    """Create a synchronous test client for FastAPI integration tests."""
    from fastapi.testclient import TestClient

    return TestClient(gap_filler_app)


# ---------------------------------------------------------------------------
# Mock service builder
# ---------------------------------------------------------------------------


class _MockModel:
    """A lightweight object that supports model_dump(mode=...) calls."""

    def __init__(self, data):
        self._data = data
        # Also expose dict keys as attributes for .passed, .suitable_for_fill etc.
        for k, v in data.items():
            setattr(self, k, v)

    def model_dump(self, mode=None):
        return dict(self._data)

    def __getitem__(self, key):
        return self._data[key]

    def __contains__(self, key):
        return key in self._data


def _build_mock_service():
    """Build a mock TimeSeriesGapFillerService with deterministic responses.

    Each method is a real callable (side_effect on a MagicMock) that
    matches the exact signatures used by the router endpoints.
    """
    svc = MagicMock()
    svc._jobs = {}
    svc._calendars = {}

    # ------------------------------------------------------------------
    # create_job(config={...})
    # ------------------------------------------------------------------
    def _create_job(config=None, **kwargs):
        config = config or {}
        job_id = str(uuid.uuid4())
        job = {
            "job_id": job_id,
            "series_id": config.get("series_id", ""),
            "status": "pending",
            "strategy": config.get("strategy", "auto"),
            "provenance_hash": "abc123" + str(uuid.uuid4())[:8],
        }
        svc._jobs[job_id] = job
        return job

    svc.create_job.side_effect = _create_job

    # ------------------------------------------------------------------
    # list_jobs(limit=50, offset=0)
    # ------------------------------------------------------------------
    def _list_jobs(limit=50, offset=0, **kwargs):
        jobs = list(svc._jobs.values())
        return {
            "jobs": jobs[offset:offset + limit],
            "count": len(jobs),
            "total": len(jobs),
            "limit": limit,
            "offset": offset,
        }

    svc.list_jobs.side_effect = _list_jobs

    # ------------------------------------------------------------------
    # get_job(job_id)
    # ------------------------------------------------------------------
    def _get_job(job_id):
        return svc._jobs.get(job_id)

    svc.get_job.side_effect = _get_job

    # ------------------------------------------------------------------
    # delete_job(job_id) -- router catches ValueError as 404
    # ------------------------------------------------------------------
    def _delete_job(job_id):
        if job_id not in svc._jobs:
            raise ValueError(f"Job {job_id} not found")
        svc._jobs[job_id]["status"] = "deleted"
        return {"deleted": True, "job_id": job_id}

    svc.delete_job.side_effect = _delete_job

    # ------------------------------------------------------------------
    # detect_gaps(series=..., timestamps=..., frequency=...)
    # Returns a _MockModel with model_dump()
    # ------------------------------------------------------------------
    def _detect_gaps(series=None, timestamps=None, frequency=None, **kwargs):
        if series is not None and len(series) > 0 and all(v is None for v in series):
            raise ValueError("All values are missing")
        gaps = []
        total_missing = 0
        if series:
            for i, v in enumerate(series):
                if v is None:
                    total_missing += 1
                    gaps.append({"start_index": i, "end_index": i, "length": 1})
        return _MockModel({
            "detection_id": str(uuid.uuid4()),
            "total_gaps": len(gaps),
            "total_missing": total_missing,
            "total_points": len(series) if series else 0,
            "series_length": len(series) if series else 0,
            "gap_pct": total_missing / len(series) if series else 0.0,
            "gaps": gaps,
            "gap_types": {"missing": total_missing},
            "avg_gap_length": 1.0 if total_missing > 0 else 0.0,
            "max_gap_length": 1 if total_missing > 0 else 0,
            "processing_time_ms": 0.5,
            "provenance_hash": "detect_hash_" + str(uuid.uuid4())[:8],
        })

    svc.detect_gaps.side_effect = _detect_gaps

    # ------------------------------------------------------------------
    # fill_gaps(series=..., timestamps=..., gaps=..., strategy=...)
    # Returns a _MockModel with model_dump()
    # ------------------------------------------------------------------
    def _fill_gaps(series=None, timestamps=None, gaps=None, strategy=None,
                   **kwargs):
        if series is None:
            raise ValueError("series is required")
        strategy = strategy or "linear"
        filled_values = []
        gaps_filled_count = 0
        for i, v in enumerate(series):
            if v is None:
                filled_values.append({"index": i, "value": 0.0, "confidence": 0.85})
                gaps_filled_count += 1
            else:
                filled_values.append({"index": i, "value": v, "confidence": 1.0})
        return _MockModel({
            "fill_id": str(uuid.uuid4()),
            "series_name": "",
            "strategy": strategy,
            "total_filled": gaps_filled_count,
            "total_gaps": gaps_filled_count,
            "fill_rate": gaps_filled_count / len(series) if series else 0.0,
            "filled_values": filled_values,
            "avg_confidence": 0.85,
            "min_confidence": 0.85,
            "distribution_preserved": True,
            "processing_time_ms": 1.0,
            "provenance_hash": "fill_hash_" + str(uuid.uuid4())[:8],
        })

    svc.fill_gaps.side_effect = _fill_gaps

    # ------------------------------------------------------------------
    # validate_fills(fills=..., original_series=...)
    # Returns a list of _MockModel objects
    # ------------------------------------------------------------------
    def _validate_fills(fills=None, original_series=None, **kwargs):
        result = _MockModel({
            "validation_id": str(uuid.uuid4()),
            "fill_id": "",
            "passed": True,
            "total_checks": 3,
            "passed_checks": 3,
            "failed_checks": 0,
            "checks": [],
            "overall_confidence": 0.85,
            "distribution_test": "passed",
            "processing_time_ms": 0.5,
            "provenance_hash": "val_hash_" + str(uuid.uuid4())[:8],
        })
        return [result]

    svc.validate_fills.side_effect = _validate_fills

    # ------------------------------------------------------------------
    # analyze_frequency(timestamps=...)
    # Returns a _MockModel with model_dump()
    # ------------------------------------------------------------------
    def _analyze_frequency(timestamps=None, **kwargs):
        return _MockModel({
            "analysis_id": str(uuid.uuid4()),
            "detected_frequency": "hourly",
            "frequency_seconds": 3600.0,
            "frequency_level": "hourly",
            "regularity_score": 0.95,
            "confidence": 0.9,
            "num_points": len(timestamps) if timestamps else 0,
            "median_interval": 3600.0,
            "std_interval": 0.0,
            "is_regular": True,
            "processing_time_ms": 0.3,
            "provenance_hash": "freq_hash_" + str(uuid.uuid4())[:8],
        })

    svc.analyze_frequency.side_effect = _analyze_frequency

    # ------------------------------------------------------------------
    # health_check()
    # Returns a plain dict
    # ------------------------------------------------------------------
    svc.health_check.return_value = {
        "status": "healthy",
        "service": "time-series-gap-filler",
        "started": True,
        "engine_count": 4,
        "engines": {
            "gap_detector": True,
            "frequency_analyzer": True,
            "interpolation": True,
            "seasonal_filler": True,
        },
        "stores": {},
        "uptime_seconds": 100.0,
        "provenance_entries": 0,
        "prometheus_available": False,
    }

    # ------------------------------------------------------------------
    # get_statistics()
    # Returns a _MockModel with model_dump()
    # ------------------------------------------------------------------
    svc.get_statistics.return_value = _MockModel({
        "total_jobs": 0,
        "completed_jobs": 0,
        "failed_jobs": 0,
        "cancelled_jobs": 0,
        "total_gaps_detected": 0,
        "total_gaps_filled": 0,
        "total_validations": 0,
        "total_frequency_analyses": 0,
        "total_correlations": 0,
        "total_calendars": 0,
        "active_jobs": 0,
        "avg_gap_pct": 0.0,
        "avg_fill_confidence": 0.0,
        "by_strategy": {},
        "by_gap_type": {},
        "by_frequency": {},
        "provenance_entries": 0,
    })

    # ------------------------------------------------------------------
    # create_calendar(calendar={...})
    # Returns a _MockModel with model_dump()
    # ------------------------------------------------------------------
    def _create_calendar(calendar=None, **kwargs):
        calendar = calendar or {}
        cal_id = str(uuid.uuid4())
        cal_data = {
            "calendar_id": cal_id,
            "name": calendar.get("name", ""),
            "calendar_type": calendar.get("calendar_type", "business"),
            "timezone_name": calendar.get("timezone", "UTC"),
            "business_days": calendar.get("business_days", [0, 1, 2, 3, 4]),
            "holidays": calendar.get("holidays", []),
            "fiscal_year_start_month": calendar.get("fiscal_year_start_month", 1),
            "reporting_periods": calendar.get("reporting_periods", []),
            "active": True,
            "created_at": "2025-01-01T00:00:00Z",
            "provenance_hash": "cal_hash_" + str(uuid.uuid4())[:8],
        }
        svc._calendars[cal_id] = cal_data
        return _MockModel(cal_data)

    svc.create_calendar.side_effect = _create_calendar

    # ------------------------------------------------------------------
    # list_calendars(limit=50, offset=0)
    # Returns a plain dict
    # ------------------------------------------------------------------
    def _list_calendars(limit=50, offset=0, **kwargs):
        cals = list(svc._calendars.values())
        return {
            "calendars": cals[offset:offset + limit],
            "count": len(cals),
            "total": len(cals),
            "limit": limit,
            "offset": offset,
        }

    svc.list_calendars.side_effect = _list_calendars

    return svc
