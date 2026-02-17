# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for AGENT-DATA-013 Outlier Detection Agent tests.

Provides reusable test fixtures for configuration, sample data, mock objects,
pre-built detection results, and domain thresholds used across all test modules.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-013 Outlier Detection (GL-DATA-X-016)
"""

from __future__ import annotations

import math
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Autouse fixture: clean GL_OD_ environment variables before each test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_od_env(monkeypatch):
    """Remove all GL_OD_ environment variables before each test.

    This prevents leakage of environment state between tests that set
    GL_OD_ prefixed variables via monkeypatch or os.environ.
    """
    keys_to_remove = [k for k in os.environ if k.startswith("GL_OD_")]
    for key in keys_to_remove:
        monkeypatch.delenv(key, raising=False)

    # Also reset the singleton config so each test starts fresh
    from greenlang.outlier_detector.config import reset_config
    reset_config()

    yield

    # Post-test cleanup: reset singleton again
    reset_config()


# ---------------------------------------------------------------------------
# Configuration fixtures
# ---------------------------------------------------------------------------


def _make_config(**overrides):
    """Create an OutlierDetectorConfig with optional overrides."""
    from greenlang.outlier_detector.config import OutlierDetectorConfig

    defaults = dict(
        database_url="postgresql://test:test@localhost:5432/testdb",
        redis_url="redis://localhost:6379/0",
        s3_bucket_url="s3://test-od-bucket",
        log_level="INFO",
        batch_size=1000,
        max_records=100_000,
        iqr_multiplier=1.5,
        zscore_threshold=3.0,
        mad_threshold=3.5,
        grubbs_alpha=0.05,
        lof_neighbors=20,
        isolation_trees=100,
        ensemble_method="weighted_average",
        min_consensus=2,
        enable_contextual=True,
        enable_temporal=True,
        enable_multivariate=True,
        default_treatment="flag",
        winsorize_pct=0.05,
        worker_count=4,
        pool_min_size=2,
        pool_max_size=10,
        cache_ttl=3600,
        rate_limit_rpm=120,
        rate_limit_burst=20,
        enable_provenance=True,
        provenance_hash_algorithm="sha256",
        enable_metrics=True,
    )
    defaults.update(overrides)
    return OutlierDetectorConfig(**defaults)


@pytest.fixture
def config():
    """Create an OutlierDetectorConfig with test defaults."""
    return _make_config()


# ---------------------------------------------------------------------------
# Helper: build service with mocked engines
# ---------------------------------------------------------------------------


def _build_service(cfg=None):
    """Build an OutlierDetectorService using the given config.

    All sub-engines are real instances wired to the same config.
    """
    if cfg is None:
        cfg = _make_config()

    from greenlang.outlier_detector.setup import OutlierDetectorService
    return OutlierDetectorService(config=cfg)


# ---------------------------------------------------------------------------
# Sample numeric data with known outliers
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_numeric_data() -> List[float]:
    """Return 100 values with 5 known outliers.

    Normal range: 1..95, Outliers: 500, 600, 700, -200, -300.
    """
    normal = [float(i) for i in range(1, 96)]
    outliers = [500.0, 600.0, 700.0, -200.0, -300.0]
    return normal + outliers


@pytest.fixture
def sample_records() -> List[Dict[str, Any]]:
    """Return 10 dicts with numeric/categorical columns and some outlier values."""
    return [
        {"id": "r-001", "emissions": 10.0, "sector": "energy", "temperature": 22.0},
        {"id": "r-002", "emissions": 12.0, "sector": "energy", "temperature": 23.0},
        {"id": "r-003", "emissions": 11.0, "sector": "transport", "temperature": 21.0},
        {"id": "r-004", "emissions": 500.0, "sector": "energy", "temperature": 22.5},
        {"id": "r-005", "emissions": 9.0, "sector": "transport", "temperature": 24.0},
        {"id": "r-006", "emissions": 13.0, "sector": "energy", "temperature": -50.0},
        {"id": "r-007", "emissions": 10.5, "sector": "industry", "temperature": 22.0},
        {"id": "r-008", "emissions": 11.5, "sector": "industry", "temperature": 23.0},
        {"id": "r-009", "emissions": 10.0, "sector": "transport", "temperature": 21.5},
        {"id": "r-010", "emissions": 12.0, "sector": "energy", "temperature": 22.0},
    ]


@pytest.fixture
def sample_time_series() -> List[float]:
    """Return 30 values with 3 known anomalies at indices 10, 20, 25."""
    values = []
    for i in range(30):
        if i == 10:
            values.append(500.0)
        elif i == 20:
            values.append(-300.0)
        elif i == 25:
            values.append(800.0)
        else:
            values.append(10.0 + math.sin(i * 0.5) * 3.0)
    return values


@pytest.fixture
def sample_multivariate() -> List[Dict[str, float]]:
    """Return 20 records with 3 numeric columns, 2 multivariate outliers."""
    records = []
    for i in range(18):
        records.append({
            "x": float(i),
            "y": float(i * 2),
            "z": float(i * 3),
        })
    # Multivariate outliers
    records.append({"x": 100.0, "y": 200.0, "z": 300.0})
    records.append({"x": -50.0, "y": -100.0, "z": -150.0})
    return records


@pytest.fixture
def sample_detections():
    """Return pre-built OutlierScore and DetectionResult objects."""
    from greenlang.outlier_detector.models import (
        DetectionMethod,
        DetectionResult,
        OutlierScore,
        SeverityLevel,
    )

    scores = [
        OutlierScore(
            record_index=0, column_name="val", value=10.0,
            method=DetectionMethod.IQR, score=0.1, is_outlier=False,
            threshold=1.5, severity=SeverityLevel.INFO, confidence=0.55,
            provenance_hash="a" * 64,
        ),
        OutlierScore(
            record_index=3, column_name="val", value=500.0,
            method=DetectionMethod.IQR, score=0.95, is_outlier=True,
            threshold=1.5, severity=SeverityLevel.CRITICAL, confidence=0.97,
            provenance_hash="b" * 64,
        ),
    ]

    result = DetectionResult(
        column_name="val",
        method=DetectionMethod.IQR,
        total_points=10,
        outliers_found=1,
        outlier_pct=0.1,
        scores=scores,
        lower_fence=-5.0,
        upper_fence=50.0,
        processing_time_ms=1.5,
        provenance_hash="c" * 64,
    )

    return {"scores": scores, "result": result}


@pytest.fixture
def sample_thresholds():
    """Return 3 DomainThreshold objects."""
    from greenlang.outlier_detector.models import DomainThreshold, ThresholdSource

    return [
        DomainThreshold(
            column_name="emissions",
            lower_bound=0.0,
            upper_bound=100.0,
            source=ThresholdSource.DOMAIN,
            description="Max expected emissions",
        ),
        DomainThreshold(
            column_name="temperature",
            lower_bound=-40.0,
            upper_bound=60.0,
            source=ThresholdSource.REGULATORY,
            description="Operating temperature range",
        ),
        DomainThreshold(
            column_name="pressure",
            lower_bound=0.5,
            upper_bound=10.0,
            source=ThresholdSource.STATISTICAL,
            description="Normal pressure range",
        ),
    ]


# ---------------------------------------------------------------------------
# Mock service fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_service():
    """Create a mock OutlierDetectorService."""
    service = MagicMock()
    service.health_check.return_value = {"status": "healthy"}
    service.get_statistics.return_value = MagicMock(total_jobs=0)
    return service


# ---------------------------------------------------------------------------
# Mock prometheus
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_prometheus():
    """Mock prometheus_client Counter, Histogram, and Gauge classes."""
    mock_counter = MagicMock()
    mock_counter.labels.return_value.inc = MagicMock()

    mock_histogram = MagicMock()
    mock_histogram.labels.return_value.observe = MagicMock()

    mock_gauge = MagicMock()
    mock_gauge.set = MagicMock()

    return {
        "Counter": mock_counter,
        "Histogram": mock_histogram,
        "Gauge": mock_gauge,
    }
