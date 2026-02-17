# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for AGENT-DATA-012 Missing Value Imputer Agent tests.

Provides reusable test fixtures for configuration, sample data, mock objects,
and pre-computed results used across all test modules.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-012 Missing Value Imputer (GL-DATA-X-015)
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
# Autouse fixture: clean GL_MVI_ environment variables before each test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_mvi_env(monkeypatch):
    """Remove all GL_MVI_ environment variables before each test.

    This prevents leakage of environment state between tests that set
    GL_MVI_ prefixed variables via monkeypatch or os.environ.
    """
    keys_to_remove = [k for k in os.environ if k.startswith("GL_MVI_")]
    for key in keys_to_remove:
        monkeypatch.delenv(key, raising=False)

    # Also reset the singleton config so each test starts fresh
    from greenlang.missing_value_imputer.config import reset_config
    reset_config()

    yield

    # Post-test cleanup: reset singleton again
    reset_config()


# ---------------------------------------------------------------------------
# Configuration fixtures
# ---------------------------------------------------------------------------


def _make_config(**overrides):
    """Create a MissingValueImputerConfig with optional overrides."""
    from greenlang.missing_value_imputer.config import MissingValueImputerConfig

    defaults = dict(
        database_url="postgresql://test:test@localhost:5432/testdb",
        redis_url="redis://localhost:6379/0",
        s3_bucket_url="s3://test-mvi-bucket",
        log_level="INFO",
        batch_size=1000,
        max_records=100_000,
        default_strategy="auto",
        confidence_threshold=0.7,
        max_missing_pct=0.8,
        enable_statistical=True,
        knn_neighbors=5,
        max_knn_dataset_size=50_000,
        mice_iterations=10,
        multiple_imputations=5,
        enable_ml_imputation=True,
        enable_timeseries=True,
        interpolation_method="linear",
        seasonal_period=12,
        trend_window=6,
        enable_rule_based=True,
        validation_split=0.2,
        worker_count=4,
        pool_min_size=2,
        pool_max_size=10,
        cache_ttl=3600,
        rate_limit_rpm=120,
        rate_limit_burst=20,
        enable_provenance=True,
        provenance_hash_algorithm="sha256",
        enable_metrics=True,
        default_confidence_method="ensemble",
    )
    defaults.update(overrides)
    return MissingValueImputerConfig(**defaults)


@pytest.fixture
def config():
    """Create a MissingValueImputerConfig with test defaults."""
    return _make_config()


# ---------------------------------------------------------------------------
# Helper: build service with mocked engines
# ---------------------------------------------------------------------------


def _build_service(cfg=None):
    """Build a MissingValueImputerService using the given config.

    All sub-engines are real instances wired to the same config.
    """
    if cfg is None:
        cfg = _make_config()

    from greenlang.missing_value_imputer.missingness_analyzer import MissingnessAnalyzerEngine
    from greenlang.missing_value_imputer.statistical_imputer import StatisticalImputerEngine
    from greenlang.missing_value_imputer.ml_imputer import MLImputerEngine
    from greenlang.missing_value_imputer.rule_based_imputer import RuleBasedImputerEngine
    from greenlang.missing_value_imputer.time_series_imputer import TimeSeriesImputerEngine
    from greenlang.missing_value_imputer.validation_engine import ValidationEngine
    from greenlang.missing_value_imputer.imputation_pipeline import ImputationPipelineEngine

    analyzer = MissingnessAnalyzerEngine(cfg)
    stat_imputer = StatisticalImputerEngine(cfg)
    ml_imputer = MLImputerEngine(cfg)
    rule_imputer = RuleBasedImputerEngine(cfg)
    ts_imputer = TimeSeriesImputerEngine(cfg)
    validator = ValidationEngine(cfg)
    pipeline = ImputationPipelineEngine(
        cfg, analyzer, stat_imputer, ml_imputer,
        rule_imputer, ts_imputer, validator,
    )
    return {
        "config": cfg,
        "analyzer": analyzer,
        "stat_imputer": stat_imputer,
        "ml_imputer": ml_imputer,
        "rule_imputer": rule_imputer,
        "ts_imputer": ts_imputer,
        "validator": validator,
        "pipeline": pipeline,
    }


# ---------------------------------------------------------------------------
# Sample records with missing values
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_records_with_missing() -> List[Dict[str, Any]]:
    """Return 10 records with various missing patterns (None, empty, NaN)."""
    return [
        {"id": "rec-001", "temperature": 22.5, "category": "office", "emission_factor": 0.5, "date": "2025-01-01"},
        {"id": "rec-002", "temperature": None, "category": "office", "emission_factor": 0.6, "date": "2025-02-01"},
        {"id": "rec-003", "temperature": 25.0, "category": "", "emission_factor": None, "date": "2025-03-01"},
        {"id": "rec-004", "temperature": 21.3, "category": "warehouse", "emission_factor": 0.8, "date": "2025-04-01"},
        {"id": "rec-005", "temperature": float("nan"), "category": "factory", "emission_factor": 1.2, "date": "2025-05-01"},
        {"id": "rec-006", "temperature": 23.8, "category": "office", "emission_factor": None, "date": "2025-06-01"},
        {"id": "rec-007", "temperature": None, "category": None, "emission_factor": 0.55, "date": "2025-07-01"},
        {"id": "rec-008", "temperature": 26.1, "category": "factory", "emission_factor": 1.1, "date": ""},
        {"id": "rec-009", "temperature": 20.0, "category": "warehouse", "emission_factor": 0.9, "date": "2025-09-01"},
        {"id": "rec-010", "temperature": None, "category": "office", "emission_factor": None, "date": "2025-10-01"},
    ]


@pytest.fixture
def sample_complete_records() -> List[Dict[str, Any]]:
    """Return 10 complete records with no missing values."""
    return [
        {"id": f"cmp-{i:03d}", "temperature": 20.0 + i * 0.5, "category": "office", "emission_factor": 0.4 + i * 0.1, "quantity": 100 + i * 10}
        for i in range(1, 11)
    ]


@pytest.fixture
def sample_time_series() -> List[float]:
    """Return 20 timestamped values with gaps (None)."""
    values = []
    for i in range(20):
        if i in (3, 4, 8, 12, 13, 14, 18):
            values.append(None)
        else:
            values.append(10.0 + math.sin(i * 0.5) * 5.0)
    return values


@pytest.fixture
def sample_rules() -> List[Dict[str, Any]]:
    """Return 5 domain rules for testing rule-based imputation."""
    return [
        {
            "name": "office_ef_default",
            "target_column": "emission_factor",
            "conditions": [{"field_name": "category", "condition_type": "equals", "value": "office"}],
            "impute_value": 0.55,
            "priority": "high",
        },
        {
            "name": "warehouse_ef_default",
            "target_column": "emission_factor",
            "conditions": [{"field_name": "category", "condition_type": "equals", "value": "warehouse"}],
            "impute_value": 0.85,
            "priority": "medium",
        },
        {
            "name": "factory_ef_default",
            "target_column": "emission_factor",
            "conditions": [{"field_name": "category", "condition_type": "equals", "value": "factory"}],
            "impute_value": 1.15,
            "priority": "medium",
        },
        {
            "name": "high_temp_ef",
            "target_column": "emission_factor",
            "conditions": [{"field_name": "temperature", "condition_type": "greater_than", "value": 25}],
            "impute_value": 1.0,
            "priority": "low",
        },
        {
            "name": "catch_all_ef",
            "target_column": "emission_factor",
            "conditions": [],
            "impute_value": 0.7,
            "priority": "default",
        },
    ]


# ---------------------------------------------------------------------------
# Mock service fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_service():
    """Create a mock MissingValueImputerService."""
    service = MagicMock()
    service.health_check.return_value = {"status": "healthy"}
    service.get_statistics.return_value = {"total_jobs": 0}
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
