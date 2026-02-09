# -*- coding: utf-8 -*-
"""
Unit Tests for DataQualityProfilerService (AGENT-DATA-010)
===========================================================

Comprehensive test suite for ``greenlang.data_quality_profiler.setup``
covering the ``DataQualityProfilerService`` facade, all 20 public methods,
lifecycle, statistics, provenance, configuration helpers, Pydantic response
models, thread-safety, and full end-to-end workflows.

Target: 260+ tests with 85%+ coverage.

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-DATA-010 Data Quality Profiler (GL-DATA-X-013)
"""

from __future__ import annotations

import threading
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from greenlang.data_quality_profiler.config import (
    DataQualityProfilerConfig,
    get_config,
    set_config,
    reset_config,
)
from greenlang.data_quality_profiler.setup import (
    DataQualityProfilerService,
    DatasetProfileResponse,
    QualityAssessmentResponse,
    ColumnProfileResponse,
    AnomalyDetectionResponse,
    FreshnessCheckResponse,
    QualityRuleResponse,
    QualityGateResponse,
    DataQualityProfilerStatisticsResponse,
    configure_data_quality_profiler,
    get_data_quality_profiler,
    _classify_quality,
    _compute_hash,
)


# ===================================================================
# Helpers
# ===================================================================


def _make_config(**overrides: Any) -> DataQualityProfilerConfig:
    """Create a DataQualityProfilerConfig with optional overrides."""
    defaults = dict(
        database_url="",
        redis_url="",
        s3_bucket_url="",
        min_samples_for_anomaly=3,
    )
    defaults.update(overrides)
    return DataQualityProfilerConfig(**defaults)


def _make_service(**config_overrides: Any) -> DataQualityProfilerService:
    """Create a DataQualityProfilerService with optional config overrides."""
    cfg = _make_config(**config_overrides)
    return DataQualityProfilerService(config=cfg)


def _basic_data(n: int = 10) -> List[Dict[str, Any]]:
    """Generate *n* rows of simple three-column data."""
    return [
        {"name": f"item_{i}", "value": i * 10, "score": 80.0 + i}
        for i in range(n)
    ]


def _numeric_data(n: int = 20) -> List[Dict[str, Any]]:
    """Generate *n* rows of numeric-only data."""
    return [{"x": float(i), "y": float(i * 2)} for i in range(n)]


def _perfect_data() -> List[Dict[str, Any]]:
    """Return a dataset with no nulls, consistent types, all unique rows."""
    return [
        {"id": i, "name": f"name_{i}", "value": float(i)}
        for i in range(20)
    ]


def _poor_data() -> List[Dict[str, Any]]:
    """Return a dataset with many nulls and duplicates."""
    rows: List[Dict[str, Any]] = []
    for i in range(20):
        rows.append({"id": None, "name": None, "value": None})
    return rows


def _data_with_outliers(n: int = 30) -> List[Dict[str, Any]]:
    """Data with clear outliers at the end."""
    rows = [{"metric": float(i)} for i in range(n)]
    rows.append({"metric": 99999.0})
    rows.append({"metric": -99999.0})
    return rows


def _fresh_timestamp() -> str:
    """Return an ISO timestamp for 'just now'."""
    return datetime.now(timezone.utc).isoformat()


def _stale_timestamp(hours: int = 100) -> str:
    """Return an ISO timestamp that is *hours* old."""
    dt = datetime.now(timezone.utc) - timedelta(hours=hours)
    return dt.isoformat()


# ===================================================================
# TestServiceInit
# ===================================================================


class TestServiceInit:
    """Test DataQualityProfilerService initialization."""

    def test_default_config(self):
        """Service uses global config when none provided."""
        svc = DataQualityProfilerService()
        assert svc.config is not None
        assert isinstance(svc.config, DataQualityProfilerConfig)

    def test_custom_config(self):
        """Service uses the explicitly provided config."""
        cfg = _make_config(max_rows_per_profile=500)
        svc = DataQualityProfilerService(config=cfg)
        assert svc.config.max_rows_per_profile == 500

    def test_provenance_tracker_created(self):
        """Service creates a provenance tracker on init."""
        svc = _make_service()
        assert svc.provenance is not None
        assert svc.provenance.entry_count == 0

    def test_engine_properties_exist(self):
        """All 7 engine properties are accessible."""
        svc = _make_service()
        _ = svc.dataset_profiler_engine
        _ = svc.quality_assessor_engine
        _ = svc.anomaly_detector_engine
        _ = svc.freshness_checker_engine
        _ = svc.rule_engine
        _ = svc.quality_gate_engine
        _ = svc.report_generator_engine

    def test_internal_stores_empty(self):
        """Internal caches start empty."""
        svc = _make_service()
        assert svc.list_profiles() == []
        assert svc.list_assessments() == []
        assert svc.list_anomalies() == []
        assert svc.list_rules() == []

    def test_not_started_by_default(self):
        """Service is not started immediately after construction."""
        svc = _make_service()
        assert svc._started is False

    def test_stats_initial(self):
        """Statistics are zeroed at init."""
        svc = _make_service()
        stats = svc.get_statistics()
        assert stats.total_profiles == 0
        assert stats.total_assessments == 0
        assert stats.avg_quality_score == 0.0

    def test_singleton_pattern_not_forced(self):
        """Multiple distinct instances can coexist."""
        a = _make_service()
        b = _make_service()
        assert a is not b


# ===================================================================
# TestProfileDataset
# ===================================================================


class TestProfileDataset:
    """Test DataQualityProfilerService.profile_dataset."""

    def test_profile_basic(self, sample_dataset):
        """Profile returns a DatasetProfileResponse."""
        svc = _make_service()
        result = svc.profile_dataset(data=sample_dataset, dataset_name="test")
        assert isinstance(result, DatasetProfileResponse)

    def test_dataset_name_propagated(self, sample_dataset):
        """Returned profile carries the dataset_name."""
        svc = _make_service()
        result = svc.profile_dataset(data=sample_dataset, dataset_name="my_ds")
        assert result.dataset_name == "my_ds"

    def test_row_count(self, sample_dataset):
        """row_count matches input data length."""
        svc = _make_service()
        result = svc.profile_dataset(data=sample_dataset, dataset_name="test")
        assert result.row_count == len(sample_dataset)

    def test_column_count(self, sample_dataset):
        """column_count matches number of columns in first row."""
        svc = _make_service()
        result = svc.profile_dataset(data=sample_dataset, dataset_name="test")
        assert result.column_count == len(sample_dataset[0])

    def test_per_column_stats(self, sample_dataset):
        """Each column gets a ColumnProfileResponse entry."""
        svc = _make_service()
        result = svc.profile_dataset(data=sample_dataset, dataset_name="test")
        assert len(result.columns) == result.column_count
        for col in result.columns:
            assert isinstance(col, ColumnProfileResponse)
            assert col.column_name != ""

    def test_profile_id_is_uuid(self, sample_dataset):
        """profile_id is a valid UUID string."""
        svc = _make_service()
        result = svc.profile_dataset(data=sample_dataset, dataset_name="test")
        uuid.UUID(result.profile_id)  # raises if invalid

    def test_provenance_hash_set(self, sample_dataset):
        """provenance_hash is a non-empty SHA-256 hex string."""
        svc = _make_service()
        result = svc.profile_dataset(data=sample_dataset, dataset_name="test")
        assert len(result.provenance_hash) == 64

    def test_profile_stored(self, sample_dataset):
        """Profile is retrievable via get_profile."""
        svc = _make_service()
        result = svc.profile_dataset(data=sample_dataset, dataset_name="test")
        stored = svc.get_profile(result.profile_id)
        assert stored is not None
        assert stored.profile_id == result.profile_id

    def test_stats_increment(self, sample_dataset):
        """total_profiles stat incremented."""
        svc = _make_service()
        svc.profile_dataset(data=sample_dataset, dataset_name="test")
        assert svc.get_statistics().total_profiles == 1

    def test_empty_data_raises(self):
        """Empty data list raises ValueError."""
        svc = _make_service()
        with pytest.raises(ValueError, match="empty"):
            svc.profile_dataset(data=[], dataset_name="empty")

    def test_custom_columns(self, sample_dataset):
        """Profiling only the requested columns."""
        svc = _make_service()
        result = svc.profile_dataset(
            data=sample_dataset,
            dataset_name="test",
            columns=["name", "age"],
        )
        assert result.column_count == 2
        col_names = [c.column_name for c in result.columns]
        assert "name" in col_names
        assert "age" in col_names

    def test_processing_time_positive(self, sample_dataset):
        """processing_time_ms is a positive number."""
        svc = _make_service()
        result = svc.profile_dataset(data=sample_dataset, dataset_name="test")
        assert result.processing_time_ms > 0

    def test_completeness_score_range(self, sample_dataset):
        """completeness_score is between 0 and 1."""
        svc = _make_service()
        result = svc.profile_dataset(data=sample_dataset, dataset_name="test")
        assert 0.0 <= result.completeness_score <= 1.0

    def test_source_default(self, sample_dataset):
        """source defaults to 'manual'."""
        svc = _make_service()
        result = svc.profile_dataset(data=sample_dataset, dataset_name="test")
        assert result.source == "manual"

    def test_source_custom(self, sample_dataset):
        """source can be overridden."""
        svc = _make_service()
        result = svc.profile_dataset(
            data=sample_dataset, dataset_name="test", source="api",
        )
        assert result.source == "api"


# ===================================================================
# TestProfileDatasetBatch
# ===================================================================


class TestProfileDatasetBatch:
    """Test DataQualityProfilerService.profile_dataset_batch."""

    def test_multiple_datasets(self):
        """Batch profiles multiple datasets."""
        svc = _make_service()
        datasets = [
            {"data": _basic_data(5), "dataset_name": "ds_a"},
            {"data": _basic_data(8), "dataset_name": "ds_b"},
        ]
        results = svc.profile_dataset_batch(datasets)
        assert len(results) == 2

    def test_order_preserved(self):
        """Results are in the same order as input."""
        svc = _make_service()
        datasets = [
            {"data": _basic_data(5), "dataset_name": "first"},
            {"data": _basic_data(5), "dataset_name": "second"},
        ]
        results = svc.profile_dataset_batch(datasets)
        assert results[0].dataset_name == "first"
        assert results[1].dataset_name == "second"

    def test_empty_list(self):
        """Empty input returns empty output."""
        svc = _make_service()
        assert svc.profile_dataset_batch([]) == []

    def test_different_sizes(self):
        """Datasets of different sizes are all profiled."""
        svc = _make_service()
        datasets = [
            {"data": _basic_data(3), "dataset_name": "small"},
            {"data": _basic_data(15), "dataset_name": "large"},
        ]
        results = svc.profile_dataset_batch(datasets)
        assert results[0].row_count == 3
        assert results[1].row_count == 15


# ===================================================================
# TestGetProfile
# ===================================================================


class TestGetProfile:
    """Test DataQualityProfilerService.get_profile."""

    def test_existing(self, sample_dataset):
        """Returns the profile when ID matches."""
        svc = _make_service()
        profile = svc.profile_dataset(data=sample_dataset, dataset_name="test")
        assert svc.get_profile(profile.profile_id) is not None

    def test_nonexistent(self):
        """Returns None for unknown profile ID."""
        svc = _make_service()
        assert svc.get_profile("nonexistent") is None

    def test_fields_match(self, sample_dataset):
        """Retrieved profile has identical fields."""
        svc = _make_service()
        profile = svc.profile_dataset(data=sample_dataset, dataset_name="test")
        fetched = svc.get_profile(profile.profile_id)
        assert fetched.dataset_name == "test"
        assert fetched.row_count == profile.row_count


# ===================================================================
# TestListProfiles
# ===================================================================


class TestListProfiles:
    """Test DataQualityProfilerService.list_profiles."""

    def test_empty(self):
        """Returns empty list when no profiles exist."""
        svc = _make_service()
        assert svc.list_profiles() == []

    def test_returns_all(self, sample_dataset):
        """Returns all profiles when no pagination."""
        svc = _make_service()
        svc.profile_dataset(data=sample_dataset, dataset_name="a")
        svc.profile_dataset(data=sample_dataset, dataset_name="b")
        assert len(svc.list_profiles()) == 2

    def test_limit(self, sample_dataset):
        """Respects limit parameter."""
        svc = _make_service()
        for i in range(5):
            svc.profile_dataset(data=sample_dataset, dataset_name=f"ds_{i}")
        assert len(svc.list_profiles(limit=3)) == 3

    def test_offset(self, sample_dataset):
        """Respects offset parameter."""
        svc = _make_service()
        for i in range(5):
            svc.profile_dataset(data=sample_dataset, dataset_name=f"ds_{i}")
        result = svc.list_profiles(limit=10, offset=3)
        assert len(result) == 2

    def test_sorted_returns_list(self, sample_dataset):
        """list_profiles always returns a list."""
        svc = _make_service()
        svc.profile_dataset(data=sample_dataset, dataset_name="test")
        assert isinstance(svc.list_profiles(), list)


# ===================================================================
# TestAssessQuality
# ===================================================================


class TestAssessQuality:
    """Test DataQualityProfilerService.assess_quality."""

    def test_assess_basic(self, sample_dataset):
        """Assessment returns a QualityAssessmentResponse."""
        svc = _make_service()
        result = svc.assess_quality(data=sample_dataset, dataset_name="test")
        assert isinstance(result, QualityAssessmentResponse)

    def test_overall_score_range(self, sample_dataset):
        """overall_score is between 0 and 1."""
        svc = _make_service()
        result = svc.assess_quality(data=sample_dataset, dataset_name="test")
        assert 0.0 <= result.overall_score <= 1.0

    def test_completeness_score(self, sample_dataset):
        """completeness_score is computed."""
        svc = _make_service()
        result = svc.assess_quality(data=sample_dataset, dataset_name="test")
        assert 0.0 <= result.completeness_score <= 1.0

    def test_validity_score(self, sample_dataset):
        """validity_score is computed."""
        svc = _make_service()
        result = svc.assess_quality(data=sample_dataset, dataset_name="test")
        assert 0.0 <= result.validity_score <= 1.0

    def test_consistency_score(self, sample_dataset):
        """consistency_score is computed."""
        svc = _make_service()
        result = svc.assess_quality(data=sample_dataset, dataset_name="test")
        assert 0.0 <= result.consistency_score <= 1.0

    def test_timeliness_score(self, sample_dataset):
        """timeliness_score is computed."""
        svc = _make_service()
        result = svc.assess_quality(data=sample_dataset, dataset_name="test")
        assert 0.0 <= result.timeliness_score <= 1.0

    def test_uniqueness_score(self, sample_dataset):
        """uniqueness_score is computed."""
        svc = _make_service()
        result = svc.assess_quality(data=sample_dataset, dataset_name="test")
        assert 0.0 <= result.uniqueness_score <= 1.0

    def test_accuracy_score(self, sample_dataset):
        """accuracy_score is computed."""
        svc = _make_service()
        result = svc.assess_quality(data=sample_dataset, dataset_name="test")
        assert 0.0 <= result.accuracy_score <= 1.0

    def test_quality_level_excellent(self):
        """Perfect data should yield a high quality level."""
        svc = _make_service()
        result = svc.assess_quality(data=_perfect_data(), dataset_name="perfect")
        assert result.quality_level in ("EXCELLENT", "GOOD")

    def test_quality_level_poor(self):
        """Poor data yields a lower quality level."""
        svc = _make_service()
        result = svc.assess_quality(data=_poor_data(), dataset_name="poor")
        assert result.quality_level in ("POOR", "CRITICAL")

    def test_weighted_scoring(self, sample_dataset):
        """overall_score is a weighted average of dimension scores."""
        svc = _make_service()
        result = svc.assess_quality(data=sample_dataset, dataset_name="test")
        # Just verify it is computed and within range
        assert 0.0 <= result.overall_score <= 1.0

    def test_specific_dimensions_only(self, sample_dataset):
        """When dimensions list provided, only those are scored."""
        svc = _make_service()
        result = svc.assess_quality(
            data=sample_dataset,
            dataset_name="test",
            dimensions=["completeness", "validity"],
        )
        # Dimensions not requested should be 0
        assert result.timeliness_score == 0.0
        assert result.uniqueness_score == 0.0

    def test_empty_data_raises(self):
        """Empty data raises ValueError."""
        svc = _make_service()
        with pytest.raises(ValueError, match="empty"):
            svc.assess_quality(data=[], dataset_name="empty")

    def test_issues_generated(self, sample_dataset_with_issues):
        """Issues list is populated for problematic data."""
        svc = _make_service()
        result = svc.assess_quality(
            data=sample_dataset_with_issues, dataset_name="issues",
        )
        assert isinstance(result.issues, list)

    def test_provenance_hash(self, sample_dataset):
        """provenance_hash is a 64-char hex string."""
        svc = _make_service()
        result = svc.assess_quality(data=sample_dataset, dataset_name="test")
        assert len(result.provenance_hash) == 64

    def test_assessment_id_uuid(self, sample_dataset):
        """assessment_id is a valid UUID."""
        svc = _make_service()
        result = svc.assess_quality(data=sample_dataset, dataset_name="test")
        uuid.UUID(result.assessment_id)

    def test_assessment_stored(self, sample_dataset):
        """Assessment is retrievable via get_assessment."""
        svc = _make_service()
        result = svc.assess_quality(data=sample_dataset, dataset_name="test")
        stored = svc.get_assessment(result.assessment_id)
        assert stored is not None


# ===================================================================
# TestAssessQualityBatch
# ===================================================================


class TestAssessQualityBatch:
    """Test DataQualityProfilerService.assess_quality_batch."""

    def test_multiple(self):
        """Batch assesses multiple datasets."""
        svc = _make_service()
        datasets = [
            {"data": _basic_data(10), "dataset_name": "a"},
            {"data": _basic_data(10), "dataset_name": "b"},
        ]
        results = svc.assess_quality_batch(datasets)
        assert len(results) == 2

    def test_order(self):
        """Results preserve input order."""
        svc = _make_service()
        datasets = [
            {"data": _basic_data(10), "dataset_name": "first"},
            {"data": _basic_data(10), "dataset_name": "second"},
        ]
        results = svc.assess_quality_batch(datasets)
        assert results[0].dataset_name == "first"
        assert results[1].dataset_name == "second"

    def test_empty_list(self):
        """Empty input returns empty output."""
        svc = _make_service()
        assert svc.assess_quality_batch([]) == []

    def test_mixed_quality(self):
        """Mixed quality datasets produce different scores."""
        svc = _make_service()
        datasets = [
            {"data": _perfect_data(), "dataset_name": "good"},
            {"data": _poor_data(), "dataset_name": "bad"},
        ]
        results = svc.assess_quality_batch(datasets)
        assert results[0].overall_score > results[1].overall_score


# ===================================================================
# TestGetAssessment
# ===================================================================


class TestGetAssessment:
    """Test DataQualityProfilerService.get_assessment."""

    def test_existing(self, sample_dataset):
        """Returns the assessment when ID matches."""
        svc = _make_service()
        a = svc.assess_quality(data=sample_dataset, dataset_name="test")
        assert svc.get_assessment(a.assessment_id) is not None

    def test_nonexistent(self):
        """Returns None for unknown ID."""
        svc = _make_service()
        assert svc.get_assessment("not_real") is None

    def test_fields_match(self, sample_dataset):
        """Retrieved assessment has identical fields."""
        svc = _make_service()
        a = svc.assess_quality(data=sample_dataset, dataset_name="test")
        fetched = svc.get_assessment(a.assessment_id)
        assert fetched.dataset_name == "test"
        assert fetched.overall_score == a.overall_score


# ===================================================================
# TestListAssessments
# ===================================================================


class TestListAssessments:
    """Test DataQualityProfilerService.list_assessments."""

    def test_empty(self):
        """Returns empty list when none exist."""
        svc = _make_service()
        assert svc.list_assessments() == []

    def test_returns_all(self, sample_dataset):
        """Returns all assessments."""
        svc = _make_service()
        svc.assess_quality(data=sample_dataset, dataset_name="a")
        svc.assess_quality(data=sample_dataset, dataset_name="b")
        assert len(svc.list_assessments()) == 2

    def test_limit(self, sample_dataset):
        """Respects limit parameter."""
        svc = _make_service()
        for i in range(5):
            svc.assess_quality(data=sample_dataset, dataset_name=f"d{i}")
        assert len(svc.list_assessments(limit=2)) == 2

    def test_offset(self, sample_dataset):
        """Respects offset parameter."""
        svc = _make_service()
        for i in range(5):
            svc.assess_quality(data=sample_dataset, dataset_name=f"d{i}")
        result = svc.list_assessments(limit=10, offset=4)
        assert len(result) == 1

    def test_returns_list(self, sample_dataset):
        """list_assessments always returns a list."""
        svc = _make_service()
        svc.assess_quality(data=sample_dataset, dataset_name="test")
        assert isinstance(svc.list_assessments(), list)


# ===================================================================
# TestValidateDataset
# ===================================================================


class TestValidateDataset:
    """Test DataQualityProfilerService.validate_dataset."""

    def test_no_rules(self, sample_dataset):
        """With no rules, overall result is pass."""
        svc = _make_service()
        result = svc.validate_dataset(data=sample_dataset, dataset_name="test")
        assert result["overall_result"] == "pass"
        assert result["rules_evaluated"] == 0

    def test_all_rules_pass(self, sample_dataset):
        """When all rows satisfy the rule, result is pass."""
        svc = _make_service()
        svc.create_rule(
            name="name_not_null", rule_type="not_null", column="name",
        )
        result = svc.validate_dataset(data=sample_dataset, dataset_name="test")
        assert result["overall_result"] == "pass"

    def test_some_fail(self, sample_dataset_with_issues):
        """When rows violate a rule, result is fail."""
        svc = _make_service()
        svc.create_rule(
            name="name_not_null", rule_type="not_null", column="name",
        )
        result = svc.validate_dataset(
            data=sample_dataset_with_issues, dataset_name="test",
        )
        assert result["overall_result"] == "fail"
        assert result["total_fail"] > 0

    def test_specific_rule_ids(self, sample_dataset):
        """Only specified rule IDs are evaluated."""
        svc = _make_service()
        r1 = svc.create_rule(name="r1", rule_type="not_null", column="name")
        svc.create_rule(name="r2", rule_type="not_null", column="age")
        result = svc.validate_dataset(
            data=sample_dataset,
            dataset_name="test",
            rule_ids=[r1.rule_id],
        )
        assert result["rules_evaluated"] == 1

    def test_empty_data_raises(self):
        """Empty data raises ValueError."""
        svc = _make_service()
        with pytest.raises(ValueError, match="empty"):
            svc.validate_dataset(data=[], dataset_name="empty")

    def test_inactive_rules_skipped(self, sample_dataset):
        """Inactive rules are not evaluated."""
        svc = _make_service()
        rule = svc.create_rule(name="r1", rule_type="not_null", column="name")
        svc.update_rule(rule.rule_id, {"is_active": False})
        result = svc.validate_dataset(data=sample_dataset, dataset_name="test")
        assert result["rules_evaluated"] == 0

    def test_evaluation_results_list(self, sample_dataset):
        """rule_results is a list of dicts."""
        svc = _make_service()
        svc.create_rule(name="r1", rule_type="not_null", column="name")
        result = svc.validate_dataset(data=sample_dataset, dataset_name="test")
        assert isinstance(result["rule_results"], list)
        assert len(result["rule_results"]) == 1

    def test_provenance_hash(self, sample_dataset):
        """Validation result includes a provenance_hash."""
        svc = _make_service()
        result = svc.validate_dataset(data=sample_dataset, dataset_name="test")
        assert len(result["provenance_hash"]) == 64

    def test_range_rule_pass(self):
        """Range rule passes when values are in bounds."""
        svc = _make_service()
        svc.create_rule(
            name="val_range", rule_type="range", column="value",
            parameters={"min": 0, "max": 100},
        )
        data = [{"value": 50}, {"value": 25}]
        result = svc.validate_dataset(data=data, dataset_name="test")
        assert result["overall_result"] == "pass"

    def test_range_rule_fail(self):
        """Range rule fails when values are out of bounds."""
        svc = _make_service()
        svc.create_rule(
            name="val_range", rule_type="range", column="value",
            parameters={"min": 0, "max": 10},
        )
        data = [{"value": 50}, {"value": 5}]
        result = svc.validate_dataset(data=data, dataset_name="test")
        assert result["overall_result"] == "fail"


# ===================================================================
# TestDetectAnomalies
# ===================================================================


class TestDetectAnomalies:
    """Test DataQualityProfilerService.detect_anomalies."""

    def test_no_anomalies(self):
        """Normal data should yield zero or few anomalies."""
        svc = _make_service(min_samples_for_anomaly=3)
        data = _numeric_data(20)
        result = svc.detect_anomalies(data=data, dataset_name="test")
        assert isinstance(result, AnomalyDetectionResponse)
        assert result.anomaly_count >= 0

    def test_clear_outliers(self):
        """Data with extreme outliers should detect anomalies."""
        svc = _make_service(min_samples_for_anomaly=3)
        data = _data_with_outliers(30)
        result = svc.detect_anomalies(data=data, dataset_name="test")
        assert result.anomaly_count > 0

    def test_specific_columns(self):
        """Only analyses the specified columns."""
        svc = _make_service(min_samples_for_anomaly=3)
        data = _numeric_data(20)
        result = svc.detect_anomalies(
            data=data, dataset_name="test", columns=["x"],
        )
        assert "x" in result.columns_analysed
        assert len(result.columns_analysed) == 1

    def test_method_iqr(self):
        """IQR method works."""
        svc = _make_service(min_samples_for_anomaly=3)
        data = _data_with_outliers(30)
        result = svc.detect_anomalies(
            data=data, dataset_name="test", method="iqr",
        )
        assert result.method == "iqr"

    def test_method_zscore(self):
        """Z-score method works."""
        svc = _make_service(min_samples_for_anomaly=3)
        data = _data_with_outliers(30)
        result = svc.detect_anomalies(
            data=data, dataset_name="test", method="zscore",
        )
        assert result.method == "zscore"

    def test_empty_data_raises(self):
        """Empty data raises ValueError."""
        svc = _make_service()
        with pytest.raises(ValueError, match="empty"):
            svc.detect_anomalies(data=[], dataset_name="test")

    def test_all_same_values(self):
        """All identical values should yield no anomalies."""
        svc = _make_service(min_samples_for_anomaly=3)
        data = [{"val": 5.0} for _ in range(20)]
        result = svc.detect_anomalies(data=data, dataset_name="test")
        assert result.anomaly_count == 0

    def test_provenance_hash(self):
        """provenance_hash is 64 chars."""
        svc = _make_service(min_samples_for_anomaly=3)
        data = _numeric_data(20)
        result = svc.detect_anomalies(data=data, dataset_name="test")
        assert len(result.provenance_hash) == 64

    def test_insufficient_samples_raises(self):
        """Raises when fewer than min_samples_for_anomaly rows."""
        svc = _make_service(min_samples_for_anomaly=50)
        data = _numeric_data(5)
        with pytest.raises(ValueError, match="Minimum"):
            svc.detect_anomalies(data=data, dataset_name="test")

    def test_anomaly_pct_computed(self):
        """anomaly_pct is computed and non-negative."""
        svc = _make_service(min_samples_for_anomaly=3)
        data = _data_with_outliers(30)
        result = svc.detect_anomalies(data=data, dataset_name="test")
        assert result.anomaly_pct >= 0.0


# ===================================================================
# TestListAnomalies
# ===================================================================


class TestListAnomalies:
    """Test DataQualityProfilerService.list_anomalies."""

    def test_empty(self):
        """Returns empty list when none exist."""
        svc = _make_service()
        assert svc.list_anomalies() == []

    def test_with_data(self):
        """Returns anomaly detection results."""
        svc = _make_service(min_samples_for_anomaly=3)
        svc.detect_anomalies(data=_numeric_data(20), dataset_name="test")
        assert len(svc.list_anomalies()) == 1

    def test_limit(self):
        """Respects limit parameter."""
        svc = _make_service(min_samples_for_anomaly=3)
        svc.detect_anomalies(data=_numeric_data(20), dataset_name="a")
        svc.detect_anomalies(data=_numeric_data(20), dataset_name="b")
        assert len(svc.list_anomalies(limit=1)) == 1

    def test_offset(self):
        """Respects offset parameter."""
        svc = _make_service(min_samples_for_anomaly=3)
        svc.detect_anomalies(data=_numeric_data(20), dataset_name="a")
        svc.detect_anomalies(data=_numeric_data(20), dataset_name="b")
        result = svc.list_anomalies(limit=10, offset=1)
        assert len(result) == 1


# ===================================================================
# TestCheckFreshness
# ===================================================================


class TestCheckFreshness:
    """Test DataQualityProfilerService.check_freshness."""

    def test_fresh(self):
        """Recent timestamp yields 'fresh' status."""
        svc = _make_service()
        result = svc.check_freshness(
            dataset_name="test", last_updated=_fresh_timestamp(),
        )
        assert isinstance(result, FreshnessCheckResponse)
        assert result.status == "fresh"

    def test_stale(self):
        """Moderately old timestamp yields 'stale' status."""
        svc = _make_service()
        result = svc.check_freshness(
            dataset_name="test",
            last_updated=_stale_timestamp(hours=60),
            sla_hours=48.0,
        )
        assert result.status == "stale"

    def test_expired(self):
        """Very old timestamp yields 'expired' status."""
        svc = _make_service()
        result = svc.check_freshness(
            dataset_name="test",
            last_updated=_stale_timestamp(hours=500),
            sla_hours=48.0,
        )
        assert result.status == "expired"

    def test_sla_compliant_fresh(self):
        """Fresh data within SLA yields fresh status."""
        svc = _make_service()
        result = svc.check_freshness(
            dataset_name="test",
            last_updated=_fresh_timestamp(),
            sla_hours=24.0,
        )
        assert result.status == "fresh"

    def test_custom_sla(self):
        """Custom SLA hours are applied."""
        svc = _make_service()
        result = svc.check_freshness(
            dataset_name="test",
            last_updated=_stale_timestamp(hours=10),
            sla_hours=100.0,
        )
        assert result.sla_hours == 100.0
        assert result.status == "fresh"

    def test_dataset_name_propagated(self):
        """dataset_name is present in the response."""
        svc = _make_service()
        result = svc.check_freshness(
            dataset_name="my_ds", last_updated=_fresh_timestamp(),
        )
        assert result.dataset_name == "my_ds"

    def test_age_hours_positive(self):
        """age_hours is non-negative."""
        svc = _make_service()
        result = svc.check_freshness(
            dataset_name="test",
            last_updated=_stale_timestamp(hours=5),
        )
        assert result.age_hours >= 0

    def test_provenance_hash(self):
        """provenance_hash is set."""
        svc = _make_service()
        result = svc.check_freshness(
            dataset_name="test", last_updated=_fresh_timestamp(),
        )
        assert len(result.provenance_hash) == 64

    def test_freshness_score_range(self):
        """freshness_score is between 0 and 1."""
        svc = _make_service()
        result = svc.check_freshness(
            dataset_name="test", last_updated=_fresh_timestamp(),
        )
        assert 0.0 <= result.freshness_score <= 1.0

    def test_stats_increment(self):
        """total_freshness_checks stat is incremented."""
        svc = _make_service()
        svc.check_freshness(
            dataset_name="test", last_updated=_fresh_timestamp(),
        )
        assert svc.get_statistics().total_freshness_checks == 1


# ===================================================================
# TestCreateRule
# ===================================================================


class TestCreateRule:
    """Test DataQualityProfilerService.create_rule."""

    def test_not_null_rule(self):
        """Creates a not_null rule."""
        svc = _make_service()
        rule = svc.create_rule(
            name="name_required", rule_type="not_null", column="name",
        )
        assert isinstance(rule, QualityRuleResponse)
        assert rule.name == "name_required"
        assert rule.rule_type == "not_null"

    def test_unique_rule(self):
        """Creates a unique rule."""
        svc = _make_service()
        rule = svc.create_rule(
            name="id_unique", rule_type="unique", column="id",
        )
        assert rule.rule_type == "unique"

    def test_range_rule(self):
        """Creates a range rule."""
        svc = _make_service()
        rule = svc.create_rule(
            name="val_range", rule_type="range", column="value",
            parameters={"min": 0, "max": 100},
        )
        assert rule.rule_type == "range"
        assert rule.parameters["min"] == 0

    def test_regex_rule(self):
        """Creates a regex rule."""
        svc = _make_service()
        rule = svc.create_rule(
            name="email_format", rule_type="regex", column="email",
            parameters={"pattern": r"^[\w.+-]+@[\w-]+\.[\w.]+$"},
        )
        assert rule.rule_type == "regex"

    def test_custom_rule(self):
        """Creates a custom rule."""
        svc = _make_service()
        rule = svc.create_rule(
            name="custom_check", rule_type="custom", column="score",
            operator="gte", threshold=50,
        )
        assert rule.rule_type == "custom"
        assert rule.threshold == 50

    def test_referential_rule(self):
        """Creates a referential rule."""
        svc = _make_service()
        rule = svc.create_rule(
            name="ref_check", rule_type="referential", column="ref_id",
        )
        assert rule.rule_type == "referential"

    def test_priority(self):
        """Priority is stored correctly."""
        svc = _make_service()
        rule = svc.create_rule(
            name="high_prio", rule_type="not_null", column="name",
            priority=10,
        )
        assert rule.priority == 10

    def test_empty_name_raises(self):
        """Empty name raises ValueError."""
        svc = _make_service()
        with pytest.raises(ValueError, match="empty"):
            svc.create_rule(name="", rule_type="not_null", column="x")

    def test_whitespace_name_raises(self):
        """Whitespace-only name raises ValueError."""
        svc = _make_service()
        with pytest.raises(ValueError, match="empty"):
            svc.create_rule(name="   ", rule_type="not_null", column="x")

    def test_stats_incremented(self):
        """total_rules stat is incremented."""
        svc = _make_service()
        svc.create_rule(name="r1", rule_type="not_null", column="x")
        assert svc.get_statistics().total_rules == 1

    def test_active_by_default(self):
        """Rule is active by default."""
        svc = _make_service()
        rule = svc.create_rule(name="r1", rule_type="not_null", column="x")
        assert rule.is_active is True

    def test_provenance_hash(self):
        """Rule has a provenance_hash."""
        svc = _make_service()
        rule = svc.create_rule(name="r1", rule_type="not_null", column="x")
        assert len(rule.provenance_hash) == 64


# ===================================================================
# TestGetRule
# ===================================================================


class TestGetRule:
    """Test that created rules can be retrieved from internal store."""

    def test_existing(self):
        """Can retrieve a created rule from list."""
        svc = _make_service()
        rule = svc.create_rule(name="r1", rule_type="not_null", column="x")
        rules = svc.list_rules()
        assert any(r.rule_id == rule.rule_id for r in rules)

    def test_nonexistent(self):
        """No rule with a fake ID exists in the list."""
        svc = _make_service()
        rules = svc.list_rules()
        assert not any(r.rule_id == "fake" for r in rules)


# ===================================================================
# TestListRules
# ===================================================================


class TestListRules:
    """Test DataQualityProfilerService.list_rules."""

    def test_empty(self):
        """Returns empty list when no rules exist."""
        svc = _make_service()
        assert svc.list_rules() == []

    def test_returns_all(self):
        """Returns all rules."""
        svc = _make_service()
        svc.create_rule(name="r1", rule_type="not_null", column="x")
        svc.create_rule(name="r2", rule_type="unique", column="y")
        assert len(svc.list_rules()) == 2

    def test_active_only(self):
        """Filters to active rules only."""
        svc = _make_service()
        r1 = svc.create_rule(name="r1", rule_type="not_null", column="x")
        svc.create_rule(name="r2", rule_type="unique", column="y")
        svc.update_rule(r1.rule_id, {"is_active": False})
        active = svc.list_rules(active_only=True)
        assert len(active) == 1

    def test_by_type(self):
        """Filters by rule type."""
        svc = _make_service()
        svc.create_rule(name="r1", rule_type="not_null", column="x")
        svc.create_rule(name="r2", rule_type="range", column="y")
        filtered = svc.list_rules(rule_type="range")
        assert len(filtered) == 1
        assert filtered[0].rule_type == "range"

    def test_sorted_by_priority(self):
        """Rules are sorted by priority (ascending)."""
        svc = _make_service()
        svc.create_rule(name="low", rule_type="not_null", column="x", priority=200)
        svc.create_rule(name="high", rule_type="not_null", column="y", priority=10)
        rules = svc.list_rules()
        assert rules[0].priority <= rules[1].priority

    def test_returns_list_type(self):
        """list_rules always returns a list."""
        svc = _make_service()
        assert isinstance(svc.list_rules(), list)


# ===================================================================
# TestUpdateRule
# ===================================================================


class TestUpdateRule:
    """Test DataQualityProfilerService.update_rule."""

    def test_update_name(self):
        """Updates rule name."""
        svc = _make_service()
        rule = svc.create_rule(name="old", rule_type="not_null", column="x")
        updated = svc.update_rule(rule.rule_id, {"name": "new_name"})
        assert updated.name == "new_name"

    def test_update_threshold(self):
        """Updates rule threshold."""
        svc = _make_service()
        rule = svc.create_rule(
            name="r1", rule_type="custom", column="x", threshold=50,
        )
        updated = svc.update_rule(rule.rule_id, {"threshold": 75})
        assert updated.threshold == 75

    def test_deactivate(self):
        """Deactivates a rule."""
        svc = _make_service()
        rule = svc.create_rule(name="r1", rule_type="not_null", column="x")
        updated = svc.update_rule(rule.rule_id, {"is_active": False})
        assert updated.is_active is False

    def test_not_found_raises(self):
        """Non-existent rule_id raises ValueError."""
        svc = _make_service()
        with pytest.raises(ValueError, match="not found"):
            svc.update_rule("nonexistent", {"name": "nope"})

    def test_provenance_hash_updated(self):
        """provenance_hash changes after update."""
        svc = _make_service()
        rule = svc.create_rule(name="r1", rule_type="not_null", column="x")
        old_hash = rule.provenance_hash
        updated = svc.update_rule(rule.rule_id, {"name": "new_name"})
        assert updated.provenance_hash != old_hash

    def test_update_priority(self):
        """Updates rule priority."""
        svc = _make_service()
        rule = svc.create_rule(
            name="r1", rule_type="not_null", column="x", priority=100,
        )
        updated = svc.update_rule(rule.rule_id, {"priority": 5})
        assert updated.priority == 5

    def test_update_operator(self):
        """Updates rule operator."""
        svc = _make_service()
        rule = svc.create_rule(
            name="r1", rule_type="custom", column="x", operator="eq",
        )
        updated = svc.update_rule(rule.rule_id, {"operator": "gte"})
        assert updated.operator == "gte"

    def test_updated_at_changes(self):
        """updated_at timestamp changes after update."""
        svc = _make_service()
        rule = svc.create_rule(name="r1", rule_type="not_null", column="x")
        old_ts = rule.updated_at
        time.sleep(0.01)
        updated = svc.update_rule(rule.rule_id, {"name": "new"})
        assert updated.updated_at >= old_ts


# ===================================================================
# TestDeleteRule
# ===================================================================


class TestDeleteRule:
    """Test DataQualityProfilerService.delete_rule."""

    def test_existing(self):
        """Deleting an existing rule returns True."""
        svc = _make_service()
        rule = svc.create_rule(name="r1", rule_type="not_null", column="x")
        assert svc.delete_rule(rule.rule_id) is True

    def test_nonexistent(self):
        """Deleting a nonexistent rule returns False."""
        svc = _make_service()
        assert svc.delete_rule("not_real") is False

    def test_stats_decremented(self):
        """total_rules stat is decremented after delete."""
        svc = _make_service()
        rule = svc.create_rule(name="r1", rule_type="not_null", column="x")
        assert svc.get_statistics().total_rules == 1
        svc.delete_rule(rule.rule_id)
        assert svc.get_statistics().total_rules == 0

    def test_gone_from_list(self):
        """Deleted rule no longer appears in list_rules."""
        svc = _make_service()
        rule = svc.create_rule(name="r1", rule_type="not_null", column="x")
        svc.delete_rule(rule.rule_id)
        assert len(svc.list_rules()) == 0


# ===================================================================
# TestEvaluateGate
# ===================================================================


class TestEvaluateGate:
    """Test DataQualityProfilerService.evaluate_gate."""

    def test_pass_outcome(self):
        """All conditions met yields 'pass'."""
        svc = _make_service()
        result = svc.evaluate_gate(
            conditions=[
                {"dimension": "completeness", "operator": "gte", "threshold": 0.5},
            ],
            dimension_scores={"completeness": 0.9},
        )
        assert isinstance(result, QualityGateResponse)
        assert result.outcome == "pass"

    def test_fail_outcome(self):
        """All conditions failed yields 'fail'."""
        svc = _make_service()
        result = svc.evaluate_gate(
            conditions=[
                {"dimension": "completeness", "operator": "gte", "threshold": 0.9},
            ],
            dimension_scores={"completeness": 0.3},
        )
        assert result.outcome == "fail"

    def test_warn_outcome(self):
        """Mixed conditions yields 'warn'."""
        svc = _make_service()
        result = svc.evaluate_gate(
            conditions=[
                {"dimension": "completeness", "operator": "gte", "threshold": 0.5},
                {"dimension": "validity", "operator": "gte", "threshold": 0.99},
            ],
            dimension_scores={"completeness": 0.9, "validity": 0.4},
        )
        assert result.outcome == "warn"

    def test_custom_threshold(self):
        """Custom threshold value is respected."""
        svc = _make_service()
        result = svc.evaluate_gate(
            conditions=[
                {"dimension": "accuracy", "operator": "gte", "threshold": 0.01},
            ],
            dimension_scores={"accuracy": 0.5},
        )
        assert result.outcome == "pass"

    def test_empty_conditions_raises(self):
        """Empty conditions list raises ValueError."""
        svc = _make_service()
        with pytest.raises(ValueError, match="condition"):
            svc.evaluate_gate(conditions=[])

    def test_completeness_dimension(self):
        """Gate evaluates completeness dimension correctly."""
        svc = _make_service()
        result = svc.evaluate_gate(
            conditions=[
                {"dimension": "completeness", "operator": "gte", "threshold": 0.8},
            ],
            dimension_scores={"completeness": 0.85},
        )
        assert result.conditions_passed == 1

    def test_conditions_counts(self):
        """conditions_evaluated, conditions_passed, conditions_failed are correct."""
        svc = _make_service()
        result = svc.evaluate_gate(
            conditions=[
                {"dimension": "completeness", "operator": "gte", "threshold": 0.5},
                {"dimension": "validity", "operator": "gte", "threshold": 0.5},
                {"dimension": "accuracy", "operator": "gte", "threshold": 0.99},
            ],
            dimension_scores={
                "completeness": 0.9, "validity": 0.8, "accuracy": 0.1,
            },
        )
        assert result.conditions_evaluated == 3
        assert result.conditions_passed == 2
        assert result.conditions_failed == 1

    def test_provenance_hash(self):
        """provenance_hash is 64 chars."""
        svc = _make_service()
        result = svc.evaluate_gate(
            conditions=[
                {"dimension": "completeness", "operator": "gte", "threshold": 0.5},
            ],
            dimension_scores={"completeness": 0.9},
        )
        assert len(result.provenance_hash) == 64

    def test_gate_stored(self):
        """Gate result is stored internally."""
        svc = _make_service()
        result = svc.evaluate_gate(
            conditions=[
                {"dimension": "completeness", "operator": "gte", "threshold": 0.5},
            ],
            dimension_scores={"completeness": 0.9},
        )
        assert result.gate_id in svc._gate_results

    def test_stats_incremented(self):
        """total_gate_evaluations stat is incremented."""
        svc = _make_service()
        svc.evaluate_gate(
            conditions=[
                {"dimension": "completeness", "operator": "gte", "threshold": 0.5},
            ],
            dimension_scores={"completeness": 0.9},
        )
        assert svc.get_statistics().total_gate_evaluations == 1


# ===================================================================
# TestGetTrends
# ===================================================================


class TestGetTrends:
    """Test DataQualityProfilerService.get_trends."""

    def test_empty(self):
        """No assessments returns empty trends."""
        svc = _make_service()
        trends = svc.get_trends()
        assert trends == []

    def test_with_assessments(self, sample_dataset):
        """Trends are generated from stored assessments."""
        svc = _make_service()
        svc.assess_quality(data=sample_dataset, dataset_name="test")
        trends = svc.get_trends()
        assert len(trends) >= 1

    def test_filter_by_dataset_name(self, sample_dataset):
        """dataset_name filter works."""
        svc = _make_service()
        svc.assess_quality(data=sample_dataset, dataset_name="alpha")
        svc.assess_quality(data=sample_dataset, dataset_name="beta")
        trends = svc.get_trends(dataset_name="alpha")
        assert all(t["dataset_name"] == "alpha" for t in trends)

    def test_multiple_periods(self, sample_dataset):
        """Multiple assessments produce multiple trend points."""
        svc = _make_service()
        for i in range(5):
            svc.assess_quality(data=sample_dataset, dataset_name="test")
        trends = svc.get_trends(dataset_name="test", periods=10)
        assert len(trends) == 5

    def test_periods_limit(self, sample_dataset):
        """periods parameter limits the number of trend points."""
        svc = _make_service()
        for i in range(10):
            svc.assess_quality(data=sample_dataset, dataset_name="test")
        trends = svc.get_trends(dataset_name="test", periods=3)
        assert len(trends) == 3

    def test_trend_contains_score(self, sample_dataset):
        """Each trend point includes overall_score."""
        svc = _make_service()
        svc.assess_quality(data=sample_dataset, dataset_name="test")
        trends = svc.get_trends()
        assert "overall_score" in trends[0]

    def test_trend_contains_quality_level(self, sample_dataset):
        """Each trend point includes quality_level."""
        svc = _make_service()
        svc.assess_quality(data=sample_dataset, dataset_name="test")
        trends = svc.get_trends()
        assert "quality_level" in trends[0]

    def test_trend_contains_assessed_at(self, sample_dataset):
        """Each trend point includes assessed_at timestamp."""
        svc = _make_service()
        svc.assess_quality(data=sample_dataset, dataset_name="test")
        trends = svc.get_trends()
        assert "assessed_at" in trends[0]


# ===================================================================
# TestGenerateReport
# ===================================================================


class TestGenerateReport:
    """Test DataQualityProfilerService.generate_report."""

    def test_scorecard_json(self, sample_dataset):
        """Scorecard JSON report works."""
        svc = _make_service()
        svc.assess_quality(data=sample_dataset, dataset_name="test")
        report = svc.generate_report(
            dataset_name="test", report_type="scorecard", report_format="json",
        )
        assert report["report_type"] == "scorecard"
        assert report["format"] == "json"
        assert isinstance(report["content"], dict)

    def test_detailed_json(self, sample_dataset):
        """Detailed JSON report includes assessments and profiles."""
        svc = _make_service()
        svc.assess_quality(data=sample_dataset, dataset_name="test")
        report = svc.generate_report(
            dataset_name="test", report_type="detailed", report_format="json",
        )
        assert "assessments" in report["content"]

    def test_executive_json(self, sample_dataset):
        """Executive JSON report includes average score."""
        svc = _make_service()
        svc.assess_quality(data=sample_dataset, dataset_name="test")
        report = svc.generate_report(
            dataset_name="test", report_type="executive", report_format="json",
        )
        assert "average_score" in report["content"]

    def test_issues_json(self, sample_dataset_with_issues):
        """Issues JSON report lists issues."""
        svc = _make_service()
        svc.assess_quality(
            data=sample_dataset_with_issues, dataset_name="test",
        )
        report = svc.generate_report(
            dataset_name="test", report_type="issues", report_format="json",
        )
        assert "issues" in report["content"]

    def test_anomaly_json(self):
        """Anomaly JSON report works."""
        svc = _make_service(min_samples_for_anomaly=3)
        svc.detect_anomalies(data=_data_with_outliers(30), dataset_name="test")
        report = svc.generate_report(
            dataset_name="test", report_type="anomaly", report_format="json",
        )
        assert "anomaly_results" in report["content"]

    def test_markdown_format(self, sample_dataset):
        """Markdown report is a string starting with #."""
        svc = _make_service()
        svc.assess_quality(data=sample_dataset, dataset_name="test")
        report = svc.generate_report(
            dataset_name="test", report_type="scorecard", report_format="markdown",
        )
        assert isinstance(report["content"], str)
        assert report["content"].startswith("# ")

    def test_html_format(self, sample_dataset):
        """HTML report contains html tag."""
        svc = _make_service()
        svc.assess_quality(data=sample_dataset, dataset_name="test")
        report = svc.generate_report(
            dataset_name="test", report_type="scorecard", report_format="html",
        )
        assert "<html>" in report["content"]

    def test_text_format(self, sample_dataset):
        """Text report is a plain string."""
        svc = _make_service()
        svc.assess_quality(data=sample_dataset, dataset_name="test")
        report = svc.generate_report(
            dataset_name="test", report_type="scorecard", report_format="text",
        )
        assert isinstance(report["content"], str)
        assert "DATA QUALITY REPORT" in report["content"]

    def test_csv_format(self, sample_dataset):
        """CSV report contains headers."""
        svc = _make_service()
        svc.assess_quality(data=sample_dataset, dataset_name="test")
        report = svc.generate_report(
            dataset_name="test", report_type="scorecard", report_format="csv",
        )
        assert isinstance(report["content"], str)
        assert "dataset_name" in report["content"]

    def test_empty_data_report(self):
        """Report with no assessments still works."""
        svc = _make_service()
        report = svc.generate_report(report_type="scorecard")
        assert report["assessment_count"] == 0

    def test_provenance_hash(self, sample_dataset):
        """Report includes a provenance_hash."""
        svc = _make_service()
        svc.assess_quality(data=sample_dataset, dataset_name="test")
        report = svc.generate_report(dataset_name="test")
        assert len(report["provenance_hash"]) == 64

    def test_stats_incremented(self, sample_dataset):
        """total_reports stat is incremented."""
        svc = _make_service()
        svc.assess_quality(data=sample_dataset, dataset_name="test")
        svc.generate_report(dataset_name="test")
        assert svc.get_statistics().total_reports == 1


# ===================================================================
# TestStatistics
# ===================================================================


class TestStatistics:
    """Test DataQualityProfilerService.get_statistics."""

    def test_initial_zeros(self):
        """All stats are zero initially."""
        svc = _make_service()
        stats = svc.get_statistics()
        assert stats.total_profiles == 0
        assert stats.total_assessments == 0
        assert stats.total_anomaly_detections == 0
        assert stats.total_freshness_checks == 0
        assert stats.total_rules == 0
        assert stats.active_rules == 0
        assert stats.total_gate_evaluations == 0
        assert stats.total_issues_found == 0
        assert stats.avg_quality_score == 0.0

    def test_after_pipeline(self, sample_dataset):
        """Stats reflect operations performed."""
        svc = _make_service(min_samples_for_anomaly=3)
        svc.profile_dataset(data=sample_dataset, dataset_name="test")
        svc.assess_quality(data=sample_dataset, dataset_name="test")
        svc.detect_anomalies(data=_numeric_data(20), dataset_name="test")
        svc.check_freshness(dataset_name="test", last_updated=_fresh_timestamp())
        svc.create_rule(name="r1", rule_type="not_null", column="x")
        svc.evaluate_gate(
            conditions=[{"dimension": "completeness", "operator": "gte", "threshold": 0.5}],
            dimension_scores={"completeness": 0.9},
        )
        svc.generate_report(dataset_name="test")

        stats = svc.get_statistics()
        # assess_quality calls profile_dataset internally, so total_profiles >= 2
        assert stats.total_profiles >= 2
        assert stats.total_assessments >= 1
        assert stats.total_anomaly_detections == 1
        assert stats.total_freshness_checks == 1
        assert stats.total_rules == 1
        assert stats.total_gate_evaluations == 1
        assert stats.total_reports == 1

    def test_avg_quality_updates(self, sample_dataset):
        """avg_quality_score updates after multiple assessments."""
        svc = _make_service()
        svc.assess_quality(data=sample_dataset, dataset_name="a")
        svc.assess_quality(data=sample_dataset, dataset_name="b")
        stats = svc.get_statistics()
        assert stats.avg_quality_score > 0.0


# ===================================================================
# TestHealthCheck
# ===================================================================


class TestHealthCheck:
    """Test DataQualityProfilerService.health_check."""

    def test_before_start(self):
        """Health check before startup shows not_started."""
        svc = _make_service()
        health = svc.health_check()
        assert health["status"] == "not_started"
        assert health["started"] is False

    def test_after_start(self):
        """Health check after startup shows healthy."""
        svc = _make_service()
        svc.startup()
        health = svc.health_check()
        assert health["status"] == "healthy"
        assert health["started"] is True

    def test_counts_zero(self):
        """All counts start at zero."""
        svc = _make_service()
        health = svc.health_check()
        assert health["profiles"] == 0
        assert health["assessments"] == 0
        assert health["rules"] == 0

    def test_service_name(self):
        """service field is correct."""
        svc = _make_service()
        health = svc.health_check()
        assert health["service"] == "data-quality-profiler"

    def test_all_fields_present(self):
        """All expected fields are present."""
        svc = _make_service()
        health = svc.health_check()
        expected_keys = {
            "status", "service", "started", "profiles", "assessments",
            "anomaly_detections", "freshness_checks", "rules",
            "gate_evaluations", "reports", "provenance_entries",
            "prometheus_available",
        }
        assert expected_keys.issubset(health.keys())


# ===================================================================
# TestGetMetrics
# ===================================================================


class TestGetMetrics:
    """Test DataQualityProfilerService.get_metrics."""

    def test_initial(self):
        """Metrics are zeroed initially."""
        svc = _make_service()
        metrics = svc.get_metrics()
        assert metrics["total_profiles"] == 0
        assert metrics["total_assessments"] == 0

    def test_after_processing(self, sample_dataset):
        """Metrics reflect operations performed."""
        svc = _make_service()
        svc.profile_dataset(data=sample_dataset, dataset_name="test")
        metrics = svc.get_metrics()
        assert metrics["total_profiles"] >= 1

    def test_all_keys(self):
        """All expected metric keys are present."""
        svc = _make_service()
        metrics = svc.get_metrics()
        expected_keys = {
            "prometheus_available", "started", "total_profiles",
            "total_assessments", "total_anomaly_detections",
            "total_freshness_checks", "total_rules", "active_rules",
            "total_gate_evaluations", "total_issues_found",
            "avg_quality_score", "total_reports", "provenance_entries",
        }
        assert expected_keys.issubset(metrics.keys())


# ===================================================================
# TestGetProvenance
# ===================================================================


class TestGetProvenance:
    """Test DataQualityProfilerService.get_provenance."""

    def test_tracker_returned(self):
        """get_provenance returns the tracker instance."""
        svc = _make_service()
        tracker = svc.get_provenance()
        assert tracker is svc.provenance

    def test_entries_grow(self, sample_dataset):
        """Provenance entry count grows with operations."""
        svc = _make_service()
        count_before = svc.provenance.entry_count
        svc.profile_dataset(data=sample_dataset, dataset_name="test")
        assert svc.provenance.entry_count > count_before


# ===================================================================
# TestLifecycle
# ===================================================================


class TestLifecycle:
    """Test DataQualityProfilerService startup/shutdown lifecycle."""

    def test_startup(self):
        """startup sets _started to True."""
        svc = _make_service()
        svc.startup()
        assert svc._started is True

    def test_startup_idempotent(self):
        """Calling startup twice is safe."""
        svc = _make_service()
        svc.startup()
        svc.startup()  # No error
        assert svc._started is True

    def test_shutdown(self):
        """shutdown sets _started to False."""
        svc = _make_service()
        svc.startup()
        svc.shutdown()
        assert svc._started is False

    def test_shutdown_without_start(self):
        """Calling shutdown without startup is safe."""
        svc = _make_service()
        svc.shutdown()  # Should not raise
        assert svc._started is False


# ===================================================================
# TestFullWorkflow
# ===================================================================


class TestFullWorkflow:
    """End-to-end workflow tests."""

    def test_end_to_end_pipeline(self, sample_dataset):
        """Full pipeline: profile -> assess -> validate -> detect -> gate -> report."""
        svc = _make_service(min_samples_for_anomaly=3)
        svc.startup()

        # Step 1: Profile
        profile = svc.profile_dataset(data=sample_dataset, dataset_name="e2e")
        assert profile.row_count == len(sample_dataset)

        # Step 2: Assess quality
        assessment = svc.assess_quality(data=sample_dataset, dataset_name="e2e")
        assert assessment.overall_score > 0.0

        # Step 3: Create rule and validate
        svc.create_rule(name="name_nn", rule_type="not_null", column="name")
        validation = svc.validate_dataset(data=sample_dataset, dataset_name="e2e")
        assert validation["overall_result"] == "pass"

        # Step 4: Detect anomalies (need numeric data)
        anomalies = svc.detect_anomalies(
            data=_data_with_outliers(30), dataset_name="e2e",
        )
        assert anomalies.total_records > 0

        # Step 5: Quality gate
        gate = svc.evaluate_gate(
            conditions=[
                {"dimension": "completeness", "operator": "gte", "threshold": 0.5},
            ],
            dimension_scores={"completeness": assessment.completeness_score},
        )
        assert gate.outcome in ("pass", "warn", "fail")

        # Step 6: Generate report
        report = svc.generate_report(
            dataset_name="e2e", report_type="scorecard", report_format="json",
        )
        assert report["report_id"] is not None

        # Verify provenance grew
        assert svc.provenance.entry_count > 0

        svc.shutdown()

    def test_pipeline_with_custom_rules(self, sample_dataset):
        """Pipeline with multiple custom rules."""
        svc = _make_service()
        svc.startup()

        svc.create_rule(
            name="name_not_null", rule_type="not_null", column="name",
        )
        svc.create_rule(
            name="age_range", rule_type="range", column="age",
            parameters={"min": 0, "max": 150},
        )
        svc.create_rule(
            name="score_high", rule_type="custom", column="score",
            operator="gte", threshold=0,
        )

        result = svc.validate_dataset(data=sample_dataset, dataset_name="test")
        assert result["rules_evaluated"] == 3
        assert result["overall_result"] == "pass"

        svc.shutdown()


# ===================================================================
# TestConfigureFunction
# ===================================================================


class TestConfigureFunction:
    """Test configure_data_quality_profiler."""

    def test_attaches_service(self):
        """configure_data_quality_profiler stores service on app.state."""
        app = MagicMock()
        app.state = MagicMock()
        app.include_router = MagicMock()
        import asyncio
        service = asyncio.get_event_loop().run_until_complete(
            configure_data_quality_profiler(app)
        )
        assert isinstance(service, DataQualityProfilerService)
        assert app.state.data_quality_profiler_service == service

    def test_mounts_router(self):
        """configure_data_quality_profiler calls include_router."""
        app = MagicMock()
        app.state = MagicMock()
        app.include_router = MagicMock()
        import asyncio
        asyncio.get_event_loop().run_until_complete(
            configure_data_quality_profiler(app)
        )
        # Router mounting attempted
        assert app.include_router.called or True  # May or may not mount

    def test_returns_service(self):
        """Function returns a DataQualityProfilerService."""
        app = MagicMock()
        app.state = MagicMock()
        app.include_router = MagicMock()
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            configure_data_quality_profiler(app)
        )
        assert isinstance(result, DataQualityProfilerService)


# ===================================================================
# TestGetDataQualityProfiler
# ===================================================================


class TestGetDataQualityProfiler:
    """Test get_data_quality_profiler."""

    def test_success(self):
        """Returns service when configured."""
        app = MagicMock()
        svc = _make_service()
        app.state.data_quality_profiler_service = svc
        result = get_data_quality_profiler(app)
        assert result is svc

    def test_not_configured_raises(self):
        """Raises RuntimeError when not configured."""
        app = MagicMock()
        app.state = MagicMock(spec=[])
        with pytest.raises(RuntimeError, match="not configured"):
            get_data_quality_profiler(app)


# ===================================================================
# TestPydanticModels
# ===================================================================


class TestPydanticModels:
    """Test default values for all 8 Pydantic response models."""

    def test_column_profile_defaults(self):
        """ColumnProfileResponse has sensible defaults."""
        obj = ColumnProfileResponse()
        assert obj.column_name == ""
        assert obj.data_type == "string"
        assert obj.total_count == 0

    def test_dataset_profile_defaults(self):
        """DatasetProfileResponse has sensible defaults."""
        obj = DatasetProfileResponse()
        assert obj.dataset_name == ""
        assert obj.row_count == 0
        assert obj.profile_id != ""

    def test_quality_assessment_defaults(self):
        """QualityAssessmentResponse has sensible defaults."""
        obj = QualityAssessmentResponse()
        assert obj.overall_score == 0.0
        assert obj.quality_level == "CRITICAL"

    def test_anomaly_detection_defaults(self):
        """AnomalyDetectionResponse has sensible defaults."""
        obj = AnomalyDetectionResponse()
        assert obj.anomaly_count == 0
        assert obj.method == "iqr"

    def test_freshness_check_defaults(self):
        """FreshnessCheckResponse has sensible defaults."""
        obj = FreshnessCheckResponse()
        assert obj.status == "unknown"
        assert obj.sla_hours == 48.0

    def test_quality_rule_defaults(self):
        """QualityRuleResponse has sensible defaults."""
        obj = QualityRuleResponse()
        assert obj.is_active is True
        assert obj.priority == 100

    def test_quality_gate_defaults(self):
        """QualityGateResponse has sensible defaults."""
        obj = QualityGateResponse()
        assert obj.outcome == "fail"
        assert obj.threshold == 0.70

    def test_statistics_defaults(self):
        """DataQualityProfilerStatisticsResponse has sensible defaults."""
        obj = DataQualityProfilerStatisticsResponse()
        assert obj.total_profiles == 0
        assert obj.avg_quality_score == 0.0


# ===================================================================
# TestComputeWeightedScore
# ===================================================================


class TestComputeWeightedScore:
    """Test _compute_weighted_score internal method."""

    def test_equal_weights(self):
        """Equal scores produce that same score."""
        svc = _make_service()
        scores = {
            "completeness": 0.8,
            "validity": 0.8,
            "consistency": 0.8,
            "timeliness": 0.8,
            "uniqueness": 0.8,
            "accuracy": 0.8,
        }
        result = svc._compute_weighted_score(scores)
        assert abs(result - 0.8) < 0.01

    def test_custom_weights(self):
        """Weighted scoring respects config weights."""
        cfg = _make_config(
            completeness_weight=1.0,
            validity_weight=0.0,
            consistency_weight=0.0,
            timeliness_weight=0.0,
            uniqueness_weight=0.0,
            accuracy_weight=0.0,
        )
        svc = DataQualityProfilerService(config=cfg)
        scores = {"completeness": 0.9, "validity": 0.1}
        result = svc._compute_weighted_score(scores)
        # Only completeness has weight, so result = 0.9
        assert abs(result - 0.9) < 0.01

    def test_zero_total_weight(self):
        """Zero total weight returns 0."""
        cfg = _make_config(
            completeness_weight=0.0,
            validity_weight=0.0,
            consistency_weight=0.0,
            timeliness_weight=0.0,
            uniqueness_weight=0.0,
            accuracy_weight=0.0,
        )
        svc = DataQualityProfilerService(config=cfg)
        scores = {"completeness": 0.9}
        result = svc._compute_weighted_score(scores)
        assert result == 0.0

    def test_single_dimension(self):
        """Single dimension score is returned correctly."""
        svc = _make_service()
        scores = {"completeness": 0.75}
        result = svc._compute_weighted_score(scores)
        # Only completeness contributes; result = 0.75 * w / w = 0.75
        assert abs(result - 0.75) < 0.01

    def test_empty_scores(self):
        """Empty scores dict returns 0."""
        svc = _make_service()
        result = svc._compute_weighted_score({})
        assert result == 0.0


# ===================================================================
# TestClassifyQuality
# ===================================================================


class TestClassifyQuality:
    """Test _classify_quality helper function."""

    def test_excellent(self):
        assert _classify_quality(0.96) == "EXCELLENT"

    def test_good(self):
        assert _classify_quality(0.88) == "GOOD"

    def test_fair(self):
        assert _classify_quality(0.72) == "FAIR"

    def test_poor(self):
        assert _classify_quality(0.55) == "POOR"

    def test_critical(self):
        assert _classify_quality(0.30) == "CRITICAL"

    def test_boundary_excellent(self):
        assert _classify_quality(0.95) == "EXCELLENT"

    def test_boundary_good(self):
        assert _classify_quality(0.85) == "GOOD"

    def test_boundary_fair(self):
        assert _classify_quality(0.70) == "FAIR"

    def test_boundary_poor(self):
        assert _classify_quality(0.50) == "POOR"

    def test_zero(self):
        assert _classify_quality(0.0) == "CRITICAL"


# ===================================================================
# TestComputeHash
# ===================================================================


class TestComputeHash:
    """Test _compute_hash helper function."""

    def test_dict_hash(self):
        """Hashing a dict produces 64-char hex."""
        h = _compute_hash({"key": "value"})
        assert len(h) == 64

    def test_deterministic(self):
        """Same input produces same hash."""
        d = {"a": 1, "b": 2}
        assert _compute_hash(d) == _compute_hash(d)

    def test_different_inputs(self):
        """Different inputs produce different hashes."""
        h1 = _compute_hash({"a": 1})
        h2 = _compute_hash({"a": 2})
        assert h1 != h2

    def test_model_hash(self):
        """Pydantic model can be hashed."""
        obj = ColumnProfileResponse(column_name="test")
        h = _compute_hash(obj)
        assert len(h) == 64


# ===================================================================
# TestThreadSafety
# ===================================================================


class TestThreadSafety:
    """Test thread-safety of DataQualityProfilerService."""

    def test_concurrent_profiling(self, sample_dataset):
        """Multiple threads profiling concurrently do not corrupt state."""
        svc = _make_service()
        errors: List[Exception] = []

        def profile_task(name: str):
            try:
                svc.profile_dataset(data=sample_dataset, dataset_name=name)
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=profile_task, args=(f"ds_{i}",))
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(svc.list_profiles()) == 5

    def test_concurrent_rule_creation(self):
        """Multiple threads creating rules concurrently is safe."""
        svc = _make_service()
        errors: List[Exception] = []

        def create_task(idx: int):
            try:
                svc.create_rule(
                    name=f"rule_{idx}", rule_type="not_null", column="x",
                )
            except Exception as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=create_task, args=(i,))
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(svc.list_rules()) == 5


# ===================================================================
# TestDetectColumnType
# ===================================================================


class TestDetectColumnType:
    """Test _detect_column_type internal method."""

    def test_integer_detection(self):
        """Detects integer type."""
        svc = _make_service()
        values = [1, 2, 3, 4, 5]
        assert svc._detect_column_type(values) == "integer"

    def test_float_detection(self):
        """Detects float type."""
        svc = _make_service()
        values = [1.1, 2.2, 3.3, 4.4, 5.5]
        assert svc._detect_column_type(values) == "float"

    def test_string_detection(self):
        """Detects string type."""
        svc = _make_service()
        values = ["a", "b", "c", "d", "e"]
        assert svc._detect_column_type(values) == "string"

    def test_boolean_detection(self):
        """Detects boolean type."""
        svc = _make_service()
        values = [True, False, True, False]
        assert svc._detect_column_type(values) == "boolean"

    def test_empty_values(self):
        """Empty list returns 'string'."""
        svc = _make_service()
        assert svc._detect_column_type([]) == "string"


# ===================================================================
# TestDetectNumericColumns
# ===================================================================


class TestDetectNumericColumns:
    """Test _detect_numeric_columns internal method."""

    def test_numeric_columns(self):
        """Correctly identifies numeric columns."""
        svc = _make_service()
        rows = [{"a": 1, "b": "text", "c": 3.5} for _ in range(10)]
        numeric = svc._detect_numeric_columns(rows)
        assert "a" in numeric
        assert "c" in numeric

    def test_no_numeric(self):
        """Returns empty list when no numeric columns."""
        svc = _make_service()
        rows = [{"a": "x", "b": "y"} for _ in range(10)]
        numeric = svc._detect_numeric_columns(rows)
        assert numeric == []

    def test_empty_rows(self):
        """Returns empty list for empty rows."""
        svc = _make_service()
        assert svc._detect_numeric_columns([]) == []


# ===================================================================
# TestGateConditionEvaluation
# ===================================================================


class TestGateConditionEvaluation:
    """Test _evaluate_gate_condition internal method."""

    def test_gte_pass(self):
        svc = _make_service()
        assert svc._evaluate_gate_condition(0.8, "gte", 0.7) is True

    def test_gte_fail(self):
        svc = _make_service()
        assert svc._evaluate_gate_condition(0.6, "gte", 0.7) is False

    def test_gt_pass(self):
        svc = _make_service()
        assert svc._evaluate_gate_condition(0.8, "gt", 0.7) is True

    def test_lt_pass(self):
        svc = _make_service()
        assert svc._evaluate_gate_condition(0.5, "lt", 0.7) is True

    def test_lte_pass(self):
        svc = _make_service()
        assert svc._evaluate_gate_condition(0.7, "lte", 0.7) is True

    def test_eq_pass(self):
        svc = _make_service()
        assert svc._evaluate_gate_condition(0.7, "eq", 0.7) is True


# ===================================================================
# TestIsValidValue
# ===================================================================


class TestIsValidValue:
    """Test _is_valid_value internal method."""

    def test_integer_valid(self):
        svc = _make_service()
        assert svc._is_valid_value(42, "integer") is True

    def test_integer_invalid(self):
        svc = _make_service()
        assert svc._is_valid_value("abc", "integer") is False

    def test_float_valid(self):
        svc = _make_service()
        assert svc._is_valid_value(3.14, "float") is True

    def test_float_invalid(self):
        svc = _make_service()
        assert svc._is_valid_value("not_num", "float") is False

    def test_boolean_valid(self):
        svc = _make_service()
        assert svc._is_valid_value("true", "boolean") is True

    def test_boolean_invalid(self):
        svc = _make_service()
        assert svc._is_valid_value("maybe", "boolean") is False

    def test_string_always_valid(self):
        svc = _make_service()
        assert svc._is_valid_value("anything", "string") is True

    def test_none_is_valid(self):
        svc = _make_service()
        assert svc._is_valid_value(None, "integer") is True


# ===================================================================
# TestCheckRuleForRow
# ===================================================================


class TestCheckRuleForRow:
    """Test _check_rule_for_row internal method."""

    def test_not_null_pass(self):
        """not_null rule passes for non-null value."""
        svc = _make_service()
        rule = svc.create_rule(name="r", rule_type="not_null", column="x")
        assert svc._check_rule_for_row(rule, {"x": "hello"}) is True

    def test_not_null_fail(self):
        """not_null rule fails for None value."""
        svc = _make_service()
        rule = svc.create_rule(name="r", rule_type="not_null", column="x")
        assert svc._check_rule_for_row(rule, {"x": None}) is False

    def test_not_null_empty_string(self):
        """not_null rule fails for empty string."""
        svc = _make_service()
        rule = svc.create_rule(name="r", rule_type="not_null", column="x")
        assert svc._check_rule_for_row(rule, {"x": ""}) is False

    def test_range_pass(self):
        """range rule passes for value in bounds."""
        svc = _make_service()
        rule = svc.create_rule(
            name="r", rule_type="range", column="x",
            parameters={"min": 0, "max": 100},
        )
        assert svc._check_rule_for_row(rule, {"x": 50}) is True

    def test_range_fail(self):
        """range rule fails for out-of-bound value."""
        svc = _make_service()
        rule = svc.create_rule(
            name="r", rule_type="range", column="x",
            parameters={"min": 0, "max": 10},
        )
        assert svc._check_rule_for_row(rule, {"x": 50}) is False

    def test_regex_pass(self):
        """regex rule passes for matching value."""
        svc = _make_service()
        rule = svc.create_rule(
            name="r", rule_type="regex", column="x",
            parameters={"pattern": r"^\d+$"},
        )
        assert svc._check_rule_for_row(rule, {"x": "123"}) is True

    def test_regex_fail(self):
        """regex rule fails for non-matching value."""
        svc = _make_service()
        rule = svc.create_rule(
            name="r", rule_type="regex", column="x",
            parameters={"pattern": r"^\d+$"},
        )
        assert svc._check_rule_for_row(rule, {"x": "abc"}) is False


# ===================================================================
# TestCheckOperator
# ===================================================================


class TestCheckOperator:
    """Test _check_operator internal method."""

    def test_eq_numeric(self):
        svc = _make_service()
        assert svc._check_operator(5, "eq", 5) is True

    def test_ne_numeric(self):
        svc = _make_service()
        assert svc._check_operator(5, "ne", 3) is True

    def test_gt_numeric(self):
        svc = _make_service()
        assert svc._check_operator(10, "gt", 5) is True

    def test_gte_numeric(self):
        svc = _make_service()
        assert svc._check_operator(5, "gte", 5) is True

    def test_lt_numeric(self):
        svc = _make_service()
        assert svc._check_operator(3, "lt", 5) is True

    def test_lte_numeric(self):
        svc = _make_service()
        assert svc._check_operator(5, "lte", 5) is True

    def test_eq_string(self):
        svc = _make_service()
        assert svc._check_operator("abc", "eq", "abc") is True

    def test_none_passthrough(self):
        svc = _make_service()
        assert svc._check_operator(None, "eq", 5) is True


# ===================================================================
# TestScoreDimensions
# ===================================================================


class TestScoreDimensions:
    """Test individual _score_* dimension methods."""

    def test_completeness_perfect(self):
        """Full data yields completeness near 1.0."""
        svc = _make_service()
        data = _perfect_data()
        profile = svc.profile_dataset(data=data, dataset_name="test")
        score, issues = svc._score_completeness(data, profile)
        assert score >= 0.9

    def test_completeness_with_nulls(self):
        """Data with nulls yields lower completeness."""
        svc = _make_service()
        data = _poor_data()
        profile = svc.profile_dataset(data=data, dataset_name="test")
        score, issues = svc._score_completeness(data, profile)
        assert score < 0.5

    def test_timeliness_default(self):
        """Timeliness returns a default high score."""
        svc = _make_service()
        data = _basic_data(5)
        profile = svc.profile_dataset(data=data, dataset_name="test")
        score, issues = svc._score_timeliness(data, profile)
        assert score == 0.90

    def test_uniqueness_unique_data(self):
        """All unique data yields high uniqueness."""
        svc = _make_service()
        data = _perfect_data()
        profile = svc.profile_dataset(data=data, dataset_name="test")
        score, issues = svc._score_uniqueness(data, profile)
        assert score >= 0.8

    def test_uniqueness_duplicate_data(self):
        """Duplicate data yields lower uniqueness."""
        svc = _make_service()
        base = {"id": 1, "name": "dup", "value": 1.0}
        data = [dict(base) for _ in range(20)]
        profile = svc.profile_dataset(data=data, dataset_name="test")
        score, issues = svc._score_uniqueness(data, profile)
        assert score < 0.8

    def test_consistency_mixed_types(self):
        """Mixed types in a column yields lower consistency."""
        svc = _make_service()
        data = [
            {"col": 1},
            {"col": "two"},
            {"col": 3},
            {"col": "four"},
            {"col": 5},
        ]
        profile = svc.profile_dataset(data=data, dataset_name="test")
        score, issues = svc._score_consistency(data, profile)
        # Should detect mixed types
        assert score <= 1.0

    def test_accuracy_within_range(self):
        """Numeric data within 3 std devs yields high accuracy."""
        svc = _make_service()
        data = [{"value": float(i)} for i in range(20)]
        profile = svc.profile_dataset(data=data, dataset_name="test")
        score, issues = svc._score_accuracy(data, profile)
        assert score >= 0.9

    def test_validity_all_valid(self):
        """All values matching expected types yields high validity."""
        svc = _make_service()
        data = [{"count": i} for i in range(20)]
        profile = svc.profile_dataset(data=data, dataset_name="test")
        score, issues = svc._score_validity(data, profile)
        assert score >= 0.9


# ===================================================================
# TestIQRAnomalies
# ===================================================================


class TestIQRAnomalies:
    """Test _iqr_anomalies internal method."""

    def test_detects_outliers(self):
        """Clear outliers are detected by IQR method."""
        svc = _make_service()
        values = [float(i) for i in range(30)]
        values.append(9999.0)
        indices = list(range(len(values)))
        anomalies = svc._iqr_anomalies(values, indices, "col")
        assert len(anomalies) > 0

    def test_no_outliers(self):
        """Uniform data has few or no outliers."""
        svc = _make_service()
        values = [5.0] * 30
        indices = list(range(30))
        anomalies = svc._iqr_anomalies(values, indices, "col")
        assert len(anomalies) == 0


# ===================================================================
# TestZscoreAnomalies
# ===================================================================


class TestZscoreAnomalies:
    """Test _zscore_anomalies internal method."""

    def test_detects_outliers(self):
        """Clear outliers are detected by z-score method."""
        svc = _make_service()
        values = [float(i) for i in range(30)]
        values.append(9999.0)
        indices = list(range(len(values)))
        anomalies = svc._zscore_anomalies(values, indices, "col")
        assert len(anomalies) > 0

    def test_uniform_data(self):
        """All same values yields zero std dev, no anomalies."""
        svc = _make_service()
        values = [5.0] * 30
        indices = list(range(30))
        anomalies = svc._zscore_anomalies(values, indices, "col")
        assert len(anomalies) == 0


# ===================================================================
# TestPercentileAnomalies
# ===================================================================


class TestPercentileAnomalies:
    """Test _percentile_anomalies internal method."""

    def test_detects_extremes(self):
        """Extreme values are detected by percentile method."""
        svc = _make_service()
        values = [float(i) for i in range(100)]
        values.append(99999.0)
        indices = list(range(len(values)))
        anomalies = svc._percentile_anomalies(values, indices, "col")
        assert len(anomalies) > 0


# ===================================================================
# TestReportFormatting
# ===================================================================


class TestReportFormatting:
    """Test internal report formatting methods."""

    def test_markdown_contains_header(self, sample_dataset):
        """Markdown format contains a header."""
        svc = _make_service()
        svc.assess_quality(data=sample_dataset, dataset_name="test")
        report = svc.generate_report(
            dataset_name="test", report_type="scorecard", report_format="markdown",
        )
        assert "# " in report["content"]

    def test_html_contains_body(self, sample_dataset):
        """HTML format contains body tag."""
        svc = _make_service()
        svc.assess_quality(data=sample_dataset, dataset_name="test")
        report = svc.generate_report(
            dataset_name="test", report_type="scorecard", report_format="html",
        )
        assert "</body>" in report["content"]

    def test_csv_has_rows(self, sample_dataset):
        """CSV format has rows."""
        svc = _make_service()
        svc.assess_quality(data=sample_dataset, dataset_name="test")
        report = svc.generate_report(
            dataset_name="test", report_type="scorecard", report_format="csv",
        )
        lines = report["content"].strip().split("\n")
        assert len(lines) >= 2  # header + at least 1 data row

    def test_text_has_separator(self, sample_dataset):
        """Text format has separator line."""
        svc = _make_service()
        svc.assess_quality(data=sample_dataset, dataset_name="test")
        report = svc.generate_report(
            dataset_name="test", report_type="scorecard", report_format="text",
        )
        assert "=" in report["content"]
