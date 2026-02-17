# -*- coding: utf-8 -*-
"""
Unit tests for OutlierDetectorService (setup.py facade) - AGENT-DATA-013

Tests service initialization, job CRUD, detect_outliers, detect_batch,
classify_outliers, apply_treatment, thresholds, feedback, impact analysis,
run_pipeline, get_statistics, health_check, lifecycle, and helper functions.
Target: 85+ tests.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-013 Outlier Detection (GL-DATA-X-016)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from greenlang.outlier_detector.setup import (
    OutlierDetectorService,
    DetectionResponse,
    BatchDetectionResponse,
    ClassificationResponse,
    TreatmentResponse,
    ThresholdResponse,
    FeedbackResponse,
    PipelineResponse,
    StatsResponse,
    _ProvenanceTracker,
    _compute_hash,
    _is_numeric,
    _safe_float,
    _auto_detect_numeric_columns,
    _utcnow,
)


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def service(config):
    return OutlierDetectorService(config)


@pytest.fixture
def started_service(config):
    svc = OutlierDetectorService(config)
    svc.startup()
    return svc


@pytest.fixture
def sample_records() -> List[Dict[str, Any]]:
    """10 records with outlier at index 3."""
    return [
        {"val": 10.0, "cat": "A"},
        {"val": 12.0, "cat": "B"},
        {"val": 11.0, "cat": "A"},
        {"val": 500.0, "cat": "A"},
        {"val": 9.0, "cat": "B"},
        {"val": 13.0, "cat": "A"},
        {"val": 11.5, "cat": "B"},
        {"val": 10.0, "cat": "A"},
        {"val": 12.0, "cat": "B"},
        {"val": 11.0, "cat": "A"},
    ]


@pytest.fixture
def large_records() -> List[Dict[str, Any]]:
    """100 records with 5 outliers."""
    records = [{"val": float(i)} for i in range(95)]
    records.extend([
        {"val": 500.0}, {"val": 600.0}, {"val": 700.0},
        {"val": -200.0}, {"val": -300.0},
    ])
    return records


# =========================================================================
# Helper function tests
# =========================================================================


class TestHelperFunctions:
    """Tests for module-level helper functions."""

    def test_utcnow_no_microseconds(self):
        dt = _utcnow()
        assert dt.microsecond == 0

    def test_compute_hash_dict(self):
        h = _compute_hash({"a": 1, "b": 2})
        assert isinstance(h, str)
        assert len(h) == 64

    def test_compute_hash_deterministic(self):
        h1 = _compute_hash({"a": 1})
        h2 = _compute_hash({"a": 1})
        assert h1 == h2

    def test_compute_hash_different_data(self):
        h1 = _compute_hash({"a": 1})
        h2 = _compute_hash({"a": 2})
        assert h1 != h2

    def test_is_numeric_int(self):
        assert _is_numeric(42) is True

    def test_is_numeric_float(self):
        assert _is_numeric(3.14) is True

    def test_is_numeric_string_number(self):
        assert _is_numeric("123") is True

    def test_is_numeric_string_text(self):
        assert _is_numeric("abc") is False

    def test_is_numeric_none(self):
        assert _is_numeric(None) is False

    def test_is_numeric_empty(self):
        assert _is_numeric("") is False

    def test_safe_float_int(self):
        assert _safe_float(42) == 42.0

    def test_safe_float_string(self):
        assert _safe_float("3.14") == pytest.approx(3.14)

    def test_safe_float_none(self):
        assert _safe_float(None) is None

    def test_safe_float_bad_string(self):
        assert _safe_float("abc") is None

    def test_safe_float_empty_string(self):
        assert _safe_float("") is None

    def test_auto_detect_numeric_columns(self):
        records = [
            {"val": 10.0, "name": "Alice"},
            {"val": 20.0, "name": "Bob"},
            {"val": 30.0, "name": "Charlie"},
        ]
        cols = _auto_detect_numeric_columns(records)
        assert "val" in cols
        assert "name" not in cols

    def test_auto_detect_numeric_columns_empty(self):
        assert _auto_detect_numeric_columns([]) == []

    def test_auto_detect_numeric_columns_all_numeric(self):
        records = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
        cols = _auto_detect_numeric_columns(records)
        assert "a" in cols
        assert "b" in cols


# =========================================================================
# ProvenanceTracker
# =========================================================================


class TestProvenanceTracker:
    """Tests for _ProvenanceTracker helper class."""

    def test_initial_count(self):
        pt = _ProvenanceTracker()
        assert pt.entry_count == 0

    def test_record_increments_count(self):
        pt = _ProvenanceTracker()
        pt.record("test", "id-1", "create", "abc123")
        assert pt.entry_count == 1

    def test_record_returns_hash(self):
        pt = _ProvenanceTracker()
        h = pt.record("test", "id-1", "create", "abc123")
        assert isinstance(h, str)
        assert len(h) == 64

    def test_record_hashes_different(self):
        pt = _ProvenanceTracker()
        h1 = pt.record("test", "id-1", "create", "abc123")
        h2 = pt.record("test", "id-2", "create", "def456")
        assert h1 != h2


# =========================================================================
# Service initialization
# =========================================================================


class TestServiceInit:
    """Tests for OutlierDetectorService initialization."""

    def test_creates_instance(self, config):
        svc = OutlierDetectorService(config)
        assert svc is not None

    def test_config_stored(self, config):
        svc = OutlierDetectorService(config)
        assert svc.config is config

    def test_provenance_tracker_created(self, config):
        svc = OutlierDetectorService(config)
        assert svc.provenance is not None

    def test_default_config(self):
        svc = OutlierDetectorService()
        assert svc.config is not None

    def test_engines_initialized(self, service):
        # At minimum statistical engine should be available
        assert service.statistical_engine is not None

    def test_classifier_engine_available(self, service):
        assert service.classifier_engine is not None

    def test_treatment_engine_available(self, service):
        assert service.treatment_engine is not None


# =========================================================================
# Lifecycle
# =========================================================================


class TestLifecycle:
    """Tests for startup/shutdown lifecycle."""

    def test_startup(self, service):
        service.startup()
        health = service.health_check()
        assert health["started"] is True

    def test_shutdown(self, started_service):
        started_service.shutdown()
        health = started_service.health_check()
        assert health["started"] is False

    def test_startup_then_shutdown(self, service):
        service.startup()
        assert service.health_check()["status"] == "healthy"
        service.shutdown()
        assert service.health_check()["status"] == "not_started"


# =========================================================================
# Health check
# =========================================================================


class TestHealthCheck:
    """Tests for health_check method."""

    def test_returns_dict(self, service):
        result = service.health_check()
        assert isinstance(result, dict)

    def test_has_status(self, service):
        result = service.health_check()
        assert "status" in result

    def test_not_started_status(self, service):
        result = service.health_check()
        assert result["status"] == "not_started"

    def test_healthy_status(self, started_service):
        result = started_service.health_check()
        assert result["status"] == "healthy"

    def test_has_engine_flags(self, service):
        result = service.health_check()
        assert "engines" in result
        assert isinstance(result["engines"], dict)

    def test_has_counters(self, service):
        result = service.health_check()
        assert "jobs" in result
        assert "detections" in result


# =========================================================================
# Job CRUD
# =========================================================================


class TestJobCRUD:
    """Tests for job creation, listing, retrieval, and deletion."""

    def test_create_job(self, service, sample_records):
        job = service.create_job({"records": sample_records})
        assert "job_id" in job
        assert job["status"] == "pending"

    def test_create_job_has_provenance(self, service, sample_records):
        job = service.create_job({"records": sample_records})
        assert len(job["provenance_hash"]) == 64

    def test_create_job_record_count(self, service, sample_records):
        job = service.create_job({"records": sample_records})
        assert job["total_records"] == len(sample_records)

    def test_list_jobs_empty(self, service):
        jobs = service.list_jobs()
        assert jobs == []

    def test_list_jobs_after_create(self, service, sample_records):
        service.create_job({"records": sample_records})
        jobs = service.list_jobs()
        assert len(jobs) == 1

    def test_list_jobs_with_status_filter(self, service, sample_records):
        service.create_job({"records": sample_records})
        pending = service.list_jobs(status="pending")
        assert len(pending) == 1
        completed = service.list_jobs(status="completed")
        assert len(completed) == 0

    def test_list_jobs_with_limit(self, service, sample_records):
        for _ in range(5):
            service.create_job({"records": sample_records})
        jobs = service.list_jobs(limit=3)
        assert len(jobs) == 3

    def test_get_job_exists(self, service, sample_records):
        job = service.create_job({"records": sample_records})
        retrieved = service.get_job(job["job_id"])
        assert retrieved is not None
        assert retrieved["job_id"] == job["job_id"]

    def test_get_job_not_found(self, service):
        result = service.get_job("nonexistent")
        assert result is None

    def test_delete_job_exists(self, service, sample_records):
        job = service.create_job({"records": sample_records})
        deleted = service.delete_job(job["job_id"])
        assert deleted is True

    def test_delete_job_sets_cancelled(self, service, sample_records):
        job = service.create_job({"records": sample_records})
        service.delete_job(job["job_id"])
        retrieved = service.get_job(job["job_id"])
        assert retrieved["status"] == "cancelled"

    def test_delete_job_not_found(self, service):
        deleted = service.delete_job("nonexistent")
        assert deleted is False


# =========================================================================
# Detect outliers
# =========================================================================


class TestDetectOutliers:
    """Tests for detect_outliers method."""

    def test_returns_detection_response(self, service, sample_records):
        result = service.detect_outliers(sample_records, "val")
        assert isinstance(result, DetectionResponse)

    def test_column_name_stored(self, service, sample_records):
        result = service.detect_outliers(sample_records, "val")
        assert result.column_name == "val"

    def test_outliers_found(self, service, sample_records):
        result = service.detect_outliers(sample_records, "val")
        assert result.outliers_found > 0

    def test_total_points_correct(self, service, sample_records):
        result = service.detect_outliers(sample_records, "val")
        assert result.total_points > 0

    def test_scores_present(self, service, sample_records):
        result = service.detect_outliers(sample_records, "val")
        assert isinstance(result.scores, list)

    def test_provenance_hash_present(self, service, sample_records):
        result = service.detect_outliers(sample_records, "val")
        assert len(result.provenance_hash) == 64

    def test_processing_time_nonnegative(self, service, sample_records):
        result = service.detect_outliers(sample_records, "val")
        assert result.processing_time_ms >= 0.0

    def test_empty_records_raises(self, service):
        with pytest.raises(ValueError, match="empty"):
            service.detect_outliers([], "val")

    def test_detection_stored(self, service, sample_records):
        result = service.detect_outliers(sample_records, "val")
        stored = service.get_detection(result.detection_id)
        assert stored is not None

    def test_detection_id_unique(self, service, sample_records):
        r1 = service.detect_outliers(sample_records, "val")
        r2 = service.detect_outliers(sample_records, "val")
        assert r1.detection_id != r2.detection_id


# =========================================================================
# Batch detect
# =========================================================================


class TestDetectBatch:
    """Tests for detect_batch method."""

    def test_returns_batch_response(self, service, sample_records):
        result = service.detect_batch(sample_records)
        assert isinstance(result, BatchDetectionResponse)

    def test_auto_detect_columns(self, service, sample_records):
        result = service.detect_batch(sample_records)
        assert result.total_columns >= 1

    def test_explicit_columns(self, service, sample_records):
        result = service.detect_batch(sample_records, columns=["val"])
        assert result.total_columns >= 1

    def test_total_outliers_nonnegative(self, service, sample_records):
        result = service.detect_batch(sample_records)
        assert result.total_outliers >= 0

    def test_provenance_hash_present(self, service, sample_records):
        result = service.detect_batch(sample_records)
        assert len(result.provenance_hash) == 64

    def test_empty_records_raises(self, service):
        with pytest.raises(ValueError, match="empty"):
            service.detect_batch([])


# =========================================================================
# Classify outliers
# =========================================================================


class TestClassifyOutliers:
    """Tests for classify_outliers method."""

    def test_returns_classification_response(self, service, sample_records):
        det_result = service.detect_outliers(sample_records, "val")
        detections = det_result.scores
        result = service.classify_outliers(detections, sample_records)
        assert isinstance(result, ClassificationResponse)

    def test_classifications_present(self, service, sample_records):
        det_result = service.detect_outliers(sample_records, "val")
        detections = det_result.scores
        result = service.classify_outliers(detections, sample_records)
        assert isinstance(result.classifications, list)

    def test_provenance_hash_present(self, service, sample_records):
        det_result = service.detect_outliers(sample_records, "val")
        detections = det_result.scores
        result = service.classify_outliers(detections, sample_records)
        assert len(result.provenance_hash) == 64

    def test_empty_detections_raises(self, service, sample_records):
        with pytest.raises(ValueError, match="empty"):
            service.classify_outliers([], sample_records)

    def test_classification_stored(self, service, sample_records):
        det_result = service.detect_outliers(sample_records, "val")
        detections = det_result.scores
        result = service.classify_outliers(detections, sample_records)
        stored = service.get_classification(result.classification_id)
        assert stored is not None


# =========================================================================
# Apply treatment
# =========================================================================


class TestApplyTreatment:
    """Tests for apply_treatment method."""

    def test_returns_treatment_response(self, service, sample_records):
        det_result = service.detect_outliers(sample_records, "val")
        detections = det_result.scores
        result = service.apply_treatment(
            sample_records, detections, "flag",
        )
        assert isinstance(result, TreatmentResponse)

    def test_provenance_hash_present(self, service, sample_records):
        det_result = service.detect_outliers(sample_records, "val")
        detections = det_result.scores
        result = service.apply_treatment(
            sample_records, detections, "flag",
        )
        assert len(result.provenance_hash) == 64


# =========================================================================
# Thresholds
# =========================================================================


class TestThresholds:
    """Tests for threshold management."""

    def test_create_threshold(self, service):
        result = service.create_threshold(
            column="emissions", min_val=0.0, max_val=1000.0,
        )
        assert isinstance(result, ThresholdResponse)

    def test_threshold_column_stored(self, service):
        result = service.create_threshold(
            column="temp", min_val=-40.0, max_val=60.0,
        )
        assert result.column_name == "temp"

    def test_list_thresholds_empty(self, service):
        result = service.list_thresholds()
        assert result == []

    def test_list_thresholds_after_create(self, service):
        service.create_threshold(column="val", min_val=0.0, max_val=100.0)
        result = service.list_thresholds()
        assert len(result) == 1


# =========================================================================
# Feedback
# =========================================================================


class TestFeedback:
    """Tests for feedback submission."""

    def test_submit_feedback(self, service):
        result = service.submit_feedback(
            detection_id="det-001",
            feedback_type="confirmed_outlier",
            notes="Valid outlier",
        )
        assert isinstance(result, FeedbackResponse)

    def test_feedback_detection_id(self, service):
        result = service.submit_feedback(
            detection_id="det-002",
            feedback_type="false_positive",
        )
        assert result.detection_id == "det-002"

    def test_feedback_accepted(self, service):
        result = service.submit_feedback(
            detection_id="det-003",
            feedback_type="confirmed_outlier",
        )
        assert result.accepted is True


# =========================================================================
# Impact analysis
# =========================================================================


class TestImpactAnalysis:
    """Tests for analyze_impact method."""

    def test_returns_dict(self, service):
        original = [{"val": 10.0}, {"val": 500.0}]
        treated = [{"val": 10.0}, {"val": 50.0}]
        result = service.analyze_impact(
            original=original, treated=treated,
        )
        assert isinstance(result, dict)


# =========================================================================
# Pipeline
# =========================================================================


class TestRunPipeline:
    """Tests for run_pipeline method."""

    def test_returns_pipeline_response(self, service, sample_records):
        result = service.run_pipeline(sample_records)
        assert isinstance(result, PipelineResponse)

    def test_pipeline_status(self, service, sample_records):
        result = service.run_pipeline(sample_records)
        assert result.status in ("completed", "failed")

    def test_provenance_hash_present(self, service, sample_records):
        result = service.run_pipeline(sample_records)
        assert len(result.provenance_hash) == 64


# =========================================================================
# Statistics
# =========================================================================


class TestStatistics:
    """Tests for get_statistics method."""

    def test_returns_stats_response(self, service):
        result = service.get_statistics()
        assert isinstance(result, StatsResponse)

    def test_initial_stats(self, service):
        result = service.get_statistics()
        assert result.total_jobs == 0

    def test_stats_after_job(self, service, sample_records):
        service.create_job({"records": sample_records})
        result = service.get_statistics()
        assert result.total_jobs == 1

    def test_stats_after_detection(self, service, sample_records):
        service.detect_outliers(sample_records, "val")
        result = service.get_statistics()
        assert result.total_outliers_detected >= 0
        assert result.total_records_processed > 0


# =========================================================================
# Response model tests
# =========================================================================


class TestResponseModels:
    """Tests for Pydantic response model defaults."""

    def test_detection_response_defaults(self):
        r = DetectionResponse()
        assert r.column_name == ""
        assert r.method == "iqr"
        assert r.total_points == 0
        assert r.outliers_found == 0

    def test_batch_detection_response_defaults(self):
        r = BatchDetectionResponse()
        assert r.total_columns == 0
        assert r.total_outliers == 0

    def test_classification_response_defaults(self):
        r = ClassificationResponse()
        assert r.total_classified == 0

    def test_treatment_response_defaults(self):
        r = TreatmentResponse()
        assert r.strategy == "flag"
        assert r.total_treated == 0

    def test_threshold_response_defaults(self):
        r = ThresholdResponse()
        assert r.column_name == ""
        assert r.active is True

    def test_feedback_response_defaults(self):
        r = FeedbackResponse()
        assert r.accepted is True

    def test_pipeline_response_defaults(self):
        r = PipelineResponse()
        assert r.status == "completed"
        assert r.total_records == 0

    def test_stats_response_defaults(self):
        r = StatsResponse()
        assert r.total_jobs == 0
        assert r.total_outliers_detected == 0


# =========================================================================
# Edge cases
# =========================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_get_detections_initially_empty(self, service):
        result = service.get_detections()
        assert result == []

    def test_get_detection_not_found(self, service):
        result = service.get_detection("nonexistent")
        assert result is None

    def test_get_classification_not_found(self, service):
        result = service.get_classification("nonexistent")
        assert result is None

    def test_get_treatment_not_found(self, service):
        result = service.get_treatment("nonexistent")
        assert result is None

    def test_deterministic_detection(self, service, sample_records):
        r1 = service.detect_outliers(sample_records, "val")
        r2 = service.detect_outliers(sample_records, "val")
        assert r1.outliers_found == r2.outliers_found
        assert r1.outlier_pct == r2.outlier_pct

    def test_large_dataset(self, service, large_records):
        result = service.detect_outliers(large_records, "val")
        assert isinstance(result, DetectionResponse)
        assert result.total_points > 0
