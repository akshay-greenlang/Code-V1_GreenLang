# -*- coding: utf-8 -*-
"""
Unit tests for OutlierPipelineEngine - AGENT-DATA-013

Tests run_pipeline, detect_stage, classify_stage, treat_stage,
validate_stage, document_stage, checkpointing, ensemble combination,
statistics, and error handling.
Target: 40+ tests.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-013 Outlier Detection (GL-DATA-X-016)
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from greenlang.outlier_detector.outlier_pipeline import (
    OutlierPipelineEngine,
    _safe_mean,
    _severity_from_score,
    _utcnow,
)
from greenlang.outlier_detector.models import (
    BatchDetectionResult,
    DetectionMethod,
    DetectionResult,
    EnsembleMethod,
    EnsembleResult,
    OutlierClassification,
    OutlierScore,
    OutlierStatus,
    PipelineConfig,
    PipelineResult,
    PipelineStage,
    SeverityLevel,
    TreatmentResult,
    TreatmentStrategy,
)


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def engine(config):
    return OutlierPipelineEngine(config)


@pytest.fixture
def sample_records() -> List[Dict[str, Any]]:
    """20 records with 2 clear outliers in 'val' column."""
    records = [{"val": 10.0 + i * 0.5, "cat": "A"} for i in range(18)]
    records.append({"val": 500.0, "cat": "A"})
    records.append({"val": -200.0, "cat": "A"})
    return records


@pytest.fixture
def minimal_records() -> List[Dict[str, Any]]:
    """5 records for quick pipeline tests."""
    return [
        {"val": 10.0}, {"val": 12.0}, {"val": 11.0},
        {"val": 500.0}, {"val": 9.0},
    ]


@pytest.fixture
def pipeline_config() -> PipelineConfig:
    """Default pipeline configuration."""
    return PipelineConfig()


def _make_outlier_score(
    value: Any = 500.0,
    score: float = 0.9,
    is_outlier: bool = True,
    record_index: int = 0,
    column_name: str = "val",
) -> OutlierScore:
    """Create a test OutlierScore."""
    return OutlierScore(
        record_index=record_index,
        column_name=column_name,
        value=value,
        method=DetectionMethod.IQR,
        score=score,
        is_outlier=is_outlier,
        threshold=1.5,
        severity=SeverityLevel.HIGH,
        details={},
        confidence=0.8,
        provenance_hash="a" * 64,
    )


# =========================================================================
# Helper function tests
# =========================================================================


class TestHelperFunctions:
    """Tests for module-level helpers."""

    def test_safe_mean_empty(self):
        assert _safe_mean([]) == 0.0

    def test_safe_mean_normal(self):
        assert _safe_mean([2.0, 4.0]) == pytest.approx(3.0)

    def test_severity_critical(self):
        assert _severity_from_score(0.96) == SeverityLevel.CRITICAL

    def test_severity_info(self):
        assert _severity_from_score(0.1) == SeverityLevel.INFO

    def test_utcnow_returns_datetime(self):
        dt = _utcnow()
        assert dt.microsecond == 0


# =========================================================================
# Full pipeline (run_pipeline)
# =========================================================================


class TestRunPipeline:
    """Tests for run_pipeline method."""

    def test_returns_pipeline_result(self, engine, sample_records):
        result = engine.run_pipeline(sample_records)
        assert isinstance(result, PipelineResult)

    def test_status_completed(self, engine, sample_records):
        result = engine.run_pipeline(sample_records)
        assert result.status == OutlierStatus.COMPLETED

    def test_pipeline_id_present(self, engine, sample_records):
        result = engine.run_pipeline(sample_records)
        assert result.pipeline_id is not None

    def test_detection_results_populated(self, engine, sample_records):
        result = engine.run_pipeline(sample_records)
        assert isinstance(result.detection_results, list)

    def test_ensemble_results_populated(self, engine, sample_records):
        result = engine.run_pipeline(sample_records)
        assert isinstance(result.ensemble_results, list)

    def test_processing_time_nonnegative(self, engine, sample_records):
        result = engine.run_pipeline(sample_records)
        assert result.total_processing_time_ms >= 0.0

    def test_provenance_hash_present(self, engine, sample_records):
        result = engine.run_pipeline(sample_records)
        assert len(result.provenance_hash) == 64

    def test_report_generated(self, engine, sample_records):
        result = engine.run_pipeline(sample_records)
        assert result.report is not None

    def test_validation_summary_present(self, engine, sample_records):
        result = engine.run_pipeline(sample_records)
        assert isinstance(result.validation_summary, dict)

    def test_with_custom_config(self, engine, sample_records):
        cfg = PipelineConfig(
            methods=[DetectionMethod.IQR, DetectionMethod.ZSCORE],
            ensemble_method=EnsembleMethod.MEAN_SCORE,
        )
        result = engine.run_pipeline(sample_records, config=cfg)
        assert result.status == OutlierStatus.COMPLETED

    def test_minimal_records(self, engine, minimal_records):
        result = engine.run_pipeline(minimal_records)
        assert result.status == OutlierStatus.COMPLETED


# =========================================================================
# Detect stage
# =========================================================================


class TestDetectStage:
    """Tests for detect_stage method."""

    def test_returns_batch_detection_result(self, engine, sample_records):
        result = engine.detect_stage(sample_records)
        assert isinstance(result, BatchDetectionResult)

    def test_results_populated(self, engine, sample_records):
        result = engine.detect_stage(sample_records)
        assert isinstance(result.results, list)
        assert len(result.results) > 0

    def test_auto_detect_columns(self, engine, sample_records):
        result = engine.detect_stage(sample_records)
        assert result.columns_analyzed >= 1

    def test_explicit_columns(self, engine, sample_records):
        result = engine.detect_stage(sample_records, columns=["val"])
        assert result.columns_analyzed >= 1

    def test_processing_time_nonnegative(self, engine, sample_records):
        result = engine.detect_stage(sample_records)
        assert result.processing_time_ms >= 0.0

    def test_provenance_hash_present(self, engine, sample_records):
        result = engine.detect_stage(sample_records)
        assert len(result.provenance_hash) == 64

    def test_total_outliers_nonnegative(self, engine, sample_records):
        result = engine.detect_stage(sample_records)
        assert result.total_outliers >= 0


# =========================================================================
# Classify stage
# =========================================================================


class TestClassifyStage:
    """Tests for classify_stage method."""

    def test_returns_list(self, engine, sample_records):
        detections = [
            _make_outlier_score(value=500.0, score=0.9, is_outlier=True, record_index=18),
        ]
        result = engine.classify_stage(detections, sample_records)
        assert isinstance(result, list)

    def test_only_outliers_classified(self, engine, sample_records):
        detections = [
            _make_outlier_score(value=10.0, score=0.1, is_outlier=False, record_index=0),
            _make_outlier_score(value=500.0, score=0.9, is_outlier=True, record_index=18),
        ]
        result = engine.classify_stage(detections, sample_records)
        assert len(result) == 1

    def test_empty_detections(self, engine, sample_records):
        result = engine.classify_stage([], sample_records)
        assert result == []


# =========================================================================
# Treat stage
# =========================================================================


class TestTreatStage:
    """Tests for treat_stage method."""

    def test_returns_list(self, engine, sample_records):
        detections = [
            _make_outlier_score(value=500.0, score=0.9, is_outlier=True, record_index=18),
        ]
        result = engine.treat_stage(
            sample_records, [], TreatmentStrategy.FLAG, detections,
        )
        assert isinstance(result, list)

    def test_no_detections_empty(self, engine, sample_records):
        result = engine.treat_stage(sample_records, [])
        assert result == []

    def test_flag_strategy(self, engine, sample_records):
        detections = [
            _make_outlier_score(value=500.0, score=0.9, is_outlier=True, record_index=18),
        ]
        result = engine.treat_stage(
            sample_records, [], TreatmentStrategy.FLAG, detections,
        )
        for r in result:
            assert r.strategy == TreatmentStrategy.FLAG


# =========================================================================
# Validate stage
# =========================================================================


class TestValidateStage:
    """Tests for validate_stage method."""

    def test_no_treatments_valid(self, engine, sample_records):
        result = engine.validate_stage(sample_records, [])
        assert result["valid"] is True
        assert result["status"] == "no_treatments"

    def test_with_treatments(self, engine, sample_records):
        from greenlang.outlier_detector.models import TreatmentResult, TreatmentStrategy
        treatments = [
            TreatmentResult(
                record_index=18,
                column_name="val",
                original_value=500.0,
                treated_value=10.0,
                strategy=TreatmentStrategy.CAP,
                reason="test",
                reversible=True,
                confidence=0.8,
                provenance_hash="a" * 64,
            ),
        ]
        result = engine.validate_stage(sample_records, treatments)
        assert "columns" in result
        assert "val" in result["columns"]


# =========================================================================
# Document stage
# =========================================================================


class TestDocumentStage:
    """Tests for document_stage method."""

    def test_returns_dict(self, engine, sample_records):
        doc = engine.document_stage(
            sample_records, [], [], [], [], {},
        )
        assert isinstance(doc, dict)

    def test_has_methodology(self, engine, sample_records):
        doc = engine.document_stage(
            sample_records, [], [], [], [], {},
        )
        assert "methodology" in doc

    def test_has_summary(self, engine, sample_records):
        doc = engine.document_stage(
            sample_records, [], [], [], [], {},
        )
        assert "summary" in doc
        assert doc["summary"]["total_records"] == len(sample_records)

    def test_has_provenance(self, engine, sample_records):
        doc = engine.document_stage(
            sample_records, [], [], [], [], {},
        )
        assert "provenance" in doc

    def test_has_generated_at(self, engine, sample_records):
        doc = engine.document_stage(
            sample_records, [], [], [], [], {},
        )
        assert "generated_at" in doc


# =========================================================================
# Checkpointing
# =========================================================================


class TestCheckpointing:
    """Tests for checkpoint and resume methods."""

    def test_create_checkpoint(self, engine):
        key = engine.create_checkpoint("pipe-1", "detect", {"outliers": 5})
        assert "pipe-1" in key
        assert "detect" in key

    def test_resume_checkpoint(self, engine):
        engine.create_checkpoint("pipe-2", "classify", {"n": 3})
        data = engine.resume_from_checkpoint("pipe-2", "classify")
        assert data is not None
        assert data["data"]["n"] == 3

    def test_resume_missing_checkpoint(self, engine):
        result = engine.resume_from_checkpoint("missing", "stage")
        assert result is None


# =========================================================================
# Statistics
# =========================================================================


class TestStatistics:
    """Tests for get_pipeline_stats method."""

    def test_returns_dict(self, engine):
        stats = engine.get_pipeline_stats()
        assert isinstance(stats, dict)

    def test_initial_stats(self, engine):
        stats = engine.get_pipeline_stats()
        assert stats["pipelines_run"] == 0
        assert stats["total_records"] == 0

    def test_stats_updated_after_pipeline(self, engine, minimal_records):
        engine.run_pipeline(minimal_records)
        stats = engine.get_pipeline_stats()
        assert stats["pipelines_run"] == 1
        assert stats["total_records"] >= len(minimal_records)


# =========================================================================
# Edge cases and error handling
# =========================================================================


class TestEdgeCases:
    """Tests for error recovery and edge cases."""

    def test_empty_records_pipeline(self, engine):
        result = engine.run_pipeline([])
        # Should handle gracefully (either COMPLETED or FAILED)
        assert isinstance(result, PipelineResult)

    def test_no_numeric_columns(self, engine):
        records = [{"text": "hello"}, {"text": "world"}]
        result = engine.run_pipeline(records)
        assert isinstance(result, PipelineResult)

    def test_deterministic_pipeline(self, engine, minimal_records):
        r1 = engine.run_pipeline(minimal_records)
        r2 = engine.run_pipeline(minimal_records)
        assert r1.status == r2.status

    def test_pipeline_with_all_methods(self, engine, sample_records):
        cfg = PipelineConfig(
            methods=[
                DetectionMethod.IQR,
                DetectionMethod.ZSCORE,
                DetectionMethod.MAD,
            ],
        )
        result = engine.run_pipeline(sample_records, config=cfg)
        assert isinstance(result, PipelineResult)
