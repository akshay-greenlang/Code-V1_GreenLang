# -*- coding: utf-8 -*-
"""
Unit tests for Outlier Detection data models - AGENT-DATA-013

Tests all 13 enums have correct members, all 20 models create with defaults,
8 request models validate. Target: 50+ tests.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-013 Outlier Detection (GL-DATA-X-016)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from greenlang.outlier_detector.models import (
    # Enums
    DetectionMethod, OutlierClass, TreatmentStrategy, OutlierStatus,
    EnsembleMethod, ContextType, TemporalMethod, SeverityLevel,
    ReportFormat, PipelineStage, ThresholdSource, FeedbackType,
    DataColumnType,
    # SDK models
    OutlierScore, DetectionResult, ContextualResult, TemporalResult,
    MultivariateResult, EnsembleResult, OutlierClassification,
    TreatmentResult, TreatmentRecord, DomainThreshold, FeedbackEntry,
    ImpactAnalysis, OutlierReport, PipelineConfig, PipelineResult,
    DetectionJobConfig, OutlierStatistics, ThresholdConfig,
    ColumnOutlierSummary, BatchDetectionResult,
    # Request models
    CreateDetectionJobRequest, DetectOutliersRequest,
    ClassifyOutliersRequest, TreatOutliersRequest,
    SubmitFeedbackRequest, RunPipelineRequest,
    BatchDetectRequest, ConfigureThresholdsRequest,
    # Constants
    DEFAULT_METHOD_WEIGHTS, SEVERITY_THRESHOLDS,
    CLASSIFICATION_CONFIDENCE, PIPELINE_STAGE_ORDER,
    DETECTION_METHODS, TREATMENT_STRATEGIES,
    REPORT_FORMAT_OPTIONS, TEMPORAL_METHODS,
)


# =============================================================================
# Enum tests
# =============================================================================


class TestEnums:
    """Test all 13 enums have correct members."""

    def test_detection_method_count(self):
        assert len(DetectionMethod) == 13

    def test_detection_method_iqr(self):
        assert DetectionMethod.IQR.value == "iqr"

    def test_detection_method_zscore(self):
        assert DetectionMethod.ZSCORE.value == "zscore"

    def test_detection_method_temporal(self):
        assert DetectionMethod.TEMPORAL.value == "temporal"

    def test_outlier_class_members(self):
        assert set(OutlierClass) == {
            OutlierClass.ERROR, OutlierClass.GENUINE_EXTREME,
            OutlierClass.DATA_ENTRY, OutlierClass.REGIME_CHANGE,
            OutlierClass.SENSOR_FAULT,
        }

    def test_treatment_strategy_count(self):
        assert len(TreatmentStrategy) == 6

    def test_treatment_strategy_cap(self):
        assert TreatmentStrategy.CAP.value == "cap"

    def test_outlier_status_members(self):
        assert set(OutlierStatus) == {
            OutlierStatus.PENDING, OutlierStatus.DETECTING,
            OutlierStatus.CLASSIFYING, OutlierStatus.TREATING,
            OutlierStatus.COMPLETED, OutlierStatus.FAILED,
        }

    def test_ensemble_method_members(self):
        assert set(EnsembleMethod) == {
            EnsembleMethod.WEIGHTED_AVERAGE, EnsembleMethod.MAJORITY_VOTE,
            EnsembleMethod.MAX_SCORE, EnsembleMethod.MEAN_SCORE,
        }

    def test_context_type_count(self):
        assert len(ContextType) == 6

    def test_context_type_facility(self):
        assert ContextType.FACILITY.value == "facility"

    def test_temporal_method_count(self):
        assert len(TemporalMethod) == 5
        assert TemporalMethod.CUSUM.value == "cusum"

    def test_severity_level_members(self):
        assert set(SeverityLevel) == {
            SeverityLevel.CRITICAL, SeverityLevel.HIGH,
            SeverityLevel.MEDIUM, SeverityLevel.LOW, SeverityLevel.INFO,
        }

    def test_report_format_count(self):
        assert len(ReportFormat) == 4
        assert ReportFormat.JSON.value == "json"

    def test_pipeline_stage_count(self):
        assert len(PipelineStage) == 5
        assert PipelineStage.DETECT.value == "detect"

    def test_threshold_source_count(self):
        assert len(ThresholdSource) == 5
        assert ThresholdSource.DOMAIN.value == "domain"

    def test_feedback_type_members(self):
        assert set(FeedbackType) == {
            FeedbackType.CONFIRMED_OUTLIER, FeedbackType.FALSE_POSITIVE,
            FeedbackType.RECLASSIFIED, FeedbackType.UNKNOWN,
        }

    def test_data_column_type_members(self):
        assert len(DataColumnType) == 5
        assert DataColumnType.NUMERIC.value == "numeric"


# =============================================================================
# SDK model creation tests
# =============================================================================


class TestSDKModels:
    """Test SDK models create with defaults and validate constraints."""

    def test_outlier_score_requires_record_index(self):
        s = OutlierScore(record_index=0, method=DetectionMethod.IQR)
        assert s.record_index == 0
        assert s.score == 0.0
        assert s.is_outlier is False

    def test_outlier_score_score_bounds(self):
        with pytest.raises(ValidationError):
            OutlierScore(record_index=0, method=DetectionMethod.IQR, score=1.5)

    def test_detection_result_requires_column_name(self):
        with pytest.raises(ValidationError):
            DetectionResult(column_name="", method=DetectionMethod.IQR)

    def test_detection_result_valid(self):
        r = DetectionResult(column_name="val", method=DetectionMethod.IQR)
        assert len(r.result_id) == 36

    def test_contextual_result_defaults(self):
        cr = ContextualResult()
        assert cr.context_type == ContextType.CUSTOM
        assert cr.group_size == 0

    def test_temporal_result_requires_method(self):
        tr = TemporalResult(method=TemporalMethod.CUSUM)
        assert tr.series_length == 0
        assert tr.anomalies_found == 0

    def test_multivariate_result_defaults(self):
        mr = MultivariateResult(method=DetectionMethod.MAHALANOBIS)
        assert mr.total_points == 0
        assert mr.columns == []

    def test_ensemble_result_requires_record_index(self):
        er = EnsembleResult(record_index=5)
        assert er.ensemble_score == 0.0
        assert er.ensemble_method == EnsembleMethod.WEIGHTED_AVERAGE

    def test_outlier_classification_requires_fields(self):
        oc = OutlierClassification(
            record_index=0, outlier_class=OutlierClass.ERROR,
        )
        assert oc.confidence == 0.0
        assert oc.recommended_treatment == TreatmentStrategy.FLAG

    def test_treatment_result_requires_fields(self):
        tr = TreatmentResult(
            record_index=0, strategy=TreatmentStrategy.CAP,
        )
        assert tr.reversible is True

    def test_treatment_record_defaults(self):
        tr = TreatmentRecord()
        assert tr.undone is False
        assert tr.strategy == TreatmentStrategy.FLAG

    def test_domain_threshold_requires_column(self):
        with pytest.raises(ValidationError):
            DomainThreshold(column_name="")

    def test_domain_threshold_valid(self):
        dt = DomainThreshold(column_name="temp", lower_bound=0.0, upper_bound=100.0)
        assert dt.active is True
        assert dt.source == ThresholdSource.DOMAIN

    def test_feedback_entry_requires_fields(self):
        fe = FeedbackEntry(
            record_index=0, feedback_type=FeedbackType.CONFIRMED_OUTLIER,
        )
        assert fe.user_id == "system"

    def test_impact_analysis_defaults(self):
        ia = ImpactAnalysis()
        assert ia.records_affected == 0
        assert ia.mean_change_pct == 0.0

    def test_outlier_report_defaults(self):
        r = OutlierReport()
        assert r.total_records == 0
        assert r.by_method == {}

    def test_pipeline_config_defaults(self):
        pc = PipelineConfig()
        assert pc.confidence_threshold == 0.5
        assert pc.enable_classification is True
        assert len(pc.methods) == 3

    def test_pipeline_result_defaults(self):
        pr = PipelineResult()
        assert pr.status == OutlierStatus.COMPLETED
        assert pr.stage == PipelineStage.DOCUMENT

    def test_detection_job_is_active_pending(self):
        jc = DetectionJobConfig(status=OutlierStatus.PENDING)
        assert jc.is_active is True

    def test_detection_job_not_active_completed(self):
        jc = DetectionJobConfig(status=OutlierStatus.COMPLETED)
        assert jc.is_active is False

    def test_detection_job_progress_pct(self):
        jc = DetectionJobConfig(status=OutlierStatus.PENDING, stage=PipelineStage.DETECT)
        assert jc.progress_pct == 20.0

    def test_detection_job_progress_completed(self):
        jc = DetectionJobConfig(status=OutlierStatus.COMPLETED)
        assert jc.progress_pct == 100.0

    def test_detection_job_progress_failed(self):
        jc = DetectionJobConfig(status=OutlierStatus.FAILED)
        assert jc.progress_pct == 0.0

    def test_outlier_statistics_defaults(self):
        stats = OutlierStatistics()
        assert stats.total_jobs == 0
        assert stats.avg_outlier_pct == 0.0

    def test_threshold_config_requires_column(self):
        with pytest.raises(ValidationError):
            ThresholdConfig(column_name="", method=DetectionMethod.IQR, value=1.5)

    def test_threshold_config_valid(self):
        tc = ThresholdConfig(column_name="val", method=DetectionMethod.IQR, value=1.5)
        assert tc.source == ThresholdSource.STATISTICAL

    def test_column_outlier_summary_requires_column(self):
        with pytest.raises(ValidationError):
            ColumnOutlierSummary(column_name="")

    def test_column_outlier_summary_valid(self):
        cos = ColumnOutlierSummary(column_name="temp")
        assert cos.total_points == 0
        assert cos.methods_used == []

    def test_batch_detection_result_defaults(self):
        bdr = BatchDetectionResult()
        assert bdr.total_outliers == 0
        assert bdr.results == []


# =============================================================================
# Request model tests
# =============================================================================


class TestRequestModels:
    """Test all 8 request models."""

    def test_create_job_requires_records(self):
        with pytest.raises(ValidationError):
            CreateDetectionJobRequest(records=[])

    def test_create_job_valid(self):
        req = CreateDetectionJobRequest(records=[{"a": 1}])
        assert len(req.records) == 1

    def test_detect_outliers_request_valid(self):
        req = DetectOutliersRequest(records=[{"a": 1}])
        assert len(req.methods) == 2

    def test_classify_outliers_request_valid(self):
        req = ClassifyOutliersRequest(
            records=[{"a": 1}], detections=[{"score": 0.9}],
        )
        assert len(req.records) == 1

    def test_treat_outliers_request_valid(self):
        req = TreatOutliersRequest(
            records=[{"a": 1}], detections=[{"score": 0.9}],
        )
        assert req.strategy == TreatmentStrategy.FLAG

    def test_submit_feedback_requires_fields(self):
        req = SubmitFeedbackRequest(
            record_index=0, feedback_type=FeedbackType.CONFIRMED_OUTLIER,
        )
        assert req.comment == ""

    def test_run_pipeline_request_valid(self):
        req = RunPipelineRequest(records=[{"a": 1}])
        assert req.dataset_id == ""

    def test_batch_detect_requires_datasets(self):
        with pytest.raises(ValidationError):
            BatchDetectRequest(datasets=[])

    def test_batch_detect_valid(self):
        req = BatchDetectRequest(datasets=[[{"a": 1}]])
        assert len(req.datasets) == 1

    def test_configure_thresholds_valid(self):
        req = ConfigureThresholdsRequest()
        assert req.thresholds == []


# =============================================================================
# Constants tests
# =============================================================================


class TestConstants:
    """Test module-level constants."""

    def test_default_method_weights(self):
        assert DEFAULT_METHOD_WEIGHTS["iqr"] == 1.0
        assert DEFAULT_METHOD_WEIGHTS["lof"] == 1.3

    def test_severity_thresholds(self):
        assert SEVERITY_THRESHOLDS["critical"] == 0.95
        assert SEVERITY_THRESHOLDS["info"] == 0.0

    def test_classification_confidence(self):
        assert CLASSIFICATION_CONFIDENCE["high"] == 0.85
        assert CLASSIFICATION_CONFIDENCE["low"] == 0.40

    def test_pipeline_stage_order(self):
        assert PIPELINE_STAGE_ORDER == ("detect", "classify", "treat", "validate", "document")

    def test_detection_methods_tuple(self):
        assert "iqr" in DETECTION_METHODS
        assert "temporal" in DETECTION_METHODS
        assert len(DETECTION_METHODS) == 13

    def test_treatment_strategies_tuple(self):
        assert "cap" in TREATMENT_STRATEGIES
        assert "flag" in TREATMENT_STRATEGIES
        assert len(TREATMENT_STRATEGIES) == 6

    def test_report_format_options(self):
        assert "json" in REPORT_FORMAT_OPTIONS
        assert len(REPORT_FORMAT_OPTIONS) == 4

    def test_temporal_methods(self):
        assert "cusum" in TEMPORAL_METHODS
        assert len(TEMPORAL_METHODS) == 5
