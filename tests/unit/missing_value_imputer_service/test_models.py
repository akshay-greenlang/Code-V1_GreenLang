# -*- coding: utf-8 -*-
"""
Unit tests for Missing Value Imputer data models - AGENT-DATA-012

Tests all 12 enums, 20 SDK models, 8 request models, field validators,
UUID generation, and edge cases. Target: 50+ tests.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-012 Missing Value Imputer (GL-DATA-X-015)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from greenlang.missing_value_imputer.models import (
    # Enums
    MissingnessType,
    ImputationStrategy,
    ImputationStatus,
    ConfidenceLevel,
    DataColumnType,
    ValidationMethod,
    RuleConditionType,
    RulePriority,
    ReportFormat,
    PipelineStage,
    PatternType,
    TimeSeriesFrequency,
    # SDK models
    MissingnessPattern,
    ColumnAnalysis,
    MissingnessReport,
    ImputationRule,
    RuleCondition,
    LookupTable,
    LookupEntry,
    ImputedValue,
    ImputationResult,
    ImputationBatch,
    ValidationResult,
    ValidationReport,
    ImputationTemplate,
    PipelineConfig,
    PipelineResult,
    ImputationJobConfig,
    ImputationStatistics,
    TimeSeriesConfig,
    MLModelConfig,
    StrategySelection,
    # Request models
    CreateJobRequest,
    AnalyzeMissingnessRequest,
    ImputeValuesRequest,
    BatchImputeRequest,
    ValidateRequest,
    CreateRuleRequest,
    CreateTemplateRequest,
    RunPipelineRequest,
    # Constants
    DEFAULT_CONFIDENCE_THRESHOLDS,
    STRATEGY_BY_COLUMN_TYPE,
    KNN_NEIGHBORS_BY_SIZE,
    IMPUTATION_STRATEGIES,
    VALIDATION_METHODS,
    PIPELINE_STAGE_ORDER,
)


# =============================================================================
# Enum tests
# =============================================================================


class TestEnums:
    """Test all 12 enums have correct members."""

    def test_missingness_type_members(self):
        assert set(MissingnessType) == {
            MissingnessType.MCAR, MissingnessType.MAR,
            MissingnessType.MNAR, MissingnessType.UNKNOWN,
        }

    def test_imputation_strategy_count(self):
        assert len(ImputationStrategy) == 19

    def test_imputation_strategy_mean(self):
        assert ImputationStrategy.MEAN.value == "mean"

    def test_imputation_strategy_mice(self):
        assert ImputationStrategy.MICE.value == "mice"

    def test_imputation_status_members(self):
        assert set(ImputationStatus) == {
            ImputationStatus.PENDING, ImputationStatus.ANALYZING,
            ImputationStatus.IMPUTING, ImputationStatus.VALIDATING,
            ImputationStatus.COMPLETED, ImputationStatus.FAILED,
        }

    def test_confidence_level_members(self):
        assert set(ConfidenceLevel) == {
            ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM,
            ConfidenceLevel.LOW, ConfidenceLevel.VERY_LOW,
        }

    def test_data_column_type_members(self):
        assert len(DataColumnType) == 5
        assert DataColumnType.NUMERIC.value == "numeric"

    def test_validation_method_members(self):
        assert len(ValidationMethod) == 5
        assert ValidationMethod.KS_TEST.value == "ks_test"

    def test_rule_condition_type_members(self):
        assert len(RuleConditionType) == 8
        assert RuleConditionType.EQUALS.value == "equals"
        assert RuleConditionType.IS_NULL.value == "is_null"

    def test_rule_priority_members(self):
        assert set(RulePriority) == {
            RulePriority.CRITICAL, RulePriority.HIGH,
            RulePriority.MEDIUM, RulePriority.LOW, RulePriority.DEFAULT,
        }

    def test_report_format_members(self):
        assert len(ReportFormat) == 5
        assert ReportFormat.JSON.value == "json"

    def test_pipeline_stage_members(self):
        assert len(PipelineStage) == 5
        assert PipelineStage.ANALYZE.value == "analyze"

    def test_pattern_type_members(self):
        assert set(PatternType) == {
            PatternType.UNIVARIATE, PatternType.MONOTONE,
            PatternType.ARBITRARY, PatternType.PLANNED,
        }

    def test_time_series_frequency_members(self):
        assert len(TimeSeriesFrequency) == 6
        assert TimeSeriesFrequency.MONTHLY.value == "monthly"


# =============================================================================
# SDK model creation tests
# =============================================================================


class TestSDKModels:
    """Test SDK models create with defaults and validate constraints."""

    def test_missingness_pattern_defaults(self):
        p = MissingnessPattern()
        assert p.pattern_type == PatternType.ARBITRARY
        assert p.missingness_type == MissingnessType.UNKNOWN
        assert p.total_missing == 0
        assert len(p.pattern_id) == 36  # UUID

    def test_column_analysis_requires_column_name(self):
        with pytest.raises(ValidationError):
            ColumnAnalysis(column_name="")

    def test_column_analysis_valid(self):
        ca = ColumnAnalysis(column_name="temperature", missing_count=5, total_values=100)
        assert ca.column_name == "temperature"
        assert ca.missing_count == 5

    def test_column_analysis_missing_pct_bounds(self):
        with pytest.raises(ValidationError):
            ColumnAnalysis(column_name="x", missing_pct=1.5)

    def test_missingness_report_defaults(self):
        r = MissingnessReport()
        assert r.total_records == 0
        assert r.columns == []

    def test_rule_condition_requires_field_name(self):
        with pytest.raises(ValidationError):
            RuleCondition(field_name="", condition_type=RuleConditionType.EQUALS)

    def test_rule_condition_valid(self):
        rc = RuleCondition(field_name="category", condition_type=RuleConditionType.EQUALS, value="office")
        assert rc.field_name == "category"

    def test_imputation_rule_requires_name(self):
        with pytest.raises(ValidationError):
            ImputationRule(name="", target_column="col")

    def test_imputation_rule_requires_target_column(self):
        with pytest.raises(ValidationError):
            ImputationRule(name="test", target_column="")

    def test_imputation_rule_valid(self):
        rule = ImputationRule(name="test_rule", target_column="emission_factor")
        assert rule.active is True
        assert rule.priority == RulePriority.MEDIUM

    def test_lookup_entry_requires_key(self):
        with pytest.raises(ValidationError):
            LookupEntry(key="", value=1.0)

    def test_lookup_table_valid(self):
        lt = LookupTable(name="fuel_factors", key_column="fuel_type", target_column="ef")
        assert len(lt.table_id) == 36

    def test_imputed_value_requires_column_name(self):
        with pytest.raises(ValidationError):
            ImputedValue(record_index=0, column_name="", imputed_value=1.0, strategy=ImputationStrategy.MEAN)

    def test_imputed_value_valid(self):
        iv = ImputedValue(record_index=0, column_name="temp", imputed_value=22.5, strategy=ImputationStrategy.MEAN)
        assert iv.confidence == 0.0
        assert iv.confidence_level == ConfidenceLevel.MEDIUM

    def test_imputation_result_requires_column_name(self):
        with pytest.raises(ValidationError):
            ImputationResult(column_name="", strategy=ImputationStrategy.MEAN)

    def test_imputation_batch_defaults(self):
        b = ImputationBatch()
        assert b.total_values_imputed == 0
        assert b.results == []

    def test_validation_result_requires_column_name(self):
        with pytest.raises(ValidationError):
            ValidationResult(column_name="", method=ValidationMethod.KS_TEST)

    def test_validation_result_p_value_bounds(self):
        with pytest.raises(ValidationError):
            ValidationResult(column_name="x", method=ValidationMethod.KS_TEST, p_value=1.5)

    def test_validation_report_defaults(self):
        vr = ValidationReport()
        assert vr.overall_passed is False
        assert vr.results == []

    def test_imputation_template_requires_name(self):
        with pytest.raises(ValidationError):
            ImputationTemplate(name="")

    def test_imputation_template_valid(self):
        t = ImputationTemplate(name="standard")
        assert t.active is True
        assert t.default_strategy == ImputationStrategy.MEAN

    def test_time_series_config_requires_time_column(self):
        with pytest.raises(ValidationError):
            TimeSeriesConfig(time_column="")

    def test_time_series_config_seasonal_period_min(self):
        with pytest.raises(ValidationError):
            TimeSeriesConfig(time_column="date", seasonal_period=1)

    def test_ml_model_config_defaults(self):
        mc = MLModelConfig()
        assert mc.n_estimators == 100
        assert mc.random_state == 42

    def test_ml_model_config_n_estimators_bounds(self):
        with pytest.raises(ValidationError):
            MLModelConfig(n_estimators=0)
        with pytest.raises(ValidationError):
            MLModelConfig(n_estimators=1001)

    def test_strategy_selection_requires_column_name(self):
        with pytest.raises(ValidationError):
            StrategySelection(column_name="", recommended_strategy=ImputationStrategy.MEAN)

    def test_pipeline_config_defaults(self):
        pc = PipelineConfig()
        assert pc.confidence_threshold == 0.7
        assert pc.enable_ml is True

    def test_pipeline_result_defaults(self):
        pr = PipelineResult()
        assert pr.status == ImputationStatus.COMPLETED

    def test_imputation_job_config_is_active(self):
        jc = ImputationJobConfig(status=ImputationStatus.PENDING)
        assert jc.is_active is True

    def test_imputation_job_config_not_active_when_completed(self):
        jc = ImputationJobConfig(status=ImputationStatus.COMPLETED)
        assert jc.is_active is False

    def test_imputation_job_config_progress_pct(self):
        jc = ImputationJobConfig(status=ImputationStatus.PENDING, stage=PipelineStage.ANALYZE)
        assert jc.progress_pct == 15.0

    def test_imputation_job_config_progress_completed(self):
        jc = ImputationJobConfig(status=ImputationStatus.COMPLETED)
        assert jc.progress_pct == 100.0

    def test_imputation_job_config_progress_failed(self):
        jc = ImputationJobConfig(status=ImputationStatus.FAILED)
        assert jc.progress_pct == 0.0

    def test_imputation_statistics_defaults(self):
        stats = ImputationStatistics()
        assert stats.total_jobs == 0
        assert stats.avg_confidence == 0.0


# =============================================================================
# Request model tests
# =============================================================================


class TestRequestModels:
    """Test all 8 request models."""

    def test_create_job_request_requires_records(self):
        with pytest.raises(ValidationError):
            CreateJobRequest(records=[])

    def test_create_job_request_valid(self):
        req = CreateJobRequest(records=[{"a": 1}])
        assert len(req.records) == 1

    def test_analyze_missingness_request_valid(self):
        req = AnalyzeMissingnessRequest(records=[{"a": None}])
        assert req.columns == []

    def test_impute_values_request_valid(self):
        req = ImputeValuesRequest(records=[{"a": None}])
        assert req.confidence_threshold == 0.7

    def test_batch_impute_request_requires_datasets(self):
        with pytest.raises(ValidationError):
            BatchImputeRequest(datasets=[])

    def test_validate_request_valid(self):
        req = ValidateRequest(
            original_records=[{"a": 1}],
            imputed_records=[{"a": 1.1}],
        )
        assert req.methods[0] == ValidationMethod.PLAUSIBILITY_RANGE

    def test_create_rule_request_requires_name(self):
        with pytest.raises(ValidationError):
            CreateRuleRequest(name="", target_column="col")

    def test_create_template_request_requires_name(self):
        with pytest.raises(ValidationError):
            CreateTemplateRequest(name="")

    def test_run_pipeline_request_valid(self):
        req = RunPipelineRequest(records=[{"a": 1}])
        assert req.dataset_id == ""


# =============================================================================
# Constants tests
# =============================================================================


class TestConstants:
    """Test module-level constants."""

    def test_default_confidence_thresholds(self):
        assert DEFAULT_CONFIDENCE_THRESHOLDS["high"] == 0.85
        assert DEFAULT_CONFIDENCE_THRESHOLDS["very_low"] == 0.30

    def test_strategy_by_column_type(self):
        assert STRATEGY_BY_COLUMN_TYPE["numeric"] == "mean"
        assert STRATEGY_BY_COLUMN_TYPE["categorical"] == "mode"

    def test_knn_neighbors_by_size(self):
        assert KNN_NEIGHBORS_BY_SIZE["small"] == 3
        assert KNN_NEIGHBORS_BY_SIZE["very_large"] == 11

    def test_imputation_strategies_tuple(self):
        assert "mean" in IMPUTATION_STRATEGIES
        assert "mice" in IMPUTATION_STRATEGIES
        assert len(IMPUTATION_STRATEGIES) == 19

    def test_validation_methods_tuple(self):
        assert "ks_test" in VALIDATION_METHODS
        assert len(VALIDATION_METHODS) == 5

    def test_pipeline_stage_order(self):
        assert PIPELINE_STAGE_ORDER == ("analyze", "strategize", "impute", "validate", "document")
