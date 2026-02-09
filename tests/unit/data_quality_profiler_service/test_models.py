# -*- coding: utf-8 -*-
"""
Unit Tests for Data Quality Profiler Models - AGENT-DATA-010

Tests all enumerations (13), SDK models (15), request models (7), utility
helpers, constants, and re-exported Layer 1 symbols from
``greenlang.data_quality_profiler.models``.

Target: 230+ tests, 85%+ coverage of greenlang.data_quality_profiler.models

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

import pytest
from pydantic import ValidationError

from greenlang.data_quality_profiler.models import (
    # Enumerations (13)
    QualityDimension,
    DataType,
    ProfileStatus,
    AssessmentStatus,
    AnomalyMethod,
    AnomalySeverity,
    RuleType,
    RuleOperator,
    GateOutcome,
    IssueSeverity,
    MissingPattern,
    ReportFormat,
    TrendDirection,
    # SDK models (15)
    ColumnProfile,
    DatasetProfile,
    DimensionScore,
    QualityAssessment,
    QualityIssue,
    AnomalyResult,
    FreshnessResult,
    QualityRule,
    RuleEvaluation,
    QualityGate,
    QualityTrend,
    QualityScorecardRow,
    QualityScorecard,
    DataQualityProfilerStatistics,
    ProfileSummary,
    # Request models (7)
    ProfileDatasetRequest,
    AssessQualityRequest,
    ValidateDatasetRequest,
    DetectAnomaliesRequest,
    CheckFreshnessRequest,
    CreateRuleRequest,
    GenerateReportRequest,
    # Constants
    DEFAULT_DIMENSION_WEIGHTS,
    QUALITY_LEVEL_THRESHOLDS,
    DEFAULT_ANOMALY_THRESHOLDS,
    FRESHNESS_BOUNDARIES_HOURS,
    ALL_QUALITY_DIMENSIONS,
    SUPPORTED_DATA_TYPES,
    ANOMALY_METHOD_NAMES,
    REPORT_FORMAT_OPTIONS,
    GATE_OUTCOME_VALUES,
    ISSUE_SEVERITY_ORDER,
    # Helper
    _utcnow,
    # Layer 1 re-exports
    QualityLevel,
    DataQualityReport,
    DataQualityScorer,
)


# ============================================================================
# Enum Tests (13 enums)
# ============================================================================


class TestQualityDimension:
    """QualityDimension enum tests."""

    def test_member_count(self):
        assert len(QualityDimension) == 6

    def test_completeness_value(self):
        assert QualityDimension.COMPLETENESS.value == "completeness"

    def test_validity_value(self):
        assert QualityDimension.VALIDITY.value == "validity"

    def test_consistency_value(self):
        assert QualityDimension.CONSISTENCY.value == "consistency"

    def test_timeliness_value(self):
        assert QualityDimension.TIMELINESS.value == "timeliness"

    def test_uniqueness_value(self):
        assert QualityDimension.UNIQUENESS.value == "uniqueness"

    def test_accuracy_value(self):
        assert QualityDimension.ACCURACY.value == "accuracy"

    def test_is_str_enum(self):
        assert isinstance(QualityDimension.COMPLETENESS, str)

    def test_is_enum(self):
        assert issubclass(QualityDimension, Enum)


class TestDataType:
    """DataType enum tests."""

    def test_member_count(self):
        assert len(DataType) == 13

    @pytest.mark.parametrize("member,value", [
        ("STRING", "string"),
        ("INTEGER", "integer"),
        ("FLOAT", "float"),
        ("BOOLEAN", "boolean"),
        ("DATE", "date"),
        ("DATETIME", "datetime"),
        ("EMAIL", "email"),
        ("URL", "url"),
        ("PHONE", "phone"),
        ("IP_ADDRESS", "ip_address"),
        ("UUID", "uuid"),
        ("JSON", "json"),
        ("UNKNOWN", "unknown"),
    ])
    def test_member_value(self, member, value):
        assert DataType[member].value == value

    def test_is_str_enum(self):
        assert isinstance(DataType.STRING, str)


class TestProfileStatus:
    """ProfileStatus enum tests."""

    def test_member_count(self):
        assert len(ProfileStatus) == 4

    @pytest.mark.parametrize("member,value", [
        ("PENDING", "pending"),
        ("RUNNING", "running"),
        ("COMPLETED", "completed"),
        ("FAILED", "failed"),
    ])
    def test_member_value(self, member, value):
        assert ProfileStatus[member].value == value

    def test_is_str_enum(self):
        assert isinstance(ProfileStatus.PENDING, str)


class TestAssessmentStatus:
    """AssessmentStatus enum tests."""

    def test_member_count(self):
        assert len(AssessmentStatus) == 5

    @pytest.mark.parametrize("member,value", [
        ("PENDING", "pending"),
        ("RUNNING", "running"),
        ("PASSED", "passed"),
        ("WARNING", "warning"),
        ("FAILED", "failed"),
    ])
    def test_member_value(self, member, value):
        assert AssessmentStatus[member].value == value

    def test_is_str_enum(self):
        assert isinstance(AssessmentStatus.PASSED, str)


class TestAnomalyMethod:
    """AnomalyMethod enum tests."""

    def test_member_count(self):
        assert len(AnomalyMethod) == 5

    @pytest.mark.parametrize("member,value", [
        ("IQR", "iqr"),
        ("ZSCORE", "zscore"),
        ("MAD", "mad"),
        ("GRUBBS", "grubbs"),
        ("MODIFIED_ZSCORE", "modified_zscore"),
    ])
    def test_member_value(self, member, value):
        assert AnomalyMethod[member].value == value

    def test_is_str_enum(self):
        assert isinstance(AnomalyMethod.IQR, str)


class TestAnomalySeverity:
    """AnomalySeverity enum tests."""

    def test_member_count(self):
        assert len(AnomalySeverity) == 4

    @pytest.mark.parametrize("member,value", [
        ("LOW", "low"),
        ("MEDIUM", "medium"),
        ("HIGH", "high"),
        ("CRITICAL", "critical"),
    ])
    def test_member_value(self, member, value):
        assert AnomalySeverity[member].value == value

    def test_is_str_enum(self):
        assert isinstance(AnomalySeverity.LOW, str)


class TestRuleType:
    """RuleType enum tests."""

    def test_member_count(self):
        assert len(RuleType) == 6

    @pytest.mark.parametrize("member,value", [
        ("COMPLETENESS", "completeness"),
        ("RANGE", "range"),
        ("FORMAT", "format"),
        ("UNIQUENESS", "uniqueness"),
        ("CUSTOM", "custom"),
        ("FRESHNESS", "freshness"),
    ])
    def test_member_value(self, member, value):
        assert RuleType[member].value == value

    def test_is_str_enum(self):
        assert isinstance(RuleType.COMPLETENESS, str)


class TestRuleOperator:
    """RuleOperator enum tests."""

    def test_member_count(self):
        assert len(RuleOperator) == 8

    @pytest.mark.parametrize("member,value", [
        ("EQUALS", "equals"),
        ("NOT_EQUALS", "not_equals"),
        ("GREATER_THAN", "greater_than"),
        ("LESS_THAN", "less_than"),
        ("BETWEEN", "between"),
        ("MATCHES", "matches"),
        ("CONTAINS", "contains"),
        ("IN_SET", "in_set"),
    ])
    def test_member_value(self, member, value):
        assert RuleOperator[member].value == value

    def test_is_str_enum(self):
        assert isinstance(RuleOperator.EQUALS, str)


class TestGateOutcome:
    """GateOutcome enum tests."""

    def test_member_count(self):
        assert len(GateOutcome) == 3

    @pytest.mark.parametrize("member,value", [
        ("PASS", "pass"),
        ("WARN", "warn"),
        ("FAIL", "fail"),
    ])
    def test_member_value(self, member, value):
        assert GateOutcome[member].value == value

    def test_is_str_enum(self):
        assert isinstance(GateOutcome.PASS, str)


class TestIssueSeverity:
    """IssueSeverity enum tests."""

    def test_member_count(self):
        assert len(IssueSeverity) == 4

    @pytest.mark.parametrize("member,value", [
        ("INFO", "info"),
        ("WARNING", "warning"),
        ("ERROR", "error"),
        ("CRITICAL", "critical"),
    ])
    def test_member_value(self, member, value):
        assert IssueSeverity[member].value == value

    def test_is_str_enum(self):
        assert isinstance(IssueSeverity.INFO, str)


class TestMissingPattern:
    """MissingPattern enum tests."""

    def test_member_count(self):
        assert len(MissingPattern) == 4

    @pytest.mark.parametrize("member,value", [
        ("MCAR", "mcar"),
        ("MAR", "mar"),
        ("MNAR", "mnar"),
        ("UNKNOWN", "unknown"),
    ])
    def test_member_value(self, member, value):
        assert MissingPattern[member].value == value

    def test_is_str_enum(self):
        assert isinstance(MissingPattern.MCAR, str)


class TestReportFormat:
    """ReportFormat enum tests."""

    def test_member_count(self):
        assert len(ReportFormat) == 5

    @pytest.mark.parametrize("member,value", [
        ("JSON", "json"),
        ("MARKDOWN", "markdown"),
        ("HTML", "html"),
        ("TEXT", "text"),
        ("CSV", "csv"),
    ])
    def test_member_value(self, member, value):
        assert ReportFormat[member].value == value

    def test_is_str_enum(self):
        assert isinstance(ReportFormat.JSON, str)


class TestTrendDirection:
    """TrendDirection enum tests."""

    def test_member_count(self):
        assert len(TrendDirection) == 4

    @pytest.mark.parametrize("member,value", [
        ("IMPROVING", "improving"),
        ("STABLE", "stable"),
        ("DEGRADING", "degrading"),
        ("UNKNOWN", "unknown"),
    ])
    def test_member_value(self, member, value):
        assert TrendDirection[member].value == value

    def test_is_str_enum(self):
        assert isinstance(TrendDirection.IMPROVING, str)


# ============================================================================
# Helper Tests
# ============================================================================


class TestUtcnowHelper:
    """_utcnow helper function tests."""

    def test_returns_datetime(self):
        result = _utcnow()
        assert isinstance(result, datetime)

    def test_is_utc(self):
        result = _utcnow()
        assert result.tzinfo == timezone.utc

    def test_microseconds_zeroed(self):
        result = _utcnow()
        assert result.microsecond == 0


# ============================================================================
# SDK Model Tests (15 models)
# ============================================================================


class TestColumnProfile:
    """ColumnProfile model tests."""

    def test_minimal_creation(self):
        cp = ColumnProfile(name="age")
        assert cp.name == "age"

    def test_default_data_type(self):
        cp = ColumnProfile(name="col")
        assert cp.data_type == DataType.UNKNOWN

    def test_default_total(self):
        cp = ColumnProfile(name="col")
        assert cp.total == 0

    def test_default_non_null(self):
        cp = ColumnProfile(name="col")
        assert cp.non_null == 0

    def test_default_null_count(self):
        cp = ColumnProfile(name="col")
        assert cp.null_count == 0

    def test_default_null_pct(self):
        cp = ColumnProfile(name="col")
        assert cp.null_pct == 0.0

    def test_default_unique_count(self):
        cp = ColumnProfile(name="col")
        assert cp.unique_count == 0

    def test_default_cardinality(self):
        cp = ColumnProfile(name="col")
        assert cp.cardinality == 0.0

    def test_default_optional_fields_none(self):
        cp = ColumnProfile(name="col")
        assert cp.min_val is None
        assert cp.max_val is None
        assert cp.mean is None
        assert cp.median is None
        assert cp.stddev is None
        assert cp.p25 is None
        assert cp.p50 is None
        assert cp.p75 is None
        assert cp.p95 is None
        assert cp.p99 is None
        assert cp.pattern is None

    def test_default_most_common(self):
        cp = ColumnProfile(name="col")
        assert cp.most_common == []

    def test_default_missing_pattern(self):
        cp = ColumnProfile(name="col")
        assert cp.missing_pattern == MissingPattern.UNKNOWN

    def test_default_provenance_hash(self):
        cp = ColumnProfile(name="col")
        assert cp.provenance_hash == ""

    def test_validator_empty_name_raises(self):
        with pytest.raises(ValidationError):
            ColumnProfile(name="")

    def test_validator_whitespace_name_raises(self):
        with pytest.raises(ValidationError):
            ColumnProfile(name="   ")

    def test_model_dump_keys(self):
        cp = ColumnProfile(name="test")
        d = cp.model_dump()
        assert "name" in d
        assert "data_type" in d

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            ColumnProfile(name="test", unknown_field=42)

    def test_full_creation(self):
        cp = ColumnProfile(
            name="salary",
            data_type=DataType.FLOAT,
            total=100,
            non_null=95,
            null_count=5,
            null_pct=5.0,
            unique_count=90,
            cardinality=0.95,
            min_val="30000",
            max_val="200000",
            mean=75000.0,
            median=72000.0,
            stddev=15000.0,
            p25=60000.0,
            p50=72000.0,
            p75=90000.0,
            p95=150000.0,
            p99=195000.0,
        )
        assert cp.total == 100
        assert cp.null_pct == pytest.approx(5.0)

    def test_null_pct_boundary_zero(self):
        cp = ColumnProfile(name="c", null_pct=0.0)
        assert cp.null_pct == 0.0

    def test_null_pct_boundary_hundred(self):
        cp = ColumnProfile(name="c", null_pct=100.0)
        assert cp.null_pct == 100.0

    def test_null_pct_over_hundred_raises(self):
        with pytest.raises(ValidationError):
            ColumnProfile(name="c", null_pct=100.1)

    def test_negative_total_raises(self):
        with pytest.raises(ValidationError):
            ColumnProfile(name="c", total=-1)


class TestDatasetProfile:
    """DatasetProfile model tests."""

    def test_minimal_creation(self):
        dp = DatasetProfile(dataset_name="test_ds")
        assert dp.dataset_name == "test_ds"

    def test_default_profile_id_is_uuid(self):
        dp = DatasetProfile(dataset_name="ds")
        uuid.UUID(dp.profile_id)  # should not raise

    def test_default_row_count(self):
        dp = DatasetProfile(dataset_name="ds")
        assert dp.row_count == 0

    def test_default_column_count(self):
        dp = DatasetProfile(dataset_name="ds")
        assert dp.column_count == 0

    def test_default_columns_empty(self):
        dp = DatasetProfile(dataset_name="ds")
        assert dp.columns == []

    def test_default_status(self):
        dp = DatasetProfile(dataset_name="ds")
        assert dp.status == ProfileStatus.PENDING

    def test_default_profiling_duration(self):
        dp = DatasetProfile(dataset_name="ds")
        assert dp.profiling_duration_ms == 0.0

    def test_default_created_at_is_datetime(self):
        dp = DatasetProfile(dataset_name="ds")
        assert isinstance(dp.created_at, datetime)

    def test_default_profiled_at_none(self):
        dp = DatasetProfile(dataset_name="ds")
        assert dp.profiled_at is None

    def test_default_provenance_hash(self):
        dp = DatasetProfile(dataset_name="ds")
        assert dp.provenance_hash == ""

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError):
            DatasetProfile(dataset_name="")

    def test_whitespace_name_raises(self):
        with pytest.raises(ValidationError):
            DatasetProfile(dataset_name="  ")

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            DatasetProfile(dataset_name="ds", foo="bar")

    def test_model_dump(self):
        dp = DatasetProfile(dataset_name="ds", row_count=10)
        d = dp.model_dump()
        assert d["row_count"] == 10


class TestDimensionScore:
    """DimensionScore model tests."""

    def test_creation(self):
        ds = DimensionScore(
            dimension=QualityDimension.COMPLETENESS,
            score=0.95,
            weight=0.20,
            weighted_score=0.19,
        )
        assert ds.dimension == QualityDimension.COMPLETENESS

    def test_default_details_empty(self):
        ds = DimensionScore(
            dimension=QualityDimension.VALIDITY,
            score=0.80,
            weight=0.20,
            weighted_score=0.16,
        )
        assert ds.details == {}

    def test_default_issues_count(self):
        ds = DimensionScore(
            dimension=QualityDimension.ACCURACY,
            score=0.90,
            weight=0.10,
            weighted_score=0.09,
        )
        assert ds.issues_count == 0

    def test_score_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            DimensionScore(
                dimension=QualityDimension.TIMELINESS,
                score=1.5,
                weight=0.15,
                weighted_score=0.225,
            )

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            DimensionScore(
                dimension=QualityDimension.UNIQUENESS,
                score=0.5,
                weight=0.15,
                weighted_score=0.075,
                extra_field="x",
            )


class TestQualityIssue:
    """QualityIssue model tests."""

    def test_creation(self):
        qi = QualityIssue(
            dimension=QualityDimension.COMPLETENESS,
            description="Missing values in column X",
        )
        assert qi.description == "Missing values in column X"

    def test_default_severity(self):
        qi = QualityIssue(
            dimension=QualityDimension.VALIDITY,
            description="Invalid format",
        )
        assert qi.severity == IssueSeverity.WARNING

    def test_default_issue_id_is_uuid(self):
        qi = QualityIssue(
            dimension=QualityDimension.CONSISTENCY,
            description="Inconsistent values",
        )
        uuid.UUID(qi.issue_id)

    def test_empty_description_raises(self):
        with pytest.raises(ValidationError):
            QualityIssue(
                dimension=QualityDimension.COMPLETENESS,
                description="",
            )

    def test_whitespace_description_raises(self):
        with pytest.raises(ValidationError):
            QualityIssue(
                dimension=QualityDimension.COMPLETENESS,
                description="   ",
            )

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            QualityIssue(
                dimension=QualityDimension.COMPLETENESS,
                description="desc",
                unknown=True,
            )

    def test_row_index_negative_raises(self):
        with pytest.raises(ValidationError):
            QualityIssue(
                dimension=QualityDimension.VALIDITY,
                description="row issue",
                row_index=-1,
            )


class TestQualityAssessment:
    """QualityAssessment model tests."""

    def test_creation(self):
        qa = QualityAssessment(dataset_name="test_ds")
        assert qa.dataset_name == "test_ds"

    def test_default_overall_score(self):
        qa = QualityAssessment(dataset_name="ds")
        assert qa.overall_score == 0.0

    def test_default_status(self):
        qa = QualityAssessment(dataset_name="ds")
        assert qa.status == AssessmentStatus.PENDING

    def test_default_total_issues(self):
        qa = QualityAssessment(dataset_name="ds")
        assert qa.total_issues == 0

    def test_default_issues_empty(self):
        qa = QualityAssessment(dataset_name="ds")
        assert qa.issues == []

    def test_default_dimensions_empty(self):
        qa = QualityAssessment(dataset_name="ds")
        assert qa.dimensions == []

    def test_default_assessment_id_uuid(self):
        qa = QualityAssessment(dataset_name="ds")
        uuid.UUID(qa.assessment_id)

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError):
            QualityAssessment(dataset_name="")

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            QualityAssessment(dataset_name="ds", foo=1)

    def test_model_dump(self):
        qa = QualityAssessment(dataset_name="ds", overall_score=0.85)
        d = qa.model_dump()
        assert d["overall_score"] == pytest.approx(0.85)


class TestAnomalyResult:
    """AnomalyResult model tests."""

    def test_creation(self):
        ar = AnomalyResult(column="salary", method=AnomalyMethod.IQR)
        assert ar.column == "salary"

    def test_default_severity(self):
        ar = AnomalyResult(column="age", method=AnomalyMethod.ZSCORE)
        assert ar.severity == AnomalySeverity.MEDIUM

    def test_default_value_none(self):
        ar = AnomalyResult(column="c", method=AnomalyMethod.MAD)
        assert ar.value is None

    def test_default_row_indices_empty(self):
        ar = AnomalyResult(column="c", method=AnomalyMethod.GRUBBS)
        assert ar.row_indices == []

    def test_default_anomaly_id_uuid(self):
        ar = AnomalyResult(column="c", method=AnomalyMethod.IQR)
        uuid.UUID(ar.anomaly_id)

    def test_empty_column_raises(self):
        with pytest.raises(ValidationError):
            AnomalyResult(column="", method=AnomalyMethod.IQR)

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            AnomalyResult(column="c", method=AnomalyMethod.IQR, extra=1)


class TestFreshnessResult:
    """FreshnessResult model tests."""

    def test_creation(self):
        fr = FreshnessResult(
            dataset_name="ds",
            last_updated="2025-01-01T00:00:00Z",
            age_hours=48.0,
            freshness_level=QualityLevel.GOOD,
            sla_hours=72,
            sla_compliant=True,
        )
        assert fr.sla_compliant is True

    def test_empty_dataset_name_raises(self):
        with pytest.raises(ValidationError):
            FreshnessResult(
                dataset_name="",
                last_updated="2025-01-01T00:00:00Z",
                age_hours=10.0,
                freshness_level=QualityLevel.EXCELLENT,
                sla_hours=24,
                sla_compliant=True,
            )

    def test_empty_last_updated_raises(self):
        with pytest.raises(ValidationError):
            FreshnessResult(
                dataset_name="ds",
                last_updated="",
                age_hours=10.0,
                freshness_level=QualityLevel.EXCELLENT,
                sla_hours=24,
                sla_compliant=True,
            )

    def test_negative_age_raises(self):
        with pytest.raises(ValidationError):
            FreshnessResult(
                dataset_name="ds",
                last_updated="2025-01-01T00:00:00Z",
                age_hours=-1.0,
                freshness_level=QualityLevel.EXCELLENT,
                sla_hours=24,
                sla_compliant=True,
            )

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            FreshnessResult(
                dataset_name="ds",
                last_updated="2025-01-01T00:00:00Z",
                age_hours=10.0,
                freshness_level=QualityLevel.EXCELLENT,
                sla_hours=24,
                sla_compliant=True,
                extra_field=True,
            )


class TestQualityRule:
    """QualityRule model tests."""

    def test_creation(self):
        qr = QualityRule(name="test_rule", rule_type=RuleType.COMPLETENESS)
        assert qr.name == "test_rule"

    def test_default_operator(self):
        qr = QualityRule(name="r", rule_type=RuleType.RANGE)
        assert qr.operator == RuleOperator.GREATER_THAN

    def test_default_active(self):
        qr = QualityRule(name="r", rule_type=RuleType.FORMAT)
        assert qr.active is True

    def test_default_priority(self):
        qr = QualityRule(name="r", rule_type=RuleType.UNIQUENESS)
        assert qr.priority == 100

    def test_default_parameters_empty(self):
        qr = QualityRule(name="r", rule_type=RuleType.CUSTOM)
        assert qr.parameters == {}

    def test_default_rule_id_uuid(self):
        qr = QualityRule(name="r", rule_type=RuleType.FRESHNESS)
        uuid.UUID(qr.rule_id)

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError):
            QualityRule(name="", rule_type=RuleType.COMPLETENESS)

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            QualityRule(name="r", rule_type=RuleType.RANGE, foo=1)

    def test_negative_priority_raises(self):
        with pytest.raises(ValidationError):
            QualityRule(name="r", rule_type=RuleType.RANGE, priority=-1)


class TestRuleEvaluation:
    """RuleEvaluation model tests."""

    def test_creation(self):
        re_ = RuleEvaluation(rule_id="rule-123", passed=True)
        assert re_.passed is True

    def test_default_rule_name(self):
        re_ = RuleEvaluation(rule_id="r1", passed=False)
        assert re_.rule_name == ""

    def test_default_message(self):
        re_ = RuleEvaluation(rule_id="r1", passed=True)
        assert re_.message == ""

    def test_default_evaluation_id_uuid(self):
        re_ = RuleEvaluation(rule_id="r1", passed=True)
        uuid.UUID(re_.evaluation_id)

    def test_empty_rule_id_raises(self):
        with pytest.raises(ValidationError):
            RuleEvaluation(rule_id="", passed=True)

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            RuleEvaluation(rule_id="r1", passed=True, extra="x")


class TestQualityGate:
    """QualityGate model tests."""

    def test_creation(self):
        qg = QualityGate(name="production_gate")
        assert qg.name == "production_gate"

    def test_default_outcome(self):
        qg = QualityGate(name="gate")
        assert qg.outcome == GateOutcome.FAIL

    def test_default_threshold(self):
        qg = QualityGate(name="gate")
        assert qg.threshold == pytest.approx(0.70)

    def test_default_overall_score(self):
        qg = QualityGate(name="gate")
        assert qg.overall_score == 0.0

    def test_default_conditions_empty(self):
        qg = QualityGate(name="gate")
        assert qg.conditions == []

    def test_default_evaluations_empty(self):
        qg = QualityGate(name="gate")
        assert qg.evaluations == []

    def test_default_gate_id_uuid(self):
        qg = QualityGate(name="gate")
        uuid.UUID(qg.gate_id)

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError):
            QualityGate(name="")

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            QualityGate(name="gate", junk=1)


class TestQualityTrend:
    """QualityTrend model tests."""

    def test_creation(self):
        qt = QualityTrend(dataset_name="ds")
        assert qt.dataset_name == "ds"

    def test_default_direction(self):
        qt = QualityTrend(dataset_name="ds")
        assert qt.direction == TrendDirection.UNKNOWN

    def test_default_scores_empty(self):
        qt = QualityTrend(dataset_name="ds")
        assert qt.scores == []

    def test_default_change_pct(self):
        qt = QualityTrend(dataset_name="ds")
        assert qt.change_pct == 0.0

    def test_default_data_points(self):
        qt = QualityTrend(dataset_name="ds")
        assert qt.data_points == 0

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError):
            QualityTrend(dataset_name="")

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            QualityTrend(dataset_name="ds", bogus=True)


class TestQualityScorecardRow:
    """QualityScorecardRow model tests."""

    def test_creation(self):
        row = QualityScorecardRow(
            dimension=QualityDimension.COMPLETENESS,
            score=0.95,
            weight=0.20,
            weighted_score=0.19,
        )
        assert row.score == pytest.approx(0.95)

    def test_default_issues_count(self):
        row = QualityScorecardRow(
            dimension=QualityDimension.VALIDITY,
            score=0.80,
            weight=0.20,
            weighted_score=0.16,
        )
        assert row.issues_count == 0

    def test_default_trend(self):
        row = QualityScorecardRow(
            dimension=QualityDimension.ACCURACY,
            score=0.90,
            weight=0.10,
            weighted_score=0.09,
        )
        assert row.trend == TrendDirection.UNKNOWN

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            QualityScorecardRow(
                dimension=QualityDimension.COMPLETENESS,
                score=0.95,
                weight=0.20,
                weighted_score=0.19,
                extra=1,
            )


class TestQualityScorecard:
    """QualityScorecard model tests."""

    def test_creation(self):
        sc = QualityScorecard(dataset_name="ds")
        assert sc.dataset_name == "ds"

    def test_default_overall_score(self):
        sc = QualityScorecard(dataset_name="ds")
        assert sc.overall_score == 0.0

    def test_default_rows_empty(self):
        sc = QualityScorecard(dataset_name="ds")
        assert sc.rows == []

    def test_default_total_issues(self):
        sc = QualityScorecard(dataset_name="ds")
        assert sc.total_issues == 0

    def test_default_scorecard_id_uuid(self):
        sc = QualityScorecard(dataset_name="ds")
        uuid.UUID(sc.scorecard_id)

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError):
            QualityScorecard(dataset_name="")

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            QualityScorecard(dataset_name="ds", nope=1)


class TestDataQualityProfilerStatistics:
    """DataQualityProfilerStatistics model tests."""

    def test_creation(self):
        stats = DataQualityProfilerStatistics()
        assert stats.datasets_profiled == 0

    def test_default_assessments_completed(self):
        stats = DataQualityProfilerStatistics()
        assert stats.assessments_completed == 0

    def test_default_rules_evaluated(self):
        stats = DataQualityProfilerStatistics()
        assert stats.rules_evaluated == 0

    def test_default_anomalies_detected(self):
        stats = DataQualityProfilerStatistics()
        assert stats.anomalies_detected == 0

    def test_default_gates_evaluated(self):
        stats = DataQualityProfilerStatistics()
        assert stats.gates_evaluated == 0

    def test_default_avg_quality_score(self):
        stats = DataQualityProfilerStatistics()
        assert stats.avg_quality_score == 0.0

    def test_default_by_dimension_empty(self):
        stats = DataQualityProfilerStatistics()
        assert stats.by_dimension == {}

    def test_default_by_quality_level_empty(self):
        stats = DataQualityProfilerStatistics()
        assert stats.by_quality_level == {}

    def test_default_timestamp_is_datetime(self):
        stats = DataQualityProfilerStatistics()
        assert isinstance(stats.timestamp, datetime)

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            DataQualityProfilerStatistics(extra=1)


class TestProfileSummary:
    """ProfileSummary model tests."""

    def test_creation(self):
        ps = ProfileSummary(profile_id="p1", dataset_name="ds")
        assert ps.profile_id == "p1"

    def test_default_row_count(self):
        ps = ProfileSummary(profile_id="p1", dataset_name="ds")
        assert ps.row_count == 0

    def test_default_column_count(self):
        ps = ProfileSummary(profile_id="p1", dataset_name="ds")
        assert ps.column_count == 0

    def test_default_overall_quality(self):
        ps = ProfileSummary(profile_id="p1", dataset_name="ds")
        assert ps.overall_quality == 0.0

    def test_default_status(self):
        ps = ProfileSummary(profile_id="p1", dataset_name="ds")
        assert ps.status == ProfileStatus.COMPLETED

    def test_default_created_at(self):
        ps = ProfileSummary(profile_id="p1", dataset_name="ds")
        assert isinstance(ps.created_at, datetime)

    def test_empty_profile_id_raises(self):
        with pytest.raises(ValidationError):
            ProfileSummary(profile_id="", dataset_name="ds")

    def test_empty_dataset_name_raises(self):
        with pytest.raises(ValidationError):
            ProfileSummary(profile_id="p1", dataset_name="")

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            ProfileSummary(profile_id="p1", dataset_name="ds", extra=1)

    def test_model_dump(self):
        ps = ProfileSummary(profile_id="p1", dataset_name="ds", row_count=50)
        d = ps.model_dump()
        assert d["row_count"] == 50


# ============================================================================
# Request Model Tests (7 models)
# ============================================================================


class TestProfileDatasetRequest:
    """ProfileDatasetRequest model tests."""

    def test_creation(self):
        req = ProfileDatasetRequest(
            dataset_name="ds", data=[{"a": 1}],
        )
        assert req.dataset_name == "ds"

    def test_data_required_nonempty(self):
        with pytest.raises(ValidationError):
            ProfileDatasetRequest(dataset_name="ds", data=[])

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError):
            ProfileDatasetRequest(dataset_name="", data=[{"a": 1}])

    def test_default_columns_none(self):
        req = ProfileDatasetRequest(dataset_name="ds", data=[{"a": 1}])
        assert req.columns is None

    def test_default_enable_schema_inference_none(self):
        req = ProfileDatasetRequest(dataset_name="ds", data=[{"a": 1}])
        assert req.enable_schema_inference is None

    def test_default_enable_cardinality_none(self):
        req = ProfileDatasetRequest(dataset_name="ds", data=[{"a": 1}])
        assert req.enable_cardinality_analysis is None

    def test_default_sample_size_none(self):
        req = ProfileDatasetRequest(dataset_name="ds", data=[{"a": 1}])
        assert req.sample_size is None

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            ProfileDatasetRequest(dataset_name="ds", data=[{"a": 1}], extra=1)


class TestAssessQualityRequest:
    """AssessQualityRequest model tests."""

    def test_creation(self):
        req = AssessQualityRequest(
            dataset_name="ds", data=[{"x": 1}],
        )
        assert req.dataset_name == "ds"

    def test_empty_data_raises(self):
        with pytest.raises(ValidationError):
            AssessQualityRequest(dataset_name="ds", data=[])

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError):
            AssessQualityRequest(dataset_name="", data=[{"x": 1}])

    def test_default_include_issues(self):
        req = AssessQualityRequest(dataset_name="ds", data=[{"x": 1}])
        assert req.include_issues is True

    def test_default_dimensions_none(self):
        req = AssessQualityRequest(dataset_name="ds", data=[{"x": 1}])
        assert req.dimensions is None

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            AssessQualityRequest(dataset_name="ds", data=[{"x": 1}], extra=1)


class TestValidateDatasetRequest:
    """ValidateDatasetRequest model tests."""

    def test_creation(self):
        req = ValidateDatasetRequest(
            dataset_name="ds", data=[{"a": 1}],
        )
        assert req.dataset_name == "ds"

    def test_empty_data_raises(self):
        with pytest.raises(ValidationError):
            ValidateDatasetRequest(dataset_name="ds", data=[])

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError):
            ValidateDatasetRequest(dataset_name="", data=[{"a": 1}])

    def test_default_fail_fast(self):
        req = ValidateDatasetRequest(dataset_name="ds", data=[{"a": 1}])
        assert req.fail_fast is False

    def test_default_rule_ids_none(self):
        req = ValidateDatasetRequest(dataset_name="ds", data=[{"a": 1}])
        assert req.rule_ids is None

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            ValidateDatasetRequest(dataset_name="ds", data=[{"a": 1}], extra=1)


class TestDetectAnomaliesRequest:
    """DetectAnomaliesRequest model tests."""

    def test_creation(self):
        req = DetectAnomaliesRequest(
            dataset_name="ds", data=[{"v": 10.0}],
        )
        assert req.dataset_name == "ds"

    def test_empty_data_raises(self):
        with pytest.raises(ValidationError):
            DetectAnomaliesRequest(dataset_name="ds", data=[])

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError):
            DetectAnomaliesRequest(dataset_name="", data=[{"v": 1}])

    def test_default_method_none(self):
        req = DetectAnomaliesRequest(dataset_name="ds", data=[{"v": 1}])
        assert req.method is None

    def test_default_severity_none(self):
        req = DetectAnomaliesRequest(dataset_name="ds", data=[{"v": 1}])
        assert req.severity_threshold is None

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            DetectAnomaliesRequest(dataset_name="ds", data=[{"v": 1}], extra=1)


class TestCheckFreshnessRequest:
    """CheckFreshnessRequest model tests."""

    def test_creation(self):
        req = CheckFreshnessRequest(
            dataset_name="ds", last_updated="2025-06-01T00:00:00Z",
        )
        assert req.dataset_name == "ds"

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError):
            CheckFreshnessRequest(
                dataset_name="", last_updated="2025-01-01T00:00:00Z",
            )

    def test_empty_last_updated_raises(self):
        with pytest.raises(ValidationError):
            CheckFreshnessRequest(
                dataset_name="ds", last_updated="",
            )

    def test_default_sla_hours_none(self):
        req = CheckFreshnessRequest(
            dataset_name="ds", last_updated="2025-01-01T00:00:00Z",
        )
        assert req.sla_hours is None

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            CheckFreshnessRequest(
                dataset_name="ds",
                last_updated="2025-01-01T00:00:00Z",
                extra=1,
            )


class TestCreateRuleRequest:
    """CreateRuleRequest model tests."""

    def test_creation(self):
        req = CreateRuleRequest(
            name="my_rule", rule_type=RuleType.COMPLETENESS,
        )
        assert req.name == "my_rule"

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError):
            CreateRuleRequest(name="", rule_type=RuleType.RANGE)

    def test_default_operator(self):
        req = CreateRuleRequest(name="r", rule_type=RuleType.FORMAT)
        assert req.operator == RuleOperator.GREATER_THAN

    def test_default_priority(self):
        req = CreateRuleRequest(name="r", rule_type=RuleType.CUSTOM)
        assert req.priority == 100

    def test_default_parameters_empty(self):
        req = CreateRuleRequest(name="r", rule_type=RuleType.FRESHNESS)
        assert req.parameters == {}

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            CreateRuleRequest(
                name="r", rule_type=RuleType.RANGE, extra="x",
            )


class TestGenerateReportRequest:
    """GenerateReportRequest model tests."""

    def test_creation(self):
        req = GenerateReportRequest(dataset_name="ds")
        assert req.dataset_name == "ds"

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError):
            GenerateReportRequest(dataset_name="")

    def test_default_report_type(self):
        req = GenerateReportRequest(dataset_name="ds")
        assert req.report_type == "comprehensive"

    def test_default_format_none(self):
        req = GenerateReportRequest(dataset_name="ds")
        assert req.format is None

    def test_default_include_profile(self):
        req = GenerateReportRequest(dataset_name="ds")
        assert req.include_profile is True

    def test_default_include_assessment(self):
        req = GenerateReportRequest(dataset_name="ds")
        assert req.include_assessment is True

    def test_default_include_anomalies(self):
        req = GenerateReportRequest(dataset_name="ds")
        assert req.include_anomalies is True

    def test_default_include_trends(self):
        req = GenerateReportRequest(dataset_name="ds")
        assert req.include_trends is True

    def test_invalid_report_type_raises(self):
        with pytest.raises(ValidationError):
            GenerateReportRequest(dataset_name="ds", report_type="invalid")

    @pytest.mark.parametrize("rt", [
        "comprehensive", "summary", "executive", "compliance",
    ])
    def test_valid_report_types(self, rt):
        req = GenerateReportRequest(dataset_name="ds", report_type=rt)
        assert req.report_type == rt

    def test_extra_forbidden(self):
        with pytest.raises(ValidationError):
            GenerateReportRequest(dataset_name="ds", extra=1)


# ============================================================================
# Layer 1 Re-Export Tests
# ============================================================================


class TestLayer1ReExports:
    """Verify Layer 1 re-exports are importable and correct."""

    def test_quality_level_importable(self):
        assert QualityLevel is not None

    def test_quality_level_is_enum(self):
        assert issubclass(QualityLevel, Enum)

    def test_quality_level_excellent(self):
        assert QualityLevel.EXCELLENT.value == "excellent"

    def test_quality_level_critical(self):
        assert QualityLevel.CRITICAL.value == "critical"

    def test_quality_level_member_count(self):
        assert len(QualityLevel) == 5

    def test_data_quality_report_importable(self):
        assert DataQualityReport is not None

    def test_data_quality_scorer_importable(self):
        assert DataQualityScorer is not None


# ============================================================================
# Constants Tests
# ============================================================================


class TestConstants:
    """Verify module-level constants."""

    def test_default_dimension_weights_keys(self):
        expected_keys = {
            "completeness", "validity", "consistency",
            "timeliness", "uniqueness", "accuracy",
        }
        assert set(DEFAULT_DIMENSION_WEIGHTS.keys()) == expected_keys

    def test_default_dimension_weights_sum_to_one(self):
        total = sum(DEFAULT_DIMENSION_WEIGHTS.values())
        assert total == pytest.approx(1.0)

    def test_quality_level_thresholds_keys(self):
        expected = {"excellent", "good", "fair", "poor", "critical"}
        assert set(QUALITY_LEVEL_THRESHOLDS.keys()) == expected

    def test_default_anomaly_thresholds_has_iqr(self):
        assert "iqr" in DEFAULT_ANOMALY_THRESHOLDS

    def test_freshness_boundaries_has_excellent(self):
        assert "excellent" in FRESHNESS_BOUNDARIES_HOURS

    def test_all_quality_dimensions_count(self):
        assert len(ALL_QUALITY_DIMENSIONS) == 6

    def test_supported_data_types_count(self):
        assert len(SUPPORTED_DATA_TYPES) == 13

    def test_anomaly_method_names_count(self):
        assert len(ANOMALY_METHOD_NAMES) == 5

    def test_report_format_options_count(self):
        assert len(REPORT_FORMAT_OPTIONS) == 5

    def test_gate_outcome_values_count(self):
        assert len(GATE_OUTCOME_VALUES) == 3

    def test_issue_severity_order_count(self):
        assert len(ISSUE_SEVERITY_ORDER) == 4


# ============================================================================
# Module Exports Tests
# ============================================================================


class TestModuleExports:
    """Verify __all__ completeness."""

    def test_all_list_exists(self):
        from greenlang.data_quality_profiler import models as mod
        assert hasattr(mod, "__all__")

    def test_all_contains_quality_dimension(self):
        from greenlang.data_quality_profiler import models as mod
        assert "QualityDimension" in mod.__all__

    def test_all_contains_column_profile(self):
        from greenlang.data_quality_profiler import models as mod
        assert "ColumnProfile" in mod.__all__

    def test_all_contains_profile_dataset_request(self):
        from greenlang.data_quality_profiler import models as mod
        assert "ProfileDatasetRequest" in mod.__all__

    def test_all_contains_quality_level(self):
        from greenlang.data_quality_profiler import models as mod
        assert "QualityLevel" in mod.__all__

    def test_all_minimum_count(self):
        from greenlang.data_quality_profiler import models as mod
        # 6 L1 re-exports + 10 constants + 13 enums + 15 models + 7 requests = 51+
        assert len(mod.__all__) >= 45
