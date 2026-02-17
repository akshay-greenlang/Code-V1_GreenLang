# -*- coding: utf-8 -*-
"""
Unit tests for Time Series Gap Filler models - AGENT-DATA-014

Tests all 13 enumerations, FillMethod enum, 22 SDK Pydantic models,
the _utcnow helper, field validators, model serialization, extra-field
rejection, constants, request models, and Layer 1 re-exports.
Target: 55+ tests.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-014 Time Series Gap Filler (GL-DATA-X-017)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from greenlang.time_series_gap_filler.models import (
    # Helper
    _utcnow,
    # Constants
    DEFAULT_STRATEGY_WEIGHTS,
    GAP_SEVERITY_THRESHOLDS,
    FILL_CONFIDENCE_THRESHOLDS,
    PIPELINE_STAGE_ORDER,
    FILL_STRATEGIES,
    GAP_TYPES,
    REPORT_FORMAT_OPTIONS,
    FREQUENCY_STRINGS,
    CALENDAR_TYPES,
    RESOLUTION_LEVELS,
    # Enums (13)
    FrequencyLevel,
    GapType,
    GapSeverity,
    FillStrategy,
    FillStatus,
    ValidationResult,
    CalendarType,
    SeasonalPattern,
    TrendType,
    PipelineStage,
    ReportFormat,
    ConfidenceLevel,
    DataResolution,
    # FillMethod enum (engine-internal)
    FillMethod,
    # SDK data models (22)
    GapRecord,
    GapDetectionResult,
    FrequencyResult,
    FillValue,
    FillResult,
    ValidationReport,
    CalendarDefinition,
    ReferenceSeries,
    GapFillStrategy,
    ImpactAssessment,
    GapFillReport,
    PipelineConfig,
    PipelineResult,
    GapFillerJobConfig,
    GapFillerStatistics,
    SeriesMetadata,
    BatchDetectionResult,
    CrossSeriesResult,
    SeasonalDecomposition,
    GapCharacterization,
    FillPoint,
    TrendAnalysis,
    # Request models (8)
    CreateJobRequest,
    DetectGapsRequest,
    BatchDetectRequest,
    AnalyzeFrequencyRequest,
    FillGapsRequest,
    ValidateFillsRequest,
    CreateCalendarRequest,
    RunPipelineRequest,
    # Layer 1 re-exports
    TimeSeriesImputerEngine,
    ImputedValue_L1,
    ConfidenceLevel_L1,
)


# ======================================================================
# 1. Helper: _utcnow
# ======================================================================


class TestUtcnow:
    """Test _utcnow helper function."""

    def test_returns_datetime(self):
        result = _utcnow()
        assert isinstance(result, datetime)

    def test_is_utc(self):
        result = _utcnow()
        assert result.tzinfo == timezone.utc

    def test_microseconds_zeroed(self):
        result = _utcnow()
        assert result.microsecond == 0


# ======================================================================
# 2. Constants
# ======================================================================


class TestConstants:
    """Test module-level constants for type, length, and key content."""

    def test_default_strategy_weights_is_dict(self):
        assert isinstance(DEFAULT_STRATEGY_WEIGHTS, dict)

    def test_default_strategy_weights_has_10_entries(self):
        assert len(DEFAULT_STRATEGY_WEIGHTS) == 10

    def test_default_strategy_weights_keys(self):
        expected = {
            "linear", "cubic_spline", "polynomial", "akima", "nearest",
            "seasonal", "trend", "cross_series", "calendar", "auto",
        }
        assert set(DEFAULT_STRATEGY_WEIGHTS.keys()) == expected

    def test_gap_severity_thresholds_is_dict(self):
        assert isinstance(GAP_SEVERITY_THRESHOLDS, dict)

    def test_gap_severity_thresholds_keys(self):
        assert set(GAP_SEVERITY_THRESHOLDS.keys()) == {
            "critical", "high", "medium", "low",
        }

    def test_gap_severity_thresholds_values(self):
        assert GAP_SEVERITY_THRESHOLDS["critical"] == 50
        assert GAP_SEVERITY_THRESHOLDS["low"] == 1

    def test_fill_confidence_thresholds_is_dict(self):
        assert isinstance(FILL_CONFIDENCE_THRESHOLDS, dict)

    def test_fill_confidence_thresholds_keys(self):
        assert set(FILL_CONFIDENCE_THRESHOLDS.keys()) == {
            "very_high", "high", "medium", "low", "very_low",
        }

    def test_fill_confidence_thresholds_values(self):
        assert FILL_CONFIDENCE_THRESHOLDS["very_high"] == pytest.approx(0.95)
        assert FILL_CONFIDENCE_THRESHOLDS["very_low"] == pytest.approx(0.0)

    def test_pipeline_stage_order_is_tuple(self):
        assert isinstance(PIPELINE_STAGE_ORDER, tuple)

    def test_pipeline_stage_order_has_6_items(self):
        assert len(PIPELINE_STAGE_ORDER) == 6

    def test_pipeline_stage_order_content(self):
        assert PIPELINE_STAGE_ORDER == (
            "detect", "characterize", "select_strategy",
            "fill", "validate", "document",
        )

    def test_fill_strategies_is_tuple_with_10_items(self):
        assert isinstance(FILL_STRATEGIES, tuple)
        assert len(FILL_STRATEGIES) == 10
        assert "linear" in FILL_STRATEGIES
        assert "auto" in FILL_STRATEGIES

    def test_gap_types_is_tuple_with_5_items(self):
        assert isinstance(GAP_TYPES, tuple)
        assert len(GAP_TYPES) == 5
        assert "short_gap" in GAP_TYPES
        assert "systematic_gap" in GAP_TYPES

    def test_report_format_options_is_tuple(self):
        assert isinstance(REPORT_FORMAT_OPTIONS, tuple)
        assert REPORT_FORMAT_OPTIONS == ("json", "text", "markdown", "html")

    def test_frequency_strings_is_tuple_with_9_items(self):
        assert isinstance(FREQUENCY_STRINGS, tuple)
        assert len(FREQUENCY_STRINGS) == 9
        assert "sub_hourly" in FREQUENCY_STRINGS
        assert "annual" in FREQUENCY_STRINGS

    def test_calendar_types_is_tuple_with_3_items(self):
        assert isinstance(CALENDAR_TYPES, tuple)
        assert len(CALENDAR_TYPES) == 3
        assert "business_days" in CALENDAR_TYPES

    def test_resolution_levels_is_tuple_with_7_items(self):
        assert isinstance(RESOLUTION_LEVELS, tuple)
        assert len(RESOLUTION_LEVELS) == 7
        assert "raw" in RESOLUTION_LEVELS
        assert "annual" in RESOLUTION_LEVELS


# ======================================================================
# 3. Enum: FrequencyLevel (9 members)
# ======================================================================


class TestFrequencyLevelEnum:
    """Test FrequencyLevel enum."""

    def test_member_count(self):
        assert len(FrequencyLevel) == 9

    def test_sample_values(self):
        assert FrequencyLevel.SUB_HOURLY.value == "sub_hourly"
        assert FrequencyLevel.HOURLY.value == "hourly"
        assert FrequencyLevel.DAILY.value == "daily"
        assert FrequencyLevel.BIWEEKLY.value == "biweekly"
        assert FrequencyLevel.ANNUAL.value == "annual"

    def test_is_str_enum(self):
        assert isinstance(FrequencyLevel.HOURLY, str)


# ======================================================================
# 4. Enum: GapType (5 members)
# ======================================================================


class TestGapTypeEnum:
    """Test GapType enum."""

    def test_member_count(self):
        assert len(GapType) == 5

    def test_sample_values(self):
        assert GapType.SHORT_GAP.value == "short_gap"
        assert GapType.LONG_GAP.value == "long_gap"
        assert GapType.PERIODIC_GAP.value == "periodic_gap"
        assert GapType.RANDOM_GAP.value == "random_gap"
        assert GapType.SYSTEMATIC_GAP.value == "systematic_gap"


# ======================================================================
# 5. Enum: GapSeverity (4 members)
# ======================================================================


class TestGapSeverityEnum:
    """Test GapSeverity enum."""

    def test_member_count(self):
        assert len(GapSeverity) == 4

    def test_sample_values(self):
        assert GapSeverity.LOW.value == "low"
        assert GapSeverity.MEDIUM.value == "medium"
        assert GapSeverity.HIGH.value == "high"
        assert GapSeverity.CRITICAL.value == "critical"


# ======================================================================
# 6. Enum: FillStrategy (10 members)
# ======================================================================


class TestFillStrategyEnum:
    """Test FillStrategy enum."""

    def test_member_count(self):
        assert len(FillStrategy) == 10

    def test_sample_values(self):
        assert FillStrategy.LINEAR.value == "linear"
        assert FillStrategy.CUBIC_SPLINE.value == "cubic_spline"
        assert FillStrategy.AUTO.value == "auto"
        assert FillStrategy.CROSS_SERIES.value == "cross_series"
        assert FillStrategy.CALENDAR.value == "calendar"


# ======================================================================
# 7. Enum: FillStatus (5 members)
# ======================================================================


class TestFillStatusEnum:
    """Test FillStatus enum."""

    def test_member_count(self):
        assert len(FillStatus) == 5

    def test_sample_values(self):
        assert FillStatus.PENDING.value == "pending"
        assert FillStatus.IN_PROGRESS.value == "in_progress"
        assert FillStatus.COMPLETED.value == "completed"
        assert FillStatus.FAILED.value == "failed"
        assert FillStatus.CANCELLED.value == "cancelled"


# ======================================================================
# 8. Enum: ValidationResult (3 members)
# ======================================================================


class TestValidationResultEnum:
    """Test ValidationResult enum (this is an Enum, not a Pydantic model)."""

    def test_member_count(self):
        assert len(ValidationResult) == 3

    def test_sample_values(self):
        assert ValidationResult.PASSED.value == "passed"
        assert ValidationResult.FAILED.value == "failed"
        assert ValidationResult.WARNING.value == "warning"

    def test_is_str_enum(self):
        assert isinstance(ValidationResult.PASSED, str)


# ======================================================================
# 9. Enum: CalendarType (3 members)
# ======================================================================


class TestCalendarTypeEnum:
    """Test CalendarType enum."""

    def test_member_count(self):
        assert len(CalendarType) == 3

    def test_sample_values(self):
        assert CalendarType.BUSINESS_DAYS.value == "business_days"
        assert CalendarType.FISCAL_YEAR.value == "fiscal_year"
        assert CalendarType.CUSTOM.value == "custom"


# ======================================================================
# 10. Enum: SeasonalPattern (5 members)
# ======================================================================


class TestSeasonalPatternEnum:
    """Test SeasonalPattern enum."""

    def test_member_count(self):
        assert len(SeasonalPattern) == 5

    def test_sample_values(self):
        assert SeasonalPattern.DAILY.value == "daily"
        assert SeasonalPattern.WEEKLY.value == "weekly"
        assert SeasonalPattern.ANNUAL.value == "annual"


# ======================================================================
# 11. Enum: TrendType (7 members)
# ======================================================================


class TestTrendTypeEnum:
    """Test TrendType enum."""

    def test_member_count(self):
        assert len(TrendType) == 7

    def test_sample_values(self):
        assert TrendType.LINEAR.value == "linear"
        assert TrendType.MODERATE_LINEAR.value == "moderate_linear"
        assert TrendType.EXPONENTIAL.value == "exponential"
        assert TrendType.STATIONARY.value == "stationary"
        assert TrendType.NONE.value == "none"
        assert TrendType.UNKNOWN.value == "unknown"


# ======================================================================
# 12. Enum: PipelineStage (6 members)
# ======================================================================


class TestPipelineStageEnum:
    """Test PipelineStage enum."""

    def test_member_count(self):
        assert len(PipelineStage) == 6

    def test_sample_values(self):
        assert PipelineStage.DETECT.value == "detect"
        assert PipelineStage.CHARACTERIZE.value == "characterize"
        assert PipelineStage.SELECT_STRATEGY.value == "select_strategy"
        assert PipelineStage.FILL.value == "fill"
        assert PipelineStage.VALIDATE.value == "validate"
        assert PipelineStage.DOCUMENT.value == "document"


# ======================================================================
# 13. Enum: ReportFormat (4 members)
# ======================================================================


class TestReportFormatEnum:
    """Test ReportFormat enum."""

    def test_member_count(self):
        assert len(ReportFormat) == 4

    def test_sample_values(self):
        assert ReportFormat.JSON.value == "json"
        assert ReportFormat.TEXT.value == "text"
        assert ReportFormat.MARKDOWN.value == "markdown"
        assert ReportFormat.HTML.value == "html"


# ======================================================================
# 14. Enum: ConfidenceLevel (5 members)
# ======================================================================


class TestConfidenceLevelEnum:
    """Test ConfidenceLevel enum."""

    def test_member_count(self):
        assert len(ConfidenceLevel) == 5

    def test_sample_values(self):
        assert ConfidenceLevel.VERY_LOW.value == "very_low"
        assert ConfidenceLevel.LOW.value == "low"
        assert ConfidenceLevel.MEDIUM.value == "medium"
        assert ConfidenceLevel.HIGH.value == "high"
        assert ConfidenceLevel.VERY_HIGH.value == "very_high"


# ======================================================================
# 15. Enum: DataResolution (7 members)
# ======================================================================


class TestDataResolutionEnum:
    """Test DataResolution enum."""

    def test_member_count(self):
        assert len(DataResolution) == 7

    def test_sample_values(self):
        assert DataResolution.RAW.value == "raw"
        assert DataResolution.HOURLY.value == "hourly"
        assert DataResolution.ANNUAL.value == "annual"


# ======================================================================
# 16. Enum: FillMethod (15 members)
# ======================================================================


class TestFillMethodEnum:
    """Test FillMethod enum (engine-internal)."""

    def test_member_count(self):
        assert len(FillMethod) == 15

    def test_sample_values(self):
        assert FillMethod.LINEAR.value == "linear"
        assert FillMethod.LINEAR_TREND.value == "linear_trend"
        assert FillMethod.PCHIP.value == "pchip"
        assert FillMethod.HOLT_WINTERS.value == "holt_winters"
        assert FillMethod.CALENDAR.value == "calendar"


# ======================================================================
# 17. Model: GapRecord
# ======================================================================


class TestGapRecordModel:
    """Test GapRecord model creation, defaults, and validation."""

    def test_required_fields(self):
        """start_index and end_index are required."""
        gap = GapRecord(start_index=3, end_index=5)
        assert gap.start_index == 3
        assert gap.end_index == 5

    def test_defaults(self):
        gap = GapRecord(start_index=0, end_index=0)
        assert gap.gap_length == 1
        assert gap.gap_type == GapType.RANDOM_GAP
        assert gap.severity == GapSeverity.LOW
        assert gap.provenance_hash == ""

    def test_gap_id_is_uuid(self):
        gap = GapRecord(start_index=0, end_index=0)
        parsed = uuid.UUID(gap.gap_id)
        assert str(parsed) == gap.gap_id

    def test_end_index_less_than_start_index_rejected(self):
        with pytest.raises(ValidationError):
            GapRecord(start_index=5, end_index=3)

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            GapRecord(start_index=0, end_index=0, unknown_field="x")

    def test_model_dump(self):
        gap = GapRecord(start_index=2, end_index=7, gap_length=6)
        d = gap.model_dump()
        assert d["start_index"] == 2
        assert d["end_index"] == 7
        assert d["gap_length"] == 6

    def test_negative_start_index_rejected(self):
        with pytest.raises(ValidationError):
            GapRecord(start_index=-1, end_index=0)


# ======================================================================
# 18. Model: GapDetectionResult
# ======================================================================


class TestGapDetectionResultModel:
    """Test GapDetectionResult model."""

    def test_series_id_required(self):
        gdr = GapDetectionResult(series_id="ts-001")
        assert gdr.series_id == "ts-001"
        assert gdr.gaps == []
        assert gdr.gap_count == 0

    def test_result_id_is_uuid(self):
        gdr = GapDetectionResult(series_id="ts-001")
        parsed = uuid.UUID(gdr.result_id)
        assert str(parsed) == gdr.result_id

    def test_empty_series_id_rejected(self):
        with pytest.raises(ValidationError):
            GapDetectionResult(series_id="")

    def test_whitespace_series_id_rejected(self):
        with pytest.raises(ValidationError):
            GapDetectionResult(series_id="   ")

    def test_gap_pct_above_one_rejected(self):
        with pytest.raises(ValidationError):
            GapDetectionResult(series_id="ts", gap_pct=1.5)

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            GapDetectionResult(series_id="ts", extra="x")


# ======================================================================
# 19. Model: FrequencyResult
# ======================================================================


class TestFrequencyResultModel:
    """Test FrequencyResult model."""

    def test_required_fields(self):
        fr = FrequencyResult(
            series_id="ts-001",
            detected_frequency=FrequencyLevel.DAILY,
        )
        assert fr.detected_frequency == FrequencyLevel.DAILY
        assert fr.confidence == pytest.approx(0.0)

    def test_confidence_above_one_rejected(self):
        with pytest.raises(ValidationError):
            FrequencyResult(
                series_id="ts",
                detected_frequency=FrequencyLevel.HOURLY,
                confidence=1.5,
            )

    def test_empty_series_id_rejected(self):
        with pytest.raises(ValidationError):
            FrequencyResult(
                series_id="",
                detected_frequency=FrequencyLevel.DAILY,
            )

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            FrequencyResult(
                series_id="ts",
                detected_frequency=FrequencyLevel.DAILY,
                bogus=True,
            )


# ======================================================================
# 20. Model: FillValue
# ======================================================================


class TestFillValueModel:
    """Test FillValue model."""

    def test_required_fields(self):
        fv = FillValue(index=5, filled_value=3.14, method=FillStrategy.LINEAR)
        assert fv.index == 5
        assert fv.filled_value == pytest.approx(3.14)
        assert fv.method == FillStrategy.LINEAR

    def test_defaults(self):
        fv = FillValue(index=0, filled_value=1.0, method=FillStrategy.NEAREST)
        assert fv.original_state == "missing"
        assert fv.confidence == pytest.approx(0.0)
        assert fv.confidence_level == ConfidenceLevel.MEDIUM
        assert fv.lower_bound is None
        assert fv.upper_bound is None

    def test_fill_id_is_uuid(self):
        fv = FillValue(index=0, filled_value=0.0, method=FillStrategy.LINEAR)
        parsed = uuid.UUID(fv.fill_id)
        assert str(parsed) == fv.fill_id

    def test_negative_confidence_rejected(self):
        with pytest.raises(ValidationError):
            FillValue(
                index=0, filled_value=1.0,
                method=FillStrategy.LINEAR, confidence=-0.1,
            )

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            FillValue(
                index=0, filled_value=1.0,
                method=FillStrategy.LINEAR, extra="x",
            )


# ======================================================================
# 21. Model: FillResult
# ======================================================================


class TestFillResultModel:
    """Test FillResult model."""

    def test_required_fields(self):
        fr = FillResult(series_id="ts-001", method=FillStrategy.CUBIC_SPLINE)
        assert fr.series_id == "ts-001"
        assert fr.method == FillStrategy.CUBIC_SPLINE
        assert fr.fills_count == 0
        assert fr.fill_values == []

    def test_result_id_is_uuid(self):
        fr = FillResult(series_id="ts-001", method=FillStrategy.LINEAR)
        parsed = uuid.UUID(fr.result_id)
        assert str(parsed) == fr.result_id

    def test_empty_series_id_rejected(self):
        with pytest.raises(ValidationError):
            FillResult(series_id="", method=FillStrategy.LINEAR)

    def test_model_dump(self):
        fr = FillResult(
            series_id="ts", method=FillStrategy.AUTO,
            fills_count=5, avg_confidence=0.8,
        )
        d = fr.model_dump()
        assert d["fills_count"] == 5
        assert d["avg_confidence"] == pytest.approx(0.8)

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            FillResult(series_id="ts", method=FillStrategy.LINEAR, extra="x")


# ======================================================================
# 22. Model: ValidationReport
# ======================================================================


class TestValidationReportModel:
    """Test ValidationReport model."""

    def test_defaults(self):
        vr = ValidationReport(series_id="ts-001")
        assert vr.result == ValidationResult.PASSED
        assert vr.distribution_preserved is True
        assert vr.plausibility_passed is True
        assert vr.p_value == pytest.approx(1.0)
        assert vr.warnings == []
        assert vr.details == {}

    def test_empty_series_id_rejected(self):
        with pytest.raises(ValidationError):
            ValidationReport(series_id="")

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            ValidationReport(series_id="ts", extra="x")


# ======================================================================
# 23. Model: CalendarDefinition
# ======================================================================


class TestCalendarDefinitionModel:
    """Test CalendarDefinition model."""

    def test_defaults(self):
        cd = CalendarDefinition(name="Default")
        assert cd.calendar_type == CalendarType.BUSINESS_DAYS
        assert cd.business_days == [1, 2, 3, 4, 5]
        assert cd.fiscal_start_month == 1
        assert cd.timezone == "UTC"
        assert cd.active is True

    def test_empty_name_rejected(self):
        with pytest.raises(ValidationError):
            CalendarDefinition(name="")

    def test_whitespace_name_rejected(self):
        with pytest.raises(ValidationError):
            CalendarDefinition(name="   ")

    def test_business_day_0_rejected(self):
        with pytest.raises(ValidationError):
            CalendarDefinition(name="Test", business_days=[0, 1, 2])

    def test_business_day_8_rejected(self):
        with pytest.raises(ValidationError):
            CalendarDefinition(name="Test", business_days=[1, 8])

    def test_fiscal_start_month_0_rejected(self):
        with pytest.raises(ValidationError):
            CalendarDefinition(name="Test", fiscal_start_month=0)

    def test_fiscal_start_month_13_rejected(self):
        with pytest.raises(ValidationError):
            CalendarDefinition(name="Test", fiscal_start_month=13)

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            CalendarDefinition(name="Test", extra="x")

    def test_created_at_is_utc(self):
        cd = CalendarDefinition(name="Test")
        assert cd.created_at.tzinfo == timezone.utc


# ======================================================================
# 24. Model: ReferenceSeries
# ======================================================================


class TestReferenceSeriesModel:
    """Test ReferenceSeries model."""

    def test_required_fields(self):
        rs = ReferenceSeries(series_id="donor-001")
        assert rs.series_id == "donor-001"
        assert rs.correlation == pytest.approx(0.0)
        assert rs.active is True

    def test_empty_series_id_rejected(self):
        with pytest.raises(ValidationError):
            ReferenceSeries(series_id="")

    def test_correlation_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            ReferenceSeries(series_id="d", correlation=1.5)
        with pytest.raises(ValidationError):
            ReferenceSeries(series_id="d", correlation=-1.5)

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            ReferenceSeries(series_id="d", extra="x")


# ======================================================================
# 25. Model: GapFillStrategy
# ======================================================================


class TestGapFillStrategyModel:
    """Test GapFillStrategy model."""

    def test_required_fields(self):
        gfs = GapFillStrategy(
            gap_id="gap-001", strategy=FillStrategy.SEASONAL,
        )
        assert gfs.gap_id == "gap-001"
        assert gfs.strategy == FillStrategy.SEASONAL
        assert gfs.auto_selected is False

    def test_empty_gap_id_rejected(self):
        with pytest.raises(ValidationError):
            GapFillStrategy(gap_id="", strategy=FillStrategy.LINEAR)

    def test_selection_score_above_one_rejected(self):
        with pytest.raises(ValidationError):
            GapFillStrategy(
                gap_id="g", strategy=FillStrategy.LINEAR,
                selection_score=1.5,
            )

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            GapFillStrategy(
                gap_id="g", strategy=FillStrategy.LINEAR, extra="x",
            )


# ======================================================================
# 26. Model: ImpactAssessment
# ======================================================================


class TestImpactAssessmentModel:
    """Test ImpactAssessment model."""

    def test_defaults(self):
        ia = ImpactAssessment(series_id="ts-001")
        assert ia.total_impact_pct == pytest.approx(0.0)
        assert ia.data_completeness == pytest.approx(1.0)
        assert ia.pre_fill_completeness == pytest.approx(1.0)
        assert ia.risk_level == GapSeverity.LOW
        assert ia.recommendations == []

    def test_total_impact_above_100_rejected(self):
        with pytest.raises(ValidationError):
            ImpactAssessment(series_id="ts", total_impact_pct=101.0)

    def test_empty_series_id_rejected(self):
        with pytest.raises(ValidationError):
            ImpactAssessment(series_id="")

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            ImpactAssessment(series_id="ts", extra="x")


# ======================================================================
# 27. Model: GapFillReport
# ======================================================================


class TestGapFillReportModel:
    """Test GapFillReport model."""

    def test_defaults(self):
        r = GapFillReport(series_id="ts-001")
        assert r.format == ReportFormat.JSON
        assert r.validation_passed is False
        assert r.regulatory_compliance is False
        assert r.impact_assessment is None
        assert r.content == ""

    def test_empty_series_id_rejected(self):
        with pytest.raises(ValidationError):
            GapFillReport(series_id="")

    def test_generated_at_is_utc(self):
        r = GapFillReport(series_id="ts")
        assert r.generated_at.tzinfo == timezone.utc

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            GapFillReport(series_id="ts", extra="x")


# ======================================================================
# 28. Model: PipelineConfig
# ======================================================================


class TestPipelineConfigModel:
    """Test PipelineConfig model."""

    def test_defaults(self):
        pc = PipelineConfig()
        assert pc.series_id == ""
        assert pc.strategy == FillStrategy.AUTO
        assert pc.enable_validation is True
        assert pc.enable_calendar is False
        assert pc.enable_cross_series is False
        assert pc.fallback_chain == [FillStrategy.LINEAR, FillStrategy.NEAREST]
        assert pc.min_confidence == pytest.approx(0.5)
        assert pc.max_gap_length == 100
        assert pc.report_format == ReportFormat.JSON
        assert pc.auto_detect_frequency is True
        assert pc.polynomial_degree == 3
        assert pc.validation_alpha == pytest.approx(0.05)

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            PipelineConfig(extra="x")


# ======================================================================
# 29. Model: PipelineResult
# ======================================================================


class TestPipelineResultModel:
    """Test PipelineResult model."""

    def test_defaults(self):
        pr = PipelineResult()
        assert pr.job_id == ""
        assert pr.series_id == ""
        assert pr.stage == PipelineStage.DOCUMENT
        assert pr.status == FillStatus.COMPLETED
        assert pr.gaps_detected == 0
        assert pr.fill_results == []
        assert pr.provenance_hash == ""

    def test_pipeline_id_is_uuid(self):
        pr = PipelineResult()
        parsed = uuid.UUID(pr.pipeline_id)
        assert str(parsed) == pr.pipeline_id

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            PipelineResult(extra="x")


# ======================================================================
# 30. Model: GapFillerJobConfig
# ======================================================================


class TestGapFillerJobConfigModel:
    """Test GapFillerJobConfig model including computed properties."""

    def test_defaults(self):
        jc = GapFillerJobConfig()
        assert jc.status == FillStatus.PENDING
        assert jc.stage == PipelineStage.DETECT
        assert jc.strategy == FillStrategy.AUTO
        assert jc.auto_detect_frequency is True
        assert jc.pipeline_config is None
        assert jc.error_message is None

    def test_job_id_is_uuid(self):
        jc = GapFillerJobConfig()
        parsed = uuid.UUID(jc.job_id)
        assert str(parsed) == jc.job_id

    def test_is_active_pending(self):
        jc = GapFillerJobConfig(status=FillStatus.PENDING)
        assert jc.is_active is True

    def test_is_active_in_progress(self):
        jc = GapFillerJobConfig(status=FillStatus.IN_PROGRESS)
        assert jc.is_active is True

    def test_is_active_completed(self):
        jc = GapFillerJobConfig(status=FillStatus.COMPLETED)
        assert jc.is_active is False

    def test_is_active_failed(self):
        jc = GapFillerJobConfig(status=FillStatus.FAILED)
        assert jc.is_active is False

    def test_is_active_cancelled(self):
        jc = GapFillerJobConfig(status=FillStatus.CANCELLED)
        assert jc.is_active is False

    def test_progress_pct_completed(self):
        jc = GapFillerJobConfig(status=FillStatus.COMPLETED)
        assert jc.progress_pct == pytest.approx(100.0)

    def test_progress_pct_failed(self):
        jc = GapFillerJobConfig(status=FillStatus.FAILED)
        assert jc.progress_pct == pytest.approx(0.0)

    def test_progress_pct_cancelled(self):
        jc = GapFillerJobConfig(status=FillStatus.CANCELLED)
        assert jc.progress_pct == pytest.approx(0.0)

    def test_progress_pct_detect_stage(self):
        jc = GapFillerJobConfig(
            status=FillStatus.IN_PROGRESS, stage=PipelineStage.DETECT,
        )
        assert jc.progress_pct == pytest.approx(16.7)

    def test_progress_pct_fill_stage(self):
        jc = GapFillerJobConfig(
            status=FillStatus.IN_PROGRESS, stage=PipelineStage.FILL,
        )
        assert jc.progress_pct == pytest.approx(66.7)

    def test_progress_pct_validate_stage(self):
        jc = GapFillerJobConfig(
            status=FillStatus.IN_PROGRESS, stage=PipelineStage.VALIDATE,
        )
        assert jc.progress_pct == pytest.approx(83.3)

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            GapFillerJobConfig(extra="x")


# ======================================================================
# 31. Model: GapFillerStatistics
# ======================================================================


class TestGapFillerStatisticsModel:
    """Test GapFillerStatistics model."""

    def test_defaults(self):
        gs = GapFillerStatistics()
        assert gs.total_jobs == 0
        assert gs.total_gaps_detected == 0
        assert gs.total_gaps_filled == 0
        assert gs.total_gaps_failed == 0
        assert gs.avg_confidence == pytest.approx(0.0)
        assert gs.by_status == {}
        assert gs.by_strategy == {}

    def test_timestamp_is_utc(self):
        gs = GapFillerStatistics()
        assert gs.timestamp.tzinfo == timezone.utc

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            GapFillerStatistics(extra="x")


# ======================================================================
# 32. Model: SeriesMetadata
# ======================================================================


class TestSeriesMetadataModel:
    """Test SeriesMetadata model."""

    def test_required_fields(self):
        sm = SeriesMetadata(series_id="ts-001")
        assert sm.series_id == "ts-001"
        assert sm.resolution == DataResolution.RAW
        assert sm.tags == {}

    def test_empty_series_id_rejected(self):
        with pytest.raises(ValidationError):
            SeriesMetadata(series_id="")

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            SeriesMetadata(series_id="ts", extra="x")


# ======================================================================
# 33. Model: BatchDetectionResult
# ======================================================================


class TestBatchDetectionResultModel:
    """Test BatchDetectionResult model."""

    def test_defaults(self):
        bdr = BatchDetectionResult()
        assert bdr.series_count == 0
        assert bdr.total_gaps == 0
        assert bdr.total_missing == 0
        assert bdr.per_series_results == []

    def test_batch_id_is_uuid(self):
        bdr = BatchDetectionResult()
        parsed = uuid.UUID(bdr.batch_id)
        assert str(parsed) == bdr.batch_id

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            BatchDetectionResult(extra="x")


# ======================================================================
# 34. Model: CrossSeriesResult
# ======================================================================


class TestCrossSeriesResultModel:
    """Test CrossSeriesResult model."""

    def test_required_fields(self):
        csr = CrossSeriesResult(target_id="ts-001", donor_id="ts-002")
        assert csr.target_id == "ts-001"
        assert csr.donor_id == "ts-002"
        assert csr.scaling_factor == pytest.approx(1.0)

    def test_empty_target_id_rejected(self):
        with pytest.raises(ValidationError):
            CrossSeriesResult(target_id="", donor_id="ts-002")

    def test_empty_donor_id_rejected(self):
        with pytest.raises(ValidationError):
            CrossSeriesResult(target_id="ts-001", donor_id="")

    def test_correlation_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            CrossSeriesResult(
                target_id="t", donor_id="d", correlation=1.5,
            )

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            CrossSeriesResult(target_id="t", donor_id="d", extra="x")


# ======================================================================
# 35. Model: SeasonalDecomposition
# ======================================================================


class TestSeasonalDecompositionModel:
    """Test SeasonalDecomposition model."""

    def test_defaults(self):
        sd = SeasonalDecomposition()
        assert sd.series_id == ""
        assert sd.trend == []
        assert sd.seasonal == []
        assert sd.residual == []
        assert sd.period == 1
        assert sd.trend_type == TrendType.NONE
        assert sd.seasonal_strength == pytest.approx(0.0)

    def test_period_zero_rejected(self):
        with pytest.raises(ValidationError):
            SeasonalDecomposition(period=0)

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            SeasonalDecomposition(extra="x")


# ======================================================================
# 36. Model: GapCharacterization
# ======================================================================


class TestGapCharacterizationModel:
    """Test GapCharacterization model including neighbor_trend validator."""

    def test_required_fields(self):
        gc = GapCharacterization(gap_id="gap-001")
        assert gc.gap_id == "gap-001"
        assert gc.gap_type == GapType.RANDOM_GAP
        assert gc.severity == GapSeverity.LOW
        assert gc.recommended_strategy == FillStrategy.LINEAR

    def test_empty_gap_id_rejected(self):
        with pytest.raises(ValidationError):
            GapCharacterization(gap_id="")

    def test_neighbor_trend_valid_values(self):
        for val in (-1, 0, 1):
            gc = GapCharacterization(gap_id="g", neighbor_trend=val)
            assert gc.neighbor_trend == val

    def test_neighbor_trend_2_rejected(self):
        with pytest.raises(ValidationError):
            GapCharacterization(gap_id="g", neighbor_trend=2)

    def test_neighbor_trend_minus_2_rejected(self):
        with pytest.raises(ValidationError):
            GapCharacterization(gap_id="g", neighbor_trend=-2)

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            GapCharacterization(gap_id="g", extra="x")


# ======================================================================
# 37. Model: FillPoint (engine-internal)
# ======================================================================


class TestFillPointModel:
    """Test FillPoint model."""

    def test_required_fields(self):
        fp = FillPoint(index=3, filled_value=2.5, method=FillMethod.LINEAR)
        assert fp.index == 3
        assert fp.filled_value == pytest.approx(2.5)
        assert fp.method == FillMethod.LINEAR

    def test_defaults(self):
        fp = FillPoint(index=0, filled_value=0.0, method=FillMethod.NEAREST)
        assert fp.was_missing is True
        assert fp.confidence == pytest.approx(0.0)
        assert fp.provenance_hash == ""
        assert fp.original_value is None

    def test_negative_index_rejected(self):
        with pytest.raises(ValidationError):
            FillPoint(index=-1, filled_value=0.0, method=FillMethod.LINEAR)

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            FillPoint(
                index=0, filled_value=0.0, method=FillMethod.LINEAR, extra="x",
            )


# ======================================================================
# 38. Model: TrendAnalysis (engine-internal)
# ======================================================================


class TestTrendAnalysisModel:
    """Test TrendAnalysis model."""

    def test_defaults(self):
        ta = TrendAnalysis()
        assert ta.trend_type == TrendType.UNKNOWN
        assert ta.slope == pytest.approx(0.0)
        assert ta.intercept == pytest.approx(0.0)
        assert ta.r_squared == pytest.approx(0.0)
        assert ta.confidence == pytest.approx(0.0)
        assert ta.series_length == 0

    def test_r_squared_above_one_rejected(self):
        with pytest.raises(ValidationError):
            TrendAnalysis(r_squared=1.5)

    def test_negative_series_length_rejected(self):
        with pytest.raises(ValidationError):
            TrendAnalysis(series_length=-1)

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            TrendAnalysis(extra="x")


# ======================================================================
# 39. Request: CreateJobRequest
# ======================================================================


class TestCreateJobRequestModel:
    """Test CreateJobRequest model."""

    def test_defaults(self):
        req = CreateJobRequest(values=[1.0, None, 3.0])
        assert req.strategy == FillStrategy.AUTO
        assert req.auto_detect_frequency is True
        assert req.pipeline_config is None
        assert req.series_name == ""

    def test_values_min_length_2(self):
        with pytest.raises(ValidationError):
            CreateJobRequest(values=[1.0])

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            CreateJobRequest(values=[1.0, 2.0], extra="x")


# ======================================================================
# 40. Request: DetectGapsRequest
# ======================================================================


class TestDetectGapsRequestModel:
    """Test DetectGapsRequest model."""

    def test_defaults(self):
        req = DetectGapsRequest(values=[1.0, None, 3.0])
        assert req.series_id == ""
        assert req.min_gap_length == 1

    def test_values_min_length_1(self):
        req = DetectGapsRequest(values=[None])
        assert len(req.values) == 1

    def test_empty_values_rejected(self):
        with pytest.raises(ValidationError):
            DetectGapsRequest(values=[])

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            DetectGapsRequest(values=[1.0], extra="x")


# ======================================================================
# 41. Request: BatchDetectRequest
# ======================================================================


class TestBatchDetectRequestModel:
    """Test BatchDetectRequest model."""

    def test_required_fields(self):
        req = BatchDetectRequest(series_list=[[1.0, None]])
        assert len(req.series_list) == 1
        assert req.series_ids == []
        assert req.shared_frequency is None

    def test_empty_series_list_rejected(self):
        with pytest.raises(ValidationError):
            BatchDetectRequest(series_list=[])

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            BatchDetectRequest(series_list=[[1.0]], extra="x")


# ======================================================================
# 42. Request: AnalyzeFrequencyRequest
# ======================================================================


class TestAnalyzeFrequencyRequestModel:
    """Test AnalyzeFrequencyRequest model."""

    def test_defaults(self):
        req = AnalyzeFrequencyRequest(
            timestamps=["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"],
        )
        assert req.series_id == ""
        assert req.max_sample_size == 1000

    def test_timestamps_min_length_2(self):
        with pytest.raises(ValidationError):
            AnalyzeFrequencyRequest(timestamps=["2026-01-01T00:00:00Z"])

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            AnalyzeFrequencyRequest(
                timestamps=["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"],
                extra="x",
            )


# ======================================================================
# 43. Request: FillGapsRequest
# ======================================================================


class TestFillGapsRequestModel:
    """Test FillGapsRequest model."""

    def test_defaults(self):
        req = FillGapsRequest(values=[1.0, None, 3.0])
        assert req.strategy == FillStrategy.AUTO
        assert req.min_confidence == pytest.approx(0.5)
        assert req.max_gap_length == 100
        assert req.polynomial_degree == 3

    def test_values_min_length_2(self):
        with pytest.raises(ValidationError):
            FillGapsRequest(values=[1.0])

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            FillGapsRequest(values=[1.0, 2.0], extra="x")


# ======================================================================
# 44. Request: ValidateFillsRequest
# ======================================================================


class TestValidateFillsRequestModel:
    """Test ValidateFillsRequest model."""

    def test_defaults(self):
        req = ValidateFillsRequest(
            original_values=[1.0, None, 3.0],
            filled_values=[1.0, 2.0, 3.0],
        )
        assert req.alpha == pytest.approx(0.05)
        assert req.series_id == ""
        assert req.fill_indices == []
        assert req.ground_truth == []

    def test_original_values_min_length_1(self):
        with pytest.raises(ValidationError):
            ValidateFillsRequest(original_values=[], filled_values=[1.0])

    def test_filled_values_min_length_1(self):
        with pytest.raises(ValidationError):
            ValidateFillsRequest(original_values=[1.0], filled_values=[])

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            ValidateFillsRequest(
                original_values=[1.0], filled_values=[1.0], extra="x",
            )


# ======================================================================
# 45. Request: CreateCalendarRequest
# ======================================================================


class TestCreateCalendarRequestModel:
    """Test CreateCalendarRequest model."""

    def test_defaults(self):
        req = CreateCalendarRequest(name="TestCal")
        assert req.calendar_type == CalendarType.BUSINESS_DAYS
        assert req.business_days == [1, 2, 3, 4, 5]
        assert req.fiscal_start_month == 1
        assert req.timezone == "UTC"
        assert req.description == ""

    def test_empty_name_rejected(self):
        with pytest.raises(ValidationError):
            CreateCalendarRequest(name="")

    def test_business_day_0_rejected(self):
        with pytest.raises(ValidationError):
            CreateCalendarRequest(name="Cal", business_days=[0, 1])

    def test_business_day_8_rejected(self):
        with pytest.raises(ValidationError):
            CreateCalendarRequest(name="Cal", business_days=[1, 8])

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            CreateCalendarRequest(name="Cal", extra="x")


# ======================================================================
# 46. Request: RunPipelineRequest
# ======================================================================


class TestRunPipelineRequestModel:
    """Test RunPipelineRequest model."""

    def test_defaults(self):
        req = RunPipelineRequest(values=[1.0, None, 3.0])
        assert req.series_id == ""
        assert req.pipeline_config is None
        assert req.reference_series == []
        assert req.calendar_id is None
        assert req.options == {}

    def test_values_min_length_2(self):
        with pytest.raises(ValidationError):
            RunPipelineRequest(values=[1.0])

    def test_extra_field_rejected(self):
        with pytest.raises(ValidationError):
            RunPipelineRequest(values=[1.0, 2.0], extra="x")


# ======================================================================
# 47. Layer 1 re-exports
# ======================================================================


class TestLayer1ReExports:
    """Verify that Layer 1 re-exports exist or are gracefully None."""

    def test_all_exports_importable(self):
        """All items in __all__ should be importable attributes."""
        from greenlang.time_series_gap_filler import models as mod
        for name in mod.__all__:
            assert hasattr(mod, name), f"Missing export: {name}"

    def test_time_series_imputer_engine_available(self):
        """TimeSeriesImputerEngine is either a class or None."""
        assert TimeSeriesImputerEngine is None or callable(TimeSeriesImputerEngine)

    def test_imputed_value_l1_available(self):
        """ImputedValue_L1 is either a class or None."""
        assert ImputedValue_L1 is None or callable(ImputedValue_L1)

    def test_confidence_level_l1_available(self):
        """ConfidenceLevel_L1 is either a class or None."""
        assert ConfidenceLevel_L1 is None or callable(ConfidenceLevel_L1)
