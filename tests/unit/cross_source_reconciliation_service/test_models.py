# -*- coding: utf-8 -*-
"""
Unit tests for Cross-Source Reconciliation data models - AGENT-DATA-015

Tests all 13 enumerations, 22 SDK models, 8 request models, 10 constants,
Layer 1 re-exports, and the _utcnow() helper defined in
greenlang.cross_source_reconciliation.models.

Total: 110+ tests organized into classes by component category.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-015 Cross-Source Reconciliation (GL-DATA-X-018)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Dict, List

import pytest
from pydantic import ValidationError

from greenlang.cross_source_reconciliation.models import (
    # Helper
    _utcnow,
    # Constants
    DEFAULT_MATCH_THRESHOLD,
    DEFAULT_TOLERANCE_PCT,
    DEFAULT_TOLERANCE_ABS,
    MAX_SOURCES,
    MAX_MATCH_CANDIDATES,
    CRITICAL_THRESHOLD_PCT,
    HIGH_THRESHOLD_PCT,
    MEDIUM_THRESHOLD_PCT,
    SUPPORTED_UNITS,
    SUPPORTED_CURRENCIES,
    # Enumerations
    SourceType,
    SourceStatus,
    MatchStrategy,
    MatchStatus,
    ComparisonResult,
    DiscrepancyType,
    DiscrepancySeverity,
    ResolutionStrategy,
    ResolutionStatus,
    FieldType,
    ReconciliationStatus,
    TemporalGranularity,
    CredibilityFactor,
    # SDK models
    SourceDefinition,
    SchemaMapping,
    MatchKey,
    MatchResult,
    FieldComparison,
    Discrepancy,
    ResolutionDecision,
    GoldenRecord,
    SourceCredibility,
    ToleranceRule,
    ReconciliationReport,
    ReconciliationJobConfig,
    BatchMatchResult,
    DiscrepancySummary,
    ResolutionSummary,
    TemporalAlignment,
    FieldLineage,
    ReconciliationStats,
    SourceHealthMetrics,
    ComparisonSummary,
    PipelineStageResult,
    ReconciliationEvent,
    # Request models
    CreateJobRequest,
    RegisterSourceRequest,
    UpdateSourceRequest,
    MatchRequest,
    CompareRequest,
    ResolveRequest,
    PipelineRequest,
    GoldenRecordRequest,
    # Layer 1 re-exports
    ConsistencyAnalyzer,
    SimilarityScorer,
    MatchClassifier,
    FactorReconciler,
    ConflictResolutionStrategy,
)


# ======================================================================
# 1. _utcnow() helper
# ======================================================================


class TestUtcnowHelper:
    """Test the _utcnow() helper function."""

    def test_utcnow_returns_datetime(self):
        result = _utcnow()
        assert isinstance(result, datetime)

    def test_utcnow_has_utc_timezone(self):
        result = _utcnow()
        assert result.tzinfo == timezone.utc

    def test_utcnow_microseconds_zeroed(self):
        result = _utcnow()
        assert result.microsecond == 0

    def test_utcnow_successive_calls_non_decreasing(self):
        a = _utcnow()
        b = _utcnow()
        assert b >= a


# ======================================================================
# 2. Constants (10)
# ======================================================================


class TestConstants:
    """Test module-level constant values and types."""

    def test_default_match_threshold_value(self):
        assert DEFAULT_MATCH_THRESHOLD == pytest.approx(0.85)

    def test_default_match_threshold_type(self):
        assert isinstance(DEFAULT_MATCH_THRESHOLD, float)

    def test_default_tolerance_pct_value(self):
        assert DEFAULT_TOLERANCE_PCT == pytest.approx(5.0)

    def test_default_tolerance_pct_type(self):
        assert isinstance(DEFAULT_TOLERANCE_PCT, float)

    def test_default_tolerance_abs_value(self):
        assert DEFAULT_TOLERANCE_ABS == pytest.approx(0.01)

    def test_default_tolerance_abs_type(self):
        assert isinstance(DEFAULT_TOLERANCE_ABS, float)

    def test_max_sources_value(self):
        assert MAX_SOURCES == 20

    def test_max_sources_type(self):
        assert isinstance(MAX_SOURCES, int)

    def test_max_match_candidates_value(self):
        assert MAX_MATCH_CANDIDATES == 100

    def test_max_match_candidates_type(self):
        assert isinstance(MAX_MATCH_CANDIDATES, int)

    def test_critical_threshold_pct_value(self):
        assert CRITICAL_THRESHOLD_PCT == pytest.approx(50.0)

    def test_high_threshold_pct_value(self):
        assert HIGH_THRESHOLD_PCT == pytest.approx(25.0)

    def test_medium_threshold_pct_value(self):
        assert MEDIUM_THRESHOLD_PCT == pytest.approx(10.0)

    def test_supported_units_is_dict(self):
        assert isinstance(SUPPORTED_UNITS, dict)

    def test_supported_units_has_expected_keys(self):
        expected_keys = [
            "kg_to_tonnes", "tonnes_to_kg", "g_to_kg", "kg_to_g",
            "lb_to_kg", "kg_to_lb", "MWh_to_kWh", "kWh_to_MWh",
            "GJ_to_MWh", "MWh_to_GJ", "m3_to_litre", "litre_to_m3",
            "gallon_to_litre", "litre_to_gallon", "mile_to_km",
            "km_to_mile", "tCO2e_to_kgCO2e", "kgCO2e_to_tCO2e",
        ]
        for key in expected_keys:
            assert key in SUPPORTED_UNITS

    def test_supported_units_count(self):
        assert len(SUPPORTED_UNITS) == 18

    def test_supported_units_values_are_floats(self):
        for v in SUPPORTED_UNITS.values():
            assert isinstance(v, float)

    def test_supported_currencies_is_list(self):
        assert isinstance(SUPPORTED_CURRENCIES, list)

    def test_supported_currencies_count(self):
        assert len(SUPPORTED_CURRENCIES) == 30

    def test_supported_currencies_contains_usd(self):
        assert "USD" in SUPPORTED_CURRENCIES

    def test_supported_currencies_contains_eur(self):
        assert "EUR" in SUPPORTED_CURRENCIES

    def test_threshold_ordering(self):
        """Thresholds must be ordered: medium < high < critical."""
        assert MEDIUM_THRESHOLD_PCT < HIGH_THRESHOLD_PCT
        assert HIGH_THRESHOLD_PCT < CRITICAL_THRESHOLD_PCT


# ======================================================================
# 3. Enumerations (13)
# ======================================================================


class TestSourceTypeEnum:
    """Test SourceType enumeration."""

    def test_member_count(self):
        assert len(SourceType) == 10

    def test_erp_value(self):
        assert SourceType.ERP.value == "erp"

    def test_utility_value(self):
        assert SourceType.UTILITY.value == "utility"

    def test_meter_value(self):
        assert SourceType.METER.value == "meter"

    def test_iot_value(self):
        assert SourceType.IOT.value == "iot"

    def test_membership(self):
        assert "erp" in [m.value for m in SourceType]
        assert "other" in [m.value for m in SourceType]

    def test_str_enum_subclass(self):
        assert isinstance(SourceType.ERP, str)


class TestSourceStatusEnum:
    """Test SourceStatus enumeration."""

    def test_member_count(self):
        assert len(SourceStatus) == 4

    def test_active_value(self):
        assert SourceStatus.ACTIVE.value == "active"

    def test_pending_validation_value(self):
        assert SourceStatus.PENDING_VALIDATION.value == "pending_validation"

    def test_error_value(self):
        assert SourceStatus.ERROR.value == "error"


class TestMatchStrategyEnum:
    """Test MatchStrategy enumeration."""

    def test_member_count(self):
        assert len(MatchStrategy) == 5

    def test_exact_value(self):
        assert MatchStrategy.EXACT.value == "exact"

    def test_fuzzy_value(self):
        assert MatchStrategy.FUZZY.value == "fuzzy"

    def test_composite_value(self):
        assert MatchStrategy.COMPOSITE.value == "composite"

    def test_temporal_value(self):
        assert MatchStrategy.TEMPORAL.value == "temporal"

    def test_blocking_value(self):
        assert MatchStrategy.BLOCKING.value == "blocking"


class TestMatchStatusEnum:
    """Test MatchStatus enumeration."""

    def test_member_count(self):
        assert len(MatchStatus) == 4

    def test_matched_value(self):
        assert MatchStatus.MATCHED.value == "matched"

    def test_unmatched_value(self):
        assert MatchStatus.UNMATCHED.value == "unmatched"

    def test_ambiguous_value(self):
        assert MatchStatus.AMBIGUOUS.value == "ambiguous"

    def test_pending_review_value(self):
        assert MatchStatus.PENDING_REVIEW.value == "pending_review"


class TestComparisonResultEnum:
    """Test ComparisonResult enumeration."""

    def test_member_count(self):
        assert len(ComparisonResult) == 6

    def test_match_value(self):
        assert ComparisonResult.MATCH.value == "match"

    def test_mismatch_value(self):
        assert ComparisonResult.MISMATCH.value == "mismatch"

    def test_within_tolerance_value(self):
        assert ComparisonResult.WITHIN_TOLERANCE.value == "within_tolerance"

    def test_missing_left_value(self):
        assert ComparisonResult.MISSING_LEFT.value == "missing_left"


class TestDiscrepancyTypeEnum:
    """Test DiscrepancyType enumeration."""

    def test_member_count(self):
        assert len(DiscrepancyType) == 7

    def test_value_mismatch_value(self):
        assert DiscrepancyType.VALUE_MISMATCH.value == "value_mismatch"

    def test_timing_difference_value(self):
        assert DiscrepancyType.TIMING_DIFFERENCE.value == "timing_difference"

    def test_unit_difference_value(self):
        assert DiscrepancyType.UNIT_DIFFERENCE.value == "unit_difference"

    def test_aggregation_mismatch_value(self):
        assert DiscrepancyType.AGGREGATION_MISMATCH.value == "aggregation_mismatch"


class TestDiscrepancySeverityEnum:
    """Test DiscrepancySeverity enumeration."""

    def test_member_count(self):
        assert len(DiscrepancySeverity) == 5

    def test_critical_value(self):
        assert DiscrepancySeverity.CRITICAL.value == "critical"

    def test_high_value(self):
        assert DiscrepancySeverity.HIGH.value == "high"

    def test_info_value(self):
        assert DiscrepancySeverity.INFO.value == "info"


class TestResolutionStrategyEnum:
    """Test ResolutionStrategy enumeration."""

    def test_member_count(self):
        assert len(ResolutionStrategy) == 7

    def test_priority_wins_value(self):
        assert ResolutionStrategy.PRIORITY_WINS.value == "priority_wins"

    def test_most_recent_value(self):
        assert ResolutionStrategy.MOST_RECENT.value == "most_recent"

    def test_consensus_value(self):
        assert ResolutionStrategy.CONSENSUS.value == "consensus"

    def test_custom_value(self):
        assert ResolutionStrategy.CUSTOM.value == "custom"


class TestResolutionStatusEnum:
    """Test ResolutionStatus enumeration."""

    def test_member_count(self):
        assert len(ResolutionStatus) == 5

    def test_resolved_value(self):
        assert ResolutionStatus.RESOLVED.value == "resolved"

    def test_escalated_value(self):
        assert ResolutionStatus.ESCALATED.value == "escalated"

    def test_deferred_value(self):
        assert ResolutionStatus.DEFERRED.value == "deferred"

    def test_rejected_value(self):
        assert ResolutionStatus.REJECTED.value == "rejected"


class TestFieldTypeEnum:
    """Test FieldType enumeration."""

    def test_member_count(self):
        assert len(FieldType) == 7

    def test_numeric_value(self):
        assert FieldType.NUMERIC.value == "numeric"

    def test_currency_value(self):
        assert FieldType.CURRENCY.value == "currency"

    def test_unit_value_value(self):
        assert FieldType.UNIT_VALUE.value == "unit_value"

    def test_categorical_value(self):
        assert FieldType.CATEGORICAL.value == "categorical"


class TestReconciliationStatusEnum:
    """Test ReconciliationStatus enumeration."""

    def test_member_count(self):
        assert len(ReconciliationStatus) == 6

    def test_pending_value(self):
        assert ReconciliationStatus.PENDING.value == "pending"

    def test_running_value(self):
        assert ReconciliationStatus.RUNNING.value == "running"

    def test_completed_value(self):
        assert ReconciliationStatus.COMPLETED.value == "completed"

    def test_partial_value(self):
        assert ReconciliationStatus.PARTIAL.value == "partial"


class TestTemporalGranularityEnum:
    """Test TemporalGranularity enumeration."""

    def test_member_count(self):
        assert len(TemporalGranularity) == 6

    def test_hourly_value(self):
        assert TemporalGranularity.HOURLY.value == "hourly"

    def test_quarterly_value(self):
        assert TemporalGranularity.QUARTERLY.value == "quarterly"

    def test_annual_value(self):
        assert TemporalGranularity.ANNUAL.value == "annual"


class TestCredibilityFactorEnum:
    """Test CredibilityFactor enumeration."""

    def test_member_count(self):
        assert len(CredibilityFactor) == 5

    def test_completeness_value(self):
        assert CredibilityFactor.COMPLETENESS.value == "completeness"

    def test_accuracy_value(self):
        assert CredibilityFactor.ACCURACY.value == "accuracy"

    def test_certification_value(self):
        assert CredibilityFactor.CERTIFICATION.value == "certification"


# ======================================================================
# 4. SDK Models (22)
# ======================================================================


class TestSourceDefinition:
    """Test SourceDefinition model."""

    def test_construction_with_name_only(self):
        sd = SourceDefinition(name="Test Source")
        assert sd.name == "Test Source"
        assert sd.source_type == SourceType.OTHER
        assert sd.priority == 50
        assert sd.credibility_score == pytest.approx(0.5)
        assert sd.status == SourceStatus.PENDING_VALIDATION

    def test_construction_with_all_fields(self, sample_source_data):
        sd = SourceDefinition(**sample_source_data)
        assert sd.name == "SAP ERP Production"
        assert sd.source_type == SourceType.ERP
        assert sd.priority == 80

    def test_id_auto_generated(self):
        sd = SourceDefinition(name="A")
        assert len(sd.id) == 36  # UUID format

    def test_created_at_auto_set(self):
        sd = SourceDefinition(name="A")
        assert isinstance(sd.created_at, datetime)
        assert sd.created_at.tzinfo == timezone.utc

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError, match="name must be non-empty"):
            SourceDefinition(name="")

    def test_whitespace_name_raises(self):
        with pytest.raises(ValidationError, match="name must be non-empty"):
            SourceDefinition(name="   ")

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            SourceDefinition(name="A", unknown_field="x")

    def test_priority_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            SourceDefinition(name="A", priority=0)
        with pytest.raises(ValidationError):
            SourceDefinition(name="A", priority=101)

    def test_credibility_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            SourceDefinition(name="A", credibility_score=-0.1)
        with pytest.raises(ValidationError):
            SourceDefinition(name="A", credibility_score=1.1)


class TestSchemaMapping:
    """Test SchemaMapping model."""

    def test_construction(self):
        sm = SchemaMapping(source_column="col_a", canonical_column="canonical_a")
        assert sm.source_column == "col_a"
        assert sm.canonical_column == "canonical_a"
        assert sm.transform is None
        assert sm.unit_from is None

    def test_with_optional_fields(self):
        sm = SchemaMapping(
            source_column="consumption_kg",
            canonical_column="weight_tonnes",
            transform="value * 0.001",
            unit_from="kg",
            unit_to="tonnes",
            date_format="%Y-%m-%d",
        )
        assert sm.transform == "value * 0.001"
        assert sm.unit_from == "kg"

    def test_empty_source_column_raises(self):
        with pytest.raises(ValidationError, match="source_column must be non-empty"):
            SchemaMapping(source_column="", canonical_column="b")

    def test_empty_canonical_column_raises(self):
        with pytest.raises(ValidationError, match="canonical_column must be non-empty"):
            SchemaMapping(source_column="a", canonical_column="")

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            SchemaMapping(source_column="a", canonical_column="b", extra="x")


class TestMatchKey:
    """Test MatchKey model."""

    def test_construction(self, sample_match_key_a_kwargs):
        mk = MatchKey(**sample_match_key_a_kwargs)
        assert mk.entity_id == "facility-001"
        assert mk.period == "2025-Q1"
        assert mk.metric_name == "electricity_kwh"
        assert mk.source_id == "source-a"

    def test_composite_key_auto_computed(self, sample_match_key_a_kwargs):
        mk = MatchKey(**sample_match_key_a_kwargs)
        assert len(mk.composite_key) == 16  # first 16 chars of SHA-256

    def test_composite_key_deterministic(self, sample_match_key_a_kwargs):
        mk1 = MatchKey(**sample_match_key_a_kwargs)
        mk2 = MatchKey(**sample_match_key_a_kwargs)
        assert mk1.composite_key == mk2.composite_key

    def test_different_sources_different_composite_keys(
        self, sample_match_key_a_kwargs, sample_match_key_b_kwargs
    ):
        mk_a = MatchKey(**sample_match_key_a_kwargs)
        mk_b = MatchKey(**sample_match_key_b_kwargs)
        assert mk_a.composite_key != mk_b.composite_key

    def test_empty_entity_id_raises(self):
        with pytest.raises(ValidationError, match="entity_id must be non-empty"):
            MatchKey(entity_id="", period="Q1", metric_name="x", source_id="s")

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            MatchKey(
                entity_id="e", period="Q1", metric_name="x",
                source_id="s", extra="y",
            )


class TestMatchResult:
    """Test MatchResult model."""

    def test_construction(self, sample_match_key_a_kwargs, sample_match_key_b_kwargs):
        key_a = MatchKey(**sample_match_key_a_kwargs)
        key_b = MatchKey(**sample_match_key_b_kwargs)
        mr = MatchResult(source_a_key=key_a, source_b_key=key_b, confidence=0.92)
        assert mr.confidence == pytest.approx(0.92)
        assert mr.strategy == MatchStrategy.EXACT
        assert mr.status == MatchStatus.PENDING_REVIEW

    def test_match_id_auto_generated(self, sample_match_key_a_kwargs, sample_match_key_b_kwargs):
        key_a = MatchKey(**sample_match_key_a_kwargs)
        key_b = MatchKey(**sample_match_key_b_kwargs)
        mr = MatchResult(source_a_key=key_a, source_b_key=key_b)
        assert len(mr.match_id) == 36

    def test_confidence_out_of_range_raises(self, sample_match_key_a_kwargs, sample_match_key_b_kwargs):
        key_a = MatchKey(**sample_match_key_a_kwargs)
        key_b = MatchKey(**sample_match_key_b_kwargs)
        with pytest.raises(ValidationError):
            MatchResult(source_a_key=key_a, source_b_key=key_b, confidence=1.5)

    def test_extra_field_forbidden(self, sample_match_key_a_kwargs, sample_match_key_b_kwargs):
        key_a = MatchKey(**sample_match_key_a_kwargs)
        key_b = MatchKey(**sample_match_key_b_kwargs)
        with pytest.raises(ValidationError):
            MatchResult(source_a_key=key_a, source_b_key=key_b, extra="z")


class TestFieldComparison:
    """Test FieldComparison model."""

    def test_construction(self):
        fc = FieldComparison(
            field_name="electricity_kwh",
            source_a_value=12500.0,
            source_b_value=12650.0,
            absolute_diff=150.0,
            relative_diff_pct=1.2,
            result=ComparisonResult.WITHIN_TOLERANCE,
        )
        assert fc.field_name == "electricity_kwh"
        assert fc.result == ComparisonResult.WITHIN_TOLERANCE

    def test_defaults(self):
        fc = FieldComparison(field_name="test")
        assert fc.field_type == FieldType.NUMERIC
        assert fc.result == ComparisonResult.INCOMPARABLE
        assert fc.absolute_diff is None

    def test_empty_field_name_raises(self):
        with pytest.raises(ValidationError, match="field_name must be non-empty"):
            FieldComparison(field_name="")

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            FieldComparison(field_name="f", extra="x")


class TestDiscrepancy:
    """Test Discrepancy model."""

    def test_construction(self):
        d = Discrepancy(
            match_id="match-001",
            field_name="value",
            discrepancy_type=DiscrepancyType.VALUE_MISMATCH,
            severity=DiscrepancySeverity.HIGH,
            source_a_value=100.0,
            source_b_value=130.0,
            deviation_pct=30.0,
        )
        assert d.severity == DiscrepancySeverity.HIGH
        assert d.deviation_pct == pytest.approx(30.0)

    def test_defaults(self):
        d = Discrepancy(match_id="m1", field_name="f")
        assert d.discrepancy_type == DiscrepancyType.VALUE_MISMATCH
        assert d.severity == DiscrepancySeverity.LOW
        assert d.description == ""

    def test_empty_match_id_raises(self):
        with pytest.raises(ValidationError, match="match_id must be non-empty"):
            Discrepancy(match_id="", field_name="f")

    def test_empty_field_name_raises(self):
        with pytest.raises(ValidationError, match="field_name must be non-empty"):
            Discrepancy(match_id="m", field_name="")

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            Discrepancy(match_id="m", field_name="f", extra="x")


class TestResolutionDecision:
    """Test ResolutionDecision model."""

    def test_construction(self):
        rd = ResolutionDecision(
            discrepancy_id="disc-001",
            strategy=ResolutionStrategy.PRIORITY_WINS,
            winning_source_id="source-a",
            resolved_value=100.0,
            confidence=0.95,
        )
        assert rd.confidence == pytest.approx(0.95)

    def test_defaults(self):
        rd = ResolutionDecision(discrepancy_id="d1")
        assert rd.strategy == ResolutionStrategy.PRIORITY_WINS
        assert rd.confidence == 0.0
        assert rd.reviewer is None

    def test_empty_discrepancy_id_raises(self):
        with pytest.raises(ValidationError, match="discrepancy_id must be non-empty"):
            ResolutionDecision(discrepancy_id="")

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            ResolutionDecision(discrepancy_id="d", extra="x")


class TestGoldenRecord:
    """Test GoldenRecord model."""

    def test_construction(self):
        gr = GoldenRecord(
            entity_id="facility-001",
            period="2025-Q1",
            fields={"electricity_kwh": 12575.0},
            field_sources={"electricity_kwh": "source-a"},
            field_confidences={"electricity_kwh": 0.95},
            total_confidence=0.95,
        )
        assert gr.entity_id == "facility-001"
        assert gr.total_confidence == pytest.approx(0.95)

    def test_defaults(self):
        gr = GoldenRecord(entity_id="e1", period="Q1")
        assert gr.fields == {}
        assert gr.total_confidence == 0.0

    def test_empty_entity_id_raises(self):
        with pytest.raises(ValidationError, match="entity_id must be non-empty"):
            GoldenRecord(entity_id="", period="Q1")

    def test_empty_period_raises(self):
        with pytest.raises(ValidationError, match="period must be non-empty"):
            GoldenRecord(entity_id="e1", period="")

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            GoldenRecord(entity_id="e1", period="Q1", extra="x")


class TestSourceCredibility:
    """Test SourceCredibility model."""

    def test_construction(self):
        sc = SourceCredibility(
            source_id="src-1",
            completeness_score=0.9,
            timeliness_score=0.8,
            consistency_score=0.85,
            accuracy_score=0.92,
            certification_score=1.0,
            overall_score=0.89,
            sample_size=500,
        )
        assert sc.overall_score == pytest.approx(0.89)
        assert sc.sample_size == 500

    def test_defaults(self):
        sc = SourceCredibility(source_id="s")
        assert sc.completeness_score == 0.0
        assert sc.overall_score == 0.0
        assert isinstance(sc.last_assessed, datetime)

    def test_empty_source_id_raises(self):
        with pytest.raises(ValidationError, match="source_id must be non-empty"):
            SourceCredibility(source_id="")

    def test_score_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            SourceCredibility(source_id="s", completeness_score=1.5)

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            SourceCredibility(source_id="s", extra="x")


class TestToleranceRule:
    """Test ToleranceRule model."""

    def test_construction(self):
        tr = ToleranceRule(
            field_name="weight",
            tolerance_abs=0.5,
            tolerance_pct=2.0,
            rounding_digits=2,
        )
        assert tr.field_name == "weight"
        assert tr.unit_conversion_epsilon == pytest.approx(1e-6)

    def test_defaults(self):
        tr = ToleranceRule(field_name="f")
        assert tr.field_type == FieldType.NUMERIC
        assert tr.tolerance_abs is None
        assert tr.custom_comparator is None

    def test_empty_field_name_raises(self):
        with pytest.raises(ValidationError, match="field_name must be non-empty"):
            ToleranceRule(field_name="")

    def test_negative_tolerance_raises(self):
        with pytest.raises(ValidationError):
            ToleranceRule(field_name="f", tolerance_abs=-1.0)

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            ToleranceRule(field_name="f", extra="x")


class TestReconciliationReport:
    """Test ReconciliationReport model."""

    def test_construction(self):
        rr = ReconciliationReport(
            job_id="job-001",
            total_records=1000,
            matched_records=800,
            discrepancies_found=50,
            discrepancies_resolved=45,
            golden_records_created=800,
        )
        assert rr.total_records == 1000
        assert rr.unresolved_count == 0

    def test_empty_job_id_raises(self):
        with pytest.raises(ValidationError, match="job_id must be non-empty"):
            ReconciliationReport(job_id="")

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            ReconciliationReport(job_id="j", extra="x")


class TestReconciliationJobConfig:
    """Test ReconciliationJobConfig model."""

    def test_construction(self):
        rjc = ReconciliationJobConfig(source_ids=["s1", "s2"])
        assert rjc.match_strategy == MatchStrategy.COMPOSITE
        assert rjc.resolution_strategy == ResolutionStrategy.PRIORITY_WINS
        assert rjc.enable_golden_records is True
        assert rjc.enable_temporal_alignment is False

    def test_min_two_sources_required(self):
        with pytest.raises(ValidationError):
            ReconciliationJobConfig(source_ids=["s1"])

    def test_exceeds_max_sources_raises(self):
        source_ids = [f"s{i}" for i in range(MAX_SOURCES + 1)]
        with pytest.raises(ValidationError, match="cannot exceed"):
            ReconciliationJobConfig(source_ids=source_ids)

    def test_empty_source_id_string_raises(self):
        with pytest.raises(ValidationError, match="non-empty strings"):
            ReconciliationJobConfig(source_ids=["s1", ""])

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            ReconciliationJobConfig(source_ids=["s1", "s2"], extra="x")


class TestBatchMatchResult:
    """Test BatchMatchResult model."""

    def test_construction_defaults(self):
        bmr = BatchMatchResult()
        assert bmr.total_pairs == 0
        assert bmr.matched == 0
        assert bmr.match_rate == 0.0
        assert bmr.results == []

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            BatchMatchResult(extra="x")


class TestDiscrepancySummary:
    """Test DiscrepancySummary model."""

    def test_construction_defaults(self):
        ds = DiscrepancySummary()
        assert ds.total == 0
        assert ds.by_type == {}
        assert ds.critical_count == 0

    def test_with_data(self):
        ds = DiscrepancySummary(
            total=10,
            by_type={"value_mismatch": 7, "missing_in_source": 3},
            by_severity={"critical": 2, "high": 3, "medium": 5},
            critical_count=2,
        )
        assert ds.total == 10

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            DiscrepancySummary(extra="x")


class TestResolutionSummary:
    """Test ResolutionSummary model."""

    def test_construction_defaults(self):
        rs = ResolutionSummary()
        assert rs.total_resolved == 0
        assert rs.average_confidence == 0.0

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            ResolutionSummary(extra="x")


class TestTemporalAlignment:
    """Test TemporalAlignment model."""

    def test_construction(self):
        ta = TemporalAlignment(
            source_granularity=TemporalGranularity.DAILY,
            target_granularity=TemporalGranularity.MONTHLY,
            aggregation_method="sum",
            records_aligned=100,
        )
        assert ta.source_granularity == TemporalGranularity.DAILY
        assert ta.records_interpolated == 0

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            TemporalAlignment(
                source_granularity=TemporalGranularity.DAILY,
                target_granularity=TemporalGranularity.MONTHLY,
                extra="x",
            )


class TestFieldLineage:
    """Test FieldLineage model."""

    def test_construction(self):
        fl = FieldLineage(
            field_name="electricity_kwh",
            source_id="src-1",
            original_value=12500.0,
            resolved_value=12575.0,
            confidence=0.9,
        )
        assert fl.field_name == "electricity_kwh"
        assert fl.confidence == pytest.approx(0.9)

    def test_empty_field_name_raises(self):
        with pytest.raises(ValidationError, match="field_name must be non-empty"):
            FieldLineage(field_name="", source_id="s")

    def test_empty_source_id_raises(self):
        with pytest.raises(ValidationError, match="source_id must be non-empty"):
            FieldLineage(field_name="f", source_id="")

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            FieldLineage(field_name="f", source_id="s", extra="x")


class TestReconciliationStats:
    """Test ReconciliationStats model."""

    def test_construction_defaults(self):
        rs = ReconciliationStats()
        assert rs.total_jobs == 0
        assert rs.total_sources == 0
        assert rs.avg_match_confidence == 0.0

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            ReconciliationStats(extra="x")


class TestSourceHealthMetrics:
    """Test SourceHealthMetrics model."""

    def test_construction(self):
        shm = SourceHealthMetrics(
            source_id="src-1",
            records_contributed=5000,
            discrepancy_rate=0.05,
            avg_credibility=0.85,
        )
        assert shm.records_contributed == 5000
        assert shm.last_refresh is None

    def test_empty_source_id_raises(self):
        with pytest.raises(ValidationError, match="source_id must be non-empty"):
            SourceHealthMetrics(source_id="")

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            SourceHealthMetrics(source_id="s", extra="x")


class TestComparisonSummary:
    """Test ComparisonSummary model."""

    def test_construction_defaults(self):
        cs = ComparisonSummary()
        assert cs.total_fields_compared == 0
        assert cs.match_rate == 0.0

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            ComparisonSummary(extra="x")


class TestPipelineStageResult:
    """Test PipelineStageResult model."""

    def test_construction(self):
        psr = PipelineStageResult(
            stage_name="matching",
            status=ReconciliationStatus.COMPLETED,
            records_processed=500,
            duration_ms=120.5,
        )
        assert psr.stage_name == "matching"
        assert psr.errors == []

    def test_defaults(self):
        psr = PipelineStageResult(stage_name="compare")
        assert psr.status == ReconciliationStatus.PENDING
        assert psr.duration_ms == 0.0

    def test_empty_stage_name_raises(self):
        with pytest.raises(ValidationError, match="stage_name must be non-empty"):
            PipelineStageResult(stage_name="")

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            PipelineStageResult(stage_name="s", extra="x")


class TestReconciliationEvent:
    """Test ReconciliationEvent model."""

    def test_construction(self):
        re = ReconciliationEvent(
            job_id="job-001",
            event_type="match_started",
            details={"source_count": 3},
        )
        assert re.event_type == "match_started"
        assert re.details == {"source_count": 3}
        assert isinstance(re.timestamp, datetime)

    def test_empty_job_id_raises(self):
        with pytest.raises(ValidationError, match="job_id must be non-empty"):
            ReconciliationEvent(job_id="", event_type="x")

    def test_empty_event_type_raises(self):
        with pytest.raises(ValidationError, match="event_type must be non-empty"):
            ReconciliationEvent(job_id="j", event_type="")

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            ReconciliationEvent(job_id="j", event_type="e", extra="x")


# ======================================================================
# 5. Request Models (8)
# ======================================================================


class TestCreateJobRequest:
    """Test CreateJobRequest model."""

    def test_construction(self):
        req = CreateJobRequest(source_ids=["s1", "s2"])
        assert req.match_strategy == MatchStrategy.COMPOSITE
        assert req.enable_golden_records is True
        assert req.enable_temporal_alignment is False

    def test_with_all_fields(self):
        req = CreateJobRequest(
            job_name="Q1 Reconciliation",
            source_ids=["s1", "s2", "s3"],
            match_strategy=MatchStrategy.FUZZY,
            resolution_strategy=ResolutionStrategy.CONSENSUS,
            enable_golden_records=False,
            enable_temporal_alignment=True,
            temporal_granularity=TemporalGranularity.MONTHLY,
        )
        assert req.job_name == "Q1 Reconciliation"
        assert req.temporal_granularity == TemporalGranularity.MONTHLY

    def test_min_two_sources_required(self):
        with pytest.raises(ValidationError):
            CreateJobRequest(source_ids=["s1"])

    def test_empty_source_id_raises(self):
        with pytest.raises(ValidationError, match="non-empty strings"):
            CreateJobRequest(source_ids=["s1", ""])

    def test_exceeds_max_sources_raises(self):
        source_ids = [f"s{i}" for i in range(MAX_SOURCES + 1)]
        with pytest.raises(ValidationError, match="cannot exceed"):
            CreateJobRequest(source_ids=source_ids)

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            CreateJobRequest(source_ids=["s1", "s2"], extra="x")


class TestRegisterSourceRequest:
    """Test RegisterSourceRequest model."""

    def test_construction(self):
        req = RegisterSourceRequest(name="Test Source")
        assert req.source_type == SourceType.OTHER
        assert req.priority == 50
        assert req.refresh_cadence == "daily"
        assert req.tags == []

    def test_empty_name_raises(self):
        with pytest.raises(ValidationError, match="name must be non-empty"):
            RegisterSourceRequest(name="")

    def test_priority_range(self):
        with pytest.raises(ValidationError):
            RegisterSourceRequest(name="A", priority=0)
        with pytest.raises(ValidationError):
            RegisterSourceRequest(name="A", priority=101)

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            RegisterSourceRequest(name="A", extra="x")


class TestUpdateSourceRequest:
    """Test UpdateSourceRequest model."""

    def test_all_fields_optional(self):
        req = UpdateSourceRequest()
        assert req.name is None
        assert req.priority is None
        assert req.credibility_score is None
        assert req.status is None
        assert req.schema_info is None

    def test_partial_update(self):
        req = UpdateSourceRequest(name="Updated", priority=90)
        assert req.name == "Updated"
        assert req.priority == 90

    def test_empty_name_when_provided_raises(self):
        with pytest.raises(ValidationError, match="name must be non-empty when provided"):
            UpdateSourceRequest(name="")

    def test_credibility_out_of_range_raises(self):
        with pytest.raises(ValidationError):
            UpdateSourceRequest(credibility_score=1.5)

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            UpdateSourceRequest(extra="x")


class TestMatchRequest:
    """Test MatchRequest model."""

    def test_construction(self):
        req = MatchRequest(source_ids=["s1", "s2"])
        assert req.match_strategy == MatchStrategy.COMPOSITE
        assert req.match_threshold == pytest.approx(DEFAULT_MATCH_THRESHOLD)
        assert req.enable_fuzzy is False

    def test_min_two_sources(self):
        with pytest.raises(ValidationError):
            MatchRequest(source_ids=["s1"])

    def test_empty_source_id_raises(self):
        with pytest.raises(ValidationError, match="non-empty strings"):
            MatchRequest(source_ids=["s1", ""])

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            MatchRequest(source_ids=["s1", "s2"], extra="x")


class TestCompareRequest:
    """Test CompareRequest model."""

    def test_construction(self):
        req = CompareRequest(match_ids=["m1"])
        assert req.tolerance_rules == []
        assert req.field_types == {}

    def test_min_one_match_id(self):
        with pytest.raises(ValidationError):
            CompareRequest(match_ids=[])

    def test_empty_match_id_raises(self):
        with pytest.raises(ValidationError, match="non-empty strings"):
            CompareRequest(match_ids=[""])

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            CompareRequest(match_ids=["m1"], extra="x")


class TestResolveRequest:
    """Test ResolveRequest model."""

    def test_construction(self):
        req = ResolveRequest(discrepancy_ids=["d1"])
        assert req.strategy == ResolutionStrategy.PRIORITY_WINS
        assert req.manual_values is None

    def test_min_one_discrepancy_id(self):
        with pytest.raises(ValidationError):
            ResolveRequest(discrepancy_ids=[])

    def test_empty_discrepancy_id_raises(self):
        with pytest.raises(ValidationError, match="non-empty strings"):
            ResolveRequest(discrepancy_ids=[""])

    def test_with_manual_values(self):
        req = ResolveRequest(
            discrepancy_ids=["d1"],
            strategy=ResolutionStrategy.MANUAL_REVIEW,
            manual_values={"d1": 42.0},
        )
        assert req.manual_values == {"d1": 42.0}

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            ResolveRequest(discrepancy_ids=["d1"], extra="x")


class TestPipelineRequest:
    """Test PipelineRequest model."""

    def test_construction(self):
        req = PipelineRequest(source_ids=["s1", "s2"])
        assert req.match_strategy == MatchStrategy.COMPOSITE
        assert req.enable_golden_records is True

    def test_min_two_sources(self):
        with pytest.raises(ValidationError):
            PipelineRequest(source_ids=["s1"])

    def test_exceeds_max_sources_raises(self):
        source_ids = [f"s{i}" for i in range(MAX_SOURCES + 1)]
        with pytest.raises(ValidationError, match="cannot exceed"):
            PipelineRequest(source_ids=source_ids)

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            PipelineRequest(source_ids=["s1", "s2"], extra="x")


class TestGoldenRecordRequest:
    """Test GoldenRecordRequest model."""

    def test_construction(self):
        req = GoldenRecordRequest(entity_id="e1", period="2025-Q1")
        assert req.source_priority_overrides is None

    def test_with_overrides(self):
        req = GoldenRecordRequest(
            entity_id="e1",
            period="2025-Q1",
            source_priority_overrides={"s1": 90, "s2": 50},
        )
        assert req.source_priority_overrides["s1"] == 90

    def test_empty_entity_id_raises(self):
        with pytest.raises(ValidationError, match="entity_id must be non-empty"):
            GoldenRecordRequest(entity_id="", period="Q1")

    def test_empty_period_raises(self):
        with pytest.raises(ValidationError, match="period must be non-empty"):
            GoldenRecordRequest(entity_id="e1", period="")

    def test_extra_field_forbidden(self):
        with pytest.raises(ValidationError):
            GoldenRecordRequest(entity_id="e1", period="Q1", extra="x")


# ======================================================================
# 6. Layer 1 re-exports
# ======================================================================


class TestLayer1ReExports:
    """Test that Layer 1 symbols are available in __all__ and importable."""

    def test_consistency_analyzer_available(self):
        """ConsistencyAnalyzer should be importable (None if dep missing)."""
        from greenlang.cross_source_reconciliation import models
        assert hasattr(models, "ConsistencyAnalyzer")

    def test_similarity_scorer_available(self):
        from greenlang.cross_source_reconciliation import models
        assert hasattr(models, "SimilarityScorer")

    def test_match_classifier_available(self):
        from greenlang.cross_source_reconciliation import models
        assert hasattr(models, "MatchClassifier")

    def test_factor_reconciler_available(self):
        from greenlang.cross_source_reconciliation import models
        assert hasattr(models, "FactorReconciler")

    def test_conflict_resolution_strategy_available(self):
        from greenlang.cross_source_reconciliation import models
        assert hasattr(models, "ConflictResolutionStrategy")

    def test_all_exports_list_contains_layer1(self):
        from greenlang.cross_source_reconciliation.models import __all__
        expected = [
            "ConsistencyAnalyzer",
            "SimilarityScorer",
            "MatchClassifier",
            "FactorReconciler",
            "ConflictResolutionStrategy",
        ]
        for name in expected:
            assert name in __all__

    def test_all_exports_list_contains_all_enums(self):
        from greenlang.cross_source_reconciliation.models import __all__
        enum_names = [
            "SourceType", "SourceStatus", "MatchStrategy", "MatchStatus",
            "ComparisonResult", "DiscrepancyType", "DiscrepancySeverity",
            "ResolutionStrategy", "ResolutionStatus", "FieldType",
            "ReconciliationStatus", "TemporalGranularity", "CredibilityFactor",
        ]
        for name in enum_names:
            assert name in __all__

    def test_all_exports_list_contains_all_sdk_models(self):
        from greenlang.cross_source_reconciliation.models import __all__
        model_names = [
            "SourceDefinition", "SchemaMapping", "MatchKey", "MatchResult",
            "FieldComparison", "Discrepancy", "ResolutionDecision",
            "GoldenRecord", "SourceCredibility", "ToleranceRule",
            "ReconciliationReport", "ReconciliationJobConfig",
            "BatchMatchResult", "DiscrepancySummary", "ResolutionSummary",
            "TemporalAlignment", "FieldLineage", "ReconciliationStats",
            "SourceHealthMetrics", "ComparisonSummary",
            "PipelineStageResult", "ReconciliationEvent",
        ]
        for name in model_names:
            assert name in __all__

    def test_all_exports_list_contains_all_request_models(self):
        from greenlang.cross_source_reconciliation.models import __all__
        request_names = [
            "CreateJobRequest", "RegisterSourceRequest",
            "UpdateSourceRequest", "MatchRequest", "CompareRequest",
            "ResolveRequest", "PipelineRequest", "GoldenRecordRequest",
        ]
        for name in request_names:
            assert name in __all__

    def test_all_exports_list_contains_constants(self):
        from greenlang.cross_source_reconciliation.models import __all__
        const_names = [
            "DEFAULT_MATCH_THRESHOLD", "DEFAULT_TOLERANCE_PCT",
            "DEFAULT_TOLERANCE_ABS", "MAX_SOURCES",
            "MAX_MATCH_CANDIDATES", "CRITICAL_THRESHOLD_PCT",
            "HIGH_THRESHOLD_PCT", "MEDIUM_THRESHOLD_PCT",
            "SUPPORTED_UNITS", "SUPPORTED_CURRENCIES",
        ]
        for name in const_names:
            assert name in __all__

    def test_all_exports_list_contains_utcnow(self):
        from greenlang.cross_source_reconciliation.models import __all__
        assert "_utcnow" in __all__
