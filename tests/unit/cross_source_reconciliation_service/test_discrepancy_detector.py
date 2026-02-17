# -*- coding: utf-8 -*-
"""
Unit tests for DiscrepancyDetectorEngine - AGENT-DATA-015

Tests all public methods of DiscrepancyDetectorEngine with 50+ test cases.
Validates detection, type classification, severity grading, pattern detection,
field hotspots, source reliability, filtering, summarization, prioritization,
impact scoring, and description generation.

Author: GL-TestEngineer
Date: February 2026
"""

from __future__ import annotations

import pytest
from typing import List

from greenlang.cross_source_reconciliation.discrepancy_detector import (
    DiscrepancyDetectorEngine,
    FieldComparison,
    ComparisonResult,
    FieldType,
    Discrepancy,
    DiscrepancyType,
    DiscrepancySeverity,
    DiscrepancySummary,
    _safe_abs,
    _safe_divide,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    """Create a fresh DiscrepancyDetectorEngine for each test."""
    return DiscrepancyDetectorEngine()


@pytest.fixture
def engine_custom_thresholds():
    """Engine with custom severity thresholds."""
    return DiscrepancyDetectorEngine(
        critical_pct=30.0,
        high_pct=15.0,
        medium_pct=5.0,
    )


@pytest.fixture
def mismatch_numeric_comp():
    """A numeric field comparison with MISMATCH result."""
    return FieldComparison(
        field_name="emissions_total",
        field_type=FieldType.NUMERIC,
        result=ComparisonResult.MISMATCH,
        value_a=1000.0,
        value_b=1200.0,
        deviation_pct=16.7,
    )


@pytest.fixture
def mismatch_date_comp():
    """A date field comparison with MISMATCH result."""
    return FieldComparison(
        field_name="report_date",
        field_type=FieldType.DATE,
        result=ComparisonResult.MISMATCH,
        value_a="2025-01-15",
        value_b="2025-02-01",
        deviation_pct=0.0,
    )


@pytest.fixture
def match_comp():
    """A field comparison with MATCH result."""
    return FieldComparison(
        field_name="entity_id",
        field_type=FieldType.STRING,
        result=ComparisonResult.MATCH,
        value_a="E001",
        value_b="E001",
    )


@pytest.fixture
def within_tolerance_comp():
    """A field comparison with WITHIN_TOLERANCE result."""
    return FieldComparison(
        field_name="spend",
        field_type=FieldType.NUMERIC,
        result=ComparisonResult.WITHIN_TOLERANCE,
        value_a=100.0,
        value_b=102.0,
        deviation_pct=2.0,
    )


@pytest.fixture
def missing_left_comp():
    """A field comparison with MISSING_LEFT result."""
    return FieldComparison(
        field_name="vendor_code",
        field_type=FieldType.STRING,
        result=ComparisonResult.MISSING_LEFT,
        value_a=None,
        value_b="V001",
    )


@pytest.fixture
def missing_right_comp():
    """A field comparison with MISSING_RIGHT result."""
    return FieldComparison(
        field_name="invoice_id",
        field_type=FieldType.IDENTIFIER,
        result=ComparisonResult.MISSING_RIGHT,
        value_a="INV-001",
        value_b=None,
    )


@pytest.fixture
def not_comparable_comp():
    """A field comparison with NOT_COMPARABLE result."""
    return FieldComparison(
        field_name="notes",
        field_type=FieldType.STRING,
        result=ComparisonResult.NOT_COMPARABLE,
        value_a="text_a",
        value_b="text_b",
    )


@pytest.fixture
def unit_diff_comp():
    """A numeric comparison with different units."""
    return FieldComparison(
        field_name="weight",
        field_type=FieldType.NUMERIC,
        result=ComparisonResult.MISMATCH,
        value_a=1000.0,
        value_b=1.0,
        deviation_pct=99.9,
        unit_a="kg",
        unit_b="tonnes",
    )


@pytest.fixture
def rounding_diff_comp():
    """A numeric comparison with rounding difference."""
    return FieldComparison(
        field_name="amount",
        field_type=FieldType.NUMERIC,
        result=ComparisonResult.MISMATCH,
        value_a=100.004,
        value_b=100.005,
        deviation_pct=0.001,
    )


@pytest.fixture
def categorical_mismatch_comp():
    """A categorical field comparison with MISMATCH."""
    return FieldComparison(
        field_name="fuel_type",
        field_type=FieldType.CATEGORICAL,
        result=ComparisonResult.MISMATCH,
        value_a="diesel",
        value_b="petrol",
        deviation_pct=0.0,
    )


@pytest.fixture
def boolean_mismatch_comp():
    """A boolean field comparison with MISMATCH."""
    return FieldComparison(
        field_name="is_renewable",
        field_type=FieldType.BOOLEAN,
        result=ComparisonResult.MISMATCH,
        value_a=True,
        value_b=False,
        deviation_pct=0.0,
    )


@pytest.fixture
def critical_comp():
    """A numeric comparison with CRITICAL-level deviation."""
    return FieldComparison(
        field_name="co2_emissions",
        field_type=FieldType.NUMERIC,
        result=ComparisonResult.MISMATCH,
        value_a=100.0,
        value_b=200.0,
        deviation_pct=66.7,
    )


@pytest.fixture
def high_comp():
    """A numeric comparison with HIGH-level deviation."""
    return FieldComparison(
        field_name="energy_usage",
        field_type=FieldType.NUMERIC,
        result=ComparisonResult.MISMATCH,
        value_a=100.0,
        value_b=130.0,
        deviation_pct=30.0,
    )


@pytest.fixture
def medium_comp():
    """A numeric comparison with MEDIUM-level deviation."""
    return FieldComparison(
        field_name="water_usage",
        field_type=FieldType.NUMERIC,
        result=ComparisonResult.MISMATCH,
        value_a=100.0,
        value_b=115.0,
        deviation_pct=15.0,
    )


@pytest.fixture
def low_comp():
    """A numeric comparison with LOW-level deviation."""
    return FieldComparison(
        field_name="waste_output",
        field_type=FieldType.NUMERIC,
        result=ComparisonResult.MISMATCH,
        value_a=100.0,
        value_b=103.0,
        deviation_pct=3.0,
    )


def _make_discrepancy(
    field_name: str = "test_field",
    severity: DiscrepancySeverity = DiscrepancySeverity.MEDIUM,
    disc_type: DiscrepancyType = DiscrepancyType.VALUE_MISMATCH,
    field_type: FieldType = FieldType.NUMERIC,
    deviation_pct: float = 15.0,
    signed_deviation_pct: float = 15.0,
    source_a_id: str = "erp",
    source_b_id: str = "invoice",
    entity_id: str = "",
    period: str = "",
    confidence: float = 1.0,
    match_id: str = "m-001",
) -> Discrepancy:
    """Helper to create a Discrepancy for testing."""
    return Discrepancy(
        discrepancy_id=f"disc-{field_name}",
        match_id=match_id,
        source_a_id=source_a_id,
        source_b_id=source_b_id,
        field_name=field_name,
        field_type=field_type,
        discrepancy_type=disc_type,
        severity=severity,
        value_a=100.0,
        value_b=115.0,
        deviation_pct=deviation_pct,
        signed_deviation_pct=signed_deviation_pct,
        impact_score=0.5,
        description=f"Test discrepancy for {field_name}",
        confidence=confidence,
        requires_manual_review=False,
        entity_id=entity_id,
        period=period,
    )


# ---------------------------------------------------------------------------
# TestDiscrepancyDetectorEngine: Primary Detection
# ---------------------------------------------------------------------------


class TestDetectDiscrepancies:
    """Tests for detect_discrepancies method."""

    def test_detect_generates_discrepancy_for_mismatch(
        self, engine, mismatch_numeric_comp,
    ):
        """Mismatch comparison produces a Discrepancy."""
        discs = engine.detect_discrepancies(
            [mismatch_numeric_comp], "m-001", "erp", "invoice",
        )
        assert len(discs) == 1
        assert isinstance(discs[0], Discrepancy)

    def test_detect_no_discrepancy_for_match(self, engine, match_comp):
        """MATCH comparison produces no discrepancy."""
        discs = engine.detect_discrepancies(
            [match_comp], "m-001", "erp", "invoice",
        )
        assert len(discs) == 0

    def test_detect_no_discrepancy_for_within_tolerance(
        self, engine, within_tolerance_comp,
    ):
        """WITHIN_TOLERANCE comparison produces no discrepancy."""
        discs = engine.detect_discrepancies(
            [within_tolerance_comp], "m-001", "erp", "invoice",
        )
        assert len(discs) == 0

    def test_detect_generates_for_missing_left(self, engine, missing_left_comp):
        """MISSING_LEFT comparison produces a discrepancy."""
        discs = engine.detect_discrepancies(
            [missing_left_comp], "m-001", "erp", "invoice",
        )
        assert len(discs) == 1
        assert discs[0].discrepancy_type == DiscrepancyType.MISSING_IN_SOURCE

    def test_detect_generates_for_missing_right(self, engine, missing_right_comp):
        """MISSING_RIGHT comparison produces a discrepancy."""
        discs = engine.detect_discrepancies(
            [missing_right_comp], "m-001", "erp", "invoice",
        )
        assert len(discs) == 1
        assert discs[0].discrepancy_type == DiscrepancyType.MISSING_IN_SOURCE

    def test_detect_skips_not_comparable(self, engine, not_comparable_comp):
        """NOT_COMPARABLE comparison is skipped (no discrepancy)."""
        discs = engine.detect_discrepancies(
            [not_comparable_comp], "m-001", "erp", "invoice",
        )
        assert len(discs) == 0

    def test_detect_populates_source_ids(self, engine, mismatch_numeric_comp):
        """Discrepancy has correct source IDs."""
        discs = engine.detect_discrepancies(
            [mismatch_numeric_comp], "m-001", "erp", "invoice",
        )
        assert discs[0].source_a_id == "erp"
        assert discs[0].source_b_id == "invoice"

    def test_detect_populates_match_id(self, engine, mismatch_numeric_comp):
        """Discrepancy has correct match_id."""
        discs = engine.detect_discrepancies(
            [mismatch_numeric_comp], "m-999", "erp", "invoice",
        )
        assert discs[0].match_id == "m-999"

    def test_detect_populates_entity_and_period(
        self, engine, mismatch_numeric_comp,
    ):
        """Discrepancy stores entity_id and period."""
        discs = engine.detect_discrepancies(
            [mismatch_numeric_comp], "m-001", "erp", "invoice",
            entity_id="E001", period="2025-Q1",
        )
        assert discs[0].entity_id == "E001"
        assert discs[0].period == "2025-Q1"

    def test_detect_provenance_hash_populated(self, engine, mismatch_numeric_comp):
        """Discrepancy has a non-empty provenance hash."""
        discs = engine.detect_discrepancies(
            [mismatch_numeric_comp], "m-001", "erp", "invoice",
        )
        assert discs[0].provenance_hash != ""
        assert len(discs[0].provenance_hash) == 64

    def test_detect_sorted_by_severity(self, engine, critical_comp, low_comp):
        """Discrepancies are sorted with CRITICAL first."""
        discs = engine.detect_discrepancies(
            [low_comp, critical_comp], "m-001", "erp", "invoice",
        )
        assert discs[0].severity == DiscrepancySeverity.CRITICAL
        assert discs[1].severity == DiscrepancySeverity.LOW

    def test_detect_multiple_comparisons(
        self, engine, mismatch_numeric_comp, missing_left_comp, match_comp,
    ):
        """Multiple comparisons produce the correct number of discrepancies."""
        discs = engine.detect_discrepancies(
            [mismatch_numeric_comp, missing_left_comp, match_comp],
            "m-001", "erp", "invoice",
        )
        # match_comp does not produce a discrepancy
        assert len(discs) == 2

    def test_detect_empty_comparisons(self, engine):
        """Empty comparison list produces no discrepancies."""
        discs = engine.detect_discrepancies([], "m-001", "erp", "invoice")
        assert discs == []


# ---------------------------------------------------------------------------
# TestDiscrepancyDetectorEngine: Type Classification
# ---------------------------------------------------------------------------


class TestClassifyType:
    """Tests for classify_type method."""

    def test_missing_left_is_missing_in_source(self, engine, missing_left_comp):
        """MISSING_LEFT maps to MISSING_IN_SOURCE."""
        dtype = engine.classify_type(missing_left_comp)
        assert dtype == DiscrepancyType.MISSING_IN_SOURCE

    def test_missing_right_is_missing_in_source(self, engine, missing_right_comp):
        """MISSING_RIGHT maps to MISSING_IN_SOURCE."""
        dtype = engine.classify_type(missing_right_comp)
        assert dtype == DiscrepancyType.MISSING_IN_SOURCE

    def test_date_mismatch_is_timing_difference(self, engine, mismatch_date_comp):
        """DATE field mismatch maps to TIMING_DIFFERENCE."""
        dtype = engine.classify_type(mismatch_date_comp)
        assert dtype == DiscrepancyType.TIMING_DIFFERENCE

    def test_unit_difference_detected(self, engine, unit_diff_comp):
        """Mismatched units map to UNIT_DIFFERENCE."""
        dtype = engine.classify_type(unit_diff_comp)
        assert dtype == DiscrepancyType.UNIT_DIFFERENCE

    def test_rounding_difference_detected(self, engine, rounding_diff_comp):
        """Small rounding differences map to ROUNDING_DIFFERENCE."""
        dtype = engine.classify_type(rounding_diff_comp)
        assert dtype == DiscrepancyType.ROUNDING_DIFFERENCE

    def test_categorical_mismatch_is_classification(
        self, engine, categorical_mismatch_comp,
    ):
        """Categorical field mismatch maps to CLASSIFICATION_MISMATCH."""
        dtype = engine.classify_type(categorical_mismatch_comp)
        assert dtype == DiscrepancyType.CLASSIFICATION_MISMATCH

    def test_boolean_mismatch_is_classification(self, engine, boolean_mismatch_comp):
        """Boolean field mismatch maps to CLASSIFICATION_MISMATCH."""
        dtype = engine.classify_type(boolean_mismatch_comp)
        assert dtype == DiscrepancyType.CLASSIFICATION_MISMATCH

    def test_numeric_mismatch_is_value_mismatch(self, engine, mismatch_numeric_comp):
        """Numeric field mismatch (not rounding/unit) maps to VALUE_MISMATCH."""
        dtype = engine.classify_type(mismatch_numeric_comp)
        assert dtype == DiscrepancyType.VALUE_MISMATCH

    def test_aggregation_mismatch_from_metadata(self, engine):
        """Aggregation mismatch hint in metadata maps correctly."""
        comp = FieldComparison(
            field_name="total",
            field_type=FieldType.NUMERIC,
            result=ComparisonResult.MISMATCH,
            value_a=100, value_b=200,
            deviation_pct=66.7,
            metadata={"aggregation_mismatch": True},
        )
        dtype = engine.classify_type(comp)
        assert dtype == DiscrepancyType.AGGREGATION_MISMATCH

    def test_format_difference_from_metadata(self, engine):
        """Format difference hint in metadata maps correctly."""
        comp = FieldComparison(
            field_name="phone",
            field_type=FieldType.STRING,
            result=ComparisonResult.MISMATCH,
            value_a="+1-555-1234", value_b="5551234",
            metadata={"format_difference": True},
        )
        dtype = engine.classify_type(comp)
        assert dtype == DiscrepancyType.FORMAT_DIFFERENCE


# ---------------------------------------------------------------------------
# TestDiscrepancyDetectorEngine: Severity Classification
# ---------------------------------------------------------------------------


class TestClassifySeverity:
    """Tests for classify_severity method."""

    def test_critical_threshold(self, engine):
        """Deviation >= 50% (default) is CRITICAL."""
        sev = engine.classify_severity(50.0)
        assert sev == DiscrepancySeverity.CRITICAL

    def test_above_critical(self, engine):
        """Deviation > 50% is still CRITICAL."""
        sev = engine.classify_severity(75.0)
        assert sev == DiscrepancySeverity.CRITICAL

    def test_high_threshold(self, engine):
        """Deviation >= 25% (default) is HIGH."""
        sev = engine.classify_severity(25.0)
        assert sev == DiscrepancySeverity.HIGH

    def test_medium_threshold(self, engine):
        """Deviation >= 10% (default) is MEDIUM."""
        sev = engine.classify_severity(10.0)
        assert sev == DiscrepancySeverity.MEDIUM

    def test_low_deviation(self, engine):
        """Deviation > 0 but < 10% is LOW."""
        sev = engine.classify_severity(5.0)
        assert sev == DiscrepancySeverity.LOW

    def test_zero_deviation_is_info(self, engine):
        """Zero deviation is INFO."""
        sev = engine.classify_severity(0.0)
        assert sev == DiscrepancySeverity.INFO

    def test_custom_thresholds(self, engine_custom_thresholds):
        """Custom thresholds are respected."""
        # custom: critical=30, high=15, medium=5
        assert engine_custom_thresholds.classify_severity(35.0) == DiscrepancySeverity.CRITICAL
        assert engine_custom_thresholds.classify_severity(20.0) == DiscrepancySeverity.HIGH
        assert engine_custom_thresholds.classify_severity(8.0) == DiscrepancySeverity.MEDIUM
        assert engine_custom_thresholds.classify_severity(3.0) == DiscrepancySeverity.LOW

    def test_override_thresholds_per_call(self, engine):
        """classify_severity accepts per-call threshold overrides."""
        sev = engine.classify_severity(
            12.0, critical_pct=40.0, high_pct=20.0, medium_pct=10.0,
        )
        assert sev == DiscrepancySeverity.MEDIUM

    def test_negative_deviation_treated_as_absolute(self, engine):
        """Negative deviation is treated via _safe_abs (absolute value)."""
        sev = engine.classify_severity(-55.0)
        assert sev == DiscrepancySeverity.CRITICAL


# ---------------------------------------------------------------------------
# TestDiscrepancyDetectorEngine: Pattern Detection
# ---------------------------------------------------------------------------


class TestDetectPatterns:
    """Tests for detect_patterns method."""

    def test_detect_patterns_empty_list(self, engine):
        """Empty discrepancy list returns empty patterns."""
        patterns = engine.detect_patterns([])
        assert patterns["total_analysed"] == 0
        assert patterns["systematic_bias"] == {}
        assert patterns["field_hotspots"] == []

    def test_detect_patterns_returns_all_keys(self, engine):
        """Pattern result contains all expected keys."""
        disc = _make_discrepancy()
        patterns = engine.detect_patterns([disc])
        expected_keys = {
            "systematic_bias", "type_distribution", "field_hotspots",
            "source_correlation", "temporal_patterns", "total_analysed",
        }
        assert expected_keys == set(patterns.keys())

    def test_detect_patterns_total_analysed(self, engine):
        """total_analysed reflects the number of input discrepancies."""
        discs = [_make_discrepancy(field_name=f"f{i}") for i in range(5)]
        patterns = engine.detect_patterns(discs)
        assert patterns["total_analysed"] == 5


# ---------------------------------------------------------------------------
# TestDiscrepancyDetectorEngine: Systematic Bias
# ---------------------------------------------------------------------------


class TestDetectSystematicBias:
    """Tests for detect_systematic_bias method."""

    def test_positive_bias_detected(self, engine):
        """Consistently positive signed deviation indicates A > B bias."""
        discs = [
            _make_discrepancy(
                field_name=f"f{i}",
                signed_deviation_pct=10.0,
                field_type=FieldType.NUMERIC,
            )
            for i in range(5)
        ]
        bias = engine.detect_systematic_bias(discs)
        assert "erp|invoice" in bias
        assert bias["erp|invoice"] > 0

    def test_negative_bias_detected(self, engine):
        """Consistently negative signed deviation indicates B > A bias."""
        discs = [
            _make_discrepancy(
                field_name=f"f{i}",
                signed_deviation_pct=-10.0,
                field_type=FieldType.NUMERIC,
            )
            for i in range(5)
        ]
        bias = engine.detect_systematic_bias(discs)
        assert "erp|invoice" in bias
        assert bias["erp|invoice"] < 0

    def test_no_bias_for_non_numeric(self, engine):
        """Non-numeric discrepancies are excluded from bias detection."""
        discs = [
            _make_discrepancy(
                field_type=FieldType.STRING,
                signed_deviation_pct=0.0,
            )
        ]
        bias = engine.detect_systematic_bias(discs)
        assert bias == {}

    def test_empty_list_no_bias(self, engine):
        """Empty discrepancy list returns empty bias dict."""
        bias = engine.detect_systematic_bias([])
        assert bias == {}


# ---------------------------------------------------------------------------
# TestDiscrepancyDetectorEngine: Field Hotspots
# ---------------------------------------------------------------------------


class TestDetectFieldHotspots:
    """Tests for detect_field_hotspots method."""

    def test_hotspots_counts_and_sorts(self, engine):
        """Hotspots are sorted by count descending."""
        discs = [
            _make_discrepancy(field_name="amount"),
            _make_discrepancy(field_name="amount"),
            _make_discrepancy(field_name="amount"),
            _make_discrepancy(field_name="date"),
            _make_discrepancy(field_name="vendor"),
        ]
        hotspots = engine.detect_field_hotspots(discs)
        assert hotspots[0][0] == "amount"
        assert hotspots[0][1] == 3

    def test_hotspots_includes_severity_weight(self, engine):
        """Each hotspot tuple includes average severity weight."""
        discs = [
            _make_discrepancy(
                field_name="f1",
                severity=DiscrepancySeverity.CRITICAL,
            ),
        ]
        hotspots = engine.detect_field_hotspots(discs)
        assert len(hotspots) == 1
        # (field_name, count, avg_severity_weight)
        fname, count, avg_weight = hotspots[0]
        assert fname == "f1"
        assert count == 1
        assert avg_weight == 1.0  # CRITICAL weight is 1.0

    def test_hotspots_empty_list(self, engine):
        """Empty discrepancy list returns empty hotspots."""
        hotspots = engine.detect_field_hotspots([])
        assert hotspots == []


# ---------------------------------------------------------------------------
# TestDiscrepancyDetectorEngine: Source Reliability
# ---------------------------------------------------------------------------


class TestDetectSourceReliabilityIssues:
    """Tests for detect_source_reliability_issues method."""

    def test_per_source_rates(self, engine):
        """Per-source discrepancy rates are computed correctly."""
        discs = [
            _make_discrepancy(source_a_id="erp", source_b_id="invoice"),
            _make_discrepancy(source_a_id="erp", source_b_id="invoice"),
        ]
        rates = engine.detect_source_reliability_issues(
            discs, ["erp", "invoice"], total_comparisons=10,
        )
        # Both sources involved in 2 discrepancies each out of 10
        assert rates["erp"] == pytest.approx(0.2)
        assert rates["invoice"] == pytest.approx(0.2)

    def test_source_not_involved(self, engine):
        """Source not involved in any discrepancy has rate 0."""
        discs = [
            _make_discrepancy(source_a_id="erp", source_b_id="invoice"),
        ]
        rates = engine.detect_source_reliability_issues(
            discs, ["erp", "invoice", "api"], total_comparisons=10,
        )
        assert rates["api"] == 0.0

    def test_zero_total_comparisons(self, engine):
        """Zero total comparisons uses discrepancy count as denominator."""
        discs = [_make_discrepancy()]
        rates = engine.detect_source_reliability_issues(
            discs, ["erp", "invoice"], total_comparisons=0,
        )
        # denominator = max(0, 1, 1) = 1
        assert all(0.0 <= r <= 1.0 for r in rates.values())


# ---------------------------------------------------------------------------
# TestDiscrepancyDetectorEngine: Filtering
# ---------------------------------------------------------------------------


class TestFilterDiscrepancies:
    """Tests for filter_discrepancies method."""

    def test_filter_by_severity(self, engine):
        """Filtering by min_severity returns only matching or higher."""
        discs = [
            _make_discrepancy(severity=DiscrepancySeverity.CRITICAL),
            _make_discrepancy(severity=DiscrepancySeverity.HIGH),
            _make_discrepancy(severity=DiscrepancySeverity.MEDIUM),
            _make_discrepancy(severity=DiscrepancySeverity.LOW),
        ]
        result = engine.filter_discrepancies(
            discs, min_severity=DiscrepancySeverity.HIGH,
        )
        assert len(result) == 2  # CRITICAL + HIGH

    def test_filter_by_type(self, engine):
        """Filtering by discrepancy_type keeps only that type."""
        discs = [
            _make_discrepancy(disc_type=DiscrepancyType.VALUE_MISMATCH),
            _make_discrepancy(disc_type=DiscrepancyType.MISSING_IN_SOURCE),
            _make_discrepancy(disc_type=DiscrepancyType.VALUE_MISMATCH),
        ]
        result = engine.filter_discrepancies(
            discs, discrepancy_type=DiscrepancyType.VALUE_MISMATCH,
        )
        assert len(result) == 2

    def test_filter_by_field_name(self, engine):
        """Filtering by field_name keeps only that field."""
        discs = [
            _make_discrepancy(field_name="amount"),
            _make_discrepancy(field_name="date"),
            _make_discrepancy(field_name="amount"),
        ]
        result = engine.filter_discrepancies(discs, field_name="amount")
        assert len(result) == 2

    def test_filter_by_source_id(self, engine):
        """Filtering by source_id keeps discrepancies involving that source."""
        discs = [
            _make_discrepancy(source_a_id="erp", source_b_id="invoice"),
            _make_discrepancy(source_a_id="api", source_b_id="spreadsheet"),
        ]
        result = engine.filter_discrepancies(discs, source_id="erp")
        assert len(result) == 1

    def test_filter_no_criteria_returns_all(self, engine):
        """No filter criteria returns all discrepancies."""
        discs = [_make_discrepancy() for _ in range(5)]
        result = engine.filter_discrepancies(discs)
        assert len(result) == 5

    def test_filter_combined_criteria(self, engine):
        """Multiple filters are combined with AND logic."""
        discs = [
            _make_discrepancy(
                field_name="amount",
                severity=DiscrepancySeverity.CRITICAL,
            ),
            _make_discrepancy(
                field_name="amount",
                severity=DiscrepancySeverity.LOW,
            ),
            _make_discrepancy(
                field_name="date",
                severity=DiscrepancySeverity.CRITICAL,
            ),
        ]
        result = engine.filter_discrepancies(
            discs,
            min_severity=DiscrepancySeverity.CRITICAL,
            field_name="amount",
        )
        assert len(result) == 1


# ---------------------------------------------------------------------------
# TestDiscrepancyDetectorEngine: Summarization
# ---------------------------------------------------------------------------


class TestSummarize:
    """Tests for summarize method."""

    def test_summary_total_count(self, engine):
        """Summary total matches input count."""
        discs = [_make_discrepancy() for _ in range(5)]
        summary = engine.summarize(discs)
        assert summary.total == 5

    def test_summary_by_type(self, engine):
        """Summary counts by discrepancy type."""
        discs = [
            _make_discrepancy(disc_type=DiscrepancyType.VALUE_MISMATCH),
            _make_discrepancy(disc_type=DiscrepancyType.VALUE_MISMATCH),
            _make_discrepancy(disc_type=DiscrepancyType.MISSING_IN_SOURCE),
        ]
        summary = engine.summarize(discs)
        assert summary.by_type.get("value_mismatch", 0) == 2
        assert summary.by_type.get("missing_in_source", 0) == 1

    def test_summary_by_severity(self, engine):
        """Summary counts by severity."""
        discs = [
            _make_discrepancy(severity=DiscrepancySeverity.CRITICAL),
            _make_discrepancy(severity=DiscrepancySeverity.CRITICAL),
            _make_discrepancy(severity=DiscrepancySeverity.LOW),
        ]
        summary = engine.summarize(discs)
        assert summary.critical_count == 2
        assert summary.by_severity.get("low", 0) == 1

    def test_summary_mean_deviation(self, engine):
        """Summary mean deviation is correctly computed."""
        discs = [
            _make_discrepancy(deviation_pct=10.0),
            _make_discrepancy(deviation_pct=20.0),
        ]
        summary = engine.summarize(discs)
        assert summary.mean_deviation_pct == pytest.approx(15.0, abs=0.01)

    def test_summary_max_deviation(self, engine):
        """Summary max deviation is the highest value."""
        discs = [
            _make_discrepancy(deviation_pct=10.0),
            _make_discrepancy(deviation_pct=50.0),
            _make_discrepancy(deviation_pct=25.0),
        ]
        summary = engine.summarize(discs)
        assert summary.max_deviation_pct == pytest.approx(50.0, abs=0.01)

    def test_summary_top_fields(self, engine):
        """Summary top_fields are sorted by count descending."""
        discs = [
            _make_discrepancy(field_name="amount"),
            _make_discrepancy(field_name="amount"),
            _make_discrepancy(field_name="date"),
        ]
        summary = engine.summarize(discs)
        assert summary.top_fields[0][0] == "amount"
        assert summary.top_fields[0][1] == 2

    def test_summary_empty_list(self, engine):
        """Empty discrepancy list produces zero-count summary."""
        summary = engine.summarize([])
        assert summary.total == 0
        assert summary.critical_count == 0

    def test_summary_provenance_hash(self, engine):
        """Summary has a provenance hash."""
        discs = [_make_discrepancy()]
        summary = engine.summarize(discs)
        assert summary.provenance_hash != ""
        assert len(summary.provenance_hash) == 64


# ---------------------------------------------------------------------------
# TestDiscrepancyDetectorEngine: Prioritization
# ---------------------------------------------------------------------------


class TestPrioritize:
    """Tests for prioritize method."""

    def test_prioritize_critical_first(self, engine):
        """Prioritized list has CRITICAL first."""
        discs = [
            _make_discrepancy(severity=DiscrepancySeverity.LOW, deviation_pct=5.0),
            _make_discrepancy(severity=DiscrepancySeverity.CRITICAL, deviation_pct=60.0),
            _make_discrepancy(severity=DiscrepancySeverity.MEDIUM, deviation_pct=15.0),
        ]
        prioritized = engine.prioritize(discs)
        assert prioritized[0].severity == DiscrepancySeverity.CRITICAL

    def test_prioritize_same_severity_by_deviation_desc(self, engine):
        """Within same severity, higher deviation comes first."""
        discs = [
            _make_discrepancy(severity=DiscrepancySeverity.HIGH, deviation_pct=25.0),
            _make_discrepancy(severity=DiscrepancySeverity.HIGH, deviation_pct=40.0),
        ]
        prioritized = engine.prioritize(discs)
        assert prioritized[0].deviation_pct >= prioritized[1].deviation_pct

    def test_prioritize_empty_list(self, engine):
        """Prioritizing empty list returns empty list."""
        result = engine.prioritize([])
        assert result == []

    def test_prioritize_updates_manual_review_flags(self, engine):
        """Prioritize updates requires_manual_review flags."""
        discs = [
            _make_discrepancy(
                severity=DiscrepancySeverity.CRITICAL,
                confidence=0.5,
            ),
        ]
        prioritized = engine.prioritize(discs)
        # CRITICAL always requires review
        assert prioritized[0].requires_manual_review is True


# ---------------------------------------------------------------------------
# TestDiscrepancyDetectorEngine: Impact Score
# ---------------------------------------------------------------------------


class TestComputeImpactScore:
    """Tests for compute_impact_score method."""

    def test_impact_score_in_range(self, engine):
        """Impact score is always in [0, 1]."""
        disc = _make_discrepancy(
            severity=DiscrepancySeverity.CRITICAL,
            deviation_pct=100.0,
        )
        score = engine.compute_impact_score(disc)
        assert 0.0 <= score <= 1.0

    def test_critical_high_deviation_near_one(self, engine):
        """CRITICAL severity + 100% deviation produces score near 1.0."""
        disc = _make_discrepancy(
            severity=DiscrepancySeverity.CRITICAL,
            deviation_pct=100.0,
        )
        score = engine.compute_impact_score(disc)
        assert score >= 0.9

    def test_info_zero_deviation_low_score(self, engine):
        """INFO severity + 0% deviation produces low score."""
        disc = _make_discrepancy(
            severity=DiscrepancySeverity.INFO,
            deviation_pct=0.0,
        )
        score = engine.compute_impact_score(disc)
        assert score <= 0.15

    def test_impact_formula(self, engine):
        """Impact = severity_weight * 0.6 + deviation_norm * 0.4."""
        # HIGH weight = 0.75, deviation = 30% -> deviation_norm = 0.3
        disc = _make_discrepancy(
            severity=DiscrepancySeverity.HIGH,
            deviation_pct=30.0,
        )
        expected = round(0.75 * 0.6 + 0.3 * 0.4, 4)
        score = engine.compute_impact_score(disc)
        assert score == pytest.approx(expected, abs=0.001)


# ---------------------------------------------------------------------------
# TestDiscrepancyDetectorEngine: Description Generation
# ---------------------------------------------------------------------------


class TestGenerateDescription:
    """Tests for _generate_description method."""

    def test_value_mismatch_description(self, engine, mismatch_numeric_comp):
        """Value mismatch description includes field name and values."""
        desc = engine._generate_description(
            mismatch_numeric_comp, DiscrepancyType.VALUE_MISMATCH,
        )
        assert "emissions_total" in desc
        assert "value mismatch" in desc.lower()

    def test_missing_left_description(self, engine, missing_left_comp):
        """Missing left description mentions source A."""
        desc = engine._generate_description(
            missing_left_comp, DiscrepancyType.MISSING_IN_SOURCE,
        )
        assert "missing in source A" in desc

    def test_missing_right_description(self, engine, missing_right_comp):
        """Missing right description mentions source B."""
        desc = engine._generate_description(
            missing_right_comp, DiscrepancyType.MISSING_IN_SOURCE,
        )
        assert "missing in source B" in desc

    def test_timing_difference_description(self, engine, mismatch_date_comp):
        """Timing difference description includes field name."""
        desc = engine._generate_description(
            mismatch_date_comp, DiscrepancyType.TIMING_DIFFERENCE,
        )
        assert "report_date" in desc
        assert "timing difference" in desc.lower()

    def test_unit_difference_description(self, engine, unit_diff_comp):
        """Unit difference description includes units."""
        desc = engine._generate_description(
            unit_diff_comp, DiscrepancyType.UNIT_DIFFERENCE,
        )
        assert "kg" in desc
        assert "tonnes" in desc

    def test_classification_mismatch_description(
        self, engine, categorical_mismatch_comp,
    ):
        """Classification mismatch description includes values."""
        desc = engine._generate_description(
            categorical_mismatch_comp, DiscrepancyType.CLASSIFICATION_MISMATCH,
        )
        assert "diesel" in desc
        assert "petrol" in desc

    def test_rounding_difference_description(self, engine, rounding_diff_comp):
        """Rounding difference description is generated."""
        desc = engine._generate_description(
            rounding_diff_comp, DiscrepancyType.ROUNDING_DIFFERENCE,
        )
        assert "rounding" in desc.lower()


# ---------------------------------------------------------------------------
# TestDiscrepancyDetectorEngine: Helper Functions
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    """Tests for module-level helper functions."""

    def test_safe_abs_positive(self):
        """_safe_abs returns absolute value of positive."""
        assert _safe_abs(5.0) == 5.0

    def test_safe_abs_negative(self):
        """_safe_abs returns absolute value of negative."""
        assert _safe_abs(-5.0) == 5.0

    def test_safe_abs_nan(self):
        """_safe_abs returns 0.0 for NaN."""
        assert _safe_abs(float("nan")) == 0.0

    def test_safe_abs_inf(self):
        """_safe_abs returns 0.0 for infinity."""
        assert _safe_abs(float("inf")) == 0.0

    def test_safe_divide_normal(self):
        """_safe_divide returns correct quotient."""
        assert _safe_divide(10.0, 2.0) == 5.0

    def test_safe_divide_by_zero(self):
        """_safe_divide returns 0.0 for division by zero."""
        assert _safe_divide(10.0, 0.0) == 0.0

    def test_safe_divide_nan_denominator(self):
        """_safe_divide returns 0.0 for NaN denominator."""
        assert _safe_divide(10.0, float("nan")) == 0.0


# ---------------------------------------------------------------------------
# TestDiscrepancyDetectorEngine: Provenance
# ---------------------------------------------------------------------------


class TestProvenance:
    """Tests for provenance tracking across operations."""

    def test_provenance_chain_grows(self, engine, mismatch_numeric_comp):
        """Provenance chain grows after detecting discrepancies."""
        chain_before = engine._provenance.get_chain()
        engine.detect_discrepancies(
            [mismatch_numeric_comp], "m-001", "erp", "invoice",
        )
        chain_after = engine._provenance.get_chain()
        assert len(chain_after) > len(chain_before)

    def test_provenance_chain_grows_on_summarize(self, engine):
        """Provenance chain grows after summarization."""
        chain_before = engine._provenance.get_chain()
        engine.summarize([_make_discrepancy()])
        chain_after = engine._provenance.get_chain()
        assert len(chain_after) > len(chain_before)

    def test_provenance_verify(self, engine, mismatch_numeric_comp):
        """Provenance chain can be verified."""
        engine.detect_discrepancies(
            [mismatch_numeric_comp], "m-001", "erp", "invoice",
        )
        is_valid, chain = engine.verify_provenance()
        assert is_valid is True

    def test_reset_provenance(self, engine, mismatch_numeric_comp):
        """Resetting provenance clears the chain."""
        engine.detect_discrepancies(
            [mismatch_numeric_comp], "m-001", "erp", "invoice",
        )
        engine.reset_provenance()
        chain = engine.get_provenance_chain()
        assert len(chain) == 0


# ---------------------------------------------------------------------------
# TestDiscrepancyDetectorEngine: Manual Review Flagging
# ---------------------------------------------------------------------------


class TestManualReviewFlagging:
    """Tests for manual review flagging logic."""

    def test_critical_always_flagged(self, engine, critical_comp):
        """CRITICAL discrepancies are always flagged for review."""
        discs = engine.detect_discrepancies(
            [critical_comp], "m-001", "erp", "invoice",
        )
        assert discs[0].requires_manual_review is True

    def test_high_low_confidence_flagged(self, engine):
        """HIGH severity with low confidence is flagged."""
        comp = FieldComparison(
            field_name="test",
            field_type=FieldType.NUMERIC,
            result=ComparisonResult.MISMATCH,
            value_a=100, value_b=130,
            deviation_pct=30.0,
            confidence=0.5,  # Below default 0.7 threshold
        )
        discs = engine.detect_discrepancies(
            [comp], "m-001", "erp", "invoice",
        )
        assert discs[0].requires_manual_review is True

    def test_high_high_confidence_not_flagged(self, engine):
        """HIGH severity with high confidence is NOT flagged."""
        comp = FieldComparison(
            field_name="test",
            field_type=FieldType.NUMERIC,
            result=ComparisonResult.MISMATCH,
            value_a=100, value_b=130,
            deviation_pct=30.0,
            confidence=0.9,  # Above default 0.7 threshold
        )
        discs = engine.detect_discrepancies(
            [comp], "m-001", "erp", "invoice",
        )
        assert discs[0].requires_manual_review is False

    def test_medium_never_flagged(self, engine, medium_comp):
        """MEDIUM severity discrepancies are not flagged for review."""
        discs = engine.detect_discrepancies(
            [medium_comp], "m-001", "erp", "invoice",
        )
        assert discs[0].requires_manual_review is False
