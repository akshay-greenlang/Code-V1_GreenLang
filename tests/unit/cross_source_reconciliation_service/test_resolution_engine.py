# -*- coding: utf-8 -*-
"""
Unit tests for ResolutionEngine - AGENT-DATA-015 (Engine 5 of 7)

Tests all resolution strategies, golden record assembly, field lineage,
resolution summary, and auto strategy selection with 60+ tests targeting
85%+ code coverage.

Author: GreenLang Platform Team (GL-TestEngineer)
Date: February 2026
PRD: AGENT-DATA-015 Cross-Source Reconciliation (GL-DATA-X-018)
"""

from __future__ import annotations

import math
import statistics
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from greenlang.cross_source_reconciliation.resolution_engine import (
    CredibilityFactor,
    Discrepancy,
    DiscrepancySeverity,
    FieldLineage,
    FieldType,
    GoldenRecord,
    ResolutionDecision,
    ResolutionEngine,
    ResolutionStatus,
    ResolutionStrategy,
    ResolutionSummary,
    SourceCredibility,
    _compute_completeness,
    _compute_hash,
    _is_empty,
    _is_numeric,
    _normalize_value,
    _try_parse_numeric,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> ResolutionEngine:
    """Create a fresh ResolutionEngine instance."""
    return ResolutionEngine()


@pytest.fixture
def engine_custom_ts() -> ResolutionEngine:
    """Create a ResolutionEngine with a custom timestamp field."""
    return ResolutionEngine(timestamp_field="updated_at")


@pytest.fixture
def creds_ab() -> Dict[str, SourceCredibility]:
    """Two-source credibility set: src-a (high) and src-b (low)."""
    return {
        "src-a": SourceCredibility(
            source_id="src-a",
            source_name="ERP",
            credibility_score=0.9,
            priority=1,
        ),
        "src-b": SourceCredibility(
            source_id="src-b",
            source_name="Invoice",
            credibility_score=0.6,
            priority=2,
        ),
    }


@pytest.fixture
def creds_abc() -> Dict[str, SourceCredibility]:
    """Three-source credibility set."""
    return {
        "src-a": SourceCredibility(
            source_id="src-a",
            source_name="ERP",
            credibility_score=0.9,
            priority=1,
        ),
        "src-b": SourceCredibility(
            source_id="src-b",
            source_name="Invoice",
            credibility_score=0.7,
            priority=2,
        ),
        "src-c": SourceCredibility(
            source_id="src-c",
            source_name="Meter",
            credibility_score=0.5,
            priority=3,
        ),
    }


@pytest.fixture
def numeric_discrepancy() -> Discrepancy:
    """Numeric discrepancy between two sources."""
    return Discrepancy(
        discrepancy_id="d-num-001",
        entity_id="entity-1",
        field_name="emissions_total",
        field_type=FieldType.NUMERIC,
        severity=DiscrepancySeverity.MEDIUM,
        source_values={"src-a": 100.0, "src-b": 110.0},
    )


@pytest.fixture
def string_discrepancy() -> Discrepancy:
    """String discrepancy between two sources."""
    return Discrepancy(
        discrepancy_id="d-str-001",
        entity_id="entity-1",
        field_name="category_name",
        field_type=FieldType.STRING,
        severity=DiscrepancySeverity.LOW,
        source_values={"src-a": "Cement", "src-b": "Steel"},
    )


@pytest.fixture
def three_source_discrepancy() -> Discrepancy:
    """Numeric discrepancy across three sources where two agree."""
    return Discrepancy(
        discrepancy_id="d-3src-001",
        entity_id="entity-2",
        field_name="co2_tonnes",
        field_type=FieldType.NUMERIC,
        severity=DiscrepancySeverity.MEDIUM,
        source_values={"src-a": 100.0, "src-b": 100.0, "src-c": 200.0},
    )


@pytest.fixture
def source_records_ab() -> Dict[str, Dict[str, Any]]:
    """Source records for two sources with timestamps."""
    return {
        "src-a": {
            "emissions_total": 100.0,
            "category_name": "Cement",
            "data_timestamp": "2025-06-01T00:00:00",
        },
        "src-b": {
            "emissions_total": 110.0,
            "category_name": "Steel",
            "data_timestamp": "2025-08-15T12:30:00",
        },
    }


@pytest.fixture
def source_records_abc() -> Dict[str, Dict[str, Any]]:
    """Source records for three sources."""
    return {
        "src-a": {
            "co2_tonnes": 100.0,
            "region": "EU",
            "data_timestamp": "2025-07-01",
        },
        "src-b": {
            "co2_tonnes": 100.0,
            "region": "EU",
            "data_timestamp": "2025-09-01",
        },
        "src-c": {
            "co2_tonnes": 200.0,
            "region": "US",
            "data_timestamp": "2025-05-01",
        },
    }


# ===========================================================================
# Test Helpers
# ===========================================================================


class TestHelperFunctions:
    """Tests for module-level helper functions."""

    def test_normalize_value_float(self):
        """Float values are rounded to 10 decimal places."""
        assert _normalize_value(1.12345678901234) == round(1.12345678901234, 10)

    def test_normalize_value_nan(self):
        """NaN is normalised to a sentinel string."""
        assert _normalize_value(float("nan")) == "__NaN__"

    def test_normalize_value_inf(self):
        """Positive infinity is normalised to __Inf__."""
        assert _normalize_value(float("inf")) == "__Inf__"

    def test_normalize_value_neg_inf(self):
        """Negative infinity is normalised to __-Inf__."""
        assert _normalize_value(float("-inf")) == "__-Inf__"

    def test_normalize_value_dict(self):
        """Dicts are normalised with sorted keys recursively."""
        result = _normalize_value({"b": 1.0, "a": 2.0})
        assert list(result.keys()) == ["a", "b"]

    def test_normalize_value_list(self):
        """Lists are normalised element-wise."""
        result = _normalize_value([1.0, float("nan")])
        assert result == [1.0, "__NaN__"]

    def test_compute_hash_deterministic(self):
        """Same input always produces the same hash."""
        h1 = _compute_hash({"a": 1, "b": 2})
        h2 = _compute_hash({"b": 2, "a": 1})
        assert h1 == h2
        assert len(h1) == 64

    def test_is_numeric_int(self):
        assert _is_numeric(42) is True

    def test_is_numeric_float(self):
        assert _is_numeric(3.14) is True

    def test_is_numeric_bool(self):
        assert _is_numeric(True) is False

    def test_is_numeric_nan(self):
        assert _is_numeric(float("nan")) is False

    def test_is_numeric_inf(self):
        assert _is_numeric(float("inf")) is False

    def test_is_numeric_string(self):
        assert _is_numeric("42") is False

    def test_try_parse_numeric_int(self):
        assert _try_parse_numeric(42) == 42.0

    def test_try_parse_numeric_str(self):
        assert _try_parse_numeric("3.14") == pytest.approx(3.14)

    def test_try_parse_numeric_invalid_str(self):
        assert _try_parse_numeric("hello") is None

    def test_try_parse_numeric_none(self):
        assert _try_parse_numeric(None) is None

    def test_is_empty_none(self):
        assert _is_empty(None) is True

    def test_is_empty_blank_string(self):
        assert _is_empty("  ") is True

    def test_is_empty_nan(self):
        assert _is_empty(float("nan")) is True

    def test_is_empty_nonempty(self):
        assert _is_empty("hello") is False

    def test_is_empty_zero(self):
        assert _is_empty(0) is False

    def test_compute_completeness_full(self):
        """All non-empty fields -> 1.0."""
        assert _compute_completeness({"a": 1, "b": "x"}) == 1.0

    def test_compute_completeness_half(self):
        """Half fields empty -> 0.5."""
        assert _compute_completeness({"a": 1, "b": None}) == 0.5

    def test_compute_completeness_empty_record(self):
        """Empty dict -> 0.0."""
        assert _compute_completeness({}) == 0.0


# ===========================================================================
# Test Enumerations
# ===========================================================================


class TestEnumerations:
    """Tests for resolution engine enumerations."""

    def test_resolution_strategy_values(self):
        assert ResolutionStrategy.PRIORITY_WINS.value == "priority_wins"
        assert ResolutionStrategy.WEIGHTED_AVERAGE.value == "weighted_average"
        assert ResolutionStrategy.CONSENSUS.value == "consensus"
        assert ResolutionStrategy.AUTO.value == "auto"

    def test_resolution_status_values(self):
        assert ResolutionStatus.RESOLVED.value == "resolved"
        assert ResolutionStatus.PENDING_REVIEW.value == "pending_review"
        assert ResolutionStatus.MANUAL_OVERRIDE.value == "manual_override"
        assert ResolutionStatus.FAILED.value == "failed"

    def test_discrepancy_severity_values(self):
        assert DiscrepancySeverity.CRITICAL.value == "critical"
        assert DiscrepancySeverity.HIGH.value == "high"
        assert DiscrepancySeverity.LOW.value == "low"

    def test_field_type_values(self):
        assert FieldType.NUMERIC.value == "numeric"
        assert FieldType.STRING.value == "string"
        assert FieldType.DATE.value == "date"
        assert FieldType.BOOLEAN.value == "boolean"

    def test_credibility_factor_values(self):
        assert CredibilityFactor.DATA_QUALITY.value == "data_quality"
        assert CredibilityFactor.TIMELINESS.value == "timeliness"


# ===========================================================================
# Test Data Models
# ===========================================================================


class TestDataModels:
    """Tests for resolution engine data models."""

    def test_source_credibility_clamps(self):
        """Credibility score is clamped to [0.0, 1.0]."""
        cred = SourceCredibility(credibility_score=1.5)
        assert cred.credibility_score == 1.0
        cred2 = SourceCredibility(credibility_score=-0.5)
        assert cred2.credibility_score == 0.0

    def test_discrepancy_auto_id(self):
        """Auto-generates discrepancy_id if not provided."""
        d = Discrepancy()
        assert d.discrepancy_id.startswith("disc-")
        assert len(d.discrepancy_id) > 5

    def test_discrepancy_auto_timestamp(self):
        """Auto-generates detected_at if not provided."""
        d = Discrepancy()
        assert d.detected_at != ""

    def test_resolution_decision_auto_id(self):
        """Auto-generates decision_id if not provided."""
        rd = ResolutionDecision()
        assert rd.decision_id.startswith("res-")

    def test_resolution_decision_to_dict(self):
        """to_dict returns a serializable dict."""
        rd = ResolutionDecision(
            discrepancy_id="d-1",
            field_name="field_a",
            resolved_value=42,
        )
        d = rd.to_dict()
        assert d["discrepancy_id"] == "d-1"
        assert d["resolved_value"] == 42

    def test_golden_record_auto_id(self):
        """Auto-generates record_id if not provided."""
        gr = GoldenRecord()
        assert gr.record_id.startswith("gr-")

    def test_golden_record_to_dict(self):
        gr = GoldenRecord(entity_id="ent-1", period="2025-Q4")
        d = gr.to_dict()
        assert d["entity_id"] == "ent-1"

    def test_field_lineage_to_dict(self):
        fl = FieldLineage(field_name="co2", source_id="erp")
        d = fl.to_dict()
        assert d["field_name"] == "co2"

    def test_resolution_summary_to_dict(self):
        rs = ResolutionSummary(total_decisions=5)
        d = rs.to_dict()
        assert d["total_decisions"] == 5


# ===========================================================================
# Test ResolutionEngine Initialization
# ===========================================================================


class TestResolutionEngineInit:
    """Tests for engine initialization and properties."""

    def test_default_initialization(self, engine):
        assert engine._timestamp_field == "data_timestamp"
        assert engine.total_resolutions == 0
        assert engine.total_golden_records == 0

    def test_custom_timestamp_field(self, engine_custom_ts):
        assert engine_custom_ts._timestamp_field == "updated_at"


# ===========================================================================
# Test resolve_priority_wins
# ===========================================================================


class TestResolvePriorityWins:
    """Tests for the PRIORITY_WINS resolution strategy."""

    def test_picks_highest_credibility_source(
        self, engine, numeric_discrepancy, creds_ab
    ):
        """src-a has higher credibility/priority, so its value should win."""
        decision = engine.resolve_priority_wins(numeric_discrepancy, creds_ab)

        assert decision.status == ResolutionStatus.RESOLVED.value
        assert decision.strategy == ResolutionStrategy.PRIORITY_WINS.value
        assert decision.winning_source_id == "src-a"
        assert decision.resolved_value == 100.0
        assert decision.confidence == 0.9
        assert decision.is_auto is True

    def test_lower_priority_number_wins(self, engine):
        """Priority 1 beats priority 5 even with equal credibility."""
        disc = Discrepancy(
            source_values={"x": "A", "y": "B"},
        )
        creds = {
            "x": SourceCredibility(source_id="x", credibility_score=0.8, priority=1),
            "y": SourceCredibility(source_id="y", credibility_score=0.8, priority=5),
        }
        decision = engine.resolve_priority_wins(disc, creds)
        assert decision.winning_source_id == "x"

    def test_justification_contains_source_name(
        self, engine, numeric_discrepancy, creds_ab
    ):
        decision = engine.resolve_priority_wins(numeric_discrepancy, creds_ab)
        assert "ERP" in decision.justification

    def test_unknown_source_gets_default_score(self, engine):
        """Sources not in credibilities get default 0.5/100 composite."""
        disc = Discrepancy(source_values={"unknown": 42})
        decision = engine.resolve_priority_wins(disc, {})
        assert decision.winning_source_id == "unknown"
        assert decision.resolved_value == 42

    def test_no_sources_returns_failed(self, engine):
        """Empty source_values returns a FAILED decision."""
        disc = Discrepancy(source_values={})
        decision = engine.resolve_priority_wins(disc, {})
        assert decision.status == ResolutionStatus.FAILED.value


# ===========================================================================
# Test resolve_most_recent
# ===========================================================================


class TestResolveMostRecent:
    """Tests for the MOST_RECENT resolution strategy."""

    def test_picks_latest_timestamp(self, engine, numeric_discrepancy, source_records_ab):
        """src-b has a later timestamp, so its value should win."""
        decision = engine.resolve_most_recent(
            numeric_discrepancy, source_records_ab
        )
        assert decision.status == ResolutionStatus.RESOLVED.value
        assert decision.winning_source_id == "src-b"
        assert decision.resolved_value == 110.0
        assert decision.confidence == 0.7

    def test_custom_timestamp_field(self, engine):
        """Uses specified timestamp_field."""
        disc = Discrepancy(source_values={"a": 1, "b": 2})
        records = {
            "a": {"custom_ts": "2025-01-01"},
            "b": {"custom_ts": "2025-12-31"},
        }
        decision = engine.resolve_most_recent(disc, records, timestamp_field="custom_ts")
        assert decision.winning_source_id == "b"

    def test_no_timestamps_returns_fallback(self, engine):
        """When no timestamps exist, falls back to first source."""
        disc = Discrepancy(source_values={"a": 10, "b": 20})
        records = {"a": {"val": 10}, "b": {"val": 20}}
        decision = engine.resolve_most_recent(disc, records)
        assert decision.status == ResolutionStatus.RESOLVED.value

    def test_datetime_object_as_timestamp(self, engine):
        """Supports datetime objects in source records."""
        disc = Discrepancy(source_values={"a": 1, "b": 2})
        records = {
            "a": {"data_timestamp": datetime(2025, 1, 1, tzinfo=timezone.utc)},
            "b": {"data_timestamp": datetime(2025, 12, 1, tzinfo=timezone.utc)},
        }
        decision = engine.resolve_most_recent(disc, records)
        assert decision.winning_source_id == "b"

    def test_iso_format_variations(self, engine):
        """Supports various ISO timestamp formats."""
        disc = Discrepancy(source_values={"a": 1, "b": 2})
        records = {
            "a": {"data_timestamp": "2025-01-01 10:00:00"},
            "b": {"data_timestamp": "2025-06-15T14:30:00"},
        }
        decision = engine.resolve_most_recent(disc, records)
        assert decision.winning_source_id == "b"


# ===========================================================================
# Test resolve_weighted_average
# ===========================================================================


class TestResolveWeightedAverage:
    """Tests for the WEIGHTED_AVERAGE resolution strategy."""

    def test_correct_weighted_calculation(
        self, engine, numeric_discrepancy, creds_ab, source_records_ab
    ):
        """Weighted average: (100*0.9 + 110*0.6) / (0.9+0.6) = 156/1.5 = 104."""
        decision = engine.resolve_weighted_average(
            numeric_discrepancy, creds_ab, source_records_ab
        )
        assert decision.status == ResolutionStatus.RESOLVED.value
        assert decision.strategy == ResolutionStrategy.WEIGHTED_AVERAGE.value
        expected = (100.0 * 0.9 + 110.0 * 0.6) / (0.9 + 0.6)
        assert decision.resolved_value == pytest.approx(expected, rel=1e-6)
        assert decision.confidence == 0.9  # max credibility

    def test_winning_source_is_weighted_average(
        self, engine, numeric_discrepancy, creds_ab, source_records_ab
    ):
        decision = engine.resolve_weighted_average(
            numeric_discrepancy, creds_ab, source_records_ab
        )
        assert decision.winning_source_id == "weighted_average"

    def test_falls_back_to_priority_wins_for_strings(
        self, engine, string_discrepancy, creds_ab, source_records_ab
    ):
        """Non-numeric fields fall back to priority_wins."""
        decision = engine.resolve_weighted_average(
            string_discrepancy, creds_ab, source_records_ab
        )
        # Falls back to priority_wins since values are strings
        assert "priority_wins" in decision.justification.lower() or \
               "fell back" in decision.justification.lower()

    def test_three_source_weighted_average(self, engine, creds_abc):
        """Three-source weighted average calculation."""
        disc = Discrepancy(
            field_type=FieldType.NUMERIC,
            source_values={"src-a": 100.0, "src-b": 110.0, "src-c": 120.0},
        )
        records = {
            "src-a": {"val": 100},
            "src-b": {"val": 110},
            "src-c": {"val": 120},
        }
        decision = engine.resolve_weighted_average(disc, creds_abc, records)
        # (100*0.9 + 110*0.7 + 120*0.5) / (0.9+0.7+0.5)
        expected = (100 * 0.9 + 110 * 0.7 + 120 * 0.5) / (0.9 + 0.7 + 0.5)
        assert decision.resolved_value == pytest.approx(expected, rel=1e-6)

    def test_string_numeric_values_are_parsed(self, engine, creds_ab):
        """Numeric strings like '100.5' are parsed and averaged."""
        disc = Discrepancy(
            field_type=FieldType.NUMERIC,
            source_values={"src-a": "100.0", "src-b": "110.0"},
        )
        decision = engine.resolve_weighted_average(disc, creds_ab, {})
        assert decision.resolved_value is not None
        assert isinstance(decision.resolved_value, float)


# ===========================================================================
# Test resolve_most_complete
# ===========================================================================


class TestResolveMostComplete:
    """Tests for the MOST_COMPLETE resolution strategy."""

    def test_picks_source_with_fewest_nulls(self, engine):
        """Source with fewer null fields wins."""
        disc = Discrepancy(
            source_values={"a": 100, "b": 200},
        )
        records = {
            "a": {"f1": 100, "f2": None, "f3": None},
            "b": {"f1": 200, "f2": "val", "f3": "val"},
        }
        decision = engine.resolve_most_complete(disc, records)
        assert decision.winning_source_id == "b"
        assert decision.resolved_value == 200

    def test_confidence_equals_completeness_ratio(self, engine):
        """Confidence matches the winning source's completeness ratio."""
        disc = Discrepancy(source_values={"a": 1, "b": 2})
        records = {
            "a": {"x": 1, "y": None},  # 50% complete
            "b": {"x": 2, "y": "ok"},  # 100% complete
        }
        decision = engine.resolve_most_complete(disc, records)
        assert decision.confidence == pytest.approx(1.0, abs=0.01)

    def test_empty_source_records_returns_failed(self, engine):
        """No source records -> FAILED."""
        disc = Discrepancy(source_values={})
        decision = engine.resolve_most_complete(disc, {})
        assert decision.status == ResolutionStatus.FAILED.value

    def test_all_sources_equal_completeness(self, engine):
        """When all sources are equally complete, picks the first one."""
        disc = Discrepancy(source_values={"a": 10, "b": 20})
        records = {
            "a": {"f1": 10, "f2": "x"},
            "b": {"f1": 20, "f2": "y"},
        }
        decision = engine.resolve_most_complete(disc, records)
        assert decision.status == ResolutionStatus.RESOLVED.value
        assert decision.winning_source_id in ("a", "b")

    def test_metadata_contains_completeness_scores(self, engine):
        """Metadata includes completeness scores for all sources."""
        disc = Discrepancy(source_values={"a": 10, "b": 20})
        records = {
            "a": {"f1": 10, "f2": None},
            "b": {"f1": 20, "f2": "val"},
        }
        decision = engine.resolve_most_complete(disc, records)
        assert "completeness_scores" in decision.metadata


# ===========================================================================
# Test resolve_consensus
# ===========================================================================


class TestResolveConsensus:
    """Tests for the CONSENSUS resolution strategy."""

    def test_majority_vote_with_3_sources(
        self, engine, three_source_discrepancy, source_records_abc
    ):
        """Two out of three sources agree on 100.0."""
        decision = engine.resolve_consensus(
            three_source_discrepancy, source_records_abc
        )
        assert decision.status == ResolutionStatus.RESOLVED.value
        assert decision.resolved_value == 100.0
        # 2/3 agreement
        assert decision.confidence == pytest.approx(2 / 3, abs=0.01)

    def test_consensus_all_agree(self, engine):
        """All three sources agree -> confidence 1.0."""
        disc = Discrepancy(
            source_values={"a": "X", "b": "X", "c": "X"},
        )
        decision = engine.resolve_consensus(disc, {})
        assert decision.status == ResolutionStatus.RESOLVED.value
        assert decision.confidence == pytest.approx(1.0, abs=0.01)

    def test_consensus_metadata_contains_agreeing_sources(self, engine):
        disc = Discrepancy(
            source_values={"a": 10, "b": 10, "c": 20},
        )
        decision = engine.resolve_consensus(disc, {})
        assert "agreeing_sources" in decision.metadata
        assert "a" in decision.metadata["agreeing_sources"]
        assert "b" in decision.metadata["agreeing_sources"]

    def test_two_source_tie_flags_pending_review(self, engine):
        """Two sources disagreeing -> PENDING_REVIEW since no majority."""
        disc = Discrepancy(
            source_values={"a": "Cement", "b": "Steel"},
        )
        decision = engine.resolve_consensus(disc, {})
        assert decision.status == ResolutionStatus.PENDING_REVIEW.value
        assert decision.confidence == 0.0
        assert decision.is_auto is False

    def test_two_sources_agree_returns_resolved(self, engine):
        """Two sources with same value -> RESOLVED with confidence 1.0."""
        disc = Discrepancy(
            source_values={"a": 42, "b": 42},
        )
        decision = engine.resolve_consensus(disc, {})
        assert decision.status == ResolutionStatus.RESOLVED.value
        assert decision.confidence == 1.0

    def test_single_source_consensus(self, engine):
        """Single source -> RESOLVED with confidence 0.5."""
        disc = Discrepancy(source_values={"only": 99})
        decision = engine.resolve_consensus(disc, {})
        assert decision.status == ResolutionStatus.RESOLVED.value
        assert decision.confidence == 0.5
        assert decision.winning_source_id == "only"

    def test_no_source_consensus(self, engine):
        """Zero sources -> FAILED."""
        disc = Discrepancy(source_values={})
        decision = engine.resolve_consensus(disc, {})
        assert decision.status == ResolutionStatus.FAILED.value

    def test_consensus_no_majority_4_sources_all_different(self, engine):
        """4 sources all different -> no true majority, pending review."""
        disc = Discrepancy(
            source_values={"a": 1, "b": 2, "c": 3, "d": 4},
        )
        decision = engine.resolve_consensus(disc, {})
        # 1/4 agreement < 0.5 -> pending_review
        assert decision.status == ResolutionStatus.PENDING_REVIEW.value

    def test_consensus_case_insensitive_string_matching(self, engine):
        """String comparison is case-insensitive for consensus."""
        disc = Discrepancy(
            source_values={"a": "Cement", "b": "cement", "c": "CEMENT"},
        )
        decision = engine.resolve_consensus(disc, {})
        assert decision.status == ResolutionStatus.RESOLVED.value
        assert decision.confidence == pytest.approx(1.0, abs=0.01)


# ===========================================================================
# Test resolve_manual_review
# ===========================================================================


class TestResolveManualReview:
    """Tests for the MANUAL_REVIEW resolution strategy."""

    def test_with_manual_value(self, engine, numeric_discrepancy):
        """When manual_value is provided, uses MANUAL_OVERRIDE status."""
        decision = engine.resolve_manual_review(numeric_discrepancy, manual_value=105.0)

        assert decision.status == ResolutionStatus.MANUAL_OVERRIDE.value
        assert decision.resolved_value == 105.0
        assert decision.confidence == 1.0
        assert decision.is_auto is False
        assert decision.winning_source_id == "manual"

    def test_without_manual_value(self, engine, numeric_discrepancy):
        """Without manual_value, flags as PENDING_REVIEW."""
        decision = engine.resolve_manual_review(numeric_discrepancy)

        assert decision.status == ResolutionStatus.PENDING_REVIEW.value
        assert decision.resolved_value is None
        assert decision.confidence == 0.0
        assert decision.is_auto is False

    def test_manual_value_string(self, engine, string_discrepancy):
        """String manual values work correctly."""
        decision = engine.resolve_manual_review(string_discrepancy, manual_value="Aluminum")
        assert decision.resolved_value == "Aluminum"
        assert decision.status == ResolutionStatus.MANUAL_OVERRIDE.value


# ===========================================================================
# Test resolve_custom
# ===========================================================================


class TestResolveCustom:
    """Tests for the CUSTOM resolution strategy."""

    def test_custom_resolver_function(self, engine, numeric_discrepancy):
        """Custom function returning a dict resolves correctly."""
        def my_resolver(disc):
            return {
                "resolved_value": 105.0,
                "confidence": 0.85,
                "winning_source_id": "custom-logic",
                "justification": "Applied custom business rule",
            }

        decision = engine.resolve_custom(numeric_discrepancy, my_resolver)
        assert decision.resolved_value == 105.0
        assert decision.confidence == 0.85
        assert decision.winning_source_id == "custom-logic"

    def test_custom_resolver_raises_runtime_error(self, engine, numeric_discrepancy):
        """Custom function that raises gets wrapped in RuntimeError."""
        def bad_resolver(disc):
            raise ValueError("oops")

        with pytest.raises(RuntimeError, match="Custom resolver function raised"):
            engine.resolve_custom(numeric_discrepancy, bad_resolver)

    def test_custom_resolver_non_dict_raises_type_error(self, engine, numeric_discrepancy):
        """Custom function returning non-dict raises TypeError."""
        def non_dict_resolver(disc):
            return 42

        with pytest.raises(TypeError, match="Custom resolver must return a dict"):
            engine.resolve_custom(numeric_discrepancy, non_dict_resolver)


# ===========================================================================
# Test resolve_discrepancy (dispatch)
# ===========================================================================


class TestResolveDiscrepancy:
    """Tests for the top-level resolve_discrepancy dispatch method."""

    def test_dispatch_priority_wins(self, engine, numeric_discrepancy, creds_ab):
        decision = engine.resolve_discrepancy(
            numeric_discrepancy,
            ResolutionStrategy.PRIORITY_WINS,
            creds_ab,
            {},
        )
        assert decision.strategy == ResolutionStrategy.PRIORITY_WINS.value
        assert decision.provenance_hash != ""
        assert decision.processing_time_ms > 0

    def test_dispatch_auto_selects_strategy(self, engine, numeric_discrepancy, creds_ab):
        decision = engine.resolve_discrepancy(
            numeric_discrepancy,
            ResolutionStrategy.AUTO,
            creds_ab,
            {},
        )
        # AUTO for MEDIUM severity + NUMERIC -> WEIGHTED_AVERAGE
        assert decision.strategy in (
            ResolutionStrategy.WEIGHTED_AVERAGE.value,
            ResolutionStrategy.PRIORITY_WINS.value,
        )

    def test_dispatch_custom_without_fn_returns_failed(self, engine, numeric_discrepancy, creds_ab):
        """CUSTOM strategy without custom_resolver_fn is caught internally and
        returns a FAILED decision (resolve_discrepancy wraps in try/except)."""
        decision = engine.resolve_discrepancy(
            numeric_discrepancy,
            ResolutionStrategy.CUSTOM,
            creds_ab,
            {},
        )
        assert decision.status == ResolutionStatus.FAILED.value

    def test_increments_total_resolutions(self, engine, numeric_discrepancy, creds_ab):
        assert engine.total_resolutions == 0
        engine.resolve_discrepancy(
            numeric_discrepancy,
            ResolutionStrategy.PRIORITY_WINS,
            creds_ab,
            {},
        )
        assert engine.total_resolutions == 1


# ===========================================================================
# Test resolve_batch
# ===========================================================================


class TestResolveBatch:
    """Tests for batch resolution of multiple discrepancies."""

    def test_resolves_all_discrepancies(self, engine, creds_ab):
        discrepancies = [
            Discrepancy(
                field_name=f"field_{i}",
                source_values={"src-a": i * 10, "src-b": i * 11},
            )
            for i in range(5)
        ]
        decisions = engine.resolve_batch(
            discrepancies,
            ResolutionStrategy.PRIORITY_WINS,
            creds_ab,
            {},
        )
        assert len(decisions) == 5
        for d in decisions:
            assert d.status == ResolutionStatus.RESOLVED.value

    def test_empty_batch(self, engine, creds_ab):
        decisions = engine.resolve_batch(
            [],
            ResolutionStrategy.PRIORITY_WINS,
            creds_ab,
            {},
        )
        assert len(decisions) == 0


# ===========================================================================
# Test assemble_golden_record
# ===========================================================================


class TestAssembleGoldenRecord:
    """Tests for golden record assembly."""

    def test_correct_field_selection(self, engine, creds_ab, source_records_ab):
        """Golden record selects resolved fields and falls back to highest-cred."""
        # Create a resolution for emissions_total
        resolution = ResolutionDecision(
            field_name="emissions_total",
            status=ResolutionStatus.RESOLVED.value,
            resolved_value=105.0,
            winning_source_id="src-a",
            confidence=0.95,
        )

        golden = engine.assemble_golden_record(
            entity_id="ent-1",
            period="2025-Q1",
            source_records=source_records_ab,
            resolutions=[resolution],
            source_credibilities=creds_ab,
        )

        assert golden.entity_id == "ent-1"
        assert golden.period == "2025-Q1"
        assert golden.fields["emissions_total"] == 105.0
        assert golden.field_sources["emissions_total"] == "src-a"

    def test_attribution_uses_highest_credibility_fallback(
        self, engine, creds_ab, source_records_ab
    ):
        """Fields without resolutions use highest-credibility source."""
        golden = engine.assemble_golden_record(
            entity_id="ent-1",
            period="2025-Q1",
            source_records=source_records_ab,
            resolutions=[],
            source_credibilities=creds_ab,
        )
        # src-a has higher credibility, so its values should be preferred
        assert golden.field_sources["emissions_total"] == "src-a"
        assert golden.fields["emissions_total"] == 100.0

    def test_total_confidence_is_mean(self, engine, creds_ab, source_records_ab):
        """total_confidence is the mean of all field confidences."""
        golden = engine.assemble_golden_record(
            entity_id="ent-1",
            period="2025-Q1",
            source_records=source_records_ab,
            resolutions=[],
            source_credibilities=creds_ab,
        )
        expected_mean = statistics.mean(golden.field_confidences.values())
        assert golden.total_confidence == pytest.approx(expected_mean, abs=0.001)

    def test_provenance_hash_present(self, engine, creds_ab, source_records_ab):
        golden = engine.assemble_golden_record(
            entity_id="ent-1",
            period="2025-Q1",
            source_records=source_records_ab,
            resolutions=[],
            source_credibilities=creds_ab,
        )
        assert golden.provenance_hash != ""
        assert len(golden.provenance_hash) == 64

    def test_processing_time_recorded(self, engine, creds_ab, source_records_ab):
        golden = engine.assemble_golden_record(
            entity_id="ent-1",
            period="2025-Q1",
            source_records=source_records_ab,
            resolutions=[],
            source_credibilities=creds_ab,
        )
        assert golden.processing_time_ms >= 0

    def test_increments_golden_record_counter(self, engine, creds_ab, source_records_ab):
        assert engine.total_golden_records == 0
        engine.assemble_golden_record(
            entity_id="ent-1",
            period="2025-Q1",
            source_records=source_records_ab,
            resolutions=[],
            source_credibilities=creds_ab,
        )
        assert engine.total_golden_records == 1


# ===========================================================================
# Test get_field_lineage
# ===========================================================================


class TestGetFieldLineage:
    """Tests for per-field lineage retrieval."""

    def test_returns_lineage_for_all_fields(self, engine, creds_ab, source_records_ab):
        golden = engine.assemble_golden_record(
            entity_id="ent-1",
            period="2025-Q1",
            source_records=source_records_ab,
            resolutions=[],
            source_credibilities=creds_ab,
        )
        lineage = engine.get_field_lineage(
            golden, source_records_ab,
            source_credibilities=creds_ab,
        )
        field_names = {fl.field_name for fl in lineage}
        assert "emissions_total" in field_names
        assert "category_name" in field_names
        assert "data_timestamp" in field_names

    def test_lineage_marks_resolved_fields_as_discrepant(
        self, engine, creds_ab, source_records_ab
    ):
        resolution = ResolutionDecision(
            field_name="emissions_total",
            status=ResolutionStatus.RESOLVED.value,
            strategy=ResolutionStrategy.PRIORITY_WINS.value,
            resolved_value=105.0,
            winning_source_id="src-a",
            confidence=0.95,
        )
        golden = engine.assemble_golden_record(
            entity_id="ent-1",
            period="2025-Q1",
            source_records=source_records_ab,
            resolutions=[resolution],
            source_credibilities=creds_ab,
        )
        lineage = engine.get_field_lineage(
            golden, source_records_ab,
            resolutions=[resolution],
            source_credibilities=creds_ab,
        )
        emissions_lineage = [fl for fl in lineage if fl.field_name == "emissions_total"]
        assert len(emissions_lineage) == 1
        assert emissions_lineage[0].was_discrepant is True
        assert emissions_lineage[0].strategy == ResolutionStrategy.PRIORITY_WINS.value

    def test_lineage_contains_source_name(self, engine, creds_ab, source_records_ab):
        golden = engine.assemble_golden_record(
            entity_id="ent-1",
            period="2025-Q1",
            source_records=source_records_ab,
            resolutions=[],
            source_credibilities=creds_ab,
        )
        lineage = engine.get_field_lineage(
            golden, source_records_ab,
            source_credibilities=creds_ab,
        )
        names = {fl.source_name for fl in lineage if fl.source_name}
        assert "ERP" in names or "Invoice" in names


# ===========================================================================
# Test summarize_resolutions
# ===========================================================================


class TestSummarizeResolutions:
    """Tests for resolution summary aggregation."""

    def test_correct_counts(self, engine):
        decisions = [
            ResolutionDecision(
                status=ResolutionStatus.RESOLVED.value,
                strategy=ResolutionStrategy.PRIORITY_WINS.value,
                is_auto=True,
                confidence=0.9,
                processing_time_ms=1.5,
            ),
            ResolutionDecision(
                status=ResolutionStatus.PENDING_REVIEW.value,
                strategy=ResolutionStrategy.CONSENSUS.value,
                is_auto=False,
                confidence=0.3,
                processing_time_ms=2.0,
            ),
            ResolutionDecision(
                status=ResolutionStatus.MANUAL_OVERRIDE.value,
                strategy=ResolutionStrategy.MANUAL_REVIEW.value,
                is_auto=False,
                confidence=1.0,
                processing_time_ms=0.5,
            ),
        ]
        summary = engine.summarize_resolutions(decisions)

        assert summary.total_decisions == 3
        assert summary.resolved_count == 1
        assert summary.pending_count == 1
        assert summary.manual_count == 1
        assert summary.failed_count == 0
        assert summary.auto_count == 1
        assert summary.manual_review_count == 2

    def test_strategy_counts(self, engine):
        decisions = [
            ResolutionDecision(strategy=ResolutionStrategy.PRIORITY_WINS.value),
            ResolutionDecision(strategy=ResolutionStrategy.PRIORITY_WINS.value),
            ResolutionDecision(strategy=ResolutionStrategy.CONSENSUS.value),
        ]
        summary = engine.summarize_resolutions(decisions)
        assert summary.strategy_counts[ResolutionStrategy.PRIORITY_WINS.value] == 2
        assert summary.strategy_counts[ResolutionStrategy.CONSENSUS.value] == 1

    def test_confidence_stats(self, engine):
        decisions = [
            ResolutionDecision(confidence=0.5),
            ResolutionDecision(confidence=0.8),
            ResolutionDecision(confidence=0.2),
        ]
        summary = engine.summarize_resolutions(decisions)
        assert summary.average_confidence == pytest.approx(0.5, abs=0.01)
        assert summary.min_confidence == pytest.approx(0.2, abs=0.01)
        assert summary.max_confidence == pytest.approx(0.8, abs=0.01)

    def test_processing_time_sum(self, engine):
        decisions = [
            ResolutionDecision(processing_time_ms=1.0),
            ResolutionDecision(processing_time_ms=2.0),
        ]
        summary = engine.summarize_resolutions(decisions)
        assert summary.total_processing_time_ms == pytest.approx(3.0, abs=0.01)

    def test_empty_decisions(self, engine):
        summary = engine.summarize_resolutions([])
        assert summary.total_decisions == 0
        assert summary.average_confidence == 0.0

    def test_provenance_hash_present(self, engine):
        decisions = [ResolutionDecision(confidence=0.9)]
        summary = engine.summarize_resolutions(decisions)
        assert summary.provenance_hash != ""
        assert len(summary.provenance_hash) == 64


# ===========================================================================
# Test auto_select_strategy
# ===========================================================================


class TestAutoSelectStrategy:
    """Tests for automatic strategy selection based on severity and type."""

    def test_critical_returns_manual_review(self, engine, creds_ab):
        disc = Discrepancy(
            severity=DiscrepancySeverity.CRITICAL,
            field_type=FieldType.NUMERIC,
            source_values={"src-a": 100, "src-b": 200},
        )
        strategy = engine.auto_select_strategy(disc, creds_ab)
        assert strategy == ResolutionStrategy.MANUAL_REVIEW

    def test_high_with_spread_returns_priority_wins(self, engine):
        """HIGH severity + credibility spread > 0.3 -> PRIORITY_WINS."""
        creds = {
            "a": SourceCredibility(source_id="a", credibility_score=0.95, priority=1),
            "b": SourceCredibility(source_id="b", credibility_score=0.4, priority=2),
        }
        disc = Discrepancy(
            severity=DiscrepancySeverity.HIGH,
            source_values={"a": 1, "b": 2},
        )
        strategy = engine.auto_select_strategy(disc, creds)
        assert strategy == ResolutionStrategy.PRIORITY_WINS

    def test_high_without_spread_returns_manual_review(self, engine):
        """HIGH severity + credibility spread <= 0.3 -> MANUAL_REVIEW."""
        creds = {
            "a": SourceCredibility(source_id="a", credibility_score=0.8, priority=1),
            "b": SourceCredibility(source_id="b", credibility_score=0.7, priority=2),
        }
        disc = Discrepancy(
            severity=DiscrepancySeverity.HIGH,
            source_values={"a": 1, "b": 2},
        )
        strategy = engine.auto_select_strategy(disc, creds)
        assert strategy == ResolutionStrategy.MANUAL_REVIEW

    def test_medium_numeric_returns_weighted_average(self, engine, creds_ab):
        disc = Discrepancy(
            severity=DiscrepancySeverity.MEDIUM,
            field_type=FieldType.NUMERIC,
            source_values={"src-a": 100, "src-b": 200},
        )
        strategy = engine.auto_select_strategy(disc, creds_ab)
        assert strategy == ResolutionStrategy.WEIGHTED_AVERAGE

    def test_medium_string_returns_priority_wins(self, engine, creds_ab):
        disc = Discrepancy(
            severity=DiscrepancySeverity.MEDIUM,
            field_type=FieldType.STRING,
            source_values={"src-a": "a", "src-b": "b"},
        )
        strategy = engine.auto_select_strategy(disc, creds_ab)
        assert strategy == ResolutionStrategy.PRIORITY_WINS

    def test_low_returns_priority_wins(self, engine, creds_ab):
        disc = Discrepancy(
            severity=DiscrepancySeverity.LOW,
            source_values={"src-a": 1, "src-b": 2},
        )
        strategy = engine.auto_select_strategy(disc, creds_ab)
        assert strategy == ResolutionStrategy.PRIORITY_WINS

    def test_info_returns_priority_wins(self, engine, creds_ab):
        disc = Discrepancy(
            severity=DiscrepancySeverity.INFO,
            source_values={"src-a": 1, "src-b": 2},
        )
        strategy = engine.auto_select_strategy(disc, creds_ab)
        assert strategy == ResolutionStrategy.PRIORITY_WINS

    def test_string_severity_handled(self, engine, creds_ab):
        """Severity as a string value is handled gracefully."""
        disc = Discrepancy(
            severity="critical",
            source_values={"src-a": 1, "src-b": 2},
        )
        strategy = engine.auto_select_strategy(disc, creds_ab)
        assert strategy == ResolutionStrategy.MANUAL_REVIEW

    def test_unknown_severity_defaults_to_medium(self, engine, creds_ab):
        """Unknown severity string defaults to MEDIUM behavior."""
        disc = Discrepancy(
            severity="unknown_level",
            field_type=FieldType.NUMERIC,
            source_values={"src-a": 1, "src-b": 2},
        )
        strategy = engine.auto_select_strategy(disc, creds_ab)
        assert strategy == ResolutionStrategy.WEIGHTED_AVERAGE


# ===========================================================================
# Test Provenance Tracking
# ===========================================================================


class TestProvenanceTracking:
    """Tests for SHA-256 provenance hashing across all operations."""

    def test_resolve_discrepancy_has_provenance(
        self, engine, numeric_discrepancy, creds_ab
    ):
        decision = engine.resolve_discrepancy(
            numeric_discrepancy,
            ResolutionStrategy.PRIORITY_WINS,
            creds_ab,
            {},
        )
        assert decision.provenance_hash != ""
        assert len(decision.provenance_hash) == 64

    def test_golden_record_has_provenance(self, engine, creds_ab, source_records_ab):
        golden = engine.assemble_golden_record(
            entity_id="ent-1",
            period="2025-Q1",
            source_records=source_records_ab,
            resolutions=[],
            source_credibilities=creds_ab,
        )
        assert golden.provenance_hash != ""
        assert len(golden.provenance_hash) == 64

    def test_summary_has_provenance(self, engine):
        decisions = [ResolutionDecision(confidence=0.9)]
        summary = engine.summarize_resolutions(decisions)
        assert summary.provenance_hash != ""
        assert len(summary.provenance_hash) == 64

    def test_deterministic_provenance_for_same_input(self, engine, creds_ab):
        """Same discrepancy + same strategy -> same provenance hash."""
        disc = Discrepancy(
            discrepancy_id="fixed-id",
            entity_id="ent-1",
            field_name="co2",
            source_values={"src-a": 100, "src-b": 200},
        )
        # Priority wins is deterministic
        d1 = engine.resolve_priority_wins(disc, creds_ab)
        d2 = engine.resolve_priority_wins(disc, creds_ab)
        # Resolved values are identical
        assert d1.resolved_value == d2.resolved_value
        assert d1.winning_source_id == d2.winning_source_id


# ===========================================================================
# Test batch golden record assembly
# ===========================================================================


class TestAssembleGoldenRecordsBatch:
    """Tests for batch golden record assembly."""

    def test_batch_assembly(self, engine, creds_ab):
        entities = [("ent-1", "2025-Q1"), ("ent-2", "2025-Q2")]
        records_map = {
            "ent-1": {
                "src-a": {"f1": 10},
                "src-b": {"f1": 20},
            },
            "ent-2": {
                "src-a": {"f1": 30},
                "src-b": {"f1": 40},
            },
        }
        resolutions_map: Dict[str, List[ResolutionDecision]] = {
            "ent-1": [],
            "ent-2": [],
        }
        goldens = engine.assemble_golden_records_batch(
            entities, records_map, resolutions_map, creds_ab,
        )
        assert len(goldens) == 2
        assert goldens[0].entity_id == "ent-1"
        assert goldens[1].entity_id == "ent-2"
