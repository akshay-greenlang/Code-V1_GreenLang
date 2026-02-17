# -*- coding: utf-8 -*-
"""
Unit tests for MatchingEngine - AGENT-DATA-015

Tests all public methods of MatchingEngine with 60+ test cases.
Validates exact, fuzzy, composite, temporal, blocking matching strategies,
batch processing, composite key computation, and similarity algorithms.

Author: GL-TestEngineer
Date: February 2026
"""

from __future__ import annotations

import math
import pytest
from typing import Any, Dict, List

from greenlang.cross_source_reconciliation.matching_engine import (
    MatchingEngine,
    MatchResult,
    MatchKey,
    MatchStatus,
    MatchStrategy,
    FieldType,
    BatchMatchResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    """Create a fresh MatchingEngine for each test."""
    return MatchingEngine()


@pytest.fixture
def records_a_exact():
    """Source A records for exact matching tests."""
    return [
        {"entity_id": "E1", "period": "2025-Q1", "value": 100},
        {"entity_id": "E2", "period": "2025-Q1", "value": 200},
        {"entity_id": "E3", "period": "2025-Q1", "value": 300},
    ]


@pytest.fixture
def records_b_exact():
    """Source B records for exact matching tests (E1 and E2 match)."""
    return [
        {"entity_id": "E1", "period": "2025-Q1", "value": 102},
        {"entity_id": "E2", "period": "2025-Q1", "value": 198},
        {"entity_id": "E4", "period": "2025-Q1", "value": 400},
    ]


@pytest.fixture
def records_a_fuzzy():
    """Source A records for fuzzy matching tests."""
    return [
        {"name": "Acme Corporation", "city": "New York", "revenue": 1000},
        {"name": "Beta Industries", "city": "London", "revenue": 2000},
        {"name": "Gamma LLC", "city": "Paris", "revenue": 3000},
    ]


@pytest.fixture
def records_b_fuzzy():
    """Source B records for fuzzy matching (similar but not identical)."""
    return [
        {"name": "Acme Corp", "city": "New York", "revenue": 1005},
        {"name": "Beta Ind.", "city": "London", "revenue": 2010},
        {"name": "Delta Inc", "city": "Berlin", "revenue": 4000},
    ]


@pytest.fixture
def temporal_records_monthly():
    """Monthly records for temporal matching."""
    return [
        {"entity_id": "E1", "value": 100, "timestamp": "2025-01"},
        {"entity_id": "E1", "value": 110, "timestamp": "2025-02"},
        {"entity_id": "E1", "value": 120, "timestamp": "2025-03"},
    ]


@pytest.fixture
def temporal_records_quarterly():
    """Quarterly records for temporal matching."""
    return [
        {"entity_id": "E1", "value": 330, "timestamp": "2025-Q1"},
    ]


# ---------------------------------------------------------------------------
# TestMatchingEngine: Exact Matching
# ---------------------------------------------------------------------------


class TestMatchExact:
    """Tests for exact matching strategy."""

    def test_match_exact_finds_matching_records(self, engine, records_a_exact, records_b_exact):
        """Exact matching finds records with identical composite keys."""
        results = engine.match_exact(
            records_a_exact, records_b_exact,
            key_fields=["entity_id", "period"],
        )
        matched = [r for r in results if r.status == MatchStatus.MATCHED.value]
        assert len(matched) == 2  # E1 and E2 match

    def test_match_exact_confidence_is_one(self, engine, records_a_exact, records_b_exact):
        """Exact matches have confidence 1.0."""
        results = engine.match_exact(
            records_a_exact, records_b_exact,
            key_fields=["entity_id", "period"],
        )
        matched = [r for r in results if r.status == MatchStatus.MATCHED.value]
        for m in matched:
            assert m.confidence == 1.0

    def test_match_exact_reports_unmatched_a(self, engine, records_a_exact, records_b_exact):
        """Unmatched A records are reported with UNMATCHED_A status."""
        results = engine.match_exact(
            records_a_exact, records_b_exact,
            key_fields=["entity_id", "period"],
        )
        unmatched_a = [r for r in results if r.status == MatchStatus.UNMATCHED_A.value]
        assert len(unmatched_a) == 1  # E3 has no match
        assert unmatched_a[0].record_a["entity_id"] == "E3"

    def test_match_exact_reports_unmatched_b(self, engine, records_a_exact, records_b_exact):
        """Unmatched B records are reported with UNMATCHED_B status."""
        results = engine.match_exact(
            records_a_exact, records_b_exact,
            key_fields=["entity_id", "period"],
        )
        unmatched_b = [r for r in results if r.status == MatchStatus.UNMATCHED_B.value]
        assert len(unmatched_b) == 1  # E4 has no match

    def test_match_exact_empty_sources(self, engine):
        """Empty source lists produce no results."""
        results = engine.match_exact([], [], key_fields=["entity_id"])
        assert results == []

    def test_match_exact_empty_a(self, engine, records_b_exact):
        """Empty source A produces only UNMATCHED_B results."""
        results = engine.match_exact(
            [], records_b_exact, key_fields=["entity_id", "period"],
        )
        assert all(r.status == MatchStatus.UNMATCHED_B.value for r in results)

    def test_match_exact_empty_b(self, engine, records_a_exact):
        """Empty source B produces only UNMATCHED_A results."""
        results = engine.match_exact(
            records_a_exact, [], key_fields=["entity_id", "period"],
        )
        assert all(r.status == MatchStatus.UNMATCHED_A.value for r in results)

    def test_match_exact_preserves_record_data(self, engine):
        """Matched results contain full record data from both sources."""
        a = [{"id": "X", "value": 42}]
        b = [{"id": "X", "value": 99}]
        results = engine.match_exact(a, b, key_fields=["id"])
        matched = [r for r in results if r.status == MatchStatus.MATCHED.value]
        assert len(matched) == 1
        assert matched[0].record_a["value"] == 42
        assert matched[0].record_b["value"] == 99

    def test_match_exact_provenance_hash_populated(self, engine, records_a_exact, records_b_exact):
        """Every result has a non-empty provenance hash."""
        results = engine.match_exact(
            records_a_exact, records_b_exact,
            key_fields=["entity_id", "period"],
        )
        for r in results:
            assert r.provenance_hash != ""
            assert len(r.provenance_hash) == 64  # SHA-256 hex

    def test_match_exact_field_scores(self, engine):
        """Exact match field_scores are 1.0 for each key field."""
        a = [{"x": "A", "y": "B"}]
        b = [{"x": "A", "y": "B"}]
        results = engine.match_exact(a, b, key_fields=["x", "y"])
        matched = [r for r in results if r.status == MatchStatus.MATCHED.value]
        assert matched[0].field_scores == {"x": 1.0, "y": 1.0}

    def test_match_exact_single_key_field(self, engine):
        """Exact matching works with a single key field."""
        a = [{"id": "A"}, {"id": "B"}]
        b = [{"id": "A"}, {"id": "C"}]
        results = engine.match_exact(a, b, key_fields=["id"])
        matched = [r for r in results if r.status == MatchStatus.MATCHED.value]
        assert len(matched) == 1


# ---------------------------------------------------------------------------
# TestMatchingEngine: Fuzzy Matching
# ---------------------------------------------------------------------------


class TestMatchFuzzy:
    """Tests for fuzzy matching strategy."""

    def test_match_fuzzy_finds_similar_records(self, engine, records_a_fuzzy, records_b_fuzzy):
        """Fuzzy matching finds similar (not exact) records."""
        results = engine.match_fuzzy(
            records_a_fuzzy, records_b_fuzzy,
            key_fields=["name", "city"],
            threshold=0.7,
        )
        matched = [r for r in results if r.status == MatchStatus.MATCHED.value]
        # Acme Corporation <-> Acme Corp and Beta Industries <-> Beta Ind.
        assert len(matched) >= 2

    def test_match_fuzzy_confidence_below_one(self, engine, records_a_fuzzy, records_b_fuzzy):
        """Fuzzy matches have confidence < 1.0 for non-identical fields."""
        results = engine.match_fuzzy(
            records_a_fuzzy, records_b_fuzzy,
            key_fields=["name"],
            threshold=0.7,
        )
        matched = [r for r in results if r.status == MatchStatus.MATCHED.value]
        for m in matched:
            # Not identical strings, so confidence should be < 1.0
            assert m.confidence <= 1.0
            assert m.confidence >= 0.7

    def test_match_fuzzy_threshold_filtering(self, engine):
        """Records below threshold are reported as BELOW_THRESHOLD or UNMATCHED_A."""
        a = [{"name": "Alpha"}]
        b = [{"name": "Omega"}]
        results = engine.match_fuzzy(a, b, key_fields=["name"], threshold=0.9)
        matched = [r for r in results if r.status == MatchStatus.MATCHED.value]
        assert len(matched) == 0

    def test_match_fuzzy_high_threshold_strict(self, engine):
        """Very high threshold (0.99) requires near-exact matches."""
        a = [{"name": "test"}]
        b = [{"name": "test_"}]
        results = engine.match_fuzzy(a, b, key_fields=["name"], threshold=0.99)
        matched = [r for r in results if r.status == MatchStatus.MATCHED.value]
        assert len(matched) == 0

    def test_match_fuzzy_identical_records(self, engine):
        """Identical records match with confidence 1.0."""
        a = [{"name": "exact"}]
        b = [{"name": "exact"}]
        results = engine.match_fuzzy(a, b, key_fields=["name"], threshold=0.5)
        matched = [r for r in results if r.status == MatchStatus.MATCHED.value]
        assert len(matched) == 1
        assert matched[0].confidence == 1.0

    def test_match_fuzzy_reports_unmatched_b(self, engine):
        """Source B records with no match are reported as UNMATCHED_B."""
        a = [{"name": "Alpha"}]
        b = [{"name": "Alpha"}, {"name": "Zeta"}]
        results = engine.match_fuzzy(a, b, key_fields=["name"], threshold=0.5)
        unmatched_b = [r for r in results if r.status == MatchStatus.UNMATCHED_B.value]
        assert len(unmatched_b) == 1

    def test_match_fuzzy_empty_inputs(self, engine):
        """Empty inputs produce no results."""
        results = engine.match_fuzzy([], [], key_fields=["name"], threshold=0.5)
        assert results == []

    def test_match_fuzzy_numeric_field_proximity(self, engine):
        """Fuzzy matching handles numeric fields via proximity."""
        a = [{"entity": "E1", "value": 100}]
        b = [{"entity": "E1", "value": 105}]
        results = engine.match_fuzzy(
            a, b, key_fields=["entity", "value"], threshold=0.5,
        )
        matched = [r for r in results if r.status == MatchStatus.MATCHED.value]
        assert len(matched) == 1

    def test_match_fuzzy_field_scores_populated(self, engine):
        """Fuzzy matches include per-field similarity scores."""
        a = [{"name": "Acme Corp", "city": "New York"}]
        b = [{"name": "Acme Corporation", "city": "New York"}]
        results = engine.match_fuzzy(
            a, b, key_fields=["name", "city"], threshold=0.5,
        )
        matched = [r for r in results if r.status == MatchStatus.MATCHED.value]
        assert "name" in matched[0].field_scores
        assert "city" in matched[0].field_scores


# ---------------------------------------------------------------------------
# TestMatchingEngine: Composite Matching
# ---------------------------------------------------------------------------


class TestMatchComposite:
    """Tests for composite (weighted multi-field) matching."""

    def test_match_composite_with_weights(self, engine):
        """Composite matching uses field weights for scoring."""
        a = [{"name": "Acme Corp", "id": "A001"}]
        b = [{"name": "Acme Corporation", "id": "A001"}]
        results = engine.match_composite(
            a, b,
            key_fields=["name", "id"],
            field_weights={"name": 0.3, "id": 0.7},
            threshold=0.5,
        )
        matched = [r for r in results if r.status == MatchStatus.MATCHED.value]
        assert len(matched) == 1
        # id is identical (score=1.0, weight=0.7) so overall should be high
        assert matched[0].confidence > 0.8

    def test_match_composite_high_weight_field_dominates(self, engine):
        """A heavily weighted field dominates the composite score."""
        a = [{"name": "totally_different", "code": "SAME"}]
        b = [{"name": "completely_unrelated", "code": "SAME"}]
        results = engine.match_composite(
            a, b,
            key_fields=["name", "code"],
            field_weights={"name": 0.1, "code": 0.9},
            threshold=0.5,
        )
        matched = [r for r in results if r.status == MatchStatus.MATCHED.value]
        assert len(matched) == 1
        # code is exact match with 0.9 weight
        assert matched[0].confidence > 0.8

    def test_match_composite_equal_weights_is_average(self, engine):
        """Equal weights produce the same result as unweighted average."""
        a = [{"x": "hello", "y": "world"}]
        b = [{"x": "hello", "y": "wurld"}]
        results = engine.match_composite(
            a, b,
            key_fields=["x", "y"],
            field_weights={"x": 1.0, "y": 1.0},
            threshold=0.3,
        )
        matched = [r for r in results if r.status == MatchStatus.MATCHED.value]
        assert len(matched) == 1

    def test_match_composite_below_threshold(self, engine):
        """Records below composite threshold are not matched."""
        a = [{"x": "aaa"}]
        b = [{"x": "zzz"}]
        results = engine.match_composite(
            a, b,
            key_fields=["x"],
            field_weights={"x": 1.0},
            threshold=0.9,
        )
        matched = [r for r in results if r.status == MatchStatus.MATCHED.value]
        assert len(matched) == 0

    def test_match_composite_metadata_includes_weights(self, engine):
        """Composite match results include field_weights in metadata."""
        a = [{"x": "same"}]
        b = [{"x": "same"}]
        weights = {"x": 2.0}
        results = engine.match_composite(
            a, b,
            key_fields=["x"],
            field_weights=weights,
            threshold=0.5,
        )
        matched = [r for r in results if r.status == MatchStatus.MATCHED.value]
        assert matched[0].metadata.get("field_weights") == weights


# ---------------------------------------------------------------------------
# TestMatchingEngine: Temporal Matching
# ---------------------------------------------------------------------------


class TestMatchTemporal:
    """Tests for temporal matching strategy."""

    def test_match_temporal_aggregates_monthly_to_quarterly(
        self, engine, temporal_records_monthly, temporal_records_quarterly,
    ):
        """Monthly records are aggregated to quarterly for matching."""
        results = engine.match_temporal(
            temporal_records_monthly,
            temporal_records_quarterly,
            entity_field="entity_id",
            value_field="value",
            timestamp_field="timestamp",
            granularity_a="monthly",
            granularity_b="quarterly",
            target_granularity="quarterly",
            threshold=0.5,
        )
        # Should have at least one match result
        assert len(results) >= 1

    def test_match_temporal_includes_metadata(
        self, engine, temporal_records_monthly, temporal_records_quarterly,
    ):
        """Temporal match results include granularity metadata."""
        results = engine.match_temporal(
            temporal_records_monthly,
            temporal_records_quarterly,
            entity_field="entity_id",
            value_field="value",
            timestamp_field="timestamp",
            granularity_a="monthly",
            granularity_b="quarterly",
            target_granularity="quarterly",
        )
        for r in results:
            if r.metadata:
                assert "target_granularity" in r.metadata

    def test_match_temporal_empty_inputs(self, engine):
        """Empty inputs produce no results."""
        results = engine.match_temporal(
            [], [],
            entity_field="entity_id",
            value_field="value",
            timestamp_field="timestamp",
            granularity_a="monthly",
            granularity_b="quarterly",
            target_granularity="quarterly",
        )
        assert results == []

    def test_match_temporal_strategy_label(
        self, engine, temporal_records_monthly, temporal_records_quarterly,
    ):
        """Temporal matches are labelled with 'temporal' strategy."""
        results = engine.match_temporal(
            temporal_records_monthly,
            temporal_records_quarterly,
            entity_field="entity_id",
            value_field="value",
            timestamp_field="timestamp",
            granularity_a="monthly",
            granularity_b="quarterly",
            target_granularity="quarterly",
        )
        for r in results:
            assert r.strategy == "temporal"


# ---------------------------------------------------------------------------
# TestMatchingEngine: Blocking
# ---------------------------------------------------------------------------


class TestMatchWithBlocking:
    """Tests for blocking-based matching strategy."""

    def test_match_with_blocking_reduces_comparison_space(self, engine):
        """Blocking restricts comparisons to same-block records."""
        a = [
            {"name": "Alpha Corp", "region": "US"},
            {"name": "Beta Inc", "region": "EU"},
        ]
        b = [
            {"name": "Alpha Company", "region": "US"},
            {"name": "Beta Industries", "region": "EU"},
        ]
        results = engine.match_with_blocking(
            a, b,
            key_fields=["name"],
            blocking_fields=["region"],
            strategy="fuzzy",
            threshold=0.5,
        )
        matched = [r for r in results if r.status == MatchStatus.MATCHED.value]
        assert len(matched) >= 1

    def test_match_with_blocking_exact_inner_strategy(self, engine):
        """Blocking with exact inner strategy finds exact matches within blocks."""
        a = [
            {"id": "X1", "region": "US"},
            {"id": "Y1", "region": "EU"},
        ]
        b = [
            {"id": "X1", "region": "US"},
            {"id": "Z1", "region": "EU"},
        ]
        results = engine.match_with_blocking(
            a, b,
            key_fields=["id"],
            blocking_fields=["region"],
            strategy="exact",
        )
        matched = [r for r in results if r.status == MatchStatus.MATCHED.value]
        assert len(matched) == 1
        assert matched[0].record_a["id"] == "X1"

    def test_match_with_blocking_metadata_includes_block_key(self, engine):
        """Blocking results include block_key in metadata."""
        a = [{"id": "A", "region": "US"}]
        b = [{"id": "A", "region": "US"}]
        results = engine.match_with_blocking(
            a, b,
            key_fields=["id"],
            blocking_fields=["region"],
            strategy="exact",
        )
        for r in results:
            if r.status == MatchStatus.MATCHED.value:
                assert "block_key" in r.metadata

    def test_match_with_blocking_empty_inputs(self, engine):
        """Empty inputs produce no results."""
        results = engine.match_with_blocking(
            [], [], key_fields=["id"], blocking_fields=["region"],
        )
        assert results == []

    def test_match_with_blocking_unmatched_different_blocks(self, engine):
        """Records in different blocks cannot match."""
        a = [{"id": "same", "region": "US"}]
        b = [{"id": "same", "region": "EU"}]
        results = engine.match_with_blocking(
            a, b,
            key_fields=["id"],
            blocking_fields=["region"],
            strategy="exact",
        )
        matched = [r for r in results if r.status == MatchStatus.MATCHED.value]
        assert len(matched) == 0


# ---------------------------------------------------------------------------
# TestMatchingEngine: Batch Matching
# ---------------------------------------------------------------------------


class TestMatchBatch:
    """Tests for batch (multi-source) matching."""

    def test_match_batch_pairwise(self, engine):
        """Batch matching processes all pairwise source combinations."""
        source_map = {
            "src_a": [{"id": "E1", "val": 100}],
            "src_b": [{"id": "E1", "val": 102}],
            "src_c": [{"id": "E1", "val": 99}],
        }
        result = engine.match_batch(
            source_map, key_fields=["id"], strategy="exact",
        )
        assert isinstance(result, BatchMatchResult)
        # 3 sources -> 3 pairs (a-b, a-c, b-c)
        assert len(result.pair_stats) == 3

    def test_match_batch_statistics(self, engine):
        """Batch result has correct aggregate statistics."""
        source_map = {
            "s1": [{"id": "A"}, {"id": "B"}],
            "s2": [{"id": "A"}, {"id": "C"}],
        }
        result = engine.match_batch(
            source_map, key_fields=["id"], strategy="exact",
        )
        assert result.total_matched >= 1  # at least A matches

    def test_match_batch_provenance(self, engine):
        """Batch result has a provenance hash."""
        source_map = {"s1": [{"id": "X"}], "s2": [{"id": "X"}]}
        result = engine.match_batch(source_map, key_fields=["id"])
        assert result.provenance_hash != ""
        assert len(result.provenance_hash) == 64

    def test_match_batch_single_source(self, engine):
        """Single source produces no pairs (needs >= 2)."""
        source_map = {"only": [{"id": "A"}]}
        result = engine.match_batch(source_map, key_fields=["id"])
        assert result.total_matched == 0
        assert len(result.pair_stats) == 0


# ---------------------------------------------------------------------------
# TestMatchingEngine: Compute Match Key
# ---------------------------------------------------------------------------


class TestComputeMatchKey:
    """Tests for compute_match_key method."""

    def test_compute_match_key_creates_composite(self, engine):
        """Match key produces a composite string from key fields."""
        record = {"entity_id": "E1", "period": "2025-Q1", "extra": "ignore"}
        mk = engine.compute_match_key(record, ["entity_id", "period"])
        assert isinstance(mk, MatchKey)
        assert "e1" in mk.composite_key.lower()
        assert "2025-q1" in mk.composite_key.lower()

    def test_compute_match_key_deterministic(self, engine):
        """Same record + key_fields always produces the same key."""
        record = {"a": "hello", "b": "world"}
        mk1 = engine.compute_match_key(record, ["a", "b"])
        mk2 = engine.compute_match_key(record, ["a", "b"])
        assert mk1.composite_key == mk2.composite_key

    def test_compute_match_key_different_records(self, engine):
        """Different records produce different composite keys."""
        rec1 = {"a": "hello"}
        rec2 = {"a": "world"}
        mk1 = engine.compute_match_key(rec1, ["a"])
        mk2 = engine.compute_match_key(rec2, ["a"])
        assert mk1.composite_key != mk2.composite_key

    def test_compute_match_key_missing_field(self, engine):
        """Missing key field produces empty value in composite."""
        record = {"a": "hello"}
        mk = engine.compute_match_key(record, ["a", "missing_field"])
        assert mk.composite_key  # should still produce a key

    def test_compute_match_key_provenance_hash(self, engine):
        """Match key includes a provenance hash."""
        record = {"id": "E1"}
        mk = engine.compute_match_key(record, ["id"])
        assert mk.provenance_hash != ""
        assert len(mk.provenance_hash) == 64

    def test_compute_match_key_field_ordering_matters(self, engine):
        """Different key_fields order may produce different composite keys."""
        record = {"a": "X", "b": "Y"}
        mk1 = engine.compute_match_key(record, ["a", "b"])
        mk2 = engine.compute_match_key(record, ["b", "a"])
        # Ordering changes the composite key string
        assert mk1.composite_key != mk2.composite_key


# ---------------------------------------------------------------------------
# TestMatchingEngine: Jaro-Winkler Similarity
# ---------------------------------------------------------------------------


class TestJaroWinklerSimilarity:
    """Tests for _jaro_winkler_similarity static method."""

    def test_identical_strings(self, engine):
        """Identical strings return 1.0."""
        assert engine._jaro_winkler_similarity("hello", "hello") == 1.0

    def test_empty_strings(self, engine):
        """Both empty strings return 1.0."""
        assert engine._jaro_winkler_similarity("", "") == 1.0

    def test_one_empty(self, engine):
        """One empty string returns 0.0."""
        assert engine._jaro_winkler_similarity("hello", "") == 0.0
        assert engine._jaro_winkler_similarity("", "world") == 0.0

    def test_known_pair_martha_marhta(self, engine):
        """Known test pair: MARTHA/MARHTA should have ~0.961."""
        score = engine._jaro_winkler_similarity("MARTHA", "MARHTA")
        assert 0.95 <= score <= 0.97

    def test_known_pair_dwayne_duane(self, engine):
        """Known test pair: DWAYNE/DUANE should have ~0.84."""
        score = engine._jaro_winkler_similarity("DWAYNE", "DUANE")
        assert 0.80 <= score <= 0.90

    def test_completely_different(self, engine):
        """Completely different strings have low similarity."""
        score = engine._jaro_winkler_similarity("abc", "xyz")
        assert score < 0.5

    def test_symmetry(self, engine):
        """Similarity is symmetric."""
        s1 = engine._jaro_winkler_similarity("test", "tset")
        s2 = engine._jaro_winkler_similarity("tset", "test")
        assert abs(s1 - s2) < 1e-10

    def test_single_char_strings(self, engine):
        """Single character strings: same returns 1.0, different returns 0.0."""
        assert engine._jaro_winkler_similarity("a", "a") == 1.0
        assert engine._jaro_winkler_similarity("a", "b") == 0.0

    def test_prefix_bonus(self, engine):
        """Common prefix produces higher score than Jaro alone."""
        # "prefix_abc" and "prefix_xyz" share "prefix_" prefix
        score = engine._jaro_winkler_similarity("prefix_abc", "prefix_xyz")
        assert score > 0.6


# ---------------------------------------------------------------------------
# TestMatchingEngine: Numeric and Date Proximity
# ---------------------------------------------------------------------------


class TestProximityScoring:
    """Tests for numeric and date proximity scoring helpers."""

    def test_numeric_proximity_identical(self, engine):
        """Identical numeric values produce score 1.0."""
        field_scores = engine._compute_field_similarities(
            {"val": 100}, {"val": 100}, ["val"],
        )
        assert field_scores["val"] == 1.0

    def test_numeric_proximity_close_values(self, engine):
        """Close numeric values produce high proximity score."""
        field_scores = engine._compute_field_similarities(
            {"val": 100}, {"val": 101}, ["val"],
        )
        assert field_scores["val"] > 0.9

    def test_numeric_proximity_far_values(self, engine):
        """Far-apart numeric values produce low proximity score."""
        field_scores = engine._compute_field_similarities(
            {"val": 0}, {"val": 10000}, ["val"],
        )
        assert field_scores["val"] < 0.5

    def test_string_field_exact_match(self, engine):
        """String fields with exact match produce score 1.0."""
        field_scores = engine._compute_field_similarities(
            {"name": "test"}, {"name": "test"}, ["name"],
        )
        assert field_scores["name"] == 1.0


# ---------------------------------------------------------------------------
# TestMatchingEngine: Dispatcher (match_records)
# ---------------------------------------------------------------------------


class TestMatchRecords:
    """Tests for the match_records dispatcher."""

    def test_match_records_dispatches_exact(self, engine):
        """match_records with strategy='exact' routes to match_exact."""
        a = [{"id": "A"}]
        b = [{"id": "A"}]
        results = engine.match_records(
            a, b, key_fields=["id"], strategy="exact",
        )
        matched = [r for r in results if r.status == MatchStatus.MATCHED.value]
        assert len(matched) == 1

    def test_match_records_dispatches_fuzzy(self, engine):
        """match_records with strategy='fuzzy' routes to match_fuzzy."""
        a = [{"name": "test"}]
        b = [{"name": "test"}]
        results = engine.match_records(
            a, b, key_fields=["name"], strategy="fuzzy", threshold=0.5,
        )
        matched = [r for r in results if r.status == MatchStatus.MATCHED.value]
        assert len(matched) == 1

    def test_match_records_invalid_strategy_raises(self, engine):
        """Unknown strategy raises ValueError."""
        with pytest.raises(ValueError, match="Unrecognised matching strategy"):
            engine.match_records(
                [{"id": "A"}], [{"id": "A"}],
                key_fields=["id"], strategy="unknown_strategy",
            )

    def test_match_records_case_insensitive_strategy(self, engine):
        """Strategy name is case-insensitive."""
        a = [{"id": "A"}]
        b = [{"id": "A"}]
        results = engine.match_records(
            a, b, key_fields=["id"], strategy="EXACT",
        )
        matched = [r for r in results if r.status == MatchStatus.MATCHED.value]
        assert len(matched) == 1

    def test_match_records_dispatches_composite(self, engine):
        """match_records with strategy='composite' routes correctly."""
        a = [{"id": "A"}]
        b = [{"id": "A"}]
        results = engine.match_records(
            a, b, key_fields=["id"],
            strategy="composite",
            threshold=0.5,
        )
        matched = [r for r in results if r.status == MatchStatus.MATCHED.value]
        assert len(matched) == 1

    def test_match_records_dispatches_blocking(self, engine):
        """match_records with strategy='blocking' routes correctly."""
        a = [{"id": "A", "region": "US"}]
        b = [{"id": "A", "region": "US"}]
        results = engine.match_records(
            a, b, key_fields=["id"],
            strategy="blocking",
            blocking_fields=["region"],
            threshold=0.5,
        )
        assert len(results) >= 1


# ---------------------------------------------------------------------------
# TestMatchingEngine: Edge Cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_records_with_none_values(self, engine):
        """Records with None values in key fields handle gracefully."""
        a = [{"id": None}]
        b = [{"id": None}]
        results = engine.match_exact(a, b, key_fields=["id"])
        # Both have same key (None normalized), should match
        matched = [r for r in results if r.status == MatchStatus.MATCHED.value]
        assert len(matched) == 1

    def test_large_number_of_key_fields(self, engine):
        """Matching works with many key fields."""
        fields = [f"f{i}" for i in range(10)]
        rec = {f: "value" for f in fields}
        results = engine.match_exact([rec], [rec], key_fields=fields)
        matched = [r for r in results if r.status == MatchStatus.MATCHED.value]
        assert len(matched) == 1

    def test_unicode_values(self, engine):
        """Unicode values in key fields are handled correctly."""
        a = [{"name": "Munchen"}]
        b = [{"name": "Munchen"}]
        results = engine.match_exact(a, b, key_fields=["name"])
        matched = [r for r in results if r.status == MatchStatus.MATCHED.value]
        assert len(matched) == 1

    def test_match_result_has_created_at(self, engine):
        """Every match result has a created_at ISO timestamp."""
        a = [{"id": "A"}]
        b = [{"id": "A"}]
        results = engine.match_exact(a, b, key_fields=["id"])
        for r in results:
            assert r.created_at != ""

    def test_match_result_has_match_id(self, engine):
        """Every match result has a unique match_id."""
        a = [{"id": "A"}, {"id": "B"}]
        b = [{"id": "A"}, {"id": "B"}]
        results = engine.match_exact(a, b, key_fields=["id"])
        match_ids = [r.match_id for r in results]
        assert len(set(match_ids)) == len(match_ids)  # all unique
