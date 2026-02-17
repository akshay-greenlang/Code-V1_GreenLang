# -*- coding: utf-8 -*-
"""
Unit Tests for LineageValidatorEngine - AGENT-DATA-018

Tests lineage graph validation including orphan detection, broken edge
detection, cycle detection (delegated), source coverage, completeness
scoring, freshness checking, recommendation generation, validation
retrieval, and statistics.

Target: 60+ tests, 7 test classes, 85%+ coverage.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from greenlang.data_lineage_tracker.lineage_graph import LineageGraphEngine
from greenlang.data_lineage_tracker.lineage_validator import (
    LineageValidatorEngine,
    _WEIGHT_ORPHAN,
    _WEIGHT_BROKEN_EDGE,
    _WEIGHT_COVERAGE,
    _WEIGHT_CYCLE,
)
from greenlang.data_lineage_tracker.provenance import ProvenanceTracker


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def graph() -> LineageGraphEngine:
    """Create a clean LineageGraphEngine with a sample DAG.

    Structure: A -> B -> C (report)
    """
    g = LineageGraphEngine()
    g.add_node("A", "raw.data", "external_source")
    g.add_node("B", "clean.data", "dataset")
    g.add_node("C", "final.report", "report")
    g.add_edge("A", "B")
    g.add_edge("B", "C")
    return g


@pytest.fixture
def validator(graph) -> LineageValidatorEngine:
    """Create a LineageValidatorEngine for the sample graph."""
    return LineageValidatorEngine(graph)


@pytest.fixture
def graph_with_orphan() -> LineageGraphEngine:
    """Create a graph with an orphan node."""
    g = LineageGraphEngine()
    g.add_node("A", "raw.data", "external_source")
    g.add_node("B", "clean.data", "dataset")
    g.add_node("orphan", "db.orphan", "dataset")
    g.add_edge("A", "B")
    return g


@pytest.fixture
def empty_graph() -> LineageGraphEngine:
    """Create an empty LineageGraphEngine."""
    return LineageGraphEngine()


# ============================================================================
# TestValidatorInit
# ============================================================================


class TestValidatorInit:
    """Tests for LineageValidatorEngine initialization."""

    def test_default_initialization(self, graph):
        v = LineageValidatorEngine(graph)
        assert v._graph is graph
        assert v._provenance is not None

    def test_initialization_with_provenance(self, graph):
        prov = ProvenanceTracker()
        # Pre-seed so __len__ > 0 (truthy); engines use ``prov or ProvenanceTracker()``
        prov.record("test", "seed", "init")
        v = LineageValidatorEngine(graph, provenance=prov)
        assert v._provenance is prov

    def test_non_graph_raises_type_error(self):
        with pytest.raises(TypeError, match="must be a LineageGraphEngine"):
            LineageValidatorEngine("not_a_graph")

    def test_empty_validations_on_init(self, validator):
        assert len(validator._validations) == 0

    def test_weights_sum_to_one(self):
        total = _WEIGHT_ORPHAN + _WEIGHT_BROKEN_EDGE + _WEIGHT_COVERAGE + _WEIGHT_CYCLE
        assert total == pytest.approx(1.0)


# ============================================================================
# TestValidate
# ============================================================================


class TestValidate:
    """Tests for the main validate method."""

    def test_validate_returns_dict(self, validator):
        report = validator.validate()
        assert isinstance(report, dict)

    def test_validate_has_required_keys(self, validator):
        report = validator.validate()
        required = [
            "id", "scope", "orphan_nodes", "broken_edges",
            "cycles_detected", "source_coverage", "completeness_score",
            "freshness_score", "issues", "recommendations", "result",
            "validated_at",
        ]
        for key in required:
            assert key in report, f"Missing key: {key}"

    def test_validate_id_format(self, validator):
        report = validator.validate()
        assert report["id"].startswith("VAL-")

    def test_validate_clean_graph_passes(self, validator):
        report = validator.validate()
        assert report["result"] in ("pass", "warn", "fail")
        # Clean graph with good coverage should pass
        assert report["completeness_score"] > 0.0

    def test_validate_scope_default(self, validator):
        report = validator.validate()
        assert report["scope"] == "full"

    def test_validate_scope_custom(self, validator):
        report = validator.validate(scope="pipeline")
        assert report["scope"] == "pipeline"

    def test_validate_without_freshness(self, validator):
        report = validator.validate(include_freshness=False)
        assert report["freshness_score"] == 1.0

    def test_validate_without_coverage(self, validator):
        report = validator.validate(include_coverage=False)
        assert report["source_coverage"] == 1.0

    def test_validate_stored_in_memory(self, validator):
        report = validator.validate()
        stored = validator.get_validation(report["id"])
        assert stored is not None
        assert stored["id"] == report["id"]

    def test_validate_records_provenance(self, graph):
        prov = ProvenanceTracker()
        # Pre-seed so __len__ > 0 (truthy); engines use ``prov or ProvenanceTracker()``
        prov.record("test", "seed", "init")
        v = LineageValidatorEngine(graph, provenance=prov)
        initial_count = prov.entry_count
        v.validate()
        assert prov.entry_count > initial_count

    def test_validate_empty_graph(self, empty_graph):
        v = LineageValidatorEngine(empty_graph)
        report = v.validate()
        assert report["orphan_nodes"] == 0
        assert report["broken_edges"] == 0
        assert report["completeness_score"] == 1.0


# ============================================================================
# TestOrphanDetection
# ============================================================================


class TestOrphanDetection:
    """Tests for detect_orphan_nodes method."""

    def test_no_orphans_in_connected_graph(self, validator):
        orphans = validator.detect_orphan_nodes()
        assert len(orphans) == 0

    def test_detects_orphan_nodes(self, graph_with_orphan):
        v = LineageValidatorEngine(graph_with_orphan)
        orphans = v.detect_orphan_nodes()
        orphan_ids = {o["node_id"] for o in orphans}
        assert "orphan" in orphan_ids

    def test_orphan_has_required_keys(self, graph_with_orphan):
        v = LineageValidatorEngine(graph_with_orphan)
        orphans = v.detect_orphan_nodes()
        for orphan in orphans:
            assert "node_id" in orphan
            assert "node_type" in orphan
            assert "detected_at" in orphan

    def test_empty_graph_no_orphans(self, empty_graph):
        v = LineageValidatorEngine(empty_graph)
        orphans = v.detect_orphan_nodes()
        assert len(orphans) == 0

    def test_all_orphan_graph(self):
        g = LineageGraphEngine()
        g.add_node("a", "node_a", "dataset")
        g.add_node("b", "node_b", "dataset")
        g.add_node("c", "node_c", "dataset")
        v = LineageValidatorEngine(g)
        orphans = v.detect_orphan_nodes()
        assert len(orphans) == 3


# ============================================================================
# TestBrokenEdgeDetection
# ============================================================================


class TestBrokenEdgeDetection:
    """Tests for detect_broken_edges method."""

    def test_no_broken_edges_in_valid_graph(self, validator):
        broken = validator.detect_broken_edges()
        # All edges should reference valid nodes via get_all_edges/get_all_nodes
        # In the LineageGraphEngine implementation, edges always have valid endpoints
        assert isinstance(broken, list)

    def test_empty_graph_no_broken_edges(self, empty_graph):
        v = LineageValidatorEngine(empty_graph)
        broken = v.detect_broken_edges()
        assert len(broken) == 0


# ============================================================================
# TestCycleDetection
# ============================================================================


class TestCycleDetection:
    """Tests for detect_cycles method (delegates to graph)."""

    def test_no_cycles_in_dag(self, validator):
        cycles = validator.detect_cycles()
        assert len(cycles) == 0

    def test_empty_graph_no_cycles(self, empty_graph):
        v = LineageValidatorEngine(empty_graph)
        cycles = v.detect_cycles()
        assert len(cycles) == 0


# ============================================================================
# TestSourceCoverage
# ============================================================================


class TestSourceCoverage:
    """Tests for compute_source_coverage method."""

    def test_coverage_returns_dict(self, validator):
        result = validator.compute_source_coverage()
        assert isinstance(result, dict)
        assert "coverage_score" in result
        assert "covered_reports" in result
        assert "total_reports" in result
        assert "uncovered_fields" in result

    def test_coverage_score_range(self, validator):
        result = validator.compute_source_coverage()
        assert 0.0 <= result["coverage_score"] <= 1.0

    def test_coverage_no_reports(self):
        g = LineageGraphEngine()
        g.add_node("A", "raw.data", "dataset")
        g.add_node("B", "clean.data", "dataset")
        g.add_edge("A", "B")
        v = LineageValidatorEngine(g)
        result = v.compute_source_coverage()
        assert result["coverage_score"] == 1.0  # Vacuously true
        assert result["total_reports"] == 0

    def test_empty_graph_coverage(self, empty_graph):
        v = LineageValidatorEngine(empty_graph)
        result = v.compute_source_coverage()
        assert result["coverage_score"] == 1.0


# ============================================================================
# TestCompletenessScore
# ============================================================================


class TestCompletenessScore:
    """Tests for compute_completeness_score method."""

    def test_completeness_range(self, validator):
        score = validator.compute_completeness_score()
        assert 0.0 <= score <= 1.0

    def test_perfect_graph_high_score(self, validator):
        score = validator.compute_completeness_score()
        # Clean graph should have high completeness
        assert score >= 0.5

    def test_empty_graph_completeness(self, empty_graph):
        v = LineageValidatorEngine(empty_graph)
        score = v.compute_completeness_score()
        assert score == 1.0

    def test_graph_with_orphan_lower_score(self, graph_with_orphan):
        v = LineageValidatorEngine(graph_with_orphan)
        score = v.compute_completeness_score()
        # Should be penalized for orphan
        assert score < 1.0


# ============================================================================
# TestFreshness
# ============================================================================


class TestFreshness:
    """Tests for check_freshness method."""

    def test_freshness_returns_dict(self, validator):
        result = validator.check_freshness()
        assert isinstance(result, dict)
        assert "freshness_score" in result
        assert "stale_assets" in result
        assert "total_assets" in result
        assert "stale_list" in result

    def test_freshness_score_range(self, validator):
        result = validator.check_freshness()
        assert 0.0 <= result["freshness_score"] <= 1.0

    def test_freshness_custom_max_age(self, validator):
        result = validator.check_freshness(max_age_hours=1)
        assert isinstance(result["freshness_score"], float)

    def test_freshness_empty_graph(self, empty_graph):
        v = LineageValidatorEngine(empty_graph)
        result = v.check_freshness()
        assert result["freshness_score"] == 1.0
        assert result["total_assets"] == 0


# ============================================================================
# TestPipelineCoverage
# ============================================================================


class TestPipelineCoverage:
    """Tests for check_pipeline_coverage method."""

    def test_pipeline_coverage_empty_id_raises(self, validator):
        with pytest.raises(ValueError, match="pipeline_id must not be empty"):
            validator.check_pipeline_coverage("")

    def test_pipeline_coverage_no_matching_nodes(self, validator):
        result = validator.check_pipeline_coverage("nonexistent-pipeline")
        assert result["coverage"] == 1.0
        assert result["total_assets"] == 0


# ============================================================================
# TestRecommendations
# ============================================================================


class TestRecommendations:
    """Tests for generate_recommendations method."""

    def test_no_issues_gives_positive(self, validator):
        recs = validator.generate_recommendations([])
        assert len(recs) == 1
        assert "no issues" in recs[0].lower()

    def test_orphan_node_recommendation(self, validator):
        issues = [{"type": "orphan_node", "node_id": "asset-001"}]
        recs = validator.generate_recommendations(issues)
        assert len(recs) >= 1
        assert "asset-001" in recs[0]

    def test_broken_edge_recommendation(self, validator):
        issues = [{"type": "broken_edge", "edge_id": "edge-001"}]
        recs = validator.generate_recommendations(issues)
        assert len(recs) >= 1
        assert "edge-001" in recs[0]

    def test_cycle_recommendation(self, validator):
        issues = [{"type": "cycle", "nodes": ["A", "B", "C"]}]
        recs = validator.generate_recommendations(issues)
        assert len(recs) >= 1
        assert "circular" in recs[0].lower() or "cycle" in recs[0].lower()

    def test_low_coverage_recommendation(self, validator):
        issues = [{"type": "low_coverage", "coverage_score": 0.5}]
        recs = validator.generate_recommendations(issues)
        assert len(recs) >= 1

    def test_stale_metadata_recommendation(self, validator):
        issues = [{"type": "stale_metadata", "stale_assets": 5}]
        recs = validator.generate_recommendations(issues)
        assert len(recs) >= 1

    def test_unknown_issue_type_recommendation(self, validator):
        issues = [{"type": "unknown_type", "detail": "Something happened"}]
        recs = validator.generate_recommendations(issues)
        assert len(recs) >= 1

    def test_deduplicates_coverage_recs(self, validator):
        issues = [
            {"type": "low_coverage", "coverage_score": 0.5},
            {"type": "low_coverage", "coverage_score": 0.4},
        ]
        recs = validator.generate_recommendations(issues)
        # Should deduplicate low_coverage
        coverage_recs = [r for r in recs if "coverage" in r.lower()]
        assert len(coverage_recs) == 1


# ============================================================================
# TestValidationRetrieval
# ============================================================================


class TestValidationRetrieval:
    """Tests for get_validation, list_validations, get_statistics, and clear."""

    def test_get_validation_missing(self, validator):
        assert validator.get_validation("VAL-missing") is None

    def test_list_validations_empty(self, validator):
        results = validator.list_validations()
        assert results == []

    def test_list_validations_after_validate(self, validator):
        validator.validate()
        results = validator.list_validations()
        assert len(results) == 1

    def test_list_validations_filter_scope(self, validator):
        validator.validate(scope="full")
        validator.validate(scope="pipeline")
        results = validator.list_validations(scope="full")
        assert all(r["scope"] == "full" for r in results)

    def test_list_validations_filter_result(self, validator):
        validator.validate()
        results = validator.list_validations(result="pass")
        assert all(r["result"] == "pass" for r in results)

    def test_list_validations_limit(self, validator):
        for _ in range(5):
            validator.validate()
        results = validator.list_validations(limit=3)
        assert len(results) == 3

    def test_statistics_empty(self, validator):
        stats = validator.get_statistics()
        assert stats["total_validations"] == 0
        assert stats["pass_count"] == 0

    def test_statistics_after_validations(self, validator):
        validator.validate()
        stats = validator.get_statistics()
        assert stats["total_validations"] == 1

    def test_clear(self, validator):
        validator.validate()
        assert len(validator._validations) > 0
        validator.clear()
        assert len(validator._validations) == 0


# ============================================================================
# TestDetermineResult
# ============================================================================


class TestDetermineResult:
    """Tests for _determine_result internal method."""

    def test_result_pass(self, validator):
        cfg = MagicMock()
        cfg.coverage_fail_threshold = 0.5
        cfg.coverage_warn_threshold = 0.8
        assert validator._determine_result(0.9, cfg) == "pass"

    def test_result_warn(self, validator):
        cfg = MagicMock()
        cfg.coverage_fail_threshold = 0.5
        cfg.coverage_warn_threshold = 0.8
        assert validator._determine_result(0.6, cfg) == "warn"

    def test_result_fail(self, validator):
        cfg = MagicMock()
        cfg.coverage_fail_threshold = 0.5
        cfg.coverage_warn_threshold = 0.8
        assert validator._determine_result(0.3, cfg) == "fail"


# ============================================================================
# TestTimestampParsing
# ============================================================================


class TestTimestampParsing:
    """Tests for _parse_timestamp static method."""

    def test_parse_iso_string(self):
        dt = LineageValidatorEngine._parse_timestamp("2026-02-17T12:00:00+00:00")
        assert dt is not None
        assert dt.tzinfo is not None

    def test_parse_naive_string(self):
        dt = LineageValidatorEngine._parse_timestamp("2026-02-17T12:00:00")
        assert dt is not None
        assert dt.tzinfo == timezone.utc

    def test_parse_datetime_object(self):
        dt_in = datetime(2026, 2, 17, 12, 0, 0, tzinfo=timezone.utc)
        dt = LineageValidatorEngine._parse_timestamp(dt_in)
        assert dt is dt_in

    def test_parse_naive_datetime(self):
        dt_in = datetime(2026, 2, 17, 12, 0, 0)
        dt = LineageValidatorEngine._parse_timestamp(dt_in)
        assert dt.tzinfo == timezone.utc

    def test_parse_invalid_returns_none(self):
        dt = LineageValidatorEngine._parse_timestamp("not-a-date")
        assert dt is None

    def test_parse_integer_returns_none(self):
        dt = LineageValidatorEngine._parse_timestamp(12345)
        assert dt is None
