# -*- coding: utf-8 -*-
"""
Unit Tests for ImpactAnalyzerEngine - AGENT-DATA-018

Tests forward/backward lineage traversal, blast radius calculation,
impact severity scoring, dependency matrix generation, root cause analysis,
critical path discovery, full bidirectional analysis, analysis storage,
and statistics.

Target: 70+ tests, 8 test classes, 85%+ coverage.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from greenlang.data_lineage_tracker.lineage_graph import LineageGraphEngine
from greenlang.data_lineage_tracker.impact_analyzer import (
    ImpactAnalyzerEngine,
    SEVERITY_THRESHOLDS,
    _ASSET_TYPE_WEIGHTS,
    _DEFAULT_ASSET_TYPE_WEIGHT,
)
from greenlang.data_lineage_tracker.provenance import ProvenanceTracker


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def graph() -> LineageGraphEngine:
    """Create a fresh LineageGraphEngine with a sample DAG.

    Structure:
        src1 -> mid1 -> report1
        src2 -> mid1
                mid1 -> metric1
    """
    g = LineageGraphEngine()
    g.add_node("src1", "raw.invoices", "external_source")
    g.add_node("src2", "raw.spend", "external_source")
    g.add_node("mid1", "clean.data", "dataset")
    g.add_node("report1", "reports.emissions", "report")
    g.add_node("metric1", "metrics.co2", "metric")

    g.add_edge("src1", "mid1")
    g.add_edge("src2", "mid1")
    g.add_edge("mid1", "report1")
    g.add_edge("mid1", "metric1")

    return g


@pytest.fixture
def analyzer(graph) -> ImpactAnalyzerEngine:
    """Create an ImpactAnalyzerEngine with the sample graph."""
    return ImpactAnalyzerEngine(graph)


@pytest.fixture
def analyzer_with_provenance(graph) -> ImpactAnalyzerEngine:
    """Create an ImpactAnalyzerEngine with explicit provenance tracker."""
    prov = ProvenanceTracker()
    return ImpactAnalyzerEngine(graph, provenance=prov)


# ============================================================================
# TestImpactAnalyzerInit
# ============================================================================


class TestImpactAnalyzerInit:
    """Tests for ImpactAnalyzerEngine initialization."""

    def test_default_initialization(self, graph):
        a = ImpactAnalyzerEngine(graph)
        assert a._graph is graph
        assert a._provenance is not None

    def test_initialization_with_provenance(self, graph):
        prov = ProvenanceTracker()
        # Pre-seed so __len__ > 0 (truthy); engines use ``prov or ProvenanceTracker()``
        prov.record("test", "seed", "init")
        a = ImpactAnalyzerEngine(graph, provenance=prov)
        assert a._provenance is prov

    def test_empty_analyses_on_init(self, analyzer):
        assert len(analyzer._analyses) == 0

    def test_constants_defined(self):
        assert "critical" in SEVERITY_THRESHOLDS
        assert "high" in SEVERITY_THRESHOLDS
        assert "medium" in SEVERITY_THRESHOLDS
        assert "low" in SEVERITY_THRESHOLDS

    def test_asset_type_weights_defined(self):
        assert "report" in _ASSET_TYPE_WEIGHTS
        assert "metric" in _ASSET_TYPE_WEIGHTS
        assert "dataset" in _ASSET_TYPE_WEIGHTS


# ============================================================================
# TestBackwardAnalysis
# ============================================================================


class TestBackwardAnalysis:
    """Tests for analyze_backward method."""

    def test_backward_from_report(self, analyzer):
        result = analyzer.analyze_backward("report1")
        assert result["root_asset_id"] == "report1"
        assert result["direction"] == "backward"
        assert result["affected_count"] >= 1

    def test_backward_analysis_id_unique(self, analyzer):
        r1 = analyzer.analyze_backward("report1")
        r2 = analyzer.analyze_backward("report1")
        assert r1["analysis_id"] != r2["analysis_id"]

    def test_backward_has_provenance_hash(self, analyzer):
        result = analyzer.analyze_backward("report1")
        assert result["provenance_hash"] is not None
        assert len(result["provenance_hash"]) == 64

    def test_backward_source_count(self, analyzer):
        result = analyzer.analyze_backward("report1")
        assert result["source_count"] >= 0

    def test_backward_has_created_at(self, analyzer):
        result = analyzer.analyze_backward("report1")
        assert "created_at" in result

    def test_backward_empty_id_raises(self, analyzer):
        with pytest.raises(ValueError, match="asset_id must not be empty"):
            analyzer.analyze_backward("")

    def test_backward_missing_node_raises(self, analyzer):
        with pytest.raises(ValueError, match="not found"):
            analyzer.analyze_backward("nonexistent")

    def test_backward_from_root_node(self, analyzer):
        result = analyzer.analyze_backward("src1")
        assert result["affected_count"] == 0  # No upstream ancestors

    def test_backward_with_max_depth(self, analyzer):
        result = analyzer.analyze_backward("report1", max_depth=1)
        assert result["depth"] <= 1


# ============================================================================
# TestForwardAnalysis
# ============================================================================


class TestForwardAnalysis:
    """Tests for analyze_forward method."""

    def test_forward_from_source(self, analyzer):
        result = analyzer.analyze_forward("src1")
        assert result["root_asset_id"] == "src1"
        assert result["direction"] == "forward"
        assert result["affected_count"] >= 1

    def test_forward_has_blast_radius(self, analyzer):
        result = analyzer.analyze_forward("src1")
        assert "blast_radius" in result
        assert 0.0 <= result["blast_radius"] <= 1.0

    def test_forward_has_severity_summary(self, analyzer):
        result = analyzer.analyze_forward("src1")
        summary = result["severity_summary"]
        assert "critical" in summary
        assert "high" in summary
        assert "medium" in summary
        assert "low" in summary

    def test_forward_from_leaf_node(self, analyzer):
        result = analyzer.analyze_forward("report1")
        assert result["affected_count"] == 0  # No downstream

    def test_forward_empty_id_raises(self, analyzer):
        with pytest.raises(ValueError, match="asset_id must not be empty"):
            analyzer.analyze_forward("")

    def test_forward_missing_node_raises(self, analyzer):
        with pytest.raises(ValueError, match="not found"):
            analyzer.analyze_forward("nonexistent")

    def test_forward_analysis_stored(self, analyzer):
        result = analyzer.analyze_forward("src1")
        stored = analyzer.get_analysis(result["analysis_id"])
        assert stored is not None
        assert stored["analysis_id"] == result["analysis_id"]

    def test_forward_with_max_depth(self, analyzer):
        result = analyzer.analyze_forward("src1", max_depth=1)
        assert result["depth"] <= 1


# ============================================================================
# TestBlastRadius
# ============================================================================


class TestBlastRadius:
    """Tests for compute_blast_radius method."""

    def test_blast_radius_from_mid(self, analyzer):
        radius = analyzer.compute_blast_radius("mid1")
        assert 0.0 <= radius <= 1.0
        # mid1 reaches report1 and metric1 (2 out of 4 other nodes)
        assert radius == pytest.approx(2 / 4, abs=0.01)

    def test_blast_radius_from_leaf(self, analyzer):
        radius = analyzer.compute_blast_radius("report1")
        assert radius == 0.0  # No downstream consumers

    def test_blast_radius_from_root(self, analyzer):
        radius = analyzer.compute_blast_radius("src1")
        assert radius > 0.0  # Has downstream consumers

    def test_blast_radius_empty_id_raises(self, analyzer):
        with pytest.raises(ValueError, match="asset_id must not be empty"):
            analyzer.compute_blast_radius("")

    def test_blast_radius_missing_node_raises(self, analyzer):
        with pytest.raises(ValueError, match="not found"):
            analyzer.compute_blast_radius("nonexistent")


# ============================================================================
# TestImpactSeverity
# ============================================================================


class TestImpactSeverity:
    """Tests for compute_impact_severity method."""

    def test_report_at_distance_1(self, analyzer):
        affected = {"asset_type": "report", "distance": 1, "transformations_applied": []}
        severity = analyzer.compute_impact_severity("src1", affected)
        assert severity in ("critical", "high", "medium", "low")
        # report(1.0)*0.4 + 1/1*0.3 + 1.0*0.3 = 0.4+0.3+0.3 = 1.0 -> critical
        assert severity == "critical"

    def test_dataset_at_distance_3(self, analyzer):
        affected = {"asset_type": "dataset", "distance": 3, "transformations_applied": ["a", "b"]}
        severity = analyzer.compute_impact_severity("src1", affected)
        assert severity in ("critical", "high", "medium", "low")

    def test_unknown_asset_type_uses_default_weight(self, analyzer):
        affected = {"asset_type": "custom_type", "distance": 1, "transformations_applied": []}
        severity = analyzer.compute_impact_severity("src1", affected)
        assert severity in ("critical", "high", "medium", "low")

    def test_severity_decreases_with_distance(self, analyzer):
        close = {"asset_type": "report", "distance": 1, "transformations_applied": []}
        far = {"asset_type": "report", "distance": 10, "transformations_applied": []}
        sev_close = analyzer.compute_impact_severity("src1", close)
        sev_far = analyzer.compute_impact_severity("src1", far)
        # Close should be at least as severe as far
        severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
        assert severity_order[sev_close] >= severity_order[sev_far]

    @pytest.mark.parametrize("asset_type,expected_weight", [
        ("report", 1.0),
        ("metric", 0.85),
        ("dashboard", 0.80),
        ("pipeline", 0.65),
        ("dataset", 0.50),
        ("agent", 0.45),
        ("field", 0.40),
        ("external_source", 0.30),
    ])
    def test_asset_type_weights(self, asset_type, expected_weight):
        assert _ASSET_TYPE_WEIGHTS[asset_type] == expected_weight

    def test_default_weight(self):
        assert _DEFAULT_ASSET_TYPE_WEIGHT == 0.35


# ============================================================================
# TestDependencyMatrix
# ============================================================================


class TestDependencyMatrix:
    """Tests for get_dependency_matrix method."""

    def test_matrix_structure(self, analyzer):
        ids = ["src1", "mid1", "report1"]
        matrix = analyzer.get_dependency_matrix(ids)
        assert isinstance(matrix, dict)
        for aid in ids:
            assert aid in matrix
            for other in ids:
                assert other in matrix[aid]

    def test_diagonal_is_false(self, analyzer):
        ids = ["src1", "mid1", "report1"]
        matrix = analyzer.get_dependency_matrix(ids)
        for aid in ids:
            assert matrix[aid][aid] is False

    def test_report_depends_on_source(self, analyzer):
        ids = ["src1", "mid1", "report1"]
        matrix = analyzer.get_dependency_matrix(ids)
        # report1 depends on mid1 and src1 (upstream)
        assert matrix["report1"]["mid1"] is True
        assert matrix["report1"]["src1"] is True

    def test_source_does_not_depend_on_report(self, analyzer):
        ids = ["src1", "report1"]
        matrix = analyzer.get_dependency_matrix(ids)
        assert matrix["src1"]["report1"] is False

    def test_empty_list(self, analyzer):
        matrix = analyzer.get_dependency_matrix([])
        assert matrix == {}


# ============================================================================
# TestRootCauses
# ============================================================================


class TestRootCauses:
    """Tests for find_root_causes method."""

    def test_root_causes_for_report(self, analyzer):
        causes = analyzer.find_root_causes("report1")
        cause_ids = {c["asset_id"] for c in causes}
        assert "src1" in cause_ids
        assert "src2" in cause_ids

    def test_root_causes_for_source(self, analyzer):
        causes = analyzer.find_root_causes("src1")
        assert len(causes) == 0  # src1 is itself a root

    def test_root_causes_empty_id_raises(self, analyzer):
        with pytest.raises(ValueError, match="asset_id must not be empty"):
            analyzer.find_root_causes("")

    def test_root_causes_have_required_keys(self, analyzer):
        causes = analyzer.find_root_causes("report1")
        for cause in causes:
            assert "asset_id" in cause
            assert "asset_type" in cause
            assert "distance" in cause
            assert "path" in cause


# ============================================================================
# TestCriticalPaths
# ============================================================================


class TestCriticalPaths:
    """Tests for find_critical_paths method."""

    def test_critical_paths_exist(self, analyzer):
        paths = analyzer.find_critical_paths("src1", "report1")
        assert len(paths) >= 1
        assert paths[0][0] == "src1"
        assert paths[0][-1] == "report1"

    def test_critical_paths_same_node(self, analyzer):
        paths = analyzer.find_critical_paths("src1", "src1")
        assert paths == [["src1"]]

    def test_critical_paths_no_path(self, analyzer):
        paths = analyzer.find_critical_paths("report1", "src1")
        assert len(paths) == 0

    def test_critical_paths_sorted_by_length(self, analyzer):
        # Add extra path: src1 -> report1 directly
        analyzer._graph.add_edge("src1", "report1")
        paths = analyzer.find_critical_paths("src1", "report1")
        assert len(paths) >= 2
        # Shortest first
        for i in range(len(paths) - 1):
            assert len(paths[i]) <= len(paths[i + 1])

    def test_critical_paths_empty_source_raises(self, analyzer):
        with pytest.raises(ValueError, match="asset_id must not be empty"):
            analyzer.find_critical_paths("", "report1")

    def test_critical_paths_missing_target_raises(self, analyzer):
        with pytest.raises(ValueError, match="not found"):
            analyzer.find_critical_paths("src1", "nonexistent")


# ============================================================================
# TestFullAnalysis
# ============================================================================


class TestFullAnalysis:
    """Tests for analyze_full method."""

    def test_full_analysis_structure(self, analyzer):
        result = analyzer.analyze_full("mid1")
        assert result["direction"] == "bidirectional"
        assert "backward" in result
        assert "forward" in result
        assert "root_causes" in result
        assert "blast_radius" in result
        assert "total_affected_count" in result

    def test_full_analysis_combines_directions(self, analyzer):
        result = analyzer.analyze_full("mid1")
        assert result["backward"]["direction"] == "backward"
        assert result["forward"]["direction"] == "forward"

    def test_full_analysis_has_provenance(self, analyzer):
        result = analyzer.analyze_full("mid1")
        assert result["provenance_hash"] is not None
        assert len(result["provenance_hash"]) == 64

    def test_full_analysis_stored(self, analyzer):
        result = analyzer.analyze_full("mid1")
        stored = analyzer.get_analysis(result["analysis_id"])
        assert stored is not None

    def test_full_analysis_empty_id_raises(self, analyzer):
        with pytest.raises(ValueError, match="asset_id must not be empty"):
            analyzer.analyze_full("")

    def test_full_analysis_total_affected_deduped(self, analyzer):
        result = analyzer.analyze_full("mid1")
        # total_affected_count should be the union of backward and forward
        backward_ids = {a["asset_id"] for a in result["backward"]["affected_assets"]}
        forward_ids = {a["asset_id"] for a in result["forward"]["affected_assets"]}
        expected = len(backward_ids | forward_ids)
        assert result["total_affected_count"] == expected


# ============================================================================
# TestAnalysisStorage
# ============================================================================


class TestAnalysisStorage:
    """Tests for get_analysis, list_analyses, get_statistics, and clear."""

    def test_get_analysis_returns_none_for_missing(self, analyzer):
        assert analyzer.get_analysis("missing-id") is None

    def test_list_analyses_empty(self, analyzer):
        results = analyzer.list_analyses()
        assert results == []

    def test_list_analyses_filters_by_direction(self, analyzer):
        analyzer.analyze_forward("src1")
        analyzer.analyze_backward("report1")
        forward_only = analyzer.list_analyses(direction="forward")
        assert all(a["direction"] == "forward" for a in forward_only)

    def test_list_analyses_filters_by_asset(self, analyzer):
        analyzer.analyze_forward("src1")
        analyzer.analyze_forward("mid1")
        results = analyzer.list_analyses(asset_id="src1")
        assert all(a["root_asset_id"] == "src1" for a in results)

    def test_list_analyses_respects_limit(self, analyzer):
        for _ in range(5):
            analyzer.analyze_forward("src1")
        results = analyzer.list_analyses(limit=3)
        assert len(results) == 3

    def test_statistics_empty(self, analyzer):
        stats = analyzer.get_statistics()
        assert stats["total_analyses"] == 0
        assert stats["avg_affected_count"] == 0.0

    def test_statistics_after_analyses(self, analyzer):
        analyzer.analyze_forward("src1")
        analyzer.analyze_backward("report1")
        stats = analyzer.get_statistics()
        assert stats["total_analyses"] == 2
        assert stats["forward_count"] == 1
        assert stats["backward_count"] == 1

    def test_clear(self, analyzer):
        analyzer.analyze_forward("src1")
        assert len(analyzer._analyses) > 0
        analyzer.clear()
        assert len(analyzer._analyses) == 0

    def test_statistics_severity_distribution(self, analyzer):
        analyzer.analyze_forward("src1")
        stats = analyzer.get_statistics()
        assert "severity_distribution" in stats


# ============================================================================
# TestInternalHelpers
# ============================================================================


class TestInternalHelpers:
    """Tests for internal helper methods."""

    def test_score_to_severity_critical(self, analyzer):
        assert analyzer._score_to_severity(0.95) == "critical"

    def test_score_to_severity_high(self, analyzer):
        assert analyzer._score_to_severity(0.75) == "high"

    def test_score_to_severity_medium(self, analyzer):
        assert analyzer._score_to_severity(0.5) == "medium"

    def test_score_to_severity_low(self, analyzer):
        assert analyzer._score_to_severity(0.3) == "low"

    def test_freshness_sensitivity_distance_1(self, analyzer):
        assert analyzer._compute_freshness_sensitivity(1) == 1.0

    def test_freshness_sensitivity_distance_2(self, analyzer):
        assert analyzer._compute_freshness_sensitivity(2) == 0.5

    def test_freshness_sensitivity_distance_0(self, analyzer):
        assert analyzer._compute_freshness_sensitivity(0) == 1.0

    def test_blast_radius_value_empty_graph(self, analyzer):
        # Mock graph with 0 nodes
        analyzer._graph = MagicMock()
        analyzer._graph.node_count.return_value = 0
        assert analyzer._compute_blast_radius_value(5) == 0.0

    def test_blast_radius_value_single_node(self, analyzer):
        analyzer._graph = MagicMock()
        analyzer._graph.node_count.return_value = 1
        assert analyzer._compute_blast_radius_value(0) == 0.0

    def test_highest_severity_all_zero(self, analyzer):
        counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        assert analyzer._highest_severity(counts) == "none"

    def test_highest_severity_picks_highest(self, analyzer):
        counts = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        assert analyzer._highest_severity(counts) == "high"

    def test_compute_provenance_hash_deterministic(self, analyzer):
        data = {"key": "value", "provenance_hash": ""}
        h1 = analyzer._compute_provenance_hash(data)
        h2 = analyzer._compute_provenance_hash(data)
        assert h1 == h2
        assert len(h1) == 64
