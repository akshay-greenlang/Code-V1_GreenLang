# -*- coding: utf-8 -*-
"""
Unit Tests for LineageReporterEngine - AGENT-DATA-018

Tests report generation across all visualization formats (Mermaid, DOT,
JSON graph, D3, text, HTML, PDF placeholder) and compliance report types
(CSRD/ESRS, GHG Protocol, SOC 2), as well as custom reports, report
retrieval, listing, statistics, and the internal graph adapter layer.

Target: 60+ tests, 9 test classes, 85%+ coverage.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import json
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from greenlang.data_lineage_tracker.lineage_graph import LineageGraphEngine
from greenlang.data_lineage_tracker.lineage_reporter import (
    LineageReporterEngine,
    VALID_REPORT_TYPES,
    VALID_FORMATS,
    _sanitize_label,
    _compute_sha256,
)
from greenlang.data_lineage_tracker.provenance import ProvenanceTracker


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def graph() -> LineageGraphEngine:
    """Create a LineageGraphEngine with a sample DAG.

    Nodes: src(external_source), ds(dataset), rpt(report), met(metric)
    Edges: src -> ds -> rpt, ds -> met
    """
    g = LineageGraphEngine()
    g.add_node("src", "raw.invoices", "external_source")
    g.add_node("ds", "clean.invoices", "dataset")
    g.add_node("rpt", "reports.emissions", "report")
    g.add_node("met", "metrics.co2", "metric")

    g.add_edge("src", "ds")
    g.add_edge("ds", "rpt")
    g.add_edge("ds", "met")

    return g


@pytest.fixture
def reporter(graph) -> LineageReporterEngine:
    """Create a LineageReporterEngine for the sample graph."""
    return LineageReporterEngine(graph)


@pytest.fixture
def reporter_with_provenance(graph) -> LineageReporterEngine:
    """Create a reporter with explicit provenance tracker."""
    prov = ProvenanceTracker()
    return LineageReporterEngine(graph, provenance=prov)


@pytest.fixture
def empty_reporter() -> LineageReporterEngine:
    """Create a reporter for an empty graph."""
    return LineageReporterEngine(LineageGraphEngine())


# ============================================================================
# TestReporterInit
# ============================================================================


class TestReporterInit:
    """Tests for LineageReporterEngine initialization."""

    def test_default_initialization(self, graph):
        r = LineageReporterEngine(graph)
        assert r._graph is graph
        assert r._provenance is not None
        assert len(r._reports) == 0

    def test_initialization_with_provenance(self, graph):
        prov = ProvenanceTracker()
        # Pre-seed so __len__ > 0 (truthy); engines use ``prov or ProvenanceTracker()``
        prov.record("test", "seed", "init")
        r = LineageReporterEngine(graph, provenance=prov)
        assert r._provenance is prov

    def test_valid_report_types(self):
        expected = {"csrd_esrs", "ghg_protocol", "soc2", "custom", "visualization"}
        assert VALID_REPORT_TYPES == expected

    def test_valid_formats(self):
        expected = {"mermaid", "dot", "json", "d3", "text", "html", "pdf"}
        assert VALID_FORMATS == expected


# ============================================================================
# TestGenerateReport
# ============================================================================


class TestGenerateReport:
    """Tests for the main generate_report method."""

    def test_generate_report_returns_dict(self, reporter):
        report = reporter.generate_report("visualization", format="json")
        assert isinstance(report, dict)

    def test_generate_report_has_required_keys(self, reporter):
        report = reporter.generate_report("visualization", format="json")
        required = [
            "id", "report_type", "format", "scope", "content",
            "report_hash", "generated_by", "generated_at",
        ]
        for key in required:
            assert key in report, f"Missing key: {key}"

    def test_generate_report_invalid_type_raises(self, reporter):
        with pytest.raises(ValueError, match="Invalid report_type"):
            reporter.generate_report("invalid_type")

    def test_generate_report_invalid_format_raises(self, reporter):
        with pytest.raises(ValueError, match="Invalid format"):
            reporter.generate_report("visualization", format="xlsx")

    def test_generate_report_stored_in_memory(self, reporter):
        report = reporter.generate_report("visualization", format="json")
        stored = reporter.get_report(report["id"])
        assert stored is not None

    def test_generate_report_has_sha256_hash(self, reporter):
        report = reporter.generate_report("visualization", format="json")
        assert len(report["report_hash"]) == 64

    def test_generate_report_custom_scope(self, reporter):
        report = reporter.generate_report("visualization", format="json", scope="my_scope")
        assert report["scope"] == "my_scope"

    def test_generate_report_records_provenance(self, reporter_with_provenance):
        prov = reporter_with_provenance._provenance
        initial = prov.entry_count
        reporter_with_provenance.generate_report("visualization", format="json")
        assert prov.entry_count > initial


# ============================================================================
# TestMermaidFormat
# ============================================================================


class TestMermaidFormat:
    """Tests for Mermaid visualization generation."""

    def test_mermaid_starts_with_graph(self, reporter):
        content = reporter.generate_mermaid()
        assert content.startswith("graph TD")

    def test_mermaid_contains_node_labels(self, reporter):
        content = reporter.generate_mermaid()
        assert "external_source" in content or "dataset" in content

    def test_mermaid_contains_edges(self, reporter):
        content = reporter.generate_mermaid()
        assert "-->" in content

    def test_mermaid_via_generate_report(self, reporter):
        report = reporter.generate_report("visualization", format="mermaid")
        assert report["format"] == "mermaid"
        assert "graph TD" in report["content"]

    def test_mermaid_empty_graph(self, empty_reporter):
        content = empty_reporter.generate_mermaid()
        assert "graph TD" in content


# ============================================================================
# TestDotFormat
# ============================================================================


class TestDotFormat:
    """Tests for DOT/Graphviz visualization generation."""

    def test_dot_contains_digraph(self, reporter):
        content = reporter.generate_dot()
        assert "digraph DataLineage" in content

    def test_dot_contains_nodes(self, reporter):
        content = reporter.generate_dot()
        assert "shape=" in content

    def test_dot_contains_edges(self, reporter):
        content = reporter.generate_dot()
        assert " -> " in content

    def test_dot_via_generate_report(self, reporter):
        report = reporter.generate_report("visualization", format="dot")
        assert report["format"] == "dot"
        assert "digraph" in report["content"]

    def test_dot_empty_graph(self, empty_reporter):
        content = empty_reporter.generate_dot()
        assert "digraph DataLineage" in content


# ============================================================================
# TestJsonFormat
# ============================================================================


class TestJsonFormat:
    """Tests for JSON graph visualization generation."""

    def test_json_is_valid(self, reporter):
        content = reporter.generate_json_graph()
        data = json.loads(content)
        assert "nodes" in data
        assert "edges" in data
        assert "metadata" in data

    def test_json_node_count(self, reporter):
        content = reporter.generate_json_graph()
        data = json.loads(content)
        assert len(data["nodes"]) == 4

    def test_json_edge_count(self, reporter):
        content = reporter.generate_json_graph()
        data = json.loads(content)
        assert len(data["edges"]) == 3

    def test_json_via_generate_report(self, reporter):
        report = reporter.generate_report("visualization", format="json")
        data = json.loads(report["content"])
        assert "nodes" in data

    def test_json_empty_graph(self, empty_reporter):
        content = empty_reporter.generate_json_graph()
        data = json.loads(content)
        assert len(data["nodes"]) == 0


# ============================================================================
# TestD3Format
# ============================================================================


class TestD3Format:
    """Tests for D3-compatible force-directed graph JSON."""

    def test_d3_has_nodes_and_links(self, reporter):
        content = reporter.generate_d3()
        data = json.loads(content)
        assert "nodes" in data
        assert "links" in data

    def test_d3_node_has_group(self, reporter):
        content = reporter.generate_d3()
        data = json.loads(content)
        for node in data["nodes"]:
            assert "group" in node
            assert "label" in node

    def test_d3_link_has_value(self, reporter):
        content = reporter.generate_d3()
        data = json.loads(content)
        for link in data["links"]:
            assert "source" in link
            assert "target" in link
            assert "value" in link

    def test_d3_via_generate_report(self, reporter):
        report = reporter.generate_report("visualization", format="d3")
        data = json.loads(report["content"])
        assert "links" in data


# ============================================================================
# TestTextFormat
# ============================================================================


class TestTextFormat:
    """Tests for plain-text summary generation."""

    def test_text_contains_header(self, reporter):
        content = reporter.generate_text_summary()
        assert "DATA LINEAGE SUMMARY REPORT" in content

    def test_text_contains_stats(self, reporter):
        content = reporter.generate_text_summary()
        assert "Total Assets" in content
        assert "Total Edges" in content

    def test_text_contains_asset_types(self, reporter):
        content = reporter.generate_text_summary()
        assert "ASSET TYPE BREAKDOWN" in content

    def test_text_via_generate_report(self, reporter):
        report = reporter.generate_report("visualization", format="text")
        assert "DATA LINEAGE SUMMARY REPORT" in report["content"]

    def test_text_empty_graph(self, empty_reporter):
        content = empty_reporter.generate_text_summary()
        assert "Total Assets" in content


# ============================================================================
# TestHtmlFormat
# ============================================================================


class TestHtmlFormat:
    """Tests for HTML report generation."""

    def test_html_is_valid_document(self, reporter):
        content = reporter.generate_html()
        assert "<!DOCTYPE html>" in content
        assert "</html>" in content

    def test_html_contains_mermaid(self, reporter):
        content = reporter.generate_html()
        assert "mermaid" in content

    def test_html_via_generate_report(self, reporter):
        report = reporter.generate_report("visualization", format="html")
        assert "<!DOCTYPE html>" in report["content"]


# ============================================================================
# TestComplianceReports
# ============================================================================


class TestComplianceReports:
    """Tests for compliance report generators (CSRD, GHG, SOC2)."""

    def test_csrd_esrs_report(self, reporter):
        content = reporter.generate_csrd_esrs_report()
        assert "CSRD/ESRS" in content
        assert "Data Sources" in content
        assert "Data Flow" in content

    def test_csrd_via_generate_report(self, reporter):
        report = reporter.generate_report("csrd_esrs", format="json")
        assert report["report_type"] == "csrd_esrs"
        assert "CSRD/ESRS" in report["content"]

    def test_ghg_protocol_report(self, reporter):
        content = reporter.generate_ghg_protocol_report()
        assert "GHG Protocol" in content
        assert "Calculation" in content

    def test_ghg_via_generate_report(self, reporter):
        report = reporter.generate_report("ghg_protocol", format="json")
        assert "GHG Protocol" in report["content"]

    def test_soc2_report(self, reporter):
        content = reporter.generate_soc2_report()
        assert "SOC 2" in content
        assert "Audit Trail" in content

    def test_soc2_via_generate_report(self, reporter):
        report = reporter.generate_report("soc2", format="json")
        assert "SOC 2" in report["content"]

    def test_csrd_empty_graph(self, empty_reporter):
        content = empty_reporter.generate_csrd_esrs_report()
        assert "CSRD/ESRS" in content

    def test_ghg_empty_graph(self, empty_reporter):
        content = empty_reporter.generate_ghg_protocol_report()
        assert "GHG Protocol" in content

    def test_soc2_empty_graph(self, empty_reporter):
        content = empty_reporter.generate_soc2_report()
        assert "SOC 2" in content


# ============================================================================
# TestCustomReport
# ============================================================================


class TestCustomReport:
    """Tests for custom report generation."""

    def test_custom_report_via_generate_report(self, reporter):
        report = reporter.generate_report("custom", format="json")
        assert report["report_type"] == "custom"
        data = json.loads(report["content"])
        assert "nodes" in data
        assert "edges" in data

    def test_custom_report_with_filter(self, reporter):
        report = reporter.generate_report(
            "custom", format="json",
            parameters={"asset_type_filter": "dataset"},
        )
        data = json.loads(report["content"])
        for node in data["nodes"]:
            assert node.get("asset_type") == "dataset"

    def test_custom_report_empty_params(self, reporter):
        report = reporter.generate_report("custom", format="json", parameters={})
        data = json.loads(report["content"])
        assert data["report_type"] == "custom"


# ============================================================================
# TestPdfPlaceholder
# ============================================================================


class TestPdfPlaceholder:
    """Tests for PDF placeholder format."""

    def test_pdf_returns_placeholder(self, reporter):
        report = reporter.generate_report("visualization", format="pdf")
        data = json.loads(report["content"])
        assert data["format"] == "pdf"
        assert data["status"] == "requires_external_render"
        assert data["html_fallback"] is True


# ============================================================================
# TestReportRetrieval
# ============================================================================


class TestReportRetrieval:
    """Tests for get_report, list_reports, get_statistics, and clear."""

    def test_get_report_missing(self, reporter):
        assert reporter.get_report("missing-id") is None

    def test_list_reports_empty(self, reporter):
        results = reporter.list_reports()
        assert results == []

    def test_list_reports_after_generation(self, reporter):
        reporter.generate_report("visualization", format="json")
        results = reporter.list_reports()
        assert len(results) == 1

    def test_list_reports_filter_type(self, reporter):
        reporter.generate_report("visualization", format="json")
        reporter.generate_report("csrd_esrs", format="json")
        results = reporter.list_reports(report_type="csrd_esrs")
        assert all(r["report_type"] == "csrd_esrs" for r in results)

    def test_list_reports_filter_format(self, reporter):
        reporter.generate_report("visualization", format="json")
        reporter.generate_report("visualization", format="mermaid")
        results = reporter.list_reports(format="json")
        assert all(r["format"] == "json" for r in results)

    def test_list_reports_limit(self, reporter):
        for _ in range(5):
            reporter.generate_report("visualization", format="json")
        results = reporter.list_reports(limit=3)
        assert len(results) == 3

    def test_list_reports_omits_content(self, reporter):
        reporter.generate_report("visualization", format="json")
        results = reporter.list_reports()
        for r in results:
            assert "content" not in r

    def test_statistics(self, reporter):
        reporter.generate_report("visualization", format="json")
        reporter.generate_report("csrd_esrs", format="json")
        stats = reporter.get_statistics()
        assert stats["total_reports"] == 2
        assert "by_report_type" in stats
        assert "by_format" in stats

    def test_clear(self, reporter):
        reporter.generate_report("visualization", format="json")
        assert len(reporter._reports) > 0
        reporter.clear()
        assert len(reporter._reports) == 0


# ============================================================================
# TestHelpers
# ============================================================================


class TestHelpers:
    """Tests for module-level helper functions."""

    def test_sanitize_label_removes_quotes(self):
        assert '"' not in _sanitize_label('hello "world"')

    def test_sanitize_label_removes_angles(self):
        result = _sanitize_label("<script>")
        assert "<" not in result
        assert ">" not in result

    def test_sanitize_label_empty(self):
        assert _sanitize_label("") == "unnamed"

    def test_compute_sha256_deterministic(self):
        h1 = _compute_sha256("hello world")
        h2 = _compute_sha256("hello world")
        assert h1 == h2
        assert len(h1) == 64

    def test_compute_sha256_different_inputs(self):
        h1 = _compute_sha256("hello")
        h2 = _compute_sha256("world")
        assert h1 != h2

    def test_index_to_alias(self):
        assert LineageReporterEngine._index_to_alias(0) == "A"
        assert LineageReporterEngine._index_to_alias(25) == "Z"
        assert LineageReporterEngine._index_to_alias(26) == "AA"

    def test_build_name_map(self):
        nodes = [
            {"id": "n1", "display_name": "Node One"},
            {"id": "n2", "qualified_name": "db.table2"},
            {"id": "n3"},
        ]
        name_map = LineageReporterEngine._build_name_map(nodes)
        assert name_map["n1"] == "Node One"
        assert name_map["n2"] == "db.table2"
        # n3 should fall back to truncated id
        assert name_map["n3"] == "n3"
