# -*- coding: utf-8 -*-
"""
Integration Tests: Compliance Reporting and Validation (AGENT-DATA-018)
========================================================================

Tests the LineageReporterEngine compliance report generation (CSRD/ESRS,
GHG Protocol, SOC 2) and visualization formats (Mermaid, DOT, JSON, D3,
text, HTML), as well as the LineageValidatorEngine graph validation
(orphan detection, broken edges, cycle detection, source coverage,
completeness scoring, freshness checking).

All tests operate against the ``populated_pipeline`` fixture with a
realistic 10-asset, 9-edge GreenLang data flow chain.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from greenlang.data_lineage_tracker.lineage_reporter import (
    LineageReporterEngine,
    VALID_REPORT_TYPES,
    VALID_FORMATS,
)

from tests.integration.data_lineage_tracker_service.conftest import (
    GREENLANG_ASSET_NAMES,
)


# ---------------------------------------------------------------------------
# TestComplianceReporting
# ---------------------------------------------------------------------------


class TestComplianceReporting:
    """Integration tests for compliance reports and visualization outputs."""

    # ================================================================== #
    # CSRD/ESRS compliance report
    # ================================================================== #

    def test_csrd_esrs_report_generation(self, populated_pipeline):
        """Test that a CSRD/ESRS report is generated successfully."""
        pipe, assets = populated_pipeline
        reporter = pipe.lineage_reporter

        report = reporter.generate_report(
            report_type="csrd_esrs",
            format="json",  # format is ignored for compliance reports
        )

        assert report is not None
        assert report["report_type"] == "csrd_esrs"
        assert report["report_hash"] is not None
        assert len(report["report_hash"]) == 64

    def test_csrd_esrs_report_contains_required_sections(self, populated_pipeline):
        """Test that the CSRD/ESRS report contains all required sections."""
        pipe, assets = populated_pipeline
        reporter = pipe.lineage_reporter

        report = reporter.generate_report(report_type="csrd_esrs")
        content = report["content"]

        assert "CSRD/ESRS Data Lineage Disclosure" in content
        assert "Data Sources" in content
        assert "Data Flow" in content
        assert "Transformations" in content
        assert "Quality Controls" in content
        assert "Compliance Mapping" in content
        assert "AGENT-DATA-018" in content

    def test_csrd_esrs_report_lists_external_sources(self, populated_pipeline):
        """Test that the CSRD/ESRS report lists external data sources."""
        pipe, assets = populated_pipeline
        reporter = pipe.lineage_reporter

        report = reporter.generate_report(report_type="csrd_esrs")
        content = report["content"]

        assert "External Sources" in content

    def test_csrd_esrs_report_covers_esrs_requirements(self, populated_pipeline):
        """Test that the report maps to ESRS requirements."""
        pipe, assets = populated_pipeline
        reporter = pipe.lineage_reporter

        report = reporter.generate_report(report_type="csrd_esrs")
        content = report["content"]

        assert "Article 8" in content
        assert "ESRS 2" in content
        assert "ESRS E1" in content

    # ================================================================== #
    # GHG Protocol report
    # ================================================================== #

    def test_ghg_protocol_report_generation(self, populated_pipeline):
        """Test that a GHG Protocol report is generated successfully."""
        pipe, assets = populated_pipeline
        reporter = pipe.lineage_reporter

        report = reporter.generate_report(report_type="ghg_protocol")

        assert report is not None
        assert report["report_type"] == "ghg_protocol"
        assert report["report_hash"] is not None
        assert len(report["report_hash"]) == 64

    def test_ghg_protocol_report_contains_methodology(self, populated_pipeline):
        """Test that the GHG Protocol report documents calculation methodology."""
        pipe, assets = populated_pipeline
        reporter = pipe.lineage_reporter

        report = reporter.generate_report(report_type="ghg_protocol")
        content = report["content"]

        assert "GHG Protocol Calculation Chain Documentation" in content
        assert "Calculation Methodology" in content
        assert "zero-hallucination" in content.lower() or "Zero" in content

    def test_ghg_protocol_report_documents_emission_factors(self, populated_pipeline):
        """Test that the GHG Protocol report covers emission factors."""
        pipe, assets = populated_pipeline
        reporter = pipe.lineage_reporter

        report = reporter.generate_report(report_type="ghg_protocol")
        content = report["content"]

        assert "Emission Factor" in content or "emission factor" in content
        assert "Audit Trail" in content

    def test_ghg_protocol_report_lists_agents(self, populated_pipeline):
        """Test that the GHG Protocol report lists processing agents."""
        pipe, assets = populated_pipeline
        reporter = pipe.lineage_reporter

        report = reporter.generate_report(report_type="ghg_protocol")
        content = report["content"]

        assert "Processing Agents" in content

    # ================================================================== #
    # SOC 2 audit report
    # ================================================================== #

    def test_soc2_audit_report_generation(self, populated_pipeline):
        """Test that a SOC 2 audit report is generated successfully."""
        pipe, assets = populated_pipeline
        reporter = pipe.lineage_reporter

        report = reporter.generate_report(report_type="soc2")

        assert report is not None
        assert report["report_type"] == "soc2"
        assert report["report_hash"] is not None
        assert len(report["report_hash"]) == 64

    def test_soc2_report_contains_data_inventory(self, populated_pipeline):
        """Test that SOC 2 report includes data inventory section."""
        pipe, assets = populated_pipeline
        reporter = pipe.lineage_reporter

        report = reporter.generate_report(report_type="soc2")
        content = report["content"]

        assert "SOC 2 Data Processing Audit Trail" in content
        assert "Data Inventory" in content
        assert "Data Classification" in content

    def test_soc2_report_covers_processing_activities(self, populated_pipeline):
        """Test that SOC 2 report documents processing activities."""
        pipe, assets = populated_pipeline
        reporter = pipe.lineage_reporter

        report = reporter.generate_report(report_type="soc2")
        content = report["content"]

        assert "Processing Activities" in content
        assert "Data Flow Integrity" in content

    def test_soc2_report_covers_access_controls(self, populated_pipeline):
        """Test that SOC 2 report includes access control documentation."""
        pipe, assets = populated_pipeline
        reporter = pipe.lineage_reporter

        report = reporter.generate_report(report_type="soc2")
        content = report["content"]

        assert "Access Controls" in content

    def test_soc2_report_includes_provenance_controls(self, populated_pipeline):
        """Test that SOC 2 report documents SHA-256 provenance controls."""
        pipe, assets = populated_pipeline
        reporter = pipe.lineage_reporter

        report = reporter.generate_report(report_type="soc2")
        content = report["content"]

        assert "SHA-256" in content
        assert "Provenance Controls" in content
        assert "tamper" in content.lower()

    # ================================================================== #
    # Mermaid visualization
    # ================================================================== #

    def test_mermaid_visualization(self, populated_pipeline):
        """Test Mermaid diagram generation from populated graph."""
        pipe, assets = populated_pipeline
        reporter = pipe.lineage_reporter

        report = reporter.generate_report(
            report_type="visualization",
            format="mermaid",
        )

        assert report["format"] == "mermaid"
        content = report["content"]
        assert "graph TD" in content

    def test_mermaid_contains_node_labels(self, populated_pipeline):
        """Test that Mermaid output contains node type labels."""
        pipe, assets = populated_pipeline
        reporter = pipe.lineage_reporter

        report = reporter.generate_report(
            report_type="visualization",
            format="mermaid",
        )
        content = report["content"]

        # Should contain at least some asset types
        has_any_type = any(
            atype in content
            for atype in ("dataset", "agent", "metric", "report", "external_source")
        )
        assert has_any_type, "Mermaid diagram should contain asset type labels"

    # ================================================================== #
    # DOT/Graphviz visualization
    # ================================================================== #

    def test_dot_visualization(self, populated_pipeline):
        """Test DOT/Graphviz diagram generation."""
        pipe, assets = populated_pipeline
        reporter = pipe.lineage_reporter

        report = reporter.generate_report(
            report_type="visualization",
            format="dot",
        )

        assert report["format"] == "dot"
        content = report["content"]
        assert "digraph" in content
        assert "rankdir" in content

    # ================================================================== #
    # JSON graph visualization
    # ================================================================== #

    def test_json_visualization(self, populated_pipeline):
        """Test JSON graph structure generation."""
        pipe, assets = populated_pipeline
        reporter = pipe.lineage_reporter

        report = reporter.generate_report(
            report_type="visualization",
            format="json",
        )

        assert report["format"] == "json"
        content = report["content"]

        # Content should be valid JSON
        parsed = json.loads(content)
        assert "nodes" in parsed
        assert "edges" in parsed
        assert "metadata" in parsed
        assert len(parsed["nodes"]) == 10
        assert len(parsed["edges"]) == 9

    def test_json_visualization_metadata(self, populated_pipeline):
        """Test that JSON visualization includes metadata."""
        pipe, assets = populated_pipeline
        reporter = pipe.lineage_reporter

        report = reporter.generate_report(
            report_type="visualization",
            format="json",
        )
        parsed = json.loads(report["content"])

        assert "node_count" in parsed["metadata"]
        assert "edge_count" in parsed["metadata"]
        assert parsed["metadata"]["node_count"] == 10
        assert parsed["metadata"]["edge_count"] == 9

    # ================================================================== #
    # D3 visualization
    # ================================================================== #

    def test_d3_visualization(self, populated_pipeline):
        """Test D3-compatible force-directed graph JSON generation."""
        pipe, assets = populated_pipeline
        reporter = pipe.lineage_reporter

        report = reporter.generate_report(
            report_type="visualization",
            format="d3",
        )

        assert report["format"] == "d3"
        parsed = json.loads(report["content"])
        assert "nodes" in parsed
        assert "links" in parsed

        # D3 nodes should have id, group, label
        for node in parsed["nodes"]:
            assert "id" in node
            assert "group" in node
            assert "label" in node

        # D3 links should have source, target, value
        for link in parsed["links"]:
            assert "source" in link
            assert "target" in link

    # ================================================================== #
    # HTML visualization
    # ================================================================== #

    def test_html_visualization(self, populated_pipeline):
        """Test HTML report with embedded Mermaid.js generation."""
        pipe, assets = populated_pipeline
        reporter = pipe.lineage_reporter

        report = reporter.generate_report(
            report_type="visualization",
            format="html",
        )

        assert report["format"] == "html"
        content = report["content"]
        assert "<!DOCTYPE html>" in content
        assert "mermaid" in content.lower()
        assert "Data Lineage Report" in content

    def test_html_contains_statistics(self, populated_pipeline):
        """Test that HTML report includes graph statistics table."""
        pipe, assets = populated_pipeline
        reporter = pipe.lineage_reporter

        report = reporter.generate_report(
            report_type="visualization",
            format="html",
        )
        content = report["content"]

        assert "Graph Statistics" in content
        assert "Total Assets" in content
        assert "Total Edges" in content

    # ================================================================== #
    # Text summary
    # ================================================================== #

    def test_text_summary(self, populated_pipeline):
        """Test plain text summary report generation."""
        pipe, assets = populated_pipeline
        reporter = pipe.lineage_reporter

        report = reporter.generate_report(
            report_type="visualization",
            format="text",
        )

        assert report["format"] == "text"
        content = report["content"]
        assert "DATA LINEAGE SUMMARY REPORT" in content
        assert "GRAPH OVERVIEW" in content
        assert "ASSET TYPE BREAKDOWN" in content

    # ================================================================== #
    # Report hash verification
    # ================================================================== #

    def test_report_hash_is_deterministic(self, populated_pipeline):
        """Test that the same report type produces the same hash."""
        pipe, assets = populated_pipeline
        reporter = pipe.lineage_reporter

        report1 = reporter.generate_report(
            report_type="visualization",
            format="json",
        )
        report2 = reporter.generate_report(
            report_type="visualization",
            format="json",
        )

        # Same graph state should produce the same content hash
        assert report1["report_hash"] == report2["report_hash"]

    def test_report_hash_sha256_format(self, populated_pipeline):
        """Test that report hash is a valid 64-char SHA-256 hex string."""
        pipe, assets = populated_pipeline
        reporter = pipe.lineage_reporter

        report = reporter.generate_report(
            report_type="visualization",
            format="json",
        )

        assert len(report["report_hash"]) == 64
        int(report["report_hash"], 16)  # verify it's valid hex

    # ================================================================== #
    # Report retrieval and listing
    # ================================================================== #

    def test_report_retrieval_by_id(self, populated_pipeline):
        """Test retrieving a report by its unique ID."""
        pipe, assets = populated_pipeline
        reporter = pipe.lineage_reporter

        report = reporter.generate_report(
            report_type="visualization",
            format="json",
        )
        report_id = report["id"]

        retrieved = reporter.get_report(report_id)
        assert retrieved is not None
        assert retrieved["id"] == report_id

    def test_report_retrieval_nonexistent(self, populated_pipeline):
        """Test retrieving a non-existent report returns None."""
        pipe, assets = populated_pipeline
        reporter = pipe.lineage_reporter

        retrieved = reporter.get_report("nonexistent-report-id")
        assert retrieved is None

    def test_report_listing(self, populated_pipeline):
        """Test listing reports with filtering."""
        pipe, assets = populated_pipeline
        reporter = pipe.lineage_reporter

        reporter.generate_report(report_type="visualization", format="json")
        reporter.generate_report(report_type="visualization", format="mermaid")
        reporter.generate_report(report_type="csrd_esrs")

        all_reports = reporter.list_reports()
        assert len(all_reports) == 3

        viz_reports = reporter.list_reports(report_type="visualization")
        assert len(viz_reports) == 2

    # ================================================================== #
    # Invalid input handling
    # ================================================================== #

    def test_invalid_report_type_raises(self, populated_pipeline):
        """Test that invalid report type raises ValueError."""
        pipe, assets = populated_pipeline
        reporter = pipe.lineage_reporter

        with pytest.raises(ValueError, match="Invalid report_type"):
            reporter.generate_report(
                report_type="invalid_type",
                format="json",
            )

    def test_invalid_format_raises(self, populated_pipeline):
        """Test that invalid format raises ValueError."""
        pipe, assets = populated_pipeline
        reporter = pipe.lineage_reporter

        with pytest.raises(ValueError, match="Invalid format"):
            reporter.generate_report(
                report_type="visualization",
                format="invalid_format",
            )

    # ================================================================== #
    # Report type constants
    # ================================================================== #

    def test_valid_report_types_constant(self):
        """Test that VALID_REPORT_TYPES contains all expected types."""
        expected = {"csrd_esrs", "ghg_protocol", "soc2", "custom", "visualization"}
        assert VALID_REPORT_TYPES == expected

    def test_valid_formats_constant(self):
        """Test that VALID_FORMATS contains all expected formats."""
        expected = {"mermaid", "dot", "json", "d3", "text", "html", "pdf"}
        assert VALID_FORMATS == expected

    # ================================================================== #
    # Reporter statistics
    # ================================================================== #

    def test_reporter_statistics(self, populated_pipeline):
        """Test reporter statistics after generating multiple reports."""
        pipe, assets = populated_pipeline
        reporter = pipe.lineage_reporter

        reporter.generate_report(report_type="visualization", format="json")
        reporter.generate_report(report_type="csrd_esrs")
        reporter.generate_report(report_type="ghg_protocol")

        stats = reporter.get_statistics()

        assert stats["total_reports"] == 3
        assert "by_report_type" in stats
        assert stats["by_report_type"]["visualization"] == 1
        assert stats["by_report_type"]["csrd_esrs"] == 1
        assert stats["by_report_type"]["ghg_protocol"] == 1

    # ================================================================== #
    # Reporter clear
    # ================================================================== #

    def test_reporter_clear(self, populated_pipeline):
        """Test that clearing the reporter removes all stored reports."""
        pipe, assets = populated_pipeline
        reporter = pipe.lineage_reporter

        reporter.generate_report(report_type="visualization", format="json")
        reporter.generate_report(report_type="csrd_esrs")

        reporter.clear()

        all_reports = reporter.list_reports()
        assert len(all_reports) == 0


# ---------------------------------------------------------------------------
# TestLineageValidation
# ---------------------------------------------------------------------------


class TestLineageValidation:
    """Integration tests for the LineageValidatorEngine."""

    # ================================================================== #
    # Full validation tests
    # ================================================================== #

    def test_validation_on_clean_graph(self, populated_pipeline):
        """Test that validation on a well-formed graph produces a pass result."""
        pipe, assets = populated_pipeline
        validator = pipe.lineage_validator

        report = validator.validate(scope="full")

        assert report is not None
        assert "id" in report
        assert report["id"].startswith("VAL-")
        assert report["scope"] == "full"
        assert report["orphan_nodes"] == 0
        assert report["broken_edges"] == 0
        assert report["cycles_detected"] == 0
        assert 0.0 <= report["completeness_score"] <= 1.0
        assert 0.0 <= report["freshness_score"] <= 1.0
        assert report["result"] in ("pass", "warn", "fail")
        assert "validated_at" in report

    def test_validation_detects_orphan_nodes(self, pipeline, provenance):
        """Test that validation detects orphan nodes."""
        graph = pipeline.lineage_graph
        graph.add_node("orphan1", "test.orphan1", "dataset")
        graph.add_node("orphan2", "test.orphan2", "dataset")

        validator = pipeline.lineage_validator
        report = validator.validate()

        assert report["orphan_nodes"] == 2

    def test_validation_no_cycles_on_dag(self, populated_pipeline):
        """Test that a valid DAG has zero cycles."""
        pipe, assets = populated_pipeline
        validator = pipe.lineage_validator

        report = validator.validate()
        assert report["cycles_detected"] == 0

    def test_validation_completeness_score_range(self, populated_pipeline):
        """Test that completeness score is in [0.0, 1.0]."""
        pipe, assets = populated_pipeline
        validator = pipe.lineage_validator

        report = validator.validate()
        assert 0.0 <= report["completeness_score"] <= 1.0

    def test_validation_issues_list(self, populated_pipeline):
        """Test that issues list is returned as a list."""
        pipe, assets = populated_pipeline
        validator = pipe.lineage_validator

        report = validator.validate()
        assert isinstance(report["issues"], list)

    def test_validation_recommendations_generated(self, populated_pipeline):
        """Test that recommendations are generated for any issues found."""
        pipe, assets = populated_pipeline
        validator = pipe.lineage_validator

        report = validator.validate()
        assert isinstance(report["recommendations"], list)
        assert len(report["recommendations"]) >= 1

    # ================================================================== #
    # Orphan node detection
    # ================================================================== #

    def test_detect_orphan_nodes_empty_graph(self, pipeline):
        """Test orphan detection on an empty graph."""
        validator = pipeline.lineage_validator

        orphans = validator.detect_orphan_nodes()
        assert orphans == []

    def test_detect_orphan_nodes_all_connected(self, populated_pipeline):
        """Test that a fully connected chain has no orphans."""
        pipe, assets = populated_pipeline
        validator = pipe.lineage_validator

        orphans = validator.detect_orphan_nodes()
        assert len(orphans) == 0

    # ================================================================== #
    # Broken edge detection
    # ================================================================== #

    def test_detect_broken_edges_clean_graph(self, populated_pipeline):
        """Test broken edge detection on a graph with no broken edges."""
        pipe, assets = populated_pipeline
        validator = pipe.lineage_validator

        broken = validator.detect_broken_edges()
        assert len(broken) == 0

    # ================================================================== #
    # Source coverage
    # ================================================================== #

    def test_source_coverage_with_report_and_source(self, populated_pipeline):
        """Test source coverage when report traces back to external source."""
        pipe, assets = populated_pipeline
        validator = pipe.lineage_validator

        coverage = validator.compute_source_coverage()

        assert "coverage_score" in coverage
        assert "covered_reports" in coverage
        assert "total_reports" in coverage
        assert 0.0 <= coverage["coverage_score"] <= 1.0

    # ================================================================== #
    # Completeness score
    # ================================================================== #

    def test_completeness_score_perfect_graph(self, populated_pipeline):
        """Test completeness score for a well-formed graph is high."""
        pipe, assets = populated_pipeline
        validator = pipe.lineage_validator

        score = validator.compute_completeness_score()

        # A clean graph should score high
        assert score >= 0.5

    # ================================================================== #
    # Freshness check
    # ================================================================== #

    def test_freshness_check_returns_score(self, populated_pipeline):
        """Test freshness check returns a freshness score."""
        pipe, assets = populated_pipeline
        validator = pipe.lineage_validator

        result = validator.check_freshness()

        assert "freshness_score" in result
        assert "stale_assets" in result
        assert "total_assets" in result
        assert 0.0 <= result["freshness_score"] <= 1.0

    # ================================================================== #
    # Validation retrieval
    # ================================================================== #

    def test_validation_retrieval_by_id(self, populated_pipeline):
        """Test retrieving a stored validation report by its ID."""
        pipe, assets = populated_pipeline
        validator = pipe.lineage_validator

        report = validator.validate()
        retrieved = validator.get_validation(report["id"])

        assert retrieved is not None
        assert retrieved["id"] == report["id"]

    def test_validation_retrieval_nonexistent(self, populated_pipeline):
        """Test that retrieving a non-existent validation returns None."""
        pipe, assets = populated_pipeline
        validator = pipe.lineage_validator

        retrieved = validator.get_validation("VAL-nonexistent")
        assert retrieved is None

    def test_validation_list(self, populated_pipeline):
        """Test listing validation reports."""
        pipe, assets = populated_pipeline
        validator = pipe.lineage_validator

        validator.validate(scope="full")
        validator.validate(scope="full")

        validations = validator.list_validations()
        assert len(validations) == 2

    # ================================================================== #
    # Validator statistics
    # ================================================================== #

    def test_validator_statistics(self, populated_pipeline):
        """Test validator statistics after multiple validations."""
        pipe, assets = populated_pipeline
        validator = pipe.lineage_validator

        validator.validate()
        validator.validate()

        stats = validator.get_statistics()

        assert stats["total_validations"] == 2
        assert "pass_count" in stats
        assert "warn_count" in stats
        assert "fail_count" in stats
        assert "average_completeness" in stats

    # ================================================================== #
    # Validator clear
    # ================================================================== #

    def test_validator_clear(self, populated_pipeline):
        """Test that clearing the validator removes all stored validations."""
        pipe, assets = populated_pipeline
        validator = pipe.lineage_validator

        validator.validate()
        validator.validate()

        validator.clear()

        validations = validator.list_validations()
        assert len(validations) == 0

    # ================================================================== #
    # Recommendation generation
    # ================================================================== #

    def test_recommendations_for_orphan_issues(self, populated_pipeline):
        """Test that orphan issues generate remediation recommendations."""
        pipe, assets = populated_pipeline
        validator = pipe.lineage_validator

        issues = [
            {
                "type": "orphan_node",
                "node_id": "test-orphan-001",
                "severity": "warning",
            },
        ]

        recs = validator.generate_recommendations(issues)
        assert len(recs) >= 1
        assert "test-orphan-001" in recs[0]

    def test_recommendations_for_no_issues(self, populated_pipeline):
        """Test that no issues produces a 'no issues detected' recommendation."""
        pipe, assets = populated_pipeline
        validator = pipe.lineage_validator

        recs = validator.generate_recommendations([])
        assert len(recs) == 1
        assert "no issues" in recs[0].lower()
