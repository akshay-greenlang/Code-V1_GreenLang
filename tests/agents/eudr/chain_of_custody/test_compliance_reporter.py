# -*- coding: utf-8 -*-
"""
Tests for ComplianceReporter - AGENT-EUDR-009 Engine 8: Compliance Reporting

Comprehensive test suite covering:
- Article 9 traceability report with GPS coordinates (F8.1)
- Mass balance period report (F8.2)
- Chain integrity report with gaps (F8.3)
- Document completeness report (F8.4)
- Batch genealogy report (F8.8)
- JSON/CSV/PDF/EUDR XML export formats (F8.5)
- DDS submission package assembly (F8.6)
- Provenance chain in all reports

Test count: 45+ tests
Coverage target: >= 85% of ComplianceReporter module

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-009 Chain of Custody Agent (GL-EUDR-COC-009)
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List

import pytest

from tests.agents.eudr.chain_of_custody.conftest import (
    EUDR_COMMODITIES,
    REPORT_FORMATS,
    SHA256_HEX_LENGTH,
    FAC_ID_MILL_ID,
    FAC_ID_PROC_GH,
    FAC_ID_WAREHOUSE_NL,
    BATCH_ID_COCOA_COOP_GH,
    BATCH_ID_COCOA_FARM_GH,
    PLOT_ID_COCOA_GH_1,
    PLOT_ID_COCOA_GH_2,
    build_cocoa_chain,
    build_palm_oil_chain,
    build_coffee_chain,
    build_linear_genealogy,
    assert_valid_provenance_hash,
)


# ===========================================================================
# 1. Article 9 Traceability Report (F8.1)
# ===========================================================================


class TestArticle9TraceabilityReport:
    """Test Article 9 traceability report generation."""

    def test_generate_traceability_report(self, compliance_reporter):
        """Generate an Article 9 traceability report."""
        chain = build_cocoa_chain()
        report = compliance_reporter.generate_traceability_report(chain)
        assert report is not None
        assert report["report_type"] == "article_9_traceability"

    def test_report_includes_gps_coordinates(self, compliance_reporter):
        """Traceability report includes GPS coordinates of origin plots."""
        chain = build_cocoa_chain()
        report = compliance_reporter.generate_traceability_report(chain)
        assert "origin_plots" in report
        for plot in report["origin_plots"]:
            assert "gps_lat" in plot
            assert "gps_lon" in plot
            assert isinstance(plot["gps_lat"], (int, float))
            assert isinstance(plot["gps_lon"], (int, float))

    def test_report_links_product_to_origin(self, compliance_reporter):
        """Report shows product -> custody chain -> origin plot linkage."""
        chain = build_cocoa_chain()
        report = compliance_reporter.generate_traceability_report(chain)
        assert "custody_chain" in report or "traceability_links" in report

    def test_report_includes_date_range(self, compliance_reporter):
        """Report includes production date or time range per Art. 9(1)(e)."""
        chain = build_cocoa_chain()
        report = compliance_reporter.generate_traceability_report(chain)
        assert "date_range" in report or "production_date" in report

    def test_report_includes_quantity(self, compliance_reporter):
        """Report includes quantity/weight per Art. 9(1)(f)."""
        chain = build_cocoa_chain()
        report = compliance_reporter.generate_traceability_report(chain)
        assert "total_quantity_kg" in report or "quantity" in report

    @pytest.mark.parametrize("commodity", EUDR_COMMODITIES)
    def test_traceability_all_commodities(self, compliance_reporter, commodity):
        """Traceability report works for all 7 EUDR commodities."""
        from tests.agents.eudr.chain_of_custody.conftest import make_batch, make_event
        batch = make_batch(commodity=commodity, batch_id=f"BATCH-ART9-{commodity}")
        chain = {
            "batches": [batch],
            "events": [make_event("receipt", f"BATCH-ART9-{commodity}")],
            "transformations": [],
            "documents": [],
        }
        report = compliance_reporter.generate_traceability_report(chain)
        assert report is not None


# ===========================================================================
# 2. Mass Balance Period Report (F8.2)
# ===========================================================================


class TestMassBalancePeriodReport:
    """Test mass balance period report generation."""

    def test_generate_mass_balance_report(self, compliance_reporter):
        """Generate a mass balance period report."""
        report = compliance_reporter.generate_mass_balance_report(
            facility_id=FAC_ID_MILL_ID,
            commodity="palm_oil",
            period_start="2026-01-01",
            period_end="2026-03-31",
        )
        assert report is not None
        assert report["report_type"] == "mass_balance_period"

    def test_report_includes_totals(self, compliance_reporter):
        """Mass balance report includes input/output totals."""
        report = compliance_reporter.generate_mass_balance_report(
            facility_id=FAC_ID_MILL_ID,
            commodity="palm_oil",
            period_start="2026-01-01",
            period_end="2026-03-31",
        )
        assert "total_input_kg" in report
        assert "total_output_kg" in report

    def test_report_includes_variance(self, compliance_reporter):
        """Mass balance report includes variance analysis."""
        report = compliance_reporter.generate_mass_balance_report(
            facility_id=FAC_ID_MILL_ID,
            commodity="palm_oil",
            period_start="2026-01-01",
            period_end="2026-03-31",
        )
        assert "variance_kg" in report or "variance_pct" in report

    def test_report_includes_facility_info(self, compliance_reporter):
        """Report includes facility identification."""
        report = compliance_reporter.generate_mass_balance_report(
            facility_id=FAC_ID_MILL_ID,
            commodity="palm_oil",
            period_start="2026-01-01",
            period_end="2026-03-31",
        )
        assert report["facility_id"] == FAC_ID_MILL_ID


# ===========================================================================
# 3. Chain Integrity Report (F8.3)
# ===========================================================================


class TestChainIntegrityReport:
    """Test chain integrity report with gap analysis."""

    def test_generate_integrity_report(self, compliance_reporter):
        """Generate a chain integrity report."""
        chain = build_cocoa_chain()
        report = compliance_reporter.generate_integrity_report(chain)
        assert report is not None
        assert report["report_type"] == "chain_integrity"

    def test_report_includes_gaps(self, compliance_reporter):
        """Integrity report lists temporal gaps found."""
        chain = build_cocoa_chain()
        report = compliance_reporter.generate_integrity_report(chain)
        assert "temporal_gaps" in report or "gaps" in report

    def test_report_includes_completeness_score(self, compliance_reporter):
        """Integrity report includes the completeness score."""
        chain = build_cocoa_chain()
        report = compliance_reporter.generate_integrity_report(chain)
        assert "completeness_score" in report
        assert 0.0 <= report["completeness_score"] <= 100.0

    def test_report_includes_findings(self, compliance_reporter):
        """Integrity report includes verification findings."""
        chain = build_cocoa_chain()
        report = compliance_reporter.generate_integrity_report(chain)
        assert "findings" in report or "issues" in report


# ===========================================================================
# 4. Document Completeness Report (F8.4)
# ===========================================================================


class TestDocumentCompletenessReport:
    """Test document completeness report generation."""

    def test_generate_document_report(self, compliance_reporter):
        """Generate a document completeness report."""
        chain = build_cocoa_chain()
        report = compliance_reporter.generate_document_completeness_report(chain)
        assert report is not None
        assert report["report_type"] == "document_completeness"

    def test_report_lists_missing_docs(self, compliance_reporter):
        """Report lists events with missing required documents."""
        chain = build_cocoa_chain()
        report = compliance_reporter.generate_document_completeness_report(chain)
        assert "missing_documents" in report or "gaps" in report

    def test_report_includes_coverage_score(self, compliance_reporter):
        """Report includes a document coverage score."""
        chain = build_cocoa_chain()
        report = compliance_reporter.generate_document_completeness_report(chain)
        assert "coverage_score" in report or "completeness_score" in report


# ===========================================================================
# 5. Batch Genealogy Report (F8.8)
# ===========================================================================


class TestBatchGenealogyReport:
    """Test batch genealogy report generation."""

    def test_generate_genealogy_report(self, compliance_reporter):
        """Generate a batch genealogy report from root batch."""
        chain = build_cocoa_chain()
        report = compliance_reporter.generate_genealogy_report(
            chain, root_batch_id=BATCH_ID_COCOA_FARM_GH
        )
        assert report is not None
        assert report["report_type"] == "batch_genealogy"

    def test_report_includes_tree_structure(self, compliance_reporter):
        """Genealogy report includes tree structure."""
        chain = build_cocoa_chain()
        report = compliance_reporter.generate_genealogy_report(
            chain, root_batch_id=BATCH_ID_COCOA_FARM_GH
        )
        assert "tree" in report or "nodes" in report

    def test_report_depth(self, compliance_reporter):
        """Genealogy report captures full depth of tree."""
        batches = build_linear_genealogy(depth=5)
        chain = {
            "batches": batches,
            "events": [],
            "transformations": [],
            "documents": [],
        }
        report = compliance_reporter.generate_genealogy_report(
            chain, root_batch_id=batches[0]["batch_id"]
        )
        assert report.get("depth", 0) >= 4 or len(report.get("nodes", [])) >= 5

    def test_report_from_leaf_batch(self, compliance_reporter):
        """Genealogy report can start from a leaf batch (upstream view)."""
        chain = build_cocoa_chain()
        report = compliance_reporter.generate_genealogy_report(
            chain, root_batch_id=BATCH_ID_COCOA_COOP_GH
        )
        assert report is not None


# ===========================================================================
# 6. Export Formats (F8.5)
# ===========================================================================


class TestExportFormats:
    """Test report export in JSON, CSV, PDF, EUDR XML formats."""

    @pytest.mark.parametrize("fmt", REPORT_FORMATS)
    def test_export_all_formats(self, compliance_reporter, fmt):
        """Reports can be exported in all 4 supported formats."""
        chain = build_cocoa_chain()
        report = compliance_reporter.generate_traceability_report(chain)
        exported = compliance_reporter.export_report(report, format=fmt)
        assert exported is not None

    def test_json_export_valid(self, compliance_reporter):
        """JSON export produces valid JSON."""
        chain = build_cocoa_chain()
        report = compliance_reporter.generate_traceability_report(chain)
        exported = compliance_reporter.export_report(report, format="json")
        parsed = json.loads(exported)
        assert isinstance(parsed, dict)

    def test_csv_export_has_headers(self, compliance_reporter):
        """CSV export includes column headers."""
        chain = build_cocoa_chain()
        report = compliance_reporter.generate_traceability_report(chain)
        exported = compliance_reporter.export_report(report, format="csv")
        assert isinstance(exported, str)
        lines = exported.strip().split("\n")
        assert len(lines) >= 2  # Header + at least one data row

    def test_eudr_xml_valid_structure(self, compliance_reporter):
        """EUDR XML export has valid XML structure."""
        chain = build_cocoa_chain()
        report = compliance_reporter.generate_traceability_report(chain)
        exported = compliance_reporter.export_report(report, format="eudr_xml")
        assert isinstance(exported, str)
        assert exported.strip().startswith("<") or exported.strip().startswith("<?xml")

    def test_invalid_format_raises(self, compliance_reporter):
        """Exporting in an unsupported format raises ValueError."""
        chain = build_cocoa_chain()
        report = compliance_reporter.generate_traceability_report(chain)
        with pytest.raises(ValueError):
            compliance_reporter.export_report(report, format="docx")


# ===========================================================================
# 7. DDS Submission Package (F8.6)
# ===========================================================================


class TestDDSSubmissionPackage:
    """Test DDS submission data package assembly."""

    def test_assemble_dds_package(self, compliance_reporter):
        """Assemble a DDS submission package."""
        chain = build_cocoa_chain()
        package = compliance_reporter.assemble_dds_package(chain)
        assert package is not None
        assert "operator_info" in package or "submission_data" in package

    def test_dds_includes_article_9_fields(self, compliance_reporter):
        """DDS package includes all Article 9 required fields."""
        chain = build_cocoa_chain()
        package = compliance_reporter.assemble_dds_package(chain)
        # Article 9(1)(d): geolocation
        assert "geolocation" in package or "origin_plots" in package
        # Article 9(1)(f): quantity
        assert "quantity" in package or "total_quantity_kg" in package

    def test_dds_package_completeness(self, compliance_reporter):
        """DDS package reports its completeness status."""
        chain = build_cocoa_chain()
        package = compliance_reporter.assemble_dds_package(chain)
        assert "is_complete" in package or "completeness" in package

    def test_dds_for_palm_oil(self, compliance_reporter):
        """DDS package can be assembled for palm oil chains."""
        chain = build_palm_oil_chain()
        package = compliance_reporter.assemble_dds_package(chain)
        assert package is not None


# ===========================================================================
# 8. Provenance Chain in Reports
# ===========================================================================


class TestProvenanceInReports:
    """Test that all reports include provenance chain information."""

    def test_traceability_report_provenance(self, compliance_reporter):
        """Traceability report includes provenance hash."""
        chain = build_cocoa_chain()
        report = compliance_reporter.generate_traceability_report(chain)
        assert "provenance_hash" in report
        assert_valid_provenance_hash(report["provenance_hash"])

    def test_integrity_report_provenance(self, compliance_reporter):
        """Integrity report includes provenance hash."""
        chain = build_cocoa_chain()
        report = compliance_reporter.generate_integrity_report(chain)
        assert "provenance_hash" in report
        assert_valid_provenance_hash(report["provenance_hash"])

    def test_mass_balance_report_provenance(self, compliance_reporter):
        """Mass balance report includes provenance hash."""
        report = compliance_reporter.generate_mass_balance_report(
            facility_id=FAC_ID_MILL_ID,
            commodity="palm_oil",
            period_start="2026-01-01",
            period_end="2026-03-31",
        )
        assert "provenance_hash" in report
        assert_valid_provenance_hash(report["provenance_hash"])

    def test_report_timestamp(self, compliance_reporter):
        """All reports include a generation timestamp."""
        chain = build_cocoa_chain()
        report = compliance_reporter.generate_traceability_report(chain)
        assert "generated_at" in report or "timestamp" in report

    def test_report_id_assigned(self, compliance_reporter):
        """All reports have a unique report ID."""
        chain = build_cocoa_chain()
        r1 = compliance_reporter.generate_traceability_report(chain)
        r2 = compliance_reporter.generate_traceability_report(chain)
        assert r1.get("report_id") is not None
        assert r2.get("report_id") is not None
        assert r1["report_id"] != r2["report_id"]


# ===========================================================================
# 9. Competent Authority Audit Report (F8.7)
# ===========================================================================


class TestCompetentAuthorityAuditReport:
    """Test competent authority audit report generation."""

    def test_generate_audit_report(self, compliance_reporter):
        """Generate a competent authority audit report."""
        chain = build_cocoa_chain()
        report = compliance_reporter.generate_audit_report(chain)
        assert report is not None
        assert report["report_type"] == "competent_authority_audit"

    def test_audit_report_includes_operator_info(self, compliance_reporter):
        """Audit report includes operator identification per Art. 9(1)(a)."""
        chain = build_cocoa_chain()
        report = compliance_reporter.generate_audit_report(chain)
        assert "operator_info" in report or "operator" in report

    def test_audit_report_includes_compliance_status(self, compliance_reporter):
        """Audit report includes overall compliance status."""
        chain = build_cocoa_chain()
        report = compliance_reporter.generate_audit_report(chain)
        assert "compliance_status" in report
        assert report["compliance_status"] in ("compliant", "non_compliant", "pending")

    def test_audit_report_includes_risk_assessment(self, compliance_reporter):
        """Audit report includes risk assessment summary."""
        chain = build_cocoa_chain()
        report = compliance_reporter.generate_audit_report(chain)
        assert "risk_assessment" in report or "risk_score" in report

    def test_audit_report_includes_provenance(self, compliance_reporter):
        """Audit report includes provenance hash."""
        chain = build_cocoa_chain()
        report = compliance_reporter.generate_audit_report(chain)
        assert "provenance_hash" in report
        assert_valid_provenance_hash(report["provenance_hash"])

    def test_audit_report_for_coffee_chain(self, compliance_reporter):
        """Audit report can be generated for coffee chains."""
        chain = build_coffee_chain()
        report = compliance_reporter.generate_audit_report(chain)
        assert report is not None


# ===========================================================================
# 10. Report Edge Cases and Cross-Cutting Concerns
# ===========================================================================


class TestReportEdgeCases:
    """Test edge cases in report generation."""

    def test_empty_chain_traceability(self, compliance_reporter):
        """Traceability report handles empty chain gracefully."""
        chain = {"batches": [], "events": [], "transformations": [], "documents": []}
        report = compliance_reporter.generate_traceability_report(chain)
        assert report is not None
        assert report.get("total_quantity_kg", 0) == 0 or report.get("quantity", 0) == 0

    def test_single_batch_chain(self, compliance_reporter):
        """Traceability report works with a single-batch chain."""
        from tests.agents.eudr.chain_of_custody.conftest import make_batch, make_event
        batch = make_batch(commodity="soya", batch_id="BATCH-SINGLE-SOYA")
        chain = {
            "batches": [batch],
            "events": [make_event("receipt", "BATCH-SINGLE-SOYA")],
            "transformations": [],
            "documents": [],
        }
        report = compliance_reporter.generate_traceability_report(chain)
        assert report is not None

    def test_report_includes_version(self, compliance_reporter):
        """All reports include a report format version."""
        chain = build_cocoa_chain()
        report = compliance_reporter.generate_traceability_report(chain)
        assert "version" in report or "report_version" in report

    def test_mass_balance_invalid_date_range(self, compliance_reporter):
        """Mass balance report with end before start raises ValueError."""
        with pytest.raises(ValueError):
            compliance_reporter.generate_mass_balance_report(
                facility_id=FAC_ID_MILL_ID,
                commodity="palm_oil",
                period_start="2026-06-01",
                period_end="2026-01-01",
            )

    def test_genealogy_unknown_batch_raises(self, compliance_reporter):
        """Genealogy report with unknown root batch raises ValueError."""
        chain = build_cocoa_chain()
        with pytest.raises((ValueError, KeyError)):
            compliance_reporter.generate_genealogy_report(
                chain, root_batch_id="BATCH-DOES-NOT-EXIST"
            )

    def test_dds_for_coffee_chain(self, compliance_reporter):
        """DDS package can be assembled for coffee chains."""
        chain = build_coffee_chain()
        package = compliance_reporter.assemble_dds_package(chain)
        assert package is not None
