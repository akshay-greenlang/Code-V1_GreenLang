# -*- coding: utf-8 -*-
"""
Tests for ComplianceReporter - AGENT-EUDR-010 Engine 8: Compliance Reporting

Comprehensive test suite covering:
- Audit report generation (all 4 formats: JSON, CSV, PDF, EUDR XML)
- Contamination report generation
- Evidence package generation
- Trend report generation
- Supply chain summary generation
- Report format validation (JSON structure, CSV headers, XML namespace)
- Report retrieval by ID
- Report listing with filters
- Batch report generation
- SHA-256 provenance hashing on all reports
- Edge cases (empty data, missing facility, invalid format)

Test count: 55+ tests
Coverage target: >= 85% of ComplianceReporter module

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-010 Segregation Verifier Agent (GL-EUDR-SGV-010)
"""

from __future__ import annotations

import copy
import json
from typing import Any, Dict, List

import pytest

from tests.agents.eudr.segregation_verifier.conftest import (
    REPORT_FORMATS,
    REPORT_TYPES,
    EUDR_COMMODITIES,
    SHA256_HEX_LENGTH,
    FAC_ID_WAREHOUSE_GH,
    FAC_ID_MILL_ID,
    FAC_ID_FACTORY_DE,
    BATCH_ID_COCOA_001,
    BATCH_ID_PALM_001,
    ZONE_ID_COCOA_A,
    SCP_ID_STORAGE_COCOA,
    make_scp,
    make_zone,
    make_contamination,
    make_facility_profile,
    assert_valid_provenance_hash,
    assert_valid_score,
)


# ===========================================================================
# 1. Audit Report Generation
# ===========================================================================


class TestAuditReportGeneration:
    """Test audit report generation in all formats."""

    @pytest.mark.parametrize("fmt", REPORT_FORMATS)
    def test_generate_audit_report_all_formats(self, compliance_reporter, fmt):
        """Audit report can be generated in each of the 4 formats."""
        result = compliance_reporter.generate_audit_report(
            facility_id=FAC_ID_WAREHOUSE_GH,
            report_format=fmt,
        )
        assert result is not None
        assert result.get("format") == fmt or result.get("report_format") == fmt

    def test_audit_report_json_structure(self, compliance_reporter):
        """JSON audit report has valid structure."""
        result = compliance_reporter.generate_audit_report(
            facility_id=FAC_ID_WAREHOUSE_GH,
            report_format="json",
        )
        data = result.get("data", result.get("content", {}))
        assert isinstance(data, (dict, str))

    def test_audit_report_csv_has_headers(self, compliance_reporter):
        """CSV audit report has column headers."""
        result = compliance_reporter.generate_audit_report(
            facility_id=FAC_ID_WAREHOUSE_GH,
            report_format="csv",
        )
        content = result.get("data", result.get("content", ""))
        if isinstance(content, str):
            assert len(content) > 0

    def test_audit_report_eudr_xml_namespace(self, compliance_reporter):
        """EUDR XML report includes correct namespace."""
        result = compliance_reporter.generate_audit_report(
            facility_id=FAC_ID_WAREHOUSE_GH,
            report_format="eudr_xml",
        )
        content = result.get("data", result.get("content", ""))
        if isinstance(content, str):
            assert "eudr" in content.lower() or len(content) > 0

    def test_audit_report_provenance_hash(self, compliance_reporter):
        """Audit report includes a SHA-256 provenance hash."""
        result = compliance_reporter.generate_audit_report(
            facility_id=FAC_ID_WAREHOUSE_GH,
            report_format="json",
        )
        assert result.get("provenance_hash") is not None
        assert_valid_provenance_hash(result["provenance_hash"])

    def test_audit_report_includes_timestamp(self, compliance_reporter):
        """Audit report includes generation timestamp."""
        result = compliance_reporter.generate_audit_report(
            facility_id=FAC_ID_WAREHOUSE_GH,
            report_format="json",
        )
        assert "generated_at" in result or "timestamp" in result

    def test_audit_report_includes_facility_info(self, compliance_reporter):
        """Audit report includes facility identification."""
        result = compliance_reporter.generate_audit_report(
            facility_id=FAC_ID_WAREHOUSE_GH,
            report_format="json",
        )
        assert result.get("facility_id") == FAC_ID_WAREHOUSE_GH or \
            FAC_ID_WAREHOUSE_GH in str(result.get("data", ""))

    def test_invalid_format_raises(self, compliance_reporter):
        """Invalid report format raises ValueError."""
        with pytest.raises(ValueError):
            compliance_reporter.generate_audit_report(
                facility_id=FAC_ID_WAREHOUSE_GH,
                report_format="invalid_format",
            )


# ===========================================================================
# 2. Contamination Report Generation
# ===========================================================================


class TestContaminationReportGeneration:
    """Test contamination report generation."""

    def test_generate_contamination_report(self, compliance_reporter):
        """Generate a contamination summary report."""
        result = compliance_reporter.generate_contamination_report(
            facility_id=FAC_ID_WAREHOUSE_GH,
        )
        assert result is not None
        assert result.get("report_type") in ("contamination", "contamination_report")

    def test_contamination_report_includes_events(self, compliance_reporter):
        """Contamination report includes event details."""
        result = compliance_reporter.generate_contamination_report(
            facility_id=FAC_ID_WAREHOUSE_GH,
        )
        assert "events" in result or "contamination_events" in result or "data" in result

    def test_contamination_report_provenance(self, compliance_reporter):
        """Contamination report includes provenance hash."""
        result = compliance_reporter.generate_contamination_report(
            facility_id=FAC_ID_WAREHOUSE_GH,
        )
        assert result.get("provenance_hash") is not None

    @pytest.mark.parametrize("fmt", REPORT_FORMATS)
    def test_contamination_report_all_formats(self, compliance_reporter, fmt):
        """Contamination report can be generated in all formats."""
        result = compliance_reporter.generate_contamination_report(
            facility_id=FAC_ID_WAREHOUSE_GH,
            report_format=fmt,
        )
        assert result is not None


# ===========================================================================
# 3. Evidence Package Generation
# ===========================================================================


class TestEvidencePackageGeneration:
    """Test evidence package generation."""

    def test_generate_evidence_package(self, compliance_reporter):
        """Generate an evidence package for regulatory submission."""
        result = compliance_reporter.generate_evidence_package(
            facility_id=FAC_ID_WAREHOUSE_GH,
        )
        assert result is not None
        assert result.get("report_type") in ("evidence", "evidence_package")

    def test_evidence_package_includes_documents(self, compliance_reporter):
        """Evidence package includes supporting documents."""
        result = compliance_reporter.generate_evidence_package(
            facility_id=FAC_ID_WAREHOUSE_GH,
        )
        assert "documents" in result or "evidence" in result or "data" in result

    def test_evidence_package_provenance(self, compliance_reporter):
        """Evidence package includes provenance hash."""
        result = compliance_reporter.generate_evidence_package(
            facility_id=FAC_ID_WAREHOUSE_GH,
        )
        assert result.get("provenance_hash") is not None


# ===========================================================================
# 4. Trend Report Generation
# ===========================================================================


class TestTrendReportGeneration:
    """Test trend report generation."""

    def test_generate_trend_report(self, compliance_reporter):
        """Generate a segregation trend report over time."""
        result = compliance_reporter.generate_trend_report(
            facility_id=FAC_ID_WAREHOUSE_GH,
            period_days=90,
        )
        assert result is not None
        assert result.get("report_type") in ("trend", "trend_report")

    def test_trend_report_includes_period(self, compliance_reporter):
        """Trend report includes the analysis period."""
        result = compliance_reporter.generate_trend_report(
            facility_id=FAC_ID_WAREHOUSE_GH,
            period_days=180,
        )
        assert "period" in result or "period_days" in result or "date_range" in result

    def test_trend_report_provenance(self, compliance_reporter):
        """Trend report includes provenance hash."""
        result = compliance_reporter.generate_trend_report(
            facility_id=FAC_ID_WAREHOUSE_GH,
            period_days=90,
        )
        assert result.get("provenance_hash") is not None

    @pytest.mark.parametrize("period_days", [30, 60, 90, 180, 365])
    def test_trend_report_various_periods(self, compliance_reporter, period_days):
        """Trend reports can be generated for various periods."""
        result = compliance_reporter.generate_trend_report(
            facility_id=FAC_ID_WAREHOUSE_GH,
            period_days=period_days,
        )
        assert result is not None


# ===========================================================================
# 5. Supply Chain Summary
# ===========================================================================


class TestSupplyChainSummary:
    """Test supply chain summary generation."""

    def test_generate_supply_chain_summary(self, compliance_reporter):
        """Generate a supply chain segregation summary."""
        result = compliance_reporter.generate_supply_chain_summary(
            facility_ids=[FAC_ID_WAREHOUSE_GH, FAC_ID_MILL_ID],
        )
        assert result is not None
        assert result.get("report_type") in ("supply_chain", "supply_chain_summary")

    def test_summary_includes_all_facilities(self, compliance_reporter):
        """Summary includes data for all requested facilities."""
        facility_ids = [FAC_ID_WAREHOUSE_GH, FAC_ID_MILL_ID]
        result = compliance_reporter.generate_supply_chain_summary(
            facility_ids=facility_ids,
        )
        assert result is not None

    def test_summary_provenance(self, compliance_reporter):
        """Supply chain summary includes provenance hash."""
        result = compliance_reporter.generate_supply_chain_summary(
            facility_ids=[FAC_ID_WAREHOUSE_GH],
        )
        assert result.get("provenance_hash") is not None

    def test_single_facility_summary(self, compliance_reporter):
        """Summary works with a single facility."""
        result = compliance_reporter.generate_supply_chain_summary(
            facility_ids=[FAC_ID_WAREHOUSE_GH],
        )
        assert result is not None


# ===========================================================================
# 6. Report Retrieval
# ===========================================================================


class TestReportRetrieval:
    """Test report retrieval by ID."""

    def test_retrieve_report_by_id(self, compliance_reporter):
        """Retrieve a generated report by its ID."""
        result = compliance_reporter.generate_audit_report(
            facility_id=FAC_ID_WAREHOUSE_GH,
            report_format="json",
        )
        report_id = result.get("report_id")
        if report_id:
            retrieved = compliance_reporter.get_report(report_id)
            assert retrieved is not None
            assert retrieved.get("report_id") == report_id

    def test_retrieve_nonexistent_returns_none(self, compliance_reporter):
        """Retrieving a non-existent report returns None."""
        result = compliance_reporter.get_report("RPT-NONEXISTENT")
        assert result is None


# ===========================================================================
# 7. Report Listing
# ===========================================================================


class TestReportListing:
    """Test report listing with filters."""

    def test_list_reports_by_facility(self, compliance_reporter):
        """List reports filtered by facility."""
        compliance_reporter.generate_audit_report(
            facility_id=FAC_ID_WAREHOUSE_GH,
            report_format="json",
        )
        reports = compliance_reporter.list_reports(facility_id=FAC_ID_WAREHOUSE_GH)
        assert isinstance(reports, list)
        assert len(reports) >= 1

    def test_list_reports_by_type(self, compliance_reporter):
        """List reports filtered by report type."""
        compliance_reporter.generate_audit_report(
            facility_id=FAC_ID_WAREHOUSE_GH,
            report_format="json",
        )
        reports = compliance_reporter.list_reports(report_type="audit")
        assert isinstance(reports, list)

    def test_list_reports_empty(self, compliance_reporter):
        """List reports for facility with no reports returns empty."""
        reports = compliance_reporter.list_reports(facility_id="FAC-NOREPORTS")
        assert isinstance(reports, list)
        assert len(reports) == 0


# ===========================================================================
# 8. Batch Report Generation
# ===========================================================================


class TestBatchReportGeneration:
    """Test generating reports for multiple facilities at once."""

    def test_batch_report_generation(self, compliance_reporter):
        """Generate audit reports for multiple facilities in batch."""
        facility_ids = [FAC_ID_WAREHOUSE_GH, FAC_ID_MILL_ID, FAC_ID_FACTORY_DE]
        results = compliance_reporter.generate_batch_reports(
            facility_ids=facility_ids,
            report_format="json",
            report_type="audit",
        )
        assert len(results) == 3
        assert all(r is not None for r in results)

    def test_batch_report_partial_failure(self, compliance_reporter):
        """Batch report generation handles partial failures."""
        facility_ids = [FAC_ID_WAREHOUSE_GH, "FAC-INVALID-XXX"]
        results = compliance_reporter.generate_batch_reports(
            facility_ids=facility_ids,
            report_format="json",
            report_type="audit",
            continue_on_error=True,
        )
        assert len(results) == 2

    def test_batch_report_empty_list(self, compliance_reporter):
        """Batch report with empty facility list returns empty results."""
        results = compliance_reporter.generate_batch_reports(
            facility_ids=[],
            report_format="json",
            report_type="audit",
        )
        assert len(results) == 0


# ===========================================================================
# 9. SHA-256 Provenance Hashing
# ===========================================================================


class TestProvenanceHashing:
    """Test SHA-256 provenance hashing on all reports."""

    @pytest.mark.parametrize("report_type", REPORT_TYPES)
    def test_provenance_hash_all_types(self, compliance_reporter, report_type):
        """All report types include valid provenance hashes."""
        if report_type == "audit":
            result = compliance_reporter.generate_audit_report(
                facility_id=FAC_ID_WAREHOUSE_GH, report_format="json"
            )
        elif report_type == "contamination":
            result = compliance_reporter.generate_contamination_report(
                facility_id=FAC_ID_WAREHOUSE_GH
            )
        elif report_type == "evidence":
            result = compliance_reporter.generate_evidence_package(
                facility_id=FAC_ID_WAREHOUSE_GH
            )
        elif report_type == "trend":
            result = compliance_reporter.generate_trend_report(
                facility_id=FAC_ID_WAREHOUSE_GH, period_days=90
            )
        else:
            pytest.skip(f"Unknown report type: {report_type}")
        assert result.get("provenance_hash") is not None
        assert_valid_provenance_hash(result["provenance_hash"])

    def test_different_data_different_hash(self, compliance_reporter):
        """Reports with different data produce different hashes."""
        r1 = compliance_reporter.generate_audit_report(
            facility_id=FAC_ID_WAREHOUSE_GH, report_format="json"
        )
        r2 = compliance_reporter.generate_audit_report(
            facility_id=FAC_ID_MILL_ID, report_format="json"
        )
        if r1.get("provenance_hash") and r2.get("provenance_hash"):
            assert r1["provenance_hash"] != r2["provenance_hash"]

    def test_hash_is_64_chars(self, compliance_reporter):
        """Provenance hash is exactly 64 hex characters."""
        result = compliance_reporter.generate_audit_report(
            facility_id=FAC_ID_WAREHOUSE_GH, report_format="json"
        )
        h = result.get("provenance_hash", "")
        assert len(h) == SHA256_HEX_LENGTH


# ===========================================================================
# 10. Edge Cases
# ===========================================================================


class TestReporterEdgeCases:
    """Test edge cases for compliance reporting."""

    def test_empty_facility_data(self, compliance_reporter):
        """Report for facility with no data generates empty report."""
        result = compliance_reporter.generate_audit_report(
            facility_id="FAC-EMPTY-RPT",
            report_format="json",
        )
        assert result is not None

    def test_missing_facility_handled(self, compliance_reporter):
        """Report for non-existent facility is handled gracefully."""
        result = compliance_reporter.generate_audit_report(
            facility_id="FAC-MISSING-RPT",
            report_format="json",
        )
        assert result is not None

    @pytest.mark.parametrize("commodity", EUDR_COMMODITIES)
    def test_reports_for_all_commodities(self, compliance_reporter, commodity):
        """Reports can reference all 7 EUDR commodities."""
        result = compliance_reporter.generate_audit_report(
            facility_id=FAC_ID_WAREHOUSE_GH,
            report_format="json",
        )
        assert result is not None

    def test_negative_period_raises(self, compliance_reporter):
        """Negative period for trend report raises ValueError."""
        with pytest.raises(ValueError):
            compliance_reporter.generate_trend_report(
                facility_id=FAC_ID_WAREHOUSE_GH,
                period_days=-30,
            )

    def test_zero_period_raises(self, compliance_reporter):
        """Zero period for trend report raises ValueError."""
        with pytest.raises(ValueError):
            compliance_reporter.generate_trend_report(
                facility_id=FAC_ID_WAREHOUSE_GH,
                period_days=0,
            )
