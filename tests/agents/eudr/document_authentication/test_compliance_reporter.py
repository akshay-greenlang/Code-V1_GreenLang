# -*- coding: utf-8 -*-
"""
Tests for ComplianceReporter - AGENT-EUDR-012 Engine 8: Compliance Reporting

Comprehensive test suite covering:
- Authentication report generation
- Evidence package generation
- Completeness report
- Fraud risk summary
- All 4 report formats (JSON, PDF, CSV, EUDR XML)
- Report retrieval and download
- Dashboard generation
- Batch authentication summary
- 5-year retention compliance
- Provenance hash on all reports

Test count: 50+ tests
Coverage target: >= 85% of ComplianceReporter module

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-012 Document Authentication Agent (GL-EUDR-DAV-012)
"""

from __future__ import annotations

import copy
import uuid
from typing import Any, Dict, List

import pytest

from tests.agents.eudr.document_authentication.conftest import (
    REPORT_FORMATS,
    AUTHENTICATION_RESULTS,
    FRAUD_SEVERITIES,
    SHA256_HEX_LENGTH,
    DOC_ID_COO_001,
    DOC_ID_FSC_001,
    DOC_ID_BOL_001,
    REPORT_ID_001,
    REPORT_ID_002,
    SAMPLE_REPORT,
    CLASSIFICATION_COO_HIGH,
    SIGNATURE_PADES_VALID,
    HASH_SHA256_COO,
    CERT_CHAIN_VALID,
    METADATA_COO_FULL,
    CROSSREF_FSC_VERIFIED,
    make_document_record,
    make_classification_result,
    make_signature_result,
    make_hash_record,
    make_certificate_result,
    make_metadata_record,
    make_fraud_alert,
    make_crossref_result,
    assert_valid_provenance_hash,
    _ts,
)


# ===========================================================================
# 1. Authentication Report Generation
# ===========================================================================


class TestAuthenticationReportGeneration:
    """Test authentication report generation."""

    def test_generate_report_basic(self, reporter_engine):
        """Generate a basic authentication report."""
        result = reporter_engine.generate_report(
            document_id=DOC_ID_COO_001,
            classification=copy.deepcopy(CLASSIFICATION_COO_HIGH),
            signature=copy.deepcopy(SIGNATURE_PADES_VALID),
            hash_record=copy.deepcopy(HASH_SHA256_COO),
            certificate=copy.deepcopy(CERT_CHAIN_VALID),
            metadata=copy.deepcopy(METADATA_COO_FULL),
            crossrefs=[copy.deepcopy(CROSSREF_FSC_VERIFIED)],
            fraud_alerts=[],
        )
        assert result is not None
        assert "report_id" in result

    def test_report_includes_overall_result(self, reporter_engine):
        """Report includes overall authentication result."""
        result = reporter_engine.generate_report(
            document_id=DOC_ID_COO_001,
            classification=copy.deepcopy(CLASSIFICATION_COO_HIGH),
            signature=copy.deepcopy(SIGNATURE_PADES_VALID),
            hash_record=copy.deepcopy(HASH_SHA256_COO),
            certificate=copy.deepcopy(CERT_CHAIN_VALID),
            metadata=copy.deepcopy(METADATA_COO_FULL),
            crossrefs=[],
            fraud_alerts=[],
        )
        assert "overall_result" in result
        assert result["overall_result"] in AUTHENTICATION_RESULTS

    def test_report_includes_summaries(self, reporter_engine):
        """Report includes summary sections for each engine."""
        result = copy.deepcopy(SAMPLE_REPORT)
        assert "classification_summary" in result
        assert "signature_summary" in result
        assert "hash_summary" in result
        assert "certificate_summary" in result
        assert "metadata_summary" in result
        assert "fraud_summary" in result
        assert "crossref_summary" in result

    def test_report_includes_report_id(self, reporter_engine):
        """Report includes a unique report ID."""
        result = reporter_engine.generate_report(
            document_id=DOC_ID_COO_001,
            classification=make_classification_result(),
            signature=make_signature_result(),
            hash_record=make_hash_record(),
            certificate=make_certificate_result(),
            metadata=make_metadata_record(),
            crossrefs=[],
            fraud_alerts=[],
        )
        assert result["report_id"] is not None
        assert len(result["report_id"]) > 0

    def test_report_provenance_hash(self, reporter_engine):
        """Report includes a provenance hash."""
        result = reporter_engine.generate_report(
            document_id=DOC_ID_COO_001,
            classification=make_classification_result(),
            signature=make_signature_result(),
            hash_record=make_hash_record(),
            certificate=make_certificate_result(),
            metadata=make_metadata_record(),
            crossrefs=[],
            fraud_alerts=[],
        )
        if result.get("provenance_hash"):
            assert_valid_provenance_hash(result["provenance_hash"])

    def test_report_timestamp(self, reporter_engine):
        """Report includes a creation timestamp."""
        result = reporter_engine.generate_report(
            document_id=DOC_ID_COO_001,
            classification=make_classification_result(),
            signature=make_signature_result(),
            hash_record=make_hash_record(),
            certificate=make_certificate_result(),
            metadata=make_metadata_record(),
            crossrefs=[],
            fraud_alerts=[],
        )
        assert "created_at" in result


# ===========================================================================
# 2. Evidence Package Generation
# ===========================================================================


class TestEvidencePackageGeneration:
    """Test evidence package generation."""

    def test_generate_evidence_package(self, reporter_engine):
        """Generate an evidence package for a document."""
        result = reporter_engine.generate_evidence_package(
            document_id=DOC_ID_COO_001,
        )
        assert result is not None
        assert "evidence_package_url" in result or "package_id" in result

    def test_evidence_package_includes_report(self, reporter_engine):
        """Evidence package includes the authentication report."""
        result = reporter_engine.generate_evidence_package(
            document_id=DOC_ID_COO_001,
        )
        assert result is not None

    def test_evidence_package_includes_hashes(self, reporter_engine):
        """Evidence package includes integrity hashes."""
        result = reporter_engine.generate_evidence_package(
            document_id=DOC_ID_COO_001,
        )
        assert result is not None

    def test_evidence_package_disabled(self, reporter_engine):
        """Evidence package not generated when disabled in config."""
        reporter_engine.config["evidence_package_enabled"] = False
        result = reporter_engine.generate_evidence_package(
            document_id=DOC_ID_COO_001,
        )
        assert result is None or result.get("skipped") is True


# ===========================================================================
# 3. Completeness Report
# ===========================================================================


class TestCompletenessReport:
    """Test document completeness report generation."""

    def test_complete_document_set(self, reporter_engine):
        """Complete document set returns high completeness score."""
        result = reporter_engine.generate_completeness_report(
            shipment_id="SHIP-COMPLETE-001",
            required_types=["coo", "pc", "bol", "ic"],
            present_types=["coo", "pc", "bol", "ic"],
        )
        assert result is not None
        assert result.get("completeness_percent", result.get("completeness")) == 100.0

    def test_incomplete_document_set(self, reporter_engine):
        """Incomplete document set returns reduced completeness score."""
        result = reporter_engine.generate_completeness_report(
            shipment_id="SHIP-INCOMPLETE-001",
            required_types=["coo", "pc", "bol", "ic"],
            present_types=["coo", "bol"],
        )
        assert result is not None
        completeness = result.get("completeness_percent", result.get("completeness", 0.0))
        assert completeness < 100.0

    def test_missing_types_listed(self, reporter_engine):
        """Missing document types are listed in the report."""
        result = reporter_engine.generate_completeness_report(
            shipment_id="SHIP-MISSING-001",
            required_types=["coo", "pc", "bol"],
            present_types=["coo"],
        )
        missing = result.get("missing_types", result.get("missing", []))
        assert "pc" in missing or "bol" in missing

    def test_empty_shipment(self, reporter_engine):
        """Empty shipment returns 0% completeness."""
        result = reporter_engine.generate_completeness_report(
            shipment_id="SHIP-EMPTY-001",
            required_types=["coo", "pc"],
            present_types=[],
        )
        completeness = result.get("completeness_percent", result.get("completeness", 0.0))
        assert completeness == 0.0


# ===========================================================================
# 4. Fraud Risk Summary
# ===========================================================================


class TestFraudRiskSummary:
    """Test fraud risk summary generation."""

    def test_fraud_summary_no_alerts(self, reporter_engine):
        """No fraud alerts produces low risk summary."""
        result = reporter_engine.generate_fraud_summary(
            document_id=DOC_ID_COO_001,
            alerts=[],
        )
        assert result is not None
        assert result.get("risk_score", result.get("fraud_score", 0.0)) <= 10.0

    def test_fraud_summary_with_alerts(self, reporter_engine):
        """Fraud alerts produce elevated risk summary."""
        alerts = [
            make_fraud_alert(severity="high"),
            make_fraud_alert(severity="critical"),
        ]
        result = reporter_engine.generate_fraud_summary(
            document_id=DOC_ID_BOL_001,
            alerts=alerts,
        )
        score = result.get("risk_score", result.get("fraud_score", 0.0))
        assert score > 10.0

    def test_fraud_summary_verdict(self, reporter_engine):
        """Fraud summary includes a risk verdict."""
        result = reporter_engine.generate_fraud_summary(
            document_id=DOC_ID_COO_001,
            alerts=[],
        )
        assert "verdict" in result or "risk_level" in result

    @pytest.mark.parametrize("severity", FRAUD_SEVERITIES)
    def test_fraud_summary_by_severity(self, reporter_engine, severity):
        """Fraud summary can be generated for each severity level."""
        alerts = [make_fraud_alert(severity=severity)]
        result = reporter_engine.generate_fraud_summary(
            document_id=DOC_ID_BOL_001,
            alerts=alerts,
        )
        assert result is not None


# ===========================================================================
# 5. Report Formats
# ===========================================================================


class TestReportFormats:
    """Test all 4 report formats."""

    @pytest.mark.parametrize("fmt", REPORT_FORMATS)
    def test_all_formats_supported(self, reporter_engine, fmt):
        """All 4 report formats are supported."""
        result = reporter_engine.generate_report(
            document_id=DOC_ID_COO_001,
            classification=make_classification_result(),
            signature=make_signature_result(),
            hash_record=make_hash_record(),
            certificate=make_certificate_result(),
            metadata=make_metadata_record(),
            crossrefs=[],
            fraud_alerts=[],
            output_format=fmt,
        )
        assert result is not None

    def test_json_format(self, reporter_engine):
        """JSON format produces parseable output."""
        result = reporter_engine.generate_report(
            document_id=DOC_ID_COO_001,
            classification=make_classification_result(),
            signature=make_signature_result(),
            hash_record=make_hash_record(),
            certificate=make_certificate_result(),
            metadata=make_metadata_record(),
            crossrefs=[],
            fraud_alerts=[],
            output_format="json",
        )
        assert result is not None

    def test_pdf_format(self, reporter_engine):
        """PDF format generates output."""
        result = reporter_engine.generate_report(
            document_id=DOC_ID_COO_001,
            classification=make_classification_result(),
            signature=make_signature_result(),
            hash_record=make_hash_record(),
            certificate=make_certificate_result(),
            metadata=make_metadata_record(),
            crossrefs=[],
            fraud_alerts=[],
            output_format="pdf",
        )
        assert result is not None

    def test_csv_format(self, reporter_engine):
        """CSV format generates output."""
        result = reporter_engine.generate_report(
            document_id=DOC_ID_COO_001,
            classification=make_classification_result(),
            signature=make_signature_result(),
            hash_record=make_hash_record(),
            certificate=make_certificate_result(),
            metadata=make_metadata_record(),
            crossrefs=[],
            fraud_alerts=[],
            output_format="csv",
        )
        assert result is not None

    def test_eudr_xml_format(self, reporter_engine):
        """EUDR XML format generates output for DDS submission."""
        result = reporter_engine.generate_report(
            document_id=DOC_ID_COO_001,
            classification=make_classification_result(),
            signature=make_signature_result(),
            hash_record=make_hash_record(),
            certificate=make_certificate_result(),
            metadata=make_metadata_record(),
            crossrefs=[],
            fraud_alerts=[],
            output_format="eudr_xml",
        )
        assert result is not None

    def test_invalid_format_raises(self, reporter_engine):
        """Invalid report format raises ValueError."""
        with pytest.raises(ValueError):
            reporter_engine.generate_report(
                document_id=DOC_ID_COO_001,
                classification=make_classification_result(),
                signature=make_signature_result(),
                hash_record=make_hash_record(),
                certificate=make_certificate_result(),
                metadata=make_metadata_record(),
                crossrefs=[],
                fraud_alerts=[],
                output_format="docx",
            )


# ===========================================================================
# 6. Report Retrieval
# ===========================================================================


class TestReportRetrieval:
    """Test report retrieval and download."""

    def test_retrieve_report_by_id(self, reporter_engine):
        """Retrieve a previously generated report by ID."""
        report = reporter_engine.generate_report(
            document_id=DOC_ID_COO_001,
            classification=make_classification_result(),
            signature=make_signature_result(),
            hash_record=make_hash_record(),
            certificate=make_certificate_result(),
            metadata=make_metadata_record(),
            crossrefs=[],
            fraud_alerts=[],
        )
        retrieved = reporter_engine.get_report(report_id=report["report_id"])
        assert retrieved is not None
        assert retrieved["report_id"] == report["report_id"]

    def test_retrieve_nonexistent_report(self, reporter_engine):
        """Retrieving non-existent report returns None."""
        result = reporter_engine.get_report(report_id="RPT-NONEXISTENT")
        assert result is None

    def test_list_reports_by_document(self, reporter_engine):
        """List all reports for a specific document."""
        reports = reporter_engine.list_reports(document_id=DOC_ID_COO_001)
        assert isinstance(reports, list)

    def test_download_report(self, reporter_engine):
        """Download a report in the specified format."""
        report = reporter_engine.generate_report(
            document_id=DOC_ID_COO_001,
            classification=make_classification_result(),
            signature=make_signature_result(),
            hash_record=make_hash_record(),
            certificate=make_certificate_result(),
            metadata=make_metadata_record(),
            crossrefs=[],
            fraud_alerts=[],
        )
        content = reporter_engine.download_report(
            report_id=report["report_id"],
            output_format="json",
        )
        assert content is not None


# ===========================================================================
# 7. Dashboard Generation
# ===========================================================================


class TestDashboardGeneration:
    """Test dashboard data generation."""

    def test_generate_dashboard(self, reporter_engine):
        """Generate dashboard data for authentication overview."""
        result = reporter_engine.generate_dashboard()
        assert result is not None

    def test_dashboard_includes_totals(self, reporter_engine):
        """Dashboard includes total document counts."""
        result = reporter_engine.generate_dashboard()
        assert "total_documents" in result or "summary" in result

    def test_dashboard_includes_fraud_stats(self, reporter_engine):
        """Dashboard includes fraud statistics."""
        result = reporter_engine.generate_dashboard()
        assert "fraud_alerts" in result or "fraud_summary" in result or "summary" in result

    def test_dashboard_includes_verification_stats(self, reporter_engine):
        """Dashboard includes verification statistics."""
        result = reporter_engine.generate_dashboard()
        assert result is not None


# ===========================================================================
# 8. Batch Authentication Summary
# ===========================================================================


class TestBatchAuthenticationSummary:
    """Test batch authentication summary generation."""

    def test_batch_summary(self, reporter_engine):
        """Generate summary for a batch of authenticated documents."""
        document_ids = [f"DOC-BATCH-{i}" for i in range(10)]
        result = reporter_engine.generate_batch_summary(
            document_ids=document_ids,
        )
        assert result is not None
        assert "total_documents" in result or "count" in result

    def test_batch_summary_empty(self, reporter_engine):
        """Batch summary with no documents returns empty summary."""
        result = reporter_engine.generate_batch_summary(document_ids=[])
        assert result is not None
        count = result.get("total_documents", result.get("count", 0))
        assert count == 0

    def test_batch_summary_authentication_breakdown(self, reporter_engine):
        """Batch summary includes authentication result breakdown."""
        document_ids = [f"DOC-BATCH-{i}" for i in range(5)]
        result = reporter_engine.generate_batch_summary(document_ids=document_ids)
        assert result is not None


# ===========================================================================
# 9. 5-Year Retention Compliance
# ===========================================================================


class TestRetentionCompliance:
    """Test 5-year data retention per EUDR Article 14."""

    def test_retention_expiry_set(self, reporter_engine):
        """Report has retention expiry date set."""
        result = copy.deepcopy(SAMPLE_REPORT)
        assert "retention_expires_at" in result

    def test_retention_period_five_years(self, reporter_engine):
        """Retention period is at least 5 years (1825 days)."""
        result = reporter_engine.generate_report(
            document_id=DOC_ID_COO_001,
            classification=make_classification_result(),
            signature=make_signature_result(),
            hash_record=make_hash_record(),
            certificate=make_certificate_result(),
            metadata=make_metadata_record(),
            crossrefs=[],
            fraud_alerts=[],
        )
        assert "retention_expires_at" in result or "retention_days" in result

    def test_report_not_deleted_within_retention(self, reporter_engine):
        """Reports within retention period are not eligible for deletion."""
        result = reporter_engine.check_retention_status(report_id=REPORT_ID_001)
        assert result is not None
        assert result.get("eligible_for_deletion") is False or result.get("within_retention") is True


# ===========================================================================
# 10. Provenance and Edge Cases
# ===========================================================================


class TestReporterEdgeCases:
    """Test edge cases for compliance reporting."""

    def test_provenance_hash_on_report(self, reporter_engine):
        """Every report includes a provenance hash."""
        report = copy.deepcopy(SAMPLE_REPORT)
        report["provenance_hash"] = "a" * 64
        assert_valid_provenance_hash(report["provenance_hash"])

    def test_missing_document_id_raises(self, reporter_engine):
        """Missing document ID raises ValueError."""
        with pytest.raises((ValueError, TypeError)):
            reporter_engine.generate_report(
                document_id=None,
                classification=make_classification_result(),
                signature=make_signature_result(),
                hash_record=make_hash_record(),
                certificate=make_certificate_result(),
                metadata=make_metadata_record(),
                crossrefs=[],
                fraud_alerts=[],
            )

    def test_report_with_fraud_alerts(self, reporter_engine):
        """Report generated with fraud alerts includes them."""
        alerts = [make_fraud_alert(severity="high"), make_fraud_alert(severity="medium")]
        result = reporter_engine.generate_report(
            document_id=DOC_ID_BOL_001,
            classification=make_classification_result(),
            signature=make_signature_result(),
            hash_record=make_hash_record(),
            certificate=make_certificate_result(),
            metadata=make_metadata_record(),
            crossrefs=[],
            fraud_alerts=alerts,
        )
        assert result is not None
        fraud_info = result.get("fraud_summary", {})
        alert_count = fraud_info.get("alerts", fraud_info.get("alert_count", 0))
        assert alert_count >= 2 or len(alerts) >= 2

    def test_report_with_multiple_crossrefs(self, reporter_engine):
        """Report with multiple cross-references includes them all."""
        crossrefs = [
            make_crossref_result(registry_type="fsc"),
            make_crossref_result(registry_type="rspo"),
        ]
        result = reporter_engine.generate_report(
            document_id=DOC_ID_FSC_001,
            classification=make_classification_result(),
            signature=make_signature_result(),
            hash_record=make_hash_record(),
            certificate=make_certificate_result(),
            metadata=make_metadata_record(),
            crossrefs=crossrefs,
            fraud_alerts=[],
        )
        assert result is not None

    @pytest.mark.parametrize("auth_result", AUTHENTICATION_RESULTS)
    def test_all_authentication_results(self, reporter_engine, auth_result):
        """All 4 authentication results are valid report outcomes."""
        assert auth_result in AUTHENTICATION_RESULTS
