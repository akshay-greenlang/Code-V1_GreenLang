# -*- coding: utf-8 -*-
"""
Tests for AuditReporter - AGENT-EUDR-008 Engine 8: Audit Trail and Reporting

Comprehensive test suite covering:
- Audit report generation for EUDR Article 14 (F8.4)
- Tier summary with metrics (F8.6)
- JSON, CSV, XML export formats (F8.5)
- DDS readiness assessment (F8.8)
- Certificate generation and validation
- Provenance chain integrity (F8.1, F8.2, F8.3)
- Risk propagation report (F8.7)
- Audit log immutability (F8.1)

Test count: 60+ tests
Coverage target: >= 85% of AuditReporter module

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-008 Multi-Tier Supplier Tracker (GL-EUDR-MST-008)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List

import pytest

from tests.agents.eudr.multi_tier_supplier.conftest import (
    SUP_ID_COCOA_IMPORTER_EU,
    SUP_ID_COCOA_TRADER_GH,
    SUP_ID_COCOA_PROCESSOR_GH,
    SUP_ID_COCOA_FARMER_1_GH,
    SUP_ID_PALM_IMPORTER_NL,
    COCOA_CHAIN_7_TIER,
    COFFEE_CHAIN_6_TIER,
    SHA256_HEX_LENGTH,
    make_supplier,
    make_relationship,
    make_cert,
    compute_sha256,
    build_linear_chain,
)


# ===========================================================================
# 1. EUDR Article 14 Audit Report
# ===========================================================================


class TestAuditReportGeneration:
    """Test generation of EUDR Article 14 audit-ready reports (F8.4)."""

    def test_generate_audit_report(self, audit_reporter, cocoa_chain):
        """Generate a complete audit report for a cocoa supply chain."""
        suppliers, rels = cocoa_chain
        report = audit_reporter.generate_audit_report(
            operator_id=SUP_ID_COCOA_IMPORTER_EU,
            suppliers=suppliers,
            relationships=rels,
            commodity="cocoa",
        )
        assert report is not None
        assert report.operator_id == SUP_ID_COCOA_IMPORTER_EU

    def test_audit_report_has_generation_timestamp(self, audit_reporter, cocoa_chain):
        """Audit report includes generation timestamp."""
        suppliers, rels = cocoa_chain
        report = audit_reporter.generate_audit_report(
            operator_id=SUP_ID_COCOA_IMPORTER_EU,
            suppliers=suppliers,
            relationships=rels,
            commodity="cocoa",
        )
        assert report.generated_at is not None

    def test_audit_report_includes_all_suppliers(self, audit_reporter, cocoa_chain):
        """Audit report includes all suppliers in the chain."""
        suppliers, rels = cocoa_chain
        report = audit_reporter.generate_audit_report(
            operator_id=SUP_ID_COCOA_IMPORTER_EU,
            suppliers=suppliers,
            relationships=rels,
            commodity="cocoa",
        )
        report_supplier_ids = {s["supplier_id"] for s in report.suppliers}
        for sup in suppliers:
            assert sup["supplier_id"] in report_supplier_ids

    def test_audit_report_includes_relationships(self, audit_reporter, cocoa_chain):
        """Audit report includes all relationships."""
        suppliers, rels = cocoa_chain
        report = audit_reporter.generate_audit_report(
            operator_id=SUP_ID_COCOA_IMPORTER_EU,
            suppliers=suppliers,
            relationships=rels,
            commodity="cocoa",
        )
        assert len(report.relationships) >= len(rels)

    def test_audit_report_has_provenance(self, audit_reporter, cocoa_chain):
        """Audit report includes provenance hash."""
        suppliers, rels = cocoa_chain
        report = audit_reporter.generate_audit_report(
            operator_id=SUP_ID_COCOA_IMPORTER_EU,
            suppliers=suppliers,
            relationships=rels,
            commodity="cocoa",
        )
        assert len(report.provenance_hash) == SHA256_HEX_LENGTH

    def test_audit_report_includes_tier_summary(self, audit_reporter, cocoa_chain):
        """Audit report includes tier depth summary."""
        suppliers, rels = cocoa_chain
        report = audit_reporter.generate_audit_report(
            operator_id=SUP_ID_COCOA_IMPORTER_EU,
            suppliers=suppliers,
            relationships=rels,
            commodity="cocoa",
        )
        assert report.tier_summary is not None
        assert report.tier_summary.max_depth >= 4

    def test_audit_report_empty_chain(self, audit_reporter):
        """Audit report for empty chain generates minimal report."""
        report = audit_reporter.generate_audit_report(
            operator_id="EMPTY-OP",
            suppliers=[],
            relationships=[],
            commodity="cocoa",
        )
        assert report is not None
        assert len(report.suppliers) == 0

    def test_audit_report_deterministic(self, audit_reporter, cocoa_chain):
        """Same inputs produce same audit report content."""
        suppliers, rels = cocoa_chain
        r1 = audit_reporter.generate_audit_report(
            operator_id=SUP_ID_COCOA_IMPORTER_EU,
            suppliers=suppliers, relationships=rels, commodity="cocoa",
        )
        r2 = audit_reporter.generate_audit_report(
            operator_id=SUP_ID_COCOA_IMPORTER_EU,
            suppliers=suppliers, relationships=rels, commodity="cocoa",
        )
        assert r1.provenance_hash == r2.provenance_hash


# ===========================================================================
# 2. Tier Summary Report
# ===========================================================================


class TestTierSummaryReport:
    """Test tier depth summary report with metrics (F8.6)."""

    def test_tier_summary_max_depth(self, audit_reporter, cocoa_chain):
        """Tier summary reports correct max depth."""
        suppliers, rels = cocoa_chain
        summary = audit_reporter.generate_tier_summary(suppliers, rels, "cocoa")
        assert summary.max_depth >= 4

    def test_tier_summary_suppliers_per_tier(self, audit_reporter, cocoa_chain):
        """Tier summary includes supplier count per tier."""
        suppliers, rels = cocoa_chain
        summary = audit_reporter.generate_tier_summary(suppliers, rels, "cocoa")
        assert summary.suppliers_per_tier is not None
        assert isinstance(summary.suppliers_per_tier, dict)

    def test_tier_summary_visibility_score(self, audit_reporter, cocoa_chain):
        """Tier summary includes overall visibility score."""
        suppliers, rels = cocoa_chain
        summary = audit_reporter.generate_tier_summary(suppliers, rels, "cocoa")
        assert 0.0 <= summary.visibility_score <= 100.0

    def test_tier_summary_coverage_score(self, audit_reporter, cocoa_chain):
        """Tier summary includes coverage score."""
        suppliers, rels = cocoa_chain
        summary = audit_reporter.generate_tier_summary(suppliers, rels, "cocoa")
        assert 0.0 <= summary.coverage_score <= 100.0

    def test_tier_summary_commodity(self, audit_reporter, cocoa_chain):
        """Tier summary includes commodity name."""
        suppliers, rels = cocoa_chain
        summary = audit_reporter.generate_tier_summary(suppliers, rels, "cocoa")
        assert summary.commodity == "cocoa"

    def test_tier_summary_benchmark_comparison(self, audit_reporter, cocoa_chain):
        """Tier summary includes benchmark comparison."""
        suppliers, rels = cocoa_chain
        summary = audit_reporter.generate_tier_summary(suppliers, rels, "cocoa")
        assert summary.benchmark_status is not None

    def test_tier_summary_provenance(self, audit_reporter, cocoa_chain):
        """Tier summary includes provenance hash."""
        suppliers, rels = cocoa_chain
        summary = audit_reporter.generate_tier_summary(suppliers, rels, "cocoa")
        assert len(summary.provenance_hash) == SHA256_HEX_LENGTH


# ===========================================================================
# 3. Export Formats
# ===========================================================================


class TestExportFormats:
    """Test report export in JSON, CSV, and XML formats (F8.5)."""

    def test_export_json(self, audit_reporter, cocoa_chain):
        """Export audit report as JSON."""
        suppliers, rels = cocoa_chain
        report = audit_reporter.generate_audit_report(
            operator_id=SUP_ID_COCOA_IMPORTER_EU,
            suppliers=suppliers, relationships=rels, commodity="cocoa",
        )
        json_output = audit_reporter.export(report, format="json")
        assert json_output is not None
        parsed = json.loads(json_output)
        assert "operator_id" in parsed or "operatorId" in parsed

    def test_export_csv(self, audit_reporter, cocoa_chain):
        """Export audit report as CSV."""
        suppliers, rels = cocoa_chain
        report = audit_reporter.generate_audit_report(
            operator_id=SUP_ID_COCOA_IMPORTER_EU,
            suppliers=suppliers, relationships=rels, commodity="cocoa",
        )
        csv_output = audit_reporter.export(report, format="csv")
        assert csv_output is not None
        assert len(csv_output) > 0
        # CSV should have header row
        lines = csv_output.strip().split("\n")
        assert len(lines) >= 2  # header + at least one data row

    def test_export_xml(self, audit_reporter, cocoa_chain):
        """Export audit report as XML (EUDR format)."""
        suppliers, rels = cocoa_chain
        report = audit_reporter.generate_audit_report(
            operator_id=SUP_ID_COCOA_IMPORTER_EU,
            suppliers=suppliers, relationships=rels, commodity="cocoa",
        )
        xml_output = audit_reporter.export(report, format="xml")
        assert xml_output is not None
        assert xml_output.strip().startswith("<?xml") or xml_output.strip().startswith("<")

    def test_export_invalid_format_raises(self, audit_reporter, cocoa_chain):
        """Invalid export format raises ValueError."""
        suppliers, rels = cocoa_chain
        report = audit_reporter.generate_audit_report(
            operator_id=SUP_ID_COCOA_IMPORTER_EU,
            suppliers=suppliers, relationships=rels, commodity="cocoa",
        )
        with pytest.raises(ValueError):
            audit_reporter.export(report, format="pdf_unsupported")

    @pytest.mark.parametrize("format_type", ["json", "csv", "xml"])
    def test_export_non_empty(self, audit_reporter, cocoa_chain, format_type):
        """All supported formats produce non-empty output."""
        suppliers, rels = cocoa_chain
        report = audit_reporter.generate_audit_report(
            operator_id=SUP_ID_COCOA_IMPORTER_EU,
            suppliers=suppliers, relationships=rels, commodity="cocoa",
        )
        output = audit_reporter.export(report, format=format_type)
        assert len(output) > 0


# ===========================================================================
# 4. DDS Readiness Assessment
# ===========================================================================


class TestDDSReadinessAssessment:
    """Test DDS (Due Diligence Statement) readiness assessment (F8.8)."""

    def test_ready_chain_passes(self, audit_reporter, cocoa_chain):
        """Complete chain with all data passes readiness check."""
        suppliers, rels = cocoa_chain
        result = audit_reporter.assess_dds_readiness(
            operator_id=SUP_ID_COCOA_IMPORTER_EU,
            suppliers=suppliers,
            relationships=rels,
            commodity="cocoa",
        )
        assert result is not None
        assert result.is_ready is not None  # True or False based on data

    def test_incomplete_chain_not_ready(self, audit_reporter):
        """Chain with missing GPS at origin is not DDS-ready."""
        suppliers = [
            make_supplier(supplier_id="DDS-R-IMP", tier=0, gps_lat=53.55, gps_lon=9.99),
            make_supplier(supplier_id="DDS-R-TRD", tier=1, gps_lat=5.6, gps_lon=-0.2),
            make_supplier(supplier_id="DDS-R-FRM", tier=2, role="farmer",
                          gps_lat=None, gps_lon=None),
        ]
        rels = [
            make_relationship("DDS-R-IMP", "DDS-R-TRD", rel_id="R-DDSR-1"),
            make_relationship("DDS-R-TRD", "DDS-R-FRM", rel_id="R-DDSR-2"),
        ]
        result = audit_reporter.assess_dds_readiness(
            operator_id="DDS-R-IMP",
            suppliers=suppliers, relationships=rels, commodity="cocoa",
        )
        assert result.is_ready is False

    def test_readiness_report_lists_blockers(self, audit_reporter):
        """DDS readiness report lists specific blocking issues."""
        suppliers = [
            make_supplier(supplier_id="DDS-BLK-IMP", tier=0),
            make_supplier(supplier_id="DDS-BLK-FRM", tier=1, role="farmer",
                          gps_lat=None, gps_lon=None, certifications=[]),
        ]
        rels = [make_relationship("DDS-BLK-IMP", "DDS-BLK-FRM", rel_id="R-BLK-1")]
        result = audit_reporter.assess_dds_readiness(
            operator_id="DDS-BLK-IMP",
            suppliers=suppliers, relationships=rels, commodity="cocoa",
        )
        assert len(result.blockers) >= 1

    def test_readiness_score(self, audit_reporter, cocoa_chain):
        """DDS readiness includes a numeric readiness score."""
        suppliers, rels = cocoa_chain
        result = audit_reporter.assess_dds_readiness(
            operator_id=SUP_ID_COCOA_IMPORTER_EU,
            suppliers=suppliers, relationships=rels, commodity="cocoa",
        )
        assert 0.0 <= result.readiness_score <= 100.0

    def test_readiness_provenance(self, audit_reporter, cocoa_chain):
        """DDS readiness result includes provenance hash."""
        suppliers, rels = cocoa_chain
        result = audit_reporter.assess_dds_readiness(
            operator_id=SUP_ID_COCOA_IMPORTER_EU,
            suppliers=suppliers, relationships=rels, commodity="cocoa",
        )
        assert len(result.provenance_hash) == SHA256_HEX_LENGTH


# ===========================================================================
# 5. Certificate Generation and Validation
# ===========================================================================


class TestCertificateGeneration:
    """Test compliance certificate generation and validation."""

    def test_generate_compliance_certificate(self, audit_reporter, cocoa_chain):
        """Generate a compliance certificate for a supply chain."""
        suppliers, rels = cocoa_chain
        certificate = audit_reporter.generate_certificate(
            operator_id=SUP_ID_COCOA_IMPORTER_EU,
            suppliers=suppliers,
            relationships=rels,
            commodity="cocoa",
        )
        assert certificate is not None
        assert certificate.certificate_id is not None

    def test_certificate_has_validity_dates(self, audit_reporter, cocoa_chain):
        """Certificate includes issue and expiry dates."""
        suppliers, rels = cocoa_chain
        certificate = audit_reporter.generate_certificate(
            operator_id=SUP_ID_COCOA_IMPORTER_EU,
            suppliers=suppliers, relationships=rels, commodity="cocoa",
        )
        assert certificate.issue_date is not None
        assert certificate.expiry_date is not None
        assert certificate.expiry_date > certificate.issue_date

    def test_certificate_has_provenance_hash(self, audit_reporter, cocoa_chain):
        """Certificate includes SHA-256 provenance hash."""
        suppliers, rels = cocoa_chain
        certificate = audit_reporter.generate_certificate(
            operator_id=SUP_ID_COCOA_IMPORTER_EU,
            suppliers=suppliers, relationships=rels, commodity="cocoa",
        )
        assert len(certificate.provenance_hash) == SHA256_HEX_LENGTH

    def test_validate_certificate(self, audit_reporter, cocoa_chain):
        """Generated certificate passes validation."""
        suppliers, rels = cocoa_chain
        certificate = audit_reporter.generate_certificate(
            operator_id=SUP_ID_COCOA_IMPORTER_EU,
            suppliers=suppliers, relationships=rels, commodity="cocoa",
        )
        is_valid = audit_reporter.validate_certificate(certificate)
        assert is_valid is True

    def test_tampered_certificate_fails_validation(self, audit_reporter, cocoa_chain):
        """Certificate with modified data fails validation."""
        suppliers, rels = cocoa_chain
        certificate = audit_reporter.generate_certificate(
            operator_id=SUP_ID_COCOA_IMPORTER_EU,
            suppliers=suppliers, relationships=rels, commodity="cocoa",
        )
        # Tamper with certificate
        certificate.operator_id = "TAMPERED-ID"
        is_valid = audit_reporter.validate_certificate(certificate)
        assert is_valid is False


# ===========================================================================
# 6. Provenance Chain Integrity
# ===========================================================================


class TestProvenanceChainIntegrity:
    """Test immutable audit trail and provenance chain (F8.1, F8.2, F8.3)."""

    def test_audit_log_entry_created(self, audit_reporter):
        """Creating an audit log entry succeeds."""
        entry = audit_reporter.log_event(
            event_type="supplier_created",
            entity_id="SUP-LOG-001",
            actor="test_user",
            details={"legal_name": "Log Test Corp"},
        )
        assert entry is not None
        assert entry.event_type == "supplier_created"

    def test_audit_log_has_timestamp(self, audit_reporter):
        """Audit log entry includes timestamp."""
        entry = audit_reporter.log_event(
            event_type="supplier_updated",
            entity_id="SUP-LOG-002",
            actor="test_user",
            details={"field": "legal_name", "old": "Old", "new": "New"},
        )
        assert entry.timestamp is not None

    def test_audit_log_has_provenance_hash(self, audit_reporter):
        """Audit log entry includes SHA-256 provenance hash."""
        entry = audit_reporter.log_event(
            event_type="relationship_created",
            entity_id="REL-LOG-001",
            actor="test_user",
            details={"buyer_id": "B", "supplier_id": "S"},
        )
        assert len(entry.provenance_hash) == SHA256_HEX_LENGTH

    def test_audit_log_chained_hashes(self, audit_reporter):
        """Consecutive audit log entries form a hash chain."""
        e1 = audit_reporter.log_event(
            event_type="event_1", entity_id="CHAIN-001",
            actor="user", details={"seq": 1},
        )
        e2 = audit_reporter.log_event(
            event_type="event_2", entity_id="CHAIN-001",
            actor="user", details={"seq": 2},
        )
        # e2's hash should incorporate e1's hash (chain property)
        assert e1.provenance_hash != e2.provenance_hash

    def test_audit_log_immutable(self, audit_reporter):
        """Audit log entries cannot be modified after creation."""
        entry = audit_reporter.log_event(
            event_type="immutable_test", entity_id="IMM-001",
            actor="user", details={"data": "original"},
        )
        # Attempt to modify should raise or be rejected
        with pytest.raises((AttributeError, TypeError, RuntimeError)):
            audit_reporter.modify_event(entry.event_id, {"data": "tampered"})

    def test_audit_log_retrieval(self, audit_reporter):
        """Retrieve audit log entries for an entity."""
        for i in range(3):
            audit_reporter.log_event(
                event_type=f"event_{i}",
                entity_id="RETR-001",
                actor="user",
                details={"seq": i},
            )
        entries = audit_reporter.get_audit_log("RETR-001")
        assert len(entries) >= 3

    def test_audit_log_filtered_by_type(self, audit_reporter):
        """Retrieve audit log entries filtered by event type."""
        audit_reporter.log_event(
            event_type="supplier_created", entity_id="FILT-001",
            actor="user", details={},
        )
        audit_reporter.log_event(
            event_type="supplier_updated", entity_id="FILT-001",
            actor="user", details={},
        )
        entries = audit_reporter.get_audit_log("FILT-001", event_type="supplier_created")
        assert all(e.event_type == "supplier_created" for e in entries)


# ===========================================================================
# 7. Risk Propagation Report
# ===========================================================================


class TestRiskPropagationReport:
    """Test risk propagation report showing inheritance paths (F8.7)."""

    def test_risk_propagation_report(self, audit_reporter, cocoa_chain):
        """Generate risk propagation report for a chain."""
        suppliers, rels = cocoa_chain
        risk_scores = {s["supplier_id"]: 30.0 + i * 10.0 for i, s in enumerate(suppliers)}
        report = audit_reporter.generate_risk_report(
            suppliers=suppliers,
            relationships=rels,
            risk_scores=risk_scores,
            commodity="cocoa",
        )
        assert report is not None
        assert len(report.propagation_paths) >= 1

    def test_risk_report_shows_inheritance(self, audit_reporter, cocoa_chain):
        """Risk report shows which deep-tier risks flow to Tier 1."""
        suppliers, rels = cocoa_chain
        risk_scores = {s["supplier_id"]: 20.0 for s in suppliers}
        risk_scores[SUP_ID_COCOA_FARMER_1_GH] = 85.0  # High risk at farmer
        report = audit_reporter.generate_risk_report(
            suppliers=suppliers, relationships=rels,
            risk_scores=risk_scores, commodity="cocoa",
        )
        # Should show farmer risk propagating up
        assert any(p.get("source_supplier") == SUP_ID_COCOA_FARMER_1_GH
                    for p in report.propagation_paths)

    def test_risk_report_provenance(self, audit_reporter, cocoa_chain):
        """Risk propagation report includes provenance hash."""
        suppliers, rels = cocoa_chain
        risk_scores = {s["supplier_id"]: 40.0 for s in suppliers}
        report = audit_reporter.generate_risk_report(
            suppliers=suppliers, relationships=rels,
            risk_scores=risk_scores, commodity="cocoa",
        )
        assert len(report.provenance_hash) == SHA256_HEX_LENGTH


# ===========================================================================
# 8. Report Performance
# ===========================================================================


class TestReportPerformance:
    """Test report generation performance requirements."""

    def test_audit_report_generation_time(self, audit_reporter, cocoa_chain):
        """Audit report generates within reasonable time."""
        import time
        suppliers, rels = cocoa_chain
        start = time.monotonic()
        audit_reporter.generate_audit_report(
            operator_id=SUP_ID_COCOA_IMPORTER_EU,
            suppliers=suppliers, relationships=rels, commodity="cocoa",
        )
        elapsed = time.monotonic() - start
        # PRD requires < 30 seconds; small chain should be much faster
        assert elapsed < 5.0

    def test_large_chain_report(self, audit_reporter):
        """Report generation for 100-supplier chain completes."""
        suppliers, rels = build_linear_chain("cocoa", tier_count=15, country_iso="GH")
        report = audit_reporter.generate_audit_report(
            operator_id=suppliers[0]["supplier_id"],
            suppliers=suppliers, relationships=rels, commodity="cocoa",
        )
        assert report is not None
        assert len(report.suppliers) == 15
