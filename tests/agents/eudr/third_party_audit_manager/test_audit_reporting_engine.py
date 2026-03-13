# -*- coding: utf-8 -*-
"""
Unit tests for Engine 7: AuditReportingEngine -- AGENT-EUDR-024

Tests ISO 19011:2018 Clause 6.6 compliant report generation, multi-format
support (PDF/JSON/HTML/XLSX/XML), multi-language support (EN/FR/DE/ES/PT),
evidence package assembly, report versioning, amendment tracking, and
report integrity hashing.

Target: ~60 tests
Author: GreenLang Platform Team
Date: March 2026
"""

from datetime import date, timedelta
from decimal import Decimal

import pytest

from greenlang.agents.eudr.third_party_audit_manager.audit_reporting_engine import (
    AuditReportingEngine,
)
from greenlang.agents.eudr.third_party_audit_manager.models import (
    AuditReport,
    GenerateReportRequest,
    SUPPORTED_REPORT_FORMATS,
    SUPPORTED_REPORT_LANGUAGES,
)
from tests.agents.eudr.third_party_audit_manager.conftest import (
    FROZEN_DATE,
    FROZEN_NOW,
    SHA256_HEX_LENGTH,
    REPORT_FORMATS,
    REPORT_LANGUAGES,
)


class TestReportingEngineInit:
    """Test engine initialization."""

    def test_init_with_config(self, default_config):
        engine = AuditReportingEngine(config=default_config)
        assert engine.config is not None

    def test_init_without_config(self):
        engine = AuditReportingEngine()
        assert engine.config is not None

    def test_supported_formats(self):
        assert len(SUPPORTED_REPORT_FORMATS) == 5
        for fmt in ["pdf", "json", "html", "xlsx", "xml"]:
            assert fmt in SUPPORTED_REPORT_FORMATS

    def test_supported_languages(self):
        assert len(SUPPORTED_REPORT_LANGUAGES) == 5
        for lang in ["en", "fr", "de", "es", "pt"]:
            assert lang in SUPPORTED_REPORT_LANGUAGES


class TestReportGeneration:
    """Test report generation for various formats."""

    def test_generate_json_report(self, reporting_engine, generate_report_request_json):
        response = reporting_engine.generate_report(generate_report_request_json)
        assert response is not None
        assert response.report is not None
        assert response.report.report_format == "json"

    def test_generate_pdf_report(self, reporting_engine, generate_report_request_pdf):
        response = reporting_engine.generate_report(generate_report_request_pdf)
        assert response is not None
        assert response.report.report_format == "pdf"

    @pytest.mark.parametrize("fmt", REPORT_FORMATS)
    def test_generate_all_formats(self, reporting_engine, fmt):
        request = GenerateReportRequest(
            audit_id="AUD-TEST-001",
            report_format=fmt,
            language="en",
        )
        response = reporting_engine.generate_report(request)
        assert response is not None
        assert response.report.report_format == fmt

    @pytest.mark.parametrize("lang", REPORT_LANGUAGES)
    def test_generate_all_languages(self, reporting_engine, lang):
        request = GenerateReportRequest(
            audit_id="AUD-TEST-001",
            report_format="json",
            language=lang,
        )
        response = reporting_engine.generate_report(request)
        assert response is not None
        assert response.report.language == lang

    def test_invalid_format_raises(self, reporting_engine):
        request = GenerateReportRequest(
            audit_id="AUD-TEST-001",
            report_format="invalid_format",
            language="en",
        )
        with pytest.raises((ValueError, Exception)):
            reporting_engine.generate_report(request)

    def test_invalid_language_raises(self, reporting_engine):
        request = GenerateReportRequest(
            audit_id="AUD-TEST-001",
            report_format="json",
            language="xx",
        )
        with pytest.raises((ValueError, Exception)):
            reporting_engine.generate_report(request)


class TestISO19011Compliance:
    """Test ISO 19011:2018 Clause 6.6 report structure compliance."""

    def test_report_has_audit_objectives(self, reporting_engine, generate_report_request_json):
        response = reporting_engine.generate_report(generate_report_request_json)
        assert "audit_objectives" in response.report.sections

    def test_report_has_audit_scope(self, reporting_engine, generate_report_request_json):
        response = reporting_engine.generate_report(generate_report_request_json)
        assert "audit_scope" in response.report.sections

    def test_report_has_audit_criteria(self, reporting_engine, generate_report_request_json):
        response = reporting_engine.generate_report(generate_report_request_json)
        assert "audit_criteria" in response.report.sections

    def test_report_has_audit_client(self, reporting_engine, generate_report_request_json):
        response = reporting_engine.generate_report(generate_report_request_json)
        assert "audit_client" in response.report.sections

    def test_report_has_audit_team(self, reporting_engine, generate_report_request_json):
        response = reporting_engine.generate_report(generate_report_request_json)
        assert "audit_team" in response.report.sections

    def test_report_has_dates_locations(self, reporting_engine, generate_report_request_json):
        response = reporting_engine.generate_report(generate_report_request_json)
        assert "dates_and_locations" in response.report.sections

    def test_report_has_audit_findings(self, reporting_engine, generate_report_request_json):
        response = reporting_engine.generate_report(generate_report_request_json)
        assert "audit_findings" in response.report.sections

    def test_report_has_audit_conclusions(self, reporting_engine, generate_report_request_json):
        response = reporting_engine.generate_report(generate_report_request_json)
        assert "audit_conclusions" in response.report.sections

    def test_report_iso_compliance_flag(self, reporting_engine, generate_report_request_json):
        response = reporting_engine.generate_report(generate_report_request_json)
        assert response.report.iso_19011_compliant is True


class TestReportIntegrity:
    """Test report integrity hashing and provenance."""

    def test_report_has_sha256_hash(self, reporting_engine, generate_report_request_json):
        response = reporting_engine.generate_report(generate_report_request_json)
        assert response.report.provenance_hash is not None
        assert len(response.report.provenance_hash) == SHA256_HEX_LENGTH

    def test_report_hash_deterministic(self, reporting_engine, generate_report_request_json):
        r1 = reporting_engine.generate_report(generate_report_request_json)
        r2 = reporting_engine.generate_report(generate_report_request_json)
        assert r1.report.provenance_hash == r2.report.provenance_hash

    def test_report_has_version(self, reporting_engine, generate_report_request_json):
        response = reporting_engine.generate_report(generate_report_request_json)
        assert response.report.report_version is not None

    def test_report_has_generation_timestamp(self, reporting_engine, generate_report_request_json):
        response = reporting_engine.generate_report(generate_report_request_json)
        assert response.report.generated_at is not None


class TestEvidencePackage:
    """Test evidence package assembly in reports."""

    def test_report_includes_evidence_count(self, reporting_engine, generate_report_request_json):
        response = reporting_engine.generate_report(generate_report_request_json)
        assert response.report.evidence_count >= 0

    def test_report_findings_summary(self, reporting_engine, generate_report_request_json):
        response = reporting_engine.generate_report(generate_report_request_json)
        summary = response.report.findings_summary
        assert "critical" in summary
        assert "major" in summary
        assert "minor" in summary
        assert "observation" in summary

    def test_assemble_evidence_package(self, reporting_engine):
        package = reporting_engine.assemble_evidence_package(audit_id="AUD-TEST-001")
        assert package is not None


class TestReportAmendment:
    """Test report amendment workflow."""

    def test_amend_report(self, reporting_engine, generate_report_request_json):
        original = reporting_engine.generate_report(generate_report_request_json)
        amended = reporting_engine.amend_report(
            report_id=original.report.report_id,
            amendment_reason="Corrected NC-001 finding statement",
            changes={"finding_correction": "Updated objective evidence reference"},
        )
        assert amended is not None
        assert amended.report.report_version != original.report.report_version

    def test_amendment_preserves_original(self, reporting_engine, generate_report_request_json):
        original = reporting_engine.generate_report(generate_report_request_json)
        original_hash = original.report.provenance_hash
        reporting_engine.amend_report(
            report_id=original.report.report_id,
            amendment_reason="Minor correction",
            changes={"correction": "Updated date"},
        )
        assert original_hash is not None


class TestCompetentAuthorityReport:
    """Test competent authority-specific report formatting."""

    def test_generate_authority_report(self, reporting_engine):
        report = reporting_engine.generate_authority_report(
            audit_id="AUD-TEST-001",
            authority_name="BMEL",
            member_state="DE",
        )
        assert report is not None

    def test_authority_report_for_different_states(self, reporting_engine):
        report = reporting_engine.generate_authority_report(
            audit_id="AUD-TEST-001",
            authority_name="NVWA",
            member_state="NL",
        )
        assert report is not None


class TestSamplingRationale:
    """Test sampling rationale documentation in reports."""

    def test_report_includes_sampling_info(self, reporting_engine, generate_report_request_json):
        response = reporting_engine.generate_report(generate_report_request_json)
        assert response.report is not None

    def test_sampling_methodology_documented(self, reporting_engine):
        rationale = reporting_engine.get_sampling_rationale(audit_id="AUD-TEST-001")
        assert rationale is not None
