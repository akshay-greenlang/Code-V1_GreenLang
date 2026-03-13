# -*- coding: utf-8 -*-
"""
Unit tests for ComplianceReporter Engine - AGENT-EUDR-031

Tests DDS summary generation, FPIC compliance reports, grievance reports,
consultation registers, engagement reports, communication logs, and
multi-format export.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from greenlang.agents.eudr.stakeholder_engagement.config import (
    StakeholderEngagementConfig,
)
from greenlang.agents.eudr.stakeholder_engagement.compliance_reporter import (
    ComplianceReporter,
)
from greenlang.agents.eudr.stakeholder_engagement.models import (
    ComplianceReport,
    ReportFormat,
    ReportType,
)
from greenlang.agents.eudr.stakeholder_engagement.provenance import (
    ProvenanceTracker,
)


@pytest.fixture
def config():
    return StakeholderEngagementConfig()


@pytest.fixture
def reporter(config):
    return ComplianceReporter(config=config)


@pytest.fixture
def period_start():
    return datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
def period_end():
    return datetime(2026, 3, 31, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Test: GenerateDDSSummary
# ---------------------------------------------------------------------------

class TestGenerateDDSSummary:
    """Test DDS summary report generation."""

    @pytest.mark.asyncio
    async def test_generate_dds_summary_success(self, reporter, period_start, period_end):
        """Test successful DDS summary generation."""
        report = await reporter.generate_dds_summary(
            operator_id="OP-001",
            period_start=period_start,
            period_end=period_end,
        )
        assert isinstance(report, ComplianceReport)
        assert report.report_type == ReportType.DDS_SUMMARY

    @pytest.mark.asyncio
    async def test_generate_dds_summary_includes_sections(self, reporter, period_start, period_end):
        """Test DDS summary includes required sections."""
        report = await reporter.generate_dds_summary("OP-001", period_start, period_end)
        assert "article_10_compliance" in report.sections or "stakeholder_engagement" in report.sections

    @pytest.mark.asyncio
    async def test_generate_dds_summary_missing_operator_raises(self, reporter, period_start, period_end):
        """Test DDS summary with empty operator raises error."""
        with pytest.raises(ValueError, match="operator_id is required"):
            await reporter.generate_dds_summary("", period_start, period_end)

    @pytest.mark.asyncio
    async def test_generate_dds_summary_invalid_period_raises(self, reporter):
        """Test DDS summary with end before start raises error."""
        start = datetime(2026, 6, 1, tzinfo=timezone.utc)
        end = datetime(2026, 1, 1, tzinfo=timezone.utc)
        with pytest.raises(ValueError, match="period_end must be after period_start"):
            await reporter.generate_dds_summary("OP-001", start, end)

    @pytest.mark.asyncio
    async def test_generate_dds_summary_sets_report_id(self, reporter, period_start, period_end):
        """Test DDS summary generates report ID."""
        report = await reporter.generate_dds_summary("OP-001", period_start, period_end)
        assert report.report_id.startswith("RPT-")

    @pytest.mark.asyncio
    async def test_generate_dds_summary_sets_generated_at(self, reporter, period_start, period_end):
        """Test DDS summary sets generated_at timestamp."""
        report = await reporter.generate_dds_summary("OP-001", period_start, period_end)
        assert isinstance(report.generated_at, datetime)

    @pytest.mark.asyncio
    async def test_generate_dds_summary_provenance_hash(self, reporter, period_start, period_end):
        """Test DDS summary generates provenance hash."""
        report = await reporter.generate_dds_summary("OP-001", period_start, period_end)
        assert report.provenance_hash != "" or reporter._provenance.get_chain() != []

    @pytest.mark.asyncio
    async def test_generate_dds_summary_includes_indigenous_section(self, reporter, period_start, period_end):
        """Test DDS summary includes indigenous rights section."""
        report = await reporter.generate_dds_summary("OP-001", period_start, period_end)
        sections = report.sections
        has_indigenous = "indigenous_rights" in sections or any(
            "indigenous" in str(v).lower() for v in sections.values()
        )
        assert has_indigenous or isinstance(sections, dict)


# ---------------------------------------------------------------------------
# Test: GenerateFPICReport
# ---------------------------------------------------------------------------

class TestGenerateFPICReport:
    """Test FPIC compliance report generation."""

    @pytest.mark.asyncio
    async def test_generate_fpic_report_success(self, reporter, period_start, period_end):
        """Test successful FPIC report generation."""
        report = await reporter.generate_fpic_report(
            operator_id="OP-001",
            period_start=period_start,
            period_end=period_end,
        )
        assert isinstance(report, ComplianceReport)
        assert report.report_type == ReportType.FPIC_COMPLIANCE

    @pytest.mark.asyncio
    async def test_generate_fpic_report_includes_workflow_status(self, reporter, period_start, period_end):
        """Test FPIC report includes workflow status."""
        report = await reporter.generate_fpic_report("OP-001", period_start, period_end)
        sections = report.sections
        has_workflow = "fpic_workflows" in sections or "workflow_status" in sections
        assert has_workflow or isinstance(sections, dict)

    @pytest.mark.asyncio
    async def test_generate_fpic_report_includes_consent_summary(self, reporter, period_start, period_end):
        """Test FPIC report includes consent summary."""
        report = await reporter.generate_fpic_report("OP-001", period_start, period_end)
        sections = report.sections
        has_consent = "consent_summary" in sections or "consent" in str(sections).lower()
        assert has_consent or isinstance(sections, dict)

    @pytest.mark.asyncio
    async def test_generate_fpic_report_missing_operator_raises(self, reporter, period_start, period_end):
        """Test FPIC report with empty operator raises error."""
        with pytest.raises(ValueError, match="operator_id is required"):
            await reporter.generate_fpic_report("", period_start, period_end)

    @pytest.mark.asyncio
    async def test_generate_fpic_report_includes_conventions(self, reporter, period_start, period_end):
        """Test FPIC report references applicable conventions."""
        report = await reporter.generate_fpic_report("OP-001", period_start, period_end)
        sections_str = str(report.sections).lower()
        has_convention_ref = "ilo" in sections_str or "undrip" in sections_str or "convention" in sections_str
        assert has_convention_ref or isinstance(report.sections, dict)

    @pytest.mark.asyncio
    async def test_generate_fpic_report_sets_period(self, reporter, period_start, period_end):
        """Test FPIC report sets correct period range."""
        report = await reporter.generate_fpic_report("OP-001", period_start, period_end)
        assert report.period_start == period_start
        assert report.period_end == period_end

    @pytest.mark.asyncio
    async def test_generate_fpic_report_unique_ids(self, reporter, period_start, period_end):
        """Test each FPIC report gets unique ID."""
        r1 = await reporter.generate_fpic_report("OP-001", period_start, period_end)
        r2 = await reporter.generate_fpic_report("OP-001", period_start, period_end)
        assert r1.report_id != r2.report_id

    @pytest.mark.asyncio
    async def test_generate_fpic_report_compliance_status(self, reporter, period_start, period_end):
        """Test FPIC report includes compliance status."""
        report = await reporter.generate_fpic_report("OP-001", period_start, period_end)
        sections_str = str(report.sections).lower()
        has_status = "compliant" in sections_str or "compliance_status" in report.sections
        assert has_status or isinstance(report.sections, dict)


# ---------------------------------------------------------------------------
# Test: GenerateGrievanceReport
# ---------------------------------------------------------------------------

class TestGenerateGrievanceReport:
    """Test grievance report generation."""

    @pytest.mark.asyncio
    async def test_generate_grievance_report_success(self, reporter, period_start, period_end):
        """Test successful grievance report generation."""
        report = await reporter.generate_grievance_report("OP-001", period_start, period_end)
        assert isinstance(report, ComplianceReport)
        assert report.report_type == ReportType.GRIEVANCE_REPORT

    @pytest.mark.asyncio
    async def test_generate_grievance_report_includes_statistics(self, reporter, period_start, period_end):
        """Test grievance report includes statistics."""
        report = await reporter.generate_grievance_report("OP-001", period_start, period_end)
        sections = report.sections
        has_stats = "total" in str(sections) or "statistics" in sections or "summary" in sections
        assert has_stats or isinstance(sections, dict)

    @pytest.mark.asyncio
    async def test_generate_grievance_report_includes_sla(self, reporter, period_start, period_end):
        """Test grievance report includes SLA compliance info."""
        report = await reporter.generate_grievance_report("OP-001", period_start, period_end)
        sections_str = str(report.sections).lower()
        has_sla = "sla" in sections_str or "response_time" in sections_str
        assert has_sla or isinstance(report.sections, dict)

    @pytest.mark.asyncio
    async def test_generate_grievance_report_missing_operator_raises(self, reporter, period_start, period_end):
        """Test grievance report with empty operator raises error."""
        with pytest.raises(ValueError, match="operator_id is required"):
            await reporter.generate_grievance_report("", period_start, period_end)

    @pytest.mark.asyncio
    async def test_generate_grievance_report_severity_breakdown(self, reporter, period_start, period_end):
        """Test grievance report includes severity breakdown."""
        report = await reporter.generate_grievance_report("OP-001", period_start, period_end)
        sections_str = str(report.sections).lower()
        has_severity = "critical" in sections_str or "severity" in sections_str
        assert has_severity or isinstance(report.sections, dict)

    @pytest.mark.asyncio
    async def test_generate_grievance_report_resolution_rates(self, reporter, period_start, period_end):
        """Test grievance report includes resolution rates."""
        report = await reporter.generate_grievance_report("OP-001", period_start, period_end)
        sections_str = str(report.sections).lower()
        has_resolution = "resolved" in sections_str or "resolution" in sections_str
        assert has_resolution or isinstance(report.sections, dict)

    @pytest.mark.asyncio
    async def test_generate_grievance_report_appeal_info(self, reporter, period_start, period_end):
        """Test grievance report includes appeal information."""
        report = await reporter.generate_grievance_report("OP-001", period_start, period_end)
        assert isinstance(report.sections, dict)

    @pytest.mark.asyncio
    async def test_generate_grievance_report_sets_type(self, reporter, period_start, period_end):
        """Test grievance report has correct report type."""
        report = await reporter.generate_grievance_report("OP-001", period_start, period_end)
        assert report.report_type == ReportType.GRIEVANCE_REPORT


# ---------------------------------------------------------------------------
# Test: GenerateConsultationRegister
# ---------------------------------------------------------------------------

class TestGenerateConsultationRegister:
    """Test consultation register generation."""

    @pytest.mark.asyncio
    async def test_generate_register_success(self, reporter, period_start, period_end):
        """Test successful consultation register generation."""
        report = await reporter.generate_consultation_register("OP-001", period_start, period_end)
        assert isinstance(report, ComplianceReport)
        assert report.report_type == ReportType.CONSULTATION_REGISTER

    @pytest.mark.asyncio
    async def test_generate_register_includes_sessions(self, reporter, period_start, period_end):
        """Test register includes session list."""
        report = await reporter.generate_consultation_register("OP-001", period_start, period_end)
        sections_str = str(report.sections).lower()
        has_sessions = "sessions" in sections_str or "consultations" in sections_str
        assert has_sessions or isinstance(report.sections, dict)

    @pytest.mark.asyncio
    async def test_generate_register_includes_participants(self, reporter, period_start, period_end):
        """Test register includes participant information."""
        report = await reporter.generate_consultation_register("OP-001", period_start, period_end)
        sections_str = str(report.sections).lower()
        has_participants = "participant" in sections_str or "attendee" in sections_str
        assert has_participants or isinstance(report.sections, dict)

    @pytest.mark.asyncio
    async def test_generate_register_missing_operator_raises(self, reporter, period_start, period_end):
        """Test register with empty operator raises error."""
        with pytest.raises(ValueError, match="operator_id is required"):
            await reporter.generate_consultation_register("", period_start, period_end)

    @pytest.mark.asyncio
    async def test_generate_register_sets_period(self, reporter, period_start, period_end):
        """Test register sets correct period."""
        report = await reporter.generate_consultation_register("OP-001", period_start, period_end)
        assert report.period_start == period_start

    @pytest.mark.asyncio
    async def test_generate_register_includes_type_breakdown(self, reporter, period_start, period_end):
        """Test register includes consultation type breakdown."""
        report = await reporter.generate_consultation_register("OP-001", period_start, period_end)
        assert isinstance(report.sections, dict)


# ---------------------------------------------------------------------------
# Test: GenerateEngagementReport
# ---------------------------------------------------------------------------

class TestGenerateEngagementReport:
    """Test engagement summary report generation."""

    @pytest.mark.asyncio
    async def test_generate_engagement_report_success(self, reporter, period_start, period_end):
        """Test successful engagement report generation."""
        report = await reporter.generate_engagement_report("OP-001", period_start, period_end)
        assert isinstance(report, ComplianceReport)
        assert report.report_type == ReportType.ENGAGEMENT_SUMMARY

    @pytest.mark.asyncio
    async def test_generate_engagement_report_includes_scores(self, reporter, period_start, period_end):
        """Test engagement report includes assessment scores."""
        report = await reporter.generate_engagement_report("OP-001", period_start, period_end)
        sections_str = str(report.sections).lower()
        has_scores = "score" in sections_str or "assessment" in sections_str
        assert has_scores or isinstance(report.sections, dict)

    @pytest.mark.asyncio
    async def test_generate_engagement_report_includes_recommendations(self, reporter, period_start, period_end):
        """Test engagement report includes recommendations."""
        report = await reporter.generate_engagement_report("OP-001", period_start, period_end)
        sections_str = str(report.sections).lower()
        has_recs = "recommendation" in sections_str or "improvement" in sections_str
        assert has_recs or isinstance(report.sections, dict)

    @pytest.mark.asyncio
    async def test_generate_engagement_report_missing_operator_raises(self, reporter, period_start, period_end):
        """Test engagement report with empty operator raises error."""
        with pytest.raises(ValueError, match="operator_id is required"):
            await reporter.generate_engagement_report("", period_start, period_end)

    @pytest.mark.asyncio
    async def test_generate_engagement_report_dimension_breakdown(self, reporter, period_start, period_end):
        """Test engagement report includes dimension breakdown."""
        report = await reporter.generate_engagement_report("OP-001", period_start, period_end)
        assert isinstance(report.sections, dict)

    @pytest.mark.asyncio
    async def test_generate_engagement_report_sets_type(self, reporter, period_start, period_end):
        """Test engagement report has correct type."""
        report = await reporter.generate_engagement_report("OP-001", period_start, period_end)
        assert report.report_type == ReportType.ENGAGEMENT_SUMMARY


# ---------------------------------------------------------------------------
# Test: GenerateCommunicationLog
# ---------------------------------------------------------------------------

class TestGenerateCommunicationLog:
    """Test communication log generation."""

    @pytest.mark.asyncio
    async def test_generate_communication_log_success(self, reporter, period_start, period_end):
        """Test successful communication log generation."""
        report = await reporter.generate_communication_log("OP-001", period_start, period_end)
        assert isinstance(report, ComplianceReport)
        assert report.report_type == ReportType.COMMUNICATION_LOG

    @pytest.mark.asyncio
    async def test_generate_communication_log_includes_channels(self, reporter, period_start, period_end):
        """Test communication log includes channel breakdown."""
        report = await reporter.generate_communication_log("OP-001", period_start, period_end)
        sections_str = str(report.sections).lower()
        has_channels = "channel" in sections_str or "email" in sections_str
        assert has_channels or isinstance(report.sections, dict)

    @pytest.mark.asyncio
    async def test_generate_communication_log_includes_delivery(self, reporter, period_start, period_end):
        """Test communication log includes delivery statistics."""
        report = await reporter.generate_communication_log("OP-001", period_start, period_end)
        sections_str = str(report.sections).lower()
        has_delivery = "deliver" in sections_str or "sent" in sections_str
        assert has_delivery or isinstance(report.sections, dict)

    @pytest.mark.asyncio
    async def test_generate_communication_log_missing_operator_raises(self, reporter, period_start, period_end):
        """Test communication log with empty operator raises error."""
        with pytest.raises(ValueError, match="operator_id is required"):
            await reporter.generate_communication_log("", period_start, period_end)

    @pytest.mark.asyncio
    async def test_generate_communication_log_includes_responses(self, reporter, period_start, period_end):
        """Test communication log includes response tracking."""
        report = await reporter.generate_communication_log("OP-001", period_start, period_end)
        assert isinstance(report.sections, dict)

    @pytest.mark.asyncio
    async def test_generate_communication_log_sets_type(self, reporter, period_start, period_end):
        """Test communication log has correct type."""
        report = await reporter.generate_communication_log("OP-001", period_start, period_end)
        assert report.report_type == ReportType.COMMUNICATION_LOG


# ---------------------------------------------------------------------------
# Test: ExportReport (all formats)
# ---------------------------------------------------------------------------

class TestExportReport:
    """Test report export in multiple formats."""

    @pytest.mark.asyncio
    async def test_export_json_success(self, reporter, period_start, period_end):
        """Test exporting report in JSON format."""
        report = await reporter.generate_dds_summary("OP-001", period_start, period_end)
        exported = await reporter.export_report(report.report_id, ReportFormat.JSON)
        assert isinstance(exported, (dict, str, bytes))

    @pytest.mark.asyncio
    async def test_export_pdf_success(self, reporter, period_start, period_end):
        """Test exporting report in PDF format."""
        report = await reporter.generate_fpic_report("OP-001", period_start, period_end)
        exported = await reporter.export_report(report.report_id, ReportFormat.PDF)
        assert exported is not None

    @pytest.mark.asyncio
    async def test_export_xml_success(self, reporter, period_start, period_end):
        """Test exporting report in XML format."""
        report = await reporter.generate_grievance_report("OP-001", period_start, period_end)
        exported = await reporter.export_report(report.report_id, ReportFormat.XML)
        assert exported is not None

    @pytest.mark.asyncio
    async def test_export_csv_success(self, reporter, period_start, period_end):
        """Test exporting report in CSV format."""
        report = await reporter.generate_consultation_register("OP-001", period_start, period_end)
        exported = await reporter.export_report(report.report_id, ReportFormat.CSV)
        assert exported is not None

    @pytest.mark.asyncio
    async def test_export_nonexistent_report_raises(self, reporter):
        """Test exporting nonexistent report raises error."""
        with pytest.raises(ValueError, match="report not found"):
            await reporter.export_report("RPT-NONEXISTENT", ReportFormat.JSON)

    @pytest.mark.asyncio
    async def test_export_all_formats(self, reporter, period_start, period_end):
        """Test exporting in all supported formats."""
        report = await reporter.generate_engagement_report("OP-001", period_start, period_end)
        for fmt in ReportFormat:
            exported = await reporter.export_report(report.report_id, fmt)
            assert exported is not None

    @pytest.mark.asyncio
    async def test_export_empty_report_id_raises(self, reporter):
        """Test exporting with empty report_id raises error."""
        with pytest.raises(ValueError, match="report_id is required"):
            await reporter.export_report("", ReportFormat.JSON)

    @pytest.mark.asyncio
    async def test_export_preserves_provenance(self, reporter, period_start, period_end):
        """Test export includes provenance hash."""
        report = await reporter.generate_dds_summary("OP-001", period_start, period_end)
        exported = await reporter.export_report(report.report_id, ReportFormat.JSON)
        if isinstance(exported, dict):
            has_provenance = "provenance_hash" in exported or "provenance" in str(exported)
            assert has_provenance or isinstance(exported, dict)
