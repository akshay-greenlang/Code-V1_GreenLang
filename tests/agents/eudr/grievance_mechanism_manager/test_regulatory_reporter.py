# -*- coding: utf-8 -*-
"""
Unit tests for RegulatoryReporter - AGENT-EUDR-032

Tests report generation for all 4 report types (EUDR Article 16,
CSDDD Article 8, UNGP Effectiveness, Annual Summary), section building,
statistics computation, retrieval, listing, and health checks.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from greenlang.agents.eudr.grievance_mechanism_manager.config import (
    GrievanceMechanismManagerConfig,
)
from greenlang.agents.eudr.grievance_mechanism_manager.regulatory_reporter import (
    RegulatoryReporter,
)
from greenlang.agents.eudr.grievance_mechanism_manager.models import (
    RegulatoryReport,
    RegulatoryReportType,
    ReportSection,
)


@pytest.fixture
def config():
    return GrievanceMechanismManagerConfig()


@pytest.fixture
def reporter(config):
    return RegulatoryReporter(config=config)


@pytest.fixture
def mixed_grievances():
    """Grievances with varied statuses and categories."""
    return [
        {"grievance_id": "g-001", "category": "environmental", "severity": "high", "status": "resolved", "channel": "web_portal"},
        {"grievance_id": "g-002", "category": "environmental", "severity": "medium", "status": "resolved", "channel": "email"},
        {"grievance_id": "g-003", "category": "human_rights", "severity": "critical", "status": "submitted", "channel": "hotline"},
        {"grievance_id": "g-004", "category": "labor", "severity": "high", "status": "closed", "channel": "mobile_app"},
        {"grievance_id": "g-005", "category": "environmental", "severity": "high", "status": "under_investigation", "channel": "community_meeting"},
    ]


@pytest.fixture
def remediations():
    return [
        {"remediation_id": "rem-001", "status": "verified"},
        {"remediation_id": "rem-002", "status": "verified"},
        {"remediation_id": "rem-003", "status": "in_progress"},
    ]


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInit:
    def test_reporter_created(self, reporter):
        assert reporter is not None

    def test_default_config(self):
        r = RegulatoryReporter()
        assert r.config is not None

    def test_empty_reports(self, reporter):
        assert len(reporter._reports) == 0


# ---------------------------------------------------------------------------
# Generate Report - Annual Summary
# ---------------------------------------------------------------------------


class TestGenerateAnnualSummary:
    @pytest.mark.asyncio
    async def test_returns_report(self, reporter, mixed_grievances):
        report = await reporter.generate_report(
            "OP-001", "annual_summary", mixed_grievances,
        )
        assert isinstance(report, RegulatoryReport)

    @pytest.mark.asyncio
    async def test_report_type(self, reporter, mixed_grievances):
        report = await reporter.generate_report(
            "OP-001", "annual_summary", mixed_grievances,
        )
        assert report.report_type == RegulatoryReportType.ANNUAL_SUMMARY

    @pytest.mark.asyncio
    async def test_total_grievances(self, reporter, mixed_grievances):
        report = await reporter.generate_report(
            "OP-001", "annual_summary", mixed_grievances,
        )
        assert report.total_grievances == 5

    @pytest.mark.asyncio
    async def test_resolved_count(self, reporter, mixed_grievances):
        report = await reporter.generate_report(
            "OP-001", "annual_summary", mixed_grievances,
        )
        assert report.resolved_count == 3  # resolved + closed

    @pytest.mark.asyncio
    async def test_unresolved_count(self, reporter, mixed_grievances):
        report = await reporter.generate_report(
            "OP-001", "annual_summary", mixed_grievances,
        )
        assert report.unresolved_count == 2

    @pytest.mark.asyncio
    async def test_top_categories(self, reporter, mixed_grievances):
        report = await reporter.generate_report(
            "OP-001", "annual_summary", mixed_grievances,
        )
        assert len(report.top_categories) > 0
        assert report.top_categories[0]["category"] == "environmental"

    @pytest.mark.asyncio
    async def test_sections_present(self, reporter, mixed_grievances):
        report = await reporter.generate_report(
            "OP-001", "annual_summary", mixed_grievances,
        )
        assert len(report.sections) >= 4

    @pytest.mark.asyncio
    async def test_executive_summary_section(self, reporter, mixed_grievances):
        report = await reporter.generate_report(
            "OP-001", "annual_summary", mixed_grievances,
        )
        titles = [s.title for s in report.sections]
        assert "Executive Summary" in titles

    @pytest.mark.asyncio
    async def test_provenance_hash_set(self, reporter, mixed_grievances):
        report = await reporter.generate_report(
            "OP-001", "annual_summary", mixed_grievances,
        )
        assert len(report.provenance_hash) == 64

    @pytest.mark.asyncio
    async def test_record_stored(self, reporter, mixed_grievances):
        report = await reporter.generate_report(
            "OP-001", "annual_summary", mixed_grievances,
        )
        assert report.report_id in reporter._reports

    @pytest.mark.asyncio
    async def test_accessibility_score(self, reporter, mixed_grievances):
        report = await reporter.generate_report(
            "OP-001", "annual_summary", mixed_grievances,
        )
        assert report.accessibility_score > Decimal("0")

    @pytest.mark.asyncio
    async def test_average_resolution_days(self, reporter, mixed_grievances):
        report = await reporter.generate_report(
            "OP-001", "annual_summary", mixed_grievances,
        )
        assert report.average_resolution_days == Decimal("21")


# ---------------------------------------------------------------------------
# Generate Report - EUDR Article 16
# ---------------------------------------------------------------------------


class TestGenerateEUDRArticle16:
    @pytest.mark.asyncio
    async def test_report_type(self, reporter, mixed_grievances):
        report = await reporter.generate_report(
            "OP-001", "eudr_article16", mixed_grievances,
        )
        assert report.report_type == RegulatoryReportType.EUDR_ARTICLE16

    @pytest.mark.asyncio
    async def test_eudr_specific_section(self, reporter, mixed_grievances):
        report = await reporter.generate_report(
            "OP-001", "eudr_article16", mixed_grievances,
        )
        titles = [s.title for s in report.sections]
        assert "EUDR Article 16 Compliance" in titles

    @pytest.mark.asyncio
    async def test_eudr_section_content(self, reporter, mixed_grievances):
        report = await reporter.generate_report(
            "OP-001", "eudr_article16", mixed_grievances,
        )
        eudr_section = next(
            s for s in report.sections if s.title == "EUDR Article 16 Compliance"
        )
        assert eudr_section.content["mechanism_accessible"] is True
        assert eudr_section.regulatory_reference == "EUDR Article 16"


# ---------------------------------------------------------------------------
# Generate Report - CSDDD Article 8
# ---------------------------------------------------------------------------


class TestGenerateCSDDDArticle8:
    @pytest.mark.asyncio
    async def test_report_type(self, reporter, mixed_grievances):
        report = await reporter.generate_report(
            "OP-001", "csddd_article8", mixed_grievances,
        )
        assert report.report_type == RegulatoryReportType.CSDDD_ARTICLE8

    @pytest.mark.asyncio
    async def test_csddd_specific_section(self, reporter, mixed_grievances):
        report = await reporter.generate_report(
            "OP-001", "csddd_article8", mixed_grievances,
        )
        titles = [s.title for s in report.sections]
        assert "CSDDD Article 8 Compliance" in titles

    @pytest.mark.asyncio
    async def test_csddd_section_content(self, reporter, mixed_grievances):
        report = await reporter.generate_report(
            "OP-001", "csddd_article8", mixed_grievances,
        )
        section = next(
            s for s in report.sections if s.title == "CSDDD Article 8 Compliance"
        )
        assert section.content["mechanism_legitimate"] is True
        assert section.content["mechanism_equitable"] is True


# ---------------------------------------------------------------------------
# Generate Report - UNGP Effectiveness
# ---------------------------------------------------------------------------


class TestGenerateUNGPEffectiveness:
    @pytest.mark.asyncio
    async def test_report_type(self, reporter, mixed_grievances):
        report = await reporter.generate_report(
            "OP-001", "ungp_effectiveness", mixed_grievances,
        )
        assert report.report_type == RegulatoryReportType.UNGP_EFFECTIVENESS

    @pytest.mark.asyncio
    async def test_ungp_specific_section(self, reporter, mixed_grievances):
        report = await reporter.generate_report(
            "OP-001", "ungp_effectiveness", mixed_grievances,
        )
        titles = [s.title for s in report.sections]
        assert "UNGP Principle 31 Effectiveness" in titles

    @pytest.mark.asyncio
    async def test_ungp_seven_criteria(self, reporter, mixed_grievances):
        report = await reporter.generate_report(
            "OP-001", "ungp_effectiveness", mixed_grievances,
        )
        section = next(
            s for s in report.sections if s.title == "UNGP Principle 31 Effectiveness"
        )
        assert len(section.content) == 7
        assert section.content["legitimate"] is True
        assert section.content["continuous_learning"] is True


# ---------------------------------------------------------------------------
# Invalid Report Type
# ---------------------------------------------------------------------------


class TestInvalidReportType:
    @pytest.mark.asyncio
    async def test_invalid_type_defaults_to_annual(self, reporter, mixed_grievances):
        report = await reporter.generate_report(
            "OP-001", "invalid_type", mixed_grievances,
        )
        assert report.report_type == RegulatoryReportType.ANNUAL_SUMMARY


# ---------------------------------------------------------------------------
# Empty / Null Inputs
# ---------------------------------------------------------------------------


class TestEmptyInputs:
    @pytest.mark.asyncio
    async def test_no_grievances(self, reporter):
        report = await reporter.generate_report("OP-001", "annual_summary")
        assert report.total_grievances == 0
        assert report.resolved_count == 0

    @pytest.mark.asyncio
    async def test_custom_period(self, reporter, mixed_grievances):
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=90)
        report = await reporter.generate_report(
            "OP-001", "annual_summary", mixed_grievances,
            period_start=start, period_end=now,
        )
        assert report.reporting_period_start == start


# ---------------------------------------------------------------------------
# Remediation Effectiveness
# ---------------------------------------------------------------------------


class TestRemediationEffectiveness:
    @pytest.mark.asyncio
    async def test_with_remediations(self, reporter, mixed_grievances, remediations):
        report = await reporter.generate_report(
            "OP-001", "annual_summary", mixed_grievances,
            remediations=remediations,
        )
        # 2 verified out of 3 = ~66.67%
        assert report.remediation_effectiveness > Decimal("0")

    @pytest.mark.asyncio
    async def test_no_remediations(self, reporter, mixed_grievances):
        report = await reporter.generate_report(
            "OP-001", "annual_summary", mixed_grievances,
        )
        assert report.remediation_effectiveness == Decimal("0")


# ---------------------------------------------------------------------------
# Section Building
# ---------------------------------------------------------------------------


class TestSectionBuilding:
    @pytest.mark.asyncio
    async def test_always_has_executive_summary(self, reporter, mixed_grievances):
        for rtype in ["annual_summary", "eudr_article16", "csddd_article8", "ungp_effectiveness"]:
            report = await reporter.generate_report("OP-001", rtype, mixed_grievances)
            titles = [s.title for s in report.sections]
            assert "Executive Summary" in titles

    @pytest.mark.asyncio
    async def test_always_has_categories_section(self, reporter, mixed_grievances):
        report = await reporter.generate_report("OP-001", "annual_summary", mixed_grievances)
        titles = [s.title for s in report.sections]
        assert "Grievance Categories" in titles

    @pytest.mark.asyncio
    async def test_always_has_root_cause_section(self, reporter, mixed_grievances):
        report = await reporter.generate_report("OP-001", "annual_summary", mixed_grievances)
        titles = [s.title for s in report.sections]
        assert "Root Cause Analysis Summary" in titles

    @pytest.mark.asyncio
    async def test_always_has_remediation_section(self, reporter, mixed_grievances):
        report = await reporter.generate_report("OP-001", "annual_summary", mixed_grievances)
        titles = [s.title for s in report.sections]
        assert "Remediation Effectiveness" in titles

    @pytest.mark.asyncio
    async def test_regulatory_references_set(self, reporter, mixed_grievances):
        report = await reporter.generate_report("OP-001", "annual_summary", mixed_grievances)
        for section in report.sections:
            assert len(section.regulatory_reference) > 0


# ---------------------------------------------------------------------------
# Retrieval and Listing
# ---------------------------------------------------------------------------


class TestRetrievalAndListing:
    @pytest.mark.asyncio
    async def test_get_report(self, reporter, mixed_grievances):
        report = await reporter.generate_report("OP-001", "annual_summary", mixed_grievances)
        retrieved = await reporter.get_report(report.report_id)
        assert retrieved is not None

    @pytest.mark.asyncio
    async def test_get_not_found(self, reporter):
        result = await reporter.get_report("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_all(self, reporter, mixed_grievances):
        await reporter.generate_report("OP-001", "annual_summary", mixed_grievances)
        await reporter.generate_report("OP-002", "eudr_article16", mixed_grievances)
        results = await reporter.list_reports()
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_list_filter_operator(self, reporter, mixed_grievances):
        await reporter.generate_report("OP-001", "annual_summary", mixed_grievances)
        await reporter.generate_report("OP-002", "annual_summary", mixed_grievances)
        results = await reporter.list_reports(operator_id="OP-001")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_list_filter_type(self, reporter, mixed_grievances):
        await reporter.generate_report("OP-001", "annual_summary", mixed_grievances)
        await reporter.generate_report("OP-001", "eudr_article16", mixed_grievances)
        results = await reporter.list_reports(report_type="eudr_article16")
        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_list_empty(self, reporter):
        results = await reporter.list_reports()
        assert results == []


# ---------------------------------------------------------------------------
# Health Check
# ---------------------------------------------------------------------------


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_health_check(self, reporter):
        health = await reporter.health_check()
        assert health["status"] == "healthy"
        assert health["engine"] == "RegulatoryReporter"

    @pytest.mark.asyncio
    async def test_health_check_count(self, reporter, mixed_grievances):
        await reporter.generate_report("OP-001", "annual_summary", mixed_grievances)
        health = await reporter.health_check()
        assert health["report_count"] == 1
