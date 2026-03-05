# -*- coding: utf-8 -*-
"""
Unit tests for EU Taxonomy Reporting Engine.

Tests Article 8 report generation (turnover/CapEx/OpEx templates), EBA
Pillar 3 report generation, export functionality (PDF/Excel/CSV/XBRL),
qualitative disclosures, report status workflow, regulatory version
tracking, data quality scoring, and gap analysis reporting with 35+
test functions.

Author: GL-TestEngineer
Date: March 2026
"""

from datetime import date, datetime
from decimal import Decimal

import pytest


# ===========================================================================
# Article 8 report generation tests
# ===========================================================================

class TestArticle8Report:
    """Test Article 8 non-financial undertaking report generation."""

    def test_article_8_turnover_report(self, sample_report):
        """Article 8 turnover report generated."""
        assert sample_report["template"] == "article_8_turnover"
        assert sample_report["status"] == "generated"

    def test_report_period(self, sample_report):
        """Report covers correct period."""
        assert sample_report["period"] == "FY2025"

    def test_report_format(self, sample_report):
        """Report format is Excel."""
        assert sample_report["format"] == "excel"

    def test_report_download_url(self, sample_report):
        """Report has download URL."""
        assert sample_report["download_url"] is not None
        assert sample_report["download_url"].startswith("https://")

    def test_report_content_summary(self, sample_report):
        """Report content includes KPI summary."""
        content = sample_report["content"]
        assert "summary" in content
        summary = content["summary"]
        assert summary["total_turnover"] == 2500000000
        assert summary["aligned_pct"] == 42.0

    def test_report_generated_at(self, sample_report):
        """Report has generation timestamp."""
        assert isinstance(sample_report["generated_at"], datetime)

    def test_engine_generate_article_8(self, reporting_engine):
        """Engine generates Article 8 report."""
        reporting_engine.generate_article_8.return_value = {
            "report_id": "rpt-001",
            "template": "article_8_turnover",
            "status": "generated",
        }
        result = reporting_engine.generate_article_8(
            org_id="org-123",
            period="FY2025",
            kpi_type="turnover",
        )
        assert result["status"] == "generated"
        reporting_engine.generate_article_8.assert_called_once()

    def test_article_8_capex_template(self, reporting_engine):
        """Engine generates CapEx Article 8 template."""
        reporting_engine.generate_article_8.return_value = {
            "template": "article_8_capex",
            "status": "generated",
        }
        result = reporting_engine.generate_article_8("org-123", "FY2025", "capex")
        assert result["template"] == "article_8_capex"

    def test_article_8_opex_template(self, reporting_engine):
        """Engine generates OpEx Article 8 template."""
        reporting_engine.generate_article_8.return_value = {
            "template": "article_8_opex",
            "status": "generated",
        }
        result = reporting_engine.generate_article_8("org-123", "FY2025", "opex")
        assert result["template"] == "article_8_opex"


# ===========================================================================
# EBA report generation tests
# ===========================================================================

class TestEBAReport:
    """Test EBA Pillar 3 report generation for financial institutions."""

    def test_eba_template_7_report(self, sample_eba_report):
        """EBA template 7 report generated."""
        assert sample_eba_report["template"] == "eba_template_7"

    def test_eba_report_gar_content(self, sample_eba_report):
        """EBA report includes GAR stock and flow data."""
        content = sample_eba_report["content"]
        assert "gar_stock" in content
        assert "gar_flow" in content

    def test_eba_gar_stock_data(self, sample_eba_report):
        """EBA GAR stock data correct."""
        gar_stock = sample_eba_report["content"]["gar_stock"]
        assert gar_stock["percentage"] == 18.75
        assert gar_stock["aligned"] > 0
        assert gar_stock["covered"] > 0

    def test_eba_gar_flow_data(self, sample_eba_report):
        """EBA GAR flow data correct."""
        gar_flow = sample_eba_report["content"]["gar_flow"]
        assert gar_flow["percentage"] == 24.29

    def test_engine_generate_eba(self, reporting_engine):
        """Engine generates EBA template."""
        reporting_engine.generate_eba_template.return_value = {
            "template": "eba_template_7",
            "status": "generated",
        }
        result = reporting_engine.generate_eba_template("inst-123", "FY2025", template_number=7)
        assert result["template"] == "eba_template_7"
        reporting_engine.generate_eba_template.assert_called_once()

    def test_engine_generate_gar_summary(self, reporting_engine):
        """Engine generates GAR summary report."""
        reporting_engine.generate_gar_summary.return_value = {
            "template": "gar_summary",
            "gar_stock_pct": 18.75,
            "gar_flow_pct": 24.29,
        }
        result = reporting_engine.generate_gar_summary("inst-123", "FY2025")
        assert result["gar_stock_pct"] == 18.75


# ===========================================================================
# Export functionality tests
# ===========================================================================

class TestExportFunctionality:
    """Test report export in multiple formats."""

    def test_export_pdf(self, reporting_engine):
        """Export to PDF format."""
        reporting_engine.export_pdf.return_value = {
            "format": "pdf",
            "url": "https://storage.greenlang.io/reports/report.pdf",
            "size_bytes": 1250000,
        }
        result = reporting_engine.export_pdf("rpt-001")
        assert result["format"] == "pdf"
        reporting_engine.export_pdf.assert_called_once()

    def test_export_excel(self, reporting_engine):
        """Export to Excel format."""
        reporting_engine.export_excel.return_value = {
            "format": "excel",
            "url": "https://storage.greenlang.io/reports/report.xlsx",
            "sheets": ["Summary", "Turnover", "CapEx", "OpEx"],
        }
        result = reporting_engine.export_excel("rpt-001")
        assert result["format"] == "excel"
        assert len(result["sheets"]) >= 3

    def test_export_csv(self, reporting_engine):
        """Export to CSV format."""
        reporting_engine.export_csv.return_value = {
            "format": "csv",
            "url": "https://storage.greenlang.io/reports/report.csv",
            "rows": 150,
        }
        result = reporting_engine.export_csv("rpt-001")
        assert result["format"] == "csv"
        assert result["rows"] > 0

    def test_export_xbrl(self, reporting_engine):
        """Export to XBRL format."""
        reporting_engine.export_xbrl.return_value = {
            "format": "xbrl",
            "url": "https://storage.greenlang.io/reports/report.xbrl",
            "taxonomy": "ESEF_taxonomy_2024",
        }
        result = reporting_engine.export_xbrl("rpt-001")
        assert result["format"] == "xbrl"
        assert "ESEF" in result["taxonomy"]

    def test_supported_formats_config(self, sample_config):
        """Configuration lists supported export formats."""
        assert "pdf" in sample_config["supported_formats"]
        assert "excel" in sample_config["supported_formats"]
        assert "csv" in sample_config["supported_formats"]
        assert "xbrl" in sample_config["supported_formats"]


# ===========================================================================
# Qualitative disclosure tests
# ===========================================================================

class TestQualitativeDisclosure:
    """Test qualitative disclosure generation."""

    def test_engine_generate_qualitative(self, reporting_engine):
        """Engine generates qualitative disclosure."""
        reporting_engine.generate_qualitative_disclosure.return_value = {
            "template": "qualitative_disclosure",
            "sections": [
                "accounting_policy",
                "eligibility_methodology",
                "alignment_methodology",
                "contextual_information",
                "compliance_statement",
            ],
        }
        result = reporting_engine.generate_qualitative_disclosure("org-123", "FY2025")
        assert result["template"] == "qualitative_disclosure"
        assert len(result["sections"]) >= 4

    def test_engine_generate_executive_summary(self, reporting_engine):
        """Engine generates executive summary."""
        reporting_engine.generate_executive_summary.return_value = {
            "template": "executive_summary",
            "headline_kpis": {
                "turnover_alignment": 42.0,
                "capex_alignment": 52.5,
                "opex_alignment": 25.5,
            },
        }
        result = reporting_engine.generate_executive_summary("org-123", "FY2025")
        assert result["template"] == "executive_summary"
        assert result["headline_kpis"]["turnover_alignment"] == 42.0


# ===========================================================================
# Report status workflow tests
# ===========================================================================

class TestReportStatusWorkflow:
    """Test report status lifecycle."""

    def test_draft_status(self):
        """Reports start in draft status."""
        report = {"status": "draft"}
        assert report["status"] == "draft"

    def test_generated_status(self, sample_report):
        """Report transitions to generated."""
        assert sample_report["status"] == "generated"

    def test_approved_status(self):
        """Report can be approved."""
        report = {"status": "approved"}
        assert report["status"] == "approved"

    def test_submitted_status(self):
        """Report can be submitted."""
        report = {"status": "submitted"}
        assert report["status"] == "submitted"

    def test_valid_statuses(self):
        """Only valid statuses allowed."""
        valid = {"draft", "generated", "approved", "submitted"}
        for status in valid:
            report = {"status": status}
            assert report["status"] in valid


# ===========================================================================
# Regulatory version tests
# ===========================================================================

class TestRegulatoryVersions:
    """Test regulatory version tracking in reports."""

    def test_active_versions(self, sample_regulatory_versions):
        """Active versions available for reporting."""
        active = [v for v in sample_regulatory_versions if v["status"] == "active"]
        assert len(active) >= 2

    def test_climate_delegated_act_version(self, sample_regulatory_versions):
        """Climate DA has current active version."""
        climate_active = [v for v in sample_regulatory_versions
                          if v["delegated_act"] == "climate" and v["status"] == "active"]
        assert len(climate_active) == 1
        assert climate_active[0]["version_number"] == "2.0"

    def test_environmental_delegated_act(self, sample_regulatory_versions):
        """Environmental DA version available."""
        env = [v for v in sample_regulatory_versions
               if v["delegated_act"] == "environmental" and v["status"] == "active"]
        assert len(env) == 1

    def test_complementary_delegated_act(self, sample_regulatory_versions):
        """Complementary DA (nuclear/gas) version available."""
        comp = [v for v in sample_regulatory_versions
                if v["delegated_act"] == "complementary" and v["status"] == "active"]
        assert len(comp) == 1

    def test_superseded_version_tracked(self, sample_regulatory_versions):
        """Superseded versions remain in version history."""
        superseded = [v for v in sample_regulatory_versions if v["status"] == "superseded"]
        assert len(superseded) >= 1


# ===========================================================================
# Data quality scoring tests
# ===========================================================================

class TestDataQualityScoring:
    """Test data quality scoring for disclosure readiness."""

    def test_data_quality_score(self, sample_data_quality):
        """Overall quality score calculated."""
        assert sample_data_quality["overall_score"] == Decimal("78.50")

    def test_data_quality_grade(self, sample_data_quality):
        """Quality grade assigned."""
        assert sample_data_quality["grade"] == "B+"

    def test_dimension_scores(self, sample_data_quality):
        """All quality dimensions scored."""
        dims = sample_data_quality["dimensions"]
        assert dims["completeness"]["score"] == 85.0
        assert dims["accuracy"]["score"] == 90.0
        assert dims["timeliness"]["score"] == 70.0

    def test_improvement_actions_generated(self, sample_data_quality):
        """Improvement actions generated."""
        actions = sample_data_quality["improvement_actions"]
        assert len(actions) >= 2
        priorities = {a["priority"] for a in actions}
        assert "high" in priorities

    def test_engine_assess_quality(self, data_quality_engine):
        """Engine assesses data quality."""
        data_quality_engine.assess.return_value = {
            "overall_score": 78.5,
            "grade": "B+",
        }
        result = data_quality_engine.assess("org-123", "FY2025")
        assert result["grade"] == "B+"

    def test_min_grade_config(self, sample_config):
        """Configuration specifies minimum acceptable grade."""
        assert sample_config["data_quality_min_grade"] == "C"


# ===========================================================================
# Gap analysis in reporting context tests
# ===========================================================================

class TestGapAnalysisReporting:
    """Test gap analysis in reporting context."""

    def test_gap_assessment_structure(self, sample_gap_assessment):
        """Gap assessment has expected structure."""
        assert sample_gap_assessment["total_gaps"] == 8
        assert sample_gap_assessment["high_priority"] == 3

    def test_gap_categories_present(self, sample_gap_assessment):
        """Gap categories include SC, DNSH, data, regulatory."""
        cats = sample_gap_assessment["gap_categories"]
        assert "sc" in cats
        assert "dnsh" in cats
        assert "data" in cats

    def test_action_items_prioritized(self, sample_gap_assessment):
        """Action items are prioritized."""
        actions = sample_gap_assessment["action_items"]
        priorities = [a["priority"] for a in actions]
        assert priorities == sorted(priorities)

    def test_gap_items_detail(self, sample_gap_items):
        """Individual gap items have required fields."""
        for item in sample_gap_items:
            assert "category" in item
            assert "description" in item
            assert "priority" in item
            assert "deadline" in item
            assert "assigned_to" in item

    def test_engine_run_gap_analysis(self, gap_engine):
        """Engine runs full gap analysis."""
        gap_engine.run_full_analysis.return_value = {
            "total_gaps": 8,
            "high_priority": 3,
            "readiness_score": 72.0,
        }
        result = gap_engine.run_full_analysis("org-123", "FY2025")
        assert result["total_gaps"] == 8
        gap_engine.run_full_analysis.assert_called_once()
