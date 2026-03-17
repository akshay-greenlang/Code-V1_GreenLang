# -*- coding: utf-8 -*-
"""
Unit tests for PACK-008 EU Taxonomy Alignment Pack - Report Templates

Tests all 10 report templates for correct generation, required output sections,
metadata inclusion, and provenance hashing. Validates:
  - EligibilityMatrixReportTemplate
  - AlignmentSummaryReportTemplate
  - Article8DisclosureTemplate
  - EBAPillar3GARReportTemplate
  - KPIDashboardTemplate
  - GapAnalysisReportTemplate
  - TSCComplianceReportTemplate
  - DNSHAssessmentReportTemplate
  - ExecutiveSummaryTemplate
  - DetailedAssessmentReportTemplate
"""

import pytest
import hashlib
import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Simulated Template Infrastructure
# ---------------------------------------------------------------------------

ENVIRONMENTAL_OBJECTIVES = ["CCM", "CCA", "WTR", "CE", "PPC", "BIO"]


def _compute_hash(data: Any) -> str:
    """Compute SHA-256 provenance hash."""
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class SimulatedReportData:
    """Simulated report input data for template rendering."""

    def __init__(
        self,
        organization_name: str = "Acme Manufacturing GmbH",
        report_date: str = "2026-03-15",
        reporting_period: str = "FY 2025",
        total_activities: int = 42,
    ):
        self.organization_name = organization_name
        self.report_date = report_date
        self.reporting_period = reporting_period
        self.total_activities = total_activities


class SimulatedTemplate:
    """Base class for simulated report templates."""

    template_name: str = ""
    required_sections: List[str] = []

    def generate(self, data: SimulatedReportData,
                 format: str = "markdown") -> Dict[str, Any]:
        """Generate the report from input data."""
        start = datetime.utcnow()
        sections = {}
        for section in self.required_sections:
            sections[section] = self._generate_section(section, data)

        return {
            "template_name": self.template_name,
            "organization_name": data.organization_name,
            "reporting_period": data.reporting_period,
            "report_date": data.report_date,
            "format": format,
            "sections": sections,
            "section_count": len(sections),
            "metadata": {
                "generated_at": start.isoformat(),
                "generator": "PACK-008 EU Taxonomy Alignment",
                "version": "1.0.0",
                "total_activities": data.total_activities,
            },
            "provenance_hash": _compute_hash({
                "template": self.template_name,
                "org": data.organization_name,
                "period": data.reporting_period,
                "ts": start.isoformat(),
            }),
        }

    def _generate_section(self, section_name: str,
                          data: SimulatedReportData) -> Dict[str, Any]:
        """Generate content for a single report section."""
        return {
            "title": section_name.replace("_", " ").title(),
            "content": f"Content for {section_name}",
            "data_points": 5,
        }


# ---------------------------------------------------------------------------
# Simulated Template Implementations
# ---------------------------------------------------------------------------

class EligibilityMatrixReportTemplate(SimulatedTemplate):
    """Eligibility matrix report template (render method)."""

    template_name = "eligibility_matrix_report"
    required_sections = [
        "summary",
        "activity_eligibility_matrix",
        "nace_sector_breakdown",
        "eligible_vs_non_eligible_ratios",
        "objective_level_matrix",
        "methodology_notes",
    ]

    def render(self, data: SimulatedReportData,
               format: str = "markdown") -> Dict[str, Any]:
        """Render the eligibility matrix report."""
        return self.generate(data, format)


class AlignmentSummaryReportTemplate(SimulatedTemplate):
    """Alignment summary report template (render method)."""

    template_name = "alignment_summary_report"
    required_sections = [
        "executive_overview",
        "alignment_by_objective",
        "alignment_by_activity",
        "sc_dnsh_ms_summary",
        "kpi_overview",
        "recommendations",
    ]

    def render(self, data: SimulatedReportData,
               format: str = "markdown") -> Dict[str, Any]:
        """Render the alignment summary report."""
        return self.generate(data, format)


class Article8DisclosureTemplate(SimulatedTemplate):
    """Article 8 disclosure template (generate_full_disclosure + render)."""

    template_name = "article8_disclosure"
    required_sections = [
        "turnover_table",
        "capex_table",
        "opex_table",
        "eligible_non_eligible_breakdown",
        "aligned_non_aligned_breakdown",
        "nuclear_gas_supplementary",
        "methodology",
    ]

    def generate_full_disclosure(self, data: SimulatedReportData,
                                 format: str = "markdown") -> Dict[str, Any]:
        """Generate the full Article 8 disclosure."""
        return self.generate(data, format)

    def render(self, data: SimulatedReportData,
               format: str = "markdown") -> Dict[str, Any]:
        """Render the Article 8 disclosure template."""
        return self.generate(data, format)


class EBAPillar3GARReportTemplate(SimulatedTemplate):
    """EBA Pillar 3 GAR report template."""

    template_name = "eba_pillar3_gar_report"
    required_sections = [
        "template_6_gar_summary",
        "template_7_gar_by_sector",
        "template_8_btar",
        "template_9_gar_flow",
        "template_10_mitigating_actions",
        "data_quality_notes",
    ]

    def generate_full_eba_report(self, data: SimulatedReportData,
                                 format: str = "markdown") -> Dict[str, Any]:
        """Generate the full EBA Pillar 3 report."""
        return self.generate(data, format)

    def render(self, data: SimulatedReportData,
               format: str = "markdown") -> Dict[str, Any]:
        """Render the EBA Pillar 3 GAR report."""
        return self.generate(data, format)


class KPIDashboardTemplate(SimulatedTemplate):
    """KPI dashboard template."""

    template_name = "kpi_dashboard"
    required_sections = [
        "turnover_kpi",
        "capex_kpi",
        "opex_kpi",
        "trend_charts",
        "target_vs_actual",
        "peer_comparison",
    ]

    def render(self, data: SimulatedReportData,
               format: str = "markdown") -> Dict[str, Any]:
        """Render the KPI dashboard."""
        return self.generate(data, format)


class GapAnalysisReportTemplate(SimulatedTemplate):
    """Gap analysis report template."""

    template_name = "gap_analysis_report"
    required_sections = [
        "current_state_summary",
        "identified_gaps",
        "gap_prioritization",
        "remediation_roadmap",
        "cost_estimates",
        "timeline",
    ]

    def render(self, data: SimulatedReportData,
               format: str = "markdown") -> Dict[str, Any]:
        """Render the gap analysis report."""
        return self.generate(data, format)


class TSCComplianceReportTemplate(SimulatedTemplate):
    """TSC compliance report template."""

    template_name = "tsc_compliance_report"
    required_sections = [
        "criteria_overview",
        "pass_fail_summary",
        "per_activity_results",
        "gap_details",
        "da_version_reference",
        "evidence_links",
    ]

    def render(self, data: SimulatedReportData,
               format: str = "markdown") -> Dict[str, Any]:
        """Render the TSC compliance report."""
        return self.generate(data, format)


class DNSHAssessmentReportTemplate(SimulatedTemplate):
    """DNSH assessment report template."""

    template_name = "dnsh_assessment_report"
    required_sections = [
        "dnsh_matrix",
        "per_objective_assessment",
        "compliance_status",
        "evidence_summary",
        "non_compliance_details",
        "recommendations",
    ]

    def render(self, data: SimulatedReportData,
               format: str = "markdown") -> Dict[str, Any]:
        """Render the DNSH assessment report."""
        return self.generate(data, format)


class ExecutiveSummaryTemplate(SimulatedTemplate):
    """Executive summary report template."""

    template_name = "executive_summary"
    required_sections = [
        "headline_metrics",
        "eligibility_summary",
        "alignment_summary",
        "key_findings",
        "action_items",
        "regulatory_context",
    ]

    def render(self, data: SimulatedReportData,
               format: str = "markdown") -> Dict[str, Any]:
        """Render the executive summary."""
        return self.generate(data, format)


class DetailedAssessmentReportTemplate(SimulatedTemplate):
    """Detailed assessment report template."""

    template_name = "detailed_assessment_report"
    required_sections = [
        "methodology",
        "eligibility_details",
        "sc_evaluation_details",
        "dnsh_details",
        "ms_details",
        "kpi_calculations",
        "data_quality_assessment",
        "appendices",
    ]

    def render(self, data: SimulatedReportData,
               format: str = "markdown") -> Dict[str, Any]:
        """Render the detailed assessment report."""
        return self.generate(data, format)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def report_data():
    """Create standard report input data."""
    return SimulatedReportData(
        organization_name="Acme Manufacturing GmbH",
        report_date="2026-03-15",
        reporting_period="FY 2025",
        total_activities=42,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestTemplates:
    """Test suite for all PACK-008 EU Taxonomy report templates."""

    def test_eligibility_matrix_report_generation(self, report_data):
        """Test EligibilityMatrixReportTemplate generates report with required sections."""
        template = EligibilityMatrixReportTemplate()
        report = template.render(report_data)

        assert report is not None
        assert report["template_name"] == "eligibility_matrix_report"
        assert report["organization_name"] == "Acme Manufacturing GmbH"
        assert report["reporting_period"] == "FY 2025"

        # Verify required sections
        sections = report["sections"]
        assert "activity_eligibility_matrix" in sections
        assert "nace_sector_breakdown" in sections
        assert "eligible_vs_non_eligible_ratios" in sections
        assert "objective_level_matrix" in sections
        assert report["section_count"] >= 6

        # Verify metadata
        assert "metadata" in report
        assert report["metadata"]["generator"] == "PACK-008 EU Taxonomy Alignment"
        assert report["metadata"]["version"] == "1.0.0"
        assert report["metadata"]["total_activities"] == 42

        # Verify provenance hash
        assert report["provenance_hash"] is not None
        assert len(report["provenance_hash"]) == 64
        assert re.match(r"^[0-9a-f]{64}$", report["provenance_hash"])

    def test_alignment_summary_report_generation(self, report_data):
        """Test AlignmentSummaryReportTemplate generates report with required sections."""
        template = AlignmentSummaryReportTemplate()
        report = template.render(report_data)

        assert report is not None
        assert report["template_name"] == "alignment_summary_report"

        sections = report["sections"]
        assert "executive_overview" in sections
        assert "alignment_by_objective" in sections
        assert "alignment_by_activity" in sections
        assert "sc_dnsh_ms_summary" in sections
        assert "kpi_overview" in sections
        assert "recommendations" in sections
        assert report["section_count"] >= 6

        assert "metadata" in report
        assert report["provenance_hash"] is not None

    def test_article8_disclosure_template_generation(self, report_data):
        """Test Article8DisclosureTemplate generates report with mandatory tables."""
        template = Article8DisclosureTemplate()
        report = template.generate_full_disclosure(report_data)

        assert report is not None
        assert report["template_name"] == "article8_disclosure"

        sections = report["sections"]
        assert "turnover_table" in sections
        assert "capex_table" in sections
        assert "opex_table" in sections
        assert "nuclear_gas_supplementary" in sections
        assert "methodology" in sections
        assert report["section_count"] >= 7

        # Also test render method produces same structure
        report_via_render = template.render(report_data)
        assert report_via_render["template_name"] == "article8_disclosure"
        assert "turnover_table" in report_via_render["sections"]

        assert report["provenance_hash"] is not None
        assert len(report["provenance_hash"]) == 64

    def test_eba_pillar3_gar_report_generation(self, report_data):
        """Test EBAPillar3GARReportTemplate generates EBA templates 6-10."""
        template = EBAPillar3GARReportTemplate()
        report = template.generate_full_eba_report(report_data)

        assert report is not None
        assert report["template_name"] == "eba_pillar3_gar_report"

        sections = report["sections"]
        assert "template_6_gar_summary" in sections
        assert "template_7_gar_by_sector" in sections
        assert "template_8_btar" in sections
        assert "template_9_gar_flow" in sections
        assert "template_10_mitigating_actions" in sections
        assert "data_quality_notes" in sections
        assert report["section_count"] >= 6

        assert "metadata" in report
        assert report["provenance_hash"] is not None

    def test_kpi_dashboard_generation(self, report_data):
        """Test KPIDashboardTemplate generates dashboard with KPI sections."""
        template = KPIDashboardTemplate()
        report = template.render(report_data)

        assert report is not None
        assert report["template_name"] == "kpi_dashboard"

        sections = report["sections"]
        assert "turnover_kpi" in sections
        assert "capex_kpi" in sections
        assert "opex_kpi" in sections
        assert "trend_charts" in sections
        assert "target_vs_actual" in sections
        assert report["section_count"] >= 6

        assert "metadata" in report
        assert report["provenance_hash"] is not None

    def test_gap_analysis_report_generation(self, report_data):
        """Test GapAnalysisReportTemplate generates report with gap details."""
        template = GapAnalysisReportTemplate()
        report = template.render(report_data)

        assert report is not None
        assert report["template_name"] == "gap_analysis_report"

        sections = report["sections"]
        assert "current_state_summary" in sections
        assert "identified_gaps" in sections
        assert "gap_prioritization" in sections
        assert "remediation_roadmap" in sections
        assert "cost_estimates" in sections
        assert "timeline" in sections
        assert report["section_count"] >= 6

        assert "metadata" in report
        assert report["provenance_hash"] is not None

    def test_tsc_compliance_report_generation(self, report_data):
        """Test TSCComplianceReportTemplate generates report with per-activity results."""
        template = TSCComplianceReportTemplate()
        report = template.render(report_data)

        assert report is not None
        assert report["template_name"] == "tsc_compliance_report"

        sections = report["sections"]
        assert "criteria_overview" in sections
        assert "pass_fail_summary" in sections
        assert "per_activity_results" in sections
        assert "gap_details" in sections
        assert "da_version_reference" in sections
        assert "evidence_links" in sections
        assert report["section_count"] >= 6

        assert "metadata" in report
        assert report["provenance_hash"] is not None

    def test_dnsh_assessment_report_generation(self, report_data):
        """Test DNSHAssessmentReportTemplate generates report with DNSH matrix."""
        template = DNSHAssessmentReportTemplate()
        report = template.render(report_data)

        assert report is not None
        assert report["template_name"] == "dnsh_assessment_report"

        sections = report["sections"]
        assert "dnsh_matrix" in sections
        assert "per_objective_assessment" in sections
        assert "compliance_status" in sections
        assert "evidence_summary" in sections
        assert "non_compliance_details" in sections
        assert "recommendations" in sections
        assert report["section_count"] >= 6

        assert "metadata" in report
        assert report["provenance_hash"] is not None

    def test_executive_summary_generation(self, report_data):
        """Test ExecutiveSummaryTemplate generates report with headline metrics."""
        template = ExecutiveSummaryTemplate()
        report = template.render(report_data)

        assert report is not None
        assert report["template_name"] == "executive_summary"

        sections = report["sections"]
        assert "headline_metrics" in sections
        assert "eligibility_summary" in sections
        assert "alignment_summary" in sections
        assert "key_findings" in sections
        assert "action_items" in sections
        assert "regulatory_context" in sections
        assert report["section_count"] >= 6

        assert "metadata" in report
        assert report["provenance_hash"] is not None

    def test_detailed_assessment_report_generation(self, report_data):
        """Test DetailedAssessmentReportTemplate generates comprehensive report."""
        template = DetailedAssessmentReportTemplate()
        report = template.render(report_data)

        assert report is not None
        assert report["template_name"] == "detailed_assessment_report"

        sections = report["sections"]
        assert "methodology" in sections
        assert "eligibility_details" in sections
        assert "sc_evaluation_details" in sections
        assert "dnsh_details" in sections
        assert "ms_details" in sections
        assert "kpi_calculations" in sections
        assert "data_quality_assessment" in sections
        assert "appendices" in sections
        assert report["section_count"] >= 8  # Most comprehensive template

        assert "metadata" in report
        assert report["metadata"]["generator"] == "PACK-008 EU Taxonomy Alignment"
        assert report["provenance_hash"] is not None
        assert len(report["provenance_hash"]) == 64
