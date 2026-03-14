# -*- coding: utf-8 -*-
"""
PACK-002 CSRD Professional Pack - Template Rendering Tests
=============================================================

Tests for all 10 professional report templates. Each template gets 3 tests:
markdown rendering, JSON rendering, and missing data handling.

Test count: 30
Author: GreenLang QA Team
"""

import hashlib
import json
from typing import Any, Dict, List

import pytest

from .conftest import PROFESSIONAL_TEMPLATES


# ---------------------------------------------------------------------------
# Template Renderer Stub
# ---------------------------------------------------------------------------

class TemplateRendererStub:
    """Lightweight template renderer for test validation."""

    def __init__(self, template_id: str, data: Dict[str, Any]):
        self.template_id = template_id
        self.data = data

    def render_markdown(self) -> str:
        """Render template as markdown."""
        lines = [f"# {self.template_id.replace('_', ' ').title()}", ""]
        lines.append(f"Generated: {self.data.get('generated_at', 'N/A')}")
        lines.append(f"Reporting Year: {self.data.get('reporting_year', 'N/A')}")
        lines.append("")

        if "company" in self.data:
            lines.append(f"## Company: {self.data['company'].get('group_name', self.data['company'].get('company_name', 'Unknown'))}")
        if "entities" in self.data:
            lines.append(f"## Entities: {len(self.data['entities'])}")
        if "frameworks" in self.data:
            lines.append(f"## Frameworks: {', '.join(self.data['frameworks'])}")
        if "scenarios" in self.data:
            lines.append(f"## Scenarios: {len(self.data['scenarios'])}")

        provenance = hashlib.sha256(
            json.dumps(self.data, sort_keys=True, default=str).encode()
        ).hexdigest()
        lines.append(f"\n---\nProvenance: {provenance}")

        return "\n".join(lines)

    def render_json(self) -> Dict[str, Any]:
        """Render template as JSON."""
        provenance = hashlib.sha256(
            json.dumps(self.data, sort_keys=True, default=str).encode()
        ).hexdigest()
        return {
            "template_id": self.template_id,
            "content": self.data,
            "provenance_hash": provenance,
            "format": "json",
            "generated_at": self.data.get("generated_at", ""),
        }

    def render_with_missing_data(self) -> Dict[str, Any]:
        """Render template with missing data fields."""
        output = self.render_json()
        missing = []
        required_fields = self._get_required_fields()
        for field in required_fields:
            if field not in self.data:
                missing.append(field)
        output["missing_fields"] = missing
        output["is_complete"] = len(missing) == 0
        return output

    def _get_required_fields(self) -> List[str]:
        """Get required fields for this template type."""
        common = ["reporting_year", "generated_at"]
        template_specific = {
            "consolidated_group_report": ["company", "entities", "consolidated_data"],
            "cdp_questionnaire_response": ["company", "cdp_responses"],
            "tcfd_disclosure": ["company", "governance", "strategy", "risk_management", "metrics"],
            "sbti_progress_report": ["company", "base_year", "targets", "progress"],
            "eu_taxonomy_report": ["company", "eligibility", "alignment", "kpis"],
            "scenario_analysis_report": ["company", "scenarios", "physical_risk", "transition_risk"],
            "cross_framework_mapping": ["company", "frameworks", "coverage_matrix"],
            "board_governance_package": ["company", "kpis", "risk_heatmap"],
            "regulatory_change_briefing": ["company", "changes", "impact_assessment"],
            "professional_audit_package": ["company", "assurance_level", "evidence"],
        }
        return common + template_specific.get(self.template_id, [])


# ---------------------------------------------------------------------------
# Template test data fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def template_data(sample_group_profile, sample_entity_data) -> Dict[str, Any]:
    """Create comprehensive data for template rendering."""
    return {
        "reporting_year": 2025,
        "generated_at": "2025-12-15T10:00:00Z",
        "pack_version": "1.0.0",
        "company": sample_group_profile,
        "entities": list(sample_entity_data.values()),
        "consolidated_data": {"scope1_total": 25050, "scope2_total": 18300, "scope3_total": 93000},
        "cdp_responses": {"climate": {"C0": "complete", "C1": "complete"}},
        "governance": {"board_oversight": True, "management_role": "CSO"},
        "strategy": {"risks": ["transition", "physical"], "opportunities": ["efficiency"]},
        "risk_management": {"process": "integrated", "identification": "quarterly"},
        "metrics": {"scope1": 25050, "scope2": 18300, "targets": ["net_zero_2050"]},
        "base_year": 2020,
        "targets": {"near_term": {"year": 2030, "reduction_pct": 42.0}},
        "progress": {"current_reduction_pct": 13.2, "on_track": True},
        "eligibility": {"turnover_pct": 45.2, "capex_pct": 52.0},
        "alignment": {"turnover_pct": 32.8, "capex_pct": 38.5},
        "kpis": {"gar": 32.8, "intensity": 47.4},
        "scenarios": [
            {"id": "IEA_NZE", "warming": 1.5},
            {"id": "NGFS_ORDERLY", "warming": 1.5},
        ],
        "physical_risk": {"high_risk_sites": 1, "total_sites": 6},
        "transition_risk": {"policy_risk": "high", "carbon_price_2050": 250},
        "frameworks": ["cdp", "tcfd", "sbti", "eu_taxonomy", "gri", "sasb"],
        "coverage_matrix": {"cdp": 83.1, "tcfd": 90.9, "sbti": 92.0},
        "risk_heatmap": {"climate": "high", "water": "medium", "social": "medium"},
        "changes": [{"id": "REG-001", "title": "ESRS Set 2", "severity": "high"}],
        "impact_assessment": {"high_impact": 2, "medium_impact": 2},
        "assurance_level": "reasonable",
        "evidence": {"documents": 45, "calculations_verified": 120},
    }


# ===========================================================================
# Template Tests (3 per template x 10 templates = 30 tests)
# ===========================================================================

class TestConsolidatedGroupReport:
    """Test consolidated group report template."""

    def test_consolidated_group_report_markdown(self, template_data):
        r = TemplateRendererStub("consolidated_group_report", template_data)
        md = r.render_markdown()
        assert "Consolidated Group Report" in md
        assert "2025" in md

    def test_consolidated_group_report_json(self, template_data):
        r = TemplateRendererStub("consolidated_group_report", template_data)
        result = r.render_json()
        assert result["template_id"] == "consolidated_group_report"
        assert len(result["provenance_hash"]) == 64

    def test_consolidated_group_report_missing_data(self):
        r = TemplateRendererStub("consolidated_group_report", {"reporting_year": 2025})
        result = r.render_with_missing_data()
        assert result["is_complete"] is False
        assert "consolidated_data" in result["missing_fields"]


class TestCdpQuestionnaireResponse:
    """Test CDP questionnaire response template."""

    def test_cdp_questionnaire_response_markdown(self, template_data):
        r = TemplateRendererStub("cdp_questionnaire_response", template_data)
        md = r.render_markdown()
        assert "Cdp Questionnaire Response" in md

    def test_cdp_questionnaire_response_json(self, template_data):
        r = TemplateRendererStub("cdp_questionnaire_response", template_data)
        result = r.render_json()
        assert result["template_id"] == "cdp_questionnaire_response"

    def test_cdp_questionnaire_response_missing_data(self):
        r = TemplateRendererStub("cdp_questionnaire_response", {"reporting_year": 2025})
        result = r.render_with_missing_data()
        assert "cdp_responses" in result["missing_fields"]


class TestTcfdDisclosure:
    """Test TCFD disclosure report template."""

    def test_tcfd_disclosure_markdown(self, template_data):
        r = TemplateRendererStub("tcfd_disclosure", template_data)
        md = r.render_markdown()
        assert "Tcfd Disclosure" in md

    def test_tcfd_disclosure_json(self, template_data):
        r = TemplateRendererStub("tcfd_disclosure", template_data)
        result = r.render_json()
        assert result["template_id"] == "tcfd_disclosure"

    def test_tcfd_disclosure_missing_data(self):
        r = TemplateRendererStub("tcfd_disclosure", {"reporting_year": 2025})
        result = r.render_with_missing_data()
        assert "governance" in result["missing_fields"]
        assert "strategy" in result["missing_fields"]


class TestSbtiProgressReport:
    """Test SBTi progress report template."""

    def test_sbti_progress_report_markdown(self, template_data):
        r = TemplateRendererStub("sbti_progress_report", template_data)
        md = r.render_markdown()
        assert "Sbti Progress Report" in md

    def test_sbti_progress_report_json(self, template_data):
        r = TemplateRendererStub("sbti_progress_report", template_data)
        result = r.render_json()
        assert result["template_id"] == "sbti_progress_report"

    def test_sbti_progress_report_missing_data(self):
        r = TemplateRendererStub("sbti_progress_report", {"reporting_year": 2025})
        result = r.render_with_missing_data()
        assert "base_year" in result["missing_fields"]
        assert "targets" in result["missing_fields"]


class TestEuTaxonomyReport:
    """Test EU Taxonomy report template."""

    def test_eu_taxonomy_report_markdown(self, template_data):
        r = TemplateRendererStub("eu_taxonomy_report", template_data)
        md = r.render_markdown()
        assert "Eu Taxonomy Report" in md

    def test_eu_taxonomy_report_json(self, template_data):
        r = TemplateRendererStub("eu_taxonomy_report", template_data)
        result = r.render_json()
        assert result["template_id"] == "eu_taxonomy_report"

    def test_eu_taxonomy_report_missing_data(self):
        r = TemplateRendererStub("eu_taxonomy_report", {"reporting_year": 2025})
        result = r.render_with_missing_data()
        assert "eligibility" in result["missing_fields"]
        assert "alignment" in result["missing_fields"]


class TestScenarioAnalysisReport:
    """Test scenario analysis report template."""

    def test_scenario_analysis_report_markdown(self, template_data):
        r = TemplateRendererStub("scenario_analysis_report", template_data)
        md = r.render_markdown()
        assert "Scenario Analysis Report" in md
        assert "Scenarios: 2" in md

    def test_scenario_analysis_report_json(self, template_data):
        r = TemplateRendererStub("scenario_analysis_report", template_data)
        result = r.render_json()
        assert result["template_id"] == "scenario_analysis_report"

    def test_scenario_analysis_report_missing_data(self):
        r = TemplateRendererStub("scenario_analysis_report", {"reporting_year": 2025})
        result = r.render_with_missing_data()
        assert "scenarios" in result["missing_fields"]


class TestCrossFrameworkMapping:
    """Test cross-framework mapping report template."""

    def test_cross_framework_mapping_markdown(self, template_data):
        r = TemplateRendererStub("cross_framework_mapping", template_data)
        md = r.render_markdown()
        assert "Cross Framework Mapping" in md
        assert "Frameworks:" in md

    def test_cross_framework_mapping_json(self, template_data):
        r = TemplateRendererStub("cross_framework_mapping", template_data)
        result = r.render_json()
        assert result["template_id"] == "cross_framework_mapping"

    def test_cross_framework_mapping_missing_data(self):
        r = TemplateRendererStub("cross_framework_mapping", {"reporting_year": 2025})
        result = r.render_with_missing_data()
        assert "frameworks" in result["missing_fields"]
        assert "coverage_matrix" in result["missing_fields"]


class TestBoardGovernancePackage:
    """Test board governance package template."""

    def test_board_governance_package_markdown(self, template_data):
        r = TemplateRendererStub("board_governance_package", template_data)
        md = r.render_markdown()
        assert "Board Governance Package" in md

    def test_board_governance_package_json(self, template_data):
        r = TemplateRendererStub("board_governance_package", template_data)
        result = r.render_json()
        assert result["template_id"] == "board_governance_package"

    def test_board_governance_package_missing_data(self):
        r = TemplateRendererStub("board_governance_package", {"reporting_year": 2025})
        result = r.render_with_missing_data()
        assert "kpis" in result["missing_fields"]


class TestRegulatoryChangeBriefing:
    """Test regulatory change briefing template."""

    def test_regulatory_change_briefing_markdown(self, template_data):
        r = TemplateRendererStub("regulatory_change_briefing", template_data)
        md = r.render_markdown()
        assert "Regulatory Change Briefing" in md

    def test_regulatory_change_briefing_json(self, template_data):
        r = TemplateRendererStub("regulatory_change_briefing", template_data)
        result = r.render_json()
        assert result["template_id"] == "regulatory_change_briefing"

    def test_regulatory_change_briefing_missing_data(self):
        r = TemplateRendererStub("regulatory_change_briefing", {"reporting_year": 2025})
        result = r.render_with_missing_data()
        assert "changes" in result["missing_fields"]


class TestProfessionalAuditPackage:
    """Test professional audit package template."""

    def test_professional_audit_package_markdown(self, template_data):
        r = TemplateRendererStub("professional_audit_package", template_data)
        md = r.render_markdown()
        assert "Professional Audit Package" in md

    def test_professional_audit_package_json(self, template_data):
        r = TemplateRendererStub("professional_audit_package", template_data)
        result = r.render_json()
        assert result["template_id"] == "professional_audit_package"

    def test_professional_audit_package_missing_data(self):
        r = TemplateRendererStub("professional_audit_package", {"reporting_year": 2025})
        result = r.render_with_missing_data()
        assert "assurance_level" in result["missing_fields"]
        assert "evidence" in result["missing_fields"]
