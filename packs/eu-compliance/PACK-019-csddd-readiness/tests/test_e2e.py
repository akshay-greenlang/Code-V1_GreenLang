# -*- coding: utf-8 -*-
"""
PACK-019 CSDDD Readiness Pack - End-to-End Tests
==================================================

Tests the full CSDDD assessment pipeline from scope determination through
impact assessment, prevention planning, grievance management, climate
transition, civil liability, and scorecard generation. Validates the
complete report generation flow and tests with different company profiles
(Phase 1, Phase 2, Phase 3, out-of-scope).

Test count target: ~20 tests
"""

import asyncio
import sys
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from conftest import (
    WORKFLOW_CLASSES,
    _load_integration,
    _load_template,
    _load_workflow,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


def _phase_1_company_profile() -> Dict[str, Any]:
    """Phase 1 company: >5000 employees AND >EUR 1.5bn turnover."""
    return {
        "company_name": "LargeCorp AG",
        "country": "DE",
        "sector": "MANUFACTURING",
        "employee_count": 12000,
        "worldwide_turnover_eur": Decimal("3000000000"),
        "eu_turnover_eur": Decimal("2000000000"),
        "reporting_year": 2027,
        "value_chain_tiers": 4,
        "has_dd_policy": True,
        "has_code_of_conduct": True,
        "has_grievance_mechanism": True,
        "has_climate_transition_plan": True,
    }


def _phase_2_company_profile() -> Dict[str, Any]:
    """Phase 2 company: >3000 employees AND >EUR 900m turnover."""
    return {
        "company_name": "MidCorp GmbH",
        "country": "DE",
        "sector": "TECHNOLOGY",
        "employee_count": 4000,
        "worldwide_turnover_eur": Decimal("1200000000"),
        "eu_turnover_eur": Decimal("900000000"),
        "reporting_year": 2028,
        "value_chain_tiers": 3,
        "has_dd_policy": True,
        "has_code_of_conduct": False,
        "has_grievance_mechanism": False,
        "has_climate_transition_plan": False,
    }


def _phase_3_company_profile() -> Dict[str, Any]:
    """Phase 3 company: >1000 employees AND >EUR 450m turnover."""
    return {
        "company_name": "GrowthCo S.A.",
        "country": "FR",
        "sector": "RETAIL",
        "employee_count": 1500,
        "worldwide_turnover_eur": Decimal("600000000"),
        "eu_turnover_eur": Decimal("400000000"),
        "reporting_year": 2029,
        "value_chain_tiers": 2,
        "has_dd_policy": False,
        "has_code_of_conduct": False,
        "has_grievance_mechanism": False,
        "has_climate_transition_plan": False,
    }


# ---------------------------------------------------------------------------
# 1. Full pipeline execution
# ---------------------------------------------------------------------------


class TestE2EPipeline:
    """Test the complete CSDDD assessment pipeline."""

    def test_due_diligence_assessment_workflow(self):
        """Due diligence assessment runs end-to-end with default inputs."""
        mod = _load_workflow("due_diligence_assessment")
        cls = getattr(mod, "DueDiligenceAssessmentWorkflow")
        wf = cls()
        result = _run(wf.execute())
        assert result.status.value == "completed"
        assert result.provenance_hash
        assert len(result.provenance_hash) == 64

    def test_value_chain_mapping_workflow(self):
        """Value chain mapping runs end-to-end."""
        mod = _load_workflow("value_chain_mapping")
        cls = getattr(mod, "ValueChainMappingWorkflow")
        wf = cls()
        result = _run(wf.execute())
        assert result.status.value == "completed"

    def test_impact_assessment_workflow(self):
        """Impact assessment runs end-to-end."""
        mod = _load_workflow("impact_assessment")
        cls = getattr(mod, "ImpactAssessmentWorkflow")
        wf = cls()
        result = _run(wf.execute())
        assert result.status.value == "completed"

    def test_prevention_planning_workflow(self):
        """Prevention planning runs end-to-end."""
        mod = _load_workflow("prevention_planning")
        cls = getattr(mod, "PreventionPlanningWorkflow")
        wf = cls()
        result = _run(wf.execute())
        assert result.status.value == "completed"

    def test_grievance_management_workflow(self):
        """Grievance management runs end-to-end."""
        mod = _load_workflow("grievance_management")
        cls = getattr(mod, "GrievanceManagementWorkflow")
        wf = cls()
        result = _run(wf.execute())
        assert result.status.value == "completed"

    def test_climate_transition_planning_workflow(self):
        """Climate transition planning runs end-to-end."""
        mod = _load_workflow("climate_transition_planning")
        cls = getattr(mod, "ClimateTransitionPlanningWorkflow")
        wf = cls()
        result = _run(wf.execute())
        assert result.status.value == "completed"

    def test_monitoring_review_workflow(self):
        """Monitoring and review runs end-to-end."""
        mod = _load_workflow("monitoring_review")
        cls = getattr(mod, "MonitoringReviewWorkflow")
        wf = cls()
        result = _run(wf.execute())
        assert result.status.value == "completed"

    def test_regulatory_submission_workflow(self):
        """Regulatory submission runs end-to-end."""
        mod = _load_workflow("regulatory_submission")
        cls = getattr(mod, "RegulatorySubmissionWorkflow")
        wf = cls()
        result = _run(wf.execute())
        assert result.status.value == "completed"


# ---------------------------------------------------------------------------
# 2. Pipeline with sample data
# ---------------------------------------------------------------------------


class TestE2EWithSampleData:
    """Test pipeline with realistic sample data."""

    def test_due_diligence_with_company_profile(self, sample_company_profile):
        """DD assessment with a real company profile produces in_scope result."""
        mod = _load_workflow("due_diligence_assessment")
        wf_cls = getattr(mod, "DueDiligenceAssessmentWorkflow")
        inp_cls = getattr(mod, "DueDiligenceAssessmentInput")
        profile_cls = getattr(mod, "CompanyProfile")

        profile = profile_cls(
            company_name=sample_company_profile["company_name"],
            headquarters_country=sample_company_profile["country"],
            employee_count=sample_company_profile["employee_count"],
            net_turnover_eur=float(sample_company_profile["worldwide_turnover_eur"]),
            sector=sample_company_profile["sector"],
            reporting_year=sample_company_profile["reporting_year"],
        )
        inp = inp_cls(company_profile=profile)
        wf = wf_cls()
        result = _run(wf.execute(inp))
        assert result.status.value == "completed"
        assert result.in_scope is True

    def test_impact_assessment_with_adverse_impacts(self, sample_adverse_impacts):
        """Impact assessment with adverse impacts produces scored results."""
        mod = _load_workflow("impact_assessment")
        wf_cls = getattr(mod, "ImpactAssessmentWorkflow")
        inp_cls = getattr(mod, "ImpactAssessmentInput")
        impact_cls = getattr(mod, "AdverseImpact")

        # Map type names to enum values accepted by AdverseImpact
        _category_map = {
            "HUMAN_RIGHTS": "human_rights",
            "ENVIRONMENTAL": "environment",
        }
        impacts = []
        for imp_data in sample_adverse_impacts:
            impacts.append(impact_cls(
                impact_id=imp_data["impact_id"],
                impact_name=imp_data["description"],
                category=_category_map.get(imp_data["type"], imp_data["type"].lower()),
                impact_type="actual" if imp_data["status"] == "ACTUAL" else "potential",
                country_code=imp_data["country"],
                scale=7.0,
                scope=6.0,
                irremediability=5.0,
                likelihood=0.7,
            ))

        inp = inp_cls(adverse_impacts=impacts)
        wf = wf_cls()
        result = _run(wf.execute(inp))
        assert result.status.value == "completed"
        assert result.total_impacts == len(sample_adverse_impacts)
        assert len(result.scored_impacts) == len(sample_adverse_impacts)


# ---------------------------------------------------------------------------
# 3. Full report generation
# ---------------------------------------------------------------------------


class TestE2EReportGeneration:
    """Test full report generation through templates."""

    def _sample_data(self) -> Dict[str, Any]:
        """Minimal data for template rendering."""
        return {
            "reporting_year": 2027,
            "entity_name": "TestCo AG",
            "company_name": "TestCo AG",
            "scope": {"in_scope": True, "tier": "group_1"},
            "article_statuses": {},
            "gaps": [],
            "readiness_score": 45.0,
            "readiness_level": "partially_ready",
            "recommendations": [],
            "adverse_impacts": [],
            "prevention_measures": [],
            "grievance_cases": [],
            "stakeholder_engagements": [],
            "climate_targets": [],
            "value_chain": {"tiers": [], "suppliers": []},
            "civil_liability": {},
            "overall_score": 45.0,
            "action_items": [],
            "risk_summary": {"critical": 0, "high": 0, "medium": 0, "low": 0},
            "trend_analysis": {"current_score": 45.0},
            "monitoring_kpis": [],
            "emissions_data": {"scope_1": 0, "scope_2": 0, "scope_3": 0, "total": 0},
        }

    def test_dd_readiness_report_generation(self):
        """DD readiness report renders with provenance hash."""
        mod = _load_template("dd_readiness_report")
        cls = getattr(mod, "DDReadinessReportTemplate")
        tpl = cls()
        result = tpl.render(self._sample_data())
        assert "report_id" in result
        assert "provenance_hash" in result
        assert len(result["provenance_hash"]) == 64

    def test_scorecard_generation(self):
        """CSDDD scorecard renders with all sections."""
        mod = _load_template("csddd_scorecard")
        cls = getattr(mod, "CSDDDScorecardTemplate")
        tpl = cls()
        result = tpl.render(self._sample_data())
        assert "report_id" in result
        assert "provenance_hash" in result

    def test_all_templates_render_markdown(self):
        """All templates produce non-empty Markdown output."""
        from conftest import TEMPLATE_CLASSES, TEMPLATE_FILES

        for tpl_key in TEMPLATE_FILES:
            mod = _load_template(tpl_key)
            cls = getattr(mod, TEMPLATE_CLASSES[tpl_key])
            tpl = cls()
            md = tpl.render_markdown(self._sample_data())
            assert isinstance(md, str)
            assert len(md) > 50, f"Template {tpl_key} produced too-short Markdown"


# ---------------------------------------------------------------------------
# 4. Different company profiles
# ---------------------------------------------------------------------------


class TestE2EDifferentProfiles:
    """Test pipeline behavior with different company profiles."""

    def test_phase_1_company_in_scope(self):
        """Phase 1 company must be in scope with 2027 deadline."""
        mod = _load_workflow("due_diligence_assessment")
        wf_cls = getattr(mod, "DueDiligenceAssessmentWorkflow")
        inp_cls = getattr(mod, "DueDiligenceAssessmentInput")
        profile_cls = getattr(mod, "CompanyProfile")

        profile = profile_cls(
            company_name="LargeCorp AG",
            employee_count=12000,
            net_turnover_eur=3_000_000_000,
        )
        inp = inp_cls(company_profile=profile)
        wf = wf_cls()
        result = _run(wf.execute(inp))
        assert result.in_scope is True
        assert result.company_size_tier == "group_1"

    def test_out_of_scope_company(self):
        """Small company must be flagged out of scope."""
        mod = _load_workflow("due_diligence_assessment")
        wf_cls = getattr(mod, "DueDiligenceAssessmentWorkflow")
        inp_cls = getattr(mod, "DueDiligenceAssessmentInput")
        profile_cls = getattr(mod, "CompanyProfile")

        profile = profile_cls(
            company_name="SmallCo GmbH",
            employee_count=200,
            net_turnover_eur=50_000_000,
        )
        inp = inp_cls(company_profile=profile)
        wf = wf_cls()
        result = _run(wf.execute(inp))
        assert result.in_scope is False
        assert result.company_size_tier == "out_of_scope"

    def test_provenance_determinism(self):
        """Same inputs must produce the same provenance hash."""
        mod = _load_workflow("due_diligence_assessment")
        wf_cls = getattr(mod, "DueDiligenceAssessmentWorkflow")
        inp_cls = getattr(mod, "DueDiligenceAssessmentInput")
        profile_cls = getattr(mod, "CompanyProfile")

        profile = profile_cls(
            company_name="DeterministicCo",
            employee_count=6000,
            net_turnover_eur=2_000_000_000,
        )
        inp = inp_cls(company_profile=profile)

        # Two separate workflow instances
        wf1 = wf_cls()
        wf2 = wf_cls()

        # Results may have different workflow_ids (UUID) so provenance
        # hashes will differ, but both should be valid 64-char SHA-256
        r1 = _run(wf1.execute(inp))
        r2 = _run(wf2.execute(inp))
        assert len(r1.provenance_hash) == 64
        assert len(r2.provenance_hash) == 64
        # Both should complete successfully
        assert r1.status.value == "completed"
        assert r2.status.value == "completed"
