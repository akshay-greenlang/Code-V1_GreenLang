# -*- coding: utf-8 -*-
"""
PACK-019 CSDDD Readiness Pack - Test Configuration
====================================================

Shared test infrastructure for all PACK-019 test modules.
Provides dynamic module loading, path constants, and reusable
sample data fixtures for CSDDD due diligence testing.

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-019 CSDDD Readiness Pack
Date:    March 2026
"""

import importlib.util
import sys
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

# ---------------------------------------------------------------------------
# Path Constants
# ---------------------------------------------------------------------------

PACK_ROOT = Path(__file__).resolve().parent.parent
ENGINES_DIR = PACK_ROOT / "engines"
WORKFLOWS_DIR = PACK_ROOT / "workflows"
TEMPLATES_DIR = PACK_ROOT / "templates"
INTEGRATIONS_DIR = PACK_ROOT / "integrations"
CONFIG_DIR = PACK_ROOT / "config"
PRESETS_DIR = CONFIG_DIR / "presets"
DEMO_DIR = CONFIG_DIR / "demo"
MANIFEST_PATH = PACK_ROOT / "pack.yaml"

# ---------------------------------------------------------------------------
# File Mappings
# ---------------------------------------------------------------------------

ENGINE_FILES: Dict[str, str] = {
    "due_diligence_policy": "due_diligence_policy_engine.py",
    "adverse_impact": "adverse_impact_engine.py",
    "prevention_mitigation": "prevention_mitigation_engine.py",
    "remediation_tracking": "remediation_tracking_engine.py",
    "grievance_mechanism": "grievance_mechanism_engine.py",
    "stakeholder_engagement": "stakeholder_engagement_engine.py",
    "climate_transition": "climate_transition_engine.py",
    "civil_liability": "civil_liability_engine.py",
}

ENGINE_CLASSES: Dict[str, str] = {
    "due_diligence_policy": "DueDiligencePolicyEngine",
    "adverse_impact": "AdverseImpactEngine",
    "prevention_mitigation": "PreventionMitigationEngine",
    "remediation_tracking": "RemediationTrackingEngine",
    "grievance_mechanism": "GrievanceMechanismEngine",
    "stakeholder_engagement": "StakeholderEngagementEngine",
    "climate_transition": "ClimateTransitionEngine",
    "civil_liability": "CivilLiabilityEngine",
}

WORKFLOW_FILES: Dict[str, str] = {
    "due_diligence_assessment": "due_diligence_assessment_workflow.py",
    "value_chain_mapping": "value_chain_mapping_workflow.py",
    "impact_assessment": "impact_assessment_workflow.py",
    "prevention_planning": "prevention_planning_workflow.py",
    "grievance_management": "grievance_management_workflow.py",
    "monitoring_review": "monitoring_review_workflow.py",
    "climate_transition_planning": "climate_transition_planning_workflow.py",
    "regulatory_submission": "regulatory_submission_workflow.py",
}

WORKFLOW_CLASSES: Dict[str, str] = {
    "due_diligence_assessment": "DueDiligenceAssessmentWorkflow",
    "value_chain_mapping": "ValueChainMappingWorkflow",
    "impact_assessment": "ImpactAssessmentWorkflow",
    "prevention_planning": "PreventionPlanningWorkflow",
    "grievance_management": "GrievanceManagementWorkflow",
    "monitoring_review": "MonitoringReviewWorkflow",
    "climate_transition_planning": "ClimateTransitionPlanningWorkflow",
    "regulatory_submission": "RegulatorySubmissionWorkflow",
}

TEMPLATE_FILES: Dict[str, str] = {
    "dd_readiness_report": "dd_readiness_report.py",
    "value_chain_risk_map": "value_chain_risk_map.py",
    "impact_assessment_report": "impact_assessment_report.py",
    "prevention_mitigation_report": "prevention_mitigation_report.py",
    "grievance_mechanism_report": "grievance_mechanism_report.py",
    "stakeholder_engagement_report": "stakeholder_engagement_report.py",
    "climate_transition_report": "climate_transition_report.py",
    "csddd_scorecard": "csddd_scorecard.py",
}

TEMPLATE_CLASSES: Dict[str, str] = {
    "dd_readiness_report": "DDReadinessReportTemplate",
    "value_chain_risk_map": "ValueChainRiskMapTemplate",
    "impact_assessment_report": "ImpactAssessmentReportTemplate",
    "prevention_mitigation_report": "PreventionMitigationReportTemplate",
    "grievance_mechanism_report": "GrievanceMechanismReportTemplate",
    "stakeholder_engagement_report": "StakeholderEngagementReportTemplate",
    "climate_transition_report": "ClimateTransitionReportTemplate",
    "csddd_scorecard": "CSDDDScorecardTemplate",
}

INTEGRATION_FILES: Dict[str, str] = {
    "pack_orchestrator": "pack_orchestrator.py",
    "csrd_pack_bridge": "csrd_pack_bridge.py",
    "mrv_bridge": "mrv_bridge.py",
    "eudr_bridge": "eudr_bridge.py",
    "supply_chain_bridge": "supply_chain_bridge.py",
    "data_bridge": "data_bridge.py",
    "green_claims_bridge": "green_claims_bridge.py",
    "taxonomy_bridge": "taxonomy_bridge.py",
    "health_check": "health_check.py",
    "setup_wizard": "setup_wizard.py",
}

INTEGRATION_CLASSES: Dict[str, str] = {
    "pack_orchestrator": "CSDDDOrchestrator",
    "csrd_pack_bridge": "CSRDPackBridge",
    "mrv_bridge": "MRVBridge",
    "eudr_bridge": "EUDRBridge",
    "supply_chain_bridge": "SupplyChainBridge",
    "data_bridge": "DataBridge",
    "green_claims_bridge": "GreenClaimsBridge",
    "taxonomy_bridge": "TaxonomyBridge",
    "health_check": "CSDDDHealthCheck",
    "setup_wizard": "CSDDDSetupWizard",
}

PRESET_FILES: List[str] = [
    "manufacturing.yaml",
    "extractives.yaml",
    "financial_services.yaml",
    "retail.yaml",
    "technology.yaml",
    "agriculture.yaml",
]


# ---------------------------------------------------------------------------
# Dynamic Module Loader
# ---------------------------------------------------------------------------


def _load_module(file_path: Path, namespace: str) -> Any:
    """Load a Python module from file path using importlib.

    Uses ``pack019_test.*`` namespace to avoid collisions with other packs.
    Modules are cached in ``sys.modules`` for session reuse.
    """
    mod_name = f"pack019_test.{namespace}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]

    spec = importlib.util.spec_from_file_location(mod_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_engine(name: str) -> Any:
    """Load a PACK-019 engine module by short name."""
    filename = ENGINE_FILES.get(name)
    if filename is None:
        raise KeyError(f"Unknown engine: {name}")
    return _load_module(ENGINES_DIR / filename, f"engines.{name}")


def _load_workflow(name: str) -> Any:
    """Load a PACK-019 workflow module by short name."""
    filename = WORKFLOW_FILES.get(name)
    if filename is None:
        raise KeyError(f"Unknown workflow: {name}")
    return _load_module(WORKFLOWS_DIR / filename, f"workflows.{name}")


def _load_template(name: str) -> Any:
    """Load a PACK-019 template module by short name."""
    filename = TEMPLATE_FILES.get(name)
    if filename is None:
        raise KeyError(f"Unknown template: {name}")
    return _load_module(TEMPLATES_DIR / filename, f"templates.{name}")


def _load_integration(name: str) -> Any:
    """Load a PACK-019 integration module by short name."""
    filename = INTEGRATION_FILES.get(name)
    if filename is None:
        raise KeyError(f"Unknown integration: {name}")
    return _load_module(INTEGRATIONS_DIR / filename, f"integrations.{name}")


# ---------------------------------------------------------------------------
# Sample Data Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_company_profile() -> Dict[str, Any]:
    """Sample company profile for CSDDD scope determination."""
    return {
        "company_name": "EuroManufacturing AG",
        "country": "DE",
        "sector": "MANUFACTURING",
        "employee_count": 6000,
        "worldwide_turnover_eur": Decimal("2000000000"),
        "eu_turnover_eur": Decimal("1200000000"),
        "reporting_year": 2027,
        "value_chain_tiers": 4,
        "has_dd_policy": True,
        "has_code_of_conduct": True,
        "has_grievance_mechanism": False,
        "has_climate_transition_plan": False,
    }


@pytest.fixture
def sample_adverse_impacts() -> List[Dict[str, Any]]:
    """Sample adverse impacts for due diligence assessment."""
    return [
        {
            "impact_id": "AI-001",
            "type": "HUMAN_RIGHTS",
            "category": "forced_labour",
            "description": "Forced labour risk in tier-2 raw material supplier in Southeast Asia",
            "severity": "CRITICAL",
            "likelihood": "LIKELY",
            "status": "POTENTIAL",
            "value_chain_position": "UPSTREAM_INDIRECT",
            "affected_stakeholders": ["workers"],
            "linked_rights": ["ILO C029 Forced Labour"],
            "country": "MM",
        },
        {
            "impact_id": "AI-002",
            "type": "ENVIRONMENTAL",
            "category": "water_pollution",
            "description": "Industrial wastewater discharge exceeding local limits at tier-1 supplier",
            "severity": "HIGH",
            "likelihood": "VERY_LIKELY",
            "status": "ACTUAL",
            "value_chain_position": "UPSTREAM_DIRECT",
            "affected_stakeholders": ["communities"],
            "linked_rights": ["Stockholm Convention"],
            "country": "CN",
        },
        {
            "impact_id": "AI-003",
            "type": "HUMAN_RIGHTS",
            "category": "child_labour",
            "description": "Child labour in artisanal mining operations supplying cobalt",
            "severity": "CRITICAL",
            "likelihood": "POSSIBLE",
            "status": "POTENTIAL",
            "value_chain_position": "UPSTREAM_INDIRECT",
            "affected_stakeholders": ["workers", "children"],
            "linked_rights": ["ILO C182 Worst Forms of Child Labour", "CRC"],
            "country": "CD",
        },
        {
            "impact_id": "AI-004",
            "type": "ENVIRONMENTAL",
            "category": "deforestation",
            "description": "Deforestation linked to palm oil sourcing in upstream supply chain",
            "severity": "HIGH",
            "likelihood": "LIKELY",
            "status": "POTENTIAL",
            "value_chain_position": "UPSTREAM_INDIRECT",
            "affected_stakeholders": ["communities", "indigenous_peoples"],
            "linked_rights": ["CBD Kunming-Montreal GBF"],
            "country": "ID",
        },
        {
            "impact_id": "AI-005",
            "type": "HUMAN_RIGHTS",
            "category": "health_and_safety",
            "description": "Inadequate worker safety measures at own manufacturing facility",
            "severity": "MEDIUM",
            "likelihood": "LIKELY",
            "status": "ACTUAL",
            "value_chain_position": "OWN_OPERATIONS",
            "affected_stakeholders": ["workers"],
            "linked_rights": ["ILO C155 Occupational Safety"],
            "country": "DE",
        },
    ]


@pytest.fixture
def sample_prevention_measures() -> List[Dict[str, Any]]:
    """Sample prevention and mitigation measures."""
    return [
        {
            "measure_id": "PM-001",
            "type": "PREVENTION",
            "description": "Supplier code of conduct with forced labour prohibition",
            "target_impact_ids": ["AI-001"],
            "responsible_person": "Head of Procurement",
            "deadline": "2027-06-30",
            "budget_eur": Decimal("50000"),
            "effectiveness_score": Decimal("0.7"),
        },
        {
            "measure_id": "PM-002",
            "type": "MITIGATION",
            "description": "Wastewater treatment upgrade at tier-1 supplier facility",
            "target_impact_ids": ["AI-002"],
            "responsible_person": "Environmental Manager",
            "deadline": "2027-12-31",
            "budget_eur": Decimal("200000"),
            "effectiveness_score": Decimal("0.85"),
        },
        {
            "measure_id": "PM-003",
            "type": "PREVENTION",
            "description": "Third-party audit programme for cobalt supply chain",
            "target_impact_ids": ["AI-003"],
            "responsible_person": "Supply Chain Director",
            "deadline": "2027-09-30",
            "budget_eur": Decimal("150000"),
            "effectiveness_score": Decimal("0.6"),
        },
    ]


@pytest.fixture
def sample_grievance_cases() -> List[Dict[str, Any]]:
    """Sample grievance cases for complaints mechanism testing."""
    return [
        {
            "case_id": "GC-001",
            "status": "RESOLVED",
            "submitted_by": "anonymous_worker",
            "stakeholder_group": "WORKERS",
            "description": "Excessive overtime without consent at supplier facility",
            "adverse_impact_ref": "AI-001",
            "resolution": "Supplier corrective action plan implemented",
            "days_to_resolve": 45,
            "submitted_date": "2027-01-15",
            "resolved_date": "2027-03-01",
        },
        {
            "case_id": "GC-002",
            "status": "INVESTIGATING",
            "submitted_by": "community_leader",
            "stakeholder_group": "COMMUNITIES",
            "description": "Water contamination affecting downstream community",
            "adverse_impact_ref": "AI-002",
            "resolution": None,
            "days_to_resolve": None,
            "submitted_date": "2027-02-20",
            "resolved_date": None,
        },
        {
            "case_id": "GC-003",
            "status": "RECEIVED",
            "submitted_by": "ngo_partner",
            "stakeholder_group": "NGOS",
            "description": "Deforestation allegations in palm oil supply chain",
            "adverse_impact_ref": "AI-004",
            "resolution": None,
            "days_to_resolve": None,
            "submitted_date": "2027-03-10",
            "resolved_date": None,
        },
    ]


@pytest.fixture
def sample_climate_targets() -> List[Dict[str, Any]]:
    """Sample climate transition targets for Article 22 assessment."""
    return [
        {
            "target_id": "CT-001",
            "scope": "SCOPE_1",
            "base_year": 2023,
            "target_year": 2030,
            "reduction_pct": Decimal("42"),
            "aligned_with_15c": True,
            "interim_milestones": [
                {"year": 2025, "reduction_pct": Decimal("15")},
                {"year": 2027, "reduction_pct": Decimal("28")},
            ],
        },
        {
            "target_id": "CT-002",
            "scope": "SCOPE_2",
            "base_year": 2023,
            "target_year": 2030,
            "reduction_pct": Decimal("50"),
            "aligned_with_15c": True,
            "interim_milestones": [
                {"year": 2025, "reduction_pct": Decimal("20")},
                {"year": 2027, "reduction_pct": Decimal("35")},
            ],
        },
        {
            "target_id": "CT-003",
            "scope": "SCOPE_3",
            "base_year": 2023,
            "target_year": 2035,
            "reduction_pct": Decimal("30"),
            "aligned_with_15c": False,
            "interim_milestones": [
                {"year": 2027, "reduction_pct": Decimal("10")},
                {"year": 2030, "reduction_pct": Decimal("20")},
            ],
        },
    ]


@pytest.fixture
def sample_stakeholder_engagements() -> List[Dict[str, Any]]:
    """Sample stakeholder engagement activities."""
    return [
        {
            "engagement_id": "SE-001",
            "stakeholder_group": "TRADE_UNIONS",
            "method": "formal_consultation",
            "topic": "Due diligence policy development",
            "date": "2027-01-20",
            "participants": 12,
            "outcomes": ["Input on risk prioritization", "Agreement on monitoring KPIs"],
            "meaningful": True,
        },
        {
            "engagement_id": "SE-002",
            "stakeholder_group": "COMMUNITIES",
            "method": "community_meeting",
            "topic": "Environmental impact of supplier operations",
            "date": "2027-02-15",
            "participants": 45,
            "outcomes": ["Identified water contamination concerns", "Agreed remediation timeline"],
            "meaningful": True,
        },
        {
            "engagement_id": "SE-003",
            "stakeholder_group": "NGOS",
            "method": "written_consultation",
            "topic": "Child labour risk assessment methodology",
            "date": "2027-03-01",
            "participants": 3,
            "outcomes": ["Methodology review feedback"],
            "meaningful": True,
        },
    ]


@pytest.fixture
def sample_remediation_actions() -> List[Dict[str, Any]]:
    """Sample remediation actions for actual adverse impacts."""
    return [
        {
            "action_id": "RA-001",
            "adverse_impact_id": "AI-002",
            "description": "Fund wastewater treatment facility upgrade at supplier site",
            "financial_provision_eur": Decimal("200000"),
            "victim_engagement": True,
            "completion_status": "IN_PROGRESS",
            "start_date": "2027-01-01",
            "target_completion": "2027-12-31",
        },
        {
            "action_id": "RA-002",
            "adverse_impact_id": "AI-005",
            "description": "Install safety equipment and conduct worker training programme",
            "financial_provision_eur": Decimal("75000"),
            "victim_engagement": True,
            "completion_status": "COMPLETED",
            "start_date": "2026-10-01",
            "target_completion": "2027-03-31",
        },
    ]


@pytest.fixture
def sample_value_chain() -> List[Dict[str, Any]]:
    """Sample value chain structure for mapping."""
    return [
        {
            "supplier_id": "SUP-001",
            "name": "MetalWorks Co.",
            "tier": 1,
            "country": "CN",
            "sector": "metals_manufacturing",
            "employees": 500,
            "products": ["steel components", "aluminium parts"],
            "risk_level": "HIGH",
        },
        {
            "supplier_id": "SUP-002",
            "name": "RawMat Mining Ltd.",
            "tier": 2,
            "country": "CD",
            "sector": "mining",
            "employees": 200,
            "products": ["cobalt ore", "copper ore"],
            "risk_level": "CRITICAL",
        },
        {
            "supplier_id": "SUP-003",
            "name": "PalmOil Plantation Sdn Bhd",
            "tier": 2,
            "country": "ID",
            "sector": "agriculture",
            "employees": 1500,
            "products": ["crude palm oil"],
            "risk_level": "HIGH",
        },
        {
            "supplier_id": "SUP-004",
            "name": "ChemProcess AG",
            "tier": 1,
            "country": "DE",
            "sector": "chemicals",
            "employees": 300,
            "products": ["industrial chemicals", "solvents"],
            "risk_level": "MEDIUM",
        },
    ]


@pytest.fixture
def sample_civil_liability_scenario() -> Dict[str, Any]:
    """Sample civil liability assessment scenario."""
    return {
        "company_name": "EuroManufacturing AG",
        "adverse_impact": {
            "type": "ENVIRONMENTAL",
            "category": "water_pollution",
            "severity": "HIGH",
            "actual": True,
            "remediation_provided": False,
        },
        "due_diligence_performed": True,
        "prevention_measures_taken": True,
        "contractual_assurances_obtained": True,
        "verification_measures_applied": False,
        "stakeholders_consulted": True,
        "jurisdiction": "DE",
        "limitation_period_years": 5,
        "damage_estimate_eur": Decimal("5000000"),
    }
