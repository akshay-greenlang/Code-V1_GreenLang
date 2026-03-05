# -*- coding: utf-8 -*-
"""
Shared pytest fixtures for GL-TCFD-APP v1.0 test suite.

Provides reusable fixtures for configuration, organizations, governance
assessments, climate risks/opportunities, scenario definitions/results,
physical/transition risk assessments, financial impacts, risk management,
metrics/targets, disclosures, gap assessments, ISSB mappings, and mock
database sessions used across all 16 test modules.

Author: GL-TestEngineer
Date: March 2026
"""

import sys
import os
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

# ---------------------------------------------------------------------------
# Path setup -- ensure the TCFD services package is importable
# ---------------------------------------------------------------------------
_SERVICES_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "..",
    "applications", "GL-TCFD-APP", "TCFD-Disclosure-Platform",
)
_SERVICES_DIR = os.path.normpath(_SERVICES_DIR)
if _SERVICES_DIR not in sys.path:
    sys.path.insert(0, _SERVICES_DIR)

from services.config import (
    TCFDAppConfig,
    TCFDPillar,
    RiskType,
    OpportunityCategory,
    ScenarioType,
    TimeHorizon,
    TemperatureOutcome,
    AssetType,
    DisclosureStatus,
    RiskLikelihood,
    RiskImpact,
    MaturityLevel,
    TargetType,
    FinancialStatementArea,
    PhysicalHazard,
    TransitionDriver,
    ISSBMetricType,
    SectorType,
    SCENARIO_LIBRARY,
    TCFD_DISCLOSURES,
    TCFD_TO_IFRS_S2_MAPPING,
    LIKELIHOOD_SCORES,
    IMPACT_SCORES,
    MATURITY_SCORES,
    HAZARD_EXPOSURE_MATRICES,
    RISK_MATRIX_THRESHOLDS,
    SECTOR_TRANSITION_PROFILES,
    ISSB_CROSS_INDUSTRY_METRICS,
    PILLAR_NAMES,
    TIME_HORIZON_YEARS,
    REGULATORY_JURISDICTIONS,
)
from services.models import (
    GovernanceRole,
    GovernanceAssessment,
    ClimateRisk,
    ClimateOpportunity,
    ScenarioParameter,
    ScenarioDefinition,
    ScenarioResult,
    AssetLocation,
    PhysicalRiskAssessment,
    TransitionRiskAssessment,
    FinancialImpact,
    RiskManagementRecord,
    ClimateMetric,
    ClimateTarget,
    TargetProgress,
    DisclosureSection,
    TCFDDisclosure,
    GapAssessment,
    ISSBMapping,
    Recommendation,
    ApiError,
    ApiResponse,
    PaginatedResponse,
    CreateGovernanceAssessmentRequest,
    CreateClimateRiskRequest,
    CreateClimateOpportunityRequest,
    CreateScenarioRequest,
    RunScenarioAnalysisRequest,
    RegisterAssetRequest,
    CreateTargetRequest,
    RecordMetricRequest,
    CreateDisclosureRequest,
    UpdateDisclosureStatusRequest,
    CreateRiskManagementRecordRequest,
    _new_id,
    _now,
    _sha256,
)


# ============================================================================
# CONFIGURATION FIXTURES
# ============================================================================

@pytest.fixture
def default_config() -> TCFDAppConfig:
    """Default TCFD application configuration."""
    return TCFDAppConfig()


@pytest.fixture
def custom_config() -> TCFDAppConfig:
    """Custom configuration with adjusted parameters for testing."""
    return TCFDAppConfig(
        reporting_year=2025,
        default_scenario_type=ScenarioType.NGFS_BELOW_2C,
        scenario_monte_carlo_iterations=5000,
        financial_discount_rate=Decimal("0.10"),
    )


# ============================================================================
# ORGANIZATION FIXTURES
# ============================================================================

@pytest.fixture
def sample_org_id() -> str:
    """Stable organization ID for cross-fixture referencing."""
    return str(uuid4())


@pytest.fixture
def sample_organization(sample_org_id) -> Dict[str, Any]:
    """Sample TCFD organization data."""
    return {
        "id": sample_org_id,
        "tenant_id": str(uuid4()),
        "name": "Acme Energy Corp",
        "sector": "energy",
        "industry": "Oil & Gas",
        "jurisdiction": "US",
        "reporting_currency": "USD",
        "fiscal_year_end": "12-31",
    }


@pytest.fixture
def financial_org() -> Dict[str, Any]:
    """Financial services organization for banking-specific tests."""
    return {
        "id": str(uuid4()),
        "tenant_id": str(uuid4()),
        "name": "Global Finance Holdings",
        "sector": "banking",
        "industry": "Commercial Banking",
        "jurisdiction": "UK",
        "reporting_currency": "GBP",
        "fiscal_year_end": "03-31",
    }


# ============================================================================
# GOVERNANCE FIXTURES
# ============================================================================

@pytest.fixture
def sample_governance_assessment(sample_org_id) -> GovernanceAssessment:
    """Sample governance assessment for testing."""
    return GovernanceAssessment(
        org_id=sample_org_id,
        board_oversight_score=4,
        board_committees=["Sustainability Committee", "Risk Committee"],
        meeting_frequency=4,
        climate_competency_score=3,
        incentive_linkage=True,
        incentive_pct=Decimal("15.0"),
        maturity_scores={
            "board_oversight": 4,
            "management_roles": 3,
            "climate_competency": 3,
            "meeting_frequency": 4,
            "reporting_structure": 3,
            "incentive_alignment": 3,
            "risk_integration": 2,
            "strategy_integration": 3,
        },
        overall_maturity=MaturityLevel.DEFINED,
        notes="Annual governance assessment for TCFD disclosure.",
    )


@pytest.fixture
def sample_governance_role(sample_org_id) -> GovernanceRole:
    """Sample governance role."""
    return GovernanceRole(
        org_id=sample_org_id,
        role_title="Chief Sustainability Officer",
        person_name="Jane Smith",
        responsibility_description="Overall climate strategy and TCFD compliance",
        climate_accountability=True,
        reporting_line="CEO",
        competency_areas=["climate science", "ESG reporting", "risk management"],
    )


# ============================================================================
# CLIMATE RISK FIXTURES
# ============================================================================

@pytest.fixture
def sample_climate_risk(sample_org_id) -> ClimateRisk:
    """Sample physical acute climate risk."""
    return ClimateRisk(
        org_id=sample_org_id,
        risk_type=RiskType.PHYSICAL_ACUTE,
        name="Coastal Flooding",
        description="Increased frequency of coastal flooding at Texas facilities",
        category="physical",
        time_horizon=TimeHorizon.MEDIUM_TERM,
        likelihood=RiskLikelihood.LIKELY,
        impact=RiskImpact.MAJOR,
        financial_impact_estimate=Decimal("25000000"),
        affected_assets=["houston-refinery", "galveston-terminal"],
        response_strategy="Elevate critical equipment, install flood barriers",
        owner="VP Operations",
        status="active",
    )


@pytest.fixture
def sample_transition_risk(sample_org_id) -> ClimateRisk:
    """Sample transition policy risk."""
    return ClimateRisk(
        org_id=sample_org_id,
        risk_type=RiskType.TRANSITION_POLICY,
        name="Carbon Tax",
        description="Introduction of carbon pricing mechanism in operating jurisdictions",
        category="transition",
        time_horizon=TimeHorizon.SHORT_TERM,
        likelihood=RiskLikelihood.ALMOST_CERTAIN,
        impact=RiskImpact.MODERATE,
        financial_impact_estimate=Decimal("50000000"),
        response_strategy="Accelerate emissions reduction, internal carbon pricing",
    )


# ============================================================================
# CLIMATE OPPORTUNITY FIXTURES
# ============================================================================

@pytest.fixture
def sample_climate_opportunity(sample_org_id) -> ClimateOpportunity:
    """Sample climate opportunity."""
    return ClimateOpportunity(
        org_id=sample_org_id,
        category=OpportunityCategory.ENERGY_SOURCE,
        name="Renewable Energy Transition",
        description="Revenue from renewable energy products and services",
        revenue_potential=Decimal("100000000"),
        cost_savings=Decimal("15000000"),
        investment_required=Decimal("50000000"),
        roi_estimate=Decimal("0.25"),
        timeline=TimeHorizon.MEDIUM_TERM,
        feasibility_score=4,
        priority_score=5,
        status="planned",
    )


# ============================================================================
# SCENARIO FIXTURES
# ============================================================================

@pytest.fixture
def sample_scenario_definition() -> ScenarioDefinition:
    """Sample IEA NZE scenario definition."""
    lib = SCENARIO_LIBRARY[ScenarioType.IEA_NZE]
    return ScenarioDefinition(
        name=lib["name"],
        scenario_type=ScenarioType.IEA_NZE,
        temperature_outcome=TemperatureOutcome.BELOW_1_5C,
        description=lib["description"],
        carbon_price_trajectory=lib["carbon_price_trajectory"],
        energy_mix_trajectory=lib["energy_mix_trajectory"],
    )


@pytest.fixture
def custom_scenario_definition(sample_org_id) -> ScenarioDefinition:
    """Custom scenario definition for testing."""
    return ScenarioDefinition(
        name="Custom High Ambition",
        scenario_type=ScenarioType.CUSTOM,
        temperature_outcome=TemperatureOutcome.BELOW_1_5C,
        description="Custom scenario with accelerated transition",
        carbon_price_trajectory={
            2025: Decimal("100"), 2030: Decimal("200"),
            2040: Decimal("350"), 2050: Decimal("500"),
        },
        energy_mix_trajectory={
            2030: {"renewable_pct": 60, "fossil_pct": 30, "nuclear_pct": 10},
            2050: {"renewable_pct": 95, "fossil_pct": 0, "nuclear_pct": 5},
        },
    )


@pytest.fixture
def sample_scenario_result(sample_org_id) -> ScenarioResult:
    """Sample scenario result."""
    return ScenarioResult(
        scenario_id=_new_id(),
        org_id=sample_org_id,
        revenue_impact_pct=Decimal("-8.5"),
        cost_impact_pct=Decimal("12.3"),
        asset_impairment_pct=Decimal("15.0"),
        capex_required=Decimal("250000000"),
        npv=Decimal("-120000000"),
        confidence_interval_lower=Decimal("-180000000"),
        confidence_interval_upper=Decimal("-60000000"),
        key_assumptions=["Carbon price $130/tCO2e by 2030", "50% renewable by 2030"],
        narrative="Under IEA NZE, the organization faces moderate transition risk.",
    )


@pytest.fixture
def sample_scenario_parameter() -> ScenarioParameter:
    """Sample scenario parameter."""
    return ScenarioParameter(
        parameter_name="carbon_price",
        parameter_category="carbon",
        year=2030,
        value=Decimal("130.00"),
        unit="USD/tCO2e",
        source="IEA World Energy Outlook 2023",
    )


# ============================================================================
# PHYSICAL RISK FIXTURES
# ============================================================================

@pytest.fixture
def sample_asset_location(sample_org_id) -> AssetLocation:
    """Sample asset location for physical risk assessment."""
    return AssetLocation(
        org_id=sample_org_id,
        asset_name="Houston Refinery",
        asset_type=AssetType.BUILDING,
        latitude=Decimal("29.7604"),
        longitude=Decimal("-95.3698"),
        country="US",
        region="Texas",
        elevation=Decimal("15"),
        building_type="Industrial",
        replacement_value=Decimal("500000000"),
        year_built=1995,
        insurance_coverage=Decimal("400000000"),
    )


@pytest.fixture
def sample_physical_risk_assessment(sample_org_id) -> PhysicalRiskAssessment:
    """Sample physical risk assessment."""
    return PhysicalRiskAssessment(
        org_id=sample_org_id,
        asset_id=_new_id(),
        hazard_type=PhysicalHazard.FLOOD,
        rcp_scenario="ssp2_45",
        exposure_score=4,
        vulnerability_score=3,
        adaptive_capacity_score=2,
        composite_risk_score=Decimal("68.0"),
        financial_damage_estimate=Decimal("25000000"),
        insurance_cost_impact=Decimal("2000000"),
    )


# ============================================================================
# TRANSITION RISK FIXTURES
# ============================================================================

@pytest.fixture
def sample_transition_risk_assessment(sample_org_id) -> TransitionRiskAssessment:
    """Sample transition risk assessment."""
    return TransitionRiskAssessment(
        org_id=sample_org_id,
        risk_type=RiskType.TRANSITION_POLICY,
        driver=TransitionDriver.CARBON_PRICING,
        sector=SectorType.ENERGY,
        current_exposure=Decimal("100000000"),
        projected_exposure_2030=Decimal("250000000"),
        projected_exposure_2050=Decimal("500000000"),
        financial_impact=Decimal("150000000"),
        stranding_probability=Decimal("25.0"),
        mitigation_actions=["Diversify into renewables", "Internal carbon price"],
    )


# ============================================================================
# FINANCIAL IMPACT FIXTURES
# ============================================================================

@pytest.fixture
def sample_financial_impact(sample_org_id) -> FinancialImpact:
    """Sample financial impact on income statement."""
    return FinancialImpact(
        org_id=sample_org_id,
        scenario_id=_new_id(),
        statement_area=FinancialStatementArea.INCOME_STATEMENT,
        line_item="Revenue",
        current_value=Decimal("2500000000"),
        projected_value=Decimal("2287500000"),
        time_horizon=TimeHorizon.MEDIUM_TERM,
        confidence_level=Decimal("0.7"),
        assumptions=["Carbon price $130/tCO2e", "8.5% revenue decline"],
    )


# ============================================================================
# RISK MANAGEMENT FIXTURES
# ============================================================================

@pytest.fixture
def sample_risk_management_record(sample_org_id) -> RiskManagementRecord:
    """Sample risk management record."""
    return RiskManagementRecord(
        org_id=sample_org_id,
        risk_id=_new_id(),
        identification_process="Annual climate risk assessment workshop",
        assessment_methodology="5x5 likelihood-impact matrix",
        likelihood_score=4,
        impact_score=4,
        response_type="mitigate",
        response_actions=["Install flood barriers", "Relocate critical equipment"],
        owner="VP Risk",
        review_date=date(2026, 6, 30),
        erm_integrated=True,
    )


# ============================================================================
# METRICS & TARGETS FIXTURES
# ============================================================================

@pytest.fixture
def sample_climate_metric(sample_org_id) -> ClimateMetric:
    """Sample GHG emissions metric."""
    return ClimateMetric(
        org_id=sample_org_id,
        metric_type=ISSBMetricType.GHG_EMISSIONS,
        metric_name="Total Scope 1 GHG Emissions",
        value=Decimal("125000"),
        unit="tCO2e",
        reporting_year=2025,
        scope="scope_1",
        data_quality_score=4,
        source="MRV Agent Pipeline",
    )


@pytest.fixture
def sample_climate_target(sample_org_id) -> ClimateTarget:
    """Sample absolute emissions reduction target."""
    return ClimateTarget(
        org_id=sample_org_id,
        target_type=TargetType.ABSOLUTE,
        target_name="50% Scope 1+2 Reduction by 2030",
        base_year=2020,
        base_value=Decimal("200000"),
        target_year=2030,
        target_value=Decimal("100000"),
        unit="tCO2e",
        interim_milestones={2025: Decimal("150000"), 2028: Decimal("120000")},
        sbti_aligned=True,
        status="active",
    )


@pytest.fixture
def sample_target_progress() -> TargetProgress:
    """Sample target progress record."""
    return TargetProgress(
        target_id=_new_id(),
        reporting_year=2025,
        current_value=Decimal("155000"),
        progress_pct=Decimal("45.0"),
        gap_to_target=Decimal("55000"),
        on_track=True,
        notes="On track per linear trajectory",
    )


# ============================================================================
# DISCLOSURE FIXTURES
# ============================================================================

@pytest.fixture
def sample_disclosure_section() -> DisclosureSection:
    """Sample governance disclosure section."""
    return DisclosureSection(
        pillar=TCFDPillar.GOVERNANCE,
        disclosure_ref="gov_a",
        title="Board Oversight",
        content="The Board of Directors oversees climate-related risks through...",
        evidence_refs=["governance-charter-2025", "board-minutes-q4-2025"],
        compliance_score=75,
    )


@pytest.fixture
def sample_disclosure(sample_org_id) -> TCFDDisclosure:
    """Sample TCFD disclosure document."""
    sections = [
        DisclosureSection(
            pillar=TCFDPillar.GOVERNANCE,
            disclosure_ref="gov_a",
            title="Board Oversight",
            content="Board oversight of climate risks and opportunities.",
            compliance_score=80,
        ),
        DisclosureSection(
            pillar=TCFDPillar.GOVERNANCE,
            disclosure_ref="gov_b",
            title="Management Role",
            content="Management assesses and manages climate-related risks.",
            compliance_score=70,
        ),
        DisclosureSection(
            pillar=TCFDPillar.STRATEGY,
            disclosure_ref="str_a",
            title="Risks and Opportunities",
            content="Climate risks identified across short, medium, and long term.",
            compliance_score=65,
        ),
    ]
    return TCFDDisclosure(
        org_id=sample_org_id,
        reporting_year=2025,
        version=1,
        status=DisclosureStatus.DRAFT,
        sections=sections,
    )


# ============================================================================
# GAP ASSESSMENT FIXTURES
# ============================================================================

@pytest.fixture
def sample_gap_assessment(sample_org_id) -> GapAssessment:
    """Sample gap assessment."""
    return GapAssessment(
        org_id=sample_org_id,
        pillar_scores={
            "governance": 3,
            "strategy": 2,
            "risk_management": 3,
            "metrics_targets": 2,
        },
        overall_maturity=MaturityLevel.DEVELOPING,
        gaps=[
            {"pillar": "strategy", "disclosure": "str_c", "gap": "No scenario analysis performed"},
            {"pillar": "metrics_targets", "disclosure": "mt_c", "gap": "No SBTi-aligned targets"},
        ],
        actions=[
            {"action": "Conduct scenario analysis", "priority": "high", "timeline": "6 months"},
            {"action": "Set SBTi-validated target", "priority": "high", "timeline": "12 months"},
        ],
        peer_benchmark_percentile=35,
    )


# ============================================================================
# ISSB MAPPING FIXTURES
# ============================================================================

@pytest.fixture
def sample_issb_mappings() -> List[ISSBMapping]:
    """Sample TCFD-to-ISSB mappings."""
    return [
        ISSBMapping(
            tcfd_disclosure_ref="gov_a",
            ifrs_s2_paragraph="5-6",
            mapping_status="fully_mapped",
        ),
        ISSBMapping(
            tcfd_disclosure_ref="str_c",
            ifrs_s2_paragraph="22",
            mapping_status="enhanced",
            gap_description="IFRS S2 requires climate resilience assessment",
            action_required="Extend scenario analysis to cover resilience",
        ),
        ISSBMapping(
            tcfd_disclosure_ref="mt_b",
            ifrs_s2_paragraph="29(a)",
            mapping_status="enhanced",
            gap_description="IFRS S2 mandates Scope 3 for all entities",
            action_required="Report all 15 Scope 3 categories",
        ),
    ]


# ============================================================================
# RECOMMENDATION FIXTURES
# ============================================================================

@pytest.fixture
def sample_recommendation(sample_org_id) -> Recommendation:
    """Sample improvement recommendation."""
    return Recommendation(
        org_id=sample_org_id,
        category="strategy",
        priority=1,
        title="Conduct Multi-Scenario Analysis",
        description="Perform quantitative scenario analysis using IEA NZE and NGFS pathways",
        estimated_impact="high",
        estimated_effort="high",
        implementation_guidance="1. Select scenarios 2. Gather financial data 3. Model impacts",
    )


# ============================================================================
# REQUEST MODEL FIXTURES
# ============================================================================

@pytest.fixture
def governance_assessment_request() -> CreateGovernanceAssessmentRequest:
    """Sample governance assessment request."""
    return CreateGovernanceAssessmentRequest(
        board_oversight_score=4,
        board_committees=["Sustainability Committee"],
        meeting_frequency=4,
        climate_competency_score=3,
        incentive_linkage=True,
        incentive_pct=Decimal("15.0"),
    )


@pytest.fixture
def climate_risk_request() -> CreateClimateRiskRequest:
    """Sample climate risk creation request."""
    return CreateClimateRiskRequest(
        risk_type=RiskType.PHYSICAL_ACUTE,
        name="Wildfire Risk",
        description="Increasing wildfire frequency near California facilities",
        time_horizon=TimeHorizon.MEDIUM_TERM,
        likelihood=RiskLikelihood.LIKELY,
        impact=RiskImpact.MAJOR,
        financial_impact_estimate=Decimal("10000000"),
    )


@pytest.fixture
def scenario_request() -> CreateScenarioRequest:
    """Sample scenario creation request."""
    return CreateScenarioRequest(
        name="Custom Delayed Transition",
        scenario_type=ScenarioType.CUSTOM,
        temperature_outcome=TemperatureOutcome.AROUND_2C,
        description="Custom delayed transition scenario",
        carbon_price_trajectory={
            2025: Decimal("20"), 2030: Decimal("50"),
            2040: Decimal("150"), 2050: Decimal("300"),
        },
    )


@pytest.fixture
def register_asset_request() -> RegisterAssetRequest:
    """Sample asset registration request."""
    return RegisterAssetRequest(
        asset_name="London Headquarters",
        asset_type=AssetType.BUILDING,
        latitude=Decimal("51.5074"),
        longitude=Decimal("-0.1278"),
        country="GB",
        region="Greater London",
        elevation=Decimal("11"),
        building_type="Office",
        replacement_value=Decimal("200000000"),
        year_built=2005,
    )


@pytest.fixture
def create_target_request() -> CreateTargetRequest:
    """Sample target creation request."""
    return CreateTargetRequest(
        target_type=TargetType.NET_ZERO,
        target_name="Net Zero by 2050",
        base_year=2020,
        base_value=Decimal("200000"),
        target_year=2050,
        target_value=Decimal("0"),
        unit="tCO2e",
        sbti_aligned=True,
    )


# ============================================================================
# MOCK DATABASE SESSION FIXTURES
# ============================================================================

@pytest.fixture
def mock_db_session():
    """Mock async database session for integration-style tests."""
    session = MagicMock()
    session.execute = AsyncMock(return_value=MagicMock())
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    session.refresh = AsyncMock()
    return session


@pytest.fixture
def mock_db_results():
    """Mock database result set."""
    result = MagicMock()
    result.scalars = MagicMock(return_value=MagicMock())
    result.scalars.return_value.all = MagicMock(return_value=[])
    result.scalars.return_value.first = MagicMock(return_value=None)
    return result
