# -*- coding: utf-8 -*-
"""
Unit tests for TCFD Platform Domain Models.

Tests all Pydantic v2 domain models including helpers, enum validation,
model creation, serialization/deserialization, field constraints,
nested model relationships, and provenance hashing with 30+ test functions.

Author: GL-TestEngineer
Date: March 2026
"""

from datetime import date, datetime
from decimal import Decimal

import pytest

from services.config import (
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
    _new_id,
    _now,
    _sha256,
)


# ===========================================================================
# Helper tests
# ===========================================================================

class TestHelpers:
    """Test utility functions."""

    def test_new_id_returns_string(self):
        result = _new_id()
        assert isinstance(result, str)
        assert len(result) == 36

    def test_new_id_unique(self):
        ids = {_new_id() for _ in range(200)}
        assert len(ids) == 200

    def test_now_returns_datetime(self):
        result = _now()
        assert isinstance(result, datetime)

    def test_now_no_microseconds(self):
        result = _now()
        assert result.microsecond == 0

    def test_sha256_deterministic(self):
        h1 = _sha256("tcfd_test_payload")
        h2 = _sha256("tcfd_test_payload")
        assert h1 == h2

    def test_sha256_length_64(self):
        result = _sha256("tcfd data")
        assert len(result) == 64

    def test_sha256_different_inputs(self):
        h1 = _sha256("input_alpha")
        h2 = _sha256("input_beta")
        assert h1 != h2


# ===========================================================================
# Enum tests
# ===========================================================================

class TestEnums:
    """Test all TCFD-specific enums."""

    def test_tcfd_pillar_values(self):
        pillars = [e.value for e in TCFDPillar]
        assert "governance" in pillars
        assert "strategy" in pillars
        assert "risk_management" in pillars
        assert "metrics_targets" in pillars
        assert len(pillars) == 4

    def test_risk_type_values(self):
        types = [e.value for e in RiskType]
        assert len(types) == 6
        assert "physical_acute" in types
        assert "transition_policy" in types

    def test_opportunity_category_values(self):
        cats = [e.value for e in OpportunityCategory]
        assert len(cats) == 5
        for expected in ["resource_efficiency", "energy_source", "products_services",
                         "markets", "resilience"]:
            assert expected in cats

    def test_scenario_type_values(self):
        types = [e.value for e in ScenarioType]
        assert len(types) == 8
        assert "iea_nze" in types
        assert "custom" in types

    def test_time_horizon_values(self):
        horizons = [e.value for e in TimeHorizon]
        assert len(horizons) == 3

    def test_temperature_outcome_values(self):
        temps = [e.value for e in TemperatureOutcome]
        assert len(temps) == 4
        assert "below_1_5c" in temps

    def test_physical_hazard_values(self):
        hazards = [e.value for e in PhysicalHazard]
        assert len(hazards) == 9
        assert "flood" in hazards

    def test_maturity_level_values(self):
        levels = [e.value for e in MaturityLevel]
        assert len(levels) == 5
        assert "initial" in levels
        assert "optimized" in levels

    def test_target_type_values(self):
        types = [e.value for e in TargetType]
        assert len(types) == 5
        assert "net_zero" in types

    def test_sector_type_values(self):
        sectors = [e.value for e in SectorType]
        assert len(sectors) == 11

    def test_issb_metric_type_values(self):
        metrics = [e.value for e in ISSBMetricType]
        assert len(metrics) == 7

    def test_disclosure_status_lifecycle(self):
        statuses = [e.value for e in DisclosureStatus]
        assert statuses == ["draft", "review", "approved", "published"]


# ===========================================================================
# Governance model tests
# ===========================================================================

class TestGovernanceRole:
    """Test GovernanceRole model."""

    def test_create_role(self, sample_governance_role):
        assert sample_governance_role.role_title == "Chief Sustainability Officer"
        assert sample_governance_role.climate_accountability is True
        assert len(sample_governance_role.id) == 36

    def test_role_competency_areas(self, sample_governance_role):
        assert "climate science" in sample_governance_role.competency_areas
        assert len(sample_governance_role.competency_areas) == 3


class TestGovernanceAssessment:
    """Test GovernanceAssessment model."""

    def test_create_assessment(self, sample_governance_assessment):
        assert sample_governance_assessment.board_oversight_score == 4
        assert sample_governance_assessment.overall_maturity == MaturityLevel.DEFINED

    def test_assessment_provenance_hash(self, sample_governance_assessment):
        assert len(sample_governance_assessment.provenance_hash) == 64

    def test_assessment_maturity_scores(self, sample_governance_assessment):
        scores = sample_governance_assessment.maturity_scores
        assert "board_oversight" in scores
        assert scores["board_oversight"] == 4

    def test_assessment_timestamps(self, sample_governance_assessment):
        assert isinstance(sample_governance_assessment.created_at, datetime)
        assert isinstance(sample_governance_assessment.updated_at, datetime)


# ===========================================================================
# Climate risk model tests
# ===========================================================================

class TestClimateRisk:
    """Test ClimateRisk model."""

    def test_create_climate_risk(self, sample_climate_risk):
        assert sample_climate_risk.risk_type == RiskType.PHYSICAL_ACUTE
        assert sample_climate_risk.name == "Coastal Flooding"
        assert sample_climate_risk.financial_impact_estimate == Decimal("25000000")

    def test_risk_provenance_hash(self, sample_climate_risk):
        assert len(sample_climate_risk.provenance_hash) == 64

    def test_risk_affected_assets(self, sample_climate_risk):
        assert len(sample_climate_risk.affected_assets) == 2
        assert "houston-refinery" in sample_climate_risk.affected_assets

    def test_risk_serialization(self, sample_climate_risk):
        data = sample_climate_risk.model_dump()
        assert data["risk_type"] == "physical_acute"
        assert "provenance_hash" in data


# ===========================================================================
# Climate opportunity model tests
# ===========================================================================

class TestClimateOpportunity:
    """Test ClimateOpportunity model."""

    def test_create_opportunity(self, sample_climate_opportunity):
        assert sample_climate_opportunity.category == OpportunityCategory.ENERGY_SOURCE
        assert sample_climate_opportunity.revenue_potential == Decimal("100000000")

    def test_opportunity_roi(self, sample_climate_opportunity):
        assert sample_climate_opportunity.roi_estimate == Decimal("0.25")

    def test_opportunity_provenance(self, sample_climate_opportunity):
        assert len(sample_climate_opportunity.provenance_hash) == 64


# ===========================================================================
# Scenario model tests
# ===========================================================================

class TestScenarioDefinition:
    """Test ScenarioDefinition model."""

    def test_create_scenario(self, sample_scenario_definition):
        assert sample_scenario_definition.scenario_type == ScenarioType.IEA_NZE
        assert sample_scenario_definition.temperature_outcome == TemperatureOutcome.BELOW_1_5C

    def test_scenario_carbon_trajectory(self, sample_scenario_definition):
        traj = sample_scenario_definition.carbon_price_trajectory
        assert traj[2030] == Decimal("130")
        assert traj[2050] == Decimal("250")

    def test_custom_scenario(self, custom_scenario_definition):
        assert custom_scenario_definition.scenario_type == ScenarioType.CUSTOM
        assert custom_scenario_definition.carbon_price_trajectory[2050] == Decimal("500")


class TestScenarioResult:
    """Test ScenarioResult model."""

    def test_create_result(self, sample_scenario_result):
        assert sample_scenario_result.revenue_impact_pct == Decimal("-8.5")
        assert sample_scenario_result.npv == Decimal("-120000000")

    def test_result_provenance(self, sample_scenario_result):
        assert len(sample_scenario_result.provenance_hash) == 64

    def test_result_confidence_interval(self, sample_scenario_result):
        assert sample_scenario_result.confidence_interval_lower < sample_scenario_result.confidence_interval_upper


class TestScenarioParameter:
    """Test ScenarioParameter model."""

    def test_create_parameter(self, sample_scenario_parameter):
        assert sample_scenario_parameter.parameter_name == "carbon_price"
        assert sample_scenario_parameter.year == 2030
        assert sample_scenario_parameter.value == Decimal("130.00")


# ===========================================================================
# Physical risk model tests
# ===========================================================================

class TestAssetLocation:
    """Test AssetLocation model."""

    def test_create_asset(self, sample_asset_location):
        assert sample_asset_location.asset_name == "Houston Refinery"
        assert sample_asset_location.latitude == Decimal("29.7604")

    def test_asset_replacement_value(self, sample_asset_location):
        assert sample_asset_location.replacement_value == Decimal("500000000")


class TestPhysicalRiskAssessment:
    """Test PhysicalRiskAssessment model."""

    def test_create_assessment(self, sample_physical_risk_assessment):
        assert sample_physical_risk_assessment.hazard_type == PhysicalHazard.FLOOD
        assert sample_physical_risk_assessment.composite_risk_score == Decimal("68.0")

    def test_assessment_provenance(self, sample_physical_risk_assessment):
        assert len(sample_physical_risk_assessment.provenance_hash) == 64


# ===========================================================================
# Financial impact model tests
# ===========================================================================

class TestFinancialImpact:
    """Test FinancialImpact model."""

    def test_create_impact(self, sample_financial_impact):
        assert sample_financial_impact.statement_area == FinancialStatementArea.INCOME_STATEMENT
        assert sample_financial_impact.line_item == "Revenue"

    def test_impact_auto_calculation(self, sample_financial_impact):
        expected_amount = Decimal("2287500000") - Decimal("2500000000")
        assert sample_financial_impact.impact_amount == expected_amount

    def test_impact_percentage_calculation(self, sample_financial_impact):
        assert sample_financial_impact.impact_pct < Decimal("0")

    def test_impact_provenance(self, sample_financial_impact):
        assert len(sample_financial_impact.provenance_hash) == 64


# ===========================================================================
# Risk management model tests
# ===========================================================================

class TestRiskManagementRecord:
    """Test RiskManagementRecord model."""

    def test_create_record(self, sample_risk_management_record):
        assert sample_risk_management_record.likelihood_score == 4
        assert sample_risk_management_record.erm_integrated is True

    def test_risk_score_auto_compute(self, sample_risk_management_record):
        assert sample_risk_management_record.risk_score == 16  # 4 * 4


# ===========================================================================
# Target model tests
# ===========================================================================

class TestClimateTarget:
    """Test ClimateTarget model."""

    def test_create_target(self, sample_climate_target):
        assert sample_climate_target.target_type == TargetType.ABSOLUTE
        assert sample_climate_target.sbti_aligned is True

    def test_target_year_after_base(self):
        with pytest.raises(Exception):
            ClimateTarget(
                org_id=_new_id(),
                target_type=TargetType.ABSOLUTE,
                target_name="Invalid Target",
                base_year=2030,
                base_value=Decimal("100000"),
                target_year=2020,
                target_value=Decimal("50000"),
            )

    def test_target_milestones(self, sample_climate_target):
        milestones = sample_climate_target.interim_milestones
        assert 2025 in milestones
        assert milestones[2025] == Decimal("150000")


class TestTargetProgress:
    """Test TargetProgress model."""

    def test_create_progress(self, sample_target_progress):
        assert sample_target_progress.progress_pct == Decimal("45.0")
        assert sample_target_progress.on_track is True

    def test_progress_provenance(self, sample_target_progress):
        assert len(sample_target_progress.provenance_hash) == 64


# ===========================================================================
# Disclosure model tests
# ===========================================================================

class TestTCFDDisclosure:
    """Test TCFDDisclosure model."""

    def test_create_disclosure(self, sample_disclosure):
        assert sample_disclosure.reporting_year == 2025
        assert sample_disclosure.status == DisclosureStatus.DRAFT

    def test_disclosure_auto_completeness(self, sample_disclosure):
        assert sample_disclosure.completeness_score > Decimal("0")

    def test_disclosure_provenance(self, sample_disclosure):
        assert len(sample_disclosure.provenance_hash) == 64

    def test_disclosure_sections_count(self, sample_disclosure):
        assert len(sample_disclosure.sections) == 3


# ===========================================================================
# Gap assessment model tests
# ===========================================================================

class TestGapAssessment:
    """Test GapAssessment model."""

    def test_create_gap_assessment(self, sample_gap_assessment):
        assert sample_gap_assessment.overall_maturity == MaturityLevel.DEVELOPING
        assert sample_gap_assessment.peer_benchmark_percentile == 35

    def test_gap_assessment_gaps(self, sample_gap_assessment):
        assert len(sample_gap_assessment.gaps) == 2

    def test_gap_assessment_provenance(self, sample_gap_assessment):
        assert len(sample_gap_assessment.provenance_hash) == 64


# ===========================================================================
# ISSB mapping model tests
# ===========================================================================

class TestISSBMapping:
    """Test ISSBMapping model."""

    def test_create_mapping(self, sample_issb_mappings):
        assert len(sample_issb_mappings) == 3
        assert sample_issb_mappings[0].mapping_status == "fully_mapped"

    def test_enhanced_mapping(self, sample_issb_mappings):
        enhanced = sample_issb_mappings[1]
        assert enhanced.mapping_status == "enhanced"
        assert enhanced.gap_description is not None


# ===========================================================================
# API response model tests
# ===========================================================================

class TestApiModels:
    """Test API response models."""

    def test_api_response(self):
        resp = ApiResponse(data={"key": "value"})
        assert resp.success is True
        assert resp.data == {"key": "value"}

    def test_api_error(self):
        err = ApiError(code="VALIDATION_ERROR", message="Invalid input")
        assert err.code == "VALIDATION_ERROR"

    def test_paginated_response(self):
        resp = PaginatedResponse(items=[1, 2, 3], total=100, page=1, page_size=50)
        assert resp.total_pages == 2
        assert resp.has_next is True
        assert resp.has_previous is False

    def test_paginated_last_page(self):
        resp = PaginatedResponse(items=[1], total=100, page=2, page_size=50)
        assert resp.has_next is False
        assert resp.has_previous is True
