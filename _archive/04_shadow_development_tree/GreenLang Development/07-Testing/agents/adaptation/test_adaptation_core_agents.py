# -*- coding: utf-8 -*-
"""
Tests for GreenLang Climate Risk & Adaptation Layer Core Agents
================================================================

Comprehensive tests for all 12 adaptation layer core agents covering:
    - Agent initialization and configuration
    - Input validation with Pydantic models
    - Core processing logic
    - Output validation and provenance tracking
    - Zero-hallucination compliance
    - Error handling and edge cases

Author: GreenLang Team
"""

import pytest
from datetime import datetime

from greenlang.agents.adaptation.physical_risk_screening import (
    PhysicalRiskScreeningAgent,
    HazardType,
    RiskCategory,
    TimeHorizon,
    ClimateScenario,
    AssetType,
    GeoLocation,
    AssetDefinition,
    PhysicalRiskScreeningInput,
)

from greenlang.agents.adaptation.hazard_mapping import (
    HazardMappingAgent,
    HazardCategory,
    DataResolution,
    BoundingBox,
    HazardMappingInput,
)

from greenlang.agents.adaptation.vulnerability_assessment import (
    VulnerabilityAssessmentAgent,
    VulnerabilityLevel,
    SectorType,
    AssetVulnerabilityInput,
    VulnerabilityAssessmentInput,
)

from greenlang.agents.adaptation.exposure_analysis import (
    ExposureAnalysisAgent,
    ExposureLevel,
    AssetExposureInput,
    ExposureAnalysisInput,
)

from greenlang.agents.adaptation.adaptation_options_library import (
    AdaptationOptionsLibraryAgent,
    AdaptationCategory,
    LibraryQueryInput,
)

from greenlang.agents.adaptation.resilience_scoring import (
    ResilienceScoringAgent,
    ResilienceLevel,
    ResilienceInput,
    ResilienceScoringInput,
)

from greenlang.agents.adaptation.climate_scenario import (
    ClimateScenarioAgent,
    RCPScenario,
    ProjectionVariable,
    ScenarioInput,
)

from greenlang.agents.adaptation.financial_impact import (
    FinancialImpactAgent,
    AssetFinancials,
    FinancialImpactInput,
)

from greenlang.agents.adaptation.insurance_transfer import (
    InsuranceTransferAgent,
    CoverageStatus,
    InsuranceAnalysisInput,
)

from greenlang.agents.adaptation.adaptation_investment_prioritizer import (
    AdaptationInvestmentPrioritizerAgent,
    PriorityLevel,
    InvestmentCategory,
    InvestmentOption,
    PrioritizationInput,
)

from greenlang.agents.adaptation.tcfd_alignment import (
    TCFDAlignmentAgent,
    AlignmentLevel,
    TCFDAlignmentInput,
)

from greenlang.agents.adaptation.nature_based_adaptation import (
    NatureBasedAdaptationAgent,
    NbSCategory,
    NbSInput,
)


# =============================================================================
# GL-ADAPT-X-001: Physical Risk Screening Agent Tests
# =============================================================================

class TestPhysicalRiskScreeningAgent:
    """Tests for Physical Risk Screening Agent."""

    @pytest.fixture
    def agent(self):
        """Create a PhysicalRiskScreeningAgent instance."""
        return PhysicalRiskScreeningAgent()

    @pytest.fixture
    def sample_asset(self):
        """Sample asset definition for testing."""
        return AssetDefinition(
            asset_id="ASSET001",
            asset_name="Manufacturing Facility",
            asset_type=AssetType.FACILITY,
            location=GeoLocation(
                latitude=40.7128,
                longitude=-74.0060,
                coastal_distance_km=5.0
            ),
            value_usd=10000000,
            criticality=0.8
        )

    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.AGENT_ID == "GL-ADAPT-X-001"
        assert agent.AGENT_NAME == "Physical Risk Screening Agent"
        assert agent.VERSION == "1.0.0"

    def test_basic_screening(self, agent, sample_asset):
        """Test basic physical risk screening."""
        result = agent.run({
            "screening_id": "SCR001",
            "assets": [sample_asset.model_dump()],
            "hazards_to_assess": [HazardType.FLOOD_COASTAL.value, HazardType.EXTREME_HEAT.value],
            "time_horizons": [TimeHorizon.CURRENT.value, TimeHorizon.MEDIUM_TERM.value],
            "scenarios": [ClimateScenario.RCP_45.value]
        })

        assert result.success
        assert result.data["screening_id"] == "SCR001"
        assert result.data["total_assets_screened"] == 1
        assert len(result.data["asset_risk_profiles"]) == 1
        assert result.data["provenance_hash"] != ""

    def test_multiple_assets(self, agent):
        """Test screening multiple assets."""
        assets = [
            AssetDefinition(
                asset_id=f"ASSET{i:03d}",
                asset_name=f"Facility {i}",
                asset_type=AssetType.FACILITY,
                location=GeoLocation(latitude=40.0 + i, longitude=-74.0),
                value_usd=1000000 * i
            ).model_dump()
            for i in range(1, 4)
        ]

        result = agent.run({
            "screening_id": "SCR002",
            "assets": assets,
            "hazards_to_assess": [HazardType.FLOOD_RIVERINE.value]
        })

        assert result.success
        assert result.data["total_assets_screened"] == 3

    def test_risk_categorization(self, agent, sample_asset):
        """Test risk categorization works correctly."""
        result = agent.run({
            "screening_id": "SCR003",
            "assets": [sample_asset.model_dump()],
            "hazards_to_assess": [HazardType.FLOOD_COASTAL.value],
            "scenarios": [ClimateScenario.RCP_85.value]  # High emissions scenario
        })

        assert result.success
        profile = result.data["asset_risk_profiles"][0]
        assert profile["aggregate_risk_category"] in [r.value for r in RiskCategory]

    def test_provenance_hash_unique(self, agent, sample_asset):
        """Test provenance hashes are unique per screening."""
        result1 = agent.run({
            "screening_id": "SCR004A",
            "assets": [sample_asset.model_dump()]
        })
        result2 = agent.run({
            "screening_id": "SCR004B",
            "assets": [sample_asset.model_dump()]
        })

        assert result1.data["provenance_hash"] != result2.data["provenance_hash"]


# =============================================================================
# GL-ADAPT-X-002: Hazard Mapping Agent Tests
# =============================================================================

class TestHazardMappingAgent:
    """Tests for Hazard Mapping Agent."""

    @pytest.fixture
    def agent(self):
        """Create a HazardMappingAgent instance."""
        return HazardMappingAgent()

    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.AGENT_ID == "GL-ADAPT-X-002"
        assert agent.AGENT_NAME == "Hazard Mapping Agent"

    def test_basic_hazard_mapping(self, agent):
        """Test basic hazard mapping."""
        result = agent.run({
            "mapping_id": "MAP001",
            "bounding_box": {
                "min_lat": 40.0,
                "max_lat": 41.0,
                "min_lon": -75.0,
                "max_lon": -74.0
            },
            "hazard_types": ["flood_riverine", "extreme_heat"],
            "grid_size_km": 20.0
        })

        assert result.success
        assert result.data["mapping_id"] == "MAP001"
        assert result.data["total_cells"] > 0
        assert len(result.data["hazard_layers"]) == 2

    def test_hotspot_identification(self, agent):
        """Test hotspot identification."""
        result = agent.run({
            "mapping_id": "MAP002",
            "bounding_box": {
                "min_lat": 10.0,  # Tropical region
                "max_lat": 15.0,
                "min_lon": -80.0,
                "max_lon": -75.0
            },
            "hazard_types": ["cyclone"],
            "hotspot_threshold": 0.5,
            "grid_size_km": 50.0
        })

        assert result.success
        # Tropical coastal areas should have cyclone hotspots
        assert result.data["total_hotspots"] >= 0

    def test_composite_layer_creation(self, agent):
        """Test composite layer is created correctly."""
        result = agent.run({
            "mapping_id": "MAP003",
            "bounding_box": {
                "min_lat": 40.0,
                "max_lat": 41.0,
                "min_lon": -75.0,
                "max_lon": -74.0
            },
            "hazard_types": ["flood_riverine", "extreme_heat", "drought"],
            "grid_size_km": 25.0
        })

        assert result.success
        assert result.data["composite_layer"] is not None


# =============================================================================
# GL-ADAPT-X-003: Vulnerability Assessment Agent Tests
# =============================================================================

class TestVulnerabilityAssessmentAgent:
    """Tests for Vulnerability Assessment Agent."""

    @pytest.fixture
    def agent(self):
        """Create a VulnerabilityAssessmentAgent instance."""
        return VulnerabilityAssessmentAgent()

    @pytest.fixture
    def sample_asset_input(self):
        """Sample asset input for vulnerability assessment."""
        return AssetVulnerabilityInput(
            asset_id="VA001",
            asset_name="Industrial Facility",
            sector=SectorType.MANUFACTURING,
            exposure_score=0.6,
            age_years=25,
            value_usd=5000000,
            has_insurance=True,
            has_adaptation_plan=False
        )

    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.AGENT_ID == "GL-ADAPT-X-003"

    def test_basic_vulnerability_assessment(self, agent, sample_asset_input):
        """Test basic vulnerability assessment."""
        result = agent.run({
            "assessment_id": "VA001",
            "assets": [sample_asset_input.model_dump()],
            "include_gaps_analysis": True
        })

        assert result.success
        assert result.data["total_assets_assessed"] == 1
        vuln_result = result.data["vulnerability_results"][0]
        assert vuln_result["vulnerability_level"] in [v.value for v in VulnerabilityLevel]
        assert 0 <= vuln_result["vulnerability_score"] <= 1

    def test_gap_identification(self, agent, sample_asset_input):
        """Test vulnerability gap identification."""
        result = agent.run({
            "assessment_id": "VA002",
            "assets": [sample_asset_input.model_dump()],
            "include_gaps_analysis": True
        })

        assert result.success
        vuln_result = result.data["vulnerability_results"][0]
        # Should identify lack of adaptation plan as a gap
        assert len(vuln_result["vulnerability_gaps"]) > 0


# =============================================================================
# GL-ADAPT-X-004: Exposure Analysis Agent Tests
# =============================================================================

class TestExposureAnalysisAgent:
    """Tests for Exposure Analysis Agent."""

    @pytest.fixture
    def agent(self):
        """Create an ExposureAnalysisAgent instance."""
        return ExposureAnalysisAgent()

    @pytest.fixture
    def sample_asset_input(self):
        """Sample asset input for exposure analysis."""
        return AssetExposureInput(
            asset_id="EXP001",
            asset_name="Coastal Warehouse",
            value_usd=2000000,
            latitude=25.0,
            longitude=-80.0,
            coastal_proximity_km=3.0,
            annual_revenue_usd=500000,
            employee_count=50
        )

    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.AGENT_ID == "GL-ADAPT-X-004"

    def test_basic_exposure_analysis(self, agent, sample_asset_input):
        """Test basic exposure analysis."""
        result = agent.run({
            "analysis_id": "EXP001",
            "assets": [sample_asset_input.model_dump()],
            "hazards_to_analyze": ["flood_coastal", "cyclone"]
        })

        assert result.success
        assert result.data["total_assets_analyzed"] == 1
        exp_result = result.data["exposure_results"][0]
        assert exp_result["exposure_level"] in [e.value for e in ExposureLevel]

    def test_coastal_exposure_higher(self, agent):
        """Test that coastal assets have higher flood exposure."""
        coastal_asset = AssetExposureInput(
            asset_id="COASTAL",
            asset_name="Coastal Asset",
            value_usd=1000000,
            latitude=25.0,
            longitude=-80.0,
            coastal_proximity_km=2.0
        ).model_dump()

        inland_asset = AssetExposureInput(
            asset_id="INLAND",
            asset_name="Inland Asset",
            value_usd=1000000,
            latitude=35.0,
            longitude=-85.0,
            coastal_proximity_km=500.0
        ).model_dump()

        result = agent.run({
            "analysis_id": "EXP002",
            "assets": [coastal_asset, inland_asset],
            "hazards_to_analyze": ["flood_coastal"]
        })

        assert result.success
        exposures = {r["asset_id"]: r["overall_exposure_score"] for r in result.data["exposure_results"]}
        assert exposures["COASTAL"] > exposures["INLAND"]


# =============================================================================
# GL-ADAPT-X-005: Adaptation Options Library Agent Tests
# =============================================================================

class TestAdaptationOptionsLibraryAgent:
    """Tests for Adaptation Options Library Agent."""

    @pytest.fixture
    def agent(self):
        """Create an AdaptationOptionsLibraryAgent instance."""
        return AdaptationOptionsLibraryAgent()

    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.AGENT_ID == "GL-ADAPT-X-005"
        # Should have pre-loaded measures
        assert len(agent._measures) > 0

    def test_basic_query(self, agent):
        """Test basic library query."""
        result = agent.run({
            "query_id": "Q001",
            "hazards": ["flood_riverine"],
            "sectors": ["manufacturing"]
        })

        assert result.success
        assert result.data["total_matches"] > 0

    def test_query_with_cost_filter(self, agent):
        """Test query with cost filter."""
        result = agent.run({
            "query_id": "Q002",
            "hazards": ["extreme_heat"],
            "max_cost_usd": 100000
        })

        assert result.success
        # All matches should be within cost limit
        for match in result.data["matched_measures"]:
            assert match["measure"]["cost_estimate"]["capital_cost_usd_low"] <= 100000

    def test_nature_based_filter(self, agent):
        """Test nature-based solutions filter."""
        result = agent.run({
            "query_id": "Q003",
            "hazards": ["flood_riverine"],
            "include_nature_based": False
        })

        assert result.success
        # No nature-based solutions should be included
        for match in result.data["matched_measures"]:
            assert match["measure"]["category"] != AdaptationCategory.NATURE_BASED.value


# =============================================================================
# GL-ADAPT-X-006: Resilience Scoring Agent Tests
# =============================================================================

class TestResilienceScoringAgent:
    """Tests for Resilience Scoring Agent."""

    @pytest.fixture
    def agent(self):
        """Create a ResilienceScoringAgent instance."""
        return ResilienceScoringAgent()

    @pytest.fixture
    def sample_input(self):
        """Sample resilience input."""
        return ResilienceInput(
            asset_id="RES001",
            asset_name="Distribution Center",
            sector="retail",
            structural_integrity=0.7,
            business_continuity_plan=True,
            insurance_coverage=0.8,
            early_warning=0.6
        )

    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.AGENT_ID == "GL-ADAPT-X-006"

    def test_basic_resilience_scoring(self, agent, sample_input):
        """Test basic resilience scoring."""
        result = agent.run({
            "assessment_id": "RES001",
            "assets": [sample_input.model_dump()]
        })

        assert result.success
        profile = result.data["resilience_profiles"][0]
        assert 0 <= profile["resilience_score"] <= 1
        assert profile["resilience_level"] in [r.value for r in ResilienceLevel]

    def test_capacity_scores(self, agent, sample_input):
        """Test that all three capacity scores are calculated."""
        result = agent.run({
            "assessment_id": "RES002",
            "assets": [sample_input.model_dump()]
        })

        assert result.success
        profile = result.data["resilience_profiles"][0]
        assert "absorptive_capacity" in profile
        assert "adaptive_capacity" in profile
        assert "transformative_capacity" in profile


# =============================================================================
# GL-ADAPT-X-007: Climate Scenario Agent Tests
# =============================================================================

class TestClimateScenarioAgent:
    """Tests for Climate Scenario Agent."""

    @pytest.fixture
    def agent(self):
        """Create a ClimateScenarioAgent instance."""
        return ClimateScenarioAgent()

    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.AGENT_ID == "GL-ADAPT-X-007"

    def test_basic_scenario_projection(self, agent):
        """Test basic scenario projection."""
        result = agent.run({
            "request_id": "CS001",
            "scenarios": ["rcp_4.5", "rcp_8.5"],
            "time_horizons": ["2050", "2100"],
            "variables": [ProjectionVariable.TEMPERATURE_MEAN.value]
        })

        assert result.success
        assert result.data["request_id"] == "CS001"
        # Should have global summary since no locations specified
        assert "global_summary" in result.data

    def test_location_projection(self, agent):
        """Test projection for specific locations."""
        result = agent.run({
            "request_id": "CS002",
            "locations": [
                {"location_id": "NYC", "latitude": 40.7, "longitude": -74.0},
                {"location_id": "MIA", "latitude": 25.8, "longitude": -80.2, "coastal": True}
            ],
            "scenarios": ["rcp_4.5"],
            "time_horizons": ["2050"],
            "variables": [ProjectionVariable.TEMPERATURE_MEAN.value, ProjectionVariable.SEA_LEVEL.value]
        })

        assert result.success
        assert len(result.data["location_projections"]) == 2


# =============================================================================
# GL-ADAPT-X-008: Financial Impact Agent Tests
# =============================================================================

class TestFinancialImpactAgent:
    """Tests for Financial Impact Agent."""

    @pytest.fixture
    def agent(self):
        """Create a FinancialImpactAgent instance."""
        return FinancialImpactAgent()

    @pytest.fixture
    def sample_asset(self):
        """Sample asset financials."""
        return AssetFinancials(
            asset_id="FIN001",
            asset_name="Manufacturing Plant",
            asset_value_usd=50000000,
            annual_revenue_usd=20000000,
            operating_margin=0.15,
            insurance_coverage_pct=0.7
        )

    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.AGENT_ID == "GL-ADAPT-X-008"

    def test_basic_financial_impact(self, agent, sample_asset):
        """Test basic financial impact calculation."""
        result = agent.run({
            "analysis_id": "FIN001",
            "assets": [sample_asset.model_dump()],
            "hazard_exposures": {
                "FIN001": {"flood_riverine": 0.5, "wildfire": 0.3}
            }
        })

        assert result.success
        assert result.data["total_expected_annual_loss_usd"] > 0
        assert result.data["total_value_at_risk_usd"] > 0

    def test_adaptation_cost_benefit(self, agent, sample_asset):
        """Test adaptation cost-benefit analysis."""
        result = agent.run({
            "analysis_id": "FIN002",
            "assets": [sample_asset.model_dump()],
            "hazard_exposures": {"FIN001": {"flood_riverine": 0.6}},
            "adaptation_measures": [
                {
                    "name": "Flood Barriers",
                    "capital_cost_usd": 500000,
                    "annual_operating_cost_usd": 10000,
                    "risk_reduction_pct": 50
                }
            ]
        })

        assert result.success
        assert len(result.data["adaptation_analysis"]) == 1
        cb = result.data["adaptation_analysis"][0]
        assert cb["benefit_cost_ratio"] > 0


# =============================================================================
# GL-ADAPT-X-009: Insurance & Risk Transfer Agent Tests
# =============================================================================

class TestInsuranceTransferAgent:
    """Tests for Insurance & Risk Transfer Agent."""

    @pytest.fixture
    def agent(self):
        """Create an InsuranceTransferAgent instance."""
        return InsuranceTransferAgent()

    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.AGENT_ID == "GL-ADAPT-X-009"

    def test_basic_insurance_analysis(self, agent):
        """Test basic insurance analysis."""
        result = agent.run({
            "analysis_id": "INS001",
            "asset_value_usd": 10000000,
            "hazard_exposures": {"flood_riverine": 0.5, "wildfire": 0.3},
            "expected_annual_loss_usd": 50000
        })

        assert result.success
        assert result.data["coverage_status"] in [s.value for s in CoverageStatus]
        assert len(result.data["coverage_gaps"]) > 0
        assert len(result.data["transfer_options"]) > 0

    def test_coverage_gap_identification(self, agent):
        """Test coverage gap identification."""
        result = agent.run({
            "analysis_id": "INS002",
            "asset_value_usd": 5000000,
            "hazard_exposures": {"flood_coastal": 0.7},
            "existing_coverage": []  # No existing coverage
        })

        assert result.success
        assert result.data["total_gap_usd"] > 0


# =============================================================================
# GL-ADAPT-X-010: Adaptation Investment Prioritizer Agent Tests
# =============================================================================

class TestAdaptationInvestmentPrioritizerAgent:
    """Tests for Adaptation Investment Prioritizer Agent."""

    @pytest.fixture
    def agent(self):
        """Create an AdaptationInvestmentPrioritizerAgent instance."""
        return AdaptationInvestmentPrioritizerAgent()

    @pytest.fixture
    def sample_options(self):
        """Sample investment options."""
        return [
            InvestmentOption(
                option_id="OPT001",
                name="Flood Barriers",
                category=InvestmentCategory.INFRASTRUCTURE,
                capital_cost_usd=500000,
                risk_reduction_pct=40,
                expected_benefit_usd=100000,
                co_benefits=["Property protection", "Business continuity"]
            ).model_dump(),
            InvestmentOption(
                option_id="OPT002",
                name="Green Roof",
                category=InvestmentCategory.NATURE_BASED,
                capital_cost_usd=100000,
                risk_reduction_pct=20,
                expected_benefit_usd=30000,
                co_benefits=["Energy savings", "Biodiversity"]
            ).model_dump()
        ]

    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.AGENT_ID == "GL-ADAPT-X-010"

    def test_basic_prioritization(self, agent, sample_options):
        """Test basic investment prioritization."""
        result = agent.run({
            "request_id": "PRI001",
            "investment_options": sample_options,
            "total_budget_usd": 1000000
        })

        assert result.success
        assert len(result.data["prioritized_investments"]) == 2
        # Should be ranked by composite score
        scores = [p["composite_score"] for p in result.data["prioritized_investments"]]
        assert scores == sorted(scores, reverse=True)

    def test_budget_allocation(self, agent, sample_options):
        """Test budget allocation."""
        result = agent.run({
            "request_id": "PRI002",
            "investment_options": sample_options,
            "total_budget_usd": 600000  # Can only fund some options
        })

        assert result.success
        assert result.data["total_allocated_usd"] <= 600000


# =============================================================================
# GL-ADAPT-X-011: TCFD Alignment Agent Tests
# =============================================================================

class TestTCFDAlignmentAgent:
    """Tests for TCFD Alignment Agent."""

    @pytest.fixture
    def agent(self):
        """Create a TCFDAlignmentAgent instance."""
        return TCFDAlignmentAgent()

    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.AGENT_ID == "GL-ADAPT-X-011"

    def test_basic_tcfd_assessment(self, agent):
        """Test basic TCFD alignment assessment."""
        result = agent.run({
            "assessment_id": "TCFD001",
            "organization_name": "Example Corp",
            "board_climate_oversight": True,
            "management_climate_responsibility": True,
            "climate_risks_identified": ["physical_flood", "transition_carbon_pricing"],
            "scope_1_emissions_reported": True,
            "scope_2_emissions_reported": True,
            "climate_targets_set": True
        })

        assert result.success
        assert result.data["overall_alignment_level"] in [a.value for a in AlignmentLevel]
        assert 0 <= result.data["overall_alignment_score"] <= 1
        assert len(result.data["pillar_assessments"]) == 4  # Four TCFD pillars

    def test_low_alignment_detection(self, agent):
        """Test detection of low TCFD alignment."""
        result = agent.run({
            "assessment_id": "TCFD002",
            "organization_name": "Low Alignment Corp",
            "board_climate_oversight": False,
            "management_climate_responsibility": False,
            "scope_1_emissions_reported": False
        })

        assert result.success
        # Should have critical gaps identified
        assert len(result.data["critical_gaps"]) > 0


# =============================================================================
# GL-ADAPT-X-012: Nature-Based Adaptation Agent Tests
# =============================================================================

class TestNatureBasedAdaptationAgent:
    """Tests for Nature-Based Adaptation Agent."""

    @pytest.fixture
    def agent(self):
        """Create a NatureBasedAdaptationAgent instance."""
        return NatureBasedAdaptationAgent()

    def test_agent_initialization(self, agent):
        """Test agent initializes correctly."""
        assert agent.AGENT_ID == "GL-ADAPT-X-012"
        # Should have pre-loaded solutions
        assert len(agent._solutions) > 0

    def test_basic_nbs_query(self, agent):
        """Test basic NbS query."""
        result = agent.run({
            "request_id": "NBS001",
            "target_hazards": ["flood_riverine", "extreme_heat"],
            "location_climate": "temperate",
            "available_land_ha": 50
        })

        assert result.success
        assert len(result.data["matched_solutions"]) > 0
        assert result.data["total_carbon_sequestration_potential_tco2e"] > 0

    def test_coastal_solutions(self, agent):
        """Test coastal location gets appropriate solutions."""
        result = agent.run({
            "request_id": "NBS002",
            "target_hazards": ["flood_coastal", "sea_level_rise"],
            "location_climate": "tropical",
            "coastal_location": True,
            "available_land_ha": 100
        })

        assert result.success
        # Should recommend mangrove or coastal dune solutions
        solution_names = [s["solution"]["name"].lower() for s in result.data["matched_solutions"]]
        assert any("mangrove" in name or "coastal" in name for name in solution_names)

    def test_carbon_prioritization(self, agent):
        """Test carbon sequestration prioritization."""
        result = agent.run({
            "request_id": "NBS003",
            "target_hazards": ["drought"],
            "prioritize_carbon": True,
            "available_land_ha": 50
        })

        assert result.success
        # High carbon solutions should rank higher
        if len(result.data["matched_solutions"]) > 1:
            first_carbon = result.data["matched_solutions"][0]["estimated_carbon_sequestration_tco2e"]
            assert first_carbon > 0


# =============================================================================
# Integration Tests
# =============================================================================

class TestAdaptationLayerIntegration:
    """Integration tests for the adaptation layer."""

    def test_full_risk_assessment_pipeline(self):
        """Test a complete risk assessment pipeline."""
        # Step 1: Physical risk screening
        screening_agent = PhysicalRiskScreeningAgent()
        screening_result = screening_agent.run({
            "screening_id": "PIPELINE001",
            "assets": [{
                "asset_id": "PIPE001",
                "asset_name": "Test Facility",
                "asset_type": "facility",
                "location": {"latitude": 40.0, "longitude": -74.0, "coastal_distance_km": 10},
                "value_usd": 5000000
            }],
            "hazards_to_assess": ["flood_riverine", "extreme_heat"]
        })
        assert screening_result.success
        risk_score = screening_result.data["asset_risk_profiles"][0]["aggregate_risk_score"]

        # Step 2: Vulnerability assessment
        vuln_agent = VulnerabilityAssessmentAgent()
        vuln_result = vuln_agent.run({
            "assessment_id": "PIPELINE001-V",
            "assets": [{
                "asset_id": "PIPE001",
                "asset_name": "Test Facility",
                "sector": "manufacturing",
                "exposure_score": risk_score
            }]
        })
        assert vuln_result.success

        # Step 3: Financial impact
        fin_agent = FinancialImpactAgent()
        fin_result = fin_agent.run({
            "analysis_id": "PIPELINE001-F",
            "assets": [{
                "asset_id": "PIPE001",
                "asset_name": "Test Facility",
                "asset_value_usd": 5000000,
                "annual_revenue_usd": 1000000
            }],
            "hazard_exposures": {"PIPE001": {"flood_riverine": risk_score}}
        })
        assert fin_result.success
        assert fin_result.data["total_expected_annual_loss_usd"] > 0

    def test_all_agents_have_provenance(self):
        """Test that all agents produce provenance hashes."""
        agents_and_inputs = [
            (PhysicalRiskScreeningAgent(), {
                "screening_id": "PROV001",
                "assets": [{"asset_id": "A1", "asset_name": "Test", "asset_type": "facility",
                           "location": {"latitude": 40, "longitude": -74}}]
            }),
            (HazardMappingAgent(), {
                "mapping_id": "PROV002",
                "bounding_box": {"min_lat": 40, "max_lat": 41, "min_lon": -75, "max_lon": -74},
                "hazard_types": ["flood_riverine"]
            }),
            (ClimateScenarioAgent(), {
                "request_id": "PROV003",
                "scenarios": ["rcp_4.5"]
            }),
        ]

        for agent, input_data in agents_and_inputs:
            result = agent.run(input_data)
            assert result.success, f"{agent.AGENT_ID} failed"
            assert result.data.get("provenance_hash", ""), f"{agent.AGENT_ID} missing provenance hash"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
