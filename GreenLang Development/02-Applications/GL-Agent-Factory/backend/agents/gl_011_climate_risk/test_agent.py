"""
Unit Tests for GL-011: Climate Risk Assessment Agent

Comprehensive test coverage for the Climate Risk Agent including:
- Physical risk assessment (acute and chronic)
- Transition risk assessment (policy, technology, market, reputation)
- Scenario analysis (RCP/SSP pathways)
- Financial impact quantification
- Risk scoring calculations
- TCFD-aligned output generation
- Edge cases and validation

Test coverage target: 85%+
"""

import hashlib
import json
import pytest
from datetime import datetime
from typing import Dict, List, Any

from .agent import (
    ClimateRiskAgent,
    ClimateRiskInput,
    ClimateRiskOutput,
    GeoLocation,
    Asset,
    RevenueStream,
    CarbonExposure,
    MitigationMeasure,
    PhysicalRiskType,
    TransitionRiskType,
    ClimateScenario,
    TimeHorizon,
    RiskCategory,
    AssetType,
    SectorType,
    RiskScore,
    PhysicalRiskAssessment,
    TransitionRiskAssessment,
    ScenarioImpact,
    FinancialExposure,
    RiskRegister,
    ResilienceRecommendation,
    SCENARIO_PARAMETERS,
    SECTOR_TRANSITION_SENSITIVITY,
    PHYSICAL_RISK_BASELINE,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def agent() -> ClimateRiskAgent:
    """Create a ClimateRiskAgent instance for testing."""
    return ClimateRiskAgent()


@pytest.fixture
def sample_location() -> GeoLocation:
    """Create a sample coastal location (Miami)."""
    return GeoLocation(
        latitude=25.7617,
        longitude=-80.1918,
        country="US",
        region="Florida",
        city="Miami",
        elevation_m=2,
        coastal_distance_km=5,
    )


@pytest.fixture
def sample_inland_location() -> GeoLocation:
    """Create a sample inland location (Denver)."""
    return GeoLocation(
        latitude=39.7392,
        longitude=-104.9903,
        country="US",
        region="Colorado",
        city="Denver",
        elevation_m=1609,
        coastal_distance_km=1500,
    )


@pytest.fixture
def sample_assets() -> List[Asset]:
    """Create sample assets for testing."""
    return [
        Asset(
            name="Headquarters Building",
            asset_type=AssetType.REAL_ESTATE,
            value_usd=50_000_000,
            useful_life_years=30,
            insurance_coverage_usd=40_000_000,
        ),
        Asset(
            name="Manufacturing Plant",
            asset_type=AssetType.INFRASTRUCTURE,
            value_usd=100_000_000,
            useful_life_years=25,
            insurance_coverage_usd=80_000_000,
            carbon_intensity=50000,
        ),
        Asset(
            name="Equipment",
            asset_type=AssetType.EQUIPMENT,
            value_usd=20_000_000,
            useful_life_years=10,
            insurance_coverage_usd=15_000_000,
        ),
    ]


@pytest.fixture
def sample_revenue_streams() -> List[RevenueStream]:
    """Create sample revenue streams."""
    return [
        RevenueStream(
            name="Product Sales",
            annual_revenue_usd=500_000_000,
            sector=SectorType.MANUFACTURING,
            climate_sensitivity=0.6,
            geographic_exposure=["US", "EU", "APAC"],
        ),
        RevenueStream(
            name="Services",
            annual_revenue_usd=100_000_000,
            sector=SectorType.TECHNOLOGY,
            climate_sensitivity=0.3,
        ),
    ]


@pytest.fixture
def sample_carbon_exposure() -> CarbonExposure:
    """Create sample carbon exposure."""
    return CarbonExposure(
        annual_emissions_tco2e=100_000,
        scope1_emissions=30_000,
        scope2_emissions=20_000,
        scope3_emissions=50_000,
        carbon_intensity_revenue=166.67,
    )


@pytest.fixture
def sample_mitigation_measures() -> List[MitigationMeasure]:
    """Create sample mitigation measures."""
    return [
        MitigationMeasure(
            name="Flood barriers",
            risk_type="flood",
            effectiveness=0.5,
            implementation_cost_usd=2_000_000,
            implementation_status="implemented",
        ),
        MitigationMeasure(
            name="Carbon reduction program",
            risk_type="carbon_pricing",
            effectiveness=0.3,
            implementation_cost_usd=5_000_000,
            implementation_status="in_progress",
        ),
    ]


@pytest.fixture
def basic_input(sample_location: GeoLocation) -> ClimateRiskInput:
    """Create basic input for simple tests."""
    return ClimateRiskInput(
        organization_name="Test Corp",
        sector=SectorType.MANUFACTURING,
        location=sample_location,
        time_horizon_years=10,
        scenario=ClimateScenario.RCP_4_5,
    )


@pytest.fixture
def comprehensive_input(
    sample_location: GeoLocation,
    sample_assets: List[Asset],
    sample_revenue_streams: List[RevenueStream],
    sample_carbon_exposure: CarbonExposure,
    sample_mitigation_measures: List[MitigationMeasure],
) -> ClimateRiskInput:
    """Create comprehensive input for full tests."""
    return ClimateRiskInput(
        organization_name="Comprehensive Corp",
        sector=SectorType.MANUFACTURING,
        assets=sample_assets,
        revenue_streams=sample_revenue_streams,
        location=sample_location,
        carbon_exposure=sample_carbon_exposure,
        mitigation_measures=sample_mitigation_measures,
        time_horizon_years=15,
        scenario=ClimateScenario.RCP_4_5,
        scenarios_to_compare=[
            ClimateScenario.RCP_2_6,
            ClimateScenario.RCP_8_5,
        ],
        discount_rate=0.05,
    )


# =============================================================================
# Basic Agent Tests
# =============================================================================


class TestClimateRiskAgentInitialization:
    """Tests for agent initialization."""

    def test_agent_initialization(self, agent: ClimateRiskAgent):
        """Test agent initializes correctly."""
        assert agent is not None
        assert agent.AGENT_ID == "risk/climate_risk_v1"
        assert agent.VERSION == "1.0.0"

    def test_agent_with_config(self):
        """Test agent initialization with config."""
        config = {"custom_setting": "value"}
        agent = ClimateRiskAgent(config=config)
        assert agent.config == config

    def test_scenario_parameters_loaded(self, agent: ClimateRiskAgent):
        """Test scenario parameters are loaded."""
        assert len(agent.scenario_parameters) > 0
        assert ClimateScenario.RCP_2_6 in agent.scenario_parameters
        assert ClimateScenario.RCP_4_5 in agent.scenario_parameters
        assert ClimateScenario.RCP_8_5 in agent.scenario_parameters

    def test_sector_sensitivity_loaded(self, agent: ClimateRiskAgent):
        """Test sector sensitivity data is loaded."""
        assert len(agent.sector_sensitivity) > 0
        assert SectorType.ENERGY in agent.sector_sensitivity
        assert SectorType.MANUFACTURING in agent.sector_sensitivity


# =============================================================================
# Input Validation Tests
# =============================================================================


class TestInputValidation:
    """Tests for input validation."""

    def test_valid_basic_input(self, basic_input: ClimateRiskInput):
        """Test valid basic input passes validation."""
        assert basic_input.organization_name == "Test Corp"
        assert basic_input.scenario == ClimateScenario.RCP_4_5

    def test_valid_comprehensive_input(self, comprehensive_input: ClimateRiskInput):
        """Test comprehensive input passes validation."""
        assert len(comprehensive_input.assets) == 3
        assert len(comprehensive_input.revenue_streams) == 2
        assert comprehensive_input.carbon_exposure is not None

    def test_geo_location_validation(self):
        """Test geographic location validation."""
        # Valid location
        loc = GeoLocation(latitude=45.0, longitude=-90.0, country="US")
        assert loc.latitude == 45.0

        # Invalid latitude should raise
        with pytest.raises(ValueError):
            GeoLocation(latitude=100.0, longitude=0, country="US")

        # Invalid longitude should raise
        with pytest.raises(ValueError):
            GeoLocation(latitude=0, longitude=200.0, country="US")

    def test_asset_validation(self):
        """Test asset validation."""
        # Valid asset
        asset = Asset(
            name="Test Asset",
            asset_type=AssetType.REAL_ESTATE,
            value_usd=1_000_000,
        )
        assert asset.value_usd == 1_000_000

        # Negative value should raise
        with pytest.raises(ValueError):
            Asset(
                name="Invalid Asset",
                asset_type=AssetType.REAL_ESTATE,
                value_usd=-1000,
            )

    def test_time_horizon_validation(self):
        """Test time horizon validation."""
        # Valid time horizons
        for years in [1, 10, 30, 50]:
            input_data = ClimateRiskInput(
                organization_name="Test",
                location=GeoLocation(latitude=0, longitude=0, country="US"),
                time_horizon_years=years,
            )
            assert input_data.time_horizon_years == years

        # Invalid time horizon should raise
        with pytest.raises(ValueError):
            ClimateRiskInput(
                organization_name="Test",
                location=GeoLocation(latitude=0, longitude=0, country="US"),
                time_horizon_years=0,
            )

        with pytest.raises(ValueError):
            ClimateRiskInput(
                organization_name="Test",
                location=GeoLocation(latitude=0, longitude=0, country="US"),
                time_horizon_years=100,
            )


# =============================================================================
# Physical Risk Assessment Tests
# =============================================================================


class TestPhysicalRiskAssessment:
    """Tests for physical risk assessment."""

    def test_physical_risk_assessment_runs(
        self,
        agent: ClimateRiskAgent,
        comprehensive_input: ClimateRiskInput,
    ):
        """Test physical risk assessment executes successfully."""
        result = agent._assess_physical_risks(comprehensive_input)

        assert result is not None
        assert isinstance(result, PhysicalRiskAssessment)
        assert len(result.acute_risks) > 0
        assert len(result.chronic_risks) > 0

    def test_acute_risks_assessed(
        self,
        agent: ClimateRiskAgent,
        comprehensive_input: ClimateRiskInput,
    ):
        """Test acute physical risks are assessed."""
        result = agent._assess_physical_risks(comprehensive_input)

        acute_types = [r.risk_type for r in result.acute_risks]
        assert "flood" in acute_types
        assert "cyclone" in acute_types
        assert "wildfire" in acute_types
        assert "extreme_heat" in acute_types
        assert "drought" in acute_types

    def test_chronic_risks_assessed(
        self,
        agent: ClimateRiskAgent,
        comprehensive_input: ClimateRiskInput,
    ):
        """Test chronic physical risks are assessed."""
        result = agent._assess_physical_risks(comprehensive_input)

        chronic_types = [r.risk_type for r in result.chronic_risks]
        assert "sea_level_rise" in chronic_types
        assert "temperature_increase" in chronic_types

    def test_coastal_location_higher_flood_risk(
        self,
        agent: ClimateRiskAgent,
        sample_location: GeoLocation,
        sample_inland_location: GeoLocation,
    ):
        """Test coastal locations have higher flood/sea level rise risk."""
        coastal_input = ClimateRiskInput(
            organization_name="Coastal Corp",
            location=sample_location,
            assets=[Asset(name="Building", asset_type=AssetType.REAL_ESTATE, value_usd=10_000_000)],
        )

        inland_input = ClimateRiskInput(
            organization_name="Inland Corp",
            location=sample_inland_location,
            assets=[Asset(name="Building", asset_type=AssetType.REAL_ESTATE, value_usd=10_000_000)],
        )

        coastal_result = agent._assess_physical_risks(coastal_input)
        inland_result = agent._assess_physical_risks(inland_input)

        # Find sea level rise risk
        coastal_slr = next(
            (r for r in coastal_result.chronic_risks if r.risk_type == "sea_level_rise"),
            None
        )
        inland_slr = next(
            (r for r in inland_result.chronic_risks if r.risk_type == "sea_level_rise"),
            None
        )

        assert coastal_slr is not None
        assert inland_slr is not None
        assert coastal_slr.adjusted_score >= inland_slr.adjusted_score

    def test_risk_score_formula(
        self,
        agent: ClimateRiskAgent,
        comprehensive_input: ClimateRiskInput,
    ):
        """Test risk score follows formula: likelihood * impact * (1 - mitigation)."""
        result = agent._assess_physical_risks(comprehensive_input)

        for risk in result.acute_risks + result.chronic_risks:
            expected_raw = risk.likelihood * risk.impact
            expected_adjusted = expected_raw * (1 - risk.mitigation_effectiveness)

            assert abs(risk.raw_score - expected_raw) < 0.01
            assert abs(risk.adjusted_score - expected_adjusted) < 0.01

    def test_mitigation_reduces_risk_score(
        self,
        agent: ClimateRiskAgent,
        sample_location: GeoLocation,
        sample_assets: List[Asset],
    ):
        """Test mitigation measures reduce risk scores."""
        # Without mitigation
        input_no_mitigation = ClimateRiskInput(
            organization_name="Test Corp",
            location=sample_location,
            assets=sample_assets,
            mitigation_measures=[],
        )

        # With mitigation
        input_with_mitigation = ClimateRiskInput(
            organization_name="Test Corp",
            location=sample_location,
            assets=sample_assets,
            mitigation_measures=[
                MitigationMeasure(
                    name="Flood protection",
                    risk_type="flood",
                    effectiveness=0.6,
                    implementation_status="implemented",
                )
            ],
        )

        result_no_mit = agent._assess_physical_risks(input_no_mitigation)
        result_with_mit = agent._assess_physical_risks(input_with_mitigation)

        flood_no_mit = next(
            r for r in result_no_mit.acute_risks if r.risk_type == "flood"
        )
        flood_with_mit = next(
            r for r in result_with_mit.acute_risks if r.risk_type == "flood"
        )

        assert flood_with_mit.adjusted_score < flood_no_mit.adjusted_score


# =============================================================================
# Transition Risk Assessment Tests
# =============================================================================


class TestTransitionRiskAssessment:
    """Tests for transition risk assessment."""

    def test_transition_risk_assessment_runs(
        self,
        agent: ClimateRiskAgent,
        comprehensive_input: ClimateRiskInput,
    ):
        """Test transition risk assessment executes successfully."""
        result = agent._assess_transition_risks(comprehensive_input)

        assert result is not None
        assert isinstance(result, TransitionRiskAssessment)

    def test_policy_risks_assessed(
        self,
        agent: ClimateRiskAgent,
        comprehensive_input: ClimateRiskInput,
    ):
        """Test policy risks are assessed."""
        result = agent._assess_transition_risks(comprehensive_input)

        assert len(result.policy_risks) > 0
        policy_types = [r.risk_type for r in result.policy_risks]
        assert "carbon_pricing" in policy_types
        assert "regulation" in policy_types

    def test_technology_risks_assessed(
        self,
        agent: ClimateRiskAgent,
        comprehensive_input: ClimateRiskInput,
    ):
        """Test technology risks are assessed."""
        result = agent._assess_transition_risks(comprehensive_input)

        assert len(result.technology_risks) > 0
        tech_types = [r.risk_type for r in result.technology_risks]
        assert "technology_disruption" in tech_types

    def test_market_risks_assessed(
        self,
        agent: ClimateRiskAgent,
        comprehensive_input: ClimateRiskInput,
    ):
        """Test market risks are assessed."""
        result = agent._assess_transition_risks(comprehensive_input)

        assert len(result.market_risks) > 0
        market_types = [r.risk_type for r in result.market_risks]
        assert "demand_shift" in market_types

    def test_reputation_risks_assessed(
        self,
        agent: ClimateRiskAgent,
        comprehensive_input: ClimateRiskInput,
    ):
        """Test reputation risks are assessed."""
        result = agent._assess_transition_risks(comprehensive_input)

        assert len(result.reputation_risks) > 0
        rep_types = [r.risk_type for r in result.reputation_risks]
        assert "stakeholder_concern" in rep_types

    def test_carbon_pricing_impact_calculated(
        self,
        agent: ClimateRiskAgent,
        comprehensive_input: ClimateRiskInput,
    ):
        """Test carbon pricing impact is calculated."""
        result = agent._assess_transition_risks(comprehensive_input)

        assert result.carbon_price_impact_usd is not None
        assert result.carbon_price_impact_usd > 0

    def test_high_carbon_sector_higher_transition_risk(
        self,
        agent: ClimateRiskAgent,
        sample_location: GeoLocation,
    ):
        """Test high-carbon sectors have higher transition risk."""
        energy_input = ClimateRiskInput(
            organization_name="Energy Corp",
            sector=SectorType.ENERGY,
            location=sample_location,
            carbon_exposure=CarbonExposure(annual_emissions_tco2e=500_000),
        )

        tech_input = ClimateRiskInput(
            organization_name="Tech Corp",
            sector=SectorType.TECHNOLOGY,
            location=sample_location,
            carbon_exposure=CarbonExposure(annual_emissions_tco2e=10_000),
        )

        energy_result = agent._assess_transition_risks(energy_input)
        tech_result = agent._assess_transition_risks(tech_input)

        assert energy_result.total_transition_risk_score > tech_result.total_transition_risk_score


# =============================================================================
# Scenario Analysis Tests
# =============================================================================


class TestScenarioAnalysis:
    """Tests for scenario analysis."""

    def test_scenario_analysis_runs(
        self,
        agent: ClimateRiskAgent,
        comprehensive_input: ClimateRiskInput,
    ):
        """Test scenario analysis executes successfully."""
        result = agent._analyze_scenario(
            comprehensive_input,
            ClimateScenario.RCP_4_5
        )

        assert result is not None
        assert isinstance(result, ScenarioImpact)

    def test_all_scenarios_analyzable(
        self,
        agent: ClimateRiskAgent,
        basic_input: ClimateRiskInput,
    ):
        """Test all scenarios can be analyzed."""
        for scenario in ClimateScenario:
            result = agent._analyze_scenario(basic_input, scenario)
            assert result is not None
            assert result.scenario == scenario.value

    def test_higher_scenario_higher_physical_risk(
        self,
        agent: ClimateRiskAgent,
        basic_input: ClimateRiskInput,
    ):
        """Test higher scenarios have higher physical risk multipliers."""
        rcp26_result = agent._analyze_scenario(basic_input, ClimateScenario.RCP_2_6)
        rcp45_result = agent._analyze_scenario(basic_input, ClimateScenario.RCP_4_5)
        rcp85_result = agent._analyze_scenario(basic_input, ClimateScenario.RCP_8_5)

        assert rcp26_result.physical_risk_multiplier < rcp45_result.physical_risk_multiplier
        assert rcp45_result.physical_risk_multiplier < rcp85_result.physical_risk_multiplier

    def test_higher_scenario_lower_transition_risk(
        self,
        agent: ClimateRiskAgent,
        basic_input: ClimateRiskInput,
    ):
        """Test higher scenarios have lower transition risk (less policy action)."""
        rcp26_result = agent._analyze_scenario(basic_input, ClimateScenario.RCP_2_6)
        rcp85_result = agent._analyze_scenario(basic_input, ClimateScenario.RCP_8_5)

        # RCP 2.6 has high transition risk (strong policy)
        # RCP 8.5 has low transition risk (minimal policy)
        assert rcp26_result.transition_risk_multiplier > rcp85_result.transition_risk_multiplier

    def test_temperature_projections_increase_with_time(
        self,
        agent: ClimateRiskAgent,
        sample_location: GeoLocation,
    ):
        """Test temperature projections increase with time horizon."""
        short_input = ClimateRiskInput(
            organization_name="Test",
            location=sample_location,
            time_horizon_years=5,
            scenario=ClimateScenario.RCP_4_5,
        )

        long_input = ClimateRiskInput(
            organization_name="Test",
            location=sample_location,
            time_horizon_years=30,
            scenario=ClimateScenario.RCP_4_5,
        )

        short_result = agent._analyze_scenario(short_input, ClimateScenario.RCP_4_5)
        long_result = agent._analyze_scenario(long_input, ClimateScenario.RCP_4_5)

        assert long_result.temperature_increase_c > short_result.temperature_increase_c


# =============================================================================
# Financial Exposure Tests
# =============================================================================


class TestFinancialExposure:
    """Tests for financial exposure calculations."""

    def test_financial_exposure_calculated(
        self,
        agent: ClimateRiskAgent,
        comprehensive_input: ClimateRiskInput,
    ):
        """Test financial exposure is calculated."""
        physical = agent._assess_physical_risks(comprehensive_input)
        transition = agent._assess_transition_risks(comprehensive_input)
        scenario = agent._analyze_scenario(comprehensive_input, comprehensive_input.scenario)

        result = agent._calculate_financial_exposure(
            comprehensive_input, physical, transition, scenario
        )

        assert result is not None
        assert isinstance(result, FinancialExposure)
        assert result.total_financial_exposure_usd > 0

    def test_asset_value_at_risk_calculated(
        self,
        agent: ClimateRiskAgent,
        comprehensive_input: ClimateRiskInput,
    ):
        """Test asset value at risk is calculated."""
        physical = agent._assess_physical_risks(comprehensive_input)
        transition = agent._assess_transition_risks(comprehensive_input)
        scenario = agent._analyze_scenario(comprehensive_input, comprehensive_input.scenario)

        result = agent._calculate_financial_exposure(
            comprehensive_input, physical, transition, scenario
        )

        assert result.total_asset_value_at_risk_usd >= 0

    def test_insurance_gap_calculated(
        self,
        agent: ClimateRiskAgent,
        comprehensive_input: ClimateRiskInput,
    ):
        """Test insurance gap is calculated."""
        physical = agent._assess_physical_risks(comprehensive_input)
        transition = agent._assess_transition_risks(comprehensive_input)
        scenario = agent._analyze_scenario(comprehensive_input, comprehensive_input.scenario)

        result = agent._calculate_financial_exposure(
            comprehensive_input, physical, transition, scenario
        )

        # Insurance gap should be non-negative
        assert result.insurance_gap_usd >= 0

    def test_carbon_liability_with_emissions(
        self,
        agent: ClimateRiskAgent,
        comprehensive_input: ClimateRiskInput,
    ):
        """Test carbon liability is calculated when emissions present."""
        physical = agent._assess_physical_risks(comprehensive_input)
        transition = agent._assess_transition_risks(comprehensive_input)
        scenario = agent._analyze_scenario(comprehensive_input, comprehensive_input.scenario)

        result = agent._calculate_financial_exposure(
            comprehensive_input, physical, transition, scenario
        )

        # Should have carbon liability since we have emissions
        assert result.carbon_liability_usd > 0


# =============================================================================
# Risk Scoring Tests
# =============================================================================


class TestRiskScoring:
    """Tests for risk scoring calculations."""

    def test_probability_to_score_conversion(self, agent: ClimateRiskAgent):
        """Test probability to score conversion."""
        assert agent._probability_to_score(0.0) == 1
        assert agent._probability_to_score(0.1) == 1
        assert agent._probability_to_score(0.3) == 2
        assert agent._probability_to_score(0.5) == 3
        assert agent._probability_to_score(0.7) == 4
        assert agent._probability_to_score(0.9) == 5
        assert agent._probability_to_score(1.0) == 5

    def test_score_to_category_conversion(self, agent: ClimateRiskAgent):
        """Test score to category conversion."""
        assert agent._score_to_category(1) == RiskCategory.MINIMAL
        assert agent._score_to_category(4) == RiskCategory.LOW
        assert agent._score_to_category(8) == RiskCategory.MEDIUM
        assert agent._score_to_category(15) == RiskCategory.HIGH
        assert agent._score_to_category(25) == RiskCategory.CRITICAL

    def test_total_risk_score_calculation(self, agent: ClimateRiskAgent):
        """Test total risk score calculation formula."""
        # Formula: total = (physical * 0.5) + (transition * 0.5)
        physical_score = 10.0
        transition_score = 14.0

        expected = (physical_score * 0.5) + (transition_score * 0.5)
        result = agent._calculate_total_risk_score(physical_score, transition_score)

        assert result == expected

    def test_climate_zone_determination(self, agent: ClimateRiskAgent):
        """Test climate zone determination from latitude."""
        assert agent._get_climate_zone(0) == "tropical"
        assert agent._get_climate_zone(15) == "tropical"
        assert agent._get_climate_zone(25) == "subtropical"
        assert agent._get_climate_zone(40) == "temperate"
        assert agent._get_climate_zone(60) == "continental"
        assert agent._get_climate_zone(70) == "polar"

    def test_coastal_category_determination(self, agent: ClimateRiskAgent):
        """Test coastal category determination."""
        assert agent._get_coastal_category(None) == "inland"
        assert agent._get_coastal_category(5) == "coastal_0_10km"
        assert agent._get_coastal_category(25) == "coastal_10_50km"
        assert agent._get_coastal_category(75) == "coastal_50_100km"
        assert agent._get_coastal_category(200) == "inland"

    def test_time_horizon_determination(self, agent: ClimateRiskAgent):
        """Test time horizon category determination."""
        assert agent._get_time_horizon(3) == TimeHorizon.SHORT_TERM
        assert agent._get_time_horizon(5) == TimeHorizon.SHORT_TERM
        assert agent._get_time_horizon(10) == TimeHorizon.MEDIUM_TERM
        assert agent._get_time_horizon(15) == TimeHorizon.MEDIUM_TERM
        assert agent._get_time_horizon(20) == TimeHorizon.LONG_TERM
        assert agent._get_time_horizon(30) == TimeHorizon.LONG_TERM


# =============================================================================
# Full Agent Run Tests
# =============================================================================


class TestFullAgentRun:
    """Tests for complete agent execution."""

    def test_basic_run(
        self,
        agent: ClimateRiskAgent,
        basic_input: ClimateRiskInput,
    ):
        """Test basic agent run completes successfully."""
        result = agent.run(basic_input)

        assert result is not None
        assert isinstance(result, ClimateRiskOutput)
        assert result.organization_name == "Test Corp"

    def test_comprehensive_run(
        self,
        agent: ClimateRiskAgent,
        comprehensive_input: ClimateRiskInput,
    ):
        """Test comprehensive agent run completes successfully."""
        result = agent.run(comprehensive_input)

        assert result is not None
        assert result.organization_name == "Comprehensive Corp"
        assert result.total_risk_score > 0
        assert result.provenance_hash is not None

    def test_output_contains_all_components(
        self,
        agent: ClimateRiskAgent,
        comprehensive_input: ClimateRiskInput,
    ):
        """Test output contains all required components."""
        result = agent.run(comprehensive_input)

        # Core assessments
        assert result.physical_risk_assessment is not None
        assert result.transition_risk_assessment is not None

        # Scores
        assert result.total_risk_score >= 0
        assert result.overall_risk_level is not None

        # Scenario analysis
        assert result.primary_scenario_impact is not None
        assert len(result.scenario_comparison) > 0

        # Financial exposure
        assert result.financial_exposure is not None

        # TCFD outputs
        assert result.risk_register is not None
        assert len(result.resilience_recommendations) >= 0

        # Audit trail
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256
        assert result.processing_time_ms > 0

    def test_provenance_hash_unique_per_run(
        self,
        agent: ClimateRiskAgent,
        basic_input: ClimateRiskInput,
    ):
        """Test provenance hash is unique per run (includes timestamp)."""
        result1 = agent.run(basic_input)
        result2 = agent.run(basic_input)

        # Hashes should be different due to timestamp
        assert result1.provenance_hash != result2.provenance_hash

    def test_risk_register_generated(
        self,
        agent: ClimateRiskAgent,
        comprehensive_input: ClimateRiskInput,
    ):
        """Test risk register is properly generated."""
        result = agent.run(comprehensive_input)
        register = result.risk_register

        assert register.organization == "Comprehensive Corp"
        assert register.total_risks_assessed > 0
        assert len(register.physical_risks) > 0
        assert len(register.transition_risks) > 0

    def test_recommendations_generated(
        self,
        agent: ClimateRiskAgent,
        comprehensive_input: ClimateRiskInput,
    ):
        """Test recommendations are generated for high risks."""
        result = agent.run(comprehensive_input)

        # Should have some recommendations
        assert len(result.resilience_recommendations) > 0

        # Recommendations should be prioritized
        priorities = [r.priority for r in result.resilience_recommendations]
        assert priorities == sorted(priorities)


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_assets(
        self,
        agent: ClimateRiskAgent,
        sample_location: GeoLocation,
    ):
        """Test handling of zero assets."""
        input_data = ClimateRiskInput(
            organization_name="No Assets Corp",
            location=sample_location,
            assets=[],
        )

        result = agent.run(input_data)
        assert result is not None
        assert result.financial_exposure.total_asset_value_at_risk_usd == 0

    def test_zero_emissions(
        self,
        agent: ClimateRiskAgent,
        sample_location: GeoLocation,
    ):
        """Test handling of zero emissions."""
        input_data = ClimateRiskInput(
            organization_name="Clean Corp",
            location=sample_location,
            carbon_exposure=CarbonExposure(annual_emissions_tco2e=0),
        )

        result = agent.run(input_data)
        assert result is not None
        assert result.financial_exposure.carbon_liability_usd == 0

    def test_extreme_latitude_polar(
        self,
        agent: ClimateRiskAgent,
    ):
        """Test polar location risk assessment."""
        polar_location = GeoLocation(
            latitude=75.0,
            longitude=0.0,
            country="NO",
        )

        input_data = ClimateRiskInput(
            organization_name="Arctic Corp",
            location=polar_location,
        )

        result = agent.run(input_data)
        assert result is not None

        # Polar regions should have lower cyclone risk
        cyclone_risk = next(
            (r for r in result.physical_risk_assessment.acute_risks
             if r.risk_type == "cyclone"),
            None
        )
        assert cyclone_risk is not None
        assert cyclone_risk.likelihood <= 2

    def test_extreme_latitude_tropical(
        self,
        agent: ClimateRiskAgent,
    ):
        """Test tropical location risk assessment."""
        tropical_location = GeoLocation(
            latitude=5.0,
            longitude=-75.0,
            country="CO",
        )

        input_data = ClimateRiskInput(
            organization_name="Tropical Corp",
            location=tropical_location,
        )

        result = agent.run(input_data)
        assert result is not None

        # Tropical regions should have higher cyclone risk
        cyclone_risk = next(
            (r for r in result.physical_risk_assessment.acute_risks
             if r.risk_type == "cyclone"),
            None
        )
        assert cyclone_risk is not None
        assert cyclone_risk.likelihood >= 3

    def test_maximum_time_horizon(
        self,
        agent: ClimateRiskAgent,
        sample_location: GeoLocation,
    ):
        """Test maximum time horizon (50 years)."""
        input_data = ClimateRiskInput(
            organization_name="Long Term Corp",
            location=sample_location,
            time_horizon_years=50,
        )

        result = agent.run(input_data)
        assert result is not None

    def test_all_sectors(
        self,
        agent: ClimateRiskAgent,
        sample_location: GeoLocation,
    ):
        """Test all sector types can be assessed."""
        for sector in SectorType:
            input_data = ClimateRiskInput(
                organization_name=f"{sector.value} Corp",
                sector=sector,
                location=sample_location,
            )

            result = agent.run(input_data)
            assert result is not None


# =============================================================================
# Public API Tests
# =============================================================================


class TestPublicAPI:
    """Tests for public API methods."""

    def test_get_supported_scenarios(self, agent: ClimateRiskAgent):
        """Test get supported scenarios."""
        scenarios = agent.get_supported_scenarios()

        assert len(scenarios) > 0
        assert all("id" in s for s in scenarios)
        assert all("name" in s for s in scenarios)
        assert all("temperature_2050" in s for s in scenarios)

    def test_get_sector_sensitivity(self, agent: ClimateRiskAgent):
        """Test get sector sensitivity."""
        for sector in SectorType:
            sensitivity = agent.get_sector_sensitivity(sector)
            assert sensitivity is not None
            assert "carbon_pricing" in sensitivity
            assert "regulation" in sensitivity

    def test_get_physical_risk_types(self, agent: ClimateRiskAgent):
        """Test get physical risk types."""
        risk_types = agent.get_physical_risk_types()

        assert len(risk_types) > 0
        assert "flood" in risk_types
        assert "cyclone" in risk_types
        assert "sea_level_rise" in risk_types

    def test_get_transition_risk_types(self, agent: ClimateRiskAgent):
        """Test get transition risk types."""
        risk_types = agent.get_transition_risk_types()

        assert len(risk_types) > 0
        assert "carbon_pricing" in risk_types
        assert "regulation" in risk_types
        assert "technology_disruption" in risk_types


# =============================================================================
# Data Integrity Tests
# =============================================================================


class TestDataIntegrity:
    """Tests for reference data integrity."""

    def test_scenario_parameters_complete(self):
        """Test all scenario parameters are complete."""
        required_keys = [
            "temperature_2050",
            "temperature_2100",
            "sea_level_2100_m",
            "physical_risk_multiplier",
            "transition_risk_multiplier",
            "carbon_price_2030",
            "carbon_price_2050",
            "description",
        ]

        for scenario, params in SCENARIO_PARAMETERS.items():
            for key in required_keys:
                assert key in params, f"Missing {key} in {scenario}"

    def test_sector_sensitivity_complete(self):
        """Test all sector sensitivities are complete."""
        required_keys = [
            "carbon_pricing",
            "regulation",
            "technology_disruption",
            "demand_shift",
            "reputation",
        ]

        for sector, sensitivity in SECTOR_TRANSITION_SENSITIVITY.items():
            for key in required_keys:
                assert key in sensitivity, f"Missing {key} in {sector}"

    def test_sensitivity_values_valid(self):
        """Test sensitivity values are in valid range [0, 1]."""
        for sector, sensitivity in SECTOR_TRANSITION_SENSITIVITY.items():
            for key, value in sensitivity.items():
                assert 0 <= value <= 1, f"Invalid sensitivity {key}={value} for {sector}"

    def test_scenario_temperatures_reasonable(self):
        """Test scenario temperatures are reasonable."""
        for scenario, params in SCENARIO_PARAMETERS.items():
            temp_2050 = params["temperature_2050"]
            temp_2100 = params["temperature_2100"]

            # Temperature should be positive
            assert temp_2050 > 0
            assert temp_2100 > 0

            # 2100 should be >= 2050
            assert temp_2100 >= temp_2050

            # Should be within reasonable bounds
            assert temp_2100 <= 6.0  # Even worst case < 6C


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
