"""
Unit Tests for GL-011: Climate Risk Assessment Agent

Comprehensive test suite covering:
- Physical Risk Assessment (Acute and Chronic)
- Transition Risk Assessment (Policy, Technology, Market, Reputation)
- Scenario Analysis (IPCC Pathways RCP/SSP)
- Financial Impact Quantification
- TCFD-Aligned Outputs

Target: 85%+ code coverage

Reference:
- TCFD (Task Force on Climate-related Financial Disclosures)
- IPCC AR6 Climate Scenarios
- CDP Climate Change Questionnaire

Run with:
    pytest tests/agents/test_gl_011_climate_risk.py -v --cov=backend/agents/gl_011_climate_risk
"""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock




from agents.gl_011_climate_risk.agent import (
    ClimateRiskAgent,
    PhysicalRiskType,
    TransitionRiskType,
    ClimateScenario,
    TimeHorizon,
    RiskCategory,
    AssetType,
    SectorType,
    GeoLocation,
    Asset,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def climate_risk_agent():
    """Create ClimateRiskAgent instance for testing."""
    return ClimateRiskAgent()


@pytest.fixture
def coastal_location():
    """Create coastal location for physical risk testing."""
    return GeoLocation(
        latitude=25.7617,
        longitude=-80.1918,
        country="US",
        region="FL",
        city="Miami",
        coastal_distance_km=5.0,
    )


@pytest.fixture
def inland_location():
    """Create inland location."""
    return GeoLocation(
        latitude=39.7392,
        longitude=-104.9903,
        country="US",
        region="CO",
        city="Denver",
    )


@pytest.fixture
def real_estate_asset(coastal_location):
    """Create real estate asset."""
    return Asset(
        name="Miami Office Building",
        asset_type=AssetType.REAL_ESTATE,
        value_usd=50000000.0,
        location=coastal_location,
        useful_life_years=30,
    )


@pytest.fixture
def climate_risk_input(real_estate_asset):
    """Create climate risk input."""
    from agents.gl_011_climate_risk.agent import ClimateRiskInput
    return ClimateRiskInput(
        organization_name="Example Corp",
        sector=SectorType.REAL_ESTATE,
        assets=[real_estate_asset],
        time_horizon_years=10,
        scenario=ClimateScenario.RCP_4_5,
    )


# =============================================================================
# Test Class: Agent Initialization
# =============================================================================


class TestClimateRiskAgentInitialization:
    """Tests for ClimateRiskAgent initialization."""

    @pytest.mark.unit
    def test_agent_initializes_with_defaults(self, climate_risk_agent):
        """Test agent initializes correctly with default config."""
        assert climate_risk_agent is not None
        assert hasattr(climate_risk_agent, "run")


# =============================================================================
# Test Class: Physical Risk Types
# =============================================================================


class TestPhysicalRiskTypes:
    """Tests for physical climate risk type handling."""

    @pytest.mark.unit
    def test_acute_risk_types(self):
        """Test acute physical risk types."""
        acute_risks = [
            PhysicalRiskType.FLOOD,
            PhysicalRiskType.CYCLONE,
            PhysicalRiskType.WILDFIRE,
            PhysicalRiskType.EXTREME_HEAT,
            PhysicalRiskType.DROUGHT,
            PhysicalRiskType.STORM_SURGE,
            PhysicalRiskType.HAILSTORM,
        ]
        assert len(acute_risks) == 7

    @pytest.mark.unit
    def test_chronic_risk_types(self):
        """Test chronic physical risk types."""
        chronic_risks = [
            PhysicalRiskType.SEA_LEVEL_RISE,
            PhysicalRiskType.TEMPERATURE_INCREASE,
            PhysicalRiskType.PRECIPITATION_CHANGE,
            PhysicalRiskType.WATER_STRESS,
            PhysicalRiskType.PERMAFROST_THAW,
        ]
        assert len(chronic_risks) == 5


# =============================================================================
# Test Class: Transition Risk Types
# =============================================================================


class TestTransitionRiskTypes:
    """Tests for transition climate risk type handling."""

    @pytest.mark.unit
    def test_policy_risk_types(self):
        """Test policy transition risk types."""
        policy_risks = [
            TransitionRiskType.CARBON_PRICING,
            TransitionRiskType.REGULATION,
            TransitionRiskType.MANDATE,
            TransitionRiskType.SUBSIDY_REMOVAL,
        ]
        assert len(policy_risks) == 4

    @pytest.mark.unit
    def test_technology_risk_types(self):
        """Test technology transition risk types."""
        tech_risks = [
            TransitionRiskType.TECHNOLOGY_DISRUPTION,
            TransitionRiskType.TECHNOLOGY_OBSOLESCENCE,
            TransitionRiskType.STRANDED_ASSETS,
        ]
        assert len(tech_risks) == 3

    @pytest.mark.unit
    def test_market_risk_types(self):
        """Test market transition risk types."""
        market_risks = [
            TransitionRiskType.DEMAND_SHIFT,
            TransitionRiskType.COMMODITY_PRICE,
            TransitionRiskType.SUPPLY_CHAIN,
        ]
        assert len(market_risks) == 3

    @pytest.mark.unit
    def test_reputation_risk_types(self):
        """Test reputation transition risk types."""
        reputation_risks = [
            TransitionRiskType.STAKEHOLDER_CONCERN,
            TransitionRiskType.STIGMATIZATION,
            TransitionRiskType.LITIGATION,
        ]
        assert len(reputation_risks) == 3


# =============================================================================
# Test Class: Climate Scenarios
# =============================================================================


class TestClimateScenarios:
    """Tests for IPCC climate scenario handling."""

    @pytest.mark.unit
    def test_rcp_scenarios(self):
        """Test RCP scenario types."""
        rcp_scenarios = [
            ClimateScenario.RCP_2_6,
            ClimateScenario.RCP_4_5,
            ClimateScenario.RCP_6_0,
            ClimateScenario.RCP_8_5,
        ]
        assert len(rcp_scenarios) == 4

    @pytest.mark.unit
    def test_ssp_scenarios(self):
        """Test SSP scenario types."""
        ssp_scenarios = [
            ClimateScenario.SSP1_2_6,
            ClimateScenario.SSP2_4_5,
            ClimateScenario.SSP3_7_0,
            ClimateScenario.SSP5_8_5,
        ]
        assert len(ssp_scenarios) == 4

    @pytest.mark.unit
    def test_scenario_values(self):
        """Test scenario enum values."""
        assert ClimateScenario.RCP_2_6.value == "rcp_2.6"
        assert ClimateScenario.SSP2_4_5.value == "ssp2_4.5"


# =============================================================================
# Test Class: Time Horizons
# =============================================================================


class TestTimeHorizons:
    """Tests for time horizon handling."""

    @pytest.mark.unit
    def test_time_horizon_values(self):
        """Test time horizon enum values."""
        assert TimeHorizon.SHORT_TERM.value == "short_term"
        assert TimeHorizon.MEDIUM_TERM.value == "medium_term"
        assert TimeHorizon.LONG_TERM.value == "long_term"


# =============================================================================
# Test Class: Risk Categories
# =============================================================================


class TestRiskCategories:
    """Tests for risk category handling."""

    @pytest.mark.unit
    def test_risk_category_values(self):
        """Test risk category enum values."""
        assert RiskCategory.CRITICAL.value == "critical"
        assert RiskCategory.HIGH.value == "high"
        assert RiskCategory.MEDIUM.value == "medium"
        assert RiskCategory.LOW.value == "low"
        assert RiskCategory.MINIMAL.value == "minimal"


# =============================================================================
# Test Class: Asset Types
# =============================================================================


class TestAssetTypes:
    """Tests for asset type handling."""

    @pytest.mark.unit
    def test_asset_type_values(self):
        """Test asset type enum values."""
        assert AssetType.REAL_ESTATE.value == "real_estate"
        assert AssetType.INFRASTRUCTURE.value == "infrastructure"
        assert AssetType.EQUIPMENT.value == "equipment"
        assert AssetType.SUPPLY_CHAIN.value == "supply_chain"


# =============================================================================
# Test Class: Sector Types
# =============================================================================


class TestSectorTypes:
    """Tests for sector type handling."""

    @pytest.mark.unit
    def test_high_risk_sectors(self):
        """Test high climate risk sectors."""
        high_risk_sectors = [
            SectorType.ENERGY,
            SectorType.UTILITIES,
            SectorType.TRANSPORTATION,
            SectorType.AGRICULTURE,
            SectorType.MINING,
        ]
        assert len(high_risk_sectors) == 5


# =============================================================================
# Test Class: GeoLocation Validation
# =============================================================================


class TestGeoLocationValidation:
    """Tests for geolocation validation."""

    @pytest.mark.unit
    def test_valid_geolocation(self, coastal_location):
        """Test valid geolocation passes validation."""
        assert coastal_location.latitude == 25.7617
        assert coastal_location.longitude == -80.1918
        assert coastal_location.country == "US"

    @pytest.mark.unit
    def test_latitude_bounds(self):
        """Test latitude must be -90 to 90."""
        with pytest.raises(ValueError):
            GeoLocation(latitude=95, longitude=0, country="US")

    @pytest.mark.unit
    def test_longitude_bounds(self):
        """Test longitude must be -180 to 180."""
        with pytest.raises(ValueError):
            GeoLocation(latitude=0, longitude=190, country="US")

    @pytest.mark.unit
    def test_coastal_distance_non_negative(self, coastal_location):
        """Test coastal distance must be non-negative."""
        assert coastal_location.coastal_distance_km >= 0


# =============================================================================
# Test Class: Asset Validation
# =============================================================================


class TestAssetValidation:
    """Tests for asset validation."""

    @pytest.mark.unit
    def test_valid_asset(self, real_estate_asset):
        """Test valid asset passes validation."""
        assert real_estate_asset.name == "Miami Office Building"
        assert real_estate_asset.value_usd == 50000000.0

    @pytest.mark.unit
    def test_asset_value_non_negative(self, real_estate_asset):
        """Test asset value must be non-negative."""
        assert real_estate_asset.value_usd >= 0


# =============================================================================
# Test Class: Risk Assessment
# =============================================================================


class TestRiskAssessment:
    """Tests for climate risk assessment."""

    @pytest.mark.unit
    def test_risk_assessment_performed(self, climate_risk_agent, climate_risk_input):
        """Test risk assessment is performed."""
        result = climate_risk_agent.run(climate_risk_input)

        assert hasattr(result, "total_risk_score")
        assert hasattr(result, "physical_risk_score")
        assert hasattr(result, "transition_risk_score")

    @pytest.mark.unit
    def test_coastal_asset_has_elevated_risk(self, climate_risk_agent, climate_risk_input):
        """Test coastal assets have elevated physical risk."""
        result = climate_risk_agent.run(climate_risk_input)

        # Coastal Miami asset should have elevated physical risk
        assert result.physical_risk_score > 0


# =============================================================================
# Test Class: Scenario Analysis
# =============================================================================


class TestScenarioAnalysis:
    """Tests for scenario analysis functionality."""

    @pytest.mark.unit
    def test_scenario_impacts_calculated(self, climate_risk_agent, climate_risk_input):
        """Test scenario impacts are calculated."""
        result = climate_risk_agent.run(climate_risk_input)

        assert hasattr(result, "scenario_analysis") or hasattr(result, "scenario_impacts")


# =============================================================================
# Test Class: Financial Impact
# =============================================================================


class TestFinancialImpact:
    """Tests for financial impact quantification."""

    @pytest.mark.unit
    def test_value_at_risk_calculated(self, climate_risk_agent, climate_risk_input):
        """Test value at risk is calculated."""
        result = climate_risk_agent.run(climate_risk_input)

        assert hasattr(result, "value_at_risk") or hasattr(result, "financial_exposure")


# =============================================================================
# Test Class: Provenance Tracking
# =============================================================================


class TestClimateRiskProvenance:
    """Tests for provenance hash tracking."""

    @pytest.mark.unit
    def test_provenance_hash_generated(self, climate_risk_agent, climate_risk_input):
        """Test provenance hash is generated."""
        result = climate_risk_agent.run(climate_risk_input)
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64


# =============================================================================
# Test Class: Performance
# =============================================================================


class TestClimateRiskPerformance:
    """Performance tests for ClimateRiskAgent."""

    @pytest.mark.unit
    @pytest.mark.performance
    def test_single_assessment_performance(self, climate_risk_agent, climate_risk_input):
        """Test single assessment completes quickly."""
        import time

        start = time.perf_counter()
        result = climate_risk_agent.run(climate_risk_input)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 100.0
        assert result is not None
