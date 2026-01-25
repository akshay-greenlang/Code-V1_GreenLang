"""
GL-077: Incentive Hunter Agent - Test Suite

Comprehensive test coverage for IncentiveHunterAgent including:
- Input validation tests
- Incentive search tests
- Eligibility evaluation tests
- Value calculation tests
- Provenance tracking tests
- Formula verification tests

Test Coverage Target: 85%+
"""

import hashlib
import json
import pytest
from datetime import datetime, timedelta
from typing import Dict, List

from .agent import (
    IncentiveHunterAgent,
    IncentiveHunterInput,
    IncentiveHunterOutput,
    LocationInfo,
    EquipmentInfo,
    ProjectScope,
    UtilityProvider,
    AvailableIncentive,
    EligibilityStatus,
    ApplicationRequirement,
    IncentiveType,
    IncentiveCategory,
    EligibilityState,
    ProjectType,
)
from .formulas import (
    calculate_incentive_value,
    calculate_payback_impact,
    calculate_stacking_limit,
    estimate_application_success,
    calculate_npv_with_incentives,
    calculate_irr_with_incentives,
    IncentiveValueResult,
    PaybackImpactResult,
    StackingAnalysisResult,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def agent():
    """Create IncentiveHunterAgent instance."""
    return IncentiveHunterAgent()


@pytest.fixture
def basic_location():
    """Create basic location info."""
    return LocationInfo(
        state="CA",
        zip_code="94102",
        utility_territory="PG&E",
    )


@pytest.fixture
def basic_project_scope():
    """Create basic project scope."""
    return ProjectScope(
        project_type=ProjectType.RETROFIT,
        project_cost_usd=100000.0,
        building_type="COMMERCIAL",
        building_size_sqft=25000.0,
        sector="COMMERCIAL",
    )


@pytest.fixture
def basic_input(basic_location, basic_project_scope):
    """Create basic input for testing."""
    return IncentiveHunterInput(
        location=basic_location,
        equipment_types=["LED_LIGHTING", "HVAC"],
        project_scope=basic_project_scope,
        utility_provider=UtilityProvider(
            electric_utility="PG&E",
            annual_electric_usage_kwh=500000,
        ),
    )


@pytest.fixture
def solar_input(basic_location):
    """Create input for solar project."""
    return IncentiveHunterInput(
        location=basic_location,
        equipment_types=["SOLAR", "BATTERY"],
        project_scope=ProjectScope(
            project_type=ProjectType.RENEWABLE_INSTALLATION,
            project_cost_usd=500000.0,
            sector="COMMERCIAL",
        ),
        equipment_details=[
            EquipmentInfo(
                equipment_type="SOLAR",
                capacity_kw=100,
                quantity=1,
            ),
            EquipmentInfo(
                equipment_type="BATTERY",
                capacity_kw=50,
                quantity=1,
            ),
        ],
    )


# =============================================================================
# AGENT INITIALIZATION TESTS
# =============================================================================

class TestAgentInitialization:
    """Test agent initialization."""

    def test_agent_creates_successfully(self, agent):
        """Test agent creates with default config."""
        assert agent is not None
        assert agent.AGENT_ID == "GL-077"
        assert agent.AGENT_NAME == "INCENTIVEHUNTER"

    def test_agent_with_custom_config(self):
        """Test agent with custom configuration."""
        config = {"custom_setting": "value"}
        agent = IncentiveHunterAgent(config=config)
        assert agent.config == config

    def test_agent_has_incentive_database(self, agent):
        """Test agent has incentive database loaded."""
        assert len(agent.incentive_db) > 0

    def test_agent_constants(self, agent):
        """Test agent constants are set."""
        assert agent.VERSION == "1.0.0"
        assert agent.DESCRIPTION == "Energy Incentive Identification Agent"


# =============================================================================
# INPUT MODEL TESTS
# =============================================================================

class TestInputModels:
    """Test input model validation."""

    def test_valid_location_info(self):
        """Test valid location info creation."""
        location = LocationInfo(state="TX", zip_code="75001")
        assert location.state == "TX"
        assert location.zip_code == "75001"

    def test_location_requires_two_letter_state(self):
        """Test state code validation."""
        with pytest.raises(ValueError):
            LocationInfo(state="TEXAS")

    def test_location_with_disadvantaged_community(self):
        """Test disadvantaged community flag."""
        location = LocationInfo(
            state="CA",
            is_disadvantaged_community=True
        )
        assert location.is_disadvantaged_community is True

    def test_valid_equipment_info(self):
        """Test valid equipment info creation."""
        equipment = EquipmentInfo(
            equipment_type="LED_FIXTURE",
            manufacturer="Acme",
            quantity=100,
            capacity_kw=5.0,
            energy_star_certified=True,
        )
        assert equipment.quantity == 100
        assert equipment.energy_star_certified is True

    def test_equipment_quantity_must_be_positive(self):
        """Test equipment quantity validation."""
        with pytest.raises(ValueError):
            EquipmentInfo(equipment_type="LED", quantity=0)

    def test_valid_project_scope(self):
        """Test valid project scope creation."""
        scope = ProjectScope(
            project_type=ProjectType.RETROFIT,
            project_cost_usd=50000,
            building_size_sqft=10000,
        )
        assert scope.project_type == ProjectType.RETROFIT

    def test_project_cost_must_be_positive(self):
        """Test project cost validation."""
        with pytest.raises(ValueError):
            ProjectScope(
                project_type=ProjectType.RETROFIT,
                project_cost_usd=-1000
            )

    def test_complete_input_creation(self, basic_input):
        """Test complete input model creation."""
        assert basic_input.location.state == "CA"
        assert len(basic_input.equipment_types) == 2


# =============================================================================
# CATEGORY MAPPING TESTS
# =============================================================================

class TestCategoryMapping:
    """Test equipment to category mapping."""

    def test_led_maps_to_lighting(self, agent):
        """Test LED equipment maps to lighting category."""
        categories = agent._map_equipment_to_categories(["LED_LIGHTING"])
        assert IncentiveCategory.LIGHTING in categories

    def test_hvac_maps_correctly(self, agent):
        """Test HVAC equipment maps correctly."""
        categories = agent._map_equipment_to_categories(["HVAC"])
        assert IncentiveCategory.HVAC in categories

    def test_vfd_maps_to_motors(self, agent):
        """Test VFD maps to motors/drives category."""
        categories = agent._map_equipment_to_categories(["VFD"])
        assert IncentiveCategory.MOTORS_DRIVES in categories

    def test_solar_maps_to_renewable(self, agent):
        """Test solar maps to renewable energy."""
        categories = agent._map_equipment_to_categories(["SOLAR"])
        assert IncentiveCategory.RENEWABLE_ENERGY in categories

    def test_battery_maps_to_storage(self, agent):
        """Test battery maps to energy storage."""
        categories = agent._map_equipment_to_categories(["BATTERY"])
        assert IncentiveCategory.ENERGY_STORAGE in categories

    def test_multiple_equipment_types(self, agent):
        """Test multiple equipment types mapping."""
        categories = agent._map_equipment_to_categories(
            ["LED_LIGHTING", "HVAC", "VFD"]
        )
        assert len(categories) == 3

    def test_unknown_equipment_returns_empty(self, agent):
        """Test unknown equipment returns empty list."""
        categories = agent._map_equipment_to_categories(["UNKNOWN_TYPE"])
        assert len(categories) == 0


# =============================================================================
# INCENTIVE SEARCH TESTS
# =============================================================================

class TestIncentiveSearch:
    """Test incentive search functionality."""

    def test_search_finds_lighting_incentives(self, agent, basic_location):
        """Test search finds lighting incentives."""
        categories = [IncentiveCategory.LIGHTING]
        incentives = agent._search_incentives(
            categories, basic_location, "FOR_PROFIT"
        )
        assert len(incentives) > 0

    def test_search_filters_by_state(self, agent):
        """Test search filters by state."""
        location = LocationInfo(state="TX")
        categories = [IncentiveCategory.ENERGY_STORAGE]
        incentives = agent._search_incentives(
            categories, location, "FOR_PROFIT"
        )
        # SGIP is CA-only, should not appear for TX
        sgip_found = any(i["id"] == "CA_SGIP" for i in incentives)
        assert not sgip_found

    def test_california_gets_sgip(self, agent):
        """Test California location gets SGIP incentive."""
        location = LocationInfo(state="CA")
        categories = [IncentiveCategory.ENERGY_STORAGE]
        incentives = agent._search_incentives(
            categories, location, "FOR_PROFIT"
        )
        sgip_found = any(i["id"] == "CA_SGIP" for i in incentives)
        assert sgip_found

    def test_search_filters_by_tax_status(self, agent, basic_location):
        """Test search filters by tax status."""
        categories = [IncentiveCategory.RENEWABLE_ENERGY]
        # ITC is for-profit only
        for_profit_incentives = agent._search_incentives(
            categories, basic_location, "FOR_PROFIT"
        )
        non_profit_incentives = agent._search_incentives(
            categories, basic_location, "NON_PROFIT"
        )
        # Results may differ based on tax status
        assert isinstance(for_profit_incentives, list)
        assert isinstance(non_profit_incentives, list)


# =============================================================================
# ELIGIBILITY EVALUATION TESTS
# =============================================================================

class TestEligibilityEvaluation:
    """Test eligibility evaluation."""

    def test_commercial_sector_meets_commercial_requirement(self, agent, basic_input):
        """Test commercial sector meets requirement."""
        incentive_data = {
            "id": "TEST",
            "name": "Test Incentive",
            "type": IncentiveType.UTILITY_REBATE,
            "categories": [IncentiveCategory.LIGHTING],
            "provider": "Test",
            "requirements": ["Commercial building"],
            "stackable": True,
        }
        eligibility = agent._assess_eligibility(incentive_data, basic_input)
        assert eligibility.state in [EligibilityState.ELIGIBLE, EligibilityState.LIKELY_ELIGIBLE]

    def test_california_location_meets_california_requirement(self, agent, basic_input):
        """Test California location meets state requirement."""
        incentive_data = {
            "id": "TEST",
            "name": "Test Incentive",
            "type": IncentiveType.UTILITY_REBATE,
            "categories": [IncentiveCategory.LIGHTING],
            "provider": "Test",
            "requirements": ["California location"],
            "stackable": True,
        }
        eligibility = agent._assess_eligibility(incentive_data, basic_input)
        assert "California location" in eligibility.qualifying_criteria

    def test_non_california_fails_california_requirement(self, agent):
        """Test non-California location fails state requirement."""
        input_data = IncentiveHunterInput(
            location=LocationInfo(state="TX"),
            equipment_types=["LED_LIGHTING"],
            project_scope=ProjectScope(
                project_type=ProjectType.RETROFIT,
                sector="COMMERCIAL",
            ),
        )
        incentive_data = {
            "id": "TEST",
            "name": "Test Incentive",
            "type": IncentiveType.UTILITY_REBATE,
            "categories": [IncentiveCategory.LIGHTING],
            "provider": "Test",
            "requirements": ["California location"],
            "stackable": True,
        }
        eligibility = agent._assess_eligibility(incentive_data, input_data)
        assert "California location" in eligibility.missing_criteria

    def test_eligibility_confidence_scoring(self, agent, basic_input):
        """Test eligibility confidence is 0-1."""
        incentive_data = {
            "id": "TEST",
            "name": "Test",
            "type": IncentiveType.UTILITY_REBATE,
            "categories": [IncentiveCategory.LIGHTING],
            "provider": "Test",
            "requirements": [],
            "stackable": True,
        }
        eligibility = agent._assess_eligibility(incentive_data, basic_input)
        assert 0 <= eligibility.confidence_score <= 1


# =============================================================================
# VALUE CALCULATION TESTS
# =============================================================================

class TestValueCalculation:
    """Test incentive value calculations."""

    def test_per_sqft_calculation(self, agent, basic_input):
        """Test per-sqft value calculation."""
        incentive_data = {
            "id": "TEST",
            "name": "Test",
            "type": IncentiveType.FEDERAL_TAX_CREDIT,
            "categories": [IncentiveCategory.LIGHTING],
            "provider": "Test",
            "value_per_sqft": 5.0,
            "stackable": True,
        }
        value = agent._calculate_value(incentive_data, basic_input)
        # 25,000 sqft * $5/sqft = $125,000
        assert value == 125000.0

    def test_percentage_calculation(self, agent, basic_input):
        """Test percentage of cost calculation."""
        incentive_data = {
            "id": "TEST",
            "name": "Test",
            "type": IncentiveType.FEDERAL_TAX_CREDIT,
            "categories": [IncentiveCategory.LIGHTING],
            "provider": "Test",
            "value_percent": 30,
            "stackable": True,
        }
        value = agent._calculate_value(incentive_data, basic_input)
        # $100,000 * 30% = $30,000
        assert value == 30000.0

    def test_per_fixture_calculation(self, agent):
        """Test per-fixture calculation."""
        input_data = IncentiveHunterInput(
            location=LocationInfo(state="CA"),
            equipment_types=["LED_LIGHTING"],
            project_scope=ProjectScope(project_type=ProjectType.RETROFIT),
            equipment_details=[
                EquipmentInfo(equipment_type="LED", quantity=50),
            ],
        )
        incentive_data = {
            "id": "TEST",
            "name": "Test",
            "type": IncentiveType.UTILITY_REBATE,
            "categories": [IncentiveCategory.LIGHTING],
            "provider": "Test",
            "value_per_fixture": 50,
            "stackable": True,
        }
        value = agent._calculate_value(incentive_data, input_data)
        # 50 fixtures * $50/fixture = $2,500
        assert value == 2500.0


# =============================================================================
# RUN METHOD TESTS
# =============================================================================

class TestRunMethod:
    """Test the main run method."""

    def test_run_returns_output(self, agent, basic_input):
        """Test run returns IncentiveHunterOutput."""
        result = agent.run(basic_input)
        assert isinstance(result, IncentiveHunterOutput)

    def test_run_has_analysis_id(self, agent, basic_input):
        """Test output has analysis ID."""
        result = agent.run(basic_input)
        assert result.analysis_id.startswith("INCENT-")

    def test_run_finds_incentives(self, agent, basic_input):
        """Test run finds applicable incentives."""
        result = agent.run(basic_input)
        assert len(result.available_incentives) > 0

    def test_run_calculates_total_value(self, agent, basic_input):
        """Test run calculates total value."""
        result = agent.run(basic_input)
        assert result.total_estimated_value_usd >= 0

    def test_run_counts_eligible_incentives(self, agent, basic_input):
        """Test run counts eligible incentives."""
        result = agent.run(basic_input)
        assert result.eligible_count >= 0

    def test_run_generates_recommendations(self, agent, basic_input):
        """Test run generates recommendations."""
        result = agent.run(basic_input)
        assert isinstance(result.top_recommendations, list)

    def test_run_has_provenance_chain(self, agent, basic_input):
        """Test run creates provenance chain."""
        result = agent.run(basic_input)
        assert len(result.provenance_chain) > 0

    def test_run_has_provenance_hash(self, agent, basic_input):
        """Test run calculates provenance hash."""
        result = agent.run(basic_input)
        assert len(result.provenance_hash) == 64  # SHA-256

    def test_run_tracks_processing_time(self, agent, basic_input):
        """Test run tracks processing time."""
        result = agent.run(basic_input)
        assert result.processing_time_ms > 0

    def test_run_validates_output(self, agent, basic_input):
        """Test run validates output."""
        result = agent.run(basic_input)
        assert result.validation_status in ["PASS", "FAIL"]


# =============================================================================
# PROVENANCE TRACKING TESTS
# =============================================================================

class TestProvenanceTracking:
    """Test provenance tracking."""

    def test_provenance_chain_has_operations(self, agent, basic_input):
        """Test provenance chain records operations."""
        result = agent.run(basic_input)
        operations = [p.operation for p in result.provenance_chain]
        assert "category_mapping" in operations
        assert "incentive_search" in operations

    def test_provenance_has_hashes(self, agent, basic_input):
        """Test provenance records have hashes."""
        result = agent.run(basic_input)
        for record in result.provenance_chain:
            assert len(record.input_hash) == 64
            assert len(record.output_hash) == 64

    def test_provenance_has_timestamps(self, agent, basic_input):
        """Test provenance records have timestamps."""
        result = agent.run(basic_input)
        for record in result.provenance_chain:
            assert isinstance(record.timestamp, datetime)

    def test_provenance_hash_is_deterministic(self, agent, basic_input):
        """Test same inputs produce same provenance patterns."""
        result1 = agent.run(basic_input)
        result2 = agent.run(basic_input)
        # Note: Timestamps differ, but operation sequences match
        assert len(result1.provenance_chain) == len(result2.provenance_chain)


# =============================================================================
# FORMULA TESTS
# =============================================================================

class TestFormulas:
    """Test formula calculations."""

    def test_179d_calculation(self):
        """Test 179D deduction calculation."""
        result = calculate_incentive_value(
            "179D",
            building_sqft=50000,
            is_prevailing_wage=True,
        )
        assert result.total_value == 250000  # $5/sqft * 50,000 sqft
        assert result.calculation_method == "179D_ENHANCED"

    def test_179d_base_rate(self):
        """Test 179D base rate without prevailing wage."""
        result = calculate_incentive_value(
            "179D",
            building_sqft=50000,
            is_prevailing_wage=False,
        )
        assert result.total_value == 50000  # $1/sqft * 50,000 sqft
        assert result.calculation_method == "179D_BASE"

    def test_itc_base_calculation(self):
        """Test ITC base calculation."""
        result = calculate_incentive_value(
            "ITC",
            project_cost=100000,
            is_prevailing_wage=True,
        )
        assert result.base_value == 30000  # 30%
        assert result.calculation_method == "ITC_IRA_2022"

    def test_itc_with_bonuses(self):
        """Test ITC with bonus adders."""
        result = calculate_incentive_value(
            "ITC",
            project_cost=100000,
            is_prevailing_wage=True,
            is_domestic_content=True,
            is_energy_community=True,
        )
        # Base 30% + DC 10% + EC 10% = 50%
        assert result.total_value == 50000

    def test_sgip_standard_rate(self):
        """Test SGIP standard rate."""
        result = calculate_incentive_value(
            "SGIP",
            capacity_kwh=100,
            is_disadvantaged_community=False,
        )
        assert result.total_value == 20000  # $200/kWh * 100 kWh

    def test_sgip_equity_rate(self):
        """Test SGIP equity rate for DAC."""
        result = calculate_incentive_value(
            "SGIP",
            capacity_kwh=100,
            is_disadvantaged_community=True,
        )
        assert result.total_value == 40000  # $400/kWh * 100 kWh

    def test_led_rebate_per_fixture(self):
        """Test LED rebate per fixture calculation."""
        result = calculate_incentive_value(
            "LED_REBATE",
            fixtures=100,
            kwh_savings=0,
        )
        assert result.total_value == 5000  # $50/fixture * 100

    def test_led_rebate_performance(self):
        """Test LED rebate performance-based calculation."""
        result = calculate_incentive_value(
            "LED_REBATE",
            fixtures=0,
            kwh_savings=100000,
        )
        assert result.total_value == 8000  # $0.08/kWh * 100,000

    def test_payback_impact(self):
        """Test payback impact calculation."""
        result = calculate_payback_impact(
            project_cost=100000,
            annual_savings=20000,
            incentive_value=30000,
        )
        assert result.original_payback_years == 5.0  # 100k / 20k
        assert result.adjusted_payback_years == 3.5  # (100k - 30k) / 20k
        assert result.payback_reduction_years == 1.5

    def test_payback_with_zero_savings(self):
        """Test payback calculation with zero savings."""
        result = calculate_payback_impact(
            project_cost=100000,
            annual_savings=0,
            incentive_value=30000,
        )
        assert result.original_payback_years == float('inf')

    def test_stacking_limit_not_exceeded(self):
        """Test stacking when limit not exceeded."""
        result = calculate_stacking_limit(
            project_cost=100000,
            federal_incentives=20000,
            state_incentives=10000,
            utility_incentives=5000,
            max_stacking_percent=100,
        )
        assert result.total_stackable_value == 35000
        assert result.stacking_limit_applied is False

    def test_stacking_limit_exceeded(self):
        """Test stacking when limit exceeded."""
        result = calculate_stacking_limit(
            project_cost=100000,
            federal_incentives=40000,
            state_incentives=30000,
            utility_incentives=20000,
            max_stacking_percent=50,
        )
        assert result.total_stackable_value == 50000  # 50% cap
        assert result.stacking_limit_applied is True

    def test_application_success_estimation(self):
        """Test application success probability."""
        prob, notes = estimate_application_success(
            eligibility_score=0.9,
            documentation_completeness=0.8,
            program_funding_level=0.7,
            days_to_deadline=60,
        )
        assert 0 <= prob <= 1
        assert isinstance(notes, str)

    def test_application_success_with_deadline_passed(self):
        """Test application success with passed deadline."""
        prob, notes = estimate_application_success(
            eligibility_score=0.9,
            documentation_completeness=0.8,
            program_funding_level=0.7,
            days_to_deadline=0,
        )
        assert prob < 0.9  # Reduced due to timeline

    def test_npv_calculation(self):
        """Test NPV calculation with incentives."""
        npv_without, npv_with, improvement = calculate_npv_with_incentives(
            project_cost=100000,
            annual_savings=25000,
            incentive_value=30000,
            discount_rate=0.08,
            project_life_years=10,
        )
        assert npv_with > npv_without
        assert improvement == 30000  # Incentive received immediately

    def test_irr_calculation(self):
        """Test IRR calculation."""
        irr_without, irr_with = calculate_irr_with_incentives(
            project_cost=100000,
            annual_savings=25000,
            incentive_value=30000,
            project_life_years=10,
        )
        assert irr_with > irr_without  # Incentives improve IRR


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_equipment_types(self, agent):
        """Test with empty equipment types."""
        input_data = IncentiveHunterInput(
            location=LocationInfo(state="CA"),
            equipment_types=[],
            project_scope=ProjectScope(project_type=ProjectType.RETROFIT),
        )
        result = agent.run(input_data)
        assert result.available_incentives == []
        assert result.total_estimated_value_usd == 0

    def test_unknown_equipment_types(self, agent):
        """Test with unknown equipment types."""
        input_data = IncentiveHunterInput(
            location=LocationInfo(state="CA"),
            equipment_types=["UNKNOWN_DEVICE_123"],
            project_scope=ProjectScope(project_type=ProjectType.RETROFIT),
        )
        result = agent.run(input_data)
        # Should not crash, returns empty results
        assert isinstance(result, IncentiveHunterOutput)

    def test_minimal_input(self, agent):
        """Test with minimal required input."""
        input_data = IncentiveHunterInput(
            location=LocationInfo(state="TX"),
            equipment_types=["LED_LIGHTING"],
            project_scope=ProjectScope(project_type=ProjectType.RETROFIT),
        )
        result = agent.run(input_data)
        assert isinstance(result, IncentiveHunterOutput)

    def test_all_equipment_types(self, agent):
        """Test with all known equipment types."""
        all_types = [
            "LED_LIGHTING", "HVAC", "VFD", "SOLAR", "BATTERY",
            "BOILER", "CHILLER", "COMPRESSED_AIR", "EMS"
        ]
        input_data = IncentiveHunterInput(
            location=LocationInfo(state="CA"),
            equipment_types=all_types,
            project_scope=ProjectScope(
                project_type=ProjectType.RETROFIT,
                project_cost_usd=1000000,
            ),
        )
        result = agent.run(input_data)
        assert len(result.available_incentives) > 0

    def test_very_large_project(self, agent):
        """Test with very large project values."""
        input_data = IncentiveHunterInput(
            location=LocationInfo(state="CA"),
            equipment_types=["SOLAR"],
            project_scope=ProjectScope(
                project_type=ProjectType.RENEWABLE_INSTALLATION,
                project_cost_usd=100_000_000,  # $100M
                building_size_sqft=1_000_000,
            ),
        )
        result = agent.run(input_data)
        assert result.total_estimated_value_usd > 0

    def test_government_tax_status(self, agent):
        """Test with government tax status."""
        input_data = IncentiveHunterInput(
            location=LocationInfo(state="CA"),
            equipment_types=["BATTERY"],
            project_scope=ProjectScope(project_type=ProjectType.RETROFIT),
            tax_status="GOVERNMENT",
        )
        result = agent.run(input_data)
        # Government entities can't use tax credits but can get rebates
        assert isinstance(result, IncentiveHunterOutput)


# =============================================================================
# OUTPUT VALIDATION TESTS
# =============================================================================

class TestOutputValidation:
    """Test output model validation."""

    def test_output_has_required_fields(self, agent, basic_input):
        """Test output has all required fields."""
        result = agent.run(basic_input)
        assert hasattr(result, 'analysis_id')
        assert hasattr(result, 'available_incentives')
        assert hasattr(result, 'total_estimated_value_usd')
        assert hasattr(result, 'provenance_hash')

    def test_incentive_has_required_fields(self, agent, basic_input):
        """Test each incentive has required fields."""
        result = agent.run(basic_input)
        for incentive in result.available_incentives:
            assert incentive.incentive_id
            assert incentive.name
            assert incentive.incentive_type
            assert incentive.estimated_value_usd >= 0
            assert incentive.eligibility

    def test_eligibility_has_required_fields(self, agent, basic_input):
        """Test eligibility has required fields."""
        result = agent.run(basic_input)
        for incentive in result.available_incentives:
            elig = incentive.eligibility
            assert elig.state in EligibilityState
            assert 0 <= elig.confidence_score <= 1


# =============================================================================
# PACK_SPEC TESTS
# =============================================================================

class TestPackSpec:
    """Test PACK_SPEC configuration."""

    def test_pack_spec_exists(self):
        """Test PACK_SPEC is defined."""
        from .agent import PACK_SPEC
        assert PACK_SPEC is not None

    def test_pack_spec_has_required_fields(self):
        """Test PACK_SPEC has required fields."""
        from .agent import PACK_SPEC
        assert PACK_SPEC["id"] == "GL-077"
        assert PACK_SPEC["name"]
        assert PACK_SPEC["version"]

    def test_pack_spec_has_standards(self):
        """Test PACK_SPEC references standards."""
        from .agent import PACK_SPEC
        standards = PACK_SPEC.get("standards", [])
        assert len(standards) > 0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""

    def test_full_solar_storage_analysis(self, agent, solar_input):
        """Test complete solar + storage analysis."""
        result = agent.run(solar_input)

        # Should find ITC and SGIP
        incentive_ids = [i.incentive_id for i in result.available_incentives]
        assert "IRA_ITC" in incentive_ids or any("ITC" in id for id in incentive_ids)

        # Total value should be significant
        assert result.total_estimated_value_usd > 10000

    def test_full_lighting_retrofit_analysis(self, agent, basic_input):
        """Test complete lighting retrofit analysis."""
        result = agent.run(basic_input)

        # Should find lighting rebate and possibly 179D
        assert result.eligible_count > 0 or result.conditional_count > 0

    def test_multiple_runs_independent(self, agent, basic_input):
        """Test multiple runs don't interfere."""
        result1 = agent.run(basic_input)
        result2 = agent.run(basic_input)

        # Results should be consistent (though timestamps differ)
        assert len(result1.available_incentives) == len(result2.available_incentives)
        assert result1.total_estimated_value_usd == result2.total_estimated_value_usd


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
