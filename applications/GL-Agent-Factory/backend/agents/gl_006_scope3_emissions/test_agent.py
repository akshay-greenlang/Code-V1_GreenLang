"""
Unit Tests for GL-006: Scope 3 Emissions Agent

Comprehensive test coverage for the Scope 3 Supply Chain Emissions Agent including:
- All 15 GHG Protocol Scope 3 categories
- Spend-based, activity-based, and supplier-specific calculation methods
- Transport emissions (Categories 4, 9)
- Business travel emissions (Category 6)
- Waste emissions (Category 5)
- Data quality assessment
- Golden tests with zero-hallucination calculations
- Determinism verification tests
- Edge cases and boundary conditions

Test coverage target: 85%+
Total tests: 75+ golden tests covering all Scope 3 calculation scenarios

Formula Documentation:
----------------------
All emission calculations follow GHG Protocol Corporate Value Chain Standard:

Spend-Based Method:
    emissions (kgCO2e) = spend (USD) * EEIO_factor (kgCO2e/USD)

Transport Method (Categories 4, 9):
    emissions (kgCO2e) = distance (km) * weight (tonnes) * transport_factor (kgCO2e/tonne-km)

Travel Method (Category 6):
    emissions (kgCO2e) = distance (km) * travel_factor (kgCO2e/passenger-km)

Waste Method (Category 5):
    emissions (kgCO2e) = quantity (kg) * waste_factor (kgCO2e/kg)

Emission Factors (from agent.py):
---------------------------------
Spend Factors (EPA EEIO 2024, kgCO2e/USD):
- Steel: 0.85
- Aluminum: 1.20
- Plastics: 0.75
- Chemicals: 0.65
- Paper: 0.45
- Electronics: 0.55
- Machinery: 0.40
- Textiles: 0.50
- Food: 0.60
- Construction: 0.35
- Services: 0.15
- IT Services: 0.18
- Professional Services: 0.12
- Default: 0.40

Transport Factors (GLEC Framework, kgCO2e/tonne-km):
- Road Truck: 0.089
- Road Van: 0.195
- Rail Freight: 0.028
- Sea Container: 0.016
- Sea Bulk: 0.008
- Air Freight: 0.602
- Air Belly: 0.301
- Barge: 0.031
- Pipeline: 0.025

Travel Factors (DEFRA 2024, kgCO2e/passenger-km):
- Air Short Haul (<1500km): 0.255
- Air Medium Haul (1500-4000km): 0.156
- Air Long Haul (>4000km): 0.195
- Rail Average: 0.041
- Rail High Speed: 0.006
- Car Average: 0.171
- Car Electric: 0.053
- Bus: 0.089
- Taxi: 0.203

Waste Factors (DEFRA 2024, kgCO2e/kg):
- Landfill Mixed: 0.586
- Landfill Organic: 0.700
- Incineration: 0.021
- Recycling Paper: -0.900 (negative = avoided)
- Recycling Plastic: -1.400
- Recycling Metal: -1.800
- Recycling Glass: -0.300
- Composting: 0.010
"""

import hashlib
import json
import pytest
from datetime import datetime
from typing import Dict, List, Any, Optional

from .agent import (
    # Main Agent and I/O
    Scope3EmissionsAgent,
    Scope3Input,
    Scope3Output,
    # Data Models
    SpendData,
    ActivityData,
    TransportData,
    TravelData,
    SpendEmissionFactor,
    # Enumerations
    Scope3Category,
    CalculationMethod,
    DataQualityScore,
)


# =============================================================================
# Test Constants - Expected Emission Factors
# =============================================================================

# Spend-based factors (EPA EEIO 2024, kgCO2e/USD)
EF_STEEL = 0.85
EF_ALUMINUM = 1.20
EF_PLASTICS = 0.75
EF_CHEMICALS = 0.65
EF_PAPER = 0.45
EF_ELECTRONICS = 0.55
EF_MACHINERY = 0.40
EF_TEXTILES = 0.50
EF_FOOD = 0.60
EF_CONSTRUCTION = 0.35
EF_SERVICES = 0.15
EF_IT_SERVICES = 0.18
EF_PROFESSIONAL_SERVICES = 0.12
EF_DEFAULT = 0.40

# Transport factors (GLEC Framework, kgCO2e/tonne-km)
TF_ROAD_TRUCK = 0.089
TF_ROAD_VAN = 0.195
TF_RAIL_FREIGHT = 0.028
TF_SEA_CONTAINER = 0.016
TF_SEA_BULK = 0.008
TF_AIR_FREIGHT = 0.602
TF_AIR_BELLY = 0.301
TF_BARGE = 0.031
TF_PIPELINE = 0.025

# Travel factors (DEFRA 2024, kgCO2e/passenger-km)
TV_AIR_SHORT_HAUL = 0.255
TV_AIR_MEDIUM_HAUL = 0.156
TV_AIR_LONG_HAUL = 0.195
TV_RAIL_AVERAGE = 0.041
TV_RAIL_HIGHSPEED = 0.006
TV_CAR_AVERAGE = 0.171
TV_CAR_ELECTRIC = 0.053
TV_BUS = 0.089
TV_TAXI = 0.203

# Waste factors (DEFRA 2024, kgCO2e/kg)
WF_LANDFILL_MIXED = 0.586
WF_LANDFILL_ORGANIC = 0.700
WF_INCINERATION = 0.021
WF_RECYCLING_PAPER = -0.900
WF_RECYCLING_PLASTIC = -1.400
WF_RECYCLING_METAL = -1.800
WF_RECYCLING_GLASS = -0.300
WF_COMPOSTING = 0.010


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def agent() -> Scope3EmissionsAgent:
    """Create a Scope3EmissionsAgent instance for testing."""
    return Scope3EmissionsAgent()


@pytest.fixture
def agent_with_config() -> Scope3EmissionsAgent:
    """Create agent with custom configuration."""
    return Scope3EmissionsAgent(config={"custom_setting": "value"})


@pytest.fixture
def sample_spend_data_steel() -> List[SpendData]:
    """
    Create sample spend data for steel.

    Expected calculation:
        $1,000,000 * 0.85 kgCO2e/USD = 850,000 kgCO2e
    """
    return [SpendData(
        category="steel",
        spend_usd=1_000_000,
        supplier_name="Steel Supplier Inc",
        country="US"
    )]


@pytest.fixture
def sample_spend_data_multiple() -> List[SpendData]:
    """
    Create sample spend data with multiple categories.

    Expected calculation:
        Steel: $500,000 * 0.85 = 425,000 kgCO2e
        Plastics: $200,000 * 0.75 = 150,000 kgCO2e
        Services: $100,000 * 0.15 = 15,000 kgCO2e
        Total: 590,000 kgCO2e
    """
    return [
        SpendData(category="steel", spend_usd=500_000),
        SpendData(category="plastics", spend_usd=200_000),
        SpendData(category="services", spend_usd=100_000),
    ]


@pytest.fixture
def sample_transport_data_truck() -> List[TransportData]:
    """
    Create sample transport data for road truck.

    Expected calculation:
        1000 km * 50 tonnes * 0.089 kgCO2e/tonne-km = 4,450 kgCO2e
    """
    return [TransportData(
        mode="road_truck",
        distance_km=1000.0,
        weight_tonnes=50.0
    )]


@pytest.fixture
def sample_transport_data_multimodal() -> List[TransportData]:
    """
    Create sample multimodal transport data.

    Expected calculation:
        Truck: 500 km * 20 tonnes * 0.089 = 890 kgCO2e
        Sea: 5000 km * 20 tonnes * 0.016 = 1,600 kgCO2e
        Rail: 200 km * 20 tonnes * 0.028 = 112 kgCO2e
        Total: 2,602 kgCO2e
    """
    return [
        TransportData(mode="road_truck", distance_km=500.0, weight_tonnes=20.0),
        TransportData(mode="sea_container", distance_km=5000.0, weight_tonnes=20.0),
        TransportData(mode="rail_freight", distance_km=200.0, weight_tonnes=20.0),
    ]


@pytest.fixture
def sample_travel_data_air() -> List[TravelData]:
    """
    Create sample business travel data - short haul air.

    Expected calculation:
        1000 km (one way) * 0.255 kgCO2e/pkm = 255 kgCO2e
    """
    return [TravelData(
        mode="air",
        distance_km=1000.0,
        trip_type="one_way"
    )]


@pytest.fixture
def sample_travel_data_mixed() -> List[TravelData]:
    """
    Create sample mixed business travel data.

    Expected calculation:
        Air (short): 1000 km * 0.255 = 255 kgCO2e
        Rail: 500 km (round trip = 1000) * 0.041 = 41 kgCO2e
        Car: 200 km * 0.171 = 34.2 kgCO2e
        Total: 330.2 kgCO2e
    """
    return [
        TravelData(mode="air", distance_km=1000.0, trip_type="one_way"),
        TravelData(mode="rail_average", distance_km=500.0, trip_type="round_trip"),
        TravelData(mode="car_average", distance_km=200.0, trip_type="one_way"),
    ]


@pytest.fixture
def sample_waste_data() -> List[ActivityData]:
    """
    Create sample waste activity data.

    Expected calculation:
        Landfill: 5000 kg * 0.586 = 2,930 kgCO2e
        Recycling plastic: 2000 kg * -1.400 = -2,800 kgCO2e (avoided)
        Total: 130 kgCO2e
    """
    return [
        ActivityData(activity_type="landfill_mixed", quantity=5000.0, unit="kg"),
        ActivityData(activity_type="recycling_plastic", quantity=2000.0, unit="kg"),
    ]


@pytest.fixture
def cat1_input(sample_spend_data_steel) -> Scope3Input:
    """Create Category 1 (Purchased Goods) input."""
    return Scope3Input(
        category=Scope3Category.CAT_1_PURCHASED_GOODS,
        reporting_year=2024,
        spend_data=sample_spend_data_steel,
        calculation_method=CalculationMethod.SPEND_BASED,
    )


@pytest.fixture
def cat4_input(sample_transport_data_truck) -> Scope3Input:
    """Create Category 4 (Upstream Transport) input."""
    return Scope3Input(
        category=Scope3Category.CAT_4_UPSTREAM_TRANSPORT,
        reporting_year=2024,
        transport_data=sample_transport_data_truck,
        calculation_method=CalculationMethod.AVERAGE_DATA,
    )


@pytest.fixture
def cat5_input(sample_waste_data) -> Scope3Input:
    """Create Category 5 (Waste) input."""
    return Scope3Input(
        category=Scope3Category.CAT_5_WASTE,
        reporting_year=2024,
        activity_data=sample_waste_data,
        calculation_method=CalculationMethod.AVERAGE_DATA,
    )


@pytest.fixture
def cat6_input(sample_travel_data_air) -> Scope3Input:
    """Create Category 6 (Business Travel) input."""
    return Scope3Input(
        category=Scope3Category.CAT_6_BUSINESS_TRAVEL,
        reporting_year=2024,
        travel_data=sample_travel_data_air,
        calculation_method=CalculationMethod.AVERAGE_DATA,
    )


# =============================================================================
# Test 1-10: Agent Initialization and Basic Tests
# =============================================================================


class TestAgentInitialization:
    """Tests for agent initialization."""

    def test_01_agent_initialization(self, agent: Scope3EmissionsAgent):
        """Test 1: Agent initializes correctly with default config."""
        assert agent is not None
        assert agent.AGENT_ID == "emissions/scope3_v1"
        assert agent.VERSION == "1.0.0"
        assert agent.DESCRIPTION == "Scope 3 supply chain emissions calculator"

    def test_02_agent_with_custom_config(self, agent_with_config: Scope3EmissionsAgent):
        """Test 2: Agent initializes with custom configuration."""
        assert agent_with_config.config == {"custom_setting": "value"}

    def test_03_spend_factors_loaded(self, agent: Scope3EmissionsAgent):
        """Test 3: Spend-based emission factors are loaded correctly."""
        factors = agent.SPEND_FACTORS
        assert "steel" in factors
        assert "aluminum" in factors
        assert "plastics" in factors
        assert "default" in factors
        assert factors["steel"].factor_kgco2e_per_usd == EF_STEEL

    def test_04_transport_factors_loaded(self, agent: Scope3EmissionsAgent):
        """Test 4: Transport emission factors are loaded correctly."""
        factors = agent.TRANSPORT_FACTORS
        assert "road_truck" in factors
        assert "rail_freight" in factors
        assert "air_freight" in factors
        assert factors["road_truck"] == TF_ROAD_TRUCK

    def test_05_travel_factors_loaded(self, agent: Scope3EmissionsAgent):
        """Test 5: Travel emission factors are loaded correctly."""
        factors = agent.TRAVEL_FACTORS
        assert "air_short_haul" in factors
        assert "rail_average" in factors
        assert "car_average" in factors
        assert factors["air_short_haul"] == TV_AIR_SHORT_HAUL

    def test_06_waste_factors_loaded(self, agent: Scope3EmissionsAgent):
        """Test 6: Waste emission factors are loaded correctly."""
        factors = agent.WASTE_FACTORS
        assert "landfill_mixed" in factors
        assert "recycling_plastic" in factors
        assert "composting" in factors
        assert factors["landfill_mixed"] == WF_LANDFILL_MIXED

    def test_07_all_15_categories_defined(self, agent: Scope3EmissionsAgent):
        """Test 7: All 15 Scope 3 categories are defined."""
        categories = agent.CATEGORY_NAMES
        assert len(categories) == 15

        # Verify all category numbers 1-15
        category_numbers = [info[0] for info in categories.values()]
        assert sorted(category_numbers) == list(range(1, 16))

    def test_08_get_categories_method(self, agent: Scope3EmissionsAgent):
        """Test 8: get_categories returns all 15 categories."""
        categories = agent.get_categories()
        assert len(categories) == 15

        # Verify structure
        for cat in categories:
            assert "category" in cat
            assert "number" in cat
            assert "name" in cat

    def test_09_basic_run_completes(
        self,
        agent: Scope3EmissionsAgent,
        cat1_input: Scope3Input,
    ):
        """Test 9: Basic agent run completes successfully."""
        result = agent.run(cat1_input)
        assert result is not None
        assert isinstance(result, Scope3Output)

    def test_10_run_returns_provenance_hash(
        self,
        agent: Scope3EmissionsAgent,
        cat1_input: Scope3Input,
    ):
        """Test 10: Run returns valid SHA-256 provenance hash."""
        result = agent.run(cat1_input)
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256 hex string


# =============================================================================
# Test 11-25: Category 1 & 2 - Purchased/Capital Goods (Spend-Based)
# =============================================================================


class TestSpendBasedCategories:
    """Tests for spend-based categories (Cat 1, 2)."""

    @pytest.mark.golden
    def test_11_cat1_steel_1m_usd(
        self,
        agent: Scope3EmissionsAgent,
        cat1_input: Scope3Input,
    ):
        """
        GT-11: Category 1 - Steel spend $1M USD

        ZERO-HALLUCINATION CALCULATION:
        emissions = $1,000,000 * 0.85 kgCO2e/USD = 850,000 kgCO2e
        """
        result = agent.run(cat1_input)
        expected = 1_000_000 * EF_STEEL  # 850,000 kgCO2e
        assert result.total_emissions_kgco2e == pytest.approx(expected, rel=1e-6)
        assert result.category_number == 1
        assert result.category_name == "Purchased Goods and Services"

    @pytest.mark.golden
    def test_12_cat1_aluminum_500k_usd(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """
        GT-12: Category 1 - Aluminum spend $500K USD

        ZERO-HALLUCINATION CALCULATION:
        emissions = $500,000 * 1.20 kgCO2e/USD = 600,000 kgCO2e
        """
        input_data = Scope3Input(
            category=Scope3Category.CAT_1_PURCHASED_GOODS,
            reporting_year=2024,
            spend_data=[SpendData(category="aluminum", spend_usd=500_000)],
        )
        result = agent.run(input_data)
        expected = 500_000 * EF_ALUMINUM  # 600,000 kgCO2e
        assert result.total_emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.golden
    def test_13_cat1_multiple_categories(
        self,
        agent: Scope3EmissionsAgent,
        sample_spend_data_multiple: List[SpendData],
    ):
        """
        GT-13: Category 1 - Multiple spend categories

        ZERO-HALLUCINATION CALCULATION:
        Steel: $500,000 * 0.85 = 425,000 kgCO2e
        Plastics: $200,000 * 0.75 = 150,000 kgCO2e
        Services: $100,000 * 0.15 = 15,000 kgCO2e
        Total: 590,000 kgCO2e
        """
        input_data = Scope3Input(
            category=Scope3Category.CAT_1_PURCHASED_GOODS,
            reporting_year=2024,
            spend_data=sample_spend_data_multiple,
        )
        result = agent.run(input_data)

        expected_steel = 500_000 * EF_STEEL
        expected_plastics = 200_000 * EF_PLASTICS
        expected_services = 100_000 * EF_SERVICES
        expected_total = expected_steel + expected_plastics + expected_services

        assert result.total_emissions_kgco2e == pytest.approx(expected_total, rel=1e-6)

    @pytest.mark.golden
    def test_14_cat1_default_factor(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """
        GT-14: Category 1 - Unknown category uses default factor

        ZERO-HALLUCINATION CALCULATION:
        emissions = $100,000 * 0.40 kgCO2e/USD = 40,000 kgCO2e
        """
        input_data = Scope3Input(
            category=Scope3Category.CAT_1_PURCHASED_GOODS,
            reporting_year=2024,
            spend_data=[SpendData(category="unknown_category", spend_usd=100_000)],
        )
        result = agent.run(input_data)
        expected = 100_000 * EF_DEFAULT  # 40,000 kgCO2e
        assert result.total_emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.golden
    def test_15_cat2_capital_goods_machinery(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """
        GT-15: Category 2 - Capital goods (machinery)

        ZERO-HALLUCINATION CALCULATION:
        emissions = $2,000,000 * 0.40 kgCO2e/USD = 800,000 kgCO2e
        """
        input_data = Scope3Input(
            category=Scope3Category.CAT_2_CAPITAL_GOODS,
            reporting_year=2024,
            spend_data=[SpendData(category="machinery", spend_usd=2_000_000)],
        )
        result = agent.run(input_data)
        expected = 2_000_000 * EF_MACHINERY  # 800,000 kgCO2e
        assert result.total_emissions_kgco2e == pytest.approx(expected, rel=1e-6)
        assert result.category_number == 2
        assert result.category_name == "Capital Goods"

    @pytest.mark.golden
    def test_16_cat1_electronics(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """
        GT-16: Category 1 - Electronics spend

        ZERO-HALLUCINATION CALCULATION:
        emissions = $300,000 * 0.55 kgCO2e/USD = 165,000 kgCO2e
        """
        input_data = Scope3Input(
            category=Scope3Category.CAT_1_PURCHASED_GOODS,
            reporting_year=2024,
            spend_data=[SpendData(category="electronics", spend_usd=300_000)],
        )
        result = agent.run(input_data)
        expected = 300_000 * EF_ELECTRONICS  # 165,000 kgCO2e
        assert result.total_emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.golden
    def test_17_cat1_chemicals(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """
        GT-17: Category 1 - Chemicals spend

        ZERO-HALLUCINATION CALCULATION:
        emissions = $750,000 * 0.65 kgCO2e/USD = 487,500 kgCO2e
        """
        input_data = Scope3Input(
            category=Scope3Category.CAT_1_PURCHASED_GOODS,
            reporting_year=2024,
            spend_data=[SpendData(category="chemicals", spend_usd=750_000)],
        )
        result = agent.run(input_data)
        expected = 750_000 * EF_CHEMICALS  # 487,500 kgCO2e
        assert result.total_emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.golden
    def test_18_cat1_textiles(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """
        GT-18: Category 1 - Textiles spend

        ZERO-HALLUCINATION CALCULATION:
        emissions = $400,000 * 0.50 kgCO2e/USD = 200,000 kgCO2e
        """
        input_data = Scope3Input(
            category=Scope3Category.CAT_1_PURCHASED_GOODS,
            reporting_year=2024,
            spend_data=[SpendData(category="textiles", spend_usd=400_000)],
        )
        result = agent.run(input_data)
        expected = 400_000 * EF_TEXTILES  # 200,000 kgCO2e
        assert result.total_emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.golden
    def test_19_cat1_food(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """
        GT-19: Category 1 - Food products spend

        ZERO-HALLUCINATION CALCULATION:
        emissions = $250,000 * 0.60 kgCO2e/USD = 150,000 kgCO2e
        """
        input_data = Scope3Input(
            category=Scope3Category.CAT_1_PURCHASED_GOODS,
            reporting_year=2024,
            spend_data=[SpendData(category="food", spend_usd=250_000)],
        )
        result = agent.run(input_data)
        expected = 250_000 * EF_FOOD  # 150,000 kgCO2e
        assert result.total_emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.golden
    def test_20_cat1_paper(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """
        GT-20: Category 1 - Paper products spend

        ZERO-HALLUCINATION CALCULATION:
        emissions = $50,000 * 0.45 kgCO2e/USD = 22,500 kgCO2e
        """
        input_data = Scope3Input(
            category=Scope3Category.CAT_1_PURCHASED_GOODS,
            reporting_year=2024,
            spend_data=[SpendData(category="paper", spend_usd=50_000)],
        )
        result = agent.run(input_data)
        expected = 50_000 * EF_PAPER  # 22,500 kgCO2e
        assert result.total_emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.golden
    def test_21_cat1_construction(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """
        GT-21: Category 1 - Construction materials spend

        ZERO-HALLUCINATION CALCULATION:
        emissions = $1,500,000 * 0.35 kgCO2e/USD = 525,000 kgCO2e
        """
        input_data = Scope3Input(
            category=Scope3Category.CAT_1_PURCHASED_GOODS,
            reporting_year=2024,
            spend_data=[SpendData(category="construction", spend_usd=1_500_000)],
        )
        result = agent.run(input_data)
        expected = 1_500_000 * EF_CONSTRUCTION  # 525,000 kgCO2e
        assert result.total_emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.golden
    def test_22_cat1_it_services(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """
        GT-22: Category 1 - IT services spend

        ZERO-HALLUCINATION CALCULATION:
        emissions = $800,000 * 0.18 kgCO2e/USD = 144,000 kgCO2e
        """
        input_data = Scope3Input(
            category=Scope3Category.CAT_1_PURCHASED_GOODS,
            reporting_year=2024,
            spend_data=[SpendData(category="it_services", spend_usd=800_000)],
        )
        result = agent.run(input_data)
        expected = 800_000 * EF_IT_SERVICES  # 144,000 kgCO2e
        assert result.total_emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.golden
    def test_23_cat1_professional_services(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """
        GT-23: Category 1 - Professional services spend

        ZERO-HALLUCINATION CALCULATION:
        emissions = $500,000 * 0.12 kgCO2e/USD = 60,000 kgCO2e
        """
        input_data = Scope3Input(
            category=Scope3Category.CAT_1_PURCHASED_GOODS,
            reporting_year=2024,
            spend_data=[SpendData(category="professional_services", spend_usd=500_000)],
        )
        result = agent.run(input_data)
        expected = 500_000 * EF_PROFESSIONAL_SERVICES  # 60,000 kgCO2e
        assert result.total_emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.golden
    def test_24_cat1_emission_factor_info(
        self,
        agent: Scope3EmissionsAgent,
        cat1_input: Scope3Input,
    ):
        """GT-24: Emission factor information is returned."""
        result = agent.run(cat1_input)
        assert len(result.emission_factors_used) > 0
        factor = result.emission_factors_used[0]
        assert factor["category"] == "steel"
        assert factor["factor"] == EF_STEEL
        assert factor["unit"] == "kgCO2e/USD"
        assert "EPA" in factor["source"]

    @pytest.mark.golden
    def test_25_cat1_zero_spend(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """
        GT-25: Zero spend returns zero emissions

        ZERO-HALLUCINATION CALCULATION:
        emissions = $0 * 0.85 kgCO2e/USD = 0 kgCO2e
        """
        input_data = Scope3Input(
            category=Scope3Category.CAT_1_PURCHASED_GOODS,
            reporting_year=2024,
            spend_data=[SpendData(category="steel", spend_usd=0)],
        )
        result = agent.run(input_data)
        assert result.total_emissions_kgco2e == 0.0


# =============================================================================
# Test 26-35: Category 4 & 9 - Transport (Upstream/Downstream)
# =============================================================================


class TestTransportCategories:
    """Tests for transport categories (Cat 4, 9)."""

    @pytest.mark.golden
    def test_26_cat4_road_truck(
        self,
        agent: Scope3EmissionsAgent,
        cat4_input: Scope3Input,
    ):
        """
        GT-26: Category 4 - Road truck transport

        ZERO-HALLUCINATION CALCULATION:
        emissions = 1000 km * 50 tonnes * 0.089 kgCO2e/tkm = 4,450 kgCO2e
        """
        result = agent.run(cat4_input)
        expected = 1000.0 * 50.0 * TF_ROAD_TRUCK  # 4,450 kgCO2e
        assert result.total_emissions_kgco2e == pytest.approx(expected, rel=1e-6)
        assert result.category_number == 4
        assert result.category_name == "Upstream Transportation and Distribution"

    @pytest.mark.golden
    def test_27_cat4_rail_freight(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """
        GT-27: Category 4 - Rail freight transport

        ZERO-HALLUCINATION CALCULATION:
        emissions = 2000 km * 100 tonnes * 0.028 kgCO2e/tkm = 5,600 kgCO2e
        """
        input_data = Scope3Input(
            category=Scope3Category.CAT_4_UPSTREAM_TRANSPORT,
            reporting_year=2024,
            transport_data=[TransportData(mode="rail_freight", distance_km=2000.0, weight_tonnes=100.0)],
        )
        result = agent.run(input_data)
        expected = 2000.0 * 100.0 * TF_RAIL_FREIGHT  # 5,600 kgCO2e
        assert result.total_emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.golden
    def test_28_cat4_sea_container(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """
        GT-28: Category 4 - Sea container transport

        ZERO-HALLUCINATION CALCULATION:
        emissions = 10000 km * 500 tonnes * 0.016 kgCO2e/tkm = 80,000 kgCO2e
        """
        input_data = Scope3Input(
            category=Scope3Category.CAT_4_UPSTREAM_TRANSPORT,
            reporting_year=2024,
            transport_data=[TransportData(mode="sea_container", distance_km=10000.0, weight_tonnes=500.0)],
        )
        result = agent.run(input_data)
        expected = 10000.0 * 500.0 * TF_SEA_CONTAINER  # 80,000 kgCO2e
        assert result.total_emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.golden
    def test_29_cat4_air_freight(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """
        GT-29: Category 4 - Air freight transport (highest emission factor)

        ZERO-HALLUCINATION CALCULATION:
        emissions = 5000 km * 10 tonnes * 0.602 kgCO2e/tkm = 30,100 kgCO2e
        """
        input_data = Scope3Input(
            category=Scope3Category.CAT_4_UPSTREAM_TRANSPORT,
            reporting_year=2024,
            transport_data=[TransportData(mode="air_freight", distance_km=5000.0, weight_tonnes=10.0)],
        )
        result = agent.run(input_data)
        expected = 5000.0 * 10.0 * TF_AIR_FREIGHT  # 30,100 kgCO2e
        assert result.total_emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.golden
    def test_30_cat4_multimodal(
        self,
        agent: Scope3EmissionsAgent,
        sample_transport_data_multimodal: List[TransportData],
    ):
        """
        GT-30: Category 4 - Multimodal transport

        ZERO-HALLUCINATION CALCULATION:
        Truck: 500 km * 20 tonnes * 0.089 = 890 kgCO2e
        Sea: 5000 km * 20 tonnes * 0.016 = 1,600 kgCO2e
        Rail: 200 km * 20 tonnes * 0.028 = 112 kgCO2e
        Total: 2,602 kgCO2e
        """
        input_data = Scope3Input(
            category=Scope3Category.CAT_4_UPSTREAM_TRANSPORT,
            reporting_year=2024,
            transport_data=sample_transport_data_multimodal,
        )
        result = agent.run(input_data)

        expected_truck = 500.0 * 20.0 * TF_ROAD_TRUCK
        expected_sea = 5000.0 * 20.0 * TF_SEA_CONTAINER
        expected_rail = 200.0 * 20.0 * TF_RAIL_FREIGHT
        expected_total = expected_truck + expected_sea + expected_rail

        assert result.total_emissions_kgco2e == pytest.approx(expected_total, rel=1e-6)

    @pytest.mark.golden
    def test_31_cat9_downstream_transport(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """
        GT-31: Category 9 - Downstream transport (same calculation as Cat 4)

        ZERO-HALLUCINATION CALCULATION:
        emissions = 800 km * 30 tonnes * 0.089 kgCO2e/tkm = 2,136 kgCO2e
        """
        input_data = Scope3Input(
            category=Scope3Category.CAT_9_DOWNSTREAM_TRANSPORT,
            reporting_year=2024,
            transport_data=[TransportData(mode="road_truck", distance_km=800.0, weight_tonnes=30.0)],
        )
        result = agent.run(input_data)
        expected = 800.0 * 30.0 * TF_ROAD_TRUCK  # 2,136 kgCO2e
        assert result.total_emissions_kgco2e == pytest.approx(expected, rel=1e-6)
        assert result.category_number == 9
        assert result.category_name == "Downstream Transportation and Distribution"

    @pytest.mark.golden
    def test_32_cat4_road_van(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """
        GT-32: Category 4 - Road van (last mile delivery)

        ZERO-HALLUCINATION CALCULATION:
        emissions = 100 km * 2 tonnes * 0.195 kgCO2e/tkm = 39 kgCO2e
        """
        input_data = Scope3Input(
            category=Scope3Category.CAT_4_UPSTREAM_TRANSPORT,
            reporting_year=2024,
            transport_data=[TransportData(mode="road_van", distance_km=100.0, weight_tonnes=2.0)],
        )
        result = agent.run(input_data)
        expected = 100.0 * 2.0 * TF_ROAD_VAN  # 39 kgCO2e
        assert result.total_emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.golden
    def test_33_cat4_sea_bulk(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """
        GT-33: Category 4 - Sea bulk carrier

        ZERO-HALLUCINATION CALCULATION:
        emissions = 15000 km * 5000 tonnes * 0.008 kgCO2e/tkm = 600,000 kgCO2e
        """
        input_data = Scope3Input(
            category=Scope3Category.CAT_4_UPSTREAM_TRANSPORT,
            reporting_year=2024,
            transport_data=[TransportData(mode="sea_bulk", distance_km=15000.0, weight_tonnes=5000.0)],
        )
        result = agent.run(input_data)
        expected = 15000.0 * 5000.0 * TF_SEA_BULK  # 600,000 kgCO2e
        assert result.total_emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.golden
    def test_34_cat4_barge(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """
        GT-34: Category 4 - Barge transport

        ZERO-HALLUCINATION CALCULATION:
        emissions = 500 km * 200 tonnes * 0.031 kgCO2e/tkm = 3,100 kgCO2e
        """
        input_data = Scope3Input(
            category=Scope3Category.CAT_4_UPSTREAM_TRANSPORT,
            reporting_year=2024,
            transport_data=[TransportData(mode="barge", distance_km=500.0, weight_tonnes=200.0)],
        )
        result = agent.run(input_data)
        expected = 500.0 * 200.0 * TF_BARGE  # 3,100 kgCO2e
        assert result.total_emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.golden
    def test_35_cat4_pipeline(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """
        GT-35: Category 4 - Pipeline transport

        ZERO-HALLUCINATION CALCULATION:
        emissions = 1000 km * 10000 tonnes * 0.025 kgCO2e/tkm = 250,000 kgCO2e
        """
        input_data = Scope3Input(
            category=Scope3Category.CAT_4_UPSTREAM_TRANSPORT,
            reporting_year=2024,
            transport_data=[TransportData(mode="pipeline", distance_km=1000.0, weight_tonnes=10000.0)],
        )
        result = agent.run(input_data)
        expected = 1000.0 * 10000.0 * TF_PIPELINE  # 250,000 kgCO2e
        assert result.total_emissions_kgco2e == pytest.approx(expected, rel=1e-6)


# =============================================================================
# Test 36-45: Category 6 - Business Travel
# =============================================================================


class TestBusinessTravelCategory:
    """Tests for Category 6 - Business Travel."""

    @pytest.mark.golden
    def test_36_cat6_air_short_haul(
        self,
        agent: Scope3EmissionsAgent,
        cat6_input: Scope3Input,
    ):
        """
        GT-36: Category 6 - Air short haul (<1500km)

        ZERO-HALLUCINATION CALCULATION:
        emissions = 1000 km * 0.255 kgCO2e/pkm = 255 kgCO2e
        """
        result = agent.run(cat6_input)
        expected = 1000.0 * TV_AIR_SHORT_HAUL  # 255 kgCO2e
        assert result.total_emissions_kgco2e == pytest.approx(expected, rel=1e-6)
        assert result.category_number == 6
        assert result.category_name == "Business Travel"

    @pytest.mark.golden
    def test_37_cat6_air_medium_haul(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """
        GT-37: Category 6 - Air medium haul (1500-4000km)

        ZERO-HALLUCINATION CALCULATION:
        emissions = 3000 km * 0.156 kgCO2e/pkm = 468 kgCO2e
        """
        input_data = Scope3Input(
            category=Scope3Category.CAT_6_BUSINESS_TRAVEL,
            reporting_year=2024,
            travel_data=[TravelData(mode="air", distance_km=3000.0, trip_type="one_way")],
        )
        result = agent.run(input_data)
        expected = 3000.0 * TV_AIR_MEDIUM_HAUL  # 468 kgCO2e
        assert result.total_emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.golden
    def test_38_cat6_air_long_haul(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """
        GT-38: Category 6 - Air long haul (>4000km)

        ZERO-HALLUCINATION CALCULATION:
        emissions = 8000 km * 0.195 kgCO2e/pkm = 1,560 kgCO2e
        """
        input_data = Scope3Input(
            category=Scope3Category.CAT_6_BUSINESS_TRAVEL,
            reporting_year=2024,
            travel_data=[TravelData(mode="air", distance_km=8000.0, trip_type="one_way")],
        )
        result = agent.run(input_data)
        expected = 8000.0 * TV_AIR_LONG_HAUL  # 1,560 kgCO2e
        assert result.total_emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.golden
    def test_39_cat6_rail_average(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """
        GT-39: Category 6 - Rail travel (average)

        ZERO-HALLUCINATION CALCULATION:
        emissions = 500 km * 0.041 kgCO2e/pkm = 20.5 kgCO2e
        """
        input_data = Scope3Input(
            category=Scope3Category.CAT_6_BUSINESS_TRAVEL,
            reporting_year=2024,
            travel_data=[TravelData(mode="rail_average", distance_km=500.0, trip_type="one_way")],
        )
        result = agent.run(input_data)
        expected = 500.0 * TV_RAIL_AVERAGE  # 20.5 kgCO2e
        assert result.total_emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.golden
    def test_40_cat6_rail_highspeed(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """
        GT-40: Category 6 - High speed rail (lowest factor)

        ZERO-HALLUCINATION CALCULATION:
        emissions = 800 km * 0.006 kgCO2e/pkm = 4.8 kgCO2e
        """
        input_data = Scope3Input(
            category=Scope3Category.CAT_6_BUSINESS_TRAVEL,
            reporting_year=2024,
            travel_data=[TravelData(mode="rail_highspeed", distance_km=800.0, trip_type="one_way")],
        )
        result = agent.run(input_data)
        expected = 800.0 * TV_RAIL_HIGHSPEED  # 4.8 kgCO2e
        assert result.total_emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.golden
    def test_41_cat6_car_average(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """
        GT-41: Category 6 - Car travel (average)

        ZERO-HALLUCINATION CALCULATION:
        emissions = 300 km * 0.171 kgCO2e/pkm = 51.3 kgCO2e
        """
        input_data = Scope3Input(
            category=Scope3Category.CAT_6_BUSINESS_TRAVEL,
            reporting_year=2024,
            travel_data=[TravelData(mode="car_average", distance_km=300.0, trip_type="one_way")],
        )
        result = agent.run(input_data)
        expected = 300.0 * TV_CAR_AVERAGE  # 51.3 kgCO2e
        assert result.total_emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.golden
    def test_42_cat6_round_trip(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """
        GT-42: Category 6 - Round trip doubles distance

        ZERO-HALLUCINATION CALCULATION:
        emissions = 500 km * 2 (round trip) * 0.255 kgCO2e/pkm = 255 kgCO2e
        """
        input_data = Scope3Input(
            category=Scope3Category.CAT_6_BUSINESS_TRAVEL,
            reporting_year=2024,
            travel_data=[TravelData(mode="air", distance_km=500.0, trip_type="round_trip")],
        )
        result = agent.run(input_data)
        expected = 500.0 * 2 * TV_AIR_SHORT_HAUL  # 255 kgCO2e
        assert result.total_emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.golden
    def test_43_cat6_taxi(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """
        GT-43: Category 6 - Taxi travel

        ZERO-HALLUCINATION CALCULATION:
        emissions = 50 km * 0.203 kgCO2e/pkm = 10.15 kgCO2e
        """
        input_data = Scope3Input(
            category=Scope3Category.CAT_6_BUSINESS_TRAVEL,
            reporting_year=2024,
            travel_data=[TravelData(mode="taxi", distance_km=50.0, trip_type="one_way")],
        )
        result = agent.run(input_data)
        expected = 50.0 * TV_TAXI  # 10.15 kgCO2e
        assert result.total_emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.golden
    def test_44_cat6_bus(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """
        GT-44: Category 6 - Bus travel

        ZERO-HALLUCINATION CALCULATION:
        emissions = 200 km * 0.089 kgCO2e/pkm = 17.8 kgCO2e
        """
        input_data = Scope3Input(
            category=Scope3Category.CAT_6_BUSINESS_TRAVEL,
            reporting_year=2024,
            travel_data=[TravelData(mode="bus", distance_km=200.0, trip_type="one_way")],
        )
        result = agent.run(input_data)
        expected = 200.0 * TV_BUS  # 17.8 kgCO2e
        assert result.total_emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.golden
    def test_45_cat6_mixed_travel(
        self,
        agent: Scope3EmissionsAgent,
        sample_travel_data_mixed: List[TravelData],
    ):
        """
        GT-45: Category 6 - Mixed travel modes

        ZERO-HALLUCINATION CALCULATION:
        Air (short): 1000 km * 0.255 = 255 kgCO2e
        Rail (round): 500 km * 2 * 0.041 = 41 kgCO2e
        Car: 200 km * 0.171 = 34.2 kgCO2e
        Total: 330.2 kgCO2e
        """
        input_data = Scope3Input(
            category=Scope3Category.CAT_6_BUSINESS_TRAVEL,
            reporting_year=2024,
            travel_data=sample_travel_data_mixed,
        )
        result = agent.run(input_data)

        expected_air = 1000.0 * TV_AIR_SHORT_HAUL
        expected_rail = 500.0 * 2 * TV_RAIL_AVERAGE  # round trip
        expected_car = 200.0 * TV_CAR_AVERAGE
        expected_total = expected_air + expected_rail + expected_car

        assert result.total_emissions_kgco2e == pytest.approx(expected_total, rel=1e-6)


# =============================================================================
# Test 46-55: Category 5 - Waste Generated in Operations
# =============================================================================


class TestWasteCategory:
    """Tests for Category 5 - Waste Generated in Operations."""

    @pytest.mark.golden
    def test_46_cat5_landfill_mixed(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """
        GT-46: Category 5 - Landfill (mixed waste)

        ZERO-HALLUCINATION CALCULATION:
        emissions = 10000 kg * 0.586 kgCO2e/kg = 5,860 kgCO2e
        """
        input_data = Scope3Input(
            category=Scope3Category.CAT_5_WASTE,
            reporting_year=2024,
            activity_data=[ActivityData(activity_type="landfill_mixed", quantity=10000.0, unit="kg")],
        )
        result = agent.run(input_data)
        expected = 10000.0 * WF_LANDFILL_MIXED  # 5,860 kgCO2e
        assert result.total_emissions_kgco2e == pytest.approx(expected, rel=1e-6)
        assert result.category_number == 5
        assert result.category_name == "Waste Generated in Operations"

    @pytest.mark.golden
    def test_47_cat5_landfill_organic(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """
        GT-47: Category 5 - Landfill (organic waste)

        ZERO-HALLUCINATION CALCULATION:
        emissions = 5000 kg * 0.700 kgCO2e/kg = 3,500 kgCO2e
        """
        input_data = Scope3Input(
            category=Scope3Category.CAT_5_WASTE,
            reporting_year=2024,
            activity_data=[ActivityData(activity_type="landfill_organic", quantity=5000.0, unit="kg")],
        )
        result = agent.run(input_data)
        expected = 5000.0 * WF_LANDFILL_ORGANIC  # 3,500 kgCO2e
        assert result.total_emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.golden
    def test_48_cat5_incineration(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """
        GT-48: Category 5 - Incineration (low emissions due to energy recovery)

        ZERO-HALLUCINATION CALCULATION:
        emissions = 8000 kg * 0.021 kgCO2e/kg = 168 kgCO2e
        """
        input_data = Scope3Input(
            category=Scope3Category.CAT_5_WASTE,
            reporting_year=2024,
            activity_data=[ActivityData(activity_type="incineration", quantity=8000.0, unit="kg")],
        )
        result = agent.run(input_data)
        expected = 8000.0 * WF_INCINERATION  # 168 kgCO2e
        assert result.total_emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.golden
    def test_49_cat5_recycling_paper(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """
        GT-49: Category 5 - Paper recycling (negative = avoided emissions)

        ZERO-HALLUCINATION CALCULATION:
        emissions = 3000 kg * -0.900 kgCO2e/kg = -2,700 kgCO2e
        """
        input_data = Scope3Input(
            category=Scope3Category.CAT_5_WASTE,
            reporting_year=2024,
            activity_data=[ActivityData(activity_type="recycling_paper", quantity=3000.0, unit="kg")],
        )
        result = agent.run(input_data)
        expected = 3000.0 * WF_RECYCLING_PAPER  # -2,700 kgCO2e
        assert result.total_emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.golden
    def test_50_cat5_recycling_plastic(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """
        GT-50: Category 5 - Plastic recycling

        ZERO-HALLUCINATION CALCULATION:
        emissions = 2000 kg * -1.400 kgCO2e/kg = -2,800 kgCO2e
        """
        input_data = Scope3Input(
            category=Scope3Category.CAT_5_WASTE,
            reporting_year=2024,
            activity_data=[ActivityData(activity_type="recycling_plastic", quantity=2000.0, unit="kg")],
        )
        result = agent.run(input_data)
        expected = 2000.0 * WF_RECYCLING_PLASTIC  # -2,800 kgCO2e
        assert result.total_emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.golden
    def test_51_cat5_recycling_metal(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """
        GT-51: Category 5 - Metal recycling (highest avoided emissions)

        ZERO-HALLUCINATION CALCULATION:
        emissions = 1500 kg * -1.800 kgCO2e/kg = -2,700 kgCO2e
        """
        input_data = Scope3Input(
            category=Scope3Category.CAT_5_WASTE,
            reporting_year=2024,
            activity_data=[ActivityData(activity_type="recycling_metal", quantity=1500.0, unit="kg")],
        )
        result = agent.run(input_data)
        expected = 1500.0 * WF_RECYCLING_METAL  # -2,700 kgCO2e
        assert result.total_emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.golden
    def test_52_cat5_recycling_glass(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """
        GT-52: Category 5 - Glass recycling

        ZERO-HALLUCINATION CALCULATION:
        emissions = 1000 kg * -0.300 kgCO2e/kg = -300 kgCO2e
        """
        input_data = Scope3Input(
            category=Scope3Category.CAT_5_WASTE,
            reporting_year=2024,
            activity_data=[ActivityData(activity_type="recycling_glass", quantity=1000.0, unit="kg")],
        )
        result = agent.run(input_data)
        expected = 1000.0 * WF_RECYCLING_GLASS  # -300 kgCO2e
        assert result.total_emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.golden
    def test_53_cat5_composting(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """
        GT-53: Category 5 - Composting

        ZERO-HALLUCINATION CALCULATION:
        emissions = 4000 kg * 0.010 kgCO2e/kg = 40 kgCO2e
        """
        input_data = Scope3Input(
            category=Scope3Category.CAT_5_WASTE,
            reporting_year=2024,
            activity_data=[ActivityData(activity_type="composting", quantity=4000.0, unit="kg")],
        )
        result = agent.run(input_data)
        expected = 4000.0 * WF_COMPOSTING  # 40 kgCO2e
        assert result.total_emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.golden
    def test_54_cat5_tonnes_conversion(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """
        GT-54: Category 5 - Tonnes to kg conversion

        ZERO-HALLUCINATION CALCULATION:
        5 tonnes = 5000 kg
        emissions = 5000 kg * 0.586 kgCO2e/kg = 2,930 kgCO2e
        """
        input_data = Scope3Input(
            category=Scope3Category.CAT_5_WASTE,
            reporting_year=2024,
            activity_data=[ActivityData(activity_type="landfill_mixed", quantity=5.0, unit="tonnes")],
        )
        result = agent.run(input_data)
        expected = 5.0 * 1000 * WF_LANDFILL_MIXED  # 2,930 kgCO2e
        assert result.total_emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.golden
    def test_55_cat5_mixed_waste_streams(
        self,
        agent: Scope3EmissionsAgent,
        sample_waste_data: List[ActivityData],
    ):
        """
        GT-55: Category 5 - Mixed waste streams

        ZERO-HALLUCINATION CALCULATION:
        Landfill: 5000 kg * 0.586 = 2,930 kgCO2e
        Recycling plastic: 2000 kg * -1.400 = -2,800 kgCO2e
        Total: 130 kgCO2e
        """
        input_data = Scope3Input(
            category=Scope3Category.CAT_5_WASTE,
            reporting_year=2024,
            activity_data=sample_waste_data,
        )
        result = agent.run(input_data)

        expected_landfill = 5000.0 * WF_LANDFILL_MIXED
        expected_recycling = 2000.0 * WF_RECYCLING_PLASTIC
        expected_total = expected_landfill + expected_recycling

        assert result.total_emissions_kgco2e == pytest.approx(expected_total, rel=1e-6)


# =============================================================================
# Test 56-65: Other Scope 3 Categories (3, 7, 8, 10-15)
# =============================================================================


class TestOtherCategories:
    """Tests for other Scope 3 categories using spend-based fallback."""

    @pytest.mark.golden
    def test_56_cat3_fuel_energy(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """
        GT-56: Category 3 - Fuel and Energy Related Activities
        """
        input_data = Scope3Input(
            category=Scope3Category.CAT_3_FUEL_ENERGY,
            reporting_year=2024,
            spend_data=[SpendData(category="energy_services", spend_usd=200_000)],
        )
        result = agent.run(input_data)
        assert result.category_number == 3
        assert result.category_name == "Fuel- and Energy-Related Activities"
        expected = 200_000 * EF_DEFAULT  # Uses default factor
        assert result.total_emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.golden
    def test_57_cat7_commuting(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """
        GT-57: Category 7 - Employee Commuting
        """
        input_data = Scope3Input(
            category=Scope3Category.CAT_7_COMMUTING,
            reporting_year=2024,
            spend_data=[SpendData(category="commuting_allowance", spend_usd=100_000)],
        )
        result = agent.run(input_data)
        assert result.category_number == 7
        assert result.category_name == "Employee Commuting"

    @pytest.mark.golden
    def test_58_cat8_upstream_leased(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """
        GT-58: Category 8 - Upstream Leased Assets
        """
        input_data = Scope3Input(
            category=Scope3Category.CAT_8_UPSTREAM_LEASED,
            reporting_year=2024,
            spend_data=[SpendData(category="leased_equipment", spend_usd=500_000)],
        )
        result = agent.run(input_data)
        assert result.category_number == 8
        assert result.category_name == "Upstream Leased Assets"

    @pytest.mark.golden
    def test_59_cat10_processing(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """
        GT-59: Category 10 - Processing of Sold Products
        """
        input_data = Scope3Input(
            category=Scope3Category.CAT_10_PROCESSING,
            reporting_year=2024,
            spend_data=[SpendData(category="processing", spend_usd=300_000)],
        )
        result = agent.run(input_data)
        assert result.category_number == 10
        assert result.category_name == "Processing of Sold Products"

    @pytest.mark.golden
    def test_60_cat11_use_of_products(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """
        GT-60: Category 11 - Use of Sold Products
        """
        input_data = Scope3Input(
            category=Scope3Category.CAT_11_USE_OF_PRODUCTS,
            reporting_year=2024,
            spend_data=[SpendData(category="product_energy_use", spend_usd=1_000_000)],
        )
        result = agent.run(input_data)
        assert result.category_number == 11
        assert result.category_name == "Use of Sold Products"

    @pytest.mark.golden
    def test_61_cat12_end_of_life(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """
        GT-61: Category 12 - End-of-Life Treatment of Sold Products
        """
        input_data = Scope3Input(
            category=Scope3Category.CAT_12_END_OF_LIFE,
            reporting_year=2024,
            spend_data=[SpendData(category="end_of_life", spend_usd=150_000)],
        )
        result = agent.run(input_data)
        assert result.category_number == 12
        assert result.category_name == "End-of-Life Treatment of Sold Products"

    @pytest.mark.golden
    def test_62_cat13_downstream_leased(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """
        GT-62: Category 13 - Downstream Leased Assets
        """
        input_data = Scope3Input(
            category=Scope3Category.CAT_13_DOWNSTREAM_LEASED,
            reporting_year=2024,
            spend_data=[SpendData(category="leased_assets_out", spend_usd=400_000)],
        )
        result = agent.run(input_data)
        assert result.category_number == 13
        assert result.category_name == "Downstream Leased Assets"

    @pytest.mark.golden
    def test_63_cat14_franchises(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """
        GT-63: Category 14 - Franchises
        """
        input_data = Scope3Input(
            category=Scope3Category.CAT_14_FRANCHISES,
            reporting_year=2024,
            spend_data=[SpendData(category="franchise_operations", spend_usd=2_000_000)],
        )
        result = agent.run(input_data)
        assert result.category_number == 14
        assert result.category_name == "Franchises"

    @pytest.mark.golden
    def test_64_cat15_investments(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """
        GT-64: Category 15 - Investments
        """
        input_data = Scope3Input(
            category=Scope3Category.CAT_15_INVESTMENTS,
            reporting_year=2024,
            spend_data=[SpendData(category="investment_portfolio", spend_usd=10_000_000)],
        )
        result = agent.run(input_data)
        assert result.category_number == 15
        assert result.category_name == "Investments"

    def test_65_all_categories_processable(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """GT-65: All 15 categories can be processed."""
        for category in Scope3Category:
            input_data = Scope3Input(
                category=category,
                reporting_year=2024,
                spend_data=[SpendData(category="test", spend_usd=100_000)],
            )
            result = agent.run(input_data)
            assert result is not None
            assert result.category == category.value


# =============================================================================
# Test 66-70: Data Quality and Intensity Metrics
# =============================================================================


class TestDataQualityAndMetrics:
    """Tests for data quality assessment and intensity metrics."""

    def test_66_data_quality_spend_based(
        self,
        agent: Scope3EmissionsAgent,
        cat1_input: Scope3Input,
    ):
        """GT-66: Spend-based method gets FAIR data quality."""
        result = agent.run(cat1_input)
        assert result.data_quality_score in ["fair", "poor"]
        assert result.data_coverage_pct == pytest.approx(50.0, rel=0.1)

    def test_67_data_quality_supplier_specific(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """GT-67: Supplier-specific method gets VERY_GOOD data quality."""
        input_data = Scope3Input(
            category=Scope3Category.CAT_1_PURCHASED_GOODS,
            reporting_year=2024,
            spend_data=[SpendData(category="steel", spend_usd=1_000_000)],
            calculation_method=CalculationMethod.SUPPLIER_SPECIFIC,
        )
        result = agent.run(input_data)
        assert result.data_quality_score == "very_good"
        assert result.data_coverage_pct == pytest.approx(95.0, rel=0.1)

    def test_68_data_quality_hybrid(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """GT-68: Hybrid method gets GOOD data quality."""
        input_data = Scope3Input(
            category=Scope3Category.CAT_1_PURCHASED_GOODS,
            reporting_year=2024,
            spend_data=[SpendData(category="steel", spend_usd=1_000_000)],
            calculation_method=CalculationMethod.HYBRID,
        )
        result = agent.run(input_data)
        assert result.data_quality_score == "good"
        assert result.data_coverage_pct == pytest.approx(80.0, rel=0.1)

    def test_69_emissions_per_revenue(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """GT-69: Emissions per revenue intensity metric."""
        input_data = Scope3Input(
            category=Scope3Category.CAT_1_PURCHASED_GOODS,
            reporting_year=2024,
            spend_data=[SpendData(category="steel", spend_usd=1_000_000)],
            revenue_usd=10_000_000,
        )
        result = agent.run(input_data)

        # emissions = 850,000 kgCO2e / $10M = 0.085 kgCO2e/USD
        expected_intensity = 850_000 / 10_000_000
        assert result.emissions_per_revenue == pytest.approx(expected_intensity, rel=1e-4)

    def test_70_emissions_per_employee(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """GT-70: Emissions per employee intensity metric."""
        input_data = Scope3Input(
            category=Scope3Category.CAT_1_PURCHASED_GOODS,
            reporting_year=2024,
            spend_data=[SpendData(category="steel", spend_usd=1_000_000)],
            employees=1000,
        )
        result = agent.run(input_data)

        # emissions = 850,000 kgCO2e / 1000 employees = 850 kgCO2e/employee
        expected_intensity = 850_000 / 1000
        assert result.emissions_per_employee == pytest.approx(expected_intensity, rel=1e-4)


# =============================================================================
# Test 71-75: Determinism and Provenance
# =============================================================================


class TestDeterminismAndProvenance:
    """Tests for deterministic calculations and provenance tracking."""

    @pytest.mark.golden
    def test_71_deterministic_same_inputs(
        self,
        agent: Scope3EmissionsAgent,
        cat1_input: Scope3Input,
    ):
        """
        GT-71: Same inputs produce same emissions (zero-hallucination)

        Verifies the calculation is deterministic - no LLM involved in math.
        """
        result1 = agent.run(cat1_input)
        result2 = agent.run(cat1_input)
        result3 = agent.run(cat1_input)

        assert result1.total_emissions_kgco2e == result2.total_emissions_kgco2e
        assert result2.total_emissions_kgco2e == result3.total_emissions_kgco2e

    @pytest.mark.golden
    def test_72_deterministic_across_instances(
        self,
        cat1_input: Scope3Input,
    ):
        """
        GT-72: Different agent instances produce same results

        Verifies the calculation doesn't depend on instance state.
        """
        agent1 = Scope3EmissionsAgent()
        agent2 = Scope3EmissionsAgent()
        agent3 = Scope3EmissionsAgent()

        result1 = agent1.run(cat1_input)
        result2 = agent2.run(cat1_input)
        result3 = agent3.run(cat1_input)

        assert result1.total_emissions_kgco2e == result2.total_emissions_kgco2e
        assert result2.total_emissions_kgco2e == result3.total_emissions_kgco2e

    def test_73_provenance_hash_format(
        self,
        agent: Scope3EmissionsAgent,
        cat1_input: Scope3Input,
    ):
        """GT-73: Provenance hash is valid SHA-256 format."""
        result = agent.run(cat1_input)

        # SHA-256 produces 64 hex characters
        assert len(result.provenance_hash) == 64
        # Should be valid hex
        assert all(c in "0123456789abcdef" for c in result.provenance_hash)

    def test_74_provenance_hash_changes_with_inputs(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """GT-74: Provenance hash changes when inputs change."""
        input1 = Scope3Input(
            category=Scope3Category.CAT_1_PURCHASED_GOODS,
            reporting_year=2024,
            spend_data=[SpendData(category="steel", spend_usd=1_000_000)],
        )
        input2 = Scope3Input(
            category=Scope3Category.CAT_1_PURCHASED_GOODS,
            reporting_year=2024,
            spend_data=[SpendData(category="steel", spend_usd=2_000_000)],  # Different
        )

        result1 = agent.run(input1)
        result2 = agent.run(input2)

        # Different inputs should produce different provenance hashes
        assert result1.provenance_hash != result2.provenance_hash

    def test_75_provenance_hash_unique_per_run(
        self,
        agent: Scope3EmissionsAgent,
        cat1_input: Scope3Input,
    ):
        """GT-75: Provenance hash is unique per run (includes timestamp)."""
        result1 = agent.run(cat1_input)
        result2 = agent.run(cat1_input)

        # Same input should produce different hashes due to timestamp
        assert result1.provenance_hash != result2.provenance_hash


# =============================================================================
# Test 76-80: Edge Cases and Boundary Conditions
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_76_zero_quantity(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """GT-76: Zero spend returns zero emissions."""
        input_data = Scope3Input(
            category=Scope3Category.CAT_1_PURCHASED_GOODS,
            reporting_year=2024,
            spend_data=[SpendData(category="steel", spend_usd=0)],
        )
        result = agent.run(input_data)
        assert result.total_emissions_kgco2e == 0.0

    def test_77_very_small_quantity(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """
        GT-77: Very small quantity (precision test)

        ZERO-HALLUCINATION CALCULATION:
        emissions = $0.01 * 0.85 kgCO2e/USD = 0.0085 kgCO2e
        """
        input_data = Scope3Input(
            category=Scope3Category.CAT_1_PURCHASED_GOODS,
            reporting_year=2024,
            spend_data=[SpendData(category="steel", spend_usd=0.01)],
        )
        result = agent.run(input_data)
        expected = 0.01 * EF_STEEL  # 0.0085
        assert result.total_emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    def test_78_very_large_quantity(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """
        GT-78: Very large quantity (corporate-level)

        ZERO-HALLUCINATION CALCULATION:
        emissions = $1,000,000,000 * 0.85 kgCO2e/USD = 850,000,000 kgCO2e
        """
        input_data = Scope3Input(
            category=Scope3Category.CAT_1_PURCHASED_GOODS,
            reporting_year=2024,
            spend_data=[SpendData(category="steel", spend_usd=1_000_000_000)],
        )
        result = agent.run(input_data)
        expected = 1_000_000_000 * EF_STEEL  # 850,000,000 kgCO2e
        assert result.total_emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    def test_79_empty_spend_data(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """GT-79: Empty spend data returns zero emissions."""
        input_data = Scope3Input(
            category=Scope3Category.CAT_1_PURCHASED_GOODS,
            reporting_year=2024,
            spend_data=[],
        )
        result = agent.run(input_data)
        assert result.total_emissions_kgco2e == 0.0

    def test_80_empty_transport_data(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """GT-80: Empty transport data returns zero emissions."""
        input_data = Scope3Input(
            category=Scope3Category.CAT_4_UPSTREAM_TRANSPORT,
            reporting_year=2024,
            transport_data=[],
        )
        result = agent.run(input_data)
        assert result.total_emissions_kgco2e == 0.0


# =============================================================================
# Test 81-85: Improvement Recommendations and Output Validation
# =============================================================================


class TestImprovementsAndOutput:
    """Tests for improvement recommendations and output validation."""

    def test_81_improvement_recommendations_generated(
        self,
        agent: Scope3EmissionsAgent,
        cat1_input: Scope3Input,
    ):
        """GT-81: Improvement recommendations are generated."""
        result = agent.run(cat1_input)
        assert len(result.improvement_opportunities) > 0

    def test_82_spend_based_recommendations(
        self,
        agent: Scope3EmissionsAgent,
        cat1_input: Scope3Input,
    ):
        """GT-82: Spend-based method gets supplier data recommendations."""
        result = agent.run(cat1_input)
        recommendations = " ".join(result.improvement_opportunities).lower()
        assert "supplier" in recommendations or "primary data" in recommendations

    def test_83_travel_category_recommendations(
        self,
        agent: Scope3EmissionsAgent,
        cat6_input: Scope3Input,
    ):
        """GT-83: Business travel category gets travel-specific recommendations."""
        result = agent.run(cat6_input)
        recommendations = " ".join(result.improvement_opportunities).lower()
        assert "virtual" in recommendations or "rail" in recommendations or "trip" in recommendations

    def test_84_output_timestamp(
        self,
        agent: Scope3EmissionsAgent,
        cat1_input: Scope3Input,
    ):
        """GT-84: Output includes calculated_at timestamp."""
        result = agent.run(cat1_input)
        assert result.calculated_at is not None
        assert isinstance(result.calculated_at, datetime)

    def test_85_output_emissions_by_source(
        self,
        agent: Scope3EmissionsAgent,
    ):
        """GT-85: Output includes emissions breakdown by source."""
        input_data = Scope3Input(
            category=Scope3Category.CAT_1_PURCHASED_GOODS,
            reporting_year=2024,
            spend_data=[
                SpendData(category="steel", spend_usd=500_000, supplier_name="Supplier A"),
                SpendData(category="steel", spend_usd=300_000, supplier_name="Supplier B"),
            ],
        )
        result = agent.run(input_data)

        assert len(result.emissions_by_source) > 0
        assert "Supplier A" in result.emissions_by_source
        assert "Supplier B" in result.emissions_by_source


# =============================================================================
# Test 86-90: Input Validation
# =============================================================================


class TestInputValidation:
    """Tests for input validation."""

    def test_86_spend_data_model_validates_non_negative(self):
        """GT-86: SpendData requires non-negative spend."""
        with pytest.raises(ValueError):
            SpendData(category="steel", spend_usd=-100)

    def test_87_transport_data_validates_non_negative(self):
        """GT-87: TransportData requires non-negative values."""
        with pytest.raises(ValueError):
            TransportData(mode="road_truck", distance_km=-100, weight_tonnes=10)

    def test_88_travel_data_validates_non_negative(self):
        """GT-88: TravelData requires non-negative distance."""
        with pytest.raises(ValueError):
            TravelData(mode="air", distance_km=-500)

    def test_89_activity_data_validates_non_negative(self):
        """GT-89: ActivityData requires non-negative quantity."""
        with pytest.raises(ValueError):
            ActivityData(activity_type="landfill_mixed", quantity=-1000, unit="kg")

    def test_90_scope3_input_validates_year(self):
        """GT-90: Scope3Input validates reporting year >= 2020."""
        with pytest.raises(ValueError):
            Scope3Input(
                category=Scope3Category.CAT_1_PURCHASED_GOODS,
                reporting_year=2019,  # Invalid - before 2020
                spend_data=[SpendData(category="steel", spend_usd=100_000)],
            )


# =============================================================================
# Run Tests
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
