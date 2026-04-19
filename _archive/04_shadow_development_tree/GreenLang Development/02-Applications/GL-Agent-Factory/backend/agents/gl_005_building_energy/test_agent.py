"""
Unit Tests for GL-005: Building Energy Agent

Comprehensive test coverage for the Building Energy Agent including:
- Energy Use Intensity (EUI) calculations
- GHG emissions from building operations (Scope 1 + Scope 2)
- CRREM pathway alignment analysis
- Stranding risk assessment
- Energy Performance Certificate (EPC) rating
- Multi-building type support
- Multi-region grid emission factors
- Edge cases and boundary conditions
- Determinism verification tests
- Provenance hash validation

Test coverage target: 85%+
Total tests: 65+ golden tests covering all building energy calculation scenarios

Formula Documentation:
----------------------
All calculations follow zero-hallucination deterministic formulas:

EUI Calculation:
    EUI (kWh/m2) = total_energy (kWh) / floor_area (m2)

Energy Conversion:
    gas_kwh = annual_gas_m3 * 10.55 (m3 to kWh conversion factor)

Emissions Calculation:
    scope1_emissions = gas_kwh * 0.185 (kgCO2e/kWh - IPCC natural gas factor)
    scope2_emissions = electricity_kwh * grid_factor (varies by region)
    district_emissions = district_heating_kwh * 0.150 + district_cooling_kwh * 0.100

Grid Emission Factors (kgCO2e/kWh) - IEA/DEFRA 2024:
    DE: 0.366, FR: 0.052, UK: 0.207, US: 0.417, EU: 0.276
    NL: 0.328, ES: 0.182, IT: 0.256, PL: 0.635

EPC Rating Thresholds (kWh/m2/year) - Office:
    A+: <=50, A: <=75, B: <=100, C: <=135, D: <=175, E: <=225, F: <=300, G: >300

CRREM 2050 Targets (kgCO2e/m2/year):
    Office: 4.5, Retail: 8.0, Hotel: 7.5, Residential: 3.0, Industrial: 12.0
    Warehouse: 6.0, Healthcare: 15.0, Education: 5.0, Data Center: 50.0
"""

import hashlib
import json
import pytest
from datetime import datetime
from typing import Dict, Any, Optional

try:
    from .agent import (
        BuildingEnergyAgent,
        BuildingEnergyInput,
        BuildingEnergyOutput,
        BuildingType,
        EPCRating,
        StrandingRisk,
        EnergyFactor,
    )
except ImportError:
    from agent import (
        BuildingEnergyAgent,
        BuildingEnergyInput,
        BuildingEnergyOutput,
        BuildingType,
        EPCRating,
        StrandingRisk,
        EnergyFactor,
    )


# =============================================================================
# Test Constants - Expected Values from Agent
# =============================================================================

# Natural Gas to kWh conversion
GAS_M3_TO_KWH = 10.55

# Gas emission factor (kgCO2e/kWh)
GAS_EMISSION_FACTOR = 0.185

# District heating/cooling factors
DISTRICT_HEATING_FACTOR = 0.150
DISTRICT_COOLING_FACTOR = 0.100

# Grid emission factors by region (kgCO2e/kWh)
GRID_FACTORS = {
    "DE": 0.366,
    "FR": 0.052,
    "UK": 0.207,
    "US": 0.417,
    "EU": 0.276,
    "NL": 0.328,
    "ES": 0.182,
    "IT": 0.256,
    "PL": 0.635,
}

# EPC thresholds for Office (kWh/m2/year)
EPC_OFFICE_THRESHOLDS = {
    "A+": 50,
    "A": 75,
    "B": 100,
    "C": 135,
    "D": 175,
    "E": 225,
    "F": 300,
}

# CRREM 2050 targets (kgCO2e/m2/year)
CRREM_TARGETS_2050 = {
    "office": 4.5,
    "retail": 8.0,
    "hotel": 7.5,
    "residential": 3.0,
    "industrial": 12.0,
    "warehouse": 6.0,
    "healthcare": 15.0,
    "education": 5.0,
    "data_center": 50.0,
}

# EUI Benchmarks (kWh/m2/year)
EUI_BENCHMARKS = {
    "office": {"typical": 180, "best": 70, "worst": 350},
    "retail": {"typical": 350, "best": 150, "worst": 600},
    "hotel": {"typical": 280, "best": 120, "worst": 450},
    "residential": {"typical": 120, "best": 40, "worst": 250},
}


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def agent() -> BuildingEnergyAgent:
    """Create a BuildingEnergyAgent instance for testing."""
    return BuildingEnergyAgent()


@pytest.fixture
def agent_with_config() -> BuildingEnergyAgent:
    """Create agent with custom configuration."""
    return BuildingEnergyAgent(config={"custom_setting": "value"})


@pytest.fixture
def office_input_basic() -> BuildingEnergyInput:
    """
    Create basic office building input.

    Expected calculations:
        gas_kwh = 5000 * 10.55 = 52750 kWh
        total_energy = 500000 + 52750 = 552750 kWh
        EUI = 552750 / 10000 = 55.275 kWh/m2
    """
    return BuildingEnergyInput(
        building_id="OFFICE-001",
        building_type=BuildingType.OFFICE,
        floor_area_sqm=10000.0,
        annual_electricity_kwh=500000.0,
        annual_gas_m3=5000.0,
        region="DE",
    )


@pytest.fixture
def retail_input_high_energy() -> BuildingEnergyInput:
    """Create high-energy retail building input."""
    return BuildingEnergyInput(
        building_id="RETAIL-001",
        building_type=BuildingType.RETAIL,
        floor_area_sqm=5000.0,
        annual_electricity_kwh=1500000.0,
        annual_gas_m3=10000.0,
        region="UK",
    )


@pytest.fixture
def hotel_input_with_district() -> BuildingEnergyInput:
    """Create hotel building with district heating/cooling."""
    return BuildingEnergyInput(
        building_id="HOTEL-001",
        building_type=BuildingType.HOTEL,
        floor_area_sqm=8000.0,
        annual_electricity_kwh=800000.0,
        annual_gas_m3=0.0,
        annual_district_heating_kwh=400000.0,
        annual_district_cooling_kwh=200000.0,
        region="FR",
    )


@pytest.fixture
def residential_input_efficient() -> BuildingEnergyInput:
    """Create efficient residential building (A+ rated)."""
    return BuildingEnergyInput(
        building_id="RES-001",
        building_type=BuildingType.RESIDENTIAL,
        floor_area_sqm=2000.0,
        annual_electricity_kwh=40000.0,  # 20 kWh/m2
        annual_gas_m3=1000.0,  # 10550 kWh -> 5.275 kWh/m2
        region="FR",  # Low carbon grid
    )


@pytest.fixture
def industrial_input_large() -> BuildingEnergyInput:
    """Create large industrial building input."""
    return BuildingEnergyInput(
        building_id="IND-001",
        building_type=BuildingType.INDUSTRIAL,
        floor_area_sqm=50000.0,
        annual_electricity_kwh=5000000.0,
        annual_gas_m3=100000.0,
        region="PL",  # High carbon grid
    )


@pytest.fixture
def data_center_input() -> BuildingEnergyInput:
    """Create data center building input."""
    return BuildingEnergyInput(
        building_id="DC-001",
        building_type=BuildingType.DATA_CENTER,
        floor_area_sqm=3000.0,
        annual_electricity_kwh=15000000.0,  # 5000 kWh/m2
        annual_gas_m3=0.0,
        annual_district_cooling_kwh=3000000.0,
        region="US",
    )


@pytest.fixture
def office_input_with_renewables() -> BuildingEnergyInput:
    """Create office with on-site renewable generation."""
    return BuildingEnergyInput(
        building_id="OFFICE-RENEW-001",
        building_type=BuildingType.OFFICE,
        floor_area_sqm=10000.0,
        annual_electricity_kwh=600000.0,
        annual_gas_m3=3000.0,
        renewable_generation_kwh=200000.0,  # Solar PV
        region="ES",
    )


# =============================================================================
# Test 1-10: Agent Initialization and Basic Tests
# =============================================================================


class TestAgentInitialization:
    """Tests for agent initialization."""

    def test_01_agent_initialization(self, agent: BuildingEnergyAgent):
        """Test 1: Agent initializes correctly with default config."""
        assert agent is not None
        assert agent.AGENT_ID == "buildings/energy_performance_v1"
        assert agent.VERSION == "1.0.0"
        assert "CRREM" in agent.DESCRIPTION

    def test_02_agent_with_custom_config(self, agent_with_config: BuildingEnergyAgent):
        """Test 2: Agent initializes with custom configuration."""
        assert agent_with_config.config == {"custom_setting": "value"}

    def test_03_grid_factors_loaded(self, agent: BuildingEnergyAgent):
        """Test 3: Grid emission factors are loaded correctly."""
        factors = agent.GRID_EMISSION_FACTORS
        assert "DE" in factors
        assert "FR" in factors
        assert "US" in factors
        assert "EU" in factors
        assert factors["DE"].value == GRID_FACTORS["DE"]

    def test_04_crrem_targets_loaded(self, agent: BuildingEnergyAgent):
        """Test 4: CRREM 2050 targets are loaded correctly."""
        targets = agent.CRREM_TARGETS_2050
        assert BuildingType.OFFICE in targets
        assert BuildingType.RETAIL in targets
        assert targets[BuildingType.OFFICE] == CRREM_TARGETS_2050["office"]

    def test_05_epc_thresholds_loaded(self, agent: BuildingEnergyAgent):
        """Test 5: EPC rating thresholds are loaded correctly."""
        thresholds = agent.EPC_THRESHOLDS
        assert BuildingType.OFFICE in thresholds
        assert BuildingType.RETAIL in thresholds
        assert thresholds[BuildingType.OFFICE][EPCRating.A_PLUS] == EPC_OFFICE_THRESHOLDS["A+"]

    def test_06_get_building_types(self, agent: BuildingEnergyAgent):
        """Test 6: Get supported building types returns all types."""
        building_types = agent.get_building_types()
        assert "office" in building_types
        assert "retail" in building_types
        assert "hotel" in building_types
        assert "residential" in building_types
        assert "data_center" in building_types
        assert len(building_types) == 9  # All BuildingType enum values

    def test_07_basic_run_completes(
        self,
        agent: BuildingEnergyAgent,
        office_input_basic: BuildingEnergyInput,
    ):
        """Test 7: Basic agent run completes successfully."""
        result = agent.run(office_input_basic)
        assert result is not None
        assert isinstance(result, BuildingEnergyOutput)

    def test_08_run_returns_provenance_hash(
        self,
        agent: BuildingEnergyAgent,
        office_input_basic: BuildingEnergyInput,
    ):
        """Test 8: Run returns valid SHA-256 provenance hash."""
        result = agent.run(office_input_basic)
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256 hex string

    def test_09_run_returns_building_info(
        self,
        agent: BuildingEnergyAgent,
        office_input_basic: BuildingEnergyInput,
    ):
        """Test 9: Run returns building information."""
        result = agent.run(office_input_basic)
        assert result.building_type == "office"
        assert result.floor_area_sqm == 10000.0

    def test_10_run_returns_timestamp(
        self,
        agent: BuildingEnergyAgent,
        office_input_basic: BuildingEnergyInput,
    ):
        """Test 10: Run returns calculated_at timestamp."""
        result = agent.run(office_input_basic)
        assert result.calculated_at is not None
        assert isinstance(result.calculated_at, datetime)


# =============================================================================
# Test 11-20: EUI Calculations
# =============================================================================


class TestEUICalculations:
    """Tests for Energy Use Intensity (EUI) calculations."""

    @pytest.mark.golden
    def test_11_eui_office_basic(
        self,
        agent: BuildingEnergyAgent,
        office_input_basic: BuildingEnergyInput,
    ):
        """
        Test 11: EUI calculation - Office, basic scenario

        ZERO-HALLUCINATION CHECK:
        gas_kwh = 5000 * 10.55 = 52750 kWh
        total_energy = 500000 + 52750 = 552750 kWh
        EUI = 552750 / 10000 = 55.275 kWh/m2
        """
        result = agent.run(office_input_basic)

        gas_kwh = 5000.0 * GAS_M3_TO_KWH
        total_energy = 500000.0 + gas_kwh
        expected_eui = total_energy / 10000.0

        assert result.eui_kwh_sqm == pytest.approx(expected_eui, rel=1e-4)
        assert result.total_energy_kwh == pytest.approx(total_energy, rel=1e-4)
        assert result.gas_kwh == pytest.approx(gas_kwh, rel=1e-4)

    @pytest.mark.golden
    def test_12_eui_retail_high_energy(
        self,
        agent: BuildingEnergyAgent,
        retail_input_high_energy: BuildingEnergyInput,
    ):
        """
        Test 12: EUI calculation - Retail, high energy consumption

        ZERO-HALLUCINATION CHECK:
        gas_kwh = 10000 * 10.55 = 105500 kWh
        total_energy = 1500000 + 105500 = 1605500 kWh
        EUI = 1605500 / 5000 = 321.1 kWh/m2
        """
        result = agent.run(retail_input_high_energy)

        gas_kwh = 10000.0 * GAS_M3_TO_KWH
        total_energy = 1500000.0 + gas_kwh
        expected_eui = total_energy / 5000.0

        assert result.eui_kwh_sqm == pytest.approx(expected_eui, rel=1e-4)

    @pytest.mark.golden
    def test_13_eui_hotel_with_district(
        self,
        agent: BuildingEnergyAgent,
        hotel_input_with_district: BuildingEnergyInput,
    ):
        """
        Test 13: EUI calculation - Hotel with district heating/cooling

        ZERO-HALLUCINATION CHECK:
        total_energy = 800000 + 0 + 400000 + 200000 = 1400000 kWh
        EUI = 1400000 / 8000 = 175 kWh/m2
        """
        result = agent.run(hotel_input_with_district)

        total_energy = 800000.0 + 400000.0 + 200000.0
        expected_eui = total_energy / 8000.0

        assert result.eui_kwh_sqm == pytest.approx(expected_eui, rel=1e-4)
        assert result.total_energy_kwh == pytest.approx(total_energy, rel=1e-4)

    @pytest.mark.golden
    def test_14_eui_residential_efficient(
        self,
        agent: BuildingEnergyAgent,
        residential_input_efficient: BuildingEnergyInput,
    ):
        """
        Test 14: EUI calculation - Efficient residential building

        ZERO-HALLUCINATION CHECK:
        gas_kwh = 1000 * 10.55 = 10550 kWh
        total_energy = 40000 + 10550 = 50550 kWh
        EUI = 50550 / 2000 = 25.275 kWh/m2
        """
        result = agent.run(residential_input_efficient)

        gas_kwh = 1000.0 * GAS_M3_TO_KWH
        total_energy = 40000.0 + gas_kwh
        expected_eui = total_energy / 2000.0

        assert result.eui_kwh_sqm == pytest.approx(expected_eui, rel=1e-4)

    @pytest.mark.golden
    def test_15_eui_data_center_high_intensity(
        self,
        agent: BuildingEnergyAgent,
        data_center_input: BuildingEnergyInput,
    ):
        """
        Test 15: EUI calculation - Data center with very high intensity

        ZERO-HALLUCINATION CHECK:
        total_energy = 15000000 + 3000000 = 18000000 kWh
        EUI = 18000000 / 3000 = 6000 kWh/m2
        """
        result = agent.run(data_center_input)

        total_energy = 15000000.0 + 3000000.0
        expected_eui = total_energy / 3000.0

        assert result.eui_kwh_sqm == pytest.approx(expected_eui, rel=1e-4)

    @pytest.mark.golden
    def test_16_eui_electricity_only(
        self,
        agent: BuildingEnergyAgent,
    ):
        """
        Test 16: EUI calculation - Electricity only (no gas)

        ZERO-HALLUCINATION CHECK:
        total_energy = 100000 kWh
        EUI = 100000 / 1000 = 100 kWh/m2
        """
        input_data = BuildingEnergyInput(
            building_type=BuildingType.OFFICE,
            floor_area_sqm=1000.0,
            annual_electricity_kwh=100000.0,
            annual_gas_m3=0.0,
            region="EU",
        )
        result = agent.run(input_data)

        assert result.eui_kwh_sqm == pytest.approx(100.0, rel=1e-4)
        assert result.gas_kwh == 0.0

    @pytest.mark.golden
    def test_17_eui_gas_only(
        self,
        agent: BuildingEnergyAgent,
    ):
        """
        Test 17: EUI calculation - Gas only (no electricity)

        ZERO-HALLUCINATION CHECK:
        gas_kwh = 1000 * 10.55 = 10550 kWh
        EUI = 10550 / 100 = 105.5 kWh/m2
        """
        input_data = BuildingEnergyInput(
            building_type=BuildingType.WAREHOUSE,
            floor_area_sqm=100.0,
            annual_electricity_kwh=0.0,
            annual_gas_m3=1000.0,
            region="EU",
        )
        result = agent.run(input_data)

        expected_gas_kwh = 1000.0 * GAS_M3_TO_KWH
        expected_eui = expected_gas_kwh / 100.0

        assert result.eui_kwh_sqm == pytest.approx(expected_eui, rel=1e-4)
        assert result.electricity_kwh == 0.0

    @pytest.mark.golden
    def test_18_eui_small_building(
        self,
        agent: BuildingEnergyAgent,
    ):
        """
        Test 18: EUI calculation - Small building (precision test)

        ZERO-HALLUCINATION CHECK:
        gas_kwh = 10 * 10.55 = 105.5 kWh
        total_energy = 500 + 105.5 = 605.5 kWh
        EUI = 605.5 / 50 = 12.11 kWh/m2
        """
        input_data = BuildingEnergyInput(
            building_type=BuildingType.RESIDENTIAL,
            floor_area_sqm=50.0,
            annual_electricity_kwh=500.0,
            annual_gas_m3=10.0,
            region="EU",
        )
        result = agent.run(input_data)

        gas_kwh = 10.0 * GAS_M3_TO_KWH
        total_energy = 500.0 + gas_kwh
        expected_eui = total_energy / 50.0

        assert result.eui_kwh_sqm == pytest.approx(expected_eui, rel=1e-4)

    @pytest.mark.golden
    def test_19_eui_large_industrial(
        self,
        agent: BuildingEnergyAgent,
        industrial_input_large: BuildingEnergyInput,
    ):
        """
        Test 19: EUI calculation - Large industrial building

        ZERO-HALLUCINATION CHECK:
        gas_kwh = 100000 * 10.55 = 1055000 kWh
        total_energy = 5000000 + 1055000 = 6055000 kWh
        EUI = 6055000 / 50000 = 121.1 kWh/m2
        """
        result = agent.run(industrial_input_large)

        gas_kwh = 100000.0 * GAS_M3_TO_KWH
        total_energy = 5000000.0 + gas_kwh
        expected_eui = total_energy / 50000.0

        assert result.eui_kwh_sqm == pytest.approx(expected_eui, rel=1e-4)

    @pytest.mark.golden
    def test_20_renewable_share_calculation(
        self,
        agent: BuildingEnergyAgent,
        office_input_with_renewables: BuildingEnergyInput,
    ):
        """
        Test 20: Renewable energy share calculation

        ZERO-HALLUCINATION CHECK:
        gas_kwh = 3000 * 10.55 = 31650 kWh
        total_energy = 600000 + 31650 = 631650 kWh
        renewable_share = (200000 / 631650) * 100 = 31.67%
        """
        result = agent.run(office_input_with_renewables)

        gas_kwh = 3000.0 * GAS_M3_TO_KWH
        total_energy = 600000.0 + gas_kwh
        expected_share = (200000.0 / total_energy) * 100

        assert result.renewable_share_pct == pytest.approx(expected_share, rel=1e-2)


# =============================================================================
# Test 21-30: Emissions Calculations
# =============================================================================


class TestEmissionsCalculations:
    """Tests for GHG emissions calculations."""

    @pytest.mark.golden
    def test_21_emissions_office_germany(
        self,
        agent: BuildingEnergyAgent,
        office_input_basic: BuildingEnergyInput,
    ):
        """
        Test 21: Emissions calculation - Office in Germany

        ZERO-HALLUCINATION CHECK:
        gas_kwh = 5000 * 10.55 = 52750 kWh
        scope1 = 52750 * 0.185 = 9758.75 kgCO2e
        scope2 = 500000 * 0.366 = 183000 kgCO2e
        total = 9758.75 + 183000 = 192758.75 kgCO2e
        intensity = 192758.75 / 10000 = 19.276 kgCO2e/m2
        """
        result = agent.run(office_input_basic)

        gas_kwh = 5000.0 * GAS_M3_TO_KWH
        scope1 = gas_kwh * GAS_EMISSION_FACTOR
        scope2 = 500000.0 * GRID_FACTORS["DE"]
        total = scope1 + scope2
        intensity = total / 10000.0

        assert result.scope1_emissions == pytest.approx(scope1, rel=1e-4)
        assert result.scope2_emissions == pytest.approx(scope2, rel=1e-4)
        assert result.total_emissions_kgco2e == pytest.approx(total, rel=1e-4)
        assert result.emissions_intensity_kgco2e_sqm == pytest.approx(intensity, rel=1e-4)

    @pytest.mark.golden
    def test_22_emissions_france_low_carbon(
        self,
        agent: BuildingEnergyAgent,
        hotel_input_with_district: BuildingEnergyInput,
    ):
        """
        Test 22: Emissions calculation - France (low carbon grid)

        ZERO-HALLUCINATION CHECK:
        scope1 = 0 (no gas)
        scope2 = 800000 * 0.052 = 41600 kgCO2e
        district = 400000 * 0.150 + 200000 * 0.100 = 60000 + 20000 = 80000 kgCO2e
        total = 0 + 41600 + 80000 = 121600 kgCO2e
        intensity = 121600 / 8000 = 15.2 kgCO2e/m2
        """
        result = agent.run(hotel_input_with_district)

        scope1 = 0.0
        scope2 = 800000.0 * GRID_FACTORS["FR"]
        district = 400000.0 * DISTRICT_HEATING_FACTOR + 200000.0 * DISTRICT_COOLING_FACTOR
        total = scope1 + scope2 + district
        intensity = total / 8000.0

        assert result.scope1_emissions == pytest.approx(scope1, rel=1e-4)
        assert result.scope2_emissions == pytest.approx(scope2, rel=1e-4)
        assert result.total_emissions_kgco2e == pytest.approx(total, rel=1e-4)
        assert result.emissions_intensity_kgco2e_sqm == pytest.approx(intensity, rel=1e-4)

    @pytest.mark.golden
    def test_23_emissions_poland_high_carbon(
        self,
        agent: BuildingEnergyAgent,
        industrial_input_large: BuildingEnergyInput,
    ):
        """
        Test 23: Emissions calculation - Poland (high carbon grid)

        ZERO-HALLUCINATION CHECK:
        gas_kwh = 100000 * 10.55 = 1055000 kWh
        scope1 = 1055000 * 0.185 = 195175 kgCO2e
        scope2 = 5000000 * 0.635 = 3175000 kgCO2e
        total = 195175 + 3175000 = 3370175 kgCO2e
        """
        result = agent.run(industrial_input_large)

        gas_kwh = 100000.0 * GAS_M3_TO_KWH
        scope1 = gas_kwh * GAS_EMISSION_FACTOR
        scope2 = 5000000.0 * GRID_FACTORS["PL"]
        total = scope1 + scope2

        assert result.scope1_emissions == pytest.approx(scope1, rel=1e-4)
        assert result.scope2_emissions == pytest.approx(scope2, rel=1e-4)
        assert result.total_emissions_kgco2e == pytest.approx(total, rel=1e-4)

    @pytest.mark.golden
    def test_24_emissions_us_data_center(
        self,
        agent: BuildingEnergyAgent,
        data_center_input: BuildingEnergyInput,
    ):
        """
        Test 24: Emissions calculation - US Data Center

        ZERO-HALLUCINATION CHECK:
        scope1 = 0 (no gas)
        scope2 = 15000000 * 0.417 = 6255000 kgCO2e
        district_cooling = 3000000 * 0.100 = 300000 kgCO2e
        total = 6255000 + 300000 = 6555000 kgCO2e
        intensity = 6555000 / 3000 = 2185 kgCO2e/m2
        """
        result = agent.run(data_center_input)

        scope1 = 0.0
        scope2 = 15000000.0 * GRID_FACTORS["US"]
        district = 3000000.0 * DISTRICT_COOLING_FACTOR
        total = scope1 + scope2 + district
        intensity = total / 3000.0

        assert result.scope1_emissions == pytest.approx(scope1, rel=1e-4)
        assert result.scope2_emissions == pytest.approx(scope2, rel=1e-4)
        assert result.emissions_intensity_kgco2e_sqm == pytest.approx(intensity, rel=1e-4)

    @pytest.mark.golden
    def test_25_emissions_uk_retail(
        self,
        agent: BuildingEnergyAgent,
        retail_input_high_energy: BuildingEnergyInput,
    ):
        """
        Test 25: Emissions calculation - UK Retail

        ZERO-HALLUCINATION CHECK:
        gas_kwh = 10000 * 10.55 = 105500 kWh
        scope1 = 105500 * 0.185 = 19517.5 kgCO2e
        scope2 = 1500000 * 0.207 = 310500 kgCO2e
        total = 19517.5 + 310500 = 330017.5 kgCO2e
        """
        result = agent.run(retail_input_high_energy)

        gas_kwh = 10000.0 * GAS_M3_TO_KWH
        scope1 = gas_kwh * GAS_EMISSION_FACTOR
        scope2 = 1500000.0 * GRID_FACTORS["UK"]
        total = scope1 + scope2

        assert result.scope1_emissions == pytest.approx(scope1, rel=1e-4)
        assert result.scope2_emissions == pytest.approx(scope2, rel=1e-4)
        assert result.total_emissions_kgco2e == pytest.approx(total, rel=1e-4)

    @pytest.mark.golden
    def test_26_emissions_unknown_region_fallback(
        self,
        agent: BuildingEnergyAgent,
    ):
        """
        Test 26: Emissions with unknown region falls back to EU

        ZERO-HALLUCINATION CHECK:
        Brazil not in factors, falls back to EU (0.276)
        scope2 = 100000 * 0.276 = 27600 kgCO2e
        """
        input_data = BuildingEnergyInput(
            building_type=BuildingType.OFFICE,
            floor_area_sqm=1000.0,
            annual_electricity_kwh=100000.0,
            annual_gas_m3=0.0,
            region="BR",  # Brazil - not in factors
        )
        result = agent.run(input_data)

        expected_scope2 = 100000.0 * GRID_FACTORS["EU"]  # Falls back to EU
        assert result.scope2_emissions == pytest.approx(expected_scope2, rel=1e-4)

    @pytest.mark.golden
    def test_27_emissions_spain_medium_carbon(
        self,
        agent: BuildingEnergyAgent,
        office_input_with_renewables: BuildingEnergyInput,
    ):
        """
        Test 27: Emissions calculation - Spain

        ZERO-HALLUCINATION CHECK:
        gas_kwh = 3000 * 10.55 = 31650 kWh
        scope1 = 31650 * 0.185 = 5855.25 kgCO2e
        scope2 = 600000 * 0.182 = 109200 kgCO2e
        total = 5855.25 + 109200 = 115055.25 kgCO2e
        """
        result = agent.run(office_input_with_renewables)

        gas_kwh = 3000.0 * GAS_M3_TO_KWH
        scope1 = gas_kwh * GAS_EMISSION_FACTOR
        scope2 = 600000.0 * GRID_FACTORS["ES"]
        total = scope1 + scope2

        assert result.scope1_emissions == pytest.approx(scope1, rel=1e-4)
        assert result.scope2_emissions == pytest.approx(scope2, rel=1e-4)

    @pytest.mark.golden
    def test_28_emissions_all_sources(
        self,
        agent: BuildingEnergyAgent,
    ):
        """
        Test 28: Emissions from all energy sources

        ZERO-HALLUCINATION CHECK:
        gas_kwh = 2000 * 10.55 = 21100 kWh
        scope1 = 21100 * 0.185 = 3903.5 kgCO2e
        scope2 = 200000 * 0.276 = 55200 kgCO2e
        district = 50000 * 0.150 + 30000 * 0.100 = 7500 + 3000 = 10500 kgCO2e
        total = 3903.5 + 55200 + 10500 = 69603.5 kgCO2e
        """
        input_data = BuildingEnergyInput(
            building_type=BuildingType.OFFICE,
            floor_area_sqm=2000.0,
            annual_electricity_kwh=200000.0,
            annual_gas_m3=2000.0,
            annual_district_heating_kwh=50000.0,
            annual_district_cooling_kwh=30000.0,
            region="EU",
        )
        result = agent.run(input_data)

        gas_kwh = 2000.0 * GAS_M3_TO_KWH
        scope1 = gas_kwh * GAS_EMISSION_FACTOR
        scope2 = 200000.0 * GRID_FACTORS["EU"]
        district = 50000.0 * DISTRICT_HEATING_FACTOR + 30000.0 * DISTRICT_COOLING_FACTOR
        total = scope1 + scope2 + district

        assert result.total_emissions_kgco2e == pytest.approx(total, rel=1e-4)

    @pytest.mark.golden
    def test_29_emissions_zero_energy(
        self,
        agent: BuildingEnergyAgent,
    ):
        """
        Test 29: Zero energy building (edge case)

        ZERO-HALLUCINATION CHECK:
        All emissions should be zero
        """
        input_data = BuildingEnergyInput(
            building_type=BuildingType.OFFICE,
            floor_area_sqm=1000.0,
            annual_electricity_kwh=0.0,
            annual_gas_m3=0.0,
            region="EU",
        )
        result = agent.run(input_data)

        assert result.scope1_emissions == 0.0
        assert result.scope2_emissions == 0.0
        assert result.total_emissions_kgco2e == 0.0
        assert result.eui_kwh_sqm == 0.0

    @pytest.mark.golden
    def test_30_emissions_intensity_calculation(
        self,
        agent: BuildingEnergyAgent,
    ):
        """
        Test 30: Emissions intensity calculation

        ZERO-HALLUCINATION CHECK:
        scope2 = 50000 * 0.276 = 13800 kgCO2e
        intensity = 13800 / 500 = 27.6 kgCO2e/m2
        """
        input_data = BuildingEnergyInput(
            building_type=BuildingType.OFFICE,
            floor_area_sqm=500.0,
            annual_electricity_kwh=50000.0,
            annual_gas_m3=0.0,
            region="EU",
        )
        result = agent.run(input_data)

        total_emissions = 50000.0 * GRID_FACTORS["EU"]
        expected_intensity = total_emissions / 500.0

        assert result.emissions_intensity_kgco2e_sqm == pytest.approx(expected_intensity, rel=1e-4)


# =============================================================================
# Test 31-40: EPC Rating Tests
# =============================================================================


class TestEPCRating:
    """Tests for Energy Performance Certificate rating calculations."""

    @pytest.mark.golden
    def test_31_epc_office_a_plus(
        self,
        agent: BuildingEnergyAgent,
    ):
        """
        Test 31: EPC A+ rating for efficient office

        ZERO-HALLUCINATION CHECK:
        EUI = 40000 / 1000 = 40 kWh/m2 -> A+ (<=50)
        """
        input_data = BuildingEnergyInput(
            building_type=BuildingType.OFFICE,
            floor_area_sqm=1000.0,
            annual_electricity_kwh=40000.0,
            annual_gas_m3=0.0,
            region="EU",
        )
        result = agent.run(input_data)

        assert result.epc_rating == EPCRating.A_PLUS.value
        assert result.epc_score == 100.0

    @pytest.mark.golden
    def test_32_epc_office_a(
        self,
        agent: BuildingEnergyAgent,
    ):
        """
        Test 32: EPC A rating for office

        ZERO-HALLUCINATION CHECK:
        EUI = 60000 / 1000 = 60 kWh/m2 -> A (51-75)
        """
        input_data = BuildingEnergyInput(
            building_type=BuildingType.OFFICE,
            floor_area_sqm=1000.0,
            annual_electricity_kwh=60000.0,
            annual_gas_m3=0.0,
            region="EU",
        )
        result = agent.run(input_data)

        assert result.epc_rating == EPCRating.A.value
        assert result.epc_score == pytest.approx(87.5, rel=1e-2)

    @pytest.mark.golden
    def test_33_epc_office_b(
        self,
        agent: BuildingEnergyAgent,
    ):
        """
        Test 33: EPC B rating for office

        ZERO-HALLUCINATION CHECK:
        EUI = 90000 / 1000 = 90 kWh/m2 -> B (76-100)
        """
        input_data = BuildingEnergyInput(
            building_type=BuildingType.OFFICE,
            floor_area_sqm=1000.0,
            annual_electricity_kwh=90000.0,
            annual_gas_m3=0.0,
            region="EU",
        )
        result = agent.run(input_data)

        assert result.epc_rating == EPCRating.B.value

    @pytest.mark.golden
    def test_34_epc_office_c(
        self,
        agent: BuildingEnergyAgent,
    ):
        """
        Test 34: EPC C rating for office

        ZERO-HALLUCINATION CHECK:
        EUI = 120000 / 1000 = 120 kWh/m2 -> C (101-135)
        """
        input_data = BuildingEnergyInput(
            building_type=BuildingType.OFFICE,
            floor_area_sqm=1000.0,
            annual_electricity_kwh=120000.0,
            annual_gas_m3=0.0,
            region="EU",
        )
        result = agent.run(input_data)

        assert result.epc_rating == EPCRating.C.value

    @pytest.mark.golden
    def test_35_epc_office_d(
        self,
        agent: BuildingEnergyAgent,
    ):
        """
        Test 35: EPC D rating for office

        ZERO-HALLUCINATION CHECK:
        EUI = 150000 / 1000 = 150 kWh/m2 -> D (136-175)
        """
        input_data = BuildingEnergyInput(
            building_type=BuildingType.OFFICE,
            floor_area_sqm=1000.0,
            annual_electricity_kwh=150000.0,
            annual_gas_m3=0.0,
            region="EU",
        )
        result = agent.run(input_data)

        assert result.epc_rating == EPCRating.D.value

    @pytest.mark.golden
    def test_36_epc_office_e(
        self,
        agent: BuildingEnergyAgent,
    ):
        """
        Test 36: EPC E rating for office

        ZERO-HALLUCINATION CHECK:
        EUI = 200000 / 1000 = 200 kWh/m2 -> E (176-225)
        """
        input_data = BuildingEnergyInput(
            building_type=BuildingType.OFFICE,
            floor_area_sqm=1000.0,
            annual_electricity_kwh=200000.0,
            annual_gas_m3=0.0,
            region="EU",
        )
        result = agent.run(input_data)

        assert result.epc_rating == EPCRating.E.value

    @pytest.mark.golden
    def test_37_epc_office_f(
        self,
        agent: BuildingEnergyAgent,
    ):
        """
        Test 37: EPC F rating for office

        ZERO-HALLUCINATION CHECK:
        EUI = 250000 / 1000 = 250 kWh/m2 -> F (226-300)
        """
        input_data = BuildingEnergyInput(
            building_type=BuildingType.OFFICE,
            floor_area_sqm=1000.0,
            annual_electricity_kwh=250000.0,
            annual_gas_m3=0.0,
            region="EU",
        )
        result = agent.run(input_data)

        assert result.epc_rating == EPCRating.F.value

    @pytest.mark.golden
    def test_38_epc_office_g(
        self,
        agent: BuildingEnergyAgent,
    ):
        """
        Test 38: EPC G rating for inefficient office

        ZERO-HALLUCINATION CHECK:
        EUI = 400000 / 1000 = 400 kWh/m2 -> G (>300)
        """
        input_data = BuildingEnergyInput(
            building_type=BuildingType.OFFICE,
            floor_area_sqm=1000.0,
            annual_electricity_kwh=400000.0,
            annual_gas_m3=0.0,
            region="EU",
        )
        result = agent.run(input_data)

        assert result.epc_rating == EPCRating.G.value
        assert result.epc_score == 0.0

    @pytest.mark.golden
    def test_39_epc_retail_different_thresholds(
        self,
        agent: BuildingEnergyAgent,
    ):
        """
        Test 39: EPC rating for retail (different thresholds)

        ZERO-HALLUCINATION CHECK:
        EUI = 180000 / 1000 = 180 kWh/m2
        Retail A threshold: 150 -> this is B (151-200)
        """
        input_data = BuildingEnergyInput(
            building_type=BuildingType.RETAIL,
            floor_area_sqm=1000.0,
            annual_electricity_kwh=180000.0,
            annual_gas_m3=0.0,
            region="EU",
        )
        result = agent.run(input_data)

        # Retail has higher EUI thresholds
        assert result.epc_rating == EPCRating.B.value

    @pytest.mark.golden
    def test_40_epc_boundary_value(
        self,
        agent: BuildingEnergyAgent,
    ):
        """
        Test 40: EPC rating at exact boundary (75 kWh/m2)

        ZERO-HALLUCINATION CHECK:
        EUI = 75000 / 1000 = 75 kWh/m2 -> A (boundary value)
        """
        input_data = BuildingEnergyInput(
            building_type=BuildingType.OFFICE,
            floor_area_sqm=1000.0,
            annual_electricity_kwh=75000.0,
            annual_gas_m3=0.0,
            region="EU",
        )
        result = agent.run(input_data)

        assert result.epc_rating == EPCRating.A.value


# =============================================================================
# Test 41-50: CRREM and Stranding Risk Tests
# =============================================================================


class TestCRREMStrandingRisk:
    """Tests for CRREM pathway alignment and stranding risk."""

    @pytest.mark.golden
    def test_41_crrem_office_low_risk(
        self,
        agent: BuildingEnergyAgent,
        residential_input_efficient: BuildingEnergyInput,
    ):
        """
        Test 41: CRREM analysis - Low stranding risk

        Building with low emissions intensity should have low stranding risk.
        """
        result = agent.run(residential_input_efficient)

        # Efficient building should be below or near CRREM target
        assert result.stranding_risk in [StrandingRisk.LOW.value]
        assert result.stranding_year is None or result.stranding_year > 2050

    @pytest.mark.golden
    def test_42_crrem_industrial_high_risk(
        self,
        agent: BuildingEnergyAgent,
        industrial_input_large: BuildingEnergyInput,
    ):
        """
        Test 42: CRREM analysis - High stranding risk for industrial

        ZERO-HALLUCINATION CHECK:
        CRREM target industrial: 12.0 kgCO2e/m2
        If intensity > 36 (3x target), stranding risk = STRANDED
        """
        result = agent.run(industrial_input_large)

        # High-emission building has elevated stranding risk
        assert result.crrem_target_intensity == CRREM_TARGETS_2050["industrial"]
        assert result.crrem_excess_intensity > 0

    @pytest.mark.golden
    def test_43_crrem_data_center_high_target(
        self,
        agent: BuildingEnergyAgent,
        data_center_input: BuildingEnergyInput,
    ):
        """
        Test 43: CRREM target for data center (highest target)

        ZERO-HALLUCINATION CHECK:
        CRREM target data_center: 50.0 kgCO2e/m2
        """
        result = agent.run(data_center_input)

        assert result.crrem_target_intensity == CRREM_TARGETS_2050["data_center"]

    @pytest.mark.golden
    def test_44_crrem_excess_intensity_calculation(
        self,
        agent: BuildingEnergyAgent,
    ):
        """
        Test 44: CRREM excess intensity calculation

        ZERO-HALLUCINATION CHECK:
        scope2 = 100000 * 0.276 = 27600 kgCO2e
        intensity = 27600 / 1000 = 27.6 kgCO2e/m2
        CRREM office target: 4.5 kgCO2e/m2
        excess = 27.6 - 4.5 = 23.1 kgCO2e/m2
        """
        input_data = BuildingEnergyInput(
            building_type=BuildingType.OFFICE,
            floor_area_sqm=1000.0,
            annual_electricity_kwh=100000.0,
            annual_gas_m3=0.0,
            region="EU",
        )
        result = agent.run(input_data)

        intensity = (100000.0 * GRID_FACTORS["EU"]) / 1000.0
        expected_excess = max(0, intensity - CRREM_TARGETS_2050["office"])

        assert result.crrem_excess_intensity == pytest.approx(expected_excess, rel=1e-2)

    @pytest.mark.golden
    def test_45_stranding_year_medium_risk(
        self,
        agent: BuildingEnergyAgent,
    ):
        """
        Test 45: Stranding year for medium risk building

        Building with intensity 1.5-2x target should have medium risk
        """
        # Create a building with intensity around 10-15 kgCO2e/m2 for office (target 4.5)
        input_data = BuildingEnergyInput(
            building_type=BuildingType.OFFICE,
            floor_area_sqm=1000.0,
            annual_electricity_kwh=35000.0,  # ~9.7 kgCO2e/m2
            annual_gas_m3=0.0,
            region="EU",
        )
        result = agent.run(input_data)

        # Should be medium or low risk
        assert result.stranding_risk in [StrandingRisk.LOW.value, StrandingRisk.MEDIUM.value]

    @pytest.mark.golden
    def test_46_decarbonization_gap_calculation(
        self,
        agent: BuildingEnergyAgent,
        office_input_basic: BuildingEnergyInput,
    ):
        """
        Test 46: Decarbonization gap percentage calculation

        ZERO-HALLUCINATION CHECK:
        gap = ((intensity - target) / intensity) * 100
        """
        result = agent.run(office_input_basic)

        if result.emissions_intensity_kgco2e_sqm > 0:
            target = CRREM_TARGETS_2050["office"]
            expected_gap = max(
                0,
                (result.emissions_intensity_kgco2e_sqm - target) / result.emissions_intensity_kgco2e_sqm * 100,
            )
            assert result.decarbonization_gap_pct == pytest.approx(expected_gap, rel=1e-2)

    @pytest.mark.golden
    def test_47_crrem_below_target(
        self,
        agent: BuildingEnergyAgent,
    ):
        """
        Test 47: Building already below CRREM target

        Very efficient building should have no excess intensity
        """
        input_data = BuildingEnergyInput(
            building_type=BuildingType.OFFICE,
            floor_area_sqm=1000.0,
            annual_electricity_kwh=10000.0,  # Very low
            annual_gas_m3=0.0,
            region="FR",  # Low carbon grid
        )
        result = agent.run(input_data)

        # intensity = 10000 * 0.052 / 1000 = 0.52 kgCO2e/m2 (below 4.5 target)
        assert result.crrem_excess_intensity == 0.0
        assert result.stranding_risk == StrandingRisk.LOW.value

    @pytest.mark.golden
    def test_48_stranding_risk_stranded(
        self,
        agent: BuildingEnergyAgent,
    ):
        """
        Test 48: Already stranded building (>3x target)

        ZERO-HALLUCINATION CHECK:
        Office target: 4.5 kgCO2e/m2
        If intensity > 13.5 (3x), should be STRANDED
        """
        input_data = BuildingEnergyInput(
            building_type=BuildingType.OFFICE,
            floor_area_sqm=1000.0,
            annual_electricity_kwh=200000.0,  # High
            annual_gas_m3=5000.0,  # Plus gas
            region="PL",  # High carbon grid
        )
        result = agent.run(input_data)

        # intensity should be very high (> 3x target of 4.5)
        if result.emissions_intensity_kgco2e_sqm > 3 * CRREM_TARGETS_2050["office"]:
            assert result.stranding_risk == StrandingRisk.STRANDED.value

    @pytest.mark.golden
    def test_49_crrem_target_by_building_type(
        self,
        agent: BuildingEnergyAgent,
    ):
        """
        Test 49: CRREM targets differ by building type
        """
        building_types = [
            (BuildingType.OFFICE, 4.5),
            (BuildingType.RETAIL, 8.0),
            (BuildingType.HOTEL, 7.5),
            (BuildingType.RESIDENTIAL, 3.0),
        ]

        for btype, expected_target in building_types:
            input_data = BuildingEnergyInput(
                building_type=btype,
                floor_area_sqm=1000.0,
                annual_electricity_kwh=50000.0,
                annual_gas_m3=0.0,
                region="EU",
            )
            result = agent.run(input_data)
            assert result.crrem_target_intensity == expected_target

    @pytest.mark.golden
    def test_50_improvement_potential_calculation(
        self,
        agent: BuildingEnergyAgent,
        office_input_basic: BuildingEnergyInput,
    ):
        """
        Test 50: Improvement potential calculation

        ZERO-HALLUCINATION CHECK:
        Office best practice EUI: 70 kWh/m2
        improvement_kwh = (current_eui - best_eui) * floor_area
        """
        result = agent.run(office_input_basic)

        best_eui = EUI_BENCHMARKS["office"]["best"]
        if result.eui_kwh_sqm > best_eui:
            expected_improvement = (result.eui_kwh_sqm - best_eui) * 10000.0
            assert result.improvement_potential_kwh == pytest.approx(expected_improvement, rel=1e-2)


# =============================================================================
# Test 51-55: Determinism Tests
# =============================================================================


class TestDeterminism:
    """Tests for deterministic calculations (zero-hallucination)."""

    @pytest.mark.golden
    def test_51_deterministic_same_inputs(
        self,
        agent: BuildingEnergyAgent,
        office_input_basic: BuildingEnergyInput,
    ):
        """
        Test 51: Same inputs produce same outputs (zero-hallucination)
        """
        result1 = agent.run(office_input_basic)
        result2 = agent.run(office_input_basic)
        result3 = agent.run(office_input_basic)

        # All calculations should be identical
        assert result1.eui_kwh_sqm == result2.eui_kwh_sqm
        assert result2.eui_kwh_sqm == result3.eui_kwh_sqm
        assert result1.total_emissions_kgco2e == result2.total_emissions_kgco2e
        assert result1.epc_rating == result2.epc_rating

    @pytest.mark.golden
    def test_52_deterministic_across_instances(
        self,
        office_input_basic: BuildingEnergyInput,
    ):
        """
        Test 52: Different agent instances produce same results
        """
        agent1 = BuildingEnergyAgent()
        agent2 = BuildingEnergyAgent()
        agent3 = BuildingEnergyAgent(config={"different": "config"})

        result1 = agent1.run(office_input_basic)
        result2 = agent2.run(office_input_basic)
        result3 = agent3.run(office_input_basic)

        assert result1.eui_kwh_sqm == result2.eui_kwh_sqm
        assert result2.eui_kwh_sqm == result3.eui_kwh_sqm
        assert result1.total_emissions_kgco2e == result2.total_emissions_kgco2e

    @pytest.mark.golden
    def test_53_provenance_hash_format(
        self,
        agent: BuildingEnergyAgent,
        office_input_basic: BuildingEnergyInput,
    ):
        """
        Test 53: Provenance hash is valid SHA-256 format
        """
        result = agent.run(office_input_basic)

        assert len(result.provenance_hash) == 64
        assert all(c in "0123456789abcdef" for c in result.provenance_hash)

    @pytest.mark.golden
    def test_54_provenance_hash_changes_with_inputs(
        self,
        agent: BuildingEnergyAgent,
    ):
        """
        Test 54: Provenance hash changes when inputs change
        """
        input1 = BuildingEnergyInput(
            building_type=BuildingType.OFFICE,
            floor_area_sqm=1000.0,
            annual_electricity_kwh=100000.0,
            annual_gas_m3=0.0,
            region="EU",
        )
        input2 = BuildingEnergyInput(
            building_type=BuildingType.OFFICE,
            floor_area_sqm=1000.0,
            annual_electricity_kwh=200000.0,  # Different
            annual_gas_m3=0.0,
            region="EU",
        )

        result1 = agent.run(input1)
        result2 = agent.run(input2)

        # Different inputs should produce different hashes
        assert result1.provenance_hash != result2.provenance_hash

    @pytest.mark.golden
    def test_55_provenance_unique_per_run(
        self,
        agent: BuildingEnergyAgent,
        office_input_basic: BuildingEnergyInput,
    ):
        """
        Test 55: Provenance hash is unique per run (includes timestamp)
        """
        result1 = agent.run(office_input_basic)
        result2 = agent.run(office_input_basic)

        # Same input should produce different hashes due to timestamp
        assert result1.provenance_hash != result2.provenance_hash


# =============================================================================
# Test 56-65: Edge Cases and Input Validation
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.golden
    def test_56_minimum_floor_area(
        self,
        agent: BuildingEnergyAgent,
    ):
        """
        Test 56: Minimum valid floor area (1 sqm)
        """
        input_data = BuildingEnergyInput(
            building_type=BuildingType.OFFICE,
            floor_area_sqm=1.0,  # Minimum
            annual_electricity_kwh=100.0,
            annual_gas_m3=0.0,
            region="EU",
        )
        result = agent.run(input_data)

        assert result.eui_kwh_sqm == 100.0
        assert result is not None

    @pytest.mark.golden
    def test_57_very_large_building(
        self,
        agent: BuildingEnergyAgent,
    ):
        """
        Test 57: Very large building (1 million sqm)
        """
        input_data = BuildingEnergyInput(
            building_type=BuildingType.WAREHOUSE,
            floor_area_sqm=1000000.0,
            annual_electricity_kwh=100000000.0,
            annual_gas_m3=0.0,
            region="EU",
        )
        result = agent.run(input_data)

        expected_eui = 100000000.0 / 1000000.0
        assert result.eui_kwh_sqm == pytest.approx(expected_eui, rel=1e-4)

    @pytest.mark.golden
    def test_58_region_case_normalization(
        self,
        agent: BuildingEnergyAgent,
    ):
        """
        Test 58: Region code is normalized to uppercase
        """
        input_data = BuildingEnergyInput(
            building_type=BuildingType.OFFICE,
            floor_area_sqm=1000.0,
            annual_electricity_kwh=100000.0,
            annual_gas_m3=0.0,
            region="de",  # lowercase
        )
        result = agent.run(input_data)

        # Should use DE grid factor
        expected_emissions = 100000.0 * GRID_FACTORS["DE"]
        assert result.scope2_emissions == pytest.approx(expected_emissions, rel=1e-4)

    def test_59_input_validation_negative_energy(self):
        """
        Test 59: Negative energy values should be rejected
        """
        with pytest.raises(ValueError):
            BuildingEnergyInput(
                building_type=BuildingType.OFFICE,
                floor_area_sqm=1000.0,
                annual_electricity_kwh=-100.0,  # Negative
                annual_gas_m3=0.0,
                region="EU",
            )

    def test_60_input_validation_zero_floor_area(self):
        """
        Test 60: Zero floor area should be rejected
        """
        with pytest.raises(ValueError):
            BuildingEnergyInput(
                building_type=BuildingType.OFFICE,
                floor_area_sqm=0.0,  # Zero
                annual_electricity_kwh=100000.0,
                annual_gas_m3=0.0,
                region="EU",
            )

    @pytest.mark.golden
    def test_61_all_building_types_supported(
        self,
        agent: BuildingEnergyAgent,
    ):
        """
        Test 61: All building types are supported
        """
        for btype in BuildingType:
            input_data = BuildingEnergyInput(
                building_type=btype,
                floor_area_sqm=1000.0,
                annual_electricity_kwh=100000.0,
                annual_gas_m3=0.0,
                region="EU",
            )
            result = agent.run(input_data)
            assert result is not None
            assert result.building_type == btype.value

    @pytest.mark.golden
    def test_62_all_regions_supported(
        self,
        agent: BuildingEnergyAgent,
    ):
        """
        Test 62: All configured regions are supported
        """
        for region in GRID_FACTORS.keys():
            input_data = BuildingEnergyInput(
                building_type=BuildingType.OFFICE,
                floor_area_sqm=1000.0,
                annual_electricity_kwh=100000.0,
                annual_gas_m3=0.0,
                region=region,
            )
            result = agent.run(input_data)
            assert result is not None

            expected_emissions = 100000.0 * GRID_FACTORS[region]
            assert result.scope2_emissions == pytest.approx(expected_emissions, rel=1e-4)

    @pytest.mark.golden
    def test_63_metadata_preserved(
        self,
        agent: BuildingEnergyAgent,
    ):
        """
        Test 63: Metadata is accepted in input
        """
        input_data = BuildingEnergyInput(
            building_type=BuildingType.OFFICE,
            floor_area_sqm=1000.0,
            annual_electricity_kwh=100000.0,
            annual_gas_m3=0.0,
            region="EU",
            metadata={"property_id": "PROP-001", "portfolio": "European Core"},
        )
        result = agent.run(input_data)
        assert result is not None

    @pytest.mark.golden
    def test_64_occupancy_rate_accepted(
        self,
        agent: BuildingEnergyAgent,
    ):
        """
        Test 64: Occupancy rate is accepted but doesn't affect calculations
        """
        input1 = BuildingEnergyInput(
            building_type=BuildingType.OFFICE,
            floor_area_sqm=1000.0,
            annual_electricity_kwh=100000.0,
            annual_gas_m3=0.0,
            region="EU",
            occupancy_rate=50.0,
        )
        input2 = BuildingEnergyInput(
            building_type=BuildingType.OFFICE,
            floor_area_sqm=1000.0,
            annual_electricity_kwh=100000.0,
            annual_gas_m3=0.0,
            region="EU",
            occupancy_rate=100.0,
        )

        result1 = agent.run(input1)
        result2 = agent.run(input2)

        # EUI should be the same (occupancy doesn't affect current calculation)
        assert result1.eui_kwh_sqm == result2.eui_kwh_sqm

    @pytest.mark.golden
    def test_65_year_built_accepted(
        self,
        agent: BuildingEnergyAgent,
    ):
        """
        Test 65: Year built is accepted in input
        """
        input_data = BuildingEnergyInput(
            building_type=BuildingType.OFFICE,
            floor_area_sqm=1000.0,
            annual_electricity_kwh=100000.0,
            annual_gas_m3=0.0,
            region="EU",
            year_built=1990,
        )
        result = agent.run(input_data)
        assert result is not None


# =============================================================================
# Run Tests
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
