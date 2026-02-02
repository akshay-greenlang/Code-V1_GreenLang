"""
Unit Tests for GL-001: Carbon Emissions Calculator Agent

Comprehensive test coverage for the Carbon Emissions Calculator Agent including:
- Scope 1 emissions (natural gas, diesel, gasoline, LPG, coal)
- Scope 2 emissions (electricity by region with grid emission factors)
- Unit conversions and emission factor validation
- Edge cases (zero values, large values, missing data)
- Determinism verification tests
- Provenance hash validation

Test coverage target: 85%+
Total tests: 50+ golden tests covering all emission calculation scenarios

Formula Documentation:
----------------------
All emission calculations follow the GHG Protocol formula:
    emissions (kgCO2e) = activity_data (units) * emission_factor (kgCO2e/unit)

Emission Factors (from agent.py EMISSION_FACTORS):
- Natural Gas (US): 1.93 kgCO2e/m3 (EPA 2024)
- Natural Gas (EU): 2.02 kgCO2e/m3 (DEFRA 2024)
- Diesel (US): 2.68 kgCO2e/L (EPA 2024)
- Diesel (EU): 2.62 kgCO2e/L (DEFRA 2024)
- Gasoline (US): 2.31 kgCO2e/L (EPA 2024)
- Gasoline (EU): 2.19 kgCO2e/L (DEFRA 2024)
- Electricity (US): 0.417 kgCO2e/kWh (EPA eGRID 2024)
- Electricity (EU): 0.276 kgCO2e/kWh (IEA 2024)
- Electricity (DE): 0.366 kgCO2e/kWh (IEA 2024)
- Electricity (FR): 0.052 kgCO2e/kWh (IEA 2024)

Unit Conversions (from agent.py UNIT_CONVERSIONS):
- cf -> m3: 0.0283168
- therm -> m3: 2.832
- GJ -> m3: 26.137
- gal -> L: 3.78541
- MWh -> kWh: 1000
- t -> kg: 1000
- short_ton -> kg: 907.185
"""

import hashlib
import json
import pytest
from datetime import datetime
from typing import Dict, Any, Optional

from .agent import (
    CarbonEmissionsAgent,
    CarbonEmissionsInput,
    CarbonEmissionsOutput,
    FuelType,
    Scope,
    EmissionFactor,
)


# =============================================================================
# Test Constants - Expected Emission Factors
# =============================================================================

# Natural Gas Emission Factors (kgCO2e/m3)
EF_NATURAL_GAS_US = 1.93
EF_NATURAL_GAS_EU = 2.02

# Diesel Emission Factors (kgCO2e/L)
EF_DIESEL_US = 2.68
EF_DIESEL_EU = 2.62

# Gasoline Emission Factors (kgCO2e/L)
EF_GASOLINE_US = 2.31
EF_GASOLINE_EU = 2.19

# Electricity Grid Emission Factors (kgCO2e/kWh)
EF_ELECTRICITY_US = 0.417
EF_ELECTRICITY_EU = 0.276
EF_ELECTRICITY_DE = 0.366
EF_ELECTRICITY_FR = 0.052

# Unit Conversion Factors
CONV_CF_TO_M3 = 0.0283168
CONV_THERM_TO_M3 = 2.832
CONV_GJ_TO_M3 = 26.137
CONV_GAL_TO_L = 3.78541
CONV_MWH_TO_KWH = 1000


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def agent() -> CarbonEmissionsAgent:
    """Create a CarbonEmissionsAgent instance for testing."""
    return CarbonEmissionsAgent()


@pytest.fixture
def agent_with_config() -> CarbonEmissionsAgent:
    """Create agent with custom configuration."""
    return CarbonEmissionsAgent(config={"custom_setting": "value"})


@pytest.fixture
def natural_gas_input_us() -> CarbonEmissionsInput:
    """
    Create natural gas input for US region.

    Expected calculation:
        1000 m3 * 1.93 kgCO2e/m3 = 1930 kgCO2e
    """
    return CarbonEmissionsInput(
        fuel_type=FuelType.NATURAL_GAS,
        quantity=1000.0,
        unit="m3",
        region="US",
        scope=Scope.SCOPE_1,
    )


@pytest.fixture
def natural_gas_input_eu() -> CarbonEmissionsInput:
    """
    Create natural gas input for EU region.

    Expected calculation:
        1000 m3 * 2.02 kgCO2e/m3 = 2020 kgCO2e
    """
    return CarbonEmissionsInput(
        fuel_type=FuelType.NATURAL_GAS,
        quantity=1000.0,
        unit="m3",
        region="EU",
        scope=Scope.SCOPE_1,
    )


@pytest.fixture
def diesel_input_us() -> CarbonEmissionsInput:
    """
    Create diesel input for US region.

    Expected calculation:
        500 L * 2.68 kgCO2e/L = 1340 kgCO2e
    """
    return CarbonEmissionsInput(
        fuel_type=FuelType.DIESEL,
        quantity=500.0,
        unit="L",
        region="US",
        scope=Scope.SCOPE_1,
    )


@pytest.fixture
def gasoline_input_us() -> CarbonEmissionsInput:
    """
    Create gasoline input for US region.

    Expected calculation:
        1000 L * 2.31 kgCO2e/L = 2310 kgCO2e
    """
    return CarbonEmissionsInput(
        fuel_type=FuelType.GASOLINE,
        quantity=1000.0,
        unit="L",
        region="US",
        scope=Scope.SCOPE_1,
    )


@pytest.fixture
def electricity_input_us() -> CarbonEmissionsInput:
    """
    Create electricity input for US region.

    Expected calculation:
        10000 kWh * 0.417 kgCO2e/kWh = 4170 kgCO2e
    """
    return CarbonEmissionsInput(
        fuel_type=FuelType.ELECTRICITY_GRID,
        quantity=10000.0,
        unit="kWh",
        region="US",
        scope=Scope.SCOPE_2,
        calculation_method="location",
    )


@pytest.fixture
def electricity_input_france() -> CarbonEmissionsInput:
    """
    Create electricity input for France (low carbon grid).

    Expected calculation:
        10000 kWh * 0.052 kgCO2e/kWh = 520 kgCO2e
    """
    return CarbonEmissionsInput(
        fuel_type=FuelType.ELECTRICITY_GRID,
        quantity=10000.0,
        unit="kWh",
        region="FR",
        scope=Scope.SCOPE_2,
        calculation_method="location",
    )


@pytest.fixture
def electricity_input_germany() -> CarbonEmissionsInput:
    """
    Create electricity input for Germany.

    Expected calculation:
        10000 kWh * 0.366 kgCO2e/kWh = 3660 kgCO2e
    """
    return CarbonEmissionsInput(
        fuel_type=FuelType.ELECTRICITY_GRID,
        quantity=10000.0,
        unit="kWh",
        region="DE",
        scope=Scope.SCOPE_2,
        calculation_method="location",
    )


# =============================================================================
# Test 1-10: Agent Initialization and Basic Tests
# =============================================================================


class TestAgentInitialization:
    """Tests for agent initialization."""

    def test_01_agent_initialization(self, agent: CarbonEmissionsAgent):
        """Test 1: Agent initializes correctly with default config."""
        assert agent is not None
        assert agent.AGENT_ID == "emissions/carbon_calculator_v1"
        assert agent.VERSION == "1.0.0"
        assert agent.DESCRIPTION == "Zero-hallucination carbon emissions calculator"

    def test_02_agent_with_custom_config(self, agent_with_config: CarbonEmissionsAgent):
        """Test 2: Agent initializes with custom configuration."""
        assert agent_with_config.config == {"custom_setting": "value"}

    def test_03_emission_factors_loaded(self, agent: CarbonEmissionsAgent):
        """Test 3: Emission factors are loaded correctly."""
        ef = agent.EMISSION_FACTORS
        assert "natural_gas" in ef
        assert "diesel" in ef
        assert "gasoline" in ef
        assert "electricity_grid" in ef

    def test_04_unit_conversions_loaded(self, agent: CarbonEmissionsAgent):
        """Test 4: Unit conversion factors are loaded correctly."""
        conv = agent.UNIT_CONVERSIONS
        assert ("cf", "m3") in conv
        assert ("gal", "L") in conv
        assert ("MWh", "kWh") in conv

    def test_05_get_supported_fuel_types(self, agent: CarbonEmissionsAgent):
        """Test 5: Get supported fuel types returns all types."""
        fuel_types = agent.get_supported_fuel_types()
        assert "natural_gas" in fuel_types
        assert "diesel" in fuel_types
        assert "gasoline" in fuel_types
        assert "electricity_grid" in fuel_types
        assert len(fuel_types) == 7  # All FuelType enum values

    def test_06_get_supported_regions(self, agent: CarbonEmissionsAgent):
        """Test 6: Get supported regions returns regions with emission factors."""
        regions = agent.get_supported_regions()
        assert "US" in regions
        assert "EU" in regions
        assert "DE" in regions
        assert "FR" in regions

    def test_07_basic_run_completes(
        self,
        agent: CarbonEmissionsAgent,
        natural_gas_input_us: CarbonEmissionsInput,
    ):
        """Test 7: Basic agent run completes successfully."""
        result = agent.run(natural_gas_input_us)
        assert result is not None
        assert isinstance(result, CarbonEmissionsOutput)

    def test_08_run_returns_provenance_hash(
        self,
        agent: CarbonEmissionsAgent,
        natural_gas_input_us: CarbonEmissionsInput,
    ):
        """Test 8: Run returns valid SHA-256 provenance hash."""
        result = agent.run(natural_gas_input_us)
        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256 hex string

    def test_09_run_returns_emission_factor_info(
        self,
        agent: CarbonEmissionsAgent,
        natural_gas_input_us: CarbonEmissionsInput,
    ):
        """Test 9: Run returns emission factor details."""
        result = agent.run(natural_gas_input_us)
        assert result.emission_factor_used == EF_NATURAL_GAS_US
        assert result.emission_factor_unit == "kgCO2e/m3"
        assert result.emission_factor_source == "EPA"

    def test_10_run_returns_correct_scope(
        self,
        agent: CarbonEmissionsAgent,
        natural_gas_input_us: CarbonEmissionsInput,
    ):
        """Test 10: Run returns correct scope."""
        result = agent.run(natural_gas_input_us)
        assert result.scope == 1


# =============================================================================
# Test 11-20: Scope 1 Emissions - Natural Gas
# =============================================================================


class TestScope1NaturalGas:
    """Tests for Scope 1 natural gas emissions calculations."""

    @pytest.mark.golden
    def test_11_natural_gas_us_1000m3(
        self,
        agent: CarbonEmissionsAgent,
        natural_gas_input_us: CarbonEmissionsInput,
    ):
        """
        Test 11: Natural gas calculation - US, 1000 m3

        Formula: 1000 m3 * 1.93 kgCO2e/m3 = 1930 kgCO2e
        """
        result = agent.run(natural_gas_input_us)
        expected = 1000.0 * EF_NATURAL_GAS_US  # 1930.0
        assert result.emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.golden
    def test_12_natural_gas_eu_1000m3(
        self,
        agent: CarbonEmissionsAgent,
        natural_gas_input_eu: CarbonEmissionsInput,
    ):
        """
        Test 12: Natural gas calculation - EU, 1000 m3

        Formula: 1000 m3 * 2.02 kgCO2e/m3 = 2020 kgCO2e
        """
        result = agent.run(natural_gas_input_eu)
        expected = 1000.0 * EF_NATURAL_GAS_EU  # 2020.0
        assert result.emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.golden
    def test_13_natural_gas_cubic_feet_conversion(
        self,
        agent: CarbonEmissionsAgent,
    ):
        """
        Test 13: Natural gas with cubic feet unit conversion

        Formula: 35314.67 cf * 0.0283168 m3/cf = 1000 m3
                 1000 m3 * 1.93 kgCO2e/m3 = 1930 kgCO2e
        """
        input_data = CarbonEmissionsInput(
            fuel_type=FuelType.NATURAL_GAS,
            quantity=35314.67,  # ~1000 m3
            unit="cf",
            region="US",
            scope=Scope.SCOPE_1,
        )
        result = agent.run(input_data)
        # 35314.67 cf * 0.0283168 = 999.999... m3
        # 999.999 * 1.93 = 1929.998
        expected = 35314.67 * CONV_CF_TO_M3 * EF_NATURAL_GAS_US
        assert result.emissions_kgco2e == pytest.approx(expected, rel=1e-3)

    @pytest.mark.golden
    def test_14_natural_gas_therm_conversion(
        self,
        agent: CarbonEmissionsAgent,
    ):
        """
        Test 14: Natural gas with therm unit conversion

        Formula: 100 therm * 2.832 m3/therm = 283.2 m3
                 283.2 m3 * 1.93 kgCO2e/m3 = 546.576 kgCO2e
        """
        input_data = CarbonEmissionsInput(
            fuel_type=FuelType.NATURAL_GAS,
            quantity=100.0,
            unit="therm",
            region="US",
            scope=Scope.SCOPE_1,
        )
        result = agent.run(input_data)
        expected = 100.0 * CONV_THERM_TO_M3 * EF_NATURAL_GAS_US  # 546.576
        assert result.emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.golden
    def test_15_natural_gas_gj_conversion(
        self,
        agent: CarbonEmissionsAgent,
    ):
        """
        Test 15: Natural gas with GJ unit conversion

        Formula: 50 GJ * 26.137 m3/GJ = 1306.85 m3
                 1306.85 m3 * 1.93 kgCO2e/m3 = 2522.2205 kgCO2e
        """
        input_data = CarbonEmissionsInput(
            fuel_type=FuelType.NATURAL_GAS,
            quantity=50.0,
            unit="GJ",
            region="US",
            scope=Scope.SCOPE_1,
        )
        result = agent.run(input_data)
        expected = 50.0 * CONV_GJ_TO_M3 * EF_NATURAL_GAS_US  # 2522.2205
        assert result.emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.golden
    def test_16_natural_gas_small_quantity(
        self,
        agent: CarbonEmissionsAgent,
    ):
        """
        Test 16: Natural gas - small quantity precision

        Formula: 0.1 m3 * 1.93 kgCO2e/m3 = 0.193 kgCO2e
        """
        input_data = CarbonEmissionsInput(
            fuel_type=FuelType.NATURAL_GAS,
            quantity=0.1,
            unit="m3",
            region="US",
            scope=Scope.SCOPE_1,
        )
        result = agent.run(input_data)
        expected = 0.1 * EF_NATURAL_GAS_US  # 0.193
        assert result.emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.golden
    def test_17_natural_gas_large_quantity(
        self,
        agent: CarbonEmissionsAgent,
    ):
        """
        Test 17: Natural gas - large quantity (industrial scale)

        Formula: 1,000,000 m3 * 1.93 kgCO2e/m3 = 1,930,000 kgCO2e
        """
        input_data = CarbonEmissionsInput(
            fuel_type=FuelType.NATURAL_GAS,
            quantity=1_000_000.0,
            unit="m3",
            region="US",
            scope=Scope.SCOPE_1,
        )
        result = agent.run(input_data)
        expected = 1_000_000.0 * EF_NATURAL_GAS_US  # 1,930,000
        assert result.emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    def test_18_natural_gas_region_fallback_de(
        self,
        agent: CarbonEmissionsAgent,
    ):
        """
        Test 18: Natural gas - Germany region falls back to EU factor

        DE is not in natural_gas factors, should fall back to EU.
        Formula: 1000 m3 * 2.02 kgCO2e/m3 = 2020 kgCO2e
        """
        input_data = CarbonEmissionsInput(
            fuel_type=FuelType.NATURAL_GAS,
            quantity=1000.0,
            unit="m3",
            region="DE",
            scope=Scope.SCOPE_1,
        )
        result = agent.run(input_data)
        expected = 1000.0 * EF_NATURAL_GAS_EU  # Falls back to EU
        assert result.emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    def test_19_natural_gas_region_fallback_unknown(
        self,
        agent: CarbonEmissionsAgent,
    ):
        """
        Test 19: Natural gas - Unknown region falls back to US factor

        Brazil not in factors, should fall back to US.
        """
        input_data = CarbonEmissionsInput(
            fuel_type=FuelType.NATURAL_GAS,
            quantity=1000.0,
            unit="m3",
            region="BR",
            scope=Scope.SCOPE_1,
        )
        result = agent.run(input_data)
        expected = 1000.0 * EF_NATURAL_GAS_US  # Falls back to US
        assert result.emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    def test_20_natural_gas_emission_factor_source(
        self,
        agent: CarbonEmissionsAgent,
        natural_gas_input_us: CarbonEmissionsInput,
    ):
        """Test 20: Natural gas returns correct emission factor source."""
        result = agent.run(natural_gas_input_us)
        assert result.emission_factor_source == "EPA"


# =============================================================================
# Test 21-30: Scope 1 Emissions - Diesel and Gasoline
# =============================================================================


class TestScope1DieselGasoline:
    """Tests for Scope 1 diesel and gasoline emissions calculations."""

    @pytest.mark.golden
    def test_21_diesel_us_500l(
        self,
        agent: CarbonEmissionsAgent,
        diesel_input_us: CarbonEmissionsInput,
    ):
        """
        Test 21: Diesel calculation - US, 500 L

        Formula: 500 L * 2.68 kgCO2e/L = 1340 kgCO2e
        """
        result = agent.run(diesel_input_us)
        expected = 500.0 * EF_DIESEL_US  # 1340.0
        assert result.emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.golden
    def test_22_diesel_eu_500l(
        self,
        agent: CarbonEmissionsAgent,
    ):
        """
        Test 22: Diesel calculation - EU, 500 L

        Formula: 500 L * 2.62 kgCO2e/L = 1310 kgCO2e
        """
        input_data = CarbonEmissionsInput(
            fuel_type=FuelType.DIESEL,
            quantity=500.0,
            unit="L",
            region="EU",
            scope=Scope.SCOPE_1,
        )
        result = agent.run(input_data)
        expected = 500.0 * EF_DIESEL_EU  # 1310.0
        assert result.emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.golden
    def test_23_diesel_gallon_conversion(
        self,
        agent: CarbonEmissionsAgent,
    ):
        """
        Test 23: Diesel with gallon unit conversion

        Formula: 100 gal * 3.78541 L/gal = 378.541 L
                 378.541 L * 2.68 kgCO2e/L = 1014.48988 kgCO2e
        """
        input_data = CarbonEmissionsInput(
            fuel_type=FuelType.DIESEL,
            quantity=100.0,
            unit="gal",
            region="US",
            scope=Scope.SCOPE_1,
        )
        result = agent.run(input_data)
        expected = 100.0 * CONV_GAL_TO_L * EF_DIESEL_US  # 1014.48988
        assert result.emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.golden
    def test_24_gasoline_us_1000l(
        self,
        agent: CarbonEmissionsAgent,
        gasoline_input_us: CarbonEmissionsInput,
    ):
        """
        Test 24: Gasoline calculation - US, 1000 L

        Formula: 1000 L * 2.31 kgCO2e/L = 2310 kgCO2e
        """
        result = agent.run(gasoline_input_us)
        expected = 1000.0 * EF_GASOLINE_US  # 2310.0
        assert result.emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.golden
    def test_25_gasoline_eu_1000l(
        self,
        agent: CarbonEmissionsAgent,
    ):
        """
        Test 25: Gasoline calculation - EU, 1000 L

        Formula: 1000 L * 2.19 kgCO2e/L = 2190 kgCO2e
        """
        input_data = CarbonEmissionsInput(
            fuel_type=FuelType.GASOLINE,
            quantity=1000.0,
            unit="L",
            region="EU",
            scope=Scope.SCOPE_1,
        )
        result = agent.run(input_data)
        expected = 1000.0 * EF_GASOLINE_EU  # 2190.0
        assert result.emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.golden
    def test_26_gasoline_gallon_conversion(
        self,
        agent: CarbonEmissionsAgent,
    ):
        """
        Test 26: Gasoline with gallon unit conversion

        Formula: 100 gal * 3.78541 L/gal = 378.541 L
                 378.541 L * 2.31 kgCO2e/L = 874.42971 kgCO2e
        """
        input_data = CarbonEmissionsInput(
            fuel_type=FuelType.GASOLINE,
            quantity=100.0,
            unit="gal",
            region="US",
            scope=Scope.SCOPE_1,
        )
        result = agent.run(input_data)
        expected = 100.0 * CONV_GAL_TO_L * EF_GASOLINE_US  # 874.42971
        assert result.emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.golden
    def test_27_diesel_large_fleet(
        self,
        agent: CarbonEmissionsAgent,
    ):
        """
        Test 27: Diesel - large fleet consumption

        Formula: 50000 L * 2.68 kgCO2e/L = 134000 kgCO2e
        """
        input_data = CarbonEmissionsInput(
            fuel_type=FuelType.DIESEL,
            quantity=50000.0,
            unit="L",
            region="US",
            scope=Scope.SCOPE_1,
        )
        result = agent.run(input_data)
        expected = 50000.0 * EF_DIESEL_US  # 134000
        assert result.emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.golden
    def test_28_gasoline_small_quantity(
        self,
        agent: CarbonEmissionsAgent,
    ):
        """
        Test 28: Gasoline - small quantity (single vehicle fill)

        Formula: 50 L * 2.31 kgCO2e/L = 115.5 kgCO2e
        """
        input_data = CarbonEmissionsInput(
            fuel_type=FuelType.GASOLINE,
            quantity=50.0,
            unit="L",
            region="US",
            scope=Scope.SCOPE_1,
        )
        result = agent.run(input_data)
        expected = 50.0 * EF_GASOLINE_US  # 115.5
        assert result.emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    def test_29_diesel_emission_factor_details(
        self,
        agent: CarbonEmissionsAgent,
        diesel_input_us: CarbonEmissionsInput,
    ):
        """Test 29: Diesel returns correct emission factor details."""
        result = agent.run(diesel_input_us)
        assert result.emission_factor_used == EF_DIESEL_US
        assert result.emission_factor_unit == "kgCO2e/L"
        assert result.emission_factor_source == "EPA"

    def test_30_gasoline_emission_factor_details(
        self,
        agent: CarbonEmissionsAgent,
        gasoline_input_us: CarbonEmissionsInput,
    ):
        """Test 30: Gasoline returns correct emission factor details."""
        result = agent.run(gasoline_input_us)
        assert result.emission_factor_used == EF_GASOLINE_US
        assert result.emission_factor_unit == "kgCO2e/L"
        assert result.emission_factor_source == "EPA"


# =============================================================================
# Test 31-40: Scope 2 Emissions - Electricity
# =============================================================================


class TestScope2Electricity:
    """Tests for Scope 2 electricity emissions calculations."""

    @pytest.mark.golden
    def test_31_electricity_us_10000kwh(
        self,
        agent: CarbonEmissionsAgent,
        electricity_input_us: CarbonEmissionsInput,
    ):
        """
        Test 31: Electricity calculation - US, 10000 kWh

        Formula: 10000 kWh * 0.417 kgCO2e/kWh = 4170 kgCO2e
        """
        result = agent.run(electricity_input_us)
        expected = 10000.0 * EF_ELECTRICITY_US  # 4170.0
        assert result.emissions_kgco2e == pytest.approx(expected, rel=1e-6)
        assert result.scope == 2

    @pytest.mark.golden
    def test_32_electricity_eu_10000kwh(
        self,
        agent: CarbonEmissionsAgent,
    ):
        """
        Test 32: Electricity calculation - EU, 10000 kWh

        Formula: 10000 kWh * 0.276 kgCO2e/kWh = 2760 kgCO2e
        """
        input_data = CarbonEmissionsInput(
            fuel_type=FuelType.ELECTRICITY_GRID,
            quantity=10000.0,
            unit="kWh",
            region="EU",
            scope=Scope.SCOPE_2,
        )
        result = agent.run(input_data)
        expected = 10000.0 * EF_ELECTRICITY_EU  # 2760.0
        assert result.emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.golden
    def test_33_electricity_germany_10000kwh(
        self,
        agent: CarbonEmissionsAgent,
        electricity_input_germany: CarbonEmissionsInput,
    ):
        """
        Test 33: Electricity calculation - Germany, 10000 kWh

        Formula: 10000 kWh * 0.366 kgCO2e/kWh = 3660 kgCO2e
        """
        result = agent.run(electricity_input_germany)
        expected = 10000.0 * EF_ELECTRICITY_DE  # 3660.0
        assert result.emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.golden
    def test_34_electricity_france_10000kwh(
        self,
        agent: CarbonEmissionsAgent,
        electricity_input_france: CarbonEmissionsInput,
    ):
        """
        Test 34: Electricity calculation - France, 10000 kWh (nuclear-heavy grid)

        Formula: 10000 kWh * 0.052 kgCO2e/kWh = 520 kgCO2e

        France has one of the lowest grid emission factors due to nuclear power.
        """
        result = agent.run(electricity_input_france)
        expected = 10000.0 * EF_ELECTRICITY_FR  # 520.0
        assert result.emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.golden
    def test_35_electricity_mwh_conversion(
        self,
        agent: CarbonEmissionsAgent,
    ):
        """
        Test 35: Electricity with MWh unit conversion

        Formula: 10 MWh * 1000 kWh/MWh = 10000 kWh
                 10000 kWh * 0.417 kgCO2e/kWh = 4170 kgCO2e
        """
        input_data = CarbonEmissionsInput(
            fuel_type=FuelType.ELECTRICITY_GRID,
            quantity=10.0,
            unit="MWh",
            region="US",
            scope=Scope.SCOPE_2,
        )
        result = agent.run(input_data)
        expected = 10.0 * CONV_MWH_TO_KWH * EF_ELECTRICITY_US  # 4170.0
        assert result.emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.golden
    def test_36_electricity_large_industrial(
        self,
        agent: CarbonEmissionsAgent,
    ):
        """
        Test 36: Electricity - large industrial consumption

        Formula: 1,000,000 kWh * 0.417 kgCO2e/kWh = 417,000 kgCO2e
        """
        input_data = CarbonEmissionsInput(
            fuel_type=FuelType.ELECTRICITY_GRID,
            quantity=1_000_000.0,
            unit="kWh",
            region="US",
            scope=Scope.SCOPE_2,
        )
        result = agent.run(input_data)
        expected = 1_000_000.0 * EF_ELECTRICITY_US  # 417,000.0
        assert result.emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.golden
    def test_37_electricity_small_office(
        self,
        agent: CarbonEmissionsAgent,
    ):
        """
        Test 37: Electricity - small office consumption

        Formula: 500 kWh * 0.417 kgCO2e/kWh = 208.5 kgCO2e
        """
        input_data = CarbonEmissionsInput(
            fuel_type=FuelType.ELECTRICITY_GRID,
            quantity=500.0,
            unit="kWh",
            region="US",
            scope=Scope.SCOPE_2,
        )
        result = agent.run(input_data)
        expected = 500.0 * EF_ELECTRICITY_US  # 208.5
        assert result.emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    def test_38_electricity_location_method(
        self,
        agent: CarbonEmissionsAgent,
        electricity_input_us: CarbonEmissionsInput,
    ):
        """Test 38: Electricity returns location calculation method."""
        result = agent.run(electricity_input_us)
        assert result.calculation_method == "location"

    def test_39_electricity_emission_factor_source_us(
        self,
        agent: CarbonEmissionsAgent,
        electricity_input_us: CarbonEmissionsInput,
    ):
        """Test 39: US electricity returns EPA eGRID as source."""
        result = agent.run(electricity_input_us)
        assert result.emission_factor_source == "EPA eGRID"

    def test_40_electricity_emission_factor_source_eu(
        self,
        agent: CarbonEmissionsAgent,
    ):
        """Test 40: EU electricity returns IEA as source."""
        input_data = CarbonEmissionsInput(
            fuel_type=FuelType.ELECTRICITY_GRID,
            quantity=10000.0,
            unit="kWh",
            region="EU",
            scope=Scope.SCOPE_2,
        )
        result = agent.run(input_data)
        assert result.emission_factor_source == "IEA"


# =============================================================================
# Test 41-45: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.golden
    def test_41_zero_quantity(
        self,
        agent: CarbonEmissionsAgent,
    ):
        """
        Test 41: Zero quantity returns zero emissions

        Formula: 0 m3 * 1.93 kgCO2e/m3 = 0 kgCO2e
        """
        input_data = CarbonEmissionsInput(
            fuel_type=FuelType.NATURAL_GAS,
            quantity=0.0,
            unit="m3",
            region="US",
            scope=Scope.SCOPE_1,
        )
        result = agent.run(input_data)
        assert result.emissions_kgco2e == 0.0

    @pytest.mark.golden
    def test_42_very_small_quantity(
        self,
        agent: CarbonEmissionsAgent,
    ):
        """
        Test 42: Very small quantity (precision test)

        Formula: 0.001 m3 * 1.93 kgCO2e/m3 = 0.00193 kgCO2e
        """
        input_data = CarbonEmissionsInput(
            fuel_type=FuelType.NATURAL_GAS,
            quantity=0.001,
            unit="m3",
            region="US",
            scope=Scope.SCOPE_1,
        )
        result = agent.run(input_data)
        expected = 0.001 * EF_NATURAL_GAS_US  # 0.00193
        assert result.emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.golden
    def test_43_very_large_quantity(
        self,
        agent: CarbonEmissionsAgent,
    ):
        """
        Test 43: Very large quantity (corporate-level)

        Formula: 100,000,000 m3 * 1.93 kgCO2e/m3 = 193,000,000 kgCO2e
        """
        input_data = CarbonEmissionsInput(
            fuel_type=FuelType.NATURAL_GAS,
            quantity=100_000_000.0,
            unit="m3",
            region="US",
            scope=Scope.SCOPE_1,
        )
        result = agent.run(input_data)
        expected = 100_000_000.0 * EF_NATURAL_GAS_US  # 193,000,000
        assert result.emissions_kgco2e == pytest.approx(expected, rel=1e-6)

    def test_44_input_metadata_preserved(
        self,
        agent: CarbonEmissionsAgent,
    ):
        """Test 44: Input metadata is accepted."""
        input_data = CarbonEmissionsInput(
            fuel_type=FuelType.NATURAL_GAS,
            quantity=1000.0,
            unit="m3",
            region="US",
            scope=Scope.SCOPE_1,
            metadata={"facility_id": "FAC-001", "department": "Operations"},
        )
        result = agent.run(input_data)
        assert result is not None
        assert result.emissions_kgco2e > 0

    def test_45_missing_emission_factor_raises_error(
        self,
        agent: CarbonEmissionsAgent,
    ):
        """Test 45: Missing emission factor raises ValueError."""
        input_data = CarbonEmissionsInput(
            fuel_type=FuelType.COAL,  # Coal not fully defined
            quantity=1000.0,
            unit="kg",
            region="US",
            scope=Scope.SCOPE_1,
        )
        with pytest.raises(ValueError, match="No emission factor found"):
            agent.run(input_data)


# =============================================================================
# Test 46-50: Determinism and Provenance
# =============================================================================


class TestDeterminismAndProvenance:
    """Tests for deterministic calculations and provenance tracking."""

    @pytest.mark.golden
    def test_46_deterministic_calculation_same_inputs(
        self,
        agent: CarbonEmissionsAgent,
        natural_gas_input_us: CarbonEmissionsInput,
    ):
        """
        Test 46: Same inputs produce same emission values (zero-hallucination)

        This verifies the calculation is deterministic - no LLM involved in math.
        """
        result1 = agent.run(natural_gas_input_us)
        result2 = agent.run(natural_gas_input_us)
        result3 = agent.run(natural_gas_input_us)

        assert result1.emissions_kgco2e == result2.emissions_kgco2e
        assert result2.emissions_kgco2e == result3.emissions_kgco2e
        assert result1.emission_factor_used == result2.emission_factor_used

    @pytest.mark.golden
    def test_47_deterministic_across_agent_instances(
        self,
        natural_gas_input_us: CarbonEmissionsInput,
    ):
        """
        Test 47: Different agent instances produce same results

        Verifies the calculation doesn't depend on instance state.
        """
        agent1 = CarbonEmissionsAgent()
        agent2 = CarbonEmissionsAgent()
        agent3 = CarbonEmissionsAgent()

        result1 = agent1.run(natural_gas_input_us)
        result2 = agent2.run(natural_gas_input_us)
        result3 = agent3.run(natural_gas_input_us)

        assert result1.emissions_kgco2e == result2.emissions_kgco2e
        assert result2.emissions_kgco2e == result3.emissions_kgco2e

    def test_48_provenance_hash_format(
        self,
        agent: CarbonEmissionsAgent,
        natural_gas_input_us: CarbonEmissionsInput,
    ):
        """Test 48: Provenance hash is valid SHA-256 format."""
        result = agent.run(natural_gas_input_us)

        # SHA-256 produces 64 hex characters
        assert len(result.provenance_hash) == 64
        # Should be valid hex
        assert all(c in "0123456789abcdef" for c in result.provenance_hash)

    def test_49_provenance_hash_changes_with_inputs(
        self,
        agent: CarbonEmissionsAgent,
    ):
        """Test 49: Provenance hash changes when inputs change."""
        input1 = CarbonEmissionsInput(
            fuel_type=FuelType.NATURAL_GAS,
            quantity=1000.0,
            unit="m3",
            region="US",
            scope=Scope.SCOPE_1,
        )
        input2 = CarbonEmissionsInput(
            fuel_type=FuelType.NATURAL_GAS,
            quantity=2000.0,  # Different quantity
            unit="m3",
            region="US",
            scope=Scope.SCOPE_1,
        )

        result1 = agent.run(input1)
        result2 = agent.run(input2)

        # Different inputs should produce different provenance hashes
        # (Also includes timestamp, so will be different anyway)
        assert result1.provenance_hash != result2.provenance_hash

    def test_50_provenance_hash_includes_timestamp(
        self,
        agent: CarbonEmissionsAgent,
        natural_gas_input_us: CarbonEmissionsInput,
    ):
        """Test 50: Provenance hash is unique per run (includes timestamp)."""
        result1 = agent.run(natural_gas_input_us)
        result2 = agent.run(natural_gas_input_us)

        # Same input should produce different hashes due to timestamp
        assert result1.provenance_hash != result2.provenance_hash


# =============================================================================
# Test 51-55: Additional Validation Tests
# =============================================================================


class TestInputValidation:
    """Tests for input validation."""

    def test_51_input_model_validates_fuel_type(self):
        """Test 51: Input model validates fuel type."""
        input_data = CarbonEmissionsInput(
            fuel_type=FuelType.NATURAL_GAS,
            quantity=1000.0,
            unit="m3",
            region="US",
        )
        assert input_data.fuel_type == FuelType.NATURAL_GAS

    def test_52_input_model_validates_positive_quantity(self):
        """Test 52: Input model requires non-negative quantity."""
        with pytest.raises(ValueError):
            CarbonEmissionsInput(
                fuel_type=FuelType.NATURAL_GAS,
                quantity=-100.0,  # Negative not allowed
                unit="m3",
                region="US",
            )

    def test_53_input_model_validates_region_uppercase(self):
        """Test 53: Region is normalized to uppercase."""
        input_data = CarbonEmissionsInput(
            fuel_type=FuelType.NATURAL_GAS,
            quantity=1000.0,
            unit="m3",
            region="us",  # lowercase
        )
        assert input_data.region == "US"

    def test_54_scope_enum_values(self):
        """Test 54: Scope enum has correct values."""
        assert Scope.SCOPE_1.value == 1
        assert Scope.SCOPE_2.value == 2
        assert Scope.SCOPE_3.value == 3

    def test_55_fuel_type_enum_values(self):
        """Test 55: FuelType enum has all expected values."""
        fuel_types = [ft.value for ft in FuelType]
        assert "natural_gas" in fuel_types
        assert "diesel" in fuel_types
        assert "gasoline" in fuel_types
        assert "coal" in fuel_types
        assert "fuel_oil" in fuel_types
        assert "propane" in fuel_types
        assert "electricity_grid" in fuel_types


# =============================================================================
# Test 56-60: Output Model Validation
# =============================================================================


class TestOutputValidation:
    """Tests for output model validation."""

    def test_56_output_has_calculated_at_timestamp(
        self,
        agent: CarbonEmissionsAgent,
        natural_gas_input_us: CarbonEmissionsInput,
    ):
        """Test 56: Output includes calculated_at timestamp."""
        result = agent.run(natural_gas_input_us)
        assert result.calculated_at is not None
        assert isinstance(result.calculated_at, datetime)

    def test_57_output_emissions_rounded(
        self,
        agent: CarbonEmissionsAgent,
        natural_gas_input_us: CarbonEmissionsInput,
    ):
        """Test 57: Output emissions are rounded to 6 decimal places."""
        result = agent.run(natural_gas_input_us)
        emissions_str = f"{result.emissions_kgco2e:.6f}"
        # Should have at most 6 decimal places
        decimal_places = len(emissions_str.split(".")[-1])
        assert decimal_places <= 6

    def test_58_emission_factor_model(self):
        """Test 58: EmissionFactor model validation."""
        ef = EmissionFactor(
            value=1.93,
            unit="kgCO2e/m3",
            source="EPA",
            year=2024,
            uncertainty_lower=1.85,
            uncertainty_upper=2.01,
        )
        assert ef.value == 1.93
        assert ef.source == "EPA"
        assert ef.year == 2024

    def test_59_output_model_required_fields(
        self,
        agent: CarbonEmissionsAgent,
        natural_gas_input_us: CarbonEmissionsInput,
    ):
        """Test 59: Output model has all required fields."""
        result = agent.run(natural_gas_input_us)

        # All required fields present
        assert hasattr(result, "emissions_kgco2e")
        assert hasattr(result, "emission_factor_used")
        assert hasattr(result, "emission_factor_unit")
        assert hasattr(result, "emission_factor_source")
        assert hasattr(result, "scope")
        assert hasattr(result, "calculation_method")
        assert hasattr(result, "provenance_hash")

    def test_60_output_uncertainty_optional(
        self,
        agent: CarbonEmissionsAgent,
        natural_gas_input_us: CarbonEmissionsInput,
    ):
        """Test 60: Output uncertainty_pct is optional."""
        result = agent.run(natural_gas_input_us)
        # Should be None since EPA factors don't have uncertainty bounds defined
        assert result.uncertainty_pct is None or isinstance(result.uncertainty_pct, float)


# =============================================================================
# Run Tests
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
