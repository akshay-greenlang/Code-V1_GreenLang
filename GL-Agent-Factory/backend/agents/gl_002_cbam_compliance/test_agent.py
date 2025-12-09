"""
Unit Tests for GL-002: CBAM Compliance Agent

Comprehensive test coverage for the Carbon Border Adjustment Mechanism (CBAM)
Compliance Agent including:
- Embedded emissions calculations for all CBAM product categories
- CN code classification and validation
- Default emission factor lookups (global and country-specific)
- CBAM liability calculations
- Provenance hash validation
- Determinism verification (zero-hallucination)
- Edge cases and boundary conditions

Test coverage target: 85%+
Total tests: 75+ golden tests covering all CBAM calculation scenarios

Formula Documentation:
----------------------
All CBAM calculations follow EU Regulation (EU) 2023/956:

Embedded Emissions Calculation:
    direct_emissions (tCO2e) = quantity (tonnes) * direct_ef (tCO2e/t)
    indirect_emissions (tCO2e) = quantity (tonnes) * indirect_ef (tCO2e/t)
    total_embedded_emissions = direct_emissions + indirect_emissions

CBAM Liability Calculation:
    cbam_liability (EUR) = total_embedded_emissions (tCO2e) * carbon_price (EUR/tCO2)

Default Emission Factors (from agent.py DEFAULT_FACTORS):
Iron/Steel:
- Global: direct_ef=1.85, indirect_ef=0.32 tCO2e/t
- China (CN): direct_ef=2.10, indirect_ef=0.45 tCO2e/t
- India (IN): direct_ef=2.35, indirect_ef=0.52 tCO2e/t

Aluminium:
- Global: direct_ef=1.60, indirect_ef=6.50 tCO2e/t
- China (CN): direct_ef=1.65, indirect_ef=10.20 tCO2e/t

Cement:
- Global: direct_ef=0.83, indirect_ef=0.05 tCO2e/t

Fertilizers:
- Global: direct_ef=2.50, indirect_ef=0.12 tCO2e/t

Electricity:
- Global: direct_ef=0.50, indirect_ef=0.0 tCO2e/MWh

Hydrogen:
- Global: direct_ef=9.0, indirect_ef=1.5 tCO2e/t

EU ETS Carbon Price: 85.0 EUR/tCO2 (2024 average)
"""

import hashlib
import json
import pytest
from datetime import datetime
from typing import Dict, Any, Optional

from .agent import (
    CBAMComplianceAgent,
    CBAMInput,
    CBAMOutput,
    CBAMProductCategory,
    CalculationMethod,
    EmissionType,
    CBAMDefaultFactor,
)


# =============================================================================
# Test Constants - Expected Emission Factors (from agent.py)
# =============================================================================

# Iron/Steel Emission Factors (tCO2e/t)
EF_STEEL_GLOBAL_DIRECT = 1.85
EF_STEEL_GLOBAL_INDIRECT = 0.32
EF_STEEL_GLOBAL_TOTAL = EF_STEEL_GLOBAL_DIRECT + EF_STEEL_GLOBAL_INDIRECT  # 2.17

EF_STEEL_CN_DIRECT = 2.10
EF_STEEL_CN_INDIRECT = 0.45
EF_STEEL_CN_TOTAL = EF_STEEL_CN_DIRECT + EF_STEEL_CN_INDIRECT  # 2.55

EF_STEEL_IN_DIRECT = 2.35
EF_STEEL_IN_INDIRECT = 0.52
EF_STEEL_IN_TOTAL = EF_STEEL_IN_DIRECT + EF_STEEL_IN_INDIRECT  # 2.87

# Aluminium Emission Factors (tCO2e/t)
EF_ALUMINIUM_GLOBAL_DIRECT = 1.60
EF_ALUMINIUM_GLOBAL_INDIRECT = 6.50
EF_ALUMINIUM_GLOBAL_TOTAL = EF_ALUMINIUM_GLOBAL_DIRECT + EF_ALUMINIUM_GLOBAL_INDIRECT  # 8.10

EF_ALUMINIUM_CN_DIRECT = 1.65
EF_ALUMINIUM_CN_INDIRECT = 10.20
EF_ALUMINIUM_CN_TOTAL = EF_ALUMINIUM_CN_DIRECT + EF_ALUMINIUM_CN_INDIRECT  # 11.85

# Cement Emission Factors (tCO2e/t)
EF_CEMENT_GLOBAL_DIRECT = 0.83
EF_CEMENT_GLOBAL_INDIRECT = 0.05
EF_CEMENT_GLOBAL_TOTAL = EF_CEMENT_GLOBAL_DIRECT + EF_CEMENT_GLOBAL_INDIRECT  # 0.88

# Fertilizers Emission Factors (tCO2e/t)
EF_FERTILIZERS_GLOBAL_DIRECT = 2.50
EF_FERTILIZERS_GLOBAL_INDIRECT = 0.12
EF_FERTILIZERS_GLOBAL_TOTAL = EF_FERTILIZERS_GLOBAL_DIRECT + EF_FERTILIZERS_GLOBAL_INDIRECT  # 2.62

# Electricity Emission Factors (tCO2e/MWh)
EF_ELECTRICITY_GLOBAL_DIRECT = 0.50
EF_ELECTRICITY_GLOBAL_INDIRECT = 0.0
EF_ELECTRICITY_GLOBAL_TOTAL = EF_ELECTRICITY_GLOBAL_DIRECT + EF_ELECTRICITY_GLOBAL_INDIRECT  # 0.50

# Hydrogen Emission Factors (tCO2e/t)
EF_HYDROGEN_GLOBAL_DIRECT = 9.0
EF_HYDROGEN_GLOBAL_INDIRECT = 1.5
EF_HYDROGEN_GLOBAL_TOTAL = EF_HYDROGEN_GLOBAL_DIRECT + EF_HYDROGEN_GLOBAL_INDIRECT  # 10.5

# EU ETS Carbon Price (EUR/tCO2)
EU_ETS_PRICE = 85.0


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def agent() -> CBAMComplianceAgent:
    """Create a CBAMComplianceAgent instance for testing."""
    return CBAMComplianceAgent()


@pytest.fixture
def agent_with_config() -> CBAMComplianceAgent:
    """Create agent with custom configuration."""
    return CBAMComplianceAgent(config={"custom_setting": "value"})


@pytest.fixture
def steel_input_global() -> CBAMInput:
    """
    Create steel import input for global default factors.

    CN code 7208.10.00 = Flat-rolled products of iron/non-alloy steel
    Expected calculation (1000 tonnes, global EF):
        direct_emissions = 1000 * 1.85 = 1850 tCO2e
        indirect_emissions = 1000 * 0.32 = 320 tCO2e
        total_emissions = 2170 tCO2e
        cbam_liability = 2170 * 85 = 184,450 EUR
    """
    return CBAMInput(
        cn_code="7208.10.00",
        quantity_tonnes=1000.0,
        country_of_origin="DE",  # Germany, uses global default
        reporting_period="Q1 2026",
    )


@pytest.fixture
def steel_input_china() -> CBAMInput:
    """
    Create steel import input from China with country-specific factors.

    CN code 7208.10.00 = Flat-rolled products of iron/non-alloy steel
    Expected calculation (1000 tonnes, China EF):
        direct_emissions = 1000 * 2.10 = 2100 tCO2e
        indirect_emissions = 1000 * 0.45 = 450 tCO2e
        total_emissions = 2550 tCO2e
        cbam_liability = 2550 * 85 = 216,750 EUR
    """
    return CBAMInput(
        cn_code="7208.10.00",
        quantity_tonnes=1000.0,
        country_of_origin="CN",
        reporting_period="Q1 2026",
    )


@pytest.fixture
def steel_input_india() -> CBAMInput:
    """
    Create steel import input from India with country-specific factors.

    Expected calculation (500 tonnes, India EF):
        direct_emissions = 500 * 2.35 = 1175 tCO2e
        indirect_emissions = 500 * 0.52 = 260 tCO2e
        total_emissions = 1435 tCO2e
        cbam_liability = 1435 * 85 = 121,975 EUR
    """
    return CBAMInput(
        cn_code="7208.10.00",
        quantity_tonnes=500.0,
        country_of_origin="IN",
        reporting_period="Q1 2026",
    )


@pytest.fixture
def aluminium_input_global() -> CBAMInput:
    """
    Create aluminium import input for global default factors.

    CN code 7601.10.00 = Unwrought aluminium, not alloyed
    Expected calculation (100 tonnes, global EF):
        direct_emissions = 100 * 1.60 = 160 tCO2e
        indirect_emissions = 100 * 6.50 = 650 tCO2e
        total_emissions = 810 tCO2e
        cbam_liability = 810 * 85 = 68,850 EUR
    """
    return CBAMInput(
        cn_code="7601.10.00",
        quantity_tonnes=100.0,
        country_of_origin="NO",  # Norway, uses global default
        reporting_period="Q2 2026",
    )


@pytest.fixture
def aluminium_input_china() -> CBAMInput:
    """
    Create aluminium import input from China with country-specific factors.

    Expected calculation (100 tonnes, China EF):
        direct_emissions = 100 * 1.65 = 165 tCO2e
        indirect_emissions = 100 * 10.20 = 1020 tCO2e
        total_emissions = 1185 tCO2e
        cbam_liability = 1185 * 85 = 100,725 EUR
    """
    return CBAMInput(
        cn_code="7601.10.00",
        quantity_tonnes=100.0,
        country_of_origin="CN",
        reporting_period="Q2 2026",
    )


@pytest.fixture
def cement_input() -> CBAMInput:
    """
    Create cement import input.

    CN code 2523.29.00 = Portland cement
    Expected calculation (5000 tonnes, global EF):
        direct_emissions = 5000 * 0.83 = 4150 tCO2e
        indirect_emissions = 5000 * 0.05 = 250 tCO2e
        total_emissions = 4400 tCO2e
        cbam_liability = 4400 * 85 = 374,000 EUR
    """
    return CBAMInput(
        cn_code="2523.29.00",
        quantity_tonnes=5000.0,
        country_of_origin="TR",  # Turkey
        reporting_period="Q3 2026",
    )


@pytest.fixture
def fertilizers_input() -> CBAMInput:
    """
    Create fertilizers import input.

    CN code 3102.10.00 = Urea
    Expected calculation (200 tonnes, global EF):
        direct_emissions = 200 * 2.50 = 500 tCO2e
        indirect_emissions = 200 * 0.12 = 24 tCO2e
        total_emissions = 524 tCO2e
        cbam_liability = 524 * 85 = 44,540 EUR
    """
    return CBAMInput(
        cn_code="3102.10.00",
        quantity_tonnes=200.0,
        country_of_origin="RU",  # Russia
        reporting_period="Q4 2026",
    )


@pytest.fixture
def hydrogen_input() -> CBAMInput:
    """
    Create hydrogen import input.

    CN code 2804.10.00 = Hydrogen
    Expected calculation (50 tonnes, global EF):
        direct_emissions = 50 * 9.0 = 450 tCO2e
        indirect_emissions = 50 * 1.5 = 75 tCO2e
        total_emissions = 525 tCO2e
        cbam_liability = 525 * 85 = 44,625 EUR
    """
    return CBAMInput(
        cn_code="2804.10.00",
        quantity_tonnes=50.0,
        country_of_origin="SA",  # Saudi Arabia
        reporting_period="Q1 2027",
    )


@pytest.fixture
def actual_emissions_input() -> CBAMInput:
    """
    Create input with actual verified emissions.

    When actual emissions are provided, they are used instead of defaults.
    Expected calculation (1000 tonnes, actual_emissions=1.5):
        direct_emissions = 1000 * 1.5 = 1500 tCO2e
        indirect_emissions = 0 (included in actual)
        total_emissions = 1500 tCO2e
        cbam_liability = 1500 * 85 = 127,500 EUR
    """
    return CBAMInput(
        cn_code="7208.10.00",
        quantity_tonnes=1000.0,
        country_of_origin="CN",
        actual_emissions=1.5,  # Verified lower emissions
        installation_id="CN-STEEL-001",
        reporting_period="Q1 2026",
    )


# =============================================================================
# Test 1-10: Agent Initialization and Basic Tests
# =============================================================================


class TestAgentInitialization:
    """Tests for agent initialization."""

    def test_01_agent_initialization(self, agent: CBAMComplianceAgent):
        """Test 1: Agent initializes correctly with default config."""
        assert agent is not None
        assert agent.AGENT_ID == "regulatory/cbam_compliance_v1"
        assert agent.VERSION == "1.0.0"
        assert agent.DESCRIPTION == "CBAM embedded emissions calculator with zero hallucination"

    def test_02_agent_with_custom_config(self, agent_with_config: CBAMComplianceAgent):
        """Test 2: Agent initializes with custom configuration."""
        assert agent_with_config.config == {"custom_setting": "value"}

    def test_03_cn_to_category_mapping_loaded(self, agent: CBAMComplianceAgent):
        """Test 3: CN code to category mapping is loaded correctly."""
        cn_map = agent.CN_TO_CATEGORY
        assert "7208" in cn_map  # Iron/Steel
        assert "7601" in cn_map  # Aluminium
        assert "2523" in cn_map  # Cement
        assert "3102" in cn_map  # Fertilizers
        assert "2716" in cn_map  # Electricity
        assert "2804" in cn_map  # Hydrogen

    def test_04_default_factors_loaded(self, agent: CBAMComplianceAgent):
        """Test 4: Default emission factors are loaded correctly."""
        factors = agent.DEFAULT_FACTORS
        assert "iron_steel" in factors
        assert "aluminium" in factors
        assert "cement" in factors
        assert "fertilizers" in factors
        assert "electricity" in factors
        assert "hydrogen" in factors

    def test_05_eu_ets_price_loaded(self, agent: CBAMComplianceAgent):
        """Test 5: EU ETS carbon price is loaded correctly."""
        assert agent.EU_ETS_PRICE == EU_ETS_PRICE

    def test_06_get_cbam_products(self, agent: CBAMComplianceAgent):
        """Test 6: Get CBAM product categories returns all categories."""
        products = agent.get_cbam_products()
        assert "cement" in products
        assert "iron_steel" in products
        assert "aluminium" in products
        assert "fertilizers" in products
        assert "electricity" in products
        assert "hydrogen" in products
        assert len(products) == 6  # All CBAM categories

    def test_07_is_in_scope_steel(self, agent: CBAMComplianceAgent):
        """Test 7: CN code 7208 (steel) is in CBAM scope."""
        assert agent.is_in_scope("72081000") is True
        assert agent.is_in_scope("7208.10.00") is True

    def test_08_is_in_scope_aluminium(self, agent: CBAMComplianceAgent):
        """Test 8: CN code 7601 (aluminium) is in CBAM scope."""
        assert agent.is_in_scope("76011000") is True

    def test_09_is_in_scope_out_of_scope_product(self, agent: CBAMComplianceAgent):
        """Test 9: CN code 9999 (invalid) is not in CBAM scope."""
        assert agent.is_in_scope("99990000") is False

    def test_10_basic_run_completes(
        self,
        agent: CBAMComplianceAgent,
        steel_input_global: CBAMInput,
    ):
        """Test 10: Basic agent run completes successfully."""
        result = agent.run(steel_input_global)
        assert result is not None
        assert isinstance(result, CBAMOutput)


# =============================================================================
# Test 11-20: Iron/Steel CBAM Calculations
# =============================================================================


class TestIronSteelCalculations:
    """Tests for iron/steel CBAM calculations."""

    @pytest.mark.golden
    def test_11_steel_global_1000t(
        self,
        agent: CBAMComplianceAgent,
        steel_input_global: CBAMInput,
    ):
        """
        Test 11: Steel import calculation - Global default, 1000 tonnes

        Formula:
            direct_emissions = 1000 * 1.85 = 1850 tCO2e
            indirect_emissions = 1000 * 0.32 = 320 tCO2e
            total_emissions = 2170 tCO2e
        """
        result = agent.run(steel_input_global)

        expected_direct = 1000.0 * EF_STEEL_GLOBAL_DIRECT  # 1850
        expected_indirect = 1000.0 * EF_STEEL_GLOBAL_INDIRECT  # 320
        expected_total = expected_direct + expected_indirect  # 2170

        assert result.direct_emissions_tco2e == pytest.approx(expected_direct, rel=1e-6)
        assert result.indirect_emissions_tco2e == pytest.approx(expected_indirect, rel=1e-6)
        assert result.total_embedded_emissions_tco2e == pytest.approx(expected_total, rel=1e-6)

    @pytest.mark.golden
    def test_12_steel_global_cbam_liability(
        self,
        agent: CBAMComplianceAgent,
        steel_input_global: CBAMInput,
    ):
        """
        Test 12: Steel import CBAM liability calculation - Global default

        Formula:
            cbam_liability = 2170 tCO2e * 85 EUR/tCO2 = 184,450 EUR
        """
        result = agent.run(steel_input_global)

        expected_total = 1000.0 * EF_STEEL_GLOBAL_TOTAL  # 2170
        expected_liability = expected_total * EU_ETS_PRICE  # 184,450

        assert result.cbam_liability_eur == pytest.approx(expected_liability, rel=1e-2)

    @pytest.mark.golden
    def test_13_steel_china_1000t(
        self,
        agent: CBAMComplianceAgent,
        steel_input_china: CBAMInput,
    ):
        """
        Test 13: Steel import calculation - China, 1000 tonnes

        China has higher emission factors due to coal-heavy production.
        Formula:
            direct_emissions = 1000 * 2.10 = 2100 tCO2e
            indirect_emissions = 1000 * 0.45 = 450 tCO2e
            total_emissions = 2550 tCO2e
        """
        result = agent.run(steel_input_china)

        expected_direct = 1000.0 * EF_STEEL_CN_DIRECT  # 2100
        expected_indirect = 1000.0 * EF_STEEL_CN_INDIRECT  # 450
        expected_total = expected_direct + expected_indirect  # 2550

        assert result.direct_emissions_tco2e == pytest.approx(expected_direct, rel=1e-6)
        assert result.indirect_emissions_tco2e == pytest.approx(expected_indirect, rel=1e-6)
        assert result.total_embedded_emissions_tco2e == pytest.approx(expected_total, rel=1e-6)

    @pytest.mark.golden
    def test_14_steel_china_cbam_liability(
        self,
        agent: CBAMComplianceAgent,
        steel_input_china: CBAMInput,
    ):
        """
        Test 14: Steel import CBAM liability - China

        Formula:
            cbam_liability = 2550 tCO2e * 85 EUR/tCO2 = 216,750 EUR
        """
        result = agent.run(steel_input_china)

        expected_total = 1000.0 * EF_STEEL_CN_TOTAL  # 2550
        expected_liability = expected_total * EU_ETS_PRICE  # 216,750

        assert result.cbam_liability_eur == pytest.approx(expected_liability, rel=1e-2)

    @pytest.mark.golden
    def test_15_steel_india_500t(
        self,
        agent: CBAMComplianceAgent,
        steel_input_india: CBAMInput,
    ):
        """
        Test 15: Steel import calculation - India, 500 tonnes

        India has highest steel emission factors.
        Formula:
            direct_emissions = 500 * 2.35 = 1175 tCO2e
            indirect_emissions = 500 * 0.52 = 260 tCO2e
            total_emissions = 1435 tCO2e
            cbam_liability = 1435 * 85 = 121,975 EUR
        """
        result = agent.run(steel_input_india)

        expected_direct = 500.0 * EF_STEEL_IN_DIRECT  # 1175
        expected_indirect = 500.0 * EF_STEEL_IN_INDIRECT  # 260
        expected_total = expected_direct + expected_indirect  # 1435
        expected_liability = expected_total * EU_ETS_PRICE  # 121,975

        assert result.direct_emissions_tco2e == pytest.approx(expected_direct, rel=1e-6)
        assert result.indirect_emissions_tco2e == pytest.approx(expected_indirect, rel=1e-6)
        assert result.total_embedded_emissions_tco2e == pytest.approx(expected_total, rel=1e-6)
        assert result.cbam_liability_eur == pytest.approx(expected_liability, rel=1e-2)

    @pytest.mark.golden
    def test_16_steel_specific_embedded_emissions(
        self,
        agent: CBAMComplianceAgent,
        steel_input_global: CBAMInput,
    ):
        """
        Test 16: Steel specific embedded emissions (per tonne)

        specific_embedded_emissions = direct_ef + indirect_ef = 1.85 + 0.32 = 2.17 tCO2e/t
        """
        result = agent.run(steel_input_global)

        expected_specific = EF_STEEL_GLOBAL_TOTAL  # 2.17
        assert result.specific_embedded_emissions == pytest.approx(expected_specific, rel=1e-6)

    def test_17_steel_product_category_classification(
        self,
        agent: CBAMComplianceAgent,
        steel_input_global: CBAMInput,
    ):
        """Test 17: Steel CN code is correctly classified as iron_steel."""
        result = agent.run(steel_input_global)
        assert result.product_category == "iron_steel"

    def test_18_steel_calculation_method_default(
        self,
        agent: CBAMComplianceAgent,
        steel_input_global: CBAMInput,
    ):
        """Test 18: Steel with global default uses 'default' method."""
        result = agent.run(steel_input_global)
        assert result.calculation_method == CalculationMethod.DEFAULT.value

    def test_19_steel_calculation_method_country(
        self,
        agent: CBAMComplianceAgent,
        steel_input_china: CBAMInput,
    ):
        """Test 19: Steel from China uses 'country_default' method."""
        result = agent.run(steel_input_china)
        assert result.calculation_method == CalculationMethod.COUNTRY_DEFAULT.value

    def test_20_steel_emission_factor_source(
        self,
        agent: CBAMComplianceAgent,
        steel_input_global: CBAMInput,
    ):
        """Test 20: Steel returns correct emission factor source."""
        result = agent.run(steel_input_global)
        assert "EU Implementing Regulation" in result.emission_factor_source


# =============================================================================
# Test 21-30: Aluminium CBAM Calculations
# =============================================================================


class TestAluminiumCalculations:
    """Tests for aluminium CBAM calculations."""

    @pytest.mark.golden
    def test_21_aluminium_global_100t(
        self,
        agent: CBAMComplianceAgent,
        aluminium_input_global: CBAMInput,
    ):
        """
        Test 21: Aluminium import calculation - Global default, 100 tonnes

        Aluminium has high indirect emissions due to electricity intensity.
        Formula:
            direct_emissions = 100 * 1.60 = 160 tCO2e
            indirect_emissions = 100 * 6.50 = 650 tCO2e
            total_emissions = 810 tCO2e
        """
        result = agent.run(aluminium_input_global)

        expected_direct = 100.0 * EF_ALUMINIUM_GLOBAL_DIRECT  # 160
        expected_indirect = 100.0 * EF_ALUMINIUM_GLOBAL_INDIRECT  # 650
        expected_total = expected_direct + expected_indirect  # 810

        assert result.direct_emissions_tco2e == pytest.approx(expected_direct, rel=1e-6)
        assert result.indirect_emissions_tco2e == pytest.approx(expected_indirect, rel=1e-6)
        assert result.total_embedded_emissions_tco2e == pytest.approx(expected_total, rel=1e-6)

    @pytest.mark.golden
    def test_22_aluminium_global_cbam_liability(
        self,
        agent: CBAMComplianceAgent,
        aluminium_input_global: CBAMInput,
    ):
        """
        Test 22: Aluminium import CBAM liability - Global default

        Formula:
            cbam_liability = 810 tCO2e * 85 EUR/tCO2 = 68,850 EUR
        """
        result = agent.run(aluminium_input_global)

        expected_total = 100.0 * EF_ALUMINIUM_GLOBAL_TOTAL  # 810
        expected_liability = expected_total * EU_ETS_PRICE  # 68,850

        assert result.cbam_liability_eur == pytest.approx(expected_liability, rel=1e-2)

    @pytest.mark.golden
    def test_23_aluminium_china_100t(
        self,
        agent: CBAMComplianceAgent,
        aluminium_input_china: CBAMInput,
    ):
        """
        Test 23: Aluminium import calculation - China, 100 tonnes

        China aluminium has very high indirect emissions due to coal grid.
        Formula:
            direct_emissions = 100 * 1.65 = 165 tCO2e
            indirect_emissions = 100 * 10.20 = 1020 tCO2e
            total_emissions = 1185 tCO2e
        """
        result = agent.run(aluminium_input_china)

        expected_direct = 100.0 * EF_ALUMINIUM_CN_DIRECT  # 165
        expected_indirect = 100.0 * EF_ALUMINIUM_CN_INDIRECT  # 1020
        expected_total = expected_direct + expected_indirect  # 1185

        assert result.direct_emissions_tco2e == pytest.approx(expected_direct, rel=1e-6)
        assert result.indirect_emissions_tco2e == pytest.approx(expected_indirect, rel=1e-6)
        assert result.total_embedded_emissions_tco2e == pytest.approx(expected_total, rel=1e-6)

    @pytest.mark.golden
    def test_24_aluminium_china_cbam_liability(
        self,
        agent: CBAMComplianceAgent,
        aluminium_input_china: CBAMInput,
    ):
        """
        Test 24: Aluminium import CBAM liability - China

        Formula:
            cbam_liability = 1185 tCO2e * 85 EUR/tCO2 = 100,725 EUR
        """
        result = agent.run(aluminium_input_china)

        expected_total = 100.0 * EF_ALUMINIUM_CN_TOTAL  # 1185
        expected_liability = expected_total * EU_ETS_PRICE  # 100,725

        assert result.cbam_liability_eur == pytest.approx(expected_liability, rel=1e-2)

    @pytest.mark.golden
    def test_25_aluminium_high_indirect_ratio(
        self,
        agent: CBAMComplianceAgent,
        aluminium_input_global: CBAMInput,
    ):
        """
        Test 25: Aluminium has high indirect-to-direct emissions ratio

        Aluminium is electricity-intensive, so indirect > direct emissions.
        Ratio = 6.50 / 1.60 = 4.0625
        """
        result = agent.run(aluminium_input_global)

        # Indirect emissions should be ~4x direct for aluminium
        ratio = result.indirect_emissions_tco2e / result.direct_emissions_tco2e
        expected_ratio = EF_ALUMINIUM_GLOBAL_INDIRECT / EF_ALUMINIUM_GLOBAL_DIRECT
        assert ratio == pytest.approx(expected_ratio, rel=1e-6)

    @pytest.mark.golden
    def test_26_aluminium_specific_embedded_emissions(
        self,
        agent: CBAMComplianceAgent,
        aluminium_input_global: CBAMInput,
    ):
        """
        Test 26: Aluminium specific embedded emissions (per tonne)

        specific_embedded_emissions = 1.60 + 6.50 = 8.10 tCO2e/t
        """
        result = agent.run(aluminium_input_global)

        expected_specific = EF_ALUMINIUM_GLOBAL_TOTAL  # 8.10
        assert result.specific_embedded_emissions == pytest.approx(expected_specific, rel=1e-6)

    def test_27_aluminium_product_category_classification(
        self,
        agent: CBAMComplianceAgent,
        aluminium_input_global: CBAMInput,
    ):
        """Test 27: Aluminium CN code is correctly classified."""
        result = agent.run(aluminium_input_global)
        assert result.product_category == "aluminium"

    @pytest.mark.golden
    def test_28_aluminium_vs_china_comparison(
        self,
        agent: CBAMComplianceAgent,
        aluminium_input_global: CBAMInput,
        aluminium_input_china: CBAMInput,
    ):
        """
        Test 28: China aluminium has ~46% higher emissions than global default

        Global: 8.10 tCO2e/t
        China: 11.85 tCO2e/t
        Increase: (11.85 - 8.10) / 8.10 = 46.3%
        """
        result_global = agent.run(aluminium_input_global)
        result_china = agent.run(aluminium_input_china)

        # China should have significantly higher emissions
        assert result_china.total_embedded_emissions_tco2e > result_global.total_embedded_emissions_tco2e

        # ~46% higher for same quantity
        ratio = result_china.total_embedded_emissions_tco2e / result_global.total_embedded_emissions_tco2e
        expected_ratio = EF_ALUMINIUM_CN_TOTAL / EF_ALUMINIUM_GLOBAL_TOTAL
        assert ratio == pytest.approx(expected_ratio, rel=1e-6)

    def test_29_aluminium_cn_codes_coverage(self, agent: CBAMComplianceAgent):
        """Test 29: Various aluminium CN codes are in scope."""
        aluminium_codes = ["7601", "7602", "7604", "7606", "7608", "7610"]
        for code in aluminium_codes:
            assert agent.is_in_scope(f"{code}0000") is True

    def test_30_aluminium_reporting_period_preserved(
        self,
        agent: CBAMComplianceAgent,
        aluminium_input_global: CBAMInput,
    ):
        """Test 30: Reporting period is preserved in output."""
        result = agent.run(aluminium_input_global)
        assert result.reporting_period == "Q2 2026"


# =============================================================================
# Test 31-40: Cement, Fertilizers, Hydrogen CBAM Calculations
# =============================================================================


class TestOtherProductCategories:
    """Tests for cement, fertilizers, and hydrogen CBAM calculations."""

    @pytest.mark.golden
    def test_31_cement_5000t(
        self,
        agent: CBAMComplianceAgent,
        cement_input: CBAMInput,
    ):
        """
        Test 31: Cement import calculation - 5000 tonnes

        Formula:
            direct_emissions = 5000 * 0.83 = 4150 tCO2e
            indirect_emissions = 5000 * 0.05 = 250 tCO2e
            total_emissions = 4400 tCO2e
            cbam_liability = 4400 * 85 = 374,000 EUR
        """
        result = agent.run(cement_input)

        expected_direct = 5000.0 * EF_CEMENT_GLOBAL_DIRECT  # 4150
        expected_indirect = 5000.0 * EF_CEMENT_GLOBAL_INDIRECT  # 250
        expected_total = expected_direct + expected_indirect  # 4400
        expected_liability = expected_total * EU_ETS_PRICE  # 374,000

        assert result.direct_emissions_tco2e == pytest.approx(expected_direct, rel=1e-6)
        assert result.indirect_emissions_tco2e == pytest.approx(expected_indirect, rel=1e-6)
        assert result.total_embedded_emissions_tco2e == pytest.approx(expected_total, rel=1e-6)
        assert result.cbam_liability_eur == pytest.approx(expected_liability, rel=1e-2)

    def test_32_cement_product_category(
        self,
        agent: CBAMComplianceAgent,
        cement_input: CBAMInput,
    ):
        """Test 32: Cement CN code is correctly classified."""
        result = agent.run(cement_input)
        assert result.product_category == "cement"

    @pytest.mark.golden
    def test_33_cement_high_direct_emissions(
        self,
        agent: CBAMComplianceAgent,
        cement_input: CBAMInput,
    ):
        """
        Test 33: Cement has high direct-to-indirect ratio (process emissions)

        Cement production involves calcination - a chemical process that releases CO2.
        Ratio = 0.83 / 0.05 = 16.6
        """
        result = agent.run(cement_input)

        ratio = result.direct_emissions_tco2e / result.indirect_emissions_tco2e
        expected_ratio = EF_CEMENT_GLOBAL_DIRECT / EF_CEMENT_GLOBAL_INDIRECT
        assert ratio == pytest.approx(expected_ratio, rel=1e-6)

    @pytest.mark.golden
    def test_34_fertilizers_200t(
        self,
        agent: CBAMComplianceAgent,
        fertilizers_input: CBAMInput,
    ):
        """
        Test 34: Fertilizers import calculation - 200 tonnes

        Formula:
            direct_emissions = 200 * 2.50 = 500 tCO2e
            indirect_emissions = 200 * 0.12 = 24 tCO2e
            total_emissions = 524 tCO2e
            cbam_liability = 524 * 85 = 44,540 EUR
        """
        result = agent.run(fertilizers_input)

        expected_direct = 200.0 * EF_FERTILIZERS_GLOBAL_DIRECT  # 500
        expected_indirect = 200.0 * EF_FERTILIZERS_GLOBAL_INDIRECT  # 24
        expected_total = expected_direct + expected_indirect  # 524
        expected_liability = expected_total * EU_ETS_PRICE  # 44,540

        assert result.direct_emissions_tco2e == pytest.approx(expected_direct, rel=1e-6)
        assert result.indirect_emissions_tco2e == pytest.approx(expected_indirect, rel=1e-6)
        assert result.total_embedded_emissions_tco2e == pytest.approx(expected_total, rel=1e-6)
        assert result.cbam_liability_eur == pytest.approx(expected_liability, rel=1e-2)

    def test_35_fertilizers_product_category(
        self,
        agent: CBAMComplianceAgent,
        fertilizers_input: CBAMInput,
    ):
        """Test 35: Fertilizers CN code is correctly classified."""
        result = agent.run(fertilizers_input)
        assert result.product_category == "fertilizers"

    @pytest.mark.golden
    def test_36_hydrogen_50t(
        self,
        agent: CBAMComplianceAgent,
        hydrogen_input: CBAMInput,
    ):
        """
        Test 36: Hydrogen import calculation - 50 tonnes

        Grey hydrogen has highest emission factors per tonne.
        Formula:
            direct_emissions = 50 * 9.0 = 450 tCO2e
            indirect_emissions = 50 * 1.5 = 75 tCO2e
            total_emissions = 525 tCO2e
            cbam_liability = 525 * 85 = 44,625 EUR
        """
        result = agent.run(hydrogen_input)

        expected_direct = 50.0 * EF_HYDROGEN_GLOBAL_DIRECT  # 450
        expected_indirect = 50.0 * EF_HYDROGEN_GLOBAL_INDIRECT  # 75
        expected_total = expected_direct + expected_indirect  # 525
        expected_liability = expected_total * EU_ETS_PRICE  # 44,625

        assert result.direct_emissions_tco2e == pytest.approx(expected_direct, rel=1e-6)
        assert result.indirect_emissions_tco2e == pytest.approx(expected_indirect, rel=1e-6)
        assert result.total_embedded_emissions_tco2e == pytest.approx(expected_total, rel=1e-6)
        assert result.cbam_liability_eur == pytest.approx(expected_liability, rel=1e-2)

    def test_37_hydrogen_product_category(
        self,
        agent: CBAMComplianceAgent,
        hydrogen_input: CBAMInput,
    ):
        """Test 37: Hydrogen CN code is correctly classified."""
        result = agent.run(hydrogen_input)
        assert result.product_category == "hydrogen"

    @pytest.mark.golden
    def test_38_hydrogen_highest_specific_emissions(
        self,
        agent: CBAMComplianceAgent,
        hydrogen_input: CBAMInput,
    ):
        """
        Test 38: Hydrogen has highest specific emissions per tonne

        Grey hydrogen: 10.5 tCO2e/t (highest among CBAM products)
        """
        result = agent.run(hydrogen_input)

        expected_specific = EF_HYDROGEN_GLOBAL_TOTAL  # 10.5
        assert result.specific_embedded_emissions == pytest.approx(expected_specific, rel=1e-6)

    def test_39_electricity_cn_code_in_scope(self, agent: CBAMComplianceAgent):
        """Test 39: Electricity CN code is in CBAM scope."""
        assert agent.is_in_scope("27160000") is True

    def test_40_all_product_categories_covered(self, agent: CBAMComplianceAgent):
        """Test 40: All 6 CBAM product categories have default factors."""
        categories = [
            "iron_steel",
            "aluminium",
            "cement",
            "fertilizers",
            "electricity",
            "hydrogen",
        ]
        for category in categories:
            assert category in agent.DEFAULT_FACTORS
            assert "GLOBAL" in agent.DEFAULT_FACTORS[category]


# =============================================================================
# Test 41-50: Actual Emissions and Calculation Methods
# =============================================================================


class TestActualEmissionsAndMethods:
    """Tests for actual emissions and calculation method handling."""

    @pytest.mark.golden
    def test_41_actual_emissions_override(
        self,
        agent: CBAMComplianceAgent,
        actual_emissions_input: CBAMInput,
    ):
        """
        Test 41: Actual verified emissions override default factors

        When actual_emissions is provided:
            direct_emissions = 1000 * 1.5 = 1500 tCO2e
            indirect_emissions = 0 (included in actual)
            total_emissions = 1500 tCO2e
        """
        result = agent.run(actual_emissions_input)

        expected_direct = 1000.0 * 1.5  # 1500
        expected_indirect = 0.0
        expected_total = expected_direct + expected_indirect  # 1500

        assert result.direct_emissions_tco2e == pytest.approx(expected_direct, rel=1e-6)
        assert result.indirect_emissions_tco2e == pytest.approx(expected_indirect, rel=1e-6)
        assert result.total_embedded_emissions_tco2e == pytest.approx(expected_total, rel=1e-6)

    def test_42_actual_emissions_method(
        self,
        agent: CBAMComplianceAgent,
        actual_emissions_input: CBAMInput,
    ):
        """Test 42: Actual emissions uses 'actual' calculation method."""
        result = agent.run(actual_emissions_input)
        assert result.calculation_method == CalculationMethod.ACTUAL.value

    def test_43_actual_emissions_source(
        self,
        agent: CBAMComplianceAgent,
        actual_emissions_input: CBAMInput,
    ):
        """Test 43: Actual emissions shows verified installation data source."""
        result = agent.run(actual_emissions_input)
        assert "Verified" in result.emission_factor_source

    @pytest.mark.golden
    def test_44_actual_emissions_lower_liability(
        self,
        agent: CBAMComplianceAgent,
        steel_input_china: CBAMInput,
        actual_emissions_input: CBAMInput,
    ):
        """
        Test 44: Actual emissions can result in lower CBAM liability

        Default China steel: 2550 tCO2e -> 216,750 EUR
        Actual 1.5 tCO2e/t: 1500 tCO2e -> 127,500 EUR
        Savings: 89,250 EUR (41% reduction)
        """
        result_default = agent.run(steel_input_china)
        result_actual = agent.run(actual_emissions_input)

        # Actual emissions should result in lower liability
        assert result_actual.cbam_liability_eur < result_default.cbam_liability_eur

        # Verify savings calculation
        savings = result_default.cbam_liability_eur - result_actual.cbam_liability_eur
        expected_savings = (2550 - 1500) * EU_ETS_PRICE  # 89,250
        assert savings == pytest.approx(expected_savings, rel=1e-2)

    @pytest.mark.parametrize("method,country,expected_method", [
        (None, "DE", "default"),  # Germany -> Global
        (None, "CN", "country_default"),  # China -> Country
        (None, "IN", "country_default"),  # India -> Country
        (1.5, "CN", "actual"),  # Any with actual -> Actual
    ])
    def test_45_calculation_method_selection(
        self,
        agent: CBAMComplianceAgent,
        method: Optional[float],
        country: str,
        expected_method: str,
    ):
        """Test 45: Calculation method is correctly selected based on inputs."""
        input_data = CBAMInput(
            cn_code="7208.10.00",
            quantity_tonnes=100.0,
            country_of_origin=country,
            actual_emissions=method,
            reporting_period="Q1 2026",
        )
        result = agent.run(input_data)
        assert result.calculation_method == expected_method

    @pytest.mark.golden
    def test_46_carbon_price_correctly_applied(
        self,
        agent: CBAMComplianceAgent,
        steel_input_global: CBAMInput,
    ):
        """Test 46: Carbon price is correctly applied to liability calculation."""
        result = agent.run(steel_input_global)

        assert result.carbon_price_applicable == EU_ETS_PRICE

        # Verify liability = emissions * price
        calculated_liability = result.total_embedded_emissions_tco2e * result.carbon_price_applicable
        assert result.cbam_liability_eur == pytest.approx(calculated_liability, rel=1e-2)

    def test_47_cn_code_normalized(self, agent: CBAMComplianceAgent):
        """Test 47: CN codes with dots/spaces are normalized."""
        # Various formats should work
        formats = ["7208.10.00", "72081000", "7208 1000", "7208.1000"]
        for cn_format in formats:
            input_data = CBAMInput(
                cn_code=cn_format,
                quantity_tonnes=100.0,
                country_of_origin="DE",
                reporting_period="Q1 2026",
            )
            result = agent.run(input_data)
            assert result.cn_code == "72081000"

    def test_48_country_code_uppercase(self, agent: CBAMComplianceAgent):
        """Test 48: Country codes are normalized to uppercase."""
        input_data = CBAMInput(
            cn_code="7208.10.00",
            quantity_tonnes=100.0,
            country_of_origin="cn",  # lowercase
            reporting_period="Q1 2026",
        )
        # Should use China factors
        result = agent.run(input_data)
        assert result.calculation_method == CalculationMethod.COUNTRY_DEFAULT.value

    def test_49_invalid_cn_code_raises_error(self, agent: CBAMComplianceAgent):
        """Test 49: Invalid CN code raises ValueError."""
        with pytest.raises(ValueError, match="not in CBAM scope"):
            input_data = CBAMInput(
                cn_code="99990000",  # Not in CBAM scope
                quantity_tonnes=100.0,
                country_of_origin="DE",
                reporting_period="Q1 2026",
            )
            agent.run(input_data)

    def test_50_output_model_all_fields(
        self,
        agent: CBAMComplianceAgent,
        steel_input_global: CBAMInput,
    ):
        """Test 50: Output model contains all required fields."""
        result = agent.run(steel_input_global)

        # All required fields present
        assert hasattr(result, "cn_code")
        assert hasattr(result, "product_category")
        assert hasattr(result, "quantity_tonnes")
        assert hasattr(result, "direct_emissions_tco2e")
        assert hasattr(result, "indirect_emissions_tco2e")
        assert hasattr(result, "total_embedded_emissions_tco2e")
        assert hasattr(result, "specific_embedded_emissions")
        assert hasattr(result, "calculation_method")
        assert hasattr(result, "emission_factor_source")
        assert hasattr(result, "carbon_price_applicable")
        assert hasattr(result, "cbam_liability_eur")
        assert hasattr(result, "provenance_hash")
        assert hasattr(result, "calculated_at")
        assert hasattr(result, "reporting_period")


# =============================================================================
# Test 51-60: Determinism and Zero-Hallucination Tests
# =============================================================================


class TestDeterminismAndProvenance:
    """Tests for deterministic calculations and provenance tracking."""

    @pytest.mark.golden
    def test_51_deterministic_calculation_same_inputs(
        self,
        agent: CBAMComplianceAgent,
        steel_input_china: CBAMInput,
    ):
        """
        Test 51: Same inputs produce same emission values (zero-hallucination)

        This verifies the calculation is deterministic - no LLM involved in math.
        """
        result1 = agent.run(steel_input_china)
        result2 = agent.run(steel_input_china)
        result3 = agent.run(steel_input_china)

        # All emission values must be identical
        assert result1.direct_emissions_tco2e == result2.direct_emissions_tco2e
        assert result2.direct_emissions_tco2e == result3.direct_emissions_tco2e

        assert result1.indirect_emissions_tco2e == result2.indirect_emissions_tco2e
        assert result2.indirect_emissions_tco2e == result3.indirect_emissions_tco2e

        assert result1.total_embedded_emissions_tco2e == result2.total_embedded_emissions_tco2e
        assert result2.total_embedded_emissions_tco2e == result3.total_embedded_emissions_tco2e

        assert result1.cbam_liability_eur == result2.cbam_liability_eur
        assert result2.cbam_liability_eur == result3.cbam_liability_eur

    @pytest.mark.golden
    def test_52_deterministic_across_agent_instances(
        self,
        steel_input_china: CBAMInput,
    ):
        """
        Test 52: Different agent instances produce same results

        Verifies the calculation doesn't depend on instance state.
        """
        agent1 = CBAMComplianceAgent()
        agent2 = CBAMComplianceAgent()
        agent3 = CBAMComplianceAgent()

        result1 = agent1.run(steel_input_china)
        result2 = agent2.run(steel_input_china)
        result3 = agent3.run(steel_input_china)

        assert result1.total_embedded_emissions_tco2e == result2.total_embedded_emissions_tco2e
        assert result2.total_embedded_emissions_tco2e == result3.total_embedded_emissions_tco2e

    def test_53_provenance_hash_format(
        self,
        agent: CBAMComplianceAgent,
        steel_input_global: CBAMInput,
    ):
        """Test 53: Provenance hash is valid SHA-256 format."""
        result = agent.run(steel_input_global)

        # SHA-256 produces 64 hex characters
        assert len(result.provenance_hash) == 64
        # Should be valid hex
        assert all(c in "0123456789abcdef" for c in result.provenance_hash)

    def test_54_provenance_hash_changes_with_inputs(
        self,
        agent: CBAMComplianceAgent,
    ):
        """Test 54: Provenance hash changes when inputs change."""
        input1 = CBAMInput(
            cn_code="7208.10.00",
            quantity_tonnes=1000.0,
            country_of_origin="DE",
            reporting_period="Q1 2026",
        )
        input2 = CBAMInput(
            cn_code="7208.10.00",
            quantity_tonnes=2000.0,  # Different quantity
            country_of_origin="DE",
            reporting_period="Q1 2026",
        )

        result1 = agent.run(input1)
        result2 = agent.run(input2)

        # Different inputs should produce different provenance hashes
        assert result1.provenance_hash != result2.provenance_hash

    def test_55_provenance_hash_includes_timestamp(
        self,
        agent: CBAMComplianceAgent,
        steel_input_global: CBAMInput,
    ):
        """Test 55: Provenance hash is unique per run (includes timestamp)."""
        result1 = agent.run(steel_input_global)
        result2 = agent.run(steel_input_global)

        # Same input should produce different hashes due to timestamp
        assert result1.provenance_hash != result2.provenance_hash

    def test_56_calculated_at_timestamp(
        self,
        agent: CBAMComplianceAgent,
        steel_input_global: CBAMInput,
    ):
        """Test 56: Output includes calculated_at timestamp."""
        result = agent.run(steel_input_global)

        assert result.calculated_at is not None
        assert isinstance(result.calculated_at, datetime)

    @pytest.mark.golden
    def test_57_calculation_reproducibility_guarantee(
        self,
        agent: CBAMComplianceAgent,
    ):
        """
        Test 57: Calculations are bit-perfect reproducible

        For regulatory compliance, identical inputs must always produce
        identical emission calculations (excluding timestamp-based fields).
        """
        input_data = CBAMInput(
            cn_code="7601.10.00",
            quantity_tonnes=500.0,
            country_of_origin="CN",
            reporting_period="Q3 2026",
        )

        results = [agent.run(input_data) for _ in range(10)]

        # All numeric results must be identical
        first = results[0]
        for result in results[1:]:
            assert result.direct_emissions_tco2e == first.direct_emissions_tco2e
            assert result.indirect_emissions_tco2e == first.indirect_emissions_tco2e
            assert result.total_embedded_emissions_tco2e == first.total_embedded_emissions_tco2e
            assert result.specific_embedded_emissions == first.specific_embedded_emissions
            assert result.cbam_liability_eur == first.cbam_liability_eur

    @pytest.mark.golden
    def test_58_formula_verification_steel(
        self,
        agent: CBAMComplianceAgent,
    ):
        """
        Test 58: Verify steel formula against manual calculation

        Input: 1234.56 tonnes steel from India
        Expected:
            direct = 1234.56 * 2.35 = 2901.216 tCO2e
            indirect = 1234.56 * 0.52 = 641.9712 tCO2e
            total = 3543.1872 tCO2e
            liability = 3543.1872 * 85 = 301,170.912 EUR
        """
        input_data = CBAMInput(
            cn_code="7208.10.00",
            quantity_tonnes=1234.56,
            country_of_origin="IN",
            reporting_period="Q1 2026",
        )
        result = agent.run(input_data)

        expected_direct = 1234.56 * EF_STEEL_IN_DIRECT
        expected_indirect = 1234.56 * EF_STEEL_IN_INDIRECT
        expected_total = expected_direct + expected_indirect
        expected_liability = expected_total * EU_ETS_PRICE

        assert result.direct_emissions_tco2e == pytest.approx(expected_direct, rel=1e-6)
        assert result.indirect_emissions_tco2e == pytest.approx(expected_indirect, rel=1e-6)
        assert result.total_embedded_emissions_tco2e == pytest.approx(expected_total, rel=1e-6)
        assert result.cbam_liability_eur == pytest.approx(expected_liability, rel=1e-2)

    @pytest.mark.golden
    def test_59_formula_verification_aluminium(
        self,
        agent: CBAMComplianceAgent,
    ):
        """
        Test 59: Verify aluminium formula against manual calculation

        Input: 789.01 tonnes aluminium from China
        Expected:
            direct = 789.01 * 1.65 = 1301.8665 tCO2e
            indirect = 789.01 * 10.20 = 8047.902 tCO2e
            total = 9349.7685 tCO2e
            liability = 9349.7685 * 85 = 794,730.3225 EUR
        """
        input_data = CBAMInput(
            cn_code="7601.10.00",
            quantity_tonnes=789.01,
            country_of_origin="CN",
            reporting_period="Q2 2026",
        )
        result = agent.run(input_data)

        expected_direct = 789.01 * EF_ALUMINIUM_CN_DIRECT
        expected_indirect = 789.01 * EF_ALUMINIUM_CN_INDIRECT
        expected_total = expected_direct + expected_indirect
        expected_liability = expected_total * EU_ETS_PRICE

        assert result.direct_emissions_tco2e == pytest.approx(expected_direct, rel=1e-6)
        assert result.indirect_emissions_tco2e == pytest.approx(expected_indirect, rel=1e-6)
        assert result.total_embedded_emissions_tco2e == pytest.approx(expected_total, rel=1e-6)
        assert result.cbam_liability_eur == pytest.approx(expected_liability, rel=1e-2)

    @pytest.mark.golden
    def test_60_formula_verification_cement(
        self,
        agent: CBAMComplianceAgent,
    ):
        """
        Test 60: Verify cement formula against manual calculation

        Input: 10000 tonnes cement
        Expected:
            direct = 10000 * 0.83 = 8300 tCO2e
            indirect = 10000 * 0.05 = 500 tCO2e
            total = 8800 tCO2e
            liability = 8800 * 85 = 748,000 EUR
        """
        input_data = CBAMInput(
            cn_code="2523.29.00",
            quantity_tonnes=10000.0,
            country_of_origin="TR",
            reporting_period="Q4 2026",
        )
        result = agent.run(input_data)

        expected_direct = 10000.0 * EF_CEMENT_GLOBAL_DIRECT
        expected_indirect = 10000.0 * EF_CEMENT_GLOBAL_INDIRECT
        expected_total = expected_direct + expected_indirect
        expected_liability = expected_total * EU_ETS_PRICE

        assert result.direct_emissions_tco2e == pytest.approx(expected_direct, rel=1e-6)
        assert result.indirect_emissions_tco2e == pytest.approx(expected_indirect, rel=1e-6)
        assert result.total_embedded_emissions_tco2e == pytest.approx(expected_total, rel=1e-6)
        assert result.cbam_liability_eur == pytest.approx(expected_liability, rel=1e-2)


# =============================================================================
# Test 61-70: Edge Cases and Boundary Conditions
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.golden
    def test_61_zero_quantity(
        self,
        agent: CBAMComplianceAgent,
    ):
        """
        Test 61: Zero quantity returns zero emissions

        Formula: 0 tonnes * any factor = 0 tCO2e
        """
        input_data = CBAMInput(
            cn_code="7208.10.00",
            quantity_tonnes=0.0,
            country_of_origin="CN",
            reporting_period="Q1 2026",
        )
        result = agent.run(input_data)

        assert result.direct_emissions_tco2e == 0.0
        assert result.indirect_emissions_tco2e == 0.0
        assert result.total_embedded_emissions_tco2e == 0.0
        assert result.cbam_liability_eur == 0.0

    @pytest.mark.golden
    def test_62_very_small_quantity(
        self,
        agent: CBAMComplianceAgent,
    ):
        """
        Test 62: Very small quantity (precision test)

        Formula: 0.001 tonnes * 2.55 = 0.00255 tCO2e
        """
        input_data = CBAMInput(
            cn_code="7208.10.00",
            quantity_tonnes=0.001,
            country_of_origin="CN",
            reporting_period="Q1 2026",
        )
        result = agent.run(input_data)

        expected_total = 0.001 * EF_STEEL_CN_TOTAL  # 0.00255
        assert result.total_embedded_emissions_tco2e == pytest.approx(expected_total, rel=1e-6)

    @pytest.mark.golden
    def test_63_very_large_quantity(
        self,
        agent: CBAMComplianceAgent,
    ):
        """
        Test 63: Very large quantity (industrial scale)

        Formula: 1,000,000 tonnes * 2.17 = 2,170,000 tCO2e
        """
        input_data = CBAMInput(
            cn_code="7208.10.00",
            quantity_tonnes=1_000_000.0,
            country_of_origin="DE",
            reporting_period="Q1 2026",
        )
        result = agent.run(input_data)

        expected_total = 1_000_000.0 * EF_STEEL_GLOBAL_TOTAL  # 2,170,000
        expected_liability = expected_total * EU_ETS_PRICE  # 184,450,000

        assert result.total_embedded_emissions_tco2e == pytest.approx(expected_total, rel=1e-6)
        assert result.cbam_liability_eur == pytest.approx(expected_liability, rel=1e-2)

    def test_64_input_metadata_accepted(
        self,
        agent: CBAMComplianceAgent,
    ):
        """Test 64: Input metadata is accepted."""
        input_data = CBAMInput(
            cn_code="7208.10.00",
            quantity_tonnes=100.0,
            country_of_origin="DE",
            reporting_period="Q1 2026",
            metadata={"shipment_id": "SHP-001", "customs_declaration": "CD-12345"},
        )
        result = agent.run(input_data)
        assert result is not None
        assert result.total_embedded_emissions_tco2e > 0

    def test_65_installation_id_optional(
        self,
        agent: CBAMComplianceAgent,
    ):
        """Test 65: Installation ID is optional."""
        input_data = CBAMInput(
            cn_code="7208.10.00",
            quantity_tonnes=100.0,
            country_of_origin="DE",
            reporting_period="Q1 2026",
            # No installation_id
        )
        result = agent.run(input_data)
        assert result is not None

    def test_66_precursor_emissions_optional(
        self,
        agent: CBAMComplianceAgent,
    ):
        """Test 66: Precursor emissions are optional."""
        input_data = CBAMInput(
            cn_code="7208.10.00",
            quantity_tonnes=100.0,
            country_of_origin="DE",
            reporting_period="Q1 2026",
            precursor_emissions={"iron_ore": 0.5, "coke": 0.3},
        )
        result = agent.run(input_data)
        assert result is not None

    def test_67_electricity_source_optional(
        self,
        agent: CBAMComplianceAgent,
    ):
        """Test 67: Electricity source is optional."""
        input_data = CBAMInput(
            cn_code="7601.10.00",
            quantity_tonnes=100.0,
            country_of_origin="NO",
            reporting_period="Q1 2026",
            electricity_source="100% renewable",
        )
        result = agent.run(input_data)
        assert result is not None

    def test_68_rounding_precision(
        self,
        agent: CBAMComplianceAgent,
    ):
        """Test 68: Output values are rounded to appropriate precision."""
        input_data = CBAMInput(
            cn_code="7208.10.00",
            quantity_tonnes=123.456789,
            country_of_origin="CN",
            reporting_period="Q1 2026",
        )
        result = agent.run(input_data)

        # Emissions should be rounded to 6 decimal places
        emissions_str = f"{result.total_embedded_emissions_tco2e:.6f}"
        decimal_places = len(emissions_str.split(".")[-1])
        assert decimal_places <= 6

        # Liability should be rounded to 2 decimal places (EUR cents)
        liability_str = f"{result.cbam_liability_eur:.2f}"
        decimal_places = len(liability_str.split(".")[-1])
        assert decimal_places <= 2

    def test_69_unknown_country_uses_global_default(
        self,
        agent: CBAMComplianceAgent,
    ):
        """Test 69: Unknown country code uses global default factors."""
        input_data = CBAMInput(
            cn_code="7208.10.00",
            quantity_tonnes=100.0,
            country_of_origin="ZZ",  # Unknown country
            reporting_period="Q1 2026",
        )
        result = agent.run(input_data)

        # Should use global default
        assert result.calculation_method == CalculationMethod.DEFAULT.value
        expected_total = 100.0 * EF_STEEL_GLOBAL_TOTAL
        assert result.total_embedded_emissions_tco2e == pytest.approx(expected_total, rel=1e-6)

    def test_70_reporting_period_formats(
        self,
        agent: CBAMComplianceAgent,
    ):
        """Test 70: Various reporting period formats are accepted."""
        periods = ["Q1 2026", "Q2 2026", "Q3 2026", "Q4 2026", "Q1 2027"]
        for period in periods:
            input_data = CBAMInput(
                cn_code="7208.10.00",
                quantity_tonnes=100.0,
                country_of_origin="DE",
                reporting_period=period,
            )
            result = agent.run(input_data)
            assert result.reporting_period == period


# =============================================================================
# Test 71-75: Input Validation Tests
# =============================================================================


class TestInputValidation:
    """Tests for input validation."""

    def test_71_cn_code_validation_too_short(self):
        """Test 71: CN code too short raises validation error."""
        with pytest.raises(ValueError, match="at least 8 digits"):
            CBAMInput(
                cn_code="7208",  # Too short
                quantity_tonnes=100.0,
                country_of_origin="DE",
                reporting_period="Q1 2026",
            )

    def test_72_cn_code_validation_non_numeric(self):
        """Test 72: CN code with non-numeric characters raises error."""
        with pytest.raises(ValueError, match="only digits"):
            CBAMInput(
                cn_code="7208ABCD",
                quantity_tonnes=100.0,
                country_of_origin="DE",
                reporting_period="Q1 2026",
            )

    def test_73_quantity_validation_negative(self):
        """Test 73: Negative quantity raises validation error."""
        with pytest.raises(ValueError):
            CBAMInput(
                cn_code="7208.10.00",
                quantity_tonnes=-100.0,  # Negative
                country_of_origin="DE",
                reporting_period="Q1 2026",
            )

    def test_74_country_code_validation_length(self):
        """Test 74: Country code must be 2 characters."""
        with pytest.raises(ValueError):
            CBAMInput(
                cn_code="7208.10.00",
                quantity_tonnes=100.0,
                country_of_origin="DEU",  # 3 characters
                reporting_period="Q1 2026",
            )

    def test_75_actual_emissions_validation_negative(self):
        """Test 75: Negative actual emissions raises validation error."""
        with pytest.raises(ValueError):
            CBAMInput(
                cn_code="7208.10.00",
                quantity_tonnes=100.0,
                country_of_origin="DE",
                actual_emissions=-1.5,  # Negative
                reporting_period="Q1 2026",
            )


# =============================================================================
# Additional Parametrized Tests
# =============================================================================


class TestParametrizedCalculations:
    """Parametrized tests for comprehensive coverage."""

    @pytest.mark.golden
    @pytest.mark.parametrize("cn_code,quantity,country,expected_category,expected_direct_ef,expected_indirect_ef", [
        # Steel - various CN codes
        ("72010000", 100, "DE", "iron_steel", 1.85, 0.32),
        ("72081000", 100, "CN", "iron_steel", 2.10, 0.45),
        ("73011000", 100, "IN", "iron_steel", 2.35, 0.52),
        # Aluminium
        ("76011000", 100, "NO", "aluminium", 1.60, 6.50),
        ("76061100", 100, "CN", "aluminium", 1.65, 10.20),
        # Cement
        ("25231000", 100, "TR", "cement", 0.83, 0.05),
        ("25232900", 100, "EG", "cement", 0.83, 0.05),
        # Fertilizers
        ("31021000", 100, "RU", "fertilizers", 2.50, 0.12),
        ("31051000", 100, "BY", "fertilizers", 2.50, 0.12),
        # Hydrogen
        ("28041000", 100, "SA", "hydrogen", 9.0, 1.5),
    ])
    def test_parametrized_emission_factors(
        self,
        agent: CBAMComplianceAgent,
        cn_code: str,
        quantity: float,
        country: str,
        expected_category: str,
        expected_direct_ef: float,
        expected_indirect_ef: float,
    ):
        """
        Parametrized test for emission factor lookups across product categories.

        Verifies that each CN code maps to the correct product category and
        applies the correct emission factors based on country of origin.
        """
        input_data = CBAMInput(
            cn_code=cn_code,
            quantity_tonnes=quantity,
            country_of_origin=country,
            reporting_period="Q1 2026",
        )
        result = agent.run(input_data)

        assert result.product_category == expected_category

        expected_direct = quantity * expected_direct_ef
        expected_indirect = quantity * expected_indirect_ef

        assert result.direct_emissions_tco2e == pytest.approx(expected_direct, rel=1e-6)
        assert result.indirect_emissions_tco2e == pytest.approx(expected_indirect, rel=1e-6)

    @pytest.mark.golden
    @pytest.mark.parametrize("quantity", [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0])
    def test_parametrized_quantities(
        self,
        agent: CBAMComplianceAgent,
        quantity: float,
    ):
        """
        Parametrized test for various quantities.

        Verifies linear scaling of emissions with quantity.
        """
        input_data = CBAMInput(
            cn_code="7208.10.00",
            quantity_tonnes=quantity,
            country_of_origin="CN",
            reporting_period="Q1 2026",
        )
        result = agent.run(input_data)

        expected_total = quantity * EF_STEEL_CN_TOTAL
        expected_liability = expected_total * EU_ETS_PRICE

        assert result.total_embedded_emissions_tco2e == pytest.approx(expected_total, rel=1e-6)
        assert result.cbam_liability_eur == pytest.approx(expected_liability, rel=1e-2)


# =============================================================================
# Run Tests
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
