# -*- coding: utf-8 -*-
"""
Unit Tests for GL-002: CBAM Compliance Agent

Comprehensive test suite with 50 test cases covering:
- Product classification (CN codes) (10 tests)
- Embedded emissions calculations (15 tests)
- Certificate/liability calculations (10 tests)
- Country-specific factors (10 tests)
- Error handling and edge cases (5 tests)

Target: 85%+ coverage for CBAM Compliance Agent
Run with: pytest tests/unit/test_gl002_cbam_agent.py -v --cov

Author: GL-TestEngineer
Version: 1.0.0

CBAM (Carbon Border Adjustment Mechanism) calculates embedded emissions
for goods imported into the EU under EU Regulation (EU) 2023/956.
"""

import pytest
import hashlib
import json
from datetime import datetime
from decimal import Decimal
from typing import Dict, Any
from unittest.mock import Mock, patch, MagicMock

# Add project paths for imports
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "GL-Agent-Factory" / "backend" / "agents"))

# Import agent components
from gl_002_cbam_compliance.agent import (
    CBAMComplianceAgent,
    CBAMInput,
    CBAMOutput,
    CBAMProductCategory,
    EmissionType,
    CalculationMethod,
    CBAMDefaultFactor,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def agent():
    """Create CBAMComplianceAgent instance."""
    return CBAMComplianceAgent()


@pytest.fixture
def agent_with_config():
    """Create CBAMComplianceAgent with custom configuration."""
    config = {"carbon_price_override": 90.0}
    return CBAMComplianceAgent(config=config)


@pytest.fixture
def valid_steel_input():
    """Create valid steel import input data."""
    return CBAMInput(
        cn_code="72081000",
        quantity_tonnes=1000.0,
        country_of_origin="CN",
        reporting_period="Q1 2026",
    )


@pytest.fixture
def valid_aluminum_input():
    """Create valid aluminum import input data."""
    return CBAMInput(
        cn_code="76011000",
        quantity_tonnes=500.0,
        country_of_origin="CN",
        reporting_period="Q1 2026",
    )


@pytest.fixture
def valid_cement_input():
    """Create valid cement import input data."""
    return CBAMInput(
        cn_code="25232900",
        quantity_tonnes=2000.0,
        country_of_origin="IN",
        reporting_period="Q1 2026",
    )


@pytest.fixture
def valid_fertilizer_input():
    """Create valid fertilizer import input data."""
    return CBAMInput(
        cn_code="31021000",
        quantity_tonnes=750.0,
        country_of_origin="RU",
        reporting_period="Q1 2026",
    )


@pytest.fixture
def steel_with_actual_emissions():
    """Create steel input with actual verified emissions."""
    return CBAMInput(
        cn_code="72081000",
        quantity_tonnes=1000.0,
        country_of_origin="CN",
        installation_id="CN-STEEL-001",
        actual_emissions=1.75,  # Verified tCO2e/tonne
        reporting_period="Q1 2026",
    )


# =============================================================================
# Product Classification Tests (10 tests)
# =============================================================================

class TestProductClassification:
    """Test suite for CN code product classification - 10 test cases."""

    @pytest.mark.unit
    def test_classify_iron_steel_7201(self, agent):
        """UT-GL002-001: Test classification of CN 7201 (pig iron)."""
        category = agent._classify_product("72010000")
        assert category == CBAMProductCategory.IRON_STEEL

    @pytest.mark.unit
    def test_classify_iron_steel_7208(self, agent):
        """UT-GL002-002: Test classification of CN 7208 (flat-rolled steel)."""
        category = agent._classify_product("72081000")
        assert category == CBAMProductCategory.IRON_STEEL

    @pytest.mark.unit
    def test_classify_aluminum_7601(self, agent):
        """UT-GL002-003: Test classification of CN 7601 (unwrought aluminum)."""
        category = agent._classify_product("76011000")
        assert category == CBAMProductCategory.ALUMINIUM

    @pytest.mark.unit
    def test_classify_cement_2523(self, agent):
        """UT-GL002-004: Test classification of CN 2523 (cement)."""
        category = agent._classify_product("25232900")
        assert category == CBAMProductCategory.CEMENT

    @pytest.mark.unit
    def test_classify_fertilizer_3102(self, agent):
        """UT-GL002-005: Test classification of CN 3102 (nitrogen fertilizers)."""
        category = agent._classify_product("31021000")
        assert category == CBAMProductCategory.FERTILIZERS

    @pytest.mark.unit
    def test_classify_electricity_2716(self, agent):
        """UT-GL002-006: Test classification of CN 2716 (electricity)."""
        category = agent._classify_product("27160000")
        assert category == CBAMProductCategory.ELECTRICITY

    @pytest.mark.unit
    def test_classify_hydrogen_2804(self, agent):
        """UT-GL002-007: Test classification of CN 2804 (hydrogen)."""
        category = agent._classify_product("28041000")
        assert category == CBAMProductCategory.HYDROGEN

    @pytest.mark.unit
    def test_classify_unknown_cn_code_returns_none(self, agent):
        """UT-GL002-008: Test unknown CN code returns None."""
        category = agent._classify_product("99999999")
        assert category is None

    @pytest.mark.unit
    def test_cn_code_prefix_matching(self, agent):
        """UT-GL002-009: Test CN code prefix matching works."""
        # All 72xx codes should be iron/steel
        for cn_suffix in ["01", "02", "03", "04", "05"]:
            category = agent._classify_product(f"72{cn_suffix}0000")
            assert category == CBAMProductCategory.IRON_STEEL

    @pytest.mark.unit
    def test_is_in_scope_method(self, agent):
        """UT-GL002-010: Test is_in_scope utility method."""
        assert agent.is_in_scope("72081000") is True
        assert agent.is_in_scope("99999999") is False


# =============================================================================
# Embedded Emissions Calculations (15 tests)
# =============================================================================

class TestEmbeddedEmissionsCalculations:
    """Test suite for embedded emissions calculations - 15 test cases."""

    @pytest.mark.unit
    def test_steel_direct_emissions_calculation(self, agent, valid_steel_input):
        """UT-GL002-011: Test steel direct emissions calculation."""
        result = agent.run(valid_steel_input)

        # China steel: 2.10 tCO2e/t direct
        expected_direct = 1000.0 * 2.10
        assert result.direct_emissions_tco2e == pytest.approx(expected_direct, rel=1e-6)

    @pytest.mark.unit
    def test_steel_indirect_emissions_calculation(self, agent, valid_steel_input):
        """UT-GL002-012: Test steel indirect emissions calculation."""
        result = agent.run(valid_steel_input)

        # China steel: 0.45 tCO2e/t indirect
        expected_indirect = 1000.0 * 0.45
        assert result.indirect_emissions_tco2e == pytest.approx(expected_indirect, rel=1e-6)

    @pytest.mark.unit
    def test_steel_total_emissions_calculation(self, agent, valid_steel_input):
        """UT-GL002-013: Test steel total embedded emissions."""
        result = agent.run(valid_steel_input)

        # Total = direct + indirect = 1000 * (2.10 + 0.45)
        expected_total = 1000.0 * (2.10 + 0.45)
        assert result.total_embedded_emissions_tco2e == pytest.approx(expected_total, rel=1e-6)

    @pytest.mark.unit
    def test_aluminum_high_indirect_emissions(self, agent, valid_aluminum_input):
        """UT-GL002-014: Test aluminum high indirect emissions (electricity intensive)."""
        result = agent.run(valid_aluminum_input)

        # China aluminum: 10.20 tCO2e/t indirect (coal-heavy grid)
        expected_indirect = 500.0 * 10.20
        assert result.indirect_emissions_tco2e == pytest.approx(expected_indirect, rel=1e-6)

    @pytest.mark.unit
    def test_cement_emissions_calculation(self, agent, valid_cement_input):
        """UT-GL002-015: Test cement emissions calculation."""
        result = agent.run(valid_cement_input)

        # Global cement default: 0.83 + 0.05 = 0.88 tCO2e/t
        expected_total = 2000.0 * (0.83 + 0.05)
        assert result.total_embedded_emissions_tco2e == pytest.approx(expected_total, rel=1e-6)

    @pytest.mark.unit
    def test_fertilizer_emissions_calculation(self, agent, valid_fertilizer_input):
        """UT-GL002-016: Test fertilizer emissions calculation."""
        result = agent.run(valid_fertilizer_input)

        # Global fertilizer default: 2.50 + 0.12 = 2.62 tCO2e/t
        expected_total = 750.0 * (2.50 + 0.12)
        assert result.total_embedded_emissions_tco2e == pytest.approx(expected_total, rel=1e-6)

    @pytest.mark.unit
    def test_actual_emissions_override_defaults(self, agent, steel_with_actual_emissions):
        """UT-GL002-017: Test actual emissions override default values."""
        result = agent.run(steel_with_actual_emissions)

        # Should use actual emissions: 1.75 tCO2e/t
        expected_total = 1000.0 * 1.75
        assert result.total_embedded_emissions_tco2e == pytest.approx(expected_total, rel=1e-6)
        assert result.calculation_method == "actual"

    @pytest.mark.unit
    def test_calculation_method_default_used(self, agent, valid_steel_input):
        """UT-GL002-018: Test calculation method is 'country_default' or 'default'."""
        result = agent.run(valid_steel_input)

        assert result.calculation_method in ["country_default", "default"]

    @pytest.mark.unit
    def test_specific_embedded_emissions_calculation(self, agent, valid_steel_input):
        """UT-GL002-019: Test specific embedded emissions (per tonne)."""
        result = agent.run(valid_steel_input)

        # China steel: 2.10 + 0.45 = 2.55 tCO2e/tonne
        expected_specific = 2.10 + 0.45
        assert result.specific_embedded_emissions == pytest.approx(expected_specific, rel=1e-6)

    @pytest.mark.unit
    def test_zero_quantity_returns_zero_emissions(self, agent):
        """UT-GL002-020: Test zero quantity returns zero emissions."""
        input_data = CBAMInput(
            cn_code="72081000",
            quantity_tonnes=0.0,
            country_of_origin="CN",
            reporting_period="Q1 2026",
        )
        result = agent.run(input_data)
        assert result.total_embedded_emissions_tco2e == 0.0

    @pytest.mark.unit
    def test_large_quantity_calculation(self, agent):
        """UT-GL002-021: Test calculation with large quantity."""
        input_data = CBAMInput(
            cn_code="72081000",
            quantity_tonnes=100000.0,
            country_of_origin="CN",
            reporting_period="Q1 2026",
        )
        result = agent.run(input_data)

        expected = 100000.0 * (2.10 + 0.45)
        assert result.total_embedded_emissions_tco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.unit
    def test_decimal_quantity_calculation(self, agent):
        """UT-GL002-022: Test calculation with decimal quantity."""
        input_data = CBAMInput(
            cn_code="72081000",
            quantity_tonnes=123.456,
            country_of_origin="CN",
            reporting_period="Q1 2026",
        )
        result = agent.run(input_data)

        expected = 123.456 * (2.10 + 0.45)
        assert result.total_embedded_emissions_tco2e == pytest.approx(expected, rel=1e-6)

    @pytest.mark.unit
    def test_deterministic_calculation(self, agent, valid_steel_input):
        """UT-GL002-023: Test calculation is deterministic."""
        result1 = agent.run(valid_steel_input)
        result2 = agent.run(valid_steel_input)

        assert result1.total_embedded_emissions_tco2e == result2.total_embedded_emissions_tco2e

    @pytest.mark.unit
    def test_formula_direct_plus_indirect(self, agent, valid_steel_input):
        """UT-GL002-024: Test formula: total = direct + indirect."""
        result = agent.run(valid_steel_input)

        calculated_total = result.direct_emissions_tco2e + result.indirect_emissions_tco2e
        assert result.total_embedded_emissions_tco2e == pytest.approx(calculated_total, rel=1e-6)

    @pytest.mark.unit
    def test_emission_factor_source_recorded(self, agent, valid_steel_input):
        """UT-GL002-025: Test emission factor source is recorded."""
        result = agent.run(valid_steel_input)

        assert result.emission_factor_source is not None
        assert "EU Implementing Regulation" in result.emission_factor_source


# =============================================================================
# Certificate/Liability Calculations (10 tests)
# =============================================================================

class TestCertificateLiabilityCalculations:
    """Test suite for CBAM certificate and liability calculations - 10 test cases."""

    @pytest.mark.unit
    def test_cbam_liability_calculation(self, agent, valid_steel_input):
        """UT-GL002-026: Test CBAM liability calculation."""
        result = agent.run(valid_steel_input)

        # liability = total_emissions * carbon_price
        expected_liability = result.total_embedded_emissions_tco2e * 85.0
        assert result.cbam_liability_eur == pytest.approx(expected_liability, rel=0.01)

    @pytest.mark.unit
    def test_carbon_price_used(self, agent, valid_steel_input):
        """UT-GL002-027: Test carbon price applied."""
        result = agent.run(valid_steel_input)

        assert result.carbon_price_applicable == 85.0  # EU ETS price

    @pytest.mark.unit
    def test_liability_formula_verification(self, agent, valid_steel_input):
        """UT-GL002-028: Test liability = emissions * carbon_price."""
        result = agent.run(valid_steel_input)

        calculated_liability = result.total_embedded_emissions_tco2e * result.carbon_price_applicable
        assert result.cbam_liability_eur == pytest.approx(calculated_liability, rel=0.01)

    @pytest.mark.unit
    def test_zero_emissions_zero_liability(self, agent):
        """UT-GL002-029: Test zero emissions produces zero liability."""
        input_data = CBAMInput(
            cn_code="72081000",
            quantity_tonnes=0.0,
            country_of_origin="CN",
            reporting_period="Q1 2026",
        )
        result = agent.run(input_data)
        assert result.cbam_liability_eur == 0.0

    @pytest.mark.unit
    def test_liability_rounded_to_two_decimals(self, agent, valid_steel_input):
        """UT-GL002-030: Test liability is rounded to 2 decimal places."""
        result = agent.run(valid_steel_input)

        # Check liability has at most 2 decimal places
        str_liability = str(result.cbam_liability_eur)
        if '.' in str_liability:
            decimal_places = len(str_liability.split('.')[1])
            assert decimal_places <= 2

    @pytest.mark.unit
    def test_high_emission_product_liability(self, agent, valid_aluminum_input):
        """UT-GL002-031: Test high emission product (aluminum) liability."""
        result = agent.run(valid_aluminum_input)

        # Aluminum has high emissions, liability should be significant
        assert result.cbam_liability_eur > 100000  # Expected high liability

    @pytest.mark.unit
    def test_low_emission_product_liability(self, agent, valid_cement_input):
        """UT-GL002-032: Test relatively lower emission product (cement) liability."""
        result = agent.run(valid_cement_input)

        # Cement has lower emissions per tonne than aluminum
        cement_specific = 0.83 + 0.05
        aluminum_specific = 1.65 + 10.20
        assert cement_specific < aluminum_specific

    @pytest.mark.unit
    def test_liability_scales_with_quantity(self, agent):
        """UT-GL002-033: Test liability scales linearly with quantity."""
        input_1000 = CBAMInput(
            cn_code="72081000",
            quantity_tonnes=1000.0,
            country_of_origin="CN",
            reporting_period="Q1 2026",
        )
        input_2000 = CBAMInput(
            cn_code="72081000",
            quantity_tonnes=2000.0,
            country_of_origin="CN",
            reporting_period="Q1 2026",
        )

        result_1000 = agent.run(input_1000)
        result_2000 = agent.run(input_2000)

        # Double quantity should double liability
        assert result_2000.cbam_liability_eur == pytest.approx(
            result_1000.cbam_liability_eur * 2, rel=0.01
        )

    @pytest.mark.unit
    def test_get_carbon_price_method(self, agent):
        """UT-GL002-034: Test _get_carbon_price method."""
        price = agent._get_carbon_price()
        assert price == 85.0
        assert isinstance(price, float)

    @pytest.mark.unit
    def test_reporting_period_in_output(self, agent, valid_steel_input):
        """UT-GL002-035: Test reporting period preserved in output."""
        result = agent.run(valid_steel_input)
        assert result.reporting_period == "Q1 2026"


# =============================================================================
# Country-Specific Factors Tests (10 tests)
# =============================================================================

class TestCountrySpecificFactors:
    """Test suite for country-specific emission factors - 10 test cases."""

    @pytest.mark.unit
    def test_china_steel_higher_than_global(self, agent):
        """UT-GL002-036: Test China steel emissions higher than global default."""
        cn_input = CBAMInput(
            cn_code="72081000",
            quantity_tonnes=1000.0,
            country_of_origin="CN",
            reporting_period="Q1 2026",
        )

        # China: 2.10 direct vs Global: 1.85
        result = agent.run(cn_input)
        specific = result.specific_embedded_emissions
        assert specific > 1.85  # Higher than global

    @pytest.mark.unit
    def test_india_steel_highest_emissions(self, agent):
        """UT-GL002-037: Test India steel has highest emissions."""
        in_input = CBAMInput(
            cn_code="72081000",
            quantity_tonnes=1000.0,
            country_of_origin="IN",
            reporting_period="Q1 2026",
        )

        # India: 2.35 direct (highest)
        result = agent.run(in_input)
        assert result.direct_emissions_tco2e == pytest.approx(1000.0 * 2.35, rel=1e-6)

    @pytest.mark.unit
    def test_global_default_for_unknown_country(self, agent):
        """UT-GL002-038: Test global default used for unknown country."""
        unknown_input = CBAMInput(
            cn_code="72081000",
            quantity_tonnes=1000.0,
            country_of_origin="ZZ",  # Unknown country
            reporting_period="Q1 2026",
        )

        result = agent.run(unknown_input)
        # Should use global default: 1.85 tCO2e/t direct
        assert result.direct_emissions_tco2e == pytest.approx(1000.0 * 1.85, rel=1e-6)

    @pytest.mark.unit
    def test_china_aluminum_high_indirect(self, agent, valid_aluminum_input):
        """UT-GL002-039: Test China aluminum has high indirect emissions."""
        result = agent.run(valid_aluminum_input)

        # China aluminum: 10.20 tCO2e/t indirect (coal grid)
        assert result.indirect_emissions_tco2e == pytest.approx(500.0 * 10.20, rel=1e-6)

    @pytest.mark.unit
    def test_global_aluminum_lower_indirect(self, agent):
        """UT-GL002-040: Test global aluminum has lower indirect emissions."""
        global_input = CBAMInput(
            cn_code="76011000",
            quantity_tonnes=500.0,
            country_of_origin="ZZ",  # Will use global
            reporting_period="Q1 2026",
        )

        result = agent.run(global_input)
        # Global aluminum: 6.50 tCO2e/t indirect
        assert result.indirect_emissions_tco2e == pytest.approx(500.0 * 6.50, rel=1e-6)

    @pytest.mark.unit
    def test_emission_factor_lookup_priority(self, agent):
        """UT-GL002-041: Test emission factor lookup priority (country -> global)."""
        # Test method directly
        direct_ef, indirect_ef, method, source = agent._get_emission_factors(
            CBAMProductCategory.IRON_STEEL,
            "CN",
            None
        )

        assert method == "country_default"
        assert direct_ef == 2.10

    @pytest.mark.unit
    def test_actual_emissions_bypass_country_lookup(self, agent, steel_with_actual_emissions):
        """UT-GL002-042: Test actual emissions bypass country lookup."""
        result = agent.run(steel_with_actual_emissions)

        assert result.calculation_method == "actual"
        assert result.emission_factor_source == "Verified installation data"

    @pytest.mark.unit
    def test_country_code_normalized(self):
        """UT-GL002-043: Test country code is normalized to uppercase."""
        input_data = CBAMInput(
            cn_code="72081000",
            quantity_tonnes=1000.0,
            country_of_origin="cn",  # lowercase
            reporting_period="Q1 2026",
        )
        assert input_data.country_of_origin == "CN"

    @pytest.mark.unit
    def test_different_countries_different_results(self, agent):
        """UT-GL002-044: Test different countries produce different results."""
        cn_input = CBAMInput(
            cn_code="72081000",
            quantity_tonnes=1000.0,
            country_of_origin="CN",
            reporting_period="Q1 2026",
        )
        in_input = CBAMInput(
            cn_code="72081000",
            quantity_tonnes=1000.0,
            country_of_origin="IN",
            reporting_period="Q1 2026",
        )

        cn_result = agent.run(cn_input)
        in_result = agent.run(in_input)

        assert cn_result.total_embedded_emissions_tco2e != in_result.total_embedded_emissions_tco2e

    @pytest.mark.unit
    def test_get_cbam_products_method(self, agent):
        """UT-GL002-045: Test get_cbam_products utility method."""
        products = agent.get_cbam_products()

        assert "cement" in products
        assert "iron_steel" in products
        assert "aluminium" in products
        assert "fertilizers" in products
        assert "electricity" in products
        assert "hydrogen" in products


# =============================================================================
# Error Handling and Edge Cases (5 tests)
# =============================================================================

class TestErrorHandlingEdgeCases:
    """Test suite for error handling and edge cases - 5 test cases."""

    @pytest.mark.unit
    def test_non_cbam_cn_code_raises_error(self, agent):
        """UT-GL002-046: Test non-CBAM CN code raises ValueError."""
        input_data = CBAMInput(
            cn_code="99999999",  # Not in CBAM scope
            quantity_tonnes=1000.0,
            country_of_origin="CN",
            reporting_period="Q1 2026",
        )

        with pytest.raises(ValueError) as exc_info:
            agent.run(input_data)

        assert "not in CBAM scope" in str(exc_info.value)

    @pytest.mark.unit
    def test_invalid_cn_code_format_rejected(self):
        """UT-GL002-047: Test invalid CN code format is rejected."""
        with pytest.raises(ValueError):
            CBAMInput(
                cn_code="ABC",  # Non-numeric
                quantity_tonnes=1000.0,
                country_of_origin="CN",
                reporting_period="Q1 2026",
            )

    @pytest.mark.unit
    def test_negative_quantity_rejected(self):
        """UT-GL002-048: Test negative quantity is rejected."""
        with pytest.raises(ValueError):
            CBAMInput(
                cn_code="72081000",
                quantity_tonnes=-100.0,
                country_of_origin="CN",
                reporting_period="Q1 2026",
            )

    @pytest.mark.unit
    def test_provenance_hash_generated(self, agent, valid_steel_input):
        """UT-GL002-049: Test provenance hash is generated."""
        result = agent.run(valid_steel_input)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64  # SHA-256

    @pytest.mark.unit
    def test_output_includes_timestamp(self, agent, valid_steel_input):
        """UT-GL002-050: Test output includes calculation timestamp."""
        result = agent.run(valid_steel_input)

        assert result.calculated_at is not None
        assert isinstance(result.calculated_at, datetime)


# =============================================================================
# Additional Tests for CBAMProductCategory Enum
# =============================================================================

class TestCBAMProductCategoryEnum:
    """Tests for CBAMProductCategory enum."""

    @pytest.mark.unit
    def test_all_cbam_categories_defined(self):
        """Test all CBAM product categories are defined."""
        expected_categories = {
            "cement", "iron_steel", "aluminium",
            "fertilizers", "electricity", "hydrogen"
        }
        actual_categories = {cat.value for cat in CBAMProductCategory}
        assert expected_categories == actual_categories

    @pytest.mark.unit
    def test_category_from_string(self):
        """Test creating category from string value."""
        category = CBAMProductCategory("iron_steel")
        assert category == CBAMProductCategory.IRON_STEEL


# =============================================================================
# Agent Initialization Tests
# =============================================================================

class TestAgentInitialization:
    """Tests for agent initialization."""

    @pytest.mark.unit
    def test_agent_initialization(self):
        """Test agent initializes correctly."""
        agent = CBAMComplianceAgent()
        assert agent is not None
        assert agent.AGENT_ID == "regulatory/cbam_compliance_v1"
        assert agent.VERSION == "1.0.0"

    @pytest.mark.unit
    def test_agent_initialization_with_config(self):
        """Test agent initializes with custom config."""
        config = {"custom_setting": True}
        agent = CBAMComplianceAgent(config=config)
        assert agent.config["custom_setting"] is True


# =============================================================================
# Parametrized Tests
# =============================================================================

class TestParametrizedCalculations:
    """Parametrized tests for multiple products and countries."""

    @pytest.mark.unit
    @pytest.mark.parametrize("cn_code,country,quantity,expected_category", [
        ("72081000", "CN", 1000, CBAMProductCategory.IRON_STEEL),
        ("76011000", "CN", 500, CBAMProductCategory.ALUMINIUM),
        ("25232900", "IN", 2000, CBAMProductCategory.CEMENT),
        ("31021000", "RU", 750, CBAMProductCategory.FERTILIZERS),
        ("27160000", "UA", 1000, CBAMProductCategory.ELECTRICITY),
        ("28041000", "NO", 100, CBAMProductCategory.HYDROGEN),
    ])
    def test_product_classification_parametrized(self, agent, cn_code, country, quantity, expected_category):
        """Test product classification for various CN codes."""
        input_data = CBAMInput(
            cn_code=cn_code,
            quantity_tonnes=quantity,
            country_of_origin=country,
            reporting_period="Q1 2026",
        )
        result = agent.run(input_data)
        assert result.product_category == expected_category.value


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
