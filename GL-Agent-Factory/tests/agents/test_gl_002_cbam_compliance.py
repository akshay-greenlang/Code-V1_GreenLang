"""
Unit Tests for GL-002: CBAM Compliance Agent

Comprehensive test suite covering:
- CN code classification and validation
- Embedded emissions calculation accuracy
- Default vs actual emission factors
- CBAM liability calculation
- Country-specific factors (China, India, etc.)
- Provenance hash generation
- Regulatory compliance testing

Target: 85%+ code coverage

Reference:
- EU Regulation (EU) 2023/956 (CBAM Regulation)
- EU Implementing Regulation 2023/1773

Run with:
    pytest tests/agents/test_gl_002_cbam_compliance.py -v --cov=backend/agents/gl_002_cbam_compliance
"""

import pytest
from datetime import datetime
from decimal import Decimal
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend"))

from agents.gl_002_cbam_compliance.agent import (
    CBAMComplianceAgent,
    CBAMInput,
    CBAMOutput,
    CBAMProductCategory,
    CalculationMethod,
    EmissionType,
)


# =============================================================================
# Test Class: Agent Initialization
# =============================================================================


class TestCBAMAgentInitialization:
    """Tests for CBAMComplianceAgent initialization."""

    @pytest.mark.unit
    def test_agent_initializes_with_defaults(self):
        """Test agent initializes correctly with default config."""
        agent = CBAMComplianceAgent()

        assert agent is not None
        assert agent.AGENT_ID == "regulatory/cbam_compliance_v1"
        assert agent.VERSION == "1.0.0"

    @pytest.mark.unit
    def test_agent_has_cn_to_category_mapping(self):
        """Test agent has CN code to category mapping."""
        agent = CBAMComplianceAgent()

        assert hasattr(agent, "CN_TO_CATEGORY")
        assert "7208" in agent.CN_TO_CATEGORY  # Iron/steel
        assert "7601" in agent.CN_TO_CATEGORY  # Aluminium
        assert "2523" in agent.CN_TO_CATEGORY  # Cement

    @pytest.mark.unit
    def test_agent_has_default_factors(self):
        """Test agent has default emission factors."""
        agent = CBAMComplianceAgent()

        assert hasattr(agent, "DEFAULT_FACTORS")
        assert "iron_steel" in agent.DEFAULT_FACTORS
        assert "aluminium" in agent.DEFAULT_FACTORS
        assert "cement" in agent.DEFAULT_FACTORS

    @pytest.mark.unit
    def test_agent_has_eu_ets_price(self):
        """Test agent has EU ETS carbon price."""
        agent = CBAMComplianceAgent()

        assert hasattr(agent, "EU_ETS_PRICE")
        assert agent.EU_ETS_PRICE > 0


# =============================================================================
# Test Class: Input Validation
# =============================================================================


class TestCBAMInputValidation:
    """Tests for CBAMInput Pydantic model validation."""

    @pytest.mark.unit
    def test_valid_steel_input(self):
        """Test valid steel import input passes validation."""
        input_data = CBAMInput(
            cn_code="72081000",
            quantity_tonnes=1000.0,
            country_of_origin="CN",
            reporting_period="Q1 2026",
        )

        assert input_data.cn_code == "72081000"
        assert input_data.quantity_tonnes == 1000.0
        assert input_data.country_of_origin == "CN"

    @pytest.mark.unit
    def test_valid_aluminium_input(self):
        """Test valid aluminium import input passes validation."""
        input_data = CBAMInput(
            cn_code="76011000",
            quantity_tonnes=500.0,
            country_of_origin="RU",
            reporting_period="Q2 2026",
        )

        assert input_data.cn_code == "76011000"

    @pytest.mark.unit
    def test_cn_code_normalized(self):
        """Test CN code is normalized (dots removed)."""
        input_data = CBAMInput(
            cn_code="7208.10.00",  # With dots
            quantity_tonnes=100.0,
            country_of_origin="CN",
            reporting_period="Q1 2026",
        )

        # Should be cleaned to first 8 digits
        assert input_data.cn_code == "72081000"

    @pytest.mark.unit
    def test_cn_code_minimum_length(self):
        """Test CN code must be at least 8 digits."""
        with pytest.raises(ValueError):
            CBAMInput(
                cn_code="7208",  # Too short
                quantity_tonnes=100.0,
                country_of_origin="CN",
                reporting_period="Q1 2026",
            )

    @pytest.mark.unit
    def test_cn_code_only_digits(self):
        """Test CN code must contain only digits."""
        with pytest.raises(ValueError):
            CBAMInput(
                cn_code="7208ABCD",  # Contains letters
                quantity_tonnes=100.0,
                country_of_origin="CN",
                reporting_period="Q1 2026",
            )

    @pytest.mark.unit
    def test_negative_quantity_rejected(self):
        """Test negative quantity is rejected."""
        with pytest.raises(ValueError):
            CBAMInput(
                cn_code="72081000",
                quantity_tonnes=-100.0,
                country_of_origin="CN",
                reporting_period="Q1 2026",
            )

    @pytest.mark.unit
    def test_country_code_normalized(self):
        """Test country code is normalized to uppercase."""
        input_data = CBAMInput(
            cn_code="72081000",
            quantity_tonnes=100.0,
            country_of_origin="cn",  # lowercase
            reporting_period="Q1 2026",
        )

        assert input_data.country_of_origin == "CN"

    @pytest.mark.unit
    def test_country_code_length(self):
        """Test country code must be 2 characters."""
        with pytest.raises(ValueError):
            CBAMInput(
                cn_code="72081000",
                quantity_tonnes=100.0,
                country_of_origin="CHN",  # 3 chars
                reporting_period="Q1 2026",
            )

    @pytest.mark.unit
    def test_actual_emissions_optional(self):
        """Test actual emissions field is optional."""
        input_data = CBAMInput(
            cn_code="72081000",
            quantity_tonnes=100.0,
            country_of_origin="CN",
            reporting_period="Q1 2026",
            actual_emissions=1.85,  # Optional actual value
        )

        assert input_data.actual_emissions == 1.85


# =============================================================================
# Test Class: Product Classification
# =============================================================================


class TestCBAMProductClassification:
    """Tests for CN code to CBAM product category classification."""

    @pytest.mark.unit
    def test_iron_steel_classification(self, cbam_agent):
        """Test iron/steel products correctly classified."""
        # Chapter 72 - Iron and steel
        steel_codes = ["7201", "7208", "7210", "7219", "7228"]

        for code in steel_codes:
            assert cbam_agent._classify_product(f"{code}0000") == CBAMProductCategory.IRON_STEEL

    @pytest.mark.unit
    def test_aluminium_classification(self, cbam_agent):
        """Test aluminium products correctly classified."""
        # Chapter 76 - Aluminium
        alu_codes = ["7601", "7604", "7606", "7610"]

        for code in alu_codes:
            assert cbam_agent._classify_product(f"{code}0000") == CBAMProductCategory.ALUMINIUM

    @pytest.mark.unit
    def test_cement_classification(self, cbam_agent):
        """Test cement products correctly classified."""
        assert cbam_agent._classify_product("25231000") == CBAMProductCategory.CEMENT

    @pytest.mark.unit
    def test_fertilizers_classification(self, cbam_agent):
        """Test fertilizer products correctly classified."""
        fert_codes = ["2808", "3102", "3105"]

        for code in fert_codes:
            result = cbam_agent._classify_product(f"{code}0000")
            assert result == CBAMProductCategory.FERTILIZERS

    @pytest.mark.unit
    def test_electricity_classification(self, cbam_agent):
        """Test electricity correctly classified."""
        assert cbam_agent._classify_product("27160000") == CBAMProductCategory.ELECTRICITY

    @pytest.mark.unit
    def test_hydrogen_classification(self, cbam_agent):
        """Test hydrogen correctly classified."""
        assert cbam_agent._classify_product("28041000") == CBAMProductCategory.HYDROGEN

    @pytest.mark.unit
    def test_non_cbam_product_returns_none(self, cbam_agent):
        """Test non-CBAM products return None."""
        # Textiles (Chapter 50-63) not in CBAM scope
        assert cbam_agent._classify_product("50010000") is None


# =============================================================================
# Test Class: Calculation Accuracy
# =============================================================================


class TestCBAMCalculationAccuracy:
    """Tests for CBAM calculation accuracy against known values."""

    @pytest.mark.unit
    @pytest.mark.compliance
    def test_steel_china_default_calculation(self, cbam_agent):
        """
        Test steel import from China using default factors.

        Default factors (China):
        - Direct: 2.10 tCO2e/t
        - Indirect: 0.45 tCO2e/t
        - Total: 2.55 tCO2e/t

        Input: 1000 tonnes
        Expected: 2550 tCO2e
        """
        input_data = CBAMInput(
            cn_code="72081000",
            quantity_tonnes=1000.0,
            country_of_origin="CN",
            reporting_period="Q1 2026",
        )

        result = cbam_agent.run(input_data)

        # Verify embedded emissions
        assert result.direct_emissions_tco2e == pytest.approx(2100.0, rel=1e-4)
        assert result.indirect_emissions_tco2e == pytest.approx(450.0, rel=1e-4)
        assert result.total_embedded_emissions_tco2e == pytest.approx(2550.0, rel=1e-4)
        assert result.specific_embedded_emissions == pytest.approx(2.55, rel=1e-4)
        assert result.calculation_method == "country_default"

    @pytest.mark.unit
    @pytest.mark.compliance
    def test_steel_india_calculation(self, cbam_agent):
        """
        Test steel import from India.

        Default factors (India):
        - Direct: 2.35 tCO2e/t
        - Indirect: 0.52 tCO2e/t
        - Total: 2.87 tCO2e/t
        """
        input_data = CBAMInput(
            cn_code="72101000",
            quantity_tonnes=500.0,
            country_of_origin="IN",
            reporting_period="Q1 2026",
        )

        result = cbam_agent.run(input_data)

        assert result.direct_emissions_tco2e == pytest.approx(500 * 2.35, rel=1e-4)
        assert result.indirect_emissions_tco2e == pytest.approx(500 * 0.52, rel=1e-4)

    @pytest.mark.unit
    @pytest.mark.compliance
    def test_steel_global_default(self, cbam_agent):
        """
        Test steel import from country without specific factors.

        Global default factors:
        - Direct: 1.85 tCO2e/t
        - Indirect: 0.32 tCO2e/t
        """
        input_data = CBAMInput(
            cn_code="72081000",
            quantity_tonnes=200.0,
            country_of_origin="BR",  # No Brazil-specific factors
            reporting_period="Q1 2026",
        )

        result = cbam_agent.run(input_data)

        assert result.direct_emissions_tco2e == pytest.approx(200 * 1.85, rel=1e-4)
        assert result.indirect_emissions_tco2e == pytest.approx(200 * 0.32, rel=1e-4)
        assert result.calculation_method == "default"

    @pytest.mark.unit
    @pytest.mark.compliance
    def test_aluminium_china_calculation(self, cbam_agent):
        """
        Test aluminium import from China.

        China has high indirect emissions due to coal-heavy grid.
        - Direct: 1.65 tCO2e/t
        - Indirect: 10.20 tCO2e/t (very high!)
        """
        input_data = CBAMInput(
            cn_code="76011000",
            quantity_tonnes=100.0,
            country_of_origin="CN",
            reporting_period="Q1 2026",
        )

        result = cbam_agent.run(input_data)

        assert result.direct_emissions_tco2e == pytest.approx(165.0, rel=1e-4)
        assert result.indirect_emissions_tco2e == pytest.approx(1020.0, rel=1e-4)
        assert result.total_embedded_emissions_tco2e == pytest.approx(1185.0, rel=1e-4)

    @pytest.mark.unit
    @pytest.mark.compliance
    def test_cement_calculation(self, cbam_agent):
        """
        Test cement import calculation.

        Global default:
        - Direct: 0.83 tCO2e/t (mainly process emissions)
        - Indirect: 0.05 tCO2e/t
        """
        input_data = CBAMInput(
            cn_code="25231000",
            quantity_tonnes=2000.0,
            country_of_origin="TR",
            reporting_period="Q2 2026",
        )

        result = cbam_agent.run(input_data)

        expected_direct = 2000 * 0.83
        expected_indirect = 2000 * 0.05

        assert result.direct_emissions_tco2e == pytest.approx(expected_direct, rel=1e-4)
        assert result.indirect_emissions_tco2e == pytest.approx(expected_indirect, rel=1e-4)
        assert result.product_category == "cement"

    @pytest.mark.unit
    @pytest.mark.compliance
    def test_actual_emissions_override(self, cbam_agent):
        """Test actual emissions override default values."""
        input_data = CBAMInput(
            cn_code="72081000",
            quantity_tonnes=1000.0,
            country_of_origin="CN",
            actual_emissions=1.50,  # Verified actual value
            reporting_period="Q1 2026",
        )

        result = cbam_agent.run(input_data)

        # Should use actual emissions instead of defaults
        assert result.direct_emissions_tco2e == pytest.approx(1500.0, rel=1e-4)
        assert result.indirect_emissions_tco2e == 0.0
        assert result.calculation_method == "actual"
        assert result.emission_factor_source == "Verified installation data"


# =============================================================================
# Test Class: CBAM Liability Calculation
# =============================================================================


class TestCBAMLiabilityCalculation:
    """Tests for CBAM certificate liability calculation."""

    @pytest.mark.unit
    @pytest.mark.compliance
    def test_cbam_liability_calculation(self, cbam_agent):
        """
        Test CBAM liability calculation.

        CBAM Liability = Total Embedded Emissions * Carbon Price
        """
        input_data = CBAMInput(
            cn_code="72081000",
            quantity_tonnes=1000.0,
            country_of_origin="CN",
            reporting_period="Q1 2026",
        )

        result = cbam_agent.run(input_data)

        # Total emissions: 2550 tCO2e
        # Carbon price: 85 EUR/tCO2
        # Liability: 2550 * 85 = 216,750 EUR
        expected_liability = 2550.0 * 85.0

        assert result.cbam_liability_eur == pytest.approx(expected_liability, rel=1e-2)
        assert result.carbon_price_applicable == 85.0

    @pytest.mark.unit
    def test_zero_quantity_zero_liability(self, cbam_agent):
        """Test zero quantity results in zero liability."""
        input_data = CBAMInput(
            cn_code="72081000",
            quantity_tonnes=0.0,
            country_of_origin="CN",
            reporting_period="Q1 2026",
        )

        result = cbam_agent.run(input_data)

        assert result.total_embedded_emissions_tco2e == 0.0
        assert result.cbam_liability_eur == 0.0


# =============================================================================
# Test Class: Provenance Hash
# =============================================================================


class TestCBAMProvenanceHash:
    """Tests for CBAM provenance hash generation."""

    @pytest.mark.unit
    def test_provenance_hash_exists(self, cbam_agent, cbam_steel_input):
        """Test output includes provenance hash."""
        result = cbam_agent.run(cbam_steel_input)

        assert result.provenance_hash is not None
        assert len(result.provenance_hash) == 64

    @pytest.mark.unit
    def test_provenance_hash_valid_format(self, cbam_agent, cbam_steel_input):
        """Test provenance hash is valid SHA-256."""
        result = cbam_agent.run(cbam_steel_input)

        assert all(c in "0123456789abcdef" for c in result.provenance_hash.lower())

    @pytest.mark.unit
    def test_different_inputs_different_hashes(self, cbam_agent):
        """Test different inputs produce different hashes."""
        input1 = CBAMInput(
            cn_code="72081000",
            quantity_tonnes=1000.0,
            country_of_origin="CN",
            reporting_period="Q1 2026",
        )
        input2 = CBAMInput(
            cn_code="72081000",
            quantity_tonnes=2000.0,  # Different quantity
            country_of_origin="CN",
            reporting_period="Q1 2026",
        )

        result1 = cbam_agent.run(input1)
        result2 = cbam_agent.run(input2)

        assert result1.provenance_hash != result2.provenance_hash


# =============================================================================
# Test Class: Error Handling
# =============================================================================


class TestCBAMErrorHandling:
    """Tests for CBAM error handling."""

    @pytest.mark.unit
    def test_non_cbam_product_raises_error(self, cbam_agent):
        """Test non-CBAM product raises ValueError."""
        input_data = CBAMInput(
            cn_code="50010000",  # Silk - not CBAM product
            quantity_tonnes=100.0,
            country_of_origin="CN",
            reporting_period="Q1 2026",
        )

        with pytest.raises(ValueError, match="not in CBAM scope"):
            cbam_agent.run(input_data)

    @pytest.mark.unit
    def test_missing_required_fields(self):
        """Test missing required fields raise error."""
        with pytest.raises(ValueError):
            CBAMInput(
                cn_code="72081000",
                # quantity_tonnes missing
                country_of_origin="CN",
                reporting_period="Q1 2026",
            )


# =============================================================================
# Test Class: Edge Cases
# =============================================================================


class TestCBAMEdgeCases:
    """Tests for CBAM edge cases and boundary conditions."""

    @pytest.mark.unit
    def test_very_small_quantity(self, cbam_agent):
        """Test very small import quantity."""
        input_data = CBAMInput(
            cn_code="72081000",
            quantity_tonnes=0.001,
            country_of_origin="CN",
            reporting_period="Q1 2026",
        )

        result = cbam_agent.run(input_data)

        assert result.total_embedded_emissions_tco2e > 0
        assert result.cbam_liability_eur > 0

    @pytest.mark.unit
    def test_very_large_quantity(self, cbam_agent):
        """Test very large import quantity."""
        input_data = CBAMInput(
            cn_code="72081000",
            quantity_tonnes=1_000_000.0,  # 1 million tonnes
            country_of_origin="CN",
            reporting_period="Q1 2026",
        )

        result = cbam_agent.run(input_data)

        expected_emissions = 1_000_000.0 * 2.55
        assert result.total_embedded_emissions_tco2e == pytest.approx(expected_emissions, rel=1e-4)

    @pytest.mark.unit
    def test_all_cbam_product_categories(self, cbam_agent):
        """Test all CBAM product categories can be processed."""
        test_cases = [
            ("72081000", "iron_steel"),
            ("76011000", "aluminium"),
            ("25231000", "cement"),
            ("31021000", "fertilizers"),
            ("27160000", "electricity"),
            ("28041000", "hydrogen"),
        ]

        for cn_code, expected_category in test_cases:
            input_data = CBAMInput(
                cn_code=cn_code,
                quantity_tonnes=100.0,
                country_of_origin="CN",
                reporting_period="Q1 2026",
            )

            result = cbam_agent.run(input_data)

            assert result.product_category == expected_category

    @pytest.mark.unit
    def test_various_reporting_periods(self, cbam_agent):
        """Test various reporting periods."""
        periods = ["Q1 2026", "Q2 2026", "Q3 2026", "Q4 2026"]

        for period in periods:
            input_data = CBAMInput(
                cn_code="72081000",
                quantity_tonnes=100.0,
                country_of_origin="CN",
                reporting_period=period,
            )

            result = cbam_agent.run(input_data)

            assert result.reporting_period == period


# =============================================================================
# Test Class: Helper Methods
# =============================================================================


class TestCBAMHelperMethods:
    """Tests for CBAM helper methods."""

    @pytest.mark.unit
    def test_get_cbam_products(self, cbam_agent):
        """Test get_cbam_products returns all categories."""
        products = cbam_agent.get_cbam_products()

        assert isinstance(products, list)
        assert "cement" in products
        assert "iron_steel" in products
        assert "aluminium" in products
        assert "fertilizers" in products
        assert "electricity" in products
        assert "hydrogen" in products

    @pytest.mark.unit
    def test_is_in_scope_true(self, cbam_agent):
        """Test is_in_scope returns True for CBAM products."""
        assert cbam_agent.is_in_scope("72081000") is True
        assert cbam_agent.is_in_scope("76011000") is True
        assert cbam_agent.is_in_scope("25231000") is True

    @pytest.mark.unit
    def test_is_in_scope_false(self, cbam_agent):
        """Test is_in_scope returns False for non-CBAM products."""
        assert cbam_agent.is_in_scope("50010000") is False  # Silk
        assert cbam_agent.is_in_scope("84713000") is False  # Computers


# =============================================================================
# Test Class: Output Model
# =============================================================================


class TestCBAMOutput:
    """Tests for CBAMOutput model."""

    @pytest.mark.unit
    def test_output_has_all_required_fields(self, cbam_agent, cbam_steel_input):
        """Test output includes all required fields."""
        result = cbam_agent.run(cbam_steel_input)

        assert hasattr(result, "cn_code")
        assert hasattr(result, "product_category")
        assert hasattr(result, "quantity_tonnes")
        assert hasattr(result, "direct_emissions_tco2e")
        assert hasattr(result, "indirect_emissions_tco2e")
        assert hasattr(result, "total_embedded_emissions_tco2e")
        assert hasattr(result, "specific_embedded_emissions")
        assert hasattr(result, "calculation_method")
        assert hasattr(result, "carbon_price_applicable")
        assert hasattr(result, "cbam_liability_eur")
        assert hasattr(result, "provenance_hash")
        assert hasattr(result, "reporting_period")

    @pytest.mark.unit
    def test_output_calculated_at_timestamp(self, cbam_agent, cbam_steel_input):
        """Test calculated_at timestamp is set."""
        before = datetime.utcnow()
        result = cbam_agent.run(cbam_steel_input)
        after = datetime.utcnow()

        assert before <= result.calculated_at <= after


# =============================================================================
# Test Class: Performance
# =============================================================================


class TestCBAMPerformance:
    """Performance tests for CBAMComplianceAgent."""

    @pytest.mark.unit
    @pytest.mark.performance
    def test_single_calculation_under_10ms(self, cbam_agent, cbam_steel_input, performance_timer):
        """Test single calculation completes in under 10ms."""
        performance_timer.start()
        result = cbam_agent.run(cbam_steel_input)
        performance_timer.stop()

        assert performance_timer.elapsed_ms < 10.0

    @pytest.mark.unit
    @pytest.mark.performance
    def test_batch_shipment_processing(self, cbam_agent, performance_timer):
        """Test batch shipment processing throughput."""
        num_shipments = 500
        shipments = [
            CBAMInput(
                cn_code="72081000",
                quantity_tonnes=float(i * 100),
                country_of_origin="CN",
                reporting_period="Q1 2026",
            )
            for i in range(1, num_shipments + 1)
        ]

        performance_timer.start()
        results = [cbam_agent.run(s) for s in shipments]
        performance_timer.stop()

        assert len(results) == num_shipments
        throughput = num_shipments / (performance_timer.elapsed_ms / 1000)
        assert throughput >= 50, f"Throughput {throughput:.0f} rec/sec below target"
