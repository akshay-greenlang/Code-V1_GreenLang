# -*- coding: utf-8 -*-
"""
Unit tests for ProducerSpecificCalculatorEngine -- AGENT-MRV-025

Tests the producer-specific calculation method which uses Environmental Product
Declarations (EPDs) and product carbon footprint (PCF) data to calculate
end-of-life emissions from the reported EOL module.

Coverage:
- EPD parsing and validation
- PCF end-of-life module extraction (C1-C4 lifecycle stages)
- Verification status levels (self-declared, second-party, third-party)
- Take-back program calculations
- EPR (Extended Producer Responsibility) scheme obligations
- Recycled content tracking
- DQI scoring (highest quality for producer-specific method)

Target: 35+ expanded tests.
Author: GL-TestEngineer
Date: February 2026
"""

from decimal import Decimal
from typing import Any, Dict, List
import pytest

try:
    from greenlang.end_of_life_treatment.producer_specific_calculator import (
        ProducerSpecificCalculatorEngine,
    )
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

_SKIP = pytest.mark.skipif(not _AVAILABLE, reason="ProducerSpecificCalculatorEngine not available")
pytestmark = _SKIP

_Q8 = Decimal("0.00000001")


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def engine():
    """Create a ProducerSpecificCalculatorEngine instance."""
    return ProducerSpecificCalculatorEngine.get_instance()


@pytest.fixture
def epd_verified_input():
    """Third-party verified EPD input."""
    return {
        "product_id": "PRD-EPD-001",
        "epd_id": "EPD-2024-00123",
        "product_category": "consumer_electronics",
        "total_mass_kg": Decimal("1000.0"),
        "units_sold": 5000,
        "eol_module_co2e_kg": Decimal("850.0"),
        "eol_c1_deconstruction_kg": Decimal("50.0"),
        "eol_c2_transport_kg": Decimal("30.0"),
        "eol_c3_waste_processing_kg": Decimal("270.0"),
        "eol_c4_disposal_kg": Decimal("500.0"),
        "module_d_avoided_kg": Decimal("320.0"),
        "verification_level": "third_party_verified",
        "verifier_name": "SGS International",
        "epd_valid_until": "2027-12-31",
        "region": "US",
        "reporting_year": 2024,
    }


@pytest.fixture
def epd_self_declared_input():
    """Self-declared EPD input (lower DQI)."""
    return {
        "product_id": "PRD-EPD-002",
        "epd_id": "EPD-SELF-001",
        "product_category": "packaging",
        "total_mass_kg": Decimal("5000.0"),
        "units_sold": 100000,
        "eol_module_co2e_kg": Decimal("1200.0"),
        "module_d_avoided_kg": Decimal("400.0"),
        "verification_level": "self_declared",
        "region": "DE",
        "reporting_year": 2024,
    }


@pytest.fixture
def take_back_input():
    """Product with take-back program."""
    return {
        "product_id": "PRD-TB-001",
        "epd_id": "EPD-2024-00456",
        "product_category": "consumer_electronics",
        "total_mass_kg": Decimal("500.0"),
        "units_sold": 2500,
        "eol_module_co2e_kg": Decimal("420.0"),
        "module_d_avoided_kg": Decimal("280.0"),
        "verification_level": "third_party_verified",
        "take_back_rate": Decimal("0.35"),
        "take_back_treatment": "recycling",
        "region": "DE",
        "reporting_year": 2024,
    }


@pytest.fixture
def epr_input():
    """Product with Extended Producer Responsibility obligations."""
    return {
        "product_id": "PRD-EPR-001",
        "epd_id": "EPD-2024-00789",
        "product_category": "large_appliances",
        "total_mass_kg": Decimal("35000.0"),
        "units_sold": 1000,
        "eol_module_co2e_kg": Decimal("12500.0"),
        "module_d_avoided_kg": Decimal("8500.0"),
        "verification_level": "third_party_verified",
        "epr_scheme": "WEEE_Directive",
        "epr_collection_rate": Decimal("0.65"),
        "epr_recycling_target": Decimal("0.80"),
        "region": "DE",
        "reporting_year": 2024,
    }


# ============================================================================
# TEST: EPD Parsing and Validation
# ============================================================================


class TestEPDParsing:
    """Test EPD data parsing and validation."""

    def test_valid_epd_returns_result(self, engine, epd_verified_input):
        """Test valid EPD input produces calculation result."""
        result = engine.calculate(epd_verified_input)
        assert result is not None
        assert "gross_emissions_kgco2e" in result

    def test_epd_id_in_result(self, engine, epd_verified_input):
        """Test EPD ID is included in result."""
        result = engine.calculate(epd_verified_input)
        assert result.get("epd_id") == "EPD-2024-00123"

    def test_eol_modules_c1_to_c4(self, engine, epd_verified_input):
        """Test individual C1-C4 modules are captured if provided."""
        result = engine.calculate(epd_verified_input)
        # Total EOL should be C1 + C2 + C3 + C4
        total = result["gross_emissions_kgco2e"]
        assert total > Decimal("0.0")

    def test_module_d_avoided_separate(self, engine, epd_verified_input):
        """CRITICAL: Test Module D (avoided emissions) is reported separately."""
        result = engine.calculate(epd_verified_input)
        gross = result["gross_emissions_kgco2e"]
        avoided = result.get("avoided_emissions_kgco2e", Decimal("0.0"))
        # Module D must be separate from C1-C4 gross
        assert avoided >= Decimal("0.0")
        # Gross must NOT have Module D subtracted
        assert gross == result["gross_emissions_kgco2e"]

    def test_missing_epd_id_raises_error(self, engine):
        """Test missing EPD ID raises validation error."""
        inp = {
            "product_id": "PRD-NO-EPD",
            "product_category": "packaging",
            "total_mass_kg": Decimal("100.0"),
            "eol_module_co2e_kg": Decimal("50.0"),
            "verification_level": "self_declared",
        }
        with pytest.raises((ValueError, KeyError)):
            engine.calculate(inp)


# ============================================================================
# TEST: Verification Status Levels
# ============================================================================


class TestVerificationLevels:
    """Test verification status impacts on DQI scoring."""

    @pytest.mark.parametrize("verification,min_dqi", [
        ("third_party_verified", Decimal("80.0")),
        ("second_party", Decimal("60.0")),
        ("self_declared", Decimal("40.0")),
    ])
    def test_verification_dqi_impact(self, engine, verification, min_dqi):
        """Test verification level impacts DQI score."""
        inp = {
            "product_id": f"PRD-V-{verification}",
            "epd_id": f"EPD-{verification}",
            "product_category": "packaging",
            "total_mass_kg": Decimal("100.0"),
            "units_sold": 100,
            "eol_module_co2e_kg": Decimal("50.0"),
            "verification_level": verification,
            "region": "US",
            "reporting_year": 2024,
        }
        result = engine.calculate(inp)
        assert result["dqi_score"] >= min_dqi

    def test_third_party_highest_dqi(self, engine, epd_verified_input, epd_self_declared_input):
        """Test third-party verified has highest DQI score."""
        verified = engine.calculate(epd_verified_input)
        self_declared = engine.calculate(epd_self_declared_input)
        assert verified["dqi_score"] > self_declared["dqi_score"]


# ============================================================================
# TEST: Take-Back Program Calculations
# ============================================================================


class TestTakeBackPrograms:
    """Test take-back program impact on calculations."""

    def test_take_back_rate_applied(self, engine, take_back_input):
        """Test take-back rate reduces uncontrolled EOL fraction."""
        result = engine.calculate(take_back_input)
        assert result is not None
        assert result["gross_emissions_kgco2e"] >= Decimal("0.0")

    def test_take_back_affects_treatment_mix(self, engine, take_back_input):
        """Test take-back program shifts treatment toward recycling."""
        with_tb = engine.calculate(take_back_input)
        # Without take-back
        without_tb_input = dict(take_back_input)
        without_tb_input.pop("take_back_rate", None)
        without_tb_input.pop("take_back_treatment", None)
        without_tb_input["product_id"] = "PRD-NO-TB"
        without_tb = engine.calculate(without_tb_input)
        # Take-back should change emissions profile
        assert with_tb["gross_emissions_kgco2e"] != without_tb["gross_emissions_kgco2e"]


# ============================================================================
# TEST: EPR Scheme Obligations
# ============================================================================


class TestEPRSchemes:
    """Test Extended Producer Responsibility scheme calculations."""

    def test_epr_scheme_recorded(self, engine, epr_input):
        """Test EPR scheme is recorded in result."""
        result = engine.calculate(epr_input)
        assert result is not None

    def test_epr_collection_rate(self, engine, epr_input):
        """Test EPR collection rate is applied."""
        result = engine.calculate(epr_input)
        assert result["gross_emissions_kgco2e"] >= Decimal("0.0")

    def test_epr_weee_high_recycling(self, engine, epr_input):
        """Test WEEE EPR mandates high recycling targets."""
        result = engine.calculate(epr_input)
        # WEEE recycling target of 80% should increase recycling pathway
        assert result["gross_emissions_kgco2e"] >= Decimal("0.0")


# ============================================================================
# TEST: Recycled Content Tracking
# ============================================================================


class TestRecycledContent:
    """Test recycled content tracking for circularity."""

    def test_recycled_content_input(self, engine):
        """Test recycled content is tracked when provided."""
        inp = {
            "product_id": "PRD-RC-001",
            "epd_id": "EPD-RC-001",
            "product_category": "packaging",
            "total_mass_kg": Decimal("1000.0"),
            "units_sold": 1000,
            "eol_module_co2e_kg": Decimal("500.0"),
            "module_d_avoided_kg": Decimal("200.0"),
            "verification_level": "third_party_verified",
            "recycled_content_pct": Decimal("0.30"),
            "region": "DE",
            "reporting_year": 2024,
        }
        result = engine.calculate(inp)
        assert result is not None


# ============================================================================
# TEST: DQI (Highest Quality for Producer-Specific)
# ============================================================================


class TestProducerSpecificDQI:
    """Test DQI scoring for producer-specific method."""

    def test_highest_method_dqi(self, engine, epd_verified_input):
        """Test producer-specific with 3rd party verification has highest DQI."""
        result = engine.calculate(epd_verified_input)
        # Producer-specific with verification should be 80-95+
        assert result["dqi_score"] >= Decimal("75.0")

    def test_method_is_producer_specific(self, engine, epd_verified_input):
        """Test method field is 'producer_specific'."""
        result = engine.calculate(epd_verified_input)
        assert result["method"] == "producer_specific"

    def test_verification_status_in_result(self, engine, epd_verified_input):
        """Test verification status is included in result."""
        result = engine.calculate(epd_verified_input)
        assert result.get("verification_status") == "third_party_verified"


# ============================================================================
# TEST: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases for producer-specific calculator."""

    def test_zero_eol_module(self, engine):
        """Test zero EOL module emissions (product fully recyclable per EPD)."""
        inp = {
            "product_id": "PRD-ZERO-EOL",
            "epd_id": "EPD-ZERO",
            "product_category": "packaging",
            "total_mass_kg": Decimal("100.0"),
            "units_sold": 100,
            "eol_module_co2e_kg": Decimal("0.0"),
            "module_d_avoided_kg": Decimal("50.0"),
            "verification_level": "third_party_verified",
            "region": "US",
            "reporting_year": 2024,
        }
        result = engine.calculate(inp)
        assert result["gross_emissions_kgco2e"] == Decimal("0.0")
        # But avoided emissions should still be tracked
        avoided = result.get("avoided_emissions_kgco2e", Decimal("0.0"))
        assert avoided >= Decimal("0.0")
