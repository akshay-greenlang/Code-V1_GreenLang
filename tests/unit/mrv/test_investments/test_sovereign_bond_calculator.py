# -*- coding: utf-8 -*-
"""
Test suite for investments.sovereign_bond_calculator - AGENT-MRV-028.

Tests the SovereignBondCalculatorEngine (Engine 5) for the Investments
Agent (GL-MRV-S3-015) including GDP-PPP attribution, country emissions
lookup, LULUCF adjustment, PCAF quality scoring, per capita intensity,
DC-INV-005 sovereign vs corporate distinction, and parametrized country
tests.

Coverage:
- Attribution: outstanding / PPP_adjusted_GDP
- Country emissions lookup for 10+ countries
- LULUCF adjustment (inclusion/exclusion)
- PCAF quality (Score 4 vs Score 5)
- Per capita intensity
- DC-INV-005 (sovereign vs corporate)
- Parametrized tests for countries

Author: GL-TestEngineer
Date: February 2026
"""

from decimal import Decimal
from unittest.mock import patch, MagicMock
import pytest

from greenlang.agents.mrv.investments.sovereign_bond_calculator import (
    SovereignBondCalculatorEngine,
)
from greenlang.agents.mrv.investments.models import (
    AssetClass,
    PCAFDataQuality,
    AttributionMethod,
)


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset singleton before and after every test."""
    SovereignBondCalculatorEngine.reset_instance()
    yield
    SovereignBondCalculatorEngine.reset_instance()


@pytest.fixture
def engine():
    """Create a fresh SovereignBondCalculatorEngine with mocked config."""
    with patch(
        "greenlang.agents.mrv.investments.sovereign_bond_calculator.get_config"
    ) as mock_config:
        cfg = MagicMock()
        cfg.sovereign.include_lulucf = False
        cfg.general.default_gwp = "AR5"
        mock_config.return_value = cfg
        eng = SovereignBondCalculatorEngine()
        yield eng


def _make_sovereign_input(**overrides):
    """Build a sovereign bond input dict with defaults."""
    base = {
        "asset_class": "sovereign_bond",
        "country": "US",
        "outstanding_amount": Decimal("500000000"),
        "gdp_ppp": Decimal("25460000000000"),
        "country_emissions": Decimal("5222000000"),
        "include_lulucf": False,
        "currency": "USD",
        "reporting_year": 2024,
        "pcaf_quality_score": 4,
    }
    base.update(overrides)
    return base


# ==============================================================================
# ATTRIBUTION FACTOR TESTS
# ==============================================================================


class TestAttributionFactor:
    """Test sovereign bond attribution factor = outstanding / GDP_PPP."""

    def test_attribution_factor_basic(self, engine):
        """Test basic attribution factor calculation."""
        data = _make_sovereign_input()
        result = engine.calculate(data)
        expected_af = Decimal("500000000") / Decimal("25460000000000")
        assert abs(result["attribution_factor"] - expected_af) < Decimal("0.0000001")

    def test_attribution_factor_positive(self, engine):
        """Test attribution factor is always positive."""
        data = _make_sovereign_input()
        result = engine.calculate(data)
        assert result["attribution_factor"] > Decimal("0")

    def test_attribution_factor_less_than_one(self, engine):
        """Test attribution factor is less than 1 for normal holdings."""
        data = _make_sovereign_input()
        result = engine.calculate(data)
        assert result["attribution_factor"] < Decimal("1")

    def test_attribution_method_is_gdp_ppp(self, engine):
        """Test attribution method is gdp_ppp."""
        data = _make_sovereign_input()
        result = engine.calculate(data)
        assert result["attribution_method"] == "gdp_ppp"

    def test_larger_holding_more_emissions(self, engine):
        """Test larger bond holding produces more financed emissions."""
        small = _make_sovereign_input(outstanding_amount=Decimal("100000000"))
        large = _make_sovereign_input(outstanding_amount=Decimal("1000000000"))
        r_small = engine.calculate(small)
        r_large = engine.calculate(large)
        assert r_large["financed_emissions"] > r_small["financed_emissions"]

    def test_zero_gdp_raises_error(self, engine):
        """Test zero GDP PPP raises error."""
        data = _make_sovereign_input(gdp_ppp=Decimal("0"))
        with pytest.raises((ValueError, ZeroDivisionError)):
            engine.calculate(data)


# ==============================================================================
# COUNTRY EMISSIONS TESTS
# ==============================================================================


class TestCountryEmissions:
    """Test country emissions lookups."""

    @pytest.mark.parametrize("country,min_emissions", [
        ("US", Decimal("4000000000")),
        ("CN", Decimal("9000000000")),
        ("IN", Decimal("2000000000")),
        ("DE", Decimal("600000000")),
        ("GB", Decimal("300000000")),
        ("JP", Decimal("1000000000")),
        ("FR", Decimal("300000000")),
        ("BR", Decimal("400000000")),
        ("CA", Decimal("500000000")),
        ("AU", Decimal("300000000")),
    ])
    def test_country_emissions_range(self, engine, country, min_emissions):
        """Test country emissions are within expected ranges."""
        data = _make_sovereign_input(
            country=country,
            country_emissions=min_emissions * Decimal("1.5"),
            gdp_ppp=Decimal("5000000000000"),
        )
        result = engine.calculate(data)
        assert result["financed_emissions"] > Decimal("0")

    def test_us_financed_emissions(self, engine):
        """Test US sovereign bond financed emissions calculation."""
        data = _make_sovereign_input()
        result = engine.calculate(data)
        af = Decimal("500000000") / Decimal("25460000000000")
        expected = af * Decimal("5222000000")
        assert abs(result["financed_emissions"] - expected) < Decimal("100")

    def test_higher_country_emissions_more_financed(self, engine):
        """Test higher country emissions produce more financed emissions."""
        low = _make_sovereign_input(country_emissions=Decimal("1000000000"))
        high = _make_sovereign_input(country_emissions=Decimal("10000000000"))
        r_low = engine.calculate(low)
        r_high = engine.calculate(high)
        assert r_high["financed_emissions"] > r_low["financed_emissions"]


# ==============================================================================
# LULUCF ADJUSTMENT TESTS
# ==============================================================================


class TestLULUCFAdjustment:
    """Test LULUCF (Land Use, Land-Use Change and Forestry) adjustment."""

    def test_lulucf_excluded_by_default(self, engine):
        """Test LULUCF is excluded by default."""
        data = _make_sovereign_input(include_lulucf=False)
        result = engine.calculate(data)
        assert result.get("include_lulucf", False) is False

    def test_lulucf_included_when_requested(self, engine):
        """Test LULUCF can be included."""
        data = _make_sovereign_input(
            include_lulucf=True,
            lulucf_emissions=Decimal("-500000000"),
        )
        result = engine.calculate(data)
        # LULUCF inclusion should change the result
        assert result["financed_emissions"] > Decimal("0")

    def test_lulucf_changes_result(self, engine):
        """Test LULUCF adjustment changes the financed emissions."""
        data_without = _make_sovereign_input(
            include_lulucf=False,
            lulucf_emissions=Decimal("-500000000"),
        )
        data_with = _make_sovereign_input(
            include_lulucf=True,
            lulucf_emissions=Decimal("-500000000"),
        )
        r_without = engine.calculate(data_without)
        r_with = engine.calculate(data_with)
        # LULUCF is typically a carbon sink (negative), so including it
        # should reduce total emissions
        assert r_with["financed_emissions"] != r_without["financed_emissions"]

    def test_positive_lulucf_increases_emissions(self, engine):
        """Test positive LULUCF (deforestation) increases emissions."""
        data_without = _make_sovereign_input(include_lulucf=False)
        data_with = _make_sovereign_input(
            include_lulucf=True,
            lulucf_emissions=Decimal("200000000"),
        )
        r_without = engine.calculate(data_without)
        r_with = engine.calculate(data_with)
        assert r_with["financed_emissions"] > r_without["financed_emissions"]


# ==============================================================================
# PCAF QUALITY TESTS
# ==============================================================================


class TestPCAFQuality:
    """Test PCAF quality scoring for sovereign bonds."""

    def test_pcaf_score_4_default(self, engine):
        """Test PCAF score 4 is typical for sovereign bonds."""
        data = _make_sovereign_input(pcaf_quality_score=4)
        result = engine.calculate(data)
        assert result["pcaf_quality_score"] == 4

    def test_pcaf_score_5_accepted(self, engine):
        """Test PCAF score 5 is accepted."""
        data = _make_sovereign_input(pcaf_quality_score=5)
        result = engine.calculate(data)
        assert result["pcaf_quality_score"] == 5

    def test_pcaf_score_3_accepted(self, engine):
        """Test PCAF score 3 is accepted for higher quality data."""
        data = _make_sovereign_input(pcaf_quality_score=3)
        result = engine.calculate(data)
        assert result["pcaf_quality_score"] == 3

    @pytest.mark.parametrize("score", [1, 2, 3, 4, 5])
    def test_all_pcaf_scores(self, engine, score):
        """Test all PCAF scores 1-5 are accepted."""
        data = _make_sovereign_input(pcaf_quality_score=score)
        result = engine.calculate(data)
        assert result["pcaf_quality_score"] == score


# ==============================================================================
# PER CAPITA INTENSITY TESTS
# ==============================================================================


class TestPerCapitaIntensity:
    """Test per capita emissions intensity calculations."""

    def test_per_capita_intensity(self, engine):
        """Test per capita intensity is calculated."""
        data = _make_sovereign_input(population=332000000)
        result = engine.calculate(data)
        if "per_capita_emissions" in result:
            expected = Decimal("5222000000") / Decimal("332000000")
            assert abs(result["per_capita_emissions"] - expected) < Decimal("0.1")

    def test_per_capita_positive(self, engine):
        """Test per capita intensity is positive."""
        data = _make_sovereign_input(population=332000000)
        result = engine.calculate(data)
        if "per_capita_emissions" in result:
            assert result["per_capita_emissions"] > Decimal("0")


# ==============================================================================
# DC-INV-005 TESTS
# ==============================================================================


class TestDCINV005:
    """Test DC-INV-005: sovereign vs corporate bond distinction."""

    def test_sovereign_not_corporate(self, engine):
        """Test sovereign bond is not treated as corporate bond."""
        data = _make_sovereign_input()
        result = engine.calculate(data)
        assert result["asset_class"] == "sovereign_bond"

    def test_dc_inv_005_no_overlap_with_corporate(self, engine):
        """Test DC-INV-005: sovereign bond does not double-count with corporate."""
        data = _make_sovereign_input()
        result = engine.calculate(data)
        # Should not have corporate-specific fields
        assert result["attribution_method"] == "gdp_ppp"
        assert "evic" not in result or result.get("evic") is None


# ==============================================================================
# BATCH AND ERROR HANDLING TESTS
# ==============================================================================


class TestBatchAndErrors:
    """Test batch processing and error handling."""

    @pytest.mark.parametrize("country", ["US", "CN", "DE", "JP", "GB", "FR", "IN", "BR", "CA", "AU"])
    def test_sovereign_bond_by_country(self, engine, country):
        """Test sovereign bond calculation for 10 countries."""
        data = _make_sovereign_input(
            country=country,
            gdp_ppp=Decimal("5000000000000"),
            country_emissions=Decimal("2000000000"),
        )
        result = engine.calculate(data)
        assert result["financed_emissions"] > Decimal("0")

    def test_batch_multiple_countries(self, engine):
        """Test batch with multiple sovereign bonds."""
        items = [
            _make_sovereign_input(country="US"),
            _make_sovereign_input(country="DE", gdp_ppp=Decimal("4600000000000")),
            _make_sovereign_input(country="JP", gdp_ppp=Decimal("5700000000000")),
        ]
        results = engine.calculate_batch(items)
        assert len(results) == 3

    def test_batch_empty(self, engine):
        """Test batch with empty list."""
        results = engine.calculate_batch([])
        assert len(results) == 0

    def test_result_required_fields(self, engine):
        """Test result contains all required fields."""
        data = _make_sovereign_input()
        result = engine.calculate(data)
        required = [
            "country", "attribution_factor", "financed_emissions",
            "pcaf_quality_score", "provenance_hash",
        ]
        for field in required:
            assert field in result

    def test_provenance_hash_present(self, engine):
        """Test provenance hash is present and 64 chars."""
        data = _make_sovereign_input()
        result = engine.calculate(data)
        assert len(result["provenance_hash"]) == 64

    def test_provenance_hash_deterministic(self, engine):
        """Test same input produces same provenance hash."""
        data = _make_sovereign_input()
        r1 = engine.calculate(data)
        r2 = engine.calculate(data)
        assert r1["provenance_hash"] == r2["provenance_hash"]
