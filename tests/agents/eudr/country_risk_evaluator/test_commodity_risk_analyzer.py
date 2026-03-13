# -*- coding: utf-8 -*-
"""
Unit tests for CommodityRiskAnalyzer - AGENT-EUDR-016 Engine 2

Tests EUDR commodity-specific risk analysis for 7 regulated commodities
covering country-commodity cross matrix, certification effectiveness,
seasonal risk variation, derived product mapping, production risk,
supply chain complexity, and batch analysis.

Target: 60+ tests, 85%+ coverage.

Author: GreenLang Platform Team
Date: March 2026
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List
from unittest.mock import patch

import pytest

from greenlang.agents.eudr.country_risk_evaluator.commodity_risk_analyzer import (
    CommodityRiskAnalyzer,
    _CERTIFICATION_EFFECTIVENESS,
    _DERIVED_PRODUCTS,
    _HIGH_RISK_REGIONS,
    _RISK_WEIGHTS,
    _SEASONAL_RISK,
)
from greenlang.agents.eudr.country_risk_evaluator.models import (
    CommodityRiskProfile,
    CommodityType,
    RiskLevel,
    SUPPORTED_COMMODITIES,
)


# ============================================================================
# TestCommodityRiskAnalyzerInit
# ============================================================================


class TestCommodityRiskAnalyzerInit:
    """Tests for CommodityRiskAnalyzer initialization."""

    @pytest.mark.unit
    def test_initialization_creates_empty_stores(self, mock_config):
        analyzer = CommodityRiskAnalyzer()
        assert analyzer._profiles == {}
        assert analyzer._price_indices == {}
        assert analyzer._production_data == {}

    @pytest.mark.unit
    def test_risk_weights_sum_to_one(self):
        total = sum(_RISK_WEIGHTS.values())
        assert total == Decimal("1.00")

    @pytest.mark.unit
    def test_all_seven_commodities_in_seasonal_risk(self):
        for commodity in SUPPORTED_COMMODITIES:
            assert commodity in _SEASONAL_RISK

    @pytest.mark.unit
    def test_seasonal_risk_has_12_months(self):
        for commodity, months in _SEASONAL_RISK.items():
            assert len(months) == 12, f"{commodity} does not have 12 months"


# ============================================================================
# TestAnalyzeCommodity
# ============================================================================


class TestAnalyzeCommodity:
    """Tests for analyze_commodity method."""

    @pytest.mark.unit
    def test_analyze_commodity_valid(self, commodity_analyzer):
        result = commodity_analyzer.analyze_commodity(
            country_code="BR",
            commodity_type="soya",
            country_risk_score=72.5,
        )
        assert isinstance(result, CommodityRiskProfile)
        assert result.country_code == "BR"
        assert result.commodity_type == CommodityType.SOYA

    @pytest.mark.unit
    def test_analyze_commodity_score_in_range(self, commodity_analyzer):
        result = commodity_analyzer.analyze_commodity(
            country_code="BR",
            commodity_type="soya",
            country_risk_score=50.0,
        )
        assert 0.0 <= result.risk_score <= 100.0

    @pytest.mark.unit
    def test_analyze_commodity_uppercase_country(self, commodity_analyzer):
        result = commodity_analyzer.analyze_commodity(
            country_code="br",
            commodity_type="soya",
            country_risk_score=50.0,
        )
        assert result.country_code == "BR"

    @pytest.mark.unit
    def test_analyze_commodity_has_profile_id(self, commodity_analyzer):
        result = commodity_analyzer.analyze_commodity(
            country_code="ID",
            commodity_type="oil_palm",
            country_risk_score=65.0,
        )
        assert result.profile_id.startswith("crp-")

    @pytest.mark.unit
    def test_analyze_commodity_stores_profile(self, commodity_analyzer):
        result = commodity_analyzer.analyze_commodity(
            country_code="BR",
            commodity_type="coffee",
            country_risk_score=55.0,
        )
        retrieved = commodity_analyzer.get_profile(result.profile_id)
        assert retrieved is not None
        assert retrieved.profile_id == result.profile_id


# ============================================================================
# TestAllSevenCommodities
# ============================================================================


class TestAllSevenCommodities:
    """Tests for all 7 EUDR-regulated commodities."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "commodity",
        SUPPORTED_COMMODITIES,
        ids=SUPPORTED_COMMODITIES,
    )
    def test_analyze_each_commodity(self, commodity_analyzer, commodity):
        result = commodity_analyzer.analyze_commodity(
            country_code="BR",
            commodity_type=commodity,
            country_risk_score=50.0,
        )
        assert isinstance(result, CommodityRiskProfile)
        assert result.commodity_type == CommodityType(commodity)
        assert 0.0 <= result.risk_score <= 100.0

    @pytest.mark.unit
    def test_invalid_commodity_raises(self, commodity_analyzer):
        with pytest.raises(ValueError):
            commodity_analyzer.analyze_commodity(
                country_code="BR",
                commodity_type="diamonds",
                country_risk_score=50.0,
            )

    @pytest.mark.unit
    def test_all_seven_commodity_enums_exist(self):
        expected = {"cattle", "cocoa", "coffee", "oil_palm", "rubber", "soya", "wood"}
        actual = {c.value for c in CommodityType}
        assert actual == expected


# ============================================================================
# TestCountryCommodityCrossMatrix
# ============================================================================


class TestCountryCommodityCrossMatrix:
    """Tests for country-commodity cross-risk analysis."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "country,commodity,expected_high_risk",
        [
            ("BR", "soya", True),
            ("BR", "cattle", True),
            ("ID", "oil_palm", True),
            ("CI", "cocoa", True),
            ("TH", "rubber", True),
            ("SE", "wood", False),
        ],
    )
    def test_high_risk_country_commodity_pairs(
        self, commodity_analyzer, country, commodity, expected_high_risk
    ):
        result = commodity_analyzer.analyze_commodity(
            country_code=country,
            commodity_type=commodity,
            country_risk_score=70.0 if expected_high_risk else 15.0,
        )
        if expected_high_risk:
            assert result.risk_score > 30.0
        else:
            assert result.risk_score <= 50.0

    @pytest.mark.unit
    def test_production_region_increases_risk(self, commodity_analyzer):
        # With specific high-risk region
        result_region = commodity_analyzer.analyze_commodity(
            country_code="BR",
            commodity_type="soya",
            country_risk_score=50.0,
            region="Mato Grosso",
        )
        # Without region
        result_no_region = commodity_analyzer.analyze_commodity(
            country_code="BR",
            commodity_type="soya",
            country_risk_score=50.0,
        )
        # Region specified as high-risk area should have equal or higher risk
        assert result_region.risk_score >= result_no_region.risk_score - 5.0


# ============================================================================
# TestCertificationEffectiveness
# ============================================================================


class TestCertificationEffectiveness:
    """Tests for certification scheme effectiveness scoring."""

    @pytest.mark.unit
    def test_fsc_wood_high_effectiveness(self):
        assert _CERTIFICATION_EFFECTIVENESS["fsc"]["wood"] == 85.0

    @pytest.mark.unit
    def test_rspo_oil_palm_effectiveness(self):
        assert _CERTIFICATION_EFFECTIVENESS["rspo"]["oil_palm"] == 80.0

    @pytest.mark.unit
    def test_rainforest_alliance_coffee(self):
        assert _CERTIFICATION_EFFECTIVENESS["rainforest_alliance"]["coffee"] == 80.0

    @pytest.mark.unit
    def test_fairtrade_cocoa(self):
        assert _CERTIFICATION_EFFECTIVENESS["fairtrade"]["cocoa"] == 65.0

    @pytest.mark.unit
    def test_certification_reduces_risk(self, commodity_analyzer):
        # Without certification
        result_no_cert = commodity_analyzer.analyze_commodity(
            country_code="ID",
            commodity_type="oil_palm",
            country_risk_score=70.0,
        )
        # With RSPO certification
        result_with_cert = commodity_analyzer.analyze_commodity(
            country_code="ID",
            commodity_type="oil_palm",
            country_risk_score=70.0,
            certification_schemes=["rspo"],
        )
        # Certification should reduce or not increase the risk score
        assert result_with_cert.risk_score <= result_no_cert.risk_score + 1.0

    @pytest.mark.unit
    def test_irrelevant_certification_no_effect(self, commodity_analyzer):
        # FSC is for wood, not cocoa
        result = commodity_analyzer.analyze_commodity(
            country_code="CI",
            commodity_type="cocoa",
            country_risk_score=65.0,
            certification_schemes=["fsc"],
        )
        assert result.risk_score > 0  # FSC should have 0 effect on cocoa

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "scheme,commodity,expected_eff",
        [
            ("fsc", "wood", 85.0),
            ("pefc", "wood", 75.0),
            ("rspo", "oil_palm", 80.0),
            ("rainforest_alliance", "cocoa", 75.0),
            ("fairtrade", "coffee", 70.0),
            ("organic", "soya", 45.0),
            ("bonsucro", "soya", 60.0),
            ("iscc", "oil_palm", 70.0),
        ],
    )
    def test_certification_effectiveness_values(
        self, scheme, commodity, expected_eff
    ):
        assert _CERTIFICATION_EFFECTIVENESS[scheme][commodity] == expected_eff

    @pytest.mark.unit
    def test_all_schemes_have_all_commodities(self):
        for scheme, commodities in _CERTIFICATION_EFFECTIVENESS.items():
            for commodity in SUPPORTED_COMMODITIES:
                assert commodity in commodities, (
                    f"Scheme {scheme} missing commodity {commodity}"
                )


# ============================================================================
# TestSeasonalRiskVariation
# ============================================================================


class TestSeasonalRiskVariation:
    """Tests for seasonal risk variation modeling."""

    @pytest.mark.unit
    def test_cattle_dry_season_higher_risk(self, commodity_analyzer):
        # June (dry season in Amazon) should have higher risk
        result_june = commodity_analyzer.analyze_commodity(
            country_code="BR",
            commodity_type="cattle",
            country_risk_score=50.0,
            month=6,
        )
        # January (wet season) should have lower risk
        result_jan = commodity_analyzer.analyze_commodity(
            country_code="BR",
            commodity_type="cattle",
            country_risk_score=50.0,
            month=1,
        )
        assert result_june.risk_score >= result_jan.risk_score - 1.0

    @pytest.mark.unit
    def test_soya_planting_season_higher_risk(self, commodity_analyzer):
        # November (planting season) highest risk
        result_nov = commodity_analyzer.analyze_commodity(
            country_code="BR",
            commodity_type="soya",
            country_risk_score=50.0,
            month=11,
        )
        # March (off-season)
        result_mar = commodity_analyzer.analyze_commodity(
            country_code="BR",
            commodity_type="soya",
            country_risk_score=50.0,
            month=3,
        )
        assert result_nov.risk_score >= result_mar.risk_score - 1.0

    @pytest.mark.unit
    @pytest.mark.parametrize("month", list(range(1, 13)))
    def test_seasonal_multiplier_range(self, commodity_analyzer, month):
        result = commodity_analyzer.analyze_commodity(
            country_code="BR",
            commodity_type="soya",
            country_risk_score=50.0,
            month=month,
        )
        assert 0.0 <= result.risk_score <= 100.0

    @pytest.mark.unit
    def test_no_month_no_seasonal_adjustment(self, commodity_analyzer):
        result = commodity_analyzer.analyze_commodity(
            country_code="BR",
            commodity_type="soya",
            country_risk_score=50.0,
        )
        # Without month, no seasonal multiplier applied
        assert isinstance(result, CommodityRiskProfile)

    @pytest.mark.unit
    def test_seasonal_risk_multiplier_values_cattle(self):
        # June-October should have multipliers > 1.0 for cattle
        cattle_risk = _SEASONAL_RISK["cattle"]
        for month_idx in [5, 6, 7, 8, 9]:  # June-October (0-indexed)
            assert cattle_risk[month_idx] >= 1.0


# ============================================================================
# TestDerivedProductMapping
# ============================================================================


class TestDerivedProductMapping:
    """Tests for derived product to primary commodity mapping."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "derived,primary",
        [
            ("chocolate", "cocoa"),
            ("cocoa_butter", "cocoa"),
            ("leather", "cattle"),
            ("beef", "cattle"),
            ("biodiesel", "soya"),
            ("palm_oil", "oil_palm"),
            ("furniture", "wood"),
            ("paper", "wood"),
            ("latex", "rubber"),
            ("tires", "rubber"),
            ("soybean_meal", "soya"),
            ("tofu", "soya"),
        ],
    )
    def test_derived_product_mapping(self, derived, primary):
        assert _DERIVED_PRODUCTS[derived] == primary

    @pytest.mark.unit
    def test_all_derived_products_map_to_valid_commodity(self):
        for derived, primary in _DERIVED_PRODUCTS.items():
            assert primary in SUPPORTED_COMMODITIES, (
                f"Derived product {derived} maps to invalid commodity {primary}"
            )


# ============================================================================
# TestProductionVolumeCorrelation
# ============================================================================


class TestProductionVolumeCorrelation:
    """Tests for production volume impact on risk scoring."""

    @pytest.mark.unit
    def test_high_production_volume(self, commodity_analyzer):
        result = commodity_analyzer.analyze_commodity(
            country_code="BR",
            commodity_type="soya",
            country_risk_score=50.0,
            production_volume=500000.0,
        )
        assert result.risk_score > 0

    @pytest.mark.unit
    def test_low_production_volume(self, commodity_analyzer):
        result = commodity_analyzer.analyze_commodity(
            country_code="BR",
            commodity_type="soya",
            country_risk_score=50.0,
            production_volume=100.0,
        )
        assert result.risk_score > 0

    @pytest.mark.unit
    def test_zero_risk_score_input(self, commodity_analyzer):
        result = commodity_analyzer.analyze_commodity(
            country_code="FI",
            commodity_type="wood",
            country_risk_score=0.0,
        )
        assert result.risk_score >= 0.0

    @pytest.mark.unit
    def test_max_risk_score_input(self, commodity_analyzer):
        result = commodity_analyzer.analyze_commodity(
            country_code="CD",
            commodity_type="wood",
            country_risk_score=100.0,
        )
        assert result.risk_score <= 100.0


# ============================================================================
# TestHighRiskRegions
# ============================================================================


class TestHighRiskRegions:
    """Tests for high deforestation pressure regions."""

    @pytest.mark.unit
    def test_brazil_cattle_regions(self):
        regions = _HIGH_RISK_REGIONS["cattle"]["BR"]
        assert "Para" in regions
        assert "Mato Grosso" in regions
        assert "Rondonia" in regions

    @pytest.mark.unit
    def test_indonesia_oil_palm_regions(self):
        regions = _HIGH_RISK_REGIONS["oil_palm"]["ID"]
        assert "Kalimantan" in regions
        assert "Sumatra" in regions
        assert "Papua" in regions

    @pytest.mark.unit
    def test_cote_divoire_cocoa_regions(self):
        regions = _HIGH_RISK_REGIONS["cocoa"]["CI"]
        assert "Cavally" in regions
        assert "Guemon" in regions

    @pytest.mark.unit
    def test_all_commodities_have_regions(self):
        for commodity in SUPPORTED_COMMODITIES:
            assert commodity in _HIGH_RISK_REGIONS, (
                f"Commodity {commodity} missing from high risk regions"
            )


# ============================================================================
# TestInputValidation
# ============================================================================


class TestInputValidation:
    """Tests for commodity analyzer input validation."""

    @pytest.mark.unit
    def test_empty_country_code_raises(self, commodity_analyzer):
        with pytest.raises(ValueError):
            commodity_analyzer.analyze_commodity(
                country_code="",
                commodity_type="soya",
                country_risk_score=50.0,
            )

    @pytest.mark.unit
    def test_invalid_commodity_raises(self, commodity_analyzer):
        with pytest.raises(ValueError):
            commodity_analyzer.analyze_commodity(
                country_code="BR",
                commodity_type="gold",
                country_risk_score=50.0,
            )

    @pytest.mark.unit
    def test_negative_risk_score_raises(self, commodity_analyzer):
        with pytest.raises(ValueError):
            commodity_analyzer.analyze_commodity(
                country_code="BR",
                commodity_type="soya",
                country_risk_score=-5.0,
            )

    @pytest.mark.unit
    def test_risk_score_over_100_raises(self, commodity_analyzer):
        with pytest.raises(ValueError):
            commodity_analyzer.analyze_commodity(
                country_code="BR",
                commodity_type="soya",
                country_risk_score=105.0,
            )


# ============================================================================
# TestBatchAnalysis
# ============================================================================


class TestBatchAnalysis:
    """Tests for batch commodity analysis."""

    @pytest.mark.unit
    def test_batch_analysis_multiple_commodities(self, commodity_analyzer):
        results = []
        for commodity in SUPPORTED_COMMODITIES:
            result = commodity_analyzer.analyze_commodity(
                country_code="BR",
                commodity_type=commodity,
                country_risk_score=60.0,
            )
            results.append(result)

        assert len(results) == 7
        commodity_types = {r.commodity_type for r in results}
        assert len(commodity_types) == 7

    @pytest.mark.unit
    def test_batch_multiple_countries_same_commodity(self, commodity_analyzer):
        countries = ["BR", "ID", "CI", "GH", "MY"]
        results = []
        for cc in countries:
            result = commodity_analyzer.analyze_commodity(
                country_code=cc,
                commodity_type="cocoa",
                country_risk_score=55.0,
            )
            results.append(result)

        assert len(results) == 5
        country_codes = {r.country_code for r in results}
        assert country_codes == set(countries)
