# -*- coding: utf-8 -*-
"""
PACK-006 EUDR Starter Pack - Commodity Classification Engine Tests
====================================================================

Validates the commodity classification engine including CN code
classification for all 7 EUDR commodities, EUDR coverage checks,
derived product identification, HS-to-CN mapping, CN code search,
multi-commodity identification, and CN code validation.

Test count: 15
Author: GreenLang QA Team
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from typing import Any, Dict, List

import pytest

from conftest import (
    EUDR_COMMODITIES,
    ANNEX_I_CN_CODES,
    ALL_CN_CODES,
)


# ---------------------------------------------------------------------------
# Commodity Classification Engine Simulator
# ---------------------------------------------------------------------------

class CommodityClassificationSimulator:
    """Simulates commodity classification engine operations."""

    CN_TO_COMMODITY = {}
    for commodity, codes in ANNEX_I_CN_CODES.items():
        for code in codes:
            CN_TO_COMMODITY[code] = commodity

    DERIVED_PRODUCTS = {
        "palm_oil": ["margarine", "biodiesel", "soap", "cosmetics", "snack_foods"],
        "cocoa": ["chocolate", "cocoa_powder", "cocoa_butter", "confectionery"],
        "coffee": ["instant_coffee", "coffee_extract", "roasted_coffee"],
        "cattle": ["leather", "beef", "gelatin", "tallow"],
        "rubber": ["tires", "gloves", "gaskets", "footwear"],
        "soya": ["soybean_meal", "soy_oil", "tofu", "animal_feed"],
        "wood": ["furniture", "plywood", "paper", "charcoal", "pulp"],
    }

    def classify_cn_code(self, cn_code: str) -> Dict[str, Any]:
        """Classify a CN code to its EUDR commodity."""
        normalized = cn_code.strip()
        commodity = self.CN_TO_COMMODITY.get(normalized)
        if commodity:
            return {
                "cn_code": normalized,
                "commodity": commodity,
                "eudr_covered": True,
                "annex_i": True,
            }
        return {
            "cn_code": normalized,
            "commodity": None,
            "eudr_covered": False,
            "annex_i": False,
        }

    def is_eudr_covered(self, cn_code: str) -> bool:
        """Check if a CN code is covered by EUDR."""
        return cn_code.strip() in self.CN_TO_COMMODITY

    def get_derived_products(self, commodity: str) -> List[str]:
        """Get list of derived products for a commodity."""
        return self.DERIVED_PRODUCTS.get(commodity, [])

    def map_hs_to_cn(self, hs_code: str) -> List[str]:
        """Map an HS code (first 6 digits) to matching CN codes."""
        hs_prefix = hs_code.replace(" ", "")[:6]
        matches = []
        for cn_code in ALL_CN_CODES:
            cn_normalized = cn_code.replace(" ", "")
            if cn_normalized.startswith(hs_prefix):
                matches.append(cn_code)
        return matches

    def search_cn_codes(self, query: str) -> List[Dict[str, str]]:
        """Search CN codes by commodity name or prefix."""
        results = []
        query_lower = query.lower().replace("_", " ")
        # Search by commodity name
        for commodity, codes in ANNEX_I_CN_CODES.items():
            if query_lower in commodity.lower().replace("_", " "):
                for code in codes:
                    results.append({"cn_code": code, "commodity": commodity})
        # Search by CN code prefix
        if not results:
            for code in ALL_CN_CODES:
                if code.startswith(query):
                    commodity = self.CN_TO_COMMODITY.get(code, "unknown")
                    results.append({"cn_code": code, "commodity": commodity})
        return results

    def get_all_annex_i_codes(self) -> Dict[str, List[str]]:
        """Get all Annex I CN codes grouped by commodity."""
        return ANNEX_I_CN_CODES.copy()

    def identify_multi_commodity(self, cn_codes: List[str]) -> Dict[str, Any]:
        """Identify multiple commodities from a list of CN codes."""
        commodities_found = {}
        for code in cn_codes:
            result = self.classify_cn_code(code)
            if result["eudr_covered"]:
                commodity = result["commodity"]
                if commodity not in commodities_found:
                    commodities_found[commodity] = []
                commodities_found[commodity].append(code)
        return {
            "commodities": list(commodities_found.keys()),
            "commodity_count": len(commodities_found),
            "codes_by_commodity": commodities_found,
            "total_eudr_codes": sum(len(v) for v in commodities_found.values()),
        }

    def validate_cn_code(self, cn_code: str) -> Dict[str, Any]:
        """Validate CN code format (XXXX XX XX)."""
        normalized = cn_code.strip()
        parts = normalized.split(" ")
        if len(parts) == 3 and len(parts[0]) == 4 and len(parts[1]) == 2 and len(parts[2]) == 2:
            all_digits = all(p.isdigit() for p in parts)
            return {
                "cn_code": normalized,
                "format_valid": all_digits,
                "eudr_covered": self.is_eudr_covered(normalized) if all_digits else False,
            }
        return {
            "cn_code": normalized,
            "format_valid": False,
            "eudr_covered": False,
        }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCommodityClassification:
    """Tests for the commodity classification engine."""

    @pytest.fixture
    def engine(self) -> CommodityClassificationSimulator:
        return CommodityClassificationSimulator()

    # 1
    def test_classify_cattle_cn_code(self, engine):
        """Cattle CN code 0102 29 10 classifies correctly."""
        result = engine.classify_cn_code("0102 29 10")
        assert result["commodity"] == "cattle"
        assert result["eudr_covered"] is True

    # 2
    def test_classify_cocoa_cn_code(self, engine):
        """Cocoa CN code 1801 00 00 classifies correctly."""
        result = engine.classify_cn_code("1801 00 00")
        assert result["commodity"] == "cocoa"
        assert result["eudr_covered"] is True

    # 3
    def test_classify_coffee_cn_code(self, engine):
        """Coffee CN code 0901 11 00 classifies correctly."""
        result = engine.classify_cn_code("0901 11 00")
        assert result["commodity"] == "coffee"
        assert result["eudr_covered"] is True

    # 4
    def test_classify_palm_oil_cn_code(self, engine):
        """Palm oil CN code 1511 10 90 classifies correctly."""
        result = engine.classify_cn_code("1511 10 90")
        assert result["commodity"] == "palm_oil"
        assert result["eudr_covered"] is True

    # 5
    def test_classify_rubber_cn_code(self, engine):
        """Rubber CN code 4001 10 00 classifies correctly."""
        result = engine.classify_cn_code("4001 10 00")
        assert result["commodity"] == "rubber"
        assert result["eudr_covered"] is True

    # 6
    def test_classify_soya_cn_code(self, engine):
        """Soya CN code 1201 90 00 classifies correctly."""
        result = engine.classify_cn_code("1201 90 00")
        assert result["commodity"] == "soya"
        assert result["eudr_covered"] is True

    # 7
    def test_classify_wood_cn_code(self, engine):
        """Wood CN code 4403 49 00 classifies correctly."""
        result = engine.classify_cn_code("4403 49 00")
        assert result["commodity"] == "wood"
        assert result["eudr_covered"] is True

    # 8
    def test_is_eudr_covered_true(self, engine):
        """Known EUDR CN codes return True."""
        assert engine.is_eudr_covered("1511 10 90") is True
        assert engine.is_eudr_covered("0901 11 00") is True

    # 9
    def test_is_eudr_covered_false(self, engine):
        """Non-EUDR CN codes return False."""
        assert engine.is_eudr_covered("8471 30 00") is False  # computers
        assert engine.is_eudr_covered("9999 99 99") is False

    # 10
    def test_derived_products(self, engine):
        """Each commodity has associated derived products."""
        for commodity in EUDR_COMMODITIES:
            products = engine.get_derived_products(commodity)
            assert len(products) >= 3, (
                f"Expected >= 3 derived products for {commodity}, got {len(products)}"
            )

    # 11
    def test_map_hs_to_cn(self, engine):
        """HS code 1511 maps to palm oil CN codes."""
        matches = engine.map_hs_to_cn("151110")
        assert len(matches) >= 1
        for code in matches:
            assert code.startswith("1511 10")

    # 12
    def test_search_cn_codes(self, engine):
        """Search by commodity name returns matching codes."""
        results = engine.search_cn_codes("palm oil")
        assert len(results) >= 5
        for r in results:
            assert r["commodity"] == "palm_oil"

    # 13
    def test_all_annex_i_codes(self, engine):
        """All Annex I codes grouped by commodity are accessible."""
        all_codes = engine.get_all_annex_i_codes()
        assert len(all_codes) == 7
        for commodity in EUDR_COMMODITIES:
            assert commodity in all_codes
            assert len(all_codes[commodity]) > 0

    # 14
    def test_multi_commodity_identification(self, engine):
        """Multiple commodities identified from mixed CN code list."""
        cn_codes = [
            "1511 10 90",   # palm_oil
            "0901 11 00",   # coffee
            "4403 49 00",   # wood
            "8471 30 00",   # not EUDR
        ]
        result = engine.identify_multi_commodity(cn_codes)
        assert result["commodity_count"] == 3
        assert "palm_oil" in result["commodities"]
        assert "coffee" in result["commodities"]
        assert "wood" in result["commodities"]
        assert result["total_eudr_codes"] == 3

    # 15
    def test_validate_cn_code(self, engine):
        """CN code format validation detects valid and invalid formats."""
        valid = engine.validate_cn_code("1511 10 90")
        assert valid["format_valid"] is True
        assert valid["eudr_covered"] is True

        invalid = engine.validate_cn_code("15111090")
        assert invalid["format_valid"] is False
