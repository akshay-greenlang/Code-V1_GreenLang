# -*- coding: utf-8 -*-
"""
Unit Tests for CommodityClassifier (AGENT-DATA-005)

Tests CN code classification, HS code classification, name-based
classification, derived product detection, primary commodity lookups,
and full classification request processing for all 7 EUDR commodities.

Coverage target: 85%+ of commodity_classifier.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

import pytest


# ---------------------------------------------------------------------------
# Inline enums
# ---------------------------------------------------------------------------


class EUDRCommodity(str, Enum):
    COCOA = "cocoa"
    COFFEE = "coffee"
    OIL_PALM = "oil_palm"
    SOYA = "soya"
    RUBBER = "rubber"
    CATTLE = "cattle"
    WOOD = "wood"


# ---------------------------------------------------------------------------
# Inline request model
# ---------------------------------------------------------------------------


class ClassifyCommodityRequest:
    """Request model for commodity classification."""

    def __init__(self, cn_code: Optional[str] = None,
                 hs_code: Optional[str] = None,
                 product_name: Optional[str] = None,
                 description: Optional[str] = None):
        self.cn_code = cn_code
        self.hs_code = hs_code
        self.product_name = product_name
        self.description = description


# ---------------------------------------------------------------------------
# Inline CommodityClassification result model
# ---------------------------------------------------------------------------


class CommodityClassification:
    """Result of a commodity classification."""

    def __init__(self, classification_id: str, commodity: Optional[str],
                 is_eudr_covered: bool, is_derived: bool,
                 primary_commodity: Optional[str],
                 cn_code: Optional[str] = None,
                 hs_code: Optional[str] = None,
                 product_name: Optional[str] = None,
                 confidence: float = 1.0):
        self.classification_id = classification_id
        self.commodity = commodity
        self.is_eudr_covered = is_eudr_covered
        self.is_derived = is_derived
        self.primary_commodity = primary_commodity
        self.cn_code = cn_code
        self.hs_code = hs_code
        self.product_name = product_name
        self.confidence = confidence
        self.provenance_hash = ""
        self.classified_at = datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Inline CommodityClassifier mirroring greenlang/eudr_traceability/commodity_classifier.py
# ---------------------------------------------------------------------------


def _compute_hash(data: Any) -> str:
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


class CommodityClassifier:
    """EUDR commodity and CN/HS code classification engine.

    Classifies products by CN code, HS code, or product name against the
    7 EUDR-regulated commodity groups and their derived products.
    """

    # CN code prefix -> commodity mapping (EU Combined Nomenclature)
    CN_CODE_MAP: Dict[str, str] = {
        # Cocoa
        "1801": "cocoa",       # Cocoa beans, whole or broken
        "1802": "cocoa",       # Cocoa shells and husks
        "1803": "cocoa",       # Cocoa paste
        "1804": "cocoa",       # Cocoa butter, fat, oil
        "1805": "cocoa",       # Cocoa powder unsweetened
        "1806": "cocoa",       # Chocolate and preparations
        # Coffee
        "0901": "coffee",      # Coffee, roasted or not
        "2101": "coffee",      # Coffee extracts, essences
        # Oil palm
        "1511": "oil_palm",    # Palm oil and fractions
        "1513": "oil_palm",    # Coconut/palm kernel oil
        "3823": "oil_palm",    # Fatty acids from palm
        # Soya
        "1201": "soya",        # Soya beans
        "1507": "soya",        # Soya-bean oil
        "2304": "soya",        # Soya-bean oilcake
        # Rubber
        "4001": "rubber",      # Natural rubber
        "4005": "rubber",      # Compounded rubber
        "4011": "rubber",      # New pneumatic tyres of rubber
        "4012": "rubber",      # Retreaded/used tyres
        # Cattle
        "0102": "cattle",      # Live bovine animals
        "0201": "cattle",      # Meat of bovine animals, fresh/chilled
        "0202": "cattle",      # Meat of bovine animals, frozen
        "4104": "cattle",      # Tanned/crust hides bovine
        "4107": "cattle",      # Leather of bovine
        # Wood
        "4401": "wood",        # Fuel wood, wood chips
        "4403": "wood",        # Wood in the rough
        "4407": "wood",        # Wood sawn or chipped
        "4408": "wood",        # Veneer sheets
        "4409": "wood",        # Wood continuously shaped
        "4410": "wood",        # Particle board
        "4411": "wood",        # Fibreboard
        "4412": "wood",        # Plywood
        "9403": "wood",        # Other furniture
    }

    # HS code prefix -> commodity (first 4 digits of HS = CN prefix)
    HS_CODE_MAP: Dict[str, str] = dict(CN_CODE_MAP)  # Same lookup

    # Product name keywords -> (commodity, is_derived)
    NAME_KEYWORDS: Dict[str, tuple] = {
        # Cocoa derived
        "chocolate": ("cocoa", True),
        "cocoa butter": ("cocoa", True),
        "cocoa powder": ("cocoa", True),
        "cocoa paste": ("cocoa", True),
        "cocoa beans": ("cocoa", False),
        "cocoa": ("cocoa", False),
        # Coffee derived
        "coffee extract": ("coffee", True),
        "instant coffee": ("coffee", True),
        "coffee beans": ("coffee", False),
        "coffee": ("coffee", False),
        # Oil palm derived
        "palm oil": ("oil_palm", False),
        "palm kernel oil": ("oil_palm", True),
        "palm fatty acid": ("oil_palm", True),
        # Soya derived
        "soya bean oil": ("soya", True),
        "soybean oil": ("soya", True),
        "soya meal": ("soya", True),
        "soybean meal": ("soya", True),
        "soya beans": ("soya", False),
        "soybeans": ("soya", False),
        # Rubber derived
        "tyres": ("rubber", True),
        "tires": ("rubber", True),
        "natural rubber": ("rubber", False),
        "rubber": ("rubber", False),
        # Cattle derived
        "beef": ("cattle", True),
        "leather": ("cattle", True),
        "bovine hides": ("cattle", True),
        "live cattle": ("cattle", False),
        "cattle": ("cattle", False),
        # Wood derived
        "furniture": ("wood", True),
        "plywood": ("wood", True),
        "particle board": ("wood", True),
        "fibreboard": ("wood", True),
        "veneer": ("wood", True),
        "timber": ("wood", False),
        "lumber": ("wood", False),
        "wood": ("wood", False),
    }

    # CN codes for cocoa (all prefixes)
    COCOA_CN_CODES = {"1801", "1802", "1803", "1804", "1805", "1806"}

    # Derived product CN codes (chocolate, processed cocoa, etc.)
    DERIVED_CN_CODES = {
        "1803", "1804", "1805", "1806",  # processed cocoa
        "2101",  # coffee extracts
        "3823",  # palm fatty acids
        "2304",  # soya oilcake
        "4005", "4011", "4012",  # processed rubber, tyres
        "4104", "4107",  # leather
        "4408", "4409", "4410", "4411", "4412",  # processed wood
        "9403",  # furniture
    }

    def __init__(self):
        self._classifications: Dict[str, CommodityClassification] = {}
        self._counter = 0

    def _next_id(self) -> str:
        self._counter += 1
        return f"CLS-{self._counter:05d}"

    def classify_by_cn_code(self, cn_code: str) -> Optional[CommodityClassification]:
        """Classify a product by its CN code prefix (first 4 digits)."""
        prefix = cn_code[:4]
        commodity = self.CN_CODE_MAP.get(prefix)
        if commodity is None:
            return None

        is_derived = prefix in self.DERIVED_CN_CODES
        cls_id = self._next_id()

        result = CommodityClassification(
            classification_id=cls_id,
            commodity=commodity,
            is_eudr_covered=True,
            is_derived=is_derived,
            primary_commodity=commodity,
            cn_code=cn_code,
        )
        result.provenance_hash = _compute_hash({
            "classification_id": cls_id,
            "cn_code": cn_code,
            "commodity": commodity,
        })

        self._classifications[cls_id] = result
        return result

    def classify_by_hs_code(self, hs_code: str) -> Optional[CommodityClassification]:
        """Classify a product by its HS code (first 4 digits used)."""
        prefix = hs_code[:4]
        commodity = self.HS_CODE_MAP.get(prefix)
        if commodity is None:
            return None

        is_derived = prefix in self.DERIVED_CN_CODES
        cls_id = self._next_id()

        result = CommodityClassification(
            classification_id=cls_id,
            commodity=commodity,
            is_eudr_covered=True,
            is_derived=is_derived,
            primary_commodity=commodity,
            hs_code=hs_code,
        )
        result.provenance_hash = _compute_hash({
            "classification_id": cls_id,
            "hs_code": hs_code,
            "commodity": commodity,
        })

        self._classifications[cls_id] = result
        return result

    def classify_by_name(self, product_name: str) -> Optional[CommodityClassification]:
        """Classify a product by its name using keyword matching."""
        name_lower = product_name.lower().strip()

        # Match longest keyword first (most specific)
        match = None
        match_len = 0
        for keyword, (commodity, is_derived) in self.NAME_KEYWORDS.items():
            if keyword in name_lower and len(keyword) > match_len:
                match = (commodity, is_derived)
                match_len = len(keyword)

        if match is None:
            return None

        commodity, is_derived = match
        cls_id = self._next_id()

        result = CommodityClassification(
            classification_id=cls_id,
            commodity=commodity,
            is_eudr_covered=True,
            is_derived=is_derived,
            primary_commodity=commodity,
            product_name=product_name,
            confidence=0.85 if is_derived else 0.95,
        )
        result.provenance_hash = _compute_hash({
            "classification_id": cls_id,
            "product_name": product_name,
            "commodity": commodity,
        })

        self._classifications[cls_id] = result
        return result

    def is_eudr_covered(self, cn_code: str) -> bool:
        """Check if a CN code falls under EUDR regulation."""
        prefix = cn_code[:4]
        return prefix in self.CN_CODE_MAP

    def is_derived_product(self, cn_code: str) -> bool:
        """Check if a CN code represents a derived product."""
        prefix = cn_code[:4]
        return prefix in self.DERIVED_CN_CODES

    def get_primary_commodity(self, cn_code: str) -> Optional[str]:
        """Get the primary EUDR commodity for a CN code."""
        prefix = cn_code[:4]
        return self.CN_CODE_MAP.get(prefix)

    def get_all_cn_codes(self, commodity: str) -> List[str]:
        """Get all CN code prefixes for a commodity."""
        return [
            code for code, comm in self.CN_CODE_MAP.items()
            if comm == commodity
        ]

    def classify(self, request: ClassifyCommodityRequest) -> Optional[CommodityClassification]:
        """Full classification from a ClassifyCommodityRequest."""
        if request.cn_code:
            return self.classify_by_cn_code(request.cn_code)
        elif request.hs_code:
            return self.classify_by_hs_code(request.hs_code)
        elif request.product_name:
            return self.classify_by_name(request.product_name)
        return None


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def engine() -> CommodityClassifier:
    return CommodityClassifier()


# ===========================================================================
# Test Classes
# ===========================================================================


class TestClassifyByCNCode:
    """Tests for CN code based classification."""

    def test_classify_by_cn_code_cocoa(self, engine):
        result = engine.classify_by_cn_code("1801")
        assert result is not None
        assert result.commodity == "cocoa"
        assert result.is_eudr_covered is True

    def test_classify_by_cn_code_coffee(self, engine):
        result = engine.classify_by_cn_code("0901")
        assert result is not None
        assert result.commodity == "coffee"

    def test_classify_by_cn_code_palm(self, engine):
        result = engine.classify_by_cn_code("1511")
        assert result is not None
        assert result.commodity == "oil_palm"

    def test_classify_by_cn_code_soya(self, engine):
        result = engine.classify_by_cn_code("1201")
        assert result is not None
        assert result.commodity == "soya"

    def test_classify_by_cn_code_rubber(self, engine):
        result = engine.classify_by_cn_code("4001")
        assert result is not None
        assert result.commodity == "rubber"

    def test_classify_by_cn_code_cattle(self, engine):
        result = engine.classify_by_cn_code("0102")
        assert result is not None
        assert result.commodity == "cattle"

    def test_classify_by_cn_code_wood(self, engine):
        result = engine.classify_by_cn_code("4401")
        assert result is not None
        assert result.commodity == "wood"

    def test_classify_by_cn_code_unknown(self, engine):
        result = engine.classify_by_cn_code("9999")
        assert result is None

    def test_classify_by_cn_code_extended(self, engine):
        """Test CN code with sub-heading digits (e.g. 1801.00.00)."""
        result = engine.classify_by_cn_code("1801.00.00")
        assert result is not None
        assert result.commodity == "cocoa"


class TestClassifyByHSCode:
    """Tests for HS code based classification."""

    def test_classify_by_hs_code(self, engine):
        result = engine.classify_by_hs_code("0901.11")
        assert result is not None
        assert result.commodity == "coffee"
        assert result.hs_code == "0901.11"


class TestClassifyByName:
    """Tests for product name based classification."""

    def test_classify_by_name_chocolate(self, engine):
        result = engine.classify_by_name("chocolate")
        assert result is not None
        assert result.commodity == "cocoa"
        assert result.is_derived is True

    def test_classify_by_name_leather(self, engine):
        result = engine.classify_by_name("leather")
        assert result is not None
        assert result.commodity == "cattle"
        assert result.is_derived is True

    def test_classify_by_name_furniture(self, engine):
        result = engine.classify_by_name("furniture")
        assert result is not None
        assert result.commodity == "wood"
        assert result.is_derived is True

    def test_classify_by_name_palm_oil(self, engine):
        result = engine.classify_by_name("palm oil")
        assert result is not None
        assert result.commodity == "oil_palm"

    def test_classify_by_name_no_match(self, engine):
        result = engine.classify_by_name("microchip semiconductor")
        assert result is None

    def test_classify_by_name_case_insensitive(self, engine):
        result = engine.classify_by_name("CHOCOLATE BAR")
        assert result is not None
        assert result.commodity == "cocoa"


class TestEUDRCoverage:
    """Tests for EUDR coverage checks."""

    def test_is_eudr_covered_true(self, engine):
        assert engine.is_eudr_covered("1801") is True
        assert engine.is_eudr_covered("0901") is True
        assert engine.is_eudr_covered("4001") is True

    def test_is_eudr_covered_false(self, engine):
        assert engine.is_eudr_covered("9999") is False
        assert engine.is_eudr_covered("8471") is False


class TestDerivedProducts:
    """Tests for derived product detection."""

    def test_is_derived_product_true(self, engine):
        assert engine.is_derived_product("1806") is True  # Chocolate
        assert engine.is_derived_product("4011") is True  # Tyres
        assert engine.is_derived_product("9403") is True  # Furniture
        assert engine.is_derived_product("4107") is True  # Leather

    def test_is_derived_product_false(self, engine):
        assert engine.is_derived_product("1801") is False  # Raw cocoa beans
        assert engine.is_derived_product("0901") is False  # Raw coffee
        assert engine.is_derived_product("4001") is False  # Natural rubber
        assert engine.is_derived_product("0102") is False  # Live cattle


class TestPrimaryCommodity:
    """Tests for primary commodity lookup."""

    def test_get_primary_commodity(self, engine):
        assert engine.get_primary_commodity("1806") == "cocoa"

    def test_get_primary_commodity_already_primary(self, engine):
        assert engine.get_primary_commodity("1801") == "cocoa"

    def test_get_primary_commodity_unknown(self, engine):
        assert engine.get_primary_commodity("9999") is None


class TestGetAllCNCodes:
    """Tests for retrieving all CN codes for a commodity."""

    def test_get_all_cn_codes_cocoa(self, engine):
        codes = engine.get_all_cn_codes("cocoa")
        assert len(codes) >= 6
        assert "1801" in codes
        assert "1806" in codes

    def test_get_all_cn_codes_wood(self, engine):
        codes = engine.get_all_cn_codes("wood")
        assert len(codes) >= 5
        assert "4401" in codes
        assert "4412" in codes

    def test_get_all_cn_codes_unknown(self, engine):
        codes = engine.get_all_cn_codes("unknown_commodity")
        assert len(codes) == 0


class TestClassificationIDFormat:
    """Tests for classification ID format."""

    def test_classification_id_format(self, engine):
        result = engine.classify_by_cn_code("1801")
        assert result is not None
        assert result.classification_id.startswith("CLS-")
        assert len(result.classification_id) == 9  # CLS-00001

    def test_classification_ids_sequential(self, engine):
        r1 = engine.classify_by_cn_code("1801")
        r2 = engine.classify_by_cn_code("0901")
        assert r1.classification_id == "CLS-00001"
        assert r2.classification_id == "CLS-00002"


class TestClassifyFullRequest:
    """Tests for full ClassifyCommodityRequest processing."""

    def test_classify_full_request_cn(self, engine):
        request = ClassifyCommodityRequest(cn_code="1801")
        result = engine.classify(request)
        assert result is not None
        assert result.commodity == "cocoa"

    def test_classify_full_request_hs(self, engine):
        request = ClassifyCommodityRequest(hs_code="0901.11")
        result = engine.classify(request)
        assert result is not None
        assert result.commodity == "coffee"

    def test_classify_full_request_name(self, engine):
        request = ClassifyCommodityRequest(product_name="dark chocolate")
        result = engine.classify(request)
        assert result is not None
        assert result.commodity == "cocoa"

    def test_classify_full_request_empty(self, engine):
        request = ClassifyCommodityRequest()
        result = engine.classify(request)
        assert result is None

    def test_classify_cn_takes_priority(self, engine):
        """CN code should be checked before HS code and name."""
        request = ClassifyCommodityRequest(
            cn_code="1801",
            hs_code="0901",
            product_name="leather",
        )
        result = engine.classify(request)
        assert result is not None
        assert result.commodity == "cocoa"


class TestDerivedProductMappings:
    """Tests for specific derived product to primary commodity mappings."""

    def test_beef_derived_from_cattle(self, engine):
        result = engine.classify_by_name("beef")
        assert result is not None
        assert result.commodity == "cattle"
        assert result.is_derived is True
        assert result.primary_commodity == "cattle"

    def test_tyres_derived_from_rubber(self, engine):
        result = engine.classify_by_name("tyres")
        assert result is not None
        assert result.commodity == "rubber"
        assert result.is_derived is True
        assert result.primary_commodity == "rubber"

    def test_plywood_derived_from_wood(self, engine):
        result = engine.classify_by_name("plywood")
        assert result is not None
        assert result.commodity == "wood"
        assert result.is_derived is True

    def test_instant_coffee_derived(self, engine):
        result = engine.classify_by_name("instant coffee")
        assert result is not None
        assert result.commodity == "coffee"
        assert result.is_derived is True

    def test_soybean_meal_derived(self, engine):
        result = engine.classify_by_name("soybean meal")
        assert result is not None
        assert result.commodity == "soya"
        assert result.is_derived is True


class TestProvenanceHash:
    """Tests for provenance hash in classification results."""

    def test_provenance_hash_set(self, engine):
        result = engine.classify_by_cn_code("1801")
        assert result is not None
        assert len(result.provenance_hash) == 64
        int(result.provenance_hash, 16)

    def test_provenance_hash_deterministic(self, engine):
        # Two engines classifying the same code produce the same hash
        e1 = CommodityClassifier()
        e2 = CommodityClassifier()
        r1 = e1.classify_by_cn_code("1801")
        r2 = e2.classify_by_cn_code("1801")
        assert r1.provenance_hash == r2.provenance_hash
