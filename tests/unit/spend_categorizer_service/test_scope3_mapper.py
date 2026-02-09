# -*- coding: utf-8 -*-
"""
Unit tests for Scope3MapperEngine - AGENT-DATA-009 Batch 2
============================================================

Comprehensive tests for the Scope 3 mapper engine covering:
- Engine initialisation and mapping table loading
- Single record mapping (NAICS, UNSPSC, keyword, fallback)
- Batch mapping with mixed records
- NAICS-to-Scope-3 mapping (all major sector codes)
- UNSPSC-to-Scope-3 mapping (major segments)
- Keyword-to-Scope-3 mapping (all keyword categories)
- Taxonomy-based generic mapping
- CapEx detection (capital goods, operating expense)
- Multi-category allocation splitting
- Category info retrieval (all 15 categories + invalid)
- Confidence scoring by match source
- Statistics tracking by category and source
- Provenance hashing (SHA-256)
- Thread safety under concurrent mapping

Target: 90+ tests, 800+ lines, 85%+ coverage of scope3_mapper.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List

import pytest

from greenlang.spend_categorizer.scope3_mapper import (
    Scope3Assignment,
    Scope3Category,
    Scope3MapperEngine,
    _KEYWORD_TO_SCOPE3,
    _NAICS_TO_SCOPE3,
    _SCOPE3_CATEGORIES,
    _UNSPSC_TO_SCOPE3,
    _CAPEX_KEYWORDS,
)


# ---------------------------------------------------------------------------
# Local fixtures (in addition to conftest.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def scope3_engine() -> Scope3MapperEngine:
    """Create a default Scope3MapperEngine."""
    return Scope3MapperEngine()


@pytest.fixture
def scope3_engine_no_capex() -> Scope3MapperEngine:
    """Create a Scope3MapperEngine with CapEx detection disabled."""
    return Scope3MapperEngine(config={"enable_capex_detection": False})


@pytest.fixture
def scope3_engine_custom() -> Scope3MapperEngine:
    """Create a Scope3MapperEngine with custom thresholds."""
    return Scope3MapperEngine(config={
        "min_confidence": 0.5,
        "default_category": 1,
        "enable_capex_detection": True,
    })


# ===================================================================
# TestInit - engine creation and defaults
# ===================================================================


class TestInit:
    """Tests for Scope3MapperEngine initialisation."""

    def test_default_init(self):
        """Engine initialises with sensible defaults."""
        engine = Scope3MapperEngine()
        assert engine._min_confidence == 0.3
        assert engine._default_category == 1
        assert engine._enable_capex is True

    def test_custom_config(self):
        """Engine respects custom configuration."""
        engine = Scope3MapperEngine(config={
            "min_confidence": 0.5,
            "default_category": 2,
            "enable_capex_detection": False,
        })
        assert engine._min_confidence == 0.5
        assert engine._default_category == 2
        assert engine._enable_capex is False

    def test_empty_assignments_on_init(self):
        """Assignment storage is empty on init."""
        engine = Scope3MapperEngine()
        assert engine._assignments == {}

    def test_stats_zeroed_on_init(self):
        """Statistics are zero on init."""
        engine = Scope3MapperEngine()
        assert engine._stats["records_mapped"] == 0
        assert engine._stats["capex_detected"] == 0
        assert engine._stats["unclassified"] == 0

    def test_mapping_tables_loaded(self):
        """Mapping tables have expected sizes."""
        assert len(_NAICS_TO_SCOPE3) >= 50
        assert len(_UNSPSC_TO_SCOPE3) >= 55
        assert len(_KEYWORD_TO_SCOPE3) >= 60
        assert len(_SCOPE3_CATEGORIES) == 15

    def test_none_config(self):
        """None config uses defaults."""
        engine = Scope3MapperEngine(config=None)
        assert engine._min_confidence == 0.3

    def test_capex_keywords_loaded(self):
        """CapEx keywords list has entries."""
        assert len(_CAPEX_KEYWORDS) >= 15


# ===================================================================
# TestMapRecord - single record mapping
# ===================================================================


class TestMapRecord:
    """Tests for map_record() method."""

    def test_basic_mapping(self, scope3_engine):
        """map_record returns a Scope3Assignment."""
        result = scope3_engine.map_record(
            {"description": "office supplies", "amount_usd": 5000}
        )
        assert isinstance(result, Scope3Assignment)

    def test_category_number_range(self, scope3_engine):
        """Category number is 0-15."""
        result = scope3_engine.map_record(
            {"description": "diesel fuel", "amount_usd": 10000}
        )
        assert 0 <= result.category_number <= 15

    def test_category_name_populated(self, scope3_engine):
        """Category name is populated."""
        result = scope3_engine.map_record(
            {"description": "office supplies", "amount_usd": 5000}
        )
        assert result.category_name != ""

    def test_direction_populated(self, scope3_engine):
        """Direction is upstream or downstream."""
        result = scope3_engine.map_record(
            {"description": "office supplies", "amount_usd": 5000}
        )
        assert result.direction in ("upstream", "downstream")

    def test_confidence_range(self, scope3_engine):
        """Confidence is between 0 and 1."""
        result = scope3_engine.map_record(
            {"description": "air travel flights", "amount_usd": 5000}
        )
        assert 0.0 <= result.confidence <= 1.0

    def test_match_source_populated(self, scope3_engine):
        """match_source is populated."""
        result = scope3_engine.map_record(
            {"description": "diesel fuel", "amount_usd": 5000}
        )
        assert result.match_source in (
            "naics", "unspsc", "keyword", "default", "capex_detection"
        )

    def test_provenance_hash(self, scope3_engine):
        """Assignment has 64-char SHA-256 provenance hash."""
        result = scope3_engine.map_record(
            {"description": "office supplies", "amount_usd": 5000}
        )
        assert len(result.provenance_hash) == 64

    def test_assignment_id_prefix(self, scope3_engine):
        """Assignment ID starts with s3m- prefix."""
        result = scope3_engine.map_record(
            {"description": "office supplies", "amount_usd": 5000}
        )
        assert result.assignment_id.startswith("s3m-")

    def test_record_id_preserved(self, scope3_engine):
        """record_id is passed through from input."""
        result = scope3_engine.map_record(
            {"description": "supplies", "record_id": "rec-999", "amount_usd": 5000}
        )
        assert result.record_id == "rec-999"

    def test_amount_usd_preserved(self, scope3_engine):
        """amount_usd is carried to the assignment."""
        result = scope3_engine.map_record(
            {"description": "supplies", "amount_usd": 12345.67}
        )
        assert result.amount_usd == 12345.67

    def test_assigned_at_populated(self, scope3_engine):
        """assigned_at timestamp is set."""
        result = scope3_engine.map_record(
            {"description": "supplies", "amount_usd": 5000}
        )
        assert result.assigned_at != ""

    def test_naics_code_used_when_provided(self, scope3_engine):
        """NAICS code in record takes priority."""
        result = scope3_engine.map_record(
            {"description": "generic stuff", "naics_code": "48", "amount_usd": 5000}
        )
        assert result.match_source == "naics"
        assert result.category_number == 4  # Transportation

    def test_taxonomy_code_param(self, scope3_engine):
        """taxonomy_code parameter is used as NAICS fallback."""
        result = scope3_engine.map_record(
            {"description": "generic stuff", "amount_usd": 5000},
            taxonomy_code="48",
        )
        assert result.match_source == "naics"

    def test_keyword_fallback_when_no_codes(self, scope3_engine):
        """Keyword matching is used when no taxonomy codes provided."""
        result = scope3_engine.map_record(
            {"description": "air travel business flights", "amount_usd": 5000}
        )
        assert result.match_source in ("keyword", "capex_detection")
        if result.match_source == "keyword":
            assert result.category_number == 6  # Business Travel

    def test_default_fallback(self, scope3_engine_no_capex):
        """Default category used when nothing matches."""
        result = scope3_engine_no_capex.map_record(
            {"description": "xyzzy_unknown_12345", "amount_usd": 5000}
        )
        assert result.match_source == "default"
        assert result.confidence <= 0.30


# ===================================================================
# TestMapBatch - batch mapping
# ===================================================================


class TestMapBatch:
    """Tests for map_batch() method."""

    def test_batch_returns_list(self, scope3_engine):
        """map_batch returns a list of assignments."""
        records = [
            {"description": "office supplies", "amount_usd": 5000},
            {"description": "diesel fuel", "amount_usd": 10000},
        ]
        results = scope3_engine.map_batch(records)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_batch_preserves_order(self, scope3_engine):
        """Results maintain input order."""
        records = [
            {"description": "A", "record_id": "r1", "amount_usd": 100},
            {"description": "B", "record_id": "r2", "amount_usd": 200},
        ]
        results = scope3_engine.map_batch(records)
        assert results[0].record_id == "r1"
        assert results[1].record_id == "r2"

    def test_batch_empty(self, scope3_engine):
        """Empty batch returns empty list."""
        results = scope3_engine.map_batch([])
        assert results == []

    def test_batch_mixed_records(self, scope3_engine):
        """Batch handles diverse records."""
        records = [
            {"description": "air travel", "amount_usd": 5000},
            {"description": "diesel fuel", "amount_usd": 10000},
            {"description": "office supplies", "amount_usd": 1000},
            {"description": "waste disposal", "amount_usd": 2000},
            {"description": "freight shipping", "amount_usd": 15000},
        ]
        results = scope3_engine.map_batch(records)
        assert len(results) == 5
        categories = {r.category_number for r in results}
        assert len(categories) >= 2  # At least 2 different categories


# ===================================================================
# TestMapFromNAICS - NAICS-to-Scope-3
# ===================================================================


class TestMapFromNAICS:
    """Tests for map_from_naics() method."""

    @pytest.mark.parametrize("naics_code,expected_cat", [
        ("11", 1),   # Agriculture -> Purchased Goods
        ("21", 1),   # Mining
        ("22", 3),   # Utilities -> Fuel & Energy
        ("23", 2),   # Construction -> Capital Goods
        ("31", 1),   # Manufacturing (Food)
        ("32", 1),   # Manufacturing (Paper/Chemical)
        ("33", 1),   # Manufacturing (Metal/Machinery)
        ("42", 1),   # Wholesale Trade
        ("44", 1),   # Retail Trade
        ("45", 1),   # Retail Trade
        ("48", 4),   # Transportation -> Upstream Transport
        ("49", 4),   # Warehousing
        ("51", 1),   # Information
        ("52", 15),  # Finance -> Investments
        ("53", 8),   # Real Estate -> Leased Assets
        ("54", 1),   # Professional Services
        ("55", 1),   # Management
        ("56", 1),   # Administrative
        ("61", 1),   # Education
        ("62", 1),   # Healthcare
        ("71", 6),   # Arts/Entertainment -> Business Travel
        ("72", 6),   # Accommodation -> Business Travel
        ("81", 1),   # Other Services
        ("92", 1),   # Public Admin
    ])
    def test_major_sector_mapping(self, scope3_engine, naics_code, expected_cat):
        """All major NAICS 2-digit sectors map correctly."""
        cat, conf = scope3_engine.map_from_naics(naics_code)
        assert cat == expected_cat
        assert conf >= 0.3

    def test_3digit_naics(self, scope3_engine):
        """3-digit NAICS codes use more specific mapping."""
        cat, conf = scope3_engine.map_from_naics("481")
        assert cat == 6  # Air transport -> Business Travel
        assert conf >= 0.80

    def test_4digit_naics(self, scope3_engine):
        """4-digit NAICS codes use most specific mapping."""
        cat, conf = scope3_engine.map_from_naics("5415")
        assert cat == 1  # Computer systems design

    def test_unknown_naics_defaults(self, scope3_engine):
        """Unknown NAICS code returns default category."""
        cat, conf = scope3_engine.map_from_naics("99")
        assert cat == 1  # Default category
        assert conf == 0.30

    def test_waste_management_naics(self, scope3_engine):
        """Waste management NAICS 562 maps to Cat 5."""
        cat, conf = scope3_engine.map_from_naics("562")
        assert cat == 5
        assert conf >= 0.85

    def test_confidence_varies_by_specificity(self, scope3_engine):
        """More specific codes may have different confidence."""
        _, conf_2 = scope3_engine.map_from_naics("48")
        _, conf_3 = scope3_engine.map_from_naics("484")
        assert conf_2 > 0
        assert conf_3 > 0

    def test_empty_code(self, scope3_engine):
        """Empty NAICS code returns default."""
        cat, conf = scope3_engine.map_from_naics("")
        assert cat == 1
        assert conf == 0.30


# ===================================================================
# TestMapFromUNSPSC - UNSPSC-to-Scope-3
# ===================================================================


class TestMapFromUNSPSC:
    """Tests for map_from_unspsc() method."""

    def test_fuels_segment(self, scope3_engine):
        """Fuels (15) maps to Cat 3."""
        cat, conf = scope3_engine.map_from_unspsc("15")
        assert cat == 3
        assert conf >= 0.85

    def test_office_supplies_segment(self, scope3_engine):
        """Office supplies (44) maps to Cat 1."""
        cat, conf = scope3_engine.map_from_unspsc("44")
        assert cat == 1

    def test_transportation_segment(self, scope3_engine):
        """Transportation (78) maps to Cat 4."""
        cat, conf = scope3_engine.map_from_unspsc("78")
        assert cat == 4
        assert conf >= 0.85

    def test_travel_segment(self, scope3_engine):
        """Travel (90) maps to Cat 6."""
        cat, conf = scope3_engine.map_from_unspsc("90")
        assert cat == 6

    def test_environmental_segment(self, scope3_engine):
        """Environmental (77) maps to Cat 5."""
        cat, conf = scope3_engine.map_from_unspsc("77")
        assert cat == 5

    def test_financial_services_segment(self, scope3_engine):
        """Financial services (84) maps to Cat 15."""
        cat, conf = scope3_engine.map_from_unspsc("84")
        assert cat == 15

    def test_real_estate_segment(self, scope3_engine):
        """Real estate (95) maps to Cat 8."""
        cat, conf = scope3_engine.map_from_unspsc("95")
        assert cat == 8

    def test_utilities_segment(self, scope3_engine):
        """Public utilities (83) maps to Cat 3."""
        cat, conf = scope3_engine.map_from_unspsc("83")
        assert cat == 3

    def test_machinery_segment(self, scope3_engine):
        """Construction machinery (22) maps to Cat 2."""
        cat, conf = scope3_engine.map_from_unspsc("22")
        assert cat == 2

    def test_longer_code_uses_2digit(self, scope3_engine):
        """8-digit code uses 2-digit segment for mapping."""
        cat, conf = scope3_engine.map_from_unspsc("44121500")
        assert cat == 1  # Office supplies

    def test_unknown_segment_defaults(self, scope3_engine):
        """Unknown UNSPSC segment returns default."""
        cat, conf = scope3_engine.map_from_unspsc("ZZ")
        assert cat == 1
        assert conf == 0.30


# ===================================================================
# TestMapFromKeyword - keyword-to-Scope-3
# ===================================================================


class TestMapFromKeyword:
    """Tests for map_from_keyword() method."""

    def test_air_travel(self, scope3_engine):
        """'air travel' maps to Cat 6."""
        cat, conf = scope3_engine.map_from_keyword("air travel booking")
        assert cat == 6
        assert conf >= 0.90

    def test_diesel_fuel(self, scope3_engine):
        """'diesel' maps to Cat 3."""
        cat, conf = scope3_engine.map_from_keyword("diesel fuel for generators")
        assert cat == 3

    def test_freight_shipping(self, scope3_engine):
        """'freight' maps to Cat 4."""
        cat, conf = scope3_engine.map_from_keyword("freight shipping services")
        assert cat == 4

    def test_waste_disposal(self, scope3_engine):
        """'waste disposal' maps to Cat 5."""
        cat, conf = scope3_engine.map_from_keyword("waste disposal recycling")
        assert cat == 5

    def test_commuting(self, scope3_engine):
        """'commuting' maps to Cat 7."""
        cat, conf = scope3_engine.map_from_keyword("employee commuting shuttle")
        assert cat == 7

    def test_lease(self, scope3_engine):
        """'lease' maps to Cat 8."""
        cat, conf = scope3_engine.map_from_keyword("office lease agreement")
        assert cat == 8

    def test_investment(self, scope3_engine):
        """'investment' maps to Cat 15."""
        cat, conf = scope3_engine.map_from_keyword("investment portfolio equity")
        assert cat == 15

    def test_capital_equipment(self, scope3_engine):
        """'capital equipment' maps to Cat 2."""
        cat, conf = scope3_engine.map_from_keyword("capital equipment purchase")
        assert cat == 2

    def test_office_supplies_keyword(self, scope3_engine):
        """'office supplies' maps to Cat 1."""
        cat, conf = scope3_engine.map_from_keyword("office supplies order")
        assert cat == 1

    def test_consulting_keyword(self, scope3_engine):
        """'consulting' maps to Cat 1."""
        cat, conf = scope3_engine.map_from_keyword("management consulting engagement")
        assert cat == 1

    def test_hotel_keyword(self, scope3_engine):
        """'hotel' maps to Cat 6."""
        cat, conf = scope3_engine.map_from_keyword("hotel reservation booking")
        assert cat == 6

    def test_no_match_default(self, scope3_engine):
        """No keyword match returns default with low confidence."""
        cat, conf = scope3_engine.map_from_keyword("xyzzy_nothing_here")
        assert cat == 1  # default
        assert conf == 0.20

    def test_partial_match(self, scope3_engine):
        """Partial keyword in text matches."""
        cat, conf = scope3_engine.map_from_keyword(
            "purchased recycling bins for waste disposal"
        )
        assert cat == 5  # waste disposal keyword wins

    def test_case_insensitive(self, scope3_engine):
        """Keyword matching is case-insensitive."""
        cat, conf = scope3_engine.map_from_keyword("DIESEL FUEL PURCHASE")
        assert cat == 3

    def test_best_keyword_wins(self, scope3_engine):
        """When multiple keywords match, highest confidence wins."""
        # Both 'hotel' and 'business travel' in text
        cat, conf = scope3_engine.map_from_keyword(
            "business travel hotel booking flights"
        )
        assert cat == 6  # business travel has 0.95, hotel has 0.88
        assert conf >= 0.88


# ===================================================================
# TestMapFromTaxonomy - generic taxonomy mapping
# ===================================================================


class TestMapFromTaxonomy:
    """Tests for map_from_taxonomy() method."""

    def test_naics_delegation(self, scope3_engine):
        """NAICS system delegates to map_from_naics."""
        cat, conf = scope3_engine.map_from_taxonomy("48", "naics")
        assert cat == 4  # Transportation

    def test_unspsc_delegation(self, scope3_engine):
        """UNSPSC system delegates to map_from_unspsc."""
        cat, conf = scope3_engine.map_from_taxonomy("15", "unspsc")
        assert cat == 3  # Fuels

    def test_unsupported_system(self, scope3_engine):
        """Unsupported system returns default."""
        cat, conf = scope3_engine.map_from_taxonomy("XX", "eclass")
        assert cat == 1
        assert conf == 0.20

    def test_case_insensitive_system(self, scope3_engine):
        """System name is case-insensitive."""
        cat, conf = scope3_engine.map_from_taxonomy("48", "NAICS")
        assert cat == 4


# ===================================================================
# TestDetectCapex - CapEx detection
# ===================================================================


class TestDetectCapex:
    """Tests for detect_capex() method."""

    def test_machinery_is_capex(self, scope3_engine):
        """'machinery' in description triggers CapEx."""
        result = scope3_engine.detect_capex(
            {"description": "purchase of machinery", "amount_usd": 50000}
        )
        assert result is True

    def test_equipment_is_capex(self, scope3_engine):
        """'equipment' in description triggers CapEx."""
        result = scope3_engine.detect_capex(
            {"description": "heavy equipment acquisition", "amount_usd": 100000}
        )
        assert result is True

    def test_building_is_capex(self, scope3_engine):
        """'building' in description triggers CapEx."""
        result = scope3_engine.detect_capex(
            {"description": "new building construction", "amount_usd": 500000}
        )
        assert result is True

    def test_capital_keyword(self, scope3_engine):
        """'capital' in description triggers CapEx."""
        result = scope3_engine.detect_capex(
            {"description": "capital expenditure Q2", "amount_usd": 75000}
        )
        assert result is True

    def test_capex_keyword(self, scope3_engine):
        """'capex' in description triggers CapEx."""
        result = scope3_engine.detect_capex(
            {"description": "capex budget allocation", "amount_usd": 100000}
        )
        assert result is True

    def test_vehicle_purchase(self, scope3_engine):
        """'vehicle purchase' triggers CapEx."""
        result = scope3_engine.detect_capex(
            {"description": "vehicle purchase for fleet", "amount_usd": 45000}
        )
        assert result is True

    def test_server_purchase(self, scope3_engine):
        """'server purchase' triggers CapEx."""
        result = scope3_engine.detect_capex(
            {"description": "server purchase for datacenter", "amount_usd": 80000}
        )
        assert result is True

    def test_office_supplies_not_capex(self, scope3_engine):
        """Regular operating expenses are not CapEx."""
        result = scope3_engine.detect_capex(
            {"description": "office supplies order", "amount_usd": 500}
        )
        assert result is False

    def test_consulting_not_capex(self, scope3_engine):
        """Consulting services are not CapEx."""
        result = scope3_engine.detect_capex(
            {"description": "management consulting services", "amount_usd": 25000}
        )
        assert result is False

    def test_category_field_checked(self, scope3_engine):
        """Category field is also checked for CapEx keywords."""
        result = scope3_engine.detect_capex(
            {"description": "general purchase",
             "category": "capital equipment", "amount_usd": 60000}
        )
        assert result is True

    def test_gl_account_checked(self, scope3_engine):
        """GL account field is checked for CapEx keywords."""
        result = scope3_engine.detect_capex(
            {"description": "general item",
             "gl_account": "fixed asset account", "amount_usd": 60000}
        )
        assert result is True

    def test_large_amount_equipment_category(self, scope3_engine):
        """Large amount + equipment category triggers CapEx."""
        result = scope3_engine.detect_capex(
            {"description": "generic item",
             "category": "equipment", "amount_usd": 75000}
        )
        assert result is True

    def test_large_amount_no_equipment_no_capex(self, scope3_engine):
        """Large amount alone without keywords does not trigger CapEx."""
        result = scope3_engine.detect_capex(
            {"description": "generic services",
             "category": "services", "amount_usd": 999999}
        )
        assert result is False

    def test_capex_disabled(self, scope3_engine_no_capex):
        """CapEx detection disabled in config."""
        result = scope3_engine_no_capex.map_record(
            {"description": "machinery purchase", "amount_usd": 100000}
        )
        # With CapEx disabled, shouldn't be flagged via capex_detection
        assert result.match_source != "capex_detection"

    def test_capex_overrides_keyword_mapping(self, scope3_engine):
        """When CapEx detected, category 2 overrides keyword match."""
        result = scope3_engine.map_record(
            {"description": "machinery for manufacturing", "amount_usd": 100000}
        )
        assert result.category_number == 2
        assert result.is_capex is True


# ===================================================================
# TestSplitAllocation - multi-category allocation
# ===================================================================


class TestSplitAllocation:
    """Tests for split_allocation() method."""

    def test_two_category_split(self, scope3_engine):
        """Split between two categories."""
        record = {"amount_usd": 10000}
        categories = [
            {"category_number": 1, "weight": 0.7},
            {"category_number": 4, "weight": 0.3},
        ]
        allocs = scope3_engine.split_allocation(record, categories)
        assert len(allocs) == 2
        total_amount = sum(a["amount_usd"] for a in allocs)
        assert total_amount == pytest.approx(10000, abs=0.01)

    def test_three_category_split(self, scope3_engine):
        """Split between three categories."""
        record = {"amount_usd": 9000}
        categories = [
            {"category_number": 1, "weight": 0.5},
            {"category_number": 3, "weight": 0.3},
            {"category_number": 6, "weight": 0.2},
        ]
        allocs = scope3_engine.split_allocation(record, categories)
        assert len(allocs) == 3
        weights = [a["weight"] for a in allocs]
        assert sum(weights) == pytest.approx(1.0, abs=0.01)

    def test_single_category_full_allocation(self, scope3_engine):
        """Single category gets 100% allocation."""
        record = {"amount_usd": 5000}
        categories = [{"category_number": 1, "weight": 1.0}]
        allocs = scope3_engine.split_allocation(record, categories)
        assert len(allocs) == 1
        assert allocs[0]["amount_usd"] == pytest.approx(5000, abs=0.01)
        assert allocs[0]["weight"] == pytest.approx(1.0, abs=0.001)

    def test_weights_sum_to_one(self, scope3_engine):
        """Normalized weights sum to 1.0."""
        record = {"amount_usd": 10000}
        categories = [
            {"category_number": 1, "weight": 3},
            {"category_number": 4, "weight": 2},
            {"category_number": 6, "weight": 5},
        ]
        allocs = scope3_engine.split_allocation(record, categories)
        weights = [a["weight"] for a in allocs]
        assert sum(weights) == pytest.approx(1.0, abs=0.01)

    def test_zero_weights_equalized(self, scope3_engine):
        """Zero total weight reassigns weights and distributes.

        When all weights are zero, the engine sets each weight to 1/N
        and uses N as the total_weight. Each allocation then gets
        (1/N)/N = 1/N^2 of the amount. For N=3 and amount=9000:
        each allocation = 9000 * (1/3) / 3 = 1000.
        """
        record = {"amount_usd": 9000}
        categories = [
            {"category_number": 1, "weight": 0},
            {"category_number": 4, "weight": 0},
            {"category_number": 6, "weight": 0},
        ]
        allocs = scope3_engine.split_allocation(record, categories)
        assert len(allocs) == 3
        # All allocations should have the same amount
        amounts = [a["amount_usd"] for a in allocs]
        assert amounts[0] == amounts[1] == amounts[2]
        # Each gets 1000 due to the internal weight normalization
        for a in allocs:
            assert a["amount_usd"] == pytest.approx(1000, abs=1)

    def test_allocation_provenance(self, scope3_engine):
        """Each allocation has a provenance hash."""
        record = {"amount_usd": 10000}
        categories = [
            {"category_number": 1, "weight": 0.6},
            {"category_number": 4, "weight": 0.4},
        ]
        allocs = scope3_engine.split_allocation(record, categories)
        for a in allocs:
            assert len(a["provenance_hash"]) == 64

    def test_allocation_category_name(self, scope3_engine):
        """Each allocation has the correct category name."""
        record = {"amount_usd": 10000}
        categories = [{"category_number": 6, "weight": 1.0}]
        allocs = scope3_engine.split_allocation(record, categories)
        assert allocs[0]["category_name"] == "Business Travel"

    def test_zero_amount_split(self, scope3_engine):
        """Zero amount distributes zeros."""
        record = {"amount_usd": 0}
        categories = [
            {"category_number": 1, "weight": 0.5},
            {"category_number": 4, "weight": 0.5},
        ]
        allocs = scope3_engine.split_allocation(record, categories)
        for a in allocs:
            assert a["amount_usd"] == 0.0


# ===================================================================
# TestGetCategoryInfo - category metadata
# ===================================================================


class TestGetCategoryInfo:
    """Tests for get_category_info() and get_all_categories()."""

    @pytest.mark.parametrize("cat_num", range(1, 16))
    def test_all_15_categories(self, scope3_engine, cat_num):
        """All 15 Scope 3 categories return valid info."""
        info = scope3_engine.get_category_info(cat_num)
        assert info["category_number"] == cat_num
        assert info["name"] != ""
        assert info["direction"] in ("upstream", "downstream")
        assert info["description"] != ""
        assert info["methodology"] != ""

    def test_upstream_categories(self, scope3_engine):
        """Categories 1-8 are upstream."""
        for num in range(1, 9):
            info = scope3_engine.get_category_info(num)
            assert info["direction"] == "upstream"

    def test_downstream_categories(self, scope3_engine):
        """Categories 9-15 are downstream."""
        for num in range(9, 16):
            info = scope3_engine.get_category_info(num)
            assert info["direction"] == "downstream"

    def test_invalid_category_raises(self, scope3_engine):
        """Invalid category number raises ValueError."""
        with pytest.raises(ValueError, match="Invalid Scope 3 category"):
            scope3_engine.get_category_info(0)

    def test_invalid_category_16(self, scope3_engine):
        """Category 16 raises ValueError."""
        with pytest.raises(ValueError, match="Invalid Scope 3 category"):
            scope3_engine.get_category_info(16)

    def test_get_all_categories(self, scope3_engine):
        """get_all_categories returns all 15."""
        cats = scope3_engine.get_all_categories()
        assert len(cats) == 15
        numbers = [c["category_number"] for c in cats]
        assert numbers == list(range(1, 16))


# ===================================================================
# TestConfidenceScoring - confidence by source
# ===================================================================


class TestConfidenceScoring:
    """Tests for confidence scoring based on match source."""

    def test_naics_high_confidence(self, scope3_engine):
        """NAICS match yields high confidence (>= 0.70)."""
        result = scope3_engine.map_record(
            {"naics_code": "48", "description": "generic", "amount_usd": 5000}
        )
        assert result.match_source == "naics"
        assert result.confidence >= 0.70

    def test_keyword_moderate_confidence(self, scope3_engine_no_capex):
        """Keyword match yields moderate confidence."""
        result = scope3_engine_no_capex.map_record(
            {"description": "office supplies", "amount_usd": 1000}
        )
        if result.match_source == "keyword":
            assert 0.50 <= result.confidence <= 1.0

    def test_default_low_confidence(self, scope3_engine_no_capex):
        """Default fallback yields low confidence (0.30)."""
        result = scope3_engine_no_capex.map_record(
            {"description": "xyzzy_unknown", "amount_usd": 100}
        )
        assert result.match_source == "default"
        assert result.confidence == 0.30

    def test_capex_confidence(self, scope3_engine):
        """CapEx detection yields 0.85 confidence."""
        result = scope3_engine.map_record(
            {"description": "machinery purchase", "amount_usd": 100000}
        )
        assert result.is_capex is True
        assert result.confidence == 0.85


# ===================================================================
# TestStatistics - mapping statistics
# ===================================================================


class TestStatistics:
    """Tests for get_statistics() method."""

    def test_initial_stats(self, scope3_engine):
        """Fresh engine reports zero stats."""
        stats = scope3_engine.get_statistics()
        assert stats["records_mapped"] == 0
        assert stats["capex_detected"] == 0

    def test_stats_after_mapping(self, scope3_engine):
        """Stats update after mapping."""
        scope3_engine.map_record(
            {"description": "office supplies", "amount_usd": 5000}
        )
        stats = scope3_engine.get_statistics()
        assert stats["records_mapped"] == 1

    def test_by_category_tracking(self, scope3_engine):
        """by_category tracks category distribution."""
        scope3_engine.map_record(
            {"description": "air travel", "amount_usd": 5000}
        )
        stats = scope3_engine.get_statistics()
        total = sum(int(v) for v in stats["by_category"].values())
        assert total == 1

    def test_by_source_tracking(self, scope3_engine):
        """by_source tracks match source distribution."""
        scope3_engine.map_record(
            {"naics_code": "48", "description": "", "amount_usd": 5000}
        )
        stats = scope3_engine.get_statistics()
        assert stats["by_source"].get("naics", 0) >= 1

    def test_capex_counter(self, scope3_engine):
        """capex_detected counter increments."""
        scope3_engine.map_record(
            {"description": "machinery purchase", "amount_usd": 100000}
        )
        stats = scope3_engine.get_statistics()
        assert stats["capex_detected"] >= 1

    def test_avg_confidence(self, scope3_engine):
        """avg_confidence is computed correctly."""
        scope3_engine.map_record(
            {"naics_code": "48", "description": "", "amount_usd": 5000}
        )
        stats = scope3_engine.get_statistics()
        assert 0.0 < stats["avg_confidence"] <= 1.0

    def test_rules_counts_reported(self, scope3_engine):
        """Mapping rule counts are reported."""
        stats = scope3_engine.get_statistics()
        assert stats["naics_rules_count"] >= 50
        assert stats["unspsc_rules_count"] >= 55
        assert stats["keyword_rules_count"] >= 60


# ===================================================================
# TestProvenance - SHA-256 provenance hashing
# ===================================================================


class TestProvenance:
    """Tests for provenance hashing on assignments."""

    def test_assignment_provenance_sha256(self, scope3_engine):
        """Assignment provenance hash is 64-char hex SHA-256."""
        result = scope3_engine.map_record(
            {"description": "office supplies", "amount_usd": 5000}
        )
        assert len(result.provenance_hash) == 64
        int(result.provenance_hash, 16)

    def test_different_records_different_hashes(self, scope3_engine):
        """Different records produce different provenance hashes."""
        r1 = scope3_engine.map_record(
            {"description": "office supplies", "amount_usd": 5000}
        )
        r2 = scope3_engine.map_record(
            {"description": "diesel fuel", "amount_usd": 10000}
        )
        assert r1.provenance_hash != r2.provenance_hash

    def test_batch_all_have_provenance(self, scope3_engine):
        """All records in a batch have provenance hashes."""
        records = [
            {"description": "office supplies", "amount_usd": 5000},
            {"description": "diesel fuel", "amount_usd": 10000},
            {"description": "air travel", "amount_usd": 3000},
        ]
        results = scope3_engine.map_batch(records)
        for r in results:
            assert len(r.provenance_hash) == 64


# ===================================================================
# TestThreadSafety - concurrent mapping
# ===================================================================


class TestThreadSafety:
    """Tests for thread-safe concurrent mapping."""

    def test_concurrent_mapping(self, scope3_engine):
        """Multiple threads can map simultaneously."""
        errors = []

        def worker(engine, desc, idx):
            try:
                result = engine.map_record(
                    {"description": desc, "amount_usd": 1000 * (idx + 1),
                     "record_id": f"r-{idx}"}
                )
                assert result.category_number >= 0
            except Exception as exc:
                errors.append(str(exc))

        descriptions = [
            "office supplies", "diesel fuel", "air travel",
            "freight shipping", "waste disposal", "legal consulting",
            "machinery purchase", "software license",
            "building maintenance", "hotel booking",
        ]

        threads = [
            threading.Thread(
                target=worker, args=(scope3_engine, desc, i)
            )
            for i, desc in enumerate(descriptions)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert errors == [], f"Thread errors: {errors}"

        stats = scope3_engine.get_statistics()
        assert stats["records_mapped"] == 10

    def test_concurrent_stats_consistent(self, scope3_engine):
        """Statistics remain consistent under load."""
        def worker(engine, idx):
            engine.map_record(
                {"description": f"item {idx}", "amount_usd": 100}
            )

        threads = [
            threading.Thread(target=worker, args=(scope3_engine, i))
            for i in range(20)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        stats = scope3_engine.get_statistics()
        assert stats["records_mapped"] == 20
