# -*- coding: utf-8 -*-
"""
Test suite for CategoryDatabaseEngine - AGENT-MRV-029 Engine 1.

Tests the CategoryDatabaseEngine for the Scope 3 Category Mapper Agent
(GL-MRV-X-040) including NAICS 2022 lookups (2-digit sectors, 3-digit
subsector overrides), ISIC Rev 4 lookups (section and division levels),
GL account range lookups, keyword dictionary lookups, cross-reference
concordance, category info retrieval, and provenance hashing.

Coverage:
- NAICS 2-digit sector lookups (all 20 sectors mapped)
- NAICS 3-digit subsector overrides (precedence over 2-digit)
- NAICS invalid/empty code handling
- NAICS secondary categories and confidence values
- ISIC section lookups (all 21 sections A-U)
- ISIC division overrides (precedence over section)
- ISIC-NAICS cross-reference concordance
- GL account range lookups (all 20 ranges)
- GL boundary conditions (start, end, mid-range, out-of-range)
- Keyword lookups (all 15 categories via keywords)
- Keyword case insensitivity and partial matching
- Keyword batch processing
- Category info for all 15 categories
- Upstream/downstream direction validation
- Category info required fields
- Mapping version and database summary
- Singleton pattern and thread safety
- Provenance hash determinism

Total: ~120 tests

Author: GL-TestEngineer
Date: March 2026
"""

import threading
from decimal import Decimal
from typing import Any, Dict, List, Optional

import pytest

from greenlang.agents.mrv.scope3_category_mapper.category_database import (
    AGENT_COMPONENT,
    AGENT_ID,
    MAPPING_VERSION,
    VERSION,
    CategoryDatabaseEngine,
    CategoryInfo,
    GLLookupResult,
    ISICLookupResult,
    KeywordLookupResult,
    NAICSLookupResult,
    Scope3Category,
    ValueChainDirection,
    get_category_database_engine,
    reset_category_database_engine,
)


# ==============================================================================
# LOCAL FIXTURES
# ==============================================================================


@pytest.fixture(autouse=True)
def _reset_category_db_singleton():
    """Reset the CategoryDatabaseEngine singleton before and after each test."""
    reset_category_database_engine()
    yield
    reset_category_database_engine()


@pytest.fixture
def engine() -> CategoryDatabaseEngine:
    """Create a fresh CategoryDatabaseEngine instance."""
    return CategoryDatabaseEngine()


# ==============================================================================
# SINGLETON PATTERN TESTS
# ==============================================================================


class TestSingletonPattern:
    """Test CategoryDatabaseEngine singleton implementation."""

    def test_singleton_same_instance(self, engine):
        """Two calls to CategoryDatabaseEngine() return the same object."""
        engine2 = CategoryDatabaseEngine()
        assert engine is engine2

    def test_singleton_via_accessor(self):
        """get_category_database_engine returns singleton instance."""
        e1 = get_category_database_engine()
        e2 = get_category_database_engine()
        assert e1 is e2

    def test_singleton_across_threads(self):
        """Singleton is consistent across multiple threads."""
        instances: List[CategoryDatabaseEngine] = []

        def create():
            instances.append(CategoryDatabaseEngine())

        threads = [threading.Thread(target=create) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(instances) == 10
        for inst in instances:
            assert inst is instances[0]

    def test_reset_creates_new_instance(self, engine):
        """After reset, a new instance is created."""
        old_id = id(engine)
        reset_category_database_engine()
        engine2 = CategoryDatabaseEngine()
        assert id(engine2) != old_id


# ==============================================================================
# NAICS LOOKUP TESTS (~40)
# ==============================================================================


class TestNAICSLookups:
    """Test NAICS code to Scope 3 category lookups."""

    # --- 2-digit sector-level tests ---

    def test_lookup_naics_2digit_11_agriculture(self, engine):
        """NAICS 11 (Agriculture) maps to Cat 1."""
        result = engine.lookup_naics("11")
        assert result.primary_category == Scope3Category.CAT_1
        assert result.description == "Agriculture, Forestry, Fishing and Hunting"

    def test_lookup_naics_2digit_22_utilities(self, engine):
        """NAICS 22 (Utilities) maps to Cat 3."""
        result = engine.lookup_naics("22")
        assert result.primary_category == Scope3Category.CAT_3

    def test_lookup_naics_2digit_23_construction(self, engine):
        """NAICS 23 (Construction) maps to Cat 2."""
        result = engine.lookup_naics("23")
        assert result.primary_category == Scope3Category.CAT_2

    def test_lookup_naics_2digit_48_transportation(self, engine):
        """NAICS 48 (Transportation) maps to Cat 4."""
        result = engine.lookup_naics("48")
        assert result.primary_category == Scope3Category.CAT_4

    def test_lookup_naics_2digit_52_finance(self, engine):
        """NAICS 52 (Finance) maps to Cat 15."""
        result = engine.lookup_naics("52")
        assert result.primary_category == Scope3Category.CAT_15

    def test_lookup_naics_2digit_53_real_estate(self, engine):
        """NAICS 53 (Real Estate) maps to Cat 8."""
        result = engine.lookup_naics("53")
        assert result.primary_category == Scope3Category.CAT_8

    def test_lookup_naics_2digit_72_accommodation(self, engine):
        """NAICS 72 (Accommodation) maps to Cat 6."""
        result = engine.lookup_naics("72")
        assert result.primary_category == Scope3Category.CAT_6

    @pytest.mark.parametrize("naics_code,expected_cat", [
        ("11", Scope3Category.CAT_1),   # Agriculture
        ("21", Scope3Category.CAT_1),   # Mining
        ("22", Scope3Category.CAT_3),   # Utilities
        ("23", Scope3Category.CAT_2),   # Construction
        ("31", Scope3Category.CAT_1),   # Mfg (Food/Beverage/Textile)
        ("32", Scope3Category.CAT_1),   # Mfg (Wood/Paper/Chemical)
        ("33", Scope3Category.CAT_1),   # Mfg (Metal/Machinery/Electronics)
        ("42", Scope3Category.CAT_1),   # Wholesale Trade
        ("44", Scope3Category.CAT_1),   # Retail Trade
        ("45", Scope3Category.CAT_1),   # Retail Trade
        ("48", Scope3Category.CAT_4),   # Transportation
        ("49", Scope3Category.CAT_4),   # Postal/Couriers
        ("51", Scope3Category.CAT_1),   # Information
        ("52", Scope3Category.CAT_15),  # Finance
        ("53", Scope3Category.CAT_8),   # Real Estate
        ("54", Scope3Category.CAT_1),   # Professional Services
        ("55", Scope3Category.CAT_1),   # Mgmt of Companies
        ("56", Scope3Category.CAT_1),   # Administrative Services
        ("61", Scope3Category.CAT_1),   # Education
        ("62", Scope3Category.CAT_1),   # Health Care
        ("71", Scope3Category.CAT_1),   # Arts/Entertainment
        ("72", Scope3Category.CAT_6),   # Accommodation
        ("81", Scope3Category.CAT_1),   # Other Services
        ("92", Scope3Category.CAT_1),   # Public Admin
    ])
    def test_lookup_naics_all_2digit_sectors(self, engine, naics_code, expected_cat):
        """All 2-digit NAICS sectors return the expected primary category."""
        result = engine.lookup_naics(naics_code)
        assert result.primary_category == expected_cat

    def test_lookup_naics_all_2digit_sectors_mapped(self, engine):
        """All 20 NAICS 2-digit sectors (11-92) have valid mappings."""
        sector_codes = [
            "11", "21", "22", "23", "31", "32", "33", "42", "44", "45",
            "48", "49", "51", "52", "53", "54", "55", "56", "61", "62",
            "71", "72", "81", "92",
        ]
        for code in sector_codes:
            result = engine.lookup_naics(code)
            assert result is not None
            assert 1 <= result.primary_category.value <= 15

    # --- 3-digit subsector override tests ---

    def test_lookup_naics_3digit_333_machinery(self, engine):
        """NAICS 333 (Machinery Mfg) maps to Cat 2 (capital goods)."""
        result = engine.lookup_naics("333")
        assert result.primary_category == Scope3Category.CAT_2
        assert result.matched_code == "333"

    def test_lookup_naics_3digit_334_electronics(self, engine):
        """NAICS 334 (Computer/Electronic) maps to Cat 2."""
        result = engine.lookup_naics("334")
        assert result.primary_category == Scope3Category.CAT_2

    def test_lookup_naics_3digit_481_air_transport(self, engine):
        """NAICS 481 (Air Transportation) maps to Cat 6."""
        result = engine.lookup_naics("481")
        assert result.primary_category == Scope3Category.CAT_6

    def test_lookup_naics_3digit_484_trucking(self, engine):
        """NAICS 484 (Truck Transportation) maps to Cat 4."""
        result = engine.lookup_naics("484")
        assert result.primary_category == Scope3Category.CAT_4

    def test_lookup_naics_3digit_562_waste(self, engine):
        """NAICS 562 (Waste Management) maps to Cat 5."""
        result = engine.lookup_naics("562")
        assert result.primary_category == Scope3Category.CAT_5

    def test_lookup_naics_3digit_overrides_2digit(self, engine):
        """3-digit subsector override takes precedence over 2-digit sector.

        NAICS 48 (Transportation) -> Cat 4, but 481 (Air) -> Cat 6.
        """
        sector_result = engine.lookup_naics("48")
        subsector_result = engine.lookup_naics("481")

        assert sector_result.primary_category == Scope3Category.CAT_4
        assert subsector_result.primary_category == Scope3Category.CAT_6
        assert subsector_result.matched_code == "481"

    def test_lookup_naics_6digit_uses_3digit_prefix(self, engine):
        """A 6-digit NAICS code resolves via its 3-digit prefix override."""
        result = engine.lookup_naics("484110")
        assert result.primary_category == Scope3Category.CAT_4
        assert result.naics_code == "484110"
        assert result.matched_code == "484"

    def test_lookup_naics_4digit_falls_to_3digit(self, engine):
        """A 4-digit code uses its 3-digit prefix if available."""
        result = engine.lookup_naics("4841")
        assert result.primary_category == Scope3Category.CAT_4
        assert result.matched_code == "484"

    # --- Result field validation ---

    def test_lookup_naics_returns_secondary_categories(self, engine):
        """NAICS lookup returns secondary categories list."""
        result = engine.lookup_naics("481")
        assert isinstance(result.secondary_categories, list)
        assert Scope3Category.CAT_4 in result.secondary_categories

    def test_lookup_naics_returns_confidence(self, engine):
        """NAICS lookup returns a confidence value in [0, 1]."""
        result = engine.lookup_naics("484")
        assert isinstance(result.confidence, Decimal)
        assert Decimal("0") < result.confidence <= Decimal("1")

    def test_lookup_naics_returns_provenance_hash(self, engine):
        """NAICS lookup returns a 64-character SHA-256 provenance hash."""
        result = engine.lookup_naics("11")
        assert isinstance(result.provenance_hash, str)
        assert len(result.provenance_hash) == 64

    def test_lookup_naics_provenance_deterministic(self, engine):
        """Same NAICS input produces identical provenance hash."""
        r1 = engine.lookup_naics("484")
        r2 = engine.lookup_naics("484")
        assert r1.provenance_hash == r2.provenance_hash

    def test_lookup_naics_result_type(self, engine):
        """NAICS lookup returns NAICSLookupResult model."""
        result = engine.lookup_naics("11")
        assert isinstance(result, NAICSLookupResult)

    def test_lookup_naics_result_frozen(self, engine):
        """NAICSLookupResult is frozen (immutable)."""
        result = engine.lookup_naics("11")
        with pytest.raises(Exception):
            result.primary_category = Scope3Category.CAT_2

    # --- Error handling ---

    def test_lookup_naics_invalid_code(self, engine):
        """Non-existent NAICS code raises ValueError."""
        with pytest.raises(ValueError, match="No NAICS mapping found"):
            engine.lookup_naics("99")

    def test_lookup_naics_empty_string(self, engine):
        """Empty string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid NAICS code"):
            engine.lookup_naics("")

    def test_lookup_naics_non_numeric(self, engine):
        """Non-numeric string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid NAICS code"):
            engine.lookup_naics("ABC")

    def test_lookup_naics_whitespace_only(self, engine):
        """Whitespace-only string raises ValueError."""
        with pytest.raises(ValueError):
            engine.lookup_naics("   ")

    def test_lookup_naics_strips_whitespace(self, engine):
        """Leading/trailing whitespace is stripped before lookup."""
        result = engine.lookup_naics("  484  ")
        assert result.primary_category == Scope3Category.CAT_4


# ==============================================================================
# ISIC LOOKUP TESTS (~20)
# ==============================================================================


class TestISICLookups:
    """Test ISIC Rev 4 code to Scope 3 category lookups."""

    def test_lookup_isic_section_A(self, engine):
        """ISIC section A (Agriculture) maps to Cat 1."""
        result = engine.lookup_isic("A")
        assert result.primary_category == Scope3Category.CAT_1

    def test_lookup_isic_section_C(self, engine):
        """ISIC section C (Manufacturing) maps to Cat 1."""
        result = engine.lookup_isic("C")
        assert result.primary_category == Scope3Category.CAT_1

    def test_lookup_isic_section_D(self, engine):
        """ISIC section D (Electricity/Gas) maps to Cat 3."""
        result = engine.lookup_isic("D")
        assert result.primary_category == Scope3Category.CAT_3

    def test_lookup_isic_section_E(self, engine):
        """ISIC section E (Water/Waste) maps to Cat 5."""
        result = engine.lookup_isic("E")
        assert result.primary_category == Scope3Category.CAT_5

    def test_lookup_isic_section_F(self, engine):
        """ISIC section F (Construction) maps to Cat 2."""
        result = engine.lookup_isic("F")
        assert result.primary_category == Scope3Category.CAT_2

    def test_lookup_isic_section_H(self, engine):
        """ISIC section H (Transport) maps to Cat 4."""
        result = engine.lookup_isic("H")
        assert result.primary_category == Scope3Category.CAT_4

    def test_lookup_isic_section_K(self, engine):
        """ISIC section K (Finance) maps to Cat 15."""
        result = engine.lookup_isic("K")
        assert result.primary_category == Scope3Category.CAT_15

    @pytest.mark.parametrize("isic_section,expected_cat", [
        ("A", Scope3Category.CAT_1),   # Agriculture
        ("B", Scope3Category.CAT_1),   # Mining
        ("C", Scope3Category.CAT_1),   # Manufacturing
        ("D", Scope3Category.CAT_3),   # Electricity/Gas
        ("E", Scope3Category.CAT_5),   # Water/Waste
        ("F", Scope3Category.CAT_2),   # Construction
        ("G", Scope3Category.CAT_1),   # Wholesale/Retail
        ("H", Scope3Category.CAT_4),   # Transport
        ("I", Scope3Category.CAT_6),   # Accommodation
        ("J", Scope3Category.CAT_1),   # Information
        ("K", Scope3Category.CAT_15),  # Finance
        ("L", Scope3Category.CAT_8),   # Real Estate
        ("M", Scope3Category.CAT_1),   # Professional Services
        ("N", Scope3Category.CAT_1),   # Administrative
        ("O", Scope3Category.CAT_1),   # Public Admin
        ("P", Scope3Category.CAT_1),   # Education
        ("Q", Scope3Category.CAT_1),   # Health
        ("R", Scope3Category.CAT_1),   # Arts/Entertainment
        ("S", Scope3Category.CAT_1),   # Other Services
        ("T", Scope3Category.CAT_1),   # Households
        ("U", Scope3Category.CAT_1),   # Extraterritorial
    ])
    def test_lookup_isic_all_21_sections_mapped(self, engine, isic_section, expected_cat):
        """All 21 ISIC sections (A-U) return the expected primary category."""
        result = engine.lookup_isic(isic_section)
        assert result.primary_category == expected_cat

    def test_lookup_isic_division_override(self, engine):
        """ISIC 2-digit division override takes precedence over section.

        Section B (Mining) -> Cat 1, but division 06 (Oil/Gas) -> Cat 3.
        """
        result = engine.lookup_isic("06")
        assert result.primary_category == Scope3Category.CAT_3
        assert result.matched_code == "06"

    def test_lookup_isic_case_insensitive(self, engine):
        """ISIC section lookup is case-insensitive."""
        upper = engine.lookup_isic("A")
        lower = engine.lookup_isic("a")
        assert upper.primary_category == lower.primary_category

    def test_lookup_isic_naics_cross_reference(self, engine):
        """NAICS sector 48 cross-references to ISIC section H."""
        isic_section = engine.get_naics_to_isic("48")
        assert isic_section == "H"

        naics_sectors = engine.get_isic_to_naics("H")
        assert "48" in naics_sectors

    def test_lookup_isic_result_type(self, engine):
        """ISIC lookup returns ISICLookupResult model."""
        result = engine.lookup_isic("A")
        assert isinstance(result, ISICLookupResult)

    def test_lookup_isic_returns_provenance_hash(self, engine):
        """ISIC lookup returns a 64-character SHA-256 hash."""
        result = engine.lookup_isic("K")
        assert len(result.provenance_hash) == 64

    def test_lookup_isic_invalid(self, engine):
        """Invalid ISIC code raises ValueError."""
        with pytest.raises(ValueError, match="No ISIC mapping found"):
            engine.lookup_isic("Z")

    def test_lookup_isic_empty(self, engine):
        """Empty ISIC code raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            engine.lookup_isic("")

    def test_lookup_isic_returns_secondary(self, engine):
        """ISIC lookup returns secondary categories."""
        result = engine.lookup_isic("H")
        assert isinstance(result.secondary_categories, list)
        assert Scope3Category.CAT_6 in result.secondary_categories

    def test_lookup_isic_returns_confidence(self, engine):
        """ISIC lookup returns confidence in (0, 1]."""
        result = engine.lookup_isic("D")
        assert Decimal("0") < result.confidence <= Decimal("1")

    def test_get_naics_to_isic_invalid(self, engine):
        """Invalid NAICS code for cross-reference returns None."""
        result = engine.get_naics_to_isic("ABC")
        assert result is None

    def test_get_isic_to_naics_manufacturing(self, engine):
        """ISIC C (Manufacturing) maps to NAICS 31, 32, 33."""
        naics_codes = engine.get_isic_to_naics("C")
        assert "31" in naics_codes
        assert "32" in naics_codes
        assert "33" in naics_codes

    def test_get_isic_to_naics_no_match(self, engine):
        """ISIC section with no NAICS correspondence returns empty list."""
        naics_codes = engine.get_isic_to_naics("T")
        assert naics_codes == []


# ==============================================================================
# GL ACCOUNT LOOKUP TESTS (~20)
# ==============================================================================


class TestGLAccountLookups:
    """Test GL account code to Scope 3 category lookups."""

    def test_lookup_gl_5000_materials(self, engine):
        """GL 5000 (COGS - Materials) maps to Cat 1."""
        result = engine.lookup_gl_account("5000")
        assert result.primary_category == Scope3Category.CAT_1

    def test_lookup_gl_5300_freight(self, engine):
        """GL 5300 (Freight In) maps to Cat 4."""
        result = engine.lookup_gl_account("5300")
        assert result.primary_category == Scope3Category.CAT_4

    def test_lookup_gl_6400_travel(self, engine):
        """GL 6400 (Travel & Entertainment) maps to Cat 6."""
        result = engine.lookup_gl_account("6400")
        assert result.primary_category == Scope3Category.CAT_6

    def test_lookup_gl_6600_utilities(self, engine):
        """GL 6600 (Utilities) maps to Cat 3."""
        result = engine.lookup_gl_account("6600")
        assert result.primary_category == Scope3Category.CAT_3

    def test_lookup_gl_6700_rent(self, engine):
        """GL 6700 (Rent & Leases) maps to Cat 8."""
        result = engine.lookup_gl_account("6700")
        assert result.primary_category == Scope3Category.CAT_8

    def test_lookup_gl_7000_capex(self, engine):
        """GL 7000 (Capital Expenditures) maps to Cat 2."""
        result = engine.lookup_gl_account("7000")
        assert result.primary_category == Scope3Category.CAT_2

    def test_lookup_gl_8000_waste(self, engine):
        """GL 8000 (Waste Disposal) maps to Cat 5."""
        result = engine.lookup_gl_account("8000")
        assert result.primary_category == Scope3Category.CAT_5

    def test_lookup_gl_8100_distribution(self, engine):
        """GL 8100 (Distribution / Outbound) maps to Cat 9."""
        result = engine.lookup_gl_account("8100")
        assert result.primary_category == Scope3Category.CAT_9

    def test_lookup_gl_8200_franchise(self, engine):
        """GL 8200 (Franchise Fees) maps to Cat 14."""
        result = engine.lookup_gl_account("8200")
        assert result.primary_category == Scope3Category.CAT_14

    def test_lookup_gl_8300_investment(self, engine):
        """GL 8300 (Investment Expenses) maps to Cat 15."""
        result = engine.lookup_gl_account("8300")
        assert result.primary_category == Scope3Category.CAT_15

    def test_lookup_gl_8400_commuting(self, engine):
        """GL 8400 (Employee Commuting Programs) maps to Cat 7."""
        result = engine.lookup_gl_account("8400")
        assert result.primary_category == Scope3Category.CAT_7

    def test_lookup_gl_6500_fleet(self, engine):
        """GL 6500 (Vehicle / Fleet) maps to Cat 3."""
        result = engine.lookup_gl_account("6500")
        assert result.primary_category == Scope3Category.CAT_3

    def test_lookup_gl_midrange_value(self, engine):
        """A code in the middle of a range (5150 within 5000-5199) resolves correctly."""
        result = engine.lookup_gl_account("5150")
        assert result.primary_category == Scope3Category.CAT_1
        assert result.matched_range == "5000-5199"

    def test_lookup_gl_boundary_start(self, engine):
        """First code in range (5000) matches."""
        result = engine.lookup_gl_account("5000")
        assert result.matched_range == "5000-5199"

    def test_lookup_gl_boundary_end(self, engine):
        """Last code in range (5199) matches."""
        result = engine.lookup_gl_account("5199")
        assert result.matched_range == "5000-5199"
        assert result.primary_category == Scope3Category.CAT_1

    def test_lookup_gl_out_of_range(self, engine):
        """Code outside any defined range raises ValueError."""
        with pytest.raises(ValueError, match="No GL account mapping found"):
            engine.lookup_gl_account("9999")

    def test_lookup_gl_non_numeric(self, engine):
        """Non-numeric GL code raises ValueError."""
        with pytest.raises(ValueError, match="Invalid GL account code"):
            engine.lookup_gl_account("ABCD")

    def test_lookup_gl_result_type(self, engine):
        """GL lookup returns GLLookupResult model."""
        result = engine.lookup_gl_account("5000")
        assert isinstance(result, GLLookupResult)

    def test_lookup_gl_returns_provenance_hash(self, engine):
        """GL lookup returns a 64-character provenance hash."""
        result = engine.lookup_gl_account("6400")
        assert len(result.provenance_hash) == 64

    @pytest.mark.parametrize("gl_code,expected_cat", [
        ("5000", Scope3Category.CAT_1),   # Materials
        ("5300", Scope3Category.CAT_4),   # Freight In
        ("5500", Scope3Category.CAT_1),   # Packaging
        ("6100", Scope3Category.CAT_1),   # Office Supplies
        ("6400", Scope3Category.CAT_6),   # Travel
        ("6500", Scope3Category.CAT_3),   # Vehicle/Fleet
        ("6600", Scope3Category.CAT_3),   # Utilities
        ("6700", Scope3Category.CAT_8),   # Rent
        ("6800", Scope3Category.CAT_1),   # Insurance
        ("7000", Scope3Category.CAT_2),   # CapEx
        ("8000", Scope3Category.CAT_5),   # Waste
        ("8100", Scope3Category.CAT_9),   # Distribution
        ("8200", Scope3Category.CAT_14),  # Franchise
        ("8300", Scope3Category.CAT_15),  # Investment
        ("8400", Scope3Category.CAT_7),   # Commuting
    ])
    def test_lookup_gl_all_ranges_mapped(self, engine, gl_code, expected_cat):
        """All defined GL account ranges return the expected category."""
        result = engine.lookup_gl_account(gl_code)
        assert result.primary_category == expected_cat

    def test_lookup_gl_secondary_categories(self, engine):
        """GL ranges with secondary categories return them correctly."""
        result = engine.lookup_gl_account("5300")
        assert Scope3Category.CAT_9 in result.secondary_categories

    def test_lookup_gl_confidence(self, engine):
        """GL lookup returns confidence of 0.85."""
        result = engine.lookup_gl_account("5000")
        assert result.confidence == Decimal("0.85000000")


# ==============================================================================
# KEYWORD LOOKUP TESTS (~20)
# ==============================================================================


class TestKeywordLookups:
    """Test keyword dictionary to Scope 3 category lookups."""

    def test_lookup_keyword_raw_materials(self, engine):
        """Keyword 'raw materials' maps to Cat 1."""
        result = engine.lookup_keyword("Purchase of raw materials")
        assert result.primary_category == Scope3Category.CAT_1

    def test_lookup_keyword_machinery(self, engine):
        """Keyword 'machinery' maps to Cat 2."""
        result = engine.lookup_keyword("Industrial machinery purchase")
        assert result.primary_category == Scope3Category.CAT_2

    def test_lookup_keyword_electricity(self, engine):
        """Keyword 'electricity' maps to Cat 3."""
        result = engine.lookup_keyword("Monthly electricity bill")
        assert result.primary_category == Scope3Category.CAT_3

    def test_lookup_keyword_freight(self, engine):
        """Keyword 'freight' maps to Cat 4."""
        result = engine.lookup_keyword("Inbound freight charges")
        assert result.primary_category == Scope3Category.CAT_4

    def test_lookup_keyword_waste_disposal(self, engine):
        """Keyword 'waste disposal' maps to Cat 5."""
        result = engine.lookup_keyword("Hazardous waste disposal fee")
        assert result.primary_category == Scope3Category.CAT_5

    def test_lookup_keyword_air_travel(self, engine):
        """Keyword 'air travel' maps to Cat 6."""
        result = engine.lookup_keyword("Employee air travel booking")
        assert result.primary_category == Scope3Category.CAT_6

    def test_lookup_keyword_commuting(self, engine):
        """Keyword 'commuting' maps to Cat 7."""
        result = engine.lookup_keyword("Employee commuting program")
        assert result.primary_category == Scope3Category.CAT_7

    def test_lookup_keyword_office_lease(self, engine):
        """Keyword 'office lease' maps to Cat 8."""
        result = engine.lookup_keyword("Monthly office lease payment")
        assert result.primary_category == Scope3Category.CAT_8

    def test_lookup_keyword_distribution(self, engine):
        """Keyword 'outbound freight' maps to Cat 9."""
        result = engine.lookup_keyword("Outbound freight to customer")
        assert result.primary_category == Scope3Category.CAT_9

    def test_lookup_keyword_processing_sold(self, engine):
        """Keyword 'intermediate product' maps to Cat 10."""
        result = engine.lookup_keyword("Sale of intermediate product")
        assert result.primary_category == Scope3Category.CAT_10

    def test_lookup_keyword_use_sold_products(self, engine):
        """Keyword 'product use' maps to Cat 11."""
        result = engine.lookup_keyword("Estimating product use emissions")
        assert result.primary_category == Scope3Category.CAT_11

    def test_lookup_keyword_end_of_life(self, engine):
        """Keyword 'end of life' maps to Cat 12."""
        result = engine.lookup_keyword("Product end of life treatment")
        assert result.primary_category == Scope3Category.CAT_12

    def test_lookup_keyword_downstream_lease(self, engine):
        """Keyword 'leased to tenant' maps to Cat 13."""
        result = engine.lookup_keyword("Asset leased to tenant")
        assert result.primary_category == Scope3Category.CAT_13

    def test_lookup_keyword_franchise(self, engine):
        """Keyword 'franchise' maps to Cat 14."""
        result = engine.lookup_keyword("Franchise fee revenue")
        assert result.primary_category == Scope3Category.CAT_14

    def test_lookup_keyword_investments(self, engine):
        """Keyword 'investment' maps to Cat 15."""
        result = engine.lookup_keyword("Equity investment portfolio")
        assert result.primary_category == Scope3Category.CAT_15

    def test_lookup_keyword_case_insensitive(self, engine):
        """Keyword lookup is case-insensitive."""
        lower = engine.lookup_keyword("raw materials")
        upper = engine.lookup_keyword("RAW MATERIALS")
        mixed = engine.lookup_keyword("Raw Materials")

        assert lower.primary_category == upper.primary_category
        assert upper.primary_category == mixed.primary_category

    def test_lookup_keyword_partial_match(self, engine):
        """Keyword is found as a substring within longer text."""
        result = engine.lookup_keyword(
            "Invoice #4521 for waste disposal services rendered in Q3"
        )
        assert result.primary_category == Scope3Category.CAT_5
        assert "waste disposal" in result.matched_keyword

    def test_lookup_keyword_longest_match_wins(self, engine):
        """When multiple keywords match, the longest is selected."""
        result = engine.lookup_keyword("inbound freight charge from carrier")
        # "inbound freight" (15 chars) should beat "freight" (7 chars)
        assert len(result.matched_keyword) >= len("freight")

    def test_lookup_keyword_no_match(self, engine):
        """Text with no matching keyword raises ValueError."""
        with pytest.raises(ValueError, match="No keyword match found"):
            engine.lookup_keyword("xyz123 completely unrelated gibberish")

    def test_lookup_keyword_empty_text(self, engine):
        """Empty text raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            engine.lookup_keyword("")

    def test_lookup_keyword_result_type(self, engine):
        """Keyword lookup returns KeywordLookupResult model."""
        result = engine.lookup_keyword("raw materials purchase")
        assert isinstance(result, KeywordLookupResult)

    def test_lookup_keyword_returns_keyword_group(self, engine):
        """Keyword lookup includes the keyword group in the result."""
        result = engine.lookup_keyword("waste disposal fee")
        assert result.keyword_group == "waste_disposal"

    def test_lookup_keyword_returns_provenance_hash(self, engine):
        """Keyword lookup returns a 64-character provenance hash."""
        result = engine.lookup_keyword("raw materials")
        assert len(result.provenance_hash) == 64

    def test_lookup_keyword_provenance_deterministic(self, engine):
        """Same keyword input produces identical provenance hash."""
        r1 = engine.lookup_keyword("air travel booking")
        r2 = engine.lookup_keyword("air travel booking")
        assert r1.provenance_hash == r2.provenance_hash

    def test_lookup_keywords_batch(self, engine):
        """Batch keyword lookup processes multiple texts successfully."""
        texts = [
            "raw materials purchase",
            "air travel expense report",
            "waste disposal fee",
            "electricity bill payment",
        ]
        results = engine.lookup_keywords_batch(texts)
        assert len(results) == 4

        categories = [r.primary_category for r in results]
        assert Scope3Category.CAT_1 in categories
        assert Scope3Category.CAT_6 in categories
        assert Scope3Category.CAT_5 in categories
        assert Scope3Category.CAT_3 in categories

    def test_lookup_keywords_batch_skips_unmatched(self, engine):
        """Batch lookup skips texts with no keyword match."""
        texts = [
            "raw materials purchase",
            "xyz123 completely unrelated gibberish",
            "waste disposal fee",
        ]
        results = engine.lookup_keywords_batch(texts)
        assert len(results) == 2

    def test_lookup_keywords_batch_empty(self, engine):
        """Batch lookup on empty list returns empty list."""
        results = engine.lookup_keywords_batch([])
        assert results == []


# ==============================================================================
# CATEGORY INFO TESTS (~20)
# ==============================================================================


class TestCategoryInfo:
    """Test category metadata retrieval for all 15 Scope 3 categories."""

    @pytest.mark.parametrize("cat_num", list(range(1, 16)))
    def test_get_category_info_all_15(self, engine, cat_num):
        """Each category number (1-15) returns valid CategoryInfo."""
        info = engine.get_category_info(cat_num)
        assert isinstance(info, CategoryInfo)
        assert info.number == cat_num
        assert len(info.name) > 0
        assert len(info.description) > 0

    @pytest.mark.parametrize("cat_num", [1, 2, 3, 4, 5, 6, 7, 8])
    def test_get_category_info_upstream_cats(self, engine, cat_num):
        """Categories 1-8 have direction=upstream."""
        info = engine.get_category_info(cat_num)
        assert info.direction == ValueChainDirection.UPSTREAM

    @pytest.mark.parametrize("cat_num", [9, 10, 11, 12, 13, 14, 15])
    def test_get_category_info_downstream_cats(self, engine, cat_num):
        """Categories 9-15 have direction=downstream."""
        info = engine.get_category_info(cat_num)
        assert info.direction == ValueChainDirection.DOWNSTREAM

    def test_get_all_categories_returns_15(self, engine):
        """get_all_categories returns exactly 15 CategoryInfo objects."""
        categories = engine.get_all_categories()
        assert len(categories) == 15
        numbers = [c.number for c in categories]
        assert numbers == list(range(1, 16))

    def test_get_mapping_version(self, engine):
        """get_mapping_version returns the expected version string."""
        version = engine.get_mapping_version()
        assert version == MAPPING_VERSION
        assert version == "2026.1.0"

    def test_category_info_has_required_fields(self, engine):
        """CategoryInfo includes all required metadata fields."""
        info = engine.get_category_info(1)
        assert info.number == 1
        assert len(info.name) > 0
        assert len(info.description) > 0
        assert info.direction in (ValueChainDirection.UPSTREAM, ValueChainDirection.DOWNSTREAM)
        assert len(info.ghg_protocol_chapter) > 0
        assert len(info.reporter_role) > 0
        assert isinstance(info.typical_data_sources, list)
        assert len(info.typical_data_sources) > 0
        assert info.downstream_agent.startswith("AGENT-MRV-")

    @pytest.mark.parametrize("cat_num,expected_agent", [
        (1, "AGENT-MRV-014"),
        (2, "AGENT-MRV-015"),
        (3, "AGENT-MRV-016"),
        (4, "AGENT-MRV-017"),
        (5, "AGENT-MRV-018"),
        (6, "AGENT-MRV-019"),
        (7, "AGENT-MRV-020"),
        (8, "AGENT-MRV-021"),
        (9, "AGENT-MRV-022"),
        (10, "AGENT-MRV-023"),
        (11, "AGENT-MRV-024"),
        (12, "AGENT-MRV-025"),
        (13, "AGENT-MRV-026"),
        (14, "AGENT-MRV-027"),
        (15, "AGENT-MRV-028"),
    ])
    def test_category_info_downstream_agent_mapping(self, engine, cat_num, expected_agent):
        """Each category maps to the correct downstream MRV agent."""
        info = engine.get_category_info(cat_num)
        assert info.downstream_agent == expected_agent

    def test_get_category_info_invalid_zero(self, engine):
        """Category number 0 raises ValueError."""
        with pytest.raises(ValueError, match="Invalid category number"):
            engine.get_category_info(0)

    def test_get_category_info_invalid_16(self, engine):
        """Category number 16 raises ValueError."""
        with pytest.raises(ValueError, match="Invalid category number"):
            engine.get_category_info(16)

    def test_get_category_info_negative(self, engine):
        """Negative category number raises ValueError."""
        with pytest.raises(ValueError, match="Invalid category number"):
            engine.get_category_info(-1)


# ==============================================================================
# DATABASE SUMMARY AND STATISTICS TESTS
# ==============================================================================


class TestDatabaseSummary:
    """Test database summary and statistics methods."""

    def test_database_summary_keys(self, engine):
        """Summary dict contains all expected keys."""
        summary = engine.get_database_summary()
        expected_keys = {
            "agent_id", "agent_component", "version", "mapping_version",
            "naics_sector_count", "naics_subsector_count",
            "isic_section_count", "isic_division_count",
            "naics_isic_concordance_entries",
            "gl_account_ranges", "keyword_count", "category_count",
            "total_lookups",
        }
        assert expected_keys.issubset(set(summary.keys()))

    def test_database_summary_counts(self, engine):
        """Summary reflects correct table sizes."""
        summary = engine.get_database_summary()
        assert summary["naics_sector_count"] == 24  # 20 sectors but 24 entries (31/32/33, 44/45, 48/49)
        assert summary["isic_section_count"] == 21
        assert summary["category_count"] == 15
        assert summary["keyword_count"] > 400

    def test_lookup_count_increments(self, engine):
        """Lookup count increments with each lookup call."""
        initial = engine.get_lookup_count()
        engine.lookup_naics("11")
        engine.lookup_isic("A")
        engine.lookup_gl_account("5000")
        engine.lookup_keyword("raw materials")
        assert engine.get_lookup_count() == initial + 4

    def test_agent_metadata(self, engine):
        """Agent metadata constants are correct."""
        assert AGENT_ID == "GL-MRV-X-040"
        assert AGENT_COMPONENT == "AGENT-MRV-029"
        assert VERSION == "1.0.0"
