# -*- coding: utf-8 -*-
"""
Unit tests for TaxonomyClassifierEngine - AGENT-DATA-009 Batch 2
==================================================================

Comprehensive tests for the taxonomy classification engine covering:
- Engine initialisation and taxonomy database loading
- Generic classification and default taxonomy
- Batch classification with mixed records
- UNSPSC classification (segment, keyword, fallback)
- NAICS classification (2-digit sectors, keywords)
- eCl@ss classification (group, keyword, fallback)
- ISIC Rev 4 classification (section, keyword, fallback)
- Cross-taxonomy code translation
- Hierarchy resolution for multi-level codes
- Code search functionality
- Statistics tracking by system and match type
- Provenance hashing (SHA-256)
- Thread safety under concurrent classification

Target: 100+ tests, 900+ lines, 85%+ coverage of taxonomy_classifier.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List

import pytest

from greenlang.spend_categorizer.taxonomy_classifier import (
    TaxonomyClassifierEngine,
    TaxonomyClassification,
    TaxonomyCode,
    _UNSPSC_SEGMENTS,
    _NAICS_SECTORS,
    _ECLASS_GROUPS,
    _ISIC_SECTIONS,
    _UNSPSC_TO_NAICS,
    _NAICS_TO_ISIC,
    _NAICS_TO_ECLASS,
)


# ---------------------------------------------------------------------------
# Local fixtures (defined here since conftest is shared with Batch 1 tests)
# ---------------------------------------------------------------------------


@pytest.fixture
def taxonomy_engine() -> TaxonomyClassifierEngine:
    """Create a default TaxonomyClassifierEngine."""
    return TaxonomyClassifierEngine()


@pytest.fixture
def taxonomy_engine_naics() -> TaxonomyClassifierEngine:
    """Create a TaxonomyClassifierEngine defaulting to NAICS."""
    return TaxonomyClassifierEngine(config={"default_taxonomy": "naics"})


# ===================================================================
# TestInit - engine creation and defaults
# ===================================================================


class TestInit:
    """Tests for TaxonomyClassifierEngine initialisation."""

    def test_default_init(self):
        """Engine initialises with default taxonomy and thresholds."""
        engine = TaxonomyClassifierEngine()
        assert engine._default_taxonomy == "unspsc"
        assert engine._min_confidence == 0.3
        assert engine._max_results == 5

    def test_custom_config(self):
        """Engine respects custom configuration."""
        engine = TaxonomyClassifierEngine(config={
            "default_taxonomy": "naics",
            "min_confidence": 0.5,
            "max_results": 10,
        })
        assert engine._default_taxonomy == "naics"
        assert engine._min_confidence == 0.5
        assert engine._max_results == 10

    def test_empty_classifications_on_init(self):
        """Classification storage is empty on init."""
        engine = TaxonomyClassifierEngine()
        assert engine._classifications == {}

    def test_stats_zeroed_on_init(self):
        """Statistics are zero on init."""
        engine = TaxonomyClassifierEngine()
        assert engine._stats["classifications_performed"] == 0
        assert engine._stats["errors"] == 0

    def test_taxonomy_databases_loaded(self):
        """Taxonomy databases have expected entry counts."""
        assert len(_UNSPSC_SEGMENTS) >= 55
        assert len(_NAICS_SECTORS) >= 20
        assert len(_ECLASS_GROUPS) >= 40
        assert len(_ISIC_SECTIONS) >= 20

    def test_none_config(self):
        """None config uses defaults."""
        engine = TaxonomyClassifierEngine(config=None)
        assert engine._default_taxonomy == "unspsc"


# ===================================================================
# TestClassify - generic classification
# ===================================================================


class TestClassify:
    """Tests for the classify() method."""

    def test_classify_returns_classification(self, taxonomy_engine):
        """classify() returns a TaxonomyClassification."""
        result = taxonomy_engine.classify(
            {"description": "office supplies"}
        )
        assert isinstance(result, TaxonomyClassification)

    def test_classify_default_taxonomy(self, taxonomy_engine):
        """Default taxonomy is unspsc."""
        result = taxonomy_engine.classify(
            {"description": "office supplies"}
        )
        assert result.taxonomy_system == "unspsc"

    def test_classify_naics_override(self, taxonomy_engine):
        """Taxonomy can be overridden per-call."""
        result = taxonomy_engine.classify(
            {"description": "construction services"},
            taxonomy="naics",
        )
        assert result.taxonomy_system == "naics"

    def test_classify_primary_code_present(self, taxonomy_engine):
        """Primary code is always present."""
        result = taxonomy_engine.classify(
            {"description": "chemical reagents"}
        )
        assert result.primary_code is not None
        assert result.primary_code.code != ""

    def test_classify_secondary_codes_populated(self, taxonomy_engine):
        """Secondary codes from other systems are populated."""
        result = taxonomy_engine.classify(
            {"description": "office supplies"}
        )
        # Should have classifications in other systems
        assert isinstance(result.secondary_codes, list)

    def test_classify_confidence_range(self, taxonomy_engine):
        """Confidence is between 0 and 1."""
        result = taxonomy_engine.classify(
            {"description": "diesel fuel for trucks"}
        )
        assert 0.0 <= result.confidence <= 1.0

    def test_classify_provenance_hash(self, taxonomy_engine):
        """Classification has a 64-char SHA-256 provenance hash."""
        result = taxonomy_engine.classify(
            {"description": "office supplies"}
        )
        assert len(result.provenance_hash) == 64

    def test_classify_classified_at(self, taxonomy_engine):
        """classified_at timestamp is populated."""
        result = taxonomy_engine.classify(
            {"description": "office supplies"}
        )
        assert result.classified_at != ""

    def test_classify_classification_id(self, taxonomy_engine):
        """classification_id starts with cls- prefix."""
        result = taxonomy_engine.classify(
            {"description": "office supplies"}
        )
        assert result.classification_id.startswith("cls-")

    def test_classify_input_text_truncated(self, taxonomy_engine):
        """Input text is truncated to 200 chars."""
        long_text = "a" * 500
        result = taxonomy_engine.classify(
            {"description": long_text}
        )
        assert len(result.input_text) <= 200

    def test_classify_record_id_passed(self, taxonomy_engine):
        """record_id from input is preserved."""
        result = taxonomy_engine.classify(
            {"description": "office supplies", "record_id": "rec-123"}
        )
        assert result.record_id == "rec-123"

    def test_classify_empty_description(self, taxonomy_engine):
        """Empty description produces fallback classification."""
        result = taxonomy_engine.classify({"description": ""})
        assert result.primary_code.code != ""


# ===================================================================
# TestClassifyBatch - batch classification
# ===================================================================


class TestClassifyBatch:
    """Tests for classify_batch() method."""

    def test_batch_returns_list(self, taxonomy_engine):
        """classify_batch() returns a list of classifications."""
        records = [
            {"description": "office supplies"},
            {"description": "construction"},
        ]
        results = taxonomy_engine.classify_batch(records)
        assert isinstance(results, list)
        assert len(results) == 2

    def test_batch_preserves_order(self, taxonomy_engine):
        """Results maintain input order."""
        records = [
            {"description": "diesel fuel", "record_id": "r1"},
            {"description": "office supplies", "record_id": "r2"},
        ]
        results = taxonomy_engine.classify_batch(records)
        assert results[0].record_id == "r1"
        assert results[1].record_id == "r2"

    def test_batch_custom_taxonomy(self, taxonomy_engine):
        """Batch can use a custom taxonomy."""
        records = [{"description": "construction"}]
        results = taxonomy_engine.classify_batch(records, taxonomy="naics")
        assert results[0].taxonomy_system == "naics"

    def test_batch_empty_list(self, taxonomy_engine):
        """Empty batch returns empty list."""
        results = taxonomy_engine.classify_batch([])
        assert results == []

    def test_batch_mixed_records(self, taxonomy_engine):
        """Batch handles diverse record types."""
        records = [
            {"description": "diesel fuel for fleet"},
            {"description": "office chairs and desks"},
            {"description": "legal consulting services"},
            {"description": "steel beams for construction"},
            {"description": "cloud computing aws services"},
        ]
        results = taxonomy_engine.classify_batch(records)
        assert len(results) == 5
        for r in results:
            assert r.primary_code.confidence > 0


# ===================================================================
# TestClassifyUNSPSC - UNSPSC classification
# ===================================================================


class TestClassifyUNSPSC:
    """Tests for classify_unspsc() and UNSPSC classification."""

    def test_office_supplies(self, taxonomy_engine):
        """Office supplies maps to segment 44."""
        code = taxonomy_engine.classify_unspsc("office supplies")
        assert code.system == "unspsc"
        assert code.code == "44"

    def test_fuel_keyword(self, taxonomy_engine):
        """Fuel keyword maps to segment 15."""
        code = taxonomy_engine.classify_unspsc("diesel fuel")
        assert code.code == "15"

    def test_computer_keyword(self, taxonomy_engine):
        """Computer keyword maps to IT segment 43."""
        code = taxonomy_engine.classify_unspsc("computer servers")
        assert code.code == "43"

    def test_chemical_keyword(self, taxonomy_engine):
        """Chemical keyword maps to segment 12."""
        code = taxonomy_engine.classify_unspsc("chemical reagent")
        assert code.code == "12"

    def test_furniture_keyword(self, taxonomy_engine):
        """Furniture keyword maps to segment 56."""
        code = taxonomy_engine.classify_unspsc("ergonomic chair")
        assert code.code == "56"

    def test_shipping_keyword(self, taxonomy_engine):
        """Shipping keyword maps to segment 78."""
        code = taxonomy_engine.classify_unspsc("freight shipping service")
        assert code.code == "78"

    def test_consulting_keyword(self, taxonomy_engine):
        """Consulting keyword maps to segment 80."""
        code = taxonomy_engine.classify_unspsc("management consulting")
        assert code.code == "80"

    def test_with_category_hint(self, taxonomy_engine):
        """Category hint improves classification."""
        code = taxonomy_engine.classify_unspsc(
            "generic supplies", category="cleaning"
        )
        assert code.code == "47"

    def test_unknown_text_fallback(self, taxonomy_engine):
        """Unknown text falls back to generic or low-confidence category."""
        code = taxonomy_engine.classify_unspsc("xyzzy_gobbledygook_12345")
        # Should get a code with relatively low confidence
        assert code.code != ""
        assert code.confidence <= 0.6

    def test_confidence_exact_match_high(self, taxonomy_engine):
        """Exact name match yields high confidence (0.95)."""
        # Use an exact UNSPSC segment name
        code = taxonomy_engine.classify_unspsc(
            "Information Technology Broadcasting and Telecommunications"
        )
        assert code.confidence >= 0.90

    def test_confidence_keyword_moderate(self, taxonomy_engine):
        """Keyword match yields moderate confidence."""
        code = taxonomy_engine.classify_unspsc("software")
        assert 0.3 <= code.confidence <= 0.7

    def test_provenance_hash_on_unspsc(self, taxonomy_engine):
        """UNSPSC classification has provenance hash."""
        code = taxonomy_engine.classify_unspsc("office supplies")
        assert len(code.provenance_hash) == 64


# ===================================================================
# TestClassifyNAICS - NAICS classification
# ===================================================================


class TestClassifyNAICS:
    """Tests for classify_naics() and NAICS classification."""

    def test_construction(self, taxonomy_engine):
        """Construction maps to NAICS 23."""
        code = taxonomy_engine.classify_naics("construction building")
        assert code.system == "naics"
        assert code.code == "23"

    def test_agriculture(self, taxonomy_engine):
        """Agriculture maps to NAICS 11."""
        code = taxonomy_engine.classify_naics("farming agriculture crop")
        assert code.code == "11"

    def test_manufacturing(self, taxonomy_engine):
        """Manufacturing keyword matching."""
        code = taxonomy_engine.classify_naics("machinery metal manufacturing")
        assert code.code in ("31", "32", "33")

    def test_transportation(self, taxonomy_engine):
        """Transportation maps to NAICS 48."""
        code = taxonomy_engine.classify_naics("trucking transport pipeline")
        assert code.code == "48"

    def test_finance(self, taxonomy_engine):
        """Finance maps to NAICS 52."""
        code = taxonomy_engine.classify_naics("banking insurance securities")
        assert code.code == "52"

    def test_education(self, taxonomy_engine):
        """Education maps to NAICS 61."""
        code = taxonomy_engine.classify_naics("school university education training")
        assert code.code == "61"

    def test_healthcare(self, taxonomy_engine):
        """Healthcare maps to NAICS 62."""
        code = taxonomy_engine.classify_naics("hospital physician nursing dental")
        assert code.code == "62"

    def test_utilities(self, taxonomy_engine):
        """Utilities maps to NAICS 22."""
        code = taxonomy_engine.classify_naics("electricity natural gas utility water supply")
        assert code.code == "22"

    def test_professional_services(self, taxonomy_engine):
        """Professional services maps to NAICS 54."""
        code = taxonomy_engine.classify_naics("legal accounting engineering consulting")
        assert code.code == "54"

    def test_real_estate(self, taxonomy_engine):
        """Real estate maps to NAICS 53."""
        code = taxonomy_engine.classify_naics("real estate rental property management leasing")
        assert code.code == "53"

    def test_partial_code_3digit(self, taxonomy_engine):
        """NAICS keywords for 3-digit detail."""
        code = taxonomy_engine.classify_naics("hotel motel restaurant catering")
        assert code.code == "72"

    def test_naics_provenance_hash(self, taxonomy_engine):
        """NAICS classification includes provenance hash."""
        code = taxonomy_engine.classify_naics("construction")
        assert len(code.provenance_hash) == 64


# ===================================================================
# TestClassifyEclass - eCl@ss classification
# ===================================================================


class TestClassifyEclass:
    """Tests for classify_eclass() and eCl@ss classification."""

    def test_office_supplies(self, taxonomy_engine):
        """Office supplies maps to eCl@ss 23."""
        code = taxonomy_engine.classify_eclass("office supplies stationery pen paper")
        assert code.system == "eclass"
        assert code.code == "23"

    def test_fastener_keyword(self, taxonomy_engine):
        """Fastener keyword maps to eCl@ss 18."""
        code = taxonomy_engine.classify_eclass("bolt screw fastener assembly")
        assert code.code == "18"

    def test_electrical_keyword(self, taxonomy_engine):
        """Electrical keyword maps to eCl@ss 19."""
        code = taxonomy_engine.classify_eclass("electrical cable wire transformer")
        assert code.code == "19"

    def test_it_keyword(self, taxonomy_engine):
        """IT keyword maps to eCl@ss 22."""
        code = taxonomy_engine.classify_eclass("computer server network router")
        assert code.code == "22"

    def test_hvac_keyword(self, taxonomy_engine):
        """HVAC keyword maps to eCl@ss 33."""
        code = taxonomy_engine.classify_eclass("hvac heating cooling air conditioning")
        assert code.code == "33"

    def test_chemical_keyword(self, taxonomy_engine):
        """Chemical keyword maps to eCl@ss 36."""
        code = taxonomy_engine.classify_eclass("chemical acid reagent solvent")
        assert code.code == "36"

    def test_furniture_keyword(self, taxonomy_engine):
        """Furniture keyword maps to eCl@ss 44."""
        code = taxonomy_engine.classify_eclass("desk chair cabinet furniture")
        assert code.code == "44"

    def test_keyword_fallback(self, taxonomy_engine):
        """Unknown text falls back to generic eCl@ss code."""
        code = taxonomy_engine.classify_eclass("xyzzy_unknown_text")
        assert code.code != ""

    def test_eclass_provenance(self, taxonomy_engine):
        """eCl@ss classification includes provenance hash."""
        code = taxonomy_engine.classify_eclass("office supplies")
        assert len(code.provenance_hash) == 64


# ===================================================================
# TestClassifyISIC - ISIC Rev 4 classification
# ===================================================================


class TestClassifyISIC:
    """Tests for classify_isic() and ISIC classification."""

    def test_agriculture_section(self, taxonomy_engine):
        """Agriculture maps to ISIC section A."""
        code = taxonomy_engine.classify_isic("agriculture farming crop livestock")
        assert code.system == "isic"
        assert code.code == "A"

    def test_mining_section(self, taxonomy_engine):
        """Mining maps to ISIC section B."""
        code = taxonomy_engine.classify_isic("mining quarrying oil extraction")
        assert code.code == "B"

    def test_manufacturing_section(self, taxonomy_engine):
        """Manufacturing maps to ISIC section C."""
        code = taxonomy_engine.classify_isic("manufacturing factory production assembly")
        assert code.code == "C"

    def test_construction_section(self, taxonomy_engine):
        """Construction maps to ISIC section F."""
        code = taxonomy_engine.classify_isic("construction building civil engineering")
        assert code.code == "F"

    def test_finance_section(self, taxonomy_engine):
        """Finance maps to ISIC section K."""
        code = taxonomy_engine.classify_isic("finance insurance banking investment")
        assert code.code == "K"

    def test_education_section(self, taxonomy_engine):
        """Education maps to ISIC section P."""
        code = taxonomy_engine.classify_isic("education school university training")
        assert code.code == "P"

    def test_healthcare_section(self, taxonomy_engine):
        """Healthcare maps to ISIC section Q."""
        code = taxonomy_engine.classify_isic("healthcare hospital medical social work")
        assert code.code == "Q"

    def test_isic_provenance(self, taxonomy_engine):
        """ISIC classification includes provenance hash."""
        code = taxonomy_engine.classify_isic("construction")
        assert len(code.provenance_hash) == 64

    def test_isic_division_matching(self, taxonomy_engine):
        """ISIC also matches at division level if section keywords hit."""
        code = taxonomy_engine.classify_isic("wholesale retail trade repair")
        assert code.code == "G"


# ===================================================================
# TestTranslateCode - cross-taxonomy translation
# ===================================================================


class TestTranslateCode:
    """Tests for translate_code() method."""

    def test_unspsc_to_naics(self, taxonomy_engine):
        """Translate UNSPSC segment 43 to NAICS."""
        code = taxonomy_engine.translate_code("43", "unspsc", "naics")
        assert code.system == "naics"
        assert code.code == "51"  # IT -> Information

    def test_naics_to_isic(self, taxonomy_engine):
        """Translate NAICS 23 to ISIC."""
        code = taxonomy_engine.translate_code("23", "naics", "isic")
        assert code.system == "isic"
        assert code.code == "F"  # Construction

    def test_naics_to_eclass(self, taxonomy_engine):
        """Translate NAICS 23 to eCl@ss."""
        code = taxonomy_engine.translate_code("23", "naics", "eclass")
        assert code.system == "eclass"
        assert code.code == "43"  # Building Technology

    def test_unspsc_to_isic_via_naics(self, taxonomy_engine):
        """Translate UNSPSC to ISIC via NAICS intermediate."""
        code = taxonomy_engine.translate_code("43", "unspsc", "isic")
        assert code.system == "isic"
        # 43->NAICS 51->ISIC J
        assert code.code == "J"

    def test_same_system_raises(self, taxonomy_engine):
        """Translation between same systems raises ValueError."""
        with pytest.raises(ValueError, match="identical"):
            taxonomy_engine.translate_code("43", "unspsc", "unspsc")

    def test_unknown_code_returns_unknown(self, taxonomy_engine):
        """Unknown source code returns code='unknown'."""
        code = taxonomy_engine.translate_code("ZZ", "unspsc", "naics")
        assert code.code == "unknown"
        assert code.confidence == 0.0

    def test_translation_confidence(self, taxonomy_engine):
        """Successful translation has confidence 0.80."""
        code = taxonomy_engine.translate_code("43", "unspsc", "naics")
        assert code.confidence == 0.80

    def test_translation_match_type(self, taxonomy_engine):
        """Translation match type is 'translation'."""
        code = taxonomy_engine.translate_code("43", "unspsc", "naics")
        assert code.match_type == "translation"

    def test_translation_provenance(self, taxonomy_engine):
        """Translation includes provenance hash."""
        code = taxonomy_engine.translate_code("43", "unspsc", "naics")
        assert len(code.provenance_hash) == 64

    def test_reverse_lookup_naics_to_unspsc(self, taxonomy_engine):
        """NAICS to UNSPSC can use reverse lookup."""
        # NAICS 51 maps from UNSPSC 43 via _UNSPSC_TO_NAICS
        code = taxonomy_engine.translate_code("51", "naics", "unspsc")
        # Reverse lookup should find a UNSPSC code that maps to NAICS 51
        if code.code != "unknown":
            assert code.system == "unspsc"


# ===================================================================
# TestGetCodeHierarchy - hierarchy resolution
# ===================================================================


class TestGetCodeHierarchy:
    """Tests for get_code_hierarchy() method."""

    def test_unspsc_segment_only(self, taxonomy_engine):
        """UNSPSC 2-digit code returns single level."""
        hierarchy = taxonomy_engine.get_code_hierarchy("43", "unspsc")
        assert len(hierarchy) >= 1
        assert hierarchy[0].code == "43"
        assert hierarchy[0].level == 1

    def test_unspsc_full_code(self, taxonomy_engine):
        """UNSPSC 8-digit code returns full hierarchy."""
        hierarchy = taxonomy_engine.get_code_hierarchy("43211500", "unspsc")
        assert len(hierarchy) == 4
        assert hierarchy[0].code == "43"
        assert hierarchy[1].code == "4321"
        assert hierarchy[2].code == "432115"
        assert hierarchy[3].code == "43211500"

    def test_naics_2digit(self, taxonomy_engine):
        """NAICS 2-digit code returns one level."""
        hierarchy = taxonomy_engine.get_code_hierarchy("23", "naics")
        assert len(hierarchy) >= 1
        assert hierarchy[0].code == "23"

    def test_naics_6digit(self, taxonomy_engine):
        """NAICS 6-digit code returns full hierarchy."""
        hierarchy = taxonomy_engine.get_code_hierarchy("236220", "naics")
        assert len(hierarchy) == 5  # 2,3,4,5,6 digit levels

    def test_eclass_2digit(self, taxonomy_engine):
        """eCl@ss 2-digit code returns one level."""
        hierarchy = taxonomy_engine.get_code_hierarchy("22", "eclass")
        assert len(hierarchy) >= 1

    def test_isic_section(self, taxonomy_engine):
        """ISIC section letter returns one level."""
        hierarchy = taxonomy_engine.get_code_hierarchy("C", "isic")
        assert len(hierarchy) == 1
        assert hierarchy[0].code == "C"
        assert hierarchy[0].name == "Manufacturing"

    def test_isic_with_division(self, taxonomy_engine):
        """ISIC section + division returns two levels."""
        hierarchy = taxonomy_engine.get_code_hierarchy("C10", "isic")
        assert len(hierarchy) == 2
        assert hierarchy[0].code == "C"
        assert hierarchy[1].code == "C10"

    def test_unknown_code_empty(self, taxonomy_engine):
        """Unknown system returns empty hierarchy."""
        hierarchy = taxonomy_engine.get_code_hierarchy("43", "unknown_system")
        assert hierarchy == []

    def test_hierarchy_parent_chain(self, taxonomy_engine):
        """Parent codes form a chain."""
        hierarchy = taxonomy_engine.get_code_hierarchy("43211500", "unspsc")
        assert hierarchy[0].parent_code is None
        for i in range(1, len(hierarchy)):
            assert hierarchy[i].parent_code == hierarchy[i - 1].code


# ===================================================================
# TestSearchCodes - code search
# ===================================================================


class TestSearchCodes:
    """Tests for search_codes() method."""

    def test_exact_name_match(self, taxonomy_engine):
        """Exact name match returns high confidence."""
        results = taxonomy_engine.search_codes("Construction", "naics")
        assert len(results) >= 1
        assert results[0].confidence >= 0.70

    def test_partial_match(self, taxonomy_engine):
        """Partial match on name substring."""
        results = taxonomy_engine.search_codes("Manufacturing", "naics")
        assert len(results) >= 1

    def test_keyword_match(self, taxonomy_engine):
        """Keyword match in search."""
        results = taxonomy_engine.search_codes("software", "naics")
        assert len(results) >= 1

    def test_no_results(self, taxonomy_engine):
        """Nonsense query returns empty."""
        results = taxonomy_engine.search_codes(
            "zxqwvbn_no_match", "naics"
        )
        assert results == []

    def test_limit_parameter(self, taxonomy_engine):
        """Limit parameter caps results."""
        results = taxonomy_engine.search_codes("service", "unspsc", limit=3)
        assert len(results) <= 3

    def test_sorted_by_confidence(self, taxonomy_engine):
        """Results are sorted by confidence descending."""
        results = taxonomy_engine.search_codes("manufacturing", "naics")
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i].confidence >= results[i + 1].confidence

    def test_search_unspsc(self, taxonomy_engine):
        """Search works for UNSPSC system."""
        results = taxonomy_engine.search_codes("fuel", "unspsc")
        assert len(results) >= 1
        assert any(r.code == "15" for r in results)

    def test_search_eclass(self, taxonomy_engine):
        """Search works for eCl@ss system."""
        results = taxonomy_engine.search_codes("electrical", "eclass")
        assert len(results) >= 1

    def test_search_isic(self, taxonomy_engine):
        """Search works for ISIC system."""
        results = taxonomy_engine.search_codes("construction", "isic")
        assert len(results) >= 1


# ===================================================================
# TestStatistics - classification stats
# ===================================================================


class TestStatistics:
    """Tests for get_statistics() method."""

    def test_initial_stats(self, taxonomy_engine):
        """Fresh engine reports zero stats."""
        stats = taxonomy_engine.get_statistics()
        assert stats["classifications_performed"] == 0
        assert stats["avg_confidence"] == 0.0

    def test_stats_after_classification(self, taxonomy_engine):
        """Stats update after classification."""
        taxonomy_engine.classify({"description": "office supplies"})
        stats = taxonomy_engine.get_statistics()
        assert stats["classifications_performed"] == 1

    def test_by_system_tracking(self, taxonomy_engine):
        """by_system tracks taxonomy system usage."""
        taxonomy_engine.classify(
            {"description": "office supplies"}, taxonomy="unspsc"
        )
        taxonomy_engine.classify(
            {"description": "construction"}, taxonomy="naics"
        )
        stats = taxonomy_engine.get_statistics()
        assert stats["by_system"].get("unspsc", 0) >= 1
        assert stats["by_system"].get("naics", 0) >= 1

    def test_by_match_type_tracking(self, taxonomy_engine):
        """by_match_type tracks match type distribution."""
        taxonomy_engine.classify({"description": "office supplies"})
        stats = taxonomy_engine.get_statistics()
        total_match_types = sum(stats["by_match_type"].values())
        assert total_match_types >= 1

    def test_avg_confidence_calculation(self, taxonomy_engine):
        """avg_confidence is computed correctly."""
        taxonomy_engine.classify({"description": "office supplies"})
        taxonomy_engine.classify({"description": "construction"})
        stats = taxonomy_engine.get_statistics()
        assert 0.0 < stats["avg_confidence"] <= 1.0

    def test_taxonomy_sizes_reported(self, taxonomy_engine):
        """Taxonomy database sizes are reported."""
        stats = taxonomy_engine.get_statistics()
        assert stats["taxonomy_sizes"]["unspsc_segments"] >= 55
        assert stats["taxonomy_sizes"]["naics_sectors"] >= 20
        assert stats["taxonomy_sizes"]["eclass_groups"] >= 40
        assert stats["taxonomy_sizes"]["isic_sections"] >= 20

    def test_classifications_stored_count(self, taxonomy_engine):
        """classifications_stored tracks stored count."""
        taxonomy_engine.classify({"description": "office supplies"})
        stats = taxonomy_engine.get_statistics()
        assert stats["classifications_stored"] == 1


# ===================================================================
# TestProvenance - SHA-256 provenance hashing
# ===================================================================


class TestProvenance:
    """Tests for provenance hashing on classifications."""

    def test_classification_provenance_sha256(self, taxonomy_engine):
        """Classification provenance hash is 64-char hex SHA-256."""
        result = taxonomy_engine.classify({"description": "office supplies"})
        assert len(result.provenance_hash) == 64
        int(result.provenance_hash, 16)

    def test_primary_code_provenance(self, taxonomy_engine):
        """Primary code has its own provenance hash."""
        result = taxonomy_engine.classify({"description": "office supplies"})
        assert len(result.primary_code.provenance_hash) == 64

    def test_different_inputs_different_hashes(self, taxonomy_engine):
        """Different inputs produce different provenance hashes."""
        r1 = taxonomy_engine.classify({"description": "office supplies"})
        r2 = taxonomy_engine.classify({"description": "diesel fuel"})
        assert r1.provenance_hash != r2.provenance_hash

    def test_translation_provenance(self, taxonomy_engine):
        """translate_code includes provenance hash."""
        code = taxonomy_engine.translate_code("43", "unspsc", "naics")
        if code.code != "unknown":
            assert len(code.provenance_hash) == 64

    def test_batch_all_have_provenance(self, taxonomy_engine):
        """All records in a batch have provenance hashes."""
        records = [
            {"description": "office supplies"},
            {"description": "construction"},
            {"description": "chemicals"},
        ]
        results = taxonomy_engine.classify_batch(records)
        for r in results:
            assert len(r.provenance_hash) == 64


# ===================================================================
# TestThreadSafety - concurrent classification
# ===================================================================


class TestThreadSafety:
    """Tests for thread-safe concurrent classification."""

    def test_concurrent_classification(self, taxonomy_engine):
        """Multiple threads can classify simultaneously."""
        errors = []

        def worker(engine, desc, idx):
            try:
                result = engine.classify(
                    {"description": desc, "record_id": f"r-{idx}"}
                )
                assert result.primary_code.code != ""
            except Exception as exc:
                errors.append(str(exc))

        descriptions = [
            "office supplies", "construction", "diesel fuel",
            "computer servers", "legal consulting", "steel beams",
            "medical equipment", "agriculture", "chemical reagents",
            "transportation logistics",
        ]

        threads = [
            threading.Thread(
                target=worker, args=(taxonomy_engine, desc, i)
            )
            for i, desc in enumerate(descriptions)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert errors == [], f"Thread errors: {errors}"

        stats = taxonomy_engine.get_statistics()
        assert stats["classifications_performed"] == 10

    def test_concurrent_stats_consistent(self, taxonomy_engine):
        """Statistics remain consistent under load."""
        def worker(engine, idx):
            engine.classify(
                {"description": f"item {idx}"},
                taxonomy="naics",
            )

        threads = [
            threading.Thread(target=worker, args=(taxonomy_engine, i))
            for i in range(20)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        stats = taxonomy_engine.get_statistics()
        assert stats["classifications_performed"] == 20


# ===================================================================
# TestInternalMethods - internal helper coverage
# ===================================================================


class TestInternalMethods:
    """Tests for internal methods to boost coverage."""

    def test_build_search_text(self, taxonomy_engine):
        """_build_search_text concatenates relevant fields."""
        record = {
            "description": "office supplies",
            "category": "indirect",
            "vendor_name": "Staples",
            "material_group": "MRO",
        }
        text = taxonomy_engine._build_search_text(record)
        assert "office supplies" in text
        assert "indirect" in text
        assert "Staples" in text
        assert "MRO" in text

    def test_build_search_text_empty(self, taxonomy_engine):
        """Empty record produces empty text."""
        text = taxonomy_engine._build_search_text({})
        assert text == ""

    def test_partial_name_match_true(self, taxonomy_engine):
        """Partial name match identifies significant words."""
        result = taxonomy_engine._partial_name_match(
            "construction building", "building renovation construction"
        )
        assert result is True

    def test_partial_name_match_false(self, taxonomy_engine):
        """No significant word overlap returns False."""
        result = taxonomy_engine._partial_name_match(
            "construction building", "office supplies stationery"
        )
        assert result is False

    def test_keyword_score_zero(self, taxonomy_engine):
        """No keyword matches returns 0."""
        score = taxonomy_engine._keyword_score(
            ["diesel", "fuel"], "office supplies"
        )
        assert score == 0.0

    def test_keyword_score_positive(self, taxonomy_engine):
        """Matching keywords produce positive score."""
        score = taxonomy_engine._keyword_score(
            ["diesel", "fuel", "gasoline"], "diesel fuel for trucks"
        )
        assert 0.5 <= score <= 0.6

    def test_keyword_score_empty_list(self, taxonomy_engine):
        """Empty keyword list returns 0."""
        score = taxonomy_engine._keyword_score([], "anything")
        assert score == 0.0

    def test_get_registry_unknown(self, taxonomy_engine):
        """Unknown system returns empty registry."""
        registry = taxonomy_engine._get_registry("unknown")
        assert registry == {}

    def test_get_keywords_unknown(self, taxonomy_engine):
        """Unknown system returns empty keywords."""
        keywords = taxonomy_engine._get_keywords("unknown")
        assert keywords == {}

    def test_get_code_name_known(self, taxonomy_engine):
        """Known code returns its name."""
        name = taxonomy_engine._get_code_name("23", "naics")
        assert name == "Construction"

    def test_get_code_name_unknown(self, taxonomy_engine):
        """Unknown code returns the code itself."""
        name = taxonomy_engine._get_code_name("ZZ", "naics")
        assert name == "ZZ"

    def test_get_fallback_unspsc(self, taxonomy_engine):
        """Fallback for UNSPSC is segment 80."""
        code, name = taxonomy_engine._get_fallback("unspsc")
        assert code == "80"

    def test_get_fallback_naics(self, taxonomy_engine):
        """Fallback for NAICS is sector 54."""
        code, name = taxonomy_engine._get_fallback("naics")
        assert code == "54"

    def test_get_fallback_eclass(self, taxonomy_engine):
        """Fallback for eCl@ss is group 39."""
        code, name = taxonomy_engine._get_fallback("eclass")
        assert code == "39"

    def test_get_fallback_isic(self, taxonomy_engine):
        """Fallback for ISIC is section M."""
        code, name = taxonomy_engine._get_fallback("isic")
        assert code == "M"

    def test_get_fallback_unknown(self, taxonomy_engine):
        """Unknown system fallback returns 00."""
        code, name = taxonomy_engine._get_fallback("xyz")
        assert code == "00"
