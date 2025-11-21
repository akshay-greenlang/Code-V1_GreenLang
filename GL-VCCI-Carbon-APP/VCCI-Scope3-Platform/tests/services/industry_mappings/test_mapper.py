# -*- coding: utf-8 -*-
"""
Tests for Industry Mapping Engine
"""

import pytest
from services.industry_mappings.mapper import (
    IndustryMapper,
    MatchingEngine,
    create_mapper,
    quick_map
)
from services.industry_mappings.models import MappingStrategy, ConfidenceLevel
from services.industry_mappings.config import get_default_config


class TestMatchingEngine:
    """Test the core matching engine"""

    @pytest.fixture
    def engine(self):
        """Create matching engine instance"""
        return MatchingEngine()

    def test_exact_code_match_naics(self, engine):
        """Test exact NAICS code matching"""
        result = engine.exact_code_match("331110", "NAICS")
        assert result is not None
        assert result.code == "331110"

    def test_exact_code_match_isic(self, engine):
        """Test exact ISIC code matching"""
        result = engine.exact_code_match("C2410", "ISIC")
        assert result is not None
        assert result.section == "C"

    def test_exact_code_match_invalid(self, engine):
        """Test exact match with invalid code"""
        result = engine.exact_code_match("999999", "NAICS")
        assert result is None

    def test_keyword_search_all(self, engine):
        """Test keyword search across all databases"""
        results = engine.keyword_search("steel manufacturing", "all", max_results=10)
        assert len(results) > 0
        assert any("steel" in str(r[0]).lower() for r in results)

    def test_keyword_search_naics_only(self, engine):
        """Test keyword search in NAICS only"""
        results = engine.keyword_search("petroleum refining", "naics", max_results=5)
        assert len(results) > 0
        assert len(results) <= 5

    def test_keyword_search_taxonomy_only(self, engine):
        """Test keyword search in taxonomy only"""
        results = engine.keyword_search("cement", "taxonomy", max_results=5)
        assert len(results) > 0

    def test_fuzzy_match(self, engine):
        """Test fuzzy string matching"""
        candidates = ["steel manufacturing", "aluminum production", "copper smelting"]
        results = engine.fuzzy_match("steal manufactring", candidates, threshold=0.6)  # Typos
        assert len(results) > 0
        # Should match "steel manufacturing" despite typos

    def test_hierarchical_match(self, engine):
        """Test hierarchical code matching"""
        hierarchy = engine.hierarchical_match("331110", "NAICS")
        assert len(hierarchy) > 0
        codes = [c.code for c in hierarchy]
        assert "331110" in codes

    def test_crosswalk_match_naics_to_isic(self, engine):
        """Test NAICS to ISIC crosswalk"""
        results = engine.crosswalk_match("331110", "NAICS", "ISIC")
        # Should find ISIC equivalents
        assert isinstance(results, list)

    def test_crosswalk_match_isic_to_naics(self, engine):
        """Test ISIC to NAICS crosswalk"""
        results = engine.crosswalk_match("C2410", "ISIC", "NAICS")
        assert isinstance(results, list)


class TestIndustryMapper:
    """Test the main IndustryMapper class"""

    @pytest.fixture
    def mapper(self):
        """Create mapper instance"""
        return IndustryMapper()

    def test_map_simple_product(self, mapper):
        """Test mapping a simple product"""
        result = mapper.map("steel rebar")
        assert result.matched is True
        assert result.confidence_score > 0.5
        assert result.matched_title is not None

    def test_map_with_taxonomy_preference(self, mapper):
        """Test mapping with taxonomy preference"""
        result = mapper.map("portland cement", prefer_taxonomy=True)
        assert result.matched is True
        # Should prefer taxonomy entry over NAICS
        if result.taxonomy_id:
            assert "CEMENT" in result.taxonomy_id

    def test_map_with_alternatives(self, mapper):
        """Test mapping with alternative matches"""
        result = mapper.map("concrete", include_alternatives=True, max_alternatives=3)
        if result.matched and len(result.alternative_matches) > 0:
            assert len(result.alternative_matches) <= 3
            assert all("title" in alt for alt in result.alternative_matches)

    def test_map_electricity(self, mapper):
        """Test mapping electricity"""
        result = mapper.map("grid electricity")
        assert result.matched is True
        assert "electric" in result.matched_title.lower() or "power" in result.matched_title.lower()

    def test_map_transportation(self, mapper):
        """Test mapping transportation"""
        result = mapper.map("diesel fuel trucking")
        assert result.matched is True

    def test_map_services(self, mapper):
        """Test mapping services"""
        result = mapper.map("management consulting services")
        assert result.matched is True

    def test_map_complex_description(self, mapper):
        """Test mapping complex product description"""
        result = mapper.map("reinforced concrete ready-mix for building construction")
        assert result.matched is True
        assert result.confidence_score > 0.4

    def test_map_batch(self, mapper):
        """Test batch mapping"""
        products = [
            "steel rebar",
            "cement",
            "diesel fuel",
            "electricity",
            "aluminum"
        ]
        results = mapper.map_batch(products)
        assert len(results) == len(products)
        assert all(isinstance(r.matched, bool) for r in results)

    def test_get_by_code_naics(self, mapper):
        """Test getting entry by NAICS code"""
        code = mapper.get_by_code("331110", "NAICS")
        assert code is not None
        assert code.code == "331110"

    def test_get_by_code_isic(self, mapper):
        """Test getting entry by ISIC code"""
        code = mapper.get_by_code("C2410", "ISIC")
        if code:
            assert code.section == "C"

    def test_convert_code(self, mapper):
        """Test code conversion"""
        isic_codes = mapper.convert_code("331110", "NAICS", "ISIC")
        assert isinstance(isic_codes, list)

    def test_get_hierarchy(self, mapper):
        """Test getting code hierarchy"""
        hierarchy = mapper.get_hierarchy("331110", "NAICS")
        assert len(hierarchy) > 0

    def test_search_all(self, mapper):
        """Test searching all databases"""
        results = mapper.search("cement manufacturing", "all", max_results=5)
        assert len(results) > 0
        assert len(results) <= 5

    def test_suggest_codes(self, mapper):
        """Test code suggestion"""
        suggestions = mapper.suggest_codes("steel reinforcement bars", max_suggestions=5)
        assert len(suggestions) > 0
        assert all("confidence" in s for s in suggestions)
        assert all("title" in s for s in suggestions)

    def test_cache_functionality(self, mapper):
        """Test caching works"""
        # Clear cache
        mapper.clear_cache()

        # First call - cache miss
        result1 = mapper.map("steel rebar")
        stats1 = mapper.get_cache_stats()
        assert stats1["cache_misses"] >= 1

        # Second call - should hit cache
        result2 = mapper.map("steel rebar")
        stats2 = mapper.get_cache_stats()
        assert stats2["cache_hits"] >= 1

        # Results should be identical
        assert result1.confidence_score == result2.confidence_score

    def test_processing_time_recorded(self, mapper):
        """Test that processing time is recorded"""
        result = mapper.map("steel")
        assert result.processing_time_ms > 0
        assert result.processing_time_ms < 1000  # Should be fast

    def test_confidence_levels(self, mapper):
        """Test confidence level classification"""
        result = mapper.map("steel rebar manufacturing")
        assert result.confidence_level in [
            ConfidenceLevel.HIGH,
            ConfidenceLevel.MEDIUM,
            ConfidenceLevel.LOW,
            ConfidenceLevel.VERY_LOW
        ]

    def test_warnings_for_low_confidence(self, mapper):
        """Test warnings are added for low confidence"""
        result = mapper.map("some vague product description xyz123")
        if result.confidence_score < 0.7:
            assert len(result.warnings) > 0


class TestMappingStrategies:
    """Test different mapping strategies"""

    @pytest.fixture
    def mapper(self):
        return IndustryMapper()

    def test_exact_match_strategy(self, mapper):
        """Test exact matching works"""
        # Should find exact matches with high confidence
        result = mapper.map("portland cement")
        if result.matched:
            assert result.confidence_score > 0.7

    def test_keyword_strategy(self, mapper):
        """Test keyword matching"""
        result = mapper.map("steel manufacturing facility")
        assert result.matched is True
        assert result.strategy_used in [MappingStrategy.KEYWORD_SEARCH, MappingStrategy.FUZZY_MATCH]

    def test_fuzzy_strategy(self, mapper):
        """Test fuzzy matching handles typos"""
        result = mapper.map("steal manufactring")  # Typos
        # Should still find steel manufacturing
        assert result.matched is True


class TestEdgeCases:
    """Test edge cases and error handling"""

    @pytest.fixture
    def mapper(self):
        return IndustryMapper()

    def test_empty_input(self, mapper):
        """Test handling of empty input"""
        result = mapper.map("")
        assert isinstance(result.matched, bool)

    def test_very_long_input(self, mapper):
        """Test handling of very long input"""
        long_input = "steel " * 1000
        result = mapper.map(long_input)
        assert isinstance(result.matched, bool)

    def test_special_characters(self, mapper):
        """Test handling of special characters"""
        result = mapper.map("steel & aluminum manufacturing (USA) 2023")
        assert isinstance(result.matched, bool)

    def test_non_english_characters(self, mapper):
        """Test handling of non-English characters"""
        result = mapper.map("béton ciment")
        assert isinstance(result.matched, bool)

    def test_numbers_only(self, mapper):
        """Test handling of numbers only"""
        result = mapper.map("331110")
        # Should match as exact code
        assert result.matched is True
        assert result.naics_code == "331110"

    def test_unmapped_product(self, mapper):
        """Test handling of unmapped product"""
        result = mapper.map("xyz completely unknown product 12345")
        assert result.matched is False
        assert len(result.warnings) > 0


class TestPerformance:
    """Test performance requirements"""

    @pytest.fixture
    def mapper(self):
        return IndustryMapper()

    def test_single_mapping_performance(self, mapper):
        """Test single mapping meets performance requirement"""
        result = mapper.map("steel rebar")
        assert result.processing_time_ms < 50  # Should be well under 10ms average

    def test_batch_mapping_performance(self, mapper):
        """Test batch mapping performance"""
        import time
        products = ["steel", "cement", "aluminum", "copper", "plastic"] * 20  # 100 products

        start = time.time()
        results = mapper.map_batch(products)
        elapsed_ms = (time.time() - start) * 1000

        avg_time = elapsed_ms / len(products)
        assert avg_time < 50  # Average should be well under threshold


class TestConvenienceFunctions:
    """Test module-level convenience functions"""

    def test_create_mapper(self):
        """Test create_mapper function"""
        mapper = create_mapper()
        assert isinstance(mapper, IndustryMapper)

    def test_quick_map(self):
        """Test quick_map function"""
        result = quick_map("steel rebar")
        assert result.matched is True
        assert result.confidence_score > 0.5


class TestRealWorldScenarios:
    """Test real-world usage scenarios"""

    @pytest.fixture
    def mapper(self):
        return IndustryMapper()

    def test_construction_materials(self, mapper):
        """Test mapping construction materials"""
        materials = [
            "steel rebar",
            "concrete ready-mix",
            "cement",
            "aluminum window frames",
            "glass panels"
        ]

        for material in materials:
            result = mapper.map(material)
            assert result.matched is True, f"Failed to map: {material}"
            assert result.confidence_score > 0.5

    def test_energy_products(self, mapper):
        """Test mapping energy products"""
        energy = [
            "electricity",
            "natural gas",
            "diesel fuel",
            "gasoline",
            "coal"
        ]

        for product in energy:
            result = mapper.map(product)
            assert result.matched is True, f"Failed to map: {product}"

    def test_transportation_services(self, mapper):
        """Test mapping transportation services"""
        transport = [
            "air freight",
            "ocean freight",
            "trucking",
            "rail transport"
        ]

        for service in transport:
            result = mapper.map(service)
            assert result.matched is True, f"Failed to map: {service}"

    def test_manufactured_goods(self, mapper):
        """Test mapping manufactured goods"""
        goods = [
            "automobile",
            "computer",
            "plastic bottle",
            "steel pipe",
            "aluminum can"
        ]

        for good in goods:
            result = mapper.map(good)
            assert result.matched is True, f"Failed to map: {good}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
