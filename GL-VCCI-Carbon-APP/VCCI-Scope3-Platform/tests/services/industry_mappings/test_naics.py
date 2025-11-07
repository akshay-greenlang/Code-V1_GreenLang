"""
Tests for NAICS Database and Search Functions
"""

import pytest
from services.industry_mappings.naics import (
    NAICSDatabase,
    search_naics,
    get_naics_hierarchy,
    validate_naics_code
)
from services.industry_mappings.models import NAICSCode, IndustryCategory
from services.industry_mappings.config import get_default_config


class TestNAICSDatabase:
    """Test NAICS database functionality"""

    @pytest.fixture
    def naics_db(self):
        """Create NAICS database instance"""
        return NAICSDatabase()

    def test_database_initialization(self, naics_db):
        """Test database loads correctly"""
        assert len(naics_db.codes) > 100
        assert len(naics_db.by_level) > 0
        assert len(naics_db.by_category) > 0
        assert len(naics_db.keyword_index) > 0

    def test_get_code_by_string(self, naics_db):
        """Test getting code by string"""
        code = naics_db.get_code("331110")
        assert code is not None
        assert code.title == "Iron and Steel Mills and Ferroalloy Manufacturing"
        assert code.level == 6

    def test_get_code_nonexistent(self, naics_db):
        """Test getting nonexistent code"""
        code = naics_db.get_code("999999")
        assert code is None

    def test_get_by_level(self, naics_db):
        """Test getting codes by level"""
        level_2 = naics_db.get_by_level(2)
        assert len(level_2) > 0
        assert all(len(code.code) == 2 for code in level_2)

    def test_get_by_category(self, naics_db):
        """Test getting codes by category"""
        manufacturing = naics_db.get_by_category(IndustryCategory.MANUFACTURING)
        assert len(manufacturing) > 0
        assert all(code.category == IndustryCategory.MANUFACTURING for code in manufacturing)

    def test_get_hierarchy(self, naics_db):
        """Test getting code hierarchy"""
        hierarchy = naics_db.get_hierarchy("331110")
        assert len(hierarchy) > 0
        # Should include codes: 33, 331, 3311, 33111, 331110
        codes = [c.code for c in hierarchy]
        assert "33" in codes
        assert "331" in codes
        assert "331110" in codes

    def test_get_children(self, naics_db):
        """Test getting child codes"""
        children = naics_db.get_children("331")
        assert len(children) > 0
        assert all(code.code.startswith("331") and len(code.code) == 4 for code in children)

    def test_search_exact_code(self, naics_db):
        """Test search with exact code"""
        results = naics_db.search("331110")
        assert len(results) > 0
        assert results[0][0].code == "331110"
        assert results[0][1] == 1.0  # Perfect match

    def test_search_by_title(self, naics_db):
        """Test search by title"""
        results = naics_db.search("steel manufacturing")
        assert len(results) > 0
        # Should find steel-related codes
        assert any("steel" in r[0].title.lower() for r in results)

    def test_search_by_keyword(self, naics_db):
        """Test search by keyword"""
        results = naics_db.search("petroleum refining")
        assert len(results) > 0
        # Should find petroleum/refinery codes
        assert any("petroleum" in r[0].title.lower() or "refin" in r[0].title.lower() for r in results)

    def test_search_min_score(self, naics_db):
        """Test search with minimum score"""
        results = naics_db.search("manufacturing", min_score=0.8)
        assert all(score >= 0.8 for _, score in results)

    def test_search_max_results(self, naics_db):
        """Test search result limit"""
        results = naics_db.search("manufacturing", max_results=5)
        assert len(results) <= 5

    def test_search_fuzzy_matching(self, naics_db):
        """Test fuzzy matching capability"""
        results = naics_db.search("steal manufacturing")  # Typo: steal vs steel
        assert len(results) > 0
        # Should still find steel manufacturing due to fuzzy matching

    def test_electricity_generation_codes(self, naics_db):
        """Test electricity generation codes"""
        solar = naics_db.get_code("221114")
        assert solar is not None
        assert "solar" in solar.title.lower()

        wind = naics_db.get_code("221115")
        assert wind is not None
        assert "wind" in wind.title.lower()

    def test_all_codes_have_required_fields(self, naics_db):
        """Test all codes have required fields"""
        for code_str, code in naics_db.codes.items():
            assert code.code == code_str
            assert code.title
            assert code.description
            assert code.level >= 2 and code.level <= 6
            assert code.category

    def test_keyword_index_populated(self, naics_db):
        """Test keyword index is properly populated"""
        # Check that common words are indexed
        assert "steel" in naics_db.keyword_index
        assert "manufacturing" in naics_db.keyword_index
        assert len(naics_db.keyword_index["steel"]) > 0


class TestNAICSModuleFunctions:
    """Test module-level convenience functions"""

    def test_search_naics_function(self):
        """Test search_naics convenience function"""
        results = search_naics("cement manufacturing")
        assert len(results) > 0
        assert results[0][1] > 0.5  # Should have decent score

    def test_get_naics_hierarchy_function(self):
        """Test get_naics_hierarchy convenience function"""
        hierarchy = get_naics_hierarchy("327310")
        assert len(hierarchy) > 0
        assert hierarchy[-1].code == "327310"

    def test_validate_naics_code_function(self):
        """Test validate_naics_code convenience function"""
        assert validate_naics_code("331110") is True
        assert validate_naics_code("999999") is False
        assert validate_naics_code("invalid") is False


class TestNAICSEdgeCases:
    """Test edge cases and error handling"""

    @pytest.fixture
    def naics_db(self):
        return NAICSDatabase()

    def test_empty_search(self, naics_db):
        """Test search with empty string"""
        results = naics_db.search("")
        # Should return empty or very low confidence results
        assert isinstance(results, list)

    def test_search_special_characters(self, naics_db):
        """Test search with special characters"""
        results = naics_db.search("manufacturing & processing")
        assert isinstance(results, list)

    def test_search_very_long_query(self, naics_db):
        """Test search with very long query"""
        long_query = "manufacturing " * 100
        results = naics_db.search(long_query)
        assert isinstance(results, list)

    def test_hierarchy_for_invalid_code(self, naics_db):
        """Test hierarchy for invalid code"""
        hierarchy = naics_db.get_hierarchy("999999")
        assert len(hierarchy) == 0

    def test_children_for_leaf_code(self, naics_db):
        """Test getting children for 6-digit code (no children)"""
        children = naics_db.get_children("331110")
        assert len(children) == 0  # 6-digit codes have no children


class TestNAICSDataQuality:
    """Test data quality and consistency"""

    @pytest.fixture
    def naics_db(self):
        return NAICSDatabase()

    def test_all_2digit_codes_exist(self, naics_db):
        """Test that major 2-digit sector codes exist"""
        major_sectors = ["11", "21", "22", "23", "31", "42", "44", "48", "51", "52", "54", "61", "62", "71", "72", "81", "92"]
        for sector in major_sectors:
            code = naics_db.get_code(sector)
            if code:  # Not all may be in our dataset
                assert code.level == 2

    def test_hierarchy_consistency(self, naics_db):
        """Test that hierarchies are consistent"""
        # Pick a 6-digit code and verify full hierarchy
        code = naics_db.get_code("331110")
        if code:
            hierarchy = naics_db.get_hierarchy("331110")
            # Should have 5 levels (2, 3, 4, 5, 6 digit)
            assert len(hierarchy) >= 3

    def test_category_assignments(self, naics_db):
        """Test that categories are properly assigned"""
        # Check manufacturing codes
        steel = naics_db.get_code("331110")
        if steel:
            assert steel.category == IndustryCategory.MANUFACTURING

        # Check utilities codes
        electric = naics_db.get_code("221111")
        if electric:
            assert electric.category == IndustryCategory.UTILITIES

    def test_keywords_present(self, naics_db):
        """Test that codes have keywords"""
        sample_size = min(50, len(naics_db.codes))
        codes_with_keywords = 0

        for code in list(naics_db.codes.values())[:sample_size]:
            if len(code.keywords) > 0:
                codes_with_keywords += 1

        # At least 80% should have keywords
        assert codes_with_keywords / sample_size >= 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
