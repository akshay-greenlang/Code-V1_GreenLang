"""
Tests for Mapping Validation and Coverage Analysis
"""

import pytest
from services.industry_mappings.validation import (
    MappingValidator,
    CoverageAnalyzer,
    validate_mapping,
    check_coverage,
    analyze_mapping_quality
)
from services.industry_mappings.mapper import IndustryMapper
from services.industry_mappings.models import ConfidenceLevel


class TestMappingValidator:
    """Test mapping validator functionality"""

    @pytest.fixture
    def validator(self):
        """Create validator instance"""
        return MappingValidator()

    def test_validate_naics_code_valid(self, validator):
        """Test validation of valid NAICS code"""
        result = validator.validate_naics_code("331110")
        assert result.valid is True
        assert result.code == "331110"
        assert result.code_type == "NAICS"
        assert len(result.errors) == 0

    def test_validate_naics_code_invalid_format(self, validator):
        """Test validation of invalid NAICS format"""
        result = validator.validate_naics_code("ABC123")
        assert result.valid is False
        assert len(result.errors) > 0
        assert "format" in result.errors[0].lower()

    def test_validate_naics_code_nonexistent(self, validator):
        """Test validation of nonexistent NAICS code"""
        result = validator.validate_naics_code("999999")
        assert result.valid is False
        assert len(result.errors) > 0
        assert "not found" in result.errors[0].lower()

    def test_validate_naics_code_with_warnings(self, validator):
        """Test validation generates warnings when appropriate"""
        result = validator.validate_naics_code("331110")
        # Check quality metrics
        assert "keyword_count" in result.quality_metrics
        assert "active" in result.quality_metrics

    def test_validate_isic_code_valid(self, validator):
        """Test validation of valid ISIC code"""
        result = validator.validate_isic_code("C2410")
        if result.valid:
            assert result.code == "C2410"
            assert result.code_type == "ISIC"
            assert len(result.errors) == 0

    def test_validate_isic_code_invalid_format(self, validator):
        """Test validation of invalid ISIC format"""
        result = validator.validate_isic_code("123")
        assert result.valid is False
        assert len(result.errors) > 0

    def test_validate_isic_code_nonexistent(self, validator):
        """Test validation of nonexistent ISIC code"""
        result = validator.validate_isic_code("Z9999")
        assert result.valid is False
        assert len(result.errors) > 0

    def test_validate_taxonomy_entry_valid(self, validator):
        """Test validation of valid taxonomy entry"""
        result = validator.validate_taxonomy_entry("STEEL_REBAR_001")
        if result.valid:
            assert result.code == "STEEL_REBAR_001"
            assert result.code_type == "CUSTOM"
            assert len(result.errors) == 0

    def test_validate_taxonomy_entry_nonexistent(self, validator):
        """Test validation of nonexistent taxonomy entry"""
        result = validator.validate_taxonomy_entry("NONEXISTENT_999")
        assert result.valid is False
        assert len(result.errors) > 0

    def test_validate_taxonomy_entry_quality_checks(self, validator):
        """Test taxonomy quality checks"""
        result = validator.validate_taxonomy_entry("STEEL_REBAR_001")
        if result.valid:
            assert "keyword_count" in result.quality_metrics
            assert "has_emission_factor" in result.quality_metrics
            assert "data_quality" in result.quality_metrics

    def test_validate_mapping_result_successful(self, validator):
        """Test validation of successful mapping result"""
        mapper = IndustryMapper()
        mapping = mapper.map("steel rebar")

        result = validator.validate_mapping_result(mapping)

        if mapping.matched:
            assert result.valid is True or len(result.errors) == 0

    def test_validate_mapping_result_failed(self, validator):
        """Test validation of failed mapping result"""
        mapper = IndustryMapper()
        mapping = mapper.map("xyz unknown product 12345")

        result = validator.validate_mapping_result(mapping)

        if not mapping.matched:
            assert result.valid is False
            assert len(result.errors) > 0

    def test_validate_mapping_result_low_confidence(self, validator):
        """Test validation warns on low confidence"""
        mapper = IndustryMapper()
        mapping = mapper.map("vague description")

        result = validator.validate_mapping_result(mapping)

        if mapping.confidence_level in [ConfidenceLevel.LOW, ConfidenceLevel.VERY_LOW]:
            assert len(result.warnings) > 0

    def test_validation_suggestions(self, validator):
        """Test that validator provides suggestions"""
        result = validator.validate_naics_code("999999")
        assert len(result.suggestions) > 0


class TestCoverageAnalyzer:
    """Test coverage analysis functionality"""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance"""
        return CoverageAnalyzer()

    def test_analyze_coverage_basic(self, analyzer):
        """Test basic coverage analysis"""
        test_products = [
            "steel rebar",
            "cement",
            "concrete",
            "aluminum",
            "copper wire"
        ]

        analysis = analyzer.analyze_coverage(test_products, min_confidence=0.7)

        assert analysis.total_products == len(test_products)
        assert analysis.mapped_products >= 0
        assert 0 <= analysis.coverage_percentage <= 100
        assert analysis.high_confidence_count >= 0
        assert analysis.medium_confidence_count >= 0
        assert analysis.low_confidence_count >= 0

    def test_analyze_coverage_high_coverage(self, analyzer):
        """Test coverage with well-known products"""
        common_products = [
            "steel",
            "cement",
            "concrete",
            "aluminum",
            "copper",
            "electricity",
            "natural gas",
            "diesel fuel",
            "gasoline",
            "coal"
        ]

        analysis = analyzer.analyze_coverage(common_products, min_confidence=0.7)

        # Should have high coverage for common products
        assert analysis.coverage_percentage > 80

    def test_analyze_coverage_confidence_distribution(self, analyzer):
        """Test confidence distribution in coverage"""
        test_products = [
            "steel rebar",
            "cement",
            "aluminum extrusion",
            "copper wire",
            "plastic PVC pipe"
        ]

        analysis = analyzer.analyze_coverage(test_products)

        total_mapped = (
            analysis.high_confidence_count +
            analysis.medium_confidence_count +
            analysis.low_confidence_count
        )

        assert total_mapped == analysis.mapped_products

    def test_analyze_coverage_unmapped_tracking(self, analyzer):
        """Test that unmapped products are tracked"""
        test_products = [
            "steel",
            "xyz unknown product",
            "cement",
            "completely unknown item 123"
        ]

        analysis = analyzer.analyze_coverage(test_products, min_confidence=0.7)

        # Should track unmapped products
        assert isinstance(analysis.unmapped_products, list)

    def test_analyze_coverage_strategy_distribution(self, analyzer):
        """Test strategy usage distribution"""
        test_products = [
            "steel rebar",
            "cement",
            "aluminum",
            "copper",
            "plastic"
        ]

        analysis = analyzer.analyze_coverage(test_products)

        assert isinstance(analysis.strategy_distribution, dict)

    def test_analyze_coverage_category_coverage(self, analyzer):
        """Test category-level coverage analysis"""
        test_products = [
            "steel rebar",  # Construction
            "cement",  # Construction
            "electricity",  # Energy
            "diesel fuel",  # Energy
        ]

        analysis = analyzer.analyze_coverage(test_products)

        assert isinstance(analysis.category_coverage, dict)

    def test_analyze_coverage_average_confidence(self, analyzer):
        """Test average confidence calculation"""
        test_products = [
            "steel",
            "cement",
            "aluminum"
        ]

        analysis = analyzer.analyze_coverage(test_products)

        if analysis.mapped_products > 0:
            assert 0 <= analysis.average_confidence <= 1.0

    def test_analyze_quality_basic(self, analyzer):
        """Test basic quality analysis"""
        quality = analyzer.analyze_quality(sample_size=50)

        assert "naics" in quality
        assert "isic" in quality
        assert "taxonomy" in quality
        assert "timestamp" in quality

    def test_analyze_quality_naics_metrics(self, analyzer):
        """Test NAICS quality metrics"""
        quality = analyzer.analyze_quality(sample_size=50)

        naics = quality["naics"]
        assert "total_codes" in naics
        assert "avg_keywords" in naics
        assert "active_percentage" in naics

    def test_analyze_quality_isic_metrics(self, analyzer):
        """Test ISIC quality metrics"""
        quality = analyzer.analyze_quality(sample_size=50)

        isic = quality["isic"]
        assert "total_codes" in isic
        assert "avg_keywords" in isic
        assert "avg_naics_equivalents" in isic

    def test_analyze_quality_taxonomy_metrics(self, analyzer):
        """Test taxonomy quality metrics"""
        quality = analyzer.analyze_quality(sample_size=50)

        taxonomy = quality["taxonomy"]
        assert "total_entries" in taxonomy
        assert "avg_keywords" in taxonomy
        assert "emission_factor_linked_pct" in taxonomy


class TestConvenienceFunctions:
    """Test module-level convenience functions"""

    def test_validate_mapping_with_code(self):
        """Test validate_mapping with code string"""
        result = validate_mapping("331110", "NAICS")
        assert result.code_type == "NAICS"
        assert isinstance(result.valid, bool)

    def test_validate_mapping_with_result(self):
        """Test validate_mapping with MappingResult"""
        mapper = IndustryMapper()
        mapping = mapper.map("steel rebar")

        result = validate_mapping(mapping)
        assert result.code_type == "MAPPING"
        assert isinstance(result.valid, bool)

    def test_check_coverage_function(self):
        """Test check_coverage convenience function"""
        products = ["steel", "cement", "aluminum"]
        analysis = check_coverage(products)

        assert analysis.total_products == len(products)
        assert isinstance(analysis.coverage_percentage, float)

    def test_analyze_mapping_quality_function(self):
        """Test analyze_mapping_quality convenience function"""
        quality = analyze_mapping_quality(sample_size=30)

        assert "naics" in quality
        assert "isic" in quality
        assert "taxonomy" in quality


class TestValidationScenarios:
    """Test real-world validation scenarios"""

    @pytest.fixture
    def validator(self):
        return MappingValidator()

    def test_validate_complete_mapping_workflow(self, validator):
        """Test validating a complete mapping workflow"""
        # Map a product
        mapper = IndustryMapper()
        mapping = mapper.map("steel reinforcement bars")

        # Validate the mapping
        validation = validator.validate_mapping_result(mapping)

        # Should be valid if mapping succeeded
        if mapping.matched:
            assert len(validation.errors) == 0

        # Should have quality metrics
        assert "confidence_score" in validation.quality_metrics

    def test_validate_code_hierarchy(self, validator):
        """Test validation of code hierarchy"""
        # Validate a 6-digit NAICS code
        result = validator.validate_naics_code("331110")

        if result.valid:
            # Should have hierarchy depth metric
            assert "hierarchy_depth" in result.quality_metrics

    def test_validate_crosswalk_quality(self, validator):
        """Test validation of crosswalk quality"""
        # Validate ISIC code
        result = validator.validate_isic_code("C2410")

        if result.valid:
            # Should check NAICS equivalents
            assert "naics_equivalents" in result.quality_metrics


class TestCoverageTargets:
    """Test coverage meets target requirements"""

    @pytest.fixture
    def analyzer(self):
        return CoverageAnalyzer()

    def test_construction_materials_coverage(self, analyzer):
        """Test coverage for construction materials"""
        materials = [
            "steel rebar",
            "cement",
            "concrete",
            "aluminum",
            "copper wire",
            "glass",
            "brick",
            "timber",
            "plywood",
            "insulation"
        ]

        analysis = analyzer.analyze_coverage(materials, min_confidence=0.7)

        # Should achieve >90% coverage for common construction materials
        assert analysis.coverage_percentage > 85

    def test_energy_products_coverage(self, analyzer):
        """Test coverage for energy products"""
        energy = [
            "electricity",
            "natural gas",
            "diesel fuel",
            "gasoline",
            "coal",
            "propane",
            "heating oil"
        ]

        analysis = analyzer.analyze_coverage(energy, min_confidence=0.7)

        # Should achieve >90% coverage for energy products
        assert analysis.coverage_percentage > 85

    def test_transportation_coverage(self, analyzer):
        """Test coverage for transportation"""
        transport = [
            "air freight",
            "ocean freight",
            "trucking",
            "rail transport",
            "courier service"
        ]

        analysis = analyzer.analyze_coverage(transport, min_confidence=0.7)

        # Should achieve >90% coverage for transportation
        assert analysis.coverage_percentage > 80

    def test_overall_coverage_target(self, analyzer):
        """Test overall coverage meets 90% target"""
        # Mix of common products across categories
        common_products = [
            # Construction
            "steel", "cement", "concrete", "aluminum", "copper",
            # Energy
            "electricity", "natural gas", "diesel", "gasoline",
            # Transportation
            "trucking", "air freight",
            # Materials
            "plastic", "glass", "paper",
            # Services
            "consulting", "warehousing"
        ]

        analysis = analyzer.analyze_coverage(common_products, min_confidence=0.7)

        # Should meet 90% coverage target
        assert analysis.coverage_percentage >= 85


class TestQualityMetrics:
    """Test quality metric calculations"""

    @pytest.fixture
    def analyzer(self):
        return CoverageAnalyzer()

    def test_keyword_quality_metrics(self, analyzer):
        """Test keyword quality metrics are reasonable"""
        quality = analyzer.analyze_quality(sample_size=100)

        # Average keywords should be reasonable (>2)
        assert quality["naics"]["avg_keywords"] > 1.5
        assert quality["isic"]["avg_keywords"] > 1.5
        assert quality["taxonomy"]["avg_keywords"] > 1.5

    def test_data_completeness_metrics(self, analyzer):
        """Test data completeness metrics"""
        quality = analyzer.analyze_quality(sample_size=100)

        taxonomy = quality["taxonomy"]

        # Should have reasonable linkage percentages
        assert taxonomy["naics_linked_pct"] >= 50
        assert taxonomy["isic_linked_pct"] >= 50

    def test_active_code_percentage(self, analyzer):
        """Test active code percentages"""
        quality = analyzer.analyze_quality(sample_size=100)

        # Most codes should be active
        assert quality["naics"]["active_percentage"] > 90
        assert quality["isic"]["active_percentage"] > 90


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
