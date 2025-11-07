"""
Mapping Validation and Quality Assurance

Validation rules, coverage analysis, and quality metrics for industry mappings.
"""

from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
from datetime import datetime
import re

from .models import (
    ValidationResult,
    CoverageAnalysis,
    MappingResult,
    ConfidenceLevel,
    IndustryCategory
)
from .naics import NAICSDatabase
from .isic import ISICDatabase
from .custom_taxonomy import CustomTaxonomy
from .mapper import IndustryMapper
from .config import IndustryMappingConfig, get_default_config


class MappingValidator:
    """Validator for industry mappings"""

    def __init__(self, config: Optional[IndustryMappingConfig] = None):
        """Initialize validator"""
        self.config = config or get_default_config()
        self.naics_db = NAICSDatabase(config)
        self.isic_db = ISICDatabase(config)
        self.taxonomy = CustomTaxonomy(config)

    def validate_naics_code(self, code: str) -> ValidationResult:
        """
        Validate a NAICS code

        Args:
            code: NAICS code to validate

        Returns:
            ValidationResult with validation details
        """
        errors = []
        warnings = []
        suggestions = []

        # Format validation
        if not re.match(r"^\d{2,6}$", code):
            errors.append(f"Invalid NAICS code format: {code}. Must be 2-6 digits.")
            return ValidationResult(
                valid=False,
                code=code,
                code_type="NAICS",
                errors=errors,
                warnings=warnings,
                suggestions=suggestions
            )

        # Existence validation
        naics_obj = self.naics_db.get_code(code)
        if not naics_obj:
            errors.append(f"NAICS code {code} not found in database")
            suggestions.append("Check if code is from a different NAICS year")
            suggestions.append("Verify code is not deprecated")

            return ValidationResult(
                valid=False,
                code=code,
                code_type="NAICS",
                errors=errors,
                warnings=warnings,
                suggestions=suggestions
            )

        # Active status check
        if not naics_obj.active:
            warnings.append(f"NAICS code {code} is marked as inactive")
            suggestions.append("Consider using a more current code")

        # Hierarchy validation
        hierarchy = self.naics_db.get_hierarchy(code)
        if len(hierarchy) < naics_obj.level:
            warnings.append("Incomplete hierarchy chain")

        # Level consistency
        if len(code) != naics_obj.level:
            warnings.append(f"Code length {len(code)} doesn't match level {naics_obj.level}")

        # Keyword quality check
        if len(naics_obj.keywords) < 3:
            warnings.append("Limited keywords - may affect matching quality")
            suggestions.append("Consider adding more keywords for better matches")

        return ValidationResult(
            valid=True,
            code=code,
            code_type="NAICS",
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            quality_metrics={
                "keyword_count": len(naics_obj.keywords),
                "example_count": len(naics_obj.examples),
                "hierarchy_depth": len(hierarchy),
                "active": naics_obj.active
            }
        )

    def validate_isic_code(self, code: str) -> ValidationResult:
        """
        Validate an ISIC code

        Args:
            code: ISIC code to validate

        Returns:
            ValidationResult with validation details
        """
        errors = []
        warnings = []
        suggestions = []

        # Format validation
        if not re.match(r"^[A-U]\d{0,3}$", code.upper()):
            errors.append(f"Invalid ISIC code format: {code}. Must be section letter + 0-3 digits.")
            return ValidationResult(
                valid=False,
                code=code,
                code_type="ISIC",
                errors=errors,
                warnings=warnings,
                suggestions=suggestions
            )

        # Existence validation
        isic_obj = self.isic_db.get_code(code.upper())
        if not isic_obj:
            errors.append(f"ISIC code {code} not found in database")
            suggestions.append("Check if code is from ISIC Rev 4")
            suggestions.append("Verify code spelling and format")

            return ValidationResult(
                valid=False,
                code=code,
                code_type="ISIC",
                errors=errors,
                warnings=warnings,
                suggestions=suggestions
            )

        # Active status check
        if not isic_obj.active:
            warnings.append(f"ISIC code {code} is marked as inactive")
            suggestions.append("Consider using a more current code")

        # Hierarchy validation
        hierarchy = self.isic_db.get_hierarchy(code.upper())
        if not hierarchy:
            warnings.append("No hierarchy found")

        # Crosswalk validation
        if not isic_obj.naics_equivalents:
            warnings.append("No NAICS equivalents defined - crosswalk may not work")

        # Keyword quality check
        if len(isic_obj.keywords) < 3:
            warnings.append("Limited keywords - may affect matching quality")

        return ValidationResult(
            valid=True,
            code=code.upper(),
            code_type="ISIC",
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            quality_metrics={
                "keyword_count": len(isic_obj.keywords),
                "example_count": len(isic_obj.examples),
                "naics_equivalents": len(isic_obj.naics_equivalents),
                "hierarchy_depth": len(hierarchy),
                "active": isic_obj.active
            }
        )

    def validate_taxonomy_entry(self, entry_id: str) -> ValidationResult:
        """
        Validate a taxonomy entry

        Args:
            entry_id: Taxonomy entry ID

        Returns:
            ValidationResult with validation details
        """
        errors = []
        warnings = []
        suggestions = []

        # Existence validation
        entry = self.taxonomy.get_entry(entry_id)
        if not entry:
            errors.append(f"Taxonomy entry {entry_id} not found")
            return ValidationResult(
                valid=False,
                code=entry_id,
                code_type="CUSTOM",
                errors=errors,
                warnings=warnings,
                suggestions=suggestions
            )

        # Active status
        if not entry.active:
            warnings.append("Entry is marked as inactive")

        # Completeness checks
        if not entry.naics_codes:
            warnings.append("No NAICS codes linked")
            suggestions.append("Link to relevant NAICS codes for better integration")

        if not entry.isic_codes:
            warnings.append("No ISIC codes linked")
            suggestions.append("Link to relevant ISIC codes for international use")

        if not entry.emission_factor_id:
            warnings.append("No emission factor linked")
            suggestions.append("Link emission factor for Scope 3 calculations")

        # Keyword quality
        if len(entry.keywords) < 3:
            warnings.append("Limited keywords")
            suggestions.append("Add more keywords to improve matching")

        if not entry.synonyms:
            warnings.append("No synonyms defined")
            suggestions.append("Add synonyms to catch variations")

        # Data quality check
        if entry.data_quality == "low":
            warnings.append("Data quality is marked as low")
            suggestions.append("Review and update data quality")

        # Unit validation
        valid_units = [
            "kg", "g", "t", "lb", "oz",
            "m3", "l", "ml", "gal",
            "kWh", "MWh", "GJ", "MJ",
            "km", "mi", "m",
            "unit", "item", "piece",
            "hour", "day", "month"
        ]
        if entry.unit not in valid_units:
            warnings.append(f"Unusual unit: {entry.unit}")

        return ValidationResult(
            valid=True,
            code=entry_id,
            code_type="CUSTOM",
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            quality_metrics={
                "keyword_count": len(entry.keywords),
                "synonym_count": len(entry.synonyms),
                "naics_links": len(entry.naics_codes),
                "isic_links": len(entry.isic_codes),
                "has_emission_factor": bool(entry.emission_factor_id),
                "data_quality": entry.data_quality,
                "active": entry.active
            }
        )

    def validate_mapping_result(self, result: MappingResult) -> ValidationResult:
        """
        Validate a mapping result

        Args:
            result: MappingResult to validate

        Returns:
            ValidationResult
        """
        errors = []
        warnings = []
        suggestions = []

        # Check if mapping succeeded
        if not result.matched:
            errors.append("Mapping failed - no match found")
            suggestions.append("Try rephrasing the input")
            suggestions.append("Add more specific keywords")
            return ValidationResult(
                valid=False,
                code=None,
                code_type="MAPPING",
                errors=errors,
                warnings=warnings,
                suggestions=suggestions
            )

        # Confidence checks
        if result.confidence_level == ConfidenceLevel.VERY_LOW:
            warnings.append("Very low confidence match")
            suggestions.append("Manual verification strongly recommended")
        elif result.confidence_level == ConfidenceLevel.LOW:
            warnings.append("Low confidence match")
            suggestions.append("Manual verification recommended")
        elif result.confidence_level == ConfidenceLevel.MEDIUM:
            suggestions.append("Medium confidence - review recommended")

        # Check for code validity
        if result.naics_code:
            naics_validation = self.validate_naics_code(result.naics_code)
            if not naics_validation.valid:
                errors.extend(naics_validation.errors)
            warnings.extend(naics_validation.warnings)

        if result.isic_code:
            isic_validation = self.validate_isic_code(result.isic_code)
            if not isic_validation.valid:
                errors.extend(isic_validation.errors)
            warnings.extend(isic_validation.warnings)

        if result.taxonomy_id:
            taxonomy_validation = self.validate_taxonomy_entry(result.taxonomy_id)
            if not taxonomy_validation.valid:
                errors.extend(taxonomy_validation.errors)
            warnings.extend(taxonomy_validation.warnings)

        # Check for alternatives
        if not result.alternative_matches and result.confidence_score < 0.9:
            warnings.append("No alternative matches found - limited options")

        # Performance check
        if result.processing_time_ms > 50:
            warnings.append(f"Slow query: {result.processing_time_ms}ms")

        return ValidationResult(
            valid=len(errors) == 0,
            code=result.naics_code or result.isic_code or result.taxonomy_id,
            code_type="MAPPING",
            errors=errors,
            warnings=warnings,
            suggestions=suggestions,
            quality_metrics={
                "confidence_score": result.confidence_score,
                "confidence_level": result.confidence_level.value,
                "has_alternatives": len(result.alternative_matches) > 0,
                "processing_time_ms": result.processing_time_ms,
                "keywords_matched": len(result.keywords_matched)
            }
        )


class CoverageAnalyzer:
    """Analyze mapping coverage and quality"""

    def __init__(self, config: Optional[IndustryMappingConfig] = None):
        """Initialize coverage analyzer"""
        self.config = config or get_default_config()
        self.mapper = IndustryMapper(config)
        self.validator = MappingValidator(config)

    def analyze_coverage(
        self,
        test_products: List[str],
        min_confidence: float = 0.7
    ) -> CoverageAnalysis:
        """
        Analyze mapping coverage for a list of products

        Args:
            test_products: List of product names to test
            min_confidence: Minimum confidence threshold for "mapped"

        Returns:
            CoverageAnalysis with detailed metrics
        """
        total = len(test_products)
        mapped = 0
        high_conf = 0
        medium_conf = 0
        low_conf = 0
        unmapped = []
        confidence_scores = []
        strategy_counts = defaultdict(int)
        category_coverage = defaultdict(lambda: {"total": 0, "mapped": 0})

        # Process each product
        for product in test_products:
            result = self.mapper.map(product, include_alternatives=False)

            if result.matched and result.confidence_score >= min_confidence:
                mapped += 1
                confidence_scores.append(result.confidence_score)

                # Count by confidence level
                if result.confidence_level == ConfidenceLevel.HIGH:
                    high_conf += 1
                elif result.confidence_level == ConfidenceLevel.MEDIUM:
                    medium_conf += 1
                elif result.confidence_level == ConfidenceLevel.LOW:
                    low_conf += 1

                # Track strategy usage
                strategy_counts[result.strategy_used.value] += 1

                # Track by category
                if result.category:
                    category_coverage[result.category.value]["mapped"] += 1
            else:
                unmapped.append(product)

            # Track category total
            if result.category:
                category_coverage[result.category.value]["total"] += 1

        # Calculate average confidence
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

        # Calculate category coverage percentages
        category_pct = {}
        for category, counts in category_coverage.items():
            if counts["total"] > 0:
                pct = (counts["mapped"] / counts["total"]) * 100
                category_pct[category] = round(pct, 2)

        return CoverageAnalysis(
            total_products=total,
            mapped_products=mapped,
            coverage_percentage=round((mapped / total) * 100, 2) if total > 0 else 0.0,
            high_confidence_count=high_conf,
            medium_confidence_count=medium_conf,
            low_confidence_count=low_conf,
            unmapped_products=unmapped[:100],  # Limit to 100 for display
            category_coverage=category_pct,
            strategy_distribution=dict(strategy_counts),
            average_confidence=round(avg_confidence, 3)
        )

    def analyze_quality(
        self,
        sample_size: int = 100
    ) -> Dict[str, any]:
        """
        Analyze overall mapping quality

        Args:
            sample_size: Number of entries to sample from each database

        Returns:
            Quality metrics dictionary
        """
        quality_metrics = {
            "naics": self._analyze_naics_quality(sample_size),
            "isic": self._analyze_isic_quality(sample_size),
            "taxonomy": self._analyze_taxonomy_quality(sample_size),
            "timestamp": datetime.utcnow().isoformat()
        }

        return quality_metrics

    def _analyze_naics_quality(self, sample_size: int) -> Dict[str, any]:
        """Analyze NAICS database quality"""
        naics_db = NAICSDatabase(self.config)
        total_codes = len(naics_db.codes)

        # Sample codes
        sample_codes = list(naics_db.codes.values())[:sample_size]

        keyword_counts = [len(code.keywords) for code in sample_codes]
        example_counts = [len(code.examples) for code in sample_codes]
        active_count = sum(1 for code in sample_codes if code.active)

        return {
            "total_codes": total_codes,
            "sample_size": len(sample_codes),
            "avg_keywords": round(sum(keyword_counts) / len(keyword_counts), 2) if keyword_counts else 0,
            "avg_examples": round(sum(example_counts) / len(example_counts), 2) if example_counts else 0,
            "active_percentage": round((active_count / len(sample_codes)) * 100, 2) if sample_codes else 0,
            "levels_distribution": self._get_level_distribution(naics_db.by_level)
        }

    def _analyze_isic_quality(self, sample_size: int) -> Dict[str, any]:
        """Analyze ISIC database quality"""
        isic_db = ISICDatabase(self.config)
        total_codes = len(isic_db.codes)

        sample_codes = list(isic_db.codes.values())[:sample_size]

        keyword_counts = [len(code.keywords) for code in sample_codes]
        example_counts = [len(code.examples) for code in sample_codes]
        naics_equiv_counts = [len(code.naics_equivalents) for code in sample_codes]
        active_count = sum(1 for code in sample_codes if code.active)

        return {
            "total_codes": total_codes,
            "sample_size": len(sample_codes),
            "avg_keywords": round(sum(keyword_counts) / len(keyword_counts), 2) if keyword_counts else 0,
            "avg_examples": round(sum(example_counts) / len(example_counts), 2) if example_counts else 0,
            "avg_naics_equivalents": round(sum(naics_equiv_counts) / len(naics_equiv_counts), 2) if naics_equiv_counts else 0,
            "active_percentage": round((active_count / len(sample_codes)) * 100, 2) if sample_codes else 0,
            "sections_distribution": self._get_section_distribution(isic_db.by_section)
        }

    def _analyze_taxonomy_quality(self, sample_size: int) -> Dict[str, any]:
        """Analyze custom taxonomy quality"""
        taxonomy = CustomTaxonomy(self.config)
        total_entries = len(taxonomy.entries)

        sample_entries = list(taxonomy.entries.values())[:sample_size]

        keyword_counts = [len(entry.keywords) for entry in sample_entries]
        synonym_counts = [len(entry.synonyms) for entry in sample_entries]
        has_naics = sum(1 for entry in sample_entries if entry.naics_codes)
        has_isic = sum(1 for entry in sample_entries if entry.isic_codes)
        has_ef = sum(1 for entry in sample_entries if entry.emission_factor_id)
        active_count = sum(1 for entry in sample_entries if entry.active)

        return {
            "total_entries": total_entries,
            "sample_size": len(sample_entries),
            "avg_keywords": round(sum(keyword_counts) / len(keyword_counts), 2) if keyword_counts else 0,
            "avg_synonyms": round(sum(synonym_counts) / len(synonym_counts), 2) if synonym_counts else 0,
            "naics_linked_pct": round((has_naics / len(sample_entries)) * 100, 2) if sample_entries else 0,
            "isic_linked_pct": round((has_isic / len(sample_entries)) * 100, 2) if sample_entries else 0,
            "emission_factor_linked_pct": round((has_ef / len(sample_entries)) * 100, 2) if sample_entries else 0,
            "active_percentage": round((active_count / len(sample_entries)) * 100, 2) if sample_entries else 0,
            "category_distribution": self._get_category_distribution(taxonomy.by_category)
        }

    def _get_level_distribution(self, by_level: Dict[int, List]) -> Dict[int, int]:
        """Get distribution of codes by level"""
        return {level: len(codes) for level, codes in by_level.items()}

    def _get_section_distribution(self, by_section: Dict[str, List]) -> Dict[str, int]:
        """Get distribution of codes by section"""
        return {section: len(codes) for section, codes in by_section.items()}

    def _get_category_distribution(self, by_category: Dict[str, List]) -> Dict[str, int]:
        """Get distribution by category"""
        return {category: len(entries) for category, entries in by_category.items()}


# Convenience functions
def validate_mapping(
    code_or_result: Union[str, MappingResult],
    code_type: str = "NAICS",
    config: Optional[IndustryMappingConfig] = None
) -> ValidationResult:
    """
    Validate a code or mapping result

    Args:
        code_or_result: Code string or MappingResult
        code_type: Type of code (NAICS, ISIC, CUSTOM)
        config: Configuration

    Returns:
        ValidationResult
    """
    validator = MappingValidator(config)

    if isinstance(code_or_result, MappingResult):
        return validator.validate_mapping_result(code_or_result)
    elif code_type == "NAICS":
        return validator.validate_naics_code(code_or_result)
    elif code_type == "ISIC":
        return validator.validate_isic_code(code_or_result)
    elif code_type == "CUSTOM":
        return validator.validate_taxonomy_entry(code_or_result)
    else:
        raise ValueError(f"Unknown code type: {code_type}")


def check_coverage(
    test_products: List[str],
    min_confidence: float = 0.7,
    config: Optional[IndustryMappingConfig] = None
) -> CoverageAnalysis:
    """
    Check mapping coverage for a product list

    Args:
        test_products: List of products to test
        min_confidence: Minimum confidence threshold
        config: Configuration

    Returns:
        CoverageAnalysis
    """
    analyzer = CoverageAnalyzer(config)
    return analyzer.analyze_coverage(test_products, min_confidence)


def analyze_mapping_quality(
    sample_size: int = 100,
    config: Optional[IndustryMappingConfig] = None
) -> Dict[str, any]:
    """
    Analyze overall mapping quality

    Args:
        sample_size: Sample size for analysis
        config: Configuration

    Returns:
        Quality metrics dictionary
    """
    analyzer = CoverageAnalyzer(config)
    return analyzer.analyze_quality(sample_size)
