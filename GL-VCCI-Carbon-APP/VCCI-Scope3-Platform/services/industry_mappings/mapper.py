# -*- coding: utf-8 -*-
"""
Industry Mapping Engine

Multi-strategy matching engine for automatic categorization of products and
services into industry codes with confidence scoring and ambiguity resolution.
"""

import time
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
import logging

from .models import (
    MappingResult,
    MappingStrategy,
    ConfidenceLevel,
    IndustryCategory,
    NAICSCode,
    ISICCode,
    TaxonomyEntry
)
from .naics import NAICSDatabase
from .isic import ISICDatabase
from .custom_taxonomy import CustomTaxonomy
from .config import IndustryMappingConfig, get_default_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import rapidfuzz
try:
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    from difflib import SequenceMatcher


class MatchingEngine:
    """Core matching engine with multiple strategies"""

    def __init__(self, config: Optional[IndustryMappingConfig] = None):
        """Initialize matching engine"""
        self.config = config or get_default_config()
        self.naics_db = NAICSDatabase(config)
        self.isic_db = ISICDatabase(config)
        self.taxonomy = CustomTaxonomy(config)

    def exact_code_match(
        self,
        code: str,
        code_type: str = "NAICS"
    ) -> Optional[Union[NAICSCode, ISICCode]]:
        """
        Strategy 1: Exact code matching

        Args:
            code: Industry code to match
            code_type: Type of code (NAICS, ISIC)

        Returns:
            Matched code object or None
        """
        if code_type == "NAICS":
            return self.naics_db.get_code(code)
        elif code_type == "ISIC":
            return self.isic_db.get_code(code.upper())
        return None

    def keyword_search(
        self,
        query: str,
        search_space: str = "all",
        max_results: int = 5
    ) -> List[Tuple[Union[NAICSCode, ISICCode, TaxonomyEntry], float]]:
        """
        Strategy 2: Keyword-based search

        Args:
            query: Search query
            search_space: Where to search (all, naics, isic, taxonomy)
            max_results: Maximum results to return

        Returns:
            List of (code/entry, score) tuples
        """
        results = []

        if search_space in ["all", "naics"]:
            naics_results = self.naics_db.search(
                query,
                max_results=max_results,
                min_score=self.config.match_thresholds.keyword_match
            )
            results.extend(naics_results)

        if search_space in ["all", "isic"]:
            isic_results = self.isic_db.search(
                query,
                max_results=max_results,
                min_score=self.config.match_thresholds.keyword_match
            )
            results.extend(isic_results)

        if search_space in ["all", "taxonomy"]:
            taxonomy_results = self.taxonomy.search(
                query,
                max_results=max_results,
                min_score=self.config.match_thresholds.keyword_match
            )
            results.extend(taxonomy_results)

        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_results]

    def fuzzy_match(
        self,
        query: str,
        candidates: List[str],
        threshold: float = 0.8
    ) -> List[Tuple[str, float]]:
        """
        Strategy 3: Fuzzy string matching

        Args:
            query: Query string
            candidates: List of candidate strings
            threshold: Minimum match threshold

        Returns:
            List of (candidate, score) tuples
        """
        if not RAPIDFUZZ_AVAILABLE:
            # Fallback to difflib
            results = []
            for candidate in candidates:
                score = SequenceMatcher(None, query.lower(), candidate.lower()).ratio()
                if score >= threshold:
                    results.append((candidate, score))
            results.sort(key=lambda x: x[1], reverse=True)
            return results

        # Use rapidfuzz for better performance
        matches = process.extract(
            query,
            candidates,
            scorer=fuzz.token_set_ratio,
            limit=None
        )

        # Filter by threshold and convert to 0-1 scale
        results = [(match[0], match[1] / 100.0) for match in matches if match[1] / 100.0 >= threshold]
        return results

    def hierarchical_match(
        self,
        code: str,
        code_type: str = "NAICS"
    ) -> List[Union[NAICSCode, ISICCode]]:
        """
        Strategy 4: Hierarchical code matching

        Get the full hierarchy for a code

        Args:
            code: Industry code
            code_type: Type of code (NAICS, ISIC)

        Returns:
            List of codes in hierarchy
        """
        if code_type == "NAICS":
            return self.naics_db.get_hierarchy(code)
        elif code_type == "ISIC":
            return self.isic_db.get_hierarchy(code.upper())
        return []

    def crosswalk_match(
        self,
        code: str,
        from_system: str = "NAICS",
        to_system: str = "ISIC"
    ) -> List[Union[NAICSCode, ISICCode]]:
        """
        Strategy 5: Cross-system code mapping

        Args:
            code: Code to convert
            from_system: Source classification system
            to_system: Target classification system

        Returns:
            List of equivalent codes
        """
        if from_system == "NAICS" and to_system == "ISIC":
            return self.isic_db.naics_to_isic(code)
        elif from_system == "ISIC" and to_system == "NAICS":
            naics_codes = self.isic_db.isic_to_naics(code.upper())
            return [self.naics_db.get_code(c) for c in naics_codes if self.naics_db.get_code(c)]
        return []


class IndustryMapper:
    """
    Main Industry Mapper class

    Provides high-level API for industry code mapping with automatic
    strategy selection and confidence scoring.
    """

    def __init__(self, config: Optional[IndustryMappingConfig] = None):
        """Initialize industry mapper"""
        self.config = config or get_default_config()
        self.engine = MatchingEngine(config)

        # Cache for performance
        self._cache: Dict[str, MappingResult] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def map(
        self,
        input_text: str,
        prefer_taxonomy: bool = True,
        include_alternatives: bool = True,
        max_alternatives: int = 5
    ) -> MappingResult:
        """
        Map input text to industry codes

        Main entry point for industry mapping. Automatically selects
        and applies the best matching strategy.

        Args:
            input_text: Text to map (product name, description, etc.)
            prefer_taxonomy: Prefer custom taxonomy over standard codes
            include_alternatives: Include alternative matches
            max_alternatives: Maximum alternative matches to return

        Returns:
            MappingResult with best match and alternatives
        """
        start_time = time.time()

        # Check cache
        if self.config.cache.enable_cache:
            cache_key = self._get_cache_key(input_text, prefer_taxonomy)
            if cache_key in self._cache:
                self._cache_hits += 1
                cached_result = self._cache[cache_key]
                logger.debug(f"Cache hit for: {input_text}")
                return cached_result
            self._cache_misses += 1

        # Initialize result
        result = MappingResult(
            input_text=input_text,
            matched=False,
            confidence_score=0.0,
            confidence_level=ConfidenceLevel.VERY_LOW,
            strategy_used=MappingStrategy.KEYWORD_SEARCH,
            processing_time_ms=0.0,
            warnings=[]
        )

        try:
            # Strategy selection and execution
            if prefer_taxonomy:
                # Try custom taxonomy first
                taxonomy_result = self._map_to_taxonomy(input_text)
                if taxonomy_result and taxonomy_result.confidence_score >= self.config.match_thresholds.medium_confidence:
                    result = taxonomy_result
                else:
                    # Fall back to standard codes
                    standard_result = self._map_to_standard_codes(input_text)
                    if standard_result.confidence_score > (taxonomy_result.confidence_score if taxonomy_result else 0):
                        result = standard_result
                    elif taxonomy_result:
                        result = taxonomy_result
            else:
                # Try standard codes first
                result = self._map_to_standard_codes(input_text)

            # Add alternative matches if requested
            if include_alternatives and result.matched:
                result.alternative_matches = self._get_alternatives(
                    input_text,
                    exclude_id=result.taxonomy_id or result.naics_code or result.isic_code,
                    max_alternatives=max_alternatives
                )

            # Add warnings
            if result.confidence_score < self.config.match_thresholds.medium_confidence:
                result.warnings.append("Low confidence match - manual verification recommended")

            if not result.matched:
                result.warnings.append("No suitable match found in database")

        except Exception as e:
            logger.error(f"Error mapping '{input_text}': {e}")
            result.warnings.append(f"Mapping error: {str(e)}")

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        result.processing_time_ms = round(processing_time, 2)

        # Check performance threshold
        if processing_time > self.config.performance.max_processing_time_ms:
            result.warnings.append(f"Slow query: {processing_time:.2f}ms")
            if self.config.performance.log_slow_queries:
                logger.warning(f"Slow query ({processing_time:.2f}ms): {input_text}")

        # Cache result
        if self.config.cache.enable_cache:
            self._cache[cache_key] = result
            # Simple cache size management
            if len(self._cache) > self.config.cache.max_cache_size:
                # Remove oldest 10% of entries
                remove_count = int(self.config.cache.max_cache_size * 0.1)
                for key in list(self._cache.keys())[:remove_count]:
                    del self._cache[key]

        return result

    def _map_to_taxonomy(self, input_text: str) -> Optional[MappingResult]:
        """Map to custom taxonomy"""
        taxonomy_results = self.engine.taxonomy.search(
            input_text,
            max_results=1,
            min_score=self.config.match_thresholds.minimum_match
        )

        if not taxonomy_results:
            return None

        entry, score = taxonomy_results[0]

        # Get matched keywords
        matched_keywords = self._extract_matched_keywords(input_text, entry.keywords + entry.synonyms)

        result = MappingResult(
            input_text=input_text,
            matched=True,
            taxonomy_id=entry.id,
            confidence_score=score,
            confidence_level=self._score_to_level(score),
            strategy_used=MappingStrategy.KEYWORD_SEARCH,
            matched_title=entry.name,
            keywords_matched=matched_keywords,
            processing_time_ms=0.0,
            metadata={
                "category": entry.category,
                "subcategory": entry.subcategory,
                "material_type": entry.material_type,
                "unit": entry.unit,
                "emission_factor_id": entry.emission_factor_id
            }
        )

        # Also populate NAICS/ISIC if available
        if entry.naics_codes:
            result.naics_code = entry.naics_codes[0]
        if entry.isic_codes:
            result.isic_code = entry.isic_codes[0]

        return result

    def _map_to_standard_codes(self, input_text: str) -> MappingResult:
        """Map to standard NAICS/ISIC codes"""
        # Try NAICS first
        naics_results = self.engine.naics_db.search(
            input_text,
            max_results=1,
            min_score=self.config.match_thresholds.minimum_match
        )

        if naics_results:
            code, score = naics_results[0]
            matched_keywords = self._extract_matched_keywords(input_text, code.keywords)

            result = MappingResult(
                input_text=input_text,
                matched=True,
                naics_code=code.code,
                confidence_score=score,
                confidence_level=self._score_to_level(score),
                strategy_used=MappingStrategy.KEYWORD_SEARCH,
                matched_title=code.title,
                category=code.category,
                keywords_matched=matched_keywords,
                processing_time_ms=0.0,
                metadata={
                    "code_level": code.level,
                    "parent_code": code.parent_code,
                    "examples": code.examples
                }
            )

            # Try to get ISIC equivalent
            if self.config.enable_crosswalk:
                isic_equivalents = self.engine.isic_db.naics_to_isic(code.code)
                if isic_equivalents:
                    result.isic_code = isic_equivalents[0].code

            return result

        # Fall back to ISIC
        isic_results = self.engine.isic_db.search(
            input_text,
            max_results=1,
            min_score=self.config.match_thresholds.minimum_match
        )

        if isic_results:
            code, score = isic_results[0]
            matched_keywords = self._extract_matched_keywords(input_text, code.keywords)

            result = MappingResult(
                input_text=input_text,
                matched=True,
                isic_code=code.code,
                confidence_score=score,
                confidence_level=self._score_to_level(score),
                strategy_used=MappingStrategy.KEYWORD_SEARCH,
                matched_title=code.title,
                category=code.category,
                keywords_matched=matched_keywords,
                processing_time_ms=0.0,
                metadata={
                    "section": code.section,
                    "division": code.division,
                    "examples": code.examples
                }
            )

            return result

        # No match found
        return MappingResult(
            input_text=input_text,
            matched=False,
            confidence_score=0.0,
            confidence_level=ConfidenceLevel.VERY_LOW,
            strategy_used=MappingStrategy.KEYWORD_SEARCH,
            processing_time_ms=0.0,
            warnings=["No matching codes found"]
        )

    def _get_alternatives(
        self,
        input_text: str,
        exclude_id: Optional[str] = None,
        max_alternatives: int = 5
    ) -> List[Dict[str, Union[str, float]]]:
        """Get alternative matches"""
        alternatives = []

        # Get from all sources
        all_results = self.engine.keyword_search(input_text, search_space="all", max_results=max_alternatives + 1)

        for item, score in all_results:
            # Skip if it's the excluded ID
            if exclude_id:
                item_id = None
                if isinstance(item, TaxonomyEntry):
                    item_id = item.id
                elif isinstance(item, NAICSCode):
                    item_id = item.code
                elif isinstance(item, ISICCode):
                    item_id = item.code

                if item_id == exclude_id:
                    continue

            # Add to alternatives
            if isinstance(item, TaxonomyEntry):
                alternatives.append({
                    "type": "taxonomy",
                    "id": item.id,
                    "title": item.name,
                    "score": score
                })
            elif isinstance(item, NAICSCode):
                alternatives.append({
                    "type": "naics",
                    "code": item.code,
                    "title": item.title,
                    "score": score
                })
            elif isinstance(item, ISICCode):
                alternatives.append({
                    "type": "isic",
                    "code": item.code,
                    "title": item.title,
                    "score": score
                })

            if len(alternatives) >= max_alternatives:
                break

        return alternatives

    def _extract_matched_keywords(self, input_text: str, keywords: List[str]) -> List[str]:
        """Extract keywords that matched from input text"""
        input_lower = input_text.lower()
        matched = []

        for keyword in keywords:
            if keyword.lower() in input_lower:
                matched.append(keyword)

        return matched

    def _score_to_level(self, score: float) -> ConfidenceLevel:
        """Convert score to confidence level"""
        if score >= self.config.match_thresholds.high_confidence:
            return ConfidenceLevel.HIGH
        elif score >= self.config.match_thresholds.medium_confidence:
            return ConfidenceLevel.MEDIUM
        elif score >= self.config.match_thresholds.low_confidence:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW

    def _get_cache_key(self, input_text: str, prefer_taxonomy: bool) -> str:
        """Generate cache key"""
        return f"{input_text.lower()}_{prefer_taxonomy}"

    def map_batch(
        self,
        inputs: List[str],
        prefer_taxonomy: bool = True,
        show_progress: bool = False
    ) -> List[MappingResult]:
        """
        Map multiple inputs in batch

        Args:
            inputs: List of input texts
            prefer_taxonomy: Prefer custom taxonomy
            show_progress: Show progress indicator

        Returns:
            List of mapping results
        """
        results = []

        for i, input_text in enumerate(inputs):
            if show_progress and i % 100 == 0:
                logger.info(f"Processing {i}/{len(inputs)}")

            result = self.map(input_text, prefer_taxonomy=prefer_taxonomy, include_alternatives=False)
            results.append(result)

        return results

    def get_cache_stats(self) -> Dict[str, Union[int, float]]:
        """Get cache statistics"""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0

        return {
            "cache_size": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
            "total_requests": total_requests
        }

    def clear_cache(self):
        """Clear the cache"""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("Cache cleared")

    def get_by_code(
        self,
        code: str,
        code_type: str = "NAICS"
    ) -> Optional[Union[NAICSCode, ISICCode, TaxonomyEntry]]:
        """
        Get entry by exact code

        Args:
            code: Code to lookup
            code_type: Type of code (NAICS, ISIC, TAXONOMY)

        Returns:
            Code object or None
        """
        if code_type == "NAICS":
            return self.engine.naics_db.get_code(code)
        elif code_type == "ISIC":
            return self.engine.isic_db.get_code(code.upper())
        elif code_type == "TAXONOMY":
            return self.engine.taxonomy.get_entry(code.upper())
        return None

    def convert_code(
        self,
        code: str,
        from_system: str = "NAICS",
        to_system: str = "ISIC"
    ) -> List[str]:
        """
        Convert between code systems

        Args:
            code: Code to convert
            from_system: Source system (NAICS, ISIC)
            to_system: Target system (NAICS, ISIC)

        Returns:
            List of equivalent codes
        """
        results = self.engine.crosswalk_match(code, from_system, to_system)
        return [r.code for r in results if hasattr(r, 'code')]

    def get_hierarchy(
        self,
        code: str,
        code_type: str = "NAICS"
    ) -> List[Union[NAICSCode, ISICCode]]:
        """
        Get code hierarchy

        Args:
            code: Code to get hierarchy for
            code_type: Type of code (NAICS, ISIC)

        Returns:
            List of codes in hierarchy
        """
        return self.engine.hierarchical_match(code, code_type)

    def search(
        self,
        query: str,
        search_space: str = "all",
        max_results: int = 10
    ) -> List[Tuple[Union[NAICSCode, ISICCode, TaxonomyEntry], float]]:
        """
        Search all databases

        Args:
            query: Search query
            search_space: Where to search (all, naics, isic, taxonomy)
            max_results: Maximum results

        Returns:
            List of (item, score) tuples
        """
        return self.engine.keyword_search(query, search_space, max_results)

    def suggest_codes(
        self,
        description: str,
        max_suggestions: int = 5
    ) -> List[Dict[str, Union[str, float]]]:
        """
        Suggest possible industry codes for a description

        Args:
            description: Product/service description
            max_suggestions: Maximum suggestions to return

        Returns:
            List of suggested codes with scores
        """
        result = self.map(
            description,
            prefer_taxonomy=True,
            include_alternatives=True,
            max_alternatives=max_suggestions - 1
        )

        suggestions = []

        # Add primary match
        if result.matched:
            primary = {
                "rank": 1,
                "confidence": result.confidence_score,
                "confidence_level": result.confidence_level.value,
                "title": result.matched_title
            }

            if result.taxonomy_id:
                primary["type"] = "taxonomy"
                primary["id"] = result.taxonomy_id
            elif result.naics_code:
                primary["type"] = "naics"
                primary["code"] = result.naics_code
            elif result.isic_code:
                primary["type"] = "isic"
                primary["code"] = result.isic_code

            suggestions.append(primary)

        # Add alternatives
        for i, alt in enumerate(result.alternative_matches[:max_suggestions - 1], 2):
            alt_suggestion = {
                "rank": i,
                "confidence": alt["score"],
                "confidence_level": self._score_to_level(alt["score"]).value,
                "type": alt["type"],
                "title": alt["title"]
            }

            if "id" in alt:
                alt_suggestion["id"] = alt["id"]
            if "code" in alt:
                alt_suggestion["code"] = alt["code"]

            suggestions.append(alt_suggestion)

        return suggestions


# Convenience functions
def create_mapper(config: Optional[IndustryMappingConfig] = None) -> IndustryMapper:
    """Create an industry mapper instance"""
    return IndustryMapper(config)


def quick_map(input_text: str, config: Optional[IndustryMappingConfig] = None) -> MappingResult:
    """Quick mapping function"""
    mapper = IndustryMapper(config)
    return mapper.map(input_text)
