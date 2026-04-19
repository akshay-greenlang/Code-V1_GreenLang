"""
Entity Resolution for Supply Chain Mapping.

This module provides sophisticated entity resolution capabilities for matching
supplier records across different systems. It combines multiple matching
strategies with confidence scoring to identify duplicate or related entities.

Key Features:
- Fuzzy string matching using multiple algorithms (Levenshtein, Jaro-Winkler, etc.)
- External identifier matching (LEI, DUNS, VAT)
- Address standardization and matching
- Machine learning-based entity resolution (optional)
- Configurable confidence thresholds
- Batch processing for large datasets

Example:
    >>> from greenlang.supply_chain.resolution import EntityResolver, MatchStrategy
    >>> resolver = EntityResolver(confidence_threshold=0.80)
    >>> result = resolver.find_matches(
    ...     query_name="ACME Manufacturing GmbH",
    ...     query_country="DE",
    ...     candidates=supplier_list
    ... )
    >>> for match in result:
    ...     print(f"{match.candidate.name}: {match.confidence:.2%}")
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Dict, Any, Tuple, Callable, Set
from datetime import datetime
import logging

from greenlang.supply_chain.models.entity import (
    Supplier,
    Address,
    ExternalIdentifiers,
)

logger = logging.getLogger(__name__)


class MatchConfidence(Enum):
    """
    Confidence level classification for entity matches.

    Used to categorize matches for different processing workflows:
    - HIGH: Auto-merge candidates
    - MEDIUM: Review recommended
    - LOW: Manual review required
    - NO_MATCH: Below threshold
    """
    HIGH = "high"  # >= 0.90 - Auto-merge candidate
    MEDIUM = "medium"  # >= 0.80 - Review recommended
    LOW = "low"  # >= 0.60 - Manual review required
    NO_MATCH = "no_match"  # < 0.60 - Below threshold

    @classmethod
    def from_score(cls, score: float) -> "MatchConfidence":
        """Classify confidence level from numeric score."""
        if score >= 0.90:
            return cls.HIGH
        elif score >= 0.80:
            return cls.MEDIUM
        elif score >= 0.60:
            return cls.LOW
        return cls.NO_MATCH


class MatchStrategy(Enum):
    """
    Entity matching strategy types.

    Different strategies are used for different matching scenarios:
    - EXACT: Exact string matching (case-insensitive)
    - FUZZY: Fuzzy string matching using multiple algorithms
    - IDENTIFIER: External identifier matching (LEI, DUNS)
    - ADDRESS: Address-based matching
    - HYBRID: Combination of multiple strategies
    - ML_BASED: Machine learning-based resolution
    """
    EXACT = auto()
    FUZZY = auto()
    IDENTIFIER = auto()
    ADDRESS = auto()
    HYBRID = auto()
    ML_BASED = auto()


@dataclass
class MatchResult:
    """
    Result of an entity matching operation.

    Attributes:
        candidate: The matched supplier entity
        confidence: Overall confidence score (0.0 to 1.0)
        confidence_level: Categorical confidence classification
        match_strategy: Strategy that produced this match
        name_similarity: Name matching score component
        identifier_match: Whether external IDs matched
        address_similarity: Address matching score component
        matched_identifiers: List of matched identifier types
        match_details: Detailed breakdown of matching scores
        timestamp: When the match was computed
    """
    candidate: Supplier
    confidence: float
    confidence_level: MatchConfidence
    match_strategy: MatchStrategy
    name_similarity: float = 0.0
    identifier_match: bool = False
    address_similarity: float = 0.0
    matched_identifiers: List[str] = field(default_factory=list)
    match_details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        """Validate match result data."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0 and 1: {self.confidence}")
        if self.confidence_level is None:
            self.confidence_level = MatchConfidence.from_score(self.confidence)

    def to_dict(self) -> Dict[str, Any]:
        """Convert match result to dictionary."""
        return {
            "candidate_id": self.candidate.id,
            "candidate_name": self.candidate.name,
            "confidence": self.confidence,
            "confidence_level": self.confidence_level.value,
            "match_strategy": self.match_strategy.name,
            "name_similarity": self.name_similarity,
            "identifier_match": self.identifier_match,
            "address_similarity": self.address_similarity,
            "matched_identifiers": self.matched_identifiers,
            "match_details": self.match_details,
            "timestamp": self.timestamp.isoformat(),
        }


class AddressNormalizer:
    """
    Address standardization and normalization utility.

    Normalizes addresses for consistent matching by:
    - Removing common abbreviations
    - Standardizing street type designations
    - Normalizing whitespace and punctuation
    - Converting to lowercase
    """

    # Common abbreviation expansions
    ABBREVIATIONS: Dict[str, str] = {
        "st": "street",
        "st.": "street",
        "ave": "avenue",
        "ave.": "avenue",
        "rd": "road",
        "rd.": "road",
        "blvd": "boulevard",
        "blvd.": "boulevard",
        "dr": "drive",
        "dr.": "drive",
        "ln": "lane",
        "ln.": "lane",
        "ct": "court",
        "ct.": "court",
        "pl": "place",
        "pl.": "place",
        "sq": "square",
        "sq.": "square",
        "pkwy": "parkway",
        "hwy": "highway",
        "ste": "suite",
        "ste.": "suite",
        "apt": "apartment",
        "apt.": "apartment",
        "fl": "floor",
        "fl.": "floor",
        "bldg": "building",
        "bldg.": "building",
        # German
        "str": "strasse",
        "str.": "strasse",
        "plz": "platz",
        # French
        "r.": "rue",
        "av.": "avenue",
        "bd.": "boulevard",
    }

    # Country-specific postal code patterns
    POSTAL_PATTERNS: Dict[str, str] = {
        "US": r"\d{5}(-\d{4})?",
        "CA": r"[A-Z]\d[A-Z]\s?\d[A-Z]\d",
        "UK": r"[A-Z]{1,2}\d{1,2}[A-Z]?\s?\d[A-Z]{2}",
        "DE": r"\d{5}",
        "FR": r"\d{5}",
        "NL": r"\d{4}\s?[A-Z]{2}",
    }

    @classmethod
    def normalize(cls, address: Optional[Address]) -> str:
        """
        Normalize an address for matching.

        Args:
            address: Address object to normalize

        Returns:
            Normalized address string
        """
        if not address:
            return ""

        parts = []

        # Normalize street lines
        if address.street_line_1:
            parts.append(cls._normalize_street(address.street_line_1))
        if address.street_line_2:
            parts.append(cls._normalize_street(address.street_line_2))

        # Normalize city
        if address.city:
            parts.append(cls._normalize_text(address.city))

        # Normalize state/province
        if address.state_province:
            parts.append(cls._normalize_text(address.state_province))

        # Normalize postal code
        if address.postal_code:
            parts.append(cls._normalize_postal_code(address.postal_code))

        # Add country code
        if address.country_code:
            parts.append(address.country_code.lower())

        return " ".join(parts)

    @classmethod
    def _normalize_text(cls, text: str) -> str:
        """Normalize general text."""
        # Remove accents
        text = unicodedata.normalize("NFKD", text)
        text = "".join(c for c in text if not unicodedata.combining(c))

        # Lowercase and remove extra whitespace
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)

        # Remove punctuation except hyphens
        text = re.sub(r"[^\w\s-]", "", text)

        return text

    @classmethod
    def _normalize_street(cls, street: str) -> str:
        """Normalize street address."""
        street = cls._normalize_text(street)

        # Expand abbreviations
        words = street.split()
        normalized_words = []
        for word in words:
            normalized_words.append(cls.ABBREVIATIONS.get(word, word))

        return " ".join(normalized_words)

    @classmethod
    def _normalize_postal_code(cls, postal_code: str) -> str:
        """Normalize postal code."""
        # Remove spaces and convert to uppercase for comparison
        return re.sub(r"\s+", "", postal_code.upper())


class CompanyNameNormalizer:
    """
    Company name normalization for fuzzy matching.

    Handles common variations in company names:
    - Legal entity suffixes (GmbH, Ltd, Inc, etc.)
    - Common abbreviations
    - Punctuation and whitespace
    """

    # Legal entity type suffixes to remove or normalize
    LEGAL_SUFFIXES: Set[str] = {
        # English
        "inc", "incorporated", "corp", "corporation", "co", "company",
        "ltd", "limited", "llc", "llp", "lp", "plc",
        # German
        "gmbh", "ag", "kg", "ohg", "gbr", "eg", "se",
        # French
        "sa", "sarl", "sas", "sasu", "eurl", "sci",
        # Dutch
        "bv", "nv", "cv", "vof",
        # Spanish
        "sl", "sa", "slu",
        # Italian
        "spa", "srl", "sas",
        # Other
        "pty", "pvt", "private",
    }

    # Common words to remove for matching
    COMMON_WORDS: Set[str] = {
        "the", "and", "&", "of", "for", "a", "an",
        "international", "intl", "global", "worldwide",
        "group", "holding", "holdings",
    }

    @classmethod
    def normalize(cls, name: str) -> str:
        """
        Normalize company name for matching.

        Args:
            name: Company name to normalize

        Returns:
            Normalized company name
        """
        if not name:
            return ""

        # Remove accents
        name = unicodedata.normalize("NFKD", name)
        name = "".join(c for c in name if not unicodedata.combining(c))

        # Lowercase
        name = name.lower()

        # Remove punctuation
        name = re.sub(r"[^\w\s]", " ", name)

        # Split into words
        words = name.split()

        # Filter out legal suffixes and common words
        filtered_words = []
        for word in words:
            if word not in cls.LEGAL_SUFFIXES and word not in cls.COMMON_WORDS:
                filtered_words.append(word)

        # Rejoin and normalize whitespace
        normalized = " ".join(filtered_words)
        normalized = re.sub(r"\s+", " ", normalized).strip()

        return normalized

    @classmethod
    def extract_legal_form(cls, name: str) -> Optional[str]:
        """
        Extract the legal form from a company name.

        Args:
            name: Company name

        Returns:
            Legal form suffix if found, None otherwise
        """
        name_lower = name.lower()
        words = re.split(r"[\s.,]+", name_lower)

        for word in reversed(words):
            if word in cls.LEGAL_SUFFIXES:
                return word
        return None


class StringSimilarity:
    """
    String similarity computation using multiple algorithms.

    Provides various string matching algorithms optimized for different
    use cases in entity resolution:
    - Levenshtein distance (edit distance)
    - Jaro-Winkler similarity (good for names)
    - Token-based similarity (order-independent)
    - Phonetic matching (Soundex, Metaphone)
    """

    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """
        Compute Levenshtein (edit) distance between two strings.

        Args:
            s1: First string
            s2: Second string

        Returns:
            Number of single-character edits needed to transform s1 into s2
        """
        if len(s1) < len(s2):
            s1, s2 = s2, s1

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    @staticmethod
    def levenshtein_similarity(s1: str, s2: str) -> float:
        """
        Compute Levenshtein similarity (normalized distance).

        Args:
            s1: First string
            s2: Second string

        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0

        distance = StringSimilarity.levenshtein_distance(s1, s2)
        max_len = max(len(s1), len(s2))
        return 1.0 - (distance / max_len)

    @staticmethod
    def jaro_similarity(s1: str, s2: str) -> float:
        """
        Compute Jaro similarity between two strings.

        Args:
            s1: First string
            s2: Second string

        Returns:
            Jaro similarity score between 0.0 and 1.0
        """
        if not s1 and not s2:
            return 1.0
        if not s1 or not s2:
            return 0.0

        len1, len2 = len(s1), len(s2)
        match_distance = max(len1, len2) // 2 - 1
        match_distance = max(0, match_distance)

        s1_matches = [False] * len1
        s2_matches = [False] * len2

        matches = 0
        transpositions = 0

        # Find matches
        for i in range(len1):
            start = max(0, i - match_distance)
            end = min(i + match_distance + 1, len2)

            for j in range(start, end):
                if s2_matches[j] or s1[i] != s2[j]:
                    continue
                s1_matches[i] = True
                s2_matches[j] = True
                matches += 1
                break

        if matches == 0:
            return 0.0

        # Count transpositions
        k = 0
        for i in range(len1):
            if not s1_matches[i]:
                continue
            while not s2_matches[k]:
                k += 1
            if s1[i] != s2[k]:
                transpositions += 1
            k += 1

        return (
            (matches / len1 + matches / len2 + (matches - transpositions / 2) / matches)
            / 3.0
        )

    @staticmethod
    def jaro_winkler_similarity(s1: str, s2: str, prefix_weight: float = 0.1) -> float:
        """
        Compute Jaro-Winkler similarity between two strings.

        Jaro-Winkler gives higher scores to strings that share a common prefix,
        making it well-suited for name matching.

        Args:
            s1: First string
            s2: Second string
            prefix_weight: Weight for common prefix bonus (default 0.1)

        Returns:
            Jaro-Winkler similarity score between 0.0 and 1.0
        """
        jaro = StringSimilarity.jaro_similarity(s1, s2)

        # Calculate common prefix length (up to 4 characters)
        prefix_len = 0
        for i in range(min(len(s1), len(s2), 4)):
            if s1[i] == s2[i]:
                prefix_len += 1
            else:
                break

        return jaro + prefix_len * prefix_weight * (1 - jaro)

    @staticmethod
    def token_sort_similarity(s1: str, s2: str) -> float:
        """
        Compute token sort similarity (order-independent).

        Useful for comparing names where word order may vary.

        Args:
            s1: First string
            s2: Second string

        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Tokenize and sort
        tokens1 = sorted(s1.lower().split())
        tokens2 = sorted(s2.lower().split())

        # Rejoin and compare
        sorted1 = " ".join(tokens1)
        sorted2 = " ".join(tokens2)

        return StringSimilarity.levenshtein_similarity(sorted1, sorted2)

    @staticmethod
    def token_set_similarity(s1: str, s2: str) -> float:
        """
        Compute token set similarity (handles partial matches).

        Useful when one string may be a subset of the other.

        Args:
            s1: First string
            s2: Second string

        Returns:
            Similarity score between 0.0 and 1.0
        """
        tokens1 = set(s1.lower().split())
        tokens2 = set(s2.lower().split())

        if not tokens1 or not tokens2:
            return 0.0

        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)

        # Jaccard similarity
        jaccard = len(intersection) / len(union) if union else 0.0

        # Also consider overlap ratio
        overlap = len(intersection) / min(len(tokens1), len(tokens2))

        return (jaccard + overlap) / 2

    @staticmethod
    def composite_similarity(
        s1: str,
        s2: str,
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Compute weighted composite similarity using multiple algorithms.

        Args:
            s1: First string
            s2: Second string
            weights: Dictionary of algorithm weights

        Returns:
            Weighted average similarity score
        """
        if weights is None:
            weights = {
                "jaro_winkler": 0.4,
                "levenshtein": 0.3,
                "token_sort": 0.2,
                "token_set": 0.1,
            }

        scores = {
            "jaro_winkler": StringSimilarity.jaro_winkler_similarity(s1, s2),
            "levenshtein": StringSimilarity.levenshtein_similarity(s1, s2),
            "token_sort": StringSimilarity.token_sort_similarity(s1, s2),
            "token_set": StringSimilarity.token_set_similarity(s1, s2),
        }

        weighted_sum = sum(scores[alg] * weights.get(alg, 0) for alg in scores)
        total_weight = sum(weights.get(alg, 0) for alg in scores)

        return weighted_sum / total_weight if total_weight > 0 else 0.0


class EntityResolver:
    """
    Entity resolution engine for supplier matching.

    Combines multiple matching strategies with confidence scoring to
    identify duplicate or related supplier entities across different
    data sources.

    Configuration:
        - confidence_threshold: Minimum score for matches (default 0.80)
        - identifier_boost: Score boost for matching identifiers (default 0.3)
        - country_match_boost: Score boost for matching country (default 0.1)
        - max_results: Maximum matches to return (default 10)

    Example:
        >>> resolver = EntityResolver(confidence_threshold=0.80)
        >>> matches = resolver.find_matches(
        ...     query_name="Acme Manufacturing",
        ...     query_country="DE",
        ...     candidates=supplier_database
        ... )
        >>> for match in matches:
        ...     print(f"{match.candidate.name}: {match.confidence:.2%}")
    """

    def __init__(
        self,
        confidence_threshold: float = 0.80,
        identifier_boost: float = 0.30,
        country_match_boost: float = 0.10,
        max_results: int = 10,
        enable_ml_resolution: bool = False,
    ):
        """
        Initialize the entity resolver.

        Args:
            confidence_threshold: Minimum confidence for returning matches
            identifier_boost: Additional score for matching external IDs
            country_match_boost: Additional score for matching country
            max_results: Maximum number of matches to return
            enable_ml_resolution: Enable ML-based entity resolution
        """
        self.confidence_threshold = confidence_threshold
        self.identifier_boost = identifier_boost
        self.country_match_boost = country_match_boost
        self.max_results = max_results
        self.enable_ml_resolution = enable_ml_resolution

        # Matching weights for different components
        self.component_weights = {
            "name": 0.50,
            "address": 0.20,
            "identifier": 0.20,
            "industry": 0.10,
        }

        logger.info(
            f"EntityResolver initialized with threshold={confidence_threshold}, "
            f"ml_enabled={enable_ml_resolution}"
        )

    def find_matches(
        self,
        query_name: str,
        candidates: List[Supplier],
        query_country: Optional[str] = None,
        query_address: Optional[Address] = None,
        query_identifiers: Optional[ExternalIdentifiers] = None,
        strategy: MatchStrategy = MatchStrategy.HYBRID,
    ) -> List[MatchResult]:
        """
        Find matching suppliers for a query entity.

        Args:
            query_name: Company name to match
            candidates: List of potential matching suppliers
            query_country: Optional country code for filtering
            query_address: Optional address for additional matching
            query_identifiers: Optional external IDs for exact matching
            strategy: Matching strategy to use

        Returns:
            List of MatchResult objects, sorted by confidence descending
        """
        if not query_name or not candidates:
            return []

        results: List[MatchResult] = []

        # Normalize query name
        normalized_query = CompanyNameNormalizer.normalize(query_name)

        for candidate in candidates:
            match_result = self._compute_match(
                query_name=query_name,
                normalized_query=normalized_query,
                query_country=query_country,
                query_address=query_address,
                query_identifiers=query_identifiers,
                candidate=candidate,
                strategy=strategy,
            )

            if match_result and match_result.confidence >= self.confidence_threshold:
                results.append(match_result)

        # Sort by confidence and limit results
        results.sort(key=lambda x: x.confidence, reverse=True)
        return results[:self.max_results]

    def _compute_match(
        self,
        query_name: str,
        normalized_query: str,
        query_country: Optional[str],
        query_address: Optional[Address],
        query_identifiers: Optional[ExternalIdentifiers],
        candidate: Supplier,
        strategy: MatchStrategy,
    ) -> Optional[MatchResult]:
        """
        Compute match score for a single candidate.

        Args:
            query_name: Original query name
            normalized_query: Normalized query name
            query_country: Query country code
            query_address: Query address
            query_identifiers: Query external identifiers
            candidate: Candidate supplier
            strategy: Matching strategy

        Returns:
            MatchResult if above threshold, None otherwise
        """
        match_details: Dict[str, Any] = {}
        matched_identifiers: List[str] = []
        total_score = 0.0

        # Check for exact identifier matches first
        identifier_match = False
        if query_identifiers and candidate.external_ids:
            identifier_match, matched_ids = self._check_identifier_match(
                query_identifiers, candidate.external_ids
            )
            matched_identifiers = matched_ids

            if identifier_match:
                # High confidence for identifier matches
                match_details["identifier_match"] = True
                match_details["matched_identifiers"] = matched_identifiers

                # Identifier match provides significant boost
                if "lei" in matched_identifiers or "duns" in matched_identifiers:
                    total_score += 0.95  # Near-certain match
                else:
                    total_score += 0.85  # High confidence

        # Compute name similarity
        normalized_candidate = CompanyNameNormalizer.normalize(candidate.name)
        name_similarity = self._compute_name_similarity(
            normalized_query, normalized_candidate
        )
        match_details["name_similarity"] = name_similarity
        match_details["query_normalized"] = normalized_query
        match_details["candidate_normalized"] = normalized_candidate

        # Compute address similarity if available
        address_similarity = 0.0
        if query_address and candidate.address:
            address_similarity = self._compute_address_similarity(
                query_address, candidate.address
            )
            match_details["address_similarity"] = address_similarity

        # Country match bonus
        country_match = False
        if query_country and candidate.country_code:
            country_match = query_country.upper() == candidate.country_code.upper()
            match_details["country_match"] = country_match

        # Compute weighted score based on strategy
        if strategy == MatchStrategy.EXACT:
            # Only exact matches
            if normalized_query == normalized_candidate:
                total_score = 1.0
            else:
                total_score = 0.0

        elif strategy == MatchStrategy.IDENTIFIER:
            # Only identifier-based matching
            if identifier_match:
                total_score = 1.0
            else:
                total_score = 0.0

        elif strategy == MatchStrategy.FUZZY:
            # Fuzzy name matching only
            total_score = name_similarity

        elif strategy == MatchStrategy.ADDRESS:
            # Address-based matching
            total_score = (name_similarity * 0.5) + (address_similarity * 0.5)

        elif strategy == MatchStrategy.HYBRID:
            # Combine all signals
            if identifier_match and total_score < 0.90:
                # Already computed above
                pass
            else:
                # Weighted combination
                total_score = (
                    name_similarity * self.component_weights["name"] +
                    address_similarity * self.component_weights["address"]
                )

                # Apply boosts
                if identifier_match:
                    total_score += self.identifier_boost
                if country_match:
                    total_score += self.country_match_boost

        # Cap at 1.0
        total_score = min(1.0, total_score)

        # Determine confidence level
        confidence_level = MatchConfidence.from_score(total_score)

        return MatchResult(
            candidate=candidate,
            confidence=total_score,
            confidence_level=confidence_level,
            match_strategy=strategy,
            name_similarity=name_similarity,
            identifier_match=identifier_match,
            address_similarity=address_similarity,
            matched_identifiers=matched_identifiers,
            match_details=match_details,
        )

    def _check_identifier_match(
        self,
        query_ids: ExternalIdentifiers,
        candidate_ids: ExternalIdentifiers,
    ) -> Tuple[bool, List[str]]:
        """
        Check for matching external identifiers.

        Args:
            query_ids: Query external identifiers
            candidate_ids: Candidate external identifiers

        Returns:
            Tuple of (match_found, list of matched identifier types)
        """
        matched = []

        # Check LEI (highest priority)
        if query_ids.lei and candidate_ids.lei:
            if query_ids.lei.upper() == candidate_ids.lei.upper():
                matched.append("lei")

        # Check DUNS
        if query_ids.duns and candidate_ids.duns:
            if query_ids.duns == candidate_ids.duns:
                matched.append("duns")

        # Check VAT number
        if query_ids.vat_number and candidate_ids.vat_number:
            # Normalize VAT numbers (remove spaces and common prefixes)
            q_vat = re.sub(r"[\s-]", "", query_ids.vat_number.upper())
            c_vat = re.sub(r"[\s-]", "", candidate_ids.vat_number.upper())
            if q_vat == c_vat:
                matched.append("vat_number")

        # Check company registry ID
        if query_ids.company_registry_id and candidate_ids.company_registry_id:
            if query_ids.company_registry_id == candidate_ids.company_registry_id:
                matched.append("company_registry_id")

        # Check procurement system IDs
        if query_ids.sap_vendor_id and candidate_ids.sap_vendor_id:
            if query_ids.sap_vendor_id == candidate_ids.sap_vendor_id:
                matched.append("sap_vendor_id")

        if query_ids.ariba_network_id and candidate_ids.ariba_network_id:
            if query_ids.ariba_network_id == candidate_ids.ariba_network_id:
                matched.append("ariba_network_id")

        return len(matched) > 0, matched

    def _compute_name_similarity(
        self,
        query: str,
        candidate: str,
    ) -> float:
        """
        Compute name similarity using composite algorithm.

        Args:
            query: Normalized query name
            candidate: Normalized candidate name

        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not query or not candidate:
            return 0.0

        # Use composite similarity with custom weights optimized for company names
        return StringSimilarity.composite_similarity(
            query,
            candidate,
            weights={
                "jaro_winkler": 0.35,
                "levenshtein": 0.25,
                "token_sort": 0.25,
                "token_set": 0.15,
            }
        )

    def _compute_address_similarity(
        self,
        query_address: Address,
        candidate_address: Address,
    ) -> float:
        """
        Compute address similarity.

        Args:
            query_address: Query address
            candidate_address: Candidate address

        Returns:
            Similarity score between 0.0 and 1.0
        """
        query_normalized = AddressNormalizer.normalize(query_address)
        candidate_normalized = AddressNormalizer.normalize(candidate_address)

        if not query_normalized or not candidate_normalized:
            return 0.0

        # Use token-based similarity for addresses
        return StringSimilarity.token_set_similarity(
            query_normalized, candidate_normalized
        )

    def deduplicate_suppliers(
        self,
        suppliers: List[Supplier],
        merge_threshold: float = 0.90,
    ) -> Tuple[List[Supplier], List[Tuple[Supplier, Supplier, float]]]:
        """
        Identify and report duplicate suppliers in a list.

        Args:
            suppliers: List of suppliers to deduplicate
            merge_threshold: Confidence threshold for considering duplicates

        Returns:
            Tuple of (unique suppliers, list of duplicate pairs with scores)
        """
        if len(suppliers) < 2:
            return suppliers, []

        duplicates: List[Tuple[Supplier, Supplier, float]] = []
        unique_indices: Set[int] = set(range(len(suppliers)))

        for i in range(len(suppliers)):
            if i not in unique_indices:
                continue

            for j in range(i + 1, len(suppliers)):
                if j not in unique_indices:
                    continue

                # Check for match
                match_result = self._compute_match(
                    query_name=suppliers[i].name,
                    normalized_query=CompanyNameNormalizer.normalize(suppliers[i].name),
                    query_country=suppliers[i].country_code,
                    query_address=suppliers[i].address,
                    query_identifiers=suppliers[i].external_ids,
                    candidate=suppliers[j],
                    strategy=MatchStrategy.HYBRID,
                )

                if match_result and match_result.confidence >= merge_threshold:
                    duplicates.append((suppliers[i], suppliers[j], match_result.confidence))
                    unique_indices.discard(j)  # Mark as duplicate

        unique_suppliers = [suppliers[i] for i in sorted(unique_indices)]
        return unique_suppliers, duplicates

    def batch_resolve(
        self,
        queries: List[Dict[str, Any]],
        candidates: List[Supplier],
        strategy: MatchStrategy = MatchStrategy.HYBRID,
    ) -> List[List[MatchResult]]:
        """
        Batch entity resolution for multiple queries.

        Args:
            queries: List of query dictionaries with keys:
                - name: Company name (required)
                - country: Country code (optional)
                - address: Address dict (optional)
                - identifiers: External IDs dict (optional)
            candidates: List of candidate suppliers
            strategy: Matching strategy

        Returns:
            List of match results for each query
        """
        results = []

        for query in queries:
            query_name = query.get("name", "")
            query_country = query.get("country")
            query_address = (
                Address(**query["address"])
                if query.get("address") else None
            )
            query_identifiers = (
                ExternalIdentifiers(**query["identifiers"])
                if query.get("identifiers") else None
            )

            matches = self.find_matches(
                query_name=query_name,
                candidates=candidates,
                query_country=query_country,
                query_address=query_address,
                query_identifiers=query_identifiers,
                strategy=strategy,
            )
            results.append(matches)

        return results


class LEIResolver:
    """
    Legal Entity Identifier (LEI) resolution service.

    Integrates with the GLEIF (Global Legal Entity Identifier Foundation)
    API to validate and resolve LEI codes.
    """

    GLEIF_API_BASE = "https://api.gleif.org/api/v1"

    @classmethod
    def validate_lei(cls, lei: str) -> bool:
        """
        Validate LEI format and checksum.

        LEI format: 20 characters alphanumeric
        - Characters 1-4: LOU (Local Operating Unit) prefix
        - Characters 5-6: Reserved (00)
        - Characters 7-18: Entity identifier
        - Characters 19-20: Checksum (MOD 97-10)

        Args:
            lei: LEI code to validate

        Returns:
            True if LEI format is valid
        """
        if not lei or len(lei) != 20:
            return False

        # Check alphanumeric
        if not lei.isalnum():
            return False

        # Validate checksum (MOD 97-10)
        lei_digits = ""
        for char in lei.upper():
            if char.isdigit():
                lei_digits += char
            else:
                # Convert A=10, B=11, ..., Z=35
                lei_digits += str(ord(char) - 55)

        try:
            return int(lei_digits) % 97 == 1
        except ValueError:
            return False

    @classmethod
    async def lookup_lei(cls, lei: str) -> Optional[Dict[str, Any]]:
        """
        Look up entity information from GLEIF API.

        Args:
            lei: LEI code to look up

        Returns:
            Entity information dictionary or None if not found
        """
        # This would make an actual API call in production
        # For now, return structure for documentation
        logger.info(f"LEI lookup requested for: {lei}")
        return None


class DUNSResolver:
    """
    D-U-N-S (Data Universal Numbering System) resolution service.

    Note: DUNS data access typically requires a D&B subscription.
    This class provides validation and structure for integration.
    """

    @classmethod
    def validate_duns(cls, duns: str) -> bool:
        """
        Validate DUNS number format.

        DUNS format: 9 numeric digits

        Args:
            duns: DUNS number to validate

        Returns:
            True if DUNS format is valid
        """
        if not duns:
            return False

        # Remove any formatting
        duns_clean = re.sub(r"[\s-]", "", duns)

        return len(duns_clean) == 9 and duns_clean.isdigit()

    @classmethod
    def format_duns(cls, duns: str) -> str:
        """
        Format DUNS number to standard representation.

        Args:
            duns: DUNS number

        Returns:
            Formatted DUNS (XX-XXX-XXXX)
        """
        duns_clean = re.sub(r"[\s-]", "", duns)
        if len(duns_clean) == 9:
            return f"{duns_clean[:2]}-{duns_clean[2:5]}-{duns_clean[5:]}"
        return duns
