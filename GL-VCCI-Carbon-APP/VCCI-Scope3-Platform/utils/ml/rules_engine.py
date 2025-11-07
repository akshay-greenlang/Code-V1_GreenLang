# GL-VCCI ML Module - Rules Engine
# Spend Classification ML System - Rule-Based Classifier

"""
Rules Engine
============

Rule-based classification engine for Scope 3 spend categorization.

Provides fallback classification when LLM confidence is low, using:
- Keyword matching
- Regex patterns
- Fuzzy string matching
- Decision tree logic

Classification Methods:
----------------------
1. Exact keyword matching (e.g., "freight" → Category 4)
2. Regex pattern matching (e.g., r"\\b(flight|airfare)\\b" → Category 6)
3. Fuzzy matching (e.g., "elecricity" → "electricity" → Category 3)
4. Multi-keyword scoring (combine evidence from multiple keywords)

Usage:
------
```python
from utils.ml.rules_engine import RulesEngine
from utils.ml.config import MLConfig

# Initialize engine
config = MLConfig()
engine = RulesEngine(config)

# Classify spend
result = engine.classify("Office furniture purchase")
print(f"Category: {result.category}, Confidence: {result.confidence}")

# Get all matching rules
matches = engine.get_all_matches("Freight shipping to warehouse")
for category, score in matches:
    print(f"{category}: {score}")
```
"""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple

from fuzzywuzzy import fuzz
from pydantic import BaseModel, Field

from .config import MLConfig, RulesEngineConfig, Scope3Category
from .exceptions import RuleEvaluationException, RulesEngineException

logger = logging.getLogger(__name__)


# ============================================================================
# Response Models
# ============================================================================

class RuleMatch(BaseModel):
    """
    Rule match result.

    Attributes:
        category: Matched Scope 3 category
        confidence: Match confidence score (0.0-1.0)
        matched_keywords: Keywords that matched
        matched_patterns: Regex patterns that matched
        rule_type: Type of rule that matched (keyword, regex, fuzzy)
    """
    category: str = Field(description="Matched Scope 3 category")
    confidence: float = Field(ge=0.0, le=1.0, description="Match confidence")
    matched_keywords: List[str] = Field(default_factory=list, description="Matched keywords")
    matched_patterns: List[str] = Field(default_factory=list, description="Matched patterns")
    rule_type: str = Field(description="Rule type (keyword, regex, fuzzy)")


# ============================================================================
# Rule Definitions
# ============================================================================

# Category 1: Purchased Goods & Services
CATEGORY_1_KEYWORDS = [
    "material", "materials", "component", "components", "raw material", "supplies",
    "packaging", "office supplies", "stationary", "consumables", "parts",
    "ingredients", "chemicals", "paper", "toner", "ink"
]

CATEGORY_1_PATTERNS = [
    r"\b(raw\s+material|office\s+suppl(y|ies)|consumable)\b"
]

# Category 2: Capital Goods
CATEGORY_2_KEYWORDS = [
    "equipment", "machinery", "building", "facility", "construction",
    "computer", "laptop", "server", "hardware", "furniture", "vehicle",
    "truck", "forklift", "manufacturing equipment", "hvac", "renovation"
]

CATEGORY_2_PATTERNS = [
    r"\b(capital\s+equipment|manufacturing\s+equipment|IT\s+equipment)\b",
    r"\b(building|facility|construction|renovation)\b"
]

# Category 3: Fuel and Energy Related Activities
CATEGORY_3_KEYWORDS = [
    "electricity", "power", "energy", "fuel", "gas", "diesel", "gasoline",
    "natural gas", "heating", "cooling", "utility", "utilities", "kwh", "mwh",
    "electricity bill", "power bill", "energy bill"
]

CATEGORY_3_PATTERNS = [
    r"\b(electric(ity)?|power|energy|fuel|gas)\b",
    r"\b(kwh|mwh|kw|mw|utility|utilities)\b"
]

# Category 4: Upstream Transportation & Distribution
CATEGORY_4_KEYWORDS = [
    "freight", "shipping", "logistics", "transportation", "transport",
    "delivery", "courier", "fedex", "ups", "dhl", "trucking", "rail",
    "ocean freight", "air freight", "distribution", "warehousing"
]

CATEGORY_4_PATTERNS = [
    r"\b(freight|shipping|logistics|transport(ation)?)\b",
    r"\b(fedex|ups|dhl|courier|delivery)\b"
]

# Category 5: Waste Generated in Operations
CATEGORY_5_KEYWORDS = [
    "waste", "disposal", "recycling", "recycle", "trash", "garbage",
    "landfill", "waste management", "hazardous waste", "e-waste",
    "scrap", "waste removal", "waste collection"
]

CATEGORY_5_PATTERNS = [
    r"\b(waste|disposal|recycl(e|ing)|trash|garbage)\b",
    r"\b(landfill|waste\s+management|hazardous\s+waste)\b"
]

# Category 6: Business Travel
CATEGORY_6_KEYWORDS = [
    "flight", "airfare", "airline", "hotel", "accommodation", "lodging",
    "taxi", "uber", "lyft", "rental car", "car rental", "business travel",
    "conference", "meeting travel", "per diem", "travel expense"
]

CATEGORY_6_PATTERNS = [
    r"\b(flight|airfare|airline|air\s+travel)\b",
    r"\b(hotel|accommodat(ion|ions)|lodging)\b",
    r"\b(taxi|uber|lyft|ride\s*share)\b"
]

# Category 7: Employee Commuting
CATEGORY_7_KEYWORDS = [
    "commute", "commuting", "parking", "public transit", "metro", "subway",
    "bus pass", "train pass", "employee transportation", "shuttle",
    "carpool", "bike share", "parking permit"
]

CATEGORY_7_PATTERNS = [
    r"\b(commut(e|ing)|parking|employee\s+transport)\b",
    r"\b(public\s+transit|metro|subway|bus\s+pass)\b"
]

# Category 8: Upstream Leased Assets
CATEGORY_8_KEYWORDS = [
    "lease", "rental", "rent", "leased equipment", "leased facility",
    "office lease", "equipment rental", "operating lease"
]

CATEGORY_8_PATTERNS = [
    r"\b(lease|rental|rent|leased)\b",
    r"\b(operating\s+lease|office\s+lease)\b"
]

# Category 9: Downstream Transportation & Distribution
CATEGORY_9_KEYWORDS = [
    "customer delivery", "product shipping", "distribution to customer",
    "last mile", "product transport", "outbound logistics"
]

CATEGORY_9_PATTERNS = [
    r"\b(customer\s+delivery|product\s+shipping|outbound)\b",
    r"\b(last\s+mile|distribution\s+to\s+customer)\b"
]

# Category 10: Processing of Sold Products
CATEGORY_10_KEYWORDS = [
    "product processing", "intermediate processing", "downstream processing",
    "product manufacturing", "assembly by customer"
]

CATEGORY_10_PATTERNS = [
    r"\b(product\s+processing|downstream\s+processing)\b"
]

# Category 11: Use of Sold Products
CATEGORY_11_KEYWORDS = [
    "product use", "customer use", "end use", "product operation",
    "product consumption", "product usage"
]

CATEGORY_11_PATTERNS = [
    r"\b(product\s+use|customer\s+use|end\s+use)\b"
]

# Category 12: End-of-Life Treatment of Sold Products
CATEGORY_12_KEYWORDS = [
    "product disposal", "end of life", "product recycling", "product waste",
    "eol", "product decommission", "take-back program"
]

CATEGORY_12_PATTERNS = [
    r"\b(end[\s-]*of[\s-]*life|eol|product\s+disposal)\b",
    r"\b(take[\s-]*back|product\s+recycl(e|ing))\b"
]

# Category 13: Downstream Leased Assets
CATEGORY_13_KEYWORDS = [
    "leased to customer", "asset lease out", "equipment lease out",
    "downstream lease", "customer lease"
]

CATEGORY_13_PATTERNS = [
    r"\b(downstream\s+lease|lease\s+(to|out)|customer\s+lease)\b"
]

# Category 14: Franchises
CATEGORY_14_KEYWORDS = [
    "franchise", "franchisee", "franchise operation", "franchise fees",
    "royalty", "franchise agreement"
]

CATEGORY_14_PATTERNS = [
    r"\b(franchise(e|s)?|royalt(y|ies))\b"
]

# Category 15: Investments
CATEGORY_15_KEYWORDS = [
    "investment", "portfolio", "equity", "shares", "stock", "bond",
    "securities", "financial investment", "fund", "asset management"
]

CATEGORY_15_PATTERNS = [
    r"\b(investment|portfolio|equity|shares|stock|bond)\b",
    r"\b(securit(y|ies)|financial\s+investment)\b"
]


# ============================================================================
# Rules Engine
# ============================================================================

class RulesEngine:
    """
    Rule-based classification engine for Scope 3 categorization.

    Uses keyword matching, regex patterns, and fuzzy matching to classify
    procurement spend when LLM confidence is low.
    """

    def __init__(self, config: MLConfig):
        """
        Initialize rules engine.

        Args:
            config: ML configuration
        """
        self.config = config
        self.rules_config = config.rules_engine

        # Build rule database
        self.keyword_rules = self._build_keyword_rules()
        self.pattern_rules = self._build_pattern_rules()

        logger.info(
            f"Initialized rules engine: {len(self.keyword_rules)} keyword rules, "
            f"{len(self.pattern_rules)} pattern rules"
        )

    def classify(self, description: str) -> RuleMatch:
        """
        Classify spend description using rule-based approach.

        Args:
            description: Procurement description

        Returns:
            Rule match result with highest confidence

        Raises:
            RulesEngineException: If classification fails
        """
        try:
            # Get all matches
            all_matches = self._get_all_rule_matches(description)

            if not all_matches:
                # No matches - return unknown
                return RuleMatch(
                    category="unknown",
                    confidence=0.0,
                    rule_type="none"
                )

            # Return match with highest confidence
            return max(all_matches, key=lambda m: m.confidence)

        except Exception as e:
            logger.error(f"Rule evaluation failed: {e}", exc_info=True)
            raise RuleEvaluationException(
                message=f"Failed to evaluate rules: {str(e)}",
                input_data={"description": description[:100]},
                original_error=e
            )

    def get_all_matches(
        self,
        description: str,
        min_confidence: float = 0.0
    ) -> List[Tuple[str, float]]:
        """
        Get all matching categories with confidence scores.

        Args:
            description: Procurement description
            min_confidence: Minimum confidence threshold

        Returns:
            List of (category, confidence) tuples sorted by confidence
        """
        matches = self._get_all_rule_matches(description)

        # Filter by minimum confidence
        filtered = [
            (m.category, m.confidence)
            for m in matches
            if m.confidence >= min_confidence
        ]

        # Sort by confidence (descending)
        return sorted(filtered, key=lambda x: x[1], reverse=True)

    def _get_all_rule_matches(self, description: str) -> List[RuleMatch]:
        """Get all rule matches for description."""
        matches = []

        # Normalize description
        desc_lower = description.lower() if not self.rules_config.keyword_case_sensitive else description

        # 1. Keyword matching
        if self.rules_config.enable_keyword_matching:
            keyword_matches = self._match_keywords(desc_lower)
            matches.extend(keyword_matches)

        # 2. Regex pattern matching
        if self.rules_config.enable_regex_patterns:
            pattern_matches = self._match_patterns(desc_lower)
            matches.extend(pattern_matches)

        # 3. Fuzzy matching (if enabled)
        if self.rules_config.enable_fuzzy_matching:
            fuzzy_matches = self._match_fuzzy(desc_lower)
            matches.extend(fuzzy_matches)

        # Aggregate matches by category (combine evidence)
        aggregated = self._aggregate_matches(matches)

        return aggregated

    def _match_keywords(self, description: str) -> List[RuleMatch]:
        """Match keywords against description."""
        matches = []

        for category, keywords in self.keyword_rules.items():
            matched_keywords = []
            for keyword in keywords:
                if keyword.lower() in description:
                    matched_keywords.append(keyword)

            if matched_keywords:
                # Confidence based on number of matched keywords
                # Base: 0.6, +0.05 per additional keyword (max 0.95)
                confidence = min(0.6 + (len(matched_keywords) - 1) * 0.05, 0.95)

                matches.append(RuleMatch(
                    category=category,
                    confidence=confidence,
                    matched_keywords=matched_keywords,
                    rule_type="keyword"
                ))

        return matches

    def _match_patterns(self, description: str) -> List[RuleMatch]:
        """Match regex patterns against description."""
        matches = []

        for category, patterns in self.pattern_rules.items():
            matched_patterns = []
            for pattern in patterns:
                if re.search(pattern, description, re.IGNORECASE):
                    matched_patterns.append(pattern)

            if matched_patterns:
                # Confidence based on pattern matches
                # Base: 0.7, +0.05 per additional pattern (max 0.95)
                confidence = min(0.7 + (len(matched_patterns) - 1) * 0.05, 0.95)

                matches.append(RuleMatch(
                    category=category,
                    confidence=confidence,
                    matched_patterns=matched_patterns,
                    rule_type="regex"
                ))

        return matches

    def _match_fuzzy(self, description: str) -> List[RuleMatch]:
        """Fuzzy match keywords against description."""
        matches = []

        # Extract words from description
        desc_words = set(re.findall(r'\b\w+\b', description.lower()))

        for category, keywords in self.keyword_rules.items():
            best_fuzzy_score = 0
            matched_keywords = []

            for keyword in keywords:
                keyword_words = set(keyword.lower().split())

                # Check fuzzy match for each word
                for desc_word in desc_words:
                    for keyword_word in keyword_words:
                        score = fuzz.ratio(desc_word, keyword_word) / 100.0

                        if score >= self.rules_config.fuzzy_threshold:
                            best_fuzzy_score = max(best_fuzzy_score, score)
                            matched_keywords.append(f"{keyword} (fuzzy)")

            if best_fuzzy_score >= self.rules_config.fuzzy_threshold:
                # Confidence = fuzzy score * 0.8 (penalize fuzzy matches)
                confidence = min(best_fuzzy_score * 0.8, 0.85)

                matches.append(RuleMatch(
                    category=category,
                    confidence=confidence,
                    matched_keywords=list(set(matched_keywords)),
                    rule_type="fuzzy"
                ))

        return matches

    def _aggregate_matches(self, matches: List[RuleMatch]) -> List[RuleMatch]:
        """
        Aggregate multiple matches per category.

        Combines evidence from different rule types (keyword, regex, fuzzy)
        to produce final confidence score.
        """
        category_matches: Dict[str, List[RuleMatch]] = {}

        # Group by category
        for match in matches:
            if match.category not in category_matches:
                category_matches[match.category] = []
            category_matches[match.category].append(match)

        # Aggregate
        aggregated = []
        for category, cat_matches in category_matches.items():
            # Combine confidence scores (take weighted average)
            # Weights: regex=0.4, keyword=0.35, fuzzy=0.25
            weights = {"regex": 0.4, "keyword": 0.35, "fuzzy": 0.25}

            total_confidence = 0.0
            total_weight = 0.0

            all_keywords = []
            all_patterns = []

            for match in cat_matches:
                weight = weights.get(match.rule_type, 0.3)
                total_confidence += match.confidence * weight
                total_weight += weight

                all_keywords.extend(match.matched_keywords)
                all_patterns.extend(match.matched_patterns)

            # Final confidence
            final_confidence = total_confidence / total_weight if total_weight > 0 else 0.0

            # Boost confidence if multiple rule types matched
            if len(cat_matches) > 1:
                final_confidence = min(final_confidence * 1.15, 0.95)

            aggregated.append(RuleMatch(
                category=category,
                confidence=final_confidence,
                matched_keywords=list(set(all_keywords)),
                matched_patterns=list(set(all_patterns)),
                rule_type="aggregated"
            ))

        return aggregated

    def _build_keyword_rules(self) -> Dict[str, List[str]]:
        """Build keyword-based rules for all categories."""
        return {
            Scope3Category.CATEGORY_1.value: CATEGORY_1_KEYWORDS,
            Scope3Category.CATEGORY_2.value: CATEGORY_2_KEYWORDS,
            Scope3Category.CATEGORY_3.value: CATEGORY_3_KEYWORDS,
            Scope3Category.CATEGORY_4.value: CATEGORY_4_KEYWORDS,
            Scope3Category.CATEGORY_5.value: CATEGORY_5_KEYWORDS,
            Scope3Category.CATEGORY_6.value: CATEGORY_6_KEYWORDS,
            Scope3Category.CATEGORY_7.value: CATEGORY_7_KEYWORDS,
            Scope3Category.CATEGORY_8.value: CATEGORY_8_KEYWORDS,
            Scope3Category.CATEGORY_9.value: CATEGORY_9_KEYWORDS,
            Scope3Category.CATEGORY_10.value: CATEGORY_10_KEYWORDS,
            Scope3Category.CATEGORY_11.value: CATEGORY_11_KEYWORDS,
            Scope3Category.CATEGORY_12.value: CATEGORY_12_KEYWORDS,
            Scope3Category.CATEGORY_13.value: CATEGORY_13_KEYWORDS,
            Scope3Category.CATEGORY_14.value: CATEGORY_14_KEYWORDS,
            Scope3Category.CATEGORY_15.value: CATEGORY_15_KEYWORDS,
        }

    def _build_pattern_rules(self) -> Dict[str, List[str]]:
        """Build regex pattern rules for all categories."""
        return {
            Scope3Category.CATEGORY_1.value: CATEGORY_1_PATTERNS,
            Scope3Category.CATEGORY_2.value: CATEGORY_2_PATTERNS,
            Scope3Category.CATEGORY_3.value: CATEGORY_3_PATTERNS,
            Scope3Category.CATEGORY_4.value: CATEGORY_4_PATTERNS,
            Scope3Category.CATEGORY_5.value: CATEGORY_5_PATTERNS,
            Scope3Category.CATEGORY_6.value: CATEGORY_6_PATTERNS,
            Scope3Category.CATEGORY_7.value: CATEGORY_7_PATTERNS,
            Scope3Category.CATEGORY_8.value: CATEGORY_8_PATTERNS,
            Scope3Category.CATEGORY_9.value: CATEGORY_9_PATTERNS,
            Scope3Category.CATEGORY_10.value: CATEGORY_10_PATTERNS,
            Scope3Category.CATEGORY_11.value: CATEGORY_11_PATTERNS,
            Scope3Category.CATEGORY_12.value: CATEGORY_12_PATTERNS,
            Scope3Category.CATEGORY_13.value: CATEGORY_13_PATTERNS,
            Scope3Category.CATEGORY_14.value: CATEGORY_14_PATTERNS,
            Scope3Category.CATEGORY_15.value: CATEGORY_15_PATTERNS,
        }
