# -*- coding: utf-8 -*-
"""
InvestmentScreenerBridge - SFDR Product Classification and Screening
=====================================================================

This module connects PACK-010 (SFDR Article 8) with the green investment
screener for SFDR product classification, investment screening against
exclusion lists, positive criteria evaluation, and binding element
enforcement.

Architecture:
    Portfolio Holdings --> InvestmentScreenerBridge --> Classification
                               |
                               v
    Exclusion Lists + Positive Criteria --> Pass/Fail per Holding

Example:
    >>> config = InvestmentScreenerBridgeConfig()
    >>> bridge = InvestmentScreenerBridge(config)
    >>> result = bridge.classify_product(holdings)
    >>> print(f"Classification: {result['classification']}")

Author: GreenLang Platform Team
Date: March 2026
Pack: PACK-010 SFDR Article 8
Version: 1.0.0
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Utility Helpers
# =============================================================================


def _utcnow() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(timezone.utc)


def _hash_data(data: Any) -> str:
    """Compute a SHA-256 hash of arbitrary data."""
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, default=str).encode()
    ).hexdigest()


# =============================================================================
# Agent Stub
# =============================================================================


class _AgentStub:
    """Deferred agent loader for lazy initialization."""

    def __init__(self, agent_id: str, module_path: str, class_name: str) -> None:
        self.agent_id = agent_id
        self.module_path = module_path
        self.class_name = class_name
        self._instance: Optional[Any] = None

    def load(self) -> Any:
        """Load and return the agent instance."""
        if self._instance is not None:
            return self._instance
        try:
            import importlib
            mod = importlib.import_module(self.module_path)
            cls = getattr(mod, self.class_name)
            self._instance = cls()
            return self._instance
        except Exception as exc:
            logger.warning(
                "AgentStub: failed to load %s: %s", self.agent_id, exc,
            )
            return None

    @property
    def is_loaded(self) -> bool:
        """Whether the agent has been loaded."""
        return self._instance is not None


# =============================================================================
# Enums
# =============================================================================


class SFDRClassification(str, Enum):
    """SFDR product classification."""
    ARTICLE_6 = "article_6"
    ARTICLE_8 = "article_8"
    ARTICLE_8_PLUS = "article_8_plus"
    ARTICLE_9 = "article_9"


class ScreeningVerdict(str, Enum):
    """Screening outcome for a holding."""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    EXCLUDED = "excluded"
    PENDING_REVIEW = "pending_review"


class ExclusionCategory(str, Enum):
    """Categories of excluded activities."""
    CONTROVERSIAL_WEAPONS = "controversial_weapons"
    TOBACCO = "tobacco"
    THERMAL_COAL = "thermal_coal"
    FOSSIL_FUEL_EXPLORATION = "fossil_fuel_exploration"
    OIL_SANDS = "oil_sands"
    ARCTIC_DRILLING = "arctic_drilling"
    UNGC_VIOLATORS = "ungc_violators"
    SEVERE_CONTROVERSIES = "severe_controversies"
    GAMBLING = "gambling"
    ADULT_ENTERTAINMENT = "adult_entertainment"
    NUCLEAR_WEAPONS = "nuclear_weapons"
    PALM_OIL_UNSUSTAINABLE = "palm_oil_unsustainable"


# =============================================================================
# Data Models
# =============================================================================


class InvestmentScreenerBridgeConfig(BaseModel):
    """Configuration for the Investment Screener Bridge."""
    screening_criteria: str = Field(
        default="article_8",
        description="Screening criteria set (article_8, article_8_plus)",
    )
    sfdr_target: SFDRClassification = Field(
        default=SFDRClassification.ARTICLE_8,
        description="Target SFDR classification",
    )
    min_esg_rating: str = Field(
        default="BBB",
        description="Minimum ESG rating for inclusion",
    )
    revenue_threshold_pct: float = Field(
        default=5.0, ge=0.0, le=100.0,
        description="Revenue threshold for exclusion (%)",
    )
    enable_controversy_screening: bool = Field(
        default=True,
        description="Enable controversy-based screening",
    )
    max_controversy_level: int = Field(
        default=4, ge=1, le=5,
        description="Maximum acceptable controversy level (1-5)",
    )
    custom_exclusions: List[str] = Field(
        default_factory=list,
        description="Additional custom exclusion categories",
    )


class ScreeningResult(BaseModel):
    """Result of screening a single holding."""
    isin: str = Field(default="", description="Holding ISIN")
    name: str = Field(default="", description="Holding name")
    verdict: ScreeningVerdict = Field(
        default=ScreeningVerdict.PASS, description="Screening verdict"
    )
    exclusion_flags: List[str] = Field(
        default_factory=list, description="Triggered exclusion categories"
    )
    positive_criteria_met: List[str] = Field(
        default_factory=list, description="Positive criteria satisfied"
    )
    esg_rating: str = Field(default="", description="ESG rating")
    esg_rating_pass: bool = Field(default=False, description="Meets minimum rating")
    controversy_level: int = Field(default=0, description="Controversy level (1-5)")
    notes: List[str] = Field(default_factory=list, description="Screening notes")


class ClassificationResult(BaseModel):
    """Result of product classification analysis."""
    classification: SFDRClassification = Field(
        default=SFDRClassification.ARTICLE_8,
        description="Determined SFDR classification",
    )
    target_met: bool = Field(
        default=False, description="Whether target classification is met"
    )
    holdings_screened: int = Field(default=0, description="Holdings screened")
    holdings_passed: int = Field(default=0, description="Holdings that passed")
    holdings_excluded: int = Field(default=0, description="Holdings excluded")
    pass_rate_pct: float = Field(default=0.0, description="Pass rate %")
    exclusion_summary: Dict[str, int] = Field(
        default_factory=dict, description="Count per exclusion category"
    )
    positive_criteria_summary: Dict[str, int] = Field(
        default_factory=dict, description="Count per positive criteria"
    )
    screening_results: List[ScreeningResult] = Field(
        default_factory=list, description="Per-holding screening results"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")
    screened_at: str = Field(default="", description="Screening timestamp")


# =============================================================================
# Exclusion & Positive Criteria Definitions
# =============================================================================


EXCLUSION_CATEGORIES: Dict[str, Dict[str, Any]] = {
    ExclusionCategory.CONTROVERSIAL_WEAPONS.value: {
        "name": "Controversial weapons",
        "description": "Cluster munitions, anti-personnel mines, biological/chemical weapons",
        "revenue_threshold_pct": 0.0,
        "article_8": True,
        "article_8_plus": True,
    },
    ExclusionCategory.TOBACCO.value: {
        "name": "Tobacco production",
        "description": "Companies deriving revenue from tobacco production",
        "revenue_threshold_pct": 5.0,
        "article_8": True,
        "article_8_plus": True,
    },
    ExclusionCategory.THERMAL_COAL.value: {
        "name": "Thermal coal extraction/generation",
        "description": "Companies with significant thermal coal exposure",
        "revenue_threshold_pct": 10.0,
        "article_8": True,
        "article_8_plus": True,
    },
    ExclusionCategory.FOSSIL_FUEL_EXPLORATION.value: {
        "name": "Fossil fuel exploration",
        "description": "New fossil fuel exploration and expansion",
        "revenue_threshold_pct": 10.0,
        "article_8": False,
        "article_8_plus": True,
    },
    ExclusionCategory.OIL_SANDS.value: {
        "name": "Oil sands extraction",
        "description": "Companies involved in oil sands extraction",
        "revenue_threshold_pct": 5.0,
        "article_8": False,
        "article_8_plus": True,
    },
    ExclusionCategory.ARCTIC_DRILLING.value: {
        "name": "Arctic oil and gas drilling",
        "description": "Companies involved in Arctic drilling",
        "revenue_threshold_pct": 5.0,
        "article_8": False,
        "article_8_plus": True,
    },
    ExclusionCategory.UNGC_VIOLATORS.value: {
        "name": "UN Global Compact violators",
        "description": "Companies in violation of UN Global Compact principles",
        "revenue_threshold_pct": 0.0,
        "article_8": True,
        "article_8_plus": True,
    },
    ExclusionCategory.SEVERE_CONTROVERSIES.value: {
        "name": "Severe ESG controversies",
        "description": "Companies with severe or very severe controversies",
        "revenue_threshold_pct": 0.0,
        "article_8": False,
        "article_8_plus": True,
    },
    ExclusionCategory.NUCLEAR_WEAPONS.value: {
        "name": "Nuclear weapons",
        "description": "Companies involved in nuclear weapons production",
        "revenue_threshold_pct": 0.0,
        "article_8": True,
        "article_8_plus": True,
    },
}

POSITIVE_CRITERIA: Dict[str, Dict[str, Any]] = {
    "esg_rating_minimum": {
        "name": "Minimum ESG rating",
        "article_8_threshold": "BBB",
        "article_8_plus_threshold": "A",
        "description": "Minimum ESG rating for inclusion",
    },
    "taxonomy_alignment": {
        "name": "EU Taxonomy alignment",
        "article_8_required": False,
        "article_8_plus_required": True,
        "description": "Holding contributes to taxonomy-aligned activities",
    },
    "sustainability_certification": {
        "name": "Sustainability certification",
        "article_8_required": False,
        "article_8_plus_required": False,
        "description": "Holding has sustainability certification (ISO 14001, etc.)",
    },
    "science_based_target": {
        "name": "Science-based target",
        "article_8_required": False,
        "article_8_plus_required": True,
        "description": "Company has validated science-based target",
    },
    "climate_transition_plan": {
        "name": "Climate transition plan",
        "article_8_required": False,
        "article_8_plus_required": False,
        "description": "Company has published climate transition plan",
    },
    "good_governance": {
        "name": "Good governance practices",
        "article_8_required": True,
        "article_8_plus_required": True,
        "description": "Sound management structures, employee relations, remuneration, tax compliance",
    },
}

ESG_RATING_ORDER: List[str] = ["CCC", "B", "BB", "BBB", "A", "AA", "AAA"]


# =============================================================================
# Investment Screener Bridge
# =============================================================================


class InvestmentScreenerBridge:
    """Bridge connecting SFDR Article 8 to green investment screening.

    Implements SFDR-specific investment screening with exclusion lists,
    positive criteria, ESG rating thresholds, and controversy screening
    for Article 8 and Article 8+ product classification.

    Attributes:
        config: Bridge configuration.
        _screener: Deferred screener agent stub.

    Example:
        >>> bridge = InvestmentScreenerBridge(
        ...     InvestmentScreenerBridgeConfig(sfdr_target="article_8")
        ... )
        >>> result = bridge.classify_product(holdings)
        >>> print(f"Pass rate: {result.pass_rate_pct:.1f}%")
    """

    def __init__(
        self, config: Optional[InvestmentScreenerBridgeConfig] = None
    ) -> None:
        """Initialize the Investment Screener Bridge.

        Args:
            config: Bridge configuration. Uses defaults if not provided.
        """
        self.config = config or InvestmentScreenerBridgeConfig()
        self.logger = logger

        self._screener = _AgentStub(
            "GL-INV-SCREENER",
            "greenlang.apps.investment.green_investment_screener",
            "GreenInvestmentScreener",
        )

        self.logger.info(
            "InvestmentScreenerBridge initialized: target=%s, min_esg=%s",
            self.config.sfdr_target.value,
            self.config.min_esg_rating,
        )

    # -------------------------------------------------------------------------
    # Public Methods
    # -------------------------------------------------------------------------

    def classify_product(
        self,
        holdings: List[Dict[str, Any]],
    ) -> ClassificationResult:
        """Classify a financial product under SFDR based on holdings screening.

        Screens all holdings against exclusion lists and positive criteria,
        then determines if the product meets the target SFDR classification.

        Args:
            holdings: List of portfolio holdings to screen.

        Returns:
            ClassificationResult with classification and per-holding details.
        """
        screening_results: List[ScreeningResult] = []
        exclusion_summary: Dict[str, int] = {}
        positive_summary: Dict[str, int] = {}
        passed = 0
        excluded = 0

        for holding in holdings:
            result = self.screen_investment(holding)
            screening_results.append(result)

            if result.verdict in (ScreeningVerdict.PASS, ScreeningVerdict.WARNING):
                passed += 1
            elif result.verdict in (
                ScreeningVerdict.EXCLUDED, ScreeningVerdict.FAIL
            ):
                excluded += 1

            for flag in result.exclusion_flags:
                exclusion_summary[flag] = exclusion_summary.get(flag, 0) + 1

            for criteria in result.positive_criteria_met:
                positive_summary[criteria] = positive_summary.get(criteria, 0) + 1

        total = len(holdings) or 1
        pass_rate = round((passed / total) * 100, 1)

        # Determine classification
        classification = self._determine_classification(
            pass_rate, exclusion_summary, positive_summary, holdings
        )
        target_met = self._classification_meets_target(classification)

        result = ClassificationResult(
            classification=classification,
            target_met=target_met,
            holdings_screened=len(holdings),
            holdings_passed=passed,
            holdings_excluded=excluded,
            pass_rate_pct=pass_rate,
            exclusion_summary=exclusion_summary,
            positive_criteria_summary=positive_summary,
            screening_results=screening_results,
            screened_at=_utcnow().isoformat(),
        )
        result.provenance_hash = _hash_data({
            "screened": len(holdings),
            "passed": passed,
            "excluded": excluded,
            "classification": classification.value,
        })

        self.logger.info(
            "Product classified as %s: %d/%d passed (%.1f%%), target_met=%s",
            classification.value, passed, len(holdings),
            pass_rate, target_met,
        )
        return result

    def screen_investment(
        self,
        holding: Dict[str, Any],
    ) -> ScreeningResult:
        """Screen a single investment against SFDR criteria.

        Args:
            holding: Holding data with ISIN, sector, ESG data.

        Returns:
            ScreeningResult with verdict and details.
        """
        isin = holding.get("isin", "")
        name = holding.get("name", "")
        is_article_8_plus = (
            self.config.sfdr_target == SFDRClassification.ARTICLE_8_PLUS
        )

        exclusion_flags: List[str] = []
        positive_met: List[str] = []
        notes: List[str] = []

        # Exclusion screening
        for cat_key, cat_def in EXCLUSION_CATEGORIES.items():
            applicable = (
                cat_def.get("article_8_plus", False) if is_article_8_plus
                else cat_def.get("article_8", False)
            )
            if not applicable:
                continue

            if self._check_exclusion(holding, cat_key, cat_def):
                exclusion_flags.append(cat_key)

        # Custom exclusions
        for custom in self.config.custom_exclusions:
            if holding.get("exclusions", {}).get(custom, False):
                exclusion_flags.append(custom)

        # ESG rating check
        esg_rating = holding.get("esg_rating", "")
        min_rating = (
            "A" if is_article_8_plus else self.config.min_esg_rating
        )
        rating_pass = self._esg_rating_meets_minimum(esg_rating, min_rating)
        if rating_pass:
            positive_met.append("esg_rating_minimum")

        # Positive criteria check
        for criteria_key, criteria_def in POSITIVE_CRITERIA.items():
            if criteria_key == "esg_rating_minimum":
                continue
            required = (
                criteria_def.get("article_8_plus_required", False)
                if is_article_8_plus
                else criteria_def.get("article_8_required", False)
            )
            met = holding.get("positive_criteria", {}).get(criteria_key, False)
            if met:
                positive_met.append(criteria_key)
            elif required:
                notes.append(
                    f"Required positive criteria '{criteria_key}' not met"
                )

        # Controversy screening
        controversy_level = int(holding.get("controversy_level", 0))
        if (
            self.config.enable_controversy_screening
            and controversy_level > self.config.max_controversy_level
        ):
            exclusion_flags.append("severe_controversies")
            notes.append(
                f"Controversy level {controversy_level} exceeds max "
                f"{self.config.max_controversy_level}"
            )

        # Determine verdict
        if exclusion_flags:
            verdict = ScreeningVerdict.EXCLUDED
        elif not rating_pass and esg_rating:
            verdict = ScreeningVerdict.FAIL
            notes.append(
                f"ESG rating '{esg_rating}' below minimum '{min_rating}'"
            )
        elif not esg_rating:
            verdict = ScreeningVerdict.WARNING
            notes.append("ESG rating not available")
        else:
            verdict = ScreeningVerdict.PASS

        return ScreeningResult(
            isin=isin,
            name=name,
            verdict=verdict,
            exclusion_flags=exclusion_flags,
            positive_criteria_met=positive_met,
            esg_rating=esg_rating,
            esg_rating_pass=rating_pass,
            controversy_level=controversy_level,
            notes=notes,
        )

    def get_exclusion_list(self) -> List[Dict[str, Any]]:
        """Get the active exclusion categories for the configured SFDR target.

        Returns:
            List of exclusion category definitions with applicability.
        """
        is_article_8_plus = (
            self.config.sfdr_target == SFDRClassification.ARTICLE_8_PLUS
        )

        result: List[Dict[str, Any]] = []
        for cat_key, cat_def in EXCLUSION_CATEGORIES.items():
            applicable = (
                cat_def.get("article_8_plus", False) if is_article_8_plus
                else cat_def.get("article_8", False)
            )
            if applicable:
                result.append({
                    "category": cat_key,
                    "name": cat_def["name"],
                    "description": cat_def["description"],
                    "revenue_threshold_pct": cat_def["revenue_threshold_pct"],
                    "applicable": True,
                })

        for custom in self.config.custom_exclusions:
            result.append({
                "category": custom,
                "name": custom.replace("_", " ").title(),
                "description": "Custom exclusion",
                "revenue_threshold_pct": 0.0,
                "applicable": True,
            })

        return result

    def get_positive_criteria(self) -> List[Dict[str, Any]]:
        """Get the active positive criteria for the configured SFDR target.

        Returns:
            List of positive criteria definitions with requirements.
        """
        is_article_8_plus = (
            self.config.sfdr_target == SFDRClassification.ARTICLE_8_PLUS
        )

        result: List[Dict[str, Any]] = []
        for criteria_key, criteria_def in POSITIVE_CRITERIA.items():
            required = (
                criteria_def.get("article_8_plus_required", False)
                if is_article_8_plus
                else criteria_def.get("article_8_required", False)
            )
            threshold = (
                criteria_def.get("article_8_plus_threshold", "")
                if is_article_8_plus
                else criteria_def.get("article_8_threshold", "")
            )
            result.append({
                "criteria": criteria_key,
                "name": criteria_def["name"],
                "description": criteria_def["description"],
                "required": required,
                "threshold": threshold,
            })

        return result

    # -------------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------------

    def _check_exclusion(
        self,
        holding: Dict[str, Any],
        category: str,
        definition: Dict[str, Any],
    ) -> bool:
        """Check if a holding triggers an exclusion category.

        Args:
            holding: Holding data.
            category: Exclusion category key.
            definition: Exclusion category definition.

        Returns:
            True if the holding should be excluded.
        """
        # Check direct exclusion flag
        exclusions = holding.get("exclusions", {})
        if exclusions.get(category, False):
            return True

        # Check revenue threshold
        threshold = definition.get("revenue_threshold_pct", 0.0)
        if threshold == 0.0:
            return exclusions.get(category, False)

        revenue_pct = float(
            holding.get("revenue_exposure", {}).get(category, 0.0)
        )
        return revenue_pct > threshold

    def _esg_rating_meets_minimum(
        self,
        rating: str,
        minimum: str,
    ) -> bool:
        """Check if an ESG rating meets the minimum threshold.

        Args:
            rating: ESG rating to check.
            minimum: Minimum acceptable rating.

        Returns:
            True if rating >= minimum.
        """
        if not rating or not minimum:
            return False

        rating_upper = rating.upper()
        minimum_upper = minimum.upper()

        if rating_upper not in ESG_RATING_ORDER:
            return False
        if minimum_upper not in ESG_RATING_ORDER:
            return False

        return ESG_RATING_ORDER.index(rating_upper) >= ESG_RATING_ORDER.index(
            minimum_upper
        )

    def _determine_classification(
        self,
        pass_rate: float,
        exclusion_summary: Dict[str, int],
        positive_summary: Dict[str, int],
        holdings: List[Dict[str, Any]],
    ) -> SFDRClassification:
        """Determine SFDR classification based on screening results.

        Args:
            pass_rate: Percentage of holdings that passed screening.
            exclusion_summary: Count per exclusion category triggered.
            positive_summary: Count per positive criteria met.
            holdings: Original holdings list.

        Returns:
            Determined SFDRClassification.
        """
        total = len(holdings) or 1

        # Article 8+ requires higher bar
        taxonomy_count = positive_summary.get("taxonomy_alignment", 0)
        taxonomy_pct = (taxonomy_count / total) * 100
        sbt_count = positive_summary.get("science_based_target", 0)
        sbt_pct = (sbt_count / total) * 100

        if pass_rate >= 90 and taxonomy_pct >= 20 and sbt_pct >= 30:
            return SFDRClassification.ARTICLE_8_PLUS

        if pass_rate >= 80:
            return SFDRClassification.ARTICLE_8

        return SFDRClassification.ARTICLE_6

    def _classification_meets_target(
        self, classification: SFDRClassification
    ) -> bool:
        """Check if determined classification meets the target.

        Args:
            classification: Determined classification.

        Returns:
            True if classification meets or exceeds target.
        """
        order = [
            SFDRClassification.ARTICLE_6,
            SFDRClassification.ARTICLE_8,
            SFDRClassification.ARTICLE_8_PLUS,
            SFDRClassification.ARTICLE_9,
        ]
        determined_idx = order.index(classification)
        target_idx = order.index(self.config.sfdr_target)
        return determined_idx >= target_idx
