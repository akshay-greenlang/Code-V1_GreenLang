"""
Significance Engine -- ISO 14064-1:2018 Clause 5.2 Implementation

Implements multi-criteria significance assessment for indirect emission
categories (Categories 2-6) per ISO 14064-1:2018.  The organization shall
assess the significance of each indirect emission category and document
the rationale for including or excluding categories from the inventory.

Five assessment criteria evaluated on a 0-100 scale:
  1. Magnitude   -- Absolute emissions relative to total
  2. Influence   -- Organization's ability to reduce emissions
  3. Risk        -- Exposure to regulatory/market/reputational risks
  4. Stakeholder -- External expectations from stakeholders
  5. Data Availability -- Feasibility of quantification

A composite score determines significance.  Categories exceeding
the composite threshold AND/OR the magnitude threshold are classified
as significant.

Reference: ISO 14064-1:2018 Clause 5.2.2.

Example:
    >>> engine = SignificanceEngine(config)
    >>> result = engine.assess_category("inv-1", ISOCategory.CATEGORY_3_TRANSPORT,
    ...     criteria=SignificanceCriteria(magnitude=Decimal("80"), influence=Decimal("60")),
    ...     estimated_emissions=Decimal("500"), total_emissions=Decimal("10000"))
    >>> result.result
    <SignificanceLevel.SIGNIFICANT: 'significant'>
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from .config import (
    ISO14064AppConfig,
    ISOCategory,
    ISO_CATEGORY_NAMES,
    SignificanceLevel,
)
from .models import (
    SignificanceAssessment,
    SignificanceCriteria,
    _new_id,
    _now,
    _sha256,
)

logger = logging.getLogger(__name__)


class SignificanceEngine:
    """
    Multi-criteria significance assessment for ISO 14064-1 indirect categories.

    Evaluates each indirect category against weighted criteria to
    determine whether it is significant and should be quantified in the
    GHG inventory.  Follows ISO 14064-1:2018 Clause 5.2.2.

    Attributes:
        config: Application configuration.
        _assessments: In-memory store keyed by inventory_id.
        _yoy_history: Year-over-year tracking data.
    """

    INDIRECT_CATEGORIES: List[ISOCategory] = [
        ISOCategory.CATEGORY_2_ENERGY,
        ISOCategory.CATEGORY_3_TRANSPORT,
        ISOCategory.CATEGORY_4_PRODUCTS_USED,
        ISOCategory.CATEGORY_5_PRODUCTS_FROM_ORG,
        ISOCategory.CATEGORY_6_OTHER,
    ]

    def __init__(
        self,
        config: Optional[ISO14064AppConfig] = None,
    ) -> None:
        """
        Initialize SignificanceEngine.

        Args:
            config: Application configuration.
        """
        self.config = config or ISO14064AppConfig()
        self._magnitude_threshold = self.config.significance_threshold_percent
        self._composite_threshold = Decimal("50.0")  # 50/100 scale
        self._assessments: Dict[str, List[SignificanceAssessment]] = {}
        self._yoy_history: Dict[str, Dict[int, List[SignificanceAssessment]]] = {}

        logger.info(
            "SignificanceEngine initialized (magnitude_threshold=%.1f%%, "
            "composite_threshold=%.1f)",
            self._magnitude_threshold,
            self._composite_threshold,
        )

    # ------------------------------------------------------------------
    # Public Interface
    # ------------------------------------------------------------------

    def assess_category(
        self,
        inventory_id: str,
        category: ISOCategory,
        criteria: SignificanceCriteria,
        estimated_emissions: Decimal = Decimal("0"),
        total_emissions: Decimal = Decimal("0"),
        threshold_pct: Optional[Decimal] = None,
        assessed_by: str = "",
    ) -> SignificanceAssessment:
        """
        Assess significance of a single indirect category.

        Args:
            inventory_id: Parent inventory ID.
            category: ISO category to assess.
            criteria: Criteria scores (0-100 per dimension).
            estimated_emissions: Estimated emissions for this category (tCO2e).
            total_emissions: Total inventory emissions (tCO2e).
            threshold_pct: Custom threshold (defaults to config value).
            assessed_by: ID of the assessor.

        Returns:
            SignificanceAssessment with determination and justification.

        Raises:
            ValueError: If category is Category 1 (always direct/significant).
        """
        start = datetime.utcnow()

        if category == ISOCategory.CATEGORY_1_DIRECT:
            raise ValueError(
                "Category 1 (direct) is always significant and does not "
                "require significance assessment."
            )

        composite_score = criteria.composite_score
        magnitude_pct = self._calculate_magnitude_pct(
            estimated_emissions, total_emissions,
        )
        effective_threshold = threshold_pct or self._magnitude_threshold
        determination = self._determine_significance(
            composite_score, magnitude_pct, effective_threshold,
        )
        justification = self._generate_justification(
            category, determination, composite_score, magnitude_pct, effective_threshold,
        )

        assessment = SignificanceAssessment(
            iso_category=category,
            criteria=criteria,
            threshold_pct=effective_threshold,
            result=determination,
            justification=justification,
            estimated_emissions_tco2e=estimated_emissions,
            assessed_by=assessed_by,
        )

        if inventory_id not in self._assessments:
            self._assessments[inventory_id] = []
        # Replace existing assessment for same category
        self._assessments[inventory_id] = [
            a for a in self._assessments[inventory_id]
            if a.iso_category != category
        ]
        self._assessments[inventory_id].append(assessment)

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "Assessed %s: composite=%.1f, magnitude=%.1f%%, determination=%s in %.1f ms",
            category.value,
            composite_score,
            magnitude_pct,
            determination.value,
            elapsed_ms,
        )
        return assessment

    def batch_assess_all_indirect(
        self,
        inventory_id: str,
        category_emissions: Dict[ISOCategory, Decimal],
        total_emissions: Decimal,
        category_criteria: Optional[Dict[ISOCategory, SignificanceCriteria]] = None,
        assessed_by: str = "",
    ) -> List[SignificanceAssessment]:
        """
        Batch assess all indirect categories (2-6).

        Args:
            inventory_id: Parent inventory ID.
            category_emissions: Emissions per category.
            total_emissions: Total inventory emissions.
            category_criteria: Criteria per category (defaults used if absent).
            assessed_by: Assessor ID.

        Returns:
            List of SignificanceAssessment results.
        """
        start = datetime.utcnow()
        results: List[SignificanceAssessment] = []

        for category in self.INDIRECT_CATEGORIES:
            emissions = category_emissions.get(category, Decimal("0"))
            criteria = None
            if category_criteria:
                criteria = category_criteria.get(category)
            if criteria is None:
                criteria = self._default_criteria()

            assessment = self.assess_category(
                inventory_id=inventory_id,
                category=category,
                criteria=criteria,
                estimated_emissions=emissions,
                total_emissions=total_emissions,
                assessed_by=assessed_by,
            )
            results.append(assessment)

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        significant_count = sum(
            1 for r in results if r.result == SignificanceLevel.SIGNIFICANT
        )
        logger.info(
            "Batch assessment for inventory %s: %d/%d significant in %.1f ms",
            inventory_id,
            significant_count,
            len(results),
            elapsed_ms,
        )
        return results

    def get_assessments(
        self,
        inventory_id: str,
    ) -> List[SignificanceAssessment]:
        """Retrieve all assessments for an inventory."""
        return self._assessments.get(inventory_id, [])

    def get_assessment_by_category(
        self,
        inventory_id: str,
        category: ISOCategory,
    ) -> Optional[SignificanceAssessment]:
        """Retrieve assessment for a specific category."""
        for a in self._assessments.get(inventory_id, []):
            if a.iso_category == category:
                return a
        return None

    def get_significant_categories(
        self,
        inventory_id: str,
    ) -> List[ISOCategory]:
        """
        Get categories determined as significant.

        Category 1 is always included.
        """
        significant = [ISOCategory.CATEGORY_1_DIRECT]

        for assessment in self._assessments.get(inventory_id, []):
            if assessment.result == SignificanceLevel.SIGNIFICANT:
                significant.append(assessment.iso_category)

        return significant

    def get_exclusion_justifications(
        self,
        inventory_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Generate exclusion justifications for non-significant categories.

        Per ISO 14064-1, excluded categories must have documented justification.
        """
        exclusions: List[Dict[str, Any]] = []

        for assessment in self._assessments.get(inventory_id, []):
            if assessment.result == SignificanceLevel.NOT_SIGNIFICANT:
                exclusions.append({
                    "category": assessment.iso_category.value,
                    "category_name": ISO_CATEGORY_NAMES.get(
                        assessment.iso_category, assessment.iso_category.value,
                    ),
                    "determination": assessment.result.value,
                    "composite_score": str(assessment.criteria.composite_score),
                    "threshold_pct": str(assessment.threshold_pct),
                    "justification": assessment.justification,
                    "assessed_at": assessment.assessed_at.isoformat(),
                    "assessed_by": assessment.assessed_by or "",
                })

        return exclusions

    def record_yoy_assessment(
        self,
        inventory_id: str,
        year: int,
    ) -> None:
        """Record current assessments for year-over-year tracking."""
        assessments = self._assessments.get(inventory_id, [])
        if not assessments:
            return

        if inventory_id not in self._yoy_history:
            self._yoy_history[inventory_id] = {}
        self._yoy_history[inventory_id][year] = list(assessments)

        logger.info(
            "Recorded %d assessments for inventory %s year %d",
            len(assessments),
            inventory_id,
            year,
        )

    def get_yoy_significance_trend(
        self,
        inventory_id: str,
    ) -> Dict[str, Any]:
        """Get year-over-year significance tracking data."""
        history = self._yoy_history.get(inventory_id, {})
        if not history:
            return {"message": "No historical significance data available"}

        trend: Dict[str, Any] = {
            "inventory_id": inventory_id,
            "years": sorted(history.keys()),
            "by_category": {},
        }

        for category in self.INDIRECT_CATEGORIES:
            cat_trend: List[Dict[str, Any]] = []
            for year in sorted(history.keys()):
                year_assessments = history[year]
                cat_assessment = next(
                    (a for a in year_assessments if a.iso_category == category),
                    None,
                )
                if cat_assessment:
                    cat_trend.append({
                        "year": year,
                        "composite_score": str(cat_assessment.criteria.composite_score),
                        "threshold_pct": str(cat_assessment.threshold_pct),
                        "determination": cat_assessment.result.value,
                    })
            trend["by_category"][category.value] = cat_trend

        return trend

    def update_thresholds(
        self,
        magnitude_threshold: Optional[Decimal] = None,
        composite_threshold: Optional[Decimal] = None,
    ) -> None:
        """Update significance thresholds."""
        if magnitude_threshold is not None:
            self._magnitude_threshold = magnitude_threshold
        if composite_threshold is not None:
            self._composite_threshold = composite_threshold

        logger.info(
            "Updated thresholds: magnitude=%.1f%%, composite=%.1f",
            self._magnitude_threshold,
            self._composite_threshold,
        )

    def generate_summary(
        self,
        inventory_id: str,
    ) -> Dict[str, Any]:
        """Generate a summary of all significance assessments."""
        assessments = self._assessments.get(inventory_id, [])
        if not assessments:
            return {"message": "No assessments found", "inventory_id": inventory_id}

        significant = [a for a in assessments if a.result == SignificanceLevel.SIGNIFICANT]
        not_significant = [a for a in assessments if a.result == SignificanceLevel.NOT_SIGNIFICANT]
        under_review = [a for a in assessments if a.result == SignificanceLevel.UNDER_REVIEW]

        return {
            "inventory_id": inventory_id,
            "total_assessed": len(assessments),
            "significant_count": len(significant),
            "not_significant_count": len(not_significant),
            "under_review_count": len(under_review),
            "significant_categories": [a.iso_category.value for a in significant],
            "excluded_categories": [a.iso_category.value for a in not_significant],
            "under_review_categories": [a.iso_category.value for a in under_review],
            "magnitude_threshold_pct": str(self._magnitude_threshold),
            "composite_threshold": str(self._composite_threshold),
        }

    # ------------------------------------------------------------------
    # Core Calculation
    # ------------------------------------------------------------------

    @staticmethod
    def _calculate_magnitude_pct(
        category_emissions: Decimal,
        total_emissions: Decimal,
    ) -> Decimal:
        """Calculate category emissions as percentage of total."""
        if total_emissions <= 0:
            return Decimal("0")
        return (category_emissions / total_emissions * 100).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP,
        )

    def _determine_significance(
        self,
        composite_score: Decimal,
        magnitude_pct: Decimal,
        magnitude_threshold: Decimal,
    ) -> SignificanceLevel:
        """Determine significance from composite score and magnitude."""
        score_significant = composite_score >= self._composite_threshold
        magnitude_significant = magnitude_pct >= magnitude_threshold

        if score_significant or magnitude_significant:
            return SignificanceLevel.SIGNIFICANT

        # Near-threshold: flag for review (within 10% of thresholds)
        score_margin = self._composite_threshold - composite_score
        magnitude_margin = magnitude_threshold - magnitude_pct
        if score_margin <= Decimal("5.0") or magnitude_margin <= Decimal("0.5"):
            return SignificanceLevel.UNDER_REVIEW

        return SignificanceLevel.NOT_SIGNIFICANT

    @staticmethod
    def _generate_justification(
        category: ISOCategory,
        determination: SignificanceLevel,
        composite_score: Decimal,
        magnitude_pct: Decimal,
        threshold: Decimal,
    ) -> str:
        """Generate human-readable justification text."""
        cat_name = ISO_CATEGORY_NAMES.get(category, category.value)

        if determination == SignificanceLevel.SIGNIFICANT:
            return (
                f"{cat_name} is assessed as SIGNIFICANT. "
                f"Composite score={composite_score}, "
                f"magnitude={magnitude_pct}% of total."
            )

        if determination == SignificanceLevel.NOT_SIGNIFICANT:
            return (
                f"{cat_name} is assessed as NOT SIGNIFICANT. "
                f"Composite score ({composite_score}) is below threshold and "
                f"magnitude ({magnitude_pct}%) is below {threshold}%. "
                f"Excluded from quantified inventory "
                f"per ISO 14064-1:2018 Clause 5.2.2."
            )

        return (
            f"{cat_name} is UNDER REVIEW. Composite score "
            f"({composite_score}) and/or magnitude "
            f"({magnitude_pct}%) are near thresholds."
        )

    @staticmethod
    def _default_criteria() -> SignificanceCriteria:
        """Generate default criteria with mid-range scores."""
        return SignificanceCriteria(
            magnitude=Decimal("50"),
            influence=Decimal("50"),
            risk=Decimal("50"),
            stakeholder=Decimal("50"),
            data_availability=Decimal("50"),
        )
