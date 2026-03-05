"""
Scope 3 Screening Engine -- SBTi Scope 3 Significance and Coverage Assessment

Implements the SBTi Scope 3 screening process:
  - Scope 3 trigger assessment (C8): S3 >= 40% of S1+S2+S3
  - Category-level emissions breakdown
  - Hotspot identification (top categories by emissions)
  - Coverage calculation for selected target categories
  - Category recommendation for target boundary
  - Data quality assessment per category
  - Near-term (67%) and long-term (90%) coverage calculations
  - Minimum boundary definitions
  - Screening report generation

All calculations are deterministic (zero-hallucination).

Reference:
    - SBTi Criteria C8 and C9 (Scope 3 trigger and coverage)
    - GHG Protocol Scope 3 Standard (2011)
    - SBTi Scope 3 Calculation Guidance

Example:
    >>> engine = Scope3ScreeningEngine(config)
    >>> trigger = engine.assess_scope3_trigger(inventory)
    >>> trigger.is_required
    True
"""

from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from .config import (
    SCOPE3_CATEGORY_NAMES,
    SCOPE3_NEAR_TERM_COVERAGE,
    SCOPE3_TRIGGER_THRESHOLD,
    SBTiAppConfig,
)
from .models import (
    CategoryBreakdown,
    EmissionsInventory,
    Scope3CategoryEmissions,
    Scope3Screening,
    _new_id,
    _now,
    _sha256,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Screening Result Data Classes
# ---------------------------------------------------------------------------

class TriggerAssessment:
    """Result of Scope 3 trigger assessment per criterion C8."""

    def __init__(
        self,
        scope3_total_tco2e: float,
        total_s1_s2_s3_tco2e: float,
        scope3_pct: float,
        threshold_pct: float,
        is_required: bool,
        message: str,
    ) -> None:
        self.scope3_total_tco2e = scope3_total_tco2e
        self.total_s1_s2_s3_tco2e = total_s1_s2_s3_tco2e
        self.scope3_pct = scope3_pct
        self.threshold_pct = threshold_pct
        self.is_required = is_required
        self.message = message


class Hotspot:
    """A Scope 3 hotspot (high-emissions category)."""

    def __init__(
        self,
        category_number: int,
        category_name: str,
        emissions_tco2e: float,
        pct_of_scope3: float,
        pct_of_total: float,
        rank: int,
    ) -> None:
        self.category_number = category_number
        self.category_name = category_name
        self.emissions_tco2e = emissions_tco2e
        self.pct_of_scope3 = pct_of_scope3
        self.pct_of_total = pct_of_total
        self.rank = rank


class CoverageResult:
    """Result of coverage calculation for selected categories."""

    def __init__(
        self,
        selected_emissions_tco2e: float,
        total_scope3_tco2e: float,
        coverage_pct: float,
        threshold_pct: float,
        is_sufficient: bool,
        selected_categories: List[int],
        message: str,
    ) -> None:
        self.selected_emissions_tco2e = selected_emissions_tco2e
        self.total_scope3_tco2e = total_scope3_tco2e
        self.coverage_pct = coverage_pct
        self.threshold_pct = threshold_pct
        self.is_sufficient = is_sufficient
        self.selected_categories = selected_categories
        self.message = message


class CategoryRecommendation:
    """Recommended categories for target boundary."""

    def __init__(
        self,
        recommended_categories: List[int],
        total_coverage_pct: float,
        near_term_sufficient: bool,
        long_term_sufficient: bool,
        category_details: List[Dict[str, Any]],
    ) -> None:
        self.recommended_categories = recommended_categories
        self.total_coverage_pct = total_coverage_pct
        self.near_term_sufficient = near_term_sufficient
        self.long_term_sufficient = long_term_sufficient
        self.category_details = category_details


class CategoryDataQuality:
    """Data quality assessment for a Scope 3 category."""

    def __init__(
        self,
        category_number: int,
        category_name: str,
        data_quality: str,
        methodology: str,
        confidence_pct: float,
        improvement_actions: List[str],
    ) -> None:
        self.category_number = category_number
        self.category_name = category_name
        self.data_quality = data_quality
        self.methodology = methodology
        self.confidence_pct = confidence_pct
        self.improvement_actions = improvement_actions


class MinimumBoundary:
    """Minimum boundary definition for a Scope 3 category."""

    def __init__(
        self,
        category_number: int,
        category_name: str,
        included: bool,
        reason: str,
    ) -> None:
        self.category_number = category_number
        self.category_name = category_name
        self.included = included
        self.reason = reason


class ScreeningReport:
    """Comprehensive Scope 3 screening report."""

    def __init__(
        self,
        org_id: str,
        trigger_result: TriggerAssessment,
        category_breakdown: List[Dict[str, Any]],
        hotspots: List[Dict[str, Any]],
        coverage: CoverageResult,
        recommendations: CategoryRecommendation,
        data_quality: List[Dict[str, Any]],
        generated_at: str,
    ) -> None:
        self.org_id = org_id
        self.trigger_result = trigger_result
        self.category_breakdown = category_breakdown
        self.hotspots = hotspots
        self.coverage = coverage
        self.recommendations = recommendations
        self.data_quality = data_quality
        self.generated_at = generated_at


class Scope3ScreeningEngine:
    """
    SBTi Scope 3 Screening and Coverage Assessment Engine.

    Assesses Scope 3 significance, identifies emission hotspots,
    calculates coverage for selected target categories, and generates
    screening reports.

    Attributes:
        config: Application configuration.
        _screenings: In-memory store of screening results keyed by org ID.
        _inventories: In-memory store of inventories keyed by org ID.
    """

    def __init__(self, config: Optional[SBTiAppConfig] = None) -> None:
        """
        Initialize Scope3ScreeningEngine.

        Args:
            config: Application configuration instance.
        """
        self.config = config or SBTiAppConfig()
        self._screenings: Dict[str, Scope3Screening] = {}
        self._inventories: Dict[str, EmissionsInventory] = {}
        logger.info("Scope3ScreeningEngine initialized")

    def register_inventory(self, inventory: EmissionsInventory) -> None:
        """Register an inventory for screening."""
        self._inventories[inventory.org_id] = inventory

    # ------------------------------------------------------------------
    # Scope 3 Trigger Assessment
    # ------------------------------------------------------------------

    def assess_scope3_trigger(self, inventory: EmissionsInventory) -> TriggerAssessment:
        """
        Assess whether Scope 3 target is required per criterion C8.

        Formula: S3 / (S1 + S2 + S3) >= 0.40

        Args:
            inventory: Emissions inventory with Scope 1, 2, 3 totals.

        Returns:
            TriggerAssessment with pass/fail and percentages.
        """
        s3 = float(inventory.scope3_total_tco2e)
        total = float(inventory.total_s1_s2_s3_tco2e)

        if total <= 0:
            return TriggerAssessment(
                scope3_total_tco2e=s3,
                total_s1_s2_s3_tco2e=total,
                scope3_pct=0.0,
                threshold_pct=SCOPE3_TRIGGER_THRESHOLD * 100,
                is_required=False,
                message="Total emissions are zero; cannot assess Scope 3 trigger.",
            )

        s3_pct = (s3 / total) * 100
        is_required = s3_pct >= (SCOPE3_TRIGGER_THRESHOLD * 100)

        logger.info(
            "Scope 3 trigger: %.1f%% (threshold %.0f%%) -> %s",
            s3_pct, SCOPE3_TRIGGER_THRESHOLD * 100,
            "REQUIRED" if is_required else "NOT REQUIRED",
        )

        return TriggerAssessment(
            scope3_total_tco2e=s3,
            total_s1_s2_s3_tco2e=total,
            scope3_pct=round(s3_pct, 2),
            threshold_pct=SCOPE3_TRIGGER_THRESHOLD * 100,
            is_required=is_required,
            message=(
                f"Scope 3 is {s3_pct:.1f}% of total emissions. "
                f"{'Scope 3 target IS required.' if is_required else 'Scope 3 target is NOT required.'}"
            ),
        )

    # ------------------------------------------------------------------
    # Category Breakdown
    # ------------------------------------------------------------------

    def calculate_category_breakdown(
        self,
        inventory: EmissionsInventory,
    ) -> List[CategoryBreakdown]:
        """
        Calculate emissions breakdown by Scope 3 category.

        Args:
            inventory: Emissions inventory with per-category data.

        Returns:
            List of CategoryBreakdown sorted by emissions (descending).
        """
        s3_total = float(inventory.scope3_total_tco2e)
        total = float(inventory.total_s1_s2_s3_tco2e)

        breakdowns: List[CategoryBreakdown] = []

        for cat in inventory.scope3_categories:
            emissions = float(cat.emissions_tco2e)
            pct_of_s3 = (emissions / s3_total * 100) if s3_total > 0 else 0
            pct_of_total = (emissions / total * 100) if total > 0 else 0
            is_material = pct_of_total >= 1.0

            breakdowns.append(CategoryBreakdown(
                category=cat.category,
                category_number=cat.category_number,
                category_name=cat.category_name or SCOPE3_CATEGORY_NAMES.get(cat.category_number, ""),
                emissions_tco2e=cat.emissions_tco2e,
                percentage_of_scope3=Decimal(str(round(pct_of_s3, 2))),
                percentage_of_total=Decimal(str(round(pct_of_total, 2))),
                data_quality=cat.data_quality,
                included_in_target=cat.included_in_target,
                is_material=is_material,
            ))

        breakdowns.sort(key=lambda b: float(b.emissions_tco2e), reverse=True)

        logger.info(
            "Category breakdown: %d categories, %d material (>1%%)",
            len(breakdowns), sum(1 for b in breakdowns if b.is_material),
        )
        return breakdowns

    # ------------------------------------------------------------------
    # Hotspot Identification
    # ------------------------------------------------------------------

    def identify_hotspots(
        self,
        inventory: EmissionsInventory,
        top_n: int = 5,
    ) -> List[Hotspot]:
        """
        Identify top Scope 3 emission hotspot categories.

        Args:
            inventory: Emissions inventory.
            top_n: Number of top categories to return.

        Returns:
            List of Hotspot sorted by emissions (descending).
        """
        breakdowns = self.calculate_category_breakdown(inventory)

        hotspots: List[Hotspot] = []
        for rank, bd in enumerate(breakdowns[:top_n], start=1):
            hotspots.append(Hotspot(
                category_number=bd.category_number,
                category_name=bd.category_name,
                emissions_tco2e=float(bd.emissions_tco2e),
                pct_of_scope3=float(bd.percentage_of_scope3),
                pct_of_total=float(bd.percentage_of_total),
                rank=rank,
            ))

        logger.info("Identified %d hotspots from %d categories", len(hotspots), len(breakdowns))
        return hotspots

    # ------------------------------------------------------------------
    # Coverage Calculation
    # ------------------------------------------------------------------

    def calculate_coverage(
        self,
        selected_categories: List[int],
        inventory: EmissionsInventory,
    ) -> CoverageResult:
        """
        Calculate coverage of selected categories against total Scope 3.

        Coverage = sum(selected) / total_scope3

        Args:
            selected_categories: List of category numbers to include.
            inventory: Emissions inventory.

        Returns:
            CoverageResult with coverage percentage and sufficiency.
        """
        s3_total = float(inventory.scope3_total_tco2e)
        if s3_total <= 0:
            return CoverageResult(
                selected_emissions_tco2e=0,
                total_scope3_tco2e=0,
                coverage_pct=0,
                threshold_pct=SCOPE3_NEAR_TERM_COVERAGE * 100,
                is_sufficient=False,
                selected_categories=selected_categories,
                message="Total Scope 3 emissions are zero.",
            )

        selected_total = 0.0
        for cat in inventory.scope3_categories:
            if cat.category_number in selected_categories:
                selected_total += float(cat.emissions_tco2e)

        coverage_pct = (selected_total / s3_total) * 100
        threshold = SCOPE3_NEAR_TERM_COVERAGE * 100
        is_sufficient = coverage_pct >= threshold

        logger.info(
            "Coverage: %d categories, %.1f tCO2e, %.1f%% (threshold %.0f%%)",
            len(selected_categories), selected_total, coverage_pct, threshold,
        )

        return CoverageResult(
            selected_emissions_tco2e=round(selected_total, 2),
            total_scope3_tco2e=round(s3_total, 2),
            coverage_pct=round(coverage_pct, 2),
            threshold_pct=threshold,
            is_sufficient=is_sufficient,
            selected_categories=selected_categories,
            message=(
                f"Selected categories cover {coverage_pct:.1f}% of Scope 3. "
                f"{'Sufficient' if is_sufficient else 'Insufficient'} "
                f"(threshold: {threshold:.0f}%)."
            ),
        )

    # ------------------------------------------------------------------
    # Category Recommendations
    # ------------------------------------------------------------------

    def recommend_target_categories(
        self,
        inventory: EmissionsInventory,
    ) -> CategoryRecommendation:
        """
        Recommend Scope 3 categories for the target boundary.

        Selects categories in descending order of emissions until
        the near-term coverage threshold (67%) is met.

        Args:
            inventory: Emissions inventory.

        Returns:
            CategoryRecommendation with ordered category list.
        """
        breakdowns = self.calculate_category_breakdown(inventory)
        s3_total = float(inventory.scope3_total_tco2e)

        if s3_total <= 0:
            return CategoryRecommendation(
                recommended_categories=[],
                total_coverage_pct=0,
                near_term_sufficient=False,
                long_term_sufficient=False,
                category_details=[],
            )

        recommended: List[int] = []
        cumulative = 0.0
        details: List[Dict[str, Any]] = []

        for bd in breakdowns:
            emissions = float(bd.emissions_tco2e)
            cumulative += emissions
            coverage = (cumulative / s3_total) * 100
            recommended.append(bd.category_number)
            details.append({
                "category_number": bd.category_number,
                "category_name": bd.category_name,
                "emissions_tco2e": emissions,
                "pct_of_scope3": float(bd.percentage_of_scope3),
                "cumulative_coverage_pct": round(coverage, 2),
            })

        total_coverage = (cumulative / s3_total) * 100 if s3_total > 0 else 0
        near_term_ok = total_coverage >= (SCOPE3_NEAR_TERM_COVERAGE * 100)
        long_term_ok = total_coverage >= (self.config.scope3_long_term_coverage_min * 100)

        # Find minimum set for near-term coverage
        min_categories: List[int] = []
        cum = 0.0
        for bd in breakdowns:
            cum += float(bd.emissions_tco2e)
            min_categories.append(bd.category_number)
            if (cum / s3_total) * 100 >= (SCOPE3_NEAR_TERM_COVERAGE * 100):
                break

        logger.info(
            "Recommended %d categories for %.1f%% coverage (min %d for 67%%)",
            len(recommended), total_coverage, len(min_categories),
        )

        return CategoryRecommendation(
            recommended_categories=min_categories,
            total_coverage_pct=round(total_coverage, 2),
            near_term_sufficient=near_term_ok,
            long_term_sufficient=long_term_ok,
            category_details=details,
        )

    # ------------------------------------------------------------------
    # Data Quality Assessment
    # ------------------------------------------------------------------

    def assess_data_quality_by_category(
        self,
        inventory: EmissionsInventory,
    ) -> List[CategoryDataQuality]:
        """
        Assess data quality for each Scope 3 category.

        Args:
            inventory: Emissions inventory.

        Returns:
            List of CategoryDataQuality assessments.
        """
        results: List[CategoryDataQuality] = []

        dq_confidence = {
            "measured": 95.0, "calculated": 80.0, "estimated": 60.0,
            "proxy": 40.0, "default": 20.0,
        }

        for cat in inventory.scope3_categories:
            dq = cat.data_quality.value if hasattr(cat.data_quality, 'value') else str(cat.data_quality)
            confidence = dq_confidence.get(dq, 50.0)

            actions: List[str] = []
            if confidence < 60:
                actions.append(f"Improve data quality for category {cat.category_number} from {dq} to calculated or measured.")
            if confidence < 40:
                actions.append("Engage suppliers for primary data collection.")
            if cat.methodology is None:
                actions.append("Document methodology used for this category.")

            results.append(CategoryDataQuality(
                category_number=cat.category_number,
                category_name=cat.category_name or SCOPE3_CATEGORY_NAMES.get(cat.category_number, ""),
                data_quality=dq,
                methodology=cat.methodology or "Not specified",
                confidence_pct=confidence,
                improvement_actions=actions,
            ))

        logger.info("Assessed data quality for %d categories", len(results))
        return results

    # ------------------------------------------------------------------
    # Near-Term and Long-Term Coverage
    # ------------------------------------------------------------------

    def calculate_near_term_coverage(
        self,
        selected_cats: List[int],
        inventory: EmissionsInventory,
    ) -> float:
        """
        Calculate near-term Scope 3 coverage (target 67%).

        Args:
            selected_cats: Selected category numbers.
            inventory: Emissions inventory.

        Returns:
            Coverage percentage.
        """
        result = self.calculate_coverage(selected_cats, inventory)
        return result.coverage_pct

    def calculate_long_term_coverage(
        self,
        selected_cats: List[int],
        inventory: EmissionsInventory,
    ) -> float:
        """
        Calculate long-term Scope 3 coverage (target 90%).

        Args:
            selected_cats: Selected category numbers.
            inventory: Emissions inventory.

        Returns:
            Coverage percentage.
        """
        result = self.calculate_coverage(selected_cats, inventory)
        return result.coverage_pct

    # ------------------------------------------------------------------
    # Minimum Boundaries
    # ------------------------------------------------------------------

    def get_minimum_boundaries(
        self,
        categories: List[Scope3CategoryEmissions],
    ) -> List[MinimumBoundary]:
        """
        Determine minimum boundary inclusion for each Scope 3 category.

        Categories are included if they are relevant and not explicitly
        excluded with justification.

        Args:
            categories: List of Scope 3 category emissions data.

        Returns:
            List of MinimumBoundary definitions.
        """
        boundaries: List[MinimumBoundary] = []

        for cat in categories:
            name = cat.category_name or SCOPE3_CATEGORY_NAMES.get(cat.category_number, "")
            included = cat.is_relevant and float(cat.emissions_tco2e) > 0

            reason = "Relevant and quantified" if included else "Not relevant or zero emissions"
            if not cat.is_relevant and cat.exclusion_reason:
                reason = cat.exclusion_reason

            boundaries.append(MinimumBoundary(
                category_number=cat.category_number,
                category_name=name,
                included=included,
                reason=reason,
            ))

        logger.info(
            "Minimum boundaries: %d included, %d excluded",
            sum(1 for b in boundaries if b.included),
            sum(1 for b in boundaries if not b.included),
        )
        return boundaries

    # ------------------------------------------------------------------
    # Screening Report
    # ------------------------------------------------------------------

    def generate_screening_report(self, org_id: str) -> ScreeningReport:
        """
        Generate a comprehensive Scope 3 screening report.

        Args:
            org_id: Organization ID.

        Returns:
            ScreeningReport with all screening results.
        """
        start = datetime.utcnow()
        inventory = self._inventories.get(org_id)

        if inventory is None:
            raise ValueError(f"No inventory registered for org {org_id}")

        trigger = self.assess_scope3_trigger(inventory)
        breakdowns = self.calculate_category_breakdown(inventory)
        hotspots = self.identify_hotspots(inventory)
        recommendations = self.recommend_target_categories(inventory)
        coverage = self.calculate_coverage(recommendations.recommended_categories, inventory)
        data_quality = self.assess_data_quality_by_category(inventory)

        report = ScreeningReport(
            org_id=org_id,
            trigger_result=trigger,
            category_breakdown=[{
                "category_number": b.category_number,
                "category_name": b.category_name,
                "emissions_tco2e": float(b.emissions_tco2e),
                "pct_of_scope3": float(b.percentage_of_scope3),
                "pct_of_total": float(b.percentage_of_total),
                "is_material": b.is_material,
            } for b in breakdowns],
            hotspots=[{
                "rank": h.rank,
                "category_number": h.category_number,
                "category_name": h.category_name,
                "emissions_tco2e": h.emissions_tco2e,
                "pct_of_scope3": h.pct_of_scope3,
            } for h in hotspots],
            coverage=coverage,
            recommendations=recommendations,
            data_quality=[{
                "category_number": dq.category_number,
                "data_quality": dq.data_quality,
                "confidence_pct": dq.confidence_pct,
                "actions": dq.improvement_actions,
            } for dq in data_quality],
            generated_at=_now().isoformat(),
        )

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "Screening report for org %s: trigger=%s categories=%d coverage=%.1f%% in %.1f ms",
            org_id, trigger.is_required, len(breakdowns), coverage.coverage_pct, elapsed_ms,
        )
        return report
