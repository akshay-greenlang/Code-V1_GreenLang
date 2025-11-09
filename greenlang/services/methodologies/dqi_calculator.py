"""
Data Quality Index (DQI) Calculator

This module implements comprehensive Data Quality Index calculation for
emission factors and activity data. DQI combines multiple quality dimensions
into a single 0-100 score.

Key Features:
- Tier-based scoring (Tier 1=Primary, Tier 2=Secondary, Tier 3=Estimated)
- Factor source quality assessment
- Pedigree matrix integration
- Composite DQI calculation
- Quality rating labels (Excellent/Good/Fair/Poor)

References:
- GHG Protocol: Corporate Value Chain (Scope 3) Standard
- Quantis International: "Data Quality in LCA"
- ecoinvent Data Quality Guidelines

Version: 1.0.0
Date: 2025-10-30
"""

import logging
from typing import Dict, Optional, Tuple, List
from datetime import datetime

from .models import DQIScore, PedigreeScore
from .constants import (
    DQI_QUALITY_LABELS,
    FACTOR_SOURCE_SCORES,
    PEDIGREE_TO_DQI_MAPPING,
    DQI_SCORE_MIN,
    DQI_SCORE_MAX,
)
from .config import get_config
from .pedigree_matrix import PedigreeMatrixEvaluator

logger = logging.getLogger(__name__)


# ============================================================================
# DATA QUALITY INDEX CALCULATOR
# ============================================================================

class DQICalculator:
    """
    Calculator for Data Quality Index (DQI) scores.

    DQI provides a standardized 0-100 score representing overall data quality,
    combining multiple assessment dimensions:
    - Pedigree matrix scores (5 dimensions)
    - Factor source quality
    - Data tier (primary/secondary/estimated)
    """

    def __init__(self):
        """Initialize DQI Calculator."""
        self.config = get_config()
        self.pedigree_evaluator = PedigreeMatrixEvaluator()
        logger.info("Initialized DQICalculator")

    def get_quality_label(self, dqi_score: float) -> str:
        """
        Get quality label for a DQI score.

        Args:
            dqi_score: DQI score (0-100)

        Returns:
            Quality label: "Excellent", "Good", "Fair", or "Poor"
        """
        for label, (min_score, max_score) in DQI_QUALITY_LABELS.items():
            if min_score <= dqi_score <= max_score:
                return label

        return "Poor"

    def get_source_quality_score(self, source: str) -> float:
        """
        Get quality score for an emission factor source.

        Args:
            source: Source name (e.g., "ecoinvent", "defra", "primary_measured")

        Returns:
            Source quality score (0-100)
        """
        source_lower = source.lower().replace(" ", "_")

        # Try exact match first
        if source_lower in FACTOR_SOURCE_SCORES:
            score = FACTOR_SOURCE_SCORES[source_lower]
            logger.debug(f"Source quality score for '{source}': {score}")
            return score

        # Try partial matches
        for key, value in FACTOR_SOURCE_SCORES.items():
            if key in source_lower or source_lower in key:
                logger.debug(
                    f"Source quality score for '{source}' (matched '{key}'): {value}"
                )
                return value

        # Default to unknown
        logger.warning(
            f"Unknown source '{source}', using default score: "
            f"{FACTOR_SOURCE_SCORES['unknown']}"
        )
        return FACTOR_SOURCE_SCORES["unknown"]

    def get_tier_score(self, tier: int) -> float:
        """
        Get quality score for a data tier.

        Args:
            tier: Data tier (1=primary, 2=secondary, 3=estimated)

        Returns:
            Tier quality score (0-100)
        """
        tier_scores = {
            1: 100.0,  # Primary data (supplier-specific, measured)
            2: 80.0,   # Secondary data (database averages)
            3: 50.0,   # Estimated data (proxies, extrapolations)
        }

        if tier not in tier_scores:
            logger.warning(f"Invalid tier {tier}, using tier 3 score")
            return tier_scores[3]

        score = tier_scores[tier]
        logger.debug(f"Tier {tier} quality score: {score}")
        return score

    def pedigree_to_dqi_score(
        self, pedigree: PedigreeScore, use_interpolation: bool = True
    ) -> float:
        """
        Convert pedigree scores to DQI score component.

        Args:
            pedigree: Pedigree matrix scores
            use_interpolation: Use linear interpolation between mapping points

        Returns:
            DQI score component (0-100)
        """
        avg_score = pedigree.average_score

        if not use_interpolation:
            # Use direct formula
            dqi = 100.0 * (6.0 - avg_score) / 4.0
            return max(0.0, min(100.0, dqi))

        # Use interpolation with mapping table
        # Find surrounding points in mapping
        mapping_scores = sorted(PEDIGREE_TO_DQI_MAPPING.keys())

        # Exact match
        if avg_score in mapping_scores:
            return PEDIGREE_TO_DQI_MAPPING[avg_score]

        # Find interpolation points
        lower = None
        upper = None

        for score in mapping_scores:
            if score < avg_score:
                lower = score
            elif score > avg_score:
                upper = score
                break

        # Handle edge cases
        if lower is None:
            return PEDIGREE_TO_DQI_MAPPING[mapping_scores[0]]
        if upper is None:
            return PEDIGREE_TO_DQI_MAPPING[mapping_scores[-1]]

        # Linear interpolation
        lower_dqi = PEDIGREE_TO_DQI_MAPPING[lower]
        upper_dqi = PEDIGREE_TO_DQI_MAPPING[upper]

        # Interpolation formula
        fraction = (avg_score - lower) / (upper - lower)
        interpolated_dqi = lower_dqi + fraction * (upper_dqi - lower_dqi)

        logger.debug(
            f"Pedigree to DQI interpolation: avg_score={avg_score:.2f}, "
            f"lower={lower}→{lower_dqi}, upper={upper}→{upper_dqi}, "
            f"result={interpolated_dqi:.2f}"
        )

        return interpolated_dqi

    def calculate_composite_dqi(
        self,
        pedigree_score: Optional[PedigreeScore] = None,
        factor_source: Optional[str] = None,
        data_tier: Optional[int] = None,
        custom_weights: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Calculate composite DQI score from multiple components.

        DQI = w_p × DQI_pedigree + w_s × DQI_source + w_t × DQI_tier

        Args:
            pedigree_score: Pedigree matrix scores (optional)
            factor_source: Emission factor source (optional)
            data_tier: Data tier 1-3 (optional)
            custom_weights: Custom weights (default from config)

        Returns:
            Composite DQI score (0-100)
        """
        # Get weights
        if custom_weights:
            w_pedigree = custom_weights.get("pedigree", 0.5)
            w_source = custom_weights.get("source", 0.3)
            w_tier = custom_weights.get("tier", 0.2)
        else:
            w_pedigree = self.config.dqi.pedigree_weight
            w_source = self.config.dqi.source_weight
            w_tier = self.config.dqi.tier_weight

        # Calculate component scores
        components = []
        weights = []

        if pedigree_score is not None:
            pedigree_dqi = self.pedigree_to_dqi_score(
                pedigree_score, self.config.dqi.use_pedigree_interpolation
            )
            components.append(pedigree_dqi)
            weights.append(w_pedigree)
            logger.debug(f"Pedigree DQI component: {pedigree_dqi:.2f} (weight={w_pedigree})")

        if factor_source is not None:
            source_dqi = self.get_source_quality_score(factor_source)
            components.append(source_dqi)
            weights.append(w_source)
            logger.debug(f"Source DQI component: {source_dqi:.2f} (weight={w_source})")

        if data_tier is not None:
            tier_dqi = self.get_tier_score(data_tier)

            # Apply tier penalty if configured
            if self.config.dqi.apply_tier_penalty and data_tier > 1:
                penalty = (data_tier - 1) * 10  # 10 point penalty per tier
                tier_dqi = max(0.0, tier_dqi - penalty)
                logger.debug(f"Applied tier penalty: -{penalty} points")

            components.append(tier_dqi)
            weights.append(w_tier)
            logger.debug(f"Tier DQI component: {tier_dqi:.2f} (weight={w_tier})")

        # Calculate weighted average
        if not components:
            logger.warning("No DQI components provided, returning 0")
            return 0.0

        # Normalize weights to sum to 1.0
        total_weight = sum(weights)
        if total_weight == 0:
            logger.warning("Total weight is 0, returning 0")
            return 0.0

        normalized_weights = [w / total_weight for w in weights]

        # Calculate composite DQI
        composite_dqi = sum(
            comp * weight for comp, weight in zip(components, normalized_weights)
        )

        # Clamp to valid range
        composite_dqi = max(DQI_SCORE_MIN, min(DQI_SCORE_MAX, composite_dqi))

        logger.info(
            f"Calculated composite DQI: {composite_dqi:.2f} "
            f"(components={len(components)}, total_weight={total_weight:.2f})"
        )

        return composite_dqi

    def calculate_dqi(
        self,
        pedigree_score: Optional[PedigreeScore] = None,
        factor_source: Optional[str] = None,
        data_tier: Optional[int] = None,
        assessed_by: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> DQIScore:
        """
        Calculate complete DQI score with all components.

        Args:
            pedigree_score: Pedigree matrix scores
            factor_source: Emission factor source
            data_tier: Data tier (1-3)
            assessed_by: Assessor identifier
            notes: Additional notes

        Returns:
            DQIScore object with complete assessment

        Example:
            >>> calculator = DQICalculator()
            >>> pedigree = PedigreeScore(
            ...     reliability=1, completeness=2, temporal=1,
            ...     geographical=2, technological=1
            ... )
            >>> dqi = calculator.calculate_dqi(
            ...     pedigree_score=pedigree,
            ...     factor_source="ecoinvent",
            ...     data_tier=1
            ... )
            >>> dqi.score
            92.5
        """
        # Calculate component contributions
        pedigree_contribution = 0.0
        source_contribution = 0.0
        tier_contribution = 0.0

        if pedigree_score is not None:
            pedigree_contribution = self.pedigree_to_dqi_score(
                pedigree_score, self.config.dqi.use_pedigree_interpolation
            )

        if factor_source is not None:
            source_contribution = self.get_source_quality_score(factor_source)

        if data_tier is not None:
            tier_contribution = self.get_tier_score(data_tier)
            if self.config.dqi.apply_tier_penalty and data_tier > 1:
                penalty = (data_tier - 1) * 10
                tier_contribution = max(0.0, tier_contribution - penalty)

        # Calculate composite score
        composite_score = self.calculate_composite_dqi(
            pedigree_score=pedigree_score,
            factor_source=factor_source,
            data_tier=data_tier,
        )

        # Get quality label
        quality_label = self.get_quality_label(composite_score)

        # Create DQIScore object
        dqi_score = DQIScore(
            score=composite_score,
            quality_label=quality_label,
            pedigree_contribution=pedigree_contribution,
            source_contribution=source_contribution,
            tier_contribution=tier_contribution,
            pedigree_score=pedigree_score,
            factor_source=factor_source,
            data_tier=data_tier,
            assessed_at=datetime.utcnow(),
            assessed_by=assessed_by,
            notes=notes,
        )

        logger.info(
            f"Calculated DQI: score={composite_score:.2f}, "
            f"quality={quality_label}, "
            f"components=[pedigree={pedigree_contribution:.2f}, "
            f"source={source_contribution:.2f}, tier={tier_contribution:.2f}]"
        )

        return dqi_score

    def generate_dqi_report(self, dqi_score: DQIScore) -> Dict[str, any]:
        """
        Generate comprehensive DQI assessment report.

        Args:
            dqi_score: DQI score object

        Returns:
            Dictionary with detailed DQI information
        """
        report = {
            "dqi_score": round(dqi_score.score, 2),
            "quality_label": dqi_score.quality_label,
            "components": {
                "pedigree": round(dqi_score.pedigree_contribution, 2),
                "source": round(dqi_score.source_contribution, 2),
                "tier": round(dqi_score.tier_contribution, 2),
            },
            "weights": {
                "pedigree": self.config.dqi.pedigree_weight,
                "source": self.config.dqi.source_weight,
                "tier": self.config.dqi.tier_weight,
            },
            "metadata": {
                "factor_source": dqi_score.factor_source,
                "data_tier": dqi_score.data_tier,
                "assessed_at": dqi_score.assessed_at.isoformat(),
                "assessed_by": dqi_score.assessed_by,
            },
        }

        # Add pedigree details if available
        if dqi_score.pedigree_score is not None:
            report["pedigree_details"] = {
                "scores": dqi_score.pedigree_score.to_dict(),
                "average": round(dqi_score.pedigree_score.average_score, 2),
                "quality_label": dqi_score.pedigree_score.quality_label,
            }

        # Add improvement recommendations
        recommendations = []
        if dqi_score.score < self.config.dqi.good_threshold:
            recommendations.append(
                "Consider obtaining higher quality data (primary or supplier-specific)"
            )
        if dqi_score.data_tier and dqi_score.data_tier > 1:
            recommendations.append(
                f"Upgrade from Tier {dqi_score.data_tier} to Tier 1 (primary) data"
            )
        if dqi_score.pedigree_score:
            worst_score = max(dqi_score.pedigree_score.to_dict().values())
            if worst_score >= 4:
                recommendations.append(
                    "Improve low-scoring pedigree dimensions (scores of 4-5)"
                )

        report["recommendations"] = recommendations

        logger.debug(f"Generated DQI report: {len(recommendations)} recommendations")

        return report

    def compare_dqi(self, dqi1: DQIScore, dqi2: DQIScore) -> Dict[str, any]:
        """
        Compare two DQI scores.

        Args:
            dqi1: First DQI score
            dqi2: Second DQI score

        Returns:
            Comparison report
        """
        comparison = {
            "scores": {
                "dqi1": round(dqi1.score, 2),
                "dqi2": round(dqi2.score, 2),
                "difference": round(dqi2.score - dqi1.score, 2),
                "percent_change": round(
                    ((dqi2.score - dqi1.score) / dqi1.score * 100) if dqi1.score > 0 else 0,
                    2
                ),
            },
            "quality_labels": {
                "dqi1": dqi1.quality_label,
                "dqi2": dqi2.quality_label,
                "improved": dqi2.quality_label != dqi1.quality_label
                and DQI_QUALITY_LABELS[dqi2.quality_label][0]
                > DQI_QUALITY_LABELS[dqi1.quality_label][0],
            },
            "components": {
                "pedigree_change": round(
                    dqi2.pedigree_contribution - dqi1.pedigree_contribution, 2
                ),
                "source_change": round(
                    dqi2.source_contribution - dqi1.source_contribution, 2
                ),
                "tier_change": round(dqi2.tier_contribution - dqi1.tier_contribution, 2),
            },
        }

        return comparison


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def calculate_dqi(
    pedigree_score: Optional[PedigreeScore] = None,
    factor_source: Optional[str] = None,
    data_tier: Optional[int] = None,
) -> DQIScore:
    """
    Quick DQI calculation.

    Args:
        pedigree_score: Pedigree matrix scores
        factor_source: Emission factor source
        data_tier: Data tier (1-3)

    Returns:
        DQIScore
    """
    calculator = DQICalculator()
    return calculator.calculate_dqi(pedigree_score, factor_source, data_tier)


def assess_factor_quality(source: str, tier: int = 2) -> float:
    """
    Quick quality assessment for an emission factor.

    Args:
        source: Source name
        tier: Data tier (default: 2)

    Returns:
        DQI score (0-100)
    """
    calculator = DQICalculator()
    return calculator.calculate_composite_dqi(factor_source=source, data_tier=tier)


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    "DQICalculator",
    "calculate_dqi",
    "assess_factor_quality",
]
