# -*- coding: utf-8 -*-
"""
ILCD Pedigree Matrix Implementation

This module implements the ILCD (International Reference Life Cycle Data System)
Pedigree Matrix for data quality assessment in Life Cycle Assessment (LCA).

The Pedigree Matrix evaluates data quality across 5 dimensions:
1. Reliability: Verification and source quality
2. Completeness: Sample size and representativeness
3. Temporal: Age of data relative to reference year
4. Geographical: Geographic representativeness
5. Technological: Technology representativeness

Each dimension is scored on a 1-5 scale (1=excellent, 5=poor), and these scores
are converted to uncertainty factors that can be used in Monte Carlo simulations
and uncertainty propagation.

References:
- ILCD Handbook (2010), Chapter 5: Data Quality Guidelines
- ecoinvent methodology v3.9
- Weidema et al. (2013): "The Pedigree Approach to Data Quality"

Version: 1.0.0
Date: 2025-10-30
"""

import math
import logging
from typing import Dict, Optional, Tuple, List
from datetime import datetime

from .models import PedigreeScore, UncertaintyResult
from greenlang.utilities.determinism import DeterministicClock
from .constants import (
    PedigreeIndicator,
    PEDIGREE_MATRIX,
    PEDIGREE_DESCRIPTIONS,
    RELIABILITY_FACTORS,
    COMPLETENESS_FACTORS,
    TEMPORAL_FACTORS,
    GEOGRAPHICAL_FACTORS,
    TECHNOLOGICAL_FACTORS,
    PEDIGREE_SCORE_MIN,
    PEDIGREE_SCORE_MAX,
    DistributionType,
)
from .config import get_config

logger = logging.getLogger(__name__)


# ============================================================================
# PEDIGREE MATRIX EVALUATOR
# ============================================================================

class PedigreeMatrixEvaluator:
    """
    Evaluator for ILCD Pedigree Matrix data quality assessment.

    This class provides methods to:
    - Validate pedigree scores
    - Calculate uncertainty factors from pedigree scores
    - Convert pedigree scores to DQI scores
    - Assess temporal correlation automatically
    - Generate comprehensive quality reports
    """

    def __init__(self):
        """Initialize the Pedigree Matrix Evaluator."""
        self.config = get_config()
        logger.info("Initialized PedigreeMatrixEvaluator")

    def validate_score(self, score: int, dimension: str) -> bool:
        """
        Validate a pedigree score.

        Args:
            score: Score value (1-5)
            dimension: Dimension name for error messages

        Returns:
            True if valid

        Raises:
            ValueError: If score is invalid
        """
        if not isinstance(score, int):
            raise ValueError(
                f"{dimension} score must be an integer, got {type(score).__name__}"
            )

        if not (PEDIGREE_SCORE_MIN <= score <= PEDIGREE_SCORE_MAX):
            raise ValueError(
                f"{dimension} score must be between {PEDIGREE_SCORE_MIN} and "
                f"{PEDIGREE_SCORE_MAX}, got {score}"
            )

        return True

    def validate_pedigree_score(self, pedigree: PedigreeScore) -> bool:
        """
        Validate all dimensions of a pedigree score.

        Args:
            pedigree: PedigreeScore instance

        Returns:
            True if all scores are valid

        Raises:
            ValueError: If any score is invalid
        """
        self.validate_score(pedigree.reliability, "Reliability")
        self.validate_score(pedigree.completeness, "Completeness")
        self.validate_score(pedigree.temporal, "Temporal")
        self.validate_score(pedigree.geographical, "Geographical")
        self.validate_score(pedigree.technological, "Technological")

        logger.debug(f"Validated pedigree score: {pedigree.to_dict()}")
        return True

    def assess_temporal_score(
        self, data_year: int, reference_year: Optional[int] = None
    ) -> int:
        """
        Automatically assess temporal correlation score based on data age.

        Args:
            data_year: Year when data was collected
            reference_year: Reference year (defaults to config or current year)

        Returns:
            Temporal score (1-5)

        Example:
            >>> evaluator = PedigreeMatrixEvaluator()
            >>> evaluator.assess_temporal_score(2023, 2024)  # 1 year old
            1
            >>> evaluator.assess_temporal_score(2015, 2024)  # 9 years old
            3
        """
        if reference_year is None:
            if self.config.pedigree.enable_temporal_adjustment:
                reference_year = self.config.pedigree.reference_year
            else:
                reference_year = DeterministicClock.now().year

        age = abs(reference_year - data_year)

        # Apply ILCD temporal correlation scoring
        if age < 3:
            score = 1
        elif age < 6:
            score = 2
        elif age < 10:
            score = 3
        elif age < 15:
            score = 4
        else:
            score = 5

        logger.debug(
            f"Assessed temporal score: age={age} years, score={score}, "
            f"data_year={data_year}, reference_year={reference_year}"
        )

        return score

    def get_uncertainty_factor(
        self, dimension: PedigreeIndicator, score: int
    ) -> float:
        """
        Get the uncertainty factor for a specific dimension and score.

        Args:
            dimension: Pedigree dimension (reliability, completeness, etc.)
            score: Quality score (1-5)

        Returns:
            Uncertainty factor (geometric standard deviation)

        Example:
            >>> evaluator = PedigreeMatrixEvaluator()
            >>> evaluator.get_uncertainty_factor(
            ...     PedigreeIndicator.RELIABILITY, 1
            ... )
            1.0
        """
        if dimension not in PEDIGREE_MATRIX:
            raise ValueError(f"Invalid pedigree dimension: {dimension}")

        factors = PEDIGREE_MATRIX[dimension]
        if score not in factors:
            raise ValueError(
                f"Invalid score {score} for dimension {dimension}. "
                f"Must be between {PEDIGREE_SCORE_MIN} and {PEDIGREE_SCORE_MAX}"
            )

        factor = factors[score]
        logger.debug(f"Uncertainty factor for {dimension}[{score}] = {factor}")

        return factor

    def get_dimension_description(
        self, dimension: PedigreeIndicator, score: int
    ) -> str:
        """
        Get the description for a specific dimension and score.

        Args:
            dimension: Pedigree dimension
            score: Quality score (1-5)

        Returns:
            Human-readable description of the score level
        """
        if dimension not in PEDIGREE_DESCRIPTIONS:
            raise ValueError(f"Invalid pedigree dimension: {dimension}")

        descriptions = PEDIGREE_DESCRIPTIONS[dimension]
        if score not in descriptions:
            raise ValueError(f"Invalid score {score} for dimension {dimension}")

        return descriptions[score]

    def calculate_combined_uncertainty(
        self, pedigree: PedigreeScore, base_uncertainty: float = 0.0
    ) -> float:
        """
        Calculate combined uncertainty from all pedigree dimensions.

        The combined uncertainty is calculated using geometric mean of
        uncertainty factors (multiplicative model).

        Formula:
            σ_combined = σ_base × Π(f_i)
            where f_i is the uncertainty factor for dimension i

        Args:
            pedigree: Pedigree scores for all dimensions
            base_uncertainty: Base uncertainty before pedigree adjustment (default 0.0)

        Returns:
            Combined relative uncertainty (coefficient of variation)

        Example:
            >>> evaluator = PedigreeMatrixEvaluator()
            >>> pedigree = PedigreeScore(
            ...     reliability=1, completeness=2, temporal=1,
            ...     geographical=2, technological=1
            ... )
            >>> evaluator.calculate_combined_uncertainty(pedigree, 0.1)
            0.10824...
        """
        self.validate_pedigree_score(pedigree)

        # Get uncertainty factors for each dimension
        reliability_factor = self.get_uncertainty_factor(
            PedigreeIndicator.RELIABILITY, pedigree.reliability
        )
        completeness_factor = self.get_uncertainty_factor(
            PedigreeIndicator.COMPLETENESS, pedigree.completeness
        )
        temporal_factor = self.get_uncertainty_factor(
            PedigreeIndicator.TEMPORAL, pedigree.temporal
        )
        geographical_factor = self.get_uncertainty_factor(
            PedigreeIndicator.GEOGRAPHICAL, pedigree.geographical
        )
        technological_factor = self.get_uncertainty_factor(
            PedigreeIndicator.TECHNOLOGICAL, pedigree.technological
        )

        # Calculate combined factor (multiplicative model)
        combined_factor = (
            reliability_factor
            * completeness_factor
            * temporal_factor
            * geographical_factor
            * technological_factor
        )

        # Apply to base uncertainty
        # If base_uncertainty is 0, use the combined factor as the uncertainty
        if base_uncertainty == 0.0:
            combined_uncertainty = combined_factor - 1.0  # Convert factor to CV
        else:
            combined_uncertainty = base_uncertainty * combined_factor

        # Apply floor and ceiling if configured
        if self.config.uncertainty.apply_floor:
            combined_uncertainty = max(
                combined_uncertainty, self.config.uncertainty.min_uncertainty
            )
        if self.config.uncertainty.apply_ceiling:
            combined_uncertainty = min(
                combined_uncertainty, self.config.uncertainty.max_uncertainty
            )

        logger.info(
            f"Combined uncertainty: base={base_uncertainty:.4f}, "
            f"factor={combined_factor:.4f}, combined={combined_uncertainty:.4f}"
        )

        return combined_uncertainty

    def calculate_dimension_uncertainties(
        self, pedigree: PedigreeScore
    ) -> Dict[str, float]:
        """
        Calculate uncertainty contribution from each dimension.

        Args:
            pedigree: Pedigree scores for all dimensions

        Returns:
            Dictionary mapping dimension names to uncertainty factors
        """
        self.validate_pedigree_score(pedigree)

        uncertainties = {
            "reliability": self.get_uncertainty_factor(
                PedigreeIndicator.RELIABILITY, pedigree.reliability
            ),
            "completeness": self.get_uncertainty_factor(
                PedigreeIndicator.COMPLETENESS, pedigree.completeness
            ),
            "temporal": self.get_uncertainty_factor(
                PedigreeIndicator.TEMPORAL, pedigree.temporal
            ),
            "geographical": self.get_uncertainty_factor(
                PedigreeIndicator.GEOGRAPHICAL, pedigree.geographical
            ),
            "technological": self.get_uncertainty_factor(
                PedigreeIndicator.TECHNOLOGICAL, pedigree.technological
            ),
        }

        return uncertainties

    def pedigree_to_uncertainty_result(
        self, mean: float, pedigree: PedigreeScore, base_uncertainty: float = 0.0
    ) -> UncertaintyResult:
        """
        Convert pedigree scores to a full uncertainty result.

        Args:
            mean: Mean value of the parameter
            pedigree: Pedigree scores
            base_uncertainty: Base uncertainty (default 0.0)

        Returns:
            UncertaintyResult with calculated uncertainty bounds

        Example:
            >>> evaluator = PedigreeMatrixEvaluator()
            >>> pedigree = PedigreeScore(
            ...     reliability=1, completeness=2, temporal=1,
            ...     geographical=2, technological=1
            ... )
            >>> result = evaluator.pedigree_to_uncertainty_result(
            ...     1000.0, pedigree, 0.1
            ... )
            >>> result.mean
            1000.0
        """
        # Calculate combined relative uncertainty
        relative_std_dev = self.calculate_combined_uncertainty(
            pedigree, base_uncertainty
        )

        # Calculate absolute standard deviation
        std_dev = abs(mean * relative_std_dev)

        # For lognormal distribution (typical for environmental data)
        # Calculate 95% confidence interval
        # Using lognormal: ln(x) ~ N(μ, σ²)
        # μ = ln(mean) - 0.5 * σ²
        # σ = sqrt(ln(1 + CV²))

        if mean > 0:
            cv = relative_std_dev
            sigma_ln = math.sqrt(math.log(1 + cv**2))
            mu_ln = math.log(mean) - 0.5 * sigma_ln**2

            # 95% CI for lognormal
            confidence_95_lower = math.exp(mu_ln - 1.96 * sigma_ln)
            confidence_95_upper = math.exp(mu_ln + 1.96 * sigma_ln)

            # 90% CI for lognormal
            confidence_90_lower = math.exp(mu_ln - 1.645 * sigma_ln)
            confidence_90_upper = math.exp(mu_ln + 1.645 * sigma_ln)

            # Median for lognormal
            median = math.exp(mu_ln)
        else:
            # For zero or negative values, use normal distribution
            confidence_95_lower = mean - 1.96 * std_dev
            confidence_95_upper = mean + 1.96 * std_dev
            confidence_90_lower = mean - 1.645 * std_dev
            confidence_90_upper = mean + 1.645 * std_dev
            median = mean

        # Create uncertainty result
        result = UncertaintyResult(
            mean=mean,
            median=median,
            std_dev=std_dev,
            relative_std_dev=relative_std_dev,
            variance=std_dev**2,
            confidence_95_lower=max(0, confidence_95_lower),  # Emissions can't be negative
            confidence_95_upper=confidence_95_upper,
            confidence_90_lower=max(0, confidence_90_lower),
            confidence_90_upper=confidence_90_upper,
            distribution_type=DistributionType.LOGNORMAL,
            distribution_params={
                "mu": mu_ln if mean > 0 else mean,
                "sigma": sigma_ln if mean > 0 else relative_std_dev,
            },
            uncertainty_sources=["pedigree_matrix"],
            pedigree_score=pedigree,
            methodology="pedigree_matrix",
        )

        logger.info(
            f"Generated uncertainty result: mean={mean:.2f}, "
            f"std_dev={std_dev:.2f}, CV={relative_std_dev:.4f}"
        )

        return result

    def pedigree_to_dqi(self, pedigree: PedigreeScore) -> float:
        """
        Convert pedigree scores to a Data Quality Index (DQI) score.

        The DQI is calculated on a 0-100 scale, where higher is better.
        Conversion formula: DQI = 100 × (6 - avg_score) / 4

        Args:
            pedigree: Pedigree scores

        Returns:
            DQI score (0-100)

        Example:
            >>> evaluator = PedigreeMatrixEvaluator()
            >>> pedigree = PedigreeScore(
            ...     reliability=1, completeness=2, temporal=1,
            ...     geographical=2, technological=1
            ... )
            >>> evaluator.pedigree_to_dqi(pedigree)
            92.5
        """
        self.validate_pedigree_score(pedigree)

        # Calculate average pedigree score (1-5)
        avg_score = pedigree.average_score

        # Convert to DQI (0-100)
        # Score of 1.0 (perfect) -> DQI of 100
        # Score of 5.0 (worst) -> DQI of 0
        dqi = 100.0 * (6.0 - avg_score) / 4.0

        # Clamp to valid range
        dqi = max(0.0, min(100.0, dqi))

        logger.debug(f"Converted pedigree to DQI: avg_score={avg_score:.2f}, DQI={dqi:.2f}")

        return dqi

    def generate_quality_report(
        self, pedigree: PedigreeScore
    ) -> Dict[str, any]:
        """
        Generate a comprehensive quality assessment report.

        Args:
            pedigree: Pedigree scores

        Returns:
            Dictionary with detailed quality information
        """
        self.validate_pedigree_score(pedigree)

        # Calculate metrics
        avg_score = pedigree.average_score
        dqi = self.pedigree_to_dqi(pedigree)
        combined_uncertainty = self.calculate_combined_uncertainty(pedigree)
        dimension_uncertainties = self.calculate_dimension_uncertainties(pedigree)

        # Get descriptions for each dimension
        descriptions = {
            "reliability": self.get_dimension_description(
                PedigreeIndicator.RELIABILITY, pedigree.reliability
            ),
            "completeness": self.get_dimension_description(
                PedigreeIndicator.COMPLETENESS, pedigree.completeness
            ),
            "temporal": self.get_dimension_description(
                PedigreeIndicator.TEMPORAL, pedigree.temporal
            ),
            "geographical": self.get_dimension_description(
                PedigreeIndicator.GEOGRAPHICAL, pedigree.geographical
            ),
            "technological": self.get_dimension_description(
                PedigreeIndicator.TECHNOLOGICAL, pedigree.technological
            ),
        }

        # Identify worst dimensions (highest scores)
        scores_dict = pedigree.to_dict()
        sorted_dimensions = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
        worst_dimensions = [
            dim for dim, score in sorted_dimensions if score >= 3
        ]

        report = {
            "pedigree_scores": scores_dict,
            "average_score": round(avg_score, 2),
            "quality_label": pedigree.quality_label,
            "dqi_score": round(dqi, 2),
            "combined_uncertainty": round(combined_uncertainty, 4),
            "dimension_uncertainties": {
                k: round(v, 4) for k, v in dimension_uncertainties.items()
            },
            "dimension_descriptions": descriptions,
            "improvement_opportunities": worst_dimensions,
            "assessment_timestamp": DeterministicClock.utcnow().isoformat(),
        }

        logger.info(f"Generated quality report: quality={pedigree.quality_label}, DQI={dqi:.2f}")

        return report


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_pedigree_score(
    reliability: int,
    completeness: int,
    temporal: int,
    geographical: int,
    technological: int,
    **kwargs,
) -> PedigreeScore:
    """
    Create a validated PedigreeScore instance.

    Args:
        reliability: Reliability score (1-5)
        completeness: Completeness score (1-5)
        temporal: Temporal correlation score (1-5)
        geographical: Geographical correlation score (1-5)
        technological: Technological correlation score (1-5)
        **kwargs: Additional metadata (reference_year, data_year, notes)

    Returns:
        PedigreeScore instance

    Example:
        >>> score = create_pedigree_score(1, 2, 1, 2, 1)
        >>> score.average_score
        1.4
    """
    evaluator = PedigreeMatrixEvaluator()

    score = PedigreeScore(
        reliability=reliability,
        completeness=completeness,
        temporal=temporal,
        geographical=geographical,
        technological=technological,
        **kwargs,
    )

    evaluator.validate_pedigree_score(score)
    return score


def assess_data_quality(
    reliability: int,
    completeness: int,
    temporal: int,
    geographical: int,
    technological: int,
) -> Dict[str, any]:
    """
    Quick data quality assessment.

    Args:
        reliability: Reliability score (1-5)
        completeness: Completeness score (1-5)
        temporal: Temporal correlation score (1-5)
        geographical: Geographical correlation score (1-5)
        technological: Technological correlation score (1-5)

    Returns:
        Quality assessment report
    """
    evaluator = PedigreeMatrixEvaluator()
    pedigree = create_pedigree_score(
        reliability, completeness, temporal, geographical, technological
    )
    return evaluator.generate_quality_report(pedigree)


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    "PedigreeMatrixEvaluator",
    "create_pedigree_score",
    "assess_data_quality",
]
