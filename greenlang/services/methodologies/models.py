# -*- coding: utf-8 -*-
"""
Methodologies Data Models

Pydantic models for uncertainty quantification, data quality assessment,
and Monte Carlo simulation results.

Key Models:
- PedigreeScore: ILCD Pedigree Matrix scores (5 dimensions)
- DQIScore: Data Quality Index assessment
- UncertaintyResult: Uncertainty quantification results
- MonteCarloResult: Monte Carlo simulation outputs

References:
- ILCD Handbook (2010)
- GHG Protocol uncertainty guidance
- ISO 14044:2006

Version: 1.0.0
Date: 2025-10-30
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator, model_validator
from enum import Enum

from .constants import (
    PedigreeIndicator,
    GWPVersion,
    DistributionType,
    PEDIGREE_SCORE_MIN,
    PEDIGREE_SCORE_MAX,
    DQI_SCORE_MIN,
    DQI_SCORE_MAX,
    UNCERTAINTY_MIN,
    UNCERTAINTY_MAX,
    MC_MIN_ITERATIONS,
    MC_MAX_ITERATIONS,
)


# ============================================================================
# PEDIGREE SCORE MODELS
# ============================================================================

class PedigreeScore(BaseModel):
    """
    ILCD Pedigree Matrix score for a single data point.

    The Pedigree Matrix assesses data quality across 5 dimensions:
    1. Reliability: Verification and source quality
    2. Completeness: Sample size and representativeness
    3. Temporal: Age of data relative to reference year
    4. Geographical: Geographic representativeness
    5. Technological: Technology representativeness

    Each dimension is scored on a 1-5 scale (1=excellent, 5=poor).

    Example:
        >>> score = PedigreeScore(
        ...     reliability=1,
        ...     completeness=2,
        ...     temporal=1,
        ...     geographical=2,
        ...     technological=1
        ... )
        >>> score.average_score
        1.4
    """

    reliability: int = Field(
        ...,
        ge=PEDIGREE_SCORE_MIN,
        le=PEDIGREE_SCORE_MAX,
        description="Reliability score (1=verified measurements, 5=non-qualified estimate)",
    )
    completeness: int = Field(
        ...,
        ge=PEDIGREE_SCORE_MIN,
        le=PEDIGREE_SCORE_MAX,
        description="Completeness score (1=sufficient sample/period, 5=unknown)",
    )
    temporal: int = Field(
        ...,
        ge=PEDIGREE_SCORE_MIN,
        le=PEDIGREE_SCORE_MAX,
        description="Temporal correlation (1=<3 years old, 5=>15 years or unknown)",
    )
    geographical: int = Field(
        ...,
        ge=PEDIGREE_SCORE_MIN,
        le=PEDIGREE_SCORE_MAX,
        description="Geographical correlation (1=same area, 5=unknown/different area)",
    )
    technological: int = Field(
        ...,
        ge=PEDIGREE_SCORE_MIN,
        le=PEDIGREE_SCORE_MAX,
        description="Technological correlation (1=same technology, 5=different technology)",
    )

    # Metadata
    reference_year: Optional[int] = Field(
        None, description="Reference year for temporal assessment"
    )
    data_year: Optional[int] = Field(None, description="Year of data collection")
    notes: Optional[str] = Field(None, description="Additional notes on data quality")

    @property
    def average_score(self) -> float:
        """Calculate average pedigree score across all dimensions."""
        return (
            self.reliability
            + self.completeness
            + self.temporal
            + self.geographical
            + self.technological
        ) / 5.0

    @property
    def quality_label(self) -> str:
        """Get quality label based on average score."""
        avg = self.average_score
        if avg <= 1.5:
            return "Excellent"
        elif avg <= 2.5:
            return "Good"
        elif avg <= 3.5:
            return "Fair"
        else:
            return "Poor"

    def to_dict(self) -> Dict[str, int]:
        """Convert scores to dictionary format."""
        return {
            "reliability": self.reliability,
            "completeness": self.completeness,
            "temporal": self.temporal,
            "geographical": self.geographical,
            "technological": self.technological,
        }

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "reliability": 1,
                    "completeness": 2,
                    "temporal": 1,
                    "geographical": 2,
                    "technological": 1,
                    "reference_year": 2024,
                    "data_year": 2023,
                    "notes": "Primary data from supplier, verified measurements",
                }
            ]
        }
    }


# ============================================================================
# DATA QUALITY INDEX (DQI) MODELS
# ============================================================================

class DQIScore(BaseModel):
    """
    Data Quality Index (DQI) assessment result.

    DQI provides a composite score (0-100) representing overall data quality,
    combining pedigree matrix scores, factor source quality, and data tier.

    Example:
        >>> dqi = DQIScore(
        ...     score=85.5,
        ...     quality_label="Good",
        ...     pedigree_score=PedigreeScore(...),
        ...     factor_source="ecoinvent",
        ...     data_tier=1
        ... )
    """

    score: float = Field(
        ...,
        ge=DQI_SCORE_MIN,
        le=DQI_SCORE_MAX,
        description="Composite DQI score (0-100, higher is better)",
    )
    quality_label: str = Field(
        ..., description="Quality label: Excellent, Good, Fair, or Poor"
    )

    # Component scores
    pedigree_contribution: float = Field(
        ..., ge=0.0, le=100.0, description="Contribution from pedigree matrix (0-100)"
    )
    source_contribution: float = Field(
        ..., ge=0.0, le=100.0, description="Contribution from factor source (0-100)"
    )
    tier_contribution: float = Field(
        ..., ge=0.0, le=100.0, description="Contribution from data tier (0-100)"
    )

    # Source information
    pedigree_score: Optional[PedigreeScore] = Field(
        None, description="Associated pedigree matrix scores"
    )
    factor_source: Optional[str] = Field(
        None, description="Emission factor source (e.g., ecoinvent, DEFRA)"
    )
    data_tier: Optional[int] = Field(
        None, ge=1, le=3, description="Data tier (1=primary, 2=secondary, 3=estimated)"
    )

    # Metadata
    assessed_at: datetime = Field(
        default_factory=datetime.utcnow, description="Assessment timestamp"
    )
    assessed_by: Optional[str] = Field(None, description="Assessor identifier")
    notes: Optional[str] = Field(None, description="Additional notes")

    @field_validator("quality_label")
    @classmethod
    def validate_quality_label(cls, v: str) -> str:
        """Validate quality label is one of the allowed values."""
        allowed = ["Excellent", "Good", "Fair", "Poor"]
        if v not in allowed:
            raise ValueError(f"Quality label must be one of {allowed}")
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "score": 85.5,
                    "quality_label": "Good",
                    "pedigree_contribution": 80.0,
                    "source_contribution": 90.0,
                    "tier_contribution": 100.0,
                    "factor_source": "ecoinvent",
                    "data_tier": 1,
                    "assessed_by": "system",
                }
            ]
        }
    }


# ============================================================================
# UNCERTAINTY MODELS
# ============================================================================

class UncertaintyResult(BaseModel):
    """
    Uncertainty quantification result for an emission value.

    Represents the uncertainty bounds and distribution parameters
    for a calculated emission value.

    Example:
        >>> uncertainty = UncertaintyResult(
        ...     mean=1000.0,
        ...     std_dev=150.0,
        ...     relative_std_dev=0.15,
        ...     confidence_95_lower=700.0,
        ...     confidence_95_upper=1300.0,
        ...     distribution_type="lognormal"
        ... )
    """

    # Central tendency
    mean: float = Field(..., description="Mean value (expected value)")
    median: Optional[float] = Field(None, description="Median value (p50)")

    # Dispersion measures
    std_dev: float = Field(
        ..., ge=0.0, description="Standard deviation (absolute uncertainty)"
    )
    relative_std_dev: float = Field(
        ...,
        ge=UNCERTAINTY_MIN,
        le=UNCERTAINTY_MAX,
        description="Relative standard deviation (coefficient of variation)",
    )
    variance: Optional[float] = Field(None, ge=0.0, description="Variance")

    # Confidence intervals (95% by default)
    confidence_95_lower: float = Field(..., description="95% CI lower bound (p2.5)")
    confidence_95_upper: float = Field(..., description="95% CI upper bound (p97.5)")
    confidence_90_lower: Optional[float] = Field(
        None, description="90% CI lower bound (p5)"
    )
    confidence_90_upper: Optional[float] = Field(
        None, description="90% CI upper bound (p95)"
    )

    # Distribution information
    distribution_type: DistributionType = Field(
        ..., description="Probability distribution type"
    )
    distribution_params: Optional[Dict[str, float]] = Field(
        None, description="Distribution-specific parameters"
    )

    # Source of uncertainty
    uncertainty_sources: Optional[List[str]] = Field(
        None, description="Sources contributing to uncertainty"
    )
    pedigree_score: Optional[PedigreeScore] = Field(
        None, description="Associated pedigree scores"
    )

    # Metadata
    calculated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Calculation timestamp"
    )
    methodology: str = Field(
        default="pedigree_matrix", description="Uncertainty quantification methodology"
    )

    @property
    def confidence_95_range(self) -> float:
        """Calculate 95% confidence interval range."""
        return self.confidence_95_upper - self.confidence_95_lower

    @property
    def coefficient_of_variation(self) -> float:
        """Calculate coefficient of variation (CV = std_dev / mean)."""
        if self.mean == 0:
            return 0.0
        return self.std_dev / abs(self.mean)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "mean": 1000.0,
                    "median": 985.0,
                    "std_dev": 150.0,
                    "relative_std_dev": 0.15,
                    "variance": 22500.0,
                    "confidence_95_lower": 700.0,
                    "confidence_95_upper": 1300.0,
                    "confidence_90_lower": 750.0,
                    "confidence_90_upper": 1250.0,
                    "distribution_type": "lognormal",
                    "uncertainty_sources": ["emission_factor", "activity_data"],
                    "methodology": "pedigree_matrix",
                }
            ]
        }
    }


# ============================================================================
# MONTE CARLO SIMULATION MODELS
# ============================================================================

class MonteCarloInput(BaseModel):
    """
    Input parameter for Monte Carlo simulation.

    Example:
        >>> mc_input = MonteCarloInput(
        ...     name="activity_data",
        ...     mean=1000.0,
        ...     std_dev=50.0,
        ...     distribution="normal"
        ... )
    """

    name: str = Field(..., description="Parameter name")
    mean: float = Field(..., description="Mean value")
    std_dev: float = Field(..., ge=0.0, description="Standard deviation")
    distribution: DistributionType = Field(
        default=DistributionType.LOGNORMAL, description="Distribution type"
    )
    min_value: Optional[float] = Field(None, description="Minimum value (for uniform)")
    max_value: Optional[float] = Field(None, description="Maximum value (for uniform)")
    correlation_matrix: Optional[Dict[str, float]] = Field(
        None, description="Correlations with other parameters"
    )


class MonteCarloResult(BaseModel):
    """
    Monte Carlo simulation result with statistical summary.

    Provides comprehensive statistics from Monte Carlo uncertainty propagation,
    including percentiles, confidence intervals, and distribution metrics.

    Example:
        >>> mc_result = MonteCarloResult(
        ...     iterations=10000,
        ...     mean=1000.0,
        ...     std_dev=150.0,
        ...     p5=750.0,
        ...     p50=985.0,
        ...     p95=1320.0
        ... )
    """

    # Simulation parameters
    iterations: int = Field(
        ...,
        ge=MC_MIN_ITERATIONS,
        le=MC_MAX_ITERATIONS,
        description="Number of Monte Carlo iterations",
    )
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")

    # Statistical summary
    mean: float = Field(..., description="Mean of simulation results")
    median: float = Field(..., description="Median (p50) of results")
    std_dev: float = Field(..., ge=0.0, description="Standard deviation")
    variance: float = Field(..., ge=0.0, description="Variance")

    # Percentiles
    p5: float = Field(..., description="5th percentile")
    p10: Optional[float] = Field(None, description="10th percentile")
    p25: Optional[float] = Field(None, description="25th percentile (Q1)")
    p50: float = Field(..., description="50th percentile (median)")
    p75: Optional[float] = Field(None, description="75th percentile (Q3)")
    p90: Optional[float] = Field(None, description="90th percentile")
    p95: float = Field(..., description="95th percentile")

    # Range
    min_value: float = Field(..., description="Minimum simulated value")
    max_value: float = Field(..., description="Maximum simulated value")

    # Distribution shape
    skewness: Optional[float] = Field(None, description="Skewness of distribution")
    kurtosis: Optional[float] = Field(None, description="Kurtosis of distribution")

    # Input parameters
    input_parameters: Optional[List[MonteCarloInput]] = Field(
        None, description="Input parameters used in simulation"
    )

    # Sensitivity analysis
    sensitivity_indices: Optional[Dict[str, float]] = Field(
        None, description="Sensitivity indices for each input parameter"
    )
    top_contributors: Optional[List[str]] = Field(
        None, description="Top contributing parameters to uncertainty"
    )

    # Metadata
    simulated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Simulation timestamp"
    )
    computation_time: Optional[float] = Field(
        None, ge=0.0, description="Computation time in seconds"
    )

    @property
    def coefficient_of_variation(self) -> float:
        """Calculate coefficient of variation (CV)."""
        if self.mean == 0:
            return 0.0
        return self.std_dev / abs(self.mean)

    @property
    def interquartile_range(self) -> Optional[float]:
        """Calculate interquartile range (IQR = Q3 - Q1)."""
        if self.p75 is not None and self.p25 is not None:
            return self.p75 - self.p25
        return None

    @property
    def confidence_90_range(self) -> float:
        """Calculate 90% confidence range (p95 - p5)."""
        return self.p95 - self.p5

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "iterations": 10000,
                    "seed": 42,
                    "mean": 1000.0,
                    "median": 985.0,
                    "std_dev": 150.0,
                    "variance": 22500.0,
                    "p5": 750.0,
                    "p10": 800.0,
                    "p25": 900.0,
                    "p50": 985.0,
                    "p75": 1100.0,
                    "p90": 1250.0,
                    "p95": 1320.0,
                    "min_value": 600.0,
                    "max_value": 1500.0,
                    "skewness": 0.15,
                    "kurtosis": 3.2,
                    "computation_time": 0.85,
                }
            ]
        }
    }


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    "PedigreeScore",
    "DQIScore",
    "UncertaintyResult",
    "MonteCarloInput",
    "MonteCarloResult",
]
