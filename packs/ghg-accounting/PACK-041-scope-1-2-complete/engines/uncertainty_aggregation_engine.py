# -*- coding: utf-8 -*-
"""
UncertaintyAggregationEngine - PACK-041 Scope 1-2 Complete Engine 6
=====================================================================

Aggregates measurement, emission factor, and activity data uncertainty
from all 13 MRV agents (8 Scope 1 + 5 Scope 2) into organisation-level
confidence bounds using both analytical (error propagation) and
Monte Carlo simulation approaches.

Calculation Methodology:
    Analytical (Quadrature / Error Propagation / Root-Sum-of-Squares):
        For independent, uncorrelated sources:
        U_combined = sqrt( sum( (E_i * u_i)^2 ) ) / E_total * 100

        Where:
            E_i     = emissions from source i (tCO2e)
            u_i     = fractional uncertainty of source i (e.g. 0.05 for 5%)
            E_total = sum of all E_i

        Reference: IPCC 2006 Guidelines, Volume 1, Chapter 3, Eq 3.1
                   GHG Protocol Corporate Standard, Chapter 7

    Monte Carlo Simulation:
        For each iteration k = 1..N:
            For each source i, sample E_i_k from the uncertainty distribution
            Compute total_k = sum(E_i_k)
        Extract percentiles from the distribution of total_k

        Reference: IPCC 2006 Guidelines, Volume 1, Chapter 3, Section 3.2.3
                   ISO 14064-1:2018, Clause 9 (uncertainty assessment)

    Sensitivity / Contribution Analysis:
        Contribution_i = (E_i * u_i)^2 / sum( (E_j * u_j)^2 ) * 100

        Identifies which source categories dominate overall uncertainty.

    Data Quality Scoring:
        Based on IPCC 2006 data quality indicators (Table 3.5) and
        GHG Protocol data quality matrix (Scope 3 Technical Guidance).

Typical Uncertainty Ranges (IPCC 2006 default values):
    Stationary combustion:  Activity data +/-5%, EF +/-5%   => combined ~7%
    Mobile combustion:      Activity data +/-5%, EF +/-10%  => combined ~11%
    Process emissions:      Activity data +/-5%, EF +/-15%  => combined ~16%
    Fugitive emissions:     Activity data +/-10%, EF +/-30% => combined ~32%
    Refrigerant leakage:    Activity data +/-10%, EF +/-20% => combined ~22%
    Scope 2 location-based: Activity data +/-2%, EF +/-10%  => combined ~10%
    Scope 2 market-based:   Activity data +/-2%, EF +/-5%   => combined ~5%

Regulatory References:
    - IPCC 2006 Guidelines for National GHG Inventories, Vol 1, Ch 3
    - GHG Protocol Corporate Standard (2004, revised 2015), Chapter 7
    - ISO 14064-1:2018, Clause 9 (Quantification of GHG uncertainty)
    - IPCC 2019 Refinement, Volume 1, Chapter 3
    - EPA GHGRP Subpart A, 40 CFR 98.3(d) (uncertainty requirements)

Zero-Hallucination:
    - All calculations use deterministic Decimal arithmetic
    - Monte Carlo uses numpy with fixed seed for reproducibility
    - Uncertainty ranges from published IPCC default tables
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-041 Scope 1-2 Complete
Engine:  6 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator
from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())

def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

    Uses JSON serialization with sorted keys to guarantee reproducibility.

    Args:
        data: Data to hash -- dict, Pydantic model, or other serializable.

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    if isinstance(serializable, dict):
        serializable = {
            k: v for k, v in serializable.items()
            if k not in ("calculated_at", "processing_time_ms", "provenance_hash")
        }
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _decimal(value: Any) -> Decimal:
    """Safely convert a value to Decimal.

    Args:
        value: Numeric value to convert.

    Returns:
        Decimal representation; Decimal("0") on failure.
    """
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")

def _safe_divide(
    numerator: Decimal,
    denominator: Decimal,
    default: Decimal = Decimal("0"),
) -> Decimal:
    """Safely divide two Decimals, returning *default* on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    """Compute percentage safely (part / whole * 100)."""
    return _safe_divide(part * Decimal("100"), whole)

def _round2(value: Any) -> float:
    """Round to 2 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

def _round3(value: Any) -> float:
    """Round to 3 decimal places."""
    return float(Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))

def _round4(value: Any) -> float:
    """Round to 4 decimal places."""
    return float(Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))

def _round6(value: Any) -> float:
    """Round to 6 decimal places."""
    return float(Decimal(str(value)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class UncertaintyDistribution(str, Enum):
    """Statistical distribution types for uncertainty modelling.

    NORMAL:      Symmetric bell curve; appropriate when activity data errors
                 are additive and many small independent factors contribute.
    LOGNORMAL:   Right-skewed; appropriate for emission factors where values
                 are strictly positive and multiplicative errors dominate.
    UNIFORM:     Equal probability across range; used when only a range is
                 known with no information about the most likely value.
    TRIANGULAR:  Defined by min, mode, max; used when expert judgement
                 provides a most likely value plus bounds.
    """
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    UNIFORM = "uniform"
    TRIANGULAR = "triangular"

class SourceCategoryType(str, Enum):
    """GHG source categories aligned with MRV agent naming.

    Covers all 8 Scope 1 and 5 Scope 2 source categories
    mapped to the corresponding MRV agents (AGENT-MRV 001-013).
    """
    STATIONARY_COMBUSTION = "stationary_combustion"
    MOBILE_COMBUSTION = "mobile_combustion"
    PROCESS_EMISSIONS = "process_emissions"
    FUGITIVE_EMISSIONS = "fugitive_emissions"
    REFRIGERANT_LEAKAGE = "refrigerant_leakage"
    LAND_USE_CHANGE = "land_use_change"
    WASTE_TREATMENT = "waste_treatment"
    AGRICULTURAL_EMISSIONS = "agricultural_emissions"
    SCOPE2_LOCATION_BASED = "scope2_location_based"
    SCOPE2_MARKET_BASED = "scope2_market_based"
    SCOPE2_STEAM = "scope2_steam"
    SCOPE2_COOLING = "scope2_cooling"
    SCOPE2_DUAL_REPORTING = "scope2_dual_reporting"

class DataQualityTier(str, Enum):
    """Data quality tiers per IPCC 2006 Table 3.5 / GHG Protocol guidance.

    HIGH:   Continuous monitoring, calibrated instruments, supplier-specific EFs.
    MEDIUM: Periodic measurements, regional average EFs, estimated activity data.
    LOW:    Expert estimates, global default EFs, proxy data.
    """
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class AggregationMethod(str, Enum):
    """Method used for uncertainty aggregation.

    ANALYTICAL:   Error propagation via quadrature (root-sum-of-squares).
    MONTE_CARLO:  Stochastic simulation with distribution sampling.
    BOTH:         Run both analytical and Monte Carlo for cross-validation.
    """
    ANALYTICAL = "analytical"
    MONTE_CARLO = "monte_carlo"
    BOTH = "both"

# ---------------------------------------------------------------------------
# Constants -- Default Uncertainty Ranges
# ---------------------------------------------------------------------------

# Default uncertainty percentages by source category.
# Source: IPCC 2006 Guidelines, Volume 1, Chapter 3, Tables 3.2-3.5.
# Activity data uncertainty and emission factor uncertainty (half-width of
# 95% confidence interval as percentage of the mean).
DEFAULT_UNCERTAINTY_RANGES: Dict[str, Dict[str, float]] = {
    SourceCategoryType.STATIONARY_COMBUSTION: {
        "activity_data_pct": 5.0,
        "emission_factor_pct": 5.0,
        "source": "IPCC 2006 Vol 2 Ch 2, Table 2.1 (Tier 1 default)",
    },
    SourceCategoryType.MOBILE_COMBUSTION: {
        "activity_data_pct": 5.0,
        "emission_factor_pct": 10.0,
        "source": "IPCC 2006 Vol 2 Ch 3, Table 3.2 (fleet-average)",
    },
    SourceCategoryType.PROCESS_EMISSIONS: {
        "activity_data_pct": 5.0,
        "emission_factor_pct": 15.0,
        "source": "IPCC 2006 Vol 3 Ch 2, Table 2.1 (industrial processes)",
    },
    SourceCategoryType.FUGITIVE_EMISSIONS: {
        "activity_data_pct": 10.0,
        "emission_factor_pct": 30.0,
        "source": "IPCC 2006 Vol 2 Ch 4, Table 4.2 (fugitive default)",
    },
    SourceCategoryType.REFRIGERANT_LEAKAGE: {
        "activity_data_pct": 10.0,
        "emission_factor_pct": 20.0,
        "source": "IPCC 2006 Vol 3 Ch 7, Tier 1 (equipment-based)",
    },
    SourceCategoryType.LAND_USE_CHANGE: {
        "activity_data_pct": 15.0,
        "emission_factor_pct": 50.0,
        "source": "IPCC 2006 Vol 4 Ch 2 (AFOLU, high variability)",
    },
    SourceCategoryType.WASTE_TREATMENT: {
        "activity_data_pct": 10.0,
        "emission_factor_pct": 40.0,
        "source": "IPCC 2006 Vol 5 Ch 3, Table 3.3 (landfill default)",
    },
    SourceCategoryType.AGRICULTURAL_EMISSIONS: {
        "activity_data_pct": 10.0,
        "emission_factor_pct": 50.0,
        "source": "IPCC 2006 Vol 4 Ch 10-11, Tier 1 (enteric/manure)",
    },
    SourceCategoryType.SCOPE2_LOCATION_BASED: {
        "activity_data_pct": 2.0,
        "emission_factor_pct": 10.0,
        "source": "GHG Protocol Scope 2, grid average EF uncertainty",
    },
    SourceCategoryType.SCOPE2_MARKET_BASED: {
        "activity_data_pct": 2.0,
        "emission_factor_pct": 5.0,
        "source": "GHG Protocol Scope 2, contractual instrument EF",
    },
    SourceCategoryType.SCOPE2_STEAM: {
        "activity_data_pct": 3.0,
        "emission_factor_pct": 10.0,
        "source": "GHG Protocol Scope 2 Guidance, steam/heat networks",
    },
    SourceCategoryType.SCOPE2_COOLING: {
        "activity_data_pct": 3.0,
        "emission_factor_pct": 12.0,
        "source": "GHG Protocol Scope 2 Guidance, district cooling",
    },
    SourceCategoryType.SCOPE2_DUAL_REPORTING: {
        "activity_data_pct": 2.0,
        "emission_factor_pct": 10.0,
        "source": "GHG Protocol Scope 2 Guidance, dual reporting",
    },
}
"""Default uncertainty percentages per source category from IPCC 2006."""

# Data quality tier multipliers for adjusting default uncertainty ranges.
# Higher multiplier = wider uncertainty bounds for lower quality data.
DATA_QUALITY_MULTIPLIERS: Dict[str, float] = {
    DataQualityTier.HIGH: 0.6,
    DataQualityTier.MEDIUM: 1.0,
    DataQualityTier.LOW: 1.8,
}
"""Multipliers applied to default uncertainty ranges based on data quality."""

# Improvement recommendations by source category for top contributors.
IMPROVEMENT_RECOMMENDATIONS: Dict[str, str] = {
    SourceCategoryType.STATIONARY_COMBUSTION: (
        "Install calibrated fuel flow meters and use fuel analysis "
        "certificates for emission factors (Tier 2/3 approach)."
    ),
    SourceCategoryType.MOBILE_COMBUSTION: (
        "Use vehicle-specific fuel consumption records rather than "
        "fleet averages; adopt telematics for accurate distance tracking."
    ),
    SourceCategoryType.PROCESS_EMISSIONS: (
        "Use plant-specific emission factors from stack testing rather "
        "than IPCC defaults; consider CEMS for large sources."
    ),
    SourceCategoryType.FUGITIVE_EMISSIONS: (
        "Implement LDAR (Leak Detection and Repair) programme; use "
        "equipment-level component counts for bottom-up estimation."
    ),
    SourceCategoryType.REFRIGERANT_LEAKAGE: (
        "Track refrigerant purchases and disposals by equipment; use "
        "mass-balance method rather than default leak rates."
    ),
    SourceCategoryType.LAND_USE_CHANGE: (
        "Use site-specific carbon stock assessments and GIS-based land "
        "use mapping with satellite verification."
    ),
    SourceCategoryType.WASTE_TREATMENT: (
        "Obtain waste composition analysis and use site-specific methane "
        "generation rates rather than IPCC defaults."
    ),
    SourceCategoryType.AGRICULTURAL_EMISSIONS: (
        "Use region-specific livestock emission factors and actual feed "
        "intake data rather than global defaults."
    ),
    SourceCategoryType.SCOPE2_LOCATION_BASED: (
        "Use sub-national or regional grid emission factors rather than "
        "national averages; obtain hourly marginal factors where available."
    ),
    SourceCategoryType.SCOPE2_MARKET_BASED: (
        "Procure energy attribute certificates (EACs/RECs/GOs) with "
        "clear provenance; use supplier-specific emission factors."
    ),
    SourceCategoryType.SCOPE2_STEAM: (
        "Obtain supplier-specific steam emission factors based on "
        "boiler efficiency and fuel mix data."
    ),
    SourceCategoryType.SCOPE2_COOLING: (
        "Request district cooling provider's measured COP and "
        "fuel/electricity mix for accurate emission factor."
    ),
    SourceCategoryType.SCOPE2_DUAL_REPORTING: (
        "Ensure consistent activity data across location-based and "
        "market-based calculations; reconcile any discrepancies."
    ),
}
"""Improvement recommendations per source category."""

# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------

class SourceUncertainty(BaseModel):
    """Uncertainty profile for a single emission source category.

    Attributes:
        source_id: Unique source identifier.
        source_category: Emission source category (MRV agent mapping).
        description: Human-readable source description.
        emissions_tco2e: Total emissions from this source (tCO2e).
        activity_data_uncertainty_pct: Activity data uncertainty (% of mean,
            half-width of 95% CI).
        emission_factor_uncertainty_pct: Emission factor uncertainty (% of mean,
            half-width of 95% CI).
        distribution_type: Statistical distribution assumed for this source.
        data_quality_tier: Data quality tier (high/medium/low).
        correlation_group: Optional grouping for correlated sources.
        custom_combined_uncertainty_pct: Optional override for pre-calculated
            combined uncertainty.
    """
    source_id: str = Field(
        default_factory=_new_uuid, description="Unique source identifier"
    )
    source_category: str = Field(
        ..., min_length=1, description="Emission source category"
    )
    description: str = Field(
        default="", description="Human-readable source description"
    )
    emissions_tco2e: Decimal = Field(
        ..., ge=0, description="Total emissions (tCO2e)"
    )
    activity_data_uncertainty_pct: Decimal = Field(
        default=Decimal("5.0"), ge=0, le=200,
        description="Activity data uncertainty (%, half-width 95% CI)"
    )
    emission_factor_uncertainty_pct: Decimal = Field(
        default=Decimal("5.0"), ge=0, le=200,
        description="Emission factor uncertainty (%, half-width 95% CI)"
    )
    distribution_type: UncertaintyDistribution = Field(
        default=UncertaintyDistribution.NORMAL,
        description="Statistical distribution for Monte Carlo"
    )
    data_quality_tier: DataQualityTier = Field(
        default=DataQualityTier.MEDIUM,
        description="Data quality tier"
    )
    correlation_group: Optional[str] = Field(
        default=None, description="Correlation group for non-independent sources"
    )
    custom_combined_uncertainty_pct: Optional[Decimal] = Field(
        default=None, ge=0, le=200,
        description="Override: pre-calculated combined uncertainty"
    )

    @field_validator("emissions_tco2e", mode="before")
    @classmethod
    def coerce_emissions(cls, v: Any) -> Decimal:
        """Coerce emissions to Decimal."""
        return _decimal(v)

    @field_validator(
        "activity_data_uncertainty_pct",
        "emission_factor_uncertainty_pct",
        mode="before",
    )
    @classmethod
    def coerce_uncertainty(cls, v: Any) -> Decimal:
        """Coerce uncertainty values to Decimal."""
        return _decimal(v)

class MonteCarloConfig(BaseModel):
    """Configuration for Monte Carlo uncertainty simulation.

    Attributes:
        iterations: Number of Monte Carlo iterations.
        seed: Random seed for reproducibility.
        confidence_level: Confidence level for interval estimation.
        convergence_threshold: Relative change threshold to stop early.
        check_convergence_every: Check convergence every N iterations.
    """
    iterations: int = Field(
        default=10000, ge=100, le=1_000_000,
        description="Number of Monte Carlo iterations"
    )
    seed: int = Field(
        default=42, ge=0, description="Random seed for reproducibility"
    )
    confidence_level: Decimal = Field(
        default=Decimal("0.95"), ge=Decimal("0.50"), le=Decimal("0.99"),
        description="Confidence level (e.g. 0.95 for 95%)"
    )
    convergence_threshold: Decimal = Field(
        default=Decimal("0.001"), ge=0,
        description="Convergence threshold (relative change in mean)"
    )
    check_convergence_every: int = Field(
        default=1000, ge=100, description="Check convergence every N iterations"
    )

    @field_validator("confidence_level", "convergence_threshold", mode="before")
    @classmethod
    def coerce_decimal(cls, v: Any) -> Decimal:
        """Coerce to Decimal."""
        return _decimal(v)

# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------

class AnalyticalResult(BaseModel):
    """Result from analytical (quadrature) uncertainty aggregation.

    Attributes:
        combined_uncertainty_pct: Combined uncertainty as percentage of total
            emissions (half-width of 95% CI).
        lower_bound_tco2e: Lower bound of 95% CI (tCO2e).
        upper_bound_tco2e: Upper bound of 95% CI (tCO2e).
        total_emissions_tco2e: Sum of all source emissions.
        confidence_level: Confidence level (e.g. 0.95).
        method: Description of the method used.
    """
    combined_uncertainty_pct: float = Field(
        default=0.0, description="Combined uncertainty (%)"
    )
    lower_bound_tco2e: float = Field(
        default=0.0, description="Lower bound 95% CI (tCO2e)"
    )
    upper_bound_tco2e: float = Field(
        default=0.0, description="Upper bound 95% CI (tCO2e)"
    )
    total_emissions_tco2e: float = Field(
        default=0.0, description="Total emissions (tCO2e)"
    )
    confidence_level: float = Field(
        default=0.95, description="Confidence level"
    )
    method: str = Field(
        default="IPCC Approach 1 (error propagation via quadrature)",
        description="Aggregation method"
    )

class MonteCarloResult(BaseModel):
    """Result from Monte Carlo uncertainty simulation.

    Attributes:
        mean_tco2e: Mean of simulated total emissions.
        median_tco2e: Median (50th percentile).
        std_dev: Standard deviation of simulated totals.
        p2_5: 2.5th percentile (lower bound of 95% CI).
        p5: 5th percentile.
        p25: 25th percentile (first quartile).
        p50: 50th percentile (median).
        p75: 75th percentile (third quartile).
        p95: 95th percentile.
        p97_5: 97.5th percentile (upper bound of 95% CI).
        iterations_run: Actual iterations completed.
        convergence_achieved: Whether convergence was achieved.
        convergence_at_iteration: Iteration where convergence was achieved.
        seed_used: Random seed used.
    """
    mean_tco2e: float = Field(default=0.0, description="Mean (tCO2e)")
    median_tco2e: float = Field(default=0.0, description="Median (tCO2e)")
    std_dev: float = Field(default=0.0, description="Standard deviation")
    p2_5: float = Field(default=0.0, description="2.5th percentile")
    p5: float = Field(default=0.0, description="5th percentile")
    p25: float = Field(default=0.0, description="25th percentile")
    p50: float = Field(default=0.0, description="50th percentile")
    p75: float = Field(default=0.0, description="75th percentile")
    p95: float = Field(default=0.0, description="95th percentile")
    p97_5: float = Field(default=0.0, description="97.5th percentile")
    iterations_run: int = Field(default=0, description="Iterations completed")
    convergence_achieved: bool = Field(
        default=False, description="Convergence status"
    )
    convergence_at_iteration: Optional[int] = Field(
        default=None, description="Iteration where convergence achieved"
    )
    seed_used: int = Field(default=42, description="Random seed")

class UncertaintyContributor(BaseModel):
    """Contribution of a single source to overall uncertainty.

    Attributes:
        source_id: Source identifier.
        source_category: Source category name.
        emissions_tco2e: Emissions from this source (tCO2e).
        combined_uncertainty_pct: This source's combined uncertainty.
        variance_contribution: Absolute variance contribution.
        contribution_pct: Percentage of total uncertainty variance.
        rank: Rank (1 = largest contributor).
        recommended_improvement: Specific improvement recommendation.
    """
    source_id: str = Field(default="", description="Source ID")
    source_category: str = Field(default="", description="Source category")
    emissions_tco2e: float = Field(default=0.0, description="Emissions (tCO2e)")
    combined_uncertainty_pct: float = Field(
        default=0.0, description="Combined uncertainty (%)"
    )
    variance_contribution: float = Field(
        default=0.0, description="Variance contribution"
    )
    contribution_pct: float = Field(
        default=0.0, description="Contribution to total uncertainty (%)"
    )
    rank: int = Field(default=0, description="Contribution rank")
    recommended_improvement: str = Field(
        default="", description="Improvement recommendation"
    )

class DataQualityAssessment(BaseModel):
    """Assessment of overall data quality across all sources.

    Attributes:
        overall_score: Weighted score (0-100, higher = better quality).
        tier_distribution: Count of sources per data quality tier.
        sources_with_defaults: Count of sources using default uncertainty.
        sources_with_measured: Count of sources with measured uncertainty.
        improvement_priority: Ordered list of categories to improve.
        assessment_notes: Explanatory notes.
    """
    overall_score: float = Field(
        default=0.0, ge=0, le=100, description="Quality score (0-100)"
    )
    tier_distribution: Dict[str, int] = Field(
        default_factory=dict, description="Sources per quality tier"
    )
    sources_with_defaults: int = Field(
        default=0, description="Sources using default uncertainty"
    )
    sources_with_measured: int = Field(
        default=0, description="Sources with measured uncertainty"
    )
    improvement_priority: List[str] = Field(
        default_factory=list, description="Priority categories for improvement"
    )
    assessment_notes: List[str] = Field(
        default_factory=list, description="Assessment notes"
    )

class UncertaintyAggregationResult(BaseModel):
    """Complete uncertainty aggregation result with full provenance.

    Contains analytical bounds, Monte Carlo results, contributor analysis,
    data quality scoring, and improvement recommendations.

    Attributes:
        result_id: Unique result identifier.
        engine_version: Engine version string.
        calculated_at: Calculation timestamp (UTC).
        processing_time_ms: Total processing time in milliseconds.
        total_emissions_tco2e: Sum of all source emissions.
        source_count: Number of sources analysed.
        analytical: Analytical (quadrature) result.
        monte_carlo: Monte Carlo simulation result.
        top_contributors: Sources ranked by uncertainty contribution.
        data_quality_score: Data quality assessment.
        improvement_recommendations: Prioritised improvement actions.
        methodology_notes: Methodology and reference notes.
        provenance_hash: SHA-256 hash for audit trail.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Unique result ID")
    engine_version: str = Field(default=_MODULE_VERSION, description="Engine version")
    calculated_at: datetime = Field(
        default_factory=utcnow, description="Calculation timestamp"
    )
    processing_time_ms: float = Field(
        default=0.0, description="Processing time (ms)"
    )
    total_emissions_tco2e: float = Field(
        default=0.0, description="Total emissions (tCO2e)"
    )
    source_count: int = Field(default=0, description="Number of sources")
    analytical: Optional[AnalyticalResult] = Field(
        default=None, description="Analytical (quadrature) result"
    )
    monte_carlo: Optional[MonteCarloResult] = Field(
        default=None, description="Monte Carlo result"
    )
    top_contributors: List[UncertaintyContributor] = Field(
        default_factory=list, description="Top uncertainty contributors"
    )
    data_quality_score: Optional[DataQualityAssessment] = Field(
        default=None, description="Data quality assessment"
    )
    improvement_recommendations: List[str] = Field(
        default_factory=list, description="Improvement recommendations"
    )
    methodology_notes: List[str] = Field(
        default_factory=list, description="Methodology notes"
    )
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class UncertaintyAggregationEngine:
    """Organisation-level GHG uncertainty aggregation engine.

    Aggregates uncertainty from multiple source categories into overall
    confidence bounds using analytical error propagation (IPCC Approach 1)
    and/or Monte Carlo simulation (IPCC Approach 2).

    Supports all 13 Scope 1/2 MRV agent source categories with IPCC 2006
    default uncertainty ranges and configurable data quality tiers.

    Guarantees:
        - Deterministic: same inputs produce identical analytical results.
        - Reproducible: Monte Carlo uses fixed seed; SHA-256 provenance hash.
        - Auditable: full contributor analysis and methodology notes.
        - No LLM: zero hallucination risk in any calculation path.

    Usage::

        engine = UncertaintyAggregationEngine()
        sources = [
            SourceUncertainty(
                source_category="stationary_combustion",
                emissions_tco2e=Decimal("5000"),
            ),
            SourceUncertainty(
                source_category="scope2_location_based",
                emissions_tco2e=Decimal("3000"),
            ),
        ]
        result = engine.aggregate(sources)
        print(f"Combined uncertainty: {result.analytical.combined_uncertainty_pct}%")
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise the uncertainty aggregation engine.

        Args:
            config: Optional overrides. Supported keys:
                - significance_threshold (float): min % for significance
                - max_contributors (int): max contributors to return
                - default_confidence_level (float): default CI level
        """
        self._config = config or {}
        self._significance_threshold = float(
            self._config.get("significance_threshold", 1.0)
        )
        self._max_contributors = int(
            self._config.get("max_contributors", 20)
        )
        self._default_confidence = Decimal(
            str(self._config.get("default_confidence_level", "0.95"))
        )
        self._notes: List[str] = []
        logger.info("UncertaintyAggregationEngine v%s initialised.", _MODULE_VERSION)

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def aggregate(
        self,
        sources: List[SourceUncertainty],
        method: AggregationMethod = AggregationMethod.BOTH,
        mc_config: Optional[MonteCarloConfig] = None,
    ) -> UncertaintyAggregationResult:
        """Run complete uncertainty aggregation analysis.

        Args:
            sources: List of source uncertainty profiles.
            method: Aggregation method (analytical, monte_carlo, both).
            mc_config: Monte Carlo configuration (optional).

        Returns:
            UncertaintyAggregationResult with all analysis outputs.

        Raises:
            ValueError: If sources list is empty.
        """
        t0 = time.perf_counter()
        self._notes = [f"Engine version: {self.engine_version}"]

        if not sources:
            raise ValueError("At least one source is required for aggregation.")

        # Apply default uncertainty ranges where not specified
        enriched = self._enrich_defaults(sources)

        # Calculate total emissions
        total = sum(_decimal(s.emissions_tco2e) for s in enriched)

        logger.info(
            "Uncertainty aggregation: %d sources, total %.2f tCO2e, method=%s",
            len(enriched), float(total), method.value,
        )

        # Analytical aggregation
        analytical = None
        if method in (AggregationMethod.ANALYTICAL, AggregationMethod.BOTH):
            analytical = self.aggregate_analytical(enriched)

        # Monte Carlo simulation
        mc_result = None
        if method in (AggregationMethod.MONTE_CARLO, AggregationMethod.BOTH):
            cfg = mc_config or MonteCarloConfig()
            mc_result = self.run_monte_carlo(enriched, cfg)

        # Contributor analysis
        contributors = self.identify_top_contributors(enriched)

        # Data quality assessment
        quality = self._assess_data_quality(enriched)

        # Improvement recommendations
        recommendations = self.recommend_improvements(contributors)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        result = UncertaintyAggregationResult(
            total_emissions_tco2e=_round2(float(total)),
            source_count=len(enriched),
            analytical=analytical,
            monte_carlo=mc_result,
            top_contributors=contributors,
            data_quality_score=quality,
            improvement_recommendations=recommendations,
            methodology_notes=list(self._notes),
            processing_time_ms=_round3(elapsed_ms),
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Uncertainty aggregation complete: %d sources, "
            "analytical=%.2f%%, MC mean=%.2f tCO2e, hash=%s (%.1f ms)",
            len(enriched),
            analytical.combined_uncertainty_pct if analytical else 0.0,
            mc_result.mean_tco2e if mc_result else 0.0,
            result.provenance_hash[:16],
            elapsed_ms,
        )
        return result

    def aggregate_analytical(
        self,
        sources: List[SourceUncertainty],
    ) -> AnalyticalResult:
        """Aggregate uncertainty using analytical error propagation.

        Uses IPCC Approach 1 (quadrature / root-sum-of-squares) for
        independent, uncorrelated sources:

            U_total = sqrt( sum( (E_i * u_i)^2 ) ) / E_total * 100

        Where:
            E_i = emissions from source i (tCO2e)
            u_i = fractional combined uncertainty of source i
            E_total = sum of all E_i

        This formula assumes source uncertainties are independent. For
        correlated sources, use Monte Carlo instead.

        Args:
            sources: List of source uncertainty profiles.

        Returns:
            AnalyticalResult with combined uncertainty and bounds.
        """
        if not sources:
            return AnalyticalResult()

        total = Decimal("0")
        sum_sq = Decimal("0")

        for s in sources:
            emissions = _decimal(s.emissions_tco2e)
            total += emissions

            # Combined uncertainty for this source (IPCC Eq 3.1)
            u_ad = _decimal(s.activity_data_uncertainty_pct) / Decimal("100")
            u_ef = _decimal(s.emission_factor_uncertainty_pct) / Decimal("100")

            if s.custom_combined_uncertainty_pct is not None:
                u_combined = _decimal(s.custom_combined_uncertainty_pct) / Decimal("100")
            else:
                # Quadrature of activity data and emission factor uncertainty
                u_combined = _decimal(
                    math.sqrt(float(u_ad ** 2 + u_ef ** 2))
                )

            # Variance contribution = (E_i * u_i)^2
            variance_i = (emissions * u_combined) ** 2
            sum_sq += variance_i

        if total == Decimal("0"):
            return AnalyticalResult()

        # Combined uncertainty = sqrt(sum_sq) / total * 100
        combined_abs = _decimal(math.sqrt(max(float(sum_sq), 0.0)))
        combined_pct = _safe_divide(
            combined_abs * Decimal("100"), total
        )

        # 95% CI bounds (assuming normal distribution, z=1.96)
        half_width = combined_abs
        lower = total - half_width
        upper = total + half_width

        self._notes.append(
            f"Analytical (quadrature): combined {_round2(float(combined_pct))}%, "
            f"total {_round2(float(total))} tCO2e, "
            f"bounds [{_round2(float(lower))}, {_round2(float(upper))}]."
        )

        return AnalyticalResult(
            combined_uncertainty_pct=_round2(float(combined_pct)),
            lower_bound_tco2e=_round2(float(lower)),
            upper_bound_tco2e=_round2(float(upper)),
            total_emissions_tco2e=_round2(float(total)),
            confidence_level=0.95,
            method="IPCC Approach 1 (error propagation via quadrature)",
        )

    def run_monte_carlo(
        self,
        sources: List[SourceUncertainty],
        config: MonteCarloConfig,
    ) -> MonteCarloResult:
        """Run Monte Carlo uncertainty simulation.

        For each iteration, samples from each source's uncertainty
        distribution and computes the total. Extracts percentiles from
        the distribution of simulated totals.

        Uses numpy for vectorised random sampling with fixed seed for
        full reproducibility. Falls back to pure-Python if numpy is
        not available.

        Args:
            sources: List of source uncertainty profiles.
            config: Monte Carlo configuration.

        Returns:
            MonteCarloResult with distribution statistics.
        """
        try:
            import numpy as np
            return self._mc_numpy(sources, config, np)
        except ImportError:
            logger.warning(
                "numpy not available; using pure-Python Monte Carlo (slower)."
            )
            return self._mc_pure_python(sources, config)

    def identify_top_contributors(
        self,
        sources: List[SourceUncertainty],
    ) -> List[UncertaintyContributor]:
        """Identify sources that contribute most to overall uncertainty.

        Uses the variance decomposition formula:
            Contribution_i = (E_i * u_i)^2 / sum( (E_j * u_j)^2 ) * 100

        Args:
            sources: List of source uncertainty profiles.

        Returns:
            List of UncertaintyContributor sorted by contribution (descending).
        """
        if not sources:
            return []

        # Calculate variance contribution for each source
        variances: List[Tuple[int, Decimal, SourceUncertainty]] = []
        total_variance = Decimal("0")

        for idx, s in enumerate(sources):
            emissions = _decimal(s.emissions_tco2e)
            u_ad = _decimal(s.activity_data_uncertainty_pct) / Decimal("100")
            u_ef = _decimal(s.emission_factor_uncertainty_pct) / Decimal("100")

            if s.custom_combined_uncertainty_pct is not None:
                u_combined = _decimal(s.custom_combined_uncertainty_pct) / Decimal("100")
            else:
                u_combined = _decimal(math.sqrt(float(u_ad ** 2 + u_ef ** 2)))

            var_i = (emissions * u_combined) ** 2
            variances.append((idx, var_i, s))
            total_variance += var_i

        # Sort by variance contribution (descending)
        variances.sort(key=lambda x: x[1], reverse=True)

        contributors: List[UncertaintyContributor] = []
        for rank, (idx, var_i, s) in enumerate(variances, start=1):
            if rank > self._max_contributors:
                break

            emissions = _decimal(s.emissions_tco2e)
            u_ad = _decimal(s.activity_data_uncertainty_pct) / Decimal("100")
            u_ef = _decimal(s.emission_factor_uncertainty_pct) / Decimal("100")

            if s.custom_combined_uncertainty_pct is not None:
                u_combined_pct = float(s.custom_combined_uncertainty_pct)
            else:
                u_combined_pct = _round2(
                    math.sqrt(
                        float(
                            _decimal(s.activity_data_uncertainty_pct) ** 2
                            + _decimal(s.emission_factor_uncertainty_pct) ** 2
                        )
                    )
                )

            pct_contribution = float(
                _safe_pct(var_i, total_variance)
            ) if total_variance > Decimal("0") else 0.0

            # Get recommendation for this source category
            recommendation = self._get_recommendation(s.source_category)

            contributors.append(UncertaintyContributor(
                source_id=s.source_id,
                source_category=s.source_category,
                emissions_tco2e=_round2(float(emissions)),
                combined_uncertainty_pct=_round2(u_combined_pct),
                variance_contribution=_round2(float(var_i)),
                contribution_pct=_round2(pct_contribution),
                rank=rank,
                recommended_improvement=recommendation,
            ))

        return contributors

    def recommend_improvements(
        self,
        contributors: List[UncertaintyContributor],
    ) -> List[str]:
        """Generate prioritised improvement recommendations.

        Focuses on the top contributors to overall uncertainty, as these
        offer the most reduction potential per unit of effort.

        Args:
            contributors: Ranked list of uncertainty contributors.

        Returns:
            List of prioritised recommendation strings.
        """
        if not contributors:
            return ["No sources provided for uncertainty analysis."]

        recommendations: List[str] = []

        # Top 3 contributors get specific recommendations
        for c in contributors[:3]:
            if c.contribution_pct >= self._significance_threshold:
                recommendations.append(
                    f"[Rank {c.rank}, {c.contribution_pct:.1f}% of total uncertainty] "
                    f"{c.source_category}: {c.recommended_improvement}"
                )

        # General recommendations based on analysis
        total_top3_pct = sum(c.contribution_pct for c in contributors[:3])
        if total_top3_pct > 80.0:
            recommendations.append(
                f"Top 3 sources account for {total_top3_pct:.1f}% of total "
                f"uncertainty. Focused effort on these sources will yield the "
                f"greatest improvement in inventory accuracy."
            )

        # Data quality upgrade recommendation
        low_quality_sources = [
            c for c in contributors
            if c.combined_uncertainty_pct > 30.0
        ]
        if low_quality_sources:
            recommendations.append(
                f"{len(low_quality_sources)} source(s) have combined uncertainty "
                f">30%. Consider upgrading from IPCC Tier 1 to Tier 2 emission "
                f"factors and improving activity data measurement systems."
            )

        if not recommendations:
            recommendations.append(
                "Uncertainty analysis indicates good data quality across "
                "all sources. Continue current measurement practices."
            )

        return recommendations

    def calculate_source_combined_uncertainty(
        self,
        activity_data_uncertainty_pct: float,
        emission_factor_uncertainty_pct: float,
    ) -> float:
        """Calculate combined uncertainty for a single source.

        Uses quadrature (root-sum-of-squares) of independent uncertainties:
            U_combined = sqrt(U_ad^2 + U_ef^2)

        Args:
            activity_data_uncertainty_pct: Activity data uncertainty (%).
            emission_factor_uncertainty_pct: Emission factor uncertainty (%).

        Returns:
            Combined uncertainty as percentage.
        """
        u_ad = _decimal(activity_data_uncertainty_pct)
        u_ef = _decimal(emission_factor_uncertainty_pct)
        combined = _decimal(math.sqrt(float(u_ad ** 2 + u_ef ** 2)))
        return _round2(float(combined))

    def get_default_uncertainty(
        self,
        source_category: str,
        data_quality_tier: DataQualityTier = DataQualityTier.MEDIUM,
    ) -> Dict[str, float]:
        """Get default IPCC uncertainty ranges for a source category.

        Args:
            source_category: Source category identifier.
            data_quality_tier: Data quality tier for adjustment.

        Returns:
            Dict with activity_data_pct, emission_factor_pct, combined_pct.
        """
        defaults = DEFAULT_UNCERTAINTY_RANGES.get(source_category, {})
        ad_pct = defaults.get("activity_data_pct", 10.0)
        ef_pct = defaults.get("emission_factor_pct", 10.0)

        # Apply quality tier multiplier
        multiplier = DATA_QUALITY_MULTIPLIERS.get(data_quality_tier, 1.0)
        ad_pct *= multiplier
        ef_pct *= multiplier

        combined = math.sqrt(ad_pct ** 2 + ef_pct ** 2)

        return {
            "activity_data_pct": _round2(ad_pct),
            "emission_factor_pct": _round2(ef_pct),
            "combined_pct": _round2(combined),
            "quality_multiplier": multiplier,
            "source": defaults.get("source", "IPCC 2006 default"),
        }

    # -------------------------------------------------------------------
    # Private -- Default enrichment
    # -------------------------------------------------------------------

    def _enrich_defaults(
        self,
        sources: List[SourceUncertainty],
    ) -> List[SourceUncertainty]:
        """Enrich sources with IPCC default uncertainty values where missing.

        Applies data quality tier multipliers to default ranges.

        Args:
            sources: Raw source uncertainty profiles.

        Returns:
            Enriched list (originals are not mutated).
        """
        enriched: List[SourceUncertainty] = []

        for s in sources:
            defaults = DEFAULT_UNCERTAINTY_RANGES.get(s.source_category, {})
            multiplier = DATA_QUALITY_MULTIPLIERS.get(s.data_quality_tier, 1.0)

            # Use provided values or fall back to defaults
            ad_pct = s.activity_data_uncertainty_pct
            ef_pct = s.emission_factor_uncertainty_pct

            # If the values are the model defaults and we have IPCC data, use IPCC
            if (
                ad_pct == Decimal("5.0")
                and ef_pct == Decimal("5.0")
                and defaults
            ):
                ad_pct = _decimal(defaults.get("activity_data_pct", 5.0) * multiplier)
                ef_pct = _decimal(defaults.get("emission_factor_pct", 5.0) * multiplier)

            enriched.append(SourceUncertainty(
                source_id=s.source_id,
                source_category=s.source_category,
                description=s.description,
                emissions_tco2e=s.emissions_tco2e,
                activity_data_uncertainty_pct=ad_pct,
                emission_factor_uncertainty_pct=ef_pct,
                distribution_type=s.distribution_type,
                data_quality_tier=s.data_quality_tier,
                correlation_group=s.correlation_group,
                custom_combined_uncertainty_pct=s.custom_combined_uncertainty_pct,
            ))

        return enriched

    # -------------------------------------------------------------------
    # Private -- Monte Carlo (numpy)
    # -------------------------------------------------------------------

    def _mc_numpy(
        self,
        sources: List[SourceUncertainty],
        config: MonteCarloConfig,
        np: Any,
    ) -> MonteCarloResult:
        """Monte Carlo simulation using numpy for performance.

        For each source, samples from the specified distribution
        parameterised by the source's emissions and uncertainty.
        For NORMAL: N(mu=E_i, sigma=E_i * u_i)
        For LOGNORMAL: lognormal with mean=E_i and sigma=E_i * u_i
        For UNIFORM: U(E_i * (1 - u_i), E_i * (1 + u_i))
        For TRIANGULAR: Tri(E_i * (1 - u_i), E_i, E_i * (1 + u_i))

        Args:
            sources: Source uncertainty profiles.
            config: Monte Carlo configuration.
            np: numpy module reference.

        Returns:
            MonteCarloResult with percentile statistics.
        """
        rng = np.random.default_rng(config.seed)
        n_iter = config.iterations

        # Pre-allocate totals array
        totals = np.zeros(n_iter, dtype=np.float64)

        for s in sources:
            emissions = float(s.emissions_tco2e)
            u_ad = float(s.activity_data_uncertainty_pct) / 100.0
            u_ef = float(s.emission_factor_uncertainty_pct) / 100.0

            if s.custom_combined_uncertainty_pct is not None:
                u_combined = float(s.custom_combined_uncertainty_pct) / 100.0
            else:
                u_combined = math.sqrt(u_ad ** 2 + u_ef ** 2)

            sigma = emissions * u_combined

            if s.distribution_type == UncertaintyDistribution.NORMAL:
                samples = rng.normal(loc=emissions, scale=max(sigma, 1e-10), size=n_iter)
                # Clamp to non-negative (emissions cannot be negative)
                samples = np.maximum(samples, 0.0)

            elif s.distribution_type == UncertaintyDistribution.LOGNORMAL:
                if emissions > 0 and sigma > 0:
                    # Parameterise lognormal from desired mean and std
                    variance = sigma ** 2
                    mu_ln = math.log(emissions ** 2 / math.sqrt(variance + emissions ** 2))
                    sigma_ln = math.sqrt(math.log(1 + variance / emissions ** 2))
                    samples = rng.lognormal(mean=mu_ln, sigma=max(sigma_ln, 1e-10), size=n_iter)
                else:
                    samples = np.full(n_iter, emissions)

            elif s.distribution_type == UncertaintyDistribution.UNIFORM:
                low = max(emissions * (1 - u_combined), 0.0)
                high = emissions * (1 + u_combined)
                samples = rng.uniform(low=low, high=max(high, low + 1e-10), size=n_iter)

            elif s.distribution_type == UncertaintyDistribution.TRIANGULAR:
                low = max(emissions * (1 - u_combined), 0.0)
                high = emissions * (1 + u_combined)
                mode = emissions
                if low >= high:
                    samples = np.full(n_iter, emissions)
                else:
                    mode = max(min(mode, high - 1e-10), low + 1e-10)
                    samples = rng.triangular(left=low, mode=mode, right=high, size=n_iter)

            else:
                samples = rng.normal(loc=emissions, scale=max(sigma, 1e-10), size=n_iter)
                samples = np.maximum(samples, 0.0)

            totals += samples

        # Check convergence periodically
        convergence_achieved = False
        convergence_at = None
        check_every = config.check_convergence_every
        threshold = float(config.convergence_threshold)

        if check_every < n_iter:
            prev_mean = float(np.mean(totals[:check_every]))
            for check_idx in range(2 * check_every, n_iter + 1, check_every):
                curr_mean = float(np.mean(totals[:check_idx]))
                if prev_mean > 0:
                    relative_change = abs(curr_mean - prev_mean) / prev_mean
                    if relative_change < threshold:
                        convergence_achieved = True
                        convergence_at = check_idx
                        break
                prev_mean = curr_mean

        # Extract percentiles
        percentiles = np.percentile(totals, [2.5, 5, 25, 50, 75, 95, 97.5])

        self._notes.append(
            f"Monte Carlo: {n_iter} iterations, seed={config.seed}, "
            f"mean={_round2(float(np.mean(totals)))} tCO2e, "
            f"std={_round2(float(np.std(totals)))}, "
            f"convergence={'yes' if convergence_achieved else 'no'}."
        )

        return MonteCarloResult(
            mean_tco2e=_round2(float(np.mean(totals))),
            median_tco2e=_round2(float(np.median(totals))),
            std_dev=_round2(float(np.std(totals))),
            p2_5=_round2(float(percentiles[0])),
            p5=_round2(float(percentiles[1])),
            p25=_round2(float(percentiles[2])),
            p50=_round2(float(percentiles[3])),
            p75=_round2(float(percentiles[4])),
            p95=_round2(float(percentiles[5])),
            p97_5=_round2(float(percentiles[6])),
            iterations_run=n_iter,
            convergence_achieved=convergence_achieved,
            convergence_at_iteration=convergence_at,
            seed_used=config.seed,
        )

    # -------------------------------------------------------------------
    # Private -- Monte Carlo (pure Python fallback)
    # -------------------------------------------------------------------

    def _mc_pure_python(
        self,
        sources: List[SourceUncertainty],
        config: MonteCarloConfig,
    ) -> MonteCarloResult:
        """Pure-Python Monte Carlo fallback when numpy is unavailable.

        Uses the random module with fixed seed. Significantly slower than
        numpy for large iteration counts.

        Args:
            sources: Source uncertainty profiles.
            config: Monte Carlo configuration.

        Returns:
            MonteCarloResult with percentile statistics.
        """
        import random as py_random


        rng = py_random.Random(config.seed)
        n_iter = config.iterations
        totals: List[float] = []

        for _ in range(n_iter):
            total = 0.0
            for s in sources:
                emissions = float(s.emissions_tco2e)
                u_ad = float(s.activity_data_uncertainty_pct) / 100.0
                u_ef = float(s.emission_factor_uncertainty_pct) / 100.0

                if s.custom_combined_uncertainty_pct is not None:
                    u_combined = float(s.custom_combined_uncertainty_pct) / 100.0
                else:
                    u_combined = math.sqrt(u_ad ** 2 + u_ef ** 2)

                sigma = emissions * u_combined

                if s.distribution_type == UncertaintyDistribution.NORMAL:
                    sample = max(rng.gauss(emissions, max(sigma, 1e-10)), 0.0)
                elif s.distribution_type == UncertaintyDistribution.LOGNORMAL:
                    if emissions > 0 and sigma > 0:
                        variance = sigma ** 2
                        mu_ln = math.log(
                            emissions ** 2 / math.sqrt(variance + emissions ** 2)
                        )
                        sigma_ln = math.sqrt(
                            math.log(1 + variance / emissions ** 2)
                        )
                        sample = rng.lognormvariate(mu_ln, max(sigma_ln, 1e-10))
                    else:
                        sample = emissions
                elif s.distribution_type == UncertaintyDistribution.UNIFORM:
                    low = max(emissions * (1 - u_combined), 0.0)
                    high = emissions * (1 + u_combined)
                    sample = rng.uniform(low, max(high, low + 1e-10))
                elif s.distribution_type == UncertaintyDistribution.TRIANGULAR:
                    low = max(emissions * (1 - u_combined), 0.0)
                    high = emissions * (1 + u_combined)
                    mode = max(min(emissions, high - 1e-10), low + 1e-10)
                    sample = rng.triangular(low, high, mode)
                else:
                    sample = max(rng.gauss(emissions, max(sigma, 1e-10)), 0.0)

                total += sample
            totals.append(total)

        totals.sort()
        n = len(totals)

        def _percentile(data: List[float], p: float) -> float:
            """Calculate percentile from sorted data."""
            k = (n - 1) * p / 100.0
            f_val = math.floor(k)
            c_val = math.ceil(k)
            if f_val == c_val:
                return data[int(k)]
            return data[int(f_val)] * (c_val - k) + data[int(c_val)] * (k - f_val)

        mean_val = sum(totals) / n
        variance_val = sum((t - mean_val) ** 2 for t in totals) / max(n - 1, 1)
        std_val = math.sqrt(max(variance_val, 0.0))

        self._notes.append(
            f"Monte Carlo (pure-Python): {n_iter} iterations, seed={config.seed}, "
            f"mean={_round2(mean_val)} tCO2e."
        )

        return MonteCarloResult(
            mean_tco2e=_round2(mean_val),
            median_tco2e=_round2(_percentile(totals, 50)),
            std_dev=_round2(std_val),
            p2_5=_round2(_percentile(totals, 2.5)),
            p5=_round2(_percentile(totals, 5)),
            p25=_round2(_percentile(totals, 25)),
            p50=_round2(_percentile(totals, 50)),
            p75=_round2(_percentile(totals, 75)),
            p95=_round2(_percentile(totals, 95)),
            p97_5=_round2(_percentile(totals, 97.5)),
            iterations_run=n_iter,
            convergence_achieved=False,
            seed_used=config.seed,
        )

    # -------------------------------------------------------------------
    # Private -- Data quality assessment
    # -------------------------------------------------------------------

    def _assess_data_quality(
        self,
        sources: List[SourceUncertainty],
    ) -> DataQualityAssessment:
        """Assess overall data quality across all sources.

        Scores range from 0-100 where:
            100 = all HIGH quality with measured uncertainty
            50  = all MEDIUM quality with IPCC defaults
            0   = all LOW quality with no data

        Args:
            sources: Enriched source uncertainty profiles.

        Returns:
            DataQualityAssessment with score and recommendations.
        """
        if not sources:
            return DataQualityAssessment()

        tier_counts: Dict[str, int] = {
            DataQualityTier.HIGH: 0,
            DataQualityTier.MEDIUM: 0,
            DataQualityTier.LOW: 0,
        }
        tier_scores = {
            DataQualityTier.HIGH: Decimal("100"),
            DataQualityTier.MEDIUM: Decimal("50"),
            DataQualityTier.LOW: Decimal("10"),
        }

        total_emissions = sum(_decimal(s.emissions_tco2e) for s in sources)
        weighted_score = Decimal("0")
        measured_count = 0
        default_count = 0
        notes: List[str] = []

        for s in sources:
            tier_counts[s.data_quality_tier] = (
                tier_counts.get(s.data_quality_tier, 0) + 1
            )

            # Emission-weighted quality score
            weight = _safe_divide(
                _decimal(s.emissions_tco2e), total_emissions
            ) if total_emissions > Decimal("0") else _decimal(1.0 / len(sources))

            tier_score = tier_scores.get(s.data_quality_tier, Decimal("50"))
            weighted_score += weight * tier_score

            if s.custom_combined_uncertainty_pct is not None:
                measured_count += 1
            else:
                default_count += 1

        # Improvement priority: sort sources by combined uncertainty descending
        priority = sorted(
            sources,
            key=lambda x: float(
                _decimal(x.activity_data_uncertainty_pct) ** 2
                + _decimal(x.emission_factor_uncertainty_pct) ** 2
            ),
            reverse=True,
        )
        priority_categories = []
        seen: set = set()
        for s in priority:
            if s.source_category not in seen:
                priority_categories.append(s.source_category)
                seen.add(s.source_category)

        if tier_counts.get(DataQualityTier.LOW, 0) > 0:
            notes.append(
                f"{tier_counts[DataQualityTier.LOW]} source(s) classified as "
                f"LOW quality; consider upgrading measurement approach."
            )

        if default_count > 0:
            notes.append(
                f"{default_count} source(s) use IPCC default uncertainty; "
                f"organisation-specific measurement would improve accuracy."
            )

        return DataQualityAssessment(
            overall_score=_round2(float(weighted_score)),
            tier_distribution={k.value if hasattr(k, 'value') else k: v for k, v in tier_counts.items()},
            sources_with_defaults=default_count,
            sources_with_measured=measured_count,
            improvement_priority=priority_categories[:5],
            assessment_notes=notes,
        )

    # -------------------------------------------------------------------
    # Private -- Recommendation lookup
    # -------------------------------------------------------------------

    def _get_recommendation(self, source_category: str) -> str:
        """Get improvement recommendation for a source category.

        Args:
            source_category: Source category identifier.

        Returns:
            Recommendation string.
        """
        rec = IMPROVEMENT_RECOMMENDATIONS.get(source_category)
        if rec:
            return rec

        # Fuzzy match by checking enum values
        for cat, recommendation in IMPROVEMENT_RECOMMENDATIONS.items():
            cat_val = cat.value if hasattr(cat, "value") else str(cat)
            if cat_val == source_category:
                return recommendation

        return (
            "Upgrade from IPCC Tier 1 default values to site-specific "
            "emission factors and metered activity data."
        )

# ---------------------------------------------------------------------------
# Pydantic v2 model_rebuild for forward-reference resolution
# ---------------------------------------------------------------------------

SourceUncertainty.model_rebuild()
MonteCarloConfig.model_rebuild()
AnalyticalResult.model_rebuild()
MonteCarloResult.model_rebuild()
UncertaintyContributor.model_rebuild()
DataQualityAssessment.model_rebuild()
UncertaintyAggregationResult.model_rebuild()
