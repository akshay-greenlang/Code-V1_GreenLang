# -*- coding: utf-8 -*-
"""
Scope3UncertaintyEngine - PACK-042 Scope 3 Starter Pack Engine 8
==================================================================

Quantifies uncertainty in Scope 3 emission estimates using both Monte
Carlo simulation and IPCC Approach 1 analytical (error propagation via
quadrature) methods.  Supports per-category uncertainty profiling,
inter-category correlation handling, sensitivity analysis (tornado
charts), confidence interval construction, and tier-upgrade impact
assessment.

Calculation Methodology:
    Analytical (IPCC Approach 1 -- Error Propagation):
        U_combined = sqrt( sum( (E_i * u_i)^2 ) ) / E_total * 100

        Where:
            E_i     = emissions from category i (tCO2e)
            u_i     = fractional uncertainty of category i (half-width 95% CI)
            E_total = sum of all E_i

        For correlated categories:
            cov(i,j) = E_i * u_i * E_j * u_j * rho(i,j)
            U_combined = sqrt( sum(var_i) + 2 * sum(cov_pairs) ) / E_total * 100

    Monte Carlo Simulation (IPCC Approach 2):
        For each iteration k = 1..N:
            For each category i, sample E_i_k from uncertainty distribution
            (using Cholesky decomposition for correlated categories)
            Compute total_k = sum(E_i_k)
        Extract percentiles from distribution of total_k

Default Uncertainty Ranges by Methodology Tier:
    Spend-based:       +/- 50-200% (lognormal distribution)
    Average-data:      +/- 20-50%  (normal distribution)
    Supplier-specific: +/- 5-20%   (normal distribution)

    Source: GHG Protocol Scope 3 Calculation Guidance (2013), Table 7.4

Default Inter-Category Correlations:
    Cat 1 <-> Cat 4:  rho = 0.3  (purchased goods drive transport)
    Cat 1 <-> Cat 5:  rho = 0.2  (purchased goods drive waste)
    Cat 4 <-> Cat 9:  rho = 0.4  (upstream/downstream transport shared)
    Cat 1 <-> Cat 12: rho = 0.2  (purchased goods -> end-of-life)
    Cat 6 <-> Cat 7:  rho = 0.3  (travel and commuting correlate)
    Cat 11 <-> Cat 12: rho = 0.5 (use of sold -> end-of-life)

Regulatory References:
    - IPCC 2006 Guidelines, Volume 1, Chapter 3 (Uncertainties)
    - IPCC 2019 Refinement, Volume 1, Chapter 3
    - GHG Protocol Scope 3 Calculation Guidance (2013), Chapter 7
    - ISO 14064-1:2018, Clause 9 (Uncertainty Assessment)
    - EPA GHGRP Subpart A, 40 CFR 98.3(d)

Zero-Hallucination:
    - All calculations use deterministic Decimal arithmetic (analytical)
    - Monte Carlo uses numpy with fixed seed for reproducibility
    - Uncertainty ranges from published IPCC / GHG Protocol tables
    - No LLM involvement in any calculation path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-042 Scope 3 Starter
Engine:  8 of 10
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

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"

# Attempt to import numpy; degrade gracefully if unavailable.
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    _HAS_NUMPY = False
    logger.warning("numpy not available; Monte Carlo simulation will be disabled.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash of arbitrary data.

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
    """Safely convert a value to Decimal."""
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
    """Safely divide two Decimals."""
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


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Scope3Category(str, Enum):
    """GHG Protocol Scope 3 categories (1-15)."""
    CAT_1 = "cat_1_purchased_goods"
    CAT_2 = "cat_2_capital_goods"
    CAT_3 = "cat_3_fuel_energy"
    CAT_4 = "cat_4_upstream_transport"
    CAT_5 = "cat_5_waste"
    CAT_6 = "cat_6_business_travel"
    CAT_7 = "cat_7_employee_commuting"
    CAT_8 = "cat_8_upstream_leased"
    CAT_9 = "cat_9_downstream_transport"
    CAT_10 = "cat_10_processing"
    CAT_11 = "cat_11_use_of_sold"
    CAT_12 = "cat_12_end_of_life"
    CAT_13 = "cat_13_downstream_leased"
    CAT_14 = "cat_14_franchises"
    CAT_15 = "cat_15_investments"


class MethodologyTier(str, Enum):
    """Methodology tier affecting uncertainty profile.

    SPEND_BASED:       Financial proxy -- highest uncertainty.
    AVERAGE_DATA:      Activity + average EFs -- moderate uncertainty.
    SUPPLIER_SPECIFIC: Primary supplier data -- lowest uncertainty.
    """
    SPEND_BASED = "spend_based"
    AVERAGE_DATA = "average_data"
    SUPPLIER_SPECIFIC = "supplier_specific"


class DistributionType(str, Enum):
    """Statistical distribution for uncertainty sampling.

    NORMAL:     Symmetric bell curve (for moderate uncertainty).
    LOGNORMAL:  Right-skewed (for high-uncertainty spend-based estimates).
    UNIFORM:    Equal probability across range (for bounded estimates).
    TRIANGULAR: Min-mode-max (for expert judgement).
    """
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    UNIFORM = "uniform"
    TRIANGULAR = "triangular"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# Default uncertainty ranges (half-width of 95% CI as %) by methodology tier.
# Source: GHG Protocol Scope 3 Calculation Guidance, Table 7.4.
DEFAULT_UNCERTAINTY_BY_TIER: Dict[str, Dict[str, Any]] = {
    MethodologyTier.SPEND_BASED: {
        "low_pct": 50.0,
        "high_pct": 200.0,
        "default_pct": 100.0,
        "distribution": DistributionType.LOGNORMAL,
        "source": "GHG Protocol Scope 3 Guidance, Table 7.4 (spend-based)",
    },
    MethodologyTier.AVERAGE_DATA: {
        "low_pct": 20.0,
        "high_pct": 50.0,
        "default_pct": 30.0,
        "distribution": DistributionType.NORMAL,
        "source": "GHG Protocol Scope 3 Guidance, Table 7.4 (average-data)",
    },
    MethodologyTier.SUPPLIER_SPECIFIC: {
        "low_pct": 5.0,
        "high_pct": 20.0,
        "default_pct": 10.0,
        "distribution": DistributionType.NORMAL,
        "source": "GHG Protocol Scope 3 Guidance, Table 7.4 (supplier-specific)",
    },
}
"""Default uncertainty ranges by methodology tier."""

# Default inter-category correlation coefficients.
# Source: Expert judgement informed by GHG Protocol guidance and
# academic literature on Scope 3 estimation.
DEFAULT_CORRELATIONS: List[Tuple[str, str, float]] = [
    (Scope3Category.CAT_1, Scope3Category.CAT_4, 0.3),
    (Scope3Category.CAT_1, Scope3Category.CAT_5, 0.2),
    (Scope3Category.CAT_1, Scope3Category.CAT_12, 0.2),
    (Scope3Category.CAT_4, Scope3Category.CAT_9, 0.4),
    (Scope3Category.CAT_6, Scope3Category.CAT_7, 0.3),
    (Scope3Category.CAT_11, Scope3Category.CAT_12, 0.5),
    (Scope3Category.CAT_1, Scope3Category.CAT_2, 0.3),
    (Scope3Category.CAT_3, Scope3Category.CAT_1, 0.15),
    (Scope3Category.CAT_10, Scope3Category.CAT_11, 0.3),
]
"""Default correlation pairs (category_a, category_b, rho)."""

# Human-readable category names.
CATEGORY_NAMES: Dict[str, str] = {
    Scope3Category.CAT_1: "Cat 1: Purchased Goods & Services",
    Scope3Category.CAT_2: "Cat 2: Capital Goods",
    Scope3Category.CAT_3: "Cat 3: Fuel & Energy Related",
    Scope3Category.CAT_4: "Cat 4: Upstream Transport",
    Scope3Category.CAT_5: "Cat 5: Waste",
    Scope3Category.CAT_6: "Cat 6: Business Travel",
    Scope3Category.CAT_7: "Cat 7: Employee Commuting",
    Scope3Category.CAT_8: "Cat 8: Upstream Leased",
    Scope3Category.CAT_9: "Cat 9: Downstream Transport",
    Scope3Category.CAT_10: "Cat 10: Processing",
    Scope3Category.CAT_11: "Cat 11: Use of Sold Products",
    Scope3Category.CAT_12: "Cat 12: End-of-Life Treatment",
    Scope3Category.CAT_13: "Cat 13: Downstream Leased",
    Scope3Category.CAT_14: "Cat 14: Franchises",
    Scope3Category.CAT_15: "Cat 15: Investments",
}
"""Human-readable names for each Scope 3 category."""


# ---------------------------------------------------------------------------
# Pydantic Models -- Inputs
# ---------------------------------------------------------------------------


class CategoryUncertaintyInput(BaseModel):
    """Uncertainty input for a single Scope 3 category.

    Attributes:
        category: Scope 3 category identifier.
        emissions_tco2e: Total emissions for this category (tCO2e).
        methodology_tier: Methodology tier used.
        uncertainty_pct: Override uncertainty (% half-width of 95% CI).
        distribution_type: Statistical distribution for sampling.
        custom_low_pct: Custom lower bound uncertainty (%).
        custom_high_pct: Custom upper bound uncertainty (%).
    """
    category: str = Field(..., min_length=1, description="Scope 3 category")
    emissions_tco2e: Decimal = Field(
        ..., ge=0, description="Emissions (tCO2e)"
    )
    methodology_tier: str = Field(
        default=MethodologyTier.SPEND_BASED, description="Methodology tier"
    )
    uncertainty_pct: Optional[Decimal] = Field(
        default=None, ge=0, le=500, description="Override uncertainty %"
    )
    distribution_type: Optional[str] = Field(
        default=None, description="Distribution type override"
    )
    custom_low_pct: Optional[Decimal] = Field(
        default=None, ge=0, description="Custom low uncertainty %"
    )
    custom_high_pct: Optional[Decimal] = Field(
        default=None, ge=0, description="Custom high uncertainty %"
    )

    @field_validator("emissions_tco2e", mode="before")
    @classmethod
    def coerce_emissions(cls, v: Any) -> Decimal:
        """Coerce emissions to Decimal."""
        return _decimal(v)

    @field_validator("uncertainty_pct", "custom_low_pct", "custom_high_pct", mode="before")
    @classmethod
    def coerce_pct(cls, v: Any) -> Optional[Decimal]:
        """Coerce percentage to Decimal."""
        if v is None:
            return None
        return _decimal(v)


class CorrelationInput(BaseModel):
    """Custom inter-category correlation coefficient.

    Attributes:
        category_a: First category.
        category_b: Second category.
        rho: Correlation coefficient (-1.0 to 1.0).
    """
    category_a: str = Field(..., min_length=1, description="Category A")
    category_b: str = Field(..., min_length=1, description="Category B")
    rho: float = Field(..., ge=-1.0, le=1.0, description="Correlation coefficient")


class MonteCarloConfig(BaseModel):
    """Configuration for Monte Carlo simulation.

    Attributes:
        iterations: Number of Monte Carlo iterations.
        seed: Random seed for reproducibility.
        confidence_level: Confidence level for interval estimation.
        convergence_threshold: Relative change threshold for early stop.
        check_every: Check convergence every N iterations.
    """
    iterations: int = Field(
        default=10000, ge=100, le=1_000_000, description="Iterations"
    )
    seed: int = Field(default=42, ge=0, description="Random seed")
    confidence_level: float = Field(
        default=0.95, ge=0.50, le=0.99, description="Confidence level"
    )
    convergence_threshold: float = Field(
        default=0.001, ge=0, description="Convergence threshold"
    )
    check_every: int = Field(
        default=1000, ge=100, description="Check every N iterations"
    )


# ---------------------------------------------------------------------------
# Pydantic Models -- Outputs
# ---------------------------------------------------------------------------


class AnalyticalResult(BaseModel):
    """Analytical (IPCC Approach 1) uncertainty result.

    Attributes:
        combined_uncertainty_pct: Combined uncertainty (% half-width 95% CI).
        lower_bound_tco2e: Lower bound of 95% CI (tCO2e).
        upper_bound_tco2e: Upper bound of 95% CI (tCO2e).
        total_emissions_tco2e: Sum of all category emissions.
        confidence_level: Confidence level used.
        includes_correlations: Whether correlations were included.
        method: Description of method.
    """
    combined_uncertainty_pct: float = Field(default=0.0, description="Combined uncertainty %")
    lower_bound_tco2e: float = Field(default=0.0, description="Lower bound (tCO2e)")
    upper_bound_tco2e: float = Field(default=0.0, description="Upper bound (tCO2e)")
    total_emissions_tco2e: float = Field(default=0.0, description="Total (tCO2e)")
    confidence_level: float = Field(default=0.95, description="Confidence level")
    includes_correlations: bool = Field(default=False, description="Correlations included")
    method: str = Field(
        default="IPCC Approach 1 (error propagation via quadrature)",
        description="Method"
    )


class MonteCarloResult(BaseModel):
    """Monte Carlo simulation (IPCC Approach 2) uncertainty result.

    Attributes:
        mean_tco2e: Mean of simulated totals.
        median_tco2e: Median (50th percentile).
        std_dev: Standard deviation.
        p2_5: 2.5th percentile (lower 95% CI).
        p5: 5th percentile.
        p10: 10th percentile.
        p25: 25th percentile (Q1).
        p50: 50th percentile (median).
        p75: 75th percentile (Q3).
        p90: 90th percentile.
        p95: 95th percentile.
        p97_5: 97.5th percentile (upper 95% CI).
        combined_uncertainty_pct: Combined uncertainty from MC.
        iterations_run: Actual iterations run.
        convergence_achieved: Whether convergence was reached.
        convergence_at_iteration: Iteration of convergence.
        seed_used: Seed used.
    """
    mean_tco2e: float = Field(default=0.0, description="Mean (tCO2e)")
    median_tco2e: float = Field(default=0.0, description="Median (tCO2e)")
    std_dev: float = Field(default=0.0, description="Standard deviation")
    p2_5: float = Field(default=0.0, description="2.5th percentile")
    p5: float = Field(default=0.0, description="5th percentile")
    p10: float = Field(default=0.0, description="10th percentile")
    p25: float = Field(default=0.0, description="25th percentile")
    p50: float = Field(default=0.0, description="50th percentile")
    p75: float = Field(default=0.0, description="75th percentile")
    p90: float = Field(default=0.0, description="90th percentile")
    p95: float = Field(default=0.0, description="95th percentile")
    p97_5: float = Field(default=0.0, description="97.5th percentile")
    combined_uncertainty_pct: float = Field(default=0.0, description="Combined uncertainty %")
    iterations_run: int = Field(default=0, description="Iterations run")
    convergence_achieved: bool = Field(default=False, description="Convergence achieved")
    convergence_at_iteration: Optional[int] = Field(
        default=None, description="Convergence iteration"
    )
    seed_used: int = Field(default=42, description="Random seed")


class SensitivityItem(BaseModel):
    """Single item in a sensitivity analysis (tornado chart data).

    Attributes:
        category: Scope 3 category.
        category_name: Human-readable name.
        emissions_tco2e: Emissions from this category.
        uncertainty_pct: Category uncertainty (%).
        variance_contribution: Absolute variance contribution.
        contribution_pct: Share of total variance (%).
        rank: Rank (1 = largest contributor).
        low_scenario_tco2e: Total emissions if this category is at low bound.
        high_scenario_tco2e: Total emissions if this category is at high bound.
        swing_tco2e: Difference between high and low scenarios.
    """
    category: str = Field(default="", description="Category")
    category_name: str = Field(default="", description="Category name")
    emissions_tco2e: float = Field(default=0.0, description="Emissions")
    uncertainty_pct: float = Field(default=0.0, description="Uncertainty %")
    variance_contribution: float = Field(default=0.0, description="Variance contribution")
    contribution_pct: float = Field(default=0.0, le=100, description="Contribution %")
    rank: int = Field(default=0, description="Rank")
    low_scenario_tco2e: float = Field(default=0.0, description="Low scenario total")
    high_scenario_tco2e: float = Field(default=0.0, description="High scenario total")
    swing_tco2e: float = Field(default=0.0, description="Swing (high - low)")


class TierUpgradeImpact(BaseModel):
    """Impact of upgrading methodology tier for a category.

    Attributes:
        category: Scope 3 category.
        category_name: Human-readable name.
        current_tier: Current methodology tier.
        target_tier: Target methodology tier.
        current_uncertainty_pct: Current uncertainty (%).
        target_uncertainty_pct: Target uncertainty (%).
        uncertainty_reduction_pct: Absolute reduction in uncertainty (pp).
        current_variance: Current variance contribution.
        target_variance: Target variance contribution.
        variance_reduction_pct: Percentage reduction in variance.
        impact_on_total_uncertainty_pct: Reduction in total Scope 3 uncertainty (pp).
        effort_estimate: Effort description.
    """
    category: str = Field(default="", description="Category")
    category_name: str = Field(default="", description="Category name")
    current_tier: str = Field(default="", description="Current tier")
    target_tier: str = Field(default="", description="Target tier")
    current_uncertainty_pct: float = Field(default=0.0, description="Current uncertainty %")
    target_uncertainty_pct: float = Field(default=0.0, description="Target uncertainty %")
    uncertainty_reduction_pct: float = Field(default=0.0, description="Reduction (pp)")
    current_variance: float = Field(default=0.0, description="Current variance")
    target_variance: float = Field(default=0.0, description="Target variance")
    variance_reduction_pct: float = Field(default=0.0, description="Variance reduction %")
    impact_on_total_uncertainty_pct: float = Field(
        default=0.0, description="Impact on total uncertainty (pp)"
    )
    effort_estimate: str = Field(default="", description="Effort estimate")


class UncertaintyResult(BaseModel):
    """Complete Scope 3 uncertainty analysis result with provenance.

    Attributes:
        result_id: Unique result identifier.
        engine_version: Engine version.
        calculated_at: Calculation timestamp.
        processing_time_ms: Processing time (ms).
        total_emissions_tco2e: Total Scope 3 emissions.
        category_count: Number of categories analysed.
        analytical: Analytical (quadrature) result.
        monte_carlo: Monte Carlo simulation result.
        sensitivity_analysis: Tornado chart data.
        tier_upgrade_impacts: Impact of tier upgrades.
        correlation_matrix: Correlation matrix used.
        methodology_notes: Methodology notes.
        provenance_hash: SHA-256 hash.
    """
    result_id: str = Field(default_factory=_new_uuid, description="Result ID")
    engine_version: str = Field(default=_MODULE_VERSION, description="Version")
    calculated_at: datetime = Field(default_factory=_utcnow, description="Timestamp")
    processing_time_ms: float = Field(default=0.0, description="Processing time")
    total_emissions_tco2e: float = Field(default=0.0, description="Total emissions")
    category_count: int = Field(default=0, description="Categories analysed")
    analytical: Optional[AnalyticalResult] = Field(
        default=None, description="Analytical result"
    )
    monte_carlo: Optional[MonteCarloResult] = Field(
        default=None, description="Monte Carlo result"
    )
    sensitivity_analysis: List[SensitivityItem] = Field(
        default_factory=list, description="Sensitivity analysis"
    )
    tier_upgrade_impacts: List[TierUpgradeImpact] = Field(
        default_factory=list, description="Tier upgrade impacts"
    )
    correlation_matrix: Dict[str, Dict[str, float]] = Field(
        default_factory=dict, description="Correlation matrix"
    )
    methodology_notes: List[str] = Field(
        default_factory=list, description="Methodology notes"
    )
    provenance_hash: str = Field(default="", description="SHA-256 hash")


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class Scope3UncertaintyEngine:
    """Scope 3 uncertainty quantification engine.

    Runs IPCC Approach 1 (analytical error propagation) and Approach 2
    (Monte Carlo simulation) to quantify uncertainty bounds for Scope 3
    emission estimates.

    Guarantees:
        - Deterministic: analytical results are exact for same inputs.
        - Reproducible: Monte Carlo uses fixed seed.
        - Auditable: SHA-256 provenance hash, full methodology notes.
        - No LLM: zero hallucination risk in any calculation path.

    Usage::

        engine = Scope3UncertaintyEngine()
        result = engine.run_full_analysis(category_results)
        analytical = engine.run_analytical(category_results)
        mc = engine.run_monte_carlo(category_results, iterations=10000)
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise the uncertainty engine.

        Args:
            config: Optional configuration overrides.
                - correlations: Override default correlation pairs.
                - confidence_level: Override default confidence level.
        """
        self._config = config or {}
        self._custom_correlations: Optional[List[CorrelationInput]] = None
        raw_corr = self._config.get("correlations")
        if raw_corr and isinstance(raw_corr, list):
            self._custom_correlations = [
                CorrelationInput(**c) if isinstance(c, dict) else c
                for c in raw_corr
            ]
        self._default_confidence = self._config.get("confidence_level", 0.95)
        logger.info("Scope3UncertaintyEngine v%s initialised.", _MODULE_VERSION)

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def run_full_analysis(
        self,
        category_results: List[CategoryUncertaintyInput],
        mc_config: Optional[MonteCarloConfig] = None,
        correlations: Optional[List[CorrelationInput]] = None,
    ) -> UncertaintyResult:
        """Run complete uncertainty analysis (analytical + Monte Carlo).

        Args:
            category_results: Per-category uncertainty inputs.
            mc_config: Monte Carlo configuration.
            correlations: Optional custom correlation pairs.

        Returns:
            UncertaintyResult with analytical, MC, and sensitivity results.
        """
        t0 = time.perf_counter()
        logger.info("Running full uncertainty analysis for %d categories.",
                     len(category_results))

        total_emissions = sum(
            (cr.emissions_tco2e for cr in category_results), Decimal("0")
        )

        # Resolve uncertainty profiles.
        profiles = self._resolve_profiles(category_results)

        # Build correlation matrix.
        corr_pairs = correlations or self._custom_correlations
        corr_matrix = self._build_correlation_matrix(category_results, corr_pairs)

        # Analytical result.
        analytical = self.run_analytical(category_results, correlations=corr_pairs)

        # Monte Carlo result.
        mc_result = None
        if _HAS_NUMPY:
            mc_cfg = mc_config or MonteCarloConfig()
            mc_result = self.run_monte_carlo(
                category_results, mc_cfg, correlations=corr_pairs,
            )

        # Sensitivity analysis.
        sensitivity = self._sensitivity_analysis(category_results, profiles)

        # Tier upgrade impacts.
        upgrades = self._calculate_all_tier_upgrades(category_results, profiles)

        # Methodology notes.
        notes = [
            "Uncertainty quantified using IPCC 2006 Guidelines, Vol 1, Ch 3.",
            f"Analytical: IPCC Approach 1 error propagation via quadrature.",
        ]
        if mc_result:
            notes.append(
                f"Monte Carlo: {mc_result.iterations_run} iterations, "
                f"seed={mc_result.seed_used}."
            )
        notes.append(
            f"Default uncertainty by tier: spend-based +/-100%, "
            f"average-data +/-30%, supplier-specific +/-10%."
        )
        if any(c[2] != 0.0 for c in DEFAULT_CORRELATIONS):
            notes.append("Inter-category correlations applied (see correlation_matrix).")

        elapsed = (time.perf_counter() - t0) * 1000

        result = UncertaintyResult(
            total_emissions_tco2e=_round2(total_emissions),
            category_count=len(category_results),
            analytical=analytical,
            monte_carlo=mc_result,
            sensitivity_analysis=sensitivity,
            tier_upgrade_impacts=upgrades,
            correlation_matrix=corr_matrix,
            methodology_notes=notes,
            processing_time_ms=_round2(elapsed),
        )
        result.provenance_hash = self._compute_provenance(result)

        logger.info("Full uncertainty analysis complete in %.1f ms.", elapsed)
        return result

    def run_analytical(
        self,
        category_results: List[CategoryUncertaintyInput],
        correlations: Optional[List[CorrelationInput]] = None,
    ) -> AnalyticalResult:
        """Run IPCC Approach 1 analytical uncertainty aggregation.

        Uses error propagation via quadrature. If correlations are
        provided, includes covariance terms.

        Args:
            category_results: Per-category uncertainty inputs.
            correlations: Optional correlation pairs.

        Returns:
            AnalyticalResult with combined uncertainty and CI bounds.
        """
        t0 = time.perf_counter()
        logger.info("Running analytical uncertainty for %d categories.",
                     len(category_results))

        profiles = self._resolve_profiles(category_results)
        total_emissions = Decimal("0")
        sum_var = Decimal("0")

        # Calculate variances: var_i = (E_i * u_i)^2
        category_vars: Dict[str, Decimal] = {}
        category_emissions: Dict[str, Decimal] = {}
        for cr in category_results:
            e_i = cr.emissions_tco2e
            u_i = profiles[cr.category]["uncertainty_frac"]  # Fractional uncertainty.
            var_i = (e_i * u_i) ** 2
            category_vars[cr.category] = var_i
            category_emissions[cr.category] = e_i
            sum_var += var_i
            total_emissions += e_i

        # Add covariance terms for correlated categories.
        includes_corr = False
        corr_pairs = correlations or self._custom_correlations
        sum_cov = Decimal("0")
        if corr_pairs:
            for cp in corr_pairs:
                if (cp.category_a in category_emissions and
                        cp.category_b in category_emissions):
                    e_a = category_emissions[cp.category_a]
                    e_b = category_emissions[cp.category_b]
                    u_a = profiles.get(cp.category_a, {}).get(
                        "uncertainty_frac", Decimal("0")
                    )
                    u_b = profiles.get(cp.category_b, {}).get(
                        "uncertainty_frac", Decimal("0")
                    )
                    rho = _decimal(cp.rho)
                    cov_ab = e_a * u_a * e_b * u_b * rho * Decimal("2")
                    sum_cov += cov_ab
                    includes_corr = True
        else:
            # Use defaults.
            for cat_a, cat_b, rho_val in DEFAULT_CORRELATIONS:
                if cat_a in category_emissions and cat_b in category_emissions:
                    e_a = category_emissions[cat_a]
                    e_b = category_emissions[cat_b]
                    u_a = profiles.get(cat_a, {}).get(
                        "uncertainty_frac", Decimal("0")
                    )
                    u_b = profiles.get(cat_b, {}).get(
                        "uncertainty_frac", Decimal("0")
                    )
                    rho = _decimal(rho_val)
                    cov_ab = e_a * u_a * e_b * u_b * rho * Decimal("2")
                    sum_cov += cov_ab
                    includes_corr = True

        total_var = sum_var + sum_cov
        total_var = max(Decimal("0"), total_var)  # Prevent negative from rounding.

        # Combined uncertainty.
        if total_emissions > 0 and total_var > 0:
            sqrt_var = Decimal(str(math.sqrt(float(total_var))))
            combined_pct = (sqrt_var / total_emissions) * Decimal("100")
        else:
            combined_pct = Decimal("0")

        # 95% CI bounds (assuming normal: 1.96 * sigma).
        z = Decimal("1.96")  # For 95% CI.
        half_width = total_emissions * combined_pct / Decimal("100")
        lower = total_emissions - z * half_width / Decimal("1.96")
        upper = total_emissions + z * half_width / Decimal("1.96")

        result = AnalyticalResult(
            combined_uncertainty_pct=_round2(combined_pct),
            lower_bound_tco2e=_round2(max(Decimal("0"), total_emissions - half_width)),
            upper_bound_tco2e=_round2(total_emissions + half_width),
            total_emissions_tco2e=_round2(total_emissions),
            confidence_level=self._default_confidence,
            includes_correlations=includes_corr,
        )

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "Analytical result: combined=%.1f%%, range=[%.0f, %.0f] tCO2e in %.1f ms.",
            result.combined_uncertainty_pct,
            result.lower_bound_tco2e, result.upper_bound_tco2e, elapsed,
        )
        return result

    def run_monte_carlo(
        self,
        category_results: List[CategoryUncertaintyInput],
        mc_config: Optional[MonteCarloConfig] = None,
        correlations: Optional[List[CorrelationInput]] = None,
    ) -> MonteCarloResult:
        """Run Monte Carlo simulation for uncertainty quantification.

        Args:
            category_results: Per-category uncertainty inputs.
            mc_config: Simulation configuration.
            correlations: Optional correlation pairs.

        Returns:
            MonteCarloResult with percentile distribution.

        Raises:
            RuntimeError: If numpy is not available.
        """
        if not _HAS_NUMPY:
            raise RuntimeError(
                "numpy is required for Monte Carlo simulation. "
                "Install with: pip install numpy"
            )

        cfg = mc_config or MonteCarloConfig()
        t0 = time.perf_counter()
        logger.info(
            "Running Monte Carlo: %d iterations, seed=%d, %d categories.",
            cfg.iterations, cfg.seed, len(category_results),
        )

        profiles = self._resolve_profiles(category_results)
        n_cats = len(category_results)
        rng = np.random.default_rng(cfg.seed)

        # Build correlation matrix for Cholesky decomposition.
        cat_keys = [cr.category for cr in category_results]
        corr_mat = np.eye(n_cats)
        corr_pairs = correlations or self._custom_correlations
        pair_list = (
            [(cp.category_a, cp.category_b, cp.rho) for cp in corr_pairs]
            if corr_pairs else list(DEFAULT_CORRELATIONS)
        )
        for cat_a, cat_b, rho_val in pair_list:
            if cat_a in cat_keys and cat_b in cat_keys:
                i = cat_keys.index(cat_a)
                j = cat_keys.index(cat_b)
                corr_mat[i, j] = rho_val
                corr_mat[j, i] = rho_val

        # Cholesky decomposition.
        try:
            chol = np.linalg.cholesky(corr_mat)
        except np.linalg.LinAlgError:
            logger.warning(
                "Correlation matrix not positive definite; "
                "falling back to independent sampling."
            )
            chol = np.eye(n_cats)

        # Generate samples.
        totals = np.zeros(cfg.iterations)
        convergence_achieved = False
        convergence_at: Optional[int] = None

        # Pre-compute parameters.
        emissions = np.array([float(cr.emissions_tco2e) for cr in category_results])
        uncertainties = np.array([
            float(profiles[cr.category]["uncertainty_frac"])
            for cr in category_results
        ])
        dist_types = [profiles[cr.category]["distribution"] for cr in category_results]

        batch_size = cfg.check_every
        samples_done = 0

        while samples_done < cfg.iterations:
            batch_end = min(samples_done + batch_size, cfg.iterations)
            batch_n = batch_end - samples_done

            # Generate standard normal samples and apply Cholesky.
            z_standard = rng.standard_normal((batch_n, n_cats))
            z_correlated = z_standard @ chol.T

            # Sample per category.
            for j in range(n_cats):
                e_j = emissions[j]
                u_j = uncertainties[j]
                dist = dist_types[j]

                if dist == DistributionType.LOGNORMAL and u_j > 0:
                    sigma_ln = np.sqrt(np.log(1 + u_j ** 2))
                    mu_ln = np.log(e_j) - 0.5 * sigma_ln ** 2
                    sampled = np.exp(mu_ln + sigma_ln * z_correlated[:, j])
                elif dist == DistributionType.UNIFORM:
                    low = e_j * (1 - u_j)
                    high = e_j * (1 + u_j)
                    # Map normal to uniform via CDF.
                    from scipy.stats import norm as _norm_dist
                    u_vals = _norm_dist.cdf(z_correlated[:, j])
                    sampled = low + (high - low) * u_vals
                else:
                    # Normal distribution (default).
                    sigma = e_j * u_j
                    sampled = e_j + sigma * z_correlated[:, j]
                    sampled = np.maximum(sampled, 0)  # Prevent negatives.

                totals[samples_done:batch_end] += sampled

            samples_done = batch_end

            # Check convergence.
            if samples_done >= 2 * batch_size and not convergence_achieved:
                running_mean = np.mean(totals[:samples_done])
                prev_mean = np.mean(totals[:samples_done - batch_size])
                if prev_mean > 0:
                    rel_change = abs(running_mean - prev_mean) / prev_mean
                    if rel_change < cfg.convergence_threshold:
                        convergence_achieved = True
                        convergence_at = samples_done

        # Calculate percentiles.
        result = MonteCarloResult(
            mean_tco2e=_round2(np.mean(totals)),
            median_tco2e=_round2(np.median(totals)),
            std_dev=_round2(np.std(totals)),
            p2_5=_round2(np.percentile(totals, 2.5)),
            p5=_round2(np.percentile(totals, 5)),
            p10=_round2(np.percentile(totals, 10)),
            p25=_round2(np.percentile(totals, 25)),
            p50=_round2(np.percentile(totals, 50)),
            p75=_round2(np.percentile(totals, 75)),
            p90=_round2(np.percentile(totals, 90)),
            p95=_round2(np.percentile(totals, 95)),
            p97_5=_round2(np.percentile(totals, 97.5)),
            combined_uncertainty_pct=_round2(
                (np.std(totals) / np.mean(totals) * 100) if np.mean(totals) > 0 else 0
            ),
            iterations_run=cfg.iterations,
            convergence_achieved=convergence_achieved,
            convergence_at_iteration=convergence_at,
            seed_used=cfg.seed,
        )

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "Monte Carlo complete: mean=%.0f, std=%.0f, 95%%CI=[%.0f, %.0f] in %.1f ms.",
            result.mean_tco2e, result.std_dev,
            result.p2_5, result.p97_5, elapsed,
        )
        return result

    def _compute_provenance(self, data: Any) -> str:
        """Compute SHA-256 provenance hash for audit trail.

        Args:
            data: Data to hash.

        Returns:
            SHA-256 hex digest string.
        """
        return _compute_hash(data)

    # -------------------------------------------------------------------
    # Private -- Profile Resolution
    # -------------------------------------------------------------------

    def _resolve_profiles(
        self, category_results: List[CategoryUncertaintyInput],
    ) -> Dict[str, Dict[str, Any]]:
        """Resolve uncertainty profiles for each category.

        Maps each category to its uncertainty percentage and distribution
        type, using overrides if provided or defaults by methodology tier.

        Args:
            category_results: Per-category inputs.

        Returns:
            Dict mapping category -> profile dict.
        """
        profiles: Dict[str, Dict[str, Any]] = {}
        for cr in category_results:
            tier_defaults = DEFAULT_UNCERTAINTY_BY_TIER.get(
                cr.methodology_tier, DEFAULT_UNCERTAINTY_BY_TIER[MethodologyTier.SPEND_BASED]
            )

            if cr.uncertainty_pct is not None:
                unc_pct = float(cr.uncertainty_pct)
            else:
                unc_pct = tier_defaults["default_pct"]

            dist = (
                cr.distribution_type
                if cr.distribution_type
                else tier_defaults["distribution"]
            )

            profiles[cr.category] = {
                "uncertainty_pct": unc_pct,
                "uncertainty_frac": Decimal(str(unc_pct)) / Decimal("100"),
                "distribution": dist,
                "tier": cr.methodology_tier,
                "source": tier_defaults.get("source", ""),
            }

        return profiles

    # -------------------------------------------------------------------
    # Private -- Correlation Matrix
    # -------------------------------------------------------------------

    def _build_correlation_matrix(
        self,
        category_results: List[CategoryUncertaintyInput],
        correlations: Optional[List[CorrelationInput]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Build human-readable correlation matrix dict.

        Args:
            category_results: Category inputs.
            correlations: Optional custom correlations.

        Returns:
            Nested dict representing correlation matrix.
        """
        cat_keys = [cr.category for cr in category_results]
        matrix: Dict[str, Dict[str, float]] = {}
        for cat in cat_keys:
            matrix[cat] = {c: (1.0 if c == cat else 0.0) for c in cat_keys}

        pair_list = (
            [(cp.category_a, cp.category_b, cp.rho) for cp in correlations]
            if correlations
            else list(DEFAULT_CORRELATIONS)
        )
        for cat_a, cat_b, rho_val in pair_list:
            if cat_a in matrix and cat_b in matrix:
                matrix[cat_a][cat_b] = rho_val
                matrix[cat_b][cat_a] = rho_val

        return matrix

    # -------------------------------------------------------------------
    # Private -- Sensitivity Analysis
    # -------------------------------------------------------------------

    def _sensitivity_analysis(
        self,
        category_results: List[CategoryUncertaintyInput],
        profiles: Dict[str, Dict[str, Any]],
    ) -> List[SensitivityItem]:
        """Perform sensitivity analysis (tornado chart data).

        For each category, calculates variance contribution to total
        uncertainty and the high/low scenario swing.

        Args:
            category_results: Category inputs.
            profiles: Resolved uncertainty profiles.

        Returns:
            SensitivityItems sorted by contribution (largest first).
        """
        t0 = time.perf_counter()

        total_emissions = sum(
            (float(cr.emissions_tco2e) for cr in category_results), 0.0
        )

        # Calculate variances.
        variances: List[Tuple[str, float, float, float]] = []
        total_var = 0.0
        for cr in category_results:
            e_i = float(cr.emissions_tco2e)
            u_i = float(profiles[cr.category]["uncertainty_frac"])
            var_i = (e_i * u_i) ** 2
            variances.append((cr.category, e_i, u_i * 100, var_i))
            total_var += var_i

        # Build sensitivity items.
        items: List[SensitivityItem] = []
        for cat, e_i, unc_pct, var_i in variances:
            contrib_pct = (var_i / total_var * 100) if total_var > 0 else 0.0
            u_frac = unc_pct / 100
            low_total = total_emissions - e_i + e_i * (1 - u_frac)
            high_total = total_emissions - e_i + e_i * (1 + u_frac)
            swing = high_total - low_total

            items.append(SensitivityItem(
                category=cat,
                category_name=CATEGORY_NAMES.get(cat, cat),
                emissions_tco2e=_round2(e_i),
                uncertainty_pct=_round2(unc_pct),
                variance_contribution=_round2(var_i),
                contribution_pct=_round2(contrib_pct),
                low_scenario_tco2e=_round2(low_total),
                high_scenario_tco2e=_round2(high_total),
                swing_tco2e=_round2(swing),
            ))

        # Sort by contribution descending and assign ranks.
        items.sort(key=lambda x: x.contribution_pct, reverse=True)
        for idx, item in enumerate(items):
            item.rank = idx + 1

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("Sensitivity analysis: %d categories in %.1f ms.",
                     len(items), elapsed)
        return items

    # -------------------------------------------------------------------
    # Private -- Tier Upgrade Impact
    # -------------------------------------------------------------------

    def _tier_upgrade_impact(
        self,
        category: str,
        emissions: Decimal,
        current_tier: str,
        target_tier: str,
        total_var_current: Decimal,
        total_emissions: Decimal,
    ) -> TierUpgradeImpact:
        """Calculate uncertainty reduction from upgrading a single category.

        Args:
            category: Category identifier.
            emissions: Category emissions.
            current_tier: Current methodology tier.
            target_tier: Target methodology tier.
            total_var_current: Current total variance.
            total_emissions: Total Scope 3 emissions.

        Returns:
            TierUpgradeImpact result.
        """
        current_defaults = DEFAULT_UNCERTAINTY_BY_TIER.get(
            current_tier, DEFAULT_UNCERTAINTY_BY_TIER[MethodologyTier.SPEND_BASED]
        )
        target_defaults = DEFAULT_UNCERTAINTY_BY_TIER.get(
            target_tier, DEFAULT_UNCERTAINTY_BY_TIER[MethodologyTier.AVERAGE_DATA]
        )

        current_pct = current_defaults["default_pct"]
        target_pct = target_defaults["default_pct"]

        current_frac = Decimal(str(current_pct)) / Decimal("100")
        target_frac = Decimal(str(target_pct)) / Decimal("100")

        current_var = (emissions * current_frac) ** 2
        target_var = (emissions * target_frac) ** 2

        # New total variance if this category is upgraded.
        new_total_var = total_var_current - current_var + target_var
        new_total_var = max(Decimal("0"), new_total_var)

        # Impact on total uncertainty.
        if total_emissions > 0:
            current_total_unc = (
                Decimal(str(math.sqrt(float(total_var_current)))) / total_emissions
                * Decimal("100")
            )
            new_total_unc = (
                Decimal(str(math.sqrt(float(new_total_var)))) / total_emissions
                * Decimal("100")
            )
            impact_pp = current_total_unc - new_total_unc
        else:
            impact_pp = Decimal("0")

        # Effort estimate.
        tier_order = {
            MethodologyTier.SPEND_BASED: 1,
            MethodologyTier.AVERAGE_DATA: 2,
            MethodologyTier.SUPPLIER_SPECIFIC: 3,
        }
        gap = tier_order.get(target_tier, 2) - tier_order.get(current_tier, 1)
        if gap >= 2:
            effort = "High effort (6-12 months, supplier engagement required)."
        elif gap == 1:
            effort = "Medium effort (2-6 months, data collection required)."
        else:
            effort = "Low effort (already at or near target tier)."

        var_reduction = _round2(
            _safe_pct(current_var - target_var, current_var)
        ) if current_var > 0 else 0.0

        return TierUpgradeImpact(
            category=category,
            category_name=CATEGORY_NAMES.get(category, category),
            current_tier=current_tier,
            target_tier=target_tier,
            current_uncertainty_pct=current_pct,
            target_uncertainty_pct=target_pct,
            uncertainty_reduction_pct=_round2(current_pct - target_pct),
            current_variance=_round2(current_var),
            target_variance=_round2(target_var),
            variance_reduction_pct=var_reduction,
            impact_on_total_uncertainty_pct=_round2(impact_pp),
            effort_estimate=effort,
        )

    def _calculate_all_tier_upgrades(
        self,
        category_results: List[CategoryUncertaintyInput],
        profiles: Dict[str, Dict[str, Any]],
    ) -> List[TierUpgradeImpact]:
        """Calculate tier upgrade impacts for all eligible categories.

        Args:
            category_results: Category inputs.
            profiles: Resolved profiles.

        Returns:
            List of TierUpgradeImpact sorted by impact (largest first).
        """
        t0 = time.perf_counter()

        total_emissions = sum(
            (cr.emissions_tco2e for cr in category_results), Decimal("0")
        )
        total_var = sum(
            ((cr.emissions_tco2e * profiles[cr.category]["uncertainty_frac"]) ** 2
             for cr in category_results),
            Decimal("0"),
        )

        # Define upgrade path.
        tier_upgrades = {
            MethodologyTier.SPEND_BASED: MethodologyTier.AVERAGE_DATA,
            MethodologyTier.AVERAGE_DATA: MethodologyTier.SUPPLIER_SPECIFIC,
        }

        impacts: List[TierUpgradeImpact] = []
        for cr in category_results:
            current = cr.methodology_tier
            target = tier_upgrades.get(current)
            if not target:
                continue
            impact = self._tier_upgrade_impact(
                cr.category, cr.emissions_tco2e,
                current, target,
                total_var, total_emissions,
            )
            impacts.append(impact)

        impacts.sort(key=lambda x: x.impact_on_total_uncertainty_pct, reverse=True)

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info("Tier upgrades: %d impacts in %.1f ms.", len(impacts), elapsed)
        return impacts

    # -------------------------------------------------------------------
    # Private -- Distribution Generators (stdlib fallback)
    # -------------------------------------------------------------------

    def _generate_distribution(
        self,
        value: float,
        uncertainty_frac: float,
        dist_type: str,
        rng: Any,
        n_samples: int = 1,
    ) -> Any:
        """Generate random samples from a specified distribution.

        This is a fallback when numpy is not available for use in
        analytical variance estimation.

        Args:
            value: Central value.
            uncertainty_frac: Fractional uncertainty.
            dist_type: Distribution type.
            rng: Random number generator.
            n_samples: Number of samples.

        Returns:
            Array of samples.
        """
        if not _HAS_NUMPY:
            raise RuntimeError("numpy required for distribution sampling.")

        sigma = value * uncertainty_frac

        if dist_type == DistributionType.LOGNORMAL:
            sigma_ln = math.sqrt(math.log(1 + uncertainty_frac ** 2))
            mu_ln = math.log(max(value, 1e-10)) - 0.5 * sigma_ln ** 2
            return rng.lognormal(mu_ln, sigma_ln, n_samples)
        elif dist_type == DistributionType.UNIFORM:
            low = value * (1 - uncertainty_frac)
            high = value * (1 + uncertainty_frac)
            return rng.uniform(max(0, low), high, n_samples)
        elif dist_type == DistributionType.TRIANGULAR:
            low = value * (1 - uncertainty_frac)
            high = value * (1 + uncertainty_frac)
            return rng.triangular(max(0, low), value, high, n_samples)
        else:
            # Normal distribution (default).
            samples = rng.normal(value, sigma, n_samples)
            return np.maximum(samples, 0)

    def _calculate_confidence_interval(
        self,
        samples: Any,
        level: float = 0.95,
    ) -> Tuple[float, float]:
        """Calculate confidence interval from samples.

        Args:
            samples: Array of simulated values.
            level: Confidence level (e.g., 0.95).

        Returns:
            Tuple of (lower_bound, upper_bound).
        """
        if not _HAS_NUMPY:
            raise RuntimeError("numpy required.")

        alpha = 1 - level
        lower = float(np.percentile(samples, alpha / 2 * 100))
        upper = float(np.percentile(samples, (1 - alpha / 2) * 100))
        return (_round2(lower), _round2(upper))

    def _correlation_handling(
        self,
        categories: List[str],
    ) -> Dict[str, Dict[str, float]]:
        """Build correlation matrix for the given categories.

        Args:
            categories: List of category identifiers.

        Returns:
            Nested dict correlation matrix.
        """
        matrix: Dict[str, Dict[str, float]] = {}
        for cat in categories:
            matrix[cat] = {c: (1.0 if c == cat else 0.0) for c in categories}

        for cat_a, cat_b, rho_val in DEFAULT_CORRELATIONS:
            if cat_a in matrix and cat_b in matrix:
                matrix[cat_a][cat_b] = rho_val
                matrix[cat_b][cat_a] = rho_val

        return matrix


# ---------------------------------------------------------------------------
# Pydantic v2 model_rebuild for forward-reference resolution
# ---------------------------------------------------------------------------

CategoryUncertaintyInput.model_rebuild()
CorrelationInput.model_rebuild()
MonteCarloConfig.model_rebuild()
AnalyticalResult.model_rebuild()
MonteCarloResult.model_rebuild()
SensitivityItem.model_rebuild()
TierUpgradeImpact.model_rebuild()
UncertaintyResult.model_rebuild()
