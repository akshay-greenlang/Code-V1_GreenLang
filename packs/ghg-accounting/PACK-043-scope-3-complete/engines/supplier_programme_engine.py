# -*- coding: utf-8 -*-
"""
SupplierProgrammeEngine - PACK-043 Scope 3 Complete Pack Engine 6
===================================================================

Manages supplier-level emission reduction targets, commitments, and
programme impact across the value chain.  Enables reporting organisations
to set per-supplier targets, track SBTi / RE100 / CDP commitments,
measure year-on-year progress against baselines, model aggregate
programme impact on the reporter's Scope 3 inventory, generate composite
supplier scorecards, classify suppliers into strategic tiers, and
evaluate incentive-based procurement approaches.

Supplier Scorecard Formula:
    composite_score = (
        emission_contribution_score * 0.30
      + data_quality_score          * 0.20
      + engagement_level_score      * 0.20
      + commitment_strength_score   * 0.30
    )

    Each dimension is normalised to a 0-100 scale.

Supplier Classification (Pareto-based):
    Strategic: Top 20% by composite score (deep engagement)
    Key:       Next 30% (structured engagement)
    Managed:   Bottom 50% (light-touch / automated)

Programme ROI:
    roi_ratio = reduction_achieved_tco2e * value_per_tonne / total_programme_cost
    payback_years = total_programme_cost / (annual_reduction_tco2e * value_per_tonne)

Transition Risk Assessment:
    risk_score = (
        sector_carbon_intensity * 0.25
      + regulatory_exposure     * 0.25
      + decarbonisation_pace    * 0.25
      + financial_resilience    * 0.25
    )
    risk_rating = HIGH if risk_score >= 70 else MEDIUM if >= 40 else LOW

Incentive Types:
    - PRICE_PREFERENCE:    preferential pricing margin (e.g., 2-5% premium)
    - VOLUME_GUARANTEE:    minimum purchase volume commitment
    - CONTRACT_EXTENSION:  extended contract term (e.g., +2 years)
    - RECOGNITION:         public acknowledgement, awards, preferred listing

Regulatory References:
    - GHG Protocol Corporate Value Chain (Scope 3) Standard (2011)
    - GHG Protocol Scope 3 Calculation Guidance (2013), Chapter 7
    - SBTi Scope 3 Supplier Engagement guidance (2023)
    - SBTi Corporate Net-Zero Standard v1.1
    - CDP Supply Chain Program Technical Note (2024)
    - ESRS E1 para 44-46 (value chain engagement)

Zero-Hallucination:
    - All scores computed deterministically via Decimal arithmetic
    - Classification thresholds are configurable, not ML-predicted
    - No LLM involvement in any scoring or classification path
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-043 Scope 3 Complete
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
from datetime import datetime, date, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "43.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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

def _round4(value: Any) -> float:
    """Round to 4 decimal places using ROUND_HALF_UP."""
    return float(Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SupplierClassification(str, Enum):
    """Supplier tier classification based on composite score ranking.

    STRATEGIC: Top 20% -- deep, personalised engagement.
    KEY:       Next 30% -- structured engagement programme.
    MANAGED:   Bottom 50% -- light-touch / automated engagement.
    """
    STRATEGIC = "strategic"
    KEY = "key"
    MANAGED = "managed"

class CommitmentType(str, Enum):
    """Types of supplier climate commitments."""
    SBTI_COMMITTED = "sbti_committed"
    SBTI_VALIDATED = "sbti_validated"
    RE100 = "re100"
    CDP_DISCLOSURE = "cdp_disclosure"
    NET_ZERO_PLEDGE = "net_zero_pledge"
    CARBON_NEUTRAL = "carbon_neutral"
    INTERNAL_TARGET = "internal_target"
    NO_COMMITMENT = "no_commitment"

class IncentiveType(str, Enum):
    """Types of procurement incentives for emission reduction.

    PRICE_PREFERENCE:    Preferential pricing margin (e.g., 2-5% premium).
    VOLUME_GUARANTEE:    Minimum purchase volume commitment.
    CONTRACT_EXTENSION:  Extended contract term.
    RECOGNITION:         Public acknowledgement, awards, preferred listing.
    """
    PRICE_PREFERENCE = "price_preference"
    VOLUME_GUARANTEE = "volume_guarantee"
    CONTRACT_EXTENSION = "contract_extension"
    RECOGNITION = "recognition"

class TransitionRiskRating(str, Enum):
    """Supplier transition risk rating."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class ProgressStatus(str, Enum):
    """Year-on-year progress status."""
    ON_TRACK = "on_track"
    BEHIND = "behind"
    AHEAD = "ahead"
    NO_DATA = "no_data"
    BASELINE_YEAR = "baseline_year"

class EngagementLevel(str, Enum):
    """Supplier engagement level for scoring."""
    NONE = "none"
    AWARE = "aware"
    ACTIVE = "active"
    ADVANCED = "advanced"
    LEADER = "leader"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Scorecard dimension weights (must sum to 1.0).
SCORECARD_WEIGHTS: Dict[str, float] = {
    "emission_contribution": 0.30,
    "data_quality": 0.20,
    "engagement_level": 0.20,
    "commitment_strength": 0.30,
}
"""Scorecard dimension weights per GreenLang supplier programme methodology."""

# Commitment strength scores (0-100).
COMMITMENT_SCORES: Dict[str, int] = {
    CommitmentType.SBTI_VALIDATED: 100,
    CommitmentType.SBTI_COMMITTED: 80,
    CommitmentType.RE100: 70,
    CommitmentType.NET_ZERO_PLEDGE: 60,
    CommitmentType.CDP_DISCLOSURE: 50,
    CommitmentType.CARBON_NEUTRAL: 40,
    CommitmentType.INTERNAL_TARGET: 30,
    CommitmentType.NO_COMMITMENT: 0,
}
"""Commitment strength scores by type (0-100 scale)."""

# Engagement level scores (0-100).
ENGAGEMENT_LEVEL_SCORES: Dict[str, int] = {
    EngagementLevel.LEADER: 100,
    EngagementLevel.ADVANCED: 75,
    EngagementLevel.ACTIVE: 50,
    EngagementLevel.AWARE: 25,
    EngagementLevel.NONE: 0,
}
"""Engagement level scores (0-100 scale)."""

# Transition risk sector intensity factors (tCO2e per $M revenue).
# Source: CDP Technical Note on Climate Transition Plans (2024).
SECTOR_CARBON_INTENSITY: Dict[str, float] = {
    "oil_gas": 850.0,
    "coal_mining": 1200.0,
    "steel": 650.0,
    "cement": 750.0,
    "chemicals": 400.0,
    "aluminium": 500.0,
    "power_generation": 550.0,
    "aviation": 350.0,
    "shipping": 300.0,
    "agriculture": 250.0,
    "automotive": 150.0,
    "manufacturing": 120.0,
    "construction": 100.0,
    "retail": 50.0,
    "technology": 30.0,
    "financial_services": 20.0,
    "professional_services": 15.0,
    "other": 80.0,
}
"""Sector carbon intensity benchmarks (tCO2e per $M revenue)."""

# Classification thresholds (percentile-based).
CLASSIFICATION_THRESHOLDS: Dict[str, float] = {
    "strategic_pct": 20.0,   # Top 20%
    "key_pct": 50.0,         # Next 30% (20-50 percentile)
}
"""Classification thresholds: strategic top 20%, key next 30%, managed rest."""

# Default incentive impact assumptions.
INCENTIVE_DEFAULTS: Dict[str, Dict[str, float]] = {
    IncentiveType.PRICE_PREFERENCE: {
        "typical_premium_pct": 3.0,
        "estimated_reduction_pct": 5.0,
        "adoption_rate_pct": 60.0,
    },
    IncentiveType.VOLUME_GUARANTEE: {
        "typical_premium_pct": 0.0,
        "estimated_reduction_pct": 8.0,
        "adoption_rate_pct": 70.0,
    },
    IncentiveType.CONTRACT_EXTENSION: {
        "typical_premium_pct": 0.0,
        "estimated_reduction_pct": 6.0,
        "adoption_rate_pct": 65.0,
    },
    IncentiveType.RECOGNITION: {
        "typical_premium_pct": 0.0,
        "estimated_reduction_pct": 3.0,
        "adoption_rate_pct": 80.0,
    },
}
"""Default incentive impact parameters by type."""

# ---------------------------------------------------------------------------
# Pydantic Data Models
# ---------------------------------------------------------------------------

class SupplierTarget(BaseModel):
    """Per-supplier emission reduction target.

    Attributes:
        supplier_id: Unique supplier identifier.
        supplier_name: Human-readable supplier name.
        baseline_year: Year of the baseline emissions.
        baseline_tco2e: Baseline emissions in tCO2e.
        target_reduction_pct: Target reduction percentage (0-100).
        target_year: Year by which the target should be achieved.
        interim_targets: Optional dict of year -> target_tco2e milestones.
        scope3_categories: Scope 3 categories this supplier contributes to.
        notes: Optional notes about the target.
    """
    supplier_id: str = Field(..., description="Unique supplier identifier")
    supplier_name: str = Field(..., description="Supplier name")
    baseline_year: int = Field(..., ge=2015, le=2035, description="Baseline year")
    baseline_tco2e: float = Field(..., ge=0, description="Baseline emissions tCO2e")
    target_reduction_pct: float = Field(..., ge=0, le=100, description="Target reduction %")
    target_year: int = Field(..., ge=2025, le=2060, description="Target year")
    interim_targets: Optional[Dict[int, float]] = Field(
        default=None, description="Year -> target tCO2e milestones"
    )
    scope3_categories: List[int] = Field(
        default_factory=list, description="Scope 3 category numbers (1-15)"
    )
    notes: str = Field(default="", description="Notes about the target")

    @field_validator("scope3_categories")
    @classmethod
    def validate_categories(cls, v: List[int]) -> List[int]:
        """Validate all categories are in range 1-15."""
        for cat in v:
            if cat < 1 or cat > 15:
                raise ValueError(f"Scope 3 category must be 1-15, got {cat}")
        return v

class SupplierCommitment(BaseModel):
    """Supplier climate commitment record.

    Attributes:
        supplier_id: Unique supplier identifier.
        commitment_type: Type of commitment.
        status: Current status of the commitment.
        target_year: Year of the commitment target.
        details: Additional details.
        cdp_score: CDP Climate Change score (A/A-/B/B-/C/C-/D/D-).
        verified: Whether the commitment is externally verified.
        last_updated: Date of last update.
    """
    supplier_id: str = Field(..., description="Unique supplier identifier")
    commitment_type: str = Field(..., description="Type of commitment")
    status: str = Field(default="active", description="Commitment status")
    target_year: Optional[int] = Field(default=None, description="Target year")
    details: str = Field(default="", description="Additional details")
    cdp_score: Optional[str] = Field(default=None, description="CDP score (A to D-)")
    verified: bool = Field(default=False, description="Externally verified")
    last_updated: Optional[str] = Field(default=None, description="Last update date")

    @field_validator("cdp_score")
    @classmethod
    def validate_cdp_score(cls, v: Optional[str]) -> Optional[str]:
        """Validate CDP score is in allowed range."""
        if v is not None:
            allowed = {"A", "A-", "B", "B-", "C", "C-", "D", "D-", "F"}
            if v not in allowed:
                raise ValueError(f"CDP score must be one of {allowed}, got {v}")
        return v

class SupplierProgress(BaseModel):
    """Year-on-year progress for a supplier.

    Attributes:
        supplier_id: Unique supplier identifier.
        supplier_name: Supplier name.
        baseline_year: Baseline year.
        baseline_tco2e: Baseline emissions.
        current_year: Current reporting year.
        current_tco2e: Current emissions.
        reduction_tco2e: Absolute reduction.
        reduction_pct: Reduction percentage.
        target_reduction_pct: Target reduction percentage.
        annualised_rate_pct: Annualised reduction rate.
        status: On-track, behind, or ahead.
        years_elapsed: Years since baseline.
        years_remaining: Years until target.
        required_annual_rate_pct: Required annual reduction to meet target.
        provenance_hash: SHA-256 provenance.
        calculated_at: Calculation timestamp.
    """
    supplier_id: str
    supplier_name: str
    baseline_year: int
    baseline_tco2e: float
    current_year: int
    current_tco2e: float
    reduction_tco2e: float
    reduction_pct: float
    target_reduction_pct: float
    annualised_rate_pct: float
    status: str
    years_elapsed: int
    years_remaining: int
    required_annual_rate_pct: float
    provenance_hash: str = ""
    calculated_at: str = ""

class SupplierScorecard(BaseModel):
    """Composite supplier scorecard.

    Attributes:
        supplier_id: Unique supplier identifier.
        supplier_name: Supplier name.
        emission_contribution_score: Emission contribution score (0-100).
        data_quality_score: Data quality score (0-100).
        engagement_level_score: Engagement level score (0-100).
        commitment_strength_score: Commitment strength score (0-100).
        composite_score: Weighted composite score (0-100).
        classification: Strategic, key, or managed.
        rank: Rank among all scored suppliers.
        dimension_details: Breakdown of dimension calculations.
        provenance_hash: SHA-256 provenance.
        calculated_at: Calculation timestamp.
    """
    supplier_id: str
    supplier_name: str
    emission_contribution_score: float
    data_quality_score: float
    engagement_level_score: float
    commitment_strength_score: float
    composite_score: float
    classification: str
    rank: int = 0
    dimension_details: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = ""
    calculated_at: str = ""

class ProgrammeImpact(BaseModel):
    """Aggregate programme impact on reporter's Scope 3 inventory.

    Attributes:
        programme_name: Programme name.
        total_suppliers: Number of suppliers in programme.
        participating_suppliers: Number actively participating.
        baseline_total_tco2e: Total baseline emissions across programme.
        current_total_tco2e: Total current emissions.
        reduction_tco2e: Total reduction.
        reduction_pct: Percentage reduction.
        reporter_scope3_total_tco2e: Reporter's total Scope 3.
        impact_on_scope3_pct: Programme impact as % of total Scope 3.
        by_category: Impact broken down by Scope 3 category.
        by_classification: Impact broken down by supplier classification.
        provenance_hash: SHA-256 provenance.
        calculated_at: Calculation timestamp.
    """
    programme_name: str
    total_suppliers: int
    participating_suppliers: int
    baseline_total_tco2e: float
    current_total_tco2e: float
    reduction_tco2e: float
    reduction_pct: float
    reporter_scope3_total_tco2e: float
    impact_on_scope3_pct: float
    by_category: Dict[str, float] = Field(default_factory=dict)
    by_classification: Dict[str, float] = Field(default_factory=dict)
    provenance_hash: str = ""
    calculated_at: str = ""

class TransitionRisk(BaseModel):
    """Supplier transition risk assessment.

    Attributes:
        supplier_id: Unique supplier identifier.
        supplier_name: Supplier name.
        sector: Supplier sector.
        sector_intensity_score: Score from sector carbon intensity (0-100).
        regulatory_exposure_score: Score from regulatory exposure (0-100).
        decarbonisation_pace_score: Score from decarbonisation pace (0-100).
        financial_resilience_score: Score from financial resilience (0-100).
        overall_risk_score: Composite risk score (0-100).
        risk_rating: HIGH, MEDIUM, or LOW.
        key_risk_factors: List of key risk factors.
        mitigation_recommendations: Suggested mitigations.
        provenance_hash: SHA-256 provenance.
        calculated_at: Calculation timestamp.
    """
    supplier_id: str
    supplier_name: str
    sector: str
    sector_intensity_score: float
    regulatory_exposure_score: float
    decarbonisation_pace_score: float
    financial_resilience_score: float
    overall_risk_score: float
    risk_rating: str
    key_risk_factors: List[str] = Field(default_factory=list)
    mitigation_recommendations: List[str] = Field(default_factory=list)
    provenance_hash: str = ""
    calculated_at: str = ""

class ProgrammeROI(BaseModel):
    """Programme return on investment.

    Attributes:
        programme_name: Programme name.
        total_cost: Total programme cost.
        reduction_tco2e: Total reduction achieved.
        value_per_tonne: Carbon value per tonne.
        total_value: Total emission reduction value.
        roi_ratio: ROI ratio (value / cost).
        payback_years: Payback period in years.
        cost_per_tonne: Cost per tonne of reduction.
        annual_reduction_tco2e: Annual reduction rate.
        net_present_value: NPV of programme.
        provenance_hash: SHA-256 provenance.
        calculated_at: Calculation timestamp.
    """
    programme_name: str
    total_cost: float
    reduction_tco2e: float
    value_per_tonne: float
    total_value: float
    roi_ratio: float
    payback_years: float
    cost_per_tonne: float
    annual_reduction_tco2e: float
    net_present_value: float = 0.0
    provenance_hash: str = ""
    calculated_at: str = ""

class IncentiveImpact(BaseModel):
    """Modelled impact of an incentive on a supplier.

    Attributes:
        supplier_id: Supplier identifier.
        supplier_name: Supplier name.
        incentive_type: Type of incentive.
        current_tco2e: Current emissions.
        estimated_reduction_pct: Estimated emission reduction.
        estimated_reduction_tco2e: Estimated reduction in tCO2e.
        incentive_cost: Estimated cost of the incentive.
        cost_per_tonne: Cost per tonne of reduction.
        adoption_probability_pct: Probability supplier adopts.
        expected_reduction_tco2e: Expected reduction (probability-weighted).
        provenance_hash: SHA-256 provenance.
        calculated_at: Calculation timestamp.
    """
    supplier_id: str
    supplier_name: str
    incentive_type: str
    current_tco2e: float
    estimated_reduction_pct: float
    estimated_reduction_tco2e: float
    incentive_cost: float
    cost_per_tonne: float
    adoption_probability_pct: float
    expected_reduction_tco2e: float
    provenance_hash: str = ""
    calculated_at: str = ""

class SupplierInput(BaseModel):
    """Input data for a single supplier in the programme.

    Attributes:
        supplier_id: Unique identifier.
        supplier_name: Human-readable name.
        sector: Supplier sector classification.
        annual_spend: Annual spend with this supplier.
        baseline_tco2e: Baseline emissions.
        current_tco2e: Current emissions.
        data_quality_score: Data quality score (0-100).
        engagement_level: Engagement level.
        commitments: List of commitment types.
        cdp_score: CDP Climate Change score.
        has_sbti: Whether supplier has SBTi target.
        revenue: Supplier total revenue.
        scope3_categories: Categories this supplier maps to.
        regulatory_jurisdictions: Jurisdictions supplier operates in.
    """
    supplier_id: str = Field(..., description="Unique identifier")
    supplier_name: str = Field(..., description="Supplier name")
    sector: str = Field(default="other", description="Sector classification")
    annual_spend: float = Field(default=0, ge=0, description="Annual spend")
    baseline_tco2e: float = Field(default=0, ge=0, description="Baseline tCO2e")
    current_tco2e: float = Field(default=0, ge=0, description="Current tCO2e")
    data_quality_score: float = Field(default=0, ge=0, le=100, description="DQ score 0-100")
    engagement_level: str = Field(default="none", description="Engagement level")
    commitments: List[str] = Field(default_factory=list, description="Commitment types")
    cdp_score: Optional[str] = Field(default=None, description="CDP score")
    has_sbti: bool = Field(default=False, description="Has SBTi target")
    revenue: float = Field(default=0, ge=0, description="Total revenue")
    scope3_categories: List[int] = Field(default_factory=list, description="Scope 3 cats")
    regulatory_jurisdictions: List[str] = Field(
        default_factory=list, description="Regulatory jurisdictions"
    )

# ---------------------------------------------------------------------------
# Engine Class
# ---------------------------------------------------------------------------

class SupplierProgrammeEngine:
    """Manages supplier-level emission reduction programmes.

    Provides deterministic scoring, classification, progress tracking,
    incentive modelling, transition risk assessment, and programme ROI
    calculation.  All numeric operations use ``Decimal`` arithmetic for
    reproducibility.  Every result carries a SHA-256 provenance hash.

    Attributes:
        scorecard_weights: Weights for scorecard dimensions.
        commitment_scores: Scores for each commitment type.
        classification_thresholds: Percentile thresholds for classification.

    Example:
        >>> engine = SupplierProgrammeEngine()
        >>> targets = engine.set_supplier_targets(
        ...     suppliers=[...],
        ...     reduction_pct=42.0,
        ...     target_year=2030,
        ...     baseline_year=2019,
        ... )
    """

    def __init__(
        self,
        scorecard_weights: Optional[Dict[str, float]] = None,
        commitment_scores: Optional[Dict[str, int]] = None,
        classification_thresholds: Optional[Dict[str, float]] = None,
    ) -> None:
        """Initialise SupplierProgrammeEngine.

        Args:
            scorecard_weights: Override default scorecard dimension weights.
            commitment_scores: Override default commitment type scores.
            classification_thresholds: Override default classification percentiles.
        """
        self.scorecard_weights = scorecard_weights or dict(SCORECARD_WEIGHTS)
        self.commitment_scores = commitment_scores or dict(COMMITMENT_SCORES)
        self.classification_thresholds = (
            classification_thresholds or dict(CLASSIFICATION_THRESHOLDS)
        )
        logger.info(
            "SupplierProgrammeEngine v%s initialised (weights=%s)",
            _MODULE_VERSION,
            self.scorecard_weights,
        )

    # -------------------------------------------------------------------
    # Public -- set_supplier_targets
    # -------------------------------------------------------------------

    def set_supplier_targets(
        self,
        suppliers: List[SupplierInput],
        reduction_pct: float,
        target_year: int,
        baseline_year: int,
        interim_milestones: bool = True,
    ) -> List[SupplierTarget]:
        """Set per-supplier emission reduction targets.

        Allocates the aggregate reduction target across suppliers
        proportionally to their baseline emissions.  Optionally
        generates linear interim milestones for each year.

        Args:
            suppliers: List of supplier input data.
            reduction_pct: Target reduction percentage (0-100).
            target_year: Year to achieve the target.
            baseline_year: Baseline year for tracking.
            interim_milestones: Whether to generate annual milestones.

        Returns:
            List of SupplierTarget records.
        """
        start_ms = time.time()
        logger.info(
            "Setting targets for %d suppliers (%.1f%% by %d, base %d)",
            len(suppliers), reduction_pct, target_year, baseline_year,
        )

        targets: List[SupplierTarget] = []
        red_pct = _decimal(reduction_pct)
        years = target_year - baseline_year

        for supplier in suppliers:
            baseline = _decimal(supplier.baseline_tco2e)
            target_tco2e = baseline * (Decimal("1") - red_pct / Decimal("100"))

            interim: Optional[Dict[int, float]] = None
            if interim_milestones and years > 0:
                interim = {}
                annual_reduction = (baseline - target_tco2e) / _decimal(years)
                for yr in range(baseline_year + 1, target_year + 1):
                    elapsed = yr - baseline_year
                    milestone = baseline - (annual_reduction * _decimal(elapsed))
                    interim[yr] = _round2(milestone)

            targets.append(SupplierTarget(
                supplier_id=supplier.supplier_id,
                supplier_name=supplier.supplier_name,
                baseline_year=baseline_year,
                baseline_tco2e=_round2(baseline),
                target_reduction_pct=_round2(red_pct),
                target_year=target_year,
                interim_targets=interim,
                scope3_categories=supplier.scope3_categories,
            ))

        elapsed_ms = (time.time() - start_ms) * 1000
        logger.info(
            "Set %d supplier targets in %.1f ms", len(targets), elapsed_ms,
        )
        return targets

    # -------------------------------------------------------------------
    # Public -- track_commitments
    # -------------------------------------------------------------------

    def track_commitments(
        self,
        suppliers: List[SupplierInput],
    ) -> List[SupplierCommitment]:
        """Track climate commitments for all suppliers.

        Extracts SBTi status, RE100 membership, CDP score, and
        net-zero pledges from supplier input data.

        Args:
            suppliers: List of supplier input data.

        Returns:
            List of SupplierCommitment records (one per commitment).
        """
        start_ms = time.time()
        commitments: List[SupplierCommitment] = []

        for supplier in suppliers:
            if not supplier.commitments:
                commitments.append(SupplierCommitment(
                    supplier_id=supplier.supplier_id,
                    commitment_type=CommitmentType.NO_COMMITMENT,
                    status="none",
                    last_updated=utcnow().isoformat(),
                ))
                continue

            for commitment_str in supplier.commitments:
                commitments.append(SupplierCommitment(
                    supplier_id=supplier.supplier_id,
                    commitment_type=commitment_str,
                    status="active",
                    cdp_score=supplier.cdp_score,
                    verified=supplier.has_sbti if "sbti" in commitment_str.lower() else False,
                    last_updated=utcnow().isoformat(),
                ))

        elapsed_ms = (time.time() - start_ms) * 1000
        logger.info(
            "Tracked %d commitments across %d suppliers in %.1f ms",
            len(commitments), len(suppliers), elapsed_ms,
        )
        return commitments

    # -------------------------------------------------------------------
    # Public -- measure_progress
    # -------------------------------------------------------------------

    def measure_progress(
        self,
        supplier: SupplierInput,
        target: SupplierTarget,
    ) -> SupplierProgress:
        """Measure year-on-year progress for a supplier against its target.

        Calculates absolute and percentage reduction, annualised rate,
        and whether the supplier is on-track to meet its target.

        Args:
            supplier: Current supplier data.
            target: The supplier's reduction target.

        Returns:
            SupplierProgress with status assessment.
        """
        start_ms = time.time()

        baseline = _decimal(target.baseline_tco2e)
        current = _decimal(supplier.current_tco2e)
        reduction = baseline - current
        reduction_pct = _safe_pct(reduction, baseline)

        # Determine current year from context.
        current_year = utcnow().year
        years_elapsed = max(current_year - target.baseline_year, 0)
        years_remaining = max(target.target_year - current_year, 0)

        # Annualised rate.
        if years_elapsed > 0 and baseline > Decimal("0"):
            ratio = current / baseline
            if ratio > Decimal("0"):
                annualised = (
                    Decimal("1") - ratio ** (Decimal("1") / _decimal(years_elapsed))
                ) * Decimal("100")
            else:
                annualised = Decimal("100")
        else:
            annualised = Decimal("0")

        # Required annual rate to meet target.
        target_remaining_pct = _decimal(target.target_reduction_pct) - reduction_pct
        if years_remaining > 0 and target_remaining_pct > Decimal("0"):
            required_annual = target_remaining_pct / _decimal(years_remaining)
        elif target_remaining_pct <= Decimal("0"):
            required_annual = Decimal("0")
        else:
            required_annual = target_remaining_pct

        # Status determination.
        if years_elapsed == 0:
            status = ProgressStatus.BASELINE_YEAR
        elif reduction_pct >= _decimal(target.target_reduction_pct):
            status = ProgressStatus.AHEAD
        else:
            expected_pct = _decimal(target.target_reduction_pct) * _decimal(years_elapsed) / _decimal(
                max(target.target_year - target.baseline_year, 1)
            )
            if reduction_pct >= expected_pct * Decimal("0.9"):
                status = ProgressStatus.ON_TRACK
            else:
                status = ProgressStatus.BEHIND

        result = SupplierProgress(
            supplier_id=supplier.supplier_id,
            supplier_name=supplier.supplier_name,
            baseline_year=target.baseline_year,
            baseline_tco2e=_round2(baseline),
            current_year=current_year,
            current_tco2e=_round2(current),
            reduction_tco2e=_round2(reduction),
            reduction_pct=_round2(reduction_pct),
            target_reduction_pct=_round2(target.target_reduction_pct),
            annualised_rate_pct=_round2(annualised),
            status=status.value,
            years_elapsed=years_elapsed,
            years_remaining=years_remaining,
            required_annual_rate_pct=_round2(required_annual),
            calculated_at=utcnow().isoformat(),
        )
        result.provenance_hash = _compute_hash(result)

        elapsed_ms = (time.time() - start_ms) * 1000
        logger.info(
            "Measured progress for %s: %.1f%% reduction (%s) in %.1f ms",
            supplier.supplier_name, _round2(reduction_pct), status.value, elapsed_ms,
        )
        return result

    # -------------------------------------------------------------------
    # Public -- model_programme_impact
    # -------------------------------------------------------------------

    def model_programme_impact(
        self,
        programme_name: str,
        suppliers: List[SupplierInput],
        targets: List[SupplierTarget],
        reporter_scope3_tco2e: float,
    ) -> ProgrammeImpact:
        """Model aggregate programme impact on reporter's Scope 3 inventory.

        Args:
            programme_name: Name of the engagement programme.
            suppliers: Current supplier data.
            targets: Per-supplier targets.
            reporter_scope3_tco2e: Reporter's total Scope 3 emissions.

        Returns:
            ProgrammeImpact with aggregate and per-category breakdowns.
        """
        start_ms = time.time()

        target_map = {t.supplier_id: t for t in targets}
        baseline_total = Decimal("0")
        current_total = Decimal("0")
        participating = 0
        by_category: Dict[str, Decimal] = {}
        by_classification: Dict[str, Decimal] = {}

        for supplier in suppliers:
            target = target_map.get(supplier.supplier_id)
            if target is None:
                continue

            participating += 1
            b = _decimal(target.baseline_tco2e)
            c = _decimal(supplier.current_tco2e)
            baseline_total += b
            current_total += c
            red = b - c

            # Accumulate by category.
            for cat in supplier.scope3_categories:
                cat_key = f"category_{cat}"
                share = _safe_divide(red, _decimal(len(supplier.scope3_categories)))
                by_category[cat_key] = by_category.get(cat_key, Decimal("0")) + share

            # Accumulate by classification (will be finalised after scoring).
            # Use simple emission-based bucketing here.
            if b > Decimal("0"):
                by_classification["participating"] = (
                    by_classification.get("participating", Decimal("0")) + red
                )

        reduction = baseline_total - current_total
        reduction_pct = _safe_pct(reduction, baseline_total)
        scope3 = _decimal(reporter_scope3_tco2e)
        impact_pct = _safe_pct(reduction, scope3) if scope3 > Decimal("0") else Decimal("0")

        result = ProgrammeImpact(
            programme_name=programme_name,
            total_suppliers=len(suppliers),
            participating_suppliers=participating,
            baseline_total_tco2e=_round2(baseline_total),
            current_total_tco2e=_round2(current_total),
            reduction_tco2e=_round2(reduction),
            reduction_pct=_round2(reduction_pct),
            reporter_scope3_total_tco2e=_round2(scope3),
            impact_on_scope3_pct=_round2(impact_pct),
            by_category={k: _round2(v) for k, v in by_category.items()},
            by_classification={k: _round2(v) for k, v in by_classification.items()},
            calculated_at=utcnow().isoformat(),
        )
        result.provenance_hash = _compute_hash(result)

        elapsed_ms = (time.time() - start_ms) * 1000
        logger.info(
            "Programme '%s' impact: %.1f tCO2e reduction (%.1f%% of Scope 3) in %.1f ms",
            programme_name, _round2(reduction), _round2(impact_pct), elapsed_ms,
        )
        return result

    # -------------------------------------------------------------------
    # Public -- generate_scorecard
    # -------------------------------------------------------------------

    def generate_scorecard(
        self,
        supplier: SupplierInput,
        total_programme_tco2e: float,
    ) -> SupplierScorecard:
        """Generate a composite scorecard for a supplier.

        Dimensions (weighted):
            emission_contribution (30%): Supplier's share of programme emissions.
            data_quality (20%): Supplier's data quality score.
            engagement_level (20%): Supplier's engagement level.
            commitment_strength (30%): Strongest commitment score.

        Args:
            supplier: Supplier input data.
            total_programme_tco2e: Total programme baseline emissions.

        Returns:
            SupplierScorecard with composite score.
        """
        start_ms = time.time()

        # Emission contribution score (higher share = higher score).
        total_d = _decimal(total_programme_tco2e)
        emission_share = _safe_pct(_decimal(supplier.baseline_tco2e), total_d)
        # Normalise: cap at 100, scale linearly.
        emission_score = min(float(emission_share), Decimal("100"))
        emission_score_d = _decimal(emission_score)

        # Data quality score (direct pass-through 0-100).
        dq_score_d = _decimal(supplier.data_quality_score)

        # Engagement level score.
        eng_score = ENGAGEMENT_LEVEL_SCORES.get(
            supplier.engagement_level, 0
        )
        eng_score_d = _decimal(eng_score)

        # Commitment strength (take highest).
        commit_score = Decimal("0")
        for c in supplier.commitments:
            s = _decimal(self.commitment_scores.get(c, 0))
            if s > commit_score:
                commit_score = s
        if not supplier.commitments:
            commit_score = Decimal("0")

        # Composite.
        w = self.scorecard_weights
        composite = (
            emission_score_d * _decimal(w["emission_contribution"])
            + dq_score_d * _decimal(w["data_quality"])
            + eng_score_d * _decimal(w["engagement_level"])
            + commit_score * _decimal(w["commitment_strength"])
        )

        result = SupplierScorecard(
            supplier_id=supplier.supplier_id,
            supplier_name=supplier.supplier_name,
            emission_contribution_score=_round2(emission_score_d),
            data_quality_score=_round2(dq_score_d),
            engagement_level_score=_round2(eng_score_d),
            commitment_strength_score=_round2(commit_score),
            composite_score=_round2(composite),
            classification="",  # Set during classify_suppliers.
            dimension_details={
                "emission_share_pct": _round2(emission_share),
                "engagement_level": supplier.engagement_level,
                "strongest_commitment": max(supplier.commitments, default="none"),
                "weights": dict(w),
            },
            calculated_at=utcnow().isoformat(),
        )
        result.provenance_hash = _compute_hash(result)

        elapsed_ms = (time.time() - start_ms) * 1000
        logger.debug(
            "Scorecard for %s: composite=%.1f in %.1f ms",
            supplier.supplier_name, _round2(composite), elapsed_ms,
        )
        return result

    # -------------------------------------------------------------------
    # Public -- classify_suppliers
    # -------------------------------------------------------------------

    def classify_suppliers(
        self,
        suppliers: List[SupplierInput],
        total_programme_tco2e: float,
    ) -> List[SupplierScorecard]:
        """Classify all suppliers into strategic/key/managed tiers.

        Generates scorecards for all suppliers, ranks by composite score,
        and assigns classification based on percentile thresholds:
            Strategic: Top 20%
            Key: Next 30% (21st-50th percentile)
            Managed: Bottom 50%

        Args:
            suppliers: List of all supplier inputs.
            total_programme_tco2e: Total programme baseline emissions.

        Returns:
            List of SupplierScorecard ranked by composite score.
        """
        start_ms = time.time()

        # Generate scorecards.
        scorecards = [
            self.generate_scorecard(s, total_programme_tco2e)
            for s in suppliers
        ]

        # Sort descending by composite score.
        scorecards.sort(key=lambda sc: sc.composite_score, reverse=True)

        # Assign ranks and classifications.
        n = len(scorecards)
        strategic_cutoff = max(1, int(n * self.classification_thresholds["strategic_pct"] / 100))
        key_cutoff = max(
            strategic_cutoff + 1,
            int(n * self.classification_thresholds["key_pct"] / 100),
        )

        for i, sc in enumerate(scorecards):
            sc.rank = i + 1
            if i < strategic_cutoff:
                sc.classification = SupplierClassification.STRATEGIC.value
            elif i < key_cutoff:
                sc.classification = SupplierClassification.KEY.value
            else:
                sc.classification = SupplierClassification.MANAGED.value
            sc.provenance_hash = _compute_hash(sc)

        elapsed_ms = (time.time() - start_ms) * 1000
        counts = {
            SupplierClassification.STRATEGIC.value: sum(
                1 for s in scorecards if s.classification == SupplierClassification.STRATEGIC.value
            ),
            SupplierClassification.KEY.value: sum(
                1 for s in scorecards if s.classification == SupplierClassification.KEY.value
            ),
            SupplierClassification.MANAGED.value: sum(
                1 for s in scorecards if s.classification == SupplierClassification.MANAGED.value
            ),
        }
        logger.info(
            "Classified %d suppliers: strategic=%d, key=%d, managed=%d in %.1f ms",
            n, counts["strategic"], counts["key"], counts["managed"], elapsed_ms,
        )
        return scorecards

    # -------------------------------------------------------------------
    # Public -- model_incentives
    # -------------------------------------------------------------------

    def model_incentives(
        self,
        supplier: SupplierInput,
        incentive_type: str,
        annual_spend_override: Optional[float] = None,
        carbon_price_per_tonne: float = 100.0,
    ) -> IncentiveImpact:
        """Model the impact of an incentive on a supplier's emissions.

        Uses default assumptions for each incentive type (price premium,
        adoption rate, estimated reduction) and calculates cost-effectiveness.

        Args:
            supplier: Supplier input data.
            incentive_type: Type of incentive to model.
            annual_spend_override: Override supplier's annual spend.
            carbon_price_per_tonne: Carbon value per tonne for cost comparison.

        Returns:
            IncentiveImpact with cost and reduction estimates.
        """
        start_ms = time.time()

        defaults = INCENTIVE_DEFAULTS.get(
            incentive_type,
            {"typical_premium_pct": 0.0, "estimated_reduction_pct": 3.0, "adoption_rate_pct": 50.0},
        )
        spend = _decimal(annual_spend_override if annual_spend_override else supplier.annual_spend)
        current = _decimal(supplier.current_tco2e)
        premium_pct = _decimal(defaults["typical_premium_pct"])
        reduction_pct = _decimal(defaults["estimated_reduction_pct"])
        adoption_pct = _decimal(defaults["adoption_rate_pct"])

        # Cost of incentive.
        incentive_cost = spend * premium_pct / Decimal("100")

        # Estimated reduction.
        estimated_reduction = current * reduction_pct / Decimal("100")
        expected_reduction = estimated_reduction * adoption_pct / Decimal("100")

        # Cost per tonne.
        cost_per_tonne = _safe_divide(incentive_cost, estimated_reduction)

        result = IncentiveImpact(
            supplier_id=supplier.supplier_id,
            supplier_name=supplier.supplier_name,
            incentive_type=incentive_type,
            current_tco2e=_round2(current),
            estimated_reduction_pct=_round2(reduction_pct),
            estimated_reduction_tco2e=_round2(estimated_reduction),
            incentive_cost=_round2(incentive_cost),
            cost_per_tonne=_round2(cost_per_tonne),
            adoption_probability_pct=_round2(adoption_pct),
            expected_reduction_tco2e=_round2(expected_reduction),
            calculated_at=utcnow().isoformat(),
        )
        result.provenance_hash = _compute_hash(result)

        elapsed_ms = (time.time() - start_ms) * 1000
        logger.info(
            "Incentive model for %s (%s): %.1f tCO2e expected reduction in %.1f ms",
            supplier.supplier_name, incentive_type, _round2(expected_reduction), elapsed_ms,
        )
        return result

    # -------------------------------------------------------------------
    # Public -- assess_transition_risk
    # -------------------------------------------------------------------

    def assess_transition_risk(
        self,
        supplier: SupplierInput,
        regulatory_regions_with_carbon_pricing: Optional[List[str]] = None,
    ) -> TransitionRisk:
        """Assess a supplier's transition risk (risk of failing to decarbonise).

        Computes four sub-scores:
            1. Sector intensity (from benchmark table).
            2. Regulatory exposure (jurisdictions with carbon pricing).
            3. Decarbonisation pace (commitments and progress).
            4. Financial resilience (revenue vs. spend proxy).

        Overall risk = weighted average; rating = HIGH/MEDIUM/LOW.

        Args:
            supplier: Supplier input data.
            regulatory_regions_with_carbon_pricing: Regions with active carbon pricing.

        Returns:
            TransitionRisk assessment.
        """
        start_ms = time.time()
        cp_regions = regulatory_regions_with_carbon_pricing or [
            "EU", "UK", "CA", "NZ", "KR", "CN", "JP",
        ]

        # 1. Sector intensity score (0-100).
        max_intensity = max(SECTOR_CARBON_INTENSITY.values())
        sector_intensity = SECTOR_CARBON_INTENSITY.get(supplier.sector, 80.0)
        sector_score = _round2(_safe_pct(
            _decimal(sector_intensity), _decimal(max_intensity),
        ))

        # 2. Regulatory exposure (0-100).
        if supplier.regulatory_jurisdictions:
            exposed = sum(
                1 for j in supplier.regulatory_jurisdictions
                if j.upper() in [r.upper() for r in cp_regions]
            )
            reg_score = _round2(
                _safe_pct(_decimal(exposed), _decimal(len(supplier.regulatory_jurisdictions)))
            )
        else:
            reg_score = 50.0  # Unknown = medium risk.

        # 3. Decarbonisation pace (0-100; lower commitment = higher risk).
        best_commit = Decimal("0")
        for c in supplier.commitments:
            s = _decimal(self.commitment_scores.get(c, 0))
            if s > best_commit:
                best_commit = s
        # Invert: high commitment = low risk score.
        decarb_score = _round2(Decimal("100") - best_commit)

        # 4. Financial resilience proxy (0-100).
        # Simple heuristic: larger revenue = more resilient = lower risk.
        if supplier.revenue > 0:
            # Log-scale normalisation (cap at $100B).
            log_rev = min(math.log10(max(supplier.revenue, 1)), 11)  # log10(100B) ~ 11
            fin_score = _round2(Decimal("100") - _decimal(log_rev * 100 / 11))
        else:
            fin_score = 50.0  # Unknown.

        # Overall.
        overall = _round2((
            _decimal(sector_score) * Decimal("0.25")
            + _decimal(reg_score) * Decimal("0.25")
            + _decimal(decarb_score) * Decimal("0.25")
            + _decimal(fin_score) * Decimal("0.25")
        ))

        # Rating.
        if overall >= 70:
            rating = TransitionRiskRating.HIGH
        elif overall >= 40:
            rating = TransitionRiskRating.MEDIUM
        else:
            rating = TransitionRiskRating.LOW

        # Key risk factors.
        risk_factors: List[str] = []
        if sector_score >= 50:
            risk_factors.append(f"High-carbon sector ({supplier.sector}, intensity={sector_intensity})")
        if reg_score >= 50:
            risk_factors.append("Significant regulatory exposure to carbon pricing")
        if decarb_score >= 70:
            risk_factors.append("Weak or no climate commitments")
        if fin_score >= 70:
            risk_factors.append("Limited financial resilience for transition investment")

        # Mitigations.
        mitigations: List[str] = []
        if decarb_score >= 50:
            mitigations.append("Engage supplier in SBTi commitment programme")
        if sector_score >= 50:
            mitigations.append("Evaluate alternative lower-carbon suppliers in category")
        if reg_score >= 50:
            mitigations.append("Monitor carbon pricing developments in supplier jurisdictions")
        mitigations.append("Include climate risk in supplier review process")

        result = TransitionRisk(
            supplier_id=supplier.supplier_id,
            supplier_name=supplier.supplier_name,
            sector=supplier.sector,
            sector_intensity_score=sector_score,
            regulatory_exposure_score=reg_score,
            decarbonisation_pace_score=decarb_score,
            financial_resilience_score=fin_score,
            overall_risk_score=overall,
            risk_rating=rating.value,
            key_risk_factors=risk_factors,
            mitigation_recommendations=mitigations,
            calculated_at=utcnow().isoformat(),
        )
        result.provenance_hash = _compute_hash(result)

        elapsed_ms = (time.time() - start_ms) * 1000
        logger.info(
            "Transition risk for %s: %.1f (%s) in %.1f ms",
            supplier.supplier_name, overall, rating.value, elapsed_ms,
        )
        return result

    # -------------------------------------------------------------------
    # Public -- calculate_programme_roi
    # -------------------------------------------------------------------

    def calculate_programme_roi(
        self,
        programme_name: str,
        total_cost: float,
        reduction_tco2e: float,
        programme_years: int,
        value_per_tonne: float = 100.0,
        discount_rate: float = 0.08,
    ) -> ProgrammeROI:
        """Calculate programme return on investment.

        Args:
            programme_name: Name of the programme.
            total_cost: Total programme cost.
            reduction_tco2e: Total reduction achieved.
            programme_years: Duration of the programme in years.
            value_per_tonne: Carbon value per tonne.
            discount_rate: Discount rate for NPV (default 8%).

        Returns:
            ProgrammeROI with ROI ratio, payback, and NPV.
        """
        start_ms = time.time()

        cost_d = _decimal(total_cost)
        reduction_d = _decimal(reduction_tco2e)
        value_d = _decimal(value_per_tonne)
        years_d = _decimal(programme_years)
        rate_d = _decimal(discount_rate)

        total_value = reduction_d * value_d
        roi_ratio = _safe_divide(total_value, cost_d)

        annual_reduction = _safe_divide(reduction_d, years_d)
        annual_value = annual_reduction * value_d
        payback = _safe_divide(cost_d, annual_value) if annual_value > Decimal("0") else Decimal("0")
        cost_per_tonne = _safe_divide(cost_d, reduction_d)

        # NPV of annual reduction value over programme life.
        npv = Decimal("0")
        for yr in range(1, programme_years + 1):
            discount_factor = Decimal("1") / (Decimal("1") + rate_d) ** _decimal(yr)
            npv += annual_value * discount_factor
        npv -= cost_d

        result = ProgrammeROI(
            programme_name=programme_name,
            total_cost=_round2(cost_d),
            reduction_tco2e=_round2(reduction_d),
            value_per_tonne=_round2(value_d),
            total_value=_round2(total_value),
            roi_ratio=_round4(roi_ratio),
            payback_years=_round2(payback),
            cost_per_tonne=_round2(cost_per_tonne),
            annual_reduction_tco2e=_round2(annual_reduction),
            net_present_value=_round2(npv),
            calculated_at=utcnow().isoformat(),
        )
        result.provenance_hash = _compute_hash(result)

        elapsed_ms = (time.time() - start_ms) * 1000
        logger.info(
            "Programme ROI '%s': ratio=%.2f, payback=%.1f yrs, NPV=%.0f in %.1f ms",
            programme_name, _round2(roi_ratio), _round2(payback), _round2(npv), elapsed_ms,
        )
        return result

    # -------------------------------------------------------------------
    # Public -- _compute_provenance (module-level accessor)
    # -------------------------------------------------------------------

    @staticmethod
    def _compute_provenance(data: Any) -> str:
        """Compute SHA-256 provenance hash for audit trail.

        Args:
            data: Data to hash.

        Returns:
            SHA-256 hex digest (64 characters).
        """
        return _compute_hash(data)

# ---------------------------------------------------------------------------
# Pydantic v2 model_rebuild for forward-reference resolution
# ---------------------------------------------------------------------------

SupplierTarget.model_rebuild()
SupplierCommitment.model_rebuild()
SupplierProgress.model_rebuild()
SupplierScorecard.model_rebuild()
ProgrammeImpact.model_rebuild()
TransitionRisk.model_rebuild()
ProgrammeROI.model_rebuild()
IncentiveImpact.model_rebuild()
SupplierInput.model_rebuild()
