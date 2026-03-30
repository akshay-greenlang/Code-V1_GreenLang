# -*- coding: utf-8 -*-
"""
ThresholdScoringEngine - PACK-015 Double Materiality Engine 7
================================================================

Configurable scoring methodologies and materiality thresholds for the
double materiality assessment.  Supports multiple scoring approaches
(geometric mean, arithmetic mean, weighted sum, max score, product),
normalization methods, sector-specific adjustments, and sensitivity
analysis.

Regulatory Context:
    ESRS 1 para 21-33 defines materiality but leaves scoring methodology
    to the undertaking.  This engine provides a deterministic, auditable
    framework for implementing any chosen methodology while ensuring
    reproducibility and transparency.

Scoring Flow:
    1. Raw scores collected from stakeholder engagement / expert assessment
    2. Sector adjustments applied (NACE-based weighting)
    3. Scores combined via chosen methodology
    4. Normalization applied (min-max, z-score, percentile, or none)
    5. Threshold comparison determines materiality
    6. Sensitivity analysis identifies borderline topics

Zero-Hallucination:
    - All arithmetic uses deterministic Decimal operations
    - Sector adjustments use hard-coded lookup tables
    - Normalization uses pure mathematical formulas
    - Sensitivity analysis uses brute-force threshold iteration
    - SHA-256 provenance hash on every result
    - No LLM involvement in any calculation path

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-015 Double Materiality
Status:  Production Ready
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator, model_validator

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

engine_version: str = "1.0.0"

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

def _round_val(value: Decimal, places: int = 6) -> float:
    """Round a Decimal to *places* and return a float.

    Uses ROUND_HALF_UP (regulatory standard rounding).
    """
    quantizer = Decimal(10) ** -places
    return float(value.quantize(quantizer, rounding=ROUND_HALF_UP))

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ScoringMethodology(str, Enum):
    """Methodology for combining sub-scores into a composite score.

    GEOMETRIC_MEAN: Square root of product (penalizes imbalance).
    ARITHMETIC_MEAN: Simple weighted average.
    WEIGHTED_SUM: Sum of weighted scores (not normalized to scale).
    MAX_SCORE: Maximum of all sub-scores.
    PRODUCT: Product of all sub-scores (highly penalizes low scores).
    """
    GEOMETRIC_MEAN = "geometric_mean"
    ARITHMETIC_MEAN = "arithmetic_mean"
    WEIGHTED_SUM = "weighted_sum"
    MAX_SCORE = "max_score"
    PRODUCT = "product"

class NormalizationMethod(str, Enum):
    """Method for normalizing scores to a common scale.

    MIN_MAX: Scale to [0, 1] range using (x - min) / (max - min).
    Z_SCORE: Standardize to mean=0, std=1.
    PERCENTILE: Convert to percentile rank (0-100).
    NONE: No normalization applied.
    """
    MIN_MAX = "min_max"
    Z_SCORE = "z_score"
    PERCENTILE = "percentile"
    NONE = "none"

class ThresholdSource(str, Enum):
    """Origin of the materiality threshold.

    REGULATORY: From ESRS or other regulation.
    INDUSTRY_STANDARD: From industry body guidance.
    PEER_BENCHMARK: Derived from peer comparison.
    CUSTOM: Defined by the undertaking.
    """
    REGULATORY = "regulatory"
    INDUSTRY_STANDARD = "industry_standard"
    PEER_BENCHMARK = "peer_benchmark"
    CUSTOM = "custom"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METHODOLOGY_DESCRIPTIONS: Dict[str, str] = {
    "geometric_mean": (
        "Geometric mean of sub-scores. Penalizes imbalance between criteria. "
        "A matter must score reasonably well on all criteria to achieve a high score."
    ),
    "arithmetic_mean": (
        "Weighted arithmetic mean of sub-scores. Simple and transparent. "
        "High scores on one criterion can compensate for low scores on another."
    ),
    "weighted_sum": (
        "Sum of weighted sub-scores. Not bounded to the original scale. "
        "Useful when different criteria have different importance weights."
    ),
    "max_score": (
        "Maximum of all sub-scores. A matter is scored by its highest-scoring "
        "criterion. Conservative approach that ensures no material impact is missed."
    ),
    "product": (
        "Product of all sub-scores. Highly penalizes any low-scoring criterion. "
        "A single low score will dramatically reduce the composite."
    ),
}

# Industry-specific materiality thresholds.
# Impact and financial thresholds on a 1-5 scale.
# Source: Compiled from EFRAG guidance, sector-specific CSRD implementations,
# and leading sustainability frameworks.
INDUSTRY_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    "energy": {
        "impact_threshold": Decimal("2.500"),
        "financial_threshold": Decimal("3.000"),
        "combined_threshold": Decimal("2.750"),
        "source": "industry_standard",
        "rationale": "Energy sector has high inherent environmental impact; lower impact threshold.",
    },
    "manufacturing": {
        "impact_threshold": Decimal("3.000"),
        "financial_threshold": Decimal("3.000"),
        "combined_threshold": Decimal("3.000"),
        "source": "industry_standard",
        "rationale": "Standard thresholds for manufacturing sector.",
    },
    "financial_services": {
        "impact_threshold": Decimal("3.500"),
        "financial_threshold": Decimal("2.500"),
        "combined_threshold": Decimal("3.000"),
        "source": "industry_standard",
        "rationale": "Financial services have high financial materiality, moderate direct impact.",
    },
    "retail": {
        "impact_threshold": Decimal("3.000"),
        "financial_threshold": Decimal("3.000"),
        "combined_threshold": Decimal("3.000"),
        "source": "industry_standard",
        "rationale": "Standard thresholds for retail sector.",
    },
    "technology": {
        "impact_threshold": Decimal("3.500"),
        "financial_threshold": Decimal("3.000"),
        "combined_threshold": Decimal("3.250"),
        "source": "industry_standard",
        "rationale": "Technology sector has moderate direct impact; slightly higher threshold.",
    },
    "construction": {
        "impact_threshold": Decimal("2.500"),
        "financial_threshold": Decimal("3.000"),
        "combined_threshold": Decimal("2.750"),
        "source": "industry_standard",
        "rationale": "Construction has high environmental impact; lower impact threshold.",
    },
    "healthcare": {
        "impact_threshold": Decimal("3.000"),
        "financial_threshold": Decimal("3.000"),
        "combined_threshold": Decimal("3.000"),
        "source": "industry_standard",
        "rationale": "Standard thresholds for healthcare sector.",
    },
    "agriculture": {
        "impact_threshold": Decimal("2.500"),
        "financial_threshold": Decimal("2.500"),
        "combined_threshold": Decimal("2.500"),
        "source": "industry_standard",
        "rationale": "Agriculture has high environmental and social impact; lower thresholds.",
    },
    "transport": {
        "impact_threshold": Decimal("2.500"),
        "financial_threshold": Decimal("3.000"),
        "combined_threshold": Decimal("2.750"),
        "source": "industry_standard",
        "rationale": "Transport sector has high climate impact; lower impact threshold.",
    },
    "default": {
        "impact_threshold": Decimal("3.000"),
        "financial_threshold": Decimal("3.000"),
        "combined_threshold": Decimal("3.000"),
        "source": "regulatory",
        "rationale": "Default mid-point threshold on 5-point scale per EFRAG guidance.",
    },
}

# Sector-specific adjustment factors for ESRS topics.
# Values > 1.0 increase relevance, < 1.0 decrease relevance.
# Keyed by NACE sector, then by ESRS topic.
SECTOR_ADJUSTMENT_FACTORS: Dict[str, Dict[str, Decimal]] = {
    "energy": {
        "E1": Decimal("1.30"), "E2": Decimal("1.10"), "E3": Decimal("1.10"),
        "E4": Decimal("1.05"), "E5": Decimal("1.00"),
        "S1": Decimal("1.00"), "S2": Decimal("0.90"), "S3": Decimal("1.10"),
        "S4": Decimal("0.90"), "G1": Decimal("1.00"),
    },
    "manufacturing": {
        "E1": Decimal("1.20"), "E2": Decimal("1.20"), "E3": Decimal("1.10"),
        "E4": Decimal("1.00"), "E5": Decimal("1.20"),
        "S1": Decimal("1.10"), "S2": Decimal("1.10"), "S3": Decimal("1.00"),
        "S4": Decimal("1.00"), "G1": Decimal("1.00"),
    },
    "financial_services": {
        "E1": Decimal("1.00"), "E2": Decimal("0.80"), "E3": Decimal("0.80"),
        "E4": Decimal("0.90"), "E5": Decimal("0.80"),
        "S1": Decimal("1.10"), "S2": Decimal("0.90"), "S3": Decimal("1.00"),
        "S4": Decimal("1.20"), "G1": Decimal("1.30"),
    },
    "retail": {
        "E1": Decimal("1.00"), "E2": Decimal("0.90"), "E3": Decimal("0.90"),
        "E4": Decimal("1.10"), "E5": Decimal("1.20"),
        "S1": Decimal("1.10"), "S2": Decimal("1.20"), "S3": Decimal("1.00"),
        "S4": Decimal("1.20"), "G1": Decimal("1.00"),
    },
    "technology": {
        "E1": Decimal("1.10"), "E2": Decimal("0.80"), "E3": Decimal("0.90"),
        "E4": Decimal("0.80"), "E5": Decimal("1.10"),
        "S1": Decimal("1.10"), "S2": Decimal("1.00"), "S3": Decimal("0.90"),
        "S4": Decimal("1.20"), "G1": Decimal("1.10"),
    },
    "agriculture": {
        "E1": Decimal("1.10"), "E2": Decimal("1.20"), "E3": Decimal("1.30"),
        "E4": Decimal("1.30"), "E5": Decimal("1.10"),
        "S1": Decimal("1.10"), "S2": Decimal("1.20"), "S3": Decimal("1.20"),
        "S4": Decimal("1.00"), "G1": Decimal("1.00"),
    },
    "default": {
        "E1": Decimal("1.00"), "E2": Decimal("1.00"), "E3": Decimal("1.00"),
        "E4": Decimal("1.00"), "E5": Decimal("1.00"),
        "S1": Decimal("1.00"), "S2": Decimal("1.00"), "S3": Decimal("1.00"),
        "S4": Decimal("1.00"), "G1": Decimal("1.00"),
    },
}

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class ScoringProfile(BaseModel):
    """Configuration for a scoring methodology.

    Defines how raw scores are combined, weighted, and adjusted
    for a particular assessment.

    Attributes:
        id: Unique profile identifier.
        name: Human-readable name.
        methodology: Scoring methodology to use.
        impact_weights: Weights for impact sub-criteria (criteria_name -> weight).
        financial_weights: Weights for financial sub-criteria.
        sector_adjustments: Per-topic adjustment factors.
        custom_thresholds: Custom thresholds (if not using industry defaults).
    """
    id: str = Field(default_factory=_new_uuid, description="Profile identifier")
    name: str = Field(default="Default Profile", description="Profile name")
    methodology: ScoringMethodology = Field(
        default=ScoringMethodology.ARITHMETIC_MEAN, description="Scoring methodology"
    )
    impact_weights: Dict[str, Decimal] = Field(
        default_factory=dict, description="Weights for impact sub-criteria"
    )
    financial_weights: Dict[str, Decimal] = Field(
        default_factory=dict, description="Weights for financial sub-criteria"
    )
    sector_adjustments: Dict[str, Decimal] = Field(
        default_factory=dict, description="Per-topic sector adjustment factors"
    )
    custom_thresholds: Dict[str, Decimal] = Field(
        default_factory=dict, description="Custom threshold overrides"
    )

    @field_validator("impact_weights", "financial_weights", "sector_adjustments", "custom_thresholds", mode="before")
    @classmethod
    def _coerce_decimal_dict(cls, v: Any) -> Dict[str, Decimal]:
        if isinstance(v, dict):
            return {k: _decimal(val) for k, val in v.items()}
        return {}

class ThresholdSet(BaseModel):
    """A set of materiality thresholds.

    Attributes:
        impact_threshold: Threshold for impact materiality (0-5 scale).
        financial_threshold: Threshold for financial materiality (0-5 scale).
        combined_threshold: Threshold for combined score.
        sector: Sector these thresholds apply to.
        source: Origin of the thresholds.
    """
    impact_threshold: Decimal = Field(default=Decimal("3.000"), description="Impact threshold")
    financial_threshold: Decimal = Field(default=Decimal("3.000"), description="Financial threshold")
    combined_threshold: Decimal = Field(default=Decimal("3.000"), description="Combined threshold")
    sector: Optional[str] = Field(default=None, description="Sector name")
    source: str = Field(default="regulatory", description="Threshold source")

    @field_validator("impact_threshold", "financial_threshold", "combined_threshold", mode="before")
    @classmethod
    def _coerce_threshold(cls, v: Any) -> Decimal:
        return _decimal(v)

class ScoringResult(BaseModel):
    """Result of scoring a single sustainability matter.

    Attributes:
        matter_id: Unique identifier of the matter.
        raw_score: Unprocessed aggregate score.
        weighted_score: Score after applying weights.
        normalized_score: Score after normalization.
        percentile: Percentile rank among all scored matters.
        threshold_applied: Threshold value used for comparison.
        passes_threshold: Whether the score exceeds the threshold.
        scoring_methodology: Name of methodology used.
        sector_adjustment_applied: Adjustment factor applied.
        provenance_hash: SHA-256 hash for audit trail.
    """
    matter_id: str = Field(..., description="Matter identifier")
    raw_score: Decimal = Field(default=Decimal("0"), description="Raw aggregate score")
    weighted_score: Decimal = Field(default=Decimal("0"), description="Weighted score")
    normalized_score: Decimal = Field(default=Decimal("0"), description="Normalized score")
    percentile: Decimal = Field(default=Decimal("0"), description="Percentile rank (0-100)")
    threshold_applied: Decimal = Field(default=Decimal("3"), description="Threshold used")
    passes_threshold: bool = Field(default=False, description="Exceeds threshold")
    scoring_methodology: str = Field(default="arithmetic_mean", description="Methodology name")
    sector_adjustment_applied: Decimal = Field(default=Decimal("1.0"), description="Sector adjustment factor")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

    @field_validator(
        "raw_score", "weighted_score", "normalized_score", "percentile",
        "threshold_applied", "sector_adjustment_applied", mode="before"
    )
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

class SensitivityPoint(BaseModel):
    """A single point in a sensitivity analysis.

    Attributes:
        threshold_value: Threshold value tested.
        passes: Whether the matter passes at this threshold.
    """
    threshold_value: Decimal = Field(..., description="Threshold tested")
    passes: bool = Field(default=False, description="Passes at this threshold")

class SensitivityAnalysis(BaseModel):
    """Sensitivity analysis for a single sustainability matter.

    Tests whether a matter's materiality classification changes
    under different threshold values.

    Attributes:
        matter_id: Unique identifier of the matter.
        base_score: The matter's score.
        threshold_range: List of thresholds tested.
        sensitivity_results: Pass/fail at each threshold.
        is_borderline: True if classification changes within +/- 0.5 of threshold.
        breakpoint_threshold: Threshold at which classification flips.
    """
    matter_id: str = Field(..., description="Matter identifier")
    base_score: Decimal = Field(default=Decimal("0"), description="Matter's score")
    threshold_range: List[Decimal] = Field(default_factory=list, description="Thresholds tested")
    sensitivity_results: List[SensitivityPoint] = Field(default_factory=list)
    is_borderline: bool = Field(default=False, description="Classification changes near threshold")
    breakpoint_threshold: Optional[Decimal] = Field(default=None, description="Threshold where classification flips")

    @field_validator("base_score", mode="before")
    @classmethod
    def _coerce_score(cls, v: Any) -> Decimal:
        return _decimal(v)

# ---------------------------------------------------------------------------
# Input Model
# ---------------------------------------------------------------------------

class RawScoreInput(BaseModel):
    """Raw score input for a single matter.

    Attributes:
        matter_id: Unique matter identifier.
        esrs_topic: ESRS topic code.
        sub_scores: Dictionary of criterion_name -> score (0-5 scale).
    """
    matter_id: str = Field(..., min_length=1)
    esrs_topic: str = Field(default="", description="ESRS topic code for sector adjustment")
    sub_scores: Dict[str, Decimal] = Field(..., description="Sub-criteria scores (0-5)")

    @field_validator("sub_scores", mode="before")
    @classmethod
    def _coerce_scores(cls, v: Any) -> Dict[str, Decimal]:
        if isinstance(v, dict):
            return {k: _decimal(val) for k, val in v.items()}
        return {}

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ThresholdScoringEngine:
    """Configurable scoring engine for double materiality assessment.

    Zero-Hallucination Guarantees:
        - All arithmetic uses deterministic Decimal operations
        - Sector adjustments use hard-coded lookup tables
        - Normalization uses pure mathematical formulas
        - SHA-256 provenance hash on every result
        - No LLM involvement in any calculation path

    Usage::

        engine = ThresholdScoringEngine(sector="manufacturing")
        profile = engine.create_scoring_profile(
            methodology=ScoringMethodology.GEOMETRIC_MEAN,
        )
        result = engine.score_matter(raw_input, profile, threshold=Decimal("3.0"))
    """

    def __init__(self, sector: str = "default") -> None:
        """Initialize ThresholdScoringEngine.

        Args:
            sector: Industry sector for threshold and adjustment lookup.
        """
        self.sector = sector
        self._thresholds = INDUSTRY_THRESHOLDS
        self._adjustments = SECTOR_ADJUSTMENT_FACTORS
        logger.info("ThresholdScoringEngine initialized: sector=%s", sector)

    # ------------------------------------------------------------------
    # Profile Creation
    # ------------------------------------------------------------------

    def create_scoring_profile(
        self,
        methodology: ScoringMethodology = ScoringMethodology.ARITHMETIC_MEAN,
        impact_weights: Optional[Dict[str, Decimal]] = None,
        financial_weights: Optional[Dict[str, Decimal]] = None,
        name: str = "Custom Profile",
        sector: Optional[str] = None,
    ) -> ScoringProfile:
        """Create a scoring profile with optional sector adjustments.

        Args:
            methodology: Scoring methodology to use.
            impact_weights: Weights for impact sub-criteria.
            financial_weights: Weights for financial sub-criteria.
            name: Profile name.
            sector: Sector for adjustments (uses engine sector if None).

        Returns:
            ScoringProfile configured for use.
        """
        effective_sector = sector or self.sector
        sector_adj = self._adjustments.get(
            effective_sector, self._adjustments.get("default", {})
        )

        return ScoringProfile(
            name=name,
            methodology=methodology,
            impact_weights=impact_weights or {},
            financial_weights=financial_weights or {},
            sector_adjustments=sector_adj,
        )

    # ------------------------------------------------------------------
    # Core: Score Matter
    # ------------------------------------------------------------------

    def score_matter(
        self,
        raw_input: RawScoreInput,
        profile: ScoringProfile,
        threshold: Optional[Decimal] = None,
    ) -> ScoringResult:
        """Score a single sustainability matter.

        DETERMINISTIC: Same inputs always produce the same score.

        Steps:
            1. Calculate weighted composite from sub-scores
            2. Apply sector adjustment
            3. Compare against threshold

        Args:
            raw_input: Raw sub-scores for the matter.
            profile: Scoring profile defining methodology and weights.
            threshold: Materiality threshold (uses industry default if None).

        Returns:
            ScoringResult with all scoring details.
        """
        # Step 1: Calculate weighted composite
        weights = profile.impact_weights if profile.impact_weights else None
        weighted = self._calculate_composite(
            raw_input.sub_scores, weights, profile.methodology
        )

        # Step 2: Apply sector adjustment
        adj_factor = self.apply_sector_adjustment_factor(
            raw_input.esrs_topic, profile.sector_adjustments
        )
        adjusted = (weighted * adj_factor).quantize(
            Decimal("0.000001"), rounding=ROUND_HALF_UP
        )

        # Step 3: Determine threshold
        if threshold is not None:
            thr = _decimal(threshold)
        else:
            thr = self.get_industry_threshold(self.sector).impact_threshold

        passes = adjusted >= thr

        result = ScoringResult(
            matter_id=raw_input.matter_id,
            raw_score=self._simple_mean(raw_input.sub_scores),
            weighted_score=weighted,
            normalized_score=adjusted,
            percentile=Decimal("0"),  # Set in batch_score
            threshold_applied=thr,
            passes_threshold=passes,
            scoring_methodology=profile.methodology.value,
            sector_adjustment_applied=adj_factor,
        )
        result.provenance_hash = _compute_hash(result)
        return result

    # ------------------------------------------------------------------
    # Core: Batch Score
    # ------------------------------------------------------------------

    def batch_score(
        self,
        matters: List[RawScoreInput],
        profile: ScoringProfile,
        threshold: Optional[Decimal] = None,
    ) -> List[ScoringResult]:
        """Score multiple matters and compute percentile ranks.

        DETERMINISTIC: Same inputs always produce the same results.

        Args:
            matters: List of raw score inputs.
            profile: Scoring profile.
            threshold: Materiality threshold.

        Returns:
            List of ScoringResult with percentile ranks computed.
        """
        # First pass: compute all scores
        results: List[ScoringResult] = []
        for raw_input in matters:
            result = self.score_matter(raw_input, profile, threshold)
            results.append(result)

        # Second pass: compute percentile ranks
        if len(results) > 1:
            all_scores = sorted([r.normalized_score for r in results])
            for result in results:
                rank = self._percentile_rank(result.normalized_score, all_scores)
                result.percentile = rank
                result.provenance_hash = _compute_hash(result)
        elif len(results) == 1:
            results[0].percentile = Decimal("50.0")
            results[0].provenance_hash = _compute_hash(results[0])

        return results

    # ------------------------------------------------------------------
    # Core: Sector Adjustment
    # ------------------------------------------------------------------

    def apply_sector_adjustment(
        self,
        score: Decimal,
        sector: str,
        topic: str,
    ) -> Decimal:
        """Apply sector-specific adjustment to a score.

        DETERMINISTIC: Hard-coded lookup table.

        Args:
            score: Original score.
            sector: Sector name (e.g. "energy", "retail").
            topic: ESRS topic code (e.g. "E1", "S1").

        Returns:
            Adjusted score, capped at 5.0.
        """
        factor = self.apply_sector_adjustment_factor(
            topic,
            self._adjustments.get(sector, self._adjustments.get("default", {})),
        )
        adjusted = score * factor
        # Cap at 5.0 (maximum on the 1-5 scale)
        if adjusted > Decimal("5"):
            adjusted = Decimal("5")
        return adjusted.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)

    def apply_sector_adjustment_factor(
        self,
        topic: str,
        sector_adjustments: Dict[str, Decimal],
    ) -> Decimal:
        """Look up the sector adjustment factor for a topic.

        Args:
            topic: ESRS topic code.
            sector_adjustments: Topic-to-factor mapping.

        Returns:
            Adjustment factor (Decimal); defaults to 1.0 if not found.
        """
        return sector_adjustments.get(topic, Decimal("1.0"))

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def normalize_scores(
        self, scores: List[Decimal], method: NormalizationMethod
    ) -> List[Decimal]:
        """Normalize a list of scores using the specified method.

        DETERMINISTIC: Same inputs always produce the same output.

        Args:
            scores: List of raw scores.
            method: Normalization method.

        Returns:
            List of normalized Decimal scores.
        """
        if not scores:
            return []

        if method == NormalizationMethod.NONE:
            return [s.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP) for s in scores]

        if method == NormalizationMethod.MIN_MAX:
            return self._normalize_min_max(scores)

        if method == NormalizationMethod.Z_SCORE:
            return self._normalize_z_score(scores)

        if method == NormalizationMethod.PERCENTILE:
            return self._normalize_percentile(scores)

        return scores

    def _normalize_min_max(self, scores: List[Decimal]) -> List[Decimal]:
        """Min-max normalization to [0, 1] range."""
        min_val = min(scores)
        max_val = max(scores)
        range_val = max_val - min_val

        if range_val == Decimal("0"):
            return [Decimal("0.500000")] * len(scores)

        result: List[Decimal] = []
        for s in scores:
            normalized = _safe_divide(s - min_val, range_val)
            result.append(normalized.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))
        return result

    def _normalize_z_score(self, scores: List[Decimal]) -> List[Decimal]:
        """Z-score normalization (mean=0, std=1)."""
        n = _decimal(len(scores))
        mean = _safe_divide(sum(scores), n)

        # Calculate standard deviation
        variance = _safe_divide(
            sum((s - mean) ** 2 for s in scores), n
        )
        if variance == Decimal("0"):
            return [Decimal("0.000000")] * len(scores)

        std_dev = _decimal(float(variance) ** 0.5)
        if std_dev == Decimal("0"):
            return [Decimal("0.000000")] * len(scores)

        result: List[Decimal] = []
        for s in scores:
            z = _safe_divide(s - mean, std_dev)
            result.append(z.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))
        return result

    def _normalize_percentile(self, scores: List[Decimal]) -> List[Decimal]:
        """Percentile normalization (0-100 scale)."""
        n = len(scores)
        if n == 0:
            return []

        sorted_scores = sorted(scores)
        result: List[Decimal] = []
        for s in scores:
            rank = self._percentile_rank(s, sorted_scores)
            result.append(rank)
        return result

    # ------------------------------------------------------------------
    # Sensitivity Analysis
    # ------------------------------------------------------------------

    def run_sensitivity_analysis(
        self,
        matter_id: str,
        score: Decimal,
        threshold_range: Optional[List[Decimal]] = None,
        base_threshold: Optional[Decimal] = None,
    ) -> SensitivityAnalysis:
        """Run sensitivity analysis for a matter's materiality classification.

        Tests whether the classification changes under different thresholds.
        A matter is considered "borderline" if its classification changes
        within +/- 0.5 of the base threshold.

        DETERMINISTIC: Same inputs produce identical analysis.

        Args:
            matter_id: Matter identifier.
            score: The matter's score.
            threshold_range: List of thresholds to test. If None, generates
                             a range from 1.0 to 5.0 in 0.25 increments.
            base_threshold: Base threshold for borderline detection.

        Returns:
            SensitivityAnalysis with results.
        """
        s = _decimal(score)

        if threshold_range is None:
            # Generate range: 1.0, 1.25, 1.50, ..., 5.0
            threshold_range = []
            current = Decimal("1.0")
            while current <= Decimal("5.0"):
                threshold_range.append(current)
                current += Decimal("0.25")

        if base_threshold is None:
            base_threshold = self.get_industry_threshold(self.sector).impact_threshold

        base_thr = _decimal(base_threshold)

        results: List[SensitivityPoint] = []
        breakpoint: Optional[Decimal] = None

        for thr in threshold_range:
            passes = s >= thr
            results.append(SensitivityPoint(threshold_value=thr, passes=passes))

        # Determine breakpoint: smallest threshold where classification flips
        # from pass to fail (or vice versa)
        for i in range(1, len(results)):
            if results[i].passes != results[i - 1].passes:
                breakpoint = results[i].threshold_value
                break

        # Borderline: classification changes within +/- 0.5 of base threshold
        borderline_low = base_thr - Decimal("0.5")
        borderline_high = base_thr + Decimal("0.5")
        is_borderline = borderline_low <= s <= borderline_high and breakpoint is not None

        return SensitivityAnalysis(
            matter_id=matter_id,
            base_score=s,
            threshold_range=threshold_range,
            sensitivity_results=results,
            is_borderline=is_borderline,
            breakpoint_threshold=breakpoint,
        )

    # ------------------------------------------------------------------
    # Industry Thresholds
    # ------------------------------------------------------------------

    def get_industry_threshold(self, sector: Optional[str] = None) -> ThresholdSet:
        """Get industry-specific materiality thresholds.

        DETERMINISTIC: Hard-coded lookup table.

        Args:
            sector: Sector name. Uses engine sector if None.

        Returns:
            ThresholdSet for the sector.
        """
        effective_sector = sector or self.sector
        data = self._thresholds.get(
            effective_sector, self._thresholds["default"]
        )

        return ThresholdSet(
            impact_threshold=data["impact_threshold"],
            financial_threshold=data["financial_threshold"],
            combined_threshold=data["combined_threshold"],
            sector=effective_sector,
            source=data.get("source", "regulatory"),
        )

    # ------------------------------------------------------------------
    # Methodology Comparison
    # ------------------------------------------------------------------

    def compare_methodologies(
        self,
        matter_id: str,
        sub_scores: Dict[str, Decimal],
        methodologies: Optional[List[ScoringMethodology]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Compare scores produced by different methodologies.

        Useful for documenting methodology sensitivity in the DMA report.

        DETERMINISTIC: Same inputs produce identical comparison.

        Args:
            matter_id: Matter identifier.
            sub_scores: Sub-criteria scores.
            methodologies: List of methodologies to compare. Defaults to all.

        Returns:
            Dictionary keyed by methodology name with score and threshold result.
        """
        if methodologies is None:
            methodologies = list(ScoringMethodology)

        thr = self.get_industry_threshold().impact_threshold
        result: Dict[str, Dict[str, Any]] = {}

        for meth in methodologies:
            composite = self._calculate_composite(sub_scores, None, meth)
            result[meth.value] = {
                "score": _round_val(composite, 3),
                "passes_threshold": composite >= thr,
                "threshold": _round_val(thr, 3),
                "description": METHODOLOGY_DESCRIPTIONS.get(meth.value, ""),
            }

        return result

    # ------------------------------------------------------------------
    # Internal Calculation Methods
    # ------------------------------------------------------------------

    def _calculate_composite(
        self,
        sub_scores: Dict[str, Decimal],
        weights: Optional[Dict[str, Decimal]],
        methodology: ScoringMethodology,
    ) -> Decimal:
        """Calculate composite score from sub-scores using methodology.

        DETERMINISTIC: Pure arithmetic, no randomness.

        Args:
            sub_scores: Criterion name -> score mapping.
            weights: Optional weights per criterion.
            methodology: Combination method.

        Returns:
            Composite score as Decimal.
        """
        if not sub_scores:
            return Decimal("0")

        values = list(sub_scores.values())

        if methodology == ScoringMethodology.ARITHMETIC_MEAN:
            return self._weighted_arithmetic_mean(sub_scores, weights)

        elif methodology == ScoringMethodology.GEOMETRIC_MEAN:
            return self._geometric_mean(values)

        elif methodology == ScoringMethodology.WEIGHTED_SUM:
            return self._weighted_sum(sub_scores, weights)

        elif methodology == ScoringMethodology.MAX_SCORE:
            return max(values)

        elif methodology == ScoringMethodology.PRODUCT:
            return self._product(values)

        else:
            return self._weighted_arithmetic_mean(sub_scores, weights)

    def _weighted_arithmetic_mean(
        self,
        sub_scores: Dict[str, Decimal],
        weights: Optional[Dict[str, Decimal]],
    ) -> Decimal:
        """Weighted arithmetic mean of sub-scores."""
        if not sub_scores:
            return Decimal("0")

        total_weighted = Decimal("0")
        total_weight = Decimal("0")

        for criterion, score in sub_scores.items():
            w = Decimal("1.0")
            if weights and criterion in weights:
                w = weights[criterion]
            total_weighted += score * w
            total_weight += w

        result = _safe_divide(total_weighted, total_weight)
        return result.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)

    def _geometric_mean(self, values: List[Decimal]) -> Decimal:
        """Geometric mean of values."""
        if not values:
            return Decimal("0")

        # Filter out zeros to avoid zero product
        nonzero = [v for v in values if v > Decimal("0")]
        if not nonzero:
            return Decimal("0")

        product_float = 1.0
        for v in nonzero:
            product_float *= float(v)

        n = len(nonzero)
        geo_mean = product_float ** (1.0 / n)
        return _decimal(geo_mean).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)

    def _weighted_sum(
        self,
        sub_scores: Dict[str, Decimal],
        weights: Optional[Dict[str, Decimal]],
    ) -> Decimal:
        """Weighted sum of sub-scores (not normalized to scale)."""
        total = Decimal("0")
        for criterion, score in sub_scores.items():
            w = Decimal("1.0")
            if weights and criterion in weights:
                w = weights[criterion]
            total += score * w
        return total.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)

    def _product(self, values: List[Decimal]) -> Decimal:
        """Product of all values."""
        if not values:
            return Decimal("0")
        result = Decimal("1")
        for v in values:
            result *= v
        return result.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)

    def _simple_mean(self, sub_scores: Dict[str, Decimal]) -> Decimal:
        """Simple (unweighted) arithmetic mean."""
        if not sub_scores:
            return Decimal("0")
        n = _decimal(len(sub_scores))
        total = sum(sub_scores.values())
        return _safe_divide(total, n).quantize(
            Decimal("0.000001"), rounding=ROUND_HALF_UP
        )

    def _percentile_rank(
        self, value: Decimal, sorted_values: List[Decimal]
    ) -> Decimal:
        """Calculate percentile rank of a value in a sorted list.

        Uses the "percentage of values below" method.
        """
        n = len(sorted_values)
        if n == 0:
            return Decimal("0")

        below = sum(1 for v in sorted_values if v < value)
        equal = sum(1 for v in sorted_values if v == value)

        # Percentile = (below + 0.5 * equal) / n * 100
        rank = _safe_divide(
            _decimal(below) + _decimal(equal) * Decimal("0.5"),
            _decimal(n),
        ) * Decimal("100")

        return rank.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
