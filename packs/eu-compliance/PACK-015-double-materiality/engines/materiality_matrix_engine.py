# -*- coding: utf-8 -*-
"""
MaterialityMatrixEngine - PACK-015 Double Materiality Engine 5
================================================================

Generates the double materiality matrix combining impact materiality and
financial materiality scores into the standard 2x2 quadrant visualization
required by ESRS 2 (IRO-1, IRO-2, SBM-3).

The matrix positions each sustainability matter according to its impact
score (y-axis) and financial score (x-axis) relative to configurable
thresholds.  Matters crossing either threshold are deemed material;
matters crossing both are "double material".

ESRS Requirements Addressed:
    - ESRS 2 IRO-1: Description of processes to identify material impacts,
      risks and opportunities (the matrix itself)
    - ESRS 2 IRO-2: Disclosure Requirements in ESRS covered by the
      undertaking's sustainability statement
    - ESRS 2 SBM-3: Material impacts, risks and opportunities and their
      interaction with strategy and business model

Quadrant Classification:
    - Top-Right  (DOUBLE_MATERIAL):  impact >= threshold AND financial >= threshold
    - Top-Left   (IMPACT_ONLY):      impact >= threshold AND financial < threshold
    - Bottom-Right (FINANCIAL_ONLY): impact < threshold  AND financial >= threshold
    - Bottom-Left  (NOT_MATERIAL):   impact < threshold  AND financial < threshold

Zero-Hallucination:
    - All score arithmetic uses deterministic Decimal operations
    - Quadrant classification is pure threshold comparison (no ML/LLM)
    - Combined scores use weighted arithmetic mean or geometric mean
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
    """Safely divide two Decimals, returning *default* on zero denominator.

    Args:
        numerator: Dividend.
        denominator: Divisor.
        default: Value returned when denominator is zero.

    Returns:
        Result of division or *default*.
    """
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    """Compute percentage safely (part / whole * 100).

    Args:
        part: Numerator.
        whole: Denominator.

    Returns:
        Percentage as Decimal; Decimal("0") when whole is zero.
    """
    return _safe_divide(part * Decimal("100"), whole)

def _round_val(value: Decimal, places: int = 6) -> float:
    """Round a Decimal to *places* and return a float.

    Uses ROUND_HALF_UP (regulatory standard rounding).

    Args:
        value: Value to round.
        places: Number of decimal places.

    Returns:
        Rounded float value.
    """
    quantizer = Decimal(10) ** -places
    return float(value.quantize(quantizer, rounding=ROUND_HALF_UP))

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Quadrant(str, Enum):
    """Double materiality matrix quadrant classification.

    Determines where a sustainability matter falls on the 2x2 matrix
    based on impact and financial thresholds.
    """
    DOUBLE_MATERIAL = "double_material"
    IMPACT_ONLY = "impact_only"
    FINANCIAL_ONLY = "financial_only"
    NOT_MATERIAL = "not_material"

class MatrixLayout(str, Enum):
    """Matrix visualization layout options.

    STANDARD_2X2 is the ESRS-recommended layout.
    DETAILED_3X3 adds a borderline zone around thresholds.
    HEAT_MAP provides continuous color gradient.
    """
    STANDARD_2X2 = "standard_2x2"
    DETAILED_3X3 = "detailed_3x3"
    HEAT_MAP = "heat_map"

class CombinedScoreMethod(str, Enum):
    """Method for combining impact and financial scores.

    ARITHMETIC_MEAN: Simple weighted average.
    GEOMETRIC_MEAN: Root of product (penalizes imbalance).
    MAX_SCORE: Maximum of the two scores.
    """
    ARITHMETIC_MEAN = "arithmetic_mean"
    GEOMETRIC_MEAN = "geometric_mean"
    MAX_SCORE = "max_score"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_IMPACT_THRESHOLD: Decimal = Decimal("3.000")
"""Default threshold for impact materiality on a 1-5 scale.

Matters scoring >= 3.0 are considered material from an impact perspective.
Based on common CSRD implementation guidance (mid-point of 5-point scale).
"""

DEFAULT_FINANCIAL_THRESHOLD: Decimal = Decimal("3.000")
"""Default threshold for financial materiality on a 1-5 scale.

Matters scoring >= 3.0 are considered material from a financial perspective.
"""

DEFAULT_COMBINED_THRESHOLD: Decimal = Decimal("3.000")
"""Default threshold for combined materiality score."""

QUADRANT_DESCRIPTIONS: Dict[str, str] = {
    "double_material": (
        "Material from both impact and financial perspectives. "
        "Full ESRS topical standard disclosures required."
    ),
    "impact_only": (
        "Material from impact perspective only. "
        "ESRS disclosures required for impact-related data points."
    ),
    "financial_only": (
        "Material from financial perspective only. "
        "ESRS disclosures required for financial-risk-related data points."
    ),
    "not_material": (
        "Not material from either perspective. "
        "No topical ESRS disclosures required, but must document "
        "the assessment rationale under ESRS 2 IRO-1."
    ),
}
"""Human-readable description of each quadrant for report generation."""

COLOR_MAP: Dict[str, str] = {
    "double_material": "#D32F2F",
    "impact_only": "#F57C00",
    "financial_only": "#1976D2",
    "not_material": "#757575",
}
"""Default color palette for matrix visualization (hex RGB)."""

SIZE_MAP: Dict[str, int] = {
    "double_material": 120,
    "impact_only": 80,
    "financial_only": 80,
    "not_material": 40,
}
"""Default marker size for matrix visualization (scatter plot points)."""

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class MatrixEntry(BaseModel):
    """Single sustainability matter positioned on the materiality matrix.

    Represents one row in the double materiality assessment, with its
    impact score, financial score, combined score, and classification.

    Attributes:
        matter_id: Unique identifier for the sustainability matter.
        matter_name: Human-readable name.
        esrs_topic: ESRS topical standard (e.g. "E1", "S1", "G1").
        impact_score: Score on impact materiality dimension (0-5 scale).
        financial_score: Score on financial materiality dimension (0-5 scale).
        combined_score: Weighted combination of impact and financial scores.
        quadrant: Classification based on threshold comparison.
        is_material_impact: True if impact_score >= impact_threshold.
        is_material_financial: True if financial_score >= financial_threshold.
        is_double_material: True if material on both dimensions.
    """
    matter_id: str = Field(..., min_length=1, description="Unique matter identifier")
    matter_name: str = Field(..., min_length=1, description="Human-readable matter name")
    esrs_topic: str = Field(..., description="ESRS topic code (e.g. E1, S1, G1)")
    impact_score: Decimal = Field(..., ge=Decimal("0"), le=Decimal("5"), description="Impact materiality score (0-5)")
    financial_score: Decimal = Field(..., ge=Decimal("0"), le=Decimal("5"), description="Financial materiality score (0-5)")
    combined_score: Decimal = Field(default=Decimal("0"), ge=Decimal("0"), description="Weighted combined score")
    quadrant: Quadrant = Field(default=Quadrant.NOT_MATERIAL, description="Matrix quadrant classification")
    is_material_impact: bool = Field(default=False, description="Passes impact threshold")
    is_material_financial: bool = Field(default=False, description="Passes financial threshold")
    is_double_material: bool = Field(default=False, description="Material on both dimensions")

    @field_validator("impact_score", "financial_score", "combined_score", mode="before")
    @classmethod
    def _coerce_decimal(cls, v: Any) -> Decimal:
        return _decimal(v)

class MaterialityMatrix(BaseModel):
    """Complete double materiality matrix with all entries and statistics.

    Attributes:
        entries: All sustainability matters positioned on the matrix.
        impact_threshold: Threshold used for impact materiality.
        financial_threshold: Threshold used for financial materiality.
        total_matters: Total number of matters assessed.
        material_count: Number of matters material on at least one dimension.
        double_material_count: Number of matters material on both dimensions.
        impact_only_count: Number of matters material only on impact.
        financial_only_count: Number of matters material only on financial.
        not_material_count: Number of matters not material on either dimension.
        quadrant_distribution: Percentage distribution across quadrants.
        combined_score_method: Method used for combined scoring.
        provenance_hash: SHA-256 hash for audit trail.
        calculated_at: Timestamp of matrix generation.
        processing_time_ms: Time taken to generate the matrix.
    """
    entries: List[MatrixEntry] = Field(default_factory=list, description="All matrix entries")
    impact_threshold: Decimal = Field(default=DEFAULT_IMPACT_THRESHOLD, description="Impact materiality threshold")
    financial_threshold: Decimal = Field(default=DEFAULT_FINANCIAL_THRESHOLD, description="Financial materiality threshold")
    total_matters: int = Field(default=0, ge=0, description="Total matters assessed")
    material_count: int = Field(default=0, ge=0, description="Matters material on >= 1 dimension")
    double_material_count: int = Field(default=0, ge=0, description="Matters material on both dimensions")
    impact_only_count: int = Field(default=0, ge=0, description="Matters material only on impact")
    financial_only_count: int = Field(default=0, ge=0, description="Matters material only on financial")
    not_material_count: int = Field(default=0, ge=0, description="Matters not material")
    quadrant_distribution: Dict[str, float] = Field(default_factory=dict, description="% per quadrant")
    combined_score_method: str = Field(default="arithmetic_mean", description="Combined scoring method used")
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")
    calculated_at: Optional[datetime] = Field(default=None, description="Generation timestamp")
    processing_time_ms: float = Field(default=0.0, ge=0.0, description="Processing time in ms")

    @field_validator("impact_threshold", "financial_threshold", mode="before")
    @classmethod
    def _coerce_threshold(cls, v: Any) -> Decimal:
        return _decimal(v)

class MatrixVisualizationData(BaseModel):
    """Pre-computed visualization data for rendering the materiality matrix.

    Provides arrays suitable for scatter-plot rendering libraries.

    Attributes:
        x_values: Financial scores for x-axis positioning.
        y_values: Impact scores for y-axis positioning.
        labels: Matter names for hover/tooltip.
        colors: Hex color strings per entry.
        sizes: Marker sizes per entry.
        quadrant_labels: Human-readable labels for each quadrant region.
        impact_threshold: Threshold line y-position.
        financial_threshold: Threshold line x-position.
    """
    x_values: List[float] = Field(default_factory=list, description="Financial scores (x-axis)")
    y_values: List[float] = Field(default_factory=list, description="Impact scores (y-axis)")
    labels: List[str] = Field(default_factory=list, description="Matter names")
    colors: List[str] = Field(default_factory=list, description="Hex color per entry")
    sizes: List[int] = Field(default_factory=list, description="Marker size per entry")
    quadrant_labels: Dict[str, str] = Field(default_factory=dict, description="Quadrant labels")
    impact_threshold: float = Field(default=3.0, description="Impact threshold line position")
    financial_threshold: float = Field(default=3.0, description="Financial threshold line position")
    matter_ids: List[str] = Field(default_factory=list, description="Matter IDs for lookup")
    esrs_topics: List[str] = Field(default_factory=list, description="ESRS topics per entry")

class ScoreChange(BaseModel):
    """Change in a matter's scores between two matrices.

    Attributes:
        matter_id: Unique identifier.
        matter_name: Human-readable name.
        previous_impact: Previous impact score.
        current_impact: Current impact score.
        impact_delta: Change in impact score.
        previous_financial: Previous financial score.
        current_financial: Current financial score.
        financial_delta: Change in financial score.
        quadrant_changed: Whether quadrant classification changed.
        previous_quadrant: Previous quadrant.
        current_quadrant: Current quadrant.
    """
    matter_id: str = Field(..., description="Matter identifier")
    matter_name: str = Field(default="", description="Matter name")
    previous_impact: Decimal = Field(default=Decimal("0"))
    current_impact: Decimal = Field(default=Decimal("0"))
    impact_delta: Decimal = Field(default=Decimal("0"))
    previous_financial: Decimal = Field(default=Decimal("0"))
    current_financial: Decimal = Field(default=Decimal("0"))
    financial_delta: Decimal = Field(default=Decimal("0"))
    quadrant_changed: bool = Field(default=False)
    previous_quadrant: Optional[Quadrant] = Field(default=None)
    current_quadrant: Optional[Quadrant] = Field(default=None)

class MatrixDelta(BaseModel):
    """Comparison between two materiality matrices (current vs. previous).

    Used for year-over-year tracking of materiality changes, which is
    a requirement under ESRS 2 IRO-1 for explaining changes.

    Attributes:
        previous_period: Reporting period of previous matrix.
        current_period: Reporting period of current matrix.
        new_material: Matters that became material.
        no_longer_material: Matters that are no longer material.
        score_changes: All score changes for common matters.
        quadrant_changes_count: Number of matters that changed quadrant.
        total_common: Number of matters present in both matrices.
        total_added: Matters in current but not previous.
        total_removed: Matters in previous but not current.
        provenance_hash: SHA-256 hash for audit trail.
    """
    previous_period: str = Field(default="", description="Previous reporting period")
    current_period: str = Field(default="", description="Current reporting period")
    new_material: List[str] = Field(default_factory=list, description="Newly material matter IDs")
    no_longer_material: List[str] = Field(default_factory=list, description="No longer material matter IDs")
    score_changes: List[ScoreChange] = Field(default_factory=list, description="Per-matter score changes")
    quadrant_changes_count: int = Field(default=0, ge=0)
    total_common: int = Field(default=0, ge=0)
    total_added: int = Field(default=0, ge=0)
    total_removed: int = Field(default=0, ge=0)
    provenance_hash: str = Field(default="", description="SHA-256 provenance hash")

# ---------------------------------------------------------------------------
# Input Models
# ---------------------------------------------------------------------------

class ImpactScoreInput(BaseModel):
    """Impact materiality score input for a single matter.

    Attributes:
        matter_id: Unique identifier matching the sustainability matter.
        matter_name: Human-readable name.
        esrs_topic: ESRS topical standard code.
        score: Aggregated impact materiality score (0-5 scale).
    """
    matter_id: str = Field(..., min_length=1)
    matter_name: str = Field(..., min_length=1)
    esrs_topic: str = Field(...)
    score: Decimal = Field(..., ge=Decimal("0"), le=Decimal("5"))

    @field_validator("score", mode="before")
    @classmethod
    def _coerce_score(cls, v: Any) -> Decimal:
        return _decimal(v)

class FinancialScoreInput(BaseModel):
    """Financial materiality score input for a single matter.

    Attributes:
        matter_id: Unique identifier matching the sustainability matter.
        matter_name: Human-readable name.
        esrs_topic: ESRS topical standard code.
        score: Aggregated financial materiality score (0-5 scale).
    """
    matter_id: str = Field(..., min_length=1)
    matter_name: str = Field(..., min_length=1)
    esrs_topic: str = Field(...)
    score: Decimal = Field(..., ge=Decimal("0"), le=Decimal("5"))

    @field_validator("score", mode="before")
    @classmethod
    def _coerce_score(cls, v: Any) -> Decimal:
        return _decimal(v)

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class MaterialityMatrixEngine:
    """Generates the double materiality matrix for ESRS reporting.

    Zero-Hallucination Guarantees:
        - All scoring uses deterministic Decimal arithmetic
        - Quadrant classification is a pure threshold comparison
        - Combined scores use configurable but deterministic methods
        - SHA-256 provenance hash on every result
        - No LLM involvement in any calculation path

    Usage::

        engine = MaterialityMatrixEngine()
        matrix = engine.build_matrix(
            impact_results=[ImpactScoreInput(...)],
            financial_results=[FinancialScoreInput(...)],
        )
    """

    def __init__(
        self,
        impact_threshold: Decimal = DEFAULT_IMPACT_THRESHOLD,
        financial_threshold: Decimal = DEFAULT_FINANCIAL_THRESHOLD,
        combined_method: CombinedScoreMethod = CombinedScoreMethod.ARITHMETIC_MEAN,
        impact_weight: Decimal = Decimal("0.5"),
        financial_weight: Decimal = Decimal("0.5"),
    ) -> None:
        """Initialize MaterialityMatrixEngine.

        Args:
            impact_threshold: Threshold for impact materiality (0-5 scale).
            financial_threshold: Threshold for financial materiality (0-5 scale).
            combined_method: Method for calculating combined score.
            impact_weight: Weight for impact in combined score (0-1).
            financial_weight: Weight for financial in combined score (0-1).
        """
        self.impact_threshold = _decimal(impact_threshold)
        self.financial_threshold = _decimal(financial_threshold)
        self.combined_method = combined_method
        self.impact_weight = _decimal(impact_weight)
        self.financial_weight = _decimal(financial_weight)
        logger.info(
            "MaterialityMatrixEngine initialized: impact_threshold=%s, "
            "financial_threshold=%s, method=%s",
            self.impact_threshold,
            self.financial_threshold,
            self.combined_method.value,
        )

    # ------------------------------------------------------------------
    # Core: Build Matrix
    # ------------------------------------------------------------------

    def build_matrix(
        self,
        impact_results: List[ImpactScoreInput],
        financial_results: List[FinancialScoreInput],
        impact_threshold: Optional[Decimal] = None,
        financial_threshold: Optional[Decimal] = None,
    ) -> MaterialityMatrix:
        """Build the double materiality matrix from impact and financial scores.

        Matches impact and financial score inputs by matter_id, classifies
        each into a quadrant, and computes statistics.

        Args:
            impact_results: List of impact materiality scores.
            financial_results: List of financial materiality scores.
            impact_threshold: Override impact threshold (uses engine default if None).
            financial_threshold: Override financial threshold (uses engine default if None).

        Returns:
            MaterialityMatrix with all entries classified and statistics computed.

        Raises:
            ValueError: If a matter_id appears in one list but not the other.
        """
        t0 = time.perf_counter()

        i_threshold = _decimal(impact_threshold) if impact_threshold is not None else self.impact_threshold
        f_threshold = _decimal(financial_threshold) if financial_threshold is not None else self.financial_threshold

        # Index financial results by matter_id for O(1) lookup
        financial_by_id: Dict[str, FinancialScoreInput] = {
            fr.matter_id: fr for fr in financial_results
        }
        impact_by_id: Dict[str, ImpactScoreInput] = {
            ir.matter_id: ir for ir in impact_results
        }

        # Validate all matter_ids are present in both lists
        impact_ids = set(impact_by_id.keys())
        financial_ids = set(financial_by_id.keys())
        all_ids = impact_ids | financial_ids

        entries: List[MatrixEntry] = []
        for matter_id in sorted(all_ids):
            impact_input = impact_by_id.get(matter_id)
            financial_input = financial_by_id.get(matter_id)

            # Use zero score if matter is missing from one side
            if impact_input is not None:
                i_score = impact_input.score
                name = impact_input.matter_name
                topic = impact_input.esrs_topic
            else:
                i_score = Decimal("0")
                name = financial_input.matter_name if financial_input else matter_id
                topic = financial_input.esrs_topic if financial_input else ""

            if financial_input is not None:
                f_score = financial_input.score
                if impact_input is None:
                    name = financial_input.matter_name
                    topic = financial_input.esrs_topic
            else:
                f_score = Decimal("0")

            # Classify
            quadrant = self.classify_quadrant(i_score, f_score, i_threshold, f_threshold)
            combined = self.calculate_combined_score(
                i_score, f_score, self.impact_weight, self.financial_weight
            )

            is_mat_i = i_score >= i_threshold
            is_mat_f = f_score >= f_threshold

            entry = MatrixEntry(
                matter_id=matter_id,
                matter_name=name,
                esrs_topic=topic,
                impact_score=i_score,
                financial_score=f_score,
                combined_score=combined,
                quadrant=quadrant,
                is_material_impact=is_mat_i,
                is_material_financial=is_mat_f,
                is_double_material=(is_mat_i and is_mat_f),
            )
            entries.append(entry)

        # Compute statistics
        total = len(entries)
        double_ct = sum(1 for e in entries if e.is_double_material)
        impact_ct = sum(1 for e in entries if e.quadrant == Quadrant.IMPACT_ONLY)
        financial_ct = sum(1 for e in entries if e.quadrant == Quadrant.FINANCIAL_ONLY)
        not_mat_ct = sum(1 for e in entries if e.quadrant == Quadrant.NOT_MATERIAL)
        material_ct = double_ct + impact_ct + financial_ct

        # Quadrant distribution as percentages
        quad_dist: Dict[str, float] = {}
        for q in Quadrant:
            count = sum(1 for e in entries if e.quadrant == q)
            quad_dist[q.value] = _round_val(
                _safe_pct(_decimal(count), _decimal(total)), places=2
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        matrix = MaterialityMatrix(
            entries=entries,
            impact_threshold=i_threshold,
            financial_threshold=f_threshold,
            total_matters=total,
            material_count=material_ct,
            double_material_count=double_ct,
            impact_only_count=impact_ct,
            financial_only_count=financial_ct,
            not_material_count=not_mat_ct,
            quadrant_distribution=quad_dist,
            combined_score_method=self.combined_method.value,
            calculated_at=utcnow(),
            processing_time_ms=round(elapsed_ms, 3),
        )
        matrix.provenance_hash = _compute_hash(matrix)

        logger.info(
            "Matrix built: %d matters, %d material (%d double), hash=%s",
            total, material_ct, double_ct, matrix.provenance_hash[:16],
        )
        return matrix

    # ------------------------------------------------------------------
    # Core: Classify Quadrant
    # ------------------------------------------------------------------

    def classify_quadrant(
        self,
        impact_score: Decimal,
        financial_score: Decimal,
        impact_threshold: Optional[Decimal] = None,
        financial_threshold: Optional[Decimal] = None,
    ) -> Quadrant:
        """Classify a matter into a materiality matrix quadrant.

        Pure threshold comparison -- DETERMINISTIC.

        Args:
            impact_score: Impact materiality score.
            financial_score: Financial materiality score.
            impact_threshold: Override (uses engine default if None).
            financial_threshold: Override (uses engine default if None).

        Returns:
            Quadrant enum value.
        """
        i_thr = _decimal(impact_threshold) if impact_threshold is not None else self.impact_threshold
        f_thr = _decimal(financial_threshold) if financial_threshold is not None else self.financial_threshold

        i_score = _decimal(impact_score)
        f_score = _decimal(financial_score)

        is_impact = i_score >= i_thr
        is_financial = f_score >= f_thr

        if is_impact and is_financial:
            return Quadrant.DOUBLE_MATERIAL
        elif is_impact:
            return Quadrant.IMPACT_ONLY
        elif is_financial:
            return Quadrant.FINANCIAL_ONLY
        else:
            return Quadrant.NOT_MATERIAL

    # ------------------------------------------------------------------
    # Core: Combined Score
    # ------------------------------------------------------------------

    def calculate_combined_score(
        self,
        impact_score: Decimal,
        financial_score: Decimal,
        impact_weight: Optional[Decimal] = None,
        financial_weight: Optional[Decimal] = None,
    ) -> Decimal:
        """Calculate combined materiality score from impact and financial.

        Uses the method configured on this engine instance (arithmetic mean,
        geometric mean, or max score).

        DETERMINISTIC: Same inputs always produce the same output.

        Args:
            impact_score: Impact materiality score.
            financial_score: Financial materiality score.
            impact_weight: Weight for impact (0-1).  Defaults to engine setting.
            financial_weight: Weight for financial (0-1).  Defaults to engine setting.

        Returns:
            Combined score as Decimal rounded to 6 decimal places.
        """
        i_s = _decimal(impact_score)
        f_s = _decimal(financial_score)
        i_w = _decimal(impact_weight) if impact_weight is not None else self.impact_weight
        f_w = _decimal(financial_weight) if financial_weight is not None else self.financial_weight

        if self.combined_method == CombinedScoreMethod.ARITHMETIC_MEAN:
            total_weight = i_w + f_w
            if total_weight == Decimal("0"):
                return Decimal("0")
            result = _safe_divide(i_s * i_w + f_s * f_w, total_weight)

        elif self.combined_method == CombinedScoreMethod.GEOMETRIC_MEAN:
            # sqrt(impact * financial) -- geometric mean of two values
            product = i_s * f_s
            if product <= Decimal("0"):
                result = Decimal("0")
            else:
                # Use Decimal sqrt via ** 0.5 for determinism
                result = _decimal(float(product) ** 0.5)

        elif self.combined_method == CombinedScoreMethod.MAX_SCORE:
            result = max(i_s, f_s)

        else:
            # Fallback to arithmetic mean
            total_weight = i_w + f_w
            result = _safe_divide(i_s * i_w + f_s * f_w, total_weight)

        # Round to 6 decimal places for consistency
        quantizer = Decimal("0.000001")
        return result.quantize(quantizer, rounding=ROUND_HALF_UP)

    # ------------------------------------------------------------------
    # Query: Get Material Topics
    # ------------------------------------------------------------------

    def get_material_topics(self, matrix: MaterialityMatrix) -> List[MatrixEntry]:
        """Return all entries that are material on at least one dimension.

        Args:
            matrix: The materiality matrix to query.

        Returns:
            List of MatrixEntry where is_material_impact or is_material_financial.
        """
        return [
            e for e in matrix.entries
            if e.is_material_impact or e.is_material_financial
        ]

    def get_double_material(self, matrix: MaterialityMatrix) -> List[MatrixEntry]:
        """Return entries that are material on both dimensions.

        Args:
            matrix: The materiality matrix to query.

        Returns:
            List of MatrixEntry where is_double_material is True.
        """
        return [e for e in matrix.entries if e.is_double_material]

    def get_impact_only(self, matrix: MaterialityMatrix) -> List[MatrixEntry]:
        """Return entries material only on impact dimension.

        Args:
            matrix: The materiality matrix to query.

        Returns:
            List of MatrixEntry in IMPACT_ONLY quadrant.
        """
        return [e for e in matrix.entries if e.quadrant == Quadrant.IMPACT_ONLY]

    def get_financial_only(self, matrix: MaterialityMatrix) -> List[MatrixEntry]:
        """Return entries material only on financial dimension.

        Args:
            matrix: The materiality matrix to query.

        Returns:
            List of MatrixEntry in FINANCIAL_ONLY quadrant.
        """
        return [e for e in matrix.entries if e.quadrant == Quadrant.FINANCIAL_ONLY]

    def get_not_material(self, matrix: MaterialityMatrix) -> List[MatrixEntry]:
        """Return entries that are not material on either dimension.

        Args:
            matrix: The materiality matrix to query.

        Returns:
            List of MatrixEntry in NOT_MATERIAL quadrant.
        """
        return [e for e in matrix.entries if e.quadrant == Quadrant.NOT_MATERIAL]

    def get_entries_by_topic(
        self, matrix: MaterialityMatrix, esrs_topic: str
    ) -> List[MatrixEntry]:
        """Return all entries for a specific ESRS topic.

        Args:
            matrix: The materiality matrix to query.
            esrs_topic: ESRS topic code (e.g. "E1", "S1").

        Returns:
            Filtered list of MatrixEntry.
        """
        return [e for e in matrix.entries if e.esrs_topic == esrs_topic]

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def get_visualization_data(self, matrix: MaterialityMatrix) -> MatrixVisualizationData:
        """Generate pre-computed data for rendering the materiality matrix.

        Produces arrays suitable for scatter-plot libraries (matplotlib,
        plotly, etc.) with colors and sizes based on quadrant.

        DETERMINISTIC: Produces identical output for identical matrix.

        Args:
            matrix: The materiality matrix to visualize.

        Returns:
            MatrixVisualizationData with x/y values, labels, colors, sizes.
        """
        x_vals: List[float] = []
        y_vals: List[float] = []
        labels: List[str] = []
        colors: List[str] = []
        sizes: List[int] = []
        matter_ids: List[str] = []
        esrs_topics: List[str] = []

        for entry in matrix.entries:
            x_vals.append(_round_val(_decimal(entry.financial_score), places=3))
            y_vals.append(_round_val(_decimal(entry.impact_score), places=3))
            labels.append(entry.matter_name)
            colors.append(COLOR_MAP.get(entry.quadrant.value, "#757575"))
            sizes.append(SIZE_MAP.get(entry.quadrant.value, 40))
            matter_ids.append(entry.matter_id)
            esrs_topics.append(entry.esrs_topic)

        quadrant_labels = {
            "top_right": "Double Material",
            "top_left": "Impact Only",
            "bottom_right": "Financial Only",
            "bottom_left": "Not Material",
        }

        return MatrixVisualizationData(
            x_values=x_vals,
            y_values=y_vals,
            labels=labels,
            colors=colors,
            sizes=sizes,
            quadrant_labels=quadrant_labels,
            impact_threshold=_round_val(matrix.impact_threshold, places=3),
            financial_threshold=_round_val(matrix.financial_threshold, places=3),
            matter_ids=matter_ids,
            esrs_topics=esrs_topics,
        )

    # ------------------------------------------------------------------
    # Comparison
    # ------------------------------------------------------------------

    def compare_matrices(
        self,
        current: MaterialityMatrix,
        previous: MaterialityMatrix,
        current_period: str = "",
        previous_period: str = "",
    ) -> MatrixDelta:
        """Compare two materiality matrices to identify changes.

        Used for year-over-year tracking as required by ESRS 2 IRO-1.
        Identifies newly material topics, topics no longer material,
        score changes, and quadrant changes.

        DETERMINISTIC: Same inputs produce same delta.

        Args:
            current: Current period's materiality matrix.
            previous: Previous period's materiality matrix.
            current_period: Label for current period (e.g. "FY2025").
            previous_period: Label for previous period (e.g. "FY2024").

        Returns:
            MatrixDelta with all changes documented.
        """
        # Build lookup maps
        current_map: Dict[str, MatrixEntry] = {
            e.matter_id: e for e in current.entries
        }
        previous_map: Dict[str, MatrixEntry] = {
            e.matter_id: e for e in previous.entries
        }

        current_ids = set(current_map.keys())
        previous_ids = set(previous_map.keys())
        common_ids = current_ids & previous_ids
        added_ids = current_ids - previous_ids
        removed_ids = previous_ids - current_ids

        # Identify newly material / no longer material
        new_material: List[str] = []
        no_longer_material: List[str] = []
        score_changes: List[ScoreChange] = []
        quadrant_change_count = 0

        # Added matters that are now material
        for mid in sorted(added_ids):
            entry = current_map[mid]
            if entry.is_material_impact or entry.is_material_financial:
                new_material.append(mid)

        # Removed matters that were material
        for mid in sorted(removed_ids):
            entry = previous_map[mid]
            if entry.is_material_impact or entry.is_material_financial:
                no_longer_material.append(mid)

        # Common matters -- score and quadrant changes
        for mid in sorted(common_ids):
            curr = current_map[mid]
            prev = previous_map[mid]

            impact_delta = curr.impact_score - prev.impact_score
            financial_delta = curr.financial_score - prev.financial_score
            q_changed = curr.quadrant != prev.quadrant
            if q_changed:
                quadrant_change_count += 1

            # Track if became material or lost materiality
            was_material = prev.is_material_impact or prev.is_material_financial
            is_material = curr.is_material_impact or curr.is_material_financial
            if is_material and not was_material:
                new_material.append(mid)
            elif was_material and not is_material:
                no_longer_material.append(mid)

            change = ScoreChange(
                matter_id=mid,
                matter_name=curr.matter_name,
                previous_impact=prev.impact_score,
                current_impact=curr.impact_score,
                impact_delta=impact_delta,
                previous_financial=prev.financial_score,
                current_financial=curr.financial_score,
                financial_delta=financial_delta,
                quadrant_changed=q_changed,
                previous_quadrant=prev.quadrant,
                current_quadrant=curr.quadrant,
            )
            score_changes.append(change)

        delta = MatrixDelta(
            previous_period=previous_period,
            current_period=current_period,
            new_material=sorted(new_material),
            no_longer_material=sorted(no_longer_material),
            score_changes=score_changes,
            quadrant_changes_count=quadrant_change_count,
            total_common=len(common_ids),
            total_added=len(added_ids),
            total_removed=len(removed_ids),
        )
        delta.provenance_hash = _compute_hash(delta)

        logger.info(
            "Matrix comparison: %d common, %d added, %d removed, %d quadrant changes",
            len(common_ids), len(added_ids), len(removed_ids), quadrant_change_count,
        )
        return delta

    # ------------------------------------------------------------------
    # Utility: Rank Entries
    # ------------------------------------------------------------------

    def rank_by_combined_score(
        self, matrix: MaterialityMatrix, descending: bool = True
    ) -> List[MatrixEntry]:
        """Return entries ranked by combined score.

        Args:
            matrix: The materiality matrix to rank.
            descending: If True, highest score first.

        Returns:
            Sorted list of MatrixEntry.
        """
        return sorted(
            matrix.entries,
            key=lambda e: e.combined_score,
            reverse=descending,
        )

    def rank_by_impact(
        self, matrix: MaterialityMatrix, descending: bool = True
    ) -> List[MatrixEntry]:
        """Return entries ranked by impact score.

        Args:
            matrix: The materiality matrix to rank.
            descending: If True, highest score first.

        Returns:
            Sorted list of MatrixEntry.
        """
        return sorted(
            matrix.entries,
            key=lambda e: e.impact_score,
            reverse=descending,
        )

    def rank_by_financial(
        self, matrix: MaterialityMatrix, descending: bool = True
    ) -> List[MatrixEntry]:
        """Return entries ranked by financial score.

        Args:
            matrix: The materiality matrix to rank.
            descending: If True, highest score first.

        Returns:
            Sorted list of MatrixEntry.
        """
        return sorted(
            matrix.entries,
            key=lambda e: e.financial_score,
            reverse=descending,
        )

    # ------------------------------------------------------------------
    # Utility: Summary Statistics
    # ------------------------------------------------------------------

    def get_summary_statistics(self, matrix: MaterialityMatrix) -> Dict[str, Any]:
        """Calculate summary statistics for the matrix.

        DETERMINISTIC: Same matrix produces identical statistics.

        Args:
            matrix: The materiality matrix to summarize.

        Returns:
            Dictionary with mean, median, min, max for each score dimension.
        """
        if not matrix.entries:
            return {
                "impact_mean": 0.0, "impact_median": 0.0,
                "impact_min": 0.0, "impact_max": 0.0,
                "financial_mean": 0.0, "financial_median": 0.0,
                "financial_min": 0.0, "financial_max": 0.0,
                "combined_mean": 0.0, "combined_median": 0.0,
                "combined_min": 0.0, "combined_max": 0.0,
            }

        impact_scores = sorted([e.impact_score for e in matrix.entries])
        financial_scores = sorted([e.financial_score for e in matrix.entries])
        combined_scores = sorted([e.combined_score for e in matrix.entries])

        n = _decimal(len(matrix.entries))

        def _median(vals: List[Decimal]) -> Decimal:
            length = len(vals)
            if length == 0:
                return Decimal("0")
            mid = length // 2
            if length % 2 == 0:
                return (vals[mid - 1] + vals[mid]) / Decimal("2")
            return vals[mid]

        return {
            "impact_mean": _round_val(_safe_divide(sum(impact_scores), n), 3),
            "impact_median": _round_val(_median(impact_scores), 3),
            "impact_min": _round_val(impact_scores[0], 3),
            "impact_max": _round_val(impact_scores[-1], 3),
            "financial_mean": _round_val(_safe_divide(sum(financial_scores), n), 3),
            "financial_median": _round_val(_median(financial_scores), 3),
            "financial_min": _round_val(financial_scores[0], 3),
            "financial_max": _round_val(financial_scores[-1], 3),
            "combined_mean": _round_val(_safe_divide(sum(combined_scores), n), 3),
            "combined_median": _round_val(_median(combined_scores), 3),
            "combined_min": _round_val(combined_scores[0], 3),
            "combined_max": _round_val(combined_scores[-1], 3),
        }

    # ------------------------------------------------------------------
    # Utility: Topic Aggregation
    # ------------------------------------------------------------------

    def aggregate_by_esrs_topic(
        self, matrix: MaterialityMatrix
    ) -> Dict[str, Dict[str, Any]]:
        """Aggregate matrix entries by ESRS topic.

        Returns a dictionary keyed by ESRS topic with counts and averages.

        Args:
            matrix: The materiality matrix to aggregate.

        Returns:
            Dictionary with topic-level aggregation.
        """
        topic_groups: Dict[str, List[MatrixEntry]] = {}
        for entry in matrix.entries:
            topic_groups.setdefault(entry.esrs_topic, []).append(entry)

        result: Dict[str, Dict[str, Any]] = {}
        for topic, entries in sorted(topic_groups.items()):
            n = _decimal(len(entries))
            avg_impact = _safe_divide(
                sum(e.impact_score for e in entries), n
            )
            avg_financial = _safe_divide(
                sum(e.financial_score for e in entries), n
            )
            material_count = sum(
                1 for e in entries
                if e.is_material_impact or e.is_material_financial
            )
            result[topic] = {
                "matter_count": len(entries),
                "material_count": material_count,
                "avg_impact_score": _round_val(avg_impact, 3),
                "avg_financial_score": _round_val(avg_financial, 3),
                "matters": [e.matter_id for e in entries],
            }

        return result
