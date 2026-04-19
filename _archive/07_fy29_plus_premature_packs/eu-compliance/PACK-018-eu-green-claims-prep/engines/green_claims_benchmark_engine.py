# -*- coding: utf-8 -*-
"""
GreenClaimsBenchmarkEngine - PACK-018 EU Green Claims Prep Engine 8
====================================================================

Provides cross-portfolio scoring, peer comparison, and maturity
assessment for entities preparing for the EU Green Claims Directive
(proposed Directive COM/2023/166 final) and the Empowering Consumers
Directive (Directive 2024/825).

This engine aggregates outputs from the other PACK-018 engines
(substantiation, label governance, comparative claims, greenwashing
detection, and trader obligations) into a unified readiness score
and maturity classification.

Maturity Framework:
    Level 1 - INITIAL (0-20):
        Ad-hoc environmental claims with no systematic substantiation.
        No evidence management, no governance structure.
    Level 2 - DEVELOPING (20-40):
        Basic awareness of Green Claims Directive requirements.
        Some evidence collection, but inconsistent and incomplete.
    Level 3 - DEFINED (40-60):
        Defined processes for claim substantiation.
        Partial evidence coverage, some label governance in place.
    Level 4 - MANAGED (60-80):
        Systematic claim management with comprehensive evidence.
        Regular reviews, verifier engagement, greenwashing controls.
    Level 5 - OPTIMIZING (80-100):
        Continuous improvement of claims program.
        Full regulatory alignment, proactive risk management,
        best-in-class substantiation and transparency.

Benchmark Dimensions:
    - Substantiation Quality: Depth and rigour of claim substantiation
    - Evidence Completeness: Coverage and accessibility of supporting evidence
    - Label Governance: Compliance of labels with certification requirements
    - Greenwashing Risk: Inverse of greenwashing detection score
    - Verification Readiness: Preparedness for independent verification
    - Regulatory Alignment: Overall alignment with Green Claims Directive

Zero-Hallucination:
    - Portfolio metrics use deterministic aggregation
    - Maturity scoring uses fixed threshold ranges
    - Peer benchmarking uses deterministic percentile calculations
    - Roadmap priorities use deterministic gap analysis
    - SHA-256 provenance hash on every result
    - No LLM involvement in any calculation path

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-018 EU Green Claims Prep
Status:  Production Ready
"""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

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

    Args:
        data: Data to hash (dict, Pydantic model, or other).

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    if hasattr(data, "model_dump"):
        serializable = data.model_dump(mode="json")
    elif isinstance(data, dict):
        serializable = data
    else:
        serializable = str(data)
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _decimal(value: Any) -> Decimal:
    """Convert value to Decimal safely.

    Args:
        value: Numeric value (int, float, str, or Decimal).

    Returns:
        Decimal representation.
    """
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))

def _safe_divide(
    numerator: Decimal, denominator: Decimal, default: Decimal = Decimal("0")
) -> Decimal:
    """Safely divide two Decimal numbers, returning *default* on zero denominator."""
    if denominator == Decimal("0"):
        return default
    return numerator / denominator

def _round_val(value: Decimal, places: int = 3) -> Decimal:
    """Round a Decimal value to the specified number of decimal places.

    Uses ROUND_HALF_UP for regulatory consistency.

    Args:
        value: Decimal value to round.
        places: Number of decimal places (default 3).

    Returns:
        Rounded Decimal value.
    """
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MaturityLevel(str, Enum):
    """Green Claims readiness maturity level.

    Five-level maturity model aligned with capability maturity
    frameworks, adapted for EU Green Claims Directive preparedness.
    """
    INITIAL = "initial"
    DEVELOPING = "developing"
    DEFINED = "defined"
    MANAGED = "managed"
    OPTIMIZING = "optimizing"

class BenchmarkDimension(str, Enum):
    """Dimensions of the Green Claims readiness benchmark.

    Each dimension captures a distinct aspect of regulatory
    preparedness for the Green Claims Directive.
    """
    SUBSTANTIATION_QUALITY = "substantiation_quality"
    EVIDENCE_COMPLETENESS = "evidence_completeness"
    LABEL_GOVERNANCE = "label_governance"
    GREENWASHING_RISK = "greenwashing_risk"
    VERIFICATION_READINESS = "verification_readiness"
    REGULATORY_ALIGNMENT = "regulatory_alignment"

class ImprovementTimeframe(str, Enum):
    """Timeframe for improvement actions in the roadmap.

    Categorizes recommended actions by urgency and implementation
    timeline.
    """
    IMMEDIATE = "immediate"
    SHORT_TERM = "short_term"
    MEDIUM_TERM = "medium_term"
    LONG_TERM = "long_term"

class PeerComparisonOutcome(str, Enum):
    """Outcome of peer benchmarking comparison.

    Classifies how the entity performs relative to peers.
    """
    LEADING = "leading"
    ABOVE_AVERAGE = "above_average"
    AVERAGE = "average"
    BELOW_AVERAGE = "below_average"
    LAGGING = "lagging"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maturity level thresholds (lower bound inclusive).
MATURITY_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    MaturityLevel.INITIAL.value: {
        "lower_bound": Decimal("0"),
        "upper_bound": Decimal("20"),
        "description": (
            "Ad-hoc environmental claims with no systematic "
            "substantiation processes in place"
        ),
        "characteristics": [
            "No formal claim management process",
            "Claims made without structured evidence",
            "No awareness of Green Claims Directive requirements",
            "High greenwashing risk",
        ],
        "level_number": 1,
    },
    MaturityLevel.DEVELOPING.value: {
        "lower_bound": Decimal("20"),
        "upper_bound": Decimal("40"),
        "description": (
            "Basic awareness of Green Claims Directive requirements "
            "with initial evidence collection efforts"
        ),
        "characteristics": [
            "Some awareness of regulatory requirements",
            "Initial evidence collection for key claims",
            "Inconsistent substantiation practices",
            "Limited label governance",
        ],
        "level_number": 2,
    },
    MaturityLevel.DEFINED.value: {
        "lower_bound": Decimal("40"),
        "upper_bound": Decimal("60"),
        "description": (
            "Defined processes for claim substantiation with "
            "partial evidence coverage and basic governance"
        ),
        "characteristics": [
            "Documented claim substantiation procedures",
            "Partial evidence coverage across claims",
            "Basic label governance framework",
            "Initial greenwashing screening",
        ],
        "level_number": 3,
    },
    MaturityLevel.MANAGED.value: {
        "lower_bound": Decimal("60"),
        "upper_bound": Decimal("80"),
        "description": (
            "Systematic claim management with comprehensive "
            "evidence and regular review cycles"
        ),
        "characteristics": [
            "Comprehensive evidence for all claims",
            "Regular claim review and update cycles",
            "Verifier engagement in progress or complete",
            "Active greenwashing risk management",
        ],
        "level_number": 4,
    },
    MaturityLevel.OPTIMIZING.value: {
        "lower_bound": Decimal("80"),
        "upper_bound": Decimal("100"),
        "description": (
            "Continuous improvement with full regulatory alignment "
            "and proactive risk management"
        ),
        "characteristics": [
            "Full compliance with Green Claims Directive",
            "Continuous improvement of claims program",
            "Best-in-class substantiation and transparency",
            "Proactive greenwashing prevention",
        ],
        "level_number": 5,
    },
}

# Dimension descriptions and weights for overall score calculation.
DIMENSION_SPECS: Dict[str, Dict[str, Any]] = {
    BenchmarkDimension.SUBSTANTIATION_QUALITY.value: {
        "name": "Substantiation Quality",
        "description": (
            "Depth and rigour of claim substantiation methodology, "
            "including LCA coverage and data quality"
        ),
        "weight": Decimal("0.25"),
        "max_score": Decimal("100"),
    },
    BenchmarkDimension.EVIDENCE_COMPLETENESS.value: {
        "name": "Evidence Completeness",
        "description": (
            "Coverage and accessibility of supporting evidence "
            "across all environmental claims"
        ),
        "weight": Decimal("0.20"),
        "max_score": Decimal("100"),
    },
    BenchmarkDimension.LABEL_GOVERNANCE.value: {
        "name": "Label Governance",
        "description": (
            "Compliance of sustainability labels with certification "
            "requirements and public authority standards"
        ),
        "weight": Decimal("0.15"),
        "max_score": Decimal("100"),
    },
    BenchmarkDimension.GREENWASHING_RISK.value: {
        "name": "Greenwashing Risk (Inverse)",
        "description": (
            "Inverse of greenwashing detection risk score; lower "
            "greenwashing risk yields higher benchmark score"
        ),
        "weight": Decimal("0.15"),
        "max_score": Decimal("100"),
    },
    BenchmarkDimension.VERIFICATION_READINESS.value: {
        "name": "Verification Readiness",
        "description": (
            "Preparedness for independent third-party verification "
            "of environmental claims"
        ),
        "weight": Decimal("0.15"),
        "max_score": Decimal("100"),
    },
    BenchmarkDimension.REGULATORY_ALIGNMENT.value: {
        "name": "Regulatory Alignment",
        "description": (
            "Overall alignment with the EU Green Claims Directive "
            "and Empowering Consumers Directive requirements"
        ),
        "weight": Decimal("0.10"),
        "max_score": Decimal("100"),
    },
}

# Peer comparison percentile thresholds.
PEER_PERCENTILE_THRESHOLDS: Dict[str, Decimal] = {
    PeerComparisonOutcome.LEADING.value: Decimal("80"),
    PeerComparisonOutcome.ABOVE_AVERAGE.value: Decimal("60"),
    PeerComparisonOutcome.AVERAGE.value: Decimal("40"),
    PeerComparisonOutcome.BELOW_AVERAGE.value: Decimal("20"),
    PeerComparisonOutcome.LAGGING.value: Decimal("0"),
}

# Improvement action templates by dimension.
IMPROVEMENT_ACTIONS: Dict[str, List[Dict[str, Any]]] = {
    BenchmarkDimension.SUBSTANTIATION_QUALITY.value: [
        {
            "action": "Conduct PEF/OEF-compliant life-cycle assessment for all claims",
            "timeframe": ImprovementTimeframe.MEDIUM_TERM.value,
            "effort_hours": 120,
            "impact": "high",
        },
        {
            "action": "Improve primary data coverage to meet Data Quality Requirements",
            "timeframe": ImprovementTimeframe.SHORT_TERM.value,
            "effort_hours": 80,
            "impact": "high",
        },
        {
            "action": "Establish peer review process for substantiation assessments",
            "timeframe": ImprovementTimeframe.MEDIUM_TERM.value,
            "effort_hours": 40,
            "impact": "medium",
        },
    ],
    BenchmarkDimension.EVIDENCE_COMPLETENESS.value: [
        {
            "action": "Create centralized evidence repository for all environmental claims",
            "timeframe": ImprovementTimeframe.SHORT_TERM.value,
            "effort_hours": 60,
            "impact": "high",
        },
        {
            "action": "Map evidence gaps and prioritize collection for highest-risk claims",
            "timeframe": ImprovementTimeframe.IMMEDIATE.value,
            "effort_hours": 24,
            "impact": "high",
        },
        {
            "action": "Implement consumer-facing evidence access (QR codes, web portal)",
            "timeframe": ImprovementTimeframe.MEDIUM_TERM.value,
            "effort_hours": 80,
            "impact": "medium",
        },
    ],
    BenchmarkDimension.LABEL_GOVERNANCE.value: [
        {
            "action": "Audit all sustainability labels for certification compliance",
            "timeframe": ImprovementTimeframe.IMMEDIATE.value,
            "effort_hours": 32,
            "impact": "critical",
        },
        {
            "action": "Remove or replace uncertified/self-awarded environmental labels",
            "timeframe": ImprovementTimeframe.SHORT_TERM.value,
            "effort_hours": 16,
            "impact": "critical",
        },
        {
            "action": "Establish label approval governance process",
            "timeframe": ImprovementTimeframe.MEDIUM_TERM.value,
            "effort_hours": 40,
            "impact": "medium",
        },
    ],
    BenchmarkDimension.GREENWASHING_RISK.value: [
        {
            "action": "Review and remove vague environmental terms from all communications",
            "timeframe": ImprovementTimeframe.IMMEDIATE.value,
            "effort_hours": 16,
            "impact": "high",
        },
        {
            "action": "Eliminate carbon neutrality claims based on offsets",
            "timeframe": ImprovementTimeframe.SHORT_TERM.value,
            "effort_hours": 8,
            "impact": "critical",
        },
        {
            "action": "Implement pre-publication greenwashing screening process",
            "timeframe": ImprovementTimeframe.MEDIUM_TERM.value,
            "effort_hours": 40,
            "impact": "high",
        },
    ],
    BenchmarkDimension.VERIFICATION_READINESS.value: [
        {
            "action": "Identify and engage accredited independent verifier",
            "timeframe": ImprovementTimeframe.SHORT_TERM.value,
            "effort_hours": 24,
            "impact": "high",
        },
        {
            "action": "Prepare verification-ready documentation package",
            "timeframe": ImprovementTimeframe.MEDIUM_TERM.value,
            "effort_hours": 80,
            "impact": "high",
        },
        {
            "action": "Conduct internal pre-verification readiness assessment",
            "timeframe": ImprovementTimeframe.SHORT_TERM.value,
            "effort_hours": 40,
            "impact": "medium",
        },
    ],
    BenchmarkDimension.REGULATORY_ALIGNMENT.value: [
        {
            "action": "Complete trader obligation assessment and gap analysis",
            "timeframe": ImprovementTimeframe.IMMEDIATE.value,
            "effort_hours": 16,
            "impact": "high",
        },
        {
            "action": "Develop Green Claims Directive compliance implementation plan",
            "timeframe": ImprovementTimeframe.SHORT_TERM.value,
            "effort_hours": 40,
            "impact": "high",
        },
        {
            "action": "Assign compliance responsibilities and establish governance",
            "timeframe": ImprovementTimeframe.SHORT_TERM.value,
            "effort_hours": 24,
            "impact": "medium",
        },
    ],
}

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class PortfolioMetrics(BaseModel):
    """Aggregate metrics for a portfolio of environmental claims.

    Captures the summary statistics from claim-level assessments
    to feed into the benchmark scoring engine.
    """
    total_claims: int = Field(
        ...,
        ge=0,
        description="Total number of environmental claims in the portfolio",
    )
    compliant_claims: int = Field(
        0,
        ge=0,
        description="Number of claims that are fully compliant",
    )
    non_compliant_claims: int = Field(
        0,
        ge=0,
        description="Number of claims that are non-compliant",
    )
    average_substantiation_score: Decimal = Field(
        Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Average substantiation quality score (0-100)",
    )
    greenwashing_risk_score: Decimal = Field(
        Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Aggregate greenwashing risk score (0-100)",
    )
    evidence_coverage_pct: Decimal = Field(
        Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Percentage of claims with complete evidence",
    )
    verification_ready_pct: Decimal = Field(
        Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Percentage of claims ready for independent verification",
    )
    label_compliance_pct: Decimal = Field(
        Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Percentage of labels that are certification-compliant",
    )
    regulatory_alignment_pct: Decimal = Field(
        Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Percentage alignment with Directive requirements",
    )

    @field_validator("compliant_claims")
    @classmethod
    def compliant_not_exceed_total(cls, v: int, info: Any) -> int:
        """Validate compliant_claims does not exceed total_claims."""
        total = info.data.get("total_claims", 0)
        if v > total:
            raise ValueError(
                f"compliant_claims ({v}) cannot exceed total_claims ({total})"
            )
        return v

class BenchmarkResult(BaseModel):
    """Result of a benchmark assessment for an entity.

    Encapsulates the overall score, maturity level, dimension-level
    scores, and improvement recommendations.
    """
    entity_name: str = Field(
        ...,
        description="Name of the entity being benchmarked",
    )
    overall_score: Decimal = Field(
        Decimal("0"),
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Overall benchmark score (0-100)",
    )
    maturity_level: str = Field(
        MaturityLevel.INITIAL.value,
        description="Current maturity level classification",
    )
    dimension_scores: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Scores for each benchmark dimension (0-100)",
    )
    strengths: List[str] = Field(
        default_factory=list,
        description="Identified strengths based on high-scoring dimensions",
    )
    weaknesses: List[str] = Field(
        default_factory=list,
        description="Identified weaknesses based on low-scoring dimensions",
    )
    improvement_priorities: List[str] = Field(
        default_factory=list,
        description="Prioritized list of improvement areas",
    )

class PeerDataPoint(BaseModel):
    """A single peer entity's benchmark data for comparison.

    Captures peer entity scores for peer-to-peer benchmarking.
    """
    entity_name: str = Field(
        ...,
        description="Name of the peer entity",
    )
    overall_score: Decimal = Field(
        ...,
        ge=Decimal("0"),
        le=Decimal("100"),
        description="Peer overall benchmark score (0-100)",
    )
    dimension_scores: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Peer dimension scores",
    )
    industry_sector: Optional[str] = Field(
        None,
        description="Industry sector of the peer entity",
    )

# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class GreenClaimsBenchmarkEngine:
    """Engine for cross-portfolio scoring and maturity assessment.

    Aggregates results from other PACK-018 engines into a unified
    readiness score, determines maturity level, performs peer
    benchmarking, and generates improvement roadmaps.

    Attributes:
        engine_id: Unique identifier for this engine instance.
        version: Module version string.

    Example:
        >>> engine = GreenClaimsBenchmarkEngine()
        >>> metrics = PortfolioMetrics(
        ...     total_claims=20,
        ...     compliant_claims=12,
        ...     average_substantiation_score=Decimal("65"),
        ...     evidence_coverage_pct=Decimal("70"),
        ...     greenwashing_risk_score=Decimal("25"),
        ...     verification_ready_pct=Decimal("50"),
        ... )
        >>> result = engine.calculate_portfolio_metrics(metrics)
        >>> assert "provenance_hash" in result
    """

    def __init__(self) -> None:
        """Initialize GreenClaimsBenchmarkEngine."""
        self.engine_id: str = _new_uuid()
        self.version: str = _MODULE_VERSION
        logger.info(
            "GreenClaimsBenchmarkEngine initialized | engine_id=%s version=%s",
            self.engine_id,
            self.version,
        )

    # ------------------------------------------------------------------
    # Public Methods
    # ------------------------------------------------------------------

    def calculate_portfolio_metrics(
        self,
        claims_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Calculate aggregate portfolio metrics from individual claim results.

        Processes a list of individual claim assessment results and
        produces summary statistics for the portfolio.

        Each entry in claims_results should contain:
            - substantiation_score (Decimal/float): 0-100 score
            - evidence_complete (bool): Whether evidence is complete
            - verification_ready (bool): Whether claim is verification-ready
            - greenwashing_risk_score (Decimal/float): 0-100 risk score
            - compliant (bool): Whether claim is compliant
            - label_compliant (bool, optional): Label compliance status

        Args:
            claims_results: List of individual claim result dicts.

        Returns:
            Dict with aggregated portfolio metrics, dimension
            scores, and provenance_hash.
        """
        logger.info(
            "Calculating portfolio metrics | claims_count=%d",
            len(claims_results),
        )
        timestamp = utcnow()
        calc_id = _new_uuid()

        total = len(claims_results)
        if total == 0:
            empty_result = {
                "calculation_id": calc_id,
                "timestamp": str(timestamp),
                "total_claims": 0,
                "compliant_claims": 0,
                "non_compliant_claims": 0,
                "portfolio_metrics": {},
                "dimension_scores": {},
                "note": "No claims to analyze",
                "engine_id": self.engine_id,
                "version": self.version,
            }
            empty_result["provenance_hash"] = _compute_hash(empty_result)
            return empty_result

        # Aggregate metrics
        compliant_count = 0
        non_compliant_count = 0
        total_substantiation = Decimal("0")
        total_risk = Decimal("0")
        evidence_complete_count = 0
        verification_ready_count = 0
        label_compliant_count = 0
        label_total_count = 0

        for entry in claims_results:
            if entry.get("compliant", False):
                compliant_count += 1
            else:
                non_compliant_count += 1

            sub_score = _decimal(entry.get("substantiation_score", 0))
            total_substantiation += sub_score

            risk_score = _decimal(entry.get("greenwashing_risk_score", 0))
            total_risk += risk_score

            if entry.get("evidence_complete", False):
                evidence_complete_count += 1

            if entry.get("verification_ready", False):
                verification_ready_count += 1

            if "label_compliant" in entry:
                label_total_count += 1
                if entry["label_compliant"]:
                    label_compliant_count += 1

        total_d = _decimal(total)

        avg_substantiation = _round_val(
            _safe_divide(total_substantiation, total_d), 2,
        )
        avg_risk = _round_val(
            _safe_divide(total_risk, total_d), 2,
        )
        evidence_pct = _round_val(
            _safe_divide(
                _decimal(evidence_complete_count) * Decimal("100"), total_d,
            ), 2,
        )
        verification_pct = _round_val(
            _safe_divide(
                _decimal(verification_ready_count) * Decimal("100"), total_d,
            ), 2,
        )
        compliance_pct = _round_val(
            _safe_divide(
                _decimal(compliant_count) * Decimal("100"), total_d,
            ), 2,
        )
        label_pct = _round_val(
            _safe_divide(
                _decimal(label_compliant_count) * Decimal("100"),
                _decimal(label_total_count) if label_total_count > 0 else Decimal("1"),
            ), 2,
        )

        # Calculate dimension scores
        dimension_scores: Dict[str, str] = {
            BenchmarkDimension.SUBSTANTIATION_QUALITY.value: str(avg_substantiation),
            BenchmarkDimension.EVIDENCE_COMPLETENESS.value: str(evidence_pct),
            BenchmarkDimension.LABEL_GOVERNANCE.value: str(label_pct),
            BenchmarkDimension.GREENWASHING_RISK.value: str(
                _round_val(Decimal("100") - avg_risk, 2)
            ),
            BenchmarkDimension.VERIFICATION_READINESS.value: str(verification_pct),
            BenchmarkDimension.REGULATORY_ALIGNMENT.value: str(compliance_pct),
        }

        portfolio_metrics = {
            "total_claims": total,
            "compliant_claims": compliant_count,
            "non_compliant_claims": non_compliant_count,
            "average_substantiation_score": str(avg_substantiation),
            "greenwashing_risk_score": str(avg_risk),
            "evidence_coverage_pct": str(evidence_pct),
            "verification_ready_pct": str(verification_pct),
            "compliance_pct": str(compliance_pct),
            "label_compliance_pct": str(label_pct),
        }

        result = {
            "calculation_id": calc_id,
            "timestamp": str(timestamp),
            "total_claims": total,
            "compliant_claims": compliant_count,
            "non_compliant_claims": non_compliant_count,
            "portfolio_metrics": portfolio_metrics,
            "dimension_scores": dimension_scores,
            "engine_id": self.engine_id,
            "version": self.version,
        }
        result["provenance_hash"] = _compute_hash(result)
        logger.info(
            "Portfolio metrics calculated | calc_id=%s total=%d compliant=%d",
            calc_id,
            total,
            compliant_count,
        )
        return result

    def determine_maturity_level(
        self, score: Decimal,
    ) -> Dict[str, Any]:
        """Determine the maturity level for a given score.

        Maps a numeric score (0-100) to one of the five maturity
        levels based on fixed threshold ranges.

        Args:
            score: Overall benchmark score (0-100).

        Returns:
            Dict with maturity level, description, characteristics,
            next level targets, and provenance_hash.
        """
        logger.info("Determining maturity level | score=%s", str(score))
        timestamp = utcnow()
        assessment_id = _new_uuid()

        score_d = _decimal(score)
        score_d = max(Decimal("0"), min(Decimal("100"), score_d))

        # Determine level
        maturity = MaturityLevel.INITIAL.value
        for level_key, spec in MATURITY_THRESHOLDS.items():
            lower = spec["lower_bound"]
            upper = spec["upper_bound"]
            if lower <= score_d < upper:
                maturity = level_key
                break
            if score_d >= Decimal("100") and level_key == MaturityLevel.OPTIMIZING.value:
                maturity = level_key

        current_spec = MATURITY_THRESHOLDS.get(
            maturity, MATURITY_THRESHOLDS[MaturityLevel.INITIAL.value],
        )

        # Determine next level
        level_order = [
            MaturityLevel.INITIAL.value,
            MaturityLevel.DEVELOPING.value,
            MaturityLevel.DEFINED.value,
            MaturityLevel.MANAGED.value,
            MaturityLevel.OPTIMIZING.value,
        ]
        current_idx = level_order.index(maturity)
        next_level = None
        next_level_spec = None
        points_to_next = Decimal("0")

        if current_idx < len(level_order) - 1:
            next_level = level_order[current_idx + 1]
            next_level_spec = MATURITY_THRESHOLDS.get(next_level)
            if next_level_spec:
                points_to_next = _round_val(
                    next_level_spec["lower_bound"] - score_d, 2,
                )
                points_to_next = max(Decimal("0"), points_to_next)

        result = {
            "assessment_id": assessment_id,
            "timestamp": str(timestamp),
            "score": str(_round_val(score_d, 2)),
            "maturity_level": maturity,
            "level_number": current_spec.get("level_number", 1),
            "description": current_spec.get("description", ""),
            "characteristics": current_spec.get("characteristics", []),
            "score_range": {
                "lower_bound": str(current_spec["lower_bound"]),
                "upper_bound": str(current_spec["upper_bound"]),
            },
            "next_level": next_level,
            "next_level_description": (
                next_level_spec.get("description", "") if next_level_spec else None
            ),
            "points_to_next_level": str(points_to_next),
            "engine_id": self.engine_id,
            "version": self.version,
        }
        result["provenance_hash"] = _compute_hash(result)
        logger.info(
            "Maturity level determined | assessment_id=%s level=%s (%d/5)",
            assessment_id,
            maturity,
            current_spec.get("level_number", 1),
        )
        return result

    def benchmark_against_peers(
        self,
        metrics: PortfolioMetrics,
        peer_data: List[Dict[str, Any]],
        *,
        entity_name: str = "Target Entity",
    ) -> Dict[str, Any]:
        """Benchmark entity metrics against peer data.

        Calculates percentile ranking and relative performance
        for each dimension against a peer cohort.

        Each entry in peer_data should contain:
            - entity_name (str): Peer name.
            - overall_score (Decimal/float): Peer overall score.
            - dimension_scores (dict, optional): Dimension scores.

        Args:
            metrics: Portfolio metrics for the entity.
            peer_data: List of peer entity data dicts.
            entity_name: Name of the entity being benchmarked.

        Returns:
            Dict with percentile, peer comparison outcome,
            dimension comparisons, and provenance_hash.
        """
        logger.info(
            "Benchmarking against peers | entity=%s peer_count=%d",
            entity_name,
            len(peer_data),
        )
        timestamp = utcnow()
        benchmark_id = _new_uuid()

        # Calculate entity overall score from metrics
        entity_dimension_scores = self._metrics_to_dimension_scores(metrics)
        overall_result = self.calculate_overall_score(entity_dimension_scores)
        entity_score = _decimal(overall_result.get("overall_score", "0"))

        # Collect peer scores
        peer_scores: List[Decimal] = []
        for peer in peer_data:
            peer_score = _decimal(peer.get("overall_score", 0))
            peer_scores.append(peer_score)

        # Calculate percentile
        if not peer_scores:
            percentile = Decimal("50")
            comparison_outcome = PeerComparisonOutcome.AVERAGE.value
        else:
            below_count = sum(1 for ps in peer_scores if ps < entity_score)
            percentile = _round_val(
                _safe_divide(
                    _decimal(below_count) * Decimal("100"),
                    _decimal(len(peer_scores)),
                ), 2,
            )
            comparison_outcome = self._determine_peer_outcome(percentile)

        # Peer statistics
        if peer_scores:
            peer_avg = _round_val(
                _safe_divide(
                    sum(peer_scores), _decimal(len(peer_scores)),
                ), 2,
            )
            peer_min = _round_val(min(peer_scores), 2)
            peer_max = _round_val(max(peer_scores), 2)
            sorted_peers = sorted(peer_scores)
            mid = len(sorted_peers) // 2
            peer_median = _round_val(
                sorted_peers[mid] if len(sorted_peers) % 2 == 1
                else (sorted_peers[mid - 1] + sorted_peers[mid]) / 2,
                2,
            )
        else:
            peer_avg = Decimal("0")
            peer_min = Decimal("0")
            peer_max = Decimal("0")
            peer_median = Decimal("0")

        # Dimension-level comparison
        dimension_comparisons: Dict[str, Dict[str, Any]] = {}
        for dim_key, dim_score in entity_dimension_scores.items():
            peer_dim_scores: List[Decimal] = []
            for peer in peer_data:
                pdim = peer.get("dimension_scores", {})
                if dim_key in pdim:
                    peer_dim_scores.append(_decimal(pdim[dim_key]))

            if peer_dim_scores:
                dim_avg = _round_val(
                    _safe_divide(
                        sum(peer_dim_scores), _decimal(len(peer_dim_scores)),
                    ), 2,
                )
                delta = _round_val(dim_score - dim_avg, 2)
            else:
                dim_avg = Decimal("0")
                delta = Decimal("0")

            dimension_comparisons[dim_key] = {
                "entity_score": str(_round_val(dim_score, 2)),
                "peer_average": str(dim_avg),
                "delta": str(delta),
                "above_average": delta > Decimal("0"),
            }

        result = {
            "benchmark_id": benchmark_id,
            "entity_name": entity_name,
            "timestamp": str(timestamp),
            "entity_overall_score": str(_round_val(entity_score, 2)),
            "percentile": str(percentile),
            "comparison_outcome": comparison_outcome,
            "peer_count": len(peer_data),
            "peer_statistics": {
                "average": str(peer_avg),
                "median": str(peer_median),
                "minimum": str(peer_min),
                "maximum": str(peer_max),
            },
            "dimension_comparisons": dimension_comparisons,
            "entity_dimension_scores": {
                k: str(_round_val(v, 2)) for k, v in entity_dimension_scores.items()
            },
            "engine_id": self.engine_id,
            "version": self.version,
        }
        result["provenance_hash"] = _compute_hash(result)
        logger.info(
            "Peer benchmark complete | benchmark_id=%s percentile=%s outcome=%s",
            benchmark_id,
            str(percentile),
            comparison_outcome,
        )
        return result

    def generate_improvement_roadmap(
        self,
        metrics: PortfolioMetrics,
        *,
        entity_name: str = "Target Entity",
        max_actions: int = 15,
    ) -> Dict[str, Any]:
        """Generate an improvement roadmap based on portfolio metrics.

        Identifies gaps, prioritizes improvement actions, and
        produces a phased roadmap with estimated effort.

        Args:
            metrics: Portfolio metrics for the entity.
            entity_name: Name of the entity.
            max_actions: Maximum number of actions to include.

        Returns:
            Dict with roadmap phases, actions, estimated effort,
            and provenance_hash.
        """
        logger.info(
            "Generating improvement roadmap | entity=%s", entity_name,
        )
        timestamp = utcnow()
        roadmap_id = _new_uuid()

        # Calculate dimension scores
        dimension_scores = self._metrics_to_dimension_scores(metrics)

        # Overall score and maturity
        overall_result = self.calculate_overall_score(dimension_scores)
        overall_score = _decimal(overall_result.get("overall_score", "0"))
        maturity_result = self.determine_maturity_level(overall_score)

        # Identify weaknesses (dimensions below 60)
        weaknesses: List[Dict[str, Any]] = []
        strengths: List[Dict[str, Any]] = []

        for dim_key, dim_score in dimension_scores.items():
            spec = DIMENSION_SPECS.get(dim_key, {})
            if dim_score < Decimal("60"):
                weaknesses.append({
                    "dimension": dim_key,
                    "name": spec.get("name", dim_key),
                    "score": str(_round_val(dim_score, 2)),
                    "gap": str(_round_val(Decimal("60") - dim_score, 2)),
                })
            elif dim_score >= Decimal("80"):
                strengths.append({
                    "dimension": dim_key,
                    "name": spec.get("name", dim_key),
                    "score": str(_round_val(dim_score, 2)),
                })

        # Sort weaknesses by gap size (largest first = highest priority)
        weaknesses.sort(key=lambda w: _decimal(w["gap"]), reverse=True)

        # Build action list from weakness dimensions
        actions: List[Dict[str, Any]] = []
        action_number = 0

        for weakness in weaknesses:
            dim_key = weakness["dimension"]
            dim_actions = IMPROVEMENT_ACTIONS.get(dim_key, [])
            for act in dim_actions:
                if action_number >= max_actions:
                    break
                action_number += 1
                actions.append({
                    "action_number": action_number,
                    "dimension": dim_key,
                    "dimension_name": weakness["name"],
                    "action": act["action"],
                    "timeframe": act["timeframe"],
                    "effort_hours": act["effort_hours"],
                    "impact": act["impact"],
                    "current_score": weakness["score"],
                    "target_score": "60.00",
                })
            if action_number >= max_actions:
                break

        # Group by timeframe
        phases: Dict[str, List[Dict[str, Any]]] = {
            ImprovementTimeframe.IMMEDIATE.value: [],
            ImprovementTimeframe.SHORT_TERM.value: [],
            ImprovementTimeframe.MEDIUM_TERM.value: [],
            ImprovementTimeframe.LONG_TERM.value: [],
        }

        for act in actions:
            tf = act.get("timeframe", ImprovementTimeframe.MEDIUM_TERM.value)
            if tf in phases:
                phases[tf].append(act)

        # Total estimated effort
        total_effort = sum(a.get("effort_hours", 0) for a in actions)

        result = {
            "roadmap_id": roadmap_id,
            "entity_name": entity_name,
            "timestamp": str(timestamp),
            "current_overall_score": str(_round_val(overall_score, 2)),
            "current_maturity_level": maturity_result.get("maturity_level", "initial"),
            "target_maturity_level": maturity_result.get("next_level", "optimizing"),
            "strengths": strengths,
            "weaknesses": weaknesses,
            "improvement_actions": actions,
            "action_count": len(actions),
            "phases": {
                k: {
                    "action_count": len(v),
                    "total_effort_hours": sum(a.get("effort_hours", 0) for a in v),
                    "actions": v,
                }
                for k, v in phases.items()
            },
            "total_estimated_effort_hours": total_effort,
            "dimension_scores": {
                k: str(_round_val(v, 2)) for k, v in dimension_scores.items()
            },
            "engine_id": self.engine_id,
            "version": self.version,
        }
        result["provenance_hash"] = _compute_hash(result)
        logger.info(
            "Roadmap generated | roadmap_id=%s actions=%d effort=%dh",
            roadmap_id,
            len(actions),
            total_effort,
        )
        return result

    def calculate_overall_score(
        self,
        dimension_scores: Dict[str, Decimal],
    ) -> Dict[str, Any]:
        """Calculate the weighted overall benchmark score.

        Applies dimension weights to produce a single overall
        readiness score from individual dimension scores.

        Args:
            dimension_scores: Dict mapping dimension keys to scores (0-100).

        Returns:
            Dict with overall score, dimension contributions,
            maturity level, and provenance_hash.
        """
        logger.info(
            "Calculating overall score | dimensions=%d",
            len(dimension_scores),
        )
        timestamp = utcnow()
        calc_id = _new_uuid()

        weighted_sum = Decimal("0")
        total_weight = Decimal("0")
        contributions: Dict[str, Dict[str, str]] = {}

        for dim_key in BenchmarkDimension:
            spec = DIMENSION_SPECS.get(dim_key.value, {})
            weight = spec.get("weight", Decimal("0.10"))
            score = _decimal(
                dimension_scores.get(dim_key.value, Decimal("0"))
            )

            # Clamp score to 0-100
            score = max(Decimal("0"), min(Decimal("100"), score))

            contribution = score * weight
            weighted_sum += contribution
            total_weight += weight

            contributions[dim_key.value] = {
                "name": spec.get("name", dim_key.value),
                "score": str(_round_val(score, 2)),
                "weight": str(weight),
                "contribution": str(_round_val(contribution, 4)),
            }

        # Normalize
        overall_score = _safe_divide(weighted_sum, total_weight)
        overall_score = _round_val(overall_score, 2)

        # Determine maturity level inline
        maturity = MaturityLevel.INITIAL.value
        for level_key, spec in MATURITY_THRESHOLDS.items():
            if spec["lower_bound"] <= overall_score < spec["upper_bound"]:
                maturity = level_key
                break
            if (
                overall_score >= Decimal("100")
                and level_key == MaturityLevel.OPTIMIZING.value
            ):
                maturity = level_key

        # Identify strengths and weaknesses
        strengths: List[str] = []
        weaknesses: List[str] = []
        improvement_priorities: List[str] = []

        for dim_key, contrib in contributions.items():
            dim_score = _decimal(contrib["score"])
            dim_name = contrib["name"]
            if dim_score >= Decimal("80"):
                strengths.append(f"{dim_name} ({contrib['score']})")
            elif dim_score < Decimal("50"):
                weaknesses.append(f"{dim_name} ({contrib['score']})")
                improvement_priorities.append(dim_name)

        result = {
            "calculation_id": calc_id,
            "timestamp": str(timestamp),
            "overall_score": str(overall_score),
            "maturity_level": maturity,
            "dimension_contributions": contributions,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "improvement_priorities": improvement_priorities,
            "total_weight": str(_round_val(total_weight, 2)),
            "engine_id": self.engine_id,
            "version": self.version,
        }
        result["provenance_hash"] = _compute_hash(result)
        logger.info(
            "Overall score calculated | calc_id=%s score=%s maturity=%s",
            calc_id,
            str(overall_score),
            maturity,
        )
        return result

    # ------------------------------------------------------------------
    # Private Methods
    # ------------------------------------------------------------------

    def _metrics_to_dimension_scores(
        self, metrics: PortfolioMetrics,
    ) -> Dict[str, Decimal]:
        """Convert portfolio metrics to dimension scores.

        Maps portfolio metric fields to the six benchmark dimensions.

        Args:
            metrics: Portfolio metrics.

        Returns:
            Dict mapping dimension keys to scores (0-100).
        """
        # Greenwashing risk is inverted (low risk = high score)
        greenwashing_score = Decimal("100") - _decimal(
            metrics.greenwashing_risk_score
        )
        greenwashing_score = max(Decimal("0"), greenwashing_score)

        return {
            BenchmarkDimension.SUBSTANTIATION_QUALITY.value: _decimal(
                metrics.average_substantiation_score
            ),
            BenchmarkDimension.EVIDENCE_COMPLETENESS.value: _decimal(
                metrics.evidence_coverage_pct
            ),
            BenchmarkDimension.LABEL_GOVERNANCE.value: _decimal(
                metrics.label_compliance_pct
            ),
            BenchmarkDimension.GREENWASHING_RISK.value: greenwashing_score,
            BenchmarkDimension.VERIFICATION_READINESS.value: _decimal(
                metrics.verification_ready_pct
            ),
            BenchmarkDimension.REGULATORY_ALIGNMENT.value: _decimal(
                metrics.regulatory_alignment_pct
            ),
        }

    def _determine_peer_outcome(
        self, percentile: Decimal,
    ) -> str:
        """Determine peer comparison outcome from percentile.

        Args:
            percentile: Percentile ranking (0-100).

        Returns:
            Peer comparison outcome string.
        """
        if percentile >= Decimal("80"):
            return PeerComparisonOutcome.LEADING.value
        elif percentile >= Decimal("60"):
            return PeerComparisonOutcome.ABOVE_AVERAGE.value
        elif percentile >= Decimal("40"):
            return PeerComparisonOutcome.AVERAGE.value
        elif percentile >= Decimal("20"):
            return PeerComparisonOutcome.BELOW_AVERAGE.value
        else:
            return PeerComparisonOutcome.LAGGING.value
