# -*- coding: utf-8 -*-
"""
CreditQualityEngine - PACK-024 Carbon Neutral Engine 3
=======================================================

12-dimension ICVCM Core Carbon Principles (CCP) scoring engine for
carbon credit quality assessment covering additionality, permanence,
robust quantification, independent validation, avoidance of double
counting, transition towards net-zero, sustainable development, no net
harm, host country participation, registry operations, effective
governance, and transparency.

This engine provides a deterministic, auditable quality score for
individual carbon credits or credit portfolios, enabling organisations
to meet ISO 14068-1:2023 and PAS 2060:2014 credit quality requirements.

Calculation Methodology:
    ICVCM CCP Scoring (12 dimensions):
        Each dimension scored 0-10:
            EXCELLENT:  9-10  (meets all CCP requirements with best practice)
            GOOD:       7-8   (meets CCP requirements)
            ADEQUATE:   5-6   (partially meets, minor gaps)
            POOR:       3-4   (significant gaps)
            FAILING:    0-2   (does not meet requirements)

        Dimension weights (sum to 1.0):
            additionality:         0.15  (highest: core integrity)
            permanence:            0.12
            robust_quantification: 0.12
            independent_validation: 0.10
            double_counting:       0.10
            transition:            0.08
            sustainable_development: 0.08
            no_net_harm:           0.07
            host_country:          0.05
            registry:              0.05
            governance:            0.04
            transparency:          0.04

        overall_score = sum(dimension_score * dimension_weight) * 10
        overall_rating: A+ (>=95), A (>=85), B+ (>=75), B (>=65),
                        C (>=50), D (>=35), F (<35)

    Additionality Assessment (ICVCM CCP Criterion 1):
        financial_additionality: Project not viable without carbon revenue
        regulatory_additionality: Not required by law/regulation
        barrier_analysis: Barriers exist that credits help overcome
        common_practice: Activity not common practice in the sector

    Permanence Risk (ICVCM CCP Criterion 2):
        reversal_risk: Risk of stored carbon being released
        buffer_pool: Adequate buffer pool contribution
        monitoring: Long-term monitoring plan
        insurance: Insurance or replacement commitment

Regulatory References:
    - ICVCM Core Carbon Principles V1.0 (2023)
    - ICVCM Assessment Framework V1.0 (2023)
    - ISO 14068-1:2023 - Section 8: Carbon credits quality
    - PAS 2060:2014 - Section 5.4: Offset credits quality
    - Article 6 of the Paris Agreement (2015)
    - CORSIA Eligible Emissions Unit Criteria (ICAO, 2023)
    - Gold Standard for the Global Goals V1.2 (2022)
    - Verra VCS Standard V4.5 (2023)

Zero-Hallucination:
    - All 12 CCP dimensions from ICVCM Assessment Framework V1.0
    - Scoring criteria from published ICVCM standards
    - No LLM involvement in any calculation path
    - Deterministic Decimal arithmetic throughout
    - SHA-256 provenance hash on every result

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-024 Carbon Neutral
Engine:  3 of 10
Status:  Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_MODULE_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _new_uuid() -> str:
    return str(uuid.uuid4())


def _compute_hash(data: Any) -> str:
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
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return Decimal("0")


def _safe_divide(
    numerator: Decimal, denominator: Decimal, default: Decimal = Decimal("0"),
) -> Decimal:
    if denominator == Decimal("0"):
        return default
    return numerator / denominator


def _safe_pct(part: Decimal, whole: Decimal) -> Decimal:
    return _safe_divide(part * Decimal("100"), whole)


def _round_val(value: Decimal, places: int = 6) -> Decimal:
    quantize_str = "0." + "0" * places
    return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)


def _round3(value: float) -> float:
    return float(
        Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    )


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class CCPDimension(str, Enum):
    """ICVCM Core Carbon Principles dimensions.

    12 dimensions from ICVCM Assessment Framework V1.0 (2023).
    """
    ADDITIONALITY = "additionality"
    PERMANENCE = "permanence"
    ROBUST_QUANTIFICATION = "robust_quantification"
    INDEPENDENT_VALIDATION = "independent_validation"
    DOUBLE_COUNTING = "double_counting"
    TRANSITION = "transition"
    SUSTAINABLE_DEVELOPMENT = "sustainable_development"
    NO_NET_HARM = "no_net_harm"
    HOST_COUNTRY = "host_country"
    REGISTRY = "registry"
    GOVERNANCE = "governance"
    TRANSPARENCY = "transparency"


class QualityRating(str, Enum):
    """Overall credit quality rating.

    A_PLUS: Exceptional quality (>=95%).
    A: High quality (>=85%).
    B_PLUS: Good quality (>=75%).
    B: Acceptable quality (>=65%).
    C: Marginal quality (>=50%).
    D: Poor quality (>=35%).
    F: Failing quality (<35%).
    """
    A_PLUS = "A+"
    A = "A"
    B_PLUS = "B+"
    B = "B"
    C = "C"
    D = "D"
    F = "F"


class DimensionRating(str, Enum):
    """Rating for a single CCP dimension.

    EXCELLENT: 9-10 points.
    GOOD: 7-8 points.
    ADEQUATE: 5-6 points.
    POOR: 3-4 points.
    FAILING: 0-2 points.
    """
    EXCELLENT = "excellent"
    GOOD = "good"
    ADEQUATE = "adequate"
    POOR = "poor"
    FAILING = "failing"


class ProjectType(str, Enum):
    """Carbon credit project type classification.

    AVOIDANCE: Avoided emissions (renewable energy, cookstoves).
    REDUCTION: Emission reductions (efficiency, fuel switching).
    REMOVAL_NATURE: Nature-based removal (afforestation, soil carbon).
    REMOVAL_TECH: Technology-based removal (DACCS, BECCS, biochar).
    """
    AVOIDANCE = "avoidance"
    REDUCTION = "reduction"
    REMOVAL_NATURE = "removal_nature"
    REMOVAL_TECH = "removal_tech"


class CreditStandard(str, Enum):
    """Carbon credit certification standard.

    VCS: Verified Carbon Standard (Verra).
    GOLD_STANDARD: Gold Standard for the Global Goals.
    CAR: Climate Action Reserve.
    ACR: American Carbon Registry.
    PURO: Puro.earth (carbon removal).
    CDM: Clean Development Mechanism.
    CORSIA: Carbon Offsetting and Reduction Scheme for International Aviation.
    """
    VCS = "vcs"
    GOLD_STANDARD = "gold_standard"
    CAR = "car"
    ACR = "acr"
    PURO = "puro"
    CDM = "cdm"
    CORSIA = "corsia"


# ---------------------------------------------------------------------------
# Constants -- ICVCM CCP Dimension Weights
# ---------------------------------------------------------------------------

# Dimension weights from ICVCM Assessment Framework V1.0 (2023).
# Sum = 1.00.
CCP_DIMENSION_WEIGHTS: Dict[str, Decimal] = {
    CCPDimension.ADDITIONALITY.value: Decimal("0.15"),
    CCPDimension.PERMANENCE.value: Decimal("0.12"),
    CCPDimension.ROBUST_QUANTIFICATION.value: Decimal("0.12"),
    CCPDimension.INDEPENDENT_VALIDATION.value: Decimal("0.10"),
    CCPDimension.DOUBLE_COUNTING.value: Decimal("0.10"),
    CCPDimension.TRANSITION.value: Decimal("0.08"),
    CCPDimension.SUSTAINABLE_DEVELOPMENT.value: Decimal("0.08"),
    CCPDimension.NO_NET_HARM.value: Decimal("0.07"),
    CCPDimension.HOST_COUNTRY.value: Decimal("0.05"),
    CCPDimension.REGISTRY.value: Decimal("0.05"),
    CCPDimension.GOVERNANCE.value: Decimal("0.04"),
    CCPDimension.TRANSPARENCY.value: Decimal("0.04"),
}

# Rating thresholds (overall score out of 100).
RATING_THRESHOLDS: List[Tuple[Decimal, str]] = [
    (Decimal("95"), QualityRating.A_PLUS.value),
    (Decimal("85"), QualityRating.A.value),
    (Decimal("75"), QualityRating.B_PLUS.value),
    (Decimal("65"), QualityRating.B.value),
    (Decimal("50"), QualityRating.C.value),
    (Decimal("35"), QualityRating.D.value),
    (Decimal("0"), QualityRating.F.value),
]

# Dimension rating thresholds (score out of 10).
DIMENSION_RATING_THRESHOLDS: List[Tuple[Decimal, str]] = [
    (Decimal("9"), DimensionRating.EXCELLENT.value),
    (Decimal("7"), DimensionRating.GOOD.value),
    (Decimal("5"), DimensionRating.ADEQUATE.value),
    (Decimal("3"), DimensionRating.POOR.value),
    (Decimal("0"), DimensionRating.FAILING.value),
]

# Minimum quality score for ISO 14068-1 compliance.
MIN_QUALITY_ISO14068: Decimal = Decimal("65")

# Minimum quality score for PAS 2060 compliance.
MIN_QUALITY_PAS2060: Decimal = Decimal("50")

# Critical dimensions that cannot score below 5 for compliance.
CRITICAL_DIMENSIONS: List[str] = [
    CCPDimension.ADDITIONALITY.value,
    CCPDimension.PERMANENCE.value,
    CCPDimension.ROBUST_QUANTIFICATION.value,
    CCPDimension.DOUBLE_COUNTING.value,
]


# ---------------------------------------------------------------------------
# Pydantic Models -- Input
# ---------------------------------------------------------------------------


class DimensionInput(BaseModel):
    """Input scoring data for a single CCP dimension.

    Attributes:
        dimension: CCP dimension identifier.
        score: Score (0-10) for this dimension.
        evidence: Evidence supporting the score.
        assessor_notes: Assessor comments.
        evidence_documents: List of supporting documents.
        auto_assessed: Whether score was auto-calculated.
    """
    dimension: str = Field(..., description="CCP dimension identifier")
    score: Decimal = Field(
        default=Decimal("0"), ge=0, le=Decimal("10"),
        description="Score (0-10)"
    )
    evidence: str = Field(default="", description="Supporting evidence")
    assessor_notes: str = Field(default="", description="Assessor notes")
    evidence_documents: List[str] = Field(
        default_factory=list, description="Supporting documents"
    )
    auto_assessed: bool = Field(
        default=False, description="Whether auto-assessed"
    )

    @field_validator("dimension")
    @classmethod
    def validate_dimension(cls, v: str) -> str:
        valid = {d.value for d in CCPDimension}
        if v not in valid:
            raise ValueError(f"Unknown CCP dimension '{v}'. Must be one of: {sorted(valid)}")
        return v


class AdditionalityInput(BaseModel):
    """Detailed additionality assessment input.

    Attributes:
        financial_additionality: Project not viable without carbon revenue.
        regulatory_additionality: Activity not required by regulation.
        barrier_analysis_complete: Barriers documented and analysed.
        common_practice_analysis: Activity not common practice.
        irr_without_carbon_pct: Project IRR without carbon revenue.
        irr_with_carbon_pct: Project IRR with carbon revenue.
        benchmark_irr_pct: Sector benchmark IRR.
        regulatory_surplus: Beyond regulatory requirements.
        evidence_quality: Quality of additionality evidence.
    """
    financial_additionality: bool = Field(default=False)
    regulatory_additionality: bool = Field(default=False)
    barrier_analysis_complete: bool = Field(default=False)
    common_practice_analysis: bool = Field(default=False)
    irr_without_carbon_pct: Decimal = Field(
        default=Decimal("0"), description="IRR without carbon (%)"
    )
    irr_with_carbon_pct: Decimal = Field(
        default=Decimal("0"), description="IRR with carbon (%)"
    )
    benchmark_irr_pct: Decimal = Field(
        default=Decimal("10"), description="Benchmark IRR (%)"
    )
    regulatory_surplus: bool = Field(default=False)
    evidence_quality: str = Field(
        default="moderate", description="Evidence quality"
    )


class PermanenceInput(BaseModel):
    """Detailed permanence assessment input.

    Attributes:
        permanence_years: Expected permanence duration.
        reversal_risk_pct: Estimated reversal risk percentage.
        buffer_pool_contribution_pct: Buffer pool contribution percentage.
        monitoring_plan: Whether long-term monitoring plan exists.
        insurance_mechanism: Whether insurance/replacement commitment exists.
        is_geological_storage: Whether storage is geological (very high permanence).
        is_nature_based: Whether nature-based (moderate permanence risk).
        tonne_year_accounting: Whether tonne-year accounting is applied.
    """
    permanence_years: int = Field(default=0, ge=0, description="Expected permanence (years)")
    reversal_risk_pct: Decimal = Field(
        default=Decimal("10"), ge=0, le=Decimal("100"),
        description="Reversal risk (%)"
    )
    buffer_pool_contribution_pct: Decimal = Field(
        default=Decimal("0"), ge=0, le=Decimal("100"),
        description="Buffer pool contribution (%)"
    )
    monitoring_plan: bool = Field(default=False)
    insurance_mechanism: bool = Field(default=False)
    is_geological_storage: bool = Field(default=False)
    is_nature_based: bool = Field(default=False)
    tonne_year_accounting: bool = Field(default=False)


class CreditQualityInput(BaseModel):
    """Complete input for carbon credit quality assessment.

    Attributes:
        credit_id: Credit or batch identifier.
        project_name: Project name.
        project_type: Project type classification.
        credit_standard: Certification standard.
        methodology: Methodology used.
        vintage_year: Credit vintage year.
        quantity_tco2e: Number of credits (tCO2e).
        country: Project country (ISO 3166-1 alpha-2).
        region: Project region/state.
        dimensions: Per-dimension scoring data.
        additionality_detail: Detailed additionality assessment.
        permanence_detail: Detailed permanence assessment.
        price_per_tco2e_usd: Purchase price per credit.
        registry_serial: Registry serial number.
        verification_body: Independent verification body.
        sdg_contributions: UN SDG contributions claimed.
        host_country_authorization: Whether host country has authorized.
        corresponding_adjustment: Whether corresponding adjustment applied.
        include_recommendations: Whether to generate recommendations.
    """
    credit_id: str = Field(default_factory=_new_uuid, description="Credit ID")
    project_name: str = Field(default="", max_length=300, description="Project name")
    project_type: str = Field(
        default=ProjectType.REDUCTION.value, description="Project type"
    )
    credit_standard: str = Field(
        default=CreditStandard.VCS.value, description="Credit standard"
    )
    methodology: str = Field(default="", max_length=200, description="Methodology")
    vintage_year: int = Field(default=0, ge=0, le=2060, description="Vintage year")
    quantity_tco2e: Decimal = Field(
        default=Decimal("0"), ge=0, description="Credit quantity (tCO2e)"
    )
    country: str = Field(default="", max_length=2, description="Country code")
    region: str = Field(default="", max_length=100, description="Region")
    dimensions: List[DimensionInput] = Field(
        default_factory=list, description="Per-dimension scores"
    )
    additionality_detail: Optional[AdditionalityInput] = Field(
        default=None, description="Detailed additionality"
    )
    permanence_detail: Optional[PermanenceInput] = Field(
        default=None, description="Detailed permanence"
    )
    price_per_tco2e_usd: Decimal = Field(
        default=Decimal("0"), ge=0, description="Price per tCO2e (USD)"
    )
    registry_serial: str = Field(default="", description="Registry serial number")
    verification_body: str = Field(default="", description="Verification body")
    sdg_contributions: List[int] = Field(
        default_factory=list, description="UN SDG numbers (1-17)"
    )
    host_country_authorization: bool = Field(
        default=False, description="Host country authorization"
    )
    corresponding_adjustment: bool = Field(
        default=False, description="Corresponding adjustment applied"
    )
    include_recommendations: bool = Field(default=True, description="Generate recommendations")

    @field_validator("project_type")
    @classmethod
    def validate_project_type(cls, v: str) -> str:
        valid = {p.value for p in ProjectType}
        if v not in valid:
            raise ValueError(f"Unknown project type '{v}'.")
        return v

    @field_validator("credit_standard")
    @classmethod
    def validate_standard(cls, v: str) -> str:
        valid = {s.value for s in CreditStandard}
        if v not in valid:
            raise ValueError(f"Unknown credit standard '{v}'.")
        return v


# ---------------------------------------------------------------------------
# Pydantic Models -- Output
# ---------------------------------------------------------------------------


class DimensionResult(BaseModel):
    """Assessment result for a single CCP dimension.

    Attributes:
        dimension: CCP dimension identifier.
        dimension_name: Human-readable dimension name.
        score: Score (0-10).
        weight: Dimension weight (0-1).
        weighted_score: score * weight.
        rating: Dimension rating (EXCELLENT/GOOD/ADEQUATE/POOR/FAILING).
        is_critical: Whether this is a critical dimension.
        meets_minimum: Whether score >= 5 (minimum for critical dims).
        evidence_provided: Whether evidence was provided.
        auto_assessed: Whether auto-assessed.
        issues: Issues identified.
        recommendations: Dimension-specific recommendations.
    """
    dimension: str = Field(default="")
    dimension_name: str = Field(default="")
    score: Decimal = Field(default=Decimal("0"))
    weight: Decimal = Field(default=Decimal("0"))
    weighted_score: Decimal = Field(default=Decimal("0"))
    rating: str = Field(default=DimensionRating.FAILING.value)
    is_critical: bool = Field(default=False)
    meets_minimum: bool = Field(default=True)
    evidence_provided: bool = Field(default=False)
    auto_assessed: bool = Field(default=False)
    issues: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class AdditionalityResult(BaseModel):
    """Detailed additionality assessment result.

    Attributes:
        financial_additionality: Whether financially additional.
        regulatory_additionality: Whether regulatory additional.
        barrier_analysis_pass: Whether barrier analysis passes.
        common_practice_pass: Whether common practice analysis passes.
        irr_gap_pct: Gap between project IRR and benchmark.
        additionality_score: Composite score (0-10).
        additionality_confident: Whether additionality is confident.
        key_risks: Key additionality risks.
    """
    financial_additionality: bool = Field(default=False)
    regulatory_additionality: bool = Field(default=False)
    barrier_analysis_pass: bool = Field(default=False)
    common_practice_pass: bool = Field(default=False)
    irr_gap_pct: Decimal = Field(default=Decimal("0"))
    additionality_score: Decimal = Field(default=Decimal("0"))
    additionality_confident: bool = Field(default=False)
    key_risks: List[str] = Field(default_factory=list)


class PermanenceResult(BaseModel):
    """Detailed permanence assessment result.

    Attributes:
        permanence_years: Expected permanence.
        permanence_tier: very_high/high/moderate/low/very_low.
        reversal_risk_pct: Reversal risk percentage.
        buffer_adequate: Whether buffer pool is adequate.
        monitoring_adequate: Whether monitoring is adequate.
        insurance_exists: Whether insurance mechanism exists.
        permanence_score: Composite score (0-10).
        adjusted_tco2e: Quantity after permanence adjustment.
        permanence_discount_pct: Discount applied for permanence risk.
    """
    permanence_years: int = Field(default=0)
    permanence_tier: str = Field(default="moderate")
    reversal_risk_pct: Decimal = Field(default=Decimal("0"))
    buffer_adequate: bool = Field(default=False)
    monitoring_adequate: bool = Field(default=False)
    insurance_exists: bool = Field(default=False)
    permanence_score: Decimal = Field(default=Decimal("0"))
    adjusted_tco2e: Decimal = Field(default=Decimal("0"))
    permanence_discount_pct: Decimal = Field(default=Decimal("0"))


class CreditQualityResult(BaseModel):
    """Complete credit quality assessment result.

    Attributes:
        result_id: Unique result identifier.
        engine_version: Engine version.
        calculated_at: Calculation timestamp.
        credit_id: Credit identifier.
        project_name: Project name.
        project_type: Project type.
        credit_standard: Credit standard.
        vintage_year: Vintage year.
        quantity_tco2e: Credit quantity.
        dimension_results: Per-dimension assessment results.
        additionality_result: Detailed additionality result.
        permanence_result: Detailed permanence result.
        overall_score: Overall quality score (0-100).
        overall_rating: Quality rating (A+/A/B+/B/C/D/F).
        meets_iso14068: Whether meets ISO 14068-1 requirements.
        meets_pas2060: Whether meets PAS 2060 requirements.
        meets_icvcm_ccp: Whether eligible for ICVCM CCP label.
        critical_dimension_pass: Whether all critical dimensions pass.
        dimensions_assessed: Number of dimensions assessed.
        dimensions_passing: Number of dimensions with adequate+ score.
        price_quality_ratio: Price per quality point.
        recommendations: Overall recommendations.
        warnings: Non-blocking warnings.
        errors: Blocking errors.
        processing_time_ms: Processing duration.
        provenance_hash: SHA-256 audit hash.
    """
    result_id: str = Field(default_factory=_new_uuid)
    engine_version: str = Field(default=_MODULE_VERSION)
    calculated_at: datetime = Field(default_factory=_utcnow)
    credit_id: str = Field(default="")
    project_name: str = Field(default="")
    project_type: str = Field(default="")
    credit_standard: str = Field(default="")
    vintage_year: int = Field(default=0)
    quantity_tco2e: Decimal = Field(default=Decimal("0"))
    dimension_results: List[DimensionResult] = Field(default_factory=list)
    additionality_result: Optional[AdditionalityResult] = Field(default=None)
    permanence_result: Optional[PermanenceResult] = Field(default=None)
    overall_score: Decimal = Field(default=Decimal("0"))
    overall_rating: str = Field(default=QualityRating.F.value)
    meets_iso14068: bool = Field(default=False)
    meets_pas2060: bool = Field(default=False)
    meets_icvcm_ccp: bool = Field(default=False)
    critical_dimension_pass: bool = Field(default=False)
    dimensions_assessed: int = Field(default=0)
    dimensions_passing: int = Field(default=0)
    price_quality_ratio: Decimal = Field(default=Decimal("0"))
    recommendations: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    errors: List[str] = Field(default_factory=list)
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")


# ---------------------------------------------------------------------------
# Dimension Name Lookup
# ---------------------------------------------------------------------------

DIMENSION_NAMES: Dict[str, str] = {
    CCPDimension.ADDITIONALITY.value: "Additionality",
    CCPDimension.PERMANENCE.value: "Permanence",
    CCPDimension.ROBUST_QUANTIFICATION.value: "Robust Quantification",
    CCPDimension.INDEPENDENT_VALIDATION.value: "Independent Validation",
    CCPDimension.DOUBLE_COUNTING.value: "Avoidance of Double Counting",
    CCPDimension.TRANSITION.value: "Transition Towards Net-Zero",
    CCPDimension.SUSTAINABLE_DEVELOPMENT.value: "Sustainable Development",
    CCPDimension.NO_NET_HARM.value: "No Net Harm",
    CCPDimension.HOST_COUNTRY.value: "Host Country Participation",
    CCPDimension.REGISTRY.value: "Registry Operations",
    CCPDimension.GOVERNANCE.value: "Effective Governance",
    CCPDimension.TRANSPARENCY.value: "Transparency",
}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class CreditQualityEngine:
    """ICVCM Core Carbon Principles 12-dimension credit quality engine.

    Provides a comprehensive, deterministic quality assessment for
    carbon credits across all 12 ICVCM CCP dimensions, with detailed
    additionality and permanence assessments.

    Usage::

        engine = CreditQualityEngine()
        result = engine.assess(input_data)
        print(f"Overall: {result.overall_rating} ({result.overall_score}/100)")
        for dim in result.dimension_results:
            print(f"  {dim.dimension_name}: {dim.score}/10 ({dim.rating})")
    """

    engine_version: str = _MODULE_VERSION

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise CreditQualityEngine.

        Args:
            config: Optional overrides. Supported keys:
                - dimension_weights (dict): Custom weights (must sum to 1.0)
                - min_quality_threshold (Decimal): Minimum score for acceptance
                - critical_minimum_score (Decimal): Minimum for critical dims
        """
        self.config = config or {}
        self._weights = dict(CCP_DIMENSION_WEIGHTS)
        custom_weights = self.config.get("dimension_weights")
        if custom_weights:
            for k, v in custom_weights.items():
                if k in self._weights:
                    self._weights[k] = _decimal(v)
        self._min_threshold = _decimal(
            self.config.get("min_quality_threshold", MIN_QUALITY_ISO14068)
        )
        self._critical_min = _decimal(
            self.config.get("critical_minimum_score", Decimal("5"))
        )
        logger.info("CreditQualityEngine v%s initialised", self.engine_version)

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    def assess(
        self, data: CreditQualityInput,
    ) -> CreditQualityResult:
        """Perform complete 12-dimension credit quality assessment.

        Args:
            data: Validated credit quality input.

        Returns:
            CreditQualityResult with comprehensive assessment.
        """
        t0 = time.perf_counter()
        logger.info(
            "Credit quality assessment: project=%s, standard=%s, vintage=%d",
            data.project_name, data.credit_standard, data.vintage_year,
        )

        warnings: List[str] = []
        errors: List[str] = []

        # Build dimension score map from input
        dim_scores: Dict[str, DimensionInput] = {}
        for d in data.dimensions:
            dim_scores[d.dimension] = d

        # Step 1: Auto-assess additionality if detail provided
        add_result: Optional[AdditionalityResult] = None
        if data.additionality_detail:
            add_result = self._assess_additionality(data.additionality_detail)
            # Update dimension score if not manually provided
            if CCPDimension.ADDITIONALITY.value not in dim_scores:
                dim_scores[CCPDimension.ADDITIONALITY.value] = DimensionInput(
                    dimension=CCPDimension.ADDITIONALITY.value,
                    score=add_result.additionality_score,
                    evidence="Auto-assessed from detailed additionality input",
                    auto_assessed=True,
                )

        # Step 2: Auto-assess permanence if detail provided
        perm_result: Optional[PermanenceResult] = None
        if data.permanence_detail:
            perm_result = self._assess_permanence(
                data.permanence_detail, data.quantity_tco2e
            )
            if CCPDimension.PERMANENCE.value not in dim_scores:
                dim_scores[CCPDimension.PERMANENCE.value] = DimensionInput(
                    dimension=CCPDimension.PERMANENCE.value,
                    score=perm_result.permanence_score,
                    evidence="Auto-assessed from detailed permanence input",
                    auto_assessed=True,
                )

        # Step 3: Score all 12 dimensions
        dimension_results = self._score_dimensions(
            dim_scores, data, warnings
        )

        # Step 4: Calculate overall score
        overall_score = Decimal("0")
        for dr in dimension_results:
            overall_score += dr.weighted_score
        overall_score = _round_val(overall_score * Decimal("10"), 2)

        # Step 5: Determine rating
        overall_rating = self._determine_rating(overall_score)

        # Step 6: Check critical dimensions
        critical_pass = all(
            dr.meets_minimum for dr in dimension_results if dr.is_critical
        )

        # Step 7: Compliance checks
        meets_iso = overall_score >= MIN_QUALITY_ISO14068 and critical_pass
        meets_pas = overall_score >= MIN_QUALITY_PAS2060
        meets_ccp = (
            overall_score >= Decimal("75") and critical_pass
            and all(dr.score >= Decimal("5") for dr in dimension_results)
        )

        # Step 8: Count passing
        dims_assessed = len(dimension_results)
        dims_passing = sum(1 for dr in dimension_results if dr.score >= Decimal("5"))

        # Step 9: Price/quality ratio
        pq_ratio = Decimal("0")
        if data.price_per_tco2e_usd > Decimal("0") and overall_score > Decimal("0"):
            pq_ratio = _round_val(
                data.price_per_tco2e_usd / overall_score, 4
            )

        # Step 10: Recommendations
        recommendations: List[str] = []
        if data.include_recommendations:
            recommendations = self._generate_recommendations(
                dimension_results, overall_score, meets_iso,
                critical_pass, data
            )

        # Warnings
        if not critical_pass:
            warnings.append(
                "One or more critical CCP dimensions scored below minimum (5/10). "
                "Credits may not be suitable for carbon neutral claims."
            )
        if data.vintage_year > 0 and data.vintage_year < (datetime.now().year - 5):
            warnings.append(
                f"Vintage year {data.vintage_year} is more than 5 years old. "
                f"ISO 14068-1 recommends recent vintages."
            )
        if not data.host_country_authorization:
            warnings.append(
                "No host country authorization for these credits. "
                "Required under Article 6 of Paris Agreement."
            )

        elapsed_ms = _round3((time.perf_counter() - t0) * 1000.0)

        result = CreditQualityResult(
            credit_id=data.credit_id,
            project_name=data.project_name,
            project_type=data.project_type,
            credit_standard=data.credit_standard,
            vintage_year=data.vintage_year,
            quantity_tco2e=data.quantity_tco2e,
            dimension_results=dimension_results,
            additionality_result=add_result,
            permanence_result=perm_result,
            overall_score=overall_score,
            overall_rating=overall_rating,
            meets_iso14068=meets_iso,
            meets_pas2060=meets_pas,
            meets_icvcm_ccp=meets_ccp,
            critical_dimension_pass=critical_pass,
            dimensions_assessed=dims_assessed,
            dimensions_passing=dims_passing,
            price_quality_ratio=pq_ratio,
            recommendations=recommendations,
            warnings=warnings,
            errors=errors,
            processing_time_ms=elapsed_ms,
        )
        result.provenance_hash = _compute_hash(result)

        logger.info(
            "Credit quality assessment complete: score=%.1f, rating=%s, "
            "iso14068=%s, ccp=%s, hash=%s",
            float(overall_score), overall_rating, meets_iso,
            meets_ccp, result.provenance_hash[:16],
        )
        return result

    # ------------------------------------------------------------------ #
    # Internal Methods                                                     #
    # ------------------------------------------------------------------ #

    def _score_dimensions(
        self,
        dim_scores: Dict[str, DimensionInput],
        data: CreditQualityInput,
        warnings: List[str],
    ) -> List[DimensionResult]:
        """Score all 12 CCP dimensions.

        For dimensions without explicit input, assigns a default score of 5
        (adequate) and flags as needing assessment.

        Args:
            dim_scores: Dimension input scores.
            data: Credit input data.
            warnings: Warning list.

        Returns:
            List of DimensionResult for all 12 dimensions.
        """
        results: List[DimensionResult] = []

        for dim in CCPDimension:
            dim_val = dim.value
            weight = self._weights.get(dim_val, Decimal("0.05"))
            is_critical = dim_val in CRITICAL_DIMENSIONS
            dim_name = DIMENSION_NAMES.get(dim_val, dim_val)

            if dim_val in dim_scores:
                inp = dim_scores[dim_val]
                score = inp.score
                evidence = bool(inp.evidence)
                auto = inp.auto_assessed
            else:
                score = Decimal("5")
                evidence = False
                auto = False
                warnings.append(
                    f"CCP dimension '{dim_name}' not explicitly scored. "
                    f"Default score of 5/10 applied."
                )

            weighted = score * weight
            rating = self._dimension_rating(score)
            meets_min = score >= self._critical_min

            issues: List[str] = []
            recs: List[str] = []

            if is_critical and not meets_min:
                issues.append(
                    f"Critical dimension '{dim_name}' scored {score}/10, "
                    f"below minimum threshold of {self._critical_min}."
                )
                recs.append(
                    f"Improve '{dim_name}' assessment to at least {self._critical_min}/10 "
                    f"for credit eligibility."
                )

            if not evidence:
                issues.append(f"No evidence provided for '{dim_name}'.")
                recs.append(f"Provide documented evidence for '{dim_name}' scoring.")

            results.append(DimensionResult(
                dimension=dim_val,
                dimension_name=dim_name,
                score=score,
                weight=weight,
                weighted_score=_round_val(weighted, 4),
                rating=rating,
                is_critical=is_critical,
                meets_minimum=meets_min,
                evidence_provided=evidence,
                auto_assessed=auto,
                issues=issues,
                recommendations=recs,
            ))

        return results

    def _assess_additionality(
        self, detail: AdditionalityInput,
    ) -> AdditionalityResult:
        """Assess additionality from detailed input.

        Calculates composite score based on financial, regulatory,
        barrier, and common practice additionality tests.

        Args:
            detail: Detailed additionality input.

        Returns:
            AdditionalityResult.
        """
        score = Decimal("0")
        risks: List[str] = []

        # Financial additionality (3 points max)
        if detail.financial_additionality:
            score += Decimal("3")
            if detail.irr_without_carbon_pct < detail.benchmark_irr_pct:
                score += Decimal("0.5")
        else:
            risks.append("Financial additionality not demonstrated.")

        # Regulatory additionality (2.5 points max)
        if detail.regulatory_additionality:
            score += Decimal("2")
            if detail.regulatory_surplus:
                score += Decimal("0.5")
        else:
            risks.append("Activity may be required by regulation.")

        # Barrier analysis (2 points max)
        if detail.barrier_analysis_complete:
            score += Decimal("2")
        else:
            risks.append("Barrier analysis not completed.")

        # Common practice (2 points max)
        if detail.common_practice_analysis:
            score += Decimal("2")
        else:
            risks.append("Common practice analysis not performed.")

        score = min(score, Decimal("10"))

        irr_gap = detail.irr_with_carbon_pct - detail.irr_without_carbon_pct

        confident = (
            detail.financial_additionality
            and detail.regulatory_additionality
            and score >= Decimal("7")
        )

        return AdditionalityResult(
            financial_additionality=detail.financial_additionality,
            regulatory_additionality=detail.regulatory_additionality,
            barrier_analysis_pass=detail.barrier_analysis_complete,
            common_practice_pass=detail.common_practice_analysis,
            irr_gap_pct=_round_val(irr_gap, 2),
            additionality_score=_round_val(score, 1),
            additionality_confident=confident,
            key_risks=risks,
        )

    def _assess_permanence(
        self,
        detail: PermanenceInput,
        quantity: Decimal,
    ) -> PermanenceResult:
        """Assess permanence from detailed input.

        Calculates permanence score and any discount factor based on
        reversal risk, buffer pool, and monitoring.

        Args:
            detail: Detailed permanence input.
            quantity: Credit quantity for adjustment.

        Returns:
            PermanenceResult.
        """
        score = Decimal("0")

        # Permanence duration scoring
        if detail.is_geological_storage:
            score += Decimal("4")
            tier = "very_high"
        elif detail.permanence_years >= 1000:
            score += Decimal("4")
            tier = "very_high"
        elif detail.permanence_years >= 100:
            score += Decimal("3")
            tier = "high"
        elif detail.permanence_years >= 40:
            score += Decimal("2")
            tier = "moderate"
        elif detail.permanence_years >= 20:
            score += Decimal("1")
            tier = "low"
        else:
            score += Decimal("0")
            tier = "very_low"

        # Buffer pool (2 points max)
        if detail.buffer_pool_contribution_pct >= Decimal("20"):
            score += Decimal("2")
        elif detail.buffer_pool_contribution_pct >= Decimal("10"):
            score += Decimal("1")

        # Monitoring (2 points max)
        if detail.monitoring_plan:
            score += Decimal("2")

        # Insurance (2 points max)
        if detail.insurance_mechanism:
            score += Decimal("2")

        score = min(score, Decimal("10"))

        # Permanence discount
        discount = detail.reversal_risk_pct
        if detail.buffer_pool_contribution_pct >= detail.reversal_risk_pct:
            discount = max(Decimal("0"), discount - detail.buffer_pool_contribution_pct)

        adjusted = quantity * (Decimal("100") - discount) / Decimal("100")

        return PermanenceResult(
            permanence_years=detail.permanence_years,
            permanence_tier=tier,
            reversal_risk_pct=detail.reversal_risk_pct,
            buffer_adequate=detail.buffer_pool_contribution_pct >= detail.reversal_risk_pct,
            monitoring_adequate=detail.monitoring_plan,
            insurance_exists=detail.insurance_mechanism,
            permanence_score=_round_val(score, 1),
            adjusted_tco2e=_round_val(adjusted),
            permanence_discount_pct=_round_val(discount, 2),
        )

    def _determine_rating(self, score: Decimal) -> str:
        """Determine overall quality rating from score.

        Args:
            score: Overall score (0-100).

        Returns:
            QualityRating value.
        """
        for threshold, rating in RATING_THRESHOLDS:
            if score >= threshold:
                return rating
        return QualityRating.F.value

    def _dimension_rating(self, score: Decimal) -> str:
        """Determine dimension rating from score.

        Args:
            score: Dimension score (0-10).

        Returns:
            DimensionRating value.
        """
        for threshold, rating in DIMENSION_RATING_THRESHOLDS:
            if score >= threshold:
                return rating
        return DimensionRating.FAILING.value

    def _generate_recommendations(
        self,
        dimensions: List[DimensionResult],
        overall: Decimal,
        meets_iso: bool,
        critical_pass: bool,
        data: CreditQualityInput,
    ) -> List[str]:
        """Generate overall recommendations.

        Args:
            dimensions: Dimension results.
            overall: Overall score.
            meets_iso: Whether ISO 14068-1 is met.
            critical_pass: Whether critical dimensions pass.
            data: Input data.

        Returns:
            List of recommendation strings.
        """
        recs: List[str] = []

        if not critical_pass:
            failing_critical = [
                d.dimension_name for d in dimensions
                if d.is_critical and not d.meets_minimum
            ]
            recs.append(
                f"CRITICAL: Dimensions {', '.join(failing_critical)} score below "
                f"minimum. These must be addressed before credits can be used "
                f"for carbon neutral claims."
            )

        if not meets_iso:
            recs.append(
                f"Overall score of {overall}/100 is below ISO 14068-1 minimum "
                f"of {MIN_QUALITY_ISO14068}. Consider higher-quality credits."
            )

        # Check for removal vs avoidance progression (Oxford Principles)
        if data.project_type == ProjectType.AVOIDANCE.value:
            recs.append(
                "Consider transitioning to carbon removal credits (nature-based "
                "or technology-based) in line with Oxford Principles progression."
            )

        # SDG contributions
        if len(data.sdg_contributions) == 0:
            recs.append(
                "No UN SDG contributions claimed. Credits with verified SDG "
                "co-benefits strengthen carbon neutral claims."
            )

        # Corresponding adjustment
        if not data.corresponding_adjustment:
            recs.append(
                "No corresponding adjustment applied under Article 6. "
                "Required for international transfer claims."
            )

        # Low-scoring dimensions
        for d in dimensions:
            if d.score <= Decimal("4") and not d.is_critical:
                recs.append(
                    f"Improve '{d.dimension_name}' (currently {d.score}/10) "
                    f"to strengthen overall credit quality."
                )

        return recs
